"""Tests for the persistent settings (TOML, environment, precedence)."""

from __future__ import annotations

from pathlib import Path

import pytest

from minipcai.settings import (
    Settings,
    load_settings,
    save_settings,
    settings_summary,
    thresholds_from_metadata,
)


class TestDefaults:
    def test_safe_defaults(self):
        settings = Settings()
        assert settings.executor_mode == "dry-run"
        assert settings.language == "de"
        assert settings.auto_confirm is False
        assert settings.store_text_in_audit is True
        assert settings.allow_untrusted_apps is False
        assert set(settings.confirm_intents) == {"close_app", "web_search"}

    def test_validate_accepts_defaults(self):
        assert Settings().validate() == []

    @pytest.mark.parametrize(
        ("field", "value", "fragment"),
        [
            ("language", "fr", "language must be one of"),
            ("executor_mode", "turbo", "unknown executor mode"),
            ("audit_max_files", 0, "audit_max_files"),
            ("audit_max_bytes", -1, "audit_max_bytes"),
            ("max_active_timers", 0, "max_active_timers"),
            ("find_file_time_budget", -0.5, "find_file_time_budget"),
        ],
    )
    def test_validate_reports_problems(self, field, value, fragment):
        settings = Settings(**{field: value})
        problems = settings.validate()
        assert any(fragment in problem for problem in problems)


class TestPrecedence:
    def test_toml_values_are_loaded(self, tmp_path: Path):
        config = tmp_path / "config.toml"
        config.write_text(
            "[minipcai]\n"
            'language = "de"\n'
            'executor_mode = "windows"\n'
            "auto_confirm = true\n"
            "audit_max_files = 2\n"
            'confirm_intents = ["close_app"]\n'
            'theme = "dark"\n',
            encoding="utf-8",
        )
        settings = load_settings(config, use_environment=False)
        assert settings.language == "de"
        assert settings.executor_mode == "windows"
        assert settings.auto_confirm is True
        assert settings.audit_max_files == 2
        assert settings.confirm_intents == ("close_app",)
        assert settings.theme == "dark"

    def test_unknown_keys_are_ignored(self, tmp_path: Path):
        config = tmp_path / "config.toml"
        config.write_text('[minipcai]\nlanguage = "de"\nfuture_option = 7\n', encoding="utf-8")
        assert load_settings(config, use_environment=False).language == "de"

    def test_wrong_type_is_reported(self, tmp_path: Path):
        config = tmp_path / "config.toml"
        config.write_text('[minipcai]\nlanguage = 3\n', encoding="utf-8")
        with pytest.raises(ValueError, match="language must be a string"):
            load_settings(config, use_environment=False)

    def test_invalid_toml_is_reported(self, tmp_path: Path):
        config = tmp_path / "config.toml"
        config.write_text("[minipcai\n", encoding="utf-8")
        with pytest.raises(ValueError, match="cannot read"):
            load_settings(config, use_environment=False)

    def test_missing_file_returns_defaults(self, tmp_path: Path):
        assert load_settings(tmp_path / "nope.toml", use_environment=False) == Settings()

    def test_environment_overrides_the_file(self, tmp_path: Path):
        config = tmp_path / "config.toml"
        config.write_text('[minipcai]\nlanguage = "en"\n', encoding="utf-8")
        settings = load_settings(
            config,
            use_environment=False,
        ).apply_environment(
            {
                "MINIPCAI_LANGUAGE": "de",
                "MINIPCAI_EXECUTOR": "windows",
                "MINIPCAI_MODEL": "/tmp/model.joblib",
                "MINIPCAI_REGISTRY": "/tmp/registry.json",
                "MINIPCAI_AUDIT": "/tmp/audit.jsonl",
                "MINIPCAI_NO_CONFIRM": "1",
                "MINIPCAI_PRIVACY_AUDIT": "yes",
                "MINIPCAI_ALLOW_UNTRUSTED_APPS": "true",
                "MINIPCAI_EXTRA_TRUSTED_ROOTS": r"C:\tools;C:\opt",
            }
        )
        assert settings.language == "de"
        assert settings.executor_mode == "windows"
        assert settings.model_path == "/tmp/model.joblib"
        assert settings.registry_path == "/tmp/registry.json"
        assert settings.audit_path == "/tmp/audit.jsonl"
        assert settings.auto_confirm is True
        assert settings.store_text_in_audit is False
        assert settings.allow_untrusted_apps is True
        assert settings.extra_trusted_roots == (r"C:\tools", r"C:\opt")


class TestRoundTrip:
    def test_save_and_load_round_trip(self, tmp_path: Path):
        settings = Settings(
            language="en",
            executor_mode="windows",
            auto_confirm=True,
            confirm_intents=("close_app", "web_search"),
            extra_trusted_roots=(r"C:\tools",),
            theme="dark",
            hotkey="Ctrl+Shift+M",
        )
        path = save_settings(settings, tmp_path / "config.toml")
        assert path.is_file()
        loaded = load_settings(path, use_environment=False)
        assert loaded == settings

    def test_summary_contains_resolved_paths(self, tmp_path: Path):
        summary = settings_summary(
            Settings(model_path=str(tmp_path / "m.joblib"), registry_path=str(tmp_path / "r.json"))
        )
        assert summary["model_path"] == str(tmp_path / "m.joblib")
        assert summary["registry_path"] == str(tmp_path / "r.json")
        assert summary["confirm_intents"] == ["close_app", "web_search"]


class TestPolicyBridge:
    def test_to_policy_defaults_to_confirmations(self):
        policy = Settings().to_policy()
        assert policy.confirm_intents == frozenset({"close_app", "web_search"})
        assert policy.enforce_trusted_app_roots is True

    def test_to_policy_honours_escape_hatches(self):
        policy = Settings(allow_untrusted_apps=True, confirm_intents=()).to_policy()
        assert policy.enforce_trusted_app_roots is False
        assert policy.confirm_intents == frozenset()

    def test_to_policy_adds_extra_roots(self):
        policy = Settings(extra_trusted_roots=(r"C:\tools",)).to_policy()
        assert policy.is_trusted_app_path(r"C:\tools\x.exe")

    def test_to_policy_respects_the_timer_cap(self):
        assert Settings(max_active_timers=3).to_policy().max_active_timers == 3


class TestThresholdsFromMetadata:
    def test_reads_calibrated_values(self):
        thresholds = thresholds_from_metadata(
            {"thresholds": {"min_confidence": 0.7, "min_margin": 0.25}}
        )
        assert thresholds.min_confidence == pytest.approx(0.7)
        assert thresholds.min_margin == pytest.approx(0.25)

    @pytest.mark.parametrize(
        "metadata",
        [{}, None, {"thresholds": None}, {"thresholds": {"min_confidence": "x"}}],
    )
    def test_falls_back_to_defaults(self, metadata):
        from minipcai.config import Thresholds

        assert thresholds_from_metadata(metadata) == Thresholds()
