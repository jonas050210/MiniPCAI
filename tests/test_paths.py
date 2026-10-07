"""Tests for the file-system layout helpers (packaged data, user state dirs)."""

from __future__ import annotations

import json
from pathlib import Path

from minipcai import paths


class TestPackagedData:
    def test_packaged_data_dir_ships_the_registry_and_dataset(self):
        assert paths.PACKAGED_DATA_DIR.is_dir()
        assert paths.packaged_registry_path().is_file()
        assert paths.default_dataset_path().is_file()

    def test_discover_dataset_prefers_the_highest_version(self, tmp_path):
        for version in (1, 2, 10, 9):
            (tmp_path / f"intent_dataset.v{version}.jsonl").write_text("{}", encoding="utf-8")
        (tmp_path / "intent_dataset.jsonl").write_text("{}", encoding="utf-8")
        assert paths.discover_dataset_path(tmp_path).name == "intent_dataset.v10.jsonl"

    def test_discover_dataset_without_files_falls_back_to_v1(self, tmp_path):
        assert paths.discover_dataset_path(tmp_path) == tmp_path / "intent_dataset.v1.jsonl"

    def test_discover_dataset_on_unreadable_dir_does_not_raise(self, tmp_path):
        missing = tmp_path / "nope"
        assert paths.discover_dataset_path(missing).name == "intent_dataset.v1.jsonl"


class TestStateDir:
    def test_windows_uses_localappdata(self, monkeypatch):
        monkeypatch.setattr(paths.sys, "platform", "win32")
        environ = {"LOCALAPPDATA": r"C:\Users\me\AppData\Local"}
        assert paths.state_dir(environ) == Path(r"C:\Users\me\AppData\Local") / "MiniPCAI"
        # Windows without LOCALAPPDATA still resolves somewhere writable
        assert paths.state_dir({}).name == ".minipcai"

    def test_posix_uses_home(self):
        assert paths.state_dir({}) == Path.home() / ".minipcai"

    def test_derived_paths(self):
        assert paths.user_config_path({}) == paths.state_dir({}) / "config.toml"
        assert paths.user_registry_path({}) == paths.state_dir({}) / "registry.json"
        assert paths.user_models_dir({}) == paths.state_dir({}) / "models"
        assert paths.default_audit_path() == paths.state_dir() / "audit.jsonl"


class TestRegistryResolution:
    def test_explicit_path_wins(self, tmp_path, monkeypatch):
        target = tmp_path / "custom.json"
        monkeypatch.setenv("MINIPCAI_REGISTRY", str(tmp_path / "env.json"))
        assert paths.resolve_registry_path(target) == target

    def test_environment_path_second(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MINIPCAI_REGISTRY", str(tmp_path / "env.json"))
        assert paths.resolve_registry_path() == tmp_path / "env.json"

    def test_source_checkout_falls_back_to_packaged_registry(self, monkeypatch):
        monkeypatch.delenv("MINIPCAI_REGISTRY", raising=False)
        monkeypatch.setattr(paths, "is_source_checkout", lambda: True)
        monkeypatch.setattr(
            paths, "user_registry_path", lambda environ=None: Path("/nonexistent/x")
        )
        assert paths.resolve_registry_path() == paths.packaged_registry_path()

    def test_seed_user_registry_copies_and_never_overwrites(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            paths, "user_registry_path", lambda environ=None: tmp_path / "registry.json"
        )
        seeded = paths.seed_user_registry()
        assert seeded.is_file()
        # a hand-edited user registry survives a re-seed
        seeded.write_text('{"version": 1, "apps": []}', encoding="utf-8")
        paths.seed_user_registry()
        assert json.loads(seeded.read_text(encoding="utf-8"))["apps"] == []
        # ... unless the caller asks for a fresh copy
        paths.seed_user_registry(overwrite=True)
        assert paths.registry_is_user_copy(seeded) is True

    def test_describe_layout_reports_expected_keys(self):
        layout = paths.describe_layout()
        assert set(layout) >= {"package", "packaged data", "state", "models", "checkout"}
        assert layout["checkout"] in {"yes", "no"}


class TestModelsDir:
    def test_default_models_dir_is_writable(self):
        directory = paths.default_models_dir()
        assert directory.name in {"models", "MiniPCAI"}
        assert str(directory)

    def test_installed_layout_prefers_user_dir(self, monkeypatch):
        monkeypatch.setattr(paths, "is_source_checkout", lambda: False)
        assert paths.default_models_dir() == paths.user_models_dir()
