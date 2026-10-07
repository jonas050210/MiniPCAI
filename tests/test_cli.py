"""Tests for the command line interface."""

from __future__ import annotations

import json

import pytest

from minipcai import __version__
from minipcai.cli import main


def _common(registry_path, model_path, audit_path):
    return [
        "--registry", str(registry_path),
        "--model", str(model_path),
        "--audit", str(audit_path),
    ]


class TestAsk:
    def test_ok_request(self, capsys, trained_model, registry_factory, tmp_path):
        code = main([
            "ask", "öffne notepad",
            *_common(registry_factory(), trained_model, tmp_path / "audit.jsonl"),
        ])
        assert code == 0
        out = capsys.readouterr().out
        assert "[ok]" in out
        assert "notepad" in out

    def test_rejected_request(self, capsys, trained_model, registry_factory, tmp_path):
        code = main([
            "ask", "wie wird das wetter morgen",
            *_common(registry_factory(), trained_model, tmp_path / "audit.jsonl"),
        ])
        assert code == 1
        assert "[rejected]" in capsys.readouterr().out

    def test_json_output(self, capsys, trained_model, registry_factory, tmp_path):
        code = main([
            "ask", "was ist 12*4", "--json",
            *_common(registry_factory(), trained_model, tmp_path / "audit.jsonl"),
        ])
        assert code == 0
        data = json.loads(capsys.readouterr().out)
        assert data["status"] == "ok"
        assert data["intent"] == "calc"

    def test_missing_model_fails_cleanly(self, capsys, registry_factory, tmp_path):
        code = main([
            "ask", "öffne notepad",
            *_common(registry_factory(), tmp_path / "missing.joblib",
                     tmp_path / "audit.jsonl"),
        ])
        assert code == 2
        assert "minipcai-train" in capsys.readouterr().err


class TestRegistryCommand:
    def test_lists_entries(self, capsys, registry_factory):
        code = main(["registry", "--registry", str(registry_factory())])
        assert code == 0
        out = capsys.readouterr().out
        assert "OK: registry v1" in out
        assert "notepad" in out
        assert "wikipedia" in out
        assert "[searchable]" in out

    def test_invalid_registry_fails(self, capsys, tmp_path):
        bad = tmp_path / "registry.json"
        bad.write_text("{}", encoding="utf-8")
        code = main(["registry", "--registry", str(bad)])
        assert code == 2
        assert "Invalid registry" in capsys.readouterr().err


class TestAuditCommand:
    def _run_ask(self, registry_path, model_path, audit_path, text):
        return main(["ask", text, *_common(registry_path, model_path, audit_path)])

    def test_empty_log(self, capsys, tmp_path):
        audit_path = tmp_path / "missing.jsonl"
        code = main(["audit", "--audit", str(audit_path)])
        assert code == 0
        assert "No audit log yet" in capsys.readouterr().out

    def test_shows_recent_records(self, capsys, trained_model, registry_factory,
                                  tmp_path):
        audit_path = tmp_path / "audit.jsonl"
        registry_path = registry_factory()
        self._run_ask(registry_path, trained_model, audit_path, "öffne notepad")
        self._run_ask(registry_path, trained_model, audit_path,
                      "wie wird das wetter morgen")
        capsys.readouterr()  # drop the ask output
        code = main(["audit", "--audit", str(audit_path)])
        assert code == 0
        out = capsys.readouterr().out
        assert "accepted" in out and "open_app" in out
        assert "rejected" in out and "unknown_request" in out
        assert "action ok" in out

    def test_limit_and_json(self, capsys, trained_model, registry_factory, tmp_path):
        audit_path = tmp_path / "audit.jsonl"
        registry_path = registry_factory()
        self._run_ask(registry_path, trained_model, audit_path, "öffne notepad")
        capsys.readouterr()
        code = main(["audit", "--audit", str(audit_path), "--limit", "1", "--json"])
        assert code == 0
        lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["event"] == "action_result"

    def test_corrupt_lines_are_skipped(self, capsys, tmp_path):
        audit_path = tmp_path / "audit.jsonl"
        audit_path.write_text("not json\n", encoding="utf-8")
        code = main(["audit", "--audit", str(audit_path)])
        assert code == 0
        captured = capsys.readouterr()
        assert "no readable records" in captured.out
        assert "unreadable" in captured.err

    def test_audit_command_does_not_modify_the_log(self, trained_model,
                                                   registry_factory, tmp_path):
        audit_path = tmp_path / "audit.jsonl"
        self._run_ask(registry_factory(), trained_model, audit_path, "öffne notepad")
        before = audit_path.read_bytes()
        main(["audit", "--audit", str(audit_path)])
        assert audit_path.read_bytes() == before


class TestVersion:
    def test_version_flag(self, capsys):
        with pytest.raises(SystemExit) as excinfo:
            main(["--version"])
        assert excinfo.value.code == 0
        assert __version__ in capsys.readouterr().out


class TestTrainCommand:
    def test_train_writes_artifacts(self, capsys, trained_model, tmp_path):
        # The session-scoped trained_model fixture already exercised train();
        # here we only verify the CLI wiring with a tiny fresh run.
        from minipcai.config import DEFAULT_DATASET_PATH

        code = main([
            "train", "--dataset", str(DEFAULT_DATASET_PATH),
            "--models-dir", str(tmp_path / "cli_models"),
        ])
        assert code == 0
        assert (tmp_path / "cli_models" / "model.joblib").is_file()
        assert (tmp_path / "cli_models" / "metadata.json").is_file()
        assert (tmp_path / "cli_models" / "metrics.json").is_file()
        assert "training report" in capsys.readouterr().out


class TestConfirmationCommand:
    def test_non_interactive_confirmation_requires_yes(
        self, capsys, trained_model, registry_factory, tmp_path
    ):
        code = main([
            "ask", "schließe notepad",
            *_common(registry_factory(), trained_model, tmp_path / "audit.jsonl"),
        ])
        assert code == 1
        captured = capsys.readouterr()
        assert "[confirmation]" in captured.out
        assert "--yes" in captured.err
        # nothing was executed
        assert not (tmp_path / "audit.jsonl").read_text(encoding="utf-8").count(
            '"event": "action_result"'
        )

    def test_yes_confirms_automatically(self, capsys, trained_model, registry_factory, tmp_path):
        audit_path = tmp_path / "audit.jsonl"
        code = main([
            "ask", "schließe notepad", "--yes",
            *_common(registry_factory(), trained_model, audit_path),
        ])
        assert code == 0
        assert "[ok]" in capsys.readouterr().out
        assert '"event": "action_result"' in audit_path.read_text(encoding="utf-8")

    def test_language_flag_switches_the_reply(
        self, capsys, trained_model, registry_factory, tmp_path
    ):
        code = main([
            "ask", "öffne notepad", "--language", "de",
            *_common(registry_factory(), trained_model, tmp_path / "audit.jsonl"),
        ])
        assert code == 0
        assert "öffnen" in capsys.readouterr().out.lower()

    def test_environment_no_confirm_escape_hatch(
        self, capsys, monkeypatch, trained_model, registry_factory, tmp_path
    ):
        monkeypatch.setenv("MINIPCAI_NO_CONFIRM", "1")
        code = main([
            "ask", "schließe notepad",
            *_common(registry_factory(), trained_model, tmp_path / "audit.jsonl"),
        ])
        assert code == 0

    def test_audit_survives_confirmation_decline(
        self, capsys, trained_model, registry_factory, tmp_path
    ):
        from minipcai.audit import verify_chain

        audit_path = tmp_path / "audit.jsonl"
        main(["ask", "schließe notepad",
              *_common(registry_factory(), trained_model, audit_path)])
        main(["ask", "öffne notepad",
              *_common(registry_factory(), trained_model, audit_path)])
        ok, checked, bad_line = verify_chain(audit_path)
        assert ok is True, bad_line
        # confirmation ("required"), accepted + action result for the ok request
        assert checked >= 3


class TestSetupCommand:
    def test_setup_creates_user_files(self, capsys, monkeypatch, tmp_path):
        from minipcai import paths

        monkeypatch.setattr(paths, "state_dir", lambda environ=None: tmp_path / "state")
        code = main(["setup", "--no-file-check"])
        # the packaged registry contains Windows paths, so the report warns,
        # but nothing fails structurally
        assert code in (0, 1)
        out = capsys.readouterr().out
        assert "Configuration:" in out
        assert (tmp_path / "state" / "registry.json").is_file()
        config = (tmp_path / "state" / "config.toml").read_text(encoding="utf-8")
        assert "[minipcai]" in config

    def test_setup_language_and_privacy_flags(self, capsys, monkeypatch, tmp_path):
        from minipcai import paths

        monkeypatch.setattr(paths, "state_dir", lambda environ=None: tmp_path / "state")
        main(["setup", "--no-file-check", "--language", "de", "--private-audit"])
        config = (tmp_path / "state" / "config.toml").read_text(encoding="utf-8")
        assert 'language = "de"' in config
        assert "store_text_in_audit = false" in config

    def test_setup_keeps_an_existing_registry(self, capsys, monkeypatch, tmp_path):
        from minipcai import paths

        monkeypatch.setattr(paths, "state_dir", lambda environ=None: tmp_path / "state")
        state = tmp_path / "state"
        state.mkdir(parents=True)
        (state / "registry.json").write_text('{"version": 1, "apps": []}', encoding="utf-8")
        main(["setup", "--no-file-check"])
        assert "Keeping the existing registry" in capsys.readouterr().out
        assert json.loads((state / "registry.json").read_text(encoding="utf-8"))["apps"] == []


class TestDoctorCommand:
    def test_healthy_installation(self, capsys, trained_model, registry_factory, tmp_path):
        code = main([
            "doctor", "--no-gui-check",
            "--model", str(trained_model),
            "--registry", str(registry_factory()),
            "--audit", str(tmp_path / "audit.jsonl"),
        ])
        assert code == 0
        out = capsys.readouterr().out
        assert "[OK  ] model" in out
        assert "Everything essential works" in out

    def test_json_output_and_failure_code(self, capsys, tmp_path):
        code = main([
            "doctor", "--no-gui-check", "--json",
            "--model", str(tmp_path / "missing.joblib"),
            "--registry", str(tmp_path / "missing.json"),
            "--audit", str(tmp_path / "audit.jsonl"),
        ])
        assert code == 1
        data = json.loads(capsys.readouterr().out)
        statuses = {entry["name"]: entry["status"] for entry in data}
        assert statuses["model"] == "fail"
        assert statuses["registry"] == "fail"


class TestAuditVerifyCommand:
    def test_verify_detects_tampering(self, capsys, tmp_path):
        from minipcai.audit import AuditLogger

        path = tmp_path / "audit.jsonl"
        audit = AuditLogger(path)
        audit.log_event("request_result", index=0)
        audit.log_event("request_result", index=1)
        lines = path.read_text(encoding="utf-8").splitlines()
        first = json.loads(lines[0])
        first["index"] = 42
        path.write_text(lines[0].replace('"index": 0', '"index": 42') + "\n" + lines[1] + "\n",
                        encoding="utf-8")
        code = main(["audit", "--audit", str(path), "--verify"])
        assert code == 1
        assert "hash chain broken" in capsys.readouterr().err

    def test_verify_ok(self, capsys, tmp_path):
        from minipcai.audit import AuditLogger

        path = tmp_path / "audit.jsonl"
        audit = AuditLogger(path)
        audit.log_event("request_result", index=0)
        code = main(["audit", "--audit", str(path), "--verify"])
        assert code == 0
        assert "hash chain intact" in capsys.readouterr().out

class TestUiCommand:
    def test_ui_and_gui_are_registered(self):
        from minipcai.cli import _cmd_ui, build_parser

        parser = build_parser()
        # both names must resolve to the same handler
        assert parser.parse_args(["ui"]).func is parser.parse_args(["gui"]).func is _cmd_ui

    def test_missing_pyside6_is_reported_friendly(self, capsys, monkeypatch):
        from minipcai import ui as ui_package
        from minipcai.cli import main

        def boom(*args, **kwargs):
            raise ui_package.GuiUnavailable(ui_package.GUI_MISSING_HINT)

        monkeypatch.setattr(ui_package, "run_ui", boom)
        code = main(["ui"])
        assert code == 2
        err = capsys.readouterr().err
        assert "PySide6" in err
        assert "minipcai[gui]" in err
        assert "Traceback" not in err

    def test_gui_available_matches_the_environment(self):
        from minipcai.ui import gui_available

        assert isinstance(gui_available(), bool)
