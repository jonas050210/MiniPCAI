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
