"""Tests for ``minipcai doctor`` (installation self-check)."""

from __future__ import annotations

import json

import pytest

from minipcai.doctor import (
    FAIL,
    OK,
    WARN,
    format_report,
    has_failures,
    run,
    run_checks,
    to_dict,
)
from minipcai.settings import Settings


@pytest.fixture()
def healthy_settings(trained_model, registry_factory, tmp_path):
    registry = registry_factory()
    return Settings(
        model_path=str(trained_model),
        registry_path=str(registry),
        audit_path=str(tmp_path / "audit.jsonl"),
    )


def _by_name(checks):
    return {check.name: check for check in checks}


class TestHealthyInstallation:
    def test_no_failures(self, healthy_settings):
        checks = run_checks(healthy_settings, check_files=True, check_gui=False)
        assert not has_failures(checks)
        assert all(check.status in {OK, WARN} for check in checks)
        assert "environment" in _by_name(checks)
        assert "registry" in _by_name(checks)
        assert "model" in _by_name(checks)
        assert "audit log" in _by_name(checks)
        assert "security policy" in _by_name(checks)

    def test_exit_code_and_report(self, healthy_settings):
        code, report = run(healthy_settings, check_files=True, check_gui=False)
        assert code == 0
        assert "Everything essential works" in report

    def test_json_report_is_valid(self, healthy_settings):
        code, report = run(
            healthy_settings, check_files=True, check_gui=False, json_output=True
        )
        assert code == 0
        data = json.loads(report)
        assert {entry["name"] for entry in data} >= {"environment", "registry", "model"}

    def test_check_objects_serialize(self, healthy_settings):
        payload = to_dict(run_checks(healthy_settings, check_files=False, check_gui=False))
        assert all({"name", "status", "detail"} <= set(entry) for entry in payload)

    def test_format_report_forwards_warnings(self, healthy_settings):
        checks = run_checks(healthy_settings, check_files=False, check_gui=False)
        text = format_report(checks)
        assert "registry" in text

    def test_gui_check_is_included_on_request(self, healthy_settings):
        checks = run_checks(healthy_settings, check_files=False, check_gui=True)
        assert "gui" in _by_name(checks)


class TestFailureModes:
    def test_missing_model_fails(self, healthy_settings):
        healthy_settings.model_path = "/nonexistent/model.joblib"
        checks = _by_name(run_checks(healthy_settings, check_gui=False))
        assert checks["model"].status == FAIL
        assert has_failures(list(checks.values())) is True
        code, report = run(healthy_settings, check_gui=False)
        assert code == 1 and "model" in report

    def test_missing_registry_fails(self, healthy_settings):
        healthy_settings.registry_path = "/nonexistent/registry.json"
        checks = _by_name(run_checks(healthy_settings, check_files=False, check_gui=False))
        assert checks["registry"].status == FAIL
        assert "minipcai setup" in checks["registry"].detail

    def test_unusable_targets_warn(self, healthy_settings, tmp_path):
        registry = tmp_path / "registry.json"
        registry.write_text(
            json.dumps({"version": 1, "files": [
                {"id": "ghost", "aliases": ["ghost"],
                 "path": str(tmp_path / "missing.txt")},
            ]}),
            encoding="utf-8",
        )
        healthy_settings.registry_path = str(registry)
        checks = _by_name(run_checks(healthy_settings, check_files=True, check_gui=False))
        assert checks["registry"].status == WARN
        assert "ghost" in checks["registry"].detail
        # ... and the exit code stays 0: usability problems are warnings
        assert run(healthy_settings, check_files=True, check_gui=False)[0] == 0

    def test_unwritable_audit_fails(self, healthy_settings, tmp_path):
        blocked = tmp_path / "audit_dir"
        blocked.mkdir()
        healthy_settings.audit_path = str(blocked)
        checks = _by_name(run_checks(healthy_settings, check_files=False, check_gui=False))
        assert checks["audit log"].status == FAIL

    def test_broken_chain_fails(self, healthy_settings, tmp_path):
        from minipcai.audit import AuditLogger

        audit = AuditLogger(tmp_path / "audit.jsonl")
        audit.log_event("request_result", index=0)
        audit.log_event("request_result", index=1)
        lines = (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()
        tampered = json.loads(lines[0])
        tampered["index"] = 99
        (tmp_path / "audit.jsonl").write_text(
            json.dumps(tampered) + "\n" + lines[1] + "\n", encoding="utf-8"
        )
        checks = _by_name(run_checks(healthy_settings, check_files=False, check_gui=False))
        assert checks["audit log"].status == FAIL
        assert "hash chain broken" in checks["audit log"].detail

    def test_rotated_files_are_reported(self, healthy_settings, tmp_path):
        from minipcai.audit import AuditLogger

        path = tmp_path / "audit.jsonl"
        audit = AuditLogger(path, max_bytes=300, max_files=3)
        for index in range(30):
            audit.log_event("request_result", index=index, padding="x" * 50)
        checks = _by_name(run_checks(healthy_settings, check_files=False, check_gui=False))
        assert checks["audit log"].status == OK
        assert "rotated file(s)" in checks["audit log"].detail


class TestPolicyCheck:
    def test_default_policy_refuses_untrusted_paths(self, healthy_settings):
        check = _by_name(run_checks(healthy_settings, check_files=False, check_gui=False))[
            "security policy"
        ]
        assert check.status == OK
        assert check.data["enforce_trusted_app_roots"] is True
        assert "close_app" in check.detail and "web_search" in check.detail

    def test_escape_hatch_is_visible(self, healthy_settings):
        healthy_settings.allow_untrusted_apps = True
        healthy_settings.confirm_intents = ()
        check = _by_name(run_checks(healthy_settings, check_files=False, check_gui=False))[
            "security policy"
        ]
        assert check.data["enforce_trusted_app_roots"] is False
        assert "untrusted app paths: allowed" in check.detail
        assert "confirmations: off" in check.detail
