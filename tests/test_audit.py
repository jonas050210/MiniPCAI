"""Tests for the audit logger."""

from __future__ import annotations

import json

import pytest

from minipcai.audit import AuditError, AuditLogger, read_audit_events


class TestAuditLogger:
    def test_creates_parent_directories(self, tmp_path):
        audit = AuditLogger(tmp_path / "deep" / "nested" / "audit.jsonl")
        audit.log_event("request_result", decision="rejected")
        assert (tmp_path / "deep" / "nested" / "audit.jsonl").is_file()

    def test_events_are_valid_json_lines(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        audit = AuditLogger(path)
        audit.log_event("request_result", text="öffne notepad", decision="rejected",
                        reason="unknown_request")
        audit.log_event("request_accepted", text="was ist 12*4", intent="calc")
        events = read_audit_events(path)
        assert len(events) == 2
        assert events[0]["event"] == "request_result"
        assert events[0]["decision"] == "rejected"
        assert "ts" in events[0] and "T" in events[0]["ts"]
        assert events[0]["schema_version"] == 1
        assert events[1]["intent"] == "calc"

    def test_german_text_written_as_utf8(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        AuditLogger(path).log_event("request_result", text="schließe die tür")
        raw = path.read_bytes().decode("utf-8")
        assert "schließe die tür" in raw

    def test_unwritable_path_raises(self, tmp_path):
        blocked = tmp_path / "as_directory"
        blocked.mkdir()
        audit = AuditLogger(blocked)  # a directory cannot be appended to
        with pytest.raises(AuditError):
            audit.log_event("request_result")

    def test_check_writable_raises_for_directory(self, tmp_path):
        audit = AuditLogger(tmp_path / "as_directory")
        (tmp_path / "as_directory").mkdir()
        with pytest.raises(AuditError):
            audit.check_writable()

    def test_check_writable_ok(self, tmp_path):
        AuditLogger(tmp_path / "ok.jsonl").check_writable()

    def test_wrapper_methods(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        audit = AuditLogger(path)
        audit.log_rejected("r1", "öffne das", None, None, "ambiguous_intent", "msg")
        audit.log_accepted("r2", "öffne notepad", "open_app", 0.9, "apps", "notepad",
                           "dry-run")
        audit.log_action_result("r2", "open_app", True, "opened", "dry-run", 12)
        audit.log_timer_elapsed("r3", 300)
        events = read_audit_events(path)
        assert [event["event"] for event in events] == [
            "request_result", "request_accepted", "action_result", "timer_elapsed",
        ]
        assert events[1]["target_id"] == "notepad"
        assert events[3]["duration_seconds"] == 300

    def test_thread_safety(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        audit = AuditLogger(path)

        def write_many():
            for index in range(25):
                audit.log_event("spam", index=index)

        threads = [__import__("threading").Thread(target=write_many) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 100
        for line in lines:
            json.loads(line)  # every line must be valid JSON
