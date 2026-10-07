"""Tests for the audit logger."""

from __future__ import annotations

import json

import pytest

from minipcai.audit import SCHEMA_VERSION, AuditError, AuditLogger, read_audit_events


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
        assert events[0]["schema_version"] == SCHEMA_VERSION
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


class TestHashChain:
    def test_chain_is_built_and_verifies(self, tmp_path):
        from minipcai.audit import verify_chain

        path = tmp_path / "audit.jsonl"
        audit = AuditLogger(path)
        for index in range(5):
            audit.log_event("request_result", index=index)
        # consecutive records written by the same logger must link correctly
        ok, checked, bad_line = verify_chain(path)
        assert (ok, checked, bad_line) == (True, 5, 0)
        events = read_audit_events(path)
        assert events[0]["prev"] == ""
        assert events[1]["prev"] == events[0]["hash"][:12]

    def test_missing_hash_is_ignored_not_failing(self, tmp_path):
        from minipcai.audit import verify_chain

        path = tmp_path / "audit.jsonl"
        audit = AuditLogger(path, hash_chain=False)
        audit.log_event("request_result", index=0)
        ok, checked, _ = verify_chain(path)
        assert ok is True and checked == 1

    def test_tampering_is_detected(self, tmp_path):
        from minipcai.audit import verify_chain

        path = tmp_path / "audit.jsonl"
        audit = AuditLogger(path)
        for index in range(3):
            audit.log_event("request_result", index=index)
        lines = path.read_text(encoding="utf-8").splitlines()
        records = [json.loads(line) for line in lines]
        records[1]["index"] = 99  # edit the middle record
        path.write_text(
            "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
            encoding="utf-8",
        )
        ok, _checked, bad_line = verify_chain(path)
        assert ok is False and bad_line == 2

    def test_removed_record_is_detected(self, tmp_path):
        from minipcai.audit import verify_chain

        path = tmp_path / "audit.jsonl"
        audit = AuditLogger(path)
        for index in range(3):
            audit.log_event("request_result", index=index)
        lines = path.read_text(encoding="utf-8").splitlines()
        path.write_text("\n".join([lines[0], lines[2]]) + "\n", encoding="utf-8")
        ok, _checked, bad_line = verify_chain(path)
        assert ok is False and bad_line == 2

    def test_missing_file_verifies_as_empty(self, tmp_path):
        from minipcai.audit import verify_chain

        assert verify_chain(tmp_path / "nope.jsonl") == (True, 0, 0)


class TestRotation:
    def test_rotation_keeps_bounded_file_count(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        audit = AuditLogger(path, max_bytes=400, max_files=3)
        for index in range(60):
            audit.log_event("request_result", index=index, padding="x" * 60)
        rotated = sorted(tmp_path.glob("audit.jsonl*"))
        assert len(rotated) <= 3
        assert path.is_file()
        # the active file stays small
        assert path.stat().st_size <= 400 * 2

    def test_rotated_files_still_verify(self, tmp_path):
        from minipcai.audit import verify_chain

        path = tmp_path / "audit.jsonl"
        audit = AuditLogger(path, max_bytes=400, max_files=3)
        for index in range(60):
            audit.log_event("request_result", index=index, padding="x" * 60)
        for candidate in tmp_path.glob("audit.jsonl*"):
            ok, _checked, bad_line = verify_chain(candidate)
            assert ok is True, (candidate, bad_line)

    def test_rotation_disabled(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        audit = AuditLogger(path, max_bytes=0, max_files=1)
        for index in range(10):
            audit.log_event("request_result", index=index, padding="x" * 200)
        assert path.stat().st_size > 1000
        assert not list(tmp_path.glob("audit.jsonl.1"))


class TestPrivacyMode:
    def test_plain_mode_stores_the_text(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        AuditLogger(path, store_text=True).log_event("request_result", text="geheim")
        assert read_audit_events(path)[0]["text"] == "geheim"

    def test_private_mode_redacts_but_keeps_a_fingerprint(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        AuditLogger(path, store_text=False).log_event("request_result", text="geheim")
        event = read_audit_events(path)[0]
        assert event["text"] == "[redacted]"
        assert event["text_len"] == len("geheim")
        assert len(event["text_sha256"]) == 16
        assert "geheim" not in path.read_text(encoding="utf-8")

    def test_fingerprint_helpers(self):
        from minipcai.audit import text_fingerprint

        first = text_fingerprint("öffne notepad")
        second = text_fingerprint("öffne notepad")
        third = text_fingerprint("beende notepad")
        assert first == second != third
