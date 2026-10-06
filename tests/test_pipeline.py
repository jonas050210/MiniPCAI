"""End-to-end pipeline tests: German requests in, safe results out."""

from __future__ import annotations

import pytest

from minipcai.audit import read_audit_events
from minipcai.pipeline import Assistant


class TestGermanRequestsSucceed:
    """One happy-path case per implemented MVP action."""

    CASES = [
        ("öffne notepad", "open_app"),
        ("starte den editor", "open_app"),
        ("schließe notepad", "close_app"),
        ("beende den browser", "close_app"),
        ("öffne wikipedia", "open_url"),
        ("öffne die webseite wikipedia", "open_url"),
        ("öffne die notizen", "open_file"),
        ("öffne die downloads", "open_folder"),
        ("finde die datei rechnung", "find_file"),
        ("wie hoch ist die cpu auslastung", "sys_cpu"),
        ("wie viel arbeitsspeicher ist frei", "sys_ram"),
        ("wie viel platz ist auf der festplatte", "sys_disk"),
        ("wie läuft der pc", "sys_summary"),
        ("was ist 12*4", "calc"),
        ("stelle einen timer auf 10 minuten", "timer"),
    ]

    @pytest.mark.parametrize(("text", "intent"), CASES)
    def test_request(self, make_assistant, text, intent):
        assistant = make_assistant()
        result = assistant.handle(text)
        assert result.status == "ok", f"{text}: {result.message}"
        assert result.intent == intent
        assert result.confidence >= assistant.thresholds.min_confidence


class TestTypoTolerance:
    """Typos in the *request wording* are tolerated by the classifier; typos in
    the *target name* are never guessed at - the request is rejected safely,
    with a suggestion."""

    @pytest.mark.parametrize(
        ("text", "intent"),
        [
            ("öfne notepad", "open_app"),
            ("wi viel arbeitsspeicher ist belegt", "sys_ram"),
            ("berchne 12*4", "calc"),
            ("stelle einen timr auf 5 miniten", "timer"),
        ],
    )
    def test_typos_still_work(self, make_assistant, text, intent):
        result = make_assistant().handle(text)
        assert result.status == "ok", f"{text}: {result.message}"
        assert result.intent == intent

    def test_typo_in_target_name_rejects_with_suggestion(self, make_assistant):
        result = make_assistant().handle("öffne die donloads")
        assert result.status == "rejected"
        assert result.reason == "target_not_found"
        assert "downloads" in result.message  # helpful hint, but no execution



class TestRejections:
    def test_empty_input(self, make_assistant):
        result = make_assistant().handle("   ")
        assert result.status == "rejected"
        assert result.reason == "invalid_request"

    def test_too_long_input(self, make_assistant):
        result = make_assistant().handle("öffne " + "notepad " * 100)
        assert result.status == "rejected"
        assert result.reason == "invalid_request"

    def test_unknown_request(self, make_assistant):
        result = make_assistant().handle("wie wird das wetter morgen")
        assert result.status == "rejected"
        assert result.reason == "unknown_request"

    def test_unsupported_destructive_request(self, make_assistant):
        result = make_assistant().handle("lösche die datei notizen")
        assert result.status == "rejected"
        assert result.reason in {
            "unknown_request", "low_confidence", "ambiguous_intent",
            "target_not_found", "target_mismatch", "target_ambiguous",
            "unsafe_request", "invalid_parameter",
        }

    def test_ambiguous_intent_request(self, make_assistant):
        result = make_assistant().handle("öffne das")
        assert result.status == "rejected"

    def test_ambiguous_target_rejected(self, make_assistant):
        result = make_assistant().handle("öffne firefox und chrome")
        assert result.status == "rejected"
        assert result.reason == "target_ambiguous"

    def test_target_not_found(self, make_assistant):
        result = make_assistant().handle("starte gimp")
        assert result.status == "rejected"
        assert result.reason == "target_not_found"

    def test_path_injection_rejected(self, make_assistant):
        result = make_assistant().handle("öffne C:\\Users\\hacker\\secrets.txt")
        assert result.status == "rejected"
        assert result.reason in {"target_not_found", "target_mismatch"}

    def test_cmd_not_in_registry(self, make_assistant):
        result = make_assistant().handle("starte cmd")
        assert result.status == "rejected"
        assert result.reason == "target_not_found"

    def test_calc_power_overflow_rejected(self, make_assistant):
        result = make_assistant().handle("was ist 2**2**2**2**2")
        assert result.status == "rejected"
        assert result.reason == "invalid_parameter"

    def test_calc_python_injection_rejected(self, make_assistant):
        result = make_assistant().handle("berechne __import__('os')")
        assert result.status == "rejected"

    def test_timer_out_of_bounds(self, make_assistant):
        result = make_assistant().handle("stelle einen timer auf 100000 stunden")
        assert result.status == "rejected"
        assert result.reason == "invalid_parameter"

    def test_find_file_search_is_confined_to_registry_roots(self, make_assistant):
        result = make_assistant().handle("finde ../../etc/passwd")
        # Either rejected, or executed as a harmless plain-name search.
        if result.status == "ok":
            assert "/" not in result.details.get("search_term", "")
        else:
            assert result.reason != "internal_error"

    def test_no_raw_user_text_leaks_into_plans(self, make_assistant):
        assistant = make_assistant()
        assistant.handle("öffne bitte mal den editor, danke")
        assert assistant.executor.plans, "dry-run executor should have recorded a plan"
        plan = assistant.executor.plans[-1]
        assert plan.intent == "open_app"
        assert plan.entry.id == "notepad"
        # The plan carries only registry data, never the raw sentence.
        assert "editor" not in str(plan.expression)
        assert plan.expression is None
        assert plan.search_term is None


class TestAuditTrail:
    def test_executed_request_logs_accept_and_result(self, make_assistant, tmp_path):
        audit_path = tmp_path / "audit.jsonl"
        assistant = make_assistant(audit_path=audit_path)
        result = assistant.handle("öffne notepad")
        assert result.status == "ok"
        events = read_audit_events(audit_path)
        assert [event["event"] for event in events] == [
            "request_accepted", "action_result",
        ]
        assert events[0]["request_id"] == result.request_id
        assert events[0]["intent"] == "open_app"
        assert events[0]["target_id"] == "notepad"
        assert events[0]["executor_mode"] == "dry-run"
        assert events[1]["ok"] is True

    def test_rejected_request_logs_single_record(self, make_assistant, tmp_path):
        audit_path = tmp_path / "audit.jsonl"
        assistant = make_assistant(audit_path=audit_path)
        assistant.handle("wie wird das wetter morgen")
        events = read_audit_events(audit_path)
        assert len(events) == 1
        assert events[0]["decision"] == "rejected"
        assert events[0]["reason"] == "unknown_request"
        assert assistant.executor.plans == []

    def test_audit_failure_blocks_execution(self, make_assistant, tmp_path):
        blocked = tmp_path / "audit_as_dir.jsonl"
        blocked.mkdir()  # make the audit target unwritable
        assistant = make_assistant(audit_path=blocked)
        result = assistant.handle("öffne notepad")
        assert result.status == "rejected"
        assert result.reason == "audit_unavailable"
        assert assistant.executor.plans == []

    def test_executor_unavailable_is_audited(self, make_assistant):
        from minipcai.actions import ExecutorUnavailable

        class UnavailableExecutor:
            mode = "test-unavailable"

            def execute(self, plan):
                raise ExecutorUnavailable("action not available here")

        assistant = make_assistant()
        assistant.executor = UnavailableExecutor()
        result = assistant.handle("öffne notepad")
        assert result.status == "error"
        assert result.reason == "executor_unavailable"
        # The audit trail must contain BOTH records: acceptance and outcome.
        events = read_audit_events(assistant.audit.path)
        assert [event["event"] for event in events] == [
            "request_accepted", "action_result",
        ]
        assert events[1]["ok"] is False
        assert "not available" in events[1]["summary"]


class TestWindowsExecutorMode:
    def test_os_action_requires_windows(self, make_assistant):
        assistant = make_assistant(executor_mode="windows")
        result = assistant.handle("öffne notepad")
        if result.status != "ok":  # on non-Windows test machines
            assert result.status == "error"
            assert result.reason == "executor_unavailable"
        else:  # genuinely on Windows: the action must have run for real
            assert result.action_summary

    def test_sys_info_works_with_windows_executor(self, make_assistant):
        assistant = make_assistant(executor_mode="windows")
        result = assistant.handle("wie viel arbeitsspeicher ist frei")
        assert result.status == "ok"
        assert result.intent == "sys_ram"
        assert "RAM" in result.message

    def test_find_file_with_windows_executor(self, make_assistant):
        assistant = make_assistant(executor_mode="windows")
        result = assistant.handle("finde die datei rechnung")
        assert result.status == "ok"
        assert result.intent == "find_file"


class TestResultObject:
    def test_to_dict_roundtrip(self, make_assistant):
        result = make_assistant().handle("was ist 12*4")
        data = result.to_dict()
        assert data["status"] == "ok"
        assert data["intent"] == "calc"
        assert data["request_id"] == result.request_id
        assert data["details"]["result"] == "48"

    def test_request_ids_are_unique(self, make_assistant):
        assistant = make_assistant()
        first = assistant.handle("öffne notepad")
        second = assistant.handle("öffne notepad")
        assert first.request_id != second.request_id

    def test_assistant_is_assistant_instance(self, make_assistant):
        assert isinstance(make_assistant(), Assistant)
