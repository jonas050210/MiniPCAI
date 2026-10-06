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


class TestGeneralizationToNewPhrasings:
    """Requests that are NOT part of the training dataset.

    They verify that the classifier generalizes to unseen German phrasings
    instead of memorizing the dataset. The set is deliberately kept small and
    generic; it must never contain a text that also exists in the dataset
    (``test_no_case_is_part_of_the_dataset`` enforces exact-string disjointness).

    Scope of this evidence: it is a smoke test, not a statistical estimate of
    generalization. The set is small, and two of its in-scope cases
    (``was ergibt 25 mal 4``, ``stelle bitte einen timer auf 6 minuten``) are
    within edit distance 1 of a dataset entry, so near-duplicate robustness is
    not what this suite measures.
    """

    CASES = [
        ("ich hätte gerne den editor gestartet", "open_app"),
        ("bring mir bitte den browser hoch", "open_app"),
        ("kannst du den texteditor öffnen", "open_app"),
        ("mach den editor bitte zu", "close_app"),
        ("der browser soll sich schließen", "close_app"),
        ("beende firefox sofort bitte", "close_app"),
        ("zeig mir bitte die wikipedia seite", "open_url"),
        ("ich möchte zu wikipedia", "open_url"),
        ("öffne mir die datei rechnung", "open_file"),
        ("kannst du die rechnung aufmachen", "open_file"),
        ("zeig mir bitte die dokumente", "open_folder"),
        ("mach den bilder ordner mal auf", "open_folder"),
        ("wo finde ich die datei tabelle", "find_file"),
        ("kannst du nach der rechnung suchen", "find_file"),
        ("sag mir bitte die prozessorauslastung", "sys_cpu"),
        ("wie viel arbeitsspeicher ist belegt bitte", "sys_ram"),
        ("wie viel speicherplatz habe ich noch", "sys_disk"),
        ("wie geht es dem rechner gerade so", "sys_summary"),
        ("was ergibt 25 mal 4", "calc"),
        ("stelle bitte einen timer auf 6 minuten", "timer"),
    ]

    @pytest.mark.parametrize(("text", "intent"), CASES)
    def test_unseen_phrasing_is_understood(self, make_assistant, text, intent):
        result = make_assistant().handle(text)
        assert result.status == "ok", f"{text}: {result.message}"
        assert result.intent == intent

    @pytest.mark.parametrize(
        "text",
        ["wie wird das wetter in berlin", "schreib eine nachricht an max"],
    )
    def test_unseen_out_of_scope_request_is_rejected(self, make_assistant, text):
        result = make_assistant().handle(text)
        assert result.status == "rejected"
        assert result.reason in {
            "unknown_request", "low_confidence", "ambiguous_intent",
            "target_not_found", "target_mismatch", "target_ambiguous",
            "invalid_parameter", "unsafe_request",
        }

    def test_no_case_is_part_of_the_dataset(self):
        from minipcai.config import DEFAULT_DATASET_PATH
        from minipcai.dataset import load_dataset

        dataset_texts = {example.text.lower() for example in load_dataset(
            DEFAULT_DATASET_PATH
        ).examples}
        for text, _ in self.CASES:
            assert text.lower() not in dataset_texts, text


class TestRobustness:
    def test_model_failure_is_reported_not_raised(self, make_assistant):
        class BrokenModel:
            labels = ("open_app", "unknown")
            metadata: dict = {}

            def predict(self, text):
                raise RuntimeError("model exploded")

        assistant = make_assistant()
        assistant.model = BrokenModel()
        result = assistant.handle("öffne notepad")
        assert result.status == "rejected"
        assert result.reason == "internal_error"
        assert "model" in result.message

    def test_model_without_ranked_predictions(self, make_assistant):
        from minipcai.model import Prediction

        class EmptyModel:
            labels = ("open_app", "unknown")
            metadata: dict = {}

            def predict(self, text):
                return Prediction(label="open_app", confidence=0.0, ranked=())

        assistant = make_assistant()
        assistant.model = EmptyModel()
        result = assistant.handle("öffne notepad")
        assert result.status == "rejected"
        assert result.reason == "internal_error"

    def test_non_string_input_is_rejected(self, make_assistant):
        result = make_assistant().handle(None)  # type: ignore[arg-type]
        assert result.status == "rejected"
        assert result.reason == "invalid_request"

    def test_unknown_executor_mode_is_refused(self, trained_model, registry_factory,
                                              tmp_path):
        from minipcai.pipeline import build_assistant

        with pytest.raises(ValueError, match="Unknown executor mode"):
            build_assistant(
                model_path=trained_model,
                registry_path=registry_factory(),
                audit_path=tmp_path / "audit.jsonl",
                executor_mode="shell",
            )


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
