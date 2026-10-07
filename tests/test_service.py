"""Tests for the conversational layer (sessions, clarifications, confirmations)."""

from __future__ import annotations

import threading
import time

import pytest

from minipcai.policy import default_policy
from minipcai.service import (
    STATUS_CLARIFICATION_REQUIRED,
    AssistantService,
)


@pytest.fixture()
def service_factory(make_assistant):
    """Build a service around an assistant; the caller controls auto_confirm."""

    def make(auto_confirm: bool = False, language: str = "en", policy=None):
        assistant = make_assistant(
            policy=policy or default_policy(), auto_confirm=auto_confirm, language=language
        )
        service = AssistantService(assistant, language=language)
        return service

    return make


@pytest.fixture()
def service(service_factory):
    service = service_factory()
    yield service
    service.shutdown()


class TestBasicFlow:
    def test_ask_returns_a_result_and_emits_events(self, service):
        events: list[str] = []
        service.subscribe(lambda event: events.append(event.kind))
        result = service.ask("öffne notepad")
        assert result.status == "ok"
        assert result.intent == "open_app"
        assert events == ["request", "result"]

    def test_unsubscribe_stops_events(self, service):
        events: list[str] = []
        unsubscribe = service.subscribe(lambda event: events.append(event.kind))
        unsubscribe()
        service.ask("öffne notepad")
        assert events == []

    def test_listener_errors_do_not_break_the_request(self, service):
        def boom(event):
            raise RuntimeError("listener bug")

        service.subscribe(boom)
        assert service.ask("öffne notepad").status == "ok"

    def test_german_replies(self, service_factory):
        service = service_factory(language="de")
        result = service.ask("öffne notepad")
        assert "öffnen" in result.message.lower()
        assert result.text("de") == result.message
        assert result.text("en") != result.message
        service.shutdown()


class TestConfirmation:
    def test_close_app_asks_first(self, service):
        result = service.ask("schließe notepad")
        assert result.status == "confirmation_required"
        assert result.details["pending_request_id"] == result.request_id

    def test_yes_executes(self, service):
        service.ask("schließe notepad")
        result = service.ask("ja")
        assert result.status == "ok"
        assert result.intent == "close_app"

    def test_no_declines(self, service):
        service.ask("schließe notepad")
        result = service.ask("nein")
        assert result.status == "rejected"
        assert result.reason == "confirmation_declined"
        # the plan must not have been executed
        assert service.assistant.executor.plans == []

    @pytest.mark.parametrize("word", ["ja", "Ja", "yes", "ok", "bitte."])
    def test_affirmative_variants(self, service, word):
        service.ask("schließe notepad")
        assert service.ask(word).status == "ok"

    def test_confirm_without_pending_is_rejected(self, service):
        result = service.confirm(True)
        assert result.status == "rejected"
        assert result.reason == "confirmation_not_pending"

    def test_unrelated_text_drops_the_pending_confirmation(self, service):
        service.ask("schließe notepad")
        result = service.ask("was ist 2+2")
        assert result.status == "ok"
        assert result.intent == "calc"
        assert service.confirm(True).reason == "confirmation_not_pending"

    def test_expired_confirmation_is_not_executed(self, make_assistant):
        policy = default_policy().with_confirmation_timeout(0.01)
        assistant = make_assistant(policy=policy, auto_confirm=False)
        service = AssistantService(assistant)
        assert service.ask("schließe notepad").status == "confirmation_required"
        time.sleep(0.05)
        result = service.ask("ja")
        assert result.reason == "confirmation_not_pending"
        assert assistant.executor.plans == []
        service.shutdown()

    def test_confirmation_event_is_emitted(self, service):
        events = []
        service.subscribe(lambda event: events.append(event.kind))
        service.ask("schließe notepad")
        service.ask("ja")
        assert "confirmation" in events

    def test_auto_confirm_skips_the_question(self, service_factory):
        service = service_factory(auto_confirm=True)
        assert service.ask("schließe notepad").status == "ok"
        service.shutdown()


class TestClarification:
    def test_ambiguous_target_opens_a_question(self, service):
        result = service.ask("öffne firefox und chrome")
        assert result.status == STATUS_CLARIFICATION_REQUIRED
        assert result.details["options"] == ["firefox", "chrome"]
        assert "1)" in result.message and "2)" in result.message

    def test_choosing_a_number_resolves_to_that_target(self, service):
        service.ask("öffne firefox und chrome")
        result = service.ask("2")
        assert result.status == "ok"
        assert result.intent == "open_app"
        plan = service.assistant.executor.plans[-1]
        assert plan.entry.id == "chrome"

    def test_out_of_range_number_keeps_the_question_open(self, service):
        service.ask("öffne firefox und chrome")
        result = service.ask("7")
        assert result.status == STATUS_CLARIFICATION_REQUIRED
        assert result.reason == "clarification_invalid"
        # a valid answer afterwards still works
        assert service.ask("1").status == "ok"

    def test_yes_is_not_a_valid_choice_for_multiple_options(self, service):
        service.ask("öffne firefox und chrome")
        result = service.ask("ja")
        assert result.reason == "confirmation_not_pending"

    def test_declining_a_single_suggestion_cancels(self, service):
        service.ask("öffne notizn")
        result = service.ask("nein")
        assert result.status == "rejected"
        assert result.reason == "confirmation_declined"

    def test_typo_suggestion_can_be_accepted(self, service):
        result = service.ask("öffne notizn")
        assert result.status == STATUS_CLARIFICATION_REQUIRED
        assert result.details["options"]  # the suggested entry id
        assert "notizen" in result.message  # ... shown with the user's wording
        accepted = service.ask("1")
        assert accepted.status == "ok"
        assert service.assistant.executor.plans[-1].entry.id == "notes"

    def test_unrelated_text_drops_the_question(self, service):
        service.ask("öffne firefox und chrome")
        result = service.ask("was ist 2+2")
        assert result.status == "ok"
        assert result.intent == "calc"


class TestAsync:
    def test_run_async_returns_the_result(self, service):
        handle = service.run_async("was ist 12*4")
        result = handle.result(timeout=10)
        assert result.status == "ok"
        assert result.details["result"] == "48"
        assert handle.done() is True

    def test_run_async_callback(self, service):
        results = []
        done = threading.Event()

        def on_result(result):
            results.append(result)
            done.set()

        service.run_async("öffne notepad", on_result=on_result)
        assert done.wait(10)
        assert results[0].status == "ok"

    def test_cancelled_task_reports_cancelled(self, service):
        handle = service.run_async("öffne notepad")
        handle.cancel()
        assert handle.cancelled is True
        assert handle.result(timeout=10).reason == "cancelled"

    def test_confirm_async_runs_in_the_worker(self, make_assistant):
        """The GUI must be able to approve without blocking its event loop."""
        assistant = make_assistant(auto_confirm=False)
        service = AssistantService(assistant)
        assert service.ask("schließe notepad").status == "confirmation_required"
        results = []
        done = threading.Event()

        def on_result(result):
            results.append(result)
            done.set()

        handle = service.confirm_async(True, on_result=on_result)
        assert done.wait(10)
        assert handle.result(timeout=10).status == "ok"
        assert [plan.intent for plan in assistant.executor.plans] == ["close_app"]
        service.shutdown()

    def test_confirm_async_reports_the_declined_event(self, make_assistant):
        assistant = make_assistant(auto_confirm=False)
        service = AssistantService(assistant)
        service.ask("schließe notepad")
        result = service.confirm_async(False).result(timeout=10)
        assert result.reason == "confirmation_declined"
        assert assistant.executor.plans == []
        service.shutdown()

    def test_shutdown_is_idempotent(self, service):
        service.shutdown()
        service.shutdown()


class TestTimers:
    def test_dry_run_timers_are_tracked_and_capped(self, make_assistant):
        from dataclasses import replace

        from minipcai.policy import default_policy as strict_policy

        policy = replace(strict_policy(), max_active_timers=2)
        service = AssistantService(make_assistant(policy=policy, auto_confirm=True))
        assert service.ask("stelle einen timer auf 5 minuten").status == "ok"
        assert service.ask("stelle einen timer auf 60 sekunden").status == "ok"
        assert len(service.active_timers()) == 2
        capped = service.ask("stelle einen timer auf 90 sekunden")
        assert capped.status == "error"
        assert capped.reason == "execution_error"
        assert service.cancel_timers() == 2
        assert service.active_timers() == []
        service.shutdown()

    def test_cancel_single_timer(self, make_assistant):
        service = AssistantService(make_assistant(auto_confirm=True))
        service.ask("stelle einen timer auf 5 minuten")
        service.ask("stelle einen timer auf 4 minuten")
        timers = service.active_timers()
        assert service.cancel_timers(timers[0]["request_id"]) == 1
        assert len(service.active_timers()) == 1
        service.shutdown()

    def test_expired_timers_disappear(self, make_assistant):
        service = AssistantService(make_assistant(auto_confirm=True))
        service.ask("stelle einen timer auf 1 sekunde")
        assert len(service.active_timers()) == 1
        time.sleep(1.1)
        assert service.active_timers() == []
        service.shutdown()


class TestTypoClarification:
    """A typo must produce "Meintest du ...?" and then the right action."""

    def test_mistyped_target_asks_back_and_then_executes(self, make_assistant):
        assistant = make_assistant()
        service = AssistantService(assistant)
        question = service.ask("öffne dokumnete")
        assert question.status == "clarification_required"
        assert "dokumente" in question.message
        assert question.details["options"] == ["documents"]
        confirmed = service.ask("ja")
        assert confirmed.status == "ok"
        assert confirmed.details["target_id"] == "documents"
        service.shutdown()

    def test_destructive_typo_is_never_executed_without_confirmation(self, make_assistant):
        assistant = make_assistant(auto_confirm=False)
        service = AssistantService(assistant)
        result = service.ask("schließe notepda")
        # either a clarification question or a refusal - never an execution
        assert result.status in {"clarification_required", "rejected"}
        assert assistant.executor.plans == []
        service.shutdown()
