"""The conversational service layer used by the CLI, the GUI and the server.

The pipeline answers *one* request. Users, however, expect a conversation:
they want to be asked back when several targets match, they want to confirm
before an app is closed, and they want to cancel a search that takes too long.
:class:`AssistantService` adds exactly that on top of the pipeline, without any
frontend or threading assumptions:

* **state machine** - a request can end in ``ok``/``rejected``/``error`` or wait
  for a *clarification* (which target did you mean?) or a *confirmation*
  (may I close this?);
* **events** - every transition is published to listeners, so a UI can render
  progress lines, timers and audit notifications without polling;
* **cancellation and background work** - :meth:`run_async` executes a request in
  a worker thread and returns a handle whose ``cancel()`` reaches the executor;
* **timers** - the service owns the timer list, so a frontend can display and
  cancel active timers.

Everything is plain Python (no Qt), which keeps it testable and reusable.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from minipcai.audit import AuditError
from minipcai.i18n import t
from minipcai.pipeline import (
    REASON_TARGET_AMBIGUOUS,
    REASON_TARGET_NOT_FOUND,
    STATUS_OK,
    STATUS_REJECTED,
    Assistant,
    AssistantResult,
)

logger = logging.getLogger("minipcai.service")

STATUS_CLARIFICATION_REQUIRED = "clarification_required"

# Words that answer a yes/no clarification or confirmation.
_AFFIRMATIVE = {
    "ja", "j", "yes", "y", "ok", "okay", "bitte", "sicher", "gerne", "mach das", "ja bitte",
}
_NEGATIVE = {
    "nein", "n", "no", "abbrechen", "stop", "stopp", "nicht", "lieber nicht", "nein danke",
}


@dataclass(frozen=True)
class ServiceEvent:
    """One observable transition of the service."""

    kind: str  # "request", "result", "confirmation", "clarification", "timer", "error"
    data: dict = field(default_factory=dict)


@dataclass
class _Clarification:
    """An open question the user has to answer."""

    original_text: str
    question: str
    options: list[str]  # registry entry ids
    reason: str
    labels: dict[str, str] = field(default_factory=dict)  # id -> display name


class TaskHandle:
    """Handle for a background request."""

    def __init__(self, future: Future, cancel_event: threading.Event):
        self.future = future
        self.cancel_event = cancel_event

    def cancel(self) -> None:
        self.cancel_event.set()

    @property
    def cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def result(self, timeout: float | None = None) -> AssistantResult:
        return self.future.result(timeout=timeout)

    def done(self) -> bool:
        return self.future.done()


class AssistantService:
    """Turn-by-turn conversation on top of :class:`~minipcai.pipeline.Assistant`."""

    def __init__(
        self,
        assistant: Assistant,
        max_workers: int = 2,
        language: str | None = None,
    ):
        self.assistant = assistant
        self.language = language or assistant.language
        self._listeners: list[Callable[[ServiceEvent], None]] = []
        self._clarification: _Clarification | None = None
        self._executor_pool = ThreadPoolExecutor(
            max_workers=max(1, max_workers), thread_name_prefix="minipcai"
        )

    # -- events ---------------------------------------------------------------
    def subscribe(self, listener: Callable[[ServiceEvent], None]) -> Callable[[], None]:
        self._listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return unsubscribe

    def _emit(self, kind: str, **data: Any) -> None:
        event = ServiceEvent(kind=kind, data=data)
        for listener in list(self._listeners):
            try:
                listener(event)
            except Exception:  # pragma: no cover - listener bugs must not break chat
                logger.exception("Service event listener failed")

    # -- synchronous API -------------------------------------------------------
    def ask(self, text: str) -> AssistantResult:
        """Handle one user turn, including clarification/confirmation state."""
        if self._clarification is not None:
            answer = self._interpret_answer(text)
            if answer is not None:
                return self._resolve_clarification(answer)
            # Not a usable answer: drop the question and treat the text as new.
            self._clarification = None

        if self._pending_confirmation() is not None:
            answer = self._interpret_answer(text)
            if answer is not None:
                return self.confirm(answer)
            self._clear_pending_confirmation()

        if self._clarification is None:
            stray = self._interpret_answer(text)
            if stray is not None:
                # A bare "yes"/"no" without an open question must not be sent to
                # the classifier; answer plainly instead.
                self._emit("request", text=text)
                message = t("confirmation.not_pending", self.language)
                result = AssistantResult(
                    status=STATUS_REJECTED,
                    message=message,
                    reason="confirmation_not_pending",
                    message_key="confirmation.not_pending",
                )
                self._emit("result", status=result.status, message=message)
                return result

        self._emit("request", text=text)
        result = self.assistant.handle(text)
        return self._postprocess(text, result)

    def confirm(self, approved: bool) -> AssistantResult:
        """Answer the pending confirmation."""
        pending = self._pending_confirmation()
        if pending is None:
            message = t("confirmation.not_pending", self.language)
            return AssistantResult(
                status=STATUS_REJECTED,
                message=message,
                reason="confirmation_not_pending",
                message_key="confirmation.not_pending",
            )
        result = self.assistant.confirm(pending.request_id, approved=approved)
        self._emit(
            "confirmation",
            approved=approved,
            request_id=pending.request_id,
            intent=pending.plan.intent,
        )
        if approved:
            return self._postprocess(pending.text, result)
        return result

    def run_async(
        self,
        text: str,
        on_result: Callable[[AssistantResult], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> TaskHandle:
        """Run :meth:`ask` in a worker thread (keeps a GUI responsive)."""
        return self._submit(self.ask, text, on_result=on_result, cancel_event=cancel_event)

    def confirm_async(
        self,
        approved: bool,
        on_result: Callable[[AssistantResult], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> TaskHandle:
        """Run :meth:`confirm` in a worker thread.

        Confirming executes an action which may block (starting a program,
        closing an application); a GUI must not freeze while that happens.
        """
        return self._submit(self.confirm, approved, on_result=on_result, cancel_event=cancel_event)

    def _submit(
        self,
        function: Callable[..., AssistantResult],
        *args: Any,
        on_result: Callable[[AssistantResult], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> TaskHandle:
        """Run one service call in the worker pool and report its result."""
        event = cancel_event or threading.Event()

        def work() -> AssistantResult:
            if event.is_set():
                return self._cancelled_result()
            result = function(*args)
            if event.is_set() and result.status == STATUS_OK:
                return self._cancelled_result()
            return result

        future = self._executor_pool.submit(work)
        if on_result is not None:
            future.add_done_callback(
                lambda done: _safe_call(on_result, done.result() if not done.cancelled() else None)
            )
        return TaskHandle(future, event)

    def shutdown(self) -> None:
        self._executor_pool.shutdown(wait=False, cancel_futures=True)

    # -- timers ----------------------------------------------------------------
    def active_timers(self) -> list[dict[str, Any]]:
        """Running timers of the active executor (real or dry-run)."""
        executor = self.assistant.executor
        reader = getattr(executor, "active_timers", None)
        if callable(reader):
            try:
                return list(reader())
            except Exception:  # pragma: no cover - executor bugs must not crash the UI
                logger.exception("active_timers() failed")
        return []

    def cancel_timers(self, request_id: str | None = None) -> int:
        executor = self.assistant.executor
        canceller = getattr(executor, "cancel_timers", None)
        cancelled = 0
        if callable(canceller):
            try:
                cancelled = int(canceller(request_id))
            except Exception:  # pragma: no cover - executor bugs must not crash the UI
                logger.exception("cancel_timers() failed")
                cancelled = 0
        self._emit("timer", cancelled=cancelled, request_id=request_id)
        return cancelled

    # -- internals --------------------------------------------------------------
    def _pending_confirmation(self):
        pending = self.assistant.pending_requests
        return pending[-1] if pending else None

    def _clear_pending_confirmation(self) -> None:
        for pending in self.assistant.pending_requests:
            self.assistant.confirm(pending.request_id, approved=False)

    def _interpret_answer(self, text: str) -> bool | int | None:
        """Interpret ``text`` as 'yes'/'no'/option number, else ``None``."""
        normalized = (text or "").strip().lower().strip(".!")
        if not normalized:
            return None
        clarification = self._clarification
        if normalized.isdigit():
            if clarification is None:
                return None
            # Out-of-range numbers are returned as well: ``_resolve_clarification``
            # re-asks instead of silently dropping the question.
            return int(normalized)
        if normalized in _AFFIRMATIVE:
            return True
        if normalized in _NEGATIVE:
            return False
        return None

    def _postprocess(self, text: str, result: AssistantResult) -> AssistantResult:
        """Publish events and open clarifications for ambiguous targets."""
        self._emit(
            "result",
            status=result.status,
            intent=result.intent,
            reason=result.reason,
            message=result.message,
            request_id=result.request_id,
        )
        if result.status == STATUS_CLARIFICATION_REQUIRED:
            return result
        if (
            result.status == STATUS_REJECTED
            and result.reason == REASON_TARGET_AMBIGUOUS
        ):
            return self._open_clarification(text, result)
        if (
            result.status == STATUS_REJECTED
            and result.reason == REASON_TARGET_NOT_FOUND
            and result.details.get("suggestion")
        ):
            return self._open_clarification(text, result, single=True)
        return result

    def _open_clarification(
        self, text: str, result: AssistantResult, single: bool = False
    ) -> AssistantResult:
        candidates = result.details.get("candidates") or []
        suggestion = result.details.get("suggestion")
        labels = dict(result.details.get("option_labels") or {})
        if suggestion and result.details.get("suggestion_alias"):
            labels[suggestion] = result.details["suggestion_alias"]
        options = candidates if not single and candidates else (
            [suggestion] if suggestion else []
        )
        if not options:
            return result
        if len(options) == 1:
            display = labels.get(options[0], options[0])
            listed = f"'{display}'"
        else:
            listed = ", ".join(
                f"{index + 1}) '{labels.get(option, option)}'"
                for index, option in enumerate(options)
            )
        question = t("clarification.prompt", self.language, options=listed)
        self._clarification = _Clarification(
            original_text=text,
            question=question,
            options=list(options),
            reason=result.reason or "",
            labels=labels,
        )
        self._emit(
            "clarification",
            options=list(options),
            question=question,
            intent=result.intent,
        )
        return AssistantResult(
            status=STATUS_CLARIFICATION_REQUIRED,
            message=question,
            intent=result.intent,
            confidence=result.confidence,
            reason=result.reason,
            details={**result.details, "options": list(options)},
            request_id=result.request_id,
            message_key="clarification.prompt",
            message_fields={"options": listed},
        )

    def _resolve_clarification(self, answer: bool | int) -> AssistantResult:
        clarification = self._clarification
        assert clarification is not None
        self._clarification = None
        if answer is False:
            message = t("confirmation.declined", self.language)
            return AssistantResult(
                status=STATUS_REJECTED,
                message=message,
                reason="confirmation_declined",
                message_key="confirmation.declined",
            )
        if answer is True:
            if len(clarification.options) != 1:
                message = t("confirmation.not_pending", self.language)
                return AssistantResult(
                    status=STATUS_REJECTED,
                    message=message,
                    reason="confirmation_not_pending",
                )
            index = 1
        else:
            index = int(answer)
            if not 1 <= index <= len(clarification.options):
                message = t(
                    "clarification.invalid_choice",
                    self.language,
                    count=len(clarification.options),
                )
                self._clarification = clarification
                return AssistantResult(
                    status=STATUS_CLARIFICATION_REQUIRED,
                    message=message,
                    reason="clarification_invalid",
                    message_key="clarification.invalid_choice",
                    message_fields={"count": len(clarification.options)},
                )
        chosen = clarification.options[index - 1]
        # The chosen registry id is a *hint* to the resolver, not an addition to
        # the sentence: appending it would keep every original alias match alive
        # and the request would stay ambiguous forever.
        try:
            self.assistant.audit.log_clarification(
                "", None, "resolved", options=clarification.options, text=chosen
            )
        except AuditError:
            logger.exception("Could not record the clarification decision")
        result = self.assistant.handle(
            clarification.original_text, preferred_target=chosen
        )
        return self._postprocess(clarification.original_text, result)

    def _cancelled_result(self) -> AssistantResult:
        message = t("confirmation.declined", self.language)
        return AssistantResult(
            status=STATUS_REJECTED,
            message=message,
            reason="cancelled",
            message_key="confirmation.declined",
        )


def _safe_call(callback: Callable[[Any], None], value: Any) -> None:
    try:
        callback(value)
    except Exception:  # pragma: no cover - callback bugs must not kill threads
        logger.exception("async result callback failed")


def build_service(
    settings=None,
    auto_confirm: bool | None = None,
    language: str | None = None,
    executor_mode: str | None = None,
    check_files: bool = False,
) -> AssistantService:
    """Build a fully configured service from :class:`~minipcai.settings.Settings`."""
    from minipcai.pipeline import build_assistant
    from minipcai.settings import Settings

    settings = settings or Settings()
    assistant = build_assistant(
        model_path=settings.resolved_model_path(),
        registry_path=settings.resolved_registry_path(),
        audit_path=settings.resolved_audit_path(),
        executor_mode=executor_mode or settings.executor_mode,
        policy=settings.to_policy(),
        language=language or settings.language,
        auto_confirm=settings.auto_confirm if auto_confirm is None else auto_confirm,
        store_text=settings.store_text_in_audit,
        check_files=check_files,
    )
    return AssistantService(assistant, language=language or settings.language)
