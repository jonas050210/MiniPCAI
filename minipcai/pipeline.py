"""The assistant pipeline.

Flow for every request:

    user text
      -> input validation
      -> intent classification (model)
      -> confidence / ambiguity gates
      -> logical target resolution (registry + strict parsers)
      -> security validation
      -> fail-closed audit checkpoint
      -> executor (dry-run or Windows)
      -> audit result record

Every request ends in an :class:`AssistantResult` with an English, human
readable message and a machine readable reason code. Rejections never execute
anything.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from minipcai.actions import (
    ActionResult,
    DryRunExecutor,
    ExecutorUnavailable,
    WindowsExecutor,
)
from minipcai.audit import AuditError, AuditLogger
from minipcai.config import MAX_INPUT_LENGTH, Thresholds
from minipcai.intents import UNKNOWN_LABEL
from minipcai.model import IntentModel, ModelError, SklearnIntentClassifier
from minipcai.registry import Registry
from minipcai.security import SecurityError, SecurityValidator
from minipcai.targets import ActionPlan, TargetError, resolve_target

logger = logging.getLogger("minipcai.pipeline")

# Status values.
STATUS_OK = "ok"
STATUS_REJECTED = "rejected"
STATUS_ERROR = "error"

# Reason codes for rejected/error results.
REASON_INVALID_REQUEST = "invalid_request"
REASON_UNKNOWN_REQUEST = "unknown_request"
REASON_LOW_CONFIDENCE = "low_confidence"
REASON_AMBIGUOUS_INTENT = "ambiguous_intent"
REASON_TARGET_NOT_FOUND = "target_not_found"
REASON_TARGET_AMBIGUOUS = "target_ambiguous"
REASON_TARGET_MISMATCH = "target_mismatch"
REASON_INVALID_PARAMETER = "invalid_parameter"
REASON_UNSAFE_REQUEST = "unsafe_request"
REASON_AUDIT_UNAVAILABLE = "audit_unavailable"
REASON_EXECUTION_ERROR = "execution_error"
REASON_EXECUTOR_UNAVAILABLE = "executor_unavailable"
REASON_INTERNAL_ERROR = "internal_error"

_TARGET_ERROR_REASONS = {
    "target_not_found": REASON_TARGET_NOT_FOUND,
    "target_ambiguous": REASON_TARGET_AMBIGUOUS,
    "target_mismatch": REASON_TARGET_MISMATCH,
    "invalid_parameter": REASON_INVALID_PARAMETER,
}


class Executor(Protocol):
    """Interface every executor must implement."""

    mode: str

    def execute(self, plan: ActionPlan) -> ActionResult: ...


@dataclass(frozen=True)
class AssistantResult:
    """The outcome of one handled request."""

    status: str  # "ok" | "rejected" | "error"
    message: str  # English, human readable
    intent: str | None = None
    confidence: float | None = None
    reason: str | None = None
    action_summary: str | None = None
    details: dict = field(default_factory=dict)
    request_id: str = ""

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "message": self.message,
            "intent": self.intent,
            "confidence": self.confidence,
            "reason": self.reason,
            "action_summary": self.action_summary,
            "details": self.details,
            "request_id": self.request_id,
        }


class Assistant:
    """Orchestrates model, registry, executor and audit log."""

    def __init__(
        self,
        model: IntentModel,
        registry: Registry,
        executor: Executor,
        audit: AuditLogger,
        thresholds: Thresholds | None = None,
    ):
        self.model = model
        self.registry = registry
        self.executor = executor
        self.audit = audit
        self.thresholds = thresholds or Thresholds()

    # -- helpers ---------------------------------------------------------------
    def _reject(
        self,
        request_id: str,
        text: str,
        intent: str | None,
        confidence: float | None,
        reason: str,
        message: str,
        details: dict | None = None,
    ) -> AssistantResult:
        try:
            self.audit.log_rejected(request_id, text, intent, confidence, reason, message)
        except AuditError:
            logger.exception("Audit log failure while recording a rejection")
        return AssistantResult(
            status=STATUS_REJECTED,
            message=message,
            intent=intent,
            confidence=confidence,
            reason=reason,
            details=details or {},
            request_id=request_id,
        )

    # -- main entry point -------------------------------------------------------
    def handle(self, text: str) -> AssistantResult:
        request_id = uuid.uuid4().hex[:12]
        started = time.perf_counter()

        if not isinstance(text, str) or not text.strip():
            return self._reject(
                request_id, "", None, None, REASON_INVALID_REQUEST, "Please enter a request."
            )
        if len(text) > MAX_INPUT_LENGTH:
            return self._reject(
                request_id,
                text,
                None,
                None,
                REASON_INVALID_REQUEST,
                f"Request is too long (maximum {MAX_INPUT_LENGTH} characters).",
            )

        # 1. Intent classification.
        try:
            prediction = self.model.predict(text)
        except Exception as exc:  # noqa: BLE001 - last-resort guard for model bugs
            logger.exception("Intent model failed")
            return self._reject(
                request_id, text, None, None, REASON_INTERNAL_ERROR,
                "The intent model failed to process this request.",
                details={"error": str(exc)},
            )

        top_label, top_confidence = prediction.ranked[0]
        second_label, second_confidence = (
            prediction.ranked[1] if len(prediction.ranked) > 1 else ("", 0.0)
        )
        top3 = [(label, round(conf, 4)) for label, conf in prediction.ranked[:3]]

        # 2. Confidence and ambiguity gates.
        if top_label == UNKNOWN_LABEL:
            return self._reject(
                request_id, text, top_label, top_confidence, REASON_UNKNOWN_REQUEST,
                "Sorry, I don't know how to do that. I can open and close registered "
                "apps, files, folders and websites, find files, report system info, "
                "calculate and set timers.",
                details={"top_intents": top3},
            )
        if top_confidence < self.thresholds.min_confidence:
            return self._reject(
                request_id, text, top_label, top_confidence, REASON_LOW_CONFIDENCE,
                f"I'm not confident enough about this request ({top_confidence:.0%}). "
                "Could you rephrase it more explicitly?",
                details={"top_intents": top3},
            )
        margin = top_confidence - second_confidence
        if margin < self.thresholds.min_margin:
            return self._reject(
                request_id, text, top_label, top_confidence, REASON_AMBIGUOUS_INTENT,
                f"This request is ambiguous between '{top_label}' ({top_confidence:.0%}) "
                f"and '{second_label}' ({second_confidence:.0%}). Please be more specific.",
                details={"top_intents": top3, "margin": round(margin, 4)},
            )

        # 3. Logical target resolution.
        try:
            plan = resolve_target(top_label, text, self.registry)
        except TargetError as exc:
            reason = _TARGET_ERROR_REASONS.get(exc.reason, REASON_TARGET_NOT_FOUND)
            return self._reject(
                request_id, text, top_label, top_confidence, reason, exc.message,
                details={**exc.details, "top_intents": top3},
            )

        # 4. Security validation.
        validator = SecurityValidator(self.registry)
        try:
            validator.validate_plan(plan)
        except SecurityError as exc:
            return self._reject(
                request_id, text, top_label, top_confidence, exc.reason, exc.message,
                details={"top_intents": top3},
            )

        # 5. Fail-closed audit checkpoint: the acceptance record MUST be
        #    writable before anything is executed.
        plan = ActionPlan(
            intent=plan.intent,
            entry=plan.entry,
            expression=plan.expression,
            duration_seconds=plan.duration_seconds,
            search_term=plan.search_term,
            search_roots=plan.search_roots,
            request_id=request_id,
        )
        target_kind = plan.entry.section if plan.entry else None
        target_id = plan.entry.id if plan.entry else None
        try:
            self.audit.log_accepted(
                request_id, text, plan.intent, top_confidence,
                target_kind, target_id, self.executor.mode,
            )
        except AuditError:
            logger.exception("Audit log unavailable; refusing to execute")
            return self._reject(
                request_id, text, top_label, top_confidence, REASON_AUDIT_UNAVAILABLE,
                "The audit log is currently unavailable, so I refuse to execute "
                "anything for safety reasons.",
                details={"top_intents": top3},
            )

        # 6. Execution.
        duration_ms: int | None = None
        try:
            action_result = self.executor.execute(plan)
            duration_ms = int((time.perf_counter() - started) * 1000)
        except ExecutorUnavailable as exc:
            return AssistantResult(
                status=STATUS_ERROR,
                message=str(exc),
                intent=plan.intent,
                confidence=top_confidence,
                reason=REASON_EXECUTOR_UNAVAILABLE,
                action_summary=None,
                details={"top_intents": top3},
                request_id=request_id,
            )
        except Exception as exc:  # noqa: BLE001 - executor bugs must not crash the UI
            logger.exception("Executor failed")
            action_result = ActionResult(
                ok=False, summary=f"Execution failed unexpectedly: {exc}"
            )

        # 7. Audit the action result (never blocks the user response).
        try:
            self.audit.log_action_result(
                request_id, plan.intent, action_result.ok, action_result.summary,
                self.executor.mode, duration_ms,
            )
        except AuditError:
            logger.exception("Could not write action result audit record")

        if action_result.ok:
            return AssistantResult(
                status=STATUS_OK,
                message=action_result.summary,
                intent=plan.intent,
                confidence=top_confidence,
                action_summary=action_result.summary,
                details={**action_result.details, "top_intents": top3},
                request_id=request_id,
            )
        return AssistantResult(
            status=STATUS_ERROR,
            message=action_result.summary,
            intent=plan.intent,
            confidence=top_confidence,
            reason=REASON_EXECUTION_ERROR,
            action_summary=action_result.summary,
            details={**action_result.details, "top_intents": top3},
            request_id=request_id,
        )


def build_assistant(
    model_path: Path | str,
    registry_path: Path | str,
    audit_path: Path | str,
    executor_mode: str = "dry-run",
) -> Assistant:
    """Convenience factory used by the CLI and the UI."""
    model = SklearnIntentClassifier.load(model_path)
    if UNKNOWN_LABEL not in model.labels:
        raise ModelError("model artifact does not support the 'unknown' label")
    registry = Registry.load(registry_path)
    audit = AuditLogger(audit_path)
    thresholds = Thresholds()
    stored = model.metadata.get("thresholds")
    if isinstance(stored, dict):
        try:
            thresholds = Thresholds(
                min_confidence=float(stored["min_confidence"]),
                min_margin=float(stored["min_margin"]),
            )
        except (KeyError, TypeError, ValueError):
            logger.warning("Ignoring invalid thresholds in model metadata")
    executor: Executor
    if executor_mode == "windows":
        executor = WindowsExecutor(audit=audit)
    elif executor_mode == "dry-run":
        executor = DryRunExecutor()
    else:
        raise ValueError(f"Unknown executor mode: {executor_mode!r}")
    return Assistant(
        model=model, registry=registry, executor=executor, audit=audit,
        thresholds=thresholds,
    )
