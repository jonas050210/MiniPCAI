"""The assistant pipeline.

Flow for every request:

    user text
      -> input validation
      -> intent classification (model)
      -> confidence / ambiguity gates
      -> logical target resolution (registry + strict parsers)
      -> security validation (policy: blocked names, extensions, trusted roots)
      -> confirmation gate for state-changing intents
      -> fail-closed audit checkpoint
      -> executor (dry-run or Windows)
      -> audit result record

Every request ends in an :class:`AssistantResult` with a human readable message
(in the configured language), a machine readable reason code and the i18n key
plus fields used to build that message, so frontends can render it differently
without re-deriving anything. Rejections never execute anything.

Confirmation: intents listed in ``policy.confirm_intents`` (``close_app`` and
``web_search`` by default) do NOT execute on the first call. ``handle`` returns
``status == "confirmation_required"`` together with a ``request_id``; the caller
then either calls :meth:`Assistant.confirm` or passes ``confirmed=True`` /
``auto_confirm=True`` (used by ``--yes`` in scripts).
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from minipcai import i18n
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
from minipcai.paths import resolve_registry_path
from minipcai.policy import (
    SecurityPolicy,
    contains_shell_syntax,
    default_policy,
    looks_like_path,
)
from minipcai.registry import Registry
from minipcai.security import SecurityError, SecurityValidator
from minipcai.targets import ActionPlan, TargetError, resolve_target

logger = logging.getLogger("minipcai.pipeline")

# Status values.
STATUS_OK = "ok"
STATUS_REJECTED = "rejected"
STATUS_ERROR = "error"
STATUS_CONFIRMATION_REQUIRED = "confirmation_required"

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
REASON_UNTRUSTED_TARGET = "untrusted_target"
REASON_AUDIT_UNAVAILABLE = "audit_unavailable"
REASON_EXECUTION_ERROR = "execution_error"
REASON_EXECUTOR_UNAVAILABLE = "executor_unavailable"
REASON_INTERNAL_ERROR = "internal_error"
REASON_CONFIRMATION_DECLINED = "confirmation_declined"
REASON_CONFIRMATION_NOT_PENDING = "confirmation_not_pending"
REASON_CANCELLED = "cancelled"

_TARGET_ERROR_REASONS = {
    "target_not_found": REASON_TARGET_NOT_FOUND,
    "target_ambiguous": REASON_TARGET_AMBIGUOUS,
    "target_mismatch": REASON_TARGET_MISMATCH,
    "invalid_parameter": REASON_INVALID_PARAMETER,
}

# i18n key per reason code (used when an error carries no dedicated key).
_REASON_KEYS = {
    REASON_INVALID_REQUEST: "invalid.empty",
    REASON_UNKNOWN_REQUEST: "unknown.request",
    REASON_LOW_CONFIDENCE: "low_confidence",
    REASON_AMBIGUOUS_INTENT: "ambiguous_intent",
    REASON_TARGET_NOT_FOUND: "target.not_found",
    REASON_TARGET_AMBIGUOUS: "target.ambiguous",
    REASON_TARGET_MISMATCH: "target.mismatch.generic",
    REASON_INVALID_PARAMETER: "invalid.empty",
    REASON_UNSAFE_REQUEST: "unsafe_request",
    REASON_UNTRUSTED_TARGET: "untrusted_target",
    REASON_AUDIT_UNAVAILABLE: "audit_unavailable",
    REASON_EXECUTION_ERROR: "execution_error",
    REASON_EXECUTOR_UNAVAILABLE: "executor_unavailable",
    REASON_INTERNAL_ERROR: "internal.model",
    REASON_CONFIRMATION_DECLINED: "confirmation.declined",
    REASON_CONFIRMATION_NOT_PENDING: "confirmation.not_pending",
    REASON_CANCELLED: "cancelled",
}

_MAX_PENDING = 16


class Executor(Protocol):
    """Interface every executor must implement."""

    mode: str

    def execute(self, plan: ActionPlan) -> ActionResult: ...


@dataclass(frozen=True)
class AssistantResult:
    """The outcome of one handled request."""

    status: str  # "ok" | "rejected" | "error" | "confirmation_required"
    message: str  # human readable, in the configured language
    intent: str | None = None
    confidence: float | None = None
    reason: str | None = None
    action_summary: str | None = None
    details: dict = field(default_factory=dict)
    request_id: str = ""
    message_key: str = ""
    message_fields: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK

    @property
    def needs_confirmation(self) -> bool:
        return self.status == STATUS_CONFIRMATION_REQUIRED

    def text(self, language: str | None = None) -> str:
        """The message rendered in ``language`` (falls back to the stored one)."""
        return i18n.render(self.message_key, language, self.message, **self.message_fields)

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
            "message_key": self.message_key,
            "message_fields": self.message_fields,
        }


@dataclass
class PendingRequest:
    """A validated, audited-but-not-yet-executed plan waiting for confirmation."""

    request_id: str
    plan: ActionPlan
    text: str
    confidence: float
    top_intents: list[tuple[str, float]]
    description: str
    question: str
    created_at: float = 0.0


class Assistant:
    """Orchestrates model, registry, executor and audit log."""

    def __init__(
        self,
        model: IntentModel,
        registry: Registry,
        executor: Executor,
        audit: AuditLogger,
        thresholds: Thresholds | None = None,
        policy: SecurityPolicy | None = None,
        language: str = "de",
        auto_confirm: bool = False,
    ):
        self.model = model
        self.registry = registry
        self.executor = executor
        self.audit = audit
        self.thresholds = thresholds or Thresholds()
        self.policy = policy or registry.policy or default_policy()
        self.language = i18n.normalize_language(language)
        self.auto_confirm = bool(auto_confirm)
        self._validator = SecurityValidator(registry, self.policy)
        self._pending: OrderedDict[str, PendingRequest] = OrderedDict()

    # -- helpers ---------------------------------------------------------------
    def _render(self, key: str, fallback: str = "", **fields) -> str:
        if not key:
            return fallback
        return i18n.render(key, self.language, fallback, **fields)

    def _reject(
        self,
        request_id: str,
        text: str,
        intent: str | None,
        confidence: float | None,
        reason: str,
        message: str,
        details: dict | None = None,
        message_key: str = "",
        message_fields: dict | None = None,
    ) -> AssistantResult:
        fields = message_fields or {}
        key = message_key or _REASON_KEYS.get(reason, "")
        localized = self._render(key, message, **fields)
        try:
            self.audit.log_rejected(request_id, text, intent, confidence, reason, message)
        except AuditError:
            logger.exception("Audit log failure while recording a rejection")
        return AssistantResult(
            status=STATUS_REJECTED,
            message=localized,
            intent=intent,
            confidence=confidence,
            reason=reason,
            details=details or {},
            request_id=request_id,
            message_key=key,
            message_fields=fields,
        )

    def _capabilities(self) -> dict[str, str]:
        return {"capabilities": self._render("unknown.capabilities")}

    # -- confirmation -----------------------------------------------------------
    @property
    def pending_requests(self) -> list[PendingRequest]:
        self._prune_pending()
        return list(self._pending.values())

    def _prune_pending(self) -> None:
        """Forget confirmations that were never answered in time."""
        timeout = float(getattr(self.policy, "confirmation_timeout", 0.0) or 0.0)
        if timeout <= 0:
            return
        deadline = time.monotonic() - timeout
        for request_id, pending in list(self._pending.items()):
            if pending.created_at and pending.created_at < deadline:
                self._pending.pop(request_id, None)

    def _needs_confirmation(self, plan: ActionPlan, confirmed: bool) -> bool:
        if confirmed or self.auto_confirm:
            return False
        return plan.intent in self.policy.confirm_intents

    def _remember(self, pending: PendingRequest) -> None:
        self._pending[pending.request_id] = pending
        while len(self._pending) > _MAX_PENDING:
            self._pending.popitem(last=False)

    def _ask_for_confirmation(
        self, text: str, plan: ActionPlan, confidence: float,
        top3: list[tuple[str, float]], request_id: str,
    ) -> AssistantResult:
        description = self._describe_plan(plan)
        answer = self._render("confirmation.yes") + "/" + self._render("confirmation.no")
        question = self._render(
            "confirmation.required", action=description + " ", yes=answer, no=answer
        )
        pending = PendingRequest(
            created_at=time.monotonic(),
            request_id=request_id,
            plan=plan,
            text=text,
            confidence=confidence,
            top_intents=top3,
            description=description,
            question=question,
        )
        self._remember(pending)
        try:
            self.audit.log_confirmation(request_id, plan.intent, "required", description)
        except AuditError:
            logger.exception("Could not record the confirmation request")
        return AssistantResult(
            status=STATUS_CONFIRMATION_REQUIRED,
            message=question,
            intent=plan.intent,
            confidence=confidence,
            action_summary=description,
            details={
                "pending_request_id": request_id,
                "description": description,
                "top_intents": top3,
            },
            request_id=request_id,
            message_key="confirmation.required",
            message_fields={"action": description + " ", "yes": answer, "no": answer},
        )

    def _describe_plan(self, plan: ActionPlan) -> str:
        entry = plan.entry
        if plan.intent == "close_app" and entry is not None:
            return f"close '{entry.id}' ({entry.path})"
        if plan.intent == "web_search" and entry is not None:
            return f"search the web for '{plan.query}' via '{entry.id}'"
        return plan.describe()

    def confirm(self, request_id: str, approved: bool = True) -> AssistantResult:
        """Execute a pending request after the user's decision."""
        self._prune_pending()
        pending = self._pending.pop(request_id, None)
        if pending is None:
            message = self._render("confirmation.not_pending")
            return AssistantResult(
                status=STATUS_REJECTED,
                message=message,
                reason=REASON_CONFIRMATION_NOT_PENDING,
                request_id=request_id,
                message_key="confirmation.not_pending",
            )
        try:
            self.audit.log_confirmation(
                request_id, pending.plan.intent,
                "approved" if approved else "declined", pending.description,
            )
        except AuditError:
            logger.exception("Could not record the confirmation decision")
        if not approved:
            return AssistantResult(
                status=STATUS_REJECTED,
                message=self._render("confirmation.declined"),
                intent=pending.plan.intent,
                confidence=pending.confidence,
                reason=REASON_CONFIRMATION_DECLINED,
                details={"description": pending.description},
                request_id=request_id,
                message_key="confirmation.declined",
            )
        return self._execute(
            pending.text, pending.plan, pending.confidence, pending.top_intents, request_id
        )

    # -- main entry point -------------------------------------------------------
    def handle(
        self,
        text: str,
        confirmed: bool = False,
        preferred_target: str | None = None,
    ) -> AssistantResult:
        request_id = uuid.uuid4().hex[:12]
        started = time.perf_counter()

        if not isinstance(text, str):
            return self._reject(
                request_id, "", None, None, REASON_INVALID_REQUEST,
                "Please enter your request as text.", message_key="invalid.not_text",
            )
        if not text.strip():
            return self._reject(
                request_id, "", None, None, REASON_INVALID_REQUEST,
                "Please enter a request.", message_key="invalid.empty",
            )
        if len(text) > MAX_INPUT_LENGTH:
            return self._reject(
                request_id,
                text,
                None,
                None,
                REASON_INVALID_REQUEST,
                f"Request is too long (maximum {MAX_INPUT_LENGTH} characters).",
                message_key="invalid.too_long",
                message_fields={"limit": MAX_INPUT_LENGTH},
            )

        # 0. Path guard: MiniPCAI never resolves targets from raw user paths,
        #    so a request that spells out a filesystem path is refused before
        #    the classifier can map it onto an action.
        if contains_shell_syntax(text):
            return self._reject(
                request_id, text, None, None, REASON_UNSAFE_REQUEST,
                "I do not run shell commands. Ask me for one registered action instead.",
                message_key="unsafe.shell_syntax",
            )

        if looks_like_path(text):
            return self._reject(
                request_id, text, None, None, REASON_TARGET_NOT_FOUND,
                "I only open names from your registry, not file paths. "
                "Ask me for the name of a registered entry instead.",
                details={"path_like": True},
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

        if not prediction.ranked:
            return self._reject(
                request_id, text, None, None, REASON_INTERNAL_ERROR,
                "The intent model returned no usable prediction.",
                message_key="internal.prediction",
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
                "Sorry, I don't know how to do that.",
                details={"top_intents": top3, **self._capabilities()},
                message_key="unknown.request",
                message_fields=self._capabilities(),
            )
        if top_confidence < self.thresholds.min_confidence:
            return self._reject(
                request_id, text, top_label, top_confidence, REASON_LOW_CONFIDENCE,
                f"I'm not confident enough about this request ({top_confidence:.0%}). "
                "Could you rephrase it more explicitly?",
                details={"top_intents": top3},
                message_fields={"confidence": f"{top_confidence:.0%}"},
            )
        margin = top_confidence - second_confidence
        if margin < self.thresholds.min_margin:
            return self._reject(
                request_id, text, top_label, top_confidence, REASON_AMBIGUOUS_INTENT,
                f"This request is ambiguous between '{top_label}' ({top_confidence:.0%}) "
                f"and '{second_label}' ({second_confidence:.0%}). Please be more specific.",
                details={"top_intents": top3, "margin": round(margin, 4)},
                message_fields={
                    "top": top_label,
                    "top_confidence": f"{top_confidence:.0%}",
                    "second": second_label,
                    "second_confidence": f"{second_confidence:.0%}",
                },
            )

        # 3. Logical target resolution.
        try:
            plan = resolve_target(
                top_label, text, self.registry, preferred_target=preferred_target
            )
        except TargetError as exc:
            reason = _TARGET_ERROR_REASONS.get(exc.reason, REASON_TARGET_NOT_FOUND)
            return self._reject(
                request_id, text, top_label, top_confidence, reason, exc.message,
                details={**exc.details, "top_intents": top3},
                message_key=exc.message_key,
                message_fields=exc.message_fields,
            )

        # 4. Security validation.
        try:
            self._validator.validate_plan(plan)
        except SecurityError as exc:
            return self._reject(
                request_id, text, top_label, top_confidence, exc.reason, exc.message,
                details={"top_intents": top3, "security_reason": exc.reason},
            )

        plan = self._with_request_id(plan, request_id)

        # 5. Confirmation gate for state-changing / outbound intents.
        if self._needs_confirmation(plan, confirmed):
            return self._ask_for_confirmation(
                text, plan, top_confidence, top3, request_id
            )

        return self._execute(text, plan, top_confidence, top3, request_id, started)

    # -- execution --------------------------------------------------------------
    @staticmethod
    def _with_request_id(plan: ActionPlan, request_id: str) -> ActionPlan:
        return ActionPlan(
            intent=plan.intent,
            entry=plan.entry,
            expression=plan.expression,
            duration_seconds=plan.duration_seconds,
            search_term=plan.search_term,
            search_roots=plan.search_roots,
            query=plan.query,
            request_id=request_id,
        )

    def _execute(
        self,
        text: str,
        plan: ActionPlan,
        top_confidence: float,
        top3: list[tuple[str, float]],
        request_id: str,
        started: float | None = None,
    ) -> AssistantResult:
        started = time.perf_counter() if started is None else started
        # Fail-closed audit checkpoint: the acceptance record MUST be writable
        # before anything is executed.
        try:
            self.audit.log_accepted(
                request_id, text, plan.intent, top_confidence,
                plan.entry.section if plan.entry else None,
                plan.entry.id if plan.entry else None,
                self.executor.mode,
            )
        except AuditError:
            logger.exception("Audit log unavailable; refusing to execute")
            return AssistantResult(
                status=STATUS_REJECTED,
                message=self._render("audit_unavailable"),
                intent=plan.intent,
                confidence=top_confidence,
                reason=REASON_AUDIT_UNAVAILABLE,
                details={"top_intents": top3},
                request_id=request_id,
                message_key="audit_unavailable",
            )

        duration_ms: int | None = None
        try:
            action_result = self.executor.execute(plan)
            duration_ms = int((time.perf_counter() - started) * 1000)
        except ExecutorUnavailable as exc:
            try:
                self.audit.log_action_result(
                    request_id, plan.intent, False, str(exc), self.executor.mode, None,
                )
            except AuditError:
                logger.exception("Could not write action result audit record")
            return AssistantResult(
                status=STATUS_ERROR,
                message=self._render("executor_unavailable", "", detail=str(exc)),
                intent=plan.intent,
                confidence=top_confidence,
                reason=REASON_EXECUTOR_UNAVAILABLE,
                action_summary=None,
                details={"top_intents": top3},
                request_id=request_id,
                message_key="executor_unavailable",
                message_fields={"detail": str(exc)},
            )
        except Exception as exc:  # noqa: BLE001 - executor bugs must not crash the UI
            logger.exception("Executor failed")
            action_result = ActionResult(
                ok=False, summary=f"Execution failed unexpectedly: {exc}"
            )

        # Audit the action result (never blocks the user response).
        try:
            self.audit.log_action_result(
                request_id, plan.intent, action_result.ok, action_result.summary,
                self.executor.mode, duration_ms,
            )
        except AuditError:
            logger.exception("Could not write action result audit record")

        summary = action_result.localized_summary(self.language)
        if action_result.ok:
            return AssistantResult(
                status=STATUS_OK,
                message=summary,
                intent=plan.intent,
                confidence=top_confidence,
                action_summary=action_result.summary,
                details={**action_result.details, "top_intents": top3},
                request_id=request_id,
                message_key=getattr(action_result, "key", ""),
                message_fields=getattr(action_result, "fields", {}),
            )
        return AssistantResult(
            status=STATUS_ERROR,
            message=summary,
            intent=plan.intent,
            confidence=top_confidence,
            reason=REASON_EXECUTION_ERROR,
            action_summary=action_result.summary,
            details={**action_result.details, "top_intents": top3},
            request_id=request_id,
            message_key=getattr(action_result, "key", "") or "execution_error",
            message_fields=getattr(action_result, "fields", {}) or {"detail": summary},
        )


def build_assistant(
    model_path: Path | str | None = None,
    registry_path: Path | str | None = None,
    audit_path: Path | str | None = None,
    executor_mode: str = "dry-run",
    policy: SecurityPolicy | None = None,
    language: str = "de",
    auto_confirm: bool = False,
    store_text: bool = True,
    check_files: bool = False,
) -> Assistant:
    """Convenience factory used by the CLI, the UI and the service layer.

    Paths default to the values from :mod:`minipcai.paths`: a checkout keeps
    using ``models/`` and ``data/``, an installed MiniPCAI uses its packaged
    defaults plus the writable per-user directory.
    """
    from minipcai.config import DEFAULT_MODEL_PATH, default_config

    if executor_mode not in ("dry-run", "windows"):
        raise ValueError(f"Unknown executor mode: {executor_mode!r}")

    config = default_config()
    policy = policy or config.policy
    model = SklearnIntentClassifier.load(model_path or DEFAULT_MODEL_PATH)
    if UNKNOWN_LABEL not in model.labels:
        raise ModelError("model artifact does not support the 'unknown' label")
    # Refuse to run with thresholds that are not backed by calibration data:
    # silently falling back to defaults would weaken the safety gates.
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
    else:
        logger.warning(
            "Model metadata contains no calibrated thresholds; using defaults "
            "(min_confidence=%s, min_margin=%s). Retrain with 'minipcai-train'.",
            thresholds.min_confidence,
            thresholds.min_margin,
        )

    registry = Registry.load(
        resolve_registry_path(registry_path), policy=policy, check_files=check_files
    )
    audit = AuditLogger(audit_path or config.audit_path, store_text=store_text)
    # Early warning only: the authoritative, fail-closed check happens per
    # request right before execution.
    try:
        audit.check_writable()
    except AuditError as exc:
        logger.warning("Audit log problem at startup: %s", exc)

    if executor_mode == "windows":
        executor: Executor = WindowsExecutor(
            audit=audit, max_active_timers=policy.max_active_timers
        )
    else:
        executor = DryRunExecutor(max_active_timers=policy.max_active_timers)
    return Assistant(
        model=model,
        registry=registry,
        executor=executor,
        audit=audit,
        thresholds=thresholds,
        policy=policy,
        language=language,
        auto_confirm=auto_confirm or config.auto_confirm,
    )
