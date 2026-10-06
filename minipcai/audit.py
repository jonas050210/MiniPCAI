"""Append-only JSONL audit logging for every request and action.

Security-relevant property: the audit log is *fail-closed*. Before an accepted
request is executed, the pipeline writes an ``request_accepted`` record; if
that write fails, the action is NOT executed and the request is rejected with
reason ``audit_unavailable``.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


class AuditError(RuntimeError):
    """Raised when the audit log cannot be written."""


class AuditLogger:
    """Thread-safe append-only JSONL audit logger."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._lock = threading.Lock()

    def check_writable(self) -> None:
        """Ensure the audit file can be written; raise :class:`AuditError` otherwise."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8"):
                pass
        except OSError as exc:
            raise AuditError(f"audit log is not writable at {self.path}: {exc}") from exc

    def log_event(self, event: str, **fields: Any) -> None:
        """Append one JSON line with ``event`` plus arbitrary structured fields."""
        record: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "schema_version": SCHEMA_VERSION,
            "event": event,
        }
        record.update(fields)
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            except (OSError, TypeError, ValueError) as exc:
                raise AuditError(f"failed to write audit record: {exc}") from exc

    # -- convenience wrappers ---------------------------------------------------
    def log_rejected(self, request_id: str, text: str, intent: str | None,
                     confidence: float | None, reason: str, message: str) -> None:
        self.log_event(
            "request_result",
            request_id=request_id,
            text=text,
            intent=intent,
            confidence=confidence,
            decision="rejected",
            reason=reason,
            message=message,
        )

    def log_accepted(self, request_id: str, text: str, intent: str,
                     confidence: float, target_kind: str | None,
                     target_id: str | None, executor_mode: str) -> None:
        self.log_event(
            "request_accepted",
            request_id=request_id,
            text=text,
            intent=intent,
            confidence=confidence,
            target_kind=target_kind,
            target_id=target_id,
            executor_mode=executor_mode,
        )

    def log_action_result(self, request_id: str, intent: str, ok: bool,
                          summary: str, executor_mode: str,
                          duration_ms: int | None = None) -> None:
        self.log_event(
            "action_result",
            request_id=request_id,
            intent=intent,
            ok=ok,
            summary=summary,
            executor_mode=executor_mode,
            duration_ms=duration_ms,
        )

    def log_timer_elapsed(self, request_id: str, duration_seconds: int) -> None:
        self.log_event(
            "timer_elapsed",
            request_id=request_id,
            duration_seconds=duration_seconds,
        )


def read_audit_events(path: Path | str) -> list[dict[str, Any]]:
    """Read an audit log back as a list of dictionaries (test helper)."""
    events: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events
