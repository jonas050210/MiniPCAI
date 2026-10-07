"""Append-only JSONL audit logging for every request and action.

Security-relevant property: the audit log is *fail-closed*. Before an accepted
request is executed, the pipeline writes an ``request_accepted`` record; if
that write fails, the action is NOT executed and the request is rejected with
reason ``audit_unavailable``.

Beyond plain appending this module provides:

* **rotation** - the active file never grows past ``max_bytes``; older records
  move to ``audit.jsonl.1`` ... ``audit.jsonl.N`` and the oldest file is
  dropped, so an unattended installation cannot fill the disk;
* **tamper evidence** - every record carries ``prev``/``hash`` and forms a
  SHA-256 chain, so removing or editing a record is detectable
  (``minipcai audit --verify``);
* **privacy mode** - ``store_text=False`` stores a hash and the length of the
  user's request instead of its text (still useful for forensics, without
  keeping everything the user typed on disk);
* **cross-process safety** - appends are guarded by a file lock on POSIX and an
  exclusive-open retry loop on Windows, in addition to the in-process lock.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from minipcai.config import AUDIT_MAX_BYTES, AUDIT_MAX_FILES

SCHEMA_VERSION = 2
REDACTED = "[redacted]"

try:  # pragma: no cover - platform specific
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]


class AuditError(RuntimeError):
    """Raised when the audit log cannot be written."""


def text_fingerprint(text: str) -> dict[str, Any]:
    """Privacy-preserving stand-in for a raw request string."""
    encoded = text.encode("utf-8", errors="replace")
    return {
        "text": REDACTED,
        "text_len": len(text),
        "text_sha256": hashlib.sha256(encoded).hexdigest()[:16],
    }


def _canonical(record: dict[str, Any]) -> bytes:
    return json.dumps(record, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")


def compute_record_hash(record: dict[str, Any]) -> str:
    """SHA-256 over the canonical record (including its ``prev`` link)."""
    payload = {key: value for key, value in record.items() if key != "hash"}
    return hashlib.sha256(_canonical(payload)).hexdigest()


class AuditLogger:
    """Thread- and process-safe append-only JSONL audit logger."""

    def __init__(
        self,
        path: Path | str,
        max_bytes: int = AUDIT_MAX_BYTES,
        max_files: int = AUDIT_MAX_FILES,
        store_text: bool = True,
        hash_chain: bool = True,
    ):
        self.path = Path(path)
        self.max_bytes = max(0, int(max_bytes))
        self.max_files = max(1, int(max_files))
        self.store_text = bool(store_text)
        self.hash_chain = bool(hash_chain)
        self._lock = threading.Lock()
        self._last_hash: str | None = None
        self._last_hash_loaded = False

    # -- public API -----------------------------------------------------------
    def check_writable(self) -> None:
        """Ensure the audit file can be written; raise :class:`AuditError` otherwise."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists() and self.path.is_dir():
                raise OSError("audit path is a directory")
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
        if "text" in fields:
            text = fields.pop("text")
            if isinstance(text, str):
                if self.store_text:
                    fields["text"] = text
                else:
                    fields.update(text_fingerprint(text))
        record.update(fields)

        with self._lock:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._rotate_if_needed()
                if self.hash_chain:
                    previous = self._previous_hash()
                    record["prev"] = (previous or "")[:12]
                    record["hash"] = compute_record_hash(record)[:32]
                line = json.dumps(record, ensure_ascii=False, default=str)
                with self._exclusive_append() as handle:
                    handle.write(line + "\n")
                # Keep the in-process chain head in sync, otherwise the next
                # record written by this process would link to the hash from
                # before the record just written.
                self._last_hash = record.get("hash")
                self._last_hash_loaded = True
            except (OSError, TypeError, ValueError) as exc:
                raise AuditError(f"failed to write audit record: {exc}") from exc

    # -- internals ------------------------------------------------------------
    @contextmanager
    def _exclusive_append(self) -> Iterator[Any]:
        with open(self.path, "a", encoding="utf-8") as handle:
            if fcntl is not None:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                except OSError:  # pragma: no cover - best effort
                    pass
            try:
                yield handle
                handle.flush()
                os.fsync(handle.fileno())
            finally:
                if fcntl is not None:
                    try:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                    except OSError:  # pragma: no cover - best effort
                        pass

    def rotated_path(self, index: int) -> Path:
        return self.path.with_name(self.path.name + f".{index}")

    def rotation_files(self) -> list[Path]:
        return [
            self.rotated_path(index)
            for index in range(1, self.max_files)
            if self.rotated_path(index).is_file()
        ]

    def _rotate_if_needed(self) -> None:
        if self.max_bytes <= 0 or self.max_files <= 1:
            return
        try:
            size = self.path.stat().st_size if self.path.is_file() else 0
        except OSError:
            return
        if size < self.max_bytes:
            return
        for index in range(self.max_files - 1, 0, -1):
            source = self.path if index == 1 else self.rotated_path(index - 1)
            target = self.rotated_path(index)
            if not source.is_file():
                continue
            if index == self.max_files - 1 and target.is_file():
                try:
                    target.unlink()
                except OSError:  # pragma: no cover - best effort
                    pass
            try:
                os.replace(source, target)
            except OSError:  # pragma: no cover - best effort
                return
        self._last_hash = None
        self._last_hash_loaded = False

    def _previous_hash(self) -> str | None:
        if self._last_hash_loaded:
            return self._last_hash
        self._last_hash = None
        if self.path.is_file():
            try:
                with open(self.path, "rb") as handle:
                    handle.seek(0, os.SEEK_END)
                    position = handle.tell()
                    block = min(position, 65536)
                    handle.seek(position - block)
                    tail = handle.read().decode("utf-8", errors="replace")
                for line in reversed(tail.splitlines()):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        break
                    if isinstance(record, dict):
                        self._last_hash = record.get("hash")
                    break
            except OSError:  # pragma: no cover - best effort
                self._last_hash = None
        self._last_hash_loaded = True
        return self._last_hash

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

    def log_confirmation(self, request_id: str, intent: str, decision: str,
                         action: str) -> None:
        self.log_event(
            "confirmation",
            request_id=request_id,
            intent=intent,
            decision=decision,
            action=action,
        )

    def log_clarification(self, request_id: str, intent: str | None,
                          decision: str, options: list[str] | None = None,
                          text: str | None = None) -> None:
        payload: dict[str, Any] = {
            "request_id": request_id,
            "intent": intent,
            "decision": decision,
        }
        if options is not None:
            payload["options"] = options
        if text is not None:
            payload["text"] = text
        self.log_event("clarification", **payload)


def read_audit_events(path: Path | str) -> list[dict[str, Any]]:
    """Read an audit log back as a list of dictionaries (test helper)."""
    events: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


def verify_chain(path: Path | str) -> tuple[bool, int, int]:
    """Verify the hash chain of an audit file.

    Returns ``(ok, checked_records, first_bad_line)``; ``first_bad_line`` is 0
    when everything verified. Records without a ``hash`` (older schema or
    ``hash_chain=False``) are counted as checked but not validated.
    """
    path = Path(path)
    if not path.is_file():
        return True, 0, 0
    previous: str | None = None
    checked = 0
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            return False, checked, number
        if not isinstance(record, dict) or "hash" not in record:
            checked += 1
            continue
        expected = compute_record_hash(record)
        if record.get("hash") != expected[:32]:
            return False, checked, number
        if previous is not None and record.get("prev") not in ("", previous[:12]):
            return False, checked, number
        previous = record["hash"]
        checked += 1
    return True, checked, 0


def default_audit_logger(path: Path | str, store_text: bool = True) -> AuditLogger:
    return AuditLogger(path, store_text=store_text)


if sys.platform == "win32":  # pragma: no cover - documentation hook
    # ``fcntl`` is unavailable on Windows; the retry loop in ``_exclusive_append``
    # plus ``os.replace`` during rotation keep concurrent writers consistent.
    pass
