"""Action executors: ``DryRunExecutor`` for testing and ``WindowsExecutor`` for real work.

Executors receive fully validated :class:`~minipcai.targets.ActionPlan` objects
and never touch raw user text. The real executor always uses safe invocation
primitives:

* applications: ``subprocess.Popen([exe])`` with a list argv and ``shell=False``;
* files/folders: ``os.startfile`` (Windows shell integration, no shell);
* websites: ``webbrowser.open`` (registered URL only);
* web search: a *fixed* registry URL template with a percent-encoded query, so
  the host can never be influenced by the request;
* processes are only ever terminated when their executable path exactly
  matches a registry entry;
* ``find_file`` only walks registry-approved searchable folders and is bounded
  by both a visit counter and a wall-clock budget.

Results are structured (``key`` + ``fields`` + English ``summary``) so a
frontend can render them in any supported language without re-parsing text.

OS-integration actions (open/close app, open file/folder) require Windows;
informational actions (system info, file search, URLs, web search, calc) are
pure Python and run on any platform.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote

import psutil

from minipcai.audit import AuditLogger
from minipcai.calc_engine import evaluate as evaluate_expression
from minipcai.config import (
    FIND_FILE_MAX_RESULTS,
    FIND_FILE_MAX_VISITED,
    FIND_FILE_TIME_BUDGET_SECONDS,
)
from minipcai.i18n import t
from minipcai.targets import ActionPlan

logger = logging.getLogger("minipcai.actions")


@dataclass(frozen=True)
class ActionResult:
    """Outcome of one executed action.

    ``summary`` is the English rendering and stays stable for tests/logs;
    ``key`` + ``fields`` allow localized rendering via
    :meth:`localized_summary`.
    """

    ok: bool
    summary: str
    details: dict = field(default_factory=dict)
    key: str = ""
    fields: dict = field(default_factory=dict)
    cancelled: bool = False

    def localized_summary(self, language: str | None = None) -> str:
        if not self.key:
            return self.summary
        rendered = t(self.key, language, **self.fields)
        if self.key.startswith("action.dry_run") and not rendered.startswith("["):
            return rendered
        return rendered


class ExecutorUnavailable(RuntimeError):
    """Raised when an action cannot run on the current platform/configuration."""


def _require_windows(action: str) -> None:
    if sys.platform != "win32":
        raise ExecutorUnavailable(
            f"Action '{action}' requires Windows (current platform: {sys.platform}). "
            "Use the dry-run executor for testing."
        )


def _format_gb(value: float) -> str:
    return f"{value / (1024 ** 3):.1f} GB"


def _safe_resolve(value: str | Path) -> Path | None:
    """``Path(value).resolve()`` that returns ``None`` instead of raising."""
    try:
        return Path(value).resolve()
    except OSError:  # unreadable/invalid path: treat as "not the target"
        return None


def _system_drive() -> str:
    """The drive reported by ``sys_disk`` (Windows system drive, else POSIX root)."""
    if sys.platform != "win32":
        return "/"
    # Prefer the Windows system drive so the report matches the machine
    # instead of assuming "C:".
    drive = os.environ.get("SystemDrive", "").strip()
    if not drive:
        return "C:\\"
    return drive if drive.endswith("\\") else drive + "\\"


def _launch_kwargs() -> dict[str, Any]:
    """Detached-launch arguments (Windows only, so tests stay portable)."""
    if sys.platform != "win32":
        return {}
    flags = 0
    if hasattr(subprocess, "DETACHED_PROCESS"):
        flags |= subprocess.DETACHED_PROCESS  # type: ignore[attr-defined]
    if hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
        flags |= subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    return {
        "creationflags": flags,
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "close_fds": True,
    }


def build_search_url(template: str, query: str) -> str:
    """Percent-encode ``query`` into an approved search URL template."""
    encoded = quote(query, safe="")
    if "{query}" in template:
        return template.replace("{query}", encoded)
    return template + encoded


# ---------------------------------------------------------------------------
# Dry-run executor
# ---------------------------------------------------------------------------


class DryRunExecutor:
    """Describes actions instead of performing them; records every plan.

    Pure computations (``calc``) are still evaluated because they have no side
    effects. Everything that would touch the OS, processes or the file system
    is only described. The recorded plans make assertions in tests easy.
    """

    mode = "dry-run"

    def __init__(self, max_active_timers: int = 32) -> None:
        self.plans: list[ActionPlan] = []
        # Dry-run timers are *virtual*: nothing fires, but the timer panel and
        # the cap behave exactly like the real executor so the deployment can
        # be exercised without side effects.
        self._virtual_timers: list[dict[str, Any]] = []
        self._max_active_timers = max(1, int(max_active_timers))

    def execute(self, plan: ActionPlan, cancel_event=None, progress=None) -> ActionResult:
        self.plans.append(plan)
        entry = plan.entry
        if plan.intent == "open_app":
            return ActionResult(
                ok=True,
                summary=f"[dry-run] Would open app '{entry.id}' ({entry.path}).",
                key="action.dry_run.open_app",
                fields={"id": entry.id, "path": entry.path},
            )
        if plan.intent == "close_app":
            return ActionResult(
                ok=True,
                summary=f"[dry-run] Would terminate processes of '{entry.id}' ({entry.path}).",
                key="action.dry_run.close_app",
                fields={"id": entry.id, "path": entry.path},
            )
        if plan.intent == "open_url":
            return ActionResult(
                ok=True,
                summary=f"[dry-run] Would open website '{entry.id}' ({entry.url}).",
                key="action.dry_run.open_url",
                fields={"id": entry.id, "url": entry.url},
            )
        if plan.intent == "open_file":
            return ActionResult(
                ok=True,
                summary=f"[dry-run] Would open file '{entry.id}' ({entry.path}).",
                key="action.dry_run.open_file",
                fields={"id": entry.id, "path": entry.path},
            )
        if plan.intent == "open_folder":
            return ActionResult(
                ok=True,
                summary=f"[dry-run] Would open folder '{entry.id}' ({entry.path}).",
                key="action.dry_run.open_folder",
                fields={"id": entry.id, "path": entry.path},
            )
        if plan.intent == "find_file":
            roots = ", ".join(str(root) for root in plan.search_roots)
            return ActionResult(
                ok=True,
                summary=(
                    f"[dry-run] Would search for '{plan.search_term}' "
                    f"in {len(plan.search_roots)} approved folder(s): {roots}."
                ),
                details={"search_term": plan.search_term, "folders": roots},
                key="action.dry_run.find_file",
                fields={"term": plan.search_term, "count": len(plan.search_roots)},
            )
        if plan.intent == "web_search":
            url = build_search_url(plan.entry.url_template, plan.query or "")
            return ActionResult(
                ok=True,
                summary=f"[dry-run] Would search the web for '{plan.query}' ({url}).",
                details={"query": plan.query, "url": url},
                key="action.dry_run.web_search",
                fields={"query": plan.query, "url": url},
            )
        if plan.intent == "calc":
            result = evaluate_expression(plan.expression)
            return ActionResult(
                ok=True,
                summary=f"Calculated {plan.expression} = {result} (no side effects).",
                details={"expression": plan.expression, "result": result},
                key="action.calc.dry_run",
                fields={"expression": plan.expression, "result": result},
            )
        if plan.intent == "timer":
            self._prune_virtual_timers()
            if len(self._virtual_timers) >= self._max_active_timers:
                return ActionResult(
                    ok=False,
                    summary=(
                        f"{self._max_active_timers} timers are already running; "
                        "please wait or stop one."
                    ),
                    key="action.timer.limit",
                    fields={"max": self._max_active_timers},
                )
            self._virtual_timers.append(
                {
                    "request_id": plan.request_id,
                    "duration_seconds": plan.duration_seconds,
                    "expires_at": time.time() + max(0, plan.duration_seconds),
                }
            )
            return ActionResult(
                ok=True,
                summary=f"[dry-run] Would start a timer of {plan.duration_seconds} seconds.",
                details={"duration_seconds": plan.duration_seconds},
                key="action.dry_run.timer",
                fields={"seconds": plan.duration_seconds},
            )
        if plan.intent in {"sys_cpu", "sys_ram", "sys_disk", "sys_summary"}:
            return ActionResult(
                ok=True,
                summary=f"[dry-run] Would query {plan.intent.replace('sys_', '').upper()}.",
                key="action.dry_run.sys",
            )
        raise ExecutorUnavailable(f"Unknown intent '{plan.intent}'.")

    # -- virtual timer control (mirrors WindowsExecutor) -------------------------
    def _prune_virtual_timers(self) -> None:
        now = time.time()
        self._virtual_timers = [
            entry for entry in self._virtual_timers if entry["expires_at"] > now
        ]

    def active_timers(self) -> list[dict[str, Any]]:
        self._prune_virtual_timers()
        now = time.time()
        return [
            dict(
                entry,
                remaining_seconds=max(0, int(entry["expires_at"] - now)),
            )
            for entry in self._virtual_timers
        ]

    def cancel_timers(self, request_id: str | None = None) -> int:
        self._prune_virtual_timers()
        if request_id is None:
            cancelled = len(self._virtual_timers)
            self._virtual_timers = []
            return cancelled
        keep = [
            entry for entry in self._virtual_timers if entry["request_id"] != request_id
        ]
        cancelled = len(self._virtual_timers) - len(keep)
        self._virtual_timers = keep
        return cancelled


# ---------------------------------------------------------------------------
# Real (Windows) executor
# ---------------------------------------------------------------------------


class WindowsExecutor:
    """Performs the real actions on the Windows machine."""

    mode = "windows"

    def __init__(
        self,
        audit: AuditLogger | None = None,
        max_active_timers: int = 32,
        find_file_time_budget: float = FIND_FILE_TIME_BUDGET_SECONDS,
    ):
        self._audit = audit
        self._timers: list[threading.Timer] = []
        self._timer_info: dict[int, dict[str, Any]] = {}
        self._max_active_timers = max(1, int(max_active_timers))
        self._find_file_time_budget = max(0.0, float(find_file_time_budget))

    # -- public API -------------------------------------------------------------
    def execute(self, plan: ActionPlan, cancel_event=None, progress=None) -> ActionResult:
        handler = {
            "open_app": self._open_app,
            "close_app": self._close_app,
            "open_url": self._open_url,
            "open_file": self._open_file,
            "open_folder": self._open_file,  # same primitive, open with the shell
            "find_file": self._find_file,
            "web_search": self._web_search,
            "sys_cpu": self._sys_cpu,
            "sys_ram": self._sys_ram,
            "sys_disk": self._sys_disk,
            "sys_summary": self._sys_summary,
            "calc": self._calc,
            "timer": self._timer,
        }.get(plan.intent)
        if handler is None:
            raise ExecutorUnavailable(f"Unknown intent '{plan.intent}'.")
        if plan.intent == "find_file":
            return handler(plan, cancel_event=cancel_event, progress=progress)
        return handler(plan)

    # -- applications ------------------------------------------------------------
    def _open_app(self, plan: ActionPlan) -> ActionResult:
        _require_windows(plan.intent)
        executable = Path(plan.entry.path)
        if not executable.is_file():
            return ActionResult(
                ok=False,
                summary=f"Application not found: {executable}.",
                key="action.open_app.missing",
                fields={"path": str(executable)},
            )
        # Safe launch: list argv, shell disabled, no user-controlled parts.
        # The working directory is the application directory (derived from the
        # registry path, never from user input): many Windows apps fail to
        # start correctly without it.
        try:
            subprocess.Popen(
                [str(executable)],
                shell=False,
                cwd=str(executable.parent),
                **_launch_kwargs(),
            )
        except OSError as exc:
            logger.warning("Could not start %s: %s", executable, exc)
            return ActionResult(
                ok=False,
                summary=f"Could not start '{plan.entry.id}': {exc}.",
                key="action.open_app.failed",
                fields={"id": plan.entry.id, "detail": str(exc)},
            )
        return ActionResult(
            ok=True,
            summary=f"Opened app '{plan.entry.id}'.",
            key="action.open_app.ok",
            fields={"id": plan.entry.id},
        )

    def _close_app(self, plan: ActionPlan) -> ActionResult:
        _require_windows(plan.intent)
        target = Path(plan.entry.path)
        target_resolved = _safe_resolve(target)
        if target_resolved is None:
            return ActionResult(
                ok=False,
                summary=f"Could not resolve the registered path: {target}.",
                key="action.close_app.unresolved",
                fields={"path": str(target)},
            )
        terminated: list[int] = []
        errors: list[str] = []
        for process in psutil.process_iter(["pid", "exe"]):
            try:
                exe = process.info.get("exe")
            except psutil.Error:
                continue
            if not exe:
                continue
            # Security: only processes whose executable is exactly the
            # registered one are terminated.
            if _safe_resolve(exe) == target_resolved:
                try:
                    process.terminate()
                    terminated.append(process.pid)
                except (psutil.NoSuchProcess, psutil.AccessDenied) as exc:
                    errors.append(f"pid {process.pid}: {exc}")
        if terminated:
            summary = f"Terminated {len(terminated)} process(es) of '{plan.entry.id}'."
            key = "action.close_app.ok"
        else:
            summary = f"No running process of '{plan.entry.id}' found."
            key = "action.close_app.none"
        if errors:
            summary += f" ({len(errors)} could not be terminated.)"
        return ActionResult(
            ok=True,
            summary=summary,
            details={"terminated_pids": terminated, "errors": errors},
            key=key,
            fields={"id": plan.entry.id, "count": len(terminated)},
        )

    # -- urls / files / folders ---------------------------------------------------
    def _open_url(self, plan: ActionPlan) -> ActionResult:
        opened = webbrowser.open(plan.entry.url)
        if not opened:
            return ActionResult(
                ok=False, summary="No web browser available.",
                key="action.open_url.no_browser",
            )
        return ActionResult(
            ok=True,
            summary=f"Opened website '{plan.entry.id}' in the browser.",
            key="action.open_url.ok",
            fields={"id": plan.entry.id},
        )

    def _web_search(self, plan: ActionPlan) -> ActionResult:
        url = build_search_url(plan.entry.url_template, plan.query or "")
        opened = webbrowser.open(url)
        if not opened:
            return ActionResult(
                ok=False, summary="No web browser available.",
                key="action.web_search.no_browser",
            )
        return ActionResult(
            ok=True,
            summary=f"Started a web search for '{plan.query}' ({url}).",
            details={"query": plan.query, "url": url, "provider": plan.entry.id},
            key="action.web_search.ok",
            fields={"query": plan.query, "url": url},
        )

    def _open_file(self, plan: ActionPlan) -> ActionResult:
        _require_windows(plan.intent)
        target = Path(plan.entry.path)
        if not target.exists():
            return ActionResult(
                ok=False,
                summary=f"Path not found: {target}.",
                key="action.open_path.missing",
                fields={"path": str(target)},
            )
        if plan.intent == "open_folder" and not target.is_dir():
            return ActionResult(
                ok=False,
                summary=f"That target is not a folder: {target}.",
                key="action.open_folder.not_a_folder",
                fields={"path": str(target)},
            )
        # os.startfile uses the Windows shell association; no shell command is run.
        os.startfile(str(target))  # type: ignore[attr-defined]  # Windows only
        if plan.intent == "open_file":
            return ActionResult(
                ok=True,
                summary=f"Opened file '{plan.entry.id}'.",
                key="action.open_file.ok",
                fields={"id": plan.entry.id},
            )
        return ActionResult(
            ok=True,
            summary=f"Opened folder '{plan.entry.id}'.",
            key="action.open_folder.ok",
            fields={"id": plan.entry.id},
        )

    # -- file search -----------------------------------------------------------------
    def _find_file(self, plan: ActionPlan, cancel_event=None, progress=None) -> ActionResult:
        words = [word.lower() for word in plan.search_term.split() if word.strip()]
        matches: list[str] = []
        visited = 0
        truncated = False
        deadline = (
            time.monotonic() + self._find_file_time_budget
            if self._find_file_time_budget > 0
            else None
        )
        for root in plan.search_roots:
            if truncated:
                break
            try:
                for candidate in root.rglob("*"):
                    if cancel_event is not None and cancel_event.is_set():
                        truncated = True
                        break
                    visited += 1
                    if visited > FIND_FILE_MAX_VISITED or (
                        deadline is not None and time.monotonic() > deadline
                    ):
                        truncated = True
                        break
                    try:
                        if not candidate.is_file():
                            continue
                    except OSError:
                        continue
                    name = candidate.name.lower()
                    if all(word in name for word in words):
                        matches.append(str(candidate))
                        if len(matches) >= FIND_FILE_MAX_RESULTS:
                            break
            except OSError as exc:
                logger.warning("Could not search folder %s: %s", root, exc)
            if len(matches) >= FIND_FILE_MAX_RESULTS:
                break
            if progress is not None:
                try:
                    progress(f"searched {visited} entries, {len(matches)} matches")
                except Exception:  # pragma: no cover - progress must never break a search
                    logger.debug("progress callback failed", exc_info=True)
        if matches:
            listing = "\n".join(f"- {match}" for match in matches)
            summary = f"Found {len(matches)} file(s) matching '{plan.search_term}':\n{listing}"
            key = "action.find_file.results"
            fields = {"count": len(matches), "term": plan.search_term, "listing": listing}
        else:
            summary = (
                f"No files matching '{plan.search_term}' found in the approved folders."
            )
            key = "action.find_file.none"
            fields = {"term": plan.search_term}
        if truncated:
            summary += (
                f" (Search stopped after {visited} entries / "
                f"{self._find_file_time_budget:g} s; the result may be incomplete.)"
            )
        return ActionResult(
            ok=True,
            summary=summary,
            details={"matches": matches, "visited": visited, "truncated": truncated},
            key=key,
            fields=fields,
        )

    # -- system information -----------------------------------------------------------
    def _sys_cpu(self, plan: ActionPlan) -> ActionResult:
        usage = psutil.cpu_percent(interval=0.15)
        cores = psutil.cpu_count(logical=True)
        return ActionResult(
            ok=True,
            summary=f"CPU usage: {usage:.0f}% ({cores} logical cores).",
            details={"usage_percent": usage, "logical_cores": cores},
            key="action.sys_cpu",
            fields={"usage": f"{usage:.0f}", "cores": cores},
        )

    def _sys_ram(self, plan: ActionPlan) -> ActionResult:
        memory = psutil.virtual_memory()
        return ActionResult(
            ok=True,
            summary=(
                f"RAM: {_format_gb(memory.used)} of {_format_gb(memory.total)} used "
                f"({memory.percent:.0f}%), {_format_gb(memory.available)} available."
            ),
            details={
                "total_bytes": memory.total,
                "used_bytes": memory.used,
                "available_bytes": memory.available,
                "percent": memory.percent,
            },
            key="action.sys_ram",
            fields={
                "used": _format_gb(memory.used),
                "total": _format_gb(memory.total),
                "available": _format_gb(memory.available),
                "percent": f"{memory.percent:.0f}",
            },
        )

    def _sys_disk(self, plan: ActionPlan) -> ActionResult:
        root = _system_drive()
        usage = shutil.disk_usage(root)
        return ActionResult(
            ok=True,
            summary=(
                f"Disk ({root}): {_format_gb(usage.used)} of {_format_gb(usage.total)} used "
                f"({usage.used / usage.total * 100:.0f}%), {_format_gb(usage.free)} free."
            ),
            details={
                "root": root,
                "total_bytes": usage.total,
                "used_bytes": usage.used,
                "free_bytes": usage.free,
            },
            key="action.sys_disk",
            fields={
                "root": root,
                "used": _format_gb(usage.used),
                "total": _format_gb(usage.total),
                "free": _format_gb(usage.free),
                "percent": f"{usage.used / usage.total * 100:.0f}",
            },
        )

    def _sys_summary(self, plan: ActionPlan) -> ActionResult:
        cpu = self._sys_cpu(plan)
        ram = self._sys_ram(plan)
        disk = self._sys_disk(plan)
        body = "\n".join([f"CPU: {cpu.summary}", f"RAM: {ram.summary}", f"Disk: {disk.summary}"])
        return ActionResult(
            ok=True,
            summary="System summary:\n" + body,
            details=cpu.details | ram.details | disk.details,
            key="action.sys_summary",
            fields={"body": body},
        )

    # -- calc / timer -------------------------------------------------------------------
    def _calc(self, plan: ActionPlan) -> ActionResult:
        result = evaluate_expression(plan.expression)
        return ActionResult(
            ok=True,
            summary=f"{plan.expression} = {result}",
            details={"expression": plan.expression, "result": result},
            key="action.calc",
            fields={"expression": plan.expression, "result": result},
        )

    def _timer(self, plan: ActionPlan) -> ActionResult:
        duration = plan.duration_seconds
        self._prune_timers()
        if len(self._timers) >= self._max_active_timers:
            return ActionResult(
                ok=False,
                summary=(
                    f"{self._max_active_timers} timers are already running; "
                    "please wait or stop one."
                ),
                key="action.timer.limit",
                fields={"max": self._max_active_timers},
            )
        timer = threading.Timer(
            duration, self._on_timer_elapsed, args=(plan.request_id, duration)
        )
        timer.daemon = True
        timer.start()
        self._timers.append(timer)
        self._timer_info[id(timer)] = {
            "request_id": plan.request_id,
            "duration_seconds": duration,
            "started_at": time.time(),
        }
        minutes = duration // 60
        seconds = duration % 60
        human = f"{minutes} min {seconds} s" if minutes else f"{seconds} s"
        return ActionResult(
            ok=True,
            summary=f"Timer started: {human}.",
            details={"duration_seconds": duration},
            key="action.timer.started",
            fields={"human": human},
        )

    def _prune_timers(self) -> None:
        """Drop finished timers so the list cannot grow without bound."""
        alive = [timer for timer in self._timers if timer.is_alive()]
        keep = {id(timer) for timer in alive}
        self._timers = alive
        self._timer_info = {
            key: value for key, value in self._timer_info.items() if key in keep
        }

    # -- timer control (used by the UI/CLI timer panel) --------------------------
    def active_timers(self) -> list[dict[str, Any]]:
        self._prune_timers()
        return [
            dict(info, remaining_seconds=max(
                0, int(info["duration_seconds"] - (time.time() - info["started_at"]))
            ))
            for info in self._timer_info.values()
        ]

    def cancel_timers(self, request_id: str | None = None) -> int:
        """Cancel active timers (all of them, or only one request id)."""
        cancelled = 0
        for timer in list(self._timers):
            info = self._timer_info.get(id(timer), {})
            if request_id is not None and info.get("request_id") != request_id:
                continue
            if timer.is_alive():
                timer.cancel()
                cancelled += 1
        self._prune_timers()
        return cancelled

    def _on_timer_elapsed(self, request_id: str, duration: int) -> None:
        """Called from a timer thread when a timer finishes."""
        logger.info("Timer of %s seconds elapsed.", duration)
        if self._audit is not None:
            try:
                self._audit.log_timer_elapsed(request_id, duration)
            except Exception:  # pragma: no cover - defensive
                logger.exception("Could not write timer audit record")
