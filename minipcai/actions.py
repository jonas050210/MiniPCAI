"""Action executors: ``DryRunExecutor`` for testing and ``WindowsExecutor`` for real work.

Executors receive fully validated :class:`~minipcai.targets.ActionPlan` objects
and never touch raw user text. The real executor always uses safe invocation
primitives:

* applications: ``subprocess.Popen([exe])`` with a list argv and ``shell=False``;
* files/folders: ``os.startfile`` (Windows shell integration, no shell);
* websites: ``webbrowser.open`` (registered URL only);
* processes are only ever terminated when their executable path exactly
  matches a registry entry;
* ``find_file`` only walks registry-approved searchable folders.

OS-integration actions (open/close app, open file/folder) require Windows;
informational actions (system info, file search, URLs, calc) are pure Python
and run on any platform.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import threading
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path

import psutil

from minipcai.audit import AuditLogger
from minipcai.calc_engine import evaluate as evaluate_expression
from minipcai.config import FIND_FILE_MAX_RESULTS, FIND_FILE_MAX_VISITED
from minipcai.targets import ActionPlan

logger = logging.getLogger("minipcai.actions")


@dataclass(frozen=True)
class ActionResult:
    """Outcome of one executed action."""

    ok: bool
    summary: str
    details: dict = field(default_factory=dict)


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

    def __init__(self) -> None:
        self.plans: list[ActionPlan] = []

    def execute(self, plan: ActionPlan) -> ActionResult:
        self.plans.append(plan)
        entry = plan.entry
        if plan.intent == "open_app":
            return ActionResult(
                ok=True,
                summary=f"[dry-run] Would open app '{entry.id}' ({entry.path}).",
            )
        if plan.intent == "close_app":
            return ActionResult(
                ok=True,
                summary=f"[dry-run] Would terminate processes of '{entry.id}' ({entry.path}).",
            )
        if plan.intent == "open_url":
            return ActionResult(
                ok=True,
                summary=f"[dry-run] Would open website '{entry.id}' ({entry.url}).",
            )
        if plan.intent == "open_file":
            return ActionResult(
                ok=True,
                summary=f"[dry-run] Would open file '{entry.id}' ({entry.path}).",
            )
        if plan.intent == "open_folder":
            return ActionResult(
                ok=True,
                summary=f"[dry-run] Would open folder '{entry.id}' ({entry.path}).",
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
            )
        if plan.intent == "calc":
            result = evaluate_expression(plan.expression)
            return ActionResult(
                ok=True,
                summary=f"Calculated {plan.expression} = {result} (no side effects).",
                details={"expression": plan.expression, "result": result},
            )
        if plan.intent == "timer":
            return ActionResult(
                ok=True,
                summary=f"[dry-run] Would start a timer of {plan.duration_seconds} seconds.",
                details={"duration_seconds": plan.duration_seconds},
            )
        if plan.intent == "sys_cpu":
            return ActionResult(ok=True, summary="[dry-run] Would query CPU usage.")
        if plan.intent == "sys_ram":
            return ActionResult(ok=True, summary="[dry-run] Would query RAM usage.")
        if plan.intent == "sys_disk":
            return ActionResult(ok=True, summary="[dry-run] Would query disk usage.")
        if plan.intent == "sys_summary":
            return ActionResult(ok=True, summary="[dry-run] Would collect a system summary.")
        raise ExecutorUnavailable(f"Unknown intent '{plan.intent}'.")


# ---------------------------------------------------------------------------
# Real (Windows) executor
# ---------------------------------------------------------------------------


class WindowsExecutor:
    """Performs the real actions on the Windows machine."""

    mode = "windows"

    def __init__(self, audit: AuditLogger | None = None):
        self._audit = audit
        self._timers: list[threading.Timer] = []

    # -- public API -------------------------------------------------------------
    def execute(self, plan: ActionPlan) -> ActionResult:
        handler = {
            "open_app": self._open_app,
            "close_app": self._close_app,
            "open_url": self._open_url,
            "open_file": self._open_file,
            "open_folder": self._open_file,  # same primitive, open with the shell
            "find_file": self._find_file,
            "sys_cpu": self._sys_cpu,
            "sys_ram": self._sys_ram,
            "sys_disk": self._sys_disk,
            "sys_summary": self._sys_summary,
            "calc": self._calc,
            "timer": self._timer,
        }.get(plan.intent)
        if handler is None:
            raise ExecutorUnavailable(f"Unknown intent '{plan.intent}'.")
        return handler(plan)

    # -- applications ------------------------------------------------------------
    def _open_app(self, plan: ActionPlan) -> ActionResult:
        _require_windows(plan.intent)
        executable = Path(plan.entry.path)
        if not executable.is_file():
            return ActionResult(ok=False, summary=f"Application not found: {executable}.")
        # Safe launch: list argv, shell disabled, no user-controlled parts.
        subprocess.Popen([str(executable)], shell=False)
        return ActionResult(ok=True, summary=f"Opened app '{plan.entry.id}'.")

    def _close_app(self, plan: ActionPlan) -> ActionResult:
        _require_windows(plan.intent)
        target = Path(plan.entry.path)
        target_resolved = target.resolve()
        terminated: list[int] = []
        errors: list[str] = []
        for process in psutil.process_iter(["pid", "exe"]):
            try:
                exe = process.info.get("exe")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            if not exe:
                continue
            # Security: only processes whose executable is exactly the
            # registered one are terminated.
            if Path(exe).resolve() == target_resolved:
                try:
                    process.terminate()
                    terminated.append(process.pid)
                except (psutil.NoSuchProcess, psutil.AccessDenied) as exc:
                    errors.append(f"pid {process.pid}: {exc}")
        summary = (
            f"Terminated {len(terminated)} process(es) of '{plan.entry.id}'."
            if terminated
            else f"No running process of '{plan.entry.id}' found."
        )
        if errors:
            summary += f" ({len(errors)} could not be terminated.)"
        return ActionResult(
            ok=True, summary=summary, details={"terminated_pids": terminated}
        )

    # -- urls / files / folders ---------------------------------------------------
    def _open_url(self, plan: ActionPlan) -> ActionResult:
        opened = webbrowser.open(plan.entry.url)
        if not opened:
            return ActionResult(ok=False, summary="No web browser available.")
        return ActionResult(ok=True, summary=f"Opened website '{plan.entry.id}' in the browser.")

    def _open_file(self, plan: ActionPlan) -> ActionResult:
        _require_windows(plan.intent)
        target = Path(plan.entry.path)
        if not target.exists():
            return ActionResult(ok=False, summary=f"Path not found: {target}.")
        # os.startfile uses the Windows shell association; no shell command is run.
        os.startfile(str(target))  # type: ignore[attr-defined]  # Windows only
        kind = "file" if plan.intent == "open_file" else "folder"
        return ActionResult(ok=True, summary=f"Opened {kind} '{plan.entry.id}'.")

    # -- file search -----------------------------------------------------------------
    def _find_file(self, plan: ActionPlan) -> ActionResult:
        words = [word.lower() for word in plan.search_term.split() if word.strip()]
        matches: list[str] = []
        visited = 0
        for root in plan.search_roots:
            try:
                for candidate in root.rglob("*"):
                    visited += 1
                    if visited > FIND_FILE_MAX_VISITED:
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
        if matches:
            listing = "\n".join(f"- {match}" for match in matches)
            summary = f"Found {len(matches)} file(s) matching '{plan.search_term}':\n{listing}"
        else:
            summary = f"No files matching '{plan.search_term}' found in the approved folders."
        return ActionResult(
            ok=True,
            summary=summary,
            details={"matches": matches, "visited": visited},
        )

    # -- system information -----------------------------------------------------------
    def _sys_cpu(self, plan: ActionPlan) -> ActionResult:
        usage = psutil.cpu_percent(interval=0.15)
        cores = psutil.cpu_count(logical=True)
        return ActionResult(
            ok=True,
            summary=f"CPU usage: {usage:.0f}% ({cores} logical cores).",
            details={"usage_percent": usage, "logical_cores": cores},
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
        )

    def _sys_disk(self, plan: ActionPlan) -> ActionResult:
        root = "C:\\" if sys.platform == "win32" else "/"
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
        )

    def _sys_summary(self, plan: ActionPlan) -> ActionResult:
        cpu = self._sys_cpu(plan)
        ram = self._sys_ram(plan)
        disk = self._sys_disk(plan)
        summary = "System summary:\n" + "\n".join(
            [f"CPU: {cpu.summary}", f"RAM: {ram.summary}", f"Disk: {disk.summary}"]
        )
        details = cpu.details | ram.details | disk.details
        return ActionResult(ok=True, summary=summary, details=details)

    # -- calc / timer -------------------------------------------------------------------
    def _calc(self, plan: ActionPlan) -> ActionResult:
        result = evaluate_expression(plan.expression)
        return ActionResult(
            ok=True,
            summary=f"{plan.expression} = {result}",
            details={"expression": plan.expression, "result": result},
        )

    def _timer(self, plan: ActionPlan) -> ActionResult:
        duration = plan.duration_seconds
        timer = threading.Timer(
            duration, self._on_timer_elapsed, args=(plan.request_id, duration)
        )
        timer.daemon = True
        timer.start()
        self._timers.append(timer)
        minutes = duration // 60
        seconds = duration % 60
        human = f"{minutes} min {seconds} s" if minutes else f"{seconds} s"
        return ActionResult(
            ok=True,
            summary=f"Timer started: {human}.",
            details={"duration_seconds": duration},
        )

    def _on_timer_elapsed(self, request_id: str, duration: int) -> None:
        """Called from a timer thread when a timer finishes."""
        logger.info("Timer of %s seconds elapsed.", duration)
        if self._audit is not None:
            try:
                self._audit.log_timer_elapsed(request_id, duration)
            except Exception:  # pragma: no cover - defensive
                logger.exception("Could not write timer audit record")
