"""Defense-in-depth security validation of action plans.

Even though targets can only originate from the validated registry and
parameters only from strict parsers, every plan is re-checked here right
before execution. This module is the last line of defense; it never touches
the user's raw text.
"""

from __future__ import annotations

import re

from minipcai.calc_engine import CalcError
from minipcai.calc_engine import validate as validate_expression
from minipcai.config import (
    CALC_MAX_EXPRESSION_LENGTH,
    FIND_FILE_MAX_RESULTS,
    TIMER_MAX_SECONDS,
    TIMER_MIN_SECONDS,
)
from minipcai.intents import ACTION_INTENTS, TARGETED_INTENTS
from minipcai.registry import Registry
from minipcai.targets import ActionPlan

# Reason codes attached to SecurityError.
UNSAFE_REQUEST = "unsafe_request"
INVALID_PARAMETER = "invalid_parameter"

# Executables that must never be launchable through MiniPCAI, even if someone
# adds them to the registry by accident. The assistant has no use case for
# shells and script hosts; opening one would defeat the whole security model.
_BLOCKED_EXECUTABLES = frozenset(
    {
        # Windows command interpreters and script hosts
        "cmd.exe",
        "powershell.exe",
        "powershell_ise.exe",
        "pwsh.exe",
        "wscript.exe",
        "cscript.exe",
        "mshta.exe",
        # Troubleshooter/proxy execution helpers (known LOLBin abuse)
        "msdt.exe",
        "fodhelper.exe",
        "computerdefaults.exe",
        # Registration / control-panel tooling
        "regedit.exe",
        "regsvr32.exe",
        "rundll32.exe",
        "control.exe",
        # POSIX shells inside Windows
        "wsl.exe",
        "bash.exe",
        "sh.exe",
        "dash.exe",
        # Console host (never useful to launch directly)
        "conhost.exe",
    }
)

# A conservative search term: plain words, digits, spaces and a few harmless
# separators. No path separators, no glob characters, no "..".
_SEARCH_TERM_RE = re.compile(r"^[\w\s.\-()]{1,100}$", re.UNICODE)

# Which plan fields each intent may use.
_ALLOWED_FIELDS: dict[str, frozenset[str]] = {
    "open_app": frozenset({"entry"}),
    "close_app": frozenset({"entry"}),
    "open_url": frozenset({"entry"}),
    "open_file": frozenset({"entry"}),
    "open_folder": frozenset({"entry"}),
    "find_file": frozenset({"search_term", "search_roots"}),
    "sys_cpu": frozenset(),
    "sys_ram": frozenset(),
    "sys_disk": frozenset(),
    "sys_summary": frozenset(),
    "calc": frozenset({"expression"}),
    "timer": frozenset({"duration_seconds"}),
}


class SecurityError(Exception):
    """Raised when an action plan violates the security policy."""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason
        self.message = message


class SecurityValidator:
    """Validates action plans against the security policy."""

    def __init__(self, registry: Registry):
        self._registry = registry

    @property
    def registry(self) -> Registry:
        return self._registry

    def validate_plan(self, plan: ActionPlan) -> None:
        """Raise :class:`SecurityError` if the plan is invalid or unsafe."""
        if plan.intent not in ACTION_INTENTS:
            raise SecurityError(UNSAFE_REQUEST, f"Intent '{plan.intent}' is not supported.")

        # Discipline check: only the fields allowed for this intent may be set.
        allowed = _ALLOWED_FIELDS[plan.intent]
        populated = {
            field_name
            for field_name in (
                "entry", "expression", "duration_seconds", "search_term", "search_roots",
            )
            if getattr(plan, field_name) not in (None, ())
        }
        unexpected = populated - allowed
        if unexpected:
            raise SecurityError(
                UNSAFE_REQUEST,
                f"Plan for '{plan.intent}' carries unexpected data: {sorted(unexpected)}.",
            )
        missing = allowed - populated
        if missing:
            raise SecurityError(
                UNSAFE_REQUEST,
                f"Plan for '{plan.intent}' is missing required data: {sorted(missing)}.",
            )

        if plan.intent in TARGETED_INTENTS:
            self._validate_registry_entry(plan)
        elif plan.intent == "calc":
            self._validate_calc(plan)
        elif plan.intent == "timer":
            self._validate_timer(plan)
        elif plan.intent == "find_file":
            self._validate_find_file(plan)

    # -- per-intent checks ----------------------------------------------------
    def _validate_registry_entry(self, plan: ActionPlan) -> None:
        entry = plan.entry
        if entry is None:
            raise SecurityError(UNSAFE_REQUEST, "Plan has no registry entry.")
        # Provenance check: the entry must be EXACTLY an entry of THIS registry.
        # Comparing full identity (not just a known id) rejects forged entries
        # that reuse a registry id with a different executable/path/URL.
        registered = self._registry.by_id(entry.id)
        if registered is None or registered != entry:
            raise SecurityError(
                UNSAFE_REQUEST,
                f"Target '{entry.id}' is not part of the validated registry.",
            )
        if entry.section != TARGETED_INTENTS[plan.intent]:
            raise SecurityError(
                UNSAFE_REQUEST,
                f"Target '{entry.id}' is of kind '{entry.section}', "
                f"which does not match intent '{plan.intent}'.",
            )
        if entry.section == "apps":
            path = entry.path
            name = path.rsplit("\\", 1)[-1].rsplit("/", 1)[-1].lower()
            if not name.endswith(".exe"):
                raise SecurityError(
                    UNSAFE_REQUEST,
                    f"Registered executable for '{entry.id}' must be an .exe file.",
                )
            if name in _BLOCKED_EXECUTABLES:
                raise SecurityError(
                    UNSAFE_REQUEST,
                    f"Executable '{name}' is blocked by the security policy.",
                )

    def _validate_calc(self, plan: ActionPlan) -> None:
        expression = plan.expression or ""
        if len(expression) > CALC_MAX_EXPRESSION_LENGTH:
            raise SecurityError(INVALID_PARAMETER, "Expression is too long.")
        try:
            validate_expression(expression)
        except CalcError as exc:
            raise SecurityError(
                INVALID_PARAMETER, f"Unsafe or invalid expression: {exc}"
            ) from exc

    def _validate_timer(self, plan: ActionPlan) -> None:
        seconds = plan.duration_seconds
        if not isinstance(seconds, int):
            raise SecurityError(INVALID_PARAMETER, "Timer duration must be an integer.")
        if not TIMER_MIN_SECONDS <= seconds <= TIMER_MAX_SECONDS:
            raise SecurityError(
                INVALID_PARAMETER,
                f"Timer duration must be between {TIMER_MIN_SECONDS} and "
                f"{TIMER_MAX_SECONDS} seconds.",
            )

    def _validate_find_file(self, plan: ActionPlan) -> None:
        term = plan.search_term or ""
        if not _SEARCH_TERM_RE.match(term) or ".." in term:
            raise SecurityError(
                INVALID_PARAMETER,
                "Search term contains invalid characters.",
            )
        if not any(ch.isalnum() for ch in term):
            raise SecurityError(INVALID_PARAMETER, "Search term is too vague.")
        approved_roots = {folder.path for folder in self._registry.searchable_folders()}
        for root in plan.search_roots:
            if str(root) not in approved_roots:
                raise SecurityError(
                    UNSAFE_REQUEST,
                    "Search root is not an approved, searchable registry folder.",
                )
        if FIND_FILE_MAX_RESULTS < 1:
            raise SecurityError(UNSAFE_REQUEST, "Search result limit is misconfigured.")
