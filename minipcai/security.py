"""Defense-in-depth security validation of action plans.

Even though targets can only originate from the validated registry and
parameters only from strict parsers, every plan is re-checked here right
before execution. This module is the last line of defense; it never touches
the user's raw text.

The rules themselves live in :mod:`minipcai.policy` (data, not code), so a
deployment can tighten them without patching the validator:

* blocked executable names (shells, script hosts, LOLBins) are refused even if
  someone put them in the registry;
* ``open_file`` refuses anything that the Windows shell would *execute*
  (``.exe``, ``.lnk``, ``.bat``, ``.ps1``, ...) instead of opening;
* applications must live in a trusted location unless the policy explicitly
  allows untrusted paths;
* optional SHA-256 pins are verified against the real binary;
* plan field discipline: every intent may only carry the fields it needs.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from minipcai.calc_engine import CalcError
from minipcai.calc_engine import validate as validate_expression
from minipcai.config import (
    CALC_MAX_EXPRESSION_LENGTH,
    FIND_FILE_MAX_RESULTS,
    TIMER_MAX_SECONDS,
    TIMER_MIN_SECONDS,
    WEB_SEARCH_MAX_QUERY_LENGTH,
)
from minipcai.intents import ACTION_INTENTS, TARGETED_INTENTS
from minipcai.policy import (
    SecurityPolicy,
    default_policy,
    executable_name,
    expand_env,
)
from minipcai.registry import Registry
from minipcai.targets import ActionPlan

# Reason codes attached to SecurityError.
UNSAFE_REQUEST = "unsafe_request"
INVALID_PARAMETER = "invalid_parameter"
UNTRUSTED_TARGET = "untrusted_target"

# A conservative search term: plain words, digits, spaces and a few harmless
# separators. No path separators, no glob characters, no "..".
_SEARCH_TERM_RE = re.compile(r"^[\w\s.\-()]{1,100}$", re.UNICODE)

# Web queries may contain more punctuation (questions, quotes, operators) but
# never control characters and never anything that could change the target
# host: the executor percent-encodes the query into a fixed registry template.
_WEB_QUERY_RE = re.compile(
    r"^[\w\s.,;:!?'\"()\[\]+&%$#@~^*<>=/\-]{1,"
    + str(WEB_SEARCH_MAX_QUERY_LENGTH)
    + r"}$",
    re.UNICODE,
)

# Which plan fields each intent may use.
_ALLOWED_FIELDS: dict[str, frozenset[str]] = {
    "open_app": frozenset({"entry"}),
    "close_app": frozenset({"entry"}),
    "open_url": frozenset({"entry"}),
    "open_file": frozenset({"entry"}),
    "open_folder": frozenset({"entry"}),
    "find_file": frozenset({"search_term", "search_roots"}),
    "web_search": frozenset({"entry", "query"}),
    "sys_cpu": frozenset(),
    "sys_ram": frozenset(),
    "sys_disk": frozenset(),
    "sys_summary": frozenset(),
    "calc": frozenset({"expression"}),
    "timer": frozenset({"duration_seconds"}),
}

_PLAN_FIELDS = (
    "entry",
    "expression",
    "duration_seconds",
    "search_term",
    "search_roots",
    "query",
)


class SecurityError(Exception):
    """Raised when an action plan violates the security policy."""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason
        self.message = message


class SecurityValidator:
    """Validates action plans against the security policy."""

    def __init__(self, registry: Registry, policy: SecurityPolicy | None = None):
        self._registry = registry
        self._policy = policy or registry.policy or default_policy()

    @property
    def registry(self) -> Registry:
        return self._registry

    @property
    def policy(self) -> SecurityPolicy:
        return self._policy

    def validate_plan(self, plan: ActionPlan) -> None:
        """Raise :class:`SecurityError` if the plan is invalid or unsafe."""
        if plan.intent not in ACTION_INTENTS:
            raise SecurityError(UNSAFE_REQUEST, f"Intent '{plan.intent}' is not supported.")

        # Discipline check: only the fields allowed for this intent may be set.
        allowed = _ALLOWED_FIELDS[plan.intent]
        populated = {
            field_name
            for field_name in _PLAN_FIELDS
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
        elif plan.intent == "web_search":
            self._validate_web_search(plan)

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
            self._validate_app_entry(entry)
        elif entry.section == "files":
            self._validate_file_entry(entry)
        elif entry.section == "folders":
            self._validate_folder_entry(entry)

    def _validate_app_entry(self, entry) -> None:
        path = entry.path
        name = executable_name(path)
        if not name.endswith(".exe"):
            raise SecurityError(
                UNSAFE_REQUEST,
                f"Registered executable for '{entry.id}' must be an .exe file.",
            )
        if name in self._policy.blocked_executables:
            raise SecurityError(
                UNSAFE_REQUEST,
                f"Executable '{name}' is blocked by the security policy.",
            )
        if self._policy.enforce_trusted_app_roots and not self._policy.is_trusted_app_path(
            path
        ):
            raise SecurityError(
                UNTRUSTED_TARGET,
                f"'{entry.id}' points to '{path}', which is outside the trusted "
                "application locations. Update the registry or add the location to "
                "the configuration before this can run.",
            )
        pinned = entry.pinned_sha256 or self._policy.hash_for(path)
        if pinned:
            actual = _sha256_of_file(path)
            if actual is None:
                raise SecurityError(
                    UNTRUSTED_TARGET,
                    f"Could not read '{path}' to verify its pinned hash.",
                )
            if actual != pinned:
                raise SecurityError(
                    UNTRUSTED_TARGET,
                    f"'{entry.id}' does not match its pinned hash; refusing to run a "
                    "modified binary.",
                )

    def _validate_file_entry(self, entry) -> None:
        reason = self._policy.is_blocked_file_target(entry.path)
        if reason:
            raise SecurityError(
                UNSAFE_REQUEST,
                f"Registered file '{entry.id}' cannot be opened safely: {reason}.",
            )

    def _validate_folder_entry(self, entry) -> None:
        from minipcai.policy import extension_of

        extension = extension_of(entry.path)
        if extension and extension in self._policy.blocked_file_extensions:
            raise SecurityError(
                UNSAFE_REQUEST,
                f"Registered folder '{entry.id}' ends with the executable extension "
                f"'{extension}'.",
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

    def _validate_web_search(self, plan: ActionPlan) -> None:
        entry = plan.entry
        if entry is None:
            raise SecurityError(UNSAFE_REQUEST, "Web search has no search provider.")
        registered = self._registry.by_id(entry.id)
        if registered is None or registered != entry or entry.section != "searchers":
            raise SecurityError(
                UNSAFE_REQUEST,
                "Search provider is not part of the validated registry.",
            )
        if "{query}" not in entry.url_template:
            raise SecurityError(
                UNSAFE_REQUEST, "Search provider template is missing '{query}'."
            )
        query = plan.query or ""
        if not _WEB_QUERY_RE.match(query):
            raise SecurityError(
                INVALID_PARAMETER, "The search query contains invalid characters."
            )
        if not any(ch.isalnum() for ch in query):
            raise SecurityError(INVALID_PARAMETER, "The search query is too vague.")
        if len(query) > WEB_SEARCH_MAX_QUERY_LENGTH:
            raise SecurityError(INVALID_PARAMETER, "The search query is too long.")


def _sha256_of_file(path: str | Path) -> str | None:
    """Best-effort SHA-256 of a file; ``None`` when it cannot be read."""
    try:
        digest = hashlib.sha256()
        with open(expand_env(str(path)), "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None
