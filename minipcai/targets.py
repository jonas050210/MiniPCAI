"""Logical target resolution: user text -> a validated :class:`ActionPlan`.

The resolver NEVER constructs Windows paths from user input. Targets are
looked up exclusively via alias matching against the validated registry.
Parameterized intents (``calc``, ``timer``, ``find_file``) get their values
from strict parsers in :mod:`minipcai.textutils`; those values are re-checked
by :mod:`minipcai.security` before execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from minipcai.intents import TARGETED_INTENTS
from minipcai.registry import Registry, RegistryEntry
from minipcai.textutils import (
    damerau_levenshtein,
    extract_math_expression,
    extract_search_term,
    normalize,
    parse_duration,
)

# Human-readable names per registry section, used in rejection messages.
_SECTION_KINDS = {"apps": "app", "files": "file", "folders": "folder",
                  "websites": "website"}

# Reason codes for target resolution failures.
TARGET_NOT_FOUND = "target_not_found"
TARGET_AMBIGUOUS = "target_ambiguous"
TARGET_MISMATCH = "target_mismatch"
INVALID_PARAMETER = "invalid_parameter"


class TargetError(Exception):
    """Raised when no safe, unambiguous target can be resolved."""

    def __init__(self, reason: str, message: str, details: dict | None = None):
        super().__init__(message)
        self.reason = reason
        self.message = message
        self.details = details or {}


@dataclass(frozen=True)
class ActionPlan:
    """A fully resolved, security-checkable action description.

    Only fields relevant to the intent are populated; the security validator
    enforces this discipline.
    """

    intent: str
    entry: RegistryEntry | None = None          # open/close app, url, file, folder
    expression: str | None = None               # calc
    duration_seconds: int | None = None         # timer
    search_term: str | None = None              # find_file
    search_roots: tuple[Path, ...] = field(default_factory=tuple)  # find_file
    request_id: str = ""                        # audit correlation


def _suggest_alias(text: str, registry: Registry, section: str) -> str | None:
    """Find a registry alias close to a word of the request (for hints only)."""
    words = normalize(text).split()
    best: tuple[int, str] | None = None
    for entry in registry.section(section):
        for alias in entry.aliases:
            normalized_alias = normalize(alias)
            for word in words:
                if len(word) < 4 or len(normalized_alias) < 4:
                    continue
                distance = damerau_levenshtein(word, normalized_alias)
                if distance <= (2 if len(normalized_alias) >= 7 else 1):
                    if best is None or distance < best[0] or (
                        distance == best[0] and len(normalized_alias) > len(best[1])
                    ):
                        best = (distance, normalized_alias)
    return best[1] if best else None


def _resolve_registry_target(intent: str, text: str, registry: Registry) -> ActionPlan:
    section = TARGETED_INTENTS[intent]
    matches = registry.find_matches(text, section)

    if len(matches) > 1:
        names = ", ".join(f"'{match.entry.id}'" for match in matches)
        raise TargetError(
            TARGET_AMBIGUOUS,
            f"Multiple {section} match this request ({names}). Please name exactly one.",
            details={"candidates": [match.entry.id for match in matches]},
        )

    if len(matches) == 1:
        return ActionPlan(intent=intent, entry=matches[0].entry)

    # No match in the expected section: check whether another section matches,
    # which usually means the user wants something of a different kind
    # (e.g. "öffne die downloads" classified as open_app).
    other_matches = {
        other_section: registry.find_matches(text, other_section)
        for other_section in set(TARGETED_INTENTS.values()) - {section}
    }
    hit_sections = [s for s, ms in other_matches.items() if ms]
    if len(hit_sections) == 1 and len(other_matches[hit_sections[0]]) == 1:
        entry = other_matches[hit_sections[0]][0].entry
        kind = _SECTION_KINDS[entry.section]
        raise TargetError(
            TARGET_MISMATCH,
            f"'{entry.id}' is a registered {kind}, but this request sounds like a "
            f"different action. Please rephrase and mention the {kind} explicitly.",
            details={"suggestion_entry": entry.id, "suggestion_section": entry.section},
        )

    suggestion = _suggest_alias(text, registry, section)
    message = (
        f"No approved {section[:-1]} matches this request. "
        "Only targets from the registry can be used."
    )
    if suggestion:
        message += f" Did you mean '{suggestion}'?"
    raise TargetError(
        TARGET_NOT_FOUND,
        message,
        details={"text_snippet": text[:80], "suggestion": suggestion},
    )


def _resolve_find_file(text: str, registry: Registry) -> ActionPlan:
    term = extract_search_term(text)
    if not term:
        raise TargetError(
            INVALID_PARAMETER,
            "No search term found. Example: 'finde die datei rechnung'.",
        )
    folders = registry.searchable_folders()
    if not folders:
        raise TargetError(
            INVALID_PARAMETER,
            "No searchable folders are configured in the registry.",
        )
    return ActionPlan(
        intent="find_file",
        search_term=term,
        search_roots=tuple(Path(folder.path) for folder in folders),
    )


def _resolve_calc(text: str) -> ActionPlan:
    expression = extract_math_expression(text)
    if expression is None:
        raise TargetError(
            INVALID_PARAMETER,
            "No arithmetic expression found. Example: 'was ist 12*4'.",
        )
    return ActionPlan(intent="calc", expression=expression)


def _resolve_timer(text: str) -> ActionPlan:
    seconds = parse_duration(text)
    if seconds is None:
        raise TargetError(
            INVALID_PARAMETER,
            "No duration found. Example: 'stelle einen timer auf 10 minuten'.",
        )
    return ActionPlan(intent="timer", duration_seconds=seconds)


def resolve_target(intent: str, text: str, registry: Registry) -> ActionPlan:
    """Resolve the logical target for ``intent`` from ``text``."""
    if intent in TARGETED_INTENTS:
        return _resolve_registry_target(intent, text, registry)
    if intent == "find_file":
        return _resolve_find_file(text, registry)
    if intent == "calc":
        return _resolve_calc(text)
    if intent == "timer":
        return _resolve_timer(text)
    # System information intents need no target.
    return ActionPlan(intent=intent)
