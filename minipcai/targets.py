"""Logical target resolution: user text -> a validated :class:`ActionPlan`.

The resolver NEVER constructs Windows paths from user input. Targets are
looked up exclusively via alias matching against the validated registry.
Parameterized intents (``calc``, ``timer``, ``find_file``, ``web_search``) get
their values from strict parsers in :mod:`minipcai.textutils`; those values are
re-checked by :mod:`minipcai.security` before execution - and in the case of a
web search the executor percent-encodes the query into a *fixed* registry URL
template, so no user text can ever change the target host.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from minipcai.intents import TARGETED_INTENTS
from minipcai.registry import Registry, RegistryEntry
from minipcai.textutils import (
    bigram_dice,
    damerau_levenshtein,
    extract_math_expression,
    extract_search_term,
    extract_web_query,
    fold,
    has_web_search_trigger,
    parse_duration,
)

# Human-readable names per registry section, used in messages and details.
_SECTION_KINDS = {
    "apps": "app",
    "files": "file",
    "folders": "folder",
    "websites": "website",
    "searchers": "search provider",
}

# Reason codes for target resolution failures.
TARGET_NOT_FOUND = "target_not_found"
TARGET_AMBIGUOUS = "target_ambiguous"
TARGET_MISMATCH = "target_mismatch"
INVALID_PARAMETER = "invalid_parameter"


class TargetError(Exception):
    """Raised when no safe, unambiguous target can be resolved.

    Carries a reason code, an English fallback message, structured details and
    (where available) an i18n key plus fields so frontends can render the same
    failure in German without re-deriving it.
    """

    def __init__(
        self,
        reason: str,
        message: str,
        details: dict | None = None,
        message_key: str = "",
        message_fields: dict | None = None,
    ):
        super().__init__(message)
        self.reason = reason
        self.message = message
        self.details = details or {}
        self.message_key = message_key
        self.message_fields = message_fields or {}


@dataclass(frozen=True)
class ActionPlan:
    """A fully resolved, security-checkable action description.

    Only fields relevant to the intent are populated; the security validator
    enforces this discipline.
    """

    intent: str
    entry: RegistryEntry | None = None          # open/close app, url, file, folder, web search
    expression: str | None = None               # calc
    duration_seconds: int | None = None         # timer
    search_term: str | None = None              # find_file
    search_roots: tuple[Path, ...] = field(default_factory=tuple)  # find_file
    query: str | None = None                    # web_search
    request_id: str = ""                        # audit correlation

    def target_id(self) -> str | None:
        return self.entry.id if self.entry else None

    def describe(self) -> str:
        """Short, human-readable plan description (used in confirmations)."""
        if self.intent in TARGETED_INTENTS or self.intent == "web_search":
            entry = self.entry
            return f"{self.intent}: {entry.id if entry else '?'}"
        if self.intent == "calc":
            return f"calc: {self.expression}"
        if self.intent == "timer":
            return f"timer: {self.duration_seconds}s"
        if self.intent == "find_file":
            return f"find_file: {self.search_term}"
        return self.intent


#: A hint is only offered when the spelling really resembles the alias.
_MIN_SUGGESTION_SIMILARITY = 0.45


def _suggest_alias(
    text: str, registry: Registry, section: str
) -> tuple[RegistryEntry, str] | None:
    """Find the registry entry whose alias is closest to a word in ``text``.

    Returns ``(entry, matched_alias)`` for hints only: the *entry* is what the
    clarification flow offers (stable id), the alias is what the user saw.
    """
    words = fold(text).split()
    candidates: list[tuple[tuple[float, float, int, str, str], str, RegistryEntry]] = []
    for entry in registry.section(section):
        for alias in entry.aliases:
            folded_alias = fold(alias)
            for word in words:
                if len(word) < 4 or len(folded_alias) < 4:
                    continue
                distance = damerau_levenshtein(word, folded_alias)
                limit = 2 if len(folded_alias) >= 7 else 1
                if distance > limit:
                    continue
                similarity = bigram_dice(word, folded_alias)
                if similarity < _MIN_SUGGESTION_SIMILARITY:
                    continue
                # Ranking: lowest edit distance, then the highest bigram
                # similarity, then the alias closest in length; the ids make
                # the result deterministic instead of registry-order dependent.
                score = (
                    float(distance),
                    -similarity,
                    abs(len(word) - len(folded_alias)),
                    folded_alias,
                    entry.id,
                )
                candidates.append((score, folded_alias, entry))
    if not candidates:
        return None
    best = min(candidates, key=lambda item: item[0])
    return best[2], best[1]


def _resolve_registry_target(intent: str, text: str, registry: Registry) -> ActionPlan:
    section = TARGETED_INTENTS[intent]
    matches = registry.find_matches(text, section)

    if len(matches) > 1:
        names = ", ".join(f"'{match.entry.id}'" for match in matches)
        candidates = [match.entry.id for match in matches]
        raise TargetError(
            TARGET_AMBIGUOUS,
            f"Multiple {section} match this request ({names}). Please name exactly one.",
            details={"candidates": candidates, "options": candidates},
            message_key="target.ambiguous",
            message_fields={"names": names, "kind": _SECTION_KINDS[section]},
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
            message_key="target.mismatch",
            message_fields={"entry": entry.id, "kind": kind},
        )

    hint = _suggest_alias(text, registry, section)
    kind = _SECTION_KINDS[section]
    alias = hint[1] if hint else ""
    message = (
        f"No approved {kind} matches this request. Only targets from the registry "
        "can be used."
    )
    if hint:
        message += f" Did you mean '{alias}'?"
    raise TargetError(
        TARGET_NOT_FOUND,
        message,
        details={
            "text_snippet": text[:80],
            "suggestion": hint[0].id if hint else None,
            "suggestion_alias": alias,
            "kind": kind,
            "section": section,
        },
        message_key="target.not_found" + (".suggestion" if hint else ""),
        message_fields={"suggestion": alias, "kind": kind},
    )


def _resolve_find_file(text: str, registry: Registry) -> ActionPlan:
    term = extract_search_term(text)
    if not term:
        raise TargetError(
            INVALID_PARAMETER,
            "No search term found. Example: 'finde die datei rechnung'.",
            message_key="parameter.search",
        )
    folders = registry.searchable_folders()
    if not folders:
        raise TargetError(
            INVALID_PARAMETER,
            "No searchable folders are configured in the registry.",
            message_key="parameter.no_searchable_folders",
        )
    return ActionPlan(
        intent="find_file",
        search_term=term,
        search_roots=tuple(Path(folder.path) for folder in folders),
    )


def _resolve_web_search(text: str, registry: Registry) -> ActionPlan:
    searchers = registry.searchers()
    if not searchers:
        raise TargetError(
            INVALID_PARAMETER,
            "No search provider is configured in the registry.",
            message_key="parameter.no_searcher",
        )
    matches = registry.find_matches(text, "searchers")
    searcher: RegistryEntry = matches[0].entry if matches else searchers[0]
    strip_words = [alias for alias in searcher.aliases] + [searcher.id]
    query = extract_web_query(text, strip_aliases=tuple(strip_words))
    # A web search must be *asked for*: either a registered provider is named
    # or the request contains an explicit search word. Without this gate the
    # intent would swallow every sentence with a content word.
    if not query or (not matches and not has_web_search_trigger(text)):
        raise TargetError(
            INVALID_PARAMETER,
            "No search query found. Example: 'suche im internet nach katzen'.",
            message_key="parameter.web_query",
        )
    return ActionPlan(intent="web_search", entry=searcher, query=query)


def _resolve_calc(text: str) -> ActionPlan:
    expression = extract_math_expression(text)
    if expression is None:
        raise TargetError(
            INVALID_PARAMETER,
            "No arithmetic expression found. Example: 'was ist 12*4'.",
            message_key="parameter.calc",
        )
    return ActionPlan(intent="calc", expression=expression)


def _resolve_timer(text: str) -> ActionPlan:
    seconds = parse_duration(text)
    if seconds is None:
        raise TargetError(
            INVALID_PARAMETER,
            "No duration found. Example: 'stelle einen timer auf 10 minuten'.",
            message_key="parameter.timer",
        )
    return ActionPlan(intent="timer", duration_seconds=seconds)


def resolve_target(
    intent: str,
    text: str,
    registry: Registry,
    preferred_target: str | None = None,
) -> ActionPlan:
    """Resolve the logical target for ``intent`` from ``text``.

    ``preferred_target`` is an *internal* hint: the id of a registry entry the
    caller (the clarification flow) has already offered the user. It never
    comes from raw user text, is only accepted when the entry exists and the
    entry's kind matches the intent, and short-circuits the alias lookup so a
    disambiguated request cannot fall back into ambiguity.
    """
    if preferred_target and intent in TARGETED_INTENTS:
        entry = registry.by_id(preferred_target)
        if entry is not None and entry.section == TARGETED_INTENTS[intent]:
            return ActionPlan(intent=intent, entry=entry)
    if intent in TARGETED_INTENTS:
        return _resolve_registry_target(intent, text, registry)
    if intent == "find_file":
        return _resolve_find_file(text, registry)
    if intent == "web_search":
        return _resolve_web_search(text, registry)
    if intent == "calc":
        return _resolve_calc(text)
    if intent == "timer":
        return _resolve_timer(text)
    # System information intents need no target.
    return ActionPlan(intent=intent)
