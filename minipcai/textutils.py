"""Text utilities: normalization, German math/duration parsing, search terms.

All helpers are pure functions without side effects. They operate on German
user input and produce validated, structured values that downstream components
(security, executors) can rely on.
"""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"[^\w\s]+", re.UNICODE)  # \w includes umlauts and sz in Python 3
_SPACES_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase, strip punctuation and collapse whitespace."""
    text = _WORD_RE.sub(" ", text.lower())
    return _SPACES_RE.sub(" ", text).strip()


# ---------------------------------------------------------------------------
# Calculator input
# ---------------------------------------------------------------------------

# German math words -> symbols. Order matters ("geteilt durch" before "durch").
_MATH_WORD_OPERATIONS: tuple[tuple[str, str], ...] = (
    ("geteilt durch", "/"),
    ("geteilt", "/"),
    ("dividiert durch", "/"),
    ("durch", "/"),
    ("plus", "+"),
    ("minus", "-"),
    ("weniger", "-"),
    ("mal", "*"),
    ("multipliziert mit", "*"),
    ("hoch", "**"),
    ("quadrat", "**2"),
)

# A "math run" is a maximal sequence of digits, operators, dots, parens, spaces.
_MATH_RUN_RE = re.compile(r"[\d\s+\-*/().%]+")
_DECIMAL_COMMA_RE = re.compile(r"(\d),(\d)")


def _replace_math_words(text: str) -> str:
    text = f" {text.lower()} "
    for word, symbol in _MATH_WORD_OPERATIONS:
        text = text.replace(f" {word} ", f" {symbol} ")
    return text


def extract_math_expression(text: str) -> str | None:
    """Extract an arithmetic expression from a German request.

    Returns ``None`` when the text contains no usable expression. The returned
    string still has to pass the strict AST validation in ``calc_engine``.
    """
    prepared = _DECIMAL_COMMA_RE.sub(r"\1.\2", text)
    prepared = _replace_math_words(prepared)
    best: str | None = None
    for run in _MATH_RUN_RE.findall(prepared):
        candidate = run.strip()
        if not candidate or not any(ch.isdigit() for ch in candidate):
            continue
        if best is None or len(candidate) > len(best):
            best = candidate
    if best is None:
        return None
    # Drop stray trailing operators such as "12*" (e.g. from "12* bitte").
    best = best.rstrip("+-*/% ").strip()
    if not best or not any(ch.isdigit() for ch in best):
        return None
    return best


# ---------------------------------------------------------------------------
# Timer input
# ---------------------------------------------------------------------------


def damerau_levenshtein(a: str, b: str) -> int:
    """Edit distance (insert/delete/substitute/swap) between two strings."""
    if a == b:
        return 0
    if not a or not b:
        return max(len(a), len(b))
    previous = list(range(len(b) + 1))
    for index_a, char_a in enumerate(a, start=1):
        current = [index_a]
        for index_b, char_b in enumerate(b, start=1):
            cost = 0 if char_a == char_b else 1
            current.append(
                min(
                    current[index_b - 1] + 1,        # insertion
                    previous[index_b] + 1,           # deletion
                    previous[index_b - 1] + cost,    # substitution
                )
            )
        previous = current
    return previous[-1]


# Long unit words that may contain typos ("miniten", "stundn", ...).
_FUZZY_UNIT_WORDS = ("sekunde", "sekunden", "minute", "minuten", "stunde", "stunden")


def _fuzzy_fix_unit_words(text: str) -> str:
    """Replace words that are within edit distance 1 of a time unit word."""
    fixed = []
    for word in text.split():
        if len(word) >= 4 and word not in _UNIT_SECONDS:
            best: tuple[int, str] | None = None
            for unit in _FUZZY_UNIT_WORDS:
                if abs(len(unit) - len(word)) > 2:
                    continue
                distance = damerau_levenshtein(word, unit)
                if best is None or distance < best[0]:
                    best = (distance, unit)
            if best is not None and best[0] <= 1:
                word = best[1]
        fixed.append(word)
    return " ".join(fixed)

_NUMBER_WORDS: dict[str, int] = {
    "ein": 1, "eine": 1, "einen": 1, "einem": 1, "einer": 1,
    "zwei": 2, "drei": 3, "vier": 4, "fünf": 5, "fuenf": 5,
    "sechs": 6, "sieben": 7, "acht": 8, "neun": 9, "zehn": 10,
    "elf": 11, "zwölf": 12, "zwoelf": 12, "fünfzehn": 15, "fuenfzehn": 15,
    "zwanzig": 20, "dreißig": 30, "dreissig": 30, "vierzig": 40,
    "fünfzig": 50, "fuenfzig": 50, "sechzig": 60,
}

_UNIT_SECONDS: dict[str, int] = {
    "sekunde": 1, "sekunden": 1, "sec": 1, "s": 1,
    "minute": 60, "minuten": 60, "min": 60, "m": 60,
    "stunde": 3600, "stunden": 3600, "std": 3600, "h": 3600,
}

_DURATION_RE = re.compile(
    r"(\d+(?:[.,]\d+)?|" + "|".join(_NUMBER_WORDS) + r")\s*"
    r"(sekunden|sekunde|stunden|stunde|minuten|minute|sec|min|std|s|m|h)\b"
)

# Fractions of hours/minutes expressed in words, normalized to minutes/seconds.
_FRACTION_PREPROCESSING: tuple[tuple[str, str], ...] = (
    ("halbe stunde", "30 minuten"),
    ("halben stunde", "30 minuten"),
    ("halbe minute", "30 sekunden"),
    ("viertelstunde", "15 minuten"),
    ("viertel stunde", "15 minuten"),
    ("einer viertel stunde", "15 minuten"),
)


def parse_duration(text: str) -> int | None:
    """Parse a duration in seconds from a German timer request.

    Supports digits and simple number words, compound durations
    ("2 minuten und 30 sekunden") and fractional expressions such as
    "eine halbe stunde". Returns ``None`` if no duration is found.
    """
    prepared = f" {text.lower()} "
    for word, replacement in _FRACTION_PREPROCESSING:
        prepared = prepared.replace(f" {word} ", f" {replacement} ")
    prepared = _fuzzy_fix_unit_words(prepared)
    total = 0
    for raw_number, raw_unit in _DURATION_RE.findall(prepared):
        number: float
        if raw_number.isdigit():
            number = float(raw_number)
        else:
            try:
                number = float(raw_number.replace(",", "."))
            except ValueError:
                number = _NUMBER_WORDS.get(raw_number, 0)
        total += number * _UNIT_SECONDS[raw_unit]
    if total <= 0:
        return None
    seconds = round(total)
    return seconds if seconds > 0 else None


# ---------------------------------------------------------------------------
# find_file search terms
# ---------------------------------------------------------------------------

# Tokens that carry no search meaning and are stripped from the request.
_SEARCH_STOP_TOKENS = frozenset(
    {
        "finde", "find", "suche", "such", "suchst", "durchsuche",
        "wo", "ist", "sind", "liegt", "liegen", "habe", "hab", "ich",
        "die", "das", "den", "der", "dem", "ein", "eine", "einen", "mein",
        "meine", "mir", "mal", "bitte", "nach", "datei", "dateien",
        "file", "files", "dokument", "dokumente", "gespeichert", "kannst",
        "du", "könntest", "the", "and", "auf",
    }
)


def extract_search_term(text: str) -> str:
    """Extract the file search term from a German "find file" request.

    Stop words are removed; everything the user actually said about the file
    name remains, e.g. "finde die datei rechnung pdf" -> "rechnung pdf".
    """
    tokens = [t for t in normalize(text).split() if t not in _SEARCH_STOP_TOKENS]
    return " ".join(tokens)
