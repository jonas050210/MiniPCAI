"""Text utilities: normalization, German math/duration parsing, search terms.

All helpers are pure functions without side effects. They operate on German
user input and produce validated, structured values that downstream components
(security, executors) can rely on.

Three normalizations exist on purpose:

``normalize``
    Unicode NFC, lowercase, punctuation removed, whitespace collapsed. This is
    the "human readable" form used for search terms and error messages.
``fold``
    Like ``normalize``, but additionally folds umlauts/ß and strips every
    diacritic, so ``"öffne"``, ``"oeffne"`` and NFD-encoded ``"o\u0308ffne"``
    all collapse to the same key. Used for alias matching only, so that a
    registry written in NFC still matches keyboard layouts and mail clients
    that emit NFD.
``normalize_for_model``
    Like ``normalize`` but keeps arithmetic operators, used as the vectorizer
    preprocessor so that ``"12*4"`` survives while punctuation/casing does not.
"""

from __future__ import annotations

import re
import unicodedata

_WORD_RE = re.compile(r"[^\w\s]+", re.UNICODE)  # \w includes umlauts and sz in Python 3
_SPACES_RE = re.compile(r"\s+")

# Characters that survive model-side normalization: arithmetic must stay
# recognizable ("was ist 12*4" must not become "was ist 12 4").
_MODEL_KEEP_RE = re.compile(r"[^\w\s+\-*/%().]+", re.UNICODE)

# Fold table for alias matching: German umlauts/ß plus a few common Latin
# letters that appear in product names ("Ø", "Å").
_FOLD_TABLE = str.maketrans(
    {"ä": "a", "ö": "o", "ü": "u", "ß": "ss", "å": "a", "æ": "ae", "ø": "o", "œ": "oe"}
)


def to_nfc(text: str) -> str:
    """Return the NFC (composed) form of ``text``.

    Windows input arrives in whatever form the keyboard/browser produced:
    an emoji picker, a mail client or a copy/paste from macOS can deliver NFD
    (``o`` + combining diaeresis), which would otherwise be stripped as
    "punctuation" and turn ``"öffne"`` into ``"o ffne"``.
    """
    return unicodedata.normalize("NFC", text)


def normalize(text: str) -> str:
    """Lowercase, strip punctuation and collapse whitespace."""
    text = _WORD_RE.sub(" ", to_nfc(text).lower())
    return _SPACES_RE.sub(" ", text).strip()


def fold(text: str) -> str:
    """Canonical alias key: case-, accent- and punctuation-insensitive."""
    folded = normalize(text).translate(_FOLD_TABLE)
    # Drop remaining diacritics that survived NFC (e.g. "ć"), then collapse
    # any whitespace the decomposition introduced.
    decomposed = unicodedata.normalize("NFD", folded)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return _SPACES_RE.sub(" ", stripped).strip()


def normalize_for_model(text: str) -> str:
    """Like :func:`normalize`, but keeps arithmetic operators.

    Used as the vectorizer preprocessor so that punctuation and casing
    variants of a request map to the same features, while calculator requests
    keep their operators for the character n-grams.
    """
    text = _MODEL_KEEP_RE.sub(" ", to_nfc(text).lower())
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
    text = f" {_SPACES_RE.sub(' ', text.lower())} "
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
    """Damerau-Levenshtein distance (insert/delete/substitute/swap).

    The adjacent-transposition term is what distinguishes this from the plain
    Levenshtein distance: transposed characters ("notepda" -> "notepad") are
    one of the most common typing mistakes and must cost 1, not 2.
    """
    if a == b:
        return 0
    if not a or not b:
        return max(len(a), len(b))
    previous_previous: list[int] = []
    previous = list(range(len(b) + 1))
    for index_a, char_a in enumerate(a, start=1):
        current = [index_a]
        for index_b, char_b in enumerate(b, start=1):
            cost = 0 if char_a == char_b else 1
            value = min(
                current[index_b - 1] + 1,        # insertion
                previous[index_b] + 1,           # deletion
                previous[index_b - 1] + cost,    # substitution
            )
            if (
                index_a > 1
                and index_b > 1
                and char_a == b[index_b - 2]
                and char_b == a[index_a - 2]
            ):
                value = min(value, previous_previous[index_b - 2] + 1)  # transposition
            current.append(value)
        previous_previous = previous
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


# ---------------------------------------------------------------------------
# web_search queries
# ---------------------------------------------------------------------------

# Words that only introduce a web search and carry no search meaning.
_WEB_SEARCH_STOP_TOKENS = frozenset(
    {
        "suche", "such", "suchst", "sucht", "durchsuche", "recherchiere",
        "im", "in", "dem", "der", "den", "das", "die", "ein", "eine", "einen",
        "nach", "web", "internet", "websuche", "online", "netz", "google",
        "bitte", "mal", "mach", "mir", "bei", "auf", "für", "fuer", "und",
        "was", "ist", "sind", "über", "ueber", "zu", "zum", "zur", "von",
        "the", "and", "search", "for", "about",
    }
)


# Words that must be present for a *default* web search. Without one of them a
# request is not a search - "hallo" must not silently become "search the web
# for hallo".
_WEB_SEARCH_TRIGGERS = frozenset(
    {
        "suche", "such", "suchst", "sucht", "durchsuche", "recherchiere",
        "recherche", "googeln", "google", "internet", "web", "websuche",
        "online", "nachschlagen", "schau", "guck",
    }
)


def has_web_search_trigger(text: str) -> bool:
    """True when the text explicitly asks for a web search."""
    tokens = set(normalize(text).split())
    return bool(tokens & _WEB_SEARCH_TRIGGERS)


def extract_web_query(text: str, strip_aliases: tuple[str, ...] = ()) -> str:
    """Extract the search query of a "search the web" request.

    ``strip_aliases`` are registry aliases (e.g. the name of the search
    provider) that are removed from the query. Matching happens on the folded
    text, but the *original* characters are returned, so ``"Bäume"`` stays
    ``"Bäume"`` instead of becoming ``"baume"``.
    """
    tokens = normalize(text).split()
    if not tokens:
        return ""
    folded = [fold(token) for token in tokens]
    drop = [False] * len(tokens)

    for alias in strip_aliases:
        parts = fold(alias).split()
        if not parts:
            continue
        width = len(parts)
        for start in range(len(folded) - width + 1):
            if folded[start : start + width] == parts:
                for index in range(start, start + width):
                    drop[index] = True

    for index, _token in enumerate(tokens):
        if folded[index] in _WEB_SEARCH_STOP_TOKENS:
            drop[index] = True

    return " ".join(token for token, dropped in zip(tokens, drop, strict=True) if not dropped)

