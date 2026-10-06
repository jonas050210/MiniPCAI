"""Tests for text normalization and the German parameter parsers."""

from __future__ import annotations

import pytest

from minipcai.textutils import (
    damerau_levenshtein,
    extract_math_expression,
    extract_search_term,
    normalize,
    normalize_for_model,
    parse_duration,
)


class TestNormalize:
    def test_lowercase_and_punctuation(self):
        assert normalize("Öffne den Download-Ordner, bitte!") == "öffne den download ordner bitte"

    def test_whitespace_collapse(self):
        assert normalize("  cpu   auslastung  ") == "cpu auslastung"

    def test_umlauts_and_sz_kept(self):
        assert normalize("schließe die Tür") == "schließe die tür"


class TestNormalizeForModel:
    def test_keeps_arithmetic_operators(self):
        assert normalize_for_model("was ist 12*4?") == "was ist 12*4"

    def test_strips_other_punctuation(self):
        assert normalize_for_model("Öffne Notepad, bitte!") == "öffne notepad bitte"

    def test_collapses_whitespace(self):
        assert normalize_for_model("  cpu   auslastung  ") == "cpu auslastung"

    def test_matches_normalize_without_math(self):
        text = "Schließe die Tür bitte!"
        assert normalize_for_model(text) == normalize(text)


class TestDamerauLevenshtein:
    def test_identity_and_empty(self):
        assert damerau_levenshtein("abc", "abc") == 0
        assert damerau_levenshtein("", "abc") == 3
        assert damerau_levenshtein("abc", "") == 3

    def test_transposition_costs_one(self):
        # A plain Levenshtein implementation would return 2 here.
        assert damerau_levenshtein("ca", "ac") == 1
        assert damerau_levenshtein("notepda", "notepad") == 1
        assert damerau_levenshtein("donloads", "downloads") == 1

    def test_classic_distance(self):
        assert damerau_levenshtein("kitten", "sitting") == 3

    def test_symmetric(self):
        for a, b in (("notepad", "notepda"), ("rechnung", "rechnug"), ("abc", "abd")):
            assert damerau_levenshtein(a, b) == damerau_levenshtein(b, a)


class TestMathExtraction:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("was ist 12*4", "12*4"),
            ("berechne 12*4", "12*4"),
            ("rechne 3+4", "3+4"),
            ("wie viel ist 15 geteilt durch 3", "15 / 3"),
            ("was ergibt 2 hoch 10", "2 ** 10"),
            ("rechne 17,5 plus 2,5", "17.5 + 2.5"),
            ("berechne (2+3)*7", "(2+3)*7"),
            ("was ist 7 mal 8 minus 2", "7 * 8 - 2"),
            ("rechne  12   plus   4", "12 + 4"),
            ("was ist 12 * 4", "12 * 4"),
            ("berechne 2 hoch 10 bitte", "2 ** 10"),
        ],
    )
    def test_extracts_expression(self, text, expected):
        assert extract_math_expression(text) == expected

    @pytest.mark.parametrize(
        "text",
        [
            "hallo welt",
            "wie geht es dir",
            "öffne notepad",
            "danke",
        ],
    )
    def test_no_expression(self, text):
        assert extract_math_expression(text) is None


class TestDurationParsing:
    @pytest.mark.parametrize(
        ("text", "seconds"),
        [
            ("timer auf 5 minuten", 300),
            ("stelle einen timer auf 30 sekunden", 30),
            ("timer 10 minuten", 600),
            ("setze einen timer auf 2 stunden", 7200),
            ("timer auf 90 sekunden", 90),
            ("timer auf 2 minuten und 30 sekunden", 150),
            ("timer auf 10 minuten und 20 sekunden", 620),
            ("erinner mich in 1 stunde", 3600),
            ("timer auf eine halbe stunde", 1800),
            ("timer auf eine viertelstunde", 900),
            ("erinner mich in fünf minuten", 300),
            ("timer auf zwei stunden", 7200),
            ("timer 45 sekunden", 45),
            ("timer auf 1 minute", 60),
            ("wecker auf 10 minuten", 600),
            ("timer auf 1,5 stunden", 5400),
        ],
    )
    def test_parses_duration(self, text, seconds):
        assert parse_duration(text) == seconds

    @pytest.mark.parametrize(
        "text",
        [
            "öffne notepad",
            "was ist 12*4",
            "hallo",
            "timer",  # no duration at all
            "stelle einen timer auf eine pizza",
        ],
    )
    def test_no_duration(self, text):
        assert parse_duration(text) is None


class TestSearchTermExtraction:
    @pytest.mark.parametrize(
        ("text", "term"),
        [
            ("finde die datei rechnung", "rechnung"),
            ("suche rechnung", "rechnung"),
            ("wo ist die datei rechnung pdf", "rechnung pdf"),
            ("finde die pdf rechnung", "pdf rechnung"),
            ("suche bitte nach dem protokoll", "protokoll"),
            ("wo habe ich die rechnung gespeichert", "rechnung"),
        ],
    )
    def test_extracts_term(self, text, term):
        assert extract_search_term(text) == term

    def test_stop_words_only_yields_empty_term(self):
        assert extract_search_term("finde die datei") == ""

    def test_path_characters_are_structurally_impossible(self):
        # normalize() strips separators and dots, so traversal cannot survive.
        assert extract_search_term("finde ../../etc/passwd") == "etc passwd"
        term = extract_search_term("suche C:\\Windows\\system32")
        assert "/" not in term and "\\" not in term
