"""Tests for text normalization and the German parameter parsers."""

from __future__ import annotations

import pytest

from minipcai.textutils import (
    extract_math_expression,
    extract_search_term,
    normalize,
    parse_duration,
)


class TestNormalize:
    def test_lowercase_and_punctuation(self):
        assert normalize("Öffne den Download-Ordner, bitte!") == "öffne den download ordner bitte"

    def test_whitespace_collapse(self):
        assert normalize("  cpu   auslastung  ") == "cpu auslastung"

    def test_umlauts_and_sz_kept(self):
        assert normalize("schließe die Tür") == "schließe die tür"


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
