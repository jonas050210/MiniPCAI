"""Tests for the bilingual message catalog (English default, German switch)."""

from __future__ import annotations

import pytest

from minipcai import i18n


class TestCatalog:
    def test_both_catalogs_have_identical_keys(self):
        assert set(i18n._EN) == set(i18n._DE)
        assert len(i18n._EN) > 50

    def test_nested_placeholder_keys_exist(self):
        # Keys used dynamically by the pipeline must exist in both languages.
        for key in (
            "target.not_found",
            "target.not_found.suggestion",
            "target.mismatch",
            "target.mismatch.generic",
            "parameter.calc",
            "parameter.timer",
            "parameter.search",
            "parameter.web_query",
            "parameter.no_searcher",
            "parameter.no_searchable_folders",
            "confirmation.required",
            "confirmation.declined",
            "confirmation.not_pending",
            "clarification.prompt",
            "clarification.invalid_choice",
            "action.dry_run.open_app",
            "action.open_app.ok",
            "action.web_search.ok",
            "action.timer.started",
            "timer.elapsed",
        ):
            assert i18n.has_key(key, "en"), key
            assert i18n.has_key(key, "de"), key

    def test_has_key_rejects_unknown_keys(self):
        assert i18n.has_key("nope.not.a.key") is False


class TestNormalizeLanguage:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("de", "de"),
            ("DE", "de"),
            ("de-DE", "de"),
            ("de_AT", "de"),
            ("en", "en"),
            ("en-US", "en"),
            ("fr", "de"),
            ("", "de"),
            (None, "de"),
        ],
    )
    def test_normalization(self, value, expected):
        assert i18n.normalize_language(value) == expected

    def test_supported_languages(self):
        assert set(i18n.LANGUAGES) == {"en", "de"}


class TestRendering:
    def test_t_returns_translated_text(self):
        assert i18n.t("confirmation.yes", "de") == "ja"
        assert i18n.t("confirmation.yes", "en") == "yes"
        assert i18n.t("confirmation.declined", "de") != i18n.t("confirmation.declined", "en")

    def test_t_formats_fields(self):
        assert i18n.t("action.calc", "en", expression="2+2", result="4") == "2+2 = 4"

    def test_t_unknown_key_returns_key(self):
        assert i18n.t("does.not.exist", "de") == "does.not.exist"

    def test_render_uses_fallback_and_fields(self):
        assert i18n.render("", "en", "Fallback {n}", n=1) == "Fallback 1"
        rendered = i18n.render("action.open_app.ok", "de", "fallback", id="notepad")
        assert "notepad" in rendered and rendered != "fallback"

    def test_placeholders_are_always_filled(self):
        # Every placeholder in a translated message must be supplied by the
        # caller; this test renders the catalog with dummy values to prove the
        # two languages use the same placeholder names.
        import re

        for key in i18n._EN:
            names_en = set(re.findall(r"\{(\w+)\}", i18n._EN[key]))
            names_de = set(re.findall(r"\{(\w+)\}", i18n._DE[key]))
            assert names_en == names_de, key
            fields = {name: "X" for name in names_en}
            assert "{" not in i18n.t(key, "de", **fields)
            assert "{" not in i18n.t(key, "en", **fields)

    def test_unknown_language_falls_back_to_the_default(self):
        assert i18n.t("confirmation.yes", "fr") == i18n.t("confirmation.yes", "de")
        assert i18n.t("confirmation.yes", "en") == "yes"
