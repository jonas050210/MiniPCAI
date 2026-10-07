"""Tests for logical target resolution."""

from __future__ import annotations

import pytest

from minipcai.registry import Registry
from minipcai.targets import (
    INVALID_PARAMETER,
    TARGET_AMBIGUOUS,
    TARGET_MISMATCH,
    TARGET_NOT_FOUND,
    ActionPlan,
    TargetError,
    resolve_target,
)


@pytest.fixture()
def registry(registry_factory) -> Registry:
    return Registry.load(registry_factory())


class TestRegistryTargets:
    def test_open_app_resolves_entry(self, registry):
        plan = resolve_target("open_app", "starte den editor", registry)
        assert isinstance(plan, ActionPlan)
        assert plan.intent == "open_app"
        assert plan.entry.id == "notepad"

    def test_open_folder_resolves_entry(self, registry):
        plan = resolve_target("open_folder", "öffne die downloads", registry)
        assert plan.entry.id == "downloads"

    def test_open_url_resolves_entry(self, registry):
        plan = resolve_target("open_url", "geh auf wikipedia", registry)
        assert plan.entry.id == "wikipedia"

    def test_open_file_resolves_entry(self, registry):
        plan = resolve_target("open_file", "öffne die notizen", registry)
        assert plan.entry.id == "notes"

    def test_close_app_resolves_entry(self, registry):
        plan = resolve_target("close_app", "beende firefox", registry)
        assert plan.entry.id == "firefox"

    def test_ambiguous_target_rejected(self, registry):
        with pytest.raises(TargetError) as excinfo:
            resolve_target("open_app", "öffne firefox und chrome", registry)
        assert excinfo.value.reason == TARGET_AMBIGUOUS
        assert set(excinfo.value.details["candidates"]) == {"firefox", "chrome"}

    def test_target_not_found(self, registry):
        with pytest.raises(TargetError) as excinfo:
            resolve_target("open_app", "starte gimp", registry)
        assert excinfo.value.reason == TARGET_NOT_FOUND

    def test_path_injection_never_resolves(self, registry):
        with pytest.raises(TargetError) as excinfo:
            resolve_target("open_file", "öffne C:\\Users\\hacker\\secrets.txt", registry)
        assert excinfo.value.reason in (TARGET_NOT_FOUND, TARGET_MISMATCH)

    def test_section_mismatch_gives_hint(self, registry):
        with pytest.raises(TargetError) as excinfo:
            resolve_target("close_app", "schließe die downloads", registry)
        assert excinfo.value.reason == TARGET_MISMATCH
        assert excinfo.value.details["suggestion_entry"] == "downloads"


class TestResolutionDeterminism:
    def test_multiple_other_sections_fall_back_to_not_found(self, registry):
        # Text mentions a folder AND a website, but the intent is open_app:
        # with more than one foreign section matching, no guess is made.
        with pytest.raises(TargetError) as excinfo:
            resolve_target("open_app", "öffne die downloads und wikipedia", registry)
        assert excinfo.value.reason == TARGET_NOT_FOUND

    def test_single_foreign_section_gives_a_deterministic_hint(self, registry):
        first = None
        for _ in range(5):  # no set-iteration order may leak into the result
            with pytest.raises(TargetError) as excinfo:
                resolve_target("close_app", "schließe die notizen", registry)
            detail = (excinfo.value.reason, excinfo.value.details.get("suggestion_entry"))
            first = first or detail
            assert detail == first
        assert first[0] == TARGET_MISMATCH
        assert first[1] == "notes"

    def test_longest_alias_wins_within_one_entry(self, registry_factory):
        registry = Registry.load(registry_factory(folders=[
            {"id": "downloads", "aliases": ["downloads", "download ordner"],
             "path": "/tmp/downloads", "searchable": True},
        ]))
        matches = registry.find_matches("öffne den download ordner", "folders")
        assert [match.alias for match in matches] == ["download ordner"]


class TestFindFile:
    def test_resolves_term_and_roots(self, registry):
        plan = resolve_target("find_file", "finde die datei rechnung", registry)
        assert plan.search_term == "rechnung"
        assert {root.name for root in plan.search_roots} == {"Downloads", "Documents"}

    def test_missing_term_rejected(self, registry):
        with pytest.raises(TargetError) as excinfo:
            resolve_target("find_file", "finde die datei", registry)
        assert excinfo.value.reason == INVALID_PARAMETER

    def test_no_searchable_folders_rejected(self, registry_factory):
        registry = Registry.load(registry_factory(folders=[
            {"id": "downloads", "aliases": ["downloads"], "path": "/tmp/downloads"}
        ]))
        with pytest.raises(TargetError) as excinfo:
            resolve_target("find_file", "finde rechnung", registry)
        assert excinfo.value.reason == INVALID_PARAMETER


class TestCalcAndTimer:
    def test_calc_expression_extracted(self, registry):
        plan = resolve_target("calc", "was ist 12*4", registry)
        assert plan.expression == "12*4"

    def test_calc_without_expression_rejected(self, registry):
        with pytest.raises(TargetError) as excinfo:
            resolve_target("calc", "danke", registry)
        assert excinfo.value.reason == INVALID_PARAMETER

    def test_timer_duration_extracted(self, registry):
        plan = resolve_target("timer", "timer auf 5 minuten", registry)
        assert plan.duration_seconds == 300

    def test_timer_without_duration_rejected(self, registry):
        with pytest.raises(TargetError) as excinfo:
            resolve_target("timer", "stelle einen timer", registry)
        assert excinfo.value.reason == INVALID_PARAMETER


class TestSystemIntents:
    @pytest.mark.parametrize("intent", ["sys_cpu", "sys_ram", "sys_disk", "sys_summary"])
    def test_no_target_needed(self, registry, intent):
        plan = resolve_target(intent, "irgendein text", registry)
        assert plan.intent == intent
        assert plan.entry is None


class TestWebSearch:
    """The web-search intent only ever opens *registered* search providers."""

    @pytest.fixture()
    def search_registry(self, registry_factory):
        from minipcai.registry import Registry

        path = registry_factory(searchers=[
            {"id": "google_search", "aliases": ["google", "websuche"],
             "url_template": "https://www.google.com/search?q={query}"},
        ])
        return Registry.load(path)

    def test_searcher_provider_resolves(self, search_registry):
        from minipcai.actions import build_search_url

        plan = resolve_target("web_search", "google nach katzen", search_registry)
        assert plan.intent == "web_search"
        assert plan.entry.id == "google_search"
        assert plan.query == "katzen"
        url = build_search_url(plan.entry.url_template, plan.query)
        assert "{query}" not in url
        assert url.startswith("https://www.google.com/search?q=")

    def test_query_is_url_encoded_when_the_url_is_built(self, search_registry):
        from minipcai.actions import build_search_url

        plan = resolve_target("web_search", "suche nach katzen und hunden", search_registry)
        # filler words are dropped, the content words stay in order
        assert plan.query == "katzen hunden"
        url = build_search_url(plan.entry.url_template, plan.query)
        assert "katzen%20hunden" in url

    def test_without_provider_the_request_is_rejected(self, registry):
        with pytest.raises(TargetError) as excinfo:
            resolve_target("web_search", "suche im internet nach katzen", registry)
        assert excinfo.value.reason == "invalid_parameter"

    def test_search_word_alone_is_not_enough(self, registry_factory):
        from minipcai.registry import Registry

        registry = Registry.load(registry_factory(searchers=[
            {"id": "google_search", "aliases": ["google"],
             "url_template": "https://www.google.com/search?q={query}"},
        ]))
        # A bare provider name without a trigger word is not a search request:
        # the classifier would have to guess which words are the query.
        with pytest.raises(TargetError) as excinfo:
            resolve_target("web_search", "google", registry)
        assert excinfo.value.reason == "invalid_parameter"

    @pytest.mark.parametrize(
        "text",
        [
            "hallo",
            "danke",
            "was ist 12*4",
            "wie ist die speicherauslastung",
        ],
    )
    def test_trigger_words_do_not_turn_anything_into_a_search(self, search_registry, text):
        with pytest.raises(TargetError) as excinfo:
            resolve_target("web_search", text, search_registry)
        assert excinfo.value.reason == "invalid_parameter"

    def test_trigger_plus_generic_words(self, search_registry):
        for text, expected in (
            ("suche nach katzenbildern", "katzenbildern"),
            ("suche im internet nach katzen", "katzen"),
            ("google nach wetter", "wetter"),
            ("recherchiere mal die beste route", "beste route"),
        ):
            plan = resolve_target("web_search", text, search_registry)
            assert plan.query == expected

    def test_searcher_without_placeholder_is_a_registry_error(self, tmp_path):
        import json

        from minipcai.registry import RegistryError

        path = tmp_path / "registry.json"
        path.write_text(
            json.dumps({"version": 1, "searchers": [
                {"id": "bad", "aliases": ["bad"], "url_template": "https://example.org/search"},
            ]}),
            encoding="utf-8",
        )
        with pytest.raises(RegistryError, match="placeholder"):
            Registry.load(path)
