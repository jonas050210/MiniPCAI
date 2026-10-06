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
