"""The golden hard set: quality gate for realistic phrasing (roadmap P3).

Two suites:

* ``test_golden_hard_set`` - the curated cases in ``data/golden_hard.jsonl``.
* ``test_registry_sweep`` - hundreds of phrasings/typos derived from the
  registry itself, so every alias is exercised in several shapes.

Both gate on two numbers: **zero** wrong accepts (nothing out of scope, no
injection, no wrong target is ever executed) and an unnecessary-reject rate
below :data:`minipcai.golden.MAX_UNNECESSARY_REJECT_RATE`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from minipcai.golden import (
    CLARIFY_OFFER,
    CORRECT,
    CORRECT_AMBIGUOUS,
    MAX_UNNECESSARY_REJECT_RATE,
    UNNECESSARY_REJECT,
    WRONG_ACCEPT,
    WRONG_TARGET,
    evaluate_cases,
    format_report,
    load_cases,
    summarize,
)

GOLDEN_PATH = Path(__file__).parent / "data" / "golden_hard.jsonl"

SEARCHERS = [
    {
        "id": "ddg",
        "aliases": ["duckduckgo", "web", "suchdienst"],
        "url_template": "https://duckduckgo.com/?q={query}",
    }
]


def _golden_assistant(make_assistant, registry_factory):
    """An assistant with the shipped-shaped registry (incl. a search provider)."""
    path = registry_factory(searchers=SEARCHERS)
    return make_assistant(registry_path=path)


class TestGoldenHardSet:
    @pytest.fixture()
    def outcomes(self, make_assistant, registry_factory):
        """Grade the whole curated set (the assistant is cheap to build)."""
        assistant = _golden_assistant(make_assistant, registry_factory)
        return evaluate_cases(assistant, load_cases(GOLDEN_PATH))

    def test_the_set_is_loaded_completely(self, outcomes):
        assert len(outcomes) == 71

    def test_no_request_is_wrongly_executed(self, outcomes):
        fatal = [o for o in outcomes if o.verdict in {WRONG_ACCEPT, WRONG_TARGET}]
        assert not fatal, "\n".join(format_report(summarize(outcomes), outcomes).splitlines()[-6:])

    def test_unnecessary_reject_rate(self, outcomes):
        summary = summarize(outcomes)
        assert summary["unnecessary_reject_rate"] <= MAX_UNNECESSARY_REJECT_RATE, (
            format_report(summary, outcomes)
        )

    def test_every_case_gets_a_known_verdict(self, outcomes):
        known = {CORRECT, CORRECT_AMBIGUOUS, CLARIFY_OFFER, UNNECESSARY_REJECT,
                 WRONG_ACCEPT, WRONG_TARGET}
        assert {o.verdict for o in outcomes} <= known

    def test_typos_produce_a_clarification_offer_that_names_the_right_target(self, outcomes):
        """Fuzzy target matching: a typo must ask "did you mean X?" - never act silently."""
        typo_cases = [o for o in outcomes if o.category == "typo"]
        assert typo_cases
        handled = [o for o in typo_cases if o.verdict in {CORRECT, CLARIFY_OFFER}]
        assert len(handled) == len(typo_cases), format_report(summarize(outcomes), outcomes)

    def test_jsonl_entries_are_well_formed(self):
        for case in load_cases(GOLDEN_PATH):
            assert case["expect"] in {"resolve", "refuse", "ambiguous"}
            assert case["text"].strip()
            if case["expect"] == "resolve":
                assert case.get("intent")


class TestRegistrySweep:
    """Mechanical coverage: every alias in several phrasings and mutations."""

    @staticmethod
    def _mutations(alias: str) -> list[str]:
        variants = [alias]
        folded = (
            alias.replace("ä", "ae").replace("ö", "oe")
            .replace("ü", "ue").replace("ß", "ss")
        )
        if folded != alias:
            variants.append(folded)
        if len(alias) > 3:  # swapped neighbours
            variants.append(alias[:-2] + alias[-1] + alias[-2])
        if len(alias) > 4:  # dropped character
            variants.append(alias[:2] + alias[3:])
        if len(alias) > 4:  # doubled character
            variants.append(alias[:2] + alias[1] + alias[2:])
        return variants

    @classmethod
    def build_cases(cls, registry) -> list[dict]:
        templates = {
            "apps": ["öffne {alias}", "starte {alias}", "mach {alias} auf",
                     "kannst du {alias} öffnen", "ich möchte {alias} starten",
                     "öffne {alias} bitte"],
            "files": ["öffne {alias}", "zeig mir {alias}", "kannst du {alias} öffnen"],
            "folders": ["öffne {alias}", "öffne den ordner {alias}",
                        "zeig mir den ordner {alias}", "wo ist der ordner {alias}"],
            "websites": ["öffne {alias}", "geh auf {alias}", "mach {alias} auf",
                         "ich möchte {alias} öffnen"],
        }
        cases: list[dict] = []
        seen: set[str] = set()
        for section, phrasings in templates.items():
            for entry in registry.section(section):
                for alias in entry.aliases:
                    for phrasing in phrasings:
                        for variant in cls._mutations(alias):
                            phrase = phrasing.format(alias=variant).strip()
                            if not phrase or phrase.casefold() in seen:
                                continue
                            seen.add(phrase.casefold())
                            cases.append({
                                "text": phrase,
                                "expect": "resolve",
                                "target": entry.id,
                                "category": f"sweep_{section}",
                            })
        return cases

    def test_sweep(self, make_assistant, registry_factory):
        assistant = _golden_assistant(make_assistant, registry_factory)
        cases = self.build_cases(assistant.registry)
        assert len(cases) >= 300
        outcomes = evaluate_cases(assistant, cases)
        summary = summarize(outcomes)
        assert summary["wrong_accepts"] == 0, format_report(summary, outcomes)
        assert summary["wrong_targets"] == 0, format_report(summary, outcomes)
        # A sweep full of deliberate typos is allowed to ask back, but it must
        # not simply give up on a third of the phrasings.
        assert summary["unnecessary_reject_rate"] <= 0.20, format_report(summary, outcomes)

    def test_sweep_cases_are_serializable(self, registry_factory):
        from minipcai.registry import Registry

        registry = Registry.load(registry_factory(searchers=SEARCHERS))
        for case in self.build_cases(registry):
            json.dumps(case, ensure_ascii=False)
