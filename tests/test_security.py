"""Tests for the security validator and a static source-code safety scan."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import minipcai
from minipcai.registry import Registry
from minipcai.security import (
    SecurityError,
    SecurityValidator,
)
from minipcai.targets import ActionPlan, RegistryEntry, resolve_target


@pytest.fixture()
def registry(registry_factory) -> Registry:
    return Registry.load(registry_factory())


@pytest.fixture()
def validator(registry) -> SecurityValidator:
    return SecurityValidator(registry)


def _entry(section: str, entry_id: str, target: str) -> RegistryEntry:
    return RegistryEntry(section=section, id=entry_id, aliases=(entry_id,), target=target)


class TestValidPlans:
    @pytest.mark.parametrize(
        ("intent", "text"),
        [
            ("open_app", "öffne notepad"),
            ("close_app", "schließe notepad"),
            ("open_url", "öffne wikipedia"),
            ("open_file", "öffne die notizen"),
            ("open_folder", "öffne die downloads"),
            ("find_file", "finde die datei rechnung"),
            ("sys_cpu", "cpu auslastung"),
            ("sys_ram", "ram auslastung"),
            ("sys_disk", "festplatte"),
            ("sys_summary", "wie läuft der pc"),
            ("calc", "was ist 12*4"),
            ("timer", "timer auf 5 minuten"),
        ],
    )
    def test_all_intent_plans_pass(self, registry, validator, intent, text):
        plan = resolve_target(intent, text, registry)
        validator.validate_plan(plan)  # must not raise


class TestRegistryProvenance:
    def test_entry_from_foreign_registry_rejected(self, validator):
        foreign = _entry("apps", "evil", "C:\\Windows\\System32\\evil.exe")
        with pytest.raises(SecurityError, match="not part of the validated registry"):
            validator.validate_plan(ActionPlan(intent="open_app", entry=foreign))

    def test_wrong_section_rejected(self, validator):
        website = _entry("websites", "wikipedia", "https://www.wikipedia.org")
        with pytest.raises(SecurityError, match="does not match intent"):
            validator.validate_plan(ActionPlan(intent="open_app", entry=website))

    @pytest.mark.parametrize(
        "executable",
        [
            "C:\\Windows\\System32\\cmd.exe",
            "C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe",
            "C:\\Windows\\System32\\wscript.exe",
        ],
    )
    def test_blocked_executables(self, registry_factory, executable):
        registry = Registry.load(registry_factory(apps=[
            {"id": "shell", "aliases": ["shell"], "executable": executable}
        ]))
        validator = SecurityValidator(registry)
        with pytest.raises(SecurityError, match="blocked"):
            validator.validate_plan(ActionPlan(intent="open_app", entry=registry.by_id("shell")))

    def test_non_exe_executable_rejected(self, registry_factory):
        registry = Registry.load(registry_factory(apps=[
            {"id": "script", "aliases": ["script"], "executable": "C:\\tools\\script.bat"}
        ]))
        validator = SecurityValidator(registry)
        with pytest.raises(SecurityError, match="must be an .exe"):
            validator.validate_plan(ActionPlan(intent="open_app", entry=registry.by_id("script")))

    def test_missing_entry_rejected(self, validator):
        with pytest.raises(SecurityError, match="missing required data"):
            validator.validate_plan(ActionPlan(intent="open_app"))


class TestFieldDiscipline:
    def test_calc_with_entry_rejected(self, validator):
        entry = _entry("apps", "notepad", "C:\\Windows\\System32\\notepad.exe")
        with pytest.raises(SecurityError, match="unexpected data"):
            validator.validate_plan(ActionPlan(intent="calc", entry=entry))

    def test_open_app_with_expression_rejected(self, validator):
        entry = _entry("apps", "notepad", "C:\\Windows\\System32\\notepad.exe")
        with pytest.raises(SecurityError, match="unexpected data"):
            validator.validate_plan(
                ActionPlan(intent="open_app", entry=entry, expression="1+1")
            )

    def test_calc_without_expression_rejected(self, validator):
        with pytest.raises(SecurityError, match="missing required data"):
            validator.validate_plan(ActionPlan(intent="calc"))

    def test_unsupported_intent_rejected(self, validator):
        with pytest.raises(SecurityError, match="not supported"):
            validator.validate_plan(ActionPlan(intent="format_disk"))


class TestCalcLimits:
    @pytest.mark.parametrize(
        "expression",
        [
            "__import__('os')",
            "open('x')",
            "2**2**2**2**2",
            "2 ** 1001",
            "1" + "+1" * 100,
        ],
    )
    def test_unsafe_expressions_rejected(self, validator, expression):
        with pytest.raises(SecurityError):
            validator.validate_plan(ActionPlan(intent="calc", expression=expression))

    def test_too_long_expression_rejected(self, validator):
        with pytest.raises(SecurityError, match="too long"):
            validator.validate_plan(
                ActionPlan(intent="calc", expression="1+1" * 50)
            )

    def test_safe_expression_passes(self, validator):
        validator.validate_plan(ActionPlan(intent="calc", expression="(2+3)*7"))


class TestTimerLimits:
    @pytest.mark.parametrize("seconds", [0, -5, 86401, 10**9])
    def test_out_of_bounds_rejected(self, validator, seconds):
        with pytest.raises(SecurityError, match="between"):
            validator.validate_plan(ActionPlan(intent="timer", duration_seconds=seconds))

    @pytest.mark.parametrize("seconds", [1, 60, 3600, 86400])
    def test_in_bounds_passes(self, validator, seconds):
        validator.validate_plan(ActionPlan(intent="timer", duration_seconds=seconds))

    def test_non_integer_rejected(self, validator):
        with pytest.raises(SecurityError, match="integer"):
            validator.validate_plan(ActionPlan(intent="timer", duration_seconds="300"))


class TestFindFileLimits:
    def test_path_like_term_rejected(self, validator):
        with pytest.raises(SecurityError, match="invalid characters"):
            validator.validate_plan(
                ActionPlan(intent="find_file", search_term="..",
                           search_roots=(Path("/approved"),))
            )

    def test_glob_term_rejected(self, validator):
        with pytest.raises(SecurityError, match="invalid characters"):
            validator.validate_plan(
                ActionPlan(intent="find_file", search_term="*.exe",
                           search_roots=(Path("/approved"),))
            )

    def test_unapproved_root_rejected(self, registry, validator):
        with pytest.raises(SecurityError, match="not an approved"):
            validator.validate_plan(
                ActionPlan(intent="find_file", search_term="rechnung",
                           search_roots=(Path("/not/approved"),))
            )

    def test_empty_term_rejected(self, validator):
        with pytest.raises(SecurityError, match="too vague|invalid"):
            validator.validate_plan(
                ActionPlan(intent="find_file", search_term="   ",
                           search_roots=(Path("/approved"),))
            )


class TestSourceCodeSafetyScan:
    """Statically verify that no unsafe execution primitive is used."""

    FORBIDDEN_PATTERNS = [
        (r"shell\s*=\s*True", "shell=True is forbidden"),
        (r"\bos\.system\s*\(", "os.system() is forbidden"),
        (r"\bos\.popen\s*\(", "os.popen() is forbidden"),
        (r"(?<![\w.])eval\s*\(", "eval() is forbidden"),
        (r"(?<![\w.])exec\s*\(", "exec() is forbidden"),
        (r"\b__import__\s*\(", "__import__ is forbidden"),
        (r"subprocess\.(call|run|check_output|check_call|getoutput|getstatusoutput)\s*\(",
         "only subprocess.Popen is allowed"),
    ]

    def test_no_unsafe_constructs_in_package_source(self):
        root = Path(minipcai.__file__).resolve().parent
        sources = list(root.rglob("*.py"))
        assert len(sources) >= 12
        for source in sources:
            text = source.read_text(encoding="utf-8")
            for pattern, message in self.FORBIDDEN_PATTERNS:
                assert not re.search(pattern, text), (
                    f"{source.relative_to(root)}: {message}"
                )

    def test_popen_only_used_in_windows_executor(self):
        root = Path(minipcai.__file__).resolve().parent
        for source in root.rglob("*.py"):
            if "Popen" in source.read_text(encoding="utf-8"):
                assert source.name == "actions.py"
