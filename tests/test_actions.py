"""Tests for the dry-run and the real (Windows) executor."""

from __future__ import annotations

import sys
import time as time_module
from pathlib import Path

import pytest

import minipcai.actions as actions_module
from minipcai.actions import (
    ActionResult,
    DryRunExecutor,
    ExecutorUnavailable,
    WindowsExecutor,
)
from minipcai.audit import AuditLogger, read_audit_events
from minipcai.registry import Registry
from minipcai.targets import ActionPlan, resolve_target

IS_WINDOWS = sys.platform == "win32"


@pytest.fixture()
def registry(registry_factory) -> Registry:
    return Registry.load(registry_factory())


def _plan(registry: Registry, intent: str, text: str) -> ActionPlan:
    return resolve_target(intent, text, registry)


class TestDryRunExecutor:
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
            ("sys_disk", "festplatten belegung"),
            ("sys_summary", "wie läuft der pc"),
            ("calc", "was ist 12*4"),
            ("timer", "timer auf 5 minuten"),
        ],
    )
    def test_all_intents_produce_results(self, registry, intent, text):
        executor = DryRunExecutor()
        result = executor.execute(_plan(registry, intent, text))
        assert isinstance(result, ActionResult)
        assert result.ok
        assert result.summary

    def test_plans_are_recorded(self, registry):
        executor = DryRunExecutor()
        executor.execute(_plan(registry, "open_app", "öffne notepad"))
        executor.execute(_plan(registry, "calc", "was ist 12*4"))
        assert [plan.intent for plan in executor.plans] == ["open_app", "calc"]

    def test_descriptions_are_dry_run(self, registry):
        executor = DryRunExecutor()
        result = executor.execute(_plan(registry, "open_app", "öffne notepad"))
        assert result.summary.startswith("[dry-run]")

    def test_calc_is_actually_computed(self, registry):
        executor = DryRunExecutor()
        result = executor.execute(_plan(registry, "calc", "was ist 12*4"))
        assert "48" in result.summary
        assert result.details["result"] == "48"

    def test_unknown_intent_raises(self):
        with pytest.raises(ExecutorUnavailable):
            DryRunExecutor().execute(ActionPlan(intent="nonsense"))


class TestWindowsExecutorSystemInfo:
    """System info actions are pure Python and run on every platform."""

    @pytest.mark.parametrize("intent", ["sys_cpu", "sys_ram", "sys_disk", "sys_summary"])
    def test_system_info(self, registry, intent):
        result = WindowsExecutor().execute(ActionPlan(intent=intent))
        assert result.ok
        assert result.summary

    def test_sys_ram_details(self):
        result = WindowsExecutor().execute(ActionPlan(intent="sys_ram"))
        assert result.details["total_bytes"] > 0
        assert 0 <= result.details["percent"] <= 100


class TestWindowsExecutorFindFile:
    def test_finds_files_by_name(self, registry, tmp_path):
        documents = Path(registry.by_id("documents").path)
        documents.joinpath("invoice_2024.pdf").touch()
        documents.joinpath("invoice_2025.txt").touch()
        documents.joinpath("photo.png").touch()
        result = WindowsExecutor().execute(
            _plan(registry, "find_file", "finde die datei invoice")
        )
        assert result.ok
        assert len(result.details["matches"]) == 2

    def test_multiword_terms_require_all_words(self, registry):
        documents = Path(registry.by_id("documents").path)
        documents.joinpath("invoice_2024.pdf").touch()
        result = WindowsExecutor().execute(
            ActionPlan(intent="find_file", search_term="invoice pdf",
                       search_roots=tuple(
                           Path(f.path) for f in registry.searchable_folders()
                       ))
        )
        assert "invoice_2024.pdf" in "\n".join(result.details["matches"])

    def test_result_limit(self, registry):
        downloads = Path(registry.by_id("downloads").path)
        for index in range(30):
            downloads.joinpath(f"invoice_{index:03}.pdf").touch()
        result = WindowsExecutor().execute(
            ActionPlan(intent="find_file", search_term="invoice",
                       search_roots=(downloads,))
        )
        assert len(result.details["matches"]) == 20

    def test_no_matches(self, registry):
        result = WindowsExecutor().execute(
            _plan(registry, "find_file", "finde die datei qqzz")
        )
        assert result.ok
        assert result.details["matches"] == []


class TestWindowsExecutorGatedActions:
    """OS-integration actions are refused outside Windows (and mocked inside)."""

    def test_open_app_requires_windows(self, registry, monkeypatch):
        calls: list[tuple] = []
        monkeypatch.setattr(
            actions_module.subprocess, "Popen",
            lambda argv, shell: calls.append((argv, shell)),
        )
        if IS_WINDOWS:
            result = WindowsExecutor().execute(
                _plan(registry, "open_app", "öffne notepad")
            )
            assert result.ok
            assert calls == [([str(registry.by_id("notepad").path)], False)]
        else:
            with pytest.raises(ExecutorUnavailable, match="requires Windows"):
                WindowsExecutor().execute(_plan(registry, "open_app", "öffne notepad"))
            assert calls == []

    def test_open_file_requires_windows(self, registry, monkeypatch):
        opened: list[str] = []
        monkeypatch.setattr(actions_module.os, "startfile", opened.append, raising=False)
        if IS_WINDOWS:
            result = WindowsExecutor().execute(
                _plan(registry, "open_file", "öffne die notizen")
            )
            assert result.ok
            assert opened == [str(registry.by_id("notes").path)]
        else:
            with pytest.raises(ExecutorUnavailable, match="requires Windows"):
                WindowsExecutor().execute(
                    _plan(registry, "open_file", "öffne die notizen")
                )
            assert opened == []

    def test_open_folder_requires_windows(self, registry):
        if IS_WINDOWS:
            assert WindowsExecutor().execute(
                _plan(registry, "open_folder", "öffne die downloads")
            ).ok
        else:
            with pytest.raises(ExecutorUnavailable):
                WindowsExecutor().execute(
                    _plan(registry, "open_folder", "öffne die downloads")
                )

    def test_open_app_missing_executable(self, registry, monkeypatch):
        monkeypatch.setattr(actions_module, "_require_windows", lambda action: None)
        result = WindowsExecutor().execute(
            ActionPlan(intent="open_app",
                       entry=registry.by_id("chrome"))  # file was never created
        )
        assert not result.ok
        assert "not found" in result.summary

    def test_open_url_uses_webbrowser(self, registry, monkeypatch):
        opened: list[str] = []
        def fake_open(url):
            opened.append(url)
            return True

        monkeypatch.setattr(actions_module.webbrowser, "open", fake_open)
        result = WindowsExecutor().execute(_plan(registry, "open_url", "öffne wikipedia"))
        assert result.ok
        assert opened == ["https://www.wikipedia.org"]


class TestWindowsExecutorCloseApp:
    def test_only_registered_processes_terminated(self, registry, monkeypatch):
        monkeypatch.setattr(actions_module, "_require_windows", lambda action: None)

        firefox_path = Path(registry.by_id("firefox").path)
        terminated: list[int] = []

        class FakeProcess:
            def __init__(self, pid: int, exe: str):
                self.pid = pid
                self.info = {"pid": pid, "exe": exe}

            def terminate(self):
                terminated.append(self.pid)

        def fake_process_iter(attrs=None):
            return [
                FakeProcess(101, str(firefox_path)),
                FakeProcess(102, str(firefox_path)),
                FakeProcess(103, "C:\\Windows\\System32\\some_other.exe"),
                FakeProcess(104, ""),
            ]

        monkeypatch.setattr(actions_module.psutil, "process_iter", fake_process_iter)
        result = WindowsExecutor().execute(
            _plan(registry, "close_app", "beende firefox")
        )
        assert result.ok
        assert terminated == [101, 102]
        assert result.details["terminated_pids"] == [101, 102]

    def test_no_running_process(self, registry, monkeypatch):
        monkeypatch.setattr(actions_module, "_require_windows", lambda action: None)
        monkeypatch.setattr(actions_module.psutil, "process_iter", lambda attrs=None: [])
        result = WindowsExecutor().execute(
            _plan(registry, "close_app", "beende firefox")
        )
        assert result.ok
        assert "No running process" in result.summary


class TestWindowsExecutorTimer:
    def test_timer_starts_and_audits_on_elapsed(self, registry, tmp_path):
        audit = AuditLogger(tmp_path / "audit.jsonl")
        audit.check_writable()  # create the file up front
        executor = WindowsExecutor(audit=audit)
        result = executor.execute(
            ActionPlan(intent="timer", duration_seconds=1, request_id="req-1")
        )
        assert result.ok
        assert result.details["duration_seconds"] == 1
        # Wait for the background timer to fire (plus a safety margin).
        deadline = time_module.monotonic() + 5
        while time_module.monotonic() < deadline:
            events = read_audit_events(tmp_path / "audit.jsonl")
            if any(event["event"] == "timer_elapsed" for event in events):
                break
            time_module.sleep(0.05)
        events = read_audit_events(tmp_path / "audit.jsonl")
        elapsed = [event for event in events if event["event"] == "timer_elapsed"]
        assert elapsed and elapsed[0]["request_id"] == "req-1"
        assert elapsed[0]["duration_seconds"] == 1


class TestExecutorModeAttribute:
    def test_modes(self):
        assert DryRunExecutor().mode == "dry-run"
        assert WindowsExecutor().mode == "windows"
