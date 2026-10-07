"""Shared pytest fixtures for the MiniPCAI test suite."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from minipcai.config import DEFAULT_DATASET_PATH
from minipcai.train import train


@pytest.fixture(scope="session")
def trained_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Train the real model once per test session into a temporary directory."""
    models_dir = tmp_path_factory.mktemp("models")
    train(dataset_path=DEFAULT_DATASET_PATH, models_dir=models_dir, quiet=True)
    return models_dir / "model.joblib"


@pytest.fixture()
def permissive_policy():
    """A policy without the trusted-app-root allowlist.

    Used by tests that exercise registry *validation* rules (duplicate ids,
    forbidden file types, ...) with Windows-shaped fixture paths that do not
    exist in a trusted location on the test machine.
    """
    from minipcai.policy import default_policy

    return default_policy().allow_untrusted_app_paths()


@pytest.fixture()
def registry_factory(tmp_path: Path):
    """Build a small, valid registry rooted in a temp directory.

    Returns a factory that writes ``registry.json`` and returns its path.
    Callers may pass section overrides, e.g. ``make(apps=[...])``.
    """

    def make(**overrides) -> Path:
        base = tmp_path / "targets"
        downloads = base / "Downloads"
        documents = base / "Documents"
        apps_dir = base / "apps"
        for folder in (downloads, documents, apps_dir):
            folder.mkdir(parents=True, exist_ok=True)
        notepad = apps_dir / "notepad.exe"
        firefox = apps_dir / "firefox.exe"
        for exe in (notepad, firefox):
            exe.touch(exist_ok=True)
        data = {
            "version": 1,
            "apps": [
                {"id": "notepad", "aliases": ["notepad", "editor", "texteditor"],
                 "executable": str(notepad)},
                {"id": "firefox", "aliases": ["firefox", "browser"],
                 "executable": str(firefox)},
                {"id": "chrome", "aliases": ["chrome"],
                 "executable": str(apps_dir / "chrome.exe")},
            ],
            "files": [
                {"id": "notes", "aliases": ["notizen"],
                 "path": str(documents / "notes.txt")},
                {"id": "invoice", "aliases": ["rechnung"],
                 "path": str(documents / "invoice.pdf")},
            ],
            "folders": [
                {"id": "downloads", "aliases": ["downloads", "download-ordner"],
                 "path": str(downloads), "searchable": True},
                {"id": "documents", "aliases": ["dokumente"],
                 "path": str(documents), "searchable": True},
                {"id": "pictures", "aliases": ["bilder"],
                 "path": str(base / "Pictures"), "searchable": False},
            ],
            "websites": [
                {"id": "wikipedia", "aliases": ["wikipedia", "wiki", "nachschlagewerk"],
                 "url": "https://www.wikipedia.org"},
                {"id": "google", "aliases": ["google", "suchmaschine"],
                 "url": "https://www.google.com"},
            ],
        }
        data.update(overrides)
        path = tmp_path / "registry.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    return make


@pytest.fixture()
def make_assistant(trained_model: Path, registry_factory, tmp_path: Path):
    """Factory building an Assistant wired for tests (dry-run by default).

    ``auto_confirm`` defaults to True here because the confirmation flow has
    dedicated tests (``tests/test_pipeline.py::TestConfirmation``,
    ``tests/test_service.py``); the *shipped* default is False, i.e. closing an
    application and web searches ask first.
    """

    def make(
        executor_mode: str = "dry-run",
        registry_path: Path | None = None,
        audit_path: Path | None = None,
        policy=None,
        auto_confirm: bool = True,
        language: str = "en",
        store_text: bool = True,
        registry_policy=None,
    ):
        from minipcai.actions import DryRunExecutor, WindowsExecutor
        from minipcai.audit import AuditLogger
        from minipcai.config import Thresholds
        from minipcai.model import SklearnIntentClassifier
        from minipcai.pipeline import Assistant
        from minipcai.policy import default_policy
        from minipcai.registry import Registry

        model = SklearnIntentClassifier.load(trained_model)
        registry = Registry.load(
            registry_path or registry_factory(),
            policy=policy or registry_policy,
        )
        effective_policy = policy or registry.policy or default_policy()
        audit = AuditLogger(audit_path or tmp_path / "audit.jsonl", store_text=store_text)
        executor = (
            WindowsExecutor(
                audit=audit, max_active_timers=effective_policy.max_active_timers
            )
            if executor_mode == "windows"
            else DryRunExecutor(max_active_timers=effective_policy.max_active_timers)
        )
        stored = model.metadata.get("thresholds", {})
        thresholds = Thresholds(
            min_confidence=float(stored.get("min_confidence", 0.5)),
            min_margin=float(stored.get("min_margin", 0.15)),
        )
        return Assistant(
            model=model,
            registry=registry,
            executor=executor,
            audit=audit,
            thresholds=thresholds,
            policy=policy,
            language=language,
            auto_confirm=auto_confirm,
        )

    return make


@pytest.fixture(scope="session")
def qapp():
    """A QApplication for UI tests; skips when no Qt platform can start."""
    pytest.importorskip("PySide6")
    import os

    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as exc:  # missing system graphics libraries, e.g. libGL
        pytest.skip(f"PySide6.QtWidgets cannot be imported: {exc}")

    candidates = []
    if os.environ.get("QT_QPA_PLATFORM"):
        candidates.append(os.environ["QT_QPA_PLATFORM"])
    # "minimal" first: it exercises widget logic without any rendering
    # backend, which makes it the most portable choice for logic tests.
    candidates += ["minimal", "offscreen"]
    for platform in candidates:
        try:
            os.environ["QT_QPA_PLATFORM"] = platform
            app = QApplication.instance() or QApplication([])
            _ = app.platformName()  # force platform initialization
            return app
        except Exception:  # noqa: BLE001 - try the next platform
            continue
    pytest.skip("no usable Qt platform plugin available")


# Allow running the suite on Windows and Linux alike.
IS_WINDOWS = sys.platform == "win32"
