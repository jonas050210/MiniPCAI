"""PySide6 desktop UI for MiniPCAI.

A compact chat window: type a German request, get an English answer. The
status bar shows the model and executor state; the menu allows switching
between the safe dry-run executor and the real Windows executor. All request
handling runs through the same audited pipeline as the CLI.
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtGui import QAction, QActionGroup, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStatusBar,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from minipcai import __version__
from minipcai.actions import DryRunExecutor, WindowsExecutor
from minipcai.model import SklearnIntentClassifier
from minipcai.pipeline import Assistant, AssistantResult, build_assistant
from minipcai.registry import Registry, registry_summary

_STYLE = """
QTextBrowser { background: #ffffff; border: 1px solid #d0d0d0; padding: 6px; }
QLineEdit { padding: 6px; }
QPushButton { padding: 6px 14px; }
"""

_STATUS_COLORS = {"ok": "#1b5e20", "rejected": "#e65100", "error": "#b71c1c"}


class MainWindow(QMainWindow):
    """The MiniPCAI chat window."""

    def __init__(self, assistant: Assistant, registry: Registry, model: SklearnIntentClassifier):
        super().__init__()
        self._assistant = assistant
        self._registry = registry
        self._model = model
        self._pending_timers: list[QTimer] = []

        self.setWindowTitle(f"MiniPCAI {__version__}")
        self.resize(720, 520)
        self._build_menu()
        self._build_central()
        self._build_status_bar()
        self.setStyleSheet(_STYLE)
        self._append_system_message(
            "Welcome to MiniPCAI. Ask in German, e.g. 'öffne notepad', "
            "'wie viel ram ist frei' or 'stelle einen timer auf 5 minuten'. "
            f"<br>{registry_summary(registry)} | "
            f"dataset v{model.metadata.get('dataset', {}).get('version', '?')}."
        )
        self.input_edit.setFocus()

    # -- construction -----------------------------------------------------------
    def _build_menu(self) -> None:
        executor_menu = self.menuBar().addMenu("&Executor")
        group = QActionGroup(self)
        self._dry_run_action = QAction("Dry run (no real actions)", self, checkable=True)
        self._windows_action = QAction("Windows (real actions)", self, checkable=True)
        group.addAction(self._dry_run_action)
        group.addAction(self._windows_action)
        self._dry_run_action.setChecked(isinstance(self._assistant.executor, DryRunExecutor))
        self._windows_action.setChecked(isinstance(self._assistant.executor, WindowsExecutor))
        self._dry_run_action.triggered.connect(lambda: self._switch_executor("dry-run"))
        self._windows_action.triggered.connect(lambda: self._switch_executor("windows"))
        executor_menu.addAction(self._dry_run_action)
        executor_menu.addAction(self._windows_action)

        help_menu = self.menuBar().addMenu("&Help")
        about = QAction("&About MiniPCAI", self)
        about.triggered.connect(self._show_about)
        help_menu.addAction(about)

    def _build_central(self) -> None:
        self.chat_view = QTextBrowser()
        self.chat_view.setOpenExternalLinks(False)

        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("Request in German ...")
        self.input_edit.returnPressed.connect(self._send)

        self.send_button = QPushButton("&Send")
        self.send_button.clicked.connect(self._send)

        layout = QVBoxLayout()
        layout.addWidget(self.chat_view, 1)
        row = QVBoxLayout()
        row.addWidget(self.input_edit)
        row.addWidget(self.send_button)
        layout.addLayout(row)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def _build_status_bar(self) -> None:
        status = QStatusBar()
        self.setStatusBar(status)
        self._refresh_status_bar()

    def _refresh_status_bar(self) -> None:
        executor = self._assistant.executor
        n_labels = len(self._model.labels)
        self.statusBar().showMessage(
            f"Model: {n_labels} intents | Executor: {executor.mode} | "
            f"{registry_summary(self._registry)}"
        )

    # -- behavior ------------------------------------------------------------------
    def _switch_executor(self, mode: str) -> None:
        audit = self._assistant.audit
        self._assistant.executor = (
            WindowsExecutor(audit=audit) if mode == "windows" else DryRunExecutor()
        )
        self._append_system_message(
            "Switched to the Windows executor: requests will now perform real actions."
            if mode == "windows"
            else "Switched to the dry-run executor: no real actions will be performed."
        )
        self._refresh_status_bar()

    def _send(self) -> None:
        text = self.input_edit.text().strip()
        if not text:
            return
        self.input_edit.clear()
        self._append_user_message(text)
        result = self._assistant.handle(text)
        self._append_assistant_message(result)
        if result.status == "ok" and result.intent == "timer":
            self._schedule_timer_notification(result)

    def _schedule_timer_notification(self, result: AssistantResult) -> None:
        duration = result.details.get("duration_seconds")
        if not isinstance(duration, int) or duration <= 0:
            return

        def notify() -> None:
            QApplication.beep()
            self._append_system_message(f"Timer finished ({duration} seconds).")

        timer = QTimer(self)
        timer.setSingleShot(True)
        timer.timeout.connect(notify)
        timer.start(duration * 1000)
        self._pending_timers.append(timer)

    # -- chat rendering --------------------------------------------------------------
    def _append(self, html_fragment: str) -> None:
        self.chat_view.append(html_fragment)
        self.chat_view.moveCursor(QTextCursor.MoveOperation.End)

    def _append_user_message(self, text: str) -> None:
        self._append(
            f'<div style="color:#0d47a1"><b>You:</b> {html.escape(text)}</div>'
        )

    def _append_assistant_message(self, result: AssistantResult) -> None:
        color = _STATUS_COLORS.get(result.status, "#333333")
        meta = []
        if result.intent:
            meta.append(f"intent: {result.intent}")
        if result.confidence is not None:
            meta.append(f"confidence: {result.confidence:.0%}")
        if result.reason:
            meta.append(f"reason: {result.reason}")
        meta_html = (
            f'<div style="color:#9e9e9e;font-size:small">{" | ".join(meta)}</div>'
            if meta
            else ""
        )
        body = html.escape(result.message).replace("\n", "<br>")
        self._append(
            f'<div style="color:{color}"><b>MiniPCAI [{result.status}]:</b> {body}</div>'
            f"{meta_html}"
        )

    def _append_system_message(self, text: str) -> None:
        self._append(f'<div style="color:#616161"><i>{text}</i></div>')

    def _show_about(self) -> None:
        QMessageBox.information(
            self,
            "About MiniPCAI",
            f"MiniPCAI {__version__}\n\n"
            "A small, self-trained assistant that understands German requests and "
            "performs safe, registry-based Windows actions.\n\n"
            "Security model: targets come only from a validated registry; "
            "no shell, no arbitrary commands, no invented paths. "
            "Every request is written to an audit log.",
        )


def run_ui(
    model_path: Path | str,
    registry_path: Path | str,
    audit_path: Path | str,
    executor_mode: str = "dry-run",
) -> int:
    """Create the application window and enter the Qt event loop."""
    try:
        assistant = build_assistant(
            model_path=model_path,
            registry_path=registry_path,
            audit_path=audit_path,
            executor_mode=executor_mode,
        )
    except Exception as exc:  # noqa: BLE001 - UI boundary
        app = QApplication.instance() or QApplication(sys.argv)
        QMessageBox.critical(
            None, "MiniPCAI - startup failed",
            f"Could not start MiniPCAI:\n\n{exc}\n\n"
            "Hint: train the model first with 'minipcai-train'.",
        )
        return 2

    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow(
        assistant=assistant,
        registry=assistant.registry,
        model=assistant.model,  # type: ignore[arg-type]
    )
    window.show()
    return int(app.exec())
