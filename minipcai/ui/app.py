"""PySide6 desktop application for MiniPCAI.

The window is a thin, non-blocking shell around the same audited pipeline the
CLI uses: requests run in a worker thread through :class:`AssistantService`,
results are marshalled back into the GUI thread with Qt signals, and every reply
carries a small *plan card* (intent, target, parameters, executor, dry-run
badge). The sidebar hosts the timer panel, the registry editor and the audit
viewer; a settings dialog covers executor, language, theme, privacy and the
global hotkey. All user-visible text comes from :mod:`minipcai.i18n`, so the
whole UI is German by default and English on request.
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QColor,
    QKeySequence,
    QPalette,
    QShortcut,
    QTextCursor,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDockWidget,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from minipcai import __version__, i18n
from minipcai.actions import DryRunExecutor, WindowsExecutor
from minipcai.audit import AuditError, read_audit_events, verify_chain
from minipcai.model import SklearnIntentClassifier
from minipcai.pipeline import Assistant, AssistantResult
from minipcai.registry import Registry, RegistryError, registry_summary
from minipcai.security import SecurityValidator
from minipcai.service import AssistantService, build_service
from minipcai.settings import Settings, save_settings

_STATUS_COLORS = {
    "ok": "#1b5e20",
    "rejected": "#e65100",
    "error": "#b71c1c",
    "confirmation_required": "#4a148c",
    "clarification_required": "#4527a0",
}

_LIGHT_STYLE = """
QTextBrowser { background: #ffffff; border: 1px solid #d0d0d0; padding: 6px; }
QLineEdit { padding: 6px; }
QPushButton { padding: 6px 14px; }
QTableWidget { gridline-color: #e0e0e0; }
"""

_DARK_STYLE = """
QWidget { background: #1e1e1e; color: #e8e8e8; }
QTextBrowser { background: #141414; border: 1px solid #3a3a3a; padding: 6px; color: #f0f0f0; }
QLineEdit, QComboBox, QTableWidget {
    background: #262626; color: #f0f0f0; border: 1px solid #3a3a3a;
}
QPushButton { background: #2f2f2f; color: #f0f0f0; padding: 6px 14px; border: 1px solid #454545; }
QPushButton:hover { background: #3a3a3a; }
QHeaderView::section { background: #2a2a2a; color: #e8e8e8; border: 1px solid #3a3a3a; }
QTabBar::tab { background: #262626; color: #e8e8e8; padding: 6px; }
QTabBar::tab:selected { background: #3a3a3a; }
QMenuBar, QMenu, QStatusBar { background: #202020; color: #e8e8e8; }
"""

_THEMES = ("system", "light", "dark")


class _ResultBridge(QObject):
    """Marshals worker-thread results into the GUI thread."""

    finished = Signal(object, str)  # AssistantResult, request text


class MainWindow(QMainWindow):
    """The MiniPCAI window."""

    def __init__(
        self,
        assistant: Assistant,
        registry: Registry,
        model: SklearnIntentClassifier,
        settings: Settings | None = None,
        service: AssistantService | None = None,
    ) -> None:
        super().__init__()
        self._assistant = assistant
        self._registry = registry
        self._model = model
        self._settings = settings or Settings(language=assistant.language)
        self._settings.language = assistant.language
        self._service = service or AssistantService(assistant, language=assistant.language)
        self._language = i18n.normalize_language(assistant.language)
        self._pending_timers: list[QTimer] = []
        self._cancel_event = None
        self.confirm_dialogs_enabled = True
        self._sidebar_enabled = True
        self._hotkey_registered = False

        self._bridge = _ResultBridge(self)
        self._bridge.finished.connect(self._on_result)

        self.setWindowTitle(f"MiniPCAI {__version__}")
        self.resize(1060, 680)
        self._build_menu()
        self._build_central()
        self._build_sidebar()
        self._build_status_bar()
        self._apply_theme(self._settings.theme)
        self._install_tray()
        self._install_hotkey()
        self._append_system_message(
            self._t(
                "ui.welcome",
                registry=registry_summary(registry),
                dataset=model.metadata.get("dataset", {}).get("version", "?"),
            )
        )
        self._check_audit_log()
        self._timer_tick = QTimer(self)
        self._timer_tick.setInterval(1000)
        self._timer_tick.timeout.connect(self._refresh_timers)
        self._timer_tick.start()
        self.input_edit.setFocus()

    # -- localization -------------------------------------------------------------
    def _t(self, key: str, **fields) -> str:
        return i18n.t(key, self._language, **fields)

    # -- construction -------------------------------------------------------------
    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu(self._t("ui.menu.file"))
        quit_action = QAction(self._t("ui.menu.quit"), self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        view_menu = self.menuBar().addMenu(self._t("ui.menu.view"))
        theme_menu = view_menu.addMenu(self._t("ui.menu.theme"))
        theme_group = QActionGroup(self)
        self._theme_actions: dict[str, QAction] = {}
        for theme in _THEMES:
            action = QAction(self._t(f"ui.menu.theme_{theme}"), self, checkable=True)
            action.setChecked(theme == self._settings.theme)
            action.triggered.connect(lambda _checked=False, value=theme: self._set_theme(value))
            theme_group.addAction(action)
            theme_menu.addAction(action)
            self._theme_actions[theme] = action
        view_menu.addSeparator()
        self._shown_panels: dict[str, QAction] = {}
        for key, title_key in (
            ("timer", "ui.menu.timer_panel"),
            ("registry", "ui.menu.registry_editor"),
            ("audit", "ui.menu.audit_viewer"),
        ):
            action = QAction(self._t(title_key), self, checkable=True)
            action.setChecked(True)
            action.triggered.connect(
                lambda checked, name=key: self._toggle_panel(name, checked)
            )
            view_menu.addAction(action)
            self._shown_panels[key] = action

        executor_menu = self.menuBar().addMenu(self._t("ui.menu.executor"))
        group = QActionGroup(self)
        self._dry_run_action = QAction(self._t("ui.executor.dry_run"), self, checkable=True)
        self._windows_action = QAction(self._t("ui.executor.windows"), self, checkable=True)
        group.addAction(self._dry_run_action)
        group.addAction(self._windows_action)
        self._dry_run_action.setChecked(isinstance(self._assistant.executor, DryRunExecutor))
        self._windows_action.setChecked(isinstance(self._assistant.executor, WindowsExecutor))
        self._dry_run_action.triggered.connect(lambda: self._switch_executor("dry-run"))
        self._windows_action.triggered.connect(lambda: self._switch_executor("windows"))
        executor_menu.addAction(self._dry_run_action)
        executor_menu.addAction(self._windows_action)

        chat_menu = self.menuBar().addMenu(self._t("ui.menu.chat"))
        clear_action = QAction(self._t("ui.menu.clear_chat"), self)
        clear_action.setShortcut("Ctrl+L")
        clear_action.triggered.connect(self._clear_chat)
        chat_menu.addAction(clear_action)
        settings_action = QAction(self._t("ui.menu.settings"), self)
        settings_action.setShortcut("Ctrl+,")
        settings_action.triggered.connect(self._open_settings)
        chat_menu.addAction(settings_action)

        help_menu = self.menuBar().addMenu(self._t("ui.menu.help"))
        about = QAction(self._t("ui.menu.about"), self)
        about.triggered.connect(self._show_about)
        help_menu.addAction(about)

    def _build_central(self) -> None:
        self.chat_view = QTextBrowser()
        self.chat_view.setOpenExternalLinks(False)

        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText(self._t("ui.placeholder"))
        self.input_edit.returnPressed.connect(self._send)

        self.send_button = QPushButton(self._t("ui.send"))
        self.send_button.clicked.connect(self._send)
        self.stop_button = QPushButton(self._t("ui.stop"))
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._cancel_current)

        self.completer_model = _CompleterModel()
        self.completer = _build_completer(self.input_edit, self.completer_model)
        self._refresh_completions()

        layout = QVBoxLayout()
        layout.addWidget(self.chat_view, 1)
        row = QHBoxLayout()
        row.addWidget(self.input_edit, 1)
        row.addWidget(self.send_button)
        row.addWidget(self.stop_button)
        layout.addLayout(row)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def _build_sidebar(self) -> None:
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_timer_panel(), self._t("ui.timer.title"))
        self.tabs.addTab(self._build_registry_panel(), self._t("ui.registry.title"))
        self.tabs.addTab(self._build_audit_panel(), self._t("ui.audit.title"))

        self.dock = QDockWidget(self._t("ui.menu.panels"), self)
        self.dock.setObjectName("sidebar")
        self.dock.setWidget(self.tabs)
        self.dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock)

    def _build_timer_panel(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        self.timer_table = QTableWidget(0, 3)
        self.timer_table.setHorizontalHeaderLabels(
            [self._t("ui.timer.id"), self._t("ui.timer.remaining"), self._t("ui.timer.duration")]
        )
        self.timer_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.timer_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        layout.addWidget(self.timer_table)
        self.timer_hint = QLabel(self._t("ui.timer.empty"))
        layout.addWidget(self.timer_hint)
        row = QHBoxLayout()
        cancel_selected = QPushButton(self._t("ui.timer.cancel_selected"))
        cancel_selected.clicked.connect(self._cancel_selected_timer)
        cancel_all = QPushButton(self._t("ui.timer.cancel_all"))
        cancel_all.clicked.connect(self._cancel_all_timers)
        row.addWidget(cancel_selected)
        row.addWidget(cancel_all)
        layout.addLayout(row)
        return page

    def _build_registry_panel(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        self.registry_hint = QLabel(
            self._t("ui.registry.path", path=self._registry_path())
        )
        self.registry_hint.setWordWrap(True)
        layout.addWidget(self.registry_hint)
        self.registry_table = QTableWidget(0, 4)
        self.registry_table.setHorizontalHeaderLabels(
            ["id", "section", "target", "aliases"]
        )
        self.registry_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.registry_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        layout.addWidget(self.registry_table)
        row = QHBoxLayout()
        reload_button = QPushButton(self._t("ui.registry.reload"))
        reload_button.clicked.connect(self._reload_registry_table)
        add_button = QPushButton(self._t("ui.registry.add"))
        add_button.clicked.connect(self._add_registry_entry)
        remove_button = QPushButton(self._t("ui.registry.remove"))
        remove_button.clicked.connect(self._remove_registry_entry)
        row.addWidget(reload_button)
        row.addWidget(add_button)
        row.addWidget(remove_button)
        layout.addLayout(row)
        self._reload_registry_table()
        return page

    def _build_audit_panel(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        self.audit_hint = QLabel(str(self._assistant.audit.path))
        self.audit_hint.setWordWrap(True)
        layout.addWidget(self.audit_hint)
        self.audit_table = QTableWidget(0, 3)
        self.audit_table.setHorizontalHeaderLabels(["time", "event", "details"])
        self.audit_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        layout.addWidget(self.audit_table)
        row = QHBoxLayout()
        reload_button = QPushButton(self._t("ui.audit.reload"))
        reload_button.clicked.connect(self._refresh_audit_table)
        verify_button = QPushButton(self._t("ui.audit.verify"))
        verify_button.clicked.connect(self._verify_audit_chain)
        row.addWidget(reload_button)
        row.addWidget(verify_button)
        layout.addLayout(row)
        self._refresh_audit_table()
        return page

    def _build_status_bar(self) -> None:
        status = QStatusBar()
        self.setStatusBar(status)
        self._refresh_status_bar()

    def _refresh_status_bar(self) -> None:
        executor = self._assistant.executor
        self.statusBar().showMessage(
            self._t(
                "ui.status.bar",
                labels=len(self._model.labels),
                mode=executor.mode,
                registry=registry_summary(self._registry),
                audit=self._assistant.audit.path,
            )
        )

    # -- theme / tray / hotkey ----------------------------------------------------
    def _apply_theme(self, theme: str) -> None:
        theme = theme if theme in _THEMES else "system"
        self._settings.theme = theme
        app = QApplication.instance()
        if theme == "dark":
            if app is not None:
                app.setPalette(_dark_palette())
            self.setStyleSheet(_DARK_STYLE)
            self.chat_view.setStyleSheet(_DARK_STYLE)
        elif theme == "light":
            self.setStyleSheet(_LIGHT_STYLE)
            self.chat_view.setStyleSheet("")
        else:  # system: follow the application palette
            self.setStyleSheet(_LIGHT_STYLE)
            self.chat_view.setStyleSheet("")
            if app is not None:
                app.setPalette(app.style().standardPalette())
        for name, action in self._theme_actions.items():
            action.setChecked(name == theme)

    def _set_theme(self, theme: str) -> None:
        self._apply_theme(theme)
        self._save_settings(silent=True)

    def _install_tray(self) -> None:
        self.tray_icon = None
        try:
            from PySide6.QtWidgets import QSystemTrayIcon

            if not QSystemTrayIcon.isSystemTrayAvailable():
                return
            from PySide6.QtWidgets import QStyle

            icon = self.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon)
            tray = QSystemTrayIcon(icon, self)
            tray.setToolTip(f"MiniPCAI {__version__}")
            menu = QMenu(self)
            show_action = menu.addAction(self._t("ui.menu.panels"))
            show_action.triggered.connect(self._show_window)
            quit_action = menu.addAction(self._t("ui.menu.quit"))
            quit_action.triggered.connect(self.close)
            tray.setContextMenu(menu)
            tray.activated.connect(lambda reason: self._show_window())
            tray.show()
            self.tray_icon = tray
        except Exception:  # pragma: no cover - trays are optional
            self.tray_icon = None

    def _show_window(self) -> None:
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def _install_hotkey(self) -> None:
        """Register the configured global hotkey (Windows) or a window shortcut."""
        sequence = self._settings.hotkey or "Ctrl+Alt+M"
        if sys.platform == "win32":  # pragma: no cover - Windows only
            try:
                import ctypes

                modifiers = 0
                key_part = sequence.split("+")[-1].upper()
                if "CTRL" in sequence.upper():
                    modifiers |= 0x0002
                if "ALT" in sequence.upper():
                    modifiers |= 0x0001
                if "SHIFT" in sequence.upper():
                    modifiers |= 0x0004
                vk = ord(key_part) if len(key_part) == 1 else 0x4D
                if ctypes.windll.user32.RegisterHotKey(None, 1, modifiers, vk):
                    self._hotkey_registered = True
                    self._append_system_message(
                        self._t("ui.hotkey.hint", hotkey=sequence)
                    )
                    return
            except Exception:  # pragma: no cover - best effort
                pass
        shortcut = QShortcut(QKeySequence(sequence), self)
        shortcut.activated.connect(self._show_window)
        if sys.platform != "win32":
            self._append_system_message(
                self._t("ui.hotkey.unavailable", hotkey=sequence)
            )

    # -- chat behavior ------------------------------------------------------------
    def _send(self) -> None:
        text = self.input_edit.text().strip()
        if not text:
            return
        self.input_edit.clear()
        self._append_user_message(text)
        self._remember_request(text)
        self.send_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        handle = self._service.run_async(
            text,
            on_result=lambda result: self._bridge.finished.emit(result, text),
        )
        self._cancel_event = handle.cancel_event

    def _cancel_current(self) -> None:
        if self._cancel_event is not None:
            self._cancel_event.set()
            self._append_system_message(self._t("ui.stop"))

    def _on_result(self, result: AssistantResult, text: str) -> None:
        """Runs in the GUI thread (queued from the worker)."""
        self.send_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self._cancel_event = None
        if result.status == "confirmation_required" and self.confirm_dialogs_enabled:
            self._append_assistant_message(result)
            approved = self._ask_confirmation(result)
            self._append_system_message(
                self._t("ui.confirm.approved") if approved else self._t("ui.confirm.declined")
            )
            # Executing the action may block, so it goes back to the worker.
            self.send_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            handle = self._service.confirm_async(
                approved,
                on_result=lambda follow_up: self._bridge.finished.emit(follow_up, text),
            )
            self._cancel_event = handle.cancel_event
            return
        self._append_assistant_message(result)
        if result.status == "clarification_required" and self.confirm_dialogs_enabled:
            answer = self._ask_clarification(result)
            if answer is not None:
                self._append_user_message(answer)
                handle = self._service.run_async(
                    answer,
                    on_result=lambda follow_up: self._bridge.finished.emit(follow_up, answer),
                )
                self._cancel_event = handle.cancel_event
        if result.status == "ok" and result.intent == "timer":
            self._schedule_timer_notification(result)
        self._refresh_timers()
        self._refresh_audit_table()

    def _confirm_dialog(self, question: str) -> bool:
        """Yes/no dialog (own method so tests can patch the answer)."""
        answer = QMessageBox.question(
            self,
            self._t("ui.confirm.title"),
            question,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _ask_confirmation(self, result: AssistantResult) -> bool:
        """Ask before a destructive action (close an app, web search)."""
        description = result.details.get("description") or result.message
        return self._confirm_dialog(self._t("ui.confirm.question", action=description))

    def _clarification_dialog(self, message: str, options: list[str]) -> str | None:
        """Option picker (own method so tests can patch the answer)."""
        text, ok = QInputDialog.getItem(
            self, self._t("ui.confirm.title"), message, options, 0, False
        )
        return text if ok else None

    def _ask_clarification(self, result: AssistantResult) -> str | None:
        options = [str(option) for option in result.details.get("options") or []]
        if not options:
            return None
        return self._clarification_dialog(result.message, options)

    def _schedule_timer_notification(self, result: AssistantResult) -> None:
        duration = result.details.get("duration_seconds")
        if not isinstance(duration, int) or duration <= 0:
            return

        def notify() -> None:
            QApplication.beep()
            self._append_system_message(self._t("ui.timer.finished", seconds=duration))

        timer = QTimer(self)
        timer.setSingleShot(True)
        timer.timeout.connect(notify)
        timer.start(duration * 1000)
        self._pending_timers.append(timer)

    def _switch_executor(self, mode: str) -> None:
        audit = self._assistant.audit
        max_timers = self._assistant.policy.max_active_timers
        self._assistant.executor = (
            WindowsExecutor(audit=audit, max_active_timers=max_timers)
            if mode == "windows"
            else DryRunExecutor(max_active_timers=max_timers)
        )
        self._settings.executor_mode = mode
        if mode == "windows":
            self._append_system_message(self._t("ui.executor.switched_windows"))
            if sys.platform != "win32":
                self._append_system_message(self._t("ui.executor.non_windows"))
        else:
            self._append_system_message(self._t("ui.executor.switched_dry_run"))
        self._refresh_status_bar()
        self._save_settings(silent=True)

    # -- panels -------------------------------------------------------------------
    def _sidebar_visible(self) -> bool:
        """Whether the sidebar is enabled.

        ``QDockWidget.isVisible()`` reports ``False`` while the whole window is
        hidden, so the state is tracked explicitly instead of guessed.
        """
        return self._sidebar_enabled

    def _toggle_panel(self, name: str, visible: bool) -> None:
        """Show the sidebar (and the requested tab) or hide it entirely."""
        order = ["timer", "registry", "audit"]
        any_checked = any(action.isChecked() for action in self._shown_panels.values())
        self._sidebar_enabled = any_checked
        if not any_checked:
            self.dock.setVisible(False)
            return
        self.dock.setVisible(True)
        if visible and name in order:
            self.tabs.setCurrentIndex(order.index(name))

    def _refresh_timers(self) -> None:
        timers = self._service.active_timers()
        self.timer_table.setRowCount(len(timers))
        for row, timer in enumerate(timers):
            values = (
                str(timer.get("request_id", "")),
                f"{int(timer.get('remaining_seconds', 0))} s",
                f"{int(timer.get('duration_seconds', 0))} s",
            )
            for column, value in enumerate(values):
                self.timer_table.setItem(row, column, QTableWidgetItem(value))
        self.timer_hint.setText(self._t("ui.timer.empty") if not timers else "")

    def _cancel_all_timers(self) -> None:
        cancelled = self._service.cancel_timers()
        self._append_system_message(self._t("ui.timer.cancelled", count=cancelled))
        self._refresh_timers()

    def _cancel_selected_timer(self) -> None:
        row = self.timer_table.currentRow()
        if row < 0:
            return
        item = self.timer_table.item(row, 0)
        request_id = item.text() if item is not None else None
        if request_id:
            self._service.cancel_timers(request_id)
            self._refresh_timers()

    def _registry_path(self) -> Path:
        path = getattr(self._registry, "path", None)
        if path:
            return Path(path)
        return self._settings.resolved_registry_path()

    def _reload_registry_table(self) -> None:
        entries = list(self._registry.entries)
        self.registry_table.setRowCount(len(entries))
        for row, entry in enumerate(entries):
            target = entry.url_template if entry.section == "searchers" else (
                entry.url if entry.section == "websites" else entry.path
            )
            values = (entry.id, entry.section, str(target), ", ".join(entry.aliases))
            for column, value in enumerate(values):
                self.registry_table.setItem(row, column, QTableWidgetItem(value))
        self.registry_hint.setText(self._t("ui.registry.path", path=self._registry_path()))

    def _ask_registry_section(self) -> tuple[str, bool]:
        """Which section an added entry belongs to (own method: patchable)."""
        return QInputDialog.getItem(
            self,
            self._t("ui.registry.title"),
            self._t("ui.registry.section_prompt"),
            ["apps", "files", "folders", "websites", "searchers"],
            0,
            False,
        )

    def _ask_registry_target(self) -> tuple[str, bool]:
        """``ID=target`` for a new entry (own method: patchable)."""
        return QInputDialog.getText(
            self, self._t("ui.registry.title"), self._t("ui.registry.add_prompt")
        )

    def _confirm_removal(self, entry_id: str) -> bool:
        """Ask before deleting a registry entry (own method: patchable)."""
        answer = QMessageBox.question(
            self,
            self._t("ui.registry.title"),
            self._t("ui.registry.confirm_remove", id=entry_id),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _add_registry_entry(self) -> None:
        section, ok = self._ask_registry_section()
        if not ok:
            return
        value, ok = self._ask_registry_target()
        if not ok or "=" not in value:
            return
        entry_id, target = (part.strip() for part in value.split("=", 1))
        if not entry_id or not target:
            return
        field = {
            "apps": "executable",
            "files": "path",
            "folders": "path",
            "websites": "url",
            "searchers": "url_template",
        }[section]
        payload = {entry_id: {field: target}}
        if self._write_registry_update(add={section: payload}):
            self._append_system_message(self._t("ui.registry.saved", path=self._registry_path()))

    def _remove_registry_entry(self) -> None:
        row = self.registry_table.currentRow()
        if row < 0:
            return
        entry_id_item = self.registry_table.item(row, 0)
        if entry_id_item is None:
            return
        entry_id = entry_id_item.text()
        if not self._confirm_removal(entry_id):
            return
        if self._write_registry_update(remove=[entry_id]):
            self._append_system_message(self._t("ui.registry.saved", path=self._registry_path()))

    def _write_registry_update(
        self, add: dict[str, dict] | None = None, remove: list[str] | None = None
    ) -> bool:
        """Edit the registry file with the same validation the runtime uses."""
        path = Path(self._registry_path())
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self._append_system_message(self._t("ui.registry.save_failed", detail=exc))
            return False
        for section, entries in (add or {}).items():
            data.setdefault(section, [])
            for entry_id, fields in entries.items():
                entry = {"id": entry_id, "aliases": [entry_id], **fields}
                data[section].append(entry)
        for entry_id in remove or []:
            for section, entries in list(data.items()):
                if isinstance(entries, list):
                    data[section] = [
                        item
                        for item in entries
                        if not (isinstance(item, dict) and item.get("id") == entry_id)
                    ]
        try:
            registry = Registry.load_from_dict(
                data, path=path, policy=self._assistant.policy
            )
        except RegistryError as exc:
            self._append_system_message(self._t("ui.registry.save_failed", detail=exc))
            return False
        try:
            path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
        except OSError as exc:
            self._append_system_message(self._t("ui.registry.save_failed", detail=exc))
            return False
        self._registry = registry
        self._assistant.registry = registry
        self._assistant._validator = SecurityValidator(  # noqa: SLF001 - keep the plan checks consistent
            registry, self._assistant.policy
        )
        self._reload_registry_table()
        self._refresh_status_bar()
        self._refresh_completions()
        return True

    def _refresh_audit_table(self) -> None:
        try:
            events = read_audit_events(self._assistant.audit.path)
        except Exception as exc:  # noqa: BLE001 - a broken log must not crash the UI
            self._append_system_message(self._t("ui.error", detail=exc))
            events = []
        tail = events[-100:]
        self.audit_table.setRowCount(len(tail))
        for row, event in enumerate(reversed(tail)):
            details = ", ".join(
                f"{key}={value}"
                for key, value in event.items()
                if key not in {"ts", "schema_version", "event", "prev", "hash"}
            )
            values = (str(event.get("ts", ""))[:19], str(event.get("event", "")), details)
            for column, value in enumerate(values):
                self.audit_table.setItem(row, column, QTableWidgetItem(value))
        self.audit_hint.setText(
            f"{self._assistant.audit.path} - "
            + self._t("ui.audit.records", count=len(events))
        )

    def _verify_audit_chain(self) -> None:
        ok, checked, bad_line = verify_chain(self._assistant.audit.path)
        if ok:
            self._append_system_message(self._t("ui.audit.verified", count=checked))
        else:
            self._append_system_message(self._t("ui.audit.broken", line=bad_line))

    # -- settings -----------------------------------------------------------------
    def _open_settings(self) -> None:
        dialog = SettingsDialog(self._settings, self._language, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        updated = dialog.result_settings()
        language_changed = updated.language != self._settings.language
        executor_changed = updated.executor_mode != self._settings.executor_mode
        self._settings = updated
        self._language = i18n.normalize_language(updated.language)
        self._assistant.language = self._language
        self._service.language = self._language
        self._apply_theme(updated.theme)
        if executor_changed:
            self._switch_executor(updated.executor_mode)
        if language_changed:
            self._append_system_message(self._t("ui.settings.saved", path=save_settings(updated)))
        else:
            self._save_settings()
        self._refresh_status_bar()

    def _save_settings(self, silent: bool = False) -> None:
        try:
            path = save_settings(self._settings)
            if not silent:
                self._append_system_message(self._t("ui.settings.saved", path=path))
        except OSError as exc:
            self._append_system_message(self._t("ui.settings.save_failed", detail=exc))

    # -- completions --------------------------------------------------------------
    def _remember_request(self, text: str) -> None:
        self.completer_model.add(text)
        self._refresh_completions()

    def _refresh_completions(self) -> None:
        words = [entry.id for entry in self._registry.entries]
        words += [alias for entry in self._registry.entries for alias in entry.aliases]
        self.completer_model.add_many(words)

    # -- chat rendering -----------------------------------------------------------
    def _clear_chat(self) -> None:
        self.chat_view.clear()
        self._append_system_message(self._t("ui.chat.cleared"))

    def _append(self, html_fragment: str) -> None:
        self.chat_view.append(html_fragment)
        self.chat_view.moveCursor(QTextCursor.MoveOperation.End)

    def _append_user_message(self, text: str) -> None:
        color = "#90caf9" if self._settings.theme == "dark" else "#0d47a1"
        self._append(
            f'<div style="color:{color}"><b>{html.escape(self._t("ui.you"))}:</b> '
            f"{html.escape(text)}</div>"
        )

    def _append_assistant_message(self, result: AssistantResult) -> None:
        color = _STATUS_COLORS.get(result.status, "#333333")
        if self._settings.theme == "dark":
            color = {
                "ok": "#81c784",
                "rejected": "#ffb74d",
                "error": "#e57373",
                "confirmation_required": "#ce93d8",
                "clarification_required": "#b39ddb",
            }.get(result.status, "#e0e0e0")
        body = html.escape(result.message).replace("\n", "<br>")
        self._append(
            f'<div style="color:{color}"><b>[{result.status}]</b> {body}</div>'
            f"{self._plan_card_html(result) if result.status == 'ok' else ''}"
            f"{self._meta_html(result)}"
        )

    def _plan_card_html(self, result: AssistantResult) -> str:
        details = result.details or {}
        target = (
            details.get("target_id")
            or details.get("id")
            or details.get("target")
            or "-"
        )
        parameters = {
            key: value
            for key, value in details.items()
            if key
            in {
                "path",
                "url",
                "query",
                "expression",
                "result",
                "duration_seconds",
                "search_term",
            }
        }
        rows = [f"{self._t('ui.plan.title')}"]
        rows.append(self._t("ui.plan.target", target=target))
        if parameters:
            rows.append(
                self._t(
                    "ui.plan.parameters",
                    parameters=", ".join(f"{key}={value}" for key, value in parameters.items()),
                )
            )
        rows.append(self._t("ui.plan.executor", mode=self._assistant.executor.mode))
        if isinstance(self._assistant.executor, DryRunExecutor):
            rows.append(self._t("ui.plan.dry_run"))
        background = "#f5f5f5" if self._settings.theme != "dark" else "#2a2a2a"
        text_color = "#424242" if self._settings.theme != "dark" else "#cfcfcf"
        body = "<br>".join(html.escape(row) for row in rows)
        return (
            f'<div style="background:{background};color:{text_color};font-size:small;'
            f'margin:4px 0;padding:6px;border-left:3px solid #9e9e9e">{body}</div>'
        )

    def _meta_html(self, result: AssistantResult) -> str:
        meta = []
        if result.intent:
            meta.append(f"intent: {result.intent}")
        if result.confidence is not None:
            meta.append(f"confidence: {result.confidence:.0%}")
        if result.reason:
            meta.append(f"reason: {result.reason}")
        if not meta:
            return ""
        color = "#9e9e9e" if self._settings.theme != "dark" else "#8a8a8a"
        return f'<div style="color:{color};font-size:small">{" | ".join(meta)}</div>'

    def _append_system_message(self, text: str) -> None:
        color = "#616161" if self._settings.theme != "dark" else "#9e9e9e"
        self._append(f'<div style="color:{color}"><i>{html.escape(text)}</i></div>')

    def _check_audit_log(self) -> None:
        try:
            self._assistant.audit.check_writable()
        except AuditError as exc:
            self._append_system_message(self._t("ui.audit.warning", detail=exc))

    def _show_about(self) -> None:
        QMessageBox.information(
            self,
            self._t("ui.menu.about"),
            self._t("ui.about.text", version=__version__),
        )

    # -- lifecycle ----------------------------------------------------------------
    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        try:
            self._service.shutdown()
        finally:
            super().closeEvent(event)


class _CompleterModel:
    """Case-insensitive completion over history and registry aliases."""

    def __init__(self) -> None:
        self._values: list[str] = []
        self._seen: set[str] = set()

    def add(self, value: str) -> None:
        self.add_many([value])

    def add_many(self, values) -> None:
        for value in values:
            value = str(value).strip()
            if value and value.lower() not in self._seen:
                self._seen.add(value.lower())
                self._values.append(value)
        model = getattr(self, "_model", None)
        if model is not None:
            model.setStringList(self._values)

    def attach(self, completer) -> None:
        from PySide6.QtCore import QStringListModel

        self._model = QStringListModel(self._values, completer)
        completer.setModel(self._model)


def _build_completer(line_edit: QLineEdit, model: _CompleterModel):
    from PySide6.QtWidgets import QCompleter

    completer = QCompleter(line_edit)
    completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
    completer.setFilterMode(Qt.MatchFlag.MatchContains)
    completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
    model.attach(completer)
    line_edit.setCompleter(completer)
    return completer


class SettingsDialog(QDialog):
    """Small settings dialog (executor, language, theme, privacy, hotkey)."""

    def __init__(self, settings: Settings, language: str, parent: QWidget | None = None):
        super().__init__(parent)
        self._settings = settings
        self._language = i18n.normalize_language(language)
        self.setWindowTitle(i18n.t("ui.settings.title", self._language))

        self.executor_combo = QComboBox()
        self.executor_combo.addItems(["dry-run", "windows"])
        self.executor_combo.setCurrentText(settings.executor_mode)
        self.language_combo = QComboBox()
        self.language_combo.addItems(list(i18n.LANGUAGES))
        self.language_combo.setCurrentText(self._language)
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(list(_THEMES))
        self.theme_combo.setCurrentText(settings.theme)
        self.privacy_check = QCheckBox(i18n.t("ui.settings.privacy", self._language))
        self.privacy_check.setChecked(not settings.store_text_in_audit)
        self.hotkey_edit = QLineEdit(settings.hotkey)
        self.confirmations_label = QLabel(
            i18n.t(
                "ui.settings.confirmations",
                self._language,
                intents=", ".join(settings.confirm_intents) or "-",
            )
        )
        self.confirmations_label.setWordWrap(True)

        form = QFormLayout()
        form.addRow(i18n.t("ui.settings.executor", self._language), self.executor_combo)
        form.addRow(i18n.t("ui.settings.language", self._language), self.language_combo)
        form.addRow(i18n.t("ui.settings.theme", self._language), self.theme_combo)
        form.addRow("", self.privacy_check)
        form.addRow(self.confirmations_label)
        form.addRow(i18n.t("ui.settings.hotkey", self._language), self.hotkey_edit)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def result_settings(self) -> Settings:
        from dataclasses import replace

        return replace(
            self._settings,
            executor_mode=self.executor_combo.currentText(),
            language=self.language_combo.currentText(),
            theme=self.theme_combo.currentText(),
            store_text_in_audit=not self.privacy_check.isChecked(),
            hotkey=self.hotkey_edit.text().strip() or self._settings.hotkey,
        )


def _dark_palette() -> QPalette:  # pragma: no cover - used only for the dark theme
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(30, 30, 30))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(232, 232, 232))
    palette.setColor(QPalette.ColorRole.Base, QColor(20, 20, 20))
    palette.setColor(QPalette.ColorRole.Text, QColor(240, 240, 240))
    palette.setColor(QPalette.ColorRole.Button, QColor(47, 47, 47))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(240, 240, 240))
    return palette


def run_ui(
    settings: Settings | None = None,
    model_path: Path | str | None = None,
    registry_path: Path | str | None = None,
    audit_path: Path | str | None = None,
    executor_mode: str | None = None,
) -> int:
    """Create the application window and enter the Qt event loop."""
    app = QApplication.instance() or QApplication(sys.argv)
    from dataclasses import replace

    settings = settings or Settings()
    if executor_mode:
        settings = replace(settings, executor_mode=executor_mode)
    try:
        service = build_service(
            settings,
            auto_confirm=None,
            language=settings.language,
            executor_mode=executor_mode or settings.executor_mode,
        )
    except Exception as exc:  # noqa: BLE001 - UI boundary
        QMessageBox.critical(
            None,
            "MiniPCAI - startup failed",
            f"Could not start MiniPCAI:\n\n{exc}\n\n"
            "Hint: run 'minipcai setup' and 'minipcai train' first.",
        )
        return 2
    window = MainWindow(
        assistant=service.assistant,
        registry=service.assistant.registry,
        model=service.assistant.model,  # type: ignore[arg-type]
        settings=settings,
        service=service,
    )
    window.show()
    return int(app.exec())
