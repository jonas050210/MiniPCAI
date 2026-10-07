"""Tests for the PySide6 desktop UI (offscreen/minimal platform).

Requests are asynchronous by design (the window runs them through
:class:`AssistantService` in a worker thread), so the helpers below spin the Qt
event loop until the reply has been rendered instead of assuming that
``_send()`` blocks.
"""

from __future__ import annotations

import time

import pytest

pytestmark = pytest.mark.usefixtures("qapp")


def _wait_for(predicate, qapp, timeout: float = 10.0) -> bool:
    """Process events until ``predicate`` is true (or the timeout expires)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    qapp.processEvents()
    return predicate()


def _wait_for_text(window, qapp, needle: str, timeout: float = 10.0) -> str:
    _wait_for(lambda: needle in window.chat_view.toPlainText(), qapp, timeout)
    return window.chat_view.toPlainText()


@pytest.fixture()
def main_window(make_assistant, qapp):
    from minipcai.ui.app import MainWindow

    # auto_confirm stays off so ``close_app`` really asks the window first
    assistant = make_assistant(auto_confirm=False)
    window = MainWindow(
        assistant=assistant,
        registry=assistant.registry,
        model=assistant.model,
    )
    yield window
    window.close()
    window._service.shutdown()


class TestMainWindow:
    def test_window_constructs(self, main_window):
        assert main_window.windowTitle().startswith("MiniPCAI")

    def test_send_executed_request_shows_result(self, main_window, qapp):
        main_window.input_edit.setText("öffne notepad")
        main_window._send()
        text = _wait_for_text(main_window, qapp, "[ok]")
        assert "You: öffne notepad" in text
        assert "notepad" in text

    def test_send_rejected_request_shows_reason(self, main_window, qapp):
        main_window.input_edit.setText("wie wird das wetter morgen")
        main_window._send()
        text = _wait_for_text(main_window, qapp, "[rejected]")
        assert "intent: unknown" in text

    def test_input_cleared_after_send(self, main_window, qapp):
        main_window.input_edit.setText("öffne notepad")
        main_window._send()
        assert main_window.input_edit.text() == ""
        _wait_for_text(main_window, qapp, "[ok]")

    def test_empty_input_ignored(self, main_window, qapp):
        main_window.input_edit.setText("   ")
        main_window._send()
        qapp.processEvents()
        assert "You:" not in main_window.chat_view.toPlainText()

    def test_plan_card_is_rendered(self, main_window, qapp):
        main_window.input_edit.setText("öffne notepad")
        main_window._send()
        text = _wait_for_text(main_window, qapp, "[ok]")
        assert "Plan" in text
        assert "dry-run: nothing will be executed" in text
        assert "executor: dry-run" in text

    def test_timer_schedules_notification(self, main_window, qapp):
        main_window.input_edit.setText("stelle einen timer auf 1 sekunde")
        main_window._send()
        _wait_for_text(main_window, qapp, "[ok]")
        assert len(main_window._pending_timers) == 1

    def test_html_is_escaped(self, main_window, qapp):
        main_window.input_edit.setText("<b>öffne notepad</b>")
        main_window._send()
        _wait_for(lambda: "notepad" in main_window.chat_view.toPlainText(), qapp)
        raw = main_window.chat_view.toHtml()
        assert "<b>öffne notepad</b>" not in raw  # must be escaped

    def test_executor_switching(self, main_window, qapp):
        assert main_window._assistant.executor.mode == "dry-run"
        main_window._windows_action.trigger()
        assert main_window._assistant.executor.mode == "windows"
        assert "Windows executor" in main_window.chat_view.toPlainText()
        main_window._dry_run_action.trigger()
        assert main_window._assistant.executor.mode == "dry-run"

    def test_non_windows_switch_shows_hint(self, main_window, monkeypatch):
        monkeypatch.setattr("minipcai.ui.app.sys.platform", "linux")
        main_window._windows_action.trigger()
        text = main_window.chat_view.toPlainText()
        assert "not a Windows machine" in text

    def test_status_bar_mentions_audit_log(self, main_window):
        message = main_window.statusBar().currentMessage()
        assert "Audit:" in message
        assert str(main_window._assistant.audit.path) in message

    def test_clear_chat(self, main_window, qapp):
        main_window.input_edit.setText("öffne notepad")
        main_window._send()
        _wait_for_text(main_window, qapp, "[ok]")
        assert main_window.chat_view.toPlainText().strip()
        main_window._clear_chat()
        text = main_window.chat_view.toPlainText()
        assert "öffne notepad" not in text
        assert "Chat cleared." in text

    def test_unwritable_audit_log_warns_at_startup(self, make_assistant, qapp, tmp_path):
        from minipcai.ui.app import MainWindow

        blocked = tmp_path / "audit_as_dir.jsonl"
        blocked.mkdir()
        assistant = make_assistant(audit_path=blocked)
        window = MainWindow(
            assistant=assistant, registry=assistant.registry, model=assistant.model
        )
        assert "audit log is not writable" in window.chat_view.toPlainText()
        window.close()
        window._service.shutdown()

    def test_writable_audit_log_does_not_warn(self, main_window):
        assert "audit log is not writable" not in main_window.chat_view.toPlainText()


class TestAutocomplete:
    def test_history_and_aliases_feed_the_completer(self, main_window, qapp):
        main_window.input_edit.setText("öffne notepad")
        main_window._send()
        _wait_for_text(main_window, qapp, "[ok]")
        values = main_window.completer_model._values
        assert "notepad" in values  # registry alias
        assert "öffne notepad" in values  # history

    def test_no_duplicates_in_the_completer(self, main_window):
        main_window._refresh_completions()
        main_window._refresh_completions()
        values = main_window.completer_model._values
        assert len(values) == len({value.lower() for value in values})


class TestConfirmationDialog:
    def test_confirmation_is_requested_and_executed(self, main_window, qapp, monkeypatch):
        asked: list[str] = []

        def fake_confirm(question: str) -> bool:
            asked.append(question)
            return True

        monkeypatch.setattr(main_window, "_confirm_dialog", fake_confirm)
        main_window.input_edit.setText("schließe notepad")
        main_window._send()
        text = _wait_for_text(main_window, qapp, "[confirmation_required]")
        assert asked, "the user must be asked before closing an application"
        # The approved action runs in the worker and reports back asynchronously.
        _wait_for(
            lambda: main_window._assistant.executor.plans
            and main_window._assistant.executor.plans[-1].intent == "close_app",
            qapp,
        )
        assert "confirmed" in text.lower() or "bestätigt" in text.lower()

    def test_declined_confirmation_executes_nothing(self, main_window, qapp, monkeypatch):
        monkeypatch.setattr(main_window, "_confirm_dialog", lambda question: False)
        main_window.input_edit.setText("schließe notepad")
        main_window._send()
        _wait_for_text(main_window, qapp, "[rejected]")
        assert main_window._assistant.executor.plans == []
        assert "declined" in main_window.chat_view.toPlainText().lower()


class TestPanels:
    def test_registry_table_lists_entries(self, main_window):
        rows = main_window.registry_table.rowCount()
        assert rows == len(main_window._registry.entries) > 0
        ids = [
            main_window.registry_table.item(row, 0).text() for row in range(rows)
        ]
        assert "notepad" in ids

    def test_timer_panel_shows_running_timers(self, main_window, qapp):
        assert main_window.timer_table.rowCount() == 0
        main_window.input_edit.setText("stelle einen timer auf 5 minuten")
        main_window._send()
        _wait_for_text(main_window, qapp, "[ok]")
        _wait_for(lambda: main_window.timer_table.rowCount() == 1, qapp)
        assert main_window.timer_table.item(0, 1).text().endswith("s")

    def test_timer_panel_cancel_all(self, main_window, qapp):
        main_window.input_edit.setText("stelle einen timer auf 5 minuten")
        main_window._send()
        _wait_for(lambda: main_window.timer_table.rowCount() == 1, qapp)
        main_window._cancel_all_timers()
        assert main_window.timer_table.rowCount() == 0

    def test_audit_table_shows_recent_records(self, main_window, qapp):
        main_window.input_edit.setText("öffne notepad")
        main_window._send()
        _wait_for_text(main_window, qapp, "[ok]")
        _wait_for(lambda: main_window.audit_table.rowCount() > 0, qapp)
        events = [
            main_window.audit_table.item(row, 1).text()
            for row in range(main_window.audit_table.rowCount())
        ]
        assert "request_accepted" in events

    def test_audit_verify_button(self, main_window, qapp):
        main_window.input_edit.setText("öffne notepad")
        main_window._send()
        _wait_for_text(main_window, qapp, "[ok]")
        main_window._verify_audit_chain()
        assert "hash chain intact" in main_window.chat_view.toPlainText()

    def test_panel_toggle_hides_the_sidebar(self, main_window):
        for name in ("timer", "registry", "audit"):
            main_window._shown_panels[name].setChecked(False)
            main_window._toggle_panel(name, False)
        assert main_window._sidebar_visible() is False
        main_window._shown_panels["registry"].setChecked(True)
        main_window._toggle_panel("registry", True)
        assert main_window._sidebar_visible() is True
        assert main_window.tabs.currentIndex() == 1


class TestRegistryEditor:
    def test_add_and_remove_entry_round_trip(self, main_window, qapp, monkeypatch):
        registry_path = main_window._registry.path
        before = len(main_window._registry.entries)

        monkeypatch.setattr(main_window, "_ask_registry_section", lambda: ("websites", True))
        monkeypatch.setattr(
            main_window, "_ask_registry_target", lambda: ("example=https://example.org", True)
        )
        main_window._add_registry_entry()
        assert len(main_window._registry.entries) == before + 1
        assert main_window.registry_table.rowCount() == before + 1
        assert "example" in (registry_path and registry_path.read_text(encoding="utf-8"))

        # remove it again (the confirmation dialog is patched to "Yes")
        monkeypatch.setattr(main_window, "_confirm_removal", lambda entry_id: True)
        row = [
            index
            for index in range(main_window.registry_table.rowCount())
            if main_window.registry_table.item(index, 0).text() == "example"
        ][0]
        main_window.registry_table.setCurrentCell(row, 0)
        main_window._remove_registry_entry()
        assert len(main_window._registry.entries) == before

    def test_invalid_edit_is_refused(self, main_window, monkeypatch):
        monkeypatch.setattr(
            "minipcai.ui.app.QInputDialog.getItem",
            staticmethod(lambda *args, **kwargs: ("apps", True)),
        )
        monkeypatch.setattr(
            "minipcai.ui.app.QInputDialog.getText",
            staticmethod(
                lambda *args, **kwargs: ("script=C:\\\\tools\\\\evil.bat", True)
            ),
        )
        before = len(main_window._registry.entries)
        main_window._add_registry_entry()
        assert len(main_window._registry.entries) == before
        assert "Registry was not saved" in main_window.chat_view.toPlainText()


class TestSettingsDialog:
    def test_dialog_round_trip(self, main_window, qapp):
        from minipcai.ui.app import SettingsDialog

        dialog = SettingsDialog(main_window._settings, "en")
        dialog.executor_combo.setCurrentText("windows")
        dialog.language_combo.setCurrentText("de")
        dialog.theme_combo.setCurrentText("dark")
        dialog.privacy_check.setChecked(True)
        updated = dialog.result_settings()
        assert updated.executor_mode == "windows"
        assert updated.language == "de"
        assert updated.theme == "dark"
        assert updated.store_text_in_audit is False

    def test_theme_switching_applies_styles(self, main_window):
        main_window._set_theme("dark")
        assert "#1e1e1e" in main_window.styleSheet()
        main_window._set_theme("light")
        assert "#1e1e1e" not in main_window.styleSheet()

    def test_german_ui_when_the_language_is_german(self, make_assistant, qapp):
        from minipcai.ui.app import MainWindow

        assistant = make_assistant(language="de")
        window = MainWindow(
            assistant=assistant, registry=assistant.registry, model=assistant.model
        )
        assert "Willkommen" in window.chat_view.toPlainText()
        assert window.send_button.text() == "Senden"
        assert window.stop_button.text() == "Stopp"
        assert window.menuBar().actions()[0].text() == "&Datei"
        window.close()
        window._service.shutdown()
