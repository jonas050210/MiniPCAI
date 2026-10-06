"""Smoke tests for the PySide6 UI (offscreen/minimal platform)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.usefixtures("qapp")


@pytest.fixture()
def main_window(make_assistant, qapp):
    from minipcai.ui.app import MainWindow

    assistant = make_assistant()
    window = MainWindow(
        assistant=assistant,
        registry=assistant.registry,
        model=assistant.model,
    )
    return window


class TestMainWindow:
    def test_window_constructs(self, main_window):
        assert main_window.windowTitle().startswith("MiniPCAI")

    def test_send_executed_request_shows_result(self, main_window):
        main_window.input_edit.setText("öffne notepad")
        main_window._send()
        text = main_window.chat_view.toPlainText()
        assert "You: öffne notepad" in text
        assert "[ok]" in text
        assert "notepad" in text

    def test_send_rejected_request_shows_reason(self, main_window):
        main_window.input_edit.setText("wie wird das wetter morgen")
        main_window._send()
        text = main_window.chat_view.toPlainText()
        assert "[rejected]" in text
        assert "intent: unknown" in text

    def test_input_cleared_after_send(self, main_window):
        main_window.input_edit.setText("öffne notepad")
        main_window._send()
        assert main_window.input_edit.text() == ""

    def test_empty_input_ignored(self, main_window):
        main_window.input_edit.setText("   ")
        main_window._send()
        assert "You:" not in main_window.chat_view.toPlainText()

    def test_timer_schedules_notification(self, main_window):
        main_window.input_edit.setText("stelle einen timer auf 1 sekunde")
        main_window._send()
        assert len(main_window._pending_timers) == 1

    def test_html_is_escaped(self, main_window):
        main_window.input_edit.setText("<b>öffne notepad</b>")
        main_window._send()
        raw = main_window.chat_view.toHtml()
        assert "<b>öffne notepad</b>" not in raw  # must be escaped

    def test_executor_switching(self, main_window):
        assert main_window._assistant.executor.mode == "dry-run"
        main_window._windows_action.trigger()
        assert main_window._assistant.executor.mode == "windows"
        assert "Windows executor" in main_window.chat_view.toPlainText()
        main_window._dry_run_action.trigger()
        assert main_window._assistant.executor.mode == "dry-run"
