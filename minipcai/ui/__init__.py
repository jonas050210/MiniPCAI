"""PySide6 user interface package (the optional ``gui`` extra).

The window itself lives in :mod:`minipcai.ui.app`, which imports Qt at module
level. This package therefore imports it *lazily*: a missing PySide6 turns into
:class:`GuiUnavailable` with an actionable hint instead of an ``ImportError``
traceback, and the rest of MiniPCAI (CLI, pipeline, training) keeps working
without Qt installed.
"""

from __future__ import annotations

from typing import Any

__all__ = ["GuiUnavailable", "MainWindow", "run_ui", "gui_available"]

#: Message used everywhere the GUI is missing (German default, English in brackets).
GUI_MISSING_HINT = (
    "The graphical interface needs PySide6, which is an optional extra: "
    'install it with  pip install "minipcai[gui]"  '
    "(Die Oberflaeche braucht PySide6:  pip install \"minipcai[gui]\")."
)


class GuiUnavailable(RuntimeError):
    """Raised when the desktop UI is requested but PySide6 is not installed."""


def gui_available() -> bool:
    """Whether Qt is importable (used by ``minipcai doctor`` and the CLI)."""
    try:
        import PySide6.QtWidgets  # noqa: F401
    except Exception:  # noqa: BLE001 - missing Qt *or* missing system libraries
        return False
    return True


def _app_module():
    """Import the Qt window module, mapping an ImportError to a clear error."""
    try:
        from minipcai.ui import app as app_module
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise GuiUnavailable(f"{GUI_MISSING_HINT} ({exc})") from exc
    return app_module


def run_ui(*args: Any, **kwargs: Any) -> int:
    """Start the desktop UI (see :func:`minipcai.ui.app.run_ui`)."""
    return _app_module().run_ui(*args, **kwargs)


def __getattr__(name: str) -> Any:
    """Expose :class:`MainWindow` without importing Qt at package import time."""
    if name == "MainWindow":
        return _app_module().MainWindow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
