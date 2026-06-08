"""Small Qt worker helpers for running blocking callables off the UI thread."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QThread, Signal


class CallableWorker(QThread):
    """Run a callable in a QThread and emit its return value or exception text."""

    finished_with_result = Signal(object)
    failed = Signal(str)

    def __init__(self, func: Callable[[], Any], parent=None):
        super().__init__(parent)
        self._func = func

    def run(self) -> None:  # pragma: no cover - exercised through GUI integration
        try:
            self.finished_with_result.emit(self._func())
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
