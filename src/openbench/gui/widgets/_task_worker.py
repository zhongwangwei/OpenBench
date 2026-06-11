"""Small Qt worker helpers for running blocking callables off the UI thread."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QThread, Signal


class CallableWorker(QThread):
    """Run a callable in a QThread and emit its return value or exception."""

    finished_with_result = Signal(object)
    failed = Signal(str)
    # The raw exception, for callers (e.g. call_responsive) that re-raise it.
    failed_with_exception = Signal(object)

    def __init__(self, func: Callable[[], Any], parent=None):
        super().__init__(parent)
        self._func = func

    def run(self) -> None:  # pragma: no cover - exercised through GUI integration
        try:
            self.finished_with_result.emit(self._func())
        except Exception as exc:
            self.failed_with_exception.emit(exc)
            self.failed.emit(f"{type(exc).__name__}: {exc}")


def detach_worker(worker, registry: list) -> None:
    """Keep an unparented running QThread alive until Qt emits finished.

    Appends the worker to ``registry`` (a module-level list) and removes it
    when the thread finishes. Without that reference, Python could garbage
    collect a running QThread, which aborts the whole process.
    """
    if worker is None:
        return
    registry.append(worker)

    def _forget():
        try:
            registry.remove(worker)
        except ValueError:
            pass

    try:
        worker.finished.connect(_forget)
    except RuntimeError:
        pass
