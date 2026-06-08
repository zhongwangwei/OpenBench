"""Background workers for GUI registry scans."""

from __future__ import annotations

from PySide6.QtCore import QThread, Signal


class FindDatasetsWorker(QThread):
    """Run registry discovery off the Qt main thread."""

    finished_with_result = Signal(object)
    failed = Signal(str)

    def __init__(self, data_root: str, parent=None):
        super().__init__(parent)
        self._data_root = data_root

    def run(self) -> None:  # pragma: no cover - exercised through GUI integration
        try:
            from openbench.data.registry.scanner import find_new_datasets

            self.finished_with_result.emit(find_new_datasets(self._data_root))
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
