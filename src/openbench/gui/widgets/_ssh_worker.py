"""Worker thread template for long-running SSH operations.

This module provides a reusable QThread-based worker for SSH calls so
that the Qt main thread is not blocked while the remote server runs
slow commands (e.g. ``conda env update`` or ``git clone`` over a
high-latency link). Without it, RemoteConfigWidget falls back to
calling ``QApplication.processEvents()`` in a loop while
``self._ssh_manager.execute(...)`` blocks — UI freezes for tens of
seconds to fifteen minutes, and macOS may show "Application not
responding".

Recommended usage from a widget::

    worker = SshExecuteWorker(
        ssh_manager=self._ssh_manager,
        command="conda env update -n env -f reqs.yml",
        timeout=900,
    )
    worker.line.connect(self._append_status)   # streamed output
    worker.finished_with_result.connect(self._on_install_done)
    worker.failed.connect(self._on_install_failed)
    worker.start()

The widget keeps a reference to ``worker`` until ``finished`` fires;
otherwise QThread is GC'd and Qt will warn about a dangling thread.

NOTE: This is a template — RemoteConfigWidget's existing inline
``processEvents()`` callsites (12 of them at the time of writing)
have not yet been migrated. Migration is tracked as a follow-up; the
template here is intended to keep future SSH UI work on a single
clear pattern.
"""

from __future__ import annotations

from typing import Optional

try:
    from PySide6.QtCore import QThread, Signal
except ImportError:  # pragma: no cover - GUI extra not installed
    QThread = object  # type: ignore[assignment,misc]
    Signal = None  # type: ignore[assignment]


class SshExecuteWorker(QThread):  # type: ignore[misc]
    """Run a single SSH command in a background thread.

    Emits:
      - ``line(str)``: each non-empty line of streamed stdout/stderr.
      - ``finished_with_result(int, str, str)``: exit_code, stdout, stderr
        once the command completes successfully.
      - ``failed(str)``: human-readable error message if the call raised.
    """

    if Signal is not None:  # only define when Qt is available
        line = Signal(str)
        finished_with_result = Signal(int, str, str)
        failed = Signal(str)

    def __init__(
        self,
        ssh_manager,
        command: str,
        timeout: Optional[int] = None,
        parent=None,
    ):
        super().__init__(parent)
        self._ssh_manager = ssh_manager
        self._command = command
        self._timeout = timeout

    def run(self) -> None:  # pragma: no cover - exercised only with GUI
        try:
            stdout, stderr, exit_code = self._ssh_manager.execute(self._command, timeout=self._timeout)
            for raw in (stdout or "").splitlines():
                if raw.strip():
                    self.line.emit(raw)
            for raw in (stderr or "").splitlines():
                if raw.strip():
                    self.line.emit(raw)
            self.finished_with_result.emit(exit_code, stdout or "", stderr or "")
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
