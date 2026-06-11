"""Worker thread template for long-running SSH operations.

This module provides a reusable QThread-based worker for SSH calls so
that the Qt main thread is not blocked while the remote server runs
slow commands (e.g. ``conda env update`` or ``git clone`` over a
high-latency link). Without it, RemoteConfigWidget falls back to
calling ``QApplication.processEvents()`` in a loop while
``self._ssh_manager.execute(...)`` blocks — UI freezes for tens of
seconds to fifteen minutes, and macOS may show "Application not
responding".

Usage from a widget::

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
When ``SSHManager.execute_stream`` is available, output is emitted while
the command is still running; older/fake managers fall back to
``execute`` and emit captured output after completion.
"""

from __future__ import annotations

from typing import Optional

try:
    from PySide6.QtCore import QObject, QThread, Signal, Slot
except ImportError:  # pragma: no cover - GUI extra not installed
    QObject = QThread = object  # type: ignore[assignment,misc]
    Signal = None  # type: ignore[assignment]

    def Slot(*args, **kwargs):  # type: ignore[no-redef]
        return lambda func: func


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
            if hasattr(self._ssh_manager, "execute_stream"):
                import time

                stdout_chunks: list[str] = []
                # Batch line emissions: one queued cross-thread signal per
                # line floods the GUI event loop on a 10k-line install log.
                batch: list[str] = []
                last_flush = time.monotonic()

                def _flush_lines():
                    nonlocal last_flush
                    if batch:
                        self.line.emit("".join(batch))
                        batch.clear()
                    last_flush = time.monotonic()

                try:
                    stream = self._ssh_manager.execute_stream(
                        self._command,
                        total_timeout=self._timeout,
                        should_abort=self.isInterruptionRequested,
                    )
                except TypeError:
                    # Older/fake managers without the should_abort probe;
                    # interruption then only lands between output lines.
                    stream = self._ssh_manager.execute_stream(self._command, total_timeout=self._timeout)
                while True:
                    # Interruption is only observable between lines: a hung
                    # command that produces no output keeps next() blocked
                    # until total_timeout.
                    if self.isInterruptionRequested():
                        stream.close()
                        self.failed.emit("Interrupted")
                        return
                    try:
                        raw = next(stream)
                    except StopIteration as done:
                        exit_code = int(done.value or 0)
                        break
                    stdout_chunks.append(raw)
                    if raw.strip():
                        batch.append(raw)
                        if len(batch) >= 50 or (time.monotonic() - last_flush) > 0.2:
                            _flush_lines()
                _flush_lines()
                self.finished_with_result.emit(exit_code, "".join(stdout_chunks), "")
                return

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


class _CallableWorker(QThread):  # type: ignore[misc]
    """Run an arbitrary callable in a background thread for call_responsive."""

    if Signal is not None:  # only define when Qt is available
        done = Signal(object)
        failed = Signal(object)

    def __init__(self, func, parent=None):
        super().__init__(parent)
        self._func = func

    def run(self) -> None:
        try:
            self.done.emit(self._func())
        except Exception as exc:
            self.failed.emit(exc)


def call_responsive(func):
    """Run ``func`` on a worker thread while keeping the GUI event loop alive.

    Synchronous façade: blocks the caller like a direct ``func()`` call, but
    keeps processing Qt events meanwhile, so the window repaints and never
    goes "Application not responding". Falls back to a plain call when there
    is no QApplication (CLI, tests) or when already off the GUI thread.
    Exceptions raised by ``func`` are re-raised in the caller.
    """
    try:
        from PySide6.QtCore import QEventLoop
        from PySide6.QtWidgets import QApplication
    except ImportError:  # pragma: no cover - GUI extra not installed
        return func()

    app = QApplication.instance()
    if app is None or QThread.currentThread() != app.thread():
        return func()

    result: dict = {}
    loop = QEventLoop()
    worker = _CallableWorker(func)
    worker.done.connect(lambda value: result.setdefault("value", value))
    worker.failed.connect(lambda exc: result.setdefault("error", exc))
    worker.finished.connect(loop.quit)
    worker.start()
    loop.exec()
    worker.wait()
    if "error" in result:
        raise result["error"]
    return result["value"]


def execute_responsive(ssh_manager, command: str, timeout: Optional[int] = None):
    """``ssh_manager.execute`` via :func:`call_responsive` — same return tuple."""
    return call_responsive(lambda: ssh_manager.execute(command, timeout=timeout))


class _GuiInvoker(QObject):  # type: ignore[misc]
    """Runs a stored callable in the thread the object lives in."""

    func = None
    result = None
    error = None

    @Slot()
    def invoke(self):
        try:
            self.result = self.func()
        except Exception as exc:
            self.error = exc


def call_on_gui_thread(func):
    """Run ``func`` on the GUI thread and block for its result.

    For callbacks fired on a worker thread that must show Qt UI (e.g. the
    SSH host-key confirmation dialog during a connect run via
    call_responsive). Requires the GUI thread to be processing events —
    which is exactly what call_responsive's nested loop guarantees.
    Passthrough when already on the GUI thread or without a QApplication.
    Exceptions from ``func`` are re-raised in the caller.
    """
    try:
        from PySide6.QtCore import QMetaObject, Qt
        from PySide6.QtWidgets import QApplication
    except ImportError:  # pragma: no cover - GUI extra not installed
        return func()

    app = QApplication.instance()
    if app is None or QThread.currentThread() == app.thread():
        return func()

    invoker = _GuiInvoker()
    invoker.func = func
    invoker.moveToThread(app.thread())
    QMetaObject.invokeMethod(invoker, "invoke", Qt.ConnectionType.BlockingQueuedConnection)
    invoker.deleteLater()
    if invoker.error is not None:
        raise invoker.error
    return invoker.result
