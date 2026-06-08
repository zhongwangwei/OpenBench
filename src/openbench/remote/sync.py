# core/sync_engine.py
# -*- coding: utf-8 -*-
"""
Sync engine for remote storage with local caching.
"""

import posixpath
import re
import shlex
import threading
import logging
import time
from enum import Enum
from typing import Dict, List, Optional, Callable, Set
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class SyncStatus(Enum):
    """Sync status for a file."""

    SYNCED = "synced"  # File is synced with remote
    PENDING = "pending"  # Local changes not yet synced
    SYNCING = "syncing"  # Currently syncing
    ERROR = "error"  # Sync failed


@dataclass
class SyncState:
    """State of a file in the sync engine."""

    status: SyncStatus
    error_message: Optional[str] = None
    retry_count: int = 0


class SyncEngine:
    """
    Manages local cache and background sync with remote server.

    Provides immediate local reads/writes with async sync to remote.
    """

    MAX_RETRIES = 3

    def __init__(
        self,
        ssh_manager,
        remote_project_dir: str,
        on_status_changed: Optional[Callable[[str, SyncStatus], None]] = None,
    ):
        """
        Initialize sync engine.

        Args:
            ssh_manager: SSH manager for remote operations
            remote_project_dir: Remote project directory path
            on_status_changed: Callback when file sync status changes
        """
        self._ssh = ssh_manager
        normalized_remote_dir = posixpath.normpath(remote_project_dir or ".")
        self._remote_dir = "/" if normalized_remote_dir == "/" else normalized_remote_dir.rstrip("/")
        self._on_status_changed = on_status_changed

        # Cache storage
        self._cache: Dict[str, str] = {}
        self._sync_status: Dict[str, SyncStatus] = {}
        self._sync_errors: Dict[str, str] = {}

        # Thread safety. _fetch_cv shares the underlying RLock so all the
        # `with self._lock:` blocks elsewhere in this class continue to work
        # unchanged; only the read() coordination needs Condition semantics.
        self._lock = threading.RLock()
        self._fetch_cv = threading.Condition(self._lock)
        self._fetching: Set[str] = set()  # Files currently being fetched from remote

        # Background sync
        self._pending_sync: Set[str] = set()
        self._sync_thread: Optional[threading.Thread] = None
        self._stop_sync = threading.Event()

    def _remote_path(self, path: str) -> str:
        """Get full remote path."""
        remote_root = posixpath.normpath(self._remote_dir)
        if not path:
            return remote_root
        remote_path = posixpath.normpath(posixpath.join(remote_root, path))
        if remote_root == "/":
            return remote_path
        if remote_path != remote_root and not remote_path.startswith(remote_root + "/"):
            raise ValueError(f"Path escapes remote project directory: {path}")
        return remote_path

    @staticmethod
    def _validate_glob_pattern(pattern: str) -> None:
        """Reject shell syntax in remote glob patterns before interpolation."""
        if pattern.startswith("/") or pattern == ".." or pattern.startswith("../") or "/../" in pattern:
            raise ValueError(f"unsafe glob pattern escapes remote project directory: {pattern}")
        if re.search(r"[;&|$`'\"()<>\\\r\n]", pattern):
            raise ValueError(f"unsafe glob pattern contains shell metacharacters: {pattern}")

    def read(self, path: str) -> str:
        """
        Read file, from cache if available, otherwise from remote.

        Thread-safe: avoids duplicate fetches if multiple threads request
        the same file simultaneously.

        Args:
            path: Relative path from project directory

        Returns:
            File contents
        """
        # First check: is it in cache?
        with self._lock:
            if path in self._cache:
                return self._cache[path]

            # Wait if another thread is already fetching this file.
            # Condition.wait() atomically releases the underlying lock and
            # re-acquires on wakeup; no manual release/acquire pairs that
            # could leak the lock or _fetching state on KeyboardInterrupt.
            while path in self._fetching:
                # Timeout lets us periodically re-check for the cache entry
                # even if a notify is missed (defensive — notify_all is
                # always issued by the producing thread).
                self._fetch_cv.wait(timeout=0.1)
                if path in self._cache:
                    return self._cache[path]

            # Mark as being fetched and exit the lock to do I/O unlocked
            self._fetching.add(path)

        try:
            # Fetch from remote (outside lock to avoid blocking other operations).
            # Use SFTP to read raw bytes then strict-decode as UTF-8, instead
            # of `cat` whose output is funneled through paramiko's lossy
            # errors="replace" decode — that previously corrupted any
            # non-UTF-8 byte and a follow-up write() would commit the
            # corruption back to the server.
            remote_path = self._remote_path(path)
            try:
                sftp = self._ssh.open_sftp()
                with sftp.open(remote_path, "rb") as remote_file:
                    raw = remote_file.read()
            except FileNotFoundError:
                raise FileNotFoundError(f"Remote file not found: {remote_path}")
            try:
                content = raw.decode("utf-8")
            except UnicodeDecodeError as e:
                raise UnicodeError(
                    f"Remote file {remote_path!r} is not valid UTF-8 and cannot be "
                    f"safely round-tripped through SyncedRemoteFiles.write(): {e}"
                ) from e

            with self._lock:
                self._cache[path] = content
                self._sync_status[path] = SyncStatus.SYNCED
                self._fetching.discard(path)
                self._fetch_cv.notify_all()  # Wake all waiters
            return content
        except Exception:
            # Even on failure, clear _fetching and notify waiters so they
            # don't deadlock on a permanently-stuck path.
            with self._lock:
                self._fetching.discard(path)
                self._fetch_cv.notify_all()
            raise

    def write(self, path: str, content: str) -> None:
        """
        Write to local cache and queue for sync.

        Args:
            path: Relative path from project directory
            content: Content to write
        """
        with self._lock:
            self._cache[path] = content
            self._sync_status[path] = SyncStatus.PENDING
            self._pending_sync.add(path)

        self._notify_status_changed(path, SyncStatus.PENDING)

    def get_sync_status(self, path: str) -> SyncStatus:
        """Get sync status for a file."""
        with self._lock:
            return self._sync_status.get(path, SyncStatus.SYNCED)

    def mark_synced(self, path: str, content: str) -> None:
        """Record a file as already synced after an out-of-band remote write.

        Some GUI workflows still upload via SFTP for compatibility with
        existing remote runners. Without updating the sync cache, a stale
        pending entry can later be flushed by the background sync thread and
        overwrite the just-uploaded remote file.
        """
        with self._lock:
            self._cache[path] = content
            self._sync_status[path] = SyncStatus.SYNCED
            self._pending_sync.discard(path)
            self._sync_errors.pop(path, None)

        self._notify_status_changed(path, SyncStatus.SYNCED)

    def get_overall_status(self) -> SyncStatus:
        """Get overall sync status."""
        with self._lock:
            statuses = set(self._sync_status.values())
            if SyncStatus.ERROR in statuses:
                return SyncStatus.ERROR
            if SyncStatus.SYNCING in statuses:
                return SyncStatus.SYNCING
            if SyncStatus.PENDING in statuses:
                return SyncStatus.PENDING
            return SyncStatus.SYNCED

    def sync_all(self) -> bool:
        """
        Sync all pending changes to remote.

        Returns:
            True if all syncs succeeded
        """
        with self._lock:
            pending = list(self._pending_sync)

        success = True
        for path in pending:
            if not self._sync_file(path):
                success = False

        return success

    def _sync_file(self, path: str) -> bool:
        """
        Sync a single file to remote.

        Returns:
            True if sync succeeded
        """
        with self._lock:
            if path not in self._cache:
                return True
            content = self._cache[path]
            self._sync_status[path] = SyncStatus.SYNCING

        self._notify_status_changed(path, SyncStatus.SYNCING)

        last_error: Exception | None = None
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                remote_path = self._remote_path(path)

                # Ensure remote directory exists
                remote_dir = posixpath.dirname(remote_path)
                if remote_dir:
                    _stdout, stderr, exit_code = self._ssh.execute(f"mkdir -p {shlex.quote(remote_dir)}", timeout=10)
                    if exit_code != 0:
                        raise Exception(f"Create remote directory failed: {stderr}")

                # Write through SFTP rather than embedding file contents in a
                # remote shell command.  Shell command strings are bounded by
                # ARG_MAX and cannot safely carry NUL bytes; SFTP preserves the
                # exact UTF-8 payload including no trailing newline and NULs.
                sftp = self._ssh.open_sftp()
                with sftp.open(remote_path, "wb") as remote_file:
                    remote_file.write(content.encode("utf-8"))

                with self._lock:
                    self._sync_status[path] = SyncStatus.SYNCED
                    self._pending_sync.discard(path)
                    self._sync_errors.pop(path, None)

                self._notify_status_changed(path, SyncStatus.SYNCED)
                return True

            except Exception as e:
                last_error = e
                # Don't burn retries on permission / "command not found" —
                # these won't change between attempts and only delay the
                # user seeing the actual failure.
                err_str = str(e).lower()
                is_permanent = (
                    isinstance(e, PermissionError)
                    or "permission denied" in err_str
                    or "operation not permitted" in err_str
                    or "no such file or directory" in err_str
                    or "not a directory" in err_str
                )
                if is_permanent:
                    logger.warning("Sync giving up early on permanent error for %s: %s", path, e)
                    break
                if attempt < self.MAX_RETRIES:
                    logger.warning("Sync failed for %s (attempt %d/%d): %s", path, attempt + 1, self.MAX_RETRIES + 1, e)
                    time.sleep(min(0.25 * (2**attempt), 2.0))
                    continue

        logger.error(f"Sync failed for {path}: {last_error}")
        with self._lock:
            self._sync_status[path] = SyncStatus.ERROR
            self._sync_errors[path] = str(last_error)
        self._notify_status_changed(path, SyncStatus.ERROR)
        return False

    def _notify_status_changed(self, path: str, status: SyncStatus):
        """Notify callback of status change."""
        if self._on_status_changed:
            try:
                self._on_status_changed(path, status)
            except Exception as e:
                logger.error(f"Status callback error: {e}")

    def list_dir(self, path: str) -> List[str]:
        """List remote directory contents.

        Raises:
            IOError: When the listing fails (e.g. permission denied, missing
                directory, network error). Previously this returned ``[]``
                indistinguishably from a genuinely empty directory.
        """
        remote_path = self._remote_path(path)
        # Don't swallow stderr in the shell — surface it on failure so
        # callers can tell "permission denied" / "no such directory" apart
        # from an empty directory.
        stdout, stderr, exit_code = self._ssh.execute(f"ls -1 {shlex.quote(remote_path)}", timeout=30)
        if exit_code != 0:
            raise IOError(
                f"list_dir({remote_path!r}) failed (exit {exit_code}): "
                f"{stderr.strip() or stdout.strip() or 'ls produced no diagnostics'}"
            )
        return [line.strip() for line in stdout.strip().split("\n") if line.strip()]

    def exists(self, path: str) -> bool:
        """Check if remote path exists."""
        # Check cache first
        with self._lock:
            if path in self._cache:
                return True

        remote_path = self._remote_path(path)
        stdout, stderr, exit_code = self._ssh.execute(
            f"test -e {shlex.quote(remote_path)} && echo 'exists'", timeout=10
        )
        return exit_code == 0 and "exists" in stdout

    def glob(self, pattern: str) -> List[str]:
        """Find files matching pattern on remote.

        Supports standard glob patterns including ** for recursive matching.
        """
        self._validate_glob_pattern(pattern)
        base_dir = self._remote_dir
        # base_dir must be quoted (path may contain spaces/specials);
        # pattern is intentionally left literal so bash can expand globs.
        # Wrap the whole inner script with shlex.quote to survive single
        # quotes in base_dir (the previous f"bash -c '{cmd}'" form broke).
        inner = (
            f"cd {shlex.quote(base_dir)} && shopt -s globstar nullglob && "
            f'for f in {pattern}; do [ -f "$f" ] && echo "$f"; done'
        )
        stdout, stderr, exit_code = self._ssh.execute(f"bash -c {shlex.quote(inner)}", timeout=30)
        if exit_code != 0:
            return []
        return [line.strip() for line in stdout.strip().split("\n") if line.strip()]

    def mkdir(self, path: str) -> None:
        """Create remote directory."""
        remote_path = self._remote_path(path)
        _stdout, stderr, exit_code = self._ssh.execute(f"mkdir -p {shlex.quote(remote_path)}", timeout=10)
        if exit_code != 0:
            raise Exception(f"Create remote directory failed: {stderr}")

    def delete(self, path: str) -> None:
        """Delete a remote file or directory under the project root.

        Directories are removed recursively after root/escape guards pass.
        Previously ``rm -f`` was used, which always exits 0 — a
        permission error or a path-outside-project case would go
        completely undetected and the local cache would be cleared as
        if the delete had succeeded. Use an explicit file-vs-directory
        command and a separate existence probe so genuine failures surface
        as exceptions rather than silently desynchronising local and remote
        state.
        """
        if path in {"", ".", "./"}:
            raise ValueError("Refusing to delete remote project root")
        remote_path = self._remote_path(path)
        remote_root = posixpath.normpath(self._remote_dir)
        if remote_path == remote_root:
            raise ValueError("Refusing to delete remote project root")
        quoted = shlex.quote(remote_path)
        # Probe existence first so we can distinguish "not there" (no-op,
        # like POSIX rm -f) from "could not delete" (permission etc.).
        _, _, exit_code = self._ssh.execute(f"test -e {quoted}", timeout=10)
        if exit_code == 0:
            delete_cmd = f"if [ -d {quoted} ] && [ ! -L {quoted} ]; then rm -rf {quoted}; else rm -f {quoted}; fi"
            stdout, stderr, exit_code = self._ssh.execute(delete_cmd, timeout=10)
            if exit_code != 0:
                raise OSError(
                    f"Failed to delete remote path {remote_path!r}: "
                    f"{stderr.strip() or stdout.strip() or 'rm exited '}"
                    f"(exit code {exit_code})"
                )

        with self._lock:
            self._cache.pop(path, None)
            self._sync_status.pop(path, None)
            self._pending_sync.discard(path)

    def start_background_sync(self, interval: float = 2.0):
        """Start background sync thread."""
        if self._sync_thread and self._sync_thread.is_alive():
            return

        self._stop_sync.clear()
        self._sync_thread = threading.Thread(target=self._background_sync_loop, args=(interval,), daemon=True)
        self._sync_thread.start()

    def stop_background_sync(self):
        """Stop background sync thread."""
        self._stop_sync.set()
        if self._sync_thread:
            self._sync_thread.join(timeout=5)

    def _background_sync_loop(self, interval: float):
        """Background sync loop."""
        while not self._stop_sync.wait(interval):
            with self._lock:
                pending = list(self._pending_sync)

            for path in pending:
                if self._stop_sync.is_set():
                    break
                self._sync_file(path)

    def load_project(self) -> None:
        """
        Load all project files into cache.

        Call this when opening a remote project.
        """
        # Load nml directory structure
        nml_files = self.glob("nml/**/*.yaml")
        for path in nml_files:
            try:
                self.read(path)
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")

    def get_pending_count(self) -> int:
        """Get number of files pending sync."""
        with self._lock:
            return len(self._pending_sync)

    def get_error_files(self) -> Dict[str, str]:
        """Get files with sync errors and their error messages."""
        with self._lock:
            return dict(self._sync_errors)

    def retry_errors(self) -> bool:
        """Retry syncing files that had errors."""
        with self._lock:
            error_files = [path for path, status in self._sync_status.items() if status == SyncStatus.ERROR]

        success = True
        for path in error_files:
            if not self._sync_file(path):
                success = False

        return success
