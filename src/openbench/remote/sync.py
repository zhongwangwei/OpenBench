# core/sync_engine.py
# -*- coding: utf-8 -*-
"""
Sync engine for remote storage with local caching.
"""

import os
import threading
import logging
from enum import Enum
from typing import Dict, List, Optional, Callable, Set
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class SyncStatus(Enum):
    """Sync status for a file."""
    SYNCED = "synced"      # File is synced with remote
    PENDING = "pending"    # Local changes not yet synced
    SYNCING = "syncing"    # Currently syncing
    ERROR = "error"        # Sync failed


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
        on_status_changed: Optional[Callable[[str, SyncStatus], None]] = None
    ):
        """
        Initialize sync engine.

        Args:
            ssh_manager: SSH manager for remote operations
            remote_project_dir: Remote project directory path
            on_status_changed: Callback when file sync status changes
        """
        self._ssh = ssh_manager
        self._remote_dir = remote_project_dir.rstrip('/')
        self._on_status_changed = on_status_changed

        # Cache storage
        self._cache: Dict[str, str] = {}
        self._sync_status: Dict[str, SyncStatus] = {}
        self._sync_errors: Dict[str, str] = {}

        # Thread safety
        self._lock = threading.RLock()
        self._fetching: Set[str] = set()  # Files currently being fetched from remote

        # Background sync
        self._pending_sync: Set[str] = set()
        self._sync_thread: Optional[threading.Thread] = None
        self._stop_sync = threading.Event()

    def _remote_path(self, path: str) -> str:
        """Get full remote path."""
        if not path:
            return self._remote_dir
        return f"{self._remote_dir}/{path}"

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

            # Check if another thread is fetching this file
            # If so, wait for it to complete by polling
            while path in self._fetching:
                self._lock.release()
                import time
                time.sleep(0.05)  # Small delay before retry
                self._lock.acquire()
                # Check cache again after waiting
                if path in self._cache:
                    return self._cache[path]

            # Mark as being fetched
            self._fetching.add(path)

        try:
            # Fetch from remote (outside lock to avoid blocking other operations)
            remote_path = self._remote_path(path)
            stdout, stderr, exit_code = self._ssh.execute(
                f"cat '{remote_path}'", timeout=30
            )

            if exit_code != 0:
                raise FileNotFoundError(f"Remote file not found: {remote_path}")

            content = stdout

            with self._lock:
                self._cache[path] = content
                self._sync_status[path] = SyncStatus.SYNCED

            return content
        finally:
            with self._lock:
                self._fetching.discard(path)

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

        try:
            remote_path = self._remote_path(path)

            # Ensure remote directory exists
            remote_dir = os.path.dirname(remote_path)
            if remote_dir:
                self._ssh.execute(f"mkdir -p '{remote_dir}'", timeout=10)

            # Write content using heredoc with quoted delimiter (prevents shell expansion)
            # Generate a unique delimiter that doesn't appear in the content
            delimiter = "EOFCONTENT"
            counter = 0
            while delimiter in content:
                counter += 1
                delimiter = f"EOF_SYNC_{counter}_{hash(content) & 0xFFFFFFFF:08X}"
            cmd = f"cat > '{remote_path}' << '{delimiter}'\n{content}\n{delimiter}"
            stdout, stderr, exit_code = self._ssh.execute(cmd, timeout=30)

            if exit_code != 0:
                raise Exception(f"Write failed: {stderr}")

            with self._lock:
                self._sync_status[path] = SyncStatus.SYNCED
                self._pending_sync.discard(path)
                self._sync_errors.pop(path, None)

            self._notify_status_changed(path, SyncStatus.SYNCED)
            return True

        except Exception as e:
            logger.error(f"Sync failed for {path}: {e}")
            with self._lock:
                self._sync_status[path] = SyncStatus.ERROR
                self._sync_errors[path] = str(e)
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
        """List remote directory contents."""
        remote_path = self._remote_path(path)
        stdout, stderr, exit_code = self._ssh.execute(
            f"ls -1 '{remote_path}' 2>/dev/null", timeout=30
        )
        if exit_code != 0:
            return []
        return [line.strip() for line in stdout.strip().split('\n') if line.strip()]

    def exists(self, path: str) -> bool:
        """Check if remote path exists."""
        # Check cache first
        with self._lock:
            if path in self._cache:
                return True

        remote_path = self._remote_path(path)
        stdout, stderr, exit_code = self._ssh.execute(
            f"test -e '{remote_path}' && echo 'exists'", timeout=10
        )
        return exit_code == 0 and 'exists' in stdout

    def glob(self, pattern: str) -> List[str]:
        """Find files matching pattern on remote.

        Supports standard glob patterns including ** for recursive matching.
        """
        base_dir = self._remote_dir
        # Use bash with globstar for ** support, ls to list matches
        # shopt -s globstar enables ** pattern; nullglob prevents literal pattern on no match
        cmd = f"cd '{base_dir}' && shopt -s globstar nullglob && for f in {pattern}; do [ -f \"$f\" ] && echo \"$f\"; done"
        stdout, stderr, exit_code = self._ssh.execute(
            f"bash -c '{cmd}'",
            timeout=30
        )
        if exit_code != 0:
            return []
        return [line.strip() for line in stdout.strip().split('\n') if line.strip()]

    def mkdir(self, path: str) -> None:
        """Create remote directory."""
        remote_path = self._remote_path(path)
        self._ssh.execute(f"mkdir -p '{remote_path}'", timeout=10)

    def delete(self, path: str) -> None:
        """Delete remote file or directory."""
        remote_path = self._remote_path(path)
        self._ssh.execute(f"rm -f '{remote_path}'", timeout=10)

        with self._lock:
            self._cache.pop(path, None)
            self._sync_status.pop(path, None)
            self._pending_sync.discard(path)

    def start_background_sync(self, interval: float = 2.0):
        """Start background sync thread."""
        if self._sync_thread and self._sync_thread.is_alive():
            return

        self._stop_sync.clear()
        self._sync_thread = threading.Thread(
            target=self._background_sync_loop,
            args=(interval,),
            daemon=True
        )
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
            error_files = [
                path for path, status in self._sync_status.items()
                if status == SyncStatus.ERROR
            ]

        success = True
        for path in error_files:
            if not self._sync_file(path):
                success = False

        return success
