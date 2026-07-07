# core/storage.py
# -*- coding: utf-8 -*-
"""
Unified storage interface for local and remote file operations.
"""

import os
import glob as glob_module
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from openbench.remote.sync import SyncEngine


class ProjectStorage(ABC):
    """Abstract interface for project file storage operations."""

    def __init__(self, project_dir: str):
        """
        Initialize storage with project directory.

        Args:
            project_dir: Base directory for all operations
        """
        self._project_dir = project_dir

    @property
    def project_dir(self) -> str:
        """Get the project directory."""
        return self._project_dir

    @abstractmethod
    def read_file(self, path: str) -> str:
        """
        Read file contents.

        Args:
            path: Relative path from project directory

        Returns:
            File contents as string
        """
        pass

    @abstractmethod
    def write_file(self, path: str, content: str) -> None:
        """
        Write content to file.

        Args:
            path: Relative path from project directory
            content: Content to write
        """
        pass

    @abstractmethod
    def list_dir(self, path: str) -> List[str]:
        """
        List directory contents.

        Args:
            path: Relative path from project directory

        Returns:
            List of file/directory names
        """
        pass

    @abstractmethod
    def exists(self, path: str) -> bool:
        """
        Check if path exists.

        Args:
            path: Relative path from project directory

        Returns:
            True if path exists
        """
        pass

    @abstractmethod
    def glob(self, pattern: str) -> List[str]:
        """
        Find files matching pattern.

        Args:
            pattern: Glob pattern (e.g., "nml/*.yaml")

        Returns:
            List of matching paths relative to project directory
        """
        pass

    @abstractmethod
    def mkdir(self, path: str) -> None:
        """
        Create directory (including parents).

        Args:
            path: Relative path from project directory
        """
        pass

    @abstractmethod
    def delete(self, path: str) -> None:
        """
        Delete file or empty directory.

        Args:
            path: Relative path from project directory
        """
        pass


class LocalStorage(ProjectStorage):
    """Storage implementation for local filesystem."""

    def _full_path(self, path: str) -> str:
        """Get full path from relative path.

        Raises:
            ValueError: If path attempts to escape project directory
        """
        project_root = Path(self._project_dir).resolve()
        if not path:
            return str(project_root)

        full_path = (project_root / path).resolve(strict=False)
        try:
            full_path.relative_to(project_root)
        except ValueError:
            raise ValueError(f"Path escapes project directory: {path}")
        return str(full_path)

    def read_file(self, path: str) -> str:
        full_path = self._full_path(path)
        with open(full_path, "r", encoding="utf-8") as f:
            return f.read()

    def write_file(self, path: str, content: str) -> None:
        full_path = self._full_path(path)
        # Ensure directory exists
        dir_path = os.path.dirname(full_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)

    def list_dir(self, path: str) -> List[str]:
        full_path = self._full_path(path)
        if not os.path.isdir(full_path):
            return []
        return os.listdir(full_path)

    def exists(self, path: str) -> bool:
        return os.path.exists(self._full_path(path))

    def glob(self, pattern: str) -> List[str]:
        full_pattern = self._full_path(pattern)
        # Use recursive=True to support ** patterns
        matches = glob_module.glob(full_pattern, recursive=True)
        # Return paths relative to project directory
        return [os.path.relpath(m, self._project_dir) for m in matches]

    def mkdir(self, path: str) -> None:
        os.makedirs(self._full_path(path), exist_ok=True)

    def delete(self, path: str) -> None:
        full_path = self._full_path(path)
        if os.path.isfile(full_path):
            os.remove(full_path)
        elif os.path.isdir(full_path):
            os.rmdir(full_path)
        elif not os.path.exists(full_path):
            raise FileNotFoundError(f"Path does not exist: {path}")


class RemoteStorage(ProjectStorage):
    """Storage implementation for remote server via SyncEngine."""

    def __init__(self, project_dir: str, sync_engine: "SyncEngine"):
        """
        Initialize remote storage.

        Args:
            project_dir: Remote project directory path
            sync_engine: SyncEngine instance for caching and sync
        """
        super().__init__(project_dir)
        self._sync = sync_engine

    @property
    def sync_engine(self) -> "SyncEngine":
        """Get the sync engine."""
        return self._sync

    def read_file(self, path: str) -> str:
        return self._sync.read(path)

    def write_file(self, path: str, content: str) -> None:
        self._sync.write(path, content)

    def mark_synced(self, path: str, content: str) -> None:
        """Update the sync cache after an out-of-band remote write."""
        self._sync.mark_synced(path, content)

    def list_dir(self, path: str) -> List[str]:
        return self._sync.list_dir(path)

    def exists(self, path: str) -> bool:
        return self._sync.exists(path)

    def glob(self, pattern: str) -> List[str]:
        return self._sync.glob(pattern)

    def mkdir(self, path: str) -> None:
        self._sync.mkdir(path)

    def delete(self, path: str) -> None:
        self._sync.delete(path)
