# ui/widgets/path_completer.py
# -*- coding: utf-8 -*-
"""
Path autocomplete for storage-backed path input.
"""

from typing import Optional, List
from PySide6.QtWidgets import QCompleter
from PySide6.QtCore import Qt, QStringListModel, QTimer

from openbench.remote.storage import ProjectStorage


class PathCompleterModel(QStringListModel):
    """Model that fetches path completions from storage."""

    def __init__(self, storage: Optional[ProjectStorage] = None, parent=None):
        super().__init__(parent)
        self._storage = storage
        self._cache: dict = {}

    def set_storage(self, storage: Optional[ProjectStorage]):
        """Set the storage backend."""
        self._storage = storage
        self._cache.clear()

    def update_completions(self, text: str) -> List[str]:
        """
        Update completions for the given text.

        Args:
            text: Current input text

        Returns:
            List of completion suggestions
        """
        if not self._storage or not text:
            self.setStringList([])
            return []

        # Skip absolute paths - ProjectStorage only handles relative paths
        # Absolute paths are outside the project scope
        if text.startswith("/"):
            self.setStringList([])
            return []

        # Get directory part and prefix
        if "/" in text:
            dir_part, prefix = text.rsplit("/", 1)
        else:
            dir_part = ""
            prefix = text

        # Check cache
        cache_key = dir_part
        if cache_key not in self._cache:
            try:
                items = self._storage.list_dir(dir_part)
                self._cache[cache_key] = items
            except Exception:
                self._cache[cache_key] = []

        items = self._cache.get(cache_key, [])

        # Filter by prefix and build full paths
        completions = []
        for item in items:
            if item.lower().startswith(prefix.lower()):
                if dir_part:
                    full_path = f"{dir_part}/{item}"
                else:
                    full_path = item
                completions.append(full_path)

        # Sort: directories first, then files
        completions.sort(key=str.lower)

        self.setStringList(completions)
        return completions

    def clear_cache(self):
        """Clear the completion cache."""
        self._cache.clear()


class PathCompleter(QCompleter):
    """
    Path completer with delayed fetching for storage backends.
    """

    DELAY_MS = 300  # Delay before fetching completions

    def __init__(self, storage: Optional[ProjectStorage] = None, parent=None):
        self._model = PathCompleterModel(storage, parent)
        super().__init__(self._model, parent)

        self.setCaseSensitivity(Qt.CaseInsensitive)
        self.setCompletionMode(QCompleter.PopupCompletion)
        self.setFilterMode(Qt.MatchStartsWith)

        # Delay timer for remote fetching
        self._delay_timer = QTimer(self)
        self._delay_timer.setSingleShot(True)
        self._delay_timer.timeout.connect(self._fetch_completions)
        self._pending_text = ""

    def set_storage(self, storage: Optional[ProjectStorage]):
        """Set the storage backend."""
        self._model.set_storage(storage)

    def update_completions(self, text: str):
        """
        Request completion update for text.

        Uses delayed fetching to avoid excessive remote calls.
        """
        self._pending_text = text
        self._delay_timer.start(self.DELAY_MS)

    def _fetch_completions(self):
        """Fetch completions after delay."""
        self._model.update_completions(self._pending_text)

    def clear_cache(self):
        """Clear the completion cache."""
        self._model.clear_cache()
