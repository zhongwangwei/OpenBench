# -*- coding: utf-8 -*-
"""
Base class for all wizard pages.
"""

from abc import abstractmethod
from typing import Dict, Any

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel,
    QScrollArea, QFrame
)
from PySide6.QtCore import Qt

from openbench.gui.path_utils import get_openbench_root


class BasePage(QWidget):
    """Base class for wizard pages."""

    # Override in subclasses
    PAGE_ID = "base"
    PAGE_TITLE = "Base Page"
    PAGE_SUBTITLE = ""
    # Set to True in subclasses to allow content to expand and fill available space
    # Set to False (default) to keep content at top with stretch at bottom
    CONTENT_EXPAND = False

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Setup the page UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)

        # Header
        header = QWidget()
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(5)

        title = QLabel(self.PAGE_TITLE)
        title.setProperty("heading", True)
        title.setStyleSheet("font-size: 24px; font-weight: 600; color: #333;")
        header_layout.addWidget(title)

        if self.PAGE_SUBTITLE:
            subtitle = QLabel(self.PAGE_SUBTITLE)
            subtitle.setProperty("subheading", True)
            subtitle.setStyleSheet("font-size: 14px; color: #666;")
            header_layout.addWidget(subtitle)

        layout.addWidget(header)

        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setStyleSheet("background-color: #e0e0e0;")
        separator.setFixedHeight(1)
        layout.addWidget(separator)

        # Scroll area for content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Content container
        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(0, 0, 20, 0)
        self.content_layout.setSpacing(15)

        # Let subclasses add their content
        self._setup_content()

        # Add stretch at the end only if content should not expand
        # Subclasses can set CONTENT_EXPAND = True to fill available space
        if not self.CONTENT_EXPAND:
            self.content_layout.addStretch()

        scroll.setWidget(self.content)
        layout.addWidget(scroll, 1)

    @abstractmethod
    def _setup_content(self):
        """Override to add page-specific content to self.content_layout."""
        pass

    def _connect_signals(self):
        """Connect signals. Override to add custom signal connections."""
        self.controller.config_updated.connect(self._on_config_updated)

    def _on_config_updated(self, config: Dict[str, Any]):
        """Handle config updates. Override to react to config changes."""
        pass

    def load_from_config(self):
        """Load page state from controller config. Override in subclasses."""
        pass

    def save_to_config(self):
        """Save page state to controller config. Override in subclasses."""
        pass

    def validate(self) -> bool:
        """Validate page input. Override in subclasses. Return False to prevent navigation."""
        return True

    def _get_openbench_root(self) -> str:
        """Get the OpenBench root directory.

        Uses controller's project_root if available, otherwise falls back
        to the default detected OpenBench root.

        Returns:
            Path to the OpenBench root directory
        """
        if self.controller.project_root:
            return self.controller.project_root
        return get_openbench_root()

    def showEvent(self, event):
        """Called when page becomes visible."""
        super().showEvent(event)
        self.load_from_config()
