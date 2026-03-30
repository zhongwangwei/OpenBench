# -*- coding: utf-8 -*-
"""
YAML preview widget with syntax highlighting.
"""

import re

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QPlainTextEdit, QHBoxLayout,
    QPushButton, QApplication
)
from PySide6.QtGui import (
    QSyntaxHighlighter, QTextCharFormat, QColor, QFont
)


class YamlHighlighter(QSyntaxHighlighter):
    """Syntax highlighter for YAML."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_formats()
        self._setup_rules()

    def _setup_formats(self):
        """Setup text formats for different token types."""
        # Key format (before colon)
        self.key_format = QTextCharFormat()
        self.key_format.setForeground(QColor("#569cd6"))
        self.key_format.setFontWeight(QFont.Bold)

        # String value format
        self.string_format = QTextCharFormat()
        self.string_format.setForeground(QColor("#ce9178"))

        # Number format
        self.number_format = QTextCharFormat()
        self.number_format.setForeground(QColor("#b5cea8"))

        # Boolean format
        self.bool_format = QTextCharFormat()
        self.bool_format.setForeground(QColor("#569cd6"))

        # Comment format
        self.comment_format = QTextCharFormat()
        self.comment_format.setForeground(QColor("#6a9955"))
        self.comment_format.setFontItalic(True)

        # List item format (dash)
        self.list_format = QTextCharFormat()
        self.list_format.setForeground(QColor("#d4d4d4"))

    def _setup_rules(self):
        """Setup highlighting rules."""
        self.rules = [
            # Comments
            (r'#.*$', self.comment_format),
            # Keys (word followed by colon)
            (r'^[\s]*[\w_]+(?=\s*:)', self.key_format),
            # Quoted strings
            (r'"[^"]*"', self.string_format),
            (r"'[^']*'", self.string_format),
            # Booleans
            (r'\b(true|false|True|False|yes|no|Yes|No)\b', self.bool_format),
            # Numbers
            (r'\b-?\d+\.?\d*\b', self.number_format),
            # List items
            (r'^\s*-\s', self.list_format),
        ]

    def highlightBlock(self, text: str):
        """Apply highlighting to a block of text."""
        for pattern, fmt in self.rules:
            for match in re.finditer(pattern, text, re.MULTILINE):
                start = match.start()
                length = match.end() - match.start()
                self.setFormat(start, length, fmt)


class YamlPreview(QWidget):
    """Widget for previewing YAML content with syntax highlighting."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        """Setup the widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Text editor
        self.text_edit = QPlainTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setLineWrapMode(QPlainTextEdit.NoWrap)

        # Apply monospace font (cross-platform: Monaco for macOS, Consolas for Windows)
        font = QFont()
        font.setStyleHint(QFont.Monospace)
        font.setFamily("Monaco, Menlo, Consolas, Courier New")
        font.setPointSize(11)
        self.text_edit.setFont(font)

        # Apply syntax highlighter
        self.highlighter = YamlHighlighter(self.text_edit.document())

        layout.addWidget(self.text_edit, 1)

        # Button bar
        btn_bar = QHBoxLayout()
        btn_bar.setSpacing(8)

        btn_bar.addStretch()

        self.btn_copy = QPushButton("Copy to Clipboard")
        self.btn_copy.setProperty("secondary", True)
        self.btn_copy.clicked.connect(self._copy_to_clipboard)
        btn_bar.addWidget(self.btn_copy)

        layout.addLayout(btn_bar)

    def set_content(self, content: str):
        """Set the YAML content to display."""
        self.text_edit.setPlainText(content)

    def get_content(self) -> str:
        """Get the current content."""
        return self.text_edit.toPlainText()

    def _copy_to_clipboard(self):
        """Copy content to clipboard."""
        clipboard = QApplication.clipboard()
        clipboard.setText(self.text_edit.toPlainText())
