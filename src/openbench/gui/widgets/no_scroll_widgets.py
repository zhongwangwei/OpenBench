# -*- coding: utf-8 -*-
"""
Custom widgets that completely ignore scroll wheel events to prevent accidental value changes.
Users can still use arrow buttons, keyboard arrows, or type values directly.
"""

from PySide6.QtWidgets import QSpinBox, QDoubleSpinBox, QComboBox


class NoScrollSpinBox(QSpinBox):
    """SpinBox that completely ignores scroll wheel events."""

    def wheelEvent(self, event):
        # Always ignore scroll wheel - pass to parent for page scrolling
        event.ignore()


class NoScrollDoubleSpinBox(QDoubleSpinBox):
    """DoubleSpinBox that completely ignores scroll wheel events."""

    def wheelEvent(self, event):
        # Always ignore scroll wheel - pass to parent for page scrolling
        event.ignore()


class NoScrollComboBox(QComboBox):
    """ComboBox that completely ignores scroll wheel events."""

    def wheelEvent(self, event):
        # Always ignore scroll wheel - pass to parent for page scrolling
        event.ignore()
