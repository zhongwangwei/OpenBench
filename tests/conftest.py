"""Shared fixtures for the test suite."""

import os

# Must be set before any test module imports PySide6.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest


@pytest.fixture
def qapp():
    """Process-wide QApplication for GUI tests (offscreen platform)."""
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])
