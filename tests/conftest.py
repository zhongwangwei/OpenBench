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


@pytest.fixture(autouse=True)
def _fast_credential_manager(monkeypatch):
    """Keep RemoteConfigWidget construction cheap and out of the real home dir.

    The production CredentialManager runs 100k PBKDF2 iterations and reads/
    writes a salt file in ~/.openbench_wizard on every __init__; tests that
    construct RemoteConfigWidget don't exercise credentials, so stub it.
    """
    try:
        import openbench.gui.widgets.remote_config as remote_config
    except Exception:
        return

    class _StubCredentialManager:
        def __init__(self, *args, **kwargs):
            pass

        def save_credential(self, *args, **kwargs):
            pass

        def get_credential(self, *args, **kwargs):
            return None

        def clear_all(self, *args, **kwargs):
            pass

    monkeypatch.setattr(remote_config, "CredentialManager", _StubCredentialManager)
