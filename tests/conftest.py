"""Shared fixtures for the test suite."""

import os
import sys

# Must be set before any test module imports PySide6.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest


@pytest.fixture(autouse=True)
def _isolated_user_config(monkeypatch, tmp_path):
    """Tests must never read, compact, or overwrite the developer's catalog."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.delenv("OPENBENCH_HOME", raising=False)
    _clear_loaded_registry_cache()
    yield
    _clear_loaded_registry_cache()


def _clear_loaded_registry_cache():
    # Only a loaded registry holds a cache. Importing it here would make every
    # test need PyYAML; the wheel smoke job installs only build and pytest.
    manager = sys.modules.get("openbench.data.registry.manager")
    if manager is not None:
        manager.clear_registry_cache()


@pytest.fixture
def qapp():
    """Reuse QApplication, but do not leak widgets or event filters across tests."""
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QCoreApplication, QEvent
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app

    manager = getattr(app, "_openbench_language_manager", None)
    if manager is not None:
        app.removeEventFilter(manager)
        del app._openbench_language_manager
    for widget in app.topLevelWidgets():
        # Do not invoke closeEvent: it can open confirmation dialogs in teardown.
        widget.hide()
        widget.deleteLater()
    if manager is not None:
        manager.deleteLater()
    # processEvents() alone does not deliver deferred QObject deletions.
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    app.processEvents()


@pytest.fixture(autouse=True)
def _fast_credential_manager(monkeypatch, _isolated_user_config):
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
