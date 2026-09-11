"""Exercise fixture teardown in a separate process, not the suite's QApplication."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


def test_qapp_cleans_widgets_and_language_filter_between_tests(tmp_path):
    pytest.importorskip("PySide6")
    test_file = tmp_path / "test_qapp_isolation.py"
    test_file.write_text(
        textwrap.dedent("""\
        from PySide6.QtWidgets import QWidget
        from shiboken6 import isValid
        from openbench.gui.localization import LanguageManager

        retained = {}

        def test_create_widgets(qapp):
            retained["app"] = qapp
            retained["window"] = QWidget()
            retained["child"] = QWidget(retained["window"])
            retained["pending"] = QWidget()
            retained["pending"].deleteLater()
            manager = LanguageManager(qapp, persist=False)
            retained["manager"] = manager
            qapp._openbench_language_manager = manager
            qapp.installEventFilter(manager)

        def test_previous_widgets_are_destroyed(qapp):
            assert qapp is retained["app"]
            for name in ("window", "child", "pending", "manager"):
                assert not isValid(retained[name]), name
            assert not qapp.topLevelWidgets()
            assert getattr(qapp, "_openbench_language_manager", None) is None
        """),
        encoding="utf-8",
    )
    repo = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(repo / "src"), str(repo))))
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--noconftest", "-p", "tests.conftest", str(test_file)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
