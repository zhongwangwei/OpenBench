import pytest

pytest.importorskip("PySide6")

from openbench.gui.controller import WizardController  # noqa: E402
from openbench.gui.pages.page_general import PageGeneral  # noqa: E402


def test_general_requires_confirm_after_project_name_changes(qapp, monkeypatch, tmp_path):
    controller = WizardController()
    page = PageGeneral(controller)
    page.basedir_input.set_path(str(tmp_path))
    page.basename_input.setText("demo")
    warnings = []
    monkeypatch.setattr(
        "openbench.gui.validation.QMessageBox.warning",
        lambda _parent, title, message: warnings.append((title, message)),
    )

    assert page.validate() is False
    assert "click Confirm" in warnings[0][1]

    controller.sync_namelists = lambda: None
    monkeypatch.setattr("openbench.gui.pages.page_general.QMessageBox.information", lambda *_args: None)
    page._on_confirm_project()
    assert page.validate() is True

    page.basename_input.setText("renamed")
    assert page.validate() is False
