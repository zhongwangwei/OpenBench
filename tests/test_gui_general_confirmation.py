import pytest

pytest.importorskip("PySide6")

from openbench.gui.controller import WizardController  # noqa: E402
from openbench.gui.pages.page_general import PageGeneral  # noqa: E402


@pytest.mark.parametrize("threshold", [None, 0.0, 2.35, 10.0])
def test_general_hii_threshold_load_save_and_disable(qapp, tmp_path, threshold):
    controller = WizardController()
    controller.config["general"].update(basename="hii", basedir=str(tmp_path), streamflow_max_hii=threshold)
    page = PageGeneral(controller)
    page.load_from_config()

    assert page.cb_streamflow_hii.isChecked() == (threshold is not None)
    assert page.streamflow_max_hii_spin.isEnabled() == (threshold is not None)
    if threshold is not None:
        assert page.streamflow_max_hii_spin.value() == threshold
    page._save_to_config_no_sync()
    assert controller.config["general"]["streamflow_max_hii"] == threshold

    page.cb_streamflow_hii.setChecked(True)
    page.streamflow_max_hii_spin.setValue(0.0)
    page._save_to_config_no_sync()
    assert controller.config["general"]["streamflow_max_hii"] == 0.0
    page.cb_streamflow_hii.setChecked(False)
    page._save_to_config_no_sync()
    assert not page.streamflow_max_hii_spin.isEnabled()
    assert controller.config["general"]["streamflow_max_hii"] is None


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
