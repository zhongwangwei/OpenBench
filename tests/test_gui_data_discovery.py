from PySide6.QtCore import Qt

from openbench.data.registry.scanner import DatasetGroup, ScannedDataset
from openbench.data.registry.schema import ReferenceDataset
from openbench.gui.dialogs.data_discovery import DataDiscoveryDialog, ResolutionPickerDialog
from openbench.gui.localization import CHINESE, LanguageManager


def test_resolution_picker_accepts_registry_reference_dataset(qapp):
    """Registry variants lack scan-only file_count metadata but must still render."""
    ref = ReferenceDataset(
        name="Demo_LowRes",
        description="demo",
        category="Water",
        data_type="grid",
        tim_res="Month",
        data_groupby="Month",
        timezone=0,
        years=[2000, 2001],
        variables={},
        grid_res=1.0,
    )

    dlg = ResolutionPickerDialog("Demo", {"LowRes": ref})

    assert dlg.selected_resolution() is None


def test_discovery_dialog_defaults_to_new_datasets_only(qapp):
    existing = ScannedDataset("Existing", "LowRes", "Water", "grid", "/ref", {"Runoff": "existing"})
    new = ScannedDataset("New", "LowRes", "Water", "grid", "/ref", {"Runoff": "new"})
    groups = [
        DatasetGroup("Existing", {"LowRes": existing}),
        DatasetGroup("New", {"LowRes": new}),
    ]

    dialog = DataDiscoveryDialog(groups, existing_names={existing.registry_name})

    assert dialog.tree.headerItem().text(2) == "Status"
    assert dialog.tree.topLevelItem(0).child(0).text(2) == "Registered"
    assert dialog.tree.topLevelItem(0).child(0).checkState(0) == Qt.Unchecked
    assert dialog.tree.topLevelItem(1).child(0).text(2) == "Unregistered"
    assert [item[2].registry_name for item in dialog.get_selected()] == [new.registry_name]

    manager = LanguageManager(persist=False)
    manager.language = CHINESE
    manager.apply(dialog.tree)
    assert dialog.tree.headerItem().text(4) == "变量"
    assert dialog.tree.topLevelItem(1).child(0).text(2) == "未注册"
