from openbench.data.registry.schema import ReferenceDataset
from openbench.gui.dialogs.data_discovery import ResolutionPickerDialog


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
