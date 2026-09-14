from openbench.runner.postprocessing import _GRID_ONLY_COMPARISONS, _grid_only_evaluation_items
import logging


def test_grid_only_comparison_skips_station_involved_items(caplog):
    caplog.set_level(logging.INFO)
    simulation = {
        "general": {"Grid_sim_source": "GridSim", "Mixed_sim_source": "StationSim"},
        "Grid": {"GridSim_data_type": "grid"},
        "Mixed": {"StationSim_data_type": "stn"},
    }
    reference = {
        "general": {"Grid_ref_source": "GridRef", "Mixed_ref_source": "GridRef"},
        "Grid": {"GridRef_data_type": "grid"},
        "Mixed": {"GridRef_data_type": "grid"},
    }

    items = _grid_only_evaluation_items(
        "Mean", ["Grid", "Mixed"], simulation, reference
    )

    assert items == ["Grid"]
    assert "Skipping Mean for Mixed" in caplog.text


def test_grid_only_comparisons_include_tail_statistics():
    assert {"Mann_Kendall_Trend_Test", "Standard_Deviation"} <= _GRID_ONLY_COMPARISONS
