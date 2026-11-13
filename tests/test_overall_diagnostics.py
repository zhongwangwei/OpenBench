import json
from types import SimpleNamespace

import numpy as np
import xarray as xr

from openbench.core._overall_diagnostics import write_diagnostics
from openbench.core.evaluation import Evaluation_grid
from openbench.core.scores import scores


def test_counts_weights_zero_missing_and_masks(tmp_path, caplog):
    evaluator = SimpleNamespace(casedir=tmp_path, item="Flood", ref_source="Ref", sim_source="Sim")
    # six cells with 0..5 components; the one-component cell is spatial-only.
    fields = [
        xr.DataArray([np.nan, np.nan, 0, 0, 0, 0], dims="cell"),
        xr.DataArray([np.nan, np.nan, 0, 0, 0, 0], dims="cell"),
        xr.DataArray([np.nan, np.nan, np.nan, 0, 0, 0], dims="cell"),
        xr.DataArray([np.nan, np.nan, np.nan, np.nan, 0, 0], dims="cell"),
        xr.DataArray([np.nan, 0, np.nan, np.nan, np.nan, 0], dims="cell"),
    ]
    overall = xr.DataArray([np.nan, np.nan, 0, 0, 0, 0], dims="cell")
    paired = xr.DataArray([False, True, True, True, True, True], dims="cell")
    summary = write_diagnostics(evaluator, fields, overall, paired)
    assert [summary["valid_component_counts"][str(i)]["count"] for i in range(6)] == [1] * 6
    assert summary["spatial_only"] == 1
    assert summary["finite_overall_incomplete"] == 3
    assert summary["paired_data_locations"] == 5
    target = next((tmp_path / "diagnostics/overall_score").iterdir())
    with xr.open_dataset(target / "component_validity.nc") as ds:
        np.testing.assert_array_equal(ds.valid_weight_sum, [0, 1, 3, 4, 5, 6])
        assert ds.nBiasScore_valid_mask[2] == 1
    assert json.loads((target / "summary.json").read_text()) == summary
    assert len([r for r in caplog.records if r.levelname == "WARNING"]) == 1


def test_empty_domain(tmp_path):
    evaluator = SimpleNamespace(casedir=tmp_path, item="Flood", ref_source="Ref", sim_source="Sim")
    empty = xr.DataArray(np.array([], dtype=float), dims="cell")
    summary = write_diagnostics(evaluator, [empty] * 5, empty, empty.astype(bool))
    assert summary["total_locations"] == 0
    assert summary["valid_component_counts"]["5"]["fraction"] == 0


def test_grid_integration_preserves_score_and_only_requested_diagnostics(tmp_path):
    evaluator = Evaluation_grid.__new__(Evaluation_grid)
    evaluator.casedir = str(tmp_path)
    evaluator.item, evaluator.ref_source, evaluator.sim_source = "Flood", "Ref", "Sim"
    evaluator.output_manager = None
    time = np.arange("2000-01", "2002-01", dtype="datetime64[M]")
    ref = xr.DataArray(
        np.stack([np.zeros(24), np.arange(24)], axis=1)[:, None, :],
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": [0], "lon": [0, 1]},
    )
    sim = ref + 1
    evaluator.process_score("nBiasScore", sim, ref)
    assert not (tmp_path / "diagnostics").exists()
    components = scores()._overall_nonspatial_components(sim, ref)
    supported = sum(np.isfinite(a).astype(int) for a, _ in components) > 0
    components.append((scores().nSpatialScore(sim, ref), 1))
    before = scores._weighted_score_components(components)[0].where(supported)
    evaluator.process_score("Overall_Score", sim, ref)
    with xr.open_dataset(next((tmp_path / "scores").glob("*Overall_Score.nc"))) as ds:
        np.testing.assert_array_equal(ds.Overall_Score, before)
    assert list((tmp_path / "diagnostics").rglob("summary.json"))
