"""CSV contracts, weighted reductions and sample-count validation for LC/CZ."""

import importlib
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr


@pytest.fixture
def groupby_case(tmp_path, monkeypatch):
    def build(group, weight="none", mismatch=False, kind="metric", different_masks=False):
        module = importlib.import_module(
            "openbench.core.climatezone_groupby" if group == "CZ" else "openbench.core.landcover_groupby"
        )
        # Remapping itself is outside this change; supply a known target-grid map.
        monkeypatch.setitem(
            sys.modules,
            "openbench.data.regrid",
            SimpleNamespace(Grid=lambda **kwargs: None, create_regridding_dataset=lambda grid: None),
        )
        monkeypatch.setattr(
            xr.DataArray,
            "regrid",
            property(lambda data: SimpleNamespace(most_common=lambda *args, **kwargs: data.to_dataset())),
            raising=False,
        )
        coords = {"lat": [0.0, 60.0], "lon": np.arange(20)}
        start = 0 if group == "PFT" else 1
        classes = np.array([[start] * 20, [start + 1] * 19 + [99]])
        variable = "climate_zone" if group == "CZ" else group
        static = tmp_path / f"{group}_static.nc"
        xr.Dataset({variable: (("lat", "lon"), classes)}, coords=coords).to_netcdf(static)
        monkeypatch.setattr(module, "static_dataset_path", lambda name: nullcontext(static))
        monkeypatch.setattr(module, "_open_dataset_safe", lambda path, **kwargs: xr.open_dataset(path))
        monkeypatch.setattr(
            module, "make_CZ_based_heat_map" if group == "CZ" else "make_LC_based_heat_map", lambda *args: None
        )
        values = np.arange(40, dtype=float).reshape(2, 20)
        # Monotonic statistics share their classwise and global clipping masks.
        statistics = ["bias", "RMSE"] if kind == "metric" else ["nBiasScore", "nRMSEScore"]
        folder = tmp_path / ("metrics" if kind == "metric" else "scores")
        folder.mkdir(exist_ok=True)
        for index, statistic in enumerate(statistics):
            data = values.copy() + index
            if different_masks and index:
                data[0] = data[0, ::-1]
            if mismatch and index:
                data[0, 10] = np.nan
            xr.Dataset({statistic: (("lat", "lon"), data)}, coords=coords).to_netcdf(
                folder / f"Evapotranspiration_ref_Ref_sim_Sim_{statistic}.nc"
            )
        ref = xr.DataArray(np.ones((2, 2, 20)), dims=("time", "lat", "lon"), coords={**coords, "time": [0, 1]})
        ref[:, 0, 0] = 0
        ref[:, 1, 0] = np.nan
        (tmp_path / "data").mkdir(exist_ok=True)
        ref.to_dataset(name="et").to_netcdf(tmp_path / "data/Evapotranspiration_ref_Ref_et.nc")
        main = {
            "general": dict(
                basedir=str(tmp_path.parent),
                basename=tmp_path.name,
                compare_grid_res=1,
                compare_tim_res="month",
                weight=weight,
                min_lat=-90,
                max_lat=90,
                min_lon=-180,
                max_lon=180,
            )
        }
        ref_nml = {
            "general": {"Evapotranspiration_ref_source": "Ref"},
            "Evapotranspiration": {"Ref_data_type": "grid", "Ref_varname": "et", "Ref_varunit": "mm day-1"},
        }
        sim_nml = {
            "general": {"Evapotranspiration_sim_source": "Sim"},
            "Evapotranspiration": {"Sim_data_type": "grid", "Sim_varunit": "mm day-1"},
        }
        metrics = statistics if kind == "metric" else []
        scores = statistics if kind == "score" else []
        handler = (module.CZ_groupby if group == "CZ" else module.LC_groupby)(main, scores, metrics)

        def run():
            getattr(handler, f"scenarios_{group}_groupby_comparison")(
                str(tmp_path), sim_nml, ref_nml, ["Evapotranspiration"], scores, metrics, {}
            )

        path = tmp_path / f"comparisons/{group}_groupby/Sim__Ref/Evapotranspiration__Sim__Ref__{kind}s.csv"
        return SimpleNamespace(
            run=run,
            path=path,
            values=values,
            classes=classes,
            start=start,
            main=main,
            sim=sim_nml,
            ref=ref_nml,
            metrics=metrics,
            scores=scores,
            reference=ref,
            coords=coords,
        )

    return build


@pytest.mark.parametrize("group", ["IGBP", "PFT", "CZ"])
@pytest.mark.parametrize("kind,weight", [("metric", "none"), ("score", "none"), ("score", "area"), ("score", "mass")])
def test_producer_metadata_counts_and_values(groupby_case, group, kind, weight):
    from openbench.visualization.Fig_LC_based_heat_map import _read_metrics_file

    case = groupby_case(group, weight, kind=kind)
    case.run()
    assert case.path.exists()  # Exact existing directory and filename contract.
    text = case.path.read_text(encoding="utf-8")
    assert len([line for line in text.splitlines() if line.startswith("#")]) == 5
    assert text.splitlines()[-1].startswith("n_valid\t")
    frame = _read_metrics_file(str(case.path))
    assert list(frame.index) == (case.metrics or case.scores)
    metadata = frame.attrs["metadata"]
    assert metadata["statistic_type"] == kind
    assert metadata["ref_unit"] == metadata["sim_unit"] == "mm day-1"
    assert metadata["weight"] == weight
    method = {"none": "unweighted", "area": "area_weighted", "mass": "mass_weighted"}[weight]
    assert metadata["aggregation"] == (
        "gridcell_metric -> classwise_clipped_median"
        if kind == "metric"
        else f"gridcell_score -> classwise_{method}_mean"
    )
    for column_index in (0, 1, len(frame.columns) - 1):
        selected = (
            np.ones_like(case.values, dtype=bool)
            if column_index == len(frame.columns) - 1
            else (case.classes == case.start + column_index)
        )
        data = case.values[selected]
        if kind == "metric":
            if len(data) >= 20:
                low, high = np.quantile(data, [0.05, 0.95])
                data = data[(data >= low) & (data <= high)]
            expected = np.median(data)
            count = len(data)
        else:
            weights = np.ones_like(case.values)
            if weight != "none":
                weights *= np.cos(np.deg2rad(np.array(case.coords["lat"])))[:, None]
            if weight == "mass":
                weights *= np.nan_to_num(case.reference.mean("time").values)
            expected = np.average(data, weights=weights[selected])
            count = np.count_nonzero(weights[selected])
        assert frame.iloc[0, column_index] == pytest.approx(expected, abs=0.00051)
        assert frame.attrs["n_valid"][frame.columns[column_index]] == count


@pytest.mark.parametrize("group", ["IGBP", "PFT", "CZ"])
@pytest.mark.parametrize("kind", ["metric", "score"])
def test_mismatch_stops_csv(groupby_case, group, kind):
    case = groupby_case(group, mismatch=True, kind=kind)
    with pytest.raises(ValueError, match="inconsistent n_valid.*CSV generation stopped"):
        case.run()
    assert not case.path.exists()


def test_legacy_reader(tmp_path):
    from openbench.visualization.Fig_LC_based_heat_map import _read_metrics_file

    path = tmp_path / "legacy.csv"
    path.write_text("metric\tENF\tOverall\nbias\t1.2\t2.3\n")
    frame = _read_metrics_file(str(path))
    assert frame.loc["bias", "ENF"] == 1.2
    assert frame.attrs == {"metadata": {}, "n_valid": {}}


@pytest.mark.parametrize("group", ["IGBP", "PFT", "CZ"])
def test_metric_different_clipped_cells_same_counts_write_csv(groupby_case, group):
    from openbench.visualization.Fig_LC_based_heat_map import _read_metrics_file

    case = groupby_case(group, different_masks=True)
    case.run()
    frame = _read_metrics_file(str(case.path))
    assert frame.attrs["n_valid"][frame.columns[0]] == 18
    assert frame.attrs["n_valid"][frame.columns[1]] == 19
    assert frame.attrs["n_valid"]["Overall"] == 36
