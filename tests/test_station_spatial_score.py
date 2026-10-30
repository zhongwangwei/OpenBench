from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from openbench.core.evaluation import Evaluation_stn
from openbench.core.scores import scores


def _station_evaluator(tmp_path, score_names, station_values, *, num_cores=1):
    stations = pd.DataFrame(
        {
            "ID": list(station_values),
            "sim_lon": np.arange(len(station_values), dtype=float),
            "sim_lat": np.arange(len(station_values), dtype=float),
            "use_syear": 2000,
            "use_eyear": 2001,
        }
    )
    stations.to_csv(tmp_path / "stn_Ref_Sim_list.txt", index=False)
    data_dir = tmp_path / "data" / "stn_Ref_Sim"
    data_dir.mkdir(parents=True)
    time = pd.date_range("2000-01-01", periods=24, freq="MS")
    seasonal = np.tile(np.arange(12, dtype=float), 2)
    interannual = np.repeat([0.0, 1.0], 12)
    for station_id, (ref_level, sim_level) in station_values.items():
        ref = ref_level + seasonal + interannual
        sim = sim_level + seasonal * 1.1 + interannual * 0.8
        xr.Dataset({"flow": ("time", sim)}, coords={"time": time}).to_netcdf(
            data_dir / f"Runoff_sim_{station_id}_2000_2001.nc"
        )
        xr.Dataset({"flow": ("time", ref)}, coords={"time": time}).to_netcdf(
            data_dir / f"Runoff_ref_{station_id}_2000_2001.nc"
        )

    evaluator = Evaluation_stn.__new__(Evaluation_stn)
    evaluator.casedir = str(tmp_path)
    evaluator.item = "Runoff"
    evaluator.ref_source = "Ref"
    evaluator.sim_source = "Sim"
    evaluator.ref_varname = ["flow"]
    evaluator.sim_varname = ["flow"]
    evaluator.compare_tim_res = "Month"
    evaluator.metrics = []
    evaluator.scores = list(score_names)
    evaluator.num_cores = num_cores
    evaluator.output_manager = None
    evaluator.station_plots = False
    return evaluator


def _run_and_read(evaluator, tmp_path, monkeypatch):
    monkeypatch.setattr("openbench.core.evaluation.make_plot_index_stn", lambda _self: None)
    evaluator.make_evaluation_P()
    return pd.read_csv(tmp_path / "scores" / "Runoff_stn_Ref_Sim_evaluations.csv")


def test_station_spatial_score_is_one_network_value_and_updates_overall(tmp_path, monkeypatch):
    values = {"A": (10.0, 12.0), "B": (20.0, 18.0), "C": (30.0, 33.0)}
    evaluator = _station_evaluator(tmp_path, ["nSpatialScore", "Overall_Score"], values)

    frame = _run_and_read(evaluator, tmp_path, monkeypatch)

    ref_means = xr.DataArray([level + 6.0 for level, _ in values.values()], dims="station")
    # mean(seasonal * 1.1) + mean(interannual * 0.8) = 5.5*1.1 + 0.5*0.8
    sim_means = xr.DataArray([level + 6.45 for _, level in values.values()], dims="station")
    expected_spatial = float(scores._spatial_score_from_means(sim_means, ref_means, dim="station"))
    np.testing.assert_allclose(frame["nSpatialScore"], expected_spatial)
    assert frame["nSpatialScore"].nunique() == 1

    for row_index, station_id in enumerate(values):
        worker_row = evaluator.make_evaluation_parallel(
            pd.DataFrame(
                {
                    "ID": [station_id],
                    "use_syear": [2000],
                    "use_eyear": [2001],
                    "sim_lat": [0.0],
                    "sim_lon": [0.0],
                }
            ),
            0,
        )
        expected_overall = (
            worker_row["_overall_nonspatial_numerator"] + expected_spatial
        ) / (worker_row["_overall_nonspatial_denominator"] + 1.0)
        assert frame.loc[row_index, "Overall_Score"] == pytest.approx(expected_overall)

    assert not any(column.startswith("_") for column in frame.columns)


def test_overall_implicitly_uses_spatial_score_without_output_column(tmp_path, monkeypatch):
    values = {"A": (10.0, 12.0), "B": (20.0, 18.0), "C": (30.0, 33.0)}
    evaluator = _station_evaluator(tmp_path, ["Overall_Score"], values)

    frame = _run_and_read(evaluator, tmp_path, monkeypatch)

    assert "nSpatialScore" not in frame.columns
    assert np.isfinite(frame["Overall_Score"]).all()


def test_unavailable_station_is_excluded_from_network_score(tmp_path, monkeypatch):
    values = {"A": (10.0, 12.0), "B": (20.0, 18.0), "C": (30.0, 33.0)}
    evaluator = _station_evaluator(tmp_path, ["nSpatialScore"], values)
    missing_path = tmp_path / "data" / "stn_Ref_Sim" / "Runoff_ref_C_2000_2001.nc"
    with xr.open_dataset(missing_path) as dataset:
        missing = dataset.load()
    missing["flow"][:] = np.nan
    missing.to_netcdf(missing_path, mode="w")

    frame = _run_and_read(evaluator, tmp_path, monkeypatch)

    expected = float(
        scores._spatial_score_from_means(
            xr.DataArray([12.0 + 6.45, 18.0 + 6.45], dims="station"),
            xr.DataArray([10.0 + 6.0, 20.0 + 6.0], dims="station"),
            dim="station",
        )
    )
    assert frame["ID"].tolist() == ["A", "B"]
    np.testing.assert_allclose(frame["nSpatialScore"], expected)


def test_explicit_station_spatial_score_reports_degenerate_network(tmp_path, monkeypatch):
    evaluator = _station_evaluator(tmp_path, ["nSpatialScore"], {"A": (10.0, 12.0)})
    monkeypatch.setattr("openbench.core.evaluation.make_plot_index_stn", lambda _self: None)

    with pytest.raises(RuntimeError, match="at least two stations.*non-zero"):
        evaluator.make_evaluation_P()


@pytest.mark.parametrize(
    "values",
    [
        {"A": (10.0, 12.0), "B": (10.0, 18.0)},  # zero reference spatial variance
        {"A": (10.0, 12.0), "B": (20.0, 12.0)},  # zero simulation spatial variance
    ],
)
def test_explicit_station_spatial_score_rejects_zero_spatial_variance(tmp_path, monkeypatch, values):
    evaluator = _station_evaluator(tmp_path, ["nSpatialScore"], values)
    monkeypatch.setattr("openbench.core.evaluation.make_plot_index_stn", lambda _self: None)

    with pytest.raises(RuntimeError, match="non-zero simulated and reference spatial standard deviations"):
        evaluator.make_evaluation_P()


def test_explicit_station_spatial_score_reports_all_stations_missing(tmp_path, monkeypatch):
    evaluator = _station_evaluator(tmp_path, ["nSpatialScore"], {"A": (10.0, 12.0), "B": (20.0, 18.0)})
    evaluator.make_evaluation_parallel = lambda _stations, _index: {"_skip_reason": "no shared finite pairs"}
    monkeypatch.setattr("openbench.core.evaluation.make_plot_index_stn", lambda _self: None)

    with pytest.raises(RuntimeError, match="no valid stations remain"):
        evaluator.make_evaluation_P()


def test_overall_skips_undefined_implicit_spatial_component(tmp_path, monkeypatch):
    evaluator = _station_evaluator(tmp_path, ["Overall_Score"], {"A": (10.0, 12.0)})

    frame = _run_and_read(evaluator, tmp_path, monkeypatch)

    worker = evaluator.make_evaluation_parallel(
        pd.DataFrame(
            {"ID": ["A"], "use_syear": [2000], "use_eyear": [2001], "sim_lat": [0.0], "sim_lon": [0.0]}
        ),
        0,
    )
    expected = worker["_overall_nonspatial_numerator"] / worker["_overall_nonspatial_denominator"]
    assert frame.loc[0, "Overall_Score"] == pytest.approx(expected)


def test_station_spatial_serial_and_parallel_results_match(tmp_path, monkeypatch):
    values = {"A": (10.0, 12.0), "B": (20.0, 18.0), "C": (30.0, 33.0)}
    serial_dir = tmp_path / "serial"
    parallel_dir = tmp_path / "parallel"
    serial_dir.mkdir()
    parallel_dir.mkdir()
    serial = _station_evaluator(serial_dir, ["nSpatialScore", "Overall_Score"], values, num_cores=1)
    parallel = _station_evaluator(parallel_dir, ["nSpatialScore", "Overall_Score"], values, num_cores=2)

    monkeypatch.setattr("openbench.core.evaluation._HAS_PARALLEL_ENGINE", True)
    monkeypatch.setattr(
        "openbench.core.evaluation.parallel_map",
        lambda func, items, **_kwargs: [func(index) for index in items],
    )
    serial_frame = _run_and_read(serial, serial_dir, monkeypatch)
    parallel_frame = _run_and_read(parallel, parallel_dir, monkeypatch)

    pd.testing.assert_frame_equal(serial_frame, parallel_frame)
