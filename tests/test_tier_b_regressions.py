import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr


def _make_processing_processor(processing_module):
    processor = object.__new__(processing_module.DatasetProcessing)
    processor.compare_grid_res = 1.0
    processor.compare_tim_res = "D"
    processor.sim_source = "SimA"
    processor.ref_source = "RefA"
    processor.sim_varname = ["runoff"]
    processor.ref_varname = ["runoff"]
    processor.sim_varunit = None
    processor.ref_varunit = None
    return processor


def test_merged_station_parser_rejects_2d_grid_coordinates(tmp_path, caplog):
    from openbench.data.registry.scanner import _parse_merged_station_file

    nc_path = tmp_path / "merged.nc"
    ds = xr.Dataset(
        {
            "station_id": ("station", np.array(["A", "B"], dtype=object)),
            "lat": (("y", "x"), [[10.0, 10.5], [11.0, 11.5]]),
            "lon": (("y", "x"), [[100.0, 100.5], [101.0, 101.5]]),
            "flow": (("time", "station"), [[1.0, 2.0]]),
        },
        coords={"time": [np.datetime64("2000-01-01")], "station": [0, 1], "y": [0, 1], "x": [0, 1]},
    )
    ds.to_netcdf(nc_path)

    with caplog.at_level(logging.WARNING):
        assert _parse_merged_station_file(nc_path, tmp_path) == []

    assert "dims ('y', 'x') are not indexed by station dimension station" in caplog.text


def test_merged_station_parser_warns_for_non_scalar_station_coordinate(tmp_path, caplog):
    from openbench.data.registry.scanner import _parse_merged_station_file

    nc_path = tmp_path / "merged.nc"
    ds = xr.Dataset(
        {
            "station_id": ("station", np.array(["A", "B"], dtype=object)),
            "lat": (("station", "sample"), [[10.0, 10.5], [11.0, 11.5]]),
            "lon": ("station", [100.0, 101.0]),
            "flow": (("time", "station"), [[1.0, 2.0]]),
        },
        coords={"time": [np.datetime64("2000-01-01")], "station": [0, 1], "sample": [0, 1]},
    )
    ds.to_netcdf(nc_path)

    with caplog.at_level(logging.WARNING):
        assert _parse_merged_station_file(nc_path, tmp_path) == []

    assert "Skipping merged station coordinate lat: station slice is not scalar" in caplog.text


def test_merged_station_parser_prefers_station_dimension_over_bounds_dimension(tmp_path):
    from openbench.data.registry.scanner import _parse_merged_station_file

    nc_path = tmp_path / "merged.nc"
    ds = xr.Dataset(
        {
            "station_id": ("station", np.array(["A", "B"], dtype=object)),
            "lat": ("station", [10.0, 11.0]),
            "lon": ("station", [100.0, 101.0]),
            "time_bounds": (("time", "nv"), np.zeros((2, 2))),
            "flow": (("station", "time"), [[1.0, 2.0], [3.0, 4.0]]),
        },
        coords={"time": pd.date_range("2000-01-01", periods=2), "nv": [0, 1], "station": [0, 1]},
    )
    ds.to_netcdf(nc_path)

    rows = _parse_merged_station_file(nc_path, tmp_path)

    assert [row[0] for row in rows] == ["A", "B"]
    assert rows[0][1:3] == [2000, 2000]


def test_merged_station_parser_detects_time_when_station_dimension_is_first(tmp_path):
    from openbench.data.registry.scanner import _parse_merged_station_file

    nc_path = tmp_path / "merged.nc"
    ds = xr.Dataset(
        {
            "station_id": ("station", np.array(["A"], dtype=object)),
            "lat": ("station", [10.0]),
            "lon": ("station", [100.0]),
            "flow": (("station", "time"), [[1.0, 2.0]]),
        },
        coords={"station": [0], "time": pd.date_range("2001-01-01", periods=2)},
    )
    ds.to_netcdf(nc_path)

    rows = _parse_merged_station_file(nc_path, tmp_path)

    assert rows[0][1:3] == [2001, 2001]


def test_single_station_parser_warns_when_file_parse_fails(tmp_path, caplog):
    from openbench.data.registry.scanner import _parse_single_station_file

    nc_path = tmp_path / "not-netcdf.nc"
    nc_path.write_text("not a netcdf file", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        assert _parse_single_station_file(nc_path) is None

    assert "Failed to parse station file not-netcdf.nc" in caplog.text


def test_merged_station_parser_warns_when_file_parse_fails(tmp_path, caplog):
    from openbench.data.registry.scanner import _parse_merged_station_file

    nc_path = tmp_path / "not-netcdf.nc"
    nc_path.write_text("not a netcdf file", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        assert _parse_merged_station_file(nc_path, tmp_path) == []

    assert "Failed to parse merged station file not-netcdf.nc" in caplog.text


def test_tch_uncertainty_matches_classical_three_cornered_hat_formula():
    from openbench.core.statistics.stat_three_cornered_hat import _tch_uncertainty_from_samples

    # Three orthogonal zero-mean columns, scaled so their sample variances are
    # exactly 1, 4, and 9.  Pairwise-difference variances are therefore
    # v_i + v_j, making the classical 3CH closed form exact.
    orthogonal = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ]
    )
    samples = orthogonal * np.array([1.0, 2.0, 3.0]) / np.sqrt(4.0 / 3.0)

    uncertainty, relative = _tch_uncertainty_from_samples(samples)

    np.testing.assert_allclose(uncertainty, [1.0, 2.0, 3.0], rtol=1e-12, atol=1e-12)
    assert np.isfinite(relative).all()


def test_tch_statistic_returns_finite_values_for_independent_sources():
    from openbench.core.statistics.stat_three_cornered_hat import stat_three_cornered_hat

    orthogonal = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ]
    )
    samples = orthogonal * np.array([1.0, 2.0, 3.0]) / np.sqrt(4.0 / 3.0)
    arrays = [
        xr.DataArray(
            samples[:, i, None, None],
            dims=("time", "lat", "lon"),
            coords={"time": np.arange(samples.shape[0]), "lat": [0.0], "lon": [0.0]},
            name="v",
        )
        for i in range(samples.shape[1])
    ]

    result = stat_three_cornered_hat(None, *arrays)

    np.testing.assert_allclose(result["uncertainty"].values[:, 0, 0], [1.0, 2.0, 3.0], rtol=1e-12, atol=1e-12)


def test_check_time_repairs_unparseable_time_without_dropping_level_dimension():
    from openbench.data.processing import DatasetProcessing

    processor = object.__new__(DatasetProcessing)
    data = xr.DataArray(
        np.ones((1, 2, 1, 1)),
        dims=("time", "level", "lat", "lon"),
        coords={"time": ["not-a-time"], "level": [850, 500], "lat": [0.0], "lon": [10.0]},
        name="air",
    )

    out = processor.check_time(data, 2000, 2000, "YE")

    assert out.dims == ("time", "level", "lat", "lon")
    assert out.sizes["level"] == 2
    assert np.issubdtype(out["time"].dtype, np.datetime64)


def test_grid_fallback_conversion_failure_is_fatal(monkeypatch):
    import openbench.data.processing as processing

    processor = _make_processing_processor(processing)
    processor._fb_convert_sim = "missing_name + 1"
    ds = xr.Dataset({"runoff": ("time", [1.0])}, coords={"time": [np.datetime64("2000-01-01")]})

    monkeypatch.setattr(processing.xr, "open_dataset", lambda *args, **kwargs: ds.copy())
    monkeypatch.setattr(processor, "apply_custom_filter", lambda datasource, opened_ds, varname: opened_ds[varname[0]])

    with pytest.raises(RuntimeError, match="Fallback conversion .* failed"):
        processing.BaseDatasetProcessing.select_var(processor, 2000, 2000, "D", "dummy.nc", ["runoff"], "sim")


def test_station_fallback_conversion_failure_is_fatal():
    import openbench.data.processing as processing

    processor = _make_processing_processor(processing)
    processor._fb_convert_sim = "missing_name + 1"
    ds = xr.Dataset({"runoff": ("time", [1.0])}, coords={"time": [np.datetime64("2000-01-01")]})

    with pytest.raises(RuntimeError, match="Station fallback conversion .* failed"):
        processing.DatasetProcessing.process_single_station_data(processor, ds, 2000, 2000, "sim")


def test_merged_station_processing_selects_requested_station_by_id():
    import openbench.data.processing as processing

    processor = _make_processing_processor(processing)
    ds = xr.Dataset(
        {
            "runoff": (("station", "time"), [[1.0, 2.0], [3.0, 4.0]]),
            "station_id": ("station", np.array(["A", "B"], dtype=object)),
            "lat": ("station", [10.0, 20.0]),
            "lon": ("station", [100.0, 110.0]),
        },
        coords={"station": [0, 1], "time": pd.date_range("2000-01-01", periods=2)},
    )
    station = pd.Series({"ID": "B"})

    selected = processing.DatasetProcessing._select_merged_station_data(processor, ds, station, "ref")

    assert selected["runoff"].dims == ("time",)
    assert selected["runoff"].values.tolist() == [3.0, 4.0]
    assert float(selected["lat"]) == 20.0
    assert float(selected["lon"]) == 110.0


def test_merged_station_processing_raises_when_station_id_is_absent():
    import openbench.data.processing as processing

    processor = _make_processing_processor(processing)
    ds = xr.Dataset(
        {
            "runoff": (("station", "time"), [[1.0, 2.0], [3.0, 4.0]]),
            "station_id": ("station", np.array(["A", "B"], dtype=object)),
        },
        coords={"station": [0, 1], "time": pd.date_range("2000-01-01", periods=2)},
    )
    station = pd.Series({"ID": "C"})

    with pytest.raises(ValueError, match="Station C not found"):
        processing.DatasetProcessing._select_merged_station_data(processor, ds, station, "ref")


def test_longitude_normalization_removes_duplicate_seam_cells():
    from openbench.data.processing import DatasetProcessing

    processor = object.__new__(DatasetProcessing)
    ds = xr.Dataset({"v": ("lon", [1.0, 2.0])}, coords={"lon": [0.0, 360.0]})

    out = processor._normalize_longitude_axis(ds)

    assert out.sizes["lon"] == 1
    assert out["lon"].values.tolist() == [0.0]


def test_process_extracted_data_rejects_monthly_to_daily_upsampling():
    import openbench.data.processing as processing

    processor = _make_processing_processor(processing)
    data = xr.DataArray(
        [1.0, 2.0],
        dims=("time",),
        coords={"time": pd.to_datetime(["2000-01-15", "2000-02-15"])},
        name="runoff",
    )

    with pytest.raises(ValueError, match="refusing to upsample"):
        processing.DatasetProcessing.process_extracted_data(processor, data, 2000, 2000)


def test_station_nearest_selection_requires_tolerance():
    import openbench.data.processing as processing

    processor = _make_processing_processor(processing)
    dataset = xr.Dataset(
        {"runoff": (("lat", "lon"), [[1.0]])},
        coords={"lat": [0.0], "lon": [0.0]},
    )
    station = pd.Series({"ID": "offshore", "ref_lat": 10.0, "ref_lon": 10.0})

    with pytest.raises(ValueError, match="outside tolerance"):
        processing.DatasetProcessing.extract_single_station_data(processor, dataset, station, "sim")


def test_comparison_pairwise_mask_handles_integer_arrays():
    from openbench.core._comparison_helpers import _apply_pairwise_valid_mask

    s = xr.DataArray(np.array([1, 2], dtype=np.int16), dims=("time",))
    o = xr.DataArray(np.array([1.0, np.nan]), dims=("time",))

    masked_s, masked_o = _apply_pairwise_valid_mask(s, o)

    assert np.issubdtype(masked_s.dtype, np.floating)
    assert np.isnan(masked_s.values[1])
    assert np.isnan(masked_o.values[1])


def test_missing_comparison_stat_method_fails_clearly():
    from openbench.core._comparison_helpers import _require_stat_method

    with pytest.raises(AttributeError, match="stat_not_a_stat"):
        _require_stat_method(object(), "Not_A_Stat")


def test_correlation_does_not_mutate_empty_sim_varnames(tmp_path, monkeypatch):
    import openbench.core.comparison as comparison

    case_dir = tmp_path / "case"
    data_dir = case_dir / "data"
    data_dir.mkdir(parents=True)
    arr = xr.DataArray([[1.0]], dims=("lat", "lon"), coords={"lat": [0.0], "lon": [0.0]}, name="Runoff")
    arr.to_dataset().to_netcdf(data_dir / "Runoff_sim_Sim1_Runoff.nc")
    arr.to_dataset().to_netcdf(data_dir / "Runoff_sim_Sim2_Runoff.nc")

    main_nml = {
        "general": {"basedir": str(tmp_path), "basename": "case", "compare_grid_res": 1.0, "compare_tim_res": "Month"}
    }
    sim_nml = {
        "general": {"Runoff_sim_source": ["Sim1", "Sim2"]},
        "Runoff": {
            "Sim1_varname": "",
            "Sim2_varname": None,
            "Sim1_data_type": "grid",
            "Sim2_data_type": "grid",
        },
    }
    processor = comparison.ComparisonProcessing(main_nml, scores=[], metrics=[])
    processor.stat_correlation = lambda ds1, ds2: arr
    processor.save_result = lambda *args, **kwargs: None
    monkeypatch.setattr(comparison, "make_Correlation", lambda *args, **kwargs: None)

    processor.scenarios_Correlation_comparison(str(case_dir), sim_nml, {}, ["Runoff"], [], [], {})

    assert sim_nml["Runoff"]["Sim1_varname"] == ""
    assert sim_nml["Runoff"]["Sim2_varname"] is None


def test_mann_kendall_logs_station_skips(tmp_path, caplog):
    from openbench.core.comparison import ComparisonProcessing

    main_nml = {
        "general": {"basedir": str(tmp_path), "basename": "case", "compare_grid_res": 1.0, "compare_tim_res": "Month"}
    }
    sim_nml = {
        "general": {"Runoff_sim_source": ["SimA"]},
        "Runoff": {"SimA_varname": "runoff", "SimA_data_type": "stn"},
    }
    ref_nml = {
        "general": {"Runoff_ref_source": "RefA"},
        "Runoff": {"RefA_varname": "runoff", "RefA_data_type": "stn"},
    }
    processor = ComparisonProcessing(main_nml, scores=[], metrics=[])
    processor.stat_mann_kendall_trend_test = lambda ds: ds

    caplog.set_level(logging.INFO)
    processor.scenarios_Mann_Kendall_Trend_Test_comparison(
        str(tmp_path / "case"), sim_nml, ref_nml, ["Runoff"], [], [], {"significance_level": 0.05}
    )

    assert "Skipping Mann_Kendall_Trend_Test" in caplog.text


def test_climatology_unsupported_metric_raises(tmp_path, monkeypatch):
    import openbench.core.evaluation as evaluation

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    coords = {"time": pd.date_range("2000-01-01", periods=12, freq="ME"), "lat": [0.0], "lon": [0.0]}
    xr.Dataset({"ref": (("time", "lat", "lon"), np.ones((12, 1, 1)))}, coords=coords).to_netcdf(
        data_dir / "Runoff_ref_RefA_ref.nc"
    )
    xr.Dataset({"sim": (("time", "lat", "lon"), np.ones((12, 1, 1)))}, coords=coords).to_netcdf(
        data_dir / "Runoff_sim_SimA_sim.nc"
    )

    def fake_climatology(ref_ds, sim_ds, evaluations, **kwargs):
        return ref_ds, sim_ds, ["bias"]

    monkeypatch.setattr(evaluation, "_HAS_CLIMATOLOGY", True)
    monkeypatch.setattr(evaluation, "process_climatology_evaluation", fake_climatology)

    info = {
        "casedir": str(tmp_path),
        "item": "Runoff",
        "ref_source": "RefA",
        "sim_source": "SimA",
        "ref_varname": "ref",
        "sim_varname": "sim",
        "metrics": ["bias", "RMSE"],
        "scores": [],
        "compare_tim_res": "climatology-month",
        "syear": 2000,
    }
    processor = evaluation.Evaluation_grid(info, fig_nml={})

    with pytest.raises(ValueError, match="Unsupported climatology metric"):
        processor.make_Evaluation()


def test_station_time_alignment_preserves_exact_common_times():
    from openbench.core.evaluation import Evaluation_stn

    processor = object.__new__(Evaluation_stn)
    processor.compare_tim_res = "Day"
    times = pd.to_datetime(["2000-01-01T00:00:00", "2000-01-02T00:00:00"])
    s = xr.DataArray([1.0, 2.0], dims=("time",), coords={"time": times})
    o = xr.DataArray([1.5, 2.5], dims=("time",), coords={"time": times})

    aligned_s, aligned_o = processor._align_station_times(s, o, "S1")

    np.testing.assert_array_equal(aligned_s["time"].values, times.values)
    np.testing.assert_array_equal(aligned_o["time"].values, times.values)


def test_station_single_variable_fallback_is_not_implicit():
    from openbench.core.evaluation import Evaluation_stn

    processor = object.__new__(Evaluation_stn)
    processor.ref_varname = ["typo"]
    processor.ref_source = "NoSuchRef"
    processor.sim_source = "SimA"
    dataset = xr.Dataset({"actual": ("time", [1.0])}, coords={"time": [np.datetime64("2000-01-01")]})

    with pytest.raises(KeyError):
        processor._load_station_dataset(dataset, "ref")


def test_station_plot_scalarizes_vector_metric_values(caplog):
    from openbench.core.evaluation import _scalar_plot_value

    caplog.set_level(logging.WARNING)

    value = _scalar_plot_value(np.array([1.0, 3.0]), label="RMSE", station_id="S1")

    assert value == 2.0
    assert "returned 2 values" in caplog.text


def test_parallel_coordinates_no_longer_drops_columns_for_single_nan():
    source = Path("src/openbench/core/_comparison_parallel.py").read_text(encoding="utf-8")

    assert 'df.dropna(axis=1, how="any")' not in source
    assert "all_missing_value_columns" in source


def test_relative_score_validity_does_not_use_id_column_only():
    source = Path("src/openbench/core/_comparison_relative.py").read_text(encoding="utf-8")

    assert "relative_score_columns" in source
    assert "if not combined_relative_scores.empty" not in source


def test_groupby_default_compare_tim_res_is_parseable_without_config_value(tmp_path):
    from openbench.core.climatezone_groupby import CZ_groupby
    from openbench.core.landcover_groupby import LC_groupby
    from openbench.core.statistics.Mod_Statistics import StatisticsProcessing

    main_nml = {"general": {"basedir": str(tmp_path), "basename": "case", "compare_grid_res": 1.0}}

    lc = LC_groupby(main_nml, scores=[], metrics=[])
    cz = CZ_groupby(main_nml, scores=[], metrics=[])
    stats = StatisticsProcessing(main_nml, {"general": {}}, output_dir=str(tmp_path), num_cores=1)

    assert lc.compare_tim_res == "month"
    assert cz.compare_tim_res == "month"
    assert stats.compare_tim_res == "1ME"
