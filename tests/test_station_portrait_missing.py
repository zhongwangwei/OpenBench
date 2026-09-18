import numpy as np
import pandas as pd
import pytest
import xarray as xr


def _processor(tmp_path, metrics):
    import openbench.core.comparison as comparison_module

    processor = comparison_module.ComparisonProcessing(
        {
            "general": {
                "basename": "case",
                "basedir": str(tmp_path),
                "compare_grid_res": 0.5,
                "compare_tim_res": "Month",
                "weight": "none",
                "num_cores": 1,
            }
        },
        [],
        metrics,
    )
    return processor, comparison_module


def _nmls():
    return (
        {
            "general": {"Runoff_sim_source": ["SimA"]},
            "Runoff": {"SimA_data_type": "stn", "SimA_varname": "runoff_sim"},
        },
        {
            "general": {"Runoff_ref_source": "RefA"},
            "Runoff": {"RefA_data_type": "stn", "RefA_varname": "runoff_ref"},
        },
    )


def _write_station_case(tmp_path, rows, datasets):
    metrics_dir = tmp_path / "metrics"
    data_dir = tmp_path / "data" / "stn_RefA_SimA"
    metrics_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(metrics_dir / "Runoff_stn_RefA_SimA_evaluations.csv", index=False)
    for station_id, (start, end, ref_values, sim_values, times) in datasets.items():
        coords = {"time": pd.DatetimeIndex(times)}
        xr.Dataset({"runoff_ref": ("time", ref_values)}, coords=coords).to_netcdf(
            data_dir / f"Runoff_ref_{station_id}_{start}_{end}.nc"
        )
        xr.Dataset({"runoff_sim": ("time", sim_values)}, coords=coords).to_netcdf(
            data_dir / f"Runoff_sim_{station_id}_{start}_{end}.nc"
        )
    return data_dir


def test_station_portrait_preserves_recorded_missing_station_as_nan(tmp_path, monkeypatch):
    processor, comparison_module = _processor(tmp_path, ["bias"])
    processor.bias = lambda s, o: s - o
    monkeypatch.setattr(comparison_module, "make_scenarios_comparison_Portrait_Plot_seasonal", lambda *a, **k: None)

    rows = [
        {"ID": "A", "use_syear": 2000, "use_eyear": 2000, "status": "ok", "reason": ""},
        {"ID": "B", "use_syear": 2000, "use_eyear": 2000, "status": "skipped", "reason": "missing variable"},
    ]
    data_dir = _write_station_case(
        tmp_path,
        rows,
        {
            "A": (
                2000,
                2000,
                [1.0, 2.0, 3.0, 4.0],
                [2.0, 4.0, 6.0, 8.0],
                ["2000-12-01", "2000-03-01", "2000-06-01", "2000-09-01"],
            )
        },
    )

    pd.DataFrame(
        [
            {"ID": "A", "status": "ok", "reason": ""},
            {"ID": "B", "status": "unavailable", "reason": "missing variable"},
        ]
    ).to_csv(data_dir / "Runoff_evaluation_status.csv", index=False)

    sim_nml, ref_nml = _nmls()
    processor.scenarios_Portrait_Plot_seasonal_comparison(str(tmp_path), sim_nml, ref_nml, ["Runoff"], [], ["bias"], {})

    out = pd.read_csv(tmp_path / "comparisons" / "Portrait_Plot_seasonal" / "Portrait_Plot_seasonal.csv", sep="\t")
    assert out.loc[0, ["bias_DJF", "bias_MAM", "bias_JJA", "bias_SON"]].tolist() == [1.0, 2.0, 3.0, 4.0]


def test_station_portrait_empty_seasons_are_nan_without_metric_call(tmp_path, monkeypatch):
    processor, comparison_module = _processor(tmp_path, ["only_djf"])
    monkeypatch.setattr(comparison_module, "make_scenarios_comparison_Portrait_Plot_seasonal", lambda *a, **k: None)
    calls = []

    def only_djf(s, o):
        calls.append(str(s.time.dt.season.values[0]))
        return s - o

    processor.only_djf = only_djf
    _write_station_case(
        tmp_path,
        [{"ID": "A", "use_syear": 2000, "use_eyear": 2000}],
        {"A": (2000, 2000, [1.0], [3.0], ["2000-12-01"])},
    )

    sim_nml, ref_nml = _nmls()
    processor.scenarios_Portrait_Plot_seasonal_comparison(
        str(tmp_path), sim_nml, ref_nml, ["Runoff"], [], ["only_djf"], {}
    )

    out = pd.read_csv(tmp_path / "comparisons" / "Portrait_Plot_seasonal" / "Portrait_Plot_seasonal.csv", sep="\t")
    assert out.loc[0, "only_djf_DJF"] == 2.0
    assert out.loc[0, ["only_djf_MAM", "only_djf_JJA", "only_djf_SON"]].isna().all()
    assert calls == ["DJF"]


def test_station_portrait_undefined_metric_result_stays_nan(tmp_path, monkeypatch):
    processor, comparison_module = _processor(tmp_path, ["undefined"])
    processor.undefined = lambda s, o: xr.full_like(s, np.nan)
    monkeypatch.setattr(comparison_module, "make_scenarios_comparison_Portrait_Plot_seasonal", lambda *a, **k: None)
    _write_station_case(
        tmp_path,
        [{"ID": "A", "use_syear": 2000, "use_eyear": 2000}],
        {"A": (2000, 2000, [1.0], [1.0], ["2000-12-01"])},
    )

    sim_nml, ref_nml = _nmls()
    processor.scenarios_Portrait_Plot_seasonal_comparison(
        str(tmp_path), sim_nml, ref_nml, ["Runoff"], [], ["undefined"], {}
    )

    out = pd.read_csv(tmp_path / "comparisons" / "Portrait_Plot_seasonal" / "Portrait_Plot_seasonal.csv", sep="\t")
    assert out.loc[0, ["undefined_DJF", "undefined_MAM", "undefined_JJA", "undefined_SON"]].isna().all()


def test_station_portrait_unrecorded_missing_files_are_fatal(tmp_path, monkeypatch):
    processor, comparison_module = _processor(tmp_path, ["bias"])
    processor.bias = lambda s, o: s - o
    monkeypatch.setattr(comparison_module, "make_scenarios_comparison_Portrait_Plot_seasonal", lambda *a, **k: None)
    _write_station_case(
        tmp_path,
        [{"ID": "A", "use_syear": 2000, "use_eyear": 2000}],
        {},
    )

    sim_nml, ref_nml = _nmls()
    with pytest.raises(FileNotFoundError):
        processor.scenarios_Portrait_Plot_seasonal_comparison(
            str(tmp_path), sim_nml, ref_nml, ["Runoff"], [], ["bias"], {}
        )


def test_station_portrait_scalar_metric_result_is_accepted(tmp_path, monkeypatch):
    processor, comparison_module = _processor(tmp_path, ["scalar_bias"])
    processor.scalar_bias = lambda s, o: float((s - o).mean(skipna=True))
    monkeypatch.setattr(comparison_module, "make_scenarios_comparison_Portrait_Plot_seasonal", lambda *a, **k: None)
    _write_station_case(
        tmp_path,
        [{"ID": "A", "use_syear": 2000, "use_eyear": 2000}],
        {"A": (2000, 2000, [1.0], [4.0], ["2000-12-01"])},
    )

    sim_nml, ref_nml = _nmls()
    processor.scenarios_Portrait_Plot_seasonal_comparison(
        str(tmp_path), sim_nml, ref_nml, ["Runoff"], [], ["scalar_bias"], {}
    )

    out = pd.read_csv(tmp_path / "comparisons" / "Portrait_Plot_seasonal" / "Portrait_Plot_seasonal.csv", sep="\t")
    assert out.loc[0, "scalar_bias_DJF"] == 3.0
    assert out.loc[0, ["scalar_bias_MAM", "scalar_bias_JJA", "scalar_bias_SON"]].isna().all()
