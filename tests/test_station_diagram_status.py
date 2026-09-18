import numpy as np
import pandas as pd
import xarray as xr


def _processor(tmp_path):
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
        [],
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


def _write_station_inputs(tmp_path):
    metrics_dir = tmp_path / "metrics"
    data_dir = tmp_path / "data" / "stn_RefA_SimA"
    metrics_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    rows = pd.DataFrame(
        [
            {
                "ID": "A",
                "use_syear": 2000,
                "use_eyear": 2000,
                "bias": -999.0,
                "RMSE": -999.0,
                "correlation": -999.0,
                "SMPI": -999.0,
            },
            {
                "ID": "B",
                "use_syear": 2000,
                "use_eyear": 2000,
                "bias": -999.0,
                "RMSE": -999.0,
                "correlation": -999.0,
                "SMPI": -999.0,
            },
        ]
    )
    rows.to_csv(metrics_dir / "Runoff_stn_RefA_SimA_evaluations.csv", index=False)
    pd.DataFrame(
        [
            {"ID": "A", "status": "ok", "reason": ""},
            {"ID": "B", "status": "unavailable", "reason": "known missing variable"},
        ]
    ).to_csv(data_dir / "Runoff_evaluation_status.csv", index=False)

    times = pd.date_range("2000-01-01", periods=4, freq="MS")
    for station, offset in (("A", 0.0), ("B", 1000.0)):
        # B intentionally has stale numeric files; status sidecar must win.
        xr.Dataset({"runoff_ref": ("time", [1.0, 2.0, 1.0, 2.0])}, coords={"time": times}).to_netcdf(
            data_dir / f"Runoff_ref_{station}_2000_2000.nc"
        )
        xr.Dataset({"runoff_sim": ("time", np.array([2.0, 3.0, 2.0, 3.0]) + offset)}, coords={"time": times}).to_netcdf(
            data_dir / f"Runoff_sim_{station}_2000_2000.nc"
        )


def test_target_taylor_smpi_station_outputs_preserve_unavailable_status(tmp_path, monkeypatch):
    processor, comparison_module = _processor(tmp_path)
    sim_nml, ref_nml = _nmls()
    _write_station_inputs(tmp_path)

    monkeypatch.setattr(comparison_module, "make_scenarios_comparison_Target_Diagram", lambda *a, **k: None)
    monkeypatch.setattr(comparison_module, "make_scenarios_comparison_Taylor_Diagram", lambda *a, **k: None)
    monkeypatch.setattr(
        comparison_module, "make_scenarios_comparison_Single_Model_Performance_Index", lambda *a, **k: None
    )

    processor.scenarios_Target_Diagram_comparison(str(tmp_path), sim_nml, ref_nml, ["Runoff"], [], [], {})
    processor.scenarios_Taylor_Diagram_comparison(str(tmp_path), sim_nml, ref_nml, ["Runoff"], [], [], {})
    processor.scenarios_Single_Model_Performance_Index_comparison(
        str(tmp_path), sim_nml, ref_nml, ["Runoff"], [], [], {}
    )

    outputs = [
        tmp_path / "comparisons" / "Target_Diagram" / "target_diagram_Runoff_stn_RefA_SimA.csv",
        tmp_path / "comparisons" / "Taylor_Diagram" / "taylor_diagram_Runoff_stn_RefA_SimA.csv",
        tmp_path / "comparisons" / "Single_Model_Performance_Index" / "SMPI_Runoff_stn_RefA_SimA.csv",
    ]
    for path in outputs:
        frame = pd.read_csv(path)
        assert not frame.columns.duplicated().any()
        assert -999.0 not in frame.select_dtypes(include=["number"]).to_numpy()
        unavailable = frame.loc[frame["ID"] == "B"].iloc[0]
        assert unavailable["status"] == "unavailable"
        assert unavailable["reason"] == "known missing variable"
        numeric_cols = [col for col in frame.columns if col not in {"ID", "use_syear", "use_eyear", "status", "reason"}]
        assert unavailable[numeric_cols].isna().all()
        available = frame.loc[frame["ID"] == "A"].iloc[0]
        assert available["status"] == "ok"
        assert np.isfinite(available[numeric_cols].astype(float)).any()

    target_summary = pd.read_csv(
        next((tmp_path / "comparisons" / "Target_Diagram").glob("target_diagram*Runoff*RefA.csv")), sep="\t"
    )
    assert np.isfinite(target_summary.filter(like="SimA_").iloc[0].dropna().astype(float)).all()

    taylor_summary = pd.read_csv(
        next((tmp_path / "comparisons" / "Taylor_Diagram").glob("taylor_diagram*Runoff*RefA.csv"))
    )
    assert np.isfinite(taylor_summary.filter(like="SimA_").iloc[0].astype(float)).all()

    smpi_summary = pd.read_csv(
        tmp_path / "comparisons" / "Single_Model_Performance_Index" / "SMPI_comparison.csv", sep="\t"
    )
    assert np.isfinite(smpi_summary[["SMPI", "Lower_CI", "Upper_CI"]].iloc[0].astype(float)).all()


def test_smpi_station_computed_all_undefined_is_unavailable(tmp_path, monkeypatch):
    processor, comparison_module = _processor(tmp_path)
    sim_nml, ref_nml = _nmls()

    metrics_dir = tmp_path / "metrics"
    data_dir = tmp_path / "data" / "stn_RefA_SimA"
    metrics_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    pd.DataFrame([{"ID": "C", "use_syear": 2000, "use_eyear": 2000, "SMPI": -999.0}]).to_csv(
        metrics_dir / "Runoff_stn_RefA_SimA_evaluations.csv", index=False
    )
    times = pd.date_range("2000-01-01", periods=4, freq="MS")
    xr.Dataset({"runoff_ref": ("time", [1.0, 1.0, 1.0, 1.0])}, coords={"time": times}).to_netcdf(
        data_dir / "Runoff_ref_C_2000_2000.nc"
    )
    xr.Dataset({"runoff_sim": ("time", [2.0, 2.0, 2.0, 2.0])}, coords={"time": times}).to_netcdf(
        data_dir / "Runoff_sim_C_2000_2000.nc"
    )
    monkeypatch.setattr(
        comparison_module, "make_scenarios_comparison_Single_Model_Performance_Index", lambda *a, **k: None
    )

    processor.scenarios_Single_Model_Performance_Index_comparison(
        str(tmp_path), sim_nml, ref_nml, ["Runoff"], [], [], {}
    )

    frame = pd.read_csv(tmp_path / "comparisons" / "Single_Model_Performance_Index" / "SMPI_Runoff_stn_RefA_SimA.csv")
    row = frame.iloc[0]
    assert row["status"] == "unavailable"
    assert row["reason"] == "computed SMPI metrics are undefined"
    assert row[["SMPI", "Lower_CI", "Upper_CI"]].isna().all()
    assert -999.0 not in frame.select_dtypes(include=["number"]).to_numpy()


def test_station_diagram_unknown_missing_file_remains_fatal(tmp_path, monkeypatch):
    methods = [
        ("scenarios_Target_Diagram_comparison", "make_scenarios_comparison_Target_Diagram"),
        ("scenarios_Taylor_Diagram_comparison", "make_scenarios_comparison_Taylor_Diagram"),
        (
            "scenarios_Single_Model_Performance_Index_comparison",
            "make_scenarios_comparison_Single_Model_Performance_Index",
        ),
    ]
    for method_name, renderer_name in methods:
        case_dir = tmp_path / method_name
        processor, comparison_module = _processor(case_dir)
        sim_nml, ref_nml = _nmls()
        (case_dir / "metrics").mkdir(parents=True)
        (case_dir / "data" / "stn_RefA_SimA").mkdir(parents=True)
        pd.DataFrame([{"ID": "A", "use_syear": 2000, "use_eyear": 2000}]).to_csv(
            case_dir / "metrics" / "Runoff_stn_RefA_SimA_evaluations.csv", index=False
        )
        monkeypatch.setattr(comparison_module, renderer_name, lambda *a, **k: None)

        try:
            getattr(processor, method_name)(str(case_dir), sim_nml, ref_nml, ["Runoff"], [], [], {})
        except FileNotFoundError:
            continue
        raise AssertionError(f"{method_name} hid an unrecorded station file gap")
