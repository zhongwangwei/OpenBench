import numpy as np
import pandas as pd
import xarray as xr


def test_grid_portrait_keeps_annual_cycle_scores_as_nan_for_seasonal_slices(tmp_path, monkeypatch, caplog):
    import openbench.core.comparison as comparison_module

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    times = pd.date_range("2001-01-01", periods=12, freq="MS")
    coords = {"time": times, "lat": [0.0], "lon": [0.0]}
    ref = np.arange(12, dtype=float).reshape(12, 1, 1)
    sim = np.roll(ref, 1, axis=0)
    xr.Dataset({"runoff_ref": (("time", "lat", "lon"), ref)}, coords=coords).to_netcdf(
        data_dir / "Runoff_ref_RefA_runoff_ref.nc"
    )
    xr.Dataset({"runoff_sim": (("time", "lat", "lon"), sim)}, coords=coords).to_netcdf(
        data_dir / "Runoff_sim_SimA_runoff_sim.nc"
    )

    plot_calls = []
    monkeypatch.setattr(
        comparison_module,
        "make_scenarios_comparison_Portrait_Plot_seasonal",
        lambda *args, **kwargs: plot_calls.append(args),
    )
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
        ["nPhaseScore", "nSeasonalityScore"],
        [],
    )

    processor.scenarios_Portrait_Plot_seasonal_comparison(
        str(tmp_path),
        {
            "general": {"Runoff_sim_source": ["SimA"]},
            "Runoff": {"SimA_data_type": "grid", "SimA_varname": "runoff_sim"},
        },
        {
            "general": {"Runoff_ref_source": "RefA"},
            "Runoff": {"RefA_data_type": "grid", "RefA_varname": "runoff_ref"},
        },
        ["Runoff"],
        ["nPhaseScore", "nSeasonalityScore"],
        [],
        {},
    )

    result = pd.read_csv(
        tmp_path / "comparisons" / "Portrait_Plot_seasonal" / "Portrait_Plot_seasonal.csv",
        sep="\t",
    )
    seasonal_columns = [
        f"{score}_{season}" for score in ("nPhaseScore", "nSeasonalityScore") for season in ("DJF", "MAM", "JJA", "SON")
    ]
    assert result.loc[0, seasonal_columns].isna().all()
    assert len(plot_calls) == 1
    assert "require all 12 months" in caplog.text
    assert "written as N/A" in caplog.text
    assert "not as zero scores" in caplog.text
