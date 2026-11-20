"""Regressions for high-flow quantiles, flood timing, and inundation CSI."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from openbench.cli._metric_guidance import metric_score_errors
from openbench.core._hydrology_extreme_metrics import inundation_spatial_csi, peak_timing_stats
from openbench.core.metrics import metrics
from openbench.core.registry import METRIC_LABELS, METRICS_ITEMS


def daily(values):
    return xr.DataArray(
        values,
        dims=("time",),
        coords={"time": pd.date_range("2001-01-01", periods=len(values), freq="D")},
    )


def test_quantile_bias_uses_paired_non_exceedance_quantiles():
    observed = daily([1.0, 2.0, 3.0, 4.0, 100.0])
    simulated = observed * 1.2
    evaluator = metrics()
    assert np.isclose(float(evaluator.Q95_Bias(simulated, observed)), 20.0)
    assert np.isclose(float(evaluator.Q99_Bias(simulated, observed)), 20.0)

    simulated = simulated.where(simulated.time != simulated.time[4])
    observed = observed.where(observed.time != observed.time[1])
    assert np.isclose(float(evaluator.Q95_Bias(simulated, observed)), 20.0)
    assert np.isnan(float(evaluator.Q99_Bias(daily([2.0, 2.0, 2.0]), daily([0.0, 0.0, 0.0]))))


def test_peak_timing_matches_separate_events_with_match_rate():
    observed = np.zeros(90)
    simulated = np.zeros(90)
    for day, height in ((12, 12), (45, 15), (78, 18)):
        observed[day - 1 : day + 2] = [height / 4, height, height / 4]
    for day, height in ((14, 13), (43, 16)):
        simulated[day - 1 : day + 2] = [height / 4, height, height / 4]

    evaluator = metrics()
    assert np.isclose(float(evaluator.PeakTimingError(daily(simulated), daily(observed))), 2.0)
    assert np.isclose(float(evaluator.PeakTimingMatchRate(daily(simulated), daily(observed))), 2 / 3)


def test_missing_flood_peak_and_monthly_means_are_not_perfect_scores():
    observed = np.zeros(60)
    simulated = np.zeros(60)
    observed[15], simulated[36] = 10, 10
    evaluator = metrics()
    assert np.isnan(float(evaluator.PeakTimingError(daily(simulated), daily(observed))))
    assert float(evaluator.PeakTimingMatchRate(daily(simulated), daily(observed))) == 0.0

    months = pd.date_range("2001-01-01", periods=36, freq="MS")
    values = xr.DataArray(np.arange(36), dims=("time",), coords={"time": months})
    assert np.isnan(float(evaluator.PeakTimingError(values, values)))


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_peak_timing_uses_days_for_all_datetime_precisions(monkeypatch, unit):
    observed = np.zeros(90)
    simulated = np.zeros(90)
    observed[[12, 45, 78]] = [12, 15, 18]
    simulated[[14, 43]] = [13, 16]
    sim, obs = daily(simulated), daily(observed)
    original_to_datetime = pd.to_datetime
    # Older xarray versions coerce coordinates to ns. Exercise the actual
    # parsed index resolution, including the us default used by newer pandas.
    monkeypatch.setattr(pd, "to_datetime", lambda values: original_to_datetime(values).as_unit(unit))
    error, rate = peak_timing_stats(sim, obs)
    assert float(error) == pytest.approx(2.0)
    assert float(rate) == pytest.approx(2 / 3)


def test_temporal_csi_masks_never_wet_and_invalid_fractions():
    times = pd.date_range("2001-01-01", periods=3, freq="D")
    observed = xr.DataArray(
        [[[0.2, 0.0, 0.0]], [[0.0, 0.0, 0.0]], [[0.5, 0.0, 0.0]]],
        dims=("time", "lat", "lon"),
        coords={"time": times, "lat": [0], "lon": [0, 1, 2]},
    )
    simulated = xr.DataArray(
        [[[0.1, 0.2, 0.0]], [[0.4, 0.0, 0.0]], [[0.0, 0.0, 0.0]]],
        dims=observed.dims,
        coords=observed.coords,
    )
    score = metrics().Inundation_CSI(simulated, observed)
    assert np.isclose(float(score.sel(lat=0, lon=0)), 1 / 3)
    assert np.isclose(float(score.sel(lat=0, lon=1)), 0.0)
    assert np.isnan(float(score.sel(lat=0, lon=2)))

    invalid = simulated.copy()
    invalid.loc[dict(lon=1, time=times[0])] = 20.0
    assert np.isnan(float(metrics().Inundation_CSI(invalid, observed).sel(lat=0, lon=1)))


def test_spatial_csi_area_weights_and_common_valid_pixels():
    times = pd.date_range("2001-01-01", periods=2, freq="D")
    observed = xr.DataArray(
        [[[0.2, 0.0]], [[0.0, 0.2]]],
        dims=("time", "lat", "lon"),
        coords={"time": times, "lat": [0.0], "lon": [0.0, 1.0]},
    )
    simulated = xr.DataArray(
        [[[0.3, 0.3]], [[np.nan, 0.0]]],
        dims=observed.dims,
        coords=observed.coords,
    )
    result = inundation_spatial_csi(simulated, observed)
    np.testing.assert_allclose(result.CSI_spatial.values, [0.5, 0.0])
    assert result.valid_weight.isel(time=1).item() == 1

    elevated_observed = xr.DataArray(
        np.ones((1, 2, 1)),
        dims=("time", "lat", "lon"),
        coords={"time": times[:1], "lat": [0.0, 60.0], "lon": [0.0]},
    )
    elevated_simulated = elevated_observed.copy()
    elevated_simulated.loc[dict(lat=60.0)] = 0.0
    assert np.isclose(
        inundation_spatial_csi(elevated_simulated, elevated_observed).CSI_spatial.item(),
        1 / 1.5,
    )


@pytest.mark.parametrize("vkey", ["", "_flooded_fraction"])
def test_grid_evaluation_exports_spatial_csi_csv(tmp_path, monkeypatch, vkey):
    from openbench.core.evaluation import Evaluation_grid

    times = pd.date_range("2001-01-01", periods=2, freq="MS")
    observed = xr.DataArray(
        [[[0.2, 0.0]], [[0.0, 0.3]]],
        dims=("time", "lat", "lon"),
        coords={"time": times, "lat": [0.0], "lon": [0.0, 1.0]},
    )
    simulated = xr.DataArray(
        [[[0.2, 0.3]], [[0.0, 0.3]]],
        dims=observed.dims,
        coords=observed.coords,
    )
    evaluator = object.__new__(Evaluation_grid)
    evaluator.item, evaluator.ref_source, evaluator.sim_source = "Inundation_Fraction", "OBS", "D"
    evaluator.casedir = str(tmp_path)
    monkeypatch.setattr(Evaluation_grid, "_save_metric_array", lambda *args, **kwargs: None)
    evaluator.process_metric("Inundation_CSI", simulated, observed, vkey=vkey)

    name = f"Inundation_Fraction_ref_OBS_sim_D_Inundation_CSI{vkey}"
    monthly = pd.read_csv(tmp_path / "metrics" / (name + "_spatial_timeseries.csv"))
    summary = pd.read_csv(tmp_path / "metrics" / (name + "_spatial_summary.csv"))
    np.testing.assert_allclose(monthly.CSI_spatial.to_numpy(), [0.5, 1.0])
    assert np.isclose(summary.CSI_spatial_pooled.iloc[0], 2 / 3)


def test_metric_registry_exposes_all_new_metrics():
    registered = {name for group in METRICS_ITEMS.values() for name in group}
    expected = {"Q95_Bias", "Q99_Bias", "PeakTimingError", "PeakTimingMatchRate", "Inundation_CSI"}
    assert expected <= registered
    assert expected <= METRIC_LABELS.keys()


def test_csi_fails_for_inundation_area_direct_invocation():
    evaluator = metrics()
    evaluator.item = "Inundation_Area"
    with pytest.raises(ValueError, match="Inundation_Fraction"):
        evaluator.Inundation_CSI(daily([0.1, 0.2]), daily([0.1, 0.2]))


def test_preflight_checks_peak_resolution_and_inundation_variable():
    cfg = SimpleNamespace(
        metrics=["PeakTimingError", "PeakTimingMatchRate", "Inundation_CSI"],
        scores=[],
        project=SimpleNamespace(years=(2001, 2001)),
        evaluation=SimpleNamespace(variables=["Streamflow"]),
    )
    errors = metric_score_errors(cfg, "Month")
    assert any("PeakTimingError" in message for message in errors)
    assert any("PeakTimingMatchRate" in message for message in errors)
    assert any("Inundation_CSI" in message for message in errors)

    cfg.evaluation.variables = ["Streamflow", "Inundation_Fraction"]
    assert metric_score_errors(cfg, "Day") == []


@pytest.mark.parametrize("resolution", ["Day", "Hour", "3Hour", "6Hour", "6hr", "24hr"])
def test_peak_preflight_accepts_daily_or_finer_resolutions(resolution):
    cfg = SimpleNamespace(
        metrics=["PeakTimingError"],
        scores=[],
        project=SimpleNamespace(years=(2001, 2002)),
        evaluation=SimpleNamespace(variables=["Streamflow"]),
    )
    assert metric_score_errors(cfg, resolution) == []


def test_csi_skips_other_variables_in_both_evaluation_paths(tmp_path, monkeypatch):
    from openbench.core import evaluation

    monkeypatch.setattr(evaluation, "_HAS_OUTPUT_MANAGER", False)
    config = {
        "item": "Streamflow",
        "casedir": str(tmp_path),
        "metrics": ["KGE", "Inundation_CSI", "Q95_Bias", "PeakTimingError"],
        "sim_varname": "qsim",
        "ref_varname": "qobs",
    }
    grid = evaluation.Evaluation_grid(config, {})
    station = evaluation.Evaluation_stn(config, {})
    assert grid.metrics == ["KGE", "Q95_Bias", "PeakTimingError"]
    assert station.metrics == ["KGE", "Q95_Bias", "PeakTimingError"]
    assert config["metrics"] == ["KGE", "Inundation_CSI", "Q95_Bias", "PeakTimingError"]

    flooded = dict(config, item="Inundation_Fraction")
    inundation = evaluation.Evaluation_grid(flooded, {})
    assert inundation.metrics == ["KGE", "Inundation_CSI"]


def test_climatology_rejects_event_peak_metrics():
    from openbench.data.climatology import ClimatologyProcessor

    processor = ClimatologyProcessor()
    for name in ("PeakTimingError", "PeakTimingMatchRate"):
        assert not processor.is_metric_supported(name, clim_type="monthly", time_points=12)
