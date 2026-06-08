"""Tests for scores module — verify normalized score calculations."""

import numpy as np
import pandas as pd
import xarray as xr


def make_da(values):
    """Helper: wrap a list of floats into a time-indexed xr.DataArray."""
    times = pd.date_range("2000-01-01", periods=len(values), freq="MS")
    return xr.DataArray(values, coords={"time": times}, dims=["time"])


def make_lat_da(values):
    """Helper: wrap a 2-D list into monthly time x lat xr.DataArray."""
    times = pd.date_range("2000-01-01", periods=len(values), freq="MS")
    return xr.DataArray(values, coords={"time": times, "lat": [0, 1]}, dims=["time", "lat"])


def test_scores_class_exists():
    """Verify scores class can be imported."""
    from openbench.core.scores import scores

    assert scores is not None


def test_nbias_score_perfect():
    """Perfect match should give score close to 1."""
    from openbench.core.scores import scores

    s = scores()
    obs = make_da([1.0, 2.0, 3.0, 4.0, 5.0])
    sim = make_da([1.0, 2.0, 3.0, 4.0, 5.0])
    result = float(s.nBiasScore(sim, obs))
    assert result > 0.99  # Should be very close to 1


def test_nbias_score_range():
    """Score should be between 0 and 1."""
    from openbench.core.scores import scores

    s = scores()
    obs = make_da([1.0, 2.0, 3.0, 4.0, 5.0])
    sim = make_da([10.0, 20.0, 30.0, 40.0, 50.0])  # Very biased
    result = float(s.nBiasScore(sim, obs))
    assert 0.0 <= result <= 1.0


def test_nrmse_score_perfect():
    """Perfect match should give score close to 1."""
    from openbench.core.scores import scores

    s = scores()
    obs = make_da([1.0, 2.0, 3.0, 4.0, 5.0])
    sim = make_da([1.0, 2.0, 3.0, 4.0, 5.0])
    result = float(s.nRMSEScore(sim, obs))
    assert result > 0.99


def test_scores_import_does_not_globally_ignore_runtime_warnings(monkeypatch):
    """Importing scores must not hide RuntimeWarning process-wide."""
    import importlib
    import warnings

    scores_module = importlib.import_module("openbench.core.scores")

    monkeypatch.setattr(warnings, "filters", [])
    importlib.reload(scores_module)

    assert not any(
        action == "ignore" and category is RuntimeWarning
        for action, _message, category, _module, _lineno in warnings.filters
    )


def test_scores_use_same_valid_pairs_for_normalization_terms():
    """Normalized scores should not mix pairwise-valid errors with all obs variance."""
    from openbench.core.scores import scores

    s = scores()
    obs = make_da([1.0, 2.0, 3.0])
    sim = make_da([2.0, np.nan, 4.0])

    # Valid pairs are (2, 1), (4, 3): bias=1 and obs CRMS=1.
    assert abs(float(s.nBiasScore(sim, obs)) - np.exp(-1.0)) < 1e-10


def test_overall_score_drops_unavailable_components_per_cell():
    """A NaN IAV score in one cell should not force that cell's overall score to NaN."""
    from openbench.core.scores import scores

    s = scores()
    seasonal = np.arange(1.0, 13.0)
    # lat=0 repeats the same seasonal cycle each year (IAV undefined);
    # lat=1 has interannual variability and should keep all components.
    obs_values = np.column_stack(
        [
            np.tile(seasonal, 2),
            np.concatenate([seasonal, seasonal + 1.0]),
        ]
    )
    obs = make_lat_da(obs_values)
    sim = make_lat_da(obs_values + 0.1)

    result = s.Overall_Score(sim, obs)

    assert np.isfinite(float(result.sel(lat=0)))


def test_spatial_score_constant_simulation_std_returns_nan_without_infinite_sigma():
    from openbench.core.scores import scores

    scorer = scores()
    times = pd.date_range("2000-01-01", periods=2, freq="MS")
    obs = xr.DataArray(
        np.array(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[2.0, 3.0], [4.0, 5.0]],
            ]
        ),
        coords={"time": times, "lat": [0, 1], "lon": [0, 1]},
        dims=["time", "lat", "lon"],
    )
    sim = xr.zeros_like(obs) + 1.0

    result = scorer.nSpatialScore(sim, obs)

    assert np.isnan(result).all()


def test_nseasonality_score_returns_annual_cycle_amplitude_score_without_month_dimension():
    from openbench.core.scores import scores

    scorer = scores()
    obs = make_da([1.0, 3.0] * 12)
    sim = make_da([1.0, 5.0] * 12)

    result = scorer.nSeasonalityScore(sim, obs)

    assert "month" not in result.dims
    assert abs(float(result) - np.exp(-1.0)) < 1e-12


def test_nphase_score_returns_nan_for_all_nan_cells():
    from openbench.core.scores import scores

    scorer = scores()
    times = pd.date_range("2000-01-01", periods=12, freq="MS")
    obs = xr.DataArray(
        np.column_stack([np.full(12, np.nan), np.arange(12.0)]),
        dims=("time", "lat"),
        coords={"time": times, "lat": [0, 1]},
    )
    sim = xr.DataArray(
        np.column_stack([np.full(12, np.nan), np.roll(np.arange(12.0), 1)]),
        dims=("time", "lat"),
        coords={"time": times, "lat": [0, 1]},
    )

    result = scorer.nPhaseScore(sim, obs)

    assert np.isnan(float(result.sel(lat=0)))
    assert np.isfinite(float(result.sel(lat=1)))


def test_nphase_score_returns_nan_for_flat_seasonal_cycles():
    from openbench.core.scores import scores

    scorer = scores()
    obs = make_da([1.0] * 24)
    sim = make_da([1.0] * 24)

    result = scorer.nPhaseScore(sim, obs)

    assert np.isnan(float(result))
