import numpy as np
import pytest
import xarray as xr

from openbench.core.appendix_methods import (
    brier_decomposition,
    brier_score,
    contingency_scores,
    crps_ensemble,
    fit_gev,
    fit_gpd,
    ideal_point_error,
    roc_auc,
    roc_curve,
    taylor_skill_score,
    uncertainty_factors,
)


def da(values, dim="time"):
    return xr.DataArray(values, dims=dim)


def test_uncertainty_factors_use_band_width_and_coverage():
    result = uncertainty_factors(da([0.0, 1.0, 2.0]), da([2.0, 3.0, 4.0]), da([1.0, 4.0, 3.0]))

    np.testing.assert_allclose(result.R_factor, 2.0 / np.std([1.0, 4.0, 3.0]))
    np.testing.assert_allclose(result.p_factor, 2.0 / 3.0)


def test_ipe_drops_degenerate_components():
    candidate = {"candidate": ["a", "b"]}
    result = ideal_point_error(
        xr.DataArray([0.0, 2.0], coords=candidate, dims="candidate"),
        xr.DataArray([1.0, 1.0], coords=candidate, dims="candidate"),
        xr.DataArray([0.0, 4.0], coords=candidate, dims="candidate"),
    )

    np.testing.assert_allclose(result, [1.0, 0.0])


def test_brier_score_and_decomposition_identity():
    probability = da([0.25, 0.25, 0.75, 0.75])
    outcome = da([0.0, 0.0, 1.0, 1.0])
    score = brier_score(probability, outcome)
    decomposition = brier_decomposition(probability, outcome, bins=2)

    np.testing.assert_allclose(score, 0.0625)
    np.testing.assert_allclose(
        decomposition.BS,
        decomposition.reliability
        - decomposition.resolution
        + decomposition.uncertainty
        + decomposition.binning_residual,
    )


def test_binary_event_scores_use_the_contingency_table():
    result = contingency_scores(da([1, 1, 0, 0]), da([1, 0, 1, 0]))

    np.testing.assert_allclose(result.CSI, 1.0 / 3.0)
    np.testing.assert_allclose(result.HSS, 0.0)
    np.testing.assert_allclose(result.POD, 0.5)
    np.testing.assert_allclose(result.FAR, 0.5)


def test_taylor_skill_requires_explicit_reference_correlation():
    result = taylor_skill_score(da([1.0, 2.0, 3.0]), da([1.0, 2.0, 3.0]), reference_correlation=1.0)

    np.testing.assert_allclose(result, 1.0)


def test_crps_empirical_ensemble_formula():
    ensemble = xr.DataArray([[0.0, 2.0], [1.0, 3.0]], dims=["time", "member"])
    observation = da([1.0, 2.0])

    np.testing.assert_allclose(crps_ensemble(ensemble, observation), 0.5)


def test_roc_curve_and_auc_handle_tied_probabilities():
    probability = da([0.1, 0.4, 0.4, 0.9])
    outcome = da([0, 0, 1, 1])

    np.testing.assert_allclose(roc_auc(probability, outcome), 0.875)
    curve = roc_curve(probability, outcome)
    np.testing.assert_allclose(curve.TPR.values[[0, -1]], [0.0, 1.0])
    np.testing.assert_allclose(curve.FPR.values[[0, -1]], [0.0, 1.0])


def test_probability_and_binary_domains_fail_without_eager_grid_validation():
    probability = da([0.2, 1.2]).chunk({"time": 1})
    outcome = da([0, 1]).chunk({"time": 1})
    forecast = da([0, 2]).chunk({"time": 1})

    assert np.isnan(float(brier_score(probability, outcome).compute()))
    assert np.isnan(float(brier_decomposition(probability, outcome).BS.compute()))
    assert np.isnan(float(roc_auc(probability, outcome).compute()))
    assert np.isnan(float(contingency_scores(forecast, outcome).CSI.compute()))
    with pytest.raises(ValueError, match="probabilities"):
        roc_curve(probability.compute(), outcome.compute())


def test_extreme_value_fits_return_finite_parameters():
    rng = np.random.default_rng(4)
    gev = fit_gev(da(rng.gumbel(size=200)))
    gpd = fit_gpd(da(rng.exponential(size=400)), threshold=0.5)

    assert np.isfinite(gev.to_array()).all()
    assert np.isfinite(gpd[["shape", "scale"]].to_array()).all()
    assert float(gev.scale) > 0
    assert float(gpd.scale) > 0
