"""Group-by reductions bypass flox and keep their values."""

import threading

import numpy as np
import pandas as pd
import pytest
import xarray as xr


def _monthly_grid(seed):
    values = np.random.default_rng(seed).random((48, 3, 4))
    values[:, 0, 0] = np.nan
    values[5, 1, 1] = np.nan
    times = pd.date_range("2001-01-31", periods=48, freq="ME")
    return xr.DataArray(values, dims=("time", "lat", "lon"), coords={"time": times})


def _score(name):
    from openbench.core.scores import scores

    return lambda s, o: getattr(scores(), name)(s, o)


def _ws_nse(s, o):
    from openbench.core.metrics import metrics

    return metrics().wsNSE(s, o, season_weights={"DJF": 2.0, "MAM": 1.0, "JJA": 1.0, "SON": 1.0})


def _stat_resample(s, _o):
    from openbench.core.statistics import statistics_calculate

    return statistics_calculate({}).stat_resample(s, "1YE")


@pytest.mark.parametrize(
    "compute",
    [_score("nPhaseScore"), _score("nIavScore"), _score("nSeasonalityScore"), _ws_nse, _stat_resample],
    ids=["nPhaseScore", "nIavScore", "nSeasonalityScore", "wsNSE", "stat_resample"],
)
def test_groupby_reductions_bypass_flox_with_same_values(monkeypatch, compute):
    flox_xarray = pytest.importorskip("flox.xarray")
    sim, obs = _monthly_grid(0), _monthly_grid(1)
    expected = compute(sim, obs)

    def fail_flox(*_args, **_kwargs):
        raise AssertionError("flox used for a time group-by")

    monkeypatch.setattr(flox_xarray, "xarray_reduce", fail_flox)

    result = compute(sim, obs)

    np.testing.assert_allclose(result.values, expected.values, rtol=1e-12, equal_nan=True)


def test_without_flox_restores_option_when_nested_and_threaded():
    from openbench.util.groupby import without_flox

    before = xr.get_options()["use_flox"]
    with without_flox():
        with without_flox():
            assert xr.get_options()["use_flox"] is False
        assert xr.get_options()["use_flox"] is False

    barrier = threading.Barrier(8)

    def worker():
        barrier.wait()
        for _ in range(50):
            with without_flox():
                assert xr.get_options()["use_flox"] is False

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert xr.get_options()["use_flox"] == before
