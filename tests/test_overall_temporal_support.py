import numpy as np
import pandas as pd
import pytest
import xarray as xr

from openbench.core.scores import scores


@pytest.mark.parametrize("level", [0.0, 5.0])
def test_constant_reference_cannot_inherit_spatial_score(level):
    time = pd.date_range("1997-01-01", periods=24, freq="MS")
    values = np.stack([np.full(24, level), np.arange(24) + 10.0], axis=1)[:, None, :]
    ref = xr.DataArray(values, dims=("time", "lat", "lon"), coords={"time": time})
    sim = ref + 0.1
    calculator = scores()
    assert np.isfinite(calculator.nSpatialScore(sim, ref)).all()
    result = calculator.Overall_Score(sim, ref)
    assert np.isnan(result[0, 0])
    assert np.isfinite(result[0, 1])


@pytest.mark.parametrize(
    "temporal,spatial,expected",
    [
        ([np.nan] * 4, 0.65, np.nan),
        ([np.nan] * 4, np.nan, np.nan),
        ([np.nan, 0.4, np.nan, np.nan], 0.7, 0.5),
        ([0.2, 0.4, 0.6, 0.8], np.nan, 0.48),
        ([0.2, 0.4, 0.6, 0.8], 0.6, 0.5),
    ],
)
def test_available_weights_unchanged(monkeypatch, temporal, spatial, expected):
    calculator = scores()
    for name, value in zip(
        ["nBiasScore", "nRMSEScore", "nPhaseScore", "nIavScore", "nSpatialScore"], temporal + [spatial]
    ):
        monkeypatch.setattr(calculator, name, lambda s, o, v=value: xr.DataArray(v))
    result = float(calculator.Overall_Score(None, None))
    if np.isnan(expected):
        assert np.isnan(result)
    else:
        assert result == pytest.approx(expected)
