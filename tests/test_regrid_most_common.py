"""Most-common-label regridding against a brute-force cell-by-cell mode."""

import numpy as np
import pytest
import xarray as xr

import openbench.data.regrid  # noqa: F401  register accessor


def _brute_force_mode(values, src_lat, src_lon, tgt_lat, tgt_lon, labels, res):
    out = np.full((tgt_lat.size, tgt_lon.size), np.nan)
    for i, lat in enumerate(tgt_lat):
        rows = (src_lat >= lat - res / 2) & (src_lat < lat + res / 2)
        for j, lon in enumerate(tgt_lon):
            cols = (src_lon >= lon - res / 2) & (src_lon < lon + res / 2)
            cell = values[np.ix_(rows, cols)].ravel()
            counts = [np.count_nonzero(cell == label) for label in labels]
            out[i, j] = labels[int(np.argmax(counts))]  # ties: first label listed
    inside_lat = (tgt_lat >= src_lat.min()) & (tgt_lat <= src_lat.max())
    inside_lon = (tgt_lon >= src_lon.min()) & (tgt_lon <= src_lon.max())
    out[~(inside_lat[:, None] & inside_lon[None, :])] = np.nan
    return out


def test_most_common_matches_brute_force_mode_with_ties_unlisted_labels_and_time():
    rng = np.random.default_rng(3)
    src_lat = np.arange(0.125, 6, 0.25)
    src_lon = np.arange(10.125, 18, 0.25)
    # Label 9 is not listed and must be ignored; small label sets make ties common.
    values = rng.choice([1, 2, 3, 9], size=(2, src_lat.size, src_lon.size))
    data = xr.DataArray(
        values,
        dims=("time", "lat", "lon"),
        coords={"time": [0, 1], "lat": src_lat, "lon": src_lon},
        name="landcover",
    )
    # The target reaches one cell beyond the source on each side.
    tgt_lat = np.arange(-0.5, 7, 1.0)
    tgt_lon = np.arange(9.5, 19, 1.0)
    labels = np.array([3, 1, 2])

    result = data.regrid.most_common(xr.Dataset(coords={"lat": tgt_lat, "lon": tgt_lon}), values=labels)

    assert result.dims == ("time", "lat", "lon")
    for step in range(2):
        expected = _brute_force_mode(values[step], src_lat, src_lon, tgt_lat, tgt_lon, labels, 1.0)
        np.testing.assert_array_equal(result.isel(time=step).values, expected)


def test_most_common_rejects_non_integer_labels():
    data = xr.DataArray(np.ones((2, 2)), dims=("lat", "lon"), coords={"lat": [0.5, 1.5], "lon": [0.5, 1.5]})

    with pytest.raises(ValueError, match="integer datatype"):
        data.regrid.most_common(xr.Dataset(coords={"lat": [1.0], "lon": [1.0]}), values=np.array([1]))
