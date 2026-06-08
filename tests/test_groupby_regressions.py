from pathlib import Path

import numpy as np
import xarray as xr


def test_landcover_metric_clip_operates_on_current_subset():
    from openbench.core.landcover_groupby import _clip_metric_quantiles

    ds = xr.Dataset(
        {"bias": (("lat", "lon"), np.arange(100, dtype=float).reshape(10, 10))},
        coords={"lat": np.arange(10), "lon": np.arange(10)},
    )
    subset = ds.where(ds["bias"] >= 50)

    clipped = _clip_metric_quantiles(subset, "bias")

    assert np.nanmin(clipped["bias"].values) > np.nanmin(subset["bias"].values)
    assert np.nanmax(clipped["bias"].values) < np.nanmax(subset["bias"].values)


def test_groupby_metric_loops_clip_each_class_after_masking():
    lc_source = Path("src/openbench/core/landcover_groupby.py").read_text(encoding="utf-8")
    cz_source = Path("src/openbench/core/climatezone_groupby.py").read_text(encoding="utf-8")

    assert "ds1 = _clip_metric_quantiles(ds.where(IGBPtype == i), metric)" in lc_source
    assert "ds1 = _clip_metric_quantiles(ds.where(PFTtype == i), metric)" in lc_source
    assert "ds1 = _clip_metric_quantiles(ds.where(CZtype == i), metric)" in cz_source


def test_groupby_class_netcdf_outputs_are_bundled_by_statistic():
    lc_source = Path("src/openbench/core/landcover_groupby.py").read_text(encoding="utf-8")
    cz_source = Path("src/openbench/core/climatezone_groupby.py").read_text(encoding="utf-8")
    heatmap_source = Path("src/openbench/visualization/Fig_LC_based_heat_map.py").read_text(encoding="utf-8")

    assert "groupby_class_netcdf_filename" not in lc_source
    assert "groupby_class_netcdf_filename" not in cz_source
    assert "_write_netcdf_atomic(ds1" not in lc_source
    assert "_write_netcdf_atomic(ds1" not in cz_source
    assert "__classes.nc" in lc_source
    assert "__classes.nc" in cz_source
    assert "_open_groupby_class_distribution(option, metric)" in heatmap_source
