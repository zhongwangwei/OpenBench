import numpy as np
import pytest
import xarray as xr

from openbench.data._processing_time_core import TimeCoreMixin
from openbench.data._processing_transforms import ProcessingTransformMixin
from openbench.data.compute import ComputeError
from openbench.data.coordinates import COORDINATE_MAP_WITH_VERTICAL
from openbench.data.registry.manager import RegistryManager


class _Harness(ProcessingTransformMixin, TimeCoreMixin):
    coordinate_map = COORDINATE_MAP_WITH_VERTICAL


def test_cable_model_profile_registered(tmp_path):
    profile = RegistryManager(user_dir=tmp_path).get_model("CABLE")

    assert profile is not None
    assert profile.variables["Latent_Heat"].varname == "Qle"
    assert profile.variables["Total_Runoff"].compute == "ds['Qs'] + ds['Qsb']"


def test_cable_xy_coordinates_do_not_conflict_with_aux_latlon():
    ds = xr.Dataset(
        {"Qle": (("time", "y", "x"), np.zeros((1, 2, 2)))},
        coords={
            "time": [np.datetime64("2001-01-16")],
            "x": [100.0, 101.0],
            "y": [-1.0, 0.0],
            "longitude": (("y", "x"), [[100.0, 101.0], [100.0, 101.0]]),
            "latitude": (("y", "x"), [[-1.0, -1.0], [0.0, 0.0]]),
        },
    )

    out = _Harness().check_coordinate(ds)

    assert out["Qle"].dims == ("time", "lat", "lon")
    assert np.allclose(out["lon"], [100.0, 101.0])
    assert np.allclose(out["lat"], [-1.0, 0.0])
    assert {"longitude", "latitude"} <= set(out.coords)


def test_cable_patch_output_uses_patchfrac_weighted_grid_mean():
    h = _Harness()
    h.item = "Latent_Heat"
    h.sim_source = "CableCase"
    h.CableCase_model = "CABLE"
    h.sim_varname = ["Qle"]

    ds = xr.Dataset(
        {
            "Qle": (("time", "y", "x", "patch"), [[[[10.0, 20.0]]]]),
            "patchfrac": (("y", "x", "patch"), [[[0.25, 0.75]]]),
        },
        coords={"time": [np.datetime64("2001-01-16")], "y": [0.0], "x": [100.0], "patch": [0, 1]},
    )

    out = h.apply_custom_filter("sim", ds, ["Qle"])

    assert "patch" not in out.dims
    assert float(out.squeeze()) == 17.5


def test_cable_compute_failure_does_not_fall_back_to_stale_direct_variable():
    h = _Harness()
    h.item = "Total_Runoff"
    h.sim_source = "CableCase"
    h.CableCase_model = "CABLE"
    h.sim_varname = ["Total_Runoff"]

    ds = xr.Dataset(
        {
            "Qs": (["lat"], [1.0]),
            "Total_Runoff": (["lat"], [999.0]),
        },
        coords={"lat": [10.0]},
    )

    with pytest.raises(ComputeError, match="Qsb"):
        h.apply_custom_filter("sim", ds, ["Total_Runoff"])
