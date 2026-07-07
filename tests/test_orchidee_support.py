from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from openbench.data.compute import ComputeError, execute_compute
from openbench.data.registry.manager import RegistryManager
from openbench.data.sim_scanner import scan_simulation_roots


def _write_orchidee_history(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    times = pd.date_range("2000-01-01", periods=2, freq="MS")
    ds = xr.Dataset(
        {
            "fluxlat": (["time", "lat", "lon"], np.zeros((2, 1, 1))),
            "fluxsens": (["time", "lat", "lon"], np.zeros((2, 1, 1))),
            "evap": (["time", "lat", "lon"], np.zeros((2, 1, 1))),
            "rain": (["time", "lat", "lon"], np.zeros((2, 1, 1))),
        },
        coords={"time": times, "lat": [10.0], "lon": [100.0]},
    )
    ds.to_netcdf(path)


def test_orchidee_profile_registered_with_2_0_alias(tmp_path: Path):
    mgr = RegistryManager(user_dir=tmp_path)

    for alias in ("ORCHIDEE_2_0", "ORCHIDEE_2_2", "ORCHIDEE_3", "ORCHIDEE_4", "ORCHIDEE_4_3"):
        profile = mgr.get_model(alias)
        assert profile is not None
        assert profile.name == "ORCHIDEE"

    profile = mgr.get_model("ORCHIDEE")
    assert profile.variables["Latent_Heat"].varname == "fluxlat"
    assert profile.variables["Sensible_Heat"].varname == "fluxsens"
    assert profile.variables["Evapotranspiration"].varname == "evap"
    assert profile.variables["Precipitation"].varname == "rain"
    assert profile.variables["Surface_Downward_SW_Radiation"].varname == "swdown"


def test_orchidee_weighted_pft_compute_collapses_to_grid(tmp_path: Path):
    profile = RegistryManager(user_dir=tmp_path).get_model("ORCHIDEE")
    ds = xr.Dataset(
        {
            "lai": (["nvm", "lat"], [[1.0], [3.0]]),
            "maxvegetfrac": (["nvm", "lat"], [[0.25], [0.75]]),
        },
        coords={"nvm": [1, 2], "lat": [10.0]},
    )

    out = execute_compute(ds, profile.variables["Leaf_Area_Index"].compute, "Leaf_Area_Index")

    assert out.dims == ("lat",)
    assert float(out.squeeze()) == 2.5


def test_orchidee_pft_compute_requires_vegetation_weights(tmp_path: Path):
    profile = RegistryManager(user_dir=tmp_path).get_model("ORCHIDEE")
    ds = xr.Dataset({"gpp": (["nvm", "lat"], [[1.0], [3.0]])}, coords={"nvm": [1, 2], "lat": [10.0]})

    with pytest.raises(ComputeError, match="maxvegetfrac"):
        execute_compute(ds, profile.variables["Gross_Primary_Productivity"].compute, "Gross_Primary_Productivity")


def test_orchidee_scalar_gpp_kg_c_is_converted_to_gc(tmp_path: Path):
    profile = RegistryManager(user_dir=tmp_path).get_model("ORCHIDEE")
    gpp = xr.DataArray([0.002], dims=["lat"], attrs={"units": "kg C m-2 s-1"})
    ds = xr.Dataset({"gpp": gpp}, coords={"lat": [10.0]})

    out = execute_compute(ds, profile.variables["Gross_Primary_Productivity"].compute, "Gross_Primary_Productivity")

    assert float(out.squeeze()) == 2.0


def test_orchidee_lai_prefers_native_lai_mean(tmp_path: Path):
    profile = RegistryManager(user_dir=tmp_path).get_model("ORCHIDEE")
    ds = xr.Dataset({"LAImean": (["lat"], [1.7])}, coords={"lat": [10.0]})

    out = execute_compute(ds, profile.variables["Leaf_Area_Index"].compute, "Leaf_Area_Index")

    assert float(out.squeeze()) == 1.7


def test_orchidee_albedo_prefers_broadband_albedo(tmp_path: Path):
    profile = RegistryManager(user_dir=tmp_path).get_model("ORCHIDEE")
    ds = xr.Dataset(
        {
            "Albedo": (["lat"], [0.2]),
            "albedo": (["albtyp", "lat"], [[0.1], [0.9]]),
        },
        coords={"albtyp": [0, 1], "lat": [10.0]},
    )

    out = execute_compute(ds, profile.variables["Surface_Albedo"].compute, "Surface_Albedo")

    assert float(out.squeeze()) == 0.2


def test_orchidee_mrsos_upper_10cm_water_column_to_fraction(tmp_path: Path):
    profile = RegistryManager(user_dir=tmp_path).get_model("ORCHIDEE")
    ds = xr.Dataset({"mrsos": (["lat"], [25.0])}, coords={"lat": [10.0]})

    out = execute_compute(ds, profile.variables["Surface_Soil_Moisture"].compute, "Surface_Soil_Moisture")

    assert float(out.squeeze()) == 0.25


def test_orchidee_total_runoff_sums_native_components(tmp_path: Path):
    profile = RegistryManager(user_dir=tmp_path).get_model("ORCHIDEE")
    ds = xr.Dataset({"runoff": (["lat"], [2.0]), "drainage": (["lat"], [3.0])}, coords={"lat": [10.0]})

    out = execute_compute(ds, profile.variables["Total_Runoff"].compute, "Total_Runoff")

    assert float(out.squeeze()) == 5.0


def test_scan_auto_resolves_orchidee_2_0_path(tmp_path: Path):
    root = tmp_path / "runs"
    history = root / "ORCHIDEE_2_0" / "history"
    _write_orchidee_history(history / "sechiba_history.nc")

    result = scan_simulation_roots([root], model_name="auto", case_depth=2)

    assert len(result.cases) == 1
    assert result.cases[0].model == "ORCHIDEE"
