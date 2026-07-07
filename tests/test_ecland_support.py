from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from openbench.data.compute import execute_compute
from openbench.data.registry.manager import RegistryManager
from openbench.data.sim_scanner import scan_simulation_roots


def _write_ecland_history(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    times = pd.date_range("2000-01-01", periods=2, freq="MS")
    ds = xr.Dataset(
        {
            "Qle": (["time", "lat", "lon"], np.zeros((2, 2, 2))),
            "Qh": (["time", "lat", "lon"], np.zeros((2, 2, 2))),
            "SWnet": (["time", "lat", "lon"], np.zeros((2, 2, 2))),
            "LWnet": (["time", "lat", "lon"], np.zeros((2, 2, 2))),
            "Evap": (["time", "lat", "lon"], np.zeros((2, 2, 2))),
            "Qs": (["time", "lat", "lon"], np.zeros((2, 2, 2))),
            "Qsb": (["time", "lat", "lon"], np.zeros((2, 2, 2))),
        },
        coords={"time": times, "lat": [10.0, 11.0], "lon": [100.0, 101.0]},
    )
    ds.to_netcdf(path)


def test_ecland_profile_registered_with_ecmwf_aliases(tmp_path: Path):
    mgr = RegistryManager(user_dir=tmp_path)

    for name in ("ECLand", "ecLand", "EC-Land", "ECMWF_ECLAND"):
        profile = mgr.get_model(name)
        assert profile is not None
        assert profile.name == "ECLand"

    profile = mgr.get_model("ECLand")
    assert profile.variables["Latent_Heat"].varname == "Qle"
    assert profile.variables["Surface_Net_SW_Radiation"].varname == "SWnet"
    assert profile.variables["Total_Runoff"].compute == "ds['Qs'] + ds['Qsb']"


def test_ecland_compute_mappings_cover_native_output_conventions(tmp_path: Path):
    profile = RegistryManager(user_dir=tmp_path).get_model("ECLand")
    ds = xr.Dataset(
        {
            "SWnet": (["lat"], [80.0]),
            "LWnet": (["lat"], [20.0]),
            "Qg": (["lat"], [-15.0]),
            "SWE": (["lat"], [100.0]),
            "snowdens": (["lat"], [250.0]),
            "SoilMoist": (["nlevs", "lat"], [[14.0], [42.0], [144.0], [378.0]]),
            "SoilTemp": (["nlevs", "lat"], [[280.0], [281.0], [282.0], [283.0]]),
        },
        coords={"nlevs": [1, 2, 3, 4], "lat": [10.0]},
    )

    net = execute_compute(ds, profile.variables["Net_Radiation"].compute, "Net_Radiation")
    ground = execute_compute(ds, profile.variables["Ground_Heat"].compute, "Ground_Heat")
    snow_depth = execute_compute(ds, profile.variables["Snow_Depth"].compute, "Snow_Depth")
    surface_sm = execute_compute(
        ds,
        profile.variables["Surface_Soil_Moisture"].compute,
        "Surface_Soil_Moisture",
    )
    root_sm = execute_compute(
        ds,
        profile.variables["Root_Zone_Soil_Moisture"].compute,
        "Root_Zone_Soil_Moisture",
    )
    surface_temp = execute_compute(
        ds,
        profile.variables["Surface_Soil_Temperature"].compute,
        "Surface_Soil_Temperature",
    )

    assert float(net.squeeze()) == 100.0
    assert float(ground.squeeze()) == 15.0
    assert float(snow_depth.squeeze()) == 0.4
    assert float(surface_sm.squeeze()) == 0.2
    assert float(root_sm.squeeze()) == 0.2
    assert float(surface_temp.squeeze()) == 280.0


def test_scan_auto_resolves_ecland_output_path(tmp_path: Path):
    root = tmp_path / "runs"
    output = root / "ECland" / "output"
    _write_ecland_history(output / "o_efl_200001.nc")

    result = scan_simulation_roots([root], model_name="auto", case_depth=3)

    assert len(result.cases) == 1
    assert result.cases[0].model == "ECLand"
    assert result.cases[0].data_type == "grid"
