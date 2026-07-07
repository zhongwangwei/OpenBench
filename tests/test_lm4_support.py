from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from openbench.data.compute import execute_compute
from openbench.data.registry.manager import RegistryManager
from openbench.data.registry.scanner import inspect_nc_file
from openbench.data.sim_scanner import scan_simulation_roots


def _write_lm4_history(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    times = pd.date_range("2000-01-01", periods=2, freq="MS")
    ds = xr.Dataset(
        {
            "evap": (["time", "grid_index"], np.zeros((2, 2))),
            "hflsLut": (["time", "grid_index"], np.zeros((2, 2))),
            "sens": (["time", "grid_index"], np.zeros((2, 2))),
            "runf": (["time", "grid_index"], np.zeros((2, 2))),
            "gpp": (["time", "grid_index"], np.zeros((2, 2))),
            "lai": (["time", "grid_index"], np.zeros((2, 2))),
        },
        coords={
            "time": times,
            "grid_index": [0, 1],
            "geolon_t": ("grid_index", [100.0, 101.0]),
            "geolat_t": ("grid_index", [10.0, 11.0]),
        },
    )
    ds.to_netcdf(path)


def test_lm4_profile_registered_with_nuopc_aliases(tmp_path: Path):
    mgr = RegistryManager(user_dir=tmp_path)

    for name in ("LM4", "LM4-NUOPC", "LM4_NUOPC", "GFDL_LM4"):
        profile = mgr.get_model(name)
        assert profile is not None
        assert profile.name == "LM4"

    profile = mgr.get_model("LM4")
    assert profile.variables["Evapotranspiration"].varname == "evap"
    assert profile.variables["Latent_Heat"].varname == "hflsLut"
    assert profile.variables["Total_Runoff"].compute == "ds['runf'] if 'runf' in ds else ds['wroff'] + ds['sroff']"


def test_lm4_compute_mappings_cover_native_flux_conventions(tmp_path: Path):
    profile = RegistryManager(user_dir=tmp_path).get_model("LM4")
    ds = xr.Dataset(
        {
            "fsw": (["grid_index"], [80.0]),
            "flw": (["grid_index"], [20.0]),
            "albedo_dif": (["band", "grid_index"], [[0.2], [0.4]]),
            "gpp": (["grid_index"], [31_536_000.0], {"units": "kg C/(m2 year)"}),
        },
        coords={"band": [1, 2], "grid_index": [0]},
    )

    net = execute_compute(ds, profile.variables["Net_Radiation"].compute, "Net_Radiation")
    albedo = execute_compute(ds, profile.variables["Surface_Albedo"].compute, "Surface_Albedo")
    gpp = execute_compute(ds, profile.variables["Gross_Primary_Productivity"].compute, "Gross_Primary_Productivity")

    assert float(net.squeeze()) == 100.0
    assert np.isclose(float(albedo.squeeze()), 0.3)
    assert float(gpp.squeeze()) == 1.0


def test_lm4_unstructured_history_scans_as_grid_and_auto_model(tmp_path: Path):
    root = tmp_path / "runs"
    history = root / "LM4-NUOPC" / "history"
    _write_lm4_history(history / "lm4_land_200001.nc")

    info = inspect_nc_file(history)
    assert info["detected_data_type"] == "grid"
    assert "geolon_t" not in {item["name"] for item in info["all_data_vars"]}
    assert "geolat_t" not in {item["name"] for item in info["all_data_vars"]}

    result = scan_simulation_roots([root], model_name="auto", case_depth=3)

    assert len(result.cases) == 1
    assert result.cases[0].model == "LM4"
    assert result.cases[0].data_type == "grid"
