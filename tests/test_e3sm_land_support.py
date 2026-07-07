from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from openbench.data.registry.manager import RegistryManager
from openbench.data.sim_scanner import scan_simulation_roots


def _write_elm_history(path: Path, variables: tuple[str, ...] = ("EFLX_LH_TOT", "Rainf", "PSurf", "Wind")) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    times = pd.date_range("2000-01-01", periods=2, freq="MS")
    data_vars = {name: (["time", "lat", "lon"], np.zeros((2, 1, 1))) for name in variables}
    ds = xr.Dataset(data_vars, coords={"time": times, "lat": [10.0], "lon": [100.0]})
    ds.to_netcdf(path)


def test_elm_and_e3sm_land_profiles_registered(tmp_path: Path):
    mgr = RegistryManager(user_dir=tmp_path)

    for name in ("ELM", "E3SM", "E3SM_Land"):
        profile = mgr.get_model(name)
        assert profile is not None
        assert profile.variables["Latent_Heat"].varname == "EFLX_LH_TOT"
        assert profile.variables["Evapotranspiration"].varunit == "mm s-1"
        assert profile.variables["Precipitation"].varname == "Rainf"
        assert profile.variables["Surface_Pressure"].varname == "PSurf"
        assert profile.variables["Surface_Wind_Speed"].varname == "Wind"


def test_elm_profile_accepts_clm_style_downward_radiation_names(tmp_path: Path):
    profile = RegistryManager(user_dir=tmp_path).get_model("ELM")

    sw = profile.variables["Surface_Downward_SW_Radiation"]
    lw = profile.variables["Surface_Downward_LW_Radiation"]

    assert sw.varname == "SWdown"
    assert [fb.varname for fb in sw.fallbacks] == ["FSDS"]
    assert lw.varname == "LWdown"
    assert [fb.varname for fb in lw.fallbacks] == ["FLDS"]


def test_elm_net_radiation_compute_accepts_elm_forcing_names(tmp_path: Path):
    from openbench.data.compute import execute_compute

    profile = RegistryManager(user_dir=tmp_path).get_model("ELM")
    ds = xr.Dataset(
        {
            "SWdown": (["lat"], [100.0]),
            "FSR": (["lat"], [20.0]),
            "LWdown": (["lat"], [300.0]),
            "FIRE": (["lat"], [250.0]),
        },
        coords={"lat": [10.0]},
    )

    out = execute_compute(ds, profile.variables["Net_Radiation"].compute, "Net_Radiation")

    assert float(out.squeeze()) == 130.0


def test_scan_auto_resolves_elm_history_path(tmp_path: Path):
    root = tmp_path / "runs"
    history = root / "CaseA" / "elm" / "history"
    _write_elm_history(history / "CaseA.elm.h0.2000-01.nc")

    result = scan_simulation_roots([root], model_name="auto", case_depth=3)

    assert len(result.cases) == 1
    assert result.cases[0].model == "ELM"


def test_scan_auto_resolves_e3sm_land_path(tmp_path: Path):
    root = tmp_path / "runs"
    history = root / "E3SM" / "CaseA" / "history"
    _write_elm_history(history / "CaseA.elm.h0.2000-01.nc")

    result = scan_simulation_roots([root], model_name="auto", case_depth=3)

    assert len(result.cases) == 1
    assert result.cases[0].model == "E3SM"
