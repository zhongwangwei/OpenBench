"""Flat-file variable resolution: decouple stored var name from configured name.

A fallback/convert (e.g. NEE from f_respc) relabels the saved variable to the
evaluation item, but downstream readers index by the stale configured varname
(f_respc). Readers must resolve robustly via the sole data variable.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from openbench.util.names import select_data_array


def _flat(varname: str) -> xr.Dataset:
    return xr.Dataset(
        {varname: (("time", "lat", "lon"), np.ones((2, 2, 2)))},
        coords={
            "time": xr.date_range("2000-01-01", periods=2, freq="MS", use_cftime=True),
            "lat": [10, 20],
            "lon": [30, 40],
        },
    )


def test_named_variable_found_case_insensitive():
    ds = _flat("Net_Ecosystem_Exchange")
    da = select_data_array(ds, "net_ecosystem_exchange")
    assert da.name == "Net_Ecosystem_Exchange"


def test_falls_back_to_sole_variable_when_name_absent():
    # The exact reproduction of the reported bug: file stores the item-named
    # variable, reader asks for the stale config varname 'f_respc'.
    ds = _flat("Net_Ecosystem_Exchange")
    da = select_data_array(ds, "f_respc")
    assert da.name == "Net_Ecosystem_Exchange"
    # raw indexing would raise the user's exact KeyError:
    with pytest.raises(KeyError):
        _ = ds["f_respc"]


def test_item_name_is_tried_before_sole_fallback():
    ds = _flat("Net_Ecosystem_Exchange")
    # preferred list form + item fallback both resolve
    da = select_data_array(ds, ["f_nee", "f_respc"], "Net_Ecosystem_Exchange")
    assert da.name == "Net_Ecosystem_Exchange"


def test_ambiguous_multivar_without_match_raises():
    ds = xr.Dataset({"a": ("x", [1.0]), "b": ("x", [2.0])}, coords={"x": [0]})
    with pytest.raises(KeyError):
        select_data_array(ds, "f_respc")


def test_masking_read_resolves_relabelled_variable(tmp_path):
    """apply_unified_mask must read a flat sim file whose variable was relabelled
    to the item even though sim_varname is still the source name."""
    import openbench.runner.masking as masking

    casedir = tmp_path
    (casedir / "data").mkdir()
    item, ref_src, sim_src = "Net_Ecosystem_Exchange", "FLUXCOM", "Case05"
    ref_vn, sim_vn = "NEE", "f_respc"  # configured names
    # ref flat: stored under its configured name; sim flat: relabelled to item
    _flat("NEE").to_netcdf(casedir / "data" / f"{item}_ref_{ref_src}_{ref_vn}.nc")
    _flat(item).to_netcdf(casedir / "data" / f"{item}_sim_{sim_src}_{sim_vn}.nc")

    info = {
        "casedir": str(casedir),
        "ref_varname": ref_vn,
        "sim_varname": sim_vn,
        "ref_data_type": "grid",
        "sim_data_type": "grid",
    }
    written = {}

    def fake_writer(ds, path, **kwargs):
        written["path"] = path

    # Should not raise "No variable named 'f_respc'"
    masking.apply_unified_mask(info, item, ref_src, sim_src, write_netcdf_atomic_fn=fake_writer)
    assert written  # the masked ref was written
