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


def _flat_series(varname: str, scale: float) -> xr.Dataset:
    return xr.Dataset(
        {varname: (("time", "lat", "lon"), np.arange(1.0, 17.0).reshape(4, 2, 2) * scale)},
        coords={
            "time": xr.date_range("2000-01-01", periods=4, freq="MS"),
            "lat": [10.0, 20.0],
            "lon": [30.0, 40.0],
        },
    )


def test_smpi_grid_comparison_reads_relabelled_sim_variable(tmp_path, monkeypatch):
    import openbench.core.comparison as comparison_module

    item, ref_src, sim_src = "Net_Ecosystem_Exchange", "FLUXCOM", "Case05"
    (tmp_path / "data").mkdir()
    _flat_series("NEE", scale=1.0).to_netcdf(tmp_path / "data" / f"{item}_ref_{ref_src}_NEE.nc")
    _flat_series(item, scale=1.5).to_netcdf(tmp_path / "data" / f"{item}_sim_{sim_src}_f_respc.nc")
    monkeypatch.setattr(
        comparison_module, "make_scenarios_comparison_Single_Model_Performance_Index", lambda *a, **k: None
    )
    processor = comparison_module.ComparisonProcessing(
        {
            "general": {
                "basename": "case",
                "basedir": str(tmp_path),
                "compare_grid_res": 0.5,
                "compare_tim_res": "Month",
                "weight": "none",
                "num_cores": 1,
            }
        },
        [],
        [],
    )
    sim_nml = {
        "general": {f"{item}_sim_source": [sim_src]},
        item: {f"{sim_src}_data_type": "grid", f"{sim_src}_varname": "f_respc"},
    }
    ref_nml = {
        "general": {f"{item}_ref_source": ref_src},
        item: {f"{ref_src}_data_type": "grid", f"{ref_src}_varname": "NEE"},
    }

    processor.scenarios_Single_Model_Performance_Index_comparison(str(tmp_path), sim_nml, ref_nml, [item], [], [], {})

    smpi_dir = tmp_path / "comparisons" / "Single_Model_Performance_Index"
    assert (smpi_dir / f"{item}_ref_{ref_src}_sim_{sim_src}_SMPI_grid.nc").exists()


def test_core_flat_file_readers_do_not_hard_index_configured_varname():
    import re
    from pathlib import Path

    import openbench.core

    raw_read = re.compile(r"(?:\bds|_ds|_file)\[\s*f?\"?\{?(?:sim|ref)_varname\d?\}?\"?\s*\]")
    core_dir = Path(openbench.core.__file__).parent
    offenders = [
        f"{path.relative_to(core_dir)}:{lineno}"
        for path in sorted(core_dir.rglob("*.py"))
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if raw_read.search(line)
    ]

    assert offenders == []
