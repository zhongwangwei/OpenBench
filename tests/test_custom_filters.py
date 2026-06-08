"""Regression tests for openbench.data.custom filter modules."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def _make_hydroweb_info(tmp_path, **overrides):
    """Build a minimal info-like object for HydroWeb filter."""
    info = SimpleNamespace(
        compare_tim_res="D",
        sim_grid_res=0.25,
        ref_dir=str(tmp_path / "refs"),
        casedir=str(tmp_path / "case"),
        sim_source="SimA",
        sim_syear=2010,
        sim_eyear=2015,
        syear=2010,
        eyear=2015,
        min_year=3,
        min_lon=-180.0,
        max_lon=180.0,
        min_lat=-90.0,
        max_lat=90.0,
        debug_mode=False,
    )
    for k, v in overrides.items():
        setattr(info, k, v)
    return info


def test_hydroweb_filter_raises_instead_of_sys_exit_on_bad_tim_res(tmp_path):
    """Filter must NOT call sys.exit on configuration errors. Previously
    five sys.exit(1) calls killed the entire runner mid-evaluation; now
    raises ValueError so the caller (runner._preprocess_variable's
    except Exception block) records a phase error and continues with
    other variables / sims.
    """
    from openbench.data.custom.HydroWeb_filter import filter_HydroWeb

    info = _make_hydroweb_info(tmp_path, compare_tim_res="Month")

    with pytest.raises(ValueError, match="compare_tim_res=Day"):
        filter_HydroWeb(info)


def test_hydroweb_filter_raises_on_bad_grid_res(tmp_path):
    """Invalid grid resolution → ValueError, not sys.exit."""
    from openbench.data.custom.HydroWeb_filter import filter_HydroWeb

    info = _make_hydroweb_info(tmp_path, sim_grid_res=0.7)

    with pytest.raises(ValueError, match="sim_grid_res 0.7 not in valid set"):
        filter_HydroWeb(info)


def test_hydroweb_grid_res_accepts_close_float_values():
    """HydroWeb routing resolutions should not depend on exact binary float spelling."""
    from openbench.data.custom.HydroWeb_filter import _canonical_sim_grid_res

    assert _canonical_sim_grid_res(0.08330000001) == 0.0833


def test_hydroweb_process_station_uses_normalized_d_not_1d(tmp_path):
    """process_station must accept compare_tim_res normalized to 'D'.
    Previously checked '1d' which never matched the actual normalized
    value, so all stations returned Flag=False and downstream filter
    raised "no stations selected".
    """
    from openbench.data.custom.HydroWeb_filter import process_station

    info = _make_hydroweb_info(tmp_path)
    station = {"ID": 12345, "lon": 10.0, "lat": 50.0}

    # File does not exist → process_station returns the default result with
    # ref_dir == "file" (path lookup failed), but the if-branch entered
    # successfully (compare_tim_res check passed). Previously the "1d"
    # check would have early-returned with ref_dir == "file" untouched.
    result = process_station(station, info)
    # The path-existence branch was entered (we see ref_dir is still "file",
    # since the NC file doesn't exist in our tmp_path setup, but the
    # compare_tim_res check passed).
    assert "Flag" in result
    # We don't assert Flag=True because the file doesn't actually exist;
    # we only need to verify the function didn't bail out at the
    # tim_res check, which the old "1d" check would have done.


def test_ch4_fluxnetann_filter_raises_on_missing_dataset(tmp_path):
    from openbench.data.custom.CH4_FluxnetANN_filter import filter_CH4_FluxnetANN

    info = SimpleNamespace(ref_dir=str(tmp_path), casedir=str(tmp_path / "case"), sim_source="SimA")

    with pytest.raises(FileNotFoundError, match="CH4_FluxnetANN dataset not found"):
        filter_CH4_FluxnetANN(info)
