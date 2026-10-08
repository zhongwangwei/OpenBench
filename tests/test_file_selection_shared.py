"""Runtime and preflight share variable-aware file selection."""

import pytest
import xarray as xr

from openbench.data import file_lookup


def _write(tmp_path, name, variables):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset({name: ("time", [1.0]) for name in variables}).to_netcdf(path)
    return str(path)


def test_existing_unrelated_file_falls_back_to_compute_inputs(tmp_path):
    _write(tmp_path, "runoff_2001.nc", ["unrelated"])
    inputs = [_write(tmp_path, f"{branch}/input_2001.nc", ["rain", "snow"]) for branch in ("0p25", "0p5")]
    selected, compute = file_lookup.select_data_files(
        str(tmp_path),
        "runoff_",
        "",
        2001,
        candidate_varnames=["runoff", "rain", "snow"],
        dependencies=["rain", "snow"],
    )
    assert compute and selected == inputs
    assert file_lookup.mixed_branches(str(tmp_path), {2001: selected}, dependencies=["rain", "snow"])[2001][1]


@pytest.mark.parametrize("same_name", [True, False])
def test_complementary_inputs_are_not_ambiguous(tmp_path, same_name):
    files = [_write(tmp_path, f"{var}/{('input' if same_name else var)}_2001.nc", [var]) for var in ("rain", "snow")]
    assert file_lookup.mixed_branches(str(tmp_path), {2001: files}, dependencies=["rain", "snow"]) == {}


@pytest.mark.parametrize("same_name", [True, False])
def test_overlapping_inputs_are_ambiguous(tmp_path, same_name):
    files = [
        _write(tmp_path, f"{branch}/{('input' if same_name else branch)}_2001.nc", ["rain"]) for branch in ("a", "b")
    ]
    assert file_lookup.mixed_branches(str(tmp_path), {2001: files}, dependencies=["rain"])[2001] == (
        ("a", "b"),
        same_name,
    )


def test_single_checks_each_extension_and_prefers_variable_aware_fallback(tmp_path):
    _write(tmp_path, "run.nc", ["rain"])
    _write(tmp_path, "run_alt.nc", ["other"])
    wanted = _write(tmp_path, "run_alt.nc4", ["rain"])
    assert file_lookup.select_data_files(
        str(tmp_path), "run", "", None, prefix_fallback=["_alt"], candidate_varnames=["rain"]
    ) == ([wanted], False)


def test_preindexed_year_matches_do_not_scan_again(tmp_path, monkeypatch):
    wanted = _write(tmp_path, "run_2001.nc", ["rain"])
    monkeypatch.setattr(file_lookup, "year_file_paths", lambda *args, **kwargs: pytest.fail("rescanned"))
    assert file_lookup.select_data_files(
        str(tmp_path), "run_", "", 2001, candidate_varnames=["rain"], named_matches={"run_": [wanted]}
    ) == ([wanted], False)


def test_unusable_named_files_remain_fallback_if_compute_missing(tmp_path):
    wanted = _write(tmp_path, "run_2001.nc", ["other"])
    assert file_lookup.select_data_files(
        str(tmp_path), "run_", "", 2001, candidate_varnames=["rain"], dependencies=["rain"]
    ) == ([wanted], False)


def test_same_basename_only_counts_when_dependency_overlaps(tmp_path):
    files = [
        _write(tmp_path, "a/input_2001.nc", ["rain"]),
        _write(tmp_path, "b/input_2001.nc", ["snow"]),
        _write(tmp_path, "b/rain_2001.nc", ["rain"]),
    ]
    assert file_lookup.mixed_branches(str(tmp_path), {2001: files}, dependencies=["rain", "snow"])[2001] == (
        ("a", "b"),
        False,
    )


def test_unreadable_header_preserves_named_selection_and_conservative_conflict(tmp_path):
    paths = []
    for branch in ("a", "b"):
        path = tmp_path / branch / "input_2001.nc"
        path.parent.mkdir()
        path.write_text("unreadable")
        paths.append(str(path))
    assert file_lookup.select_data_files(
        str(tmp_path),
        "input_",
        "",
        2001,
        candidate_varnames=["rain"],
        inventory=paths,
    ) == (paths, False)
    assert file_lookup.mixed_branches(str(tmp_path), {2001: paths}, dependencies=["rain"])[2001][1]


def _count_header_reads(monkeypatch):
    import xarray as xr

    from openbench.data import file_lookup

    file_lookup._variable_names.cache_clear()
    reads = []
    real = xr.open_dataset

    def counting(path, *args, **kwargs):
        reads.append(str(path))
        return real(path, *args, **kwargs)

    monkeypatch.setattr(xr, "open_dataset", counting)
    return reads


def test_one_candidate_group_without_compute_opens_no_header(tmp_path, monkeypatch):
    from openbench.data.file_lookup import select_data_files

    (tmp_path / "runoff_2001.nc").touch()
    reads = _count_header_reads(monkeypatch)

    files, used_compute = select_data_files(str(tmp_path), "runoff_", "", 2001, candidate_varnames=["runoff"])

    assert files == [str(tmp_path / "runoff_2001.nc")] and not used_compute
    assert reads == []


def test_prefix_fallback_still_reads_headers_to_choose(tmp_path, monkeypatch):
    import xarray as xr

    from openbench.data.file_lookup import select_data_files

    xr.Dataset({"other": ("t", [1.0])}).to_netcdf(tmp_path / "case_2001.nc")
    xr.Dataset({"runoff": ("t", [1.0])}).to_netcdf(tmp_path / "case_cama_2001.nc")
    reads = _count_header_reads(monkeypatch)

    files, _ = select_data_files(
        str(tmp_path), "case_", "", 2001, prefix_fallback=["_cama_"], candidate_varnames=["runoff"]
    )

    assert files == [str(tmp_path / "case_cama_2001.nc")]
    assert reads


def test_unreadable_header_is_read_once(tmp_path, monkeypatch):
    from openbench.data.file_lookup import variable_names

    broken = tmp_path / "broken_2001.nc"
    broken.write_text("not netcdf")
    reads = _count_header_reads(monkeypatch)

    assert variable_names(str(broken)) is None
    assert variable_names(str(broken)) is None
    assert reads == [str(broken)]
