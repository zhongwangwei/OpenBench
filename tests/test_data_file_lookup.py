"""Data-file naming rules shared by preprocessing and the pre-run checks."""

from __future__ import annotations

import pytest

from openbench.data.file_lookup import missing_data_files, prefix_candidates, year_file_paths


def _touch(directory, *names):
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).touch()


def test_prefix_candidates_append_fallbacks():
    assert prefix_candidates("Case01_hist_", ["_cama_", "_unitcat_"]) == [
        "Case01_hist_",
        "Case01_hist_cama_",
        "Case01_hist_unitcat_",
    ]
    assert prefix_candidates("run", None) == ["run"]


def test_single_file_needs_exact_prefix_and_suffix(tmp_path):
    _touch(tmp_path, "CN05.1_Tm_1961_2021_daily_0P25.nc")

    assert missing_data_files(str(tmp_path), "CN05.1_Tm_", "_2021_daily_0P25", "Single", [2000, 2010]) == [
        str(tmp_path / "CN05.1_Tm__2021_daily_0P25.nc")
    ]
    assert missing_data_files(str(tmp_path), "CN05.1_Tm_1961", "_2021_daily_0P25", "single", [2000]) == []


def test_yearly_files_report_each_missing_year(tmp_path):
    _touch(tmp_path, "ERA5LAND_2000_t2m_1D_0p25.nc", "ERA5LAND_2001_t2m_1D_0p25.nc")

    assert missing_data_files(str(tmp_path), "ERA5LAND_", "_t2m_1D_0p25", "Year", [2000, 2001]) == []
    assert missing_data_files(str(tmp_path), "ERA5LAND_", "_t2m_1D_0p25", "Year", [2000, 2005]) == [
        str(tmp_path / "ERA5LAND_2005*_t2m_1D_0p25.nc")
    ]


def test_yearly_lookup_accepts_year_dirs_and_prefix_fallback(tmp_path):
    _touch(tmp_path / "2001", "case_2001-01.nc")
    _touch(tmp_path, "case_cama_2002.nc")

    assert year_file_paths(str(tmp_path), "case_", 2001, "") == [str(tmp_path / "2001" / "case_2001-01.nc")]
    assert missing_data_files(str(tmp_path), "case_", "", "Month", [2001, 2002], ["_cama_"]) == []
    assert missing_data_files(str(tmp_path), "case_", "", "Month", [2002]) == [str(tmp_path / "case_2002*.nc")]


def test_yearly_lookup_rejects_letters_between_year_and_suffix(tmp_path):
    _touch(tmp_path, "E_2004_extra_GLEAM_v4.2a.nc")

    assert missing_data_files(str(tmp_path), "E_", "_GLEAM_v4.2a", "Year", [2004]) == [
        str(tmp_path / "E_2004*_GLEAM_v4.2a.nc")
    ]


def test_missing_years_walk_tree_once(tmp_path, monkeypatch):
    from openbench.data import file_lookup

    _touch(tmp_path / "2001" / "01", "actual.nc4")
    walk = file_lookup.iter_netcdf_paths
    calls = []

    def counted(dirx):
        calls.append(dirx)
        return walk(dirx)

    monkeypatch.setattr(file_lookup, "iter_netcdf_paths", counted)
    assert len(missing_data_files(str(tmp_path), "missing_", "", "Year", [2001, 2002], ["_other_"])) == 2
    assert calls == [str(tmp_path)]
    assert year_file_paths(str(tmp_path), "actual", 2001, "") == [str(tmp_path / "2001" / "01" / "actual.nc4")]


def test_inventory_directory_link_cannot_recurse_into_ancestor(tmp_path):
    from openbench.data.file_lookup import netcdf_inventory

    _touch(tmp_path / "nested", "case_2001.nc")
    try:
        (tmp_path / "nested" / "loop").symlink_to(tmp_path, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable")
    assert netcdf_inventory(str(tmp_path)) == [str(tmp_path / "nested" / "case_2001.nc")]


def test_indexed_preflight_matches_runtime_lookup(tmp_path):
    from openbench.data.file_lookup import netcdf_inventory

    _touch(tmp_path, "case_2001_invalid.nc", "case[1]_2001.nc", "2002.NC4")
    _touch(tmp_path / "2001", "case_2001.nc")
    _touch(tmp_path / "nested", "case_2002.nc4")
    _touch(tmp_path / "2003" / "01", "case_.NC", "case[1]_.nc")
    inventory = netcdf_inventory(str(tmp_path))
    years = [2001, 2002, 2003, 2004]
    for prefix in ("", "case_", "case[1]_", "missing_"):
        for suffix in ("", "_x"):
            for fallbacks in (None, ["_alt_"]):
                expected = [
                    year
                    for year in years
                    if not any(
                        year_file_paths(str(tmp_path), candidate, year, suffix, inventory=inventory)
                        for candidate in prefix_candidates(prefix, fallbacks)
                    )
                ]
                missing = missing_data_files(
                    str(tmp_path), prefix, suffix, "Year", years, fallbacks, inventory=inventory
                )
                assert len(missing) == len(expected)
                assert all(str(year) in path for year, path in zip(expected, missing))


def test_year_lookup_preserves_flat_match_priority(tmp_path):
    _touch(tmp_path, "case_2001_invalid.nc")
    _touch(tmp_path / "nested", "case_2001.nc")
    # Runtime stops at the flat naming match before filtering its invalid letters.
    assert year_file_paths(str(tmp_path), "case_", 2001, "") == []


def test_string_prefix_fallback_is_one_fallback():
    assert prefix_candidates("case_", "_cama_") == ["case_", "case_cama_"]


def test_has_netcdf_stops_at_the_first_file(tmp_path):
    from openbench.data.file_lookup import has_netcdf

    _touch(tmp_path / "2001" / "01", "x.nc4")
    assert not has_netcdf(tmp_path)
    assert has_netcdf(tmp_path, recursive=True)
    assert not has_netcdf(tmp_path / "absent", recursive=True)


def test_find_nc_dir_keeps_year_layouts_and_ignores_numeric_or_empty_folders(tmp_path):
    from openbench.config.adapter import _find_nc_dir

    years = tmp_path / "years"
    _touch(years / "2001", "ref_2001.nc")
    _touch(years / "2002", "ref_2002.nc")
    assert _find_nc_dir(str(years), str(years), None) == str(years)

    numeric = tmp_path / "numeric"
    _touch(numeric / "1000", "ERA5_2001.nc")
    (numeric / "5000").mkdir()
    assert _find_nc_dir(str(numeric), str(numeric), None) == str(numeric / "1000")

    empty_year = tmp_path / "empty_year"
    (empty_year / "2001").mkdir(parents=True)
    _touch(empty_year / "0p25deg", "data_2001.nc")
    assert _find_nc_dir(str(empty_year), str(empty_year), None) == str(empty_year / "0p25deg")


def test_walk_finds_nested_files_when_direntry_has_no_inode(tmp_path, monkeypatch):
    """Windows DirEntry.stat() reports st_ino/st_dev as 0; nesting must still be walked."""
    import os

    from openbench.data import file_lookup

    real_scandir = os.scandir

    class Entry:
        def __init__(self, entry):
            self._entry = entry

        def __getattr__(self, name):
            return getattr(self._entry, name)

        def stat(self, *, follow_symlinks=True):
            values = list(self._entry.stat(follow_symlinks=follow_symlinks))
            values[1] = values[2] = 0
            return os.stat_result(values)

    class Scan:
        def __init__(self, path):
            self._scan = real_scandir(path)

        def __enter__(self):
            self._scan.__enter__()
            return (Entry(entry) for entry in self._scan)

        def __exit__(self, *exc):
            return self._scan.__exit__(*exc)

    _touch(tmp_path / "2001" / "01", "case_2001-01.nc")
    try:
        (tmp_path / "2001" / "01" / "loop").symlink_to(tmp_path, target_is_directory=True)
    except OSError:
        pass
    monkeypatch.setattr(file_lookup.os, "scandir", Scan)

    assert file_lookup.netcdf_inventory(str(tmp_path)) == [str(tmp_path / "2001" / "01" / "case_2001-01.nc")]
    assert file_lookup.has_netcdf(tmp_path, recursive=True)


def _case_insensitive(directory) -> bool:
    probe = directory / "case_probe"
    probe.touch()
    try:
        return (directory / "CASE_PROBE").exists()
    finally:
        probe.unlink()


def test_dated_folder_file_name_follows_the_filesystem_case_rules(tmp_path):
    _touch(tmp_path / "2001" / "01", "rain.nc")

    found = year_file_paths(str(tmp_path), "RAIN", 2001, "")
    missing = missing_data_files(str(tmp_path), "RAIN", "", "Year", [2001])

    # A case-insensitive filesystem (macOS, SMB) opens rain.nc as RAIN.nc; a case-sensitive one does not.
    if _case_insensitive(tmp_path):
        assert found == [str(tmp_path / "2001" / "01" / "rain.nc")] and missing == []
    else:
        assert found == [] and missing


def test_case_variant_is_not_a_match_when_both_names_exist(tmp_path):
    _touch(tmp_path / "2001", "RAIN.nc")
    if _case_insensitive(tmp_path):
        pytest.skip("needs a case-sensitive filesystem")
    _touch(tmp_path / "2001", "rain.nc")

    assert year_file_paths(str(tmp_path), "RAIN", 2001, "") == [str(tmp_path / "2001" / "RAIN.nc")]


@pytest.mark.parametrize("year_folder", ["2001", "2001-01"])
def test_year_layout_does_not_read_missing_years_from_a_sibling_branch(tmp_path, year_folder):
    from openbench.cli.check import data_file_findings
    from openbench.config.adapter import _find_nc_dir
    from openbench.data.file_lookup import unread_folders

    _touch(tmp_path / year_folder, "T_2001.nc")
    _touch(tmp_path / "0p25", "T_2001.nc", "T_2002.nc")

    data_dir = _find_nc_dir(str(tmp_path), str(tmp_path), None, "Year")

    assert data_dir == str(tmp_path)
    assert year_file_paths(data_dir, "T_", 2001, "") == [str(tmp_path / year_folder / "T_2001.nc")]
    assert year_file_paths(data_dir, "T_", 2002, "") == []
    assert unread_folders(data_dir) == ["0p25"]
    errors, warnings = data_file_findings(
        "Reference", data_dir, prefix="T_", suffix="", data_groupby="Year", years=[2001, 2002]
    )
    assert any("2002" in error for error in errors)
    assert any("0p25" in warning and "sub_dir" in warning for warning in warnings)


def test_flat_layout_with_unrelated_folders_reads_everything(tmp_path):
    from openbench.data.file_lookup import unread_folders

    _touch(tmp_path / "part_a", "T_2001.nc")
    _touch(tmp_path / "part_b", "T_2002.nc")

    assert year_file_paths(str(tmp_path), "T_", 2002, "") == [str(tmp_path / "part_b" / "T_2002.nc")]
    assert unread_folders(str(tmp_path)) == []


def test_case_variant_needs_the_filesystem_to_agree(tmp_path, monkeypatch):
    import os

    _touch(tmp_path / "2001" / "01", "rain.nc")
    # As on a case-sensitive filesystem: RAIN.nc is not the same file as rain.nc.
    monkeypatch.setattr(os.path, "samefile", lambda left, right: False)

    assert year_file_paths(str(tmp_path), "RAIN", 2001, "") == []
    assert missing_data_files(str(tmp_path), "RAIN", "", "Year", [2001])


def test_check_reports_years_read_from_several_branch_folders(tmp_path):
    from openbench.cli.check import data_file_findings
    from openbench.config.adapter import _find_nc_dir

    for resolution in ("0p25", "0p5"):
        _touch(tmp_path / resolution / "2001", "T_2001.nc")
    _touch(tmp_path / "0p25" / "2002", "T_2002.nc")

    data_dir = _find_nc_dir(str(tmp_path), str(tmp_path), None, "Year")
    errors, warnings = data_file_findings(
        "Reference", data_dir, prefix="T_", suffix="", data_groupby="Year", years=[2001, 2002]
    )

    assert data_dir == str(tmp_path)
    assert len(errors) == 1 and "2001" in errors[0] and "2002" not in errors[0]
    assert "0p25/, 0p5/" in errors[0] and "sub_dir" in errors[0]
    assert not warnings


def test_check_warns_when_one_year_is_split_across_folders(tmp_path):
    from openbench.cli.check import data_file_findings

    _touch(tmp_path / "part_a", "T_2001_01.nc")
    _touch(tmp_path / "part_b", "T_2001_07.nc")

    errors, warnings = data_file_findings(
        "Reference", str(tmp_path), prefix="T_", suffix="", data_groupby="Year", years=[2001]
    )

    assert not errors
    assert len(warnings) == 1 and "part_a/, part_b/" in warnings[0]


def test_compute_inputs_follow_the_branch_rule(tmp_path):
    import xarray as xr

    from openbench.config.adapter import _find_nc_dir
    from openbench.data.processing import DatasetProcessing

    (tmp_path / "2001-01").mkdir()
    (tmp_path / "0p25").mkdir()
    xr.Dataset({"rain": ("time", [1.0])}).to_netcdf(tmp_path / "2001-01" / "rain_2001.nc")
    xr.Dataset({"snow": ("time", [100.0])}).to_netcdf(tmp_path / "0p25" / "snow_2001.nc")
    processor = object.__new__(DatasetProcessing)
    processor._compute_dependency_varnames_for_file_lookup = lambda datasource: ["rain", "snow"]
    data_dir = _find_nc_dir(str(tmp_path), str(tmp_path), None, "Month")

    # snow exists only in the sibling branch: the inputs are incomplete, not mixed.
    assert processor._find_compute_dependency_files(data_dir, 2001, "sim") == []

    xr.Dataset({"snow": ("time", [2.0])}).to_netcdf(tmp_path / "2001-01" / "snow_2001.nc")
    assert processor._find_compute_dependency_files(data_dir, 2001, "sim") == [
        str(tmp_path / "2001-01" / "rain_2001.nc"),
        str(tmp_path / "2001-01" / "snow_2001.nc"),
    ]


def _compute_inputs(path, value):
    import xarray as xr

    path.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset({"rain": ("time", [value]), "snow": ("time", [value])}).to_netcdf(path)


@pytest.mark.parametrize("groupby", ["Year", "Single"])
def test_check_reports_compute_inputs_read_from_several_branches(tmp_path, groupby):
    from openbench.cli.check import data_file_findings

    _compute_inputs(tmp_path / "0p25" / "2001" / "input_2001.nc", 1.0)
    _compute_inputs(tmp_path / "0p5" / "2001" / "input_2001.nc", 100.0)

    errors, warnings = data_file_findings(
        "Simulation",
        str(tmp_path),
        prefix="runoff_",
        suffix="",
        data_groupby=groupby,
        years=[2001],
        compute="ds['rain'] + ds['snow']",
    )

    assert len(errors) == 1 and "compute inputs (rain, snow)" in errors[0] and "0p25/, 0p5/" in errors[0]
    assert ("for 2001" in errors[0]) == (groupby == "Year")
    assert not warnings


def test_years_read_through_compute_inputs_are_not_reported_missing(tmp_path):
    from openbench.cli.check import data_file_findings

    _compute_inputs(tmp_path / "input_2001.nc", 1.0)

    errors, warnings = data_file_findings(
        "Simulation",
        str(tmp_path),
        prefix="runoff_",
        suffix="",
        data_groupby="Year",
        years=[2001, 2002],
        compute="ds['rain'] + ds['snow']",
    )

    assert not errors
    assert len(warnings) == 1 and "for 2002" in warnings[0] and "2001" not in warnings[0]
    assert "no files holding all compute inputs (rain, snow)" in warnings[0]
