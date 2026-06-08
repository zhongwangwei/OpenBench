"""Tests for time-resolution detection in the registry scanner."""

from pathlib import Path

from openbench.data.registry.scanner import _detect_tim_res, scan_reference_directory


def test_detect_tim_res_prefers_3hour_over_hourly(tmp_path: Path):
    dataset_dir = tmp_path / "sample_3hourly"
    dataset_dir.mkdir()
    (dataset_dir / "sample_3hourly.nc").write_text("")

    assert _detect_tim_res(dataset_dir) == "3Hour"


def test_detect_tim_res_does_not_match_3h_inside_larger_number(tmp_path: Path):
    dataset_dir = tmp_path / "13hourly_archive"
    dataset_dir.mkdir()
    (dataset_dir / "sample_13hourly.nc").write_text("")

    assert _detect_tim_res(dataset_dir) == "Hour"


def test_detect_tim_res_does_not_match_6h_inside_larger_number(tmp_path: Path):
    dataset_dir = tmp_path / "16hourly_archive"
    dataset_dir.mkdir()
    (dataset_dir / "sample_16hourly.nc").write_text("")

    assert _detect_tim_res(dataset_dir) == "Hour"


def test_detect_tim_res_detects_hourly(tmp_path: Path):
    dataset_dir = tmp_path / "sample_hourly"
    dataset_dir.mkdir()
    (dataset_dir / "sample_hourly.nc").write_text("")

    assert _detect_tim_res(dataset_dir) == "Hour"


def test_scan_reference_directory_registers_unprofiled_grid_composite(tmp_path: Path):
    ref_root = tmp_path / "ref"

    direct = ref_root / "Grid" / "LowRes" / "Water" / "VarA" / "DatasetA"
    direct.mkdir(parents=True)
    (direct / "sample.nc").write_text("")

    composite = ref_root / "Grid" / "LowRes" / "Composite" / "VarD" / "DatasetD"
    composite.mkdir(parents=True)
    (composite / "composite.nc").write_text("")

    groups = scan_reference_directory(ref_root)
    by_name = {group.base_name: group for group in groups}

    assert set(by_name) == {"DatasetA", "DatasetD"}
    variant = by_name["DatasetD"].variants["LowRes"]
    assert variant.variables == {"VarD": "Composite/VarD/DatasetD"}
    assert variant.file_count == 1


def test_scan_reference_directory_registers_multiple_standard_composite_datasets(tmp_path: Path):
    """Composite/<variable>/<dataset> with several datasets is standard layout."""
    import numpy as np
    import xarray as xr

    ref_root = tmp_path / "ref"
    for dataset_name in ("DatasetA", "DatasetB"):
        nc_dir = ref_root / "Grid" / "LowRes" / "Composite" / "VarX" / dataset_name
        nc_dir.mkdir(parents=True)
        ds = xr.Dataset(
            {"VarX": (["lat", "lon"], np.zeros((2, 2), dtype=np.float32))},
            coords={"lat": [0.0, 0.5], "lon": [0.0, 0.5]},
        )
        ds.to_netcdf(nc_dir / f"{dataset_name}_2010.nc")

    skipped = []
    groups = scan_reference_directory(ref_root, on_skip=skipped.append)
    by_name = {group.base_name: group for group in groups}

    assert set(by_name) == {"DatasetA", "DatasetB"}
    assert skipped == []
    assert by_name["DatasetA"].variants["LowRes"].variables == {"VarX": "Composite/VarX/DatasetA"}


def test_scan_reference_directory_discovers_one_level_nested_children(tmp_path: Path):
    """When NC files live one level deeper (single child), sub_dir must point
    to that child so the registration step's _inspect_nc_file finds the actual
    NCs (previously this recorded the parent path, leaving inspect to find no
    NCs and returning empty varname/prefix/suffix).
    """
    ref_root = tmp_path / "ref"

    one_level = ref_root / "Grid" / "LowRes" / "Water" / "VarB" / "DatasetB"
    (one_level / "child").mkdir(parents=True)
    (one_level / "child" / "nested.nc").write_text("")

    groups = scan_reference_directory(ref_root)
    dataset_b = next(group for group in groups if group.base_name == "DatasetB")

    assert dataset_b.variants["LowRes"].variables == {"VarB": "Water/VarB/DatasetB/child"}
    assert dataset_b.variants["LowRes"].file_count == 1


def test_scan_reference_directory_finds_grandchildren_within_3_level_depth(tmp_path: Path):
    """NC files two levels below dataset_dir (single chain) ARE supported.

    Layout: dataset_dir/<single_subdir>/<single_subdir>/*.nc — a common
    convention like 0p25deg/daily/file.nc or raw/data/file.nc.
    """
    ref_root = tmp_path / "ref"

    deep_child = ref_root / "Grid" / "LowRes" / "Water" / "VarC" / "DatasetC"
    (deep_child / "child" / "grand").mkdir(parents=True)
    (deep_child / "child" / "grand" / "found.nc").write_text("")

    groups = scan_reference_directory(ref_root)
    dataset_c = next(group for group in groups if group.base_name == "DatasetC")

    assert dataset_c.variants["LowRes"].variables == {"VarC": "Water/VarC/DatasetC/child/grand"}
    assert dataset_c.variants["LowRes"].file_count == 1


def test_scan_reference_directory_misses_great_grandchildren(tmp_path: Path):
    """NC files three levels below dataset_dir are still beyond supported depth."""
    ref_root = tmp_path / "ref"

    deeper = ref_root / "Grid" / "LowRes" / "Water" / "VarD" / "DatasetD"
    (deeper / "a" / "b" / "c").mkdir(parents=True)
    (deeper / "a" / "b" / "c" / "missed.nc").write_text("")

    groups = scan_reference_directory(ref_root)

    assert {group.base_name for group in groups} == set()


def test_scan_reference_directory_marks_mixed_shallow_and_deep_nc_children_ambiguous(tmp_path: Path):
    ref_root = tmp_path / "ref"
    dataset_dir = ref_root / "Grid" / "LowRes" / "Water" / "VarE" / "DatasetE"
    (dataset_dir / "shallow").mkdir(parents=True)
    (dataset_dir / "shallow" / "a.nc").write_text("")
    (dataset_dir / "deep" / "nested").mkdir(parents=True)
    (dataset_dir / "deep" / "nested" / "b.nc").write_text("")

    skipped = []
    groups = scan_reference_directory(ref_root, on_skip=skipped.append)

    assert "DatasetE" not in {group.base_name for group in groups}
    assert any(message.reason == "ambiguous_nc_subdirectories" for message in skipped)
