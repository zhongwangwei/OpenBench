"""Tests for time-resolution detection in the registry scanner."""

from pathlib import Path

from openbench.data.registry.scanner import _detect_tim_res, scan_reference_directory


def test_detect_tim_res_prefers_3hour_over_hourly(tmp_path: Path):
    dataset_dir = tmp_path / "sample_3hourly"
    dataset_dir.mkdir()
    (dataset_dir / "sample_3hourly.nc").write_text("")

    assert _detect_tim_res(dataset_dir) == "3Hour"


def test_detect_tim_res_detects_hourly(tmp_path: Path):
    dataset_dir = tmp_path / "sample_hourly"
    dataset_dir.mkdir()
    (dataset_dir / "sample_hourly.nc").write_text("")

    assert _detect_tim_res(dataset_dir) == "Hour"


def test_scan_reference_directory_skips_composite(tmp_path: Path):
    ref_root = tmp_path / "ref"

    direct = ref_root / "Grid" / "LowRes" / "Water" / "VarA" / "DatasetA"
    direct.mkdir(parents=True)
    (direct / "sample.nc").write_text("")

    composite = ref_root / "Grid" / "LowRes" / "Composite" / "VarD" / "DatasetD"
    composite.mkdir(parents=True)
    (composite / "composite.nc").write_text("")

    groups = scan_reference_directory(ref_root)
    names = {group.base_name for group in groups}

    assert names == {"DatasetA"}


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

    assert dataset_c.variants["LowRes"].variables == {
        "VarC": "Water/VarC/DatasetC/child/grand"
    }
    assert dataset_c.variants["LowRes"].file_count == 1


def test_scan_reference_directory_misses_great_grandchildren(tmp_path: Path):
    """NC files three levels below dataset_dir are still beyond supported depth."""
    ref_root = tmp_path / "ref"

    deeper = ref_root / "Grid" / "LowRes" / "Water" / "VarD" / "DatasetD"
    (deeper / "a" / "b" / "c").mkdir(parents=True)
    (deeper / "a" / "b" / "c" / "missed.nc").write_text("")

    groups = scan_reference_directory(ref_root)

    assert {group.base_name for group in groups} == set()
