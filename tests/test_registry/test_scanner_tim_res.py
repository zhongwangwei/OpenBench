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


def test_scan_reference_directory_skips_composite_and_deep_children(tmp_path: Path):
    ref_root = tmp_path / "ref"

    direct = ref_root / "Grid" / "LowRes" / "Water" / "VarA" / "DatasetA"
    direct.mkdir(parents=True)
    (direct / "sample.nc").write_text("")

    one_level = ref_root / "Grid" / "LowRes" / "Water" / "VarB" / "DatasetB"
    (one_level / "child").mkdir(parents=True)
    (one_level / "child" / "nested.nc").write_text("")

    deep_child = ref_root / "Grid" / "LowRes" / "Water" / "VarC" / "DatasetC"
    (deep_child / "child" / "grand").mkdir(parents=True)
    (deep_child / "child" / "grand" / "missed.nc").write_text("")

    composite = ref_root / "Grid" / "LowRes" / "Composite" / "VarD" / "DatasetD"
    composite.mkdir(parents=True)
    (composite / "composite.nc").write_text("")

    groups = scan_reference_directory(ref_root)
    names = {group.base_name for group in groups}

    assert names == {"DatasetA", "DatasetB"}
    dataset_b = next(group for group in groups if group.base_name == "DatasetB")
    assert dataset_b.variants["LowRes"].variables == {"VarB": "Water/VarB/DatasetB"}
    assert dataset_b.variants["LowRes"].file_count == 1
