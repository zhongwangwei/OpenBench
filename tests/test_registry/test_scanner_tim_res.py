"""Tests for time-resolution detection in the registry scanner."""

from pathlib import Path

from openbench.data.registry.scanner import _detect_tim_res


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
