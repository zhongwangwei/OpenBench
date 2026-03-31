"""Tests for time-resolution detection in the registry scanner."""

from pathlib import Path

from openbench.data.registry.scanner import _detect_tim_res


def test_detect_tim_res_prefers_3hour_over_hourly(tmp_path: Path):
    cases = [
        ("sample_3hourly.nc", "3Hour"),
        ("sample_hourly.nc", "Hour"),
    ]

    for filename, expected in cases:
        dataset_dir = tmp_path / filename.replace(".nc", "")
        dataset_dir.mkdir()
        (dataset_dir / filename).write_text("")

        assert _detect_tim_res(dataset_dir) == expected
