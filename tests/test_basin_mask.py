from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr


def _write_mask(path):
    xr.Dataset(
        {
            "basin_mask": (
                ("lat", "lon"),
                np.array(
                    [
                        [0, 1, 0],
                        [0, 0, 1],
                    ],
                    dtype=np.uint8,
                ),
            )
        },
        coords={"lat": [20.0, 10.0], "lon": [350.0, 0.0, 10.0]},
    ).to_netcdf(path)


def test_filter_stations_by_basin_mask_handles_descending_lat_and_cyclic_lon(tmp_path):
    from openbench.data.basin_mask import filter_stations_by_basin_mask

    mask_path = tmp_path / "basin.nc"
    _write_mask(mask_path)
    stations = pd.DataFrame(
        {
            "ID": ["inside-west", "inside-east", "outside"],
            "ref_lon": [0.1, 9.9, -9.9],
            "ref_lat": [19.9, 10.1, 10.1],
        }
    )

    filtered = filter_stations_by_basin_mask(stations, mask_path)

    assert filtered["ID"].tolist() == ["inside-west", "inside-east"]


def test_filter_stations_by_basin_mask_requires_expected_variable(tmp_path):
    from openbench.data.basin_mask import filter_stations_by_basin_mask

    mask_path = tmp_path / "invalid.nc"
    xr.Dataset(
        {"other": (("lat", "lon"), np.ones((2, 2), dtype=np.uint8))},
        coords={"lat": [0.0, 1.0], "lon": [0.0, 1.0]},
    ).to_netcdf(mask_path)

    with pytest.raises(ValueError, match="basin_mask"):
        filter_stations_by_basin_mask(
            pd.DataFrame({"ID": ["a"], "ref_lon": [0.0], "ref_lat": [0.0]}),
            mask_path,
        )


def test_general_info_reader_applies_basin_mask_after_custom_station_filter(tmp_path):
    from openbench.config.runtime_info import GeneralInfoReader

    mask_path = tmp_path / "basin.nc"
    _write_mask(mask_path)
    reader = object.__new__(GeneralInfoReader)
    reader.ref_source = "Ref"
    reader.sim_source = "Sim"
    reader.casedir = str(tmp_path)
    reader.basin_mask = str(mask_path)
    reader.stn_list = pd.DataFrame(
        {
            "ID": ["inside", "outside"],
            "ref_lon": [0.0, -10.0],
            "ref_lat": [20.0, 10.0],
        }
    )
    reader._get_custom_filter = lambda: lambda _reader: None

    reader._filter_stations()

    assert reader.stn_list["ID"].tolist() == ["inside"]
    written = pd.read_csv(tmp_path / "stn_Ref_Sim_list.csv")
    assert written["ID"].tolist() == ["inside"]


def test_basin_mask_reaches_runner_general_config(tmp_path):
    from openbench.config.adapter import build_runner_config
    from openbench.config.schema import (
        EvaluationConfig,
        OpenBenchConfig,
        ProjectConfig,
        ReferenceConfig,
        SimulationEntry,
    )

    mask_path = tmp_path / "basin.nc"
    cfg = OpenBenchConfig(
        project=ProjectConfig(
            name="case",
            output_dir=str(tmp_path),
            years=[2000, 2001],
            basin_mask=str(mask_path),
        ),
        evaluation=EvaluationConfig(variables=["Runoff"]),
        reference=ReferenceConfig(sources={"Runoff": "Ref"}),
        simulation={"Sim": SimulationEntry(model="Model", root_dir=str(tmp_path))},
    )

    runner = build_runner_config(cfg)

    assert runner.general["basin_mask"] == str(mask_path)


def test_single_file_signature_changes_when_basin_mask_changes(tmp_path):
    from openbench.runner.hashing import single_file_signature

    mask_path = tmp_path / "basin.nc"
    _write_mask(mask_path)
    first = single_file_signature(str(mask_path))

    with xr.open_dataset(mask_path) as dataset:
        changed = dataset.load()
    changed["basin_mask"].values[0, 0] = 1
    changed.to_netcdf(mask_path)
    second = single_file_signature(str(mask_path))

    assert first["exists"] is True
    assert first["sample_sha256"] != second["sample_sha256"]


def test_basin_mask_maps_to_grid_and_masks_outside_cells(tmp_path):
    from openbench.data.basin_mask import apply_basin_mask_to_grid, basin_mask_extent

    mask_path = tmp_path / "basin.nc"
    _write_mask(mask_path)
    data = xr.Dataset(
        {"flow": (("time", "lat", "lon"), np.ones((1, 2, 3), dtype=float))},
        coords={"time": [0], "lat": [10.0, 20.0], "lon": [350.0, 0.0, 10.0]},
    )

    masked = apply_basin_mask_to_grid(data, mask_path)

    assert basin_mask_extent(mask_path) == (-5.0, 15.0, 5.0, 25.0)
    expected = np.array([[np.nan, np.nan, 1.0], [np.nan, 1.0, np.nan]])
    np.testing.assert_allclose(masked["flow"].isel(time=0).values, expected, equal_nan=True)
    np.testing.assert_array_equal(masked["lon"].values, [350.0, 0.0, 10.0])


def test_grid_basin_mask_rejects_target_that_truncates_basin(tmp_path):
    from openbench.data.basin_mask import basin_mask_for_grid

    mask_path = tmp_path / "basin.nc"
    _write_mask(mask_path)

    with pytest.raises(ValueError, match="does not cover the complete basin mask"):
        basin_mask_for_grid(mask_path, [10.0, 20.0], [0.0])


def test_grid_processing_applies_basin_mask_only_to_grid_grid(tmp_path):
    from openbench.data._processing_grid_core import GridProcessingCoreMixin

    mask_path = tmp_path / "basin.nc"
    _write_mask(mask_path)
    data = xr.Dataset(
        {"flow": (("lat", "lon"), np.ones((2, 3), dtype=float))},
        coords={"lat": [10.0, 20.0], "lon": [350.0, 0.0, 10.0]},
    )
    processor = object.__new__(GridProcessingCoreMixin)
    processor.basin_mask = str(mask_path)
    processor.item = "Streamflow"
    processor.ref_data_type = "grid"
    processor.sim_data_type = "grid"

    masked = processor._apply_configured_grid_basin_mask(data)
    assert int(np.isfinite(masked["flow"].values).sum()) == 2

    processor.ref_data_type = "stn"
    unchanged = processor._apply_configured_grid_basin_mask(data)
    assert unchanged is data
