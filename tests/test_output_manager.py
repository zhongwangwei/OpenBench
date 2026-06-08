"""Regression tests for the modular output manager."""

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from openbench.util.output import create_output_manager


def test_output_manager_registers_default_formatters(tmp_path):
    manager = create_output_manager(str(tmp_path))

    assert set(manager.get_supported_formats()) >= {"netcdf", "csv", "json"}
    assert manager.get_formatter_info("json")["extension"] == ".json"


def test_netcdf_formatter_preserves_existing_file_when_write_fails(tmp_path, monkeypatch):
    """NetCDF formatter should use atomic replacement, not overwrite in place."""
    output = tmp_path / "metrics" / "result.nc"
    output.parent.mkdir()
    xr.Dataset({"value": ("time", np.array([1.0]))}).to_netcdf(output)

    manager = create_output_manager(str(tmp_path))

    def fail_to_netcdf(self, path, *args, **kwargs):
        path.write_bytes(b"partial invalid netcdf")
        raise OSError("simulated write failure")

    monkeypatch.setattr(xr.Dataset, "to_netcdf", fail_to_netcdf)

    with pytest.raises(Exception, match="simulated write failure"):
        manager.save_data(xr.Dataset({"value": ("time", np.array([2.0]))}), "metrics", "result", "netcdf")

    with xr.open_dataset(output) as ds:
        np.testing.assert_allclose(ds["value"].values, [1.0])
    assert not list(output.parent.glob(".result.nc.*.tmp.nc"))


def test_csv_formatter_preserves_existing_file_when_write_fails(tmp_path, monkeypatch):
    """CSV formatter should not leave partial bytes at the final path."""
    output = tmp_path / "metrics" / "table.csv"
    output.parent.mkdir()
    output.write_text("a\n1\n")

    manager = create_output_manager(str(tmp_path))

    def fail_to_csv(self, path, *args, **kwargs):
        path.write_text("partial")
        raise OSError("simulated csv failure")

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail_to_csv)

    with pytest.raises(Exception, match="simulated csv failure"):
        manager.save_data(pd.DataFrame({"a": [2]}), "metrics", "table", "csv")

    assert output.read_text() == "a\n1\n"
    assert not list(output.parent.glob(".table.csv.*.tmp.csv"))


def test_json_formatter_preserves_existing_file_when_write_fails(tmp_path, monkeypatch):
    """JSON formatter should not replace a valid existing file until dump succeeds."""
    import openbench.util.output as output_module

    output = tmp_path / "metrics" / "payload.json"
    output.parent.mkdir()
    output.write_text('{"ok": true}')

    manager = create_output_manager(str(tmp_path))

    def fail_dump(data, fp, *args, **kwargs):
        fp.write("partial")
        raise OSError("simulated json failure")

    monkeypatch.setattr(output_module.json, "dump", fail_dump)

    with pytest.raises(Exception, match="simulated json failure"):
        manager.save_data({"ok": False}, "metrics", "payload", "json")

    assert json.loads(output.read_text()) == {"ok": True}
    assert not list(output.parent.glob(".payload.json.*.tmp.json"))
