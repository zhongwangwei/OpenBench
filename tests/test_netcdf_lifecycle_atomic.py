import os
from pathlib import Path

import numpy as np
import pytest
import xarray as xr


def _grid_dataset(var="value", value=1.0):
    return xr.Dataset(
        {var: (("lat", "lon"), np.array([[value]], dtype=float))},
        coords={"lat": [0.0], "lon": [10.0]},
    )


def test_statistics_save_result_preserves_existing_netcdf_on_write_failure(tmp_path, monkeypatch):
    import openbench.core.statistics.Mod_Statistics as stats_module

    output_dir = tmp_path / "statistics"
    target_dir = output_dir / "Mean"
    target_dir.mkdir(parents=True)
    target = target_dir / "Mean_source_output.nc"
    _grid_dataset("Mean", 1.0).to_netcdf(target)

    processor = object.__new__(stats_module.BasicProcessing)
    processor.output_dir = str(output_dir)

    def failing_to_netcdf(self, path, *args, **kwargs):
        Path(path).write_bytes(b"partial invalid netcdf")
        raise OSError("simulated write failure")

    monkeypatch.setattr(xr.Dataset, "to_netcdf", failing_to_netcdf)

    with pytest.raises(OSError, match="simulated write failure"):
        stats_module.BasicProcessing.save_result(processor, "Mean", _grid_dataset("raw", 2.0), ["source"])

    with xr.open_dataset(target) as ds:
        np.testing.assert_allclose(ds["Mean"].values, [[1.0]])
    assert not list(target_dir.glob(".Mean_source_output.nc.*.tmp.nc"))


def test_comparison_save_result_preserves_existing_netcdf_on_write_failure(tmp_path, monkeypatch):
    import openbench.core.comparison as comparison_module

    target = tmp_path / "comparison.nc"
    _grid_dataset("bias", 1.0).to_netcdf(target)

    processor = object.__new__(comparison_module.ComparisonProcessing)

    def failing_to_netcdf(self, path, *args, **kwargs):
        Path(path).write_bytes(b"partial invalid netcdf")
        raise OSError("simulated write failure")

    monkeypatch.setattr(xr.Dataset, "to_netcdf", failing_to_netcdf)

    with pytest.raises(OSError, match="simulated write failure"):
        comparison_module.ComparisonProcessing.save_result(processor, str(target), "bias", _grid_dataset("raw", 2.0))

    with xr.open_dataset(target) as ds:
        np.testing.assert_allclose(ds["bias"].values, [[1.0]])
    assert not list(tmp_path.glob(".comparison.nc.*.tmp.nc"))


def _evaluation_processor(tmp_path):
    import openbench.core.evaluation as evaluation_module

    processor = object.__new__(evaluation_module.Evaluation_grid)
    processor.casedir = str(tmp_path)
    processor.item = "Runoff"
    processor.ref_source = "TestRef"
    processor.sim_source = "SimA"
    processor.output_manager = None
    processor.bias = lambda s, o: np.array([[2.0, 3.0], [4.0, 5.0]])
    processor.Overall_Score = lambda s, o: np.array([[0.5, 0.6], [0.7, 0.8]])
    return processor


def _lat_lon_array():
    return xr.DataArray(
        np.ones((2, 2), dtype=float),
        coords={"lat": [0.0, 1.0], "lon": [10.0, 11.0]},
        dims=("lat", "lon"),
    )


def test_evaluation_metric_save_preserves_existing_netcdf_on_write_failure(tmp_path, monkeypatch):
    processor = _evaluation_processor(tmp_path)
    target_dir = tmp_path / "metrics"
    target_dir.mkdir()
    target = target_dir / "Runoff_ref_TestRef_sim_SimA_bias.nc"
    _grid_dataset("bias", 1.0).to_netcdf(target)

    def failing_to_netcdf(self, path, *args, **kwargs):
        Path(path).write_bytes(b"partial invalid netcdf")
        raise OSError("simulated metric write failure")

    monkeypatch.setattr(xr.DataArray, "to_netcdf", failing_to_netcdf)

    with pytest.raises(OSError, match="simulated metric write failure"):
        processor.process_metric("bias", _lat_lon_array(), _lat_lon_array())

    with xr.open_dataset(target) as ds:
        np.testing.assert_allclose(ds["bias"].values, [[1.0]])
    assert not list(target_dir.glob(".Runoff_ref_TestRef_sim_SimA_bias.nc.*.tmp.nc"))


def test_evaluation_score_save_preserves_existing_netcdf_on_write_failure(tmp_path, monkeypatch):
    processor = _evaluation_processor(tmp_path)
    target_dir = tmp_path / "scores"
    target_dir.mkdir()
    target = target_dir / "Runoff_ref_TestRef_sim_SimA_Overall_Score.nc"
    _grid_dataset("Overall_Score", 1.0).to_netcdf(target)

    def failing_to_netcdf(self, path, *args, **kwargs):
        Path(path).write_bytes(b"partial invalid netcdf")
        raise OSError("simulated score write failure")

    monkeypatch.setattr(xr.DataArray, "to_netcdf", failing_to_netcdf)

    with pytest.raises(OSError, match="simulated score write failure"):
        processor.process_score("Overall_Score", _lat_lon_array(), _lat_lon_array())

    with xr.open_dataset(target) as ds:
        np.testing.assert_allclose(ds["Overall_Score"].values, [[1.0]])
    assert not list(target_dir.glob(".Runoff_ref_TestRef_sim_SimA_Overall_Score.nc.*.tmp.nc"))


def test_core_comparison_uses_atomic_netcdf_writes_only():
    source = Path("src/openbench/core/comparison.py").read_text(encoding="utf-8")

    assert ".to_netcdf(" not in source


def test_groupby_modules_use_atomic_netcdf_writes_only():
    for path in (
        Path("src/openbench/core/climatezone_groupby.py"),
        Path("src/openbench/core/landcover_groupby.py"),
    ):
        assert ".to_netcdf(" not in path.read_text(encoding="utf-8"), str(path)


@pytest.mark.parametrize(
    "module_name",
    [
        "openbench.core.climatezone_groupby",
        "openbench.core.landcover_groupby",
    ],
)
def test_groupby_table_write_preserves_existing_csv_on_failure(tmp_path, monkeypatch, module_name):
    import importlib

    module = importlib.import_module(module_name)
    target = tmp_path / "groupby.csv"
    target.write_text("old complete table\n")
    original_write_text = Path.write_text

    def failing_write_text(path, text, *args, **kwargs):
        original_write_text(path, "partial table\n", *args, **kwargs)
        raise OSError("simulated CSV write failure")

    monkeypatch.setattr(Path, "write_text", failing_write_text)

    with pytest.raises(OSError, match="simulated CSV write failure"):
        module._write_lines_atomic(str(target), ["new complete table\n"])

    assert target.read_text(encoding="utf-8") == "old complete table\n"
    assert not list(tmp_path.glob(".groupby.csv.*.tmp.csv"))


def test_groupby_modules_use_atomic_table_writes_only():
    for path in (
        Path("src/openbench/core/climatezone_groupby.py"),
        Path("src/openbench/core/landcover_groupby.py"),
    ):
        source = path.read_text(encoding="utf-8")
        assert "with open(output_file_path" not in source, str(path)
        assert "_write_lines_atomic(" in source, str(path)


def test_cdo_remap_paths_use_closed_atomic_temp_netcdf_writes():
    for path in (
        Path("src/openbench/core/statistics/Mod_Statistics.py"),
        Path("src/openbench/data/processing.py"),
    ):
        source = path.read_text(encoding="utf-8")
        assert ".to_netcdf(" not in source, str(path)
        assert 'NamedTemporaryFile(suffix=".nc")' not in source, str(path)


def test_netcdf_compression_is_opt_in_by_environment(tmp_path, monkeypatch):
    from openbench.util.netcdf import write_netcdf_atomic

    captured = {}

    def fake_to_netcdf(self, path, *args, **kwargs):
        captured.update(kwargs)

    monkeypatch.delenv("OPENBENCH_NETCDF_COMPRESSION", raising=False)
    monkeypatch.delenv("OPENBENCH_NETCDF_COMP_LEVEL", raising=False)
    monkeypatch.setattr(xr.Dataset, "to_netcdf", fake_to_netcdf)

    write_netcdf_atomic(_grid_dataset("metric", 1.0), tmp_path / "out.nc")

    assert "encoding" not in captured


def test_netcdf_compression_defaults_to_level_one_for_numeric_variables(tmp_path, monkeypatch):
    from openbench.util.netcdf import write_netcdf_atomic

    captured = {}
    dataset = xr.Dataset(
        {
            "metric": (("lat", "lon"), np.ones((2, 2), dtype=np.float32)),
            "label": ("lat", np.array(["north", "south"], dtype=object)),
        },
        coords={"lat": [0.0, 1.0], "lon": [10.0, 11.0]},
    )

    def fake_to_netcdf(self, path, *args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setenv("OPENBENCH_NETCDF_COMPRESSION", "1")
    monkeypatch.delenv("OPENBENCH_NETCDF_COMP_LEVEL", raising=False)
    monkeypatch.setattr(xr.Dataset, "to_netcdf", fake_to_netcdf)

    write_netcdf_atomic(dataset, tmp_path / "out.nc")

    assert captured["encoding"]["metric"] == {"zlib": True, "complevel": 1, "shuffle": True}
    assert "label" not in captured["encoding"]


def test_netcdf_compression_can_be_disabled_for_intermediates(tmp_path, monkeypatch):
    from openbench.util.netcdf import write_netcdf_atomic

    captured = {}

    def fake_to_netcdf(self, path, *args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setenv("OPENBENCH_NETCDF_COMPRESSION", "1")
    monkeypatch.setenv("OPENBENCH_NETCDF_COMP_LEVEL", "4")
    monkeypatch.setattr(xr.Dataset, "to_netcdf", fake_to_netcdf)

    write_netcdf_atomic(_grid_dataset("metric", 1.0), tmp_path / "scratch.nc", compression=False)

    assert "encoding" not in captured


def test_netcdf_compression_preserves_explicit_encoding(tmp_path, monkeypatch):
    from openbench.util.netcdf import write_netcdf_atomic

    captured = {}

    def fake_to_netcdf(self, path, *args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setenv("OPENBENCH_NETCDF_COMPRESSION", "true")
    monkeypatch.setenv("OPENBENCH_NETCDF_COMP_LEVEL", "4")
    monkeypatch.setattr(xr.Dataset, "to_netcdf", fake_to_netcdf)

    write_netcdf_atomic(
        _grid_dataset("metric", 1.0),
        tmp_path / "out.nc",
        encoding={"metric": {"dtype": "float32", "complevel": 9}},
    )

    assert captured["encoding"]["metric"] == {
        "dtype": "float32",
        "complevel": 9,
        "zlib": True,
        "shuffle": True,
    }


def test_netcdf_compression_skips_netcdf3_backends(tmp_path, monkeypatch):
    from openbench.util.netcdf import write_netcdf_atomic

    captured = {}

    def fake_to_netcdf(self, path, *args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setenv("OPENBENCH_NETCDF_COMPRESSION", "1")
    monkeypatch.setattr(xr.Dataset, "to_netcdf", fake_to_netcdf)

    write_netcdf_atomic(_grid_dataset("metric", 1.0), tmp_path / "out.nc", format="NETCDF3_CLASSIC")

    assert captured == {"format": "NETCDF3_CLASSIC"}


def test_atomic_file_write_fsyncs_temp_file_and_parent_directory(tmp_path, monkeypatch):
    from openbench.util.netcdf import write_file_atomic

    fsynced = []
    opened_dirs = []
    real_open = os.open
    real_close = os.close

    def fake_fsync(fd):
        fsynced.append(fd)

    def fake_open(path, flags, *args, **kwargs):
        if Path(path) == tmp_path:
            opened_dirs.append(Path(path))
            return 987654
        return real_open(path, flags, *args, **kwargs)

    def fake_close(fd):
        if fd == 987654:
            return None
        return real_close(fd)

    monkeypatch.setattr(os, "fsync", fake_fsync)
    monkeypatch.setattr(os, "open", fake_open)
    monkeypatch.setattr(os, "close", fake_close)

    write_file_atomic(tmp_path / "out.txt", lambda p: p.write_text("ok"), suffix=".tmp")

    assert (tmp_path / "out.txt").read_text(encoding="utf-8") == "ok"
    # Directory fsync is POSIX-only (os.O_DIRECTORY); Windows skips it by design.
    if hasattr(os, "O_DIRECTORY"):
        assert len(fsynced) >= 2
        assert opened_dirs == [tmp_path]
    else:
        assert len(fsynced) >= 1
        assert opened_dirs == []
