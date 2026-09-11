from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_detached_data_pipeline_module_is_removed():
    assert not (ROOT / "src/openbench/data/pipeline.py").exists()


def test_processing_module_no_longer_imports_detached_pipeline():
    source = (ROOT / "src/openbench/data/processing.py").read_text(encoding="utf-8")

    assert "openbench.data.pipeline" not in source
    assert "ProcessingPipeline" not in source
    assert "setup_data_pipeline" not in source


def test_processing_modules_drop_dead_interface_flag_and_empty_forwarder():
    processing_source = (ROOT / "src/openbench/data/processing.py").read_text(encoding="utf-8")
    config_source = (ROOT / "src/openbench/data/_processing_config.py").read_text(encoding="utf-8")

    assert "_HAS_INTERFACES" not in processing_source
    assert "_HAS_INTERFACES" not in config_source
    assert "Add any additional processing specific to this class if needed" not in processing_source


def test_dataset_processing_prepare_source_inherits_mixin_dispatch(monkeypatch, tmp_path):
    from openbench.data.processing import DatasetProcessing

    calls = []
    monkeypatch.setattr(DatasetProcessing, "initialize_attributes", lambda self, config: None)
    monkeypatch.setattr(DatasetProcessing, "setup_output_directories", lambda self: None)
    monkeypatch.setattr(DatasetProcessing, "initialize_resource_parameters", lambda self: None)
    monkeypatch.setattr(DatasetProcessing, "_preprocess", lambda self, datasource: calls.append(datasource))

    processor = DatasetProcessing({"name": "TestProcessor"})
    processor.prepare_source("sim")

    assert processor.name == "TestProcessor"
    assert calls == ["sim"]


def test_dataset_processing_process_rejects_empty_dataset_and_accepts_valid(monkeypatch):
    import xarray as xr

    from openbench.data.processing import DatasetProcessing

    processor = DatasetProcessing.__new__(DatasetProcessing)
    monkeypatch.setattr(DatasetProcessing, "check_coordinate", lambda self, ds: ds)

    try:
        processor.process(xr.Dataset())
    except ValueError as exc:
        assert "Input dataset validation failed" in str(exc)
    else:
        raise AssertionError("empty dataset was accepted")

    valid = xr.Dataset({"v": ("x", [1.0])})
    assert processor.process(valid) is valid
