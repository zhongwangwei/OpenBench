from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_detached_data_pipeline_module_is_removed():
    assert not (ROOT / "src/openbench/data/pipeline.py").exists()


def test_processing_module_no_longer_imports_detached_pipeline():
    source = (ROOT / "src/openbench/data/processing.py").read_text(encoding="utf-8")

    assert "openbench.data.pipeline" not in source
    assert "ProcessingPipeline" not in source
    assert "setup_data_pipeline" not in source
