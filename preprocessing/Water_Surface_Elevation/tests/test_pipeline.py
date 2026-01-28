import pytest
from unittest.mock import MagicMock, patch
from src.pipeline import Pipeline

def test_pipeline_initialization():
    config = {'data_root': '/tmp', 'output_dir': '/tmp/out'}
    pipeline = Pipeline(config)
    assert pipeline.config == config

def test_pipeline_step_sequence():
    config = {'data_root': '/tmp', 'output_dir': '/tmp/out', 'skip_download': True}
    pipeline = Pipeline(config)
    assert pipeline.steps == ['download', 'validate', 'cama', 'reserved', 'merge']

def test_pipeline_has_step_handlers():
    config = {'data_root': '/tmp', 'output_dir': '/tmp/out'}
    pipeline = Pipeline(config)
    assert 'download' in pipeline._step_handlers
    assert 'validate' in pipeline._step_handlers
    assert 'cama' in pipeline._step_handlers
    assert 'reserved' in pipeline._step_handlers
    assert 'merge' in pipeline._step_handlers

def test_run_step_invalid():
    config = {'data_root': '/tmp', 'output_dir': '/tmp/out'}
    pipeline = Pipeline(config)
    with pytest.raises(ValueError, match="Unknown step"):
        pipeline.run_step('invalid_step', ['hydroweb'])
