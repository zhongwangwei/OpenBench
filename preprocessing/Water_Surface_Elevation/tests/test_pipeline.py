import pytest
from unittest.mock import MagicMock, patch
from src.pipeline import Pipeline


# Pipeline steps constant (matches conftest.PIPELINE_STEPS)
PIPELINE_STEPS = ['download', 'validate', 'cama', 'reserved', 'merge']


def test_pipeline_initialization(minimal_config):
    """Test pipeline initializes correctly with config."""
    pipeline = Pipeline(minimal_config)
    assert pipeline.config == minimal_config


def test_pipeline_step_sequence(mock_config):
    """Test pipeline has correct step sequence."""
    pipeline = Pipeline(mock_config)
    assert pipeline.steps == PIPELINE_STEPS


def test_pipeline_has_step_handlers(minimal_config):
    """Test pipeline initializes all step handlers."""
    pipeline = Pipeline(minimal_config)
    for step in PIPELINE_STEPS:
        assert step in pipeline._step_handlers


def test_run_step_invalid(minimal_config):
    """Test that invalid step name raises ValueError."""
    pipeline = Pipeline(minimal_config)
    with pytest.raises(ValueError, match="Unknown step"):
        pipeline.run_step('invalid_step', ['hydroweb'])
