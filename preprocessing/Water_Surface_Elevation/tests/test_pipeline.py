import pytest
from unittest.mock import MagicMock, patch
from src.pipeline import Pipeline, PipelineResult


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


# =============================================================================
# PipelineResult Tests
# =============================================================================

class TestPipelineResult:
    """Tests for PipelineResult dataclass."""

    def test_pipeline_result_creation(self):
        """Test PipelineResult can be created with required fields."""
        result = PipelineResult(
            success=True,
            stations_processed=10,
            output_files=['/tmp/output.txt']
        )
        assert result.success is True
        assert result.stations_processed == 10
        assert result.output_files == ['/tmp/output.txt']
        assert result.error is None
        assert result.step_results == {}

    def test_pipeline_result_with_error(self):
        """Test PipelineResult with error message."""
        result = PipelineResult(
            success=False,
            stations_processed=0,
            output_files=[],
            error="Pipeline failed during validation"
        )
        assert result.success is False
        assert result.error == "Pipeline failed during validation"

    def test_pipeline_result_with_step_results(self):
        """Test PipelineResult with step results."""
        step_results = {
            'download': {'hydroweb': True},
            'validate': {'total': 100},
            'cama': {'allocated': 95},
        }
        result = PipelineResult(
            success=True,
            stations_processed=95,
            output_files=['/tmp/output.txt'],
            step_results=step_results
        )
        assert result.step_results == step_results
        assert result.step_results['validate']['total'] == 100


# =============================================================================
# Pipeline.run() Returns PipelineResult Tests
# =============================================================================

class TestPipelineRunReturnsPipelineResult:
    """Tests for Pipeline.run() returning PipelineResult."""

    def test_pipeline_run_returns_pipeline_result(self, mock_config, sample_station_list, mock_step_handlers, mock_checkpoint):
        """Test that Pipeline.run() returns a PipelineResult object."""
        with patch.object(Pipeline, '__init__', lambda self, config: None):
            pipeline = Pipeline.__new__(Pipeline)
            pipeline.config = mock_config
            pipeline.steps = PIPELINE_STEPS
            pipeline._step_handlers = mock_step_handlers
            pipeline.checkpoint = mock_checkpoint

            result = pipeline.run(['hydroweb'])

            assert isinstance(result, PipelineResult)
            assert result.success is True
            assert result.stations_processed >= 0
            assert isinstance(result.output_files, list)
            assert isinstance(result.step_results, dict)

    def test_pipeline_run_captures_step_results(self, mock_config, sample_station_list, mock_step_handlers, mock_checkpoint):
        """Test that Pipeline.run() captures results from each step."""
        with patch.object(Pipeline, '__init__', lambda self, config: None):
            pipeline = Pipeline.__new__(Pipeline)
            pipeline.config = mock_config
            pipeline.steps = PIPELINE_STEPS
            pipeline._step_handlers = mock_step_handlers
            pipeline.checkpoint = mock_checkpoint

            result = pipeline.run(['hydroweb'])

            # Should have results for each step
            assert 'download' in result.step_results
            assert 'validate' in result.step_results
            assert 'cama' in result.step_results
            assert 'merge' in result.step_results

    def test_pipeline_run_returns_output_files(self, mock_config, sample_station_list, mock_step_handlers, mock_checkpoint):
        """Test that Pipeline.run() returns output files from merge step."""
        mock_step_handlers['merge'].run.return_value = ['/tmp/output1.txt', '/tmp/output2.txt']

        with patch.object(Pipeline, '__init__', lambda self, config: None):
            pipeline = Pipeline.__new__(Pipeline)
            pipeline.config = mock_config
            pipeline.steps = PIPELINE_STEPS
            pipeline._step_handlers = mock_step_handlers
            pipeline.checkpoint = mock_checkpoint

            result = pipeline.run(['hydroweb'])

            assert result.output_files == ['/tmp/output1.txt', '/tmp/output2.txt']

    def test_pipeline_run_failure_returns_error(self, mock_config, mock_step_handlers, mock_checkpoint):
        """Test that Pipeline.run() returns error on step failure."""
        # Make validate step raise an exception
        mock_step_handlers['validate'].run.side_effect = ValueError("Validation failed")

        with patch.object(Pipeline, '__init__', lambda self, config: None):
            pipeline = Pipeline.__new__(Pipeline)
            pipeline.config = mock_config
            pipeline.steps = PIPELINE_STEPS
            pipeline._step_handlers = mock_step_handlers
            pipeline.checkpoint = mock_checkpoint

            result = pipeline.run(['hydroweb'])

            assert result.success is False
            assert "Validation failed" in result.error


class TestPipelineRunStepReturnsPipelineResult:
    """Tests for Pipeline.run_step() returning PipelineResult."""

    def test_run_step_returns_pipeline_result(self, mock_config):
        """Test that run_step returns PipelineResult."""
        from src.core.station import StationList

        pipeline = Pipeline(mock_config)
        mock_stations = StationList()

        with patch.object(pipeline._step_handlers['validate'], 'run', return_value=mock_stations):
            result = pipeline.run_step('validate', ['hydroweb'])

            assert isinstance(result, PipelineResult)
            assert result.success is True

    def test_run_step_failure_returns_error(self, mock_config):
        """Test that run_step returns error on failure."""
        pipeline = Pipeline(mock_config)

        with patch.object(pipeline._step_handlers['download'], 'run', side_effect=RuntimeError("Download failed")):
            result = pipeline.run_step('download', ['hydroweb'])

            assert result.success is False
            assert "Download failed" in result.error
