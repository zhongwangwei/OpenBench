#!/usr/bin/env python3
"""Integration tests for WSE Pipeline.

These tests verify the full pipeline flow and ensure all steps work together correctly.
"""
import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from click.testing import CliRunner


class TestFullPipelineDryRun:
    """Test full pipeline in dry-run mode."""

    def test_full_pipeline_dry_run(self):
        """Test that dry-run mode doesn't create any files.

        This test:
        1. Creates a temporary config
        2. Runs the pipeline with --dry-run flag
        3. Verifies no output files are created
        """
        from src.main import main

        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'output'

            # Run pipeline with dry-run flag
            result = runner.invoke(main, [
                '--source', 'hydroweb',
                '--output', str(output_dir),
                '--dry-run'
            ])

            # Should exit successfully
            assert result.exit_code == 0, f"CLI failed with: {result.output}"

            # Verify dry-run message is in output
            assert 'DRY RUN' in result.output or 'dry' in result.output.lower()

            # Verify no output directory was created
            assert not output_dir.exists(), \
                f"Output directory should not exist in dry-run mode, but found: {list(output_dir.iterdir()) if output_dir.exists() else 'N/A'}"

    def test_dry_run_with_all_sources(self):
        """Test dry-run with all data sources."""
        from src.main import main

        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'output'

            result = runner.invoke(main, [
                '--source', 'all',
                '--output', str(output_dir),
                '--dry-run'
            ])

            assert result.exit_code == 0
            # No files should be created
            assert not output_dir.exists()

    def test_dry_run_with_merge_flag(self):
        """Test dry-run with merge flag enabled."""
        from src.main import main

        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'output'

            result = runner.invoke(main, [
                '--source', 'hydroweb,cgls',
                '--output', str(output_dir),
                '--merge',
                '--dry-run'
            ])

            assert result.exit_code == 0
            assert not output_dir.exists()

    def test_dry_run_no_checkpoint_created(self):
        """Test that dry-run doesn't create checkpoint files."""
        from src.main import main

        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'output'

            result = runner.invoke(main, [
                '--source', 'hydroweb',
                '--output', str(output_dir),
                '--dry-run'
            ])

            assert result.exit_code == 0

            # Check no checkpoint file was created
            checkpoint_file = Path(tmpdir) / 'output' / 'checkpoint.pkl'
            assert not checkpoint_file.exists()


class TestSingleSourcePipeline:
    """Test pipeline with single source processing."""

    def test_single_source_pipeline(self, mock_config, sample_station_list, mock_step_handlers, mock_checkpoint):
        """Test pipeline with single data source using mocks.

        This test:
        1. Mocks data reading to avoid external dependencies
        2. Runs the pipeline for a single source
        3. Verifies the step sequence is correct
        """
        from src.pipeline import Pipeline, PipelineResult

        # Mock the Pipeline's __init__ to avoid initialization side effects
        with patch.object(Pipeline, '__init__', lambda self, config: None):
            pipeline = Pipeline.__new__(Pipeline)
            pipeline.config = mock_config
            pipeline.steps = ['download', 'validate', 'cama', 'reserved', 'merge']
            pipeline._step_handlers = mock_step_handlers
            pipeline.checkpoint = mock_checkpoint

            # Run the pipeline
            result = pipeline.run(['hydroweb'])

            # Verify step sequence
            mock_step_handlers['download'].run.assert_called_once()
            mock_step_handlers['validate'].run.assert_called_once_with(['hydroweb'])
            mock_step_handlers['cama'].run.assert_called_once()
            mock_step_handlers['reserved'].run.assert_called_once()
            mock_step_handlers['merge'].run.assert_called_once()

            # Verify result is PipelineResult with correct structure
            assert isinstance(result, PipelineResult)
            assert result.success is True
            assert 'download' in result.step_results
            assert 'validate' in result.step_results
            assert 'cama' in result.step_results
            assert 'merge' in result.step_results

    def test_step_sequence_order(self, mock_config):
        """Test that pipeline steps execute in correct order."""
        from src.pipeline import Pipeline

        pipeline = Pipeline(mock_config)

        # Verify step order
        expected_order = ['download', 'validate', 'cama', 'reserved', 'merge']
        assert pipeline.steps == expected_order

    def test_single_step_execution(self, mock_config):
        """Test running a single step only."""
        from src.pipeline import Pipeline
        from src.core.station import StationList

        pipeline = Pipeline(mock_config)

        # Mock the validate step
        mock_stations = StationList()

        with patch.object(pipeline._step_handlers['validate'], 'run', return_value=mock_stations) as mock_run:
            result = pipeline.run_step('validate', ['hydroweb'])

            mock_run.assert_called_once_with(['hydroweb'])

    def test_invalid_step_raises_error(self, mock_config):
        """Test that invalid step name raises ValueError."""
        from src.pipeline import Pipeline

        pipeline = Pipeline(mock_config)

        with pytest.raises(ValueError, match="Unknown step"):
            pipeline.run_step('invalid_step', ['hydroweb'])


class TestPipelineWithMockedData:
    """Test pipeline with mocked external data sources."""

    def test_pipeline_creates_output_on_success(self, temp_output_dir):
        """Test that pipeline creates output files on successful run."""
        from src.pipeline import Pipeline, PipelineResult
        from src.core.station import Station, StationList
        from src.steps import Step0Download, Step1Validate, Step2CaMa, Step3Reserved, Step4Merge

        config = {
            'data_root': '/tmp',
            'output_dir': str(temp_output_dir),
            'skip_download': True,
            'resolutions': ['glb_15min'],
        }

        # Create test stations
        stations = StationList()
        stations.add(Station(
            id='TEST001',
            name='Test Station',
            lon=10.0,
            lat=50.0,
            source='hydroweb',
            elevation=100.0,
            num_observations=50
        ))

        # Mock all steps except merge (which actually writes files)
        with patch.object(Step0Download, 'run', return_value={}), \
             patch.object(Step1Validate, 'run', return_value=stations), \
             patch.object(Step2CaMa, 'run', return_value=stations), \
             patch.object(Step3Reserved, 'run', return_value=stations):

            pipeline = Pipeline(config)
            result = pipeline.run(['hydroweb'])

            # Verify result is PipelineResult
            assert isinstance(result, PipelineResult)
            assert result.success is True

            # Verify merge step returned files
            assert 'merge' in result.step_results
            assert 'files' in result.step_results['merge']

            # Verify output file was created
            output_files = list(temp_output_dir.glob('*.txt'))
            assert len(output_files) > 0, "Expected output files to be created"

    def test_pipeline_handles_empty_station_list(self, temp_output_dir):
        """Test pipeline behavior with empty station list."""
        from src.pipeline import Pipeline, PipelineResult
        from src.core.station import StationList
        from src.steps import Step0Download, Step1Validate, Step2CaMa, Step3Reserved

        config = {
            'data_root': '/tmp',
            'output_dir': str(temp_output_dir),
            'skip_download': True,
        }

        # Return empty station list
        empty_stations = StationList()

        with patch.object(Step0Download, 'run', return_value={}), \
             patch.object(Step1Validate, 'run', return_value=empty_stations), \
             patch.object(Step2CaMa, 'run', return_value=empty_stations), \
             patch.object(Step3Reserved, 'run', return_value=empty_stations):

            pipeline = Pipeline(config)
            result = pipeline.run(['hydroweb'])

            # Pipeline should complete without errors
            assert isinstance(result, PipelineResult)
            assert result.success is True
            assert 'validate' in result.step_results
            assert result.step_results['validate']['total'] == 0
            assert result.stations_processed == 0


class TestPipelineCheckpointing:
    """Test pipeline checkpoint functionality."""

    def test_checkpoint_saves_after_each_step(self):
        """Test that checkpoint is saved after each step."""
        from src.pipeline import Pipeline
        from src.core.station import StationList
        from src.steps import Step0Download, Step1Validate, Step2CaMa, Step3Reserved, Step4Merge

        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'data_root': '/tmp',
                'output_dir': tmpdir,
                'skip_download': True,
            }

            stations = StationList()

            with patch.object(Step0Download, 'run', return_value={}), \
                 patch.object(Step1Validate, 'run', return_value=stations), \
                 patch.object(Step2CaMa, 'run', return_value=stations), \
                 patch.object(Step3Reserved, 'run', return_value=stations), \
                 patch.object(Step4Merge, 'run', return_value=[]):

                pipeline = Pipeline(config)

                # Spy on checkpoint.save
                save_calls = []
                original_save = pipeline.checkpoint.save

                def track_save(step, results):
                    save_calls.append(step)
                    return original_save(step, results)

                pipeline.checkpoint.save = track_save

                pipeline.run(['hydroweb'])

                # Verify checkpoint was called for each step
                assert 'download' in save_calls
                assert 'validate' in save_calls
                assert 'cama' in save_calls
                assert 'reserved' in save_calls
                assert 'merge' in save_calls


class TestCLIIntegration:
    """Integration tests for CLI interface."""

    def test_cli_with_config_file(self):
        """Test CLI with configuration file."""
        from src.main import main
        import yaml

        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create temporary config file
            config_path = Path(tmpdir) / 'test_config.yaml'
            config_data = {
                'data_root': '/tmp/test_data',
                'output_dir': str(Path(tmpdir) / 'output'),
            }

            with open(config_path, 'w') as f:
                yaml.dump(config_data, f)

            result = runner.invoke(main, [
                '--config', str(config_path),
                '--dry-run'
            ])

            assert result.exit_code == 0

    def test_cli_source_parsing(self):
        """Test CLI parses multiple sources correctly."""
        from src.main import main

        runner = CliRunner()

        result = runner.invoke(main, [
            '--source', 'hydroweb,cgls,icesat',
            '--dry-run'
        ])

        assert result.exit_code == 0
        # Sources should be mentioned in output
        output_lower = result.output.lower()
        assert 'hydroweb' in output_lower or 'cgls' in output_lower or 'sources' in output_lower

    def test_cli_step_only_mode(self):
        """Test CLI with --step option for running single step."""
        from src.main import main

        runner = CliRunner()

        for step in ['download', 'validate', 'cama', 'reserved', 'merge']:
            result = runner.invoke(main, [
                '--step', step,
                '--dry-run'
            ])

            assert result.exit_code == 0, f"Step {step} failed: {result.output}"


class TestPipelineExecution:
    """Test actual pipeline execution via CLI."""

    def test_cli_runs_pipeline_with_mocked_steps(self):
        """Test CLI actually runs the pipeline with mocked steps."""
        from src.main import main
        from src.pipeline import Pipeline, PipelineResult

        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a mock result
            mock_result = PipelineResult(
                success=True,
                stations_processed=10,
                output_files=[str(Path(tmpdir) / 'output.txt')],
                step_results={'validate': {'total': 10}}
            )

            with patch.object(Pipeline, 'run', return_value=mock_result) as mock_run:
                result = runner.invoke(main, [
                    '--source', 'hydroweb',
                    '--output', tmpdir,
                    '--skip-download'
                ])

                # Pipeline.run should be called
                mock_run.assert_called_once()

                # Check success message in output
                assert result.exit_code == 0
                assert 'completed successfully' in result.output or 'Pipeline' in result.output

    def test_cli_runs_single_step_with_mocked_handler(self):
        """Test CLI runs a single step correctly."""
        from src.main import main
        from src.pipeline import Pipeline, PipelineResult

        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_result = PipelineResult(
                success=True,
                stations_processed=5,
                output_files=[],
                step_results={'validate': {'total': 5}}
            )

            with patch.object(Pipeline, 'run_step', return_value=mock_result) as mock_run_step:
                result = runner.invoke(main, [
                    '--source', 'hydroweb',
                    '--output', tmpdir,
                    '--step', 'validate'
                ])

                # Pipeline.run_step should be called with correct args
                mock_run_step.assert_called_once()
                call_args = mock_run_step.call_args
                assert call_args[0][0] == 'validate'  # step name
                assert 'hydroweb' in call_args[0][1]  # sources

                assert result.exit_code == 0

    def test_cli_exits_on_pipeline_failure(self):
        """Test CLI exits with error code on pipeline failure."""
        from src.main import main
        from src.pipeline import Pipeline, PipelineResult

        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_result = PipelineResult(
                success=False,
                stations_processed=0,
                output_files=[],
                error="Validation failed: no stations found"
            )

            with patch.object(Pipeline, 'run', return_value=mock_result):
                result = runner.invoke(main, [
                    '--source', 'hydroweb',
                    '--output', tmpdir
                ])

                # Should exit with error
                assert result.exit_code == 1
                assert 'failed' in result.output.lower() or 'error' in result.output.lower()

    def test_cli_logs_output_files_on_success(self):
        """Test CLI logs output files on successful completion."""
        from src.main import main
        from src.pipeline import Pipeline, PipelineResult

        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = str(Path(tmpdir) / 'hydroweb_stations.txt')
            mock_result = PipelineResult(
                success=True,
                stations_processed=25,
                output_files=[output_file],
                step_results={'merge': {'files': [output_file]}}
            )

            with patch.object(Pipeline, 'run', return_value=mock_result):
                result = runner.invoke(main, [
                    '--source', 'hydroweb',
                    '--output', tmpdir
                ])

                assert result.exit_code == 0
                # Should log stations processed
                assert '25' in result.output or 'processed' in result.output.lower()

    def test_pipeline_integration_full_flow(self, temp_output_dir, sample_station_list):
        """Test full pipeline integration flow with mocked step handlers."""
        from src.pipeline import Pipeline, PipelineResult
        from src.steps import Step0Download, Step1Validate, Step2CaMa, Step3Reserved, Step4Merge

        config = {
            'data_root': '/tmp',
            'output_dir': str(temp_output_dir),
            'skip_download': True,
            'resolutions': ['glb_15min'],
        }

        with patch.object(Step0Download, 'run', return_value={}), \
             patch.object(Step1Validate, 'run', return_value=sample_station_list), \
             patch.object(Step2CaMa, 'run', return_value=sample_station_list), \
             patch.object(Step3Reserved, 'run', return_value=sample_station_list), \
             patch.object(Step4Merge, 'run', return_value=[str(temp_output_dir / 'output.txt')]):

            pipeline = Pipeline(config)
            result = pipeline.run(['hydroweb'])

            # Verify PipelineResult
            assert isinstance(result, PipelineResult)
            assert result.success is True
            assert result.stations_processed == 1  # sample_station_list has 1 station
            assert len(result.output_files) == 1
            assert 'download' in result.step_results
            assert 'validate' in result.step_results
            assert 'cama' in result.step_results
            assert 'merge' in result.step_results
