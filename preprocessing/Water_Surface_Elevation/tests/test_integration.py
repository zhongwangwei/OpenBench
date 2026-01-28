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

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration for testing."""
        return {
            'data_root': '/tmp/test_data',
            'output_dir': '/tmp/test_output',
            'cama_root': '/tmp/cama',
            'geoid_root': '/tmp/geoid',
            'skip_download': True,
            'resolutions': ['glb_15min'],
            'validation': {
                'min_observations': 5,
            },
        }

    def test_single_source_pipeline(self, mock_config):
        """Test pipeline with single data source using mocks.

        This test:
        1. Mocks data reading to avoid external dependencies
        2. Runs the pipeline for a single source
        3. Verifies the step sequence is correct
        """
        from src.pipeline import Pipeline
        from src.core.station import Station, StationList

        # Track which steps were called and in what order
        step_calls = []

        # Create mock stations
        mock_stations = StationList()
        mock_stations.add(Station(
            id='TEST001',
            name='Test Station 1',
            lon=10.0,
            lat=50.0,
            source='hydroweb',
            elevation=100.0,
            num_observations=100
        ))

        # Mock the step handlers
        with patch.object(Pipeline, '__init__', lambda self, config: None):
            pipeline = Pipeline.__new__(Pipeline)
            pipeline.config = mock_config
            pipeline.steps = ['download', 'validate', 'cama', 'reserved', 'merge']

            # Create mock handlers that track calls
            mock_download = MagicMock()
            mock_download.run = MagicMock(return_value={'hydroweb': True})

            mock_validate = MagicMock()
            mock_validate.run = MagicMock(return_value=mock_stations)

            mock_cama = MagicMock()
            mock_cama.run = MagicMock(return_value=mock_stations)

            mock_reserved = MagicMock()
            mock_reserved.run = MagicMock(return_value=mock_stations)

            mock_merge = MagicMock()
            mock_merge.run = MagicMock(return_value=['/tmp/test_output/hydroweb_stations.txt'])

            pipeline._step_handlers = {
                'download': mock_download,
                'validate': mock_validate,
                'cama': mock_cama,
                'reserved': mock_reserved,
                'merge': mock_merge,
            }

            # Mock checkpoint
            mock_checkpoint = MagicMock()
            pipeline.checkpoint = mock_checkpoint

            # Run the pipeline
            result = pipeline.run(['hydroweb'])

            # Verify step sequence
            mock_download.run.assert_called_once()
            mock_validate.run.assert_called_once_with(['hydroweb'])
            mock_cama.run.assert_called_once()
            mock_reserved.run.assert_called_once()
            mock_merge.run.assert_called_once()

            # Verify results structure
            assert 'download' in result
            assert 'validate' in result
            assert 'cama' in result
            assert 'merge' in result

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

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_pipeline_creates_output_on_success(self, temp_output_dir):
        """Test that pipeline creates output files on successful run."""
        from src.pipeline import Pipeline
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

            # Verify merge step returned files
            assert 'merge' in result
            assert 'files' in result['merge']

            # Verify output file was created
            output_files = list(temp_output_dir.glob('*.txt'))
            assert len(output_files) > 0, "Expected output files to be created"

    def test_pipeline_handles_empty_station_list(self, temp_output_dir):
        """Test pipeline behavior with empty station list."""
        from src.pipeline import Pipeline
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
            assert 'validate' in result
            assert result['validate']['total'] == 0


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
