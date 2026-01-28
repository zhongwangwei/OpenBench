#!/usr/bin/env python3
"""Tests for WSE Pipeline CLI."""
import pytest
from click.testing import CliRunner
from src.main import main


class TestCLI:
    """Test suite for CLI entry point."""

    def test_cli_help(self):
        """Test that --help shows usage information."""
        runner = CliRunner()
        result = runner.invoke(main, ['--help'])
        assert result.exit_code == 0
        assert 'WSE Pipeline' in result.output
        assert '--source' in result.output
        assert '--config' in result.output
        assert '--dry-run' in result.output

    def test_cli_dry_run(self):
        """Test dry-run mode exits cleanly."""
        runner = CliRunner()
        result = runner.invoke(main, ['--source', 'hydroweb', '--dry-run'])
        assert result.exit_code == 0
        assert 'DRY RUN' in result.output or 'hydroweb' in result.output.lower()

    def test_cli_source_all(self):
        """Test source=all parses correctly."""
        runner = CliRunner()
        result = runner.invoke(main, ['--source', 'all', '--dry-run'])
        assert result.exit_code == 0

    def test_cli_multiple_sources(self):
        """Test comma-separated sources."""
        runner = CliRunner()
        result = runner.invoke(main, ['--source', 'hydroweb,cgls', '--dry-run'])
        assert result.exit_code == 0

    def test_cli_invalid_config(self):
        """Test that invalid config path fails gracefully."""
        runner = CliRunner()
        result = runner.invoke(main, ['--config', '/nonexistent/path.yaml'])
        # Click's exists=True should catch this
        assert result.exit_code != 0

    def test_cli_step_option(self):
        """Test --step option with valid choices."""
        runner = CliRunner()
        for step in ['download', 'validate', 'cama', 'reserved', 'merge']:
            result = runner.invoke(main, ['--step', step, '--dry-run'])
            assert result.exit_code == 0

    def test_cli_invalid_step(self):
        """Test --step with invalid value."""
        runner = CliRunner()
        result = runner.invoke(main, ['--step', 'invalid'])
        assert result.exit_code != 0

    def test_cli_log_levels(self):
        """Test different log levels."""
        runner = CliRunner()
        for level in ['DEBUG', 'INFO', 'WARNING', 'ERROR']:
            result = runner.invoke(main, ['--log-level', level, '--dry-run'])
            assert result.exit_code == 0

    def test_cli_num_workers(self):
        """Test --num-workers option."""
        runner = CliRunner()
        result = runner.invoke(main, ['-j', '10', '--dry-run'])
        assert result.exit_code == 0

    def test_cli_output_option(self):
        """Test --output option."""
        runner = CliRunner()
        result = runner.invoke(main, ['--output', '/tmp/wse_output', '--dry-run'])
        assert result.exit_code == 0

    def test_cli_merge_flag(self):
        """Test --merge flag."""
        runner = CliRunner()
        result = runner.invoke(main, ['--merge', '--dry-run'])
        assert result.exit_code == 0

    def test_cli_skip_download_flag(self):
        """Test --skip-download flag."""
        runner = CliRunner()
        result = runner.invoke(main, ['--skip-download', '--dry-run'])
        assert result.exit_code == 0

    def test_cli_combined_options(self):
        """Test multiple options together."""
        runner = CliRunner()
        result = runner.invoke(main, [
            '--source', 'hydroweb,cgls',
            '--output', '/tmp/test',
            '--merge',
            '--skip-download',
            '-j', '8',
            '--log-level', 'DEBUG',
            '--dry-run'
        ])
        assert result.exit_code == 0
