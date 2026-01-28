#!/usr/bin/env python3
"""
Additional tests for Step 0: Data Download.

Tests cover:
- Step0Download class methods
- DataStatus dataclass
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.steps.step0_download import Step0Download, DataStatus, FILE_PATTERNS


class TestDataStatusDataclass:
    """Tests for DataStatus dataclass."""

    def test_data_status_creation(self):
        """Test DataStatus can be created."""
        status = DataStatus(
            source='hydroweb',
            current_files=100,
            min_required=500,
            is_complete=False,
            path=Path('/data/hydroweb')
        )
        assert status.source == 'hydroweb'
        assert status.current_files == 100
        assert status.min_required == 500
        assert status.is_complete is False
        assert status.path == Path('/data/hydroweb')

    def test_data_status_complete(self):
        """Test DataStatus with complete data."""
        status = DataStatus(
            source='cgls',
            current_files=1000,
            min_required=500,
            is_complete=True,
            path=Path('/data/cgls')
        )
        assert status.is_complete is True


class TestFilePatterns:
    """Tests for FILE_PATTERNS constant."""

    def test_has_all_sources(self):
        """Test FILE_PATTERNS has all expected sources."""
        expected_sources = ['hydrosat', 'hydroweb', 'cgls', 'icesat']
        for source in expected_sources:
            assert source in FILE_PATTERNS

    def test_patterns_have_required_keys(self):
        """Test each pattern has required keys."""
        for source, pattern in FILE_PATTERNS.items():
            assert 'pattern' in pattern
            assert 'subdir' in pattern


class TestStep0DownloadInit:
    """Tests for Step0Download initialization."""

    def test_default_data_root(self):
        """Test default data_root is set."""
        step = Step0Download(config={})
        assert str(step.data_root) == '/Volumes/Data01/Altimetry'

    def test_custom_data_root(self, tmp_path):
        """Test custom data_root from config."""
        step = Step0Download(config={'data_root': str(tmp_path)})
        assert step.data_root == tmp_path


class TestCheckSourceCompleteness:
    """Tests for _check_source_completeness method."""

    def test_unknown_source_uses_defaults(self, tmp_path):
        """Test unknown source uses default rules."""
        step = Step0Download(config={'data_root': str(tmp_path)})
        status = step._check_source_completeness('unknown', tmp_path)

        # Should not crash and return a status
        assert status.source == 'unknown'
        assert status.min_required == 0  # Default from rules.get()

    def test_counts_files_correctly(self, tmp_path):
        """Test files are counted correctly."""
        # Create test directory for hydrosat (no subdir, files directly in path)
        data_dir = tmp_path / "hydrosat_data"
        data_dir.mkdir(parents=True)

        # Create some .txt files
        for i in range(10):
            (data_dir / f"file{i}.txt").write_text("test")

        step = Step0Download(config={'data_root': str(tmp_path)})
        status = step._check_source_completeness('hydrosat', data_dir)

        assert status.current_files == 10
