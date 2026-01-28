#!/usr/bin/env python3
"""
Tests for src/readers/__init__.py - Reader Factory Functions and Data Status

Tests cover:
- ensure_data_and_get_reader function
- check_data_status function
- print_data_status function
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.readers import (
    get_reader,
    ensure_data_and_get_reader,
    check_data_status,
    print_data_status,
    READERS,
    DOWNLOADERS,
)
from src.readers.hydroweb_reader import HydroWebReader


class TestEnsureDataAndGetReader:
    """Tests for ensure_data_and_get_reader function."""

    def test_invalid_source_raises_value_error(self):
        """Test that invalid source raises ValueError."""
        with pytest.raises(ValueError, match="unknown_source"):
            ensure_data_and_get_reader(
                source="unknown_source",
                config={},
            )

    def test_data_exists_returns_reader_and_path(self, tmp_path):
        """Test that existing data returns reader and path without downloading."""
        # Create a hydroweb test file
        data_dir = tmp_path / "hydroweb"
        data_dir.mkdir()
        test_file = data_dir / "hydroprd_test.txt"
        test_file.write_text("""#REFERENCE LONGITUDE:: 10.0
#REFERENCE LATITUDE:: 50.0
#ID:: TEST001
2020-01-01 00:00:00 100.0 0.1
""")

        config = {
            'global_paths': {
                'data_sources': {
                    'hydroweb': {'root': str(data_dir)}
                }
            }
        }

        reader, path = ensure_data_and_get_reader(
            source="hydroweb",
            config=config,
            logger=None,
        )

        assert isinstance(reader, HydroWebReader)
        assert path == data_dir

    def test_data_not_exists_with_auto_download_false_raises(self, tmp_path):
        """Test that missing data with auto_download=False raises FileNotFoundError."""
        config = {
            'global_paths': {
                'data_sources': {
                    'hydroweb': {'root': str(tmp_path / "nonexistent")}
                }
            }
        }

        with pytest.raises(FileNotFoundError, match="hydroweb"):
            ensure_data_and_get_reader(
                source="hydroweb",
                config=config,
                auto_download=False,
            )

    def test_uses_default_path_when_not_configured(self, tmp_path):
        """Test that default path is used when source path is not configured."""
        config = {
            'global_paths': {
                'data_sources': {}
            }
        }

        with pytest.raises(FileNotFoundError):
            ensure_data_and_get_reader(
                source="hydroweb",
                config=config,
                auto_download=False,
            )


class TestCheckDataStatus:
    """Tests for check_data_status function."""

    def test_returns_status_for_all_sources(self, tmp_path):
        """Test that status is returned for all configured sources."""
        # Create empty directories for all sources to avoid ReaderError
        for source in ['hydroweb', 'cgls', 'icesat', 'hydrosat']:
            (tmp_path / source).mkdir(parents=True, exist_ok=True)

        config = {
            'global_paths': {
                'data_sources': {
                    source: {'root': str(tmp_path / source)}
                    for source in ['hydroweb', 'cgls', 'icesat', 'hydrosat']
                }
            }
        }

        status = check_data_status(config)

        # Should have status for all known sources
        for source in READERS.keys():
            assert source in status
            assert 'exists' in status[source]
            assert 'path' in status[source]
            assert 'count' in status[source]
            assert 'requires_auth' in status[source]

    def test_detects_existing_data(self, tmp_path):
        """Test that existing data is detected correctly."""
        # Create hydroweb test data
        data_dir = tmp_path / "hydroweb"
        data_dir.mkdir()
        for i in range(3):
            test_file = data_dir / f"hydroprd_test{i}.txt"
            test_file.write_text(f"""#REFERENCE LONGITUDE:: {10.0 + i}
#REFERENCE LATITUDE:: 50.0
#ID:: TEST{i:03d}
2020-01-01 00:00:00 100.0 0.1
""")

        # Create empty directories for other sources
        for source in ['cgls', 'icesat', 'hydrosat']:
            (tmp_path / source).mkdir(parents=True, exist_ok=True)

        config = {
            'global_paths': {
                'data_sources': {
                    'hydroweb': {'root': str(data_dir)},
                    'cgls': {'root': str(tmp_path / 'cgls')},
                    'icesat': {'root': str(tmp_path / 'icesat')},
                    'hydrosat': {'root': str(tmp_path / 'hydrosat')},
                }
            }
        }

        status = check_data_status(config)

        assert status['hydroweb']['exists'] is True
        assert status['hydroweb']['count'] == 3

    def test_detects_missing_data(self, tmp_path):
        """Test that missing data is detected correctly."""
        # Create empty directories for all sources
        for source in ['hydroweb', 'cgls', 'icesat', 'hydrosat']:
            (tmp_path / source).mkdir(parents=True, exist_ok=True)

        config = {
            'global_paths': {
                'data_sources': {
                    source: {'root': str(tmp_path / source)}
                    for source in ['hydroweb', 'cgls', 'icesat', 'hydrosat']
                }
            }
        }

        status = check_data_status(config)

        # Empty directories should show exists=False
        assert status['hydroweb']['exists'] is False
        assert status['hydroweb']['count'] == 0


class TestPrintDataStatus:
    """Tests for print_data_status function."""

    def test_prints_status_output(self, tmp_path, capsys):
        """Test that print_data_status outputs to stdout."""
        # Create empty directories for all sources
        for source in ['hydroweb', 'cgls', 'icesat', 'hydrosat']:
            (tmp_path / source).mkdir(parents=True, exist_ok=True)

        config = {
            'global_paths': {
                'data_sources': {
                    source: {'root': str(tmp_path / source)}
                    for source in ['hydroweb', 'cgls', 'icesat', 'hydrosat']
                }
            }
        }

        print_data_status(config)

        captured = capsys.readouterr()
        assert "hydroweb" in captured.out or "Hydroweb" in captured.out.lower()


class TestReadersDictionaryMappings:
    """Tests for READERS and DOWNLOADERS dictionaries."""

    def test_readers_contains_all_sources(self):
        """Test that READERS contains all expected sources."""
        expected = ['hydroweb', 'cgls', 'icesat', 'hydrosat']
        for source in expected:
            assert source in READERS

    def test_downloaders_contains_all_sources(self):
        """Test that DOWNLOADERS contains all expected sources."""
        expected = ['hydroweb', 'cgls', 'icesat', 'hydrosat']
        for source in expected:
            assert source in DOWNLOADERS

    def test_readers_return_correct_types(self):
        """Test that READERS return correct reader classes."""
        from src.readers.hydroweb_reader import HydroWebReader
        from src.readers.cgls_reader import CGLSReader
        from src.readers.icesat_reader import ICESatReader
        from src.readers.hydrosat_reader import HydroSatReader

        assert READERS['hydroweb'] is HydroWebReader
        assert READERS['cgls'] is CGLSReader
        assert READERS['icesat'] is ICESatReader
        assert READERS['hydrosat'] is HydroSatReader
