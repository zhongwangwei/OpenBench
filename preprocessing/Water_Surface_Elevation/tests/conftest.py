#!/usr/bin/env python3
"""Shared test fixtures and constants for WSE Pipeline tests."""
import tempfile
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from src.core.station import Station, StationList


# =============================================================================
# Test Constants
# =============================================================================

VALID_SOURCES = ['hydrosat', 'hydroweb', 'cgls', 'icesat']
PIPELINE_STEPS = ['download', 'validate', 'cama', 'reserved', 'merge']
VALID_RESOLUTIONS = ['glb_01min', 'glb_03min', 'glb_05min', 'glb_06min', 'glb_15min']


# =============================================================================
# Configuration Fixtures
# =============================================================================

@pytest.fixture
def mock_config():
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


@pytest.fixture
def minimal_config():
    """Minimal configuration for basic tests."""
    return {
        'data_root': '/tmp',
        'output_dir': '/tmp/out',
    }


# =============================================================================
# Directory Fixtures
# =============================================================================

@pytest.fixture
def temp_output_dir():
    """Create temporary output directory that auto-cleans."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_config_dir():
    """Create temporary config directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# =============================================================================
# Station Fixtures
# =============================================================================

@pytest.fixture
def sample_station():
    """Create a single sample station for testing."""
    return Station(
        id='TEST001',
        name='Test Station 1',
        lon=10.0,
        lat=50.0,
        source='hydroweb',
        elevation=100.0,
        num_observations=100
    )


@pytest.fixture
def sample_station_list(sample_station):
    """Create a StationList with one sample station."""
    stations = StationList()
    stations.add(sample_station)
    return stations


@pytest.fixture
def multi_source_station_list():
    """Create a StationList with stations from multiple sources."""
    stations = StationList()
    stations.add(Station(
        id='HW001',
        name='Hydroweb Station',
        lon=10.0,
        lat=50.0,
        source='hydroweb',
        elevation=100.0,
        num_observations=100
    ))
    stations.add(Station(
        id='CGLS001',
        name='CGLS Station',
        lon=20.0,
        lat=40.0,
        source='cgls',
        elevation=200.0,
        num_observations=50
    ))
    stations.add(Station(
        id='ICE001',
        name='ICESat Station',
        lon=30.0,
        lat=30.0,
        source='icesat',
        elevation=300.0,
        num_observations=75
    ))
    return stations


@pytest.fixture
def empty_station_list():
    """Create an empty StationList."""
    return StationList()


# =============================================================================
# Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_checkpoint():
    """Create a mock checkpoint object."""
    mock = MagicMock()
    mock.save = MagicMock()
    mock.load_stations = MagicMock(return_value=None)
    return mock


@pytest.fixture
def mock_step_handlers(sample_station_list):
    """Create mock step handlers for pipeline testing."""
    mock_download = MagicMock()
    mock_download.run = MagicMock(return_value={'hydroweb': True})

    mock_validate = MagicMock()
    mock_validate.run = MagicMock(return_value=sample_station_list)

    mock_cama = MagicMock()
    mock_cama.run = MagicMock(return_value=sample_station_list)

    mock_reserved = MagicMock()
    mock_reserved.run = MagicMock(return_value=sample_station_list)

    mock_merge = MagicMock()
    mock_merge.run = MagicMock(return_value=['/tmp/test_output/hydroweb_stations.txt'])

    return {
        'download': mock_download,
        'validate': mock_validate,
        'cama': mock_cama,
        'reserved': mock_reserved,
        'merge': mock_merge,
    }
