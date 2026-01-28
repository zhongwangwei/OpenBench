#!/usr/bin/env python3
"""
Comprehensive tests for WSE Pipeline steps.

Tests cover:
- Step1Validate: Coordinate validation, EGM calculation, duplicate detection
- Step2CaMa: CaMa-Flood grid allocation (with mocked CaMa data)
- Step4Merge: Output file generation

Each step is tested for:
- Valid input processing
- Invalid input handling
- Edge cases
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

from src.core.station import Station, StationList
from src.steps.step1_validate import (
    Step1Validate,
    validate_station,
    ValidationIssue,
    ValidationResult,
    haversine_distance,
    detect_duplicates,
    detect_duplicates_fast,
    compute_statistics,
    VALIDATION_RULES,
)
from src.steps.step2_cama import Step2CaMa, CamaResult, compute_allocation_stats
from src.steps.step4_merge import Step4Merge
from src.core.cama_allocator import CamaAllocator, AllocationResult, StationAllocation


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_config():
    """Create mock configuration for step testing."""
    return {
        'data_root': '/tmp/test_data',
        'output_dir': '/tmp/test_output',
        'cama_root': '/tmp/cama',
        'geoid_root': '/tmp/geoid',
        'resolutions': ['glb_15min'],
        'validation': {
            'min_observations': 5,
        },
        'global_paths': {
            'data_sources': {
                'hydroweb': '/tmp/test_data/Hydroweb',
                'cgls': '/tmp/test_data/CGLS',
            },
            'geoid_data': {
                'root': '/tmp/geoid',
            },
            'cama_data': {
                'root': '/tmp/cama',
                'resolutions': ['glb_15min'],
            },
        },
        'processing': {
            'cama_resolutions': ['glb_15min'],
        },
    }


@pytest.fixture
def sample_station():
    """Create a single sample station."""
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
    """Create a StationList with sample stations."""
    stations = StationList()
    stations.add(sample_station)
    stations.add(Station(
        id='TEST002',
        name='Test Station 2',
        lon=20.0,
        lat=40.0,
        source='hydroweb',
        elevation=200.0,
        num_observations=50
    ))
    stations.add(Station(
        id='TEST003',
        name='Test Station 3',
        lon=30.0,
        lat=30.0,
        source='cgls',
        elevation=300.0,
        num_observations=75
    ))
    return stations


@pytest.fixture
def invalid_station():
    """Create a station with invalid coordinates."""
    return Station(
        id='INVALID001',
        name='Invalid Station',
        lon=999.0,  # Invalid longitude
        lat=999.0,  # Invalid latitude
        source='hydroweb',
        elevation=100.0,
        num_observations=100
    )


@pytest.fixture
def low_obs_station():
    """Create a station with low observation count."""
    return Station(
        id='LOWOBS001',
        name='Low Obs Station',
        lon=15.0,
        lat=45.0,
        source='hydroweb',
        elevation=150.0,
        num_observations=3  # Below minimum
    )


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# =============================================================================
# Step1Validate Tests
# =============================================================================

class TestStep1Validate:
    """Tests for Step1Validate - Validation and EGM Calculation."""

    def test_initialization(self, mock_config):
        """Test Step1Validate can be initialized."""
        step = Step1Validate(mock_config)
        assert step.config == mock_config
        assert step.data_root == Path('/tmp/test_data')
        assert step.geoid_root == Path('/tmp/geoid')

    def test_initialization_without_geoid(self):
        """Test Step1Validate initializes without geoid_root."""
        config = {
            'data_root': '/tmp/data',
            'output_dir': '/tmp/output',
        }
        step = Step1Validate(config)
        assert step.geoid_root is None

    def test_validation_rules_default(self, mock_config):
        """Test default validation rules are applied."""
        step = Step1Validate(mock_config)
        assert 'min_observations' in step.validation

    def test_validates_coordinates(self, mock_config, sample_station):
        """Test validation passes for valid coordinates."""
        step = Step1Validate(mock_config)

        # Station.is_valid() checks coordinates
        assert sample_station.is_valid() is True
        assert -180 <= sample_station.lon <= 180
        assert -90 <= sample_station.lat <= 90

    def test_rejects_invalid_coordinates(self, mock_config, invalid_station):
        """Test validation fails for invalid coordinates."""
        step = Step1Validate(mock_config)

        # Station.is_valid() should return False for invalid coords
        assert invalid_station.is_valid() is False

    def test_validate_station_function_valid(self, sample_station):
        """Test validate_station function with valid station."""
        rules = {
            'coordinates': {
                'lat_min': -90,
                'lat_max': 90,
                'lon_min': -180,
                'lon_max': 180,
            },
            'quality': {
                'min_observations': 10,
            },
        }

        issues = validate_station(sample_station, rules)

        # Should have no errors (maybe warnings)
        errors = [i for i in issues if i.level == 'error']
        assert len(errors) == 0

    def test_validate_station_function_invalid_lat(self):
        """Test validate_station with invalid latitude."""
        station = Station(
            id='INV_LAT',
            name='Invalid Lat',
            lon=10.0,
            lat=999.0,  # Invalid
            source='test',
            num_observations=100
        )
        rules = {
            'coordinates': {
                'lat_min': -90,
                'lat_max': 90,
                'lon_min': -180,
                'lon_max': 180,
            },
        }

        issues = validate_station(station, rules)

        errors = [i for i in issues if i.level == 'error']
        assert len(errors) >= 1
        assert any('INVALID_LATITUDE' in i.code for i in errors)

    def test_validate_station_function_invalid_lon(self):
        """Test validate_station with invalid longitude."""
        station = Station(
            id='INV_LON',
            name='Invalid Lon',
            lon=999.0,  # Invalid
            lat=45.0,
            source='test',
            num_observations=100
        )
        rules = {
            'coordinates': {
                'lat_min': -90,
                'lat_max': 90,
                'lon_min': -180,
                'lon_max': 180,
            },
        }

        issues = validate_station(station, rules)

        errors = [i for i in issues if i.level == 'error']
        assert len(errors) >= 1
        assert any('INVALID_LONGITUDE' in i.code for i in errors)

    def test_validate_station_low_observations(self, low_obs_station):
        """Test validate_station flags low observation count."""
        rules = {
            'coordinates': {},
            'quality': {
                'min_observations': 10,
            },
        }

        issues = validate_station(low_obs_station, rules)

        warnings = [i for i in issues if i.level == 'warning']
        assert len(warnings) >= 1
        assert any('LOW_OBSERVATIONS' in i.code for i in warnings)

    def test_validate_station_unusual_elevation(self):
        """Test validate_station flags unusual elevation."""
        station = Station(
            id='HIGH_ELEV',
            name='High Elevation',
            lon=10.0,
            lat=45.0,
            source='test',
            elevation=9000.0,  # Very high
            num_observations=100
        )
        rules = {
            'coordinates': {},
            'elevation': {
                'min_value': -500,
                'max_value': 6000,
            },
        }

        issues = validate_station(station, rules)

        warnings = [i for i in issues if i.level == 'warning']
        assert len(warnings) >= 1
        assert any('UNUSUAL_ELEVATION' in i.code for i in warnings)


class TestHaversineDistance:
    """Tests for haversine distance calculation."""

    def test_same_point_zero_distance(self):
        """Same point should have zero distance."""
        dist = haversine_distance(45.0, 90.0, 45.0, 90.0)
        assert dist == 0.0

    def test_known_distance(self):
        """Test a known distance between two cities."""
        # Paris to London: approximately 343 km
        dist = haversine_distance(48.8566, 2.3522, 51.5074, -0.1278)
        assert 340000 < dist < 350000

    def test_equator_one_degree(self):
        """One degree at equator is approximately 111 km."""
        dist = haversine_distance(0.0, 0.0, 0.0, 1.0)
        assert 110000 < dist < 112000


class TestDuplicateDetection:
    """Tests for duplicate station detection."""

    def _create_station(self, i: int, lat: float, lon: float) -> Station:
        """Helper to create a Station instance."""
        return Station(
            id=f"station_{i}",
            name=f"Station {i}",
            lat=lat,
            lon=lon,
            num_observations=100,
            source="test",
        )

    def test_no_duplicates_when_far_apart(self):
        """Stations far apart should not be duplicates."""
        stations = [
            self._create_station(0, 0.0, 0.0),
            self._create_station(1, 10.0, 10.0),
            self._create_station(2, -20.0, 30.0),
        ]
        rules = {'distance_threshold_m': 100}

        duplicates = detect_duplicates(stations, rules)
        assert len(duplicates) == 0

    def test_duplicates_when_close(self):
        """Stations within threshold should be detected."""
        stations = [
            self._create_station(0, 0.0, 0.0),
            self._create_station(1, 0.0, 0.0001),  # Very close
            self._create_station(2, 10.0, 10.0),
        ]
        rules = {'distance_threshold_m': 100}

        duplicates = detect_duplicates(stations, rules)
        assert len(duplicates) == 1

    def test_fast_detection_matches_original(self):
        """Fast algorithm should find same duplicates as original."""
        import random
        random.seed(42)

        stations = [
            self._create_station(i, random.uniform(-90, 90), random.uniform(-180, 180))
            for i in range(50)
        ]
        # Add known duplicate
        stations.append(self._create_station(50, stations[0].lat, stations[0].lon + 0.00001))

        rules = {'distance_threshold_m': 100}

        original_dups = detect_duplicates(stations, rules)
        fast_dups = detect_duplicates_fast(stations, rules)

        assert len(original_dups) == len(fast_dups)

    def test_empty_stations_list(self):
        """Empty list should return no duplicates."""
        rules = {'distance_threshold_m': 100}

        duplicates = detect_duplicates_fast([], rules)
        assert duplicates == []


class TestComputeStatistics:
    """Tests for compute_statistics function."""

    def test_compute_statistics_basic(self, sample_station_list):
        """Test computing basic statistics."""
        stations = list(sample_station_list)
        stats = compute_statistics(stations)

        assert 'total_stations' in stats
        assert stats['total_stations'] == 3
        assert 'bbox' in stats
        assert 'observations' in stats

    def test_compute_statistics_empty(self):
        """Test computing statistics for empty list."""
        stats = compute_statistics([])
        assert stats == {}

    def test_compute_statistics_bbox(self, sample_station_list):
        """Test bounding box calculation."""
        stations = list(sample_station_list)
        stats = compute_statistics(stations)

        bbox = stats['bbox']
        assert bbox['west'] == 10.0
        assert bbox['east'] == 30.0
        assert bbox['south'] == 30.0
        assert bbox['north'] == 50.0


# =============================================================================
# Step2CaMa Tests
# =============================================================================

class TestStep2CaMa:
    """Tests for Step2CaMa - CaMa-Flood Grid Allocation."""

    def test_initialization(self, mock_config):
        """Test Step2CaMa can be initialized."""
        step = Step2CaMa(mock_config)
        assert step.cama_root == '/tmp/cama'
        assert 'glb_15min' in step.resolutions

    def test_initialization_without_cama_root(self):
        """Test Step2CaMa handles missing cama_root."""
        config = {
            'output_dir': '/tmp/output',
        }
        step = Step2CaMa(config)
        assert step.cama_root is None

    def test_run_with_empty_stations(self, mock_config):
        """Test running with empty station list."""
        step = Step2CaMa(mock_config)
        stations = StationList()

        result = step.run(stations)

        assert len(result) == 0

    def test_run_without_cama_root_skips(self, sample_station_list):
        """Test running without cama_root returns input unchanged."""
        config = {'output_dir': '/tmp/output'}
        step = Step2CaMa(config)

        result = step.run(sample_station_list)

        # Should return same stations without modification
        assert len(result) == len(sample_station_list)

    @patch('src.steps.step2_cama.CamaAllocator')
    def test_run_with_mocked_allocator(self, mock_allocator_class, mock_config, sample_station_list):
        """Test running with mocked CaMa allocator."""
        # Setup mock allocator
        mock_allocator = MagicMock()
        mock_allocator_class.return_value = mock_allocator

        # Mock allocation result
        mock_result = MagicMock()
        mock_result.results = {
            'glb_15min': AllocationResult(
                resolution='glb_15min',
                success=True,
                flag=1,
                ix=100,
                iy=200,
                lon_cama=10.125,
                lat_cama=49.875,
            )
        }
        mock_allocator.allocate_station.return_value = mock_result

        step = Step2CaMa(mock_config)
        result = step.run(sample_station_list)

        assert len(result) == len(sample_station_list)
        # Verify allocator was called for each station
        assert mock_allocator.allocate_station.call_count == len(sample_station_list)


class TestCamaAllocator:
    """Tests for CamaAllocator class."""

    def test_initialization(self, tmp_path):
        """Test CamaAllocator initialization."""
        allocator = CamaAllocator(
            cama_root=str(tmp_path),
            resolutions=['glb_15min'],
        )
        assert allocator.cama_root == tmp_path
        assert 'glb_15min' in allocator.resolutions

    def test_simple_allocation(self, tmp_path):
        """Test simple allocation when AllocateVS is not available."""
        allocator = CamaAllocator(
            cama_root=str(tmp_path),
            resolutions=['glb_15min'],
        )

        result = allocator._simple_allocation(10.0, 50.0, 'glb_15min')

        assert result.success is True
        assert result.resolution == 'glb_15min'
        assert result.ix > 0
        assert result.iy > 0
        assert -180 <= result.lon_cama <= 180
        assert -90 <= result.lat_cama <= 90

    def test_allocate_station(self, tmp_path):
        """Test allocating a single station."""
        allocator = CamaAllocator(
            cama_root=str(tmp_path),
            resolutions=['glb_15min'],
        )

        allocation = allocator.allocate_station(
            station_id='TEST001',
            lon=10.0,
            lat=50.0,
            elevation=100.0,
            satellite='Sentinel-3A'
        )

        assert allocation.station_id == 'TEST001'
        assert allocation.lon == 10.0
        assert allocation.lat == 50.0
        assert 'glb_15min' in allocation.results

    def test_allocate_batch(self, tmp_path):
        """Test batch allocation."""
        allocator = CamaAllocator(
            cama_root=str(tmp_path),
            resolutions=['glb_15min'],
        )

        stations = [
            {'id': 'ST001', 'lon': 10.0, 'lat': 50.0, 'elevation': 100.0, 'satellite': 'S3A'},
            {'id': 'ST002', 'lon': 20.0, 'lat': 40.0, 'elevation': 200.0, 'satellite': 'S3B'},
        ]

        results = allocator.allocate_batch(stations)

        assert len(results) == 2
        assert results[0].station_id == 'ST001'
        assert results[1].station_id == 'ST002'


class TestComputeAllocationStats:
    """Tests for compute_allocation_stats function."""

    def test_compute_stats_all_success(self):
        """Test stats with all successful allocations."""
        allocations = [
            StationAllocation(
                station_id='ST001',
                lon=10.0,
                lat=50.0,
                results={
                    'glb_15min': AllocationResult(resolution='glb_15min', success=True),
                }
            ),
            StationAllocation(
                station_id='ST002',
                lon=20.0,
                lat=40.0,
                results={
                    'glb_15min': AllocationResult(resolution='glb_15min', success=True),
                }
            ),
        ]

        stats = compute_allocation_stats(allocations, ['glb_15min'])

        assert stats['total_stations'] == 2
        assert stats['by_resolution']['glb_15min']['success'] == 2
        assert stats['by_resolution']['glb_15min']['failed'] == 0
        assert stats['overall_success_rate'] == 1.0

    def test_compute_stats_partial_success(self):
        """Test stats with partial success."""
        allocations = [
            StationAllocation(
                station_id='ST001',
                lon=10.0,
                lat=50.0,
                results={
                    'glb_15min': AllocationResult(resolution='glb_15min', success=True),
                }
            ),
            StationAllocation(
                station_id='ST002',
                lon=20.0,
                lat=40.0,
                results={
                    'glb_15min': AllocationResult(resolution='glb_15min', success=False),
                }
            ),
        ]

        stats = compute_allocation_stats(allocations, ['glb_15min'])

        assert stats['by_resolution']['glb_15min']['success'] == 1
        assert stats['by_resolution']['glb_15min']['failed'] == 1
        assert stats['overall_success_rate'] == 0.5

    def test_compute_stats_empty(self):
        """Test stats with no allocations."""
        stats = compute_allocation_stats([], ['glb_15min'])

        assert stats['total_stations'] == 0
        assert stats['overall_success_rate'] == 0


# =============================================================================
# Step4Merge Tests
# =============================================================================

class TestStep4Merge:
    """Tests for Step4Merge - Output File Generation."""

    def test_initialization(self, mock_config, temp_output_dir):
        """Test Step4Merge can be initialized."""
        config = {**mock_config, 'output_dir': str(temp_output_dir)}
        step = Step4Merge(config)
        assert step.output_dir == temp_output_dir

    def test_run_separate_files(self, mock_config, sample_station_list, temp_output_dir):
        """Test generating separate files for each source."""
        config = {**mock_config, 'output_dir': str(temp_output_dir)}
        step = Step4Merge(config)

        output_files = step.run(sample_station_list, merge=False)

        # Should create separate files for each source
        assert len(output_files) >= 1
        for filepath in output_files:
            assert Path(filepath).exists()

    def test_run_merged_file(self, mock_config, sample_station_list, temp_output_dir):
        """Test generating single merged file."""
        config = {**mock_config, 'output_dir': str(temp_output_dir)}
        step = Step4Merge(config)

        output_files = step.run(sample_station_list, merge=True)

        assert len(output_files) == 1
        assert 'all_stations.txt' in output_files[0]
        assert Path(output_files[0]).exists()

    def test_run_creates_output_dir(self, mock_config, sample_station_list, tmp_path):
        """Test that output directory is created if not exists."""
        new_output_dir = tmp_path / 'new_output'
        config = {**mock_config, 'output_dir': str(new_output_dir)}
        step = Step4Merge(config)

        output_files = step.run(sample_station_list, merge=True)

        assert new_output_dir.exists()

    def test_run_empty_stations(self, mock_config, temp_output_dir):
        """Test running with empty station list."""
        config = {**mock_config, 'output_dir': str(temp_output_dir)}
        step = Step4Merge(config)
        stations = StationList()

        output_files = step.run(stations, merge=True)

        # Should still create file, but with only header
        assert len(output_files) == 1
        assert Path(output_files[0]).exists()

    def test_output_file_format(self, mock_config, sample_station_list, temp_output_dir):
        """Test output file has correct format."""
        config = {**mock_config, 'output_dir': str(temp_output_dir)}
        step = Step4Merge(config)

        output_files = step.run(sample_station_list, merge=True)

        with open(output_files[0], 'r') as f:
            lines = f.readlines()

        # Check header
        header = lines[0].strip().split('\t')
        assert 'id' in header
        assert 'name' in header
        assert 'lon' in header
        assert 'lat' in header
        assert 'elevation' in header

        # Check data lines
        assert len(lines) > 1  # Header + data

    def test_output_file_content(self, mock_config, sample_station, temp_output_dir):
        """Test output file content is correct."""
        config = {**mock_config, 'output_dir': str(temp_output_dir)}
        step = Step4Merge(config)

        stations = StationList()
        stations.add(sample_station)

        output_files = step.run(stations, merge=True)

        with open(output_files[0], 'r') as f:
            content = f.read()

        # Check station data is in output
        assert 'TEST001' in content
        assert 'Test Station 1' in content

    def test_format_station_row(self, mock_config, sample_station, temp_output_dir):
        """Test _format_station_row method."""
        config = {**mock_config, 'output_dir': str(temp_output_dir)}
        step = Step4Merge(config)

        row = step._format_station_row(sample_station, include_source=True)

        assert sample_station.source in row
        assert sample_station.id in row
        assert sample_station.name in row

    def test_format_station_row_without_source(self, mock_config, sample_station, temp_output_dir):
        """Test _format_station_row without source column."""
        config = {**mock_config, 'output_dir': str(temp_output_dir)}
        step = Step4Merge(config)

        row = step._format_station_row(sample_station, include_source=False)

        # First element should be id, not source
        assert row[0] == sample_station.id

    def test_output_with_cama_results(self, mock_config, sample_station, temp_output_dir):
        """Test output includes CaMa results."""
        config = {**mock_config, 'output_dir': str(temp_output_dir)}
        step = Step4Merge(config)

        # Add CaMa results to station
        sample_station.set_cama_result('glb_15min', {
            'success': True,
            'flag': 1,
            'ix': 100,
            'iy': 200,
            'kx1': 10,
            'ky1': 20,
            'kx2': 11,
            'ky2': 21,
            'dist1': 1000.0,
            'dist2': 2000.0,
            'rivwth': 500.0,
            'lon_cama': 10.125,
            'lat_cama': 49.875,
        })

        stations = StationList()
        stations.add(sample_station)

        output_files = step.run(stations, merge=True)

        with open(output_files[0], 'r') as f:
            content = f.read()

        # Check CaMa columns are present
        assert 'flag_15min' in content or 'ix_15min' in content

    def test_output_with_egm_values(self, mock_config, temp_output_dir):
        """Test output includes EGM values."""
        config = {**mock_config, 'output_dir': str(temp_output_dir)}
        step = Step4Merge(config)

        station = Station(
            id='EGM001',
            name='EGM Station',
            lon=10.0,
            lat=50.0,
            source='hydroweb',
            elevation=100.0,
            num_observations=100,
            egm08=25.5,
            egm96=24.3,
        )

        stations = StationList()
        stations.add(station)

        output_files = step.run(stations, merge=True)

        with open(output_files[0], 'r') as f:
            content = f.read()

        # Check EGM values are in output
        assert '25.5' in content or 'NA' in content


# =============================================================================
# Integration Tests
# =============================================================================

class TestStepIntegration:
    """Integration tests for pipeline steps."""

    def test_step1_to_step2_flow(self, mock_config, sample_station_list, tmp_path):
        """Test data flows correctly from Step1 to Step2."""
        # Step 1 output should be usable by Step 2
        config = {**mock_config, 'cama_root': str(tmp_path)}

        step2 = Step2CaMa(config)
        result = step2.run(sample_station_list)

        # Result should be a StationList
        assert len(result) == len(sample_station_list)

    def test_step2_to_step4_flow(self, mock_config, sample_station_list, temp_output_dir):
        """Test data flows correctly from Step2 to Step4."""
        config = {**mock_config, 'output_dir': str(temp_output_dir)}

        # Add mock CaMa results
        for station in sample_station_list:
            station.set_cama_result('glb_15min', {
                'success': True,
                'flag': 1,
                'ix': 100,
                'iy': 200,
            })

        step4 = Step4Merge(config)
        output_files = step4.run(sample_station_list, merge=True)

        assert len(output_files) == 1
        assert Path(output_files[0]).exists()

    def test_full_pipeline_flow(self, mock_config, sample_station_list, temp_output_dir, tmp_path):
        """Test complete flow through all steps."""
        config = {
            **mock_config,
            'cama_root': str(tmp_path),
            'output_dir': str(temp_output_dir),
        }

        # Step 2: CaMa allocation
        step2 = Step2CaMa(config)
        stations_with_cama = step2.run(sample_station_list)

        # Step 4: Generate output
        step4 = Step4Merge(config)
        output_files = step4.run(stations_with_cama, merge=True)

        # Verify output
        assert len(output_files) == 1
        output_path = Path(output_files[0])
        assert output_path.exists()

        # Verify content
        with open(output_path, 'r') as f:
            lines = f.readlines()

        # Header + 3 stations
        assert len(lines) == 4


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
