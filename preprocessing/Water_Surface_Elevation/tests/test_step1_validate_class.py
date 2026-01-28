#!/usr/bin/env python3
"""
Tests for Step1Validate class and validation functions.

Tests cover:
- Step1Validate class initialization and run methods
- validate_station function
- compute_statistics function
- ValidationIssue and ValidationResult dataclasses
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.steps.step1_validate import (
    Step1Validate,
    validate_station,
    compute_statistics,
    ValidationIssue,
    ValidationResult,
    VALIDATION_RULES,
)
from src.core.station import Station, StationList


class TestStep1ValidateClass:
    """Tests for Step1Validate class."""

    def test_initialization_with_data_root(self, tmp_path):
        """Test initialization with data_root config."""
        config = {
            'data_root': str(tmp_path),
            'geoid_root': str(tmp_path / 'geoid'),
        }
        step = Step1Validate(config)
        assert step.data_root == tmp_path

    def test_initialization_without_data_root(self):
        """Test initialization uses default data_root."""
        config = {}
        step = Step1Validate(config)
        assert step.data_root == Path('./data')

    def test_initialization_without_geoid_root(self):
        """Test initialization handles missing geoid_root."""
        config = {'data_root': '/tmp'}
        step = Step1Validate(config)
        assert step.geoid_root is None

    def test_initialization_with_validation_rules(self):
        """Test initialization with custom validation rules."""
        custom_rules = {'min_observations': 50}
        config = {'validation': custom_rules}
        step = Step1Validate(config)
        assert step.validation == custom_rules

    def test_run_with_empty_sources(self, tmp_path):
        """Test run with empty sources list."""
        config = {
            'data_root': str(tmp_path),
            'global_paths': {'data_sources': {}},
        }
        step = Step1Validate(config)
        result = step.run([])
        assert isinstance(result, StationList)
        assert len(result) == 0


class TestValidateStationFunction:
    """Tests for validate_station function."""

    def test_valid_station_no_issues(self):
        """Test valid station returns no issues."""
        station = Station(
            id="TEST001",
            name="Test Station",
            lon=10.0,
            lat=50.0,
            num_observations=100,
            elevation=500.0,
            source="test"
        )
        rules = {
            'coordinates': {'lat_min': -90, 'lat_max': 90, 'lon_min': -180, 'lon_max': 180},
            'quality': {'min_observations': 10},
            'elevation': {'min_value': -500, 'max_value': 6000},
        }

        issues = validate_station(station, rules)
        errors = [i for i in issues if i.level == 'error']
        assert len(errors) == 0

    def test_invalid_latitude_returns_error(self):
        """Test invalid latitude returns error."""
        station = Station(
            id="TEST001",
            name="Test Station",
            lon=10.0,
            lat=100.0,  # Invalid: > 90
            num_observations=100,
            source="test"
        )
        rules = {
            'coordinates': {'lat_min': -90, 'lat_max': 90, 'lon_min': -180, 'lon_max': 180},
        }

        issues = validate_station(station, rules)
        lat_errors = [i for i in issues if i.code == 'INVALID_LATITUDE']
        assert len(lat_errors) == 1
        assert lat_errors[0].level == 'error'

    def test_invalid_longitude_returns_error(self):
        """Test invalid longitude returns error."""
        station = Station(
            id="TEST001",
            name="Test Station",
            lon=200.0,  # Invalid: > 180
            lat=50.0,
            num_observations=100,
            source="test"
        )
        rules = {
            'coordinates': {'lat_min': -90, 'lat_max': 90, 'lon_min': -180, 'lon_max': 180},
        }

        issues = validate_station(station, rules)
        lon_errors = [i for i in issues if i.code == 'INVALID_LONGITUDE']
        assert len(lon_errors) == 1
        assert lon_errors[0].level == 'error'

    def test_low_observations_returns_warning(self):
        """Test low observations returns warning."""
        station = Station(
            id="TEST001",
            name="Test Station",
            lon=10.0,
            lat=50.0,
            num_observations=5,  # Below threshold
            source="test"
        )
        rules = {
            'coordinates': {},
            'quality': {'min_observations': 10},
        }

        issues = validate_station(station, rules)
        obs_warnings = [i for i in issues if i.code == 'LOW_OBSERVATIONS']
        assert len(obs_warnings) == 1
        assert obs_warnings[0].level == 'warning'

    def test_unusual_elevation_returns_warning(self):
        """Test unusual elevation returns warning."""
        station = Station(
            id="TEST001",
            name="Test Station",
            lon=10.0,
            lat=50.0,
            num_observations=100,
            elevation=7000.0,  # Above max
            source="test"
        )
        rules = {
            'coordinates': {},
            'quality': {},
            'elevation': {'min_value': -500, 'max_value': 6000},
        }

        issues = validate_station(station, rules)
        elev_warnings = [i for i in issues if i.code == 'UNUSUAL_ELEVATION']
        assert len(elev_warnings) == 1
        assert elev_warnings[0].level == 'warning'


class TestComputeStatistics:
    """Tests for compute_statistics function."""

    def test_compute_stats_with_stations(self):
        """Test computing statistics with valid stations."""
        stations = [
            Station(id="S1", name="S1", lon=-10.0, lat=40.0, num_observations=50, elevation=100.0, source="test"),
            Station(id="S2", name="S2", lon=20.0, lat=60.0, num_observations=100, elevation=200.0, source="test"),
            Station(id="S3", name="S3", lon=5.0, lat=50.0, num_observations=75, elevation=150.0, source="test"),
        ]

        stats = compute_statistics(stations)

        assert stats['total_stations'] == 3
        assert stats['bbox']['west'] == -10.0
        assert stats['bbox']['east'] == 20.0
        assert stats['bbox']['south'] == 40.0
        assert stats['bbox']['north'] == 60.0
        assert stats['observations']['total'] == 225
        assert stats['observations']['min'] == 50
        assert stats['observations']['max'] == 100
        assert 'elevation' in stats

    def test_compute_stats_empty_list(self):
        """Test computing statistics with empty list."""
        stats = compute_statistics([])
        assert stats == {}

    def test_compute_stats_by_source(self):
        """Test computing statistics groups by source."""
        stations = [
            Station(id="S1", name="S1", lon=0, lat=0, num_observations=10, source="hydroweb"),
            Station(id="S2", name="S2", lon=0, lat=0, num_observations=10, source="hydroweb"),
            Station(id="S3", name="S3", lon=0, lat=0, num_observations=10, source="cgls"),
        ]

        stats = compute_statistics(stations)

        assert stats['by_source']['hydroweb'] == 2
        assert stats['by_source']['cgls'] == 1


class TestValidationDataclasses:
    """Tests for ValidationIssue and ValidationResult dataclasses."""

    def test_validation_issue_creation(self):
        """Test ValidationIssue can be created."""
        issue = ValidationIssue(
            level='error',
            code='TEST_ERROR',
            message='Test error message',
            station_id='STATION001'
        )
        assert issue.level == 'error'
        assert issue.code == 'TEST_ERROR'
        assert issue.message == 'Test error message'
        assert issue.station_id == 'STATION001'

    def test_validation_issue_without_station_id(self):
        """Test ValidationIssue can be created without station_id."""
        issue = ValidationIssue(
            level='warning',
            code='GLOBAL_WARNING',
            message='Global warning'
        )
        assert issue.station_id is None

    def test_validation_result_creation(self):
        """Test ValidationResult can be created."""
        stations = [Station(id="S1", name="S1", lon=0, lat=0, num_observations=10, source="test")]
        issues = [ValidationIssue('info', 'INFO', 'Info message')]
        stats = {'total': 1}

        result = ValidationResult(
            stations=stations,
            issues=issues,
            stats=stats
        )

        assert len(result.stations) == 1
        assert len(result.issues) == 1
        assert result.stats['total'] == 1


class TestDefaultValidationRules:
    """Tests for default validation rules constant."""

    def test_validation_rules_has_required_keys(self):
        """Test VALIDATION_RULES has required keys."""
        assert 'lon_range' in VALIDATION_RULES
        assert 'lat_range' in VALIDATION_RULES
        assert 'min_observations' in VALIDATION_RULES

    def test_validation_rules_values(self):
        """Test VALIDATION_RULES has correct values."""
        assert VALIDATION_RULES['lon_range'] == (-180, 180)
        assert VALIDATION_RULES['lat_range'] == (-90, 90)
        assert VALIDATION_RULES['min_observations'] == 10
