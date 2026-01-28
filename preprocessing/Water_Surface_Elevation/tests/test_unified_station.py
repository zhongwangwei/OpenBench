#!/usr/bin/env python3
"""
Tests for unified Station and StationMetadata types.

These tests verify that:
1. StationMetadata is an alias for Station
2. Readers return Station objects directly
3. No conversion is needed in step1_validate
"""

import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.station import Station, StationList
from src.readers.base_reader import StationMetadata
from src.readers import get_reader


class TestStationMetadataAlias:
    """Test that StationMetadata is an alias for Station."""

    def test_station_metadata_is_station(self):
        """StationMetadata should be the same class as Station."""
        assert StationMetadata is Station

    def test_station_metadata_instance_is_station(self):
        """Instance created with StationMetadata should be Station instance."""
        sm = StationMetadata(
            id="test_001",
            name="Test Station",
            lon=100.5,
            lat=30.2,
            source="hydroweb"
        )
        assert isinstance(sm, Station)

    def test_station_has_all_fields(self):
        """Station should have all fields needed by readers."""
        # These are the fields that were in StationMetadata
        s = Station(
            id="test_001",
            name="Test Station",
            lon=100.5,
            lat=30.2,
            source="hydroweb",
            elevation=150.0,
            num_observations=100,
            egm08=25.5,
            egm96=24.8,
        )

        # Core fields
        assert s.id == "test_001"
        assert s.name == "Test Station"
        assert s.lon == 100.5
        assert s.lat == 30.2
        assert s.source == "hydroweb"

        # Elevation and observation fields
        assert s.elevation == 150.0
        assert s.num_observations == 100

        # EGM fields
        assert s.egm08 == 25.5
        assert s.egm96 == 24.8

    def test_station_metadata_fields(self):
        """Station should support metadata dict for extra fields."""
        s = Station(
            id="test_001",
            name="Test Station",
            lon=100.5,
            lat=30.2,
            source="hydroweb",
            metadata={
                'river': 'Amazon',
                'basin': 'Amazon Basin',
                'country': 'Brazil',
                'satellite': 'Sentinel-3A',
                'start_date': '2020-01-01',
                'end_date': '2023-12-31',
                'filepath': '/path/to/file.txt',
            }
        )

        assert s.metadata['river'] == 'Amazon'
        assert s.metadata['basin'] == 'Amazon Basin'
        assert s.metadata['country'] == 'Brazil'


class TestReaderReturnsStation:
    """Test that readers return Station objects directly."""

    def test_reader_returns_station_type(self):
        """Reader should return Station objects, not StationMetadata."""
        # This test verifies the type annotation and behavior
        # We can't test actual reading without real data, but we verify
        # that the import structure is correct
        from src.readers.base_reader import BaseReader

        # Verify the abstract method signature expects Station return type
        import inspect
        sig = inspect.signature(BaseReader.read_station)
        # The return annotation should reference StationMetadata which is Station
        assert 'StationMetadata' in str(sig.return_annotation) or 'Station' in str(sig.return_annotation)


class TestNoConversionNeeded:
    """Test that step1_validate doesn't need conversion between types."""

    def test_station_can_be_added_to_station_list(self):
        """Station objects should work directly with StationList."""
        station_list = StationList()

        # Create station as if it came from a reader
        station = Station(
            id="test_001",
            name="Test Station",
            lon=100.5,
            lat=30.2,
            source="hydroweb",
            elevation=150.0,
            num_observations=100,
        )

        station_list.add(station)
        assert len(station_list) == 1

        # Verify the station is in the list
        for s in station_list:
            assert s.id == "test_001"
            assert isinstance(s, Station)

    def test_station_is_valid_method_works(self):
        """Station.is_valid() should work correctly."""
        # Valid station
        valid = Station(
            id="test_001",
            name="Test",
            lon=100.5,
            lat=30.2,
            source="test"
        )
        assert valid.is_valid()

        # Invalid longitude
        invalid_lon = Station(
            id="test_002",
            name="Test",
            lon=200.0,  # Invalid
            lat=30.2,
            source="test"
        )
        assert not invalid_lon.is_valid()

        # Invalid latitude
        invalid_lat = Station(
            id="test_003",
            name="Test",
            lon=100.5,
            lat=95.0,  # Invalid
            source="test"
        )
        assert not invalid_lat.is_valid()


class TestBackwardCompatibility:
    """Test backward compatibility after unification."""

    def test_import_station_metadata_from_readers(self):
        """Should be able to import StationMetadata from readers module."""
        from src.readers import StationMetadata
        assert StationMetadata is Station

    def test_import_station_metadata_from_base_reader(self):
        """Should be able to import StationMetadata from base_reader."""
        from src.readers.base_reader import StationMetadata
        assert StationMetadata is Station

    def test_existing_station_code_works(self):
        """Existing code using Station should still work."""
        from src.core.station import Station as CoreStation

        s = CoreStation(
            id="test",
            name="Test",
            lon=100.0,
            lat=30.0,
            source="test"
        )

        s.set_cama_result("15sec", {"ix": 100, "iy": 200})
        assert s.cama_results["15sec"]["ix"] == 100


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
