#!/usr/bin/env python3
"""Tests for GeoidCalculator input validation.

Security tests to verify that subprocess calls are protected by input validation.
"""
import pytest
from unittest.mock import patch, MagicMock


class TestGeoidCalculatorValidation:
    """Test coordinate validation in GeoidCalculator."""

    @pytest.fixture
    def calculator(self):
        """Create a GeoidCalculator instance with mocked initialization."""
        from src.core.geoid_calculator import GeoidCalculator

        # Patch PyGeodesy to avoid needing actual data files
        with patch.object(GeoidCalculator, '_init_pygeodesy'):
            calc = GeoidCalculator(data_dir='/tmp/test_geoid')
            calc._pygeodesy_available = False  # Force CLI backend for testing
            yield calc

    # =========================================================================
    # Invalid Type Tests
    # =========================================================================

    def test_rejects_string_latitude(self, calculator):
        """Test that string latitude is rejected."""
        with pytest.raises(ValueError, match="Invalid coordinate type"):
            calculator.get_undulation("not_a_number", 10.0)

    def test_rejects_string_longitude(self, calculator):
        """Test that string longitude is rejected."""
        with pytest.raises(ValueError, match="Invalid coordinate type"):
            calculator.get_undulation(10.0, "not_a_number")

    def test_rejects_none_latitude(self, calculator):
        """Test that None latitude is rejected."""
        with pytest.raises(ValueError, match="Invalid coordinate type"):
            calculator.get_undulation(None, 10.0)

    def test_rejects_none_longitude(self, calculator):
        """Test that None longitude is rejected."""
        with pytest.raises(ValueError, match="Invalid coordinate type"):
            calculator.get_undulation(10.0, None)

    def test_rejects_list_input(self, calculator):
        """Test that list input is rejected."""
        with pytest.raises(ValueError, match="Invalid coordinate type"):
            calculator.get_undulation([10.0], 20.0)

    def test_rejects_dict_input(self, calculator):
        """Test that dict input is rejected."""
        with pytest.raises(ValueError, match="Invalid coordinate type"):
            calculator.get_undulation({"lat": 10.0}, 20.0)

    # =========================================================================
    # Out of Range Tests
    # =========================================================================

    def test_rejects_latitude_above_90(self, calculator):
        """Test that latitude > 90 is rejected."""
        with pytest.raises(ValueError, match="Latitude out of range"):
            calculator.get_undulation(91.0, 0.0)

    def test_rejects_latitude_below_minus_90(self, calculator):
        """Test that latitude < -90 is rejected."""
        with pytest.raises(ValueError, match="Latitude out of range"):
            calculator.get_undulation(-91.0, 0.0)

    def test_rejects_longitude_above_180(self, calculator):
        """Test that longitude > 180 is rejected."""
        with pytest.raises(ValueError, match="Longitude out of range"):
            calculator.get_undulation(0.0, 181.0)

    def test_rejects_longitude_below_minus_180(self, calculator):
        """Test that longitude < -180 is rejected."""
        with pytest.raises(ValueError, match="Longitude out of range"):
            calculator.get_undulation(0.0, -181.0)

    def test_rejects_extreme_latitude(self, calculator):
        """Test that extremely large latitude is rejected."""
        with pytest.raises(ValueError, match="Latitude out of range"):
            calculator.get_undulation(1000.0, 0.0)

    def test_rejects_extreme_longitude(self, calculator):
        """Test that extremely large longitude is rejected."""
        with pytest.raises(ValueError, match="Longitude out of range"):
            calculator.get_undulation(0.0, 1000.0)

    # =========================================================================
    # Boundary Tests (Valid Inputs)
    # =========================================================================

    def test_accepts_latitude_90(self, calculator):
        """Test that latitude = 90 is accepted (boundary)."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="0.0")
            # Should not raise - will attempt subprocess call
            try:
                calculator.get_undulation(90.0, 0.0)
            except Exception as e:
                # Allow subprocess failures, but validation should pass
                assert "Latitude out of range" not in str(e)
                assert "Invalid coordinate type" not in str(e)

    def test_accepts_latitude_minus_90(self, calculator):
        """Test that latitude = -90 is accepted (boundary)."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="0.0")
            try:
                calculator.get_undulation(-90.0, 0.0)
            except Exception as e:
                assert "Latitude out of range" not in str(e)
                assert "Invalid coordinate type" not in str(e)

    def test_accepts_longitude_180(self, calculator):
        """Test that longitude = 180 is accepted (boundary)."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="0.0")
            try:
                calculator.get_undulation(0.0, 180.0)
            except Exception as e:
                assert "Longitude out of range" not in str(e)
                assert "Invalid coordinate type" not in str(e)

    def test_accepts_longitude_minus_180(self, calculator):
        """Test that longitude = -180 is accepted (boundary)."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="0.0")
            try:
                calculator.get_undulation(0.0, -180.0)
            except Exception as e:
                assert "Longitude out of range" not in str(e)
                assert "Invalid coordinate type" not in str(e)

    def test_accepts_zero_coordinates(self, calculator):
        """Test that (0, 0) is accepted."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="0.0")
            try:
                calculator.get_undulation(0.0, 0.0)
            except Exception as e:
                assert "out of range" not in str(e)
                assert "Invalid coordinate type" not in str(e)

    # =========================================================================
    # Type Coercion Tests (Valid String Numbers)
    # =========================================================================

    def test_accepts_numeric_string_latitude(self, calculator):
        """Test that numeric string is coerced to float."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="0.0")
            try:
                calculator.get_undulation("45.5", 10.0)
            except Exception as e:
                # Should either work or fail validation - not subprocess error
                assert "Invalid coordinate type" not in str(e) or "45.5" in str(e)

    def test_accepts_integer_input(self, calculator):
        """Test that integer input is accepted."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="0.0")
            try:
                calculator.get_undulation(45, 10)
            except Exception as e:
                assert "Invalid coordinate type" not in str(e)
                assert "out of range" not in str(e)

    # =========================================================================
    # Injection Prevention Tests
    # =========================================================================

    def test_rejects_shell_injection_in_latitude(self, calculator):
        """Test that shell injection attempts are rejected."""
        with pytest.raises(ValueError, match="Invalid coordinate type"):
            calculator.get_undulation("45; rm -rf /", 10.0)

    def test_rejects_command_injection_in_longitude(self, calculator):
        """Test that command injection attempts are rejected."""
        with pytest.raises(ValueError, match="Invalid coordinate type"):
            calculator.get_undulation(10.0, "10 && echo hacked")

    def test_rejects_pipe_injection(self, calculator):
        """Test that pipe injection attempts are rejected."""
        with pytest.raises(ValueError, match="Invalid coordinate type"):
            calculator.get_undulation("10 | cat /etc/passwd", 10.0)

    # =========================================================================
    # Batch Processing Validation
    # =========================================================================

    def test_batch_rejects_invalid_coordinates(self, calculator):
        """Test that batch processing validates each coordinate."""
        coords = [
            (10.0, 20.0),      # Valid
            ("invalid", 30.0), # Invalid
            (40.0, 50.0),      # Valid
        ]
        with pytest.raises(ValueError, match="Invalid coordinate type"):
            calculator.get_undulation_batch(coords)

    def test_batch_rejects_out_of_range(self, calculator):
        """Test that batch processing rejects out-of-range coordinates."""
        coords = [
            (10.0, 20.0),   # Valid
            (95.0, 30.0),   # Invalid latitude
        ]
        with pytest.raises(ValueError, match="Latitude out of range"):
            calculator.get_undulation_batch(coords)


class TestValidateCoordinatesFunction:
    """Direct tests for the _validate_coordinates method."""

    @pytest.fixture
    def calculator(self):
        """Create a GeoidCalculator instance."""
        from src.core.geoid_calculator import GeoidCalculator

        with patch.object(GeoidCalculator, '_init_pygeodesy'):
            calc = GeoidCalculator(data_dir='/tmp/test_geoid')
            yield calc

    def test_returns_tuple_of_floats(self, calculator):
        """Test that validation returns a tuple of floats."""
        result = calculator._validate_coordinates(45.5, 10.5)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)

    def test_converts_integers_to_floats(self, calculator):
        """Test that integers are converted to floats."""
        lat, lon = calculator._validate_coordinates(45, 10)
        assert lat == 45.0
        assert lon == 10.0
        assert isinstance(lat, float)
        assert isinstance(lon, float)

    def test_preserves_precision(self, calculator):
        """Test that precision is preserved."""
        lat, lon = calculator._validate_coordinates(45.123456789, 10.987654321)
        assert abs(lat - 45.123456789) < 1e-9
        assert abs(lon - 10.987654321) < 1e-9
