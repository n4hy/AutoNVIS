"""
Unit tests for data validation
"""

import pytest
from datetime import datetime, timedelta
from src.ingestion.common.data_validator import DataValidator


@pytest.fixture
def validator():
    """Create validator for testing"""
    return DataValidator(staleness_threshold_sec=300)


def test_timestamp_validation(validator):
    """Test timestamp freshness validation"""
    # Current timestamp should be valid
    now = datetime.utcnow().isoformat() + 'Z'
    is_valid, error = validator.validate_timestamp(now)
    assert is_valid is True

    # Old timestamp should be invalid
    old = (datetime.utcnow() - timedelta(hours=1)).isoformat() + 'Z'
    is_valid, error = validator.validate_timestamp(old)
    assert is_valid is False
    assert "too old" in error

    # Future timestamp should be invalid
    future = (datetime.utcnow() + timedelta(hours=1)).isoformat() + 'Z'
    is_valid, error = validator.validate_timestamp(future)
    assert is_valid is False
    assert "future" in error


def test_xray_flux_validation(validator):
    """Test X-ray flux validation"""
    # Valid fluxes
    assert validator.validate_xray_flux(1e-6)[0] is True  # C-class
    assert validator.validate_xray_flux(5e-5)[0] is True  # X-class

    # Invalid fluxes
    assert validator.validate_xray_flux(1e-12)[0] is False  # Too low
    assert validator.validate_xray_flux(1e-1)[0] is False   # Too high
    assert validator.validate_xray_flux(float('nan'))[0] is False  # NaN


def test_solar_wind_validation(validator):
    """Test solar wind parameter validation"""
    # Valid data
    valid_data = {
        'proton_density': 5.0,
        'bulk_speed': 400.0,
        'proton_temperature': 1e5,
        'bx_gsm': 2.0,
        'by_gsm': -3.0,
        'bz_gsm': 5.0,
        'bt': 6.0
    }
    is_valid, errors = validator.validate_solar_wind(valid_data)
    assert is_valid is True

    # Invalid density
    invalid_data = {**valid_data, 'proton_density': 1000}
    is_valid, errors = validator.validate_solar_wind(invalid_data)
    assert is_valid is False
    assert any('Density' in e for e in errors)

    # Invalid velocity
    invalid_data = {**valid_data, 'bulk_speed': 5000}
    is_valid, errors = validator.validate_solar_wind(invalid_data)
    assert is_valid is False
    assert any('Velocity' in e for e in errors)


def test_tec_validation(validator):
    """Test TEC validation"""
    # Valid TEC
    assert validator.validate_tec(20.0, 2.0)[0] is True

    # Invalid TEC (negative)
    assert validator.validate_tec(-5.0, 2.0)[0] is False

    # Invalid TEC (too high)
    assert validator.validate_tec(500.0, 2.0)[0] is False

    # Invalid error (larger than value)
    assert validator.validate_tec(10.0, 20.0)[0] is False


def test_ionosonde_validation(validator):
    """Test ionosonde parameter validation"""
    # Valid parameters
    is_valid, errors = validator.validate_ionosonde(8.0, 300.0)
    assert is_valid is True

    # Invalid foF2
    is_valid, errors = validator.validate_ionosonde(50.0, 300.0)
    assert is_valid is False
    assert any('foF2' in e for e in errors)

    # Invalid hmF2
    is_valid, errors = validator.validate_ionosonde(8.0, 700.0)
    assert is_valid is False
    assert any('hmF2' in e for e in errors)


def test_elevation_angle_validation(validator):
    """Test elevation angle validation"""
    # Valid elevations
    assert validator.validate_elevation_angle(30.0)[0] is True
    assert validator.validate_elevation_angle(90.0)[0] is True

    # Too low (multipath risk)
    assert validator.validate_elevation_angle(5.0)[0] is False

    # Invalid
    assert validator.validate_elevation_angle(100.0)[0] is False


def test_geographic_coords_validation(validator):
    """Test geographic coordinate validation"""
    # Valid coordinates
    assert validator.validate_geographic_coords(40.0, -105.0)[0] is True

    # Invalid latitude
    assert validator.validate_geographic_coords(100.0, -105.0)[0] is False

    # Invalid longitude
    assert validator.validate_geographic_coords(40.0, -200.0)[0] is False
