"""
Unit tests for ACE solar wind client
"""

import pytest
from src.ingestion.space_weather.ace_solar_wind import ACESolarWindClient


@pytest.fixture
def ace_client():
    """Create ACE client for testing"""
    return ACESolarWindClient(update_interval=60)


def test_dynamic_pressure_calculation(ace_client):
    """Test solar wind dynamic pressure calculation"""
    # Typical quiet conditions: 5 p/cm³, 400 km/s
    pressure = ace_client.calculate_dynamic_pressure(5, 400)
    assert 0.5 < pressure < 2.0  # ~0.8 nPa expected

    # High-speed stream: 10 p/cm³, 600 km/s
    pressure_fast = ace_client.calculate_dynamic_pressure(10, 600)
    assert pressure_fast > pressure  # Should be higher

    # CME: 50 p/cm³, 800 km/s
    pressure_cme = ace_client.calculate_dynamic_pressure(50, 800)
    assert pressure_cme > 20  # Very high pressure


def test_speed_classification(ace_client):
    """Test solar wind speed classification"""
    assert ace_client.classify_solar_wind_speed(250) == "very_slow"
    assert ace_client.classify_solar_wind_speed(350) == "slow"
    assert ace_client.classify_solar_wind_speed(450) == "moderate"
    assert ace_client.classify_solar_wind_speed(550) == "fast"
    assert ace_client.classify_solar_wind_speed(650) == "very_fast"
    assert ace_client.classify_solar_wind_speed(750) == "extreme"


def test_cme_signature_detection(ace_client):
    """Test CME signature detection"""
    # Typical solar wind (not CME)
    assert ace_client.detect_cme_signature(5, 400, 1e5) is False

    # High density but appropriate temperature (not CME)
    assert ace_client.detect_cme_signature(15, 400, 2e5) is False

    # High density and cool plasma (possible CME)
    assert ace_client.detect_cme_signature(20, 500, 5e4) is True
