"""
Unit tests for GOES X-ray client
"""

import pytest
from src.ingestion.space_weather.goes_xray_client import GOESXRayClient


@pytest.fixture
def goes_client():
    """Create GOES client for testing"""
    return GOESXRayClient(update_interval=60)


def test_flare_classification(goes_client):
    """Test solar flare classification"""
    # A-class
    assert goes_client.classify_flare(5e-9) == ('A', 5)

    # B-class
    assert goes_client.classify_flare(5e-8) == ('B', 5)

    # C-class
    assert goes_client.classify_flare(5e-7) == ('C', 5)

    # M-class
    assert goes_client.classify_flare(5e-6) == ('M', 5)

    # X-class
    assert goes_client.classify_flare(5e-5) == ('X', 5)


def test_flare_formatting(goes_client):
    """Test flare class string formatting"""
    # M2.5 flare
    flux_m25 = 2.5e-6
    assert goes_client.format_flare_class(flux_m25) == 'M2.5'

    # X9.3 flare (2017 Sept 6 event)
    flux_x93 = 9.3e-5
    formatted = goes_client.format_flare_class(flux_x93)
    assert formatted.startswith('X')
    assert '9' in formatted


def test_m1_threshold_detection(goes_client):
    """Test M1+ threshold detection for mode switching"""
    # Below threshold
    assert goes_client.is_m1_or_higher(5e-7) is False  # C5
    assert goes_client.is_m1_or_higher(9.9e-6) is False  # M0.99

    # At/above threshold
    assert goes_client.is_m1_or_higher(1e-5) is True  # M1
    assert goes_client.is_m1_or_higher(5e-5) is True  # X5
