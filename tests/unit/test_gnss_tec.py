"""
Unit tests for GNSS-TEC ingestion components
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ingestion.gnss.ntrip_client import NTRIPClient
from src.ingestion.gnss.rtcm3_parser import RTCM3Parser
from src.ingestion.gnss.tec_calculator import (
    TECCalculator,
    ReceiverPosition,
    GPS_L1_FREQ,
    GPS_L2_FREQ
)
from src.ingestion.gnss.gnss_tec_client import GNSSTECClient


class TestNTRIPClient:
    """Tests for NTRIP client"""

    def test_build_request_headers_no_auth(self):
        """Test HTTP headers without authentication"""
        client = NTRIPClient(
            host="test.host",
            port=2101,
            mountpoint="RTCM3"
        )

        headers = client._build_request_headers()

        assert headers['User-Agent'] == "NTRIP AutoNVIS/1.0"
        assert headers['Ntrip-Version'] == "Ntrip/2.0"
        assert 'Authorization' not in headers

    def test_build_request_headers_with_auth(self):
        """Test HTTP headers with authentication"""
        client = NTRIPClient(
            host="test.host",
            port=2101,
            mountpoint="RTCM3",
            username="testuser",
            password="testpass"
        )

        headers = client._build_request_headers()

        assert 'Authorization' in headers
        assert headers['Authorization'].startswith('Basic ')

    @pytest.mark.asyncio
    async def test_connection_properties(self):
        """Test connection state tracking"""
        client = NTRIPClient(
            host="test.host",
            port=2101,
            mountpoint="RTCM3"
        )

        assert not client.is_connected

        # Don't actually connect in test
        client._connected = True
        assert client.is_connected


class TestRTCM3Parser:
    """Tests for RTCM3 parser"""

    def test_parser_initialization(self):
        """Test parser initialization"""
        parser = RTCM3Parser()

        assert parser.station_info is None
        stats = parser.statistics
        assert stats['messages_parsed'] == 0
        assert stats['parse_errors'] == 0

    def test_find_preamble(self):
        """Test finding RTCM3 preamble in data"""
        parser = RTCM3Parser()

        # Add data with preamble at position 3
        parser._buffer = bytearray([0x00, 0x01, 0x02, 0xD3, 0x04])

        idx = parser._find_preamble()
        assert idx == 3

    def test_find_preamble_not_found(self):
        """Test preamble not found"""
        parser = RTCM3Parser()

        parser._buffer = bytearray([0x00, 0x01, 0x02, 0x03])

        idx = parser._find_preamble()
        assert idx == -1

    def test_compute_crc24q(self):
        """Test CRC24Q calculation"""
        parser = RTCM3Parser()

        # Test with known data (preamble + length)
        data = bytes([0xD3, 0x00, 0x13])

        crc = parser._compute_crc24q(data)

        # CRC should be 24-bit value
        assert 0 <= crc <= 0xFFFFFF

    def test_add_data_empty(self):
        """Test adding empty data"""
        parser = RTCM3Parser()

        messages = parser.add_data(b'')

        assert len(messages) == 0


class TestTECCalculator:
    """Tests for TEC calculator"""

    def test_calculator_initialization(self):
        """Test calculator initialization"""
        receiver_pos = ReceiverPosition(
            latitude=42.5,
            longitude=-71.5,
            altitude=100.0
        )

        calculator = TECCalculator(receiver_position=receiver_pos)

        assert calculator.receiver_position == receiver_pos
        assert calculator.statistics['measurements_processed'] == 0

    def test_calculate_tec_from_pseudorange(self):
        """Test TEC calculation from pseudorange"""
        calculator = TECCalculator()

        # Simulate pseudorange measurements with ionospheric delay
        # L2 should be delayed more than L1 (lower frequency)
        p1 = 20000000.0  # meters
        p2 = 20000100.0  # meters (100m more delay)

        tec = calculator.calculate_tec_from_pseudorange(
            p1, p2, GPS_L1_FREQ, GPS_L2_FREQ
        )

        # TEC should be positive (ionospheric delay)
        assert tec > 0

        # Typical TEC values: 1-100 TECU
        # For 100m delay difference, TEC should be reasonable
        assert 0 < tec < 300  # TECU

    def test_ecef_to_geodetic_conversion(self):
        """Test ECEF to geodetic coordinate conversion"""
        calculator = TECCalculator()

        # Known location: MIT (approximately)
        lat_expected = 42.36  # degrees
        lon_expected = -71.09  # degrees
        alt_expected = 10.0  # meters

        # Convert to ECEF
        x, y, z = calculator.geodetic_to_ecef(
            lat_expected, lon_expected, alt_expected
        )

        # Convert back to geodetic
        lat, lon, alt = calculator.ecef_to_geodetic(x, y, z)

        # Check round-trip conversion accuracy
        assert abs(lat - lat_expected) < 1e-6
        assert abs(lon - lon_expected) < 1e-6
        assert abs(alt - alt_expected) < 0.01

    def test_geodetic_to_ecef_conversion(self):
        """Test geodetic to ECEF coordinate conversion"""
        calculator = TECCalculator()

        # Equator at prime meridian
        x, y, z = calculator.geodetic_to_ecef(0.0, 0.0, 0.0)

        # Should be approximately at Earth's equatorial radius
        assert abs(x - 6378137.0) < 1.0  # WGS84 semi-major axis
        assert abs(y) < 1.0
        assert abs(z) < 1.0

    def test_calculate_azimuth_elevation(self):
        """Test azimuth and elevation calculation"""
        calculator = TECCalculator()

        # Receiver at origin (for simplicity)
        receiver_lat = 0.0
        receiver_lon = 0.0
        receiver_alt = 0.0

        # Satellite directly overhead (very high altitude)
        satellite_lat = 0.0
        satellite_lon = 0.0
        satellite_alt = 20200000.0  # GNSS altitude (meters)

        azimuth, elevation = calculator.calculate_azimuth_elevation(
            receiver_lat, receiver_lon, receiver_alt,
            satellite_lat, satellite_lon, satellite_alt
        )

        # Should be nearly overhead (elevation ~90°)
        assert elevation > 85.0

    def test_validate_measurement_good(self):
        """Test measurement validation with good data"""
        calculator = TECCalculator()

        is_valid, reason = calculator.validate_measurement(
            tec=25.0,  # TECU (typical value)
            elevation=45.0,  # degrees (good elevation)
            snr=30.0  # dB-Hz (good SNR)
        )

        assert is_valid
        assert reason == ""

    def test_validate_measurement_low_elevation(self):
        """Test measurement rejection for low elevation"""
        calculator = TECCalculator()

        is_valid, reason = calculator.validate_measurement(
            tec=25.0,
            elevation=5.0,  # Too low
            snr=30.0
        )

        assert not is_valid
        assert "Elevation too low" in reason

    def test_validate_measurement_negative_tec(self):
        """Test measurement rejection for negative TEC"""
        calculator = TECCalculator()

        is_valid, reason = calculator.validate_measurement(
            tec=-5.0,  # Invalid
            elevation=45.0,
            snr=30.0
        )

        assert not is_valid
        assert "Negative TEC" in reason

    def test_validate_measurement_high_tec(self):
        """Test measurement rejection for unrealistic TEC"""
        calculator = TECCalculator()

        is_valid, reason = calculator.validate_measurement(
            tec=400.0,  # Too high
            elevation=45.0,
            snr=30.0
        )

        assert not is_valid
        assert "TEC too high" in reason

    def test_validate_measurement_low_snr(self):
        """Test measurement rejection for low SNR"""
        calculator = TECCalculator()

        is_valid, reason = calculator.validate_measurement(
            tec=25.0,
            elevation=45.0,
            snr=15.0  # Too low
        )

        assert not is_valid
        assert "SNR too low" in reason

    def test_estimate_tec_error(self):
        """Test TEC error estimation"""
        calculator = TECCalculator()

        # Typical pseudorange errors (meters)
        p1_error = 1.0
        p2_error = 1.0

        tec_error = calculator.estimate_tec_error(
            p1_error, p2_error, GPS_L1_FREQ, GPS_L2_FREQ
        )

        # Error should be positive and reasonable
        assert tec_error > 0
        assert tec_error < 10  # TECU


class TestGNSSTECClient:
    """Tests for GNSS-TEC client"""

    def test_client_initialization(self):
        """Test client initialization"""
        client = GNSSTECClient(
            ntrip_host="test.host",
            ntrip_port=2101,
            ntrip_mountpoint="RTCM3"
        )

        assert client.ntrip_host == "test.host"
        assert client.ntrip_port == 2101
        assert client.ntrip_mountpoint == "RTCM3"

    def test_client_with_receiver_position(self):
        """Test client with static receiver position"""
        client = GNSSTECClient(
            ntrip_host="test.host",
            ntrip_port=2101,
            ntrip_mountpoint="RTCM3",
            receiver_lat=42.5,
            receiver_lon=-71.5,
            receiver_alt=100.0
        )

        assert client.receiver_position is not None
        assert client.receiver_position.latitude == 42.5
        assert client.receiver_position.longitude == -71.5

    def test_update_receiver_position(self):
        """Test updating receiver position from station message"""
        client = GNSSTECClient(
            ntrip_host="test.host",
            ntrip_port=2101,
            ntrip_mountpoint="RTCM3"
        )

        # Simulate station position message (ECEF coordinates)
        message = {
            'type': 'station_position',
            'station_id': 1001,
            'x': 1000000.0,
            'y': 2000000.0,
            'z': 3000000.0
        }

        client.update_receiver_position(message)

        # Should have updated receiver position
        assert client.receiver_position is not None
        assert -90 <= client.receiver_position.latitude <= 90
        assert -180 <= client.receiver_position.longitude <= 180

    def test_statistics(self):
        """Test client statistics tracking"""
        client = GNSSTECClient(
            ntrip_host="test.host",
            ntrip_port=2101,
            ntrip_mountpoint="RTCM3"
        )

        stats = client.statistics

        assert 'rtcm_messages_processed' in stats
        assert 'tec_measurements_published' in stats
        assert stats['rtcm_messages_processed'] == 0


# Integration test (requires network, can be skipped)
@pytest.mark.skip(reason="Requires network connection to NTRIP caster")
@pytest.mark.asyncio
async def test_ntrip_connection_integration():
    """Integration test for NTRIP connection"""
    # This test would connect to a real NTRIP caster
    # Skipped by default to avoid network dependency
    client = NTRIPClient(
        host="www.igs-ip.net",
        port=2101,
        mountpoint="RTCM3"
    )

    connected = await client.connect()

    if connected:
        await client.disconnect()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
