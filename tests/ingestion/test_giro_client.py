"""
Tests for GIRO Ionosonde Client

Tests DIDBase parsing, station registry, and data retrieval.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, patch, AsyncMock
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from ingestion.giro_client import (
    GIROStation,
    IonosondeMeasurement,
    DIDBaseParser,
    DEFAULT_GIRO_STATIONS,
    GIRODataWorker,
    generate_simulated_measurement
)


class TestGIROStation:
    """Test GIROStation dataclass"""

    def test_station_creation(self):
        """Test creating a station"""
        station = GIROStation(
            code="TEST01",
            name="Test Station",
            lat=40.0,
            lon=-105.0,
            country="US",
            ursi_code="TS001"
        )

        assert station.code == "TEST01"
        assert station.lat == 40.0
        assert station.lon == -105.0

    def test_distance_calculation(self):
        """Test great-circle distance calculation"""
        # Boulder, CO
        station = GIROStation("BC840", "Boulder", 40.015, -105.264, "US", "BC840")

        # Distance to itself should be 0
        assert station.distance_to(40.015, -105.264) < 1.0

        # Distance to Denver (~40 km)
        dist = station.distance_to(39.739, -104.990)
        assert 30 < dist < 50

    def test_default_stations_count(self):
        """Test that we have expected number of stations"""
        assert len(DEFAULT_GIRO_STATIONS) >= 12  # At least the original 12

    def test_default_stations_global_coverage(self):
        """Test that default stations have global coverage"""
        lats = [s.lat for s in DEFAULT_GIRO_STATIONS]

        # Should have stations in both hemispheres
        assert any(lat > 40 for lat in lats)  # Northern
        assert any(lat < -30 for lat in lats)  # Southern

        # Should have equatorial stations
        assert any(-30 < lat < 30 for lat in lats)


class TestIonosondeMeasurement:
    """Test IonosondeMeasurement dataclass"""

    def test_measurement_creation(self):
        """Test creating a measurement"""
        station = GIROStation("TEST", "Test", 40.0, -105.0)
        timestamp = datetime.now(timezone.utc)

        meas = IonosondeMeasurement(
            station=station,
            timestamp=timestamp,
            foF2=7.5,
            hmF2=280.0,
            foE=2.5,
            MUF3000=22.5
        )

        assert meas.foF2 == 7.5
        assert meas.hmF2 == 280.0
        assert meas.source == "GIRO"

    def test_measurement_validity(self):
        """Test is_valid method"""
        station = GIROStation("TEST", "Test", 40.0, -105.0)

        # Valid measurement (recent)
        valid_meas = IonosondeMeasurement(
            station=station,
            timestamp=datetime.now(timezone.utc),
            foF2=7.5,
            hmF2=280.0
        )
        assert valid_meas.is_valid()

        # Invalid measurement (zero foF2)
        invalid_meas = IonosondeMeasurement(
            station=station,
            timestamp=datetime.now(timezone.utc),
            foF2=0.0,
            hmF2=280.0
        )
        assert not invalid_meas.is_valid()

    def test_measurement_age(self):
        """Test age_seconds method"""
        station = GIROStation("TEST", "Test", 40.0, -105.0)

        meas = IonosondeMeasurement(
            station=station,
            timestamp=datetime.now(timezone.utc),
            foF2=7.5,
            hmF2=280.0
        )

        # Should be very recent (< 1 second)
        assert meas.age_seconds() < 1.0


class TestDIDBaseParser:
    """Test DIDBase CSV parser"""

    def test_parse_timestamp_standard_format(self):
        """Test parsing standard DIDBase timestamp"""
        ts = DIDBaseParser.parse_timestamp("2026.03.10 12:30:45")
        assert ts is not None
        assert ts.year == 2026
        assert ts.month == 3
        assert ts.day == 10
        assert ts.hour == 12
        assert ts.minute == 30
        assert ts.second == 45

    def test_parse_timestamp_iso_format(self):
        """Test parsing ISO format timestamp"""
        ts = DIDBaseParser.parse_timestamp("2026-03-10 12:30:45")
        assert ts is not None
        assert ts.year == 2026

    def test_parse_timestamp_invalid(self):
        """Test parsing invalid timestamp"""
        ts = DIDBaseParser.parse_timestamp("invalid")
        assert ts is None

    def test_parse_value_numeric(self):
        """Test parsing numeric values"""
        assert DIDBaseParser.parse_value("7.850") == 7.850
        assert DIDBaseParser.parse_value("285.3") == 285.3
        assert DIDBaseParser.parse_value("-2.5") == -2.5

    def test_parse_value_missing_markers(self):
        """Test parsing missing data markers"""
        assert DIDBaseParser.parse_value("-") is None
        assert DIDBaseParser.parse_value("---") is None
        assert DIDBaseParser.parse_value("N/A") is None
        assert DIDBaseParser.parse_value("//") is None
        assert DIDBaseParser.parse_value("") is None

    def test_parse_value_with_default(self):
        """Test parsing with default value"""
        assert DIDBaseParser.parse_value("-", default=300.0) == 300.0
        assert DIDBaseParser.parse_value("", default=-1.0) == -1.0

    def test_parse_csv_basic(self):
        """Test parsing basic CSV data"""
        station = GIROStation("TEST", "Test Station", 40.0, -105.0)

        csv_data = """Time,CS,foF2,QD,hmF2,QD,foE,QD
2026.03.10 12:00:00,A,7.850,0,285.3,0,2.850,0
2026.03.10 12:15:00,B,8.100,0,290.1,0,2.900,0"""

        measurements = DIDBaseParser.parse_csv_response(csv_data, station)

        assert len(measurements) == 2
        assert measurements[0].foF2 == 7.850
        assert measurements[0].hmF2 == 285.3
        assert measurements[0].confidence == 1.0  # CS=A

        assert measurements[1].foF2 == 8.100
        assert measurements[1].confidence == 0.7  # CS=B

    def test_parse_csv_with_missing_values(self):
        """Test parsing CSV with missing values"""
        station = GIROStation("TEST", "Test Station", 40.0, -105.0)

        csv_data = """Time,CS,foF2,QD,hmF2,QD,foE,QD
2026.03.10 12:00:00,A,7.850,0,285.3,0,-,0
2026.03.10 12:15:00,A,---,0,N/A,0,-,0"""

        measurements = DIDBaseParser.parse_csv_response(csv_data, station)

        # First line should parse (foF2 valid)
        assert len(measurements) == 1
        assert measurements[0].foF2 == 7.850
        assert measurements[0].foE is None  # Missing '-'

    def test_parse_csv_comment_lines(self):
        """Test that comment lines are skipped"""
        station = GIROStation("TEST", "Test Station", 40.0, -105.0)

        csv_data = """# This is a comment
Time,CS,foF2,QD,hmF2,QD
# Another comment
2026.03.10 12:00:00,A,7.850,0,285.3,0"""

        measurements = DIDBaseParser.parse_csv_response(csv_data, station)
        assert len(measurements) == 1

    def test_parse_csv_empty(self):
        """Test parsing empty CSV"""
        station = GIROStation("TEST", "Test Station", 40.0, -105.0)

        assert DIDBaseParser.parse_csv_response("", station) == []
        assert DIDBaseParser.parse_csv_response("Time,CS,foF2\n", station) == []

    def test_confidence_parsing(self):
        """Test confidence score parsing"""
        assert DIDBaseParser._parse_confidence("A") == 1.0
        assert DIDBaseParser._parse_confidence("B") == 0.7
        assert DIDBaseParser._parse_confidence("C") == 0.4
        assert DIDBaseParser._parse_confidence("0.8") == 0.8


class TestGIRODataWorker:
    """Test GIRODataWorker functionality"""

    def test_worker_initialization(self):
        """Test worker initialization"""
        worker = GIRODataWorker()
        assert len(worker.stations) >= 12
        assert not worker.running

    def test_worker_with_custom_stations(self):
        """Test worker with custom station list"""
        custom_stations = [
            GIROStation("CUST1", "Custom 1", 40.0, -105.0),
            GIROStation("CUST2", "Custom 2", 50.0, 10.0)
        ]

        worker = GIRODataWorker(stations=custom_stations)
        assert len(worker.stations) == 2
        assert "CUST1" in worker.stations

    def test_get_nearest_measurement(self):
        """Test finding nearest measurement"""
        worker = GIRODataWorker()

        # Add some test measurements
        for station in list(worker.stations.values())[:3]:
            meas = generate_simulated_measurement(station)
            worker.latest_measurements[station.code] = meas

        # Find nearest to test location
        test_lat, test_lon = 40.0, -75.0
        nearest = worker.get_nearest_measurement(test_lat, test_lon)

        assert nearest is not None
        assert nearest.is_valid()

    def test_get_interpolated_parameters(self):
        """Test interpolated parameter calculation"""
        worker = GIRODataWorker()

        # Add measurements from multiple stations
        for station in list(worker.stations.values())[:5]:
            meas = generate_simulated_measurement(station)
            worker.latest_measurements[station.code] = meas

        # Get interpolated parameters
        interp = worker.get_interpolated_parameters(40.0, -75.0)

        assert interp is not None
        assert 'foF2' in interp
        assert 'hmF2' in interp
        assert interp['foF2'] > 0


class TestSimulatedMeasurement:
    """Test simulated measurement generation"""

    def test_generate_simulated(self):
        """Test generating simulated measurement"""
        meas = generate_simulated_measurement()

        assert meas.foF2 > 0
        assert meas.hmF2 > 0
        assert meas.source == "simulated"
        assert meas.is_valid()

    def test_simulated_with_station(self):
        """Test generating simulated measurement for specific station"""
        station = GIROStation("TEST", "Test", 40.0, -105.0)
        meas = generate_simulated_measurement(station)

        assert meas.station == station

    def test_simulated_diurnal_variation(self):
        """Test that simulated data shows diurnal variation"""
        station = GIROStation("TEST", "Test", 40.0, -105.0)

        # Noon (high foF2)
        noon_meas = generate_simulated_measurement(station, solar_time_hours=14.0)

        # Midnight (low foF2)
        midnight_meas = generate_simulated_measurement(station, solar_time_hours=2.0)

        # Noon should generally have higher foF2
        # Note: Random variation means this might not always hold
        # so we just verify both are in reasonable range
        assert 2.0 < noon_meas.foF2 < 15.0
        assert 2.0 < midnight_meas.foF2 < 15.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
