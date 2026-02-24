"""
GIRO Ionosonde Network Client

Fetches real-time ionosonde data from the Global Ionospheric Radio Observatory (GIRO)
network for IRI correction and nowcasting-grade HF propagation predictions.

GIRO provides near-real-time ionograms and derived ionospheric parameters (foF2, hmF2, etc.)
from a global network of ionosondes. This client fetches the latest measurements and
makes them available for ray tracing calculations.

Data Sources:
    - GIRO DIDBASE: https://giro.uml.edu/
    - SAO Explorer: Real-time ionogram browser
    - UALR-SAMI3: Assimilative model data

Endpoints:
    - Latest character data: Latest autoscaled parameters per station
    - Ionogram images: Full ionogram spectrograms (not used here)

References:
    - Reinisch & Galkin (2011), "Global Ionospheric Radio Observatory (GIRO)"
    - IONORT paper Section 4: Real-time IRI correction
"""

import asyncio
import aiohttp
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Any, Callable
from dataclasses import dataclass, field
from PyQt6.QtCore import QObject, QThread, pyqtSignal, QTimer

logger = logging.getLogger(__name__)


@dataclass
class GIROStation:
    """GIRO ionosonde station metadata."""
    code: str
    name: str
    lat: float
    lon: float
    country: str = ""
    ursi_code: str = ""  # URSI station code

    def distance_to(self, lat: float, lon: float) -> float:
        """Calculate great-circle distance to a point in km."""
        import math
        R = 6371.0  # Earth radius km

        lat1, lon1 = math.radians(self.lat), math.radians(self.lon)
        lat2, lon2 = math.radians(lat), math.radians(lon)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

        return R * c


@dataclass
class IonosondeMeasurement:
    """Real-time ionosonde measurement from GIRO."""
    station: GIROStation
    timestamp: datetime
    foF2: float  # F2 critical frequency MHz
    hmF2: float  # F2 peak height km
    foF1: Optional[float] = None  # F1 critical frequency MHz
    hmF1: Optional[float] = None  # F1 peak height km
    foE: Optional[float] = None   # E critical frequency MHz
    hmE: Optional[float] = None   # E peak height km
    MUF3000: Optional[float] = None  # Maximum usable frequency for 3000km path
    fmin: Optional[float] = None  # Minimum frequency observed
    confidence: float = 1.0  # Data quality indicator
    source: str = "GIRO"  # Data source identifier

    def age_seconds(self) -> float:
        """Get age of measurement in seconds."""
        return (datetime.now(timezone.utc) - self.timestamp).total_seconds()

    def is_valid(self, max_age_seconds: float = 3600.0) -> bool:
        """Check if measurement is still valid."""
        return self.age_seconds() < max_age_seconds and self.foF2 > 0


# Default GIRO stations (most reliable real-time data)
DEFAULT_GIRO_STATIONS = [
    GIROStation("JR055", "Millstone Hill", 42.619, -71.491, "US", "MHJ45"),
    GIROStation("BC840", "Boulder", 40.015, -105.264, "US", "BC840"),
    GIROStation("WP937", "Wallops Island", 37.930, -75.467, "US", "WP937"),
    GIROStation("EG931", "Eglin AFB", 30.467, -86.517, "US", "EG931"),
    GIROStation("AU930", "Canberra", -35.317, 149.006, "AU", "CB53N"),
    GIROStation("DB049", "Dourbes", 50.099, 4.594, "BE", "DB049"),
    GIROStation("JJ433", "Wakkanai", 45.390, 141.688, "JP", "WK546"),
    GIROStation("RL052", "Rome", 41.900, 12.515, "IT", "RO041"),
    GIROStation("PA836", "Point Arguello", 34.567, -120.633, "US", "PA836"),
    GIROStation("GA762", "Grahamstown", -33.300, 26.533, "ZA", "GR13L"),
    GIROStation("AS00Q", "Ascension Island", -7.950, -14.400, "SH", "AS00Q"),
    GIROStation("SV611", "San Vito", 40.600, 17.800, "IT", "SV611"),
]


class GIRODataWorker(QObject):
    """Worker that fetches ionosonde data from GIRO network."""

    # Signals
    connected = pyqtSignal()
    disconnected = pyqtSignal()
    error = pyqtSignal(str)
    measurement_received = pyqtSignal(object)  # IonosondeMeasurement
    stations_updated = pyqtSignal(list)  # List of station codes with data

    # GIRO API endpoints
    # Note: GIRO has various data access methods. We use the JSON API for character data.
    GIRO_LATEST_URL = "https://giro.uml.edu/didbase/scaled.php"
    GIRO_CHARS_URL = "https://lgdc.uml.edu/common/DIDBGetValues"

    # Fallback: NOAA SWPC ionospheric data
    SWPC_IONO_URL = "https://services.swpc.noaa.gov/products/ionospheric-data.json"

    def __init__(
        self,
        stations: Optional[List[GIROStation]] = None,
        update_interval_ms: int = 300000,  # 5 minutes default
        parent=None
    ):
        super().__init__(parent)
        self.stations = {s.code: s for s in (stations or DEFAULT_GIRO_STATIONS)}
        self.update_interval_ms = update_interval_ms
        self.running = False
        self.timer: Optional[QTimer] = None
        self.latest_measurements: Dict[str, IonosondeMeasurement] = {}

    def start_fetching(self):
        """Start periodic data fetching."""
        self.running = True
        logger.info(f"Starting GIRO data fetch for {len(self.stations)} stations...")

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._fetch_data)
        self.timer.start(self.update_interval_ms)

        self.connected.emit()
        # Initial fetch
        self._fetch_data()

    def stop_fetching(self):
        """Stop fetching."""
        self.running = False
        if self.timer:
            self.timer.stop()
        self.disconnected.emit()

    def _fetch_data(self):
        """Fetch data from all sources."""
        if not self.running:
            return

        loop = None
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._async_fetch_all())
        except asyncio.TimeoutError:
            logger.warning("GIRO fetch timeout")
        except Exception as e:
            logger.error(f"GIRO fetch error: {e}")
            self.error.emit(str(e))
        finally:
            if loop:
                try:
                    loop.close()
                except Exception:
                    pass

    async def _async_fetch_all(self):
        """Fetch from multiple sources concurrently."""
        async with aiohttp.ClientSession() as session:
            tasks = [
                self._fetch_swpc_ionospheric(session),
                self._fetch_giro_latest(session),
            ]
            await asyncio.gather(*tasks, return_exceptions=True)

        # Emit list of stations with valid data
        valid_stations = [
            code for code, m in self.latest_measurements.items()
            if m.is_valid(max_age_seconds=7200)  # 2 hour validity
        ]
        if valid_stations:
            self.stations_updated.emit(valid_stations)

    async def _fetch_swpc_ionospheric(self, session: aiohttp.ClientSession):
        """Fetch ionospheric data from NOAA SWPC (backup source)."""
        try:
            async with session.get(self.SWPC_IONO_URL, timeout=30) as resp:
                if resp.status != 200:
                    return
                data = await resp.json()

            if not data:
                return

            # SWPC format: list of records with station data
            # Parse and emit measurements
            for record in data[-20:]:  # Last 20 records
                try:
                    measurement = self._parse_swpc_record(record)
                    if measurement:
                        self.latest_measurements[measurement.station.code] = measurement
                        self.measurement_received.emit(measurement)
                except Exception as e:
                    logger.debug(f"Error parsing SWPC record: {e}")

        except Exception as e:
            logger.debug(f"SWPC ionospheric fetch error: {e}")

    def _parse_swpc_record(self, record: Dict) -> Optional[IonosondeMeasurement]:
        """Parse SWPC ionospheric data record."""
        # SWPC format varies - adapt as needed
        station_code = record.get('station', record.get('Observatory', ''))
        if not station_code:
            return None

        # Find matching station or create placeholder
        if station_code in self.stations:
            station = self.stations[station_code]
        else:
            # Create placeholder station
            station = GIROStation(
                code=station_code,
                name=station_code,
                lat=record.get('latitude', 40.0),
                lon=record.get('longitude', -75.0)
            )
            self.stations[station_code] = station

        # Parse timestamp
        ts_str = record.get('time_tag', record.get('Timestamp', ''))
        try:
            if ts_str:
                timestamp = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
            else:
                timestamp = datetime.now(timezone.utc)
        except ValueError:
            timestamp = datetime.now(timezone.utc)

        # Extract parameters
        foF2 = float(record.get('foF2', record.get('fof2', 0)))
        hmF2 = float(record.get('hmF2', record.get('hmf2', 300)))

        if foF2 <= 0:
            return None

        return IonosondeMeasurement(
            station=station,
            timestamp=timestamp,
            foF2=foF2,
            hmF2=hmF2,
            foF1=record.get('foF1'),
            foE=record.get('foE'),
            MUF3000=record.get('MUF3000'),
            source="SWPC"
        )

    async def _fetch_giro_latest(self, session: aiohttp.ClientSession):
        """Fetch latest ionosonde data from GIRO.

        Note: The actual GIRO API requires specific formatting.
        This is a simplified implementation that may need adjustment
        based on GIRO's current API structure.
        """
        try:
            # Try the GIRO scaled data endpoint
            # Format: station code, parameter, time range
            for station_code, station in list(self.stations.items())[:6]:  # Top 6 stations
                try:
                    params = {
                        'ursiCode': station.ursi_code or station_code,
                        'charName': 'foF2,hmF2,foE,MUF(3000)F2',
                        'fromDate': (datetime.utcnow() - timedelta(hours=2)).strftime('%Y-%m-%d %H:%M'),
                        'toDate': datetime.utcnow().strftime('%Y-%m-%d %H:%M'),
                        'output': 'json'
                    }

                    # Note: Actual GIRO endpoint may differ
                    # This is a placeholder for the real implementation
                    url = f"{self.GIRO_CHARS_URL}?ursiCode={params['ursiCode']}"

                    async with session.get(url, timeout=15) as resp:
                        if resp.status == 200:
                            data = await resp.text()
                            measurement = self._parse_giro_response(station, data)
                            if measurement:
                                self.latest_measurements[station_code] = measurement
                                self.measurement_received.emit(measurement)
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.debug(f"GIRO fetch for {station_code}: {e}")
                    continue

        except Exception as e:
            logger.debug(f"GIRO latest fetch error: {e}")

    def _parse_giro_response(self, station: GIROStation, data: str) -> Optional[IonosondeMeasurement]:
        """Parse GIRO API response."""
        try:
            # GIRO responses are typically CSV or JSON
            # This needs to be adapted to actual GIRO format
            if not data or len(data) < 10:
                return None

            # Try JSON parsing
            try:
                records = json.loads(data)
                if isinstance(records, list) and records:
                    latest = records[-1]
                    return IonosondeMeasurement(
                        station=station,
                        timestamp=datetime.now(timezone.utc),
                        foF2=float(latest.get('foF2', 0)),
                        hmF2=float(latest.get('hmF2', 300)),
                        foE=latest.get('foE'),
                        MUF3000=latest.get('MUF(3000)F2'),
                        source="GIRO"
                    )
            except json.JSONDecodeError:
                pass

            # Try CSV parsing (common GIRO format)
            lines = data.strip().split('\n')
            if len(lines) >= 2:
                # Header line followed by data
                headers = lines[0].split(',')
                values = lines[-1].split(',')  # Latest record

                record = dict(zip(headers, values))
                foF2 = float(record.get('foF2', record.get('fof2', 0)))
                hmF2 = float(record.get('hmF2', record.get('hmf2', 300)))

                if foF2 > 0:
                    return IonosondeMeasurement(
                        station=station,
                        timestamp=datetime.now(timezone.utc),
                        foF2=foF2,
                        hmF2=hmF2,
                        foE=float(record.get('foE', 0)) if record.get('foE') else None,
                        MUF3000=float(record.get('MUF(3000)F2', 0)) if record.get('MUF(3000)F2') else None,
                        source="GIRO"
                    )

        except Exception as e:
            logger.debug(f"Error parsing GIRO response: {e}")

        return None

    def get_nearest_measurement(
        self,
        lat: float,
        lon: float,
        max_distance_km: float = 2000.0
    ) -> Optional[IonosondeMeasurement]:
        """Get the nearest valid measurement to a location."""
        nearest = None
        nearest_dist = float('inf')

        for measurement in self.latest_measurements.values():
            if not measurement.is_valid():
                continue

            dist = measurement.station.distance_to(lat, lon)
            if dist < nearest_dist and dist <= max_distance_km:
                nearest = measurement
                nearest_dist = dist

        return nearest

    def get_interpolated_parameters(
        self,
        lat: float,
        lon: float,
        max_distance_km: float = 2000.0
    ) -> Optional[Dict[str, float]]:
        """Get distance-weighted interpolated ionospheric parameters."""
        valid_measurements = [
            (m, m.station.distance_to(lat, lon))
            for m in self.latest_measurements.values()
            if m.is_valid() and m.station.distance_to(lat, lon) <= max_distance_km
        ]

        if not valid_measurements:
            return None

        # Single station case
        if len(valid_measurements) == 1:
            m = valid_measurements[0][0]
            return {
                'foF2': m.foF2,
                'hmF2': m.hmF2,
                'foE': m.foE,
                'MUF3000': m.MUF3000,
                'distance_km': valid_measurements[0][1],
                'source': m.station.code
            }

        # Distance-weighted interpolation
        total_weight = 0.0
        weighted_foF2 = 0.0
        weighted_hmF2 = 0.0

        decay_km = 500.0  # Weight decay constant

        for m, dist in valid_measurements:
            weight = 1.0 / (1.0 + dist / decay_km)
            total_weight += weight
            weighted_foF2 += weight * m.foF2
            weighted_hmF2 += weight * m.hmF2

        if total_weight > 0:
            return {
                'foF2': weighted_foF2 / total_weight,
                'hmF2': weighted_hmF2 / total_weight,
                'stations_used': len(valid_measurements),
                'source': 'interpolated'
            }

        return None


class GIROClient(QObject):
    """High-level GIRO client with threaded data fetching."""

    # Forward signals from worker
    connected = pyqtSignal()
    disconnected = pyqtSignal()
    error = pyqtSignal(str)
    measurement_received = pyqtSignal(object)
    stations_updated = pyqtSignal(list)

    def __init__(
        self,
        stations: Optional[List[GIROStation]] = None,
        update_interval_ms: int = 300000,
        parent=None
    ):
        super().__init__(parent)
        self.stations = stations
        self.update_interval_ms = update_interval_ms
        self.thread: Optional[QThread] = None
        self.worker: Optional[GIRODataWorker] = None

    def start(self):
        """Start the client in a background thread."""
        if self.thread is not None and self.thread.isRunning():
            return

        self.thread = QThread()
        self.worker = GIRODataWorker(
            stations=self.stations,
            update_interval_ms=self.update_interval_ms
        )
        self.worker.moveToThread(self.thread)

        # Connect lifecycle
        self.thread.started.connect(self.worker.start_fetching)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        # Forward signals
        self.worker.connected.connect(self.connected.emit)
        self.worker.disconnected.connect(self.disconnected.emit)
        self.worker.error.connect(self.error.emit)
        self.worker.measurement_received.connect(self.measurement_received.emit)
        self.worker.stations_updated.connect(self.stations_updated.emit)

        self.thread.start()
        logger.info("GIRO client started")

    def stop(self):
        """Stop the client."""
        if self.worker:
            self.worker.stop_fetching()
        if self.thread and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait(5000)
        self.thread = None
        self.worker = None

    def isRunning(self) -> bool:
        return self.thread is not None and self.thread.isRunning()

    def get_nearest_measurement(self, lat: float, lon: float) -> Optional[IonosondeMeasurement]:
        """Get nearest valid measurement (thread-safe via worker)."""
        if self.worker:
            return self.worker.get_nearest_measurement(lat, lon)
        return None

    def get_interpolated_parameters(self, lat: float, lon: float) -> Optional[Dict[str, float]]:
        """Get interpolated parameters (thread-safe via worker)."""
        if self.worker:
            return self.worker.get_interpolated_parameters(lat, lon)
        return None


# Simulated data for testing when network is unavailable
def generate_simulated_measurement(
    station: Optional[GIROStation] = None,
    solar_time_hours: Optional[float] = None
) -> IonosondeMeasurement:
    """Generate simulated ionosonde measurement for testing.

    Uses simple diurnal variation model:
        foF2 varies from ~4 MHz (night) to ~10 MHz (day)
        hmF2 varies from ~350 km (night) to ~280 km (day)
    """
    import math

    if station is None:
        station = DEFAULT_GIRO_STATIONS[0]

    if solar_time_hours is None:
        # Calculate local solar time
        utc_now = datetime.utcnow()
        solar_time_hours = (utc_now.hour + station.lon / 15.0) % 24

    # Diurnal model
    # Peak at 14:00 local, minimum at 02:00 local
    phase = 2 * math.pi * (solar_time_hours - 14) / 24
    diurnal_factor = 0.5 * (1 + math.cos(phase))

    # Add some random variation
    import random
    noise = random.gauss(0, 0.1)

    foF2 = 4.0 + 6.0 * diurnal_factor + noise
    foF2 = max(2.0, min(15.0, foF2))  # Clamp to reasonable range

    hmF2 = 350 - 70 * diurnal_factor + random.gauss(0, 10)
    hmF2 = max(200, min(450, hmF2))

    return IonosondeMeasurement(
        station=station,
        timestamp=datetime.now(timezone.utc),
        foF2=foF2,
        hmF2=hmF2,
        foE=2.0 + 1.5 * diurnal_factor if diurnal_factor > 0.3 else None,
        MUF3000=foF2 * 3.0,  # Rough approximation
        source="simulated"
    )


if __name__ == "__main__":
    # Test the GIRO client
    import sys
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import QTimer

    logging.basicConfig(level=logging.INFO)

    app = QApplication(sys.argv)

    print("GIRO Client Test")
    print("=" * 50)

    # Print default stations
    print("\nDefault GIRO Stations:")
    for station in DEFAULT_GIRO_STATIONS:
        print(f"  {station.code}: {station.name} ({station.lat:.2f}, {station.lon:.2f})")

    # Generate simulated measurements
    print("\nSimulated Measurements:")
    for station in DEFAULT_GIRO_STATIONS[:3]:
        m = generate_simulated_measurement(station)
        print(f"  {station.code}: foF2={m.foF2:.2f} MHz, hmF2={m.hmF2:.0f} km")

    # Test interpolation
    print("\nInterpolation Test:")
    worker = GIRODataWorker()

    # Add simulated measurements
    for station in DEFAULT_GIRO_STATIONS[:5]:
        m = generate_simulated_measurement(station)
        worker.latest_measurements[station.code] = m

    # Test nearest measurement
    test_lat, test_lon = 39.0, -77.0  # Washington DC area
    nearest = worker.get_nearest_measurement(test_lat, test_lon)
    if nearest:
        print(f"  Nearest to ({test_lat}, {test_lon}): {nearest.station.name}")
        print(f"    foF2={nearest.foF2:.2f} MHz, hmF2={nearest.hmF2:.0f} km")

    # Test interpolated
    interp = worker.get_interpolated_parameters(test_lat, test_lon)
    if interp:
        print(f"  Interpolated: foF2={interp['foF2']:.2f} MHz, hmF2={interp['hmF2']:.0f} km")

    print("\nDone!")
