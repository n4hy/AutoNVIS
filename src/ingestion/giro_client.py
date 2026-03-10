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


# Default GIRO stations - Global network with reliable real-time data
# Expanded from 12 to 27 stations for better global coverage
DEFAULT_GIRO_STATIONS = [
    # North America
    GIROStation("JR055", "Millstone Hill", 42.619, -71.491, "US", "MHJ45"),
    GIROStation("BC840", "Boulder", 40.015, -105.264, "US", "BC840"),
    GIROStation("WP937", "Wallops Island", 37.930, -75.467, "US", "WP937"),
    GIROStation("EG931", "Eglin AFB", 30.467, -86.517, "US", "EG931"),
    GIROStation("PA836", "Point Arguello", 34.567, -120.633, "US", "PA836"),
    GIROStation("PRJ18", "Puerto Rico", 18.494, -67.139, "PR", "PRJ18"),
    GIROStation("THJ76", "Thule", 77.467, -69.217, "GL", "THJ76"),

    # Europe
    GIROStation("DB049", "Dourbes", 50.099, 4.594, "BE", "DB049"),
    GIROStation("RL052", "Rome", 41.900, 12.515, "IT", "RO041"),
    GIROStation("SV611", "San Vito", 40.600, 17.800, "IT", "SV611"),
    GIROStation("JU434", "Juliusruh", 54.633, 13.383, "DE", "JR055"),
    GIROStation("AT138", "Athens", 38.050, 23.867, "GR", "AT138"),
    GIROStation("EB040", "Ebro", 40.957, 0.492, "ES", "EB040"),
    GIROStation("SO148", "Sodankyla", 67.367, 26.633, "FI", "SO148"),
    GIROStation("TR170", "Tromso", 69.667, 18.950, "NO", "TR170"),

    # Asia-Pacific
    GIROStation("JJ433", "Wakkanai", 45.390, 141.688, "JP", "WK546"),
    GIROStation("TO535", "Tokyo/Kokubunji", 35.710, 139.488, "JP", "TO535"),
    GIROStation("OKJ24", "Okinawa", 26.333, 127.800, "JP", "OKJ24"),
    GIROStation("AU930", "Canberra", -35.317, 149.006, "AU", "CB53N"),
    GIROStation("HO54K", "Hobart", -42.883, 147.317, "AU", "HO54K"),
    GIROStation("DA122", "Darwin", -12.467, 130.867, "AU", "DA122"),

    # Africa & Atlantic
    GIROStation("GA762", "Grahamstown", -33.300, 26.533, "ZA", "GR13L"),
    GIROStation("AS00Q", "Ascension Island", -7.950, -14.400, "SH", "AS00Q"),
    GIROStation("SB49P", "Sao Luis", -2.600, -44.233, "BR", "SB49P"),

    # South America
    GIROStation("JI91J", "Jicamarca", -11.950, -76.867, "PE", "JI91J"),
    GIROStation("PA839", "Port Stanley", -51.700, -57.867, "FK", "PA839"),

    # Arctic/Antarctic
    GIROStation("SY88K", "Syowa", -69.000, 39.583, "AQ", "SY88K"),
]


class DIDBaseParser:
    """
    Parser for DIDBase CSV response format from GIRO.

    DIDBase (Digital Ionogram Database) provides ionospheric parameters
    in CSV format. This parser handles the specific format and data
    quality markers used by the GIRO network.

    CSV Format:
        Time,CS,foF2,QD,hmF2,QD,foF1,QD,hmF1,QD,foE,QD,hmE,QD,MUF(3000)F2,QD,fmin,QD
        2026.03.10 12:00:00,A,7.850,0,285.3,0,4.125,0,195.2,0,2.850,0,105.5,0,22.5,0,1.850,0

    Where:
        - Time: Timestamp in "YYYY.MM.DD HH:MM:SS" format
        - CS: Confidence score (A/B/C or numeric)
        - QD: Quality descriptor (0=good, 1=fair, 2=poor, -1=missing)
        - Values: Ionospheric parameters in standard units

    Missing data markers: '-', '---', 'N/A', '//', ''
    """

    MISSING_MARKERS = {'-', '---', 'N/A', '//', '', 'nan', 'NaN'}

    # Expected columns in DIDBase CSV output
    EXPECTED_COLUMNS = [
        'Time', 'CS',
        'foF2', 'QD_foF2',
        'hmF2', 'QD_hmF2',
        'foF1', 'QD_foF1',
        'hmF1', 'QD_hmF1',
        'foE', 'QD_foE',
        'hmE', 'QD_hmE',
        'MUF(3000)F2', 'QD_MUF',
        'fmin', 'QD_fmin'
    ]

    @staticmethod
    def parse_timestamp(ts_str: str) -> Optional[datetime]:
        """
        Parse DIDBase timestamp format.

        Args:
            ts_str: Timestamp string in "YYYY.MM.DD HH:MM:SS" format

        Returns:
            datetime object or None if parsing fails
        """
        formats = [
            "%Y.%m.%d %H:%M:%S",  # Primary format
            "%Y-%m-%d %H:%M:%S",  # ISO format
            "%Y/%m/%d %H:%M:%S",  # Alternate format
            "%Y.%m.%d %H:%M",     # Without seconds
            "%Y-%m-%dT%H:%M:%SZ", # ISO with T separator
        ]

        for fmt in formats:
            try:
                return datetime.strptime(ts_str.strip(), fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue

        return None

    @staticmethod
    def parse_value(value_str: str, default: Optional[float] = None) -> Optional[float]:
        """
        Parse a numeric value, handling missing data markers.

        Args:
            value_str: Value string from CSV
            default: Default value if missing

        Returns:
            Parsed float value or default
        """
        if not value_str or value_str.strip() in DIDBaseParser.MISSING_MARKERS:
            return default

        try:
            return float(value_str.strip())
        except ValueError:
            return default

    @staticmethod
    def parse_csv_response(
        csv_data: str,
        station: GIROStation
    ) -> List[IonosondeMeasurement]:
        """
        Parse DIDBase CSV response into IonosondeMeasurement objects.

        Args:
            csv_data: CSV string from DIDBase API
            station: Station metadata

        Returns:
            List of IonosondeMeasurement objects
        """
        measurements = []

        if not csv_data or not csv_data.strip():
            return measurements

        lines = csv_data.strip().split('\n')
        if len(lines) < 2:
            return measurements

        # Parse header to determine column positions
        header = lines[0].strip()
        if header.startswith('#'):
            header = header[1:].strip()

        columns = [c.strip() for c in header.split(',')]

        # Build column index map
        col_idx = {}
        for i, col in enumerate(columns):
            col_idx[col.lower()] = i

        # Parse data lines
        for line in lines[1:]:
            if not line.strip() or line.startswith('#'):
                continue

            values = line.split(',')
            if len(values) < 3:  # Minimum: time, confidence, foF2
                continue

            try:
                # Parse timestamp
                ts_idx = col_idx.get('time', 0)
                timestamp = DIDBaseParser.parse_timestamp(values[ts_idx])
                if timestamp is None:
                    continue

                # Parse foF2 (required)
                fof2_idx = col_idx.get('fof2', 2)
                foF2 = DIDBaseParser.parse_value(values[fof2_idx] if fof2_idx < len(values) else '')
                if foF2 is None or foF2 <= 0:
                    continue

                # Parse hmF2 (required)
                hmf2_idx = col_idx.get('hmf2', 4)
                hmF2 = DIDBaseParser.parse_value(values[hmf2_idx] if hmf2_idx < len(values) else '', 300.0)

                # Parse optional parameters
                fof1_idx = col_idx.get('fof1', 6)
                foF1 = DIDBaseParser.parse_value(values[fof1_idx] if fof1_idx < len(values) else '')

                hmf1_idx = col_idx.get('hmf1', 8)
                hmF1 = DIDBaseParser.parse_value(values[hmf1_idx] if hmf1_idx < len(values) else '')

                foe_idx = col_idx.get('foe', 10)
                foE = DIDBaseParser.parse_value(values[foe_idx] if foe_idx < len(values) else '')

                muf_idx = col_idx.get('muf(3000)f2', 14)
                MUF3000 = DIDBaseParser.parse_value(values[muf_idx] if muf_idx < len(values) else '')

                fmin_idx = col_idx.get('fmin', 16)
                fmin = DIDBaseParser.parse_value(values[fmin_idx] if fmin_idx < len(values) else '')

                # Parse confidence score
                cs_idx = col_idx.get('cs', 1)
                cs_str = values[cs_idx].strip() if cs_idx < len(values) else 'C'
                confidence = DIDBaseParser._parse_confidence(cs_str)

                measurement = IonosondeMeasurement(
                    station=station,
                    timestamp=timestamp,
                    foF2=foF2,
                    hmF2=hmF2,
                    foF1=foF1,
                    hmF1=hmF1,
                    foE=foE,
                    MUF3000=MUF3000,
                    fmin=fmin,
                    confidence=confidence,
                    source="GIRO/DIDBase"
                )
                measurements.append(measurement)

            except Exception as e:
                logger.debug(f"Error parsing line: {line}: {e}")
                continue

        return measurements

    @staticmethod
    def _parse_confidence(cs_str: str) -> float:
        """
        Parse confidence score from DIDBase.

        Args:
            cs_str: Confidence string ('A', 'B', 'C' or numeric)

        Returns:
            Confidence value 0.0-1.0
        """
        cs_str = cs_str.upper().strip()

        if cs_str == 'A':
            return 1.0
        elif cs_str == 'B':
            return 0.7
        elif cs_str == 'C':
            return 0.4
        else:
            try:
                return min(1.0, max(0.0, float(cs_str)))
            except ValueError:
                return 0.5


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
                except RuntimeError:
                    pass  # Loop already closed

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
        """
        Parse GIRO API response using DIDBase parser.

        Args:
            station: Station metadata
            data: Response data (CSV or JSON format)

        Returns:
            Latest IonosondeMeasurement or None
        """
        if not data or len(data) < 10:
            return None

        try:
            # Try JSON parsing first
            try:
                records = json.loads(data)
                if isinstance(records, list) and records:
                    latest = records[-1]
                    foF2 = float(latest.get('foF2', latest.get('fof2', 0)))
                    if foF2 <= 0:
                        return None

                    # Parse timestamp if present
                    ts_str = latest.get('time', latest.get('timestamp', ''))
                    timestamp = DIDBaseParser.parse_timestamp(ts_str) if ts_str else datetime.now(timezone.utc)

                    return IonosondeMeasurement(
                        station=station,
                        timestamp=timestamp or datetime.now(timezone.utc),
                        foF2=foF2,
                        hmF2=float(latest.get('hmF2', latest.get('hmf2', 300))),
                        foF1=DIDBaseParser.parse_value(str(latest.get('foF1', ''))),
                        hmF1=DIDBaseParser.parse_value(str(latest.get('hmF1', ''))),
                        foE=DIDBaseParser.parse_value(str(latest.get('foE', ''))),
                        hmE=DIDBaseParser.parse_value(str(latest.get('hmE', ''))),
                        MUF3000=DIDBaseParser.parse_value(str(latest.get('MUF(3000)F2', latest.get('MUF3000', '')))),
                        fmin=DIDBaseParser.parse_value(str(latest.get('fmin', ''))),
                        source="GIRO/JSON"
                    )
            except json.JSONDecodeError:
                pass

            # Use DIDBase CSV parser
            measurements = DIDBaseParser.parse_csv_response(data, station)
            if measurements:
                # Return most recent measurement
                return max(measurements, key=lambda m: m.timestamp)

            # Fallback: simple CSV parsing for legacy format
            lines = data.strip().split('\n')
            if len(lines) >= 2:
                # Skip comment lines
                data_lines = [l for l in lines if not l.startswith('#')]
                if len(data_lines) >= 2:
                    headers = [h.strip().lower() for h in data_lines[0].split(',')]
                    values = data_lines[-1].split(',')  # Latest record

                    if len(headers) == len(values):
                        record = dict(zip(headers, values))
                        foF2 = DIDBaseParser.parse_value(record.get('fof2', ''))

                        if foF2 and foF2 > 0:
                            return IonosondeMeasurement(
                                station=station,
                                timestamp=datetime.now(timezone.utc),
                                foF2=foF2,
                                hmF2=DIDBaseParser.parse_value(record.get('hmf2', ''), 300.0),
                                foE=DIDBaseParser.parse_value(record.get('foe', '')),
                                MUF3000=DIDBaseParser.parse_value(record.get('muf(3000)f2', '')),
                                fmin=DIDBaseParser.parse_value(record.get('fmin', '')),
                                source="GIRO/CSV"
                            )

        except Exception as e:
            logger.debug(f"Error parsing GIRO response for {station.code}: {e}")

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
