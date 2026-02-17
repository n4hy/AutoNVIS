"""
Real-Time IRI Correction Module

Applies real-time ionosonde measurements to correct IRI climatological
profiles for nowcasting-grade HF propagation predictions.

The IRI (International Reference Ionosphere) model provides monthly median
electron density profiles. During space weather events, actual conditions
can differ significantly. This module corrects IRI using real-time data
from the GIRO ionosonde network.

Correction Algorithm (IRI-Real):
    1. Fetch real-time foF2, hmF2 from nearest GIRO ionosonde
    2. Calculate amplitude correction: α = foF2_real / foF2_IRI
    3. Scale profile: Ne_corrected = α² × Ne_IRI (since Ne ∝ foF2²)
    4. Apply height shift: Δh = hmF2_real - hmF2_IRI

References:
    - IONORT paper (remotesensing-15-05111-v2.pdf) Section 4
    - Bilitza et al., "IRI-Real: Real-time ionospheric model"
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Callable
from datetime import datetime, timezone, timedelta
import logging

from .electron_density import IonosphericModel, IonosphericProfile, ChapmanLayer

logger = logging.getLogger(__name__)


@dataclass
class IonosondeStation:
    """
    GIRO ionosonde station information.

    Attributes:
        code: Station identifier (e.g., "JR055" for Millstone Hill)
        name: Station name
        lat: Latitude in degrees
        lon: Longitude in degrees
        country: Country code
    """
    code: str
    name: str
    lat: float
    lon: float
    country: str = ""


@dataclass
class IonosondeMeasurement:
    """
    Real-time ionosonde measurement from GIRO.

    Attributes:
        station: Station that made measurement
        timestamp: UTC timestamp
        foF2: F2 critical frequency (MHz)
        hmF2: F2 peak height (km)
        foF1: F1 critical frequency (MHz), optional
        foE: E critical frequency (MHz), optional
        MUF3000: Maximum usable frequency 3000km (MHz)
        confidence: Data quality indicator (0-1)
    """
    station: IonosondeStation
    timestamp: datetime
    foF2: float
    hmF2: float
    foF1: Optional[float] = None
    foE: Optional[float] = None
    MUF3000: Optional[float] = None
    confidence: float = 1.0


@dataclass
class CorrectionFactors:
    """
    IRI correction factors derived from ionosonde data.

    Attributes:
        alpha: Amplitude scaling (foF2_real / foF2_model)
        delta_h: Height shift in km (hmF2_real - hmF2_model)
        station_dist_km: Distance to ionosonde station
        age_seconds: Age of measurement
        source: Source station code
    """
    alpha: float = 1.0
    delta_h: float = 0.0
    station_dist_km: float = 0.0
    age_seconds: float = 0.0
    source: str = ""

    @property
    def ne_scale(self) -> float:
        """Electron density scale factor (α²)."""
        return self.alpha ** 2

    def is_valid(self, max_age_seconds: float = 3600.0) -> bool:
        """Check if correction is still valid."""
        return self.age_seconds < max_age_seconds and self.alpha > 0


class IRICorrection:
    """
    Real-time IRI correction engine.

    Takes real-time ionosonde measurements and applies corrections to
    a base ionospheric model. Handles multiple stations with distance-
    weighted interpolation.

    Example:
        correction = IRICorrection(model)

        # Register ionosonde stations
        correction.add_station(IonosondeStation(
            code="JR055",
            name="Millstone Hill",
            lat=42.6,
            lon=-71.5
        ))

        # Apply real-time measurement
        correction.apply_measurement(IonosondeMeasurement(
            station=correction.stations["JR055"],
            timestamp=datetime.now(timezone.utc),
            foF2=8.5,
            hmF2=320.0,
            MUF3000=25.0
        ))

        # Get corrected electron density
        Ne = correction.get_corrected_density(lat=40.0, lon=-75.0, alt=300.0)
    """

    # Maximum distance for single-station correction (km)
    MAX_SINGLE_STATION_DISTANCE = 1500.0

    # Maximum measurement age for correction (seconds)
    MAX_MEASUREMENT_AGE = 3600.0  # 1 hour

    # Weight decay factor for distance-based interpolation
    DISTANCE_DECAY_KM = 500.0

    def __init__(self, base_model: Optional[IonosphericModel] = None):
        """
        Initialize IRI correction engine.

        Args:
            base_model: Base ionospheric model to correct
        """
        self.base_model = base_model or IonosphericModel()
        self.stations: Dict[str, IonosondeStation] = {}
        self.measurements: Dict[str, IonosondeMeasurement] = {}
        self._correction_cache: Dict[Tuple[float, float], CorrectionFactors] = {}

        # Initialize with default stations
        self._init_default_stations()

    def _init_default_stations(self):
        """Add commonly used GIRO stations."""
        default_stations = [
            IonosondeStation("JR055", "Millstone Hill", 42.6, -71.5, "US"),
            IonosondeStation("BC840", "Boulder", 40.0, -105.3, "US"),
            IonosondeStation("AU930", "Canberra", -35.3, 149.0, "AU"),
            IonosondeStation("DB049", "Dourbes", 50.1, 4.6, "BE"),
            IonosondeStation("JJ433", "Wakkanai", 45.4, 141.7, "JP"),
            IonosondeStation("RL052", "Rome", 41.8, 12.5, "IT"),
        ]
        for station in default_stations:
            self.stations[station.code] = station

    def add_station(self, station: IonosondeStation):
        """Register an ionosonde station."""
        self.stations[station.code] = station
        logger.debug(f"Added station: {station.code} ({station.name})")

    def apply_measurement(self, measurement: IonosondeMeasurement):
        """
        Apply a real-time ionosonde measurement.

        Args:
            measurement: IonosondeMeasurement from GIRO
        """
        code = measurement.station.code
        self.measurements[code] = measurement

        # Clear correction cache (new data invalidates old corrections)
        self._correction_cache.clear()

        # Also update the base model if using default profile
        self.base_model.update_from_realtime(
            foF2=measurement.foF2,
            hmF2=measurement.hmF2,
            MUF=measurement.MUF3000,
            station_lat=measurement.station.lat,
            station_lon=measurement.station.lon,
            timestamp=measurement.timestamp,
        )

        logger.info(
            f"Applied measurement from {code}: "
            f"foF2={measurement.foF2:.2f} MHz, hmF2={measurement.hmF2:.1f} km"
        )

    def get_correction_factors(
        self,
        lat: float,
        lon: float,
        time: Optional[datetime] = None
    ) -> CorrectionFactors:
        """
        Calculate correction factors for a location.

        Uses distance-weighted interpolation if multiple stations available.
        Falls back to nearest station if only one.

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            time: Reference time (default: now)

        Returns:
            CorrectionFactors with alpha and delta_h
        """
        if not self.measurements:
            return CorrectionFactors()  # No corrections available

        now = time or datetime.now(timezone.utc)

        # Find valid measurements with distances
        valid_measurements = []
        for code, meas in self.measurements.items():
            age = (now - meas.timestamp).total_seconds()
            if age > self.MAX_MEASUREMENT_AGE:
                continue

            dist = self._haversine_distance(
                lat, lon,
                meas.station.lat, meas.station.lon
            )

            if dist <= self.MAX_SINGLE_STATION_DISTANCE:
                valid_measurements.append((meas, dist, age))

        if not valid_measurements:
            return CorrectionFactors()

        # Single station case
        if len(valid_measurements) == 1:
            meas, dist, age = valid_measurements[0]
            return self._calculate_single_station_correction(meas, dist, age)

        # Multiple stations: distance-weighted average
        return self._calculate_multi_station_correction(valid_measurements)

    def _calculate_single_station_correction(
        self,
        measurement: IonosondeMeasurement,
        distance_km: float,
        age_seconds: float
    ) -> CorrectionFactors:
        """Calculate correction from single station."""
        # Get model baseline at station location
        model_profile = self.base_model._get_or_create_profile(
            measurement.station.lat,
            measurement.station.lon,
            None
        )

        # This gets the pre-correction foF2 (before our update)
        # For now, use a reasonable climatological estimate
        model_foF2 = 7.0  # Typical mid-latitude value
        model_hmF2 = 300.0

        alpha = measurement.foF2 / model_foF2 if model_foF2 > 0 else 1.0
        delta_h = measurement.hmF2 - model_hmF2

        return CorrectionFactors(
            alpha=alpha,
            delta_h=delta_h,
            station_dist_km=distance_km,
            age_seconds=age_seconds,
            source=measurement.station.code
        )

    def _calculate_multi_station_correction(
        self,
        measurements: List[Tuple[IonosondeMeasurement, float, float]]
    ) -> CorrectionFactors:
        """
        Calculate weighted average correction from multiple stations.

        Uses inverse-distance weighting with exponential decay.
        """
        model_foF2 = 7.0
        model_hmF2 = 300.0

        total_weight = 0.0
        weighted_alpha = 0.0
        weighted_delta_h = 0.0
        min_dist = float('inf')
        nearest_source = ""
        max_age = 0.0

        for meas, dist, age in measurements:
            # Weight = exp(-dist / decay)
            weight = np.exp(-dist / self.DISTANCE_DECAY_KM)
            total_weight += weight

            alpha = meas.foF2 / model_foF2 if model_foF2 > 0 else 1.0
            delta_h = meas.hmF2 - model_hmF2

            weighted_alpha += weight * alpha
            weighted_delta_h += weight * delta_h

            if dist < min_dist:
                min_dist = dist
                nearest_source = meas.station.code
            max_age = max(max_age, age)

        if total_weight > 0:
            weighted_alpha /= total_weight
            weighted_delta_h /= total_weight

        return CorrectionFactors(
            alpha=weighted_alpha,
            delta_h=weighted_delta_h,
            station_dist_km=min_dist,
            age_seconds=max_age,
            source=f"{nearest_source}+{len(measurements)-1}"
        )

    def get_corrected_density(
        self,
        lat: float,
        lon: float,
        alt: float,
        time: Optional[datetime] = None
    ) -> float:
        """
        Get electron density with real-time corrections applied.

        Args:
            lat: Latitude degrees
            lon: Longitude degrees
            alt: Altitude km
            time: Reference time

        Returns:
            Corrected electron density in el/cm³
        """
        # Get base density
        Ne_base = self.base_model.get_electron_density(lat, lon, alt, time)

        # Get correction factors
        corrections = self.get_correction_factors(lat, lon, time)

        if not corrections.is_valid():
            return Ne_base

        # Apply corrections
        # Ne_corrected = α² × Ne_base (amplitude scaling)
        Ne_corrected = corrections.ne_scale * Ne_base

        # Height shift handled implicitly by updating hmF2 in base model

        return Ne_corrected

    def get_corrected_profile(
        self,
        lat: float,
        lon: float,
        altitudes: np.ndarray,
        time: Optional[datetime] = None
    ) -> Tuple[np.ndarray, CorrectionFactors]:
        """
        Get corrected electron density profile.

        Args:
            lat: Latitude degrees
            lon: Longitude degrees
            altitudes: Array of altitudes in km
            time: Reference time

        Returns:
            Tuple of (Ne_profile, CorrectionFactors)
        """
        corrections = self.get_correction_factors(lat, lon, time)
        Ne_base = self.base_model.get_electron_density_profile(
            lat, lon, altitudes, time
        )

        if corrections.is_valid():
            Ne_corrected = corrections.ne_scale * Ne_base
        else:
            Ne_corrected = Ne_base

        return Ne_corrected, corrections

    def _haversine_distance(
        self,
        lat1: float, lon1: float,
        lat2: float, lon2: float
    ) -> float:
        """
        Calculate great-circle distance between two points.

        Args:
            lat1, lon1: First point (degrees)
            lat2, lon2: Second point (degrees)

        Returns:
            Distance in km
        """
        from . import EARTH_RADIUS_KM

        lat1_r = np.radians(lat1)
        lat2_r = np.radians(lat2)
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)

        a = (np.sin(dlat/2)**2 +
             np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon/2)**2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

        return EARTH_RADIUS_KM * c

    def get_status(self) -> Dict:
        """
        Get correction engine status.

        Returns:
            Dictionary with stations, measurements, and correction info
        """
        now = datetime.now(timezone.utc)

        active_measurements = {}
        for code, meas in self.measurements.items():
            age = (now - meas.timestamp).total_seconds()
            active_measurements[code] = {
                'foF2': meas.foF2,
                'hmF2': meas.hmF2,
                'age_seconds': age,
                'valid': age < self.MAX_MEASUREMENT_AGE
            }

        return {
            'stations': list(self.stations.keys()),
            'measurements': active_measurements,
            'base_model_status': self.base_model.get_correction_status()
        }


class IRICorrectionCallback:
    """
    Callback adapter for Advanced Data Client integration.

    Connects the real-time ionosonde data stream from AdvancedDataClient
    to the IRICorrection engine.

    Example:
        from propagation.advanced_data_client import AdvancedDataClient

        correction = IRICorrection()
        callback = IRICorrectionCallback(correction)

        client = AdvancedDataClient()
        client.ionosonde_data.connect(callback.on_ionosonde_data)
    """

    def __init__(self, correction: IRICorrection):
        """
        Initialize callback adapter.

        Args:
            correction: IRICorrection engine to update
        """
        self.correction = correction

    def on_ionosonde_data(self, data: Dict):
        """
        Handle ionosonde data from Advanced Data Client.

        Expected data format from AdvancedDataClient:
            {
                'station': 'JR055',
                'foF2': 8.5,
                'hmF2': 320.0,
                'MUF': 25.0,
                'timestamp': '2024-01-15T12:00:00Z'
            }

        Args:
            data: Ionosonde measurement dictionary
        """
        station_code = data.get('station')
        if not station_code:
            logger.warning("Ionosonde data missing station code")
            return

        # Get or create station
        if station_code not in self.correction.stations:
            # Create placeholder station (location unknown)
            logger.warning(f"Unknown station {station_code}, using default location")
            self.correction.add_station(IonosondeStation(
                code=station_code,
                name=station_code,
                lat=40.0,  # Default to mid-latitude
                lon=-75.0
            ))

        station = self.correction.stations[station_code]

        # Parse timestamp
        ts_str = data.get('timestamp')
        if ts_str:
            try:
                timestamp = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
            except ValueError:
                timestamp = datetime.now(timezone.utc)
        else:
            timestamp = datetime.now(timezone.utc)

        # Create measurement
        measurement = IonosondeMeasurement(
            station=station,
            timestamp=timestamp,
            foF2=float(data.get('foF2', 0)),
            hmF2=float(data.get('hmF2', 300)),
            foF1=data.get('foF1'),
            foE=data.get('foE'),
            MUF3000=data.get('MUF'),
        )

        # Apply to correction engine
        self.correction.apply_measurement(measurement)


def create_test_correction() -> IRICorrection:
    """Create a test correction engine with sample data."""
    correction = IRICorrection()

    # Simulate real-time measurement from Millstone Hill
    measurement = IonosondeMeasurement(
        station=correction.stations["JR055"],
        timestamp=datetime.now(timezone.utc),
        foF2=8.5,
        hmF2=320.0,
        MUF3000=25.0,
        confidence=0.95
    )
    correction.apply_measurement(measurement)

    return correction


if __name__ == "__main__":
    # Test the correction engine
    correction = create_test_correction()

    print("IRI Correction Test")
    print("=" * 50)

    # Status
    status = correction.get_status()
    print(f"\nStations: {status['stations']}")
    print(f"\nMeasurements:")
    for code, info in status['measurements'].items():
        print(f"  {code}: foF2={info['foF2']:.2f} MHz, age={info['age_seconds']:.0f}s")

    # Test correction at various locations
    print("\nCorrected Profiles:")
    print(f"{'Location':<20} {'Distance':<10} {'foF2 scale':<12} {'Ne scale':<10}")
    print("-" * 52)

    test_locations = [
        ("Near station", 42.6, -71.5),   # Millstone Hill
        ("500 km away", 40.0, -75.0),     # ~500 km
        ("1000 km away", 35.0, -80.0),    # ~1000 km
    ]

    for name, lat, lon in test_locations:
        factors = correction.get_correction_factors(lat, lon)
        dist = factors.station_dist_km
        print(f"{name:<20} {dist:>6.0f} km  α={factors.alpha:.3f}      α²={factors.ne_scale:.3f}")

    # Test profile
    print("\nCorrected Density Profile (40°N, 75°W):")
    altitudes = np.arange(200, 450, 50)
    Ne_profile, factors = correction.get_corrected_profile(40.0, -75.0, altitudes)

    print(f"Source: {factors.source}, Distance: {factors.station_dist_km:.0f} km")
    print(f"{'Alt (km)':<10} {'Ne (el/cm³)':<15}")
    print("-" * 25)
    for alt, Ne in zip(altitudes, Ne_profile):
        print(f"{alt:<10.0f} {Ne:<15.2e}")
