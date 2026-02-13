"""
TEC Calculator from GNSS Observables

Calculates Total Electron Content (TEC) from dual-frequency GNSS measurements.
Uses ionospheric dispersive property to derive slant TEC from L1/L2 observables.

Physics:
    Ionospheric delay is frequency-dependent:
        Δt ∝ TEC / f²

    Using dual-frequency measurements:
        TEC = (f1² × f2²) / (40.3 × (f1² - f2²)) × ΔP

    where ΔP = P2 - P1 (pseudorange difference)
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.common.logging_config import ServiceLogger
from src.common.constants import TECU_TO_ELECTRONS_M2


# GNSS Frequencies (Hz)
GPS_L1_FREQ = 1575.42e6  # Hz
GPS_L2_FREQ = 1227.60e6  # Hz
GLONASS_L1_FREQ = 1602.0e6  # Hz (approximate, frequency-dependent)
GLONASS_L2_FREQ = 1246.0e6  # Hz (approximate, frequency-dependent)

# Physical constants
SPEED_OF_LIGHT = 299792458.0  # m/s
IONOSPHERIC_CONSTANT = 40.3  # m³/s²

# Quality control thresholds
MIN_ELEVATION_ANGLE = 10.0  # degrees (avoid multipath)
MAX_TEC_VALUE = 300.0  # TECU (sanity check)
MIN_SNR = 20.0  # dB-Hz (minimum signal-to-noise ratio)


@dataclass
class TECMeasurement:
    """TEC measurement with metadata"""
    # Receiver location (geodetic)
    receiver_lat: float  # degrees
    receiver_lon: float  # degrees
    receiver_alt: float  # meters

    # Satellite location (geodetic)
    satellite_lat: float  # degrees
    satellite_lon: float  # degrees
    satellite_alt: float  # meters

    # Geometry
    azimuth: float  # degrees
    elevation: float  # degrees

    # TEC value
    slant_tec: float  # TECU
    tec_error: float  # TECU

    # Metadata
    timestamp: datetime
    satellite_id: int
    gnss_type: str  # 'GPS' or 'GLONASS'


@dataclass
class ReceiverPosition:
    """Receiver position in geodetic coordinates"""
    latitude: float  # degrees
    longitude: float  # degrees
    altitude: float  # meters


class TECCalculator:
    """
    Calculate TEC from dual-frequency GNSS observables

    Converts L1/L2 pseudorange and carrier phase measurements into
    slant TEC values with geometry and quality metrics.
    """

    def __init__(self, receiver_position: Optional[ReceiverPosition] = None):
        """
        Initialize TEC calculator

        Args:
            receiver_position: Receiver position (if known statically)
        """
        self.logger = ServiceLogger("ingestion", "tec_calculator")
        self.receiver_position = receiver_position

        # Statistics
        self._measurements_processed = 0
        self._measurements_rejected = 0

    def calculate_tec_from_pseudorange(
        self,
        p1: float,
        p2: float,
        f1: float,
        f2: float
    ) -> float:
        """
        Calculate slant TEC from dual-frequency pseudorange measurements

        Args:
            p1: Pseudorange on frequency 1 (meters)
            p2: Pseudorange on frequency 2 (meters)
            f1: Frequency 1 (Hz)
            f2: Frequency 2 (Hz)

        Returns:
            Slant TEC (TECU)
        """
        # Ionospheric delay difference
        delay_diff = p2 - p1  # meters

        # TEC calculation (geometry-free combination)
        # TEC = (f1² × f2²) / (40.3 × (f1² - f2²)) × ΔP
        frequency_factor = (f1**2 * f2**2) / (IONOSPHERIC_CONSTANT * (f1**2 - f2**2))

        tec_m2 = frequency_factor * delay_diff  # electrons/m²
        tec_tecu = tec_m2 / TECU_TO_ELECTRONS_M2  # TECU

        return tec_tecu

    def calculate_tec_from_phase(
        self,
        l1: float,
        l2: float,
        f1: float,
        f2: float
    ) -> float:
        """
        Calculate slant TEC from dual-frequency carrier phase measurements

        Carrier phase TEC has higher precision but integer ambiguity.
        This computes relative TEC (changes over time).

        Args:
            l1: Carrier phase on frequency 1 (cycles)
            l2: Carrier phase on frequency 2 (cycles)
            f1: Frequency 1 (Hz)
            f2: Frequency 2 (Hz)

        Returns:
            Slant TEC (TECU) - relative value
        """
        # Convert phase to range
        lambda1 = SPEED_OF_LIGHT / f1  # wavelength 1 (meters)
        lambda2 = SPEED_OF_LIGHT / f2  # wavelength 2 (meters)

        range1 = l1 * lambda1
        range2 = l2 * lambda2

        # Calculate TEC (same formula as pseudorange)
        frequency_factor = (f1**2 * f2**2) / (IONOSPHERIC_CONSTANT * (f1**2 - f2**2))

        tec_m2 = frequency_factor * (range2 - range1)
        tec_tecu = tec_m2 / TECU_TO_ELECTRONS_M2

        return tec_tecu

    def ecef_to_geodetic(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """
        Convert ECEF coordinates to geodetic (WGS84)

        Args:
            x, y, z: ECEF coordinates (meters)

        Returns:
            (latitude, longitude, altitude) in degrees and meters
        """
        # WGS84 ellipsoid parameters
        a = 6378137.0  # semi-major axis (meters)
        f = 1.0 / 298.257223563  # flattening
        e2 = 2 * f - f**2  # first eccentricity squared

        # Longitude
        lon = np.arctan2(y, x)

        # Iterative latitude calculation
        p = np.sqrt(x**2 + y**2)
        lat = np.arctan2(z, p * (1 - e2))

        for _ in range(5):  # Usually converges in 3-4 iterations
            N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
            alt = p / np.cos(lat) - N
            lat = np.arctan2(z, p * (1 - e2 * N / (N + alt)))

        # Final altitude
        N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
        alt = p / np.cos(lat) - N

        # Convert to degrees
        lat_deg = np.degrees(lat)
        lon_deg = np.degrees(lon)

        return lat_deg, lon_deg, alt

    def geodetic_to_ecef(self, lat: float, lon: float, alt: float) -> Tuple[float, float, float]:
        """
        Convert geodetic coordinates (WGS84) to ECEF

        Args:
            lat, lon: Latitude and longitude (degrees)
            alt: Altitude above ellipsoid (meters)

        Returns:
            (x, y, z) ECEF coordinates (meters)
        """
        # WGS84 parameters
        a = 6378137.0
        f = 1.0 / 298.257223563
        e2 = 2 * f - f**2

        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)

        N = a / np.sqrt(1 - e2 * np.sin(lat_rad)**2)

        x = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
        y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
        z = (N * (1 - e2) + alt) * np.sin(lat_rad)

        return x, y, z

    def calculate_azimuth_elevation(
        self,
        receiver_lat: float,
        receiver_lon: float,
        receiver_alt: float,
        satellite_lat: float,
        satellite_lon: float,
        satellite_alt: float
    ) -> Tuple[float, float]:
        """
        Calculate azimuth and elevation angles from receiver to satellite

        Args:
            receiver_lat, receiver_lon, receiver_alt: Receiver position (deg, deg, m)
            satellite_lat, satellite_lon, satellite_alt: Satellite position (deg, deg, m)

        Returns:
            (azimuth, elevation) in degrees
        """
        # Convert to ECEF
        rx, ry, rz = self.geodetic_to_ecef(receiver_lat, receiver_lon, receiver_alt)
        sx, sy, sz = self.geodetic_to_ecef(satellite_lat, satellite_lon, satellite_alt)

        # Vector from receiver to satellite (ECEF)
        dx = sx - rx
        dy = sy - ry
        dz = sz - rz

        # Convert to local ENU (East-North-Up) frame
        lat_rad = np.radians(receiver_lat)
        lon_rad = np.radians(receiver_lon)

        # Rotation matrix ECEF -> ENU
        e = -np.sin(lon_rad) * dx + np.cos(lon_rad) * dy
        n = -np.sin(lat_rad) * np.cos(lon_rad) * dx - np.sin(lat_rad) * np.sin(lon_rad) * dy + np.cos(lat_rad) * dz
        u = np.cos(lat_rad) * np.cos(lon_rad) * dx + np.cos(lat_rad) * np.sin(lon_rad) * dy + np.sin(lat_rad) * dz

        # Azimuth (from north, clockwise)
        azimuth = np.degrees(np.arctan2(e, n))
        if azimuth < 0:
            azimuth += 360.0

        # Elevation (from horizon, upward)
        slant_range = np.sqrt(e**2 + n**2 + u**2)
        elevation = np.degrees(np.arcsin(u / slant_range))

        return azimuth, elevation

    def estimate_tec_error(
        self,
        p1_error: float,
        p2_error: float,
        f1: float,
        f2: float
    ) -> float:
        """
        Estimate TEC measurement error from pseudorange errors

        Args:
            p1_error: Pseudorange error on frequency 1 (meters)
            p2_error: Pseudorange error on frequency 2 (meters)
            f1: Frequency 1 (Hz)
            f2: Frequency 2 (Hz)

        Returns:
            TEC error (TECU)
        """
        # Error propagation for geometry-free combination
        frequency_factor = (f1**2 * f2**2) / (IONOSPHERIC_CONSTANT * (f1**2 - f2**2))

        # Assume errors are uncorrelated
        combined_error = np.sqrt(p1_error**2 + p2_error**2)

        tec_error_m2 = frequency_factor * combined_error
        tec_error_tecu = tec_error_m2 / TECU_TO_ELECTRONS_M2

        return tec_error_tecu

    def validate_measurement(
        self,
        tec: float,
        elevation: float,
        snr: Optional[float] = None
    ) -> Tuple[bool, str]:
        """
        Validate TEC measurement quality

        Args:
            tec: TEC value (TECU)
            elevation: Elevation angle (degrees)
            snr: Signal-to-noise ratio (dB-Hz, optional)

        Returns:
            (is_valid, reason)
        """
        # Check elevation angle (avoid multipath)
        if elevation < MIN_ELEVATION_ANGLE:
            return False, f"Elevation too low: {elevation:.1f}° < {MIN_ELEVATION_ANGLE}°"

        # Check TEC range
        if tec < 0:
            return False, f"Negative TEC: {tec:.1f} TECU"

        if tec > MAX_TEC_VALUE:
            return False, f"TEC too high: {tec:.1f} TECU > {MAX_TEC_VALUE} TECU"

        # Check SNR if provided
        if snr is not None and snr < MIN_SNR:
            return False, f"SNR too low: {snr:.1f} dB-Hz < {MIN_SNR} dB-Hz"

        return True, ""

    def process_observable(
        self,
        observable: Dict[str, Any],
        receiver_position: Optional[ReceiverPosition] = None,
        satellite_position: Optional[Tuple[float, float, float]] = None
    ) -> Optional[TECMeasurement]:
        """
        Process GNSS observable to produce TEC measurement

        Args:
            observable: Observable dictionary from RTCM parser
            receiver_position: Receiver position (overrides default)
            satellite_position: Satellite position (lat, lon, alt) if known

        Returns:
            TECMeasurement or None if processing failed
        """
        try:
            # Get receiver position
            if receiver_position is None:
                receiver_position = self.receiver_position

            if receiver_position is None:
                self.logger.warning("No receiver position available")
                return None

            # Extract observables
            gnss_type = observable.get('gnss_type', 'GPS')
            sat_id = observable.get('satellite_id')

            l1_pseudorange = observable.get('l1_pseudorange')
            l2_pseudorange = observable.get('l2_pseudorange')

            if l1_pseudorange is None or l2_pseudorange is None:
                return None

            # Get frequencies
            if gnss_type == 'GPS':
                f1, f2 = GPS_L1_FREQ, GPS_L2_FREQ
            elif gnss_type == 'GLONASS':
                f1, f2 = GLONASS_L1_FREQ, GLONASS_L2_FREQ
            else:
                self.logger.warning(f"Unknown GNSS type: {gnss_type}")
                return None

            # Calculate TEC
            tec = self.calculate_tec_from_pseudorange(
                l1_pseudorange, l2_pseudorange, f1, f2
            )

            # Estimate error (simplified - assume 1m error on each frequency)
            tec_error = self.estimate_tec_error(1.0, 1.0, f1, f2)

            # Calculate geometry (if satellite position known)
            if satellite_position is not None:
                sat_lat, sat_lon, sat_alt = satellite_position
                azimuth, elevation = self.calculate_azimuth_elevation(
                    receiver_position.latitude,
                    receiver_position.longitude,
                    receiver_position.altitude,
                    sat_lat, sat_lon, sat_alt
                )
            else:
                # Without satellite position, cannot compute geometry
                # In production, would fetch ephemeris and compute satellite position
                self.logger.warning("No satellite position available, skipping")
                return None

            # Validate measurement
            snr = observable.get('l1_snr')
            is_valid, reason = self.validate_measurement(tec, elevation, snr)

            if not is_valid:
                self.logger.debug(f"Measurement rejected: {reason}")
                self._measurements_rejected += 1
                return None

            # Create measurement
            measurement = TECMeasurement(
                receiver_lat=receiver_position.latitude,
                receiver_lon=receiver_position.longitude,
                receiver_alt=receiver_position.altitude,
                satellite_lat=sat_lat,
                satellite_lon=sat_lon,
                satellite_alt=sat_alt,
                azimuth=azimuth,
                elevation=elevation,
                slant_tec=tec,
                tec_error=tec_error,
                timestamp=observable.get('epoch_time', datetime.utcnow()),
                satellite_id=sat_id,
                gnss_type=gnss_type
            )

            self._measurements_processed += 1
            return measurement

        except Exception as e:
            self.logger.error(f"Error processing observable: {e}", exc_info=True)
            return None

    @property
    def statistics(self) -> Dict[str, int]:
        """Get calculator statistics"""
        return {
            'measurements_processed': self._measurements_processed,
            'measurements_rejected': self._measurements_rejected
        }
