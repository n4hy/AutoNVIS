"""
Data Validation Utilities

Validates incoming observations for physical plausibility, freshness,
and data quality before assimilation.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.common.logging_config import ServiceLogger


class DataValidator:
    """
    Validates incoming observations from various sources
    """

    def __init__(self, staleness_threshold_sec: int = 300):
        """
        Initialize data validator

        Args:
            staleness_threshold_sec: Maximum age for data (seconds)
        """
        self.staleness_threshold = timedelta(seconds=staleness_threshold_sec)
        self.logger = ServiceLogger("ingestion", "validator")

    def validate_timestamp(self, timestamp_str: str) -> Tuple[bool, str]:
        """
        Validate timestamp freshness

        Args:
            timestamp_str: ISO 8601 timestamp string

        Returns:
            (is_valid, error_message)
        """
        try:
            # Parse timestamp
            if timestamp_str.endswith('Z'):
                timestamp = datetime.fromisoformat(timestamp_str[:-1])
            else:
                timestamp = datetime.fromisoformat(timestamp_str)

            # Check if too old
            age = datetime.utcnow() - timestamp
            if age > self.staleness_threshold:
                return False, f"Data too old: {age.total_seconds():.0f}s"

            # Check if in future (clock skew)
            if age < timedelta(seconds=-60):
                return False, f"Timestamp in future: {timestamp_str}"

            return True, ""

        except Exception as e:
            return False, f"Invalid timestamp format: {e}"

    def validate_xray_flux(self, flux: float) -> Tuple[bool, str]:
        """
        Validate GOES X-ray flux

        Args:
            flux: X-ray flux (W/m²)

        Returns:
            (is_valid, error_message)
        """
        # Physical range: ~10^-9 (quiet) to ~10^-3 (X10+ flare)
        if flux < 1e-10:
            return False, f"Flux too low: {flux:.2e} W/m²"

        if flux > 1e-2:
            return False, f"Flux too high: {flux:.2e} W/m²"

        if flux != flux:  # NaN check
            return False, "Flux is NaN"

        return True, ""

    def validate_solar_wind(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate ACE solar wind parameters

        Args:
            data: Solar wind data dictionary

        Returns:
            (is_valid, list_of_errors)
        """
        errors = []

        # Proton density: typically 1-20 particles/cm³, can spike to 100
        density = data.get('proton_density', 0)
        if density < 0.1 or density > 500:
            errors.append(f"Density out of range: {density} p/cm³")

        # Bulk speed: typically 300-800 km/s, can reach 1000+ during CMEs
        velocity = data.get('bulk_speed', 0)
        if velocity < 200 or velocity > 2000:
            errors.append(f"Velocity out of range: {velocity} km/s")

        # Proton temperature: typically 10^4 to 10^6 K
        temperature = data.get('proton_temperature', 0)
        if temperature > 0 and (temperature < 1e3 or temperature > 1e7):
            errors.append(f"Temperature out of range: {temperature:.2e} K")

        # Magnetic field: typically ±30 nT, can reach ±50 nT
        for component in ['bx_gsm', 'by_gsm', 'bz_gsm']:
            if component in data:
                b = data[component]
                if abs(b) > 100:
                    errors.append(f"{component} out of range: {b} nT")

        bt = data.get('bt', 0)
        if bt > 0 and (bt > 100 or bt < 0):
            errors.append(f"Total field out of range: {bt} nT")

        return len(errors) == 0, errors

    def validate_tec(self, tec: float, tec_error: float) -> Tuple[bool, str]:
        """
        Validate GNSS-TEC measurement

        Args:
            tec: TEC value (TECU)
            tec_error: TEC error estimate (TECU)

        Returns:
            (is_valid, error_message)
        """
        # TEC range: typically 1-100 TECU, can reach 200 at low latitudes
        if tec < 0 or tec > 300:
            return False, f"TEC out of range: {tec} TECU"

        # Error should be positive and reasonable
        if tec_error < 0 or tec_error > 50:
            return False, f"TEC error out of range: {tec_error} TECU"

        # Error should be smaller than measurement
        if tec_error > tec:
            return False, f"TEC error ({tec_error}) exceeds value ({tec})"

        return True, ""

    def validate_ionosonde(self, fof2: float, hmf2: float) -> Tuple[bool, List[str]]:
        """
        Validate ionosonde parameters

        Args:
            fof2: Critical frequency (MHz)
            hmf2: Peak height (km)

        Returns:
            (is_valid, list_of_errors)
        """
        errors = []

        # foF2: typically 2-15 MHz, can reach 20 at low latitudes
        if fof2 < 0.5 or fof2 > 30:
            errors.append(f"foF2 out of range: {fof2} MHz")

        # hmF2: typically 200-400 km, can vary 150-500
        if hmf2 < 100 or hmf2 > 600:
            errors.append(f"hmF2 out of range: {hmf2} km")

        return len(errors) == 0, errors

    def validate_elevation_angle(self, elevation: float) -> Tuple[bool, str]:
        """
        Validate satellite elevation angle for TEC measurement

        Args:
            elevation: Elevation angle (degrees)

        Returns:
            (is_valid, error_message)
        """
        # Minimum elevation to avoid multipath and excessive slant path
        min_elevation = 10.0

        if elevation < min_elevation:
            return False, f"Elevation too low: {elevation}° (min: {min_elevation}°)"

        if elevation > 90:
            return False, f"Elevation too high: {elevation}°"

        return True, ""

    def validate_geographic_coords(
        self,
        lat: float,
        lon: float
    ) -> Tuple[bool, str]:
        """
        Validate geographic coordinates

        Args:
            lat: Latitude (degrees)
            lon: Longitude (degrees)

        Returns:
            (is_valid, error_message)
        """
        if lat < -90 or lat > 90:
            return False, f"Latitude out of range: {lat}°"

        if lon < -180 or lon > 180:
            return False, f"Longitude out of range: {lon}°"

        return True, ""

    def log_validation_result(
        self,
        data_type: str,
        is_valid: bool,
        errors: Any,
        data: Dict[str, Any] = None
    ):
        """
        Log validation result

        Args:
            data_type: Type of data being validated
            is_valid: Validation result
            errors: Error message(s)
            data: Optional data dictionary for context
        """
        if is_valid:
            self.logger.debug(f"{data_type} validation passed")
        else:
            extra = {'data_type': data_type}
            if data:
                extra['data'] = data

            if isinstance(errors, list):
                error_msg = "; ".join(errors)
            else:
                error_msg = errors

            self.logger.warning(
                f"{data_type} validation failed: {error_msg}",
                extra=extra
            )
