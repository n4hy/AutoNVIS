"""
HF Link Budget Calculator

Calculates signal-to-noise ratio (SNR) for HF propagation paths by combining:
- Free space path loss
- Ionospheric absorption (D-layer, deviative)
- Ground reflection losses (multi-hop)
- Antenna gains
- Atmospheric/galactic noise

References:
- ITU-R P.533: HF propagation prediction method
- ITU-R P.372: Radio noise
- Davies, K. "Ionospheric Radio" (1990)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Physical constants
SPEED_OF_LIGHT = 299792.458  # km/s
BOLTZMANN_K = 1.38e-23  # J/K


class NoiseEnvironment(Enum):
    """ITU-R P.372 noise environment categories."""
    QUIET_RURAL = "quiet_rural"
    RURAL = "rural"
    RESIDENTIAL = "residential"
    URBAN = "urban"
    INDUSTRIAL = "industrial"


@dataclass
class AntennaConfig:
    """Antenna configuration for link budget."""
    gain_dbi: float = 0.0  # Gain in dBi (0 = isotropic)
    height_m: float = 10.0  # Height above ground
    efficiency: float = 0.9  # Antenna efficiency (0-1)

    @property
    def effective_gain_dbi(self) -> float:
        """Gain adjusted for efficiency."""
        return self.gain_dbi + 10 * np.log10(self.efficiency)


@dataclass
class TransmitterConfig:
    """Transmitter configuration."""
    power_watts: float = 100.0  # Transmit power
    antenna: AntennaConfig = None

    def __post_init__(self):
        if self.antenna is None:
            self.antenna = AntennaConfig()

    @property
    def power_dbw(self) -> float:
        """Power in dBW."""
        return 10 * np.log10(self.power_watts)

    @property
    def eirp_dbw(self) -> float:
        """Effective Isotropic Radiated Power in dBW."""
        return self.power_dbw + self.antenna.effective_gain_dbi


@dataclass
class ReceiverConfig:
    """Receiver configuration."""
    antenna: AntennaConfig = None
    bandwidth_hz: float = 3000.0  # Receiver bandwidth (SSB typical)
    noise_figure_db: float = 10.0  # Receiver noise figure
    noise_environment: NoiseEnvironment = NoiseEnvironment.RURAL

    def __post_init__(self):
        if self.antenna is None:
            self.antenna = AntennaConfig()


@dataclass
class PropagationLosses:
    """Breakdown of propagation losses."""
    free_space_db: float = 0.0
    d_layer_absorption_db: float = 0.0
    deviative_absorption_db: float = 0.0
    ground_reflection_db: float = 0.0
    polarization_coupling_db: float = 0.0
    auroral_absorption_db: float = 0.0
    focusing_gain_db: float = 0.0  # Can be negative (gain) or positive (loss)

    @property
    def total_db(self) -> float:
        """Total propagation loss in dB."""
        return (self.free_space_db +
                self.d_layer_absorption_db +
                self.deviative_absorption_db +
                self.ground_reflection_db +
                self.polarization_coupling_db +
                self.auroral_absorption_db -
                self.focusing_gain_db)  # Focusing gain reduces loss


@dataclass
class LinkBudgetResult:
    """Complete link budget calculation result."""
    # Signal path
    tx_power_dbw: float = 0.0
    tx_antenna_gain_dbi: float = 0.0
    eirp_dbw: float = 0.0

    # Losses
    losses: PropagationLosses = None
    total_path_loss_db: float = 0.0

    # Receive side
    rx_antenna_gain_dbi: float = 0.0
    signal_power_dbw: float = 0.0

    # Noise
    noise_power_dbw: float = 0.0
    noise_floor_dbm: float = 0.0
    external_noise_db: float = 0.0  # Above thermal

    # Result
    snr_db: float = 0.0

    # Metadata
    frequency_mhz: float = 0.0
    path_length_km: float = 0.0
    hop_count: int = 1

    def __post_init__(self):
        if self.losses is None:
            self.losses = PropagationLosses()

    @property
    def signal_report(self) -> str:
        """Return S-meter reading approximation."""
        # S9 = -73 dBm, each S-unit = 6 dB
        signal_dbm = self.signal_power_dbw + 30
        s_units = (signal_dbm + 73) / 6 + 9

        if s_units >= 9:
            over = int((s_units - 9) * 6)
            return f"S9+{over}dB" if over > 0 else "S9"
        else:
            return f"S{max(0, int(s_units))}"

    @property
    def quality_assessment(self) -> str:
        """Qualitative signal quality assessment."""
        if self.snr_db >= 30:
            return "Excellent"
        elif self.snr_db >= 20:
            return "Good"
        elif self.snr_db >= 10:
            return "Fair"
        elif self.snr_db >= 3:
            return "Marginal"
        else:
            return "Poor/Unusable"

    def to_dict(self) -> dict:
        """Convert to dictionary for display."""
        return {
            'SNR': f"{self.snr_db:.1f} dB",
            'Quality': self.quality_assessment,
            'Signal': self.signal_report,
            'EIRP': f"{self.eirp_dbw:.1f} dBW",
            'Path Loss': f"{self.total_path_loss_db:.1f} dB",
            'D-layer Abs': f"{self.losses.d_layer_absorption_db:.1f} dB",
            'Ground Loss': f"{self.losses.ground_reflection_db:.1f} dB",
            'Noise Floor': f"{self.noise_floor_dbm:.1f} dBm",
        }


class LinkBudgetCalculator:
    """
    HF Link Budget Calculator.

    Calculates expected SNR for ionospheric propagation paths.

    Example:
        calc = LinkBudgetCalculator()

        tx = TransmitterConfig(power_watts=100)
        rx = ReceiverConfig(bandwidth_hz=3000)

        result = calc.calculate(
            frequency_mhz=7.0,
            path_length_km=500,
            hop_count=1,
            reflection_height_km=300,
            solar_zenith_angle_deg=45,
            xray_flux=1e-6,
            tx_config=tx,
            rx_config=rx,
        )

        print(f"SNR: {result.snr_db:.1f} dB - {result.quality_assessment}")
    """

    def __init__(self):
        pass

    def calculate(
        self,
        frequency_mhz: float,
        path_length_km: float,
        hop_count: int = 1,
        reflection_height_km: float = 300.0,
        solar_zenith_angle_deg: float = 45.0,
        xray_flux: float = 1e-6,
        kp_index: float = 2.0,
        tx_config: Optional[TransmitterConfig] = None,
        rx_config: Optional[ReceiverConfig] = None,
        latitude_deg: float = 45.0,
        is_night: bool = False,
    ) -> LinkBudgetResult:
        """
        Calculate complete link budget.

        Args:
            frequency_mhz: Operating frequency
            path_length_km: Total ray path length (not ground distance)
            hop_count: Number of ionospheric hops
            reflection_height_km: Peak reflection altitude
            solar_zenith_angle_deg: Sun angle (0=overhead, 90=horizon)
            xray_flux: GOES X-ray flux in W/m² (affects D-layer)
            kp_index: Geomagnetic activity index (0-9)
            tx_config: Transmitter configuration
            rx_config: Receiver configuration
            latitude_deg: Path midpoint latitude (for auroral effects)
            is_night: True if propagation is during night

        Returns:
            LinkBudgetResult with complete breakdown
        """
        tx_config = tx_config or TransmitterConfig()
        rx_config = rx_config or ReceiverConfig()

        result = LinkBudgetResult(
            frequency_mhz=frequency_mhz,
            path_length_km=path_length_km,
            hop_count=hop_count,
        )

        # Transmitter
        result.tx_power_dbw = tx_config.power_dbw
        result.tx_antenna_gain_dbi = tx_config.antenna.effective_gain_dbi
        result.eirp_dbw = tx_config.eirp_dbw

        # Calculate losses
        result.losses = self._calculate_losses(
            frequency_mhz=frequency_mhz,
            path_length_km=path_length_km,
            hop_count=hop_count,
            reflection_height_km=reflection_height_km,
            solar_zenith_angle_deg=solar_zenith_angle_deg,
            xray_flux=xray_flux,
            kp_index=kp_index,
            latitude_deg=latitude_deg,
            is_night=is_night,
        )
        result.total_path_loss_db = result.losses.total_db

        # Receiver
        result.rx_antenna_gain_dbi = rx_config.antenna.effective_gain_dbi

        # Signal power at receiver
        result.signal_power_dbw = (
            result.eirp_dbw -
            result.total_path_loss_db +
            result.rx_antenna_gain_dbi
        )

        # Noise calculation
        noise_info = self._calculate_noise(
            frequency_mhz=frequency_mhz,
            bandwidth_hz=rx_config.bandwidth_hz,
            noise_figure_db=rx_config.noise_figure_db,
            environment=rx_config.noise_environment,
            latitude_deg=latitude_deg,
        )
        result.noise_power_dbw = noise_info['noise_power_dbw']
        result.noise_floor_dbm = noise_info['noise_floor_dbm']
        result.external_noise_db = noise_info['external_noise_db']

        # SNR
        result.snr_db = result.signal_power_dbw - result.noise_power_dbw

        return result

    def _calculate_losses(
        self,
        frequency_mhz: float,
        path_length_km: float,
        hop_count: int,
        reflection_height_km: float,
        solar_zenith_angle_deg: float,
        xray_flux: float,
        kp_index: float,
        latitude_deg: float,
        is_night: bool,
    ) -> PropagationLosses:
        """Calculate all propagation losses."""
        losses = PropagationLosses()

        # 1. Free space path loss
        # L_fs = 20*log10(d) + 20*log10(f) + 32.45 (d in km, f in MHz)
        losses.free_space_db = (
            20 * np.log10(path_length_km) +
            20 * np.log10(frequency_mhz) +
            32.45
        )

        # 2. D-layer absorption (non-deviative)
        losses.d_layer_absorption_db = self._d_layer_absorption(
            frequency_mhz=frequency_mhz,
            solar_zenith_angle_deg=solar_zenith_angle_deg,
            xray_flux=xray_flux,
            hop_count=hop_count,
            is_night=is_night,
        )

        # 3. Deviative absorption (near reflection point)
        losses.deviative_absorption_db = self._deviative_absorption(
            frequency_mhz=frequency_mhz,
            reflection_height_km=reflection_height_km,
        )

        # 4. Ground reflection loss (multi-hop)
        if hop_count > 1:
            # Each ground reflection loses 2-6 dB depending on ground type
            # Using average of 3 dB per reflection
            ground_reflections = hop_count - 1
            losses.ground_reflection_db = 3.0 * ground_reflections

        # 5. Polarization coupling (O/X mode)
        # Typically 1-3 dB loss
        losses.polarization_coupling_db = 1.5

        # 6. Auroral absorption (high latitudes, disturbed conditions)
        losses.auroral_absorption_db = self._auroral_absorption(
            frequency_mhz=frequency_mhz,
            kp_index=kp_index,
            latitude_deg=latitude_deg,
        )

        # 7. Focusing gain/loss
        # Ionospheric focusing can provide gain (negative loss)
        # Simplified model based on path geometry
        losses.focusing_gain_db = self._focusing_effect(
            reflection_height_km=reflection_height_km,
            path_length_km=path_length_km,
        )

        return losses

    def _d_layer_absorption(
        self,
        frequency_mhz: float,
        solar_zenith_angle_deg: float,
        xray_flux: float,
        hop_count: int,
        is_night: bool,
    ) -> float:
        """
        Calculate D-layer absorption.

        D-layer absorption is the dominant loss mechanism during daytime,
        especially at lower HF frequencies. It varies as:
        - 1/f² (inversely with frequency squared)
        - cos(χ) where χ is solar zenith angle
        - X-ray flux (solar flares increase absorption)

        Returns absorption in dB.
        """
        if is_night:
            # D-layer largely disappears at night
            return 0.5 * hop_count  # Minimal residual

        # Base absorption at 10 MHz, overhead sun
        # Typical value: 5-15 dB depending on solar activity

        # Solar zenith angle factor (0 at χ=90°, max at χ=0°)
        chi_rad = np.radians(min(solar_zenith_angle_deg, 89))
        zenith_factor = np.cos(chi_rad) ** 0.75

        if zenith_factor < 0.1:
            # Near sunset/sunrise, very low absorption
            return 1.0 * hop_count

        # X-ray flux factor
        # Quiet sun: ~1e-7 W/m², active: ~1e-5 W/m², flare: >1e-4 W/m²
        # Absorption scales roughly as sqrt(flux) relative to quiet
        flux_ref = 1e-6  # Reference flux (C-class level)
        flux_factor = np.sqrt(max(xray_flux, 1e-8) / flux_ref)
        flux_factor = np.clip(flux_factor, 0.5, 10.0)  # Limit range

        # Frequency factor: absorption ~ 1/f²
        freq_factor = (10.0 / frequency_mhz) ** 2

        # Base absorption coefficient (dB per hop at 10 MHz, overhead sun, quiet)
        base_absorption = 8.0

        absorption = (
            base_absorption *
            zenith_factor *
            flux_factor *
            freq_factor *
            hop_count
        )

        return min(absorption, 60.0)  # Cap at 60 dB (total blackout)

    def _deviative_absorption(
        self,
        frequency_mhz: float,
        reflection_height_km: float,
    ) -> float:
        """
        Calculate deviative absorption near the reflection point.

        Occurs when the ray slows down near the reflection height.
        Typically 1-5 dB, higher for lower frequencies and lower reflection heights.
        """
        # Higher reflection = less deviative absorption
        height_factor = max(0.5, (400 - reflection_height_km) / 200)

        # Lower frequency = more absorption
        freq_factor = (10.0 / frequency_mhz) ** 0.5

        base = 2.0  # Base deviative absorption in dB

        return base * height_factor * freq_factor

    def _auroral_absorption(
        self,
        frequency_mhz: float,
        kp_index: float,
        latitude_deg: float,
    ) -> float:
        """
        Calculate auroral absorption for high-latitude paths.

        Significant for paths crossing auroral zone (60-70° magnetic latitude)
        during geomagnetic disturbances.
        """
        # Auroral zone approximately 60-70° geomagnetic latitude
        # Simplified: use geographic latitude
        auroral_lat = 65.0
        lat_distance = abs(abs(latitude_deg) - auroral_lat)

        if lat_distance > 15:
            # Far from auroral zone
            return 0.0

        # Proximity factor (max at auroral_lat)
        proximity = 1.0 - (lat_distance / 15.0)

        # Kp factor (significant above Kp 4)
        if kp_index < 3:
            kp_factor = 0.0
        else:
            kp_factor = (kp_index - 3) / 3  # 0-2 for Kp 3-9

        # Frequency factor
        freq_factor = (10.0 / frequency_mhz) ** 1.5

        # Base auroral absorption
        base = 10.0  # Can be very high during storms

        return base * proximity * kp_factor * freq_factor

    def _focusing_effect(
        self,
        reflection_height_km: float,
        path_length_km: float,
    ) -> float:
        """
        Calculate ionospheric focusing gain.

        The curved ionosphere can act as a lens, providing gain (or loss).
        Typically ±3 dB effect.
        """
        # Simplified model: gain increases with reflection height
        # and decreases with path length

        # Higher reflection = more focusing
        height_factor = reflection_height_km / 300.0

        # Longer paths = less focusing (spreading)
        path_factor = 1000.0 / max(path_length_km, 100)

        # Base focusing gain
        base = 2.0

        gain = base * height_factor * path_factor

        # Limit to realistic range
        return np.clip(gain, -2.0, 4.0)

    def _calculate_noise(
        self,
        frequency_mhz: float,
        bandwidth_hz: float,
        noise_figure_db: float,
        environment: NoiseEnvironment,
        latitude_deg: float,
    ) -> dict:
        """
        Calculate noise floor based on ITU-R P.372.

        HF noise is dominated by:
        - Atmospheric noise (lightning, especially tropical)
        - Galactic noise (cosmic background)
        - Man-made noise (urban areas)

        Returns dict with noise_power_dbw, noise_floor_dbm, external_noise_db
        """
        # Thermal noise floor
        # N = kTB where k=Boltzmann, T=290K, B=bandwidth
        thermal_noise_dbw = (
            10 * np.log10(BOLTZMANN_K) +
            10 * np.log10(290) +
            10 * np.log10(bandwidth_hz)
        )

        # External noise figure (Fa) from ITU-R P.372
        # Approximate median values for different sources

        # Atmospheric noise (varies with frequency, location, season, time)
        # Higher at lower frequencies, higher in tropics
        fa_atmospheric = self._atmospheric_noise_figure(
            frequency_mhz, latitude_deg
        )

        # Galactic noise (dominates above ~20 MHz in quiet areas)
        fa_galactic = self._galactic_noise_figure(frequency_mhz)

        # Man-made noise
        fa_manmade = self._manmade_noise_figure(frequency_mhz, environment)

        # Total external noise is dominated by the largest source
        # (they add in power, but usually one dominates)
        fa_total = 10 * np.log10(
            10 ** (fa_atmospheric / 10) +
            10 ** (fa_galactic / 10) +
            10 ** (fa_manmade / 10)
        )

        # Total noise = thermal + external + receiver noise figure
        # External noise figure is relative to kTB
        total_noise_dbw = thermal_noise_dbw + max(fa_total, noise_figure_db)

        return {
            'noise_power_dbw': total_noise_dbw,
            'noise_floor_dbm': total_noise_dbw + 30,  # Convert to dBm
            'external_noise_db': fa_total,
        }

    def _atmospheric_noise_figure(
        self,
        frequency_mhz: float,
        latitude_deg: float,
    ) -> float:
        """
        Atmospheric noise figure (Fa) in dB above kTB.

        Based on ITU-R P.372 Figure 2 (simplified).
        """
        # Frequency dependence (approximately -20 dB/decade above 1 MHz)
        # Reference: Fa ≈ 55 dB at 2 MHz, tropical

        freq_factor = 55 - 25 * np.log10(frequency_mhz / 2)

        # Latitude factor (higher in tropics)
        if abs(latitude_deg) < 30:
            lat_factor = 0  # Tropical, highest noise
        elif abs(latitude_deg) < 60:
            lat_factor = -5  # Temperate
        else:
            lat_factor = -10  # Polar, lowest atmospheric

        return max(freq_factor + lat_factor, 0)

    def _galactic_noise_figure(self, frequency_mhz: float) -> float:
        """
        Galactic noise figure (Fa) in dB above kTB.

        Based on ITU-R P.372.
        """
        # Galactic noise: Fa ≈ 52 - 23*log10(f) for f in MHz
        # Significant above ~15-20 MHz
        if frequency_mhz < 15:
            return 0  # Masked by atmospheric

        return max(52 - 23 * np.log10(frequency_mhz), 0)

    def _manmade_noise_figure(
        self,
        frequency_mhz: float,
        environment: NoiseEnvironment,
    ) -> float:
        """
        Man-made noise figure (Fa) in dB above kTB.

        Based on ITU-R P.372 Figure 10.
        """
        # Base levels at 3 MHz (approximate medians)
        base_levels = {
            NoiseEnvironment.QUIET_RURAL: 35,
            NoiseEnvironment.RURAL: 40,
            NoiseEnvironment.RESIDENTIAL: 45,
            NoiseEnvironment.URBAN: 50,
            NoiseEnvironment.INDUSTRIAL: 55,
        }

        base = base_levels.get(environment, 45)

        # Frequency dependence: approximately -28 dB/decade
        freq_factor = -28 * np.log10(frequency_mhz / 3)

        return max(base + freq_factor, 0)

    def calculate_for_winner(
        self,
        winner,  # WinnerTriplet
        tx_config: Optional[TransmitterConfig] = None,
        rx_config: Optional[ReceiverConfig] = None,
        xray_flux: float = 1e-6,
        kp_index: float = 2.0,
        solar_zenith_angle_deg: float = 45.0,
        is_night: bool = False,
        latitude_deg: float = 45.0,
    ) -> LinkBudgetResult:
        """
        Calculate link budget for a winner triplet.

        Convenience method that extracts parameters from WinnerTriplet.
        """
        # Validate and extract parameters
        ground_range = winner.ground_range_km
        h = winner.reflection_height_km
        hop_count = max(winner.hop_count, 1)  # Ensure at least 1 hop

        # Validate ground_range - must be positive
        if ground_range <= 0:
            # Estimate from ray path if available
            if hasattr(winner, 'ray_path') and winner.ray_path and winner.ray_path.states:
                # Calculate from actual path positions
                states = winner.ray_path.states
                if len(states) >= 2:
                    start = states[0]
                    end = states[-1]
                    # Haversine distance
                    lat1, lon1, _ = start.lat_lon_alt()
                    lat2, lon2, _ = end.lat_lon_alt()
                    dlat = np.radians(lat2 - lat1)
                    dlon = np.radians(lon2 - lon1)
                    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
                    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
                    ground_range = 6371.0 * c
            # If still zero, use a reasonable default based on elevation
            if ground_range <= 0:
                # Higher elevation = shorter range (NVIS-like)
                # Lower elevation = longer range
                ground_range = max(100, 1000 * (1 - winner.elevation_deg / 90))
                logger.warning(f"Using estimated ground_range={ground_range:.0f}km for {winner.frequency_mhz:.1f}MHz")

        # Validate reflection height
        if h <= 0:
            h = 300.0  # Default F2 layer height
            logger.warning(f"Using default reflection height h={h:.0f}km")

        # Estimate path length from ground range and reflection height
        # Simple geometric model: path ≈ 2 * sqrt((range/2)² + h²) per hop
        hop_range = ground_range / hop_count
        path_per_hop = 2 * np.sqrt((hop_range/2)**2 + h**2)
        path_length = path_per_hop * hop_count

        # Ensure path_length is reasonable
        if path_length < 100:
            path_length = max(path_length, 2 * h)  # At minimum, vertical path

        return self.calculate(
            frequency_mhz=winner.frequency_mhz,
            path_length_km=path_length,
            hop_count=hop_count,
            reflection_height_km=h,
            solar_zenith_angle_deg=solar_zenith_angle_deg,
            xray_flux=xray_flux,
            kp_index=kp_index,
            tx_config=tx_config,
            rx_config=rx_config,
            latitude_deg=latitude_deg,
            is_night=is_night,
        )


def calculate_solar_zenith_angle(
    latitude_deg: float,
    longitude_deg: float,
    utc_time: Optional[datetime] = None,
) -> float:
    """
    Calculate solar zenith angle for a given location and time.

    Returns angle in degrees (0 = sun overhead, 90 = horizon).
    """
    if utc_time is None:
        utc_time = datetime.now(timezone.utc)

    # Day of year
    day_of_year = utc_time.timetuple().tm_yday

    # Hour angle (solar time)
    hour = utc_time.hour + utc_time.minute / 60
    solar_time = hour + longitude_deg / 15
    hour_angle = 15 * (solar_time - 12)  # degrees

    # Solar declination (approximate)
    declination = 23.45 * np.sin(np.radians(360 * (284 + day_of_year) / 365))

    # Solar zenith angle
    lat_rad = np.radians(latitude_deg)
    dec_rad = np.radians(declination)
    ha_rad = np.radians(hour_angle)

    cos_zenith = (
        np.sin(lat_rad) * np.sin(dec_rad) +
        np.cos(lat_rad) * np.cos(dec_rad) * np.cos(ha_rad)
    )

    zenith_angle = np.degrees(np.arccos(np.clip(cos_zenith, -1, 1)))

    return zenith_angle


def is_nighttime(
    latitude_deg: float,
    longitude_deg: float,
    utc_time: Optional[datetime] = None,
) -> bool:
    """Check if it's nighttime at a given location."""
    zenith = calculate_solar_zenith_angle(latitude_deg, longitude_deg, utc_time)
    return zenith > 96  # Civil twilight ends at 96°


# Test function
if __name__ == "__main__":
    print("Link Budget Calculator Test")
    print("=" * 60)

    calc = LinkBudgetCalculator()

    # Test cases
    test_cases = [
        {"freq": 7.0, "path": 500, "hops": 1, "desc": "7 MHz, 500km, 1 hop"},
        {"freq": 14.0, "path": 1000, "hops": 1, "desc": "14 MHz, 1000km, 1 hop"},
        {"freq": 7.0, "path": 3000, "hops": 2, "desc": "7 MHz, 3000km, 2 hops"},
        {"freq": 3.5, "path": 300, "hops": 1, "desc": "3.5 MHz NVIS, 300km"},
    ]

    tx = TransmitterConfig(power_watts=100)
    rx = ReceiverConfig(bandwidth_hz=3000)

    for tc in test_cases:
        result = calc.calculate(
            frequency_mhz=tc["freq"],
            path_length_km=tc["path"],
            hop_count=tc["hops"],
            reflection_height_km=300,
            solar_zenith_angle_deg=45,
            xray_flux=1e-6,
            tx_config=tx,
            rx_config=rx,
        )

        print(f"\n{tc['desc']}:")
        print(f"  Path Loss: {result.total_path_loss_db:.1f} dB")
        print(f"    Free Space: {result.losses.free_space_db:.1f} dB")
        print(f"    D-layer: {result.losses.d_layer_absorption_db:.1f} dB")
        print(f"    Ground: {result.losses.ground_reflection_db:.1f} dB")
        print(f"  Signal: {result.signal_report}")
        print(f"  SNR: {result.snr_db:.1f} dB ({result.quality_assessment})")
