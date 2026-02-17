"""
NVIS Frequency Optimizer

Find optimal frequencies for Near Vertical Incidence Skywave (NVIS)
communication using the homing algorithm.

NVIS uses high elevation angles (70-90°) for short-range HF communication,
providing coverage within the "skip zone" that normal oblique propagation
cannot reach. This is ideal for regional emergency communications, military
tactical networks, and areas with difficult terrain.

Key Features:
- Automatic frequency selection based on real-time ionospheric conditions
- Multi-hop path finding for extended range
- Signal quality estimation considering absorption and multipath
- Optimal antenna configuration recommendations

Example:
    from raytracer.nvis_optimizer import NVISOptimizer
    from raytracer.electron_density import create_test_profile

    model = create_test_profile()
    model.update_from_realtime(foF2=8.5, hmF2=320.0)

    optimizer = NVISOptimizer(model)

    # Find best frequencies for NVIS
    results = optimizer.optimize(
        tx_lat=40.0, tx_lon=-105.0,
        rx_lat=40.5, rx_lon=-104.5
    )

    for r in results:
        print(f"{r.frequency:.1f} MHz at {r.elevation}° → {r.ground_range:.0f} km")
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
from datetime import datetime, timezone
from enum import Enum
import logging

from .electron_density import IonosphericModel
from .haselgrove import HaselgroveSolver, RayMode, RayTermination
from .pharlap_interface import PHaRLAPInterface, PropagationResult

logger = logging.getLogger(__name__)


class SignalQuality(Enum):
    """Signal quality rating."""
    EXCELLENT = 5
    GOOD = 4
    FAIR = 3
    MARGINAL = 2
    UNUSABLE = 1


@dataclass
class NVISFrequency:
    """
    NVIS frequency optimization result.

    Attributes:
        frequency: Operating frequency (MHz)
        elevation: Optimal elevation angle (degrees)
        azimuth: Azimuth toward receiver (degrees)
        ground_range: Ground distance covered (km)
        path_length: Total ray path (km)
        reflection_height: Ionospheric reflection altitude (km)
        quality: Estimated signal quality
        margin_db: Estimated link margin (dB)
        fof2_ratio: Ratio of frequency to foF2
    """
    frequency: float
    elevation: float
    azimuth: float
    ground_range: float
    path_length: float
    reflection_height: float
    quality: SignalQuality
    margin_db: float
    fof2_ratio: float


@dataclass
class NVISResult:
    """
    Complete NVIS optimization result.

    Attributes:
        tx_location: Transmitter (lat, lon, alt)
        rx_location: Receiver (lat, lon, alt)
        target_range: Great circle distance Tx to Rx (km)
        working_frequencies: List of frequencies that work
        recommended_frequency: Best frequency to use
        foF2: Current F2 layer critical frequency (MHz)
        critical_frequency: Frequency where n=0 at peak
        absorption_warning: True if D-region absorption is high
    """
    tx_location: Tuple[float, float, float]
    rx_location: Tuple[float, float, float]
    target_range: float
    working_frequencies: List[NVISFrequency]
    recommended_frequency: Optional[NVISFrequency]
    foF2: float
    critical_frequency: float
    absorption_warning: bool = False


class NVISOptimizer:
    """
    NVIS frequency optimizer using homing algorithm.

    NVIS propagation characteristics:
    - Elevation angles: 70-90° (near vertical)
    - Typical ranges: 0-400 km (skip zone coverage)
    - Frequencies: 2-10 MHz (below foF2)
    - Reflection height: 200-400 km (F2 layer)

    The optimizer finds "winner triplets" (frequency, elevation, azimuth)
    that successfully connect transmitter to receiver via F2 layer
    reflection.

    Frequency selection guidelines:
    - Below foF2: Required for reflection
    - Above LUF: Required to penetrate D-layer absorption
    - Optimal: 0.7-0.85 × foF2 (FOT range)
    """

    # NVIS parameter ranges
    NVIS_ELEVATION_MIN = 70.0  # degrees
    NVIS_ELEVATION_MAX = 90.0  # degrees
    NVIS_ELEVATION_STEP = 2.0  # degrees

    NVIS_FREQ_MIN = 2.0  # MHz
    NVIS_FREQ_MAX = 10.0  # MHz
    NVIS_FREQ_STEP = 0.25  # MHz

    # Maximum useful NVIS range (km)
    NVIS_MAX_RANGE = 400.0

    def __init__(
        self,
        ionosphere: IonosphericModel,
        step_km: float = 5.0,
    ):
        """
        Initialize NVIS optimizer.

        Args:
            ionosphere: IonosphericModel for electron density
            step_km: Ray tracing step size
        """
        self.ionosphere = ionosphere
        self.interface = PHaRLAPInterface(ionosphere, default_step_km=step_km)

    def optimize(
        self,
        tx_lat: float,
        tx_lon: float,
        rx_lat: float,
        rx_lon: float,
        tx_alt: float = 0.0,
        rx_alt: float = 0.0,
        freq_range: Optional[Tuple[float, float]] = None,
    ) -> NVISResult:
        """
        Find optimal NVIS frequencies for Tx-Rx path.

        Args:
            tx_lat, tx_lon: Transmitter position (degrees)
            rx_lat, rx_lon: Receiver position (degrees)
            tx_alt, rx_alt: Altitudes (km)
            freq_range: Optional custom frequency range (MHz)

        Returns:
            NVISResult with working frequencies and recommendation
        """
        # Calculate path geometry
        target_range = self.interface._haversine_distance(
            tx_lat, tx_lon, rx_lat, rx_lon
        )
        azimuth = self.interface._bearing(tx_lat, tx_lon, rx_lat, rx_lon)

        logger.info(f"NVIS optimization: {target_range:.1f} km @ {azimuth:.0f}°")

        # Get current ionospheric conditions
        status = self.ionosphere.get_correction_status()
        foF2 = status.get('current_foF2', 7.0)

        # Check if NVIS is appropriate for this range
        if target_range > self.NVIS_MAX_RANGE:
            logger.warning(
                f"Target range {target_range:.0f} km exceeds typical NVIS range"
            )

        # Frequency search range
        if freq_range:
            freq_min, freq_max = freq_range
        else:
            freq_min = self.NVIS_FREQ_MIN
            # Don't search above foF2 (won't reflect)
            freq_max = min(self.NVIS_FREQ_MAX, foF2 * 0.95)

        # Find working frequencies using homing algorithm
        working_frequencies = self._homing_search(
            tx_lat, tx_lon, tx_alt,
            target_range, azimuth,
            freq_min, freq_max, foF2,
        )

        # Sort by quality
        working_frequencies.sort(
            key=lambda x: (-x.quality.value, abs(x.ground_range - target_range))
        )

        # Select recommendation (best quality that matches range)
        recommended = None
        for wf in working_frequencies:
            range_error = abs(wf.ground_range - target_range)
            if range_error < max(50.0, target_range * 0.2):
                recommended = wf
                break

        # Check D-region absorption (simple heuristic)
        absorption_warning = self._check_absorption()

        return NVISResult(
            tx_location=(tx_lat, tx_lon, tx_alt),
            rx_location=(rx_lat, rx_lon, rx_alt),
            target_range=target_range,
            working_frequencies=working_frequencies,
            recommended_frequency=recommended,
            foF2=foF2,
            critical_frequency=foF2,
            absorption_warning=absorption_warning,
        )

    def _homing_search(
        self,
        tx_lat: float,
        tx_lon: float,
        tx_alt: float,
        target_range: float,
        azimuth: float,
        freq_min: float,
        freq_max: float,
        foF2: float,
    ) -> List[NVISFrequency]:
        """
        Homing algorithm: find frequency/elevation combinations that work.
        """
        working = []

        frequencies = np.arange(freq_min, freq_max + self.NVIS_FREQ_STEP,
                               self.NVIS_FREQ_STEP)
        elevations = np.arange(self.NVIS_ELEVATION_MIN,
                              self.NVIS_ELEVATION_MAX + self.NVIS_ELEVATION_STEP,
                              self.NVIS_ELEVATION_STEP)

        for freq in frequencies:
            for elevation in elevations:
                result = self.interface.trace_ray(
                    frequency=freq,
                    elevation=elevation,
                    azimuth=azimuth,
                    tx_lat=tx_lat,
                    tx_lon=tx_lon,
                    tx_alt=tx_alt,
                )

                if result.success:
                    # Estimate quality
                    quality = self._estimate_quality(
                        freq, foF2,
                        result.ground_range, target_range,
                        result.max_altitude,
                    )

                    # Calculate link margin (simplified)
                    margin = self._estimate_margin(
                        freq, result.path_length,
                        result.max_altitude,
                    )

                    working.append(NVISFrequency(
                        frequency=freq,
                        elevation=elevation,
                        azimuth=azimuth,
                        ground_range=result.ground_range,
                        path_length=result.path_length,
                        reflection_height=result.max_altitude,
                        quality=quality,
                        margin_db=margin,
                        fof2_ratio=freq / foF2 if foF2 > 0 else 0,
                    ))

        return working

    def _estimate_quality(
        self,
        frequency: float,
        foF2: float,
        ground_range: float,
        target_range: float,
        reflection_height: float,
    ) -> SignalQuality:
        """
        Estimate signal quality based on propagation parameters.
        """
        # Frequency ratio (optimal is 0.7-0.85 × foF2)
        ratio = frequency / foF2 if foF2 > 0 else 0

        # Range match (how close to target)
        range_error_pct = abs(ground_range - target_range) / max(target_range, 1) * 100

        # Score calculation
        score = 5

        # Frequency ratio penalty
        if ratio > 0.95:
            score -= 2  # Too close to MUF, unreliable
        elif ratio > 0.85:
            score -= 1  # Getting marginal
        elif ratio < 0.5:
            score -= 1  # More absorption likely

        # Range error penalty
        if range_error_pct > 50:
            score -= 2
        elif range_error_pct > 20:
            score -= 1

        # Reflection height consideration
        if reflection_height < 150:
            score -= 1  # E-layer reflection, less stable

        score = max(1, min(5, score))

        return SignalQuality(score)

    def _estimate_margin(
        self,
        frequency: float,
        path_length: float,
        reflection_height: float,
    ) -> float:
        """
        Estimate link margin in dB (simplified model).

        Considers:
        - Free space path loss
        - D-region absorption
        - Ionospheric absorption
        """
        # Free space path loss (simplified for HF)
        # FSPL = 20*log10(d) + 20*log10(f) + 32.44 (d in km, f in MHz)
        fspl = 20 * np.log10(max(path_length, 1)) + 20 * np.log10(frequency) + 32.44

        # D-region absorption (increases at lower frequencies)
        # Rough model: absorption ~ (1/f)^2 × secant(zenith angle)
        d_absorption = 10 * (3.0 / frequency) ** 2  # dB, reference at 3 MHz

        # Assume 100W transmitter, 0 dBi antenna, -100 dBm sensitivity
        # Link budget = Tx power (50 dBm) - FSPL - absorption + Rx gain (0 dBi)
        margin = 50 - fspl - d_absorption - (-100)

        return margin

    def _check_absorption(self) -> bool:
        """
        Check if D-region absorption is elevated.

        Would use solar X-ray data from propagation display.
        For now, return False (no warning).
        """
        # TODO: Connect to X-ray flux data from PropagationDataClient
        return False

    def get_frequency_recommendation(
        self,
        tx_lat: float,
        tx_lon: float,
        rx_lat: float,
        rx_lon: float,
    ) -> Optional[float]:
        """
        Get single best frequency recommendation.

        Args:
            tx_lat, tx_lon: Transmitter position
            rx_lat, rx_lon: Receiver position

        Returns:
            Recommended frequency in MHz, or None if no path found
        """
        result = self.optimize(tx_lat, tx_lon, rx_lat, rx_lon)

        if result.recommended_frequency:
            return result.recommended_frequency.frequency
        elif result.working_frequencies:
            return result.working_frequencies[0].frequency
        else:
            return None

    def get_antenna_recommendation(
        self,
        optimal_elevation: float,
    ) -> Dict[str, str]:
        """
        Get antenna recommendations for NVIS operation.

        Args:
            optimal_elevation: Elevation angle in degrees

        Returns:
            Dictionary with antenna recommendations
        """
        return {
            "type": "Horizontal dipole or loop",
            "height": "λ/4 to λ/2 above ground (0.1λ minimum)",
            "orientation": "Broadside toward receiver",
            "polarization": "Horizontal (matches O-mode reflection)",
            "notes": (
                f"For {optimal_elevation:.0f}° elevation, mount antenna low "
                "to maximize high-angle radiation. NVIS antennas work best "
                "when ground-reflected waves reinforce sky wave."
            ),
        }


def test_nvis_optimizer():
    """Test the NVIS optimizer."""
    from .electron_density import create_test_profile

    print("NVIS Optimizer Test")
    print("=" * 50)

    # Create model with realistic conditions
    model = create_test_profile()
    model.update_from_realtime(foF2=8.0, hmF2=300.0)

    optimizer = NVISOptimizer(model)

    # Test optimization for short range
    print("\n1. NVIS optimization (40 km path):")
    result = optimizer.optimize(
        tx_lat=40.0, tx_lon=-105.0,
        rx_lat=40.3, rx_lon=-104.7,
    )

    print(f"   Target range: {result.target_range:.1f} km")
    print(f"   foF2: {result.foF2:.1f} MHz")
    print(f"   Working frequencies: {len(result.working_frequencies)}")

    if result.recommended_frequency:
        rf = result.recommended_frequency
        print(f"\n   Recommended: {rf.frequency:.2f} MHz")
        print(f"   Elevation: {rf.elevation:.0f}°")
        print(f"   Quality: {rf.quality.name}")
        print(f"   Link margin: {rf.margin_db:.0f} dB")
        print(f"   Reflection height: {rf.reflection_height:.0f} km")
    else:
        print("   No recommendation available")

    # Show top 5 frequencies
    print("\n   Top 5 frequencies:")
    for i, wf in enumerate(result.working_frequencies[:5]):
        print(f"   {i+1}. {wf.frequency:.2f} MHz @ {wf.elevation:.0f}° "
              f"→ {wf.ground_range:.0f} km ({wf.quality.name})")

    # Test single recommendation API
    print("\n2. Quick frequency recommendation:")
    freq = optimizer.get_frequency_recommendation(
        tx_lat=40.0, tx_lon=-105.0,
        rx_lat=40.2, rx_lon=-104.8,
    )
    if freq:
        print(f"   Recommended: {freq:.2f} MHz")
    else:
        print("   No path found")

    # Antenna recommendation
    print("\n3. Antenna recommendation:")
    antenna = optimizer.get_antenna_recommendation(85.0)
    for key, value in antenna.items():
        print(f"   {key}: {value}")


if __name__ == "__main__":
    test_nvis_optimizer()
