"""
PHaRLAP-Style Ray Tracing Interface

High-level interface for HF ray tracing providing PHaRLAP-compatible API.
Supports ray tracing, MUF calculation, and synthetic ionogram generation.

Key Functions:
    trace_ray: Single ray trace from transmitter
    trace_fan: Fan of rays at multiple elevations
    find_muf: Maximum usable frequency between two points
    generate_ionogram: Synthetic oblique ionogram
    find_skip_zone: Calculate skip distance

Example:
    from raytracer.pharlap_interface import PHaRLAPInterface
    from raytracer.electron_density import create_test_profile

    model = create_test_profile()
    rt = PHaRLAPInterface(model)

    # Trace a single ray
    path = rt.trace_ray(
        frequency=7.0,
        elevation=80.0,
        azimuth=0.0,
        tx_lat=40.0, tx_lon=-105.0, tx_alt=0.0
    )

    # Find MUF
    muf = rt.find_muf(
        tx_lat=40.0, tx_lon=-105.0,
        rx_lat=42.0, rx_lon=-100.0
    )
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
from datetime import datetime, timezone
import logging

from .electron_density import IonosphericModel
from .haselgrove import HaselgroveSolver, RayPath, RayState, RayMode, RayTermination

logger = logging.getLogger(__name__)


@dataclass
class PropagationResult:
    """
    Result of a propagation calculation.

    Attributes:
        success: Whether propagation path was found
        frequency: Frequency used (MHz)
        elevation: Launch elevation (degrees)
        azimuth: Launch azimuth (degrees)
        tx_location: Transmitter (lat, lon, alt)
        rx_location: Receiver (lat, lon, alt) if ground hit
        path_length: Total ray path length (km)
        group_path: Group path / virtual height (km)
        ground_range: Great circle distance Tx to Rx (km)
        max_altitude: Maximum altitude reached (km)
        ray_path: Full RayPath object for detailed analysis
    """
    success: bool
    frequency: float
    elevation: float
    azimuth: float
    tx_location: Tuple[float, float, float]
    rx_location: Optional[Tuple[float, float, float]] = None
    path_length: float = 0.0
    group_path: float = 0.0
    ground_range: float = 0.0
    max_altitude: float = 0.0
    ray_path: Optional[RayPath] = None


@dataclass
class MUFResult:
    """
    Maximum Usable Frequency calculation result.

    Attributes:
        muf: Maximum usable frequency (MHz)
        fot: Frequency of optimum traffic (85% of MUF)
        luf: Lowest usable frequency estimate (MHz)
        best_elevation: Elevation angle for MUF (degrees)
        path_length: Path length at MUF (km)
        max_altitude: Reflection height (km)
        skip_distance: Minimum ground range at this frequency (km)
    """
    muf: float
    fot: float  # Frequency of optimum traffic
    luf: float  # Lowest usable frequency
    best_elevation: float
    path_length: float
    max_altitude: float
    skip_distance: float


@dataclass
class IonogramPoint:
    """
    Single point on a synthetic ionogram.

    Attributes:
        frequency: Transmission frequency (MHz)
        virtual_height: Virtual/group height (km)
        elevation: Launch elevation (degrees)
        mode: O-mode or X-mode
        ground_range: Ground distance (km)
    """
    frequency: float
    virtual_height: float
    elevation: float
    mode: str
    ground_range: float


class PHaRLAPInterface:
    """
    High-level ray tracing interface.

    Provides PHaRLAP-compatible API for HF propagation calculations
    using the Haselgrove ray tracer with real-time ionospheric data.

    Example:
        model = IonosphericModel()
        model.update_from_realtime(foF2=8.5, hmF2=320.0)

        rt = PHaRLAPInterface(model)

        # Check if path exists
        result = rt.trace_ray(
            frequency=7.0,
            elevation=45.0,
            azimuth=0.0,
            tx_lat=40.0, tx_lon=-105.0
        )

        if result.success:
            print(f"Path found: {result.ground_range:.1f} km")
    """

    def __init__(
        self,
        ionosphere: IonosphericModel,
        default_step_km: float = 5.0,
        max_path_km: float = 5000.0
    ):
        """
        Initialize PHaRLAP interface.

        Args:
            ionosphere: IonosphericModel for electron density
            default_step_km: Default integration step size
            max_path_km: Maximum ray path length
        """
        self.ionosphere = ionosphere
        self.solver = HaselgroveSolver(ionosphere)
        self.default_step_km = default_step_km
        self.max_path_km = max_path_km

    def trace_ray(
        self,
        frequency: float,
        elevation: float,
        azimuth: float,
        tx_lat: float,
        tx_lon: float,
        tx_alt: float = 0.0,
        mode: RayMode = RayMode.ORDINARY,
    ) -> PropagationResult:
        """
        Trace a single ray from transmitter.

        Args:
            frequency: Transmission frequency (MHz)
            elevation: Launch elevation above horizon (degrees)
            azimuth: Launch azimuth clockwise from north (degrees)
            tx_lat: Transmitter latitude (degrees)
            tx_lon: Transmitter longitude (degrees)
            tx_alt: Transmitter altitude (km, default 0)
            mode: O-mode or X-mode

        Returns:
            PropagationResult with path details
        """
        path = self.solver.trace_ray(
            tx_lat=tx_lat,
            tx_lon=tx_lon,
            tx_alt=tx_alt,
            elevation=elevation,
            azimuth=azimuth,
            frequency_mhz=frequency,
            mode=mode,
            step_km=self.default_step_km,
            max_path_km=self.max_path_km,
        )

        # Calculate max altitude
        max_alt = max(s.altitude() for s in path.states) if path.states else 0.0

        # Build result
        result = PropagationResult(
            success=(path.termination == RayTermination.GROUND_HIT),
            frequency=frequency,
            elevation=elevation,
            azimuth=azimuth,
            tx_location=(tx_lat, tx_lon, tx_alt),
            path_length=path.total_path_length,
            group_path=path.group_path_length,
            ground_range=path.ground_range,
            max_altitude=max_alt,
            ray_path=path,
        )

        if path.landing_position:
            result.rx_location = path.landing_position

        return result

    def trace_fan(
        self,
        frequency: float,
        elevation_range: Tuple[float, float],
        elevation_step: float,
        azimuth: float,
        tx_lat: float,
        tx_lon: float,
        tx_alt: float = 0.0,
        mode: RayMode = RayMode.ORDINARY,
    ) -> List[PropagationResult]:
        """
        Trace a fan of rays at multiple elevations.

        Useful for coverage analysis and skip zone calculation.

        Args:
            frequency: Transmission frequency (MHz)
            elevation_range: (min_elevation, max_elevation) degrees
            elevation_step: Step between elevations (degrees)
            azimuth: Fixed azimuth (degrees)
            tx_lat, tx_lon, tx_alt: Transmitter position
            mode: O-mode or X-mode

        Returns:
            List of PropagationResult for each elevation
        """
        el_min, el_max = elevation_range
        elevations = np.arange(el_min, el_max + elevation_step, elevation_step)

        results = []
        for elevation in elevations:
            result = self.trace_ray(
                frequency=frequency,
                elevation=elevation,
                azimuth=azimuth,
                tx_lat=tx_lat,
                tx_lon=tx_lon,
                tx_alt=tx_alt,
                mode=mode,
            )
            results.append(result)

        return results

    def find_muf(
        self,
        tx_lat: float,
        tx_lon: float,
        rx_lat: float,
        rx_lon: float,
        tx_alt: float = 0.0,
        rx_alt: float = 0.0,
        freq_min: float = 2.0,
        freq_max: float = 30.0,
        freq_step: float = 0.5,
        elevation_range: Tuple[float, float] = (5.0, 89.0),
    ) -> MUFResult:
        """
        Find Maximum Usable Frequency between two points.

        Uses homing algorithm: for each frequency, sweep elevations
        to find paths that connect Tx to Rx.

        Args:
            tx_lat, tx_lon: Transmitter position (degrees)
            rx_lat, rx_lon: Receiver position (degrees)
            tx_alt, rx_alt: Altitudes (km)
            freq_min, freq_max: Frequency search range (MHz)
            freq_step: Frequency step (MHz)
            elevation_range: Elevation search range (degrees)

        Returns:
            MUFResult with MUF and related parameters
        """
        # Calculate target ground range and azimuth
        target_range = self._haversine_distance(tx_lat, tx_lon, rx_lat, rx_lon)
        azimuth = self._bearing(tx_lat, tx_lon, rx_lat, rx_lon)

        logger.debug(f"Finding MUF: Tx({tx_lat}, {tx_lon}) -> Rx({rx_lat}, {rx_lon})")
        logger.debug(f"Target range: {target_range:.1f} km, azimuth: {azimuth:.1f}°")

        muf = freq_min
        best_elevation = 45.0
        best_path_length = 0.0
        best_max_alt = 0.0
        skip_distance = 0.0

        # Search frequencies from high to low
        frequencies = np.arange(freq_max, freq_min - freq_step, -freq_step)

        for freq in frequencies:
            # Sweep elevations to find paths near target range
            fan = self.trace_fan(
                frequency=freq,
                elevation_range=elevation_range,
                elevation_step=2.0,
                azimuth=azimuth,
                tx_lat=tx_lat,
                tx_lon=tx_lon,
                tx_alt=tx_alt,
            )

            # Find paths that hit near target
            tolerance = max(50.0, target_range * 0.1)  # 10% or 50km tolerance

            for result in fan:
                if result.success:
                    range_error = abs(result.ground_range - target_range)
                    if range_error < tolerance:
                        # Found a path at this frequency
                        if freq > muf:
                            muf = freq
                            best_elevation = result.elevation
                            best_path_length = result.path_length
                            best_max_alt = result.max_altitude
                        break

            # Calculate skip distance (minimum ground range at this frequency)
            ground_ranges = [r.ground_range for r in fan if r.success]
            if ground_ranges:
                skip_distance = min(ground_ranges)

        return MUFResult(
            muf=muf,
            fot=muf * 0.85,  # Frequency of optimum traffic
            luf=max(2.0, muf * 0.5),  # Rough estimate
            best_elevation=best_elevation,
            path_length=best_path_length,
            max_altitude=best_max_alt,
            skip_distance=skip_distance,
        )

    def find_nvis_frequencies(
        self,
        tx_lat: float,
        tx_lon: float,
        rx_lat: float,
        rx_lon: float,
        freq_min: float = 2.0,
        freq_max: float = 10.0,
        freq_step: float = 0.25,
    ) -> List[Tuple[float, float, float]]:
        """
        Find NVIS frequencies for short-range communication.

        NVIS (Near Vertical Incidence Skywave) uses high elevation
        angles (70-90°) for short-range coverage with no skip zone.

        Args:
            tx_lat, tx_lon: Transmitter position
            rx_lat, rx_lon: Receiver position
            freq_min, freq_max: Search range (MHz)
            freq_step: Frequency step (MHz)

        Returns:
            List of (frequency, elevation, ground_range) tuples that work
        """
        target_range = self._haversine_distance(tx_lat, tx_lon, rx_lat, rx_lon)
        azimuth = self._bearing(tx_lat, tx_lon, rx_lat, rx_lon)

        # NVIS uses high elevation angles
        nvis_elevations = np.arange(70.0, 91.0, 2.0)

        working_freqs = []

        for freq in np.arange(freq_min, freq_max + freq_step, freq_step):
            for elevation in nvis_elevations:
                result = self.trace_ray(
                    frequency=freq,
                    elevation=elevation,
                    azimuth=azimuth,
                    tx_lat=tx_lat,
                    tx_lon=tx_lon,
                )

                if result.success:
                    # For NVIS, we want short ground range
                    if result.ground_range < 400:  # Typical NVIS range
                        working_freqs.append((
                            freq,
                            elevation,
                            result.ground_range
                        ))

        return working_freqs

    def generate_ionogram(
        self,
        tx_lat: float,
        tx_lon: float,
        azimuth: float,
        freq_range: Tuple[float, float] = (2.0, 15.0),
        freq_step: float = 0.5,
        elevation_range: Tuple[float, float] = (5.0, 89.0),
        elevation_step: float = 5.0,
    ) -> List[IonogramPoint]:
        """
        Generate synthetic oblique ionogram.

        Creates virtual height vs frequency data similar to an ionosonde.

        Args:
            tx_lat, tx_lon: Transmitter position
            azimuth: Direction of propagation (degrees)
            freq_range: Frequency sweep range (MHz)
            freq_step: Frequency step (MHz)
            elevation_range: Elevation sweep range (degrees)
            elevation_step: Elevation step (degrees)

        Returns:
            List of IonogramPoint for plotting
        """
        points = []
        freq_min, freq_max = freq_range

        for freq in np.arange(freq_min, freq_max + freq_step, freq_step):
            for elevation in np.arange(
                elevation_range[0],
                elevation_range[1] + elevation_step,
                elevation_step
            ):
                result = self.trace_ray(
                    frequency=freq,
                    elevation=elevation,
                    azimuth=azimuth,
                    tx_lat=tx_lat,
                    tx_lon=tx_lon,
                )

                if result.success:
                    points.append(IonogramPoint(
                        frequency=freq,
                        virtual_height=result.group_path / 2,  # One-way
                        elevation=elevation,
                        mode="O",
                        ground_range=result.ground_range,
                    ))

        return points

    def get_coverage_map(
        self,
        frequency: float,
        tx_lat: float,
        tx_lon: float,
        tx_alt: float = 0.0,
        azimuth_range: Tuple[float, float] = (0.0, 360.0),
        azimuth_step: float = 10.0,
        elevation_range: Tuple[float, float] = (5.0, 89.0),
        elevation_step: float = 5.0,
    ) -> List[PropagationResult]:
        """
        Generate coverage map by tracing rays in all directions.

        Args:
            frequency: Transmission frequency (MHz)
            tx_lat, tx_lon, tx_alt: Transmitter position
            azimuth_range: Azimuth sweep range (degrees)
            azimuth_step: Azimuth step (degrees)
            elevation_range: Elevation sweep range (degrees)
            elevation_step: Elevation step (degrees)

        Returns:
            List of PropagationResult for coverage analysis
        """
        results = []
        az_min, az_max = azimuth_range

        for azimuth in np.arange(az_min, az_max, azimuth_step):
            fan = self.trace_fan(
                frequency=frequency,
                elevation_range=elevation_range,
                elevation_step=elevation_step,
                azimuth=azimuth,
                tx_lat=tx_lat,
                tx_lon=tx_lon,
                tx_alt=tx_alt,
            )
            results.extend(fan)

        return results

    def _haversine_distance(
        self,
        lat1: float, lon1: float,
        lat2: float, lon2: float
    ) -> float:
        """Calculate great-circle distance in km."""
        from . import EARTH_RADIUS_KM

        lat1_r = np.radians(lat1)
        lat2_r = np.radians(lat2)
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)

        a = (np.sin(dlat/2)**2 +
             np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon/2)**2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

        return EARTH_RADIUS_KM * c

    def _bearing(
        self,
        lat1: float, lon1: float,
        lat2: float, lon2: float
    ) -> float:
        """Calculate initial bearing from point 1 to point 2 in degrees."""
        lat1_r = np.radians(lat1)
        lat2_r = np.radians(lat2)
        dlon = np.radians(lon2 - lon1)

        x = np.sin(dlon) * np.cos(lat2_r)
        y = (np.cos(lat1_r) * np.sin(lat2_r) -
             np.sin(lat1_r) * np.cos(lat2_r) * np.cos(dlon))

        bearing = np.degrees(np.arctan2(x, y))
        return (bearing + 360) % 360


def test_pharlap_interface():
    """Test the PHaRLAP interface."""
    from .electron_density import create_test_profile

    print("PHaRLAP Interface Test")
    print("=" * 50)

    # Create model and interface
    model = create_test_profile()
    model.update_from_realtime(foF2=10.0, hmF2=300.0)

    rt = PHaRLAPInterface(model)

    # Test single ray trace
    print("\n1. Single Ray Trace (7 MHz, 60° elevation, North):")
    result = rt.trace_ray(
        frequency=7.0,
        elevation=60.0,
        azimuth=0.0,
        tx_lat=40.0,
        tx_lon=-105.0,
    )
    print(f"   Success: {result.success}")
    print(f"   Ground range: {result.ground_range:.1f} km")
    print(f"   Max altitude: {result.max_altitude:.1f} km")
    print(f"   Path length: {result.path_length:.1f} km")

    # Test NVIS frequencies
    print("\n2. NVIS Frequencies (short range, high angles):")
    nvis = rt.find_nvis_frequencies(
        tx_lat=40.0, tx_lon=-105.0,
        rx_lat=40.5, rx_lon=-104.5,  # ~70 km away
        freq_min=3.0, freq_max=9.0,
    )
    if nvis:
        print(f"   Found {len(nvis)} working frequencies")
        for freq, el, rng in nvis[:5]:
            print(f"   {freq:.1f} MHz at {el:.0f}° → {rng:.0f} km")
    else:
        print("   No NVIS frequencies found")

    # Test MUF calculation
    print("\n3. MUF Calculation (Boulder to Denver, ~40 km):")
    muf_result = rt.find_muf(
        tx_lat=40.0, tx_lon=-105.3,  # Boulder
        rx_lat=39.7, rx_lon=-104.9,  # Denver
    )
    print(f"   MUF: {muf_result.muf:.1f} MHz")
    print(f"   FOT: {muf_result.fot:.1f} MHz")
    print(f"   Best elevation: {muf_result.best_elevation:.0f}°")

    # Test ionogram generation
    print("\n4. Synthetic Ionogram (North, 2-10 MHz):")
    ionogram = rt.generate_ionogram(
        tx_lat=40.0, tx_lon=-105.0,
        azimuth=0.0,
        freq_range=(2.0, 10.0),
        freq_step=1.0,
        elevation_range=(30.0, 85.0),
        elevation_step=10.0,
    )
    print(f"   Generated {len(ionogram)} points")
    if ionogram:
        print("   Sample points:")
        for pt in ionogram[:5]:
            print(f"   {pt.frequency:.1f} MHz: vh={pt.virtual_height:.0f} km, "
                  f"el={pt.elevation:.0f}°")


if __name__ == "__main__":
    test_pharlap_interface()
