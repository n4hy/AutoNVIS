"""
IONORT-Style Homing Algorithm

Implements the winner triplet search algorithm for finding HF propagation paths
that connect a transmitter to a receiver. A "winner triplet" is a combination of
(frequency, elevation, azimuth) that successfully propagates from Tx to Rx.

This follows the IONORT methodology described in:
    "IONORT: 3D Ray Tracing for HF Radio Propagation"
    Remote Sensing 2023, 15(21), 5111

Key Features:
- Systematic search over frequency, elevation, and azimuth deviation
- Landing accuracy check (IONORT Condition 10)
- Parallel ray tracing with ThreadPoolExecutor
- MUF/LUF/FOT calculation from winner triplets

The algorithm:
1. Calculate great circle geometry (range, azimuth) from Tx to Rx
2. Generate search grid of (frequency, elevation, azimuth_deviation) triplets
3. Trace rays in parallel using ThreadPoolExecutor
4. Check if each ray lands within tolerance of Rx (Condition 10)
5. Collect winner triplets and compute propagation statistics

Reference: IONORT paper Section 4 (Homing Algorithm)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing
import time
import logging

from .haselgrove import HaselgroveSolver, RayPath, RayMode, RayTermination, RayState

# Re-export for worker function
__all__ = ['HomingAlgorithm', 'HomingConfig', 'HomingSearchSpace', 'HomingResult',
           'WinnerTriplet', 'PropagationMode']
from .electron_density import IonosphericModel

logger = logging.getLogger(__name__)


class PropagationMode(Enum):
    """Magnetoionic propagation mode."""
    O_MODE = "O"  # Ordinary ray
    X_MODE = "X"  # Extraordinary ray


@dataclass
class WinnerTriplet:
    """
    A successful propagation path connecting Tx to Rx.

    This represents a (frequency, elevation, azimuth) combination that
    results in a ray landing within tolerance of the receiver.

    Attributes:
        frequency_mhz: Operating frequency (MHz)
        elevation_deg: Launch elevation angle (degrees)
        azimuth_deg: Launch azimuth (degrees clockwise from north)
        azimuth_deviation_deg: Deviation from great circle azimuth (degrees)
        group_delay_ms: One-way group delay (milliseconds)
        ground_range_km: Great circle distance traveled (km)
        landing_lat: Latitude where ray landed (degrees)
        landing_lon: Longitude where ray landed (degrees)
        landing_error_km: Distance from Rx to landing point (km)
        mode: Propagation mode (O or X)
        ray_path: Full ray path data (optional, for visualization)
        reflection_height_km: Approximate reflection altitude (km)
        hop_count: Number of ionospheric hops (1 = single hop, 2 = two hops)
        snr_db: Predicted signal-to-noise ratio (dB, None if not calculated)
        signal_strength_dbm: Predicted signal at receiver (dBm)
        path_loss_db: Total path loss (dB)
    """
    frequency_mhz: float
    elevation_deg: float
    azimuth_deg: float
    azimuth_deviation_deg: float
    group_delay_ms: float
    ground_range_km: float
    landing_lat: float
    landing_lon: float
    landing_error_km: float
    mode: PropagationMode
    ray_path: Optional[RayPath] = None
    reflection_height_km: float = 0.0
    hop_count: int = 1
    snr_db: Optional[float] = None
    signal_strength_dbm: Optional[float] = None
    path_loss_db: Optional[float] = None

    def __repr__(self) -> str:
        snr_str = f", SNR={self.snr_db:.0f}dB" if self.snr_db is not None else ""
        return (f"WinnerTriplet(f={self.frequency_mhz:.1f}MHz, "
                f"el={self.elevation_deg:.1f}°, "
                f"az={self.azimuth_deg:.1f}°, "
                f"err={self.landing_error_km:.1f}km, "
                f"hops={self.hop_count}, "
                f"mode={self.mode.value}{snr_str})")


@dataclass
class HomingSearchSpace:
    """
    Search space parameters for homing algorithm.

    Defines the range of frequencies, elevations, and azimuth deviations
    to search for winner triplets.

    Attributes:
        freq_range: (min_freq, max_freq) in MHz
        freq_step: Frequency step size (MHz)
        elevation_range: (min_elevation, max_elevation) in degrees
        elevation_step: Elevation step size (degrees)
        azimuth_deviation_range: (min_dev, max_dev) deviation from great circle (degrees)
        azimuth_step: Azimuth deviation step (degrees)
    """
    freq_range: Tuple[float, float] = (2.0, 30.0)
    freq_step: float = 0.5
    elevation_range: Tuple[float, float] = (5.0, 89.0)
    elevation_step: float = 2.0
    azimuth_deviation_range: Tuple[float, float] = (-15.0, 15.0)
    azimuth_step: float = 5.0

    @property
    def num_frequencies(self) -> int:
        """Number of frequencies in search grid."""
        return int((self.freq_range[1] - self.freq_range[0]) / self.freq_step) + 1

    @property
    def num_elevations(self) -> int:
        """Number of elevations in search grid."""
        return int((self.elevation_range[1] - self.elevation_range[0]) / self.elevation_step) + 1

    @property
    def num_azimuths(self) -> int:
        """Number of azimuth deviations in search grid."""
        return int((self.azimuth_deviation_range[1] - self.azimuth_deviation_range[0]) / self.azimuth_step) + 1

    @property
    def total_triplets(self) -> int:
        """Total number of triplets to evaluate."""
        return self.num_frequencies * self.num_elevations * self.num_azimuths

    def frequencies(self) -> np.ndarray:
        """Generate frequency array."""
        return np.arange(self.freq_range[0], self.freq_range[1] + self.freq_step/2, self.freq_step)

    def elevations(self) -> np.ndarray:
        """Generate elevation array."""
        return np.arange(self.elevation_range[0], self.elevation_range[1] + self.elevation_step/2, self.elevation_step)

    def azimuth_deviations(self) -> np.ndarray:
        """Generate azimuth deviation array."""
        return np.arange(
            self.azimuth_deviation_range[0],
            self.azimuth_deviation_range[1] + self.azimuth_step/2,
            self.azimuth_step
        )


@dataclass
class HomingConfig:
    """
    Configuration for homing algorithm.

    Attributes:
        lat_tolerance_deg: Maximum latitude error for winner (degrees)
        lon_tolerance_deg: Maximum longitude error for winner (degrees)
        distance_tolerance_km: Alternative: maximum distance error (km)
        use_distance_tolerance: If True, use distance instead of lat/lon
        trace_both_modes: If True, trace both O-mode and X-mode
        store_ray_paths: If True, store full ray paths in winners (memory intensive)
        max_workers: Number of parallel workers for ray tracing
        step_km: Ray integration step size in km (larger=faster, less accurate)
        use_multiprocessing: If True, use ProcessPoolExecutor (true parallelism)
        max_hops: Maximum ground reflections for multi-hop paths (default 3)
    """
    lat_tolerance_deg: float = 1.0
    lon_tolerance_deg: float = 1.0
    distance_tolerance_km: float = 100.0
    use_distance_tolerance: bool = True
    trace_both_modes: bool = True
    store_ray_paths: bool = False
    max_workers: int = 4
    step_km: float = 1.0  # Integration step size
    use_multiprocessing: bool = True  # Use processes instead of threads
    max_hops: int = 3  # Maximum ground reflections for multi-hop


# Global worker state for multiprocessing (each process gets its own copy)
_worker_solver = None
_worker_config = None


def _init_worker(ionosphere_params: dict, integrator_name: str, config_dict: dict):
    """Initialize worker process with its own solver instance."""
    global _worker_solver, _worker_config
    from .electron_density import IonosphericModel
    from .haselgrove import HaselgroveSolver

    # Recreate ionosphere model in this process
    ionosphere = IonosphericModel()
    ionosphere.update_from_realtime(
        foF2=ionosphere_params.get('foF2', 7.0),
        hmF2=ionosphere_params.get('hmF2', 300.0),
    )

    # Create solver for this process
    _worker_solver = HaselgroveSolver(ionosphere, integrator_name=integrator_name)
    _worker_config = config_dict


def _trace_ray_worker(args: tuple) -> Optional[WinnerTriplet]:
    """Worker function for parallel ray tracing (runs in separate process)."""
    global _worker_solver, _worker_config

    (freq, elev, azimuth, mode_str,
     tx_lat, tx_lon, tx_alt,
     rx_lat, rx_lon, gc_azimuth) = args

    mode = PropagationMode.O_MODE if mode_str == "O" else PropagationMode.X_MODE
    ray_mode = RayMode.ORDINARY if mode == PropagationMode.O_MODE else RayMode.EXTRAORDINARY

    try:
        # Trace the ray with multi-hop support
        path = _worker_solver.trace_ray(
            tx_lat=tx_lat,
            tx_lon=tx_lon,
            tx_alt=tx_alt,
            elevation=elev,
            azimuth=azimuth,
            frequency_mhz=freq,
            mode=ray_mode,
            step_km=_worker_config.get('step_km', 1.0),
            max_hops=_worker_config.get('max_hops', 3),
        )

        # Check termination
        if path.termination != RayTermination.GROUND_HIT:
            return None

        if path.landing_position is None:
            return None

        land_lat, land_lon, land_alt = path.landing_position

        # Calculate distance error
        lat1_r = np.radians(land_lat)
        lat2_r = np.radians(rx_lat)
        lon1_r = np.radians(land_lon)
        lon2_r = np.radians(rx_lon)

        dlat = lat2_r - lat1_r
        dlon = lon2_r - lon1_r

        a = np.sin(dlat/2)**2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        distance_error = 6371.0 * c

        # Check tolerance
        tolerance = _worker_config.get('distance_tolerance_km', 100.0)
        if distance_error > tolerance:
            return None

        # Calculate azimuth deviation
        az_deviation = azimuth - gc_azimuth
        if az_deviation > 180:
            az_deviation -= 360
        elif az_deviation < -180:
            az_deviation += 360

        # Group delay
        group_delay_ms = path.group_path_length / 299.792458

        # Reflection height
        reflection_height = max(s.altitude() for s in path.states) if path.states else 0.0

        return WinnerTriplet(
            frequency_mhz=freq,
            elevation_deg=elev,
            azimuth_deg=azimuth,
            azimuth_deviation_deg=az_deviation,
            group_delay_ms=group_delay_ms,
            ground_range_km=path.ground_range,
            landing_lat=land_lat,
            landing_lon=land_lon,
            landing_error_km=distance_error,
            mode=mode,
            ray_path=None,  # Don't store paths in multiprocessing (not picklable)
            reflection_height_km=reflection_height,
            hop_count=path.hop_count,  # Get hop count from traced path
        )

    except Exception as e:
        return None


@dataclass
class HomingResult:
    """
    Result of homing algorithm execution.

    Attributes:
        tx_position: Transmitter (lat, lon, alt) in degrees/km
        rx_position: Receiver (lat, lon, alt) in degrees/km
        great_circle_range_km: Direct great circle distance (km)
        great_circle_azimuth_deg: Azimuth from Tx to Rx (degrees)
        winner_triplets: List of successful propagation paths
        muf: Maximum Usable Frequency (MHz)
        luf: Lowest Usable Frequency (MHz)
        fot: Frequency of Optimum Traffic (MHz)
        total_rays_traced: Number of rays traced
        computation_time_s: Total computation time (seconds)
        search_space: Search parameters used
    """
    tx_position: Tuple[float, float, float]
    rx_position: Tuple[float, float, float]
    great_circle_range_km: float
    great_circle_azimuth_deg: float
    winner_triplets: List[WinnerTriplet] = field(default_factory=list)
    muf: float = 0.0
    luf: float = 0.0
    fot: float = 0.0
    total_rays_traced: int = 0
    computation_time_s: float = 0.0
    search_space: Optional[HomingSearchSpace] = None

    @property
    def num_winners(self) -> int:
        """Number of winner triplets found."""
        return len(self.winner_triplets)

    @property
    def o_mode_winners(self) -> List[WinnerTriplet]:
        """Winner triplets for O-mode only."""
        return [w for w in self.winner_triplets if w.mode == PropagationMode.O_MODE]

    @property
    def x_mode_winners(self) -> List[WinnerTriplet]:
        """Winner triplets for X-mode only."""
        return [w for w in self.winner_triplets if w.mode == PropagationMode.X_MODE]

    def frequencies_by_mode(self, mode: PropagationMode) -> List[float]:
        """Get unique frequencies for a mode."""
        return sorted(set(w.frequency_mhz for w in self.winner_triplets if w.mode == mode))

    def __repr__(self) -> str:
        return (f"HomingResult(range={self.great_circle_range_km:.0f}km, "
                f"winners={self.num_winners}, "
                f"MUF={self.muf:.1f}MHz, "
                f"time={self.computation_time_s:.1f}s)")


class HomingAlgorithm:
    """
    IONORT-style homing algorithm for HF path finding.

    Systematically searches for (frequency, elevation, azimuth) combinations
    that successfully propagate from transmitter to receiver.

    Example:
        from raytracer import IonosphericModel, HaselgroveSolver
        from raytracer.homing_algorithm import HomingAlgorithm, HomingSearchSpace

        # Setup
        ionosphere = IonosphericModel(...)
        solver = HaselgroveSolver(ionosphere)
        homing = HomingAlgorithm(solver)

        # Define search space
        search = HomingSearchSpace(
            freq_range=(3.0, 15.0),
            freq_step=0.5,
            elevation_range=(10.0, 80.0),
            elevation_step=5.0,
        )

        # Find paths
        result = homing.find_paths(
            tx_lat=40.0, tx_lon=-105.0,
            rx_lat=42.0, rx_lon=-100.0,
            search_space=search
        )

        print(f"Found {result.num_winners} winner triplets")
        print(f"MUF: {result.muf:.1f} MHz")
        for w in result.winner_triplets[:5]:
            print(f"  {w.frequency_mhz:.1f} MHz at {w.elevation_deg:.0f}° elevation")

        # Cancellation support:
        homing.cancel()  # Call from another thread to stop
    """

    # Earth radius for great circle calculations
    EARTH_RADIUS_KM = 6371.0

    def __init__(
        self,
        solver: HaselgroveSolver,
        config: Optional[HomingConfig] = None,
    ):
        """
        Initialize homing algorithm.

        Args:
            solver: Configured HaselgroveSolver instance
            config: Homing configuration (uses defaults if None)
        """
        self.solver = solver
        self.config = config or HomingConfig()
        self._cancelled = False
        self._executor = None  # Track executor for cleanup
        self._active_processes = []  # Track spawned processes

    def cancel(self):
        """Cancel the running homing algorithm."""
        self._cancelled = True
        logger.info("Homing algorithm cancellation requested")
        self._force_shutdown()

    def _force_shutdown(self):
        """Force shutdown of any running executor/processes."""
        if self._executor is not None:
            try:
                # First try graceful shutdown with short timeout
                self._executor.shutdown(wait=True, cancel_futures=True)
            except Exception as e:
                logger.debug(f"Executor shutdown: {e}")
            self._executor = None

        # Terminate any tracked processes
        for proc in self._active_processes:
            try:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=0.5)
                    if proc.is_alive():
                        proc.kill()
            except Exception:
                pass
        self._active_processes.clear()

    def is_cancelled(self) -> bool:
        """Check if cancellation was requested."""
        return self._cancelled

    def reset(self):
        """Reset cancellation flag for a new run."""
        self._cancelled = False

    def cleanup(self):
        """Clean up all resources. Call this before destroying the algorithm."""
        self.cancel()
        self._force_shutdown()

    def find_paths(
        self,
        tx_lat: float,
        tx_lon: float,
        rx_lat: float,
        rx_lon: float,
        tx_alt: float = 0.0,
        rx_alt: float = 0.0,
        search_space: Optional[HomingSearchSpace] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> HomingResult:
        """
        Find all propagation paths connecting Tx to Rx.

        Systematically traces rays over the search space and collects
        winner triplets (paths that land within tolerance of Rx).

        Args:
            tx_lat: Transmitter latitude (degrees)
            tx_lon: Transmitter longitude (degrees)
            rx_lat: Receiver latitude (degrees)
            rx_lon: Receiver longitude (degrees)
            tx_alt: Transmitter altitude (km, default 0)
            rx_alt: Receiver altitude (km, default 0)
            search_space: Search parameters (uses defaults if None)
            progress_callback: Optional callback(rays_done, total_rays) for progress

        Returns:
            HomingResult with all winner triplets and statistics
        """
        # Reset cancellation flag for new run
        self._cancelled = False

        start_time = time.time()
        search_space = search_space or HomingSearchSpace()

        # Calculate great circle geometry
        gc_range, gc_azimuth = self._great_circle_geometry(
            tx_lat, tx_lon, rx_lat, rx_lon
        )

        logger.info(f"Homing: {tx_lat:.2f}°, {tx_lon:.2f}° -> {rx_lat:.2f}°, {rx_lon:.2f}°")
        logger.info(f"Great circle: {gc_range:.0f} km at {gc_azimuth:.1f}°")
        logger.info(f"Search space: {search_space.total_triplets} triplets")

        # Initialize result
        result = HomingResult(
            tx_position=(tx_lat, tx_lon, tx_alt),
            rx_position=(rx_lat, rx_lon, rx_alt),
            great_circle_range_km=gc_range,
            great_circle_azimuth_deg=gc_azimuth,
            search_space=search_space,
        )

        # Generate triplet list
        triplets = self._generate_triplets(search_space, gc_azimuth)

        # Trace rays (parallel or sequential based on config)
        if self.config.max_workers > 1:
            winners = self._trace_parallel(
                triplets, tx_lat, tx_lon, tx_alt,
                rx_lat, rx_lon, gc_range,
                progress_callback
            )
        else:
            winners = self._trace_sequential(
                triplets, tx_lat, tx_lon, tx_alt,
                rx_lat, rx_lon, gc_range,
                progress_callback
            )

        result.winner_triplets = winners
        result.total_rays_traced = len(triplets)

        # Calculate MUF/LUF/FOT
        self._calculate_frequencies(result)

        result.computation_time_s = time.time() - start_time

        logger.info(f"Homing complete: {result.num_winners} winners in {result.computation_time_s:.1f}s")

        return result

    def _great_circle_geometry(
        self,
        lat1: float, lon1: float,
        lat2: float, lon2: float
    ) -> Tuple[float, float]:
        """
        Calculate great circle distance and initial bearing.

        Args:
            lat1, lon1: Start point (degrees)
            lat2, lon2: End point (degrees)

        Returns:
            (distance_km, azimuth_deg)
        """
        lat1_r = np.radians(lat1)
        lat2_r = np.radians(lat2)
        lon1_r = np.radians(lon1)
        lon2_r = np.radians(lon2)

        dlat = lat2_r - lat1_r
        dlon = lon2_r - lon1_r

        # Haversine formula for distance
        a = np.sin(dlat/2)**2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        distance = self.EARTH_RADIUS_KM * c

        # Initial bearing (forward azimuth)
        y = np.sin(dlon) * np.cos(lat2_r)
        x = np.cos(lat1_r) * np.sin(lat2_r) - np.sin(lat1_r) * np.cos(lat2_r) * np.cos(dlon)
        bearing = np.degrees(np.arctan2(y, x))

        # Normalize to 0-360
        bearing = (bearing + 360) % 360

        return distance, bearing

    def _generate_triplets(
        self,
        search_space: HomingSearchSpace,
        gc_azimuth: float
    ) -> List[Tuple[float, float, float, PropagationMode]]:
        """
        Generate list of (freq, elevation, azimuth, mode) triplets to trace.

        Args:
            search_space: Search parameters
            gc_azimuth: Great circle azimuth (degrees)

        Returns:
            List of (frequency, elevation, azimuth, mode) tuples
        """
        triplets = []

        for freq in search_space.frequencies():
            for elev in search_space.elevations():
                for az_dev in search_space.azimuth_deviations():
                    azimuth = (gc_azimuth + az_dev) % 360

                    # O-mode
                    triplets.append((freq, elev, azimuth, PropagationMode.O_MODE))

                    # X-mode (if configured)
                    if self.config.trace_both_modes:
                        triplets.append((freq, elev, azimuth, PropagationMode.X_MODE))

        return triplets

    def _trace_sequential(
        self,
        triplets: List[Tuple[float, float, float, PropagationMode]],
        tx_lat: float, tx_lon: float, tx_alt: float,
        rx_lat: float, rx_lon: float, gc_range: float,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[WinnerTriplet]:
        """Trace rays sequentially."""
        winners = []
        total = len(triplets)

        for i, (freq, elev, azimuth, mode) in enumerate(triplets):
            # Check for cancellation
            if self._cancelled:
                logger.info(f"Homing cancelled at {i}/{total} rays")
                break

            winner = self._trace_and_check(
                freq, elev, azimuth, mode,
                tx_lat, tx_lon, tx_alt,
                rx_lat, rx_lon, gc_range
            )
            if winner is not None:
                winners.append(winner)

            if progress_callback and ((i + 1) % 10 == 0 or i + 1 == total):
                progress_callback(i + 1, total)

        # Final progress call
        if progress_callback:
            progress_callback(total, total)

        return winners

    def _trace_parallel(
        self,
        triplets: List[Tuple[float, float, float, PropagationMode]],
        tx_lat: float, tx_lon: float, tx_alt: float,
        rx_lat: float, rx_lon: float, gc_range: float,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[WinnerTriplet]:
        """Trace rays in parallel using ProcessPoolExecutor or ThreadPoolExecutor."""
        winners = []
        total = len(triplets)
        completed = 0

        # Get great circle azimuth for workers
        gc_azimuth = self._great_circle_geometry(tx_lat, tx_lon, rx_lat, rx_lon)[1]

        if self.config.use_multiprocessing:
            # Use ProcessPoolExecutor for true parallelism (bypasses GIL)
            winners = self._trace_multiprocess(
                triplets, tx_lat, tx_lon, tx_alt,
                rx_lat, rx_lon, gc_azimuth,
                progress_callback
            )
        else:
            # Fall back to ThreadPoolExecutor
            winners = self._trace_threaded(
                triplets, tx_lat, tx_lon, tx_alt,
                rx_lat, rx_lon, gc_range,
                progress_callback
            )

        # If multiprocessing was used and we need ray paths, re-trace winners
        if self.config.use_multiprocessing and self.config.store_ray_paths and winners:
            winners = self._retrace_winners_for_paths(
                winners, tx_lat, tx_lon, tx_alt, gc_azimuth
            )

        return winners

    def _retrace_winners_for_paths(
        self,
        winners: List[WinnerTriplet],
        tx_lat: float, tx_lon: float, tx_alt: float,
        gc_azimuth: float,
    ) -> List[WinnerTriplet]:
        """Re-trace winner rays in main process to get full paths for visualization."""
        logger.info(f"Re-tracing {len(winners)} winners for visualization paths...")
        updated_winners = []
        success_count = 0
        fail_count = 0

        for w in winners:
            ray_mode = RayMode.ORDINARY if w.mode == PropagationMode.O_MODE else RayMode.EXTRAORDINARY
            try:
                path = self.solver.trace_ray(
                    tx_lat=tx_lat,
                    tx_lon=tx_lon,
                    tx_alt=tx_alt,
                    elevation=w.elevation_deg,
                    azimuth=w.azimuth_deg,
                    frequency_mhz=w.frequency_mhz,
                    mode=ray_mode,
                    step_km=self.config.step_km,
                    max_hops=self.config.max_hops,
                )

                # Verify the path has states
                if path.states and len(path.states) > 1:
                    # Create new winner with ray_path attached
                    updated_winner = WinnerTriplet(
                        frequency_mhz=w.frequency_mhz,
                        elevation_deg=w.elevation_deg,
                        azimuth_deg=w.azimuth_deg,
                        azimuth_deviation_deg=w.azimuth_deviation_deg,
                        group_delay_ms=w.group_delay_ms,
                        ground_range_km=w.ground_range_km,
                        landing_lat=w.landing_lat,
                        landing_lon=w.landing_lon,
                        landing_error_km=w.landing_error_km,
                        mode=w.mode,
                        ray_path=path,  # Now has the full path
                        reflection_height_km=w.reflection_height_km,
                        hop_count=w.hop_count,
                        snr_db=w.snr_db,
                        signal_strength_dbm=w.signal_strength_dbm,
                        path_loss_db=w.path_loss_db,
                    )
                    updated_winners.append(updated_winner)
                    success_count += 1
                    logger.debug(f"Re-traced {w.frequency_mhz:.1f} MHz: {len(path.states)} states")
                else:
                    logger.warning(f"Re-trace {w.frequency_mhz:.1f} MHz: empty path")
                    updated_winners.append(w)
                    fail_count += 1
            except Exception as e:
                logger.warning(f"Failed to re-trace {w.frequency_mhz:.1f} MHz winner: {e}")
                updated_winners.append(w)  # Keep original without path
                fail_count += 1

        logger.info(f"Re-traced {success_count}/{len(winners)} winners with paths ({fail_count} failed)")
        return updated_winners

    def _trace_multiprocess(
        self,
        triplets: List[Tuple[float, float, float, PropagationMode]],
        tx_lat: float, tx_lon: float, tx_alt: float,
        rx_lat: float, rx_lon: float, gc_azimuth: float,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[WinnerTriplet]:
        """Trace rays using ProcessPoolExecutor for true parallelism."""
        winners = []
        total = len(triplets)
        completed = 0

        # Prepare ionosphere parameters for workers
        iono_params = {
            'foF2': self.solver.ionosphere._default_profile.foF2,
            'hmF2': self.solver.ionosphere._default_profile.hmF2,
        }

        # Prepare config for workers
        config_dict = {
            'step_km': self.config.step_km,
            'distance_tolerance_km': self.config.distance_tolerance_km,
            'store_ray_paths': False,  # Can't pickle ray paths
            'max_hops': self.config.max_hops,
        }

        # Prepare work items (must be picklable)
        work_items = [
            (freq, elev, azimuth, mode.value,  # mode.value is "O" or "X" string
             tx_lat, tx_lon, tx_alt,
             rx_lat, rx_lon, gc_azimuth)
            for freq, elev, azimuth, mode in triplets
        ]

        # Get integrator name from solver
        integrator_name = getattr(self.solver, '_integrator_name', 'rk4')

        logger.info(f"Starting multiprocess ray tracing with {self.config.max_workers} workers")

        # Use spawn context for clean process isolation
        mp_context = multiprocessing.get_context('spawn')

        try:
            self._executor = ProcessPoolExecutor(
                max_workers=self.config.max_workers,
                mp_context=mp_context,
                initializer=_init_worker,
                initargs=(iono_params, integrator_name, config_dict)
            )

            try:
                # Submit all work
                futures = {self._executor.submit(_trace_ray_worker, item): item for item in work_items}

                # Collect results
                for future in as_completed(futures):
                    if self._cancelled:
                        logger.info(f"Homing cancelled at {completed}/{total} rays")
                        break

                    completed += 1
                    try:
                        winner = future.result(timeout=30)  # Timeout per ray
                        if winner is not None:
                            winners.append(winner)
                    except Exception as e:
                        logger.debug(f"Ray trace failed: {e}")

                    if progress_callback and (completed % 10 == 0 or completed == total):
                        progress_callback(completed, total)

            finally:
                # Always clean up executor properly to avoid semaphore leaks
                try:
                    self._executor.shutdown(wait=True, cancel_futures=True)
                except Exception:
                    pass
                self._executor = None

        except Exception as e:
            logger.warning(f"Multiprocessing failed, falling back to threads: {e}")
            self._executor = None
            # Fall back to threaded execution
            return self._trace_threaded(
                triplets, tx_lat, tx_lon, tx_alt,
                rx_lat, rx_lon, 0, progress_callback
            )

        if progress_callback:
            progress_callback(completed, total)

        return winners

    def _trace_threaded(
        self,
        triplets: List[Tuple[float, float, float, PropagationMode]],
        tx_lat: float, tx_lon: float, tx_alt: float,
        rx_lat: float, rx_lon: float, gc_range: float,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[WinnerTriplet]:
        """Trace rays using ThreadPoolExecutor (GIL-limited but works as fallback)."""
        winners = []
        total = len(triplets)
        completed = 0

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(
                    self._trace_and_check,
                    freq, elev, azimuth, mode,
                    tx_lat, tx_lon, tx_alt,
                    rx_lat, rx_lon, gc_range
                ): (freq, elev, azimuth, mode)
                for freq, elev, azimuth, mode in triplets
            }

            # Collect results as they complete
            for future in as_completed(futures):
                # Check for cancellation
                if self._cancelled:
                    logger.info(f"Homing cancelled at {completed}/{total} rays - cancelling pending tasks")
                    # Cancel remaining futures
                    for f in futures:
                        f.cancel()
                    break

                completed += 1
                try:
                    winner = future.result()
                    if winner is not None:
                        winners.append(winner)
                except Exception as e:
                    logger.warning(f"Ray trace failed: {e}")

                if progress_callback and (completed % 10 == 0 or completed == total):
                    progress_callback(completed, total)

        # Final progress call (even if cancelled, report where we got to)
        if progress_callback:
            progress_callback(completed, total)

        return winners

    def _trace_and_check(
        self,
        freq: float,
        elev: float,
        azimuth: float,
        mode: PropagationMode,
        tx_lat: float, tx_lon: float, tx_alt: float,
        rx_lat: float, rx_lon: float, gc_range: float,
    ) -> Optional[WinnerTriplet]:
        """
        Trace a single ray and check if it's a winner.

        Args:
            freq, elev, azimuth, mode: Ray parameters
            tx_lat, tx_lon, tx_alt: Transmitter position
            rx_lat, rx_lon: Receiver position
            gc_range: Great circle range (km)

        Returns:
            WinnerTriplet if ray lands within tolerance, else None
        """
        # Map mode to RayMode
        ray_mode = RayMode.ORDINARY if mode == PropagationMode.O_MODE else RayMode.EXTRAORDINARY

        try:
            # Trace the ray with multi-hop support
            path = self.solver.trace_ray(
                tx_lat=tx_lat,
                tx_lon=tx_lon,
                tx_alt=tx_alt,
                elevation=elev,
                azimuth=azimuth,
                frequency_mhz=freq,
                mode=ray_mode,
                step_km=self.config.step_km,
                max_hops=self.config.max_hops,
            )

            # Check termination - only ground hits are potential winners
            if path.termination != RayTermination.GROUND_HIT:
                return None

            # Get landing position
            if path.landing_position is None:
                return None

            land_lat, land_lon, land_alt = path.landing_position

            # Check landing accuracy (IONORT Condition 10)
            is_winner, landing_error = self._check_landing_accuracy(
                land_lat, land_lon, rx_lat, rx_lon
            )

            if not is_winner:
                return None

            # Calculate additional metrics
            gc_azimuth = self._great_circle_geometry(tx_lat, tx_lon, rx_lat, rx_lon)[1]
            az_deviation = azimuth - gc_azimuth
            if az_deviation > 180:
                az_deviation -= 360
            elif az_deviation < -180:
                az_deviation += 360

            # Group delay (approximate from path length)
            group_delay_ms = path.group_path_length / 299.792458  # c in km/ms

            # Find reflection height (maximum altitude reached)
            reflection_height = max(s.altitude() for s in path.states) if path.states else 0.0

            # Create winner triplet
            winner = WinnerTriplet(
                frequency_mhz=freq,
                elevation_deg=elev,
                azimuth_deg=azimuth,
                azimuth_deviation_deg=az_deviation,
                group_delay_ms=group_delay_ms,
                ground_range_km=path.ground_range,
                landing_lat=land_lat,
                landing_lon=land_lon,
                landing_error_km=landing_error,
                mode=mode,
                ray_path=path if self.config.store_ray_paths else None,
                reflection_height_km=reflection_height,
                hop_count=path.hop_count,  # Multi-hop count from traced path
            )

            return winner

        except Exception as e:
            logger.debug(f"Ray trace error at f={freq}, el={elev}: {e}")
            return None

    def _check_landing_accuracy(
        self,
        land_lat: float, land_lon: float,
        rx_lat: float, rx_lon: float,
    ) -> Tuple[bool, float]:
        """
        Check if landing position is within tolerance of receiver.

        Implements IONORT Condition (10):
            |λ_ray - λ_rx| <= Latitude_Accuracy
            |φ_ray - φ_rx| <= Longitude_Accuracy

        Or alternatively, distance-based tolerance.

        Args:
            land_lat, land_lon: Where ray landed
            rx_lat, rx_lon: Receiver position

        Returns:
            (is_winner, distance_error_km)
        """
        # Calculate distance error
        distance_error, _ = self._great_circle_geometry(
            land_lat, land_lon, rx_lat, rx_lon
        )

        if self.config.use_distance_tolerance:
            is_winner = distance_error <= self.config.distance_tolerance_km
        else:
            lat_error = abs(land_lat - rx_lat)
            lon_error = abs(land_lon - rx_lon)
            is_winner = (lat_error <= self.config.lat_tolerance_deg and
                        lon_error <= self.config.lon_tolerance_deg)

        return is_winner, distance_error

    def _calculate_frequencies(self, result: HomingResult) -> None:
        """
        Calculate MUF, LUF, and FOT from winner triplets.

        MUF: Maximum Usable Frequency (highest successful frequency)
        LUF: Lowest Usable Frequency (lowest successful frequency)
        FOT: Frequency of Optimum Traffic (typically 0.85 * MUF)
        """
        if not result.winner_triplets:
            return

        frequencies = [w.frequency_mhz for w in result.winner_triplets]

        result.muf = max(frequencies)
        result.luf = min(frequencies)
        result.fot = result.muf * 0.85

    def find_optimal_nvis(
        self,
        tx_lat: float,
        tx_lon: float,
        radius_km: float = 200.0,
        freq_range: Tuple[float, float] = (2.0, 15.0),
        freq_step: float = 0.25,
    ) -> HomingResult:
        """
        Find optimal NVIS frequencies for coverage within a radius.

        NVIS (Near Vertical Incidence Skywave) uses high elevation angles
        for short-range communication (typically within 400 km).

        Args:
            tx_lat, tx_lon: Transmitter position
            radius_km: Coverage radius (km)
            freq_range: Frequency search range (MHz)
            freq_step: Frequency step (MHz)

        Returns:
            HomingResult with optimal frequencies
        """
        # For NVIS, we want high elevation angles (60-90 degrees)
        search_space = HomingSearchSpace(
            freq_range=freq_range,
            freq_step=freq_step,
            elevation_range=(60.0, 90.0),
            elevation_step=5.0,
            azimuth_deviation_range=(-5.0, 5.0),  # NVIS is omnidirectional
            azimuth_step=90.0,  # Check 4 directions
        )

        # Use a point at the edge of coverage as Rx
        # (bearing north - arbitrary since NVIS is omnidirectional)
        rx_lat = tx_lat + (radius_km / 111.0)  # ~111 km per degree
        rx_lon = tx_lon

        # Relax tolerance for NVIS
        config = HomingConfig(
            distance_tolerance_km=radius_km * 0.5,
            use_distance_tolerance=True,
            trace_both_modes=True,
            max_workers=self.config.max_workers,
        )

        saved_config = self.config
        self.config = config

        try:
            result = self.find_paths(
                tx_lat, tx_lon, rx_lat, rx_lon,
                search_space=search_space
            )
        finally:
            self.config = saved_config

        return result

    def calculate_snr_for_winners(
        self,
        result: 'HomingResult',
        tx_power_watts: float = 100.0,
        tx_antenna_gain_dbi: float = 0.0,
        rx_antenna_gain_dbi: float = 0.0,
        rx_bandwidth_hz: float = 3000.0,
        xray_flux: float = 1e-6,
        kp_index: float = 2.0,
        is_night: bool = False,
        noise_environment: str = "rural",
        min_snr_db: float = 0.0,
    ) -> 'HomingResult':
        """
        Calculate SNR for all winner triplets in a homing result.

        Args:
            result: HomingResult with winner triplets
            tx_power_watts: Transmitter power
            tx_antenna_gain_dbi: TX antenna gain in dBi
            rx_antenna_gain_dbi: RX antenna gain in dBi
            rx_bandwidth_hz: Receiver bandwidth (Hz)
            xray_flux: GOES X-ray flux (W/m²) - affects D-layer absorption
            kp_index: Geomagnetic Kp index (0-9)
            is_night: True if nighttime propagation
            noise_environment: One of 'quiet_rural', 'rural', 'residential', 'urban', 'industrial'
            min_snr_db: Minimum usable SNR threshold (-20 to 60 dB, default 0)
                        Paths below this are filtered out. Use -20 for FT8, 0 for CW, 10+ for voice.

        Returns:
            Updated HomingResult with SNR values populated (unusable paths filtered out)
        """
        from .link_budget import (
            LinkBudgetCalculator,
            TransmitterConfig,
            ReceiverConfig,
            AntennaConfig,
            NoiseEnvironment,
            calculate_solar_zenith_angle,
        )

        calc = LinkBudgetCalculator()

        # Map noise environment string to enum
        env_map = {
            'quiet_rural': NoiseEnvironment.QUIET_RURAL,
            'rural': NoiseEnvironment.RURAL,
            'residential': NoiseEnvironment.RESIDENTIAL,
            'urban': NoiseEnvironment.URBAN,
            'industrial': NoiseEnvironment.INDUSTRIAL,
        }
        noise_env = env_map.get(noise_environment.lower(), NoiseEnvironment.RURAL)

        # Configure TX/RX
        tx_config = TransmitterConfig(
            power_watts=tx_power_watts,
            antenna=AntennaConfig(gain_dbi=tx_antenna_gain_dbi),
        )
        rx_config = ReceiverConfig(
            antenna=AntennaConfig(gain_dbi=rx_antenna_gain_dbi),
            bandwidth_hz=rx_bandwidth_hz,
            noise_environment=noise_env,
        )

        # Calculate midpoint latitude for noise calculations
        mid_lat = (result.tx_position[0] + result.rx_position[0]) / 2

        # Calculate solar zenith angle at midpoint
        mid_lon = (result.tx_position[1] + result.rx_position[1]) / 2
        try:
            solar_zenith = calculate_solar_zenith_angle(mid_lat, mid_lon)
        except Exception:
            solar_zenith = 45.0  # Default

        # Calculate SNR for each winner and filter unusable paths
        # Use configurable threshold (default 0 dB, range -20 to 60)
        valid_winners = []

        for winner in result.winner_triplets:
            try:
                link_result = calc.calculate_for_winner(
                    winner,
                    tx_config=tx_config,
                    rx_config=rx_config,
                    xray_flux=xray_flux,
                    kp_index=kp_index,
                    solar_zenith_angle_deg=solar_zenith,
                    is_night=is_night,
                    latitude_deg=mid_lat,
                )

                snr = link_result.snr_db

                # Filter unusable paths based on user-configurable threshold
                if snr < min_snr_db:
                    logger.debug(f"Winner {winner.frequency_mhz:.1f}MHz rejected: "
                                f"SNR {snr:.0f}dB below {min_snr_db:.0f}dB cutoff")
                    continue

                winner.snr_db = snr
                winner.signal_strength_dbm = link_result.signal_power_dbw + 30
                winner.path_loss_db = link_result.total_path_loss_db
                valid_winners.append(winner)

            except Exception as e:
                logger.debug(f"SNR calculation failed for winner: {e}")
                # Don't include winners with failed SNR calculation
                continue

        # Replace winner list with only valid winners
        result.winner_triplets[:] = valid_winners

        return result
