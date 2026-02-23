"""
Haselgrove Ray Equations Solver

Implements 3D magnetoionic ray tracing using Haselgrove's equations -
six coupled first-order ODEs in Hamiltonian form for HF wave propagation
through the ionosphere.

The ray equations describe the evolution of:
    - Position: (x, y, z) in Earth-centered coordinates
    - Wave vector: (kx, ky, kz) related to ray direction

Key Physics:
    - Anisotropic medium: refractive index depends on direction
    - Magnetoionic splitting: O-mode and X-mode rays follow different paths
    - Earth curvature: uses spherical coordinates internally
    - Geomagnetic field: varies with position

Integration Methods:
    - RK4 (Runge-Kutta 4th order) - default, accurate
    - Adams-Bashforth/Adams-Moulton - higher order, faster for long paths

References:
    - Haselgrove (1957) "Oblique Ray Paths in the Ionosphere"
    - IONORT paper (remotesensing-15-05111-v2.pdf)
    - Jones & Stephenson (1975) "A Versatile 3D Ray Tracing Program"
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Callable
from datetime import datetime, timezone
from enum import Enum
import logging

from . import EARTH_RADIUS_KM, SPEED_OF_LIGHT
from .electron_density import IonosphericModel
from .integrators import BaseIntegrator, IntegrationStep, create_integrator

logger = logging.getLogger(__name__)


class RayMode(Enum):
    """Magnetoionic ray mode."""
    ORDINARY = "O"      # O-mode (ordinary ray)
    EXTRAORDINARY = "X"  # X-mode (extraordinary ray)


class RayTermination(Enum):
    """Reason for ray termination."""
    GROUND_HIT = "ground"           # Ray reached Earth surface
    ESCAPE = "escape"               # Ray escaped to space
    MAX_PATH = "max_path"           # Maximum path length reached
    REFLECTION = "reflection"       # Ray reflected back down
    ABSORPTION = "absorption"       # Ray absorbed (D-region)
    ERROR = "error"                 # Numerical error


@dataclass
class RayState:
    """
    Current state of a ray during tracing.

    Uses Earth-Centered Earth-Fixed (ECEF) coordinates internally
    with local ENU (East-North-Up) for convenience.

    Attributes:
        x, y, z: Position in ECEF (km from Earth center)
        kx, ky, kz: Wave vector components (normalized)
        path_length: Total path length traveled (km)
        group_path: Group path / virtual height (km)
        time: Propagation time (seconds)
    """
    # Position (km, ECEF)
    x: float
    y: float
    z: float

    # Wave vector (normalized, direction of phase propagation)
    kx: float
    ky: float
    kz: float

    # Path metrics
    path_length: float = 0.0
    group_path: float = 0.0
    time: float = 0.0

    # Ray properties
    mode: RayMode = RayMode.ORDINARY
    frequency_mhz: float = 10.0

    def position(self) -> np.ndarray:
        """Get position as numpy array."""
        return np.array([self.x, self.y, self.z])

    def wave_vector(self) -> np.ndarray:
        """Get wave vector as numpy array."""
        return np.array([self.kx, self.ky, self.kz])

    def altitude(self) -> float:
        """Calculate altitude above Earth surface (km)."""
        r = np.sqrt(self.x**2 + self.y**2 + self.z**2)
        return r - EARTH_RADIUS_KM

    def lat_lon_alt(self) -> Tuple[float, float, float]:
        """
        Convert ECEF to geodetic coordinates.

        Returns:
            (latitude_deg, longitude_deg, altitude_km)
        """
        r = np.sqrt(self.x**2 + self.y**2 + self.z**2)
        lat = np.degrees(np.arcsin(self.z / r)) if r > 0 else 0.0
        lon = np.degrees(np.arctan2(self.y, self.x))
        alt = r - EARTH_RADIUS_KM
        return (lat, lon, alt)

    def copy(self) -> 'RayState':
        """Create a copy of this state."""
        return RayState(
            x=self.x, y=self.y, z=self.z,
            kx=self.kx, ky=self.ky, kz=self.kz,
            path_length=self.path_length,
            group_path=self.group_path,
            time=self.time,
            mode=self.mode,
            frequency_mhz=self.frequency_mhz
        )


@dataclass
class RayPath:
    """
    Complete ray path from transmitter to termination.

    Attributes:
        states: List of RayState at each integration step
        start_position: Initial (lat, lon, alt) in degrees/km
        start_direction: Initial (elevation, azimuth) in degrees
        frequency_mhz: Transmission frequency
        mode: O-mode or X-mode
        termination: Why ray stopped
        landing_position: Final (lat, lon, alt) if ground hit
    """
    states: List[RayState] = field(default_factory=list)
    start_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    start_direction: Tuple[float, float] = (45.0, 0.0)  # elevation, azimuth
    frequency_mhz: float = 10.0
    mode: RayMode = RayMode.ORDINARY
    termination: RayTermination = RayTermination.MAX_PATH
    landing_position: Optional[Tuple[float, float, float]] = None

    @property
    def total_path_length(self) -> float:
        """Total geometric path length in km."""
        return self.states[-1].path_length if self.states else 0.0

    @property
    def group_path_length(self) -> float:
        """Total group path (virtual height) in km."""
        return self.states[-1].group_path if self.states else 0.0

    @property
    def ground_range(self) -> float:
        """Great circle distance from start to landing (km)."""
        if not self.landing_position:
            return 0.0
        # Simplified: use straight-line approximation for short paths
        lat1, lon1, _ = self.start_position
        lat2, lon2, _ = self.landing_position
        return self._haversine(lat1, lon1, lat2, lon2)

    def _haversine(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate great-circle distance in km."""
        lat1_r = np.radians(lat1)
        lat2_r = np.radians(lat2)
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)

        a = (np.sin(dlat/2)**2 +
             np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon/2)**2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

        return EARTH_RADIUS_KM * c


class HaselgroveSolver:
    """
    Haselgrove ray equations solver.

    Integrates the 6 coupled ODEs:
        dx/ds = (c/n) * kx_hat
        dk/ds = -∇(n) / n

    Where:
        s = path length
        n = refractive index (from Appleton-Hartree)
        k_hat = unit wave vector

    The solver handles:
        - Anisotropic refractive index
        - Earth curvature
        - Magnetoionic mode splitting
        - Variable geomagnetic field

    Example:
        solver = HaselgroveSolver(ionospheric_model)

        # Trace a ray from Boulder at 10 MHz, 30° elevation, bearing north
        path = solver.trace_ray(
            tx_lat=40.0,
            tx_lon=-105.0,
            tx_alt=0.0,
            elevation=30.0,
            azimuth=0.0,
            frequency_mhz=10.0,
            mode=RayMode.ORDINARY
        )

        print(f"Ray landed at: {path.landing_position}")
        print(f"Ground range: {path.ground_range:.1f} km")
    """

    # Integration parameters
    DEFAULT_STEP_KM = 1.0  # Integration step size
    MAX_PATH_LENGTH_KM = 5000.0  # Maximum ray path length
    MIN_ALTITUDE_KM = 0.0  # Ground level
    MAX_ALTITUDE_KM = 1000.0  # Escape altitude

    # Numerical parameters
    GRADIENT_DELTA_KM = 1.0  # For numerical gradient calculation

    def __init__(
        self,
        ionosphere: IonosphericModel,
        integrator: Optional[BaseIntegrator] = None,
        integrator_name: Optional[str] = None,
    ):
        """
        Initialize the Haselgrove solver.

        Args:
            ionosphere: IonosphericModel for refractive index calculation
            integrator: Pre-configured integrator instance, or None for default RK4
            integrator_name: Name of integrator to create ('rk4', 'rk45', 'adams_bashforth')
                           Ignored if integrator is provided.

        Example:
            # Default RK4 (backward compatible)
            solver = HaselgroveSolver(ionosphere)

            # Use adaptive RK45
            solver = HaselgroveSolver(ionosphere, integrator_name='rk45')

            # Use custom integrator
            from raytracer.integrators import RK45Integrator
            rk45 = RK45Integrator(solver._derivatives_array, tolerance=1e-8)
            solver = HaselgroveSolver(ionosphere, integrator=rk45)
        """
        self.ionosphere = ionosphere

        # Store integrator (create later when derivative func is available)
        self._integrator = integrator
        self._integrator_name = integrator_name

        # Flag to indicate if we should use pluggable integrator
        self._use_pluggable_integrator = (integrator is not None or integrator_name is not None)

    def _get_integrator(self) -> Optional[BaseIntegrator]:
        """Get or create the integrator instance."""
        if self._integrator is not None:
            return self._integrator

        if self._integrator_name is not None:
            self._integrator = create_integrator(
                self._integrator_name,
                self._derivatives_array,
                tolerance=1e-6,
                min_step=0.01,
                max_step=self.DEFAULT_STEP_KM,
            )
            return self._integrator

        return None

    def _derivatives_array(self, state_array: np.ndarray, freq_mhz: float) -> np.ndarray:
        """
        Compute Haselgrove derivatives in array form for integrators.

        This is the interface expected by the pluggable integrators.

        Args:
            state_array: [x, y, z, kx, ky, kz] as numpy array
            freq_mhz: Frequency in MHz

        Returns:
            [dx/ds, dy/ds, dz/ds, dkx/ds, dky/ds, dkz/ds] as numpy array
        """
        # Create temporary RayState for compatibility with existing methods
        temp_state = RayState(
            x=state_array[0],
            y=state_array[1],
            z=state_array[2],
            kx=state_array[3],
            ky=state_array[4],
            kz=state_array[5],
            frequency_mhz=freq_mhz,
            mode=self._current_mode,  # Set during trace_ray
        )

        # Use existing derivatives method
        derivs = self._derivatives(temp_state)
        return np.array(derivs)

    def trace_ray(
        self,
        tx_lat: float,
        tx_lon: float,
        tx_alt: float,
        elevation: float,
        azimuth: float,
        frequency_mhz: float,
        mode: RayMode = RayMode.ORDINARY,
        step_km: float = DEFAULT_STEP_KM,
        max_path_km: float = MAX_PATH_LENGTH_KM,
    ) -> RayPath:
        """
        Trace a single ray through the ionosphere.

        Args:
            tx_lat: Transmitter latitude (degrees)
            tx_lon: Transmitter longitude (degrees)
            tx_alt: Transmitter altitude (km above ground)
            elevation: Launch elevation angle (degrees above horizon)
            azimuth: Launch azimuth (degrees clockwise from north)
            frequency_mhz: Transmission frequency (MHz)
            mode: O-mode or X-mode
            step_km: Integration step size (km)
            max_path_km: Maximum path length before termination (km)

        Returns:
            RayPath with complete trajectory
        """
        # Initialize ray state
        state = self._initialize_ray(
            tx_lat, tx_lon, tx_alt,
            elevation, azimuth,
            frequency_mhz, mode
        )

        # Store current mode for derivatives_array callback
        self._current_mode = mode

        # Create path object
        path = RayPath(
            start_position=(tx_lat, tx_lon, tx_alt),
            start_direction=(elevation, azimuth),
            frequency_mhz=frequency_mhz,
            mode=mode,
        )

        # Store initial state
        path.states.append(state.copy())
        max_alt_reached = state.altitude()

        # Get integrator (if configured)
        integrator = self._get_integrator() if self._use_pluggable_integrator else None

        # Reset integrator state for new ray (important for Adams-Bashforth)
        if integrator is not None:
            integrator.reset()

        # Integrate using configured method
        max_iterations = 10000  # Safety limit to prevent infinite loops
        iteration = 0
        while state.path_length < max_path_km and iteration < max_iterations:
            iteration += 1
            # Integration step (pluggable or built-in RK4)
            try:
                if integrator is not None:
                    state = self._integrator_step(state, step_km, integrator)
                else:
                    state = self._rk4_step(state, step_km)
            except Exception as e:
                logger.warning(f"Integration error: {e}")
                path.termination = RayTermination.ERROR
                break

            # Track maximum altitude reached
            current_alt = state.altitude()
            max_alt_reached = max(max_alt_reached, current_alt)

            # Store state
            path.states.append(state.copy())

            # Check termination conditions (after integration step)
            termination = self._check_termination(state, max_alt_reached)
            if termination:
                path.termination = termination
                break

        # Set landing position if ground hit
        if path.termination == RayTermination.GROUND_HIT:
            path.landing_position = state.lat_lon_alt()

        return path

    def trace_fan(
        self,
        tx_lat: float,
        tx_lon: float,
        tx_alt: float,
        elevation_range: Tuple[float, float],
        elevation_step: float,
        azimuth: float,
        frequency_mhz: float,
        mode: RayMode = RayMode.ORDINARY,
    ) -> List[RayPath]:
        """
        Trace a fan of rays at different elevations.

        Useful for creating ionograms or coverage maps.

        Args:
            tx_lat, tx_lon, tx_alt: Transmitter position
            elevation_range: (min_elevation, max_elevation) in degrees
            elevation_step: Step between elevations (degrees)
            azimuth: Fixed azimuth (degrees)
            frequency_mhz: Transmission frequency
            mode: O-mode or X-mode

        Returns:
            List of RayPath for each elevation
        """
        paths = []
        el_min, el_max = elevation_range
        elevations = np.arange(el_min, el_max + elevation_step, elevation_step)

        for elevation in elevations:
            path = self.trace_ray(
                tx_lat, tx_lon, tx_alt,
                elevation, azimuth,
                frequency_mhz, mode
            )
            paths.append(path)

        return paths

    def _initialize_ray(
        self,
        lat: float, lon: float, alt: float,
        elevation: float, azimuth: float,
        frequency_mhz: float, mode: RayMode
    ) -> RayState:
        """
        Initialize ray state from launch parameters.

        Converts geodetic coordinates to ECEF and calculates
        initial wave vector from elevation/azimuth.
        """
        # Convert geodetic to ECEF
        r = EARTH_RADIUS_KM + alt
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)

        x = r * np.cos(lat_rad) * np.cos(lon_rad)
        y = r * np.cos(lat_rad) * np.sin(lon_rad)
        z = r * np.sin(lat_rad)

        # Calculate initial wave vector in ENU, then rotate to ECEF
        el_rad = np.radians(elevation)
        az_rad = np.radians(azimuth)

        # ENU components (East, North, Up)
        k_e = np.cos(el_rad) * np.sin(az_rad)
        k_n = np.cos(el_rad) * np.cos(az_rad)
        k_u = np.sin(el_rad)

        # Rotate ENU to ECEF
        # R = rotation matrix from ENU to ECEF at (lat, lon)
        sin_lat = np.sin(lat_rad)
        cos_lat = np.cos(lat_rad)
        sin_lon = np.sin(lon_rad)
        cos_lon = np.cos(lon_rad)

        # ECEF components
        kx = -sin_lon * k_e - sin_lat * cos_lon * k_n + cos_lat * cos_lon * k_u
        ky = cos_lon * k_e - sin_lat * sin_lon * k_n + cos_lat * sin_lon * k_u
        kz = cos_lat * k_n + sin_lat * k_u

        return RayState(
            x=x, y=y, z=z,
            kx=kx, ky=ky, kz=kz,
            mode=mode,
            frequency_mhz=frequency_mhz
        )

    def _integrator_step(
        self,
        state: RayState,
        ds: float,
        integrator: BaseIntegrator,
    ) -> RayState:
        """
        Perform integration step using pluggable integrator.

        This method wraps the integrator interface to work with RayState
        objects and handles reflection logic.

        Args:
            state: Current ray state
            ds: Requested step size in km
            integrator: Configured integrator instance

        Returns:
            Updated ray state
        """
        # Adaptive step size based on refractive index
        n = self._get_refractive_index(state)
        n_real = max(abs(n.real), 0.01)
        adaptive_ds = ds * min(n_real, 1.0)
        adaptive_ds = max(adaptive_ds, 0.01)

        # Convert RayState to array for integrator
        state_array = np.array([
            state.x, state.y, state.z,
            state.kx, state.ky, state.kz
        ])

        # Perform integration step
        result: IntegrationStep = integrator.step(
            state_array,
            adaptive_ds,
            state.frequency_mhz
        )

        # Extract new state
        new_array = result.state

        # Create new RayState
        new_state = RayState(
            x=new_array[0],
            y=new_array[1],
            z=new_array[2],
            kx=new_array[3],
            ky=new_array[4],
            kz=new_array[5],
            path_length=state.path_length + result.step_size_used,
            group_path=state.group_path + self._group_path_increment(state, result.step_size_used),
            time=state.time + result.step_size_used / (SPEED_OF_LIGHT / 1000),
            mode=state.mode,
            frequency_mhz=state.frequency_mhz
        )

        # Check for reflection (same logic as built-in RK4)
        n_new = self._get_refractive_index(new_state)
        if n_new.real < 0.1 and abs(n_new.imag) > 0.1:
            r = np.sqrt(new_state.x**2 + new_state.y**2 + new_state.z**2)
            if r > 0:
                r_x = new_state.x / r
                r_y = new_state.y / r
                r_z = new_state.z / r

                k_radial = (new_state.kx * r_x + new_state.ky * r_y + new_state.kz * r_z)

                if k_radial > 0:
                    new_state.kx -= 2 * k_radial * r_x
                    new_state.ky -= 2 * k_radial * r_y
                    new_state.kz -= 2 * k_radial * r_z

        # Normalize wave vector
        k_mag = np.sqrt(new_state.kx**2 + new_state.ky**2 + new_state.kz**2)
        if k_mag > 0:
            new_state.kx /= k_mag
            new_state.ky /= k_mag
            new_state.kz /= k_mag

        return new_state

    def _rk4_step(self, state: RayState, ds: float) -> RayState:
        """
        Perform one RK4 integration step.

        Args:
            state: Current ray state
            ds: Step size in km

        Returns:
            Updated ray state
        """
        # Adaptive step size based on refractive index
        # When n is small, we need smaller steps to avoid jumping over reflection
        n = self._get_refractive_index(state)
        n_real = max(abs(n.real), 0.01)

        # Scale step size: smaller steps when n is small
        # At n=1, use full step. At n=0.1, use 1/10 step.
        adaptive_ds = ds * min(n_real, 1.0)
        adaptive_ds = max(adaptive_ds, 0.01)  # Minimum step 10m

        # Get derivatives at current position
        k1 = self._derivatives(state)

        # Midpoint 1
        state_mid1 = self._apply_derivatives(state, k1, adaptive_ds/2)
        k2 = self._derivatives(state_mid1)

        # Midpoint 2
        state_mid2 = self._apply_derivatives(state, k2, adaptive_ds/2)
        k3 = self._derivatives(state_mid2)

        # Endpoint
        state_end = self._apply_derivatives(state, k3, adaptive_ds)
        k4 = self._derivatives(state_end)

        # Combine derivatives (RK4 formula)
        dx = (k1[0] + 2*k2[0] + 2*k3[0] + k4[0]) / 6 * adaptive_ds
        dy = (k1[1] + 2*k2[1] + 2*k3[1] + k4[1]) / 6 * adaptive_ds
        dz = (k1[2] + 2*k2[2] + 2*k3[2] + k4[2]) / 6 * adaptive_ds
        dkx = (k1[3] + 2*k2[3] + 2*k3[3] + k4[3]) / 6 * adaptive_ds
        dky = (k1[4] + 2*k2[4] + 2*k3[4] + k4[4]) / 6 * adaptive_ds
        dkz = (k1[5] + 2*k2[5] + 2*k3[5] + k4[5]) / 6 * adaptive_ds

        # Update state
        new_state = RayState(
            x=state.x + dx,
            y=state.y + dy,
            z=state.z + dz,
            kx=state.kx + dkx,
            ky=state.ky + dky,
            kz=state.kz + dkz,
            path_length=state.path_length + adaptive_ds,
            group_path=state.group_path + self._group_path_increment(state, adaptive_ds),
            time=state.time + adaptive_ds / (SPEED_OF_LIGHT / 1000),  # km/s
            mode=state.mode,
            frequency_mhz=state.frequency_mhz
        )

        # Check for reflection: if n is imaginary (n² < 0), the ray should reflect
        # This happens when f < fp (local plasma frequency exceeds wave frequency)
        n_new = self._get_refractive_index(new_state)
        if n_new.real < 0.1 and abs(n_new.imag) > 0.1:
            # Reflection condition: reverse the radial component of wave vector
            # In ECEF, "radial" at position (x,y,z) is the unit vector (x,y,z)/|r|
            lat, lon, alt = new_state.lat_lon_alt()
            r = np.sqrt(new_state.x**2 + new_state.y**2 + new_state.z**2)
            if r > 0:
                # Radial unit vector
                r_x = new_state.x / r
                r_y = new_state.y / r
                r_z = new_state.z / r

                # Current wave vector dot radial = radial component
                k_radial = (new_state.kx * r_x + new_state.ky * r_y + new_state.kz * r_z)

                # If ray is going outward (k_radial > 0), reflect by reversing radial component
                if k_radial > 0:
                    new_state.kx -= 2 * k_radial * r_x
                    new_state.ky -= 2 * k_radial * r_y
                    new_state.kz -= 2 * k_radial * r_z

        # Normalize wave vector
        k_mag = np.sqrt(new_state.kx**2 + new_state.ky**2 + new_state.kz**2)
        if k_mag > 0:
            new_state.kx /= k_mag
            new_state.ky /= k_mag
            new_state.kz /= k_mag

        return new_state

    def _derivatives(self, state: RayState) -> Tuple[float, ...]:
        """
        Calculate Haselgrove equation derivatives.

        Returns:
            Tuple of (dx/ds, dy/ds, dz/ds, dkx/ds, dky/ds, dkz/ds)
        """
        lat, lon, alt = state.lat_lon_alt()

        # Get refractive index at current position
        n = self._get_refractive_index(state)

        # Minimum n for stable propagation (avoid division by near-zero)
        n_min = 0.1

        # Position derivatives: dr/ds = k_hat / n
        # (ray advances in direction of wave vector, scaled by 1/n)
        n_eff = max(n.real, n_min)
        dx_ds = state.kx / n_eff
        dy_ds = state.ky / n_eff
        dz_ds = state.kz / n_eff

        # Wave vector derivatives: dk/ds = -∇n / n
        # Calculate gradient numerically
        grad_n = self._gradient_n(state)

        dkx_ds = -grad_n[0] / n_eff
        dky_ds = -grad_n[1] / n_eff
        dkz_ds = -grad_n[2] / n_eff

        return (dx_ds, dy_ds, dz_ds, dkx_ds, dky_ds, dkz_ds)

    def _apply_derivatives(
        self,
        state: RayState,
        derivatives: Tuple[float, ...],
        ds: float
    ) -> RayState:
        """Apply derivatives to state for intermediate RK4 steps."""
        return RayState(
            x=state.x + derivatives[0] * ds,
            y=state.y + derivatives[1] * ds,
            z=state.z + derivatives[2] * ds,
            kx=state.kx + derivatives[3] * ds,
            ky=state.ky + derivatives[4] * ds,
            kz=state.kz + derivatives[5] * ds,
            path_length=state.path_length,
            group_path=state.group_path,
            time=state.time,
            mode=state.mode,
            frequency_mhz=state.frequency_mhz
        )

    def _get_refractive_index(self, state: RayState) -> complex:
        """Get refractive index at ray position."""
        lat, lon, alt = state.lat_lon_alt()

        # Below ground or very low: use 1.0
        if alt < 50:
            return complex(1.0, 0.0)

        # Get both modes
        n_O, n_X = self.ionosphere.get_refractive_index(
            lat, lon, alt,
            state.frequency_mhz,
            (state.kx, state.ky, state.kz)
        )

        if state.mode == RayMode.ORDINARY:
            return n_O
        else:
            return n_X

    def _gradient_n(self, state: RayState) -> np.ndarray:
        """
        Calculate gradient of refractive index numerically.

        Uses central differences in ECEF coordinates.
        """
        delta = self.GRADIENT_DELTA_KM
        grad = np.zeros(3)

        # X gradient
        state_plus = state.copy()
        state_plus.x += delta
        state_minus = state.copy()
        state_minus.x -= delta
        n_plus = self._get_refractive_index(state_plus)
        n_minus = self._get_refractive_index(state_minus)
        grad[0] = (n_plus.real - n_minus.real) / (2 * delta)

        # Y gradient
        state_plus = state.copy()
        state_plus.y += delta
        state_minus = state.copy()
        state_minus.y -= delta
        n_plus = self._get_refractive_index(state_plus)
        n_minus = self._get_refractive_index(state_minus)
        grad[1] = (n_plus.real - n_minus.real) / (2 * delta)

        # Z gradient
        state_plus = state.copy()
        state_plus.z += delta
        state_minus = state.copy()
        state_minus.z -= delta
        n_plus = self._get_refractive_index(state_plus)
        n_minus = self._get_refractive_index(state_minus)
        grad[2] = (n_plus.real - n_minus.real) / (2 * delta)

        return grad

    def _group_path_increment(self, state: RayState, ds: float) -> float:
        """Calculate group path increment (virtual height contribution)."""
        n = self._get_refractive_index(state)
        if n.real > 0.01:
            # Group path = ∫ n' ds where n' is group refractive index
            # For simplicity, use phase index (true group index requires dn/df)
            return ds / n.real
        return ds

    def _check_termination(self, state: RayState, max_alt_reached: float = 0.0) -> Optional[RayTermination]:
        """
        Check if ray should terminate.

        Args:
            state: Current ray state
            max_alt_reached: Maximum altitude reached so far

        Returns:
            RayTermination reason or None to continue
        """
        alt = state.altitude()

        # Ground hit - only if we've already gone up significantly and come back down
        # This prevents immediate termination at launch
        if alt <= self.MIN_ALTITUDE_KM and max_alt_reached > 50.0:
            return RayTermination.GROUND_HIT

        # Escape to space
        if alt >= self.MAX_ALTITUDE_KM:
            return RayTermination.ESCAPE

        return None


def test_ray_trace():
    """Test basic ray tracing."""
    from .electron_density import create_test_profile

    print("Haselgrove Ray Trace Test")
    print("=" * 50)

    # Create ionospheric model with higher foF2
    model = create_test_profile()
    # Update to higher foF2 for better reflection
    model.update_from_realtime(foF2=12.0, hmF2=300.0)

    # Create solver
    solver = HaselgroveSolver(model)

    # Test multiple frequencies
    print("\nTesting reflection at different frequencies (foF2=12.0 MHz):")
    print(f"{'Freq (MHz)':<12} {'Elevation':<12} {'Termination':<15} {'Path (km)':<10}")
    print("-" * 49)

    for freq in [5.0, 7.0, 10.0, 15.0]:
        path = solver.trace_ray(
            tx_lat=40.0,
            tx_lon=-105.0,
            tx_alt=0.0,
            elevation=60.0,
            azimuth=0.0,
            frequency_mhz=freq,
            mode=RayMode.ORDINARY
        )
        term = path.termination.value
        plen = path.total_path_length
        print(f"{freq:<12.1f} 60°          {term:<15} {plen:>7.1f}")

    # Test elevations for NVIS at 7 MHz (below foF2)
    print(f"\nNVIS test at 7 MHz (below foF2=12 MHz), bearing North:")
    print(f"{'Elevation':<12} {'Termination':<15} {'Path (km)':<12} {'Max Alt (km)':<12}")
    print("-" * 51)

    for el in [45, 60, 75, 85, 90]:
        path = solver.trace_ray(
            tx_lat=40.0,
            tx_lon=-105.0,
            tx_alt=0.0,
            elevation=el,
            azimuth=0.0,
            frequency_mhz=7.0,
            mode=RayMode.ORDINARY
        )

        term = path.termination.value
        plen = path.total_path_length

        # Find maximum altitude reached
        max_alt = max(s.altitude() for s in path.states) if path.states else 0

        print(f"{el}°          {term:<15} {plen:>8.1f}     {max_alt:>8.1f}")

    # Show ray path for one NVIS ray
    print("\n\nDetailed 80° elevation ray path:")
    path = solver.trace_ray(
        tx_lat=40.0, tx_lon=-105.0, tx_alt=0.0,
        elevation=80.0, azimuth=0.0,
        frequency_mhz=7.0,
        mode=RayMode.ORDINARY,
        step_km=10.0  # Larger steps for summary
    )

    print(f"Steps: {len(path.states)}, Termination: {path.termination.value}")
    if len(path.states) > 10:
        # Sample every 10th state
        print(f"{'Step':<6} {'Alt (km)':<12} {'n_real':<10}")
        print("-" * 28)
        for i in range(0, len(path.states), max(1, len(path.states)//10)):
            s = path.states[i]
            lat, lon, alt = s.lat_lon_alt()
            n = solver._get_refractive_index(s)
            print(f"{i:<6} {alt:>8.1f}     {n.real:>8.4f}")


if __name__ == "__main__":
    test_ray_trace()
