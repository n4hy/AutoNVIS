"""
PHaRLAP Replacement - Pure Python/C++ Implementation

This module provides a drop-in replacement for MATLAB PHaRLAP,
using a native C++ ray tracing engine with Python bindings.

No MATLAB dependency required!

Usage:
    from src.propagation.python.pharlap_replacement import RayTracer

    # Initialize with ionospheric grid from SR-UKF
    tracer = RayTracer(ne_grid, lat, lon, alt)

    # Trace rays
    paths = tracer.trace_nvis(tx_lat=40.0, tx_lon=-105.0, freq_mhz=5.0)

    # Calculate coverage
    coverage = tracer.calculate_coverage(tx_lat=40.0, tx_lon=-105.0,
                                        freq_min=2.0, freq_max=15.0)

    # Get LUF/MUF
    luf, muf = coverage['luf'], coverage['muf']
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

try:
    # Import C++ ray tracer module
    from . import raytracer
except ImportError:
    # If not built yet, provide helpful error
    raise ImportError(
        "Ray tracer C++ module not found. Please build it first:\n"
        "  cd src/propagation\n"
        "  cmake -B build && cmake --build build -j$(nproc)\n"
    )


class RayTracer:
    """
    High-level Python interface to 3D ionospheric ray tracer.

    This class provides a PHaRLAP-compatible API using our native
    C++ implementation instead of MATLAB.
    """

    def __init__(
        self,
        ne_grid: np.ndarray,
        lat: np.ndarray,
        lon: np.ndarray,
        alt: np.ndarray,
        xray_flux: float = 0.0,
        igrf_file: Optional[str] = None
    ):
        """
        Initialize ray tracer with ionospheric grid.

        Args:
            ne_grid: Electron density grid (n_lat, n_lon, n_alt) in el/m³
            lat: Latitude grid (degrees)
            lon: Longitude grid (degrees)
            alt: Altitude grid (km)
            xray_flux: GOES X-ray flux for D-region absorption (W/m²)
            igrf_file: Path to IGRF coefficients file (optional)
        """
        self.logger = logging.getLogger(__name__)

        # Store grid parameters
        self.lat = lat
        self.lon = lon
        self.alt = alt
        self.xray_flux = xray_flux

        # Flatten Ne grid to row-major order for C++
        ne_flat = ne_grid.flatten('C').tolist()

        # Create C++ ionospheric grid object
        self.iono_grid = raytracer.IonoGrid(lat, lon, alt, ne_flat)

        # Create geomagnetic field model
        if igrf_file:
            self.geomag = raytracer.GeomagneticField(igrf_file)
        else:
            self.geomag = raytracer.GeomagneticField()  # Use dipole model

        # Default ray tracing configuration
        self.config = raytracer.RayTracingConfig()
        self.config.tolerance = 1e-7
        self.config.max_path_length_km = 5000.0  # NVIS range
        self.config.initial_step_km = 0.5
        self.config.calculate_absorption = True
        self.config.mode = raytracer.Mode.O_MODE  # O-mode default for NVIS

        # Create C++ ray tracer
        self.tracer = raytracer.RayTracer3D(self.iono_grid, self.geomag, self.config)

        self.logger.info(f"Ray tracer initialized: {len(lat)}×{len(lon)}×{len(alt)} grid")

    def trace_ray(
        self,
        tx_lat: float,
        tx_lon: float,
        elevation: float,
        azimuth: float,
        freq_mhz: float
    ) -> Dict:
        """
        Trace a single ray.

        Args:
            tx_lat: Transmitter latitude (degrees)
            tx_lon: Transmitter longitude (degrees)
            elevation: Elevation angle (degrees, 0-90)
            azimuth: Azimuth angle (degrees, 0-360, clockwise from North)
            freq_mhz: Frequency (MHz)

        Returns:
            Dictionary with ray path information
        """
        path = self.tracer.trace_ray(tx_lat, tx_lon, 0.0, elevation, azimuth, freq_mhz)

        return {
            'positions': np.array([p for p in path.positions]),
            'wave_normals': np.array([w for w in path.wave_normals]),
            'path_length': path.path_lengths[-1] if path.path_lengths else 0.0,
            'ground_range': path.ground_range,
            'apex_altitude': path.apex_altitude,
            'apex_lat': path.apex_lat,
            'apex_lon': path.apex_lon,
            'absorption_db': path.absorption_db[-1] if path.absorption_db else 0.0,
            'reflected': path.reflected,
            'escaped': path.escaped,
            'absorbed': path.absorbed,
            'refractive_indices': np.array(path.refractive_indices)
        }

    def trace_nvis(
        self,
        tx_lat: float,
        tx_lon: float,
        freq_mhz: float,
        elevation_min: float = 70.0,
        elevation_max: float = 90.0,
        elevation_step: float = 2.0,
        azimuth_step: float = 15.0
    ) -> List[Dict]:
        """
        Trace NVIS ray fan.

        Args:
            tx_lat, tx_lon: Transmitter position (degrees)
            freq_mhz: Frequency (MHz)
            elevation_min: Minimum elevation angle (degrees)
            elevation_max: Maximum elevation angle (degrees)
            elevation_step: Elevation step (degrees)
            azimuth_step: Azimuth step (degrees)

        Returns:
            List of ray path dictionaries
        """
        paths = self.tracer.calculate_nvis_coverage(
            tx_lat, tx_lon, freq_mhz,
            elevation_min, elevation_max,
            elevation_step, azimuth_step
        )

        return [self._path_to_dict(p) for p in paths]

    def calculate_coverage(
        self,
        tx_lat: float,
        tx_lon: float,
        freq_min: float = 2.0,
        freq_max: float = 15.0,
        freq_step: float = 0.5,
        snr_threshold_db: float = 10.0
    ) -> Dict:
        """
        Calculate NVIS coverage map with LUF/MUF analysis.

        Args:
            tx_lat, tx_lon: Transmitter position (degrees)
            freq_min: Minimum frequency (MHz)
            freq_max: Maximum frequency (MHz)
            freq_step: Frequency step (MHz)
            snr_threshold_db: SNR threshold for usability (dB)

        Returns:
            Dictionary with coverage information:
                - luf: Lowest Usable Frequency (MHz)
                - muf: Maximum Usable Frequency (MHz)
                - optimal_freq: Optimal frequency (MHz)
                - coverage_map: Dict mapping frequencies to ray paths
                - blackout: Boolean indicating if LUF > MUF
        """
        coverage_map = {}
        frequencies = np.arange(freq_min, freq_max + freq_step, freq_step)

        for freq in frequencies:
            paths = self.trace_nvis(tx_lat, tx_lon, freq)
            coverage_map[freq] = paths

        # Calculate LUF (minimum frequency with acceptable absorption)
        luf = freq_max
        for freq in frequencies:
            paths = coverage_map[freq]
            min_absorption = min(p['absorption_db'] for p in paths if p['reflected'])

            if min_absorption < 50.0:  # Reasonable absorption threshold
                luf = freq
                break

        # Calculate MUF (maximum frequency that reflects)
        muf = freq_min
        for freq in reversed(frequencies):
            paths = coverage_map[freq]
            any_reflected = any(p['reflected'] for p in paths)

            if any_reflected:
                muf = freq
                break

        # Optimal frequency (FOT = 0.85 * MUF)
        optimal_freq = 0.85 * muf

        # Check for blackout condition
        blackout = luf > muf

        return {
            'luf': luf,
            'muf': muf,
            'optimal_freq': optimal_freq,
            'fot': optimal_freq,  # Alias
            'coverage_map': coverage_map,
            'frequencies': frequencies.tolist(),
            'blackout': blackout,
            'usable_range': (luf, muf) if not blackout else None
        }

    def calculate_luf_muf_grid(
        self,
        tx_lat: float,
        tx_lon: float,
        grid_size: int = 50
    ) -> Dict:
        """
        Calculate LUF/MUF spatial grid around transmitter.

        Args:
            tx_lat, tx_lon: Transmitter position
            grid_size: Number of grid points

        Returns:
            Dictionary with 2D LUF/MUF grids
        """
        # Create spatial grid (±500 km around transmitter)
        lat_range = np.linspace(tx_lat - 5, tx_lat + 5, grid_size)
        lon_range = np.linspace(tx_lon - 5, tx_lon + 5, grid_size)

        luf_grid = np.zeros((grid_size, grid_size))
        muf_grid = np.zeros((grid_size, grid_size))

        # For each grid point, find azimuth and calculate coverage
        for i, lat in enumerate(lat_range):
            for j, lon in enumerate(lon_range):
                # Calculate azimuth from tx to this point
                dlat = lat - tx_lat
                dlon = lon - tx_lon
                azim = np.degrees(np.arctan2(dlon, dlat))
                azim = (azim + 360) % 360  # Normalize to 0-360

                # Trace rays in this direction for multiple frequencies
                freqs = [3, 5, 7, 10, 12, 15]
                reflected = []

                for freq in freqs:
                    path = self.trace_ray(tx_lat, tx_lon, 85.0, azim, freq)
                    if path['reflected'] and path['absorption_db'] < 50.0:
                        reflected.append(freq)

                # LUF = min reflected, MUF = max reflected
                luf_grid[i, j] = min(reflected) if reflected else freqs[-1]
                muf_grid[i, j] = max(reflected) if reflected else freqs[0]

        return {
            'luf_grid': luf_grid,
            'muf_grid': muf_grid,
            'lat_range': lat_range,
            'lon_range': lon_range
        }

    def set_xray_flux(self, xray_flux: float):
        """
        Update X-ray flux for D-region absorption calculation.

        Args:
            xray_flux: GOES X-ray flux (W/m²)
        """
        self.xray_flux = xray_flux
        # Absorption is calculated on-the-fly during ray tracing

    def set_mode(self, mode: str = 'O'):
        """
        Set polarization mode.

        Args:
            mode: 'O' for O-mode, 'X' for X-mode
        """
        if mode.upper() == 'O':
            self.config.mode = raytracer.Mode.O_MODE
        elif mode.upper() == 'X':
            self.config.mode = raytracer.Mode.X_MODE
        else:
            raise ValueError("Mode must be 'O' or 'X'")

        # Recreate tracer with new config
        self.tracer = raytracer.RayTracer3D(self.iono_grid, self.geomag, self.config)

    def _path_to_dict(self, path) -> Dict:
        """Convert C++ RayPath object to Python dictionary."""
        return {
            'positions': np.array([p for p in path.positions]),
            'wave_normals': np.array([w for w in path.wave_normals]),
            'path_length': path.path_lengths[-1] if path.path_lengths else 0.0,
            'ground_range': path.ground_range,
            'apex_altitude': path.apex_altitude,
            'apex_lat': path.apex_lat,
            'apex_lon': path.apex_lon,
            'absorption_db': path.absorption_db[-1] if path.absorption_db else 0.0,
            'reflected': path.reflected,
            'escaped': path.escaped,
            'absorbed': path.absorbed,
            'refractive_indices': np.array(path.refractive_indices)
        }


def create_from_srukf_grid(ne_grid: np.ndarray, lat: np.ndarray,
                           lon: np.ndarray, alt: np.ndarray,
                           xray_flux: float = 0.0) -> RayTracer:
    """
    Convenience function to create RayTracer from SR-UKF output.

    Args:
        ne_grid: Electron density from SR-UKF (73, 73, 55)
        lat, lon, alt: Grid coordinates
        xray_flux: Current X-ray flux from GOES

    Returns:
        Configured RayTracer instance
    """
    return RayTracer(ne_grid, lat, lon, alt, xray_flux)


# Alias for PHaRLAP compatibility
raytrace_3d = RayTracer.trace_ray
nvis_coverage = RayTracer.calculate_coverage
