"""
Propagation Service - Ray Tracing Integration for Auto-NVIS

This service integrates the native C++ ray tracer with the SR-UKF filter
to produce real-time LUF/MUF predictions and propagation products.

Interfaces:
    - Input: Electron density grid from SR-UKF filter
    - Output: LUF/MUF products published to RabbitMQ
    - Configuration: Grid coordinates, transmitter location, frequencies
"""

import sys
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import numpy as np

# Add propagation module to path
sys.path.insert(0, '/home/n4hy/AutoNVIS/src/propagation/python')

from pharlap_replacement import RayTracer

# Import from products module
try:
    from ..products.luf_muf_calculator import LUFMUFCalculator, FrequencyRecommender
except ImportError:
    # Fallback for direct execution
    from pathlib import Path
    products_path = Path(__file__).parent.parent / "products"
    sys.path.insert(0, str(products_path))
    from luf_muf_calculator import LUFMUFCalculator, FrequencyRecommender


class PropagationService:
    """
    Service for real-time ionospheric propagation prediction.

    Integrates native C++ ray tracer with SR-UKF filter output to
    calculate LUF/MUF and generate frequency recommendations.
    """

    def __init__(
        self,
        tx_lat: float,
        tx_lon: float,
        tx_alt: float = 0.0,
        freq_min: float = 2.0,
        freq_max: float = 15.0,
        freq_step: float = 0.5,
        elevation_min: float = 70.0,
        elevation_max: float = 90.0,
        elevation_step: float = 2.0,
        azimuth_step: float = 15.0,
        absorption_threshold_db: float = 50.0,
        snr_threshold_db: float = 10.0
    ):
        """
        Initialize propagation service.

        Args:
            tx_lat: Transmitter latitude (degrees)
            tx_lon: Transmitter longitude (degrees)
            tx_alt: Transmitter altitude (km)
            freq_min: Minimum frequency to test (MHz)
            freq_max: Maximum frequency to test (MHz)
            freq_step: Frequency step size (MHz)
            elevation_min: Minimum elevation angle (degrees, NVIS: 70)
            elevation_max: Maximum elevation angle (degrees, NVIS: 90)
            elevation_step: Elevation step (degrees)
            azimuth_step: Azimuth step (degrees)
            absorption_threshold_db: Maximum acceptable absorption (dB)
            snr_threshold_db: Minimum acceptable SNR (dB)
        """
        self.logger = logging.getLogger(__name__)

        # Transmitter configuration
        self.tx_lat = tx_lat
        self.tx_lon = tx_lon
        self.tx_alt = tx_alt

        # Frequency scan configuration
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.freq_step = freq_step

        # Ray tracing configuration (NVIS geometry)
        self.elevation_min = elevation_min
        self.elevation_max = elevation_max
        self.elevation_step = elevation_step
        self.azimuth_step = azimuth_step

        # Quality thresholds
        self.absorption_threshold_db = absorption_threshold_db
        self.snr_threshold_db = snr_threshold_db

        # Ray tracer instance (created when grid is provided)
        self.tracer: Optional[RayTracer] = None

        # LUF/MUF calculator
        self.calculator = LUFMUFCalculator(
            absorption_threshold_db=absorption_threshold_db,
            snr_threshold_db=snr_threshold_db
        )

        # Frequency recommender
        self.recommender = FrequencyRecommender()

        self.logger.info(
            f"PropagationService initialized: TX=({tx_lat:.2f}, {tx_lon:.2f}), "
            f"Freq={freq_min}-{freq_max} MHz, Elev={elevation_min}-{elevation_max}°"
        )

    def initialize_ray_tracer(
        self,
        ne_grid: np.ndarray,
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
        alt_grid: np.ndarray,
        xray_flux: float = 0.0
    ) -> None:
        """
        Initialize ray tracer with ionospheric grid from SR-UKF filter.

        Args:
            ne_grid: Electron density grid (n_lat, n_lon, n_alt) in el/m³
            lat_grid: Latitude grid (degrees)
            lon_grid: Longitude grid (degrees)
            alt_grid: Altitude grid (km)
            xray_flux: GOES X-ray flux for D-region absorption (W/m²)
        """
        self.logger.info(
            f"Initializing ray tracer with grid: "
            f"{ne_grid.shape[0]}×{ne_grid.shape[1]}×{ne_grid.shape[2]}"
        )

        # Validate grid dimensions
        if ne_grid.shape != (len(lat_grid), len(lon_grid), len(alt_grid)):
            raise ValueError(
                f"Grid shape mismatch: ne_grid={ne_grid.shape}, "
                f"expected ({len(lat_grid)}, {len(lon_grid)}, {len(alt_grid)})"
            )

        # Create ray tracer instance
        self.tracer = RayTracer(
            ne_grid=ne_grid,
            lat=lat_grid,
            lon=lon_grid,
            alt=alt_grid,
            xray_flux=xray_flux
        )

        # Log grid statistics
        ne_max = np.max(ne_grid)
        ne_mean = np.mean(ne_grid)
        self.logger.info(
            f"Ray tracer initialized: Ne_max={ne_max:.2e} el/m³, "
            f"Ne_mean={ne_mean:.2e} el/m³, X-ray={xray_flux:.2e} W/m²"
        )

    def calculate_luf_muf(self) -> Dict[str, Any]:
        """
        Calculate LUF/MUF and propagation products.

        Returns:
            Dictionary containing:
                - luf_mhz: Lowest Usable Frequency
                - muf_mhz: Maximum Usable Frequency
                - fot_mhz: Frequency of Optimum Traffic
                - usable_range_mhz: [LUF, MUF] tuple
                - blackout: True if no usable frequencies
                - coverage_stats: Ray statistics
                - frequency_recommendations: Recommended frequencies for ALE
                - timestamp_utc: Calculation time

        Raises:
            RuntimeError: If ray tracer not initialized
        """
        if self.tracer is None:
            raise RuntimeError("Ray tracer not initialized. Call initialize_ray_tracer() first.")

        self.logger.info("Calculating LUF/MUF coverage...")
        start_time = datetime.utcnow()

        # Calculate multi-frequency coverage
        coverage = self.tracer.calculate_coverage(
            tx_lat=self.tx_lat,
            tx_lon=self.tx_lon,
            freq_min=self.freq_min,
            freq_max=self.freq_max,
            freq_step=self.freq_step,
            snr_threshold_db=self.snr_threshold_db
        )

        # Extract LUF/MUF results
        luf_mhz = coverage['luf']
        muf_mhz = coverage['muf']
        fot_mhz = coverage['fot']  # FOT = 0.85 * MUF
        blackout = coverage['blackout']

        # Calculate coverage statistics
        total_rays = 0
        reflected_rays = 0
        usable_rays = 0
        total_absorption = 0.0

        for freq, paths in coverage['coverage_map'].items():
            total_rays += len(paths)
            for path in paths:
                if path['reflected']:
                    reflected_rays += 1
                    if path['absorption_db'] < self.absorption_threshold_db:
                        usable_rays += 1
                        total_absorption += path['absorption_db']

        avg_absorption_db = total_absorption / usable_rays if usable_rays > 0 else 0.0

        coverage_stats = {
            'total_rays': total_rays,
            'reflected_rays': reflected_rays,
            'usable_rays': usable_rays,
            'reflection_rate': reflected_rays / total_rays if total_rays > 0 else 0.0,
            'usability_rate': usable_rays / total_rays if total_rays > 0 else 0.0,
            'avg_absorption_db': avg_absorption_db
        }

        # Get frequency recommendations for ALE
        recommendations = self.recommender.recommend_frequencies(
            luf_mhz=luf_mhz,
            muf_mhz=muf_mhz,
            num_frequencies=5,
            strategy='distributed'
        )

        calculation_time = (datetime.utcnow() - start_time).total_seconds()

        self.logger.info(
            f"LUF/MUF calculated in {calculation_time:.2f}s: "
            f"LUF={luf_mhz:.2f} MHz, MUF={muf_mhz:.2f} MHz, "
            f"FOT={fot_mhz:.2f} MHz, Blackout={blackout}"
        )

        return {
            'luf_mhz': float(luf_mhz),
            'muf_mhz': float(muf_mhz),
            'fot_mhz': float(fot_mhz),
            'usable_range_mhz': [float(luf_mhz), float(muf_mhz)] if not blackout else None,
            'blackout': bool(blackout),
            'coverage_stats': coverage_stats,
            'frequency_recommendations': recommendations,
            'transmitter': {
                'latitude': self.tx_lat,
                'longitude': self.tx_lon,
                'altitude_km': self.tx_alt
            },
            'timestamp_utc': datetime.utcnow().isoformat() + 'Z',
            'calculation_time_sec': calculation_time
        }

    def calculate_nvis_coverage(self, freq_mhz: float) -> Dict[str, Any]:
        """
        Calculate NVIS coverage map for a single frequency.

        Args:
            freq_mhz: Frequency to calculate (MHz)

        Returns:
            Dictionary containing:
                - frequency_mhz: Operating frequency
                - ray_paths: List of ray path dictionaries
                - coverage_summary: Statistics about coverage
                - timestamp_utc: Calculation time
        """
        if self.tracer is None:
            raise RuntimeError("Ray tracer not initialized.")

        self.logger.info(f"Calculating NVIS coverage for {freq_mhz:.2f} MHz...")

        paths = self.tracer.trace_nvis(
            tx_lat=self.tx_lat,
            tx_lon=self.tx_lon,
            freq_mhz=freq_mhz,
            elevation_min=self.elevation_min,
            elevation_max=self.elevation_max,
            elevation_step=self.elevation_step,
            azimuth_step=self.azimuth_step
        )

        # Analyze coverage
        reflected = sum(1 for p in paths if p['reflected'])
        escaped = sum(1 for p in paths if p['escaped'])
        absorbed = sum(1 for p in paths if p['absorbed'])

        avg_range = np.mean([p['ground_range'] for p in paths if p['reflected']]) if reflected > 0 else 0.0
        max_range = np.max([p['ground_range'] for p in paths if p['reflected']]) if reflected > 0 else 0.0

        coverage_summary = {
            'total_rays': len(paths),
            'reflected': reflected,
            'escaped': escaped,
            'absorbed': absorbed,
            'reflection_rate': reflected / len(paths) if len(paths) > 0 else 0.0,
            'avg_ground_range_km': float(avg_range),
            'max_ground_range_km': float(max_range)
        }

        self.logger.info(
            f"Coverage calculated: {reflected}/{len(paths)} rays reflected, "
            f"avg range={avg_range:.1f} km"
        )

        return {
            'frequency_mhz': freq_mhz,
            'ray_paths': paths,
            'coverage_summary': coverage_summary,
            'transmitter': {
                'latitude': self.tx_lat,
                'longitude': self.tx_lon,
                'altitude_km': self.tx_alt
            },
            'timestamp_utc': datetime.utcnow().isoformat() + 'Z'
        }

    def get_configuration(self) -> Dict[str, Any]:
        """
        Get current propagation service configuration.

        Returns:
            Configuration dictionary
        """
        return {
            'transmitter': {
                'latitude': self.tx_lat,
                'longitude': self.tx_lon,
                'altitude_km': self.tx_alt
            },
            'frequency_scan': {
                'min_mhz': self.freq_min,
                'max_mhz': self.freq_max,
                'step_mhz': self.freq_step
            },
            'ray_geometry': {
                'elevation_min': self.elevation_min,
                'elevation_max': self.elevation_max,
                'elevation_step': self.elevation_step,
                'azimuth_step': self.azimuth_step
            },
            'thresholds': {
                'absorption_db': self.absorption_threshold_db,
                'snr_db': self.snr_threshold_db
            },
            'tracer_initialized': self.tracer is not None
        }
