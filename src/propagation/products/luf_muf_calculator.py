"""
LUF/MUF Calculator

Calculates Lowest Usable Frequency and Maximum Usable Frequency
for NVIS propagation from ray tracing results.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging


class LUFMUFCalculator:
    """
    Calculate LUF and MUF from ray tracing results.

    LUF: Minimum frequency with acceptable D-region absorption
    MUF: Maximum frequency that reflects from ionosphere
    FOT: Optimum frequency (typically 0.85 * MUF)
    """

    def __init__(
        self,
        absorption_threshold_db: float = 50.0,
        snr_threshold_db: float = 10.0
    ):
        """
        Initialize calculator.

        Args:
            absorption_threshold_db: Maximum acceptable absorption (dB)
            snr_threshold_db: Minimum acceptable SNR (dB)
        """
        self.logger = logging.getLogger(__name__)
        self.absorption_threshold = absorption_threshold_db
        self.snr_threshold = snr_threshold_db

    def calculate_from_coverage(
        self,
        coverage_map: Dict[float, List[Dict]],
        tx_power_dbm: float = 50.0
    ) -> Dict:
        """
        Calculate LUF/MUF from multi-frequency coverage map.

        Args:
            coverage_map: Dict mapping frequencies to ray paths
            tx_power_dbm: Transmitter power (dBm)

        Returns:
            Dictionary with LUF, MUF, FOT, and analysis
        """
        frequencies = sorted(coverage_map.keys())

        # Find LUF (lowest frequency with usable paths)
        luf = None
        for freq in frequencies:
            paths = coverage_map[freq]

            # Check if any path has acceptable absorption
            usable_paths = [
                p for p in paths
                if p['reflected'] and
                   p['absorption_db'] < self.absorption_threshold and
                   not p['absorbed']
            ]

            if usable_paths:
                luf = freq
                break

        # Find MUF (highest frequency that reflects)
        muf = None
        for freq in reversed(frequencies):
            paths = coverage_map[freq]

            # Check if any path reflects
            reflected_paths = [p for p in paths if p['reflected']]

            if reflected_paths:
                muf = freq
                break

        # Calculate FOT (Frequency of Optimum Transmission)
        fot = 0.85 * muf if muf else None

        # Blackout condition
        blackout = False
        if luf and muf:
            blackout = luf > muf
        elif not luf or not muf:
            blackout = True

        # Calculate coverage statistics
        coverage_stats = self._calculate_coverage_stats(
            coverage_map, luf, muf, tx_power_dbm
        )

        return {
            'luf_mhz': luf,
            'muf_mhz': muf,
            'fot_mhz': fot,
            'usable_range_mhz': (luf, muf) if (luf and muf and not blackout) else None,
            'blackout': blackout,
            'coverage_stats': coverage_stats
        }

    def calculate_spatial_luf_muf(
        self,
        ray_tracer,
        tx_lat: float,
        tx_lon: float,
        grid_resolution: int = 50,
        freq_range: Tuple[float, float] = (2.0, 15.0),
        freq_step: float = 1.0
    ) -> Dict:
        """
        Calculate spatial LUF/MUF grid.

        Args:
            ray_tracer: RayTracer instance
            tx_lat, tx_lon: Transmitter position
            grid_resolution: Number of grid points
            freq_range: (min_freq, max_freq) in MHz
            freq_step: Frequency step in MHz

        Returns:
            Dictionary with 2D LUF/MUF grids
        """
        self.logger.info(f"Calculating spatial LUF/MUF grid ({grid_resolution}x{grid_resolution})")

        # Create spatial grid (±500 km)
        lat_range = np.linspace(tx_lat - 5, tx_lat + 5, grid_resolution)
        lon_range = np.linspace(tx_lon - 5, tx_lon + 5, grid_resolution)

        luf_grid = np.full((grid_resolution, grid_resolution), np.nan)
        muf_grid = np.full((grid_resolution, grid_resolution), np.nan)
        blackout_grid = np.zeros((grid_resolution, grid_resolution), dtype=bool)

        frequencies = np.arange(freq_range[0], freq_range[1] + freq_step, freq_step)

        for i, lat in enumerate(lat_range):
            for j, lon in enumerate(lon_range):
                # Calculate azimuth to this point
                dlat = lat - tx_lat
                dlon = lon - tx_lon
                azim = np.degrees(np.arctan2(dlon, dlat))
                azim = (azim + 360) % 360

                # Trace rays for different frequencies
                reflected_freqs = []
                usable_freqs = []

                for freq in frequencies:
                    path = ray_tracer.trace_ray(tx_lat, tx_lon, 85.0, azim, freq)

                    if path['reflected']:
                        reflected_freqs.append(freq)

                        if path['absorption_db'] < self.absorption_threshold:
                            usable_freqs.append(freq)

                # Set LUF/MUF for this grid point
                if usable_freqs:
                    luf_grid[i, j] = min(usable_freqs)
                if reflected_freqs:
                    muf_grid[i, j] = max(reflected_freqs)

                # Blackout if LUF > MUF or no usable frequencies
                if not usable_freqs or (usable_freqs and reflected_freqs and
                                       min(usable_freqs) > max(reflected_freqs)):
                    blackout_grid[i, j] = True

        return {
            'luf_grid': luf_grid,
            'muf_grid': muf_grid,
            'blackout_grid': blackout_grid,
            'lat_range': lat_range,
            'lon_range': lon_range
        }

    def _calculate_coverage_stats(
        self,
        coverage_map: Dict[float, List[Dict]],
        luf: Optional[float],
        muf: Optional[float],
        tx_power_dbm: float
    ) -> Dict:
        """Calculate statistics about coverage quality."""
        stats = {}

        for freq, paths in coverage_map.items():
            reflected = [p for p in paths if p['reflected']]
            usable = [
                p for p in reflected
                if p['absorption_db'] < self.absorption_threshold
            ]

            stats[freq] = {
                'total_rays': len(paths),
                'reflected_rays': len(reflected),
                'usable_rays': len(usable),
                'reflection_rate': len(reflected) / len(paths) if paths else 0.0,
                'usability_rate': len(usable) / len(paths) if paths else 0.0,
                'avg_absorption_db': np.mean([p['absorption_db'] for p in reflected]) if reflected else np.nan,
                'avg_ground_range_km': np.mean([p['ground_range'] for p in reflected]) if reflected else np.nan
            }

        return stats


class FrequencyRecommender:
    """
    Recommend operating frequencies based on LUF/MUF analysis.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def recommend_frequencies(
        self,
        luf_mhz: float,
        muf_mhz: float,
        num_frequencies: int = 5,
        strategy: str = 'distributed'
    ) -> List[Dict]:
        """
        Recommend operating frequencies.

        Args:
            luf_mhz: Lowest Usable Frequency
            muf_mhz: Maximum Usable Frequency
            num_frequencies: Number of frequencies to recommend
            strategy: 'distributed', 'optimal', or 'conservative'

        Returns:
            List of recommended frequencies with confidence scores
        """
        if luf_mhz > muf_mhz:
            self.logger.warning("Blackout condition: LUF > MUF")
            return []

        # Calculate FOT (0.85 * MUF)
        fot = 0.85 * muf_mhz

        recommendations = []

        if strategy == 'optimal':
            # Focus around FOT
            frequencies = np.linspace(
                max(luf_mhz, fot - 1.0),
                min(muf_mhz, fot + 1.0),
                num_frequencies
            )
            base_confidence = 0.9

        elif strategy == 'conservative':
            # Stay well below MUF
            frequencies = np.linspace(
                luf_mhz,
                min(muf_mhz * 0.7, fot),
                num_frequencies
            )
            base_confidence = 0.95

        else:  # distributed
            # Spread across usable range
            frequencies = np.linspace(luf_mhz, muf_mhz * 0.9, num_frequencies)
            base_confidence = 0.8

        for freq in frequencies:
            # Calculate confidence score
            # Higher near FOT, lower near LUF/MUF boundaries
            dist_from_fot = abs(freq - fot) / (muf_mhz - luf_mhz)
            confidence = base_confidence * (1.0 - 0.5 * dist_from_fot)

            recommendations.append({
                'frequency_mhz': float(freq),
                'confidence': float(np.clip(confidence, 0.0, 1.0)),
                'distance_from_fot_mhz': abs(freq - fot),
                'margin_to_muf_mhz': muf_mhz - freq,
                'margin_to_luf_mhz': freq - luf_mhz
            })

        # Sort by confidence
        recommendations.sort(key=lambda x: x['confidence'], reverse=True)

        return recommendations


def calculate_luf_muf_trends(
    historical_data: List[Dict],
    window_hours: int = 24
) -> Dict:
    """
    Analyze LUF/MUF trends over time.

    Args:
        historical_data: List of LUF/MUF measurements with timestamps
        window_hours: Analysis window in hours

    Returns:
        Trend analysis dictionary
    """
    if len(historical_data) < 2:
        return {'trend': 'insufficient_data'}

    luf_values = [d['luf_mhz'] for d in historical_data if d.get('luf_mhz')]
    muf_values = [d['muf_mhz'] for d in historical_data if d.get('muf_mhz')]

    if not luf_values or not muf_values:
        return {'trend': 'no_data'}

    # Calculate trends
    luf_trend = np.polyfit(range(len(luf_values)), luf_values, 1)[0]
    muf_trend = np.polyfit(range(len(muf_values)), muf_values, 1)[0]

    # Calculate statistics
    return {
        'luf_mean': np.mean(luf_values),
        'luf_std': np.std(luf_values),
        'luf_trend_mhz_per_hour': luf_trend,
        'luf_current': luf_values[-1],
        'muf_mean': np.mean(muf_values),
        'muf_std': np.std(muf_values),
        'muf_trend_mhz_per_hour': muf_trend,
        'muf_current': muf_values[-1],
        'usable_bandwidth_mhz': muf_values[-1] - luf_values[-1],
        'trend': 'improving' if (muf_trend > 0 and luf_trend < 0) else
                'degrading' if (muf_trend < 0 and luf_trend > 0) else
                'stable'
    }
