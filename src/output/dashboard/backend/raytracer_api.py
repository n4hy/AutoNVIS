"""
Raytracer API Endpoints

Provides REST API endpoints for PHaRLAP ray tracing visualization:
- Homing algorithm results
- Winner triplets
- Ray path cross-section data
- Synthetic ionograms
- MUF/LUF/FOT time series
- Link budget analysis
- Coverage maps
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, Dict, Any, List

from .state_manager import DashboardState

router = APIRouter(prefix="/api/raytracer", tags=["raytracer"])


def create_raytracer_routes(state: DashboardState) -> APIRouter:
    """
    Create raytracer API routes with state dependency.

    Args:
        state: Dashboard state manager

    Returns:
        Configured API router
    """

    @router.get("/homing/latest")
    async def get_latest_homing_result(
        max_age_seconds: float = Query(1200.0, description="Maximum result age in seconds")
    ):
        """
        GET /api/raytracer/homing/latest
        Get latest homing algorithm result with winner triplets and statistics.
        """
        result = state.get_latest_homing_result(max_age_seconds)
        if result is None:
            raise HTTPException(
                status_code=404,
                detail="No fresh homing result available"
            )

        return result

    @router.get("/winners/latest")
    async def get_latest_winners(
        limit: int = Query(100, description="Maximum number of winners to return")
    ):
        """
        GET /api/raytracer/winners/latest
        Get latest winner triplets with SNR, mode, and hop count.
        """
        result = state.get_latest_homing_result()
        if result is None:
            return {'winners': [], 'count': 0}

        winners = result.get('winner_triplets', [])[:limit]
        return {
            'winners': winners,
            'count': len(winners),
            'total_available': len(result.get('winner_triplets', [])),
            'timestamp': result.get('timestamp'),
            'muf': result.get('muf'),
            'luf': result.get('luf'),
            'fot': result.get('fot')
        }

    @router.get("/winners/by_frequency")
    async def get_winners_by_frequency(
        freq_min: float = Query(2.0, description="Minimum frequency in MHz"),
        freq_max: float = Query(30.0, description="Maximum frequency in MHz"),
        hours: int = Query(24, description="Time window in hours")
    ):
        """
        GET /api/raytracer/winners/by_frequency
        Filter winner triplets by frequency range.
        """
        winners = state.get_winner_triplets(
            hours=hours,
            frequency_min=freq_min,
            frequency_max=freq_max
        )

        return {
            'winners': winners,
            'count': len(winners),
            'frequency_range': {'min': freq_min, 'max': freq_max},
            'hours': hours
        }

    @router.get("/ray_paths")
    async def get_ray_paths():
        """
        GET /api/raytracer/ray_paths
        Get ray path data for cross-section visualization.

        Returns ground range and altitude arrays for each traced ray,
        organized by frequency and mode (O/X).
        """
        ray_paths = state.get_ray_paths()
        if ray_paths is None:
            # Return empty structure if no data
            return {
                'paths': [],
                'ionosphere_profile': None,
                'tx_position': None,
                'rx_position': None
            }

        return ray_paths

    @router.get("/frequencies/current")
    async def get_current_frequencies():
        """
        GET /api/raytracer/frequencies/current
        Get current MUF/LUF/FOT from raytracer.
        """
        result = state.get_latest_homing_result()
        if result is None:
            return {
                'muf': None,
                'luf': None,
                'fot': None,
                'timestamp': None,
                'num_winners': 0
            }

        return {
            'muf': result.get('muf'),
            'luf': result.get('luf'),
            'fot': result.get('fot'),
            'timestamp': result.get('timestamp'),
            'num_winners': result.get('num_winners', 0),
            'great_circle_range_km': result.get('great_circle_range_km'),
            'computation_time_s': result.get('computation_time_s')
        }

    @router.get("/frequencies/history")
    async def get_frequency_history(
        hours: int = Query(24, description="Time window in hours")
    ):
        """
        GET /api/raytracer/frequencies/history
        Get MUF/LUF/FOT time series from raytracer.
        """
        history = state.get_raytracer_frequency_history(hours)

        # Organize into separate arrays for plotting
        muf_data = []
        luf_data = []
        fot_data = []

        for entry in history:
            timestamp = entry.get('timestamp')
            if entry.get('muf') is not None:
                muf_data.append({'timestamp': timestamp, 'value': entry['muf']})
            if entry.get('luf') is not None:
                luf_data.append({'timestamp': timestamp, 'value': entry['luf']})
            if entry.get('fot') is not None:
                fot_data.append({'timestamp': timestamp, 'value': entry['fot']})

        return {
            'muf': muf_data,
            'luf': luf_data,
            'fot': fot_data,
            'hours': hours,
            'count': len(history)
        }

    @router.get("/ionogram/latest")
    async def get_latest_ionogram():
        """
        GET /api/raytracer/ionogram/latest
        Get synthetic ionogram data (group delay vs frequency).

        Returns O-mode and X-mode traces for ionogram display.
        """
        ionogram = state.get_ionogram()
        if ionogram is None:
            return {
                'o_mode': {'frequencies': [], 'delays': []},
                'x_mode': {'frequencies': [], 'delays': []},
                'timestamp': None
            }

        return ionogram

    @router.get("/link_budget")
    async def get_link_budget():
        """
        GET /api/raytracer/link_budget
        Get SNR and path loss breakdown for current propagation paths.

        Returns link budget components for each winner triplet.
        """
        result = state.get_latest_homing_result()
        if result is None:
            return {'link_budgets': [], 'count': 0}

        link_budgets = []
        for winner in result.get('winner_triplets', []):
            if winner.get('snr_db') is not None:
                link_budgets.append({
                    'frequency_mhz': winner.get('frequency_mhz'),
                    'mode': winner.get('mode'),
                    'hop_count': winner.get('hop_count'),
                    'snr_db': winner.get('snr_db'),
                    'signal_strength_dbm': winner.get('signal_strength_dbm'),
                    'path_loss_db': winner.get('path_loss_db'),
                    'elevation_deg': winner.get('elevation_deg'),
                    'ground_range_km': winner.get('ground_range_km'),
                    'reflection_height_km': winner.get('reflection_height_km')
                })

        return {
            'link_budgets': link_budgets,
            'count': len(link_budgets),
            'timestamp': result.get('timestamp')
        }

    @router.get("/coverage_map")
    async def get_coverage_map():
        """
        GET /api/raytracer/coverage_map
        Get winner triplet landing points for geographic display.

        Returns lat/lon landing positions with mode and SNR data.
        """
        result = state.get_latest_homing_result()
        if result is None:
            return {
                'points': [],
                'tx_position': None,
                'rx_position': None,
                'timestamp': None
            }

        points = []
        for winner in result.get('winner_triplets', []):
            points.append({
                'lat': winner.get('landing_lat'),
                'lon': winner.get('landing_lon'),
                'frequency_mhz': winner.get('frequency_mhz'),
                'mode': winner.get('mode'),
                'snr_db': winner.get('snr_db'),
                'hop_count': winner.get('hop_count'),
                'landing_error_km': winner.get('landing_error_km'),
                'elevation_deg': winner.get('elevation_deg')
            })

        return {
            'points': points,
            'tx_position': result.get('tx_position'),
            'rx_position': result.get('rx_position'),
            'great_circle_range_km': result.get('great_circle_range_km'),
            'timestamp': result.get('timestamp')
        }

    @router.get("/statistics")
    async def get_raytracer_statistics():
        """
        GET /api/raytracer/statistics
        Get raytracer performance and state statistics.
        """
        stats = state.get_raytracer_statistics()
        return stats

    return router
