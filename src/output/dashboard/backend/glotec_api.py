"""
GloTEC API Endpoints

Provides REST API endpoints for GloTEC global TEC map data:
- Latest TEC map with grid data
- TEC statistics history
- TEC value at specific lat/lon point
- Grid metadata
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from .state_manager import DashboardState

router = APIRouter(prefix="/api/glotec", tags=["glotec"])


def create_glotec_routes(state: DashboardState) -> APIRouter:
    """
    Create GloTEC API routes with state dependency.

    Args:
        state: Dashboard state manager

    Returns:
        Configured API router
    """

    @router.get("/latest")
    async def get_latest_glotec(
        include_grid: bool = Query(True, description="Include full grid arrays in response")
    ):
        """
        GET /api/glotec/latest?include_grid=true
        Get latest GloTEC map.

        Returns global TEC map with:
        - Grid arrays (tec, anomaly, hmF2, NmF2, quality)
        - Statistics (mean, max, min, std)
        - Metadata (resolution, source)
        """
        glotec = state.get_latest_glotec()

        if glotec is None:
            raise HTTPException(
                status_code=404,
                detail="No GloTEC data available"
            )

        if include_grid:
            return glotec
        else:
            # Return only statistics and metadata, not the large grid arrays
            return {
                'timestamp': glotec.get('timestamp'),
                'statistics': glotec.get('statistics', {}),
                'metadata': glotec.get('metadata', {})
            }

    @router.get("/statistics")
    async def get_glotec_statistics():
        """
        GET /api/glotec/statistics
        Get statistics from latest GloTEC map without grid data.
        """
        stats = state.get_glotec_statistics()

        if stats is None:
            raise HTTPException(
                status_code=404,
                detail="No GloTEC data available"
            )

        return stats

    @router.get("/history")
    async def get_glotec_history(
        hours: int = Query(24, description="Number of hours of history to return")
    ):
        """
        GET /api/glotec/history?hours=24
        Get GloTEC statistics history.

        Returns time series of TEC statistics (not full grids).
        """
        history = state.get_glotec_history(hours=hours)

        return {
            'hours': hours,
            'count': len(history),
            'data': history,
            'units': 'TECU'
        }

    @router.get("/point")
    async def get_glotec_point(
        lat: float = Query(..., description="Latitude in degrees (-90 to 90)"),
        lon: float = Query(..., description="Longitude in degrees (-180 to 180)")
    ):
        """
        GET /api/glotec/point?lat=40.0&lon=-105.0
        Get TEC value at a specific geographic point.

        Uses nearest-neighbor interpolation from the GloTEC grid.
        """
        # Validate coordinates
        if not -90 <= lat <= 90:
            raise HTTPException(
                status_code=400,
                detail="Latitude must be between -90 and 90 degrees"
            )

        if not -180 <= lon <= 180:
            raise HTTPException(
                status_code=400,
                detail="Longitude must be between -180 and 180 degrees"
            )

        point_data = state.get_glotec_point(lat, lon)

        if point_data is None:
            raise HTTPException(
                status_code=404,
                detail="No GloTEC data available or point outside grid"
            )

        return {
            'requested': {'lat': lat, 'lon': lon},
            'nearest_grid_point': {
                'lat': point_data.get('lat'),
                'lon': point_data.get('lon')
            },
            'values': {
                'tec': point_data.get('tec'),
                'anomaly': point_data.get('anomaly'),
                'hmF2': point_data.get('hmF2'),
                'NmF2': point_data.get('NmF2')
            },
            'timestamp': point_data.get('timestamp'),
            'units': {
                'tec': 'TECU',
                'anomaly': 'TECU',
                'hmF2': 'km',
                'NmF2': 'el/m3'
            }
        }

    @router.get("/metadata")
    async def get_glotec_metadata():
        """
        GET /api/glotec/metadata
        Get metadata about available GloTEC data.
        """
        stats = state.get_glotec_statistics()

        if stats is None:
            return {
                'available': False,
                'message': 'No GloTEC data has been received yet'
            }

        return {
            'available': True,
            'timestamp': stats.get('timestamp'),
            'age_seconds': stats.get('age_seconds'),
            'metadata': stats.get('metadata', {}),
            'data_source': 'NOAA SWPC GloTEC',
            'update_cadence_minutes': 10,
            'grid_resolution': {
                'lat_deg': 2.5,
                'lon_deg': 5.0
            }
        }

    @router.get("/summary")
    async def get_glotec_summary():
        """
        GET /api/glotec/summary
        Get a comprehensive summary of current GloTEC state.
        """
        stats = state.get_glotec_statistics()
        history = state.get_glotec_history(hours=24)

        if stats is None:
            raise HTTPException(
                status_code=404,
                detail="No GloTEC data available"
            )

        # Calculate 24h statistics from history
        if history:
            tec_means = [h['statistics'].get('tec_mean') for h in history if h.get('statistics', {}).get('tec_mean') is not None]
            history_stats = {
                'tec_mean_24h': sum(tec_means) / len(tec_means) if tec_means else None,
                'tec_min_24h': min(tec_means) if tec_means else None,
                'tec_max_24h': max(tec_means) if tec_means else None,
                'maps_received_24h': len(history)
            }
        else:
            history_stats = {
                'tec_mean_24h': None,
                'tec_min_24h': None,
                'tec_max_24h': None,
                'maps_received_24h': 0
            }

        return {
            'current': stats,
            'history_24h': history_stats,
            'total_maps_received': state.stats.get('glotec_maps_received', 0)
        }

    return router
