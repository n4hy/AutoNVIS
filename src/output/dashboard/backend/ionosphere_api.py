"""
Ionosphere API Endpoints

Provides REST API endpoints for ionospheric data visualization:
- Electron density grids
- Horizontal slices and vertical profiles
- foF2, hmF2, TEC maps
- Grid metadata and statistics
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, Dict, Any, List
import numpy as np

from .state_manager import DashboardState
from .data_processing import (
    compute_fof2,
    compute_hmf2,
    compute_tec,
    extract_horizontal_slice,
    extract_vertical_profile,
    compute_grid_statistics,
    detect_ionospheric_layers
)

router = APIRouter(prefix="/api/ionosphere", tags=["ionosphere"])


def create_ionosphere_routes(state: DashboardState) -> APIRouter:
    """
    Create ionosphere API routes with state dependency.

    Args:
        state: Dashboard state manager

    Returns:
        Configured API router
    """

    @router.get("/grid/metadata")
    async def get_grid_metadata():
        """
        GET /api/ionosphere/grid/metadata
        Get metadata about the latest electron density grid.
        """
        metadata = state.get_grid_metadata()
        if metadata is None:
            raise HTTPException(status_code=404, detail="No grid data available")

        return metadata

    @router.get("/grid/full")
    async def get_full_grid(
        max_age_seconds: float = Query(1200.0, description="Maximum grid age in seconds")
    ):
        """
        GET /api/ionosphere/grid/full
        Get complete 3D electron density grid with coordinates.

        Warning: Large payload (~2-3 MB for 73×73×55 grid).
        Use slices/profiles for smaller data transfers.
        """
        grid_data = state.get_latest_grid(max_age_seconds)
        if grid_data is None:
            raise HTTPException(
                status_code=404,
                detail="No fresh grid available"
            )

        # Convert numpy arrays to lists for JSON serialization
        return {
            'ne_grid': grid_data['ne_grid'].tolist(),
            'lat': grid_data['lat'].tolist(),
            'lon': grid_data['lon'].tolist(),
            'alt': grid_data['alt'].tolist(),
            'metadata': {
                'cycle_id': grid_data['cycle_id'],
                'timestamp': grid_data['timestamp'],
                'quality': grid_data['quality'],
                'xray_flux': grid_data['xray_flux'],
                'effective_ssn': grid_data['effective_ssn'],
                'observations_used': grid_data['observations_used'],
                'filter_converged': grid_data['filter_converged']
            }
        }

    @router.get("/slice/horizontal")
    async def get_horizontal_slice(
        altitude_km: float = Query(..., description="Altitude for slice in km"),
        max_age_seconds: float = Query(1200.0, description="Maximum grid age in seconds")
    ):
        """
        GET /api/ionosphere/slice/horizontal?altitude_km=300
        Get horizontal slice of electron density at specified altitude.
        """
        grid_data = state.get_latest_grid(max_age_seconds)
        if grid_data is None:
            raise HTTPException(status_code=404, detail="No grid data available")

        lat_grid, lon_grid, ne_slice = extract_horizontal_slice(
            grid_data['ne_grid'],
            grid_data['lat'],
            grid_data['lon'],
            grid_data['alt'],
            altitude_km
        )

        # Find actual altitude used
        alt_idx = np.argmin(np.abs(grid_data['alt'] - altitude_km))
        actual_alt = grid_data['alt'][alt_idx]

        return {
            'latitude': lat_grid.tolist(),
            'longitude': lon_grid.tolist(),
            'ne_values': ne_slice.tolist(),
            'requested_altitude_km': altitude_km,
            'actual_altitude_km': float(actual_alt),
            'units': 'el/m³',
            'timestamp': grid_data['timestamp']
        }

    @router.get("/profile/vertical")
    async def get_vertical_profile(
        latitude: float = Query(..., description="Latitude in degrees"),
        longitude: float = Query(..., description="Longitude in degrees"),
        max_age_seconds: float = Query(1200.0, description="Maximum grid age in seconds")
    ):
        """
        GET /api/ionosphere/profile/vertical?latitude=40.0&longitude=-105.0
        Get vertical electron density profile at specified location.
        """
        grid_data = state.get_latest_grid(max_age_seconds)
        if grid_data is None:
            raise HTTPException(status_code=404, detail="No grid data available")

        alt_grid, ne_profile = extract_vertical_profile(
            grid_data['ne_grid'],
            grid_data['lat'],
            grid_data['lon'],
            grid_data['alt'],
            latitude,
            longitude
        )

        # Detect layers
        layers = detect_ionospheric_layers(ne_profile, alt_grid)

        # Find actual lat/lon used
        lat_idx = np.argmin(np.abs(grid_data['lat'] - latitude))
        lon_idx = np.argmin(np.abs(grid_data['lon'] - longitude))
        actual_lat = grid_data['lat'][lat_idx]
        actual_lon = grid_data['lon'][lon_idx]

        return {
            'altitude_km': alt_grid.tolist(),
            'ne_values': ne_profile.tolist(),
            'requested_latitude': latitude,
            'requested_longitude': longitude,
            'actual_latitude': float(actual_lat),
            'actual_longitude': float(actual_lon),
            'layers': layers,
            'units': 'el/m³',
            'timestamp': grid_data['timestamp']
        }

    @router.get("/parameters/fof2")
    async def get_fof2_map(
        max_age_seconds: float = Query(1200.0, description="Maximum grid age in seconds")
    ):
        """
        GET /api/ionosphere/parameters/fof2
        Get foF2 (critical frequency) map.

        foF2 is computed from peak electron density using:
        f_p = 8.98 * sqrt(N_e) MHz
        """
        grid_data = state.get_latest_grid(max_age_seconds)
        if grid_data is None:
            raise HTTPException(status_code=404, detail="No grid data available")

        fof2_map, hmf2_map = compute_fof2(grid_data['ne_grid'], grid_data['alt'])

        return {
            'latitude': grid_data['lat'].tolist(),
            'longitude': grid_data['lon'].tolist(),
            'fof2_mhz': fof2_map.tolist(),
            'fof2_min': float(np.min(fof2_map)),
            'fof2_max': float(np.max(fof2_map)),
            'fof2_mean': float(np.mean(fof2_map)),
            'units': 'MHz',
            'timestamp': grid_data['timestamp']
        }

    @router.get("/parameters/hmf2")
    async def get_hmf2_map(
        max_age_seconds: float = Query(1200.0, description="Maximum grid age in seconds")
    ):
        """
        GET /api/ionosphere/parameters/hmf2
        Get hmF2 (F2 peak height) map.
        """
        grid_data = state.get_latest_grid(max_age_seconds)
        if grid_data is None:
            raise HTTPException(status_code=404, detail="No grid data available")

        hmf2_map = compute_hmf2(grid_data['ne_grid'], grid_data['alt'])

        return {
            'latitude': grid_data['lat'].tolist(),
            'longitude': grid_data['lon'].tolist(),
            'hmf2_km': hmf2_map.tolist(),
            'hmf2_min': float(np.min(hmf2_map)),
            'hmf2_max': float(np.max(hmf2_map)),
            'hmf2_mean': float(np.mean(hmf2_map)),
            'units': 'km',
            'timestamp': grid_data['timestamp']
        }

    @router.get("/parameters/tec")
    async def get_tec_map(
        max_age_seconds: float = Query(1200.0, description="Maximum grid age in seconds")
    ):
        """
        GET /api/ionosphere/parameters/tec
        Get vertical TEC (Total Electron Content) map.

        TEC is computed by integrating electron density along altitude.
        Units: TECU (1 TECU = 10^16 el/m²)
        """
        grid_data = state.get_latest_grid(max_age_seconds)
        if grid_data is None:
            raise HTTPException(status_code=404, detail="No grid data available")

        tec_map = compute_tec(grid_data['ne_grid'], grid_data['alt'])

        return {
            'latitude': grid_data['lat'].tolist(),
            'longitude': grid_data['lon'].tolist(),
            'tec_tecu': tec_map.tolist(),
            'tec_min': float(np.min(tec_map)),
            'tec_max': float(np.max(tec_map)),
            'tec_mean': float(np.mean(tec_map)),
            'units': 'TECU',
            'timestamp': grid_data['timestamp']
        }

    @router.get("/statistics")
    async def get_grid_statistics(
        max_age_seconds: float = Query(1200.0, description="Maximum grid age in seconds")
    ):
        """
        GET /api/ionosphere/statistics
        Get statistical summary of electron density grid.
        """
        grid_data = state.get_latest_grid(max_age_seconds)
        if grid_data is None:
            raise HTTPException(status_code=404, detail="No grid data available")

        from .data_processing import compute_grid_statistics
        stats = compute_grid_statistics(grid_data['ne_grid'])

        return {
            **stats,
            'grid_shape': grid_data['ne_grid'].shape,
            'timestamp': grid_data['timestamp'],
            'cycle_id': grid_data['cycle_id']
        }

    @router.get("/history/grids")
    async def get_grid_history(
        limit: int = Query(100, description="Maximum number of entries to return")
    ):
        """
        GET /api/ionosphere/history/grids
        Get history of recent grids (metadata only, not full grids).
        """
        with state.grid_lock:
            history = list(state.grid_history)[-limit:]

        return {
            'count': len(history),
            'grids': history
        }

    return router
