"""
Propagation API Endpoints

Provides REST API endpoints for propagation prediction visualization:
- LUF/MUF/FOT frequency plans
- Coverage maps with SNR
- Ray path trajectories
- Time series data
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, Dict, Any, List

from .state_manager import DashboardState

router = APIRouter(prefix="/api/propagation", tags=["propagation"])


def create_propagation_routes(state: DashboardState) -> APIRouter:
    """
    Create propagation API routes with state dependency.

    Args:
        state: Dashboard state manager

    Returns:
        Configured API router
    """

    @router.get("/frequency_plan/latest")
    async def get_latest_frequency_plan():
        """
        GET /api/propagation/frequency_plan/latest
        Get the most recent frequency plan (LUF/MUF/FOT).
        """
        plan = state.get_frequency_plan()
        if plan is None:
            raise HTTPException(
                status_code=404,
                detail="No frequency plan available"
            )

        return plan

    @router.get("/frequency_plan/history")
    async def get_frequency_plan_history(
        hours: int = Query(24, description="Number of hours of history to return")
    ):
        """
        GET /api/propagation/frequency_plan/history?hours=24
        Get LUF/MUF/FOT history over specified time period.
        """
        history = state.get_luf_muf_fot_history(hours=hours)

        return {
            'hours': hours,
            'luf_count': len(history['luf']),
            'muf_count': len(history['muf']),
            'fot_count': len(history['fot']),
            'luf': history['luf'],
            'muf': history['muf'],
            'fot': history['fot']
        }

    @router.get("/coverage_map/latest")
    async def get_latest_coverage_map():
        """
        GET /api/propagation/coverage_map/latest
        Get the most recent coverage map.
        """
        with state.propagation_lock:
            coverage = state.latest_coverage_map

        if coverage is None:
            raise HTTPException(
                status_code=404,
                detail="No coverage map available"
            )

        return coverage

    @router.get("/luf")
    async def get_luf_data(
        hours: int = Query(24, description="Hours of history")
    ):
        """
        GET /api/propagation/luf?hours=24
        Get Lowest Usable Frequency (LUF) time series.
        """
        history = state.get_luf_muf_fot_history(hours=hours)

        return {
            'parameter': 'LUF',
            'units': 'MHz',
            'description': 'Lowest Usable Frequency',
            'data': history['luf']
        }

    @router.get("/muf")
    async def get_muf_data(
        hours: int = Query(24, description="Hours of history")
    ):
        """
        GET /api/propagation/muf?hours=24
        Get Maximum Usable Frequency (MUF) time series.
        """
        history = state.get_luf_muf_fot_history(hours=hours)

        return {
            'parameter': 'MUF',
            'units': 'MHz',
            'description': 'Maximum Usable Frequency',
            'data': history['muf']
        }

    @router.get("/fot")
    async def get_fot_data(
        hours: int = Query(24, description="Hours of history")
    ):
        """
        GET /api/propagation/fot?hours=24
        Get Frequency of Optimum Traffic (FOT) time series.

        FOT is typically 85% of MUF for reliable communication.
        """
        history = state.get_luf_muf_fot_history(hours=hours)

        return {
            'parameter': 'FOT',
            'units': 'MHz',
            'description': 'Frequency of Optimum Traffic (85% MUF)',
            'data': history['fot']
        }

    @router.get("/statistics")
    async def get_propagation_statistics():
        """
        GET /api/propagation/statistics
        Get summary statistics for propagation predictions.
        """
        plan = state.get_frequency_plan()
        history = state.get_luf_muf_fot_history(hours=24)

        if plan is None:
            raise HTTPException(status_code=404, detail="No data available")

        # Compute statistics from history
        import numpy as np

        luf_values = [entry['value'] for entry in history['luf']]
        muf_values = [entry['value'] for entry in history['muf']]
        fot_values = [entry['value'] for entry in history['fot']]

        stats = {
            'latest': {
                'luf_mhz': plan.get('luf_mhz'),
                'muf_mhz': plan.get('muf_mhz'),
                'fot_mhz': plan.get('fot_mhz'),
                'timestamp': plan.get('timestamp')
            }
        }

        if luf_values:
            stats['luf_24h'] = {
                'min': float(np.min(luf_values)),
                'max': float(np.max(luf_values)),
                'mean': float(np.mean(luf_values)),
                'std': float(np.std(luf_values)),
                'count': len(luf_values)
            }

        if muf_values:
            stats['muf_24h'] = {
                'min': float(np.min(muf_values)),
                'max': float(np.max(muf_values)),
                'mean': float(np.mean(muf_values)),
                'std': float(np.std(muf_values)),
                'count': len(muf_values)
            }

        if fot_values:
            stats['fot_24h'] = {
                'min': float(np.min(fot_values)),
                'max': float(np.max(fot_values)),
                'mean': float(np.mean(fot_values)),
                'std': float(np.std(fot_values)),
                'count': len(fot_values)
            }

        return stats

    return router
