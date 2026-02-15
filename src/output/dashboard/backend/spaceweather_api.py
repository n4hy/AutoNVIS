"""
Space Weather API Endpoints

Provides REST API endpoints for space weather monitoring:
- X-ray flux with flare classification
- Solar wind parameters
- Geomagnetic indices
- Autonomous mode history
- System alerts
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, Dict, Any, List

from .state_manager import DashboardState

router = APIRouter(prefix="/api/spaceweather", tags=["spaceweather"])


def create_spaceweather_routes(state: DashboardState) -> APIRouter:
    """
    Create space weather API routes with state dependency.

    Args:
        state: Dashboard state manager

    Returns:
        Configured API router
    """

    @router.get("/xray/history")
    async def get_xray_history(
        hours: int = Query(24, description="Number of hours of history to return")
    ):
        """
        GET /api/spaceweather/xray/history?hours=24
        Get X-ray flux history with flare classifications.

        Flare classes:
        - C-class: 1e-6 to 1e-5 W/m²
        - M-class: 1e-5 to 1e-4 W/m² (affects polar regions)
        - X-class: >= 1e-4 W/m² (major flares, global impact)
        """
        history = state.get_xray_history(hours=hours)

        # Add flare classification
        for entry in history:
            flux = entry.get('flux_short', entry.get('flux_long', entry.get('flux_wm2', 0.0)))
            if flux >= 1e-4:
                entry['flare_class'] = 'X'
                entry['flare_magnitude'] = flux / 1e-4
            elif flux >= 1e-5:
                entry['flare_class'] = 'M'
                entry['flare_magnitude'] = flux / 1e-5
            elif flux >= 1e-6:
                entry['flare_class'] = 'C'
                entry['flare_magnitude'] = flux / 1e-6
            elif flux >= 1e-7:
                entry['flare_class'] = 'B'
                entry['flare_magnitude'] = flux / 1e-7
            else:
                entry['flare_class'] = 'A'
                entry['flare_magnitude'] = flux / 1e-8

        return {
            'hours': hours,
            'count': len(history),
            'data': history,
            'units': 'W/m²'
        }

    @router.get("/xray/latest")
    async def get_xray_latest():
        """
        GET /api/spaceweather/xray/latest
        Get most recent X-ray flux measurement.
        """
        history = state.get_xray_history(hours=1)

        if not history:
            raise HTTPException(
                status_code=404,
                detail="No X-ray data available"
            )

        latest = history[-1]

        # Classify flare
        flux = latest.get('flux_short', latest.get('flux_long', latest.get('flux_wm2', 0.0)))
        if flux >= 1e-4:
            flare_class = 'X'
            magnitude = flux / 1e-4
        elif flux >= 1e-5:
            flare_class = 'M'
            magnitude = flux / 1e-5
        elif flux >= 1e-6:
            flare_class = 'C'
            magnitude = flux / 1e-6
        elif flux >= 1e-7:
            flare_class = 'B'
            magnitude = flux / 1e-7
        else:
            flare_class = 'A'
            magnitude = flux / 1e-8

        return {
            **latest,
            'flare_class': flare_class,
            'flare_magnitude': magnitude
        }

    @router.get("/solar_wind/latest")
    async def get_solar_wind_latest():
        """
        GET /api/spaceweather/solar_wind/latest
        Get latest solar wind parameters.

        Parameters:
        - Speed (km/s): Typical 300-800 km/s, CMEs can exceed 1000 km/s
        - Density (protons/cm³): Typical 5-10 p/cm³
        - Bz (nT): North/south component of IMF, critical for geomagnetic storms
        - Bt (nT): Total IMF magnitude
        """
        latest = state.get_solar_wind_latest()

        if latest is None:
            raise HTTPException(
                status_code=404,
                detail="No solar wind data available"
            )

        return latest

    @router.get("/solar_wind/history")
    async def get_solar_wind_history(
        hours: int = Query(24, description="Number of hours of history to return")
    ):
        """
        GET /api/spaceweather/solar_wind/history?hours=24
        Get solar wind parameter history.
        """
        from datetime import datetime, timedelta

        cutoff = datetime.utcnow() - timedelta(hours=hours)

        with state.spaceweather_lock:
            history = [
                entry for entry in state.solar_wind_history
                if datetime.fromisoformat(entry['timestamp'].rstrip('Z')) > cutoff
            ]

        return {
            'hours': hours,
            'count': len(history),
            'data': history
        }

    @router.get("/mode/current")
    async def get_current_mode():
        """
        GET /api/spaceweather/mode/current
        Get current autonomous mode (QUIET or SHOCK).

        Modes:
        - QUIET: Normal ionospheric conditions
        - SHOCK: Disturbed conditions (X-ray flux > M1)
        """
        with state.spaceweather_lock:
            mode = state.current_mode

        return {
            'mode': mode,
            'description': (
                'Normal ionospheric conditions' if mode == 'QUIET'
                else 'Disturbed ionospheric conditions'
            )
        }

    @router.get("/mode/history")
    async def get_mode_history(
        hours: int = Query(24, description="Number of hours of history to return")
    ):
        """
        GET /api/spaceweather/mode/history?hours=24
        Get history of autonomous mode changes.
        """
        history = state.get_mode_history(hours=hours)

        return {
            'hours': hours,
            'count': len(history),
            'current_mode': state.current_mode,
            'data': history
        }

    @router.get("/alerts/recent")
    async def get_recent_alerts(
        hours: int = Query(24, description="Number of hours of history to return"),
        severity: Optional[str] = Query(None, description="Filter by severity (info, warning, error, critical)")
    ):
        """
        GET /api/spaceweather/alerts/recent?hours=24&severity=warning
        Get recent system alerts.
        """
        alerts = state.get_recent_alerts(hours=hours)

        # Filter by severity if specified
        if severity:
            alerts = [
                alert for alert in alerts
                if alert.get('severity', '').lower() == severity.lower()
            ]

        return {
            'hours': hours,
            'count': len(alerts),
            'severity_filter': severity,
            'data': alerts
        }

    @router.get("/statistics")
    async def get_spaceweather_statistics():
        """
        GET /api/spaceweather/statistics
        Get summary statistics for space weather data.
        """
        import numpy as np

        # X-ray statistics
        xray_history = state.get_xray_history(hours=24)
        xray_fluxes = [entry.get('flux_short', entry.get('flux_long', entry.get('flux_wm2', 0.0))) for entry in xray_history]

        # Count flares by class
        flare_counts = {'A': 0, 'B': 0, 'C': 0, 'M': 0, 'X': 0}
        max_flare = {'class': 'A', 'magnitude': 0.0, 'timestamp': None}

        for entry in xray_history:
            flux = entry.get('flux_short', entry.get('flux_long', entry.get('flux_wm2', 0.0)))
            timestamp = entry.get('timestamp')

            if flux >= 1e-4:
                flare_class = 'X'
                magnitude = flux / 1e-4
                flare_counts['X'] += 1
            elif flux >= 1e-5:
                flare_class = 'M'
                magnitude = flux / 1e-5
                flare_counts['M'] += 1
            elif flux >= 1e-6:
                flare_class = 'C'
                magnitude = flux / 1e-6
                flare_counts['C'] += 1
            elif flux >= 1e-7:
                flare_class = 'B'
                magnitude = flux / 1e-7
                flare_counts['B'] += 1
            else:
                flare_class = 'A'
                magnitude = flux / 1e-8
                flare_counts['A'] += 1

            # Track maximum flare
            if flux > max_flare['magnitude']:
                max_flare = {
                    'class': flare_class,
                    'magnitude': magnitude,
                    'flux_wm2': flux,
                    'timestamp': timestamp
                }

        # Mode statistics
        mode_history = state.get_mode_history(hours=24)
        mode_durations = {'QUIET': 0, 'SHOCK': 0}

        # Compute time in each mode
        if mode_history:
            from datetime import datetime, timedelta

            for i in range(len(mode_history)):
                mode = mode_history[i]['mode']
                start_time = datetime.fromisoformat(mode_history[i]['timestamp'])

                if i + 1 < len(mode_history):
                    end_time = datetime.fromisoformat(mode_history[i + 1]['timestamp'])
                else:
                    end_time = datetime.utcnow()

                duration = (end_time - start_time).total_seconds() / 3600.0  # hours
                if mode in mode_durations:
                    mode_durations[mode] += duration

        stats = {
            'xray': {
                'flux_min_wm2': float(np.min(xray_fluxes)) if xray_fluxes else None,
                'flux_max_wm2': float(np.max(xray_fluxes)) if xray_fluxes else None,
                'flux_mean_wm2': float(np.mean(xray_fluxes)) if xray_fluxes else None,
                'measurements_24h': len(xray_fluxes)
            },
            'flares_24h': flare_counts,
            'max_flare_24h': max_flare,
            'mode': {
                'current': state.current_mode,
                'changes_24h': len(mode_history),
                'duration_hours': mode_durations
            },
            'alerts_24h': len(state.get_recent_alerts(hours=24))
        }

        return stats

    return router
