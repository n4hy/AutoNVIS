"""
Control API Endpoints

Provides REST API endpoints for system control:
- Service management (start/stop/status)
- Filter control (trigger cycles, adjust parameters)
- Data source configuration
- System health monitoring
"""

from fastapi import APIRouter, HTTPException, Query, Body
from typing import Optional, Dict, Any, List
from pydantic import BaseModel

from .state_manager import DashboardState
from src.common.message_queue import MessageQueueClient, Topics

router = APIRouter(prefix="/api/control", tags=["control"])


# Request models
class ServiceControlRequest(BaseModel):
    """Request to start/stop a service"""
    service_name: str
    action: str  # 'start', 'stop', 'restart'


class FilterCycleRequest(BaseModel):
    """Request to trigger a filter cycle"""
    force: bool = False
    reason: str = "Manual trigger from dashboard"


class FilterParametersRequest(BaseModel):
    """Request to update filter parameters"""
    process_noise_ne: Optional[float] = None
    observation_noise_tec: Optional[float] = None
    covariance_inflation: Optional[float] = None


class ModeSwitchRequest(BaseModel):
    """Request to force mode switch"""
    mode: str  # 'QUIET' or 'SHOCK'
    reason: str = "Manual override from dashboard"


class DataSourceToggleRequest(BaseModel):
    """Request to enable/disable data source"""
    source_name: str
    enabled: bool


def create_control_routes(
    state: DashboardState,
    mq_client: Optional[MessageQueueClient] = None
) -> APIRouter:
    """
    Create control API routes with state dependency.

    Args:
        state: Dashboard state manager
        mq_client: Message queue client for sending control commands

    Returns:
        Configured API router
    """

    @router.get("/services/status")
    async def get_services_status():
        """
        GET /api/control/services/status
        Get status of all services.
        """
        services = state.get_service_status()

        return {
            'count': len(services),
            'services': services
        }

    @router.post("/services/control")
    async def control_service(request: ServiceControlRequest):
        """
        POST /api/control/services/control
        Start, stop, or restart a service.

        Body: {
            "service_name": "assimilation",
            "action": "restart"
        }
        """
        if mq_client is None:
            raise HTTPException(
                status_code=503,
                detail="Message queue not available"
            )

        valid_actions = ['start', 'stop', 'restart']
        if request.action not in valid_actions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid action. Must be one of: {valid_actions}"
            )

        # Publish service control message
        try:
            mq_client.publish(
                topic="ctrl.service",
                data={
                    'service': request.service_name,
                    'action': request.action
                },
                source="dashboard"
            )

            return {
                'status': 'success',
                'message': f"Service {request.service_name} {request.action} requested"
            }

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to send control command: {str(e)}"
            )

    @router.post("/filter/trigger_cycle")
    async def trigger_filter_cycle(request: FilterCycleRequest):
        """
        POST /api/control/filter/trigger_cycle
        Manually trigger a filter update cycle.

        Body: {
            "force": true,
            "reason": "Testing new observations"
        }
        """
        if mq_client is None:
            raise HTTPException(
                status_code=503,
                detail="Message queue not available"
            )

        try:
            mq_client.publish(
                topic=Topics.CTRL_CYCLE_TRIGGER,
                data={
                    'force': request.force,
                    'reason': request.reason,
                    'source': 'dashboard'
                },
                source="dashboard"
            )

            return {
                'status': 'success',
                'message': 'Filter cycle triggered'
            }

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to trigger cycle: {str(e)}"
            )

    @router.post("/filter/update_parameters")
    async def update_filter_parameters(request: FilterParametersRequest):
        """
        POST /api/control/filter/update_parameters
        Update filter tuning parameters.

        Body: {
            "process_noise_ne": 1e10,
            "observation_noise_tec": 2.0,
            "covariance_inflation": 1.05
        }
        """
        if mq_client is None:
            raise HTTPException(
                status_code=503,
                detail="Message queue not available"
            )

        # Build parameter update dict
        params = {}
        if request.process_noise_ne is not None:
            params['process_noise_ne'] = request.process_noise_ne
        if request.observation_noise_tec is not None:
            params['observation_noise_tec'] = request.observation_noise_tec
        if request.covariance_inflation is not None:
            params['covariance_inflation'] = request.covariance_inflation

        if not params:
            raise HTTPException(
                status_code=400,
                detail="No parameters specified"
            )

        try:
            mq_client.publish(
                topic="ctrl.filter_params",
                data=params,
                source="dashboard"
            )

            return {
                'status': 'success',
                'message': 'Filter parameters update requested',
                'parameters': params
            }

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to update parameters: {str(e)}"
            )

    @router.get("/filter/metrics")
    async def get_filter_metrics():
        """
        GET /api/control/filter/metrics
        Get filter performance metrics.
        """
        with state.health_lock:
            metrics = state.filter_metrics

        if not metrics:
            raise HTTPException(
                status_code=404,
                detail="No filter metrics available"
            )

        return metrics

    @router.post("/mode/switch")
    async def switch_mode(request: ModeSwitchRequest):
        """
        POST /api/control/mode/switch
        Force autonomous mode switch (QUIET ↔ SHOCK).

        Body: {
            "mode": "SHOCK",
            "reason": "Anticipated solar flare"
        }
        """
        if mq_client is None:
            raise HTTPException(
                status_code=503,
                detail="Message queue not available"
            )

        if request.mode not in ['QUIET', 'SHOCK']:
            raise HTTPException(
                status_code=400,
                detail="Mode must be 'QUIET' or 'SHOCK'"
            )

        try:
            mq_client.publish(
                topic=Topics.CTRL_MODE_CHANGE,
                data={
                    'new_mode': request.mode,
                    'old_mode': state.current_mode,
                    'reason': request.reason,
                    'manual_override': True
                },
                source="dashboard"
            )

            # Update local state
            state.update_mode(request.mode, request.reason)

            return {
                'status': 'success',
                'message': f'Mode switched to {request.mode}',
                'previous_mode': state.current_mode
            }

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to switch mode: {str(e)}"
            )

    @router.post("/datasource/toggle")
    async def toggle_data_source(request: DataSourceToggleRequest):
        """
        POST /api/control/datasource/toggle
        Enable or disable a data source.

        Body: {
            "source_name": "goes_xray",
            "enabled": false
        }
        """
        if mq_client is None:
            raise HTTPException(
                status_code=503,
                detail="Message queue not available"
            )

        try:
            mq_client.publish(
                topic="ctrl.datasource",
                data={
                    'source': request.source_name,
                    'enabled': request.enabled
                },
                source="dashboard"
            )

            return {
                'status': 'success',
                'message': f"Data source {request.source_name} {'enabled' if request.enabled else 'disabled'}"
            }

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to toggle data source: {str(e)}"
            )

    @router.get("/observations/counts")
    async def get_observation_counts():
        """
        GET /api/control/observations/counts
        Get counts of observations by type.
        """
        counts = state.get_observation_counts()

        return counts

    @router.get("/observations/recent")
    async def get_recent_observations(
        hours: int = Query(1, description="Number of hours of history to return"),
        obs_type: Optional[str] = Query(None, description="Filter by type (gnss_tec, ionosonde, nvis_sounder)")
    ):
        """
        GET /api/control/observations/recent?hours=1&obs_type=gnss_tec
        Get recent observations.
        """
        observations = state.get_observation_history(hours=hours)

        # Filter by type if specified
        if obs_type:
            observations = [
                obs for obs in observations
                if obs.get('obs_type') == obs_type
            ]

        return {
            'hours': hours,
            'obs_type_filter': obs_type,
            'count': len(observations),
            'data': observations
        }

    @router.get("/system/statistics")
    async def get_system_statistics():
        """
        GET /api/control/system/statistics
        Get comprehensive system statistics.
        """
        stats = state.get_statistics()

        return stats

    @router.get("/health/check")
    async def health_check():
        """
        GET /api/control/health/check
        Health check endpoint for monitoring.
        """
        metadata = state.get_grid_metadata()
        grid_healthy = metadata is not None and metadata.get('age_seconds', float('inf')) < 1800

        return {
            'status': 'healthy' if grid_healthy else 'degraded',
            'grid_age_seconds': metadata.get('age_seconds') if metadata else None,
            'stats': state.get_statistics()
        }

    return router
