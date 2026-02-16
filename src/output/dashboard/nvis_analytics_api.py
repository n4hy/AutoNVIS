"""
NVIS Analytics REST API

Provides REST endpoints and WebSocket support for the NVIS analytics dashboard.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.requests import Request
from typing import List, Dict, Any, Optional
import asyncio
import json
from datetime import datetime
from pathlib import Path
import numpy as np

from ...common.logging_config import ServiceLogger
from ...common.message_queue import MessageQueueClient, Topics
from ...analysis.network_analyzer import NetworkAnalyzer
from ...analysis.information_gain_analyzer import InformationGainAnalyzer
from ...analysis.optimal_placement import OptimalPlacementRecommender
from ...ingestion.nvis.protocol_adapters.base_adapter import SounderMetadata


class WebSocketManager:
    """Manages WebSocket connections for real-time updates"""

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.logger = ServiceLogger("websocket_manager")

    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        self.active_connections.remove(websocket)
        self.logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        if self.active_connections:
            self.logger.info(f"Broadcasting {message.get('type', 'unknown')} to {len(self.active_connections)} client(s)")
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                self.logger.error(f"Error broadcasting to client: {e}")
                disconnected.append(connection)

        # Clean up disconnected clients
        for conn in disconnected:
            if conn in self.active_connections:
                self.active_connections.remove(conn)


class NVISAnalyticsAPI:
    """NVIS Analytics API backend"""

    def __init__(
        self,
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
        alt_grid: np.ndarray,
        mq_client: Optional[MessageQueueClient] = None
    ):
        """
        Initialize NVIS analytics API

        Args:
            lat_grid: Latitude grid
            lon_grid: Longitude grid
            alt_grid: Altitude grid
            mq_client: Message queue client (optional)
        """
        self.lat_grid = lat_grid
        self.lon_grid = lon_grid
        self.alt_grid = alt_grid
        self.mq_client = mq_client

        # Initialize analyzers
        grid_shape = (len(lat_grid), len(lon_grid), len(alt_grid))
        self.info_gain_analyzer = InformationGainAnalyzer(
            grid_shape, lat_grid, lon_grid, alt_grid
        )
        self.placement_recommender = OptimalPlacementRecommender(
            lat_grid, lon_grid, alt_grid
        )
        self.network_analyzer = NetworkAnalyzer(
            lat_grid, lon_grid, alt_grid, mq_client
        )

        # WebSocket manager
        self.ws_manager = WebSocketManager()

        # State
        self.sounder_registry: Dict[str, SounderMetadata] = {}
        self.recent_observations: List[Dict[str, Any]] = []
        self.latest_analysis: Optional[Dict[str, Any]] = None
        self.prior_sqrt_cov: Optional[np.ndarray] = None

        self.logger = ServiceLogger("nvis_analytics_api")

        # Subscribe to analysis updates
        if self.mq_client:
            self._subscribe_to_updates()

    def _subscribe_to_updates(self):
        """Subscribe to analysis updates from message queue"""
        def on_analysis_update(message):
            self.latest_analysis = message.data
            # Broadcast to WebSocket clients
            asyncio.create_task(self.ws_manager.broadcast({
                'type': 'analysis_update',
                'data': message.data
            }))

        self.mq_client.subscribe(
            Topics.ANALYSIS_INFO_GAIN,
            on_analysis_update,
            queue_name="dashboard_analysis"
        )

    def update_sounder_registry(self, sounders: List[SounderMetadata]):
        """Update sounder registry"""
        for sounder in sounders:
            self.sounder_registry[sounder.sounder_id] = sounder

    def update_observations(self, observations: List[Dict[str, Any]]):
        """Update recent observations"""
        self.recent_observations = observations

    def update_prior_covariance(self, prior_sqrt_cov: np.ndarray):
        """Update prior covariance"""
        self.prior_sqrt_cov = prior_sqrt_cov

    async def get_sounders(self) -> List[Dict[str, Any]]:
        """
        GET /nvis/sounders
        List all sounders with current metrics
        """
        if not self.latest_analysis:
            return []

        sounders_data = []
        for sounder_id, sounder in self.sounder_registry.items():
            # Get contribution from latest analysis
            contribution = None
            if self.latest_analysis and 'information_gain' in self.latest_analysis:
                contrib_data = self.latest_analysis['information_gain'].get('sounder_contributions', {})
                if sounder_id in contrib_data:
                    contribution = contrib_data[sounder_id]

            # Count observations
            n_obs = sum(1 for obs in self.recent_observations if obs['sounder_id'] == sounder_id)

            # Get quality tier (from latest observation)
            quality_tier = 'unknown'
            for obs in reversed(self.recent_observations):
                if obs['sounder_id'] == sounder_id:
                    quality_tier = obs.get('quality_tier', 'unknown')
                    break

            sounders_data.append({
                'sounder_id': sounder_id,
                'name': sounder.name,
                'latitude': sounder.latitude,
                'longitude': sounder.longitude,
                'equipment_type': sounder.equipment_type,
                'calibration_status': sounder.calibration_status,
                'quality_tier': quality_tier,
                'n_observations': n_obs,
                'marginal_gain': contribution['marginal_gain'] if contribution else 0.0,
                'relative_contribution': contribution['relative_contribution'] if contribution else 0.0
            })

        # Sort by contribution (descending)
        sounders_data.sort(key=lambda x: x['relative_contribution'], reverse=True)

        return sounders_data

    async def get_sounder_detail(self, sounder_id: str) -> Dict[str, Any]:
        """
        GET /nvis/sounder/{id}
        Get detailed information for a sounder
        """
        if sounder_id not in self.sounder_registry:
            raise HTTPException(status_code=404, detail="Sounder not found")

        sounder = self.sounder_registry[sounder_id]

        # Get observations from this sounder
        sounder_obs = [obs for obs in self.recent_observations if obs['sounder_id'] == sounder_id]

        # Compute information gain
        if self.prior_sqrt_cov is not None and len(sounder_obs) > 0:
            result = self.info_gain_analyzer.compute_marginal_gain(
                sounder_id,
                self.recent_observations,
                self.prior_sqrt_cov
            )

            info_gain = {
                'marginal_gain': result.marginal_gain,
                'relative_contribution': result.relative_contribution,
                'trace_with': result.trace_with,
                'trace_without': result.trace_without
            }
        else:
            info_gain = None

        # Quality statistics
        if sounder_obs:
            quality_scores = []
            snr_values = []
            for obs in sounder_obs:
                metrics = obs.get('quality_metrics', {})
                score = np.mean([
                    metrics.get('signal_quality', 0.5),
                    metrics.get('calibration_quality', 0.5),
                    metrics.get('temporal_quality', 0.5),
                    metrics.get('spatial_quality', 0.5),
                    metrics.get('equipment_quality', 0.5),
                    metrics.get('historical_quality', 0.5)
                ])
                quality_scores.append(score)
                snr_values.append(obs.get('snr', 0.0))

            quality_stats = {
                'avg_quality_score': np.mean(quality_scores),
                'avg_snr': np.mean(snr_values),
                'quality_tier': sounder_obs[-1].get('quality_tier', 'unknown')
            }
        else:
            quality_stats = None

        return {
            'sounder_id': sounder_id,
            'name': sounder.name,
            'operator': sounder.operator,
            'location': sounder.location,
            'latitude': sounder.latitude,
            'longitude': sounder.longitude,
            'altitude': sounder.altitude,
            'equipment_type': sounder.equipment_type,
            'calibration_status': sounder.calibration_status,
            'n_observations': len(sounder_obs),
            'information_gain': info_gain,
            'quality_stats': quality_stats
        }

    async def get_network_analysis(self) -> Dict[str, Any]:
        """
        GET /nvis/network/analysis
        Get comprehensive network analysis
        """
        if not self.latest_analysis:
            # Perform analysis if not available
            if self.prior_sqrt_cov is not None and len(self.recent_observations) > 0:
                analysis = self.network_analyzer.analyze_network(
                    list(self.sounder_registry.values()),
                    self.recent_observations,
                    self.prior_sqrt_cov
                )
                return analysis
            else:
                return {
                    'error': 'Insufficient data for analysis',
                    'network_overview': {
                        'n_sounders': len(self.sounder_registry),
                        'n_observations': len(self.recent_observations)
                    }
                }

        return self.latest_analysis

    async def get_placement_recommendations(
        self,
        n_sounders: int = 3,
        assumed_tier: str = "gold"
    ) -> List[Dict[str, Any]]:
        """
        GET /nvis/placement/recommend
        Get optimal placement recommendations
        """
        recommendations = self.placement_recommender.recommend_multiple_locations(
            n_sounders=n_sounders,
            existing_sounders=list(self.sounder_registry.values()),
            recent_observations=self.recent_observations,
            prior_sqrt_cov=self.prior_sqrt_cov,
            assumed_tier=assumed_tier
        )

        return [
            {
                'priority': i + 1,
                'latitude': rec.latitude,
                'longitude': rec.longitude,
                'expected_gain': rec.expected_gain,
                'coverage_gap_score': rec.coverage_gap_score,
                'redundancy_score': rec.redundancy_score,
                'combined_score': rec.combined_score,
                'nearby_sounders': rec.nearby_sounders,
                'estimated_tier': rec.estimated_tier
            }
            for i, rec in enumerate(recommendations)
        ]

    async def simulate_placement(
        self,
        latitude: float,
        longitude: float,
        assumed_tier: str = "gold"
    ) -> Dict[str, Any]:
        """
        POST /nvis/placement/simulate
        Simulate placing a sounder at a location ('what-if')
        """
        analysis = self.placement_recommender.analyze_proposed_location(
            lat=latitude,
            lon=longitude,
            existing_sounders=list(self.sounder_registry.values()),
            recent_observations=self.recent_observations,
            prior_sqrt_cov=self.prior_sqrt_cov,
            assumed_tier=assumed_tier
        )

        return analysis

    async def get_placement_heatmap(
        self,
        resolution: int = 50
    ) -> Dict[str, Any]:
        """
        GET /nvis/placement/heatmap
        Get placement heatmap data
        """
        heatmap = self.placement_recommender.generate_placement_heatmap(
            existing_sounders=list(self.sounder_registry.values()),
            recent_observations=self.recent_observations,
            resolution=resolution
        )

        # Convert to JSON-serializable format
        lats = np.linspace(self.lat_grid.min(), self.lat_grid.max(), resolution)
        lons = np.linspace(self.lon_grid.min(), self.lon_grid.max(), resolution)

        return {
            'latitudes': lats.tolist(),
            'longitudes': lons.tolist(),
            'scores': heatmap.tolist()
        }


def create_app(
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    alt_grid: np.ndarray,
    mq_client: Optional[MessageQueueClient] = None,
    subscribers: Optional[dict] = None
) -> FastAPI:
    """
    Create FastAPI application

    Args:
        lat_grid: Latitude grid
        lon_grid: Longitude grid
        alt_grid: Altitude grid
        mq_client: Message queue client
        subscribers: Dictionary with subscriber instances and dashboard state

    Returns:
        FastAPI app instance
    """
    app = FastAPI(
        title="AutoNVIS Dashboard",
        description="Real-time ionospheric monitoring and NVIS propagation visualization",
        version="2.0.0"
    )

    # Initialize legacy API backend
    api_backend = NVISAnalyticsAPI(lat_grid, lon_grid, alt_grid, mq_client)

    # Setup new API routers if subscribers are available
    if subscribers:
        from .backend.ionosphere_api import create_ionosphere_routes
        from .backend.propagation_api import create_propagation_routes
        from .backend.spaceweather_api import create_spaceweather_routes
        from .backend.glotec_api import create_glotec_routes
        from .backend.control_api import create_control_routes

        dashboard_state = subscribers.get('state')

        # Include new API routers
        ionosphere_router = create_ionosphere_routes(dashboard_state)
        propagation_router = create_propagation_routes(dashboard_state)
        spaceweather_router = create_spaceweather_routes(dashboard_state)
        glotec_router = create_glotec_routes(dashboard_state)
        control_router = create_control_routes(dashboard_state, mq_client)

        app.include_router(ionosphere_router)
        app.include_router(propagation_router)
        app.include_router(spaceweather_router)
        app.include_router(glotec_router)
        app.include_router(control_router)

        # Note: WebSocket broadcast callbacks are now set during subscriber construction
        # in main.py, before threads start. This ensures no messages are lost.

    # Setup templates and static files
    base_path = Path(__file__).parent
    templates = Jinja2Templates(directory=str(base_path / "templates"))
    app.mount("/static", StaticFiles(directory=str(base_path / "static")), name="static")

    # Routes
    @app.get("/", response_class=HTMLResponse)
    async def overview(request: Request):
        """Overview dashboard page (main entry point)"""
        if subscribers:
            return templates.TemplateResponse("overview.html", {"request": request})
        else:
            # Fallback to legacy dashboard if no subscribers
            return templates.TemplateResponse("dashboard.html", {"request": request})

    @app.get("/network", response_class=HTMLResponse)
    async def network_view(request: Request):
        """NVIS network analysis page (legacy dashboard)"""
        return templates.TemplateResponse("dashboard.html", {"request": request})

    @app.get("/ionosphere", response_class=HTMLResponse)
    async def ionosphere_view(request: Request):
        """Ionosphere visualization page"""
        return templates.TemplateResponse("ionosphere.html", {"request": request})

    @app.get("/propagation", response_class=HTMLResponse)
    async def propagation_view(request: Request):
        """Propagation prediction page"""
        return templates.TemplateResponse("propagation.html", {"request": request})

    @app.get("/spaceweather", response_class=HTMLResponse)
    async def spaceweather_view(request: Request):
        """Space weather monitoring page"""
        return templates.TemplateResponse("spaceweather.html", {"request": request})

    @app.get("/control", response_class=HTMLResponse)
    async def control_view(request: Request):
        """System control page"""
        return templates.TemplateResponse("control.html", {"request": request})

    @app.get("/api/nvis/sounders")
    async def list_sounders():
        """List all sounders with metrics"""
        return await api_backend.get_sounders()

    @app.get("/api/nvis/sounder/{sounder_id}")
    async def get_sounder(sounder_id: str):
        """Get detailed sounder information"""
        return await api_backend.get_sounder_detail(sounder_id)

    @app.get("/api/nvis/network/analysis")
    async def network_analysis():
        """Get comprehensive network analysis"""
        return await api_backend.get_network_analysis()

    @app.get("/api/nvis/placement/recommend")
    async def placement_recommendations(
        n_sounders: int = 3,
        tier: str = "gold"
    ):
        """Get optimal placement recommendations"""
        return await api_backend.get_placement_recommendations(n_sounders, tier)

    @app.post("/api/nvis/placement/simulate")
    async def simulate_sounder_placement(
        latitude: float,
        longitude: float,
        tier: str = "gold"
    ):
        """Simulate placing a sounder ('what-if' analysis)"""
        return await api_backend.simulate_placement(latitude, longitude, tier)

    @app.get("/api/nvis/placement/heatmap")
    async def placement_heatmap(resolution: int = 50):
        """Get placement heatmap"""
        return await api_backend.get_placement_heatmap(resolution)

    # Use the shared WebSocket manager if available, otherwise fall back to api_backend's
    shared_ws_manager = subscribers.get('ws_manager') if subscribers else None
    active_ws_manager = shared_ws_manager or api_backend.ws_manager
    dashboard_state = subscribers.get('state') if subscribers else None

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket for real-time updates"""
        await active_ws_manager.connect(websocket)
        try:
            # Send current GloTEC data immediately on connect
            if dashboard_state:
                glotec = dashboard_state.get_latest_glotec()
                if glotec:
                    await websocket.send_json({
                        'type': 'glotec_update',
                        'data': glotec
                    })
                    active_ws_manager.logger.info("Sent initial GloTEC data to new client")

            while True:
                # Keep connection alive and receive messages
                data = await websocket.receive_text()
                # Echo back for now (could handle commands)
                await websocket.send_text(f"Echo: {data}")
        except WebSocketDisconnect:
            active_ws_manager.disconnect(websocket)

    # Store reference to backend for external updates
    app.state.api_backend = api_backend

    return app
