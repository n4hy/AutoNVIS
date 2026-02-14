"""
Integration Tests for AutoNVIS Dashboard

Tests the complete dashboard system including:
- API endpoints
- WebSocket connections
- Data subscribers
- State management
"""

import pytest
import asyncio
import json
from datetime import datetime
from fastapi.testclient import TestClient
import numpy as np

from src.output.dashboard.nvis_analytics_api import create_app
from src.output.dashboard.backend.state_manager import DashboardState
from src.output.dashboard.backend.data_processing import (
    compute_fof2,
    compute_hmf2,
    compute_tec,
    extract_horizontal_slice,
    extract_vertical_profile
)


class TestDataProcessing:
    """Test data processing utilities"""

    def test_compute_fof2(self):
        """Test foF2 calculation"""
        # Create simple 3D grid
        ne_grid = np.random.rand(10, 10, 20) * 1e12
        alt_grid = np.linspace(60, 600, 20)

        fof2_map, hmf2_map = compute_fof2(ne_grid, alt_grid)

        assert fof2_map.shape == (10, 10)
        assert hmf2_map.shape == (10, 10)
        assert np.all(fof2_map >= 0)
        assert np.all(hmf2_map >= 60)
        assert np.all(hmf2_map <= 600)

    def test_compute_tec(self):
        """Test TEC calculation"""
        ne_grid = np.random.rand(10, 10, 20) * 1e12
        alt_grid = np.linspace(60, 600, 20)

        tec_map = compute_tec(ne_grid, alt_grid)

        assert tec_map.shape == (10, 10)
        assert np.all(tec_map >= 0)

    def test_horizontal_slice(self):
        """Test horizontal slice extraction"""
        ne_grid = np.random.rand(10, 10, 20) * 1e12
        lat_grid = np.linspace(-90, 90, 10)
        lon_grid = np.linspace(-180, 180, 10)
        alt_grid = np.linspace(60, 600, 20)

        lat, lon, ne_slice = extract_horizontal_slice(
            ne_grid, lat_grid, lon_grid, alt_grid, 300.0
        )

        assert lat.shape == (10,)
        assert lon.shape == (10,)
        assert ne_slice.shape == (10, 10)

    def test_vertical_profile(self):
        """Test vertical profile extraction"""
        ne_grid = np.random.rand(10, 10, 20) * 1e12
        lat_grid = np.linspace(-90, 90, 10)
        lon_grid = np.linspace(-180, 180, 10)
        alt_grid = np.linspace(60, 600, 20)

        alt, ne_profile = extract_vertical_profile(
            ne_grid, lat_grid, lon_grid, alt_grid, 0.0, 0.0
        )

        assert alt.shape == (20,)
        assert ne_profile.shape == (20,)


class TestStateManager:
    """Test dashboard state manager"""

    def test_state_initialization(self):
        """Test state manager initialization"""
        state = DashboardState(retention_hours=24)

        assert state.retention_hours == 24
        assert state.latest_grid is None
        assert len(state.xray_flux_history) == 0

    def test_grid_update(self):
        """Test grid data update"""
        state = DashboardState()

        grid_data = {
            'ne_grid': np.random.rand(10, 10, 20) * 1e12,
            'lat': np.linspace(-90, 90, 10),
            'lon': np.linspace(-180, 180, 10),
            'alt': np.linspace(60, 600, 20),
            'cycle_id': 'test_001',
            'quality': 'good'
        }

        state.update_grid(grid_data, datetime.utcnow())

        assert state.latest_grid is not None
        assert state.latest_grid['cycle_id'] == 'test_001'
        assert state.stats['grids_received'] == 1

    def test_frequency_plan_update(self):
        """Test frequency plan update"""
        state = DashboardState()

        plan_data = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'luf_mhz': 2.5,
            'muf_mhz': 12.8,
            'fot_mhz': 10.9
        }

        state.update_frequency_plan(plan_data)

        assert state.latest_frequency_plan is not None
        assert len(state.luf_history) == 1
        assert len(state.muf_history) == 1
        assert len(state.fot_history) == 1

    def test_observation_tracking(self):
        """Test observation tracking"""
        state = DashboardState()

        obs_data = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'value': 15.2,
            'sounder_id': 'test_sounder'
        }

        state.add_observation(obs_data, 'gnss_tec')

        assert state.observation_counts['gnss_tec'] == 1
        assert state.observation_counts['total'] == 1


class TestAPIEndpoints:
    """Test API endpoints"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        lat_grid = np.linspace(-90, 90, 10)
        lon_grid = np.linspace(-180, 180, 10)
        alt_grid = np.linspace(60, 600, 20)

        # Create app without RabbitMQ
        app = create_app(lat_grid, lon_grid, alt_grid, mq_client=None, subscribers=None)
        return TestClient(app)

    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_ionosphere_view(self, client):
        """Test ionosphere view endpoint"""
        response = client.get("/ionosphere")
        assert response.status_code == 200

    def test_propagation_view(self, client):
        """Test propagation view endpoint"""
        response = client.get("/propagation")
        assert response.status_code == 200

    def test_spaceweather_view(self, client):
        """Test space weather view endpoint"""
        response = client.get("/spaceweather")
        assert response.status_code == 200

    def test_control_view(self, client):
        """Test control view endpoint"""
        response = client.get("/control")
        assert response.status_code == 200


class TestPerformance:
    """Performance tests"""

    def test_grid_update_performance(self):
        """Test grid update performance"""
        import time

        state = DashboardState()

        # Create large grid (73x73x55 like production)
        grid_data = {
            'ne_grid': np.random.rand(73, 73, 55) * 1e12,
            'lat': np.linspace(-90, 90, 73),
            'lon': np.linspace(-180, 180, 73),
            'alt': np.linspace(60, 600, 55),
            'cycle_id': 'perf_test',
            'quality': 'good'
        }

        start = time.time()
        state.update_grid(grid_data, datetime.utcnow())
        duration = time.time() - start

        # Should complete in less than 100ms
        assert duration < 0.1

    def test_fof2_calculation_performance(self):
        """Test foF2 calculation performance"""
        import time

        # Production-sized grid
        ne_grid = np.random.rand(73, 73, 55) * 1e12
        alt_grid = np.linspace(60, 600, 55)

        start = time.time()
        fof2_map, hmf2_map = compute_fof2(ne_grid, alt_grid)
        duration = time.time() - start

        # Should complete in less than 1 second
        assert duration < 1.0

    def test_tec_calculation_performance(self):
        """Test TEC calculation performance"""
        import time

        ne_grid = np.random.rand(73, 73, 55) * 1e12
        alt_grid = np.linspace(60, 600, 55)

        start = time.time()
        tec_map = compute_tec(ne_grid, alt_grid)
        duration = time.time() - start

        # Should complete in less than 1 second
        assert duration < 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
