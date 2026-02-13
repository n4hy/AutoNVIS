"""
Integration Tests for NVIS Analytics Dashboard API

Tests REST endpoints and WebSocket functionality.
"""

import pytest
import numpy as np
from fastapi.testclient import TestClient

from src.output.dashboard.nvis_analytics_api import create_app
from src.ingestion.nvis.protocol_adapters.base_adapter import SounderMetadata


@pytest.fixture
def app():
    """Create test app"""
    lat_grid = np.linspace(-90, 90, 7)
    lon_grid = np.linspace(-180, 180, 7)
    alt_grid = np.linspace(100, 500, 11)

    return create_app(lat_grid, lon_grid, alt_grid, mq_client=None)


@pytest.fixture
def client(app):
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def sample_sounders():
    """Create sample sounders"""
    return [
        SounderMetadata(
            sounder_id='TEST_001',
            name='Test Sounder 1',
            operator='Test Operator',
            location='Test Location 1',
            latitude=40.0,
            longitude=-105.0,
            altitude=1500.0,
            equipment_type='professional',
            calibration_status='calibrated'
        ),
        SounderMetadata(
            sounder_id='TEST_002',
            name='Test Sounder 2',
            operator='Test Operator',
            location='Test Location 2',
            latitude=35.0,
            longitude=-110.0,
            altitude=1200.0,
            equipment_type='research',
            calibration_status='calibrated'
        )
    ]


@pytest.fixture
def sample_observations():
    """Create sample observations"""
    return [
        {
            'sounder_id': 'TEST_001',
            'tx_latitude': 40.0,
            'tx_longitude': -105.0,
            'tx_altitude': 1500.0,
            'rx_latitude': 40.5,
            'rx_longitude': -104.5,
            'rx_altitude': 1600.0,
            'frequency': 7.5,
            'elevation_angle': 85.0,
            'azimuth': 45.0,
            'hop_distance': 75.0,
            'signal_strength': -80.0,
            'group_delay': 2.5,
            'snr': 20.0,
            'signal_strength_error': 2.0,
            'group_delay_error': 0.1,
            'is_o_mode': True,
            'quality_tier': 'platinum',
            'quality_metrics': {
                'signal_quality': 0.9,
                'calibration_quality': 1.0,
                'temporal_quality': 0.8,
                'spatial_quality': 0.7,
                'equipment_quality': 1.0,
                'historical_quality': 0.8
            }
        },
        {
            'sounder_id': 'TEST_002',
            'tx_latitude': 35.0,
            'tx_longitude': -110.0,
            'tx_altitude': 1200.0,
            'rx_latitude': 35.5,
            'rx_longitude': -109.5,
            'rx_altitude': 1300.0,
            'frequency': 7.5,
            'elevation_angle': 80.0,
            'azimuth': 90.0,
            'hop_distance': 80.0,
            'signal_strength': -90.0,
            'group_delay': 2.8,
            'snr': 15.0,
            'signal_strength_error': 8.0,
            'group_delay_error': 2.0,
            'is_o_mode': True,
            'quality_tier': 'silver',
            'quality_metrics': {
                'signal_quality': 0.5,
                'calibration_quality': 0.5,
                'temporal_quality': 0.4,
                'spatial_quality': 0.6,
                'equipment_quality': 0.4,
                'historical_quality': 0.5
            }
        }
    ]


class TestDashboardEndpoints:
    """Test dashboard REST endpoints"""

    def test_root_endpoint(self, client):
        """Test root endpoint returns HTML"""
        response = client.get("/")
        assert response.status_code == 200
        assert "NVIS" in response.text

    def test_sounders_endpoint_empty(self, client):
        """Test sounders endpoint with no data"""
        response = client.get("/api/nvis/sounders")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_sounders_endpoint_with_data(self, client, app, sample_sounders, sample_observations):
        """Test sounders endpoint with data"""
        # Update backend with test data
        backend = app.state.api_backend
        backend.update_sounder_registry(sample_sounders)
        backend.update_observations(sample_observations)

        response = client.get("/api/nvis/sounders")
        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 2
        assert data[0]['sounder_id'] in ['TEST_001', 'TEST_002']

    def test_sounder_detail_endpoint(self, client, app, sample_sounders, sample_observations):
        """Test sounder detail endpoint"""
        backend = app.state.api_backend
        backend.update_sounder_registry(sample_sounders)
        backend.update_observations(sample_observations)

        response = client.get("/api/nvis/sounder/TEST_001")
        assert response.status_code == 200
        data = response.json()
        assert data['sounder_id'] == 'TEST_001'
        assert 'latitude' in data
        assert 'longitude' in data

    def test_sounder_detail_not_found(self, client):
        """Test sounder detail endpoint with non-existent sounder"""
        response = client.get("/api/nvis/sounder/NONEXISTENT")
        assert response.status_code == 404

    def test_network_analysis_endpoint(self, client):
        """Test network analysis endpoint"""
        response = client.get("/api/nvis/network/analysis")
        assert response.status_code == 200
        data = response.json()
        # Should return error or analysis
        assert 'error' in data or 'network_overview' in data

    def test_placement_recommendations_endpoint(self, client, app, sample_sounders, sample_observations):
        """Test placement recommendations endpoint"""
        backend = app.state.api_backend
        backend.update_sounder_registry(sample_sounders)
        backend.update_observations(sample_observations)

        response = client.get("/api/nvis/placement/recommend?n_sounders=2&tier=gold")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2

    def test_simulate_placement_endpoint(self, client, app, sample_sounders, sample_observations):
        """Test simulate placement endpoint"""
        backend = app.state.api_backend
        backend.update_sounder_registry(sample_sounders)
        backend.update_observations(sample_observations)

        response = client.post(
            "/api/nvis/placement/simulate?latitude=45.0&longitude=-100.0&tier=gold"
        )
        assert response.status_code == 200
        data = response.json()
        assert 'scores' in data
        assert 'recommendation' in data

    def test_heatmap_endpoint(self, client, app, sample_sounders, sample_observations):
        """Test heatmap endpoint"""
        backend = app.state.api_backend
        backend.update_sounder_registry(sample_sounders)
        backend.update_observations(sample_observations)

        response = client.get("/api/nvis/placement/heatmap?resolution=10")
        assert response.status_code == 200
        data = response.json()
        assert 'latitudes' in data
        assert 'longitudes' in data
        assert 'scores' in data
        assert len(data['scores']) == 10
        assert len(data['scores'][0]) == 10


class TestWebSocket:
    """Test WebSocket functionality"""

    def test_websocket_connection(self, client):
        """Test WebSocket connection"""
        with client.websocket_connect("/ws") as websocket:
            # Send test message
            websocket.send_text("test")
            # Receive echo
            data = websocket.receive_text()
            assert "Echo" in data


class TestAPIBackend:
    """Test API backend functionality"""

    def test_sounder_registry_update(self, app, sample_sounders):
        """Test updating sounder registry"""
        backend = app.state.api_backend
        backend.update_sounder_registry(sample_sounders)

        assert len(backend.sounder_registry) == 2
        assert 'TEST_001' in backend.sounder_registry

    def test_observations_update(self, app, sample_observations):
        """Test updating observations"""
        backend = app.state.api_backend
        backend.update_observations(sample_observations)

        assert len(backend.recent_observations) == 2

    def test_prior_covariance_update(self, app):
        """Test updating prior covariance"""
        backend = app.state.api_backend
        prior_sqrt_cov = np.eye(100)
        backend.update_prior_covariance(prior_sqrt_cov)

        assert backend.prior_sqrt_cov is not None
        assert backend.prior_sqrt_cov.shape == (100, 100)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
