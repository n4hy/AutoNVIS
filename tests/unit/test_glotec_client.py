"""
Unit Tests for GloTEC Client

Tests for GloTEC data fetching and parsing.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
import json

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ingestion.space_weather.glotec_client import GloTECClient


class TestGloTECClient:
    """Unit tests for GloTECClient."""

    @pytest.fixture
    def client(self):
        """Create a GloTEC client instance."""
        return GloTECClient(mq_client=MagicMock())

    @pytest.fixture
    def sample_index(self):
        """Sample GloTEC index response."""
        return [
            {"url": "/products/glotec/geojson_2d_urt/glotec_icao_20260116T120000Z.geojson", "time_tag": "2026-01-16T12:00:00Z"},
            {"url": "/products/glotec/geojson_2d_urt/glotec_icao_20260116T121000Z.geojson", "time_tag": "2026-01-16T12:10:00Z"},
        ]

    @pytest.fixture
    def sample_geojson(self):
        """Sample GloTEC GeoJSON response."""
        return {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [-180.0, -90.0]},
                    "properties": {
                        "tec": 15.5,
                        "anomaly": -2.3,
                        "hmF2": 280.0,
                        "NmF2": 5.2e11,
                        "quality_flag": 5
                    }
                },
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [-175.0, -90.0]},
                    "properties": {
                        "tec": 16.2,
                        "anomaly": -1.8,
                        "hmF2": 285.0,
                        "NmF2": 5.5e11,
                        "quality_flag": 5
                    }
                },
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [-180.0, -87.5]},
                    "properties": {
                        "tec": 17.0,
                        "anomaly": -1.5,
                        "hmF2": 290.0,
                        "NmF2": 5.8e11,
                        "quality_flag": 5
                    }
                },
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [-175.0, -87.5]},
                    "properties": {
                        "tec": 17.8,
                        "anomaly": -1.2,
                        "hmF2": 295.0,
                        "NmF2": 6.0e11,
                        "quality_flag": 5
                    }
                }
            ]
        }

    def test_client_initialization(self, client):
        """Test client initializes with default values."""
        assert client.index_url is not None
        assert client.base_url is not None
        assert client.update_interval == 600
        assert client.last_fetch_time is None
        assert client._maps_fetched == 0

    def test_parse_tec_map(self, client, sample_geojson):
        """Test parsing GeoJSON to TEC map structure."""
        time_tag = "2026-01-16T12:00:00Z"
        result = client.parse_tec_map(sample_geojson, time_tag)

        assert 'grid' in result
        assert 'statistics' in result
        assert 'metadata' in result
        assert result['timestamp'] == time_tag

        grid = result['grid']
        assert 'lat' in grid
        assert 'lon' in grid
        assert 'tec' in grid
        assert 'anomaly' in grid
        assert 'hmF2' in grid
        assert 'NmF2' in grid

        stats = result['statistics']
        assert 'tec_mean' in stats
        assert 'tec_max' in stats
        assert 'tec_min' in stats
        assert stats['n_valid_cells'] == 4

    def test_parse_tec_map_statistics(self, client, sample_geojson):
        """Test statistics calculation from TEC map."""
        result = client.parse_tec_map(sample_geojson, "2026-01-16T12:00:00Z")
        stats = result['statistics']

        # Check calculated statistics
        assert stats['tec_min'] == 15.5
        assert stats['tec_max'] == 17.8
        assert 15.5 <= stats['tec_mean'] <= 17.8

    def test_parse_tec_map_empty_features(self, client):
        """Test parsing with empty features."""
        empty_geojson = {"type": "FeatureCollection", "features": []}
        result = client.parse_tec_map(empty_geojson, "2026-01-16T12:00:00Z")

        assert result == {}

    def test_reshape_to_grid(self, client, sample_geojson):
        """Test reshaping flat data to 2D grid."""
        import numpy as np

        values = np.array([1.0, 2.0, 3.0, 4.0])
        lats = np.array([-90.0, -90.0, -87.5, -87.5])
        lons = np.array([-180.0, -175.0, -180.0, -175.0])
        unique_lats = np.array([-90.0, -87.5])
        unique_lons = np.array([-180.0, -175.0])

        grid = client._reshape_to_grid(values, lats, lons, unique_lats, unique_lons)

        assert grid.shape == (2, 2)
        assert grid[0, 0] == 1.0  # lat=-90, lon=-180
        assert grid[0, 1] == 2.0  # lat=-90, lon=-175
        assert grid[1, 0] == 3.0  # lat=-87.5, lon=-180
        assert grid[1, 1] == 4.0  # lat=-87.5, lon=-175

    @pytest.mark.asyncio
    async def test_fetch_index_success(self, client, sample_index):
        """Test successful index fetch."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=sample_index)

        with patch('aiohttp.ClientSession') as mock_session:
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response

            index = await client.fetch_index()

            assert index is not None
            assert len(index) == 2
            assert 'url' in index[0]
            assert 'time_tag' in index[0]

    @pytest.mark.asyncio
    async def test_fetch_index_http_error(self, client):
        """Test index fetch with HTTP error."""
        mock_response = AsyncMock()
        mock_response.status = 500

        with patch('aiohttp.ClientSession') as mock_session:
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response

            index = await client.fetch_index()

            assert index is None

    @pytest.mark.asyncio
    async def test_fetch_geojson_success(self, client, sample_geojson):
        """Test successful GeoJSON fetch."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=sample_geojson)

        with patch('aiohttp.ClientSession') as mock_session:
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response

            result = await client.fetch_geojson("/products/glotec/test.geojson")

            assert result is not None
            assert result['type'] == 'FeatureCollection'
            assert len(result['features']) == 4

    def test_publish_to_queue(self, client):
        """Test publishing to message queue."""
        test_data = {
            'timestamp': '2026-01-16T12:00:00Z',
            'grid': {},
            'statistics': {'tec_mean': 25.0}
        }

        client.publish_to_queue(test_data)

        client.mq_client.publish.assert_called_once()
        call_args = client.mq_client.publish.call_args
        assert call_args[1]['topic'] == 'obs.glotec_map'
        assert call_args[1]['source'] == 'glotec_client'

    def test_statistics_property(self, client):
        """Test statistics property."""
        stats = client.statistics

        assert 'maps_fetched' in stats
        assert 'fetch_errors' in stats
        assert 'last_fetch_time' in stats
        assert 'last_file_url' in stats


class TestGloTECClientIntegration:
    """Integration tests for GloTEC client (require network)."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.skip(reason="Requires network access to NOAA SWPC")
    async def test_fetch_live_data(self):
        """Integration test: Fetch live GloTEC data."""
        client = GloTECClient()

        data = await client.fetch_latest()

        if data is not None:  # API may be unavailable
            assert 'grid' in data
            assert 'timestamp' in data
            assert data['grid']['tec'] is not None
            assert 'statistics' in data
