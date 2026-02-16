"""
GloTEC Global TEC Map Client

Fetches real-time global Total Electron Content maps from NOAA SWPC GloTEC.
GloTEC provides near-real-time global TEC derived from GNSS observations,
updated every 10 minutes.

Data source: https://www.swpc.noaa.gov/products/glotec
"""

import asyncio
import aiohttp
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.common.config import get_config
from src.common.logging_config import ServiceLogger, setup_logging
from src.common.message_queue import MessageQueueClient, Topics


class GloTECClient:
    """
    Client for fetching NOAA SWPC GloTEC global TEC maps

    GloTEC is a global 3D data assimilation system that uses a Gauss-Markov
    Kalman Filter to estimate electron density in the ionosphere. It ingests
    slant TEC from ground-based GNSS receivers and space-based Radio Occultation
    data from COSMIC-2/FORMOSAT-7.

    Output grid:
    - Resolution: 5 deg longitude x 2.5 deg latitude
    - Properties: tec, anomaly, hmF2, NmF2, quality_flag
    - Update cadence: 10 minutes
    """

    def __init__(
        self,
        index_url: str = None,
        base_url: str = None,
        update_interval: int = None,
        mq_client: MessageQueueClient = None
    ):
        """
        Initialize GloTEC client

        Args:
            index_url: URL to GloTEC file index JSON
            base_url: Base URL for constructing full file URLs
            update_interval: Update interval in seconds
            mq_client: Message queue client (optional, will create if None)
        """
        config = get_config()

        self.index_url = index_url or config.data_sources.glotec_index_url
        self.base_url = base_url or config.data_sources.glotec_base_url
        self.update_interval = update_interval or config.data_sources.glotec_update_interval

        self.logger = ServiceLogger("ingestion", "glotec")
        self.mq_client = mq_client

        # State tracking
        self.last_fetch_time: Optional[datetime] = None
        self.last_file_url: Optional[str] = None
        self.last_tec_map: Optional[Dict[str, Any]] = None

        # Statistics
        self._maps_fetched = 0
        self._fetch_errors = 0

    async def fetch_index(self) -> Optional[List[Dict[str, str]]]:
        """
        Fetch the GloTEC file index to find available data files

        Returns:
            List of file entries with 'url' and 'time_tag' fields, or None if failed
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.index_url, timeout=30) as response:
                    if response.status != 200:
                        self.logger.error(
                            f"HTTP {response.status} from GloTEC index",
                            extra={'status_code': response.status}
                        )
                        return None

                    data = await response.json()

                    if not data or len(data) == 0:
                        self.logger.warning("Empty index from GloTEC API")
                        return None

                    return data

        except asyncio.TimeoutError:
            self.logger.error("Timeout fetching GloTEC index")
            return None
        except aiohttp.ClientError as e:
            self.logger.error(f"HTTP error fetching GloTEC index: {e}", exc_info=True)
            return None
        except Exception as e:
            self.logger.error(f"Error fetching GloTEC index: {e}", exc_info=True)
            return None

    async def fetch_geojson(self, file_url: str) -> Optional[Dict[str, Any]]:
        """
        Fetch a GloTEC GeoJSON file

        Args:
            file_url: Relative URL path to the GeoJSON file

        Returns:
            Parsed GeoJSON data, or None if failed
        """
        full_url = f"{self.base_url}{file_url}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(full_url, timeout=60) as response:
                    if response.status != 200:
                        self.logger.error(
                            f"HTTP {response.status} fetching GloTEC file",
                            extra={'url': full_url, 'status_code': response.status}
                        )
                        return None

                    return await response.json()

        except asyncio.TimeoutError:
            self.logger.error(f"Timeout fetching GloTEC file: {file_url}")
            return None
        except aiohttp.ClientError as e:
            self.logger.error(f"HTTP error fetching GloTEC file: {e}", exc_info=True)
            return None
        except Exception as e:
            self.logger.error(f"Error fetching GloTEC file: {e}", exc_info=True)
            return None

    def parse_tec_map(self, geojson_data: Dict[str, Any], time_tag: str) -> Dict[str, Any]:
        """
        Parse GeoJSON FeatureCollection into structured TEC map arrays

        Args:
            geojson_data: GeoJSON FeatureCollection from GloTEC
            time_tag: ISO timestamp for this map

        Returns:
            Structured TEC map with grid arrays and statistics
        """
        features = geojson_data.get('features', [])

        if not features:
            self.logger.warning("No features in GloTEC GeoJSON")
            return {}

        # Extract coordinates and properties
        lons = []
        lats = []
        tec_values = []
        anomaly_values = []
        hmf2_values = []
        nmf2_values = []
        quality_values = []

        for feature in features:
            geometry = feature.get('geometry', {})
            properties = feature.get('properties', {})

            if geometry.get('type') == 'Point':
                coords = geometry.get('coordinates', [])
                if len(coords) >= 2:
                    lons.append(coords[0])
                    lats.append(coords[1])
                    tec_values.append(properties.get('tec', np.nan))
                    anomaly_values.append(properties.get('anomaly', np.nan))
                    hmf2_values.append(properties.get('hmF2', np.nan))
                    nmf2_values.append(properties.get('NmF2', np.nan))
                    quality_values.append(properties.get('quality_flag', 0))

        # Convert to numpy arrays
        lons = np.array(lons)
        lats = np.array(lats)
        tec = np.array(tec_values)
        anomaly = np.array(anomaly_values)
        hmf2 = np.array(hmf2_values)
        nmf2 = np.array(nmf2_values)
        quality = np.array(quality_values)

        # Determine grid dimensions
        unique_lons = np.unique(lons)
        unique_lats = np.unique(lats)
        n_lon = len(unique_lons)
        n_lat = len(unique_lats)

        # Reshape to 2D grids (lat x lon)
        try:
            tec_grid = self._reshape_to_grid(tec, lats, lons, unique_lats, unique_lons)
            anomaly_grid = self._reshape_to_grid(anomaly, lats, lons, unique_lats, unique_lons)
            hmf2_grid = self._reshape_to_grid(hmf2, lats, lons, unique_lats, unique_lons)
            nmf2_grid = self._reshape_to_grid(nmf2, lats, lons, unique_lats, unique_lons)
            quality_grid = self._reshape_to_grid(quality, lats, lons, unique_lats, unique_lons)
        except Exception as e:
            self.logger.warning(f"Could not reshape to grid, using flat arrays: {e}")
            tec_grid = tec
            anomaly_grid = anomaly
            hmf2_grid = hmf2
            nmf2_grid = nmf2
            quality_grid = quality

        # Calculate statistics
        valid_tec = tec[~np.isnan(tec)]
        statistics = {
            'tec_mean': float(np.mean(valid_tec)) if len(valid_tec) > 0 else None,
            'tec_max': float(np.max(valid_tec)) if len(valid_tec) > 0 else None,
            'tec_min': float(np.min(valid_tec)) if len(valid_tec) > 0 else None,
            'tec_std': float(np.std(valid_tec)) if len(valid_tec) > 0 else None,
            'n_valid_cells': int(len(valid_tec)),
            'n_total_cells': len(tec)
        }

        # Calculate anomaly statistics
        valid_anomaly = anomaly[~np.isnan(anomaly)]
        if len(valid_anomaly) > 0:
            statistics['anomaly_mean'] = float(np.mean(valid_anomaly))
            statistics['anomaly_min'] = float(np.min(valid_anomaly))
            statistics['anomaly_max'] = float(np.max(valid_anomaly))

        return {
            'timestamp': time_tag,
            'grid': {
                'lat': unique_lats.tolist(),
                'lon': unique_lons.tolist(),
                'tec': tec_grid.tolist() if isinstance(tec_grid, np.ndarray) else tec_grid,
                'anomaly': anomaly_grid.tolist() if isinstance(anomaly_grid, np.ndarray) else anomaly_grid,
                'hmF2': hmf2_grid.tolist() if isinstance(hmf2_grid, np.ndarray) else hmf2_grid,
                'NmF2': nmf2_grid.tolist() if isinstance(nmf2_grid, np.ndarray) else nmf2_grid,
                'quality': quality_grid.tolist() if isinstance(quality_grid, np.ndarray) else quality_grid
            },
            'statistics': statistics,
            'metadata': {
                'n_lat': n_lat,
                'n_lon': n_lon,
                'lat_range': [float(unique_lats.min()), float(unique_lats.max())],
                'lon_range': [float(unique_lons.min()), float(unique_lons.max())],
                'source': 'NOAA_SWPC_GloTEC'
            }
        }

    def _reshape_to_grid(
        self,
        values: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
        unique_lats: np.ndarray,
        unique_lons: np.ndarray
    ) -> np.ndarray:
        """
        Reshape flat values array to 2D grid based on coordinates

        Args:
            values: Flat array of values
            lats: Latitude for each value
            lons: Longitude for each value
            unique_lats: Sorted unique latitudes
            unique_lons: Sorted unique longitudes

        Returns:
            2D grid array (lat x lon)
        """
        n_lat = len(unique_lats)
        n_lon = len(unique_lons)

        # Create mapping from coordinates to indices
        lat_to_idx = {lat: idx for idx, lat in enumerate(unique_lats)}
        lon_to_idx = {lon: idx for idx, lon in enumerate(unique_lons)}

        # Initialize grid with NaN
        grid = np.full((n_lat, n_lon), np.nan)

        # Fill grid
        for i, (lat, lon, val) in enumerate(zip(lats, lons, values)):
            lat_idx = lat_to_idx.get(lat)
            lon_idx = lon_to_idx.get(lon)
            if lat_idx is not None and lon_idx is not None:
                grid[lat_idx, lon_idx] = val

        return grid

    async def fetch_latest(self) -> Optional[Dict[str, Any]]:
        """
        Fetch the latest GloTEC map

        Returns:
            Parsed TEC map dictionary, or None if failed
        """
        try:
            # Fetch index to find latest file
            index = await self.fetch_index()

            if not index:
                self._fetch_errors += 1
                return None

            # Get most recent file (last in list)
            latest_entry = index[-1]
            file_url = latest_entry.get('url', '')
            time_tag = latest_entry.get('time_tag', '')

            # Skip if we already have this file
            if file_url == self.last_file_url:
                self.logger.debug(f"No new GloTEC data (last: {time_tag})")
                return self.last_tec_map

            # Fetch the GeoJSON file
            geojson = await self.fetch_geojson(file_url)

            if not geojson:
                self._fetch_errors += 1
                return None

            # Parse to structured format
            tec_map = self.parse_tec_map(geojson, time_tag)

            if not tec_map:
                self._fetch_errors += 1
                return None

            # Update state
            self.last_fetch_time = datetime.utcnow()
            self.last_file_url = file_url
            self.last_tec_map = tec_map
            self._maps_fetched += 1

            stats = tec_map.get('statistics', {})
            self.logger.info(
                f"Fetched GloTEC map: {time_tag}",
                extra={
                    'timestamp': time_tag,
                    'tec_mean': stats.get('tec_mean'),
                    'tec_max': stats.get('tec_max'),
                    'n_cells': stats.get('n_valid_cells')
                }
            )

            return tec_map

        except Exception as e:
            self.logger.error(f"Error fetching latest GloTEC: {e}", exc_info=True)
            self._fetch_errors += 1
            return None

    def publish_to_queue(self, data: Dict[str, Any]):
        """
        Publish GloTEC map to message queue

        Args:
            data: TEC map dictionary
        """
        if self.mq_client is None:
            self.logger.warning("No message queue client configured")
            return

        try:
            self.mq_client.publish(
                topic=Topics.OBS_GLOTEC_MAP,
                data=data,
                source="glotec_client"
            )

            stats = data.get('statistics', {})
            self.logger.info(
                f"Published GloTEC to queue: TEC mean={stats.get('tec_mean'):.1f} TECU",
                extra={
                    'timestamp': data.get('timestamp'),
                    'tec_mean': stats.get('tec_mean')
                }
            )

        except Exception as e:
            self.logger.error(f"Failed to publish to queue: {e}", exc_info=True)

    async def fetch_historical(self, hours: int = 6):
        """
        Fetch historical GloTEC data to backfill graphs on startup.

        Args:
            hours: Number of hours of history to fetch
        """
        self.logger.info(f"Fetching {hours} hours of historical GloTEC data...")

        try:
            index = await self.fetch_index()
            if not index:
                self.logger.warning("Could not fetch index for historical data")
                return

            # Calculate how many files to fetch (6 per hour at 10-min intervals)
            files_to_fetch = min(hours * 6, len(index))

            # Get the most recent N files (they're in chronological order)
            historical_files = index[-files_to_fetch:]

            self.logger.info(f"Fetching {len(historical_files)} historical maps...")

            fetched = 0
            for entry in historical_files:
                file_url = entry.get('url', '')
                time_tag = entry.get('time_tag', '')

                try:
                    geojson = await self.fetch_geojson(file_url)
                    if geojson:
                        tec_map = self.parse_tec_map(geojson, time_tag)
                        if tec_map:
                            self.publish_to_queue(tec_map)
                            fetched += 1
                            # Small delay to avoid overwhelming the queue
                            await asyncio.sleep(0.1)
                except Exception as e:
                    self.logger.debug(f"Failed to fetch {time_tag}: {e}")
                    continue

            self.logger.info(f"Historical backfill complete: {fetched} maps published")

        except Exception as e:
            self.logger.error(f"Error fetching historical data: {e}", exc_info=True)

    async def run_monitoring_loop(self):
        """
        Continuous monitoring loop - fetches data at regular intervals
        """
        self.logger.info(
            f"Starting GloTEC monitoring (interval: {self.update_interval}s)"
        )

        # Fetch historical data first to populate graphs
        await self.fetch_historical(hours=6)

        last_published_url = None

        while True:
            try:
                data = await self.fetch_latest()

                if data and self.last_file_url != last_published_url:
                    self.publish_to_queue(data)
                    last_published_url = self.last_file_url
                elif not data:
                    self.logger.warning("Failed to fetch GloTEC data, will retry")

                await asyncio.sleep(self.update_interval)

            except Exception as e:
                self.logger.error(
                    f"Error in monitoring loop: {e}",
                    exc_info=True
                )
                await asyncio.sleep(self.update_interval)

    async def run(self):
        """Start the GloTEC monitoring service"""
        self.logger.info("GloTEC client starting")

        # Initialize message queue if not provided
        if self.mq_client is None:
            config = get_config()
            self.mq_client = MessageQueueClient(
                host=config.services.rabbitmq_host,
                port=config.services.rabbitmq_port,
                username=config.services.rabbitmq_user,
                password=config.services.rabbitmq_password,
                vhost=config.services.rabbitmq_vhost
            )

        await self.run_monitoring_loop()

    @property
    def statistics(self) -> Dict[str, Any]:
        """Get client statistics"""
        return {
            'maps_fetched': self._maps_fetched,
            'fetch_errors': self._fetch_errors,
            'last_fetch_time': self.last_fetch_time.isoformat() if self.last_fetch_time else None,
            'last_file_url': self.last_file_url
        }


async def main():
    """Standalone entry point for GloTEC client"""
    setup_logging("ingestion", log_level="INFO", json_format=False)

    client = GloTECClient()
    await client.run()


if __name__ == "__main__":
    asyncio.run(main())
