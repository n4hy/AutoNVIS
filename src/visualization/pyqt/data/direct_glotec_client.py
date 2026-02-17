"""
Direct GloTEC Client - Fetches directly from NOAA

No RabbitMQ, no WebSocket, no middleware.
Runs in a QThread and emits signals just like the websocket client.
"""

import aiohttp
import asyncio
import numpy as np
from PyQt6.QtCore import QObject, QThread, pyqtSignal, QTimer
from typing import Optional, Dict, Any, List
import logging
from datetime import datetime


class DirectGloTECWorker(QObject):
    """
    Worker that fetches GloTEC directly from NOAA.
    Runs in a QThread.
    """

    # Signals
    connected = pyqtSignal()
    disconnected = pyqtSignal()
    error = pyqtSignal(str)
    glotec_received = pyqtSignal(dict)

    INDEX_URL = "https://services.swpc.noaa.gov/products/glotec/geojson_2d_urt.json"
    BASE_URL = "https://services.swpc.noaa.gov"

    def __init__(self, update_interval_ms: int = 60000):
        """
        Initialize worker.

        Args:
            update_interval_ms: How often to check for new data (default 60s)
        """
        super().__init__()
        self.update_interval_ms = update_interval_ms
        self.running = False
        self.timer: Optional[QTimer] = None
        self.last_file_url: Optional[str] = None
        self.historical_loaded = False
        self.logger = logging.getLogger("direct_glotec")

    def start_fetching(self):
        """Start periodic data fetching."""
        self.running = True
        self.logger.info("Starting direct GloTEC fetch...")

        # Create timer for periodic updates
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._fetch_data)
        self.timer.start(self.update_interval_ms)

        # Emit connected signal
        self.connected.emit()

        # Initial fetch
        self._fetch_data()

    def stop_fetching(self):
        """Stop fetching."""
        self.running = False
        if self.timer:
            self.timer.stop()
        self.disconnected.emit()
        self.logger.info("Stopped direct GloTEC fetch")

    def _fetch_data(self):
        """Fetch data (runs sync wrapper around async)."""
        if not self.running:
            return

        try:
            # Run async fetch in new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(self._async_fetch())
                if result:
                    self.glotec_received.emit(result)
            finally:
                loop.close()
        except Exception as e:
            self.logger.error(f"Fetch error: {e}")
            self.error.emit(str(e))

    async def _async_fetch(self) -> Optional[Dict[str, Any]]:
        """Async fetch from NOAA."""
        try:
            async with aiohttp.ClientSession() as session:
                # Get index
                async with session.get(self.INDEX_URL, timeout=30) as resp:
                    if resp.status != 200:
                        self.logger.error(f"HTTP {resp.status} from index")
                        return None
                    index = await resp.json()

                if not index:
                    return None

                # First fetch: load historical data
                if not self.historical_loaded:
                    await self._load_historical(session, index)
                    self.historical_loaded = True

                # Get latest
                latest = index[-1]
                file_url = latest.get('url', '')
                time_tag = latest.get('time_tag', '')

                # Skip if same file
                if file_url == self.last_file_url:
                    self.logger.debug("No new data")
                    return None

                # Fetch data
                full_url = f"{self.BASE_URL}{file_url}"
                async with session.get(full_url, timeout=60) as resp:
                    if resp.status != 200:
                        self.logger.error(f"HTTP {resp.status} from data")
                        return None
                    geojson = await resp.json()

                self.last_file_url = file_url

                # Parse
                result = self._parse_geojson(geojson, time_tag)
                if result:
                    self.logger.info(f"Fetched: {time_tag}, mean={result['statistics']['tec_mean']:.1f} TECU")
                return result

        except Exception as e:
            self.logger.error(f"Async fetch error: {e}")
            return None

    async def _load_historical(self, session: aiohttp.ClientSession, index: List[Dict]):
        """Load historical GloTEC data from available index entries."""
        # Take last 24 entries (each ~10 min apart = ~4 hours of history)
        # Limit to avoid slow startup
        historical_entries = index[-24:]
        self.logger.info(f"Loading {len(historical_entries)} historical GloTEC records...")

        for entry in historical_entries:
            try:
                file_url = entry.get('url', '')
                time_tag = entry.get('time_tag', '')

                if not file_url:
                    continue

                full_url = f"{self.BASE_URL}{file_url}"
                async with session.get(full_url, timeout=30) as resp:
                    if resp.status != 200:
                        continue
                    geojson = await resp.json()

                result = self._parse_geojson(geojson, time_tag)
                if result:
                    self.glotec_received.emit(result)

            except Exception as e:
                self.logger.debug(f"Error loading historical entry: {e}")
                continue

        self.logger.info("Historical GloTEC data loaded")

    def _parse_geojson(self, geojson: Dict, time_tag: str) -> Optional[Dict[str, Any]]:
        """Parse GeoJSON to structured data."""
        features = geojson.get('features', [])
        if not features:
            return None

        lons, lats, tec_values = [], [], []
        anomaly_values, hmf2_values, nmf2_values = [], [], []

        for f in features:
            geom = f.get('geometry', {})
            props = f.get('properties', {})

            if geom.get('type') == 'Point':
                coords = geom.get('coordinates', [])
                if len(coords) >= 2:
                    lons.append(coords[0])
                    lats.append(coords[1])
                    tec_values.append(props.get('tec', np.nan))
                    anomaly_values.append(props.get('anomaly', np.nan))
                    hmf2_values.append(props.get('hmF2', np.nan))
                    nmf2_values.append(props.get('NmF2', np.nan))

        lons = np.array(lons)
        lats = np.array(lats)
        tec = np.array(tec_values)
        anomaly = np.array(anomaly_values)
        hmf2 = np.array(hmf2_values)
        nmf2 = np.array(nmf2_values)

        unique_lons = np.unique(lons)
        unique_lats = np.unique(lats)
        n_lat, n_lon = len(unique_lats), len(unique_lons)

        # Build grids
        lat_to_idx = {lat: i for i, lat in enumerate(unique_lats)}
        lon_to_idx = {lon: i for i, lon in enumerate(unique_lons)}

        def to_grid(values):
            grid = np.full((n_lat, n_lon), np.nan)
            for lat, lon, val in zip(lats, lons, values):
                li, lo = lat_to_idx.get(lat), lon_to_idx.get(lon)
                if li is not None and lo is not None:
                    grid[li, lo] = val
            return grid.tolist()

        tec_grid = to_grid(tec_values)
        anomaly_grid = to_grid(anomaly_values)
        hmf2_grid = to_grid(hmf2_values)
        nmf2_grid = to_grid(nmf2_values)

        valid_tec = tec[~np.isnan(tec)]
        valid_anomaly = anomaly[~np.isnan(anomaly)]

        stats = {
            'tec_mean': float(np.mean(valid_tec)) if len(valid_tec) > 0 else None,
            'tec_max': float(np.max(valid_tec)) if len(valid_tec) > 0 else None,
            'tec_min': float(np.min(valid_tec)) if len(valid_tec) > 0 else None,
            'tec_std': float(np.std(valid_tec)) if len(valid_tec) > 0 else None,
            'n_valid_cells': int(len(valid_tec)),
        }

        if len(valid_anomaly) > 0:
            stats['anomaly_mean'] = float(np.mean(valid_anomaly))

        return {
            'timestamp': time_tag,
            'grid': {
                'lat': unique_lats.tolist(),
                'lon': unique_lons.tolist(),
                'tec': tec_grid,
                'anomaly': anomaly_grid,
                'hmF2': hmf2_grid,
                'NmF2': nmf2_grid,
            },
            'statistics': stats,
            'metadata': {
                'n_lat': n_lat,
                'n_lon': n_lon,
                'source': 'NOAA_SWPC_GloTEC'
            }
        }


class DirectGloTECClient(QObject):
    """
    Client that fetches GloTEC directly from NOAA.
    Drop-in replacement for DashboardWebSocketClient.
    """

    # Same signals as websocket client
    connected = pyqtSignal()
    disconnected = pyqtSignal()
    error = pyqtSignal(str)
    glotec_received = pyqtSignal(dict)

    def __init__(self, update_interval_ms: int = 60000, parent=None):
        """
        Initialize client.

        Args:
            update_interval_ms: How often to check for new data (default 60s)
            parent: Parent QObject
        """
        super().__init__(parent)
        self.update_interval_ms = update_interval_ms
        self.thread: Optional[QThread] = None
        self.worker: Optional[DirectGloTECWorker] = None
        self.logger = logging.getLogger("direct_glotec")

    def start(self):
        """Start the client."""
        if self.thread is not None and self.thread.isRunning():
            self.logger.warning("Client already running")
            return

        self.thread = QThread()
        self.worker = DirectGloTECWorker(self.update_interval_ms)
        self.worker.moveToThread(self.thread)

        # Connect signals
        self.thread.started.connect(self.worker.start_fetching)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.worker.connected.connect(self.connected.emit)
        self.worker.disconnected.connect(self.disconnected.emit)
        self.worker.error.connect(self.error.emit)
        self.worker.glotec_received.connect(self.glotec_received.emit)

        self.thread.start()
        self.logger.info("Direct GloTEC client started")

    def stop(self):
        """Stop the client."""
        if self.worker:
            self.worker.stop_fetching()

        if self.thread and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait(5000)

        self.thread = None
        self.worker = None
        self.logger.info("Direct GloTEC client stopped")

    def isRunning(self) -> bool:
        """Check if running."""
        return self.thread is not None and self.thread.isRunning()
