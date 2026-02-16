"""
Direct GOES X-Ray Client - Fetches directly from NOAA

No RabbitMQ, no WebSocket, no middleware.
"""

import aiohttp
import asyncio
from PyQt6.QtCore import QObject, QThread, pyqtSignal, QTimer
from typing import Optional, Dict, Any, List
import logging
from datetime import datetime, timedelta


# Flare classification thresholds (W/m²)
FLARE_CLASSES = {
    'A': (0, 1e-7),
    'B': (1e-7, 1e-6),
    'C': (1e-6, 1e-5),
    'M': (1e-5, 1e-4),
    'X': (1e-4, float('inf'))
}


class DirectXRayWorker(QObject):
    """Worker that fetches GOES X-ray directly from NOAA."""

    connected = pyqtSignal()
    disconnected = pyqtSignal()
    error = pyqtSignal(str)
    xray_received = pyqtSignal(dict)
    xray_batch_received = pyqtSignal(dict)

    LATEST_URL = "https://services.swpc.noaa.gov/json/goes/primary/xrays-1-day.json"

    def __init__(self, update_interval_ms: int = 60000):
        super().__init__()
        self.update_interval_ms = update_interval_ms
        self.running = False
        self.timer: Optional[QTimer] = None
        self.logger = logging.getLogger("direct_xray")
        self.last_timestamp: Optional[str] = None
        self.historical_loaded = False

    def start_fetching(self):
        """Start periodic data fetching."""
        self.running = True
        self.logger.info("Starting direct GOES X-ray fetch...")

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._fetch_data)
        self.timer.start(self.update_interval_ms)

        self.connected.emit()
        self._fetch_data()

    def stop_fetching(self):
        """Stop fetching."""
        self.running = False
        if self.timer:
            self.timer.stop()
        self.disconnected.emit()

    def _classify_flare(self, flux: float) -> str:
        """Classify flare based on flux."""
        for class_letter, (min_flux, max_flux) in FLARE_CLASSES.items():
            if min_flux <= flux < max_flux:
                if class_letter == 'A':
                    magnitude = flux / 1e-9
                elif class_letter == 'B':
                    magnitude = flux / 1e-8
                elif class_letter == 'C':
                    magnitude = flux / 1e-7
                elif class_letter == 'M':
                    magnitude = flux / 1e-6
                else:
                    magnitude = flux / 1e-5
                return f"{class_letter}{magnitude:.1f}"
        return "A0.0"

    def _fetch_data(self):
        """Fetch data."""
        if not self.running:
            return

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._async_fetch())
            finally:
                loop.close()
        except Exception as e:
            self.logger.error(f"Fetch error: {e}")
            self.error.emit(str(e))

    async def _async_fetch(self):
        """Async fetch from NOAA."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.LATEST_URL, timeout=30) as resp:
                    if resp.status != 200:
                        self.logger.error(f"HTTP {resp.status}")
                        return
                    data = await resp.json()

                if not data:
                    return

                # First fetch: load historical batch
                if not self.historical_loaded:
                    await self._load_historical(data)
                    self.historical_loaded = True

                # Get latest record
                latest = data[-1]
                timestamp = latest.get('time_tag', '')

                # Skip if same as last
                if timestamp == self.last_timestamp:
                    return

                self.last_timestamp = timestamp
                flux = float(latest.get('flux', 0))
                flare_class = self._classify_flare(flux)

                result = {
                    'timestamp': timestamp,
                    'flux_short': flux,
                    'flux_long': flux,
                    'flare_class': flare_class,
                    'm1_or_higher': flux >= 1e-5,
                    'source': 'GOES-Primary'
                }

                self.logger.info(f"X-ray: {flare_class}")
                self.xray_received.emit(result)

        except Exception as e:
            self.logger.error(f"Async fetch error: {e}")

    async def _load_historical(self, data: List[Dict]):
        """Load historical data as batch."""
        cutoff = datetime.utcnow() - timedelta(hours=24)
        records = []

        # Take every 5th record for efficiency
        for i, record in enumerate(data):
            if i % 5 != 0:
                continue
            try:
                ts = datetime.fromisoformat(record['time_tag'].rstrip('Z'))
                if ts >= cutoff:
                    flux = float(record.get('flux', 0))
                    records.append({
                        'timestamp': record.get('time_tag'),
                        'flux_short': flux,
                        'flux_long': flux,
                        'flare_class': self._classify_flare(flux),
                        'm1_or_higher': flux >= 1e-5,
                    })
            except (ValueError, KeyError):
                continue

        if records:
            batch = {
                'type': 'historical_batch',
                'source': 'GOES-Historical',
                'records': records,
                'count': len(records)
            }
            self.logger.info(f"Loaded {len(records)} historical records")
            self.xray_batch_received.emit(batch)


class DirectXRayClient(QObject):
    """
    Client that fetches GOES X-ray directly from NOAA.
    Drop-in replacement for WebSocket client.
    """

    connected = pyqtSignal()
    disconnected = pyqtSignal()
    error = pyqtSignal(str)
    xray_received = pyqtSignal(dict)
    xray_batch_received = pyqtSignal(dict)

    def __init__(self, update_interval_ms: int = 60000, parent=None):
        super().__init__(parent)
        self.update_interval_ms = update_interval_ms
        self.thread: Optional[QThread] = None
        self.worker: Optional[DirectXRayWorker] = None
        self.logger = logging.getLogger("direct_xray")

    def start(self):
        """Start the client."""
        if self.thread is not None and self.thread.isRunning():
            return

        self.thread = QThread()
        self.worker = DirectXRayWorker(self.update_interval_ms)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.start_fetching)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.worker.connected.connect(self.connected.emit)
        self.worker.disconnected.connect(self.disconnected.emit)
        self.worker.error.connect(self.error.emit)
        self.worker.xray_received.connect(self.xray_received.emit)
        self.worker.xray_batch_received.connect(self.xray_batch_received.emit)

        self.thread.start()
        self.logger.info("Direct X-ray client started")

    def stop(self):
        """Stop the client."""
        if self.worker:
            self.worker.stop_fetching()
        if self.thread and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait(5000)
        self.thread = None
        self.worker = None

    def isRunning(self) -> bool:
        return self.thread is not None and self.thread.isRunning()
