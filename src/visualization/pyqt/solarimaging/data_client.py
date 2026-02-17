"""
Solar Image Data Client

Concurrent fetcher for solar images from NOAA SWPC and Helioviewer API.
Uses asyncio for parallel fetching and PyQt6 signals for thread-safe updates.
"""

import aiohttp
import asyncio
from datetime import datetime
from typing import Optional, Dict, List, Any
import logging

from PyQt6.QtCore import QObject, QThread, pyqtSignal, QTimer

from .sources import SOLAR_SOURCES, HELIOVIEWER_API_URL, get_all_sources


class SolarImageWorker(QObject):
    """Worker that fetches solar images in a background thread."""

    # Signals
    image_received = pyqtSignal(str, bytes, str)  # source_id, image_data, timestamp
    error = pyqtSignal(str, str)  # source_id, error_message
    connected = pyqtSignal()
    disconnected = pyqtSignal()
    fetch_started = pyqtSignal()
    fetch_completed = pyqtSignal(int, int)  # success_count, total_count

    def __init__(self, fast_interval_ms: int = 60000, slow_interval_ms: int = 900000):
        super().__init__()
        self.fast_interval_ms = fast_interval_ms
        self.slow_interval_ms = slow_interval_ms
        self.running = False
        self.fast_timer: Optional[QTimer] = None
        self.slow_timer: Optional[QTimer] = None
        self.logger = logging.getLogger("solar_image_client")
        self.last_fetch_time: Dict[str, datetime] = {}

    def start_fetching(self):
        """Start periodic image fetching."""
        self.running = True
        self.logger.info("Starting solar image fetch...")

        # Fast timer for SUVI, AIA, HMI (60 second cadence)
        self.fast_timer = QTimer(self)
        self.fast_timer.timeout.connect(self._fetch_fast_sources)
        self.fast_timer.start(self.fast_interval_ms)

        # Slow timer for LASCO, EIT (15 minute cadence)
        self.slow_timer = QTimer(self)
        self.slow_timer.timeout.connect(self._fetch_slow_sources)
        self.slow_timer.start(self.slow_interval_ms)

        self.connected.emit()

        # Initial fetch of all sources
        self._fetch_all_sources()

    def stop_fetching(self):
        """Stop fetching."""
        self.running = False
        if self.fast_timer:
            self.fast_timer.stop()
        if self.slow_timer:
            self.slow_timer.stop()
        self.disconnected.emit()

    def _fetch_fast_sources(self):
        """Fetch fast-cadence sources (SUVI, AIA, HMI)."""
        if not self.running:
            return
        categories = ['suvi', 'aia', 'hmi']
        self._run_fetch(categories)

    def _fetch_slow_sources(self):
        """Fetch slow-cadence sources (LASCO, EIT)."""
        if not self.running:
            return
        categories = ['lasco', 'eit']
        self._run_fetch(categories)

    def _fetch_all_sources(self):
        """Fetch all sources."""
        if not self.running:
            return
        categories = list(SOLAR_SOURCES.keys())
        self._run_fetch(categories)

    def _run_fetch(self, categories: List[str]):
        """Run async fetch in a new event loop."""
        loop = None
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._async_fetch_categories(categories))
        except Exception as e:
            self.logger.error(f"Fetch error: {e}")
        finally:
            if loop:
                try:
                    loop.close()
                except Exception:
                    pass

    async def _async_fetch_categories(self, categories: List[str]):
        """Fetch all images in specified categories concurrently."""
        self.fetch_started.emit()

        tasks = []
        async with aiohttp.ClientSession() as session:
            for cat_key in categories:
                category = SOLAR_SOURCES.get(cat_key)
                if not category:
                    continue

                for img in category['images']:
                    if category['type'] == 'noaa':
                        tasks.append(self._fetch_noaa_image(session, img))
                    else:
                        tasks.append(self._fetch_helioviewer_image(session, img, category['name']))

            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Count successes
        success_count = sum(1 for r in results if r is not None and not isinstance(r, Exception))
        self.fetch_completed.emit(success_count, len(tasks))

    async def _fetch_noaa_image(self, session: aiohttp.ClientSession, source: Dict) -> Optional[Dict]:
        """Fetch image directly from NOAA SWPC."""
        source_id = source['id']
        url = source['url']

        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status != 200:
                    self.error.emit(source_id, f"HTTP {resp.status}")
                    return None

                data = await resp.read()

                # Get timestamp from Last-Modified header or current time
                last_modified = resp.headers.get('Last-Modified')
                if last_modified:
                    try:
                        # Parse HTTP date format
                        timestamp = datetime.strptime(last_modified, '%a, %d %b %Y %H:%M:%S %Z')
                        timestamp_str = timestamp.strftime('%Y-%m-%dT%H:%M:%SZ')
                    except ValueError:
                        timestamp_str = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
                else:
                    timestamp_str = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')

                self.image_received.emit(source_id, data, timestamp_str)
                self.last_fetch_time[source_id] = datetime.utcnow()
                return {'id': source_id, 'size': len(data)}

        except asyncio.TimeoutError:
            self.error.emit(source_id, "Timeout")
            return None
        except Exception as e:
            self.error.emit(source_id, str(e))
            return None

    async def _fetch_helioviewer_image(self, session: aiohttp.ClientSession,
                                        source: Dict, category_name: str) -> Optional[Dict]:
        """Fetch image from Helioviewer takeScreenshot API."""
        source_id = source['id']
        source_id_num = source['sourceId']

        # Build request parameters
        # Sun diameter is ~1920 arcsec (1 Rs = 960 arcsec)
        # Disk imagers: 2.4 arcsec/pixel for full disk in 1024x1024
        # Coronagraphs need larger FOV: C2 ~12 Rs, C3 ~60 Rs
        image_scale = source.get('imageScale', 2.4)  # Default for disk imagers

        params = {
            'date': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
            'imageScale': image_scale,
            'layers': f"[{source_id_num},1,100]",
            'x0': 0,
            'y0': 0,
            'width': 1024,
            'height': 1024,
            'display': 'true',  # Return PNG directly
            'watermark': 'false',
        }

        try:
            async with session.get(
                HELIOVIEWER_API_URL,
                params=params,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as resp:
                if resp.status != 200:
                    self.error.emit(source_id, f"HTTP {resp.status}")
                    return None

                content_type = resp.headers.get('Content-Type', '')
                if 'image' not in content_type and 'png' not in content_type.lower():
                    # Might be JSON error response
                    text = await resp.text()
                    self.error.emit(source_id, f"Not an image: {text[:100]}")
                    return None

                data = await resp.read()
                timestamp_str = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')

                self.image_received.emit(source_id, data, timestamp_str)
                self.last_fetch_time[source_id] = datetime.utcnow()
                return {'id': source_id, 'size': len(data)}

        except asyncio.TimeoutError:
            self.error.emit(source_id, "Timeout")
            return None
        except Exception as e:
            self.error.emit(source_id, str(e))
            return None


class SolarImageDataClient(QObject):
    """Main client class that manages the worker thread."""

    # Forward signals from worker
    image_received = pyqtSignal(str, bytes, str)
    error = pyqtSignal(str, str)
    connected = pyqtSignal()
    disconnected = pyqtSignal()
    fetch_started = pyqtSignal()
    fetch_completed = pyqtSignal(int, int)

    def __init__(self, fast_interval_ms: int = 60000, slow_interval_ms: int = 900000, parent=None):
        super().__init__(parent)
        self.fast_interval_ms = fast_interval_ms
        self.slow_interval_ms = slow_interval_ms
        self.thread: Optional[QThread] = None
        self.worker: Optional[SolarImageWorker] = None
        self.logger = logging.getLogger("solar_image_client")

    def start(self):
        """Start the client and begin fetching."""
        if self.thread is not None and self.thread.isRunning():
            return

        self.thread = QThread()
        self.worker = SolarImageWorker(self.fast_interval_ms, self.slow_interval_ms)
        self.worker.moveToThread(self.thread)

        # Connect thread lifecycle
        self.thread.started.connect(self.worker.start_fetching)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        # Forward signals
        self.worker.image_received.connect(self.image_received.emit)
        self.worker.error.connect(self.error.emit)
        self.worker.connected.connect(self.connected.emit)
        self.worker.disconnected.connect(self.disconnected.emit)
        self.worker.fetch_started.connect(self.fetch_started.emit)
        self.worker.fetch_completed.connect(self.fetch_completed.emit)

        self.thread.start()
        self.logger.info("Solar image data client started")

    def stop(self):
        """Stop the client."""
        if self.worker:
            self.worker.stop_fetching()
        if self.thread and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait(5000)
        self.thread = None
        self.worker = None

    def is_running(self) -> bool:
        """Check if client is running."""
        return self.thread is not None and self.thread.isRunning()

    def refresh_all(self):
        """Trigger a manual refresh of all sources."""
        if self.worker:
            self.worker._fetch_all_sources()
