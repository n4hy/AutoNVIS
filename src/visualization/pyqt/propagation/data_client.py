"""
Propagation Data Client - Fetches all propagation data from NOAA

Fetches:
- X-ray flux (GOES) - R scale
- Kp index - G scale
- Proton flux (GOES) - S scale
- Solar wind Bz (DSCOVR)
"""

import aiohttp
import asyncio
from PyQt6.QtCore import QObject, QThread, pyqtSignal, QTimer
from typing import Optional, Dict, Any, List
import logging
from datetime import datetime, timedelta


class PropagationDataWorker(QObject):
    """Worker that fetches all propagation data from NOAA."""

    # Signals for each data type
    connected = pyqtSignal()
    disconnected = pyqtSignal()
    error = pyqtSignal(str)

    xray_received = pyqtSignal(dict)
    kp_received = pyqtSignal(dict)
    proton_received = pyqtSignal(dict)
    solarwind_received = pyqtSignal(dict)

    # NOAA endpoints
    XRAY_URL = "https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json"
    KP_URL = "https://services.swpc.noaa.gov/json/planetary_k_index_1m.json"
    PROTON_URL = "https://services.swpc.noaa.gov/json/goes/primary/integral-protons-1-day.json"
    SOLARWIND_URL = "https://services.swpc.noaa.gov/products/solar-wind/mag-1-day.json"

    def __init__(self, update_interval_ms: int = 60000):
        super().__init__()
        self.update_interval_ms = update_interval_ms
        self.running = False
        self.timer: Optional[QTimer] = None
        self.logger = logging.getLogger("propagation_data")
        self.historical_loaded = False

    def start_fetching(self):
        """Start periodic data fetching."""
        self.running = True
        self.logger.info("Starting propagation data fetch...")

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

    def _fetch_data(self):
        """Fetch all data sources."""
        if not self.running:
            return

        loop = None
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._async_fetch_all())
        except asyncio.TimeoutError:
            self.logger.warning("Fetch timeout")
        except Exception as e:
            self.logger.error(f"Fetch error: {e}")
            self.error.emit(str(e))
        finally:
            if loop:
                try:
                    loop.close()
                except Exception:
                    pass

    async def _async_fetch_all(self):
        """Fetch all data sources concurrently."""
        load_history = not self.historical_loaded
        if load_history:
            self.historical_loaded = True  # Set early to prevent race condition
            self.logger.info("First fetch - loading historical data...")

        async with aiohttp.ClientSession() as session:
            # Fetch all in parallel
            tasks = [
                self._fetch_xray(session, load_history),
                self._fetch_kp(session, load_history),
                self._fetch_proton(session, load_history),
                self._fetch_solarwind(session, load_history),
            ]
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _fetch_xray(self, session: aiohttp.ClientSession, load_history: bool = False):
        """Fetch X-ray flux data."""
        try:
            async with session.get(self.XRAY_URL, timeout=30) as resp:
                if resp.status != 200:
                    return
                data = await resp.json()

            if not data:
                return

            # Load historical on first fetch
            if load_history:
                await self._emit_xray_history(data)

            # Get latest
            latest = data[-1]
            flux = float(latest.get('flux', 0))

            result = {
                'timestamp': latest.get('time_tag', ''),
                'flux': flux,
                'flare_class': self._classify_flare(flux),
                'r_scale': self._get_r_scale(flux),
            }
            self.xray_received.emit(result)

        except Exception as e:
            self.logger.debug(f"X-ray fetch error: {e}")

    async def _emit_xray_history(self, data: List[Dict]):
        """Emit historical X-ray data."""
        # Filter to last 24 hours first, then sample
        cutoff = datetime.utcnow() - timedelta(days=1)
        filtered = []
        for record in data:
            try:
                ts = datetime.fromisoformat(record['time_tag'].rstrip('Z'))
                if ts >= cutoff:
                    filtered.append(record)
            except (ValueError, KeyError):
                continue

        # Sample every 5th from filtered data
        self.logger.info(f"Loading {len(filtered)} X-ray records from last 24h")
        for i, record in enumerate(filtered):
            if i % 5 != 0:
                continue
            try:
                flux = float(record.get('flux', 0))
                result = {
                    'timestamp': record.get('time_tag'),
                    'flux': flux,
                    'flare_class': self._classify_flare(flux),
                    'r_scale': self._get_r_scale(flux),
                }
                self.xray_received.emit(result)
            except (ValueError, KeyError):
                continue

    def _classify_flare(self, flux: float) -> str:
        """Classify flare based on flux."""
        if flux >= 1e-4:
            return f"X{flux/1e-4:.1f}"
        elif flux >= 1e-5:
            return f"M{flux/1e-5:.1f}"
        elif flux >= 1e-6:
            return f"C{flux/1e-6:.1f}"
        elif flux >= 1e-7:
            return f"B{flux/1e-7:.1f}"
        else:
            return f"A{flux/1e-8:.1f}"

    def _get_r_scale(self, flux: float) -> int:
        """Get NOAA R-scale (0-5) from X-ray flux."""
        if flux >= 1e-3:  # X10+
            return 5
        elif flux >= 1e-4:  # X1+
            return 4
        elif flux >= 5e-5:  # M5+
            return 3
        elif flux >= 1e-5:  # M1+
            return 2
        elif flux >= 1e-6:  # C1+
            return 1
        return 0

    async def _fetch_kp(self, session: aiohttp.ClientSession, load_history: bool = False):
        """Fetch Kp index data."""
        try:
            async with session.get(self.KP_URL, timeout=30) as resp:
                if resp.status != 200:
                    return
                data = await resp.json()

            if not data:
                return

            # Load historical on first fetch
            if load_history:
                await self._emit_kp_history(data)

            # Get latest
            latest = data[-1]

            result = {
                'timestamp': latest.get('time_tag', ''),
                'kp_index': latest.get('kp_index', 0),
                'estimated_kp': latest.get('estimated_kp', 0),
                'kp': latest.get('kp', '0'),
                'g_scale': self._get_g_scale(latest.get('kp_index', 0)),
            }
            self.kp_received.emit(result)

        except Exception as e:
            self.logger.debug(f"Kp fetch error: {e}")

    async def _emit_kp_history(self, data: List[Dict]):
        """Emit historical Kp data (every 10th record for efficiency)."""
        self.logger.info(f"Loading {len(data)} Kp records")
        for i, record in enumerate(data):
            if i % 10 != 0:
                continue
            try:
                result = {
                    'timestamp': record.get('time_tag', ''),
                    'kp_index': record.get('kp_index', 0),
                    'estimated_kp': record.get('estimated_kp', 0),
                    'kp': record.get('kp', '0'),
                    'g_scale': self._get_g_scale(record.get('kp_index', 0)),
                }
                self.kp_received.emit(result)
            except (ValueError, KeyError):
                continue

    def _get_g_scale(self, kp: int) -> int:
        """Get NOAA G-scale (0-5) from Kp index."""
        if kp >= 9:
            return 5
        elif kp >= 8:
            return 4
        elif kp >= 7:
            return 3
        elif kp >= 6:
            return 2
        elif kp >= 5:
            return 1
        return 0

    async def _fetch_proton(self, session: aiohttp.ClientSession, load_history: bool = False):
        """Fetch proton flux data."""
        try:
            async with session.get(self.PROTON_URL, timeout=30) as resp:
                if resp.status != 200:
                    return
                data = await resp.json()

            if not data:
                return

            # Filter for >=10 MeV (S-scale threshold)
            proton_10mev = [r for r in data if r.get('energy') == '>=10 MeV']

            if not proton_10mev:
                return

            # Load historical on first fetch
            if load_history:
                await self._emit_proton_history(proton_10mev)

            # Get latest
            latest = proton_10mev[-1]
            flux = float(latest.get('flux', 0))

            result = {
                'timestamp': latest.get('time_tag', ''),
                'flux': flux,
                'energy': '>=10 MeV',
                's_scale': self._get_s_scale(flux),
            }
            self.proton_received.emit(result)

        except Exception as e:
            self.logger.debug(f"Proton fetch error: {e}")

    async def _emit_proton_history(self, data: List[Dict]):
        """Emit historical proton data."""
        self.logger.info(f"Loading {len(data)} proton records")
        for i, record in enumerate(data):
            if i % 3 != 0:  # Every 3rd record
                continue
            try:
                flux = float(record.get('flux', 0))
                result = {
                    'timestamp': record.get('time_tag', ''),
                    'flux': flux,
                    'energy': '>=10 MeV',
                    's_scale': self._get_s_scale(flux),
                }
                self.proton_received.emit(result)
            except (ValueError, KeyError):
                continue

    def _get_s_scale(self, flux: float) -> int:
        """Get NOAA S-scale (0-5) from >=10 MeV proton flux."""
        if flux >= 1e5:
            return 5
        elif flux >= 1e4:
            return 4
        elif flux >= 1e3:
            return 3
        elif flux >= 1e2:
            return 2
        elif flux >= 10:
            return 1
        return 0

    async def _fetch_solarwind(self, session: aiohttp.ClientSession, load_history: bool = False):
        """Fetch solar wind magnetic field data."""
        try:
            async with session.get(self.SOLARWIND_URL, timeout=30) as resp:
                if resp.status != 200:
                    return
                data = await resp.json()

            if not data or len(data) < 2:
                return

            # Skip header row
            records = data[1:]

            # Load historical on first fetch
            if load_history:
                await self._emit_solarwind_history(records)

            # Get latest
            latest = records[-1]

            try:
                bz = float(latest[3]) if latest[3] else 0
                bt = float(latest[6]) if latest[6] else 0
            except (IndexError, ValueError, TypeError):
                bz = 0
                bt = 0

            result = {
                'timestamp': latest[0] if latest else '',
                'bz_gsm': bz,
                'bt': bt,
                'bz_status': 'Southward' if bz < 0 else 'Northward',
            }
            self.solarwind_received.emit(result)

        except Exception as e:
            self.logger.debug(f"Solar wind fetch error: {e}")

    async def _emit_solarwind_history(self, records: List):
        """Emit historical solar wind data."""
        self.logger.info(f"Loading {len(records)} solar wind records")
        for i, record in enumerate(records):
            if i % 10 != 0:  # Every 10th record
                continue
            try:
                bz = float(record[3]) if record[3] else 0
                bt = float(record[6]) if record[6] else 0
                result = {
                    'timestamp': record[0] if record else '',
                    'bz_gsm': bz,
                    'bt': bt,
                    'bz_status': 'Southward' if bz < 0 else 'Northward',
                }
                self.solarwind_received.emit(result)
            except (IndexError, ValueError, TypeError):
                continue


class PropagationDataClient(QObject):
    """Client that fetches all propagation data from NOAA."""

    connected = pyqtSignal()
    disconnected = pyqtSignal()
    error = pyqtSignal(str)

    xray_received = pyqtSignal(dict)
    kp_received = pyqtSignal(dict)
    proton_received = pyqtSignal(dict)
    solarwind_received = pyqtSignal(dict)

    def __init__(self, update_interval_ms: int = 60000, parent=None):
        super().__init__(parent)
        self.update_interval_ms = update_interval_ms
        self.thread: Optional[QThread] = None
        self.worker: Optional[PropagationDataWorker] = None
        self.logger = logging.getLogger("propagation_data")

    def start(self):
        """Start the client."""
        if self.thread is not None and self.thread.isRunning():
            return

        self.thread = QThread()
        self.worker = PropagationDataWorker(self.update_interval_ms)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.start_fetching)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        # Forward signals
        self.worker.connected.connect(self.connected.emit)
        self.worker.disconnected.connect(self.disconnected.emit)
        self.worker.error.connect(self.error.emit)
        self.worker.xray_received.connect(self.xray_received.emit)
        self.worker.kp_received.connect(self.kp_received.emit)
        self.worker.proton_received.connect(self.proton_received.emit)
        self.worker.solarwind_received.connect(self.solarwind_received.emit)

        self.thread.start()
        self.logger.info("Propagation data client started")

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
