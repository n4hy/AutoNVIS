"""
Advanced Propagation Data Client - Beyond the Standard Four

Fetches next-generation space weather data for enhanced HF propagation prediction:
- F10.7 Solar Flux - Baseline MUF calculation (EUV proxy)
- Hemispheric Power Index (HPI) - Aurora energy deposition
- D-RAP Absorption - HF blackout zones
- GIRO Ionosonde - Real-time foF2/MUF ground truth
- WSA-Enlil - CME arrival prediction
- Propagated Solar Wind - Near-term solar wind forecast

These sources provide the "ionospheric response" data that the Standard Four
(solar drivers) cannot capture, enabling true "nowcasting" vs mere "forecasting".
"""

import aiohttp
import asyncio
from PyQt6.QtCore import QObject, QThread, pyqtSignal, QTimer
from typing import Optional, Dict, Any, List
import logging
from datetime import datetime, timedelta
import re


class AdvancedDataWorker(QObject):
    """Worker that fetches advanced propagation data sources."""

    # Signals for connection state
    connected = pyqtSignal()
    disconnected = pyqtSignal()
    error = pyqtSignal(str)

    # Signals for each data type
    f107_received = pyqtSignal(dict)          # F10.7 Solar Flux
    hpi_received = pyqtSignal(dict)           # Hemispheric Power Index
    drap_received = pyqtSignal(dict)          # D-RAP Absorption
    ionosonde_received = pyqtSignal(dict)     # GIRO Ionosonde stations
    enlil_received = pyqtSignal(dict)         # WSA-Enlil prediction
    prop_wind_received = pyqtSignal(dict)     # Propagated solar wind

    # NOAA/External Endpoints
    F107_URL = "https://services.swpc.noaa.gov/json/f107_cm_flux.json"
    HPI_URL = "https://services.swpc.noaa.gov/text/aurora-nowcast-hemi-power.txt"
    DRAP_URL = "https://services.swpc.noaa.gov/text/drap_global_frequencies.txt"
    IONOSONDE_URL = "https://prop.kc2g.com/api/stations.json"
    ENLIL_URL = "https://services.swpc.noaa.gov/json/enlil_time_series.json"
    PROP_WIND_URL = "https://services.swpc.noaa.gov/products/geospace/propagated-solar-wind-1-hour.json"

    def __init__(self, update_interval_ms: int = 60000):
        super().__init__()
        self.update_interval_ms = update_interval_ms
        self.running = False
        self.timer: Optional[QTimer] = None
        self.logger = logging.getLogger("advanced_propagation_data")

    def start_fetching(self):
        """Start periodic data fetching."""
        self.running = True
        self.logger.info("Starting advanced data fetch...")

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
        """Fetch all advanced data sources."""
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
        """Fetch all advanced data sources concurrently."""
        async with aiohttp.ClientSession() as session:
            tasks = [
                self._fetch_f107(session),
                self._fetch_hpi(session),
                self._fetch_drap(session),
                self._fetch_ionosonde(session),
                self._fetch_enlil(session),
                self._fetch_propagated_wind(session),
            ]
            await asyncio.gather(*tasks, return_exceptions=True)

    # =========================================================================
    # F10.7 Solar Flux - Critical for foF2/MUF baseline calculation
    # =========================================================================
    async def _fetch_f107(self, session: aiohttp.ClientSession):
        """Fetch F10.7 Solar Radio Flux data.

        F10.7 is the standard proxy for solar EUV output that ionizes the F2 layer.
        Higher F10.7 = higher MUF. Used as baseline for all propagation models.
        """
        try:
            async with session.get(self.F107_URL, timeout=30) as resp:
                if resp.status != 200:
                    return
                data = await resp.json()

            if not data:
                return

            # Get latest entry
            latest = data[-1] if isinstance(data, list) else data

            flux = float(latest.get('flux', 0))
            mean_90 = float(latest.get('ninety_day_mean', flux))

            result = {
                'timestamp': latest.get('time_tag', ''),
                'flux': flux,
                'ninety_day_mean': mean_90,
                'trend': 'Rising' if flux > mean_90 else 'Falling' if flux < mean_90 else 'Stable',
                'muf_impact': self._classify_f107_impact(flux),
            }
            self.f107_received.emit(result)
            self.logger.debug(f"F10.7: {flux:.1f} sfu (90d mean: {mean_90:.1f})")

        except Exception as e:
            self.logger.debug(f"F10.7 fetch error: {e}")

    def _classify_f107_impact(self, flux: float) -> str:
        """Classify F10.7 impact on HF propagation."""
        if flux >= 200:
            return "Excellent"  # 10m/12m wide open
        elif flux >= 150:
            return "Very Good"  # 15m excellent, 10m possible
        elif flux >= 120:
            return "Good"       # 17m/20m excellent
        elif flux >= 90:
            return "Fair"       # 20m/40m good
        elif flux >= 70:
            return "Poor"       # Limited to lower bands
        else:
            return "Very Poor"  # Solar minimum conditions

    # =========================================================================
    # Hemispheric Power Index - Aurora energy deposition
    # =========================================================================
    async def _fetch_hpi(self, session: aiohttp.ClientSession):
        """Fetch Hemispheric Power Index.

        HPI indicates total energy deposited into the auroral zones (GW).
        HPI > 50 GW = aurora visible in northern US
        HPI > 100 GW = major storm, visible at mid-latitudes
        """
        try:
            async with session.get(self.HPI_URL, timeout=30) as resp:
                if resp.status != 200:
                    return
                text = await resp.text()

            # Parse the text file format
            # Format: timestamp, HPI_North, HPI_South, etc.
            lines = text.strip().split('\n')

            # Find last data line (skip headers starting with #)
            data_lines = [l for l in lines if l and not l.startswith('#')]
            if not data_lines:
                return

            latest = data_lines[-1]
            parts = latest.split()

            if len(parts) < 3:
                return

            # Parse: date time hpi_north hpi_south ...
            try:
                timestamp = f"{parts[0]} {parts[1]}"
                hpi_north = float(parts[2])
                hpi_south = float(parts[3]) if len(parts) > 3 else 0
            except (ValueError, IndexError):
                return

            result = {
                'timestamp': timestamp,
                'hpi_north': hpi_north,
                'hpi_south': hpi_south,
                'hpi_total': hpi_north + hpi_south,
                'aurora_visibility': self._classify_hpi(hpi_north),
                'storm_level': self._hpi_storm_level(hpi_north),
            }
            self.hpi_received.emit(result)
            self.logger.debug(f"HPI North: {hpi_north:.1f} GW")

        except Exception as e:
            self.logger.debug(f"HPI fetch error: {e}")

    def _classify_hpi(self, hpi: float) -> str:
        """Classify aurora visibility from HPI."""
        if hpi >= 100:
            return "Mid-latitudes (40°N)"
        elif hpi >= 50:
            return "Northern US (45°N)"
        elif hpi >= 30:
            return "Canada/Alaska (55°N)"
        elif hpi >= 15:
            return "Arctic (65°N)"
        else:
            return "Polar only (70°N+)"

    def _hpi_storm_level(self, hpi: float) -> str:
        """Get storm level from HPI."""
        if hpi >= 200:
            return "Extreme"
        elif hpi >= 100:
            return "Severe"
        elif hpi >= 50:
            return "Strong"
        elif hpi >= 30:
            return "Moderate"
        elif hpi >= 15:
            return "Minor"
        else:
            return "Quiet"

    # =========================================================================
    # D-RAP Absorption - HF blackout zones
    # =========================================================================
    async def _fetch_drap(self, session: aiohttp.ClientSession):
        """Fetch D-Region Absorption Prediction data.

        D-RAP provides the Highest Affected Frequency (HAF) at each location.
        If your frequency < HAF, you're absorbed. Must operate between HAF and MUF.
        """
        try:
            async with session.get(self.DRAP_URL, timeout=30) as resp:
                if resp.status != 200:
                    return
                text = await resp.text()

            # Parse the D-RAP grid format
            lines = text.strip().split('\n')

            # Extract metadata
            timestamp = ""
            max_haf = 0.0
            x_ray_status = "Unknown"
            proton_status = "Unknown"

            data_lines = []
            for line in lines:
                if 'Valid' in line:
                    # Extract timestamp from "Valid: YYYY-MM-DD HH:MM UTC"
                    match = re.search(r'Valid[:\s]+(.+?UTC)', line)
                    if match:
                        timestamp = match.group(1)
                elif 'X-ray' in line:
                    x_ray_status = line.split(':')[-1].strip() if ':' in line else "Unknown"
                elif 'Proton' in line:
                    proton_status = line.split(':')[-1].strip() if ':' in line else "Unknown"
                elif line and not line.startswith('#') and not line.startswith('Lat'):
                    # Data row
                    data_lines.append(line)

            # Parse grid to find max HAF
            for line in data_lines:
                parts = line.split()
                for val in parts[1:]:  # Skip latitude column
                    try:
                        haf = float(val)
                        if haf > max_haf:
                            max_haf = haf
                    except ValueError:
                        continue

            result = {
                'timestamp': timestamp,
                'max_haf': max_haf,
                'x_ray_background': x_ray_status,
                'proton_background': proton_status,
                'absorption_status': self._classify_drap(max_haf),
                'grid_rows': len(data_lines),
            }
            self.drap_received.emit(result)
            self.logger.debug(f"D-RAP max HAF: {max_haf:.1f} MHz")

        except Exception as e:
            self.logger.debug(f"D-RAP fetch error: {e}")

    def _classify_drap(self, max_haf: float) -> str:
        """Classify absorption status from max HAF."""
        if max_haf >= 20:
            return "Severe Blackout"
        elif max_haf >= 10:
            return "Major Absorption"
        elif max_haf >= 5:
            return "Moderate Absorption"
        elif max_haf >= 2:
            return "Minor Absorption"
        else:
            return "Normal"

    # =========================================================================
    # GIRO Ionosonde - Real-time foF2/MUF ground truth
    # =========================================================================
    async def _fetch_ionosonde(self, session: aiohttp.ClientSession):
        """Fetch GIRO ionosonde station data.

        This is the "ground truth" - actual measurements of the ionosphere
        vs model predictions. Provides real foF2, hmF2, and MUF values.
        """
        try:
            async with session.get(self.IONOSONDE_URL, timeout=30) as resp:
                if resp.status != 200:
                    return
                data = await resp.json()

            if not data:
                return

            # Process station data
            stations = []
            global_max_fof2 = 0
            global_max_muf = 0

            for station in data:
                try:
                    fof2 = float(station.get('fof2', 0) or 0)
                    muf = float(station.get('mufd', 0) or 0)
                    hmf2 = float(station.get('hmf2', 0) or 0)

                    if fof2 > global_max_fof2:
                        global_max_fof2 = fof2
                    if muf > global_max_muf:
                        global_max_muf = muf

                    station_info = station.get('station', {})
                    stations.append({
                        'name': station_info.get('name', 'Unknown'),
                        'code': station_info.get('code', ''),
                        'lat': station_info.get('latitude', 0),
                        'lon': station_info.get('longitude', 0),
                        'fof2': fof2,
                        'hmf2': hmf2,
                        'muf': muf,
                        'tec': float(station.get('tec', 0) or 0),
                        'time': station.get('time', ''),
                    })
                except (ValueError, TypeError, KeyError):
                    continue

            # Sort by MUF descending to find best propagation regions
            stations.sort(key=lambda x: x['muf'], reverse=True)

            result = {
                'timestamp': datetime.utcnow().isoformat(),
                'station_count': len(stations),
                'global_max_fof2': global_max_fof2,
                'global_max_muf': global_max_muf,
                'top_stations': stations[:10],  # Top 10 by MUF
                'propagation_outlook': self._classify_ionosonde(global_max_muf),
            }
            self.ionosonde_received.emit(result)
            self.logger.debug(f"Ionosonde: {len(stations)} stations, max MUF {global_max_muf:.1f} MHz")

        except Exception as e:
            self.logger.debug(f"Ionosonde fetch error: {e}")

    def _classify_ionosonde(self, max_muf: float) -> str:
        """Classify propagation from global max MUF."""
        if max_muf >= 35:
            return "Excellent - 10m open"
        elif max_muf >= 28:
            return "Very Good - 12m open"
        elif max_muf >= 21:
            return "Good - 15m open"
        elif max_muf >= 14:
            return "Fair - 20m open"
        elif max_muf >= 7:
            return "Poor - 40m only"
        else:
            return "Very Poor"

    # =========================================================================
    # WSA-Enlil - CME arrival prediction
    # =========================================================================
    async def _fetch_enlil(self, session: aiohttp.ClientSession):
        """Fetch WSA-Enlil solar wind prediction.

        Predicts solar wind conditions 1-4 days ahead, including CME arrivals.
        Critical for anticipating geomagnetic storms before they hit.
        """
        try:
            async with session.get(self.ENLIL_URL, timeout=30) as resp:
                if resp.status != 200:
                    return
                data = await resp.json()

            if not data:
                return

            # Get latest prediction
            latest = data[-1] if isinstance(data, list) else data

            # Extract key parameters
            density = float(latest.get('earth_particles_per_cm3', 0) or 0)
            speed = 0

            # Calculate speed from velocity components if available
            vr = float(latest.get('v_r', 0) or 0)
            if vr:
                speed = abs(vr)

            result = {
                'timestamp': latest.get('time_tag', ''),
                'predicted_density': density,
                'predicted_speed': speed,
                'density_status': self._classify_enlil_density(density),
                'cme_alert': density > 20,  # High density suggests CME
                'record_count': len(data) if isinstance(data, list) else 1,
            }
            self.enlil_received.emit(result)
            self.logger.debug(f"Enlil: density={density:.1f}/cm³")

        except Exception as e:
            self.logger.debug(f"Enlil fetch error: {e}")

    def _classify_enlil_density(self, density: float) -> str:
        """Classify predicted solar wind density."""
        if density >= 30:
            return "CME Likely"
        elif density >= 15:
            return "Enhanced"
        elif density >= 5:
            return "Moderate"
        else:
            return "Quiet"

    # =========================================================================
    # Propagated Solar Wind - Near-term forecast
    # =========================================================================
    async def _fetch_propagated_wind(self, session: aiohttp.ClientSession):
        """Fetch propagated solar wind (1-hour ahead forecast).

        Shows what solar wind conditions will be at Earth in the next hour.
        Provides advance warning for geomagnetic disturbances.
        """
        try:
            async with session.get(self.PROP_WIND_URL, timeout=30) as resp:
                if resp.status != 200:
                    return
                data = await resp.json()

            if not data or len(data) < 2:
                return

            # Skip header row
            header = data[0]
            records = data[1:]

            if not records:
                return

            # Get latest
            latest = records[-1]

            # Parse based on header positions
            # Expected: time_tag, propagated_time_tag, speed, density, temperature, bx, by, bz, bt
            try:
                speed = float(latest[2]) if latest[2] else 0
                density = float(latest[3]) if latest[3] else 0
                temp = float(latest[4]) if latest[4] else 0
                bz = float(latest[7]) if latest[7] else 0
                bt = float(latest[8]) if latest[8] else 0
            except (IndexError, ValueError, TypeError):
                return

            result = {
                'timestamp': latest[0] if latest else '',
                'propagated_time': latest[1] if len(latest) > 1 else '',
                'speed': speed,
                'density': density,
                'temperature': temp,
                'bz': bz,
                'bt': bt,
                'storm_potential': self._classify_storm_potential(speed, bz),
            }
            self.prop_wind_received.emit(result)
            self.logger.debug(f"Prop Wind: speed={speed:.0f} km/s, Bz={bz:.1f} nT")

        except Exception as e:
            self.logger.debug(f"Propagated wind fetch error: {e}")

    def _classify_storm_potential(self, speed: float, bz: float) -> str:
        """Classify geomagnetic storm potential from solar wind."""
        # Southward Bz + high speed = storm driver
        if bz < -10 and speed > 500:
            return "High"
        elif bz < -5 and speed > 400:
            return "Moderate"
        elif bz < 0 and speed > 350:
            return "Low"
        else:
            return "Minimal"


class AdvancedDataClient(QObject):
    """Client that fetches advanced propagation data from multiple sources."""

    # Connection signals
    connected = pyqtSignal()
    disconnected = pyqtSignal()
    error = pyqtSignal(str)

    # Data signals
    f107_received = pyqtSignal(dict)
    hpi_received = pyqtSignal(dict)
    drap_received = pyqtSignal(dict)
    ionosonde_received = pyqtSignal(dict)
    enlil_received = pyqtSignal(dict)
    prop_wind_received = pyqtSignal(dict)

    def __init__(self, update_interval_ms: int = 60000, parent=None):
        super().__init__(parent)
        self.update_interval_ms = update_interval_ms
        self.thread: Optional[QThread] = None
        self.worker: Optional[AdvancedDataWorker] = None
        self.logger = logging.getLogger("advanced_propagation_data")

    def start(self):
        """Start the client."""
        if self.thread is not None and self.thread.isRunning():
            return

        self.thread = QThread()
        self.worker = AdvancedDataWorker(self.update_interval_ms)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.start_fetching)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        # Forward signals
        self.worker.connected.connect(self.connected.emit)
        self.worker.disconnected.connect(self.disconnected.emit)
        self.worker.error.connect(self.error.emit)
        self.worker.f107_received.connect(self.f107_received.emit)
        self.worker.hpi_received.connect(self.hpi_received.emit)
        self.worker.drap_received.connect(self.drap_received.emit)
        self.worker.ionosonde_received.connect(self.ionosonde_received.emit)
        self.worker.enlil_received.connect(self.enlil_received.emit)
        self.worker.prop_wind_received.connect(self.prop_wind_received.emit)

        self.thread.start()
        self.logger.info("Advanced data client started")

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

    def set_update_interval(self, interval_ms: int):
        """Update the fetch interval."""
        self.update_interval_ms = interval_ms
        if self.worker and self.worker.timer:
            self.worker.update_interval_ms = interval_ms
            self.worker.timer.setInterval(interval_ms)
