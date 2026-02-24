"""
Integrated Live Ionospheric Data Client

Combines real-time data from multiple sources to provide comprehensive
ionospheric conditions for ray tracing calculations:

1. GIRO Ionosonde Network - foF2, hmF2, MUF from global ionosondes
2. NOAA Space Weather - X-ray flux, Kp index, solar wind (affects D-layer absorption)
3. GloTEC - Global TEC maps (optional enhancement)

This client provides a unified interface for the IONORT ray tracing demo
to receive real-time ionospheric data updates.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Callable
from dataclasses import dataclass, field
from PyQt6.QtCore import QObject, QThread, pyqtSignal, QTimer

from .giro_client import (
    GIROClient, GIRODataWorker, GIROStation,
    IonosondeMeasurement, DEFAULT_GIRO_STATIONS,
    generate_simulated_measurement
)

# Import space weather client - handle import error gracefully
try:
    # Try absolute import first (when running as module)
    from src.visualization.pyqt.propagation.data_client import PropagationDataClient
    HAS_SPACE_WEATHER = True
except ImportError:
    try:
        # Try with package prefix removed (when running from project root)
        import sys
        import os
        # Add project root to path if needed
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from src.visualization.pyqt.propagation.data_client import PropagationDataClient
        HAS_SPACE_WEATHER = True
    except ImportError:
        HAS_SPACE_WEATHER = False
        PropagationDataClient = None

logger = logging.getLogger(__name__)


@dataclass
class LiveIonosphericState:
    """Current state of ionospheric conditions from live data."""

    # Primary parameters (from ionosonde)
    foF2: float = 7.0  # MHz
    hmF2: float = 300.0  # km
    foF1: Optional[float] = None
    foE: Optional[float] = None
    MUF3000: Optional[float] = None

    # Source information
    source_station: str = ""
    source_distance_km: float = 0.0
    data_age_seconds: float = 0.0
    is_interpolated: bool = False
    stations_used: int = 0

    # Space weather conditions
    kp_index: float = 2.0
    r_scale: int = 0  # X-ray/radio blackout scale (0-5)
    s_scale: int = 0  # Solar radiation storm scale (0-5)
    g_scale: int = 0  # Geomagnetic storm scale (0-5)
    xray_flux: float = 1e-6  # W/m²
    bz_gsm: float = 0.0  # nT, negative = southward

    # D-layer absorption factor (derived from X-ray flux)
    d_layer_absorption_db: float = 0.0

    # Timestamps
    iono_timestamp: Optional[datetime] = None
    space_wx_timestamp: Optional[datetime] = None

    # Confidence/quality
    overall_confidence: float = 0.5

    def to_dict(self) -> Dict:
        """Convert to dictionary for UI display."""
        return {
            'foF2': f"{self.foF2:.2f} MHz",
            'hmF2': f"{self.hmF2:.0f} km",
            'foE': f"{self.foE:.2f} MHz" if self.foE else "N/A",
            'MUF3000': f"{self.MUF3000:.1f} MHz" if self.MUF3000 else "N/A",
            'source': self.source_station or "Default",
            'data_age': f"{self.data_age_seconds:.0f}s" if self.data_age_seconds < 3600 else f"{self.data_age_seconds/3600:.1f}h",
            'Kp': f"{self.kp_index:.1f}",
            'R-scale': f"R{self.r_scale}",
            'D-absorption': f"{self.d_layer_absorption_db:.1f} dB",
            'confidence': f"{self.overall_confidence:.0%}",
        }


class LiveIonoClient(QObject):
    """
    Integrated live ionospheric data client.

    Provides real-time ionospheric parameters by combining:
    - Ionosonde data (GIRO network)
    - Space weather data (NOAA SWPC)

    Emits signals when new data arrives, which can be connected to
    update the ionospheric model in real-time.
    """

    # Signals
    state_updated = pyqtSignal(object)  # LiveIonosphericState
    ionosonde_updated = pyqtSignal(object)  # IonosondeMeasurement
    space_weather_updated = pyqtSignal(dict)  # Combined space weather
    error = pyqtSignal(str)
    connected = pyqtSignal()
    disconnected = pyqtSignal()

    def __init__(
        self,
        reference_lat: float = 40.0,
        reference_lon: float = -105.0,
        update_interval_ms: int = 60000,
        enable_giro: bool = True,
        enable_space_weather: bool = True,
        use_simulated: bool = False,
        parent=None
    ):
        """
        Initialize live ionospheric data client.

        Args:
            reference_lat: Reference latitude for nearest station selection
            reference_lon: Reference longitude for nearest station selection
            update_interval_ms: Update interval in milliseconds
            enable_giro: Enable GIRO ionosonde data
            enable_space_weather: Enable space weather data
            use_simulated: Use simulated data (for testing without network)
            parent: Parent QObject
        """
        super().__init__(parent)

        self.reference_lat = reference_lat
        self.reference_lon = reference_lon
        self.update_interval_ms = update_interval_ms
        self.enable_giro = enable_giro
        self.enable_space_weather = enable_space_weather
        self.use_simulated = use_simulated

        # Current state
        self.state = LiveIonosphericState()

        # Sub-clients
        self.giro_client: Optional[GIROClient] = None
        self.space_weather_client = None

        # For simulated mode
        self._sim_timer: Optional[QTimer] = None

        # Latest raw data
        self._latest_iono: Optional[IonosondeMeasurement] = None
        self._latest_space_wx: Dict = {}

    def start(self):
        """Start all data clients."""
        logger.info("Starting live ionospheric data client...")

        if self.use_simulated:
            self._start_simulated()
            self.connected.emit()
            return

        # Start GIRO client
        if self.enable_giro:
            self.giro_client = GIROClient(
                update_interval_ms=self.update_interval_ms * 5  # Less frequent for ionosonde
            )
            self.giro_client.measurement_received.connect(self._on_ionosonde_data)
            self.giro_client.error.connect(self._on_giro_error)
            self.giro_client.start()

        # Start space weather client
        if self.enable_space_weather and HAS_SPACE_WEATHER:
            self.space_weather_client = PropagationDataClient(
                update_interval_ms=self.update_interval_ms
            )
            self.space_weather_client.xray_received.connect(self._on_xray_data)
            self.space_weather_client.kp_received.connect(self._on_kp_data)
            self.space_weather_client.solarwind_received.connect(self._on_solarwind_data)
            self.space_weather_client.start()

        self.connected.emit()

    def stop(self):
        """Stop all data clients."""
        if self.giro_client:
            self.giro_client.stop()
            self.giro_client = None

        if self.space_weather_client:
            self.space_weather_client.stop()
            self.space_weather_client = None

        if self._sim_timer:
            self._sim_timer.stop()
            self._sim_timer = None

        self.disconnected.emit()

    def set_reference_location(self, lat: float, lon: float):
        """Update reference location for nearest station selection."""
        self.reference_lat = lat
        self.reference_lon = lon

        # Re-evaluate current data with new reference
        self._update_state()

    def _start_simulated(self):
        """Start simulated data generation."""
        logger.info("Using simulated ionospheric data")

        self._sim_timer = QTimer(self)
        self._sim_timer.timeout.connect(self._generate_simulated_data)
        self._sim_timer.start(self.update_interval_ms)

        # Initial data
        self._generate_simulated_data()

    def _generate_simulated_data(self):
        """Generate simulated ionosonde measurement."""
        # Find nearest default station
        nearest_station = None
        nearest_dist = float('inf')
        for station in DEFAULT_GIRO_STATIONS:
            dist = station.distance_to(self.reference_lat, self.reference_lon)
            if dist < nearest_dist:
                nearest_station = station
                nearest_dist = dist

        if nearest_station:
            measurement = generate_simulated_measurement(nearest_station)
            self._on_ionosonde_data(measurement)

        # Simulated space weather (quiet conditions)
        self._on_xray_data({'flux': 1e-6, 'r_scale': 0})
        self._on_kp_data({'kp_index': 2.0, 'g_scale': 0})

    def _on_ionosonde_data(self, measurement: IonosondeMeasurement):
        """Handle ionosonde measurement."""
        # Check if this station is relevant (within range)
        dist = measurement.station.distance_to(self.reference_lat, self.reference_lon)

        # Accept if within 2000 km, prefer closer stations
        if dist > 2000:
            return

        # Update if closer than current or current is stale
        current_dist = self.state.source_distance_km
        current_age = self.state.data_age_seconds

        if (self._latest_iono is None or
            dist < current_dist * 0.8 or
            current_age > 3600):

            self._latest_iono = measurement
            self.ionosonde_updated.emit(measurement)

            logger.info(
                f"Ionosonde update from {measurement.station.name}: "
                f"foF2={measurement.foF2:.2f} MHz, hmF2={measurement.hmF2:.0f} km "
                f"(dist={dist:.0f} km)"
            )

        self._update_state()

    def _on_xray_data(self, data: Dict):
        """Handle X-ray flux data."""
        self._latest_space_wx['xray'] = data
        self._update_state()

    def _on_kp_data(self, data: Dict):
        """Handle Kp index data."""
        self._latest_space_wx['kp'] = data
        self._update_state()

    def _on_solarwind_data(self, data: Dict):
        """Handle solar wind data."""
        self._latest_space_wx['solarwind'] = data
        self._update_state()

    def _on_giro_error(self, error_msg: str):
        """Handle GIRO client error."""
        logger.warning(f"GIRO error: {error_msg}")
        self.error.emit(f"Ionosonde: {error_msg}")

    def _update_state(self):
        """Update combined state from all data sources."""
        now = datetime.now(timezone.utc)

        # Update from ionosonde data
        if self._latest_iono:
            iono = self._latest_iono
            self.state.foF2 = iono.foF2
            self.state.hmF2 = iono.hmF2
            self.state.foF1 = iono.foF1
            self.state.foE = iono.foE
            self.state.MUF3000 = iono.MUF3000
            self.state.source_station = iono.station.code
            self.state.source_distance_km = iono.station.distance_to(
                self.reference_lat, self.reference_lon
            )
            self.state.data_age_seconds = iono.age_seconds()
            self.state.iono_timestamp = iono.timestamp

            # Confidence based on distance and age
            dist_factor = max(0, 1 - self.state.source_distance_km / 2000)
            age_factor = max(0, 1 - self.state.data_age_seconds / 3600)
            self.state.overall_confidence = 0.5 + 0.5 * dist_factor * age_factor

        # Update from space weather
        if 'xray' in self._latest_space_wx:
            xray = self._latest_space_wx['xray']
            self.state.xray_flux = xray.get('flux', 1e-6)
            self.state.r_scale = xray.get('r_scale', 0)

            # Calculate D-layer absorption from X-ray flux
            # Absorption increases with solar X-ray flux
            # Rough model: absorption = 10 * log10(flux / 1e-6) dB at MF
            if self.state.xray_flux > 1e-7:
                import math
                self.state.d_layer_absorption_db = max(0, 10 * math.log10(self.state.xray_flux / 1e-6))
            else:
                self.state.d_layer_absorption_db = 0

        if 'kp' in self._latest_space_wx:
            kp = self._latest_space_wx['kp']
            self.state.kp_index = kp.get('kp_index', 2.0)
            self.state.g_scale = kp.get('g_scale', 0)

        if 'solarwind' in self._latest_space_wx:
            sw = self._latest_space_wx['solarwind']
            self.state.bz_gsm = sw.get('bz_gsm', 0)

        self.state.space_wx_timestamp = now

        # Emit combined state
        self.state_updated.emit(self.state)

        # Emit space weather summary
        if self._latest_space_wx:
            self.space_weather_updated.emit(self._latest_space_wx.copy())

    def get_current_state(self) -> LiveIonosphericState:
        """Get current ionospheric state."""
        return self.state

    def get_foF2_hmF2(self) -> tuple:
        """Get current foF2 and hmF2 values."""
        return self.state.foF2, self.state.hmF2

    def is_data_valid(self, max_age_seconds: float = 3600) -> bool:
        """Check if current data is valid (not too stale)."""
        if self._latest_iono is None:
            return False
        return self.state.data_age_seconds < max_age_seconds


class LiveIonoModelUpdater:
    """
    Utility class that connects LiveIonoClient to IonosphericModel.

    Automatically updates the ionospheric model when new live data arrives.
    """

    def __init__(
        self,
        client: LiveIonoClient,
        model,  # IonosphericModel (avoid circular import)
    ):
        self.client = client
        self.model = model

        # Connect signals
        self.client.state_updated.connect(self._on_state_updated)

    def _on_state_updated(self, state: LiveIonosphericState):
        """Update model with new live data."""
        self.model.update_from_realtime(
            foF2=state.foF2,
            hmF2=state.hmF2,
            MUF=state.MUF3000,
        )

        logger.debug(
            f"Model updated: foF2={state.foF2:.2f}, hmF2={state.hmF2:.0f}, "
            f"source={state.source_station}"
        )


if __name__ == "__main__":
    # Test the live client
    import sys
    from PyQt6.QtWidgets import QApplication

    logging.basicConfig(level=logging.INFO)

    app = QApplication(sys.argv)

    print("Live Ionospheric Data Client Test")
    print("=" * 50)

    # Test with simulated data
    client = LiveIonoClient(
        reference_lat=40.0,
        reference_lon=-105.0,
        use_simulated=True,
        update_interval_ms=2000
    )

    def on_state_update(state: LiveIonosphericState):
        print(f"\nState Update:")
        for key, val in state.to_dict().items():
            print(f"  {key}: {val}")

    client.state_updated.connect(on_state_update)
    client.start()

    # Run for a few updates
    QTimer.singleShot(10000, app.quit)

    print("\nWaiting for data updates (10 seconds)...")
    app.exec()

    client.stop()
    print("\nDone!")
