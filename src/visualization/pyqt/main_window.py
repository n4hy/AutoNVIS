"""
TEC Display Main Window

Main window integrating all TEC visualization widgets.
"""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QToolBar, QMessageBox
)
from PyQt6.QtCore import Qt, QTimer, pyqtSlot
from PyQt6.QtGui import QAction
import pyqtgraph as pg

from .widgets.tec_map_widget import TECMapWidget
from .widgets.tec_timeseries_widget import TECTimeSeriesWidget
from .widgets.ionosphere_profile_widget import IonosphereProfileWidget
from .widgets.status_bar_widget import StatusBarWidget
from .data.data_manager import DataManager


class TECDisplayMainWindow(QMainWindow):
    """
    Main window for TEC visualization application.

    Layout:
    +----------------------------------------------------------+
    | Toolbar: [Connect] [Theme] [Clear]                       |
    +----------------------------------------------------------+
    | +------------------------+ +---------------------------+ |
    | |                        | |    TEC Time Series        | |
    | |   Global TEC Map       | |    - Global mean          | |
    | |   (color-coded)        | |    - Selected point       | |
    | |                        | +---------------------------+ |
    | |                        | +---------------------------+ |
    | +------------------------+ |  Ionosphere Profile       | |
    |                            |  - hmF2, NmF2             | |
    |                            +---------------------------+ |
    +----------------------------------------------------------+
    | Status: Connected | Last: 12:30:00 UTC | TEC Mean: 25   |
    +----------------------------------------------------------+
    """

    def __init__(self, parent=None):
        """Initialize main window."""
        super().__init__(parent)

        self.setWindowTitle("AutoNVIS TEC Display")
        self.setMinimumSize(1200, 800)

        # Initialize pyqtgraph
        pg.setConfigOptions(antialias=True)

        # Data manager
        self.data_manager = DataManager()

        # WebSocket client (set externally)
        self.ws_client = None

        # Create UI
        self._setup_ui()
        self._setup_toolbar()
        self._connect_signals()

        # Update timer for periodic refresh
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self._on_refresh)
        self.refresh_timer.start(100)  # 10 Hz refresh rate

    def _setup_ui(self):
        """Create and arrange all UI elements."""
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)

        # Main horizontal splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel: TEC Map
        self.tec_map_widget = TECMapWidget()
        main_splitter.addWidget(self.tec_map_widget)

        # Right panel: Time series and profiles
        right_splitter = QSplitter(Qt.Orientation.Vertical)

        self.timeseries_widget = TECTimeSeriesWidget()
        right_splitter.addWidget(self.timeseries_widget)

        self.profile_widget = IonosphereProfileWidget()
        right_splitter.addWidget(self.profile_widget)

        # Set right panel proportions
        right_splitter.setSizes([300, 200])

        main_splitter.addWidget(right_splitter)

        # Set main splitter proportions (map gets more space)
        main_splitter.setSizes([700, 500])

        layout.addWidget(main_splitter)

        # Status bar
        self.status_bar = StatusBarWidget()
        self.setStatusBar(self.status_bar)

    def _setup_toolbar(self):
        """Create toolbar."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # Connect action
        self.connect_action = QAction("Connect", self)
        self.connect_action.setCheckable(True)
        self.connect_action.triggered.connect(self._on_connect_toggle)
        toolbar.addAction(self.connect_action)

        toolbar.addSeparator()

        # Clear data action
        clear_action = QAction("Clear Data", self)
        clear_action.triggered.connect(self._on_clear_data)
        toolbar.addAction(clear_action)

        toolbar.addSeparator()

        # About action
        about_action = QAction("About", self)
        about_action.triggered.connect(self._show_about)
        toolbar.addAction(about_action)

    def _connect_signals(self):
        """Connect widget signals."""
        # Map point selection -> update time series tracking
        self.tec_map_widget.point_selected.connect(self._on_point_selected)

        # Data manager signals
        self.data_manager.glotec_map_updated.connect(self._on_glotec_updated)
        self.data_manager.tec_statistics_updated.connect(
            self.status_bar.update_statistics
        )
        self.data_manager.tec_statistics_updated.connect(
            self.timeseries_widget.on_statistics_update
        )
        self.data_manager.connection_status_changed.connect(
            self.status_bar.set_connection_status
        )

    def set_websocket_client(self, ws_client):
        """
        Set the WebSocket client for data reception.

        Args:
            ws_client: DashboardWebSocketClient instance
        """
        self.ws_client = ws_client

        # Connect WebSocket signals to data manager
        ws_client.glotec_received.connect(self.data_manager.update_glotec_map)
        ws_client.connected.connect(
            lambda: self.data_manager.set_connection_status('websocket', True)
        )
        ws_client.disconnected.connect(
            lambda: self.data_manager.set_connection_status('websocket', False)
        )

        # Update connection action state
        ws_client.connected.connect(
            lambda: self.connect_action.setChecked(True)
        )
        ws_client.disconnected.connect(
            lambda: self.connect_action.setChecked(False)
        )

    @pyqtSlot(float, float)
    def _on_point_selected(self, lat: float, lon: float):
        """Handle point selection on map."""
        self.data_manager.set_tracked_point(lat, lon)
        self.timeseries_widget.set_tracked_point(lat, lon)

    @pyqtSlot(dict)
    def _on_glotec_updated(self, map_data: dict):
        """Handle GloTEC map update."""
        # Update map widget
        grid_arrays = self.data_manager.get_latest_grid_arrays()
        if grid_arrays:
            self.tec_map_widget.update_map(grid_arrays)

        # Update ionosphere profile
        self.profile_widget.on_glotec_update(map_data)

        # Update time series with point data if tracking
        if self.data_manager.tracked_point is not None:
            point_times, point_values = self.data_manager.get_point_history()
            global_times, global_values = self.data_manager.get_global_mean_history()

            self.timeseries_widget.update_from_history(
                global_times, global_values,
                point_times, point_values
            )

    def _on_refresh(self):
        """Periodic refresh handler."""
        # Currently handled by signals, but could add additional refresh logic
        pass

    def _on_connect_toggle(self, checked: bool):
        """Handle connect/disconnect toggle."""
        if self.ws_client is None:
            QMessageBox.warning(
                self,
                "No WebSocket Client",
                "WebSocket client not configured. Start the application with proper parameters."
            )
            self.connect_action.setChecked(False)
            return

        if checked:
            if not self.ws_client.isRunning():
                self.ws_client.start()
        else:
            self.ws_client.stop()

    def _on_clear_data(self):
        """Clear all data from widgets."""
        self.timeseries_widget.clear()
        self.profile_widget.clear()

    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About AutoNVIS TEC Display",
            "AutoNVIS TEC Display\n\n"
            "Real-time visualization of GNSS-TEC and GloTEC data.\n\n"
            "Data source: NOAA SWPC GloTEC\n"
            "Update cadence: 10 minutes\n\n"
            "Built with PyQt6 and pyqtgraph."
        )

    def closeEvent(self, event):
        """Handle window close."""
        if self.ws_client and self.ws_client.isRunning():
            self.ws_client.stop()
        event.accept()
