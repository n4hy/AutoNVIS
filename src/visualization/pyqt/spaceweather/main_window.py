"""
Space Weather Display Main Window

Main window for real-time solar X-ray flux visualization.
"""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QToolBar, QMessageBox, QSpinBox, QLabel
)
from PyQt6.QtCore import Qt, QTimer, pyqtSlot
from PyQt6.QtGui import QAction
import pyqtgraph as pg
import logging

from .flare_indicator_widget import FlareIndicatorWidget
from .xray_plot_widget import XRayPlotWidget


class SpaceWeatherDataManager:
    """
    Simple data manager for space weather display.

    Tracks X-ray flux history and connection status.
    """

    def __init__(self):
        self.connected = False
        self.last_xray_data = None
        self.data_count = 0

    def update_xray(self, data: dict):
        """Update with new X-ray data."""
        self.last_xray_data = data
        self.data_count += 1

    def set_connected(self, connected: bool):
        """Set connection status."""
        self.connected = connected


class SpaceWeatherMainWindow(QMainWindow):
    """
    Main window for Space Weather visualization.

    Layout:
    +----------------------------------------------------------+
    | Toolbar: [Connect] [Clear] [About]                       |
    +----------------------------------------------------------+
    | +------------------------+ +---------------------------+ |
    | |                        | |                           | |
    | |   Flare Indicator      | |   X-Ray Time Series       | |
    | |   - Class (A/B/C/M/X)  | |   - 24-hour history       | |
    | |   - QUIET/SHOCK mode   | |   - Class threshold lines | |
    | |   - Current flux       | |   - Logarithmic scale     | |
    | |                        | |                           | |
    | +------------------------+ +---------------------------+ |
    +----------------------------------------------------------+
    | Status: Connected | Flare: B5.6 | Mode: QUIET            |
    +----------------------------------------------------------+
    """

    def __init__(self, parent=None):
        """Initialize main window."""
        super().__init__(parent)

        self.logger = logging.getLogger("spaceweather")

        self.setWindowTitle("AutoNVIS Space Weather - GOES X-Ray Monitor")
        self.setMinimumSize(1000, 600)

        # Initialize pyqtgraph with robust settings
        pg.setConfigOptions(antialias=True, useOpenGL=False)

        # Data manager
        self.data_manager = SpaceWeatherDataManager()

        # WebSocket client (set externally)
        self.ws_client = None

        # Deferred resize handling
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self._handle_resize)

        # Create UI
        self._setup_ui()
        self._setup_toolbar()

        # Apply dark theme
        self._apply_dark_theme()

    def _setup_ui(self):
        """Create and arrange all UI elements."""
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Main horizontal splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel: Flare indicator (compact)
        self.flare_indicator = FlareIndicatorWidget()
        self.flare_indicator.setMaximumWidth(350)
        self.flare_indicator.setMinimumWidth(250)
        main_splitter.addWidget(self.flare_indicator)

        # Right panel: X-ray time series (larger)
        self.xray_plot = XRayPlotWidget()
        main_splitter.addWidget(self.xray_plot)

        # Set splitter proportions
        main_splitter.setSizes([300, 700])

        layout.addWidget(main_splitter)

        # Status bar
        self.statusBar().showMessage("Disconnected | Waiting for data...")
        self.statusBar().setStyleSheet("""
            QStatusBar {
                background-color: #2d2d2d;
                color: #aaa;
                font-size: 12px;
            }
        """)

    def _setup_toolbar(self):
        """Create toolbar."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        toolbar.setStyleSheet("""
            QToolBar {
                background-color: #2d2d2d;
                border: none;
                spacing: 5px;
                padding: 5px;
            }
            QToolButton {
                background-color: #3d3d3d;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 5px 10px;
                color: #ddd;
            }
            QToolButton:hover {
                background-color: #4d4d4d;
            }
            QToolButton:checked {
                background-color: #2a5a2a;
                border-color: #4a8a4a;
            }
        """)
        self.addToolBar(toolbar)

        # Connect action
        self.connect_action = QAction("Connect", self)
        self.connect_action.setCheckable(True)
        self.connect_action.triggered.connect(self._on_connect_toggle)
        toolbar.addAction(self.connect_action)

        toolbar.addSeparator()

        # Autoscale toggle
        self.autoscale_action = QAction("Autoscale Y", self)
        self.autoscale_action.setCheckable(True)
        self.autoscale_action.setChecked(False)
        self.autoscale_action.setToolTip("Toggle Y-axis autoscaling (vs fixed 1e-9 to 1e-3 range)")
        self.autoscale_action.triggered.connect(self._on_autoscale_toggle)
        toolbar.addAction(self.autoscale_action)

        # Normalized view toggle
        self.normalized_action = QAction("Normalized %", self)
        self.normalized_action.setCheckable(True)
        self.normalized_action.setChecked(False)
        self.normalized_action.setToolTip("Show % deviation from mean - reveals small variations")
        self.normalized_action.triggered.connect(self._on_normalized_toggle)
        toolbar.addAction(self.normalized_action)

        toolbar.addSeparator()

        # Duration controls
        duration_label = QLabel("Duration:")
        duration_label.setStyleSheet("color: #ddd; padding: 0 5px;")
        toolbar.addWidget(duration_label)

        self.days_spinbox = QSpinBox()
        self.days_spinbox.setRange(0, 100)
        self.days_spinbox.setValue(1)
        self.days_spinbox.setSuffix(" days")
        self.days_spinbox.setToolTip("Days of data to display (0-100)")
        self.days_spinbox.setStyleSheet("""
            QSpinBox {
                background-color: #3d3d3d;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 3px 8px;
                color: #ddd;
                min-width: 70px;
            }
        """)
        self.days_spinbox.valueChanged.connect(self._on_duration_changed)
        toolbar.addWidget(self.days_spinbox)

        self.hours_spinbox = QSpinBox()
        self.hours_spinbox.setRange(0, 23)
        self.hours_spinbox.setValue(0)
        self.hours_spinbox.setSuffix(" hrs")
        self.hours_spinbox.setToolTip("Hours of data to display (0-23)")
        self.hours_spinbox.setStyleSheet("""
            QSpinBox {
                background-color: #3d3d3d;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 3px 8px;
                color: #ddd;
                min-width: 60px;
            }
        """)
        self.hours_spinbox.valueChanged.connect(self._on_duration_changed)
        toolbar.addWidget(self.hours_spinbox)

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

    def _apply_dark_theme(self):
        """Apply dark theme styling."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QWidget {
                background-color: #1e1e1e;
                color: #ddd;
            }
            QSplitter::handle {
                background-color: #444;
            }
            QSplitter::handle:horizontal {
                width: 4px;
            }
        """)

    def set_websocket_client(self, ws_client):
        """
        Set the WebSocket client for data reception.

        Args:
            ws_client: DashboardWebSocketClient instance
        """
        self.ws_client = ws_client

        # Connect X-ray signals
        ws_client.xray_received.connect(self._on_xray_update)
        ws_client.xray_batch_received.connect(self._on_xray_batch)

        # Connect status signals
        ws_client.connected.connect(self._on_connected)
        ws_client.disconnected.connect(self._on_disconnected)

        # Update connect button state
        ws_client.connected.connect(
            lambda: self.connect_action.setChecked(True)
        )
        ws_client.disconnected.connect(
            lambda: self.connect_action.setChecked(False)
        )

    @pyqtSlot(dict)
    def _on_xray_update(self, data: dict):
        """Handle X-ray data update."""
        self.data_manager.update_xray(data)

        # Update flare indicator
        self.flare_indicator.on_xray_update(data)

        # Update X-ray plot
        self.xray_plot.on_xray_update(data)

        # Update status bar
        flare_class = data.get('flare_class', '--')
        m1_threshold = data.get('m1_or_higher', False)
        mode = "SHOCK" if m1_threshold else "QUIET"

        self.statusBar().showMessage(
            f"Connected | Flare: {flare_class} | Mode: {mode} | "
            f"Updates: {self.data_manager.data_count}"
        )

    @pyqtSlot(dict)
    def _on_xray_batch(self, data: dict):
        """Handle historical X-ray batch - load all at once."""
        count = data.get('count', 0)
        self.statusBar().showMessage(f"Loading {count} historical records...")

        # Load batch into plot (single update)
        self.xray_plot.on_xray_batch(data)

        # Update flare indicator with most recent record
        records = data.get('records', [])
        if records:
            latest = records[-1]
            self.flare_indicator.on_xray_update(latest)
            self.data_manager.update_xray(latest)

        self.statusBar().showMessage(
            f"Connected | Historical data loaded ({count} records) | Live updates active"
        )

    @pyqtSlot()
    def _on_connected(self):
        """Handle WebSocket connection."""
        self.data_manager.set_connected(True)
        self.statusBar().showMessage("Connected | Waiting for X-ray data...")

    @pyqtSlot()
    def _on_disconnected(self):
        """Handle WebSocket disconnection."""
        self.data_manager.set_connected(False)
        self.statusBar().showMessage("Disconnected | Connection lost")

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

    def _on_autoscale_toggle(self, checked: bool):
        """Handle autoscale toggle."""
        self.xray_plot.set_autoscale(checked)
        mode = "enabled" if checked else "disabled (fixed 1e-9 to 1e-3)"
        self.statusBar().showMessage(f"Autoscale {mode}")

    def _on_normalized_toggle(self, checked: bool):
        """Handle normalized view toggle."""
        self.xray_plot.set_normalized_view(checked)
        if checked:
            self.statusBar().showMessage("Normalized view: showing % deviation from mean")
        else:
            self.statusBar().showMessage("Absolute view: showing actual flux values")

    def _on_duration_changed(self):
        """Handle duration change."""
        days = self.days_spinbox.value()
        hours = self.hours_spinbox.value()

        # Ensure at least 1 hour
        if days == 0 and hours == 0:
            hours = 1
            self.hours_spinbox.setValue(1)

        # Clear and reload data with new duration
        self.xray_plot.clear()
        self.xray_plot.set_duration(days=days, hours=hours)

        # Trigger data reload from client
        if self.ws_client and self.ws_client.isRunning():
            self.ws_client.reload()

        if days > 0:
            self.statusBar().showMessage(f"Duration set to {days} days {hours} hours - reloading data...")
        else:
            self.statusBar().showMessage(f"Duration set to {hours} hours - reloading data...")

    def _on_clear_data(self):
        """Clear all data from widgets."""
        self.xray_plot.clear()
        self.data_manager.data_count = 0
        self.statusBar().showMessage("Data cleared")

    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About AutoNVIS Space Weather",
            "AutoNVIS Space Weather Monitor\n\n"
            "Real-time GOES X-ray flux visualization for\n"
            "solar flare monitoring and mode switching.\n\n"
            "Data source: NOAA SWPC GOES X-ray\n"
            "Update cadence: 1 minute\n\n"
            "Flare Classes:\n"
            "  A: < 1e-7 W/m² (Very Quiet)\n"
            "  B: 1e-7 to 1e-6 W/m² (Quiet)\n"
            "  C: 1e-6 to 1e-5 W/m² (Minor)\n"
            "  M: 1e-5 to 1e-4 W/m² (Moderate)\n"
            "  X: > 1e-4 W/m² (Major)\n\n"
            "QUIET mode: Below M1 threshold\n"
            "SHOCK mode: M1 or higher flare\n\n"
            "Built with PyQt6 and pyqtgraph."
        )

    def resizeEvent(self, event):
        """Handle window resize with deferred updates."""
        super().resizeEvent(event)
        # Debounce resize events
        self._resize_timer.start(150)

    def _handle_resize(self):
        """Deferred resize handling."""
        try:
            # Force plot widget to update after resize settles
            if hasattr(self, 'xray_plot') and self.xray_plot:
                self.xray_plot.update()
        except Exception as e:
            self.logger.debug(f"Resize handling error: {e}")

    def closeEvent(self, event):
        """Handle window close."""
        try:
            if self.ws_client and self.ws_client.isRunning():
                self.ws_client.stop()
        except Exception as e:
            self.logger.error(f"Error stopping client: {e}")
        event.accept()
