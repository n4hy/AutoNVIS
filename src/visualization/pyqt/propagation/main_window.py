"""
HF Propagation Conditions Main Window

Combines all four propagation indicators:
- X-ray flux (R-scale) - Radio blackouts
- Kp index (G-scale) - Geomagnetic storms
- Proton flux (S-scale) - Solar radiation storms
- Solar wind Bz - Storm precursor
"""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QGridLayout, QVBoxLayout, QHBoxLayout,
    QLabel, QFrame, QToolBar, QMessageBox, QComboBox, QCheckBox,
    QPushButton, QSizePolicy
)
from PyQt6.QtCore import Qt, QTimer, pyqtSlot
from PyQt6.QtGui import QAction
import pyqtgraph as pg
import logging

from .widgets import XRayWidget, KpWidget, ProtonWidget, SolarWindWidget


class ConditionsSummary(QWidget):
    """Overall HF conditions summary panel."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self.r_scale = 0
        self.g_scale = 0
        self.s_scale = 0
        self.bz = 0

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)

        # Title
        title = QLabel("HF CONDITIONS:")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)

        # Overall status
        self.status_label = QLabel("GOOD")
        self.status_label.setStyleSheet("""
            font-weight: bold;
            font-size: 16px;
            padding: 5px 15px;
            border-radius: 5px;
            background-color: #44aa44;
            color: white;
        """)
        layout.addWidget(self.status_label)

        layout.addStretch()

        # Individual scales
        self.r_label = QLabel("R0")
        self.g_label = QLabel("G0")
        self.s_label = QLabel("S0")

        for label in [self.r_label, self.g_label, self.s_label]:
            label.setStyleSheet("""
                font-weight: bold;
                font-size: 14px;
                padding: 3px 8px;
                border-radius: 4px;
                background-color: #44aa44;
                color: white;
                margin: 0 3px;
            """)
            layout.addWidget(label)

        # Last update
        layout.addWidget(QLabel(" | "))
        self.update_label = QLabel("Last update: --")
        self.update_label.setStyleSheet("color: #888;")
        layout.addWidget(self.update_label)

    def update_scales(self, r: int = None, g: int = None, s: int = None, bz: float = None):
        """Update scale values and overall status."""
        if r is not None:
            self.r_scale = r
        if g is not None:
            self.g_scale = g
        if s is not None:
            self.s_scale = s
        if bz is not None:
            self.bz = bz

        # Update individual labels
        self._update_scale_label(self.r_label, "R", self.r_scale)
        self._update_scale_label(self.g_label, "G", self.g_scale)
        self._update_scale_label(self.s_label, "S", self.s_scale)

        # Calculate overall status
        max_scale = max(self.r_scale, self.g_scale, self.s_scale)
        bz_bad = self.bz < -10

        if max_scale >= 4 or (max_scale >= 2 and bz_bad):
            status = "POOR"
            color = "#dd4444"
        elif max_scale >= 2 or bz_bad:
            status = "FAIR"
            color = "#ddaa44"
        elif max_scale >= 1:
            status = "MODERATE"
            color = "#aaaa44"
        else:
            status = "GOOD"
            color = "#44aa44"

        self.status_label.setText(status)
        self.status_label.setStyleSheet(f"""
            font-weight: bold;
            font-size: 16px;
            padding: 5px 15px;
            border-radius: 5px;
            background-color: {color};
            color: white;
        """)

        # Update timestamp
        from datetime import datetime
        self.update_label.setText(f"Last update: {datetime.utcnow().strftime('%H:%M:%S')} UTC")

    def _update_scale_label(self, label: QLabel, prefix: str, value: int):
        colors = {
            0: '#44aa44', 1: '#aaaa44', 2: '#ddaa44',
            3: '#dd6644', 4: '#dd4444', 5: '#aa44aa'
        }
        color = colors.get(value, '#888888')
        label.setText(f"{prefix}{value}")
        label.setStyleSheet(f"""
            font-weight: bold;
            font-size: 14px;
            padding: 3px 8px;
            border-radius: 4px;
            background-color: {color};
            color: white;
            margin: 0 3px;
        """)


class PropagationMainWindow(QMainWindow):
    """
    Main window for HF Propagation Conditions display.

    Layout:
    +----------------------------------------------------------+
    | Toolbar | HF CONDITIONS: [GOOD] | R0 G0 S0 | Last: HH:MM |
    +----------------------------------------------------------+
    | +------------------------+ +---------------------------+ |
    | |                        | |                           | |
    | |   X-Ray Flux (R)       | |   Kp Index (G)            | |
    | |   Flare/Blackout       | |   Geomagnetic Storm       | |
    | |                        | |                           | |
    | +------------------------+ +---------------------------+ |
    | +------------------------+ +---------------------------+ |
    | |                        | |                           | |
    | |   Proton Flux (S)      | |   Solar Wind Bz           | |
    | |   Radiation Storm      | |   Storm Precursor         | |
    | |                        | |                           | |
    | +------------------------+ +---------------------------+ |
    +----------------------------------------------------------+
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.logger = logging.getLogger("propagation")
        self.setWindowTitle("AutoNVIS HF Propagation Conditions")
        self.setMinimumSize(1200, 800)

        pg.setConfigOptions(antialias=True)

        self.data_client = None

        self._setup_ui()
        self._setup_toolbar()
        self._apply_dark_theme()

    def _setup_ui(self):
        """Create and arrange all UI elements."""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Summary bar
        self.summary = ConditionsSummary()
        self.summary.setStyleSheet("background-color: #2a2a2a; border-radius: 5px;")
        layout.addWidget(self.summary)

        # 2x2 grid of widgets
        grid = QGridLayout()
        grid.setSpacing(8)

        self.xray_widget = XRayWidget()
        self.kp_widget = KpWidget()
        self.proton_widget = ProtonWidget()
        self.solarwind_widget = SolarWindWidget()

        grid.addWidget(self._wrap_widget(self.xray_widget, "Radio Blackouts"), 0, 0)
        grid.addWidget(self._wrap_widget(self.kp_widget, "Geomagnetic Storms"), 0, 1)
        grid.addWidget(self._wrap_widget(self.proton_widget, "Solar Radiation"), 1, 0)
        grid.addWidget(self._wrap_widget(self.solarwind_widget, "Storm Precursor"), 1, 1)

        layout.addLayout(grid)

        # Status bar
        self.statusBar().showMessage("Disconnected | Waiting for data...")
        self.statusBar().setStyleSheet("""
            QStatusBar {
                background-color: #2d2d2d;
                color: #aaa;
            }
        """)

    def _wrap_widget(self, widget: QWidget, subtitle: str) -> QFrame:
        """Wrap a widget in a styled frame."""
        frame = QFrame()
        frame.setFrameStyle(QFrame.Shape.Box)
        frame.setStyleSheet("""
            QFrame {
                background-color: #252525;
                border: 1px solid #3a3a3a;
                border-radius: 5px;
            }
        """)

        layout = QVBoxLayout(frame)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.addWidget(widget)

        return frame

    def _setup_toolbar(self):
        """Create toolbar with controls."""
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
            QComboBox {
                background-color: #3d3d3d;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 5px 10px;
                color: #ddd;
                min-width: 80px;
            }
            QComboBox:hover {
                background-color: #4d4d4d;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #3d3d3d;
                color: #ddd;
                selection-background-color: #2a5a2a;
            }
        """)
        self.addToolBar(toolbar)

        # Refresh button
        self.refresh_btn = QPushButton("Refresh Now")
        self.refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #2a6e2a;
                border: 1px solid #4a8e4a;
                border-radius: 4px;
                padding: 5px 15px;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3a8e3a;
            }
            QPushButton:pressed {
                background-color: #1a5e1a;
            }
        """)
        self.refresh_btn.clicked.connect(self._on_refresh)
        toolbar.addWidget(self.refresh_btn)

        toolbar.addSeparator()

        # Update interval
        interval_label = QLabel("Update:")
        interval_label.setStyleSheet("color: #ddd; padding: 0 5px;")
        toolbar.addWidget(interval_label)

        self.interval_combo = QComboBox()
        self.interval_combo.addItems(["30 sec", "1 min", "2 min", "5 min"])
        self.interval_combo.setCurrentIndex(1)  # Default 1 min
        self.interval_combo.currentIndexChanged.connect(self._on_interval_changed)
        toolbar.addWidget(self.interval_combo)

        toolbar.addSeparator()

        # Time range
        range_label = QLabel("History:")
        range_label.setStyleSheet("color: #ddd; padding: 0 5px;")
        toolbar.addWidget(range_label)

        self.range_combo = QComboBox()
        self.range_combo.addItems(["1 hour", "6 hours", "12 hours", "24 hours"])
        self.range_combo.setCurrentIndex(3)  # Default 24h
        self.range_combo.currentIndexChanged.connect(self._on_range_changed)
        toolbar.addWidget(self.range_combo)

        toolbar.addSeparator()

        # Autoscale toggle
        self.autoscale_action = QAction("Autoscale Y", self)
        self.autoscale_action.setCheckable(True)
        self.autoscale_action.setChecked(False)
        self.autoscale_action.setToolTip("Toggle Y-axis autoscaling")
        self.autoscale_action.triggered.connect(self._on_autoscale_toggle)
        toolbar.addAction(self.autoscale_action)

        toolbar.addSeparator()

        # Clear data
        clear_action = QAction("Clear Data", self)
        clear_action.triggered.connect(self._on_clear_data)
        toolbar.addAction(clear_action)

        # Spacer to push About to the right
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)

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
        """)

    def set_data_client(self, client):
        """Set the data client and connect signals."""
        self.data_client = client

        client.xray_received.connect(self._on_xray)
        client.kp_received.connect(self._on_kp)
        client.proton_received.connect(self._on_proton)
        client.solarwind_received.connect(self._on_solarwind)

        client.connected.connect(self._on_connected)
        client.disconnected.connect(self._on_disconnected)

    @pyqtSlot(dict)
    def _on_xray(self, data: dict):
        self.xray_widget.on_data(data)
        self.summary.update_scales(r=data.get('r_scale', 0))

    @pyqtSlot(dict)
    def _on_kp(self, data: dict):
        self.kp_widget.on_data(data)
        self.summary.update_scales(g=data.get('g_scale', 0))

    @pyqtSlot(dict)
    def _on_proton(self, data: dict):
        self.proton_widget.on_data(data)
        self.summary.update_scales(s=data.get('s_scale', 0))

    @pyqtSlot(dict)
    def _on_solarwind(self, data: dict):
        self.solarwind_widget.on_data(data)
        self.summary.update_scales(bz=data.get('bz_gsm', 0))

    @pyqtSlot()
    def _on_connected(self):
        self.statusBar().showMessage("Connected | Receiving data...")

    @pyqtSlot()
    def _on_disconnected(self):
        self.statusBar().showMessage("Disconnected")

    def _on_refresh(self):
        """Handle manual refresh request."""
        if self.data_client:
            self.statusBar().showMessage("Refreshing data...")
            # Trigger immediate fetch by restarting the client's timer
            if hasattr(self.data_client, 'worker') and self.data_client.worker:
                self.data_client.worker._fetch_data()
            self.statusBar().showMessage("Connected | Data refreshed")

    def _on_interval_changed(self, index: int):
        """Handle update interval change."""
        intervals = [30000, 60000, 120000, 300000]  # ms
        interval_ms = intervals[index]

        if self.data_client:
            self.data_client.update_interval_ms = interval_ms
            if hasattr(self.data_client, 'worker') and self.data_client.worker:
                if self.data_client.worker.timer:
                    self.data_client.worker.timer.setInterval(interval_ms)

        interval_text = self.interval_combo.currentText()
        self.statusBar().showMessage(f"Update interval set to {interval_text}")

    def _on_range_changed(self, index: int):
        """Handle time range change."""
        ranges_hours = [1, 6, 12, 24]
        hours = ranges_hours[index]

        # Update all widgets with new time range
        self.xray_widget.set_time_range(hours)
        self.kp_widget.set_time_range(hours)
        self.proton_widget.set_time_range(hours)
        self.solarwind_widget.set_time_range(hours)

        range_text = self.range_combo.currentText()
        self.statusBar().showMessage(f"Time range set to {range_text}")

    def _on_autoscale_toggle(self, checked: bool):
        """Handle autoscale toggle."""
        self.xray_widget.set_autoscale(checked)
        self.kp_widget.set_autoscale(checked)
        self.proton_widget.set_autoscale(checked)
        self.solarwind_widget.set_autoscale(checked)

        mode = "enabled" if checked else "disabled"
        self.statusBar().showMessage(f"Y-axis autoscale {mode}")

    def _on_clear_data(self):
        """Clear all data from widgets."""
        self.xray_widget.clear_data()
        self.kp_widget.clear_data()
        self.proton_widget.clear_data()
        self.solarwind_widget.clear_data()
        self.summary.update_scales(r=0, g=0, s=0, bz=0)
        self.statusBar().showMessage("Data cleared")

    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About HF Propagation Conditions",
            "AutoNVIS HF Propagation Conditions Monitor\n\n"
            "Real-time space weather monitoring for HF radio operators.\n\n"
            "Data sources:\n"
            "  - X-ray flux: GOES satellite\n"
            "  - Kp index: NOAA SWPC\n"
            "  - Proton flux: GOES satellite\n"
            "  - Solar wind: DSCOVR satellite\n\n"
            "NOAA Scales:\n"
            "  R (Radio Blackouts): Based on X-ray flux\n"
            "  G (Geomagnetic Storms): Based on Kp index\n"
            "  S (Solar Radiation): Based on proton flux\n\n"
            "Controls:\n"
            "  - Refresh Now: Force immediate data fetch\n"
            "  - Update: Set polling interval (30s-5min)\n"
            "  - History: Set time range to display\n"
            "  - Autoscale Y: Toggle Y-axis auto-ranging\n\n"
            "Built with PyQt6 and pyqtgraph."
        )

    def closeEvent(self, event):
        """Handle window close."""
        try:
            if self.data_client and self.data_client.isRunning():
                self.data_client.stop()
        except Exception as e:
            self.logger.error(f"Error stopping client: {e}")
        event.accept()
