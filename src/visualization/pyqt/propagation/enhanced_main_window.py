"""
Enhanced Propagation Display - Complete Space Weather Dashboard

Combines the Standard Four (solar drivers) with Advanced data sources (ionospheric response)
to provide comprehensive HF propagation intelligence.

Layout:
- Top: Summary bar with overall conditions + key metrics
- Left Column: Standard Four (X-Ray, Kp, Proton, Solar Wind Bz)
- Right Column: Advanced Sources (F10.7, Ionosonde, HPI, D-RAP)
- Bottom: Prediction panel (WSA-Enlil, Propagated Wind)

This display provides the foundation for PHaRLAP ray tracing integration.
"""

import sys
import logging
from datetime import datetime
from typing import Optional

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QGridLayout, QToolBar, QComboBox, QPushButton, QSizePolicy,
    QStatusBar, QSplitter, QTabWidget
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QAction

# Standard Four components
from .data_client import PropagationDataClient
from .widgets import XRayWidget, KpWidget, ProtonWidget, SolarWindWidget

# Advanced components
from .advanced_data_client import AdvancedDataClient
from .advanced_widgets import (
    F107Widget, HPIWidget, DRAPWidget, IonosondeWidget,
    EnlilWidget, PropagatedWindWidget
)


class ConditionsSummaryBar(QFrame):
    """Enhanced summary bar showing all key metrics at a glance."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QFrame {
                background-color: #1a1a2e;
                border-radius: 6px;
            }
        """)
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 8, 16, 8)
        layout.setSpacing(24)

        # Overall HF Conditions
        self.conditions_label = QLabel("HF CONDITIONS:")
        self.conditions_label.setFont(QFont("Segoe UI", 11))
        self.conditions_label.setStyleSheet("color: #888;")
        layout.addWidget(self.conditions_label)

        self.conditions_value = QLabel("ANALYZING...")
        self.conditions_value.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        self.conditions_value.setStyleSheet("color: #4caf50;")
        layout.addWidget(self.conditions_value)

        layout.addStretch()

        # NOAA Scales
        scales_frame = QFrame()
        scales_layout = QHBoxLayout(scales_frame)
        scales_layout.setContentsMargins(0, 0, 0, 0)
        scales_layout.setSpacing(12)

        self.r_badge = self._create_badge("R0", "#4caf50")
        scales_layout.addWidget(self.r_badge)

        self.g_badge = self._create_badge("G0", "#4caf50")
        scales_layout.addWidget(self.g_badge)

        self.s_badge = self._create_badge("S0", "#4caf50")
        scales_layout.addWidget(self.s_badge)

        layout.addWidget(scales_frame)

        layout.addStretch()

        # Key metrics
        metrics_frame = QFrame()
        metrics_layout = QHBoxLayout(metrics_frame)
        metrics_layout.setContentsMargins(0, 0, 0, 0)
        metrics_layout.setSpacing(16)

        self.f107_metric = self._create_metric("F10.7", "--", "#2196f3")
        metrics_layout.addWidget(self.f107_metric)

        self.muf_metric = self._create_metric("Max MUF", "--", "#4caf50")
        metrics_layout.addWidget(self.muf_metric)

        self.hpi_metric = self._create_metric("HPI", "--", "#ab47bc")
        metrics_layout.addWidget(self.hpi_metric)

        self.haf_metric = self._create_metric("HAF", "--", "#ff9800")
        metrics_layout.addWidget(self.haf_metric)

        layout.addWidget(metrics_frame)

        layout.addStretch()

        # Timestamp
        self.time_label = QLabel("Last: --:--:-- UTC")
        self.time_label.setFont(QFont("Segoe UI", 9))
        self.time_label.setStyleSheet("color: #666;")
        layout.addWidget(self.time_label)

    def _create_badge(self, text: str, color: str) -> QLabel:
        badge = QLabel(text)
        badge.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        badge.setStyleSheet(f"""
            background-color: {color};
            color: #ffffff;
            padding: 4px 10px;
            border-radius: 4px;
        """)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setMinimumWidth(48)
        return badge

    def _create_metric(self, label: str, value: str, color: str) -> QFrame:
        frame = QFrame()
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        label_widget = QLabel(label)
        label_widget.setFont(QFont("Segoe UI", 8))
        label_widget.setStyleSheet("color: #666;")
        label_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label_widget)

        value_widget = QLabel(value)
        value_widget.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        value_widget.setStyleSheet(f"color: {color};")
        value_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
        value_widget.setObjectName(f"{label.replace(' ', '_').lower()}_value")
        layout.addWidget(value_widget)

        return frame

    def update_scales(self, r: int, g: int, s: int):
        """Update NOAA scale badges."""
        self._update_badge(self.r_badge, f"R{r}", r)
        self._update_badge(self.g_badge, f"G{g}", g)
        self._update_badge(self.s_badge, f"S{s}", s)

    def _update_badge(self, badge: QLabel, text: str, level: int):
        badge.setText(text)
        if level >= 4:
            color = "#f44336"  # Red
        elif level >= 3:
            color = "#ff9800"  # Orange
        elif level >= 2:
            color = "#ffeb3b"  # Yellow
            badge.setStyleSheet(f"background-color: {color}; color: #000; padding: 4px 10px; border-radius: 4px;")
            return
        elif level >= 1:
            color = "#8bc34a"  # Light green
        else:
            color = "#4caf50"  # Green
        badge.setStyleSheet(f"background-color: {color}; color: #ffffff; padding: 4px 10px; border-radius: 4px;")

    def update_conditions(self, status: str):
        """Update overall conditions display."""
        self.conditions_value.setText(status)
        if status == "GOOD":
            color = "#4caf50"
        elif status == "MODERATE":
            color = "#8bc34a"
        elif status == "FAIR":
            color = "#ffeb3b"
        elif status == "POOR":
            color = "#f44336"
        else:
            color = "#888888"
        self.conditions_value.setStyleSheet(f"color: {color};")

    def update_metric(self, name: str, value: str, color: str = None):
        """Update a metric value."""
        obj_name = f"{name.replace(' ', '_').lower()}_value"
        for child in self.findChildren(QLabel):
            if child.objectName() == obj_name:
                child.setText(value)
                if color:
                    child.setStyleSheet(f"color: {color};")
                break

    def update_time(self, timestamp: str):
        """Update the timestamp display."""
        self.time_label.setText(f"Last: {timestamp}")


class EnhancedPropagationWindow(QMainWindow):
    """Enhanced propagation display combining Standard Four + Advanced sources."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Enhanced HF Propagation Display - AutoNVIS")
        self.setMinimumSize(1400, 900)

        # Dark theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0d0d1a;
            }
            QWidget {
                color: #e0e0e0;
            }
            QToolBar {
                background-color: #1a1a2e;
                border: none;
                spacing: 8px;
                padding: 4px;
            }
            QToolBar QLabel {
                color: #888;
                padding: 0 4px;
            }
            QComboBox {
                background-color: #2d2d44;
                border: 1px solid #3d3d5c;
                border-radius: 4px;
                padding: 4px 8px;
                min-width: 80px;
            }
            QComboBox:hover {
                border-color: #4fc3f7;
            }
            QComboBox::drop-down {
                border: none;
            }
            QPushButton {
                background-color: #2d2d44;
                border: 1px solid #3d3d5c;
                border-radius: 4px;
                padding: 6px 12px;
            }
            QPushButton:hover {
                background-color: #3d3d5c;
                border-color: #4fc3f7;
            }
            QPushButton:pressed {
                background-color: #4fc3f7;
            }
            QStatusBar {
                background-color: #1a1a2e;
                color: #888;
            }
            QTabWidget::pane {
                border: 1px solid #2d2d44;
                background-color: #0d0d1a;
            }
            QTabBar::tab {
                background-color: #1a1a2e;
                color: #888;
                padding: 8px 16px;
                border: 1px solid #2d2d44;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #2d2d44;
                color: #4fc3f7;
            }
            QTabBar::tab:hover {
                color: #e0e0e0;
            }
        """)

        self.logger = logging.getLogger("enhanced_propagation")

        # Initialize data clients
        self.standard_client = PropagationDataClient(update_interval_ms=60000)
        self.advanced_client = AdvancedDataClient(update_interval_ms=60000)

        # Track current scale values
        self.current_r = 0
        self.current_g = 0
        self.current_s = 0
        self.current_bz = 0

        self._setup_ui()
        self._connect_signals()
        self._start_clients()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # Toolbar
        self._create_toolbar()

        # Summary bar
        self.summary_bar = ConditionsSummaryBar()
        main_layout.addWidget(self.summary_bar)

        # Main content area with tabs
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs, stretch=1)

        # Tab 1: Combined Dashboard
        dashboard = self._create_dashboard()
        self.tabs.addTab(dashboard, "Dashboard")

        # Tab 2: Standard Four (detailed)
        standard_tab = self._create_standard_four_tab()
        self.tabs.addTab(standard_tab, "Standard Four")

        # Tab 3: Advanced Sources (detailed)
        advanced_tab = self._create_advanced_tab()
        self.tabs.addTab(advanced_tab, "Advanced Ionospheric")

        # Tab 4: Predictions
        predictions_tab = self._create_predictions_tab()
        self.tabs.addTab(predictions_tab, "Predictions")

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Initializing data streams...")

    def _create_toolbar(self):
        toolbar = QToolBar("Controls")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # Refresh button
        refresh_btn = QPushButton("⟳ Refresh Now")
        refresh_btn.clicked.connect(self._on_refresh)
        toolbar.addWidget(refresh_btn)

        toolbar.addSeparator()

        # Update interval
        toolbar.addWidget(QLabel("Update:"))
        self.interval_combo = QComboBox()
        self.interval_combo.addItems(["30 sec", "1 min", "2 min", "5 min"])
        self.interval_combo.setCurrentIndex(1)
        self.interval_combo.currentIndexChanged.connect(self._on_interval_changed)
        toolbar.addWidget(self.interval_combo)

        toolbar.addSeparator()

        # History range
        toolbar.addWidget(QLabel("History:"))
        self.range_combo = QComboBox()
        self.range_combo.addItems(["1 hour", "6 hours", "12 hours", "24 hours"])
        self.range_combo.setCurrentIndex(3)
        self.range_combo.currentIndexChanged.connect(self._on_range_changed)
        toolbar.addWidget(self.range_combo)

        toolbar.addSeparator()

        # Autoscale toggle
        self.autoscale_btn = QPushButton("Autoscale Y")
        self.autoscale_btn.setCheckable(True)
        self.autoscale_btn.toggled.connect(self._on_autoscale_toggle)
        toolbar.addWidget(self.autoscale_btn)

        toolbar.addSeparator()

        # Clear data
        clear_btn = QPushButton("Clear Data")
        clear_btn.clicked.connect(self._on_clear_data)
        toolbar.addWidget(clear_btn)

    def _create_dashboard(self) -> QWidget:
        """Create the combined dashboard view."""
        widget = QWidget()
        layout = QGridLayout(widget)
        layout.setSpacing(8)

        # Standard Four (left side)
        self.xray_widget = XRayWidget()
        layout.addWidget(self.xray_widget, 0, 0)

        self.kp_widget = KpWidget()
        layout.addWidget(self.kp_widget, 0, 1)

        self.proton_widget = ProtonWidget()
        layout.addWidget(self.proton_widget, 1, 0)

        self.solarwind_widget = SolarWindWidget()
        layout.addWidget(self.solarwind_widget, 1, 1)

        # Advanced (right side)
        self.f107_widget = F107Widget()
        layout.addWidget(self.f107_widget, 0, 2)

        self.ionosonde_widget = IonosondeWidget()
        layout.addWidget(self.ionosonde_widget, 0, 3)

        self.hpi_widget = HPIWidget()
        layout.addWidget(self.hpi_widget, 1, 2)

        self.drap_widget = DRAPWidget()
        layout.addWidget(self.drap_widget, 1, 3)

        return widget

    def _create_standard_four_tab(self) -> QWidget:
        """Create detailed Standard Four tab."""
        widget = QWidget()
        layout = QGridLayout(widget)
        layout.setSpacing(8)

        # Create dedicated widgets for this tab
        self.xray_detail = XRayWidget()
        layout.addWidget(self.xray_detail, 0, 0)

        self.kp_detail = KpWidget()
        layout.addWidget(self.kp_detail, 0, 1)

        self.proton_detail = ProtonWidget()
        layout.addWidget(self.proton_detail, 1, 0)

        self.solarwind_detail = SolarWindWidget()
        layout.addWidget(self.solarwind_detail, 1, 1)

        return widget

    def _create_advanced_tab(self) -> QWidget:
        """Create detailed Advanced Ionospheric tab."""
        widget = QWidget()
        layout = QGridLayout(widget)
        layout.setSpacing(8)

        self.f107_detail = F107Widget()
        layout.addWidget(self.f107_detail, 0, 0)

        self.ionosonde_detail = IonosondeWidget()
        layout.addWidget(self.ionosonde_detail, 0, 1)

        self.hpi_detail = HPIWidget()
        layout.addWidget(self.hpi_detail, 1, 0)

        self.drap_detail = DRAPWidget()
        layout.addWidget(self.drap_detail, 1, 1)

        return widget

    def _create_predictions_tab(self) -> QWidget:
        """Create predictions tab."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setSpacing(8)

        self.enlil_widget = EnlilWidget()
        layout.addWidget(self.enlil_widget)

        self.prop_wind_widget = PropagatedWindWidget()
        layout.addWidget(self.prop_wind_widget)

        return widget

    def _connect_signals(self):
        """Connect data client signals to widget updates."""
        # Standard Four
        self.standard_client.xray_received.connect(self._on_xray)
        self.standard_client.kp_received.connect(self._on_kp)
        self.standard_client.proton_received.connect(self._on_proton)
        self.standard_client.solarwind_received.connect(self._on_solarwind)
        self.standard_client.connected.connect(lambda: self.status_bar.showMessage("Standard Four: Connected"))
        self.standard_client.error.connect(lambda e: self.status_bar.showMessage(f"Standard Error: {e}"))

        # Advanced
        self.advanced_client.f107_received.connect(self._on_f107)
        self.advanced_client.hpi_received.connect(self._on_hpi)
        self.advanced_client.drap_received.connect(self._on_drap)
        self.advanced_client.ionosonde_received.connect(self._on_ionosonde)
        self.advanced_client.enlil_received.connect(self._on_enlil)
        self.advanced_client.prop_wind_received.connect(self._on_prop_wind)
        self.advanced_client.connected.connect(lambda: self.status_bar.showMessage("Advanced: Connected"))

    def _start_clients(self):
        """Start both data clients."""
        self.standard_client.start()
        self.advanced_client.start()

    # =========================================================================
    # Standard Four Handlers
    # =========================================================================
    def _on_xray(self, data: dict):
        self.xray_widget.on_data(data)
        self.xray_detail.on_data(data)
        self.current_r = data.get('r_scale', 0)
        self._update_summary()

    def _on_kp(self, data: dict):
        self.kp_widget.on_data(data)
        self.kp_detail.on_data(data)
        self.current_g = data.get('g_scale', 0)
        self._update_summary()

    def _on_proton(self, data: dict):
        self.proton_widget.on_data(data)
        self.proton_detail.on_data(data)
        self.current_s = data.get('s_scale', 0)
        self._update_summary()

    def _on_solarwind(self, data: dict):
        self.solarwind_widget.on_data(data)
        self.solarwind_detail.on_data(data)
        self.current_bz = data.get('bz_gsm', 0)
        self._update_summary()
        self.summary_bar.update_time(datetime.utcnow().strftime("%H:%M:%S UTC"))

    # =========================================================================
    # Advanced Handlers
    # =========================================================================
    def _on_f107(self, data: dict):
        self.f107_widget.update_data(data)
        self.f107_detail.update_data(data)
        flux = data.get('flux', 0)
        self.summary_bar.update_metric("F10.7", f"{flux:.0f}")

    def _on_hpi(self, data: dict):
        self.hpi_widget.update_data(data)
        self.hpi_detail.update_data(data)
        hpi = data.get('hpi_north', 0)
        self.summary_bar.update_metric("HPI", f"{hpi:.0f}")

    def _on_drap(self, data: dict):
        self.drap_widget.update_data(data)
        self.drap_detail.update_data(data)
        haf = data.get('max_haf', 0)
        self.summary_bar.update_metric("HAF", f"{haf:.1f}")

    def _on_ionosonde(self, data: dict):
        self.ionosonde_widget.update_data(data)
        self.ionosonde_detail.update_data(data)
        muf = data.get('global_max_muf', 0)
        self.summary_bar.update_metric("Max MUF", f"{muf:.0f}")

    def _on_enlil(self, data: dict):
        self.enlil_widget.update_data(data)

    def _on_prop_wind(self, data: dict):
        self.prop_wind_widget.update_data(data)

    # =========================================================================
    # Summary Updates
    # =========================================================================
    def _update_summary(self):
        """Update the summary bar based on current conditions."""
        self.summary_bar.update_scales(self.current_r, self.current_g, self.current_s)

        # Determine overall conditions
        max_scale = max(self.current_r, self.current_g, self.current_s)
        southward_bz = self.current_bz < -10

        if max_scale >= 4 or (max_scale >= 2 and southward_bz):
            status = "POOR"
        elif max_scale >= 2 or (max_scale >= 1 and southward_bz):
            status = "FAIR"
        elif max_scale >= 1:
            status = "MODERATE"
        else:
            status = "GOOD"

        self.summary_bar.update_conditions(status)

    # =========================================================================
    # Toolbar Handlers
    # =========================================================================
    def _on_refresh(self):
        """Force immediate data refresh."""
        if self.standard_client.worker:
            self.standard_client.worker._fetch_data()
        if self.advanced_client.worker:
            self.advanced_client.worker._fetch_data()
        self.status_bar.showMessage("Refreshing all data...")

    def _on_interval_changed(self, index: int):
        """Handle update interval change."""
        intervals = [30000, 60000, 120000, 300000]
        interval = intervals[index]

        if self.standard_client.worker and self.standard_client.worker.timer:
            self.standard_client.worker.update_interval_ms = interval
            self.standard_client.worker.timer.setInterval(interval)

        self.advanced_client.set_update_interval(interval)
        self.status_bar.showMessage(f"Update interval: {self.interval_combo.currentText()}")

    def _on_range_changed(self, index: int):
        """Handle history range change."""
        ranges = [1, 6, 12, 24]
        hours = ranges[index]

        # Update all widgets
        for widget in [self.xray_widget, self.kp_widget, self.proton_widget, self.solarwind_widget,
                       self.xray_detail, self.kp_detail, self.proton_detail, self.solarwind_detail,
                       self.f107_widget, self.hpi_widget, self.f107_detail, self.hpi_detail]:
            widget.set_time_range(hours)

        self.status_bar.showMessage(f"History range: {self.range_combo.currentText()}")

    def _on_autoscale_toggle(self, checked: bool):
        """Handle autoscale toggle."""
        for widget in [self.xray_widget, self.kp_widget, self.proton_widget, self.solarwind_widget,
                       self.xray_detail, self.kp_detail, self.proton_detail, self.solarwind_detail,
                       self.f107_widget, self.hpi_widget, self.f107_detail, self.hpi_detail]:
            widget.set_autoscale(checked)

        self.status_bar.showMessage(f"Autoscale: {'On' if checked else 'Off'}")

    def _on_clear_data(self):
        """Clear all historical data."""
        for widget in [self.xray_widget, self.kp_widget, self.proton_widget, self.solarwind_widget,
                       self.xray_detail, self.kp_detail, self.proton_detail, self.solarwind_detail,
                       self.f107_widget, self.hpi_widget, self.drap_widget,
                       self.f107_detail, self.hpi_detail, self.drap_detail,
                       self.prop_wind_widget]:
            widget.clear_data()

        self.status_bar.showMessage("Data cleared")

    def closeEvent(self, event):
        """Clean up on close."""
        self.standard_client.stop()
        self.advanced_client.stop()
        event.accept()


def main():
    """Main entry point for enhanced propagation display."""
    from PyQt6.QtWidgets import QApplication

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    app = QApplication(sys.argv)
    window = EnhancedPropagationWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
