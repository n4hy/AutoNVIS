"""
Advanced Propagation Widgets - Beyond the Standard Four

Display widgets for next-generation space weather data:
- F10.7 Solar Flux Widget
- Hemispheric Power Index Widget
- D-RAP Absorption Widget
- GIRO Ionosonde Widget
- WSA-Enlil Prediction Widget
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QGridLayout,
    QSizePolicy, QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QColor
import pyqtgraph as pg
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
import numpy as np


class BaseAdvancedWidget(QWidget):
    """Base class for advanced propagation widgets."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.title = title
        self.time_range_hours = 24
        self.autoscale_enabled = False

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Header
        header = QFrame()
        header.setStyleSheet("background-color: #1a1a2e; border-radius: 4px;")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(8, 4, 8, 4)

        self.title_label = QLabel(self.title)
        self.title_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self.title_label.setStyleSheet("color: #e0e0e0;")
        header_layout.addWidget(self.title_label)

        header_layout.addStretch()

        self.value_label = QLabel("--")
        self.value_label.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        self.value_label.setStyleSheet("color: #4fc3f7;")
        header_layout.addWidget(self.value_label)

        layout.addWidget(header)

        # Status line
        self.status_label = QLabel("")
        self.status_label.setFont(QFont("Segoe UI", 9))
        self.status_label.setStyleSheet("color: #888;")
        layout.addWidget(self.status_label)

    def set_time_range(self, hours: int):
        self.time_range_hours = hours

    def set_autoscale(self, enabled: bool):
        self.autoscale_enabled = enabled

    def clear_data(self):
        pass

    def _get_status_color(self, status: str) -> str:
        """Return color based on status text."""
        status_lower = status.lower()
        if any(w in status_lower for w in ['excellent', 'quiet', 'normal', 'minimal']):
            return "#4caf50"  # Green
        elif any(w in status_lower for w in ['good', 'fair', 'low', 'minor']):
            return "#8bc34a"  # Light green
        elif any(w in status_lower for w in ['moderate', 'enhanced']):
            return "#ffeb3b"  # Yellow
        elif any(w in status_lower for w in ['poor', 'strong', 'major']):
            return "#ff9800"  # Orange
        elif any(w in status_lower for w in ['severe', 'extreme', 'high', 'cme']):
            return "#f44336"  # Red
        return "#888888"  # Gray


class F107Widget(BaseAdvancedWidget):
    """Widget displaying F10.7 Solar Radio Flux.

    F10.7 is the primary proxy for solar EUV output that maintains F2-layer ionization.
    Higher F10.7 = higher MUF = better HF propagation on higher bands.
    """

    def __init__(self, parent=None):
        super().__init__("F10.7 Solar Flux", parent)
        self._setup_plot()

        self.times: List[float] = []
        self.values: List[float] = []

    def _setup_plot(self):
        layout = self.layout()

        # Create plot
        self.plot = pg.PlotWidget()
        self.plot.setBackground('#0d0d1a')
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setLabel('left', 'SFU', color='#888')
        self.plot.setLabel('bottom', 'Time (UTC)', color='#888')

        # Set Y range for typical F10.7 values
        self.plot.setYRange(60, 250)

        # Threshold lines
        # 150 SFU = Very good propagation
        self.line_150 = pg.InfiniteLine(pos=150, angle=0, pen=pg.mkPen('#4caf50', width=1, style=Qt.PenStyle.DashLine))
        self.plot.addItem(self.line_150)

        # 100 SFU = Moderate baseline
        self.line_100 = pg.InfiniteLine(pos=100, angle=0, pen=pg.mkPen('#ff9800', width=1, style=Qt.PenStyle.DashLine))
        self.plot.addItem(self.line_100)

        # Data curve
        self.curve = self.plot.plot([], [], pen=pg.mkPen('#2196f3', width=2), name='F10.7')

        # 90-day mean line (will be updated)
        self.mean_line = pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen('#9c27b0', width=1.5))
        self.plot.addItem(self.mean_line)

        layout.addWidget(self.plot, stretch=1)

        # Info panel
        info_frame = QFrame()
        info_frame.setStyleSheet("background-color: #1a1a2e; border-radius: 4px;")
        info_layout = QGridLayout(info_frame)
        info_layout.setContentsMargins(8, 4, 8, 4)

        self.trend_label = QLabel("Trend: --")
        self.trend_label.setStyleSheet("color: #888;")
        info_layout.addWidget(self.trend_label, 0, 0)

        self.mean_label = QLabel("90d Mean: --")
        self.mean_label.setStyleSheet("color: #9c27b0;")
        info_layout.addWidget(self.mean_label, 0, 1)

        self.impact_label = QLabel("MUF Impact: --")
        self.impact_label.setStyleSheet("color: #888;")
        info_layout.addWidget(self.impact_label, 0, 2)

        layout.addWidget(info_frame)

    def update_data(self, data: dict):
        """Update widget with new F10.7 data."""
        flux = data.get('flux', 0)
        mean_90 = data.get('ninety_day_mean', flux)
        trend = data.get('trend', 'Stable')
        impact = data.get('muf_impact', 'Unknown')

        # Update value display
        self.value_label.setText(f"{flux:.1f} sfu")
        color = self._get_status_color(impact)
        self.value_label.setStyleSheet(f"color: {color};")

        # Update info labels
        self.trend_label.setText(f"Trend: {trend}")
        self.mean_label.setText(f"90d Mean: {mean_90:.1f}")
        self.impact_label.setText(f"MUF Impact: {impact}")
        self.impact_label.setStyleSheet(f"color: {color};")

        # Update mean line
        self.mean_line.setValue(mean_90)

        # Add to time series
        try:
            ts = datetime.fromisoformat(data.get('timestamp', '').rstrip('Z'))
            self.times.append(ts.timestamp())
            self.values.append(flux)
            self._update_plot()
        except (ValueError, AttributeError):
            pass

        self.status_label.setText(f"Last update: {data.get('timestamp', '--')}")

    def _update_plot(self):
        if not self.times:
            return

        # Trim to time range
        cutoff = datetime.utcnow() - timedelta(hours=self.time_range_hours)
        cutoff_ts = cutoff.timestamp()

        while self.times and self.times[0] < cutoff_ts:
            self.times.pop(0)
            self.values.pop(0)

        if self.times:
            # Normalize times for display
            t0 = self.times[0]
            x = [(t - t0) / 3600 for t in self.times]  # Hours
            self.curve.setData(x, self.values)

            if self.autoscale_enabled:
                self.plot.enableAutoRange()
            else:
                self.plot.setYRange(60, 250)

    def clear_data(self):
        self.times.clear()
        self.values.clear()
        self.curve.setData([], [])


class HPIWidget(BaseAdvancedWidget):
    """Widget displaying Hemispheric Power Index.

    HPI indicates total auroral energy input in Gigawatts.
    Higher HPI = more auroral activity, more ionospheric disturbance.
    """

    def __init__(self, parent=None):
        super().__init__("Hemispheric Power (Aurora)", parent)
        self._setup_plot()

        self.times: List[float] = []
        self.values: List[float] = []

    def _setup_plot(self):
        layout = self.layout()

        # Create plot
        self.plot = pg.PlotWidget()
        self.plot.setBackground('#0d0d1a')
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setLabel('left', 'GW', color='#888')
        self.plot.setLabel('bottom', 'Time (UTC)', color='#888')

        # Set Y range for typical HPI values
        self.plot.setYRange(0, 150)

        # Aurora visibility thresholds
        # 50 GW = Visible in northern US
        self.line_50 = pg.InfiniteLine(pos=50, angle=0, pen=pg.mkPen('#ffeb3b', width=1, style=Qt.PenStyle.DashLine))
        self.plot.addItem(self.line_50)

        # 100 GW = Visible at mid-latitudes (major storm)
        self.line_100 = pg.InfiniteLine(pos=100, angle=0, pen=pg.mkPen('#f44336', width=1, style=Qt.PenStyle.DashLine))
        self.plot.addItem(self.line_100)

        # Data curve
        self.curve = self.plot.plot([], [], pen=pg.mkPen('#ab47bc', width=2), name='HPI')

        layout.addWidget(self.plot, stretch=1)

        # Info panel
        info_frame = QFrame()
        info_frame.setStyleSheet("background-color: #1a1a2e; border-radius: 4px;")
        info_layout = QGridLayout(info_frame)
        info_layout.setContentsMargins(8, 4, 8, 4)

        self.north_label = QLabel("North: -- GW")
        self.north_label.setStyleSheet("color: #888;")
        info_layout.addWidget(self.north_label, 0, 0)

        self.south_label = QLabel("South: -- GW")
        self.south_label.setStyleSheet("color: #888;")
        info_layout.addWidget(self.south_label, 0, 1)

        self.visibility_label = QLabel("Aurora: --")
        self.visibility_label.setStyleSheet("color: #888;")
        info_layout.addWidget(self.visibility_label, 0, 2)

        layout.addWidget(info_frame)

    def update_data(self, data: dict):
        """Update widget with new HPI data."""
        hpi_north = data.get('hpi_north', 0)
        hpi_south = data.get('hpi_south', 0)
        visibility = data.get('aurora_visibility', 'Unknown')
        storm = data.get('storm_level', 'Unknown')

        # Update value display
        self.value_label.setText(f"{hpi_north:.0f} GW")
        color = self._get_status_color(storm)
        self.value_label.setStyleSheet(f"color: {color};")

        # Update info labels
        self.north_label.setText(f"North: {hpi_north:.0f} GW")
        self.south_label.setText(f"South: {hpi_south:.0f} GW")
        self.visibility_label.setText(f"Aurora: {visibility}")
        self.visibility_label.setStyleSheet(f"color: {color};")

        # Add to time series
        try:
            # HPI timestamp format: "YYYY-MM-DD HH:MM"
            ts_str = data.get('timestamp', '')
            ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M") if ts_str else datetime.utcnow()
            self.times.append(ts.timestamp())
            self.values.append(hpi_north)
            self._update_plot()
        except (ValueError, AttributeError):
            pass

        self.status_label.setText(f"Storm Level: {storm}")

    def _update_plot(self):
        if not self.times:
            return

        # Trim to time range
        cutoff = datetime.utcnow() - timedelta(hours=self.time_range_hours)
        cutoff_ts = cutoff.timestamp()

        while self.times and self.times[0] < cutoff_ts:
            self.times.pop(0)
            self.values.pop(0)

        if self.times:
            t0 = self.times[0]
            x = [(t - t0) / 3600 for t in self.times]
            self.curve.setData(x, self.values)

            if self.autoscale_enabled:
                self.plot.enableAutoRange()
            else:
                self.plot.setYRange(0, 150)

    def clear_data(self):
        self.times.clear()
        self.values.clear()
        self.curve.setData([], [])


class DRAPWidget(BaseAdvancedWidget):
    """Widget displaying D-Region Absorption Prediction.

    D-RAP shows the Highest Affected Frequency (HAF).
    Your operating frequency must be above HAF to avoid absorption.
    """

    def __init__(self, parent=None):
        super().__init__("D-RAP Absorption", parent)
        self._setup_display()

    def _setup_display(self):
        layout = self.layout()

        # Main display frame
        main_frame = QFrame()
        main_frame.setStyleSheet("background-color: #0d0d1a; border-radius: 8px;")
        main_layout = QVBoxLayout(main_frame)
        main_layout.setContentsMargins(16, 16, 16, 16)

        # Large HAF display
        self.haf_label = QLabel("--")
        self.haf_label.setFont(QFont("Segoe UI", 36, QFont.Weight.Bold))
        self.haf_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.haf_label.setStyleSheet("color: #4caf50;")
        main_layout.addWidget(self.haf_label)

        haf_subtitle = QLabel("Max Affected Freq (MHz)")
        haf_subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        haf_subtitle.setStyleSheet("color: #666;")
        main_layout.addWidget(haf_subtitle)

        # Status
        self.absorption_status = QLabel("Normal")
        self.absorption_status.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        self.absorption_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.absorption_status.setStyleSheet("color: #4caf50;")
        main_layout.addWidget(self.absorption_status)

        layout.addWidget(main_frame, stretch=1)

        # Info panel
        info_frame = QFrame()
        info_frame.setStyleSheet("background-color: #1a1a2e; border-radius: 4px;")
        info_layout = QGridLayout(info_frame)
        info_layout.setContentsMargins(8, 4, 8, 4)

        self.xray_label = QLabel("X-Ray: --")
        self.xray_label.setStyleSheet("color: #888;")
        info_layout.addWidget(self.xray_label, 0, 0)

        self.proton_label = QLabel("Proton: --")
        self.proton_label.setStyleSheet("color: #888;")
        info_layout.addWidget(self.proton_label, 0, 1)

        layout.addWidget(info_frame)

    def update_data(self, data: dict):
        """Update widget with new D-RAP data."""
        max_haf = data.get('max_haf', 0)
        status = data.get('absorption_status', 'Unknown')
        x_ray = data.get('x_ray_background', 'Unknown')
        proton = data.get('proton_background', 'Unknown')

        # Update HAF display
        self.haf_label.setText(f"{max_haf:.1f}")
        color = self._get_status_color(status)
        self.haf_label.setStyleSheet(f"color: {color};")

        # Update status
        self.absorption_status.setText(status)
        self.absorption_status.setStyleSheet(f"color: {color};")

        # Update value label
        self.value_label.setText(f"{max_haf:.1f} MHz")
        self.value_label.setStyleSheet(f"color: {color};")

        # Update info labels
        self.xray_label.setText(f"X-Ray: {x_ray}")
        self.proton_label.setText(f"Proton: {proton}")

        self.status_label.setText(f"Last update: {data.get('timestamp', '--')}")


class IonosondeWidget(BaseAdvancedWidget):
    """Widget displaying GIRO Ionosonde ground truth data.

    Shows real measured foF2 and MUF from ionosonde stations worldwide.
    This is the "truth" vs model predictions.
    """

    def __init__(self, parent=None):
        super().__init__("Ionosonde Ground Truth", parent)
        self._setup_display()

    def _setup_display(self):
        layout = self.layout()

        # Stats frame
        stats_frame = QFrame()
        stats_frame.setStyleSheet("background-color: #0d0d1a; border-radius: 4px;")
        stats_layout = QGridLayout(stats_frame)
        stats_layout.setContentsMargins(8, 8, 8, 8)

        # Global max MUF
        muf_title = QLabel("Global Max MUF")
        muf_title.setStyleSheet("color: #666;")
        stats_layout.addWidget(muf_title, 0, 0)

        self.max_muf_label = QLabel("--")
        self.max_muf_label.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        self.max_muf_label.setStyleSheet("color: #4caf50;")
        stats_layout.addWidget(self.max_muf_label, 1, 0)

        muf_unit = QLabel("MHz")
        muf_unit.setStyleSheet("color: #666;")
        stats_layout.addWidget(muf_unit, 2, 0)

        # Global max foF2
        fof2_title = QLabel("Global Max foF2")
        fof2_title.setStyleSheet("color: #666;")
        stats_layout.addWidget(fof2_title, 0, 1)

        self.max_fof2_label = QLabel("--")
        self.max_fof2_label.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        self.max_fof2_label.setStyleSheet("color: #2196f3;")
        stats_layout.addWidget(self.max_fof2_label, 1, 1)

        fof2_unit = QLabel("MHz")
        fof2_unit.setStyleSheet("color: #666;")
        stats_layout.addWidget(fof2_unit, 2, 1)

        # Station count
        count_title = QLabel("Active Stations")
        count_title.setStyleSheet("color: #666;")
        stats_layout.addWidget(count_title, 0, 2)

        self.count_label = QLabel("--")
        self.count_label.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        self.count_label.setStyleSheet("color: #9c27b0;")
        stats_layout.addWidget(self.count_label, 1, 2)

        layout.addWidget(stats_frame)

        # Top stations table
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(['Station', 'foF2', 'MUF', 'hmF2'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: #0d0d1a;
                color: #e0e0e0;
                gridline-color: #333;
            }
            QHeaderView::section {
                background-color: #1a1a2e;
                color: #888;
                padding: 4px;
                border: none;
            }
        """)
        self.table.setMaximumHeight(150)
        layout.addWidget(self.table)

        # Outlook
        self.outlook_label = QLabel("Propagation: --")
        self.outlook_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.outlook_label.setStyleSheet("color: #888;")
        layout.addWidget(self.outlook_label)

    def update_data(self, data: dict):
        """Update widget with new ionosonde data."""
        max_muf = data.get('global_max_muf', 0)
        max_fof2 = data.get('global_max_fof2', 0)
        count = data.get('station_count', 0)
        outlook = data.get('propagation_outlook', 'Unknown')
        top_stations = data.get('top_stations', [])

        # Update stats
        self.max_muf_label.setText(f"{max_muf:.1f}")
        self.max_fof2_label.setText(f"{max_fof2:.1f}")
        self.count_label.setText(str(count))

        # Update value label
        self.value_label.setText(f"MUF {max_muf:.0f} MHz")
        color = self._get_status_color(outlook)
        self.value_label.setStyleSheet(f"color: {color};")

        # Update outlook
        self.outlook_label.setText(f"Propagation: {outlook}")
        self.outlook_label.setStyleSheet(f"color: {color};")

        # Update table
        self.table.setRowCount(min(5, len(top_stations)))
        for i, station in enumerate(top_stations[:5]):
            self.table.setItem(i, 0, QTableWidgetItem(station.get('name', 'Unknown')[:20]))
            self.table.setItem(i, 1, QTableWidgetItem(f"{station.get('fof2', 0):.1f}"))
            self.table.setItem(i, 2, QTableWidgetItem(f"{station.get('muf', 0):.1f}"))
            self.table.setItem(i, 3, QTableWidgetItem(f"{station.get('hmf2', 0):.0f}"))

        self.status_label.setText(f"Updated: {data.get('timestamp', '--')[:19]}")


class EnlilWidget(BaseAdvancedWidget):
    """Widget displaying WSA-Enlil solar wind prediction.

    Shows predicted solar wind conditions and CME arrival alerts.
    """

    def __init__(self, parent=None):
        super().__init__("WSA-Enlil Prediction", parent)
        self._setup_display()

    def _setup_display(self):
        layout = self.layout()

        # Main display
        main_frame = QFrame()
        main_frame.setStyleSheet("background-color: #0d0d1a; border-radius: 8px;")
        main_layout = QVBoxLayout(main_frame)
        main_layout.setContentsMargins(16, 12, 16, 12)

        # Density display
        self.density_label = QLabel("--")
        self.density_label.setFont(QFont("Segoe UI", 28, QFont.Weight.Bold))
        self.density_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.density_label.setStyleSheet("color: #4fc3f7;")
        main_layout.addWidget(self.density_label)

        density_subtitle = QLabel("Predicted Density (p/cm³)")
        density_subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        density_subtitle.setStyleSheet("color: #666;")
        main_layout.addWidget(density_subtitle)

        # CME Alert indicator
        self.cme_indicator = QLabel("CME STATUS")
        self.cme_indicator.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self.cme_indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cme_indicator.setStyleSheet("""
            background-color: #1a1a2e;
            color: #4caf50;
            padding: 8px;
            border-radius: 4px;
        """)
        main_layout.addWidget(self.cme_indicator)

        layout.addWidget(main_frame, stretch=1)

        # Info panel
        info_frame = QFrame()
        info_frame.setStyleSheet("background-color: #1a1a2e; border-radius: 4px;")
        info_layout = QGridLayout(info_frame)
        info_layout.setContentsMargins(8, 4, 8, 4)

        self.speed_label = QLabel("Speed: -- km/s")
        self.speed_label.setStyleSheet("color: #888;")
        info_layout.addWidget(self.speed_label, 0, 0)

        self.status_detail = QLabel("Status: --")
        self.status_detail.setStyleSheet("color: #888;")
        info_layout.addWidget(self.status_detail, 0, 1)

        layout.addWidget(info_frame)

    def update_data(self, data: dict):
        """Update widget with new Enlil data."""
        density = data.get('predicted_density', 0)
        speed = data.get('predicted_speed', 0)
        status = data.get('density_status', 'Unknown')
        cme_alert = data.get('cme_alert', False)

        # Update density display
        self.density_label.setText(f"{density:.1f}")
        color = self._get_status_color(status)
        self.density_label.setStyleSheet(f"color: {color};")

        # Update CME indicator
        if cme_alert:
            self.cme_indicator.setText("⚠ CME LIKELY")
            self.cme_indicator.setStyleSheet("""
                background-color: #f44336;
                color: #ffffff;
                padding: 8px;
                border-radius: 4px;
            """)
        else:
            self.cme_indicator.setText("✓ No CME Detected")
            self.cme_indicator.setStyleSheet("""
                background-color: #1a1a2e;
                color: #4caf50;
                padding: 8px;
                border-radius: 4px;
            """)

        # Update value label
        self.value_label.setText(status)
        self.value_label.setStyleSheet(f"color: {color};")

        # Update info
        self.speed_label.setText(f"Speed: {speed:.0f} km/s")
        self.status_detail.setText(f"Status: {status}")

        self.status_label.setText(f"Last update: {data.get('timestamp', '--')[:19]}")


class PropagatedWindWidget(BaseAdvancedWidget):
    """Widget displaying propagated solar wind (1-hour forecast)."""

    def __init__(self, parent=None):
        super().__init__("Solar Wind Forecast (1h)", parent)
        self._setup_display()

        self.times: List[float] = []
        self.speeds: List[float] = []
        self.bz_values: List[float] = []

    def _setup_display(self):
        layout = self.layout()

        # Create plot
        self.plot = pg.PlotWidget()
        self.plot.setBackground('#0d0d1a')
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setLabel('left', 'Speed (km/s)', color='#888')

        # Add second Y axis for Bz
        self.p2 = pg.ViewBox()
        self.plot.scene().addItem(self.p2)
        self.plot.getAxis('right').linkToView(self.p2)
        self.p2.setXLink(self.plot)
        self.plot.getAxis('right').setLabel('Bz (nT)', color='#ab47bc')
        self.plot.showAxis('right')

        # Speed curve
        self.speed_curve = self.plot.plot([], [], pen=pg.mkPen('#4fc3f7', width=2), name='Speed')

        # Bz curve (on second axis)
        self.bz_curve = pg.PlotCurveItem(pen=pg.mkPen('#ab47bc', width=2))
        self.p2.addItem(self.bz_curve)

        # Zero line for Bz
        self.zero_line = pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen('#666', width=1, style=Qt.PenStyle.DashLine))
        self.p2.addItem(self.zero_line)

        layout.addWidget(self.plot, stretch=1)

        # Info panel
        info_frame = QFrame()
        info_frame.setStyleSheet("background-color: #1a1a2e; border-radius: 4px;")
        info_layout = QGridLayout(info_frame)
        info_layout.setContentsMargins(8, 4, 8, 4)

        self.speed_info = QLabel("Speed: -- km/s")
        self.speed_info.setStyleSheet("color: #4fc3f7;")
        info_layout.addWidget(self.speed_info, 0, 0)

        self.bz_info = QLabel("Bz: -- nT")
        self.bz_info.setStyleSheet("color: #ab47bc;")
        info_layout.addWidget(self.bz_info, 0, 1)

        self.storm_info = QLabel("Storm Potential: --")
        self.storm_info.setStyleSheet("color: #888;")
        info_layout.addWidget(self.storm_info, 0, 2)

        layout.addWidget(info_frame)

        # Handle view synchronization
        self.plot.getViewBox().sigResized.connect(self._update_views)

    def _update_views(self):
        self.p2.setGeometry(self.plot.getViewBox().sceneBoundingRect())
        self.p2.linkedViewChanged(self.plot.getViewBox(), self.p2.XAxis)

    def update_data(self, data: dict):
        """Update widget with new propagated wind data."""
        speed = data.get('speed', 0)
        bz = data.get('bz', 0)
        storm = data.get('storm_potential', 'Unknown')

        # Update value label
        self.value_label.setText(f"{speed:.0f} km/s")
        color = self._get_status_color(storm)
        self.value_label.setStyleSheet(f"color: {color};")

        # Update info
        self.speed_info.setText(f"Speed: {speed:.0f} km/s")
        self.bz_info.setText(f"Bz: {bz:.1f} nT")
        bz_color = "#f44336" if bz < -5 else "#4caf50" if bz > 0 else "#ffeb3b"
        self.bz_info.setStyleSheet(f"color: {bz_color};")

        self.storm_info.setText(f"Storm Potential: {storm}")
        self.storm_info.setStyleSheet(f"color: {color};")

        # Add to time series
        try:
            ts_str = data.get('timestamp', '')
            if ts_str:
                ts = datetime.fromisoformat(ts_str.rstrip('Z'))
                self.times.append(ts.timestamp())
                self.speeds.append(speed)
                self.bz_values.append(bz)
                self._update_plot()
        except (ValueError, AttributeError):
            pass

        self.status_label.setText(f"Arrival: {data.get('propagated_time', '--')[:19]}")

    def _update_plot(self):
        if not self.times:
            return

        # Trim to 1 hour
        cutoff = datetime.utcnow() - timedelta(hours=1)
        cutoff_ts = cutoff.timestamp()

        while self.times and self.times[0] < cutoff_ts:
            self.times.pop(0)
            self.speeds.pop(0)
            self.bz_values.pop(0)

        if self.times:
            t0 = self.times[0]
            x = [(t - t0) / 60 for t in self.times]  # Minutes

            self.speed_curve.setData(x, self.speeds)
            self.bz_curve.setData(x, self.bz_values)

            # Set ranges
            self.plot.setYRange(min(300, min(self.speeds) - 50), max(700, max(self.speeds) + 50))
            self.p2.setYRange(min(-15, min(self.bz_values) - 5), max(15, max(self.bz_values) + 5))

    def clear_data(self):
        self.times.clear()
        self.speeds.clear()
        self.bz_values.clear()
        self.speed_curve.setData([], [])
        self.bz_curve.setData([], [])
