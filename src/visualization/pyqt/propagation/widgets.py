"""
Propagation Dashboard Widgets

Individual widgets for X-ray, Kp, Proton flux, and Solar Wind displays.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QGridLayout
)
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QFont
import pyqtgraph as pg
import numpy as np
from datetime import datetime
from typing import List


class ScaleIndicator(QWidget):
    """NOAA scale indicator (R/S/G) with color coding."""

    SCALE_COLORS = {
        0: '#44aa44',  # Green - normal
        1: '#aaaa44',  # Yellow - minor
        2: '#ddaa44',  # Gold - moderate
        3: '#dd6644',  # Orange - strong
        4: '#dd4444',  # Red - severe
        5: '#aa44aa',  # Magenta - extreme
    }

    def __init__(self, scale_letter: str, parent=None):
        super().__init__(parent)
        self.scale_letter = scale_letter
        self.current_level = 0
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Scale label
        self.scale_label = QLabel(f"{self.scale_letter}0")
        font = QFont()
        font.setPointSize(24)
        font.setBold(True)
        self.scale_label.setFont(font)
        self.scale_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scale_label.setMinimumWidth(60)
        self._update_color()

        layout.addWidget(self.scale_label)

    def set_level(self, level: int):
        """Set the scale level (0-5)."""
        self.current_level = max(0, min(5, level))
        self.scale_label.setText(f"{self.scale_letter}{self.current_level}")
        self._update_color()

    def _update_color(self):
        color = self.SCALE_COLORS.get(self.current_level, '#888888')
        self.scale_label.setStyleSheet(f"""
            color: {color};
            background-color: #2a2a2a;
            border: 2px solid {color};
            border-radius: 8px;
            padding: 5px 10px;
        """)


class XRayWidget(QWidget):
    """X-ray flux display with R-scale indicator."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.times: List[float] = []
        self.fluxes: List[float] = []
        self.time_range_hours = 24
        self.autoscale_enabled = False
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Header
        header = QHBoxLayout()
        title = QLabel("X-Ray Flux (R-Scale)")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        header.addWidget(title)
        header.addStretch()

        self.scale_indicator = ScaleIndicator("R")
        header.addWidget(self.scale_indicator)

        self.class_label = QLabel("--")
        self.class_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #00ff00;")
        header.addWidget(self.class_label)

        layout.addLayout(header)

        # Plot
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#1e1e1e')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLogMode(x=False, y=True)
        self.plot_widget.setLabel('left', 'W/m²')

        time_axis = pg.DateAxisItem(orientation='bottom')
        self.plot_widget.setAxisItems({'bottom': time_axis})

        # Threshold lines
        for flux, color in [(1e-5, '#ffff44'), (1e-4, '#ff4444')]:
            line = pg.InfiniteLine(pos=np.log10(flux), angle=0,
                                   pen=pg.mkPen(color, width=1, style=Qt.PenStyle.DashLine))
            self.plot_widget.addItem(line)

        self.curve = self.plot_widget.plot(pen=pg.mkPen('#00ff00', width=2),
                                           symbol='o', symbolSize=4, symbolBrush='#00ff00')
        layout.addWidget(self.plot_widget)

    @pyqtSlot(dict)
    def on_data(self, data: dict):
        """Handle X-ray data update."""
        try:
            ts = datetime.fromisoformat(data['timestamp'].rstrip('Z'))
            ts_float = ts.timestamp()

            self.times.append(ts_float)
            self.fluxes.append(data['flux'])

            # Trim to configured time range
            cutoff = datetime.utcnow().timestamp() - (self.time_range_hours * 3600)
            while self.times and self.times[0] < cutoff:
                self.times.pop(0)
                self.fluxes.pop(0)

            self.curve.setData(self.times, self.fluxes)

            # Update indicators
            self.scale_indicator.set_level(data.get('r_scale', 0))
            self.class_label.setText(data.get('flare_class', '--'))

            if len(self.times) > 1:
                self.plot_widget.setXRange(self.times[0], self.times[-1], padding=0.02)

        except Exception:
            pass

    def set_time_range(self, hours: int):
        """Set the time range to display."""
        self.time_range_hours = hours
        # Trim existing data to new range
        cutoff = datetime.utcnow().timestamp() - (hours * 3600)
        while self.times and self.times[0] < cutoff:
            self.times.pop(0)
            self.fluxes.pop(0)
        if self.times:
            self.curve.setData(self.times, self.fluxes)
            self.plot_widget.setXRange(self.times[0], self.times[-1], padding=0.02)

    def set_autoscale(self, enabled: bool):
        """Enable or disable Y-axis autoscaling."""
        self.autoscale_enabled = enabled
        if enabled:
            self.plot_widget.enableAutoRange(axis='y')
        else:
            # Fixed range for X-ray flux (log scale)
            self.plot_widget.setYRange(np.log10(1e-9), np.log10(1e-3), padding=0)

    def clear_data(self):
        """Clear all data."""
        self.times.clear()
        self.fluxes.clear()
        self.curve.setData([], [])
        self.scale_indicator.set_level(0)
        self.class_label.setText("--")


class KpWidget(QWidget):
    """Kp index display with G-scale indicator."""

    KP_COLORS = {
        0: '#44aa44', 1: '#44aa44', 2: '#88aa44', 3: '#aaaa44',
        4: '#ccaa44', 5: '#ddaa44', 6: '#dd8844', 7: '#dd6644',
        8: '#dd4444', 9: '#aa44aa'
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.times: List[float] = []
        self.kp_values: List[float] = []
        self.time_range_hours = 24
        self.autoscale_enabled = False
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Header
        header = QHBoxLayout()
        title = QLabel("Kp Index (G-Scale)")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        header.addWidget(title)
        header.addStretch()

        self.scale_indicator = ScaleIndicator("G")
        header.addWidget(self.scale_indicator)

        self.kp_label = QLabel("Kp 0")
        self.kp_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        header.addWidget(self.kp_label)

        layout.addLayout(header)

        # Plot
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#1e1e1e')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setYRange(0, 9)
        self.plot_widget.setLabel('left', 'Kp')

        time_axis = pg.DateAxisItem(orientation='bottom')
        self.plot_widget.setAxisItems({'bottom': time_axis})

        # Storm thresholds
        for kp, color in [(5, '#ddaa44'), (7, '#dd4444')]:
            line = pg.InfiniteLine(pos=kp, angle=0,
                                   pen=pg.mkPen(color, width=1, style=Qt.PenStyle.DashLine))
            self.plot_widget.addItem(line)

        self.curve = self.plot_widget.plot(pen=pg.mkPen('#44aaff', width=2),
                                           symbol='o', symbolSize=4, symbolBrush='#44aaff')
        layout.addWidget(self.plot_widget)

    @pyqtSlot(dict)
    def on_data(self, data: dict):
        """Handle Kp data update."""
        try:
            ts = datetime.fromisoformat(data['timestamp'].rstrip('Z'))
            ts_float = ts.timestamp()

            kp = data.get('estimated_kp', data.get('kp_index', 0))

            self.times.append(ts_float)
            self.kp_values.append(kp)

            # Trim to configured time range
            cutoff = datetime.utcnow().timestamp() - (self.time_range_hours * 3600)
            while self.times and self.times[0] < cutoff:
                self.times.pop(0)
                self.kp_values.pop(0)

            self.curve.setData(self.times, self.kp_values)

            # Update indicators
            kp_int = int(round(kp))
            self.scale_indicator.set_level(data.get('g_scale', 0))
            color = self.KP_COLORS.get(kp_int, '#888888')
            self.kp_label.setText(f"Kp {kp:.1f}")
            self.kp_label.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {color};")

            if len(self.times) > 1:
                self.plot_widget.setXRange(self.times[0], self.times[-1], padding=0.02)

        except Exception:
            pass

    def set_time_range(self, hours: int):
        """Set the time range to display."""
        self.time_range_hours = hours
        cutoff = datetime.utcnow().timestamp() - (hours * 3600)
        while self.times and self.times[0] < cutoff:
            self.times.pop(0)
            self.kp_values.pop(0)
        if self.times:
            self.curve.setData(self.times, self.kp_values)
            self.plot_widget.setXRange(self.times[0], self.times[-1], padding=0.02)

    def set_autoscale(self, enabled: bool):
        """Enable or disable Y-axis autoscaling."""
        self.autoscale_enabled = enabled
        if enabled:
            self.plot_widget.enableAutoRange(axis='y')
        else:
            self.plot_widget.setYRange(0, 9, padding=0)

    def clear_data(self):
        """Clear all data."""
        self.times.clear()
        self.kp_values.clear()
        self.curve.setData([], [])
        self.scale_indicator.set_level(0)
        self.kp_label.setText("Kp 0")


class ProtonWidget(QWidget):
    """Proton flux display with S-scale indicator."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.times: List[float] = []
        self.fluxes: List[float] = []
        self.time_range_hours = 24
        self.autoscale_enabled = False
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Header
        header = QHBoxLayout()
        title = QLabel("Proton Flux >=10 MeV (S-Scale)")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        header.addWidget(title)
        header.addStretch()

        self.scale_indicator = ScaleIndicator("S")
        header.addWidget(self.scale_indicator)

        self.flux_label = QLabel("--")
        self.flux_label.setStyleSheet("font-size: 14px; color: #ffaa44;")
        header.addWidget(self.flux_label)

        layout.addLayout(header)

        # Plot
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#1e1e1e')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLogMode(x=False, y=True)
        self.plot_widget.setLabel('left', 'pfu')

        time_axis = pg.DateAxisItem(orientation='bottom')
        self.plot_widget.setAxisItems({'bottom': time_axis})

        # S1 threshold (10 pfu)
        line = pg.InfiniteLine(pos=np.log10(10), angle=0,
                               pen=pg.mkPen('#ddaa44', width=1, style=Qt.PenStyle.DashLine))
        self.plot_widget.addItem(line)

        self.curve = self.plot_widget.plot(pen=pg.mkPen('#ffaa44', width=2),
                                           symbol='o', symbolSize=4, symbolBrush='#ffaa44')
        layout.addWidget(self.plot_widget)

    @pyqtSlot(dict)
    def on_data(self, data: dict):
        """Handle proton data update."""
        try:
            ts = datetime.fromisoformat(data['timestamp'].rstrip('Z'))
            ts_float = ts.timestamp()

            flux = data.get('flux', 0)
            if flux <= 0:
                flux = 0.01  # Minimum for log scale

            self.times.append(ts_float)
            self.fluxes.append(flux)

            # Trim to configured time range
            cutoff = datetime.utcnow().timestamp() - (self.time_range_hours * 3600)
            while self.times and self.times[0] < cutoff:
                self.times.pop(0)
                self.fluxes.pop(0)

            self.curve.setData(self.times, self.fluxes)

            # Update indicators
            self.scale_indicator.set_level(data.get('s_scale', 0))
            self.flux_label.setText(f"{flux:.1f} pfu")

            if len(self.times) > 1:
                self.plot_widget.setXRange(self.times[0], self.times[-1], padding=0.02)

        except Exception:
            pass

    def set_time_range(self, hours: int):
        """Set the time range to display."""
        self.time_range_hours = hours
        cutoff = datetime.utcnow().timestamp() - (hours * 3600)
        while self.times and self.times[0] < cutoff:
            self.times.pop(0)
            self.fluxes.pop(0)
        if self.times:
            self.curve.setData(self.times, self.fluxes)
            self.plot_widget.setXRange(self.times[0], self.times[-1], padding=0.02)

    def set_autoscale(self, enabled: bool):
        """Enable or disable Y-axis autoscaling."""
        self.autoscale_enabled = enabled
        if enabled:
            self.plot_widget.enableAutoRange(axis='y')
        else:
            # Fixed range for proton flux (log scale)
            self.plot_widget.setYRange(np.log10(0.01), np.log10(1e5), padding=0)

    def clear_data(self):
        """Clear all data."""
        self.times.clear()
        self.fluxes.clear()
        self.curve.setData([], [])
        self.scale_indicator.set_level(0)
        self.flux_label.setText("--")


class SolarWindWidget(QWidget):
    """Solar wind Bz display for storm precursor monitoring."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.times: List[float] = []
        self.bz_values: List[float] = []
        self.time_range_hours = 24
        self.autoscale_enabled = False
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Header
        header = QHBoxLayout()
        title = QLabel("Solar Wind Bz (Storm Precursor)")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        header.addWidget(title)
        header.addStretch()

        self.status_label = QLabel("--")
        self.status_label.setStyleSheet("font-size: 14px; padding: 3px 8px; border-radius: 4px;")
        header.addWidget(self.status_label)

        self.bz_label = QLabel("Bz: -- nT")
        self.bz_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        header.addWidget(self.bz_label)

        layout.addLayout(header)

        # Plot
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#1e1e1e')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel('left', 'Bz (nT)')

        time_axis = pg.DateAxisItem(orientation='bottom')
        self.plot_widget.setAxisItems({'bottom': time_axis})

        # Zero line
        zero_line = pg.InfiniteLine(pos=0, angle=0,
                                    pen=pg.mkPen('#888888', width=1))
        self.plot_widget.addItem(zero_line)

        # Significant southward threshold (-10 nT)
        thresh_line = pg.InfiniteLine(pos=-10, angle=0,
                                      pen=pg.mkPen('#dd4444', width=1, style=Qt.PenStyle.DashLine))
        self.plot_widget.addItem(thresh_line)

        self.curve = self.plot_widget.plot(pen=pg.mkPen('#aa44ff', width=2),
                                           symbol='o', symbolSize=4, symbolBrush='#aa44ff')
        layout.addWidget(self.plot_widget)

    @pyqtSlot(dict)
    def on_data(self, data: dict):
        """Handle solar wind data update."""
        try:
            ts_str = data.get('timestamp', '')
            if not ts_str:
                return

            # Parse timestamp (format: "2026-02-16 01:47:00.000")
            ts = datetime.fromisoformat(ts_str.replace(' ', 'T').split('.')[0])
            ts_float = ts.timestamp()

            bz = data.get('bz_gsm', 0)

            self.times.append(ts_float)
            self.bz_values.append(bz)

            # Trim to configured time range
            cutoff = datetime.utcnow().timestamp() - (self.time_range_hours * 3600)
            while self.times and self.times[0] < cutoff:
                self.times.pop(0)
                self.bz_values.pop(0)

            self.curve.setData(self.times, self.bz_values)

            # Update indicators
            if bz < -10:
                status_color = '#dd4444'
                status_text = "SOUTHWARD"
            elif bz < 0:
                status_color = '#ddaa44'
                status_text = "Southward"
            else:
                status_color = '#44aa44'
                status_text = "Northward"

            self.status_label.setText(status_text)
            self.status_label.setStyleSheet(f"""
                font-size: 14px;
                padding: 3px 8px;
                border-radius: 4px;
                background-color: {status_color}33;
                color: {status_color};
                border: 1px solid {status_color};
            """)

            bz_color = '#dd4444' if bz < -10 else ('#ddaa44' if bz < 0 else '#44aa44')
            self.bz_label.setText(f"Bz: {bz:+.1f} nT")
            self.bz_label.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {bz_color};")

            if len(self.times) > 1:
                self.plot_widget.setXRange(self.times[0], self.times[-1], padding=0.02)

        except Exception:
            pass

    def set_time_range(self, hours: int):
        """Set the time range to display."""
        self.time_range_hours = hours
        cutoff = datetime.utcnow().timestamp() - (hours * 3600)
        while self.times and self.times[0] < cutoff:
            self.times.pop(0)
            self.bz_values.pop(0)
        if self.times:
            self.curve.setData(self.times, self.bz_values)
            self.plot_widget.setXRange(self.times[0], self.times[-1], padding=0.02)

    def set_autoscale(self, enabled: bool):
        """Enable or disable Y-axis autoscaling."""
        self.autoscale_enabled = enabled
        if enabled:
            self.plot_widget.enableAutoRange(axis='y')
        else:
            # Fixed range for Bz (-30 to +30 nT)
            self.plot_widget.setYRange(-30, 30, padding=0)

    def clear_data(self):
        """Clear all data."""
        self.times.clear()
        self.bz_values.clear()
        self.curve.setData([], [])
        self.status_label.setText("--")
        self.bz_label.setText("Bz: -- nT")
