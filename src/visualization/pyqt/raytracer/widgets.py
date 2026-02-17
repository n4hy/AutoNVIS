"""
Ray Tracer Widgets

Individual widgets for ray tracing visualization.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QGridLayout,
    QDoubleSpinBox, QSpinBox, QPushButton, QGroupBox, QProgressBar
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
import pyqtgraph as pg
import numpy as np
from typing import List, Dict, Optional


class FrequencyIndicator(QWidget):
    """LUF/MUF frequency indicator with color-coded bar."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.luf = 0.0
        self.muf = 0.0
        self.fot = 0.0
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # Title
        title = QLabel("Frequency Window")
        title.setStyleSheet("font-weight: bold; font-size: 16px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Frequency display
        freq_layout = QHBoxLayout()

        # LUF
        luf_box = QVBoxLayout()
        luf_label = QLabel("LUF")
        luf_label.setStyleSheet("color: #ff8844; font-weight: bold;")
        luf_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.luf_value = QLabel("-- MHz")
        self.luf_value.setStyleSheet("font-size: 24px; font-weight: bold; color: #ff8844;")
        self.luf_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        luf_box.addWidget(luf_label)
        luf_box.addWidget(self.luf_value)
        freq_layout.addLayout(luf_box)

        # FOT (optimal)
        fot_box = QVBoxLayout()
        fot_label = QLabel("FOT (Optimal)")
        fot_label.setStyleSheet("color: #44ff44; font-weight: bold;")
        fot_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.fot_value = QLabel("-- MHz")
        self.fot_value.setStyleSheet("font-size: 28px; font-weight: bold; color: #44ff44;")
        self.fot_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        fot_box.addWidget(fot_label)
        fot_box.addWidget(self.fot_value)
        freq_layout.addLayout(fot_box)

        # MUF
        muf_box = QVBoxLayout()
        muf_label = QLabel("MUF")
        muf_label.setStyleSheet("color: #4488ff; font-weight: bold;")
        muf_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.muf_value = QLabel("-- MHz")
        self.muf_value.setStyleSheet("font-size: 24px; font-weight: bold; color: #4488ff;")
        self.muf_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        muf_box.addWidget(muf_label)
        muf_box.addWidget(self.muf_value)
        freq_layout.addLayout(muf_box)

        layout.addLayout(freq_layout)

        # Frequency bar - simple text version (avoid LinearRegionItem issues)
        bar_layout = QHBoxLayout()

        self.range_label = QLabel("Usable: -- to -- MHz")
        self.range_label.setStyleSheet("""
            font-size: 16px;
            font-weight: bold;
            padding: 10px;
            background-color: #2a2a2a;
            border-radius: 5px;
        """)
        self.range_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        bar_layout.addWidget(self.range_label)

        layout.addLayout(bar_layout)

        # Status
        self.status_label = QLabel("Calculating...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("font-size: 14px; color: #888;")
        layout.addWidget(self.status_label)

    def update_frequencies(self, luf: float, muf: float, fot: float, blackout: bool = False):
        """Update frequency display."""
        try:
            self.luf = luf
            self.muf = muf
            self.fot = fot

            self.luf_value.setText(f"{luf:.1f} MHz")
            self.muf_value.setText(f"{muf:.1f} MHz")
            self.fot_value.setText(f"{fot:.1f} MHz")

            # Update range label
            self.range_label.setText(f"Usable: {luf:.1f} to {muf:.1f} MHz")

            if blackout:
                self.status_label.setText("BLACKOUT - No usable frequencies")
                self.status_label.setStyleSheet("font-size: 14px; color: #ff4444; font-weight: bold;")
                self.range_label.setStyleSheet("""
                    font-size: 16px; font-weight: bold; padding: 10px;
                    background-color: #442222; border-radius: 5px; color: #ff4444;
                """)
            else:
                bandwidth = muf - luf
                self.status_label.setText(f"Usable bandwidth: {bandwidth:.1f} MHz")
                self.status_label.setStyleSheet("font-size: 14px; color: #44ff44;")
                self.range_label.setStyleSheet("""
                    font-size: 16px; font-weight: bold; padding: 10px;
                    background-color: #224422; border-radius: 5px; color: #44ff44;
                """)
        except Exception:
            pass


class RayPathWidget(QWidget):
    """Displays ray paths in altitude vs ground range cross-section."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Header
        header = QHBoxLayout()
        title = QLabel("Ray Paths (Cross-Section)")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        header.addWidget(title)
        header.addStretch()

        self.info_label = QLabel("")
        self.info_label.setStyleSheet("color: #888;")
        header.addWidget(self.info_label)

        layout.addLayout(header)

        # Plot
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#1e1e1e')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel('left', 'Altitude (km)')
        self.plot_widget.setLabel('bottom', 'Ground Range (km)')
        self.plot_widget.disableAutoRange()
        self.max_alt = 0
        self.max_range = 0

        # Earth surface line
        earth_x = np.linspace(-600, 600, 100)
        self.plot_widget.plot(earth_x, np.zeros_like(earth_x),
                              pen=pg.mkPen('#444444', width=2))

        # Ionospheric layers (approximate)
        for alt, name, color in [(100, 'D', '#44444488'), (150, 'E', '#44884488'), (300, 'F', '#88444488')]:
            line = pg.InfiniteLine(pos=alt, angle=0,
                                   pen=pg.mkPen(color, width=1, style=Qt.PenStyle.DashLine))
            self.plot_widget.addItem(line)

        self.ray_curves = []
        layout.addWidget(self.plot_widget)

    def clear_rays(self):
        """Clear all ray paths."""
        for curve in self.ray_curves:
            self.plot_widget.removeItem(curve)
        self.ray_curves = []
        self.max_alt = 0
        self.max_range = 0

    def add_ray_path(self, positions, freq_mhz: float,
                     reflected: bool, color: str = None):
        """Add a ray path to the display."""
        try:
            if positions is None or len(positions) < 2:
                return

            # Handle both list and numpy array
            if hasattr(positions, 'tolist'):
                positions = positions.tolist()

            # Extract lat, lon, alt - positions are relative offsets from TX
            lats = [float(p[0]) for p in positions]
            lons = [float(p[1]) for p in positions]
            alts = [float(p[2]) for p in positions]

            # Calculate ground range from first point (TX location)
            tx_lat = lats[0]
            tx_lon = lons[0]

            ground_range = []
            for lat, lon in zip(lats, lons):
                dlat = lat - tx_lat
                dlon = lon - tx_lon
                # km per degree at equator ~111 km
                rng = np.sqrt((dlat * 111)**2 + (dlon * 111)**2)
                # Make it signed based on direction
                if dlat < 0:
                    rng = -rng
                ground_range.append(rng)

            if color is None:
                # Color based on frequency
                freq_norm = (freq_mhz - 2) / 13  # Normalize 2-15 MHz to 0-1
                freq_norm = max(0, min(1, freq_norm))
                r = int(255 * (1 - freq_norm))
                b = int(255 * freq_norm)
                color = f'#{r:02x}88{b:02x}'

            pen = pg.mkPen(color, width=2 if reflected else 1)
            if not reflected:
                pen.setStyle(Qt.PenStyle.DashLine)

            curve = self.plot_widget.plot(ground_range, alts, pen=pen)
            self.ray_curves.append(curve)

            # Only track reflected rays for scaling (ignore escaped rays that go to 800km)
            if reflected and alts:
                self.max_alt = max(self.max_alt, max(alts))
            if ground_range:
                self.max_range = max(self.max_range, max(abs(r) for r in ground_range))

        except Exception as e:
            pass  # Skip invalid paths

    def update_scale(self):
        """Update plot scale to fit all rays."""
        # Use actual max altitude, with minimum of 100km for visibility
        if self.max_alt > 0:
            y_max = self.max_alt * 1.1
        else:
            y_max = 400  # Default if no reflected rays
        self.plot_widget.setYRange(0, y_max, padding=0)

        if self.max_range > 0:
            x_max = self.max_range * 1.1
        else:
            x_max = 100
        self.plot_widget.setXRange(-x_max, x_max, padding=0)

    def set_info(self, text: str):
        """Set info label text."""
        self.info_label.setText(text)


class CoverageMapWidget(QWidget):
    """Displays NVIS coverage map."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.tx_lat = 0.0
        self.tx_lon = 0.0
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Header
        header = QHBoxLayout()
        title = QLabel("NVIS Coverage Map")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        header.addWidget(title)
        header.addStretch()
        layout.addLayout(header)

        # Plot
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#1e1e1e')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel('left', 'Latitude (deg)')
        self.plot_widget.setLabel('bottom', 'Longitude (deg)')
        self.plot_widget.setAspectLocked(True)
        self.plot_widget.disableAutoRange()

        # Coverage circles (will be added dynamically)
        self.coverage_circles = []

        # Transmitter marker
        self.tx_marker = pg.ScatterPlotItem(
            [0], [0], size=15, brush='#ff4444', symbol='star'
        )
        self.plot_widget.addItem(self.tx_marker)

        # Coverage points
        self.coverage_scatter = pg.ScatterPlotItem(size=8)
        self.plot_widget.addItem(self.coverage_scatter)

        layout.addWidget(self.plot_widget)

    def set_transmitter(self, lat: float, lon: float):
        """Set transmitter location."""
        self.tx_lat = lat
        self.tx_lon = lon
        self.tx_marker.setData([lon], [lat])

    def update_coverage(self, points: List[Dict]):
        """Update coverage points from ray trace results."""
        try:
            if not points:
                return

            # Clear old circles
            for circle in self.coverage_circles:
                self.plot_widget.removeItem(circle)
            self.coverage_circles = []

            lats = []
            lons = []
            colors = []
            distances = []  # Distance from transmitter in degrees

            for p in points:
                if p.get('reflected', False):
                    apex_lat = float(p.get('apex_lat', 0))
                    apex_lon = float(p.get('apex_lon', 0))
                    lats.append(apex_lat)
                    lons.append(apex_lon)

                    # Calculate distance from transmitter
                    dist = np.sqrt((apex_lat - self.tx_lat)**2 + (apex_lon - self.tx_lon)**2)
                    distances.append(dist)

                    # Color based on absorption
                    absorption = p.get('absorption_db', 0)
                    if absorption < 10:
                        colors.append('#44ff44')  # Good
                    elif absorption < 30:
                        colors.append('#ffff44')  # Fair
                    else:
                        colors.append('#ff4444')  # Poor

            if lats:
                brushes = [pg.mkBrush(c) for c in colors]
                self.coverage_scatter.setData(lons, lats, brush=brushes)

                # Draw circles at unique distances
                unique_distances = sorted(set([round(d, 2) for d in distances if d > 0.01]))

                # Draw a few representative circles
                if unique_distances:
                    # Take min, max, and a few in between
                    circle_radii = []
                    if len(unique_distances) >= 1:
                        circle_radii.append(min(unique_distances))
                    if len(unique_distances) >= 2:
                        circle_radii.append(max(unique_distances))
                    if len(unique_distances) >= 3:
                        circle_radii.append(unique_distances[len(unique_distances)//2])

                    for radius in sorted(set(circle_radii)):
                        # Generate circle points
                        theta = np.linspace(0, 2*np.pi, 64)
                        circle_lons = self.tx_lon + radius * np.cos(theta)
                        circle_lats = self.tx_lat + radius * np.sin(theta)

                        circle = self.plot_widget.plot(
                            circle_lons, circle_lats,
                            pen=pg.mkPen('#44ff44', width=2, style=Qt.PenStyle.DashLine)
                        )
                        self.coverage_circles.append(circle)

                # Auto-scale to fit all points plus transmitter
                all_lats = lats + [self.tx_lat]
                all_lons = lons + [self.tx_lon]

                lat_min, lat_max = min(all_lats), max(all_lats)
                lon_min, lon_max = min(all_lons), max(all_lons)

                # Add 5% padding
                lat_range = lat_max - lat_min
                lon_range = lon_max - lon_min
                lat_pad = max(0.1, lat_range * 0.05)
                lon_pad = max(0.1, lon_range * 0.05)

                self.plot_widget.setXRange(lon_min - lon_pad, lon_max + lon_pad, padding=0)
                self.plot_widget.setYRange(lat_min - lat_pad, lat_max + lat_pad, padding=0)
        except Exception:
            pass  # Skip on error


class ControlPanel(QWidget):
    """Control panel for ray tracer parameters."""

    trace_requested = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # Transmitter location
        tx_group = QGroupBox("Transmitter Location")
        tx_layout = QGridLayout(tx_group)

        tx_layout.addWidget(QLabel("Latitude:"), 0, 0)
        self.tx_lat = QDoubleSpinBox()
        self.tx_lat.setRange(-90, 90)
        self.tx_lat.setValue(40.0)
        self.tx_lat.setSuffix("°")
        tx_layout.addWidget(self.tx_lat, 0, 1)

        tx_layout.addWidget(QLabel("Longitude:"), 1, 0)
        self.tx_lon = QDoubleSpinBox()
        self.tx_lon.setRange(-180, 180)
        self.tx_lon.setValue(-105.0)
        self.tx_lon.setSuffix("°")
        tx_layout.addWidget(self.tx_lon, 1, 1)

        layout.addWidget(tx_group)

        # Frequency range
        freq_group = QGroupBox("Frequency Range")
        freq_layout = QGridLayout(freq_group)

        freq_layout.addWidget(QLabel("Min:"), 0, 0)
        self.freq_min = QDoubleSpinBox()
        self.freq_min.setRange(1, 30)
        self.freq_min.setValue(2.0)
        self.freq_min.setSuffix(" MHz")
        freq_layout.addWidget(self.freq_min, 0, 1)

        freq_layout.addWidget(QLabel("Max:"), 1, 0)
        self.freq_max = QDoubleSpinBox()
        self.freq_max.setRange(1, 30)
        self.freq_max.setValue(15.0)
        self.freq_max.setSuffix(" MHz")
        freq_layout.addWidget(self.freq_max, 1, 1)

        freq_layout.addWidget(QLabel("Step:"), 2, 0)
        self.freq_step = QDoubleSpinBox()
        self.freq_step.setRange(0.1, 5)
        self.freq_step.setValue(1.0)
        self.freq_step.setSuffix(" MHz")
        freq_layout.addWidget(self.freq_step, 2, 1)

        layout.addWidget(freq_group)

        # Ionosphere model
        iono_group = QGroupBox("Ionosphere Model")
        iono_layout = QGridLayout(iono_group)

        iono_layout.addWidget(QLabel("NmF2:"), 0, 0)
        self.nmf2 = QDoubleSpinBox()
        self.nmf2.setRange(1e10, 1e13)
        self.nmf2.setValue(5e11)
        self.nmf2.setDecimals(2)
        self.nmf2.setSingleStep(1e10)
        iono_layout.addWidget(self.nmf2, 0, 1)

        iono_layout.addWidget(QLabel("hmF2:"), 1, 0)
        self.hmf2 = QDoubleSpinBox()
        self.hmf2.setRange(150, 500)
        self.hmf2.setValue(300.0)
        self.hmf2.setSuffix(" km")
        iono_layout.addWidget(self.hmf2, 1, 1)

        layout.addWidget(iono_group)

        # Trace button
        self.trace_btn = QPushButton("Calculate Coverage")
        self.trace_btn.setStyleSheet("""
            QPushButton {
                background-color: #2a6e2a;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #3a8e3a;
            }
            QPushButton:pressed {
                background-color: #1a5e1a;
            }
        """)
        self.trace_btn.clicked.connect(self._on_trace)
        layout.addWidget(self.trace_btn)

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        layout.addStretch()

    def _on_trace(self):
        """Emit trace request with current parameters."""
        params = {
            'tx_lat': self.tx_lat.value(),
            'tx_lon': self.tx_lon.value(),
            'freq_min': self.freq_min.value(),
            'freq_max': self.freq_max.value(),
            'freq_step': self.freq_step.value(),
            'nmf2': self.nmf2.value(),
            'hmf2': self.hmf2.value(),
        }
        self.trace_requested.emit(params)

    def set_busy(self, busy: bool):
        """Set busy state."""
        self.trace_btn.setEnabled(not busy)
        self.progress.setVisible(busy)
        if busy:
            self.progress.setRange(0, 0)  # Indeterminate
