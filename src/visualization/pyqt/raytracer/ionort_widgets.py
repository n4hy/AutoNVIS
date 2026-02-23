"""
IONORT-Style Visualization Widgets

Three visualization widgets matching the IONORT paper figures:
1. AltitudeGroundRangeWidget - Ray paths in altitude vs ground range (Figures 5, 7, 9)
2. Geographic3DWidget - 3D Earth view with ray paths (Figures 7, 8 perspective)
3. SyntheticIonogramWidget - Group delay vs frequency ionogram (Figures 11-16)

Reference: IONORT paper (remotesensing-15-05111-v2.pdf)

All widgets use:
- PyQtGraph for 2D plotting
- PyQtGraph.opengl for 3D visualization
- Dark theme (#1e1e1e background)
- Rainbow frequency coloring (red=low, blue=high)
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QGridLayout,
    QTableWidget, QTableWidgetItem, QHeaderView, QSplitter, QGroupBox,
    QScrollArea, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPen
import pyqtgraph as pg
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# Try to import OpenGL widgets
try:
    import pyqtgraph.opengl as gl
    HAS_OPENGL = True
except ImportError:
    HAS_OPENGL = False


# Earth radius for calculations
EARTH_RADIUS_KM = 6371.0


def frequency_to_color(freq_mhz: float, freq_min: float = 2.0, freq_max: float = 30.0) -> str:
    """
    Map frequency to rainbow color (red -> orange -> yellow -> green -> blue).

    Following IONORT convention: low frequencies are warm colors, high are cool.

    Args:
        freq_mhz: Frequency in MHz
        freq_min: Minimum frequency for color scale
        freq_max: Maximum frequency for color scale

    Returns:
        Hex color string like '#ff8844'
    """
    # Normalize to 0-1
    norm = (freq_mhz - freq_min) / (freq_max - freq_min)
    norm = max(0.0, min(1.0, norm))

    # Rainbow: red (0) -> yellow (0.25) -> green (0.5) -> cyan (0.75) -> blue (1.0)
    if norm < 0.25:
        # Red to Yellow
        t = norm / 0.25
        r, g, b = 255, int(255 * t), 0
    elif norm < 0.5:
        # Yellow to Green
        t = (norm - 0.25) / 0.25
        r, g, b = int(255 * (1 - t)), 255, 0
    elif norm < 0.75:
        # Green to Cyan
        t = (norm - 0.5) / 0.25
        r, g, b = 0, 255, int(255 * t)
    else:
        # Cyan to Blue
        t = (norm - 0.75) / 0.25
        r, g, b = 0, int(255 * (1 - t)), 255

    return f'#{r:02x}{g:02x}{b:02x}'


@dataclass
class IonogramTrace:
    """Data for ionogram trace (O or X mode)."""
    frequencies: List[float]  # MHz
    group_delays: List[float]  # ms
    mode: str  # 'O' or 'X'
    elevations: Optional[List[float]] = None  # degrees


class AltitudeGroundRangeWidget(QWidget):
    """
    IONORT-style ray path display in altitude vs ground range cross-section.

    Features:
    - Ionospheric layer shading (D, E, F1, F2 regions)
    - Frequency-based rainbow coloring
    - Solid lines for reflected rays, dashed for escaped
    - O-mode thicker than X-mode
    - Earth curvature for paths > 500 km

    Like IONORT Figures 5, 7, 9.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ray_curves = []
        self.layer_items = []
        self.freq_min = 2.0
        self.freq_max = 30.0
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Header with title and info
        header = QHBoxLayout()
        title = QLabel("Altitude vs Ground Range")
        title.setStyleSheet("font-weight: bold; font-size: 14px; color: #ffffff;")
        header.addWidget(title)
        header.addStretch()

        self.info_label = QLabel("")
        self.info_label.setStyleSheet("color: #888888;")
        header.addWidget(self.info_label)
        layout.addLayout(header)

        # Main plot
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#1e1e1e')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.2)
        self.plot_widget.setLabel('left', 'Altitude', units='km')
        self.plot_widget.setLabel('bottom', 'Ground Range', units='km')

        # Set axis styles
        for axis in ['left', 'bottom']:
            ax = self.plot_widget.getAxis(axis)
            ax.setTextPen(pg.mkPen('#ffffff'))
            ax.setPen(pg.mkPen('#444444'))

        # Add ionospheric layer shading
        self._add_layer_shading()

        # Earth surface
        earth_x = np.linspace(-3000, 3000, 200)
        earth_y = np.zeros_like(earth_x)
        self.earth_curve = self.plot_widget.plot(
            earth_x, earth_y,
            pen=pg.mkPen('#444444', width=3),
            name='Ground'
        )

        layout.addWidget(self.plot_widget)

        # Legend
        self._add_legend(layout)

    def _add_layer_shading(self):
        """Add semi-transparent shading for ionospheric layers."""
        layers = [
            (60, 90, '#442200', 'D Region'),    # D layer: 60-90 km
            (90, 150, '#443300', 'E Region'),   # E layer: 90-150 km
            (150, 220, '#443344', 'F1 Region'), # F1 layer: 150-220 km
            (220, 450, '#334444', 'F2 Region'), # F2 layer: 220-450 km
        ]

        for y_min, y_max, color, name in layers:
            # Create LinearRegionItem for horizontal band
            region = pg.LinearRegionItem(
                values=[y_min, y_max],
                orientation='horizontal',
                brush=pg.mkBrush(color + '40'),  # 40 = 25% alpha
                pen=pg.mkPen(None),
                movable=False
            )
            self.plot_widget.addItem(region)
            self.layer_items.append(region)

            # Add label
            label = pg.TextItem(name, color='#888888', anchor=(0, 0.5))
            label.setPos(-2800, (y_min + y_max) / 2)
            self.plot_widget.addItem(label)
            self.layer_items.append(label)

    def _add_legend(self, layout):
        """Add frequency color legend."""
        legend_layout = QHBoxLayout()
        legend_layout.addStretch()

        # Color bar simulation with labels
        legend_layout.addWidget(QLabel("Frequency:"))

        for freq, color in [(2, '#ff0000'), (8, '#ffff00'), (15, '#00ff00'), (22, '#00ffff'), (30, '#0000ff')]:
            box = QLabel(f" {freq} MHz ")
            box.setStyleSheet(f"background-color: {color}; color: black; padding: 2px; border-radius: 3px;")
            legend_layout.addWidget(box)

        legend_layout.addStretch()
        layout.addLayout(legend_layout)

    def clear(self):
        """Clear all ray paths."""
        for curve in self.ray_curves:
            self.plot_widget.removeItem(curve)
        self.ray_curves = []

    def set_frequency_range(self, freq_min: float, freq_max: float):
        """Set frequency range for color mapping."""
        self.freq_min = freq_min
        self.freq_max = freq_max

    def add_ray_path(
        self,
        ground_ranges: List[float],
        altitudes: List[float],
        frequency_mhz: float,
        is_reflected: bool = True,
        is_o_mode: bool = True,
    ):
        """
        Add a single ray path to the display.

        Args:
            ground_ranges: List of ground range values (km)
            altitudes: List of altitude values (km)
            frequency_mhz: Frequency for color coding
            is_reflected: True if ray reflects (solid line), False if escapes (dashed)
            is_o_mode: True for O-mode (thick line), False for X-mode (thin)
        """
        if len(ground_ranges) < 2 or len(altitudes) < 2:
            return

        color = frequency_to_color(frequency_mhz, self.freq_min, self.freq_max)
        width = 2.5 if is_o_mode else 1.5

        pen = pg.mkPen(color, width=width)
        if not is_reflected:
            pen.setStyle(Qt.PenStyle.DashLine)

        curve = self.plot_widget.plot(
            ground_ranges, altitudes,
            pen=pen
        )
        self.ray_curves.append(curve)

    def add_ray_path_from_positions(
        self,
        positions: List[Tuple[float, float, float]],
        tx_lat: float,
        tx_lon: float,
        frequency_mhz: float,
        is_reflected: bool = True,
        is_o_mode: bool = True,
    ):
        """
        Add ray path from list of (lat, lon, alt) positions.

        Converts geodetic positions to ground range.

        Args:
            positions: List of (lat, lon, alt) tuples
            tx_lat, tx_lon: Transmitter position for range calculation
            frequency_mhz: Frequency for color coding
            is_reflected: Reflection status
            is_o_mode: Mode indicator
        """
        if len(positions) < 2:
            return

        ground_ranges = []
        altitudes = []

        for lat, lon, alt in positions:
            # Calculate ground range using haversine
            dlat = np.radians(lat - tx_lat)
            dlon = np.radians(lon - tx_lon)
            lat1_r = np.radians(tx_lat)
            lat2_r = np.radians(lat)

            a = np.sin(dlat/2)**2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon/2)**2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
            dist = EARTH_RADIUS_KM * c

            # Sign based on predominant direction
            if dlat < 0 or (dlat == 0 and dlon < 0):
                dist = -dist

            ground_ranges.append(dist)
            altitudes.append(alt)

        self.add_ray_path(ground_ranges, altitudes, frequency_mhz, is_reflected, is_o_mode)

    def auto_scale(self, max_range: float = None, max_alt: float = None):
        """Auto-scale plot axes to fit ray paths."""
        if max_range is None:
            max_range = 500
        if max_alt is None:
            max_alt = 500

        self.plot_widget.setXRange(-max_range * 1.1, max_range * 1.1, padding=0)
        self.plot_widget.setYRange(0, max_alt * 1.1, padding=0)

    def set_info(self, text: str):
        """Set info label text."""
        self.info_label.setText(text)


class Geographic3DWidget(QWidget):
    """
    3D geographic visualization of ray paths on Earth sphere.

    Features:
    - Earth sphere mesh with lat/lon grid
    - Ray paths as 3D lines colored by frequency
    - Tx marker (red star), Rx marker (green diamond)
    - Interactive rotation and zoom

    Like IONORT Figures 7, 8 perspective.

    Requires PyOpenGL.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ray_items = []
        self.marker_items = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Header
        header = QHBoxLayout()
        title = QLabel("3D Geographic View")
        title.setStyleSheet("font-weight: bold; font-size: 14px; color: #ffffff;")
        header.addWidget(title)
        header.addStretch()

        self.info_label = QLabel("")
        self.info_label.setStyleSheet("color: #888888;")
        header.addWidget(self.info_label)
        layout.addLayout(header)

        if not HAS_OPENGL:
            # Fallback if OpenGL not available
            fallback = QLabel("3D visualization requires PyOpenGL.\nInstall with: pip install PyOpenGL")
            fallback.setStyleSheet("color: #ff8888; font-size: 14px;")
            fallback.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(fallback)
            self.gl_widget = None
            return

        # Create OpenGL widget
        self.gl_widget = gl.GLViewWidget()
        self.gl_widget.setBackgroundColor('#1e1e1e')
        self.gl_widget.setCameraPosition(distance=20000, elevation=30, azimuth=45)
        layout.addWidget(self.gl_widget)

        # Add Earth sphere
        self._add_earth()

        # Add lat/lon grid
        self._add_grid()

    def _add_earth(self):
        """Add Earth sphere mesh."""
        if not self.gl_widget:
            return

        # Create sphere mesh data
        rows = 30
        cols = 60

        verts = []
        faces = []
        colors = []

        for i in range(rows + 1):
            lat = -90 + 180 * i / rows
            for j in range(cols):
                lon = -180 + 360 * j / cols

                # Convert to Cartesian
                lat_r = np.radians(lat)
                lon_r = np.radians(lon)
                x = EARTH_RADIUS_KM * np.cos(lat_r) * np.cos(lon_r)
                y = EARTH_RADIUS_KM * np.cos(lat_r) * np.sin(lon_r)
                z = EARTH_RADIUS_KM * np.sin(lat_r)

                verts.append([x, y, z])

                # Blue-green for Earth
                colors.append([0.1, 0.3, 0.5, 0.8])

        # Create faces
        for i in range(rows):
            for j in range(cols):
                p1 = i * cols + j
                p2 = i * cols + (j + 1) % cols
                p3 = (i + 1) * cols + (j + 1) % cols
                p4 = (i + 1) * cols + j

                faces.append([p1, p2, p4])
                faces.append([p2, p3, p4])

        verts = np.array(verts)
        faces = np.array(faces)
        colors = np.array(colors)

        mesh = gl.GLMeshItem(
            vertexes=verts,
            faces=faces,
            faceColors=colors[faces[:, 0]],
            smooth=True,
            drawEdges=False
        )
        self.gl_widget.addItem(mesh)
        self.earth_mesh = mesh

    def _add_grid(self):
        """Add latitude/longitude grid lines."""
        if not self.gl_widget:
            return

        # Latitude lines every 30 degrees
        for lat in range(-60, 90, 30):
            pts = []
            for lon in np.linspace(-180, 180, 73):
                lat_r = np.radians(lat)
                lon_r = np.radians(lon)
                r = EARTH_RADIUS_KM + 5  # Slightly above surface
                x = r * np.cos(lat_r) * np.cos(lon_r)
                y = r * np.cos(lat_r) * np.sin(lon_r)
                z = r * np.sin(lat_r)
                pts.append([x, y, z])

            pts = np.array(pts)
            line = gl.GLLinePlotItem(pos=pts, color=(0.5, 0.5, 0.5, 0.5), width=1)
            self.gl_widget.addItem(line)

        # Longitude lines every 30 degrees
        for lon in range(-180, 180, 30):
            pts = []
            for lat in np.linspace(-90, 90, 37):
                lat_r = np.radians(lat)
                lon_r = np.radians(lon)
                r = EARTH_RADIUS_KM + 5
                x = r * np.cos(lat_r) * np.cos(lon_r)
                y = r * np.cos(lat_r) * np.sin(lon_r)
                z = r * np.sin(lat_r)
                pts.append([x, y, z])

            pts = np.array(pts)
            line = gl.GLLinePlotItem(pos=pts, color=(0.5, 0.5, 0.5, 0.5), width=1)
            self.gl_widget.addItem(line)

    def clear(self):
        """Clear all ray paths and markers."""
        if not self.gl_widget:
            return

        for item in self.ray_items + self.marker_items:
            self.gl_widget.removeItem(item)
        self.ray_items = []
        self.marker_items = []

    def add_ray_path(
        self,
        positions: List[Tuple[float, float, float]],
        frequency_mhz: float,
        freq_min: float = 2.0,
        freq_max: float = 30.0,
    ):
        """
        Add ray path as 3D line.

        Args:
            positions: List of (lat, lon, alt) tuples
            frequency_mhz: Frequency for coloring
            freq_min, freq_max: Frequency range for color mapping
        """
        if not self.gl_widget or len(positions) < 2:
            return

        # Convert positions to ECEF
        pts = []
        for lat, lon, alt in positions:
            lat_r = np.radians(lat)
            lon_r = np.radians(lon)
            r = EARTH_RADIUS_KM + alt
            x = r * np.cos(lat_r) * np.cos(lon_r)
            y = r * np.cos(lat_r) * np.sin(lon_r)
            z = r * np.sin(lat_r)
            pts.append([x, y, z])

        pts = np.array(pts)

        # Get color
        color_hex = frequency_to_color(frequency_mhz, freq_min, freq_max)
        r = int(color_hex[1:3], 16) / 255
        g = int(color_hex[3:5], 16) / 255
        b = int(color_hex[5:7], 16) / 255

        line = gl.GLLinePlotItem(pos=pts, color=(r, g, b, 1.0), width=2)
        self.gl_widget.addItem(line)
        self.ray_items.append(line)

    def add_marker(
        self,
        lat: float,
        lon: float,
        alt: float = 0,
        color: str = '#ff0000',
        size: float = 100,
    ):
        """
        Add station marker.

        Args:
            lat, lon, alt: Position
            color: Marker color
            size: Marker size
        """
        if not self.gl_widget:
            return

        lat_r = np.radians(lat)
        lon_r = np.radians(lon)
        r = EARTH_RADIUS_KM + alt + 10  # Above surface
        x = r * np.cos(lat_r) * np.cos(lon_r)
        y = r * np.cos(lat_r) * np.sin(lon_r)
        z = r * np.sin(lat_r)

        # Parse color
        r_c = int(color[1:3], 16) / 255
        g_c = int(color[3:5], 16) / 255
        b_c = int(color[5:7], 16) / 255

        scatter = gl.GLScatterPlotItem(
            pos=np.array([[x, y, z]]),
            color=(r_c, g_c, b_c, 1.0),
            size=size
        )
        self.gl_widget.addItem(scatter)
        self.marker_items.append(scatter)

    def set_camera(self, distance: float = 20000, elevation: float = 30, azimuth: float = 45):
        """Set camera position."""
        if self.gl_widget:
            self.gl_widget.setCameraPosition(distance=distance, elevation=elevation, azimuth=azimuth)

    def focus_on(self, lat: float, lon: float):
        """Focus camera on a lat/lon position."""
        if not self.gl_widget:
            return

        # Calculate camera azimuth to look at this point
        azimuth = lon + 180
        self.gl_widget.setCameraPosition(distance=15000, elevation=30, azimuth=azimuth)


class SyntheticIonogramWidget(QWidget):
    """
    Synthetic oblique ionogram display.

    Shows group delay vs frequency with O-mode and X-mode traces,
    plus a table of winner triplets.

    Features:
    - O-mode trace (solid circles)
    - X-mode trace (X markers, offset by gyrofrequency)
    - MUF vertical marker
    - Winner triplets table with scrolling

    Like IONORT Figures 11-16.
    """

    # Signal emitted when a triplet is selected
    triplet_selected = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.o_trace_items = []
        self.x_trace_items = []
        self.markers = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Left side: ionogram plot
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        plot_layout.setContentsMargins(0, 0, 0, 0)

        # Header
        header = QHBoxLayout()
        title = QLabel("Synthetic Oblique Ionogram")
        title.setStyleSheet("font-weight: bold; font-size: 14px; color: #ffffff;")
        header.addWidget(title)
        header.addStretch()

        self.info_label = QLabel("")
        self.info_label.setStyleSheet("color: #888888;")
        header.addWidget(self.info_label)
        plot_layout.addLayout(header)

        # Main plot
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#1e1e1e')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel('left', 'Group Delay', units='ms')
        self.plot_widget.setLabel('bottom', 'Frequency', units='MHz')

        # Set axis styles
        for axis in ['left', 'bottom']:
            ax = self.plot_widget.getAxis(axis)
            ax.setTextPen(pg.mkPen('#ffffff'))
            ax.setPen(pg.mkPen('#444444'))

        # MUF line
        self.muf_line = pg.InfiniteLine(
            pos=15,
            angle=90,
            pen=pg.mkPen('#ff4444', width=2, style=Qt.PenStyle.DashLine),
            label='MUF',
            labelOpts={'color': '#ff4444', 'position': 0.9}
        )
        self.muf_line.setVisible(False)
        self.plot_widget.addItem(self.muf_line)

        # LUF line
        self.luf_line = pg.InfiniteLine(
            pos=3,
            angle=90,
            pen=pg.mkPen('#ffaa00', width=2, style=Qt.PenStyle.DashLine),
            label='LUF',
            labelOpts={'color': '#ffaa00', 'position': 0.1}
        )
        self.luf_line.setVisible(False)
        self.plot_widget.addItem(self.luf_line)

        plot_layout.addWidget(self.plot_widget)

        # Legend
        legend_layout = QHBoxLayout()
        legend_layout.addStretch()

        o_label = QLabel(" O-mode ")
        o_label.setStyleSheet("background-color: #4488ff; color: white; padding: 3px; border-radius: 3px;")
        legend_layout.addWidget(o_label)

        x_label = QLabel(" X-mode ")
        x_label.setStyleSheet("background-color: #ff88cc; color: white; padding: 3px; border-radius: 3px;")
        legend_layout.addWidget(x_label)

        legend_layout.addStretch()
        plot_layout.addLayout(legend_layout)

        layout.addWidget(plot_widget, stretch=3)

        # Right side: winner triplets table
        table_widget = QWidget()
        table_layout = QVBoxLayout(table_widget)
        table_layout.setContentsMargins(5, 0, 0, 0)

        table_title = QLabel("Winner Triplets")
        table_title.setStyleSheet("font-weight: bold; font-size: 14px; color: #ffffff;")
        table_layout.addWidget(table_title)

        self.triplet_table = QTableWidget()
        self.triplet_table.setColumnCount(5)
        self.triplet_table.setHorizontalHeaderLabels(['f (MHz)', 'El (°)', 'Az (°)', 'Delay (ms)', 'Mode'])
        self.triplet_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.triplet_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.triplet_table.setStyleSheet("""
            QTableWidget {
                background-color: #2a2a2a;
                color: #ffffff;
                gridline-color: #444444;
            }
            QTableWidget::item:selected {
                background-color: #445566;
            }
            QHeaderView::section {
                background-color: #333333;
                color: #ffffff;
                padding: 5px;
            }
        """)
        self.triplet_table.cellClicked.connect(self._on_triplet_clicked)
        table_layout.addWidget(self.triplet_table)

        # Stats
        self.stats_label = QLabel("No data")
        self.stats_label.setStyleSheet("color: #888888; font-size: 12px;")
        table_layout.addWidget(self.stats_label)

        layout.addWidget(table_widget, stretch=1)

    def clear(self):
        """Clear all traces and table."""
        for item in self.o_trace_items + self.x_trace_items + self.markers:
            self.plot_widget.removeItem(item)
        self.o_trace_items = []
        self.x_trace_items = []
        self.markers = []
        self.triplet_table.setRowCount(0)
        self.muf_line.setVisible(False)
        self.luf_line.setVisible(False)

    def set_o_mode_trace(self, frequencies: List[float], group_delays: List[float]):
        """
        Set O-mode ionogram trace.

        Args:
            frequencies: List of frequencies (MHz)
            group_delays: List of group delays (ms)
        """
        # Clear existing O-mode items
        for item in self.o_trace_items:
            self.plot_widget.removeItem(item)
        self.o_trace_items = []

        if len(frequencies) == 0:
            return

        # Plot as scatter with connected lines
        scatter = pg.ScatterPlotItem(
            x=frequencies,
            y=group_delays,
            symbol='o',
            size=10,
            brush=pg.mkBrush('#4488ff'),
            pen=pg.mkPen('#ffffff', width=1)
        )
        self.plot_widget.addItem(scatter)
        self.o_trace_items.append(scatter)

        # Connect with line
        if len(frequencies) > 1:
            # Sort by frequency for proper line
            sorted_pairs = sorted(zip(frequencies, group_delays))
            freqs_sorted = [p[0] for p in sorted_pairs]
            delays_sorted = [p[1] for p in sorted_pairs]

            line = self.plot_widget.plot(
                freqs_sorted, delays_sorted,
                pen=pg.mkPen('#4488ff', width=2)
            )
            self.o_trace_items.append(line)

    def set_x_mode_trace(self, frequencies: List[float], group_delays: List[float]):
        """
        Set X-mode ionogram trace.

        Args:
            frequencies: List of frequencies (MHz)
            group_delays: List of group delays (ms)
        """
        # Clear existing X-mode items
        for item in self.x_trace_items:
            self.plot_widget.removeItem(item)
        self.x_trace_items = []

        if len(frequencies) == 0:
            return

        # Plot as scatter with X markers
        scatter = pg.ScatterPlotItem(
            x=frequencies,
            y=group_delays,
            symbol='x',
            size=10,
            brush=pg.mkBrush('#ff88cc'),
            pen=pg.mkPen('#ff88cc', width=2)
        )
        self.plot_widget.addItem(scatter)
        self.x_trace_items.append(scatter)

        # Connect with dashed line
        if len(frequencies) > 1:
            sorted_pairs = sorted(zip(frequencies, group_delays))
            freqs_sorted = [p[0] for p in sorted_pairs]
            delays_sorted = [p[1] for p in sorted_pairs]

            line = self.plot_widget.plot(
                freqs_sorted, delays_sorted,
                pen=pg.mkPen('#ff88cc', width=1, style=Qt.PenStyle.DashLine)
            )
            self.x_trace_items.append(line)

    def set_muf(self, muf_mhz: float):
        """Set MUF vertical line."""
        self.muf_line.setValue(muf_mhz)
        self.muf_line.setVisible(True)

    def set_luf(self, luf_mhz: float):
        """Set LUF vertical line."""
        self.luf_line.setValue(luf_mhz)
        self.luf_line.setVisible(True)

    def set_winner_triplets(self, triplets: List[Dict]):
        """
        Populate winner triplets table.

        Args:
            triplets: List of dicts with keys:
                     'frequency_mhz', 'elevation_deg', 'azimuth_deg',
                     'group_delay_ms', 'mode'
        """
        self.triplet_table.setRowCount(len(triplets))

        o_count = 0
        x_count = 0

        for i, t in enumerate(triplets):
            freq = t.get('frequency_mhz', 0)
            elev = t.get('elevation_deg', 0)
            azim = t.get('azimuth_deg', 0)
            delay = t.get('group_delay_ms', 0)
            mode = t.get('mode', 'O')

            self.triplet_table.setItem(i, 0, QTableWidgetItem(f"{freq:.2f}"))
            self.triplet_table.setItem(i, 1, QTableWidgetItem(f"{elev:.1f}"))
            self.triplet_table.setItem(i, 2, QTableWidgetItem(f"{azim:.1f}"))
            self.triplet_table.setItem(i, 3, QTableWidgetItem(f"{delay:.2f}"))
            self.triplet_table.setItem(i, 4, QTableWidgetItem(mode))

            # Color row by mode
            color = QColor('#334466') if mode == 'O' else QColor('#443344')
            for col in range(5):
                self.triplet_table.item(i, col).setBackground(color)

            if mode == 'O':
                o_count += 1
            else:
                x_count += 1

        self.stats_label.setText(f"Total: {len(triplets)} (O: {o_count}, X: {x_count})")

    def auto_scale(self):
        """Auto-scale plot to fit data."""
        self.plot_widget.autoRange()

    def set_info(self, text: str):
        """Set info label."""
        self.info_label.setText(text)

    def _on_triplet_clicked(self, row: int, col: int):
        """Handle triplet table row click."""
        self.triplet_selected.emit(row)


class IONORTVisualizationPanel(QWidget):
    """
    Combined panel with all three IONORT-style visualizations.

    Contains:
    - AltitudeGroundRangeWidget (top left)
    - Geographic3DWidget (top right)
    - SyntheticIonogramWidget (bottom)

    Use this widget as a drop-in replacement for simpler visualization.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create splitter for resizable panels
        main_splitter = QSplitter(Qt.Orientation.Vertical)

        # Top row: altitude and 3D view
        top_splitter = QSplitter(Qt.Orientation.Horizontal)

        self.altitude_widget = AltitudeGroundRangeWidget()
        top_splitter.addWidget(self.altitude_widget)

        self.geographic_widget = Geographic3DWidget()
        top_splitter.addWidget(self.geographic_widget)

        main_splitter.addWidget(top_splitter)

        # Bottom: ionogram
        self.ionogram_widget = SyntheticIonogramWidget()
        main_splitter.addWidget(self.ionogram_widget)

        # Set initial sizes (60% top, 40% bottom)
        main_splitter.setSizes([600, 400])
        top_splitter.setSizes([500, 500])

        layout.addWidget(main_splitter)

    def clear(self):
        """Clear all visualizations."""
        self.altitude_widget.clear()
        self.geographic_widget.clear()
        self.ionogram_widget.clear()

    def update_from_homing_result(self, result):
        """
        Update all visualizations from a HomingResult.

        Args:
            result: HomingResult from homing_algorithm
        """
        self.clear()

        if not result or not result.winner_triplets:
            return

        # Get Tx/Rx positions
        tx_lat, tx_lon, tx_alt = result.tx_position
        rx_lat, rx_lon, rx_alt = result.rx_position

        # Add markers to 3D view
        self.geographic_widget.add_marker(tx_lat, tx_lon, tx_alt, '#ff4444', 150)  # Tx red
        self.geographic_widget.add_marker(rx_lat, rx_lon, rx_alt, '#44ff44', 150)  # Rx green

        # Focus camera on path midpoint
        mid_lat = (tx_lat + rx_lat) / 2
        mid_lon = (tx_lon + rx_lon) / 2
        self.geographic_widget.focus_on(mid_lat, mid_lon)

        # Set frequency range for coloring
        frequencies = [w.frequency_mhz for w in result.winner_triplets]
        freq_min = min(frequencies)
        freq_max = max(frequencies)
        self.altitude_widget.set_frequency_range(freq_min, freq_max)

        # Collect traces for ionogram
        o_freqs = []
        o_delays = []
        x_freqs = []
        x_delays = []

        # Process winner triplets
        triplet_dicts = []
        for w in result.winner_triplets:
            # Add to ionogram data
            if w.mode.value == 'O':
                o_freqs.append(w.frequency_mhz)
                o_delays.append(w.group_delay_ms)
            else:
                x_freqs.append(w.frequency_mhz)
                x_delays.append(w.group_delay_ms)

            # Add ray path if available
            if w.ray_path and w.ray_path.states:
                positions = [s.lat_lon_alt() for s in w.ray_path.states]
                is_reflected = w.ray_path.termination.value == 'ground'
                is_o_mode = w.mode.value == 'O'

                # Add to altitude widget
                self.altitude_widget.add_ray_path_from_positions(
                    positions, tx_lat, tx_lon,
                    w.frequency_mhz, is_reflected, is_o_mode
                )

                # Add to 3D widget
                self.geographic_widget.add_ray_path(
                    positions, w.frequency_mhz, freq_min, freq_max
                )

            # Table entry
            triplet_dicts.append({
                'frequency_mhz': w.frequency_mhz,
                'elevation_deg': w.elevation_deg,
                'azimuth_deg': w.azimuth_deg,
                'group_delay_ms': w.group_delay_ms,
                'mode': w.mode.value,
            })

        # Update ionogram
        self.ionogram_widget.set_o_mode_trace(o_freqs, o_delays)
        self.ionogram_widget.set_x_mode_trace(x_freqs, x_delays)
        self.ionogram_widget.set_muf(result.muf)
        self.ionogram_widget.set_luf(result.luf)
        self.ionogram_widget.set_winner_triplets(triplet_dicts)
        self.ionogram_widget.auto_scale()

        # Auto-scale altitude widget
        self.altitude_widget.auto_scale(result.great_circle_range_km / 2, 500)

        # Set info
        self.altitude_widget.set_info(f"Range: {result.great_circle_range_km:.0f} km")
        self.ionogram_widget.set_info(f"MUF: {result.muf:.1f} MHz, LUF: {result.luf:.1f} MHz")
