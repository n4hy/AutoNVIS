"""
TEC Map Widget

Global TEC visualization using pyqtgraph ImageItem.
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox
from PyQt6.QtCore import pyqtSignal
import pyqtgraph as pg
import numpy as np
from typing import Optional


class TECMapWidget(QWidget):
    """
    Global TEC map visualization using pyqtgraph.

    Features:
    - Color-coded TEC values (0-100 TECU range)
    - Click to select point
    - Anomaly overlay option
    - Quality flag overlay
    - Political boundaries overlay
    """

    point_selected = pyqtSignal(float, float)  # lat, lon

    def __init__(self, parent=None):
        """Initialize TEC map widget."""
        super().__init__(parent)

        self.current_layer = 'tec'  # tec, anomaly, hmF2, NmF2
        self.last_grid_data = None
        self.show_political = False
        self.political_lines = []  # Store political boundary plot items

        # Color scaling mode: 'fixed', 'auto', 'percentile'
        self.scale_mode = 'percentile'  # Default to percentile for best visibility

        self._setup_ui()
        self._create_political_boundaries()

    def _setup_ui(self):
        """Create and arrange UI elements."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Control bar
        control_layout = QHBoxLayout()

        control_layout.addWidget(QLabel("Layer:"))

        self.layer_combo = QComboBox()
        self.layer_combo.addItems(['TEC', 'Anomaly', 'hmF2', 'NmF2'])
        self.layer_combo.currentTextChanged.connect(self._on_layer_changed)
        control_layout.addWidget(self.layer_combo)

        control_layout.addWidget(QLabel("  Scale:"))

        self.scale_combo = QComboBox()
        self.scale_combo.addItems(['Percentile', 'Auto', 'Fixed'])
        self.scale_combo.setToolTip("Percentile: 5th-95th (best detail)\nAuto: actual min-max\nFixed: preset range")
        self.scale_combo.currentTextChanged.connect(self._on_scale_changed)
        control_layout.addWidget(self.scale_combo)

        control_layout.addStretch()

        self.info_label = QLabel("No data")
        control_layout.addWidget(self.info_label)

        layout.addLayout(control_layout)

        # Create plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setAspectLocked(False)
        self.plot_widget.setLabel('bottom', 'Longitude', units='deg')
        self.plot_widget.setLabel('left', 'Latitude', units='deg')
        self.plot_widget.setTitle('Global TEC Map')

        # Set axis ranges
        self.plot_widget.setXRange(-180, 180)
        self.plot_widget.setYRange(-90, 90)

        # TEC image
        self.tec_image = pg.ImageItem()
        self.plot_widget.addItem(self.tec_image)

        # Default colormap (plasma-like for TEC)
        self.colormap = pg.colormap.get('plasma')
        self.tec_image.setColorMap(self.colormap)

        # Colorbar
        self.colorbar = pg.ColorBarItem(
            values=(0, 100),
            colorMap=self.colormap,
            label='TEC (TECU)'
        )
        self.colorbar.setImageItem(self.tec_image)

        # Click handling
        self.plot_widget.scene().sigMouseClicked.connect(self._on_click)

        # Crosshair for selected point
        self.crosshair_v = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('c', width=1, style=pg.QtCore.Qt.PenStyle.DashLine))
        self.crosshair_h = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('c', width=1, style=pg.QtCore.Qt.PenStyle.DashLine))
        self.crosshair_v.setVisible(False)
        self.crosshair_h.setVisible(False)
        self.plot_widget.addItem(self.crosshair_v, ignoreBounds=True)
        self.plot_widget.addItem(self.crosshair_h, ignoreBounds=True)

        layout.addWidget(self.plot_widget)

    def update_map(self, grid_data: dict):
        """
        Update the TEC map display.

        Args:
            grid_data: Dictionary with lat, lon, tec, anomaly, hmF2, NmF2 arrays
        """
        self.last_grid_data = grid_data
        self._refresh_display()

    def _refresh_display(self):
        """Refresh display with current layer and data."""
        if self.last_grid_data is None:
            return

        grid_data = self.last_grid_data

        lat = grid_data.get('lat')
        lon = grid_data.get('lon')

        if lat is None or lon is None:
            return

        # Get data for current layer
        layer_key = self.current_layer.lower()
        if layer_key == 'hmf2':
            layer_key = 'hmF2'
        elif layer_key == 'nmf2':
            layer_key = 'NmF2'

        data = grid_data.get(layer_key)

        if data is None or len(data) == 0:
            return

        # Convert to numpy array if needed
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        if not isinstance(lat, np.ndarray):
            lat = np.array(lat)
        if not isinstance(lon, np.ndarray):
            lon = np.array(lon)

        # Ensure 2D
        if data.ndim == 1:
            # Try to reshape if we know dimensions
            n_lat = len(lat)
            n_lon = len(lon)
            if len(data) == n_lat * n_lon:
                data = data.reshape(n_lat, n_lon)
            else:
                return

        # Update image (transpose for correct orientation)
        self.tec_image.setImage(data.T)

        # Set position and scale to match geographic coordinates
        x_min = float(lon.min())
        y_min = float(lat.min())
        x_range = float(lon.max() - lon.min())
        y_range = float(lat.max() - lat.min())

        # Set transform to map image pixels to coordinates
        self.tec_image.setRect(x_min, y_min, x_range, y_range)

        # Update colorbar based on layer and data
        self._update_colorbar(data)

        # Update info label
        timestamp = grid_data.get('timestamp', 'Unknown')
        valid_data = data[~np.isnan(data)]
        if len(valid_data) > 0:
            self.info_label.setText(
                f"{timestamp} | Mean: {np.mean(valid_data):.1f} | Max: {np.max(valid_data):.1f}"
            )
        else:
            self.info_label.setText(f"{timestamp} | No valid data")

    def _update_colorbar(self, data=None):
        """Update colorbar for current layer and scale mode."""
        layer = self.current_layer.lower()

        # Fixed ranges (defaults)
        fixed_ranges = {
            'tec': (0, 100),
            'anomaly': (-20, 20),
            'hmf2': (200, 500),
            'nmf2': (0, 1e12),
        }

        # Colormaps
        colormaps = {
            'tec': 'plasma',
            'anomaly': 'CET-D1',
            'hmf2': 'viridis',
            'nmf2': 'inferno',
        }

        # Normalize layer name
        layer_key = layer
        if layer_key == 'hmf2':
            layer_key = 'hmf2'
        elif layer_key == 'nmf2':
            layer_key = 'nmf2'

        # Get colormap
        cmap_name = colormaps.get(layer_key, 'plasma')
        self.colormap = pg.colormap.get(cmap_name)
        self.tec_image.setColorMap(self.colormap)

        # Calculate range based on scale mode
        if data is not None and len(data) > 0:
            valid_data = data[~np.isnan(data)]
            if len(valid_data) > 0:
                if self.scale_mode == 'auto':
                    # Full data range
                    vmin = float(np.min(valid_data))
                    vmax = float(np.max(valid_data))
                elif self.scale_mode == 'percentile':
                    # 5th to 95th percentile (avoids outlier saturation)
                    vmin = float(np.percentile(valid_data, 5))
                    vmax = float(np.percentile(valid_data, 95))
                else:
                    # Fixed range
                    vmin, vmax = fixed_ranges.get(layer_key, (0, 100))

                # Ensure valid range
                if vmax <= vmin:
                    vmax = vmin + 1

                self.colorbar.setLevels((vmin, vmax))
                self.tec_image.setLevels((vmin, vmax))
                return

        # Fallback to fixed range
        vmin, vmax = fixed_ranges.get(layer_key, (0, 100))
        self.colorbar.setLevels((vmin, vmax))

    def _on_layer_changed(self, layer_name: str):
        """Handle layer selection change."""
        self.current_layer = layer_name.lower()
        self._refresh_display()

    def _on_scale_changed(self, scale_name: str):
        """Handle scale mode change."""
        self.scale_mode = scale_name.lower()
        self._refresh_display()

    def _on_click(self, event):
        """Handle mouse click to select point."""
        # Check if click is within plot area
        if not self.plot_widget.sceneBoundingRect().contains(event.scenePos()):
            return

        # Map scene position to view coordinates
        mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(event.scenePos())
        lon = mouse_point.x()
        lat = mouse_point.y()

        # Validate coordinates
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            # Update crosshair
            self.crosshair_v.setPos(lon)
            self.crosshair_h.setPos(lat)
            self.crosshair_v.setVisible(True)
            self.crosshair_h.setVisible(True)

            # Emit signal
            self.point_selected.emit(lat, lon)

    def set_selected_point(self, lat: float, lon: float):
        """
        Programmatically set the selected point.

        Args:
            lat: Latitude
            lon: Longitude
        """
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            self.crosshair_v.setPos(lon)
            self.crosshair_h.setPos(lat)
            self.crosshair_v.setVisible(True)
            self.crosshair_h.setVisible(True)

    def _create_political_boundaries(self):
        """Create political boundary line items (hidden by default)."""
        # Simplified world coastlines and major borders
        # Format: list of (lon, lat) coordinate pairs, None separates segments
        boundaries = self._get_world_boundaries()

        pen = pg.mkPen('#000000', width=2, style=pg.QtCore.Qt.PenStyle.DashLine)

        for segment in boundaries:
            if len(segment) > 1:
                lons = [p[0] for p in segment]
                lats = [p[1] for p in segment]
                line = self.plot_widget.plot(lons, lats, pen=pen)
                line.setVisible(False)
                self.political_lines.append(line)

    def _get_world_boundaries(self):
        """Return simplified world coastline/border coordinates."""
        # Simplified continent outlines and major borders
        # Each segment is a list of (lon, lat) tuples
        return [
            # North America coastline (simplified)
            [(-168, 65), (-165, 60), (-170, 55), (-165, 55), (-155, 58), (-150, 60),
             (-140, 60), (-135, 55), (-125, 50), (-125, 45), (-120, 35), (-115, 30),
             (-110, 25), (-105, 20), (-95, 18), (-90, 20), (-85, 22), (-80, 25),
             (-82, 30), (-80, 32), (-75, 35), (-70, 40), (-70, 45), (-65, 45),
             (-60, 50), (-55, 50), (-55, 55), (-60, 55), (-65, 60), (-70, 65),
             (-80, 70), (-90, 70), (-100, 70), (-120, 72), (-140, 70), (-160, 70), (-168, 65)],

            # South America coastline
            [(-80, 10), (-75, 10), (-70, 12), (-65, 10), (-60, 5), (-50, 0),
             (-45, -5), (-40, -10), (-40, -20), (-45, -25), (-50, -30), (-55, -35),
             (-60, -40), (-65, -45), (-70, -50), (-75, -55), (-70, -55), (-70, -45),
             (-75, -40), (-75, -30), (-70, -20), (-75, -15), (-80, -5), (-80, 0), (-80, 10)],

            # Europe coastline
            [(-10, 35), (-5, 35), (0, 40), (5, 45), (10, 45), (15, 45), (20, 40),
             (25, 40), (30, 45), (30, 50), (25, 55), (20, 55), (15, 55), (10, 55),
             (10, 60), (15, 65), (20, 70), (25, 70), (30, 70), (25, 65), (20, 60),
             (15, 58), (10, 58), (5, 50), (0, 50), (-5, 45), (-10, 45), (-10, 35)],

            # Africa coastline
            [(-15, 35), (-5, 35), (10, 35), (15, 30), (30, 30), (35, 25), (40, 15),
             (50, 10), (45, 0), (40, -5), (35, -10), (30, -20), (25, -30), (20, -35),
             (25, -35), (30, -30), (35, -25), (40, -20), (45, -25), (35, -35),
             (20, -35), (15, -30), (10, -5), (5, 5), (0, 5), (-5, 10), (-15, 15),
             (-20, 15), (-15, 25), (-15, 35)],

            # Asia coastline (simplified)
            [(30, 40), (40, 40), (50, 40), (55, 45), (60, 45), (70, 40), (75, 35),
             (80, 30), (85, 25), (90, 25), (95, 20), (100, 15), (105, 10), (110, 20),
             (120, 25), (125, 30), (130, 35), (135, 35), (140, 40), (145, 45),
             (145, 50), (140, 55), (135, 55), (130, 50), (125, 45), (120, 45),
             (115, 40), (110, 40), (105, 45), (100, 50), (90, 55), (80, 60),
             (70, 65), (60, 70), (50, 70), (40, 65), (35, 60), (30, 55), (30, 45), (30, 40)],

            # Australia coastline
            [(115, -20), (120, -15), (130, -12), (140, -12), (145, -15), (150, -20),
             (155, -25), (150, -30), (150, -35), (145, -40), (140, -38), (135, -35),
             (130, -32), (125, -30), (120, -30), (115, -32), (115, -25), (115, -20)],

            # US-Canada border (49th parallel)
            [(-125, 49), (-120, 49), (-115, 49), (-110, 49), (-105, 49), (-100, 49),
             (-95, 49), (-90, 49), (-85, 49), (-80, 45), (-75, 45)],

            # US-Mexico border (simplified)
            [(-115, 32), (-110, 31), (-105, 30), (-100, 26), (-97, 26)],

            # Russia-China border (simplified)
            [(130, 50), (125, 50), (120, 50), (115, 45), (110, 45), (100, 45), (90, 45), (80, 45)],

            # India outline
            [(68, 25), (70, 20), (72, 15), (78, 8), (82, 8), (88, 22), (92, 25),
             (88, 28), (80, 30), (75, 32), (72, 30), (68, 25)],

            # Japan
            [(130, 32), (132, 34), (135, 35), (140, 38), (142, 42), (145, 44),
             (145, 42), (140, 36), (137, 34), (135, 33), (130, 32)],

            # UK
            [(-5, 50), (0, 51), (2, 52), (0, 55), (-3, 58), (-5, 58), (-6, 55), (-5, 50)],

            # Greenland
            [(-45, 60), (-40, 65), (-35, 70), (-25, 75), (-20, 80), (-30, 82),
             (-45, 82), (-55, 80), (-60, 75), (-55, 70), (-50, 65), (-45, 60)],
        ]

    def set_political_visible(self, visible: bool):
        """
        Toggle political boundary visibility.

        Args:
            visible: True to show boundaries, False to hide
        """
        self.show_political = visible
        for line in self.political_lines:
            line.setVisible(visible)

    def toggle_political(self):
        """Toggle political boundaries on/off."""
        self.set_political_visible(not self.show_political)
        return self.show_political
