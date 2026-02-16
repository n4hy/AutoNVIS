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
    """

    point_selected = pyqtSignal(float, float)  # lat, lon

    def __init__(self, parent=None):
        """Initialize TEC map widget."""
        super().__init__(parent)

        self.current_layer = 'tec'  # tec, anomaly, hmF2, NmF2
        self.last_grid_data = None

        self._setup_ui()

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

        # Update colorbar based on layer
        self._update_colorbar()

        # Update info label
        timestamp = grid_data.get('timestamp', 'Unknown')
        valid_data = data[~np.isnan(data)]
        if len(valid_data) > 0:
            self.info_label.setText(
                f"{timestamp} | Mean: {np.mean(valid_data):.1f} | Max: {np.max(valid_data):.1f}"
            )
        else:
            self.info_label.setText(f"{timestamp} | No valid data")

    def _update_colorbar(self):
        """Update colorbar for current layer."""
        layer = self.current_layer.lower()

        if layer == 'tec':
            self.colorbar.setLevels((0, 100))
            self.colormap = pg.colormap.get('plasma')

        elif layer == 'anomaly':
            self.colorbar.setLevels((-20, 20))
            self.colormap = pg.colormap.get('CET-D1')  # Diverging colormap

        elif layer in ('hmf2', 'hmF2'):
            self.colorbar.setLevels((200, 500))
            self.colormap = pg.colormap.get('viridis')

        elif layer in ('nmf2', 'NmF2'):
            self.colorbar.setLevels((0, 1e12))
            self.colormap = pg.colormap.get('inferno')

        self.tec_image.setColorMap(self.colormap)

    def _on_layer_changed(self, layer_name: str):
        """Handle layer selection change."""
        self.current_layer = layer_name.lower()
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
