"""
Ionosphere Profile Widget

Display hmF2 (layer height) and NmF2 (peak density) history from GloTEC data.
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt6.QtCore import pyqtSlot
import pyqtgraph as pg
import numpy as np
from datetime import datetime
from typing import Optional


class IonosphereProfileWidget(QWidget):
    """
    Ionosphere profile display showing hmF2 and NmF2 time series.

    Shows:
    - hmF2 (F2 layer peak height) history
    - NmF2 (F2 layer peak density) history
    """

    def __init__(self, parent=None):
        """Initialize ionosphere profile widget."""
        super().__init__(parent)

        self.max_points = 500

        # Data buffers
        self.hmf2_times = []
        self.hmf2_values = []
        self.nmf2_times = []
        self.nmf2_values = []

        self._setup_ui()

    def _setup_ui(self):
        """Create and arrange UI elements."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Info bar
        info_layout = QHBoxLayout()
        self.info_label = QLabel("Ionosphere Profile")
        info_layout.addWidget(self.info_label)
        info_layout.addStretch()
        layout.addLayout(info_layout)

        # Create graphics layout for two plots
        self.graphics_layout = pg.GraphicsLayoutWidget()
        layout.addWidget(self.graphics_layout)

        # hmF2 plot (top)
        self.hmf2_plot = self.graphics_layout.addPlot(row=0, col=0)
        self.hmf2_plot.setLabel('left', 'hmF2', units='km')
        self.hmf2_plot.setTitle('F2 Layer Peak Height')
        self.hmf2_plot.showGrid(x=True, y=True, alpha=0.3)

        # Configure time axis for hmF2
        hmf2_time_axis = pg.DateAxisItem(orientation='bottom')
        self.hmf2_plot.setAxisItems({'bottom': hmf2_time_axis})

        self.hmf2_curve = self.hmf2_plot.plot(
            pen=pg.mkPen('g', width=2),
            symbol='o',
            symbolSize=6,
            symbolBrush='g',
            name='hmF2'
        )

        # NmF2 plot (bottom)
        self.nmf2_plot = self.graphics_layout.addPlot(row=1, col=0)
        self.nmf2_plot.setLabel('left', 'NmF2', units='el/m3')
        self.nmf2_plot.setTitle('F2 Layer Peak Density')
        self.nmf2_plot.showGrid(x=True, y=True, alpha=0.3)

        # Configure time axis for NmF2
        nmf2_time_axis = pg.DateAxisItem(orientation='bottom')
        self.nmf2_plot.setAxisItems({'bottom': nmf2_time_axis})

        self.nmf2_curve = self.nmf2_plot.plot(
            pen=pg.mkPen('m', width=2),
            symbol='o',
            symbolSize=6,
            symbolBrush='m',
            name='NmF2'
        )

        # Link X axes
        self.nmf2_plot.setXLink(self.hmf2_plot)

    def update_from_grid(self, grid_data: dict):
        """
        Update from grid data containing hmF2 and NmF2.

        Calculates global statistics from the grid.

        Args:
            grid_data: Dictionary with hmF2, NmF2 arrays and timestamp
        """
        timestamp_str = grid_data.get('timestamp')
        hmf2_grid = grid_data.get('hmF2')
        nmf2_grid = grid_data.get('NmF2')

        if timestamp_str is None:
            return

        try:
            ts = datetime.fromisoformat(timestamp_str.rstrip('Z'))
            ts_float = ts.timestamp()
        except (ValueError, AttributeError):
            ts_float = datetime.utcnow().timestamp()

        # Calculate global means
        if hmf2_grid is not None:
            if not isinstance(hmf2_grid, np.ndarray):
                hmf2_grid = np.array(hmf2_grid)
            hmf2_flat = hmf2_grid.flatten()
            valid_hmf2 = hmf2_flat[~np.isnan(hmf2_flat)]
            if len(valid_hmf2) > 0:
                self.add_hmf2_point(ts_float, float(np.mean(valid_hmf2)))

        if nmf2_grid is not None:
            if not isinstance(nmf2_grid, np.ndarray):
                nmf2_grid = np.array(nmf2_grid)
            nmf2_flat = nmf2_grid.flatten()
            valid_nmf2 = nmf2_flat[~np.isnan(nmf2_flat)]
            if len(valid_nmf2) > 0:
                self.add_nmf2_point(ts_float, float(np.mean(valid_nmf2)))

    def add_hmf2_point(self, timestamp: float, value: float):
        """
        Add hmF2 data point.

        Args:
            timestamp: Unix timestamp
            value: hmF2 value in km
        """
        self.hmf2_times.append(timestamp)
        self.hmf2_values.append(value)

        # Trim to max points
        if len(self.hmf2_times) > self.max_points:
            self.hmf2_times = self.hmf2_times[-self.max_points:]
            self.hmf2_values = self.hmf2_values[-self.max_points:]

        # Update curve
        self.hmf2_curve.setData(self.hmf2_times, self.hmf2_values)

        # Update info
        self._update_info()

    def add_nmf2_point(self, timestamp: float, value: float):
        """
        Add NmF2 data point.

        Args:
            timestamp: Unix timestamp
            value: NmF2 value in el/m3
        """
        self.nmf2_times.append(timestamp)
        self.nmf2_values.append(value)

        # Trim to max points
        if len(self.nmf2_times) > self.max_points:
            self.nmf2_times = self.nmf2_times[-self.max_points:]
            self.nmf2_values = self.nmf2_values[-self.max_points:]

        # Update curve
        self.nmf2_curve.setData(self.nmf2_times, self.nmf2_values)

        # Update info
        self._update_info()

    def _update_info(self):
        """Update info label with latest values."""
        hmf2_str = "---"
        nmf2_str = "---"

        if self.hmf2_values:
            hmf2_str = f"{self.hmf2_values[-1]:.0f} km"

        if self.nmf2_values:
            nmf2_str = f"{self.nmf2_values[-1]:.2e} el/m3"

        self.info_label.setText(f"hmF2: {hmf2_str} | NmF2: {nmf2_str}")

    @pyqtSlot(dict)
    def on_glotec_update(self, map_data: dict):
        """
        Handle GloTEC map update from data manager.

        Args:
            map_data: Full GloTEC map dictionary
        """
        grid = map_data.get('grid', {})
        if grid:
            self.update_from_grid({
                'timestamp': map_data.get('timestamp'),
                'hmF2': grid.get('hmF2'),
                'NmF2': grid.get('NmF2')
            })

    def clear(self):
        """Clear all data."""
        self.hmf2_times.clear()
        self.hmf2_values.clear()
        self.nmf2_times.clear()
        self.nmf2_values.clear()

        self.hmf2_curve.setData([], [])
        self.nmf2_curve.setData([], [])

        self.info_label.setText("Ionosphere Profile")
