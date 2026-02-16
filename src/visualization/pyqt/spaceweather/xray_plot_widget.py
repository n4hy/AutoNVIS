"""
X-Ray Flux Time Series Widget

Displays GOES X-ray flux with logarithmic scale and flare class thresholds.
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt6.QtCore import pyqtSlot
import pyqtgraph as pg
import numpy as np
from datetime import datetime
from typing import Optional


class XRayPlotWidget(QWidget):
    """
    X-ray flux time series with flare class threshold lines.

    Shows:
    - 24-hour X-ray flux history (logarithmic scale)
    - Horizontal lines at A/B/C/M/X class boundaries
    - Current flux value highlighted
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.max_points = 2000  # ~24 hours at 1-min intervals

        # Data buffers
        self.times = []
        self.flux_values = []

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Title
        title_layout = QHBoxLayout()
        self.title_label = QLabel("GOES X-Ray Flux (0.1-0.8 nm)")
        self.title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        title_layout.addWidget(self.title_label)
        title_layout.addStretch()
        self.current_label = QLabel("--")
        self.current_label.setStyleSheet("font-size: 14px; color: #00ff00;")
        title_layout.addWidget(self.current_label)
        layout.addLayout(title_layout)

        # Plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#1e1e1e')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)

        # Logarithmic Y axis
        self.plot_widget.setLogMode(x=False, y=True)
        self.plot_widget.setLabel('left', 'Flux', units='W/m²')

        # Time axis
        time_axis = pg.DateAxisItem(orientation='bottom')
        self.plot_widget.setAxisItems({'bottom': time_axis})

        # Set Y range for typical solar activity
        self.plot_widget.setYRange(-9, -3)  # 1e-9 to 1e-3 W/m²

        # Add flare class threshold lines
        self._add_threshold_lines()

        # Flux curve
        self.flux_curve = self.plot_widget.plot(
            pen=pg.mkPen('#00ff00', width=2),
            name='X-ray Flux'
        )

        layout.addWidget(self.plot_widget)

    def _add_threshold_lines(self):
        """Add horizontal lines at flare class boundaries."""
        thresholds = [
            (1e-8, 'B', '#4444ff'),   # B class
            (1e-7, 'C', '#44ff44'),   # C class
            (1e-6, 'M', '#ffff44'),   # M class
            (1e-5, 'X', '#ff4444'),   # X class
        ]

        for flux, label, color in thresholds:
            line = pg.InfiniteLine(
                pos=np.log10(flux),
                angle=0,
                pen=pg.mkPen(color, width=1, style=pg.QtCore.Qt.PenStyle.DashLine)
            )
            self.plot_widget.addItem(line)

            # Add label
            text = pg.TextItem(label, color=color, anchor=(0, 0.5))
            text.setPos(self.plot_widget.getViewBox().viewRange()[0][0], np.log10(flux))
            self.plot_widget.addItem(text)

    def add_data_point(self, timestamp: str, flux: float):
        """
        Add a new X-ray flux data point.

        Args:
            timestamp: ISO format timestamp
            flux: X-ray flux in W/m²
        """
        try:
            ts = datetime.fromisoformat(timestamp.rstrip('Z'))
            ts_float = ts.timestamp()
        except (ValueError, AttributeError):
            ts_float = datetime.utcnow().timestamp()

        self.times.append(ts_float)
        self.flux_values.append(flux)

        # Trim to max points
        if len(self.times) > self.max_points:
            self.times = self.times[-self.max_points:]
            self.flux_values = self.flux_values[-self.max_points:]

        self._update_plot()

    def _update_plot(self):
        """Update the plot with current data."""
        if len(self.times) < 2:
            return

        times_arr = np.array(self.times)
        flux_arr = np.array(self.flux_values)

        # Filter out invalid values
        valid = flux_arr > 0
        if not np.any(valid):
            return

        self.flux_curve.setData(times_arr[valid], flux_arr[valid])

        # Update current value label
        if len(self.flux_values) > 0:
            current_flux = self.flux_values[-1]
            self.current_label.setText(f"{current_flux:.2e} W/m²")

    @pyqtSlot(dict)
    def on_xray_update(self, data: dict):
        """Handle X-ray data update from WebSocket."""
        timestamp = data.get('timestamp')
        flux = data.get('flux_long') or data.get('flux')

        if timestamp and flux:
            self.add_data_point(timestamp, flux)

    def clear(self):
        """Clear all data."""
        self.times.clear()
        self.flux_values.clear()
        self.flux_curve.setData([], [])
        self.current_label.setText("--")
