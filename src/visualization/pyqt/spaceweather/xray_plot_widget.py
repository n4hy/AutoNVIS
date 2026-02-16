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
    - Both short (0.05-0.4nm) and long (0.1-0.8nm) wavelength channels
    - Horizontal lines at A/B/C/M/X class boundaries
    - Current flux value highlighted
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.max_points = 2000  # ~24 hours at 1-min intervals

        # Data buffers - both channels
        self.times = []
        self.flux_long = []   # 0.1-0.8nm (used for flare classification)
        self.flux_short = []  # 0.05-0.4nm (higher energy)

        # Autoscale mode
        self.autoscale_enabled = False

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

        # Flux curves - two channels
        self.flux_long_curve = self.plot_widget.plot(
            pen=pg.mkPen('#00ff00', width=2),
            name='Long (0.1-0.8nm)'
        )
        self.flux_short_curve = self.plot_widget.plot(
            pen=pg.mkPen('#ff6600', width=2),
            name='Short (0.05-0.4nm)'
        )

        # Add legend
        self.plot_widget.addLegend(offset=(10, 10))

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

    def add_data_point(self, timestamp: str, flux_long: float, flux_short: float = None):
        """
        Add a new X-ray flux data point.

        Args:
            timestamp: ISO format timestamp
            flux_long: Long wavelength (0.1-0.8nm) flux in W/m²
            flux_short: Short wavelength (0.05-0.4nm) flux in W/m² (optional)
        """
        try:
            ts = datetime.fromisoformat(timestamp.rstrip('Z'))
            ts_float = ts.timestamp()
        except (ValueError, AttributeError):
            ts_float = datetime.utcnow().timestamp()

        self.times.append(ts_float)
        self.flux_long.append(flux_long)
        self.flux_short.append(flux_short if flux_short else flux_long)

        # Trim to max points
        if len(self.times) > self.max_points:
            self.times = self.times[-self.max_points:]
            self.flux_long = self.flux_long[-self.max_points:]
            self.flux_short = self.flux_short[-self.max_points:]

        self._update_plot()

    def _update_plot(self):
        """Update the plot with current data."""
        if len(self.times) < 2:
            return

        times_arr = np.array(self.times)
        flux_long_arr = np.array(self.flux_long)
        flux_short_arr = np.array(self.flux_short)

        # Filter out invalid values for long wavelength
        valid_long = flux_long_arr > 0
        if np.any(valid_long):
            self.flux_long_curve.setData(times_arr[valid_long], flux_long_arr[valid_long])

        # Filter out invalid values for short wavelength
        valid_short = flux_short_arr > 0
        if np.any(valid_short):
            self.flux_short_curve.setData(times_arr[valid_short], flux_short_arr[valid_short])

        # Update current value label with both values
        if len(self.flux_long) > 0:
            current_long = self.flux_long[-1]
            current_short = self.flux_short[-1] if self.flux_short else current_long
            self.current_label.setText(f"Long: {current_long:.2e}  Short: {current_short:.2e} W/m²")

    @pyqtSlot(dict)
    def on_xray_update(self, data: dict):
        """Handle X-ray data update from WebSocket."""
        timestamp = data.get('timestamp')
        flux_long = data.get('flux_long') or data.get('flux')
        flux_short = data.get('flux_short')

        if timestamp and flux_long:
            self.add_data_point(timestamp, flux_long, flux_short)

    @pyqtSlot(dict)
    def on_xray_batch(self, data: dict):
        """Handle historical X-ray batch - load all at once without individual updates."""
        records = data.get('records', [])
        if not records:
            return

        # Process all records without updating plot each time
        for record in records:
            timestamp = record.get('timestamp')
            flux_long = record.get('flux_long') or record.get('flux_short')
            flux_short = record.get('flux_short') or flux_long

            if timestamp and flux_long:
                try:
                    ts = datetime.fromisoformat(timestamp.rstrip('Z'))
                    ts_float = ts.timestamp()
                except (ValueError, AttributeError):
                    continue

                self.times.append(ts_float)
                self.flux_long.append(flux_long)
                self.flux_short.append(flux_short)

        # Trim to max points
        if len(self.times) > self.max_points:
            self.times = self.times[-self.max_points:]
            self.flux_long = self.flux_long[-self.max_points:]
            self.flux_short = self.flux_short[-self.max_points:]

        # Single update after all data loaded
        self._update_plot()

    def clear(self):
        """Clear all data."""
        self.times.clear()
        self.flux_long.clear()
        self.flux_short.clear()
        self.flux_long_curve.setData([], [])
        self.flux_short_curve.setData([], [])
        self.current_label.setText("--")

    def set_autoscale(self, enabled: bool):
        """
        Enable or disable Y-axis autoscaling.

        Args:
            enabled: True for autoscale, False for fixed range
        """
        self.autoscale_enabled = enabled

        if enabled:
            # Enable autoscaling
            self.plot_widget.enableAutoRange(axis='y')
        else:
            # Fixed range for typical solar activity
            self.plot_widget.disableAutoRange(axis='y')
            self.plot_widget.setYRange(-9, -3)  # 1e-9 to 1e-3 W/m²

    def is_autoscale_enabled(self) -> bool:
        """Return current autoscale state."""
        return self.autoscale_enabled
