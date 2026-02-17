"""
TEC Time Series Widget

Real-time TEC time series visualization using pyqtgraph.
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox
from PyQt6.QtCore import pyqtSlot
import pyqtgraph as pg
import numpy as np
from datetime import datetime
from typing import Optional, Tuple


class TECTimeSeriesWidget(QWidget):
    """
    Real-time TEC time series display.

    Shows:
    - Global mean TEC (24h history)
    - Selected point TEC
    """

    def __init__(self, parent=None):
        """Initialize time series widget."""
        super().__init__(parent)

        self.max_points = 1000  # Maximum points to display
        self.tracked_lat: Optional[float] = None
        self.tracked_lon: Optional[float] = None

        # Data buffers
        self.global_mean_times = []
        self.global_mean_values = []
        self.point_times = []
        self.point_values = []

        self._setup_ui()

    def _setup_ui(self):
        """Create and arrange UI elements."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Control bar
        control_layout = QHBoxLayout()

        self.show_global_check = QCheckBox("Global Mean")
        self.show_global_check.setChecked(True)
        self.show_global_check.stateChanged.connect(self._update_visibility)
        control_layout.addWidget(self.show_global_check)

        self.show_point_check = QCheckBox("Selected Point")
        self.show_point_check.setChecked(True)
        self.show_point_check.stateChanged.connect(self._update_visibility)
        control_layout.addWidget(self.show_point_check)

        control_layout.addStretch()

        self.point_label = QLabel("No point selected")
        control_layout.addWidget(self.point_label)

        layout.addLayout(control_layout)

        # Create plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('bottom', 'Time', units='UTC')
        self.plot_widget.setLabel('left', 'TEC', units='TECU')
        self.plot_widget.setTitle('TEC Time Series')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.addLegend()

        # Enable auto-range on Y axis
        self.plot_widget.enableAutoRange(axis='y')

        # Global mean line (yellow) with symbols for single-point visibility
        self.global_mean_curve = self.plot_widget.plot(
            pen=pg.mkPen('y', width=2),
            symbol='o',
            symbolSize=6,
            symbolBrush='y',
            name='Global Mean'
        )

        # Selected point line (cyan) with symbols
        self.point_curve = self.plot_widget.plot(
            pen=pg.mkPen('c', width=2),
            symbol='o',
            symbolSize=6,
            symbolBrush='c',
            name='Selected Point'
        )

        # Configure time axis
        self.time_axis = pg.DateAxisItem(orientation='bottom')
        self.plot_widget.setAxisItems({'bottom': self.time_axis})

        layout.addWidget(self.plot_widget)

    def set_tracked_point(self, lat: float, lon: float):
        """
        Set the point to track in time series.

        Args:
            lat: Latitude
            lon: Longitude
        """
        self.tracked_lat = lat
        self.tracked_lon = lon
        self.point_label.setText(f"Point: ({lat:.2f}, {lon:.2f})")

        # Clear point history
        self.point_times.clear()
        self.point_values.clear()
        self.point_curve.setData([], [])

    def add_global_mean_point(self, timestamp: float, value: float):
        """
        Add a global mean data point.

        Args:
            timestamp: Unix timestamp
            value: TEC value in TECU
        """
        self.global_mean_times.append(timestamp)
        self.global_mean_values.append(value)

        # Trim to max points
        if len(self.global_mean_times) > self.max_points:
            self.global_mean_times = self.global_mean_times[-self.max_points:]
            self.global_mean_values = self.global_mean_values[-self.max_points:]

        # Update curve
        if self.show_global_check.isChecked():
            self.global_mean_curve.setData(
                self.global_mean_times,
                self.global_mean_values
            )

    def add_point_value(self, timestamp: float, value: float):
        """
        Add a selected point data value.

        Args:
            timestamp: Unix timestamp
            value: TEC value in TECU
        """
        self.point_times.append(timestamp)
        self.point_values.append(value)

        # Trim to max points
        if len(self.point_times) > self.max_points:
            self.point_times = self.point_times[-self.max_points:]
            self.point_values = self.point_values[-self.max_points:]

        # Update curve
        if self.show_point_check.isChecked():
            self.point_curve.setData(
                self.point_times,
                self.point_values
            )

    def update_from_history(
        self,
        global_times: np.ndarray,
        global_values: np.ndarray,
        point_times: Optional[np.ndarray] = None,
        point_values: Optional[np.ndarray] = None
    ):
        """
        Update plots from history arrays.

        Args:
            global_times: Array of timestamps for global mean
            global_values: Array of global mean TEC values
            point_times: Array of timestamps for selected point
            point_values: Array of TEC values for selected point
        """
        # Update global mean
        self.global_mean_times = list(global_times)
        self.global_mean_values = list(global_values)

        if self.show_global_check.isChecked() and len(global_times) > 0:
            self.global_mean_curve.setData(global_times, global_values)

        # Update point
        if point_times is not None and point_values is not None:
            self.point_times = list(point_times)
            self.point_values = list(point_values)

            if self.show_point_check.isChecked() and len(point_times) > 0:
                self.point_curve.setData(point_times, point_values)

    @pyqtSlot(dict)
    def on_statistics_update(self, stats: dict):
        """
        Handle statistics update from data manager.

        Args:
            stats: Dictionary with timestamp and tec_mean
        """
        timestamp_str = stats.get('timestamp')
        tec_mean = stats.get('tec_mean')

        if timestamp_str and tec_mean is not None:
            try:
                ts = datetime.fromisoformat(timestamp_str.rstrip('Z'))
                ts_float = ts.timestamp()
                self.add_global_mean_point(ts_float, tec_mean)
            except (ValueError, AttributeError):
                pass

    def _update_visibility(self):
        """Update curve visibility based on checkboxes."""
        if self.show_global_check.isChecked() and self.global_mean_times:
            self.global_mean_curve.setData(
                self.global_mean_times,
                self.global_mean_values
            )
        else:
            self.global_mean_curve.setData([], [])

        if self.show_point_check.isChecked() and self.point_times:
            self.point_curve.setData(
                self.point_times,
                self.point_values
            )
        else:
            self.point_curve.setData([], [])

    def clear(self):
        """Clear all data."""
        self.global_mean_times.clear()
        self.global_mean_values.clear()
        self.point_times.clear()
        self.point_values.clear()

        self.global_mean_curve.setData([], [])
        self.point_curve.setData([], [])
