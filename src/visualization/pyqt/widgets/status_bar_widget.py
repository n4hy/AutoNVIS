"""
Status Bar Widget

Connection status and data freshness display.
"""

from PyQt6.QtWidgets import QStatusBar, QLabel, QWidget, QHBoxLayout
from PyQt6.QtCore import QTimer, pyqtSlot
from datetime import datetime
from typing import Optional


class StatusBarWidget(QStatusBar):
    """
    Custom status bar showing connection status and data freshness.

    Displays:
    - WebSocket connection status
    - Last update time
    - Current TEC statistics
    """

    def __init__(self, parent=None):
        """Initialize status bar widget."""
        super().__init__(parent)

        self.last_update_time: Optional[datetime] = None
        self.tec_mean: Optional[float] = None
        self.tec_max: Optional[float] = None

        self._setup_ui()

        # Update timer for freshness display
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self._update_age_display)
        self.update_timer.start(1000)  # Update every second

    def _setup_ui(self):
        """Create status bar elements."""
        # Connection status indicator
        self.connection_widget = QWidget()
        conn_layout = QHBoxLayout(self.connection_widget)
        conn_layout.setContentsMargins(0, 0, 0, 0)

        self.connection_indicator = QLabel()
        self.connection_indicator.setFixedSize(12, 12)
        self.connection_indicator.setStyleSheet(
            "background-color: #ff4444; border-radius: 6px;"
        )
        conn_layout.addWidget(self.connection_indicator)

        self.connection_label = QLabel("Disconnected")
        conn_layout.addWidget(self.connection_label)

        self.addWidget(self.connection_widget)

        # Separator
        self.addWidget(QLabel("|"))

        # Last update time
        self.update_label = QLabel("No data")
        self.addWidget(self.update_label)

        # Separator
        self.addWidget(QLabel("|"))

        # TEC statistics
        self.tec_label = QLabel("TEC: ---")
        self.addWidget(self.tec_label)

        # Add stretch to push permanent widgets to the right
        self.addPermanentWidget(QLabel("AutoNVIS TEC Display"))

    @pyqtSlot(str, bool)
    def set_connection_status(self, source: str, connected: bool):
        """
        Update connection status display.

        Args:
            source: Connection source identifier
            connected: Whether connected
        """
        if connected:
            self.connection_indicator.setStyleSheet(
                "background-color: #44ff44; border-radius: 6px;"
            )
            self.connection_label.setText("Connected")
        else:
            self.connection_indicator.setStyleSheet(
                "background-color: #ff4444; border-radius: 6px;"
            )
            self.connection_label.setText("Disconnected")

    @pyqtSlot(dict)
    def update_statistics(self, stats: dict):
        """
        Update TEC statistics display.

        Args:
            stats: Dictionary with tec_mean, tec_max, timestamp
        """
        timestamp_str = stats.get('timestamp')
        self.tec_mean = stats.get('tec_mean')
        self.tec_max = stats.get('tec_max')

        if timestamp_str:
            try:
                self.last_update_time = datetime.fromisoformat(timestamp_str.rstrip('Z'))
            except (ValueError, AttributeError):
                self.last_update_time = datetime.utcnow()

        self._update_display()

    def _update_display(self):
        """Update all display elements."""
        # TEC statistics
        if self.tec_mean is not None and self.tec_max is not None:
            self.tec_label.setText(
                f"TEC Mean: {self.tec_mean:.1f} TECU | Max: {self.tec_max:.1f} TECU"
            )
        elif self.tec_mean is not None:
            self.tec_label.setText(f"TEC Mean: {self.tec_mean:.1f} TECU")
        else:
            self.tec_label.setText("TEC: ---")

        # Update time
        self._update_age_display()

    def _update_age_display(self):
        """Update the data age display."""
        if self.last_update_time is None:
            self.update_label.setText("No data")
            return

        age = (datetime.utcnow() - self.last_update_time).total_seconds()

        if age < 60:
            age_str = f"{int(age)}s ago"
        elif age < 3600:
            age_str = f"{int(age / 60)}m ago"
        else:
            age_str = f"{int(age / 3600)}h ago"

        # Color code based on freshness
        if age < 120:  # Fresh (< 2 min)
            color = "#44ff44"
        elif age < 600:  # Warning (< 10 min)
            color = "#ffff44"
        else:  # Stale
            color = "#ff4444"

        self.update_label.setText(f"<span style='color: {color}'>Last: {age_str}</span>")
