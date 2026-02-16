"""
Flare Class Indicator Widget

Large visual indicator showing current solar flare class and system mode.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QGridLayout
)
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QFont


class FlareIndicatorWidget(QWidget):
    """
    Large flare class indicator with color coding.

    Shows:
    - Current flare class (A/B/C/M/X) with large text
    - Color coded by severity
    - QUIET/SHOCK mode indicator
    - M1+ threshold alert
    """

    # Flare class colors
    CLASS_COLORS = {
        'A': '#4488ff',   # Blue - very quiet
        'B': '#44ff44',   # Green - quiet
        'C': '#ffff44',   # Yellow - minor
        'M': '#ff8844',   # Orange - moderate
        'X': '#ff4444',   # Red - major
    }

    def __init__(self, parent=None):
        super().__init__(parent)

        self.current_class = 'B'
        self.current_flux = 0.0
        self.m1_threshold = False

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # Main frame
        self.main_frame = QFrame()
        self.main_frame.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Raised)
        self.main_frame.setStyleSheet("""
            QFrame {
                background-color: #2d2d2d;
                border: 2px solid #444;
                border-radius: 10px;
            }
        """)

        frame_layout = QVBoxLayout(self.main_frame)

        # Title
        title = QLabel("Solar X-Ray Status")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 12px; color: #888;")
        frame_layout.addWidget(title)

        # Flare class display (large)
        self.class_label = QLabel("B5.6")
        self.class_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont()
        font.setPointSize(48)
        font.setBold(True)
        self.class_label.setFont(font)
        self.class_label.setStyleSheet(f"color: {self.CLASS_COLORS['B']};")
        frame_layout.addWidget(self.class_label)

        # Flux value
        self.flux_label = QLabel("5.6e-7 W/m²")
        self.flux_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.flux_label.setStyleSheet("font-size: 14px; color: #aaa;")
        frame_layout.addWidget(self.flux_label)

        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setStyleSheet("background-color: #444;")
        frame_layout.addWidget(separator)

        # Mode indicator
        mode_layout = QHBoxLayout()

        mode_title = QLabel("Mode:")
        mode_title.setStyleSheet("color: #888;")
        mode_layout.addWidget(mode_title)

        self.mode_label = QLabel("QUIET")
        self.mode_label.setStyleSheet("""
            font-weight: bold;
            font-size: 16px;
            color: #44ff44;
            padding: 5px 15px;
            background-color: #1a3a1a;
            border-radius: 5px;
        """)
        mode_layout.addWidget(self.mode_label)

        mode_layout.addStretch()
        frame_layout.addLayout(mode_layout)

        # Alert indicator (hidden by default)
        self.alert_label = QLabel("M1+ FLARE - SHOCK MODE TRIGGERED")
        self.alert_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.alert_label.setStyleSheet("""
            font-weight: bold;
            font-size: 12px;
            color: #ff4444;
            background-color: #3a1a1a;
            padding: 10px;
            border-radius: 5px;
        """)
        self.alert_label.hide()
        frame_layout.addWidget(self.alert_label)

        layout.addWidget(self.main_frame)

    def update_flare_class(self, flare_class: str, flux: float, m1_threshold: bool):
        """
        Update the flare class display.

        Args:
            flare_class: Flare class string (e.g., 'B5.6', 'M2.1')
            flux: X-ray flux in W/m²
            m1_threshold: Whether M1 threshold is exceeded
        """
        self.current_flux = flux
        self.m1_threshold = m1_threshold

        # Extract class letter
        if flare_class:
            self.current_class = flare_class[0].upper()

        # Update class label
        self.class_label.setText(flare_class or "--")

        # Update color
        color = self.CLASS_COLORS.get(self.current_class, '#888888')
        self.class_label.setStyleSheet(f"color: {color};")

        # Update flux label
        self.flux_label.setText(f"{flux:.2e} W/m²")

        # Update mode indicator
        if m1_threshold:
            self.mode_label.setText("SHOCK")
            self.mode_label.setStyleSheet("""
                font-weight: bold;
                font-size: 16px;
                color: #ff4444;
                padding: 5px 15px;
                background-color: #3a1a1a;
                border-radius: 5px;
            """)
            self.alert_label.show()
        else:
            self.mode_label.setText("QUIET")
            self.mode_label.setStyleSheet("""
                font-weight: bold;
                font-size: 16px;
                color: #44ff44;
                padding: 5px 15px;
                background-color: #1a3a1a;
                border-radius: 5px;
            """)
            self.alert_label.hide()

        # Update frame border color based on severity
        self.main_frame.setStyleSheet(f"""
            QFrame {{
                background-color: #2d2d2d;
                border: 2px solid {color};
                border-radius: 10px;
            }}
        """)

    @pyqtSlot(dict)
    def on_xray_update(self, data: dict):
        """Handle X-ray data update from WebSocket."""
        flare_class = data.get('flare_class', '--')
        flux = data.get('flux_long') or data.get('flux', 0.0)
        m1_threshold = data.get('m1_or_higher', False)

        self.update_flare_class(flare_class, flux, m1_threshold)
