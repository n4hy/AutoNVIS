"""
Solar Image Display Widgets

Custom PyQt6 widgets for displaying solar images with labels, timestamps,
and download functionality.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSizePolicy, QFileDialog, QMessageBox, QToolTip
)
from PyQt6.QtGui import QPixmap, QImage, QCursor
from PyQt6.QtCore import Qt, pyqtSlot, pyqtSignal


class SolarImagePanel(QWidget):
    """
    Widget for displaying a single solar image with header, timestamp, and download button.

    Layout:
    ┌─────────────────────────────┐
    │  SOURCE NAME | WAVELENGTH   │  <- Header
    ├─────────────────────────────┤
    │                             │
    │         [IMAGE]             │  <- Scaled image
    │                             │
    ├─────────────────────────────┤
    │ 2026-02-17 16:12:33 UTC [⬇] │  <- Timestamp + Download
    └─────────────────────────────┘
    """

    # Signal emitted when download is requested
    download_requested = pyqtSignal(str)  # source_id

    def __init__(self, source_id: str, source_name: str, wavelength: str,
                 description: str = "", parent=None):
        super().__init__(parent)
        self.source_id = source_id
        self.source_name = source_name
        self.wavelength = wavelength
        self.description = description

        # Store original image data for download and resize
        self._image_data: Optional[bytes] = None
        self._timestamp: Optional[str] = None
        self._original_pixmap: Optional[QPixmap] = None

        self._setup_ui()

    def _setup_ui(self):
        """Setup the widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        # Header: "GOES SUVI | 195 Angstrom"
        self.header = QLabel(f"{self.source_name} | {self.wavelength}")
        self.header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.header.setStyleSheet("""
            font-weight: bold;
            font-size: 13px;
            color: #ffffff;
            padding: 4px;
        """)
        self.header.setToolTip(self.description)

        # Image container with dark background
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(200, 200)
        self.image_label.setStyleSheet("""
            background-color: #0a0a0a;
            border: 1px solid #3a3a3a;
            border-radius: 4px;
        """)
        self.image_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )
        self.image_label.setText("Loading...")
        self.image_label.setToolTip(self.description)

        # Footer row: timestamp + download button
        footer_layout = QHBoxLayout()
        footer_layout.setSpacing(8)

        self.timestamp_label = QLabel("Waiting for data...")
        self.timestamp_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.timestamp_label.setStyleSheet("color: #888888; font-size: 11px;")

        self.download_btn = QPushButton("Save")
        self.download_btn.setFixedWidth(50)
        self.download_btn.setStyleSheet("""
            QPushButton {
                background-color: #2a5a8a;
                color: white;
                border: none;
                border-radius: 3px;
                padding: 4px 8px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #3a6a9a;
            }
            QPushButton:pressed {
                background-color: #1a4a7a;
            }
            QPushButton:disabled {
                background-color: #444444;
                color: #888888;
            }
        """)
        self.download_btn.setEnabled(False)
        self.download_btn.clicked.connect(self._on_download_clicked)
        self.download_btn.setToolTip("Save image to file")

        footer_layout.addWidget(self.timestamp_label, stretch=1)
        footer_layout.addWidget(self.download_btn)

        # Assemble layout
        layout.addWidget(self.header)
        layout.addWidget(self.image_label, stretch=1)
        layout.addLayout(footer_layout)

        # Overall widget styling
        self.setStyleSheet("""
            SolarImagePanel {
                background-color: #1e1e1e;
                border: 1px solid #3a3a3a;
                border-radius: 6px;
            }
        """)

    @pyqtSlot(bytes, str)
    def update_image(self, data: bytes, timestamp: str):
        """Update the displayed image with new data."""
        self._image_data = data
        self._timestamp = timestamp

        # Load image from bytes
        pixmap = QPixmap()
        if pixmap.loadFromData(data):
            self._original_pixmap = pixmap
            self._scale_and_display()
            self.download_btn.setEnabled(True)
        else:
            self.image_label.setText("Error loading image")
            self.download_btn.setEnabled(False)

        # Update timestamp display
        self._update_timestamp_display(timestamp)

    def _scale_and_display(self):
        """Scale the original pixmap to fit the label."""
        if self._original_pixmap is None or self._original_pixmap.isNull():
            return

        # Get available size
        available_size = self.image_label.size()

        # Scale maintaining aspect ratio
        scaled = self._original_pixmap.scaled(
            available_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.image_label.setPixmap(scaled)

    def _update_timestamp_display(self, timestamp: str):
        """Format and display the timestamp."""
        try:
            # Handle various ISO formats
            ts = timestamp.replace('Z', '+00:00')
            if '.' in ts:
                # Remove microseconds for cleaner display
                ts = ts.split('.')[0] + '+00:00'
            dt = datetime.fromisoformat(ts)
            display_str = dt.strftime('%Y-%m-%d %H:%M:%S UTC')
        except (ValueError, AttributeError):
            display_str = timestamp

        self.timestamp_label.setText(display_str)

    def _on_download_clicked(self):
        """Handle download button click."""
        if self._image_data is None:
            return

        # Generate default filename: SOURCE_WAVELENGTH_YYYYMMDD_HHMMSS.png
        timestamp_str = ""
        if self._timestamp:
            try:
                ts = self._timestamp.replace('Z', '+00:00')
                if '.' in ts:
                    ts = ts.split('.')[0] + '+00:00'
                dt = datetime.fromisoformat(ts)
                timestamp_str = dt.strftime('%Y%m%d_%H%M%S')
            except (ValueError, AttributeError):
                timestamp_str = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        else:
            timestamp_str = datetime.utcnow().strftime('%Y%m%d_%H%M%S')

        # Clean wavelength for filename (remove special chars)
        clean_wavelength = self.wavelength.replace(' ', '_').replace('/', '_')

        default_filename = f"{self.source_id}_{timestamp_str}.png"

        # Open save dialog
        downloads_dir = Path.home() / "Downloads"
        if not downloads_dir.exists():
            downloads_dir = Path.home()

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Save Solar Image",
            str(downloads_dir / default_filename),
            "PNG Images (*.png);;All Files (*)"
        )

        if filepath:
            try:
                with open(filepath, 'wb') as f:
                    f.write(self._image_data)
                # Show brief tooltip confirmation
                QToolTip.showText(
                    QCursor.pos(),
                    f"Saved: {Path(filepath).name}",
                    self,
                    self.rect(),
                    2000
                )
            except Exception as e:
                QMessageBox.warning(
                    self,
                    "Save Failed",
                    f"Could not save image: {e}"
                )

    def resizeEvent(self, event):
        """Handle resize to re-scale image."""
        super().resizeEvent(event)
        self._scale_and_display()

    def set_error(self, error_msg: str):
        """Display an error state."""
        self.image_label.setText(f"Error: {error_msg}")
        self.timestamp_label.setText("Error")
        self.timestamp_label.setStyleSheet("color: #ff6666; font-size: 11px;")

    def clear(self):
        """Clear the current image."""
        self._image_data = None
        self._timestamp = None
        self._original_pixmap = None
        self.image_label.clear()
        self.image_label.setText("Loading...")
        self.timestamp_label.setText("Waiting for data...")
        self.download_btn.setEnabled(False)


class StatusIndicator(QWidget):
    """Small status indicator widget showing connection state."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self.set_disconnected()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.indicator = QLabel()
        self.indicator.setFixedSize(10, 10)
        self.indicator.setStyleSheet("""
            background-color: #888888;
            border-radius: 5px;
        """)

        self.label = QLabel("Disconnected")
        self.label.setStyleSheet("color: #888888; font-size: 11px;")

        layout.addWidget(self.indicator)
        layout.addWidget(self.label)

    def set_connected(self):
        """Show connected state."""
        self.indicator.setStyleSheet("""
            background-color: #44ff44;
            border-radius: 5px;
        """)
        self.label.setText("Connected")
        self.label.setStyleSheet("color: #44ff44; font-size: 11px;")

    def set_disconnected(self):
        """Show disconnected state."""
        self.indicator.setStyleSheet("""
            background-color: #888888;
            border-radius: 5px;
        """)
        self.label.setText("Disconnected")
        self.label.setStyleSheet("color: #888888; font-size: 11px;")

    def set_fetching(self):
        """Show fetching state."""
        self.indicator.setStyleSheet("""
            background-color: #ffff44;
            border-radius: 5px;
        """)
        self.label.setText("Fetching...")
        self.label.setStyleSheet("color: #ffff44; font-size: 11px;")

    def set_error(self):
        """Show error state."""
        self.indicator.setStyleSheet("""
            background-color: #ff4444;
            border-radius: 5px;
        """)
        self.label.setText("Error")
        self.label.setStyleSheet("color: #ff4444; font-size: 11px;")
