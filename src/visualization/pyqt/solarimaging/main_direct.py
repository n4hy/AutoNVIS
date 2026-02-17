#!/usr/bin/env python3
"""
Solar Imaging Display - Standalone Application

Real-time multi-source solar imagery viewer.
Displays images from GOES SUVI, SDO AIA, SDO HMI, SOHO LASCO, and SOHO EIT.

Usage (from distribution directory):
    ./run.sh

Or manually:
    source venv/bin/activate
    python -m solarimaging.main_direct
"""

import sys
import logging
import signal

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QPalette, QColor
from PyQt6.QtCore import Qt


def apply_dark_theme(app: QApplication):
    """Apply a dark theme to the application."""
    app.setStyle("Fusion")

    palette = QPalette()

    # Window backgrounds
    palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)

    # Base (text inputs, lists)
    palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))

    # Tooltips
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(42, 42, 42))
    palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)

    # Text
    palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(127, 127, 127))

    # Buttons
    palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)

    # Special text
    palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
    palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))

    # Selections
    palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)

    # Disabled state
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, QColor(127, 127, 127))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(127, 127, 127))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(127, 127, 127))

    app.setPalette(palette)

    # Additional stylesheet for fine-tuning
    app.setStyleSheet("""
        QToolTip {
            background-color: #2a2a2a;
            color: white;
            border: 1px solid #555555;
            padding: 4px;
            border-radius: 4px;
        }
        QScrollBar:vertical {
            background-color: #2a2a2a;
            width: 14px;
            margin: 0;
        }
        QScrollBar::handle:vertical {
            background-color: #5a5a5a;
            min-height: 30px;
            border-radius: 7px;
            margin: 2px;
        }
        QScrollBar::handle:vertical:hover {
            background-color: #6a6a6a;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0;
        }
        QScrollBar:horizontal {
            background-color: #2a2a2a;
            height: 14px;
            margin: 0;
        }
        QScrollBar::handle:horizontal {
            background-color: #5a5a5a;
            min-width: 30px;
            border-radius: 7px;
            margin: 2px;
        }
        QScrollBar::handle:horizontal:hover {
            background-color: #6a6a6a;
        }
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
            width: 0;
        }
    """)


def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main():
    """Main entry point."""
    setup_logging()
    logger = logging.getLogger("solar_imaging")
    logger.info("Starting Solar Imaging Display...")

    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("Solar Imaging Display")
    app.setOrganizationName("AutoNVIS")

    # Apply dark theme
    apply_dark_theme(app)

    # Import here to avoid circular imports
    from .main_window import SolarImagingMainWindow
    from .data_client import SolarImageDataClient

    # Create main window
    window = SolarImagingMainWindow()

    # Create data client
    # Fast sources (SUVI, AIA, HMI): 60 seconds
    # Slow sources (LASCO, EIT): 15 minutes
    client = SolarImageDataClient(
        fast_interval_ms=60000,
        slow_interval_ms=900000
    )

    # Connect window to client
    window.set_client(client)

    # Start fetching
    client.start()

    # Show window
    window.show()
    logger.info("Solar Imaging Display started successfully")

    # Handle Ctrl+C gracefully
    def signal_handler(signum, frame):
        logger.info("Received interrupt signal, shutting down...")
        client.stop()
        app.quit()

    signal.signal(signal.SIGINT, signal_handler)

    # Run event loop
    exit_code = app.exec()

    # Cleanup
    client.stop()
    logger.info("Solar Imaging Display closed")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
