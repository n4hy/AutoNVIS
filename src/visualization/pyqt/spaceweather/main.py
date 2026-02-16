#!/usr/bin/env python3
"""
AutoNVIS Space Weather Display

Real-time GOES X-ray flux visualization with flare class indicator.

Usage:
    python -m src.visualization.pyqt.spaceweather.main --ws-url ws://localhost:8080/ws
"""

import sys
import argparse
import logging
from pathlib import Path

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from src.visualization.pyqt.spaceweather.main_window import SpaceWeatherMainWindow
from src.visualization.pyqt.data.websocket_client import DashboardWebSocketClient


def main():
    """Main entry point for Space Weather display."""
    parser = argparse.ArgumentParser(
        description="AutoNVIS Space Weather - GOES X-Ray Monitor"
    )
    parser.add_argument(
        '--ws-url',
        default='ws://localhost:8080/ws',
        help='Dashboard WebSocket URL (default: ws://localhost:8080/ws)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger('spaceweather')

    # Create Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("AutoNVIS Space Weather")
    app.setOrganizationName("AutoNVIS")

    # Create main window
    window = SpaceWeatherMainWindow()

    # Create WebSocket client
    ws_client = DashboardWebSocketClient(url=args.ws_url)
    window.set_websocket_client(ws_client)

    # Auto-connect after window shows
    def auto_connect():
        logger.info(f"Connecting to {args.ws_url}")
        ws_client.start()
        window.connect_action.setChecked(True)

    QTimer.singleShot(500, auto_connect)

    # Show window
    window.show()
    logger.info("Space Weather display started")

    # Run event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
