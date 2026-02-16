#!/usr/bin/env python3
"""
Space Weather Display - Direct from NOAA

No RabbitMQ. No WebSocket. No Dashboard.
Just fetches GOES X-ray from NOAA and displays.
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from PyQt6.QtWidgets import QApplication

from src.visualization.pyqt.spaceweather.main_window import SpaceWeatherMainWindow
from src.visualization.pyqt.spaceweather.direct_xray_client import DirectXRayClient


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("spaceweather")

    logger.info("=" * 60)
    logger.info("Space Weather - Direct from NOAA")
    logger.info("=" * 60)

    app = QApplication(sys.argv)
    app.setApplicationName("Space Weather")

    window = SpaceWeatherMainWindow()

    # Use direct client - fetches from NOAA every 60 seconds
    client = DirectXRayClient(update_interval_ms=60000)

    # Connect signals
    client.xray_received.connect(window._on_xray_update)
    client.xray_batch_received.connect(window._on_xray_batch)
    client.connected.connect(window._on_connected)
    client.disconnected.connect(window._on_disconnected)

    # Store client reference for cleanup
    window.ws_client = client

    client.start()
    window.show()

    logger.info("Application started - fetching from NOAA")

    exit_code = app.exec()

    if client.isRunning():
        client.stop()

    sys.exit(exit_code)


if __name__ == '__main__':
    main()
