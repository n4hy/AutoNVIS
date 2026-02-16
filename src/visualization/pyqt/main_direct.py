#!/usr/bin/env python3
"""
TEC Display - Direct from NOAA

No RabbitMQ. No WebSocket. No Dashboard.
Just fetches from NOAA and displays.
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from PyQt6.QtWidgets import QApplication

from src.visualization.pyqt.main_window import TECDisplayMainWindow
from src.visualization.pyqt.data.direct_glotec_client import DirectGloTECClient
from src.visualization.pyqt.themes.dark_theme import apply_dark_theme


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("tec_display")

    logger.info("=" * 60)
    logger.info("TEC Display - Direct from NOAA")
    logger.info("=" * 60)

    app = QApplication(sys.argv)
    app.setApplicationName("TEC Display")
    apply_dark_theme(app)

    window = TECDisplayMainWindow()

    # Use direct client - fetches from NOAA every 60 seconds
    client = DirectGloTECClient(update_interval_ms=60000)

    # Connect signals
    client.glotec_received.connect(window.data_manager.update_glotec_map)
    client.connected.connect(
        lambda: window.data_manager.set_connection_status('direct', True)
    )
    client.disconnected.connect(
        lambda: window.data_manager.set_connection_status('direct', False)
    )

    client.start()
    window.show()

    logger.info("Application started - fetching from NOAA")

    exit_code = app.exec()

    if client.isRunning():
        client.stop()

    sys.exit(exit_code)


if __name__ == '__main__':
    main()
