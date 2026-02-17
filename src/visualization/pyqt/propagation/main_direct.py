#!/usr/bin/env python3
"""
HF Propagation Conditions Display - Direct from NOAA

Displays all four propagation indicators:
- X-ray flux (R-scale)
- Kp index (G-scale)
- Proton flux (S-scale)
- Solar wind Bz
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from PyQt6.QtWidgets import QApplication

from src.visualization.pyqt.propagation.main_window import PropagationMainWindow
from src.visualization.pyqt.propagation.data_client import PropagationDataClient


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("propagation")

    logger.info("=" * 60)
    logger.info("HF Propagation Conditions - Direct from NOAA")
    logger.info("=" * 60)

    client = None
    try:
        app = QApplication(sys.argv)
        app.setApplicationName("HF Propagation Conditions")

        # Apply dark theme
        app.setStyle("Fusion")
        from PyQt6.QtGui import QPalette, QColor
        from PyQt6.QtCore import Qt
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Base, QColor(35, 35, 35))
        palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
        app.setPalette(palette)

        window = PropagationMainWindow()

        # Create data client - fetches every 60 seconds
        client = PropagationDataClient(update_interval_ms=60000)

        # Connect to window
        window.set_data_client(client)

        client.start()
        window.show()

        # Force initial paint
        app.processEvents()

        logger.info("Application started - fetching from NOAA")

        exit_code = app.exec()

    except Exception as e:
        logger.error(f"Application error: {e}")
        exit_code = 1
    finally:
        try:
            if client and client.isRunning():
                client.stop()
        except Exception:
            pass

    sys.exit(exit_code)


if __name__ == '__main__':
    main()
