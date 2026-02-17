#!/usr/bin/env python3
"""
HF Ray Tracer Display

Visualizes ionospheric ray tracing for NVIS propagation analysis.
Shows LUF/MUF, ray paths, and coverage maps.
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from PyQt6.QtWidgets import QApplication

from src.visualization.pyqt.raytracer.main_window import RayTracerMainWindow


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("raytracer")

    logger.info("=" * 60)
    logger.info("HF Ray Tracer - NVIS Propagation Analysis")
    logger.info("=" * 60)

    try:
        app = QApplication(sys.argv)
        app.setApplicationName("HF Ray Tracer")

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

        window = RayTracerMainWindow()
        window.show()

        # Force initial paint
        app.processEvents()

        logger.info("Application started")

        exit_code = app.exec()

    except Exception as e:
        logger.error(f"Application error: {e}")
        import traceback
        traceback.print_exc()
        exit_code = 1

    sys.exit(exit_code)


if __name__ == '__main__':
    main()
