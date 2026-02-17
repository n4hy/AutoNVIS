#!/usr/bin/env python3
"""
Enhanced HF Propagation Display - Entry Point

Launches the complete space weather dashboard combining:
- Standard Four (X-Ray, Kp, Proton, Solar Wind Bz)
- Advanced Ionospheric (F10.7, Ionosonde, HPI, D-RAP)
- Predictions (WSA-Enlil, Propagated Solar Wind)

This provides the foundation for exceeding standard PHaRLAP capabilities
through real-time data integration beyond monthly median climatology.

Usage:
    python -m propagation.enhanced_main
"""

import sys
import logging
from pathlib import Path

# Configure logging before imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("enhanced_propagation")


def main():
    """Launch the enhanced propagation display."""
    try:
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtCore import Qt
    except ImportError as e:
        logger.error(f"PyQt6 not found: {e}")
        logger.error("Install with: pip install PyQt6 pyqtgraph aiohttp")
        sys.exit(1)

    try:
        from .enhanced_main_window import EnhancedPropagationWindow
    except ImportError:
        # Direct run
        from enhanced_main_window import EnhancedPropagationWindow

    logger.info("Starting Enhanced HF Propagation Display...")
    logger.info("Data sources:")
    logger.info("  Standard Four: X-Ray, Kp, Proton Flux, Solar Wind Bz")
    logger.info("  Advanced: F10.7, GIRO Ionosonde, HPI, D-RAP")
    logger.info("  Predictions: WSA-Enlil, Propagated Solar Wind")

    app = QApplication(sys.argv)

    # Enable high DPI scaling
    app.setStyle('Fusion')

    window = EnhancedPropagationWindow()
    window.show()

    logger.info("Display ready. Fetching data...")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
