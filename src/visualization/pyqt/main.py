#!/usr/bin/env python3
"""
AutoNVIS TEC Display Application

PyQt6-based real-time TEC visualization connecting to the AutoNVIS dashboard.

Usage:
    python -m src.visualization.pyqt.main [options]

Options:
    --ws-url URL        Dashboard WebSocket URL (default: ws://localhost:8080/ws)
    --theme THEME       UI theme: dark or light (default: dark)
    --no-connect        Don't auto-connect on startup
"""

import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

from src.visualization.pyqt.main_window import TECDisplayMainWindow
from src.visualization.pyqt.data.websocket_client import DashboardWebSocketClient
from src.visualization.pyqt.themes.dark_theme import apply_dark_theme


def setup_logging(level: str = "INFO"):
    """Configure logging for the application."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def main():
    """Main entry point for TEC Display application."""
    parser = argparse.ArgumentParser(
        description='AutoNVIS TEC Display - Real-time TEC visualization'
    )
    parser.add_argument(
        '--ws-url',
        default='ws://localhost:8080/ws',
        help='Dashboard WebSocket URL (default: ws://localhost:8080/ws)'
    )
    parser.add_argument(
        '--theme',
        choices=['dark', 'light'],
        default='dark',
        help='UI theme (default: dark)'
    )
    parser.add_argument(
        '--no-connect',
        action='store_true',
        help="Don't auto-connect on startup"
    )
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    args = parser.parse_args()

    # Set up logging
    setup_logging(args.log_level)
    logger = logging.getLogger("tec_display")

    logger.info("=" * 60)
    logger.info("AutoNVIS TEC Display Application")
    logger.info("=" * 60)
    logger.info(f"WebSocket URL: {args.ws_url}")
    logger.info(f"Theme: {args.theme}")

    # Create Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("AutoNVIS TEC Display")
    app.setOrganizationName("AutoNVIS")

    # Apply theme
    if args.theme == 'dark':
        apply_dark_theme(app)
        logger.info("Applied dark theme")

    # Create main window
    window = TECDisplayMainWindow()

    # Create WebSocket client
    ws_client = DashboardWebSocketClient(url=args.ws_url)
    window.set_websocket_client(ws_client)

    # Auto-connect if not disabled
    if not args.no_connect:
        logger.info("Starting WebSocket connection...")
        ws_client.start()

    # Show window
    window.show()

    logger.info("Application started")

    # Run application
    exit_code = app.exec()

    # Cleanup
    if ws_client.isRunning():
        logger.info("Stopping WebSocket client...")
        ws_client.stop()

    logger.info("Application exiting")
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
