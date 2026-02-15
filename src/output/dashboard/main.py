"""
NVIS Analytics Dashboard Main Entry Point

Starts the FastAPI dashboard server.
"""

import sys
import argparse
from pathlib import Path
import uvicorn
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.common.config import get_config
from src.common.message_queue import MessageQueueClient
from src.output.dashboard.nvis_analytics_api import create_app, WebSocketManager
from src.common.logging_config import ServiceLogger, setup_logging
from src.output.dashboard.backend.state_manager import DashboardState
from src.output.dashboard.backend.subscribers import (
    GridDataSubscriber,
    PropagationSubscriber,
    SpaceWeatherSubscriber,
    ObservationSubscriber,
    SystemHealthSubscriber
)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='NVIS Analytics Dashboard')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8080, help='Port to bind to')
    parser.add_argument('--config', help='Path to config file')
    parser.add_argument('--no-mq', action='store_true', help='Disable message queue')
    args = parser.parse_args()

    # Set up logging
    setup_logging("dashboard", log_level="INFO", json_format=False)
    logger = ServiceLogger("dashboard_main")

    # Load configuration
    config = get_config(args.config)

    # Get grids
    lat_grid = config.grid.get_lat_grid()
    lon_grid = config.grid.get_lon_grid()
    alt_grid = config.grid.get_alt_grid()

    logger.info(f"Grid: {len(lat_grid)}×{len(lon_grid)}×{len(alt_grid)}")

    # Create message queue client (optional)
    mq_client = None
    subscribers = None

    if not args.no_mq:
        try:
            # Initialize dashboard state
            dashboard_state = DashboardState(retention_hours=24)
            logger.info("Dashboard state manager initialized")

            # Create WebSocket manager BEFORE subscribers
            ws_manager = WebSocketManager()
            logger.info("WebSocket manager initialized")

            # Initialize subscribers (each creates its own RabbitMQ connection)
            # WebSocket broadcast callback is set during construction
            grid_subscriber = GridDataSubscriber(
                config.services.rabbitmq_host,
                config.services.rabbitmq_port,
                config.services.rabbitmq_user,
                config.services.rabbitmq_password,
                config.services.rabbitmq_vhost,
                dashboard_state,
                ws_manager.broadcast
            )
            logger.info("GridDataSubscriber created with WebSocket callback")
            propagation_subscriber = PropagationSubscriber(
                config.services.rabbitmq_host,
                config.services.rabbitmq_port,
                config.services.rabbitmq_user,
                config.services.rabbitmq_password,
                config.services.rabbitmq_vhost,
                dashboard_state,
                ws_manager.broadcast
            )
            logger.info("PropagationSubscriber created with WebSocket callback")
            spaceweather_subscriber = SpaceWeatherSubscriber(
                config.services.rabbitmq_host,
                config.services.rabbitmq_port,
                config.services.rabbitmq_user,
                config.services.rabbitmq_password,
                config.services.rabbitmq_vhost,
                dashboard_state,
                ws_manager.broadcast
            )
            logger.info("SpaceWeatherSubscriber created with WebSocket callback")
            observation_subscriber = ObservationSubscriber(
                config.services.rabbitmq_host,
                config.services.rabbitmq_port,
                config.services.rabbitmq_user,
                config.services.rabbitmq_password,
                config.services.rabbitmq_vhost,
                dashboard_state,
                ws_manager.broadcast
            )
            logger.info("ObservationSubscriber created with WebSocket callback")
            health_subscriber = SystemHealthSubscriber(
                config.services.rabbitmq_host,
                config.services.rabbitmq_port,
                config.services.rabbitmq_user,
                config.services.rabbitmq_password,
                config.services.rabbitmq_vhost,
                dashboard_state,
                ws_manager.broadcast
            )
            logger.info("SystemHealthSubscriber created with WebSocket callback")

            # Create one connection for the main dashboard API (NVIS analytics)
            mq_client = MessageQueueClient(
                host=config.services.rabbitmq_host,
                port=config.services.rabbitmq_port,
                username=config.services.rabbitmq_user,
                password=config.services.rabbitmq_password,
                vhost=config.services.rabbitmq_vhost
            )
            logger.info("Main RabbitMQ connection created")

            subscribers = {
                'grid': grid_subscriber,
                'propagation': propagation_subscriber,
                'spaceweather': spaceweather_subscriber,
                'observation': observation_subscriber,
                'health': health_subscriber,
                'state': dashboard_state,
                'ws_manager': ws_manager
            }

            # Start all subscribers
            logger.info("Starting data subscribers...")
            grid_subscriber.start()
            propagation_subscriber.start()
            spaceweather_subscriber.start()
            observation_subscriber.start()
            health_subscriber.start()
            logger.info("All subscribers started successfully")

        except Exception as e:
            logger.error(f"Failed to initialize real-time updates: {type(e).__name__}: {e}")
            import traceback
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            logger.warning("Running without real-time updates")

    # Create FastAPI app
    app = create_app(lat_grid, lon_grid, alt_grid, mq_client, subscribers)

    logger.info(f"Starting dashboard on {args.host}:{args.port}")
    logger.info(f"Open http://{args.host}:{args.port} in your browser")

    # Run server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )


if __name__ == '__main__':
    main()
