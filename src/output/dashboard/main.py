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
from src.output.dashboard.nvis_analytics_api import create_app
from src.common.logging_config import ServiceLogger
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
            mq_client = MessageQueueClient(
                host=config.services.rabbitmq_host,
                port=config.services.rabbitmq_port,
                username=config.services.rabbitmq_user,
                password=config.services.rabbitmq_password
            )
            logger.info("Connected to RabbitMQ")

            # Initialize dashboard state
            dashboard_state = DashboardState(retention_hours=24)
            logger.info("Dashboard state manager initialized")

            # Initialize subscribers
            # Note: WebSocket broadcast callback will be set after app creation
            grid_subscriber = GridDataSubscriber(mq_client, dashboard_state, None)
            propagation_subscriber = PropagationSubscriber(mq_client, dashboard_state, None)
            spaceweather_subscriber = SpaceWeatherSubscriber(mq_client, dashboard_state, None)
            observation_subscriber = ObservationSubscriber(mq_client, dashboard_state, None)
            health_subscriber = SystemHealthSubscriber(mq_client, dashboard_state, None)

            subscribers = {
                'grid': grid_subscriber,
                'propagation': propagation_subscriber,
                'spaceweather': spaceweather_subscriber,
                'observation': observation_subscriber,
                'health': health_subscriber,
                'state': dashboard_state
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
            logger.warning(f"Could not connect to RabbitMQ: {e}")
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
