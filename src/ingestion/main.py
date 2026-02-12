"""
Auto-NVIS Data Ingestion Service

Main entry point for all data ingestion services. Orchestrates concurrent
monitoring of space weather, GNSS-TEC, and ionosonde data sources.
"""

import asyncio
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.common.config import get_config
from src.common.logging_config import setup_logging, ServiceLogger
from src.common.message_queue import MessageQueueClient
from src.ingestion.space_weather.goes_xray_client import GOESXRayClient
from src.ingestion.space_weather.ace_solar_wind import ACESolarWindClient


class IngestionOrchestrator:
    """
    Orchestrates all data ingestion services
    """

    def __init__(self):
        """Initialize ingestion orchestrator"""
        self.config = get_config()
        self.logger = ServiceLogger("ingestion", "orchestrator")

        # Shared message queue client
        self.mq_client = None

        # Client instances
        self.goes_client = None
        self.ace_client = None

        # Task handles
        self.tasks = []

        # Shutdown flag
        self.shutdown_event = asyncio.Event()

    async def initialize(self):
        """Initialize all components"""
        self.logger.info("Initializing ingestion services")

        # Create shared message queue client
        try:
            self.mq_client = MessageQueueClient(
                host=self.config.services.rabbitmq_host,
                port=self.config.services.rabbitmq_port,
                username=self.config.services.rabbitmq_user,
                password=self.config.services.rabbitmq_password
            )
            self.logger.info("Message queue client initialized")

        except Exception as e:
            self.logger.error(
                f"Failed to initialize message queue: {e}",
                exc_info=True
            )
            raise

        # Initialize clients
        self.goes_client = GOESXRayClient(mq_client=self.mq_client)
        self.ace_client = ACESolarWindClient(mq_client=self.mq_client)

        self.logger.info("All clients initialized")

    async def start_services(self):
        """Start all monitoring services"""
        self.logger.info("Starting ingestion services")

        # Start GOES X-ray monitoring
        goes_task = asyncio.create_task(
            self.goes_client.run_monitoring_loop(),
            name="goes_xray"
        )
        self.tasks.append(goes_task)
        self.logger.info("GOES X-ray monitoring started")

        # Start ACE solar wind monitoring
        ace_task = asyncio.create_task(
            self.ace_client.run_monitoring_loop(),
            name="ace_solar_wind"
        )
        self.tasks.append(ace_task)
        self.logger.info("ACE solar wind monitoring started")

        # TODO: Add GNSS-TEC monitoring when implemented
        # TODO: Add ionosonde monitoring when implemented

        self.logger.info(f"Started {len(self.tasks)} ingestion services")

    async def monitor_tasks(self):
        """Monitor running tasks and restart on failure"""
        while not self.shutdown_event.is_set():
            # Check for completed/failed tasks
            for task in self.tasks:
                if task.done():
                    task_name = task.get_name()

                    try:
                        # This will raise exception if task failed
                        task.result()
                        self.logger.warning(
                            f"Task {task_name} completed unexpectedly"
                        )
                    except Exception as e:
                        self.logger.error(
                            f"Task {task_name} failed: {e}",
                            exc_info=True
                        )

                    # TODO: Implement task restart logic
                    self.logger.warning(f"Task {task_name} needs restart (not yet implemented)")

            await asyncio.sleep(10)

    async def shutdown(self):
        """Graceful shutdown of all services"""
        self.logger.info("Shutting down ingestion services")

        # Signal shutdown
        self.shutdown_event.set()

        # Cancel all tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()

        # Wait for tasks to complete
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)

        # Close message queue
        if self.mq_client:
            self.mq_client.close()

        self.logger.info("Shutdown complete")

    async def run(self):
        """Main run loop"""
        self.logger.info("Auto-NVIS Ingestion Service starting")

        try:
            # Initialize
            await self.initialize()

            # Start all services
            await self.start_services()

            # Monitor tasks
            await self.monitor_tasks()

        except asyncio.CancelledError:
            self.logger.info("Received cancellation signal")
        except Exception as e:
            self.logger.error(f"Fatal error in ingestion service: {e}", exc_info=True)
            raise
        finally:
            await self.shutdown()


def handle_signal(orchestrator: IngestionOrchestrator):
    """Signal handler for graceful shutdown"""
    def _handler(signum, frame):
        logger = ServiceLogger("ingestion", "main")
        logger.info(f"Received signal {signum}, initiating shutdown")
        asyncio.create_task(orchestrator.shutdown())

    return _handler


async def main():
    """Main entry point"""
    # Set up logging
    setup_logging(
        service_name="ingestion",
        log_level="INFO",
        json_format=True
    )

    logger = ServiceLogger("ingestion", "main")
    logger.info("=" * 60)
    logger.info("Auto-NVIS Data Ingestion Service")
    logger.info("Version 0.1.0")
    logger.info("=" * 60)

    # Create orchestrator
    orchestrator = IngestionOrchestrator()

    # Set up signal handlers
    signal.signal(signal.SIGINT, handle_signal(orchestrator))
    signal.signal(signal.SIGTERM, handle_signal(orchestrator))

    # Run
    await orchestrator.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)
