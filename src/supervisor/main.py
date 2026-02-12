"""
Auto-NVIS Supervisor Service

Main entry point for the supervisor service. Coordinates mode control,
cycle orchestration, health monitoring, and alert generation.
"""

import asyncio
import signal
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.common.config import get_config
from src.common.logging_config import setup_logging, ServiceLogger
from src.common.message_queue import MessageQueueClient

from src.supervisor.mode_controller import ModeController
from src.supervisor.system_orchestrator import SystemOrchestrator
from src.supervisor.health_monitor import HealthMonitor
from src.supervisor.alert_generator import AlertGenerator


# FastAPI app for HTTP API
app = FastAPI(title="Auto-NVIS Supervisor", version="0.1.0")


class SupervisorService:
    """
    Main supervisor service coordinating all components
    """

    def __init__(self):
        """Initialize supervisor service"""
        self.config = get_config()
        self.logger = ServiceLogger("supervisor", "main")

        # Shared message queue client
        self.mq_client = None

        # Components
        self.mode_controller = None
        self.orchestrator = None
        self.health_monitor = None
        self.alert_generator = None

        # Task handles
        self.tasks = []

        # Shutdown event
        self.shutdown_event = asyncio.Event()

    async def initialize(self):
        """Initialize all components"""
        self.logger.info("Initializing supervisor service")

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

        # Initialize components
        self.mode_controller = ModeController(mq_client=self.mq_client)
        self.orchestrator = SystemOrchestrator(mq_client=self.mq_client)
        self.health_monitor = HealthMonitor()
        self.alert_generator = AlertGenerator(mq_client=self.mq_client)

        self.logger.info("All components initialized")

    async def start_components(self):
        """Start all supervisor components"""
        self.logger.info("Starting supervisor components")

        # Start mode controller (blocking, runs in background task)
        mode_task = asyncio.create_task(
            self.mode_controller.run(),
            name="mode_controller"
        )
        self.tasks.append(mode_task)
        self.logger.info("Mode controller started")

        # Start orchestrator
        orchestrator_task = asyncio.create_task(
            self.orchestrator.run(),
            name="orchestrator"
        )
        self.tasks.append(orchestrator_task)
        self.logger.info("Orchestrator started")

        # Start health monitor
        health_task = asyncio.create_task(
            self.health_monitor.run_monitoring_loop(),
            name="health_monitor"
        )
        self.tasks.append(health_task)
        self.logger.info("Health monitor started")

        self.logger.info(f"Started {len(self.tasks)} supervisor components")

    async def shutdown(self):
        """Graceful shutdown"""
        self.logger.info("Shutting down supervisor service")

        # Signal shutdown
        self.shutdown_event.set()

        # Cancel all tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()

        # Wait for tasks
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)

        # Close message queue
        if self.mq_client:
            self.mq_client.close()

        self.logger.info("Shutdown complete")

    async def run(self):
        """Main run loop"""
        self.logger.info("Auto-NVIS Supervisor Service starting")

        try:
            await self.initialize()
            await self.start_components()

            # Wait for shutdown signal
            await self.shutdown_event.wait()

        except Exception as e:
            self.logger.error(f"Fatal error in supervisor: {e}", exc_info=True)
            raise
        finally:
            await self.shutdown()


# Global supervisor instance
supervisor = SupervisorService()


# HTTP API endpoints

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse({
        "status": "healthy",
        "service": "supervisor",
        "version": "0.1.0"
    })


@app.get("/status")
async def get_status():
    """Get overall system status"""
    if supervisor.mode_controller is None:
        return JSONResponse(
            {"error": "Supervisor not initialized"},
            status_code=503
        )

    status = {
        "mode_controller": supervisor.mode_controller.get_status(),
        "orchestrator": supervisor.orchestrator.get_status(),
        "health_monitor": supervisor.health_monitor.get_status(),
        "alert_generator": supervisor.alert_generator.get_status()
    }

    return JSONResponse(status)


@app.get("/mode")
async def get_current_mode():
    """Get current operational mode"""
    if supervisor.mode_controller is None:
        return JSONResponse(
            {"error": "Supervisor not initialized"},
            status_code=503
        )

    mode = supervisor.mode_controller.get_current_mode()

    return JSONResponse({
        "current_mode": mode.value,
        "timestamp": supervisor.mode_controller.last_xray_timestamp,
        "last_flux": supervisor.mode_controller.last_xray_flux
    })


@app.post("/cycle/trigger")
async def trigger_cycle():
    """Manually trigger an update cycle"""
    if supervisor.orchestrator is None:
        return JSONResponse(
            {"error": "Supervisor not initialized"},
            status_code=503
        )

    # Trigger cycle asynchronously
    asyncio.create_task(supervisor.orchestrator.run_cycle())

    return JSONResponse({
        "message": "Cycle triggered",
        "cycle_count": supervisor.orchestrator.cycle_count + 1
    })


async def run_fastapi():
    """Run FastAPI server"""
    config = get_config()

    uvicorn_config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=config.services.supervisor_port,
        log_level="info"
    )

    server = uvicorn.Server(uvicorn_config)
    await server.serve()


async def main():
    """Main entry point"""
    # Set up logging
    setup_logging(
        service_name="supervisor",
        log_level="INFO",
        json_format=True
    )

    logger = ServiceLogger("supervisor", "main")
    logger.info("=" * 60)
    logger.info("Auto-NVIS Supervisor Service")
    logger.info("Version 0.1.0")
    logger.info("=" * 60)

    # Run supervisor and FastAPI concurrently
    await asyncio.gather(
        supervisor.run(),
        run_fastapi()
    )


def handle_signal(signum, frame):
    """Signal handler"""
    logger = ServiceLogger("supervisor", "main")
    logger.info(f"Received signal {signum}, initiating shutdown")
    asyncio.create_task(supervisor.shutdown())


if __name__ == "__main__":
    # Set up signal handlers
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)
