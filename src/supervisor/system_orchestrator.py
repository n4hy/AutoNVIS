"""
System Orchestrator

Coordinates the 15-minute update cycle for the Auto-NVIS system.
Triggers data assimilation, ray tracing, and output generation in sequence.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.common.config import get_config
from src.common.logging_config import ServiceLogger, MetricsLogger
from src.common.message_queue import MessageQueueClient, Topics


class CyclePhase:
    """Phases of the update cycle"""
    IDLE = "IDLE"
    SNAPSHOT = "SNAPSHOT"
    ASSIMILATION = "ASSIMILATION"
    PROPAGATION = "PROPAGATION"
    OUTPUT = "OUTPUT"
    COMPLETE = "COMPLETE"


class SystemOrchestrator:
    """
    Orchestrates the 15-minute update cycle

    Cycle phases:
    1. SNAPSHOT: Trigger data ingestion snapshot
    2. ASSIMILATION: Invoke SR-UKF filter step
    3. PROPAGATION: Trigger ray tracing with updated grid
    4. OUTPUT: Generate frequency plans and coverage maps
    5. COMPLETE: Publish alerts and metrics
    """

    def __init__(
        self,
        cycle_interval_sec: int = None,
        max_cycle_duration_sec: int = None,
        mq_client: MessageQueueClient = None
    ):
        """
        Initialize system orchestrator

        Args:
            cycle_interval_sec: Update cycle interval (seconds)
            max_cycle_duration_sec: Maximum allowed cycle duration (seconds)
            mq_client: Message queue client
        """
        config = get_config()

        self.cycle_interval = cycle_interval_sec or config.supervisor.update_cycle_sec
        self.max_cycle_duration = max_cycle_duration_sec or config.supervisor.max_cycle_duration_sec

        self.logger = ServiceLogger("supervisor", "orchestrator")
        self.metrics = MetricsLogger("supervisor")

        self.mq_client = mq_client
        self.scheduler = AsyncIOScheduler()

        # Cycle state
        self.current_phase = CyclePhase.IDLE
        self.cycle_count = 0
        self.last_cycle_start = None
        self.last_cycle_end = None
        self.last_cycle_duration = None
        self.cycle_in_progress = False

        # Statistics
        self.successful_cycles = 0
        self.failed_cycles = 0
        self.total_cycle_time = 0.0

        # Propagation service (initialized on demand)
        self.propagation_service = None
        self._ne_grid_cache = None  # Cache for latest Ne grid from filter

        self.logger.info(
            f"Orchestrator initialized: cycle={self.cycle_interval}s, "
            f"max_duration={self.max_cycle_duration}s"
        )

    async def trigger_snapshot(self) -> bool:
        """
        Trigger data ingestion snapshot

        Returns:
            True if successful
        """
        self.logger.info("Phase 1: Triggering data snapshot")
        self.current_phase = CyclePhase.SNAPSHOT

        # TODO: Implement actual snapshot trigger
        # For now, just wait briefly to simulate
        await asyncio.sleep(1)

        self.logger.info("Data snapshot complete")
        return True

    async def invoke_assimilation(self) -> bool:
        """
        Invoke SR-UKF assimilation step

        Returns:
            True if successful
        """
        self.logger.info("Phase 2: Invoking SR-UKF assimilation")
        self.current_phase = CyclePhase.ASSIMILATION

        try:
            # TODO: Implement gRPC call to assimilation service
            # For now, simulate
            await asyncio.sleep(2)

            self.logger.info("Assimilation complete")
            return True

        except Exception as e:
            self.logger.error(f"Assimilation failed: {e}", exc_info=True)
            return False

    async def trigger_propagation(self) -> bool:
        """
        Trigger ray tracing with updated electron density grid.

        Integrates native C++ ray tracer with SR-UKF filter output to
        calculate LUF/MUF and publish propagation products to RabbitMQ.

        Returns:
            True if successful
        """
        self.logger.info("Phase 3: Triggering propagation (ray tracing)")
        self.current_phase = CyclePhase.PROPAGATION

        try:
            config = get_config()

            # Import propagation service (deferred to avoid circular dependencies)
            from src.propagation.services import PropagationService

            # Initialize propagation service if not already done
            if self.propagation_service is None:
                self.logger.info("Initializing PropagationService...")
                self.propagation_service = PropagationService(
                    tx_lat=config.propagation.tx_lat,
                    tx_lon=config.propagation.tx_lon,
                    tx_alt=config.propagation.tx_alt_km,
                    freq_min=config.propagation.freq_min_mhz,
                    freq_max=config.propagation.freq_max_mhz,
                    freq_step=config.propagation.freq_step_mhz,
                    elevation_min=config.propagation.elevation_min_deg,
                    elevation_max=config.propagation.elevation_max_deg,
                    elevation_step=config.propagation.elevation_step_deg,
                    azimuth_step=config.propagation.azimuth_step_deg,
                    absorption_threshold_db=config.propagation.absorption_threshold_db,
                    snr_threshold_db=config.propagation.snr_threshold_db
                )

            # Get electron density grid from filter
            # NOTE: In full implementation, this would come from the assimilation service
            # via gRPC or message queue. For now, we create a placeholder.
            ne_grid, lat_grid, lon_grid, alt_grid, xray_flux = await self._get_ionospheric_grid()

            # Initialize ray tracer with current ionospheric grid
            self.logger.info("Initializing ray tracer with updated ionospheric grid...")
            self.propagation_service.initialize_ray_tracer(
                ne_grid=ne_grid,
                lat_grid=lat_grid,
                lon_grid=lon_grid,
                alt_grid=alt_grid,
                xray_flux=xray_flux
            )

            # Calculate LUF/MUF products
            self.logger.info("Calculating LUF/MUF coverage...")
            luf_muf_products = self.propagation_service.calculate_luf_muf()

            # Log results
            self.logger.info(
                f"LUF/MUF calculated: "
                f"LUF={luf_muf_products['luf_mhz']:.2f} MHz, "
                f"MUF={luf_muf_products['muf_mhz']:.2f} MHz, "
                f"FOT={luf_muf_products['fot_mhz']:.2f} MHz, "
                f"Blackout={luf_muf_products['blackout']}"
            )

            # Publish to message queue
            if self.mq_client:
                self.logger.info("Publishing propagation products to message queue...")

                # Add cycle metadata
                luf_muf_products['cycle_id'] = f"cycle_{self.cycle_count:04d}"

                await asyncio.to_thread(
                    self.mq_client.publish,
                    topic=Topics.OUT_FREQUENCY_PLAN,
                    data=luf_muf_products,
                    source="propagation"
                )

                self.logger.info("Propagation products published successfully")
            else:
                self.logger.warning("No message queue client - skipping publication")

            # Record metrics
            if hasattr(self, 'metrics'):
                self.metrics.record_metric(
                    "propagation_luf_mhz",
                    luf_muf_products['luf_mhz'],
                    {"cycle": self.cycle_count}
                )
                self.metrics.record_metric(
                    "propagation_muf_mhz",
                    luf_muf_products['muf_mhz'],
                    {"cycle": self.cycle_count}
                )
                self.metrics.record_metric(
                    "propagation_calculation_time_sec",
                    luf_muf_products['calculation_time_sec'],
                    {"cycle": self.cycle_count}
                )

            self.logger.info("Propagation complete")
            return True

        except Exception as e:
            self.logger.error(f"Propagation failed: {e}", exc_info=True)
            return False

    async def _get_ionospheric_grid(self):
        """
        Get ionospheric grid from SR-UKF filter.

        In full implementation, this would retrieve the grid from:
        1. Assimilation service via gRPC call, or
        2. Message queue (proc.grid_ready topic), or
        3. Shared memory / file system

        For now, returns cached grid or creates placeholder Chapman layer.

        Returns:
            Tuple of (ne_grid, lat_grid, lon_grid, alt_grid, xray_flux)
        """
        config = get_config()

        # Check if we have cached grid from assimilation phase
        if self._ne_grid_cache is not None:
            self.logger.debug("Using cached Ne grid from assimilation phase")
            return self._ne_grid_cache

        # TODO: Implement actual grid retrieval from assimilation service
        # Option 1: gRPC call to assimilation service
        # Option 2: Read from message queue (proc.grid_ready)
        # Option 3: Read from shared file system

        # For now, create a placeholder Chapman layer for testing
        self.logger.warning(
            "No Ne grid available from filter - creating placeholder Chapman layer"
        )

        import numpy as np

        lat_grid = config.grid.get_lat_grid()
        lon_grid = config.grid.get_lon_grid()
        alt_grid = config.grid.get_alt_grid()

        # Create Chapman layer ionosphere for testing
        Ne_max = 1e12  # el/m³ (moderate F-layer)
        h_max = 300.0  # km (F-layer peak)
        H = 50.0  # km (scale height)

        ne_grid = np.zeros((len(lat_grid), len(lon_grid), len(alt_grid)))
        for i, h in enumerate(alt_grid):
            z = (h - h_max) / H
            Ne = Ne_max * np.exp(1 - z - np.exp(-z))
            ne_grid[:, :, i] = Ne

        xray_flux = 1e-6  # W/m² (nominal)

        self.logger.info(
            f"Created Chapman layer: Ne_max={Ne_max:.2e} el/m³ at {h_max} km"
        )

        # Cache for next use
        self._ne_grid_cache = (ne_grid, lat_grid, lon_grid, alt_grid, xray_flux)

        return ne_grid, lat_grid, lon_grid, alt_grid, xray_flux

    async def generate_outputs(self) -> bool:
        """
        Generate output products (frequency plans, coverage maps)

        Returns:
            True if successful
        """
        self.logger.info("Phase 4: Generating output products")
        self.current_phase = CyclePhase.OUTPUT

        try:
            # TODO: Implement output generation
            # For now, simulate
            await asyncio.sleep(1)

            self.logger.info("Output generation complete")
            return True

        except Exception as e:
            self.logger.error(f"Output generation failed: {e}", exc_info=True)
            return False

    async def run_cycle(self):
        """
        Execute one complete update cycle

        This is the main orchestration logic that runs every 15 minutes
        """
        if self.cycle_in_progress:
            self.logger.warning(
                "Previous cycle still in progress, skipping this cycle",
                extra={'alert': 'CYCLE_OVERRUN'}
            )
            return

        self.cycle_in_progress = True
        self.cycle_count += 1
        self.last_cycle_start = datetime.utcnow()

        cycle_id = f"cycle_{self.cycle_count:04d}"

        self.logger.info(
            f"{'='*60}\n"
            f"Starting update cycle {self.cycle_count}\n"
            f"{'='*60}",
            extra={'cycle_id': cycle_id}
        )

        start_time = asyncio.get_event_loop().time()
        success = True

        try:
            # Phase 1: Snapshot
            if not await self.trigger_snapshot():
                self.logger.error("Snapshot phase failed")
                success = False
                raise Exception("Snapshot failed")

            # Phase 2: Assimilation
            if not await self.invoke_assimilation():
                self.logger.error("Assimilation phase failed")
                success = False
                raise Exception("Assimilation failed")

            # Phase 3: Propagation
            if not await self.trigger_propagation():
                self.logger.error("Propagation phase failed")
                success = False
                raise Exception("Propagation failed")

            # Phase 4: Output
            if not await self.generate_outputs():
                self.logger.error("Output phase failed")
                success = False
                raise Exception("Output generation failed")

            # Phase 5: Complete
            self.current_phase = CyclePhase.COMPLETE

        except Exception as e:
            self.logger.error(f"Cycle {cycle_id} failed: {e}", exc_info=True)
            success = False

        finally:
            # Calculate duration
            end_time = asyncio.get_event_loop().time()
            duration = end_time - start_time

            self.last_cycle_end = datetime.utcnow()
            self.last_cycle_duration = duration
            self.cycle_in_progress = False
            self.current_phase = CyclePhase.IDLE

            # Update statistics
            if success:
                self.successful_cycles += 1
            else:
                self.failed_cycles += 1

            self.total_cycle_time += duration

            # Log summary
            self.logger.info(
                f"Cycle {cycle_id} {'COMPLETED' if success else 'FAILED'} "
                f"in {duration:.1f}s",
                extra={
                    'cycle_id': cycle_id,
                    'duration_sec': duration,
                    'success': success,
                    'phase': self.current_phase
                }
            )

            # Check for overrun
            if duration > self.max_cycle_duration:
                self.logger.warning(
                    f"Cycle duration ({duration:.1f}s) exceeded maximum "
                    f"({self.max_cycle_duration}s)",
                    extra={'alert': 'CYCLE_DURATION_EXCEEDED'}
                )

            # Log metrics
            self.metrics.log_histogram("cycle_duration", duration)
            self.metrics.log_counter(
                "cycles",
                labels={'success': str(success)}
            )

    def start_scheduling(self):
        """Start the scheduled update cycles"""
        self.logger.info(f"Starting scheduled cycles (interval: {self.cycle_interval}s)")

        # Add job to scheduler
        self.scheduler.add_job(
            self.run_cycle,
            trigger=IntervalTrigger(seconds=self.cycle_interval),
            id='update_cycle',
            name='Auto-NVIS Update Cycle',
            max_instances=1,  # Prevent overlapping
            coalesce=True     # Combine missed runs
        )

        # Start scheduler
        self.scheduler.start()

        self.logger.info("Scheduler started")

    async def run(self):
        """Run the orchestrator"""
        self.logger.info("System orchestrator starting")

        if self.mq_client is None:
            config = get_config()
            self.mq_client = MessageQueueClient(
                host=config.services.rabbitmq_host,
                port=config.services.rabbitmq_port,
                username=config.services.rabbitmq_user,
                password=config.services.rabbitmq_password
            )

        # Start scheduling
        self.start_scheduling()

        # Keep running
        try:
            while True:
                await asyncio.sleep(60)
                # Could add periodic health checks here

        except asyncio.CancelledError:
            self.logger.info("Orchestrator shutdown requested")
            self.scheduler.shutdown()

    def get_status(self) -> Dict[str, Any]:
        """
        Get orchestrator status

        Returns:
            Status dictionary
        """
        avg_cycle_time = (
            self.total_cycle_time / self.successful_cycles
            if self.successful_cycles > 0 else 0
        )

        return {
            'cycle_interval_sec': self.cycle_interval,
            'current_phase': self.current_phase,
            'cycle_count': self.cycle_count,
            'successful_cycles': self.successful_cycles,
            'failed_cycles': self.failed_cycles,
            'cycle_in_progress': self.cycle_in_progress,
            'last_cycle_start': (
                self.last_cycle_start.isoformat() + 'Z'
                if self.last_cycle_start else None
            ),
            'last_cycle_end': (
                self.last_cycle_end.isoformat() + 'Z'
                if self.last_cycle_end else None
            ),
            'last_cycle_duration_sec': self.last_cycle_duration,
            'avg_cycle_duration_sec': avg_cycle_time
        }


async def main():
    """Standalone entry point for orchestrator"""
    orchestrator = SystemOrchestrator()
    await orchestrator.run()


if __name__ == "__main__":
    from src.common.logging_config import setup_logging
    setup_logging("supervisor", log_level="INFO", json_format=True)
    asyncio.run(main())
