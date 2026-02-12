"""
Mode Controller

Autonomous mode switching between Quiet and Shock modes based on
real-time space weather conditions.

Modes:
- QUIET: Normal conditions, Gauss-Markov perturbation model
- SHOCK: Solar flare (M1+) detected, physics-based D-region absorption model
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from enum import Enum
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.common.config import get_config
from src.common.constants import M1_FLARE_THRESHOLD
from src.common.logging_config import ServiceLogger, MetricsLogger
from src.common.message_queue import MessageQueueClient, Topics, Message


class OperationalMode(Enum):
    """Operational modes for Auto-NVIS"""
    QUIET = "QUIET"
    SHOCK = "SHOCK"


class ModeController:
    """
    Autonomous mode controller for Auto-NVIS system

    The mode controller monitors GOES X-ray flux and switches between
    operational modes based on solar flare activity:

    QUIET → SHOCK: When X-ray flux exceeds M1 threshold
    SHOCK → QUIET: When flux drops below threshold for hysteresis period
    """

    def __init__(
        self,
        xray_threshold: float = None,
        hysteresis_sec: int = None,
        mq_client: MessageQueueClient = None
    ):
        """
        Initialize mode controller

        Args:
            xray_threshold: X-ray flux threshold for mode switching (W/m²)
            hysteresis_sec: Hysteresis time before switching back to quiet (seconds)
            mq_client: Message queue client
        """
        config = get_config()

        self.xray_threshold = xray_threshold or config.supervisor.xray_threshold_m1
        self.hysteresis_sec = hysteresis_sec or config.supervisor.mode_switch_hysteresis_sec

        self.logger = ServiceLogger("supervisor", "mode_controller")
        self.metrics = MetricsLogger("supervisor")

        self.current_mode = OperationalMode.QUIET
        self.mq_client = mq_client

        # State tracking
        self.last_xray_flux = None
        self.last_xray_timestamp = None
        self.flux_below_threshold_since = None
        self.mode_change_count = 0

        self.logger.info(
            f"Mode controller initialized: threshold={self.xray_threshold:.2e} W/m², "
            f"hysteresis={self.hysteresis_sec}s"
        )

    def get_current_mode(self) -> OperationalMode:
        """Get current operational mode"""
        return self.current_mode

    def should_switch_to_shock(self, flux: float) -> bool:
        """
        Determine if should switch to SHOCK mode

        Args:
            flux: Current X-ray flux (W/m²)

        Returns:
            True if should switch to SHOCK
        """
        if self.current_mode == OperationalMode.SHOCK:
            return False

        return flux >= self.xray_threshold

    def should_switch_to_quiet(self, flux: float) -> bool:
        """
        Determine if should switch back to QUIET mode

        Uses hysteresis to prevent oscillation:
        - Flux must stay below threshold for hysteresis_sec

        Args:
            flux: Current X-ray flux (W/m²)

        Returns:
            True if should switch to QUIET
        """
        if self.current_mode == OperationalMode.QUIET:
            return False

        # Check if flux is below threshold
        if flux >= self.xray_threshold:
            # Above threshold, reset hysteresis timer
            self.flux_below_threshold_since = None
            return False

        # Flux is below threshold
        if self.flux_below_threshold_since is None:
            # First time below threshold, start timer
            self.flux_below_threshold_since = datetime.utcnow()
            self.logger.info(
                f"Flux below threshold, starting hysteresis timer ({self.hysteresis_sec}s)"
            )
            return False

        # Check if enough time has passed
        elapsed = (datetime.utcnow() - self.flux_below_threshold_since).total_seconds()

        if elapsed >= self.hysteresis_sec:
            self.logger.info(
                f"Hysteresis period complete ({elapsed:.0f}s), ready to switch to QUIET"
            )
            return True

        self.logger.debug(
            f"Hysteresis in progress: {elapsed:.0f}s / {self.hysteresis_sec}s"
        )
        return False

    def switch_to_shock_mode(self, flux: float, flare_class: str):
        """
        Switch to SHOCK mode

        Args:
            flux: Current X-ray flux (W/m²)
            flare_class: Flare classification string
        """
        if self.current_mode == OperationalMode.SHOCK:
            self.logger.warning("Already in SHOCK mode")
            return

        self.logger.warning(
            f"🚨 SWITCHING TO SHOCK MODE 🚨",
            extra={
                'mode_change': 'QUIET→SHOCK',
                'flux': flux,
                'flare_class': flare_class,
                'threshold': self.xray_threshold
            }
        )

        self.current_mode = OperationalMode.SHOCK
        self.mode_change_count += 1
        self.flux_below_threshold_since = None

        # Publish mode change event
        self.publish_mode_change(
            old_mode=OperationalMode.QUIET,
            new_mode=OperationalMode.SHOCK,
            reason=f"M1+ flare detected: {flare_class}",
            flux=flux
        )

        # Log metric
        self.metrics.log_counter("mode_switches", labels={'to_mode': 'SHOCK'})

    def switch_to_quiet_mode(self, flux: float):
        """
        Switch to QUIET mode

        Args:
            flux: Current X-ray flux (W/m²)
        """
        if self.current_mode == OperationalMode.QUIET:
            self.logger.warning("Already in QUIET mode")
            return

        self.logger.info(
            f"Switching to QUIET mode",
            extra={
                'mode_change': 'SHOCK→QUIET',
                'flux': flux,
                'threshold': self.xray_threshold
            }
        )

        self.current_mode = OperationalMode.QUIET
        self.mode_change_count += 1
        self.flux_below_threshold_since = None

        # Publish mode change event
        self.publish_mode_change(
            old_mode=OperationalMode.SHOCK,
            new_mode=OperationalMode.QUIET,
            reason="X-ray flux below threshold (hysteresis complete)",
            flux=flux
        )

        # Log metric
        self.metrics.log_counter("mode_switches", labels={'to_mode': 'QUIET'})

    def publish_mode_change(
        self,
        old_mode: OperationalMode,
        new_mode: OperationalMode,
        reason: str,
        flux: float
    ):
        """
        Publish mode change event to message queue

        Args:
            old_mode: Previous mode
            new_mode: New mode
            reason: Reason for mode change
            flux: Current X-ray flux
        """
        if self.mq_client is None:
            self.logger.warning("No message queue client, cannot publish mode change")
            return

        data = {
            'old_mode': old_mode.value,
            'new_mode': new_mode.value,
            'reason': reason,
            'xray_flux': flux,
            'threshold': self.xray_threshold,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }

        try:
            self.mq_client.publish(
                topic=Topics.CTRL_MODE_CHANGE,
                data=data,
                source="mode_controller"
            )

            self.logger.info(f"Published mode change event: {old_mode.value}→{new_mode.value}")

        except Exception as e:
            self.logger.error(f"Failed to publish mode change: {e}", exc_info=True)

    def process_xray_data(self, message: Message):
        """
        Process incoming GOES X-ray data and make mode decisions

        Args:
            message: Message from wx.xray topic
        """
        try:
            data = message.data
            flux = data.get('flux_long', data.get('flux_short', 0))
            flare_class = data.get('flare_class', 'UNKNOWN')
            timestamp = data.get('timestamp')

            self.last_xray_flux = flux
            self.last_xray_timestamp = timestamp

            self.logger.debug(
                f"Processing X-ray data: {flare_class} ({flux:.2e} W/m²)",
                extra={'flux': flux, 'flare_class': flare_class}
            )

            # Check for mode switch
            if self.should_switch_to_shock(flux):
                self.switch_to_shock_mode(flux, flare_class)

            elif self.should_switch_to_quiet(flux):
                self.switch_to_quiet_mode(flux)

            # Log current mode metric
            self.metrics.log_gauge(
                "current_mode",
                1 if self.current_mode == OperationalMode.SHOCK else 0,
                labels={'mode': self.current_mode.value}
            )

            # Log flux metric
            self.metrics.log_gauge("xray_flux", flux)

        except Exception as e:
            self.logger.error(f"Error processing X-ray data: {e}", exc_info=True)

    async def run(self):
        """Run mode controller (subscribe to X-ray data)"""
        self.logger.info("Mode controller starting")

        if self.mq_client is None:
            config = get_config()
            self.mq_client = MessageQueueClient(
                host=config.services.rabbitmq_host,
                port=config.services.rabbitmq_port,
                username=config.services.rabbitmq_user,
                password=config.services.rabbitmq_password
            )

        # Subscribe to X-ray data
        self.mq_client.subscribe(
            topic_pattern=Topics.WX_XRAY,
            callback=self.process_xray_data,
            queue_name="mode_controller_xray"
        )

        self.logger.info("Subscribed to X-ray data, monitoring for mode changes")

        # Start consuming (blocking)
        self.mq_client.start_consuming()

    def get_status(self) -> Dict[str, Any]:
        """
        Get current mode controller status

        Returns:
            Status dictionary
        """
        return {
            'current_mode': self.current_mode.value,
            'last_xray_flux': self.last_xray_flux,
            'last_xray_timestamp': self.last_xray_timestamp,
            'threshold': self.xray_threshold,
            'hysteresis_sec': self.hysteresis_sec,
            'mode_change_count': self.mode_change_count,
            'flux_below_threshold_since': (
                self.flux_below_threshold_since.isoformat() + 'Z'
                if self.flux_below_threshold_since else None
            )
        }


async def main():
    """Standalone entry point for mode controller"""
    controller = ModeController()
    await controller.run()


if __name__ == "__main__":
    from src.common.logging_config import setup_logging
    setup_logging("supervisor", log_level="INFO", json_format=True)
    asyncio.run(main())
