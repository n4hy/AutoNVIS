"""
Alert Generator

Generates operational alerts for blackouts, fadeouts, and system issues.
"""

from datetime import datetime
from typing import Dict, Any, Optional
from enum import Enum
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.common.logging_config import ServiceLogger
from src.common.message_queue import MessageQueueClient, Topics


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class AlertType(Enum):
    """Types of alerts"""
    BLACKOUT = "BLACKOUT"
    FADEOUT_WARNING = "FADEOUT_WARNING"
    M1_FLARE = "M1_FLARE"
    MODE_CHANGE = "MODE_CHANGE"
    SERVICE_FAILURE = "SERVICE_FAILURE"
    DATA_STALE = "DATA_STALE"
    CYCLE_OVERRUN = "CYCLE_OVERRUN"


class AlertGenerator:
    """
    Generates and publishes operational alerts
    """

    def __init__(self, mq_client: MessageQueueClient = None):
        """
        Initialize alert generator

        Args:
            mq_client: Message queue client
        """
        self.logger = ServiceLogger("supervisor", "alert_generator")
        self.mq_client = mq_client

        # Alert counters
        self.alerts_generated = 0
        self.alerts_by_type = {}

    def create_alert(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create alert dictionary

        Args:
            alert_type: Type of alert
            severity: Severity level
            message: Alert message
            details: Additional details

        Returns:
            Alert dictionary
        """
        alert = {
            'alert_id': f"{alert_type.value}_{self.alerts_generated:06d}",
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'type': alert_type.value,
            'severity': severity.value,
            'message': message,
            'details': details or {}
        }

        return alert

    def detect_blackout(self, luf: float, muf: float) -> Optional[Dict[str, Any]]:
        """
        Detect NVIS blackout condition (LUF > MUF)

        Args:
            luf: Lowest Usable Frequency (MHz)
            muf: Maximum Usable Frequency (MHz)

        Returns:
            Alert if blackout detected, None otherwise
        """
        if luf > muf:
            message = (
                f"NVIS BLACKOUT: LUF ({luf:.1f} MHz) exceeds MUF ({muf:.1f} MHz). "
                f"No usable NVIS frequencies available."
            )

            alert = self.create_alert(
                alert_type=AlertType.BLACKOUT,
                severity=AlertSeverity.CRITICAL,
                message=message,
                details={
                    'luf_mhz': luf,
                    'muf_mhz': muf,
                    'usable_window_mhz': 0.0
                }
            )

            self.logger.critical(message, extra={'alert': alert})
            return alert

        return None

    def detect_fadeout(self, luf: float, luf_trend: float) -> Optional[Dict[str, Any]]:
        """
        Detect impending fadeout (rapidly rising LUF)

        Args:
            luf: Current LUF (MHz)
            luf_trend: LUF rate of change (MHz/min)

        Returns:
            Alert if fadeout risk detected, None otherwise
        """
        # Significant rising trend (> 0.5 MHz/min)
        if luf_trend > 0.5:
            message = (
                f"FADEOUT WARNING: LUF rising rapidly ({luf_trend:.2f} MHz/min). "
                f"Current LUF: {luf:.1f} MHz. Link degradation imminent."
            )

            alert = self.create_alert(
                alert_type=AlertType.FADEOUT_WARNING,
                severity=AlertSeverity.WARNING,
                message=message,
                details={
                    'luf_mhz': luf,
                    'luf_trend_mhz_per_min': luf_trend
                }
            )

            self.logger.warning(message, extra={'alert': alert})
            return alert

        return None

    def generate_m1_flare_alert(self, flux: float, flare_class: str) -> Dict[str, Any]:
        """
        Generate alert for M1+ flare detection

        Args:
            flux: X-ray flux (W/m²)
            flare_class: Flare classification

        Returns:
            Alert dictionary
        """
        message = (
            f"M1+ SOLAR FLARE DETECTED: {flare_class} "
            f"({flux:.2e} W/m²). Expect D-region absorption increase."
        )

        alert = self.create_alert(
            alert_type=AlertType.M1_FLARE,
            severity=AlertSeverity.WARNING,
            message=message,
            details={
                'flux_w_per_m2': flux,
                'flare_class': flare_class
            }
        )

        return alert

    def generate_mode_change_alert(
        self,
        old_mode: str,
        new_mode: str,
        reason: str
    ) -> Dict[str, Any]:
        """
        Generate alert for mode change

        Args:
            old_mode: Previous operational mode
            new_mode: New operational mode
            reason: Reason for change

        Returns:
            Alert dictionary
        """
        message = f"MODE CHANGE: {old_mode} → {new_mode}. Reason: {reason}"

        alert = self.create_alert(
            alert_type=AlertType.MODE_CHANGE,
            severity=AlertSeverity.INFO,
            message=message,
            details={
                'old_mode': old_mode,
                'new_mode': new_mode,
                'reason': reason
            }
        )

        return alert

    def generate_service_failure_alert(
        self,
        service_name: str,
        error_message: str
    ) -> Dict[str, Any]:
        """
        Generate alert for service failure

        Args:
            service_name: Name of failed service
            error_message: Error description

        Returns:
            Alert dictionary
        """
        message = f"SERVICE FAILURE: {service_name} is not responding. {error_message}"

        alert = self.create_alert(
            alert_type=AlertType.SERVICE_FAILURE,
            severity=AlertSeverity.CRITICAL,
            message=message,
            details={
                'service': service_name,
                'error': error_message
            }
        )

        return alert

    def publish_alert(self, alert: Dict[str, Any]):
        """
        Publish alert to message queue

        Args:
            alert: Alert dictionary
        """
        if self.mq_client is None:
            self.logger.warning("No message queue client, cannot publish alert")
            return

        try:
            self.mq_client.publish(
                topic=Topics.OUT_ALERT,
                data=alert,
                source="alert_generator"
            )

            # Update counters
            self.alerts_generated += 1
            alert_type = alert['type']
            self.alerts_by_type[alert_type] = self.alerts_by_type.get(alert_type, 0) + 1

            self.logger.info(
                f"Published {alert['severity']} alert: {alert['type']}",
                extra={'alert_id': alert['alert_id']}
            )

        except Exception as e:
            self.logger.error(f"Failed to publish alert: {e}", exc_info=True)

    def get_status(self) -> Dict[str, Any]:
        """
        Get alert generator status

        Returns:
            Status dictionary
        """
        return {
            'alerts_generated': self.alerts_generated,
            'alerts_by_type': self.alerts_by_type
        }
