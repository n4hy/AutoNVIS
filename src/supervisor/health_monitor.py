"""
Health Monitor

Monitors the health of all Auto-NVIS services and data sources.
Detects failures and stale data.
"""

import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.common.config import get_config
from src.common.logging_config import ServiceLogger, MetricsLogger


@dataclass
class ServiceHealth:
    """Health status for a service"""
    name: str
    status: str  # 'healthy', 'degraded', 'unhealthy', 'unknown'
    last_check: datetime
    last_success: Optional[datetime]
    consecutive_failures: int
    message: str


class HealthMonitor:
    """
    Monitors health of all Auto-NVIS services
    """

    def __init__(self, check_interval_sec: int = None):
        """
        Initialize health monitor

        Args:
            check_interval_sec: Interval between health checks
        """
        config = get_config()
        self.check_interval = check_interval_sec or config.supervisor.health_check_interval_sec

        self.logger = ServiceLogger("supervisor", "health_monitor")
        self.metrics = MetricsLogger("supervisor")

        # Service endpoints
        self.services = {
            'ingestion': f"http://{config.services.rabbitmq_host}:15672/api/overview",
            'assimilation': f"http://{config.services.assimilation_host}:{config.services.assimilation_port}/health",
            'output': f"http://{config.services.output_host}:{config.services.output_port}/health"
        }

        # Health state
        self.service_health: Dict[str, ServiceHealth] = {}
        for name in self.services.keys():
            self.service_health[name] = ServiceHealth(
                name=name,
                status='unknown',
                last_check=datetime.utcnow(),
                last_success=None,
                consecutive_failures=0,
                message="Not yet checked"
            )

    async def check_service_http(self, name: str, url: str) -> ServiceHealth:
        """
        Check service health via HTTP endpoint

        Args:
            name: Service name
            url: Health check URL

        Returns:
            ServiceHealth status
        """
        health = self.service_health[name]
        health.last_check = datetime.utcnow()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=5) as response:
                    if response.status == 200:
                        health.status = 'healthy'
                        health.last_success = datetime.utcnow()
                        health.consecutive_failures = 0
                        health.message = "OK"

                        self.logger.debug(f"{name} is healthy")

                    else:
                        health.consecutive_failures += 1

                        if health.consecutive_failures >= 3:
                            health.status = 'unhealthy'
                        else:
                            health.status = 'degraded'

                        health.message = f"HTTP {response.status}"

                        self.logger.warning(
                            f"{name} health check failed: HTTP {response.status}",
                            extra={'service': name, 'status_code': response.status}
                        )

        except asyncio.TimeoutError:
            health.consecutive_failures += 1
            health.status = 'unhealthy' if health.consecutive_failures >= 3 else 'degraded'
            health.message = "Timeout"

            self.logger.warning(f"{name} health check timeout", extra={'service': name})

        except aiohttp.ClientError as e:
            health.consecutive_failures += 1
            health.status = 'unhealthy' if health.consecutive_failures >= 3 else 'degraded'
            health.message = f"Connection error: {str(e)[:50]}"

            self.logger.warning(
                f"{name} health check failed: {e}",
                extra={'service': name}
            )

        except Exception as e:
            health.consecutive_failures += 1
            health.status = 'unhealthy'
            health.message = f"Error: {str(e)[:50]}"

            self.logger.error(
                f"{name} health check error: {e}",
                extra={'service': name},
                exc_info=True
            )

        # Log metric
        status_code = 1 if health.status == 'healthy' else 0
        self.metrics.log_gauge(
            "service_health",
            status_code,
            labels={'service': name}
        )

        return health

    async def check_all_services(self):
        """Check health of all services"""
        self.logger.debug("Running health checks")

        tasks = []
        for name, url in self.services.items():
            task = self.check_service_http(name, url)
            tasks.append(task)

        await asyncio.gather(*tasks, return_exceptions=True)

        # Log overall health
        unhealthy = [
            name for name, health in self.service_health.items()
            if health.status == 'unhealthy'
        ]

        if unhealthy:
            self.logger.warning(
                f"Unhealthy services: {', '.join(unhealthy)}",
                extra={'alert': 'UNHEALTHY_SERVICES', 'services': unhealthy}
            )

    def get_overall_health(self) -> str:
        """
        Get overall system health status

        Returns:
            'healthy', 'degraded', or 'unhealthy'
        """
        statuses = [h.status for h in self.service_health.values()]

        if 'unhealthy' in statuses:
            return 'unhealthy'
        elif 'degraded' in statuses:
            return 'degraded'
        elif all(s == 'healthy' for s in statuses):
            return 'healthy'
        else:
            return 'unknown'

    async def run_monitoring_loop(self):
        """Continuous health monitoring loop"""
        self.logger.info(f"Starting health monitoring (interval: {self.check_interval}s)")

        while True:
            try:
                await self.check_all_services()
                await asyncio.sleep(self.check_interval)

            except Exception as e:
                self.logger.error(f"Error in health monitoring loop: {e}", exc_info=True)
                await asyncio.sleep(self.check_interval)

    def get_status(self) -> Dict[str, Any]:
        """
        Get health monitor status

        Returns:
            Status dictionary with all service health
        """
        return {
            'overall_health': self.get_overall_health(),
            'check_interval_sec': self.check_interval,
            'services': {
                name: {
                    'status': health.status,
                    'last_check': health.last_check.isoformat() + 'Z',
                    'last_success': (
                        health.last_success.isoformat() + 'Z'
                        if health.last_success else None
                    ),
                    'consecutive_failures': health.consecutive_failures,
                    'message': health.message
                }
                for name, health in self.service_health.items()
            }
        }


async def main():
    """Standalone entry point for health monitor"""
    monitor = HealthMonitor()
    await monitor.run_monitoring_loop()


if __name__ == "__main__":
    from src.common.logging_config import setup_logging
    setup_logging("supervisor", log_level="INFO", json_format=True)
    asyncio.run(main())
