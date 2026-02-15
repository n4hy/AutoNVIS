"""
GOES X-Ray Data Client

Fetches real-time X-ray flux data from NOAA SWPC to monitor solar flare activity.
This is the most critical data source for autonomous mode switching between
Quiet and Shock modes.
"""

import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.common.config import get_config
from src.common.constants import FLARE_CLASSES, M1_FLARE_THRESHOLD
from src.common.logging_config import ServiceLogger
from src.common.message_queue import MessageQueueClient, Topics


class GOESXRayClient:
    """
    Client for fetching GOES X-ray flux data from NOAA SWPC

    The GOES satellites monitor solar X-ray emissions in two wavelength bands:
    - Short: 0.05-0.4 nm (higher energy)
    - Long: 0.1-0.8 nm (lower energy)

    Flare classification is based on peak flux in the 0.1-0.8 nm band.
    """

    def __init__(
        self,
        api_url: str = None,
        update_interval: int = 60,
        mq_client: MessageQueueClient = None
    ):
        """
        Initialize GOES X-ray client

        Args:
            api_url: NOAA SWPC JSON API endpoint
            update_interval: Update interval in seconds
            mq_client: Message queue client (optional, will create if None)
        """
        config = get_config()

        self.api_url = api_url or config.data_sources.goes_xray_url
        self.update_interval = update_interval or config.data_sources.goes_update_interval

        self.logger = ServiceLogger("ingestion", "goes_xray")
        self.mq_client = mq_client

        self.last_fetch_time = None
        self.last_flux_value = None
        self.current_flare_class = None

    def classify_flare(self, flux: float) -> Tuple[str, int]:
        """
        Classify solar flare based on X-ray flux

        Args:
            flux: X-ray flux in W/m²

        Returns:
            (class_letter, class_number): e.g., ('M', 2) for M2 flare
        """
        if flux < 1e-8:
            return 'A', int(flux / 1e-9)

        for class_letter, (min_flux, max_flux) in FLARE_CLASSES.items():
            if min_flux <= flux < max_flux:
                # Calculate class number (0-9)
                class_number = int((flux / min_flux) % 10)
                return class_letter, class_number

        # X-class flares
        return 'X', int(flux / 1e-4)

    def format_flare_class(self, flux: float) -> str:
        """
        Format flare classification as string

        Args:
            flux: X-ray flux in W/m²

        Returns:
            Flare class string (e.g., 'M2.5', 'X9.3')
        """
        class_letter, class_number = self.classify_flare(flux)

        # Calculate precise magnitude
        if class_letter == 'A':
            magnitude = flux / 1e-9
        elif class_letter == 'B':
            magnitude = flux / 1e-8
        elif class_letter == 'C':
            magnitude = flux / 1e-7
        elif class_letter == 'M':
            magnitude = flux / 1e-6
        else:  # X
            magnitude = flux / 1e-5

        return f"{class_letter}{magnitude:.1f}"

    def is_m1_or_higher(self, flux: float) -> bool:
        """
        Check if flux indicates M1+ class flare (mode switching threshold)

        Args:
            flux: X-ray flux in W/m²

        Returns:
            True if M1 or higher
        """
        return flux >= M1_FLARE_THRESHOLD

    async def fetch_latest(self) -> Optional[Dict[str, Any]]:
        """
        Fetch latest GOES X-ray flux data from NOAA SWPC

        Returns:
            Dictionary with X-ray flux data, or None if fetch failed
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.api_url, timeout=10) as response:
                    if response.status != 200:
                        self.logger.error(
                            f"HTTP {response.status} from GOES API",
                            extra={'status_code': response.status}
                        )
                        return None

                    data = await response.json()

                    if not data or len(data) == 0:
                        self.logger.warning("Empty response from GOES API")
                        return None

                    # Get most recent measurement
                    latest = data[-1]

                    # Extract data
                    timestamp = latest.get('time_tag')
                    flux_short = float(latest.get('flux', 0))  # 0.05-0.4 nm

                    # The API provides short wavelength, but classification uses long
                    # For now, use short as proxy (actual GOES data has both)
                    flux_long = flux_short  # Approximate

                    self.last_fetch_time = datetime.utcnow()
                    self.last_flux_value = flux_long
                    self.current_flare_class = self.format_flare_class(flux_long)

                    result = {
                        'timestamp': timestamp,
                        'flux_short': flux_short,
                        'flux_long': flux_long,
                        'flare_class': self.current_flare_class,
                        'm1_or_higher': self.is_m1_or_higher(flux_long),
                        'source': 'GOES-Primary'
                    }

                    self.logger.info(
                        f"Fetched X-ray flux: {self.current_flare_class}",
                        extra={
                            'flux': flux_long,
                            'flare_class': self.current_flare_class,
                            'm1_threshold': self.is_m1_or_higher(flux_long)
                        }
                    )

                    return result

        except asyncio.TimeoutError:
            self.logger.error("Timeout fetching GOES X-ray data")
            return None
        except aiohttp.ClientError as e:
            self.logger.error(f"HTTP error fetching GOES data: {e}", exc_info=True)
            return None
        except (KeyError, ValueError, TypeError) as e:
            self.logger.error(f"Error parsing GOES data: {e}", exc_info=True)
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error fetching GOES data: {e}", exc_info=True)
            return None

    def publish_to_queue(self, data: Dict[str, Any]):
        """
        Publish X-ray flux data to message queue

        Args:
            data: X-ray flux data dictionary
        """
        if self.mq_client is None:
            self.logger.warning("No message queue client configured")
            return

        try:
            self.mq_client.publish(
                topic=Topics.WX_XRAY,
                data=data,
                source="goes_xray_client"
            )

            self.logger.debug(
                f"Published to {Topics.WX_XRAY}",
                extra={'flare_class': data.get('flare_class')}
            )

        except Exception as e:
            self.logger.error(f"Failed to publish to queue: {e}", exc_info=True)

    async def run_monitoring_loop(self):
        """
        Continuous monitoring loop - fetches data at regular intervals
        """
        self.logger.info(
            f"Starting GOES X-ray monitoring (interval: {self.update_interval}s)"
        )

        while True:
            try:
                data = await self.fetch_latest()

                if data:
                    self.publish_to_queue(data)

                    # Log M1+ events prominently
                    if data['m1_or_higher']:
                        self.logger.warning(
                            f"M1+ FLARE DETECTED: {data['flare_class']}",
                            extra={
                                'alert': 'M1_THRESHOLD',
                                'flare_class': data['flare_class'],
                                'flux': data['flux_long']
                            }
                        )
                else:
                    self.logger.warning("Failed to fetch GOES data, will retry")

                await asyncio.sleep(self.update_interval)

            except Exception as e:
                self.logger.error(
                    f"Error in monitoring loop: {e}",
                    exc_info=True
                )
                await asyncio.sleep(self.update_interval)

    async def run(self):
        """Start the GOES X-ray monitoring service"""
        self.logger.info("GOES X-ray client starting")

        # Initialize message queue if not provided
        if self.mq_client is None:
            config = get_config()
            self.mq_client = MessageQueueClient(
                host=config.services.rabbitmq_host,
                port=config.services.rabbitmq_port,
                username=config.services.rabbitmq_user,
                password=config.services.rabbitmq_password,
                vhost=config.services.rabbitmq_vhost
            )

        await self.run_monitoring_loop()


async def main():
    """Standalone entry point for GOES X-ray client"""
    client = GOESXRayClient()
    await client.run()


if __name__ == "__main__":
    asyncio.run(main())
