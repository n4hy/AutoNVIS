"""
ACE Solar Wind Data Client

Fetches real-time solar wind parameters from the Advanced Composition Explorer (ACE)
spacecraft via NOAA SWPC. Monitors proton density, velocity, and magnetic field
to detect coronal mass ejections (CMEs) and geomagnetic storm conditions.
"""

import asyncio
import aiohttp
from datetime import datetime
from typing import Dict, Any, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.common.config import get_config
from src.common.logging_config import ServiceLogger
from src.common.message_queue import MessageQueueClient, Topics


class ACESolarWindClient:
    """
    Client for fetching ACE solar wind data from NOAA SWPC

    ACE monitors solar wind at the L1 Lagrange point, providing ~1 hour
    warning before solar wind structures reach Earth.

    Key parameters:
    - Proton density (particles/cm³)
    - Bulk velocity (km/s)
    - Temperature (K)
    - Magnetic field (Bx, By, Bz in nT)
    """

    def __init__(
        self,
        swepam_url: str = None,
        mag_url: str = None,
        update_interval: int = 60,
        mq_client: MessageQueueClient = None
    ):
        """
        Initialize ACE solar wind client

        Args:
            swepam_url: SWEPAM (plasma) data URL
            mag_url: Magnetometer data URL
            update_interval: Update interval in seconds
            mq_client: Message queue client
        """
        config = get_config()

        self.swepam_url = swepam_url or config.data_sources.ace_swepam_url
        self.mag_url = mag_url or config.data_sources.ace_mag_url
        self.update_interval = update_interval or config.data_sources.ace_update_interval

        self.logger = ServiceLogger("ingestion", "ace_solar_wind")
        self.mq_client = mq_client

        self.last_fetch_time = None

    def calculate_dynamic_pressure(self, density: float, velocity: float) -> float:
        """
        Calculate solar wind dynamic pressure

        P_dyn = ρ * v² * 1.6726e-6  [nPa]

        where ρ is in particles/cm³ and v is in km/s

        Args:
            density: Proton density (particles/cm³)
            velocity: Bulk velocity (km/s)

        Returns:
            Dynamic pressure (nPa)
        """
        return density * velocity**2 * 1.6726e-6

    def classify_solar_wind_speed(self, velocity: float) -> str:
        """
        Classify solar wind speed regime

        Args:
            velocity: Bulk velocity (km/s)

        Returns:
            Speed classification
        """
        if velocity < 300:
            return "very_slow"
        elif velocity < 400:
            return "slow"
        elif velocity < 500:
            return "moderate"
        elif velocity < 600:
            return "fast"
        elif velocity < 700:
            return "very_fast"
        else:
            return "extreme"

    def detect_cme_signature(
        self,
        density: float,
        velocity: float,
        temperature: float
    ) -> bool:
        """
        Detect potential CME signature in solar wind

        CME signatures:
        - High density (> 10 particles/cm³)
        - Variable velocity
        - Low temperature (cooler than expected for velocity)

        Args:
            density: Proton density (particles/cm³)
            velocity: Bulk velocity (km/s)
            temperature: Proton temperature (K)

        Returns:
            True if CME signature detected
        """
        # Expected temperature for given velocity (empirical)
        expected_temp = (velocity / 258.0)**2 * 1e5  # Rough approximation

        high_density = density > 10.0
        cool_plasma = temperature < expected_temp * 0.5

        return high_density and cool_plasma

    async def fetch_swepam_data(self) -> Optional[Dict[str, Any]]:
        """
        Fetch SWEPAM (plasma) data

        Returns:
            Plasma parameters or None
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.swepam_url, timeout=10) as response:
                    if response.status != 200:
                        self.logger.error(
                            f"HTTP {response.status} from ACE SWEPAM API",
                            extra={'status_code': response.status}
                        )
                        return None

                    data = await response.json()

                    if not data or len(data) == 0:
                        return None

                    # Get most recent measurement
                    latest = data[-1]

                    return {
                        'timestamp': latest.get('time_tag'),
                        'proton_density': float(latest.get('proton_density', 0)),
                        'bulk_speed': float(latest.get('bulk_speed', 0)),
                        'proton_temperature': float(latest.get('proton_temperature', 0))
                    }

        except Exception as e:
            self.logger.error(f"Error fetching SWEPAM data: {e}")
            return None

    async def fetch_mag_data(self) -> Optional[Dict[str, Any]]:
        """
        Fetch magnetometer data

        Returns:
            Magnetic field parameters or None
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.mag_url, timeout=10) as response:
                    if response.status != 200:
                        self.logger.error(
                            f"HTTP {response.status} from ACE MAG API",
                            extra={'status_code': response.status}
                        )
                        return None

                    data = await response.json()

                    if not data or len(data) == 0:
                        return None

                    # Get most recent measurement
                    latest = data[-1]

                    bx = float(latest.get('bx_gsm', 0))
                    by = float(latest.get('by_gsm', 0))
                    bz = float(latest.get('bz_gsm', 0))
                    bt = float(latest.get('bt', 0))

                    return {
                        'timestamp': latest.get('time_tag'),
                        'bx_gsm': bx,
                        'by_gsm': by,
                        'bz_gsm': bz,
                        'bt': bt
                    }

        except Exception as e:
            self.logger.error(f"Error fetching MAG data: {e}")
            return None

    async def fetch_latest(self) -> Optional[Dict[str, Any]]:
        """
        Fetch latest ACE solar wind data (both plasma and magnetic field)

        Returns:
            Combined solar wind data or None
        """
        try:
            # Fetch both datasets concurrently
            swepam_task = self.fetch_swepam_data()
            mag_task = self.fetch_mag_data()

            swepam_data, mag_data = await asyncio.gather(swepam_task, mag_task)

            if not swepam_data:
                self.logger.warning("No SWEPAM data available")
                return None

            # Magnetic field is optional
            if mag_data:
                result = {**swepam_data, **mag_data}
            else:
                result = swepam_data
                self.logger.warning("No MAG data available, using SWEPAM only")

            # Add derived parameters
            density = result.get('proton_density', 0)
            velocity = result.get('bulk_speed', 0)
            temperature = result.get('proton_temperature', 0)

            if density > 0 and velocity > 0:
                result['dynamic_pressure_npa'] = self.calculate_dynamic_pressure(
                    density, velocity
                )
                result['speed_class'] = self.classify_solar_wind_speed(velocity)

                if temperature > 0:
                    result['cme_signature'] = self.detect_cme_signature(
                        density, velocity, temperature
                    )

            result['source'] = 'ACE'

            self.last_fetch_time = datetime.utcnow()

            self.logger.info(
                f"Fetched solar wind: {velocity:.0f} km/s, "
                f"{density:.1f} p/cm³, Bz={result.get('bz_gsm', 0):.1f} nT",
                extra={
                    'velocity': velocity,
                    'density': density,
                    'bz_gsm': result.get('bz_gsm'),
                    'speed_class': result.get('speed_class')
                }
            )

            return result

        except Exception as e:
            self.logger.error(f"Error fetching ACE data: {e}", exc_info=True)
            return None

    def publish_to_queue(self, data: Dict[str, Any]):
        """
        Publish solar wind data to message queue

        Args:
            data: Solar wind data dictionary
        """
        if self.mq_client is None:
            self.logger.warning("No message queue client configured")
            return

        try:
            self.mq_client.publish(
                topic=Topics.WX_SOLAR_WIND,
                data=data,
                source="ace_solar_wind_client"
            )

            self.logger.debug(f"Published to {Topics.WX_SOLAR_WIND}")

        except Exception as e:
            self.logger.error(f"Failed to publish to queue: {e}", exc_info=True)

    async def run_monitoring_loop(self):
        """
        Continuous monitoring loop
        """
        self.logger.info(
            f"Starting ACE solar wind monitoring (interval: {self.update_interval}s)"
        )

        while True:
            try:
                data = await self.fetch_latest()

                if data:
                    self.publish_to_queue(data)

                    # Log significant events
                    if data.get('cme_signature'):
                        self.logger.warning(
                            "Possible CME signature detected",
                            extra={
                                'alert': 'CME_SIGNATURE',
                                'density': data.get('proton_density'),
                                'velocity': data.get('bulk_speed')
                            }
                        )

                    bz = data.get('bz_gsm', 0)
                    if bz < -10:
                        self.logger.warning(
                            f"Strong southward IMF: Bz={bz:.1f} nT (geomagnetic storm risk)",
                            extra={'alert': 'SOUTHWARD_BZ', 'bz_gsm': bz}
                        )

                else:
                    self.logger.warning("Failed to fetch ACE data, will retry")

                await asyncio.sleep(self.update_interval)

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}", exc_info=True)
                await asyncio.sleep(self.update_interval)

    async def run(self):
        """Start the ACE solar wind monitoring service"""
        self.logger.info("ACE solar wind client starting")

        if self.mq_client is None:
            config = get_config()
            self.mq_client = MessageQueueClient(
                host=config.services.rabbitmq_host,
                port=config.services.rabbitmq_port,
                username=config.services.rabbitmq_user,
                password=config.services.rabbitmq_password
            )

        await self.run_monitoring_loop()


async def main():
    """Standalone entry point for ACE solar wind client"""
    client = ACESolarWindClient()
    await client.run()


if __name__ == "__main__":
    asyncio.run(main())
