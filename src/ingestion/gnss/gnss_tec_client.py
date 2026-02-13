"""
GNSS-TEC Data Client

High-level client for real-time GNSS-TEC data ingestion.
Connects to NTRIP streams, processes RTCM3 messages, calculates TEC,
and publishes measurements to message queue.
"""

import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.common.config import get_config
from src.common.logging_config import ServiceLogger
from src.common.message_queue import MessageQueueClient, Topics
from src.ingestion.common.data_validator import DataValidator
from .ntrip_client import NTRIPClient
from .rtcm3_parser import RTCM3Parser
from .tec_calculator import TECCalculator, ReceiverPosition


class GNSSTECClient:
    """
    GNSS-TEC data ingestion client

    Complete pipeline for real-time TEC data:
    1. Connect to NTRIP stream (IGS)
    2. Parse RTCM3 messages
    3. Calculate TEC from dual-frequency observables
    4. Validate measurements
    5. Publish to message queue
    """

    def __init__(
        self,
        ntrip_host: str = None,
        ntrip_port: int = None,
        ntrip_mountpoint: str = None,
        ntrip_username: str = None,
        ntrip_password: str = None,
        receiver_lat: float = None,
        receiver_lon: float = None,
        receiver_alt: float = None,
        mq_client: MessageQueueClient = None
    ):
        """
        Initialize GNSS-TEC client

        Args:
            ntrip_host: NTRIP caster hostname
            ntrip_port: NTRIP caster port
            ntrip_mountpoint: NTRIP mountpoint
            ntrip_username: NTRIP username (optional)
            ntrip_password: NTRIP password (optional)
            receiver_lat: Receiver latitude (degrees, optional)
            receiver_lon: Receiver longitude (degrees, optional)
            receiver_alt: Receiver altitude (meters, optional)
            mq_client: Message queue client (optional, will create if None)
        """
        config = get_config()

        # NTRIP configuration
        self.ntrip_host = ntrip_host or config.data_sources.ntrip_host
        self.ntrip_port = ntrip_port or config.data_sources.ntrip_port
        self.ntrip_mountpoint = ntrip_mountpoint or config.data_sources.ntrip_mountpoint
        self.ntrip_username = ntrip_username or config.data_sources.ntrip_username
        self.ntrip_password = ntrip_password or config.data_sources.ntrip_password

        # Receiver position (if static)
        self.receiver_position = None
        if receiver_lat is not None and receiver_lon is not None:
            self.receiver_position = ReceiverPosition(
                latitude=receiver_lat,
                longitude=receiver_lon,
                altitude=receiver_alt or 0.0
            )

        self.logger = ServiceLogger("ingestion", "gnss_tec")
        self.mq_client = mq_client
        self.validator = DataValidator()

        # Components
        self.ntrip_client = NTRIPClient(
            host=self.ntrip_host,
            port=self.ntrip_port,
            mountpoint=self.ntrip_mountpoint,
            username=self.ntrip_username,
            password=self.ntrip_password
        )
        self.rtcm_parser = RTCM3Parser()
        self.tec_calculator = TECCalculator(receiver_position=self.receiver_position)

        # Statistics
        self._tec_measurements_published = 0
        self._messages_processed = 0

    def process_rtcm_data(self, data: bytes):
        """
        Process RTCM3 data chunk

        Args:
            data: Binary RTCM3 data from NTRIP stream
        """
        try:
            # Parse RTCM3 messages
            messages = self.rtcm_parser.add_data(data)

            for message in messages:
                self._messages_processed += 1
                self.process_message(message)

        except Exception as e:
            self.logger.error(f"Error processing RTCM data: {e}", exc_info=True)

    def process_message(self, message: Dict[str, Any]):
        """
        Process parsed RTCM3 message

        Args:
            message: Parsed message dictionary
        """
        msg_type = message.get('type')

        if msg_type == 'station_position':
            # Update receiver position from station antenna position
            self.update_receiver_position(message)

        elif msg_type == 'gps_observables':
            # Process GPS observables
            self.process_observables(message)

        elif msg_type == 'glonass_observables':
            # Process GLONASS observables
            self.process_observables(message)

        # Other message types (ephemeris, etc.) can be handled here

    def update_receiver_position(self, message: Dict[str, Any]):
        """
        Update receiver position from station position message

        Args:
            message: Station position message
        """
        try:
            # Convert ECEF to geodetic
            x = message.get('x')
            y = message.get('y')
            z = message.get('z')

            if x is not None and y is not None and z is not None:
                lat, lon, alt = self.tec_calculator.ecef_to_geodetic(x, y, z)

                self.receiver_position = ReceiverPosition(
                    latitude=lat,
                    longitude=lon,
                    altitude=alt
                )

                self.tec_calculator.receiver_position = self.receiver_position

                self.logger.info(
                    f"Updated receiver position: {lat:.6f}°, {lon:.6f}°, {alt:.1f}m",
                    extra={
                        'latitude': lat,
                        'longitude': lon,
                        'altitude': alt
                    }
                )

        except Exception as e:
            self.logger.error(f"Error updating receiver position: {e}", exc_info=True)

    def process_observables(self, message: Dict[str, Any]):
        """
        Process GNSS observables to calculate TEC

        Args:
            message: Observable message (GPS or GLONASS)
        """
        # Note: Full implementation would process each satellite observable
        # For now, this is a placeholder showing the structure

        # In production:
        # 1. Extract all satellite observables from message
        # 2. For each satellite, compute satellite position from ephemeris
        # 3. Calculate TEC using tec_calculator.process_observable()
        # 4. Validate and publish each TEC measurement

        # Example (simplified):
        observables = message.get('observables', [])

        for obs in observables:
            # Would need satellite position from ephemeris
            # For now, skip (full implementation requires ephemeris handling)
            pass

        # Placeholder log
        if len(observables) > 0:
            self.logger.debug(
                f"Received {len(observables)} observables from station {message.get('station_id')}"
            )

    def publish_tec_measurement(self, measurement: Dict[str, Any]):
        """
        Publish TEC measurement to message queue

        Args:
            measurement: TEC measurement dictionary
        """
        if self.mq_client is None:
            self.logger.warning("No message queue client configured")
            return

        try:
            # Validate measurement
            is_valid, error_msg = self.validator.validate_tec(
                measurement['tec_value'],
                measurement['tec_error']
            )

            if not is_valid:
                self.logger.warning(f"TEC validation failed: {error_msg}")
                return

            # Validate elevation angle
            is_valid, error_msg = self.validator.validate_elevation_angle(
                measurement['elevation']
            )

            if not is_valid:
                self.logger.warning(f"Elevation validation failed: {error_msg}")
                return

            # Publish to queue
            self.mq_client.publish(
                topic=Topics.OBS_GNSS_TEC,
                data=measurement,
                source="gnss_tec_client"
            )

            self._tec_measurements_published += 1

            self.logger.debug(
                f"Published TEC measurement: {measurement['tec_value']:.2f} TECU",
                extra={
                    'tec': measurement['tec_value'],
                    'elevation': measurement['elevation'],
                    'satellite_id': measurement.get('satellite_id')
                }
            )

        except Exception as e:
            self.logger.error(f"Failed to publish TEC measurement: {e}", exc_info=True)

    async def run_monitoring_loop(self):
        """
        Main monitoring loop - continuously read NTRIP stream
        """
        self.logger.info(
            f"Starting GNSS-TEC monitoring: {self.ntrip_host}:{self.ntrip_port}/{self.ntrip_mountpoint}"
        )

        # Process NTRIP stream with callback
        await self.ntrip_client.read_stream(
            callback=self.process_rtcm_data,
            chunk_size=1024
        )

    async def run(self):
        """Start the GNSS-TEC monitoring service"""
        self.logger.info("GNSS-TEC client starting")

        # Initialize message queue if not provided
        if self.mq_client is None:
            config = get_config()
            self.mq_client = MessageQueueClient(
                host=config.services.rabbitmq_host,
                port=config.services.rabbitmq_port,
                username=config.services.rabbitmq_user,
                password=config.services.rabbitmq_password
            )

        # Run monitoring loop
        try:
            await self.run_monitoring_loop()
        except KeyboardInterrupt:
            self.logger.info("GNSS-TEC client stopped by user")
        except Exception as e:
            self.logger.error(f"GNSS-TEC client error: {e}", exc_info=True)
        finally:
            # Cleanup
            await self.ntrip_client.disconnect()

            # Log statistics
            self.logger.info(
                "GNSS-TEC client statistics",
                extra={
                    'rtcm_messages': self._messages_processed,
                    'tec_published': self._tec_measurements_published,
                    'parser_stats': self.rtcm_parser.statistics,
                    'calculator_stats': self.tec_calculator.statistics
                }
            )

    @property
    def statistics(self) -> Dict[str, Any]:
        """Get client statistics"""
        return {
            'rtcm_messages_processed': self._messages_processed,
            'tec_measurements_published': self._tec_measurements_published,
            'rtcm_parser': self.rtcm_parser.statistics,
            'tec_calculator': self.tec_calculator.statistics
        }


async def main():
    """Standalone entry point for GNSS-TEC client"""
    client = GNSSTECClient()
    await client.run()


if __name__ == "__main__":
    asyncio.run(main())
