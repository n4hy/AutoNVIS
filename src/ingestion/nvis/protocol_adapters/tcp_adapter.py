"""
TCP Protocol Adapter for NVIS Sounder Data

Handles real-time streaming data from professional NVIS sounders
via TCP socket connections.
"""

import asyncio
import json
from typing import Optional, AsyncIterator
from .base_adapter import BaseAdapter, NVISMeasurement, SounderMetadata
from ....common.logging_config import ServiceLogger


class TCPAdapter(BaseAdapter):
    """
    TCP socket listener for real-time NVIS sounder streams

    Protocol: JSON-encoded measurements, one per line
    """

    def __init__(self, adapter_id: str, config: dict):
        super().__init__(adapter_id, config)
        self.logger = ServiceLogger(f"tcp_adapter_{adapter_id}")
        self.host = config.get('host', '0.0.0.0')
        self.port = config.get('port', 8001)
        self.server = None
        self.measurement_queue = asyncio.Queue()
        self.sounder_registry = {}  # sounder_id → metadata

    async def start(self):
        """Start TCP server"""
        self.running = True
        self.server = await asyncio.start_server(
            self._handle_client,
            self.host,
            self.port
        )
        self.logger.info(f"TCP adapter listening on {self.host}:{self.port}")

    async def stop(self):
        """Stop TCP server"""
        self.running = False
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            self.logger.info("TCP adapter stopped")

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle incoming client connection"""
        addr = writer.get_extra_info('peername')
        self.logger.info(f"New client connection from {addr}")

        try:
            while self.running:
                # Read line-delimited JSON
                line = await reader.readline()
                if not line:
                    break

                try:
                    data = json.loads(line.decode('utf-8'))
                    measurement = self._parse_measurement(data)

                    if measurement and self.validate_measurement(measurement):
                        await self.measurement_queue.put(measurement)
                        self.measurement_count += 1
                        self.logger.debug(
                            f"Received measurement from {measurement.sounder_id}"
                        )
                    else:
                        self.logger.warning(f"Invalid measurement from {addr}")

                except json.JSONDecodeError as e:
                    self.logger.warning(f"JSON decode error from {addr}: {e}")
                except Exception as e:
                    self.logger.error(f"Error processing measurement: {e}", exc_info=True)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Client handler error: {e}", exc_info=True)
        finally:
            writer.close()
            await writer.wait_closed()
            self.logger.info(f"Client disconnected: {addr}")

    def _parse_measurement(self, data: dict) -> Optional[NVISMeasurement]:
        """
        Parse JSON data into NVISMeasurement

        Expected JSON format:
        {
            "sounder_id": "STATION_ID",
            "timestamp": "2025-01-15T12:34:56.789Z",
            "tx": {"lat": 40.0, "lon": -105.0, "alt": 1500.0},
            "rx": {"lat": 40.5, "lon": -104.5, "alt": 1600.0},
            "frequency": 7.5,
            "elevation_angle": 85.0,
            "azimuth": 45.0,
            "signal_strength": -80.0,
            "group_delay": 2.5,
            "snr": 20.0,
            "is_o_mode": true,
            "tx_power": 100.0,
            "bandwidth": 3.0
        }
        """
        try:
            tx = data.get('tx', {})
            rx = data.get('rx', {})

            # Calculate hop distance
            hop_distance = self._calculate_hop_distance(
                tx.get('lat'), tx.get('lon'),
                rx.get('lat'), rx.get('lon')
            )

            measurement = NVISMeasurement(
                tx_latitude=tx.get('lat'),
                tx_longitude=tx.get('lon'),
                tx_altitude=tx.get('alt', 0.0),
                rx_latitude=rx.get('lat'),
                rx_longitude=rx.get('lon'),
                rx_altitude=rx.get('alt', 0.0),
                frequency=data.get('frequency'),
                elevation_angle=data.get('elevation_angle', 85.0),
                azimuth=data.get('azimuth', 0.0),
                hop_distance=hop_distance,
                signal_strength=data.get('signal_strength'),
                group_delay=data.get('group_delay'),
                snr=data.get('snr'),
                sounder_id=data.get('sounder_id'),
                timestamp=data.get('timestamp'),
                is_o_mode=data.get('is_o_mode', True),
                tx_power=data.get('tx_power'),
                bandwidth=data.get('bandwidth')
            )

            return measurement

        except (KeyError, TypeError, ValueError) as e:
            self.logger.warning(f"Failed to parse measurement: {e}")
            return None

    def _calculate_hop_distance(self, lat1: float, lon1: float,
                                 lat2: float, lon2: float) -> float:
        """
        Calculate great circle distance between two points

        Args:
            lat1, lon1: First point (degrees)
            lat2, lon2: Second point (degrees)

        Returns:
            Distance in kilometers
        """
        import math

        R = 6371.0  # Earth radius in km

        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlon = math.radians(lon2 - lon1)
        dlat = math.radians(lat2 - lat1)

        a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    async def get_measurements(self) -> AsyncIterator[NVISMeasurement]:
        """Yield measurements from queue"""
        while self.running or not self.measurement_queue.empty():
            try:
                measurement = await asyncio.wait_for(
                    self.measurement_queue.get(),
                    timeout=1.0
                )
                yield measurement
            except asyncio.TimeoutError:
                continue

    def get_sounder_metadata(self, sounder_id: str) -> Optional[SounderMetadata]:
        """Retrieve sounder metadata"""
        return self.sounder_registry.get(sounder_id)

    def register_sounder(self, metadata: SounderMetadata):
        """Register a sounder with metadata"""
        self.sounder_registry[metadata.sounder_id] = metadata
        self.logger.info(f"Registered sounder: {metadata.sounder_id}")
