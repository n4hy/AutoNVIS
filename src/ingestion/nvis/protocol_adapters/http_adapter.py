"""
HTTP Protocol Adapter for NVIS Sounder Data

Handles HTTP POST submissions from NVIS sounders via REST API.
"""

import asyncio
import json
from typing import Optional, AsyncIterator
from aiohttp import web
from .base_adapter import BaseAdapter, NVISMeasurement, SounderMetadata
from ....common.logging_config import ServiceLogger


class HTTPAdapter(BaseAdapter):
    """
    HTTP REST API for NVIS sounder submissions

    Endpoints:
    - POST /measurement: Submit a single measurement
    - POST /batch: Submit multiple measurements
    - POST /register: Register sounder metadata
    """

    def __init__(self, adapter_id: str, config: dict):
        super().__init__(adapter_id, config)
        self.logger = ServiceLogger(f"http_adapter_{adapter_id}")
        self.host = config.get('host', '0.0.0.0')
        self.port = config.get('port', 8002)
        self.app = web.Application()
        self.runner = None
        self.measurement_queue = asyncio.Queue()
        self.sounder_registry = {}

        # Setup routes
        self.app.router.add_post('/measurement', self._handle_measurement)
        self.app.router.add_post('/batch', self._handle_batch)
        self.app.router.add_post('/register', self._handle_register)
        self.app.router.add_get('/health', self._handle_health)

    async def start(self):
        """Start HTTP server"""
        self.running = True
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, self.host, self.port)
        await site.start()
        self.logger.info(f"HTTP adapter listening on {self.host}:{self.port}")

    async def stop(self):
        """Stop HTTP server"""
        self.running = False
        if self.runner:
            await self.runner.cleanup()
            self.logger.info("HTTP adapter stopped")

    async def _handle_measurement(self, request: web.Request) -> web.Response:
        """Handle single measurement submission"""
        try:
            data = await request.json()
            measurement = self._parse_measurement(data)

            if measurement and self.validate_measurement(measurement):
                await self.measurement_queue.put(measurement)
                self.measurement_count += 1
                return web.json_response({
                    'status': 'success',
                    'message': 'Measurement accepted'
                })
            else:
                return web.json_response({
                    'status': 'error',
                    'message': 'Invalid measurement'
                }, status=400)

        except json.JSONDecodeError:
            return web.json_response({
                'status': 'error',
                'message': 'Invalid JSON'
            }, status=400)
        except Exception as e:
            self.logger.error(f"Error handling measurement: {e}", exc_info=True)
            return web.json_response({
                'status': 'error',
                'message': str(e)
            }, status=500)

    async def _handle_batch(self, request: web.Request) -> web.Response:
        """Handle batch measurement submission"""
        try:
            data = await request.json()
            measurements = data.get('measurements', [])

            accepted = 0
            rejected = 0

            for meas_data in measurements:
                measurement = self._parse_measurement(meas_data)
                if measurement and self.validate_measurement(measurement):
                    await self.measurement_queue.put(measurement)
                    self.measurement_count += 1
                    accepted += 1
                else:
                    rejected += 1

            return web.json_response({
                'status': 'success',
                'accepted': accepted,
                'rejected': rejected
            })

        except Exception as e:
            self.logger.error(f"Error handling batch: {e}", exc_info=True)
            return web.json_response({
                'status': 'error',
                'message': str(e)
            }, status=500)

    async def _handle_register(self, request: web.Request) -> web.Response:
        """Handle sounder registration"""
        try:
            data = await request.json()
            metadata = SounderMetadata(
                sounder_id=data['sounder_id'],
                name=data.get('name', ''),
                operator=data.get('operator', ''),
                location=data.get('location', ''),
                latitude=data['latitude'],
                longitude=data['longitude'],
                altitude=data.get('altitude', 0.0),
                equipment_type=data.get('equipment_type', 'unknown'),
                calibration_status=data.get('calibration_status', 'unknown')
            )

            self.sounder_registry[metadata.sounder_id] = metadata
            self.logger.info(f"Registered sounder: {metadata.sounder_id}")

            return web.json_response({
                'status': 'success',
                'message': f'Sounder {metadata.sounder_id} registered'
            })

        except KeyError as e:
            return web.json_response({
                'status': 'error',
                'message': f'Missing required field: {e}'
            }, status=400)
        except Exception as e:
            self.logger.error(f"Error handling registration: {e}", exc_info=True)
            return web.json_response({
                'status': 'error',
                'message': str(e)
            }, status=500)

    async def _handle_health(self, request: web.Request) -> web.Response:
        """Health check endpoint"""
        return web.json_response({
            'status': 'healthy',
            'measurement_count': self.measurement_count,
            'registered_sounders': len(self.sounder_registry)
        })

    def _parse_measurement(self, data: dict) -> Optional[NVISMeasurement]:
        """Parse JSON data into NVISMeasurement (same as TCP adapter)"""
        try:
            tx = data.get('tx', {})
            rx = data.get('rx', {})

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
        """Calculate great circle distance"""
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
