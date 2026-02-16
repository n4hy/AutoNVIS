"""
RabbitMQ Data Subscribers for Dashboard

Provides subscriber classes that consume messages from various topics
and update the dashboard state in real-time.
"""

import asyncio
import threading
from datetime import datetime
from typing import Optional, Callable
import numpy as np

from src.common.message_queue import MessageQueueClient, Topics, Message
from src.common.logging_config import ServiceLogger
from .state_manager import DashboardState


class AsyncTaskRunner:
    """
    Run async tasks from sync context without blocking.

    This allows synchronous RabbitMQ consumer threads to submit async
    tasks (like WebSocket broadcasts) without blocking message processing.
    """

    def __init__(self):
        self.loop = None
        self.thread = threading.Thread(target=self._run_loop, daemon=True, name="AsyncTaskRunner")
        self.thread.start()
        # Wait for loop to be ready
        while self.loop is None:
            threading.Event().wait(0.01)

    def _run_loop(self):
        """Background thread running the event loop"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def submit(self, coro):
        """Submit coroutine to be run in background loop (non-blocking)"""
        if self.loop and not self.loop.is_closed():
            asyncio.run_coroutine_threadsafe(coro, self.loop)

    def stop(self):
        """Stop the background loop"""
        if self.loop and not self.loop.is_closed():
            self.loop.call_soon_threadsafe(self.loop.stop)


class GridDataSubscriber:
    """
    Subscribes to proc.grid_ready messages and updates dashboard state.

    Similar to supervisor's GridSubscriber but optimized for dashboard:
    - Stores grids in dashboard state
    - Computes derived parameters (foF2, TEC, hmF2)
    - Broadcasts WebSocket updates
    """

    def __init__(
        self,
        rabbitmq_host: str,
        rabbitmq_port: int,
        rabbitmq_user: str,
        rabbitmq_password: str,
        rabbitmq_vhost: str,
        state: DashboardState,
        ws_broadcast_callback: Optional[Callable] = None
    ):
        """
        Initialize grid subscriber.

        Args:
            rabbitmq_host: RabbitMQ host
            rabbitmq_port: RabbitMQ port
            rabbitmq_user: RabbitMQ username
            rabbitmq_password: RabbitMQ password
            rabbitmq_vhost: RabbitMQ vhost
            state: Dashboard state manager
            ws_broadcast_callback: Async callback for WebSocket broadcast
        """
        self.rabbitmq_host = rabbitmq_host
        self.rabbitmq_port = rabbitmq_port
        self.rabbitmq_user = rabbitmq_user
        self.rabbitmq_password = rabbitmq_password
        self.rabbitmq_vhost = rabbitmq_vhost
        self.mq_client = None  # Will be created in thread
        self.state = state
        self.ws_broadcast = ws_broadcast_callback
        self.logger = ServiceLogger("dashboard", "grid_subscriber")
        self.async_runner = AsyncTaskRunner()  # For non-blocking async broadcasts

        self.subscriber_thread: Optional[threading.Thread] = None
        self.running = False

    def start(self):
        """Start background subscriber thread."""
        if self.running:
            self.logger.warning("Grid subscriber already running")
            return

        self.running = True
        self.subscriber_thread = threading.Thread(
            target=self._consume_thread,
            daemon=True,
            name="GridDataSubscriber"
        )
        self.subscriber_thread.start()
        self.logger.info("Grid data subscriber started")

    def stop(self):
        """Stop subscriber thread."""
        if not self.running:
            return

        self.running = False
        self.mq_client.stop_consuming()

        if self.subscriber_thread:
            self.subscriber_thread.join(timeout=5)

        self.logger.info("Grid data subscriber stopped")

    def _on_grid_message(self, message: Message):
        """Handle incoming grid message."""
        try:
            data = message.data
            cycle_id = data.get('cycle_id', 'unknown')

            self.logger.debug(f"Received grid: {cycle_id}")

            # Validate and reconstruct grid
            if not self._validate_grid_message(data):
                self.logger.error(f"Invalid grid message: {cycle_id}")
                return

            grid_data = self._reconstruct_grid(data)

            # Update state
            timestamp = datetime.fromisoformat(message.timestamp.rstrip('Z'))
            self.state.update_grid(grid_data, timestamp)

            # Broadcast WebSocket update (non-blocking)
            if self.ws_broadcast:
                try:
                    self.async_runner.submit(self.ws_broadcast({
                        'type': 'grid_update',
                        'data': {
                            'cycle_id': cycle_id,
                            'timestamp': message.timestamp,
                            'ne_max': float(np.max(grid_data['ne_grid'])),
                            'quality': grid_data.get('quality', 'unknown')
                        }
                    }))
                    self.logger.debug(f"Grid update broadcast submitted: {cycle_id}")
                except Exception as e:
                    self.logger.error(f"WebSocket broadcast submit failed: {e}")

            self.logger.info(f"Grid updated: {cycle_id}, Ne_max={np.max(grid_data['ne_grid']):.2e}")

        except Exception as e:
            self.logger.error(f"Error processing grid message: {e}", exc_info=True)

    def _validate_grid_message(self, data: dict) -> bool:
        """Validate grid message has required fields."""
        required = ['grid_shape', 'ne_grid_flat', 'lat_min', 'lat_max',
                    'lon_min', 'lon_max', 'alt_min_km', 'alt_max_km']

        for field in required:
            if field not in data:
                return False

        shape = data['grid_shape']
        expected_size = shape[0] * shape[1] * shape[2]
        if len(data['ne_grid_flat']) != expected_size:
            return False

        return True

    def _reconstruct_grid(self, data: dict) -> dict:
        """Reconstruct 3D grid from flattened message."""
        shape = data['grid_shape']
        n_lat, n_lon, n_alt = shape

        ne_flat = np.array(data['ne_grid_flat'])
        ne_grid = ne_flat.reshape(shape, order='C')

        lat = np.linspace(data['lat_min'], data['lat_max'], n_lat)
        lon = np.linspace(data['lon_min'], data['lon_max'], n_lon)
        alt = np.linspace(data['alt_min_km'], data['alt_max_km'], n_alt)

        return {
            'ne_grid': ne_grid,
            'lat': lat,
            'lon': lon,
            'alt': alt,
            'xray_flux': data.get('xray_flux_wm2', 0.0),
            'cycle_id': data.get('cycle_id', 'unknown'),
            'quality': data.get('grid_quality', 'unknown'),
            'effective_ssn': data.get('effective_ssn', 0.0),
            'timestamp': data.get('grid_timestamp_utc', ''),
            'observations_used': data.get('observations_used', 0),
            'filter_converged': data.get('filter_converged', False)
        }

    def _consume_thread(self):
        """Background thread for consuming grid messages."""
        try:
            # Create own connection in this thread
            self.mq_client = MessageQueueClient(
                host=self.rabbitmq_host,
                port=self.rabbitmq_port,
                username=self.rabbitmq_user,
                password=self.rabbitmq_password,
                vhost=self.rabbitmq_vhost
            )
            self.logger.info("Grid subscriber connected to RabbitMQ")

            self.mq_client.subscribe(
                topic_pattern=Topics.PROC_GRID_READY,
                callback=self._on_grid_message,
                queue_name="dashboard_grid_subscriber"
            )
            self.logger.info(f"Subscribed to {Topics.PROC_GRID_READY}")
            self.mq_client.start_consuming()
        except Exception as e:
            self.logger.error(f"Grid subscriber error: {e}", exc_info=True)


class PropagationSubscriber:
    """Subscribes to propagation output topics (frequency plans, coverage maps)."""

    def __init__(
        self,
        rabbitmq_host: str,
        rabbitmq_port: int,
        rabbitmq_user: str,
        rabbitmq_password: str,
        rabbitmq_vhost: str,
        state: DashboardState,
        ws_broadcast_callback: Optional[Callable] = None
    ):
        self.rabbitmq_host = rabbitmq_host
        self.rabbitmq_port = rabbitmq_port
        self.rabbitmq_user = rabbitmq_user
        self.rabbitmq_password = rabbitmq_password
        self.rabbitmq_vhost = rabbitmq_vhost
        self.mq_client = None
        self.state = state
        self.ws_broadcast = ws_broadcast_callback
        self.logger = ServiceLogger("dashboard", "propagation_subscriber")
        self.async_runner = AsyncTaskRunner()  # For non-blocking async broadcasts

        self.subscriber_thread: Optional[threading.Thread] = None
        self.running = False

    def start(self):
        """Start subscriber thread."""
        if self.running:
            return

        self.running = True
        self.subscriber_thread = threading.Thread(
            target=self._consume_thread,
            daemon=True,
            name="PropagationSubscriber"
        )
        self.subscriber_thread.start()
        self.logger.info("Propagation subscriber started")

    def stop(self):
        """Stop subscriber thread."""
        if not self.running:
            return

        self.running = False
        if self.subscriber_thread:
            self.subscriber_thread.join(timeout=5)

        self.logger.info("Propagation subscriber stopped")

    def _on_frequency_plan(self, message: Message):
        """Handle frequency plan message."""
        try:
            data = message.data
            self.state.update_frequency_plan({
                **data,
                'timestamp': message.timestamp
            })

            if self.ws_broadcast:
                self.async_runner.submit(self.ws_broadcast({
                    'type': 'frequency_plan_update',
                    'data': data
                }))

            self.logger.info(f"Frequency plan updated: LUF={data.get('luf_mhz')}, MUF={data.get('muf_mhz')}")

        except Exception as e:
            self.logger.error(f"Error processing frequency plan: {e}", exc_info=True)

    def _on_coverage_map(self, message: Message):
        """Handle coverage map message."""
        try:
            data = message.data
            self.state.update_coverage_map({
                **data,
                'timestamp': message.timestamp
            })

            if self.ws_broadcast:
                self.async_runner.submit(self.ws_broadcast({
                    'type': 'coverage_map_update',
                    'data': {'frequency': data.get('frequency_mhz')}
                }))

        except Exception as e:
            self.logger.error(f"Error processing coverage map: {e}", exc_info=True)

    def _consume_thread(self):
        """Background thread for consuming propagation messages."""
        try:
            # Create own connection
            self.mq_client = MessageQueueClient(
                host=self.rabbitmq_host,
                port=self.rabbitmq_port,
                username=self.rabbitmq_user,
                password=self.rabbitmq_password,
                vhost=self.rabbitmq_vhost
            )
            self.logger.info("Propagation subscriber connected to RabbitMQ")

            # Subscribe to frequency plans
            self.mq_client.subscribe(
                topic_pattern=Topics.OUT_FREQUENCY_PLAN,
                callback=self._on_frequency_plan,
                queue_name="dashboard_frequency_plan"
            )

            # Subscribe to coverage maps
            self.mq_client.subscribe(
                topic_pattern=Topics.OUT_COVERAGE_MAP,
                callback=self._on_coverage_map,
                queue_name="dashboard_coverage_map"
            )

            self.logger.info("Subscribed to propagation topics")
            self.mq_client.start_consuming()

        except Exception as e:
            self.logger.error(f"Propagation subscriber error: {e}", exc_info=True)


class SpaceWeatherSubscriber:
    """Subscribes to space weather topics (X-ray, solar wind, geomag)."""

    def __init__(
        self,
        rabbitmq_host: str,
        rabbitmq_port: int,
        rabbitmq_user: str,
        rabbitmq_password: str,
        rabbitmq_vhost: str,
        state: DashboardState,
        ws_broadcast_callback: Optional[Callable] = None
    ):
        self.rabbitmq_host = rabbitmq_host
        self.rabbitmq_port = rabbitmq_port
        self.rabbitmq_user = rabbitmq_user
        self.rabbitmq_password = rabbitmq_password
        self.rabbitmq_vhost = rabbitmq_vhost
        self.mq_client = None
        self.state = state
        self.ws_broadcast = ws_broadcast_callback
        self.logger = ServiceLogger("dashboard", "spaceweather_subscriber")
        self.async_runner = AsyncTaskRunner()  # For non-blocking async broadcasts

        self.subscriber_thread: Optional[threading.Thread] = None
        self.running = False

    def start(self):
        """Start subscriber thread."""
        if self.running:
            return

        self.running = True
        self.subscriber_thread = threading.Thread(
            target=self._consume_thread,
            daemon=True,
            name="SpaceWeatherSubscriber"
        )
        self.subscriber_thread.start()
        self.logger.info("Space weather subscriber started")

    def stop(self):
        """Stop subscriber thread."""
        if not self.running:
            return

        self.running = False
        if self.subscriber_thread:
            self.subscriber_thread.join(timeout=5)

        self.logger.info("Space weather subscriber stopped")

    def _on_xray(self, message: Message):
        """Handle X-ray flux message."""
        try:
            data = message.data

            # Check if this is a historical batch
            if data.get('type') == 'historical_batch':
                # Broadcast batch for PyQt to process all at once
                if self.ws_broadcast:
                    self.async_runner.submit(self.ws_broadcast({
                        'type': 'xray_historical_batch',
                        'data': {
                            'records': data.get('records', []),
                            'count': data.get('count', 0)
                        }
                    }))
                    self.logger.info(f"X-ray historical batch broadcast: {data.get('count', 0)} records")
            else:
                # Regular real-time update
                data = {**data, 'timestamp': message.timestamp}
                self.state.update_xray_flux(data)

                if self.ws_broadcast:
                    self.async_runner.submit(self.ws_broadcast({
                        'type': 'xray_update',
                        'data': data
                    }))
                    self.logger.debug(f"X-ray update broadcast submitted: flux={data.get('flux_short', 0):.2e}")

        except Exception as e:
            self.logger.error(f"Error processing X-ray message: {e}", exc_info=True)

    def _on_solar_wind(self, message: Message):
        """Handle solar wind message."""
        try:
            data = {**message.data, 'timestamp': message.timestamp}
            self.state.update_solar_wind(data)

            if self.ws_broadcast:
                self.async_runner.submit(self.ws_broadcast({
                    'type': 'solar_wind_update',
                    'data': data
                }))
                self.logger.debug("Solar wind update broadcast submitted")

        except Exception as e:
            self.logger.error(f"Error processing solar wind message: {e}", exc_info=True)

    def _on_geomag(self, message: Message):
        """Handle geomagnetic message."""
        try:
            data = {**message.data, 'timestamp': message.timestamp}
            self.state.update_geomag(data)

        except Exception as e:
            self.logger.error(f"Error processing geomag message: {e}", exc_info=True)

    def _on_mode_change(self, message: Message):
        """Handle mode change message."""
        try:
            mode = message.data.get('new_mode', 'UNKNOWN')
            reason = message.data.get('reason', '')
            self.state.update_mode(mode, reason)

            if self.ws_broadcast:
                asyncio.run(self.ws_broadcast({
                    'type': 'mode_change',
                    'data': {'mode': mode, 'reason': reason}
                }))

            self.logger.info(f"Mode changed: {mode} ({reason})")

        except Exception as e:
            self.logger.error(f"Error processing mode change: {e}", exc_info=True)

    def _consume_thread(self):
        """Background thread for consuming space weather messages."""
        try:
            # Create own connection
            self.mq_client = MessageQueueClient(
                host=self.rabbitmq_host,
                port=self.rabbitmq_port,
                username=self.rabbitmq_user,
                password=self.rabbitmq_password,
                vhost=self.rabbitmq_vhost
            )
            self.logger.info("Space weather subscriber connected to RabbitMQ")

            self.mq_client.subscribe(
                topic_pattern=Topics.WX_XRAY,
                callback=self._on_xray,
                queue_name="dashboard_xray"
            )

            self.mq_client.subscribe(
                topic_pattern=Topics.WX_SOLAR_WIND,
                callback=self._on_solar_wind,
                queue_name="dashboard_solar_wind"
            )

            self.mq_client.subscribe(
                topic_pattern=Topics.WX_GEOMAG,
                callback=self._on_geomag,
                queue_name="dashboard_geomag"
            )

            self.mq_client.subscribe(
                topic_pattern=Topics.CTRL_MODE_CHANGE,
                callback=self._on_mode_change,
                queue_name="dashboard_mode_change"
            )

            self.logger.info("Subscribed to space weather topics")
            self.mq_client.start_consuming()

        except Exception as e:
            self.logger.error(f"Space weather subscriber error: {e}", exc_info=True)


class GloTECSubscriber:
    """Subscribes to GloTEC global TEC map topics."""

    def __init__(
        self,
        rabbitmq_host: str,
        rabbitmq_port: int,
        rabbitmq_user: str,
        rabbitmq_password: str,
        rabbitmq_vhost: str,
        state: DashboardState,
        ws_broadcast_callback: Optional[Callable] = None
    ):
        self.rabbitmq_host = rabbitmq_host
        self.rabbitmq_port = rabbitmq_port
        self.rabbitmq_user = rabbitmq_user
        self.rabbitmq_password = rabbitmq_password
        self.rabbitmq_vhost = rabbitmq_vhost
        self.mq_client = None
        self.state = state
        self.ws_broadcast = ws_broadcast_callback
        self.logger = ServiceLogger("dashboard", "glotec_subscriber")
        self.async_runner = AsyncTaskRunner()

        self.subscriber_thread: Optional[threading.Thread] = None
        self.running = False

    def start(self):
        """Start subscriber thread."""
        if self.running:
            return

        self.running = True
        self.subscriber_thread = threading.Thread(
            target=self._consume_thread,
            daemon=True,
            name="GloTECSubscriber"
        )
        self.subscriber_thread.start()
        self.logger.info("GloTEC subscriber started")

    def stop(self):
        """Stop subscriber thread."""
        if not self.running:
            return

        self.running = False
        if self.subscriber_thread:
            self.subscriber_thread.join(timeout=5)

        self.logger.info("GloTEC subscriber stopped")

    def _on_glotec_map(self, message: Message):
        """Handle GloTEC map message."""
        try:
            data = message.data
            self.state.update_glotec_map(data)

            # Broadcast full grid data to WebSocket clients for visualization
            if self.ws_broadcast:
                stats = data.get('statistics', {})
                self.async_runner.submit(self.ws_broadcast({
                    'type': 'glotec_update',
                    'data': data  # Send full grid data for PyQt visualization
                }))
                self.logger.info(f"GloTEC broadcast to WebSocket: tec_mean={stats.get('tec_mean', 0):.1f}")

            self.logger.info(
                f"GloTEC map updated: {data.get('timestamp')}",
                extra={'tec_mean': stats.get('tec_mean')}
            )

        except Exception as e:
            self.logger.error(f"Error processing GloTEC message: {e}", exc_info=True)

    def _consume_thread(self):
        """Background thread for consuming GloTEC messages."""
        try:
            # Create own connection
            self.mq_client = MessageQueueClient(
                host=self.rabbitmq_host,
                port=self.rabbitmq_port,
                username=self.rabbitmq_user,
                password=self.rabbitmq_password,
                vhost=self.rabbitmq_vhost
            )
            self.logger.info("GloTEC subscriber connected to RabbitMQ")

            self.mq_client.subscribe(
                topic_pattern=Topics.OBS_GLOTEC_MAP,
                callback=self._on_glotec_map,
                queue_name="dashboard_glotec"
            )

            self.logger.info(f"Subscribed to {Topics.OBS_GLOTEC_MAP}")
            self.mq_client.start_consuming()

        except Exception as e:
            self.logger.error(f"GloTEC subscriber error: {e}", exc_info=True)


class ObservationSubscriber:
    """Subscribes to observation topics (GNSS TEC, ionosonde, NVIS)."""

    def __init__(
        self,
        rabbitmq_host: str,
        rabbitmq_port: int,
        rabbitmq_user: str,
        rabbitmq_password: str,
        rabbitmq_vhost: str,
        state: DashboardState,
        ws_broadcast_callback: Optional[Callable] = None
    ):
        self.rabbitmq_host = rabbitmq_host
        self.rabbitmq_port = rabbitmq_port
        self.rabbitmq_user = rabbitmq_user
        self.rabbitmq_password = rabbitmq_password
        self.rabbitmq_vhost = rabbitmq_vhost
        self.mq_client = None
        self.state = state
        self.ws_broadcast = ws_broadcast_callback
        self.logger = ServiceLogger("dashboard", "observation_subscriber")
        self.async_runner = AsyncTaskRunner()  # For non-blocking async broadcasts

        self.subscriber_thread: Optional[threading.Thread] = None
        self.running = False

    def start(self):
        """Start subscriber thread."""
        if self.running:
            return

        self.running = True
        self.subscriber_thread = threading.Thread(
            target=self._consume_thread,
            daemon=True,
            name="ObservationSubscriber"
        )
        self.subscriber_thread.start()
        self.logger.info("Observation subscriber started")

    def stop(self):
        """Stop subscriber thread."""
        if not self.running:
            return

        self.running = False
        if self.subscriber_thread:
            self.subscriber_thread.join(timeout=5)

        self.logger.info("Observation subscriber stopped")

    def _on_observation(self, message: Message, obs_type: str):
        """Handle observation message."""
        try:
            data = {**message.data, 'timestamp': message.timestamp}
            self.state.add_observation(data, obs_type)

            if self.ws_broadcast:
                self.async_runner.submit(self.ws_broadcast({
                    'type': 'observation_update',
                    'data': {
                        'obs_type': obs_type,
                        'count': self.state.get_observation_counts()
                    }
                }))

        except Exception as e:
            self.logger.error(f"Error processing {obs_type} observation: {e}", exc_info=True)

    def _consume_thread(self):
        """Background thread for consuming observation messages."""
        try:
            # Create own connection
            self.mq_client = MessageQueueClient(
                host=self.rabbitmq_host,
                port=self.rabbitmq_port,
                username=self.rabbitmq_user,
                password=self.rabbitmq_password,
                vhost=self.rabbitmq_vhost
            )
            self.logger.info("Observation subscriber connected to RabbitMQ")

            self.mq_client.subscribe(
                topic_pattern=Topics.OBS_GNSS_TEC,
                callback=lambda msg: self._on_observation(msg, 'gnss_tec'),
                queue_name="dashboard_gnss_tec"
            )

            self.mq_client.subscribe(
                topic_pattern=Topics.OBS_IONOSONDE,
                callback=lambda msg: self._on_observation(msg, 'ionosonde'),
                queue_name="dashboard_ionosonde"
            )

            self.mq_client.subscribe(
                topic_pattern=Topics.OBS_NVIS_SOUNDER,
                callback=lambda msg: self._on_observation(msg, 'nvis_sounder'),
                queue_name="dashboard_nvis_sounder"
            )

            self.logger.info("Subscribed to observation topics")
            self.mq_client.start_consuming()

        except Exception as e:
            self.logger.error(f"Observation subscriber error: {e}", exc_info=True)


class SystemHealthSubscriber:
    """Subscribes to system health and alert topics."""

    def __init__(
        self,
        rabbitmq_host: str,
        rabbitmq_port: int,
        rabbitmq_user: str,
        rabbitmq_password: str,
        rabbitmq_vhost: str,
        state: DashboardState,
        ws_broadcast_callback: Optional[Callable] = None
    ):
        self.rabbitmq_host = rabbitmq_host
        self.rabbitmq_port = rabbitmq_port
        self.rabbitmq_user = rabbitmq_user
        self.rabbitmq_password = rabbitmq_password
        self.rabbitmq_vhost = rabbitmq_vhost
        self.mq_client = None
        self.state = state
        self.ws_broadcast = ws_broadcast_callback
        self.logger = ServiceLogger("dashboard", "health_subscriber")
        self.async_runner = AsyncTaskRunner()  # For non-blocking async broadcasts

        self.subscriber_thread: Optional[threading.Thread] = None
        self.running = False

    def start(self):
        """Start subscriber thread."""
        if self.running:
            return

        self.running = True
        self.subscriber_thread = threading.Thread(
            target=self._consume_thread,
            daemon=True,
            name="SystemHealthSubscriber"
        )
        self.subscriber_thread.start()
        self.logger.info("System health subscriber started")

    def stop(self):
        """Stop subscriber thread."""
        if not self.running:
            return

        self.running = False
        if self.subscriber_thread:
            self.subscriber_thread.join(timeout=5)

        self.logger.info("System health subscriber stopped")

    def _on_health_status(self, message: Message):
        """Handle health status message."""
        try:
            service_name = message.data.get('service', 'unknown')
            self.state.update_service_status(service_name, message.data)

        except Exception as e:
            self.logger.error(f"Error processing health status: {e}", exc_info=True)

    def _on_alert(self, message: Message):
        """Handle alert message."""
        try:
            alert_data = {**message.data, 'timestamp': message.timestamp}
            self.state.add_alert(alert_data)

            if self.ws_broadcast:
                self.async_runner.submit(self.ws_broadcast({
                    'type': 'alert',
                    'data': alert_data
                }))

            self.logger.warning(f"Alert: {alert_data.get('message', 'Unknown')}")

        except Exception as e:
            self.logger.error(f"Error processing alert: {e}", exc_info=True)

    def _consume_thread(self):
        """Background thread for consuming health messages."""
        try:
            # Create own connection
            self.mq_client = MessageQueueClient(
                host=self.rabbitmq_host,
                port=self.rabbitmq_port,
                username=self.rabbitmq_user,
                password=self.rabbitmq_password,
                vhost=self.rabbitmq_vhost
            )
            self.logger.info("System health subscriber connected to RabbitMQ")

            self.mq_client.subscribe(
                topic_pattern=Topics.HEALTH_STATUS,
                callback=self._on_health_status,
                queue_name="dashboard_health_status"
            )

            self.mq_client.subscribe(
                topic_pattern=Topics.OUT_ALERT,
                callback=self._on_alert,
                queue_name="dashboard_alerts"
            )

            self.logger.info("Subscribed to health topics")
            self.mq_client.start_consuming()

        except Exception as e:
            self.logger.error(f"System health subscriber error: {e}", exc_info=True)
