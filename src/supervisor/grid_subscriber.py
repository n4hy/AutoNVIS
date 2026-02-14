"""
Grid Subscriber for System Orchestrator

Manages subscription to proc.grid_ready messages and provides
async access to latest electron density grids from SR-UKF filter.

This module runs a background thread that subscribes to the message queue
and caches the latest grid for async retrieval by the system orchestrator.
"""

import asyncio
import threading
from queue import Queue
from typing import Optional, Tuple, Dict, Any
import numpy as np
from datetime import datetime, timedelta

from src.common.message_queue import MessageQueueClient, Topics, Message
from src.common.logging_config import ServiceLogger


class GridSubscriber:
    """
    Subscribes to proc.grid_ready and provides async access to grids.

    This class runs a background thread that consumes grid messages from
    the message queue and stores the latest grid for retrieval by the
    system orchestrator during the propagation phase.
    """

    def __init__(self, mq_client: MessageQueueClient):
        """
        Initialize grid subscriber.

        Args:
            mq_client: Message queue client for subscribing
        """
        self.mq_client = mq_client
        self.logger = ServiceLogger("supervisor", "grid_subscriber")

        # Latest grid storage (thread-safe)
        self.latest_grid: Optional[Dict[str, Any]] = None
        self.latest_grid_time: Optional[datetime] = None
        self.grid_lock = threading.Lock()

        # Background subscription
        self.subscriber_thread: Optional[threading.Thread] = None
        self.running = False

        # Statistics
        self.grids_received = 0
        self.grids_invalid = 0

    def start(self):
        """Start background subscriber thread."""
        if self.running:
            self.logger.warning("Grid subscriber already running")
            return

        self.running = True
        self.subscriber_thread = threading.Thread(
            target=self._consume_thread,
            daemon=True,
            name="GridSubscriber"
        )
        self.subscriber_thread.start()
        self.logger.info("Grid subscriber started")

    def stop(self):
        """Stop subscriber thread."""
        if not self.running:
            return

        self.running = False
        self.mq_client.stop_consuming()

        if self.subscriber_thread:
            self.subscriber_thread.join(timeout=5)

        self.logger.info(
            f"Grid subscriber stopped: "
            f"{self.grids_received} received, {self.grids_invalid} invalid"
        )

    def _on_grid_message(self, message: Message):
        """
        Handle incoming grid message from proc.grid_ready topic.

        Args:
            message: Grid message containing electron density grid
        """
        try:
            data = message.data
            cycle_id = data.get('cycle_id', 'unknown')

            self.logger.debug(f"Received grid message: {cycle_id}")

            # Validate message
            if not self._validate_grid_message(data):
                self.logger.error(f"Invalid grid message: {cycle_id}")
                self.grids_invalid += 1
                return

            # Reconstruct grid from flattened data
            grid_data = self._reconstruct_grid(data)

            # Store latest grid (thread-safe)
            with self.grid_lock:
                self.latest_grid = grid_data
                self.latest_grid_time = datetime.fromisoformat(
                    message.timestamp.rstrip('Z')
                )

            self.grids_received += 1

            self.logger.info(
                f"Grid stored: {cycle_id}, "
                f"Ne_max={np.max(grid_data['ne_grid']):.2e} el/m³, "
                f"quality={grid_data.get('quality', 'unknown')}"
            )

        except Exception as e:
            self.logger.error(
                f"Error processing grid message: {e}",
                exc_info=True
            )
            self.grids_invalid += 1

    def _validate_grid_message(self, data: dict) -> bool:
        """
        Validate grid message contains all required fields.

        Args:
            data: Message data dictionary

        Returns:
            True if message is valid
        """
        required_fields = [
            'grid_shape', 'ne_grid_flat',
            'lat_min', 'lat_max',
            'lon_min', 'lon_max',
            'alt_min_km', 'alt_max_km'
        ]

        # Check required fields
        for field in required_fields:
            if field not in data:
                self.logger.error(f"Missing required field: {field}")
                return False

        # Validate grid shape
        shape = data['grid_shape']
        if not isinstance(shape, (list, tuple)) or len(shape) != 3:
            self.logger.error(f"Invalid grid shape: {shape}")
            return False

        # Validate grid size matches shape
        expected_size = shape[0] * shape[1] * shape[2]
        actual_size = len(data['ne_grid_flat'])

        if expected_size != actual_size:
            self.logger.error(
                f"Grid size mismatch: expected {expected_size}, "
                f"got {actual_size}"
            )
            return False

        # Validate electron density values are reasonable
        ne_flat = data['ne_grid_flat']
        if not all(isinstance(x, (int, float)) for x in ne_flat):
            self.logger.error("Invalid Ne values (not numeric)")
            return False

        # Check for NaN or Inf
        ne_array = np.array(ne_flat)
        if not np.all(np.isfinite(ne_array)):
            self.logger.error("Invalid Ne values (NaN or Inf detected)")
            return False

        # Check reasonable range (1e6 to 1e13 el/m³)
        if np.min(ne_array) < 0 or np.max(ne_array) > 1e14:
            self.logger.warning(
                f"Ne values outside expected range: "
                f"min={np.min(ne_array):.2e}, max={np.max(ne_array):.2e}"
            )

        return True

    def _reconstruct_grid(self, data: dict) -> Dict[str, Any]:
        """
        Reconstruct 3D grid arrays from flattened message data.

        Args:
            data: Message data dictionary

        Returns:
            Dictionary containing reconstructed grid and metadata
        """
        shape = data['grid_shape']
        n_lat, n_lon, n_alt = shape

        # Reconstruct Ne grid from flattened array
        ne_flat = np.array(data['ne_grid_flat'])
        ne_grid = ne_flat.reshape(shape, order='C')  # Row-major order

        # Generate coordinate grids
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
        """Background thread function for consuming grid messages."""
        try:
            # Subscribe to proc.grid_ready topic
            self.mq_client.subscribe(
                topic_pattern=Topics.PROC_GRID_READY,
                callback=self._on_grid_message,
                queue_name="propagation_grid_subscriber"
            )

            self.logger.info(
                f"Subscribed to {Topics.PROC_GRID_READY}, starting consumption"
            )

            # Start consuming (blocks until stop_consuming called)
            self.mq_client.start_consuming()

        except Exception as e:
            self.logger.error(
                f"Grid subscriber thread error: {e}",
                exc_info=True
            )
        finally:
            self.logger.info("Grid subscriber thread exiting")

    async def get_latest_grid(
        self,
        max_age_seconds: float = 1200.0,
        timeout: float = 30.0
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]]:
        """
        Get latest grid asynchronously.

        Waits for a fresh grid if none is available or current is stale.

        Args:
            max_age_seconds: Maximum acceptable grid age (default: 20 minutes)
            timeout: Maximum wait time for fresh grid (seconds)

        Returns:
            Tuple of (ne_grid, lat, lon, alt, xray_flux) if available, else None
        """
        start_time = asyncio.get_event_loop().time()
        end_time = start_time + timeout

        while asyncio.get_event_loop().time() < end_time:
            with self.grid_lock:
                if self.latest_grid is not None and self.latest_grid_time is not None:
                    # Check grid age
                    age = (datetime.utcnow() - self.latest_grid_time).total_seconds()

                    if age <= max_age_seconds:
                        # Return fresh grid
                        grid = self.latest_grid
                        self.logger.debug(
                            f"Returning grid: {grid['cycle_id']}, "
                            f"age={age:.1f}s"
                        )
                        return (
                            grid['ne_grid'],
                            grid['lat'],
                            grid['lon'],
                            grid['alt'],
                            grid['xray_flux']
                        )
                    else:
                        self.logger.warning(
                            f"Grid is stale: {age:.1f} seconds old "
                            f"(max: {max_age_seconds}), waiting for update..."
                        )

            # Wait before checking again
            await asyncio.sleep(1.0)

        self.logger.warning(
            f"No fresh grid available after {timeout}s wait"
        )
        return None

    def get_latest_grid_sync(
        self,
        max_age_seconds: float = 1200.0
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]]:
        """
        Get latest grid synchronously (non-blocking).

        Args:
            max_age_seconds: Maximum acceptable grid age (seconds)

        Returns:
            Tuple of (ne_grid, lat, lon, alt, xray_flux) if available, else None
        """
        with self.grid_lock:
            if self.latest_grid is None or self.latest_grid_time is None:
                return None

            # Check grid age
            age = (datetime.utcnow() - self.latest_grid_time).total_seconds()

            if age > max_age_seconds:
                self.logger.warning(
                    f"Grid is stale: {age:.1f} seconds old "
                    f"(max: {max_age_seconds})"
                )
                return None

            grid = self.latest_grid
            return (
                grid['ne_grid'],
                grid['lat'],
                grid['lon'],
                grid['alt'],
                grid['xray_flux']
            )

    def get_grid_metadata(self) -> Optional[Dict[str, Any]]:
        """
        Get metadata about latest grid without copying large arrays.

        Returns:
            Dictionary with grid metadata or None
        """
        with self.grid_lock:
            if self.latest_grid is None:
                return None

            return {
                'cycle_id': self.latest_grid.get('cycle_id'),
                'timestamp': self.latest_grid.get('timestamp'),
                'age_seconds': (
                    datetime.utcnow() - self.latest_grid_time
                ).total_seconds() if self.latest_grid_time else None,
                'quality': self.latest_grid.get('quality'),
                'effective_ssn': self.latest_grid.get('effective_ssn'),
                'observations_used': self.latest_grid.get('observations_used'),
                'filter_converged': self.latest_grid.get('filter_converged'),
                'ne_max': float(np.max(self.latest_grid['ne_grid'])),
                'ne_mean': float(np.mean(self.latest_grid['ne_grid'])),
                'shape': self.latest_grid['ne_grid'].shape
            }

    def get_statistics(self) -> Dict[str, int]:
        """
        Get subscriber statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            'grids_received': self.grids_received,
            'grids_invalid': self.grids_invalid,
            'has_grid': self.latest_grid is not None,
            'grid_age_seconds': (
                int((datetime.utcnow() - self.latest_grid_time).total_seconds())
                if self.latest_grid_time else -1
            )
        }
