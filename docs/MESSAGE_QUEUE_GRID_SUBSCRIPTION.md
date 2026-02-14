# Message Queue Grid Subscription Guide

**Topic**: `proc.grid_ready`
**Purpose**: Receive electron density grids from SR-UKF filter for propagation prediction
**Date**: 2026-02-13

---

## Overview

This guide provides complete instructions for subscribing to the `proc.grid_ready` message queue topic to receive real-time electron density grids from the SR-UKF filter for ray tracing and propagation prediction.

---

## Message Format

### Topic
```
proc.grid_ready
```

### Message Structure

The `proc.grid_ready` message follows the standard Auto-NVIS message format:

```python
from src.common.message_queue import Message

message = Message(
    topic="proc.grid_ready",
    timestamp="2026-02-14T02:30:00.123456Z",  # ISO format UTC
    source="assimilation",  # SR-UKF filter service
    data={
        # Grid metadata
        "cycle_id": "cycle_0042",
        "grid_shape": [73, 73, 55],  # [n_lat, n_lon, n_alt]
        "grid_timestamp_utc": "2026-02-14T02:30:00.000000Z",

        # Grid coordinates
        "lat_min": -90.0,
        "lat_max": 90.0,
        "lat_step": 2.5,
        "lon_min": -180.0,
        "lon_max": 180.0,
        "lon_step": 5.0,
        "alt_min_km": 60.0,
        "alt_max_km": 600.0,
        "alt_step_km": 10.0,

        # Electron density grid (flattened to 1D array for JSON)
        "ne_grid_flat": [1.2e11, 1.3e11, ...],  # 73*73*55 = 292,855 values
        "ne_grid_units": "el/m^3",
        "ne_grid_encoding": "row_major",  # [lat][lon][alt] order

        # Filter state
        "effective_ssn": 120.5,  # Effective sunspot number
        "state_uncertainty": 1.23e10,  # trace(P)
        "observations_used": 42,  # Number of obs assimilated

        # Space weather context
        "xray_flux_wm2": 1.5e-6,  # Current GOES X-ray flux
        "ap_index": 12.0,  # Geomagnetic index
        "f107_sfu": 150.0,  # Solar radio flux

        # Quality metrics
        "grid_quality": "good",  # "good", "fair", "poor", "stale"
        "data_coverage": 0.85,  # Fraction of grid with observations
        "filter_converged": true
    }
)
```

### Data Types

```python
# Grid shape
grid_shape: List[int]  # [n_lat, n_lon, n_alt]

# Coordinates
lat_min, lat_max, lat_step: float  # degrees
lon_min, lon_max, lon_step: float  # degrees
alt_min_km, alt_max_km, alt_step_km: float  # kilometers

# Electron density
ne_grid_flat: List[float]  # Flattened 3D array
ne_grid_units: str = "el/m^3"
ne_grid_encoding: str = "row_major"  # [lat][lon][alt]

# Filter state
effective_ssn: float
state_uncertainty: float
observations_used: int

# Space weather
xray_flux_wm2: float  # W/m²
ap_index: float
f107_sfu: float  # Solar flux units

# Quality
grid_quality: str  # Enum: "good", "fair", "poor", "stale"
data_coverage: float  # 0.0 to 1.0
filter_converged: bool
```

---

## Subscription Methods

### Method 1: Synchronous Subscription (Blocking)

**Use Case**: Dedicated subscriber service running in its own process

```python
from src.common.message_queue import MessageQueueClient, Topics, Message
from src.common.config import get_config
import numpy as np

def on_grid_ready(message: Message):
    """
    Callback function for proc.grid_ready messages

    Args:
        message: Message object containing grid data
    """
    print(f"Received grid at {message.timestamp}")

    # Extract grid data
    data = message.data
    cycle_id = data['cycle_id']
    shape = data['grid_shape']

    # Reconstruct Ne grid from flattened array
    ne_flat = np.array(data['ne_grid_flat'])
    ne_grid = ne_flat.reshape(shape)  # (73, 73, 55)

    # Get grid coordinates
    n_lat, n_lon, n_alt = shape
    lat = np.linspace(data['lat_min'], data['lat_max'], n_lat)
    lon = np.linspace(data['lon_min'], data['lon_max'], n_lon)
    alt = np.linspace(data['alt_min_km'], data['alt_max_km'], n_alt)

    # Extract space weather
    xray_flux = data['xray_flux_wm2']

    print(f"  Cycle: {cycle_id}")
    print(f"  Grid shape: {ne_grid.shape}")
    print(f"  Ne max: {np.max(ne_grid):.2e} el/m³")
    print(f"  X-ray flux: {xray_flux:.2e} W/m²")

    # TODO: Process grid (pass to ray tracer, etc.)

# Create message queue client
config = get_config()
mq_client = MessageQueueClient(
    host=config.services.rabbitmq_host,
    port=config.services.rabbitmq_port,
    username=config.services.rabbitmq_user,
    password=config.services.rabbitmq_password
)

# Subscribe to proc.grid_ready
mq_client.subscribe(
    topic_pattern=Topics.PROC_GRID_READY,
    callback=on_grid_ready,
    queue_name="propagation_grid_subscriber"  # Named queue for persistence
)

# Start consuming (blocking)
print("Waiting for grid updates...")
mq_client.start_consuming()
```

---

### Method 2: Asynchronous Subscription (Event-Driven)

**Use Case**: Integration with async system orchestrator

```python
import asyncio
import threading
from queue import Queue
from src.common.message_queue import MessageQueueClient, Topics, Message

class AsyncGridSubscriber:
    """
    Async-friendly grid subscriber using background thread
    """

    def __init__(self, mq_client: MessageQueueClient):
        self.mq_client = mq_client
        self.grid_queue = Queue(maxsize=10)
        self.subscriber_thread = None
        self.running = False

    def start(self):
        """Start subscriber in background thread"""
        self.running = True
        self.subscriber_thread = threading.Thread(
            target=self._consume_thread,
            daemon=True
        )
        self.subscriber_thread.start()

    def stop(self):
        """Stop subscriber thread"""
        self.running = False
        self.mq_client.stop_consuming()
        if self.subscriber_thread:
            self.subscriber_thread.join(timeout=5)

    def _on_grid_ready(self, message: Message):
        """Internal callback for grid messages"""
        # Put message in queue for async processing
        try:
            self.grid_queue.put(message, block=False)
        except:
            # Queue full - drop oldest message
            try:
                self.grid_queue.get_nowait()
                self.grid_queue.put(message, block=False)
            except:
                pass

    def _consume_thread(self):
        """Background thread for consuming messages"""
        self.mq_client.subscribe(
            topic_pattern=Topics.PROC_GRID_READY,
            callback=self._on_grid_ready,
            queue_name="propagation_grid_async"
        )

        try:
            self.mq_client.start_consuming()
        except Exception as e:
            print(f"Subscriber error: {e}")

    async def get_latest_grid(self, timeout: float = 5.0):
        """
        Get latest grid from queue (async)

        Args:
            timeout: Maximum wait time (seconds)

        Returns:
            Message object or None if timeout
        """
        end_time = asyncio.get_event_loop().time() + timeout

        while asyncio.get_event_loop().time() < end_time:
            if not self.grid_queue.empty():
                return self.grid_queue.get()

            await asyncio.sleep(0.1)

        return None

# Usage in async context
async def main():
    mq_client = MessageQueueClient()
    subscriber = AsyncGridSubscriber(mq_client)

    # Start background subscriber
    subscriber.start()

    try:
        while True:
            # Wait for grid update
            message = await subscriber.get_latest_grid(timeout=30.0)

            if message:
                print(f"Got grid: {message.data['cycle_id']}")
                # Process grid...
            else:
                print("No grid received in 30 seconds")

            await asyncio.sleep(1)

    finally:
        subscriber.stop()

asyncio.run(main())
```

---

### Method 3: Polling Latest Grid (Simple)

**Use Case**: Periodic polling from file system or cache

```python
import asyncio
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta

class GridPoller:
    """
    Poll for latest grid from shared storage
    (Alternative when message queue subscription is not feasible)
    """

    def __init__(self, grid_dir: Path):
        self.grid_dir = grid_dir
        self.latest_grid_path = grid_dir / "latest_ne_grid.npz"
        self.last_modified = None

    async def get_latest_grid(self) -> dict:
        """
        Get latest grid from file system

        Returns:
            Dictionary with grid data or None if no update
        """
        # Check if file exists and is updated
        if not self.latest_grid_path.exists():
            return None

        stat = self.latest_grid_path.stat()
        modified_time = datetime.fromtimestamp(stat.st_mtime)

        # Check if grid is new
        if self.last_modified and modified_time <= self.last_modified:
            return None

        # Check if grid is stale (> 20 minutes old)
        age = datetime.now() - modified_time
        if age > timedelta(minutes=20):
            print(f"Warning: Grid is {age.total_seconds()/60:.1f} min old")

        # Load grid
        data = np.load(self.latest_grid_path)

        self.last_modified = modified_time

        return {
            'ne_grid': data['ne_grid'],
            'lat': data['lat'],
            'lon': data['lon'],
            'alt': data['alt'],
            'xray_flux': float(data['xray_flux']),
            'timestamp': modified_time,
            'cycle_id': str(data.get('cycle_id', 'unknown'))
        }

# Usage
poller = GridPoller(Path("/data/grids"))

async def poll_loop():
    while True:
        grid_data = await poller.get_latest_grid()

        if grid_data:
            print(f"New grid: {grid_data['cycle_id']}")
            # Process grid...

        await asyncio.sleep(60)  # Poll every minute
```

---

## System Orchestrator Integration

### Complete Implementation

**File**: `src/supervisor/grid_subscriber.py` (new)

```python
"""
Grid Subscriber for System Orchestrator

Manages subscription to proc.grid_ready messages and provides
async access to latest electron density grids.
"""

import asyncio
import threading
from queue import Queue
from typing import Optional, Tuple
import numpy as np
from datetime import datetime, timedelta

from src.common.message_queue import MessageQueueClient, Topics, Message
from src.common.logging_config import ServiceLogger


class GridSubscriber:
    """
    Subscribes to proc.grid_ready and provides async access to grids
    """

    def __init__(self, mq_client: MessageQueueClient):
        """
        Initialize grid subscriber

        Args:
            mq_client: Message queue client
        """
        self.mq_client = mq_client
        self.logger = ServiceLogger("supervisor", "grid_subscriber")

        # Latest grid storage
        self.latest_grid = None
        self.latest_grid_time = None
        self.grid_lock = threading.Lock()

        # Background subscription
        self.grid_queue = Queue(maxsize=5)
        self.subscriber_thread = None
        self.running = False

    def start(self):
        """Start background subscriber thread"""
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
        """Stop subscriber thread"""
        if not self.running:
            return

        self.running = False
        self.mq_client.stop_consuming()

        if self.subscriber_thread:
            self.subscriber_thread.join(timeout=5)

        self.logger.info("Grid subscriber stopped")

    def _on_grid_message(self, message: Message):
        """
        Handle incoming grid message

        Args:
            message: Grid message from proc.grid_ready
        """
        try:
            data = message.data
            cycle_id = data.get('cycle_id', 'unknown')

            self.logger.info(f"Received grid: {cycle_id}")

            # Validate message
            if not self._validate_grid_message(data):
                self.logger.error(f"Invalid grid message: {cycle_id}")
                return

            # Reconstruct grid
            grid_data = self._reconstruct_grid(data)

            # Store latest grid
            with self.grid_lock:
                self.latest_grid = grid_data
                self.latest_grid_time = datetime.fromisoformat(
                    message.timestamp.rstrip('Z')
                )

            self.logger.info(
                f"Grid stored: {cycle_id}, "
                f"Ne_max={np.max(grid_data['ne_grid']):.2e} el/m³"
            )

        except Exception as e:
            self.logger.error(f"Error processing grid message: {e}", exc_info=True)

    def _validate_grid_message(self, data: dict) -> bool:
        """
        Validate grid message contains required fields

        Args:
            data: Message data dictionary

        Returns:
            True if valid
        """
        required_fields = [
            'grid_shape', 'ne_grid_flat',
            'lat_min', 'lat_max', 'lon_min', 'lon_max',
            'alt_min_km', 'alt_max_km'
        ]

        for field in required_fields:
            if field not in data:
                self.logger.error(f"Missing required field: {field}")
                return False

        # Validate grid shape
        shape = data['grid_shape']
        if len(shape) != 3:
            self.logger.error(f"Invalid grid shape: {shape}")
            return False

        # Validate grid size
        expected_size = shape[0] * shape[1] * shape[2]
        actual_size = len(data['ne_grid_flat'])

        if expected_size != actual_size:
            self.logger.error(
                f"Grid size mismatch: expected {expected_size}, got {actual_size}"
            )
            return False

        return True

    def _reconstruct_grid(self, data: dict) -> dict:
        """
        Reconstruct grid from message data

        Args:
            data: Message data dictionary

        Returns:
            Dictionary with grid arrays and metadata
        """
        shape = data['grid_shape']
        n_lat, n_lon, n_alt = shape

        # Reconstruct Ne grid
        ne_flat = np.array(data['ne_grid_flat'])
        ne_grid = ne_flat.reshape(shape)

        # Reconstruct coordinate grids
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
            'timestamp': data.get('grid_timestamp_utc', '')
        }

    def _consume_thread(self):
        """Background thread for consuming messages"""
        try:
            self.mq_client.subscribe(
                topic_pattern=Topics.PROC_GRID_READY,
                callback=self._on_grid_message,
                queue_name="propagation_grid_subscriber"
            )

            self.logger.info("Starting grid message consumption")
            self.mq_client.start_consuming()

        except Exception as e:
            self.logger.error(f"Grid subscriber thread error: {e}", exc_info=True)

    async def get_latest_grid(
        self,
        max_age_seconds: float = 1200.0,
        timeout: float = 30.0
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]]:
        """
        Get latest grid (async)

        Args:
            max_age_seconds: Maximum grid age (default: 20 minutes)
            timeout: Maximum wait time if no grid available (seconds)

        Returns:
            Tuple of (ne_grid, lat, lon, alt, xray_flux) or None
        """
        start_time = asyncio.get_event_loop().time()
        end_time = start_time + timeout

        while asyncio.get_event_loop().time() < end_time:
            with self.grid_lock:
                if self.latest_grid is not None:
                    # Check grid age
                    age = (datetime.utcnow() - self.latest_grid_time).total_seconds()

                    if age <= max_age_seconds:
                        # Return grid
                        grid = self.latest_grid
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
                            f"(max: {max_age_seconds})"
                        )

            # Wait before checking again
            await asyncio.sleep(1.0)

        self.logger.warning(f"No fresh grid available after {timeout}s wait")
        return None

    def get_latest_grid_sync(
        self,
        max_age_seconds: float = 1200.0
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]]:
        """
        Get latest grid (synchronous)

        Args:
            max_age_seconds: Maximum grid age (seconds)

        Returns:
            Tuple of (ne_grid, lat, lon, alt, xray_flux) or None
        """
        with self.grid_lock:
            if self.latest_grid is None:
                return None

            # Check grid age
            age = (datetime.utcnow() - self.latest_grid_time).total_seconds()

            if age > max_age_seconds:
                self.logger.warning(f"Grid is stale: {age:.1f} seconds old")
                return None

            grid = self.latest_grid
            return (
                grid['ne_grid'],
                grid['lat'],
                grid['lon'],
                grid['alt'],
                grid['xray_flux']
            )
```

---

## Modified System Orchestrator

**File**: `src/supervisor/system_orchestrator.py` (modifications)

```python
from .grid_subscriber import GridSubscriber

class SystemOrchestrator:
    def __init__(self, ...):
        # ... existing init ...

        # Add grid subscriber
        self.grid_subscriber = None
        if self.mq_client:
            self.grid_subscriber = GridSubscriber(self.mq_client)
            self.grid_subscriber.start()

    async def _get_ionospheric_grid(self):
        """
        Get ionospheric grid from SR-UKF filter.

        Now uses GridSubscriber to get grids from proc.grid_ready topic.
        """
        if self.grid_subscriber:
            # Try to get grid from message queue
            grid_data = await self.grid_subscriber.get_latest_grid(
                max_age_seconds=1200.0,  # 20 minutes
                timeout=30.0  # Wait up to 30 seconds
            )

            if grid_data is not None:
                self.logger.info("Using grid from message queue")
                self._ne_grid_cache = grid_data
                return grid_data
            else:
                self.logger.warning("No fresh grid from message queue")

        # Fallback to cached or placeholder
        if self._ne_grid_cache is not None:
            self.logger.info("Using cached grid")
            return self._ne_grid_cache

        # Last resort: create Chapman layer placeholder
        self.logger.warning("Creating placeholder Chapman layer")
        # ... existing Chapman layer code ...
```

---

## Testing

### Test Script

**File**: `src/propagation/test_grid_subscription.py`

```python
#!/usr/bin/env python3
"""
Test grid subscription
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.common.message_queue import MessageQueueClient
from src.common.config import get_config
from src.supervisor.grid_subscriber import GridSubscriber

async def test_subscriber():
    """Test grid subscriber"""
    print("Starting grid subscriber test...")

    config = get_config()
    mq_client = MessageQueueClient(
        host=config.services.rabbitmq_host,
        port=config.services.rabbitmq_port
    )

    subscriber = GridSubscriber(mq_client)
    subscriber.start()

    print("Waiting for grid...")

    try:
        # Wait for grid (max 60 seconds)
        grid_data = await subscriber.get_latest_grid(timeout=60.0)

        if grid_data:
            ne_grid, lat, lon, alt, xray = grid_data
            print(f"✅ Grid received!")
            print(f"   Shape: {ne_grid.shape}")
            print(f"   Max Ne: {ne_grid.max():.2e} el/m³")
            print(f"   X-ray: {xray:.2e} W/m²")
        else:
            print("❌ No grid received")

    finally:
        subscriber.stop()

if __name__ == "__main__":
    asyncio.run(test_subscriber())
```

---

## Summary

**Subscription Methods**:
1. **Synchronous** - Blocking, for dedicated services
2. **Asynchronous** - Non-blocking, for system orchestrator
3. **Polling** - File-based fallback

**Recommended Approach**: Use `GridSubscriber` class in async system orchestrator

**Key Files to Create**:
- `src/supervisor/grid_subscriber.py` - Grid subscription manager
- `src/propagation/test_grid_subscription.py` - Test script

**Key Files to Modify**:
- `src/supervisor/system_orchestrator.py` - Add GridSubscriber instance

**Next Steps**:
1. Create `GridSubscriber` class
2. Add to system orchestrator
3. Test with mock grid publisher
4. Integrate with SR-UKF filter service

---

**Status**: Implementation ready
**Estimated Time**: 2-4 hours to implement and test
