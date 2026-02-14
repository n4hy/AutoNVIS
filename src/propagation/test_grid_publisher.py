#!/usr/bin/env python3
"""
Test Grid Publisher

Publishes mock electron density grids to proc.grid_ready topic for testing.
This simulates what the SR-UKF filter service would publish.
"""

import sys
from pathlib import Path
import time
import numpy as np
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.common.message_queue import MessageQueueClient, Topics
from src.common.config import get_config


def create_chapman_layer(n_lat=73, n_lon=73, n_alt=55):
    """
    Create Chapman layer ionosphere for testing.

    Args:
        n_lat: Number of latitude points
        n_lon: Number of longitude points
        n_alt: Number of altitude points

    Returns:
        3D electron density grid (el/m³)
    """
    # Grid coordinates
    lat = np.linspace(-90, 90, n_lat)
    lon = np.linspace(-180, 180, n_lon)
    alt = np.linspace(60, 600, n_alt)

    # Chapman layer parameters
    Ne_max = 1e12  # el/m³ (moderate F-layer)
    h_max = 300.0  # km (F-layer peak)
    H = 50.0  # km (scale height)

    # Create 3D grid
    ne_grid = np.zeros((n_lat, n_lon, n_alt))

    for i, h in enumerate(alt):
        z = (h - h_max) / H
        Ne = Ne_max * np.exp(1 - z - np.exp(-z))
        ne_grid[:, :, i] = Ne

    return ne_grid, lat, lon, alt


def publish_grid(mq_client: MessageQueueClient, cycle_id: int):
    """
    Publish mock grid to proc.grid_ready topic.

    Args:
        mq_client: Message queue client
        cycle_id: Cycle number
    """
    print(f"\n{'='*60}")
    print(f"Publishing Grid: cycle_{cycle_id:04d}")
    print(f"{'='*60}")

    # Create Chapman layer
    ne_grid, lat, lon, alt = create_chapman_layer()

    print(f"Grid created:")
    print(f"  Shape: {ne_grid.shape}")
    print(f"  Ne max: {np.max(ne_grid):.2e} el/m³")
    print(f"  Ne mean: {np.mean(ne_grid):.2e} el/m³")

    # Prepare message data
    data = {
        # Grid metadata
        "cycle_id": f"cycle_{cycle_id:04d}",
        "grid_shape": list(ne_grid.shape),
        "grid_timestamp_utc": datetime.utcnow().isoformat() + 'Z',

        # Grid coordinates
        "lat_min": float(lat[0]),
        "lat_max": float(lat[-1]),
        "lat_step": float(lat[1] - lat[0]),
        "lon_min": float(lon[0]),
        "lon_max": float(lon[-1]),
        "lon_step": float(lon[1] - lon[0]),
        "alt_min_km": float(alt[0]),
        "alt_max_km": float(alt[-1]),
        "alt_step_km": float(alt[1] - alt[0]),

        # Electron density grid (flattened to 1D for JSON)
        "ne_grid_flat": ne_grid.flatten(order='C').tolist(),
        "ne_grid_units": "el/m^3",
        "ne_grid_encoding": "row_major",

        # Filter state
        "effective_ssn": 120.0 + np.random.uniform(-10, 10),
        "state_uncertainty": 1.0e10,
        "observations_used": 35 + np.random.randint(-5, 5),

        # Space weather context
        "xray_flux_wm2": 1.5e-6 * (1 + np.random.uniform(-0.2, 0.2)),
        "ap_index": 12.0,
        "f107_sfu": 150.0,

        # Quality metrics
        "grid_quality": "good",
        "data_coverage": 0.85,
        "filter_converged": True
    }

    print(f"\nPublishing to {Topics.PROC_GRID_READY}...")

    # Publish to message queue
    mq_client.publish(
        topic=Topics.PROC_GRID_READY,
        data=data,
        source="test_publisher"
    )

    print(f"✅ Grid published successfully!")
    print(f"   Cycle ID: {data['cycle_id']}")
    print(f"   Timestamp: {data['grid_timestamp_utc']}")
    print(f"   Effective SSN: {data['effective_ssn']:.1f}")
    print(f"   X-ray flux: {data['xray_flux_wm2']:.2e} W/m²")
    print(f"   Grid size: {len(data['ne_grid_flat'])} values")


def main():
    """Main test function."""
    print("=" * 60)
    print("Test Grid Publisher")
    print("=" * 60)
    print()

    # Load configuration
    config = get_config()

    # Create message queue client
    print("Connecting to RabbitMQ...")
    mq_client = MessageQueueClient(
        host=config.services.rabbitmq_host,
        port=config.services.rabbitmq_port,
        username=config.services.rabbitmq_user,
        password=config.services.rabbitmq_password
    )
    print(f"✅ Connected to {config.services.rabbitmq_host}:{config.services.rabbitmq_port}")

    print("\nThis script will publish mock grids every 15 seconds.")
    print("Use Ctrl+C to stop.")
    print()

    cycle = 0

    try:
        while True:
            cycle += 1

            # Publish grid
            publish_grid(mq_client, cycle)

            # Wait for next cycle
            print(f"\nWaiting 15 seconds for next grid...")
            time.sleep(15)

    except KeyboardInterrupt:
        print("\n\nStopping publisher...")
        mq_client.close()
        print("✅ Closed cleanly")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
