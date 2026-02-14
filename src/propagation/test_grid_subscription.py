#!/usr/bin/env python3
"""
Test Grid Subscription

Tests the GridSubscriber class by subscribing to proc.grid_ready
and displaying received grids.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.common.message_queue import MessageQueueClient
from src.common.config import get_config
from src.supervisor.grid_subscriber import GridSubscriber


async def test_async_retrieval():
    """Test async grid retrieval."""
    print("="*70)
    print("Grid Subscriber Test - Async Retrieval")
    print("="*70)
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
    print()

    # Create grid subscriber
    print("Starting GridSubscriber...")
    subscriber = GridSubscriber(mq_client)
    subscriber.start()
    print("✅ Subscriber started")
    print()

    print("Waiting for grid updates...")
    print("(Run test_grid_publisher.py in another terminal to publish grids)")
    print()

    try:
        for i in range(10):  # Wait for up to 10 grids
            print(f"\n--- Attempt {i+1} ---")

            # Get latest grid (wait up to 60 seconds)
            grid_data = await subscriber.get_latest_grid(
                max_age_seconds=1200.0,  # 20 minutes
                timeout=60.0  # Wait 60 seconds for grid
            )

            if grid_data:
                ne_grid, lat, lon, alt, xray = grid_data

                print(f"✅ Grid received!")
                print(f"   Shape: {ne_grid.shape}")
                print(f"   Lat range: {lat[0]:.1f}° to {lat[-1]:.1f}°")
                print(f"   Lon range: {lon[0]:.1f}° to {lon[-1]:.1f}°")
                print(f"   Alt range: {alt[0]:.1f} to {alt[-1]:.1f} km")
                print(f"   Ne max: {ne_grid.max():.2e} el/m³")
                print(f"   Ne mean: {ne_grid.mean():.2e} el/m³")
                print(f"   X-ray flux: {xray:.2e} W/m²")

                # Get metadata
                metadata = subscriber.get_grid_metadata()
                if metadata:
                    print(f"\n   Metadata:")
                    print(f"     Cycle ID: {metadata['cycle_id']}")
                    print(f"     Timestamp: {metadata['timestamp']}")
                    print(f"     Age: {metadata['age_seconds']:.1f} seconds")
                    print(f"     Quality: {metadata['quality']}")
                    print(f"     Effective SSN: {metadata['effective_ssn']:.1f}")
                    print(f"     Observations: {metadata['observations_used']}")
                    print(f"     Converged: {metadata['filter_converged']}")

                # Get statistics
                stats = subscriber.get_statistics()
                print(f"\n   Subscriber Stats:")
                print(f"     Grids received: {stats['grids_received']}")
                print(f"     Grids invalid: {stats['grids_invalid']}")

            else:
                print(f"❌ No grid received within 60 seconds")

            # Wait before next attempt
            await asyncio.sleep(5)

    except KeyboardInterrupt:
        print("\n\nStopping test...")

    finally:
        # Clean up
        subscriber.stop()
        mq_client.close()
        print("\n✅ Test completed")


def test_sync_retrieval():
    """Test synchronous grid retrieval."""
    print("="*70)
    print("Grid Subscriber Test - Sync Retrieval")
    print("="*70)
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
    print(f"✅ Connected")
    print()

    # Create grid subscriber
    print("Starting GridSubscriber...")
    subscriber = GridSubscriber(mq_client)
    subscriber.start()
    print("✅ Subscriber started")
    print()

    print("Polling for grid (non-blocking)...")
    print("(This will return None immediately if no grid is cached)")
    print()

    try:
        import time
        for i in range(20):  # Try 20 times
            print(f"Poll {i+1}...", end=" ")

            grid_data = subscriber.get_latest_grid_sync(max_age_seconds=1200.0)

            if grid_data:
                ne_grid, lat, lon, alt, xray = grid_data
                print(f"✅ Got grid! Shape: {ne_grid.shape}, "
                      f"Ne_max: {ne_grid.max():.2e} el/m³")
                break
            else:
                print("No grid yet")

            time.sleep(3)  # Wait 3 seconds between polls

    except KeyboardInterrupt:
        print("\n\nStopping test...")

    finally:
        subscriber.stop()
        mq_client.close()
        print("\n✅ Test completed")


def main():
    """Main test function."""
    import argparse

    parser = argparse.ArgumentParser(description="Test grid subscription")
    parser.add_argument(
        "--mode",
        choices=["async", "sync"],
        default="async",
        help="Test mode (async or sync)"
    )

    args = parser.parse_args()

    if args.mode == "async":
        asyncio.run(test_async_retrieval())
    else:
        test_sync_retrieval()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
