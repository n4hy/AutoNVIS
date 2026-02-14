"""
Brutal System Integration Tests

This test suite exercises the ENTIRE Auto-NVIS system end-to-end with:
- Massive data loads
- Concurrent operations across all services
- Full propagation pipeline
- CPU-melting computational workloads
- Memory stress testing

Goal: Bend the CPU to overheat
"""

import pytest
import numpy as np
import time
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import psutil
import os

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from common.config import GridConfig, AutoNVISConfig
from common.message_queue import MessageQueueClient, Topics

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src' / 'propagation' / 'services'))
from propagation_service import PropagationService


class TestFullDataPipeline:
    """Test complete data pipeline from observations to propagation products"""

    def test_end_to_end_single_cycle(self):
        """Test single complete assimilation → propagation cycle"""
        # Setup
        grid_config = GridConfig(lat_step=10.0, lon_step=10.0, alt_step=50.0)

        mq_client = MessageQueueClient()
        prop_service = PropagationService(
            tx_lat=40.0, tx_lon=-105.0,
            freq_step=1.0
        )

        # Create electron density grid (simulating filter output)
        # Use actual grid sizes (np.arange can create different size than n_* properties)
        lat_grid = grid_config.get_lat_grid()
        lon_grid = grid_config.get_lon_grid()
        alt_grid = grid_config.get_alt_grid()

        ne_grid = np.full(
            (len(lat_grid), len(lon_grid), len(alt_grid)),
            1e11
        )

        # Initialize propagation with ionospheric grid
        prop_service.initialize_ray_tracer(
            ne_grid=ne_grid,
            lat_grid=grid_config.get_lat_grid(),
            lon_grid=grid_config.get_lon_grid(),
            alt_grid=grid_config.get_alt_grid(),
            xray_flux=1e-6
        )

        # Calculate LUF/MUF
        result = prop_service.calculate_luf_muf()

        assert result is not None
        assert 'luf_mhz' in result
        assert 'muf_mhz' in result

        # Publish to message queue
        mq_client.publish(Topics.OUT_FREQUENCY_PLAN, result, source="integration-test")

        mq_client.close()

    def test_repeated_cycles(self):
        """Test multiple consecutive assimilation cycles (simulate 1 hour)"""
        grid_config = GridConfig(lat_step=20.0, lon_step=20.0, alt_step=100.0)

        prop_service = PropagationService(
            tx_lat=40.0, tx_lon=-105.0,
            freq_step=2.0
        )

        # Run 4 cycles (simulating 1 hour at 15-min intervals)
        results = []
        total_time = 0.0

        for cycle in range(4):
            # Create electron density grid (simulating changing ionosphere)
            lat_grid = grid_config.get_lat_grid()
            lon_grid = grid_config.get_lon_grid()
            alt_grid = grid_config.get_alt_grid()
            ne_grid = np.full(
                (len(lat_grid), len(lon_grid), len(alt_grid)),
                1e11 + cycle * 1e10
            )

            # Initialize ray tracer
            start = time.time()
            prop_service.initialize_ray_tracer(
                ne_grid=ne_grid,
                lat_grid=grid_config.get_lat_grid(),
                lon_grid=grid_config.get_lon_grid(),
                alt_grid=grid_config.get_alt_grid(),
                xray_flux=1e-6 * (1 + cycle * 0.1)
            )

            # Calculate products
            result = prop_service.calculate_luf_muf()
            elapsed = time.time() - start

            total_time += elapsed
            results.append(result)

            print(f"\nCycle {cycle+1}: {elapsed:.2f}s, MUF={result['muf_mhz']:.1f} MHz")

        print(f"\nTotal time for 4 cycles: {total_time:.2f}s")
        print(f"Average cycle time: {total_time/4:.2f}s")

        assert len(results) == 4
        assert all(r is not None for r in results)


class TestConcurrentSystemLoad:
    """Test system under heavy concurrent load"""

    def test_concurrent_propagation_calculations(self):
        """Run multiple propagation calculations concurrently"""
        grid_config = GridConfig(lat_step=20.0, lon_step=20.0, alt_step=100.0)

        def run_propagation(instance_id):
            prop_service = PropagationService(
                tx_lat=40.0, tx_lon=-105.0,
                freq_step=1.0
            )

            # Create Ne grid
            lat_grid = grid_config.get_lat_grid()
            lon_grid = grid_config.get_lon_grid()
            alt_grid = grid_config.get_alt_grid()
            ne_grid = np.full(
                (len(lat_grid), len(lon_grid), len(alt_grid)),
                1e11 * (1.0 + instance_id * 0.1)
            )

            prop_service.initialize_ray_tracer(
                ne_grid=ne_grid,
                lat_grid=grid_config.get_lat_grid(),
                lon_grid=grid_config.get_lon_grid(),
                alt_grid=grid_config.get_alt_grid(),
                xray_flux=1e-6
            )

            result = prop_service.calculate_luf_muf()
            return result['muf_mhz']

        # Run 10 concurrent propagation instances
        start = time.time()
        with ThreadPoolExecutor(max_workers=10) as executor:
            mufs = list(executor.map(run_propagation, range(10)))

        elapsed = time.time() - start

        print(f"\n10 concurrent propagation calculations: {elapsed:.2f}s")
        print(f"MUFs: {mufs}")

        assert len(mufs) == 10
        assert all(m > 0 for m in mufs)

    def test_message_queue_under_load(self):
        """Stress test message queue with high-frequency publishing"""
        mq_client = MessageQueueClient()

        received_count = [0]
        lock = threading.Lock()

        def callback(msg):
            with lock:
                received_count[0] += 1

        # Use a topic that exists
        test_topic = "test.stress"
        mq_client.subscribe(test_topic, callback)
        time.sleep(0.2)

        # Publish 10,000 messages from multiple threads
        def publish_batch(batch_id):
            for i in range(1000):
                mq_client.publish(test_topic, {
                    "batch": batch_id,
                    "seq": i,
                    "data": list(range(100))
                }, source=f"batch-{batch_id}")

        start = time.time()
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(publish_batch, i) for i in range(10)]
            for f in futures:
                f.result()

        elapsed = time.time() - start
        time.sleep(2.0)  # Allow message delivery

        throughput = 10000 / elapsed
        print(f"\nPublished 10,000 messages in {elapsed:.2f}s ({throughput:.0f} msg/s)")
        print(f"Received {received_count[0]} messages")

        assert received_count[0] >= 9000  # Allow some loss

        mq_client.close()


class TestMassiveComputationalLoad:
    """CPU-melting tests designed to maximize computational load"""

    def test_ultra_fine_frequency_sweep(self):
        """Test with extremely fine frequency resolution (CPU intensive)"""
        grid_config = GridConfig(lat_step=10.0, lon_step=10.0, alt_step=50.0)

        # Ultra-fine sweep: 0.05 MHz steps from 2-15 MHz = 261 frequencies!
        prop_service = PropagationService(
            tx_lat=40.0, tx_lon=-105.0,
            freq_min=2.0,
            freq_max=15.0,
            freq_step=0.05,
            elevation_min=75.0,
            elevation_max=90.0,
            elevation_step=1.0,
            azimuth_step=15.0
        )
        # 261 freqs * 16 elev * 24 az = 100,224 ray traces!

        lat_grid = grid_config.get_lat_grid()
        lon_grid = grid_config.get_lon_grid()
        alt_grid = grid_config.get_alt_grid()
        ne_grid = np.full(
            (len(lat_grid), len(lon_grid), len(alt_grid)),
            3e11
        )

        prop_service.initialize_ray_tracer(
            ne_grid=ne_grid,
            lat_grid=grid_config.get_lat_grid(),
            lon_grid=grid_config.get_lon_grid(),
            alt_grid=grid_config.get_alt_grid(),
            xray_flux=1e-6
        )

        # This should stress the CPU significantly
        print("\nStarting ultra-fine frequency sweep (100,000+ ray traces)...")
        start = time.time()
        result = prop_service.calculate_luf_muf()
        elapsed = time.time() - start

        print(f"Ultra-fine sweep completed in {elapsed:.2f}s")
        print(f"Estimated throughput: ~{100224/elapsed:.0f} rays/second")

        assert result is not None
        assert elapsed > 1.0  # Should take significant time

    def test_massive_grid_processing(self):
        """Test with production-scale grid (CPU and memory intensive)"""
        # Production grid: 73 x 72 x 55 = 289,080 states
        grid_config = GridConfig(
            lat_min=-90.0, lat_max=90.0, lat_step=2.5,
            lon_min=-180.0, lon_max=180.0, lon_step=5.0,
            alt_min=60.0, alt_max=600.0, alt_step=10.0
        )

        print(f"\nCreating massive grid ({grid_config.total_points:,} points)...")

        ne_grid = np.zeros((grid_config.n_lat, grid_config.n_lon, grid_config.n_alt))

        lat_grid = grid_config.get_lat_grid()
        lon_grid = grid_config.get_lon_grid()
        alt_grid = grid_config.get_alt_grid()

        print("Filling grid with Chapman layer profile...")
        start = time.time()

        for i_lat, lat in enumerate(lat_grid):
            for i_lon, lon in enumerate(lon_grid):
                for i_alt, alt in enumerate(alt_grid):
                    # Chapman layer with latitude variation
                    h = alt - 300.0
                    H = 50.0
                    lat_factor = np.cos(np.radians(lat)) ** 2
                    ne_peak = 1e12 * lat_factor
                    ne_grid[i_lat, i_lon, i_alt] = ne_peak * np.exp(
                        0.5 * (1 - h/H - np.exp(-h/H))
                    )

        elapsed = time.time() - start
        print(f"Grid filling took {elapsed:.2f}s")

        # Convert to vector (memory stress)
        print("Converting to vector...")
        start = time.time()
        vec = ne_grid.ravel()
        elapsed = time.time() - start

        print(f"Vector conversion took {elapsed:.2f}s")
        print(f"Vector size: {vec.nbytes / 1024 / 1024:.1f} MB")

        assert vec.size == grid_config.total_points

    def test_parallel_multi_location_propagation(self):
        """Test propagation from multiple transmitter locations in parallel"""
        grid_config = GridConfig(lat_step=10.0, lon_step=10.0, alt_step=50.0)

        # Multiple transmitter locations
        tx_locations = [
            (40.0, -105.0),   # Boulder, CO
            (51.5, 0.0),      # London, UK
            (-33.9, 18.4),    # Cape Town, SA
            (35.7, 139.7),    # Tokyo, Japan
            (-37.8, 144.9),   # Melbourne, Australia
        ]

        lat_grid = grid_config.get_lat_grid()
        lon_grid = grid_config.get_lon_grid()
        alt_grid = grid_config.get_alt_grid()
        ne_grid = np.full(
            (len(lat_grid), len(lon_grid), len(alt_grid)),
            2e11
        )

        def propagate_from_location(loc):
            lat, lon = loc

            prop_service = PropagationService(
                tx_lat=lat,
                tx_lon=lon,
                freq_step=1.0
            )

            prop_service.initialize_ray_tracer(
                ne_grid=ne_grid,
                lat_grid=grid_config.get_lat_grid(),
                lon_grid=grid_config.get_lon_grid(),
                alt_grid=grid_config.get_alt_grid(),
                xray_flux=1e-6
            )

            result = prop_service.calculate_luf_muf()
            return (lat, lon, result['muf_mhz'])

        print(f"\nCalculating propagation from {len(tx_locations)} locations in parallel...")
        start = time.time()

        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(propagate_from_location, tx_locations))

        elapsed = time.time() - start

        print(f"Multi-location propagation completed in {elapsed:.2f}s")
        for lat, lon, muf in results:
            print(f"  ({lat:.1f}, {lon:.1f}): MUF = {muf:.1f} MHz")

        assert len(results) == 5

    def test_sustained_cpu_load(self):
        """Sustained high CPU load for extended period"""
        grid_config = GridConfig(lat_step=10.0, lon_step=10.0, alt_step=50.0)

        prop_service = PropagationService(
            tx_lat=40.0, tx_lon=-105.0,
            freq_step=0.5
        )

        # Run continuously for 60 seconds
        print("\nRunning sustained CPU load test for 60 seconds...")
        start_time = time.time()
        iteration = 0

        while time.time() - start_time < 60.0:
            # Create Ne grid
            lat_grid = grid_config.get_lat_grid()
            lon_grid = grid_config.get_lon_grid()
            alt_grid = grid_config.get_alt_grid()
            ne_grid = np.full(
                (len(lat_grid), len(lon_grid), len(alt_grid)),
                1e11 * (1.0 + 0.1 * np.sin(iteration * 0.1))
            )

            prop_service.initialize_ray_tracer(
                ne_grid=ne_grid,
                lat_grid=grid_config.get_lat_grid(),
                lon_grid=grid_config.get_lon_grid(),
                alt_grid=grid_config.get_alt_grid(),
                xray_flux=1e-6
            )

            prop_service.calculate_luf_muf()

            iteration += 1

            # Monitor CPU usage
            if iteration % 5 == 0:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                print(f"  Iteration {iteration}, CPU: {cpu_percent:.1f}%")

        elapsed = time.time() - start_time
        print(f"\nCompleted {iteration} iterations in {elapsed:.2f}s")
        print(f"Average: {elapsed/iteration:.2f}s per iteration")


class TestMemoryStress:
    """Tests designed to stress system memory"""

    def test_multiple_large_grids(self):
        """Allocate multiple large grids simultaneously"""
        grid_config = GridConfig(
            lat_min=-90.0, lat_max=90.0, lat_step=5.0,
            lon_min=-180.0, lon_max=180.0, lon_step=5.0,
            alt_min=60.0, alt_max=600.0, alt_step=10.0
        )
        # 37 * 73 * 55 = 148,555 points per grid

        print(f"\nAllocating 10 large grids ({grid_config.total_points:,} points each)...")

        grids = []
        start = time.time()

        for i in range(10):
            lat_grid = grid_config.get_lat_grid()
            lon_grid = grid_config.get_lon_grid()
            alt_grid = grid_config.get_alt_grid()
            ne_grid = np.full(
                (len(lat_grid), len(lon_grid), len(alt_grid)),
                1e11 * (1.0 + i * 0.1)
            )
            grids.append(ne_grid)

        elapsed = time.time() - start

        # Calculate total memory usage
        total_mem_mb = sum(grid.nbytes for grid in grids) / 1024 / 1024

        print(f"Allocated 10 grids in {elapsed:.2f}s")
        print(f"Total memory: {total_mem_mb:.1f} MB")

        assert len(grids) == 10

        # Clean up
        del grids

    def test_rapid_allocate_deallocate(self):
        """Rapidly allocate and deallocate large arrays"""
        grid_config = GridConfig(lat_step=5.0, lon_step=5.0, alt_step=10.0)

        print("\nRapidly allocating/deallocating 100 grids...")
        start = time.time()

        for i in range(100):
            lat_grid = grid_config.get_lat_grid()
            lon_grid = grid_config.get_lon_grid()
            alt_grid = grid_config.get_alt_grid()
            ne_grid = np.full(
                (len(lat_grid), len(lon_grid), len(alt_grid)),
                float(i)
            )
            vec = ne_grid.ravel()

            # Immediately deallocate
            del ne_grid, vec

        elapsed = time.time() - start
        print(f"100 allocate/deallocate cycles in {elapsed:.2f}s")


class TestSystemResilience:
    """Test system resilience under stress"""

    def test_recovery_from_errors(self):
        """Test that system continues after errors"""
        grid_config = GridConfig(lat_step=20.0, lon_step=20.0, alt_step=100.0)
        prop_service = PropagationService(tx_lat=40.0, tx_lon=-105.0)

        success_count = 0
        error_count = 0

        for i in range(10):
            try:
                # Create Ne grid
                if i % 3 == 0:
                    # Intentionally create invalid data
                    lat_grid = grid_config.get_lat_grid()
                    lon_grid = grid_config.get_lon_grid()
                    alt_grid = grid_config.get_alt_grid()
                    ne_grid = np.full(
                        (len(lat_grid), len(lon_grid), len(alt_grid)),
                        -1e11  # Invalid negative Ne
                    )
                else:
                    lat_grid = grid_config.get_lat_grid()
                    lon_grid = grid_config.get_lon_grid()
                    alt_grid = grid_config.get_alt_grid()
                    ne_grid = np.full(
                        (len(lat_grid), len(lon_grid), len(alt_grid)),
                        1e11  # Valid
                    )

                prop_service.initialize_ray_tracer(
                    ne_grid=ne_grid,
                    lat_grid=grid_config.get_lat_grid(),
                    lon_grid=grid_config.get_lon_grid(),
                    alt_grid=grid_config.get_alt_grid(),
                    xray_flux=1e-6
                )

                result = prop_service.calculate_luf_muf()

                if result is not None:
                    success_count += 1

            except Exception as e:
                error_count += 1
                print(f"  Iteration {i} failed: {type(e).__name__}")

        print(f"\nSuccess: {success_count}, Errors: {error_count}")
        assert success_count > 0  # At least some should succeed


class TestFullSystemStress:
    """Ultimate stress test combining all components"""

    def test_everything_at_once(self):
        """Run everything simultaneously - the ultimate CPU melt test"""
        print("\n" + "="*70)
        print("ULTIMATE STRESS TEST - Running everything at maximum load")
        print("="*70)

        grid_config = GridConfig(lat_step=10.0, lon_step=10.0, alt_step=50.0)
        mq_client = MessageQueueClient()

        # Track what's happening
        stats = {
            'propagations_completed': 0,
            'messages_sent': 0,
            'messages_received': 0,
            'errors': 0
        }
        lock = threading.Lock()

        def callback(msg):
            with lock:
                stats['messages_received'] += 1

        test_topic = "test.ultimate"
        mq_client.subscribe(test_topic, callback)
        mq_client.subscribe(Topics.OUT_FREQUENCY_PLAN, callback)
        time.sleep(0.2)

        # Worker function that does everything
        def stress_worker(worker_id):
            try:
                # Create grid
                lat_grid = grid_config.get_lat_grid()
                lon_grid = grid_config.get_lon_grid()
                alt_grid = grid_config.get_alt_grid()
                ne_grid = np.full(
                    (len(lat_grid), len(lon_grid), len(alt_grid)),
                    1e11 * (1.0 + worker_id * 0.05)
                )

                # Propagation service
                prop_service = PropagationService(
                    tx_lat=40.0, tx_lon=-105.0,
                    freq_step=1.0
                )

                # Initialize and calculate multiple times
                for iteration in range(5):
                    prop_service.initialize_ray_tracer(
                        ne_grid=ne_grid,
                        lat_grid=grid_config.get_lat_grid(),
                        lon_grid=grid_config.get_lon_grid(),
                        alt_grid=grid_config.get_alt_grid(),
                        xray_flux=1e-6 * (1 + iteration * 0.1)
                    )

                    result = prop_service.calculate_luf_muf()

                    # Publish results
                    mq_client.publish(Topics.OUT_FREQUENCY_PLAN, result,
                                    source=f"worker-{worker_id}")

                    # Send test messages
                    for _ in range(10):
                        mq_client.publish(test_topic, {
                            "worker": worker_id,
                            "iteration": iteration,
                            "data": list(range(50))
                        }, source=f"worker-{worker_id}")

                    with lock:
                        stats['propagations_completed'] += 1
                        stats['messages_sent'] += 11  # 1 freq plan + 10 test msgs

            except Exception as e:
                with lock:
                    stats['errors'] += 1
                print(f"Worker {worker_id} error: {e}")

        # Launch many workers in parallel
        num_workers = 8
        print(f"\nLaunching {num_workers} workers, each doing 5 propagation cycles...")
        print("Expected:")
        print(f"  - {num_workers * 5} propagation calculations")
        print(f"  - {num_workers * 5 * 11} messages published")

        start = time.time()

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(stress_worker, i) for i in range(num_workers)]
            for f in futures:
                f.result()

        elapsed = time.time() - start
        time.sleep(2.0)  # Allow message delivery

        print(f"\n" + "="*70)
        print(f"STRESS TEST COMPLETED in {elapsed:.2f}s")
        print(f"="*70)
        print(f"Propagations completed: {stats['propagations_completed']}")
        print(f"Messages sent: {stats['messages_sent']}")
        print(f"Messages received: {stats['messages_received']}")
        print(f"Errors: {stats['errors']}")
        print(f"CPU time used: {elapsed:.2f}s")
        print(f"="*70)

        assert stats['propagations_completed'] >= 35  # Allow some failures
        assert stats['messages_sent'] > 0
        assert stats['messages_received'] > 0

        mq_client.close()


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-s'])
