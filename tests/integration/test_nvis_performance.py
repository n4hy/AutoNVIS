"""
Performance Benchmarking Tests for NVIS System

Tests ingestion latency, filter update time, memory usage, and throughput.
"""

import pytest
import asyncio
import numpy as np
import time
import psutil
import os
from datetime import datetime
from typing import List
from unittest.mock import Mock, AsyncMock

from src.ingestion.nvis.nvis_sounder_client import NVISSounderClient
from src.ingestion.nvis.quality_assessor import QualityTier
from src.ingestion.nvis.protocol_adapters.base_adapter import (
    NVISMeasurement, SounderMetadata
)
from src.common.message_queue import MessageQueueClient
from src.common.config import NVISIngestionConfig, QualityTierConfig


@pytest.fixture
def test_config():
    """Create test configuration"""
    tier_configs = {
        'platinum': QualityTierConfig(
            signal_error_db=2.0,
            delay_error_ms=0.1
        ),
        'gold': QualityTierConfig(
            signal_error_db=4.0,
            delay_error_ms=0.5
        ),
        'silver': QualityTierConfig(
            signal_error_db=8.0,
            delay_error_ms=2.0
        ),
        'bronze': QualityTierConfig(
            signal_error_db=15.0,
            delay_error_ms=5.0
        )
    }

    return NVISIngestionConfig(
        adapters={},
        quality_tiers=tier_configs,
        aggregation_window_sec=60,
        aggregation_rate_threshold=60
    )


@pytest.fixture
def mock_mq_client():
    """Create mock message queue client"""
    mock_client = Mock(spec=MessageQueueClient)
    mock_client.publish = AsyncMock()
    return mock_client


def generate_test_measurement(sounder_id: str) -> NVISMeasurement:
    """Generate test measurement"""
    return NVISMeasurement(
        tx_latitude=40.0,
        tx_longitude=-105.0,
        tx_altitude=1500.0,
        rx_latitude=40.5,
        rx_longitude=-104.5,
        rx_altitude=1600.0,
        frequency=7.5,
        elevation_angle=85.0,
        azimuth=45.0,
        hop_distance=75.0,
        signal_strength=-85.0 + np.random.normal(0, 2),
        group_delay=2.5 + np.random.normal(0, 0.1),
        snr=20.0,
        signal_strength_error=0.0,
        group_delay_error=0.0,
        sounder_id=sounder_id,
        timestamp=datetime.utcnow().isoformat() + 'Z',
        is_o_mode=True
    )


class TestIngestionLatency:
    """Test ingestion pipeline latency"""

    @pytest.mark.asyncio
    async def test_single_measurement_latency(self, test_config, mock_mq_client):
        """Measure latency for single measurement processing"""
        client = NVISSounderClient(test_config, mock_mq_client)

        sounder = SounderMetadata(
            sounder_id='TEST_001',
            name='Test Sounder',
            operator='Test',
            location='Test Site',
            latitude=40.0,
            longitude=-105.0,
            altitude=1500.0,
            equipment_type='professional',
            calibration_status='calibrated'
        )

        client.register_sounder(sounder)

        latencies = []

        # Measure latency for 100 measurements
        for _ in range(100):
            measurement = generate_test_measurement('TEST_001')

            start_time = time.perf_counter()
            await client.process_measurement(measurement)
            end_time = time.perf_counter()

            latencies.append((end_time - start_time) * 1000)  # Convert to ms

        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)

        print(f"\nIngestion Latency:")
        print(f"  Average: {avg_latency:.2f} ms")
        print(f"  P95: {p95_latency:.2f} ms")
        print(f"  P99: {p99_latency:.2f} ms")

        # Assert reasonable latency (< 60 seconds target)
        assert avg_latency < 1000  # Less than 1 second average
        assert p99_latency < 2000  # Less than 2 seconds p99

    @pytest.mark.asyncio
    async def test_batch_processing_latency(self, test_config, mock_mq_client):
        """Measure latency for batch processing"""
        client = NVISSounderClient(test_config, mock_mq_client)

        sounder = SounderMetadata(
            sounder_id='TEST_BATCH',
            name='Batch Test Sounder',
            operator='Test',
            location='Test Site',
            latitude=40.0,
            longitude=-105.0,
            altitude=1500.0,
            equipment_type='professional',
            calibration_status='calibrated'
        )

        client.register_sounder(sounder)

        # Process batch of 100 measurements
        measurements = [generate_test_measurement('TEST_BATCH')
                       for _ in range(100)]

        start_time = time.perf_counter()

        for measurement in measurements:
            await client.process_measurement(measurement)

        end_time = time.perf_counter()

        total_time = (end_time - start_time) * 1000  # ms
        throughput = len(measurements) / (total_time / 1000)  # obs/sec

        print(f"\nBatch Processing (100 measurements):")
        print(f"  Total Time: {total_time:.2f} ms")
        print(f"  Throughput: {throughput:.1f} obs/sec")

        # Assert reasonable throughput (should handle > 100 obs/sec)
        assert throughput > 100


class TestMemoryUsage:
    """Test memory usage and resource consumption"""

    @pytest.mark.asyncio
    async def test_memory_usage_with_thousands_of_observations(self,
                                                                test_config,
                                                                mock_mq_client):
        """Monitor memory usage with large number of observations"""
        process = psutil.Process(os.getpid())

        # Get baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

        client = NVISSounderClient(test_config, mock_mq_client)

        # Register 100 sounders
        sounders = []
        for i in range(100):
            sounder = SounderMetadata(
                sounder_id=f'MEM_TEST_{i:03d}',
                name=f'Memory Test Sounder {i}',
                operator='Test',
                location=f'Site {i}',
                latitude=40.0 + i * 0.1,
                longitude=-105.0 + i * 0.1,
                altitude=1500.0,
                equipment_type='professional',
                calibration_status='calibrated'
            )
            client.register_sounder(sounder)
            sounders.append(sounder)

        # Process 10,000 measurements (100 per sounder)
        for sounder in sounders:
            for _ in range(100):
                measurement = generate_test_measurement(sounder.sounder_id)
                await client.process_measurement(measurement)

        # Get final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - baseline_memory

        print(f"\nMemory Usage (10,000 observations):")
        print(f"  Baseline: {baseline_memory:.1f} MB")
        print(f"  Final: {final_memory:.1f} MB")
        print(f"  Increase: {memory_increase:.1f} MB")

        # Assert reasonable memory usage (< 500 MB increase)
        assert memory_increase < 500

    @pytest.mark.asyncio
    async def test_aggregation_buffer_memory_bounded(self, test_config,
                                                     mock_mq_client):
        """Verify that aggregation buffers don't grow unbounded"""
        process = psutil.Process(os.getpid())

        baseline_memory = process.memory_info().rss / 1024 / 1024

        client = NVISSounderClient(test_config, mock_mq_client)

        sounder = SounderMetadata(
            sounder_id='BUFFER_TEST',
            name='Buffer Test',
            operator='Test',
            location='Test Site',
            latitude=40.0,
            longitude=-105.0,
            altitude=1500.0,
            equipment_type='professional',
            calibration_status='calibrated'
        )

        client.register_sounder(sounder)

        # Generate high-rate stream (should trigger aggregation)
        for i in range(10000):
            measurement = generate_test_measurement('BUFFER_TEST')
            await client.process_measurement(measurement)

            # Periodically check buffer size
            if i % 1000 == 0:
                buffer_size = len(client.aggregator.buffers.get('BUFFER_TEST', []))
                # Buffer should not grow unbounded (aggregation should flush)
                assert buffer_size < 1000

        final_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = final_memory - baseline_memory

        print(f"\nBuffer Memory Test (10,000 high-rate observations):")
        print(f"  Memory Increase: {memory_increase:.1f} MB")

        # Should not consume excessive memory due to buffering
        assert memory_increase < 200


class TestThroughput:
    """Test system throughput and rate handling"""

    @pytest.mark.asyncio
    async def test_high_rate_sounder_throughput(self, test_config, mock_mq_client):
        """Test handling of 1000 obs/hour sounder"""
        client = NVISSounderClient(test_config, mock_mq_client)

        sounder = SounderMetadata(
            sounder_id='HIGH_RATE',
            name='High Rate Sounder',
            operator='Test',
            location='Test Site',
            latitude=40.0,
            longitude=-105.0,
            altitude=1500.0,
            equipment_type='professional',
            calibration_status='calibrated'
        )

        client.register_sounder(sounder)

        observations_published = []

        async def capture_publish(topic, message):
            observations_published.append(message)

        mock_mq_client.publish.side_effect = capture_publish

        # Simulate 1000 obs/hour for 1 minute (~ 17 observations)
        n_measurements = 17

        start_time = time.perf_counter()

        for _ in range(n_measurements):
            measurement = generate_test_measurement('HIGH_RATE')
            await client.process_measurement(measurement)

        end_time = time.perf_counter()

        processing_time = (end_time - start_time) * 1000  # ms

        print(f"\nHigh-Rate Throughput (17 obs in 1 min equivalent):")
        print(f"  Processing Time: {processing_time:.2f} ms")
        print(f"  Published: {len(observations_published)} observations")

        # Should complete quickly
        assert processing_time < 1000  # Less than 1 second

    @pytest.mark.asyncio
    async def test_concurrent_sounder_handling(self, test_config, mock_mq_client):
        """Test handling multiple sounders concurrently"""
        client = NVISSounderClient(test_config, mock_mq_client)

        # Register 10 sounders
        sounders = []
        for i in range(10):
            sounder = SounderMetadata(
                sounder_id=f'CONCURRENT_{i:02d}',
                name=f'Concurrent Sounder {i}',
                operator='Test',
                location=f'Site {i}',
                latitude=40.0 + i,
                longitude=-105.0 + i,
                altitude=1500.0,
                equipment_type='professional',
                calibration_status='calibrated'
            )
            client.register_sounder(sounder)
            sounders.append(sounder)

        # Process measurements from all sounders concurrently
        async def process_sounder_stream(sounder_id, n_measurements):
            for _ in range(n_measurements):
                measurement = generate_test_measurement(sounder_id)
                await client.process_measurement(measurement)

        start_time = time.perf_counter()

        # Run all sounder streams concurrently
        await asyncio.gather(*[
            process_sounder_stream(s.sounder_id, 10)
            for s in sounders
        ])

        end_time = time.perf_counter()

        total_time = (end_time - start_time) * 1000  # ms
        total_obs = 10 * 10  # 10 sounders × 10 obs
        throughput = total_obs / (total_time / 1000)  # obs/sec

        print(f"\nConcurrent Handling (10 sounders, 10 obs each):")
        print(f"  Total Time: {total_time:.2f} ms")
        print(f"  Throughput: {throughput:.1f} obs/sec")

        # Should handle concurrent streams efficiently
        assert throughput > 50  # At least 50 obs/sec with concurrency


class TestFilterCycleBudget:
    """Test that processing fits within 15-minute cycle budget"""

    @pytest.mark.asyncio
    async def test_realistic_cycle_processing_time(self, test_config,
                                                    mock_mq_client):
        """
        Test processing time for realistic 15-minute cycle workload

        Realistic workload:
        - 2 PLATINUM sounders: 500/hr × 15min = 125 obs each = 250 total
        - 5 GOLD sounders: 50/hr × 15min = 12 obs each = 60 total
        - 10 SILVER sounders: 10/hr × 15min = 2-3 obs each = 25 total
        - 20 BRONZE sounders: 2/hr × 15min = 0-1 obs each = 10 total

        Total: ~345 raw observations
        After aggregation: ~150 observations to filter
        """
        client = NVISSounderClient(test_config, mock_mq_client)

        # Register sounders
        sounders = []

        # 2 PLATINUM
        for i in range(2):
            s = SounderMetadata(
                sounder_id=f'PLAT_{i:02d}',
                name=f'Platinum {i}',
                operator='Research',
                location=f'Site {i}',
                latitude=35.0 + i * 5,
                longitude=-105.0 + i * 10,
                altitude=1500.0,
                equipment_type='professional',
                calibration_status='calibrated'
            )
            client.register_sounder(s)
            sounders.append((s, 125, QualityTier.PLATINUM))

        # 5 GOLD
        for i in range(5):
            s = SounderMetadata(
                sounder_id=f'GOLD_{i:02d}',
                name=f'Gold {i}',
                operator='University',
                location=f'Campus {i}',
                latitude=40.0 + i * 3,
                longitude=-100.0 + i * 5,
                altitude=1200.0,
                equipment_type='research',
                calibration_status='calibrated'
            )
            client.register_sounder(s)
            sounders.append((s, 12, QualityTier.GOLD))

        # 10 SILVER
        for i in range(10):
            s = SounderMetadata(
                sounder_id=f'SILV_{i:02d}',
                name=f'Silver {i}',
                operator='Club',
                location=f'Club {i}',
                latitude=42.0 + i * 2,
                longitude=-95.0 + i * 3,
                altitude=800.0,
                equipment_type='amateur_advanced',
                calibration_status='self_calibrated'
            )
            client.register_sounder(s)
            sounders.append((s, 3, QualityTier.SILVER))

        # 20 BRONZE
        for i in range(20):
            s = SounderMetadata(
                sounder_id=f'BRON_{i:02d}',
                name=f'Bronze {i}',
                operator=f'Amateur {i}',
                location=f'QTH {i}',
                latitude=38.0 + i,
                longitude=-90.0 + i * 2,
                altitude=500.0,
                equipment_type='amateur_basic',
                calibration_status='uncalibrated'
            )
            client.register_sounder(s)
            sounders.append((s, 1 if i < 10 else 0, QualityTier.BRONZE))

        observations_published = []

        async def capture_publish(topic, message):
            observations_published.append(message)

        mock_mq_client.publish.side_effect = capture_publish

        # Process all measurements
        start_time = time.perf_counter()

        for sounder, n_obs, tier in sounders:
            for _ in range(n_obs):
                measurement = generate_test_measurement(sounder.sounder_id)
                await client.process_measurement(measurement)

        end_time = time.perf_counter()

        processing_time = end_time - start_time

        print(f"\nRealistic 15-Minute Cycle:")
        print(f"  Raw observations: {sum(n for _, n, _ in sounders)}")
        print(f"  Published observations: {len(observations_published)}")
        print(f"  Processing time: {processing_time:.2f} seconds")
        print(f"  Percentage of 15-min budget: {(processing_time / 900) * 100:.2f}%")

        # Should complete well within 15-minute budget
        assert processing_time < 60  # Should complete in < 1 minute
        assert len(observations_published) < 200  # Aggregation should reduce count


class TestStability:
    """Test system stability under sustained load"""

    @pytest.mark.asyncio
    async def test_sustained_load_stability(self, test_config, mock_mq_client):
        """Test stability over sustained operation (1 hour simulated)"""
        client = NVISSounderClient(test_config, mock_mq_client)

        sounder = SounderMetadata(
            sounder_id='SUSTAINED',
            name='Sustained Test',
            operator='Test',
            location='Test Site',
            latitude=40.0,
            longitude=-105.0,
            altitude=1500.0,
            equipment_type='professional',
            calibration_status='calibrated'
        )

        client.register_sounder(sounder)

        # Simulate 1 hour at 100 obs/hour = 100 observations
        # Process in batches to simulate realistic timing

        errors = []
        latencies = []

        for batch in range(10):  # 10 batches of 10
            for _ in range(10):
                try:
                    measurement = generate_test_measurement('SUSTAINED')

                    start = time.perf_counter()
                    await client.process_measurement(measurement)
                    end = time.perf_counter()

                    latencies.append((end - start) * 1000)

                except Exception as e:
                    errors.append(str(e))

            # Small delay between batches
            await asyncio.sleep(0.01)

        print(f"\nSustained Load Test (100 observations):")
        print(f"  Errors: {len(errors)}")
        print(f"  Average latency: {np.mean(latencies):.2f} ms")
        print(f"  Max latency: {np.max(latencies):.2f} ms")

        # Should have no errors and stable latency
        assert len(errors) == 0
        assert np.mean(latencies) < 100  # Average < 100ms
        assert np.max(latencies) < 500  # Max < 500ms


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])  # -s to show print statements
