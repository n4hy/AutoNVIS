"""
Performance Benchmarking Tests for NVIS System

Tests ingestion latency, quality assessment time, memory usage, and throughput.
Uses mock infrastructure from conftest.py - no RabbitMQ required.
"""

import pytest
import asyncio
import numpy as np
import time
import psutil
import os
from datetime import datetime
from typing import List
from unittest.mock import Mock, AsyncMock, patch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from conftest import MockMessageQueueClient

from src.ingestion.nvis.protocol_adapters.base_adapter import (
    NVISMeasurement, SounderMetadata
)
from src.ingestion.nvis.quality_assessor import QualityAssessor, QualityTier
from src.common.config import (
    AutoNVISConfig, NVISIngestionConfig, NVISQualityTierConfig,
    ServiceConfig, GridConfig
)


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


def create_test_sounder(sounder_id: str) -> SounderMetadata:
    """Create a test sounder metadata"""
    return SounderMetadata(
        sounder_id=sounder_id,
        name=f'Test Sounder {sounder_id}',
        operator='Test',
        location='Test Site',
        latitude=40.0,
        longitude=-105.0,
        altitude=1500.0,
        equipment_type='professional',
        calibration_status='calibrated'
    )


class TestQualityAssessmentLatency:
    """Test quality assessment latency"""

    def test_single_measurement_assessment_latency(self, test_config):
        """Measure latency for single measurement quality assessment"""
        assessor = QualityAssessor(test_config.nvis_ingestion)
        sounder = create_test_sounder('TEST_001')

        latencies = []

        # Measure latency for 100 measurements
        for _ in range(100):
            measurement = generate_test_measurement('TEST_001')

            start_time = time.perf_counter()
            metrics = assessor.assess_measurement(measurement, sounder)
            tier = assessor.assign_tier(metrics)
            errors = assessor.map_to_error_covariance(tier)
            end_time = time.perf_counter()

            latencies.append((end_time - start_time) * 1000)  # Convert to ms

        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)

        print(f"\nQuality Assessment Latency:")
        print(f"  Average: {avg_latency:.3f} ms")
        print(f"  P95: {p95_latency:.3f} ms")
        print(f"  P99: {p99_latency:.3f} ms")

        # Assert reasonable latency (< 10 ms)
        assert avg_latency < 10
        assert p99_latency < 50

    def test_batch_assessment_throughput(self, test_config):
        """Measure throughput for batch quality assessment"""
        assessor = QualityAssessor(test_config.nvis_ingestion)
        sounder = create_test_sounder('TEST_BATCH')

        # Generate batch of 1000 measurements
        measurements = [generate_test_measurement('TEST_BATCH')
                       for _ in range(1000)]

        start_time = time.perf_counter()

        for measurement in measurements:
            metrics = assessor.assess_measurement(measurement, sounder)
            tier = assessor.assign_tier(metrics)
            errors = assessor.map_to_error_covariance(tier)

        end_time = time.perf_counter()

        total_time = (end_time - start_time) * 1000  # ms
        throughput = len(measurements) / (total_time / 1000)  # obs/sec

        print(f"\nBatch Assessment (1000 measurements):")
        print(f"  Total Time: {total_time:.2f} ms")
        print(f"  Throughput: {throughput:.1f} obs/sec")

        # Assert reasonable throughput (should handle > 1000 obs/sec)
        assert throughput > 1000


class TestMessageQueueThroughput:
    """Test message queue throughput with mock infrastructure"""

    def test_publish_throughput(self, mock_mq_client):
        """Test message publishing throughput"""
        n_messages = 5000

        start_time = time.perf_counter()

        for i in range(n_messages):
            mock_mq_client.publish(
                "obs.nvis_sounder",
                {
                    "sounder_id": f"TEST_{i:04d}",
                    "signal_strength": -85.0 + np.random.normal(0, 2),
                    "quality_tier": "gold"
                },
                source="test"
            )

        end_time = time.perf_counter()

        elapsed = end_time - start_time
        throughput = n_messages / elapsed

        print(f"\nMock MQ Publish Throughput:")
        print(f"  Messages: {n_messages}")
        print(f"  Time: {elapsed:.3f} s")
        print(f"  Throughput: {throughput:.0f} msg/sec")

        # Mock should be very fast
        assert throughput > 5000

    def test_subscribe_delivery_throughput(self, mock_mq_client):
        """Test message delivery to subscribers"""
        received = []
        n_messages = 1000

        def callback(msg):
            received.append(msg)

        mock_mq_client.subscribe("obs.nvis_sounder", callback)
        time.sleep(0.05)

        start_time = time.perf_counter()

        for i in range(n_messages):
            mock_mq_client.publish(
                "obs.nvis_sounder",
                {"seq": i},
                source="test"
            )

        # Wait for delivery
        time.sleep(0.5)

        end_time = time.perf_counter()

        elapsed = end_time - start_time
        delivery_rate = len(received) / elapsed

        print(f"\nMock MQ Subscribe Delivery:")
        print(f"  Published: {n_messages}")
        print(f"  Received: {len(received)}")
        print(f"  Delivery Rate: {delivery_rate:.0f} msg/sec")

        # Most messages should be delivered
        assert len(received) >= n_messages * 0.95


class TestMemoryUsage:
    """Test memory usage and resource consumption"""

    def test_quality_assessor_memory(self, test_config):
        """Test memory usage of quality assessor with many sounders"""
        process = psutil.Process(os.getpid())

        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

        assessor = QualityAssessor(test_config.nvis_ingestion)

        # Assess measurements from 100 different sounders
        for i in range(100):
            sounder = create_test_sounder(f'MEM_TEST_{i:03d}')
            for _ in range(100):
                measurement = generate_test_measurement(sounder.sounder_id)
                metrics = assessor.assess_measurement(measurement, sounder)
                assessor.assign_tier(metrics)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - baseline_memory

        print(f"\nQuality Assessor Memory (10,000 assessments):")
        print(f"  Baseline: {baseline_memory:.1f} MB")
        print(f"  Final: {final_memory:.1f} MB")
        print(f"  Increase: {memory_increase:.1f} MB")

        # Should not consume excessive memory
        assert memory_increase < 100

    def test_mock_mq_memory(self, mock_mq_client):
        """Test memory usage of mock MQ with many messages"""
        process = psutil.Process(os.getpid())

        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Publish many messages
        for i in range(10000):
            mock_mq_client.publish(
                "test.topic",
                {"seq": i, "data": "x" * 100},
                source="test"
            )

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - baseline_memory

        print(f"\nMock MQ Memory (10,000 messages):")
        print(f"  Baseline: {baseline_memory:.1f} MB")
        print(f"  Final: {final_memory:.1f} MB")
        print(f"  Increase: {memory_increase:.1f} MB")

        # Should not consume excessive memory
        assert memory_increase < 50


class TestThroughput:
    """Test system throughput and rate handling"""

    def test_concurrent_quality_assessment(self, test_config):
        """Test handling multiple sounders concurrently"""
        assessor = QualityAssessor(test_config.nvis_ingestion)

        # Create 10 sounders
        sounders = [create_test_sounder(f'CONCURRENT_{i:02d}') for i in range(10)]

        start_time = time.perf_counter()

        # Process measurements from all sounders
        for sounder in sounders:
            for _ in range(100):
                measurement = generate_test_measurement(sounder.sounder_id)
                metrics = assessor.assess_measurement(measurement, sounder)
                assessor.assign_tier(metrics)

        end_time = time.perf_counter()

        total_time = (end_time - start_time) * 1000  # ms
        total_obs = 10 * 100  # 10 sounders × 100 obs
        throughput = total_obs / (total_time / 1000)  # obs/sec

        print(f"\nConcurrent Assessment (10 sounders, 100 obs each):")
        print(f"  Total Time: {total_time:.2f} ms")
        print(f"  Throughput: {throughput:.1f} obs/sec")

        # Should handle concurrent streams efficiently
        assert throughput > 1000


class TestFilterCycleBudget:
    """Test that processing fits within 15-minute cycle budget"""

    def test_realistic_cycle_processing_time(self, test_config, mock_mq_client):
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
        assessor = QualityAssessor(test_config.nvis_ingestion)

        # Create sounders
        sounders = []

        # 2 PLATINUM
        for i in range(2):
            sounders.append((create_test_sounder(f'PLAT_{i:02d}'), 125, QualityTier.PLATINUM))

        # 5 GOLD
        for i in range(5):
            sounders.append((create_test_sounder(f'GOLD_{i:02d}'), 12, QualityTier.GOLD))

        # 10 SILVER
        for i in range(10):
            sounders.append((create_test_sounder(f'SILV_{i:02d}'), 3, QualityTier.SILVER))

        # 20 BRONZE
        for i in range(20):
            sounders.append((create_test_sounder(f'BRON_{i:02d}'), 1 if i < 10 else 0, QualityTier.BRONZE))

        observations_published = []

        # Process all measurements
        start_time = time.perf_counter()

        for sounder, n_obs, tier in sounders:
            for _ in range(n_obs):
                measurement = generate_test_measurement(sounder.sounder_id)
                metrics = assessor.assess_measurement(measurement, sounder)
                assigned_tier = assessor.assign_tier(metrics)
                errors = assessor.map_to_error_covariance(assigned_tier)

                # Simulate publishing
                mock_mq_client.publish(
                    "obs.nvis_sounder",
                    {
                        "sounder_id": measurement.sounder_id,
                        "signal_strength": measurement.signal_strength,
                        "quality_tier": assigned_tier.value
                    },
                    source=f"nvis_{measurement.sounder_id}"
                )
                observations_published.append(measurement)

        end_time = time.perf_counter()

        processing_time = end_time - start_time

        print(f"\nRealistic 15-Minute Cycle:")
        print(f"  Raw observations: {sum(n for _, n, _ in sounders)}")
        print(f"  Published observations: {len(observations_published)}")
        print(f"  Processing time: {processing_time:.2f} seconds")
        print(f"  Percentage of 15-min budget: {(processing_time / 900) * 100:.2f}%")

        # Should complete well within 15-minute budget
        assert processing_time < 60  # Should complete in < 1 minute


class TestStability:
    """Test system stability under sustained load"""

    def test_sustained_assessment_stability(self, test_config):
        """Test stability over sustained operation (1 hour simulated)"""
        assessor = QualityAssessor(test_config.nvis_ingestion)
        sounder = create_test_sounder('SUSTAINED')

        # Simulate 1 hour at 100 obs/hour = 100 observations
        # Process in batches to simulate realistic timing

        errors = []
        latencies = []

        for batch in range(10):  # 10 batches of 10
            for _ in range(10):
                try:
                    measurement = generate_test_measurement('SUSTAINED')

                    start = time.perf_counter()
                    metrics = assessor.assess_measurement(measurement, sounder)
                    tier = assessor.assign_tier(metrics)
                    assessor.map_to_error_covariance(tier)
                    end = time.perf_counter()

                    latencies.append((end - start) * 1000)

                except Exception as e:
                    errors.append(str(e))

            # Small delay between batches
            time.sleep(0.01)

        print(f"\nSustained Load Test (100 assessments):")
        print(f"  Errors: {len(errors)}")
        print(f"  Average latency: {np.mean(latencies):.2f} ms")
        print(f"  Max latency: {np.max(latencies):.2f} ms")

        # Should have no errors and stable latency
        assert len(errors) == 0
        assert np.mean(latencies) < 10  # Average < 10ms
        assert np.max(latencies) < 50  # Max < 50ms


class TestMeasurementGenerationPerformance:
    """Test measurement generation performance (for test utilities)"""

    def test_measurement_generation_throughput(self):
        """Test how fast we can generate test measurements"""
        n_measurements = 10000

        start_time = time.perf_counter()

        for i in range(n_measurements):
            generate_test_measurement(f'PERF_{i:05d}')

        end_time = time.perf_counter()

        elapsed = end_time - start_time
        throughput = n_measurements / elapsed

        print(f"\nMeasurement Generation:")
        print(f"  Count: {n_measurements}")
        print(f"  Time: {elapsed:.3f} s")
        print(f"  Throughput: {throughput:.0f} measurements/sec")

        # Should be able to generate test data quickly
        assert throughput > 1000


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])  # -s to show print statements
