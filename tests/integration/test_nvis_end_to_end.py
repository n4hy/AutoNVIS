"""
End-to-End Integration Tests for NVIS System

Tests complete pipeline: Protocol adapters → Quality assessment → Aggregation
→ Message queue → Filter integration → Information gain analysis → Dashboard
"""

import pytest
import asyncio
import numpy as np
import time
from datetime import datetime
from typing import List, Dict
from unittest.mock import Mock, AsyncMock, patch

from src.ingestion.nvis.nvis_sounder_client import NVISSounderClient
from src.ingestion.nvis.quality_assessor import QualityTier
from src.ingestion.nvis.protocol_adapters.base_adapter import (
    NVISMeasurement, SounderMetadata
)
from src.analysis.information_gain_analyzer import InformationGainAnalyzer
from src.analysis.network_analyzer import NetworkAnalyzer
from src.common.message_queue import MessageQueueClient, Topics
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

    config = NVISIngestionConfig(
        adapters={},
        quality_tiers=tier_configs,
        aggregation_window_sec=60,
        aggregation_rate_threshold=60
    )

    return config


@pytest.fixture
def mock_mq_client():
    """Create mock message queue client"""
    mock_client = Mock(spec=MessageQueueClient)
    mock_client.publish = AsyncMock()
    return mock_client


@pytest.fixture
def multi_tier_sounders():
    """Create simulated multi-tier sounder network"""
    sounders = []

    # 2 PLATINUM sounders (professional research stations)
    for i in range(2):
        sounders.append(SounderMetadata(
            sounder_id=f'PLAT_{i+1:03d}',
            name=f'Professional Station {i+1}',
            operator='National Research Institute',
            location=f'Research Site {i+1}',
            latitude=35.0 + i * 5.0,
            longitude=-105.0 + i * 10.0,
            altitude=1500.0,
            equipment_type='professional',
            calibration_status='calibrated'
        ))

    # 5 GOLD sounders (university stations)
    for i in range(5):
        sounders.append(SounderMetadata(
            sounder_id=f'GOLD_{i+1:03d}',
            name=f'University Station {i+1}',
            operator='University Network',
            location=f'Campus {i+1}',
            latitude=40.0 + i * 3.0,
            longitude=-100.0 + i * 5.0,
            altitude=1200.0,
            equipment_type='research',
            calibration_status='calibrated'
        ))

    # 10 SILVER sounders (amateur club stations)
    for i in range(10):
        sounders.append(SounderMetadata(
            sounder_id=f'SILV_{i+1:03d}',
            name=f'Club Station {i+1}',
            operator='Ham Radio Club',
            location=f'Club Site {i+1}',
            latitude=42.0 + i * 2.0,
            longitude=-95.0 + i * 3.0,
            altitude=800.0,
            equipment_type='amateur_advanced',
            calibration_status='self_calibrated'
        ))

    # 20 BRONZE sounders (individual amateur stations)
    for i in range(20):
        sounders.append(SounderMetadata(
            sounder_id=f'BRON_{i+1:03d}',
            name=f'Amateur Station {i+1}',
            operator=f'Individual Operator {i+1}',
            location=f'Home QTH {i+1}',
            latitude=38.0 + i * 1.0,
            longitude=-90.0 + i * 2.0,
            altitude=500.0,
            equipment_type='amateur_basic',
            calibration_status='uncalibrated'
        ))

    return sounders


def generate_measurement(sounder: SounderMetadata, tier: QualityTier,
                        noise_level: float = 1.0) -> NVISMeasurement:
    """Generate synthetic NVIS measurement with realistic noise"""
    # Realistic NVIS parameters
    frequency = 7.5  # MHz (40m band)
    elevation = 80.0 + np.random.uniform(-5, 5)  # Near-vertical
    azimuth = np.random.uniform(0, 360)

    # Simple geometry: random receiver within 100 km
    rx_offset_lat = np.random.uniform(-0.5, 0.5)
    rx_offset_lon = np.random.uniform(-0.5, 0.5)
    hop_distance = np.sqrt(
        (rx_offset_lat * 111)**2 + (rx_offset_lon * 111)**2
    )

    # Realistic signal parameters
    base_signal = -85.0  # dBm
    base_delay = 2.5  # ms

    # Add tier-dependent noise
    if tier == QualityTier.PLATINUM:
        signal_noise = np.random.normal(0, 1.0) * noise_level
        delay_noise = np.random.normal(0, 0.05) * noise_level
        snr = 25.0 + np.random.uniform(-2, 2)
    elif tier == QualityTier.GOLD:
        signal_noise = np.random.normal(0, 2.0) * noise_level
        delay_noise = np.random.normal(0, 0.2) * noise_level
        snr = 20.0 + np.random.uniform(-3, 3)
    elif tier == QualityTier.SILVER:
        signal_noise = np.random.normal(0, 4.0) * noise_level
        delay_noise = np.random.normal(0, 1.0) * noise_level
        snr = 15.0 + np.random.uniform(-5, 5)
    else:  # BRONZE
        signal_noise = np.random.normal(0, 8.0) * noise_level
        delay_noise = np.random.normal(0, 2.5) * noise_level
        snr = 10.0 + np.random.uniform(-5, 5)

    return NVISMeasurement(
        tx_latitude=sounder.latitude,
        tx_longitude=sounder.longitude,
        tx_altitude=sounder.altitude,
        rx_latitude=sounder.latitude + rx_offset_lat,
        rx_longitude=sounder.longitude + rx_offset_lon,
        rx_altitude=sounder.altitude + np.random.uniform(-50, 50),
        frequency=frequency,
        elevation_angle=elevation,
        azimuth=azimuth,
        hop_distance=hop_distance,
        signal_strength=base_signal + signal_noise,
        group_delay=base_delay + delay_noise,
        snr=snr,
        signal_strength_error=0.0,  # Will be set by quality assessor
        group_delay_error=0.0,
        sounder_id=sounder.sounder_id,
        timestamp=datetime.utcnow().isoformat() + 'Z',
        is_o_mode=True
    )


class TestMultiTierNetwork:
    """Test simulated multi-tier sounder network"""

    @pytest.mark.asyncio
    async def test_full_network_ingestion(self, test_config, mock_mq_client,
                                          multi_tier_sounders):
        """Test ingestion from complete multi-tier network"""
        client = NVISSounderClient(test_config, mock_mq_client)

        # Register all sounders
        for sounder in multi_tier_sounders:
            client.register_sounder(sounder)

        assert len(client.sounder_registry) == 37  # 2+5+10+20

        # Generate realistic observation rates
        # PLATINUM: 500/hour = ~8/min
        # GOLD: 50/hour = ~1/min
        # SILVER: 10/hour = ~1/6min
        # BRONZE: 2/hour = ~1/30min

        observations_published = []

        async def capture_publish(topic, message):
            observations_published.append(message)

        mock_mq_client.publish.side_effect = capture_publish

        # Simulate 1 minute of data collection
        for sounder in multi_tier_sounders:
            if sounder.sounder_id.startswith('PLAT'):
                # 8 measurements
                tier = QualityTier.PLATINUM
                n_obs = 8
            elif sounder.sounder_id.startswith('GOLD'):
                # 1 measurement
                tier = QualityTier.GOLD
                n_obs = 1
            elif sounder.sounder_id.startswith('SILV'):
                # 0-1 measurement (probabilistic)
                tier = QualityTier.SILVER
                n_obs = 1 if np.random.random() < 0.17 else 0
            else:  # BRONZE
                # 0-1 measurement (probabilistic)
                tier = QualityTier.BRONZE
                n_obs = 1 if np.random.random() < 0.033 else 0

            for _ in range(n_obs):
                measurement = generate_measurement(sounder, tier)
                await client.process_measurement(measurement)

        # Verify observations were published
        assert len(observations_published) > 0

        # Verify quality assessment was applied
        for obs in observations_published:
            assert 'quality_tier' in obs
            assert obs['signal_strength_error'] > 0
            assert obs['group_delay_error'] > 0

        # Verify PLATINUM observations have lowest errors
        platinum_obs = [o for o in observations_published
                       if o['sounder_id'].startswith('PLAT')]
        bronze_obs = [o for o in observations_published
                     if o['sounder_id'].startswith('BRON')]

        if platinum_obs and bronze_obs:
            avg_plat_error = np.mean([o['signal_strength_error']
                                     for o in platinum_obs])
            avg_bronze_error = np.mean([o['signal_strength_error']
                                       for o in bronze_obs])
            assert avg_plat_error < avg_bronze_error

    @pytest.mark.asyncio
    async def test_rate_limiting_effectiveness(self, test_config, mock_mq_client,
                                               multi_tier_sounders):
        """Test that high-rate sounders are properly aggregated"""
        client = NVISSounderClient(test_config, mock_mq_client)

        # Register one PLATINUM sounder
        plat_sounder = multi_tier_sounders[0]
        client.register_sounder(plat_sounder)

        observations_published = []

        async def capture_publish(topic, message):
            observations_published.append(message)

        mock_mq_client.publish.side_effect = capture_publish

        # Generate 100 measurements in rapid succession (simulating high rate)
        for i in range(100):
            measurement = generate_measurement(plat_sounder, QualityTier.PLATINUM)
            await client.process_measurement(measurement)

        # With aggregation, should have buffered most observations
        # Only aggregated bins should be published
        assert len(observations_published) < 20  # Much less than 100

    @pytest.mark.asyncio
    async def test_quality_tier_distribution(self, test_config, mock_mq_client,
                                             multi_tier_sounders):
        """Test that quality tiers are correctly distributed"""
        client = NVISSounderClient(test_config, mock_mq_client)

        for sounder in multi_tier_sounders:
            client.register_sounder(sounder)

        observations_published = []

        async def capture_publish(topic, message):
            observations_published.append(message)

        mock_mq_client.publish.side_effect = capture_publish

        # Generate one measurement per sounder
        for sounder in multi_tier_sounders:
            if sounder.sounder_id.startswith('PLAT'):
                tier = QualityTier.PLATINUM
            elif sounder.sounder_id.startswith('GOLD'):
                tier = QualityTier.GOLD
            elif sounder.sounder_id.startswith('SILV'):
                tier = QualityTier.SILVER
            else:
                tier = QualityTier.BRONZE

            measurement = generate_measurement(sounder, tier)
            await client.process_measurement(measurement)

        # Count tier distribution
        tier_counts = {
            'platinum': sum(1 for o in observations_published
                          if o['quality_tier'] == 'platinum'),
            'gold': sum(1 for o in observations_published
                       if o['quality_tier'] == 'gold'),
            'silver': sum(1 for o in observations_published
                        if o['quality_tier'] == 'silver'),
            'bronze': sum(1 for o in observations_published
                        if o['quality_tier'] == 'bronze')
        }

        # Verify distribution matches expected
        assert tier_counts['platinum'] == 2
        assert tier_counts['gold'] == 5
        assert tier_counts['silver'] == 10
        assert tier_counts['bronze'] == 20


class TestQualityAdaptation:
    """Test adaptive quality learning with biased sounders"""

    @pytest.mark.asyncio
    async def test_biased_sounder_detection(self, test_config, mock_mq_client):
        """Test that biased sounder quality degrades over time"""
        client = NVISSounderClient(test_config, mock_mq_client)

        # Register a sounder that will be biased
        biased_sounder = SounderMetadata(
            sounder_id='BIASED_001',
            name='Biased Station',
            operator='Test',
            location='Test Site',
            latitude=40.0,
            longitude=-105.0,
            altitude=1500.0,
            equipment_type='professional',
            calibration_status='calibrated'
        )

        client.register_sounder(biased_sounder)

        # Initial quality should be high (professional + calibrated)
        initial_quality = client.quality_assessor.sounder_quality_history.get(
            'BIASED_001', 0.5
        )

        # Simulate 100 measurements with systematic +10 dB bias
        for i in range(100):
            measurement = generate_measurement(biased_sounder, QualityTier.PLATINUM)
            # Add systematic bias
            measurement.signal_strength += 10.0

            # Simulate innovation that reveals bias (high NIS)
            # In real system, this would come from filter
            predicted_std = 2.0  # Expected for PLATINUM
            innovation = 10.0  # Large due to bias

            client.quality_assessor.update_historical_quality(
                'BIASED_001', innovation, predicted_std
            )

        # Quality should have degraded
        final_quality = client.quality_assessor.sounder_quality_history['BIASED_001']

        assert final_quality < initial_quality
        assert final_quality < 0.3  # Should be significantly degraded

    @pytest.mark.asyncio
    async def test_good_sounder_quality_improvement(self, test_config, mock_mq_client):
        """Test that consistently good sounder quality improves"""
        client = NVISSounderClient(test_config, mock_mq_client)

        # Register amateur sounder (starts with low quality expectation)
        good_amateur = SounderMetadata(
            sounder_id='GOOD_AMATEUR_001',
            name='Good Amateur Station',
            operator='Careful Operator',
            location='Well-Maintained QTH',
            latitude=40.0,
            longitude=-105.0,
            altitude=1500.0,
            equipment_type='amateur_basic',
            calibration_status='uncalibrated'
        )

        client.register_sounder(good_amateur)

        # Initial quality should be low (amateur + uncalibrated)
        initial_quality = client.quality_assessor.sounder_quality_history.get(
            'GOOD_AMATEUR_001', 0.5
        )

        # Simulate 100 measurements with low innovation (good performance)
        for i in range(100):
            measurement = generate_measurement(good_amateur, QualityTier.BRONZE,
                                             noise_level=0.3)  # Low noise

            # Simulate low innovation (NIS << 1)
            predicted_std = 15.0  # Expected for BRONZE
            innovation = 2.0  # Much smaller than expected

            client.quality_assessor.update_historical_quality(
                'GOOD_AMATEUR_001', innovation, predicted_std
            )

        # Quality should have improved
        final_quality = client.quality_assessor.sounder_quality_history['GOOD_AMATEUR_001']

        assert final_quality > initial_quality
        assert final_quality > 0.7  # Should be significantly improved


class TestInformationGainAnalysis:
    """Test information gain computation accuracy"""

    def test_marginal_gain_computation(self, multi_tier_sounders):
        """Test that marginal gain correctly quantifies sounder contribution"""
        # Create test grid
        lat_grid = np.linspace(30, 50, 5)
        lon_grid = np.linspace(-120, -80, 5)
        alt_grid = np.linspace(100, 500, 5)

        # Create prior covariance (diagonal for simplicity)
        n_state = len(lat_grid) * len(lon_grid) * len(alt_grid)
        prior_sqrt_cov = np.eye(n_state) * 0.3  # Prior std = 0.3

        analyzer = InformationGainAnalyzer(lat_grid, lon_grid, alt_grid)

        # Generate observations from multiple sounders
        all_observations = []
        for sounder in multi_tier_sounders[:5]:  # Use first 5 sounders
            if sounder.sounder_id.startswith('PLAT'):
                tier = QualityTier.PLATINUM
            elif sounder.sounder_id.startswith('GOLD'):
                tier = QualityTier.GOLD
            else:
                tier = QualityTier.SILVER

            # Generate 5 measurements per sounder
            for _ in range(5):
                measurement = generate_measurement(sounder, tier)

                # Convert to observation dict
                obs_dict = {
                    'sounder_id': measurement.sounder_id,
                    'tx_latitude': measurement.tx_latitude,
                    'tx_longitude': measurement.tx_longitude,
                    'signal_strength': measurement.signal_strength,
                    'group_delay': measurement.group_delay,
                    'signal_strength_error': 2.0 if tier == QualityTier.PLATINUM else 4.0,
                    'group_delay_error': 0.1 if tier == QualityTier.PLATINUM else 0.5,
                }
                all_observations.append(obs_dict)

        # Compute marginal gain for first sounder
        result = analyzer.compute_marginal_gain(
            multi_tier_sounders[0].sounder_id,
            all_observations,
            prior_sqrt_cov
        )

        # Verify result structure
        assert result.marginal_gain >= 0
        assert 0 <= result.relative_contribution <= 1.0
        assert result.trace_with < result.trace_without
        assert result.n_observations > 0

    def test_platinum_higher_gain_than_bronze(self, multi_tier_sounders):
        """Test that PLATINUM sounders have higher marginal gain than BRONZE"""
        lat_grid = np.linspace(30, 50, 5)
        lon_grid = np.linspace(-120, -80, 5)
        alt_grid = np.linspace(100, 500, 5)

        n_state = len(lat_grid) * len(lon_grid) * len(alt_grid)
        prior_sqrt_cov = np.eye(n_state) * 0.3

        analyzer = InformationGainAnalyzer(lat_grid, lon_grid, alt_grid)

        # Create observations from PLATINUM and BRONZE sounders
        plat_sounder = [s for s in multi_tier_sounders if s.sounder_id.startswith('PLAT')][0]
        bronze_sounder = [s for s in multi_tier_sounders if s.sounder_id.startswith('BRON')][0]

        observations = []

        # 10 PLATINUM observations
        for _ in range(10):
            meas = generate_measurement(plat_sounder, QualityTier.PLATINUM)
            observations.append({
                'sounder_id': meas.sounder_id,
                'tx_latitude': meas.tx_latitude,
                'tx_longitude': meas.tx_longitude,
                'signal_strength': meas.signal_strength,
                'group_delay': meas.group_delay,
                'signal_strength_error': 2.0,
                'group_delay_error': 0.1,
            })

        # 10 BRONZE observations
        for _ in range(10):
            meas = generate_measurement(bronze_sounder, QualityTier.BRONZE)
            observations.append({
                'sounder_id': meas.sounder_id,
                'tx_latitude': meas.tx_latitude,
                'tx_longitude': meas.tx_longitude,
                'signal_strength': meas.signal_strength,
                'group_delay': meas.group_delay,
                'signal_strength_error': 15.0,
                'group_delay_error': 5.0,
            })

        plat_gain = analyzer.compute_marginal_gain(
            plat_sounder.sounder_id, observations, prior_sqrt_cov
        )

        bronze_gain = analyzer.compute_marginal_gain(
            bronze_sounder.sounder_id, observations, prior_sqrt_cov
        )

        # PLATINUM should contribute more information
        assert plat_gain.marginal_gain > bronze_gain.marginal_gain


class TestNetworkAnalysis:
    """Test comprehensive network analysis"""

    def test_complete_network_analysis(self, multi_tier_sounders):
        """Test full network analysis with all components"""
        lat_grid = np.linspace(30, 50, 5)
        lon_grid = np.linspace(-120, -80, 5)
        alt_grid = np.linspace(100, 500, 5)

        n_state = len(lat_grid) * len(lon_grid) * len(alt_grid)
        prior_sqrt_cov = np.eye(n_state) * 0.3

        analyzer = NetworkAnalyzer(lat_grid, lon_grid, alt_grid)

        # Generate observations from all sounders
        all_observations = []
        for sounder in multi_tier_sounders:
            if sounder.sounder_id.startswith('PLAT'):
                tier = QualityTier.PLATINUM
                error_signal, error_delay = 2.0, 0.1
                n_obs = 10
            elif sounder.sounder_id.startswith('GOLD'):
                tier = QualityTier.GOLD
                error_signal, error_delay = 4.0, 0.5
                n_obs = 5
            elif sounder.sounder_id.startswith('SILV'):
                tier = QualityTier.SILVER
                error_signal, error_delay = 8.0, 2.0
                n_obs = 2
            else:
                tier = QualityTier.BRONZE
                error_signal, error_delay = 15.0, 5.0
                n_obs = 1

            for _ in range(n_obs):
                meas = generate_measurement(sounder, tier)
                all_observations.append({
                    'sounder_id': meas.sounder_id,
                    'tx_latitude': meas.tx_latitude,
                    'tx_longitude': meas.tx_longitude,
                    'signal_strength': meas.signal_strength,
                    'group_delay': meas.group_delay,
                    'signal_strength_error': error_signal,
                    'group_delay_error': error_delay,
                    'quality_tier': tier.value
                })

        # Run full analysis
        analysis = analyzer.analyze_network(
            multi_tier_sounders,
            all_observations,
            prior_sqrt_cov
        )

        # Verify analysis structure
        assert 'network_overview' in analysis
        assert 'information_gain' in analysis
        assert 'coverage_analysis' in analysis
        assert 'quality_analysis' in analysis
        assert 'recommendations' in analysis

        # Verify network overview
        overview = analysis['network_overview']
        assert overview['n_sounders'] == 37
        assert overview['n_observations'] == len(all_observations)
        assert 'quality_tier_distribution' in overview

        # Verify information gain section
        info_gain = analysis['information_gain']
        assert 'total_information_gain' in info_gain
        assert 'top_contributors' in info_gain
        assert len(info_gain['top_contributors']) > 0

        # Verify recommendations
        recommendations = analysis['recommendations']
        assert 'new_sounders' in recommendations
        assert 'upgrades' in recommendations


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
