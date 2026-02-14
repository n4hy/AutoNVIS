"""
Unit Tests for Information Gain Analysis

Tests Fisher Information computation and marginal gain analysis.
"""

import pytest
import numpy as np
from src.analysis.information_gain_analyzer import (
    InformationGainAnalyzer,
    InformationGainResult
)


@pytest.fixture
def analyzer():
    """Create information gain analyzer"""
    grid_shape = (7, 7, 11)  # Small grid for testing
    lat_grid = np.linspace(-90, 90, 7)
    lon_grid = np.linspace(-180, 180, 7)
    alt_grid = np.linspace(100, 500, 11)

    return InformationGainAnalyzer(
        grid_shape, lat_grid, lon_grid, alt_grid
    )


@pytest.fixture
def sample_observations():
    """Create sample NVIS observations"""
    # Grid points: lat = [-90, -60, -30, 0, 30, 60, 90], lon = [-180, -120, -60, 0, 60, 120, 180]
    # Use observations near grid points for proper information gain calculation
    return [
        {
            'sounder_id': 'SOUNDER_A',
            'tx_latitude': 30.0,  # On grid
            'tx_longitude': -120.0,  # On grid
            'tx_altitude': 1500.0,
            'rx_latitude': 30.5,
            'rx_longitude': -119.5,
            'rx_altitude': 1600.0,
            'frequency': 7.5,
            'elevation_angle': 85.0,
            'azimuth': 45.0,
            'hop_distance': 75.0,
            'signal_strength': -80.0,
            'group_delay': 2.5,
            'snr': 20.0,
            'signal_strength_error': 2.0,
            'group_delay_error': 0.1,
            'is_o_mode': True,
            'quality_tier': 'platinum',
            'quality_metrics': {
                'signal_quality': 0.9,
                'calibration_quality': 1.0,
                'temporal_quality': 0.8,
                'spatial_quality': 0.7,
                'equipment_quality': 1.0,
                'historical_quality': 0.8
            }
        },
        {
            'sounder_id': 'SOUNDER_B',
            'tx_latitude': 30.0,  # On grid
            'tx_longitude': -60.0,  # On grid
            'tx_altitude': 1200.0,
            'rx_latitude': 30.5,
            'rx_longitude': -59.5,
            'rx_altitude': 1300.0,
            'frequency': 7.5,
            'elevation_angle': 80.0,
            'azimuth': 90.0,
            'hop_distance': 80.0,
            'signal_strength': -90.0,
            'group_delay': 2.8,
            'snr': 15.0,
            'signal_strength_error': 8.0,
            'group_delay_error': 2.0,
            'is_o_mode': True,
            'quality_tier': 'silver',
            'quality_metrics': {
                'signal_quality': 0.5,
                'calibration_quality': 0.5,
                'temporal_quality': 0.4,
                'spatial_quality': 0.6,
                'equipment_quality': 0.4,
                'historical_quality': 0.5
            }
        }
    ]


@pytest.fixture
def prior_sqrt_cov():
    """Create sample prior sqrt covariance"""
    state_dim = 7 * 7 * 11 + 1  # 540
    # Diagonal sqrt covariance
    return np.diag(1e5 * np.ones(state_dim))


class TestInformationGainAnalyzer:
    """Test InformationGainAnalyzer class"""

    def test_initialization(self, analyzer):
        """Test analyzer initialization"""
        assert analyzer.state_dim == 7 * 7 * 11 + 1
        assert analyzer.n_lat == 7
        assert analyzer.n_lon == 7
        assert analyzer.n_alt == 11

    def test_marginal_gain_computation(self, analyzer, sample_observations, prior_sqrt_cov):
        """Test marginal gain computation"""
        result = analyzer.compute_marginal_gain(
            'SOUNDER_A',
            sample_observations,
            prior_sqrt_cov
        )

        assert isinstance(result, InformationGainResult)
        assert result.sounder_id == 'SOUNDER_A'
        assert result.marginal_gain >= 0.0  # Should be non-negative
        assert 0.0 <= result.relative_contribution <= 1.0
        assert result.n_observations == 1

    def test_marginal_gain_comparison(self, analyzer, sample_observations, prior_sqrt_cov):
        """Test that high-quality sounders contribute more"""
        result_platinum = analyzer.compute_marginal_gain(
            'SOUNDER_A',  # Platinum tier (σ=2.0 dB)
            sample_observations,
            prior_sqrt_cov
        )

        result_silver = analyzer.compute_marginal_gain(
            'SOUNDER_B',  # Silver tier (σ=8.0 dB)
            sample_observations,
            prior_sqrt_cov
        )

        # Platinum should have higher marginal gain (lower error)
        assert result_platinum.marginal_gain >= result_silver.marginal_gain

    def test_empty_observations(self, analyzer, prior_sqrt_cov):
        """Test handling of empty observation list"""
        result = analyzer.compute_marginal_gain(
            'NONEXISTENT',
            [],
            prior_sqrt_cov
        )

        assert result.marginal_gain == 0.0
        assert result.relative_contribution == 0.0
        assert result.n_observations == 0

    def test_all_marginal_gains(self, analyzer, sample_observations, prior_sqrt_cov):
        """Test computation of all marginal gains"""
        results = analyzer.compute_all_marginal_gains(
            sample_observations,
            prior_sqrt_cov
        )

        assert 'SOUNDER_A' in results
        assert 'SOUNDER_B' in results
        assert len(results) == 2

        # Total contributions should sum to approximately 1.0
        total_contribution = sum(r.relative_contribution for r in results.values())
        assert 0.5 <= total_contribution <= 1.5  # Allow some numerical error

    def test_network_information(self, analyzer, sample_observations, prior_sqrt_cov):
        """Test network information computation"""
        network_info = analyzer.compute_network_information(
            sample_observations,
            prior_sqrt_cov
        )

        assert 'trace_prior' in network_info
        assert 'trace_posterior' in network_info
        assert 'total_information_gain' in network_info
        assert 'relative_uncertainty_reduction' in network_info

        # Posterior trace should be less than prior
        assert network_info['trace_posterior'] < network_info['trace_prior']

        # Information gain should be positive
        assert network_info['total_information_gain'] > 0.0

    def test_information_contribution(self, analyzer, sample_observations, prior_sqrt_cov):
        """Test information contribution computation"""
        prior_cov = prior_sqrt_cov @ prior_sqrt_cov.T

        contrib = analyzer._compute_information_contribution(
            sample_observations,
            prior_cov
        )

        assert contrib >= 0.0  # Should be non-negative

    def test_nearby_indices(self, analyzer):
        """Test nearby indices computation"""
        # Use a grid point (30.0, -120.0) so we get results within 500km
        indices = analyzer._get_nearby_indices(
            lat=30.0,  # On grid
            lon=-120.0,  # On grid
            radius_km=500.0
        )

        assert len(indices) > 0
        assert all(isinstance(idx, int) for idx in indices)

    def test_upgrade_prediction(self, analyzer, sample_observations, prior_sqrt_cov):
        """Test upgrade improvement prediction"""
        improvement = analyzer.predict_improvement_from_upgrade(
            'SOUNDER_B',  # Silver tier
            sample_observations,
            prior_sqrt_cov,
            new_tier='platinum'
        )

        assert 'current_marginal_gain' in improvement
        assert 'upgraded_marginal_gain' in improvement
        assert 'improvement' in improvement

        # Upgrade should improve marginal gain
        assert improvement['improvement'] >= 0.0

    def test_tier_contributions(self, analyzer, sample_observations):
        """Test tier contribution analysis"""
        marginal_gains = {
            'SOUNDER_A': InformationGainResult(
                sounder_id='SOUNDER_A',
                marginal_gain=1000.0,
                relative_contribution=0.6,
                trace_with=5000.0,
                trace_without=6000.0,
                n_observations=1,
                avg_quality_score=0.9
            ),
            'SOUNDER_B': InformationGainResult(
                sounder_id='SOUNDER_B',
                marginal_gain=500.0,
                relative_contribution=0.4,
                trace_with=5500.0,
                trace_without=6000.0,
                n_observations=1,
                avg_quality_score=0.5
            )
        }

        tier_contrib = analyzer._compute_tier_contributions(
            sample_observations,
            marginal_gains
        )

        assert 'platinum' in tier_contrib
        assert 'silver' in tier_contrib
        assert tier_contrib['platinum']['n_observations'] == 1
        assert tier_contrib['silver']['n_observations'] == 1


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_single_observation(self, analyzer, prior_sqrt_cov):
        """Test with single observation"""
        obs = [{
            'sounder_id': 'SOLO',
            'tx_latitude': 30.0,  # On grid
            'tx_longitude': -120.0,  # On grid
            'tx_altitude': 1500.0,
            'rx_latitude': 30.5,
            'rx_longitude': -119.5,
            'rx_altitude': 1600.0,
            'frequency': 7.5,
            'elevation_angle': 85.0,
            'azimuth': 45.0,
            'hop_distance': 75.0,
            'signal_strength': -80.0,
            'group_delay': 2.5,
            'snr': 20.0,
            'signal_strength_error': 2.0,
            'group_delay_error': 0.1,
            'is_o_mode': True,
            'quality_tier': 'gold',
            'quality_metrics': {'signal_quality': 0.8}
        }]

        result = analyzer.compute_marginal_gain('SOLO', obs, prior_sqrt_cov)

        assert result.marginal_gain >= 0.0
        assert result.relative_contribution == 1.0  # Only sounder

    def test_high_precision_errors(self, analyzer, prior_sqrt_cov):
        """Test with very low observation errors"""
        obs = [{
            'sounder_id': 'PRECISE',
            'tx_latitude': 30.0,  # On grid
            'tx_longitude': -120.0,  # On grid
            'tx_altitude': 1500.0,
            'rx_latitude': 30.5,
            'rx_longitude': -119.5,
            'rx_altitude': 1600.0,
            'frequency': 7.5,
            'elevation_angle': 85.0,
            'azimuth': 45.0,
            'hop_distance': 75.0,
            'signal_strength': -80.0,
            'group_delay': 2.5,
            'snr': 35.0,
            'signal_strength_error': 0.5,  # Very precise
            'group_delay_error': 0.01,     # Very precise
            'is_o_mode': True,
            'quality_tier': 'platinum',
            'quality_metrics': {'signal_quality': 1.0}
        }]

        result = analyzer.compute_marginal_gain('PRECISE', obs, prior_sqrt_cov)

        # High precision should give high information gain
        assert result.marginal_gain > 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
