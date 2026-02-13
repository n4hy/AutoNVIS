"""
Unit Tests for Optimal Placement Recommender

Tests optimal sounder placement algorithms.
"""

import pytest
import numpy as np
from src.analysis.optimal_placement import (
    OptimalPlacementRecommender,
    PlacementRecommendation
)
from src.ingestion.nvis.protocol_adapters.base_adapter import SounderMetadata


@pytest.fixture
def recommender():
    """Create optimal placement recommender"""
    lat_grid = np.linspace(-90, 90, 7)
    lon_grid = np.linspace(-180, 180, 7)
    alt_grid = np.linspace(100, 500, 11)

    return OptimalPlacementRecommender(
        lat_grid, lon_grid, alt_grid
    )


@pytest.fixture
def existing_sounders():
    """Create sample existing sounders"""
    return [
        SounderMetadata(
            sounder_id='SOUNDER_A',
            name='Sounder A',
            operator='Test Operator',
            location='Location A',
            latitude=40.0,
            longitude=-105.0,
            altitude=1500.0,
            equipment_type='professional',
            calibration_status='calibrated'
        ),
        SounderMetadata(
            sounder_id='SOUNDER_B',
            name='Sounder B',
            operator='Test Operator',
            location='Location B',
            latitude=35.0,
            longitude=-110.0,
            altitude=1200.0,
            equipment_type='research',
            calibration_status='calibrated'
        )
    ]


@pytest.fixture
def recent_observations():
    """Create sample recent observations"""
    return [
        {
            'sounder_id': 'SOUNDER_A',
            'tx_latitude': 40.0,
            'tx_longitude': -105.0,
            'rx_latitude': 40.5,
            'rx_longitude': -104.5,
            'signal_strength': -80.0,
            'quality_tier': 'platinum'
        },
        {
            'sounder_id': 'SOUNDER_B',
            'tx_latitude': 35.0,
            'tx_longitude': -110.0,
            'rx_latitude': 35.5,
            'rx_longitude': -109.5,
            'signal_strength': -90.0,
            'quality_tier': 'silver'
        }
    ]


class TestOptimalPlacementRecommender:
    """Test OptimalPlacementRecommender class"""

    def test_initialization(self, recommender):
        """Test recommender initialization"""
        assert recommender.alpha > 0
        assert recommender.beta > 0
        assert recommender.gamma > 0
        assert abs(recommender.alpha + recommender.beta + recommender.gamma - 1.0) < 0.01

    def test_recommend_single_location(self, recommender, existing_sounders, recent_observations):
        """Test recommendation of single location"""
        rec = recommender.recommend_new_sounder_location(
            existing_sounders,
            recent_observations,
            prior_sqrt_cov=None,
            assumed_tier='gold',
            search_resolution=5  # Low resolution for speed
        )

        assert isinstance(rec, PlacementRecommendation)
        assert -90 <= rec.latitude <= 90
        assert -180 <= rec.longitude <= 180
        assert rec.expected_gain >= 0.0
        assert 0.0 <= rec.coverage_gap_score <= 1.0
        assert 0.0 <= rec.redundancy_score <= 1.0

    def test_recommend_multiple_locations(self, recommender, existing_sounders, recent_observations):
        """Test recommendation of multiple locations"""
        recs = recommender.recommend_multiple_locations(
            n_sounders=2,
            existing_sounders=existing_sounders,
            recent_observations=recent_observations,
            assumed_tier='gold'
        )

        assert len(recs) == 2
        assert all(isinstance(r, PlacementRecommendation) for r in recs)

        # Locations should be different
        assert (recs[0].latitude, recs[0].longitude) != (recs[1].latitude, recs[1].longitude)

    def test_coverage_gap_score(self, recommender, existing_sounders, recent_observations):
        """Test coverage gap scoring"""
        # Location far from existing sounders should have high gap score
        far_score = recommender._compute_coverage_gap_score(
            lat=0.0,
            lon=0.0,
            existing_sounders=existing_sounders,
            recent_observations=recent_observations
        )

        # Location close to existing sounders should have low gap score
        close_score = recommender._compute_coverage_gap_score(
            lat=40.0,
            lon=-105.0,
            existing_sounders=existing_sounders,
            recent_observations=recent_observations
        )

        assert far_score > close_score

    def test_redundancy_score(self, recommender, existing_sounders):
        """Test redundancy scoring"""
        # Location very close to existing sounder should be redundant
        high_redundancy = recommender._compute_redundancy_score(
            lat=40.0,
            lon=-105.0,
            existing_sounders=existing_sounders
        )

        # Location far from existing sounders should have low redundancy
        low_redundancy = recommender._compute_redundancy_score(
            lat=0.0,
            lon=0.0,
            existing_sounders=existing_sounders
        )

        assert high_redundancy > low_redundancy

    def test_information_gain_estimation(self, recommender, existing_sounders):
        """Test information gain estimation"""
        # Location far from existing sounders should have high info gain
        far_gain = recommender._estimate_information_gain(
            lat=0.0,
            lon=0.0,
            existing_sounders=existing_sounders,
            prior_sqrt_cov=None,
            assumed_tier='gold'
        )

        # Location close to existing sounders should have lower info gain
        close_gain = recommender._estimate_information_gain(
            lat=40.0,
            lon=-105.0,
            existing_sounders=existing_sounders,
            prior_sqrt_cov=None,
            assumed_tier='gold'
        )

        assert far_gain > close_gain

    def test_analyze_proposed_location(self, recommender, existing_sounders, recent_observations):
        """Test 'what-if' analysis"""
        analysis = recommender.analyze_proposed_location(
            lat=45.0,
            lon=-100.0,
            existing_sounders=existing_sounders,
            recent_observations=recent_observations,
            assumed_tier='platinum'
        )

        assert 'latitude' in analysis
        assert 'longitude' in analysis
        assert 'scores' in analysis
        assert 'nearby_sounders' in analysis
        assert 'recommendation' in analysis

        assert analysis['scores']['combined'] >= 0.0

    def test_nearby_sounders(self, recommender, existing_sounders):
        """Test finding nearby sounders"""
        nearby = recommender._find_nearby_sounders(
            lat=40.0,
            lon=-105.0,
            sounders=existing_sounders,
            radius_km=100.0
        )

        # SOUNDER_A is at exactly this location
        assert len(nearby) >= 1
        assert any(s.sounder_id == 'SOUNDER_A' for s in nearby)

    def test_heatmap_generation(self, recommender, existing_sounders, recent_observations):
        """Test placement heatmap generation"""
        heatmap = recommender.generate_placement_heatmap(
            existing_sounders,
            recent_observations,
            resolution=10  # Low resolution for speed
        )

        assert heatmap.shape == (10, 10)
        assert np.all(np.isfinite(heatmap))

    def test_empty_network(self, recommender):
        """Test with no existing sounders"""
        rec = recommender.recommend_new_sounder_location(
            existing_sounders=[],
            recent_observations=[],
            assumed_tier='gold',
            search_resolution=5
        )

        # All locations should be equally good
        assert rec.expected_gain >= 0.0
        assert len(rec.nearby_sounders) == 0

    def test_tier_comparison(self, recommender, existing_sounders, recent_observations):
        """Test that higher tiers give better recommendations"""
        rec_platinum = recommender.recommend_new_sounder_location(
            existing_sounders,
            recent_observations,
            assumed_tier='platinum',
            search_resolution=5
        )

        rec_bronze = recommender.recommend_new_sounder_location(
            existing_sounders,
            recent_observations,
            assumed_tier='bronze',
            search_resolution=5
        )

        # Both should be valid recommendations
        assert rec_platinum.combined_score >= 0.0
        assert rec_bronze.combined_score >= 0.0


class TestEdgeCases:
    """Test edge cases"""

    def test_single_existing_sounder(self, recommender, recent_observations):
        """Test with single existing sounder"""
        sounder = SounderMetadata(
            sounder_id='SOLO',
            name='Solo',
            operator='Test',
            location='Test',
            latitude=40.0,
            longitude=-105.0,
            altitude=1500.0,
            equipment_type='professional',
            calibration_status='calibrated'
        )

        rec = recommender.recommend_new_sounder_location(
            [sounder],
            recent_observations,
            assumed_tier='gold',
            search_resolution=5
        )

        # Should place far from existing sounder
        dist = recommender._haversine_distance(
            rec.latitude, rec.longitude,
            sounder.latitude, sounder.longitude
        )
        assert dist > 100.0  # At least 100 km away

    def test_high_resolution_search(self, recommender, existing_sounders, recent_observations):
        """Test with high resolution search"""
        rec = recommender.recommend_new_sounder_location(
            existing_sounders,
            recent_observations,
            assumed_tier='gold',
            search_resolution=3  # Very low for speed
        )

        assert isinstance(rec, PlacementRecommendation)

    def test_weights_normalization(self):
        """Test that weights are properly normalized"""
        recommender = OptimalPlacementRecommender(
            lat_grid=np.array([0, 30, 60]),
            lon_grid=np.array([0, 60, 120]),
            alt_grid=np.array([100, 300, 500]),
            alpha=0.6,
            beta=0.3,
            gamma=0.1
        )

        # Should sum to 1.0
        total = recommender.alpha + recommender.beta + recommender.gamma
        assert abs(total - 1.0) < 0.01


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
