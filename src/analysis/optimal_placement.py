"""
Optimal Placement Recommender for NVIS Sounders

Finds optimal locations for new sounders to maximize information gain
and minimize network redundancy.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

from ..common.logging_config import ServiceLogger
from ..ingestion.nvis.protocol_adapters.base_adapter import SounderMetadata
from ..ingestion.nvis.quality_assessor import QualityTier


@dataclass
class PlacementRecommendation:
    """Recommendation for new sounder placement"""
    latitude: float
    longitude: float
    expected_gain: float
    coverage_gap_score: float
    redundancy_score: float
    combined_score: float
    nearby_sounders: List[str]
    estimated_tier: str


class OptimalPlacementRecommender:
    """
    Recommends optimal locations for new NVIS sounders

    Optimization objectives:
    1. Maximize information gain (reduce state uncertainty)
    2. Fill coverage gaps (spatial diversity)
    3. Minimize redundancy (avoid clustering)

    Combined score = α × info_gain + β × coverage_gap - γ × redundancy
    """

    def __init__(
        self,
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
        alt_grid: np.ndarray,
        alpha: float = 0.5,  # Information gain weight
        beta: float = 0.3,   # Coverage gap weight
        gamma: float = 0.2   # Redundancy penalty weight
    ):
        """
        Initialize placement recommender

        Args:
            lat_grid: Latitude grid (degrees)
            lon_grid: Longitude grid (degrees)
            alt_grid: Altitude grid (km)
            alpha: Weight for information gain
            beta: Weight for coverage gaps
            gamma: Weight for redundancy penalty
        """
        self.lat_grid = lat_grid
        self.lon_grid = lon_grid
        self.alt_grid = alt_grid

        # Weights (should sum to 1.0)
        total_weight = alpha + beta + gamma
        self.alpha = alpha / total_weight
        self.beta = beta / total_weight
        self.gamma = gamma / total_weight

        self.logger = ServiceLogger("optimal_placement")

    def recommend_new_sounder_location(
        self,
        existing_sounders: List[SounderMetadata],
        recent_observations: List[Dict[str, Any]],
        prior_sqrt_cov: Optional[np.ndarray] = None,
        assumed_tier: str = "gold",
        search_resolution: int = 20
    ) -> PlacementRecommendation:
        """
        Find optimal location for a new sounder

        Args:
            existing_sounders: List of existing sounder metadata
            recent_observations: Recent observation history
            prior_sqrt_cov: Prior sqrt covariance (for info gain estimation)
            assumed_tier: Assumed quality tier for new sounder
            search_resolution: Grid search resolution

        Returns:
            PlacementRecommendation with optimal location
        """
        self.logger.info(
            f"Searching for optimal location (resolution={search_resolution}, "
            f"tier={assumed_tier})"
        )

        # Generate candidate locations
        candidate_lats = np.linspace(self.lat_grid.min(), self.lat_grid.max(), search_resolution)
        candidate_lons = np.linspace(self.lon_grid.min(), self.lon_grid.max(), search_resolution)

        best_score = -np.inf
        best_recommendation = None

        # Grid search
        for lat in candidate_lats:
            for lon in candidate_lons:
                # Compute scores
                info_gain = self._estimate_information_gain(
                    lat, lon, existing_sounders, prior_sqrt_cov, assumed_tier
                )

                coverage_gap = self._compute_coverage_gap_score(
                    lat, lon, existing_sounders, recent_observations
                )

                redundancy = self._compute_redundancy_score(
                    lat, lon, existing_sounders
                )

                # Combined score
                combined = (
                    self.alpha * info_gain +
                    self.beta * coverage_gap -
                    self.gamma * redundancy
                )

                if combined > best_score:
                    best_score = combined

                    # Find nearby sounders
                    nearby = self._find_nearby_sounders(lat, lon, existing_sounders, radius_km=500.0)

                    best_recommendation = PlacementRecommendation(
                        latitude=lat,
                        longitude=lon,
                        expected_gain=info_gain,
                        coverage_gap_score=coverage_gap,
                        redundancy_score=redundancy,
                        combined_score=combined,
                        nearby_sounders=[s.sounder_id for s in nearby],
                        estimated_tier=assumed_tier
                    )

        self.logger.info(
            f"Optimal location: ({best_recommendation.latitude:.2f}, "
            f"{best_recommendation.longitude:.2f}) with score {best_score:.3f}"
        )

        return best_recommendation

    def recommend_multiple_locations(
        self,
        n_sounders: int,
        existing_sounders: List[SounderMetadata],
        recent_observations: List[Dict[str, Any]],
        prior_sqrt_cov: Optional[np.ndarray] = None,
        assumed_tier: str = "gold"
    ) -> List[PlacementRecommendation]:
        """
        Recommend multiple new sounder locations

        Uses greedy algorithm: iteratively add best location,
        update network, repeat.

        Args:
            n_sounders: Number of sounders to recommend
            existing_sounders: Existing sounders
            recent_observations: Recent observations
            prior_sqrt_cov: Prior sqrt covariance
            assumed_tier: Assumed quality tier

        Returns:
            List of placement recommendations
        """
        recommendations = []
        current_sounders = existing_sounders.copy()

        for i in range(n_sounders):
            self.logger.info(f"Finding location {i+1}/{n_sounders}")

            # Find best location given current network
            rec = self.recommend_new_sounder_location(
                current_sounders,
                recent_observations,
                prior_sqrt_cov,
                assumed_tier
            )

            recommendations.append(rec)

            # Add to network for next iteration
            new_sounder = SounderMetadata(
                sounder_id=f"NEW_{i+1}",
                name=f"Recommended {i+1}",
                operator="Planning",
                location=f"({rec.latitude:.2f}, {rec.longitude:.2f})",
                latitude=rec.latitude,
                longitude=rec.longitude,
                altitude=0.0,
                equipment_type="professional" if assumed_tier == "platinum" else "research",
                calibration_status="calibrated" if assumed_tier in ["platinum", "gold"] else "unknown"
            )
            current_sounders.append(new_sounder)

        return recommendations

    def _estimate_information_gain(
        self,
        lat: float,
        lon: float,
        existing_sounders: List[SounderMetadata],
        prior_sqrt_cov: Optional[np.ndarray],
        assumed_tier: str
    ) -> float:
        """
        Estimate information gain from a sounder at this location

        Args:
            lat: Latitude
            lon: Longitude
            existing_sounders: Existing sounders
            prior_sqrt_cov: Prior covariance
            assumed_tier: Quality tier

        Returns:
            Normalized information gain score [0, 1]
        """
        if prior_sqrt_cov is None:
            # Simplified heuristic: distance to existing sounders
            if len(existing_sounders) == 0:
                return 1.0

            min_dist = min(
                self._haversine_distance(lat, lon, s.latitude, s.longitude)
                for s in existing_sounders
            )

            # Normalize: 0 km → 0, 1000 km → 1.0
            return np.clip(min_dist / 1000.0, 0.0, 1.0)

        # Full information gain computation would require:
        # 1. Simulate observations from this location
        # 2. Compute Fisher Information
        # 3. Compute trace reduction
        # For now, use simplified heuristic

        return self._estimate_information_gain(lat, lon, existing_sounders, None, assumed_tier)

    def _compute_coverage_gap_score(
        self,
        lat: float,
        lon: float,
        existing_sounders: List[SounderMetadata],
        recent_observations: List[Dict[str, Any]]
    ) -> float:
        """
        Compute coverage gap score

        High score = fills a gap in spatial coverage

        Args:
            lat: Latitude
            lon: Longitude
            existing_sounders: Existing sounders
            recent_observations: Recent observations

        Returns:
            Coverage gap score [0, 1]
        """
        # Find distance to nearest observation
        if len(recent_observations) == 0:
            # No observations yet, all locations equally good
            return 0.5

        min_dist = np.inf
        for obs in recent_observations:
            obs_lat = (obs['tx_latitude'] + obs['rx_latitude']) / 2.0
            obs_lon = (obs['tx_longitude'] + obs['rx_longitude']) / 2.0

            dist = self._haversine_distance(lat, lon, obs_lat, obs_lon)
            min_dist = min(min_dist, dist)

        # Normalize: 0 km → 0, 1000 km → 1.0
        gap_score = np.clip(min_dist / 1000.0, 0.0, 1.0)

        return gap_score

    def _compute_redundancy_score(
        self,
        lat: float,
        lon: float,
        existing_sounders: List[SounderMetadata]
    ) -> float:
        """
        Compute redundancy score

        High score = close to existing sounders (redundant)

        Args:
            lat: Latitude
            lon: Longitude
            existing_sounders: Existing sounders

        Returns:
            Redundancy score [0, 1]
        """
        if len(existing_sounders) == 0:
            return 0.0

        # Find distance to nearest sounder
        min_dist = min(
            self._haversine_distance(lat, lon, s.latitude, s.longitude)
            for s in existing_sounders
        )

        # Normalize: 0 km → 1.0, 500 km → 0.0
        redundancy = 1.0 - np.clip(min_dist / 500.0, 0.0, 1.0)

        return redundancy

    def _find_nearby_sounders(
        self,
        lat: float,
        lon: float,
        sounders: List[SounderMetadata],
        radius_km: float
    ) -> List[SounderMetadata]:
        """
        Find sounders within radius

        Args:
            lat: Center latitude
            lon: Center longitude
            sounders: List of sounders
            radius_km: Search radius

        Returns:
            List of nearby sounders
        """
        nearby = []
        for s in sounders:
            dist = self._haversine_distance(lat, lon, s.latitude, s.longitude)
            if dist <= radius_km:
                nearby.append(s)

        return nearby

    def analyze_proposed_location(
        self,
        lat: float,
        lon: float,
        existing_sounders: List[SounderMetadata],
        recent_observations: List[Dict[str, Any]],
        prior_sqrt_cov: Optional[np.ndarray] = None,
        assumed_tier: str = "gold"
    ) -> Dict[str, Any]:
        """
        Analyze a proposed sounder location ("what-if" analysis)

        Args:
            lat: Proposed latitude
            lon: Proposed longitude
            existing_sounders: Existing sounders
            recent_observations: Recent observations
            prior_sqrt_cov: Prior sqrt covariance
            assumed_tier: Assumed quality tier

        Returns:
            Analysis results
        """
        # Compute scores
        info_gain = self._estimate_information_gain(
            lat, lon, existing_sounders, prior_sqrt_cov, assumed_tier
        )

        coverage_gap = self._compute_coverage_gap_score(
            lat, lon, existing_sounders, recent_observations
        )

        redundancy = self._compute_redundancy_score(
            lat, lon, existing_sounders
        )

        combined = (
            self.alpha * info_gain +
            self.beta * coverage_gap -
            self.gamma * redundancy
        )

        # Find nearby sounders
        nearby = self._find_nearby_sounders(lat, lon, existing_sounders, radius_km=500.0)

        # Distance to nearest
        if len(existing_sounders) > 0:
            nearest_dist = min(
                self._haversine_distance(lat, lon, s.latitude, s.longitude)
                for s in existing_sounders
            )
            nearest_sounder = min(
                existing_sounders,
                key=lambda s: self._haversine_distance(lat, lon, s.latitude, s.longitude)
            )
        else:
            nearest_dist = None
            nearest_sounder = None

        return {
            'latitude': lat,
            'longitude': lon,
            'assumed_tier': assumed_tier,
            'scores': {
                'information_gain': info_gain,
                'coverage_gap': coverage_gap,
                'redundancy': redundancy,
                'combined': combined
            },
            'nearby_sounders': [
                {
                    'sounder_id': s.sounder_id,
                    'distance_km': self._haversine_distance(lat, lon, s.latitude, s.longitude)
                }
                for s in nearby
            ],
            'nearest_sounder': {
                'sounder_id': nearest_sounder.sounder_id if nearest_sounder else None,
                'distance_km': nearest_dist
            } if nearest_sounder else None,
            'recommendation': 'Good' if combined > 0.6 else 'Fair' if combined > 0.4 else 'Poor'
        }

    def generate_placement_heatmap(
        self,
        existing_sounders: List[SounderMetadata],
        recent_observations: List[Dict[str, Any]],
        resolution: int = 50
    ) -> np.ndarray:
        """
        Generate heatmap of placement scores

        Args:
            existing_sounders: Existing sounders
            recent_observations: Recent observations
            resolution: Grid resolution

        Returns:
            2D array of combined scores (lat × lon)
        """
        lats = np.linspace(self.lat_grid.min(), self.lat_grid.max(), resolution)
        lons = np.linspace(self.lon_grid.min(), self.lon_grid.max(), resolution)

        heatmap = np.zeros((resolution, resolution))

        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                info_gain = self._estimate_information_gain(
                    lat, lon, existing_sounders, None, "gold"
                )
                coverage_gap = self._compute_coverage_gap_score(
                    lat, lon, existing_sounders, recent_observations
                )
                redundancy = self._compute_redundancy_score(
                    lat, lon, existing_sounders
                )

                combined = (
                    self.alpha * info_gain +
                    self.beta * coverage_gap -
                    self.gamma * redundancy
                )

                heatmap[i, j] = combined

        return heatmap

    def _haversine_distance(
        self,
        lat1: float, lon1: float,
        lat2: float, lon2: float
    ) -> float:
        """
        Compute great circle distance

        Args:
            lat1, lon1: Point 1 (degrees)
            lat2, lon2: Point 2 (degrees)

        Returns:
            Distance in km
        """
        R = 6371.0  # Earth radius in km

        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        dlon = np.radians(lon2 - lon1)
        dlat = np.radians(lat2 - lat1)

        a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        return R * c
