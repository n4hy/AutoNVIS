"""
NVIS Sounder Network Analyzer

Comprehensive analysis of NVIS sounder network performance,
including information gain, coverage, and optimization recommendations.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..common.logging_config import ServiceLogger
from ..common.message_queue import MessageQueueClient, Topics
from ..ingestion.nvis.protocol_adapters.base_adapter import SounderMetadata
from .information_gain_analyzer import InformationGainAnalyzer
from .optimal_placement import OptimalPlacementRecommender


class NetworkAnalyzer:
    """
    Comprehensive NVIS sounder network analyzer

    Integrates:
    - Information gain analysis
    - Optimal placement recommendations
    - Network coverage assessment
    - Quality improvement suggestions
    """

    def __init__(
        self,
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
        alt_grid: np.ndarray,
        mq_client: Optional[MessageQueueClient] = None
    ):
        """
        Initialize network analyzer

        Args:
            lat_grid: Latitude grid (degrees)
            lon_grid: Longitude grid (degrees)
            alt_grid: Altitude grid (km)
            mq_client: Message queue client (optional)
        """
        self.lat_grid = lat_grid
        self.lon_grid = lon_grid
        self.alt_grid = alt_grid

        grid_shape = (len(lat_grid), len(lon_grid), len(alt_grid))

        # Components
        self.info_gain_analyzer = InformationGainAnalyzer(
            grid_shape, lat_grid, lon_grid, alt_grid
        )

        self.placement_recommender = OptimalPlacementRecommender(
            lat_grid, lon_grid, alt_grid
        )

        self.mq_client = mq_client
        self.logger = ServiceLogger("network_analyzer")

        # State
        self.last_analysis_time = None
        self.analysis_history = []

    def analyze_network(
        self,
        sounders: List[SounderMetadata],
        observations: List[Dict[str, Any]],
        prior_sqrt_cov: np.ndarray
    ) -> Dict[str, Any]:
        """
        Perform comprehensive network analysis

        Args:
            sounders: List of sounder metadata
            observations: Recent NVIS observations
            prior_sqrt_cov: Prior sqrt covariance

        Returns:
            Comprehensive analysis results
        """
        self.logger.info(
            f"Analyzing network: {len(sounders)} sounders, {len(observations)} observations"
        )

        analysis = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'network_overview': self._analyze_network_overview(sounders, observations),
            'information_gain': self._analyze_information_gain(observations, prior_sqrt_cov),
            'coverage_analysis': self._analyze_coverage(sounders, observations),
            'quality_analysis': self._analyze_quality(observations),
            'recommendations': self._generate_recommendations(
                sounders, observations, prior_sqrt_cov
            )
        }

        # Store in history
        self.analysis_history.append(analysis)
        if len(self.analysis_history) > 100:
            self.analysis_history = self.analysis_history[-100:]

        self.last_analysis_time = datetime.utcnow()

        # Publish to message queue
        if self.mq_client:
            self._publish_analysis(analysis)

        return analysis

    def _analyze_network_overview(
        self,
        sounders: List[SounderMetadata],
        observations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Network overview statistics"""
        # Group observations by sounder
        obs_per_sounder = {}
        for obs in observations:
            sid = obs['sounder_id']
            obs_per_sounder[sid] = obs_per_sounder.get(sid, 0) + 1

        # Quality tier distribution
        tier_counts = {}
        for obs in observations:
            tier = obs.get('quality_tier', 'unknown')
            tier_counts[tier] = tier_counts.get(tier, 0) + 1

        # Coverage area (bounding box)
        if len(observations) > 0:
            lats = [obs['tx_latitude'] for obs in observations] + \
                   [obs['rx_latitude'] for obs in observations]
            lons = [obs['tx_longitude'] for obs in observations] + \
                   [obs['rx_longitude'] for obs in observations]

            coverage_area = {
                'lat_min': min(lats),
                'lat_max': max(lats),
                'lon_min': min(lons),
                'lon_max': max(lons),
                'span_lat': max(lats) - min(lats),
                'span_lon': max(lons) - min(lons)
            }
        else:
            coverage_area = None

        return {
            'n_sounders': len(sounders),
            'n_observations': len(observations),
            'active_sounders': len(obs_per_sounder),
            'avg_obs_per_sounder': np.mean(list(obs_per_sounder.values())) if obs_per_sounder else 0,
            'quality_tier_distribution': tier_counts,
            'coverage_area': coverage_area
        }

    def _analyze_information_gain(
        self,
        observations: List[Dict[str, Any]],
        prior_sqrt_cov: np.ndarray
    ) -> Dict[str, Any]:
        """Information gain analysis"""
        if len(observations) == 0:
            return {
                'total_information_gain': 0.0,
                'sounder_contributions': {},
                'tier_contributions': {}
            }

        # Compute network information
        network_info = self.info_gain_analyzer.compute_network_information(
            observations, prior_sqrt_cov
        )

        # Sort sounders by contribution
        sorted_sounders = sorted(
            network_info['sounder_contributions'].items(),
            key=lambda x: x[1]['relative_contribution'],
            reverse=True
        )

        return {
            'total_information_gain': network_info['total_information_gain'],
            'relative_uncertainty_reduction': network_info['relative_uncertainty_reduction'],
            'trace_prior': network_info['trace_prior'],
            'trace_posterior': network_info['trace_posterior'],
            'top_contributors': [
                {
                    'sounder_id': sid,
                    'contribution': data['relative_contribution'],
                    'marginal_gain': data['marginal_gain'],
                    'n_observations': data['n_observations']
                }
                for sid, data in sorted_sounders[:10]  # Top 10
            ],
            'tier_contributions': network_info['tier_contributions']
        }

    def _analyze_coverage(
        self,
        sounders: List[SounderMetadata],
        observations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Coverage analysis"""
        # Observation density
        if len(observations) > 0:
            obs_locations = []
            for obs in observations:
                mid_lat = (obs['tx_latitude'] + obs['rx_latitude']) / 2.0
                mid_lon = (obs['tx_longitude'] + obs['rx_longitude']) / 2.0
                obs_locations.append((mid_lat, mid_lon))

            # Compute spatial clustering
            distances = []
            for i, (lat1, lon1) in enumerate(obs_locations):
                for j, (lat2, lon2) in enumerate(obs_locations[i+1:], i+1):
                    dist = self._haversine_distance(lat1, lon1, lat2, lon2)
                    distances.append(dist)

            avg_spacing = np.mean(distances) if distances else 0.0
            min_spacing = np.min(distances) if distances else 0.0
        else:
            avg_spacing = 0.0
            min_spacing = 0.0

        # Coverage gaps (regions > 500 km from any observation)
        gaps = self._identify_coverage_gaps(observations)

        return {
            'average_observation_spacing_km': avg_spacing,
            'minimum_observation_spacing_km': min_spacing,
            'coverage_gaps': gaps
        }

    def _analyze_quality(
        self,
        observations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Quality analysis"""
        if len(observations) == 0:
            return {
                'overall_quality_score': 0.0,
                'tier_statistics': {}
            }

        # Overall quality
        quality_scores = []
        for obs in observations:
            metrics = obs.get('quality_metrics', {})
            score = np.mean([
                metrics.get('signal_quality', 0.5),
                metrics.get('calibration_quality', 0.5),
                metrics.get('temporal_quality', 0.5),
                metrics.get('spatial_quality', 0.5),
                metrics.get('equipment_quality', 0.5),
                metrics.get('historical_quality', 0.5)
            ])
            quality_scores.append(score)

        overall_quality = np.mean(quality_scores)

        # Per-tier statistics
        tier_stats = {}
        for tier in ['platinum', 'gold', 'silver', 'bronze']:
            tier_obs = [obs for obs in observations if obs.get('quality_tier') == tier]
            if tier_obs:
                tier_snr = [obs['snr'] for obs in tier_obs]
                tier_stats[tier] = {
                    'count': len(tier_obs),
                    'avg_snr': np.mean(tier_snr),
                    'fraction': len(tier_obs) / len(observations)
                }

        return {
            'overall_quality_score': overall_quality,
            'tier_statistics': tier_stats,
            'avg_snr': np.mean([obs['snr'] for obs in observations]),
            'quality_score_distribution': {
                'mean': np.mean(quality_scores),
                'std': np.std(quality_scores),
                'min': np.min(quality_scores),
                'max': np.max(quality_scores)
            }
        }

    def _generate_recommendations(
        self,
        sounders: List[SounderMetadata],
        observations: List[Dict[str, Any]],
        prior_sqrt_cov: np.ndarray
    ) -> Dict[str, Any]:
        """Generate recommendations for network improvement"""
        recommendations = {
            'new_sounders': [],
            'upgrades': [],
            'coverage_priorities': []
        }

        # Recommend new sounder locations (top 3)
        if len(sounders) < 50:  # Only if network not saturated
            new_locations = self.placement_recommender.recommend_multiple_locations(
                n_sounders=3,
                existing_sounders=sounders,
                recent_observations=observations,
                prior_sqrt_cov=prior_sqrt_cov,
                assumed_tier="gold"
            )

            recommendations['new_sounders'] = [
                {
                    'priority': i + 1,
                    'latitude': loc.latitude,
                    'longitude': loc.longitude,
                    'expected_gain': loc.expected_gain,
                    'nearby_sounders': loc.nearby_sounders
                }
                for i, loc in enumerate(new_locations)
            ]

        # Recommend upgrades for low-performing sounders
        if len(observations) > 0:
            marginal_gains = self.info_gain_analyzer.compute_all_marginal_gains(
                observations, prior_sqrt_cov
            )

            # Find sounders with low contribution but high observation count
            for sounder_id, result in marginal_gains.items():
                if result.n_observations >= 10:
                    # Check if upgrade would help
                    current_tier = next(
                        (obs.get('quality_tier', 'unknown') for obs in observations
                         if obs['sounder_id'] == sounder_id),
                        'unknown'
                    )

                    if current_tier in ['bronze', 'silver']:
                        target_tier = 'gold' if current_tier == 'bronze' else 'platinum'
                        improvement = self.info_gain_analyzer.predict_improvement_from_upgrade(
                            sounder_id, observations, prior_sqrt_cov, target_tier
                        )

                        if improvement['relative_improvement'] > 0.2:  # 20% improvement
                            recommendations['upgrades'].append({
                                'sounder_id': sounder_id,
                                'current_tier': current_tier,
                                'recommended_tier': target_tier,
                                'expected_improvement': improvement['improvement'],
                                'relative_improvement': improvement['relative_improvement']
                            })

        # Coverage priorities (identify gaps)
        gaps = self._identify_coverage_gaps(observations)
        recommendations['coverage_priorities'] = [
            {
                'latitude': gap['center_lat'],
                'longitude': gap['center_lon'],
                'gap_size_km': gap['radius_km']
            }
            for gap in gaps[:5]  # Top 5 gaps
        ]

        return recommendations

    def _identify_coverage_gaps(
        self,
        observations: List[Dict[str, Any]],
        gap_threshold_km: float = 500.0
    ) -> List[Dict[str, Any]]:
        """
        Identify coverage gaps in the network

        Args:
            observations: NVIS observations
            gap_threshold_km: Minimum gap size to report

        Returns:
            List of coverage gaps
        """
        if len(observations) == 0:
            return []

        # Sample grid locations
        sample_lats = np.linspace(self.lat_grid.min(), self.lat_grid.max(), 20)
        sample_lons = np.linspace(self.lon_grid.min(), self.lon_grid.max(), 20)

        gaps = []

        for lat in sample_lats:
            for lon in sample_lons:
                # Find distance to nearest observation
                min_dist = np.inf
                for obs in observations:
                    obs_lat = (obs['tx_latitude'] + obs['rx_latitude']) / 2.0
                    obs_lon = (obs['tx_longitude'] + obs['rx_longitude']) / 2.0
                    dist = self._haversine_distance(lat, lon, obs_lat, obs_lon)
                    min_dist = min(min_dist, dist)

                if min_dist > gap_threshold_km:
                    gaps.append({
                        'center_lat': lat,
                        'center_lon': lon,
                        'radius_km': min_dist
                    })

        # Sort by gap size
        gaps.sort(key=lambda x: x['radius_km'], reverse=True)

        return gaps

    def _publish_analysis(self, analysis: Dict[str, Any]):
        """
        Publish analysis to message queue

        Args:
            analysis: Analysis results
        """
        try:
            self.mq_client.publish(
                topic=Topics.ANALYSIS_INFO_GAIN,
                data=analysis,
                source="network_analyzer"
            )
            self.logger.debug("Published analysis to message queue")
        except Exception as e:
            self.logger.error(f"Error publishing analysis: {e}", exc_info=True)

    def _haversine_distance(
        self,
        lat1: float, lon1: float,
        lat2: float, lon2: float
    ) -> float:
        """Compute great circle distance in km"""
        R = 6371.0
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        dlon = np.radians(lon2 - lon1)
        dlat = np.radians(lat2 - lat1)
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return R * c

    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of recent analyses"""
        if not self.analysis_history:
            return {
                'n_analyses': 0,
                'last_analysis_time': None
            }

        recent = self.analysis_history[-1]

        return {
            'n_analyses': len(self.analysis_history),
            'last_analysis_time': self.last_analysis_time.isoformat() if self.last_analysis_time else None,
            'last_total_information_gain': recent['information_gain']['total_information_gain'],
            'last_n_sounders': recent['network_overview']['n_sounders'],
            'last_n_observations': recent['network_overview']['n_observations']
        }
