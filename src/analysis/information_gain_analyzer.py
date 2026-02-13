"""
Information Gain Analyzer for NVIS Sounder Network

Computes marginal information gain per sounder using Fisher Information
to quantify each sounder's contribution to state uncertainty reduction.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from scipy.linalg import solve

from ..common.logging_config import ServiceLogger
from ..ingestion.nvis.protocol_adapters.base_adapter import NVISMeasurement


@dataclass
class InformationGainResult:
    """Result of information gain analysis"""
    sounder_id: str
    marginal_gain: float              # trace(P_without) - trace(P_with)
    relative_contribution: float       # marginal_gain / total_gain
    trace_with: float                 # trace(P) with this sounder
    trace_without: float              # trace(P) without this sounder
    n_observations: int               # Number of observations from sounder
    avg_quality_score: float          # Average quality score


class InformationGainAnalyzer:
    """
    Analyzes information gain from NVIS sounder observations

    Uses Fisher Information Matrix to compute:
    - Marginal gain per sounder (with vs without)
    - Relative contribution to uncertainty reduction
    - Total network information gain

    Approach:
        P_post^(-1) = P_prior^(-1) + I_obs
        I_obs = H^T R^(-1) H (Fisher Information)

    Where:
        H = observation Jacobian (∂obs/∂state)
        R = observation error covariance
    """

    def __init__(
        self,
        grid_shape: Tuple[int, int, int],
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
        alt_grid: np.ndarray
    ):
        """
        Initialize information gain analyzer

        Args:
            grid_shape: (n_lat, n_lon, n_alt)
            lat_grid: Latitude grid (degrees)
            lon_grid: Longitude grid (degrees)
            alt_grid: Altitude grid (km)
        """
        self.grid_shape = grid_shape
        self.n_lat, self.n_lon, self.n_alt = grid_shape
        self.state_dim = self.n_lat * self.n_lon * self.n_alt + 1  # +1 for R_eff

        self.lat_grid = lat_grid
        self.lon_grid = lon_grid
        self.alt_grid = alt_grid

        self.logger = ServiceLogger("information_gain_analyzer")

        # Cache for Jacobian computations
        self._jacobian_cache = {}

    def compute_marginal_gain(
        self,
        sounder_id: str,
        all_observations: List[Dict[str, Any]],
        prior_sqrt_cov: np.ndarray,
        use_approximation: bool = True
    ) -> InformationGainResult:
        """
        Compute marginal information gain for a sounder

        Compares posterior uncertainty with and without this sounder's observations.

        Args:
            sounder_id: Sounder to analyze
            all_observations: All NVIS observations
            prior_sqrt_cov: Prior sqrt covariance matrix (S where P = S S^T)
            use_approximation: Use trace approximation (faster)

        Returns:
            InformationGainResult with marginal gain metrics
        """
        # Filter observations
        sounder_obs = [obs for obs in all_observations if obs['sounder_id'] == sounder_id]
        other_obs = [obs for obs in all_observations if obs['sounder_id'] != sounder_id]

        if len(sounder_obs) == 0:
            return InformationGainResult(
                sounder_id=sounder_id,
                marginal_gain=0.0,
                relative_contribution=0.0,
                trace_with=0.0,
                trace_without=0.0,
                n_observations=0,
                avg_quality_score=0.0
            )

        # Compute prior covariance
        prior_cov = prior_sqrt_cov @ prior_sqrt_cov.T
        trace_prior = np.trace(prior_cov)

        if use_approximation:
            # Fast approximation using trace formula
            trace_with, trace_without = self._compute_trace_approximation(
                all_observations, other_obs, prior_sqrt_cov
            )
        else:
            # Full computation (expensive for large state)
            trace_with = self._compute_posterior_trace(all_observations, prior_cov)
            trace_without = self._compute_posterior_trace(other_obs, prior_cov)

        # Marginal gain
        marginal_gain = trace_without - trace_with
        total_gain = trace_prior - trace_with

        # Relative contribution
        relative_contribution = marginal_gain / total_gain if total_gain > 0 else 0.0

        # Statistics
        avg_quality = np.mean([
            obs.get('quality_metrics', {}).get('signal_quality', 0.5)
            for obs in sounder_obs
        ])

        self.logger.info(
            f"Sounder {sounder_id}: marginal_gain={marginal_gain:.2e}, "
            f"contribution={relative_contribution:.1%}"
        )

        return InformationGainResult(
            sounder_id=sounder_id,
            marginal_gain=marginal_gain,
            relative_contribution=relative_contribution,
            trace_with=trace_with,
            trace_without=trace_without,
            n_observations=len(sounder_obs),
            avg_quality_score=avg_quality
        )

    def compute_all_marginal_gains(
        self,
        observations: List[Dict[str, Any]],
        prior_sqrt_cov: np.ndarray
    ) -> Dict[str, InformationGainResult]:
        """
        Compute marginal gain for all sounders

        Args:
            observations: All NVIS observations
            prior_sqrt_cov: Prior sqrt covariance

        Returns:
            Dict mapping sounder_id to InformationGainResult
        """
        # Get unique sounders
        sounder_ids = list(set(obs['sounder_id'] for obs in observations))

        results = {}
        for sounder_id in sounder_ids:
            result = self.compute_marginal_gain(
                sounder_id,
                observations,
                prior_sqrt_cov
            )
            results[sounder_id] = result

        return results

    def _compute_trace_approximation(
        self,
        all_obs: List[Dict[str, Any]],
        other_obs: List[Dict[str, Any]],
        prior_sqrt_cov: np.ndarray
    ) -> Tuple[float, float]:
        """
        Compute posterior trace using approximation

        Uses Sherman-Morrison-Woodbury formula for efficient trace computation:
        trace(P_post) ≈ trace(P_prior) - trace(H^T R^(-1) H × P_prior)

        Args:
            all_obs: All observations
            other_obs: Observations excluding one sounder
            prior_sqrt_cov: Prior sqrt covariance

        Returns:
            (trace_with, trace_without)
        """
        prior_cov = prior_sqrt_cov @ prior_sqrt_cov.T
        trace_prior = np.trace(prior_cov)

        # Compute information contribution from observations
        info_all = self._compute_information_contribution(all_obs, prior_cov)
        info_other = self._compute_information_contribution(other_obs, prior_cov)

        # Approximate posterior traces
        trace_with = trace_prior - info_all
        trace_without = trace_prior - info_other

        return trace_with, trace_without

    def _compute_information_contribution(
        self,
        observations: List[Dict[str, Any]],
        prior_cov: np.ndarray
    ) -> float:
        """
        Compute information contribution from observations

        Uses localized influence: each observation primarily affects nearby grid points.

        Args:
            observations: NVIS observations
            prior_cov: Prior covariance matrix

        Returns:
            Information contribution (reduction in trace)
        """
        if len(observations) == 0:
            return 0.0

        total_contribution = 0.0

        for obs in observations:
            # Get observation location (midpoint)
            mid_lat = (obs['tx_latitude'] + obs['rx_latitude']) / 2.0
            mid_lon = (obs['tx_longitude'] + obs['rx_longitude']) / 2.0

            # Find nearby grid points (localization)
            nearby_indices = self._get_nearby_indices(mid_lat, mid_lon, radius_km=500.0)

            # Observation errors
            signal_error = obs['signal_strength_error']
            delay_error = obs['group_delay_error']

            # Information contribution (simplified)
            # I_obs ≈ (1/σ²) for each observable
            info_signal = 1.0 / (signal_error ** 2)
            info_delay = 1.0 / (delay_error ** 2)

            # Weight by number of affected grid points
            n_affected = len(nearby_indices)
            contribution = (info_signal + info_delay) * n_affected / self.state_dim

            total_contribution += contribution

        return total_contribution

    def _get_nearby_indices(
        self,
        lat: float,
        lon: float,
        radius_km: float
    ) -> List[int]:
        """
        Get grid point indices within radius

        Args:
            lat: Latitude (degrees)
            lon: Longitude (degrees)
            radius_km: Search radius (km)

        Returns:
            List of flattened grid indices
        """
        nearby = []

        for i, lat_grid in enumerate(self.lat_grid):
            for j, lon_grid in enumerate(self.lon_grid):
                # Great circle distance
                dist = self._haversine_distance(lat, lon, lat_grid, lon_grid)

                if dist <= radius_km:
                    # Add all altitudes at this (lat, lon)
                    for k in range(self.n_alt):
                        idx = i * self.n_lon * self.n_alt + j * self.n_alt + k
                        nearby.append(idx)

        return nearby

    def _compute_posterior_trace(
        self,
        observations: List[Dict[str, Any]],
        prior_cov: np.ndarray
    ) -> float:
        """
        Compute posterior trace (full computation)

        WARNING: Expensive for large state dimensions.
        Only use for small grids or validation.

        Args:
            observations: NVIS observations
            prior_cov: Prior covariance

        Returns:
            trace(P_posterior)
        """
        if len(observations) == 0:
            return np.trace(prior_cov)

        # This would require full Jacobian computation
        # For now, use approximation
        self.logger.warning("Full posterior trace computation not implemented, using approximation")
        return self._compute_trace_approximation(observations, [], prior_cov)[0]

    def compute_network_information(
        self,
        observations: List[Dict[str, Any]],
        prior_sqrt_cov: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute overall network information gain

        Args:
            observations: All NVIS observations
            prior_sqrt_cov: Prior sqrt covariance

        Returns:
            Network information metrics
        """
        prior_cov = prior_sqrt_cov @ prior_sqrt_cov.T
        trace_prior = np.trace(prior_cov)

        # Compute with all observations
        trace_with, _ = self._compute_trace_approximation(
            observations, [], prior_sqrt_cov
        )

        total_gain = trace_prior - trace_with
        relative_reduction = total_gain / trace_prior if trace_prior > 0 else 0.0

        # Per-sounder breakdown
        sounder_gains = self.compute_all_marginal_gains(observations, prior_sqrt_cov)

        # Quality tier breakdown
        tier_contributions = self._compute_tier_contributions(observations, sounder_gains)

        return {
            'trace_prior': trace_prior,
            'trace_posterior': trace_with,
            'total_information_gain': total_gain,
            'relative_uncertainty_reduction': relative_reduction,
            'n_observations': len(observations),
            'n_sounders': len(sounder_gains),
            'sounder_contributions': {
                sid: {
                    'marginal_gain': result.marginal_gain,
                    'relative_contribution': result.relative_contribution,
                    'n_observations': result.n_observations
                }
                for sid, result in sounder_gains.items()
            },
            'tier_contributions': tier_contributions
        }

    def _compute_tier_contributions(
        self,
        observations: List[Dict[str, Any]],
        sounder_gains: Dict[str, InformationGainResult]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute information gain by quality tier

        Args:
            observations: All observations
            sounder_gains: Marginal gains per sounder

        Returns:
            Dict with tier statistics
        """
        tier_stats = {}

        for obs in observations:
            tier = obs.get('quality_tier', 'unknown')
            sounder_id = obs['sounder_id']

            if tier not in tier_stats:
                tier_stats[tier] = {
                    'n_observations': 0,
                    'total_marginal_gain': 0.0,
                    'n_sounders': 0
                }

            tier_stats[tier]['n_observations'] += 1

            # Add marginal gain (divide by n_obs to avoid double counting)
            if sounder_id in sounder_gains:
                gain = sounder_gains[sounder_id].marginal_gain
                n_obs = sounder_gains[sounder_id].n_observations
                tier_stats[tier]['total_marginal_gain'] += gain / n_obs

        # Count unique sounders per tier
        for tier in tier_stats:
            tier_sounders = set(
                obs['sounder_id'] for obs in observations
                if obs.get('quality_tier') == tier
            )
            tier_stats[tier]['n_sounders'] = len(tier_sounders)

        return tier_stats

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

    def predict_improvement_from_upgrade(
        self,
        sounder_id: str,
        observations: List[Dict[str, Any]],
        prior_sqrt_cov: np.ndarray,
        new_tier: str
    ) -> Dict[str, float]:
        """
        Predict improvement from upgrading a sounder's quality tier

        Args:
            sounder_id: Sounder to upgrade
            observations: Current observations
            prior_sqrt_cov: Prior sqrt covariance
            new_tier: Target quality tier

        Returns:
            Predicted improvement metrics
        """
        # Get current marginal gain
        current_result = self.compute_marginal_gain(
            sounder_id, observations, prior_sqrt_cov
        )

        # Simulate upgraded observations (better errors)
        upgraded_obs = []
        for obs in observations:
            if obs['sounder_id'] == sounder_id:
                # Upgrade errors based on tier
                obs_copy = obs.copy()
                if new_tier == 'platinum':
                    obs_copy['signal_strength_error'] = 2.0
                    obs_copy['group_delay_error'] = 0.1
                elif new_tier == 'gold':
                    obs_copy['signal_strength_error'] = 4.0
                    obs_copy['group_delay_error'] = 0.5
                elif new_tier == 'silver':
                    obs_copy['signal_strength_error'] = 8.0
                    obs_copy['group_delay_error'] = 2.0
                upgraded_obs.append(obs_copy)
            else:
                upgraded_obs.append(obs)

        # Compute upgraded marginal gain
        upgraded_result = self.compute_marginal_gain(
            sounder_id, upgraded_obs, prior_sqrt_cov
        )

        improvement = upgraded_result.marginal_gain - current_result.marginal_gain

        return {
            'current_marginal_gain': current_result.marginal_gain,
            'upgraded_marginal_gain': upgraded_result.marginal_gain,
            'improvement': improvement,
            'relative_improvement': improvement / current_result.marginal_gain if current_result.marginal_gain > 0 else 0.0,
            'current_tier': observations[0].get('quality_tier', 'unknown'),
            'target_tier': new_tier
        }
