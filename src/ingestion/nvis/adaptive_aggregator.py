"""
Adaptive Aggregation for NVIS Sounder Observations

Prevents data flooding from high-rate sounders while preserving
information through quality-weighted temporal averaging.
"""

import time
import numpy as np
from collections import defaultdict
from dataclasses import replace
from typing import List, Dict, Tuple, Optional
from .protocol_adapters.base_adapter import NVISMeasurement
from .quality_assessor import QualityTier
from ...common.config import NVISIngestionConfig
from ...common.logging_config import ServiceLogger


class AdaptiveAggregator:
    """
    Adaptive temporal aggregation for NVIS observations

    High-rate sounders (>threshold): Buffer and aggregate within time windows
    Low-rate sounders (<threshold): Pass through immediately

    Aggregation uses quality-weighted averaging to preserve information
    from high-quality observations.
    """

    def __init__(self, config: NVISIngestionConfig):
        """
        Initialize adaptive aggregator

        Args:
            config: NVIS ingestion configuration
        """
        self.config = config
        self.logger = ServiceLogger("adaptive_aggregator")

        # Time window for aggregation (seconds)
        self.window_sec = config.window_seconds

        # Rate threshold for triggering aggregation (obs/hour)
        self.rate_threshold = config.rate_threshold

        # Buffers for aggregation: sounder_id → list of (measurement, quality_score, timestamp)
        self.buffers: Dict[str, List[Tuple[NVISMeasurement, float, float]]] = defaultdict(list)

        # Rate estimates: sounder_id → obs/hour
        self.rate_estimates: Dict[str, float] = {}

        # Observation timestamps for rate estimation
        self.obs_timestamps: Dict[str, List[float]] = defaultdict(list)

        # Last bin flush time: sounder_id → timestamp
        self.last_flush: Dict[str, float] = {}

    def add_measurement(
        self,
        sounder_id: str,
        measurement: NVISMeasurement,
        quality_score: float
    ) -> Optional[NVISMeasurement]:
        """
        Add measurement; aggregate or pass-through based on rate

        Args:
            sounder_id: Sounder identifier
            measurement: NVIS measurement
            quality_score: Overall quality score [0.0, 1.0]

        Returns:
            Aggregated measurement if ready, None if buffered
        """
        current_time = time.time()

        # Update rate estimate
        rate = self._update_rate_estimate(sounder_id, current_time)

        if rate > self.rate_threshold:
            # High-rate sounder: buffer for aggregation
            self.buffers[sounder_id].append((measurement, quality_score, current_time))
            self.logger.debug(
                f"Buffered measurement from {sounder_id} "
                f"(rate={rate:.1f} obs/hr, buffer_size={len(self.buffers[sounder_id])})"
            )

            # Check if window is complete
            if self._should_flush_bin(sounder_id, current_time):
                return self._aggregate_bin(sounder_id)
            else:
                return None

        else:
            # Low-rate sounder: pass through immediately
            self.logger.debug(
                f"Pass-through measurement from {sounder_id} "
                f"(rate={rate:.1f} obs/hr)"
            )
            return measurement

    def _update_rate_estimate(self, sounder_id: str, current_time: float) -> float:
        """
        Update rate estimate for sounder using sliding window

        Args:
            sounder_id: Sounder identifier
            current_time: Current timestamp

        Returns:
            Estimated rate in observations per hour
        """
        # Add current observation timestamp
        self.obs_timestamps[sounder_id].append(current_time)

        # Keep only last hour of timestamps
        cutoff_time = current_time - 3600.0  # 1 hour ago
        self.obs_timestamps[sounder_id] = [
            t for t in self.obs_timestamps[sounder_id] if t >= cutoff_time
        ]

        # Estimate rate
        count = len(self.obs_timestamps[sounder_id])
        if count == 0:
            rate = 0.0
        else:
            time_span = current_time - self.obs_timestamps[sounder_id][0]
            if time_span > 0:
                rate = count / (time_span / 3600.0)
            else:
                rate = count  # Assume 1 second span → count * 3600

        self.rate_estimates[sounder_id] = rate
        return rate

    def _should_flush_bin(self, sounder_id: str, current_time: float) -> bool:
        """
        Check if aggregation window is complete

        Args:
            sounder_id: Sounder identifier
            current_time: Current timestamp

        Returns:
            True if bin should be flushed
        """
        last_flush_time = self.last_flush.get(sounder_id, current_time)
        time_since_flush = current_time - last_flush_time

        return time_since_flush >= self.window_sec

    def _aggregate_bin(self, sounder_id: str) -> Optional[NVISMeasurement]:
        """
        Aggregate buffered measurements using quality-weighted averaging

        Args:
            sounder_id: Sounder identifier

        Returns:
            Aggregated measurement, or None if buffer empty
        """
        measurements = self.buffers[sounder_id]

        if not measurements:
            return None

        self.logger.debug(
            f"Aggregating {len(measurements)} measurements from {sounder_id}"
        )

        # Extract components
        meas_list = [m for m, q, t in measurements]
        quality_list = np.array([q for m, q, t in measurements])

        # Normalize quality weights
        total_weight = quality_list.sum()
        if total_weight == 0:
            weights = np.ones_like(quality_list) / len(quality_list)
        else:
            weights = quality_list / total_weight

        # Quality-weighted averaging of observables
        signal_avg = sum(
            m.signal_strength * w for m, w in zip(meas_list, weights)
        )
        delay_avg = sum(
            m.group_delay * w for m, w in zip(meas_list, weights)
        )
        snr_avg = sum(
            m.snr * w for m, w in zip(meas_list, weights)
        )

        # Compute variability within bin (captures uncertainty)
        signal_std = np.std([m.signal_strength for m in meas_list])
        delay_std = np.std([m.group_delay for m in meas_list])

        # Select best measurement as template
        best_idx = np.argmax(quality_list)
        best_meas = meas_list[best_idx]

        # Create aggregated measurement
        aggregated = replace(
            best_meas,
            signal_strength=signal_avg,
            group_delay=delay_avg,
            snr=snr_avg,
            # Error includes both tier error and variability
            signal_strength_error=max(signal_std, best_meas.signal_strength_error),
            group_delay_error=max(delay_std, best_meas.group_delay_error)
        )

        # Clear buffer and update flush time
        self.buffers[sounder_id] = []
        self.last_flush[sounder_id] = time.time()

        self.logger.info(
            f"Aggregated {len(measurements)} obs from {sounder_id}: "
            f"signal={signal_avg:.1f}±{signal_std:.1f} dBm, "
            f"delay={delay_avg:.2f}±{delay_std:.2f} ms"
        )

        return aggregated

    def flush_all_bins(self) -> List[NVISMeasurement]:
        """
        Force flush all pending bins (for end-of-cycle processing)

        Returns:
            List of aggregated measurements
        """
        aggregated_measurements = []

        for sounder_id in list(self.buffers.keys()):
            if self.buffers[sounder_id]:
                aggregated = self._aggregate_bin(sounder_id)
                if aggregated:
                    aggregated_measurements.append(aggregated)

        return aggregated_measurements


def apply_rate_limiting(
    observations: List[Tuple[NVISMeasurement, QualityTier]],
    config: NVISIngestionConfig
) -> List[Tuple[NVISMeasurement, QualityTier]]:
    """
    Apply rate limiting per sounder per cycle

    Limits number of observations based on quality tier:
    - PLATINUM: up to 50 obs per cycle
    - GOLD: up to 30 obs per cycle
    - SILVER: up to 15 obs per cycle
    - BRONZE: up to 5 obs per cycle

    Args:
        observations: List of (measurement, tier) tuples
        config: NVIS ingestion configuration

    Returns:
        Rate-limited list of observations
    """
    # Rate limits by tier
    max_obs = {
        QualityTier.PLATINUM: config.max_obs_per_cycle_platinum,
        QualityTier.GOLD: config.max_obs_per_cycle_gold,
        QualityTier.SILVER: config.max_obs_per_cycle_silver,
        QualityTier.BRONZE: config.max_obs_per_cycle_bronze
    }

    # Group by sounder_id
    sounder_groups = defaultdict(list)
    for meas, tier in observations:
        sounder_groups[meas.sounder_id].append((meas, tier))

    limited = []
    for sounder_id, obs_list in sounder_groups.items():
        # Determine tier (use highest tier for this sounder)
        tiers = [tier for meas, tier in obs_list]
        best_tier = min(tiers, key=lambda t: list(QualityTier).index(t))

        # Apply limit
        limit = max_obs[best_tier]

        # Sort by quality (need to get quality score - use tier as proxy)
        # Higher tier = better quality
        tier_order = {
            QualityTier.PLATINUM: 3,
            QualityTier.GOLD: 2,
            QualityTier.SILVER: 1,
            QualityTier.BRONZE: 0
        }
        obs_list_sorted = sorted(
            obs_list,
            key=lambda x: tier_order[x[1]],
            reverse=True
        )

        # Take top N observations
        limited.extend(obs_list_sorted[:limit])

    logger = ServiceLogger("rate_limiter")
    logger.info(
        f"Rate limiting: {len(observations)} → {len(limited)} observations "
        f"({len(observations) - len(limited)} dropped)"
    )

    return limited
