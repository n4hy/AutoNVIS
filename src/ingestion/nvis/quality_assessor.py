"""
Quality Assessment Engine for NVIS Sounder Observations

Implements six-dimensional quality scoring and tier assignment
to appropriately weight observations in the SR-UKF filter.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional
import numpy as np
from .protocol_adapters.base_adapter import NVISMeasurement, SounderMetadata
from ...common.config import NVISIngestionConfig
from ...common.logging_config import ServiceLogger


class QualityTier(Enum):
    """Quality tier classification"""
    PLATINUM = "platinum"  # ≥0.80 score
    GOLD = "gold"          # ≥0.60 score
    SILVER = "silver"      # ≥0.40 score
    BRONZE = "bronze"      # <0.40 score


@dataclass
class QualityMetrics:
    """
    Six-dimensional quality assessment for NVIS measurements

    Each component ranges from 0.0 (worst) to 1.0 (best)
    """
    signal_quality: float = 0.5       # SNR-based
    calibration_quality: float = 0.5  # Known calibration status
    temporal_quality: float = 0.5     # Rate and consistency
    spatial_quality: float = 0.5      # Coverage gap filling
    equipment_quality: float = 0.5    # Professional vs amateur tier
    historical_quality: float = 0.5   # Learned from NIS statistics

    def overall_score(self, weights: Dict[str, float]) -> float:
        """
        Compute weighted overall quality score

        Args:
            weights: Dict with keys matching metric names

        Returns:
            Weighted score in [0.0, 1.0]
        """
        score = (
            self.signal_quality * weights.get('signal_quality', 0.25) +
            self.calibration_quality * weights.get('calibration_quality', 0.20) +
            self.temporal_quality * weights.get('temporal_quality', 0.15) +
            self.spatial_quality * weights.get('spatial_quality', 0.15) +
            self.equipment_quality * weights.get('equipment_quality', 0.15) +
            self.historical_quality * weights.get('historical_quality', 0.10)
        )
        return np.clip(score, 0.0, 1.0)


class QualityAssessor:
    """
    Assesses quality of NVIS measurements and assigns tiers

    Uses six quality dimensions:
    1. Signal quality (SNR-based)
    2. Calibration quality (equipment calibration status)
    3. Temporal quality (observation rate and consistency)
    4. Spatial quality (fills coverage gaps)
    5. Equipment quality (professional vs amateur)
    6. Historical quality (learned performance)
    """

    def __init__(self, config: NVISIngestionConfig):
        """
        Initialize quality assessor

        Args:
            config: NVIS ingestion configuration
        """
        self.config = config
        self.logger = ServiceLogger("quality_assessor")

        # Quality weights
        self.weights = {
            'signal_quality': config.weight_signal_quality,
            'calibration_quality': config.weight_calibration_quality,
            'temporal_quality': config.weight_temporal_quality,
            'spatial_quality': config.weight_spatial_quality,
            'equipment_quality': config.weight_equipment_quality,
            'historical_quality': config.weight_historical_quality
        }

        # Historical quality tracking
        self.historical_quality = {}  # sounder_id → quality score

        # Spatial coverage tracking (for spatial quality)
        self.recent_observations = []  # Recent (lat, lon) observations

    def assess_measurement(
        self,
        measurement: NVISMeasurement,
        sounder_metadata: Optional[SounderMetadata] = None
    ) -> QualityMetrics:
        """
        Assess quality of a single measurement

        Args:
            measurement: NVIS measurement to assess
            sounder_metadata: Optional sounder metadata

        Returns:
            QualityMetrics with six component scores
        """
        metrics = QualityMetrics()

        # 1. Signal quality (SNR-based)
        metrics.signal_quality = self._assess_signal_quality(measurement)

        # 2. Calibration quality
        metrics.calibration_quality = self._assess_calibration_quality(
            sounder_metadata
        )

        # 3. Temporal quality
        metrics.temporal_quality = self._assess_temporal_quality(
            measurement.sounder_id,
            sounder_metadata
        )

        # 4. Spatial quality (coverage gap filling)
        metrics.spatial_quality = self._assess_spatial_quality(measurement)

        # 5. Equipment quality
        metrics.equipment_quality = self._assess_equipment_quality(
            sounder_metadata
        )

        # 6. Historical quality (learned)
        metrics.historical_quality = self._get_historical_quality(
            measurement.sounder_id
        )

        return metrics

    def _assess_signal_quality(self, measurement: NVISMeasurement) -> float:
        """
        Assess signal quality based on SNR

        High SNR → high quality
        Low SNR → low quality

        Args:
            measurement: NVIS measurement

        Returns:
            Signal quality score [0.0, 1.0]
        """
        snr = measurement.snr

        # SNR thresholds (dB)
        SNR_EXCELLENT = 30.0
        SNR_POOR = 5.0

        if snr >= SNR_EXCELLENT:
            return 1.0
        elif snr <= SNR_POOR:
            return 0.1
        else:
            # Linear interpolation
            return 0.1 + 0.9 * (snr - SNR_POOR) / (SNR_EXCELLENT - SNR_POOR)

    def _assess_calibration_quality(
        self,
        metadata: Optional[SounderMetadata]
    ) -> float:
        """
        Assess calibration quality from sounder metadata

        Args:
            metadata: Sounder metadata

        Returns:
            Calibration quality score [0.0, 1.0]
        """
        if metadata is None:
            return 0.5  # Unknown

        status = metadata.calibration_status.lower()

        if status == 'calibrated':
            return 1.0
        elif status == 'uncalibrated':
            return 0.3
        else:
            return 0.5  # Unknown

    def _assess_temporal_quality(
        self,
        sounder_id: str,
        metadata: Optional[SounderMetadata]
    ) -> float:
        """
        Assess temporal quality based on observation rate

        High rate → better temporal coverage → higher quality

        Args:
            sounder_id: Sounder identifier
            metadata: Sounder metadata

        Returns:
            Temporal quality score [0.0, 1.0]
        """
        if metadata is None or metadata.data_rate is None:
            return 0.5  # Unknown

        data_rate = metadata.data_rate  # observations per hour

        # Rate thresholds
        RATE_HIGH = 100.0  # High-rate professional
        RATE_LOW = 1.0     # Low-rate amateur

        if data_rate >= RATE_HIGH:
            return 1.0
        elif data_rate <= RATE_LOW:
            return 0.3
        else:
            # Logarithmic scaling (favor high rates)
            return 0.3 + 0.7 * np.log10(data_rate / RATE_LOW) / np.log10(RATE_HIGH / RATE_LOW)

    def _assess_spatial_quality(self, measurement: NVISMeasurement) -> float:
        """
        Assess spatial quality based on coverage gap filling

        Measurements in sparse regions → higher spatial quality
        Measurements in dense regions → lower spatial quality

        Args:
            measurement: NVIS measurement

        Returns:
            Spatial quality score [0.0, 1.0]
        """
        # Compute midpoint location
        mid_lat = (measurement.tx_latitude + measurement.rx_latitude) / 2
        mid_lon = (measurement.tx_longitude + measurement.rx_longitude) / 2

        # Check distance to nearest recent observation
        min_distance = float('inf')
        for lat, lon in self.recent_observations[-100:]:  # Last 100 obs
            distance = self._haversine_distance(mid_lat, mid_lon, lat, lon)
            min_distance = min(min_distance, distance)

        # Update recent observations
        self.recent_observations.append((mid_lat, mid_lon))
        if len(self.recent_observations) > 1000:
            self.recent_observations = self.recent_observations[-1000:]

        # Distance thresholds (km)
        DIST_SPARSE = 500.0   # Very sparse (high quality)
        DIST_DENSE = 50.0     # Very dense (low quality)

        if min_distance == float('inf') or min_distance >= DIST_SPARSE:
            return 1.0
        elif min_distance <= DIST_DENSE:
            return 0.3
        else:
            # Linear interpolation
            return 0.3 + 0.7 * (min_distance - DIST_DENSE) / (DIST_SPARSE - DIST_DENSE)

    def _assess_equipment_quality(
        self,
        metadata: Optional[SounderMetadata]
    ) -> float:
        """
        Assess equipment quality from sounder metadata

        Args:
            metadata: Sounder metadata

        Returns:
            Equipment quality score [0.0, 1.0]
        """
        if metadata is None:
            return 0.5  # Unknown

        equipment_type = metadata.equipment_type.lower()

        if equipment_type == 'professional':
            return 1.0
        elif equipment_type == 'research':
            return 0.8
        elif equipment_type == 'amateur':
            return 0.4
        else:
            return 0.5  # Unknown

    def _get_historical_quality(self, sounder_id: str) -> float:
        """
        Get learned historical quality for sounder

        Args:
            sounder_id: Sounder identifier

        Returns:
            Historical quality score [0.0, 1.0]
        """
        return self.historical_quality.get(sounder_id, 0.5)

    def update_historical_quality(
        self,
        sounder_id: str,
        innovation: float,
        predicted_std: float
    ):
        """
        Update historical quality based on innovation statistics

        Uses Normalized Innovation Squared (NIS):
        - NIS << 1: overestimating error → increase quality
        - NIS >> 1: underestimating error → decrease quality

        Args:
            sounder_id: Sounder identifier
            innovation: Observation innovation (obs - prediction)
            predicted_std: Predicted standard deviation
        """
        # Compute NIS
        nis = (innovation / predicted_std) ** 2
        expected_nis = 1.0

        # Adjustment based on NIS
        if nis < expected_nis * 0.5:
            adjustment = +0.01  # Increase quality
        elif nis > expected_nis * 2.0:
            adjustment = -0.01  # Decrease quality
        else:
            adjustment = 0.0

        # Update with exponential smoothing and clipping
        current_quality = self.historical_quality.get(sounder_id, 0.5)
        new_quality = np.clip(current_quality + adjustment, 0.0, 1.0)
        self.historical_quality[sounder_id] = new_quality

        self.logger.debug(
            f"Updated historical quality for {sounder_id}: "
            f"{current_quality:.3f} → {new_quality:.3f} (NIS={nis:.2f})"
        )

    def assign_tier(self, metrics: QualityMetrics) -> QualityTier:
        """
        Assign quality tier based on overall score

        Args:
            metrics: Quality metrics

        Returns:
            QualityTier (PLATINUM, GOLD, SILVER, BRONZE)
        """
        score = metrics.overall_score(self.weights)

        if score >= self.config.platinum.min_score:
            return QualityTier.PLATINUM
        elif score >= self.config.gold.min_score:
            return QualityTier.GOLD
        elif score >= self.config.silver.min_score:
            return QualityTier.SILVER
        else:
            return QualityTier.BRONZE

    def map_to_error_covariance(
        self,
        tier: QualityTier
    ) -> Dict[str, float]:
        """
        Map quality tier to observation error covariance

        Args:
            tier: Quality tier

        Returns:
            Dict with 'signal_error_db' and 'delay_error_ms'
        """
        if tier == QualityTier.PLATINUM:
            config = self.config.platinum
        elif tier == QualityTier.GOLD:
            config = self.config.gold
        elif tier == QualityTier.SILVER:
            config = self.config.silver
        else:  # BRONZE
            config = self.config.bronze

        return {
            'signal_error_db': config.signal_error_db,
            'delay_error_ms': config.delay_error_ms
        }

    def _haversine_distance(
        self,
        lat1: float, lon1: float,
        lat2: float, lon2: float
    ) -> float:
        """
        Calculate great circle distance between two points

        Args:
            lat1, lon1: First point (degrees)
            lat2, lon2: Second point (degrees)

        Returns:
            Distance in kilometers
        """
        import math

        R = 6371.0  # Earth radius in km

        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlon = math.radians(lon2 - lon1)
        dlat = math.radians(lat2 - lat1)

        a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c
