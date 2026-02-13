"""
Unit Tests for NVIS Quality Assessment

Tests the six-dimensional quality scoring and tier assignment.
"""

import pytest
import numpy as np
from src.ingestion.nvis.quality_assessor import (
    QualityMetrics,
    QualityTier,
    QualityAssessor
)
from src.ingestion.nvis.protocol_adapters.base_adapter import (
    NVISMeasurement,
    SounderMetadata
)
from src.common.config import NVISIngestionConfig


@pytest.fixture
def config():
    """Create default NVIS ingestion config"""
    return NVISIngestionConfig()


@pytest.fixture
def assessor(config):
    """Create quality assessor"""
    return QualityAssessor(config)


@pytest.fixture
def sample_measurement():
    """Create sample NVIS measurement"""
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
        signal_strength=-80.0,
        group_delay=2.5,
        snr=20.0,
        sounder_id="TEST_001",
        timestamp="2025-01-15T12:00:00Z",
        is_o_mode=True
    )


class TestQualityMetrics:
    """Test QualityMetrics dataclass"""

    def test_overall_score_default_weights(self):
        """Test overall score with default weights"""
        metrics = QualityMetrics(
            signal_quality=1.0,
            calibration_quality=1.0,
            temporal_quality=1.0,
            spatial_quality=1.0,
            equipment_quality=1.0,
            historical_quality=1.0
        )

        weights = {
            'signal_quality': 0.25,
            'calibration_quality': 0.20,
            'temporal_quality': 0.15,
            'spatial_quality': 0.15,
            'equipment_quality': 0.15,
            'historical_quality': 0.10
        }

        score = metrics.overall_score(weights)
        assert score == pytest.approx(1.0)

    def test_overall_score_mixed_quality(self):
        """Test overall score with mixed quality components"""
        metrics = QualityMetrics(
            signal_quality=1.0,    # Excellent
            calibration_quality=0.5,  # Unknown
            temporal_quality=0.3,  # Poor
            spatial_quality=1.0,   # Excellent
            equipment_quality=0.8,  # Good
            historical_quality=0.5  # Average
        )

        weights = {
            'signal_quality': 0.25,
            'calibration_quality': 0.20,
            'temporal_quality': 0.15,
            'spatial_quality': 0.15,
            'equipment_quality': 0.15,
            'historical_quality': 0.10
        }

        score = metrics.overall_score(weights)
        expected = (1.0 * 0.25 + 0.5 * 0.20 + 0.3 * 0.15 +
                   1.0 * 0.15 + 0.8 * 0.15 + 0.5 * 0.10)
        assert score == pytest.approx(expected)


class TestSignalQuality:
    """Test signal quality assessment"""

    def test_excellent_snr(self, assessor, sample_measurement):
        """Test excellent SNR (>30 dB)"""
        sample_measurement.snr = 35.0
        quality = assessor._assess_signal_quality(sample_measurement)
        assert quality == pytest.approx(1.0)

    def test_poor_snr(self, assessor, sample_measurement):
        """Test poor SNR (<5 dB)"""
        sample_measurement.snr = 3.0
        quality = assessor._assess_signal_quality(sample_measurement)
        assert quality == pytest.approx(0.1)

    def test_medium_snr(self, assessor, sample_measurement):
        """Test medium SNR (15 dB)"""
        sample_measurement.snr = 15.0
        quality = assessor._assess_signal_quality(sample_measurement)
        # Should be interpolated between 0.1 and 1.0
        assert 0.1 < quality < 1.0


class TestCalibrationQuality:
    """Test calibration quality assessment"""

    def test_calibrated_sounder(self, assessor):
        """Test calibrated sounder"""
        metadata = SounderMetadata(
            sounder_id="TEST",
            name="Test",
            operator="Test",
            location="Test",
            latitude=40.0,
            longitude=-105.0,
            altitude=1500.0,
            equipment_type="professional",
            calibration_status="calibrated"
        )
        quality = assessor._assess_calibration_quality(metadata)
        assert quality == pytest.approx(1.0)

    def test_uncalibrated_sounder(self, assessor):
        """Test uncalibrated sounder"""
        metadata = SounderMetadata(
            sounder_id="TEST",
            name="Test",
            operator="Test",
            location="Test",
            latitude=40.0,
            longitude=-105.0,
            altitude=1500.0,
            equipment_type="amateur",
            calibration_status="uncalibrated"
        )
        quality = assessor._assess_calibration_quality(metadata)
        assert quality == pytest.approx(0.3)

    def test_unknown_calibration(self, assessor):
        """Test unknown calibration status"""
        quality = assessor._assess_calibration_quality(None)
        assert quality == pytest.approx(0.5)


class TestTemporalQuality:
    """Test temporal quality assessment"""

    def test_high_rate_sounder(self, assessor):
        """Test high-rate sounder (>100 obs/hr)"""
        metadata = SounderMetadata(
            sounder_id="TEST",
            name="Test",
            operator="Test",
            location="Test",
            latitude=40.0,
            longitude=-105.0,
            altitude=1500.0,
            equipment_type="professional",
            calibration_status="calibrated",
            data_rate=200.0
        )
        quality = assessor._assess_temporal_quality("TEST", metadata)
        assert quality == pytest.approx(1.0)

    def test_low_rate_sounder(self, assessor):
        """Test low-rate sounder (<1 obs/hr)"""
        metadata = SounderMetadata(
            sounder_id="TEST",
            name="Test",
            operator="Test",
            location="Test",
            latitude=40.0,
            longitude=-105.0,
            altitude=1500.0,
            equipment_type="amateur",
            calibration_status="uncalibrated",
            data_rate=0.5
        )
        quality = assessor._assess_temporal_quality("TEST", metadata)
        assert quality == pytest.approx(0.3)


class TestEquipmentQuality:
    """Test equipment quality assessment"""

    def test_professional_equipment(self, assessor):
        """Test professional equipment"""
        metadata = SounderMetadata(
            sounder_id="TEST",
            name="Test",
            operator="Test",
            location="Test",
            latitude=40.0,
            longitude=-105.0,
            altitude=1500.0,
            equipment_type="professional",
            calibration_status="calibrated"
        )
        quality = assessor._assess_equipment_quality(metadata)
        assert quality == pytest.approx(1.0)

    def test_amateur_equipment(self, assessor):
        """Test amateur equipment"""
        metadata = SounderMetadata(
            sounder_id="TEST",
            name="Test",
            operator="Test",
            location="Test",
            latitude=40.0,
            longitude=-105.0,
            altitude=1500.0,
            equipment_type="amateur",
            calibration_status="uncalibrated"
        )
        quality = assessor._assess_equipment_quality(metadata)
        assert quality == pytest.approx(0.4)


class TestTierAssignment:
    """Test quality tier assignment"""

    def test_platinum_tier(self, assessor):
        """Test PLATINUM tier assignment (score ≥0.80)"""
        metrics = QualityMetrics(
            signal_quality=1.0,
            calibration_quality=1.0,
            temporal_quality=1.0,
            spatial_quality=1.0,
            equipment_quality=1.0,
            historical_quality=1.0
        )
        tier = assessor.assign_tier(metrics)
        assert tier == QualityTier.PLATINUM

    def test_gold_tier(self, assessor):
        """Test GOLD tier assignment (0.60 ≤ score < 0.80)"""
        metrics = QualityMetrics(
            signal_quality=0.7,
            calibration_quality=0.7,
            temporal_quality=0.7,
            spatial_quality=0.7,
            equipment_quality=0.7,
            historical_quality=0.7
        )
        tier = assessor.assign_tier(metrics)
        assert tier == QualityTier.GOLD

    def test_silver_tier(self, assessor):
        """Test SILVER tier assignment (0.40 ≤ score < 0.60)"""
        metrics = QualityMetrics(
            signal_quality=0.5,
            calibration_quality=0.5,
            temporal_quality=0.5,
            spatial_quality=0.5,
            equipment_quality=0.5,
            historical_quality=0.5
        )
        tier = assessor.assign_tier(metrics)
        assert tier == QualityTier.SILVER

    def test_bronze_tier(self, assessor):
        """Test BRONZE tier assignment (score < 0.40)"""
        metrics = QualityMetrics(
            signal_quality=0.3,
            calibration_quality=0.3,
            temporal_quality=0.3,
            spatial_quality=0.3,
            equipment_quality=0.3,
            historical_quality=0.3
        )
        tier = assessor.assign_tier(metrics)
        assert tier == QualityTier.BRONZE


class TestErrorCovariance:
    """Test error covariance mapping"""

    def test_platinum_errors(self, assessor):
        """Test PLATINUM error covariance"""
        errors = assessor.map_to_error_covariance(QualityTier.PLATINUM)
        assert errors['signal_error_db'] == pytest.approx(2.0)
        assert errors['delay_error_ms'] == pytest.approx(0.1)

    def test_gold_errors(self, assessor):
        """Test GOLD error covariance"""
        errors = assessor.map_to_error_covariance(QualityTier.GOLD)
        assert errors['signal_error_db'] == pytest.approx(4.0)
        assert errors['delay_error_ms'] == pytest.approx(0.5)

    def test_silver_errors(self, assessor):
        """Test SILVER error covariance"""
        errors = assessor.map_to_error_covariance(QualityTier.SILVER)
        assert errors['signal_error_db'] == pytest.approx(8.0)
        assert errors['delay_error_ms'] == pytest.approx(2.0)

    def test_bronze_errors(self, assessor):
        """Test BRONZE error covariance"""
        errors = assessor.map_to_error_covariance(QualityTier.BRONZE)
        assert errors['signal_error_db'] == pytest.approx(15.0)
        assert errors['delay_error_ms'] == pytest.approx(5.0)


class TestHistoricalQuality:
    """Test historical quality learning"""

    def test_quality_increase(self, assessor):
        """Test quality increase when NIS < 0.5"""
        assessor.historical_quality['TEST'] = 0.5
        assessor.update_historical_quality('TEST', innovation=1.0, predicted_std=2.0)
        # NIS = (1.0/2.0)^2 = 0.25 < 0.5 → increase quality
        assert assessor.historical_quality['TEST'] > 0.5

    def test_quality_decrease(self, assessor):
        """Test quality decrease when NIS > 2.0"""
        assessor.historical_quality['TEST'] = 0.5
        assessor.update_historical_quality('TEST', innovation=4.0, predicted_std=2.0)
        # NIS = (4.0/2.0)^2 = 4.0 > 2.0 → decrease quality
        assert assessor.historical_quality['TEST'] < 0.5

    def test_quality_stable(self, assessor):
        """Test quality stable when 0.5 < NIS < 2.0"""
        assessor.historical_quality['TEST'] = 0.5
        assessor.update_historical_quality('TEST', innovation=2.0, predicted_std=2.0)
        # NIS = (2.0/2.0)^2 = 1.0 → no change
        assert assessor.historical_quality['TEST'] == pytest.approx(0.5)

    def test_quality_clipping(self, assessor):
        """Test quality is clipped to [0.0, 1.0]"""
        assessor.historical_quality['TEST'] = 0.99
        for _ in range(10):
            assessor.update_historical_quality('TEST', innovation=1.0, predicted_std=2.0)
        assert assessor.historical_quality['TEST'] <= 1.0

        assessor.historical_quality['TEST'] = 0.01
        for _ in range(10):
            assessor.update_historical_quality('TEST', innovation=4.0, predicted_std=2.0)
        assert assessor.historical_quality['TEST'] >= 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
