"""
Tests for Historical Validation Framework

Tests the event replayer and validation metrics components.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import yaml

import sys
sys.path.insert(0, str(Path(__file__).parent))

from event_replayer import (
    EventReplayer,
    EventConfig,
    ExpectedBehavior,
    Observation,
    ObservationType,
    FilterSnapshot,
    ReplayResult,
    create_sample_event_config
)

from validation_metrics import (
    compute_statistical_metrics,
    compute_fof2_metrics,
    compute_hmf2_metrics,
    compute_tec_metrics,
    compute_response_time,
    compute_mode_detection_metrics,
    check_thresholds,
    generate_validation_report,
    StatisticalMetrics,
    IonoMetrics,
    TimeSeriesComparison,
    ValidationRunner
)


class TestStatisticalMetrics:
    """Test statistical metrics computation"""

    def test_rmse_zero_error(self):
        """Test RMSE is zero for perfect predictions"""
        truth = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        estimates = truth.copy()

        metrics = compute_statistical_metrics(estimates, truth)

        assert metrics.rmse == pytest.approx(0.0, abs=1e-10)
        assert metrics.mae == pytest.approx(0.0, abs=1e-10)
        assert metrics.bias == pytest.approx(0.0, abs=1e-10)

    def test_rmse_calculation(self):
        """Test RMSE calculation"""
        truth = np.array([1.0, 2.0, 3.0])
        estimates = np.array([2.0, 3.0, 4.0])  # All errors = 1.0

        metrics = compute_statistical_metrics(estimates, truth)

        assert metrics.rmse == pytest.approx(1.0, abs=1e-10)
        assert metrics.mae == pytest.approx(1.0, abs=1e-10)
        assert metrics.bias == pytest.approx(1.0, abs=1e-10)

    def test_bias_positive_negative(self):
        """Test bias with both positive and negative errors"""
        truth = np.array([0.0, 0.0])
        estimates = np.array([1.0, -1.0])

        metrics = compute_statistical_metrics(estimates, truth)

        assert metrics.bias == pytest.approx(0.0, abs=1e-10)
        assert metrics.mae == pytest.approx(1.0, abs=1e-10)

    def test_correlation_perfect(self):
        """Test perfect correlation"""
        truth = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        estimates = truth * 2  # Perfect linear relationship

        metrics = compute_statistical_metrics(estimates, truth)

        assert metrics.correlation == pytest.approx(1.0, abs=1e-10)

    def test_correlation_negative(self):
        """Test negative correlation"""
        truth = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        estimates = 6 - truth  # Perfect negative correlation

        metrics = compute_statistical_metrics(estimates, truth)

        assert metrics.correlation == pytest.approx(-1.0, abs=1e-10)

    def test_handles_nan(self):
        """Test handling of NaN values"""
        truth = np.array([1.0, np.nan, 3.0, 4.0])
        estimates = np.array([1.0, 2.0, np.nan, 4.0])

        metrics = compute_statistical_metrics(estimates, truth)

        # Should only use valid pairs: (1,1) and (4,4)
        assert metrics.n_samples == 2
        assert metrics.rmse == pytest.approx(0.0, abs=1e-10)

    def test_percentiles(self):
        """Test percentile calculations"""
        truth = np.zeros(100)
        estimates = np.linspace(0, 10, 100)  # Errors from 0 to 10

        metrics = compute_statistical_metrics(estimates, truth)

        assert metrics.percentile_90 == pytest.approx(9.0, abs=0.5)
        assert metrics.percentile_95 == pytest.approx(9.5, abs=0.5)

    def test_empty_arrays(self):
        """Test with empty arrays"""
        metrics = compute_statistical_metrics(np.array([]), np.array([]))

        assert metrics.n_samples == 0
        assert metrics.rmse == 0.0


class TestIonoMetrics:
    """Test ionospheric-specific metrics"""

    def test_fof2_metrics(self):
        """Test foF2 metrics computation"""
        np.random.seed(42)
        truth = 8.0 + np.random.normal(0, 0.5, 100)
        estimates = truth + np.random.normal(0, 0.3, 100)

        metrics = compute_fof2_metrics(estimates, truth)

        # Should have reasonable RMSE for ionospheric data
        assert 0 < metrics.rmse < 1.0
        assert abs(metrics.bias) < 0.5
        assert metrics.correlation > 0.5

    def test_hmf2_metrics(self):
        """Test hmF2 metrics computation"""
        np.random.seed(42)
        truth = 280 + np.random.normal(0, 20, 100)
        estimates = truth + np.random.normal(0, 15, 100)

        metrics = compute_hmf2_metrics(estimates, truth)

        assert 0 < metrics.rmse < 50
        assert abs(metrics.bias) < 20

    def test_tec_metrics(self):
        """Test TEC metrics computation"""
        np.random.seed(42)
        truth = 25 + np.random.normal(0, 5, 100)
        estimates = truth + np.random.normal(0, 2, 100)

        metrics = compute_tec_metrics(estimates, truth)

        assert 0 < metrics.rmse < 10
        assert abs(metrics.bias) < 5


class TestResponseTimeMetrics:
    """Test response time metrics computation"""

    def test_detect_step_response(self):
        """Test detection of step response"""
        n_points = 100
        event_idx = 20

        # Create step function
        estimates = np.zeros(n_points)
        estimates[event_idx:] = 5.0

        times = [datetime(2026, 1, 1, 12, 0) + timedelta(minutes=i) for i in range(n_points)]
        event_time = times[event_idx]
        baseline = 0.0

        metrics = compute_response_time(times, estimates.tolist(), event_time, baseline)

        assert metrics.detection_delay_minutes is not None
        assert metrics.detection_delay_minutes >= 0

    def test_recovery_time(self):
        """Test recovery time calculation"""
        n_points = 100

        # Create pulse: baseline -> peak -> baseline
        estimates = np.zeros(n_points)
        estimates[20:40] = 5.0  # Pulse from t=20 to t=40

        times = [datetime(2026, 1, 1, 12, 0) + timedelta(minutes=i) for i in range(n_points)]
        event_time = times[20]
        baseline = 0.0

        metrics = compute_response_time(times, estimates.tolist(), event_time, baseline)

        assert metrics.recovery_time_minutes is not None

    def test_no_change_detected(self):
        """Test when no significant change occurs"""
        n_points = 50
        estimates = [10.0 + np.random.normal(0, 0.01) for _ in range(n_points)]
        times = [datetime(2026, 1, 1, 12, 0) + timedelta(minutes=i) for i in range(n_points)]
        event_time = times[10]
        baseline = 10.0

        metrics = compute_response_time(times, estimates, event_time, baseline)

        # No significant change should be detected
        assert metrics.detection_delay_minutes is None or metrics.response_time_minutes is None


class TestModeDetectionMetrics:
    """Test mode detection metrics"""

    def test_perfect_detection(self):
        """Test perfect mode detection"""
        expected = [
            (datetime(2026, 1, 1, 12, 0), "FLARE"),
            (datetime(2026, 1, 1, 13, 0), "QUIET")
        ]
        detected = [
            (datetime(2026, 1, 1, 12, 1), "FLARE"),  # 1 min delay
            (datetime(2026, 1, 1, 13, 2), "QUIET")   # 2 min delay
        ]

        metrics = compute_mode_detection_metrics(detected, expected, tolerance_minutes=5)

        assert metrics.true_positives == 2
        assert metrics.false_positives == 0
        assert metrics.false_negatives == 0
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1_score == 1.0

    def test_missed_detection(self):
        """Test missed mode switch detection"""
        expected = [
            (datetime(2026, 1, 1, 12, 0), "FLARE"),
            (datetime(2026, 1, 1, 13, 0), "QUIET")
        ]
        detected = [
            (datetime(2026, 1, 1, 12, 1), "FLARE")  # Only detected first
        ]

        metrics = compute_mode_detection_metrics(detected, expected, tolerance_minutes=5)

        assert metrics.true_positives == 1
        assert metrics.false_negatives == 1
        assert metrics.recall < 1.0

    def test_false_detection(self):
        """Test false mode switch detection"""
        expected = [
            (datetime(2026, 1, 1, 12, 0), "FLARE")
        ]
        detected = [
            (datetime(2026, 1, 1, 12, 1), "FLARE"),
            (datetime(2026, 1, 1, 14, 0), "STORM")  # False positive
        ]

        metrics = compute_mode_detection_metrics(detected, expected, tolerance_minutes=5)

        assert metrics.true_positives == 1
        assert metrics.false_positives == 1
        assert metrics.precision < 1.0


class TestThresholdChecking:
    """Test threshold checking"""

    def test_all_pass(self):
        """Test when all metrics pass thresholds"""
        metrics = IonoMetrics()
        metrics.fof2_metrics = StatisticalMetrics(rmse=0.5, bias=0.1, correlation=0.9)
        metrics.hmf2_metrics = StatisticalMetrics(rmse=20, bias=5, correlation=0.8)
        metrics.tec_metrics = StatisticalMetrics(rmse=3, bias=1, correlation=0.9)

        results = check_thresholds(metrics)

        assert all(results.values())

    def test_some_fail(self):
        """Test when some metrics fail thresholds"""
        metrics = IonoMetrics()
        metrics.fof2_metrics = StatisticalMetrics(rmse=2.0, bias=0.1, correlation=0.9)  # RMSE too high
        metrics.hmf2_metrics = StatisticalMetrics(rmse=20, bias=5, correlation=0.8)
        metrics.tec_metrics = StatisticalMetrics(rmse=3, bias=1, correlation=0.5)  # Correlation too low

        results = check_thresholds(metrics)

        assert not results['fof2_rmse']
        assert results['hmf2_rmse']
        assert not results['tec_correlation']


class TestEventReplayer:
    """Test event replayer functionality"""

    @pytest.fixture
    def sample_event_dir(self):
        """Create sample event directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            event_dir = Path(tmpdir) / "test_event"
            create_sample_event_config(event_dir)
            yield event_dir

    def test_load_event_config(self, sample_event_dir):
        """Test loading event configuration"""
        replayer = EventReplayer(sample_event_dir / 'event_config.yaml')
        config = replayer.load_event()

        assert config.name == 'X9.3 Flare - September 6, 2017'
        assert config.event_type == 'flare'
        assert config.start_time.year == 2017

    def test_load_observations(self, sample_event_dir):
        """Test loading observations from files"""
        replayer = EventReplayer(sample_event_dir / 'event_config.yaml')
        replayer.load_event()
        observations = replayer.load_observations()

        assert len(observations) > 0

        # Check observation types loaded
        obs_types = set(obs.obs_type for obs in observations)
        assert ObservationType.IONOSONDE in obs_types or ObservationType.GOES_XRAY in obs_types

    def test_observations_sorted(self, sample_event_dir):
        """Test that observations are sorted by timestamp"""
        replayer = EventReplayer(sample_event_dir / 'event_config.yaml')
        replayer.load_event()
        observations = replayer.load_observations()

        if len(observations) > 1:
            for i in range(len(observations) - 1):
                assert observations[i].timestamp <= observations[i + 1].timestamp

    def test_create_event_config_directly(self):
        """Test creating EventConfig directly"""
        config = EventConfig(
            name="Test Event",
            description="A test event",
            start_time=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
            end_time=datetime(2026, 1, 1, 14, 0, tzinfo=timezone.utc),
            event_type="test"
        )

        replayer = EventReplayer(event_config=config)
        loaded_config = replayer.load_event()

        assert loaded_config.name == "Test Event"


class TestValidationRunner:
    """Test validation runner"""

    def test_align_time_series(self):
        """Test time series alignment"""
        runner = ValidationRunner()

        base_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

        truth = [
            (base_time + timedelta(minutes=0), 8.0),
            (base_time + timedelta(minutes=5), 8.5),
            (base_time + timedelta(minutes=10), 9.0),
        ]

        estimates = [
            (base_time + timedelta(minutes=0, seconds=30), 8.1),  # Close to truth[0]
            (base_time + timedelta(minutes=5, seconds=45), 8.6),  # Close to truth[1]
            (base_time + timedelta(minutes=20), 9.5),  # No match
        ]

        comparison = runner.align_time_series(truth, estimates)

        assert len(comparison.estimates) == 2  # Only 2 matches
        assert comparison.estimates[0] == pytest.approx(8.1)
        assert comparison.truth[0] == pytest.approx(8.0)

    def test_full_validation(self):
        """Test complete validation workflow"""
        runner = ValidationRunner()

        base_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
        times = [base_time + timedelta(minutes=i*5) for i in range(20)]

        np.random.seed(42)
        truth_fof2 = [(t, 8.0 + np.random.normal(0, 0.3)) for t in times]
        est_fof2 = [(t, v + np.random.normal(0, 0.2)) for t, v in truth_fof2]

        runner.load_truth(fof2_truth=truth_fof2)
        runner.load_estimates(fof2_estimates=est_fof2)

        report = runner.validate(event_name="Test Event", event_type="test")

        assert report.event_name == "Test Event"
        assert report.iono_metrics.fof2_metrics.n_samples > 0
        assert report.iono_metrics.fof2_metrics.rmse < 1.0


class TestObservationParsing:
    """Test observation data parsing"""

    def test_ionosonde_observation(self):
        """Test creating ionosonde observation"""
        obs = Observation(
            timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
            obs_type=ObservationType.IONOSONDE,
            latitude=40.0,
            longitude=-105.0,
            values={'foF2': 8.5, 'hmF2': 280},
            errors={'foF2': 0.3, 'hmF2': 15}
        )

        assert obs.obs_type == ObservationType.IONOSONDE
        assert obs.values['foF2'] == 8.5
        assert obs.errors['foF2'] == 0.3

    def test_tec_observation(self):
        """Test creating TEC observation"""
        obs = Observation(
            timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
            obs_type=ObservationType.TEC,
            latitude=40.0,
            longitude=-105.0,
            values={'tec': 25.5},
            metadata={'elevation': 45.0, 'azimuth': 90.0}
        )

        assert obs.obs_type == ObservationType.TEC
        assert obs.values['tec'] == 25.5
        assert obs.metadata['elevation'] == 45.0


class TestReportGeneration:
    """Test validation report generation"""

    def test_generate_report(self):
        """Test generating validation report"""
        fof2_comp = TimeSeriesComparison(
            times=[datetime(2026, 1, 1, 12, 0)],
            estimates=[8.5],
            truth=[8.0],
            errors=[0.5]
        )

        report = generate_validation_report(
            event_name="Test Event",
            event_type="test",
            duration_minutes=60,
            observations_processed=100,
            fof2_comparison=fof2_comp
        )

        assert report.event_name == "Test Event"
        assert report.duration_minutes == 60
        assert len(report.summary) > 0

    def test_report_summary_format(self):
        """Test report summary formatting"""
        fof2_comp = TimeSeriesComparison(
            times=[datetime(2026, 1, 1, 12, i) for i in range(10)],
            estimates=[8.0 + 0.1*i for i in range(10)],
            truth=[8.0 + 0.05*i for i in range(10)],
            errors=[0.05*i for i in range(10)]
        )

        report = generate_validation_report(
            event_name="Summary Test",
            event_type="test",
            duration_minutes=30,
            observations_processed=50,
            fof2_comparison=fof2_comp
        )

        assert "Summary Test" in report.summary
        assert "foF2 RMSE" in report.summary
        assert "30" in report.summary  # Duration


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
