"""
Validation Metrics for Auto-NVIS

Computes standard metrics for comparing filter estimates against ground truth:
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- Bias
- Correlation coefficient
- Response time metrics

Also computes ionospheric-specific metrics:
- foF2 accuracy
- hmF2 accuracy
- TEC accuracy
- Mode detection accuracy
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from enum import Enum

logger = logging.getLogger(__name__)


@dataclass
class TimeSeriesComparison:
    """Comparison between estimate and truth time series"""
    times: List[datetime] = field(default_factory=list)
    estimates: List[float] = field(default_factory=list)
    truth: List[float] = field(default_factory=list)
    errors: List[float] = field(default_factory=list)  # estimate - truth


@dataclass
class StatisticalMetrics:
    """Standard statistical metrics"""
    rmse: float = 0.0
    mae: float = 0.0
    bias: float = 0.0
    std: float = 0.0
    correlation: float = 0.0
    n_samples: int = 0
    min_error: float = 0.0
    max_error: float = 0.0
    percentile_90: float = 0.0
    percentile_95: float = 0.0


@dataclass
class IonoMetrics:
    """Ionospheric-specific metrics"""
    fof2_metrics: StatisticalMetrics = field(default_factory=StatisticalMetrics)
    hmf2_metrics: StatisticalMetrics = field(default_factory=StatisticalMetrics)
    tec_metrics: StatisticalMetrics = field(default_factory=StatisticalMetrics)
    ne_metrics: StatisticalMetrics = field(default_factory=StatisticalMetrics)


@dataclass
class ResponseTimeMetrics:
    """Metrics for filter response to events"""
    detection_delay_minutes: Optional[float] = None
    response_time_minutes: Optional[float] = None  # Time to 90% of peak change
    recovery_time_minutes: Optional[float] = None  # Time to return to baseline
    overshoot_percent: Optional[float] = None
    settling_time_minutes: Optional[float] = None


@dataclass
class ModeDetectionMetrics:
    """Metrics for mode detection accuracy"""
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    detection_delay_minutes: Optional[float] = None


@dataclass
class ValidationReport:
    """Complete validation report for an event"""
    event_name: str = ""
    event_type: str = ""
    duration_minutes: float = 0.0
    observations_processed: int = 0

    iono_metrics: IonoMetrics = field(default_factory=IonoMetrics)
    response_metrics: ResponseTimeMetrics = field(default_factory=ResponseTimeMetrics)
    mode_metrics: ModeDetectionMetrics = field(default_factory=ModeDetectionMetrics)

    passed_thresholds: Dict[str, bool] = field(default_factory=dict)
    summary: str = ""


def compute_statistical_metrics(
    estimates: np.ndarray,
    truth: np.ndarray,
    weights: Optional[np.ndarray] = None
) -> StatisticalMetrics:
    """
    Compute standard statistical metrics.

    Args:
        estimates: Array of estimated values
        truth: Array of ground truth values
        weights: Optional weights for weighted statistics

    Returns:
        StatisticalMetrics object
    """
    if len(estimates) == 0 or len(truth) == 0:
        return StatisticalMetrics()

    estimates = np.asarray(estimates)
    truth = np.asarray(truth)

    # Remove NaN values
    valid_mask = ~(np.isnan(estimates) | np.isnan(truth))
    if not np.any(valid_mask):
        return StatisticalMetrics()

    est = estimates[valid_mask]
    tru = truth[valid_mask]

    errors = est - tru
    n = len(errors)

    if weights is not None:
        w = weights[valid_mask]
        w = w / np.sum(w)  # Normalize weights

        rmse = np.sqrt(np.sum(w * errors**2))
        mae = np.sum(w * np.abs(errors))
        bias = np.sum(w * errors)
        var = np.sum(w * (errors - bias)**2)
        std = np.sqrt(var)
    else:
        rmse = np.sqrt(np.mean(errors**2))
        mae = np.mean(np.abs(errors))
        bias = np.mean(errors)
        std = np.std(errors)

    # Correlation coefficient
    if np.std(est) > 0 and np.std(tru) > 0:
        correlation = np.corrcoef(est, tru)[0, 1]
    else:
        correlation = 0.0

    # Percentiles of absolute error
    abs_errors = np.abs(errors)
    percentile_90 = np.percentile(abs_errors, 90)
    percentile_95 = np.percentile(abs_errors, 95)

    return StatisticalMetrics(
        rmse=float(rmse),
        mae=float(mae),
        bias=float(bias),
        std=float(std),
        correlation=float(correlation),
        n_samples=n,
        min_error=float(np.min(errors)),
        max_error=float(np.max(errors)),
        percentile_90=float(percentile_90),
        percentile_95=float(percentile_95)
    )


def compute_fof2_metrics(
    estimated_fof2: np.ndarray,
    truth_fof2: np.ndarray
) -> StatisticalMetrics:
    """
    Compute metrics for critical frequency foF2.

    Args:
        estimated_fof2: Estimated foF2 values (MHz)
        truth_fof2: Ground truth foF2 values (MHz)

    Returns:
        StatisticalMetrics for foF2
    """
    metrics = compute_statistical_metrics(estimated_fof2, truth_fof2)

    # Log summary
    logger.info(f"foF2 Metrics:")
    logger.info(f"  RMSE: {metrics.rmse:.3f} MHz")
    logger.info(f"  Bias: {metrics.bias:.3f} MHz")
    logger.info(f"  Correlation: {metrics.correlation:.3f}")

    return metrics


def compute_hmf2_metrics(
    estimated_hmf2: np.ndarray,
    truth_hmf2: np.ndarray
) -> StatisticalMetrics:
    """
    Compute metrics for peak height hmF2.

    Args:
        estimated_hmf2: Estimated hmF2 values (km)
        truth_hmf2: Ground truth hmF2 values (km)

    Returns:
        StatisticalMetrics for hmF2
    """
    metrics = compute_statistical_metrics(estimated_hmf2, truth_hmf2)

    logger.info(f"hmF2 Metrics:")
    logger.info(f"  RMSE: {metrics.rmse:.1f} km")
    logger.info(f"  Bias: {metrics.bias:.1f} km")
    logger.info(f"  Correlation: {metrics.correlation:.3f}")

    return metrics


def compute_tec_metrics(
    estimated_tec: np.ndarray,
    truth_tec: np.ndarray
) -> StatisticalMetrics:
    """
    Compute metrics for Total Electron Content.

    Args:
        estimated_tec: Estimated TEC values (TECU)
        truth_tec: Ground truth TEC values (TECU)

    Returns:
        StatisticalMetrics for TEC
    """
    metrics = compute_statistical_metrics(estimated_tec, truth_tec)

    logger.info(f"TEC Metrics:")
    logger.info(f"  RMSE: {metrics.rmse:.2f} TECU")
    logger.info(f"  Bias: {metrics.bias:.2f} TECU")
    logger.info(f"  Correlation: {metrics.correlation:.3f}")

    return metrics


def compute_response_time(
    times: List[datetime],
    estimates: List[float],
    event_time: datetime,
    baseline: float,
    target_change_percent: float = 90.0
) -> ResponseTimeMetrics:
    """
    Compute filter response time metrics.

    Measures how quickly the filter responds to sudden changes.

    Args:
        times: Timestamps of estimates
        estimates: Filter estimates over time
        event_time: Known time of event onset
        baseline: Pre-event baseline value
        target_change_percent: Percentage of peak change to use (default 90%)

    Returns:
        ResponseTimeMetrics
    """
    if len(times) < 2 or len(estimates) < 2:
        return ResponseTimeMetrics()

    times = np.array(times)
    estimates = np.array(estimates)

    # Find peak deviation from baseline
    deviations = estimates - baseline
    peak_deviation = np.max(np.abs(deviations))
    peak_idx = np.argmax(np.abs(deviations))
    peak_time = times[peak_idx]

    if peak_deviation < 0.01 * abs(baseline):
        # No significant change detected
        return ResponseTimeMetrics()

    # Detection delay: time from event to first significant change (10% of peak)
    threshold_10 = 0.1 * peak_deviation
    detection_idx = None
    for i, dev in enumerate(np.abs(deviations)):
        if dev > threshold_10:
            detection_idx = i
            break

    detection_delay = None
    if detection_idx is not None:
        detection_delay = (times[detection_idx] - event_time).total_seconds() / 60.0

    # Response time: time from event to target % of peak change
    threshold_target = (target_change_percent / 100.0) * peak_deviation
    response_idx = None
    for i, dev in enumerate(np.abs(deviations)):
        if dev >= threshold_target:
            response_idx = i
            break

    response_time = None
    if response_idx is not None:
        response_time = (times[response_idx] - event_time).total_seconds() / 60.0

    # Recovery time: time from peak to return to within 10% of baseline
    recovery_idx = None
    for i in range(peak_idx, len(deviations)):
        if np.abs(deviations[i]) < threshold_10:
            recovery_idx = i
            break

    recovery_time = None
    if recovery_idx is not None:
        recovery_time = (times[recovery_idx] - peak_time).total_seconds() / 60.0

    # Overshoot: maximum deviation beyond steady-state change
    # (This requires knowing the expected steady-state, approximate here)
    expected_change = deviations[-1] if len(deviations) > 10 else peak_deviation
    if abs(expected_change) > 0:
        overshoot = abs((peak_deviation - abs(expected_change)) / expected_change) * 100
    else:
        overshoot = None

    return ResponseTimeMetrics(
        detection_delay_minutes=detection_delay,
        response_time_minutes=response_time,
        recovery_time_minutes=recovery_time,
        overshoot_percent=overshoot
    )


def compute_mode_detection_metrics(
    filter_modes: List[Tuple[datetime, str]],
    expected_modes: List[Tuple[datetime, str]],
    tolerance_minutes: float = 10.0
) -> ModeDetectionMetrics:
    """
    Compute mode detection accuracy.

    Compares detected mode switches against expected mode switches.

    Args:
        filter_modes: List of (timestamp, mode) from filter
        expected_modes: List of (timestamp, mode) expected
        tolerance_minutes: Time tolerance for matching switches

    Returns:
        ModeDetectionMetrics
    """
    tolerance = timedelta(minutes=tolerance_minutes)

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    matched_expected = set()
    detection_delays = []

    # Match filter detections to expected
    for filter_time, filter_mode in filter_modes:
        matched = False
        for i, (expected_time, expected_mode) in enumerate(expected_modes):
            if i in matched_expected:
                continue

            time_diff = abs((filter_time - expected_time).total_seconds())
            if time_diff < tolerance.total_seconds() and filter_mode == expected_mode:
                true_positives += 1
                matched_expected.add(i)
                matched = True
                detection_delays.append(time_diff / 60.0)
                break

        if not matched:
            false_positives += 1

    # Count missed detections
    false_negatives = len(expected_modes) - len(matched_expected)

    # Calculate precision, recall, F1
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    avg_delay = np.mean(detection_delays) if detection_delays else None

    return ModeDetectionMetrics(
        true_positives=true_positives,
        false_positives=false_positives,
        true_negatives=0,  # Would need continuous mode tracking
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1_score=f1,
        detection_delay_minutes=avg_delay
    )


def check_thresholds(
    metrics: IonoMetrics,
    thresholds: Optional[Dict[str, float]] = None
) -> Dict[str, bool]:
    """
    Check if metrics meet acceptance thresholds.

    Args:
        metrics: Computed ionospheric metrics
        thresholds: Dict of threshold values (default uses standard values)

    Returns:
        Dict mapping metric names to pass/fail boolean
    """
    if thresholds is None:
        # Default thresholds based on typical ionospheric variability
        thresholds = {
            'fof2_rmse_mhz': 1.0,
            'fof2_bias_mhz': 0.3,
            'fof2_correlation': 0.8,
            'hmf2_rmse_km': 30.0,
            'hmf2_bias_km': 10.0,
            'hmf2_correlation': 0.7,
            'tec_rmse_tecu': 5.0,
            'tec_bias_tecu': 2.0,
            'tec_correlation': 0.85
        }

    results = {}

    # foF2 thresholds
    results['fof2_rmse'] = metrics.fof2_metrics.rmse <= thresholds.get('fof2_rmse_mhz', float('inf'))
    results['fof2_bias'] = abs(metrics.fof2_metrics.bias) <= thresholds.get('fof2_bias_mhz', float('inf'))
    results['fof2_correlation'] = metrics.fof2_metrics.correlation >= thresholds.get('fof2_correlation', 0)

    # hmF2 thresholds
    results['hmf2_rmse'] = metrics.hmf2_metrics.rmse <= thresholds.get('hmf2_rmse_km', float('inf'))
    results['hmf2_bias'] = abs(metrics.hmf2_metrics.bias) <= thresholds.get('hmf2_bias_km', float('inf'))
    results['hmf2_correlation'] = metrics.hmf2_metrics.correlation >= thresholds.get('hmf2_correlation', 0)

    # TEC thresholds
    results['tec_rmse'] = metrics.tec_metrics.rmse <= thresholds.get('tec_rmse_tecu', float('inf'))
    results['tec_bias'] = abs(metrics.tec_metrics.bias) <= thresholds.get('tec_bias_tecu', float('inf'))
    results['tec_correlation'] = metrics.tec_metrics.correlation >= thresholds.get('tec_correlation', 0)

    return results


def generate_validation_report(
    event_name: str,
    event_type: str,
    duration_minutes: float,
    observations_processed: int,
    fof2_comparison: Optional[TimeSeriesComparison] = None,
    hmf2_comparison: Optional[TimeSeriesComparison] = None,
    tec_comparison: Optional[TimeSeriesComparison] = None,
    filter_modes: Optional[List[Tuple[datetime, str]]] = None,
    expected_modes: Optional[List[Tuple[datetime, str]]] = None,
    event_time: Optional[datetime] = None,
    thresholds: Optional[Dict[str, float]] = None
) -> ValidationReport:
    """
    Generate a complete validation report.

    Args:
        event_name: Name of the event
        event_type: Type of event (flare, storm, quiet)
        duration_minutes: Event duration in minutes
        observations_processed: Number of observations processed
        fof2_comparison: foF2 estimate vs truth comparison
        hmf2_comparison: hmF2 estimate vs truth comparison
        tec_comparison: TEC estimate vs truth comparison
        filter_modes: Detected mode switches
        expected_modes: Expected mode switches
        event_time: Time of event onset (for response time calculation)
        thresholds: Acceptance thresholds

    Returns:
        ValidationReport object
    """
    report = ValidationReport(
        event_name=event_name,
        event_type=event_type,
        duration_minutes=duration_minutes,
        observations_processed=observations_processed
    )

    # Compute ionospheric metrics
    if fof2_comparison and fof2_comparison.estimates:
        report.iono_metrics.fof2_metrics = compute_fof2_metrics(
            np.array(fof2_comparison.estimates),
            np.array(fof2_comparison.truth)
        )

    if hmf2_comparison and hmf2_comparison.estimates:
        report.iono_metrics.hmf2_metrics = compute_hmf2_metrics(
            np.array(hmf2_comparison.estimates),
            np.array(hmf2_comparison.truth)
        )

    if tec_comparison and tec_comparison.estimates:
        report.iono_metrics.tec_metrics = compute_tec_metrics(
            np.array(tec_comparison.estimates),
            np.array(tec_comparison.truth)
        )

    # Compute response time metrics
    if fof2_comparison and event_time and fof2_comparison.estimates:
        baseline = np.mean(fof2_comparison.truth[:5]) if len(fof2_comparison.truth) >= 5 else fof2_comparison.truth[0]
        report.response_metrics = compute_response_time(
            fof2_comparison.times,
            fof2_comparison.estimates,
            event_time,
            baseline
        )

    # Compute mode detection metrics
    if filter_modes and expected_modes:
        report.mode_metrics = compute_mode_detection_metrics(filter_modes, expected_modes)

    # Check thresholds
    report.passed_thresholds = check_thresholds(report.iono_metrics, thresholds)

    # Generate summary
    n_passed = sum(report.passed_thresholds.values())
    n_total = len(report.passed_thresholds)

    report.summary = (
        f"Validation Report: {event_name}\n"
        f"Event Type: {event_type}\n"
        f"Duration: {duration_minutes:.0f} minutes\n"
        f"Observations: {observations_processed}\n"
        f"\n"
        f"Metrics Summary:\n"
        f"  foF2 RMSE: {report.iono_metrics.fof2_metrics.rmse:.3f} MHz\n"
        f"  hmF2 RMSE: {report.iono_metrics.hmf2_metrics.rmse:.1f} km\n"
        f"  TEC RMSE: {report.iono_metrics.tec_metrics.rmse:.2f} TECU\n"
        f"\n"
        f"Thresholds: {n_passed}/{n_total} passed\n"
    )

    if report.mode_metrics.f1_score > 0:
        report.summary += (
            f"\n"
            f"Mode Detection:\n"
            f"  Precision: {report.mode_metrics.precision:.2f}\n"
            f"  Recall: {report.mode_metrics.recall:.2f}\n"
            f"  F1 Score: {report.mode_metrics.f1_score:.2f}\n"
        )

    if report.response_metrics.response_time_minutes is not None:
        report.summary += (
            f"\n"
            f"Response Time:\n"
            f"  Detection: {report.response_metrics.detection_delay_minutes:.1f} min\n"
            f"  90% Response: {report.response_metrics.response_time_minutes:.1f} min\n"
        )

    return report


class ValidationRunner:
    """
    Runs validation against ground truth data.

    Usage:
        runner = ValidationRunner()
        runner.load_truth(truth_file)
        runner.load_estimates(estimates_file)
        report = runner.validate()
    """

    def __init__(self):
        self.truth_data: Dict[str, TimeSeriesComparison] = {}
        self.estimate_data: Dict[str, List[Tuple[datetime, float]]] = {}
        self.event_config: Optional[Dict[str, Any]] = None

    def load_truth(
        self,
        fof2_truth: Optional[List[Tuple[datetime, float]]] = None,
        hmf2_truth: Optional[List[Tuple[datetime, float]]] = None,
        tec_truth: Optional[List[Tuple[datetime, float]]] = None
    ) -> None:
        """Load ground truth data"""
        if fof2_truth:
            self.truth_data['fof2'] = fof2_truth
        if hmf2_truth:
            self.truth_data['hmf2'] = hmf2_truth
        if tec_truth:
            self.truth_data['tec'] = tec_truth

    def load_estimates(
        self,
        fof2_estimates: Optional[List[Tuple[datetime, float]]] = None,
        hmf2_estimates: Optional[List[Tuple[datetime, float]]] = None,
        tec_estimates: Optional[List[Tuple[datetime, float]]] = None
    ) -> None:
        """Load filter estimates"""
        if fof2_estimates:
            self.estimate_data['fof2'] = fof2_estimates
        if hmf2_estimates:
            self.estimate_data['hmf2'] = hmf2_estimates
        if tec_estimates:
            self.estimate_data['tec'] = tec_estimates

    def align_time_series(
        self,
        truth: List[Tuple[datetime, float]],
        estimates: List[Tuple[datetime, float]],
        tolerance_seconds: float = 60.0
    ) -> TimeSeriesComparison:
        """
        Align two time series for comparison.

        Matches estimate times to nearest truth times within tolerance.
        """
        comparison = TimeSeriesComparison()

        truth_dict = {t: v for t, v in truth}
        truth_times = sorted(truth_dict.keys())

        for est_time, est_value in estimates:
            # Find nearest truth time
            best_match = None
            best_diff = float('inf')

            for truth_time in truth_times:
                diff = abs((est_time - truth_time).total_seconds())
                if diff < best_diff:
                    best_diff = diff
                    best_match = truth_time

            if best_match and best_diff <= tolerance_seconds:
                truth_value = truth_dict[best_match]
                comparison.times.append(est_time)
                comparison.estimates.append(est_value)
                comparison.truth.append(truth_value)
                comparison.errors.append(est_value - truth_value)

        return comparison

    def validate(
        self,
        event_name: str = "Unknown Event",
        event_type: str = "unknown",
        event_time: Optional[datetime] = None
    ) -> ValidationReport:
        """Run validation and generate report"""

        # Align time series
        fof2_comp = None
        if 'fof2' in self.truth_data and 'fof2' in self.estimate_data:
            fof2_comp = self.align_time_series(
                self.truth_data['fof2'],
                self.estimate_data['fof2']
            )

        hmf2_comp = None
        if 'hmf2' in self.truth_data and 'hmf2' in self.estimate_data:
            hmf2_comp = self.align_time_series(
                self.truth_data['hmf2'],
                self.estimate_data['hmf2']
            )

        tec_comp = None
        if 'tec' in self.truth_data and 'tec' in self.estimate_data:
            tec_comp = self.align_time_series(
                self.truth_data['tec'],
                self.estimate_data['tec']
            )

        # Calculate duration
        all_times = []
        for comp in [fof2_comp, hmf2_comp, tec_comp]:
            if comp and comp.times:
                all_times.extend(comp.times)

        if all_times:
            duration = (max(all_times) - min(all_times)).total_seconds() / 60.0
        else:
            duration = 0.0

        n_obs = sum(len(comp.estimates) for comp in [fof2_comp, hmf2_comp, tec_comp] if comp)

        return generate_validation_report(
            event_name=event_name,
            event_type=event_type,
            duration_minutes=duration,
            observations_processed=n_obs,
            fof2_comparison=fof2_comp,
            hmf2_comparison=hmf2_comp,
            tec_comparison=tec_comp,
            event_time=event_time
        )


if __name__ == '__main__':
    # Demo usage
    logging.basicConfig(level=logging.INFO)

    # Create sample data
    np.random.seed(42)

    n_points = 100
    truth_fof2 = 8.0 + 0.5 * np.sin(np.linspace(0, 2*np.pi, n_points))
    estimated_fof2 = truth_fof2 + np.random.normal(0, 0.3, n_points)

    truth_hmf2 = 280.0 + 20 * np.sin(np.linspace(0, 2*np.pi, n_points))
    estimated_hmf2 = truth_hmf2 + np.random.normal(0, 15, n_points)

    # Compute metrics
    fof2_metrics = compute_fof2_metrics(estimated_fof2, truth_fof2)
    hmf2_metrics = compute_hmf2_metrics(estimated_hmf2, truth_hmf2)

    print(f"\nfoF2 RMSE: {fof2_metrics.rmse:.3f} MHz")
    print(f"hmF2 RMSE: {hmf2_metrics.rmse:.1f} km")
