"""
Historical Validation Framework for Auto-NVIS

This package provides tools for validating the Auto-NVIS assimilation system
against historical ionospheric events with known ground truth.

Components:
    - event_replayer: Load and replay historical observations
    - validation_metrics: Compute accuracy metrics against ground truth

Usage:
    from tests.validation import EventReplayer, ValidationRunner

    # Replay historical event
    replayer = EventReplayer("events/x9_flare_2017.yaml")
    replayer.load_event()
    replayer.set_filter(filter_instance)
    results = replayer.run()

    # Validate against truth
    runner = ValidationRunner()
    runner.load_truth(fof2_truth=...)
    runner.load_estimates(fof2_estimates=...)
    report = runner.validate()
"""

from .event_replayer import (
    EventReplayer,
    EventConfig,
    ExpectedBehavior,
    Observation,
    ObservationType,
    FilterSnapshot,
    ReplayResult,
    create_sample_event_config
)

from .validation_metrics import (
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
    ResponseTimeMetrics,
    ModeDetectionMetrics,
    ValidationReport,
    TimeSeriesComparison,
    ValidationRunner
)

__all__ = [
    'EventReplayer',
    'EventConfig',
    'ExpectedBehavior',
    'Observation',
    'ObservationType',
    'FilterSnapshot',
    'ReplayResult',
    'create_sample_event_config',
    'compute_statistical_metrics',
    'compute_fof2_metrics',
    'compute_hmf2_metrics',
    'compute_tec_metrics',
    'compute_response_time',
    'compute_mode_detection_metrics',
    'check_thresholds',
    'generate_validation_report',
    'StatisticalMetrics',
    'IonoMetrics',
    'ResponseTimeMetrics',
    'ModeDetectionMetrics',
    'ValidationReport',
    'TimeSeriesComparison',
    'ValidationRunner'
]
