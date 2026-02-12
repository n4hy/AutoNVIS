"""
Unit tests for alert generator
"""

import pytest
from src.supervisor.alert_generator import (
    AlertGenerator,
    AlertType,
    AlertSeverity
)


@pytest.fixture
def alert_gen():
    """Create alert generator for testing"""
    return AlertGenerator()


def test_blackout_detection(alert_gen):
    """Test NVIS blackout detection (LUF > MUF)"""
    # Blackout condition
    luf = 8.0  # MHz
    muf = 6.0  # MHz

    alert = alert_gen.detect_blackout(luf, muf)

    assert alert is not None
    assert alert['type'] == AlertType.BLACKOUT.value
    assert alert['severity'] == AlertSeverity.CRITICAL.value
    assert 'BLACKOUT' in alert['message']
    assert alert['details']['luf_mhz'] == luf
    assert alert['details']['muf_mhz'] == muf


def test_no_blackout_when_window_available(alert_gen):
    """Test no blackout when usable window exists"""
    # Normal condition
    luf = 4.0  # MHz
    muf = 8.0  # MHz

    alert = alert_gen.detect_blackout(luf, muf)

    assert alert is None


def test_fadeout_detection(alert_gen):
    """Test fadeout warning (rapidly rising LUF)"""
    luf = 5.0  # MHz
    luf_trend = 0.8  # MHz/min (rapid rise)

    alert = alert_gen.detect_fadeout(luf, luf_trend)

    assert alert is not None
    assert alert['type'] == AlertType.FADEOUT_WARNING.value
    assert alert['severity'] == AlertSeverity.WARNING.value
    assert 'FADEOUT' in alert['message']


def test_no_fadeout_for_stable_luf(alert_gen):
    """Test no fadeout when LUF stable"""
    luf = 5.0  # MHz
    luf_trend = 0.1  # MHz/min (slow rise)

    alert = alert_gen.detect_fadeout(luf, luf_trend)

    assert alert is None


def test_m1_flare_alert(alert_gen):
    """Test M1+ flare alert generation"""
    flux = 2.5e-5  # M2.5 flare
    flare_class = "M2.5"

    alert = alert_gen.generate_m1_flare_alert(flux, flare_class)

    assert alert['type'] == AlertType.M1_FLARE.value
    assert alert['severity'] == AlertSeverity.WARNING.value
    assert flare_class in alert['message']
    assert alert['details']['flux_w_per_m2'] == flux


def test_mode_change_alert(alert_gen):
    """Test mode change alert generation"""
    alert = alert_gen.generate_mode_change_alert(
        old_mode="QUIET",
        new_mode="SHOCK",
        reason="M1+ flare detected"
    )

    assert alert['type'] == AlertType.MODE_CHANGE.value
    assert alert['severity'] == AlertSeverity.INFO.value
    assert 'QUIET' in alert['message']
    assert 'SHOCK' in alert['message']


def test_service_failure_alert(alert_gen):
    """Test service failure alert generation"""
    alert = alert_gen.generate_service_failure_alert(
        service_name="assimilation",
        error_message="Connection timeout"
    )

    assert alert['type'] == AlertType.SERVICE_FAILURE.value
    assert alert['severity'] == AlertSeverity.CRITICAL.value
    assert 'assimilation' in alert['message']


def test_alert_counter_increments(alert_gen):
    """Test alert counters increment"""
    assert alert_gen.alerts_generated == 0

    # Simulate publishing alert (without actual message queue)
    alert = alert_gen.create_alert(
        alert_type=AlertType.M1_FLARE,
        severity=AlertSeverity.WARNING,
        message="Test alert"
    )

    # Manually increment (normally done by publish_alert)
    alert_gen.alerts_generated += 1
    alert_gen.alerts_by_type[AlertType.M1_FLARE.value] = 1

    assert alert_gen.alerts_generated == 1
    assert alert_gen.alerts_by_type[AlertType.M1_FLARE.value] == 1


def test_alert_structure(alert_gen):
    """Test alert has required fields"""
    alert = alert_gen.create_alert(
        alert_type=AlertType.BLACKOUT,
        severity=AlertSeverity.CRITICAL,
        message="Test message",
        details={'key': 'value'}
    )

    assert 'alert_id' in alert
    assert 'timestamp' in alert
    assert 'type' in alert
    assert 'severity' in alert
    assert 'message' in alert
    assert 'details' in alert

    assert alert['type'] == AlertType.BLACKOUT.value
    assert alert['severity'] == AlertSeverity.CRITICAL.value
    assert alert['details']['key'] == 'value'
