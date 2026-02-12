"""
Unit tests for mode controller
"""

import pytest
from datetime import datetime, timedelta
from src.supervisor.mode_controller import ModeController, OperationalMode


@pytest.fixture
def mode_controller():
    """Create mode controller for testing"""
    return ModeController(
        xray_threshold=1e-5,  # M1 class
        hysteresis_sec=600     # 10 minutes
    )


def test_initial_mode(mode_controller):
    """Test initial mode is QUIET"""
    assert mode_controller.get_current_mode() == OperationalMode.QUIET


def test_switch_to_shock_on_m1_flare(mode_controller):
    """Test switching to SHOCK mode on M1+ flare"""
    # M2 flare (above threshold)
    flux_m2 = 2e-5

    assert mode_controller.should_switch_to_shock(flux_m2) is True

    mode_controller.switch_to_shock_mode(flux_m2, "M2.0")

    assert mode_controller.get_current_mode() == OperationalMode.SHOCK
    assert mode_controller.mode_change_count == 1


def test_no_switch_below_threshold(mode_controller):
    """Test no switch when flux below threshold"""
    # C9 flare (below threshold)
    flux_c9 = 9e-7

    assert mode_controller.should_switch_to_shock(flux_c9) is False
    assert mode_controller.get_current_mode() == OperationalMode.QUIET


def test_hysteresis_prevents_immediate_switch_back(mode_controller):
    """Test hysteresis prevents immediate switch back to QUIET"""
    # Switch to SHOCK
    mode_controller.switch_to_shock_mode(2e-5, "M2.0")
    assert mode_controller.get_current_mode() == OperationalMode.SHOCK

    # Flux drops below threshold
    flux_below = 5e-6  # Below M1

    # Should not switch immediately
    assert mode_controller.should_switch_to_quiet(flux_below) is False
    assert mode_controller.get_current_mode() == OperationalMode.SHOCK


def test_switch_to_quiet_after_hysteresis(mode_controller):
    """Test switching to QUIET after hysteresis period"""
    # Switch to SHOCK
    mode_controller.switch_to_shock_mode(2e-5, "M2.0")

    # Simulate hysteresis period
    mode_controller.flux_below_threshold_since = datetime.utcnow() - timedelta(
        seconds=mode_controller.hysteresis_sec + 1
    )

    # Should now allow switch to QUIET
    assert mode_controller.should_switch_to_quiet(5e-6) is True

    mode_controller.switch_to_quiet_mode(5e-6)
    assert mode_controller.get_current_mode() == OperationalMode.QUIET


def test_hysteresis_reset_on_flux_increase(mode_controller):
    """Test hysteresis timer resets if flux increases again"""
    # Switch to SHOCK
    mode_controller.switch_to_shock_mode(2e-5, "M2.0")

    # Flux drops, start hysteresis
    mode_controller.should_switch_to_quiet(5e-6)
    assert mode_controller.flux_below_threshold_since is not None

    # Flux increases above threshold again
    assert mode_controller.should_switch_to_quiet(2e-5) is False

    # Hysteresis should be reset
    assert mode_controller.flux_below_threshold_since is None


def test_mode_change_count(mode_controller):
    """Test mode change counter increments"""
    assert mode_controller.mode_change_count == 0

    # Switch to SHOCK
    mode_controller.switch_to_shock_mode(2e-5, "M2.0")
    assert mode_controller.mode_change_count == 1

    # Simulate hysteresis and switch back
    mode_controller.flux_below_threshold_since = datetime.utcnow() - timedelta(
        seconds=700
    )
    mode_controller.switch_to_quiet_mode(5e-6)
    assert mode_controller.mode_change_count == 2


def test_get_status(mode_controller):
    """Test status dictionary"""
    status = mode_controller.get_status()

    assert 'current_mode' in status
    assert 'threshold' in status
    assert 'hysteresis_sec' in status
    assert 'mode_change_count' in status

    assert status['current_mode'] == 'QUIET'
    assert status['threshold'] == 1e-5
    assert status['hysteresis_sec'] == 600
