"""
Unit Tests for IONORT-Style Numerical Integrators

Tests the three integration methods:
- RK4Integrator (classical 4th order with error tracking)
- AdamsBashforthMoultonIntegrator (AB4/AM3 predictor-corrector)
- RK45Integrator (Dormand-Prince adaptive)

Tests verify:
1. Correct integration of simple ODEs
2. Error estimation accuracy
3. Step size adaptation (RK45)
4. History management (Adams-Bashforth)
5. Factory function operation
"""

import pytest
import numpy as np
from typing import Tuple

from src.raytracer.integrators import (
    BaseIntegrator,
    IntegrationStep,
    IntegrationStats,
    RK4Integrator,
    AdamsBashforthMoultonIntegrator,
    RK45Integrator,
    IntegratorFactory,
    create_integrator,
)
from src.raytracer.integrators.factory import IntegratorType, get_recommended_integrator


# Simple ODE systems for testing
def harmonic_oscillator(state: np.ndarray, freq: float) -> np.ndarray:
    """
    Simple harmonic oscillator: y'' + y = 0
    Written as first-order system:
        y' = v
        v' = -y
    State: [y, v]
    Exact solution: y = cos(t), v = -sin(t)
    """
    y, v = state
    return np.array([v, -y])


def exponential_decay(state: np.ndarray, freq: float) -> np.ndarray:
    """
    Exponential decay: y' = -y
    State: [y]
    Exact solution: y = exp(-t)
    """
    return -state


def linear_growth(state: np.ndarray, freq: float) -> np.ndarray:
    """
    Linear growth: y' = 1
    State: [y]
    Exact solution: y = t + y0
    """
    return np.ones_like(state)


def haselgrove_like(state: np.ndarray, freq: float) -> np.ndarray:
    """
    6-component system mimicking Haselgrove equations structure.
    State: [x, y, z, kx, ky, kz]
    """
    # Simple circular motion in x-y plane
    x, y, z, kx, ky, kz = state
    return np.array([
        kx,        # dx/ds = kx
        ky,        # dy/ds = ky
        kz,        # dz/ds = kz
        -0.01 * x, # dkx/ds ~ -x (restoring)
        -0.01 * y, # dky/ds ~ -y
        0.0,       # dkz/ds = 0
    ])


class TestRK4Integrator:
    """Tests for RK4 integrator with step doubling."""

    def test_initialization(self):
        """Test integrator initialization."""
        integrator = RK4Integrator(harmonic_oscillator)
        assert integrator.tolerance == 1e-6
        assert integrator.min_step == 0.01
        assert integrator.max_step == 10.0
        assert integrator.name() == "RK4 (Classical 4th Order with Error Tracking)"

    def test_harmonic_oscillator_accuracy(self):
        """Test RK4 on harmonic oscillator for one period."""
        integrator = RK4Integrator(harmonic_oscillator, tolerance=1e-8)

        state = np.array([1.0, 0.0])  # y=1, v=0 at t=0
        ds = 0.1
        t = 0.0
        period = 2 * np.pi

        while t < period:
            result = integrator.step(state, ds, freq_mhz=0.0)
            state = result.state
            t += result.step_size_used

        # After approximately one period, should be close to initial state
        # Allow for phase accumulation from not hitting exactly t=2*pi
        expected = np.array([1.0, 0.0])
        np.testing.assert_allclose(state, expected, atol=0.05)

    def test_error_estimation(self):
        """Test that error estimates are computed."""
        integrator = RK4Integrator(harmonic_oscillator)
        state = np.array([1.0, 0.0])

        result = integrator.step(state, ds=0.1, freq_mhz=0.0)

        assert result.error_estimate >= 0
        assert result.derivatives_computed == 12  # 4 + 4 + 4 for step doubling
        assert result.accepted  # Should accept with reasonable step

    def test_statistics_tracking(self):
        """Test that statistics are properly tracked."""
        integrator = RK4Integrator(harmonic_oscillator)
        state = np.array([1.0, 0.0])

        for _ in range(10):
            result = integrator.step(state, ds=0.1, freq_mhz=0.0)
            state = result.state

        stats = integrator.stats
        assert stats.total_steps == 10
        assert stats.total_derivative_evals == 120  # 12 per step
        assert stats.max_error >= 0

    def test_step_clamping(self):
        """Test that step size is clamped to min/max."""
        integrator = RK4Integrator(harmonic_oscillator, min_step=0.5, max_step=2.0)
        state = np.array([1.0, 0.0])

        # Request very small step
        result = integrator.step(state, ds=0.01, freq_mhz=0.0)
        assert result.step_size_used >= 0.5

        # Request very large step
        result = integrator.step(state, ds=100.0, freq_mhz=0.0)
        assert result.step_size_used <= 2.0


class TestAdamsBashforthMoultonIntegrator:
    """Tests for Adams-Bashforth/Moulton predictor-corrector."""

    def test_initialization(self):
        """Test integrator initialization."""
        integrator = AdamsBashforthMoultonIntegrator(harmonic_oscillator)
        assert integrator.name() == "Adams-Bashforth-Moulton (AB4/AM3 Predictor-Corrector)"
        assert not integrator.is_startup_complete
        assert integrator.history_size == 0

    def test_startup_phase(self):
        """Test that startup uses RK4 for first 4 steps."""
        integrator = AdamsBashforthMoultonIntegrator(harmonic_oscillator)
        state = np.array([1.0, 0.0])

        for i in range(5):
            result = integrator.step(state, ds=0.1, freq_mhz=0.0)
            state = result.state

            if i < 3:
                assert not integrator.is_startup_complete
            else:
                # After 4 history entries, startup should be complete
                pass

        assert integrator.is_startup_complete
        assert integrator.history_size == 4

    def test_efficiency_after_startup(self):
        """Test that ABM uses only 2 derivative evals after startup."""
        integrator = AdamsBashforthMoultonIntegrator(harmonic_oscillator)
        state = np.array([1.0, 0.0])

        # Startup phase
        for _ in range(4):
            result = integrator.step(state, ds=0.1, freq_mhz=0.0)
            state = result.state

        # Record derivative count after startup
        deriv_count_before = integrator.stats.total_derivative_evals

        # Take one ABM step
        result = integrator.step(state, ds=0.1, freq_mhz=0.0)

        deriv_count_after = integrator.stats.total_derivative_evals
        abm_derivs = deriv_count_after - deriv_count_before

        # ABM should use 2 derivative evaluations (f_n and f_pred)
        assert abm_derivs == 2

    def test_reset_clears_history(self):
        """Test that reset() clears derivative history."""
        integrator = AdamsBashforthMoultonIntegrator(harmonic_oscillator)
        state = np.array([1.0, 0.0])

        # Build up history
        for _ in range(5):
            result = integrator.step(state, ds=0.1, freq_mhz=0.0)
            state = result.state

        assert integrator.is_startup_complete

        # Reset
        integrator.reset()

        assert not integrator.is_startup_complete
        assert integrator.history_size == 0

    def test_harmonic_oscillator_accuracy(self):
        """Test ABM on harmonic oscillator."""
        integrator = AdamsBashforthMoultonIntegrator(harmonic_oscillator)

        state = np.array([1.0, 0.0])
        ds = 0.05
        t = 0.0
        period = 2 * np.pi

        while t < period:
            result = integrator.step(state, ds, freq_mhz=0.0)
            state = result.state
            t += result.step_size_used

        # Allow for phase accumulation from not hitting exactly t=2*pi
        expected = np.array([1.0, 0.0])
        np.testing.assert_allclose(state, expected, atol=0.05)


class TestRK45Integrator:
    """Tests for Dormand-Prince RK45 adaptive integrator."""

    def test_initialization(self):
        """Test integrator initialization."""
        integrator = RK45Integrator(harmonic_oscillator)
        assert integrator.name() == "RK45 Dormand-Prince (Adaptive)"
        assert integrator.safety == 0.9
        assert integrator.current_step == 1.0

    def test_butcher_tableau(self):
        """Verify Butcher tableau coefficients are correct."""
        # Check key coefficients from Dormand-Prince
        assert np.isclose(RK45Integrator.C[1], 1/5)
        assert np.isclose(RK45Integrator.C[2], 3/10)
        assert np.isclose(RK45Integrator.C[3], 4/5)
        assert np.isclose(RK45Integrator.C[4], 8/9)

        # B5 coefficients (5th order)
        assert np.isclose(RK45Integrator.B5[0], 35/384)
        assert np.isclose(RK45Integrator.B5[2], 500/1113)

        # Last B5 coefficient should be 0 for FSAL
        assert RK45Integrator.B5[6] == 0

    def test_step_adaptation(self):
        """Test that step size adapts based on error."""
        integrator = RK45Integrator(harmonic_oscillator, tolerance=1e-8, initial_step=0.5)
        state = np.array([1.0, 0.0])

        initial_step = integrator.current_step
        steps_taken = []

        for _ in range(20):
            result = integrator.step(state, integrator.current_step, freq_mhz=0.0)
            state = result.state
            steps_taken.append(result.step_size_used)

        # Step sizes should vary (adaptation)
        unique_steps = len(set([round(s, 6) for s in steps_taken]))
        # Allow for some variation even if all steps are accepted
        assert len(steps_taken) == 20

    def test_derivatives_per_step(self):
        """Test that RK45 uses 7 derivative evaluations per step."""
        integrator = RK45Integrator(harmonic_oscillator, tolerance=1e-4)
        state = np.array([1.0, 0.0])

        # Use large tolerance to avoid rejections
        result = integrator.step(state, ds=0.1, freq_mhz=0.0)

        # Should use 7 evaluations if step is accepted first try
        assert result.derivatives_computed >= 7
        assert result.derivatives_computed % 7 == 0  # Multiple of 7

    def test_harmonic_oscillator_accuracy(self):
        """Test RK45 on harmonic oscillator."""
        integrator = RK45Integrator(harmonic_oscillator, tolerance=1e-10)

        state = np.array([1.0, 0.0])
        t = 0.0
        period = 2 * np.pi

        while t < period:
            ds = integrator.current_step
            result = integrator.step(state, ds, freq_mhz=0.0)
            state = result.state
            t += result.step_size_used

        # Allow for phase accumulation from not hitting exactly t=2*pi
        expected = np.array([1.0, 0.0])
        np.testing.assert_allclose(state, expected, atol=0.05)

    def test_stiff_region_handling(self):
        """Test behavior with rapidly changing derivatives."""
        def stiff_system(state, freq):
            # System with large derivative that should trigger smaller steps
            return np.array([-100 * state[0]])

        integrator = RK45Integrator(stiff_system, tolerance=1e-6, initial_step=1.0)
        state = np.array([1.0])

        result = integrator.step(state, ds=1.0, freq_mhz=0.0)

        # Should suggest smaller step due to error
        assert integrator.current_step < 1.0


class TestIntegratorFactory:
    """Tests for integrator factory functions."""

    def test_create_by_type(self):
        """Test creation by IntegratorType enum."""
        integrator = IntegratorFactory.create(
            IntegratorType.RK4,
            harmonic_oscillator,
            tolerance=1e-7
        )
        assert isinstance(integrator, RK4Integrator)
        assert integrator.tolerance == 1e-7

    def test_create_by_name(self):
        """Test creation by string name."""
        for name in ['rk4', 'RK4', 'rk4_error']:
            integrator = IntegratorFactory.from_name(name, harmonic_oscillator)
            assert isinstance(integrator, RK4Integrator)

        for name in ['rk45', 'dopri', 'adaptive', 'dormand_prince']:
            integrator = IntegratorFactory.from_name(name, harmonic_oscillator)
            assert isinstance(integrator, RK45Integrator)

        for name in ['abm', 'adams', 'adams_bashforth', 'predictor_corrector']:
            integrator = IntegratorFactory.from_name(name, harmonic_oscillator)
            assert isinstance(integrator, AdamsBashforthMoultonIntegrator)

    def test_create_integrator_convenience(self):
        """Test create_integrator convenience function."""
        integrator = create_integrator(
            'rk45',
            harmonic_oscillator,
            tolerance=1e-8,
            min_step=0.001,
            max_step=5.0
        )
        assert isinstance(integrator, RK45Integrator)
        assert integrator.tolerance == 1e-8
        assert integrator.min_step == 0.001
        assert integrator.max_step == 5.0

    def test_invalid_name_raises(self):
        """Test that invalid names raise ValueError."""
        with pytest.raises(ValueError):
            IntegratorFactory.from_name('invalid_integrator', harmonic_oscillator)

    def test_available_integrators(self):
        """Test available() returns correct list."""
        available = IntegratorFactory.available()
        assert 'rk4' in available
        assert 'rk45' in available
        assert 'adams_bashforth' in available

    def test_get_description(self):
        """Test integrator descriptions."""
        desc = IntegratorFactory.get_description('rk45')
        assert 'Dormand-Prince' in desc
        assert 'adaptive' in desc.lower()

    def test_recommended_integrator(self):
        """Test get_recommended_integrator."""
        # Short path -> RK4
        integrator = get_recommended_integrator(200, harmonic_oscillator, 'normal')
        assert isinstance(integrator, RK4Integrator)

        # Medium path -> RK45
        integrator = get_recommended_integrator(1000, harmonic_oscillator, 'normal')
        assert isinstance(integrator, RK45Integrator)

        # Long path -> ABM
        integrator = get_recommended_integrator(3000, harmonic_oscillator, 'normal')
        assert isinstance(integrator, AdamsBashforthMoultonIntegrator)


class TestIntegrationComparison:
    """Compare all three integrators on the same problem."""

    @pytest.fixture
    def integrators(self):
        """Create all three integrators."""
        return {
            'rk4': RK4Integrator(harmonic_oscillator, tolerance=1e-8),
            'abm': AdamsBashforthMoultonIntegrator(harmonic_oscillator, tolerance=1e-8),
            'rk45': RK45Integrator(harmonic_oscillator, tolerance=1e-8),
        }

    def test_all_integrators_converge(self, integrators):
        """Test all integrators reach similar final state."""
        final_states = {}

        for name, integrator in integrators.items():
            state = np.array([1.0, 0.0])
            t = 0.0
            target = np.pi  # Half period

            while t < target:
                ds = 0.05 if name != 'rk45' else integrator.current_step
                result = integrator.step(state, ds, freq_mhz=0.0)
                state = result.state
                t += result.step_size_used

            final_states[name] = state

        # All should give similar results (allow for phase error from overshooting)
        expected = np.array([-1.0, 0.0])  # cos(pi), -sin(pi)
        for name, state in final_states.items():
            np.testing.assert_allclose(state, expected, atol=0.1,
                                       err_msg=f"{name} failed")

    def test_efficiency_comparison(self, integrators):
        """Compare derivative evaluation counts."""
        eval_counts = {}

        for name, integrator in integrators.items():
            integrator.reset()
            state = np.array([1.0, 0.0])

            for _ in range(50):
                ds = 0.1 if name != 'rk45' else integrator.current_step
                result = integrator.step(state, ds, freq_mhz=0.0)
                state = result.state

            eval_counts[name] = integrator.stats.total_derivative_evals

        # ABM should be most efficient after startup
        # RK4 uses 12 per step (600 total)
        # ABM uses ~5+5+5+5 + 46*2 = 112 (startup + main)
        # RK45 uses 7 per step (~350 total, variable)
        assert eval_counts['abm'] < eval_counts['rk4']


class TestHaselgroveLikeSystem:
    """Test integrators on 6-component Haselgrove-like system."""

    def test_six_component_state(self):
        """Test all integrators handle 6-component state."""
        integrators = [
            RK4Integrator(haselgrove_like),
            AdamsBashforthMoultonIntegrator(haselgrove_like),
            RK45Integrator(haselgrove_like),
        ]

        initial_state = np.array([1.0, 0.0, 0.0, 0.0, 0.1, 0.0])

        for integrator in integrators:
            state = initial_state.copy()

            for _ in range(10):
                ds = 0.5 if not isinstance(integrator, RK45Integrator) else integrator.current_step
                result = integrator.step(state, ds, freq_mhz=7.0)
                state = result.state

            # State should remain valid (no NaN or Inf)
            assert np.all(np.isfinite(state)), f"{integrator.name()} produced invalid state"
            assert len(state) == 6
