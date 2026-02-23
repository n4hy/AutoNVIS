"""
Enhanced 4th-Order Runge-Kutta Integrator

Implements classical RK4 with step doubling for error estimation,
providing reliable error tracking for ray tracing applications.

Error Estimation Method:
    y1 = RK4(y, h)              # One full step
    y2 = RK4(RK4(y, h/2), h/2)  # Two half steps
    error = |y1 - y2| / 15      # Richardson extrapolation

The more accurate solution (two half steps) is returned.

Reference: IONORT paper Section 2.2, Butcher (2003)
"""

import numpy as np
from typing import Callable, Optional

from .base import BaseIntegrator, IntegrationStep, IntegrationStats


class RK4Integrator(BaseIntegrator):
    """
    Classical 4th-order Runge-Kutta with error tracking.

    Uses step doubling to estimate local truncation error:
    - Compute solution with one step of size h
    - Compute solution with two steps of size h/2
    - Error ≈ |y_h - y_{h/2}| / 15 (Richardson extrapolation)

    The two-half-step solution (more accurate) is returned.

    This requires 12 derivative evaluations per step (4 + 4 + 4)
    but provides reliable error estimates for monitoring integration
    quality.

    Attributes:
        derivative_func: Function dy/ds = f(y, freq)
        tolerance: Error tolerance (for reporting, not adaptive)
        min_step: Minimum step size (km)
        max_step: Maximum step size (km)

    Example:
        def haselgrove_derivs(state, freq):
            # ... compute derivatives ...
            return np.array([dx, dy, dz, dkx, dky, dkz])

        rk4 = RK4Integrator(haselgrove_derivs, tolerance=1e-6)

        state = np.array([x, y, z, kx, ky, kz])
        result = rk4.step(state, ds=1.0, freq_mhz=7.0)

        print(f"Error estimate: {result.error_estimate:.2e}")
        state = result.state
    """

    def __init__(
        self,
        derivative_func: Callable[[np.ndarray, float], np.ndarray],
        tolerance: float = 1e-6,
        min_step: float = 0.01,
        max_step: float = 10.0,
    ):
        """
        Initialize RK4 integrator.

        Args:
            derivative_func: Function computing dy/ds given (state, freq)
            tolerance: Error tolerance for quality monitoring
            min_step: Minimum step size (km)
            max_step: Maximum step size (km)
        """
        super().__init__(derivative_func, tolerance, min_step, max_step)

    def name(self) -> str:
        """Return integrator name."""
        return "RK4 (Classical 4th Order with Error Tracking)"

    def step(
        self,
        state: np.ndarray,
        ds: float,
        freq_mhz: float,
    ) -> IntegrationStep:
        """
        Perform RK4 step with error estimation via step doubling.

        Uses Richardson extrapolation:
            y1 = RK4(y, h)
            y2 = RK4(RK4(y, h/2), h/2)
            error = |y1 - y2| / 15

        Returns the more accurate two-half-step solution.

        Args:
            state: Current state [x, y, z, kx, ky, kz]
            ds: Step size (km)
            freq_mhz: Wave frequency (MHz)

        Returns:
            IntegrationStep with new state and error estimate
        """
        # Clamp step size
        h = np.clip(ds, self.min_step, self.max_step)

        # Single full step
        y_full = self._rk4_single(state, h, freq_mhz)

        # Two half steps (more accurate)
        y_half1 = self._rk4_single(state, h / 2, freq_mhz)
        y_half2 = self._rk4_single(y_half1, h / 2, freq_mhz)

        # Error estimate via Richardson extrapolation
        # For RK4 (order 4), error ≈ (y_h - y_{h/2}) / (2^4 - 1) = (y_h - y_{h/2}) / 15
        error = np.linalg.norm(y_full - y_half2) / 15.0

        # Check if error is acceptable
        accepted = error <= self.tolerance

        # Create result (use more accurate half-step solution)
        result = IntegrationStep(
            state=y_half2,
            error_estimate=error,
            step_size_used=h,
            derivatives_computed=12,  # 4 + 4 + 4
            accepted=accepted,
        )

        # Update statistics
        self.stats.update(result)

        return result

    def _rk4_single(
        self,
        y: np.ndarray,
        h: float,
        freq: float,
    ) -> np.ndarray:
        """
        Perform single RK4 step.

        Classical 4th-order Runge-Kutta:
            k1 = f(y)
            k2 = f(y + h/2 * k1)
            k3 = f(y + h/2 * k2)
            k4 = f(y + h * k3)
            y_new = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)

        Args:
            y: Current state vector
            h: Step size
            freq: Frequency parameter

        Returns:
            New state vector
        """
        k1 = self.derivative_func(y, freq)
        k2 = self.derivative_func(y + 0.5 * h * k1, freq)
        k3 = self.derivative_func(y + 0.5 * h * k2, freq)
        k4 = self.derivative_func(y + h * k3, freq)

        return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def step_no_error(
        self,
        state: np.ndarray,
        ds: float,
        freq_mhz: float,
    ) -> IntegrationStep:
        """
        Perform RK4 step without error estimation (faster).

        Uses only 4 derivative evaluations. Use when error tracking
        is not needed.

        Args:
            state: Current state [x, y, z, kx, ky, kz]
            ds: Step size (km)
            freq_mhz: Wave frequency (MHz)

        Returns:
            IntegrationStep without error estimate
        """
        h = np.clip(ds, self.min_step, self.max_step)
        y_new = self._rk4_single(state, h, freq_mhz)

        result = IntegrationStep(
            state=y_new,
            error_estimate=0.0,  # Not computed
            step_size_used=h,
            derivatives_computed=4,
            accepted=True,
        )

        self.stats.total_steps += 1
        self.stats.total_derivative_evals += 4

        return result


class RK4IntegratorFast(BaseIntegrator):
    """
    Fast RK4 integrator without error tracking.

    Uses standard 4-stage RK4 with only 4 derivative evaluations
    per step. Suitable when error tracking is not required.

    This matches the original haselgrove.py implementation
    but in a modular, pluggable form.
    """

    def name(self) -> str:
        return "RK4 Fast (No Error Tracking)"

    def step(
        self,
        state: np.ndarray,
        ds: float,
        freq_mhz: float,
    ) -> IntegrationStep:
        """
        Perform standard RK4 step.

        Args:
            state: Current state [x, y, z, kx, ky, kz]
            ds: Step size (km)
            freq_mhz: Wave frequency (MHz)

        Returns:
            IntegrationStep with new state
        """
        h = np.clip(ds, self.min_step, self.max_step)

        k1 = self.derivative_func(state, freq_mhz)
        k2 = self.derivative_func(state + 0.5 * h * k1, freq_mhz)
        k3 = self.derivative_func(state + 0.5 * h * k2, freq_mhz)
        k4 = self.derivative_func(state + h * k3, freq_mhz)

        y_new = state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        result = IntegrationStep(
            state=y_new,
            error_estimate=0.0,
            step_size_used=h,
            derivatives_computed=4,
            accepted=True,
        )

        self.stats.total_steps += 1
        self.stats.total_derivative_evals += 4

        return result
