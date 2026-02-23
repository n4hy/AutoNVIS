"""
Dormand-Prince RK45 Adaptive Step Integrator

Implements the Dormand-Prince embedded 4(5) Runge-Kutta method
with automatic step size control.

This is the same method used in scipy.integrate.solve_ivp with
method='RK45', and matches the C++ implementation in ray_tracer_3d.cpp.

Features:
- Embedded 4th and 5th order solutions for error estimation
- Automatic step size adaptation based on local error
- FSAL (First Same As Last) optimization possible
- 7 function evaluations per step

The method uses the embedded error estimate:
    error = |y5 - y4|

And adjusts step size via:
    h_new = safety * h * (tolerance / error)^(1/5)

Reference:
- Dormand & Prince (1980)
- IONORT paper Section 2.2
- src/propagation/src/ray_tracer_3d.cpp lines 506-575
"""

import numpy as np
from typing import Callable, Optional, Tuple

from .base import BaseIntegrator, IntegrationStep, IntegrationStats


class RK45Integrator(BaseIntegrator):
    """
    Dormand-Prince RK45 adaptive step integrator.

    Uses embedded 4th/5th order pair for automatic error control
    and step size adaptation. This is the gold standard for
    non-stiff ODE integration.

    Butcher Tableau (Dormand-Prince):
        0    |
        1/5  | 1/5
        3/10 | 3/40       9/40
        4/5  | 44/45      -56/15      32/9
        8/9  | 19372/6561 -25360/2187 64448/6561 -212/729
        1    | 9017/3168  -355/33     46732/5247 49/176    -5103/18656
        1    | 35/384     0           500/1113   125/192   -2187/6784   11/84
        -----+------------------------------------------------------------
        5th  | 35/384     0           500/1113   125/192   -2187/6784   11/84     0
        4th  | 5179/57600 0           7571/16695 393/640   -92097/339200 187/2100 1/40

    Attributes:
        derivative_func: Function dy/ds = f(y, freq)
        tolerance: Local error tolerance
        safety: Safety factor for step size control (default 0.9)
        current_step: Current/suggested step size

    Example:
        rk45 = RK45Integrator(haselgrove_derivs, tolerance=1e-7)

        state = np.array([x, y, z, kx, ky, kz])
        result = rk45.step(state, ds=1.0, freq_mhz=7.0)

        # Use suggested step for next iteration
        next_step = rk45.current_step
    """

    # Dormand-Prince coefficients
    # Time coefficients (c)
    C = np.array([0.0, 1/5, 3/10, 4/5, 8/9, 1.0, 1.0])

    # Stage coefficients (A matrix, lower triangular)
    A = np.array([
        [0, 0, 0, 0, 0, 0],
        [1/5, 0, 0, 0, 0, 0],
        [3/40, 9/40, 0, 0, 0, 0],
        [44/45, -56/15, 32/9, 0, 0, 0],
        [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0],
        [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]
    ])

    # 5th order weights (b5)
    B5 = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0])

    # 4th order weights (b4) for error estimation
    B4 = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])

    # Error weights (b5 - b4)
    E = B5 - B4

    def __init__(
        self,
        derivative_func: Callable[[np.ndarray, float], np.ndarray],
        tolerance: float = 1e-7,
        min_step: float = 0.01,
        max_step: float = 20.0,
        safety: float = 0.9,
        initial_step: float = 1.0,
    ):
        """
        Initialize RK45 integrator.

        Args:
            derivative_func: Function computing dy/ds given (state, freq)
            tolerance: Local error tolerance
            min_step: Minimum step size (km)
            max_step: Maximum step size (km)
            safety: Safety factor for step adaptation (0 < safety < 1)
            initial_step: Initial step size (km)
        """
        super().__init__(derivative_func, tolerance, min_step, max_step)
        self.safety = safety
        self.current_step = initial_step
        self._last_k1: Optional[np.ndarray] = None  # For FSAL optimization

    def name(self) -> str:
        """Return integrator name."""
        return "RK45 Dormand-Prince (Adaptive)"

    def reset(self) -> None:
        """Reset integrator for new ray trace."""
        super().reset()
        self._last_k1 = None

    def step(
        self,
        state: np.ndarray,
        ds: float,
        freq_mhz: float,
    ) -> IntegrationStep:
        """
        Perform adaptive RK45 step.

        Attempts to take a step of size ds, but may reduce step size
        if error exceeds tolerance. Updates self.current_step with
        optimal step for next iteration.

        Args:
            state: Current state [x, y, z, kx, ky, kz]
            ds: Requested step size (km)
            freq_mhz: Wave frequency (MHz)

        Returns:
            IntegrationStep with new state and diagnostics

        Note:
            If step is rejected (error > tolerance), the returned
            state will still be the best attempt. Check result.accepted
            to determine if step should be retried with smaller ds.
        """
        # Start with requested step (clamped)
        h = np.clip(ds, self.min_step, self.max_step)

        max_attempts = 10
        total_derivs = 0

        for attempt in range(max_attempts):
            # Compute RK45 step
            y5, y4, k, derivs = self._rk45_step(state, h, freq_mhz)
            total_derivs += derivs

            # Compute error estimate
            error = np.linalg.norm(y5 - y4)

            # Compute optimal step size
            if error > 0:
                # h_opt = safety * h * (tol / error)^(1/5) for RK5
                factor = self.safety * (self.tolerance / error) ** 0.2
                factor = np.clip(factor, 0.1, 5.0)  # Limit change rate
            else:
                factor = 5.0  # Error is zero, can increase step

            h_new = np.clip(h * factor, self.min_step, self.max_step)

            # Accept step?
            if error <= self.tolerance or h <= self.min_step:
                # Accept this step
                self.current_step = h_new

                result = IntegrationStep(
                    state=y5,  # Use 5th order solution
                    error_estimate=error,
                    step_size_used=h,
                    derivatives_computed=total_derivs,
                    accepted=True,
                )
                self.stats.update(result)

                # Save last k7 for FSAL (if implemented)
                self._last_k1 = k[6] if len(k) > 6 else None

                return result

            # Reject step, try with smaller h
            h = max(h_new, self.min_step)
            self.stats.rejected_steps += 1

        # Failed to converge after max attempts
        # Return best result with minimum step
        y5, y4, k, derivs = self._rk45_step(state, self.min_step, freq_mhz)
        error = np.linalg.norm(y5 - y4)

        result = IntegrationStep(
            state=y5,
            error_estimate=error,
            step_size_used=self.min_step,
            derivatives_computed=total_derivs + derivs,
            accepted=False,
        )
        self.stats.update(result)

        return result

    def _rk45_step(
        self,
        y: np.ndarray,
        h: float,
        freq: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Compute single RK45 step.

        Args:
            y: Current state
            h: Step size
            freq: Frequency parameter

        Returns:
            Tuple of (y5, y4, k_stages, num_derivs):
            - y5: 5th order solution
            - y4: 4th order solution (for error)
            - k_stages: Array of k values
            - num_derivs: Number of derivative evaluations
        """
        n = len(y)
        k = np.zeros((7, n))

        # Stage 1: k1 = f(y)
        k[0] = self.derivative_func(y, freq)

        # Stages 2-7
        for i in range(1, 7):
            y_stage = y + h * sum(self.A[i][j] * k[j] for j in range(i))
            k[i] = self.derivative_func(y_stage, freq)

        # 5th order solution
        y5 = y + h * sum(self.B5[i] * k[i] for i in range(7))

        # 4th order solution (for error estimation)
        y4 = y + h * sum(self.B4[i] * k[i] for i in range(7))

        return y5, y4, k, 7

    def suggested_step(self, current_step: float) -> float:
        """
        Return suggested step size for next iteration.

        Args:
            current_step: Current step size (ignored, uses internal)

        Returns:
            Optimal step size based on last error estimate
        """
        return self.current_step

    def step_with_rejection_count(
        self,
        state: np.ndarray,
        ds: float,
        freq_mhz: float,
    ) -> Tuple[IntegrationStep, int]:
        """
        Perform step and return number of rejections.

        Useful for monitoring step size adaptation behavior.

        Args:
            state: Current state
            ds: Requested step size
            freq_mhz: Frequency

        Returns:
            Tuple of (IntegrationStep, num_rejections)
        """
        rejections_before = self.stats.rejected_steps
        result = self.step(state, ds, freq_mhz)
        rejections = self.stats.rejected_steps - rejections_before

        return result, rejections


class RK45IntegratorFast(RK45Integrator):
    """
    Fast RK45 variant with relaxed error tolerance.

    Uses larger default tolerance and step sizes for faster
    computation when high precision is not required.
    """

    def __init__(
        self,
        derivative_func: Callable[[np.ndarray, float], np.ndarray],
        tolerance: float = 1e-5,
        min_step: float = 0.1,
        max_step: float = 50.0,
        **kwargs,
    ):
        super().__init__(
            derivative_func,
            tolerance=tolerance,
            min_step=min_step,
            max_step=max_step,
            **kwargs,
        )

    def name(self) -> str:
        return "RK45 Dormand-Prince (Fast)"
