"""
Adams-Bashforth/Adams-Moulton Predictor-Corrector Integrator

Implements the AB4/AM3 predictor-corrector pair as described in
IONORT paper Section 2.2.

Predictor (Adams-Bashforth 4-step):
    y_{n+1}^p = y_n + h/24 * (55*f_n - 59*f_{n-1} + 37*f_{n-2} - 9*f_{n-3})

Corrector (Adams-Moulton 3-step):
    y_{n+1} = y_n + h/24 * (9*f(y_{n+1}^p) + 19*f_n - 5*f_{n-1} + f_{n-2})

This method requires 4 previous derivative values to start, so it
uses RK4 for the first 4 steps (startup phase).

Advantages:
- Only 2 derivative evaluations per step (vs 4-12 for RK4)
- Higher order accuracy when path is smooth
- Efficient for long ray paths

Reference: IONORT paper Section 2.2, Hairer et al. (1993)
"""

import numpy as np
from typing import Callable, List, Optional

from .base import BaseIntegrator, IntegrationStep, IntegrationStats
from .rk4 import RK4IntegratorFast


class AdamsBashforthMoultonIntegrator(BaseIntegrator):
    """
    Adams-Bashforth 4-step / Adams-Moulton 3-step predictor-corrector.

    This is a multistep method that uses the history of previous
    derivative values to achieve high accuracy with fewer function
    evaluations per step.

    PECE Mode (Predict-Evaluate-Correct-Evaluate):
    1. Predict: Use AB4 to estimate y_{n+1}
    2. Evaluate: Compute f(y_{n+1}^p)
    3. Correct: Use AM3 to refine y_{n+1}
    4. (Optionally) Evaluate: Compute f(y_{n+1}) for error

    Startup:
    - First 4 steps use RK4 to build derivative history
    - After startup, uses only 2 derivative evaluations per step

    Error Estimation:
    - PECE error ≈ (y_corrected - y_predicted) * 9/270

    Attributes:
        derivative_func: Function dy/ds = f(y, freq)
        tolerance: Error tolerance
        history: List of past 4 derivative values
        step_history: List of past step sizes (for variable step)

    Example:
        abm = AdamsBashforthMoultonIntegrator(haselgrove_derivs)

        # Reset before each new ray trace!
        abm.reset()

        for i in range(num_steps):
            result = abm.step(state, ds=1.0, freq_mhz=7.0)
            state = result.state

        print(f"Total derivative evals: {abm.stats.total_derivative_evals}")
    """

    # Adams-Bashforth 4-step coefficients: (55, -59, 37, -9) / 24
    AB4_COEFFS = np.array([55.0, -59.0, 37.0, -9.0]) / 24.0

    # Adams-Moulton 3-step coefficients: (9, 19, -5, 1) / 24
    # Note: first coefficient is for f(y_{n+1}^p), rest for history
    AM3_COEFFS = np.array([9.0, 19.0, -5.0, 1.0]) / 24.0

    # PECE error constant
    PECE_ERROR_CONST = 9.0 / 270.0

    def __init__(
        self,
        derivative_func: Callable[[np.ndarray, float], np.ndarray],
        tolerance: float = 1e-6,
        min_step: float = 0.01,
        max_step: float = 10.0,
    ):
        """
        Initialize Adams-Bashforth/Moulton integrator.

        Args:
            derivative_func: Function computing dy/ds given (state, freq)
            tolerance: Error tolerance
            min_step: Minimum step size (km)
            max_step: Maximum step size (km)
        """
        super().__init__(derivative_func, tolerance, min_step, max_step)

        # History of derivative values (most recent first)
        self.history: List[np.ndarray] = []

        # History of step sizes (for potential variable-step extension)
        self.step_history: List[float] = []

        # Startup integrator (RK4)
        self._startup_integrator = RK4IntegratorFast(
            derivative_func, tolerance, min_step, max_step
        )

        # Flag for startup complete
        self._startup_complete = False

    def name(self) -> str:
        """Return integrator name."""
        return "Adams-Bashforth-Moulton (AB4/AM3 Predictor-Corrector)"

    def reset(self) -> None:
        """
        Reset integrator for new ray trace.

        IMPORTANT: Must call this before tracing a new ray!
        Clears the derivative history and resets startup state.
        """
        super().reset()
        self.history.clear()
        self.step_history.clear()
        self._startup_complete = False
        self._startup_integrator.reset_stats()

    def step(
        self,
        state: np.ndarray,
        ds: float,
        freq_mhz: float,
    ) -> IntegrationStep:
        """
        Perform AB4/AM3 predictor-corrector step.

        During startup (first 4 steps), uses RK4 to build history.
        After startup, uses efficient AB4/AM3 with only 2 derivative
        evaluations per step.

        Args:
            state: Current state [x, y, z, kx, ky, kz]
            ds: Step size (km)
            freq_mhz: Wave frequency (MHz)

        Returns:
            IntegrationStep with new state and error estimate
        """
        h = np.clip(ds, self.min_step, self.max_step)

        # Startup phase: use RK4 to build history
        if not self._startup_complete:
            return self._startup_step(state, h, freq_mhz)

        # Main AB4/AM3 method
        return self._abm_step(state, h, freq_mhz)

    def _startup_step(
        self,
        state: np.ndarray,
        h: float,
        freq_mhz: float,
    ) -> IntegrationStep:
        """
        Startup step using RK4 to build derivative history.

        Args:
            state: Current state
            h: Step size
            freq_mhz: Frequency

        Returns:
            IntegrationStep from RK4
        """
        # Compute derivative at current state
        f_n = self.derivative_func(state, freq_mhz)

        # Store in history
        self.history.insert(0, f_n)
        self.step_history.insert(0, h)

        # Trim to 4 entries
        if len(self.history) > 4:
            self.history.pop()
            self.step_history.pop()

        # Use RK4 for the step
        result = self._startup_integrator.step(state, h, freq_mhz)

        # Check if startup is complete
        if len(self.history) >= 4:
            self._startup_complete = True

        # Update our stats
        self.stats.total_steps += 1
        self.stats.total_derivative_evals += result.derivatives_computed + 1  # +1 for f_n

        return IntegrationStep(
            state=result.state,
            error_estimate=result.error_estimate,
            step_size_used=h,
            derivatives_computed=result.derivatives_computed + 1,
            accepted=result.accepted,
        )

    def _abm_step(
        self,
        state: np.ndarray,
        h: float,
        freq_mhz: float,
    ) -> IntegrationStep:
        """
        Main Adams-Bashforth/Moulton predictor-corrector step.

        PECE implementation:
        1. Predict with AB4
        2. Evaluate f at prediction
        3. Correct with AM3
        4. Use |correction - prediction| for error estimate

        Args:
            state: Current state
            h: Step size
            freq_mhz: Frequency

        Returns:
            IntegrationStep with corrected state
        """
        # Compute derivative at current state
        f_n = self.derivative_func(state, freq_mhz)

        # Adams-Bashforth 4-step predictor (explicit)
        # y_{n+1}^p = y_n + h/24 * (55*f_n - 59*f_{n-1} + 37*f_{n-2} - 9*f_{n-3})
        y_pred = state + h * (
            self.AB4_COEFFS[0] * f_n +
            self.AB4_COEFFS[1] * self.history[0] +
            self.AB4_COEFFS[2] * self.history[1] +
            self.AB4_COEFFS[3] * self.history[2]
        )

        # Evaluate derivative at prediction
        f_pred = self.derivative_func(y_pred, freq_mhz)

        # Adams-Moulton 3-step corrector (implicit, one iteration)
        # y_{n+1} = y_n + h/24 * (9*f(y_{n+1}^p) + 19*f_n - 5*f_{n-1} + f_{n-2})
        y_corr = state + h * (
            self.AM3_COEFFS[0] * f_pred +
            self.AM3_COEFFS[1] * f_n +
            self.AM3_COEFFS[2] * self.history[0] +
            self.AM3_COEFFS[3] * self.history[1]
        )

        # Error estimate: PECE error constant * |corrector - predictor|
        error = np.linalg.norm(y_corr - y_pred) * self.PECE_ERROR_CONST

        # Update history (shift and insert new derivative)
        self.history.insert(0, f_n)
        self.step_history.insert(0, h)
        if len(self.history) > 4:
            self.history.pop()
            self.step_history.pop()

        # Check acceptance
        accepted = error <= self.tolerance

        # Create result
        result = IntegrationStep(
            state=y_corr,
            error_estimate=error,
            step_size_used=h,
            derivatives_computed=2,  # f_n and f_pred
            accepted=accepted,
        )

        # Update statistics
        self.stats.update(result)

        return result

    @property
    def is_startup_complete(self) -> bool:
        """Check if startup phase is complete."""
        return self._startup_complete

    @property
    def history_size(self) -> int:
        """Current history size."""
        return len(self.history)

    def efficiency_ratio(self) -> float:
        """
        Compute efficiency ratio vs standard RK4.

        Returns:
            Ratio of RK4 derivative evals to ABM derivative evals
            for equivalent number of steps.
        """
        if self.stats.total_steps == 0:
            return 1.0

        # RK4 with doubling uses 12 evals per step
        # ABM uses 2 per step after startup
        rk4_evals = self.stats.total_steps * 12
        abm_evals = self.stats.total_derivative_evals

        return rk4_evals / abm_evals if abm_evals > 0 else 1.0
