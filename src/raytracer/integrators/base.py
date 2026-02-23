"""
Base Classes for Ray Equation Integrators

Provides abstract base class and data structures for all numerical
integrators used in Haselgrove ray tracing.

The Haselgrove system is 6 coupled first-order ODEs:
    dr/ds = (c/n) * k_hat     (position evolution)
    dk/ds = -grad(n) / n      (wave vector evolution)

where s = path length, n = refractive index, k = wave vector.

Reference: IONORT paper Section 2, Jones & Stephenson (1975)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np


@dataclass
class IntegrationStep:
    """
    Result of a single integration step.

    Attributes:
        state: New state vector [x, y, z, kx, ky, kz] in km and normalized
        error_estimate: Local truncation error estimate
        step_size_used: Actual step size taken (km)
        derivatives_computed: Number of derivative evaluations
        accepted: Whether step was accepted (for adaptive methods)
    """
    state: np.ndarray
    error_estimate: float
    step_size_used: float
    derivatives_computed: int
    accepted: bool = True

    def __repr__(self) -> str:
        return (f"IntegrationStep(error={self.error_estimate:.2e}, "
                f"ds={self.step_size_used:.3f}km, "
                f"derivs={self.derivatives_computed}, "
                f"accepted={self.accepted})")


@dataclass
class IntegrationStats:
    """
    Statistics for a complete integration (ray trace).

    Attributes:
        total_steps: Total number of integration steps taken
        rejected_steps: Steps rejected and retried (adaptive methods)
        total_derivative_evals: Total derivative function calls
        max_error: Maximum error estimate encountered
        min_step_size: Smallest step size used (km)
        max_step_size: Largest step size used (km)
    """
    total_steps: int = 0
    rejected_steps: int = 0
    total_derivative_evals: int = 0
    max_error: float = 0.0
    min_step_size: float = float('inf')
    max_step_size: float = 0.0

    def update(self, step_result: IntegrationStep) -> None:
        """Update statistics with a step result."""
        self.total_steps += 1
        self.total_derivative_evals += step_result.derivatives_computed
        self.max_error = max(self.max_error, step_result.error_estimate)
        self.min_step_size = min(self.min_step_size, step_result.step_size_used)
        self.max_step_size = max(self.max_step_size, step_result.step_size_used)
        if not step_result.accepted:
            self.rejected_steps += 1

    def efficiency(self) -> float:
        """
        Compute integration efficiency.

        Returns:
            Ratio of accepted steps to total attempts
        """
        if self.total_steps == 0:
            return 1.0
        return (self.total_steps - self.rejected_steps) / self.total_steps

    def __repr__(self) -> str:
        return (f"IntegrationStats(steps={self.total_steps}, "
                f"rejected={self.rejected_steps}, "
                f"derivs={self.total_derivative_evals}, "
                f"max_error={self.max_error:.2e})")


class BaseIntegrator(ABC):
    """
    Abstract base class for ray equation integrators.

    All integrators solve the Haselgrove system:
        dy/ds = f(y)

    where y = [x, y, z, kx, ky, kz] is the 6-component state vector
    and s is the path length along the ray.

    The derivative function f(y) computes:
        - Position derivatives: dr/ds = k_hat / n(r)
        - Wave vector derivatives: dk/ds = -grad(n) / n

    Subclasses must implement:
        - step(): Perform one integration step
        - name(): Return integrator name for logging

    Attributes:
        derivative_func: Function dy/ds = f(y, freq) returning 6-vector
        tolerance: Local error tolerance for adaptive methods
        min_step: Minimum allowed step size (km)
        max_step: Maximum allowed step size (km)
        stats: Integration statistics

    Example:
        def my_derivative(state: np.ndarray, freq: float) -> np.ndarray:
            # Compute Haselgrove derivatives
            return np.array([dx_ds, dy_ds, dz_ds, dkx_ds, dky_ds, dkz_ds])

        integrator = RK4Integrator(my_derivative, tolerance=1e-6)

        state = np.array([x0, y0, z0, kx0, ky0, kz0])
        result = integrator.step(state, ds=1.0, freq_mhz=7.0)

        if result.accepted:
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
        Initialize integrator.

        Args:
            derivative_func: Function computing dy/ds given (state, frequency).
                            Must return 6-element numpy array.
            tolerance: Local error tolerance (used by adaptive methods)
            min_step: Minimum step size in km
            max_step: Maximum step size in km
        """
        self.derivative_func = derivative_func
        self.tolerance = tolerance
        self.min_step = min_step
        self.max_step = max_step
        self.stats = IntegrationStats()

    @abstractmethod
    def step(
        self,
        state: np.ndarray,
        ds: float,
        freq_mhz: float,
    ) -> IntegrationStep:
        """
        Perform single integration step.

        Args:
            state: Current state [x, y, z, kx, ky, kz]
            ds: Requested step size (km)
            freq_mhz: Wave frequency for refractive index calculation

        Returns:
            IntegrationStep with new state and diagnostics
        """
        pass

    @abstractmethod
    def name(self) -> str:
        """
        Return integrator name for logging.

        Returns:
            Human-readable integrator name
        """
        pass

    def reset_stats(self) -> None:
        """Reset integration statistics for new ray trace."""
        self.stats = IntegrationStats()

    def reset(self) -> None:
        """
        Reset integrator state for new ray trace.

        Override in subclasses that maintain internal state
        (e.g., Adams-Bashforth history).
        """
        self.reset_stats()

    def suggested_step(self, current_step: float) -> float:
        """
        Suggest step size for next integration.

        Override in adaptive methods to provide optimal step.

        Args:
            current_step: Current step size

        Returns:
            Suggested step size for next step
        """
        return current_step

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(tol={self.tolerance}, min={self.min_step}, max={self.max_step})"
