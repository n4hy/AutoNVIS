"""
Rauch-Tung-Striebel (RTS) Smoother for Auto-NVIS

Implements the backward pass of the RTS fixed-interval smoother
for improved state estimates using future observations.

The smoother refines filter estimates by incorporating information
from subsequent measurements, providing:
- Reduced estimation uncertainty
- Improved historical state reconstruction
- Better analysis fields for validation

Algorithm:
    Forward pass: Standard SR-UKF filter (done in autonvis_filter.py)
    Backward pass: RTS smoother equations

    For k = N-1, ..., 0:
        Gₖ = Sₖ @ Sₖ₊₁⁻ᵀ              # Smoother gain
        x̂ˢₖ = x̂ₖ + Gₖ(x̂ˢₖ₊₁ - x̂⁻ₖ₊₁)  # Smoothed state
        Sˢₖ = Sₖ @ (I + Gₖᵀ(Sˢₖ₊₁Sˢₖ₊₁ᵀ - Sₖ₊₁⁻Sₖ₊₁⁻ᵀ)Gₖ)^(1/2)

Where:
    x̂ₖ = Filter posterior mean at time k
    x̂⁻ₖ = Filter prior (predicted) mean at time k
    Sₖ = Filter posterior sqrt covariance
    S⁻ₖ = Filter prior sqrt covariance
    x̂ˢₖ = Smoothed state estimate
    Sˢₖ = Smoothed sqrt covariance
    Gₖ = Smoother gain matrix

References:
    Rauch, Tung, & Striebel (1965), "Maximum likelihood estimates of linear
    dynamic systems", AIAA Journal
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)


@dataclass
class SmootherState:
    """
    State container for smoother backward pass.

    Stores both prior and posterior estimates needed for
    RTS smoother gain calculation.
    """
    state_prior: np.ndarray      # x̂⁻ₖ (predicted state)
    state_posterior: np.ndarray  # x̂ₖ (updated state)
    sqrt_cov_prior: np.ndarray   # S⁻ₖ (predicted sqrt covariance)
    sqrt_cov_posterior: np.ndarray  # Sₖ (updated sqrt covariance)
    timestamp: datetime
    cycle_index: int = 0

    def __post_init__(self):
        """Validate state dimensions match"""
        if self.state_prior.shape != self.state_posterior.shape:
            raise ValueError("Prior and posterior state dimensions must match")
        if self.sqrt_cov_prior.shape != self.sqrt_cov_posterior.shape:
            raise ValueError("Prior and posterior covariance dimensions must match")


@dataclass
class SmoothedResult:
    """Result from RTS smoother backward pass"""
    state: np.ndarray            # Smoothed state estimate
    sqrt_cov: np.ndarray         # Smoothed sqrt covariance
    timestamp: datetime
    cycle_index: int
    gain: Optional[np.ndarray] = None  # Smoother gain (optional, for diagnostics)


class RTSSmoother:
    """
    Rauch-Tung-Striebel Fixed-Interval Smoother

    Implements backward pass smoothing using square-root formulation
    for numerical stability.

    Usage:
        smoother = RTSSmoother()

        # During forward pass, store states
        for each_cycle:
            smoother.store_state(prior, posterior, sqrt_cov_prior, sqrt_cov_posterior)

        # Run backward pass when ready
        smoothed = smoother.run_backward_pass()

        # Get smoothed state at specific time
        smoothed_state = smoothed[-1].state  # Latest smoothed
    """

    def __init__(
        self,
        max_lag: int = 3,
        regularization: float = 1e-10
    ):
        """
        Initialize RTS smoother.

        Args:
            max_lag: Maximum smoothing lag (number of future steps)
            regularization: Regularization for matrix inversion
        """
        self.max_lag = max_lag
        self.regularization = regularization

        # State history for backward pass
        self.history: List[SmootherState] = []

        # Cached smoothed results
        self._smoothed_cache: Optional[List[SmoothedResult]] = None

        logger.info(f"RTSSmoother initialized: max_lag={max_lag}")

    def store_state(
        self,
        state_prior: np.ndarray,
        state_posterior: np.ndarray,
        sqrt_cov_prior: np.ndarray,
        sqrt_cov_posterior: np.ndarray,
        timestamp: Optional[datetime] = None,
        cycle_index: int = 0
    ):
        """
        Store filter state for later smoothing.

        Should be called after each filter update cycle with both
        prior (predicted) and posterior (updated) estimates.

        Args:
            state_prior: Predicted state before update
            state_posterior: Updated state after measurement assimilation
            sqrt_cov_prior: Predicted sqrt covariance
            sqrt_cov_posterior: Updated sqrt covariance
            timestamp: Time of this estimate
            cycle_index: Filter cycle number
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        state = SmootherState(
            state_prior=state_prior.copy(),
            state_posterior=state_posterior.copy(),
            sqrt_cov_prior=sqrt_cov_prior.copy(),
            sqrt_cov_posterior=sqrt_cov_posterior.copy(),
            timestamp=timestamp,
            cycle_index=cycle_index
        )

        self.history.append(state)

        # Trim to max_lag + 1 (need extra for forward reference)
        if len(self.history) > self.max_lag + 1:
            self.history.pop(0)

        # Invalidate cache
        self._smoothed_cache = None

        logger.debug(f"Stored state for cycle {cycle_index}, history length: {len(self.history)}")

    def run_backward_pass(self) -> List[SmoothedResult]:
        """
        Execute RTS smoother backward pass.

        Processes stored state history from newest to oldest,
        computing smoothed estimates at each time step.

        Returns:
            List of SmoothedResult objects, ordered oldest to newest
        """
        if len(self.history) < 2:
            logger.warning("Insufficient history for smoothing (need >= 2 states)")
            return []

        # Return cached result if available
        if self._smoothed_cache is not None:
            return self._smoothed_cache

        logger.info(f"Running RTS backward pass over {len(self.history)} states")

        n_states = len(self.history)
        smoothed = [None] * n_states

        # Initialize with final state (no future info available)
        final_state = self.history[-1]
        smoothed[-1] = SmoothedResult(
            state=final_state.state_posterior.copy(),
            sqrt_cov=final_state.sqrt_cov_posterior.copy(),
            timestamp=final_state.timestamp,
            cycle_index=final_state.cycle_index,
            gain=None
        )

        # Backward pass: k = N-2, ..., 0
        for k in range(n_states - 2, -1, -1):
            current = self.history[k]
            next_state = self.history[k + 1]
            next_smoothed = smoothed[k + 1]

            try:
                # Compute smoother gain
                # Gₖ = Sₖ @ (S⁻ₖ₊₁)⁻ᵀ = Sₖ @ inv(S⁻ₖ₊₁ᵀ)
                gain = self._compute_gain(
                    current.sqrt_cov_posterior,
                    next_state.sqrt_cov_prior
                )

                # Compute smoothed state
                # x̂ˢₖ = x̂ₖ + Gₖ(x̂ˢₖ₊₁ - x̂⁻ₖ₊₁)
                innovation = next_smoothed.state - next_state.state_prior
                smoothed_state = current.state_posterior + gain @ innovation

                # Compute smoothed covariance (sqrt form)
                smoothed_sqrt_cov = self._compute_smoothed_covariance(
                    current.sqrt_cov_posterior,
                    next_state.sqrt_cov_prior,
                    next_smoothed.sqrt_cov,
                    gain
                )

                smoothed[k] = SmoothedResult(
                    state=smoothed_state,
                    sqrt_cov=smoothed_sqrt_cov,
                    timestamp=current.timestamp,
                    cycle_index=current.cycle_index,
                    gain=gain
                )

            except np.linalg.LinAlgError as e:
                logger.warning(f"Smoother failed at step {k}: {e}")
                # Fall back to filter estimate
                smoothed[k] = SmoothedResult(
                    state=current.state_posterior.copy(),
                    sqrt_cov=current.sqrt_cov_posterior.copy(),
                    timestamp=current.timestamp,
                    cycle_index=current.cycle_index,
                    gain=None
                )

        self._smoothed_cache = smoothed
        logger.info(f"Backward pass complete, {len(smoothed)} smoothed states")

        return smoothed

    def _compute_gain(
        self,
        S_posterior: np.ndarray,
        S_prior_next: np.ndarray
    ) -> np.ndarray:
        """
        Compute RTS smoother gain matrix.

        Gₖ = Sₖ @ (S⁻ₖ₊₁)⁻ᵀ

        Args:
            S_posterior: Current posterior sqrt covariance
            S_prior_next: Next step prior sqrt covariance

        Returns:
            Smoother gain matrix
        """
        # Regularize for numerical stability
        S_prior_reg = S_prior_next + self.regularization * np.eye(S_prior_next.shape[0])

        # Compute inverse transpose
        # G = S_post @ inv(S_prior^T)
        # Solve: S_prior^T @ G^T = S_post^T
        # i.e., G^T = solve(S_prior^T, S_post^T)

        try:
            G_T = np.linalg.solve(S_prior_reg.T, S_posterior.T)
            return G_T.T
        except np.linalg.LinAlgError:
            # Fall back to pseudo-inverse
            logger.warning("Using pseudo-inverse for smoother gain")
            S_prior_inv_T = np.linalg.pinv(S_prior_reg.T)
            return S_posterior @ S_prior_inv_T

    def _compute_smoothed_covariance(
        self,
        S_posterior: np.ndarray,
        S_prior_next: np.ndarray,
        S_smoothed_next: np.ndarray,
        gain: np.ndarray
    ) -> np.ndarray:
        """
        Compute smoothed sqrt covariance using Joseph form.

        For numerical stability, we use:
        Pˢₖ = Pₖ + Gₖ(Pˢₖ₊₁ - P⁻ₖ₊₁)Gₖᵀ

        Then compute sqrt via Cholesky.

        Args:
            S_posterior: Current posterior sqrt cov
            S_prior_next: Next prior sqrt cov
            S_smoothed_next: Next smoothed sqrt cov
            gain: Smoother gain

        Returns:
            Smoothed sqrt covariance
        """
        # Compute full covariances
        P_posterior = S_posterior @ S_posterior.T
        P_prior_next = S_prior_next @ S_prior_next.T
        P_smoothed_next = S_smoothed_next @ S_smoothed_next.T

        # Smoother covariance update
        delta_P = P_smoothed_next - P_prior_next
        P_smoothed = P_posterior + gain @ delta_P @ gain.T

        # Ensure symmetry
        P_smoothed = 0.5 * (P_smoothed + P_smoothed.T)

        # Ensure positive definiteness
        eigvals = np.linalg.eigvalsh(P_smoothed)
        min_eig = np.min(eigvals)
        if min_eig < self.regularization:
            P_smoothed += (self.regularization - min_eig + 1e-10) * np.eye(P_smoothed.shape[0])

        # Compute sqrt via Cholesky
        try:
            S_smoothed = np.linalg.cholesky(P_smoothed)
            return S_smoothed
        except np.linalg.LinAlgError:
            # Fall back to eigendecomposition
            logger.warning("Cholesky failed, using eigendecomposition for sqrt cov")
            eigvals, eigvecs = np.linalg.eigh(P_smoothed)
            eigvals = np.maximum(eigvals, self.regularization)
            return eigvecs @ np.diag(np.sqrt(eigvals))

    def get_smoothed_state(self, cycle_index: int) -> Optional[SmoothedResult]:
        """
        Get smoothed state for a specific cycle.

        Args:
            cycle_index: Filter cycle number

        Returns:
            SmoothedResult or None if not available
        """
        smoothed = self.run_backward_pass()

        for result in smoothed:
            if result.cycle_index == cycle_index:
                return result

        return None

    def get_latest_smoothed(self) -> Optional[SmoothedResult]:
        """
        Get the most recent smoothed state.

        Returns:
            Latest SmoothedResult or None if no history
        """
        smoothed = self.run_backward_pass()
        return smoothed[-1] if smoothed else None

    def clear_history(self):
        """Clear all stored history"""
        self.history.clear()
        self._smoothed_cache = None
        logger.info("Smoother history cleared")

    def get_uncertainty_reduction(self) -> Optional[float]:
        """
        Calculate uncertainty reduction from smoothing.

        Returns:
            Ratio of smoothed to filter uncertainty (< 1 indicates improvement)
        """
        if len(self.history) < 2:
            return None

        smoothed = self.run_backward_pass()
        if not smoothed:
            return None

        # Compare trace of covariances
        filter_trace = np.sum(self.history[0].sqrt_cov_posterior.diagonal() ** 2)
        smoothed_trace = np.sum(smoothed[0].sqrt_cov.diagonal() ** 2)

        if filter_trace > 0:
            return smoothed_trace / filter_trace
        return None

    def get_statistics(self) -> dict:
        """Get smoother statistics"""
        smoothed = self.run_backward_pass() if len(self.history) >= 2 else []

        return {
            'history_length': len(self.history),
            'max_lag': self.max_lag,
            'smoothed_states': len(smoothed),
            'uncertainty_reduction': self.get_uncertainty_reduction(),
            'cache_valid': self._smoothed_cache is not None
        }


def integrate_smoother_with_filter(
    filter_instance: 'AutoNVISFilter',
    smoother: RTSSmoother
) -> None:
    """
    Integration helper: Store filter state to smoother after update.

    Call this after each filter update cycle to populate smoother history.

    Args:
        filter_instance: AutoNVISFilter instance
        smoother: RTSSmoother instance
    """
    # Get current states from filter
    state_posterior = filter_instance.filter.get_state().to_numpy()
    sqrt_cov_posterior = filter_instance.filter.get_sqrt_cov()

    # Get prior states from filter's internal tracking
    # Note: This requires the filter to track prior states
    if hasattr(filter_instance, '_last_prior_state'):
        state_prior = filter_instance._last_prior_state
        sqrt_cov_prior = filter_instance._last_prior_sqrt_cov
    else:
        # Use posterior as approximation if prior not available
        state_prior = state_posterior
        sqrt_cov_prior = sqrt_cov_posterior

    smoother.store_state(
        state_prior=state_prior,
        state_posterior=state_posterior,
        sqrt_cov_prior=sqrt_cov_prior,
        sqrt_cov_posterior=sqrt_cov_posterior,
        timestamp=filter_instance.last_update_time,
        cycle_index=filter_instance.cycle_count
    )
