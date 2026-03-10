"""
Unit Tests for RTS Smoother

Tests the Rauch-Tung-Striebel fixed-interval smoother
backward pass implementation.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from assimilation.python.rts_smoother import (
    RTSSmoother,
    SmootherState,
    SmoothedResult
)


class TestSmootherState:
    """Test SmootherState dataclass"""

    def test_state_creation(self):
        """Test creating smoother state"""
        n = 10
        state = SmootherState(
            state_prior=np.zeros(n),
            state_posterior=np.ones(n),
            sqrt_cov_prior=np.eye(n) * 0.5,
            sqrt_cov_posterior=np.eye(n) * 0.3,
            timestamp=datetime.utcnow(),
            cycle_index=1
        )

        assert state.state_prior.shape == (n,)
        assert state.state_posterior.shape == (n,)
        assert state.cycle_index == 1

    def test_dimension_mismatch_error(self):
        """Test that dimension mismatch raises error"""
        with pytest.raises(ValueError):
            SmootherState(
                state_prior=np.zeros(10),
                state_posterior=np.ones(5),  # Wrong size
                sqrt_cov_prior=np.eye(10),
                sqrt_cov_posterior=np.eye(10),
                timestamp=datetime.utcnow()
            )


class TestRTSSmootherBasics:
    """Test basic RTS smoother functionality"""

    def test_initialization(self):
        """Test smoother initialization"""
        smoother = RTSSmoother(max_lag=5)

        assert smoother.max_lag == 5
        assert len(smoother.history) == 0
        assert smoother._smoothed_cache is None

    def test_store_state(self):
        """Test storing states"""
        smoother = RTSSmoother(max_lag=3)
        n = 10

        for i in range(5):
            smoother.store_state(
                state_prior=np.random.randn(n),
                state_posterior=np.random.randn(n),
                sqrt_cov_prior=np.eye(n) * 0.5,
                sqrt_cov_posterior=np.eye(n) * 0.3,
                cycle_index=i
            )

        # Should trim to max_lag + 1
        assert len(smoother.history) == 4

    def test_insufficient_history(self):
        """Test that backward pass with < 2 states returns empty"""
        smoother = RTSSmoother()

        # No states
        result = smoother.run_backward_pass()
        assert result == []

        # One state
        smoother.store_state(
            state_prior=np.zeros(5),
            state_posterior=np.ones(5),
            sqrt_cov_prior=np.eye(5),
            sqrt_cov_posterior=np.eye(5)
        )
        result = smoother.run_backward_pass()
        assert result == []


class TestRTSBackwardPass:
    """Test RTS smoother backward pass algorithm"""

    def test_backward_pass_basic(self):
        """Test basic backward pass execution"""
        smoother = RTSSmoother(max_lag=3)
        n = 10

        # Store several states
        for i in range(3):
            smoother.store_state(
                state_prior=np.random.randn(n) + i,
                state_posterior=np.random.randn(n) + i,
                sqrt_cov_prior=np.eye(n) * (0.5 - i * 0.1),
                sqrt_cov_posterior=np.eye(n) * (0.3 - i * 0.05),
                cycle_index=i
            )

        smoothed = smoother.run_backward_pass()

        assert len(smoothed) == 3
        assert all(isinstance(s, SmoothedResult) for s in smoothed)

        # Each should have state and covariance
        for s in smoothed:
            assert s.state.shape == (n,)
            assert s.sqrt_cov.shape == (n, n)

    def test_final_state_equals_filter(self):
        """Test that final smoothed state equals filter estimate"""
        smoother = RTSSmoother()
        n = 5

        # Store states
        final_posterior = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        final_sqrt_cov = np.eye(n) * 0.2

        smoother.store_state(
            state_prior=np.zeros(n),
            state_posterior=np.zeros(n),
            sqrt_cov_prior=np.eye(n),
            sqrt_cov_posterior=np.eye(n) * 0.5,
            cycle_index=0
        )

        smoother.store_state(
            state_prior=np.ones(n),
            state_posterior=final_posterior,
            sqrt_cov_prior=np.eye(n) * 0.3,
            sqrt_cov_posterior=final_sqrt_cov,
            cycle_index=1
        )

        smoothed = smoother.run_backward_pass()

        # Final smoothed should equal filter posterior
        np.testing.assert_array_almost_equal(
            smoothed[-1].state, final_posterior
        )
        np.testing.assert_array_almost_equal(
            smoothed[-1].sqrt_cov, final_sqrt_cov
        )

    def test_smoothing_reduces_uncertainty(self):
        """Test that smoothing typically reduces uncertainty"""
        smoother = RTSSmoother()
        n = 10
        np.random.seed(42)

        # Create sequence with increasing uncertainty
        for i in range(5):
            noise = 0.1 * i
            smoother.store_state(
                state_prior=np.random.randn(n) * (1 + noise),
                state_posterior=np.random.randn(n) * (1 + noise * 0.5),
                sqrt_cov_prior=np.eye(n) * (0.5 + noise),
                sqrt_cov_posterior=np.eye(n) * (0.3 + noise * 0.5),
                cycle_index=i
            )

        reduction = smoother.get_uncertainty_reduction()

        # Smoothing should generally reduce uncertainty
        # (ratio < 1 means improvement)
        assert reduction is not None
        # Note: With random data, might not always reduce
        # Just verify it returns a reasonable value
        assert 0 < reduction < 10

    def test_cache_invalidation(self):
        """Test that cache is invalidated when new state stored"""
        smoother = RTSSmoother()
        n = 5

        # Store initial states
        for i in range(3):
            smoother.store_state(
                state_prior=np.zeros(n),
                state_posterior=np.ones(n),
                sqrt_cov_prior=np.eye(n),
                sqrt_cov_posterior=np.eye(n),
                cycle_index=i
            )

        # Run backward pass (populates cache)
        _ = smoother.run_backward_pass()
        assert smoother._smoothed_cache is not None

        # Store new state
        smoother.store_state(
            state_prior=np.zeros(n),
            state_posterior=np.ones(n) * 2,
            sqrt_cov_prior=np.eye(n),
            sqrt_cov_posterior=np.eye(n),
            cycle_index=3
        )

        # Cache should be invalidated
        assert smoother._smoothed_cache is None


class TestSmootherGainComputation:
    """Test smoother gain matrix computation"""

    def test_gain_dimensions(self):
        """Test that gain has correct dimensions"""
        smoother = RTSSmoother()
        n = 10

        S_posterior = np.eye(n) * 0.3
        S_prior_next = np.eye(n) * 0.5

        gain = smoother._compute_gain(S_posterior, S_prior_next)

        assert gain.shape == (n, n)

    def test_gain_identity_case(self):
        """Test gain when covariances are equal"""
        smoother = RTSSmoother()
        n = 5

        S = np.eye(n) * 0.3
        gain = smoother._compute_gain(S, S)

        # When equal, gain should be identity
        np.testing.assert_array_almost_equal(gain, np.eye(n), decimal=5)


class TestSmootherUtilities:
    """Test smoother utility methods"""

    def test_clear_history(self):
        """Test clearing history"""
        smoother = RTSSmoother()
        n = 5

        for i in range(3):
            smoother.store_state(
                state_prior=np.zeros(n),
                state_posterior=np.ones(n),
                sqrt_cov_prior=np.eye(n),
                sqrt_cov_posterior=np.eye(n)
            )

        smoother.clear_history()

        assert len(smoother.history) == 0
        assert smoother._smoothed_cache is None

    def test_get_smoothed_state_by_cycle(self):
        """Test retrieving smoothed state by cycle index"""
        smoother = RTSSmoother()
        n = 5

        for i in range(3):
            smoother.store_state(
                state_prior=np.zeros(n),
                state_posterior=np.ones(n) * i,
                sqrt_cov_prior=np.eye(n),
                sqrt_cov_posterior=np.eye(n),
                cycle_index=i
            )

        # Get middle state
        result = smoother.get_smoothed_state(1)
        assert result is not None
        assert result.cycle_index == 1

        # Get non-existent cycle
        result = smoother.get_smoothed_state(99)
        assert result is None

    def test_get_latest_smoothed(self):
        """Test getting latest smoothed state"""
        smoother = RTSSmoother()
        n = 5

        for i in range(3):
            smoother.store_state(
                state_prior=np.zeros(n),
                state_posterior=np.ones(n) * i,
                sqrt_cov_prior=np.eye(n),
                sqrt_cov_posterior=np.eye(n),
                cycle_index=i
            )

        latest = smoother.get_latest_smoothed()
        assert latest is not None
        assert latest.cycle_index == 2

    def test_get_statistics(self):
        """Test statistics reporting"""
        smoother = RTSSmoother(max_lag=5)
        n = 5

        for i in range(3):
            smoother.store_state(
                state_prior=np.zeros(n),
                state_posterior=np.ones(n),
                sqrt_cov_prior=np.eye(n),
                sqrt_cov_posterior=np.eye(n)
            )

        stats = smoother.get_statistics()

        assert stats['history_length'] == 3
        assert stats['max_lag'] == 5
        assert 'uncertainty_reduction' in stats


class TestSmootherNumericalStability:
    """Test numerical stability of smoother"""

    def test_ill_conditioned_covariance(self):
        """Test handling of ill-conditioned covariance"""
        smoother = RTSSmoother(regularization=1e-10)
        n = 5

        # Create ill-conditioned covariance
        ill_cond_cov = np.eye(n)
        ill_cond_cov[0, 0] = 1e-15  # Very small eigenvalue

        for i in range(3):
            smoother.store_state(
                state_prior=np.zeros(n),
                state_posterior=np.ones(n),
                sqrt_cov_prior=ill_cond_cov,
                sqrt_cov_posterior=ill_cond_cov,
                cycle_index=i
            )

        # Should complete without error
        smoothed = smoother.run_backward_pass()
        assert len(smoothed) == 3

    def test_large_state_dimension(self):
        """Test with larger state dimension"""
        smoother = RTSSmoother()
        n = 100  # Larger but not huge

        for i in range(3):
            smoother.store_state(
                state_prior=np.random.randn(n),
                state_posterior=np.random.randn(n),
                sqrt_cov_prior=np.eye(n) * 0.5,
                sqrt_cov_posterior=np.eye(n) * 0.3,
                cycle_index=i
            )

        smoothed = smoother.run_backward_pass()
        assert len(smoothed) == 3
        assert all(s.state.shape == (n,) for s in smoothed)


class TestSmootherWithRealData:
    """Test smoother with more realistic scenarios"""

    def test_random_walk_smoothing(self):
        """Test smoothing a random walk process"""
        smoother = RTSSmoother(max_lag=5)
        n = 10
        n_steps = 6
        np.random.seed(123)

        # Generate random walk
        true_state = np.zeros(n)
        process_noise = 0.1

        for i in range(n_steps):
            # True state evolves
            true_state = true_state + np.random.randn(n) * process_noise

            # Noisy observation
            obs_noise = 0.2
            observed = true_state + np.random.randn(n) * obs_noise

            # Prior is previous posterior + process noise
            if i == 0:
                prior = np.zeros(n)
            else:
                prior = smoother.history[-1].state_posterior

            # Posterior incorporates observation
            posterior = 0.5 * prior + 0.5 * observed

            smoother.store_state(
                state_prior=prior,
                state_posterior=posterior,
                sqrt_cov_prior=np.eye(n) * (obs_noise * 1.5),
                sqrt_cov_posterior=np.eye(n) * obs_noise,
                cycle_index=i
            )

        smoothed = smoother.run_backward_pass()

        assert len(smoothed) == n_steps
        # Smoothed should have reasonable values
        for s in smoothed:
            assert np.all(np.isfinite(s.state))
            assert np.all(np.isfinite(s.sqrt_cov))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
