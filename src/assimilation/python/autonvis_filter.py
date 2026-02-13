"""
Auto-NVIS Filter Wrapper

Python wrapper for C++ SR-UKF that integrates with supervisor and mode controller.
Implements conditional smoother logic based on operational mode and uncertainty.
"""

import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from datetime import datetime
from enum import Enum
import numpy as np

# Add to path for import
sys.path.insert(0, str(Path(__file__).parent))
import autonvis_srukf as srukf


class OperationalMode(Enum):
    """System operational modes"""
    QUIET = "QUIET"
    SHOCK = "SHOCK"


class AutoNVISFilter:
    """
    Auto-NVIS Data Assimilation Filter

    Integrates C++ SR-UKF with Python supervisor and mode controller.
    Implements conditional smoother logic:
    - NEVER use smoother during SHOCK mode (non-stationary ionosphere)
    - ONLY use smoother when uncertainty (trace P) exceeds threshold
    """

    def __init__(
        self,
        n_lat: int,
        n_lon: int,
        n_alt: int,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
        uncertainty_threshold: float = 1e12,
        localization_radius_km: float = 500.0
    ):
        """
        Initialize Auto-NVIS filter

        Args:
            n_lat: Number of latitude grid points
            n_lon: Number of longitude grid points
            n_alt: Number of altitude grid points
            alpha: UKF scaling parameter (spread of sigma points)
            beta: UKF parameter for distribution (2 = Gaussian optimal)
            kappa: UKF parameter (0 or 3-L typical)
            uncertainty_threshold: trace(P) threshold for smoother activation
            localization_radius_km: Gaspari-Cohn localization radius
        """
        # Grid dimensions
        self.n_lat = n_lat
        self.n_lon = n_lon
        self.n_alt = n_alt

        # Operational state
        self.current_mode = OperationalMode.QUIET
        self.uncertainty_threshold = uncertainty_threshold

        # Grid coordinate vectors (will be set during initialization)
        self.lat_grid = None
        self.lon_grid = None
        self.alt_grid = None

        # Create C++ filter
        self.filter = srukf.SquareRootUKF(
            n_lat=n_lat,
            n_lon=n_lon,
            n_alt=n_alt,
            alpha=alpha,
            beta=beta,
            kappa=kappa
        )

        # Configure adaptive inflation (enabled by default)
        inflation_config = srukf.AdaptiveInflationConfig()
        inflation_config.enabled = True
        inflation_config.initial_inflation = 1.0
        inflation_config.min_inflation = 1.0
        inflation_config.max_inflation = 2.0
        inflation_config.adaptation_rate = 0.95
        inflation_config.divergence_threshold = 3.0
        self.filter.set_adaptive_inflation_config(inflation_config)

        # Configure covariance localization
        self.localization_radius_km = localization_radius_km

        # Physics model (will be set during initialization)
        self.physics_model = None

        # State history for smoother (if enabled)
        self.state_history = []
        self.sqrt_cov_history = []
        self.max_history_length = 3  # lag-3 smoother maximum

        # Statistics tracking
        self.cycle_count = 0
        self.smoother_activation_count = 0
        self.last_update_time = None

    def initialize(
        self,
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
        alt_grid: np.ndarray,
        initial_state: np.ndarray,
        initial_sqrt_cov: np.ndarray,
        correlation_time: float = 3600.0,
        process_noise_std: float = 1e10
    ):
        """
        Initialize filter with background state

        Args:
            lat_grid: Latitude grid (degrees)
            lon_grid: Longitude grid (degrees)
            alt_grid: Altitude grid (km)
            initial_state: Initial state vector (numpy array)
            initial_sqrt_cov: Initial square-root covariance (numpy array)
            correlation_time: Gauss-Markov correlation time (seconds)
            process_noise_std: Process noise standard deviation
        """
        # Store grid vectors
        self.lat_grid = lat_grid
        self.lon_grid = lon_grid
        self.alt_grid = alt_grid

        # Create physics model
        self.physics_model = srukf.GaussMarkovModel(
            correlation_time=correlation_time,
            process_noise_std=process_noise_std
        )
        self.filter.set_physics_model(self.physics_model)

        # Configure localization with grid
        localization_config = srukf.LocalizationConfig()
        localization_config.enabled = True
        localization_config.radius_km = self.localization_radius_km
        localization_config.precompute = True

        self.filter.set_localization_config(
            localization_config,
            lat_grid.tolist(),
            lon_grid.tolist(),
            alt_grid.tolist()
        )

        # Convert numpy array to StateVector
        state_vec = srukf.StateVector(self.n_lat, self.n_lon, self.n_alt)
        state_vec.from_numpy(initial_state)

        # Initialize filter state
        self.filter.initialize(state_vec, initial_sqrt_cov)

        self.last_update_time = datetime.utcnow()
        print(f"✓ Filter initialized: {self.n_lat}×{self.n_lon}×{self.n_alt} grid")
        print(f"  Localization: {self.localization_radius_km} km radius")
        print(f"  Physics model: {self.physics_model.name()}")

    def set_mode(self, mode: OperationalMode):
        """
        Set operational mode (QUIET or SHOCK)

        Args:
            mode: New operational mode
        """
        if mode != self.current_mode:
            print(f"Mode switch: {self.current_mode.value} → {mode.value}")
            self.current_mode = mode

            # Clear state history on mode change (invalidates smoother)
            if len(self.state_history) > 0:
                print("  Clearing state history (mode change)")
                self.state_history.clear()
                self.sqrt_cov_history.clear()

    def should_use_smoother(self) -> bool:
        """
        Determine if smoother should be activated

        Returns:
            True if smoother should run, False otherwise

        Logic:
            - NEVER during SHOCK mode (non-stationary ionosphere)
            - ONLY when trace(P) > threshold (high uncertainty)
        """
        # NEVER use smoother during shock events
        # Ionosphere changes too rapidly, backward pass assumptions violated
        if self.current_mode == OperationalMode.SHOCK:
            return False

        # Only activate when uncertainty is high
        # Low uncertainty → filter is confident, skip smoother computation
        sqrt_cov = self.filter.get_sqrt_cov()
        trace_P = np.sum(sqrt_cov.diagonal() ** 2)

        return trace_P > self.uncertainty_threshold

    def predict(self, dt: float):
        """
        Predict step: propagate state forward in time

        Args:
            dt: Time step (seconds)
        """
        self.filter.predict(dt)
        self.cycle_count += 1

    def update(
        self,
        observations: np.ndarray,
        obs_sqrt_cov: np.ndarray,
        obs_model: Any
    ):
        """
        Update step: assimilate observations

        Args:
            observations: Observation vector
            obs_sqrt_cov: Observation error square-root covariance
            obs_model: Observation model (TECObservationModel, etc.)
        """
        self.filter.update(obs_model, observations, obs_sqrt_cov)

        # Store state for potential smoother use
        if self.should_use_smoother():
            state = self.filter.get_state().to_numpy()
            sqrt_cov = self.filter.get_sqrt_cov()

            self.state_history.append(state.copy())
            self.sqrt_cov_history.append(sqrt_cov.copy())

            # Limit history length
            if len(self.state_history) > self.max_history_length:
                self.state_history.pop(0)
                self.sqrt_cov_history.pop(0)
        else:
            # Clear history if smoother not active
            if len(self.state_history) > 0:
                self.state_history.clear()
                self.sqrt_cov_history.clear()

        self.last_update_time = datetime.utcnow()

    def run_cycle(
        self,
        dt: float,
        observations: Optional[np.ndarray] = None,
        obs_sqrt_cov: Optional[np.ndarray] = None,
        obs_model: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Run complete filter cycle: predict + update

        Args:
            dt: Time step (seconds)
            observations: Observation vector (optional)
            obs_sqrt_cov: Observation error square-root covariance (optional)
            obs_model: Observation model (optional)

        Returns:
            Cycle results dictionary
        """
        # Predict step
        self.predict(dt)

        # Update step (if observations available)
        if observations is not None and obs_model is not None:
            self.update(observations, obs_sqrt_cov, obs_model)

        # Get statistics
        stats = self.filter.get_statistics()

        # Check if smoother would activate
        smoother_active = self.should_use_smoother()
        if smoother_active:
            self.smoother_activation_count += 1

        return {
            'cycle': self.cycle_count,
            'mode': self.current_mode.value,
            'smoother_active': smoother_active,
            'predict_time_ms': stats.last_predict_time_ms,
            'update_time_ms': stats.last_update_time_ms,
            'inflation_factor': stats.inflation_factor,
            'last_nis': stats.last_nis,
            'avg_nis': stats.avg_nis,
            'divergence_count': stats.divergence_count,
            'timestamp': self.last_update_time.isoformat() if self.last_update_time else None
        }

    def get_state_grid(self) -> np.ndarray:
        """
        Get current electron density grid

        Returns:
            3D array (n_lat × n_lon × n_alt) of electron density (el/m³)
        """
        state_vec = self.filter.get_state()
        ne_grid = np.zeros((self.n_lat, self.n_lon, self.n_alt))

        for i in range(self.n_lat):
            for j in range(self.n_lon):
                for k in range(self.n_alt):
                    ne_grid[i, j, k] = state_vec.get_ne(i, j, k)

        return ne_grid

    def get_effective_ssn(self) -> float:
        """
        Get effective sunspot number

        Returns:
            Effective sunspot number
        """
        return self.filter.get_state().get_reff()

    def get_uncertainty(self) -> float:
        """
        Get current uncertainty (trace of covariance)

        Returns:
            trace(P)
        """
        sqrt_cov = self.filter.get_sqrt_cov()
        return np.sum(sqrt_cov.diagonal() ** 2)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive filter statistics

        Returns:
            Statistics dictionary
        """
        stats = self.filter.get_statistics()

        return {
            'cycle_count': self.cycle_count,
            'smoother_activation_count': self.smoother_activation_count,
            'smoother_activation_rate': (
                self.smoother_activation_count / self.cycle_count
                if self.cycle_count > 0 else 0.0
            ),
            'predict_count': stats.predict_count,
            'update_count': stats.update_count,
            'avg_predict_time_ms': stats.avg_predict_time_ms,
            'avg_update_time_ms': stats.avg_update_time_ms,
            'inflation_factor': stats.inflation_factor,
            'avg_nis': stats.avg_nis,
            'divergence_count': stats.divergence_count,
            'min_eigenvalue': stats.min_eigenvalue,
            'max_eigenvalue': stats.max_eigenvalue,
            'current_uncertainty': self.get_uncertainty(),
            'current_mode': self.current_mode.value,
            'last_update': self.last_update_time.isoformat() if self.last_update_time else None
        }

    def apply_inflation(self, factor: float):
        """
        Manually apply covariance inflation

        Args:
            factor: Inflation factor (> 1.0)
        """
        self.filter.apply_inflation(factor)

    def save_checkpoint(self, filepath: str):
        """
        Save filter state to checkpoint file

        Args:
            filepath: Path to checkpoint file
        """
        # TODO: Implement checkpoint save/load in C++ layer
        raise NotImplementedError("Checkpoint persistence not yet implemented")

    def load_checkpoint(self, filepath: str):
        """
        Load filter state from checkpoint file

        Args:
            filepath: Path to checkpoint file
        """
        # TODO: Implement checkpoint save/load in C++ layer
        raise NotImplementedError("Checkpoint persistence not yet implemented")
