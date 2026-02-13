"""
Filter Orchestrator

Integrates Auto-NVIS filter with system supervisor for autonomous operation.
Coordinates mode switching, filter cycles, and state management.
"""

import sys
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import numpy as np
from enum import Enum

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "assimilation" / "python"))

from src.supervisor.mode_controller import ModeController, OperationalMode as SupervisorMode
from autonvis_filter import AutoNVISFilter, OperationalMode as FilterMode
from src.assimilation.models.chapman_layer import ChapmanLayerModel


class FilterOrchestrator:
    """
    Orchestrates Auto-NVIS filter operations

    Integrates:
    - Mode controller (space weather monitoring)
    - SR-UKF filter (data assimilation)
    - Chapman layer physics (background state)
    - 15-minute cycle scheduling
    """

    def __init__(
        self,
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
        alt_grid: np.ndarray,
        cycle_interval_sec: int = 900,  # 15 minutes
        uncertainty_threshold: float = 1e12,
        localization_radius_km: float = 500.0
    ):
        """
        Initialize filter orchestrator

        Args:
            lat_grid: Latitude grid (degrees)
            lon_grid: Longitude grid (degrees)
            alt_grid: Altitude grid (km)
            cycle_interval_sec: Filter cycle interval (seconds)
            uncertainty_threshold: trace(P) threshold for smoother
            localization_radius_km: Gaspari-Cohn localization radius
        """
        # Grid configuration
        self.lat_grid = lat_grid
        self.lon_grid = lon_grid
        self.alt_grid = alt_grid

        self.n_lat = len(lat_grid)
        self.n_lon = len(lon_grid)
        self.n_alt = len(alt_grid)

        # Timing
        self.cycle_interval_sec = cycle_interval_sec
        self.cycle_count = 0
        self.last_cycle_time = None

        # Mode controller (monitors GOES X-ray)
        self.mode_controller = ModeController(
            xray_threshold=1e-5,  # M1 class
            hysteresis_sec=600    # 10 minutes
        )

        # SR-UKF filter
        self.filter = AutoNVISFilter(
            n_lat=self.n_lat,
            n_lon=self.n_lon,
            n_alt=self.n_alt,
            uncertainty_threshold=uncertainty_threshold,
            localization_radius_km=localization_radius_km
        )

        # Chapman layer physics model
        self.chapman = ChapmanLayerModel()

        # System state
        self.initialized = False
        self.running = False

    def initialize(self, initial_time: datetime, initial_ssn: float = 75.0):
        """
        Initialize filter with Chapman layer background

        Args:
            initial_time: Initial UTC datetime
            initial_ssn: Initial effective sunspot number
        """
        print("=" * 60)
        print("Auto-NVIS Filter Orchestrator Initialization")
        print("=" * 60)
        print()

        # Generate Chapman layer background
        print("Generating Chapman layer background state...")
        ne_grid_3d = self.chapman.compute_3d_grid(
            self.lat_grid,
            self.lon_grid,
            self.alt_grid,
            initial_time,
            initial_ssn
        )

        metrics = self.chapman.validate_grid(ne_grid_3d)
        print(f"  Grid: {self.n_lat}×{self.n_lon}×{self.n_alt} = {self.n_lat*self.n_lon*self.n_alt} points")
        print(f"  Min Ne: {metrics['min_ne']:.2e} el/m³")
        print(f"  Max Ne: {metrics['max_ne']:.2e} el/m³")
        print(f"  Mean Ne: {metrics['mean_ne']:.2e} el/m³")
        print()

        # Create initial state vector
        state_dim = self.n_lat * self.n_lon * self.n_alt + 1
        initial_state = np.zeros(state_dim)
        initial_state[:-1] = ne_grid_3d.flatten()
        initial_state[-1] = initial_ssn

        # Create initial covariance (10% uncertainty)
        initial_sqrt_cov = np.diag(0.1 * np.abs(initial_state))

        # Initialize filter
        print("Initializing SR-UKF filter...")
        self.filter.initialize(
            lat_grid=self.lat_grid,
            lon_grid=self.lon_grid,
            alt_grid=self.alt_grid,
            initial_state=initial_state,
            initial_sqrt_cov=initial_sqrt_cov,
            correlation_time=3600.0,
            process_noise_std=1e10
        )
        print()

        # Start mode controller
        print("Initializing mode controller...")
        self.mode_controller.reset()
        print(f"  Initial mode: {self.mode_controller.current_mode.value}")
        print(f"  X-ray threshold: {self.mode_controller.xray_threshold:.2e} W/m² (M1)")
        print(f"  Hysteresis: {self.mode_controller.hysteresis_sec} sec")
        print()

        self.initialized = True
        self.last_cycle_time = initial_time

        print("✓ Orchestrator initialized and ready")
        print()

    async def process_xray_event(self, flux: float, timestamp: datetime):
        """
        Process X-ray flux event and update mode

        Args:
            flux: X-ray flux (W/m²)
            timestamp: Event timestamp
        """
        # Update mode controller
        old_mode = self.mode_controller.current_mode
        new_mode = self.mode_controller.update(flux, timestamp)

        # Sync filter mode if changed
        if new_mode != old_mode:
            if new_mode == SupervisorMode.SHOCK:
                self.filter.set_mode(FilterMode.SHOCK)
                print(f"[{timestamp.isoformat()}] MODE SWITCH: QUIET → SHOCK")
                print(f"  X-ray flux: {flux:.2e} W/m² (M-class flare detected)")
                print(f"  Smoother: DISABLED (rapid ionospheric changes)")
            else:
                self.filter.set_mode(FilterMode.QUIET)
                print(f"[{timestamp.isoformat()}] MODE SWITCH: SHOCK → QUIET")
                print(f"  X-ray flux: {flux:.2e} W/m²")
                print(f"  Smoother: ENABLED (if uncertainty high)")

            print()

    async def run_filter_cycle(
        self,
        current_time: datetime,
        observations: Optional[np.ndarray] = None,
        obs_sqrt_cov: Optional[np.ndarray] = None,
        obs_model: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Execute one filter cycle

        Args:
            current_time: Current UTC datetime
            observations: Observation vector (optional)
            obs_sqrt_cov: Observation error sqrt covariance (optional)
            obs_model: Observation model (optional)

        Returns:
            Cycle results dictionary
        """
        if not self.initialized:
            raise RuntimeError("Orchestrator not initialized")

        self.cycle_count += 1

        print("-" * 60)
        print(f"Cycle {self.cycle_count}: {current_time.isoformat()} UTC")
        print("-" * 60)

        # Compute time step
        if self.last_cycle_time:
            dt = (current_time - self.last_cycle_time).total_seconds()
        else:
            dt = self.cycle_interval_sec

        # Run filter
        result = self.filter.run_cycle(
            dt=dt,
            observations=observations,
            obs_sqrt_cov=obs_sqrt_cov,
            obs_model=obs_model
        )

        # Get current state
        ne_grid = self.filter.get_state_grid()
        reff = self.filter.get_effective_ssn()
        uncertainty = self.filter.get_uncertainty()

        # Log results
        print(f"Mode: {result['mode']}")
        print(f"Smoother active: {result['smoother_active']}")
        print(f"Predict time: {result['predict_time_ms']:.2f} ms")
        if result['update_time_ms'] > 0:
            print(f"Update time: {result['update_time_ms']:.2f} ms")
        print(f"Inflation factor: {result['inflation_factor']:.4f}")
        print(f"Uncertainty: {uncertainty:.2e}")
        print(f"R_eff (SSN): {reff:.2f}")
        print(f"Ne range: [{ne_grid.min():.2e}, {ne_grid.max():.2e}] el/m³")
        print()

        self.last_cycle_time = current_time

        return {
            **result,
            'ne_grid': ne_grid,
            'reff': reff,
            'uncertainty': uncertainty,
            'timestamp': current_time.isoformat()
        }

    async def run_autonomous(
        self,
        duration_sec: int,
        xray_flux_func=None
    ):
        """
        Run autonomous operation for specified duration

        Args:
            duration_sec: Duration to run (seconds)
            xray_flux_func: Function to get X-ray flux (time) -> flux
                            If None, uses constant quiet-time flux
        """
        if not self.initialized:
            raise RuntimeError("Orchestrator not initialized")

        print("=" * 60)
        print("Auto-NVIS Autonomous Operation")
        print("=" * 60)
        print(f"Duration: {duration_sec} sec ({duration_sec/60:.1f} min)")
        print(f"Cycle interval: {self.cycle_interval_sec} sec")
        print()

        self.running = True
        start_time = self.last_cycle_time
        elapsed = 0

        try:
            while elapsed < duration_sec and self.running:
                current_time = start_time + timedelta(seconds=elapsed)

                # Simulate X-ray flux
                if xray_flux_func:
                    flux = xray_flux_func(elapsed)
                else:
                    flux = 1e-6  # Quiet-time baseline (B-class)

                # Process X-ray event
                await self.process_xray_event(flux, current_time)

                # Run filter cycle
                result = await self.run_filter_cycle(current_time)

                # Wait for next cycle
                await asyncio.sleep(0.1)  # Simulated, would be cycle_interval_sec in production
                elapsed += self.cycle_interval_sec

        except KeyboardInterrupt:
            print("\n⚠ Autonomous operation interrupted by user")
            self.running = False

        print("=" * 60)
        print("Autonomous Operation Complete")
        print("=" * 60)
        print(f"Total cycles: {self.cycle_count}")
        print()

        # Final statistics
        stats = self.filter.get_statistics()
        print("Final Statistics:")
        print(f"  Smoother activation rate: {stats['smoother_activation_rate']:.1%}")
        print(f"  Average predict time: {stats['avg_predict_time_ms']:.2f} ms")
        print(f"  Divergence count: {stats['divergence_count']}")
        print(f"  Final uncertainty: {stats['current_uncertainty']:.2e}")
        print()

    def stop(self):
        """Stop autonomous operation"""
        self.running = False

    def get_state_grid(self) -> np.ndarray:
        """Get current electron density grid"""
        return self.filter.get_state_grid()

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        filter_stats = self.filter.get_statistics()
        mode_stats = self.mode_controller.get_statistics()

        return {
            **filter_stats,
            'mode_controller': mode_stats,
            'cycle_count': self.cycle_count,
            'orchestrator_running': self.running
        }
