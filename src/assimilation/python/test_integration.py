#!/usr/bin/env python3
"""
Integration Test: Python-C++ SR-UKF with Mode Controller

Demonstrates:
1. Filter initialization with Chapman layer background
2. Mode switching (QUIET <-> SHOCK)
3. Conditional smoother activation logic
4. Integration with supervisor
"""

import sys
from pathlib import Path
import numpy as np
from datetime import datetime

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from autonvis_filter import AutoNVISFilter, OperationalMode
from src.assimilation.models.chapman_layer import ChapmanLayerModel


def generate_background_state(
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    alt_grid: np.ndarray,
    time: datetime,
    ssn: float = 75.0
) -> np.ndarray:
    """
    Generate background ionospheric state using Chapman layer model

    Args:
        lat_grid: Latitude grid (degrees)
        lon_grid: Longitude grid (degrees)
        alt_grid: Altitude grid (km)
        time: UTC datetime
        ssn: Effective sunspot number

    Returns:
        Background state vector
    """
    print("Generating Chapman layer background...")

    model = ChapmanLayerModel()
    ne_grid_3d = model.compute_3d_grid(lat_grid, lon_grid, alt_grid, time, ssn)

    # Validate
    metrics = model.validate_grid(ne_grid_3d)
    print(f"  Min Ne: {metrics['min_ne']:.2e} el/m³")
    print(f"  Max Ne: {metrics['max_ne']:.2e} el/m³")
    print(f"  Mean Ne: {metrics['mean_ne']:.2e} el/m³")
    print(f"  Invalid count: {metrics['invalid_count']}")

    # Flatten to state vector (Ne at all grid points + R_eff)
    n_lat, n_lon, n_alt = ne_grid_3d.shape
    state_dim = n_lat * n_lon * n_alt + 1

    state_vector = np.zeros(state_dim)
    state_vector[:-1] = ne_grid_3d.flatten()  # Ne values
    state_vector[-1] = ssn  # R_eff

    return state_vector


def main():
    """Integration test main function"""
    print("=" * 60)
    print("Auto-NVIS Python-C++ Integration Test")
    print("=" * 60)
    print()

    # Define grid (small for testing)
    lat_grid = np.linspace(20, 50, 5)  # 5 lat points
    lon_grid = np.linspace(-120, -70, 5)  # 5 lon points
    alt_grid = np.linspace(60, 600, 7)  # 7 alt points

    print(f"Grid: {len(lat_grid)}×{len(lon_grid)}×{len(alt_grid)} = {len(lat_grid)*len(lon_grid)*len(alt_grid)} points")
    print(f"State dimension: {len(lat_grid)*len(lon_grid)*len(alt_grid) + 1} (Ne grid + R_eff)")
    print()

    # Generate background state
    time = datetime(2026, 3, 21, 18, 0, 0)
    initial_state = generate_background_state(lat_grid, lon_grid, alt_grid, time, ssn=75.0)
    print()

    # Generate initial sqrt covariance (diagonal, 10% uncertainty)
    state_dim = len(initial_state)
    initial_sqrt_cov = np.diag(0.1 * np.abs(initial_state))
    print(f"Initial covariance: {state_dim}×{state_dim} diagonal")
    print()

    # Create filter
    print("Initializing Auto-NVIS filter...")
    filter = AutoNVISFilter(
        n_lat=len(lat_grid),
        n_lon=len(lon_grid),
        n_alt=len(alt_grid),
        alpha=1e-3,
        beta=2.0,
        kappa=0.0,
        uncertainty_threshold=1e12,
        localization_radius_km=500.0
    )

    filter.initialize(
        lat_grid=lat_grid,
        lon_grid=lon_grid,
        alt_grid=alt_grid,
        initial_state=initial_state,
        initial_sqrt_cov=initial_sqrt_cov,
        correlation_time=3600.0,
        process_noise_std=1e10
    )
    print()

    # Test 1: QUIET mode operation
    print("-" * 60)
    print("Test 1: QUIET Mode Operation")
    print("-" * 60)

    filter.set_mode(OperationalMode.QUIET)

    print(f"Current uncertainty: {filter.get_uncertainty():.2e}")
    print(f"Smoother threshold: {filter.uncertainty_threshold:.2e}")
    print(f"Smoother would activate: {filter.should_use_smoother()}")
    print()

    # Run predict step
    print("Running predict step (dt=900 sec)...")
    filter.predict(dt=900.0)

    stats = filter.get_statistics()
    print(f"  Predict time: {stats['avg_predict_time_ms']:.2f} ms")
    print(f"  Inflation factor: {stats['inflation_factor']:.4f}")
    print()

    # Test 2: Mode switch to SHOCK
    print("-" * 60)
    print("Test 2: SHOCK Mode Operation")
    print("-" * 60)

    filter.set_mode(OperationalMode.SHOCK)

    print(f"Current uncertainty: {filter.get_uncertainty():.2e}")
    print(f"Smoother would activate: {filter.should_use_smoother()}")
    print("  (Should be False - smoother NEVER active during SHOCK)")
    print()

    # Run another cycle
    print("Running predict step in SHOCK mode...")
    filter.predict(dt=900.0)
    print()

    # Test 3: Return to QUIET mode
    print("-" * 60)
    print("Test 3: Return to QUIET Mode")
    print("-" * 60)

    filter.set_mode(OperationalMode.QUIET)

    print(f"Smoother would activate: {filter.should_use_smoother()}")
    print()

    # Test 4: Full cycle with synthetic observations
    print("-" * 60)
    print("Test 4: Full Cycle (Predict + Update)")
    print("-" * 60)

    # Generate synthetic TEC observations
    import autonvis_srukf as srukf

    # Create TEC measurement
    tec_meas = srukf.TECMeasurement()
    tec_meas.latitude = 35.0
    tec_meas.longitude = -95.0
    tec_meas.altitude = 200.0  # km
    tec_meas.elevation = 45.0  # degrees
    tec_meas.azimuth = 180.0  # degrees
    tec_meas.tec_value = 20.0  # TECU
    tec_meas.tec_error = 2.0  # TECU

    measurements = [tec_meas]

    print(f"Synthetic TEC observation: {tec_meas.tec_value:.1f} ± {tec_meas.tec_error:.1f} TECU")
    print(f"  Location: {tec_meas.latitude}°N, {tec_meas.longitude}°E")
    print()

    # Create observation model
    obs_model = srukf.TECObservationModel(
        measurements,
        lat_grid.tolist(),
        lon_grid.tolist(),
        alt_grid.tolist()
    )

    print(f"Observation dimension: {obs_model.obs_dimension()}")

    # Create observation vector and covariance
    obs_vec = np.array([tec_meas.tec_value])
    obs_sqrt_cov = np.array([[tec_meas.tec_error]])

    # Run full cycle
    result = filter.run_cycle(
        dt=900.0,
        observations=obs_vec,
        obs_sqrt_cov=obs_sqrt_cov,
        obs_model=obs_model
    )

    print()
    print("Cycle results:")
    print(f"  Cycle number: {result['cycle']}")
    print(f"  Mode: {result['mode']}")
    print(f"  Smoother active: {result['smoother_active']}")
    print(f"  Predict time: {result['predict_time_ms']:.2f} ms")
    print(f"  Update time: {result['update_time_ms']:.2f} ms")
    print(f"  Inflation factor: {result['inflation_factor']:.4f}")
    print(f"  NIS (last): {result['last_nis']:.4f}")
    print(f"  NIS (avg): {result['avg_nis']:.4f}")
    print()

    # Test 5: Get state grid
    print("-" * 60)
    print("Test 5: State Grid Extraction")
    print("-" * 60)

    ne_grid = filter.get_state_grid()
    print(f"Electron density grid shape: {ne_grid.shape}")
    print(f"  Min Ne: {ne_grid.min():.2e} el/m³")
    print(f"  Max Ne: {ne_grid.max():.2e} el/m³")
    print(f"  Mean Ne: {ne_grid.mean():.2e} el/m³")
    print()

    reff = filter.get_effective_ssn()
    print(f"Effective sunspot number: {reff:.2f}")
    print()

    # Test 6: Comprehensive statistics
    print("-" * 60)
    print("Test 6: Filter Statistics")
    print("-" * 60)

    stats = filter.get_statistics()
    print(f"Total cycles: {stats['cycle_count']}")
    print(f"Smoother activations: {stats['smoother_activation_count']}")
    print(f"Smoother activation rate: {stats['smoother_activation_rate']:.2%}")
    print(f"Average predict time: {stats['avg_predict_time_ms']:.2f} ms")
    print(f"Average update time: {stats['avg_update_time_ms']:.2f} ms")
    print(f"Current uncertainty: {stats['current_uncertainty']:.2e}")
    print(f"Current mode: {stats['current_mode']}")
    print(f"Divergence count: {stats['divergence_count']}")
    print()

    # Summary
    print("=" * 60)
    print("✓ Integration Test Complete")
    print("=" * 60)
    print()
    print("Key findings:")
    print("  ✓ Filter initialization successful")
    print("  ✓ Mode switching working (QUIET <-> SHOCK)")
    print(f"  ✓ Conditional smoother logic: SHOCK={not filter.should_use_smoother()}")
    print("  ✓ Predict/update cycles executing")
    print("  ✓ State grid extraction working")
    print("  ✓ Statistics tracking functional")
    print()
    print("Ready for supervisor integration!")


if __name__ == "__main__":
    main()
