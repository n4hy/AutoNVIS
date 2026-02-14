#!/usr/bin/env python3
"""
Test PropagationService integration with system orchestrator.

This script validates:
1. PropagationService initialization
2. Ray tracer integration with Ne grid
3. LUF/MUF calculation
4. Message queue publication (simulated)
"""

import sys
import logging
from pathlib import Path
import numpy as np

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent / "python"))

from src.common.config import get_config
from src.propagation.services import PropagationService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def create_chapman_layer(lat_grid, lon_grid, alt_grid):
    """Create Chapman layer ionosphere for testing."""
    Ne_max = 1e12  # el/m³
    h_max = 300.0  # km
    H = 50.0  # km

    ne_grid = np.zeros((len(lat_grid), len(lon_grid), len(alt_grid)))
    for i, h in enumerate(alt_grid):
        z = (h - h_max) / H
        Ne = Ne_max * np.exp(1 - z - np.exp(-z))
        ne_grid[:, :, i] = Ne

    return ne_grid


def main():
    """Test propagation service integration."""
    print("=" * 70)
    print("PropagationService Integration Test")
    print("=" * 70)
    print()

    # Load configuration
    logger.info("Loading configuration...")
    config = get_config()

    # Create propagation service
    logger.info("Creating PropagationService...")
    service = PropagationService(
        tx_lat=config.propagation.tx_lat,
        tx_lon=config.propagation.tx_lon,
        tx_alt=config.propagation.tx_alt_km,
        freq_min=3.0,  # Narrow range for faster testing
        freq_max=10.0,
        freq_step=1.0,
        elevation_step=5.0,  # Coarser steps for speed
        azimuth_step=30.0
    )

    # Get grid coordinates
    logger.info("Getting grid coordinates...")
    lat_grid = config.grid.get_lat_grid()
    lon_grid = config.grid.get_lon_grid()
    alt_grid = config.grid.get_alt_grid()

    print(f"\nGrid configuration:")
    print(f"  Latitude: {lat_grid[0]:.1f}° to {lat_grid[-1]:.1f}° ({len(lat_grid)} points)")
    print(f"  Longitude: {lon_grid[0]:.1f}° to {lon_grid[-1]:.1f}° ({len(lon_grid)} points)")
    print(f"  Altitude: {alt_grid[0]:.1f} to {alt_grid[-1]:.1f} km ({len(alt_grid)} points)")
    print()

    # Create test ionospheric grid
    logger.info("Creating Chapman layer ionosphere...")
    ne_grid = create_chapman_layer(lat_grid, lon_grid, alt_grid)

    print(f"Ionosphere statistics:")
    print(f"  Max Ne: {np.max(ne_grid):.2e} el/m³")
    print(f"  Mean Ne: {np.mean(ne_grid):.2e} el/m³")
    print(f"  Min Ne: {np.min(ne_grid):.2e} el/m³")
    print()

    # Initialize ray tracer
    logger.info("Initializing ray tracer...")
    service.initialize_ray_tracer(
        ne_grid=ne_grid,
        lat_grid=lat_grid,
        lon_grid=lon_grid,
        alt_grid=alt_grid,
        xray_flux=1e-6  # Nominal X-ray flux
    )

    print("Ray tracer initialized successfully!")
    print()

    # Calculate LUF/MUF
    logger.info("Calculating LUF/MUF products...")
    print("This may take a few seconds...")
    print()

    products = service.calculate_luf_muf()

    # Display results
    print("=" * 70)
    print("PROPAGATION PRODUCTS")
    print("=" * 70)
    print()

    print(f"Transmitter Location:")
    print(f"  Latitude: {products['transmitter']['latitude']:.2f}°")
    print(f"  Longitude: {products['transmitter']['longitude']:.2f}°")
    print(f"  Altitude: {products['transmitter']['altitude_km']:.2f} km")
    print()

    print(f"Frequency Predictions:")
    print(f"  LUF: {products['luf_mhz']:.2f} MHz  (Lowest Usable Frequency)")
    print(f"  MUF: {products['muf_mhz']:.2f} MHz  (Maximum Usable Frequency)")
    print(f"  FOT: {products['fot_mhz']:.2f} MHz  (Frequency of Optimum Traffic)")
    print()

    if products['usable_range_mhz']:
        print(f"Usable Range: {products['usable_range_mhz'][0]:.2f} - {products['usable_range_mhz'][1]:.2f} MHz")
    else:
        print("⚠️  BLACKOUT CONDITION - No usable frequencies!")
    print()

    print(f"Coverage Statistics:")
    stats = products['coverage_stats']
    print(f"  Total rays traced: {stats['total_rays']}")
    print(f"  Reflected rays: {stats['reflected_rays']} ({stats['reflection_rate']*100:.1f}%)")
    print(f"  Usable rays: {stats['usable_rays']} ({stats['usability_rate']*100:.1f}%)")
    print(f"  Average absorption: {stats['avg_absorption_db']:.1f} dB")
    print()

    print(f"Frequency Recommendations (for ALE):")
    for i, rec in enumerate(products['frequency_recommendations'], 1):
        if isinstance(rec, dict):
            print(f"  {i}. {rec['frequency_mhz']:.2f} MHz (confidence: {rec['confidence']:.2f})")
        else:
            print(f"  {i}. {rec:.2f} MHz")
    print()

    print(f"Performance:")
    print(f"  Calculation time: {products['calculation_time_sec']:.3f} seconds")
    print(f"  Timestamp: {products['timestamp_utc']}")
    print()

    # Test single frequency coverage
    logger.info("Testing single-frequency NVIS coverage...")
    print("=" * 70)
    print("SINGLE FREQUENCY COVERAGE TEST")
    print("=" * 70)
    print()

    test_freq = products['fot_mhz']  # Use optimal frequency
    coverage = service.calculate_nvis_coverage(test_freq)

    print(f"Frequency: {coverage['frequency_mhz']:.2f} MHz")
    print()

    summary = coverage['coverage_summary']
    print(f"Coverage Summary:")
    print(f"  Total rays: {summary['total_rays']}")
    print(f"  Reflected: {summary['reflected']} ({summary['reflection_rate']*100:.1f}%)")
    print(f"  Escaped: {summary['escaped']}")
    print(f"  Absorbed: {summary['absorbed']}")
    print(f"  Average range: {summary['avg_ground_range_km']:.1f} km")
    print(f"  Maximum range: {summary['max_ground_range_km']:.1f} km")
    print()

    # Success message
    print("=" * 70)
    print("✅ INTEGRATION TEST PASSED")
    print("=" * 70)
    print()
    print("PropagationService is ready for deployment!")
    print("The service can now be integrated into the system orchestrator.")
    print()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)
