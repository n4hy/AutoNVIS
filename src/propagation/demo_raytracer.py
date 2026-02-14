#!/usr/bin/env python3
"""
Demo: Native C++ Ray Tracer (No MATLAB Required!)

This script demonstrates the pure Python/C++ ray tracing engine
as a drop-in replacement for MATLAB PHaRLAP.

Build first:
    cd src/propagation
    cmake -B build && cmake --build build -j$(nproc)

Then run:
    python3 src/propagation/demo_raytracer.py
"""

import numpy as np
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Add propagation module to path
sys.path.insert(0, str(Path(__file__).parent / 'python'))

try:
    from pharlap_replacement import RayTracer
    logger.info("✓ Ray tracer module imported successfully!")
except ImportError as e:
    logger.error(f"✗ Failed to import ray tracer: {e}")
    logger.error("\nPlease build the C++ module first:")
    logger.error("  cd src/propagation")
    logger.error("  cmake -B build && cmake --build build -j$(nproc)")
    sys.exit(1)


def create_chapman_layer(lat, lon, alt):
    """
    Create simple Chapman layer ionosphere for testing.

    Args:
        lat, lon, alt: Grid coordinates

    Returns:
        Electron density grid (n_lat, n_lon, n_alt)
    """
    logger.info("Creating Chapman layer ionosphere...")

    n_lat, n_lon, n_alt = len(lat), len(lon), len(alt)
    ne_grid = np.zeros((n_lat, n_lon, n_alt))

    # Chapman layer parameters
    NmF2 = 5e11  # Peak density (el/m³)
    hmF2 = 300.0  # Peak height (km)
    H = 80.0      # Scale height (km)

    for i in range(n_lat):
        for j in range(n_lon):
            for k in range(n_alt):
                # Chapman layer profile
                z = (alt[k] - hmF2) / H
                ne_grid[i, j, k] = NmF2 * np.exp(0.5 * (1 - z - np.exp(-z)))

    logger.info(f"  Grid: {n_lat}×{n_lon}×{n_alt}")
    logger.info(f"  NmF2: {NmF2:.2e} el/m³")
    logger.info(f"  hmF2: {hmF2} km")

    return ne_grid


def demo_single_ray():
    """Demonstrate single ray tracing."""
    logger.info("\n" + "="*60)
    logger.info("DEMO 1: Single Ray Trace")
    logger.info("="*60)

    # Create small grid for fast demo
    lat = np.linspace(-20, 20, 5)
    lon = np.linspace(-20, 20, 5)
    alt = np.linspace(60, 600, 20)

    ne_grid = create_chapman_layer(lat, lon, alt)

    # Initialize ray tracer
    logger.info("\nInitializing ray tracer...")
    tracer = RayTracer(ne_grid, lat, lon, alt)

    # Trace single ray
    logger.info("\nTracing ray: 5 MHz, 85° elevation, 0° azimuth")
    path = tracer.trace_ray(
        tx_lat=0.0,
        tx_lon=0.0,
        elevation=85.0,
        azimuth=0.0,
        freq_mhz=5.0
    )

    # Display results
    logger.info("\nRay Trace Results:")
    logger.info(f"  Ground range: {path['ground_range']:.1f} km")
    logger.info(f"  Apex altitude: {path['apex_altitude']:.1f} km")
    logger.info(f"  Path length: {path['path_length']:.1f} km")
    logger.info(f"  Absorption: {path['absorption_db']:.1f} dB")
    logger.info(f"  Reflected: {path['reflected']}")
    logger.info(f"  Escaped: {path['escaped']}")
    logger.info(f"  Number of points: {len(path['positions'])}")


def demo_nvis_coverage():
    """Demonstrate NVIS coverage calculation."""
    logger.info("\n" + "="*60)
    logger.info("DEMO 2: NVIS Coverage Map")
    logger.info("="*60)

    # Create grid
    lat = np.linspace(-20, 20, 5)
    lon = np.linspace(-20, 20, 5)
    alt = np.linspace(60, 600, 20)

    ne_grid = create_chapman_layer(lat, lon, alt)

    # Initialize ray tracer
    tracer = RayTracer(ne_grid, lat, lon, alt)

    # Calculate NVIS coverage
    logger.info("\nCalculating NVIS coverage...")
    logger.info("  Frequency: 5 MHz")
    logger.info("  Elevation: 70-90° (step 5°)")
    logger.info("  Azimuth: 0-360° (step 45°)")

    paths = tracer.trace_nvis(
        tx_lat=0.0,
        tx_lon=0.0,
        freq_mhz=5.0,
        elevation_min=70.0,
        elevation_max=90.0,
        elevation_step=5.0,
        azimuth_step=45.0
    )

    logger.info(f"\nTotal rays traced: {len(paths)}")

    # Analyze results
    reflected = [p for p in paths if p['reflected']]
    escaped = [p for p in paths if p['escaped']]
    absorbed = [p for p in paths if p['absorbed']]

    logger.info(f"  Reflected: {len(reflected)} ({100*len(reflected)/len(paths):.0f}%)")
    logger.info(f"  Escaped: {len(escaped)} ({100*len(escaped)/len(paths):.0f}%)")
    logger.info(f"  Absorbed: {len(absorbed)} ({100*len(absorbed)/len(paths):.0f}%)")

    if reflected:
        ranges = [p['ground_range'] for p in reflected]
        absorptions = [p['absorption_db'] for p in reflected]

        logger.info(f"\nGround Range Statistics:")
        logger.info(f"  Min: {min(ranges):.1f} km")
        logger.info(f"  Max: {max(ranges):.1f} km")
        logger.info(f"  Mean: {np.mean(ranges):.1f} km")

        logger.info(f"\nAbsorption Statistics:")
        logger.info(f"  Min: {min(absorptions):.1f} dB")
        logger.info(f"  Max: {max(absorptions):.1f} dB")
        logger.info(f"  Mean: {np.mean(absorptions):.1f} dB")


def demo_luf_muf():
    """Demonstrate LUF/MUF calculation."""
    logger.info("\n" + "="*60)
    logger.info("DEMO 3: LUF/MUF Analysis")
    logger.info("="*60)

    # Create grid
    lat = np.linspace(-20, 20, 5)
    lon = np.linspace(-20, 20, 5)
    alt = np.linspace(60, 600, 20)

    ne_grid = create_chapman_layer(lat, lon, alt)

    # Initialize ray tracer
    tracer = RayTracer(ne_grid, lat, lon, alt)

    # Calculate coverage over frequency range
    logger.info("\nCalculating LUF/MUF...")
    logger.info("  Frequency range: 2-15 MHz")
    logger.info("  Frequency step: 2 MHz")

    coverage = tracer.calculate_coverage(
        tx_lat=0.0,
        tx_lon=0.0,
        freq_min=2.0,
        freq_max=15.0,
        freq_step=2.0
    )

    # Display results
    logger.info("\nLUF/MUF Results:")
    logger.info(f"  LUF: {coverage['luf']:.1f} MHz")
    logger.info(f"  MUF: {coverage['muf']:.1f} MHz")
    logger.info(f"  FOT (Optimal): {coverage['optimal_freq']:.1f} MHz")
    logger.info(f"  Usable bandwidth: {coverage['muf'] - coverage['luf']:.1f} MHz")
    logger.info(f"  Blackout: {'YES' if coverage['blackout'] else 'NO'}")

    if not coverage['blackout']:
        logger.info(f"  Usable range: {coverage['usable_range'][0]:.1f} - {coverage['usable_range'][1]:.1f} MHz")


def demo_performance():
    """Demonstrate performance with realistic grid."""
    logger.info("\n" + "="*60)
    logger.info("DEMO 4: Performance Test")
    logger.info("="*60)

    # Create production-size grid (small version)
    lat = np.linspace(-60, 60, 25)
    lon = np.linspace(-180, 180, 25)
    alt = np.linspace(60, 600, 30)

    logger.info(f"Grid size: {len(lat)}×{len(lon)}×{len(alt)} = {len(lat)*len(lon)*len(alt):,} points")

    ne_grid = create_chapman_layer(lat, lon, alt)

    # Initialize ray tracer
    import time
    t0 = time.time()
    tracer = RayTracer(ne_grid, lat, lon, alt)
    t1 = time.time()

    logger.info(f"\nInitialization time: {(t1-t0)*1000:.1f} ms")

    # Trace single ray
    t0 = time.time()
    path = tracer.trace_ray(0.0, 0.0, 85.0, 0.0, 5.0)
    t1 = time.time()

    logger.info(f"Single ray trace: {(t1-t0)*1000:.1f} ms")

    # Trace ray fan
    t0 = time.time()
    paths = tracer.trace_nvis(0.0, 0.0, 5.0, elevation_step=5.0, azimuth_step=30.0)
    t1 = time.time()

    logger.info(f"Ray fan ({len(paths)} rays): {(t1-t0)*1000:.1f} ms")
    logger.info(f"  Per ray: {(t1-t0)*1000/len(paths):.1f} ms")


def main():
    """Run all demonstrations."""
    logger.info("="*60)
    logger.info("Auto-NVIS Ray Tracer Demo")
    logger.info("Native C++ Implementation (No MATLAB!)")
    logger.info("="*60)

    try:
        demo_single_ray()
        demo_nvis_coverage()
        demo_luf_muf()
        demo_performance()

        logger.info("\n" + "="*60)
        logger.info("✓ All demos completed successfully!")
        logger.info("="*60)
        logger.info("\nNext steps:")
        logger.info("  1. Integrate with SR-UKF filter output")
        logger.info("  2. Add to system orchestrator")
        logger.info("  3. Publish products to RabbitMQ")
        logger.info("  4. View in dashboard")

    except Exception as e:
        logger.error(f"\n✗ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
