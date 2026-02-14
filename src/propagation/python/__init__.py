"""
Auto-NVIS Propagation Module

Pure Python/C++ implementation of 3D ionospheric ray tracing.
No MATLAB dependency required!

Main Components:
- RayTracer: High-level ray tracing interface
- ProductGenerators: LUF/MUF, coverage maps, blackout detection
- Integration with SR-UKF filter output

Example:
    from src.propagation.python import RayTracer, CoverageMap

    # Create ray tracer from SR-UKF output
    tracer = RayTracer(ne_grid, lat, lon, alt)

    # Calculate coverage
    coverage = tracer.calculate_coverage(tx_lat=40.0, tx_lon=-105.0)
    print(f"LUF: {coverage['luf']} MHz, MUF: {coverage['muf']} MHz")
"""

__version__ = "1.0.0"
__author__ = "Auto-NVIS Development Team"

from .pharlap_replacement import RayTracer, create_from_srukf_grid

__all__ = ['RayTracer', 'create_from_srukf_grid']
