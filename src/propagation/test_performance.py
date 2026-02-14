#!/usr/bin/env python3
"""
Quick performance test for optimized ray tracer
"""

import sys
import time
import numpy as np

sys.path.insert(0, 'python')

from pharlap_replacement import RayTracer

# Create simple Chapman layer ionosphere
lat = np.linspace(-60, 60, 25)
lon = np.linspace(-180, 180, 25)
alt = np.linspace(60, 600, 25)

# Chapman layer parameters
Ne_max = 5e11  # el/m³ (moderate ionosphere)
h_max = 300.0  # km (F-layer peak)
H = 50.0       # km (scale height)

ne_grid = np.zeros((len(lat), len(lon), len(alt)))
for i, h in enumerate(alt):
    z = (h - h_max) / H
    Ne = Ne_max * np.exp(1 - z - np.exp(-z))
    ne_grid[:, :, i] = Ne

print("=" * 60)
print("Ray Tracer Performance Test (Optimized Parameters)")
print("=" * 60)
print(f"Grid: {len(lat)}×{len(lon)}×{len(alt)}")
print(f"Max Ne: {Ne_max:.2e} el/m³ at {h_max} km")
print()

# Create ray tracer
tracer = RayTracer(ne_grid, lat, lon, alt)

# Print configuration
print("Configuration:")
print(f"  Tolerance: {tracer.config.tolerance:.0e}")
print(f"  Initial step: {tracer.config.initial_step_km} km")
print(f"  Min step: {tracer.config.min_step_km} km")
print(f"  Max step: {tracer.config.max_step_km} km")
print(f"  Max steps: {tracer.config.max_steps}")
print()

# Test single ray
print("Tracing single ray (85° elevation, 5 MHz)...")
t0 = time.time()
path = tracer.trace_ray(0.0, 0.0, 85.0, 0.0, 5.0)
t1 = time.time()

print()
print("=" * 60)
print("Results:")
print("=" * 60)
print(f"Time: {(t1-t0)*1000:.1f} ms")
print(f"Points traced: {len(path['positions'])}")
print(f"Ground range: {path['ground_range']:.1f} km")
print(f"Apex altitude: {path['apex_altitude']:.1f} km")
print(f"Path length: {path['path_length']:.1f} km")
print(f"Reflected: {path['reflected']}")
print(f"Escaped: {path['escaped']}")
print()

# Performance assessment
target_ms = 50.0
if (t1 - t0) * 1000 < target_ms:
    status = "✅ PASS"
else:
    status = f"⚠️  SLOW (target: <{target_ms} ms)"

print(f"Performance: {status}")
print()
