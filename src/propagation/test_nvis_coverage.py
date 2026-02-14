#!/usr/bin/env python3
"""
Test NVIS coverage calculation with optimized ray tracer
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

# Stronger Chapman layer for better reflection
Ne_max = 1e12  # el/m³ (stronger ionosphere)
h_max = 300.0  # km (F-layer peak)
H = 50.0       # km (scale height)

ne_grid = np.zeros((len(lat), len(lon), len(alt)))
for i, h in enumerate(alt):
    z = (h - h_max) / H
    Ne = Ne_max * np.exp(1 - z - np.exp(-z))
    ne_grid[:, :, i] = Ne

print("=" * 60)
print("NVIS Coverage Test")
print("=" * 60)
print(f"Grid: {len(lat)}×{len(lon)}×{len(alt)}")
print(f"Max Ne: {Ne_max:.2e} el/m³ at {h_max} km")
print()

# Create ray tracer
tracer = RayTracer(ne_grid, lat, lon, alt)

# Test NVIS coverage (reduced number of rays for speed)
print("Calculating NVIS coverage...")
print("  Elevations: 70-90° in 5° steps")
print("  Azimuths: 0-360° in 30° steps")
print("  Frequency: 5 MHz")
print()

t0 = time.time()
paths = tracer.trace_nvis(
    tx_lat=0.0,
    tx_lon=0.0,
    freq_mhz=5.0,
    elevation_min=70.0,
    elevation_max=90.0,
    elevation_step=5.0,
    azimuth_step=30.0
)
t1 = time.time()

print("=" * 60)
print("Results:")
print("=" * 60)
print(f"Time: {(t1-t0)*1000:.1f} ms")
print(f"Rays traced: {len(paths)}")
print(f"Avg time per ray: {(t1-t0)*1000/len(paths):.2f} ms")
print()

# Analyze results
reflected = sum(1 for p in paths if p['reflected'])
escaped = sum(1 for p in paths if p['escaped'])
absorbed = sum(1 for p in paths if p['absorbed'])
other = len(paths) - reflected - escaped - absorbed

print("Ray outcomes:")
print(f"  Reflected: {reflected}/{len(paths)} ({100*reflected/len(paths):.1f}%)")
print(f"  Escaped: {escaped}/{len(paths)} ({100*escaped/len(paths):.1f}%)")
print(f"  Absorbed: {absorbed}/{len(paths)} ({100*absorbed/len(paths):.1f}%)")
print(f"  Other: {other}/{len(paths)} ({100*other/len(paths):.1f}%)")
print()

if reflected > 0:
    avg_range = np.mean([p['ground_range'] for p in paths if p['reflected']])
    print(f"Average ground range (reflected): {avg_range:.1f} km")

# Performance assessment
target_sec = 30.0
if (t1 - t0) < target_sec:
    status = "✅ PASS"
else:
    status = f"⚠️  SLOW (target: <{target_sec} sec)"

print()
print(f"Performance: {status}")
print()
