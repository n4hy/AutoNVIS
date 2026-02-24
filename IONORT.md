# IONORT-Style Ray Tracing Implementation

**AutoNVIS IONORT Implementation Guide**

This document describes the IONORT-style ray tracing features implemented in AutoNVIS, based on the IONORT paper:

> "IONORT: Real-Time 3D Ray-Tracing for HF Propagation Through the Ionosphere"
> Remote Sensing 2023, 15(21), 5111
> https://www.mdpi.com/2072-4292/15/21/5111

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Numerical Integrators](#numerical-integrators)
  - [RK4 Integrator](#rk4-integrator)
  - [Adams-Bashforth/Moulton](#adams-bashforthmoulton-integrator)
  - [RK45 Dormand-Prince](#rk45-dormand-prince-integrator)
  - [Integrator Factory](#integrator-factory)
- [Homing Algorithm](#homing-algorithm)
  - [Winner Triplets](#winner-triplets)
  - [Search Space](#search-space)
  - [Landing Accuracy (Condition 10)](#landing-accuracy-condition-10)
  - [MUF/LUF/FOT Calculation](#mufluffot-calculation)
  - [Multi-Hop Propagation](#multi-hop-propagation)
- [IONORT Visualizations](#ionort-visualizations)
  - [Altitude vs Ground Range](#altitude-vs-ground-range-widget)
  - [3D Geographic View](#3d-geographic-view-widget)
  - [Synthetic Ionogram](#synthetic-ionogram-widget)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Testing](#testing)
- [References](#references)

---

## Overview

The IONORT implementation provides world-class 3D magnetoionic ray tracing capabilities:

### Key Features

| Feature | Description |
|---------|-------------|
| **Three Integrators** | RK4, Adams-Bashforth/Moulton, RK45 Dormand-Prince |
| **Pluggable Architecture** | Factory pattern for easy integrator selection |
| **Homing Algorithm** | Winner triplet search connecting Tx to Rx |
| **Parallel Tracing** | ThreadPoolExecutor for multi-core utilization |
| **IONORT Condition 10** | Landing accuracy check for winner validation |
| **Three Visualizations** | Professional IONORT-style displays |
| **51 Unit Tests** | Comprehensive test coverage |

### File Structure

```
src/raytracer/
├── integrators/
│   ├── __init__.py           # Package exports
│   ├── base.py               # BaseIntegrator, IntegrationStep, IntegrationStats
│   ├── rk4.py                # RK4 with step doubling
│   ├── adams_bashforth.py    # AB4/AM3 predictor-corrector
│   ├── rk45.py               # Dormand-Prince adaptive
│   └── factory.py            # IntegratorFactory, create_integrator()
├── homing_algorithm.py       # HomingAlgorithm, WinnerTriplet, HomingResult
├── haselgrove.py             # HaselgroveSolver (multi-hop, pluggable integrators)
├── link_budget.py            # LinkBudgetCalculator, SNR, path loss (~760 LOC)
└── ...

src/visualization/pyqt/raytracer/
├── ionort_widgets.py         # Three IONORT visualization widgets
└── ...

tests/unit/
├── test_integrators.py       # 26 integrator tests
└── test_homing_algorithm.py  # 25 homing algorithm tests
```

---

## Architecture

### Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        IONOSPHERIC MODEL                             │
│  IonosphericModel → ChapmanLayer → AppletonHartree → n(r, f, k)     │
└─────────────────────────────────────┬───────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      HASELGROVE SOLVER                               │
│  6-coupled ODEs: dr/ds = k/n, dk/ds = -∇n/n                         │
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │     RK4      │  │     ABM      │  │    RK45      │               │
│  │ 12 evals/step│  │ 2 evals/step │  │ 7 evals/step │               │
│  │ error track  │  │ efficient    │  │ adaptive     │               │
│  └──────────────┘  └──────────────┘  └──────────────┘               │
│           ↑               ↑               ↑                          │
│           └───────────────┼───────────────┘                          │
│                    IntegratorFactory                                 │
└─────────────────────────────────────┬───────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      HOMING ALGORITHM                                │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ Search Space: (frequency, elevation, azimuth_deviation)     │    │
│  │ Parallel tracing with ThreadPoolExecutor                    │    │
│  │ Landing accuracy check (Condition 10)                       │    │
│  │ Winner triplet collection                                   │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              ↓                                       │
│  HomingResult: winner_triplets, MUF, LUF, FOT                       │
└─────────────────────────────────────┬───────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    IONORT VISUALIZATIONS                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐      │
│  │ Altitude vs     │  │ 3D Geographic   │  │ Synthetic       │      │
│  │ Ground Range    │  │ View            │  │ Ionogram        │      │
│  │ (Figs 5,7,9)    │  │ (Figs 7,8)      │  │ (Figs 11-16)    │      │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Numerical Integrators

All integrators implement the `BaseIntegrator` abstract class and can be swapped interchangeably.

### Base Classes

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class IntegrationStep:
    """Result of a single integration step."""
    state: np.ndarray          # [x, y, z, kx, ky, kz]
    error_estimate: float      # Local truncation error
    step_size_used: float      # Actual step taken (km)
    derivatives_computed: int  # Number of f(y) evaluations
    accepted: bool             # Step accepted?

@dataclass
class IntegrationStats:
    """Statistics for complete ray trace."""
    total_steps: int = 0
    rejected_steps: int = 0
    total_derivative_evals: int = 0
    max_error: float = 0.0
    min_step_size: float = float('inf')
    max_step_size: float = 0.0

class BaseIntegrator(ABC):
    """Abstract base for all integrators."""

    @abstractmethod
    def step(self, state: np.ndarray, ds: float, freq_mhz: float) -> IntegrationStep:
        """Perform one integration step."""
        pass

    @abstractmethod
    def name(self) -> str:
        """Return integrator name."""
        pass
```

### RK4 Integrator

Classical 4th-order Runge-Kutta with step doubling for error estimation.

**Implementation**: `src/raytracer/integrators/rk4.py`

**Method**:
```
k1 = f(y)
k2 = f(y + h/2 * k1)
k3 = f(y + h/2 * k2)
k4 = f(y + h * k3)
y_new = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
```

**Error Estimation** (Richardson extrapolation):
```
y_full = RK4(y, h)              # One full step
y_half = RK4(RK4(y, h/2), h/2)  # Two half steps
error = |y_full - y_half| / 15
```

**Characteristics**:
- 12 derivative evaluations per step (4 + 4 + 4)
- Returns more accurate half-step solution
- Good for debugging and error tracking
- Reliable but computationally expensive

**Usage**:
```python
from src.raytracer.integrators import RK4Integrator

def haselgrove_derivs(state, freq):
    # ... compute derivatives ...
    return np.array([dx, dy, dz, dkx, dky, dkz])

integrator = RK4Integrator(haselgrove_derivs, tolerance=1e-6)
result = integrator.step(state, ds=1.0, freq_mhz=7.0)
```

### Adams-Bashforth/Moulton Integrator

4-step predictor / 3-step corrector multistep method (IONORT paper Section 2.2).

**Implementation**: `src/raytracer/integrators/adams_bashforth.py`

**Predictor (Adams-Bashforth 4-step)**:
```
y_{n+1}^p = y_n + h/24 * (55*f_n - 59*f_{n-1} + 37*f_{n-2} - 9*f_{n-3})
```

**Corrector (Adams-Moulton 3-step)**:
```
y_{n+1} = y_n + h/24 * (9*f(y_{n+1}^p) + 19*f_n - 5*f_{n-1} + f_{n-2})
```

**Characteristics**:
- Only 2 derivative evaluations per step (after startup)
- Requires RK4 startup for first 4 steps to build history
- Most efficient for long, smooth ray paths
- **CRITICAL**: Must call `reset()` before each new ray trace!

**Error Estimation**:
```
error = |y_corrected - y_predicted| * (9/270)
```

**Usage**:
```python
from src.raytracer.integrators import AdamsBashforthMoultonIntegrator

integrator = AdamsBashforthMoultonIntegrator(haselgrove_derivs, tolerance=1e-6)

# IMPORTANT: Reset before each new ray!
integrator.reset()

for step in range(num_steps):
    result = integrator.step(state, ds=1.0, freq_mhz=7.0)
    state = result.state

# Check efficiency
print(f"Derivative evaluations: {integrator.stats.total_derivative_evals}")
print(f"Efficiency vs RK4: {integrator.efficiency_ratio():.1f}x")
```

### RK45 Dormand-Prince Integrator

Embedded 4(5) pair with automatic step size adaptation.

**Implementation**: `src/raytracer/integrators/rk45.py`

**Butcher Tableau**:
```
    0    |
    1/5  | 1/5
    3/10 | 3/40       9/40
    4/5  | 44/45      -56/15      32/9
    8/9  | 19372/6561 -25360/2187 64448/6561 -212/729
    1    | 9017/3168  -355/33     46732/5247 49/176    -5103/18656
    1    | 35/384     0           500/1113   125/192   -2187/6784   11/84
    -----+------------------------------------------------------------
    5th  | 35/384     0           500/1113   125/192   -2187/6784   11/84     0
    4th  | 5179/57600 0           7571/16695 393/640   -92097/339200 187/2100 1/40
```

**Step Size Control**:
```
error = |y5 - y4|
h_new = safety * h * (tolerance / error)^(1/5)
```

**Characteristics**:
- 7 derivative evaluations per accepted step
- Automatic step size adaptation
- Best for paths with varying curvature (near reflection)
- Rejects and retries steps that exceed tolerance

**Usage**:
```python
from src.raytracer.integrators import RK45Integrator

integrator = RK45Integrator(
    haselgrove_derivs,
    tolerance=1e-7,
    safety=0.9,
    initial_step=1.0
)

while tracing:
    # Use suggested step from previous iteration
    ds = integrator.current_step
    result = integrator.step(state, ds, freq_mhz=7.0)

    if result.accepted:
        state = result.state
    # Step size automatically updated in integrator.current_step
```

### Integrator Factory

Create integrators by name or type using the factory pattern.

**Implementation**: `src/raytracer/integrators/factory.py`

**Available Names**:
| Name | Aliases | Integrator |
|------|---------|------------|
| `rk4` | `rk4_error` | RK4Integrator |
| `rk4_fast` | `rk4_simple` | RK4IntegratorFast |
| `adams_bashforth` | `abm`, `adams`, `predictor_corrector` | AdamsBashforthMoultonIntegrator |
| `rk45` | `dopri`, `dormand_prince`, `adaptive` | RK45Integrator |
| `rk45_fast` | | RK45IntegratorFast |

**Usage**:
```python
from src.raytracer.integrators import create_integrator, IntegratorFactory

# By name
integrator = create_integrator('rk45', derivative_func, tolerance=1e-7)

# List available
print(IntegratorFactory.available())  # ['rk4', 'rk4_fast', 'adams_bashforth', 'rk45', 'rk45_fast']

# Get description
print(IntegratorFactory.get_description('adams_bashforth'))

# Get recommended for path length
from src.raytracer.integrators.factory import get_recommended_integrator
integrator = get_recommended_integrator(path_length_km=2000, derivative_func, accuracy='high')
```

---

## Homing Algorithm

The homing algorithm finds all propagation paths connecting a transmitter to a receiver by systematically searching over frequency, elevation, and azimuth.

**Implementation**: `src/raytracer/homing_algorithm.py`

### Winner Triplets

A **winner triplet** is a (frequency, elevation, azimuth) combination that successfully propagates from Tx to Rx.

```python
@dataclass
class WinnerTriplet:
    frequency_mhz: float        # Operating frequency
    elevation_deg: float        # Launch elevation
    azimuth_deg: float          # Launch azimuth
    azimuth_deviation_deg: float  # Deviation from great circle
    group_delay_ms: float       # One-way delay
    ground_range_km: float      # Great circle distance
    landing_lat: float          # Where ray landed
    landing_lon: float
    landing_error_km: float     # Distance from Rx
    mode: PropagationMode       # O_MODE or X_MODE
    ray_path: Optional[RayPath] # Full path (if stored)
    reflection_height_km: float
    hop_count: int
```

### Search Space

Define the parameter ranges to search:

```python
@dataclass
class HomingSearchSpace:
    freq_range: Tuple[float, float] = (2.0, 30.0)  # MHz
    freq_step: float = 0.5
    elevation_range: Tuple[float, float] = (5.0, 89.0)  # degrees
    elevation_step: float = 2.0
    azimuth_deviation_range: Tuple[float, float] = (-15.0, 15.0)  # degrees
    azimuth_step: float = 5.0

# Calculate total triplets
search = HomingSearchSpace(freq_range=(3.0, 15.0), freq_step=0.5)
print(f"Total triplets: {search.total_triplets}")  # 25 * 43 * 7 = 7,525
```

### Landing Accuracy (Condition 10)

IONORT Condition (10) determines if a ray successfully reached the receiver:

**From paper (page 10)**:
```
|λ_ray - λ_rx| / ((π/2) - λ_rx) * 100 ≤ Latitude_Accuracy
|φ_ray - φ_rx| / φ_rx * 100 ≤ Longitude_Accuracy
```

**Implementation** (simplified distance-based):
```python
@dataclass
class HomingConfig:
    lat_tolerance_deg: float = 1.0
    lon_tolerance_deg: float = 1.0
    distance_tolerance_km: float = 100.0
    use_distance_tolerance: bool = True  # Use distance instead of lat/lon
    trace_both_modes: bool = True         # O and X modes
    store_ray_paths: bool = False         # Memory intensive
    max_workers: int = 4                  # Parallel workers
```

### MUF/LUF/FOT Calculation

Automatically computed from winner triplets:

```python
@dataclass
class HomingResult:
    winner_triplets: List[WinnerTriplet]
    muf: float  # Maximum Usable Frequency (highest winner)
    luf: float  # Lowest Usable Frequency (lowest winner)
    fot: float  # Frequency of Optimum Traffic (0.85 * MUF)
```

### Multi-Hop Propagation

For long-distance paths (>2000 km), multiple ionospheric reflections with ground bounces are required.

**Ground Reflection Model**:
```python
# Specular reflection at Earth's surface
k_new = k - 2 * (k · n̂) * n̂
```

**Configuration**:
```python
@dataclass
class HomingConfig:
    max_hops: int = 3  # Maximum ground bounces
    # ... other fields
```

**Link Budget with SNR**:

The `WinnerTriplet` now includes signal quality metrics:

```python
@dataclass
class WinnerTriplet:
    # ... existing fields ...
    snr_db: Optional[float]           # Signal-to-noise ratio
    signal_strength_dbm: Optional[float]  # Received signal level
    path_loss_db: Optional[float]     # Total path loss
    hop_count: int                    # Number of ground reflections
```

**Link Budget Calculator**:

```python
from src.raytracer.link_budget import LinkBudgetCalculator

calc = LinkBudgetCalculator(
    tx_config=TransmitterConfig(power_watts=100),
    rx_config=ReceiverConfig(bandwidth_hz=2400),
    tx_antenna=AntennaConfig(gain_dbi=2.15),
    rx_antenna=AntennaConfig(gain_dbi=6.0)
)

# Calculate SNR for a path
result = calc.calculate(
    frequency_mhz=7.0,
    path_length_km=4000,
    reflection_height_km=280,
    hop_count=2,
    tx_lat=34.0, tx_lon=-118.0
)

print(f"SNR: {result.snr_db:.1f} dB")
```

**SNR Filtering**:

Winners with SNR < 0 dB are automatically filtered out. While such paths are physically valid (the SNR is computed from real physics), they are unusable for practical communication. Only paths with positive SNR appear in the winner triplets table.

**Path Loss Components**:
- Free-space path loss (FSPL)
- D-layer absorption (varies with solar zenith angle)
- Ground reflection loss (~3 dB per hop)
- Deviative absorption (ray path through plasma)
- Ionospheric focusing gain (~2 dB)

### Usage Example

```python
from src.raytracer import IonosphericModel, HaselgroveSolver
from src.raytracer.homing_algorithm import (
    HomingAlgorithm, HomingSearchSpace, HomingConfig
)

# Setup
ionosphere = IonosphericModel()
ionosphere.update_from_realtime(foF2=10.0, hmF2=300.0)
solver = HaselgroveSolver(ionosphere, integrator_name='rk45')

# Configure homing
config = HomingConfig(
    distance_tolerance_km=75.0,
    max_workers=8,
    trace_both_modes=True
)
homing = HomingAlgorithm(solver, config)

# Define search space
search = HomingSearchSpace(
    freq_range=(3.0, 15.0),
    freq_step=0.5,
    elevation_range=(15.0, 85.0),
    elevation_step=2.0,
    azimuth_deviation_range=(-10.0, 10.0),
    azimuth_step=5.0
)

# Find paths
def progress(done, total):
    print(f"Progress: {done}/{total} ({100*done/total:.1f}%)")

result = homing.find_paths(
    tx_lat=40.0, tx_lon=-105.0,
    rx_lat=42.0, rx_lon=-100.0,
    search_space=search,
    progress_callback=progress
)

# Results
print(f"Winner triplets: {result.num_winners}")
print(f"MUF: {result.muf:.1f} MHz")
print(f"LUF: {result.luf:.1f} MHz")
print(f"FOT: {result.fot:.1f} MHz")
print(f"Computation time: {result.computation_time_s:.1f}s")

# Best paths
for w in result.winner_triplets[:5]:
    print(f"  {w.frequency_mhz:.1f} MHz at {w.elevation_deg:.0f}° "
          f"(error: {w.landing_error_km:.1f} km)")
```

---

## IONORT Visualizations

Three professional visualization widgets matching the IONORT paper figures.

**Implementation**: `src/visualization/pyqt/raytracer/ionort_widgets.py`

### Altitude vs Ground Range Widget

Like IONORT Figures 5, 7, 9.

**Features**:
- Ray paths in cross-section view
- Ionospheric layer shading (D: 60-90 km, E: 90-150 km, F1: 150-220 km, F2: 220-450 km)
- Rainbow frequency coloring (red=low MHz, blue=high MHz)
- Solid lines for reflected rays, dashed for escaped
- O-mode thicker than X-mode
- **Interactive frequency filter buttons** (click to show/hide frequency ranges)

**Frequency Filter Buttons**:
The frequency legend at the bottom consists of clickable toggle buttons:
- Click a frequency button to **exclude** that frequency range (button turns grey)
- Click again to **re-include** (button returns to original color)
- "Reset All" button restores all frequencies
- Filtering is instant (no re-tracing required)
- Frequency matching uses ±2.5 MHz tolerance

Available filter frequencies: 2, 5, 8, 12, 15, 20, 25, 30 MHz

```python
from src.visualization.pyqt.raytracer.ionort_widgets import AltitudeGroundRangeWidget

widget = AltitudeGroundRangeWidget()
widget.set_frequency_range(3.0, 15.0)

# Add ray paths
for path in ray_paths:
    widget.add_ray_path_from_positions(
        positions=[(lat, lon, alt) for state in path.states],
        tx_lat=40.0, tx_lon=-105.0,
        frequency_mhz=path.frequency_mhz,
        is_reflected=(path.termination == RayTermination.GROUND_HIT),
        is_o_mode=(path.mode == RayMode.ORDINARY)
    )

widget.auto_scale(max_range=500, max_alt=400)
```

### 3D Geographic View Widget

Like IONORT Figures 7, 8 perspective.

**Features**:
- Earth sphere mesh (6371 km radius)
- Lat/lon grid lines every 30°
- **Political boundaries** (toggleable checkbox, 20x interpolated resolution)
- **Depth occlusion** (boundaries hidden when over horizon)
- **Live camera display** (Az/El/Distance in large 33px font)
- Tx marker (red), Rx marker (green)
- Ray paths as 3D lines colored by frequency
- Interactive rotation/zoom

**Requires**: PyOpenGL (`pip install PyOpenGL`)

```python
from src.visualization.pyqt.raytracer.ionort_widgets import Geographic3DWidget

widget = Geographic3DWidget()

# Add markers
widget.add_marker(tx_lat, tx_lon, tx_alt, color='#ff4444', size=150)  # Tx
widget.add_marker(rx_lat, rx_lon, rx_alt, color='#44ff44', size=150)  # Rx

# Add ray paths
for path in ray_paths:
    positions = [(s.lat_lon_alt()) for s in path.states]
    widget.add_ray_path(positions, path.frequency_mhz, freq_min=3.0, freq_max=15.0)

# Focus camera
widget.focus_on(mid_lat, mid_lon)
```

### Synthetic Ionogram Widget

Like IONORT Figures 11-16.

**Features**:
- Group delay vs frequency plot
- O-mode trace (solid blue circles)
- X-mode trace (dashed pink X markers)
- MUF/LUF vertical line markers
- Winner triplets table with scrolling

```python
from src.visualization.pyqt.raytracer.ionort_widgets import SyntheticIonogramWidget

widget = SyntheticIonogramWidget()

# Set traces
o_freqs = [w.frequency_mhz for w in result.o_mode_winners]
o_delays = [w.group_delay_ms for w in result.o_mode_winners]
widget.set_o_mode_trace(o_freqs, o_delays)

x_freqs = [w.frequency_mhz for w in result.x_mode_winners]
x_delays = [w.group_delay_ms for w in result.x_mode_winners]
widget.set_x_mode_trace(x_freqs, x_delays)

# Set frequency markers
widget.set_muf(result.muf)
widget.set_luf(result.luf)

# Populate table
triplet_dicts = [
    {
        'frequency_mhz': w.frequency_mhz,
        'elevation_deg': w.elevation_deg,
        'azimuth_deg': w.azimuth_deg,
        'group_delay_ms': w.group_delay_ms,
        'mode': w.mode.value
    }
    for w in result.winner_triplets
]
widget.set_winner_triplets(triplet_dicts)
```

### Combined Panel

Use `IONORTVisualizationPanel` to get all three visualizations in a single widget:

```python
from src.visualization.pyqt.raytracer.ionort_widgets import IONORTVisualizationPanel

panel = IONORTVisualizationPanel()

# Update from homing result
panel.update_from_homing_result(result)

# Access individual widgets
panel.altitude_widget.set_info("Custom text")
panel.ionogram_widget.auto_scale()
```

---

## API Reference

### Integrators Module

```python
from src.raytracer.integrators import (
    # Base classes
    BaseIntegrator,
    IntegrationStep,
    IntegrationStats,

    # Integrators
    RK4Integrator,
    RK4IntegratorFast,
    AdamsBashforthMoultonIntegrator,
    RK45Integrator,
    RK45IntegratorFast,

    # Factory
    IntegratorFactory,
    create_integrator,
)
```

### Homing Algorithm Module

```python
from src.raytracer.homing_algorithm import (
    HomingAlgorithm,
    HomingResult,
    HomingSearchSpace,
    HomingConfig,
    WinnerTriplet,
    PropagationMode,  # O_MODE, X_MODE
)
```

### Visualization Widgets

```python
from src.visualization.pyqt.raytracer.ionort_widgets import (
    AltitudeGroundRangeWidget,
    Geographic3DWidget,
    SyntheticIonogramWidget,
    IONORTVisualizationPanel,
    frequency_to_color,
)
```

### HaselgroveSolver with Pluggable Integrators

```python
from src.raytracer import HaselgroveSolver, IonosphericModel

# Method 1: Use default RK4 (backward compatible)
solver = HaselgroveSolver(ionosphere)

# Method 2: Specify integrator by name
solver = HaselgroveSolver(ionosphere, integrator_name='rk45')
solver = HaselgroveSolver(ionosphere, integrator_name='adams_bashforth')

# Method 3: Provide custom integrator instance
from src.raytracer.integrators import RK45Integrator

custom_rk45 = RK45Integrator(
    solver._derivatives_array,
    tolerance=1e-8,
    safety=0.95
)
solver = HaselgroveSolver(ionosphere, integrator=custom_rk45)
```

---

## Examples

### Live Dashboard

The fastest way to use the IONORT features is the interactive dashboard:

```bash
# Simple dashboard (recommended for first use)
python scripts/ionort_simple.py

# Full-featured dashboard with command-line options
python scripts/ionort_live_demo.py --tx 40.0,-105.0 --rx 35.0,-106.0 --freq 3,15

# NVIS mode (short range, high elevation angles)
python scripts/ionort_live_demo.py --nvis
```

**Dashboard Features**:
- Configurable Tx/Rx positions
- Adjustable frequency and elevation ranges
- Ionosphere parameters (foF2, hmF2)
- Integrator selection (RK4, ABM, RK45)
- Progress bar during ray tracing
- Three synchronized IONORT visualizations
- Political boundaries on 3D globe (toggleable)
- Live camera position display

### Example 1: Compare Integrator Efficiency

```python
from src.raytracer import IonosphericModel, HaselgroveSolver, RayMode
from src.raytracer.integrators import create_integrator
import time

ionosphere = IonosphericModel()
ionosphere.update_from_realtime(foF2=10.0, hmF2=300.0)

results = {}

for name in ['rk4', 'adams_bashforth', 'rk45']:
    solver = HaselgroveSolver(ionosphere, integrator_name=name)

    start = time.time()
    paths = solver.trace_fan(
        tx_lat=40.0, tx_lon=-105.0, tx_alt=0.0,
        elevation_range=(30.0, 80.0),
        elevation_step=5.0,
        azimuth=0.0,
        frequency_mhz=7.0
    )
    elapsed = time.time() - start

    results[name] = {
        'time': elapsed,
        'paths': len(paths),
        'reflected': sum(1 for p in paths if p.termination.value == 'ground')
    }

for name, data in results.items():
    print(f"{name}: {data['time']:.3f}s, {data['reflected']}/{data['paths']} reflected")
```

### Example 2: NVIS Frequency Optimization

```python
from src.raytracer import IonosphericModel, HaselgroveSolver
from src.raytracer.homing_algorithm import HomingAlgorithm

ionosphere = IonosphericModel()
ionosphere.update_from_realtime(foF2=8.5, hmF2=320.0)

solver = HaselgroveSolver(ionosphere, integrator_name='rk45')
homing = HomingAlgorithm(solver)

# Find optimal NVIS frequencies
result = homing.find_optimal_nvis(
    tx_lat=40.0,
    tx_lon=-105.0,
    radius_km=200.0,
    freq_range=(2.0, 12.0),
    freq_step=0.25
)

print(f"NVIS MUF: {result.muf:.2f} MHz")
print(f"NVIS LUF: {result.luf:.2f} MHz")
print(f"Optimal (FOT): {result.fot:.2f} MHz")
print(f"Winner elevations: {sorted(set(w.elevation_deg for w in result.winner_triplets))}")
```

### Example 3: Full Visualization Pipeline

```python
from PyQt6.QtWidgets import QApplication
import sys

from src.raytracer import IonosphericModel, HaselgroveSolver
from src.raytracer.homing_algorithm import HomingAlgorithm, HomingSearchSpace, HomingConfig
from src.visualization.pyqt.raytracer.ionort_widgets import IONORTVisualizationPanel

# Setup
ionosphere = IonosphericModel()
ionosphere.update_from_realtime(foF2=10.0, hmF2=300.0)

solver = HaselgroveSolver(ionosphere, integrator_name='rk45')

config = HomingConfig(
    distance_tolerance_km=100.0,
    store_ray_paths=True,  # Need paths for visualization
    max_workers=4
)
homing = HomingAlgorithm(solver, config)

# Run homing
result = homing.find_paths(
    tx_lat=40.0, tx_lon=-105.0,
    rx_lat=42.0, rx_lon=-100.0,
    search_space=HomingSearchSpace(
        freq_range=(5.0, 12.0),
        freq_step=0.5,
        elevation_range=(20.0, 75.0),
        elevation_step=5.0
    )
)

# Display
app = QApplication(sys.argv)
panel = IONORTVisualizationPanel()
panel.update_from_homing_result(result)
panel.show()
sys.exit(app.exec())
```

---

## Testing

### Run Unit Tests

```bash
# Run all IONORT tests
python -m pytest tests/unit/test_integrators.py tests/unit/test_homing_algorithm.py -v

# Run just integrator tests
python -m pytest tests/unit/test_integrators.py -v

# Run just homing tests
python -m pytest tests/unit/test_homing_algorithm.py -v
```

### Test Coverage

| Test Suite | Tests | Description |
|------------|-------|-------------|
| TestRK4Integrator | 5 | Initialization, accuracy, error tracking |
| TestAdamsBashforthMoultonIntegrator | 5 | Startup, efficiency, reset |
| TestRK45Integrator | 6 | Butcher tableau, adaptation, accuracy |
| TestIntegratorFactory | 7 | Factory methods, names, recommendations |
| TestIntegrationComparison | 2 | Cross-integrator validation |
| TestHaselgroveLikeSystem | 1 | 6-component state handling |
| TestWinnerTriplet | 2 | Data structure |
| TestHomingSearchSpace | 4 | Parameter calculations |
| TestHomingConfig | 2 | Configuration |
| TestHomingResult | 3 | Result structure |
| TestHomingAlgorithmGeometry | 6 | Great circle, landing accuracy |
| TestHomingAlgorithmTripletGeneration | 4 | Triplet generation |
| TestHomingAlgorithmMUFCalculation | 2 | Frequency calculation |
| TestPropagationMode | 2 | Mode enum |

**Total: 51 tests**

---

## References

### IONORT Paper

> Settimi, A.; Pezzopane, M.; Pietrella, M.; Bianchi, C.; Scotto, C.; Zuccheretti, E.
> "IONORT: Real-Time 3D Ray-Tracing for HF Propagation Through the Ionosphere"
> Remote Sensing 2023, 15(21), 5111
> https://doi.org/10.3390/rs15215111

### Key Sections Referenced

- **Section 2.2**: Adams-Bashforth/Moulton predictor-corrector
- **Section 4**: Homing algorithm
- **Equation (10)**: Landing accuracy condition
- **Figures 5, 7, 9**: Altitude vs ground range
- **Figures 7, 8**: 3D geographic view
- **Figures 11-16**: Synthetic oblique ionogram

### Numerical Methods

- Dormand, J.R.; Prince, P.J. (1980). "A family of embedded Runge-Kutta formulae"
- Hairer, E.; Norsett, S.P.; Wanner, G. (1993). "Solving Ordinary Differential Equations I"
- Butcher, J.C. (2003). "Numerical Methods for Ordinary Differential Equations"

### Haselgrove Ray Tracing

- Haselgrove, J. (1957). "Oblique Ray Paths in the Ionosphere"
- Jones, R.M.; Stephenson, J.J. (1975). "A Versatile 3D Ray Tracing Program"

---

## Changelog

### v0.3.4 (February 24, 2026)

- **Added**: SNR filtering - paths with SNR < 0 dB are excluded from winners
  - Negative SNR paths are physically computed but unusable for communication
  - Prevents misleading negative SNR values in winner triplets table
  - Applied to demo scripts and homing algorithm
- **Added**: Tolerance parameter control in live dashboard UI
  - Configurable landing tolerance (10-500 km)
  - Exposed in control panel as spin box
- **Improved**: Integrator dropdown ordered by speed (ABM > RK45 > RK4)
  - ABM default (fastest: 2 evals/step)
  - RK45 medium (7 evals/step, most accurate)
  - RK4 slow (12 evals/step)
  - Tooltips explain trade-offs
- **Fixed**: Auto-scale now uses 90% of window for ray traces
  - Ray paths fill display area properly
  - Both width and height optimized

### v0.3.3 (February 23, 2026)

- **Added**: Interactive frequency filter buttons in altitude/range widget
  - Click to toggle frequency ranges on/off
  - Instant re-rendering without re-tracing
  - "Reset All" button to restore all frequencies
- **Added**: FrequencyFilterButton widget class with toggle styling

### v0.3.2 (February 23, 2026)

- **Added**: Multi-hop ray tracing with ground reflection for long-distance paths
- **Added**: Comprehensive link budget calculator (~760 LOC)
  - Free-space path loss
  - D-layer absorption (solar zenith angle, X-ray flux dependent)
  - Ground reflection loss per hop
  - Deviative absorption
  - Ionospheric focusing gain
  - ITU-R P.372 noise model (atmospheric, galactic, man-made)
- **Added**: SNR, signal strength, and path loss fields in WinnerTriplet
- **Added**: Radio configuration UI (TX power, antenna gains, RX bandwidth)
- **Added**: Diagnostic console in live dashboard (copyable log output)
- **Added**: GIRO ionosonde client for real-time foF2/hmF2
- **Added**: Live ionosphere client with auto-reconnect
- **Fixed**: Best SNR path visualization bug
- **Modified**: HaselgroveSolver with `_reflect_at_ground()` method
- **Modified**: Increased MAX_PATH_LENGTH_KM from 5000 to 15000

### v0.3.0 (February 22, 2026)

- **Added**: Three numerical integrators (RK4, Adams-Bashforth/Moulton, RK45)
- **Added**: Integrator factory with name-based creation
- **Added**: Pluggable integrator support in HaselgroveSolver
- **Added**: IONORT-style homing algorithm with winner triplets
- **Added**: Parallel ray tracing with ThreadPoolExecutor
- **Added**: Landing accuracy check (IONORT Condition 10)
- **Added**: Three IONORT visualization widgets
- **Added**: 51 unit tests for integrators and homing
- **Modified**: HaselgroveSolver to accept integrator parameter

---

**Last Updated**: February 24, 2026
**Version**: 0.3.4
