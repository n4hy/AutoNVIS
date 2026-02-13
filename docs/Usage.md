# Usage

**Auto-NVIS System Usage Guide**

**Document Version:** 1.0
**Last Updated:** February 12, 2026

---

## Table of Contents

1. [Getting Started](#1-getting-started)
2. [Basic Usage](#2-basic-usage)
3. [Python API Reference](#3-python-api-reference)
4. [C++ API Reference](#4-c-api-reference)
5. [Configuration](#5-configuration)
6. [Operational Modes](#6-operational-modes)
7. [Working with Data](#7-working-with-data)
8. [Advanced Usage](#8-advanced-usage)
9. [Examples](#9-examples)
10. [Best Practices](#10-best-practices)
11. [Performance Tuning](#11-performance-tuning)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Getting Started

### 1.1 Prerequisites

Before using Auto-NVIS, ensure you have:

- ✅ Completed installation (see [Installation.md](Installation.md))
- ✅ Verified basic functionality with tests
- ✅ Activated Python virtual environment (if using one)

```bash
# Activate virtual environment
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows

# Verify installation
python3 -c "import sys; sys.path.insert(0, 'src/assimilation/python'); import autonvis_srukf; print('✓ Ready')"
```

### 1.2 Directory Structure

Familiarize yourself with key directories:

```
AutoNVIS/
├── demo_standalone.py          # Standalone demonstration
├── demo_autonomous_system.py   # Full system demonstration
├── src/
│   ├── assimilation/
│   │   ├── python/
│   │   │   ├── autonvis_filter.py          # Main Python API
│   │   │   ├── test_basic_integration.py   # Integration tests
│   │   │   └── autonvis_srukf.*.so         # C++ module
│   │   ├── models/
│   │   │   └── chapman_layer.py            # Physics model
│   │   └── bindings/
│   │       └── python_bindings.cpp         # pybind11 bindings
│   └── supervisor/
│       ├── mode_controller.py              # Mode switching logic
│       └── filter_orchestrator.py          # System orchestrator
└── docs/
    ├── Usage.md                            # This file
    ├── Installation.md
    └── TheoreticalUnderpinnings.md
```

---

## 2. Basic Usage

### 2.1 Hello World: Simple Filter Cycle

```python
import sys
import numpy as np
from datetime import datetime

sys.path.insert(0, 'src/assimilation/python')
from autonvis_filter import AutoNVISFilter, OperationalMode
from src.assimilation.models.chapman_layer import ChapmanLayerModel

# Define grid (small for testing)
lat_grid = np.linspace(30, 40, 5)    # 5 latitude points
lon_grid = np.linspace(-100, -90, 5) # 5 longitude points
alt_grid = np.linspace(100, 500, 9)  # 9 altitude points

# Create filter
filter = AutoNVISFilter(
    n_lat=len(lat_grid),
    n_lon=len(lon_grid),
    n_alt=len(alt_grid)
)

# Generate background state using Chapman layer
chapman = ChapmanLayerModel()
time = datetime(2026, 3, 21, 18, 0, 0)
ne_grid_3d = chapman.compute_3d_grid(lat_grid, lon_grid, alt_grid, time, ssn=75)

# Flatten to state vector
state_dim = len(lat_grid) * len(lon_grid) * len(alt_grid) + 1
initial_state = np.zeros(state_dim)
initial_state[:-1] = ne_grid_3d.flatten()
initial_state[-1] = 75.0  # R_eff (sunspot number)

# Initial uncertainty (10%)
initial_sqrt_cov = np.diag(0.1 * np.abs(initial_state))

# Initialize filter
filter.initialize(
    lat_grid, lon_grid, alt_grid,
    initial_state, initial_sqrt_cov
)

# Set mode
filter.set_mode(OperationalMode.QUIET)

# Run one cycle (15 minutes = 900 seconds)
result = filter.run_cycle(dt=900.0)

print(f"Cycle complete!")
print(f"  Mode: {result['mode']}")
print(f"  Smoother active: {result['smoother_active']}")
print(f"  Predict time: {result['predict_time_ms']:.2f} ms")

# Get current state
ne_grid = filter.get_state_grid()
print(f"\nElectron density grid:")
print(f"  Min: {ne_grid.min():.2e} el/m³")
print(f"  Max: {ne_grid.max():.2e} el/m³")
print(f"  Mean: {ne_grid.mean():.2e} el/m³")
```

### 2.2 Running the Demonstration

The easiest way to see Auto-NVIS in action:

```bash
python3 demo_standalone.py
```

This demonstrates:
- 2-hour autonomous operation (9 cycles)
- Mode switching (QUIET → SHOCK → QUIET)
- Conditional smoother activation
- Filter stability verification

**Output Highlights:**
```
Grid: 5×5×9 = 225 points
...
MODE SWITCH: QUIET → SHOCK
  Smoother: DISABLED

MODE SWITCH: SHOCK → QUIET
  Smoother: ENABLED (if uncertainty > threshold)

✓ CRITICAL REQUIREMENT MET:
  Smoother NEVER activated during SHOCK mode
```

---

## 3. Python API Reference

### 3.1 AutoNVISFilter Class

**Location:** `src/assimilation/python/autonvis_filter.py`

#### Constructor

```python
filter = AutoNVISFilter(
    n_lat,                          # Number of latitude points
    n_lon,                          # Number of longitude points
    n_alt,                          # Number of altitude points
    alpha=1e-3,                     # UKF scaling (spread of sigma points)
    beta=2.0,                       # UKF distribution parameter
    kappa=0.0,                      # UKF parameter (typically 0 or 3-L)
    uncertainty_threshold=1e12,     # Smoother activation threshold
    localization_radius_km=500.0    # Gaspari-Cohn radius
)
```

**Parameters:**
- `alpha` (float): Controls sigma point spread. Smaller = tighter around mean. Range: [1e-4, 1]
- `beta` (float): Incorporates distribution knowledge. 2 = optimal for Gaussian
- `kappa` (float): Secondary scaling. 0 or 3-L are common choices
- `uncertainty_threshold` (float): trace(P) threshold for smoother activation
- `localization_radius_km` (float): Correlation distance (typically 300-1000 km)

#### Methods

**initialize()**

```python
filter.initialize(
    lat_grid,              # 1D array of latitudes (degrees)
    lon_grid,              # 1D array of longitudes (degrees)
    alt_grid,              # 1D array of altitudes (km)
    initial_state,         # State vector (NumPy array)
    initial_sqrt_cov,      # Square-root covariance (NumPy 2D array)
    correlation_time=3600.0,     # Gauss-Markov τ (seconds)
    process_noise_std=1e10       # Process noise σ (el/m³)
)
```

**set_mode()**

```python
filter.set_mode(OperationalMode.QUIET)  # Normal operations
# or
filter.set_mode(OperationalMode.SHOCK)  # Solar flare response
```

**should_use_smoother()**

```python
activate = filter.should_use_smoother()  # Returns bool

# Activation logic:
# - Returns False if mode == SHOCK (always)
# - Returns True if trace(P) > threshold and mode == QUIET
```

**predict()**

```python
filter.predict(dt=900.0)  # Time step in seconds
```

Propagates state forward using physics model (Gauss-Markov).

**update()**

```python
filter.update(
    observations,     # Observation vector (NumPy array)
    obs_sqrt_cov,     # Observation error sqrt covariance
    obs_model         # Observation model object
)
```

Assimilates observations into state estimate.

**run_cycle()**

```python
result = filter.run_cycle(
    dt=900.0,              # Time step (seconds)
    observations=None,     # Optional: observation vector
    obs_sqrt_cov=None,     # Optional: observation error
    obs_model=None         # Optional: observation model
)

# Returns dictionary:
# {
#     'cycle': int,
#     'mode': str ('QUIET' or 'SHOCK'),
#     'smoother_active': bool,
#     'predict_time_ms': float,
#     'update_time_ms': float,
#     'inflation_factor': float,
#     'last_nis': float,
#     'avg_nis': float,
#     'divergence_count': int,
#     'timestamp': str (ISO format)
# }
```

**get_state_grid()**

```python
ne_grid = filter.get_state_grid()  # Returns (n_lat, n_lon, n_alt) array
```

Extracts 3D electron density grid from state vector.

**get_effective_ssn()**

```python
reff = filter.get_effective_ssn()  # Returns float
```

Gets effective sunspot number (solar activity proxy).

**get_uncertainty()**

```python
uncertainty = filter.get_uncertainty()  # Returns trace(P)
```

Computes total uncertainty as trace of covariance matrix.

**get_statistics()**

```python
stats = filter.get_statistics()

# Returns dictionary:
# {
#     'cycle_count': int,
#     'smoother_activation_count': int,
#     'smoother_activation_rate': float (0-1),
#     'predict_count': int,
#     'update_count': int,
#     'avg_predict_time_ms': float,
#     'avg_update_time_ms': float,
#     'inflation_factor': float,
#     'avg_nis': float,
#     'divergence_count': int,
#     'min_eigenvalue': float,
#     'max_eigenvalue': float,
#     'current_uncertainty': float,
#     'current_mode': str,
#     'last_update': str (ISO timestamp)
# }
```

**apply_inflation()**

```python
filter.apply_inflation(factor=1.2)  # Manually inflate covariance
```

Manually apply covariance inflation (factor > 1.0).

### 3.2 Chapman Layer Model

**Location:** `src/assimilation/models/chapman_layer.py`

#### Constructor

```python
from src.assimilation.models.chapman_layer import ChapmanLayerModel

chapman = ChapmanLayerModel()
```

#### Methods

**compute_fof2_hmf2()**

```python
fof2, hmf2 = chapman.compute_fof2_hmf2(
    latitude=37.9,                      # Degrees North
    longitude=-75.5,                    # Degrees East
    time=datetime(2026, 3, 21, 18, 0),  # UTC datetime
    ssn=75.0                            # Sunspot number
)

# Returns:
# fof2: F2 critical frequency (MHz)
# hmf2: F2 peak height (km)
```

**compute_ne_profile()**

```python
alt_grid = np.arange(60, 600, 5)  # 60 to 600 km
ne_profile = chapman.compute_ne_profile(
    latitude=37.9,
    longitude=-75.5,
    altitude_km=alt_grid,
    time=datetime(2026, 3, 21, 18, 0),
    ssn=75.0
)

# Returns: 1D array of electron density (el/m³)
```

**compute_3d_grid()**

```python
ne_grid_3d = chapman.compute_3d_grid(
    lat_grid=np.linspace(-60, 60, 73),
    lon_grid=np.linspace(-180, 180, 73),
    alt_grid=np.linspace(60, 600, 55),
    time=datetime(2026, 3, 21, 18, 0),
    ssn=75.0
)

# Returns: 3D array (n_lat, n_lon, n_alt) of electron density
```

**validate_grid()**

```python
metrics = chapman.validate_grid(ne_grid_3d)

# Returns dictionary:
# {
#     'min_ne': float,
#     'max_ne': float,
#     'mean_ne': float,
#     'median_ne': float,
#     'invalid_count': int  # Values outside [1e8, 1e13]
# }
```

### 3.3 Operational Modes

**Enumeration:**

```python
from autonvis_filter import OperationalMode

OperationalMode.QUIET  # Normal operations
OperationalMode.SHOCK  # Solar flare response
```

**Mode Characteristics:**

| Mode | Trigger | Smoother | Physics | Update Interval |
|------|---------|----------|---------|-----------------|
| QUIET | X-ray < 1e-5 W/m² | Allowed* | Gauss-Markov | 15 min |
| SHOCK | X-ray ≥ 1e-5 W/m² | Disabled | Physics-based** | 15 min |

\* If uncertainty > threshold
\** Future enhancement

---

## 4. C++ API Reference

### 4.1 Direct C++ Usage (Advanced)

For direct C++ usage (without Python wrapper):

#### StateVector

```cpp
#include "state_vector.hpp"

autonvis::StateVector state(n_lat, n_lon, n_alt);

// Set electron density
state.set_ne(i, j, k, ne_value);

// Get electron density
double ne = state.get_ne(i, j, k);

// Set effective sunspot number
state.set_reff(75.0);

// Convert to/from Eigen vector
Eigen::VectorXd vec = state.to_vector();
state.from_vector(vec);
```

#### SquareRootUKF

```cpp
#include "sr_ukf.hpp"

autonvis::SquareRootUKF filter(
    n_lat, n_lon, n_alt,
    alpha=1e-3, beta=2.0, kappa=0.0
);

// Initialize
filter.initialize(initial_state, initial_sqrt_cov);

// Set physics model
auto physics = std::make_shared<autonvis::GaussMarkovModel>(3600.0, 1e10);
filter.set_physics_model(physics);

// Predict
filter.predict(dt);

// Update
filter.update(obs_model, observations, obs_sqrt_cov);

// Get state
const auto& state = filter.get_state();
const auto& sqrt_cov = filter.get_sqrt_cov();
```

#### Configuration

```cpp
// Adaptive inflation
autonvis::SquareRootUKF::AdaptiveInflationConfig inflation_cfg;
inflation_cfg.enabled = true;
inflation_cfg.initial_inflation = 1.0;
inflation_cfg.min_inflation = 1.0;
inflation_cfg.max_inflation = 2.0;
inflation_cfg.adaptation_rate = 0.95;
inflation_cfg.divergence_threshold = 3.0;

filter.set_adaptive_inflation_config(inflation_cfg);

// Covariance localization
autonvis::SquareRootUKF::LocalizationConfig loc_cfg;
loc_cfg.enabled = true;
loc_cfg.radius_km = 500.0;
loc_cfg.precompute = true;

filter.set_localization_config(loc_cfg, lat_grid, lon_grid, alt_grid);
```

### 4.2 Python Bindings (Recommended)

Access C++ functionality through pybind11:

```python
import sys
sys.path.insert(0, 'src/assimilation/python')
import autonvis_srukf as srukf

# All C++ classes available in Python
state = srukf.StateVector(5, 5, 7)
filter = srukf.SquareRootUKF(5, 5, 7)
model = srukf.GaussMarkovModel(3600.0, 1e10)

# Configuration
inflation_cfg = srukf.AdaptiveInflationConfig()
inflation_cfg.enabled = True
filter.set_adaptive_inflation_config(inflation_cfg)
```

---

## 5. Configuration

### 5.1 Filter Parameters

**Grid Configuration:**

```python
# Small grid (testing/development)
lat_grid = np.linspace(30, 40, 5)    # 5 points
lon_grid = np.linspace(-100, -90, 5) # 5 points
alt_grid = np.linspace(100, 500, 9)  # 9 points
# State dimension: 5×5×9 + 1 = 226

# Medium grid (regional)
lat_grid = np.linspace(20, 50, 16)   # 16 points
lon_grid = np.linspace(-120, -70, 26) # 26 points
alt_grid = np.linspace(60, 600, 28)  # 28 points
# State dimension: 16×26×28 + 1 = 11,649

# Full grid (global)
lat_grid = np.linspace(-60, 60, 73)     # 73 points
lon_grid = np.linspace(-180, 180, 73)   # 73 points
alt_grid = np.linspace(60, 600, 55)     # 55 points
# State dimension: 73×73×55 + 1 = 293,097
```

**UKF Parameters:**

```python
# Conservative (tight sigma points)
filter = AutoNVISFilter(..., alpha=1e-4, beta=2.0, kappa=0.0)

# Standard (recommended)
filter = AutoNVISFilter(..., alpha=1e-3, beta=2.0, kappa=0.0)

# Aggressive (wider spread)
filter = AutoNVISFilter(..., alpha=1e-1, beta=2.0, kappa=0.0)
```

**Localization Radius:**

```python
# Tight localization (300 km)
filter = AutoNVISFilter(..., localization_radius_km=300.0)
# Memory: Lower, but may lose long-range correlations

# Standard (500 km, recommended)
filter = AutoNVISFilter(..., localization_radius_km=500.0)
# Balance of memory and accuracy

# Wide localization (1000 km)
filter = AutoNVISFilter(..., localization_radius_km=1000.0)
# Memory: Higher, captures more correlations
```

**Uncertainty Threshold:**

```python
# Low threshold (smoother activates often)
filter = AutoNVISFilter(..., uncertainty_threshold=1e11)

# Standard (recommended)
filter = AutoNVISFilter(..., uncertainty_threshold=1e12)

# High threshold (smoother rarely activates)
filter = AutoNVISFilter(..., uncertainty_threshold=1e13)
```

### 5.2 Physics Model Parameters

**Gauss-Markov Model:**

```python
filter.initialize(
    ...,
    correlation_time=3600.0,    # 1 hour (standard)
    process_noise_std=1e10      # el/m³ (conservative)
)

# Shorter correlation (faster decorrelation)
correlation_time=1800.0  # 30 minutes

# Lower noise (smoother evolution)
process_noise_std=1e9
```

### 5.3 Chapman Layer Parameters

**Default Parameters (Good for most cases):**

```python
chapman = ChapmanLayerModel()  # Uses defaults

# Defaults:
# - H = 50 km (scale height)
# - hmF2_base = 300 km
# - NmF2_base = 5e11 el/m³
# - Equatorial enhancement = 30%
```

**Custom Parameters:**

```python
chapman = ChapmanLayerModel()
chapman.H = 60.0                    # Thicker layer
chapman.hmF2_base = 280.0           # Lower F2 peak
chapman.NmF2_base = 6e11            # Higher density
chapman.equatorial_enhancement = 0.4  # 40% enhancement
```

---

## 6. Operational Modes

### 6.1 Mode Switching Logic

**Automatic Mode Switching (with Mode Controller):**

```python
from src.supervisor.mode_controller import ModeController, OperationalMode

# Create mode controller
mode_ctrl = ModeController(
    xray_threshold=1e-5,     # M1 class flare
    hysteresis_sec=600       # 10-minute hysteresis
)

# Update with X-ray flux
new_mode = mode_ctrl.update(flux=5e-5, timestamp=datetime.utcnow())

# Sync with filter
if new_mode == OperationalMode.SHOCK:
    filter.set_mode(autonvis_filter.OperationalMode.SHOCK)
else:
    filter.set_mode(autonvis_filter.OperationalMode.QUIET)
```

**Manual Mode Switching:**

```python
# Monitor space weather manually
if xray_flux >= 1e-5:  # M1+ flare detected
    filter.set_mode(OperationalMode.SHOCK)
    print("Switched to SHOCK mode (solar flare)")
else:
    filter.set_mode(OperationalMode.QUIET)
    print("Switched to QUIET mode (normal)")
```

### 6.2 Mode-Specific Behavior

**QUIET Mode:**
```python
filter.set_mode(OperationalMode.QUIET)

# Check if smoother will activate
if filter.should_use_smoother():
    print("Smoother will activate (uncertainty high)")
else:
    print("Smoother disabled (uncertainty low)")

# Run cycle
result = filter.run_cycle(dt=900.0)
print(f"Smoother active: {result['smoother_active']}")
```

**SHOCK Mode:**
```python
filter.set_mode(OperationalMode.SHOCK)

# Smoother NEVER activates
assert not filter.should_use_smoother()

# Run cycle
result = filter.run_cycle(dt=900.0)
assert not result['smoother_active']  # Always False
```

### 6.3 Hysteresis

To prevent rapid mode oscillation:

```python
class HysteresisController:
    def __init__(self, quiet_to_shock=1e-5, shock_to_quiet=1e-5, delay_sec=600):
        self.quiet_to_shock = quiet_to_shock
        self.shock_to_quiet = shock_to_quiet
        self.delay_sec = delay_sec
        self.current_mode = OperationalMode.QUIET
        self.transition_time = None

    def update(self, flux, time):
        if self.current_mode == OperationalMode.QUIET:
            if flux >= self.quiet_to_shock:
                # Immediate transition to SHOCK
                self.current_mode = OperationalMode.SHOCK
                self.transition_time = time
        else:  # SHOCK mode
            if flux < self.shock_to_quiet:
                if self.transition_time is None:
                    self.transition_time = time
                elif (time - self.transition_time).total_seconds() >= self.delay_sec:
                    # Delayed transition to QUIET
                    self.current_mode = OperationalMode.QUIET
                    self.transition_time = None
            else:
                self.transition_time = None

        return self.current_mode
```

---

## 7. Working with Data

### 7.1 State Vector Format

**Structure:**

```
State Vector (length L = n_lat × n_lon × n_alt + 1):

Index 0:           Ne[0,0,0]
Index 1:           Ne[0,0,1]
...
Index k:           Ne[i,j,k]
...
Index L-2:         Ne[n_lat-1, n_lon-1, n_alt-1]
Index L-1:         R_eff
```

**Conversion:**

```python
# State vector → 3D grid
ne_grid = filter.get_state_grid()  # (n_lat, n_lon, n_alt)

# 3D grid → State vector
state_vector = np.zeros(n_lat * n_lon * n_alt + 1)
state_vector[:-1] = ne_grid.flatten()
state_vector[-1] = reff
```

**Accessing Specific Locations:**

```python
# Get Ne at specific lat/lon/alt
i = np.argmin(np.abs(lat_grid - target_lat))
j = np.argmin(np.abs(lon_grid - target_lon))
k = np.argmin(np.abs(alt_grid - target_alt))

ne_value = ne_grid[i, j, k]

# Or using C++ StateVector directly
import autonvis_srukf as srukf
state = filter.filter.get_state()
ne_value = state.get_ne(i, j, k)
```

### 7.2 Saving/Loading State

**NumPy Format:**

```python
# Save state
np.savez_compressed(
    'state_backup.npz',
    state=filter.filter.get_state().to_numpy(),
    sqrt_cov=filter.filter.get_sqrt_cov(),
    lat_grid=lat_grid,
    lon_grid=lon_grid,
    alt_grid=alt_grid,
    reff=filter.get_effective_ssn(),
    timestamp=str(datetime.utcnow())
)

# Load state
data = np.load('state_backup.npz')
state_restored = data['state']
sqrt_cov_restored = data['sqrt_cov']

# Reinitialize filter
filter.filter.initialize(
    srukf.StateVector.from_numpy(state_restored),
    sqrt_cov_restored
)
```

**HDF5 Format (Future):**

```python
import h5py

# Save (future implementation)
with h5py.File('checkpoint.h5', 'w') as f:
    f.create_dataset('state', data=filter.filter.get_state().to_numpy())
    f.create_dataset('sqrt_cov', data=filter.filter.get_sqrt_cov())
    f.attrs['timestamp'] = str(datetime.utcnow())

# Load
with h5py.File('checkpoint.h5', 'r') as f:
    state = f['state'][:]
    sqrt_cov = f['sqrt_cov'][:]
```

### 7.3 Exporting Data

**CSV Export:**

```python
import pandas as pd

# Export vertical profile
profile_data = {
    'altitude_km': alt_grid,
    'ne_el_per_m3': ne_grid[i, j, :]
}
df = pd.DataFrame(profile_data)
df.to_csv('ne_profile.csv', index=False)
```

**NetCDF Export (Meteorological Standard):**

```python
from netCDF4 import Dataset

# Create NetCDF file
nc = Dataset('ionosphere.nc', 'w', format='NETCDF4')

# Dimensions
nc.createDimension('lat', n_lat)
nc.createDimension('lon', n_lon)
nc.createDimension('alt', n_alt)

# Variables
lats = nc.createVariable('latitude', 'f4', ('lat',))
lons = nc.createVariable('longitude', 'f4', ('lon',))
alts = nc.createVariable('altitude', 'f4', ('alt',))
ne = nc.createVariable('electron_density', 'f8', ('lat', 'lon', 'alt'))

# Data
lats[:] = lat_grid
lons[:] = lon_grid
alts[:] = alt_grid
ne[:,:,:] = ne_grid

# Metadata
ne.units = 'electrons per cubic meter'
ne.long_name = 'Electron Density'

nc.close()
```

### 7.4 Visualization

**Vertical Profile:**

```python
import matplotlib.pyplot as plt

# Plot vertical Ne profile at a location
i, j = 36, 36  # Center of grid
plt.figure(figsize=(8, 10))
plt.plot(ne_grid[i, j, :] / 1e11, alt_grid)
plt.xlabel('Electron Density (×10¹¹ el/m³)')
plt.ylabel('Altitude (km)')
plt.title(f'Ne Profile at ({lat_grid[i]:.1f}°N, {lon_grid[j]:.1f}°E)')
plt.grid(True)
plt.tight_layout()
plt.savefig('ne_profile.png', dpi=150)
```

**Horizontal Slice:**

```python
# Plot Ne at specific altitude
k = 20  # Altitude index
plt.figure(figsize=(12, 8))
plt.contourf(lon_grid, lat_grid, ne_grid[:, :, k] / 1e11, levels=20, cmap='viridis')
plt.colorbar(label='Electron Density (×10¹¹ el/m³)')
plt.xlabel('Longitude (°E)')
plt.ylabel('Latitude (°N)')
plt.title(f'Ne at {alt_grid[k]:.0f} km')
plt.tight_layout()
plt.savefig('ne_horizontal.png', dpi=150)
```

**Critical Frequency Map:**

```python
# Compute foF2 at each location
fof2_map = np.zeros((n_lat, n_lon))
for i in range(n_lat):
    for j in range(n_lon):
        ne_peak = ne_grid[i, j, :].max()
        fof2_map[i, j] = 9.0 * np.sqrt(ne_peak / 1e12)

plt.figure(figsize=(12, 8))
plt.contourf(lon_grid, lat_grid, fof2_map, levels=20, cmap='jet')
plt.colorbar(label='foF2 (MHz)')
plt.xlabel('Longitude (°E)')
plt.ylabel('Latitude (°N)')
plt.title('F2 Layer Critical Frequency')
plt.tight_layout()
plt.savefig('fof2_map.png', dpi=150)
```

---

## 8. Advanced Usage

### 8.1 Custom Observation Models

**Creating a Custom Observation Model:**

```python
import autonvis_srukf as srukf

class CustomObservationModel(srukf.ObservationModel):
    """Custom observation model (pybind11 subclass)"""

    def __init__(self):
        super().__init__()

    def obs_dimension(self):
        """Return observation dimension"""
        return 1

    def forward(self, state):
        """
        Forward observation model

        Args:
            state: StateVector

        Returns:
            Eigen::VectorXd of predicted observations
        """
        # Example: Observe peak Ne
        ne_peak = 0.0
        for i in range(state.n_lat):
            for j in range(state.n_lon):
                for k in range(state.n_alt):
                    ne = state.get_ne(i, j, k)
                    if ne > ne_peak:
                        ne_peak = ne

        return np.array([ne_peak])
```

**Note:** Most observation models require C++ implementation for performance.

### 8.2 Multi-Cycle Operation

**Running Multiple Cycles:**

```python
num_cycles = 10
dt = 900.0  # 15 minutes

results = []
for cycle in range(num_cycles):
    result = filter.run_cycle(dt=dt)
    results.append(result)

    print(f"Cycle {cycle+1}/{num_cycles}: "
          f"uncertainty={filter.get_uncertainty():.2e}, "
          f"inflation={result['inflation_factor']:.4f}")

# Analyze results
import pandas as pd
df = pd.DataFrame(results)
print(df[['cycle', 'smoother_active', 'inflation_factor', 'avg_nis']])
```

**Time-Varying Parameters:**

```python
from datetime import timedelta

start_time = datetime(2026, 3, 21, 0, 0, 0)

for cycle in range(96):  # 24 hours at 15-min intervals
    current_time = start_time + timedelta(minutes=15*cycle)

    # Update SSN based on time (example)
    hour = current_time.hour
    ssn_diurnal = 75.0 + 10.0 * np.sin(2 * np.pi * hour / 24)

    # Run cycle
    result = filter.run_cycle(dt=900.0)

    # Periodic output
    if cycle % 12 == 0:  # Every 3 hours
        print(f"{current_time}: SSN={ssn_diurnal:.1f}, "
              f"uncertainty={filter.get_uncertainty():.2e}")
```

### 8.3 Parallel Processing

**Multi-Grid Processing:**

```python
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

def run_filter_for_region(region_id, lat_range, lon_range):
    """Run filter for a specific geographic region"""

    lat_grid = np.linspace(*lat_range, 10)
    lon_grid = np.linspace(*lon_range, 10)
    alt_grid = np.linspace(100, 500, 9)

    filter = AutoNVISFilter(
        n_lat=len(lat_grid),
        n_lon=len(lon_grid),
        n_alt=len(alt_grid)
    )

    # Initialize and run
    # ... (initialization code) ...

    result = filter.run_cycle(dt=900.0)
    return region_id, result

# Define regions
regions = [
    (1, (20, 40), (-120, -90)),  # Region 1
    (2, (40, 60), (-120, -90)),  # Region 2
    (3, (20, 40), (-90, -60)),   # Region 3
    (4, (40, 60), (-90, -60)),   # Region 4
]

# Run in parallel
with ProcessPoolExecutor(max_workers=4) as executor:
    futures = [
        executor.submit(run_filter_for_region, rid, lat, lon)
        for rid, lat, lon in regions
    ]

    results = [f.result() for f in futures]

for region_id, result in results:
    print(f"Region {region_id}: {result['mode']}, "
          f"smoother={result['smoother_active']}")
```

### 8.4 Event Detection

**Detecting Ionospheric Anomalies:**

```python
class AnomalyDetector:
    def __init__(self, baseline_window=10):
        self.ne_history = []
        self.baseline_window = baseline_window

    def detect(self, ne_grid, threshold_sigma=3.0):
        """
        Detect anomalies in electron density

        Returns:
            dict with anomaly information
        """
        current_mean = ne_grid.mean()
        self.ne_history.append(current_mean)

        if len(self.ne_history) < self.baseline_window:
            return {'anomaly': False}

        # Compute baseline statistics
        baseline = np.array(self.ne_history[-self.baseline_window:])
        baseline_mean = baseline.mean()
        baseline_std = baseline.std()

        # Z-score
        z_score = (current_mean - baseline_mean) / baseline_std

        anomaly = abs(z_score) > threshold_sigma

        return {
            'anomaly': anomaly,
            'z_score': z_score,
            'current_mean': current_mean,
            'baseline_mean': baseline_mean,
            'deviation_percent': ((current_mean - baseline_mean) / baseline_mean) * 100
        }

# Usage
detector = AnomalyDetector()

for cycle in range(50):
    result = filter.run_cycle(dt=900.0)
    ne_grid = filter.get_state_grid()

    anomaly_info = detector.detect(ne_grid)

    if anomaly_info['anomaly']:
        print(f"Cycle {cycle}: ANOMALY DETECTED!")
        print(f"  Z-score: {anomaly_info['z_score']:.2f}")
        print(f"  Deviation: {anomaly_info['deviation_percent']:.1f}%")
```

---

## 9. Examples

### 9.1 Example 1: Simple Predict-Only Filter

```python
#!/usr/bin/env python3
"""
Example 1: Simple predict-only filter
Demonstrates basic filter operation without observations
"""

import sys
import numpy as np
from datetime import datetime

sys.path.insert(0, 'src/assimilation/python')
from autonvis_filter import AutoNVISFilter, OperationalMode
from src.assimilation.models.chapman_layer import ChapmanLayerModel

def main():
    # Small grid for quick testing
    lat_grid = np.linspace(30, 40, 5)
    lon_grid = np.linspace(-100, -90, 5)
    alt_grid = np.linspace(100, 500, 9)

    print(f"Grid: {len(lat_grid)}×{len(lon_grid)}×{len(alt_grid)} = "
          f"{len(lat_grid)*len(lon_grid)*len(alt_grid)} points")

    # Initialize with Chapman layer
    chapman = ChapmanLayerModel()
    time = datetime(2026, 3, 21, 18, 0, 0)
    ne_grid_3d = chapman.compute_3d_grid(lat_grid, lon_grid, alt_grid, time, ssn=75)

    state_dim = len(lat_grid) * len(lon_grid) * len(alt_grid) + 1
    initial_state = np.zeros(state_dim)
    initial_state[:-1] = ne_grid_3d.flatten()
    initial_state[-1] = 75.0

    initial_sqrt_cov = np.diag(0.1 * np.abs(initial_state))

    # Create filter
    filter = AutoNVISFilter(
        n_lat=len(lat_grid),
        n_lon=len(lon_grid),
        n_alt=len(alt_grid)
    )

    filter.initialize(lat_grid, lon_grid, alt_grid, initial_state, initial_sqrt_cov)
    filter.set_mode(OperationalMode.QUIET)

    # Run 10 predict cycles
    print("\nRunning 10 predict cycles...")
    for cycle in range(10):
        result = filter.run_cycle(dt=900.0)

        print(f"Cycle {cycle+1}: "
              f"uncertainty={filter.get_uncertainty():.2e}, "
              f"inflation={result['inflation_factor']:.4f}")

    # Get final state
    ne_grid_final = filter.get_state_grid()
    print(f"\nFinal Ne range: [{ne_grid_final.min():.2e}, {ne_grid_final.max():.2e}] el/m³")

    # Statistics
    stats = filter.get_statistics()
    print(f"\nStatistics:")
    print(f"  Total cycles: {stats['cycle_count']}")
    print(f"  Avg predict time: {stats['avg_predict_time_ms']:.2f} ms")
    print(f"  Divergences: {stats['divergence_count']}")

if __name__ == "__main__":
    main()
```

### 9.2 Example 2: Mode Switching Demonstration

```python
#!/usr/bin/env python3
"""
Example 2: Mode switching demonstration
Shows QUIET → SHOCK → QUIET transitions
"""

import sys
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, 'src/assimilation/python')
from autonvis_filter import AutoNVISFilter, OperationalMode
from src.assimilation.models.chapman_layer import ChapmanLayerModel

def simulate_solar_flare(cycle):
    """Simulate X-ray flux with solar flare event"""
    if 5 <= cycle < 15:  # Flare from cycle 5 to 15
        return 5e-5  # M5 class
    else:
        return 1e-6  # Quiet (B class)

def main():
    # Setup (same as Example 1)
    lat_grid = np.linspace(30, 40, 5)
    lon_grid = np.linspace(-100, -90, 5)
    alt_grid = np.linspace(100, 500, 9)

    chapman = ChapmanLayerModel()
    time = datetime(2026, 3, 21, 18, 0, 0)
    ne_grid_3d = chapman.compute_3d_grid(lat_grid, lon_grid, alt_grid, time, ssn=75)

    state_dim = len(lat_grid) * len(lon_grid) * len(alt_grid) + 1
    initial_state = np.zeros(state_dim)
    initial_state[:-1] = ne_grid_3d.flatten()
    initial_state[-1] = 75.0
    initial_sqrt_cov = np.diag(0.1 * np.abs(initial_state))

    filter = AutoNVISFilter(
        n_lat=len(lat_grid),
        n_lon=len(lon_grid),
        n_alt=len(alt_grid)
    )

    filter.initialize(lat_grid, lon_grid, alt_grid, initial_state, initial_sqrt_cov)
    filter.set_mode(OperationalMode.QUIET)

    # Run with mode switching
    print("Simulating solar flare event...")
    print("  Cycles 0-4: QUIET mode")
    print("  Cycles 5-14: SHOCK mode (M5 flare)")
    print("  Cycles 15+: QUIET mode (flare ends)")
    print()

    smoother_activations = {'QUIET': 0, 'SHOCK': 0}
    cycle_counts = {'QUIET': 0, 'SHOCK': 0}

    for cycle in range(20):
        # Get X-ray flux
        flux = simulate_solar_flare(cycle)

        # Switch mode
        old_mode = filter.current_mode
        if flux >= 1e-5:
            new_mode = OperationalMode.SHOCK
        else:
            new_mode = OperationalMode.QUIET

        if new_mode != old_mode:
            filter.set_mode(new_mode)
            print(f"Cycle {cycle}: MODE SWITCH {old_mode.value} → {new_mode.value}")

        # Run cycle
        result = filter.run_cycle(dt=900.0)

        # Track smoother
        mode_str = result['mode']
        cycle_counts[mode_str] += 1
        if result['smoother_active']:
            smoother_activations[mode_str] += 1

        print(f"Cycle {cycle}: mode={mode_str}, "
              f"smoother={result['smoother_active']}, "
              f"flux={flux:.2e} W/m²")

    # Summary
    print("\n" + "="*60)
    print("Summary:")
    for mode in ['QUIET', 'SHOCK']:
        total = cycle_counts[mode]
        activated = smoother_activations[mode]
        rate = (activated / total * 100) if total > 0 else 0
        print(f"  {mode} mode: {activated}/{total} smoother activations ({rate:.0f}%)")

    # Verify requirement
    if smoother_activations['SHOCK'] == 0:
        print("\n✓ REQUIREMENT VERIFIED: Smoother NEVER activated during SHOCK mode")
    else:
        print("\n✗ WARNING: Smoother activated during SHOCK mode!")

if __name__ == "__main__":
    main()
```

### 9.3 Example 3: Real-Time Visualization

```python
#!/usr/bin/env python3
"""
Example 3: Real-time visualization
Displays electron density evolution over time
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import datetime

sys.path.insert(0, 'src/assimilation/python')
from autonvis_filter import AutoNVISFilter, OperationalMode
from src.assimilation.models.chapman_layer import ChapmanLayerModel

# Global variables for animation
filter_instance = None
lat_grid_global = None
lon_grid_global = None
alt_grid_global = None
k_plot = 4  # Altitude index to plot

def init_filter():
    """Initialize filter (called once)"""
    global filter_instance, lat_grid_global, lon_grid_global, alt_grid_global

    lat_grid_global = np.linspace(30, 40, 7)
    lon_grid_global = np.linspace(-100, -90, 7)
    alt_grid_global = np.linspace(100, 500, 9)

    chapman = ChapmanLayerModel()
    time = datetime(2026, 3, 21, 18, 0, 0)
    ne_grid_3d = chapman.compute_3d_grid(
        lat_grid_global, lon_grid_global, alt_grid_global, time, ssn=75
    )

    state_dim = len(lat_grid_global) * len(lon_grid_global) * len(alt_grid_global) + 1
    initial_state = np.zeros(state_dim)
    initial_state[:-1] = ne_grid_3d.flatten()
    initial_state[-1] = 75.0
    initial_sqrt_cov = np.diag(0.1 * np.abs(initial_state))

    filter_instance = AutoNVISFilter(
        n_lat=len(lat_grid_global),
        n_lon=len(lon_grid_global),
        n_alt=len(alt_grid_global)
    )

    filter_instance.initialize(
        lat_grid_global, lon_grid_global, alt_grid_global,
        initial_state, initial_sqrt_cov
    )
    filter_instance.set_mode(OperationalMode.QUIET)

def animate(frame):
    """Animation update function"""
    global filter_instance, lat_grid_global, lon_grid_global, k_plot

    # Run filter cycle
    filter_instance.run_cycle(dt=900.0)

    # Get current state
    ne_grid = filter_instance.get_state_grid()

    # Clear and replot
    plt.clf()

    plt.contourf(
        lon_grid_global, lat_grid_global,
        ne_grid[:, :, k_plot] / 1e11,
        levels=15, cmap='viridis'
    )
    plt.colorbar(label='Ne (×10¹¹ el/m³)')
    plt.xlabel('Longitude (°E)')
    plt.ylabel('Latitude (°N)')
    plt.title(f'Electron Density at {alt_grid_global[k_plot]:.0f} km '
              f'(Cycle {frame+1})')
    plt.tight_layout()

def main():
    """Main visualization"""
    init_filter()

    fig = plt.figure(figsize=(10, 8))

    # Create animation (30 frames = 7.5 hours at 15-min intervals)
    anim = FuncAnimation(fig, animate, frames=30, interval=500, repeat=False)

    plt.show()

if __name__ == "__main__":
    main()
```

---

## 10. Best Practices

### 10.1 Filter Initialization

✅ **DO:**
- Use physically realistic background states (Chapman layer, IRI-2020)
- Initialize covariance to ~10% of state magnitude
- Verify initial state is within valid bounds (Ne: 1e8 - 1e13)

❌ **DON'T:**
- Initialize with zeros (physically unrealistic)
- Use overly small uncertainty (filter won't adapt)
- Use overly large uncertainty (filter may diverge)

### 10.2 Grid Design

✅ **DO:**
- Start with small grids for testing (5×5×9)
- Increase resolution incrementally
- Ensure altitude grid covers 60-600 km

❌ **DON'T:**
- Jump directly to full 73×73×55 grid without testing
- Use irregular grid spacing (makes interpolation difficult)
- Extend below 60 km (no ionosphere) or above 1000 km (negligible density)

### 10.3 Mode Switching

✅ **DO:**
- Use hysteresis to prevent oscillation
- Log all mode transitions
- Verify smoother deactivation in SHOCK mode

❌ **DON'T:**
- Switch modes based on single noisy measurement
- Forget to update filter mode when supervisor mode changes
- Allow smoother to run during solar flares

### 10.4 Performance

✅ **DO:**
- Use localization (essential for L > 10,000)
- Compile with optimizations (`-O3`)
- Run profiling before optimization

❌ **DON'T:**
- Disable localization for large grids (memory explosion)
- Run in Debug mode for production
- Optimize prematurely

### 10.5 Data Handling

✅ **DO:**
- Validate all inputs (bounds checking)
- Save checkpoints periodically
- Use compressed formats for large grids

❌ **DON'T:**
- Assume data is always valid
- Store full-resolution history (grows unbounded)
- Use text formats for large arrays

---

## 11. Performance Tuning

### 11.1 Reducing Memory Usage

**Localization Radius:**
```python
# Tighter radius = less memory
filter = AutoNVISFilter(..., localization_radius_km=300.0)
```

**Smaller Grids:**
```python
# Coarser resolution
lat_grid = np.linspace(-60, 60, 37)  # 37 instead of 73
lon_grid = np.linspace(-180, 180, 37)
alt_grid = np.linspace(60, 600, 28)  # 28 instead of 55
```

### 11.2 Reducing Computation Time

**Parallel Compilation:**
```bash
cmake --build build -j$(nproc)  # Use all cores
```

**Optimize Compiler Flags:**
```bash
cmake -B build -DCMAKE_CXX_FLAGS="-O3 -march=native -flto"
```

**Fewer Sigma Points (Advanced):**
```python
# Reduce alpha (tighter sigma points)
filter = AutoNVISFilter(..., alpha=1e-4)  # Fewer sigma points needed
```

### 11.3 Profiling

**Python Profiling:**
```bash
python3 -m cProfile -o profile.stats demo_standalone.py
python3 << 'EOF'
import pstats
p = pstats.Stats('profile.stats')
p.sort_stats('cumulative')
p.print_stats(20)
EOF
```

**C++ Profiling:**
```bash
perf record -g python3 demo_standalone.py
perf report
```

**Memory Profiling:**
```bash
/usr/bin/time -v python3 demo_standalone.py
```

---

## 12. Troubleshooting

### 12.1 Filter Divergence

**Symptoms:**
- Uncertainty grows unbounded
- NIS consistently >> expected value
- Ne values become unphysical (< 1e8 or > 1e13)

**Solutions:**
1. Enable/increase adaptive inflation
2. Increase process noise
3. Check observation errors (not too small)
4. Verify physics model validity

**Example Fix:**
```python
# Increase inflation bounds
inflation_cfg = srukf.AdaptiveInflationConfig()
inflation_cfg.max_inflation = 3.0  # Allow more inflation
filter.filter.set_adaptive_inflation_config(inflation_cfg)

# Or manually inflate
filter.apply_inflation(1.5)
```

### 12.2 Slow Performance

**Symptoms:**
- Cycles take > 10 seconds (small grid)
- Memory usage high

**Diagnosis:**
```python
result = filter.run_cycle(dt=900.0)
print(f"Predict time: {result['predict_time_ms']} ms")
print(f"Update time: {result['update_time_ms']} ms")

# Check grid size
print(f"State dimension: {len(lat_grid) * len(lon_grid) * len(alt_grid) + 1}")
```

**Solutions:**
1. Reduce grid size
2. Enable localization
3. Rebuild with optimizations
4. Use smaller alpha (fewer sigma points)

### 12.3 Import Errors

**Error:** `ModuleNotFoundError: No module named 'autonvis_srukf'`

**Solution:**
```python
import sys
import os

# Add module directory to path
module_path = os.path.join(os.getcwd(), 'src', 'assimilation', 'python')
sys.path.insert(0, module_path)

import autonvis_srukf
```

### 12.4 Numerical Issues

**Symptoms:**
- NaN values in state
- "Cholesky decomposition failed" warnings
- Negative electron densities

**Solutions:**
1. Enable regularization (already in code)
2. Increase process noise
3. Check for zero/negative initial covariance

**Example:**
```python
# Ensure positive initial covariance
initial_sqrt_cov = np.diag(np.abs(initial_state) * 0.1 + 1e6)
```

### 12.5 Mode Switching Issues

**Symptoms:**
- Smoother activates during SHOCK mode
- Mode doesn't change when expected

**Diagnosis:**
```python
print(f"Current mode: {filter.current_mode}")
print(f"Should use smoother: {filter.should_use_smoother()}")
print(f"Uncertainty: {filter.get_uncertainty()}")
print(f"Threshold: {filter.uncertainty_threshold}")
```

**Solution:**
```python
# Verify mode is set correctly
filter.set_mode(OperationalMode.SHOCK)
assert filter.current_mode == OperationalMode.SHOCK

# Verify smoother logic
if filter.current_mode == OperationalMode.SHOCK:
    assert not filter.should_use_smoother()
```

---

## 13. FAQ

**Q1: How often should I run filter cycles?**

A: Standard interval is 15 minutes (900 sec). Can be reduced to 5 min during active conditions, but increases computational load.

**Q2: Can I run multiple filters simultaneously?**

A: Yes. Each `AutoNVISFilter` instance is independent. Can run in separate threads/processes.

**Q3: How do I know if the filter is working correctly?**

A: Check:
- Divergence count = 0
- NIS ≈ observation dimension
- Ne values in valid range (1e8 - 1e13)
- Uncertainty decreases over time (with observations)

**Q4: What grid resolution should I use?**

A: Depends on application:
- Testing: 5×5×9 (fast)
- Regional: 16×26×28 (moderate)
- Global: 73×73×55 (slow, requires localization)

**Q5: How to integrate real observations?**

A: Implement observation models in C++ (see `observation_model.cpp` for examples), or use existing TEC/ionosonde models.

**Q6: Can I use GPU acceleration?**

A: Not yet implemented. Eigen3 supports GPU but requires CUDA build. Future enhancement.

**Q7: How to validate filter performance?**

A: Compare against:
- Ground truth ionosonde data
- Independent TEC measurements
- IRI-2020 climatology
- Cross-validation (withhold observations)

**Q8: What to do if filter crashes?**

A: Check:
- Memory limits (use `ulimit -a`)
- Grid size vs available RAM
- Log file for error messages
- Run in debugger (`gdb python3`)

---

**Usage Guide Complete**

For installation instructions, see [Installation.md](Installation.md).

For theoretical background, see [TheoreticalUnderpinnings.md](TheoreticalUnderpinnings.md).

For system integration details, see [system_integration_complete.md](system_integration_complete.md).
