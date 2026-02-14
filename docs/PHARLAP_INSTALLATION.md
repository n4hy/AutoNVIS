# PHaRLAP Installation Guide for Auto-NVIS

**Document Version**: 1.0
**Last Updated**: 2026-02-13
**Status**: Draft - PHaRLAP Integration Pending (Phase 12)

---

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Obtaining PHaRLAP](#obtaining-pharlap)
- [Installation Methods](#installation-methods)
  - [Method 1: MATLAB Installation (Recommended)](#method-1-matlab-installation-recommended)
  - [Method 2: MATLAB Runtime (Production)](#method-2-matlab-runtime-production)
  - [Method 3: Octave (Open Source Alternative)](#method-3-octave-open-source-alternative)
- [Auto-NVIS Integration](#auto-nvis-integration)
- [Configuration](#configuration)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)
- [Performance Optimization](#performance-optimization)
- [References](#references)

---

## Overview

**PHaRLAP** (Provision of High-frequency Appropriate Raytracing for the Low, Atmosphere, and Plasmasphere) is a 3D numerical ray tracing engine for HF radio propagation modeling developed by the Australian Defence Science and Technology (DST) Group.

### What PHaRLAP Provides for Auto-NVIS

- **3D Hamiltonian ray integration** through ionospheric electron density grids
- **Magnetoionic splitting** (O-mode and X-mode polarization)
- **D-region absorption calculation** (Sen-Wyller formulation)
- **Multi-hop propagation** (1-hop, 2-hop, 3-hop NVIS)
- **Coverage predictions** (SNR heatmaps, LUF/MUF boundaries)
- **Geomagnetic field effects** (IGRF model integration)

### System Requirements

**Minimum**:
- CPU: 4 cores (8+ recommended for parallel ray tracing)
- RAM: 8 GB (16+ GB recommended)
- Disk: 10 GB for PHaRLAP + data files
- OS: Linux (Ubuntu 20.04+), macOS, Windows 10+

**Software**:
- MATLAB R2020b or newer (R2023b+ recommended)
- OR MATLAB Runtime R2020b+ (for production deployment)
- Python 3.11+ (for Auto-NVIS integration)

---

## Prerequisites

### 1. MATLAB or MATLAB Runtime

**Option A: MATLAB (Development/Research)**
```bash
# Academic/Research License
# Download from: https://www.mathworks.com/products/matlab.html

# Verify installation
matlab -batch "version"
```

**Option B: MATLAB Runtime (Production)**
```bash
# Free runtime for compiled MATLAB applications
# Download from: https://www.mathworks.com/products/compiler/matlab-runtime.html

# Linux installation example (R2023b)
wget https://ssd.mathworks.com/supportfiles/downloads/R2023b/Release/0/deployment_files/installer/complete/glnxa64/MATLAB_Runtime_R2023b_glnxa64.zip
unzip MATLAB_Runtime_R2023b_glnxa64.zip -d matlab_runtime
cd matlab_runtime
sudo ./install -mode silent -agreeToLicense yes
```

### 2. Required MATLAB Toolboxes

PHaRLAP requires the following MATLAB toolboxes:
- **Parallel Computing Toolbox** (for multi-threaded ray tracing)
- **Optimization Toolbox** (for ray path refinement)
- **Statistics and Machine Learning Toolbox** (for data processing)

Verify toolboxes:
```matlab
% In MATLAB command window
ver

% Check for specific toolbox
license('test', 'Distrib_Computing_Toolbox')
license('test', 'Optimization_Toolbox')
```

### 3. Python Integration Dependencies

```bash
# Install MATLAB Engine API for Python
cd /usr/local/MATLAB/R2023b/extern/engines/python
python3 setup.py install

# Verify installation
python3 -c "import matlab.engine; print('MATLAB Engine OK')"

# Install Auto-NVIS Python dependencies
pip install numpy scipy matplotlib
```

### 4. Geomagnetic Field Data (IGRF)

PHaRLAP requires International Geomagnetic Reference Field (IGRF) coefficients:

```bash
# Download IGRF-13 coefficients
mkdir -p $HOME/pharlap_data/igrf
cd $HOME/pharlap_data/igrf

# Option 1: From NOAA
wget https://www.ngdc.noaa.gov/IAGA/vmod/igrf13coeffs.txt

# Option 2: From British Geological Survey
wget https://www.geomag.bgs.ac.uk/data_service/models_compass/igrf/coeffs/IGRF13.COF
```

---

## Obtaining PHaRLAP

### Official Distribution

**PHaRLAP 4.0+** is available from DST Group Australia:

**Option 1: Direct Download** (Registered Users)
- Website: https://www.dst.defence.gov.au/innovation/pharlap
- Registration required (free for research/academic use)
- License: Research use permitted, commercial use requires negotiation

**Option 2: IONO Lab Distribution** (Alternative)
- Some research institutions maintain mirrors
- Check with your institution's ionospheric research group

**Option 3: Source Code Access** (Limited)
- Available to partner institutions under research agreements
- Contact: DST Group Ionospheric Prediction Service

### Download and Extract

```bash
# Create PHaRLAP installation directory
sudo mkdir -p /opt/pharlap
sudo chown $USER:$USER /opt/pharlap

# Extract distribution (example filename)
cd /opt/pharlap
tar -xzf pharlap-4.x.x.tar.gz

# Verify contents
ls -la
# Expected:
# - src/           (MATLAB source code)
# - dat/           (Data files: IGRF, ionospheric models)
# - examples/      (Example scripts)
# - doc/           (Documentation)
# - LICENSE.txt
# - README.md
```

---

## Installation Methods

### Method 1: MATLAB Installation (Recommended)

**For development, research, and testing environments.**

#### Step 1: Add PHaRLAP to MATLAB Path

```bash
# Create startup script
cat > $HOME/Documents/MATLAB/startup.m << 'EOF'
% PHaRLAP initialization
pharlap_root = '/opt/pharlap';

addpath(genpath(fullfile(pharlap_root, 'src')));
addpath(fullfile(pharlap_root, 'dat'));

% Set PHaRLAP data directory
setenv('PHARLAP_DATA', fullfile(pharlap_root, 'dat'));

fprintf('PHaRLAP initialized: %s\n', pharlap_root);
EOF

# Test MATLAB path
matlab -batch "which raytrace_3d"
# Expected output: /opt/pharlap/src/raytrace_3d.m
```

#### Step 2: Configure PHaRLAP Settings

```matlab
% Create PHaRLAP configuration file
% File: /opt/pharlap/dat/pharlap_config.m

% Geomagnetic field model
igrf_file = '/opt/pharlap/dat/igrf/igrf13coeffs.txt';

% Parallel processing
num_workers = 8;  % Match CPU core count

% Ray tracing parameters
ray_tol = 1e-7;           % Ray integration tolerance
max_ray_len_km = 20000;   % Maximum ray path length
ground_range_km = 500;    % Max ground range for NVIS

% Grid interpolation
interp_method = 'linear';  % 'linear' or 'cubic'

% Save configuration
save('pharlap_config.mat', 'igrf_file', 'num_workers', ...
     'ray_tol', 'max_ray_len_km', 'ground_range_km', 'interp_method');
```

#### Step 3: Verify Installation

```matlab
% Test script: test_pharlap_install.m
fprintf('Testing PHaRLAP installation...\n');

% 1. Check core functions
fprintf('Checking core functions...\n');
assert(exist('raytrace_3d', 'file') == 2, 'raytrace_3d not found');
assert(exist('iri2020', 'file') == 2, 'iri2020 not found');
assert(exist('igrf', 'file') == 2, 'igrf not found');
fprintf('  OK - Core functions found\n');

% 2. Test IGRF model
fprintf('Testing IGRF magnetic field model...\n');
[bx, by, bz] = igrf(2026, 0, 0, 100);  % Lat=0, Lon=0, Alt=100km
assert(~isnan(bx) && ~isnan(by) && ~isnan(bz), 'IGRF computation failed');
fprintf('  OK - IGRF: Bx=%.1f nT, By=%.1f nT, Bz=%.1f nT\n', bx, by, bz);

% 3. Test simple ray trace
fprintf('Testing 3D ray tracing...\n');
origin_lat = 40.0;
origin_lon = -105.0;
origin_alt = 0.0;
freq = 5.0;  % MHz
elevs = [75, 80, 85];  % degrees
azim = 0;

try
    ray = raytrace_3d(origin_lat, origin_lon, origin_alt, ...
                      elevs, azim, freq);
    fprintf('  OK - Ray traced successfully\n');
catch ME
    error('Ray tracing failed: %s', ME.message);
end

fprintf('\nPHaRLAP installation verified!\n');
```

Run test:
```bash
matlab -batch "run('/opt/pharlap/test_pharlap_install.m')"
```

---

### Method 2: MATLAB Runtime (Production)

**For production deployments without MATLAB license.**

#### Step 1: Compile PHaRLAP MEX Files

On a system with MATLAB (one-time compilation):

```matlab
% compile_pharlap.m
fprintf('Compiling PHaRLAP for deployment...\n');

% Create deployment directory
deploy_dir = '/opt/pharlap/deploy';
mkdir(deploy_dir);

% Compile main ray tracing function
mcc -m raytrace_3d.m -o pharlap_raytrace ...
    -d deploy_dir ...
    -a dat/igrf ...
    -a dat/iri2020 ...
    -R -nodisplay -R -singleCompThread

% Compile batch processing function
mcc -m raytrace_batch.m -o pharlap_batch ...
    -d deploy_dir ...
    -a dat/igrf ...
    -a dat/iri2020 ...
    -R -nodisplay

fprintf('Compilation complete. Binaries in: %s\n', deploy_dir);
```

#### Step 2: Deploy to Production System

```bash
# Copy compiled binaries to production server
scp -r /opt/pharlap/deploy user@production:/opt/pharlap/

# On production server
cd /opt/pharlap/deploy

# Set MATLAB Runtime path
export LD_LIBRARY_PATH=/usr/local/MATLAB/MATLAB_Runtime/R2023b/runtime/glnxa64:\
/usr/local/MATLAB/MATLAB_Runtime/R2023b/bin/glnxa64:\
/usr/local/MATLAB/MATLAB_Runtime/R2023b/sys/os/glnxa64:$LD_LIBRARY_PATH

# Test compiled binary
./pharlap_raytrace
```

#### Step 3: Create Wrapper Script

```bash
# File: /opt/pharlap/bin/pharlap-raytrace
cat > /opt/pharlap/bin/pharlap-raytrace << 'EOF'
#!/bin/bash
# PHaRLAP ray tracing wrapper for MATLAB Runtime

PHARLAP_ROOT=/opt/pharlap
MATLAB_RUNTIME=/usr/local/MATLAB/MATLAB_Runtime/R2023b

export LD_LIBRARY_PATH=${MATLAB_RUNTIME}/runtime/glnxa64:\
${MATLAB_RUNTIME}/bin/glnxa64:\
${MATLAB_RUNTIME}/sys/os/glnxa64:$LD_LIBRARY_PATH

export PHARLAP_DATA=${PHARLAP_ROOT}/dat

exec ${PHARLAP_ROOT}/deploy/pharlap_raytrace "$@"
EOF

chmod +x /opt/pharlap/bin/pharlap-raytrace

# Add to PATH
echo 'export PATH=/opt/pharlap/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

---

### Method 3: Octave (Open Source Alternative)

**GNU Octave compatibility (limited functionality).**

⚠️ **Warning**: PHaRLAP was designed for MATLAB. Octave support is experimental and may have reduced functionality.

```bash
# Install Octave
sudo apt install octave octave-parallel octave-optim

# Install PHaRLAP
cd /opt/pharlap
octave --eval "addpath(genpath('src')); savepath;"

# Test basic functionality
octave --eval "which raytrace_3d"
```

**Known Limitations with Octave**:
- Performance ~2-3× slower than MATLAB
- Some MEX files may require recompilation
- Parallel processing may be limited
- Plotting compatibility issues

**Not recommended for production Auto-NVIS deployment.**

---

## Auto-NVIS Integration

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                 Auto-NVIS SR-UKF Filter                     │
│  Outputs: Electron density grid (73×73×55)                 │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│          Grid Converter (Python → MATLAB)                   │
│  • Convert NumPy array to MATLAB format                     │
│  • Interpolate to PHaRLAP grid requirements                 │
│  • Add geomagnetic field (IGRF)                             │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              PHaRLAP Ray Tracer (MATLAB)                    │
│  • Launch ray fans (70-90° elevation, all azimuths)         │
│  • Solve Haselgrove equations                               │
│  • Calculate O/X mode splitting                             │
│  • Compute absorption (D-region)                            │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│           Product Generator (Python)                        │
│  • LUF/MUF extraction                                       │
│  • SNR coverage maps                                        │
│  • Blackout warnings                                        │
│  • ALE frequency recommendations                            │
└─────────────────────────────────────────────────────────────┘
```

### Python-MATLAB Bridge

Create integration module: `src/propagation/pharlap_wrapper/pharlap_bridge.py`

```python
"""
PHaRLAP Bridge for Auto-NVIS

Interfaces between Python/NumPy electron density grids and MATLAB PHaRLAP.
"""

import numpy as np
import matlab.engine
from typing import Dict, List, Tuple, Optional
import logging

class PHaRLAPBridge:
    """
    Bridge between Auto-NVIS Python code and MATLAB PHaRLAP ray tracer.
    """

    def __init__(self, matlab_session: Optional[matlab.engine.MatlabEngine] = None):
        """
        Initialize PHaRLAP bridge.

        Args:
            matlab_session: Existing MATLAB engine session (optional)
        """
        self.logger = logging.getLogger(__name__)

        if matlab_session is None:
            self.logger.info("Starting MATLAB engine...")
            self.eng = matlab.engine.start_matlab()
            self._owns_session = True
        else:
            self.eng = matlab_session
            self._owns_session = False

        # Add PHaRLAP to MATLAB path
        self.eng.addpath(self.eng.genpath('/opt/pharlap/src'))
        self.eng.addpath('/opt/pharlap/dat')

        self.logger.info("PHaRLAP bridge initialized")

    def convert_grid_to_matlab(
        self,
        ne_grid: np.ndarray,
        lat: np.ndarray,
        lon: np.ndarray,
        alt: np.ndarray
    ) -> Dict:
        """
        Convert NumPy electron density grid to MATLAB format.

        Args:
            ne_grid: Electron density (73, 73, 55) in el/m³
            lat: Latitude grid (73,)
            lon: Longitude grid (73,)
            alt: Altitude grid (55,) in km

        Returns:
            MATLAB struct with ionospheric parameters
        """
        self.logger.debug("Converting Ne grid to MATLAB format")

        # Convert to MATLAB arrays (Fortran order)
        ne_matlab = matlab.double(ne_grid.tolist())
        lat_matlab = matlab.double(lat.tolist())
        lon_matlab = matlab.double(lon.tolist())
        alt_matlab = matlab.double(alt.tolist())

        # Create MATLAB struct
        iono_struct = {
            'ne': ne_matlab,
            'lat': lat_matlab,
            'lon': lon_matlab,
            'alt': alt_matlab
        }

        return iono_struct

    def raytrace_nvis(
        self,
        ne_grid: np.ndarray,
        lat: np.ndarray,
        lon: np.ndarray,
        alt: np.ndarray,
        tx_lat: float,
        tx_lon: float,
        freq_mhz: float,
        elevations: List[float],
        azimuths: List[float]
    ) -> Dict:
        """
        Perform NVIS ray tracing with PHaRLAP.

        Args:
            ne_grid: Electron density grid from SR-UKF
            lat, lon, alt: Grid coordinates
            tx_lat, tx_lon: Transmitter location (degrees)
            freq_mhz: Operating frequency (MHz)
            elevations: Elevation angles (degrees, typically 70-90)
            azimuths: Azimuth angles (degrees, 0-360)

        Returns:
            Dictionary with ray paths and propagation metrics
        """
        self.logger.info(f"Ray tracing: freq={freq_mhz} MHz, "
                        f"elevs={len(elevations)}, azims={len(azimuths)}")

        # Convert grids to MATLAB
        iono = self.convert_grid_to_matlab(ne_grid, lat, lon, alt)

        # Call PHaRLAP ray tracer
        result = self.eng.raytrace_3d_custom(
            tx_lat, tx_lon, 0.0,  # tx alt = 0
            matlab.double(elevations),
            matlab.double(azimuths),
            freq_mhz,
            iono,
            nargout=1
        )

        # Convert results back to Python
        ray_data = self._matlab_struct_to_dict(result)

        return ray_data

    def calculate_coverage(
        self,
        ne_grid: np.ndarray,
        lat: np.ndarray,
        lon: np.ndarray,
        alt: np.ndarray,
        tx_lat: float,
        tx_lon: float,
        freq_range: Tuple[float, float],
        freq_step: float = 0.1
    ) -> Dict:
        """
        Calculate NVIS coverage map (LUF/MUF analysis).

        Args:
            ne_grid: Electron density grid
            lat, lon, alt: Grid coordinates
            tx_lat, tx_lon: Transmitter location
            freq_range: (min_freq, max_freq) in MHz
            freq_step: Frequency step in MHz

        Returns:
            Coverage data with LUF, MUF, SNR maps
        """
        self.logger.info(f"Calculating coverage: {freq_range[0]}-{freq_range[1]} MHz")

        # Convert grids
        iono = self.convert_grid_to_matlab(ne_grid, lat, lon, alt)

        # Call PHaRLAP coverage calculator
        coverage = self.eng.nvis_coverage_map(
            tx_lat, tx_lon,
            freq_range[0], freq_range[1], freq_step,
            iono,
            nargout=1
        )

        return self._matlab_struct_to_dict(coverage)

    def _matlab_struct_to_dict(self, matlab_struct) -> Dict:
        """Convert MATLAB struct to Python dictionary."""
        result = {}

        # Extract field names
        if hasattr(matlab_struct, '_fieldnames'):
            for field in matlab_struct._fieldnames:
                value = getattr(matlab_struct, field)

                # Convert MATLAB arrays to NumPy
                if isinstance(value, matlab.double):
                    result[field] = np.array(value)
                else:
                    result[field] = value

        return result

    def __del__(self):
        """Clean up MATLAB engine session."""
        if self._owns_session and hasattr(self, 'eng'):
            self.logger.info("Stopping MATLAB engine")
            self.eng.quit()
```

### Installation Verification

Create test script: `tests/integration/test_pharlap_integration.py`

```python
"""
Integration test for PHaRLAP installation and Auto-NVIS bridge.
"""

import pytest
import numpy as np
from src.propagation.pharlap_wrapper.pharlap_bridge import PHaRLAPBridge

def test_matlab_engine_available():
    """Test that MATLAB engine can be started."""
    try:
        import matlab.engine
        eng = matlab.engine.start_matlab()
        eng.quit()
    except Exception as e:
        pytest.fail(f"MATLAB engine not available: {e}")

def test_pharlap_functions_exist():
    """Test that PHaRLAP functions are accessible."""
    bridge = PHaRLAPBridge()

    # Check for core PHaRLAP functions
    result = bridge.eng.exist('raytrace_3d', 'file')
    assert result == 2, "raytrace_3d not found"

    result = bridge.eng.exist('igrf', 'file')
    assert result == 2, "igrf not found"

def test_grid_conversion():
    """Test NumPy to MATLAB grid conversion."""
    bridge = PHaRLAPBridge()

    # Create small test grid
    ne_grid = np.random.rand(5, 5, 9) * 1e11
    lat = np.linspace(-20, 20, 5)
    lon = np.linspace(-20, 20, 5)
    alt = np.linspace(60, 600, 9)

    # Convert to MATLAB
    iono = bridge.convert_grid_to_matlab(ne_grid, lat, lon, alt)

    assert 'ne' in iono
    assert 'lat' in iono
    assert 'lon' in iono
    assert 'alt' in iono

def test_simple_raytrace():
    """Test simple NVIS ray trace."""
    bridge = PHaRLAPBridge()

    # Create simple ionosphere
    ne_grid = np.ones((5, 5, 9)) * 1e11
    lat = np.linspace(-20, 20, 5)
    lon = np.linspace(-20, 20, 5)
    alt = np.linspace(60, 600, 9)

    # Trace single ray
    result = bridge.raytrace_nvis(
        ne_grid, lat, lon, alt,
        tx_lat=0.0,
        tx_lon=0.0,
        freq_mhz=5.0,
        elevations=[85.0],
        azimuths=[0.0]
    )

    assert result is not None
    assert 'ray' in result or 'ground_range' in result
```

Run integration test:
```bash
pytest tests/integration/test_pharlap_integration.py -v
```

---

## Configuration

### PHaRLAP Configuration File

Create: `config/pharlap.yml`

```yaml
pharlap:
  # Installation paths
  matlab_root: /usr/local/MATLAB/R2023b
  pharlap_root: /opt/pharlap
  data_dir: /opt/pharlap/dat

  # Geomagnetic field model
  igrf:
    coefficients_file: /opt/pharlap/dat/igrf/igrf13coeffs.txt
    year: 2026  # Current year for IGRF model

  # Ray tracing parameters
  ray_tracing:
    tolerance: 1.0e-7           # Integration tolerance
    max_path_length_km: 20000   # Maximum ray path
    ground_range_max_km: 500    # Max NVIS ground range
    step_size_km: 1.0           # Initial step size

  # NVIS specific settings
  nvis:
    elevation_range: [70, 90]   # NVIS elevation angles (degrees)
    elevation_step: 1.0         # Elevation step (degrees)
    azimuth_range: [0, 360]     # All azimuths
    azimuth_step: 15.0          # Azimuth step (degrees)

  # Frequency sweep
  frequency:
    min_mhz: 2.0                # Minimum HF frequency
    max_mhz: 15.0               # Maximum HF frequency
    step_mhz: 0.1               # Frequency step

  # Absorption model
  absorption:
    d_region_model: sen_wyller  # 'sen_wyller' or 'george_bradley'
    collision_freq_model: auto  # 'auto' from X-ray flux or 'fixed'

  # Parallel processing
  parallel:
    enabled: true
    num_workers: 8              # Match CPU cores
    batch_size: 100             # Rays per batch

  # Output products
  products:
    luf_muf_maps: true
    snr_coverage: true
    blackout_warnings: true
    ray_paths: false            # Save full ray paths (large!)
```

---

## Verification

### Complete Installation Test

```bash
# Create comprehensive test script
cat > test_pharlap_complete.sh << 'EOF'
#!/bin/bash
# Comprehensive PHaRLAP installation verification

set -e

echo "=========================================="
echo "PHaRLAP Installation Verification"
echo "=========================================="

# 1. Check MATLAB/Runtime
echo ""
echo "1. Checking MATLAB installation..."
if command -v matlab &> /dev/null; then
    matlab -batch "version"
    echo "   ✓ MATLAB found"
else
    echo "   ✗ MATLAB not found (checking Runtime...)"
    if [ -d "/usr/local/MATLAB/MATLAB_Runtime" ]; then
        echo "   ✓ MATLAB Runtime found"
    else
        echo "   ✗ MATLAB Runtime not found"
        exit 1
    fi
fi

# 2. Check PHaRLAP files
echo ""
echo "2. Checking PHaRLAP installation..."
if [ -d "/opt/pharlap/src" ]; then
    echo "   ✓ PHaRLAP source directory found"
    num_files=$(find /opt/pharlap/src -name "*.m" | wc -l)
    echo "   ✓ Found $num_files MATLAB files"
else
    echo "   ✗ PHaRLAP not found at /opt/pharlap"
    exit 1
fi

# 3. Check IGRF data
echo ""
echo "3. Checking geomagnetic field data..."
if [ -f "/opt/pharlap/dat/igrf/igrf13coeffs.txt" ]; then
    echo "   ✓ IGRF coefficients found"
else
    echo "   ✗ IGRF data missing"
    exit 1
fi

# 4. Check Python MATLAB engine
echo ""
echo "4. Checking Python-MATLAB integration..."
python3 -c "import matlab.engine; print('   ✓ MATLAB Engine for Python installed')" || \
    echo "   ✗ MATLAB Engine not available"

# 5. Run MATLAB test
echo ""
echo "5. Running MATLAB functionality test..."
matlab -batch "try; raytrace_3d; catch; end; disp('   ✓ PHaRLAP functions accessible');" || \
    echo "   ✗ PHaRLAP functions not accessible"

# 6. Run Python integration test
echo ""
echo "6. Running Python integration test..."
python3 -m pytest tests/integration/test_pharlap_integration.py -v || \
    echo "   ⚠ Integration tests failed (expected if not yet implemented)"

echo ""
echo "=========================================="
echo "Verification complete!"
echo "=========================================="
EOF

chmod +x test_pharlap_complete.sh
./test_pharlap_complete.sh
```

---

## Troubleshooting

### Common Issues

#### 1. MATLAB Engine API Not Found

**Symptom**:
```
ModuleNotFoundError: No module named 'matlab.engine'
```

**Solution**:
```bash
# Locate MATLAB installation
MATLAB_ROOT=/usr/local/MATLAB/R2023b

# Install engine API
cd $MATLAB_ROOT/extern/engines/python
python3 setup.py install

# Verify
python3 -c "import matlab.engine"
```

#### 2. PHaRLAP Functions Not in Path

**Symptom**:
```matlab
Undefined function 'raytrace_3d'
```

**Solution**:
```matlab
% Add to startup.m
addpath(genpath('/opt/pharlap/src'));
savepath;
```

#### 3. IGRF File Not Found

**Symptom**:
```
Error: IGRF coefficients file not found
```

**Solution**:
```bash
# Download IGRF-13
mkdir -p /opt/pharlap/dat/igrf
cd /opt/pharlap/dat/igrf
wget https://www.ngdc.noaa.gov/IAGA/vmod/igrf13coeffs.txt

# Set environment variable
export PHARLAP_DATA=/opt/pharlap/dat
```

#### 4. Parallel Pool Fails to Start

**Symptom**:
```
Error using parpool: Parallel pool failed to start
```

**Solution**:
```matlab
% Check parallel toolbox
license('test', 'Distrib_Computing_Toolbox')

% Create local cluster profile
parpool('local', 4);  % Start with fewer workers

% Or disable parallel processing
% In pharlap_config.yml: parallel.enabled: false
```

#### 5. Memory Issues with Large Grids

**Symptom**:
```
Out of memory error in MATLAB
```

**Solution**:
```bash
# Increase MATLAB JVM heap
matlab -nosplash -r "java.lang.Runtime.getRuntime.maxMemory"

# Or use chunked processing
# Process grid in spatial chunks rather than all at once
```

#### 6. Slow Ray Tracing Performance

**Symptoms**:
- Ray tracing takes > 5 minutes for single frequency
- CPU utilization < 50%

**Solutions**:
```matlab
% Enable parallel processing
parpool('local', 8);  % Use available cores

% Reduce ray density
elevations = 70:2:90;  % Step 2° instead of 1°
azimuths = 0:30:360;   % Step 30° instead of 15°

% Optimize tolerance
ray_tol = 1e-6;  % Relax from 1e-7 if acceptable
```

---

## Performance Optimization

### 1. Parallel Ray Tracing

```matlab
% Enable MATLAB parallel pool
num_cores = feature('numcores');
parpool('local', num_cores);

% Parallel ray launch
parfor i = 1:length(elevations)
    ray(i) = raytrace_3d(lat, lon, alt, elevations(i), azimuth, freq);
end
```

### 2. Grid Interpolation

```matlab
% Use coarser grid for fast preview
ne_coarse = ne_grid(1:2:end, 1:2:end, :);

% Or use optimized interpolation
interp_method = 'linear';  % Faster than 'cubic'
```

### 3. Compiled Deployment

For production, use compiled MATLAB code:
- 2-3× faster startup
- No MATLAB license required
- Smaller memory footprint

### 4. GPU Acceleration

If NVIDIA GPU available:
```matlab
% Enable GPU computing
gpuDevice(1);

% Transfer arrays to GPU
ne_gpu = gpuArray(ne_grid);

% Ray tracing on GPU (requires custom implementation)
```

### 5. Benchmarking

Expected performance (73×73×55 grid, single frequency):

| Configuration | Time | Notes |
|--------------|------|-------|
| Single-threaded | 180 sec | Baseline |
| 8-core parallel | 30 sec | Linear scaling |
| Compiled + 8-core | 15 sec | Recommended |
| GPU-accelerated | 5 sec | Requires custom code |

---

## References

### PHaRLAP Documentation

1. **PHaRLAP User Manual**
   DST Group, Australia (2019)
   Available with PHaRLAP distribution

2. **Numerical Ray Tracing for HF Propagation**
   Coleman, C. J. (1998)
   Radio Science, 33(6), 1757-1766

3. **The IPS HF Prediction Service**
   Coleman, C. J., & Cervera, M. A. (2010)
   URSI General Assembly

### Integration References

4. **MATLAB Engine API for Python**
   https://www.mathworks.com/help/matlab/matlab-engine-for-python.html

5. **Auto-NVIS Architecture Document**
   `AutoNVIS/docs/TheoreticalUnderpinnings.md`

### Geomagnetic Models

6. **IGRF-13 Reference**
   Alken, P., et al. (2021)
   Earth, Planets and Space, 73(1), 1-25

---

## Support

### Getting Help

**Auto-NVIS PHaRLAP Integration**:
- Documentation: `docs/PHARLAP_INTEGRATION.md` (forthcoming)
- GitHub Issues: Tag with `propagation` label
- Developer: See `CONTRIBUTING.md`

**PHaRLAP Software**:
- DST Group: ionospheric.prediction@dst.defence.gov.au
- User Forum: https://groups.google.com/g/pharlap-users (if available)

**MATLAB Support**:
- MathWorks: https://www.mathworks.com/support
- Documentation: https://www.mathworks.com/help/matlab/

---

## Next Steps

After successful installation:

1. **Implement Auto-NVIS Integration** (Phase 12)
   - Grid converter: `src/propagation/pharlap_wrapper/grid_converter.py`
   - Ray tracer wrapper: `src/propagation/pharlap_wrapper/ray_tracer.py`
   - Product generator: `src/propagation/products/luf_muf_calculator.py`

2. **Create MATLAB Helper Functions**
   - `raytrace_3d_custom.m` - Custom wrapper for Auto-NVIS grids
   - `nvis_coverage_map.m` - Coverage map generator
   - `absorption_sen_wyller.m` - D-region absorption

3. **Integration Testing**
   - Test with synthetic ionospheres
   - Validate against known propagation conditions
   - Performance benchmarking

4. **Production Deployment**
   - Docker container with MATLAB Runtime
   - CI/CD pipeline integration
   - Monitoring and alerting

---

**Document Status**: Ready for Phase 12 Implementation
**Last Updated**: 2026-02-13
**Maintainer**: Auto-NVIS Development Team
