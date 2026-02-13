# Installation

**Auto-NVIS System Installation Guide**

**Document Version:** 1.0
**Last Updated:** February 12, 2026
**Tested Platforms:** Ubuntu 22.04 LTS, Ubuntu 24.04 LTS

---

## Table of Contents

1. [System Requirements](#1-system-requirements)
2. [Quick Start](#2-quick-start)
3. [Detailed Installation](#3-detailed-installation)
4. [Dependency Installation](#4-dependency-installation)
5. [Building C++ Components](#5-building-c-components)
6. [Python Environment Setup](#6-python-environment-setup)
7. [Verification](#7-verification)
8. [Troubleshooting](#8-troubleshooting)
9. [Optional Components](#9-optional-components)
10. [Development Setup](#10-development-setup)

---

## 1. System Requirements

### 1.1 Hardware Requirements

**Minimum Configuration:**
- CPU: 4 cores, 2.0 GHz
- RAM: 8 GB
- Disk: 10 GB free space
- Network: Internet connection (for data ingestion)

**Recommended Configuration:**
- CPU: 8+ cores, 3.0+ GHz (Intel Xeon, AMD EPYC)
- RAM: 32 GB (for full 73×73×55 grid)
- Disk: 100 GB SSD
- GPU: CUDA-capable GPU (optional, for future acceleration)

**Grid Size vs Memory:**

| Grid Size | State Dimension | Memory (with localization) |
|-----------|-----------------|----------------------------|
| 5×5×9 | 226 | ~10 MB |
| 10×10×10 | 1,001 | ~50 MB |
| 73×73×55 | 293,097 | ~2 GB |

### 1.2 Software Requirements

**Operating System:**
- Linux (Ubuntu 22.04+, Debian 11+, RHEL 8+)
- macOS 12+ (experimental, limited testing)
- Windows WSL2 (not recommended)

**Compilers:**
- GCC 11+ or Clang 14+
- Support for C++17 standard

**Build Tools:**
- CMake 3.20+
- Make or Ninja
- Git

**Python:**
- Python 3.11+ (3.12 recommended)
- pip 23+

---

## 2. Quick Start

For users on Ubuntu 22.04+ with sudo access:

```bash
# Clone repository
git clone https://github.com/yourusername/AutoNVIS.git
cd AutoNVIS

# Install system dependencies
sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    git \
    libeigen3-dev \
    python3-dev \
    python3-pip \
    python3-venv

# Create Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install numpy matplotlib scipy

# Build C++ components
cd src/assimilation/bindings
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Run verification
cd /path/to/AutoNVIS
python3 src/assimilation/python/test_basic_integration.py

# Run demonstration
python3 demo_standalone.py
```

If all tests pass, you're ready to use Auto-NVIS!

---

## 3. Detailed Installation

### 3.1 Clone Repository

```bash
# HTTPS (recommended for most users)
git clone https://github.com/yourusername/AutoNVIS.git

# SSH (if you have SSH keys configured)
git clone git@github.com:yourusername/AutoNVIS.git

# Navigate to directory
cd AutoNVIS
```

### 3.2 Verify Repository Structure

```bash
ls -la
```

Expected output:
```
drwxrwxr-x  demo_autonomous_system.py
drwxrwxr-x  demo_standalone.py
drwxrwxr-x  docs/
drwxrwxr-x  src/
-rw-rw-r--  README.md
```

---

## 4. Dependency Installation

### 4.1 Ubuntu/Debian

```bash
# Update package list
sudo apt update

# Install build essentials
sudo apt install -y \
    build-essential \
    cmake \
    git \
    pkg-config

# Install C++ dependencies
sudo apt install -y \
    libeigen3-dev \
    libpython3-dev

# Install Python
sudo apt install -y \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv

# Verify installations
gcc --version       # Should be 11.0+
cmake --version     # Should be 3.20+
python3 --version   # Should be 3.11+
```

**Expected Output:**
```
gcc (Ubuntu 13.2.0-4ubuntu3) 13.2.0
cmake version 3.28.3
Python 3.12.3
```

### 4.2 RHEL/CentOS/Fedora

```bash
# Enable EPEL repository (RHEL/CentOS)
sudo yum install -y epel-release

# Install build tools
sudo yum groupinstall -y "Development Tools"
sudo yum install -y cmake3 git

# Install dependencies
sudo yum install -y \
    eigen3-devel \
    python3-devel \
    python3-pip

# Create symlink for cmake (if cmake3 installed)
sudo ln -s /usr/bin/cmake3 /usr/bin/cmake
```

### 4.3 macOS (Homebrew)

```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install cmake eigen python@3.12 git

# Verify installations
cmake --version
python3 --version
```

### 4.4 Manual Eigen3 Installation

If Eigen3 is not available via package manager:

```bash
# Download Eigen3
cd /tmp
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
tar -xzf eigen-3.4.0.tar.gz
cd eigen-3.4.0

# Install (header-only library)
sudo mkdir -p /usr/local/include
sudo cp -r Eigen /usr/local/include/

# Verify
ls /usr/local/include/Eigen
```

### 4.5 pybind11 Installation

pybind11 will be automatically downloaded by CMake if not found. For manual installation:

```bash
# Via pip (recommended)
pip install pybind11

# Or via apt (Ubuntu)
sudo apt install -y pybind11-dev

# Or from source
git clone https://github.com/pybind/pybind11.git
cd pybind11
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
sudo make install
```

---

## 5. Building C++ Components

### 5.1 Configure Build

```bash
cd /path/to/AutoNVIS/src/assimilation/bindings

# Create build directory
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=17

# View configuration summary
cmake -B build -LAH
```

**Configuration Options:**

| Option | Default | Description |
|--------|---------|-------------|
| CMAKE_BUILD_TYPE | Release | Build type (Debug, Release) |
| CMAKE_CXX_STANDARD | 17 | C++ standard version |
| Python3_EXECUTABLE | auto | Path to Python interpreter |

### 5.2 Compile

```bash
# Build with all available cores
cmake --build build -j$(nproc)

# Or specify number of parallel jobs
cmake --build build -j8
```

**Expected Output:**
```
[ 50%] Building CXX object CMakeFiles/autonvis_srukf.dir/python_bindings.cpp.o
[ 62%] Building CXX object CMakeFiles/autonvis_srukf.dir/.../sr_ukf.cpp.o
[ 75%] Building CXX object CMakeFiles/autonvis_srukf.dir/.../state_vector.cpp.o
[100%] Linking CXX shared module .../autonvis_srukf.cpython-312-x86_64-linux-gnu.so
[100%] Built target autonvis_srukf
```

### 5.3 Verify Build

```bash
# Check output module
ls -lh ../python/autonvis_srukf*.so

# Test import
python3 -c "import sys; sys.path.insert(0, '../python'); import autonvis_srukf; print(autonvis_srukf.__version__)"
```

**Expected Output:**
```
-rwxrwxr-x 1 user user 2.3M Feb 12 10:30 autonvis_srukf.cpython-312-x86_64-linux-gnu.so
0.1.0
```

### 5.4 Install (Optional)

To install to Python site-packages:

```bash
cd build
sudo make install

# Or without sudo using --user
cmake --install . --prefix ~/.local
```

**Installed Location:**
- System: `/usr/local/lib/python3.x/site-packages/`
- User: `~/.local/lib/python3.x/site-packages/`

---

## 6. Python Environment Setup

### 6.1 Create Virtual Environment

```bash
cd /path/to/AutoNVIS

# Create virtual environment
python3 -m venv venv

# Activate (Linux/macOS)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

Your prompt should change to show `(venv)`.

### 6.2 Install Python Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install core dependencies
pip install numpy matplotlib scipy

# Install optional dependencies
pip install pytest pytest-asyncio h5py

# Verify installations
pip list
```

**Expected Packages:**
```
numpy          1.26.4
matplotlib     3.8.3
scipy          1.12.0
pytest         8.0.2
h5py           3.10.0
```

### 6.3 Requirements File (Alternative)

Create `requirements.txt`:

```bash
cat > requirements.txt << 'EOF'
numpy>=1.24.0
matplotlib>=3.7.0
scipy>=1.10.0
pytest>=7.4.0
pytest-asyncio>=0.21.0
h5py>=3.9.0
EOF

# Install from requirements file
pip install -r requirements.txt
```

---

## 7. Verification

### 7.1 Test C++ Module Import

```bash
cd /path/to/AutoNVIS

python3 << 'EOF'
import sys
sys.path.insert(0, 'src/assimilation/python')

import autonvis_srukf as srukf

print(f"✓ Module imported: {srukf.__name__}")
print(f"  Version: {srukf.__version__}")

# Test instantiation
state = srukf.StateVector(5, 5, 7)
print(f"✓ StateVector created: dimension = {state.dimension()}")

filter = srukf.SquareRootUKF(5, 5, 7)
print(f"✓ SR-UKF filter created")

print("\n✓ All basic tests passed!")
EOF
```

**Expected Output:**
```
✓ Module imported: autonvis_srukf
  Version: 0.1.0
✓ StateVector created: dimension = 176
✓ SR-UKF filter created

✓ All basic tests passed!
```

### 7.2 Run Integration Tests

```bash
# Basic integration test (no external dependencies)
python3 src/assimilation/python/test_basic_integration.py
```

**Expected Output:**
```
======================================================================
Auto-NVIS Python-C++ Basic Integration Test
======================================================================

Grid: 3×3×5 = 45 points
State dimension: 46 (Ne grid + R_eff)

...

✓ Basic Integration Test PASSED
```

### 7.3 Run Demonstration

```bash
python3 demo_standalone.py
```

**Expected Output:**
```
======================================================================
AUTO-NVIS AUTONOMOUS SYSTEM DEMONSTRATION
======================================================================

Scenario:
  - Start in QUIET mode (normal operations)
  - M5-class solar flare occurs at 30 minutes
  - System switches to SHOCK mode (smoother disabled)
  - Flare ends at 90 minutes
  - System returns to QUIET mode (smoother re-enabled)

...

✓ CRITICAL REQUIREMENT MET:
  Smoother NEVER activated during SHOCK mode
  (as specified: 'never use it when shock events are happening')

System ready for production deployment!
```

### 7.4 Run Unit Tests (C++)

```bash
cd src/assimilation/build

# Run C++ tests (if built)
./test_sigma_points
./test_cholesky
./test_sr_ukf
```

**Expected Output:**
```
Test: Sigma point count... PASSED
Test: Mean recovery... PASSED (error: 5.87e-11)
Test: Weights sum to one... PASSED

All tests passed!
```

---

## 8. Troubleshooting

### 8.1 CMake Cannot Find Eigen3

**Error:**
```
CMake Error: Could NOT find Eigen3
```

**Solution 1:** Install Eigen3 via package manager
```bash
# Ubuntu
sudo apt install libeigen3-dev

# macOS
brew install eigen
```

**Solution 2:** Specify Eigen3 path manually
```bash
cmake -B build -DEigen3_DIR=/usr/local/share/eigen3/cmake
```

**Solution 3:** Manual installation (see Section 4.4)

### 8.2 CMake Cannot Find Python3

**Error:**
```
CMake Error: Could NOT find Python3
```

**Solution:** Specify Python executable
```bash
cmake -B build -DPython3_EXECUTABLE=$(which python3)
```

Or specify exact path:
```bash
cmake -B build -DPython3_EXECUTABLE=/usr/bin/python3.12
```

### 8.3 pybind11 Not Found

**Error:**
```
CMake Warning: pybind11 not found, downloading...
```

This is **not an error**. CMake will automatically download pybind11 via FetchContent.

To use system pybind11:
```bash
sudo apt install pybind11-dev
cmake -B build -Dpybind11_DIR=/usr/share/cmake/pybind11
```

### 8.4 Module Import Error

**Error:**
```python
ImportError: No module named 'autonvis_srukf'
```

**Solution 1:** Check module location
```bash
ls src/assimilation/python/autonvis_srukf*.so
```

**Solution 2:** Add to Python path
```python
import sys
sys.path.insert(0, '/path/to/AutoNVIS/src/assimilation/python')
import autonvis_srukf
```

**Solution 3:** Install module
```bash
cd src/assimilation/bindings/build
sudo make install
```

### 8.5 Undefined Symbol Errors

**Error:**
```
ImportError: undefined symbol: _ZN6Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EE...
```

**Cause:** Compiler mismatch or ABI incompatibility

**Solution:** Rebuild with same compiler
```bash
rm -rf build
CC=gcc CXX=g++ cmake -B build
cmake --build build
```

### 8.6 Chapman Layer Import Error

**Error:**
```python
ModuleNotFoundError: No module named 'src.common.constants'
```

**Solution:** Ensure AutoNVIS root is in Python path
```bash
export PYTHONPATH=/path/to/AutoNVIS:$PYTHONPATH
python3 demo_standalone.py
```

Or run from AutoNVIS root directory:
```bash
cd /path/to/AutoNVIS
python3 demo_standalone.py
```

### 8.7 Compilation Warnings

**Warning:**
```
warning: unused variable 'n' [-Wunused-variable]
```

These are **harmless** and can be ignored. To suppress:
```bash
cmake -B build -DCMAKE_CXX_FLAGS="-Wno-unused-variable"
```

### 8.8 Low Memory Issues

**Error:**
```
c++: fatal error: Killed signal terminated program cc1plus
```

**Cause:** Insufficient memory during compilation

**Solution:** Reduce parallel jobs
```bash
cmake --build build -j2  # Use only 2 cores
```

Or disable optimization:
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
```

---

## 9. Optional Components

### 9.1 Message Queue (RabbitMQ)

Required for full supervisor integration:

```bash
# Ubuntu
sudo apt install rabbitmq-server

# Start service
sudo systemctl start rabbitmq-server
sudo systemctl enable rabbitmq-server

# Install Python client
pip install pika

# Verify
sudo rabbitmqctl status
```

### 9.2 HDF5 (Checkpoint Persistence)

For state checkpoint save/load:

```bash
# Ubuntu
sudo apt install libhdf5-dev

# Python bindings
pip install h5py

# Verify
python3 -c "import h5py; print(h5py.version.info)"
```

### 9.3 GPU Support (Future)

For CUDA acceleration (future enhancement):

```bash
# Install NVIDIA CUDA Toolkit
sudo apt install nvidia-cuda-toolkit

# Verify
nvcc --version
nvidia-smi
```

### 9.4 Documentation Build

To build documentation (requires Sphinx):

```bash
pip install sphinx sphinx-rtd-theme

cd docs
make html

# View documentation
firefox _build/html/index.html
```

---

## 10. Development Setup

### 10.1 Install Development Tools

```bash
# Linters and formatters
pip install black flake8 mypy

# C++ tools
sudo apt install clang-format cppcheck valgrind

# Testing frameworks
pip install pytest pytest-cov pytest-benchmark
```

### 10.2 Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Set up hooks (if .pre-commit-config.yaml exists)
pre-commit install

# Run manually
pre-commit run --all-files
```

### 10.3 Build with Debug Symbols

```bash
cmake -B build \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_FLAGS="-g -O0"

cmake --build build
```

### 10.4 Enable Verbose Output

```bash
# CMake
cmake -B build --debug-output

# Make
cmake --build build -- VERBOSE=1

# Python tests
pytest -v -s
```

### 10.5 Memory Leak Detection

```bash
# Compile with debug symbols
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build

# Run with valgrind
valgrind --leak-check=full \
         --show-leak-kinds=all \
         python3 demo_standalone.py
```

### 10.6 Profiling

```bash
# Install profiling tools
sudo apt install linux-tools-common linux-tools-generic

# Profile with perf
perf record -g python3 demo_standalone.py
perf report

# Python profiling
python3 -m cProfile -o profile.stats demo_standalone.py
python3 -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative'); p.print_stats(20)"
```

---

## 11. Uninstallation

### 11.1 Remove Built Components

```bash
# Remove build directory
rm -rf src/assimilation/bindings/build

# Remove Python module
rm -f src/assimilation/python/autonvis_srukf*.so

# Remove virtual environment
rm -rf venv
```

### 11.2 Uninstall System-Wide

```bash
# If installed to system
cd src/assimilation/bindings/build
sudo make uninstall

# Or remove manually
sudo rm /usr/local/lib/python3.*/site-packages/autonvis_srukf*.so
```

### 11.3 Clean Repository

```bash
# Remove all build artifacts
git clean -fdx

# Be careful: this removes ALL untracked files!
```

---

## 12. Frequently Asked Questions

### Q1: Which Python version should I use?

**A:** Python 3.11 or 3.12 is recommended. Python 3.10 may work but is not tested.

### Q2: Can I use Anaconda/Miniconda?

**A:** Yes, but create a dedicated environment:
```bash
conda create -n autonvis python=3.12
conda activate autonvis
conda install numpy matplotlib scipy
```

### Q3: Do I need root/sudo access?

**A:** No for building. Yes for system-wide installation. Use virtual environment and local build for non-root installation.

### Q4: How much disk space is required?

**A:**
- Source code: ~50 MB
- Build artifacts: ~100 MB
- Virtual environment: ~500 MB
- Total: ~650 MB

### Q5: Can I build on Windows?

**A:** WSL2 (Windows Subsystem for Linux) is supported. Native Windows build is not tested.

### Q6: How do I update Auto-NVIS?

**A:**
```bash
git pull
cd src/assimilation/bindings
cmake --build build  # Rebuild
```

### Q7: Build time is very long. How to speed up?

**A:**
- Use `-j$(nproc)` for parallel compilation
- Use `ccache` for caching
- Use precompiled headers (advanced)

### Q8: How to report installation issues?

**A:** Open an issue at https://github.com/yourusername/AutoNVIS/issues with:
- OS version (`uname -a`)
- Compiler version (`gcc --version`)
- CMake output (`cmake -B build 2>&1 | tee cmake.log`)
- Build error messages

---

## 13. Installation Checklist

Use this checklist to verify complete installation:

- [ ] Repository cloned
- [ ] System dependencies installed (CMake, Eigen3, Python3)
- [ ] C++ module built successfully
- [ ] Python module imports without error
- [ ] Basic integration test passes
- [ ] Demonstration runs successfully
- [ ] (Optional) Unit tests pass
- [ ] (Optional) Message queue configured
- [ ] (Optional) HDF5 support installed

**If all checked:** Installation complete! Proceed to [Usage.md](Usage.md).

---

## 14. Platform-Specific Notes

### 14.1 Ubuntu 22.04 LTS

Fully tested and supported. Use Quick Start guide.

### 14.2 Ubuntu 24.04 LTS

Fully tested and supported. Use Quick Start guide.

### 14.3 Debian 12 (Bookworm)

Works with minor changes:
```bash
sudo apt install libeigen3-dev python3-dev cmake build-essential
```

### 14.4 RHEL 8 / Rocky Linux 8

Enable PowerTools repository:
```bash
sudo dnf config-manager --set-enabled powertools
sudo dnf install eigen3-devel python3-devel cmake gcc-c++
```

### 14.5 macOS 13+ (Ventura)

Use Homebrew. May need to specify paths:
```bash
cmake -B build \
    -DEigen3_DIR=$(brew --prefix eigen)/share/eigen3/cmake \
    -DPython3_EXECUTABLE=$(which python3)
```

### 14.6 Raspberry Pi 4 (ARM64)

Supported but slow. Reduce grid size:
```python
# Use smaller grid for testing
lat_grid = np.linspace(30, 40, 3)
lon_grid = np.linspace(-100, -90, 3)
alt_grid = np.linspace(100, 400, 5)
```

---

**Installation Guide Complete**

For usage instructions, see [Usage.md](Usage.md).

For theoretical background, see [TheoreticalUnderpinnings.md](TheoreticalUnderpinnings.md).
