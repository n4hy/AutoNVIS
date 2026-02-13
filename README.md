# Auto-NVIS: Autonomous Ionospheric Nowcasting System

**Architecture for Autonomous Near Vertical Incidence Skywave (NVIS) Propagation Prediction (2025-2026)**

**Version:** 0.1.0 | **Status:** ✅ Production Ready (Filter Core) | **Last Updated:** February 12, 2026

## Overview

Auto-NVIS is an autonomous, unattended system designed to provide real-time HF propagation forecasting for Near Vertical Incidence Skywave (NVIS) communications during the volatile conditions of Solar Cycle 25 (2024-2026 solar maximum). The system integrates real-time sensor fusion, advanced nonlinear state estimation, and deterministic ray tracing into a closed-loop control system capable of operating 24/7 without human intervention.

## Mission Statement

Provide continuous, accurate, and physically-valid ionospheric state estimation and NVIS frequency planning that automatically adapts to extreme space weather events—including solar flares, geomagnetic storms, and sudden ionospheric disturbances—ensuring reliable HF communication links during critical conditions.

## Architectural Philosophy

**"Physics-Based, Data-Driven"**

The system does not rely solely on:
- Empirical statistics (which fail during anomalies)
- Raw observational data (which is often sparse and noisy)

Instead, Auto-NVIS uses a **Square-Root Unscented Kalman Filter (SR-UKF)** to assimilate real-time observations into physics-based background models (IRI-2020 or NeQuick-G), ensuring outputs that are both physically valid and observationally accurate.

## Quick Start

### Build & Test

```bash
# Build C++ pybind11 module
cd /home/n4hy/AutoNVIS/src/assimilation/bindings
cmake -B build && cmake --build build -j$(nproc)

# Run integration test
cd /home/n4hy/AutoNVIS
python3 src/assimilation/python/test_basic_integration.py

# Run autonomous demonstration
python3 demo_standalone.py
```

### Implementation Status

✅ **Complete (Phases 1-7):**
- SR-UKF Core (C++/Eigen) with adaptive inflation and localization
- Python-C++ integration (pybind11)
- Autonomous mode switching (QUIET ↔ SHOCK)
- **Conditional smoother logic** (mode-based + uncertainty-based)
- Chapman layer physics model
- System orchestration

⏸️ **Pending:**
- GNSS-TEC real-time ingestion
- Ionosonde data ingestion
- Offline smoother RTS backward pass

📚 **Documentation:** See `docs/system_integration_complete.md` for full details.

## System Architecture

### Three-Layer Design

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA INGESTION LAYER                      │
├─────────────────────────────────────────────────────────────┤
│  • GNSS-TEC Stream (RTCM/Ntrip IGS)                         │
│  • Ionosonde Data (GIRO/DIDBase auto-scaled)                │
│  • Space Weather (GOES X-Ray / ACE Solar Wind)              │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                   CORE PROCESSING LAYER                      │
├─────────────────────────────────────────────────────────────┤
│  • Assimilation Engine (SR-UKF State Estimation)            │
│    └─ Updates 4D Electron Density Grid                      │
│  • Propagation Physics (PHaRLAP 3D Ray Tracing)             │
│    └─ Multi-threaded Hamiltonian Solver                     │
│  • Supervisor Logic (Shock/Quiet Mode Switching)            │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                 OPERATIONAL OUTPUT LAYER                     │
├─────────────────────────────────────────────────────────────┤
│  • ALE Frequency Plan (Dynamic LUF/MUF Window)              │
│  • Coverage Maps (Signal-to-Noise Ratio Heatmaps)           │
│  • System Alerts (Fadeout/Storm Warnings)                   │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. SR-UKF Assimilation Engine

The "brain" of the autonomous system maintains a four-dimensional (latitude, longitude, altitude, time) model of ionospheric electron density.

**Why SR-UKF?**
- **Numerical Stability**: Guarantees positive semi-definite covariance matrices by propagating the Cholesky factor (S) directly
- **Nonlinear Capability**: Handles the highly nonlinear plasma physics without linearization errors
- **24/7 Reliability**: Prevents filter divergence during extreme ionospheric volatility

**Execution Cycle**: Every 15 minutes
1. Generate sigma points from current state and covariance
2. Propagate through nonlinear physics model (IRI-2020)
3. Update with incoming GNSS-TEC and ionosonde observations
4. Output 3D electron density grid

### 2. Autonomous Supervisor Module

Monitors space weather conditions and switches operational modes:

**Quiet Mode (Standard Operation)**
- Uses Gauss-Markov perturbation model
- Trusts climatological background
- Estimates small deviations from monthly median
- Computationally efficient for routine operations

**Shock Mode (Disturbed Operation)**
- Triggered by M1+ class solar flares (GOES X-ray flux)
- Abandons climatology assumptions
- Activates physics-based D-region absorption model
- Calculates collision frequencies directly from X-ray flux via Appleton-Hartree equation
- Prevents "model inertia" during rapid-onset blackouts

### 3. PHaRLAP Propagation Engine

Deterministic 3D ray-tracing engine for NVIS predictions:

**Workflow**:
1. Ingest electron density grid from SR-UKF
2. Launch ray fans (70° to 90° elevation, all azimuths)
3. Solve Haselgrove equations with geomagnetic field effects
4. Account for O-mode and X-mode polarization splitting
5. Generate SNR heatmaps and identify LUF/MUF boundaries

**Output Products**:
- **LUF (Lowest Usable Frequency)**: Based on D-region absorption
- **MUF (Maximum Usable Frequency)**: Based on F-layer penetration
- **Coverage Maps**: Signal strength predictions across operational area
- **Blackout Warnings**: When LUF exceeds MUF (NVIS "Dead Zone")

## Key Innovations

1. **Real-Time Absorption Modeling**: D-region electron density updated in real-time using relationship Ne(h) ∝ √(Flux_X-ray(t))

2. **Autonomous Mode Switching**: System automatically detects and responds to space weather events without human intervention

3. **Guaranteed Numerical Stability**: Square-root formulation prevents covariance matrix corruption during extreme conditions

4. **Physics-Constrained Data Assimilation**: Observations are blended with physical models, not blindly trusted

## Operational Capabilities

### Handles Critical Scenarios

- **Solar Flares**: Automatic detection via GOES X-ray monitoring
- **Sudden Ionospheric Disturbances (SID)**: Rapid D-region absorption calculation
- **Geomagnetic Storms**: F-layer density tracking via GNSS-TEC assimilation
- **Traveling Ionospheric Disturbances (TID)**: 4D state tracking captures spatial gradients

### Prevents Communication Failures

- **Proactive LUF Warnings**: Detects rising absorption before link closure
- **Dynamic Frequency Planning**: Continuously updates optimal operating frequencies
- **Blackout Prediction**: Warns when usable NVIS window collapses (LUF > MUF)

## Deployment Specifications

### Recommended Hardware

**Compute**:
- Multi-core workstation (AMD EPYC or equivalent)
- 32+ cores for parallel ray-tracing threads

**Memory**:
- 64GB+ ECC RAM for large covariance matrices

**GPU Acceleration**:
- CUDA-capable GPU for QR decomposition in SR-UKF update step

**Software Stack**:
- Dockerized containers:
  - Supervisor Module (Python)
  - Assimilation Core (C++/Eigen)
  - PHaRLAP (Fortran/MATLAB Runtime)

## Use Cases

### Emergency Communications

Maintain HF links during:
- Natural disasters when VHF/UHF infrastructure fails
- Solar storm events disrupting satellite communications
- Remote operations requiring NVIS propagation (< 400 km range)

### Military/Government Operations

- Autonomous frequency management for tactical HF networks
- Continuous situational awareness during solar activity
- Unattended operation in remote deployment scenarios

### Amateur Radio & Research

- Real-time propagation maps for NVIS operators
- Space weather impact studies
- Citizen science ionospheric monitoring

## Technical Approach

### Data Assimilation

- **State Vector**: Electron density at grid points + effective sunspot number + neutral wind vectors
- **Observation Models**: TEC (slant path integral) and foF2/hmF2 (ionosonde)
- **Update Method**: QR decomposition for efficient Cholesky factor updates

### Propagation Physics

- **Ray Equation**: Haselgrove formulation with Hamiltonian integration
- **Ionospheric Model**: Full 3D anisotropic (magnetoionic effects included)
- **Absorption Calculation**: Sen-Wyller generalized magnetoionic formulation

## Project Timeline

**Target Deployment**: 2025-2026 (Solar Cycle 25 Maximum)

**Development Phases**:
1. Assimilation core implementation
2. PHaRLAP integration and automation
3. Supervisor logic and mode switching
4. Data ingestion pipeline development
5. Validation against historical space weather events
6. Operational deployment and monitoring

## Related Work & References

This system builds upon established techniques in:
- Ionospheric data assimilation (USU Gauss-Markov Model, IDA3D)
- Nonlinear filtering (Wan & Van Der Merwe SR-UKF)
- HF propagation modeling (PHaRLAP, VOACAP)
- Space weather forecasting (NOAA SWPC, ESA SSA)

See AutoNVIS.pdf for complete reference list and theoretical background.

## License

TBD

## Contact

TBD

---

**Status**: Architecture and design phase
**Last Updated**: February 2026
