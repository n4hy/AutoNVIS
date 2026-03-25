# Auto-NVIS Development Guide

This guide provides information for developers working on the Auto-NVIS system.

**Last Updated**: March 24, 2026 | **Version**: 0.4.4

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.11+
- C++17 compiler (for local development)
- CMake 3.20+
- Git
- pybind11 3.0.2+ (for NumPy 2.x compatibility)

### Initial Setup

1. **Clone the repository** (if not already done)
   ```bash
   git clone <repository-url>
   cd AutoNVIS
   ```

2. **Install Python dependencies** (for local development)
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   pip install pybind11>=3.0.2  # Required for NumPy 2.x
   ```

3. **Build C++ modules**
   ```bash
   # Ray tracer
   cd src/propagation
   cmake -B build && cmake --build build -j$(nproc)
   cmake --build build --target install

   # Assimilation (if needed)
   cd ../assimilation
   mkdir build && cd build
   cmake .. && make -j$(nproc)
   ```

4. **Start services with Docker Compose**
   ```bash
   cd docker
   docker-compose up -d
   ```

5. **Check service health**
   ```bash
   docker-compose ps
   ```

### Access Points

- **RabbitMQ Management UI**: http://localhost:15672 (guest/guest)
- **Supervisor API**: http://localhost:8000
- **Output Dashboard**: http://localhost:8080
- **Redis**: localhost:6379

## Project Structure

```
AutoNVIS/
├── src/
│   ├── common/              # Shared utilities
│   │   ├── config.py        # Configuration management
│   │   ├── constants.py     # Physical constants
│   │   ├── geodesy.py       # Coordinate transformations
│   │   ├── logging_config.py # Structured logging
│   │   └── message_queue.py # Message queue abstraction
│   │
│   ├── ingestion/           # Data ingestion services
│   │   ├── space_weather/   # GOES, ACE clients
│   │   ├── gnss_tec/        # GNSS-TEC processing
│   │   ├── ionosonde/       # GIRO ionosonde client
│   │   └── nvis/            # NVIS sounder integration
│   │
│   ├── assimilation/        # SR-UKF core (C++)
│   │   ├── include/         # Header files
│   │   ├── src/             # Implementation
│   │   ├── models/          # Physics models
│   │   └── CMakeLists.txt   # Build configuration
│   │
│   ├── propagation/         # Native ray tracer (C++)
│   │   ├── include/         # ray_tracer_3d.hpp
│   │   ├── src/             # ray_tracer_3d.cpp
│   │   ├── bindings/        # pybind11 Python bindings
│   │   ├── python/          # High-level Python API
│   │   ├── products/        # LUF/MUF calculators
│   │   └── services/        # PropagationService
│   │
│   ├── raytracer/           # IONORT-style ray tracing
│   │   ├── integrators/     # RK4, RK45, Adams-Bashforth
│   │   ├── homing/          # Winner triplet algorithm
│   │   └── visualizations/  # Ray path displays
│   │
│   ├── supervisor/          # Autonomous control logic
│   ├── channel_models/      # Vogler-Hoffmeyer HF channel
│   └── output/              # Output generation & dashboards
│
├── docker/                  # Docker configurations
├── config/                  # YAML configuration files
├── data/                    # Runtime data storage
├── tests/                   # Unit, integration, validation tests
└── docs/                    # Documentation
```

## Development Workflow

### Running Individual Services

**Ingestion Service (Space Weather Monitor)**
```bash
python -m src.ingestion.main
```

**Supervisor Service**
```bash
python -m src.supervisor.main
```

**Output Service**
```bash
uvicorn src.output.web_dashboard.app:app --reload
```

**PyQt Dashboards**
```bash
python -m src.visualization.pyqt.propagation.main
python -m src.visualization.pyqt.solarimaging.main
```

### Building C++ Modules

**Ray Tracer (propagation)**
```bash
cd src/propagation
cmake -B build
cmake --build build -j$(nproc)
cmake --build build --target install

# Verify
python3 -c "from src.propagation.python import raytracer; print('OK')"
```

**Assimilation Core**
```bash
cd src/assimilation
mkdir build && cd build
cmake ..
make -j$(nproc)
ctest  # Run unit tests
```

### Running Tests

**Python tests**
```bash
pytest tests/unit/
pytest tests/integration/
pytest tests/  # All tests (467 passed, 9 skipped)
```

**C++ tests**
```bash
cd src/assimilation/build
ctest --verbose
```

## Configuration

Configuration is managed through YAML files in the `config/` directory:

- `production.yml`: Production configuration
- `development.yml`: Development overrides (create as needed)

Override configuration via environment variable:
```bash
export AUTONVIS_CONFIG=/path/to/config.yml
```

## Message Queue Topics

Standard topics defined in `src/common/message_queue.py`:

**Space Weather**
- `wx.xray` - GOES X-ray flux
- `wx.solar_wind` - ACE solar wind data
- `wx.geomag` - Geomagnetic indices

**Observations**
- `obs.gnss_tec` - GNSS-TEC measurements
- `obs.ionosonde` - Ionosonde parameters
- `obs.nvis` - NVIS sounder observations

**Control**
- `ctrl.mode_change` - Mode switching events
- `ctrl.cycle_trigger` - Cycle trigger commands

**Output**
- `out.frequency_plan` - Generated frequency plans
- `out.coverage_map` - Coverage maps
- `out.luf_muf` - LUF/MUF products
- `out.alert` - System alerts

## Data Flow

1. **Ingestion** services fetch real-time data (GNSS-TEC, ionosonde, space weather)
2. **Publish** to message queue topics
3. **Supervisor** monitors space weather and triggers cycles
4. **SR-UKF** assimilates observations into state estimate
5. **Ray Tracer** computes propagation with updated ionosphere
6. **Output** generates frequency plans, coverage maps, dashboards

## Debugging

**View RabbitMQ messages**
```bash
wget http://localhost:15672/cli/rabbitmqadmin
chmod +x rabbitmqadmin
./rabbitmqadmin list queues
./rabbitmqadmin get queue=<queue_name> requeue=true
```

**View Redis state**
```bash
redis-cli
> KEYS *
> GET <key>
```

**View service logs**
```bash
docker-compose logs -f ingestion
docker-compose logs -f supervisor
docker-compose logs -f assimilation
```

---

## Implementation Status

### Completed Phases ✅

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | SR-UKF Core | ✅ Complete |
| 2 | Data Ingestion (GOES, ACE, GNSS-TEC) | ✅ Complete |
| 3 | Supervisor Logic | ✅ Complete |
| 4 | SR-UKF Algorithm & IRI-2020 | ✅ Complete |
| 5 | Mode Controller | ✅ Complete |
| 6 | GIRO Ionosonde Integration | ✅ Complete |
| 7 | RTS Smoother | ✅ Complete |
| 8 | HDF5 Persistence | ✅ Complete |
| 9 | Historical Validation | ✅ Complete |
| 10 | IONORT-Style Ray Tracing | ✅ Complete |
| 11 | Web Dashboard | ✅ Complete |
| 12 | Native Ray Tracer (C++) | ✅ Complete |
| 13 | LUF/MUF Products | ✅ Complete |
| 14 | Vogler-Hoffmeyer Channel Model | ✅ Complete |
| 15 | Solar Imaging Dashboard | ✅ Complete |
| 16 | NVIS Analytics Dashboard | ✅ Complete |

### Current Phase: Refinement & Optimization

**Test Status**: 467/476 passing (98%)

**Completed Optimizations** (Phase 18):
- [x] Phase 18.1: OpenMP parallelization for sigma point propagation
- [x] Phase 18.2: Sparse matrix optimization (efficient Cholesky updates, sparse localization)
- [x] Phase 18.3: Build optimization (compiler flags, LTO)
- [x] Phase 18.4: CUDA ray tracing (GPU acceleration with CPU fallback)

**Active TODOs**:
- ~~Fix D-region absorption model~~ ✅ Completed (ITU-R P.531 / Banks & Kockarts)
- ~~Implement full IGRF-13~~ ✅ Completed (spherical harmonics through degree 8)
- ~~Performance optimization~~ ✅ Completed (OpenMP, sparse matrices, CUDA)

---

## Phase 17: Propagation Product Delivery (REVISED)

> **Note**: Original Phase 17 planned MATLAB PHaRLAP integration. This is **OBSOLETE** - a complete native C++ ray tracer has been implemented that eliminates the need for MATLAB.

### Status: Refinement Needed

The native ray tracer is functionally complete but has items to address:

#### Completed ✅
- [x] 3D Haselgrove ray equations with RK45 integration
- [x] Appleton-Hartree magnetoionic theory (O/X mode)
- [x] LUF/MUF/FOT calculation
- [x] Coverage map generation
- [x] Python bindings (pybind11)
- [x] PropagationService integration
- [x] Blackout detection (LUF > MUF)
- [x] ALE frequency recommendations

#### Needs Work ⚠️
- [ ] **D-region absorption model** - Currently disabled (`calculate_absorption = False`)
  - Current model too simplistic
  - Needs seasonal/solar cycle variations
  - X-ray enhancement logic needs refinement

- [ ] **Full IGRF-13** - Currently uses dipole approximation
  - Framework exists in code
  - Need to implement spherical harmonic expansion
  - Parse IGRF-13 coefficient files
  - Impact: Minor for NVIS (dipole accurate within 5-10%)

- [ ] **Historical validation** - Compare against known propagation data
- [ ] **RabbitMQ integration** - Publish products to message queue

### Deliverables

1. Fix D-region absorption model (HIGH PRIORITY)
2. Validate ray tracer against known scenarios
3. Complete message queue integration
4. Update dashboard visualizations
5. Document propagation API

### Effort Estimate: 2-3 weeks

---

## Phase 18: Performance Optimization

### Overview

Current performance is already excellent (ray tracer 4,200× faster than target), but significant optimization headroom exists for large-scale operations.

### Current Performance Baseline

| Component | Operation | Current | Target | Status |
|-----------|-----------|---------|--------|--------|
| Ray Tracer | Single ray | 0.2 ms | <50 ms | ✅ 250× faster |
| Ray Tracer | NVIS (60 rays) | 7.1 ms | <30 sec | ✅ 4,200× faster |
| SR-UKF | Predict (27k sigma pts) | ~50-100 ms | <100 ms | ✅ On target |
| Localization | Covariance ops | ~5-10 sec | <500 ms | ⚠️ Needs work |

### Optimization Roadmap

#### Phase 18.1: OpenMP Parallelization (Week 1-2)
**Priority: HIGH | Effort: LOW | Gain: 4-8×**

Parallelize CPU-intensive loops in SR-UKF:

```cpp
// Sigma point propagation - currently sequential
#pragma omp parallel for schedule(dynamic)
for (size_t i = 0; i < sigma_points.size(); ++i) {
    propagated_points[i] = physics_model_->propagate(sigma_points[i], dt);
}
```

**Files to modify**:
- `src/assimilation/CMakeLists.txt` - Add OpenMP
- `src/assimilation/src/sr_ukf.cpp` - Parallelize loops (lines 269, 296, 320)
- `src/assimilation/src/cholesky_update.cpp` - Parallelize localization

**Deliverables**:
- [ ] OpenMP enabled in CMake
- [ ] Parallelized sigma point propagation
- [ ] Parallelized cross-covariance computation
- [ ] Parallelized localization matrix computation
- [ ] Benchmark: 4-8× speedup on multi-core systems

#### Phase 18.2: Sparse Matrix Optimization (Week 2-3)
**Priority: HIGH | Effort: MEDIUM | Gain: 10-50×**

Current issue: Full covariance materialization defeats sparse localization:
```cpp
// PROBLEM: Creates 1.46 GB matrix for 13,501 state elements
Eigen::MatrixXd P = S * S.transpose();  // O(L³)
```

**Solution**: Work directly with sqrt covariance, avoid materialization

**Files to modify**:
- `src/assimilation/src/sr_ukf.cpp` - Refactor covariance operations
- `src/assimilation/src/cholesky_update.cpp` - Add sparse Cholesky option

**Deliverables**:
- [ ] Avoid full covariance materialization
- [ ] Sparse Cholesky decomposition (Eigen CholmodSupport)
- [ ] Memory reduction: 1.46 GB → ~50 MB
- [ ] Benchmark: 10-50× speedup for covariance operations

#### Phase 18.3: Build Optimization (Week 1)
**Priority: MEDIUM | Effort: LOW | Gain: 2-5×**

Current ray tracer builds with `-O0 -g` (debug mode):
```cmake
# Current (suboptimal)
target_compile_options(raytracer PRIVATE -O0 -g)

# Optimized
target_compile_options(raytracer PRIVATE
    -O3 -march=native -ffast-math -DNDEBUG)
```

**Files to modify**:
- `src/propagation/CMakeLists.txt`
- `src/assimilation/CMakeLists.txt`

#### Phase 18.4: CUDA Ray Tracer (Week 4-6)
**Priority: MEDIUM | Effort: HIGH | Gain: 10-100×**

Ray tracing is embarrassingly parallel - ideal for GPU:

```cpp
// Current: Sequential
for (double azim : azimuths) {
    for (double elev : elevations) {
        paths.push_back(trace_ray(...));  // One at a time
    }
}

// CUDA: Parallel
// Launch 512 threads, each traces one ray
trace_ray_kernel<<<8, 64>>>(rays, grid, paths);
```

**New files**:
- `src/propagation/src/ray_tracer_3d.cu` - CUDA kernels
- `src/propagation/src/cuda_utils.hpp` - Memory management

**Deliverables**:
- [ ] CUDA kernels for ray tracing
- [ ] CMake CUDA configuration
- [ ] Fallback to CPU if CUDA unavailable
- [ ] Benchmark: 512 rays in <5 ms (vs 60 ms CPU)

#### Phase 18.5: CUDA Sigma Points (Week 6-8)
**Priority: LOW | Effort: HIGH | Gain: 10-20×**

Requires GPU physics model - defer unless needed.

### Phase 18 Summary

| Sub-phase | Status | Actual Gain | Notes |
|-----------|--------|-------------|-------|
| 18.1 OpenMP | ✅ Complete | 4-8× | Sigma point propagation parallelized |
| 18.2 Sparse | ✅ Complete | 10-50× | Efficient Givens/hyperbolic rotations, sparse localization |
| 18.3 Build | ✅ Complete | 2-5× | -O3, -march=native, -ffast-math, LTO |
| 18.4 CUDA Ray | ✅ Complete | 10-100× | GPU kernels with CPU fallback |
| 18.5 CUDA Sigma | Deferred | - | Low priority - OpenMP sufficient |

**Implementation details**:
- 18.2: Replaced O(n³) Cholesky recomputation with O(n²) Givens rotation updates
- 18.2: Added `extract_localized_covariance()` to avoid full O(n²) matrix materialization
- 18.4: CUDA kernel traces 512+ rays in parallel, auto-detects GPU availability

---

## Phase 19: Production Deployment

### Planned Components

- Container orchestration (Kubernetes)
- Monitoring and alerting (Prometheus/Grafana)
- Automated backups and recovery
- 24/7 operational deployment

### Target: Q4 2026

---

## Contributing

1. Create a feature branch from `main`
2. Implement changes with tests
3. Run linting: `black src/` and `flake8 src/`
4. Run tests: `pytest`
5. Create pull request

## Resources

- [README](README.md) - Project overview
- [AutoNVIS PDF](AutoNVIS.pdf) - System architecture and theory
- [NATIVE_RAYTRACER_SUMMARY.md](NATIVE_RAYTRACER_SUMMARY.md) - Ray tracer documentation
- [IMPLEMENTATION_COMPLETE.md](IMPLEMENTATION_COMPLETE.md) - Implementation status

## Support

For questions or issues, consult the documentation or create a GitHub issue.
