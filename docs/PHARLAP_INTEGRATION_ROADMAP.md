# PHaRLAP Integration Roadmap - Phase 12

**Status**: ⏸️ **Pending** - Scheduled for Q3 2026
**Dependencies**: Phases 1-8 Complete ✅
**Estimated Duration**: 6-8 weeks
**Priority**: High

---

## Executive Summary

Phase 12 will integrate the PHaRLAP 3D ray tracing engine with the Auto-NVIS SR-UKF filter to deliver operational NVIS propagation products. This is the final major component needed to complete the end-to-end autonomous NVIS frequency planning system.

**Goal**: Transform electron density grids from the SR-UKF filter into actionable frequency recommendations (LUF/MUF, coverage maps, blackout warnings) for NVIS operators.

---

## Current Status

### ✅ Complete (Dependencies)
- SR-UKF filter core (Phases 1-7)
- GNSS-TEC real-time ingestion (Phase 8)
- NVIS sounder data ingestion (NVIS Phases 1-6)
- Python-C++ integration layer
- Autonomous mode switching (QUIET/SHOCK)
- Real-time data assimilation pipeline

### ⏸️ Pending (This Phase)
- PHaRLAP installation and configuration
- Python-MATLAB bridge implementation
- Grid conversion utilities
- Ray tracing automation
- Product generation (LUF/MUF/SNR)
- Integration with supervisor orchestrator

### 📁 Existing Scaffolding
- Directory structure: `src/propagation/` with subdirectories
  - `pharlap_wrapper/` (empty)
  - `absorption/` (empty)
  - `products/` (empty)
- Stub function in system orchestrator: `trigger_propagation()` (src/supervisor/system_orchestrator.py:127)
- Documentation: `PHARLAP_INSTALLATION.md` (ready)

---

## Implementation Plan

### Week 1-2: PHaRLAP Installation & Verification

**Deliverables**:
1. PHaRLAP installed on development system
2. MATLAB Engine API for Python configured
3. Installation verification tests passing
4. IGRF geomagnetic field data loaded

**Tasks**:
- [ ] Obtain PHaRLAP license/distribution from DST Group
- [ ] Install MATLAB R2023b+ or MATLAB Runtime
- [ ] Install PHaRLAP to `/opt/pharlap`
- [ ] Download IGRF-13 coefficients
- [ ] Install MATLAB Engine API for Python
- [ ] Run installation verification script
- [ ] Document any installation issues

**Files to Create**:
- `scripts/install_pharlap.sh` - Automated installation script
- `tests/integration/test_pharlap_install.py` - Installation verification

**Acceptance Criteria**:
- ✅ `matlab.engine.start_matlab()` succeeds
- ✅ `raytrace_3d` function accessible from MATLAB
- ✅ IGRF model returns valid magnetic field vectors
- ✅ Simple ray trace completes without errors

---

### Week 3-4: Python-MATLAB Bridge Implementation

**Deliverables**:
1. PHaRLAPBridge Python class (full implementation)
2. Grid conversion utilities
3. NumPy ↔ MATLAB array conversion
4. Unit tests for bridge functionality

**Tasks**:
- [ ] Implement `PHaRLAPBridge` class (started in PHARLAP_INSTALLATION.md)
- [ ] Create grid interpolation functions
- [ ] Add coordinate system transformations
- [ ] Implement error handling and logging
- [ ] Write unit tests for all bridge methods
- [ ] Performance benchmark grid conversions

**Files to Create**:
- `src/propagation/pharlap_wrapper/pharlap_bridge.py` ✅ (partial in docs)
- `src/propagation/pharlap_wrapper/grid_converter.py`
- `src/propagation/pharlap_wrapper/coordinate_transforms.py`
- `tests/unit/test_pharlap_bridge.py`
- `tests/unit/test_grid_conversion.py`

**Key Methods**:
```python
class PHaRLAPBridge:
    def __init__(self, matlab_session=None)
    def convert_grid_to_matlab(ne_grid, lat, lon, alt) -> Dict
    def raytrace_nvis(ne_grid, tx_lat, tx_lon, freq, elevs, azims) -> Dict
    def calculate_coverage(ne_grid, tx_lat, tx_lon, freq_range) -> Dict
    def calculate_absorption(ne_grid, freq, xray_flux) -> np.ndarray
```

**Acceptance Criteria**:
- ✅ 73×73×55 grid converts to MATLAB in < 100 ms
- ✅ All NumPy arrays properly converted (Fortran order)
- ✅ Unit tests achieve 90%+ coverage
- ✅ No memory leaks in bridge operations

---

### Week 4-5: MATLAB Helper Functions

**Deliverables**:
1. Custom MATLAB functions for Auto-NVIS integration
2. NVIS-optimized ray tracing routines
3. Coverage map generators
4. Absorption calculators

**Tasks**:
- [ ] Write `raytrace_3d_custom.m` - Auto-NVIS grid wrapper
- [ ] Write `nvis_coverage_map.m` - Coverage calculator
- [ ] Write `absorption_sen_wyller.m` - D-region absorption
- [ ] Write `calculate_luf_muf.m` - Frequency limit extraction
- [ ] Add parallel processing support
- [ ] Optimize for 73×73×55 grids

**Files to Create**:
- `src/propagation/pharlap_wrapper/matlab/raytrace_3d_custom.m`
- `src/propagation/pharlap_wrapper/matlab/nvis_coverage_map.m`
- `src/propagation/pharlap_wrapper/matlab/absorption_sen_wyller.m`
- `src/propagation/pharlap_wrapper/matlab/calculate_luf_muf.m`
- `src/propagation/pharlap_wrapper/matlab/plot_ray_paths.m`

**MATLAB Function Signatures**:
```matlab
function ray_data = raytrace_3d_custom(tx_lat, tx_lon, elevs, azims, freq, iono_struct)
function coverage = nvis_coverage_map(tx_lat, tx_lon, freq_min, freq_max, freq_step, iono)
function absorption_db = absorption_sen_wyller(ray_path, ne_d_region, freq_mhz)
function [luf_mhz, muf_mhz] = calculate_luf_muf(coverage_data, snr_threshold_db)
```

**Acceptance Criteria**:
- ✅ Ray tracing completes in < 30 sec (8-core parallel)
- ✅ Coverage map generated in < 60 sec
- ✅ Results validated against known propagation conditions
- ✅ No MATLAB runtime errors

---

### Week 5-6: Product Generation

**Deliverables**:
1. LUF/MUF calculator
2. SNR coverage map generator
3. Blackout warning system
4. ALE frequency recommender

**Tasks**:
- [ ] Implement `LUFMUFCalculator` class
- [ ] Implement `CoverageMapGenerator` class
- [ ] Implement `BlackoutDetector` class
- [ ] Implement `ALEFrequencyPlanner` class
- [ ] Add product validation logic
- [ ] Create visualization utilities

**Files to Create**:
- `src/propagation/products/luf_muf_calculator.py`
- `src/propagation/products/coverage_map.py`
- `src/propagation/products/blackout_detector.py`
- `src/propagation/products/ale_planner.py`
- `src/propagation/products/visualizer.py`
- `tests/unit/test_propagation_products.py`

**Product Specifications**:

1. **LUF/MUF Map**:
   - Input: Ray tracing results + absorption
   - Output: Frequency limits for each ground point
   - Format: GeoJSON with frequency ranges
   - Update: Every 15 minutes

2. **SNR Coverage Map**:
   - Input: Ray paths + signal strength
   - Output: dB SNR heatmap (lat/lon grid)
   - Format: NumPy array + metadata
   - Visualization: Matplotlib/Leaflet overlay

3. **Blackout Warning**:
   - Condition: LUF > MUF (no usable frequencies)
   - Severity: Critical if LUF > MUF + 2 MHz
   - Output: Alert message to RabbitMQ `out.alert` topic
   - Notification: Email/SMS to operators

4. **ALE Frequency Plan**:
   - Input: LUF/MUF + time-of-day + link budget
   - Output: Ranked list of recommended frequencies
   - Format: JSON with confidence scores
   - Updates: Continuous (15-min cycles)

**Acceptance Criteria**:
- ✅ LUF/MUF calculated for 500 km range in < 5 sec
- ✅ Blackout detection accuracy > 95% (vs manual analysis)
- ✅ ALE plan contains 5-10 ranked frequencies
- ✅ All products published to message queue

---

### Week 6-7: System Integration

**Deliverables**:
1. Integration with filter orchestrator
2. Message queue publishing
3. End-to-end pipeline testing
4. Performance optimization

**Tasks**:
- [ ] Implement `trigger_propagation()` in system_orchestrator.py
- [ ] Connect PHaRLAP output to product generators
- [ ] Publish products to RabbitMQ topics
- [ ] Add error handling and retry logic
- [ ] Implement caching for repeated ray traces
- [ ] Add monitoring/metrics

**Files to Modify**:
- `src/supervisor/system_orchestrator.py` (implement stub at line 127)
- `src/supervisor/filter_orchestrator.py` (add propagation trigger)
- `src/common/message_queue.py` (add product topics)

**New Message Queue Topics**:
```python
# Propagation products
TOPIC_PROPAGATION_LUF_MUF = "propagation.luf_muf"
TOPIC_PROPAGATION_COVERAGE = "propagation.coverage"
TOPIC_PROPAGATION_ALE_PLAN = "propagation.ale_plan"

# Alerts
TOPIC_ALERT_BLACKOUT = "alert.blackout"
TOPIC_ALERT_PROPAGATION = "alert.propagation"
```

**Integration Flow**:
```
1. Filter Cycle Complete → Updated Ne Grid
2. system_orchestrator.trigger_propagation()
3. PHaRLAPBridge.raytrace_nvis() → Ray paths
4. ProductGenerators.calculate_all() → LUF/MUF/SNR/ALE
5. Publish to RabbitMQ topics
6. Dashboard updates (WebSocket)
7. Alerts generated (if blackout detected)
```

**Acceptance Criteria**:
- ✅ Full pipeline completes in < 90 sec
- ✅ Products appear in RabbitMQ queues
- ✅ Dashboard receives real-time updates
- ✅ No errors in 24-hour continuous operation

---

### Week 7-8: Testing & Validation

**Deliverables**:
1. Comprehensive integration tests
2. Historical validation results
3. Performance benchmarks
4. Documentation updates

**Tasks**:
- [ ] Create test suite with known ionospheric conditions
- [ ] Validate against historical propagation data
- [ ] Performance benchmark full pipeline
- [ ] Stress test with rapid Ne grid updates
- [ ] Document accuracy metrics
- [ ] Update user guide with propagation products

**Test Scenarios**:

1. **Quiet Ionosphere** (Chapman layer, daytime)
   - Expected: LUF ~2 MHz, MUF ~12 MHz
   - Validate: SNR > 20 dB for 3-10 MHz

2. **Disturbed Ionosphere** (Solar flare)
   - Expected: LUF ~8 MHz (high absorption), MUF ~15 MHz
   - Validate: Blackout warning triggered

3. **Nighttime Conditions**
   - Expected: LUF ~2 MHz, MUF ~6 MHz (lower F-layer)
   - Validate: Reduced frequency window

4. **Solar Maximum** (high sunspot number)
   - Expected: LUF ~3 MHz, MUF ~18 MHz
   - Validate: Wide operating window

**Files to Create**:
- `tests/validation/test_propagation_scenarios.py`
- `tests/validation/test_historical_storms.py`
- `tests/performance/benchmark_pharlap_pipeline.py`
- `docs/PHARLAP_VALIDATION_REPORT.md`

**Metrics to Collect**:
- LUF/MUF prediction accuracy (vs observed)
- Coverage map correlation (vs real-world reports)
- Ray tracing computational time
- Memory usage
- False positive/negative rates (blackout warnings)

**Acceptance Criteria**:
- ✅ LUF/MUF within ±1 MHz of observed (80% of time)
- ✅ Blackout detection: 95% accuracy
- ✅ Full pipeline: < 90 sec (15-min budget OK)
- ✅ All validation tests passing

---

## File Structure

After Phase 12 completion:

```
src/propagation/
├── pharlap_wrapper/
│   ├── __init__.py
│   ├── pharlap_bridge.py          # Python-MATLAB bridge (main)
│   ├── grid_converter.py          # Ne grid → MATLAB format
│   ├── coordinate_transforms.py   # Lat/lon/alt utilities
│   └── matlab/
│       ├── raytrace_3d_custom.m
│       ├── nvis_coverage_map.m
│       ├── absorption_sen_wyller.m
│       └── calculate_luf_muf.m
│
├── absorption/
│   ├── __init__.py
│   ├── d_region_model.py          # D-region absorption calculator
│   ├── collision_frequency.py     # From X-ray flux
│   └── sen_wyller.py              # Sen-Wyller formulation
│
├── products/
│   ├── __init__.py
│   ├── luf_muf_calculator.py      # LUF/MUF extraction
│   ├── coverage_map.py            # SNR heatmap generator
│   ├── blackout_detector.py       # LUF > MUF detection
│   ├── ale_planner.py             # ALE frequency recommender
│   └── visualizer.py              # Plotting utilities
│
└── config.py                      # Propagation configuration

tests/
├── unit/
│   ├── test_pharlap_bridge.py
│   ├── test_grid_conversion.py
│   ├── test_absorption_model.py
│   └── test_propagation_products.py
│
├── integration/
│   ├── test_pharlap_install.py
│   └── test_pharlap_pipeline.py
│
├── validation/
│   ├── test_propagation_scenarios.py
│   └── test_historical_storms.py
│
└── performance/
    └── benchmark_pharlap_pipeline.py

docs/
├── PHARLAP_INSTALLATION.md        ✅ (complete)
├── PHARLAP_INTEGRATION_ROADMAP.md ✅ (this file)
└── PHARLAP_VALIDATION_REPORT.md   ⏸️ (after testing)
```

**Estimated New Code**: ~3,500 lines
- Python: ~2,000 lines
- MATLAB: ~800 lines
- Tests: ~700 lines

---

## Dependencies & Prerequisites

### External Dependencies

1. **PHaRLAP Software**
   - Source: DST Group Australia
   - License: Research/academic (free), commercial (negotiated)
   - Status: ⏸️ Pending acquisition

2. **MATLAB or MATLAB Runtime**
   - Version: R2020b minimum, R2023b recommended
   - Toolboxes: Parallel Computing, Optimization, Statistics
   - License: Academic/commercial
   - Status: ⏸️ To be installed

3. **MATLAB Engine API for Python**
   - Included with MATLAB installation
   - Python 3.11+ compatible
   - Status: ⏸️ Pending MATLAB install

4. **IGRF Geomagnetic Field Data**
   - Version: IGRF-13 (current)
   - Source: NOAA/BGS (free download)
   - Size: ~100 KB
   - Status: ⏸️ To be downloaded

### Internal Dependencies

All complete ✅:
- SR-UKF filter producing electron density grids
- System orchestrator with 15-minute cycle management
- RabbitMQ message queue infrastructure
- Python 3.11+ environment
- NumPy, SciPy scientific stack

---

## Known Challenges & Mitigations

### Challenge 1: MATLAB License Cost

**Issue**: MATLAB licenses are expensive (~$2,500/year academic, ~$10,000 commercial)

**Mitigations**:
- Use MATLAB Runtime (free) for production deployment
- Compile MATLAB code to standalone executables
- Investigate GNU Octave compatibility (reduced functionality)
- Partner with academic institution for license sharing

### Challenge 2: Performance Bottleneck

**Issue**: Ray tracing is computationally intensive

**Mitigations**:
- Enable parallel processing (8+ cores)
- Use coarser ray grid for rapid updates
- Implement caching for repeated scenarios
- Consider GPU acceleration (future enhancement)

### Challenge 3: Python-MATLAB Bridge Overhead

**Issue**: Data transfer between Python and MATLAB can be slow

**Mitigations**:
- Use NumPy zero-copy array sharing
- Batch multiple ray traces in single MATLAB call
- Keep MATLAB session persistent (don't restart)
- Profile and optimize conversion code

### Challenge 4: PHaRLAP Distribution Access

**Issue**: PHaRLAP requires registration with DST Group

**Mitigations**:
- Begin registration process early
- Establish academic/research collaboration
- Consider alternative ray tracers (VOACAP) as backup
- Maintain good relationship with DST Group

### Challenge 5: IGRF Model Updates

**Issue**: IGRF coefficients updated every 5 years

**Mitigations**:
- Automate IGRF coefficient downloads
- Add version checking on startup
- Support multiple IGRF versions
- Document update procedure

---

## Success Criteria

Phase 12 will be considered **complete** when:

### Functional Requirements ✅
- [ ] PHaRLAP successfully installed and verified
- [ ] Python-MATLAB bridge converts grids in < 100 ms
- [ ] Ray tracing produces valid paths for NVIS elevations
- [ ] LUF/MUF calculated for operational area
- [ ] SNR coverage maps generated
- [ ] Blackout warnings trigger correctly
- [ ] ALE frequency plans published to message queue

### Performance Requirements ✅
- [ ] Full propagation pipeline completes in < 90 sec
- [ ] Ray tracing (single frequency): < 30 sec
- [ ] Coverage map generation: < 60 sec
- [ ] Memory usage: < 4 GB total
- [ ] CPU usage: 70-90% (8 cores, parallel)

### Quality Requirements ✅
- [ ] Unit test coverage: > 85%
- [ ] Integration tests: All passing
- [ ] LUF/MUF accuracy: ±1 MHz (vs historical data)
- [ ] Blackout detection: > 95% accuracy
- [ ] Zero crashes in 24-hour continuous operation

### Documentation Requirements ✅
- [ ] PHARLAP_INSTALLATION.md complete
- [ ] PHARLAP_INTEGRATION_ROADMAP.md (this document)
- [ ] PHARLAP_VALIDATION_REPORT.md (test results)
- [ ] User guide updated with propagation products
- [ ] API documentation for all new classes

---

## Timeline & Milestones

```
Week 1-2: Installation & Verification
├─ Milestone: PHaRLAP operational
└─ Deliverable: Installation tests passing

Week 3-4: Python-MATLAB Bridge
├─ Milestone: Grid conversion working
└─ Deliverable: Unit tests passing

Week 4-5: MATLAB Helper Functions
├─ Milestone: Ray tracing operational
└─ Deliverable: Coverage maps generated

Week 5-6: Product Generation
├─ Milestone: LUF/MUF calculated
└─ Deliverable: Products published to queue

Week 6-7: System Integration
├─ Milestone: End-to-end pipeline working
└─ Deliverable: Dashboard shows propagation

Week 7-8: Testing & Validation
├─ Milestone: All tests passing
└─ Deliverable: Validation report complete

Week 8: Phase 12 Complete! 🚀
```

**Start Date**: TBD (Q3 2026)
**Estimated Completion**: 8 weeks from start
**Dependencies**: PHaRLAP license acquisition

---

## Next Actions (Priority Order)

1. **Immediate** (Before Phase 12 kickoff):
   - [ ] Contact DST Group for PHaRLAP license/access
   - [ ] Procure MATLAB license (or identify Runtime deployment path)
   - [ ] Review PHaRLAP documentation
   - [ ] Assign development resources

2. **Week 1** (Installation):
   - [ ] Install MATLAB/Runtime on development workstation
   - [ ] Install PHaRLAP and verify basic functionality
   - [ ] Download IGRF-13 coefficients
   - [ ] Install MATLAB Engine API for Python
   - [ ] Run installation verification tests

3. **Week 2-3** (Development):
   - [ ] Begin Python-MATLAB bridge implementation
   - [ ] Create grid conversion utilities
   - [ ] Write MATLAB helper functions
   - [ ] Implement unit tests

4. **Week 4-8** (Integration & Testing):
   - [ ] Integrate with system orchestrator
   - [ ] Generate propagation products
   - [ ] Validate against historical data
   - [ ] Performance optimization
   - [ ] Documentation updates

---

## Resources & References

### PHaRLAP Documentation
- DST Group PHaRLAP website: https://www.dst.defence.gov.au/innovation/pharlap
- User Manual: Included with distribution
- Example scripts: `/opt/pharlap/examples/`

### Technical References
- Coleman, C. J. (1998). "Numerical ray tracing for HF propagation." Radio Science, 33(6), 1757-1766.
- Appleton-Hartree equation: Ionospheric dispersion relation
- Sen-Wyller formulation: D-region absorption model

### Auto-NVIS Documentation
- `docs/PHARLAP_INSTALLATION.md` - Installation guide
- `docs/TheoreticalUnderpinnings.md` - Physics background
- `README.md` - System overview

### MATLAB Resources
- MATLAB Engine API: https://www.mathworks.com/help/matlab/matlab-engine-for-python.html
- Parallel Computing Toolbox: https://www.mathworks.com/products/parallel-computing.html

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-02-13 | Auto-NVIS Team | Initial roadmap created |

---

**Status**: Ready for Phase 12 Implementation
**Target Start**: Q3 2026
**Priority**: High (final major component)
**Dependencies**: PHaRLAP license acquisition

🚀 **This completes the Auto-NVIS system architecture!**
