# AutoNVIS Project Instructions

## Pending Test Fixes (15 failing tests)

### GloTEC Client (2 failures) - Mock/fetch issues:
- `tests/unit/test_glotec_client.py::TestGloTECClient::test_fetch_index_success`
- `tests/unit/test_glotec_client.py::TestGloTECClient::test_fetch_geojson_success`

**Issue:** Tests return None instead of expected data. Likely mock setup issue.

### Propagation Service (13 failures) - API mismatches in test file:
Located in `tests/unit/test_propagation_service.py`

**Issues to fix:**
1. Tests call `trace_frequency` but method doesn't exist on PropagationService
2. Tests call `calculate_coverage` but actual method is `calculate_nvis_coverage`
3. Grid shape mismatches (tests use 11 altitude levels, service expects 12)

**Failing tests:**
- `TestRayTracerInitialization::test_init_with_realistic_grid`
- `TestRayTracerInitialization::test_init_with_extreme_ne_values`
- `TestLUFMUFCalculation::test_luf_muf_with_high_ne`
- `TestLUFMUFCalculation::test_luf_muf_with_low_ne`
- `TestFrequencySweep::test_single_frequency_trace`
- `TestCoverageCalculation::test_basic_coverage`
- `TestCoverageCalculation::test_coverage_many_rays`
- `TestNumericalStability::test_horizontal_propagation`
- `TestNumericalStability::test_vertical_propagation`
- `TestNumericalStability::test_very_high_frequency`
- `TestNumericalStability::test_very_low_frequency`
- `TestConcurrentOperations::test_concurrent_frequency_traces`
- `TestConcurrentOperations::test_concurrent_coverage_calculations`

**Fix approach:** Update test file to match current PropagationService API, or update service to match expected API.
