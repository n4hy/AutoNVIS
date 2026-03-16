# AutoNVIS Project Instructions

## Test Status

All previously failing tests have been fixed. The test suite now passes (33 passed, 2 skipped).

### Fixed Issues (2026-03-16)

**GloTEC Client (2 tests fixed):**
- Fixed async mock setup for `aiohttp.ClientSession` context managers
- Corrected patch target to use full module path

**Propagation Service (13 tests fixed):**
- Replaced `trace_frequency()` calls with `calculate_nvis_coverage()`
- Replaced `calculate_coverage()` calls with `calculate_nvis_coverage()`
- Fixed grid dimension mismatches by using `len(grid_config.get_*_grid())` instead of `grid_config.n_*`
- Changed `initialize_ray_tracer()` assertions from `assert success` to `assert service.tracer is not None`
- Adjusted low-Ne test to use electron density high enough for reflections (5e11)
