"""
Brutal Unit Tests for Propagation Service

Tests cover:
- Ray tracer initialization with extreme grids
- LUF/MUF calculation accuracy
- Multi-frequency sweeps (CPU intensive)
- Coverage calculations with many rays
- Integration with message queue
- Numerical stability and edge cases
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import time
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src' / 'propagation' / 'services'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from propagation_service import PropagationService
from common.config import GridConfig


class TestPropagationServiceInit:
    """Test propagation service initialization"""

    def test_init_with_default_config(self):
        """Test initialization with default configuration"""
        service = PropagationService(
            tx_lat=40.0,  # Boulder, CO
            tx_lon=-105.0
        )

        assert service is not None
        assert service.tx_lat == 40.0
        assert service.tx_lon == -105.0

    def test_init_with_custom_location(self):
        """Test initialization with custom transmitter location"""
        service = PropagationService(
            tx_lat=51.5,  # London
            tx_lon=0.0,
            freq_min=3.0,
            freq_max=20.0
        )

        assert service.tx_lat == 51.5
        assert service.tx_lon == 0.0
        assert service.freq_min == 3.0
        assert service.freq_max == 20.0

    def test_init_with_extreme_location(self):
        """Test initialization at extreme geographic locations"""
        # North Pole
        service = PropagationService(tx_lat=90.0, tx_lon=0.0)
        assert service.tx_lat == 90.0

        # South Pole
        service = PropagationService(tx_lat=-90.0, tx_lon=0.0)
        assert service.tx_lat == -90.0

        # Date line
        service = PropagationService(tx_lat=0.0, tx_lon=180.0)
        assert service.tx_lon == 180.0


class TestRayTracerInitialization:
    """Test ray tracer initialization with ionospheric grids"""

    def test_init_chapman_layer(self):
        """Test initialization with Chapman layer model"""
        from assimilation.models.chapman_layer import ChapmanLayerModel

        service = PropagationService(tx_lat=40.0, tx_lon=-105.0)

        # Generate Ne grid from Chapman layer physics model
        chapman = ChapmanLayerModel()
        lat_grid = np.linspace(-90, 90, 19)
        lon_grid = np.linspace(-180, 180, 37)
        alt_grid = np.arange(60, 600, 50)

        # Use high solar activity (SSN=150) to ensure sufficient Ne for reflections
        ne_grid = chapman.compute_3d_grid(
            lat_grid, lon_grid, alt_grid,
            time=datetime(2026, 6, 21, 18, 0, 0, tzinfo=timezone.utc),  # Summer solstice, daytime
            ssn=150.0  # High solar activity for strong ionization
        )

        # Validate grid was generated correctly
        assert ne_grid.shape == (len(lat_grid), len(lon_grid), len(alt_grid))
        assert np.all(ne_grid > 0)  # All positive electron densities
        assert np.max(ne_grid) > 1e11  # Sufficient Ne for HF reflection
        assert np.max(ne_grid) < 1e14  # Physically reasonable upper bound

        # Initialize ray tracer with Chapman-generated grid
        service.initialize_ray_tracer(
            ne_grid=ne_grid,
            lat_grid=lat_grid,
            lon_grid=lon_grid,
            alt_grid=alt_grid,
            xray_flux=1e-6
        )

        assert service.tracer is not None

        # Verify NVIS coverage can be calculated (simpler than LUF/MUF)
        result = service.calculate_nvis_coverage(freq_mhz=5.0)
        assert result is not None
        assert 'frequency_mhz' in result
        assert 'coverage_summary' in result

    def test_init_with_custom_grid(self):
        """Test initialization with custom electron density grid"""
        service = PropagationService(tx_lat=40.0, tx_lon=-105.0)
        grid_config = GridConfig(
            lat_min=-90.0, lat_max=90.0, lat_step=10.0,
            lon_min=-180.0, lon_max=180.0, lon_step=10.0,
            alt_min=100.0, alt_max=500.0, alt_step=50.0
        )

        # Create custom Ne grid
        lat_grid = grid_config.get_lat_grid()
        lon_grid = grid_config.get_lon_grid()
        alt_grid = grid_config.get_alt_grid()
        ne_grid = np.full((len(lat_grid), len(lon_grid), len(alt_grid)), 1e11)

        # initialize_ray_tracer returns None (not bool)
        service.initialize_ray_tracer(
            ne_grid=ne_grid,
            lat_grid=grid_config.get_lat_grid(),
            lon_grid=grid_config.get_lon_grid(),
            alt_grid=grid_config.get_alt_grid(),
            xray_flux=1e-6
        )

        # Verify tracer was initialized
        assert service.tracer is not None

    def test_init_with_realistic_grid(self):
        """Test initialization with realistic ionospheric profile"""
        service = PropagationService(tx_lat=40.0, tx_lon=-105.0)
        grid_config = GridConfig()

        # Create realistic F2 layer
        lat_grid = grid_config.get_lat_grid()
        lon_grid = grid_config.get_lon_grid()
        alt_grid = grid_config.get_alt_grid()

        ne_grid = np.zeros((len(lat_grid), len(lon_grid), len(alt_grid)))

        # Chapman layer profile
        for i_alt, alt in enumerate(alt_grid):
            # Peak at 300 km
            h = alt - 300.0
            H = 50.0  # Scale height
            ne_peak = 1e12
            ne_grid[:, :, i_alt] = ne_peak * np.exp(0.5 * (1 - h/H - np.exp(-h/H)))

        service.initialize_ray_tracer(
            ne_grid=ne_grid,
            lat_grid=lat_grid,
            lon_grid=lon_grid,
            alt_grid=alt_grid,
            xray_flux=1e-6
        )

        assert service.tracer is not None

    def test_init_with_extreme_ne_values(self):
        """Test initialization with extreme electron density values"""
        service = PropagationService(tx_lat=40.0, tx_lon=-105.0)
        grid_config = GridConfig(lat_step=10.0, lon_step=10.0, alt_step=50.0)

        # Get grids first to ensure consistent dimensions
        lat_grid = grid_config.get_lat_grid()
        lon_grid = grid_config.get_lon_grid()
        alt_grid = grid_config.get_alt_grid()

        # Very high Ne (solar maximum)
        ne_grid_high = np.full(
            (len(lat_grid), len(lon_grid), len(alt_grid)),
            1e13  # 10 trillion electrons/m³
        )

        service.initialize_ray_tracer(
            ne_grid=ne_grid_high,
            lat_grid=lat_grid,
            lon_grid=lon_grid,
            alt_grid=alt_grid,
            xray_flux=1e-5
        )

        assert service.tracer is not None

        # Very low Ne (nighttime/solar minimum)
        ne_grid_low = np.full(
            (len(lat_grid), len(lon_grid), len(alt_grid)),
            1e9  # 1 billion electrons/m³
        )

        service.initialize_ray_tracer(
            ne_grid=ne_grid_low,
            lat_grid=lat_grid,
            lon_grid=lon_grid,
            alt_grid=alt_grid,
            xray_flux=1e-7
        )

        assert service.tracer is not None


class TestLUFMUFCalculation:
    """Test LUF/MUF calculation"""

    def test_calculate_luf_muf_basic(self):
        """Test basic LUF/MUF calculation"""
        service = PropagationService(tx_lat=40.0, tx_lon=-105.0)

        # Initialize with Chapman layer
        grid_config = GridConfig()
        lat_grid = grid_config.get_lat_grid()
        lon_grid = grid_config.get_lon_grid()
        alt_grid = grid_config.get_alt_grid()
        ne_grid = np.full((len(lat_grid), len(lon_grid), len(alt_grid)), 1e11)
        
        service.initialize_ray_tracer(
            ne_grid=ne_grid,
            lat_grid=lat_grid,
            lon_grid=lon_grid,
            alt_grid=alt_grid,
            xray_flux=1e-6
        )

        # Calculate LUF/MUF
        result = service.calculate_luf_muf()

        assert result is not None
        assert 'luf_mhz' in result
        assert 'muf_mhz' in result
        assert 'fot_mhz' in result

        # Sanity checks
        assert 0.0 < result['luf_mhz'] < result['muf_mhz']
        assert result['luf_mhz'] < result['fot_mhz'] < result['muf_mhz']

    def test_luf_muf_with_high_ne(self):
        """Test LUF/MUF with high electron density (solar maximum)"""
        service = PropagationService(tx_lat=40.0, tx_lon=-105.0)
        grid_config = GridConfig(lat_step=10.0, lon_step=10.0, alt_step=50.0)

        # Get grids first to ensure consistent dimensions
        lat_grid = grid_config.get_lat_grid()
        lon_grid = grid_config.get_lon_grid()
        alt_grid = grid_config.get_alt_grid()

        # High Ne grid
        ne_grid = np.full(
            (len(lat_grid), len(lon_grid), len(alt_grid)),
            5e12
        )

        service.initialize_ray_tracer(
            ne_grid=ne_grid,
            lat_grid=lat_grid,
            lon_grid=lon_grid,
            alt_grid=alt_grid,
            xray_flux=1e-5
        )

        result = service.calculate_luf_muf()

        # High Ne should give high MUF
        assert result['muf_mhz'] > 10.0

    def test_luf_muf_with_low_ne(self):
        """Test LUF/MUF with low electron density (nighttime)"""
        service = PropagationService(tx_lat=40.0, tx_lon=-105.0)
        grid_config = GridConfig(lat_step=10.0, lon_step=10.0, alt_step=50.0)

        # Get grids first to ensure consistent dimensions
        lat_grid = grid_config.get_lat_grid()
        lon_grid = grid_config.get_lon_grid()
        alt_grid = grid_config.get_alt_grid()

        # Low but usable Ne grid (nighttime, but not blackout)
        # 5e11 still allows some reflection at low frequencies
        ne_grid = np.full(
            (len(lat_grid), len(lon_grid), len(alt_grid)),
            5e11
        )

        service.initialize_ray_tracer(
            ne_grid=ne_grid,
            lat_grid=lat_grid,
            lon_grid=lon_grid,
            alt_grid=alt_grid,
            xray_flux=1e-7
        )

        result = service.calculate_luf_muf()

        # Low Ne should give lower MUF than high Ne
        assert result['muf_mhz'] < 20.0


class TestFrequencySweep:
    """Test multi-frequency sweep operations"""

    def test_single_frequency_trace(self):
        """Test ray tracing at single frequency"""
        service = PropagationService(tx_lat=40.0, tx_lon=-105.0)
        grid_config = GridConfig()

        lat_grid = grid_config.get_lat_grid()
        lon_grid = grid_config.get_lon_grid()
        alt_grid = grid_config.get_alt_grid()
        ne_grid = np.full((len(lat_grid), len(lon_grid), len(alt_grid)), 1e11)
        
        service.initialize_ray_tracer(
            ne_grid=ne_grid,
            lat_grid=lat_grid,
            lon_grid=lon_grid,
            alt_grid=alt_grid,
            xray_flux=1e-6
        )

        # Trace single frequency using NVIS coverage
        result = service.calculate_nvis_coverage(freq_mhz=5.0)

        assert result is not None
        assert 'frequency_mhz' in result
        assert 'coverage_summary' in result
        assert result['coverage_summary']['total_rays'] > 0

    def test_multi_frequency_sweep(self):
        """Test sweeping across multiple frequencies"""
        service = PropagationService(
            tx_lat=40.0,
            tx_lon=-105.0,
            freq_min=2.0,
            freq_max=10.0,
            freq_step=2.0  # 5 frequencies
        )
        grid_config = GridConfig(lat_step=10.0, lon_step=10.0, alt_step=50.0)

        lat_grid = grid_config.get_lat_grid()
        lon_grid = grid_config.get_lon_grid()
        alt_grid = grid_config.get_alt_grid()
        ne_grid = np.full((len(lat_grid), len(lon_grid), len(alt_grid)), 1e11)
        
        service.initialize_ray_tracer(
            ne_grid=ne_grid,
            lat_grid=lat_grid,
            lon_grid=lon_grid,
            alt_grid=alt_grid,
            xray_flux=1e-6
        )

        # Calculate LUF/MUF (which does frequency sweep)
        result = service.calculate_luf_muf()

        assert result is not None

    def test_fine_frequency_sweep(self):
        """Test very fine frequency sweep (CPU intensive)"""
        service = PropagationService(
            tx_lat=40.0,
            tx_lon=-105.0,
            freq_min=2.0,
            freq_max=15.0,
            freq_step=0.1  # 131 frequencies!
        )
        grid_config = GridConfig(lat_step=20.0, lon_step=20.0, alt_step=100.0)

        lat_grid = grid_config.get_lat_grid()
        lon_grid = grid_config.get_lon_grid()
        alt_grid = grid_config.get_alt_grid()
        ne_grid = np.full((len(lat_grid), len(lon_grid), len(alt_grid)), 1e11)
        
        service.initialize_ray_tracer(
            ne_grid=ne_grid,
            lat_grid=lat_grid,
            lon_grid=lon_grid,
            alt_grid=alt_grid,
            xray_flux=1e-6
        )

        start = time.time()
        result = service.calculate_luf_muf()
        elapsed = time.time() - start

        print(f"\nFine frequency sweep ({131} freqs) took {elapsed:.2f}s")

        assert result is not None


class TestCoverageCalculation:
    """Test coverage map calculations"""

    def test_basic_coverage(self):
        """Test basic coverage calculation"""
        service = PropagationService(tx_lat=40.0, tx_lon=-105.0)
        grid_config = GridConfig()

        lat_grid = grid_config.get_lat_grid()
        lon_grid = grid_config.get_lon_grid()
        alt_grid = grid_config.get_alt_grid()
        ne_grid = np.full((len(lat_grid), len(lon_grid), len(alt_grid)), 1e11)
        
        service.initialize_ray_tracer(
            ne_grid=ne_grid,
            lat_grid=lat_grid,
            lon_grid=lon_grid,
            alt_grid=alt_grid,
            xray_flux=1e-6
        )

        coverage = service.calculate_nvis_coverage(freq_mhz=5.0)

        assert coverage is not None
        assert 'coverage_summary' in coverage
        assert coverage['coverage_summary']['total_rays'] > 0

    def test_coverage_many_rays(self):
        """Test coverage with many rays (CPU intensive)"""
        # 21 * 72 = 1,512 rays
        service = PropagationService(
            tx_lat=40.0,
            tx_lon=-105.0,
            elevation_min=70.0,
            elevation_max=90.0,
            elevation_step=1.0,  # 21 elevations
            azimuth_step=5.0     # 72 azimuths
        )
        grid_config = GridConfig(lat_step=20.0, lon_step=20.0, alt_step=100.0)

        lat_grid = grid_config.get_lat_grid()
        lon_grid = grid_config.get_lon_grid()
        alt_grid = grid_config.get_alt_grid()
        ne_grid = np.full((len(lat_grid), len(lon_grid), len(alt_grid)), 1e11)
        
        service.initialize_ray_tracer(
            ne_grid=ne_grid,
            lat_grid=lat_grid,
            lon_grid=lon_grid,
            alt_grid=alt_grid,
            xray_flux=1e-6
        )

        start = time.time()
        coverage = service.calculate_nvis_coverage(freq_mhz=5.0)
        elapsed = time.time() - start

        print(f"\nCoverage with {1512} rays took {elapsed:.2f}s")

        assert coverage is not None
        assert 'coverage_summary' in coverage


class TestNumericalStability:
    """Test numerical stability and edge cases"""

    def test_horizontal_propagation(self):
        """Test with horizontal rays (elevation = 0)"""
        service = PropagationService(
            tx_lat=40.0,
            tx_lon=-105.0,
            elevation_min=0.0,
            elevation_max=10.0,
            elevation_step=2.0
        )
        grid_config = GridConfig()

        lat_grid = grid_config.get_lat_grid()
        lon_grid = grid_config.get_lon_grid()
        alt_grid = grid_config.get_alt_grid()
        ne_grid = np.full((len(lat_grid), len(lon_grid), len(alt_grid)), 1e11)
        
        service.initialize_ray_tracer(
            ne_grid=ne_grid,
            lat_grid=lat_grid,
            lon_grid=lon_grid,
            alt_grid=alt_grid,
            xray_flux=1e-6
        )

        # Should not crash - uses service's elevation config
        result = service.calculate_nvis_coverage(freq_mhz=5.0)

        assert result is not None

    def test_vertical_propagation(self):
        """Test with vertical ray (elevation = 90)"""
        service = PropagationService(tx_lat=40.0, tx_lon=-105.0)
        grid_config = GridConfig()

        lat_grid = grid_config.get_lat_grid()
        lon_grid = grid_config.get_lon_grid()
        alt_grid = grid_config.get_alt_grid()
        ne_grid = np.full((len(lat_grid), len(lon_grid), len(alt_grid)), 1e11)
        
        service.initialize_ray_tracer(
            ne_grid=ne_grid,
            lat_grid=lat_grid,
            lon_grid=lon_grid,
            alt_grid=alt_grid,
            xray_flux=1e-6
        )

        result = service.calculate_nvis_coverage(freq_mhz=5.0)

        assert result is not None

    def test_very_high_frequency(self):
        """Test with very high frequency (should penetrate)"""
        service = PropagationService(tx_lat=40.0, tx_lon=-105.0)
        grid_config = GridConfig()

        lat_grid = grid_config.get_lat_grid()
        lon_grid = grid_config.get_lon_grid()
        alt_grid = grid_config.get_alt_grid()
        ne_grid = np.full((len(lat_grid), len(lon_grid), len(alt_grid)), 1e11)
        
        service.initialize_ray_tracer(
            ne_grid=ne_grid,
            lat_grid=lat_grid,
            lon_grid=lon_grid,
            alt_grid=alt_grid,
            xray_flux=1e-6
        )

        result = service.calculate_nvis_coverage(freq_mhz=100.0)  # VHF - should penetrate ionosphere

        assert result is not None

    def test_very_low_frequency(self):
        """Test with very low frequency (strong reflection)"""
        service = PropagationService(tx_lat=40.0, tx_lon=-105.0)
        grid_config = GridConfig()

        lat_grid = grid_config.get_lat_grid()
        lon_grid = grid_config.get_lon_grid()
        alt_grid = grid_config.get_alt_grid()
        ne_grid = np.full((len(lat_grid), len(lon_grid), len(alt_grid)), 1e11)
        
        service.initialize_ray_tracer(
            ne_grid=ne_grid,
            lat_grid=lat_grid,
            lon_grid=lon_grid,
            alt_grid=alt_grid,
            xray_flux=1e-6
        )

        result = service.calculate_nvis_coverage(freq_mhz=0.5)  # Very low HF

        assert result is not None


class TestConcurrentOperations:
    """Test concurrent ray tracing operations"""

    def test_concurrent_frequency_traces(self):
        """Test tracing multiple frequencies concurrently"""
        service = PropagationService(tx_lat=40.0, tx_lon=-105.0)
        grid_config = GridConfig(lat_step=20.0, lon_step=20.0, alt_step=100.0)

        lat_grid = grid_config.get_lat_grid()
        lon_grid = grid_config.get_lon_grid()
        alt_grid = grid_config.get_alt_grid()
        ne_grid = np.full((len(lat_grid), len(lon_grid), len(alt_grid)), 1e11)
        
        service.initialize_ray_tracer(
            ne_grid=ne_grid,
            lat_grid=lat_grid,
            lon_grid=lon_grid,
            alt_grid=alt_grid,
            xray_flux=1e-6
        )

        def trace_freq(freq):
            return service.calculate_nvis_coverage(freq_mhz=freq)

        freqs = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(trace_freq, freqs))

        assert len(results) == 7
        assert all(r is not None for r in results)

    def test_concurrent_coverage_calculations(self):
        """Test multiple coverage calculations in parallel"""
        service = PropagationService(tx_lat=40.0, tx_lon=-105.0)
        grid_config = GridConfig(lat_step=20.0, lon_step=20.0, alt_step=100.0)

        lat_grid = grid_config.get_lat_grid()
        lon_grid = grid_config.get_lon_grid()
        alt_grid = grid_config.get_alt_grid()
        ne_grid = np.full((len(lat_grid), len(lon_grid), len(alt_grid)), 1e11)
        
        service.initialize_ray_tracer(
            ne_grid=ne_grid,
            lat_grid=lat_grid,
            lon_grid=lon_grid,
            alt_grid=alt_grid,
            xray_flux=1e-6
        )

        def calc_coverage(freq):
            return service.calculate_nvis_coverage(freq_mhz=freq)

        freqs = [3.0, 5.0, 7.0, 9.0]

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(calc_coverage, freqs))

        assert len(results) == 4
        assert all(r is not None for r in results)


class TestCPUIntensivePropagation:
    """CPU-intensive stress tests"""

    def test_ultra_high_resolution_sweep(self):
        """Test with ultra-high resolution frequency and angle sweep"""
        # 17 freqs * 16 elev * 36 az = 9,792 ray traces
        service = PropagationService(
            tx_lat=40.0,
            tx_lon=-105.0,
            freq_min=2.0,
            freq_max=10.0,
            freq_step=0.5,  # 17 frequencies
            elevation_min=75.0,
            elevation_max=90.0,
            elevation_step=1.0,  # 16 elevations
            azimuth_step=10.0    # 36 azimuths
        )
        grid_config = GridConfig(lat_step=20.0, lon_step=20.0, alt_step=100.0)

        lat_grid = grid_config.get_lat_grid()
        lon_grid = grid_config.get_lon_grid()
        alt_grid = grid_config.get_alt_grid()
        ne_grid = np.full((len(lat_grid), len(lon_grid), len(alt_grid)), 1e11)
        
        service.initialize_ray_tracer(
            ne_grid=ne_grid,
            lat_grid=lat_grid,
            lon_grid=lon_grid,
            alt_grid=alt_grid,
            xray_flux=1e-6
        )

        start = time.time()
        result = service.calculate_luf_muf()
        elapsed = time.time() - start

        print(f"\nUltra-high resolution sweep took {elapsed:.2f}s")
        print(f"Expected ~9,792 ray traces")

        assert result is not None

    def test_repeated_luf_muf_calculations(self):
        """Test repeated LUF/MUF calculations (simulating real-time updates)"""
        service = PropagationService(tx_lat=40.0, tx_lon=-105.0)
        grid_config = GridConfig(lat_step=10.0, lon_step=10.0, alt_step=50.0)

        # Initialize once
        lat_grid = grid_config.get_lat_grid()
        lon_grid = grid_config.get_lon_grid()
        alt_grid = grid_config.get_alt_grid()
        ne_grid = np.full((len(lat_grid), len(lon_grid), len(alt_grid)), 1e11)
        
        service.initialize_ray_tracer(
            ne_grid=ne_grid,
            lat_grid=lat_grid,
            lon_grid=lon_grid,
            alt_grid=alt_grid,
            xray_flux=1e-6
        )

        # Calculate many times
        results = []
        start = time.time()
        for i in range(10):
            result = service.calculate_luf_muf()
            results.append(result)

        elapsed = time.time() - start

        print(f"\n10 LUF/MUF calculations took {elapsed:.2f}s")
        print(f"Average: {elapsed/10:.2f}s per calculation")

        assert len(results) == 10
        assert all(r is not None for r in results)

    def test_massive_grid_initialization(self):
        """Test initialization with massive ionospheric grid"""
        service = PropagationService(tx_lat=40.0, tx_lon=-105.0)

        # Large grid (production scale)
        grid_config = GridConfig(
            lat_min=-90.0, lat_max=90.0, lat_step=2.5,   # 73 points
            lon_min=-180.0, lon_max=180.0, lon_step=5.0, # 72 points
            alt_min=60.0, alt_max=600.0, alt_step=10.0   # 55 points
        )
        # 73 * 72 * 55 = 289,080 grid points

        lat_grid = grid_config.get_lat_grid()
        lon_grid = grid_config.get_lon_grid()
        alt_grid = grid_config.get_alt_grid()
        ne_grid = np.full(
            (len(lat_grid), len(lon_grid), len(alt_grid)),
            1e11
        )

        start = time.time()
        service.initialize_ray_tracer(
            ne_grid=ne_grid,
            lat_grid=lat_grid,
            lon_grid=lon_grid,
            alt_grid=alt_grid,
            xray_flux=1e-6
        )
        elapsed = time.time() - start

        print(f"\nInitialization of 289,080-point grid took {elapsed:.2f}s")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-s'])
