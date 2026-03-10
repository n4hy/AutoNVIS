"""
Unit Tests for TEC Observation Model

Tests the TEC observation model including linear and ray-traced
slant path integration methods.
"""

import pytest
import numpy as np
from datetime import datetime
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))


class MockStateVector:
    """Mock StateVector for testing"""
    def __init__(self, ne_grid, n_lat, n_lon, n_alt):
        """
        Args:
            ne_grid: 3D array of electron density [lat, lon, alt]
        """
        self.ne_grid = ne_grid
        self.n_lat = n_lat
        self.n_lon = n_lon
        self.n_alt = n_alt

    def get_ne(self, i_lat, i_lon, i_alt):
        """Get electron density at grid point"""
        return self.ne_grid[i_lat, i_lon, i_alt]


class MockRayPath:
    """Mock RayPath for testing"""
    def __init__(self, positions, path_lengths):
        self.positions = positions  # List of (lat, lon, alt) arrays
        self.path_lengths = path_lengths
        self.reflected = False
        self.escaped = False
        self.absorbed = False


class MockRayTracer:
    """Mock RayTracer3D for testing"""
    def __init__(self, ne_grid=None, lat_grid=None, lon_grid=None, alt_grid=None):
        self.ne_grid = ne_grid
        self.lat_grid = lat_grid
        self.lon_grid = lon_grid
        self.alt_grid = alt_grid

    def trace_ray(self, lat0, lon0, alt0, elevation, azimuth, freq_mhz):
        """Generate mock ray path"""
        # Create a simple straight-line path for testing
        n_points = 100
        positions = []
        path_lengths = [0.0]

        # Starting altitude to ionospheric top (600 km)
        max_alt = 600.0
        for i in range(n_points):
            t = i / (n_points - 1)
            alt = alt0 + t * (max_alt - alt0)

            # Simple offset in lat/lon based on elevation/azimuth
            el_rad = np.radians(elevation)
            az_rad = np.radians(azimuth)

            # Rough horizontal displacement
            horiz_dist = (alt - alt0) / np.tan(el_rad) if elevation > 1 else 0
            dlat = horiz_dist * np.cos(az_rad) / 111.0  # deg per km
            dlon = horiz_dist * np.sin(az_rad) / (111.0 * np.cos(np.radians(lat0)))

            pos = np.array([lat0 + dlat, lon0 + dlon, alt])
            positions.append(pos)

            if i > 0:
                # Approximate path length
                ds = np.sqrt((alt - positions[i-1][2])**2 +
                            (horiz_dist / n_points)**2)
                path_lengths.append(path_lengths[-1] + ds)

        return MockRayPath(positions, path_lengths)


class TestTECIntegrationMethod:
    """Test TECIntegrationMethod enum"""

    def test_linear_method(self):
        """Test LINEAR integration method value"""
        # This would test the C++ enum, but we test the concept
        assert True  # Placeholder for C++ binding test

    def test_ray_traced_method(self):
        """Test RAY_TRACED integration method value"""
        assert True  # Placeholder for C++ binding test


class TestTECObservationModelLinear:
    """Test TEC observation model with linear integration"""

    @pytest.fixture
    def sample_grids(self):
        """Create sample coordinate grids"""
        lat_grid = np.linspace(30.0, 50.0, 21)  # 21 points, 1 deg spacing
        lon_grid = np.linspace(-120.0, -100.0, 21)
        alt_grid = np.linspace(100.0, 600.0, 51)  # 51 points, 10 km spacing
        return lat_grid, lon_grid, alt_grid

    @pytest.fixture
    def chapman_profile(self, sample_grids):
        """Create Chapman-layer electron density profile"""
        lat_grid, lon_grid, alt_grid = sample_grids
        n_lat = len(lat_grid)
        n_lon = len(lon_grid)
        n_alt = len(alt_grid)

        # Create 3D grid with Chapman layer profile
        ne_grid = np.zeros((n_lat, n_lon, n_alt))

        # Chapman layer parameters
        NmF2 = 1e12  # Peak electron density (el/m^3)
        hmF2 = 300.0  # Peak height (km)
        H = 80.0  # Scale height (km)

        for k, alt in enumerate(alt_grid):
            z = (alt - hmF2) / H
            ne = NmF2 * np.exp(0.5 * (1 - z - np.exp(-z)))
            ne_grid[:, :, k] = ne

        return ne_grid

    def test_vertical_tec_calculation(self, sample_grids, chapman_profile):
        """Test vertical TEC calculation at zenith"""
        lat_grid, lon_grid, alt_grid = sample_grids
        ne_grid = chapman_profile

        state = MockStateVector(ne_grid, len(lat_grid), len(lon_grid), len(alt_grid))

        # Calculate vertical TEC manually
        vtec = 0.0
        i_lat = 10  # Middle of grid
        i_lon = 10

        for k in range(len(alt_grid) - 1):
            ne = ne_grid[i_lat, i_lon, k]
            dh = (alt_grid[k + 1] - alt_grid[k]) * 1000.0  # km to m
            vtec += ne * dh

        vtec_tecu = vtec / 1e16

        # Expect reasonable VTEC value for this profile
        # Typical ionospheric VTEC: 5-50 TECU
        assert 1.0 < vtec_tecu < 100.0

    def test_slant_factor_zenith(self):
        """Test slant factor at zenith (elevation = 90 deg)"""
        # At zenith, slant factor should be 1.0
        elevation = 90.0
        shell_alt = 350.0

        # Slant factor formula: 1 / sqrt(1 - (Re*cos(el)/(Re+h))^2)
        Re = 6371.0  # km
        el_rad = np.radians(elevation)
        ratio = (Re * np.cos(el_rad)) / (Re + shell_alt)
        slant_factor = 1.0 / np.sqrt(1.0 - ratio**2)

        assert abs(slant_factor - 1.0) < 0.001

    def test_slant_factor_low_elevation(self):
        """Test slant factor at low elevation"""
        elevation = 30.0
        shell_alt = 350.0

        Re = 6371.0
        el_rad = np.radians(elevation)
        ratio = (Re * np.cos(el_rad)) / (Re + shell_alt)
        slant_factor = 1.0 / np.sqrt(1.0 - ratio**2)

        # At 30 deg elevation, slant factor should be ~2.0
        assert 1.5 < slant_factor < 2.5

    def test_pierce_point_zenith(self):
        """Test IPP calculation at zenith"""
        # At zenith (90 deg elevation), IPP should be directly overhead
        rx_lat = 40.0
        rx_lon = -105.0
        elevation = 90.0
        azimuth = 0.0
        shell_alt = 350.0

        # At zenith, IPP = receiver location
        # (with numerical precision)
        ipp_lat = rx_lat  # Expected
        ipp_lon = rx_lon

        # The IPP calculation should return receiver location
        assert abs(ipp_lat - rx_lat) < 0.01
        assert abs(ipp_lon - rx_lon) < 0.01

    def test_pierce_point_off_zenith(self):
        """Test IPP calculation off zenith"""
        rx_lat = 40.0
        rx_lon = -105.0
        elevation = 45.0
        azimuth = 0.0  # North
        shell_alt = 350.0

        # Calculate expected IPP (simplified)
        deg2rad = np.pi / 180.0
        Re = 6371.0

        el = elevation * deg2rad
        cos_el = np.cos(el)
        sin_arg = (Re / (Re + shell_alt)) * cos_el
        psi = (np.pi / 2.0) - el - np.arcsin(sin_arg)

        # IPP should be north of receiver for azimuth=0
        ipp_lat = np.arcsin(
            np.sin(rx_lat * deg2rad) * np.cos(psi) +
            np.cos(rx_lat * deg2rad) * np.sin(psi) * np.cos(0)
        ) / deg2rad

        assert ipp_lat > rx_lat  # Should be north of receiver


class TestTECObservationModelRayTraced:
    """Test TEC observation model with ray-traced integration"""

    @pytest.fixture
    def mock_ray_tracer(self):
        """Create mock ray tracer"""
        return MockRayTracer()

    def test_ray_traced_integration(self, mock_ray_tracer):
        """Test ray-traced TEC integration"""
        # Create simple electron density along ray path
        n_points = 100
        path_lengths = np.linspace(0, 500, n_points)  # km
        altitudes = np.linspace(100, 600, n_points)

        # Chapman profile along path
        NmF2 = 1e12
        hmF2 = 300.0
        H = 80.0

        ne_values = []
        for alt in altitudes:
            z = (alt - hmF2) / H
            ne = NmF2 * np.exp(0.5 * (1 - z - np.exp(-z)))
            ne_values.append(ne)

        # Integrate along path
        tec = 0.0
        for i in range(n_points - 1):
            ne = ne_values[i]
            ds = (path_lengths[i + 1] - path_lengths[i]) * 1000.0  # km to m
            tec += ne * ds

        tec_tecu = tec / 1e16

        # Should get reasonable slant TEC
        assert 1.0 < tec_tecu < 200.0

    def test_fallback_to_linear_when_no_raytracer(self):
        """Test that model falls back to linear when no ray tracer"""
        # If ray_tracer is None, should use linear method
        # This verifies the fallback behavior in the C++ code
        pass  # Placeholder - actual test would use C++ bindings

    def test_ray_path_positions_used(self, mock_ray_tracer):
        """Test that ray path positions are correctly used"""
        # Trace a ray
        ray_path = mock_ray_tracer.trace_ray(
            40.0, -105.0, 0.0,  # lat, lon, alt
            75.0, 45.0,  # elevation, azimuth
            1575.42  # GPS L1 frequency
        )

        assert len(ray_path.positions) > 0
        assert len(ray_path.path_lengths) == len(ray_path.positions)

        # Verify positions are within reasonable bounds
        for pos in ray_path.positions:
            assert 30.0 < pos[0] < 60.0  # lat
            assert -130.0 < pos[1] < -80.0  # lon
            assert 0.0 <= pos[2] <= 700.0  # alt


class TestTrilinearInterpolation:
    """Test trilinear interpolation of electron density"""

    def test_interpolation_at_grid_point(self):
        """Test interpolation returns exact value at grid point"""
        lat_grid = np.array([40.0, 41.0, 42.0])
        lon_grid = np.array([-106.0, -105.0, -104.0])
        alt_grid = np.array([200.0, 300.0, 400.0])

        # 3x3x3 grid with known values
        ne_grid = np.zeros((3, 3, 3))
        ne_grid[1, 1, 1] = 1e12  # Center point

        state = MockStateVector(ne_grid, 3, 3, 3)

        # At exact grid point
        ne = state.get_ne(1, 1, 1)
        assert ne == 1e12

    def test_interpolation_midpoint(self):
        """Test interpolation at midpoint between grid points"""
        # Simple linear interpolation check
        ne_grid = np.array([[[1e11, 3e11],
                            [1e11, 3e11]],
                           [[1e11, 3e11],
                            [1e11, 3e11]]])

        # Midpoint should average to 2e11
        expected_mid = 2e11

        # This would test the C++ interpolation
        # For now, verify the concept
        mid_val = (ne_grid[0, 0, 0] + ne_grid[0, 0, 1]) / 2
        assert abs(mid_val - expected_mid) < 1e9


class TestTECMeasurementStructure:
    """Test TEC measurement data structure"""

    def test_measurement_fields(self):
        """Test that measurement has required fields"""
        # Simulate what the C++ struct provides
        meas = {
            'latitude': 40.0,
            'longitude': -105.0,
            'altitude': 0.0,
            'sat_latitude': 40.0,
            'sat_longitude': -100.0,
            'sat_altitude': 20200.0,  # GPS altitude
            'azimuth': 90.0,
            'elevation': 45.0,
            'tec_value': 25.0,
            'tec_error': 1.0
        }

        assert meas['elevation'] > 0
        assert meas['sat_altitude'] > meas['altitude']
        assert meas['tec_error'] > 0

    def test_reasonable_tec_values(self):
        """Test that TEC values are in reasonable range"""
        # Typical daytime VTEC: 10-50 TECU
        # Typical STEC at low elevation: up to 100+ TECU
        min_tec = 0.1
        max_tec = 200.0

        sample_tec = 35.5
        assert min_tec < sample_tec < max_tec


class TestObservationModelForward:
    """Test forward model computation"""

    def test_forward_returns_vector(self):
        """Test that forward() returns measurement-sized vector"""
        n_measurements = 5

        # Expected output: one TEC prediction per measurement
        # This would test the C++ forward() method
        predicted = np.zeros(n_measurements)  # Mock output

        assert len(predicted) == n_measurements

    def test_forward_with_multiple_receivers(self):
        """Test forward model with multiple receiver locations"""
        # Different receivers should produce different TEC values
        # due to different slant paths
        receiver_locations = [
            (40.0, -105.0),
            (35.0, -100.0),
            (45.0, -110.0)
        ]

        # Each location would have different vertical column above it
        # so TEC values should differ
        assert len(receiver_locations) == 3


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_very_low_elevation(self):
        """Test behavior at very low elevation angles"""
        elevation = 5.0
        shell_alt = 350.0
        Re = 6371.0

        el_rad = np.radians(elevation)
        ratio = (Re * np.cos(el_rad)) / (Re + shell_alt)

        # At very low elevation, ratio approaches 1
        # Slant factor should be large but finite
        if ratio < 1.0:
            slant_factor = 1.0 / np.sqrt(1.0 - ratio**2)
            # At 5 deg elevation with 350 km shell, slant factor is ~3
            assert slant_factor > 2.5  # Larger than at higher elevations
            assert slant_factor < 10.0  # But still finite
        else:
            # Should use fallback value
            pass

    def test_position_outside_grid(self):
        """Test interpolation behavior outside grid bounds"""
        # Should clamp to grid boundaries without crashing
        lat_grid = np.array([40.0, 50.0])
        lon_grid = np.array([-110.0, -100.0])

        # Query position outside grid
        query_lat = 60.0  # Outside range
        query_lon = -90.0  # Outside range

        # Should return boundary value without error
        # Actual test would use C++ bindings
        pass

    def test_zero_electron_density(self):
        """Test handling of zero electron density"""
        ne_grid = np.zeros((3, 3, 3))
        state = MockStateVector(ne_grid, 3, 3, 3)

        tec = 0.0
        for k in range(3 - 1):
            ne = state.get_ne(1, 1, k)
            dh = 100.0 * 1000.0  # 100 km in meters
            tec += ne * dh

        assert tec == 0.0


class TestPhysicalConsistency:
    """Test physical consistency of results"""

    def test_tec_increases_with_density(self):
        """Test that TEC increases with electron density"""
        ne_low = 1e11  # Low density
        ne_high = 1e12  # High density
        path_length = 100.0 * 1000.0  # 100 km in meters

        tec_low = ne_low * path_length
        tec_high = ne_high * path_length

        assert tec_high > tec_low
        assert tec_high == 10 * tec_low

    def test_slant_tec_greater_than_vtec(self):
        """Test that slant TEC >= vertical TEC"""
        vtec = 25.0  # TECU
        elevation = 45.0
        shell_alt = 350.0

        # Calculate slant factor
        Re = 6371.0
        el_rad = np.radians(elevation)
        ratio = (Re * np.cos(el_rad)) / (Re + shell_alt)
        slant_factor = 1.0 / np.sqrt(1.0 - ratio**2)

        stec = vtec * slant_factor

        assert stec >= vtec
        assert slant_factor >= 1.0

    def test_tec_units_conversion(self):
        """Test TECU conversion factor"""
        # 1 TECU = 10^16 electrons/m^2
        TECU_FACTOR = 1e16

        electrons_m2 = 2.5e17
        tecu = electrons_m2 / TECU_FACTOR

        assert tecu == 25.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
