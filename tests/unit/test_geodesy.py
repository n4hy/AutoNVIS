"""
Brutal Unit Tests for Geodesy Module

Tests cover:
- Coordinate transformations with extreme precision
- Geographic edge cases (poles, date line, antipodes)
- Distance calculations with numerical stability
- TEC integral computations
"""

import pytest
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from common.geodesy import (
    geographic_to_geocentric,
    geocentric_to_geographic,
    great_circle_distance,
    azimuth_elevation,
    slant_path_integral,
    normalize_longitude,
    grid_bounds_check
)


class TestCoordinateTransformPrecision:
    """Test coordinate transformations with extreme numerical precision"""

    def test_geographic_geocentric_roundtrip_equator(self):
        """Test roundtrip transformation at equator"""
        lat, lon, alt = 0.0, 0.0, 100.0

        # Forward transform
        x, y, z = geographic_to_geocentric(lat, lon, alt)

        # Reverse transform
        lat2, lon2, alt2 = geocentric_to_geographic(x, y, z)

        # Should recover original with high precision
        assert abs(lat2 - lat) < 1e-10
        assert abs(lon2 - lon) < 1e-10
        assert abs(alt2 - alt) < 1e-6  # km

    def test_geographic_geocentric_roundtrip_north_pole(self):
        """Test roundtrip transformation at North Pole"""
        lat, lon, alt = 90.0, 0.0, 1000.0

        x, y, z = geographic_to_geocentric(lat, lon, alt)
        lat2, lon2, alt2 = geocentric_to_geographic(x, y, z)

        # Latitude should be exact at pole
        assert abs(lat2 - lat) < 1e-10
        # Longitude is undefined at pole
        assert abs(alt2 - alt) < 1e-6

    def test_geographic_geocentric_roundtrip_south_pole(self):
        """Test roundtrip transformation at South Pole"""
        lat, lon, alt = -90.0, 0.0, 500.0

        x, y, z = geographic_to_geocentric(lat, lon, alt)
        lat2, lon2, alt2 = geocentric_to_geographic(x, y, z)

        assert abs(lat2 - lat) < 1e-10
        assert abs(alt2 - alt) < 1e-6

    def test_geographic_geocentric_date_line(self):
        """Test transformation at date line (180° longitude)"""
        lat, lon, alt = 0.0, 180.0, 0.0

        x, y, z = geographic_to_geocentric(lat, lon, alt)
        lat2, lon2, alt2 = geocentric_to_geographic(x, y, z)

        assert abs(lat2 - lat) < 1e-10
        # Longitude might wrap to -180, both are equivalent
        assert abs(abs(lon2) - 180.0) < 1e-10
        assert abs(alt2 - alt) < 1e-6

    def test_geographic_geocentric_high_altitude(self):
        """Test transformation at extreme altitude (GPS orbit)"""
        lat, lon, alt = 40.0, -105.0, 20200.0  # GPS altitude in km

        x, y, z = geographic_to_geocentric(lat, lon, alt)
        lat2, lon2, alt2 = geocentric_to_geographic(x, y, z)

        assert abs(lat2 - lat) < 1e-9
        assert abs(lon2 - lon) < 1e-9
        assert abs(alt2 - alt) < 0.01  # 10 meters at GPS altitude

    def test_geographic_geocentric_negative_altitude(self):
        """Test transformation below sea level"""
        lat, lon, alt = 31.5, 35.4, -0.4  # Dead Sea

        x, y, z = geographic_to_geocentric(lat, lon, alt)
        lat2, lon2, alt2 = geocentric_to_geographic(x, y, z)

        assert abs(lat2 - lat) < 1e-10
        assert abs(lon2 - lon) < 1e-10
        assert abs(alt2 - alt) < 1e-6


class TestGreatCircleDistance:
    """Test great circle distance calculations"""

    def test_distance_across_equator(self):
        """Test distance calculation across equator"""
        lat1, lon1 = -1.0, 0.0
        lat2, lon2 = 1.0, 0.0

        dist = great_circle_distance(lat1, lon1, lat2, lon2)

        # 2 degrees at equator ≈ 222 km
        assert 220.0 < dist < 224.0

    def test_distance_across_poles(self):
        """Test distance from North Pole to South Pole"""
        lat1, lon1 = 90.0, 0.0
        lat2, lon2 = -90.0, 0.0

        dist = great_circle_distance(lat1, lon1, lat2, lon2)

        # Half Earth's circumference ≈ 20,000 km
        assert 19900 < dist < 20100

    def test_distance_across_dateline(self):
        """Test distance across international date line"""
        lat1, lon1 = 0.0, 179.0
        lat2, lon2 = 0.0, -179.0

        dist = great_circle_distance(lat1, lon1, lat2, lon2)

        # 2 degrees at equator
        assert 220.0 < dist < 224.0

    def test_distance_to_same_point(self):
        """Test distance to same point is zero"""
        lat, lon = 40.0, -105.0

        dist = great_circle_distance(lat, lon, lat, lon)

        assert abs(dist) < 1e-6

    def test_distance_antipodal_points(self):
        """Test distance between antipodal points"""
        lat1, lon1 = 0.0, 0.0
        lat2, lon2 = 0.0, 180.0

        dist = great_circle_distance(lat1, lon1, lat2, lon2)

        # Half circumference at equator
        assert 19900 < dist < 20100


class TestAzimuthElevation:
    """Test azimuth and elevation calculations"""

    def test_azimuth_elevation_overhead(self):
        """Test elevation angle for point directly overhead"""
        lat, lon, alt = 40.0, -105.0, 0.0
        lat_target, lon_target, alt_target = 40.0, -105.0, 350.0

        az, el = azimuth_elevation(lat, lon, alt, lat_target, lon_target, alt_target)

        # Elevation should be close to 90 degrees (straight up)
        assert el > 85.0

    def test_azimuth_elevation_north(self):
        """Test azimuth for target to the north"""
        lat, lon, alt = 0.0, 0.0, 0.0
        lat_target, lon_target, alt_target = 10.0, 0.0, 0.0

        az, el = azimuth_elevation(lat, lon, alt, lat_target, lon_target, alt_target)

        # Azimuth should be close to 0 (north)
        assert abs(az) < 5.0 or abs(az - 360.0) < 5.0

    def test_azimuth_elevation_east(self):
        """Test azimuth for target to the east"""
        lat, lon, alt = 0.0, 0.0, 0.0
        lat_target, lon_target, alt_target = 0.0, 10.0, 0.0

        az, el = azimuth_elevation(lat, lon, alt, lat_target, lon_target, alt_target)

        # Azimuth should be close to 90 (east)
        assert 85.0 < az < 95.0


class TestSlantPathIntegral:
    """Test TEC integral calculations along slant paths"""

    def test_vertical_path_integration(self):
        """Test integration along vertical path"""
        # Simple uniform grid
        lat_grid = np.array([0.0, 10.0])
        lon_grid = np.array([0.0, 10.0])
        alt_grid = np.array([100.0, 200.0, 300.0])

        # Uniform electron density
        ne_grid = np.full((2, 2, 3), 1e11)

        # Vertical path from ground to 300 km
        tec = slant_path_integral(
            0.0, 0.0, 0.0,  # Start
            0.0, 0.0, 300.0,  # End
            ne_grid, lat_grid, lon_grid, alt_grid,
            n_steps=100
        )

        # TEC should be positive
        assert tec > 0

    def test_horizontal_path_integration(self):
        """Test integration along horizontal path"""
        lat_grid = np.array([0.0, 10.0, 20.0])
        lon_grid = np.array([0.0, 10.0, 20.0])
        alt_grid = np.array([100.0, 200.0, 300.0])

        ne_grid = np.full((3, 3, 3), 1e11)

        # Horizontal path at constant altitude
        tec = slant_path_integral(
            0.0, 0.0, 200.0,
            10.0, 0.0, 200.0,
            ne_grid, lat_grid, lon_grid, alt_grid
        )

        assert tec > 0

    def test_slant_path_with_varying_ne(self):
        """Test integration with spatially varying electron density"""
        lat_grid = np.linspace(-10, 10, 21)
        lon_grid = np.linspace(-10, 10, 21)
        alt_grid = np.linspace(100, 500, 41)

        # Create varying Ne (Chapman layer-like)
        ne_grid = np.zeros((21, 21, 41))
        for i_alt, alt in enumerate(alt_grid):
            h = alt - 300.0
            H = 50.0
            ne_grid[:, :, i_alt] = 1e12 * np.exp(0.5 * (1 - h/H - np.exp(-h/H)))

        tec = slant_path_integral(
            0.0, 0.0, 0.0,
            5.0, 5.0, 400.0,
            ne_grid, lat_grid, lon_grid, alt_grid,
            n_steps=200
        )

        # TEC should be positive and reasonable
        assert 0 < tec < 1000  # TECU


class TestNormalizeLongitude:
    """Test longitude normalization"""

    def test_normalize_in_range(self):
        """Test longitude already in range"""
        assert normalize_longitude(0.0) == 0.0
        assert normalize_longitude(180.0) == 180.0
        assert normalize_longitude(-180.0) == -180.0
        assert normalize_longitude(90.0) == 90.0

    def test_normalize_above_range(self):
        """Test longitude above 180"""
        assert abs(normalize_longitude(270.0) - (-90.0)) < 1e-10
        assert abs(normalize_longitude(360.0)) < 1e-10
        assert abs(normalize_longitude(540.0) - 180.0) < 1e-10

    def test_normalize_below_range(self):
        """Test longitude below -180"""
        assert abs(normalize_longitude(-270.0) - 90.0) < 1e-10
        assert abs(normalize_longitude(-360.0)) < 1e-10
        # -540 = -540 + 2*360 = 180 or -180 (both equivalent)
        result = normalize_longitude(-540.0)
        assert abs(result - 180.0) < 1e-10 or abs(result - (-180.0)) < 1e-10


class TestGridBoundsCheck:
    """Test grid bounds checking"""

    def test_point_inside_bounds(self):
        """Test point clearly inside bounds"""
        assert grid_bounds_check(
            0.0, 0.0, 300.0,
            -90.0, 90.0,
            -180.0, 180.0,
            0.0, 1000.0
        )

    def test_point_outside_lat_bounds(self):
        """Test point outside latitude bounds"""
        assert not grid_bounds_check(
            95.0, 0.0, 300.0,
            -90.0, 90.0,
            -180.0, 180.0,
            0.0, 1000.0
        )

    def test_point_outside_lon_bounds(self):
        """Test point outside longitude bounds"""
        assert not grid_bounds_check(
            0.0, 200.0, 300.0,
            -90.0, 90.0,
            -180.0, 180.0,
            0.0, 1000.0
        )

    def test_point_outside_alt_bounds(self):
        """Test point outside altitude bounds"""
        assert not grid_bounds_check(
            0.0, 0.0, 1500.0,
            -90.0, 90.0,
            -180.0, 180.0,
            0.0, 1000.0
        )

    def test_point_on_boundary(self):
        """Test point exactly on boundary"""
        assert grid_bounds_check(
            90.0, 180.0, 1000.0,
            -90.0, 90.0,
            -180.0, 180.0,
            0.0, 1000.0
        )


class TestNumericalStability:
    """Test numerical stability of geodesy functions"""

    def test_tiny_distances(self):
        """Test distance calculation for very small separations"""
        lat1, lon1 = 40.0, -105.0
        lat2, lon2 = 40.0 + 1e-10, -105.0 + 1e-10

        dist = great_circle_distance(lat1, lon1, lat2, lon2)

        # Should be finite and small
        assert np.isfinite(dist)
        assert dist < 0.01  # Less than 10 meters

    def test_huge_distances(self):
        """Test distance calculation for maximum separation"""
        lat1, lon1 = -90.0, 0.0
        lat2, lon2 = 90.0, 180.0

        dist = great_circle_distance(lat1, lon1, lat2, lon2)

        # Should be close to half circumference
        assert 19000 < dist < 21000
        assert np.isfinite(dist)

    def test_repeated_transformations(self):
        """Test repeated coordinate transformations don't accumulate error"""
        lat, lon, alt = 40.0, -105.0, 100.0

        for _ in range(1000):
            x, y, z = geographic_to_geocentric(lat, lon, alt)
            lat, lon, alt = geocentric_to_geographic(x, y, z)

        # Should still be close to original
        assert abs(lat - 40.0) < 1e-6
        assert abs(lon - (-105.0)) < 1e-6
        assert abs(alt - 100.0) < 1e-3


class TestConcurrency:
    """Test geodesy functions under concurrent load"""

    def test_concurrent_distance_calculations(self):
        """Calculate distances from multiple threads"""
        def calc_distance(i):
            lat1 = -90.0 + i * 0.1
            lon1 = -180.0 + i * 0.1
            lat2 = lat1 + 1.0
            lon2 = lon1 + 1.0
            return great_circle_distance(lat1, lon1, lat2, lon2)

        with ThreadPoolExecutor(max_workers=20) as executor:
            results = list(executor.map(calc_distance, range(1000)))

        assert len(results) == 1000
        assert all(np.isfinite(r) for r in results)

    def test_concurrent_transformations(self):
        """Perform coordinate transformations from multiple threads"""
        def transform(i):
            lat = -90.0 + i * 0.18
            lon = -180.0 + i * 0.36
            alt = i * 0.1

            x, y, z = geographic_to_geocentric(lat, lon, alt)
            lat2, lon2, alt2 = geocentric_to_geographic(x, y, z)

            return abs(lat2 - lat) < 1e-8

        with ThreadPoolExecutor(max_workers=20) as executor:
            results = list(executor.map(transform, range(1000)))

        assert all(results)


class TestCPUIntensiveGeodesyOperations:
    """CPU-intensive stress tests"""

    def test_massive_distance_matrix(self):
        """Compute distance matrix for many points (CPU intensive)"""
        n_points = 500

        # Generate random points
        lats = np.random.uniform(-90, 90, n_points)
        lons = np.random.uniform(-180, 180, n_points)

        # Compute full distance matrix
        distances = np.zeros((n_points, n_points))
        for i in range(n_points):
            for j in range(n_points):
                distances[i, j] = great_circle_distance(lats[i], lons[i], lats[j], lons[j])

        # Verify symmetry
        assert np.allclose(distances, distances.T, atol=1e-6)

        # Diagonal should be zero
        assert np.allclose(np.diag(distances), 0.0, atol=1e-6)

    def test_global_grid_transformations(self):
        """Transform entire global grid to geocentric (CPU/memory intensive)"""
        # 1-degree resolution global grid
        lats = np.arange(-90, 91, 1.0)
        lons = np.arange(-180, 180, 1.0)
        alts = np.arange(0, 1000, 10.0)

        LAT, LON, ALT = np.meshgrid(lats, lons, alts, indexing='ij')

        # Flatten
        lat_flat = LAT.ravel()
        lon_flat = LON.ravel()
        alt_flat = ALT.ravel()

        # Transform all points
        x = np.zeros_like(lat_flat)
        y = np.zeros_like(lat_flat)
        z = np.zeros_like(lat_flat)

        for i in range(len(lat_flat)):
            x[i], y[i], z[i] = geographic_to_geocentric(lat_flat[i], lon_flat[i], alt_flat[i])

        # Verify all finite
        assert np.all(np.isfinite(x))
        assert np.all(np.isfinite(y))
        assert np.all(np.isfinite(z))


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
