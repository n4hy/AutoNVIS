"""
Geodesy and Coordinate Transformation Utilities

This module provides functions for coordinate transformations,
distance calculations, and other geodetic operations.
"""

import numpy as np
from typing import Tuple
from .constants import EARTH_RADIUS_KM, DEG_TO_RAD, RAD_TO_DEG


def geographic_to_geocentric(lat_geo: float, lon_geo: float, alt_km: float) -> Tuple[float, float, float]:
    """
    Convert geographic (geodetic) coordinates to geocentric Cartesian coordinates

    Args:
        lat_geo: Geographic latitude (degrees)
        lon_geo: Geographic longitude (degrees)
        alt_km: Altitude above sea level (kilometers)

    Returns:
        (x, y, z): Geocentric Cartesian coordinates (kilometers)
    """
    lat_rad = lat_geo * DEG_TO_RAD
    lon_rad = lon_geo * DEG_TO_RAD
    r = EARTH_RADIUS_KM + alt_km

    x = r * np.cos(lat_rad) * np.cos(lon_rad)
    y = r * np.cos(lat_rad) * np.sin(lon_rad)
    z = r * np.sin(lat_rad)

    return x, y, z


def geocentric_to_geographic(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """
    Convert geocentric Cartesian coordinates to geographic coordinates

    Args:
        x: X coordinate (kilometers)
        y: Y coordinate (kilometers)
        z: Z coordinate (kilometers)

    Returns:
        (lat, lon, alt): Geographic latitude (deg), longitude (deg), altitude (km)
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    lat_rad = np.arcsin(z / r)
    lon_rad = np.arctan2(y, x)

    lat_geo = lat_rad * RAD_TO_DEG
    lon_geo = lon_rad * RAD_TO_DEG
    alt_km = r - EARTH_RADIUS_KM

    return lat_geo, lon_geo, alt_km


def great_circle_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate great circle distance between two points using Haversine formula

    Args:
        lat1: Latitude of point 1 (degrees)
        lon1: Longitude of point 1 (degrees)
        lat2: Latitude of point 2 (degrees)
        lon2: Longitude of point 2 (degrees)

    Returns:
        Distance in kilometers
    """
    lat1_rad = lat1 * DEG_TO_RAD
    lat2_rad = lat2 * DEG_TO_RAD
    dlat = (lat2 - lat1) * DEG_TO_RAD
    dlon = (lon2 - lon1) * DEG_TO_RAD

    a = (np.sin(dlat / 2)**2 +
         np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return EARTH_RADIUS_KM * c


def azimuth_elevation(lat_obs: float, lon_obs: float, alt_obs: float,
                      lat_target: float, lon_target: float, alt_target: float) -> Tuple[float, float]:
    """
    Calculate azimuth and elevation angle from observer to target

    Args:
        lat_obs: Observer latitude (degrees)
        lon_obs: Observer longitude (degrees)
        alt_obs: Observer altitude (kilometers)
        lat_target: Target latitude (degrees)
        lon_target: Target longitude (degrees)
        alt_target: Target altitude (kilometers)

    Returns:
        (azimuth, elevation): Azimuth (degrees, 0=North, 90=East),
                             Elevation (degrees, 0=horizon, 90=zenith)
    """
    # Convert to geocentric Cartesian
    x_obs, y_obs, z_obs = geographic_to_geocentric(lat_obs, lon_obs, alt_obs)
    x_tgt, y_tgt, z_tgt = geographic_to_geocentric(lat_target, lon_target, alt_target)

    # Vector from observer to target
    dx = x_tgt - x_obs
    dy = y_tgt - y_obs
    dz = z_tgt - z_obs

    # Local East-North-Up frame at observer
    lat_rad = lat_obs * DEG_TO_RAD
    lon_rad = lon_obs * DEG_TO_RAD

    # Transform to ENU coordinates
    east = -np.sin(lon_rad) * dx + np.cos(lon_rad) * dy
    north = -np.sin(lat_rad) * np.cos(lon_rad) * dx - np.sin(lat_rad) * np.sin(lon_rad) * dy + np.cos(lat_rad) * dz
    up = np.cos(lat_rad) * np.cos(lon_rad) * dx + np.cos(lat_rad) * np.sin(lon_rad) * dy + np.sin(lat_rad) * dz

    # Azimuth and elevation
    azimuth = np.arctan2(east, north) * RAD_TO_DEG
    if azimuth < 0:
        azimuth += 360

    horizontal_dist = np.sqrt(east**2 + north**2)
    elevation = np.arctan2(up, horizontal_dist) * RAD_TO_DEG

    return azimuth, elevation


def slant_path_integral(lat1: float, lon1: float, alt1: float,
                        lat2: float, lon2: float, alt2: float,
                        ne_grid: np.ndarray,
                        lat_grid: np.ndarray,
                        lon_grid: np.ndarray,
                        alt_grid: np.ndarray,
                        n_steps: int = 100) -> float:
    """
    Integrate electron density along a slant path (for TEC calculation)

    Args:
        lat1, lon1, alt1: Start point (degrees, degrees, km)
        lat2, lon2, alt2: End point (degrees, degrees, km)
        ne_grid: 3D electron density grid (el/m³), shape (n_lat, n_lon, n_alt)
        lat_grid: Latitude grid values (degrees)
        lon_grid: Longitude grid values (degrees)
        alt_grid: Altitude grid values (km)
        n_steps: Number of integration steps

    Returns:
        Total Electron Content along path (TECU)
    """
    from scipy.interpolate import RegularGridInterpolator

    # Create interpolator
    interp = RegularGridInterpolator(
        (lat_grid, lon_grid, alt_grid),
        ne_grid,
        bounds_error=False,
        fill_value=0.0
    )

    # Generate points along path
    lats = np.linspace(lat1, lat2, n_steps)
    lons = np.linspace(lon1, lon2, n_steps)
    alts = np.linspace(alt1, alt2, n_steps)

    # Calculate distances between consecutive points
    distances_km = np.zeros(n_steps - 1)
    for i in range(n_steps - 1):
        x1, y1, z1 = geographic_to_geocentric(lats[i], lons[i], alts[i])
        x2, y2, z2 = geographic_to_geocentric(lats[i+1], lons[i+1], alts[i+1])
        distances_km[i] = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)

    # Interpolate Ne at each point
    points = np.column_stack([lats, lons, alts])
    ne_values = interp(points)  # el/m³

    # Integrate using trapezoidal rule
    # TEC = ∫ Ne(s) ds  [electrons/m²]
    ne_avg = (ne_values[:-1] + ne_values[1:]) / 2
    tec_electrons_m2 = np.sum(ne_avg * distances_km * 1000)  # Convert km to m

    # Convert to TECU (1 TECU = 10^16 electrons/m²)
    tec_tecu = tec_electrons_m2 / 1e16

    return tec_tecu


def normalize_longitude(lon: float) -> float:
    """
    Normalize longitude to [-180, 180] range

    Args:
        lon: Longitude (degrees)

    Returns:
        Normalized longitude (degrees)
    """
    while lon > 180:
        lon -= 360
    while lon < -180:
        lon += 360
    return lon


def grid_bounds_check(lat: float, lon: float, alt: float,
                      lat_min: float, lat_max: float,
                      lon_min: float, lon_max: float,
                      alt_min: float, alt_max: float) -> bool:
    """
    Check if point is within grid bounds

    Args:
        lat, lon, alt: Point coordinates
        lat_min, lat_max: Latitude bounds
        lon_min, lon_max: Longitude bounds
        alt_min, alt_max: Altitude bounds

    Returns:
        True if point is within bounds
    """
    return (lat_min <= lat <= lat_max and
            lon_min <= lon <= lon_max and
            alt_min <= alt <= alt_max)
