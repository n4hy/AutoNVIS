"""
Data Processing Utilities for Ionospheric Parameters

Provides functions to compute derived ionospheric parameters from
electron density grids:
- foF2: Critical frequency of F2 layer (MHz)
- hmF2: Height of F2 peak (km)
- TEC: Total Electron Content (TECU)
- Horizontal slices and vertical profiles
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any


def compute_fof2(ne_grid: np.ndarray, alt_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute foF2 (critical frequency) and hmF2 (peak height) from Ne grid.

    foF2 is computed using the plasma frequency equation:
    f_p = 8.98 * sqrt(N_e) MHz, where N_e is in electrons/m³

    Args:
        ne_grid: Electron density grid (lat, lon, alt) in el/m³
        alt_grid: Altitude grid in km

    Returns:
        Tuple of (fof2_map, hmf2_map):
        - fof2_map: 2D array (lat, lon) with critical frequency in MHz
        - hmf2_map: 2D array (lat, lon) with F2 peak height in km
    """
    n_lat, n_lon, n_alt = ne_grid.shape

    fof2_map = np.zeros((n_lat, n_lon))
    hmf2_map = np.zeros((n_lat, n_lon))

    for i in range(n_lat):
        for j in range(n_lon):
            # Extract vertical profile
            ne_profile = ne_grid[i, j, :]

            # Find F2 peak (maximum Ne)
            peak_idx = np.argmax(ne_profile)
            ne_max = ne_profile[peak_idx]

            # Compute foF2 from peak density
            # f_p [MHz] = 8.98 * sqrt(N_e [el/m³])
            fof2_map[i, j] = 8.98 * np.sqrt(ne_max)

            # Get peak height
            hmf2_map[i, j] = alt_grid[peak_idx]

    return fof2_map, hmf2_map


def compute_hmf2(ne_grid: np.ndarray, alt_grid: np.ndarray) -> np.ndarray:
    """
    Compute hmF2 (height of F2 peak) from Ne grid.

    Args:
        ne_grid: Electron density grid (lat, lon, alt) in el/m³
        alt_grid: Altitude grid in km

    Returns:
        2D array (lat, lon) with F2 peak height in km
    """
    _, hmf2_map = compute_fof2(ne_grid, alt_grid)
    return hmf2_map


def compute_tec(ne_grid: np.ndarray, alt_grid: np.ndarray) -> np.ndarray:
    """
    Compute vertical Total Electron Content (TEC) from Ne grid.

    TEC is the line integral of electron density along the altitude path:
    TEC = ∫ N_e dh

    Result is converted to TECU (1 TECU = 10^16 el/m²)

    Args:
        ne_grid: Electron density grid (lat, lon, alt) in el/m³
        alt_grid: Altitude grid in km

    Returns:
        2D array (lat, lon) with vertical TEC in TECU
    """
    n_lat, n_lon, n_alt = ne_grid.shape

    tec_map = np.zeros((n_lat, n_lon))

    # Convert altitude from km to m for integration
    alt_m = alt_grid * 1000.0

    for i in range(n_lat):
        for j in range(n_lon):
            # Extract vertical profile
            ne_profile = ne_grid[i, j, :]

            # Integrate using trapezoidal rule
            # Result in el/m²
            tec_el_m2 = np.trapz(ne_profile, alt_m)

            # Convert to TECU (1 TECU = 10^16 el/m²)
            tec_map[i, j] = tec_el_m2 / 1e16

    return tec_map


def extract_horizontal_slice(
    ne_grid: np.ndarray,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    alt_grid: np.ndarray,
    altitude_km: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract horizontal slice from 3D Ne grid at specified altitude.

    If altitude doesn't match grid exactly, uses nearest neighbor.

    Args:
        ne_grid: Electron density grid (lat, lon, alt) in el/m³
        lat_grid: Latitude grid in degrees
        lon_grid: Longitude grid in degrees
        alt_grid: Altitude grid in km
        altitude_km: Altitude for slice in km

    Returns:
        Tuple of (lat_grid, lon_grid, ne_slice):
        - lat_grid: Latitude values
        - lon_grid: Longitude values
        - ne_slice: 2D array (lat, lon) with Ne at specified altitude
    """
    # Find nearest altitude index
    alt_idx = np.argmin(np.abs(alt_grid - altitude_km))
    actual_alt = alt_grid[alt_idx]

    # Extract slice at this altitude
    ne_slice = ne_grid[:, :, alt_idx]

    return lat_grid, lon_grid, ne_slice


def extract_vertical_profile(
    ne_grid: np.ndarray,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    alt_grid: np.ndarray,
    latitude: float,
    longitude: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract vertical Ne profile at specified lat/lon.

    Uses nearest neighbor if lat/lon doesn't match grid exactly.

    Args:
        ne_grid: Electron density grid (lat, lon, alt) in el/m³
        lat_grid: Latitude grid in degrees
        lon_grid: Longitude grid in degrees
        alt_grid: Altitude grid in km
        latitude: Latitude for profile in degrees
        longitude: Longitude for profile in degrees

    Returns:
        Tuple of (alt_grid, ne_profile):
        - alt_grid: Altitude values in km
        - ne_profile: Ne values along altitude at specified lat/lon
    """
    # Find nearest lat/lon indices
    lat_idx = np.argmin(np.abs(lat_grid - latitude))
    lon_idx = np.argmin(np.abs(lon_grid - longitude))

    # Extract vertical profile
    ne_profile = ne_grid[lat_idx, lon_idx, :]

    return alt_grid, ne_profile


def compute_grid_statistics(ne_grid: np.ndarray) -> Dict[str, float]:
    """
    Compute statistical properties of electron density grid.

    Args:
        ne_grid: Electron density grid in el/m³

    Returns:
        Dictionary with statistics (min, max, mean, std, median)
    """
    return {
        'ne_min': float(np.min(ne_grid)),
        'ne_max': float(np.max(ne_grid)),
        'ne_mean': float(np.mean(ne_grid)),
        'ne_std': float(np.std(ne_grid)),
        'ne_median': float(np.median(ne_grid)),
        'ne_p25': float(np.percentile(ne_grid, 25)),
        'ne_p75': float(np.percentile(ne_grid, 75))
    }


def detect_ionospheric_layers(
    ne_profile: np.ndarray,
    alt_grid: np.ndarray,
    min_prominence: float = 1e10
) -> Dict[str, Optional[Dict[str, float]]]:
    """
    Detect ionospheric layers (E, F1, F2) from vertical Ne profile.

    Uses simple peak detection with minimum prominence requirement.

    Args:
        ne_profile: Vertical Ne profile in el/m³
        alt_grid: Altitude grid in km
        min_prominence: Minimum peak prominence for detection

    Returns:
        Dictionary with layer information (height, density, fof2)
        Keys: 'E_layer', 'F1_layer', 'F2_layer'
    """
    from scipy.signal import find_peaks

    # Find peaks in Ne profile
    peaks, properties = find_peaks(ne_profile, prominence=min_prominence)

    layers = {
        'E_layer': None,
        'F1_layer': None,
        'F2_layer': None
    }

    if len(peaks) == 0:
        return layers

    # Sort peaks by altitude
    sorted_indices = np.argsort(alt_grid[peaks])
    sorted_peaks = peaks[sorted_indices]

    # Classify layers by altitude ranges
    for peak_idx in sorted_peaks:
        alt = alt_grid[peak_idx]
        ne = ne_profile[peak_idx]
        fof = 8.98 * np.sqrt(ne)

        layer_info = {
            'altitude_km': float(alt),
            'ne_el_m3': float(ne),
            'fof_mhz': float(fof)
        }

        # E layer: 90-150 km
        if 90 <= alt <= 150 and layers['E_layer'] is None:
            layers['E_layer'] = layer_info
        # F1 layer: 150-250 km
        elif 150 <= alt <= 250 and layers['F1_layer'] is None:
            layers['F1_layer'] = layer_info
        # F2 layer: 250-600 km
        elif 250 <= alt <= 600:
            # Take highest F2 peak if multiple
            if layers['F2_layer'] is None or ne > layers['F2_layer']['ne_el_m3']:
                layers['F2_layer'] = layer_info

    return layers


def compute_chapman_layer_fit(
    ne_profile: np.ndarray,
    alt_grid: np.ndarray
) -> Optional[Dict[str, float]]:
    """
    Fit Chapman layer model to vertical Ne profile.

    Chapman layer: N_e(h) = N_max * exp(0.5 * (1 - z - exp(-z)))
    where z = (h - h_max) / H

    Args:
        ne_profile: Vertical Ne profile in el/m³
        alt_grid: Altitude grid in km

    Returns:
        Dictionary with fit parameters: ne_max, h_max, scale_height
        or None if fit fails
    """
    from scipy.optimize import curve_fit

    def chapman_function(h, ne_max, h_max, H):
        z = (h - h_max) / H
        return ne_max * np.exp(0.5 * (1 - z - np.exp(-z)))

    try:
        # Initial guess: peak values
        peak_idx = np.argmax(ne_profile)
        ne_max_guess = ne_profile[peak_idx]
        h_max_guess = alt_grid[peak_idx]
        H_guess = 50.0  # Typical scale height ~50 km

        # Fit Chapman function
        popt, pcov = curve_fit(
            chapman_function,
            alt_grid,
            ne_profile,
            p0=[ne_max_guess, h_max_guess, H_guess],
            bounds=([0, 0, 10], [np.inf, 1000, 200]),
            maxfev=1000
        )

        ne_max, h_max, H = popt

        # Compute goodness of fit (R²)
        residuals = ne_profile - chapman_function(alt_grid, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((ne_profile - np.mean(ne_profile))**2)
        r_squared = 1 - (ss_res / ss_tot)

        return {
            'ne_max': float(ne_max),
            'h_max': float(h_max),
            'scale_height': float(H),
            'r_squared': float(r_squared)
        }

    except Exception:
        return None
