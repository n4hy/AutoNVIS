"""
Physical and Mathematical Constants for Auto-NVIS

This module contains fundamental constants used throughout the
ionospheric modeling and propagation calculations.
"""

import numpy as np

# Earth parameters
EARTH_RADIUS_KM = 6371.0  # Mean Earth radius in kilometers
EARTH_RADIUS_M = EARTH_RADIUS_KM * 1000.0  # Earth radius in meters

# Physical constants
SPEED_OF_LIGHT = 2.99792458e8  # Speed of light in vacuum (m/s)
ELECTRON_MASS = 9.10938356e-31  # Electron mass (kg)
ELECTRON_CHARGE = 1.602176634e-19  # Elementary charge (C)
EPSILON_0 = 8.8541878128e-12  # Permittivity of free space (F/m)
MU_0 = 1.25663706212e-6  # Permeability of free space (H/m)

# Geomagnetic field (approximate dipole parameters)
# These will be updated with IGRF-13 in full implementation
EARTH_MAGNETIC_MOMENT = 7.94e22  # Magnetic dipole moment (A⋅m²)
MAGNETIC_POLE_LATITUDE = 80.65  # Geomagnetic north pole latitude (degrees)
MAGNETIC_POLE_LONGITUDE = -72.68  # Geomagnetic north pole longitude (degrees)

# Ionospheric constants
PLASMA_FREQUENCY_CONSTANT = np.sqrt(ELECTRON_CHARGE**2 / (EPSILON_0 * ELECTRON_MASS))
# fp [Hz] = PLASMA_FREQUENCY_CONSTANT * sqrt(Ne [el/m³]) / (2π)

# Solar flux thresholds (W/m²) for X-ray classification
FLARE_CLASSES = {
    'A': (1e-8, 1e-7),
    'B': (1e-7, 1e-6),
    'C': (1e-6, 1e-5),
    'M': (1e-5, 1e-4),
    'X': (1e-4, float('inf'))
}

# Mode switching threshold
M1_FLARE_THRESHOLD = 1e-5  # M1 class threshold in W/m²

# Operational frequency ranges
HF_BAND_MIN_MHZ = 1.6  # Lower edge of HF band
HF_BAND_MAX_MHZ = 30.0  # Upper edge of HF band
NVIS_RANGE_MAX_KM = 400.0  # Maximum range for NVIS (near-vertical)

# Conversion factors
TECU_TO_ELECTRONS_M2 = 1e16  # 1 TECU = 10^16 electrons/m²
MHZ_TO_HZ = 1e6  # MHz to Hz conversion
DEG_TO_RAD = np.pi / 180.0  # Degrees to radians
RAD_TO_DEG = 180.0 / np.pi  # Radians to degrees

# Time constants
SECONDS_PER_DAY = 86400
MINUTES_PER_DAY = 1440
UPDATE_CYCLE_SECONDS = 900  # 15-minute update cycle

# Numerical stability parameters
MIN_ELECTRON_DENSITY = 1e8  # Minimum Ne (el/m³) for numerical stability
MAX_ELECTRON_DENSITY = 1e13  # Maximum physically reasonable Ne (el/m³)
MIN_COVARIANCE_EIGENVALUE = 1e-10  # Minimum eigenvalue for positive definiteness
