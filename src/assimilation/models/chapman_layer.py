"""
Chapman Layer Physics Model

Implements a simplified but physically-motivated ionospheric model using
Chapman layer theory. More realistic than Gauss-Markov, but simpler than
full IRI-2020 integration.

Chapman Layer Equation:
Ne(h) = NmF2 * exp(0.5 * (1 - z - exp(-z)))

where:
- z = (h - hmF2) / H
- H = scale height (~50 km)
- NmF2 = peak electron density (~1-10 ×10¹¹ el/m³)
- hmF2 = peak height (250-400 km)

This model includes:
- Diurnal variation (solar zenith angle dependence)
- Latitudinal variation (equatorial enhancement)
- Solar cycle variation (via effective sunspot number)
- Geomagnetic latitude effects
"""

import numpy as np
from typing import Tuple
from datetime import datetime
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.common.constants import EARTH_RADIUS_KM  # Earth radius


class ChapmanLayerModel:
    """
    Chapman layer ionospheric physics model

    Provides background ionospheric state based on Chapman layer theory
    with empirical corrections for diurnal, latitudinal, and solar cycle variations.
    """

    def __init__(self):
        """Initialize Chapman layer model with default parameters"""

        # Chapman layer parameters
        self.H = 50.0  # Scale height (km)

        # F2 layer peak parameters (climatological)
        self.hmF2_base = 300.0  # Base peak height (km)
        self.hmF2_variation = 50.0  # Diurnal variation amplitude (km)

        self.NmF2_base = 5e11  # Base peak density (el/m³)
        self.NmF2_solar_coeff = 3e9  # Solar cycle coefficient (el/m³ per SSN unit)

        # Equatorial enhancement factor
        self.equatorial_width = 30.0  # degrees
        self.equatorial_enhancement = 0.3  # 30% enhancement at equator

    def solar_zenith_angle(
        self,
        latitude: float,
        longitude: float,
        time: datetime
    ) -> float:
        """
        Compute solar zenith angle (simplified)

        Args:
            latitude: Geographic latitude (degrees)
            longitude: Geographic longitude (degrees)
            time: UTC datetime

        Returns:
            Solar zenith angle (radians)
        """
        # Hour angle (simplified - ignores equation of time)
        hour = time.hour + time.minute / 60.0 + time.second / 3600.0
        local_solar_time = hour + longitude / 15.0
        hour_angle = (local_solar_time - 12.0) * 15.0  # degrees

        # Solar declination (simplified - assumes equinox)
        # Full calculation would use day of year
        day_of_year = time.timetuple().tm_yday
        declination = 23.45 * np.sin(np.deg2rad((360.0 / 365.0) * (day_of_year - 81)))

        # Solar zenith angle
        lat_rad = np.deg2rad(latitude)
        decl_rad = np.deg2rad(declination)
        ha_rad = np.deg2rad(hour_angle)

        cos_sza = (
            np.sin(lat_rad) * np.sin(decl_rad) +
            np.cos(lat_rad) * np.cos(decl_rad) * np.cos(ha_rad)
        )

        sza = np.arccos(np.clip(cos_sza, -1.0, 1.0))

        return sza

    def chapman_function(self, z: float) -> float:
        """
        Chapman layer function

        Args:
            z: Reduced height (h - hmF2) / H

        Returns:
            Ne / NmF2 (normalized electron density)
        """
        return np.exp(0.5 * (1.0 - z - np.exp(-z)))

    def compute_fof2_hmf2(
        self,
        latitude: float,
        longitude: float,
        time: datetime,
        ssn: float = 75.0
    ) -> Tuple[float, float]:
        """
        Compute F2 layer critical frequency and peak height

        Args:
            latitude: Geographic latitude (degrees)
            longitude: Geographic longitude (degrees)
            time: UTC datetime
            ssn: Effective sunspot number (0-200)

        Returns:
            (foF2, hmF2): Critical frequency (MHz) and peak height (km)
        """
        # Solar zenith angle effect
        sza = self.solar_zenith_angle(latitude, longitude, time)
        cos_chi = np.cos(sza)

        # Diurnal variation (Chapman cosine law, but with floor)
        # Ensure cos_chi is positive before taking square root
        cos_chi_positive = np.maximum(0.01, cos_chi)  # Small positive value for night
        diurnal_factor = np.maximum(0.3, cos_chi_positive ** 0.5)

        # Latitudinal variation (equatorial enhancement)
        lat_factor = 1.0 + self.equatorial_enhancement * np.exp(
            -(latitude / self.equatorial_width) ** 2
        )

        # Solar cycle variation
        solar_factor = 1.0 + (ssn / 100.0) * 0.5  # 50% increase at SSN=100

        # Peak electron density
        NmF2 = (
            self.NmF2_base *
            diurnal_factor *
            lat_factor *
            solar_factor
        )

        # Peak height (increases during day, lower at night)
        hmF2 = self.hmF2_base + self.hmF2_variation * diurnal_factor * 0.5

        # Convert to foF2 (plasma frequency)
        # foF2 (MHz) = 9 * sqrt(NmF2 / 10^12)
        foF2 = 9.0 * np.sqrt(NmF2 / 1e12)

        return foF2, hmF2

    def compute_ne_profile(
        self,
        latitude: float,
        longitude: float,
        altitude_km: np.ndarray,
        time: datetime,
        ssn: float = 75.0
    ) -> np.ndarray:
        """
        Compute vertical electron density profile

        Args:
            latitude: Geographic latitude (degrees)
            longitude: Geographic longitude (degrees)
            altitude_km: Altitude grid (km)
            time: UTC datetime
            ssn: Effective sunspot number

        Returns:
            Electron density profile (el/m³)
        """
        # Get F2 layer parameters
        foF2, hmF2 = self.compute_fof2_hmf2(latitude, longitude, time, ssn)

        # Convert foF2 to NmF2
        NmF2 = (foF2 / 9.0) ** 2 * 1e12

        # Reduced height
        z = (altitude_km - hmF2) / self.H

        # Chapman layer
        ne_profile = NmF2 * self.chapman_function(z)

        # Add E-layer contribution (simplified)
        # E-layer peak ~110 km, much weaker than F2
        E_layer_height = 110.0
        E_layer_width = 20.0
        E_layer_peak = NmF2 * 0.1  # 10% of F2 peak

        E_contribution = E_layer_peak * np.exp(
            -((altitude_km - E_layer_height) / E_layer_width) ** 2
        )

        ne_profile += E_contribution

        # Ensure physical bounds
        ne_profile = np.clip(ne_profile, 1e8, 1e13)

        return ne_profile

    def compute_3d_grid(
        self,
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
        alt_grid: np.ndarray,
        time: datetime,
        ssn: float = 75.0
    ) -> np.ndarray:
        """
        Compute 3D ionospheric electron density grid

        Args:
            lat_grid: Latitude grid (degrees)
            lon_grid: Longitude grid (degrees)
            alt_grid: Altitude grid (km)
            time: UTC datetime
            ssn: Effective sunspot number

        Returns:
            3D electron density array (n_lat × n_lon × n_alt) in el/m³
        """
        n_lat = len(lat_grid)
        n_lon = len(lon_grid)
        n_alt = len(alt_grid)

        ne_grid = np.zeros((n_lat, n_lon, n_alt))

        for i, lat in enumerate(lat_grid):
            for j, lon in enumerate(lon_grid):
                ne_grid[i, j, :] = self.compute_ne_profile(
                    lat, lon, alt_grid, time, ssn
                )

        return ne_grid

    def validate_grid(self, ne_grid: np.ndarray) -> dict:
        """
        Validate generated electron density grid

        Args:
            ne_grid: 3D electron density array

        Returns:
            Validation metrics dictionary
        """
        return {
            'min_ne': float(np.min(ne_grid)),
            'max_ne': float(np.max(ne_grid)),
            'mean_ne': float(np.mean(ne_grid)),
            'median_ne': float(np.median(ne_grid)),
            'invalid_count': int(np.sum((ne_grid < 1e8) | (ne_grid > 1e13)))
        }


def main():
    """Demonstration of Chapman layer model"""
    import matplotlib.pyplot as plt

    print("Chapman Layer Model Demonstration")
    print("=" * 50)

    model = ChapmanLayerModel()

    # Test location: Wallops Island, VA (NASA)
    latitude = 37.9
    longitude = -75.5
    time = datetime(2026, 3, 21, 18, 0, 0)  # Vernal equinox, 6 PM UTC

    # Compute foF2 and hmF2
    foF2, hmF2 = model.compute_fof2_hmf2(latitude, longitude, time, ssn=75.0)
    print(f"\nLocation: {latitude}°N, {longitude}°E")
    print(f"Time: {time} UTC")
    print(f"foF2: {foF2:.2f} MHz")
    print(f"hmF2: {hmF2:.1f} km")

    # Compute vertical profile
    alt_grid = np.arange(60, 600, 5)
    ne_profile = model.compute_ne_profile(latitude, longitude, alt_grid, time, ssn=75.0)

    # Find peak
    peak_idx = np.argmax(ne_profile)
    peak_ne = ne_profile[peak_idx]
    peak_alt = alt_grid[peak_idx]

    print(f"\nPeak electron density: {peak_ne:.2e} el/m³")
    print(f"Peak altitude: {peak_alt:.1f} km")
    print(f"Computed foF2: {9.0 * np.sqrt(peak_ne / 1e12):.2f} MHz")

    # Plot vertical profile
    plt.figure(figsize=(10, 6))
    plt.plot(ne_profile / 1e11, alt_grid, linewidth=2)
    plt.axhline(hmF2, color='r', linestyle='--', label=f'hmF2 = {hmF2:.1f} km')
    plt.xlabel('Electron Density (×10¹¹ el/m³)', fontsize=12)
    plt.ylabel('Altitude (km)', fontsize=12)
    plt.title(f'Chapman Layer Profile\n{latitude}°N, {longitude}°E at {time} UTC', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('/tmp/chapman_layer_profile.png', dpi=150)
    print("\nPlot saved to /tmp/chapman_layer_profile.png")

    # Test 3D grid generation (small grid for demo)
    print("\nGenerating 3D grid...")
    lat_grid_small = np.linspace(-60, 60, 5)
    lon_grid_small = np.linspace(-120, 120, 5)
    alt_grid_small = np.arange(60, 600, 20)

    ne_grid_3d = model.compute_3d_grid(
        lat_grid_small, lon_grid_small, alt_grid_small, time, ssn=75.0
    )

    metrics = model.validate_grid(ne_grid_3d)
    print("\n3D Grid Validation:")
    print(f"  Min Ne: {metrics['min_ne']:.2e} el/m³")
    print(f"  Max Ne: {metrics['max_ne']:.2e} el/m³")
    print(f"  Mean Ne: {metrics['mean_ne']:.2e} el/m³")
    print(f"  Median Ne: {metrics['median_ne']:.2e} el/m³")
    print(f"  Invalid count: {metrics['invalid_count']}")

    print("\n✓ Chapman layer model demonstration complete")


if __name__ == "__main__":
    main()
