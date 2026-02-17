"""
Ionospheric Electron Density Model with Real-Time Correction

Provides electron density (Ne), collision frequency (nu), and refractive index
calculations for HF ray tracing. Supports real-time correction using GIRO
ionosonde data to overcome IRI climatological limitations.

Key Classes:
    IonosphericModel: Main interface for ionospheric parameters
    ChapmanLayer: Single ionospheric layer (E, F1, F2)
    AppletonHartree: Complex refractive index calculator

Physics:
    - Chapman function for layer profiles
    - Appleton-Hartree equation for magnetoionic propagation
    - O-mode and X-mode ray splitting
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List
from datetime import datetime, timezone
import logging

from . import (
    EARTH_RADIUS_KM,
    SPEED_OF_LIGHT,
    ELECTRON_CHARGE,
    ELECTRON_MASS,
    PERMITTIVITY_0,
    PLASMA_FREQ_CONSTANT,
)

logger = logging.getLogger(__name__)


@dataclass
class ChapmanLayer:
    """
    Chapman layer profile for ionospheric electron density.

    The Chapman function models vertical electron density distribution:
        Ne(h) = NmX * exp(0.5 * (1 - z - exp(-z)))
    where:
        z = (h - hmX) / H
        NmX = peak electron density
        hmX = height of peak
        H = scale height

    Attributes:
        name: Layer identifier (E, F1, F2)
        foX: Critical frequency in MHz (related to NmX by fp = 9*sqrt(Ne))
        hmX: Height of peak in km
        scale_height: Scale height H in km
    """
    name: str
    foX: float  # Critical frequency MHz
    hmX: float  # Peak height km
    scale_height: float  # km

    @property
    def NmX(self) -> float:
        """Peak electron density in el/cm³."""
        # foX = 9 * sqrt(NmX) where NmX in el/cm³
        # NmX = (foX / 9)²  in units of 10^6 el/cm³ = 10^12 el/m³
        return (self.foX / PLASMA_FREQ_CONSTANT) ** 2 * 1e6  # el/cm³

    def electron_density(self, altitude_km: float) -> float:
        """
        Calculate electron density at given altitude using Chapman function.

        Args:
            altitude_km: Altitude above Earth's surface in km

        Returns:
            Electron density in el/cm³
        """
        z = (altitude_km - self.hmX) / self.scale_height

        # Chapman function: exp(0.5 * (1 - z - exp(-z)))
        # Avoid overflow for large positive z
        if z > 50:
            return 0.0

        return self.NmX * np.exp(0.5 * (1 - z - np.exp(-z)))


@dataclass
class IonosphericProfile:
    """
    Complete ionospheric profile with multiple layers.

    Attributes:
        time: UTC timestamp of profile
        location: (lat, lon) in degrees
        layers: Dictionary of ChapmanLayer objects
        foF2: F2 layer critical frequency (MHz)
        hmF2: F2 layer peak height (km)
        foF1: F1 layer critical frequency (MHz), may be None
        foE: E layer critical frequency (MHz)
        MUF3000: Maximum usable frequency for 3000km path
    """
    time: datetime
    location: Tuple[float, float]  # (lat, lon)
    layers: Dict[str, ChapmanLayer] = field(default_factory=dict)

    # Key parameters (can be updated from real-time sources)
    foF2: float = 7.0  # MHz
    hmF2: float = 300.0  # km
    foF1: Optional[float] = 4.0  # MHz, daytime only
    foE: float = 2.5  # MHz
    hmF1: float = 200.0  # km
    hmE: float = 110.0  # km
    MUF3000: Optional[float] = None  # MHz

    # Scale heights (typical values)
    H_F2: float = 50.0  # km
    H_F1: float = 30.0  # km
    H_E: float = 10.0  # km

    def __post_init__(self):
        """Initialize layers from parameters."""
        self._build_layers()

    def _build_layers(self):
        """Construct Chapman layers from profile parameters."""
        self.layers['F2'] = ChapmanLayer(
            name='F2',
            foX=self.foF2,
            hmX=self.hmF2,
            scale_height=self.H_F2
        )

        if self.foF1 is not None:
            self.layers['F1'] = ChapmanLayer(
                name='F1',
                foX=self.foF1,
                hmX=self.hmF1,
                scale_height=self.H_F1
            )

        self.layers['E'] = ChapmanLayer(
            name='E',
            foX=self.foE,
            hmX=self.hmE,
            scale_height=self.H_E
        )

    def electron_density(self, altitude_km: float) -> float:
        """
        Calculate total electron density at altitude.

        Sums contributions from all layers.

        Args:
            altitude_km: Altitude in km

        Returns:
            Total electron density in el/cm³
        """
        total = 0.0
        for layer in self.layers.values():
            total += layer.electron_density(altitude_km)
        return total

    def plasma_frequency(self, altitude_km: float) -> float:
        """
        Calculate plasma frequency at altitude.

        fp = 9 * sqrt(Ne) MHz where Ne in el/cm³

        Args:
            altitude_km: Altitude in km

        Returns:
            Plasma frequency in MHz
        """
        Ne = self.electron_density(altitude_km)
        if Ne <= 0:
            return 0.0
        return PLASMA_FREQ_CONSTANT * np.sqrt(Ne / 1e6)  # MHz


class AppletonHartree:
    """
    Appleton-Hartree equation for magnetoionic refractive index.

    Calculates complex refractive index for electromagnetic waves
    propagating through magnetized plasma (the ionosphere).

    The equation accounts for:
    - Electron density (Ne)
    - Collision frequency (nu)
    - Geomagnetic field (B)
    - Wave frequency (f)
    - Propagation angle to B-field (theta)

    Returns separate indices for O-mode and X-mode rays.
    """

    @staticmethod
    def refractive_index(
        Ne: float,  # el/cm³
        nu: float,  # collision frequency Hz
        B: float,   # magnetic field nT
        f: float,   # wave frequency MHz
        theta: float,  # angle to B-field radians
    ) -> Tuple[complex, complex]:
        """
        Calculate O-mode and X-mode refractive indices.

        Appleton-Hartree equation:
            n² = 1 - X / (1 - jZ - Y²sin²θ/(2(1-X-jZ)) ± √(Y⁴sin⁴θ/(4(1-X-jZ)²) + Y²cos²θ))

        Where:
            X = (fp/f)² = ωp²/ω²
            Y = fH/f = ωH/ω (gyrofrequency ratio)
            Z = ν/ω (collision term)

        Args:
            Ne: Electron density in el/cm³
            nu: Collision frequency in Hz
            B: Magnetic field strength in nT
            f: Wave frequency in MHz
            theta: Angle between wave vector and B-field in radians

        Returns:
            Tuple of (n_ordinary, n_extraordinary) as complex numbers
        """
        # Avoid division by zero
        if f <= 0 or Ne < 0:
            return (complex(1, 0), complex(1, 0))

        # Convert units
        f_hz = f * 1e6  # Hz
        omega = 2 * np.pi * f_hz

        # Plasma frequency
        # fp = 9 * sqrt(Ne) MHz where Ne in 10^6 el/cm³
        fp_hz = 9e6 * np.sqrt(Ne / 1e6) if Ne > 0 else 0

        # Gyrofrequency
        # fH = eB/(2πm) = 2.8 * B(μT) MHz = 2.8e-3 * B(nT) MHz
        fH_hz = 2.8e3 * (B / 1e9) * 1e6 if B > 0 else 0

        # Dimensionless parameters
        X = (fp_hz / f_hz) ** 2 if f_hz > 0 else 0
        Y = fH_hz / f_hz if f_hz > 0 else 0
        Z = nu / omega if omega > 0 else 0

        # Trigonometric terms
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin2 = sin_theta ** 2
        cos2 = cos_theta ** 2

        # Appleton-Hartree equation:
        # n² = 1 - X / [1 - jZ - Y²sin²θ/(2(1-X-jZ)) ± √(Y⁴sin⁴θ/(4(1-X-jZ)²) + Y²cos²θ)]
        #
        # For the simplified case (no magnetic field, Y=0):
        # n² = 1 - X / (1 - jZ) ≈ 1 - X  (when Z≈0)

        # Handle simple case (no magnetic field) for numerical stability
        if Y < 1e-6:
            # Unmagnetized plasma: n² = 1 - X / (1 - jZ)
            denom = 1 - 1j * Z
            if abs(denom) < 1e-10:
                return (complex(0, -1), complex(0, -1))
            n2 = 1 - X / denom
            n = np.sqrt(n2 + 0j)
            if n.real < 0:
                n = -n
            return (n, n)  # O and X modes identical without B-field

        # Full magnetoionic case
        # Inner denominator for Y terms: (1 - X - jZ)
        inner_denom = 1 - X - 1j * Z

        if abs(inner_denom) < 1e-10:
            # Near critical frequency
            return (complex(0, -1), complex(0, -1))

        # Y terms
        Y2_sin4 = (Y ** 4) * (sin2 ** 2)
        Y2_cos2 = (Y ** 2) * cos2
        Y2_sin2 = (Y ** 2) * sin2

        # Square root term: √(Y⁴sin⁴θ/(4(1-X-jZ)²) + Y²cos²θ)
        sqrt_term = np.sqrt(
            Y2_sin4 / (4 * inner_denom ** 2) + Y2_cos2 + 0j
        )

        # Y²sin²θ/(2(1-X-jZ))
        Y_term = Y2_sin2 / (2 * inner_denom)

        # Full denominator: 1 - jZ - Y_term ± sqrt_term
        # Note: outer denominator starts with (1 - jZ), not (1 - X - jZ)
        denom_O = (1 - 1j * Z) - Y_term - sqrt_term  # O-mode: - sign gives larger n
        denom_X = (1 - 1j * Z) - Y_term + sqrt_term  # X-mode: + sign gives smaller n

        n2_O = 1 - X / denom_O if abs(denom_O) > 1e-10 else 0j
        n2_X = 1 - X / denom_X if abs(denom_X) > 1e-10 else 0j

        n_O = np.sqrt(n2_O)
        n_X = np.sqrt(n2_X)

        # Ensure positive real part (forward propagation)
        if n_O.real < 0:
            n_O = -n_O
        if n_X.real < 0:
            n_X = -n_X

        return (n_O, n_X)


class IonosphericModel:
    """
    Main interface for ionospheric parameters with real-time updates.

    This class provides:
    1. Electron density profiles (via Chapman layers)
    2. Collision frequency estimates
    3. Refractive index calculation (Appleton-Hartree)
    4. Real-time profile correction from ionosonde data

    The key innovation is update_from_realtime() which applies
    GIRO ionosonde measurements to correct climatological profiles:
        α = foF2_real / foF2_climatology
        Ne_corrected = α² × Ne_climatology

    Example:
        model = IonosphericModel()

        # Get climatological profile
        Ne = model.get_electron_density(lat=40.0, lon=-75.0, alt=300.0)

        # Apply real-time correction
        model.update_from_realtime(foF2=8.5, hmF2=320.0, MUF=24.0)

        # Now get corrected profile
        Ne_corrected = model.get_electron_density(lat=40.0, lon=-75.0, alt=300.0)
    """

    def __init__(self):
        """Initialize with default climatological profile."""
        self.profiles: Dict[Tuple[float, float], IonosphericProfile] = {}
        self._default_profile = self._create_default_profile()
        self._realtime_corrections: Dict[str, float] = {}
        self._last_update: Optional[datetime] = None

        # Geomagnetic field model (simplified - use IGRF for production)
        self._B_equator = 30000.0  # nT at equator

    def _create_default_profile(self) -> IonosphericProfile:
        """Create default mid-latitude daytime profile."""
        return IonosphericProfile(
            time=datetime.now(timezone.utc),
            location=(40.0, -75.0),
            foF2=7.0,
            hmF2=300.0,
            foF1=4.0,
            foE=2.5,
        )

    def get_electron_density(
        self,
        lat: float,
        lon: float,
        alt: float,
        time: Optional[datetime] = None
    ) -> float:
        """
        Get electron density at specified location and altitude.

        Args:
            lat: Latitude in degrees (-90 to 90)
            lon: Longitude in degrees (-180 to 180)
            alt: Altitude above Earth's surface in km
            time: UTC timestamp (default: now)

        Returns:
            Electron density in el/cm³
        """
        profile = self._get_or_create_profile(lat, lon, time)
        return profile.electron_density(alt)

    def get_electron_density_profile(
        self,
        lat: float,
        lon: float,
        altitudes: np.ndarray,
        time: Optional[datetime] = None
    ) -> np.ndarray:
        """
        Get electron density profile over altitude range.

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            altitudes: Array of altitudes in km
            time: UTC timestamp

        Returns:
            Array of electron densities in el/cm³
        """
        profile = self._get_or_create_profile(lat, lon, time)
        return np.array([profile.electron_density(alt) for alt in altitudes])

    def get_collision_frequency(
        self,
        lat: float,
        lon: float,
        alt: float,
        time: Optional[datetime] = None
    ) -> float:
        """
        Estimate electron-neutral collision frequency.

        Uses exponential decrease with altitude:
            nu(h) = nu_0 * exp(-h/H_nu)

        Typical values:
            nu_0 = 10^7 Hz at ground
            H_nu = 5-10 km scale height

        At HF-relevant altitudes (100-400 km):
            - 100 km: ~10^5 Hz
            - 200 km: ~10^3 Hz
            - 300 km: ~10^1 Hz

        Args:
            lat: Latitude degrees
            lon: Longitude degrees
            alt: Altitude km
            time: UTC timestamp

        Returns:
            Collision frequency in Hz
        """
        # Simple exponential model
        nu_100km = 1e5  # Hz at 100 km
        H_nu = 25.0  # scale height km

        nu = nu_100km * np.exp(-(alt - 100.0) / H_nu)
        return max(nu, 1.0)  # Minimum 1 Hz

    def get_magnetic_field(
        self,
        lat: float,
        lon: float,
        alt: float
    ) -> Tuple[float, float, float]:
        """
        Get geomagnetic field components.

        Simplified dipole model. For production use, integrate IGRF.

        Args:
            lat: Latitude degrees
            lon: Longitude degrees
            alt: Altitude km

        Returns:
            (B_north, B_east, B_down) in nT
        """
        # Simplified dipole model
        # B varies with latitude: B = B_eq * sqrt(1 + 3*sin²(lat))
        lat_rad = np.radians(lat)

        # Altitude scaling: B decreases as (R/(R+h))³
        r_ratio = EARTH_RADIUS_KM / (EARTH_RADIUS_KM + alt)

        B_total = self._B_equator * np.sqrt(1 + 3 * np.sin(lat_rad) ** 2) * (r_ratio ** 3)

        # Inclination
        I = np.arctan(2 * np.tan(lat_rad))

        B_horizontal = B_total * np.cos(I)
        B_down = B_total * np.sin(I)

        # Assume field points north (simplified)
        return (B_horizontal, 0.0, B_down)

    def get_refractive_index(
        self,
        lat: float,
        lon: float,
        alt: float,
        freq_mhz: float,
        wave_direction: Tuple[float, float, float],
        time: Optional[datetime] = None
    ) -> Tuple[complex, complex]:
        """
        Calculate O-mode and X-mode refractive indices.

        Uses Appleton-Hartree equation with local ionospheric and
        geomagnetic parameters.

        Args:
            lat: Latitude degrees
            lon: Longitude degrees
            alt: Altitude km
            freq_mhz: Wave frequency in MHz
            wave_direction: Unit vector (kx, ky, kz) in local ENU frame
            time: UTC timestamp

        Returns:
            (n_O, n_X) complex refractive indices
        """
        Ne = self.get_electron_density(lat, lon, alt, time)
        nu = self.get_collision_frequency(lat, lon, alt, time)
        B_n, B_e, B_d = self.get_magnetic_field(lat, lon, alt)

        B_total = np.sqrt(B_n**2 + B_e**2 + B_d**2)

        # Calculate angle between wave vector and B-field
        B_vec = np.array([B_n, B_e, -B_d])  # ENU frame, down is -up
        k_vec = np.array(wave_direction)

        B_mag = np.linalg.norm(B_vec)
        k_mag = np.linalg.norm(k_vec)

        if B_mag > 0 and k_mag > 0:
            cos_theta = np.dot(B_vec, k_vec) / (B_mag * k_mag)
            theta = np.arccos(np.clip(cos_theta, -1, 1))
        else:
            theta = 0.0

        return AppletonHartree.refractive_index(Ne, nu, B_total, freq_mhz, theta)

    def update_from_realtime(
        self,
        foF2: Optional[float] = None,
        hmF2: Optional[float] = None,
        MUF: Optional[float] = None,
        station_lat: Optional[float] = None,
        station_lon: Optional[float] = None,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Apply real-time ionosonde corrections to model.

        This is the key method for exceeding climatological predictions.
        Real-time ionosonde data from GIRO network provides actual foF2/hmF2
        values which are used to scale the IRI-based profile.

        Correction algorithm:
            α = foF2_real / foF2_model
            Ne_corrected = α² × Ne_model

        Args:
            foF2: Real-time F2 critical frequency (MHz)
            hmF2: Real-time F2 peak height (km)
            MUF: Real-time MUF(3000) for validation
            station_lat: Ionosonde station latitude
            station_lon: Ionosonde station longitude
            timestamp: Measurement timestamp
        """
        if foF2 is not None:
            self._realtime_corrections['foF2'] = foF2

            # Update default profile
            old_foF2 = self._default_profile.foF2
            alpha = foF2 / old_foF2 if old_foF2 > 0 else 1.0

            self._default_profile.foF2 = foF2
            self._default_profile._build_layers()

            logger.info(
                f"Applied foF2 correction: {old_foF2:.2f} → {foF2:.2f} MHz "
                f"(α = {alpha:.3f}, Ne scale = {alpha**2:.3f})"
            )

        if hmF2 is not None:
            self._realtime_corrections['hmF2'] = hmF2
            self._default_profile.hmF2 = hmF2
            self._default_profile._build_layers()
            logger.info(f"Applied hmF2 correction: {hmF2:.1f} km")

        if MUF is not None:
            self._realtime_corrections['MUF'] = MUF
            self._default_profile.MUF3000 = MUF

        self._last_update = timestamp or datetime.now(timezone.utc)

    def get_correction_status(self) -> Dict:
        """
        Get current real-time correction status.

        Returns:
            Dictionary with correction parameters and age
        """
        age = None
        if self._last_update:
            age = (datetime.now(timezone.utc) - self._last_update).total_seconds()

        return {
            'corrections': self._realtime_corrections.copy(),
            'last_update': self._last_update,
            'age_seconds': age,
            'current_foF2': self._default_profile.foF2,
            'current_hmF2': self._default_profile.hmF2,
        }

    def _get_or_create_profile(
        self,
        lat: float,
        lon: float,
        time: Optional[datetime]
    ) -> IonosphericProfile:
        """Get profile for location, creating if needed."""
        # For now, return default profile with corrections applied
        # Future: implement spatial interpolation between ionosonde stations
        return self._default_profile


def create_test_profile() -> IonosphericModel:
    """
    Create a test ionospheric model for development.

    Returns:
        IonosphericModel with typical mid-latitude daytime conditions
    """
    model = IonosphericModel()

    # Simulate receiving real-time ionosonde data
    model.update_from_realtime(
        foF2=8.5,
        hmF2=320.0,
        MUF=25.0,
    )

    return model


if __name__ == "__main__":
    # Quick test
    model = create_test_profile()

    print("IonosphericModel Test")
    print("=" * 40)

    # Test electron density profile
    altitudes = np.arange(100, 500, 50)
    print("\nElectron Density Profile (lat=40, lon=-75):")
    print(f"{'Alt (km)':<10} {'Ne (el/cm³)':<15} {'fp (MHz)':<10}")
    print("-" * 35)

    for alt in altitudes:
        Ne = model.get_electron_density(40.0, -75.0, alt)
        fp = PLASMA_FREQ_CONSTANT * np.sqrt(Ne / 1e6) if Ne > 0 else 0
        print(f"{alt:<10.0f} {Ne:<15.2e} {fp:<10.3f}")

    # Test refractive index
    print("\nRefractive Index at 300 km, 10 MHz:")
    n_O, n_X = model.get_refractive_index(40.0, -75.0, 300.0, 10.0, (0, 0, 1))
    print(f"  O-mode: n = {n_O.real:.4f} + {n_O.imag:.4f}j")
    print(f"  X-mode: n = {n_X.real:.4f} + {n_X.imag:.4f}j")

    # Test correction status
    print("\nCorrection Status:")
    status = model.get_correction_status()
    for key, val in status.items():
        print(f"  {key}: {val}")
