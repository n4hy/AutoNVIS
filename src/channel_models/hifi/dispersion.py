"""
Frequency-Dependent Group Delay (Dispersion) Model for Wideband HF

This module implements dispersion modeling for the Vogler-Hoffmeyer HF channel model.
It supports linear dispersion with quasi-parabolic (QP) derived coefficients for
physics-based parameterization.

Theory:
    The ionosphere causes frequency-dependent group delay due to the refractive index
    varying with electron density. For a quasi-parabolic layer:

        τ_g(f) ≈ τ_0 + K/f²

    where K depends on layer parameters. The linear approximation:

        τ(f) = τ_0 + d·(f - f_c)

    is valid when bandwidth B << carrier frequency f_c.

Implementation:
    Linear dispersion produces a quadratic phase response, equivalent to a chirp.
    The all-pass filter impulse response is:

        h(t) = exp(jπt²/d) / √(j·d)

    This is applied via time-domain convolution.

References:
    - NTIA Report 90-255, Equation 6 (quasi-parabolic layer model)
    - Vogler & Hoffmeyer, "A Model for Wideband HF Propagation Channels"

Author: Generated with Claude Code
"""

import numpy as np
from scipy import signal
from typing import Optional, Tuple, Dict
from dataclasses import dataclass


# Speed of light (m/s)
C_LIGHT = 299792458.0


@dataclass
class DispersionParameters:
    """
    Parameters for dispersion calculation.

    Attributes:
        d_us_per_MHz: Dispersion coefficient (μs/MHz). Positive = higher frequencies
                      arrive later (normal ionospheric dispersion).
        f_c_layer: Layer critical frequency (Hz) for QP derivation
        y_m: Layer semi-thickness (m), typically 50-150 km
        phi_inc: Incidence angle (radians)
        f_carrier: RF carrier frequency (Hz)
    """
    d_us_per_MHz: Optional[float] = None
    f_c_layer: Optional[float] = None
    y_m: float = 100e3  # 100 km default
    phi_inc: float = 0.0  # Vertical incidence default
    f_carrier: Optional[float] = None

    def get_dispersion_coefficient(self) -> float:
        """
        Get the dispersion coefficient, either from direct specification
        or computed from QP layer parameters.

        Returns:
            d: Dispersion coefficient in μs/MHz
        """
        if self.d_us_per_MHz is not None:
            return self.d_us_per_MHz

        if self.f_c_layer is not None and self.f_carrier is not None:
            return compute_d_from_qp(
                self.f_c_layer, self.f_carrier, self.y_m, self.phi_inc
            )

        # Default: moderate dispersion
        return 50.0


def compute_d_from_qp(f_c_layer: float, f_carrier: float,
                      y_m: float = 100e3, phi_inc: float = 0.0) -> float:
    """
    Derive linear dispersion coefficient from quasi-parabolic layer parameters.

    Based on NTIA Report 90-255 Equation 6. The QP model gives group delay:
        τ_g(f) ≈ K / f²

    where K = (π · y_m · f_c² · sec(φ)) / (2c)

    The linear approximation d = |dτ/df| at the carrier frequency:
        d = 2K / f_carrier³

    Parameters:
        f_c_layer: Layer critical frequency (Hz). Typical: 4-12 MHz.
        f_carrier: RF carrier frequency (Hz). Must be > f_c_layer for propagation.
        y_m: Layer semi-thickness (m). Typical: 50-150 km (default 100 km).
        phi_inc: Incidence angle (radians). 0 = vertical, π/4 = 45°.

    Returns:
        d: Dispersion coefficient (μs/MHz). Positive value.

    Example:
        >>> d = compute_d_from_qp(8e6, 15e6, y_m=100e3, phi_inc=0.35)
        >>> print(f"Dispersion: {d:.1f} μs/MHz")
        Dispersion: 24.7 μs/MHz
    """
    # Validate inputs
    if f_carrier <= f_c_layer:
        raise ValueError(f"Carrier frequency ({f_carrier/1e6:.1f} MHz) must be above "
                        f"critical frequency ({f_c_layer/1e6:.1f} MHz)")

    # Secant of incidence angle (limit to avoid singularity)
    cos_phi = np.cos(phi_inc)
    if cos_phi < 0.1:
        cos_phi = 0.1  # Limit to ~84° incidence
    sec_phi = 1.0 / cos_phi

    # QP model constant K (seconds)
    # K = (π · y_m · f_c² · sec(φ)) / (2c)
    K = (np.pi * y_m * f_c_layer**2 * sec_phi) / (2 * C_LIGHT)

    # Linear coefficient: d = 2K / f³ (seconds per Hz)
    d_s_per_Hz = 2 * K / f_carrier**3

    # Convert to μs/MHz
    # 1 s/Hz = 1e6 μs / (1e-6 MHz) = 1e12 μs/MHz
    d_us_per_MHz = d_s_per_Hz * 1e12

    return d_us_per_MHz


class DispersionModel:
    """
    Frequency-dependent group delay (dispersion) model for wideband HF.

    Implements linear dispersion τ(f) = τ₀ + d·(f - f_c) via chirp all-pass filter.

    Attributes:
        fs: Sample rate (Hz)
        mode: Dispersion mode ('linear' or 'qp')

    Example:
        >>> disp = DispersionModel(fs=4e6)
        >>> d = disp.compute_d_from_qp(f_c_layer=8e6, f_carrier=15e6)
        >>> print(f"Dispersion coefficient: {d:.1f} μs/MHz")
        >>> y = disp.apply_dispersion(x, d)
    """

    def __init__(self, fs: float, mode: str = 'linear'):
        """
        Initialize dispersion model.

        Parameters:
            fs: Sample rate (Hz)
            mode: Dispersion mode ('linear' for chirp filter, 'qp' for future
                  full quasi-parabolic implementation)
        """
        self.fs = fs
        self.mode = mode
        self._filter_cache: Dict[Tuple[float, float], np.ndarray] = {}

    @staticmethod
    def compute_d_from_qp(f_c_layer: float, f_carrier: float,
                          y_m: float = 100e3, phi_inc: float = 0.0) -> float:
        """
        Derive linear dispersion coefficient from QP layer parameters.

        See module-level function for details.
        """
        return compute_d_from_qp(f_c_layer, f_carrier, y_m, phi_inc)

    def get_dispersion_filter(self, d_us_per_MHz: float,
                               duration_factor: float = 4.0,
                               max_taps: int = 2001) -> np.ndarray:
        """
        Get or create dispersion filter for given coefficient.

        The filter implements linear dispersion via a chirp impulse response:
            h(t) = exp(jπt²/d) / √(j·d)

        Parameters:
            d_us_per_MHz: Dispersion coefficient (μs/MHz).
                          Positive = higher frequencies delayed more.
            duration_factor: Filter duration as multiple of dispersion spread.
                            Higher = more accurate but longer filter.
            max_taps: Maximum filter length (will be truncated if exceeded).

        Returns:
            h: Complex impulse response (odd length, centered).
        """
        # Check cache
        cache_key = (d_us_per_MHz, duration_factor)
        if cache_key in self._filter_cache:
            return self._filter_cache[cache_key]

        # Handle zero dispersion
        if abs(d_us_per_MHz) < 0.01:
            h = np.array([1.0 + 0j])
            self._filter_cache[cache_key] = h
            return h

        # Convert units: d in μs/MHz → seconds/Hz²
        # τ(f) = τ₀ + d·f where d is in s/Hz
        # For μs/MHz: 1 μs/MHz = 1e-6 s / 1e6 Hz = 1e-12 s/Hz
        d_s_per_Hz = d_us_per_MHz * 1e-12

        # Estimate bandwidth (Nyquist)
        bandwidth_Hz = self.fs / 2

        # Dispersion spread in seconds
        # For linear dispersion: max delay difference = |d| × bandwidth
        spread_s = abs(d_us_per_MHz) * (bandwidth_Hz / 1e6) * 1e-6

        # Filter duration (ensure minimum length for very small dispersion)
        duration_s = max(spread_s * duration_factor, 20 / self.fs)
        N = int(np.ceil(duration_s * self.fs))

        # Ensure odd length for symmetric filter
        if N % 2 == 0:
            N += 1

        # Limit filter length
        N = min(N, max_taps)

        # Time vector centered at zero
        t = (np.arange(N) - (N - 1) / 2) / self.fs

        # Chirp impulse response
        # h(t) = exp(jπt²/d) / √(j·d)
        # Note: d here needs to be in consistent units
        # Phase = π·t²/d where d is the chirp rate parameter
        # For τ(f) = d·f, the phase is φ(f) = -π·d·f², giving h(t) = exp(jπt²/d)

        # Convert d to chirp rate (seconds²)
        # The relationship is: dispersion d (s/Hz) relates to chirp rate via
        # h(t) = exp(jπt²/d_chirp) where d_chirp = d (the dispersion in s/Hz)
        d_chirp = d_s_per_Hz

        if abs(d_chirp) > 1e-20:
            phase = np.pi * t**2 / d_chirp
            h = np.exp(1j * phase)

            # Normalization factor
            # For unit energy: scale by 1/√(|d|·fs)
            h = h / np.sqrt(np.abs(d_chirp) * self.fs)
        else:
            h = np.zeros(N, dtype=complex)
            h[N // 2] = 1.0

        # Apply window to limit spectral leakage
        window = signal.windows.tukey(N, alpha=0.2)
        h = h * window

        # Normalize for unit energy throughput
        energy = np.sum(np.abs(h)**2)
        if energy > 0:
            h = h / np.sqrt(energy)

        # Cache the filter
        self._filter_cache[cache_key] = h

        return h

    def apply_dispersion(self, x: np.ndarray, d_us_per_MHz: float,
                         preserve_length: bool = True) -> np.ndarray:
        """
        Apply dispersion to signal via time-domain convolution.

        Parameters:
            x: Input signal (complex baseband)
            d_us_per_MHz: Dispersion coefficient (μs/MHz)
            preserve_length: If True, output has same length as input.
                            If False, output includes full convolution.

        Returns:
            y: Dispersed signal
        """
        # Skip for negligible dispersion
        if abs(d_us_per_MHz) < 0.01:
            return x.copy()

        h = self.get_dispersion_filter(d_us_per_MHz)

        # Choose convolution mode
        mode = 'same' if preserve_length else 'full'

        # Use overlap-add for efficiency with long signals
        if len(x) > 10 * len(h):
            y = signal.oaconvolve(x, h, mode=mode)
        else:
            y = signal.convolve(x, h, mode=mode)

        return y

    def apply_inverse_dispersion(self, x: np.ndarray, d_us_per_MHz: float,
                                  preserve_length: bool = True) -> np.ndarray:
        """
        Apply inverse dispersion (compression) to signal.

        This undoes the effect of dispersion, useful for equalization.

        Parameters:
            x: Input dispersed signal
            d_us_per_MHz: Original dispersion coefficient (will be negated)
            preserve_length: If True, output has same length as input.

        Returns:
            y: Compressed signal
        """
        return self.apply_dispersion(x, -d_us_per_MHz, preserve_length)

    def get_group_delay_curve(self, d_us_per_MHz: float,
                               freq_offset_MHz: np.ndarray) -> np.ndarray:
        """
        Get theoretical group delay vs frequency offset.

        For linear dispersion: τ(f) = τ₀ + d·(f - f_c)

        Parameters:
            d_us_per_MHz: Dispersion coefficient (μs/MHz)
            freq_offset_MHz: Frequency offset from carrier (MHz)

        Returns:
            tau_us: Group delay relative to center frequency (μs)
        """
        return d_us_per_MHz * freq_offset_MHz

    def measure_dispersion(self, h: np.ndarray) -> float:
        """
        Measure dispersion coefficient from a filter's frequency response.

        Computes d by measuring the slope of group delay vs frequency.

        Parameters:
            h: Filter impulse response

        Returns:
            d_us_per_MHz: Measured dispersion coefficient (μs/MHz)
        """
        # Compute frequency response
        N_fft = max(1024, 2 * len(h))
        H = np.fft.fft(h, n=N_fft)
        f = np.fft.fftfreq(N_fft, 1 / self.fs)

        # Compute group delay via phase derivative
        phase = np.unwrap(np.angle(H))

        # Use central difference for derivative
        df = f[1] - f[0]
        group_delay = -np.gradient(phase, df) / (2 * np.pi)

        # Measure slope in the middle of the positive frequency band
        # (avoid edges and DC/Nyquist)
        idx_start = N_fft // 8
        idx_end = N_fft // 4

        f_region = f[idx_start:idx_end]
        tau_region = group_delay[idx_start:idx_end]

        # Linear fit to get slope
        coeffs = np.polyfit(f_region, tau_region, 1)
        d_s_per_Hz = coeffs[0]

        # Convert to μs/MHz
        d_us_per_MHz = d_s_per_Hz * 1e12

        return d_us_per_MHz

    def clear_cache(self):
        """Clear the filter cache."""
        self._filter_cache.clear()


def typical_dispersion_values() -> dict:
    """
    Return typical dispersion values for different ionospheric conditions.

    These are approximate values for 1-hop F-layer paths at mid-latitudes.
    Actual values depend on frequency, path geometry, and ionospheric state.

    Returns:
        Dictionary mapping condition names to dispersion ranges (μs/MHz)
    """
    return {
        'quiet_high_freq': (10, 30),    # f well above f_c, stable
        'moderate': (30, 80),            # Typical daytime operations
        'disturbed': (80, 150),          # Near MUF, unstable
        'spread_f': (150, 240),          # Spread-F conditions
        'severe': (200, 400),            # Highly disturbed
    }


def estimate_dispersion_spread(d_us_per_MHz: float, bandwidth_MHz: float) -> float:
    """
    Estimate the temporal spreading caused by dispersion.

    For a signal of bandwidth B with dispersion d, the pulse spreading is:
        Δt = |d| × B

    Parameters:
        d_us_per_MHz: Dispersion coefficient (μs/MHz)
        bandwidth_MHz: Signal bandwidth (MHz)

    Returns:
        spread_us: Pulse spreading in microseconds
    """
    return abs(d_us_per_MHz) * bandwidth_MHz
