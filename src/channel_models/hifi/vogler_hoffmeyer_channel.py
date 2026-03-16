"""
Vogler-Hoffmeyer HF Channel Model Implementation

Based on NTIA Report 90-255: "A Model for Wideband HF Propagation Channels"
by L.E. Vogler and J.A. Hoffmeyer (1990)

This module implements a stochastic HF (High Frequency) channel model that simulates
time-varying distortion of transmitted signals due to:
    - Dispersion: Different frequency components reflecting at different ionospheric heights
    - Scattering: From ionospheric irregularities (including spread-F conditions)
    - Doppler spread: Frequency spreading due to ionospheric motion
    - Doppler shift: Systematic frequency offset from ionospheric movement

The model extends the narrowband Watterson model to wideband channels (up to 1 MHz+)
and supports both Gaussian and exponential Doppler spectrum shapes.

References:
    - NTIA Report 90-255, Part II (Stochastic model)
    - CCIR Report 549 (Watterson model baseline)

Author: Generated with Claude Code
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Literal
from enum import Enum
import json

# Import dispersion model for frequency-dependent group delay
from .dispersion import DispersionModel, compute_d_from_qp

# Try to import Numba for JIT compilation
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Create dummy decorator
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


# Numba-optimized inner processing loop
@jit(nopython=True, parallel=False, cache=True)
def _process_samples_numba(input_samples, buffer, delay_samples, T,
                           C_state, rho, innovation_coeff,
                           delay_us, tau_c, f_s, b, t_start, delta_t,
                           rng_state, direct_coeff, scatter_coeff,
                           spread_f_enabled):
    """
    Numba-optimized inner loop for channel processing.

    Processes all samples through the tapped delay line with time-varying
    fading coefficients using AR(1) process for Gaussian correlation.

    Supports Rician fading via direct_coeff and scatter_coeff:
    - direct_coeff = sqrt(K/(K+1)) for direct path
    - scatter_coeff = sqrt(1/(K+1)) for scattered path
    - For Rayleigh: direct_coeff=0, scatter_coeff=1

    Supports spread-F simulation via spread_f_enabled:
    - When True, multiplies each tap gain by random factor in [0.1, 1.0]
    """
    num_input = len(input_samples)
    num_taps = len(delay_samples)
    output = np.zeros(num_input, dtype=np.complex128)

    # Copy state to avoid modifying input
    C = C_state.copy()
    buf = buffer.copy()
    buf_len = len(buf)

    # Pre-compute 2*pi
    two_pi = 2.0 * np.pi

    for n in range(num_input):
        # Current time
        t = t_start + n * delta_t

        # Update delay line buffer (shift right, insert new sample at front)
        for i in range(buf_len - 1, 0, -1):
            buf[i] = buf[i-1]
        buf[0] = input_samples[n]

        # Generate complex Gaussian noise for AR(1) update
        # Use Box-Muller transform with simple LCG
        z = np.zeros(num_taps, dtype=np.complex128)
        for k in range(num_taps):
            # Simple LCG random number generator
            rng_state[0] = (rng_state[0] * 1103515245 + 12345) & 0x7fffffff
            # Map to (0, 1) exclusive: adding 0.5 prevents 0, dividing by 2^31 prevents 1
            u1 = (rng_state[0] + 0.5) / 2147483648.0
            rng_state[0] = (rng_state[0] * 1103515245 + 12345) & 0x7fffffff
            u2 = rng_state[0] / 2147483648.0  # Use same divisor for consistency

            # Box-Muller transform
            mag = np.sqrt(-2.0 * np.log(u1))
            theta = two_pi * u2
            z[k] = (mag * np.cos(theta) + 1j * mag * np.sin(theta)) / np.sqrt(2.0)

        # AR(1) update: C[n] = rho * C[n-1] + sqrt(1-rho^2) * z[n]
        C = rho * C + innovation_coeff * z

        # Compute phase for each tap: phi = 2*pi * (f_s + b*(tau_c - tau)) * t
        # Compute tap gains and accumulate output
        for k in range(num_taps):
            delay = delay_samples[k]
            if delay < buf_len:
                # Effective Doppler for this tap
                f_eff = f_s + b * (tau_c - delay_us[k])
                phi = two_pi * f_eff * t

                # Apply Rician fading: direct path only on first tap (LOS)
                # Other taps are scattered paths only
                if k == 0:
                    fading_gain = direct_coeff + scatter_coeff * C[k]
                else:
                    fading_gain = scatter_coeff * C[k]

                # Tap gain: T(tau) * fading_gain * exp(j*phi)
                tap_gain = T[k] * fading_gain * (np.cos(phi) + 1j * np.sin(phi))

                # Apply spread-F random amplitude multiplication
                if spread_f_enabled:
                    # Generate uniform random in [0.1, 1.0]
                    rng_state[0] = (rng_state[0] * 1103515245 + 12345) & 0x7fffffff
                    spread_factor = 0.1 + 0.9 * (rng_state[0] / 2147483648.0)
                    tap_gain = tap_gain * spread_factor

                output[n] += buf[delay] * tap_gain

    return output, buf, C, rng_state


# Numba-optimized inner processing loop for exponential correlation
@jit(nopython=True, parallel=False, cache=True)
def _process_samples_numba_exp(input_samples, buffer, delay_samples, T,
                               x_real_state, x_imag_state, lambda_param,
                               delay_us, tau_c, f_s, b, t_start, delta_t,
                               rng_state, direct_coeff, scatter_coeff,
                               spread_f_enabled):
    """
    Numba-optimized inner loop for channel processing with exponential correlation.

    Processes all samples through the tapped delay line with time-varying
    fading coefficients using exponentially correlated random process.

    The exponential correlation formula (Equation 12):
        x_n = u_n + (x_{n-1} - u_n) * lambda
        lambda = exp[-delta_t * sigma_f]

    This produces autocorrelation (Equation 13):
        rho(m) = exp[-m * delta_t * sigma_f]

    IMPORTANT: Each tap has INDEPENDENT exponentially correlated fading.
    This creates:
    - Temporal correlation within each tap (Lorentzian Doppler spectrum)
    - Spatial independence between taps (frequency-selective fading)

    The paper specifies temporal correlation for the fading process at each
    delay tap. All taps should fade independently with the same temporal
    statistics, not fade together as a single correlated process.

    The fading values are in range [-0.5, 0.5], giving complex fading with
    variance ~1/6 per tap. The channel normalization factor accounts for
    this to achieve proper power normalization.

    Supports Rician fading via direct_coeff and scatter_coeff.

    Supports spread-F simulation via spread_f_enabled:
    - When True, multiplies each tap gain by random factor in [0.1, 1.0]
    """
    num_input = len(input_samples)
    num_taps = len(delay_samples)
    output = np.zeros(num_input, dtype=np.complex128)

    # Copy state arrays - one state per tap for independent fading
    x_real = x_real_state.copy()
    x_imag = x_imag_state.copy()
    buf = buffer.copy()
    buf_len = len(buf)

    # Pre-compute constants
    two_pi = 2.0 * np.pi
    sqrt2 = np.sqrt(2.0)

    for n in range(num_input):
        # Current time
        t = t_start + n * delta_t

        # Update delay line buffer (shift right, insert new sample at front)
        for i in range(buf_len - 1, 0, -1):
            buf[i] = buf[i-1]
        buf[0] = input_samples[n]

        # Generate exponentially correlated fading for each tap INDEPENDENTLY
        # Each tap has its own AR(1) exponential process
        C = np.zeros(num_taps, dtype=np.complex128)

        for k in range(num_taps):
            # Generate independent uniform random for this tap's real part
            rng_state[0] = (rng_state[0] * 1103515245 + 12345) & 0x7fffffff
            u_real = (rng_state[0] / 2147483648.0) - 0.5  # Map to ~(-0.5, 0.5)

            # Generate independent uniform random for this tap's imag part
            rng_state[0] = (rng_state[0] * 1103515245 + 12345) & 0x7fffffff
            u_imag = (rng_state[0] / 2147483648.0) - 0.5  # Map to ~(-0.5, 0.5)

            # AR(1) update for this tap (independent of other taps)
            # x_n = u_n + (x_{n-1} - u_n) * lambda
            x_real[k] = u_real + (x_real[k] - u_real) * lambda_param
            x_imag[k] = u_imag + (x_imag[k] - u_imag) * lambda_param

            # Combine into complex fading coefficient with unit variance
            # Uniform[-0.5, 0.5] has variance 1/12, so scale by sqrt(12) for unit variance per dimension
            # Combined complex: (sqrt(12)*x_real + j*sqrt(12)*x_imag) / sqrt(2) has E[|C|^2] = 1
            sqrt12 = 3.4641016151377544  # sqrt(12)
            C[k] = sqrt12 * (x_real[k] + 1j * x_imag[k]) / sqrt2

        # Compute phase for each tap: phi = 2*pi * (f_s + b*(tau_c - tau)) * t
        # Compute tap gains and accumulate output
        for k in range(num_taps):
            delay = delay_samples[k]
            if delay < buf_len:
                # Effective Doppler for this tap
                f_eff = f_s + b * (tau_c - delay_us[k])
                phi = two_pi * f_eff * t

                # Apply Rician fading: direct path only on first tap (LOS)
                # Other taps are scattered paths only
                if k == 0:
                    fading_gain = direct_coeff + scatter_coeff * C[k]
                else:
                    fading_gain = scatter_coeff * C[k]

                # Tap gain: T(tau) * fading_gain * exp(j*phi)
                tap_gain = T[k] * fading_gain * (np.cos(phi) + 1j * np.sin(phi))

                # Apply spread-F random amplitude multiplication
                if spread_f_enabled:
                    # Generate uniform random in [0.1, 1.0]
                    rng_state[0] = (rng_state[0] * 1103515245 + 12345) & 0x7fffffff
                    spread_factor = 0.1 + 0.9 * (rng_state[0] / 2147483648.0)
                    tap_gain = tap_gain * spread_factor

                output[n] += buf[delay] * tap_gain

    return output, buf, x_real, x_imag, rng_state


class CorrelationType(Enum):
    """Doppler spectrum correlation type (Paper Section 2.3)"""
    GAUSSIAN = "gaussian"      # Bell-shaped Doppler spectrum - Eq. (7), (8)
    EXPONENTIAL = "exponential"  # Peaked/Lorentzian Doppler spectrum - Eq. (10), (11)


@dataclass
class ModeParameters:
    """
    Parameters for a single propagation mode (E-layer, F-layer low-ray, or F-layer high-ray)

    All parameter names and descriptions reference the Vogler-Hoffmeyer paper (NTIA 90-255).

    Attributes:
        name: Descriptive name for this mode (e.g., "F-layer low-ray")

        # Amplitude parameters
        amplitude: A - Relative mode amplitude [0, 1] (Eq. 3)
        floor_amplitude: A_fl - Receiver threshold/floor level (Eq. 4d, 5d)

        # Delay parameters (microseconds)
        tau_L: Minimum delay time (tau_L in paper)
        sigma_tau: Total delay spread = tau_U - tau_L (Eq. 3)
        sigma_c: Carrier delay subinterval = tau_c - tau_L (Eq. 3)

        # Doppler parameters (Hz)
        sigma_D: Half-width of Doppler spread at floor level (Eq. 7, 10)
        doppler_shift: f_s - Doppler shift at carrier delay (Eq. 15)
        doppler_shift_min_delay: f_sL - Doppler shift at minimum delay (Eq. 15b)

        # Correlation type
        correlation_type: Gaussian or Exponential Doppler spectrum shape
    """
    name: str = "F-layer"

    # Amplitude parameters (dimensionless)
    amplitude: float = 1.0          # A: mode amplitude [0, 1]
    floor_amplitude: float = 0.01   # A_fl: receiver floor/threshold

    # Delay parameters (microseconds) - Paper Table 1
    tau_L: float = 0.0              # Minimum delay time (us)
    sigma_tau: float = 100.0        # Total delay spread (us)
    sigma_c: float = 50.0           # Carrier delay subinterval (us)

    # Doppler parameters (Hz) - Paper Table 1
    sigma_D: float = 1.0            # Doppler spread half-width (Hz)
    doppler_shift: float = 0.0      # f_s: Doppler shift at carrier delay (Hz)
    doppler_shift_min_delay: float = 0.0  # f_sL: Doppler shift at min delay (Hz)

    # Correlation type
    correlation_type: CorrelationType = CorrelationType.GAUSSIAN

    # Dispersion parameters (optional, for wideband frequency-dependent delay)
    # If dispersion_us_per_MHz is set, it overrides QP-derived value
    # Otherwise, dispersion is computed from f_c_layer, y_m, phi_inc
    dispersion_us_per_MHz: Optional[float] = None  # Manual dispersion override (μs/MHz)
    f_c_layer: Optional[float] = None              # Critical frequency (Hz) for QP derivation
    y_m: float = 100e3                             # Layer semi-thickness (m), default 100 km
    phi_inc: float = 0.0                           # Incidence angle (radians)

    @property
    def tau_c(self) -> float:
        """tau_c: Delay time at carrier frequency = tau_L + sigma_c (us)"""
        return self.tau_L + self.sigma_c

    @property
    def tau_U(self) -> float:
        """tau_U: Maximum delay time = tau_L + sigma_tau (us)"""
        return self.tau_L + self.sigma_tau

    @property
    def doppler_delay_coupling(self) -> float:
        """
        b: Delay-Doppler coupling coefficient (Hz/us)

        From Equation (15b):
            b = (f_sL - f_s) / (tau_c - tau_L)
            b = (f_sL - f_s) / sigma_c

        This determines the slope of ridges in the scattering function.
        """
        if self.sigma_c == 0:
            return 0.0
        return (self.doppler_shift_min_delay - self.doppler_shift) / self.sigma_c


@dataclass
class ChannelConfig:
    """
    Complete channel model configuration

    Attributes:
        sample_rate: Sample rate in Hz
        modes: List of propagation mode parameters
        spread_f_enabled: Enable spread-F random multiplication (Paper Section 3)
        random_seed: Optional seed for reproducibility
        k_factor: Rician K-factor for fading control (None=Rayleigh, 0=Rayleigh, >0=Rician)
                  K = (direct path power) / (scattered path power)
                  K=0: Pure Rayleigh (deep fades possible)
                  K=1: Equal direct and scattered (moderate fading)
                  K=10: Strong direct path (mild fading)
                  K=inf: No fading (static channel)
    """
    sample_rate: float = 1e6        # Hz (default 1 MHz for wideband)
    modes: List[ModeParameters] = field(default_factory=lambda: [ModeParameters()])
    spread_f_enabled: bool = False  # Enable spread-F simulation
    random_seed: Optional[int] = None
    k_factor: Optional[float] = None  # Rician K-factor (None or 0 = Rayleigh)

    # Dispersion settings
    dispersion_enabled: bool = False  # Enable frequency-dependent group delay
    carrier_frequency: float = 10e6   # RF carrier frequency (Hz) for dispersion calculation

    def to_dict(self) -> dict:
        """Convert configuration to dictionary for JSON serialization"""
        return {
            'sample_rate': self.sample_rate,
            'spread_f_enabled': self.spread_f_enabled,
            'random_seed': self.random_seed,
            'k_factor': self.k_factor,
            'dispersion_enabled': self.dispersion_enabled,
            'carrier_frequency': self.carrier_frequency,
            'modes': [
                {
                    'name': m.name,
                    'amplitude': m.amplitude,
                    'floor_amplitude': m.floor_amplitude,
                    'tau_L': m.tau_L,
                    'sigma_tau': m.sigma_tau,
                    'sigma_c': m.sigma_c,
                    'sigma_D': m.sigma_D,
                    'doppler_shift': m.doppler_shift,
                    'doppler_shift_min_delay': m.doppler_shift_min_delay,
                    'correlation_type': m.correlation_type.value,
                    'dispersion_us_per_MHz': m.dispersion_us_per_MHz,
                    'f_c_layer': m.f_c_layer,
                    'y_m': m.y_m,
                    'phi_inc': m.phi_inc
                }
                for m in self.modes
            ]
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ChannelConfig':
        """Create configuration from dictionary (e.g., loaded from JSON)"""
        modes = [
            ModeParameters(
                name=m.get('name', 'mode'),
                amplitude=m.get('amplitude', 1.0),
                floor_amplitude=m.get('floor_amplitude', 0.01),
                tau_L=m.get('tau_L', 0.0),
                sigma_tau=m.get('sigma_tau', 100.0),
                sigma_c=m.get('sigma_c', 50.0),
                sigma_D=m.get('sigma_D', 1.0),
                doppler_shift=m.get('doppler_shift', 0.0),
                doppler_shift_min_delay=m.get('doppler_shift_min_delay', 0.0),
                correlation_type=CorrelationType(m.get('correlation_type', 'gaussian')),
                dispersion_us_per_MHz=m.get('dispersion_us_per_MHz', None),
                f_c_layer=m.get('f_c_layer', None),
                y_m=m.get('y_m', 100e3),
                phi_inc=m.get('phi_inc', 0.0)
            )
            for m in data.get('modes', [])
        ]
        return cls(
            sample_rate=data.get('sample_rate', 1e6),
            modes=modes if modes else [ModeParameters()],
            spread_f_enabled=data.get('spread_f_enabled', False),
            random_seed=data.get('random_seed', None),
            k_factor=data.get('k_factor', None),
            dispersion_enabled=data.get('dispersion_enabled', False),
            carrier_frequency=data.get('carrier_frequency', 10e6)
        )

    @classmethod
    def from_json_file(cls, filepath: str) -> 'ChannelConfig':
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            return cls.from_dict(json.load(f))

    def to_json_file(self, filepath: str) -> None:
        """Save configuration to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class ExponentialCorrelatedRNG:
    """
    Generator for exponentially correlated random variables (multi-channel version)

    Implements Equation (12) from the paper for INDEPENDENT channels:
        x_n = u_n + (x_{n-1} - u_n) * lambda
        x_0 = (1 - lambda) * u_0
        lambda = exp[-delta_t * sigma_f]

    This produces a sequence of random variables with exponential autocorrelation
    as specified in Equation (13):
        rho(m) = exp[-m * delta_t * sigma_f]

    IMPORTANT: For multi-tap channels, each tap has INDEPENDENT exponential
    correlation. Call next_independent_array() to get values for all taps
    where each tap maintains its own state.

    The output values are in the range [-0.5, 0.5] (bounded by innovation range).
    When combined as complex (x_real + j*x_imag)/sqrt(2), the fading coefficient
    has variance ~1/12 per dimension, ~1/6 total - comparable to Rayleigh fading.

    Used for generating exponential (Lorentzian) Doppler spectra.
    """

    def __init__(self, sigma_f: float, delta_t: float, rng: np.random.Generator,
                 num_channels: int = 1):
        """
        Initialize the exponentially correlated RNG.

        Args:
            sigma_f: Spectral width parameter (Hz) - from Eq. (10)
            delta_t: Time step between samples (seconds)
            rng: NumPy random generator for reproducibility
            num_channels: Number of independent channels (taps)
        """
        self.lambda_param = np.exp(-delta_t * sigma_f)
        self.rng = rng
        self.num_channels = num_channels

        # Initialize with uniform values to start at steady-state-like conditions
        self.x_prev = self.rng.uniform(-0.5, 0.5, num_channels)

    def next(self) -> float:
        """
        Generate next exponentially correlated random value (single channel).

        Returns:
            Random value with exponential correlation to previous values
        """
        u = self.rng.uniform(-0.5, 0.5)
        if np.isscalar(self.x_prev):
            self.x_prev = u + (self.x_prev - u) * self.lambda_param
        else:
            self.x_prev[0] = u + (self.x_prev[0] - u) * self.lambda_param
            return self.x_prev[0]
        return self.x_prev

    def next_array(self, n: int) -> np.ndarray:
        """
        Generate array of n exponentially correlated values (single sequence).

        DEPRECATED: Use next_independent_array() for multi-tap channels.
        This generates a correlated sequence which is NOT what we want for
        independent fading per tap.
        """
        result = np.zeros(n)
        for i in range(n):
            result[i] = self.next()
        return result

    def next_independent_array(self) -> np.ndarray:
        """
        Generate next values for all channels with INDEPENDENT fading.

        Each channel maintains its own exponentially correlated state.
        This is the correct method for multi-tap channel fading.

        Returns:
            Array of num_channels values, each independently correlated in time
        """
        u = self.rng.uniform(-0.5, 0.5, self.num_channels)
        self.x_prev = u + (self.x_prev - u) * self.lambda_param
        return self.x_prev


class VoglerHoffmeyerChannel:
    """
    Vogler-Hoffmeyer HF Channel Model

    Implements the stochastic channel model from NTIA Report 90-255.
    The channel is modeled as a tapped delay line with time-varying complex gains.

    Channel effects modeled:
        1. Multipath delay spread (via delay amplitude function T(tau))
        2. Doppler spread (via correlation factor C(t))
        3. Doppler shift (via phase function phi_s)
        4. Multiple propagation modes (E-layer, F-layer low/high rays)
        5. Spread-F conditions (optional random multiplication)

    The scattering function S(tau, f_D) characterizes the channel completely.
    """

    def __init__(self, config: ChannelConfig):
        """
        Initialize the channel model.

        Args:
            config: Channel configuration with mode parameters
        """
        self.config = config
        self.sample_rate = config.sample_rate
        self.delta_t = 1.0 / config.sample_rate  # Time step (seconds)

        # Initialize random number generator
        self.rng = np.random.default_rng(config.random_seed)

        # Initialize dispersion model if enabled
        self.dispersion_model = None
        if config.dispersion_enabled:
            self.dispersion_model = DispersionModel(config.sample_rate)

        # Pre-compute parameters for each mode
        self._setup_modes()

        # Time index for phase continuity
        self.time_index = 0

    def _setup_modes(self) -> None:
        """Pre-compute delay taps and parameters for each mode"""
        self.mode_data = []

        for mode in self.config.modes:
            # Convert delay parameters from microseconds to samples
            tau_L_samples = int(mode.tau_L * 1e-6 * self.sample_rate)
            tau_U_samples = int(mode.tau_U * 1e-6 * self.sample_rate)
            num_taps = max(1, tau_U_samples - tau_L_samples + 1)

            # Create delay grid (in microseconds for calculations)
            delay_samples = np.arange(tau_L_samples, tau_U_samples + 1)
            delay_us = delay_samples / self.sample_rate * 1e6

            # Compute alpha and beta for delay amplitude function - Equations (4), (5)
            alpha_low, beta_low = self._compute_alpha_beta_low(mode)
            alpha_high, beta_high = self._compute_alpha_beta_high(mode)

            # Compute sigma_f for Doppler spread - Equations (7), (10)
            sigma_f = self._compute_sigma_f(mode)

            # Initialize exponentially correlated RNG if needed
            # Create TWO independent RNGs: one for real part, one for imaginary part
            # Each RNG has num_taps channels for independent fading per tap
            exp_rng_real = None
            exp_rng_imag = None
            exp_lambda = np.exp(-self.delta_t * sigma_f)  # lambda for exponential correlation
            if mode.correlation_type == CorrelationType.EXPONENTIAL:
                exp_rng_real = ExponentialCorrelatedRNG(sigma_f, self.delta_t, self.rng, num_taps)
                exp_rng_imag = ExponentialCorrelatedRNG(sigma_f, self.delta_t, self.rng, num_taps)

            # Initialize exponential correlation state for Numba path
            # Each tap has INDEPENDENT exponential correlation state
            # This creates frequency-selective fading (different taps fade independently)
            #
            # For the AR(1) process: x_n = u_n + (x_{n-1} - u_n) * lambda
            # The steady-state distribution has values in roughly [-0.5, 0.5] range
            # (the distribution is bounded by the innovation bounds).
            #
            # Initialize each tap with independent uniform values to start at
            # steady-state-like conditions for immediate proper fading behavior.
            exp_x_real_state = self.rng.uniform(-0.5, 0.5, num_taps)
            exp_x_imag_state = self.rng.uniform(-0.5, 0.5, num_taps)

            # Compute Gaussian correlation coefficient for AR(1) process
            # ρ = exp[-π × (σ_f × Δt)²] is the correlation between consecutive samples
            # For very slow fading (small σ_f × Δt), ρ ≈ 1 (samples nearly identical)
            gauss_rho = np.exp(-np.pi * (sigma_f * self.delta_t)**2)

            # Initialize Gaussian correlation state for each tap (complex Gaussian)
            # Start with random values to avoid transient
            z_init = self.rng.standard_normal(num_taps) + 1j * self.rng.standard_normal(num_taps)
            gauss_C_state = z_init / np.sqrt(2)  # Normalize to unit variance

            # Store mode data (without normalization factor initially)
            mode_data = {
                'mode': mode,
                'tau_L_samples': tau_L_samples,
                'num_taps': num_taps,
                'delay_samples': delay_samples,
                'delay_us': delay_us,
                'alpha_low': alpha_low,
                'beta_low': beta_low,
                'alpha_high': alpha_high,
                'beta_high': beta_high,
                'sigma_f': sigma_f,
                # Exponential correlation RNGs for Python fallback path
                # Two separate RNGs for real and imaginary parts
                'exp_rng_real': exp_rng_real,
                'exp_rng_imag': exp_rng_imag,
                # Gaussian AR(1) correlation parameters
                'gauss_rho': gauss_rho,
                'gauss_C_state': gauss_C_state,
                # Exponential correlation parameters for Numba path
                'exp_lambda': exp_lambda,
                'exp_x_real_state': exp_x_real_state,
                'exp_x_imag_state': exp_x_imag_state,
                # Variance normalization scale for exponential correlation
                # AR(1) with x_n = u + (x_{n-1} - u)*lambda has Var[x] = (1-lambda)/(1+lambda) * Var[u]
                # For uniform[-0.5,0.5], Var[u] = 1/12
                # Scale by sqrt(12*(1+lambda)/(1-lambda)) for unit variance
                'exp_variance_scale': np.sqrt(12.0 * (1 + exp_lambda) / max(1 - exp_lambda, 1e-10)),
                # Buffer for delay line implementation
                'buffer': np.zeros(max(1, tau_U_samples + 1), dtype=np.complex128),
                'norm_factor': 1.0  # Will be computed below
            }

            # Compute power normalization factor
            # The channel should preserve power: E[|output|²] = E[|input|²] * A²
            # where A is the mode amplitude.
            #
            # Each tap contributes T(τ)² * E[|C(t)|²] to the total power.
            # For Gaussian C(t), E[|C(t)|²] = 1 (unit variance complex Gaussian).
            # So total unnormalized power = Σ T(τ_k)²
            #
            # We need: Σ (T(τ_k) * norm_factor)² = A²
            # So: norm_factor = A / √(Σ T(τ_k)²)
            T = self._compute_delay_amplitude(delay_us, mode_data)
            sum_T_squared = np.sum(T**2)
            if sum_T_squared > 0:
                mode_data['norm_factor'] = mode.amplitude / np.sqrt(sum_T_squared)
            else:
                mode_data['norm_factor'] = mode.amplitude

            # Compute dispersion coefficient for this mode if enabled
            mode_data['dispersion_d'] = 0.0  # Default: no dispersion
            if self.config.dispersion_enabled:
                if mode.dispersion_us_per_MHz is not None:
                    # Use explicit dispersion value
                    mode_data['dispersion_d'] = mode.dispersion_us_per_MHz
                elif mode.f_c_layer is not None:
                    # Derive from quasi-parabolic layer parameters
                    mode_data['dispersion_d'] = compute_d_from_qp(
                        mode.f_c_layer,
                        self.config.carrier_frequency,
                        mode.y_m,
                        mode.phi_inc
                    )
                # else: use default of 0.0 (no dispersion for this mode)

            self.mode_data.append(mode_data)

    def _compute_alpha_beta_low(self, mode: ModeParameters) -> Tuple[float, float]:
        """
        Compute alpha and beta for delay amplitude T(tau) when y <= 1 (tau <= tau_c)

        Implements Equations (4a)-(4d) from the paper.

        These parameters shape the rising portion of the delay profile from
        tau_L to tau_c (minimum delay to carrier delay).
        """
        # Per Equation (4d)
        y1 = 0.01
        y2 = 0.5
        A1 = mode.floor_amplitude

        # A2 computation from Equation (4d)
        # A2 = exp[(ln(A1)*(1-y2+ln(y2))) / (1-y1+ln(y1))]
        numerator = np.log(A1) * (1 - y2 + np.log(y2))
        denominator = 1 - y1 + np.log(y1)
        if abs(denominator) < 1e-10:
            A2 = A1  # Fallback
        else:
            A2 = np.exp(numerator / denominator)

        # Ensure A2 is valid
        A2 = max(A2, 1e-10)

        # Compute d from Equation (4c)
        # d = (1-y2)*ln(y1) - (1-y1)*ln(y2)
        d = (1 - y2) * np.log(y1) - (1 - y1) * np.log(y2)

        if abs(d) < 1e-10:
            return 0.0, 0.0  # Degenerate case

        # Compute alpha from Equation (4a)
        # alpha = [(1-y2)*ln(A1) - (1-y1)*ln(A2)] / d
        alpha = ((1 - y2) * np.log(A1) - (1 - y1) * np.log(A2)) / d

        # Compute beta from Equation (4b)
        # beta = [ln(y1)*ln(A2) - ln(y2)*ln(A1)] / d
        beta = (np.log(y1) * np.log(A2) - np.log(y2) * np.log(A1)) / d

        return alpha, beta

    def _compute_alpha_beta_high(self, mode: ModeParameters) -> Tuple[float, float]:
        """
        Compute alpha and beta for delay amplitude T(tau) when y > 1 (tau > tau_c)

        The formula T(tau) = A * y^alpha * exp[beta * (1-y)] has its peak at
        y = alpha/beta. To ensure the peak is at y=1 (tau = tau_c), we set
        alpha = beta.

        With this constraint, we solve for alpha such that T(y2) = A_fl:
            alpha = ln(A_fl/A) / [ln(y2) + 1 - y2]

        where y2 = sigma_tau / sigma_c is the normalized maximum delay.

        Note: The paper's Equations (5a)-(5d) use two constraint points which
        can result in the peak being at y > 1. This implementation prioritizes
        having the peak at tau_c for physical correctness.
        """
        if mode.sigma_c == 0:
            return 0.0, 0.0

        # y2 is the normalized maximum delay
        y2 = mode.sigma_tau / mode.sigma_c
        A_fl_ratio = mode.floor_amplitude / mode.amplitude if mode.amplitude > 0 else 0.01

        # Ensure valid values
        A_fl_ratio = max(A_fl_ratio, 1e-10)
        y2 = max(y2, 1.001)

        # Compute alpha = beta such that peak is at y=1 and T(y2) = A_fl
        # From: ln(A_fl/A) = alpha * ln(y2) + alpha * (1 - y2)
        #       ln(A_fl/A) = alpha * [ln(y2) + 1 - y2]
        denom = np.log(y2) + 1 - y2

        if abs(denom) < 1e-10:
            return 1.0, 1.0  # Degenerate case

        alpha = np.log(A_fl_ratio) / denom
        beta = alpha  # Equal so peak is at y = alpha/beta = 1

        return alpha, beta

    def _compute_sigma_f(self, mode: ModeParameters) -> float:
        """
        Compute sigma_f spectral width parameter for correlation factor C(t)

        For Gaussian correlation (Equation 7):
            sigma_f = sigma_D * [-ln(A_fl/A) / pi]^(-1/2)

        For Exponential correlation (Equation 10):
            sigma_f = 2*pi*sigma_D * (A_fl/A) * [1 - (A_fl/A)^2]^(-1/2)
        """
        A = mode.amplitude
        A_fl = mode.floor_amplitude
        sigma_D = mode.sigma_D

        if A <= 0 or A_fl <= 0 or sigma_D <= 0:
            return 1.0  # Default

        ratio = A_fl / A

        if mode.correlation_type == CorrelationType.GAUSSIAN:
            # Equation (7)
            log_term = -np.log(ratio)
            if log_term <= 0:
                return sigma_D
            sigma_f = sigma_D * np.sqrt(np.pi / log_term)
        else:
            # Equation (10)
            denom = 1 - ratio**2
            if denom <= 0:
                return sigma_D
            sigma_f = 2 * np.pi * sigma_D * ratio / np.sqrt(denom)

        return max(sigma_f, 1e-6)  # Ensure positive

    def _compute_delay_amplitude(self, tau_us: np.ndarray, mode_data: dict) -> np.ndarray:
        """
        Compute delay amplitude factor T(tau) for all delay taps

        Implements Equation (3):
            T(tau) = A * y^alpha * exp[beta * (1 - y)]
            where y = (tau - tau_L) / (tau_c - tau_L)

        Args:
            tau_us: Array of delay values in microseconds
            mode_data: Pre-computed mode parameters

        Returns:
            Array of amplitude values T(tau) for each delay tap
        """
        mode = mode_data['mode']

        # Compute normalized delay y - Equation (3)
        sigma_c = mode.sigma_c
        if sigma_c == 0:
            return np.ones_like(tau_us) * mode.amplitude

        y = (tau_us - mode.tau_L) / sigma_c

        # Initialize output (explicitly float to avoid truncation if tau_us is int)
        T = np.zeros_like(tau_us, dtype=float)

        # For y <= 1 (tau <= tau_c): use low parameters
        mask_low = (y > 0) & (y <= 1)
        if np.any(mask_low):
            alpha = mode_data['alpha_low']
            beta = mode_data['beta_low']
            y_low = y[mask_low]
            # T(tau) = A * y^alpha * exp[beta * (1 - y)]
            T[mask_low] = mode.amplitude * np.power(y_low, alpha) * np.exp(beta * (1 - y_low))

        # For y > 1 (tau > tau_c): use high parameters
        mask_high = y > 1
        if np.any(mask_high):
            alpha = mode_data['alpha_high']
            beta = mode_data['beta_high']
            y_high = y[mask_high]
            # Same formula with different alpha, beta
            T[mask_high] = mode.amplitude * np.power(y_high, alpha) * np.exp(beta * (1 - y_high))

        # Apply floor (minimum value) - with corrected alpha=beta, T should not exceed A
        T = np.maximum(T, mode.floor_amplitude)

        return T

    def _generate_gaussian_fading(self, num_samples: int) -> np.ndarray:
        """
        Generate Gaussian-shaped fading process using Box-Muller transform

        Implements Equation (9) for generating Gaussian random variates:
            z1 = sqrt(-2 * ln(u1)) * cos(2*pi*u2)
            z2 = sqrt(-2 * ln(u1)) * sin(2*pi*u2)

        Returns complex Gaussian samples (Rayleigh fading envelope).
        """
        # Generate pairs using Box-Muller (Equation 9)
        u1 = self.rng.uniform(0, 1, num_samples)
        u2 = self.rng.uniform(0, 1, num_samples)

        # Avoid log(0)
        u1 = np.maximum(u1, 1e-10)

        z_real = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
        z_imag = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)

        return z_real + 1j * z_imag

    def _compute_correlation_factor(self, t: float, mode_data: dict,
                                    num_taps: int) -> np.ndarray:
        """
        Compute correlation factor C(t) for all taps at time t

        For Gaussian correlation (Equation 7), the autocorrelation is:
            R(Δt) = exp[-π × (σ_f × Δt)²]

        This is implemented using an AR(1) process:
            C[n] = ρ × C[n-1] + √(1-ρ²) × z[n]

        where ρ = exp[-π × (σ_f × Δt)²] is the correlation coefficient
        and z[n] is white complex Gaussian noise.

        For Exponential correlation (Equation 10), the exponentially
        correlated RNG from Eq. (12) is used.

        Args:
            t: Time lag (seconds) - not used, kept for API compatibility
            mode_data: Pre-computed mode parameters
            num_taps: Number of delay taps

        Returns:
            Complex array of correlation factors for each tap
        """
        mode = mode_data['mode']

        if mode.correlation_type == CorrelationType.GAUSSIAN:
            # Gaussian correlation using AR(1) process
            # C[n] = ρ × C[n-1] + √(1-ρ²) × z[n]
            rho = mode_data['gauss_rho']
            C_prev = mode_data['gauss_C_state']

            # Generate new white noise
            z = self._generate_gaussian_fading(num_taps) / np.sqrt(2)

            # AR(1) update: maintains Gaussian autocorrelation
            # The innovation coefficient √(1-ρ²) ensures unit variance
            innovation_coeff = np.sqrt(1 - rho**2)
            C = rho * C_prev + innovation_coeff * z

            # Store state for next sample
            mode_data['gauss_C_state'] = C
        else:
            # Equation (10): Exponential correlation
            # Use exponentially correlated random process with INDEPENDENT fading per tap
            # Each tap has its own AR(1) process to create frequency-selective fading
            exp_rng_real = mode_data['exp_rng_real']
            exp_rng_imag = mode_data['exp_rng_imag']
            x_real = exp_rng_real.next_independent_array()
            x_imag = exp_rng_imag.next_independent_array()
            x = x_real + 1j * x_imag
            # Scale uniform[-0.5, 0.5] (variance 1/12) to unit variance by multiplying by sqrt(12)
            # Then combine as complex with 1/sqrt(2) to get E[|C|^2] = 1
            C = np.sqrt(12.0) * x / np.sqrt(2)

        return C

    def _compute_phase(self, tau_us: np.ndarray, t: float, mode: ModeParameters) -> np.ndarray:
        """
        Compute phase function phi_s(tau, f_s; t) for all delay taps

        Implements Equation (15):
            phi_s(tau, f_s; t) = 2*pi * [phi_0 + {f_s + b*(tau_c - tau)} * t]

        where b is the delay-Doppler coupling coefficient from Equation (15b):
            b = (f_sL - f_s) / (tau_c - tau_L)

        This creates the slanted ridges in the scattering function that
        couple delay and Doppler shift.

        Args:
            tau_us: Array of delay values in microseconds
            t: Current time (seconds)
            mode: Mode parameters

        Returns:
            Array of phase values (radians) for each tap
        """
        # Get delay-Doppler coupling coefficient b (Hz/us)
        b = mode.doppler_delay_coupling

        # Effective Doppler frequency for each tap - Equation (15a)
        # f_D_effective = f_s + b * (tau_c - tau)
        f_D_effective = mode.doppler_shift + b * (mode.tau_c - tau_us)

        # Phase = 2*pi * f_D * t (plus initial phase which we set to 0)
        phi = 2 * np.pi * f_D_effective * t

        return phi

    def _process_mode(self, input_samples: np.ndarray, mode_idx: int) -> np.ndarray:
        """
        Process input samples through a single propagation mode

        Implements the tapped delay line channel model where each tap has:
            h[k] = T(tau_k) * C(t) * exp(j * phi_s(tau_k, t))

        The output is the convolution of input with time-varying impulse response.

        Args:
            input_samples: Complex I/Q input samples
            mode_idx: Index of the mode to process

        Returns:
            Complex output samples for this mode
        """
        mode_data = self.mode_data[mode_idx]
        mode = mode_data['mode']
        num_taps = mode_data['num_taps']
        delay_us = mode_data['delay_us']
        delay_samples = mode_data['delay_samples']

        # Get delay amplitude (static over short time scales)
        # Apply normalization factor so total power equals mode amplitude squared
        T = self._compute_delay_amplitude(delay_us, mode_data) * mode_data['norm_factor']

        # Use Numba-optimized path when available (now supports spread-F)
        if NUMBA_AVAILABLE:
            if mode.correlation_type == CorrelationType.GAUSSIAN:
                return self._process_mode_numba(input_samples, mode_idx, T)
            else:  # EXPONENTIAL
                return self._process_mode_numba_exp(input_samples, mode_idx, T)

        # Fallback: Python implementation for spread-F or when Numba unavailable
        return self._process_mode_python(input_samples, mode_idx, T)

    def _process_mode_numba(self, input_samples: np.ndarray, mode_idx: int,
                           T: np.ndarray) -> np.ndarray:
        """Numba-optimized processing for Gaussian correlation"""
        mode_data = self.mode_data[mode_idx]
        mode = mode_data['mode']
        num_taps = mode_data['num_taps']
        delay_us = mode_data['delay_us']
        delay_samples = mode_data['delay_samples']

        # Get AR(1) parameters
        rho = mode_data['gauss_rho']
        innovation_coeff = np.sqrt(1 - rho**2)
        C_state = mode_data['gauss_C_state'].copy()

        # Get Doppler parameters
        tau_c = mode.tau_L + mode.sigma_c
        f_s = mode.doppler_shift
        b = mode.doppler_delay_coupling

        # Time parameters
        t_start = self.time_index * self.delta_t

        # Compute Rician fading coefficients
        k_factor = self.config.k_factor
        if k_factor is not None and k_factor > 0:
            direct_coeff = np.sqrt(k_factor / (k_factor + 1))
            scatter_coeff = np.sqrt(1 / (k_factor + 1))
        else:
            direct_coeff = 0.0
            scatter_coeff = 1.0

        # Initialize RNG state if not present
        if 'numba_rng_state' not in mode_data:
            mode_data['numba_rng_state'] = np.array([self.rng.integers(1, 2**31-1)], dtype=np.int64)

        # Call Numba-optimized function
        output, new_buffer, new_C, new_rng = _process_samples_numba(
            input_samples,
            mode_data['buffer'],
            delay_samples.astype(np.int64),
            T,
            C_state,
            rho,
            innovation_coeff,
            delay_us,
            tau_c,
            f_s,
            b,
            t_start,
            self.delta_t,
            mode_data['numba_rng_state'],
            direct_coeff,
            scatter_coeff,
            self.config.spread_f_enabled
        )

        # Update state
        mode_data['buffer'] = new_buffer
        mode_data['gauss_C_state'] = new_C
        mode_data['numba_rng_state'] = new_rng

        return output

    def _process_mode_numba_exp(self, input_samples: np.ndarray, mode_idx: int,
                                T: np.ndarray) -> np.ndarray:
        """Numba-optimized processing for exponential correlation"""
        mode_data = self.mode_data[mode_idx]
        mode = mode_data['mode']
        num_taps = mode_data['num_taps']
        delay_us = mode_data['delay_us']
        delay_samples = mode_data['delay_samples']

        # Get exponential correlation parameters
        lambda_param = mode_data['exp_lambda']
        x_real_state = mode_data['exp_x_real_state'].copy()
        x_imag_state = mode_data['exp_x_imag_state'].copy()

        # Get Doppler parameters
        tau_c = mode.tau_L + mode.sigma_c
        f_s = mode.doppler_shift
        b = mode.doppler_delay_coupling

        # Time parameters
        t_start = self.time_index * self.delta_t

        # Compute Rician fading coefficients
        k_factor = self.config.k_factor
        if k_factor is not None and k_factor > 0:
            direct_coeff = np.sqrt(k_factor / (k_factor + 1))
            scatter_coeff = np.sqrt(1 / (k_factor + 1))
        else:
            direct_coeff = 0.0
            scatter_coeff = 1.0

        # Initialize RNG state if not present
        if 'numba_rng_state_exp' not in mode_data:
            mode_data['numba_rng_state_exp'] = np.array([self.rng.integers(1, 2**31-1)], dtype=np.int64)

        # Call Numba-optimized function for exponential correlation
        output, new_buffer, new_x_real, new_x_imag, new_rng = _process_samples_numba_exp(
            input_samples,
            mode_data['buffer'],
            delay_samples.astype(np.int64),
            T,
            x_real_state,
            x_imag_state,
            lambda_param,
            delay_us,
            tau_c,
            f_s,
            b,
            t_start,
            self.delta_t,
            mode_data['numba_rng_state_exp'],
            direct_coeff,
            scatter_coeff,
            self.config.spread_f_enabled
        )

        # Update state
        mode_data['buffer'] = new_buffer
        mode_data['exp_x_real_state'] = new_x_real
        mode_data['exp_x_imag_state'] = new_x_imag
        mode_data['numba_rng_state_exp'] = new_rng

        return output

    def _process_mode_python(self, input_samples: np.ndarray, mode_idx: int,
                            T: np.ndarray) -> np.ndarray:
        """Python implementation (fallback for spread-F or when Numba unavailable)"""
        mode_data = self.mode_data[mode_idx]
        mode = mode_data['mode']
        num_taps = mode_data['num_taps']
        delay_us = mode_data['delay_us']
        delay_samples = mode_data['delay_samples']

        num_input = len(input_samples)
        output = np.zeros(num_input, dtype=np.complex128)

        # Compute Rician fading coefficients if k_factor is set
        k_factor = self.config.k_factor
        if k_factor is not None and k_factor > 0:
            # Rician: h = sqrt(K/(K+1)) * direct + sqrt(1/(K+1)) * scattered
            direct_coeff = np.sqrt(k_factor / (k_factor + 1))
            scatter_coeff = np.sqrt(1 / (k_factor + 1))
        else:
            # Rayleigh: h = scattered only
            direct_coeff = 0.0
            scatter_coeff = 1.0

        # Process sample by sample for time-varying channel
        buffer = mode_data['buffer'].copy()

        for n in range(num_input):
            # Current time
            t = (self.time_index + n) * self.delta_t

            # Update delay line buffer
            buffer = np.roll(buffer, 1)
            buffer[0] = input_samples[n]

            # Compute time-varying correlation factor (scattered component)
            C = self._compute_correlation_factor(t, mode_data, num_taps)

            # Apply Rician fading: direct path only on first tap (LOS)
            # Other taps are scattered paths only
            fading_gain = scatter_coeff * C  # All taps get scattered component
            fading_gain[0] = direct_coeff + scatter_coeff * C[0]  # First tap also gets direct

            # Compute phase for Doppler shift
            phi = self._compute_phase(delay_us, t, mode)

            # Compute complex tap gains: h[k] = T(tau_k) * fading_gain * exp(j*phi)
            tap_gains = T * fading_gain * np.exp(1j * phi)

            # Apply spread-F if enabled (multiply by random factor)
            if self.config.spread_f_enabled:
                spread_factor = self.rng.uniform(0.1, 1.0, num_taps)
                tap_gains *= spread_factor

            # Compute output as sum of tap contributions
            for k, delay in enumerate(delay_samples):
                if delay < len(buffer):
                    output[n] += buffer[delay] * tap_gains[k]

        # Store buffer state for continuity
        mode_data['buffer'] = buffer

        return output

    def process(self, input_samples: np.ndarray) -> np.ndarray:
        """
        Apply channel model to I/Q samples

        This is the main processing function. It applies all configured
        propagation modes to the input signal and sums the results.

        From the paper, the complete channel response combines contributions
        from all propagation modes (E-layer, F-layer low-ray, F-layer high-ray).

        If dispersion is enabled, frequency-dependent group delay is applied
        before the tapped delay line processing. Each mode can have different
        dispersion characteristics (e.g., E-layer vs F-layer).

        Args:
            input_samples: Complex I/Q input samples (numpy array)

        Returns:
            Complex I/Q output samples after channel effects
        """
        output = np.zeros(len(input_samples), dtype=np.complex128)

        # Process each mode and sum contributions
        for mode_idx in range(len(self.config.modes)):
            mode_data = self.mode_data[mode_idx]

            # Apply dispersion if enabled for this mode
            if self.dispersion_model is not None and mode_data['dispersion_d'] != 0.0:
                dispersed_input = self.dispersion_model.apply_dispersion(
                    input_samples, mode_data['dispersion_d']
                )
            else:
                dispersed_input = input_samples

            mode_output = self._process_mode(dispersed_input, mode_idx)
            output += mode_output

        # Update time index for phase continuity across blocks
        self.time_index += len(input_samples)

        return output

    def process_block(self, input_samples: np.ndarray, block_size: int = 1024) -> np.ndarray:
        """
        Process input in blocks for memory efficiency

        This method processes large files in smaller blocks while maintaining
        channel state continuity between blocks.

        Args:
            input_samples: Complex I/Q input samples
            block_size: Number of samples per block

        Returns:
            Complex I/Q output samples
        """
        num_samples = len(input_samples)
        output = np.zeros(num_samples, dtype=np.complex128)

        for start in range(0, num_samples, block_size):
            end = min(start + block_size, num_samples)
            output[start:end] = self.process(input_samples[start:end])

        return output

    def reset(self) -> None:
        """Reset channel state for processing a new signal"""
        self.time_index = 0
        self._setup_modes()

    def get_impulse_response(self, num_samples: int = 1024) -> np.ndarray:
        """
        Get the instantaneous channel impulse response

        Useful for visualization and verification.

        Args:
            num_samples: Length of impulse response

        Returns:
            Complex impulse response
        """
        # Create impulse
        impulse = np.zeros(num_samples, dtype=np.complex128)
        impulse[0] = 1.0

        # Save state
        saved_time_index = self.time_index
        saved_buffers = [md['buffer'].copy() for md in self.mode_data]

        # Reset and process
        self.reset()
        response = self.process(impulse)

        # Restore state
        self.time_index = saved_time_index
        for md, buf in zip(self.mode_data, saved_buffers):
            md['buffer'] = buf

        return response

    def compute_scattering_function(self, num_delay_bins: int = 64,
                                    num_doppler_bins: int = 64,
                                    observation_time: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the THEORETICAL channel scattering function S(tau, f_D)

        The scattering function is the primary verification tool from the paper
        (Equation 1). It shows the distribution of signal power in delay-Doppler space.

        For a Gaussian correlation (Equation 7), the theoretical scattering function is:
            S(tau, f_D) = T(tau)^2 * G(f_D - f_s(tau))

        where:
            - T(tau) is the delay amplitude function (Equation 3)
            - G(f_D) is a Gaussian centered at the Doppler shift f_s(tau)
            - f_s(tau) is computed from Equation 15

        For an exponential correlation (Equation 10), G(f_D) is Lorentzian.

        Args:
            num_delay_bins: Number of delay bins
            num_doppler_bins: Number of Doppler frequency bins
            observation_time: Not used for theoretical computation (kept for API compatibility)

        Returns:
            Tuple of (delay_axis, doppler_axis, scattering_function)
        """
        # Determine delay and Doppler ranges from all modes
        max_delay_us = max(m.tau_U for m in self.config.modes)
        max_doppler_hz = max(m.sigma_D * 3 + abs(m.doppler_shift) for m in self.config.modes)

        # Create axes
        delay_axis = np.linspace(0, max_delay_us, num_delay_bins)  # microseconds
        doppler_axis = np.linspace(-max_doppler_hz, max_doppler_hz, num_doppler_bins)  # Hz

        # Compute theoretical scattering function
        S = np.zeros((num_delay_bins, num_doppler_bins))

        for mode_data in self.mode_data:
            mode = mode_data['mode']

            # For each delay bin
            for i, tau in enumerate(delay_axis):
                # Compute delay amplitude T(tau) - Equation 3
                T = self._compute_delay_amplitude(np.array([tau]), mode_data)[0]

                # Compute Doppler shift at this delay - Equation 15
                # f_s(tau) = f_s + b * (tau_c - tau)
                # where b = (f_sL - f_s) / (tau_c - tau_L)
                tau_c = mode.tau_L + mode.sigma_c
                if mode.sigma_c > 0:
                    b = (mode.doppler_shift_min_delay - mode.doppler_shift) / mode.sigma_c
                else:
                    b = 0.0
                f_s_tau = mode.doppler_shift + b * (tau_c - tau)

                # Compute Doppler spectrum shape centered at f_s(tau)
                sigma_D = mode.sigma_D

                if mode.correlation_type == CorrelationType.GAUSSIAN:
                    # Gaussian Doppler spectrum - FT of Gaussian C(t)
                    # G(f_D) = exp(-pi * (f_D / sigma_D)^2)
                    if sigma_D > 0:
                        doppler_spectrum = np.exp(-np.pi * ((doppler_axis - f_s_tau) / sigma_D)**2)
                    else:
                        # Delta function at f_s_tau
                        doppler_spectrum = np.zeros_like(doppler_axis)
                        closest_idx = np.argmin(np.abs(doppler_axis - f_s_tau))
                        doppler_spectrum[closest_idx] = 1.0
                else:
                    # Lorentzian Doppler spectrum - FT of exponential C(t)
                    # L(f_D) = sigma_D / (sigma_D^2 + (pi * f_D)^2)
                    if sigma_D > 0:
                        f_D_shifted = doppler_axis - f_s_tau
                        doppler_spectrum = sigma_D / (sigma_D**2 + (np.pi * f_D_shifted)**2)
                    else:
                        doppler_spectrum = np.zeros_like(doppler_axis)
                        closest_idx = np.argmin(np.abs(doppler_axis - f_s_tau))
                        doppler_spectrum[closest_idx] = 1.0

                # S(tau, f_D) = T(tau)^2 * G(f_D - f_s(tau))
                S[i, :] += (T**2) * doppler_spectrum

        # Normalize
        if S.max() > 0:
            S /= S.max()

        return delay_axis, doppler_axis, S


# ============================================================================
# Preset Configurations (from Table 1 in the paper)
# ============================================================================

def create_equatorial_config(sample_rate: float = 1e6) -> ChannelConfig:
    """
    Create equatorial path configuration

    From Paper Table 1 - Equatorial Path:
    - Single F-layer mode with large delay spread
    - Moderate Doppler spread
    - Typical of trans-equatorial HF propagation
    """
    mode = ModeParameters(
        name="F-layer (equatorial)",
        amplitude=1.0,
        floor_amplitude=0.01,
        tau_L=0.0,
        sigma_tau=880.0,      # Large delay spread (us)
        sigma_c=220.0,        # Carrier delay subinterval (us)
        sigma_D=2.0,          # Doppler spread (Hz)
        doppler_shift=0.0,
        doppler_shift_min_delay=0.0,
        correlation_type=CorrelationType.GAUSSIAN
    )
    return ChannelConfig(sample_rate=sample_rate, modes=[mode])


def create_polar_config(sample_rate: float = 1e6) -> ChannelConfig:
    """
    Create polar path configuration

    From Paper Table 1 - Polar Path:
    - Multiple modes (E-layer and F-layer)
    - Larger Doppler spreads due to auroral activity
    - Typical of high-latitude HF propagation
    """
    e_layer = ModeParameters(
        name="E-layer (polar)",
        amplitude=0.7,
        floor_amplitude=0.01,
        tau_L=0.0,
        sigma_tau=250.0,
        sigma_c=100.0,
        sigma_D=16.0,         # Higher Doppler spread
        doppler_shift=0.0,
        doppler_shift_min_delay=0.0,
        correlation_type=CorrelationType.GAUSSIAN
    )

    f_low = ModeParameters(
        name="F-layer low-ray (polar)",
        amplitude=1.0,
        floor_amplitude=0.01,
        tau_L=300.0,          # Offset from E-layer
        sigma_tau=400.0,
        sigma_c=135.0,
        sigma_D=7.0,
        doppler_shift=0.0,
        doppler_shift_min_delay=0.0,
        correlation_type=CorrelationType.GAUSSIAN
    )

    return ChannelConfig(sample_rate=sample_rate, modes=[e_layer, f_low])


def create_midlatitude_config(sample_rate: float = 1e6,
                               dispersion_enabled: bool = False,
                               carrier_frequency: float = 10e6) -> ChannelConfig:
    """
    Create mid-latitude path configuration

    From Paper Table 1 - Mid-latitude Short Path:
    - Narrow delay spread
    - Very small Doppler spread (stable ionosphere)
    - Shows delay-Doppler coupling (slanted ridges in scattering function)

    Args:
        sample_rate: Sample rate in Hz
        dispersion_enabled: Enable frequency-dependent group delay
        carrier_frequency: RF carrier frequency (Hz) for dispersion calculation
    """
    mode = ModeParameters(
        name="F-layer (mid-latitude)",
        amplitude=1.0,
        floor_amplitude=0.01,
        tau_L=0.0,
        sigma_tau=50.0,       # Narrow delay spread (us)
        sigma_c=25.0,
        sigma_D=0.1,          # Very small Doppler spread (Hz)
        doppler_shift=0.2,    # Small positive shift
        doppler_shift_min_delay=-0.2,  # Different shift at min delay
        correlation_type=CorrelationType.GAUSSIAN,
        # Dispersion parameters (QP-derived from F-layer)
        f_c_layer=6e6 if dispersion_enabled else None,  # 6 MHz critical frequency
        y_m=120e3,            # 120 km semi-thickness
        phi_inc=0.35          # ~20° incidence angle
    )
    return ChannelConfig(
        sample_rate=sample_rate,
        modes=[mode],
        dispersion_enabled=dispersion_enabled,
        carrier_frequency=carrier_frequency
    )


def create_auroral_spread_f_config(sample_rate: float = 1e6) -> ChannelConfig:
    """
    Create auroral spread-F configuration

    From Paper Table 1 - Auroral Spread-F:
    - Very large delay spread
    - Moderate Doppler spread
    - Spread-F enabled for diffuse scattering
    - Typical of disturbed ionospheric conditions
    """
    mode = ModeParameters(
        name="F-layer (auroral spread-F)",
        amplitude=1.0,
        floor_amplitude=0.01,
        tau_L=0.0,
        sigma_tau=2000.0,     # Very large delay spread (us)
        sigma_c=500.0,
        sigma_D=5.0,
        doppler_shift=0.0,
        doppler_shift_min_delay=0.0,
        correlation_type=CorrelationType.EXPONENTIAL  # Use exponential for spread-F
    )
    return ChannelConfig(
        sample_rate=sample_rate,
        modes=[mode],
        spread_f_enabled=True  # Enable spread-F simulation
    )


def create_auroral_config(sample_rate: float = 1e6) -> ChannelConfig:
    """
    Create auroral configuration with exponential correlation (Numba-accelerated)

    Similar to auroral_spread_f but without the spread-F random multiplier.
    Uses exponential correlation which models the peaked/Lorentzian Doppler spectrum
    typical of disturbed polar/auroral ionosphere.

    This version is Numba-accelerated for fast processing.
    """
    mode = ModeParameters(
        name="F-layer (auroral)",
        amplitude=1.0,
        floor_amplitude=0.01,
        tau_L=0.0,
        sigma_tau=2000.0,     # Very large delay spread (us)
        sigma_c=500.0,
        sigma_D=5.0,
        doppler_shift=0.0,
        doppler_shift_min_delay=0.0,
        correlation_type=CorrelationType.EXPONENTIAL
    )
    return ChannelConfig(
        sample_rate=sample_rate,
        modes=[mode],
        spread_f_enabled=False  # Disabled for Numba acceleration
    )


def create_auroral_complex_config(sample_rate: float = 1e6) -> ChannelConfig:
    """
    Create complex auroral configuration

    From Paper Table 1 - Auroral Complex:
    - Multiple overlapping modes
    - Narrow individual spreads but complex combined response
    - Models multiple reflection paths through disturbed ionosphere
    """
    mode1 = ModeParameters(
        name="Mode 1 (auroral)",
        amplitude=1.0,
        floor_amplitude=0.01,
        tau_L=0.0,
        sigma_tau=60.0,
        sigma_c=25.0,
        sigma_D=0.1,
        doppler_shift=0.2,
        doppler_shift_min_delay=-0.1,
        correlation_type=CorrelationType.GAUSSIAN
    )

    mode2 = ModeParameters(
        name="Mode 2 (auroral)",
        amplitude=0.8,
        floor_amplitude=0.01,
        tau_L=80.0,
        sigma_tau=80.0,
        sigma_c=40.0,
        sigma_D=0.2,
        doppler_shift=-0.1,
        doppler_shift_min_delay=0.1,
        correlation_type=CorrelationType.GAUSSIAN
    )

    mode3 = ModeParameters(
        name="Mode 3 (auroral)",
        amplitude=0.6,
        floor_amplitude=0.01,
        tau_L=180.0,
        sigma_tau=100.0,
        sigma_c=50.0,
        sigma_D=0.3,
        doppler_shift=0.0,
        doppler_shift_min_delay=0.0,
        correlation_type=CorrelationType.GAUSSIAN
    )

    return ChannelConfig(sample_rate=sample_rate, modes=[mode1, mode2, mode3])


def create_wideband_dispersive_config(sample_rate: float = 2e6,
                                       carrier_frequency: float = 10e6,
                                       dispersion_us_per_MHz: float = None,
                                       condition: str = 'moderate') -> ChannelConfig:
    """
    Create wideband dispersive channel configuration

    Designed for testing wideband (>48 kHz) HF systems with realistic
    frequency-dependent group delay (dispersion).

    Args:
        sample_rate: Sample rate in Hz (default 2 MHz for wideband)
        carrier_frequency: RF carrier frequency (Hz)
        dispersion_us_per_MHz: Manual dispersion coefficient (μs/MHz).
                               If None, derived from condition.
        condition: Ionospheric condition preset:
                   'quiet' (10-30 μs/MHz), 'moderate' (30-80 μs/MHz),
                   'disturbed' (80-150 μs/MHz), 'spread_f' (150-240 μs/MHz)

    Returns:
        ChannelConfig with dispersion enabled
    """
    # Condition-dependent parameters
    condition_params = {
        'quiet': {
            'f_c_layer': 5e6,     # Lower critical frequency
            'sigma_tau': 30.0,    # Narrow delay spread
            'sigma_D': 0.2,       # Small Doppler spread
            'd_typical': 20.0     # ~20 μs/MHz
        },
        'moderate': {
            'f_c_layer': 7e6,
            'sigma_tau': 100.0,
            'sigma_D': 1.0,
            'd_typical': 50.0     # ~50 μs/MHz
        },
        'disturbed': {
            'f_c_layer': 9e6,
            'sigma_tau': 500.0,
            'sigma_D': 3.0,
            'd_typical': 100.0    # ~100 μs/MHz
        },
        'spread_f': {
            'f_c_layer': 11e6,
            'sigma_tau': 2000.0,
            'sigma_D': 8.0,
            'd_typical': 200.0    # ~200 μs/MHz
        }
    }

    if condition not in condition_params:
        raise ValueError(f"Unknown condition '{condition}'. "
                        f"Available: {list(condition_params.keys())}")

    params = condition_params[condition]

    # Use explicit dispersion or derive from QP
    if dispersion_us_per_MHz is not None:
        d_value = dispersion_us_per_MHz
        f_c = None  # Use explicit value
    else:
        d_value = None
        f_c = params['f_c_layer']

    mode = ModeParameters(
        name=f"F-layer ({condition})",
        amplitude=1.0,
        floor_amplitude=0.01,
        tau_L=0.0,
        sigma_tau=params['sigma_tau'],
        sigma_c=params['sigma_tau'] / 4,  # sigma_c = sigma_tau / 4
        sigma_D=params['sigma_D'],
        doppler_shift=0.0,
        doppler_shift_min_delay=0.0,
        correlation_type=(CorrelationType.EXPONENTIAL
                          if condition == 'spread_f'
                          else CorrelationType.GAUSSIAN),
        # Dispersion parameters
        dispersion_us_per_MHz=d_value,
        f_c_layer=f_c,
        y_m=100e3,
        phi_inc=0.3  # ~17° incidence
    )

    return ChannelConfig(
        sample_rate=sample_rate,
        modes=[mode],
        dispersion_enabled=True,
        carrier_frequency=carrier_frequency,
        spread_f_enabled=(condition == 'spread_f')
    )


# Dictionary of available presets
PRESETS = {
    'equatorial': create_equatorial_config,
    'polar': create_polar_config,
    'midlatitude': create_midlatitude_config,
    'auroral': create_auroral_config,
    'auroral_spread_f': create_auroral_spread_f_config,
    'auroral_complex': create_auroral_complex_config,
    'wideband_dispersive': create_wideband_dispersive_config,
}


def get_preset(name: str, sample_rate: float = 1e6) -> ChannelConfig:
    """
    Get a preset channel configuration by name

    Args:
        name: Preset name (equatorial, polar, midlatitude, auroral_spread_f, auroral_complex)
        sample_rate: Sample rate in Hz

    Returns:
        ChannelConfig for the specified preset
    """
    if name not in PRESETS:
        available = ', '.join(PRESETS.keys())
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")
    return PRESETS[name](sample_rate)


def list_presets() -> List[str]:
    """Return list of available preset names"""
    return list(PRESETS.keys())
