"""
SC-FDE Transmitter

Implements:
- Pilot insertion (Zadoff-Chu sequence)
- IFFT for frequency-domain processing
- Cyclic prefix addition (adaptive length)
- Frame construction
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

from .utils import compute_subcarrier_indices


@dataclass
class TransmitterConfig:
    """Transmitter configuration parameters"""
    sample_rate: float = 1.536e6  # Sample rate in Hz
    fft_size: int = 4096  # FFT size
    cp_length: int = 1536  # Cyclic prefix length (samples)
    n_pilots: int = 251  # Number of pilot subcarriers (prime for optimal ZC autocorrelation)
    pilot_spacing: int = 8  # Pilot spacing (1:N data)
    n_data_carriers: int = 1757  # Number of data subcarriers
    pilot_sequence_root: int = 25  # Zadoff-Chu root
    pilot_boost_db: float = 0.0  # Pilot power boost in dB (0-6 dB)


# Channel delay spreads for Nyquist pilot spacing calculation
# Based on NTIA Vogler-Hoffmeyer model characterization
CHANNEL_DELAY_SPREADS = {
    'benign': 167e-6,      # 167 μs - minimal multipath
    'midlatitude': 583e-6,  # 583 μs - moderate multipath
    'equatorial': 1000e-6,  # 1 ms - severe spread-F
    'auroral': 2083e-6,    # 2.08 ms - extreme auroral conditions
}

# Recommended pilot spacing for each channel type
# Based on Nyquist criterion: spacing ≤ FFT_size / (τ_max × fs)
# These values satisfy the criterion with appropriate margin
PILOT_SPACING_PRESETS = {
    'benign': 8,      # Nyquist limit ~16, using 8 for margin
    'midlatitude': 4,  # Nyquist limit ~4.6, using 4
    'equatorial': 2,   # Nyquist limit ~2.7, using 2
    'auroral': 1,      # Nyquist limit ~1.3, using 1 (every carrier is pilot)
}


def calculate_nyquist_pilot_spacing(delay_spread_s: float, fft_size: int,
                                     sample_rate: float) -> float:
    """
    Calculate maximum pilot spacing satisfying Nyquist criterion.

    The Nyquist criterion for channel estimation requires:
        pilot_spacing ≤ FFT_size / (τ_max × fs)

    where τ_max is the maximum channel delay spread.

    Args:
        delay_spread_s: Maximum delay spread in seconds
        fft_size: FFT size
        sample_rate: Sample rate in Hz

    Returns:
        Maximum allowable pilot spacing (float, may be < 1 for extreme channels)
    """
    delay_samples = delay_spread_s * sample_rate
    if delay_samples <= 0:
        return float('inf')
    return fft_size / delay_samples


def validate_pilot_spacing(pilot_spacing: int, delay_spread_s: float,
                           fft_size: int, sample_rate: float,
                           raise_on_violation: bool = False) -> Tuple[bool, float]:
    """
    Validate pilot spacing against Nyquist criterion.

    Args:
        pilot_spacing: Current pilot spacing
        delay_spread_s: Maximum delay spread in seconds
        fft_size: FFT size
        sample_rate: Sample rate in Hz
        raise_on_violation: If True, raise ValueError on violation

    Returns:
        Tuple of (is_valid, nyquist_limit)
    """
    nyquist_limit = calculate_nyquist_pilot_spacing(
        delay_spread_s, fft_size, sample_rate
    )
    is_valid = pilot_spacing <= nyquist_limit

    if not is_valid and raise_on_violation:
        raise ValueError(
            f"Pilot spacing {pilot_spacing} violates Nyquist criterion. "
            f"Maximum allowed: {nyquist_limit:.1f} for delay spread "
            f"{delay_spread_s*1e6:.0f} μs"
        )

    return is_valid, nyquist_limit


def generate_zadoff_chu(length: int, root: int) -> np.ndarray:
    """
    Generate Zadoff-Chu sequence.

    Zadoff-Chu sequences have ideal autocorrelation (zero sidelobes for
    prime lengths) and low cross-correlation between different roots.

    For best autocorrelation properties, use a prime-length sequence.
    The default n_pilots=251 is prime.

    Args:
        length: Sequence length (should be prime for best properties)
        root: Sequence root (1 to length-1)

    Returns:
        Complex Zadoff-Chu sequence
    """
    n = np.arange(length)
    # ZC formula: exp(-j * π * root * n * (n + 1) / length)
    # This formula works for both even and odd lengths
    seq = np.exp(-1j * np.pi * root * n * (n + 1) / length)
    return seq


def apply_pilot_boost(freq_block: np.ndarray,
                      pilot_indices: np.ndarray,
                      data_indices: np.ndarray,
                      boost_db: float) -> np.ndarray:
    """
    Apply pilot power boosting while maintaining total power.

    Boosts pilot symbols by boost_db while scaling data symbols down
    to maintain the same total transmit power.

    Args:
        freq_block: Frequency-domain symbols
        pilot_indices: Indices of pilot subcarriers
        data_indices: Indices of data subcarriers
        boost_db: Pilot boost in dB (0 to 6)

    Returns:
        Boosted frequency block

    Raises:
        ValueError: If boost_db is out of valid range
    """
    if boost_db < 0 or boost_db > 6:
        raise ValueError(f"Pilot boost must be 0-6 dB, got {boost_db}")

    if boost_db == 0.0:
        return freq_block.copy()

    n_pilot = len(pilot_indices)
    n_data = len(data_indices)

    # Handle edge cases
    if n_pilot == 0:
        return freq_block.copy()  # No pilots to boost

    if n_data == 0:
        # All pilots - just return as-is (can't balance)
        return freq_block.copy()

    result = freq_block.copy()

    # Amplitude boost factor
    boost_linear = 10 ** (boost_db / 20)

    # Compute actual symbol powers
    pilot_power = np.sum(np.abs(freq_block[pilot_indices])**2)
    data_power = np.sum(np.abs(freq_block[data_indices])**2)
    total_power = pilot_power + data_power

    # To maintain total power:
    # alpha_pilot^2 * P_pilot + alpha_data^2 * P_data = P_total
    # With alpha_pilot = boost_linear * alpha_data:
    # alpha_data^2 * (boost_linear^2 * P_pilot + P_data) = P_total
    # alpha_data = sqrt(P_total / (boost_linear^2 * P_pilot + P_data))
    if pilot_power > 0:
        alpha_data = np.sqrt(total_power / (boost_linear**2 * pilot_power + data_power))
        alpha_pilot = boost_linear * alpha_data
    else:
        alpha_data = 1.0
        alpha_pilot = 1.0

    result[pilot_indices] *= alpha_pilot
    result[data_indices] *= alpha_data

    return result


def compensate_pilot_boost(h_pilots: np.ndarray, boost_db: float) -> np.ndarray:
    """
    Compensate for pilot boost in channel estimation.

    The receiver must divide pilot-based channel estimates by the
    boost factor to get the true channel.

    Args:
        h_pilots: Channel estimates at pilot positions
        boost_db: Pilot boost in dB that was applied at TX

    Returns:
        Compensated channel estimates
    """
    if boost_db == 0.0:
        return h_pilots.copy()

    boost_linear = 10 ** (boost_db / 20)
    return h_pilots / boost_linear


def _nearest_prime(n: int) -> int:
    """Find the nearest prime number to n."""
    def is_prime(x):
        if x < 2:
            return False
        if x == 2:
            return True
        if x % 2 == 0:
            return False
        for i in range(3, int(np.sqrt(x)) + 1, 2):
            if x % i == 0:
                return False
        return True

    if is_prime(n):
        return n

    # Search in both directions
    lower, upper = n - 1, n + 1
    while True:
        if is_prime(lower):
            return lower
        if is_prime(upper):
            return upper
        lower -= 1
        upper += 1


class SCFDETransmitter:
    """
    SC-FDE Transmitter.

    Generates SC-FDE blocks with:
    - Zadoff-Chu pilot sequences for channel estimation
    - Adaptive cyclic prefix for ISI mitigation
    - Adaptive pilot spacing for Nyquist criterion compliance
    - Efficient IFFT-based processing
    """

    # Predefined CP lengths for different channel conditions
    # Must be ≥ τ_max × fs to avoid ISI (with small margin)
    CP_PRESETS = {
        'benign': 260,      # 167 μs @ 1.536 MSPS + margin (need ≥ 257)
        'midlatitude': 900,  # 583 μs @ 1.536 MSPS + margin (need ≥ 896)
        'equatorial': 1540,  # 1 ms @ 1.536 MSPS + margin (need ≥ 1536)
        'auroral': 3204,    # 2.08 ms @ 1.536 MSPS + margin (need ≥ 3200)
    }

    def __init__(self, config: Optional[TransmitterConfig] = None,
                 preamble: Optional[np.ndarray] = None):
        """
        Initialize transmitter.

        Args:
            config: Transmitter configuration (uses defaults if None)
            preamble: Synchronization preamble to use in frames. Should be
                      the preamble_with_cp from the Synchronizer module.
                      If None, frames will not include a preamble.
        """
        self.config = config or TransmitterConfig()
        self.preamble = preamble

        # Validate configuration
        self._validate_config()

        # Pre-generate pilot sequence
        self.pilot_sequence = generate_zadoff_chu(
            self.config.n_pilots,
            self.config.pilot_sequence_root
        )

        # Compute pilot and data indices in FFT
        self._compute_subcarrier_indices()

    def _validate_config(self):
        """Validate configuration parameters"""
        cfg = self.config

        if cfg.fft_size not in [1024, 2048, 4096, 8192]:
            raise ValueError(f"FFT size {cfg.fft_size} not supported")

        if cfg.cp_length >= cfg.fft_size:
            raise ValueError("CP length must be less than FFT size")

        # Check that pilots + data fit in usable bandwidth
        total_carriers = cfg.n_pilots + cfg.n_data_carriers
        usable_carriers = cfg.fft_size // 2  # Assuming real baseband
        if total_carriers > usable_carriers:
            raise ValueError(
                f"Total carriers ({total_carriers}) exceeds usable ({usable_carriers})")

    def _compute_subcarrier_indices(self):
        """Compute pilot and data subcarrier indices using shared utility."""
        cfg = self.config
        self.pilot_indices, self.data_indices, self.null_indices = \
            compute_subcarrier_indices(
                cfg.fft_size,
                cfg.n_pilots,
                cfg.n_data_carriers,
                cfg.pilot_spacing
            )

    def set_cp_preset(self, preset: str):
        """
        Set CP length from preset.

        Args:
            preset: 'benign', 'midlatitude', 'equatorial', or 'auroral'
        """
        if preset not in self.CP_PRESETS:
            raise ValueError(f"Unknown preset '{preset}'. "
                           f"Valid: {list(self.CP_PRESETS.keys())}")

        self.config.cp_length = self.CP_PRESETS[preset]

        # Auroral requires larger FFT
        if preset == 'auroral' and self.config.fft_size < 8192:
            print(f"Warning: Auroral preset requires FFT≥8192, "
                  f"current is {self.config.fft_size}")

    def set_pilot_preset(self, preset: str, reconfigure: bool = True):
        """
        Set pilot spacing from preset based on Nyquist criterion.

        Args:
            preset: 'benign', 'midlatitude', 'equatorial', or 'auroral'
            reconfigure: If True, recalculate pilot/data indices

        Raises:
            ValueError: If preset is unknown
        """
        if preset not in PILOT_SPACING_PRESETS:
            raise ValueError(f"Unknown preset '{preset}'. "
                           f"Valid: {list(PILOT_SPACING_PRESETS.keys())}")

        new_spacing = PILOT_SPACING_PRESETS[preset]

        # Validate against Nyquist criterion
        delay_spread = CHANNEL_DELAY_SPREADS[preset]
        is_valid, nyquist_limit = validate_pilot_spacing(
            new_spacing, delay_spread,
            self.config.fft_size, self.config.sample_rate
        )

        if not is_valid:
            print(f"Warning: Pilot spacing {new_spacing} may not be adequate "
                  f"for {preset} channel (Nyquist limit: {nyquist_limit:.1f})")

        self.config.pilot_spacing = new_spacing

        # Recalculate n_pilots and n_data_carriers for new spacing
        # Keep total active carriers roughly the same
        if reconfigure:
            total_active = self.config.n_pilots + self.config.n_data_carriers
            # With spacing S, we get approximately total_active / (S+1) pilots
            # Use nearest prime for n_pilots if possible
            n_pilots_approx = total_active // (new_spacing + 1)

            # Find nearest prime for optimal ZC autocorrelation
            self.config.n_pilots = _nearest_prime(n_pilots_approx)
            self.config.n_data_carriers = total_active - self.config.n_pilots

            # Regenerate pilot sequence
            self.pilot_sequence = generate_zadoff_chu(
                self.config.n_pilots,
                self.config.pilot_sequence_root
            )

            # Recompute indices
            self._compute_subcarrier_indices()

    def set_channel_preset(self, preset: str, reconfigure: bool = True):
        """
        Set both CP and pilot spacing for a channel type.

        This is the recommended method for configuring the transmitter
        for different HF channel conditions.

        Args:
            preset: 'benign', 'midlatitude', 'equatorial', or 'auroral'
            reconfigure: If True, recalculate pilot/data indices
        """
        self.set_cp_preset(preset)
        self.set_pilot_preset(preset, reconfigure)

    def validate_for_channel(self, channel_type: str) -> Tuple[bool, str]:
        """
        Validate current configuration for a channel type.

        Args:
            channel_type: 'benign', 'midlatitude', 'equatorial', or 'auroral'

        Returns:
            Tuple of (is_valid, message)
        """
        if channel_type not in CHANNEL_DELAY_SPREADS:
            return False, f"Unknown channel type: {channel_type}"

        delay_spread = CHANNEL_DELAY_SPREADS[channel_type]

        # Check pilot spacing Nyquist criterion
        is_valid, nyquist_limit = validate_pilot_spacing(
            self.config.pilot_spacing, delay_spread,
            self.config.fft_size, self.config.sample_rate
        )

        if not is_valid:
            return False, (
                f"Pilot spacing {self.config.pilot_spacing} violates Nyquist "
                f"for {channel_type} (limit: {nyquist_limit:.1f})"
            )

        # Check CP length
        required_cp = int(np.ceil(delay_spread * self.config.sample_rate))
        if self.config.cp_length < required_cp:
            return False, (
                f"CP length {self.config.cp_length} is less than delay spread "
                f"{required_cp} samples for {channel_type}"
            )

        return True, f"Configuration valid for {channel_type}"

    def modulate_block(self, data_symbols: np.ndarray) -> np.ndarray:
        """
        Create one SC-FDE block from data symbols.

        Args:
            data_symbols: Complex symbols for data subcarriers

        Returns:
            Time-domain samples with CP prepended
        """
        cfg = self.config

        if len(data_symbols) != cfg.n_data_carriers:
            raise ValueError(
                f"Expected {cfg.n_data_carriers} data symbols, "
                f"got {len(data_symbols)}")

        # Create frequency-domain block
        freq_block = np.zeros(cfg.fft_size, dtype=np.complex128)

        # Insert pilots
        freq_block[self.pilot_indices] = self.pilot_sequence

        # Insert data
        freq_block[self.data_indices] = data_symbols

        # Apply pilot boost if configured
        if cfg.pilot_boost_db > 0:
            freq_block = apply_pilot_boost(
                freq_block, self.pilot_indices, self.data_indices, cfg.pilot_boost_db
            )

        # IFFT to time domain
        time_block = np.fft.ifft(freq_block) * np.sqrt(cfg.fft_size)

        # Add cyclic prefix
        cp = time_block[-cfg.cp_length:]
        tx_block = np.concatenate([cp, time_block])

        return tx_block

    def modulate_frame(self, data_symbols: np.ndarray,
                       include_preamble: bool = True) -> np.ndarray:
        """
        Create complete frame from data symbols.

        Args:
            data_symbols: All data symbols for the frame
            include_preamble: Whether to include synchronization preamble

        Returns:
            Complete time-domain frame

        Raises:
            ValueError: If include_preamble is True but no preamble was provided at init
        """
        cfg = self.config

        # Calculate number of blocks needed
        n_blocks = int(np.ceil(len(data_symbols) / cfg.n_data_carriers))

        # Pad data if necessary
        total_symbols = n_blocks * cfg.n_data_carriers
        if len(data_symbols) < total_symbols:
            padding = total_symbols - len(data_symbols)
            data_symbols = np.concatenate([
                data_symbols,
                np.zeros(padding, dtype=np.complex128)
            ])

        # Build frame
        frame_parts = []

        # Add preamble if requested
        if include_preamble:
            if self.preamble is None:
                raise ValueError(
                    "Cannot include preamble: no preamble was provided at initialization. "
                    "Pass preamble=sync.preamble_with_cp when creating SCFDETransmitter."
                )
            frame_parts.append(self.preamble)

        # Add data blocks
        for i in range(n_blocks):
            start = i * cfg.n_data_carriers
            end = start + cfg.n_data_carriers
            block_symbols = data_symbols[start:end]
            block = self.modulate_block(block_symbols)
            frame_parts.append(block)

        return np.concatenate(frame_parts)

    def get_block_duration(self) -> float:
        """Get duration of one SC-FDE block in seconds"""
        cfg = self.config
        samples = cfg.fft_size + cfg.cp_length
        return samples / cfg.sample_rate

    def get_data_rate(self, bits_per_symbol: int, code_rate: float) -> float:
        """
        Calculate data rate for given modulation and coding.

        Args:
            bits_per_symbol: Bits per QAM symbol
            code_rate: FEC code rate

        Returns:
            Data rate in bits per second
        """
        cfg = self.config
        block_duration = self.get_block_duration()
        bits_per_block = cfg.n_data_carriers * bits_per_symbol * code_rate
        return bits_per_block / block_duration

    def get_spectral_efficiency(self, bits_per_symbol: int,
                                code_rate: float) -> float:
        """
        Calculate spectral efficiency.

        Args:
            bits_per_symbol: Bits per QAM symbol
            code_rate: FEC code rate

        Returns:
            Spectral efficiency in bits/s/Hz
        """
        cfg = self.config
        # Effective bandwidth
        bandwidth = cfg.sample_rate / 2  # Nyquist
        data_rate = self.get_data_rate(bits_per_symbol, code_rate)
        return data_rate / bandwidth

    def get_papr(self, signal: np.ndarray) -> float:
        """
        Calculate Peak-to-Average Power Ratio.

        Args:
            signal: Time-domain signal

        Returns:
            PAPR in dB
        """
        peak_power = np.max(np.abs(signal) ** 2)
        avg_power = np.mean(np.abs(signal) ** 2)
        return 10 * np.log10(peak_power / avg_power)


def test_transmitter():
    """Test SC-FDE transmitter"""
    from .sync import Synchronizer, SyncConfig

    print("Testing SC-FDE Transmitter...")

    # Create synchronizer for preamble
    sync_config = SyncConfig(
        fft_size=4096,
        cp_length=1536,
        sample_rate=1.536e6,
    )
    sync = Synchronizer(sync_config)

    # Create transmitter with default config
    config = TransmitterConfig(
        fft_size=4096,
        cp_length=1536,
        n_pilots=256,
        n_data_carriers=1792,
    )
    tx = SCFDETransmitter(config, preamble=sync.preamble_with_cp)

    print(f"\nConfiguration:")
    print(f"  FFT size: {config.fft_size}")
    print(f"  CP length: {config.cp_length} samples ({config.cp_length/config.sample_rate*1e6:.1f} μs)")
    print(f"  Pilots: {config.n_pilots}")
    print(f"  Data carriers: {config.n_data_carriers}")
    print(f"  Block duration: {tx.get_block_duration()*1e3:.3f} ms")
    print(f"  Preamble length: {len(tx.preamble)} samples")

    # Test data rates for various modulations
    print(f"\nData rates (with rate 1/2 LDPC):")
    for mod, bps in [('BPSK', 1), ('QPSK', 2), ('8PSK', 3), ('16QAM', 4), ('64QAM', 6)]:
        rate = tx.get_data_rate(bps, 0.5)
        print(f"  {mod}: {rate/1e3:.1f} kbps")

    # Generate test block
    print(f"\nSingle block test:")
    np.random.seed(42)
    data_symbols = (np.random.randn(config.n_data_carriers) +
                    1j * np.random.randn(config.n_data_carriers)) / np.sqrt(2)

    block = tx.modulate_block(data_symbols)
    print(f"  Input symbols: {len(data_symbols)}")
    print(f"  Output samples: {len(block)}")
    print(f"  Expected: {config.fft_size + config.cp_length}")
    print(f"  PAPR: {tx.get_papr(block):.2f} dB")

    # Verify CP is correct
    cp = block[:config.cp_length]
    end = block[-config.cp_length:]
    cp_error = np.max(np.abs(cp - end))
    print(f"  CP verification error: {cp_error:.2e}")

    # Generate frame with preamble
    print(f"\nFrame test:")
    n_symbols = config.n_data_carriers * 3  # 3 blocks of data
    frame_symbols = (np.random.randn(n_symbols) +
                     1j * np.random.randn(n_symbols)) / np.sqrt(2)

    frame = tx.modulate_frame(frame_symbols, include_preamble=True)
    print(f"  Input symbols: {n_symbols}")
    print(f"  Output samples: {len(frame)}")
    print(f"  Frame PAPR: {tx.get_papr(frame):.2f} dB")

    # Check spectrum
    print(f"\nSpectrum check:")
    spectrum = np.fft.fft(block[config.cp_length:])  # FFT of one block without CP
    power_spectrum = np.abs(spectrum) ** 2

    # Check pilot positions have power
    pilot_power = np.mean(power_spectrum[tx.pilot_indices])
    data_power = np.mean(power_spectrum[tx.data_indices])
    null_indices = np.setdiff1d(
        np.arange(config.fft_size),
        np.concatenate([tx.pilot_indices, tx.data_indices])
    )
    null_power = np.mean(power_spectrum[null_indices])

    print(f"  Pilot power: {10*np.log10(pilot_power):.1f} dB")
    print(f"  Data power: {10*np.log10(data_power):.1f} dB")
    print(f"  Null power: {10*np.log10(null_power + 1e-10):.1f} dB")

    # Test CP presets
    print(f"\nCP presets:")
    for preset, cp_len in SCFDETransmitter.CP_PRESETS.items():
        duration_us = cp_len / config.sample_rate * 1e6
        print(f"  {preset}: {cp_len} samples ({duration_us:.1f} μs)")

    print("\nTransmitter tests passed!")
    return True


if __name__ == '__main__':
    test_transmitter()
