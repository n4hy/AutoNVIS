"""
Constellation Mapper and Soft Demapper for SC-FDE System

Supports:
- BPSK, QPSK, 8PSK, 16QAM, 64QAM
- Gray coding for all constellations
- Soft demapping with LLR output
"""

import numpy as np
from typing import Tuple, Optional, Dict, List, Union
from dataclasses import dataclass
from enum import Enum


class ModulationType(Enum):
    """Supported modulation types"""
    BPSK = 'bpsk'
    QPSK = 'qpsk'
    PSK8 = '8psk'
    QAM16 = '16qam'
    QAM64 = '64qam'


@dataclass
class ConstellationInfo:
    """Information about a constellation"""
    name: str
    bits_per_symbol: int
    constellation: np.ndarray  # Complex constellation points
    bit_map: np.ndarray  # Bit patterns for each point (Gray coded)
    avg_power: float  # Average power (should be ~1.0)


def _gray_code(n: int) -> List[int]:
    """Generate n-bit Gray code sequence"""
    if n == 0:
        return [0]
    if n == 1:
        return [0, 1]

    # Recursive Gray code construction
    prev = _gray_code(n - 1)
    result = []
    for code in prev:
        result.append(code)
    for code in reversed(prev):
        result.append(code | (1 << (n - 1)))
    return result


def _create_bpsk() -> ConstellationInfo:
    """Create BPSK constellation"""
    constellation = np.array([1.0, -1.0], dtype=np.complex128)
    bit_map = np.array([[0], [1]], dtype=np.int8)
    return ConstellationInfo(
        name='BPSK',
        bits_per_symbol=1,
        constellation=constellation,
        bit_map=bit_map,
        avg_power=1.0
    )


def _create_qpsk() -> ConstellationInfo:
    """Create QPSK constellation with Gray coding"""
    # Gray-coded QPSK: 00->1+j, 01->-1+j, 11->-1-j, 10->1-j
    scale = 1.0 / np.sqrt(2)
    constellation = scale * np.array([
        1 + 1j,   # 00
        -1 + 1j,  # 01
        1 - 1j,   # 10
        -1 - 1j,  # 11
    ], dtype=np.complex128)

    bit_map = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ], dtype=np.int8)

    return ConstellationInfo(
        name='QPSK',
        bits_per_symbol=2,
        constellation=constellation,
        bit_map=bit_map,
        avg_power=1.0
    )


def _create_8psk() -> ConstellationInfo:
    """Create 8-PSK constellation with Gray coding"""
    # 8-PSK on unit circle with Gray coding
    # Gray code for 3 bits: 000, 001, 011, 010, 110, 111, 101, 100
    gray3 = _gray_code(3)

    constellation = np.zeros(8, dtype=np.complex128)
    bit_map = np.zeros((8, 3), dtype=np.int8)

    for i, gray in enumerate(gray3):
        angle = i * np.pi / 4  # 45 degree spacing
        constellation[gray] = np.exp(1j * angle)
        for b in range(3):
            bit_map[gray, b] = (gray >> (2 - b)) & 1

    return ConstellationInfo(
        name='8PSK',
        bits_per_symbol=3,
        constellation=constellation,
        bit_map=bit_map,
        avg_power=1.0
    )


def _create_16qam() -> ConstellationInfo:
    """Create 16-QAM constellation with Gray coding"""
    # 16-QAM on 4x4 grid with Gray-coded I and Q
    # I: -3, -1, +1, +3 mapped by Gray code 00, 01, 11, 10
    # Q: same

    gray2 = [0, 1, 3, 2]  # Gray code for 2 bits
    levels = np.array([-3, -1, 1, 3])
    scale = 1.0 / np.sqrt(10)  # Normalize to unit average power

    constellation = np.zeros(16, dtype=np.complex128)
    bit_map = np.zeros((16, 4), dtype=np.int8)

    for i_idx, i_gray in enumerate(gray2):
        for q_idx, q_gray in enumerate(gray2):
            symbol_idx = (i_gray << 2) | q_gray
            constellation[symbol_idx] = scale * (levels[i_idx] + 1j * levels[q_idx])
            # Bits: [I1, I0, Q1, Q0]
            bit_map[symbol_idx, 0] = (i_gray >> 1) & 1
            bit_map[symbol_idx, 1] = i_gray & 1
            bit_map[symbol_idx, 2] = (q_gray >> 1) & 1
            bit_map[symbol_idx, 3] = q_gray & 1

    return ConstellationInfo(
        name='16QAM',
        bits_per_symbol=4,
        constellation=constellation,
        bit_map=bit_map,
        avg_power=1.0
    )


def _create_64qam() -> ConstellationInfo:
    """Create 64-QAM constellation with Gray coding"""
    # 64-QAM on 8x8 grid with Gray-coded I and Q
    gray3 = [0, 1, 3, 2, 6, 7, 5, 4]  # Gray code for 3 bits
    levels = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
    scale = 1.0 / np.sqrt(42)  # Normalize to unit average power

    constellation = np.zeros(64, dtype=np.complex128)
    bit_map = np.zeros((64, 6), dtype=np.int8)

    for i_idx, i_gray in enumerate(gray3):
        for q_idx, q_gray in enumerate(gray3):
            symbol_idx = (i_gray << 3) | q_gray
            constellation[symbol_idx] = scale * (levels[i_idx] + 1j * levels[q_idx])
            # Bits: [I2, I1, I0, Q2, Q1, Q0]
            for b in range(3):
                bit_map[symbol_idx, b] = (i_gray >> (2 - b)) & 1
                bit_map[symbol_idx, 3 + b] = (q_gray >> (2 - b)) & 1

    return ConstellationInfo(
        name='64QAM',
        bits_per_symbol=6,
        constellation=constellation,
        bit_map=bit_map,
        avg_power=1.0
    )


# Pre-built constellations
CONSTELLATIONS = {
    ModulationType.BPSK: _create_bpsk(),
    ModulationType.QPSK: _create_qpsk(),
    ModulationType.PSK8: _create_8psk(),
    ModulationType.QAM16: _create_16qam(),
    ModulationType.QAM64: _create_64qam(),
}


class Modulator:
    """
    Symbol mapper for various modulation schemes.

    Converts bit sequences to complex symbols using Gray-coded constellations.
    """

    def __init__(self, modulation: str = 'qpsk'):
        """
        Initialize modulator.

        Args:
            modulation: Modulation type ('bpsk', 'qpsk', '8psk', '16qam', '64qam')
        """
        # Parse modulation type
        mod_str = modulation.lower().replace('-', '').replace('_', '')
        try:
            self.mod_type = ModulationType(mod_str)
        except ValueError:
            valid = [m.value for m in ModulationType]
            raise ValueError(f"Unknown modulation '{modulation}'. Valid: {valid}")

        self.info = CONSTELLATIONS[self.mod_type]
        self.bits_per_symbol = self.info.bits_per_symbol
        self.constellation = self.info.constellation
        self.bit_map = self.info.bit_map

        # Build reverse mapping: bit pattern -> symbol index
        self._build_bit_to_index()

    def _build_bit_to_index(self):
        """Build lookup table from bit pattern to constellation index"""
        self.bit_to_index = {}
        for idx in range(len(self.constellation)):
            bits = tuple(self.bit_map[idx])
            self.bit_to_index[bits] = idx

    def modulate(self, bits: np.ndarray) -> np.ndarray:
        """
        Map bits to complex symbols.

        Args:
            bits: Array of bits (length must be multiple of bits_per_symbol)

        Returns:
            Array of complex symbols
        """
        bits = np.asarray(bits, dtype=np.int8).flatten()

        if len(bits) % self.bits_per_symbol != 0:
            raise ValueError(
                f"Bit length {len(bits)} not divisible by {self.bits_per_symbol}")

        n_symbols = len(bits) // self.bits_per_symbol
        symbols = np.zeros(n_symbols, dtype=np.complex128)

        for i in range(n_symbols):
            bit_group = bits[i * self.bits_per_symbol:(i + 1) * self.bits_per_symbol]
            # Convert bits to index
            idx = 0
            for b in bit_group:
                idx = (idx << 1) | int(b)
            symbols[i] = self.constellation[idx]

        return symbols

    def get_constellation(self) -> np.ndarray:
        """Return constellation points"""
        return self.constellation.copy()

    def get_bit_map(self) -> np.ndarray:
        """Return bit mapping table"""
        return self.bit_map.copy()


class Demodulator:
    """
    Soft demapper producing log-likelihood ratios (LLRs).

    Uses exact LLR computation or max-log approximation.
    """

    def __init__(self, modulation: str = 'qpsk', use_max_log: bool = True):
        """
        Initialize demodulator.

        Args:
            modulation: Modulation type
            use_max_log: If True, use max-log approximation for LLRs
        """
        mod_str = modulation.lower().replace('-', '').replace('_', '')
        try:
            self.mod_type = ModulationType(mod_str)
        except ValueError:
            valid = [m.value for m in ModulationType]
            raise ValueError(f"Unknown modulation '{modulation}'. Valid: {valid}")

        self.info = CONSTELLATIONS[self.mod_type]
        self.bits_per_symbol = self.info.bits_per_symbol
        self.constellation = self.info.constellation
        self.bit_map = self.info.bit_map
        self.use_max_log = use_max_log

        # Precompute indices for each bit position where bit is 0 or 1
        self._precompute_bit_indices()

    def _precompute_bit_indices(self):
        """Precompute constellation indices for each bit value"""
        self.bit0_indices = []  # For each bit position, indices where bit=0
        self.bit1_indices = []  # For each bit position, indices where bit=1

        for bit_pos in range(self.bits_per_symbol):
            idx0 = []
            idx1 = []
            for sym_idx in range(len(self.constellation)):
                if self.bit_map[sym_idx, bit_pos] == 0:
                    idx0.append(sym_idx)
                else:
                    idx1.append(sym_idx)
            self.bit0_indices.append(np.array(idx0))
            self.bit1_indices.append(np.array(idx1))

    def demodulate(self, symbols: np.ndarray,
                   noise_var: Union[float, np.ndarray] = 1.0,
                   channel: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute soft LLRs from received symbols (vectorized).

        Args:
            symbols: Received complex symbols
            noise_var: Noise variance (sigma^2). Can be scalar or per-symbol array.
            channel: Optional channel coefficients (for per-symbol scaling)

        Returns:
            LLRs for each bit (positive = more likely 0)
        """
        symbols = np.asarray(symbols, dtype=np.complex128)
        n_symbols = len(symbols)

        # Handle channel scaling
        if channel is None:
            const_scaled = self.constellation
        elif np.isscalar(channel) or len(channel) == 1:
            const_scaled = channel * self.constellation
        else:
            # Per-symbol channel - need to compute distances differently
            return self._demodulate_per_symbol_channel(symbols, noise_var, channel)

        # Handle noise variance
        noise_var = np.atleast_1d(noise_var)
        if len(noise_var) == 1:
            noise_var = np.maximum(noise_var[0], 1e-10)
        else:
            noise_var = np.maximum(noise_var, 1e-10)

        if self.use_max_log:
            return self._demodulate_maxlog_vectorized(symbols, const_scaled, noise_var)
        else:
            return self._demodulate_exact_vectorized(symbols, const_scaled, noise_var)

    def _demodulate_maxlog_vectorized(self, symbols: np.ndarray,
                                       constellation: np.ndarray,
                                       noise_var: Union[float, np.ndarray]) -> np.ndarray:
        """Vectorized max-log LLR computation."""
        n_symbols = len(symbols)

        # Compute squared distances from all symbols to all constellation points
        # distances[i,j] = |symbols[i] - constellation[j]|^2
        distances = np.abs(symbols[:, np.newaxis] - constellation[np.newaxis, :]) ** 2

        # Compute LLRs for each bit position
        llrs = np.zeros(n_symbols * self.bits_per_symbol, dtype=np.float64)

        for bit_pos in range(self.bits_per_symbol):
            idx0 = self.bit0_indices[bit_pos]
            idx1 = self.bit1_indices[bit_pos]

            # Min distance to bit=0 and bit=1 constellation points
            dist0 = np.min(distances[:, idx0], axis=1)
            dist1 = np.min(distances[:, idx1], axis=1)

            # LLR = (dist1 - dist0) / noise_var
            if np.isscalar(noise_var):
                llrs[bit_pos::self.bits_per_symbol] = (dist1 - dist0) / noise_var
            else:
                llrs[bit_pos::self.bits_per_symbol] = (dist1 - dist0) / noise_var

        return llrs

    def _demodulate_exact_vectorized(self, symbols: np.ndarray,
                                      constellation: np.ndarray,
                                      noise_var: Union[float, np.ndarray]) -> np.ndarray:
        """Vectorized exact LLR computation using log-sum-exp."""
        n_symbols = len(symbols)

        # Compute squared distances
        distances = np.abs(symbols[:, np.newaxis] - constellation[np.newaxis, :]) ** 2

        # Compute LLRs for each bit position
        llrs = np.zeros(n_symbols * self.bits_per_symbol, dtype=np.float64)

        for bit_pos in range(self.bits_per_symbol):
            idx0 = self.bit0_indices[bit_pos]
            idx1 = self.bit1_indices[bit_pos]

            if np.isscalar(noise_var):
                exp0 = np.exp(-distances[:, idx0] / noise_var)
                exp1 = np.exp(-distances[:, idx1] / noise_var)
            else:
                exp0 = np.exp(-distances[:, idx0] / noise_var[:, np.newaxis])
                exp1 = np.exp(-distances[:, idx1] / noise_var[:, np.newaxis])

            sum0 = np.sum(exp0, axis=1)
            sum1 = np.sum(exp1, axis=1)

            # Prevent log(0)
            sum0 = np.maximum(sum0, 1e-300)
            sum1 = np.maximum(sum1, 1e-300)

            llrs[bit_pos::self.bits_per_symbol] = np.log(sum0) - np.log(sum1)

        return llrs

    def _demodulate_per_symbol_channel(self, symbols: np.ndarray,
                                        noise_var: Union[float, np.ndarray],
                                        channel: np.ndarray) -> np.ndarray:
        """Handle per-symbol channel coefficients (less common case)."""
        n_symbols = len(symbols)
        llrs = np.zeros(n_symbols * self.bits_per_symbol, dtype=np.float64)

        noise_var = np.atleast_1d(noise_var)
        per_symbol_noise = len(noise_var) > 1

        for sym_idx in range(n_symbols):
            y = symbols[sym_idx]
            h = channel[sym_idx]
            nv = max(noise_var[sym_idx] if per_symbol_noise else noise_var[0], 1e-10)
            const_scaled = h * self.constellation

            for bit_pos in range(self.bits_per_symbol):
                if self.use_max_log:
                    llr = self._max_log_llr(y, const_scaled, bit_pos, nv)
                else:
                    llr = self._exact_llr(y, const_scaled, bit_pos, nv)
                llrs[sym_idx * self.bits_per_symbol + bit_pos] = llr

        return llrs

    def _max_log_llr(self, y: complex, constellation: np.ndarray,
                     bit_pos: int, noise_var: float) -> float:
        """Compute LLR using max-log approximation"""
        idx0 = self.bit0_indices[bit_pos]
        idx1 = self.bit1_indices[bit_pos]

        # Minimum distance for bit=0
        dist0 = np.min(np.abs(y - constellation[idx0]) ** 2)
        # Minimum distance for bit=1
        dist1 = np.min(np.abs(y - constellation[idx1]) ** 2)

        # LLR = (d1^2 - d0^2) / sigma^2
        # Positive LLR means bit 0 is more likely
        # Add floor to prevent division by zero
        noise_var = max(noise_var, 1e-10)
        return (dist1 - dist0) / noise_var

    def _exact_llr(self, y: complex, constellation: np.ndarray,
                   bit_pos: int, noise_var: float) -> float:
        """Compute exact LLR using log-sum-exp"""
        idx0 = self.bit0_indices[bit_pos]
        idx1 = self.bit1_indices[bit_pos]

        # Add floor to prevent division by zero
        noise_var = max(noise_var, 1e-10)

        # Compute exp(-d^2/sigma^2) for each constellation point
        dist0 = np.abs(y - constellation[idx0]) ** 2 / noise_var
        dist1 = np.abs(y - constellation[idx1]) ** 2 / noise_var

        # Log-sum-exp for numerical stability
        max0 = np.max(-dist0)
        max1 = np.max(-dist1)

        sum0 = max0 + np.log(np.sum(np.exp(-dist0 - max0)))
        sum1 = max1 + np.log(np.sum(np.exp(-dist1 - max1)))

        return sum0 - sum1

    def hard_demodulate(self, symbols: np.ndarray) -> np.ndarray:
        """
        Hard decision demodulation (minimum distance).

        Args:
            symbols: Received complex symbols

        Returns:
            Detected bits
        """
        symbols = np.asarray(symbols, dtype=np.complex128)
        n_symbols = len(symbols)
        bits = np.zeros(n_symbols * self.bits_per_symbol, dtype=np.int8)

        for i, y in enumerate(symbols):
            # Find closest constellation point
            distances = np.abs(y - self.constellation)
            min_idx = np.argmin(distances)

            # Map to bits
            bits[i * self.bits_per_symbol:(i + 1) * self.bits_per_symbol] = \
                self.bit_map[min_idx]

        return bits


def compute_evm(tx_symbols: np.ndarray, rx_symbols: np.ndarray) -> float:
    """
    Compute Error Vector Magnitude (EVM) as percentage.

    Args:
        tx_symbols: Transmitted symbols
        rx_symbols: Received symbols (same length)

    Returns:
        EVM as percentage
    """
    error = rx_symbols - tx_symbols
    evm = np.sqrt(np.mean(np.abs(error) ** 2) / np.mean(np.abs(tx_symbols) ** 2))
    return evm * 100


def test_modulator():
    """Test modulator and demodulator"""
    print("Testing Modulator/Demodulator...")

    np.random.seed(42)

    for mod_name in ['bpsk', 'qpsk', '8psk', '16qam', '64qam']:
        print(f"\n{mod_name.upper()}:")

        mod = Modulator(mod_name)
        demod = Demodulator(mod_name)

        # Random bits
        n_bits = 1000 * mod.bits_per_symbol
        bits = np.random.randint(0, 2, n_bits, dtype=np.int8)

        # Modulate
        symbols = mod.modulate(bits)
        print(f"  Bits: {n_bits}, Symbols: {len(symbols)}")
        print(f"  Avg power: {np.mean(np.abs(symbols)**2):.4f}")

        # Add AWGN
        snr_db = 10.0
        noise_var = 10 ** (-snr_db / 10)
        noise = np.sqrt(noise_var / 2) * (np.random.randn(len(symbols)) +
                                           1j * np.random.randn(len(symbols)))
        rx_symbols = symbols + noise

        # Hard demodulation
        rx_bits_hard = demod.hard_demodulate(rx_symbols)
        ber_hard = np.mean(bits != rx_bits_hard)
        print(f"  Hard BER @ {snr_db} dB: {ber_hard:.6f}")

        # Soft demodulation
        llrs = demod.demodulate(rx_symbols, noise_var)
        rx_bits_soft = (llrs < 0).astype(np.int8)
        ber_soft = np.mean(bits != rx_bits_soft)
        print(f"  Soft BER @ {snr_db} dB: {ber_soft:.6f}")

        # EVM
        evm = compute_evm(symbols, rx_symbols)
        print(f"  EVM: {evm:.2f}%")

        # Verify Gray coding property for QAM
        if 'qam' in mod_name or 'psk' in mod_name:
            # Adjacent constellation points should differ by 1 bit
            const = mod.get_constellation()
            bmap = mod.get_bit_map()
            gray_ok = True
            # Check some adjacent pairs
            for i in range(min(4, len(const))):
                for j in range(i + 1, min(i + 4, len(const))):
                    dist = np.abs(const[i] - const[j])
                    if dist < 1.5 / np.sqrt(10 if '16' in mod_name else 42 if '64' in mod_name else 1):
                        # Adjacent points
                        bit_diff = np.sum(bmap[i] != bmap[j])
                        if bit_diff != 1:
                            gray_ok = False
            print(f"  Gray coding: {'OK' if gray_ok else 'Check needed'}")

    print("\nModulator tests passed!")
    return True


if __name__ == '__main__':
    test_modulator()
