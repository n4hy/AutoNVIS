"""
Turbo Equalization for SC-FDE

Implements iterative MMSE-SIC equalization with LDPC soft feedback.

The turbo equalizer iterates between:
1. MMSE-SIC equalization (using soft symbol estimates)
2. Soft demodulation
3. LDPC decoding (soft output)
4. Soft symbol estimation from extrinsic LLRs

This provides significant gain (2-4 dB) for frequency-selective channels
by exploiting the decoder's soft information to improve equalization.
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass

from .modulator import Modulator, Demodulator
from .ldpc import get_ldpc_codec
from .interleaver import BitInterleaver


@dataclass
class TurboEqualizerConfig:
    """Configuration for turbo equalizer"""
    max_iterations: int = 4
    min_iterations: int = 2
    convergence_threshold: float = 1e-3  # BER change threshold for early stopping
    damping_factor: float = 0.5  # Soft symbol damping (0=no feedback, 1=full feedback)


class SoftSymbolEstimator:
    """
    Compute soft symbol estimates from LLRs.

    For QPSK: s_soft = tanh(LLR_I/2) + j*tanh(LLR_Q/2)
    For higher-order: weighted sum over constellation points
    """

    def __init__(self, modulation: str):
        """
        Initialize soft symbol estimator.

        Args:
            modulation: Modulation type ('bpsk', 'qpsk', '8psk', '16qam', '64qam')
        """
        self.modulation = modulation.lower()
        self.modulator = Modulator(modulation)
        self.constellation = self.modulator.constellation
        self.bits_per_symbol = self.modulator.bits_per_symbol
        self.bit_map = self.modulator.bit_map

    def estimate(self, llrs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute soft symbol estimates and variances from LLRs.

        Fully vectorized: operates on all symbols in parallel via numpy
        broadcasting.  No Python-level loops over symbols.

        Args:
            llrs: Log-likelihood ratios (positive = bit 0 more likely)

        Returns:
            Tuple of (soft_symbols, symbol_variances)
        """
        bps = self.bits_per_symbol
        n_symbols = len(llrs) // bps

        # Reshape LLRs -> (n_symbols, bits_per_symbol)
        llrs_2d = llrs[:n_symbols * bps].reshape(n_symbols, bps)

        # Bit probabilities P(b=0) via sigmoid: (n_symbols, bps)
        bit_probs_0 = 1.0 / (1.0 + np.exp(-np.clip(llrs_2d, -20, 20)))
        bit_probs_1 = 1.0 - bit_probs_0

        # Symbol probabilities via broadcasting:
        #   bit_map: (M, bps)  — 0/1 for each constellation point & bit position
        #   factor[n, m, k] = P(b_k = bit_map[m,k])
        #                    = bit_probs_0[n,k] when bit_map[m,k]==0
        #                    = bit_probs_1[n,k] when bit_map[m,k]==1
        #
        # Shapes: bp0/bp1 (n_symbols, 1, bps), bm (1, M, bps)
        bp0 = bit_probs_0[:, np.newaxis, :]  # (N, 1, bps)
        bp1 = bit_probs_1[:, np.newaxis, :]  # (N, 1, bps)
        bm = self.bit_map[np.newaxis, :, :]  # (1, M, bps)

        factors = bp0 * (1 - bm) + bp1 * bm  # (N, M, bps)

        # Product over bits -> symbol probabilities: (N, M)
        sym_probs = np.prod(factors, axis=2)

        # Normalize (should already sum to 1; guard for numerical stability)
        sym_probs /= np.sum(sym_probs, axis=1, keepdims=True) + 1e-10

        # E[s] = sum_m P(m) * s_m  ->  matrix-vector product (N, M) @ (M,)
        soft_symbols = sym_probs @ self.constellation

        # Var(s) = E[|s|²] - |E[s]|²
        const_power = np.abs(self.constellation) ** 2  # (M,)
        mean_power = sym_probs @ const_power  # (N,)
        symbol_vars = np.maximum(0.0, mean_power - np.abs(soft_symbols) ** 2)

        return soft_symbols, symbol_vars


class MMSESICEqualizer:
    """
    MMSE Equalizer with Soft Interference Cancellation (SIC).

    Uses soft symbol estimates from decoder to cancel interference
    before equalization, then combines with MMSE output.

    Output: s_hat = W*(y - H*s_soft) + s_soft
    where W is the MMSE filter and s_soft are soft symbol estimates.
    """

    def __init__(self, snr_linear: float = 10.0):
        """
        Initialize MMSE-SIC equalizer.

        Args:
            snr_linear: Linear SNR estimate
        """
        self.snr_linear = snr_linear
        self.regularization = 1.0 / snr_linear

    def set_snr(self, snr_db: float):
        """Set SNR from dB value"""
        self.snr_linear = 10 ** (snr_db / 10)
        self.regularization = 1.0 / self.snr_linear

    def equalize(self, rx_symbols: np.ndarray,
                 channel: np.ndarray,
                 soft_symbols: Optional[np.ndarray] = None,
                 symbol_vars: Optional[np.ndarray] = None,
                 damping: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply MMSE-SIC equalization.

        Args:
            rx_symbols: Received frequency-domain symbols
            channel: Channel frequency response estimate
            soft_symbols: Soft symbol estimates from decoder (None for first iteration)
            symbol_vars: Symbol variance estimates (for refined MMSE)
            damping: Damping factor for soft feedback (0-1)

        Returns:
            Tuple of (equalized_symbols, post_snr)
        """
        h_mag_sq = np.abs(channel) ** 2

        if soft_symbols is None:
            # First iteration: standard MMSE
            w = np.conj(channel) / (h_mag_sq + self.regularization)
            eq_symbols = rx_symbols * w
        else:
            # MMSE-SIC: subtract soft interference, then equalize
            # y_sic = y - H * s_soft
            # s_hat = W * y_sic + s_soft

            soft_symbols_damped = damping * soft_symbols

            # Interference-cancelled received signal
            y_sic = rx_symbols - channel * soft_symbols_damped

            # Compute MMSE filter (optionally accounting for residual interference)
            if symbol_vars is not None:
                # Refined regularization accounting for symbol variance
                # Residual interference power ~ |H|² * var(s)
                residual_var = h_mag_sq * symbol_vars * damping**2
                reg = self.regularization + residual_var
            else:
                reg = self.regularization

            w = np.conj(channel) / (h_mag_sq + reg)

            # Equalize and add back soft estimate
            eq_symbols = y_sic * w + soft_symbols_damped

        # Compute post-equalization SNR
        post_snr = h_mag_sq * self.snr_linear / (1 + h_mag_sq * self.snr_linear)

        return eq_symbols, post_snr


class TurboEqualizer:
    """
    Turbo Equalizer for SC-FDE.

    Iterates between MMSE-SIC equalization and LDPC decoding,
    using soft symbol estimates as feedback to improve equalization.
    """

    def __init__(self, modulation: str, ldpc_n: int, ldpc_rate: float,
                 config: Optional[TurboEqualizerConfig] = None,
                 ldpc_codec=None):
        """
        Initialize turbo equalizer.

        Args:
            modulation: Modulation type
            ldpc_n: LDPC codeword length
            ldpc_rate: LDPC code rate
            config: Turbo equalizer configuration
            ldpc_codec: LDPC codec to use (None = auto-select via factory).
                        Pass an explicit codec to ensure encoding/decoding use
                        the same systematic bit positions.
        """
        self.config = config or TurboEqualizerConfig()
        self.modulation = modulation

        # Components
        self.soft_estimator = SoftSymbolEstimator(modulation)
        self.demodulator = Demodulator(modulation)
        self.ldpc = ldpc_codec if ldpc_codec is not None else get_ldpc_codec(n=ldpc_n, rate=ldpc_rate)
        self.equalizer = MMSESICEqualizer()

        self.bits_per_symbol = self.demodulator.bits_per_symbol
        self.ldpc_n = ldpc_n

    def set_snr(self, snr_db: float):
        """Set SNR estimate"""
        self.equalizer.set_snr(snr_db)

    def process(self, rx_symbols: np.ndarray,
                channel: np.ndarray,
                interleaver: Optional[BitInterleaver] = None,
                freq_interleaver=None,
                scrambler=None,
                n_symbols_per_codeword: Optional[int] = None
                ) -> Tuple[np.ndarray, np.ndarray, int, bool]:
        """
        Run turbo equalization.

        Args:
            rx_symbols: Received frequency-domain symbols after CP removal and FFT
            channel: Channel frequency response at data subcarriers
            interleaver: Bit interleaver (if used)
            freq_interleaver: Frequency-domain symbol interleaver (if used)
            scrambler: Bit scrambler (if used)
            n_symbols_per_codeword: Symbols per LDPC codeword (needed for freq interleaving)

        Returns:
            Tuple of (decoded_bits, llrs, n_iterations, converged)
        """
        cfg = self.config
        n_symbols = len(rx_symbols)
        n_bits = n_symbols * self.bits_per_symbol

        soft_symbols = None
        symbol_vars = None
        prev_decoded_bits = None  # Track previous iteration's decoded bits

        # Per-codeword latch: once a codeword's LDPC converges, preserve its
        # decoded bits and extrinsic LLRs so later iterations cannot corrupt it.
        latched_decoded: dict[int, np.ndarray] = {}   # cw_index -> info bits
        latched_extrinsic: dict[int, np.ndarray] = {} # cw_index -> extrinsic LLRs

        for iteration in range(cfg.max_iterations):
            # MMSE-SIC equalization (carrier/freq-interleaved domain)
            damping = cfg.damping_factor if iteration > 0 else 0.0
            eq_symbols, post_snr = self.equalizer.equalize(
                rx_symbols, channel, soft_symbols, symbol_vars, damping
            )

            # Frequency de-interleave per codeword (carrier → codeword order)
            if freq_interleaver is not None and n_symbols_per_codeword is not None:
                eq_symbols, post_snr = self._freq_deinterleave_pair(
                    eq_symbols, post_snr, freq_interleaver, n_symbols_per_codeword
                )

            # Per-subcarrier noise variance
            post_snr_clamped = np.maximum(post_snr, 1e-6)
            noise_var = 1.0 / post_snr_clamped

            # Soft demodulation
            llrs = self.demodulator.demodulate(eq_symbols, noise_var, channel=None)

            # Bit de-interleave if needed
            if interleaver is not None:
                n_codewords = len(llrs) // self.ldpc_n
                deinterleaved = []
                for i in range(n_codewords):
                    start = i * self.ldpc_n
                    end = start + self.ldpc_n
                    if end <= len(llrs):
                        block = interleaver.deinterleave_llr(llrs[start:end])
                        deinterleaved.append(block)
                llrs_dec = np.concatenate(deinterleaved) if deinterleaved else llrs
            else:
                llrs_dec = llrs

            # Descramble per codeword (descramble_llr is self-inverse)
            if scrambler is not None:
                n_codewords = len(llrs_dec) // self.ldpc_n
                descrambled = []
                for i in range(n_codewords):
                    start = i * self.ldpc_n
                    end = start + self.ldpc_n
                    if end <= len(llrs_dec):
                        descrambled.append(
                            scrambler.descramble_llr(llrs_dec[start:end], codeword_idx=i)
                        )
                llrs_dec = np.concatenate(descrambled) if descrambled else llrs_dec

            # LDPC decode
            decoded_bits = []
            extrinsic_llrs = []
            all_converged = True
            n_codewords = len(llrs_dec) // self.ldpc_n

            for i in range(n_codewords):
                start = i * self.ldpc_n
                end = start + self.ldpc_n
                if end <= len(llrs_dec):
                    if i in latched_decoded:
                        # This codeword already converged — reuse latched result
                        decoded_bits.append(latched_decoded[i])
                        extrinsic_llrs.append(latched_extrinsic[i])
                    else:
                        # Decode with extrinsic output
                        decoded, converged, ext_llrs = self.ldpc.decode_with_extrinsic(
                            llrs_dec[start:end]
                        )
                        info_bits = self.ldpc.get_info_bits(decoded)
                        decoded_bits.append(info_bits)
                        extrinsic_llrs.append(ext_llrs)
                        if converged:
                            latched_decoded[i] = info_bits
                            latched_extrinsic[i] = ext_llrs
                        else:
                            all_converged = False

            if not decoded_bits:
                return np.array([]), llrs, iteration + 1, False

            decoded_bits = np.concatenate(decoded_bits)
            extrinsic_llrs = np.concatenate(extrinsic_llrs)

            # Check convergence criteria
            # 1. LDPC syndrome convergence
            if all_converged and iteration >= cfg.min_iterations - 1:
                return decoded_bits, llrs, iteration + 1, True

            # 2. BER-based convergence: check if decoded bits have stabilized
            if prev_decoded_bits is not None and iteration >= cfg.min_iterations - 1:
                n_bits_compare = min(len(decoded_bits), len(prev_decoded_bits))
                if n_bits_compare > 0:
                    bit_changes = np.sum(
                        decoded_bits[:n_bits_compare] != prev_decoded_bits[:n_bits_compare]
                    )
                    change_rate = bit_changes / n_bits_compare

                    if change_rate < cfg.convergence_threshold:
                        return decoded_bits, llrs, iteration + 1, True

            prev_decoded_bits = decoded_bits.copy()

            # Prepare soft symbols for next iteration
            if iteration < cfg.max_iterations - 1:
                # Re-scramble extrinsic LLRs (descramble_llr is self-inverse)
                if scrambler is not None:
                    rescrambled = []
                    for i in range(n_codewords):
                        start = i * self.ldpc_n
                        end = start + self.ldpc_n
                        if end <= len(extrinsic_llrs):
                            rescrambled.append(
                                scrambler.descramble_llr(extrinsic_llrs[start:end], codeword_idx=i)
                            )
                    extrinsic_llrs = np.concatenate(rescrambled) if rescrambled else extrinsic_llrs

                # Bit re-interleave extrinsic LLRs
                if interleaver is not None:
                    interleaved = []
                    for i in range(n_codewords):
                        start = i * self.ldpc_n
                        end = start + self.ldpc_n
                        if end <= len(extrinsic_llrs):
                            block = interleaver.interleave_llr(extrinsic_llrs[start:end])
                            interleaved.append(block)
                    llrs_fb = np.concatenate(interleaved) if interleaved else extrinsic_llrs
                else:
                    llrs_fb = extrinsic_llrs

                # Compute soft symbol estimates from available LLRs
                n_available_bits = min(len(llrs_fb), n_bits)
                n_available_symbols = n_available_bits // self.bits_per_symbol

                if n_available_symbols > 0:
                    soft_symbols_partial, symbol_vars_partial = self.soft_estimator.estimate(
                        llrs_fb[:n_available_symbols * self.bits_per_symbol]
                    )

                    # Frequency re-interleave (codeword → carrier order)
                    if freq_interleaver is not None and n_symbols_per_codeword is not None:
                        soft_symbols_partial = self._freq_interleave(
                            soft_symbols_partial, freq_interleaver, n_symbols_per_codeword
                        )
                        symbol_vars_partial = self._freq_interleave(
                            symbol_vars_partial, freq_interleaver, n_symbols_per_codeword
                        )

                    # Pad to match full symbol count (use zeros for symbols without LLR info)
                    if len(soft_symbols_partial) < n_symbols:
                        soft_symbols = np.zeros(n_symbols, dtype=np.complex128)
                        symbol_vars = np.ones(n_symbols, dtype=np.float64)  # High variance = no info
                        soft_symbols[:len(soft_symbols_partial)] = soft_symbols_partial
                        symbol_vars[:len(symbol_vars_partial)] = symbol_vars_partial
                    else:
                        soft_symbols = soft_symbols_partial[:n_symbols]
                        symbol_vars = symbol_vars_partial[:n_symbols]
                else:
                    soft_symbols = None
                    symbol_vars = None

        return decoded_bits, llrs, cfg.max_iterations, all_converged

    @staticmethod
    def _freq_deinterleave_pair(symbols, post_snr, freq_interleaver, n_per_cw):
        """Frequency de-interleave both symbols and post-SNR per codeword."""
        n_cw = len(symbols) // n_per_cw
        deint_syms = []
        deint_snr = []
        for i in range(n_cw):
            s, e = i * n_per_cw, (i + 1) * n_per_cw
            deint_syms.append(freq_interleaver.deinterleave(symbols[s:e]))
            deint_snr.append(freq_interleaver.deinterleave(post_snr[s:e]))
        assert len(symbols) % n_per_cw == 0, (
            f"Symbol count {len(symbols)} is not a multiple of n_per_cw {n_per_cw}; "
            "partial codewords would bypass interleaving in the turbo feedback path"
        )
        return np.concatenate(deint_syms), np.concatenate(deint_snr)

    @staticmethod
    def _freq_interleave(data, freq_interleaver, n_per_cw):
        """Frequency interleave data per codeword."""
        n_cw = len(data) // n_per_cw
        parts = []
        for i in range(n_cw):
            s, e = i * n_per_cw, (i + 1) * n_per_cw
            parts.append(freq_interleaver.interleave(data[s:e]))
        assert len(data) % n_per_cw == 0, (
            f"Data length {len(data)} is not a multiple of n_per_cw {n_per_cw}; "
            "partial codewords would bypass interleaving in the turbo feedback path"
        )
        return np.concatenate(parts)


@dataclass
class TurboDecodeResult:
    """Result from turbo decode + CRC strip pipeline."""
    decoded_bits: np.ndarray
    llrs: np.ndarray
    n_iterations: int
    converged: bool
    crc_valid: bool


def run_turbo_decode(
    turbo_eq: TurboEqualizer,
    rx_data: np.ndarray,
    channel: np.ndarray,
    ldpc_k: int,
    use_crc: bool,
    interleaver=None,
    freq_interleaver=None,
    scrambler=None,
    n_symbols_per_codeword: Optional[int] = None,
) -> TurboDecodeResult:
    """
    Run turbo equalization and CRC stripping.

    Shared pipeline used by both SCFDESystem._receive_turbo() and
    StreamingReceiver._process_frame_turbo().

    Args:
        turbo_eq: Configured TurboEqualizer (with SNR already set)
        rx_data: Pre-equalization data symbols, trimmed to actual data length
        channel: Channel estimates at data subcarriers, same length as rx_data
        ldpc_k: Number of info bits per LDPC codeword (including CRC if present)
        use_crc: Whether to perform per-codeword CRC stripping
        interleaver: Bit interleaver (if used)
        freq_interleaver: Frequency-domain symbol interleaver (if used)
        scrambler: Bit scrambler (if used)
        n_symbols_per_codeword: Symbols per LDPC codeword (needed for freq interleaving)

    Returns:
        TurboDecodeResult with decoded bits, LLRs, iteration count,
        convergence flag, and CRC validity.
    """
    decoded, llrs, n_iter, converged = turbo_eq.process(
        rx_data, channel,
        interleaver=interleaver,
        freq_interleaver=freq_interleaver,
        scrambler=scrambler,
        n_symbols_per_codeword=n_symbols_per_codeword,
    )

    # CRC stripping
    all_crc_valid = True
    if use_crc:
        from .crc import check_crc
        n_codewords = len(decoded) // ldpc_k
        checked_bits = []
        for i in range(n_codewords):
            start = i * ldpc_k
            end = start + ldpc_k
            if end <= len(decoded):
                info_block, crc_valid = check_crc(decoded[start:end])
                checked_bits.append(info_block)
                if not crc_valid:
                    all_crc_valid = False
        decoded_bits = np.concatenate(checked_bits) if checked_bits else np.array([], dtype=np.int8)
    else:
        decoded_bits = decoded

    return TurboDecodeResult(
        decoded_bits=decoded_bits,
        llrs=llrs,
        n_iterations=n_iter,
        converged=converged,
        crc_valid=all_crc_valid,
    )


def test_turbo_equalizer():
    """Test turbo equalizer"""
    print("Testing Turbo Equalizer...")

    # Create components
    modulation = 'qpsk'
    ldpc_n = 648
    ldpc_rate = 0.5

    turbo_eq = TurboEqualizer(modulation, ldpc_n, ldpc_rate)
    turbo_eq.set_snr(15)

    # Generate test data
    np.random.seed(42)
    ldpc = get_ldpc_codec(n=ldpc_n, rate=ldpc_rate)
    modulator = Modulator(modulation)

    # Random info bits
    info_bits = np.random.randint(0, 2, ldpc.k, dtype=np.int8)
    coded_bits = ldpc.encode(info_bits)
    symbols = modulator.modulate(coded_bits)

    # Simple 2-tap channel
    n_sym = len(symbols)
    h = np.ones(n_sym, dtype=np.complex128)
    h[::10] = 0.3  # Some deep fades

    # Received symbols (with channel and noise)
    snr_linear = 10 ** (15 / 10)
    noise_power = 1 / snr_linear
    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(n_sym) + 1j * np.random.randn(n_sym)
    )
    rx_symbols = h * symbols + noise

    # Run turbo equalizer
    decoded, llrs, n_iter, converged = turbo_eq.process(rx_symbols, h)

    # Check results
    n_errors = np.sum(decoded != info_bits)
    ber = n_errors / len(info_bits)

    print(f"  Iterations: {n_iter}")
    print(f"  Converged: {converged}")
    print(f"  BER: {ber:.2e}")
    print(f"  Errors: {n_errors}/{len(info_bits)}")

    print("\nTurbo equalizer test passed!")
    return True


if __name__ == '__main__':
    test_turbo_equalizer()
