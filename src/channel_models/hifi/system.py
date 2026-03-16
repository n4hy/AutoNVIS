"""
SC-FDE Complete System Integration

Full TX/RX chain:
- LDPC encoding
- Interleaving
- Modulation (BPSK to 64QAM)
- SC-FDE block construction
- Channel (AWGN or HF multipath)
- Synchronization
- MMSE equalization
- Soft demodulation
- LDPC decoding
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass, field

from .ldpc import get_ldpc_codec
from .modulator import Modulator, Demodulator
from .crc import append_crc, check_crc
from .interleaver import BitInterleaver, FrequencyInterleaver, DataSymbolFrequencyInterleaver
from .transmitter import SCFDETransmitter, TransmitterConfig
from .receiver import SCFDEReceiver, ReceiverConfig
from .sync import Synchronizer, SyncConfig
from .turbo_equalizer import TurboEqualizer, TurboEqualizerConfig, run_turbo_decode
from .feature_config import FeatureConfig
from .scrambler import BitScrambler


@dataclass
class SystemConfig:
    """Complete SC-FDE system configuration"""
    # Sample rate
    sample_rate: float = 1.536e6

    # FFT parameters
    fft_size: int = 4096
    cp_length: int = 1536

    # Subcarrier allocation (defaults match benign channel)
    n_pilots: int = 251  # Prime for optimal ZC autocorrelation
    n_data_carriers: int = 1757
    pilot_spacing: int = 8

    # Modulation and coding
    modulation: str = 'qpsk'
    ldpc_n: int = 648
    ldpc_rate: float = 0.5

    # Interleaving
    use_interleaving: bool = True

    # CRC for error detection
    use_crc: bool = True

    # SNR estimate for MMSE
    snr_estimate: float = 10.0

    @classmethod
    def narrowband(cls, **kwargs) -> 'SystemConfig':
        """
        Narrowband config for benign/midlatitude channels.

        FFT=4096, good for delay spreads up to ~500 μs.
        """
        defaults = dict(
            fft_size=4096,
            cp_length=1536,
            n_pilots=251,     # Prime for optimal ZC autocorrelation
            n_data_carriers=1757,
            pilot_spacing=8,
        )
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def wideband(cls, **kwargs) -> 'SystemConfig':
        """
        Wideband config for equatorial channels.

        FFT=8192, handles delay spreads up to ~1 ms.
        Nyquist limit: 8192/1536 ≈ 5.3, so spacing=4 is used.
        """
        defaults = dict(
            fft_size=8192,
            cp_length=1536,   # 1 ms at 1.536 MHz
            n_pilots=821,     # Prime, ~4x increase for denser spacing
            n_data_carriers=3275,
            pilot_spacing=4,  # Satisfies Nyquist for 1 ms delay spread
        )
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def polar(cls, **kwargs) -> 'SystemConfig':
        """
        Config for polar/auroral channels with multi-mode propagation.

        FFT=8192 with denser pilots (spacing=2) to handle delay spreads
        up to 1.5 ms. Nyquist limit: 8192/2304 ≈ 3.6, so spacing=2 satisfies.

        With 1367 pilots at spacing 2, pilots span indices 1-2733.
        Data carriers interleaved: (1367-1) * (2-1) = 1366
        """
        defaults = dict(
            fft_size=8192,
            cp_length=2304,   # 1.5 ms at 1.536 MHz
            n_pilots=1367,    # Prime, for spacing=2
            n_data_carriers=1366,  # Fixed: was 2729, caused 1363 carriers beyond pilots
            pilot_spacing=2,
        )
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def extreme(cls, **kwargs) -> 'SystemConfig':
        """
        Config for extreme conditions: auroral spread-F, severe disturbance.

        FFT=8192 with high pilot density (spacing=2) to handle delay
        spreads up to 2 ms. Uses 2 ms CP for the large delay spread.

        With 2039 pilots at spacing 2, pilots span indices 1-4077.
        Data carriers interleaved: (2039-1) * (2-1) = 2038
        Total = 4077 ≤ 4096 (half of FFT)

        Note: spacing=1 doesn't work with interpolation as all data
        would be placed after all pilots with no surrounding pilots.

        Recommended modulation: BPSK or QPSK only.
        """
        defaults = dict(
            fft_size=8192,
            cp_length=3072,   # 2 ms at 1.536 MHz for large delay spreads
            n_pilots=2039,    # Prime, fits within usable carriers
            n_data_carriers=2038,  # Fixed: was 2043 with spacing=1
            pilot_spacing=2,  # Fixed: was 1, which put all data after pilots
        )
        defaults.update(kwargs)
        return cls(**defaults)

    # Channel-specific presets based on Vogler-Hoffmeyer characterization
    @classmethod
    def benign(cls, **kwargs) -> 'SystemConfig':
        """
        Config for benign HF channels (minimal multipath).

        Delay spread: ~167 μs
        Nyquist limit for spacing: ~16 (using 8 for margin)

        With 251 pilots at spacing 8, pilots span indices 1-2001.
        Data carriers interleaved: (251-1) * (8-1) = 1750
        """
        defaults = dict(
            fft_size=4096,
            cp_length=260,    # 167 μs @ 1.536 MSPS + margin
            n_pilots=251,     # Prime
            n_data_carriers=1750,  # Fixed: was 1757, 7 carriers were beyond pilots
            pilot_spacing=8,
        )
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def midlatitude(cls, **kwargs) -> 'SystemConfig':
        """
        Config for midlatitude HF channels (moderate multipath).

        Delay spread: ~583 μs
        Nyquist limit for spacing: ~4.6 (using 4)

        With 401 pilots at spacing 4, pilots span indices 1-1601.
        Data carriers must stay within this range for interpolation.
        Data within range = (401-1) * (4-1) = 1200
        """
        defaults = dict(
            fft_size=4096,
            cp_length=900,    # 583 μs @ 1.536 MSPS + margin
            n_pilots=401,     # Prime, increased for denser spacing
            n_data_carriers=1200,  # Fixed: was 1607, caused 407 carriers beyond pilots
            pilot_spacing=4,
        )
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def equatorial(cls, **kwargs) -> 'SystemConfig':
        """
        Config for equatorial spread-F channels (severe multipath).

        Delay spread: ~1 ms
        Nyquist limit for spacing: ~2.7 (using 2)

        With 673 pilots at spacing 2, pilots span indices 1-1345.
        Data carriers must stay within this range for interpolation.
        Data within range = (673-1) * (2-1) = 672
        """
        defaults = dict(
            fft_size=4096,
            cp_length=1540,   # 1 ms @ 1.536 MSPS + margin
            n_pilots=673,     # Prime, ~3x increase for spacing=2
            n_data_carriers=672,  # Fixed: was 1335, caused 663 carriers beyond pilots
            pilot_spacing=2,
        )
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def auroral(cls, **kwargs) -> 'SystemConfig':
        """
        Config for auroral/polar channels (extreme multipath).

        Delay spread: ~2 ms
        Nyquist limit for spacing: ~1.3 (using 2 for proper interpolation)
        Requires FFT=8192 for adequate resolution.

        With 2039 pilots at spacing 2, pilots span indices 1-4077.
        Data carriers interleaved: (2039-1) * (2-1) = 2038
        Total = 2039 + 2038 = 4077 ≤ 4096 (half of FFT)
        Note: spacing=1 doesn't work with interpolation as all data
        would be after all pilots with no surrounding pilots.
        """
        defaults = dict(
            fft_size=8192,
            cp_length=3204,   # 2.08 ms @ 1.536 MSPS + margin
            n_pilots=2039,    # Prime, fits within usable carriers
            n_data_carriers=2038,  # (n_pilots-1) * (spacing-1)
            pilot_spacing=2,  # Fixed: was 1, which put all data after pilots
        )
        defaults.update(kwargs)
        return cls(**defaults)


@dataclass
class TransmitResult:
    """Result of transmission"""
    tx_signal: np.ndarray
    tx_symbols: np.ndarray
    info_bits: np.ndarray
    coded_bits: np.ndarray
    n_blocks: int


@dataclass
class ReceiveResult:
    """Result of reception"""
    rx_bits: np.ndarray
    decoded_bits: np.ndarray
    eq_symbols: np.ndarray
    channel_est: np.ndarray
    ber: float
    fer: float
    n_bit_errors: int
    converged: bool
    crc_valid: bool = True  # True if CRC check passed (or CRC not used)


class SCFDESystem:
    """
    Complete SC-FDE transceiver system.

    Integrates all components for end-to-end transmission.
    """

    def __init__(self, config: Optional[SystemConfig] = None,
                 feature_config: Optional[FeatureConfig] = None):
        """
        Initialize system.

        Args:
            config: System configuration
            feature_config: Feature toggles (None = default with recommended features enabled)
        """
        self.config = config or SystemConfig()
        self.features = feature_config or FeatureConfig()
        self._setup_components()

    def _setup_components(self):
        """Initialize all subsystem components"""
        cfg = self.config
        feat = self.features

        # LDPC codec - use feature config block length if specified
        ldpc_n = feat.ldpc_block_length if feat.ldpc_block_length > 0 else cfg.ldpc_n
        self.ldpc = get_ldpc_codec(n=ldpc_n, rate=cfg.ldpc_rate,
                                   prefer_aff3ct=True)

        # Modulator/Demodulator
        self.modulator = Modulator(cfg.modulation)
        self.demodulator = Demodulator(cfg.modulation)
        self.bits_per_symbol = self.modulator.bits_per_symbol

        # Bit interleaver (sized for one LDPC codeword)
        # Controlled by system config - always operates on coded bits
        if cfg.use_interleaving:
            self.interleaver = BitInterleaver(ldpc_n)
        else:
            self.interleaver = None

        # Frequency interleaver (sized for actual data symbols, NOT carrier count)
        # Controlled by feature config - operates on data symbols only
        #
        # IMPORTANT: We interleave the actual data symbols (e.g., 324 for QPSK/LDPC-648)
        # NOT the full carrier allocation (e.g., 3275). This prevents the problem where
        # interleaving zero-padded blocks causes fade-corrupted zeros to scatter among
        # real data after de-interleaving.
        #
        # Symbols per codeword = ldpc_n / bits_per_symbol
        self.freq_interleaver = None
        self._n_symbols_per_codeword = ldpc_n // self.bits_per_symbol
        if feat.use_freq_interleaving:
            # Use data-symbol-sized interleaver
            self.freq_interleaver = DataSymbolFrequencyInterleaver(
                self._n_symbols_per_codeword
            )

        # Bit scrambler (randomizes coded bits to reduce PAPR)
        # Placed after LDPC encode, before interleaving (TX)
        # Reverses after de-interleaving, before LDPC decode (RX)
        self.scrambler = None
        if feat.use_scrambling:
            self.scrambler = BitScrambler(seed=feat.scrambler_seed)

        # Synchronizer (create first so transmitter can use its preamble)
        # Pass subcarrier allocation so preamble respects guard bands
        sync_config = SyncConfig(
            fft_size=cfg.fft_size,
            cp_length=cfg.cp_length,
            sample_rate=cfg.sample_rate,
            n_pilots=cfg.n_pilots,
            n_data_carriers=cfg.n_data_carriers,
            pilot_spacing=cfg.pilot_spacing,
        )
        self.sync = Synchronizer(sync_config)

        # Transmitter - with pilot boost from feature config
        tx_config = TransmitterConfig(
            sample_rate=cfg.sample_rate,
            fft_size=cfg.fft_size,
            cp_length=cfg.cp_length,
            n_pilots=cfg.n_pilots,
            n_data_carriers=cfg.n_data_carriers,
            pilot_spacing=cfg.pilot_spacing,
            pilot_boost_db=feat.pilot_boost_db,  # Wire pilot boost
        )
        self.tx = SCFDETransmitter(tx_config, preamble=self.sync.preamble_with_cp)

        # Receiver - with feature toggles
        rx_config = ReceiverConfig(
            sample_rate=cfg.sample_rate,
            fft_size=cfg.fft_size,
            cp_length=cfg.cp_length,
            n_pilots=cfg.n_pilots,
            n_data_carriers=cfg.n_data_carriers,
            pilot_spacing=cfg.pilot_spacing,
            snr_estimate=cfg.snr_estimate,
            n_channel_paths=feat.sparse_threshold_n_paths,  # Sparse thresholding
            pilot_boost_db=feat.pilot_boost_db,  # Wire pilot boost compensation
            use_mmse=feat.use_mmse_estimation,  # MMSE channel estimation
        )
        self.rx = SCFDEReceiver(rx_config)

        # AGC (optional, based on feature config)
        self.agc = None
        if feat.use_agc:
            try:
                from .agc import FastAGC
                self.agc = FastAGC(
                    target_level_dbfs=feat.agc_target_dbfs,
                    sample_rate=cfg.sample_rate,
                )
            except ImportError:
                pass  # AGC module not available

        # DD Tracker (optional, based on feature config)
        self.dd_tracker = None
        if feat.use_dd_tracking:
            try:
                from .dd_tracker import DDChannelTracker
                self.dd_tracker = DDChannelTracker(
                    n_carriers=cfg.n_data_carriers,
                    base_alpha=feat.dd_alpha,
                )
            except ImportError:
                pass  # DD tracker module not available

        # Link Adapter (optional, based on feature config)
        self.link_adapter = None
        if feat.use_link_adaptation:
            try:
                from .link_adaptation import LinkAdapter
                self.link_adapter = LinkAdapter()
            except ImportError:
                pass  # Link adaptation module not available

        # CFO Tracker (optional, based on feature config)
        self.cfo_tracker = None
        if feat.use_cfo_tracking:
            try:
                from .cfo_tracker import CFOTracker
                block_duration_ms = (cfg.fft_size + cfg.cp_length) / cfg.sample_rate * 1000
                self.cfo_tracker = CFOTracker(
                    sample_rate=cfg.sample_rate,
                    block_duration_ms=block_duration_ms,
                    loop_bandwidth_hz=feat.cfo_loop_bandwidth_hz,
                )
            except ImportError:
                pass  # CFO tracker module not available

        # Turbo equalizer (eagerly initialized when enabled, otherwise lazy)
        self._turbo_eq = None
        if feat.use_turbo_equalization:
            self._turbo_eq = self._create_turbo_equalizer()

        # Calculate system parameters
        self._calculate_parameters()

    def _create_turbo_equalizer(self) -> TurboEqualizer:
        """Create a new turbo equalizer using FeatureConfig settings."""
        cfg = self.config
        feat = self.features

        turbo_config = TurboEqualizerConfig(
            max_iterations=feat.turbo_iterations,
            # If early_termination disabled, set min_iterations = max to force all iterations
            min_iterations=1 if feat.turbo_early_termination else feat.turbo_iterations,
            convergence_threshold=1e-3 if feat.turbo_early_termination else float('inf'),
        )

        # Use the same LDPC block length as the system
        ldpc_n = self.ldpc.n

        turbo_eq = TurboEqualizer(
            modulation=cfg.modulation,
            ldpc_n=ldpc_n,
            ldpc_rate=cfg.ldpc_rate,
            config=turbo_config,
            ldpc_codec=self.ldpc,
        )
        turbo_eq.set_snr(cfg.snr_estimate)
        return turbo_eq

    def _calculate_parameters(self):
        """Calculate derived system parameters"""
        cfg = self.config

        # Bits per SC-FDE block
        self.bits_per_block = cfg.n_data_carriers * self.bits_per_symbol

        # Info bits per codeword (accounting for CRC overhead)
        info_bits_per_codeword = self.ldpc.k - 16 if cfg.use_crc else self.ldpc.k

        # Info bits per block (after LDPC decoding and CRC removal)
        codewords_per_block = self.bits_per_block // cfg.ldpc_n
        self.info_bits_per_block = codewords_per_block * info_bits_per_codeword

        # Data rate
        block_duration = (cfg.fft_size + cfg.cp_length) / cfg.sample_rate
        self.data_rate = self.info_bits_per_block / block_duration

    def transmit(self, info_bits: np.ndarray,
                 include_preamble: bool = True) -> TransmitResult:
        """
        Complete transmit chain.

        Args:
            info_bits: Information bits to transmit
            include_preamble: Whether to include sync preamble

        Returns:
            TransmitResult with signal and metadata
        """
        cfg = self.config

        # CRC reduces effective info bits per codeword by 16
        info_bits_per_codeword = self.ldpc.k - 16 if cfg.use_crc else self.ldpc.k

        # Pad info bits to multiple of info_bits_per_codeword
        n_codewords = int(np.ceil(len(info_bits) / info_bits_per_codeword))
        total_info = n_codewords * info_bits_per_codeword
        if len(info_bits) < total_info:
            info_bits = np.concatenate([
                info_bits,
                np.zeros(total_info - len(info_bits), dtype=np.int8)
            ])

        # LDPC encode (with optional CRC per codeword)
        coded_bits = []
        for i in range(n_codewords):
            start = i * info_bits_per_codeword
            end = start + info_bits_per_codeword
            block_info = info_bits[start:end]

            if cfg.use_crc:
                # Add CRC-16 to info bits (now ldpc.k bits total)
                block_with_crc = append_crc(block_info)
                codeword = self.ldpc.encode(block_with_crc)
            else:
                codeword = self.ldpc.encode(block_info)

            coded_bits.append(codeword)
        coded_bits = np.concatenate(coded_bits)

        # Scramble coded bits (per-codeword with unique offset for each codeword)
        if self.scrambler is not None:
            ldpc_n = self.ldpc.n
            scrambled = []
            for i in range(n_codewords):
                start = i * ldpc_n
                end = start + ldpc_n
                scrambled.append(self.scrambler.scramble(coded_bits[start:end], codeword_idx=i))
            coded_bits = np.concatenate(scrambled)

        # Interleave (use actual LDPC block length)
        ldpc_n = self.ldpc.n
        if self.interleaver:
            interleaved_bits = []
            for i in range(n_codewords):
                start = i * ldpc_n
                end = start + ldpc_n
                interleaved = self.interleaver.interleave(coded_bits[start:end])
                interleaved_bits.append(interleaved)
            coded_bits = np.concatenate(interleaved_bits)

        # Modulate
        symbols = self.modulator.modulate(coded_bits)

        # Frequency interleaving (per-codeword, on actual data symbols only)
        # IMPORTANT: Interleave BEFORE zero-padding to carriers.
        # This prevents fade-corrupted zeros from scattering into real data.
        #
        # Each LDPC codeword produces n_symbols_per_codeword symbols.
        # We interleave each codeword's symbols independently.
        if self.freq_interleaver is not None:
            interleaved = []
            n_syms = self._n_symbols_per_codeword
            n_codeword_blocks = len(symbols) // n_syms
            for cw_idx in range(n_codeword_blocks):
                start = cw_idx * n_syms
                end = start + n_syms
                interleaved.append(self.freq_interleaver.interleave(symbols[start:end]))
            # Handle any remaining symbols (partial codeword - unlikely but safe)
            remaining = len(symbols) % n_syms
            if remaining > 0:
                # Partial block - pad, interleave, then trim back
                partial = symbols[-remaining:]
                padded = np.concatenate([partial, np.zeros(n_syms - remaining, dtype=np.complex128)])
                interleaved_padded = self.freq_interleaver.interleave(padded)
                interleaved.append(interleaved_padded[:remaining])
            symbols = np.concatenate(interleaved) if interleaved else symbols

        # Pad symbols to multiple of data carriers per block (AFTER interleaving)
        # Use scrambled padding to avoid PAPR spikes from constant symbols
        n_blocks = int(np.ceil(len(symbols) / cfg.n_data_carriers))
        total_symbols = n_blocks * cfg.n_data_carriers
        n_pad = total_symbols - len(symbols)
        if n_pad > 0:
            if self.scrambler is not None:
                # Generate random padding symbols using deterministic seed
                # Seed based on frame content hash for reproducibility
                # This ensures padding is random but receiver can reconstruct if needed
                pad_seed = int(np.sum(coded_bits[:min(64, len(coded_bits))]) + n_codewords * 7919)
                pad_rng = np.random.RandomState(pad_seed & 0x7FFFFFFF)
                pad_bits = pad_rng.randint(0, 2, n_pad * self.bits_per_symbol, dtype=np.int8)
                pad_symbols = self.modulator.modulate(pad_bits)
            else:
                # No scrambler, use zeros (original behavior)
                pad_symbols = np.zeros(n_pad, dtype=np.complex128)
            symbols = np.concatenate([symbols, pad_symbols])

        # SC-FDE frame construction
        tx_signal = self.tx.modulate_frame(symbols, include_preamble)

        return TransmitResult(
            tx_signal=tx_signal,
            tx_symbols=symbols,
            info_bits=info_bits,
            coded_bits=coded_bits,
            n_blocks=n_blocks,
        )

    def _apply_dd_tracking(self, rx_signal: np.ndarray, eq_symbols: np.ndarray,
                           channel_est: np.ndarray, n_blocks: int,
                           cfg: 'SystemConfig') -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply decision-directed channel tracking per block.

        Args:
            rx_signal: Raw received signal
            eq_symbols: Initial equalized symbols from pilot-based estimation
            channel_est: Initial channel estimate
            n_blocks: Number of SC-FDE blocks
            cfg: System configuration

        Returns:
            Tuple of (refined_eq_symbols, refined_channel_est)
        """
        # Initialize DD tracker with first block's channel estimate
        block_size = cfg.n_data_carriers
        if len(channel_est) >= block_size:
            self.dd_tracker.initialize(channel_est[:block_size])
        else:
            # Not enough channel estimates, return unchanged
            return eq_symbols, channel_est

        # Process each block
        # Note: rx_signal already has preamble removed at this point
        block_len = cfg.fft_size + cfg.cp_length
        all_refined_symbols = []
        all_refined_channel = []

        for block_idx in range(n_blocks):
            # Get equalized symbols for this block
            sym_start = block_idx * block_size
            sym_end = sym_start + block_size

            if sym_end > len(eq_symbols):
                break

            block_eq = eq_symbols[sym_start:sym_end]

            # Make hard decisions for DD feedback
            hard_decisions = self._hard_decide(block_eq)

            # Get raw received data for this block
            block_start = block_idx * block_len
            block_end = block_start + block_len

            if block_end > len(rx_signal):
                all_refined_symbols.append(block_eq)
                all_refined_channel.append(channel_est[sym_start:sym_end])
                continue

            rx_block = rx_signal[block_start:block_end]
            rx_no_cp = rx_block[cfg.cp_length:]
            rx_freq = np.fft.fft(rx_no_cp) / np.sqrt(cfg.fft_size)
            rx_data = rx_freq[self.rx.data_indices]

            # DD update with hard decisions
            self.dd_tracker.update_dd(rx_data, hard_decisions, snr_db=cfg.snr_estimate)

            # Re-equalize with refined channel estimate
            h_refined = self.dd_tracker.get_estimate()
            refined_eq = self.rx.equalizer.equalize(rx_data, h_refined)[0]

            all_refined_symbols.append(refined_eq)
            all_refined_channel.append(h_refined)

        # Concatenate results
        if all_refined_symbols:
            return np.concatenate(all_refined_symbols), np.concatenate(all_refined_channel)
        return eq_symbols, channel_est

    def _hard_decide(self, eq_symbols: np.ndarray) -> np.ndarray:
        """Make hard decisions for DD feedback based on modulation."""
        # Get constellation points from modulator
        constellation = self.modulator.constellation
        # Find nearest constellation point for each symbol
        distances = np.abs(eq_symbols[:, np.newaxis] - constellation[np.newaxis, :])
        nearest_idx = np.argmin(distances, axis=1)
        return constellation[nearest_idx]

    def _extract_pilot_phase(self, rx_freq: np.ndarray) -> float:
        """
        Extract average pilot phase for CFO tracking.

        Computes the phase difference between received pilots and
        known transmitted pilot sequence.

        Args:
            rx_freq: Received frequency-domain symbols (one block)

        Returns:
            Average pilot phase in radians [-pi, pi]
        """
        # Get received pilots
        rx_pilots = rx_freq[self.tx.pilot_indices]

        # Known transmitted pilots (with any pilot boost already applied)
        tx_pilots = self.tx.pilot_sequence

        # Compute phase difference: arg(rx * conj(tx))
        # This gives the phase rotation applied by the channel + CFO
        phase_diff = np.angle(rx_pilots * np.conj(tx_pilots))

        # Return average phase (simple approach - could use weighted avg)
        return np.mean(phase_diff)

    def _apply_cfo_correction(self, rx_signal: np.ndarray, n_blocks: int,
                               cfg: 'SystemConfig') -> np.ndarray:
        """
        Apply CFO tracking and correction to received signal.

        For each block:
        1. Extract pilot phase
        2. Update CFO tracker
        3. Apply phase correction to the block

        Args:
            rx_signal: Raw received signal (after AGC/sync)
            n_blocks: Number of blocks in frame
            cfg: System configuration

        Returns:
            CFO-corrected signal
        """
        if self.cfo_tracker is None:
            return rx_signal

        # Make a copy to avoid modifying original
        rx_corrected = rx_signal.copy()

        # Note: rx_signal already has preamble removed at this point
        block_len = cfg.fft_size + cfg.cp_length

        for block_idx in range(n_blocks):
            block_start = block_idx * block_len
            block_end = block_start + block_len

            if block_end > len(rx_signal):
                break

            # Extract this block and convert to frequency domain
            rx_block = rx_signal[block_start:block_end]
            rx_no_cp = rx_block[cfg.cp_length:]
            rx_freq = np.fft.fft(rx_no_cp) / np.sqrt(cfg.fft_size)

            # Extract pilot phase and update tracker
            pilot_phase = self._extract_pilot_phase(rx_freq)
            self.cfo_tracker.update(pilot_phase)

            # Apply correction to this block (time domain)
            # Note: We correct the entire block including CP
            corrected_block = self.cfo_tracker.apply_correction(
                rx_block,
                sample_offset=block_idx * block_len
            )

            rx_corrected[block_start:block_end] = corrected_block

        return rx_corrected

    def receive(self, rx_signal: np.ndarray,
                tx_result: TransmitResult,
                use_sync: bool = False,
                noise_var: Optional[float] = None,
                use_per_subcarrier_snr: bool = True,
                null_subcarrier_threshold: float = 0.0) -> ReceiveResult:
        """
        Complete receive chain.

        Args:
            rx_signal: Received signal
            tx_result: Transmit result (for reference)
            use_sync: Whether to use synchronization
            noise_var: Noise variance (for soft demod), ignored if use_per_subcarrier_snr=True
            use_per_subcarrier_snr: Use per-subcarrier post-MMSE SNR for LLR computation
            null_subcarrier_threshold: If > 0, erase (set LLR=0) subcarriers with
                post-MMSE SNR below this threshold. Typical value: 0.1 to 1.0

        Returns:
            ReceiveResult with decoded bits and metrics
        """
        cfg = self.config
        feat = self.features

        # When turbo equalization is enabled at init, use the turbo receive path
        if self._turbo_eq is not None:
            return self._receive_turbo(rx_signal, tx_result)

        # AGC: Apply automatic gain control if enabled
        if self.agc is not None:
            rx_signal = self.agc.process(rx_signal)

        # Synchronization (if enabled)
        if use_sync:
            try:
                data_start, cfo, rx_signal = self.sync.synchronize(rx_signal)
                # Extract data portion - sync returns absolute data_start position
                rx_signal = rx_signal[data_start:]
            except ValueError:
                # Sync failed, assume data starts after preamble
                rx_signal = rx_signal[self.sync.preamble_length:]
        else:
            # Skip preamble directly
            if tx_result.n_blocks > 0:
                rx_signal = rx_signal[self.sync.preamble_length:]

        # CFO tracking and correction (if enabled)
        if self.cfo_tracker is not None:
            rx_signal = self._apply_cfo_correction(rx_signal, tx_result.n_blocks, cfg)

        # SC-FDE demodulation with post-equalization SNR
        # Data signal now starts at first data block (preamble already removed)
        demod_result = self.rx.demodulate_frame(
            rx_signal,
            tx_result.n_blocks,
            return_post_snr=use_per_subcarrier_snr
        )

        if use_per_subcarrier_snr:
            eq_symbols, channel_est, post_snr = demod_result
        else:
            eq_symbols, channel_est = demod_result
            post_snr = None

        # Decision-directed channel tracking (optional)
        if self.dd_tracker is not None:
            eq_symbols, channel_est = self._apply_dd_tracking(
                rx_signal, eq_symbols, channel_est, tx_result.n_blocks, cfg
            )

        # Trim to expected length
        expected_symbols = len(tx_result.tx_symbols)
        if len(eq_symbols) > expected_symbols:
            eq_symbols = eq_symbols[:expected_symbols]
            channel_est = channel_est[:expected_symbols]
            if post_snr is not None:
                post_snr = post_snr[:expected_symbols]

        # Frequency de-interleaving (per-codeword, on actual data symbols only)
        # Reverses the TX-side interleaving before soft demodulation
        #
        # IMPORTANT: We de-interleave per-codeword (n_symbols_per_codeword),
        # NOT per-block (n_data_carriers). This matches the TX interleaving.
        if self.freq_interleaver is not None:
            n_syms = self._n_symbols_per_codeword
            # Trim to actual data symbols (remove zero-padding)
            # The number of actual symbols = n_codewords * n_symbols_per_codeword
            n_actual_symbols = (len(eq_symbols) // n_syms) * n_syms
            eq_symbols = eq_symbols[:n_actual_symbols]
            if post_snr is not None:
                post_snr = post_snr[:n_actual_symbols]

            n_codeword_blocks = n_actual_symbols // n_syms
            deinterleaved_symbols = []
            deinterleaved_post_snr = [] if post_snr is not None else None

            for cw_idx in range(n_codeword_blocks):
                start = cw_idx * n_syms
                end = start + n_syms
                deinterleaved_symbols.append(
                    self.freq_interleaver.deinterleave(eq_symbols[start:end])
                )
                if post_snr is not None:
                    deinterleaved_post_snr.append(
                        self.freq_interleaver.deinterleave(post_snr[start:end])
                    )

            if deinterleaved_symbols:
                eq_symbols = np.concatenate(deinterleaved_symbols)
                if post_snr is not None:
                    post_snr = np.concatenate(deinterleaved_post_snr)

        # Compute per-subcarrier noise variance for soft demodulation
        if use_per_subcarrier_snr and post_snr is not None:
            # Post-MMSE noise variance = 1 / post_snr
            # Clamp to avoid division by zero for very low SNR subcarriers
            post_snr_clamped = np.maximum(post_snr, 1e-6)
            per_symbol_noise_var = 1.0 / post_snr_clamped
        else:
            # Fall back to global noise variance
            if noise_var is None:
                noise_var = 10 ** (-cfg.snr_estimate / 10)
            per_symbol_noise_var = noise_var

        # Soft demodulation with per-subcarrier noise variance
        # After MMSE equalization, symbols are approximately x * α + noise_enhanced
        # where α = |H|²/(|H|² + noise_var) is a REAL scaling factor.
        # Don't pass the complex channel H - pass None (unity) since MMSE equalizes to unity.
        llrs = self.demodulator.demodulate(eq_symbols, per_symbol_noise_var, channel=None)

        # Soft erasure: weight LLRs by subcarrier reliability
        # Check feature config first, then fall back to parameter
        soft_erasure_threshold = feat.soft_erasure_threshold_db
        if soft_erasure_threshold is None and null_subcarrier_threshold > 0:
            # Use parameter as fallback (convert linear threshold to dB)
            soft_erasure_threshold = 10 * np.log10(null_subcarrier_threshold)

        if soft_erasure_threshold is not None and post_snr is not None:
            # Import soft erasure functions
            from .receiver import compute_reliability_weights, apply_soft_erasure

            # Compute smooth reliability weights using sigmoid
            weights = compute_reliability_weights(
                post_snr,
                threshold_db=soft_erasure_threshold,
                steepness=0.5
            )

            # Expand weights to per-bit (each symbol has bits_per_symbol LLRs)
            bits_per_sym = self.bits_per_symbol
            bit_weights = np.repeat(weights, bits_per_sym)[:len(llrs)]

            # Apply soft erasure
            llrs = apply_soft_erasure(llrs, bit_weights)

        # Trim LLRs to expected coded bits
        expected_coded = len(tx_result.coded_bits)
        if len(llrs) > expected_coded:
            llrs = llrs[:expected_coded]

        # De-interleave (use actual LDPC block length, not config)
        ldpc_n = self.ldpc.n
        if self.interleaver:
            n_codewords = len(llrs) // ldpc_n
            deinterleaved = []
            for i in range(n_codewords):
                start = i * ldpc_n
                end = start + ldpc_n
                if end <= len(llrs):
                    block = self.interleaver.deinterleave_llr(llrs[start:end])
                    deinterleaved.append(block)
            llrs = np.concatenate(deinterleaved) if deinterleaved else llrs

        # Descramble LLRs (per-codeword with unique offset matching TX)
        if self.scrambler is not None:
            n_codewords = len(llrs) // ldpc_n
            descrambled = []
            for i in range(n_codewords):
                start = i * ldpc_n
                end = start + ldpc_n
                if end <= len(llrs):
                    descrambled.append(self.scrambler.descramble_llr(llrs[start:end], codeword_idx=i))
            llrs = np.concatenate(descrambled) if descrambled else llrs

        # LDPC decode and CRC check
        decoded_bits = []
        all_converged = True
        all_crc_valid = True
        n_codewords = len(llrs) // ldpc_n

        # Use batched decoding if available (AFF3CTLDPCCodec)
        if hasattr(self.ldpc, 'decode_batch') and n_codewords > 1:
            # Batch decode all codewords at once (SIMD parallel)
            llrs_2d = llrs[:n_codewords * ldpc_n].reshape(n_codewords, ldpc_n)
            decoded_codewords, converged_arr = self.ldpc.decode_batch(llrs_2d)
            all_converged = converged_arr.all()

            for i in range(n_codewords):
                info_with_crc = self.ldpc.get_info_bits(decoded_codewords[i])

                if cfg.use_crc:
                    info_bits_block, crc_valid = check_crc(info_with_crc)
                    decoded_bits.append(info_bits_block)
                    if not crc_valid:
                        all_crc_valid = False
                else:
                    decoded_bits.append(info_with_crc)
        else:
            # Sequential decoding fallback
            for i in range(n_codewords):
                start = i * ldpc_n
                end = start + ldpc_n
                if end <= len(llrs):
                    decoded, converged = self.ldpc.decode(
                        llrs[start:end],
                        max_iterations=feat.ldpc_max_iterations,
                        early_termination=feat.ldpc_early_termination
                    )
                    info_with_crc = self.ldpc.get_info_bits(decoded)

                    if cfg.use_crc:
                        info_bits_block, crc_valid = check_crc(info_with_crc)
                        decoded_bits.append(info_bits_block)
                        if not crc_valid:
                            all_crc_valid = False
                    else:
                        decoded_bits.append(info_with_crc)

                    if not converged:
                        all_converged = False

        decoded_bits = np.concatenate(decoded_bits) if decoded_bits else np.array([])

        # Calculate BER
        ref_bits = tx_result.info_bits[:len(decoded_bits)]
        n_errors = np.sum(ref_bits != decoded_bits)
        ber = n_errors / len(ref_bits) if len(ref_bits) > 0 else 1.0

        # Hard decision bits (before LDPC)
        rx_bits = (llrs < 0).astype(np.int8)

        return ReceiveResult(
            rx_bits=rx_bits,
            decoded_bits=decoded_bits,
            eq_symbols=eq_symbols,
            channel_est=channel_est,
            ber=ber,
            fer=0.0 if all_converged else 1.0,
            n_bit_errors=n_errors,
            converged=all_converged,
            crc_valid=all_crc_valid,
        )

    def _receive_turbo(self, rx_signal: np.ndarray,
                       tx_result: TransmitResult) -> ReceiveResult:
        """
        Internal turbo equalization receive path.

        Iterates between MMSE-SIC equalization and LDPC decoding,
        using soft symbol feedback to improve equalization.
        """
        cfg = self.config

        turbo_eq = self._turbo_eq
        turbo_eq.set_snr(cfg.snr_estimate)

        # AGC (matching default receive path)
        if self.agc is not None:
            rx_signal = self.agc.process(rx_signal)

        # Skip preamble
        if tx_result.n_blocks > 0:
            rx_signal = rx_signal[self.sync.preamble_length:]

        # CFO correction
        if self.cfo_tracker is not None:
            rx_signal = self._apply_cfo_correction(rx_signal, tx_result.n_blocks, cfg)

        # Demodulate all blocks, collecting raw freq-domain data and channel estimates
        block_len = cfg.fft_size + cfg.cp_length
        all_rx_data = []
        all_channels = []
        all_eq_symbols = []

        for block_idx in range(tx_result.n_blocks):
            block_start = block_idx * block_len
            block_end = block_start + block_len

            if block_end > len(rx_signal):
                break

            rx_block = rx_signal[block_start:block_end]

            # Get equalized symbols, channel, and raw data in one pass
            eq_result = self.rx.demodulate_block(
                rx_block, return_channel=True, return_rx_data=True)
            eq_symbols, channel, rx_data = eq_result[0], eq_result[1], eq_result[2]
            all_eq_symbols.append(eq_symbols)

            all_rx_data.append(rx_data)
            all_channels.append(channel)

        if not all_rx_data:
            return ReceiveResult(
                rx_bits=np.array([], dtype=np.int8),
                decoded_bits=np.array([], dtype=np.int8),
                eq_symbols=np.array([], dtype=np.complex128),
                channel_est=np.array([], dtype=np.complex128),
                ber=1.0, fer=1.0, n_bit_errors=0, converged=False,
            )

        # Concatenate all blocks and trim to actual data symbols (remove padding)
        rx_data = np.concatenate(all_rx_data)
        channel = np.concatenate(all_channels)
        eq_symbols = np.concatenate(all_eq_symbols)

        n_data_symbols = len(tx_result.coded_bits) // self.bits_per_symbol
        rx_data = rx_data[:n_data_symbols]
        channel = channel[:n_data_symbols]

        # Run turbo equalization + CRC stripping
        result = run_turbo_decode(
            turbo_eq, rx_data, channel,
            ldpc_k=self.ldpc.k,
            use_crc=cfg.use_crc,
            interleaver=self.interleaver,
            freq_interleaver=self.freq_interleaver,
            scrambler=self.scrambler,
            n_symbols_per_codeword=self._n_symbols_per_codeword,
        )
        decoded_bits = result.decoded_bits
        converged = result.converged

        # Calculate BER
        ref_bits = tx_result.info_bits[:len(decoded_bits)]
        n_errors = np.sum(ref_bits != decoded_bits)
        ber = n_errors / len(ref_bits) if len(ref_bits) > 0 else 1.0

        rx_bits = np.zeros(len(decoded_bits), dtype=np.int8)

        return ReceiveResult(
            rx_bits=rx_bits,
            decoded_bits=decoded_bits,
            eq_symbols=eq_symbols,
            channel_est=channel,
            ber=ber,
            fer=0.0 if converged else 1.0,
            n_bit_errors=n_errors,
            converged=converged,
            crc_valid=result.crc_valid,
        )

    def simulate_awgn(self, info_bits: np.ndarray,
                      snr_db: float) -> ReceiveResult:
        """
        Simulate transmission through AWGN channel.

        Args:
            info_bits: Information bits
            snr_db: Channel SNR in dB

        Returns:
            ReceiveResult
        """
        # Transmit
        tx_result = self.transmit(info_bits, include_preamble=True)

        # Add AWGN
        snr_linear = 10 ** (snr_db / 10)
        signal_power = np.mean(np.abs(tx_result.tx_signal) ** 2)
        noise_power = signal_power / snr_linear
        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(len(tx_result.tx_signal)) +
            1j * np.random.randn(len(tx_result.tx_signal)))

        rx_signal = tx_result.tx_signal + noise

        # Update SNR estimate
        self.rx.set_snr_estimate(snr_db)

        # Receive
        return self.receive(rx_signal, tx_result, use_sync=False,
                           noise_var=noise_power)

    def get_throughput(self) -> float:
        """Get system throughput in bits/second"""
        return self.data_rate

    def get_spectral_efficiency(self) -> float:
        """Get spectral efficiency in bits/s/Hz"""
        return self.data_rate / (self.config.sample_rate / 2)


def test_system():
    """Test complete SC-FDE system"""
    print("Testing SC-FDE System...")

    # Test configurations
    configs = [
        ('BPSK rate 1/2', SystemConfig(modulation='bpsk', ldpc_rate=0.5)),
        ('QPSK rate 1/2', SystemConfig(modulation='qpsk', ldpc_rate=0.5)),
        ('16QAM rate 1/2', SystemConfig(modulation='16qam', ldpc_rate=0.5)),
    ]

    np.random.seed(42)

    for name, cfg in configs:
        print(f"\n{name}:")
        system = SCFDESystem(cfg)

        print(f"  Info bits per block: {system.info_bits_per_block}")
        print(f"  Data rate: {system.get_throughput()/1e3:.1f} kbps")
        print(f"  Spectral efficiency: {system.get_spectral_efficiency():.2f} b/s/Hz")

        # Generate random info bits
        n_info_bits = system.ldpc.k * 4  # 4 codewords
        info_bits = np.random.randint(0, 2, n_info_bits, dtype=np.int8)

        # Test at various SNRs
        for snr_db in [5, 10, 15]:
            result = system.simulate_awgn(info_bits, snr_db)
            print(f"  SNR {snr_db:2d} dB: BER={result.ber:.2e}, "
                  f"Errors={result.n_bit_errors}, "
                  f"Converged={result.converged}")

    # Loopback test (no channel)
    print("\nLoopback test (perfect channel):")
    cfg = SystemConfig(modulation='qpsk', ldpc_rate=0.5)
    system = SCFDESystem(cfg)

    info_bits = np.random.randint(0, 2, system.ldpc.k * 2, dtype=np.int8)
    tx_result = system.transmit(info_bits)
    rx_result = system.receive(tx_result.tx_signal, tx_result, use_sync=False,
                               noise_var=1e-10)

    print(f"  TX info bits: {len(info_bits)}")
    print(f"  RX decoded bits: {len(rx_result.decoded_bits)}")
    print(f"  BER: {rx_result.ber:.2e}")
    print(f"  Converged: {rx_result.converged}")

    print("\nSystem tests completed!")
    return True


if __name__ == '__main__':
    test_system()
