#!/usr/bin/env python3
"""
Streaming Receiver for SC-FDE System

Provides a streaming interface for receiving SC-FDE frames from a continuous
sample stream (e.g., from a real radio receiver).

Usage:
    from hifi.streaming_receiver import StreamingReceiver
    from hifi.system import SCFDESystem, SystemConfig

    system = SCFDESystem(SystemConfig())
    receiver = StreamingReceiver(system, n_blocks=4)

    # Process incoming sample chunks
    while True:
        samples = get_samples_from_radio()  # Variable length chunks
        results = receiver.receive_streaming(samples)
        for result in results:
            # Process decoded frame
            print(f"Decoded {len(result.decoded_bits)} bits, BER={result.ber}")
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING
from enum import Enum

from .turbo_equalizer import run_turbo_decode

if TYPE_CHECKING:
    from .system import SCFDESystem, ReceiveResult


class ReceiverState(Enum):
    """State of the streaming receiver."""
    SEARCHING = "searching"      # Looking for preamble
    FRAME_DETECTED = "detected"  # Preamble found, accumulating frame samples


@dataclass
class StreamingReceiveResult:
    """Result from streaming receiver for one detected frame."""
    decoded_bits: np.ndarray
    eq_symbols: np.ndarray
    llr: np.ndarray
    converged: bool
    n_iterations: int
    n_bit_errors: int  # Only valid if reference bits provided
    ber: float         # Only valid if reference bits provided
    frame_start_sample: int  # Sample index where this frame started (in total stream)
    cfo_hz: float      # Estimated CFO for this frame
    crc_valid: bool = True  # True if all per-codeword CRCs passed (or CRC disabled)


class StreamingReceiver:
    """
    Streaming receiver for SC-FDE frames.

    Efficiently processes a continuous stream of samples, detecting preambles
    and decoding frames as they arrive. Handles:
    - Variable-length input chunks
    - Preambles spanning chunk boundaries
    - Partial frames waiting for more samples
    - Sequential frame detection (finishes one frame before searching for next)
    """

    def __init__(self, system: 'SCFDESystem', n_blocks: int,
                 sync_threshold: float = 0.7,
                 noise_var: Optional[float] = None,
                 num_codewords: Optional[int] = None,
                 noise_floor: Optional[float] = None,
                 min_detection_snr_db: float = 6.0,
                 use_turbo: bool = False):
        """
        Initialize streaming receiver.

        Args:
            system: Configured SCFDESystem instance
            n_blocks: Number of data blocks per frame
            sync_threshold: Preamble detection threshold (0-1)
            noise_var: Noise variance for soft demodulation (None = estimate from SNR)
            num_codewords: Number of LDPC codewords to decode (None = decode all)
            noise_floor: Noise floor power for detection filtering (None = no filtering)
            min_detection_snr_db: Minimum SNR above noise floor for valid detection
            use_turbo: Enable turbo equalization (requires system._turbo_eq)
        """
        self.system = system
        self.use_turbo = use_turbo
        if use_turbo and system._turbo_eq is None:
            raise ValueError(
                "use_turbo=True requires system to be initialized with "
                "FeatureConfig(use_turbo_equalization=True)"
            )
        self.n_blocks = n_blocks
        self.sync_threshold = sync_threshold
        self.noise_var = noise_var
        self.num_codewords = num_codewords
        self.noise_floor = noise_floor
        self.min_detection_snr_db = min_detection_snr_db

        # Cache frequently used values
        cfg = system.config
        self.fft_size = cfg.fft_size
        self.cp_length = cfg.cp_length
        self.block_len = cfg.fft_size + cfg.cp_length
        self.preamble_length = system.sync.preamble_length

        # Frame size: preamble + n_blocks * block_length
        self.frame_samples = self.preamble_length + n_blocks * self.block_len

        # Timing margin for extraction - allows for coarse timing errors
        # The Schmidl-Cox rising edge detection can be off by roughly
        # the preamble CP length (~1024 samples). Additionally, the fine
        # timing search can find the actual preamble deeper into the signal
        # (up to fine_timing_search_range = 1500 samples from coarse timing).
        # When channel delay is present, this offset can be significant.
        # Use the fine timing search range as the margin to handle all cases.
        self.timing_margin = system.sync.config.fine_timing_search_range

        # Minimum samples needed for preamble detection
        # Detection requires at least fft_size samples to compute metric
        self.min_detection_samples = self.fft_size + self.preamble_length

        # State
        self._buffer = np.array([], dtype=np.complex128)
        self._state = ReceiverState.SEARCHING
        self._frame_start = 0  # Index in buffer where detected frame starts
        self._search_start = 0  # Index in buffer where to resume searching
        self._total_samples_received = 0  # Total samples received across all calls
        self._frame_start_absolute = 0  # Absolute sample index of detected frame

    def reset(self):
        """Reset receiver state. Call when starting a new reception."""
        self._buffer = np.array([], dtype=np.complex128)
        self._state = ReceiverState.SEARCHING
        self._frame_start = 0
        self._search_start = 0
        self._total_samples_received = 0
        self._frame_start_absolute = 0

    def receive_streaming(self, samples: np.ndarray, debug: bool = False) -> List[StreamingReceiveResult]:
        """
        Process incoming samples and return any fully decoded frames.

        Args:
            samples: New samples to process (variable length)

        Returns:
            List of StreamingReceiveResult for each fully decoded frame.
            Empty list if no complete frames yet.
        """
        if len(samples) == 0:
            return []

        # Track absolute sample position before adding to buffer
        samples_before_append = len(self._buffer)

        # Append new samples to buffer
        self._buffer = np.concatenate([self._buffer, samples])
        self._total_samples_received += len(samples)

        results = []

        # Process buffer - may decode multiple frames if buffer has enough data
        while True:
            if self._state == ReceiverState.SEARCHING:
                # Look for preamble
                found = self._search_for_preamble(debug=debug)
                if not found:
                    # No preamble found, trim buffer but keep overlap for next search
                    self._trim_search_buffer()
                    break

                # Preamble found - state is now FRAME_DETECTED
                # Continue to check if we have enough samples

            if self._state == ReceiverState.FRAME_DETECTED:
                # Check if we have enough samples for complete frame + margin
                # Margin allows for coarse timing errors - fine timing will align
                samples_needed = self.frame_samples + self.timing_margin
                samples_after_frame_start = len(self._buffer) - self._frame_start
                if samples_after_frame_start < samples_needed:
                    # Not enough samples yet, wait for more
                    break

                # Extract and process frame
                result = self._process_frame(debug=debug)
                if result is not None:
                    results.append(result)

                # Remove processed samples from buffer
                self._consume_frame()

                # Reset to searching state
                self._state = ReceiverState.SEARCHING
                self._search_start = 0

                # Continue loop to check for more frames in remaining buffer

        return results

    def _search_for_preamble(self, debug: bool = False) -> bool:
        """
        Search for preamble in buffer starting from search_start.

        Returns:
            True if preamble found, False otherwise.
        """
        # Need enough samples for detection
        search_region = self._buffer[self._search_start:]
        if len(search_region) < self.min_detection_samples:
            return False

        # Use system's synchronizer to detect preamble
        # Pass noise_floor for energy-based gating to prevent false detections on noise
        timing, metric = self.system.sync.detect_preamble(
            search_region, threshold=self.sync_threshold,
            noise_floor=self.noise_floor, debug=debug
        )

        if debug:
            print(f"  [DEBUG] search_start={self._search_start}, search_len={len(search_region)}, "
                  f"timing={timing}, metric={metric:.3f}")

        if timing < 0:
            # No detection - update search_start to avoid re-scanning
            # Keep overlap for preamble that might span into next chunk
            # We can skip samples that definitely don't contain a preamble start
            safe_advance = max(0, len(search_region) - self.min_detection_samples)
            self._search_start += safe_advance
            return False

        # Preamble detected
        self._frame_start = self._search_start + timing
        self._frame_start_absolute = (self._total_samples_received - len(self._buffer)
                                       + self._frame_start)
        self._state = ReceiverState.FRAME_DETECTED

        if debug:
            print(f"  [DEBUG] Preamble found: frame_start={self._frame_start}, "
                  f"absolute={self._frame_start_absolute}, metric={metric:.3f}")

        return True

    def _trim_search_buffer(self):
        """
        Trim buffer while keeping enough overlap for preamble detection.

        Keeps min_detection_samples at the end to handle preamble spanning chunks.
        """
        if len(self._buffer) > self.min_detection_samples:
            # Keep only what's needed for overlap
            trim_amount = len(self._buffer) - self.min_detection_samples
            # But don't trim past search_start
            trim_amount = min(trim_amount, self._search_start)
            if trim_amount > 0:
                self._buffer = self._buffer[trim_amount:]
                self._search_start -= trim_amount

    def _process_frame(self, debug: bool = False) -> Optional[StreamingReceiveResult]:
        """
        Process detected frame and return result.

        Follows the same processing chain as system.receive():
        1. Synchronization (CFO estimation and fine timing)
        2. Demodulation (MMSE equalization)
        3. Frequency de-interleaving (on symbols, per codeword)
        4. Soft demodulation (symbols to LLRs)
        5. Bit de-interleaving (on LLRs, per LDPC block)
        6. LDPC decoding + CRC check

        Returns:
            StreamingReceiveResult or None if processing fails.
        """
        # Extract frame samples with margin for timing errors
        # The extra margin allows synchronize() to find the correct start
        frame_end = self._frame_start + self.frame_samples + self.timing_margin
        frame_samples = self._buffer[self._frame_start:frame_end]

        try:
            # Use synchronizer for CFO estimation and fine timing
            # Since we already detected the preamble in _search_for_preamble,
            # the extracted frame_samples starts at the preamble, so pass
            # coarse_timing=0 to skip re-detection.
            # synchronize() returns:
            #   data_start: offset from frame_samples[0] to where DATA begins
            #   cfo_hz: estimated CFO
            #   corrected: full frame_samples with CFO correction applied
            data_start, cfo_hz, corrected = self.system.sync.synchronize(
                frame_samples, coarse_timing=0, debug=debug
            )

            # Extract just the data portion (after preamble) from corrected signal
            # data_start already points to where data blocks begin
            data_signal = corrected[data_start:]

            if self.use_turbo:
                return self._process_frame_turbo(data_signal, cfo_hz, debug=debug)

            # Demodulate frame - signal starts at first data block
            demod_result = self.system.rx.demodulate_frame(
                data_signal,
                self.n_blocks,
                return_post_snr=True
            )
            eq_symbols, channel_est, post_snr = demod_result

            # Diagnostic: check signal and channel levels
            data_power = np.mean(np.abs(data_signal[:self.block_len])**2)
            channel_power = np.mean(np.abs(channel_est)**2)
            eq_power = np.mean(np.abs(eq_symbols)**2)
            regularization = self.system.rx.equalizer.regularization

            # Estimate noise from null subcarriers of first block
            first_block = data_signal[self.system.config.cp_length:self.system.config.cp_length + self.fft_size]
            rx_freq = np.fft.fft(first_block) / np.sqrt(self.fft_size)
            null_indices = self.system.rx.get_null_indices()
            noise_power = np.mean(np.abs(rx_freq[null_indices])**2)
            signal_power_est = data_power - noise_power
            snr_est = 10*np.log10(max(signal_power_est, 1e-20) / max(noise_power, 1e-20))

            if debug:
                print(f"  [DIAG] Data power: {10*np.log10(data_power + 1e-20):.1f} dB, "
                      f"Noise floor: {10*np.log10(noise_power + 1e-20):.1f} dB, "
                      f"Est SNR: {snr_est:.1f} dB, "
                      f"|H|²/reg: {channel_power/regularization:.2f}")

            # Frequency de-interleaving (on SYMBOLS, per codeword)
            # This must happen BEFORE soft demodulation, matching system.receive()
            if self.system.freq_interleaver is not None:
                n_syms = self.system._n_symbols_per_codeword
                # Trim to complete codewords only
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
                        self.system.freq_interleaver.deinterleave(eq_symbols[start:end])
                    )
                    if post_snr is not None:
                        deinterleaved_post_snr.append(
                            self.system.freq_interleaver.deinterleave(post_snr[start:end])
                        )

                if deinterleaved_symbols:
                    eq_symbols = np.concatenate(deinterleaved_symbols)
                    if post_snr is not None:
                        post_snr = np.concatenate(deinterleaved_post_snr)

            # Compute noise variance for soft demodulation
            if post_snr is not None:
                post_snr_clamped = np.maximum(post_snr, 1e-6)
                per_symbol_noise_var = 1.0 / post_snr_clamped
            elif self.noise_var is not None:
                per_symbol_noise_var = self.noise_var
            else:
                per_symbol_noise_var = 10 ** (-self.system.config.snr_estimate / 10)

            # Soft demodulation (symbols to LLRs)
            llrs = self.system.demodulator.demodulate(eq_symbols, per_symbol_noise_var, channel=None)

            if debug:
                # Diagnostic: LLR statistics
                llr_mag = np.abs(llrs)
                llr_positive = np.sum(llrs > 0) / len(llrs) * 100
                print(f"  [DIAG] LLR magnitude: mean={np.mean(llr_mag):.6f}, max={np.max(llr_mag):.6f}, "
                      f"positive: {llr_positive:.1f}%")

            # Bit de-interleaving (on LLRs, per LDPC block)
            ldpc_n = self.system.ldpc.n
            if self.system.interleaver:
                n_codewords = len(llrs) // ldpc_n
                deinterleaved = []
                for i in range(n_codewords):
                    start = i * ldpc_n
                    end = start + ldpc_n
                    if end <= len(llrs):
                        block = self.system.interleaver.deinterleave_llr(llrs[start:end])
                        deinterleaved.append(block)
                llrs = np.concatenate(deinterleaved) if deinterleaved else llrs

            # Descramble LLRs (per-codeword with unique offset matching TX)
            if self.system.scrambler is not None:
                n_codewords = len(llrs) // ldpc_n
                descrambled = []
                for i in range(n_codewords):
                    start = i * ldpc_n
                    end = start + ldpc_n
                    if end <= len(llrs):
                        descrambled.append(self.system.scrambler.descramble_llr(llrs[start:end], codeword_idx=i))
                llrs = np.concatenate(descrambled) if descrambled else llrs

            # LDPC decode - use batched decoding if available (SIMD parallel)
            decoded_bits = []
            all_converged = True
            all_crc_valid = True
            total_iters = 0
            cfg = self.system.config
            feat = self.system.features
            n_codewords = len(llrs) // ldpc_n

            # Limit to expected number of codewords if specified
            if self.num_codewords is not None and n_codewords > self.num_codewords:
                n_codewords = self.num_codewords
                llrs = llrs[:n_codewords * ldpc_n]
            if hasattr(self.system.ldpc, 'decode_batch') and n_codewords > 1:
                # Batch decode all codewords at once (much faster with SIMD)
                llrs_2d = llrs[:n_codewords * ldpc_n].reshape(n_codewords, ldpc_n)
                decoded_codewords, converged_arr = self.system.ldpc.decode_batch(llrs_2d)
                all_converged = converged_arr.all()

                for i in range(n_codewords):
                    info_with_crc = self.system.ldpc.get_info_bits(decoded_codewords[i])

                    if cfg.use_crc:
                        from .crc import check_crc
                        info_bits_block, crc_valid = check_crc(info_with_crc)
                        decoded_bits.append(info_bits_block)
                        if not crc_valid:
                            all_crc_valid = False
                    else:
                        decoded_bits.append(info_with_crc)
            else:
                # Sequential decode fallback
                for i in range(n_codewords):
                    start = i * ldpc_n
                    end = start + ldpc_n
                    if end <= len(llrs):
                        decoded, converged = self.system.ldpc.decode(
                            llrs[start:end],
                            max_iterations=feat.ldpc_max_iterations,
                            early_termination=feat.ldpc_early_termination
                        )
                        info_with_crc = self.system.ldpc.get_info_bits(decoded)

                        if cfg.use_crc:
                            from .crc import check_crc
                            info_bits_block, crc_valid = check_crc(info_with_crc)
                            decoded_bits.append(info_bits_block)
                            if not crc_valid:
                                all_crc_valid = False
                        else:
                            decoded_bits.append(info_with_crc)

                        if not converged:
                            all_converged = False

            decoded_bits = np.concatenate(decoded_bits) if decoded_bits else np.array([], dtype=np.int8)

            return StreamingReceiveResult(
                decoded_bits=decoded_bits,
                eq_symbols=eq_symbols,
                llr=llrs,
                converged=all_converged,
                n_iterations=total_iters,
                n_bit_errors=0,  # No reference available in streaming mode
                ber=0.0,
                frame_start_sample=self._frame_start_absolute,
                cfo_hz=cfo_hz,
                crc_valid=all_crc_valid,
            )

        except Exception as e:
            # Processing failed - could be sync failure, etc.
            import traceback
            print(f"  [DEBUG] Frame processing failed: {e}")
            traceback.print_exc()
            return None

    def _process_frame_turbo(self, data_signal: np.ndarray, cfo_hz: float,
                              debug: bool = False) -> Optional[StreamingReceiveResult]:
        """
        Process frame using turbo equalization (iterative MMSE-SIC + LDPC).

        Mirrors SCFDESystem._receive_turbo() but operates on the data_signal
        extracted by the streaming receiver's sync path.
        """
        system = self.system
        cfg = system.config
        turbo_eq = system._turbo_eq
        turbo_eq.set_snr(cfg.snr_estimate)

        block_len = cfg.fft_size + cfg.cp_length
        all_rx_data = []
        all_channels = []
        all_eq_symbols = []

        for block_idx in range(self.n_blocks):
            block_start = block_idx * block_len
            block_end = block_start + block_len
            if block_end > len(data_signal):
                break

            rx_block = data_signal[block_start:block_end]

            # Get equalized symbols, channel, and raw data in one pass
            eq_result = system.rx.demodulate_block(
                rx_block, return_channel=True, return_rx_data=True)
            eq_symbols_blk, channel_blk, rx_data_blk = (
                eq_result[0], eq_result[1], eq_result[2])
            all_eq_symbols.append(eq_symbols_blk)

            all_rx_data.append(rx_data_blk)
            all_channels.append(channel_blk)

        if not all_rx_data:
            return None

        rx_data = np.concatenate(all_rx_data)
        channel = np.concatenate(all_channels)
        eq_symbols = np.concatenate(all_eq_symbols)

        # Trim to actual data symbols
        n_codewords = self.num_codewords
        if n_codewords is None:
            ldpc_n = system.ldpc.n
            n_codewords = len(rx_data) // (ldpc_n // system.bits_per_symbol)
        n_data_symbols = n_codewords * system.ldpc.n // system.bits_per_symbol
        rx_data = rx_data[:n_data_symbols]
        channel = channel[:n_data_symbols]

        if debug:
            print(f"  [TURBO] n_blocks={self.n_blocks}, n_codewords={n_codewords}, "
                  f"n_data_symbols={n_data_symbols}")

        # Run turbo equalization + CRC stripping
        result = run_turbo_decode(
            turbo_eq, rx_data, channel,
            ldpc_k=system.ldpc.k,
            use_crc=cfg.use_crc,
            interleaver=system.interleaver,
            freq_interleaver=system.freq_interleaver,
            scrambler=system.scrambler,
            n_symbols_per_codeword=system._n_symbols_per_codeword,
        )

        if debug:
            print(f"  [TURBO] iterations={result.n_iterations}, converged={result.converged}, "
                  f"decoded_len={len(result.decoded_bits)}")

        return StreamingReceiveResult(
            decoded_bits=result.decoded_bits,
            eq_symbols=eq_symbols,
            llr=result.llrs,
            converged=result.converged,
            n_iterations=result.n_iterations,
            n_bit_errors=0,
            ber=0.0,
            frame_start_sample=self._frame_start_absolute,
            cfo_hz=cfo_hz,
            crc_valid=result.crc_valid,
        )

    def _consume_frame(self):
        """Remove processed frame samples from buffer."""
        # Consume frame + margin (the full extracted region)
        frame_end = self._frame_start + self.frame_samples + self.timing_margin
        self._buffer = self._buffer[frame_end:]
        self._frame_start = 0

    @property
    def state(self) -> ReceiverState:
        """Current receiver state."""
        return self._state

    @property
    def buffered_samples(self) -> int:
        """Number of samples currently buffered."""
        return len(self._buffer)

    @property
    def samples_until_frame_complete(self) -> int:
        """
        Samples needed to complete current frame (if detected).

        Returns 0 if no frame detected or frame is complete.
        """
        if self._state != ReceiverState.FRAME_DETECTED:
            return 0
        samples_have = len(self._buffer) - self._frame_start
        samples_need = self.frame_samples - samples_have
        return max(0, samples_need)

    @property
    def total_samples_received(self) -> int:
        """Total samples received since last reset."""
        return self._total_samples_received


def test_streaming_receiver():
    """Test the streaming receiver with simulated data."""
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from .system import SCFDESystem, SystemConfig

    print("Testing StreamingReceiver...")

    # Create system
    cfg = SystemConfig(modulation='qpsk', ldpc_rate=0.5)
    system = SCFDESystem(cfg)

    # Generate test frame first to get actual n_blocks
    np.random.seed(43)
    n_codewords = 4
    info_bits = np.random.randint(0, 2, system.ldpc.k * n_codewords, dtype=np.int8)
    tx_result = system.transmit(info_bits, include_preamble=True)

    # Use actual n_blocks from transmit result
    n_blocks = tx_result.n_blocks
    # Use higher threshold to avoid false positives in noise
    receiver = StreamingReceiver(system, n_blocks=n_blocks, sync_threshold=0.9)

    print(f"\nConfiguration:")
    print(f"  FFT size: {receiver.fft_size}")
    print(f"  Block length: {receiver.block_len}")
    print(f"  Preamble length: {receiver.preamble_length}")
    print(f"  n_blocks (from tx): {n_blocks}")
    print(f"  Frame samples: {receiver.frame_samples}")
    print(f"  Actual TX signal length: {len(tx_result.tx_signal)}")

    # Add some noise before and after
    noise_before = 1000
    noise_after = 500
    snr_db = 20.0
    signal_power = np.mean(np.abs(tx_result.tx_signal) ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))

    pre_noise = np.sqrt(noise_power / 2) * (
        np.random.randn(noise_before) + 1j * np.random.randn(noise_before)
    )
    post_noise = np.sqrt(noise_power / 2) * (
        np.random.randn(noise_after) + 1j * np.random.randn(noise_after)
    )

    # Add channel noise to signal
    signal_noise = np.sqrt(noise_power / 2) * (
        np.random.randn(len(tx_result.tx_signal)) +
        1j * np.random.randn(len(tx_result.tx_signal))
    )
    noisy_signal = tx_result.tx_signal + signal_noise

    # Full test signal
    full_signal = np.concatenate([pre_noise, noisy_signal, post_noise])

    print(f"\nTest signal:")
    print(f"  Noise before: {noise_before} samples")
    print(f"  Frame: {len(tx_result.tx_signal)} samples")
    print(f"  Noise after: {noise_after} samples")
    print(f"  Total: {len(full_signal)} samples")
    print(f"  SNR: {snr_db} dB")

    # Test 0: Baseline - use system.receive() directly
    print("\n--- Test 0: Baseline using system.receive() ---")
    rx_result = system.receive(noisy_signal, tx_result, use_sync=True)
    print(f"  Converged: {rx_result.converged}")
    print(f"  BER: {rx_result.ber:.2e}")

    # Test 1: Process all at once
    print("\n--- Test 1: Process all at once ---")
    print(f"  Expected preamble start: {noise_before}")
    receiver.reset()
    results = receiver.receive_streaming(full_signal, debug=True)
    print(f"  Results: {len(results)} frame(s) decoded")
    if results:
        r = results[0]
        print(f"  Converged: {r.converged}")
        print(f"  Iterations: {r.n_iterations}")
        print(f"  Frame start: {r.frame_start_sample}")
        print(f"  CFO: {r.cfo_hz:.1f} Hz")
        # Check BER - need to compare with tx_result.info_bits (after CRC removal)
        # Note: CRC is added by transmit, so compare only up to what we decoded
        min_len = min(len(r.decoded_bits), len(tx_result.info_bits))
        if min_len > 0:
            ber = np.mean(r.decoded_bits[:min_len] != tx_result.info_bits[:min_len])
            print(f"  BER: {ber:.2e}")

    # Test 2: Process in small chunks
    print("\n--- Test 2: Process in small chunks ---")
    receiver.reset()
    chunk_size = 1000
    all_results = []
    for i in range(0, len(full_signal), chunk_size):
        chunk = full_signal[i:i + chunk_size]
        results = receiver.receive_streaming(chunk)
        all_results.extend(results)
        if receiver.state == ReceiverState.FRAME_DETECTED:
            print(f"  Chunk {i//chunk_size}: Frame detected, need {receiver.samples_until_frame_complete} more samples")

    print(f"  Results: {len(all_results)} frame(s) decoded")
    if all_results:
        r = all_results[0]
        print(f"  Converged: {r.converged}")
        ber = np.mean(r.decoded_bits[:len(info_bits)] != info_bits)
        print(f"  BER: {ber:.2e}")

    # Test 3: Multiple frames
    print("\n--- Test 3: Multiple frames ---")
    # Create signal with two frames - need adequate gap for preamble detection
    gap = 2000  # Larger gap to ensure clean separation
    gap_noise = np.sqrt(noise_power / 2) * (
        np.random.randn(gap) + 1j * np.random.randn(gap)
    )
    # Add more post-frame noise so second frame has enough trailing samples
    post_noise_multi = np.sqrt(noise_power / 2) * (
        np.random.randn(1500) + 1j * np.random.randn(1500)
    )
    two_frame_signal = np.concatenate([
        pre_noise, noisy_signal, gap_noise, noisy_signal, post_noise_multi
    ])

    receiver.reset()
    results = receiver.receive_streaming(two_frame_signal)
    print(f"  Results: {len(results)} frame(s) decoded")
    for i, r in enumerate(results):
        min_len = min(len(r.decoded_bits), len(info_bits))
        ber = np.mean(r.decoded_bits[:min_len] != info_bits[:min_len]) if min_len > 0 else 1.0
        print(f"  Frame {i}: start={r.frame_start_sample}, converged={r.converged}, BER={ber:.2e}")

    # Verify we decoded both frames
    assert len(results) == 2, f"Expected 2 frames, got {len(results)}"

    print("\nStreamingReceiver tests passed!")
    return True


if __name__ == '__main__':
    test_streaming_receiver()
