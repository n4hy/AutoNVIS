"""
Synchronization for SC-FDE System

Implements dual-preamble synchronization (similar to IEEE 802.11a):
- Short preamble: 16 repetitions for coarse timing and wide-range CFO estimation (±3000 Hz)
- Long preamble: Full-bandwidth ZC sequence for fine timing

The short preamble uses every 16th subcarrier, creating 16 identical
repetitions of 256 samples each. This provides CFO estimation range of
±(sample_rate / (2 * 256)) = ±3000 Hz at 1.536 MHz sample rate.
"""

import numpy as np
from scipy.signal import correlate
from typing import Tuple, Optional
from dataclasses import dataclass

from .utils import compute_subcarrier_indices


@dataclass
class SyncConfig:
    """Synchronization configuration"""
    fft_size: int = 4096
    cp_length: int = 1536  # Data symbol CP length
    sample_rate: float = 1.536e6

    # Subcarrier allocation (must match transmitter/receiver for guard band consistency)
    # If 0, defaults are computed based on fft_size (scaled from 4096 reference)
    n_pilots: int = 0         # Number of pilot subcarriers (0 = auto)
    n_data_carriers: int = 0  # Number of data subcarriers (0 = auto)
    pilot_spacing: int = 8    # Pilot spacing

    # Short preamble parameters
    short_preamble_repetitions: int = 16  # Number of repetitions (use 16 for ±3000 Hz CFO range)
    short_preamble_cp_length: int = 256   # CP for short preamble

    # Long preamble parameters
    long_preamble_cp_length: int = 256    # CP for long preamble

    # Detection parameters
    fine_timing_search_range: int = 1500  # Search range for fine timing (increased to handle late detection)
    rising_edge_threshold: float = 0.85   # Threshold for plateau detection
    min_plateau_width: int = 50           # Minimum consecutive samples above threshold
    min_detection_snr_db: float = 6.0     # Minimum SNR (dB) above noise floor for valid detection

    # Reference values for fft_size=4096 (used for scaling defaults)
    _REF_FFT_SIZE: int = 4096
    _REF_N_PILOTS: int = 251
    _REF_N_DATA_CARRIERS: int = 1757

    def __post_init__(self):
        """Compute default subcarrier allocation if not specified."""
        scale = self.fft_size / self._REF_FFT_SIZE
        if self.n_pilots <= 0:
            self.n_pilots = max(1, int(self._REF_N_PILOTS * scale))
        if self.n_data_carriers <= 0:
            self.n_data_carriers = max(1, int(self._REF_N_DATA_CARRIERS * scale))


class Synchronizer:
    """
    Timing and frequency synchronizer for SC-FDE.

    Uses dual-preamble structure (similar to IEEE 802.11a):
    1. Short preamble (16 repetitions) for coarse timing and CFO
    2. Long preamble (full bandwidth) for fine timing

    CFO estimation range: ±(sample_rate / (2 * repetition_length))
    With 16 repetitions at 4096 FFT: ±3000 Hz
    """

    def __init__(self, config: Optional[SyncConfig] = None):
        """
        Initialize synchronizer.

        Args:
            config: Synchronization configuration
        """
        self.config = config or SyncConfig()
        self._generate_dual_preamble()

    def _generate_dual_preamble(self):
        """Generate dual preamble structure for synchronization."""
        cfg = self.config

        # =================================================================
        # Short preamble: N repetitions for coarse timing and CFO
        # =================================================================
        n_reps = cfg.short_preamble_repetitions
        subcarrier_step = n_reps  # Use every Nth subcarrier
        self.repetition_length = cfg.fft_size // n_reps

        # Determine which every-Nth carriers to use (respecting guard bands)
        all_spaced_indices = np.arange(0, cfg.fft_size, subcarrier_step)

        # Compute active bandwidth boundaries from subcarrier allocation
        pilot_indices, data_indices, null_indices = compute_subcarrier_indices(
            cfg.fft_size, cfg.n_pilots, cfg.n_data_carriers, cfg.pilot_spacing
        )
        active_set = set(pilot_indices) | set(data_indices)
        # Keep only spaced carriers that fall within active bandwidth
        # Also exclude DC (index 0)
        short_active_indices = np.array([
            idx for idx in all_spaced_indices
            if idx in active_set and idx != 0
        ], dtype=int)

        n_active_short = len(short_active_indices)
        self.short_preamble_active_indices = short_active_indices

        # ZC sequence for short preamble (root 26)
        short_root = 26
        n = np.arange(n_active_short)
        zc_short = np.exp(-1j * np.pi * short_root * n * (n + 1) / n_active_short)

        # Place on selected subcarriers (within active bandwidth, every Nth)
        freq_short = np.zeros(cfg.fft_size, dtype=np.complex128)
        freq_short[short_active_indices] = zc_short

        # Time domain - will have approximate N repetitions (exact if using all spaced carriers)
        self.short_preamble = np.fft.ifft(freq_short) * np.sqrt(cfg.fft_size)
        # Normalize
        self.short_preamble = self.short_preamble / np.sqrt(
            np.mean(np.abs(self.short_preamble)**2))

        # Short preamble with CP
        self.short_preamble_cp_length = cfg.short_preamble_cp_length
        self.short_preamble_with_cp = np.concatenate([
            self.short_preamble[-self.short_preamble_cp_length:],
            self.short_preamble
        ])

        # =================================================================
        # Long preamble: Uses same bandwidth as data (respects guard bands)
        # =================================================================
        # Use the same subcarrier allocation as data transmission
        # (pilot_indices, data_indices already computed above for short preamble)
        # Active carriers = pilots + data (sorted)
        active_indices = np.sort(np.concatenate([pilot_indices, data_indices]))
        self.long_preamble_active_indices = active_indices

        n_active_long = len(active_indices)

        # ZC sequence for long preamble (root 29, different from short)
        # Length matches number of active carriers for optimal autocorrelation
        long_root = 29
        n = np.arange(n_active_long)
        zc_long = np.exp(-1j * np.pi * long_root * n * (n + 1) / n_active_long)

        # Place ZC sequence only on active subcarriers (guard bands stay zero)
        freq_long = np.zeros(cfg.fft_size, dtype=np.complex128)
        freq_long[active_indices] = zc_long

        # Time domain
        self.long_preamble = np.fft.ifft(freq_long) * np.sqrt(cfg.fft_size)
        # Normalize
        self.long_preamble = self.long_preamble / np.sqrt(
            np.mean(np.abs(self.long_preamble)**2))

        # Long preamble with CP
        self.long_preamble_cp_length = cfg.long_preamble_cp_length
        self.long_preamble_with_cp = np.concatenate([
            self.long_preamble[-self.long_preamble_cp_length:],
            self.long_preamble
        ])

        # =================================================================
        # Combined dual preamble: [short_with_cp | long_with_cp]
        # =================================================================
        self.preamble_with_cp = np.concatenate([
            self.short_preamble_with_cp,
            self.long_preamble_with_cp
        ])
        self.preamble_length = len(self.preamble_with_cp)

        # For backward compatibility, expose short preamble as 'preamble'
        self.preamble = self.short_preamble
        self.preamble_cp_length = self.short_preamble_cp_length

    def detect_preamble(self, rx_signal: np.ndarray,
                        threshold: float = 0.7,
                        noise_floor: Optional[float] = None,
                        min_energy_above_noise_db: float = 6.0,
                        debug: bool = False) -> Tuple[int, float]:
        """
        Detect preamble and estimate coarse timing.

        Uses Schmidl-Cox autocorrelation metric on the short preamble's
        repetition structure.

        Args:
            rx_signal: Received signal
            threshold: Detection threshold (0 to 1)
            noise_floor: Known noise floor power (per sample). If provided,
                        metric is gated to zero when signal energy is not
                        significantly above noise.
            min_energy_above_noise_db: Minimum dB above noise floor for valid
                                       detection (default 6 dB)
            debug: Print debug information about metric behavior

        Returns:
            Tuple of (timing_index, peak_metric)
        """
        cfg = self.config
        rep_len = self.repetition_length

        n_samples = len(rx_signal) - cfg.fft_size
        if n_samples <= 0:
            return -1, 0.0

        # Schmidl-Cox metric using adjacent repetitions
        # P(d) = sum(conj(r[d+m]) * r[d+m+L]) for m = 0..L-1
        # where L = repetition_length
        cross_corr = np.conj(rx_signal[:len(rx_signal) - rep_len]) * rx_signal[rep_len:]
        cumsum_cross = np.cumsum(cross_corr)

        P = np.empty(n_samples, dtype=np.complex128)
        P[0] = cumsum_cross[rep_len - 1]
        P[1:] = cumsum_cross[rep_len:rep_len + n_samples - 1] - cumsum_cross[:n_samples - 1]

        # Energy term R(d) = sum of |r[d+m+L]|² for m = 0..L-1
        energy = np.abs(rx_signal[rep_len:]) ** 2
        cumsum_energy = np.cumsum(energy)

        R = np.empty(n_samples, dtype=np.float64)
        R[0] = cumsum_energy[rep_len - 1]
        R[1:] = cumsum_energy[rep_len:rep_len + n_samples - 1] - cumsum_energy[:n_samples - 1]

        # Metric: M(d) = |P(d)|² / R(d)²
        metric = np.zeros(n_samples)
        valid = R > 0
        metric[valid] = np.abs(P[valid]) ** 2 / (R[valid] ** 2)

        # Energy-based gating: set metric to zero where signal energy is not
        # significantly above the noise floor. This prevents false detections
        # on noise which can produce high metric values due to random self-correlation.
        if noise_floor is not None and noise_floor > 0:
            # R(d) is sum of rep_len samples, so expected noise energy = rep_len * noise_floor
            expected_noise_energy = rep_len * noise_floor
            min_energy_ratio = 10 ** (min_energy_above_noise_db / 10)
            min_energy_threshold = expected_noise_energy * min_energy_ratio

            # Gate the metric: set to zero where energy is below threshold
            low_energy = R < min_energy_threshold
            metric[low_energy] = 0.0

            if debug:
                n_gated = np.sum(low_energy)
                print(f"  [SC-metric] Energy gating: noise_floor={noise_floor:.2e}, "
                      f"min_threshold={min_energy_threshold:.2e}, gated={n_gated}/{n_samples} samples")

        peak_val = np.max(metric)
        peak_idx = np.argmax(metric)

        if debug:
            # Find where metric first crosses various thresholds
            thresh_50 = np.where(metric > 0.5)[0]
            thresh_70 = np.where(metric > 0.7)[0]
            thresh_85 = np.where(metric > 0.85)[0]
            print(f"  [SC-metric] peak_val={peak_val:.3f} at idx={peak_idx}")
            print(f"  [SC-metric] first > 0.50: {thresh_50[0] if len(thresh_50) > 0 else 'never'}")
            print(f"  [SC-metric] first > 0.70: {thresh_70[0] if len(thresh_70) > 0 else 'never'}")
            print(f"  [SC-metric] first > 0.85: {thresh_85[0] if len(thresh_85) > 0 else 'never'}")

            # Show metric profile around peak
            if peak_idx > 2000:
                samples_before = [-4000, -3000, -2000, -1000, -500, -200, 0]
                profile = []
                for offset in samples_before:
                    idx = peak_idx + offset
                    if 0 <= idx < len(metric):
                        profile.append(f"{offset:+5d}:{metric[idx]:.3f}")
                print(f"  [SC-metric] profile near peak: {', '.join(profile)}")

        if peak_val < threshold:
            return -1, peak_val

        # Use rising edge detection for stable timing
        rising_threshold = self.config.rising_edge_threshold
        timing = self._find_rising_edge(metric, threshold=rising_threshold, debug=debug)

        if debug:
            print(f"  [SC-metric] rising edge timing={timing}, gap from peak={peak_idx - timing}")

        return timing, metric[timing] if timing >= 0 else peak_val

    def _find_rising_edge(self, metric: np.ndarray,
                          threshold: float = 0.85,
                          min_plateau_width: int = None,
                          debug: bool = False) -> int:
        """
        Find the rising edge of the Schmidl-Cox metric plateau.

        Args:
            metric: Schmidl-Cox metric array
            threshold: Rising edge threshold
            min_plateau_width: Minimum consecutive samples above threshold
            debug: Print debug information

        Returns:
            Index of rising edge (start of plateau)
        """
        if min_plateau_width is None:
            min_plateau_width = self.config.min_plateau_width

        above_threshold = metric > threshold

        diff = np.diff(above_threshold.astype(np.int8))
        crossings = np.where(diff == 1)[0]

        if debug:
            # Count total samples above threshold
            n_above = np.sum(above_threshold)
            print(f"  [rising_edge] threshold={threshold:.2f}, samples_above={n_above}, "
                  f"num_crossings={len(crossings)}")
            if len(crossings) > 0:
                # Show first few crossings and their plateau lengths
                for i, crossing in enumerate(crossings[:5]):
                    # Measure plateau length at this crossing
                    plateau_len = 0
                    for j in range(crossing + 1, len(metric)):
                        if above_threshold[j]:
                            plateau_len += 1
                        else:
                            break
                    print(f"  [rising_edge] crossing[{i}] at idx={crossing}, plateau_len={plateau_len}, "
                          f"valid={'YES' if plateau_len >= min_plateau_width else 'NO'}")
                    if i >= 4:
                        break

        if len(crossings) == 0:
            # No rising edge found - check if metric is already above threshold at start
            if above_threshold[0]:
                # Signal starts with preamble already aligned - return 0
                if debug:
                    print(f"  [rising_edge] No crossings but metric already high at start, returning 0")
                return 0
            if debug:
                print(f"  [rising_edge] No crossings found, using argmax")
            return int(np.argmax(metric))

        for crossing in crossings:
            end_check = min(crossing + 1 + min_plateau_width, len(metric))
            if np.all(above_threshold[crossing + 1:end_check]):
                return int(crossing + 1)

        if debug:
            print(f"  [rising_edge] No valid plateau found, using argmax")
        return int(np.argmax(metric))

    def estimate_cfo(self, rx_signal: np.ndarray, timing: int) -> float:
        """
        Estimate carrier frequency offset.

        Uses phase of autocorrelation between adjacent repetitions of the
        short preamble. Averages over multiple repetition pairs for robustness.

        Unambiguous range: ±(sample_rate / (2 * repetition_length))
        With 16 reps at 4096 FFT: ±3000 Hz

        Args:
            rx_signal: Received signal
            timing: Detected timing index

        Returns:
            CFO estimate in Hz
        """
        cfg = self.config
        rep_len = self.repetition_length
        n_reps = cfg.short_preamble_repetitions

        # Average over multiple repetition pairs for robustness
        n_pairs = min(8, n_reps - 1)
        P_total = 0

        for i in range(n_pairs):
            start = timing + i * rep_len
            rep_i = rx_signal[start:start + rep_len]
            rep_next = rx_signal[start + rep_len:start + 2 * rep_len]

            if len(rep_i) < rep_len or len(rep_next) < rep_len:
                continue

            P_total += np.sum(np.conj(rep_i) * rep_next)

        angle = np.angle(P_total)
        cfo_hz = angle * cfg.sample_rate / (2 * np.pi * rep_len)

        return cfo_hz

    def get_cfo_range(self) -> float:
        """Get the unambiguous CFO estimation range in Hz."""
        return self.config.sample_rate / (2 * self.repetition_length)

    def correct_cfo(self, signal: np.ndarray, cfo_hz: float) -> np.ndarray:
        """
        Apply CFO correction to signal.

        Args:
            signal: Input signal
            cfo_hz: CFO in Hz

        Returns:
            CFO-corrected signal
        """
        cfg = self.config
        n = np.arange(len(signal))
        correction = np.exp(-1j * 2 * np.pi * cfo_hz * n / cfg.sample_rate)
        return signal * correction

    def fine_timing(self, rx_signal: np.ndarray,
                    coarse_timing: int,
                    search_range: Optional[int] = None,
                    debug: bool = False) -> int:
        """
        Fine timing adjustment using cross-correlation with long preamble.

        Should be called AFTER CFO correction for best results.

        Args:
            rx_signal: Received signal (should be CFO-corrected)
            coarse_timing: Coarse timing estimate from detect_preamble
            search_range: +/- samples to search

        Returns:
            Fine timing estimate (index of long preamble start)
        """
        if search_range is None:
            search_range = self.config.fine_timing_search_range

        # Expected position of long preamble (after short preamble)
        expected_long_start = coarse_timing + len(self.short_preamble_with_cp)

        # Search window
        start = max(0, expected_long_start - search_range)
        end = min(len(rx_signal) - len(self.long_preamble_with_cp),
                  expected_long_start + search_range)

        if end <= start:
            return expected_long_start

        # Cross-correlate with long preamble (including CP)
        search_signal = rx_signal[start:end + len(self.long_preamble_with_cp)]
        corr = np.abs(correlate(search_signal, self.long_preamble_with_cp,
                                mode='valid', method='fft'))

        best_idx = np.argmax(corr)
        fine_long_start = start + best_idx

        if debug:
            # Debug: check correlation quality
            peak_val = corr[best_idx]
            mean_val = np.mean(corr)
            peak_to_mean = peak_val / (mean_val + 1e-10)
            print(f"  [fine_timing] peak={peak_val:.2f}, mean={mean_val:.2f}, ratio={peak_to_mean:.1f}, "
                  f"best_idx={best_idx}, result={fine_long_start}")

        return fine_long_start

    def synchronize(self, rx_signal: np.ndarray,
                    threshold: float = 0.7,
                    coarse_timing: Optional[int] = None,
                    debug: bool = False) -> Tuple[int, float, np.ndarray]:
        """
        Full synchronization: timing + CFO correction.

        Performs:
        1. Coarse timing detection (short preamble) - or use provided timing
        2. CFO estimation (short preamble, ±3000 Hz range)
        3. CFO correction
        4. Fine timing (long preamble cross-correlation)

        Args:
            rx_signal: Received signal
            threshold: Detection threshold (only used if coarse_timing not provided)
            coarse_timing: If provided, skip detection and use this as preamble start.
                          Use this when preamble has already been detected.
            debug: Print debug information

        Returns:
            Tuple of (data_start_index, cfo_hz, corrected_signal)
        """
        # Step 1: Coarse timing detection (or use provided)
        if coarse_timing is not None:
            timing = coarse_timing
            if debug:
                print(f"[sync] Using provided coarse timing: {timing}")
        else:
            timing, metric = self.detect_preamble(rx_signal, threshold)

            if timing < 0:
                raise ValueError("Preamble not detected")

            if debug:
                print(f"[sync] Coarse timing: {timing}, metric: {metric:.3f}")

        # Step 2: CFO estimation (now has ±3000 Hz range!)
        cfo = self.estimate_cfo(rx_signal, timing)

        if debug:
            print(f"[sync] CFO estimate: {cfo:.2f} Hz (range: ±{self.get_cfo_range():.0f} Hz)")

        # Step 3: CFO correction
        corrected = self.correct_cfo(rx_signal, cfo)

        # Step 4: Fine timing using long preamble
        fine_long_start = self.fine_timing(corrected, timing, debug=debug)

        if debug:
            expected = timing + len(self.short_preamble_with_cp)
            print(f"[sync] Fine timing: {fine_long_start} (expected: {expected})")

        # Data starts after long preamble
        data_start = fine_long_start + len(self.long_preamble_with_cp)

        return data_start, cfo, corrected


def test_synchronizer():
    """Test synchronization with various CFO values."""
    print("Testing Synchronizer with Dual Preamble...")

    cfg = SyncConfig(fft_size=4096, cp_length=1536, sample_rate=1.536e6)
    sync = Synchronizer(cfg)

    print(f"\nConfiguration:")
    print(f"  FFT size: {cfg.fft_size}")
    print(f"  Short preamble repetitions: {cfg.short_preamble_repetitions}")
    print(f"  Repetition length: {sync.repetition_length}")
    print(f"  CFO range: ±{sync.get_cfo_range():.0f} Hz")
    print(f"  Short preamble length: {len(sync.short_preamble_with_cp)}")
    print(f"  Long preamble length: {len(sync.long_preamble_with_cp)}")
    print(f"  Total preamble length: {sync.preamble_length}")

    # Generate test signal
    np.random.seed(42)
    data_len = cfg.fft_size * 3
    data = (np.random.randn(data_len) + 1j * np.random.randn(data_len)) / np.sqrt(2)

    noise_before = 500
    noise = (np.random.randn(noise_before) + 1j * np.random.randn(noise_before)) * 0.1

    tx_signal = np.concatenate([noise, sync.preamble_with_cp, data])

    # Test CFO estimation at various offsets
    print(f"\n{'='*60}")
    print("CFO Estimation Tests:")
    print(f"{'='*60}")

    test_cfos = [0, 500, 1000, 1500, 2000, 2500, 3000, -1500, -3000]

    for snr_db in [20, 10]:
        print(f"\nSNR = {snr_db} dB:")
        print(f"{'True CFO':>10} | {'Est CFO':>10} | {'Error':>10}")
        print("-" * 40)

        for true_cfo in test_cfos:
            n = np.arange(len(tx_signal))
            rx_signal = tx_signal * np.exp(1j * 2 * np.pi * true_cfo * n / cfg.sample_rate)

            # Add noise
            sig_power = np.mean(np.abs(rx_signal)**2)
            noise_power = sig_power / (10 ** (snr_db / 10))
            awgn = np.sqrt(noise_power / 2) * (
                np.random.randn(len(rx_signal)) + 1j * np.random.randn(len(rx_signal)))
            rx_signal = rx_signal + awgn

            timing, metric = sync.detect_preamble(rx_signal, threshold=0.3)
            if timing < 0:
                print(f"{true_cfo:10.1f} | Detection failed")
                continue

            cfo_est = sync.estimate_cfo(rx_signal, timing)
            error = cfo_est - true_cfo

            status = "OK" if abs(error) < 50 else "FAIL"
            print(f"{true_cfo:10.1f} | {cfo_est:10.1f} | {error:+10.1f} | {status}")

    # Full synchronization test
    print(f"\n{'='*60}")
    print("Full Synchronization Test:")
    print(f"{'='*60}")

    true_cfo = 1500.0
    snr_db = 15

    n = np.arange(len(tx_signal))
    rx_signal = tx_signal * np.exp(1j * 2 * np.pi * true_cfo * n / cfg.sample_rate)

    sig_power = np.mean(np.abs(rx_signal)**2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    awgn = np.sqrt(noise_power / 2) * (
        np.random.randn(len(rx_signal)) + 1j * np.random.randn(len(rx_signal)))
    rx_signal = rx_signal + awgn

    try:
        data_start, cfo_est, corrected = sync.synchronize(rx_signal, threshold=0.3, debug=True)
        expected_data_start = noise_before + sync.preamble_length
        print(f"\nResults:")
        print(f"  True CFO: {true_cfo:.1f} Hz")
        print(f"  Estimated CFO: {cfo_est:.1f} Hz")
        print(f"  CFO error: {cfo_est - true_cfo:.1f} Hz")
        print(f"  Data start: {data_start}")
        print(f"  Expected data start: {expected_data_start}")
        print(f"  Timing error: {data_start - expected_data_start} samples")
    except ValueError as e:
        print(f"Sync failed: {e}")

    print("\nSynchronizer tests passed!")
    return True


if __name__ == '__main__':
    test_synchronizer()
