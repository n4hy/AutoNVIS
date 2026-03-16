"""
SC-FDE Receiver with MMSE Equalization

Implements:
- CP removal and FFT
- Channel estimation (LS + DFT interpolation)
- MMSE equalization: W = H*/(|H|² + 1/SNR)
- Soft output for LDPC decoding
"""

import numpy as np
import warnings
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

from .transmitter import TransmitterConfig, generate_zadoff_chu, compensate_pilot_boost
from .utils import compute_subcarrier_indices


def estimate_noise_variance(rx_freq: np.ndarray, null_indices: np.ndarray) -> float:
    """
    Estimate noise variance from null/guard subcarriers.

    Uses the fact that null subcarriers contain only noise (no signal),
    so their power provides a direct estimate of noise variance.

    Args:
        rx_freq: FFT output (complex), shape (fft_size,)
        null_indices: Indices of null subcarriers (DC + guard bands)

    Returns:
        Estimated noise variance (real, positive)

    Raises:
        ValueError: If no valid null subcarriers available
    """
    # Exclude DC (index 0) due to potential DC offset/leakage
    valid_indices = null_indices[null_indices != 0]

    if len(valid_indices) == 0:
        raise ValueError("No valid null subcarriers for noise estimation")

    if len(valid_indices) < 10:
        warnings.warn(
            f"Only {len(valid_indices)} null subcarriers available for noise "
            "estimation. Estimate may be unreliable.",
            UserWarning
        )

    # Extract noise samples from null subcarriers
    noise_samples = rx_freq[valid_indices]

    # Estimate variance: E[|n|^2] for complex Gaussian noise
    # For complex Gaussian with variance sigma^2, E[|n|^2] = sigma^2
    noise_variance = np.mean(np.abs(noise_samples) ** 2)

    return float(noise_variance)


def compute_reliability_weights(post_snr: np.ndarray,
                                threshold_db: float = 0.0,
                                steepness: float = 0.5) -> np.ndarray:
    """
    Compute soft reliability weights using sigmoid function.

    For soft null-subcarrier erasure: instead of binary threshold,
    use a smooth sigmoid transition based on post-equalization SNR.

    Args:
        post_snr: Post-MMSE SNR per subcarrier (linear scale)
        threshold_db: SNR threshold for 0.5 weight (dB)
        steepness: Sigmoid steepness (higher = sharper transition)

    Returns:
        Weights in [0, 1] per subcarrier
    """
    post_snr = np.asarray(post_snr, dtype=np.float64)

    # Handle zero/negative SNR gracefully
    post_snr = np.maximum(post_snr, 1e-15)

    # Convert to dB
    post_snr_db = 10 * np.log10(post_snr)

    # Sigmoid: 1 / (1 + exp(-steepness * (snr_db - threshold_db)))
    exponent = -steepness * (post_snr_db - threshold_db)

    # Clip to prevent overflow
    exponent = np.clip(exponent, -100, 100)

    weights = 1.0 / (1.0 + np.exp(exponent))

    return weights


def apply_soft_erasure(llrs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Apply soft erasure by scaling LLRs with reliability weights.

    Args:
        llrs: Log-likelihood ratios per subcarrier/bit
        weights: Reliability weights in [0, 1]

    Returns:
        Scaled LLRs (same shape as input)
    """
    return llrs * weights


def construct_channel_correlation_matrix(n_pilots: int,
                                          delay_spread_samples: int) -> np.ndarray:
    """
    Construct channel correlation matrix assuming exponential power delay profile.

    For exponential PDP, the frequency correlation follows a sinc function.
    R_hh[i,j] = sinc((k_i - k_j) * delay_spread / N)

    Args:
        n_pilots: Number of pilot subcarriers
        delay_spread_samples: Maximum delay spread in samples

    Returns:
        Channel correlation matrix R_hh of shape (n_pilots, n_pilots)
    """
    if delay_spread_samples <= 0:
        # Flat channel -> identity correlation
        return np.eye(n_pilots)

    # Pilot frequency indices (normalized)
    pilot_freqs = np.arange(n_pilots)
    freq_diff = pilot_freqs[:, None] - pilot_freqs[None, :]

    # Normalized delay spread
    tau_norm = delay_spread_samples / n_pilots

    # Sinc correlation from exponential PDP
    # sinc(x) = sin(pi*x) / (pi*x), with sinc(0) = 1
    R_hh = np.sinc(freq_diff * tau_norm)

    return R_hh


def mmse_channel_estimate(h_ls: np.ndarray,
                          R_hh: np.ndarray,
                          noise_var: float) -> np.ndarray:
    """
    MMSE channel estimation.

    Computes: h_mmse = R_hh @ (R_hh + sigma_n^2 * I)^(-1) @ h_ls

    Args:
        h_ls: LS channel estimates at pilot positions
        R_hh: Channel correlation matrix
        noise_var: Noise variance estimate

    Returns:
        MMSE channel estimate
    """
    n = len(h_ls)

    # Regularization for numerical stability
    reg = max(noise_var, 1e-10)

    # Compute (R_hh + sigma_n^2 * I)^(-1) using solve for stability
    # W_mmse = R_hh @ inv(R_hh + sigma_n^2 * I)
    A = R_hh + reg * np.eye(n)

    try:
        # Solve A @ X = R_hh for X, then h_mmse = X @ h_ls
        W_mmse = np.linalg.solve(A, R_hh.T).T
        h_mmse = W_mmse @ h_ls
    except np.linalg.LinAlgError:
        # Fallback to pseudoinverse if singular
        W_mmse = R_hh @ np.linalg.pinv(A)
        h_mmse = W_mmse @ h_ls

    return h_mmse


class MMSEChannelEstimator:
    """
    MMSE Channel Estimator with precomputed filter.

    Precomputes the MMSE filter matrix for efficient repeated estimation.
    """

    def __init__(self, n_pilots: int, delay_spread_samples: int, noise_var: float):
        """
        Initialize MMSE estimator.

        Args:
            n_pilots: Number of pilot subcarriers
            delay_spread_samples: Expected channel delay spread in samples
            noise_var: Initial noise variance estimate
        """
        self.n_pilots = n_pilots
        self.delay_spread_samples = delay_spread_samples
        self.noise_var = noise_var

        # Construct correlation matrix
        self.R_hh = construct_channel_correlation_matrix(n_pilots, delay_spread_samples)

        # Precompute MMSE filter
        self._compute_filter()

    def _compute_filter(self):
        """Precompute MMSE filter matrix"""
        reg = max(self.noise_var, 1e-10)
        A = self.R_hh + reg * np.eye(self.n_pilots)

        try:
            self.W_mmse = np.linalg.solve(A, self.R_hh.T).T
        except np.linalg.LinAlgError:
            self.W_mmse = self.R_hh @ np.linalg.pinv(A)

    def estimate(self, h_ls: np.ndarray) -> np.ndarray:
        """
        Apply MMSE estimation using precomputed filter.

        Args:
            h_ls: LS channel estimates

        Returns:
            MMSE channel estimate
        """
        return self.W_mmse @ h_ls

    def update_noise_variance(self, noise_var: float):
        """
        Update noise variance and recompute filter.

        Args:
            noise_var: New noise variance estimate
        """
        self.noise_var = noise_var
        self._compute_filter()


def adaptive_sparse_threshold(cir: np.ndarray,
                               noise_var: float,
                               p_fa: float = 0.01) -> np.ndarray:
    """
    Adaptive thresholding of channel impulse response based on noise floor.

    Uses statistical threshold derived from noise variance to separate
    signal paths from noise. Based on the Rayleigh distribution of
    noise-only tap magnitudes.

    For complex Gaussian noise, |tap|^2 ~ Exponential(noise_var)
    and |tap| ~ Rayleigh(sqrt(noise_var/2)).

    Threshold for false alarm probability P_fa:
        threshold = sqrt(-2 * noise_var * ln(P_fa))

    Args:
        cir: Channel impulse response (complex)
        noise_var: Estimated noise variance (from null subcarriers)
        p_fa: Target false alarm probability (default 0.01 = 1%)

    Returns:
        Sparsified CIR with sub-threshold taps zeroed
    """
    cir = np.asarray(cir, dtype=np.complex128)

    if len(cir) == 0:
        return cir.copy()

    # Handle zero noise variance
    if noise_var <= 0:
        return cir.copy()

    # Compute threshold from Rayleigh distribution
    # For complex Gaussian noise z = x + jy where x,y ~ N(0, noise_var/2)
    # |z| follows Rayleigh distribution with parameter sqrt(noise_var/2)
    # P(|z| > t) = exp(-t^2 / noise_var)
    # Solving: t = sqrt(-noise_var * ln(P_fa))
    threshold = np.sqrt(-noise_var * np.log(p_fa))

    # Zero taps below threshold
    cir_sparse = cir.copy()
    below_threshold = np.abs(cir) < threshold
    cir_sparse[below_threshold] = 0

    return cir_sparse


@dataclass
class ReceiverConfig:
    """Receiver configuration parameters"""
    sample_rate: float = 1.536e6
    fft_size: int = 4096
    cp_length: int = 1536
    n_pilots: int = 251  # Prime for optimal ZC autocorrelation (matches transmitter)
    pilot_spacing: int = 8
    n_data_carriers: int = 1757  # Matches transmitter
    pilot_sequence_root: int = 25
    snr_estimate: float = 10.0  # Default SNR estimate for MMSE
    n_channel_paths: int = 6  # Number of CIR taps to keep (sparse thresholding)
    interpolation_method: str = 'linear'  # 'linear' or 'dft' - linear is default (more robust)
    pilot_boost_db: float = 0.0  # Pilot boost in dB applied at TX (for compensation)
    # MMSE channel estimation options
    use_mmse: bool = False  # Enable MMSE smoothing on pilot estimates
    delay_spread_samples: int = 0  # Expected delay spread in samples (0 = use cp_length)


class ChannelEstimator:
    """
    Channel estimator using pilot symbols.

    Implements LS estimation at pilot positions followed by
    DFT-based interpolation to data subcarriers.

    Optional MMSE smoothing for noise reduction at low SNR.
    Optional sparse path thresholding for noise reduction (keeps n strongest
    CIR taps, zeros others). Based on HF channel sparsity assumption.
    """

    def __init__(self, pilot_indices: np.ndarray, data_indices: np.ndarray,
                 pilot_sequence: np.ndarray, fft_size: int,
                 n_channel_paths: int = 0, pilot_boost_db: float = 0.0,
                 use_mmse: bool = False, delay_spread_samples: int = 0,
                 noise_var: float = 0.1):
        """
        Initialize channel estimator.

        Args:
            pilot_indices: FFT indices of pilot subcarriers
            data_indices: FFT indices of data subcarriers
            pilot_sequence: Known pilot symbols
            fft_size: FFT size
            n_channel_paths: Number of CIR taps to keep (0 = no thresholding)
            pilot_boost_db: Pilot boost in dB applied at TX (for compensation)
            use_mmse: Enable MMSE smoothing on pilot estimates
            delay_spread_samples: Expected delay spread in samples
            noise_var: Initial noise variance estimate
        """
        self.pilot_indices = pilot_indices
        self.data_indices = data_indices
        self.pilot_sequence = pilot_sequence
        self.fft_size = fft_size
        self.n_channel_paths = n_channel_paths
        self.pilot_boost_db = pilot_boost_db
        self.use_mmse = use_mmse

        # Create MMSE estimator if enabled
        self.mmse_estimator = None
        if use_mmse:
            self.mmse_estimator = MMSEChannelEstimator(
                n_pilots=len(pilot_indices),
                delay_spread_samples=delay_spread_samples,
                noise_var=noise_var
            )

        # All active indices
        self.active_indices = np.sort(
            np.concatenate([pilot_indices, data_indices]))

    def estimate(self, rx_freq: np.ndarray,
                 interpolation: str = 'dft') -> np.ndarray:
        """
        Estimate channel frequency response.

        Args:
            rx_freq: Received frequency-domain samples (FFT output)
            interpolation: 'dft', 'linear', or 'none'

        Returns:
            Channel estimate at all active subcarriers
        """
        # LS estimation at pilot positions
        rx_pilots = rx_freq[self.pilot_indices]
        h_pilots = rx_pilots / self.pilot_sequence

        # Compensate for pilot boost if applied at TX
        if self.pilot_boost_db > 0:
            h_pilots = compensate_pilot_boost(h_pilots, self.pilot_boost_db)

        # Apply MMSE smoothing if enabled
        if self.mmse_estimator is not None:
            h_pilots = self.mmse_estimator.estimate(h_pilots)

        if interpolation == 'none':
            # Only return pilot estimates
            return h_pilots

        elif interpolation == 'linear':
            # Linear interpolation to data carriers
            return self._linear_interpolate(h_pilots)

        elif interpolation == 'dft':
            # DFT-based interpolation (better for multipath)
            return self._dft_interpolate(h_pilots)

        else:
            raise ValueError(f"Unknown interpolation: {interpolation}")

    def _linear_interpolate(self, h_pilots: np.ndarray) -> np.ndarray:
        """Linear interpolation from pilots to all active carriers.

        Uses vectorized np.interp for efficiency (67x faster than loop-based).
        """
        h_all = np.zeros(len(self.active_indices), dtype=np.complex128)

        # Map pilot indices to active indices
        pilot_active_idx = np.searchsorted(self.active_indices, self.pilot_indices)
        data_active_idx = np.searchsorted(self.active_indices, self.data_indices)

        # Set pilot values
        h_all[pilot_active_idx] = h_pilots

        # Vectorized interpolation for data positions using np.interp
        # np.interp works on real values, so interpolate real and imag separately
        h_real_interp = np.interp(data_active_idx, pilot_active_idx, h_pilots.real)
        h_imag_interp = np.interp(data_active_idx, pilot_active_idx, h_pilots.imag)
        h_all[data_active_idx] = h_real_interp + 1j * h_imag_interp

        return h_all

    def _dft_interpolate(self, h_pilots: np.ndarray) -> np.ndarray:
        """
        DFT-based interpolation with optional sparse path thresholding.

        WARNING: This method has aliasing issues for channels with delay spread
        exceeding n_pilots samples. For HF channels with long delay spread
        (equatorial, polar, auroral), use 'linear' interpolation instead.

        The fundamental limitation is that IDFT of M pilot estimates produces
        an M-point CIR. If the true CIR has L > M taps, aliasing occurs and
        the channel estimate is corrupted.

        Algorithm (per Zhu/He/Li 2015):
        1. IDFT pilot estimates → CIR estimate (M-point)
        2. Sparse thresholding: keep n_channel_paths strongest taps
        3. Zero-pad CIR to N_active points
        4. DFT → interpolated channel at N_active frequencies

        Sparse thresholding exploits HF channel sparsity (typically 2-4 paths)
        to suppress noise-induced spurious paths in the CIR estimate.

        Use case: Only for benign channels where delay spread < M/sample_rate.
        """
        n_pilots = len(h_pilots)
        n_active = len(self.active_indices)

        # Step 1: IDFT to get channel impulse response estimate
        # The pilots uniformly sample the channel across the active band.
        # IDFT gives us the CIR at M points (aliased if true CIR is longer).
        cir = np.fft.ifft(h_pilots)

        # Step 2: Apply sparse path thresholding if enabled
        # HF channels are sparse (typically 2-6 paths), so we keep only
        # the n_channel_paths strongest taps and zero others.
        if self.n_channel_paths > 0 and self.n_channel_paths < n_pilots:
            magnitudes = np.abs(cir)
            threshold_idx = np.argsort(magnitudes)[-self.n_channel_paths:]
            cir_sparse = np.zeros_like(cir)
            cir_sparse[threshold_idx] = cir[threshold_idx]
            cir = cir_sparse

        # Step 3: Zero-pad CIR to n_active points for interpolation
        # This is sinc interpolation in frequency domain.
        # CIR is arranged as [h[0], h[1], ..., h[M/2-1], h[-M/2], ..., h[-1]]
        # We need to preserve this circular structure when zero-padding.
        cir_padded = np.zeros(n_active, dtype=np.complex128)

        # Causal part: indices 0 to M/2-1 (positive delays)
        n_causal = (n_pilots + 1) // 2
        cir_padded[:n_causal] = cir[:n_causal]

        # Anti-causal part: indices -M/2 to -1 (negative delays / wraparound)
        n_anti = n_pilots - n_causal
        if n_anti > 0:
            cir_padded[-n_anti:] = cir[-n_anti:]

        # Step 4: DFT for interpolated frequency response
        # Output h_interp[k] is the channel at uniformly spaced frequencies
        # within the pilot bandwidth, which maps to active_indices[k].
        h_interp = np.fft.fft(cir_padded)

        return h_interp


class MMSEEqualizer:
    """
    MMSE Frequency-Domain Equalizer.

    Implements: W = H* / (|H|² + 1/SNR)

    This minimizes mean squared error between equalized symbols
    and transmitted symbols under AWGN assumption.
    """

    def __init__(self, snr_linear: float = 10.0):
        """
        Initialize MMSE equalizer.

        Args:
            snr_linear: Linear SNR estimate (not dB)
        """
        self.snr_linear = snr_linear
        self.regularization = 1.0 / snr_linear

    def set_snr(self, snr_db: float):
        """Set SNR from dB value"""
        self.snr_linear = 10 ** (snr_db / 10)
        self.regularization = 1.0 / self.snr_linear

    def equalize(self, rx_symbols: np.ndarray,
                 channel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply MMSE equalization.

        Args:
            rx_symbols: Received frequency-domain symbols
            channel: Channel frequency response estimate

        Returns:
            Tuple of (equalized_symbols, equalizer_coefficients)
        """
        # Handle NaN/Inf values in channel estimate (can occur in harsh conditions)
        channel_clean = np.copy(channel)
        invalid_mask = ~np.isfinite(channel_clean)
        if np.any(invalid_mask):
            # Replace invalid values with small number to avoid divide errors
            # These subcarriers will have poor equalization but won't crash
            channel_clean[invalid_mask] = 1e-10

        # MMSE equalizer: W = H* / (|H|² + 1/SNR)
        h_mag_sq = np.abs(channel_clean) ** 2
        w = np.conj(channel_clean) / (h_mag_sq + self.regularization)

        # Apply equalization
        eq_symbols = rx_symbols * w

        return eq_symbols, w

    def get_post_snr(self, channel: np.ndarray) -> np.ndarray:
        """
        Compute post-equalization SNR for each subcarrier.

        Used for computing accurate LLRs in soft demodulation.

        For ZF/MMSE equalization, the post-equalization SNR scales with
        channel power: SNR_post = |H|² * SNR_input

        Args:
            channel: Channel frequency response

        Returns:
            Post-equalization SNR for each subcarrier
        """
        # Handle NaN/Inf values
        channel_clean = np.where(np.isfinite(channel), channel, 1e-10)
        h_mag_sq = np.abs(channel_clean) ** 2

        # Post-equalization SNR = |H|² * input_SNR
        # Deep fades (low |H|²) yield low post-SNR, used for soft erasure weighting
        post_snr = h_mag_sq * self.snr_linear

        return post_snr


class SCFDEReceiver:
    """
    SC-FDE Receiver.

    Complete receive chain:
    1. CP removal
    2. FFT
    3. Channel estimation
    4. MMSE equalization
    5. Symbol extraction
    """

    def __init__(self, config: Optional[ReceiverConfig] = None):
        """
        Initialize receiver.

        Args:
            config: Receiver configuration
        """
        if config is None:
            config = ReceiverConfig()
        self.config = config

        # Generate pilot sequence (must match transmitter)
        self.pilot_sequence = generate_zadoff_chu(
            config.n_pilots,
            config.pilot_sequence_root
        )

        # Compute subcarrier indices (must match transmitter)
        self._compute_subcarrier_indices()

        # Compute delay spread in samples (use cp_length if not specified)
        delay_spread = config.delay_spread_samples if config.delay_spread_samples > 0 else config.cp_length

        # Compute initial noise variance from SNR estimate
        snr_linear = 10 ** (config.snr_estimate / 10)
        noise_var = 1.0 / snr_linear

        # Initialize channel estimator with sparse thresholding, pilot boost, and MMSE
        self.channel_estimator = ChannelEstimator(
            self.pilot_indices,
            self.data_indices,
            self.pilot_sequence,
            config.fft_size,
            n_channel_paths=config.n_channel_paths,
            pilot_boost_db=config.pilot_boost_db,
            use_mmse=config.use_mmse,
            delay_spread_samples=delay_spread,
            noise_var=noise_var
        )

        # Initialize MMSE equalizer
        snr_linear = 10 ** (config.snr_estimate / 10)
        self.equalizer = MMSEEqualizer(snr_linear)

    def _compute_subcarrier_indices(self):
        """Compute pilot and data subcarrier indices using shared utility (must match TX)."""
        cfg = self.config
        self.pilot_indices, self.data_indices, self.null_indices = \
            compute_subcarrier_indices(
                cfg.fft_size,
                cfg.n_pilots,
                cfg.n_data_carriers,
                cfg.pilot_spacing
            )

    def set_snr_estimate(self, snr_db: float):
        """Update SNR estimate for MMSE equalization"""
        self.config.snr_estimate = snr_db
        self.equalizer.set_snr(snr_db)

    def get_null_indices(self) -> np.ndarray:
        """
        Get indices of null (unused) subcarriers.

        Null subcarriers include:
        - DC (index 0)
        - Guard band (upper frequencies beyond active carriers)

        Returns:
            Array of null subcarrier indices
        """
        return self.null_indices

    def estimate_noise_variance(self, rx_freq: np.ndarray) -> float:
        """
        Estimate noise variance from null subcarriers.

        Uses the standalone estimate_noise_variance function with
        the receiver's computed null indices.

        Args:
            rx_freq: FFT output from received signal

        Returns:
            Estimated noise variance
        """
        null_indices = self.get_null_indices()
        return estimate_noise_variance(rx_freq, null_indices)

    def demodulate_block(self, rx_block: np.ndarray,
                         return_channel: bool = False,
                         return_post_snr: bool = False,
                         return_rx_data: bool = False
                         ) -> Tuple[np.ndarray, ...]:
        """
        Demodulate one SC-FDE block.

        Args:
            rx_block: Received time-domain samples (including CP)
            return_channel: If True, also return channel estimate
            return_post_snr: If True, also return post-equalization SNR per subcarrier
            return_rx_data: If True, also return pre-equalization data symbols

        Returns:
            Tuple of (equalized_data_symbols, [channel_estimate], [post_snr], [rx_data])
        """
        cfg = self.config

        expected_len = cfg.fft_size + cfg.cp_length
        if len(rx_block) != expected_len:
            raise ValueError(
                f"Expected {expected_len} samples, got {len(rx_block)}")

        # Remove CP
        rx_no_cp = rx_block[cfg.cp_length:]

        # FFT
        rx_freq = np.fft.fft(rx_no_cp) / np.sqrt(cfg.fft_size)

        # Channel estimation with configured interpolation method
        h_est = self.channel_estimator.estimate(rx_freq, interpolation=cfg.interpolation_method)

        # Extract indices for data
        all_indices = np.sort(
            np.concatenate([self.pilot_indices, self.data_indices]))
        data_mask = np.isin(all_indices, self.data_indices)

        # Get received data symbols
        rx_data = rx_freq[self.data_indices]

        # Get channel at data positions
        h_data = h_est[data_mask]

        # MMSE equalization
        eq_data, _ = self.equalizer.equalize(rx_data, h_data)

        # Build return tuple
        result = [eq_data]
        if return_channel:
            result.append(h_data)
        if return_post_snr:
            post_snr = self.equalizer.get_post_snr(h_data)
            result.append(post_snr)
        if return_rx_data:
            result.append(rx_data)

        return tuple(result) if len(result) > 1 else (eq_data,)

    def demodulate_frame(self, rx_frame: np.ndarray,
                         n_blocks: int,
                         return_post_snr: bool = False
                         ) -> Tuple[np.ndarray, ...]:
        """
        Demodulate complete frame.

        The input rx_frame should start at the first data block (preamble already
        removed by the synchronizer).

        Args:
            rx_frame: Received time-domain frame starting at first data block
            n_blocks: Number of data blocks in frame
            return_post_snr: If True, also return post-equalization SNR per subcarrier

        Returns:
            Tuple of (all_equalized_symbols, channel_estimates, [post_snr_all])
        """
        cfg = self.config

        block_len = cfg.fft_size + cfg.cp_length
        start_idx = 0

        all_symbols = []
        all_channels = []
        all_post_snr = []

        for i in range(n_blocks):
            block_start = start_idx + i * block_len
            block_end = block_start + block_len

            if block_end > len(rx_frame):
                break

            rx_block = rx_frame[block_start:block_end]
            result = self.demodulate_block(rx_block, return_channel=True,
                                           return_post_snr=return_post_snr)

            eq_symbols = result[0]
            h_est = result[1]
            all_symbols.append(eq_symbols)
            all_channels.append(h_est)

            if return_post_snr:
                post_snr = result[2]
                all_post_snr.append(post_snr)

        # Handle case where rx_frame is too short for any blocks
        if not all_symbols:
            raise ValueError(
                f"Received frame too short: got {len(rx_frame)} samples, "
                f"need at least {block_len} for one block"
            )

        if return_post_snr:
            return (np.concatenate(all_symbols),
                    np.concatenate(all_channels),
                    np.concatenate(all_post_snr))
        return np.concatenate(all_symbols), np.concatenate(all_channels)

    def get_noise_variance(self, rx_symbols: np.ndarray,
                           eq_symbols: np.ndarray,
                           channel: np.ndarray) -> float:
        """
        Estimate noise variance from received and equalized symbols.

        Args:
            rx_symbols: Received symbols
            eq_symbols: Equalized symbols
            channel: Channel estimate

        Returns:
            Estimated noise variance
        """
        # Reconstruct what we would have received
        expected_rx = eq_symbols * channel
        error = rx_symbols - expected_rx
        return np.mean(np.abs(error) ** 2)


def test_receiver():
    """Test SC-FDE receiver"""
    print("Testing SC-FDE Receiver...")

    from .transmitter import SCFDETransmitter, TransmitterConfig

    # Create matching TX/RX config
    tx_config = TransmitterConfig(
        fft_size=4096,
        cp_length=1536,
        n_pilots=256,
        n_data_carriers=1792,
    )
    rx_config = ReceiverConfig(
        fft_size=tx_config.fft_size,
        cp_length=tx_config.cp_length,
        n_pilots=tx_config.n_pilots,
        n_data_carriers=tx_config.n_data_carriers,
        snr_estimate=20.0,
    )

    tx = SCFDETransmitter(tx_config)
    rx = SCFDEReceiver(rx_config)

    print(f"\nLoopback test (no channel):")
    np.random.seed(42)

    # Generate QPSK symbols
    n_symbols = tx_config.n_data_carriers
    bits = np.random.randint(0, 2, n_symbols * 2, dtype=np.int8)
    tx_symbols = np.array([
        (1 - 2 * bits[2*i]) + 1j * (1 - 2 * bits[2*i + 1])
        for i in range(n_symbols)
    ]) / np.sqrt(2)

    # Transmit
    tx_block = tx.modulate_block(tx_symbols)

    # Perfect channel (loopback)
    rx_block = tx_block

    # Receive
    eq_symbols, h_est = rx.demodulate_block(rx_block, return_channel=True)

    # Check symbol error
    mse = np.mean(np.abs(tx_symbols - eq_symbols) ** 2)
    print(f"  MSE (loopback): {10*np.log10(mse):.1f} dB")

    # Test with AWGN
    print(f"\nAWGN channel test:")
    for snr_db in [10, 20, 30]:
        snr_linear = 10 ** (snr_db / 10)
        noise_power = 1 / snr_linear
        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(len(tx_block)) + 1j * np.random.randn(len(tx_block)))

        rx_block = tx_block + noise
        rx.set_snr_estimate(snr_db)

        eq_symbols, _ = rx.demodulate_block(rx_block, return_channel=True)
        mse = np.mean(np.abs(tx_symbols - eq_symbols) ** 2)
        evm = np.sqrt(mse) * 100
        print(f"  SNR {snr_db} dB: MSE={10*np.log10(mse):.1f} dB, EVM={evm:.1f}%")

    # Test with simple frequency-selective channel
    print(f"\nFrequency-selective channel test:")
    # Two-tap channel: h = [1, 0.5*exp(j*pi/4)] with delay
    h_time = np.zeros(tx_config.fft_size, dtype=np.complex128)
    h_time[0] = 1.0
    h_time[100] = 0.5 * np.exp(1j * np.pi / 4)  # 100 sample delay

    # Apply channel in frequency domain
    h_freq = np.fft.fft(h_time)

    # Transmit through channel
    tx_no_cp = tx_block[tx_config.cp_length:]
    tx_freq = np.fft.fft(tx_no_cp)
    rx_freq = tx_freq * h_freq
    rx_no_cp = np.fft.ifft(rx_freq)

    # Add CP back (channel will have caused ISI without it, but we have it)
    rx_block = np.concatenate([rx_no_cp[-tx_config.cp_length:], rx_no_cp])

    # Receive and equalize
    rx.set_snr_estimate(30)  # High SNR
    eq_symbols, h_est = rx.demodulate_block(rx_block, return_channel=True)

    mse = np.mean(np.abs(tx_symbols - eq_symbols) ** 2)
    print(f"  MSE after MMSE equalization: {10*np.log10(mse):.1f} dB")

    # Verify channel estimation
    h_at_data = h_freq[rx.data_indices]
    h_est_error = np.mean(np.abs(h_at_data - h_est) ** 2)
    print(f"  Channel estimation MSE: {10*np.log10(h_est_error):.1f} dB")

    print("\nReceiver tests passed!")
    return True


if __name__ == '__main__':
    test_receiver()
