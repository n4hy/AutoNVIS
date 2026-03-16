"""
Feature Configuration for SC-FDE System Evaluation

Provides a centralized configuration to enable/disable features
for before/after comparison testing.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FeatureConfig:
    """
    Configuration for enabling/disabling SC-FDE features.

    Used for systematic evaluation of each feature's contribution
    to system performance.
    """

    # Phase 1-2: Foundation Features
    use_noise_estimation: bool = True
    """Use blind noise estimation from null subcarriers vs fixed estimate"""

    soft_erasure_threshold_db: Optional[float] = None
    """Soft erasure threshold in dB. None = disabled (weights=1.0)"""

    sparse_threshold_n_paths: int = 0
    """Number of CIR paths to keep. 0 = disabled (keep all)"""

    pilot_boost_db: float = 3.0
    """Pilot power boost in dB. 0 = no boost, 3.0 = recommended default"""

    # Phase 3: Medium Term Features
    ldpc_block_length: int = 648
    """LDPC codeword length: 648, 1296, or 1944"""

    use_freq_interleaving: bool = True
    """Enable frequency-domain symbol interleaving"""

    use_mmse_estimation: bool = True
    """Enable MMSE smoothing on pilot channel estimates (provides ~40% BER improvement)"""

    use_agc: bool = True
    """Enable automatic gain control (normalizes signal level, improves robustness)"""

    agc_target_dbfs: float = -12.0
    """AGC target level in dBFS"""

    # Phase 4: Advanced Features
    use_cfo_tracking: bool = False
    """Enable CFO tracking via PLL"""

    cfo_loop_bandwidth_hz: float = 5.0
    """CFO tracker loop bandwidth"""

    use_dd_tracking: bool = False
    """Enable decision-directed channel tracking"""

    dd_alpha: float = 0.3
    """DD tracker smoothing factor"""

    use_link_adaptation: bool = False
    """Enable adaptive modulation and coding"""

    ibdfe_iterations: int = 1
    """IB-DFE iterations. 1 = linear MMSE only"""

    ibdfe_feedback_mode: str = 'soft'
    """IB-DFE feedback mode: 'soft' (tanh of LLRs), 'hard' (sliced decisions), 'auto'"""

    # LDPC Decoder Configuration
    ldpc_max_iterations: int = 8
    """Maximum LDPC decoder iterations. Default 8 for real-time operation (with early termination)."""

    ldpc_early_termination: bool = True
    """Enable early termination on syndrome convergence. False = always run max iterations."""

    # Turbo Equalizer Configuration
    turbo_iterations: int = 4
    """Turbo equalizer outer iterations. Default 4, use 2 for fixed-iteration mode."""

    turbo_early_termination: bool = True
    """Enable early termination on BER convergence. False = always run max iterations."""

    use_turbo_equalization: bool = False
    """Enable turbo equalization in the receive path. Uses legacy LDPC for compatibility."""

    # Scrambler Configuration
    use_scrambling: bool = True
    """Enable bit scrambling after LDPC encode to randomize transmitted bits."""

    scrambler_seed: int = 0x7F
    """LFSR seed for scrambler (default 0x7F = all ones for 7-bit LFSR)."""

    @classmethod
    def baseline(cls) -> 'FeatureConfig':
        """Create baseline config with all features disabled."""
        return cls(
            use_noise_estimation=False,
            soft_erasure_threshold_db=None,
            sparse_threshold_n_paths=0,
            pilot_boost_db=0.0,
            ldpc_block_length=648,
            use_freq_interleaving=False,
            use_mmse_estimation=False,
            use_agc=False,
            use_cfo_tracking=False,
            use_dd_tracking=False,
            use_link_adaptation=False,
            ibdfe_iterations=1,
            ibdfe_feedback_mode='soft',
            ldpc_max_iterations=8,
            ldpc_early_termination=True,
            turbo_iterations=4,
            turbo_early_termination=True,
            use_scrambling=False,
        )

    @classmethod
    def all_enabled(cls) -> 'FeatureConfig':
        """Create config with all features enabled."""
        return cls(
            use_noise_estimation=True,
            soft_erasure_threshold_db=0.0,
            sparse_threshold_n_paths=6,
            pilot_boost_db=3.0,
            ldpc_block_length=1944,
            use_freq_interleaving=True,
            use_mmse_estimation=True,
            use_agc=True,
            use_cfo_tracking=True,
            use_dd_tracking=True,
            use_link_adaptation=True,
            ibdfe_iterations=3,
            ibdfe_feedback_mode='soft',
            ldpc_max_iterations=8,
            ldpc_early_termination=True,
            turbo_iterations=4,
            turbo_early_termination=True,
        )

    @classmethod
    def fixed_iteration(cls, ldpc_iter: int = 6, turbo_iter: int = 2,
                        ibdfe_iter: int = 2) -> 'FeatureConfig':
        """
        Create config optimized for fixed-iteration (deterministic latency) operation.

        Args:
            ldpc_iter: Fixed LDPC iterations (default 6)
            turbo_iter: Fixed turbo iterations (default 2)
            ibdfe_iter: Fixed IB-DFE iterations (default 2)

        Returns:
            FeatureConfig with early termination disabled
        """
        return cls(
            use_noise_estimation=True,
            soft_erasure_threshold_db=None,
            sparse_threshold_n_paths=0,
            pilot_boost_db=3.0,
            ldpc_block_length=648,
            use_freq_interleaving=True,
            use_mmse_estimation=True,
            use_agc=True,
            use_cfo_tracking=False,
            use_dd_tracking=False,
            use_link_adaptation=False,
            ibdfe_iterations=ibdfe_iter,
            ibdfe_feedback_mode='hard',  # Hard decisions for deterministic timing
            ldpc_max_iterations=ldpc_iter,
            ldpc_early_termination=False,  # Fixed iterations
            turbo_iterations=turbo_iter,
            turbo_early_termination=False,  # Fixed iterations
        )

    def with_feature(self, feature_name: str, enabled: bool = True) -> 'FeatureConfig':
        """
        Create a copy with a single feature toggled.

        Args:
            feature_name: Name of feature to toggle
            enabled: Whether to enable (True) or disable (False)

        Returns:
            New FeatureConfig with feature toggled
        """
        import copy
        new_config = copy.copy(self)

        feature_map = {
            'noise_estimation': ('use_noise_estimation', enabled),
            'soft_erasure': ('soft_erasure_threshold_db', 0.0 if enabled else None),
            'sparse_threshold': ('sparse_threshold_n_paths', 6 if enabled else 0),
            'pilot_boost': ('pilot_boost_db', 3.0 if enabled else 0.0),
            'extended_ldpc': ('ldpc_block_length', 1944 if enabled else 648),
            'freq_interleaving': ('use_freq_interleaving', enabled),
            'mmse_estimation': ('use_mmse_estimation', enabled),
            'agc': ('use_agc', enabled),
            'cfo_tracking': ('use_cfo_tracking', enabled),
            'dd_tracking': ('use_dd_tracking', enabled),
            'link_adaptation': ('use_link_adaptation', enabled),
            'ibdfe': ('ibdfe_iterations', 3 if enabled else 1),
            'ldpc_early_termination': ('ldpc_early_termination', enabled),
            'turbo_early_termination': ('turbo_early_termination', enabled),
            'fixed_ldpc_6': ('ldpc_max_iterations', 6 if enabled else 50),
            'fixed_ldpc_10': ('ldpc_max_iterations', 10 if enabled else 50),
            'fixed_turbo_2': ('turbo_iterations', 2 if enabled else 4),
            'hard_feedback': ('ibdfe_feedback_mode', 'hard' if enabled else 'soft'),
            'scrambling': ('use_scrambling', enabled),
        }

        if feature_name not in feature_map:
            raise ValueError(f"Unknown feature: {feature_name}. Valid: {list(feature_map.keys())}")

        attr_name, value = feature_map[feature_name]
        setattr(new_config, attr_name, value)
        return new_config

    def describe(self) -> str:
        """Return human-readable description of enabled features."""
        enabled = []
        if self.use_noise_estimation:
            enabled.append("Noise Est")
        if self.soft_erasure_threshold_db is not None:
            enabled.append(f"Soft Erasure ({self.soft_erasure_threshold_db}dB)")
        if self.sparse_threshold_n_paths > 0:
            enabled.append(f"Sparse ({self.sparse_threshold_n_paths} paths)")
        if self.pilot_boost_db > 0:
            enabled.append(f"Pilot Boost ({self.pilot_boost_db}dB)")
        if self.ldpc_block_length > 648:
            enabled.append(f"LDPC n={self.ldpc_block_length}")
        if self.use_freq_interleaving:
            enabled.append("Freq Interleave")
        if self.use_mmse_estimation:
            enabled.append("MMSE Est")
        if self.use_agc:
            enabled.append("AGC")
        if self.use_cfo_tracking:
            enabled.append("CFO Track")
        if self.use_dd_tracking:
            enabled.append("DD Track")
        if self.use_link_adaptation:
            enabled.append("Link Adapt")
        if self.ibdfe_iterations > 1:
            enabled.append(f"IB-DFE ({self.ibdfe_iterations} iter, {self.ibdfe_feedback_mode})")
        if self.use_scrambling:
            enabled.append("Scrambler")
        if self.ldpc_max_iterations != 30:
            enabled.append(f"LDPC {self.ldpc_max_iterations} iter")
        if not self.ldpc_early_termination:
            enabled.append("LDPC fixed-iter")
        if self.turbo_iterations != 4:
            enabled.append(f"Turbo {self.turbo_iterations} iter")
        if not self.turbo_early_termination:
            enabled.append("Turbo fixed-iter")

        if not enabled:
            return "Baseline (all disabled)"
        return ", ".join(enabled)


# Feature metadata for evaluation
FEATURE_INFO = {
    'noise_estimation': {
        'name': 'Blind Noise Estimation',
        'description': 'Estimate noise variance from null subcarriers',
        'expected_gain_db': 0.3,
        'best_channel': 'all',
    },
    'soft_erasure': {
        'name': 'Soft Null-Subcarrier Erasure',
        'description': 'Weight LLRs by subcarrier reliability',
        'expected_gain_db': 0.8,
        'best_channel': 'equatorial',
    },
    'sparse_threshold': {
        'name': 'Adaptive Sparse Thresholding',
        'description': 'Remove noise-only CIR taps',
        'expected_gain_db': 0.5,
        'best_channel': 'midlatitude',
    },
    'pilot_boost': {
        'name': 'Pilot Power Boosting',
        'description': 'Increase pilot power for better channel estimation',
        'expected_gain_db': 1.0,
        'best_channel': 'all',
    },
    'extended_ldpc': {
        'name': 'Extended LDPC Codes',
        'description': 'Longer codewords (n=1944) for better coding gain',
        'expected_gain_db': 0.4,
        'best_channel': 'all',
    },
    'freq_interleaving': {
        'name': 'Frequency-Domain Interleaving',
        'description': 'Spread symbols across frequency for diversity',
        'expected_gain_db': 2.0,
        'best_channel': 'equatorial',
    },
    'mmse_estimation': {
        'name': 'MMSE Channel Estimation',
        'description': 'MMSE smoothing on pilot estimates for noise reduction',
        'expected_gain_db': 1.5,
        'best_channel': 'midlatitude',
    },
    'agc': {
        'name': 'Automatic Gain Control',
        'description': 'Normalize signal level for ADC dynamic range',
        'expected_gain_db': 0.0,  # Not BER gain, operational
        'best_channel': 'polar',
    },
    'cfo_tracking': {
        'name': 'CFO Tracking',
        'description': 'Track carrier frequency offset via PLL',
        'expected_gain_db': 0.0,  # Depends on CFO present
        'best_channel': 'polar',
    },
    'dd_tracking': {
        'name': 'Decision-Directed Channel Tracking',
        'description': 'Refine channel estimate using decoded symbols',
        'expected_gain_db': 1.0,
        'best_channel': 'polar',
    },
    'link_adaptation': {
        'name': 'Link Adaptation / AMC',
        'description': 'Adapt modulation and coding to channel quality',
        'expected_gain_db': 0.0,  # Throughput gain, not BER
        'best_channel': 'all',
    },
    'ibdfe': {
        'name': 'IB-DFE Equalizer',
        'description': 'Iterative block decision feedback equalization',
        'expected_gain_db': 2.0,
        'best_channel': 'equatorial',
    },
    'scrambling': {
        'name': 'Bit Scrambler',
        'description': 'LFSR-based scrambling to randomize coded bits and reduce PAPR',
        'expected_gain_db': 0.0,  # Not BER gain, PAPR reduction
        'best_channel': 'all',
    },
}
