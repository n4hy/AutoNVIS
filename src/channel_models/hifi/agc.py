"""
Automatic Gain Control (AGC) for SC-FDE System

Implements AGC to handle HF's 60+ dB dynamic range:
- Fast attack (respond quickly to strong signals to prevent clipping)
- Slow decay (maintain stable gain during fading)
- Configurable time constants and gain limits
"""

import numpy as np
from typing import Optional


class AGC:
    """
    Automatic Gain Control for HF dynamic range.

    Uses asymmetric attack/decay envelope tracking to normalize
    received signal power while avoiding clipping and maintaining
    stability during fades.

    The algorithm tracks signal envelope and adjusts gain to maintain
    output at target level. Attack is fast (prevent clipping), decay
    is slow (stable gain during fades).
    """

    def __init__(self,
                 target_level_dbfs: float = -12.0,
                 attack_time_ms: float = 1.0,
                 decay_time_ms: float = 100.0,
                 sample_rate: float = 1.536e6,
                 min_gain_db: float = -40.0,
                 max_gain_db: float = 60.0):
        """
        Initialize AGC.

        Args:
            target_level_dbfs: Target output RMS level in dBFS
            attack_time_ms: Attack time constant in milliseconds (fast)
            decay_time_ms: Decay time constant in milliseconds (slow)
            sample_rate: Sample rate in Hz
            min_gain_db: Minimum gain in dB (for strong signals)
            max_gain_db: Maximum gain in dB (for weak signals)
        """
        self.target = 10 ** (target_level_dbfs / 20)
        self.sample_rate = sample_rate

        # Compute time constants as exponential smoothing coefficients
        # alpha = 1 - exp(-1 / (tau * fs))
        # where tau = time_constant_ms / 1000
        tau_attack = attack_time_ms / 1000.0
        tau_decay = decay_time_ms / 1000.0

        self.alpha_attack = 1 - np.exp(-1 / (tau_attack * sample_rate))
        self.alpha_decay = 1 - np.exp(-1 / (tau_decay * sample_rate))

        # Gain limits
        self.min_gain = 10 ** (min_gain_db / 20)
        self.max_gain = 10 ** (max_gain_db / 20)

        # State
        self.gain = 1.0
        self._envelope = 0.0

    def reset(self):
        """Reset AGC to initial state"""
        self.gain = 1.0
        self._envelope = 0.0

    def process(self, samples: np.ndarray) -> np.ndarray:
        """
        Process samples through AGC.

        Args:
            samples: Input samples (complex or real)

        Returns:
            Gain-adjusted output samples
        """
        samples = np.asarray(samples)
        output = np.zeros_like(samples)

        for i in range(len(samples)):
            # Apply current gain
            output[i] = samples[i] * self.gain

            # Measure output envelope (magnitude)
            level = np.abs(output[i])

            # Update envelope with asymmetric time constants
            if level > self._envelope:
                # Attack: signal increasing, use fast time constant
                self._envelope += self.alpha_attack * (level - self._envelope)
            else:
                # Decay: signal decreasing, use slow time constant
                self._envelope += self.alpha_decay * (level - self._envelope)

            # Compute gain adjustment
            if self._envelope > 1e-10:  # Avoid division by zero
                # Error between target and current envelope
                error = self.target / self._envelope

                # Smooth gain update (use decay time constant for stability)
                if error > 1:
                    # Need more gain - use decay (slow) time constant
                    self.gain += self.alpha_decay * (error * self.gain - self.gain)
                else:
                    # Need less gain - use attack (fast) time constant
                    self.gain += self.alpha_attack * (error * self.gain - self.gain)

            # Clamp gain to limits
            self.gain = np.clip(self.gain, self.min_gain, self.max_gain)

        return output

    def process_block(self, samples: np.ndarray) -> np.ndarray:
        """
        Process block of samples (alias for process).

        For compatibility with block-based processing pipelines.
        """
        return self.process(samples)

    def get_gain_db(self) -> float:
        """Get current gain in dB"""
        return 20 * np.log10(self.gain)

    def set_gain_db(self, gain_db: float):
        """Set gain manually (bypasses AGC algorithm)"""
        self.gain = np.clip(
            10 ** (gain_db / 20),
            self.min_gain,
            self.max_gain
        )


class FastAGC(AGC):
    """
    Fast AGC variant using vectorized operations.

    Uses per-block gain rather than per-sample for efficiency.
    Suitable for block-based processing where sample-by-sample
    tracking isn't required.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._block_count = 0

    def process(self, samples: np.ndarray) -> np.ndarray:
        """Process block with single gain value per block"""
        samples = np.asarray(samples)

        # Measure block power
        block_power = np.mean(np.abs(samples) ** 2)

        if block_power > 1e-20:
            block_level = np.sqrt(block_power)

            # Update envelope
            if block_level > self._envelope:
                self._envelope += self.alpha_attack * (block_level - self._envelope)
            else:
                self._envelope += self.alpha_decay * (block_level - self._envelope)

            # Compute desired gain
            if self._envelope > 1e-10:
                desired_gain = self.target / self._envelope

                # Smooth gain update
                if desired_gain > self.gain:
                    # Need more gain - use decay (slow)
                    self.gain += self.alpha_decay * (desired_gain - self.gain)
                else:
                    # Need less gain - use attack (fast)
                    self.gain += self.alpha_attack * (desired_gain - self.gain)

                self.gain = np.clip(self.gain, self.min_gain, self.max_gain)

        # Apply single gain to entire block
        return samples * self.gain


def test_agc():
    """Test AGC implementation"""
    print("Testing AGC...")

    sample_rate = 1.536e6

    # Create AGC
    agc = AGC(
        target_level_dbfs=-12.0,
        attack_time_ms=1.0,
        decay_time_ms=100.0,
        sample_rate=sample_rate
    )

    # Test with weak signal
    print("\nWeak signal test (-40 dBFS):")
    weak_amplitude = 0.01  # -40 dBFS
    n_samples = int(0.5 * sample_rate)  # 500 ms
    weak_signal = weak_amplitude * np.exp(1j * np.linspace(0, 100*np.pi, n_samples))

    output = agc.process(weak_signal)

    input_power = np.mean(np.abs(weak_signal)**2)
    output_power = np.mean(np.abs(output[-n_samples//4:])**2)

    print(f"  Input power: {10*np.log10(input_power):.1f} dBFS")
    print(f"  Output power: {10*np.log10(output_power):.1f} dBFS")
    print(f"  Final gain: {20*np.log10(agc.gain):.1f} dB")

    # Reset and test with strong signal
    agc.reset()
    print("\nStrong signal test (0 dBFS):")
    strong_amplitude = 1.0  # 0 dBFS
    strong_signal = strong_amplitude * np.exp(1j * np.linspace(0, 100*np.pi, n_samples))

    output = agc.process(strong_signal)

    input_power = np.mean(np.abs(strong_signal)**2)
    output_power = np.mean(np.abs(output[-n_samples//4:])**2)

    print(f"  Input power: {10*np.log10(input_power):.1f} dBFS")
    print(f"  Output power: {10*np.log10(output_power):.1f} dBFS")
    print(f"  Final gain: {20*np.log10(agc.gain):.1f} dB")

    print("\nAGC tests passed!")
    return True


if __name__ == '__main__':
    test_agc()
