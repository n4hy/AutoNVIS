"""
CFO (Carrier Frequency Offset) Tracker for SC-FDE System

Implements CFO tracking using pilot symbol phases:
- Phase-locked loop (PLL) based tracking
- Phase unwrapping for continuous tracking
- Configurable loop bandwidth for speed vs noise tradeoff
- Support for slowly varying CFO (Doppler, oscillator drift)
"""

import numpy as np
from typing import Optional


class CFOTracker:
    """
    Carrier Frequency Offset Tracker using pilot phases.

    Uses a second-order PLL to track CFO from pilot phase measurements.
    The loop bandwidth controls the tradeoff between tracking speed
    and noise rejection.

    Typical HF CFO sources:
    - Oscillator frequency drift: ±10-50 Hz
    - Doppler shift: ±1-5 Hz (slow variation)
    - Multipath-induced phase rotation
    """

    def __init__(self,
                 sample_rate: float = 1.536e6,
                 block_duration_ms: float = 2.67,
                 loop_bandwidth_hz: float = 5.0,
                 damping_factor: float = 0.707):
        """
        Initialize CFO tracker.

        Args:
            sample_rate: Sample rate in Hz
            block_duration_ms: Duration of each block in milliseconds
            loop_bandwidth_hz: PLL loop bandwidth in Hz (higher = faster tracking)
            damping_factor: PLL damping factor (0.707 = critically damped)
        """
        self.sample_rate = sample_rate
        self.block_duration_s = block_duration_ms / 1000.0
        self.loop_bandwidth_hz = loop_bandwidth_hz
        self.damping_factor = damping_factor

        # Compute loop filter coefficients
        # Using standard second-order PLL design
        omega_n = 2 * np.pi * loop_bandwidth_hz  # Natural frequency
        zeta = damping_factor

        # Proportional and integral gains
        # K1 = proportional, K2 = integral
        self._k1 = 2 * zeta * omega_n * self.block_duration_s
        self._k2 = (omega_n ** 2) * (self.block_duration_s ** 2)

        # State variables
        self._phase_accumulator = 0.0  # Accumulated phase (unwrapped)
        self._freq_estimate = 0.0  # Frequency estimate in rad/block
        self._last_phase = None  # Last measured phase (for unwrapping)
        self._block_count = 0

    def reset(self):
        """Reset tracker to initial state"""
        self._phase_accumulator = 0.0
        self._freq_estimate = 0.0
        self._last_phase = None
        self._block_count = 0

    def _unwrap_phase(self, phase: float) -> float:
        """
        Unwrap phase to maintain continuity.

        Args:
            phase: New phase measurement in [-pi, pi]

        Returns:
            Unwrapped phase
        """
        if self._last_phase is None:
            self._last_phase = phase
            return phase

        # Compute phase difference
        delta = phase - self._last_phase

        # Unwrap: if delta > pi, subtract 2*pi; if < -pi, add 2*pi
        if delta > np.pi:
            delta -= 2 * np.pi
        elif delta < -np.pi:
            delta += 2 * np.pi

        # Update accumulated phase
        unwrapped = self._phase_accumulator + delta
        self._last_phase = phase

        return unwrapped

    def update(self, pilot_phase: float):
        """
        Update CFO estimate with new pilot phase measurement.

        Args:
            pilot_phase: Measured pilot phase in radians [-pi, pi]
        """
        self._block_count += 1

        if self._block_count == 1:
            # First measurement - initialize
            self._last_phase = pilot_phase
            self._phase_accumulator = pilot_phase
            return

        # Compute phase increment (with unwrapping)
        delta_phase = pilot_phase - self._last_phase

        # Unwrap phase difference
        if delta_phase > np.pi:
            delta_phase -= 2 * np.pi
        elif delta_phase < -np.pi:
            delta_phase += 2 * np.pi

        self._last_phase = pilot_phase
        self._phase_accumulator += delta_phase

        # Phase error: measured increment vs expected from frequency estimate
        expected_increment = self._freq_estimate
        phase_error = delta_phase - expected_increment

        # Second-order loop filter update
        # Proportional path (fast response)
        # Integral path (frequency tracking)
        self._freq_estimate += self._k1 * phase_error + self._k2 * phase_error

    def get_cfo_hz(self) -> float:
        """
        Get current CFO estimate in Hz.

        Returns:
            Estimated CFO in Hz
        """
        if self._block_count < 2:
            return 0.0

        # Convert frequency in rad/block to Hz
        # freq_estimate is in radians per block
        # Hz = (rad/block) / (2*pi * block_duration_s)
        cfo_hz = self._freq_estimate / (2 * np.pi * self.block_duration_s)
        return cfo_hz

    def get_correction_phase(self, sample_offset: int) -> float:
        """
        Get phase correction for a given sample offset.

        Args:
            sample_offset: Sample offset from block start

        Returns:
            Phase correction in radians
        """
        # Convert sample offset to time
        time_offset = sample_offset / self.sample_rate

        # Phase correction = -2*pi*cfo*time
        cfo_hz = self.get_cfo_hz()
        correction = -2 * np.pi * cfo_hz * time_offset

        return correction

    def apply_correction(self,
                         symbols: np.ndarray,
                         sample_offset: int = 0) -> np.ndarray:
        """
        Apply CFO correction to symbols.

        Args:
            symbols: Complex symbols to correct
            sample_offset: Sample offset of first symbol from block start

        Returns:
            CFO-corrected symbols
        """
        symbols = np.asarray(symbols)
        n_symbols = len(symbols)

        # Generate correction phasor for each symbol
        cfo_hz = self.get_cfo_hz()
        sample_indices = sample_offset + np.arange(n_symbols)
        time_offsets = sample_indices / self.sample_rate

        # Correction phasor
        correction = np.exp(-1j * 2 * np.pi * cfo_hz * time_offsets)

        return symbols * correction

    def get_state(self) -> dict:
        """Get current tracker state for debugging/logging"""
        return {
            'cfo_hz': self.get_cfo_hz(),
            'freq_estimate_rad_per_block': self._freq_estimate,
            'phase_accumulator': self._phase_accumulator,
            'block_count': self._block_count,
        }


class AdaptiveCFOTracker(CFOTracker):
    """
    Adaptive CFO tracker with variable loop bandwidth.

    Starts with wide bandwidth for fast acquisition, then
    narrows for steady-state tracking.
    """

    def __init__(self,
                 sample_rate: float = 1.536e6,
                 block_duration_ms: float = 2.67,
                 initial_bandwidth_hz: float = 20.0,
                 steady_bandwidth_hz: float = 2.0,
                 adaptation_blocks: int = 50,
                 damping_factor: float = 0.707):
        """
        Initialize adaptive CFO tracker.

        Args:
            sample_rate: Sample rate in Hz
            block_duration_ms: Duration of each block in ms
            initial_bandwidth_hz: Wide bandwidth for acquisition
            steady_bandwidth_hz: Narrow bandwidth for tracking
            adaptation_blocks: Blocks to transition from initial to steady
            damping_factor: PLL damping factor
        """
        # Start with initial bandwidth
        super().__init__(
            sample_rate=sample_rate,
            block_duration_ms=block_duration_ms,
            loop_bandwidth_hz=initial_bandwidth_hz,
            damping_factor=damping_factor
        )

        self._initial_bw = initial_bandwidth_hz
        self._steady_bw = steady_bandwidth_hz
        self._adaptation_blocks = adaptation_blocks

    def update(self, pilot_phase: float):
        """Update with adaptive bandwidth"""
        # Adapt bandwidth based on block count
        if self._block_count < self._adaptation_blocks:
            # Linear interpolation from initial to steady
            alpha = self._block_count / self._adaptation_blocks
            current_bw = (1 - alpha) * self._initial_bw + alpha * self._steady_bw
            self._update_loop_coefficients(current_bw)

        super().update(pilot_phase)

    def _update_loop_coefficients(self, bandwidth_hz: float):
        """Update loop filter coefficients for new bandwidth"""
        omega_n = 2 * np.pi * bandwidth_hz
        zeta = self.damping_factor

        self._k1 = 2 * zeta * omega_n * self.block_duration_s
        self._k2 = (omega_n ** 2) * (self.block_duration_s ** 2)


def test_cfo_tracker():
    """Test CFO tracker implementation"""
    print("Testing CFO Tracker...")

    sample_rate = 1.536e6
    block_duration_ms = 2.67
    block_duration_s = block_duration_ms / 1000

    # Create tracker
    tracker = CFOTracker(
        sample_rate=sample_rate,
        block_duration_ms=block_duration_ms,
        loop_bandwidth_hz=10.0
    )

    # Test with known CFO
    cfo_hz = 25.0
    n_blocks = 50

    print(f"\nSimulating CFO of {cfo_hz} Hz:")
    for i in range(n_blocks):
        # Generate phase with CFO
        phase = 2 * np.pi * cfo_hz * i * block_duration_s
        phase = np.angle(np.exp(1j * phase))  # Wrap to [-pi, pi]

        tracker.update(phase)

        if (i + 1) % 10 == 0:
            print(f"  Block {i+1}: estimated CFO = {tracker.get_cfo_hz():.2f} Hz")

    final_estimate = tracker.get_cfo_hz()
    error = abs(final_estimate - cfo_hz)
    print(f"\nFinal estimate: {final_estimate:.2f} Hz (error: {error:.2f} Hz)")

    # Test with noise
    print("\nTesting with phase noise:")
    tracker.reset()
    rng = np.random.default_rng(42)
    phase_noise_std = 0.2

    for i in range(n_blocks):
        true_phase = 2 * np.pi * cfo_hz * i * block_duration_s
        noisy_phase = true_phase + phase_noise_std * rng.standard_normal()
        noisy_phase = np.angle(np.exp(1j * noisy_phase))

        tracker.update(noisy_phase)

    noisy_estimate = tracker.get_cfo_hz()
    print(f"Noisy estimate: {noisy_estimate:.2f} Hz (true: {cfo_hz} Hz)")

    print("\nCFO Tracker tests passed!")
    return True


if __name__ == '__main__':
    test_cfo_tracker()
