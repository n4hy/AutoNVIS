"""
Bit Scrambler for SC-FDE PHY

Implements an additive LFSR-based bit scrambler that XORs coded bits with a
pseudo-random sequence. This ensures transmitted bits appear random regardless
of input data, preventing high PAPR from constant symbol patterns.

Design choices:
- Polynomial: x^7 + x^4 + 1 (7-bit LFSR, WiFi-like simplicity)
- Initialization: Fixed seed (0x7F), reset per-codeword for deterministic operation
- LLR descrambling: Flip sign when scrambler bit is 1
"""

import numpy as np
from typing import Tuple, Optional


class BitScrambler:
    """
    LFSR-based bit scrambler for randomizing coded bits.

    Uses a 7-bit LFSR with polynomial x^7 + x^4 + 1 (taps at positions 7 and 4).
    The actual period is found empirically at construction time.

    For TX: XOR coded bits with LFSR sequence
    For RX: XOR received bits with same sequence (self-inverse)
           OR flip LLR signs where sequence bit is 1
    """

    def __init__(self, polynomial: Tuple[int, int] = (7, 4),
                 seed: int = 0x7F,
                 reset_per_codeword: bool = True):
        """
        Initialize the bit scrambler.

        Args:
            polynomial: Tuple of (highest_tap, lower_tap) for LFSR polynomial.
                       Default (7, 4) gives x^7 + x^4 + 1
            seed: Initial LFSR state (default 0x7F = all ones for 7-bit)
            reset_per_codeword: If True, reset LFSR to seed at start of each
                               scramble/descramble call for deterministic operation
        """
        self.polynomial = polynomial
        self.n_bits = polynomial[0]  # Number of bits in LFSR
        self.tap_high = polynomial[0]
        self.tap_low = polynomial[1]
        self.initial_seed = seed & ((1 << self.n_bits) - 1)  # Mask to n_bits
        self.reset_per_codeword = reset_per_codeword
        self._state = self.initial_seed

        # Validate seed is non-zero (all-zero state is invalid for LFSR)
        if self.initial_seed == 0:
            raise ValueError("LFSR seed cannot be zero (degenerate state)")

        # Precompute LFSR output sequence for fast vectorized generation.
        # The polynomial may not be primitive (e.g. x^7 + x^4 + 1 factors
        # over GF(2)), so the state trajectory from the seed may have a
        # non-zero tail before entering a cycle (rho-shaped). We precompute
        # enough steps to cover any realistic use case. Since reset_per_codeword
        # restarts from the seed each time, the precomputed sequence is always
        # consumed from the beginning.
        self._precomputed_len = max((1 << self.n_bits) - 1, 4096)
        self._precomputed_seq = self._compute_sequence(self._precomputed_len)
        # Track position within the precomputed sequence
        self._pos = 0

    def _compute_sequence(self, length: int) -> np.ndarray:
        """Compute LFSR output bits from current initial_seed."""
        state = self.initial_seed
        seq = np.empty(length, dtype=np.int8)
        tap_h = self.tap_high - 1
        tap_l = self.tap_low - 1
        msb_shift = self.n_bits - 1
        for i in range(length):
            seq[i] = state & 1
            feedback = ((state >> tap_h) ^ (state >> tap_l)) & 1
            state = (state >> 1) | (feedback << msb_shift)
        return seq

    def reset(self, seed: Optional[int] = None):
        """
        Reset LFSR to initial state.

        Args:
            seed: Optional new seed. If None, uses initial_seed from __init__
        """
        if seed is not None:
            self._state = seed & ((1 << self.n_bits) - 1)
            if self._state == 0:
                raise ValueError("LFSR seed cannot be zero")
            self.initial_seed = self._state
            self._precomputed_seq = self._compute_sequence(self._precomputed_len)
        else:
            self._state = self.initial_seed
        self._pos = 0

    def _generate_sequence(self, length: int) -> np.ndarray:
        """
        Generate a sequence of LFSR output bits using the precomputed buffer.

        Args:
            length: Number of bits to generate

        Returns:
            Array of bits (0 or 1)
        """
        end = self._pos + length

        # Extend precomputed buffer if needed (rare — only if a single
        # call requests more bits than the initial buffer size)
        if end > self._precomputed_len:
            new_len = max(end, self._precomputed_len * 2)
            self._precomputed_seq = self._compute_sequence(new_len)
            self._precomputed_len = new_len

        seq = self._precomputed_seq[self._pos:end].copy()
        self._pos = end
        return seq

    def _advance(self, n_steps: int):
        """
        Advance the LFSR by n steps without generating output.

        Args:
            n_steps: Number of steps to advance
        """
        self._pos += n_steps

    def scramble(self, bits: np.ndarray, codeword_idx: int = 0) -> np.ndarray:
        """
        Scramble bits by XORing with LFSR sequence.

        Args:
            bits: Input bits (0/1 values)
            codeword_idx: Codeword index for per-codeword offset. Each codeword
                         uses a different starting point in the LFSR sequence
                         to prevent identical patterns for repeated inputs.

        Returns:
            Scrambled bits (same length)
        """
        if self.reset_per_codeword:
            self.reset()
            # Advance by codeword_idx * prime_offset to create unique sequences
            # Using a prime (53) ensures good distribution across LFSR states
            if codeword_idx > 0:
                self._advance(codeword_idx * 53)

        sequence = self._generate_sequence(len(bits))
        return (bits ^ sequence).astype(np.int8)

    def descramble(self, bits: np.ndarray) -> np.ndarray:
        """
        Descramble bits by XORing with LFSR sequence.

        This is identical to scramble() since XOR is self-inverse.

        Args:
            bits: Scrambled bits

        Returns:
            Original bits
        """
        return self.scramble(bits)

    def descramble_llr(self, llrs: np.ndarray, codeword_idx: int = 0) -> np.ndarray:
        """
        Descramble soft LLR values.

        When scrambler bit is 1, the transmitted bit was flipped, so we need
        to flip the LLR sign: llr_out = llr_in * (1 - 2*seq)
        - seq=0: multiply by +1 (no change)
        - seq=1: multiply by -1 (flip sign)

        Args:
            llrs: Log-likelihood ratios (positive = more likely 0)
            codeword_idx: Codeword index for per-codeword offset (must match TX)

        Returns:
            Descrambled LLRs
        """
        if self.reset_per_codeword:
            self.reset()
            # Advance by codeword_idx * prime_offset (must match scramble())
            if codeword_idx > 0:
                self._advance(codeword_idx * 53)

        sequence = self._generate_sequence(len(llrs))
        # (1 - 2*seq) gives +1 for seq=0, -1 for seq=1
        return llrs * (1 - 2 * sequence)

    def get_period(self) -> int:
        """
        Compute the actual LFSR period by running until state repeats.

        Note: For reducible polynomials the trajectory from the seed may
        have a non-zero tail before entering a cycle. This returns the
        cycle length (not including the tail).

        Returns:
            Period length in bits
        """
        self.reset()
        initial_state = self._state
        state = initial_state
        tap_h = self.tap_high - 1
        tap_l = self.tap_low - 1
        msb_shift = self.n_bits - 1
        max_steps = (1 << self.n_bits)
        # Walk until we see a repeated state, then measure cycle from there
        seen = {}
        for i in range(max_steps):
            if state in seen:
                return i - seen[state]
            seen[state] = i
            feedback = ((state >> tap_h) ^ (state >> tap_l)) & 1
            state = (state >> 1) | (feedback << msb_shift)
        return max_steps
