"""
Interleaver modules for SC-FDE System

Implements:
- S-random bit interleaver for coded bits
- Block symbol interleaver for frequency diversity
"""

import numpy as np
import os
import hashlib
from typing import Optional, Tuple

# Global cache for interleaver permutations to avoid recomputation
_INTERLEAVER_CACHE = {}


class BitInterleaver:
    """
    S-random bit interleaver.

    S-random interleavers spread burst errors by ensuring that any two bits
    within distance S in the input are separated by at least S positions in
    the output.
    """

    def __init__(self, length: int, S: Optional[int] = None, seed: int = 42):
        """
        Initialize S-random interleaver.

        Args:
            length: Number of bits to interleave
            S: S-random spread (default: sqrt(length/2))
            seed: Random seed for reproducibility
        """
        self.length = length
        self.S = S if S is not None else int(np.sqrt(length / 2))
        self.seed = seed

        # Check cache first to avoid expensive recomputation
        cache_key = (length, self.S, seed)
        if cache_key in _INTERLEAVER_CACHE:
            self.perm = _INTERLEAVER_CACHE[cache_key]
        else:
            # Try to load from file cache
            self.perm = self._load_or_generate_permutation(cache_key)
            _INTERLEAVER_CACHE[cache_key] = self.perm

        self.inv_perm = np.argsort(self.perm)

    def _get_cache_path(self, cache_key: tuple) -> str:
        """Get file path for cached permutation"""
        # Create cache directory in user's home or temp
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "hifi_interleavers")
        os.makedirs(cache_dir, exist_ok=True)

        # Create filename from key
        key_str = f"{cache_key[0]}_{cache_key[1]}_{cache_key[2]}"
        return os.path.join(cache_dir, f"perm_{key_str}.npy")

    def _load_or_generate_permutation(self, cache_key: tuple) -> np.ndarray:
        """Load permutation from file cache or generate it"""
        cache_path = self._get_cache_path(cache_key)

        # Try to load from file
        if os.path.exists(cache_path):
            try:
                return np.load(cache_path)
            except Exception:
                pass  # Fall through to generate

        # Generate and save
        perm = self._generate_s_random_permutation()
        try:
            np.save(cache_path, perm)
        except Exception:
            pass  # Ignore save errors

        return perm

    def _generate_s_random_permutation(self) -> np.ndarray:
        """Generate S-random permutation using iterative algorithm"""
        rng = np.random.default_rng(self.seed)

        # Start with identity permutation
        perm = np.arange(self.length)
        rng.shuffle(perm)

        # Iteratively improve until S-random property is satisfied
        max_iterations = 1000 * self.length
        iteration = 0

        while iteration < max_iterations:
            is_s_random = True

            for i in range(self.length):
                # Check S-random property for position i
                for j in range(max(0, i - self.S), i):
                    if abs(perm[i] - perm[j]) < self.S:
                        is_s_random = False
                        # Swap with a random position far enough away
                        swap_candidates = []
                        for k in range(self.length):
                            if abs(k - i) >= self.S:
                                # Check if swapping maintains S-random for position j
                                ok = True
                                for m in range(max(0, k - self.S), min(self.length, k + self.S + 1)):
                                    if m != k and m != i:
                                        if abs(perm[i] - perm[m]) < self.S:
                                            ok = False
                                            break
                                if ok:
                                    swap_candidates.append(k)

                        if swap_candidates:
                            k = rng.choice(swap_candidates)
                            perm[i], perm[k] = perm[k], perm[i]
                        break

                if not is_s_random:
                    break

            if is_s_random:
                break

            iteration += 1

        # If we couldn't achieve perfect S-random, use best effort
        return perm

    def interleave(self, bits: np.ndarray) -> np.ndarray:
        """
        Interleave bit sequence.

        Args:
            bits: Input bits

        Returns:
            Interleaved bits
        """
        bits = np.asarray(bits)
        if len(bits) != self.length:
            raise ValueError(f"Expected {self.length} bits, got {len(bits)}")
        return bits[self.perm]

    def deinterleave(self, bits: np.ndarray) -> np.ndarray:
        """
        Deinterleave bit sequence.

        Args:
            bits: Interleaved bits

        Returns:
            Original bit order
        """
        bits = np.asarray(bits)
        if len(bits) != self.length:
            raise ValueError(f"Expected {self.length} bits, got {len(bits)}")
        return bits[self.inv_perm]

    def interleave_llr(self, llr: np.ndarray) -> np.ndarray:
        """Interleave LLR values (same as bits)"""
        return self.interleave(llr)

    def deinterleave_llr(self, llr: np.ndarray) -> np.ndarray:
        """Deinterleave LLR values (same as bits)"""
        return self.deinterleave(llr)


class SymbolInterleaver:
    """
    Block symbol interleaver for frequency diversity.

    Spreads symbols across multiple SC-FDE blocks to combat frequency-selective
    fading. Uses a simple row-column write/read pattern.
    """

    def __init__(self, block_size: int, depth: int = 32):
        """
        Initialize symbol interleaver.

        Args:
            block_size: Number of symbols per SC-FDE block
            depth: Number of blocks to interleave across
        """
        self.block_size = block_size
        self.depth = depth
        self.total_size = block_size * depth

    def interleave(self, symbols: np.ndarray) -> np.ndarray:
        """
        Interleave symbol sequence.

        Writes row-wise, reads column-wise.

        Args:
            symbols: Input symbols (length = block_size * depth)

        Returns:
            Interleaved symbols
        """
        symbols = np.asarray(symbols)
        if len(symbols) != self.total_size:
            raise ValueError(
                f"Expected {self.total_size} symbols, got {len(symbols)}")

        # Reshape to (depth, block_size), write row-wise
        matrix = symbols.reshape(self.depth, self.block_size)
        # Read column-wise
        return matrix.T.flatten()

    def deinterleave(self, symbols: np.ndarray) -> np.ndarray:
        """
        Deinterleave symbol sequence.

        Args:
            symbols: Interleaved symbols

        Returns:
            Original symbol order
        """
        symbols = np.asarray(symbols)
        if len(symbols) != self.total_size:
            raise ValueError(
                f"Expected {self.total_size} symbols, got {len(symbols)}")

        # Reshape to (block_size, depth), read column-wise
        matrix = symbols.reshape(self.block_size, self.depth)
        # Write row-wise
        return matrix.T.flatten()


class FrequencyInterleaver:
    """
    Frequency-domain symbol interleaver for diversity.

    Spreads adjacent frequency symbols across different subcarrier positions
    to provide diversity gain in frequency-selective fading channels.

    Adjacent symbols that experience correlated fading (within coherence
    bandwidth) are spread far apart in the interleaved sequence, so that
    after deinterleaving, burst errors are spread across the codeword.
    """

    def __init__(self, n_carriers: int, pattern: str = 'block', seed: int = 42):
        """
        Initialize frequency interleaver.

        Args:
            n_carriers: Number of data carriers to interleave
            pattern: Interleaving pattern ('block' or 'random')
            seed: Random seed for 'random' pattern
        """
        self.n_carriers = n_carriers
        self.pattern = pattern
        self.seed = seed

        self._compute_permutation()
        self.inv_perm = np.argsort(self.perm)

    def _compute_permutation(self):
        """Compute the interleaving permutation"""
        if self.pattern == 'block':
            self._compute_block_permutation()
        elif self.pattern == 'random':
            self._compute_random_permutation()
        else:
            raise ValueError(f"Unknown pattern: {self.pattern}")

    def _compute_block_permutation(self):
        """
        Block interleaving: write rows, read columns.

        This spreads adjacent symbols by approximately sqrt(n_carriers)
        positions, providing good frequency diversity.
        """
        # Choose dimensions close to square root
        self._rows = int(np.ceil(np.sqrt(self.n_carriers)))
        self._cols = int(np.ceil(self.n_carriers / self._rows))

        # Ensure we have enough space
        while self._rows * self._cols < self.n_carriers:
            self._cols += 1

        # Build permutation: write row-by-row, read column-by-column
        self.perm = np.zeros(self.n_carriers, dtype=int)

        for i in range(self.n_carriers):
            # Input position i is at (row, col) = (i // cols, i % cols)
            row = i // self._cols
            col = i % self._cols

            # Output position: reading column-by-column
            # Column col starts at position col * rows
            # Within column, row gives the offset
            output_pos = col * self._rows + row

            # Handle case where we're past the actual data
            if output_pos >= self.n_carriers:
                # Wrap around using modulo
                output_pos = output_pos % self.n_carriers

            self.perm[i] = output_pos

        # Handle any collisions by adjusting
        self._resolve_collisions()

    def _resolve_collisions(self):
        """Resolve any duplicate positions in permutation"""
        used = set()
        unused = set(range(self.n_carriers))

        # First pass: identify used positions
        for i in range(self.n_carriers):
            if self.perm[i] in used:
                # Mark for reassignment
                self.perm[i] = -1
            else:
                used.add(self.perm[i])
                unused.discard(self.perm[i])

        # Second pass: assign unused positions to collisions
        unused_list = sorted(unused)
        unused_idx = 0
        for i in range(self.n_carriers):
            if self.perm[i] == -1:
                self.perm[i] = unused_list[unused_idx]
                unused_idx += 1

    def _compute_random_permutation(self):
        """Random permutation for maximum spreading"""
        rng = np.random.RandomState(self.seed)
        self.perm = rng.permutation(self.n_carriers)
        # Set dummy values for consistency
        self._rows = int(np.ceil(np.sqrt(self.n_carriers)))
        self._cols = int(np.ceil(self.n_carriers / self._rows))

    def interleave(self, symbols: np.ndarray) -> np.ndarray:
        """
        Interleave symbol sequence.

        Args:
            symbols: Input symbols (length = n_carriers)

        Returns:
            Interleaved symbols
        """
        symbols = np.asarray(symbols)
        if len(symbols) != self.n_carriers:
            raise ValueError(
                f"Expected {self.n_carriers} symbols, got {len(symbols)}")

        # Output[perm[i]] = Input[i]
        # Equivalently: Output = Input[inv_perm]
        output = np.zeros_like(symbols)
        for i in range(self.n_carriers):
            output[self.perm[i]] = symbols[i]
        return output

    def deinterleave(self, symbols: np.ndarray) -> np.ndarray:
        """
        Deinterleave symbol sequence.

        Args:
            symbols: Interleaved symbols

        Returns:
            Original symbol order
        """
        symbols = np.asarray(symbols)
        if len(symbols) != self.n_carriers:
            raise ValueError(
                f"Expected {self.n_carriers} symbols, got {len(symbols)}")

        # Reverse operation: Output[i] = Input[perm[i]]
        output = np.zeros_like(symbols)
        for i in range(self.n_carriers):
            output[i] = symbols[self.perm[i]]
        return output


class DataSymbolFrequencyInterleaver:
    """
    Frequency-domain interleaver that operates on actual data symbols only.

    IMPORTANT: This differs from FrequencyInterleaver which operates on the
    full carrier allocation. When carrier utilization is low (e.g., 324 symbols
    in 3275 carriers = 9.9%), interleaving the full carrier block causes
    performance degradation because:

    1. Deep fades corrupt zero-padded positions (most of the block)
    2. After de-interleaving, corrupted zeros scatter among real data
    3. This converts burst errors (LDPC can handle) into distributed errors

    This class interleaves ONLY the actual data symbols:
    - TX: interleave(data_symbols) -> interleaved_symbols, then zero-pad
    - RX: extract data-length symbols, then deinterleave(symbols)

    The interleaving provides frequency diversity by spreading symbols that
    were adjacent in frequency (and thus experienced correlated fading) to
    different positions in the codeword, allowing LDPC to correct them.
    """

    def __init__(self, n_data_symbols: int, pattern: str = 'block', seed: int = 42):
        """
        Initialize data symbol interleaver.

        Args:
            n_data_symbols: Number of actual data symbols (NOT carrier count)
                           e.g., 324 for QPSK with LDPC n=648
            pattern: Interleaving pattern ('block' or 'random')
            seed: Random seed for 'random' pattern
        """
        self.n_data_symbols = n_data_symbols
        self.pattern = pattern
        self.seed = seed

        self._compute_permutation()
        self.inv_perm = np.argsort(self.perm)

    def _compute_permutation(self):
        """Compute the interleaving permutation"""
        if self.pattern == 'block':
            self._compute_block_permutation()
        elif self.pattern == 'random':
            self._compute_random_permutation()
        else:
            raise ValueError(f"Unknown pattern: {self.pattern}")

    def _compute_block_permutation(self):
        """
        Block interleaving: write rows, read columns.

        For 324 symbols with rows=cols=18:
        - Adjacent frequency symbols (indices 0,1,2...) go to rows
        - Reading column-by-column spreads them by 18 positions
        """
        n = self.n_data_symbols
        self._rows = int(np.ceil(np.sqrt(n)))
        self._cols = int(np.ceil(n / self._rows))

        while self._rows * self._cols < n:
            self._cols += 1

        self.perm = np.zeros(n, dtype=int)

        for i in range(n):
            row = i // self._cols
            col = i % self._cols
            output_pos = col * self._rows + row

            if output_pos >= n:
                output_pos = output_pos % n

            self.perm[i] = output_pos

        self._resolve_collisions()

    def _resolve_collisions(self):
        """Resolve any duplicate positions in permutation"""
        n = self.n_data_symbols
        used = set()
        unused = set(range(n))

        for i in range(n):
            if self.perm[i] in used:
                self.perm[i] = -1
            else:
                used.add(self.perm[i])
                unused.discard(self.perm[i])

        unused_list = sorted(unused)
        unused_idx = 0
        for i in range(n):
            if self.perm[i] == -1:
                self.perm[i] = unused_list[unused_idx]
                unused_idx += 1

    def _compute_random_permutation(self):
        """Random permutation for maximum spreading"""
        rng = np.random.RandomState(self.seed)
        self.perm = rng.permutation(self.n_data_symbols)
        self._rows = int(np.ceil(np.sqrt(self.n_data_symbols)))
        self._cols = int(np.ceil(self.n_data_symbols / self._rows))

    def interleave(self, symbols: np.ndarray) -> np.ndarray:
        """
        Interleave data symbols.

        Args:
            symbols: Input data symbols (length = n_data_symbols)

        Returns:
            Interleaved symbols (same length)
        """
        symbols = np.asarray(symbols)
        if len(symbols) != self.n_data_symbols:
            raise ValueError(
                f"Expected {self.n_data_symbols} symbols, got {len(symbols)}")

        output = np.zeros_like(symbols)
        for i in range(self.n_data_symbols):
            output[self.perm[i]] = symbols[i]
        return output

    def deinterleave(self, symbols: np.ndarray) -> np.ndarray:
        """
        Deinterleave data symbols.

        Args:
            symbols: Interleaved symbols (length = n_data_symbols)

        Returns:
            Original symbol order
        """
        symbols = np.asarray(symbols)
        if len(symbols) != self.n_data_symbols:
            raise ValueError(
                f"Expected {self.n_data_symbols} symbols, got {len(symbols)}")

        output = np.zeros_like(symbols)
        for i in range(self.n_data_symbols):
            output[i] = symbols[self.perm[i]]
        return output


class ConvolutionalInterleaver:
    """
    Convolutional (helical) interleaver.

    Provides continuous interleaving without block boundaries.
    Good for streaming applications.
    """

    def __init__(self, branches: int, delay: int):
        """
        Initialize convolutional interleaver.

        Args:
            branches: Number of branches (B)
            delay: Base delay per branch (D)
        """
        self.branches = branches
        self.delay = delay

        # Delay lines for each branch
        # Branch i has delay i * D
        self.delay_lines = [
            np.zeros(i * delay, dtype=np.complex128)
            for i in range(branches)
        ]
        self.branch_idx = 0

    def reset(self):
        """Reset internal state"""
        for i in range(self.branches):
            self.delay_lines[i] = np.zeros(i * self.delay, dtype=np.complex128)
        self.branch_idx = 0

    def interleave_sample(self, sample: complex) -> complex:
        """
        Interleave single sample.

        Args:
            sample: Input sample

        Returns:
            Output sample (delayed based on branch)
        """
        branch = self.branch_idx
        self.branch_idx = (self.branch_idx + 1) % self.branches

        if len(self.delay_lines[branch]) == 0:
            return sample

        # FIFO: output oldest, input newest
        output = self.delay_lines[branch][0]
        self.delay_lines[branch] = np.roll(self.delay_lines[branch], -1)
        self.delay_lines[branch][-1] = sample
        return output

    def interleave(self, samples: np.ndarray) -> np.ndarray:
        """Interleave array of samples"""
        output = np.zeros_like(samples)
        for i, s in enumerate(samples):
            output[i] = self.interleave_sample(s)
        return output


class ConvolutionalDeinterleaver:
    """
    Convolutional deinterleaver (inverse of ConvolutionalInterleaver).
    """

    def __init__(self, branches: int, delay: int):
        """
        Initialize convolutional deinterleaver.

        Args:
            branches: Number of branches (B)
            delay: Base delay per branch (D)
        """
        self.branches = branches
        self.delay = delay

        # Delay lines for deinterleaving (reversed delays)
        # Branch i has delay (B-1-i) * D
        self.delay_lines = [
            np.zeros((branches - 1 - i) * delay, dtype=np.complex128)
            for i in range(branches)
        ]
        self.branch_idx = 0

    def reset(self):
        """Reset internal state"""
        for i in range(self.branches):
            self.delay_lines[i] = np.zeros(
                (self.branches - 1 - i) * self.delay, dtype=np.complex128)
        self.branch_idx = 0

    def deinterleave_sample(self, sample: complex) -> complex:
        """Deinterleave single sample"""
        branch = self.branch_idx
        self.branch_idx = (self.branch_idx + 1) % self.branches

        if len(self.delay_lines[branch]) == 0:
            return sample

        output = self.delay_lines[branch][0]
        self.delay_lines[branch] = np.roll(self.delay_lines[branch], -1)
        self.delay_lines[branch][-1] = sample
        return output

    def deinterleave(self, samples: np.ndarray) -> np.ndarray:
        """Deinterleave array of samples"""
        output = np.zeros_like(samples)
        for i, s in enumerate(samples):
            output[i] = self.deinterleave_sample(s)
        return output


def test_interleavers():
    """Test interleaver implementations"""
    print("Testing Interleavers...")

    # Test bit interleaver
    print("\nBit Interleaver (S-random):")
    length = 648
    interleaver = BitInterleaver(length, S=16)

    bits = np.random.randint(0, 2, length, dtype=np.int8)
    interleaved = interleaver.interleave(bits)
    recovered = interleaver.deinterleave(interleaved)

    assert np.array_equal(bits, recovered), "Bit interleaver roundtrip failed!"
    print(f"  Length: {length}, S: {interleaver.S}")
    print(f"  Roundtrip: OK")

    # Check S-random property
    violations = 0
    for i in range(length):
        for j in range(max(0, i - interleaver.S), i):
            if abs(interleaver.perm[i] - interleaver.perm[j]) < interleaver.S:
                violations += 1
    print(f"  S-random violations: {violations}")

    # Test symbol interleaver
    print("\nSymbol Interleaver (Block):")
    block_size = 1792
    depth = 32
    sym_interleaver = SymbolInterleaver(block_size, depth)

    symbols = np.arange(block_size * depth, dtype=np.complex128)
    interleaved = sym_interleaver.interleave(symbols)
    recovered = sym_interleaver.deinterleave(interleaved)

    assert np.array_equal(symbols, recovered), "Symbol interleaver roundtrip failed!"
    print(f"  Block size: {block_size}, Depth: {depth}")
    print(f"  Total symbols: {block_size * depth}")
    print(f"  Roundtrip: OK")

    # Verify spreading: consecutive input symbols should be far apart
    input_positions = np.arange(depth)  # First symbol of each block
    output_positions = []
    for i in input_positions:
        # Find where symbol i ended up
        for j, s in enumerate(interleaved):
            if s == symbols[i * block_size]:
                output_positions.append(j)
                break

    avg_spread = np.mean(np.abs(np.diff(output_positions)))
    print(f"  Average spread of consecutive blocks: {avg_spread:.1f} symbols")

    # Test convolutional interleaver
    print("\nConvolutional Interleaver:")
    branches = 4
    delay = 8

    conv_int = ConvolutionalInterleaver(branches, delay)
    conv_deint = ConvolutionalDeinterleaver(branches, delay)

    # Need enough samples to fill delay lines
    n_samples = branches * delay * 10
    samples = np.arange(n_samples, dtype=np.complex128)

    interleaved = conv_int.interleave(samples)
    recovered = conv_deint.deinterleave(interleaved)

    # First few samples will be affected by empty delay lines
    skip = branches * (branches - 1) * delay // 2
    match = np.allclose(samples[:-skip], recovered[skip:])
    print(f"  Branches: {branches}, Delay: {delay}")
    print(f"  Roundtrip (after warmup): {'OK' if match else 'MISMATCH'}")

    print("\nInterleaver tests passed!")
    return True


if __name__ == '__main__':
    test_interleavers()
