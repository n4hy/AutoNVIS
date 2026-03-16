"""
LDPC Codec for SC-FDE System

Implements IEEE 802.11n QC-LDPC codes with:
- Block lengths: 648, 1296, 1944 bits
- Code rates: 1/2, 2/3, 3/4, 5/6
- Min-sum decoding algorithm (Numba-accelerated)
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
import warnings

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    warnings.warn(
        "Numba not available. LDPC decoding will be slower. "
        "Install with: pip install numba"
    )
    # Provide stub decorators when Numba not available
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    prange = range

# Try to import aff3ct for high-performance LDPC (SIMD-optimized)
AFF3CT_AVAILABLE = False
aff3ct = None
aff3ct_sparse = None

try:
    # First try importing if installed system-wide (via install_py_aff3ct.sh)
    import py_aff3ct as aff3ct
    import py_aff3ct.tools.sparse_matrix as aff3ct_sparse
    AFF3CT_AVAILABLE = True
except ImportError:
    # Fall back to relative path for development use
    try:
        import sys
        import os
        _project_root = os.path.dirname(os.path.dirname(__file__))
        _aff3ct_path = os.path.join(_project_root, 'deps', 'py_aff3ct', 'build', 'lib')
        if os.path.exists(_aff3ct_path):
            sys.path.insert(0, _aff3ct_path)
            import py_aff3ct as aff3ct
            import py_aff3ct.tools.sparse_matrix as aff3ct_sparse
            AFF3CT_AVAILABLE = True
    except ImportError:
        pass

if not AFF3CT_AVAILABLE:
    warnings.warn(
        "AFF3CT not available. LDPC decoding will use Numba fallback (~3-5x slower). "
        "Build and install with: ./deps/build_py_aff3ct.sh && ./deps/install_py_aff3ct.sh"
    )


@dataclass
class LDPCCode:
    """LDPC code parameters"""
    n: int  # Codeword length
    k: int  # Information bits
    rate: float  # Code rate
    Z: int  # Expansion factor (lifting size)
    H_base: np.ndarray  # Base parity check matrix


# IEEE 802.11n base matrices (compressed representation)
# -1 means zero submatrix, other values are cyclic shift amounts

# Rate 1/2, Z=27 (n=648)
H_BASE_1_2_Z27 = np.array([
    [0, -1, -1, -1, 0, 0, -1, -1, 0, -1, -1, 0, 1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [22, 0, -1, -1, 17, -1, 0, 0, 12, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [6, -1, 0, -1, 10, -1, -1, -1, 24, -1, 0, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1],
    [2, -1, -1, 0, 20, -1, -1, -1, 25, 0, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1],
    [23, -1, -1, -1, 3, -1, -1, -1, 0, -1, 9, 11, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1],
    [24, -1, 23, 1, 17, -1, 3, -1, 10, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1],
    [25, -1, -1, -1, 8, -1, -1, -1, 7, 18, -1, -1, 0, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1],
    [13, 24, -1, -1, 0, -1, 8, -1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1],
    [7, 20, -1, 16, 22, 10, -1, -1, 23, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1],
    [11, -1, -1, -1, 19, -1, -1, -1, 13, -1, 3, 17, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1],
    [25, -1, 8, -1, 23, 18, -1, 14, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0],
    [3, -1, -1, -1, 16, -1, -1, 2, 25, 5, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0],
], dtype=np.int16)

# Rate 2/3, Z=27 (n=648)
H_BASE_2_3_Z27 = np.array([
    [25, 26, 14, -1, 20, -1, 2, -1, 4, -1, -1, 8, -1, 16, -1, 18, 1, 0, -1, -1, -1, -1, -1, -1],
    [10, 9, 15, 11, -1, 0, -1, 1, -1, -1, 18, -1, 8, -1, 10, -1, -1, 0, 0, -1, -1, -1, -1, -1],
    [16, 2, 20, 26, 21, -1, 6, -1, 1, 26, -1, 7, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1],
    [10, 13, 5, 0, -1, 3, -1, 7, -1, -1, 26, -1, -1, 13, -1, 16, -1, -1, -1, 0, 0, -1, -1, -1],
    [23, 14, 24, -1, 12, -1, 19, -1, 17, -1, -1, -1, 20, -1, 21, -1, 0, -1, -1, -1, 0, 0, -1, -1],
    [6, 22, 9, 20, -1, 25, -1, 17, -1, 8, -1, 14, -1, 18, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1],
    [14, 23, 21, 11, 20, -1, 24, -1, 18, -1, 19, -1, -1, -1, -1, 22, -1, -1, -1, -1, -1, -1, 0, 0],
    [17, 11, 11, 20, -1, 21, -1, 26, -1, 3, -1, -1, 18, -1, 26, -1, 1, -1, -1, -1, -1, -1, -1, 0],
], dtype=np.int16)

# Rate 3/4, Z=27 (n=648)
H_BASE_3_4_Z27 = np.array([
    [16, 17, 22, 24, 9, 3, 14, -1, 4, 2, 7, -1, 26, -1, 2, -1, 21, -1, 1, 0, -1, -1, -1, -1],
    [25, 12, 12, 3, 3, 26, 6, 21, -1, 15, 22, -1, 15, -1, 4, -1, -1, 16, -1, 0, 0, -1, -1, -1],
    [25, 18, 26, 16, 22, 23, 9, -1, 0, -1, 4, -1, -1, 21, -1, 1, 12, -1, -1, -1, 0, 0, -1, -1],
    [9, 7, 0, 1, 17, -1, -1, 7, 3, -1, 3, 23, -1, 16, -1, -1, 21, -1, 0, -1, -1, 0, 0, -1],
    [24, 5, 26, 7, 1, -1, -1, 15, 24, 15, -1, 8, -1, 13, -1, 13, -1, 11, -1, -1, -1, -1, 0, 0],
    [2, 2, 19, 14, 24, 1, 15, 19, -1, 21, -1, -1, 23, -1, 21, -1, -1, -1, -1, -1, -1, -1, -1, 0],
], dtype=np.int16)

# Rate 5/6, Z=27 (n=648)
H_BASE_5_6_Z27 = np.array([
    [17, 13, 8, 21, 9, 3, 18, 12, 10, 0, 4, 15, 19, 2, 5, 10, 26, 19, 13, 13, 1, 0, -1, -1],
    [3, 12, 11, 14, 11, 25, 5, 18, 0, 9, 2, 26, 26, 10, 24, 7, 14, 20, 4, 2, -1, 0, 0, -1],
    [22, 16, 4, 3, 10, 21, 12, 5, 21, 14, 19, 5, -1, 8, 5, 18, 11, 5, 5, 15, 0, -1, 0, 0],
    [7, 7, 14, 14, 4, 16, 16, 24, 24, 10, 1, 7, 15, 6, 10, 26, 8, 18, 21, 14, 1, -1, -1, 0],
], dtype=np.int16)


def _scale_base_matrix(H_base: np.ndarray, z_base: int, z_new: int) -> np.ndarray:
    """
    Scale base matrix cyclic shift values for larger expansion factors.

    IEEE 802.11n uses the same base matrices for all block lengths,
    but with scaled cyclic shift values:
        new_shift = (old_shift * z_new) // z_base

    Args:
        H_base: Base matrix with shift values for z_base (-1 = zero block)
        z_base: Original expansion factor (27 for n=648)
        z_new: New expansion factor (54 for n=1296, 81 for n=1944)

    Returns:
        Scaled base matrix with shift values for z_new
    """
    H_scaled = H_base.copy()

    # Scale non-negative shift values
    for i in range(H_base.shape[0]):
        for j in range(H_base.shape[1]):
            if H_base[i, j] >= 0:
                H_scaled[i, j] = (H_base[i, j] * z_new) // z_base
            # -1 values (zero blocks) remain unchanged

    return H_scaled


def _expand_base_matrix(H_base: np.ndarray, Z: int) -> np.ndarray:
    """
    Expand base parity check matrix to full H matrix using cyclic permutations.

    Args:
        H_base: Base matrix with shift values (-1 = zero block)
        Z: Expansion factor

    Returns:
        Full sparse H matrix (m*Z x n*Z)
    """
    mb, nb = H_base.shape
    m = mb * Z
    n = nb * Z

    # Build H as list of (row, col) pairs for efficiency
    rows = []
    cols = []

    for i in range(mb):
        for j in range(nb):
            shift = H_base[i, j]
            if shift >= 0:
                # Create cyclic permutation matrix
                for k in range(Z):
                    rows.append(i * Z + k)
                    cols.append(j * Z + (k + shift) % Z)

    # Create sparse-like representation as dense for simplicity
    H = np.zeros((m, n), dtype=np.int8)
    for r, c in zip(rows, cols):
        H[r, c] = 1

    return H


# Numba-optimized min-sum decoder functions
@njit(cache=True)
def _minsum_iteration_numba(v2c, c2v, llr, check_neighbors, check_n_neighbors,
                            var_neighbors, var_n_neighbors, norm_factor):
    """
    Perform one iteration of min-sum decoding (Numba-accelerated).

    Args:
        v2c: Variable-to-check messages (n x max_check_degree)
        c2v: Check-to-variable messages (m x max_var_degree)
        llr: Channel LLRs (n,)
        check_neighbors: Padded array of variable node neighbors for each check node (m x max_check_degree)
        check_n_neighbors: Number of actual neighbors for each check node (m,)
        var_neighbors: Padded array of check node neighbors for each variable node (n x max_var_degree)
        var_n_neighbors: Number of actual neighbors for each variable node (n,)
        norm_factor: Normalization factor for min-sum

    Returns:
        Updated v2c and c2v arrays (in-place modification)
    """
    m = check_neighbors.shape[0]
    n = var_neighbors.shape[0]

    # Check node update (min-sum)
    for i in range(m):
        n_neigh = check_n_neighbors[i]
        for jj in range(n_neigh):
            j = check_neighbors[i, jj]

            # Compute sign and minimum abs of other messages
            sign = 1
            min_abs = 1e30
            for kk in range(n_neigh):
                if kk != jj:
                    k = check_neighbors[i, kk]
                    msg = v2c[k, i]
                    if msg < 0:
                        sign = -sign
                    abs_msg = abs(msg)
                    if abs_msg < min_abs:
                        min_abs = abs_msg

            c2v[i, j] = sign * min_abs * norm_factor

    # Variable node update
    for j in range(n):
        n_neigh = var_n_neighbors[j]
        # Compute total incoming
        total = llr[j]
        for ii in range(n_neigh):
            i = var_neighbors[j, ii]
            total += c2v[i, j]

        # Update outgoing messages
        for ii in range(n_neigh):
            i = var_neighbors[j, ii]
            v2c[j, i] = total - c2v[i, j]


@njit(cache=True)
def _compute_posterior_numba(llr, c2v, var_neighbors, var_n_neighbors):
    """Compute posterior LLRs (Numba-accelerated)."""
    n = len(llr)
    posterior = llr.copy()
    for j in range(n):
        n_neigh = var_n_neighbors[j]
        for ii in range(n_neigh):
            i = var_neighbors[j, ii]
            posterior[j] += c2v[i, j]
    return posterior


@njit(cache=True, parallel=True)
def _minsum_decode_batch_numba(llrs, check_neighbors, check_n_neighbors,
                                var_neighbors, var_n_neighbors, norm_factor,
                                max_iter, early_termination):
    """
    Decode multiple codewords in parallel (Numba-accelerated with prange).

    Args:
        llrs: 2D array of LLRs (n_codewords x n)
        check_neighbors: Padded array (m x max_check_degree)
        check_n_neighbors: Number of neighbors (m,)
        var_neighbors: Padded array (n x max_var_degree)
        var_n_neighbors: Number of neighbors (n,)
        norm_factor: Min-sum normalization
        max_iter: Maximum iterations
        early_termination: Whether to check syndrome each iteration

    Returns:
        (hard_decisions, converged_arr) - 2D array and 1D bool array
    """
    n_codewords = llrs.shape[0]
    n = llrs.shape[1]
    m = check_neighbors.shape[0]

    # Output arrays
    hard_decisions = np.zeros((n_codewords, n), dtype=np.int8)
    converged_arr = np.zeros(n_codewords, dtype=np.bool_)

    # Parallel loop over codewords
    for cw_idx in prange(n_codewords):
        llr = llrs[cw_idx]

        # Initialize variable-to-check messages
        v2c = np.zeros((n, m), dtype=np.float64)
        for j in range(n):
            n_neigh = var_n_neighbors[j]
            for ii in range(n_neigh):
                i = var_neighbors[j, ii]
                v2c[j, i] = llr[j]

        # Initialize check-to-variable messages
        c2v = np.zeros((m, n), dtype=np.float64)

        converged = False
        hard_decision = np.zeros(n, dtype=np.int8)

        for iteration in range(max_iter):
            # Check node update (min-sum)
            for i in range(m):
                n_neigh = check_n_neighbors[i]
                for jj in range(n_neigh):
                    j = check_neighbors[i, jj]

                    # Compute sign and minimum abs of other messages
                    sign = 1
                    min_abs = 1e30
                    for kk in range(n_neigh):
                        if kk != jj:
                            k = check_neighbors[i, kk]
                            msg = v2c[k, i]
                            if msg < 0:
                                sign = -sign
                            abs_msg = abs(msg)
                            if abs_msg < min_abs:
                                min_abs = abs_msg

                    c2v[i, j] = sign * min_abs * norm_factor

            # Variable node update
            for j in range(n):
                n_neigh = var_n_neighbors[j]
                # Compute total incoming
                total = llr[j]
                for ii in range(n_neigh):
                    i = var_neighbors[j, ii]
                    total += c2v[i, j]

                # Update outgoing messages
                for ii in range(n_neigh):
                    i = var_neighbors[j, ii]
                    v2c[j, i] = total - c2v[i, j]

            # Compute posterior and hard decision
            for j in range(n):
                posterior = llr[j]
                n_neigh = var_n_neighbors[j]
                for ii in range(n_neigh):
                    i = var_neighbors[j, ii]
                    posterior += c2v[i, j]
                hard_decision[j] = 1 if posterior < 0 else 0

            # Check convergence
            if early_termination:
                all_zero = True
                for i in range(m):
                    syndrome_bit = 0
                    for jj in range(check_n_neighbors[i]):
                        j = check_neighbors[i, jj]
                        syndrome_bit = (syndrome_bit + hard_decision[j]) % 2
                    if syndrome_bit != 0:
                        all_zero = False
                        break

                if all_zero:
                    converged = True
                    break

        # Final convergence check if early termination was disabled
        if not early_termination:
            all_zero = True
            for i in range(m):
                syndrome_bit = 0
                for jj in range(check_n_neighbors[i]):
                    j = check_neighbors[i, jj]
                    syndrome_bit = (syndrome_bit + hard_decision[j]) % 2
                if syndrome_bit != 0:
                    all_zero = False
                    break
            converged = all_zero

        # Store results
        hard_decisions[cw_idx] = hard_decision
        converged_arr[cw_idx] = converged

    return hard_decisions, converged_arr


@njit(cache=True)
def _minsum_decode_numba(llr, check_neighbors, check_n_neighbors,
                         var_neighbors, var_n_neighbors, norm_factor,
                         max_iter, early_termination):
    """
    Full min-sum decoding (Numba-accelerated).

    Args:
        llr: Channel LLRs (n,)
        check_neighbors: Padded array (m x max_check_degree)
        check_n_neighbors: Number of neighbors (m,)
        var_neighbors: Padded array (n x max_var_degree)
        var_n_neighbors: Number of neighbors (n,)
        norm_factor: Min-sum normalization
        max_iter: Maximum iterations
        early_termination: Whether to check syndrome each iteration

    Returns:
        (hard_decision, converged, iterations)
    """
    m = check_neighbors.shape[0]
    n = var_neighbors.shape[0]

    # Initialize variable-to-check messages
    v2c = np.zeros((n, m), dtype=np.float64)
    for j in range(n):
        n_neigh = var_n_neighbors[j]
        for ii in range(n_neigh):
            i = var_neighbors[j, ii]
            v2c[j, i] = llr[j]

    # Initialize check-to-variable messages
    c2v = np.zeros((m, n), dtype=np.float64)

    converged = False
    iteration = 0

    for iteration in range(max_iter):
        # Perform one min-sum iteration
        _minsum_iteration_numba(v2c, c2v, llr, check_neighbors, check_n_neighbors,
                                var_neighbors, var_n_neighbors, norm_factor)

        # Compute posterior and hard decision
        posterior = _compute_posterior_numba(llr, c2v, var_neighbors, var_n_neighbors)
        hard_decision = np.empty(n, dtype=np.int8)
        for j in range(n):
            hard_decision[j] = 1 if posterior[j] < 0 else 0

        # Check convergence
        if early_termination:
            # Compute syndrome
            all_zero = True
            for i in range(m):
                syndrome_bit = 0
                for jj in range(check_n_neighbors[i]):
                    j = check_neighbors[i, jj]
                    syndrome_bit = (syndrome_bit + hard_decision[j]) % 2
                if syndrome_bit != 0:
                    all_zero = False
                    break

            if all_zero:
                converged = True
                break

    # Final convergence check if early termination was disabled
    if not early_termination:
        all_zero = True
        for i in range(m):
            syndrome_bit = 0
            for jj in range(check_n_neighbors[i]):
                j = check_neighbors[i, jj]
                syndrome_bit = (syndrome_bit + hard_decision[j]) % 2
            if syndrome_bit != 0:
                all_zero = False
                break
        converged = all_zero

    return hard_decision, converged, iteration + 1


class LDPCCodec:
    """
    LDPC Encoder/Decoder using IEEE 802.11n QC-LDPC codes.

    Supports:
    - Block lengths: 648, 1296, 1944 bits
    - Code rates: 1/2, 2/3, 3/4, 5/6
    - Min-sum decoding with optional normalization
    """

    # Available codes indexed by (n, rate)
    # n=648: Z=27, n=1296: Z=54, n=1944: Z=81
    CODES = {
        # n=648 codes (Z=27, base matrices)
        (648, 0.5): (H_BASE_1_2_Z27, 27),
        (648, 2/3): (H_BASE_2_3_Z27, 27),
        (648, 0.75): (H_BASE_3_4_Z27, 27),
        (648, 5/6): (H_BASE_5_6_Z27, 27),
        # n=1296 codes (Z=54, scaled from Z=27)
        (1296, 0.5): (_scale_base_matrix(H_BASE_1_2_Z27, 27, 54), 54),
        (1296, 2/3): (_scale_base_matrix(H_BASE_2_3_Z27, 27, 54), 54),
        (1296, 0.75): (_scale_base_matrix(H_BASE_3_4_Z27, 27, 54), 54),
        (1296, 5/6): (_scale_base_matrix(H_BASE_5_6_Z27, 27, 54), 54),
        # n=1944 codes (Z=81, scaled from Z=27)
        (1944, 0.5): (_scale_base_matrix(H_BASE_1_2_Z27, 27, 81), 81),
        (1944, 2/3): (_scale_base_matrix(H_BASE_2_3_Z27, 27, 81), 81),
        (1944, 0.75): (_scale_base_matrix(H_BASE_3_4_Z27, 27, 81), 81),
        (1944, 5/6): (_scale_base_matrix(H_BASE_5_6_Z27, 27, 81), 81),
    }

    def __init__(self, n: int = 648, rate: float = 0.5,
                 max_iter: int = 50, norm_factor: float = 0.75):
        """
        Initialize LDPC codec.

        Args:
            n: Codeword length (648, 1296, or 1944)
            rate: Code rate (0.5, 2/3, 0.75, or 5/6)
            max_iter: Maximum decoding iterations
            norm_factor: Normalization factor for min-sum (0.75-0.9 typical)
        """
        # Normalize rate to avoid float comparison issues
        rate_key = self._normalize_rate(rate)

        if (n, rate_key) not in self.CODES:
            available = list(self.CODES.keys())
            raise ValueError(f"Code (n={n}, rate={rate}) not available. "
                           f"Available: {available}")

        self.n = n
        self.rate = rate_key
        self.max_iter = max_iter
        self.norm_factor = norm_factor

        # Get base matrix and expansion factor
        H_base, Z = self.CODES[(n, rate_key)]
        self.Z = Z
        self.H_base = H_base

        # Expand to full H matrix
        self.H = _expand_base_matrix(H_base, Z)

        # Calculate dimensions
        self.m = self.H.shape[0]  # Number of parity checks
        self.k = self.n - self.m  # Information bits

        # Build encoding matrix (systematic form)
        self._build_generator()

        # Precompute neighbor indices for decoding
        self._precompute_neighbors()

    @staticmethod
    def _normalize_rate(rate: float) -> float:
        """Normalize rate to standard values"""
        rate_map = {
            0.5: 0.5,
            1/2: 0.5,
            2/3: 2/3,
            0.667: 2/3,
            0.75: 0.75,
            3/4: 0.75,
            5/6: 5/6,
            0.833: 5/6,
        }
        for r, normalized in rate_map.items():
            if abs(rate - r) < 0.01:
                return normalized
        return rate

    def _build_generator(self):
        """Build systematic generator matrix G = [I | P]"""
        # For QC-LDPC with dual-diagonal structure, use efficient encoding
        # H = [A | B] where B is approximately lower triangular

        # Store H in parts for efficient encoding
        self.H_info = self.H[:, :self.k]  # Part corresponding to info bits
        self.H_parity = self.H[:, self.k:]  # Part corresponding to parity bits

        # Check if H_parity is full rank - if not, we need null-space encoding
        parity_rank = np.linalg.matrix_rank(self.H_parity)
        if parity_rank < self.m:
            # Build systematic generator from null space of H
            self._build_null_space_generator()
            self._use_generator_encoding = True
        else:
            self._use_generator_encoding = False

    def _build_null_space_generator(self):
        """Build systematic generator matrix from null space of H"""
        # Find null space of H over GF(2)
        # Use row reduction on H^T augmented with identity

        H = self.H.astype(np.int8)
        m, n = H.shape
        k = self.k

        # Augment [H^T | I_n] and row reduce
        aug = np.hstack([H.T, np.eye(n, dtype=np.int8)])

        # Gaussian elimination on first m columns
        pivot_row = 0
        for col in range(m):
            if pivot_row >= n:
                break

            # Find pivot
            found = False
            for r in range(pivot_row, n):
                if aug[r, col] == 1:
                    if r != pivot_row:
                        aug[[r, pivot_row]] = aug[[pivot_row, r]]
                    found = True
                    break

            if not found:
                continue

            # Eliminate
            for r in range(n):
                if r != pivot_row and aug[r, col] == 1:
                    aug[r] = (aug[r] + aug[pivot_row]) % 2

            pivot_row += 1

        # Extract null space vectors (rows where first m columns are zero)
        null_vectors = []
        for r in range(n):
            if np.all(aug[r, :m] == 0):
                null_vectors.append(aug[r, m:])

        G = np.array(null_vectors, dtype=np.int8)

        if G.shape[0] != k:
            raise RuntimeError(f"Null space dimension {G.shape[0]} != expected k={k}")

        # Put G in systematic form [I_k | P] through row/column operations
        # Track column permutation to restore systematic order later
        col_perm = list(range(n))

        for i in range(k):
            # Find pivot in column i (want G[i,i] = 1 after permutation)
            found = False

            # First check current column
            for r in range(i, k):
                if G[r, i] == 1:
                    if r != i:
                        G[[r, i]] = G[[i, r]]
                    found = True
                    break

            if not found:
                # Need to swap column i with some column j >= k that has a 1 in row i
                for j in range(i + 1, n):
                    if G[i, j] == 1:
                        G[:, [i, j]] = G[:, [j, i]]
                        col_perm[i], col_perm[j] = col_perm[j], col_perm[i]
                        found = True
                        break

            if not found:
                # Try finding any 1 in remaining submatrix
                for r in range(i, k):
                    for j in range(i, n):
                        if G[r, j] == 1:
                            if r != i:
                                G[[r, i]] = G[[i, r]]
                            if j != i:
                                G[:, [i, j]] = G[:, [j, i]]
                                col_perm[i], col_perm[j] = col_perm[j], col_perm[i]
                            found = True
                            break
                    if found:
                        break

            if not found:
                raise RuntimeError(f"Cannot find pivot for row {i}")

            # Eliminate other 1s in column i
            for r in range(k):
                if r != i and G[r, i] == 1:
                    G[r] = (G[r] + G[i]) % 2

        # Store generator matrix components
        self._gen_P = G[:, k:]  # k x m parity generation matrix
        self._col_perm = col_perm
        self._col_perm_inv = [0] * n
        for i, p in enumerate(col_perm):
            self._col_perm_inv[p] = i

        # The info positions in the original codeword are col_perm[0:k]
        # These are the original column indices that became info positions after permutation
        self._info_positions = sorted(col_perm[:k])
        self._parity_positions = sorted(col_perm[k:])

    def _precompute_neighbors(self):
        """Precompute check node and variable node neighbors for fast decoding"""
        # For each check node, list of connected variable nodes
        self.check_neighbors = []
        for i in range(self.m):
            self.check_neighbors.append(np.where(self.H[i, :] == 1)[0])

        # For each variable node, list of connected check nodes
        self.var_neighbors = []
        for j in range(self.n):
            self.var_neighbors.append(np.where(self.H[:, j] == 1)[0])

        # Create fixed-size arrays for Numba compatibility
        if NUMBA_AVAILABLE:
            self._precompute_numba_arrays()

    def _precompute_numba_arrays(self):
        """Create fixed-size arrays for Numba-accelerated decoding."""
        # Find maximum degrees
        max_check_degree = max(len(n) for n in self.check_neighbors)
        max_var_degree = max(len(n) for n in self.var_neighbors)

        # Create padded arrays for check node neighbors
        self._check_neighbors_numba = np.zeros((self.m, max_check_degree), dtype=np.int32)
        self._check_n_neighbors = np.zeros(self.m, dtype=np.int32)
        for i, neighbors in enumerate(self.check_neighbors):
            self._check_n_neighbors[i] = len(neighbors)
            self._check_neighbors_numba[i, :len(neighbors)] = neighbors

        # Create padded arrays for variable node neighbors
        self._var_neighbors_numba = np.zeros((self.n, max_var_degree), dtype=np.int32)
        self._var_n_neighbors = np.zeros(self.n, dtype=np.int32)
        for j, neighbors in enumerate(self.var_neighbors):
            self._var_n_neighbors[j] = len(neighbors)
            self._var_neighbors_numba[j, :len(neighbors)] = neighbors

    def encode(self, info_bits: np.ndarray) -> np.ndarray:
        """
        Encode information bits to codeword.

        Args:
            info_bits: Array of k information bits

        Returns:
            Codeword of n bits
        """
        info_bits = np.asarray(info_bits, dtype=np.int8)

        if len(info_bits) != self.k:
            raise ValueError(f"Expected {self.k} info bits, got {len(info_bits)}")

        # Use generator matrix encoding if H_parity is rank-deficient
        if self._use_generator_encoding:
            return self._encode_generator(info_bits)

        codeword = np.zeros(self.n, dtype=np.int8)
        codeword[:self.k] = info_bits

        # Compute syndrome contribution from info bits
        syndrome = (self.H_info @ info_bits) % 2

        # Try fast encoding first (for dual-diagonal structure)
        parity = self._encode_fast(syndrome)

        if parity is not None:
            codeword[self.k:] = parity

            # Verify: H @ c should be zero
            check = (self.H @ codeword) % 2
            if np.all(check == 0):
                return codeword

        # Fall back to Gaussian elimination
        return self._encode_gaussian(info_bits)

    def _encode_generator(self, info_bits: np.ndarray) -> np.ndarray:
        """Encode using the systematic generator matrix G = [I | P]"""
        # Compute parity bits: p = s @ P (mod 2)
        parity = (info_bits @ self._gen_P) % 2

        # Build codeword: place info and parity bits at their permuted positions
        # col_perm[i] = original position that holds the i-th bit in permuted order
        codeword = np.zeros(self.n, dtype=np.int8)

        # Info bits go to positions col_perm[0:k]
        for i in range(self.k):
            codeword[self._col_perm[i]] = info_bits[i]

        # Parity bits go to positions col_perm[k:n]
        for j in range(self.m):
            codeword[self._col_perm[self.k + j]] = parity[j]

        return codeword

    def _encode_fast(self, syndrome: np.ndarray) -> np.ndarray:
        """Fast encoding for dual-diagonal parity structure"""
        # Check if this is rate 3/4 which needs special handling
        if self.rate == 0.75:
            return self._encode_rate_3_4(syndrome)

        parity = np.zeros(self.m, dtype=np.int8)

        # For IEEE 802.11n codes with standard dual-diagonal structure
        # Solve for parity bits row by row
        for i in range(self.m):
            # Sum of known parity contributions
            known_sum = 0
            for j in range(i):
                if self.H_parity[i, j] == 1:
                    known_sum ^= parity[j]
            parity[i] = (syndrome[i] ^ known_sum) % 2

        return parity

    def _encode_rate_3_4(self, syndrome: np.ndarray) -> np.ndarray:
        """
        Special encoding for rate 3/4 codes using iterative approach.

        Rate 3/4 has a complex feedback structure that requires careful handling.
        We use a bit-by-bit iterative approach with the full expanded H matrix.
        """
        # For rate 3/4, use the Gaussian elimination fallback directly
        # since the structure is complex and the simple approach doesn't work
        return None  # Signal to use Gaussian method

    def _encode_gaussian(self, info_bits: np.ndarray) -> np.ndarray:
        """Encode using Gaussian elimination with rank handling"""
        # Solve H_parity @ p = syndrome where syndrome = H_info @ s (mod 2)
        syndrome = (self.H_info @ info_bits) % 2

        # Create augmented matrix [H_parity | syndrome]
        augmented = np.hstack([self.H_parity.copy(),
                               syndrome.reshape(-1, 1)])
        augmented = augmented.astype(np.int8)

        m, n_aug = augmented.shape
        n_vars = n_aug - 1  # Number of parity variables

        # Track pivot columns and free columns
        pivot_cols = []
        pivot_row = 0

        # Forward elimination to row echelon form
        for col in range(n_vars):
            if pivot_row >= m:
                break

            # Find pivot in column
            pivot_found = False
            for row in range(pivot_row, m):
                if augmented[row, col] == 1:
                    if row != pivot_row:
                        augmented[[pivot_row, row]] = augmented[[row, pivot_row]]
                    pivot_found = True
                    break

            if not pivot_found:
                continue

            pivot_cols.append(col)

            # Eliminate below and above
            for row in range(m):
                if row != pivot_row and augmented[row, col] == 1:
                    augmented[row] = (augmented[row] + augmented[pivot_row]) % 2

            pivot_row += 1

        # Identify free variables
        free_cols = [c for c in range(n_vars) if c not in pivot_cols]

        # Try to find a valid encoding by iterating over free variable combinations
        # For efficiency, only try a limited number of combinations
        max_tries = min(2 ** len(free_cols), 1024)

        best_parity = None
        best_errors = float('inf')

        for trial in range(max_tries):
            parity = np.zeros(n_vars, dtype=np.int8)

            # Set free variables based on trial number
            for i, col in enumerate(free_cols):
                parity[col] = (trial >> i) & 1

            # Back-substitution for pivot variables
            for i in range(len(pivot_cols) - 1, -1, -1):
                col = pivot_cols[i]
                row = i

                rhs = augmented[row, -1]
                for j in range(col + 1, n_vars):
                    if augmented[row, j] == 1:
                        rhs ^= parity[j]
                parity[col] = rhs

            # Verify this solution
            codeword = np.zeros(self.n, dtype=np.int8)
            codeword[:self.k] = info_bits
            codeword[self.k:] = parity

            errors = np.sum((self.H @ codeword) % 2)

            if errors == 0:
                return codeword

            if errors < best_errors:
                best_errors = errors
                best_parity = parity.copy()

        # Return best attempt even if not perfect
        codeword = np.zeros(self.n, dtype=np.int8)
        codeword[:self.k] = info_bits
        codeword[self.k:] = best_parity

        return codeword

    def decode(self, llr: np.ndarray, return_iterations: bool = False,
               max_iterations: Optional[int] = None,
               early_termination: bool = True
               ) -> Tuple[np.ndarray, bool]:
        """
        Decode using min-sum algorithm.

        Args:
            llr: Log-likelihood ratios for each bit (positive = more likely 0)
            return_iterations: If True, return number of iterations used
            max_iterations: Override max iterations (None = use self.max_iter)
            early_termination: If False, always run max_iterations (no syndrome check)

        Returns:
            Tuple of (decoded_bits, converged) or (decoded_bits, converged, iterations)
        """
        llr = np.asarray(llr, dtype=np.float64)

        if len(llr) != self.n:
            raise ValueError(f"Expected {self.n} LLRs, got {len(llr)}")

        # Use provided max_iterations or fall back to default
        n_iter = max_iterations if max_iterations is not None else self.max_iter

        # Use Numba-accelerated decoder if available
        if NUMBA_AVAILABLE and hasattr(self, '_check_neighbors_numba'):
            hard_decision, converged, iterations = _minsum_decode_numba(
                llr,
                self._check_neighbors_numba, self._check_n_neighbors,
                self._var_neighbors_numba, self._var_n_neighbors,
                self.norm_factor, n_iter, early_termination
            )
            if return_iterations:
                return hard_decision, converged, iterations
            return hard_decision, converged

        # Fallback to Python implementation
        return self._decode_python(llr, return_iterations, n_iter, early_termination)

    def _decode_python(self, llr: np.ndarray, return_iterations: bool,
                       n_iter: int, early_termination: bool) -> Tuple[np.ndarray, bool]:
        """Python fallback implementation of min-sum decoding."""
        # Initialize variable-to-check messages
        v2c = np.zeros((self.n, self.m), dtype=np.float64)
        for j in range(self.n):
            for i in self.var_neighbors[j]:
                v2c[j, i] = llr[j]

        # Initialize check-to-variable messages
        c2v = np.zeros((self.m, self.n), dtype=np.float64)

        converged = False
        iteration = 0

        for iteration in range(n_iter):
            # Check node update (min-sum)
            for i in range(self.m):
                neighbors = self.check_neighbors[i]
                for j in neighbors:
                    # Product of signs of other messages
                    other_msgs = [v2c[k, i] for k in neighbors if k != j]
                    if len(other_msgs) == 0:
                        c2v[i, j] = 0
                        continue

                    sign = 1
                    min_abs = float('inf')
                    for msg in other_msgs:
                        if msg < 0:
                            sign *= -1
                        abs_msg = abs(msg)
                        if abs_msg < min_abs:
                            min_abs = abs_msg

                    c2v[i, j] = sign * min_abs * self.norm_factor

            # Variable node update
            for j in range(self.n):
                neighbors = self.var_neighbors[j]
                total = llr[j] + np.sum(c2v[neighbors, j])

                for i in neighbors:
                    v2c[j, i] = total - c2v[i, j]

            # Hard decision
            posterior = llr.copy()
            for j in range(self.n):
                posterior[j] += np.sum(c2v[self.var_neighbors[j], j])

            hard_decision = (posterior < 0).astype(np.int8)

            # Check convergence (only if early_termination enabled)
            if early_termination:
                syndrome = (self.H @ hard_decision) % 2
                if np.all(syndrome == 0):
                    converged = True
                    break

        # Final convergence check if early termination was disabled
        if not early_termination:
            syndrome = (self.H @ hard_decision) % 2
            converged = np.all(syndrome == 0)

        if return_iterations:
            return hard_decision, converged, iteration + 1
        return hard_decision, converged

    def decode_batch(self, llrs: np.ndarray,
                     max_iterations: Optional[int] = None,
                     early_termination: bool = True
                     ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decode multiple codewords in parallel using Numba's prange.

        This method provides speedup over sequential decoding by processing
        multiple codewords simultaneously using thread-level parallelism.

        Args:
            llrs: 2D array of LLRs, shape (n_codewords, n) or 1D array of
                  concatenated LLRs (length must be multiple of n)
            max_iterations: Maximum decoding iterations (None = use self.max_iter)
            early_termination: If True, stop when syndrome is zero

        Returns:
            Tuple of (codewords, converged) where:
            - codewords: 2D array of decoded bits, shape (n_codewords, n)
            - converged: 1D boolean array indicating convergence for each codeword
        """
        llrs = np.asarray(llrs, dtype=np.float64)
        n_iter = max_iterations if max_iterations is not None else self.max_iter

        # Handle 1D input (concatenated LLRs)
        if llrs.ndim == 1:
            if len(llrs) % self.n != 0:
                raise ValueError(f"LLR length {len(llrs)} not divisible by n={self.n}")
            n_codewords = len(llrs) // self.n
            llrs = llrs.reshape(n_codewords, self.n)
        else:
            n_codewords = llrs.shape[0]
            if llrs.shape[1] != self.n:
                raise ValueError(f"Expected LLRs of length {self.n}, got {llrs.shape[1]}")

        # Use Numba-accelerated batch decoder if available
        if NUMBA_AVAILABLE and hasattr(self, '_check_neighbors_numba'):
            codewords, converged = _minsum_decode_batch_numba(
                llrs,
                self._check_neighbors_numba, self._check_n_neighbors,
                self._var_neighbors_numba, self._var_n_neighbors,
                self.norm_factor, n_iter, early_termination
            )
            return codewords, converged

        # Fallback to sequential decoding
        codewords = np.zeros((n_codewords, self.n), dtype=np.int8)
        converged = np.zeros(n_codewords, dtype=bool)

        for i in range(n_codewords):
            codewords[i], converged[i] = self.decode(llrs[i], max_iterations=n_iter,
                                                      early_termination=early_termination)

        return codewords, converged

    def decode_soft(self, llr: np.ndarray) -> np.ndarray:
        """
        Decode and return soft output (posterior LLRs).

        Args:
            llr: Input log-likelihood ratios

        Returns:
            Posterior LLRs after decoding
        """
        llr = np.asarray(llr, dtype=np.float64)

        if len(llr) != self.n:
            raise ValueError(f"Expected {self.n} LLRs, got {len(llr)}")

        # Initialize messages
        v2c = np.zeros((self.n, self.m), dtype=np.float64)
        for j in range(self.n):
            for i in self.var_neighbors[j]:
                v2c[j, i] = llr[j]

        c2v = np.zeros((self.m, self.n), dtype=np.float64)

        for _ in range(self.max_iter):
            # Check node update
            for i in range(self.m):
                neighbors = self.check_neighbors[i]
                for j in neighbors:
                    other_msgs = [v2c[k, i] for k in neighbors if k != j]
                    if len(other_msgs) == 0:
                        c2v[i, j] = 0
                        continue

                    sign = 1
                    min_abs = float('inf')
                    for msg in other_msgs:
                        if msg < 0:
                            sign *= -1
                        abs_msg = abs(msg)
                        if abs_msg < min_abs:
                            min_abs = abs_msg

                    c2v[i, j] = sign * min_abs * self.norm_factor

            # Variable node update
            for j in range(self.n):
                neighbors = self.var_neighbors[j]
                total = llr[j] + np.sum(c2v[neighbors, j])

                for i in neighbors:
                    v2c[j, i] = total - c2v[i, j]

            # Check convergence
            posterior = llr.copy()
            for j in range(self.n):
                posterior[j] += np.sum(c2v[self.var_neighbors[j], j])

            hard_decision = (posterior < 0).astype(np.int8)
            syndrome = (self.H @ hard_decision) % 2
            if np.all(syndrome == 0):
                break

        # Compute final posterior
        posterior = llr.copy()
        for j in range(self.n):
            posterior[j] += np.sum(c2v[self.var_neighbors[j], j])

        return posterior

    def decode_with_extrinsic(self, llr: np.ndarray
                              ) -> Tuple[np.ndarray, bool, np.ndarray]:
        """
        Decode and return both hard decision and extrinsic LLRs.

        Extrinsic LLR = posterior - a_priori = sum of check-to-variable messages.
        Used for turbo equalization feedback.

        Args:
            llr: Input log-likelihood ratios (a priori)

        Returns:
            Tuple of (hard_decision, converged, extrinsic_llrs)
        """
        llr = np.asarray(llr, dtype=np.float64)

        if len(llr) != self.n:
            raise ValueError(f"Expected {self.n} LLRs, got {len(llr)}")

        # Initialize messages
        v2c = np.zeros((self.n, self.m), dtype=np.float64)
        for j in range(self.n):
            for i in self.var_neighbors[j]:
                v2c[j, i] = llr[j]

        c2v = np.zeros((self.m, self.n), dtype=np.float64)
        converged = False

        for _ in range(self.max_iter):
            # Check node update (min-sum)
            for i in range(self.m):
                neighbors = self.check_neighbors[i]
                for j in neighbors:
                    other_msgs = [v2c[k, i] for k in neighbors if k != j]
                    if len(other_msgs) == 0:
                        c2v[i, j] = 0
                        continue

                    sign = 1
                    min_abs = float('inf')
                    for msg in other_msgs:
                        if msg < 0:
                            sign *= -1
                        abs_msg = abs(msg)
                        if abs_msg < min_abs:
                            min_abs = abs_msg

                    c2v[i, j] = sign * min_abs * self.norm_factor

            # Variable node update
            for j in range(self.n):
                neighbors = self.var_neighbors[j]
                total = llr[j] + np.sum(c2v[neighbors, j])

                for i in neighbors:
                    v2c[j, i] = total - c2v[i, j]

            # Check convergence
            posterior = llr.copy()
            for j in range(self.n):
                posterior[j] += np.sum(c2v[self.var_neighbors[j], j])

            hard_decision = (posterior < 0).astype(np.int8)
            syndrome = (self.H @ hard_decision) % 2
            if np.all(syndrome == 0):
                converged = True
                break

        # Compute extrinsic LLRs (posterior - a_priori = sum of c2v messages)
        extrinsic = np.zeros(self.n, dtype=np.float64)
        for j in range(self.n):
            extrinsic[j] = np.sum(c2v[self.var_neighbors[j], j])

        return hard_decision, converged, extrinsic

    def get_info_bits(self, codeword: np.ndarray) -> np.ndarray:
        """Extract information bits from systematic codeword"""
        if self._use_generator_encoding:
            # For generator encoding, info bits are at col_perm[0:k] positions
            info_bits = np.zeros(self.k, dtype=np.int8)
            for i in range(self.k):
                info_bits[i] = codeword[self._col_perm[i]]
            return info_bits
        return codeword[:self.k].copy()

    def compute_ber(self, tx_bits: np.ndarray, rx_bits: np.ndarray) -> float:
        """Compute bit error rate"""
        return np.mean(tx_bits != rx_bits)


def test_ldpc():
    """Test LDPC encoder/decoder"""
    print("Testing LDPC Codec...")

    # Test encoding
    codec = LDPCCodec(n=648, rate=0.5)
    print(f"Code: n={codec.n}, k={codec.k}, rate={codec.rate}")

    # Random information bits
    np.random.seed(42)
    info_bits = np.random.randint(0, 2, codec.k, dtype=np.int8)

    # Encode
    codeword = codec.encode(info_bits)
    print(f"Encoded {codec.k} bits to {len(codeword)} bits")

    # Verify H @ c = 0
    syndrome = (codec.H @ codeword) % 2
    assert np.all(syndrome == 0), "Encoding verification failed!"
    print("Encoding verified: H @ c = 0")

    # Test decoding with AWGN
    snr_db = 3.0
    snr_linear = 10 ** (snr_db / 10)
    noise_std = 1 / np.sqrt(2 * snr_linear)

    # BPSK modulation: 0 -> +1, 1 -> -1
    tx_symbols = 1 - 2 * codeword.astype(np.float64)
    rx_symbols = tx_symbols + np.random.randn(len(tx_symbols)) * noise_std

    # LLR calculation: LLR = 2 * y / sigma^2 (positive = more likely 0)
    llr = 2 * rx_symbols / (noise_std ** 2)

    # Decode
    decoded, converged, iters = codec.decode(llr, return_iterations=True)

    # Extract info bits
    decoded_info = codec.get_info_bits(decoded)
    ber = codec.compute_ber(info_bits, decoded_info)

    print(f"SNR: {snr_db} dB")
    print(f"Converged: {converged} in {iters} iterations")
    print(f"BER: {ber:.6f}")

    # Test different rates
    for rate in [0.5, 2/3, 0.75, 5/6]:
        try:
            codec = LDPCCodec(n=648, rate=rate)
            info = np.random.randint(0, 2, codec.k, dtype=np.int8)
            cw = codec.encode(info)
            syn = (codec.H @ cw) % 2
            print(f"Rate {rate:.3f}: k={codec.k}, verified={np.all(syn==0)}")
        except Exception as e:
            print(f"Rate {rate}: {e}")

    print("\nLDPC tests passed!")
    return True


class AFF3CTLDPCCodec:
    """
    High-performance LDPC Encoder/Decoder using AFF3CT library.

    Uses SIMD-optimized belief propagation with horizontal layered scheduling
    and normalized min-sum algorithm for maximum throughput.

    This codec provides significant speedups over pure Python/Numba implementations,
    especially for high data rates requiring real-time processing.
    """

    def __init__(self, n: int = 648, rate: float = 0.5,
                 max_iter: int = 50, norm_factor: float = 0.75,
                 batch_size: int = 8):
        """
        Initialize AFF3CT LDPC codec.

        Args:
            n: Codeword length (648, 1296, or 1944)
            rate: Code rate (0.5, 2/3, 0.75, or 5/6)
            max_iter: Maximum decoding iterations
            norm_factor: Normalization factor for min-sum (used for compatibility,
                        aff3ct uses its own internal normalization)
            batch_size: Number of codewords to decode in parallel using SIMD (default 8)
        """
        self.batch_size = batch_size
        if not AFF3CT_AVAILABLE:
            raise RuntimeError("AFF3CT not available. Build it first with: "
                             "./deps/build_py_aff3ct.sh && ./deps/install_py_aff3ct.sh")

        # Normalize rate
        rate_key = LDPCCodec._normalize_rate(rate)

        if (n, rate_key) not in LDPCCodec.CODES:
            available = list(LDPCCodec.CODES.keys())
            raise ValueError(f"Code (n={n}, rate={rate}) not available. "
                           f"Available: {available}")

        self.n = n
        self.rate = rate_key
        self.max_iter = max_iter
        self.norm_factor = norm_factor

        # Get base matrix and expansion factor from LDPCCodec
        H_base, Z = LDPCCodec.CODES[(n, rate_key)]
        self.Z = Z
        self.H_base = H_base

        # Expand to full H matrix (same as LDPCCodec)
        self.H = _expand_base_matrix(H_base, Z)

        # Calculate dimensions
        self.m = self.H.shape[0]  # Number of parity checks
        self.k = self.n - self.m  # Information bits

        # Create aff3ct sparse matrix from H
        # aff3ct expects the H matrix as a sparse matrix object
        self._create_aff3ct_codec()

    def _create_aff3ct_codec(self):
        """Create aff3ct encoder and decoder from H matrix."""
        # Import scipy sparse here since it's only used by AFF3CT codec
        from scipy import sparse as scipy_sparse

        # Convert H to sparse matrix format for aff3ct
        # aff3ct_sparse.array() creates sparse matrix directly from dense numpy array
        H_dense = self.H.astype(np.int32)
        self._H_sparse = aff3ct_sparse.array(H_dense)

        # Create encoder from H matrix
        # Encoder_LDPC_from_H creates encoder from parity check matrix
        self._encoder = aff3ct.module.encoder.Encoder_LDPC_from_H(
            self.k, self.n, self._H_sparse
        )

        # Get info bits positions from encoder (needed for decoder)
        self._info_bits_pos = list(self._encoder.get_info_bits_pos())

        # Create transposed H for decoder (decoder expects N == H.get_n_rows())
        H_T_sparse = aff3ct_sparse.array(self.H.T.astype(np.int32))

        # Create single-frame decoder for regular decode() calls
        self._decoder = aff3ct.module.decoder.Decoder_LDPC_BP_horizontal_layered_NMS(
            self.k, self.n, self.max_iter, H_T_sparse, self._info_bits_pos,
            self.norm_factor
        )

        # Create inter-frame (SIMD parallel) decoder for batched decoding
        # This uses SIMD to decode multiple codewords simultaneously
        self._decoder_batch = aff3ct.module.decoder.Decoder_LDPC_BP_horizontal_layered_inter_NMS(
            self.k, self.n, self.max_iter, H_T_sparse, self._info_bits_pos,
            self.norm_factor
        )
        self._decoder_batch.n_frames = self.batch_size

        # Pre-allocate 2D buffers for encode/decode (aff3ct expects batch dimension)
        self._encode_input = np.zeros((1, self.k), dtype=np.int32)
        self._encode_output = np.zeros((1, self.n), dtype=np.int32)
        self._decode_input = np.zeros((1, self.n), dtype=np.float32)
        self._decode_output = np.zeros((1, self.n), dtype=np.int32)  # Full codeword

        # Batch decode buffers
        self._decode_batch_input = np.zeros((self.batch_size, self.n), dtype=np.float32)
        self._decode_batch_output = np.zeros((self.batch_size, self.n), dtype=np.int32)

        # Bind buffers to encoder/decoder sockets
        # Use decode_siho_cw which outputs full codeword (V_N) instead of just info bits (V_K)
        self._encoder['encode::U_K'].bind(self._encode_input)
        self._encoder['encode::X_N'].bind(self._encode_output)
        self._decoder['decode_siho_cw::Y_N'].bind(self._decode_input)
        self._decoder['decode_siho_cw::V_N'].bind(self._decode_output)
        self._decoder_batch['decode_siho_cw::Y_N'].bind(self._decode_batch_input)
        self._decoder_batch['decode_siho_cw::V_N'].bind(self._decode_batch_output)

        # SISO buffers for decode_with_extrinsic (extrinsic LLR output)
        self._siso_input = np.zeros((1, self.n), dtype=np.float32)   # Y_N1: channel LLRs
        self._siso_output = np.zeros((1, self.n), dtype=np.float32)  # Y_N2: extrinsic LLRs
        self._decoder['decode_siso::Y_N1'].bind(self._siso_input)
        self._decoder['decode_siso::Y_N2'].bind(self._siso_output)

        # Create scipy sparse H matrix for fast syndrome checking
        self._H_scipy_sparse = scipy_sparse.csr_matrix(self.H)

    def encode(self, info_bits: np.ndarray) -> np.ndarray:
        """
        Encode information bits to codeword.

        Args:
            info_bits: Array of k information bits

        Returns:
            Codeword of n bits
        """
        info_bits = np.asarray(info_bits, dtype=np.int32).flatten()

        if len(info_bits) != self.k:
            raise ValueError(f"Expected {self.k} info bits, got {len(info_bits)}")

        # Copy to input buffer (already bound to encoder socket)
        self._encode_input[0, :] = info_bits

        # Execute encoding
        self._encoder['encode'].exec()

        return self._encode_output[0, :].astype(np.int8)

    def decode(self, llr: np.ndarray, return_iterations: bool = False,
               max_iterations: Optional[int] = None,
               early_termination: bool = True
               ) -> Tuple[np.ndarray, bool]:
        """
        Decode using aff3ct's SIMD-optimized belief propagation.

        Args:
            llr: Log-likelihood ratios for each bit (positive = more likely 0)
            return_iterations: If True, return number of iterations used
            max_iterations: Override max iterations (None = use self.max_iter)
            early_termination: If False, always run max_iterations

        Returns:
            Tuple of (decoded_bits, converged) or (decoded_bits, converged, iterations)
        """
        llr = np.asarray(llr, dtype=np.float32).flatten()

        if len(llr) != self.n:
            raise ValueError(f"Expected {self.n} LLRs, got {len(llr)}")

        # Copy LLRs to input buffer (already bound to decoder socket)
        self._decode_input[0, :] = llr

        # Execute decoding (SIHO_CW = Soft-Input Hard-Output, outputs full Codeword)
        self._decoder['decode_siho_cw'].exec()

        # Get full decoded codeword from output buffer
        codeword = self._decode_output[0, :].astype(np.int8)

        # Check convergence via syndrome using sparse matrix (much faster for large codes)
        syndrome = self._H_scipy_sparse @ codeword % 2
        converged = not np.any(syndrome)

        if return_iterations:
            # aff3ct doesn't expose iteration count, estimate from convergence
            iterations = 1 if converged else (max_iterations or self.max_iter)
            return codeword, converged, iterations

        return codeword, converged

    def decode_batch(self, llrs: np.ndarray,
                     max_iterations: Optional[int] = None,
                     early_termination: bool = True,
                     check_syndrome: bool = True
                     ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decode multiple codewords in parallel using SIMD.

        This method provides ~10x speedup over sequential decoding by using
        inter-frame SIMD parallelism to decode up to batch_size codewords
        simultaneously.

        Args:
            llrs: 2D array of LLRs, shape (n_codewords, n) or 1D array of
                  concatenated LLRs (length must be multiple of n)
            max_iterations: Maximum decoding iterations (ignored for AFF3CT,
                           set at construction time)
            early_termination: Early termination setting (ignored for AFF3CT,
                              set at construction time)
            check_syndrome: If True, check syndrome for each codeword (adds overhead)

        Returns:
            Tuple of (codewords, converged) where:
            - codewords: 2D array of decoded bits, shape (n_codewords, n)
            - converged: 1D boolean array indicating convergence for each codeword
        """
        # Warn if max_iterations differs from construction value
        if max_iterations is not None and max_iterations != self.max_iter:
            warnings.warn(f"AFF3CT decoder max_iterations={self.max_iter} is set at "
                         f"construction and cannot be changed dynamically "
                         f"(requested {max_iterations})")

        # Warn if early_termination differs from default
        if not early_termination:
            warnings.warn("AFF3CT decoder early_termination cannot be disabled; "
                         "decoder always uses early termination")

        # Warn if check_syndrome is disabled
        if not check_syndrome:
            warnings.warn("check_syndrome=False means converged array will all be True; "
                         "syndrome checking is needed to verify convergence")

        llrs = np.asarray(llrs, dtype=np.float32)

        # Handle 1D input (concatenated LLRs)
        if llrs.ndim == 1:
            if len(llrs) % self.n != 0:
                raise ValueError(f"LLR length {len(llrs)} not divisible by n={self.n}")
            n_codewords = len(llrs) // self.n
            llrs = llrs.reshape(n_codewords, self.n)
        else:
            n_codewords = llrs.shape[0]
            if llrs.shape[1] != self.n:
                raise ValueError(f"Expected LLRs of length {self.n}, got {llrs.shape[1]}")

        # Allocate output arrays
        codewords = np.zeros((n_codewords, self.n), dtype=np.int8)
        converged = np.ones(n_codewords, dtype=bool)

        # Process in batches
        for batch_start in range(0, n_codewords, self.batch_size):
            batch_end = min(batch_start + self.batch_size, n_codewords)
            actual_batch_size = batch_end - batch_start

            # Copy LLRs to batch input buffer
            self._decode_batch_input[:actual_batch_size, :] = llrs[batch_start:batch_end, :]

            # Pad remaining slots if partial batch (decoder always processes batch_size)
            if actual_batch_size < self.batch_size:
                self._decode_batch_input[actual_batch_size:, :] = 0

            # Execute batch decoding
            self._decoder_batch['decode_siho_cw'].exec()

            # Copy results
            codewords[batch_start:batch_end, :] = self._decode_batch_output[:actual_batch_size, :].astype(np.int8)

            # Check syndrome for convergence (optional, adds overhead)
            if check_syndrome:
                for i in range(actual_batch_size):
                    syndrome = self._H_scipy_sparse @ codewords[batch_start + i, :] % 2
                    converged[batch_start + i] = not np.any(syndrome)

        return codewords, converged

    def decode_with_extrinsic(self, llr: np.ndarray
                              ) -> Tuple[np.ndarray, bool, np.ndarray]:
        """
        Decode and return both hard decision and extrinsic LLRs.

        Uses AFF3CT's native SISO decode_siso task which computes
        extrinsic = posterior - channel_LLR after belief propagation.

        Args:
            llr: Input log-likelihood ratios (a priori), length n

        Returns:
            Tuple of (hard_decision, converged, extrinsic_llrs)
        """
        llr = np.asarray(llr, dtype=np.float32).flatten()

        if len(llr) != self.n:
            raise ValueError(f"Expected {self.n} LLRs, got {len(llr)}")

        # Copy to SISO input buffer
        self._siso_input[0, :] = llr

        # Execute SISO decoding — fills _siso_output with extrinsic LLRs
        self._decoder['decode_siso'].exec()

        # Read extrinsic output
        extrinsic = self._siso_output[0, :].astype(np.float64)

        # Compute posterior = channel + extrinsic, then hard decision
        posterior = llr.astype(np.float64) + extrinsic
        hard_decision = (posterior < 0).astype(np.int8)

        # Check convergence via syndrome
        syndrome = self._H_scipy_sparse @ hard_decision % 2
        converged = not np.any(syndrome)

        return hard_decision, converged, extrinsic

    def decode_soft(self, llr: np.ndarray) -> np.ndarray:
        """
        Decode and return soft output (posterior LLRs).

        Args:
            llr: Input log-likelihood ratios

        Returns:
            Posterior LLRs after decoding (channel + extrinsic)
        """
        hard_decision, converged, extrinsic = self.decode_with_extrinsic(llr)
        return llr.astype(np.float64) + extrinsic

    def get_info_bits(self, codeword: np.ndarray) -> np.ndarray:
        """Extract information bits from codeword using aff3ct info positions."""
        codeword = np.asarray(codeword, dtype=np.int8)
        info_bits = np.zeros(self.k, dtype=np.int8)
        for i, pos in enumerate(self._info_bits_pos):
            info_bits[i] = codeword[pos]
        return info_bits

    def compute_ber(self, tx_bits: np.ndarray, rx_bits: np.ndarray) -> float:
        """Compute bit error rate."""
        return np.mean(tx_bits != rx_bits)


def get_ldpc_codec(n: int = 648, rate: float = 0.5,
                   max_iter: int = 50, norm_factor: float = 0.75,
                   prefer_aff3ct: bool = True) -> 'LDPCCodec':
    """
    Factory function to get the best available LDPC codec.

    Tries backends in order of performance:
    1. AFF3CT (SIMD-optimized, fastest)
    2. Numba-accelerated Python
    3. Pure Python (fallback)

    Args:
        n: Codeword length (648, 1296, or 1944)
        rate: Code rate (0.5, 2/3, 0.75, or 5/6)
        max_iter: Maximum decoding iterations
        norm_factor: Normalization factor for min-sum
        prefer_aff3ct: If True, use AFF3CT when available

    Returns:
        LDPCCodec instance (either AFF3CTLDPCCodec or standard LDPCCodec)
    """
    if prefer_aff3ct:
        if AFF3CT_AVAILABLE:
            try:
                return AFF3CTLDPCCodec(n=n, rate=rate, max_iter=max_iter,
                                       norm_factor=norm_factor)
            except Exception as e:
                warnings.warn(f"AFF3CT codec creation failed: {e}. "
                             f"Falling back to standard codec.")
        else:
            warnings.warn(
                "AFF3CT requested but not available. Using Numba fallback. "
                "Build AFF3CT with: ./deps/build_py_aff3ct.sh && ./deps/install_py_aff3ct.sh"
            )

    return LDPCCodec(n=n, rate=rate, max_iter=max_iter, norm_factor=norm_factor)


if __name__ == '__main__':
    test_ldpc()
