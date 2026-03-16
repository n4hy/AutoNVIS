"""
SC-FDE Shared Utilities

Common utilities shared between transmitter and receiver.
"""

import numpy as np
from typing import Tuple


def compute_subcarrier_indices(fft_size: int,
                                n_pilots: int,
                                n_data: int,
                                pilot_spacing: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute pilot, data, and null subcarrier indices.

    This function provides a single implementation of subcarrier index
    computation used by both transmitter and receiver to ensure consistency.

    The subcarrier allocation pattern is:
    - DC (index 0) is null
    - Active carriers are centered around DC (positive freq bins 1..N/2,
      negative freq bins N/2+1..N-1), with guard bands at the Nyquist edge
    - Pilots are placed every pilot_spacing carriers within active band
    - Data carriers fill in between pilots

    Args:
        fft_size: FFT size
        n_pilots: Number of pilot subcarriers
        n_data: Number of data subcarriers
        pilot_spacing: Spacing between pilots (e.g., 8 means 1 pilot per 8 carriers)

    Returns:
        Tuple of (pilot_indices, data_indices, null_indices)
        All arrays are sorted in ascending order.
    """
    # Total active carriers needed
    n_active = n_pilots + n_data

    # Center active carriers around DC in the complex baseband spectrum.
    # Positive freq carriers: bins 1, 2, ..., n_pos
    # Negative freq carriers: bins N-n_neg, ..., N-1
    # Guard bands go at the Nyquist edge (middle of bin index range).
    n_pos = n_active // 2
    n_neg = n_active - n_pos

    active_bins = np.concatenate([
        np.arange(1, 1 + n_pos),                          # positive freq
        np.arange(fft_size - n_neg, fft_size),             # negative freq
    ])
    active_bins = np.sort(active_bins)

    # Place pilots at regular spacing within the active band, data in between
    pilot_indices = []
    data_indices = []

    for i, bin_idx in enumerate(active_bins):
        if len(pilot_indices) < n_pilots and i % pilot_spacing == 0:
            pilot_indices.append(bin_idx)
        elif len(data_indices) < n_data:
            data_indices.append(bin_idx)
        else:
            pilot_indices.append(bin_idx)

    pilot_indices = np.array(pilot_indices[:n_pilots], dtype=int)
    data_indices = np.array(data_indices[:n_data], dtype=int)

    # Compute null indices (everything not pilot or data)
    active_set = set(pilot_indices) | set(data_indices)
    null_indices = np.array(
        sorted([i for i in range(fft_size) if i not in active_set]),
        dtype=int
    )

    return pilot_indices, data_indices, null_indices
