"""
CRC-16-CCITT Error Detection

Implements CRC-16-CCITT (polynomial 0x1021) for frame error detection.
Used after LDPC decoding to detect residual errors.

Standard: ITU-T X.25, HDLC
Polynomial: x^16 + x^12 + x^5 + 1
Init: 0xFFFF
"""

import numpy as np
from typing import Tuple

# CRC-16-CCITT polynomial
CRC16_POLY = 0x1021

# Precomputed lookup table for byte-at-a-time CRC computation
# Entry i contains CRC of byte i with zero initial remainder
# Using Python list instead of numpy array to avoid uint16 overflow issues
CRC_TABLE = []


def _init_crc_table():
    """Initialize CRC lookup table"""
    global CRC_TABLE
    CRC_TABLE = []

    for i in range(256):
        crc = i << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ CRC16_POLY) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
        CRC_TABLE.append(crc)


# Initialize table at module load
_init_crc_table()


def crc16_ccitt(bits: np.ndarray) -> int:
    """
    Compute CRC-16-CCITT checksum.

    Args:
        bits: Input data as bit array (0s and 1s), any length

    Returns:
        16-bit CRC value (0x0000 to 0xFFFF)
    """
    # Convert bits to bytes (pad to byte boundary)
    bits = np.asarray(bits, dtype=np.uint8)

    # Pad to multiple of 8 bits
    pad_len = (8 - len(bits) % 8) % 8
    if pad_len > 0:
        bits = np.concatenate([bits, np.zeros(pad_len, dtype=np.uint8)])

    # Pack bits into bytes
    bytes_data = np.packbits(bits)

    # Compute CRC using lookup table
    crc = 0xFFFF  # Initial value

    for byte in bytes_data:
        table_idx = ((crc >> 8) ^ byte) & 0xFF
        crc = ((crc << 8) ^ CRC_TABLE[table_idx]) & 0xFFFF

    return int(crc)


def append_crc(info_bits: np.ndarray) -> np.ndarray:
    """
    Append 16-bit CRC to information bits.

    Args:
        info_bits: Information bits (0s and 1s)

    Returns:
        Data with CRC appended (length = len(info_bits) + 16)
    """
    info_bits = np.asarray(info_bits, dtype=np.int8)

    # Compute CRC
    crc = crc16_ccitt(info_bits)

    # Convert CRC to 16 bits (MSB first)
    crc_bits = np.array([(crc >> (15 - i)) & 1 for i in range(16)], dtype=np.int8)

    return np.concatenate([info_bits, crc_bits])


def check_crc(data_with_crc: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Check CRC and extract data.

    Args:
        data_with_crc: Data with 16-bit CRC appended

    Returns:
        Tuple of (data_without_crc, crc_valid)
    """
    data_with_crc = np.asarray(data_with_crc, dtype=np.int8)

    if len(data_with_crc) < 16:
        return np.array([], dtype=np.int8), False

    # Extract data and received CRC
    data = data_with_crc[:-16]
    received_crc_bits = data_with_crc[-16:]

    # Convert received CRC bits to integer
    received_crc = 0
    for i, bit in enumerate(received_crc_bits):
        received_crc |= (int(bit) << (15 - i))

    # Compute expected CRC
    computed_crc = crc16_ccitt(data)

    # Check match
    valid = (computed_crc == received_crc)

    return data, valid
