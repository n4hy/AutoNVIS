"""
RTCM3 Message Parser

Parses RTCM3 binary messages to extract GNSS observables for TEC calculation.
Focuses on GPS (1004) and GLONASS (1012) observable messages containing
dual-frequency pseudorange and carrier phase measurements.

RTCM3 Message Format:
    - Preamble: 0xD3 (1 byte)
    - Reserved + Length: 2 bytes
    - Message: variable length
    - CRC24Q: 3 bytes

Reference: RTCM Standard 10403.3
"""

import struct
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.common.logging_config import ServiceLogger


# RTCM3 Constants
RTCM3_PREAMBLE = 0xD3
RTCM3_HEADER_SIZE = 3  # Preamble + length
RTCM3_CRC_SIZE = 3

# Message types of interest
MSG_TYPE_GPS_L1_L2 = 1004  # GPS L1/L2 observables (extended)
MSG_TYPE_GLONASS_L1_L2 = 1012  # GLONASS L1/L2 observables (extended)
MSG_TYPE_ANTENNA = 1005  # Stationary RTK reference station ARP
MSG_TYPE_GPS_EPHEMERIS = 1019  # GPS ephemeris
MSG_TYPE_GLONASS_EPHEMERIS = 1020  # GLONASS ephemeris


@dataclass
class GNSSObservable:
    """GNSS observable for a single satellite"""
    satellite_id: int  # PRN for GPS, slot for GLONASS
    gnss_type: str  # 'GPS' or 'GLONASS'
    epoch_time: datetime  # Observation epoch

    # L1 observables
    l1_pseudorange: Optional[float] = None  # meters
    l1_carrier_phase: Optional[float] = None  # cycles
    l1_snr: Optional[float] = None  # dB-Hz

    # L2 observables
    l2_pseudorange: Optional[float] = None  # meters
    l2_carrier_phase: Optional[float] = None  # cycles
    l2_snr: Optional[float] = None  # dB-Hz

    # Quality indicators
    code_indicator: int = 0
    lock_time_indicator: int = 0


@dataclass
class StationInfo:
    """GNSS station information"""
    station_id: int
    x: float  # ECEF X coordinate (meters)
    y: float  # ECEF Y coordinate (meters)
    z: float  # ECEF Z coordinate (meters)


class RTCM3Parser:
    """
    Parser for RTCM3 binary messages

    Extracts GNSS observables from RTCM3 stream for TEC calculation.
    """

    def __init__(self):
        """Initialize RTCM3 parser"""
        self.logger = ServiceLogger("ingestion", "rtcm3_parser")

        # Buffer for incomplete messages
        self._buffer = bytearray()

        # Statistics
        self._messages_parsed = 0
        self._parse_errors = 0

        # Reference station info
        self._station_info: Optional[StationInfo] = None

    def add_data(self, data: bytes) -> List[Dict[str, Any]]:
        """
        Add data to parser and extract complete messages

        Args:
            data: Binary data from NTRIP stream

        Returns:
            List of parsed message dictionaries
        """
        self._buffer.extend(data)
        messages = []

        while True:
            message = self._extract_message()
            if message is None:
                break
            messages.append(message)

        return messages

    def _extract_message(self) -> Optional[Dict[str, Any]]:
        """
        Extract one complete RTCM3 message from buffer

        Returns:
            Parsed message dictionary, or None if no complete message available
        """
        # Need at least header to determine message length
        if len(self._buffer) < RTCM3_HEADER_SIZE:
            return None

        # Find preamble
        preamble_idx = self._find_preamble()
        if preamble_idx == -1:
            # No preamble found, clear buffer
            self._buffer.clear()
            return None

        # Remove data before preamble
        if preamble_idx > 0:
            self._buffer = self._buffer[preamble_idx:]

        # Parse header
        if len(self._buffer) < RTCM3_HEADER_SIZE:
            return None

        # Extract message length from header
        # Format: 0xD3 | 00 (6 bits reserved) + Length (10 bits)
        length_bytes = struct.unpack('>H', self._buffer[1:3])[0]
        message_length = length_bytes & 0x3FF  # Lower 10 bits

        # Total frame size
        frame_size = RTCM3_HEADER_SIZE + message_length + RTCM3_CRC_SIZE

        # Check if complete message available
        if len(self._buffer) < frame_size:
            return None

        # Extract complete frame
        frame = bytes(self._buffer[:frame_size])
        self._buffer = self._buffer[frame_size:]

        # Verify CRC
        if not self._verify_crc(frame):
            self.logger.warning("RTCM3 CRC check failed")
            self._parse_errors += 1
            return None

        # Extract message payload (skip header, exclude CRC)
        payload = frame[RTCM3_HEADER_SIZE:-RTCM3_CRC_SIZE]

        # Parse message
        try:
            message = self._parse_message(payload)
            if message:
                self._messages_parsed += 1
            return message
        except Exception as e:
            self.logger.error(f"Error parsing RTCM3 message: {e}", exc_info=True)
            self._parse_errors += 1
            return None

    def _find_preamble(self) -> int:
        """
        Find RTCM3 preamble (0xD3) in buffer

        Returns:
            Index of preamble, or -1 if not found
        """
        try:
            return self._buffer.index(RTCM3_PREAMBLE)
        except ValueError:
            return -1

    def _verify_crc(self, frame: bytes) -> bool:
        """
        Verify RTCM3 CRC24Q checksum

        Args:
            frame: Complete RTCM3 frame including CRC

        Returns:
            True if CRC valid
        """
        # Extract CRC from frame
        crc_received = struct.unpack('>I', b'\x00' + frame[-3:])[0]

        # Compute CRC for data (excluding CRC bytes)
        data = frame[:-3]
        crc_computed = self._compute_crc24q(data)

        return crc_received == crc_computed

    def _compute_crc24q(self, data: bytes) -> int:
        """
        Compute RTCM3 CRC24Q checksum

        Args:
            data: Data to compute CRC over

        Returns:
            24-bit CRC value
        """
        # CRC24Q polynomial: 0x1864CFB
        crc = 0
        for byte in data:
            crc ^= (byte << 16)
            for _ in range(8):
                crc <<= 1
                if crc & 0x1000000:
                    crc ^= 0x1864CFB
        return crc & 0xFFFFFF

    def _parse_message(self, payload: bytes) -> Optional[Dict[str, Any]]:
        """
        Parse RTCM3 message payload

        Args:
            payload: Message payload (without header/CRC)

        Returns:
            Parsed message dictionary, or None if message type not handled
        """
        if len(payload) < 2:
            return None

        # Extract message type (first 12 bits)
        msg_type = struct.unpack('>H', payload[0:2])[0] >> 4

        # Dispatch to appropriate parser
        if msg_type == MSG_TYPE_GPS_L1_L2:
            return self._parse_gps_observables(payload)
        elif msg_type == MSG_TYPE_GLONASS_L1_L2:
            return self._parse_glonass_observables(payload)
        elif msg_type == MSG_TYPE_ANTENNA:
            return self._parse_station_position(payload)
        elif msg_type == MSG_TYPE_GPS_EPHEMERIS:
            # GPS ephemeris - could be used for satellite position
            # For now, we'll use external ephemeris data
            return {'type': 'gps_ephemeris', 'msg_type': msg_type}
        elif msg_type == MSG_TYPE_GLONASS_EPHEMERIS:
            return {'type': 'glonass_ephemeris', 'msg_type': msg_type}
        else:
            # Unknown or unhandled message type
            return None

    def _parse_gps_observables(self, payload: bytes) -> Optional[Dict[str, Any]]:
        """
        Parse GPS L1/L2 observable message (Type 1004)

        Args:
            payload: Message payload

        Returns:
            Dictionary with GPS observables
        """
        try:
            # Message 1004 structure (simplified - full parsing is complex)
            # This is a simplified parser focusing on key fields
            # Production code should use a library like pyrtcm

            # Convert to bit stream for easier parsing
            bits = ''.join(format(byte, '08b') for byte in payload)

            # Message type (12 bits) - already verified
            pos = 12

            # Station ID (12 bits)
            station_id = int(bits[pos:pos+12], 2)
            pos += 12

            # GPS Epoch Time (30 bits) - milliseconds in GPS week
            epoch_time_ms = int(bits[pos:pos+30], 2)
            pos += 30

            # Synchronous GNSS flag (1 bit)
            pos += 1

            # Number of satellites (5 bits)
            num_sats = int(bits[pos:pos+5], 2)
            pos += 5

            # Smoothing indicator (1 bit)
            pos += 1

            # Smoothing interval (3 bits)
            pos += 3

            # Parse satellite observables (simplified)
            observables = []
            for _ in range(num_sats):
                # This is highly simplified - actual message is much more complex
                # Full parsing would extract all fields per satellite
                # For production, use pyrtcm or similar library

                # Satellite ID (6 bits)
                if pos + 6 > len(bits):
                    break
                sat_id = int(bits[pos:pos+6], 2)
                pos += 6

                # Skip detailed parsing for now
                # In production, would parse:
                # - Code indicator, L1 pseudorange, L1 phase, L1 lock time
                # - L2 code indicator, L2-L1 pseudorange diff, L2 phase, L2 lock time
                # - CNR values

                # Placeholder observable
                observables.append({
                    'satellite_id': sat_id,
                    'gnss_type': 'GPS'
                })

            return {
                'type': 'gps_observables',
                'msg_type': MSG_TYPE_GPS_L1_L2,
                'station_id': station_id,
                'epoch_time_ms': epoch_time_ms,
                'num_satellites': num_sats,
                'observables': observables
            }

        except Exception as e:
            self.logger.error(f"Error parsing GPS observables: {e}", exc_info=True)
            return None

    def _parse_glonass_observables(self, payload: bytes) -> Optional[Dict[str, Any]]:
        """
        Parse GLONASS L1/L2 observable message (Type 1012)

        Args:
            payload: Message payload

        Returns:
            Dictionary with GLONASS observables
        """
        # Similar to GPS parser - simplified for now
        # Production code should use proper RTCM3 library
        return {
            'type': 'glonass_observables',
            'msg_type': MSG_TYPE_GLONASS_L1_L2
        }

    def _parse_station_position(self, payload: bytes) -> Optional[Dict[str, Any]]:
        """
        Parse station position message (Type 1005)

        Args:
            payload: Message payload

        Returns:
            Dictionary with station position
        """
        try:
            bits = ''.join(format(byte, '08b') for byte in payload)

            pos = 12  # Skip message type

            # Station ID (12 bits)
            station_id = int(bits[pos:pos+12], 2)
            pos += 12

            # Reserved (6 bits)
            pos += 6

            # GPS indicator (1 bit)
            pos += 1

            # GLONASS indicator (1 bit)
            pos += 1

            # Reference station indicator (1 bit)
            pos += 1

            # ECEF X (38 bits, signed, 0.0001 m resolution)
            x_raw = int(bits[pos:pos+38], 2)
            if x_raw & (1 << 37):  # Sign bit
                x_raw -= (1 << 38)
            x = x_raw * 0.0001
            pos += 38

            # Single receiver oscillator (1 bit)
            pos += 1

            # ECEF Y (38 bits, signed, 0.0001 m resolution)
            y_raw = int(bits[pos:pos+38], 2)
            if y_raw & (1 << 37):
                y_raw -= (1 << 38)
            y = y_raw * 0.0001
            pos += 38

            # Quarter cycle indicator (2 bits)
            pos += 2

            # ECEF Z (38 bits, signed, 0.0001 m resolution)
            z_raw = int(bits[pos:pos+38], 2)
            if z_raw & (1 << 37):
                z_raw -= (1 << 38)
            z = z_raw * 0.0001

            self._station_info = StationInfo(
                station_id=station_id,
                x=x,
                y=y,
                z=z
            )

            return {
                'type': 'station_position',
                'msg_type': MSG_TYPE_ANTENNA,
                'station_id': station_id,
                'x': x,
                'y': y,
                'z': z
            }

        except Exception as e:
            self.logger.error(f"Error parsing station position: {e}", exc_info=True)
            return None

    @property
    def station_info(self) -> Optional[StationInfo]:
        """Get current station information"""
        return self._station_info

    @property
    def statistics(self) -> Dict[str, int]:
        """Get parser statistics"""
        return {
            'messages_parsed': self._messages_parsed,
            'parse_errors': self._parse_errors
        }
