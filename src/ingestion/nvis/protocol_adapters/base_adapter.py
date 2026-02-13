"""
Base Protocol Adapter for NVIS Sounder Data Ingestion

Defines the abstract interface and data structures for all protocol adapters.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, AsyncIterator
from datetime import datetime


@dataclass
class NVISMeasurement:
    """
    NVIS sounder measurement data structure

    Represents a single NVIS propagation observation including
    geometry, observables, and quality metadata.
    """
    # Geometry
    tx_latitude: float  # deg
    tx_longitude: float  # deg
    tx_altitude: float  # m
    rx_latitude: float  # deg
    rx_longitude: float  # deg
    rx_altitude: float  # m

    # Propagation parameters
    frequency: float  # MHz
    elevation_angle: float  # deg (70-90° for NVIS)
    azimuth: float  # deg
    hop_distance: float  # km

    # Observables
    signal_strength: float  # dBm
    group_delay: float  # ms
    snr: float  # dB

    # Quality metadata (assigned by quality assessor)
    signal_strength_error: float = 2.0  # dB (default PLATINUM)
    group_delay_error: float = 0.1  # ms (default PLATINUM)

    # Metadata
    sounder_id: str = ""
    timestamp: str = ""
    is_o_mode: bool = True  # Ordinary vs Extraordinary wave

    # Equipment metadata
    tx_power: Optional[float] = None  # Watts
    tx_antenna_gain: Optional[float] = None  # dBi
    rx_antenna_gain: Optional[float] = None  # dBi
    bandwidth: Optional[float] = None  # kHz

    def __post_init__(self):
        """Validate measurement parameters"""
        if not -90 <= self.tx_latitude <= 90:
            raise ValueError(f"Invalid tx_latitude: {self.tx_latitude}")
        if not -180 <= self.tx_longitude <= 180:
            raise ValueError(f"Invalid tx_longitude: {self.tx_longitude}")
        if not -90 <= self.rx_latitude <= 90:
            raise ValueError(f"Invalid rx_latitude: {self.rx_latitude}")
        if not -180 <= self.rx_longitude <= 180:
            raise ValueError(f"Invalid rx_longitude: {self.rx_longitude}")
        if not 2.0 <= self.frequency <= 30.0:
            raise ValueError(f"Invalid frequency for NVIS: {self.frequency} MHz")
        if not 70.0 <= self.elevation_angle <= 90.0:
            raise ValueError(f"Invalid NVIS elevation angle: {self.elevation_angle}°")

        # Set timestamp if not provided
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat() + 'Z'


@dataclass
class SounderMetadata:
    """Metadata about a sounder station"""
    sounder_id: str
    name: str
    operator: str
    location: str
    latitude: float
    longitude: float
    altitude: float
    equipment_type: str  # "professional", "amateur", "research"
    calibration_status: str  # "calibrated", "uncalibrated", "unknown"
    data_rate: Optional[float] = None  # observations per hour
    uptime_percent: Optional[float] = None
    last_seen: Optional[str] = None


class BaseAdapter(ABC):
    """
    Abstract base class for NVIS sounder protocol adapters

    All adapters must implement:
    - start(): Begin listening for measurements
    - stop(): Gracefully shutdown
    - get_measurements(): Async iterator yielding measurements
    """

    def __init__(self, adapter_id: str, config: dict):
        """
        Initialize adapter

        Args:
            adapter_id: Unique identifier for this adapter instance
            config: Adapter-specific configuration dict
        """
        self.adapter_id = adapter_id
        self.config = config
        self.running = False
        self.measurement_count = 0

    @abstractmethod
    async def start(self):
        """Start the adapter (begin listening for data)"""
        pass

    @abstractmethod
    async def stop(self):
        """Stop the adapter (graceful shutdown)"""
        pass

    @abstractmethod
    async def get_measurements(self) -> AsyncIterator[NVISMeasurement]:
        """
        Async iterator yielding NVIS measurements as they arrive

        Yields:
            NVISMeasurement: Parsed and validated measurement
        """
        pass

    @abstractmethod
    def get_sounder_metadata(self, sounder_id: str) -> Optional[SounderMetadata]:
        """
        Retrieve metadata for a sounder

        Args:
            sounder_id: Sounder identifier

        Returns:
            SounderMetadata if found, None otherwise
        """
        pass

    def validate_measurement(self, measurement: NVISMeasurement) -> bool:
        """
        Validate measurement is complete and reasonable

        Args:
            measurement: Measurement to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            # Check required fields are present
            if not measurement.sounder_id:
                return False
            if not measurement.timestamp:
                return False

            # Check signal strength is reasonable
            if not -140 <= measurement.signal_strength <= 0:
                return False

            # Check SNR is reasonable
            if not -20 <= measurement.snr <= 60:
                return False

            # Check group delay is reasonable (< 10ms for NVIS)
            if not 0 <= measurement.group_delay <= 10:
                return False

            return True

        except Exception:
            return False
