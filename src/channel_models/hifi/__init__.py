"""
HiFi - High Frequency SC-FDE Transceiver

A complete wideband HF transceiver with:
- MMSE equalization
- LDPC coding (IEEE 802.11n)
- Support for BPSK through 64QAM
- Adaptive cyclic prefix for HF channels
- Vogler-Hoffmeyer ionospheric channel model

System Parameters:
    Sample Rate: 1.536 MSPS
    Bandwidth: 768 kHz
    FFT Size: 4096 (standard) / 8192 (severe)
    CP Length: 256-3200 samples (adaptive)
"""

# Core SC-FDE components
from .ldpc import LDPCCodec, get_ldpc_codec, AFF3CT_AVAILABLE
from .modulator import Modulator, Demodulator
from .interleaver import BitInterleaver, SymbolInterleaver
from .transmitter import SCFDETransmitter, TransmitterConfig
from .receiver import SCFDEReceiver, ReceiverConfig
from .sync import Synchronizer, SyncConfig
from .system import SCFDESystem, SystemConfig
from .streaming_receiver import StreamingReceiver
from .feature_config import FeatureConfig

# Vogler-Hoffmeyer channel model
from .vogler_hoffmeyer_channel import (
    VoglerHoffmeyerChannel,
    ChannelConfig,
    ModeParameters,
    CorrelationType,
    # Preset factory functions
    create_equatorial_config,
    create_polar_config,
    create_midlatitude_config,
    create_auroral_config,
    create_auroral_spread_f_config,
    create_auroral_complex_config,
    create_wideband_dispersive_config,
    get_preset,
    list_presets,
)

__version__ = '0.1.0'

__all__ = [
    # Version
    '__version__',
    # LDPC
    'LDPCCodec',
    'get_ldpc_codec',
    'AFF3CT_AVAILABLE',
    # Modulation
    'Modulator',
    'Demodulator',
    # Interleaving
    'BitInterleaver',
    'SymbolInterleaver',
    # Transmitter
    'SCFDETransmitter',
    'TransmitterConfig',
    # Receiver
    'SCFDEReceiver',
    'ReceiverConfig',
    'StreamingReceiver',
    # Synchronization
    'Synchronizer',
    'SyncConfig',
    # System
    'SCFDESystem',
    'SystemConfig',
    'FeatureConfig',
    # Channel model
    'VoglerHoffmeyerChannel',
    'ChannelConfig',
    'ModeParameters',
    'CorrelationType',
    'create_equatorial_config',
    'create_polar_config',
    'create_midlatitude_config',
    'create_auroral_config',
    'create_auroral_spread_f_config',
    'create_auroral_complex_config',
    'create_wideband_dispersive_config',
    'get_preset',
    'list_presets',
]
