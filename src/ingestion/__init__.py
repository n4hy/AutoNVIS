"""
Auto-NVIS Data Ingestion Services

This module handles real-time ingestion of ionospheric and space weather data
from multiple sources including GOES, ACE, GNSS-TEC, and ionosondes.

Key Components:
    - giro_client: Real-time ionosonde data from GIRO network
    - live_iono_client: Integrated live ionospheric data client
"""

__version__ = "0.1.0"

# Import main client classes for convenient access
try:
    from .giro_client import (
        GIROClient,
        GIRODataWorker,
        GIROStation,
        IonosondeMeasurement,
        DEFAULT_GIRO_STATIONS,
        generate_simulated_measurement,
    )
except ImportError:
    pass

try:
    from .live_iono_client import (
        LiveIonoClient,
        LiveIonosphericState,
        LiveIonoModelUpdater,
    )
except ImportError:
    pass
