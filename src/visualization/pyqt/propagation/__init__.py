"""
HF Propagation Conditions Display - Standard Four Monitoring

A standalone PyQt6 application for monitoring the four primary NOAA space
weather parameters that affect HF radio propagation:

1. X-Ray Flux (R-Scale) - Solar flare radio blackouts
2. Kp Index (G-Scale) - Geomagnetic storm conditions
3. Proton Flux (S-Scale) - Solar radiation storms
4. Solar Wind Bz - Storm precursor indicator
"""

__version__ = "1.1.0"
__author__ = "AutoNVIS Project"

from .main_window import PropagationMainWindow
from .data_client import PropagationDataClient
from .widgets import XRayWidget, KpWidget, ProtonWidget, SolarWindWidget

__all__ = [
    'PropagationMainWindow',
    'PropagationDataClient',
    'XRayWidget',
    'KpWidget',
    'ProtonWidget',
    'SolarWindWidget',
]
