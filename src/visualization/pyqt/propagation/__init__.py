"""
HF Propagation Conditions Display - Standard Four + Advanced Monitoring

A standalone PyQt6 application for monitoring space weather parameters
that affect HF radio propagation.

Standard Four (Solar Drivers):
1. X-Ray Flux (R-Scale) - Solar flare radio blackouts
2. Kp Index (G-Scale) - Geomagnetic storm conditions
3. Proton Flux (S-Scale) - Solar radiation storms
4. Solar Wind Bz - Storm precursor indicator

Advanced Ionospheric (Response Indicators):
5. F10.7 Solar Flux - Baseline MUF calculation (EUV proxy)
6. GIRO Ionosonde - Real-time foF2/MUF ground truth
7. Hemispheric Power Index - Aurora energy deposition
8. D-RAP Absorption - HF blackout zones

Predictions:
9. WSA-Enlil - CME arrival prediction
10. Propagated Solar Wind - Near-term forecast
"""

__version__ = "2.0.0"
__author__ = "AutoNVIS Project"

# Standard Four components
from .main_window import PropagationMainWindow
from .data_client import PropagationDataClient
from .widgets import XRayWidget, KpWidget, ProtonWidget, SolarWindWidget

# Advanced components
from .advanced_data_client import AdvancedDataClient
from .advanced_widgets import (
    F107Widget, HPIWidget, DRAPWidget, IonosondeWidget,
    EnlilWidget, PropagatedWindWidget
)

# Enhanced combined display
from .enhanced_main_window import EnhancedPropagationWindow

__all__ = [
    # Standard Four
    'PropagationMainWindow',
    'PropagationDataClient',
    'XRayWidget',
    'KpWidget',
    'ProtonWidget',
    'SolarWindWidget',
    # Advanced
    'AdvancedDataClient',
    'F107Widget',
    'HPIWidget',
    'DRAPWidget',
    'IonosondeWidget',
    'EnlilWidget',
    'PropagatedWindWidget',
    # Enhanced
    'EnhancedPropagationWindow',
]
