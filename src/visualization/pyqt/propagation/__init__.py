"""
HF Propagation Conditions Dashboard

Combines X-ray flux, Kp index, proton flux, and solar wind Bz
for comprehensive HF propagation monitoring.
"""

from .main_window import PropagationMainWindow
from .data_client import PropagationDataClient

__all__ = ['PropagationMainWindow', 'PropagationDataClient']
