"""
Backend modules for AutoNVIS GUI Dashboard

This package contains:
- Data subscribers for RabbitMQ message streams
- State management with thread-safe access
- Data processing utilities for ionospheric parameters
- API endpoint implementations
"""

from .subscribers import (
    GridDataSubscriber,
    PropagationSubscriber,
    SpaceWeatherSubscriber,
    ObservationSubscriber,
    SystemHealthSubscriber
)
from .state_manager import DashboardState
from .data_processing import (
    compute_fof2,
    compute_hmf2,
    compute_tec,
    extract_horizontal_slice,
    extract_vertical_profile
)

__all__ = [
    'GridDataSubscriber',
    'PropagationSubscriber',
    'SpaceWeatherSubscriber',
    'ObservationSubscriber',
    'SystemHealthSubscriber',
    'DashboardState',
    'compute_fof2',
    'compute_hmf2',
    'compute_tec',
    'extract_horizontal_slice',
    'extract_vertical_profile'
]
