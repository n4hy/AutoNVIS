"""
NVIS Analytics Dashboard

Provides web-based visualization and monitoring for the NVIS sounder network.
"""

from .nvis_analytics_api import create_app

__all__ = ['create_app']
