"""
Information Gain Analysis for NVIS Sounder Network

Provides tools for:
- Computing marginal information gain per sounder
- Optimal placement recommendations
- Network quality assessment
"""

from .information_gain_analyzer import InformationGainAnalyzer
from .optimal_placement import OptimalPlacementRecommender
from .network_analyzer import NetworkAnalyzer

__all__ = [
    'InformationGainAnalyzer',
    'OptimalPlacementRecommender',
    'NetworkAnalyzer'
]
