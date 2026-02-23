"""
HF Ray Tracer Visualization

Visualizes ionospheric ray tracing results:
- LUF/MUF display
- Ray path cross-sections
- NVIS coverage maps

IONORT-style visualizations:
- Altitude vs Ground Range (Figures 5, 7, 9)
- 3D Geographic View (Figures 7, 8)
- Synthetic Oblique Ionogram (Figures 11-16)
"""

from .main_window import RayTracerMainWindow
from .ionort_widgets import (
    AltitudeGroundRangeWidget,
    Geographic3DWidget,
    SyntheticIonogramWidget,
    IONORTVisualizationPanel,
    frequency_to_color,
)

__all__ = [
    'RayTracerMainWindow',
    # IONORT-style widgets
    'AltitudeGroundRangeWidget',
    'Geographic3DWidget',
    'SyntheticIonogramWidget',
    'IONORTVisualizationPanel',
    'frequency_to_color',
]
