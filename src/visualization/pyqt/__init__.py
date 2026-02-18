"""
PyQt6 TEC Display Application

Real-time visualization of GNSS-TEC and GloTEC data using PyQt6 and pyqtgraph.

Subpackages:
- propagation: HF propagation conditions display
- raytracer: Ray tracer visualization (in src/raytracer/)
"""

# Lazy imports to avoid pulling in optional dependencies (like QtWebSockets)
def __getattr__(name):
    if name == 'TECDisplayMainWindow':
        from .main_window import TECDisplayMainWindow
        return TECDisplayMainWindow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ['TECDisplayMainWindow']
