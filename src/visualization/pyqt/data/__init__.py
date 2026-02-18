"""
Data Management for PyQt TEC Display
"""

# Lazy imports to avoid pulling in optional dependencies (like QtWebSockets)
def __getattr__(name):
    if name == 'DataManager':
        from .data_manager import DataManager
        return DataManager
    if name == 'DashboardWebSocketClient':
        from .websocket_client import DashboardWebSocketClient
        return DashboardWebSocketClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ['DataManager', 'DashboardWebSocketClient']
