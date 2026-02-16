"""
Data Manager for PyQt TEC Display

Thread-safe data buffer management for real-time visualization.
"""

from PyQt6.QtCore import QObject, pyqtSignal
from collections import deque
import threading
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import bisect


class DataManager(QObject):
    """
    Thread-safe data buffer manager for PyQt TEC display.

    Handles:
    - WebSocket data reception (async -> Qt signal)
    - Data buffering with configurable retention
    - Interpolation for selected points
    - Statistics calculation
    """

    # Signals for thread-safe UI updates
    glotec_map_updated = pyqtSignal(dict)
    tec_statistics_updated = pyqtSignal(dict)
    connection_status_changed = pyqtSignal(str, bool)

    def __init__(self, retention_hours: int = 24, parent=None):
        """
        Initialize data manager.

        Args:
            retention_hours: Hours of history to retain
            parent: Parent QObject
        """
        super().__init__(parent)

        self.retention_hours = retention_hours
        self.lock = threading.Lock()

        # Latest data
        self.latest_glotec_map: Optional[Dict[str, Any]] = None
        self.latest_timestamp: Optional[datetime] = None

        # History buffers (statistics only, not full grids)
        self.global_mean_history: deque = deque(maxlen=1000)
        self.tec_max_history: deque = deque(maxlen=1000)
        self.anomaly_mean_history: deque = deque(maxlen=1000)

        # Tracked point history
        self.tracked_point: Optional[Tuple[float, float]] = None  # (lat, lon)
        self.point_history: deque = deque(maxlen=1000)

        # Connection state
        self.websocket_connected = False
        self.last_update_time: Optional[datetime] = None

    def update_glotec_map(self, map_data: Dict[str, Any]):
        """
        Update from data source (thread-safe).

        Called from WebSocket handler thread.

        Args:
            map_data: GloTEC map dictionary with grid, statistics, metadata
        """
        with self.lock:
            self.latest_glotec_map = map_data
            self.latest_timestamp = datetime.utcnow()
            self.last_update_time = self.latest_timestamp

            # Extract statistics
            stats = map_data.get('statistics', {})
            timestamp = map_data.get('timestamp', datetime.utcnow().isoformat())

            # Convert timestamp string to datetime for plotting
            try:
                ts = datetime.fromisoformat(timestamp.rstrip('Z'))
                ts_float = ts.timestamp()
            except (ValueError, AttributeError):
                ts_float = datetime.utcnow().timestamp()

            # Update history buffers
            if stats.get('tec_mean') is not None:
                self.global_mean_history.append({
                    'timestamp': ts_float,
                    'value': stats['tec_mean']
                })

            if stats.get('tec_max') is not None:
                self.tec_max_history.append({
                    'timestamp': ts_float,
                    'value': stats['tec_max']
                })

            if stats.get('anomaly_mean') is not None:
                self.anomaly_mean_history.append({
                    'timestamp': ts_float,
                    'value': stats['anomaly_mean']
                })

            # Update tracked point if set
            if self.tracked_point is not None:
                lat, lon = self.tracked_point
                point_tec = self.get_tec_at_point(lat, lon)
                if point_tec is not None:
                    self.point_history.append({
                        'timestamp': ts_float,
                        'value': point_tec
                    })

        # Emit signals for UI update (thread-safe via Qt's signal queue)
        self.glotec_map_updated.emit(map_data)

        stats_update = {
            'timestamp': timestamp,
            'tec_mean': stats.get('tec_mean'),
            'tec_max': stats.get('tec_max'),
            'anomaly_mean': stats.get('anomaly_mean')
        }
        self.tec_statistics_updated.emit(stats_update)

    def set_tracked_point(self, lat: float, lon: float):
        """
        Set a point to track in time series.

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
        """
        with self.lock:
            self.tracked_point = (lat, lon)
            self.point_history.clear()

    def get_tec_at_point(self, lat: float, lon: float) -> Optional[float]:
        """
        Interpolate TEC value at specific lat/lon from latest map.

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees

        Returns:
            TEC value at point, or None if unavailable
        """
        # Note: lock should already be held when calling this
        if self.latest_glotec_map is None:
            return None

        grid = self.latest_glotec_map.get('grid', {})
        tec_grid = grid.get('tec')
        lat_grid = grid.get('lat', [])
        lon_grid = grid.get('lon', [])

        if tec_grid is None or not lat_grid or not lon_grid:
            return None

        # Find nearest grid indices
        lat_idx = self._find_nearest_idx(lat_grid, lat)
        lon_idx = self._find_nearest_idx(lon_grid, lon)

        if lat_idx is None or lon_idx is None:
            return None

        try:
            # Handle both 2D list and flat list formats
            if isinstance(tec_grid[0], list):
                return tec_grid[lat_idx][lon_idx]
            else:
                return tec_grid[lat_idx * len(lon_grid) + lon_idx]
        except (IndexError, TypeError):
            return None

    def _find_nearest_idx(self, grid: List[float], value: float) -> Optional[int]:
        """Find index of nearest value in sorted grid."""
        if not grid:
            return None

        idx = bisect.bisect_left(grid, value)

        if idx == 0:
            return 0
        if idx == len(grid):
            return len(grid) - 1

        # Compare neighbors
        if abs(grid[idx] - value) < abs(grid[idx - 1] - value):
            return idx
        return idx - 1

    def get_global_mean_history(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get global mean TEC history as numpy arrays.

        Returns:
            Tuple of (timestamps, values) as numpy arrays
        """
        with self.lock:
            if not self.global_mean_history:
                return np.array([]), np.array([])

            timestamps = np.array([h['timestamp'] for h in self.global_mean_history])
            values = np.array([h['value'] for h in self.global_mean_history])
            return timestamps, values

    def get_point_history(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get tracked point TEC history as numpy arrays.

        Returns:
            Tuple of (timestamps, values) as numpy arrays
        """
        with self.lock:
            if not self.point_history:
                return np.array([]), np.array([])

            timestamps = np.array([h['timestamp'] for h in self.point_history])
            values = np.array([h['value'] for h in self.point_history])
            return timestamps, values

    def get_latest_grid_arrays(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Get latest grid data as numpy arrays for plotting.

        Returns:
            Dictionary with lat, lon, tec, anomaly as numpy arrays, or None
        """
        with self.lock:
            if self.latest_glotec_map is None:
                return None

            grid = self.latest_glotec_map.get('grid', {})

            try:
                lat = np.array(grid.get('lat', []))
                lon = np.array(grid.get('lon', []))
                tec = np.array(grid.get('tec', []))
                anomaly = np.array(grid.get('anomaly', []))
                hmf2 = np.array(grid.get('hmF2', []))
                nmf2 = np.array(grid.get('NmF2', []))

                return {
                    'lat': lat,
                    'lon': lon,
                    'tec': tec,
                    'anomaly': anomaly,
                    'hmF2': hmf2,
                    'NmF2': nmf2,
                    'timestamp': self.latest_glotec_map.get('timestamp')
                }
            except (ValueError, TypeError):
                return None

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get current statistics summary.

        Returns:
            Dictionary with current statistics
        """
        with self.lock:
            if self.latest_glotec_map is None:
                return {'available': False}

            stats = self.latest_glotec_map.get('statistics', {})
            age = None
            if self.last_update_time:
                age = (datetime.utcnow() - self.last_update_time).total_seconds()

            return {
                'available': True,
                'timestamp': self.latest_glotec_map.get('timestamp'),
                'age_seconds': age,
                'tec_mean': stats.get('tec_mean'),
                'tec_max': stats.get('tec_max'),
                'tec_min': stats.get('tec_min'),
                'anomaly_mean': stats.get('anomaly_mean'),
                'n_valid_cells': stats.get('n_valid_cells'),
                'websocket_connected': self.websocket_connected
            }

    def set_connection_status(self, source: str, connected: bool):
        """
        Update connection status.

        Args:
            source: Connection source identifier
            connected: Whether connected
        """
        with self.lock:
            if source == 'websocket':
                self.websocket_connected = connected

        self.connection_status_changed.emit(source, connected)
