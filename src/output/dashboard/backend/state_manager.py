"""
Dashboard State Manager

Provides thread-safe centralized state management for dashboard data.
Maintains time-series buffers and latest data cache for all data streams.
"""

import threading
from collections import deque
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Deque
import numpy as np


class DashboardState:
    """
    Thread-safe state manager for dashboard data.

    Maintains:
    - Latest electron density grids
    - Time-series buffers (24-hour retention)
    - Propagation data (LUF/MUF/FOT history)
    - Space weather data (X-ray, solar wind)
    - Observation counts and quality
    - System health metrics
    """

    def __init__(self, retention_hours: int = 24):
        """
        Initialize dashboard state.

        Args:
            retention_hours: Number of hours to retain time-series data
        """
        self.retention_hours = retention_hours
        self.retention_seconds = retention_hours * 3600

        # Thread locks for safe access
        self.grid_lock = threading.Lock()
        self.propagation_lock = threading.Lock()
        self.spaceweather_lock = threading.Lock()
        self.observation_lock = threading.Lock()
        self.health_lock = threading.Lock()

        # --- Grid Data ---
        self.latest_grid: Optional[Dict[str, Any]] = None
        self.latest_grid_time: Optional[datetime] = None
        self.grid_history: Deque[Dict[str, Any]] = deque(maxlen=100)

        # --- Propagation Data ---
        self.latest_frequency_plan: Optional[Dict[str, Any]] = None
        self.latest_coverage_map: Optional[Dict[str, Any]] = None
        self.luf_history: Deque[Dict[str, float]] = deque(maxlen=1000)
        self.muf_history: Deque[Dict[str, float]] = deque(maxlen=1000)
        self.fot_history: Deque[Dict[str, float]] = deque(maxlen=1000)

        # --- Space Weather Data ---
        self.xray_flux_history: Deque[Dict[str, Any]] = deque(maxlen=10000)
        self.solar_wind_history: Deque[Dict[str, Any]] = deque(maxlen=1000)
        self.geomag_history: Deque[Dict[str, Any]] = deque(maxlen=1000)
        self.mode_history: Deque[Dict[str, Any]] = deque(maxlen=100)
        self.current_mode: str = "QUIET"

        # --- Observation Data ---
        self.observation_history: Deque[Dict[str, Any]] = deque(maxlen=10000)
        self.observation_counts: Dict[str, int] = {
            'gnss_tec': 0,
            'ionosonde': 0,
            'nvis_sounder': 0,
            'total': 0
        }

        # --- System Health Data ---
        self.service_status: Dict[str, Dict[str, Any]] = {}
        self.filter_metrics: Dict[str, Any] = {}
        self.alert_history: Deque[Dict[str, Any]] = deque(maxlen=1000)

        # --- GloTEC Data ---
        self.glotec_lock = threading.Lock()
        self.latest_glotec_map: Optional[Dict[str, Any]] = None
        self.latest_glotec_time: Optional[datetime] = None
        self.glotec_history: Deque[Dict[str, Any]] = deque(maxlen=144)  # 24h @ 10min intervals

        # Statistics
        self.stats = {
            'grids_received': 0,
            'propagation_updates': 0,
            'spaceweather_updates': 0,
            'observations_received': 0,
            'alerts_generated': 0,
            'glotec_maps_received': 0
        }

    # --- Grid Data Methods ---

    def update_grid(self, grid_data: Dict[str, Any], timestamp: datetime):
        """
        Update latest grid data.

        Args:
            grid_data: Grid data dictionary with ne_grid, lat, lon, alt
            timestamp: Grid timestamp
        """
        with self.grid_lock:
            self.latest_grid = grid_data
            self.latest_grid_time = timestamp
            self.grid_history.append({
                'timestamp': timestamp.isoformat(),
                'cycle_id': grid_data.get('cycle_id', 'unknown'),
                'ne_max': float(np.max(grid_data['ne_grid'])),
                'quality': grid_data.get('quality', 'unknown')
            })
            self.stats['grids_received'] += 1

    def get_latest_grid(self, max_age_seconds: float = 1200.0) -> Optional[Dict[str, Any]]:
        """
        Get latest grid if fresh enough.

        Args:
            max_age_seconds: Maximum acceptable age in seconds

        Returns:
            Grid data dictionary or None
        """
        with self.grid_lock:
            if self.latest_grid is None or self.latest_grid_time is None:
                return None

            age = (datetime.utcnow() - self.latest_grid_time).total_seconds()
            if age > max_age_seconds:
                return None

            return self.latest_grid

    def get_grid_metadata(self) -> Optional[Dict[str, Any]]:
        """Get metadata about latest grid without copying large arrays."""
        with self.grid_lock:
            if self.latest_grid is None:
                return None

            return {
                'cycle_id': self.latest_grid.get('cycle_id'),
                'timestamp': self.latest_grid.get('timestamp'),
                'age_seconds': (
                    datetime.utcnow() - self.latest_grid_time
                ).total_seconds() if self.latest_grid_time else None,
                'quality': self.latest_grid.get('quality'),
                'ne_max': float(np.max(self.latest_grid['ne_grid'])),
                'shape': self.latest_grid['ne_grid'].shape
            }

    # --- Propagation Data Methods ---

    def update_frequency_plan(self, plan_data: Dict[str, Any]):
        """Update latest frequency plan."""
        with self.propagation_lock:
            self.latest_frequency_plan = plan_data
            timestamp = datetime.fromisoformat(plan_data['timestamp'].rstrip('Z'))

            # Extract LUF/MUF/FOT values for history
            if 'luf_mhz' in plan_data:
                self.luf_history.append({
                    'timestamp': timestamp.isoformat(),
                    'value': plan_data['luf_mhz']
                })
            if 'muf_mhz' in plan_data:
                self.muf_history.append({
                    'timestamp': timestamp.isoformat(),
                    'value': plan_data['muf_mhz']
                })
            if 'fot_mhz' in plan_data:
                self.fot_history.append({
                    'timestamp': timestamp.isoformat(),
                    'value': plan_data['fot_mhz']
                })

            self.stats['propagation_updates'] += 1

    def update_coverage_map(self, coverage_data: Dict[str, Any]):
        """Update latest coverage map."""
        with self.propagation_lock:
            self.latest_coverage_map = coverage_data

    def get_frequency_plan(self) -> Optional[Dict[str, Any]]:
        """Get latest frequency plan."""
        with self.propagation_lock:
            return self.latest_frequency_plan

    def get_luf_muf_fot_history(self, hours: int = 24) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get LUF/MUF/FOT history for specified time window.

        Args:
            hours: Number of hours to include

        Returns:
            Dictionary with 'luf', 'muf', 'fot' lists
        """
        cutoff = datetime.utcnow() - timedelta(hours=hours)

        with self.propagation_lock:
            return {
                'luf': [
                    entry for entry in self.luf_history
                    if datetime.fromisoformat(entry['timestamp']) > cutoff
                ],
                'muf': [
                    entry for entry in self.muf_history
                    if datetime.fromisoformat(entry['timestamp']) > cutoff
                ],
                'fot': [
                    entry for entry in self.fot_history
                    if datetime.fromisoformat(entry['timestamp']) > cutoff
                ]
            }

    # --- Space Weather Methods ---

    def update_xray_flux(self, flux_data: Dict[str, Any]):
        """Add X-ray flux measurement to history."""
        with self.spaceweather_lock:
            self.xray_flux_history.append(flux_data)
            self._cleanup_old_data(self.xray_flux_history)
            self.stats['spaceweather_updates'] += 1

    def update_solar_wind(self, wind_data: Dict[str, Any]):
        """Add solar wind measurement to history."""
        with self.spaceweather_lock:
            self.solar_wind_history.append(wind_data)
            self._cleanup_old_data(self.solar_wind_history)

    def update_geomag(self, geomag_data: Dict[str, Any]):
        """Add geomagnetic measurement to history."""
        with self.spaceweather_lock:
            self.geomag_history.append(geomag_data)
            self._cleanup_old_data(self.geomag_history)

    def update_mode(self, mode: str, reason: str = ""):
        """Update autonomous mode."""
        with self.spaceweather_lock:
            self.current_mode = mode
            self.mode_history.append({
                'timestamp': datetime.utcnow().isoformat(),
                'mode': mode,
                'reason': reason
            })

    def get_xray_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get X-ray flux history."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        with self.spaceweather_lock:
            return [
                entry for entry in self.xray_flux_history
                if datetime.fromisoformat(entry['timestamp'].rstrip('Z')) > cutoff
            ]

    def get_solar_wind_latest(self) -> Optional[Dict[str, Any]]:
        """Get latest solar wind data."""
        with self.spaceweather_lock:
            if len(self.solar_wind_history) == 0:
                return None
            return self.solar_wind_history[-1]

    def get_mode_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get mode change history."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        with self.spaceweather_lock:
            return [
                entry for entry in self.mode_history
                if datetime.fromisoformat(entry['timestamp']) > cutoff
            ]

    # --- Observation Methods ---

    def add_observation(self, obs_data: Dict[str, Any], obs_type: str):
        """
        Add observation to history.

        Args:
            obs_data: Observation data
            obs_type: One of 'gnss_tec', 'ionosonde', 'nvis_sounder'
        """
        with self.observation_lock:
            obs_entry = obs_data.copy()
            obs_entry['obs_type'] = obs_type
            self.observation_history.append(obs_entry)
            self._cleanup_old_data(self.observation_history)

            # Update counts
            if obs_type in self.observation_counts:
                self.observation_counts[obs_type] += 1
            self.observation_counts['total'] += 1
            self.stats['observations_received'] += 1

    def get_observation_history(self, hours: int = 1) -> List[Dict[str, Any]]:
        """Get recent observations."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        with self.observation_lock:
            return [
                entry for entry in self.observation_history
                if datetime.fromisoformat(entry['timestamp'].rstrip('Z')) > cutoff
            ]

    def get_observation_counts(self) -> Dict[str, int]:
        """Get observation counts."""
        with self.observation_lock:
            return self.observation_counts.copy()

    # --- GloTEC Data Methods ---

    def update_glotec_map(self, map_data: Dict[str, Any]):
        """
        Update latest GloTEC map data.

        Args:
            map_data: GloTEC map dictionary with grid, statistics, metadata
        """
        with self.glotec_lock:
            self.latest_glotec_map = map_data
            self.latest_glotec_time = datetime.utcnow()

            # Add to history (store statistics only, not full grid)
            history_entry = {
                'timestamp': map_data.get('timestamp', datetime.utcnow().isoformat()),
                'statistics': map_data.get('statistics', {}),
                'metadata': map_data.get('metadata', {})
            }
            self.glotec_history.append(history_entry)
            self.stats['glotec_maps_received'] += 1

    def get_latest_glotec(self, max_age_seconds: float = 1200.0) -> Optional[Dict[str, Any]]:
        """
        Get latest GloTEC map if fresh enough.

        Args:
            max_age_seconds: Maximum acceptable age in seconds

        Returns:
            GloTEC map dictionary or None
        """
        with self.glotec_lock:
            if self.latest_glotec_map is None or self.latest_glotec_time is None:
                return None

            age = (datetime.utcnow() - self.latest_glotec_time).total_seconds()
            if age > max_age_seconds:
                return None

            return self.latest_glotec_map

    def get_glotec_statistics(self) -> Optional[Dict[str, Any]]:
        """Get statistics from latest GloTEC map without copying large arrays."""
        with self.glotec_lock:
            if self.latest_glotec_map is None:
                return None

            return {
                'timestamp': self.latest_glotec_map.get('timestamp'),
                'statistics': self.latest_glotec_map.get('statistics', {}),
                'metadata': self.latest_glotec_map.get('metadata', {}),
                'age_seconds': (
                    datetime.utcnow() - self.latest_glotec_time
                ).total_seconds() if self.latest_glotec_time else None
            }

    def get_glotec_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get GloTEC statistics history for specified time window.

        Args:
            hours: Number of hours to include

        Returns:
            List of GloTEC statistics entries
        """
        cutoff = datetime.utcnow() - timedelta(hours=hours)

        with self.glotec_lock:
            return [
                entry for entry in self.glotec_history
                if datetime.fromisoformat(entry['timestamp'].rstrip('Z')) > cutoff
            ]

    def get_glotec_point(self, lat: float, lon: float) -> Optional[Dict[str, float]]:
        """
        Get TEC value at a specific lat/lon from latest GloTEC map.

        Uses nearest neighbor interpolation.

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees

        Returns:
            Dictionary with tec, anomaly, hmF2, NmF2 at point, or None
        """
        with self.glotec_lock:
            if self.latest_glotec_map is None:
                return None

            grid = self.latest_glotec_map.get('grid', {})
            lat_grid = grid.get('lat', [])
            lon_grid = grid.get('lon', [])
            tec_grid = grid.get('tec', [])

            if not lat_grid or not lon_grid or not tec_grid:
                return None

            # Find nearest indices
            lat_idx = self._find_nearest_idx(lat_grid, lat)
            lon_idx = self._find_nearest_idx(lon_grid, lon)

            if lat_idx is None or lon_idx is None:
                return None

            try:
                return {
                    'lat': lat_grid[lat_idx],
                    'lon': lon_grid[lon_idx],
                    'tec': tec_grid[lat_idx][lon_idx] if isinstance(tec_grid[0], list) else tec_grid[lat_idx * len(lon_grid) + lon_idx],
                    'anomaly': grid.get('anomaly', [[]])[lat_idx][lon_idx] if grid.get('anomaly') and isinstance(grid['anomaly'][0], list) else None,
                    'hmF2': grid.get('hmF2', [[]])[lat_idx][lon_idx] if grid.get('hmF2') and isinstance(grid['hmF2'][0], list) else None,
                    'NmF2': grid.get('NmF2', [[]])[lat_idx][lon_idx] if grid.get('NmF2') and isinstance(grid['NmF2'][0], list) else None,
                    'timestamp': self.latest_glotec_map.get('timestamp')
                }
            except (IndexError, TypeError):
                return None

    def _find_nearest_idx(self, grid: List[float], value: float) -> Optional[int]:
        """Find index of nearest value in sorted grid."""
        if not grid:
            return None

        # Binary search for nearest
        import bisect
        idx = bisect.bisect_left(grid, value)

        if idx == 0:
            return 0
        if idx == len(grid):
            return len(grid) - 1

        # Compare neighbors
        if abs(grid[idx] - value) < abs(grid[idx - 1] - value):
            return idx
        return idx - 1

    # --- System Health Methods ---

    def update_service_status(self, service_name: str, status_data: Dict[str, Any]):
        """Update service status."""
        with self.health_lock:
            self.service_status[service_name] = {
                **status_data,
                'last_update': datetime.utcnow().isoformat()
            }

    def update_filter_metrics(self, metrics: Dict[str, Any]):
        """Update filter performance metrics."""
        with self.health_lock:
            self.filter_metrics = metrics

    def add_alert(self, alert_data: Dict[str, Any]):
        """Add system alert."""
        with self.health_lock:
            alert_entry = {
                **alert_data,
                'timestamp': datetime.utcnow().isoformat()
            }
            self.alert_history.append(alert_entry)
            self.stats['alerts_generated'] += 1

    def get_service_status(self) -> Dict[str, Dict[str, Any]]:
        """Get all service statuses."""
        with self.health_lock:
            return self.service_status.copy()

    def get_recent_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent alerts."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        with self.health_lock:
            return [
                entry for entry in self.alert_history
                if datetime.fromisoformat(entry['timestamp']) > cutoff
            ]

    # --- Utility Methods ---

    def _cleanup_old_data(self, data_deque: Deque):
        """Remove entries older than retention period from deque."""
        if len(data_deque) == 0:
            return

        cutoff = datetime.utcnow() - timedelta(seconds=self.retention_seconds)

        # Remove from left while old
        while len(data_deque) > 0:
            entry = data_deque[0]
            timestamp_str = entry.get('timestamp', '')
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(timestamp_str.rstrip('Z'))
                    if timestamp < cutoff:
                        data_deque.popleft()
                    else:
                        break
                except ValueError:
                    break
            else:
                break

    def get_statistics(self) -> Dict[str, Any]:
        """Get dashboard statistics."""
        return {
            **self.stats,
            'grid_age_seconds': (
                (datetime.utcnow() - self.latest_grid_time).total_seconds()
                if self.latest_grid_time else None
            ),
            'current_mode': self.current_mode,
            'observation_counts': self.get_observation_counts(),
            'active_services': len(self.service_status)
        }
