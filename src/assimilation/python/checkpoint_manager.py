"""
Checkpoint Manager for Auto-NVIS Filter State Persistence

Provides HDF5-based checkpoint save/load functionality for the SR-UKF filter,
enabling system restart without loss of assimilation state.

HDF5 Schema:
/checkpoint_YYYYMMDD_HHMMSS.h5
    /state/
        ne_grid         [n_lat, n_lon, n_alt] - Electron density grid
        reff            scalar - Effective sunspot number
        timestamp       string - ISO format timestamp
        cycle_count     int - Filter cycle count
    /covariance/
        sqrt_cov        [n_state, rank] - Square-root covariance (gzip compressed)
        rank            int - Rank of sqrt covariance
    /grid/
        lat             [n_lat] - Latitude coordinates
        lon             [n_lon] - Longitude coordinates
        alt             [n_alt] - Altitude coordinates
    /history/
        states          [n_history, n_state] - State history for smoother
        sqrt_covs       [n_history, n_state, rank] - Covariance history
        timestamps      [n_history] - Timestamps for history entries
    /metadata/
        mode            string - Operational mode (QUIET/SHOCK)
        version         string - Checkpoint format version
        created         string - Creation timestamp
        filter_config   group - Filter configuration parameters
        statistics      group - Filter statistics at checkpoint time
"""

import h5py
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import logging
import json
from dataclasses import asdict

logger = logging.getLogger(__name__)

CHECKPOINT_VERSION = "1.0"


class CheckpointManager:
    """
    Manages HDF5 checkpoint save/load for Auto-NVIS filter state.

    Features:
    - Compressed HDF5 storage for efficient disk usage
    - Automatic cleanup of old checkpoints
    - Atomic writes with temp file + rename
    - Smoother history preservation
    - Filter configuration storage
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        max_checkpoints: int = 48,
        compression: str = "gzip",
        compression_level: int = 4,
        prefix: str = "checkpoint"
    ):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory for checkpoint files
            max_checkpoints: Maximum checkpoints to retain
            compression: Compression type ("gzip", "lzf", or None)
            compression_level: Compression level (1-9 for gzip)
            prefix: Checkpoint filename prefix
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.compression = compression
        self.compression_opts = compression_level if compression == "gzip" else None
        self.prefix = prefix

        # Ensure checkpoint directory exists
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"CheckpointManager initialized: {self.checkpoint_dir}")

    def save_checkpoint(
        self,
        filter_state: 'AutoNVISFilter',
        timestamp: Optional[datetime] = None,
        include_history: bool = True
    ) -> Path:
        """
        Save filter state to checkpoint file.

        Args:
            filter_state: AutoNVISFilter instance to checkpoint
            timestamp: Checkpoint timestamp (defaults to now)
            include_history: Include smoother history in checkpoint

        Returns:
            Path to saved checkpoint file
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        # Generate checkpoint filename
        ts_str = timestamp.strftime("%Y%m%d_%H%M%S")
        checkpoint_path = self.checkpoint_dir / f"{self.prefix}_{ts_str}.h5"
        temp_path = checkpoint_path.with_suffix('.h5.tmp')

        logger.info(f"Saving checkpoint: {checkpoint_path}")

        try:
            with h5py.File(temp_path, 'w') as f:
                # Create groups
                state_grp = f.create_group('state')
                cov_grp = f.create_group('covariance')
                grid_grp = f.create_group('grid')
                history_grp = f.create_group('history')
                meta_grp = f.create_group('metadata')

                # Save state
                ne_grid = filter_state.get_state_grid()
                state_grp.create_dataset(
                    'ne_grid', data=ne_grid,
                    compression=self.compression,
                    compression_opts=self.compression_opts
                )
                state_grp.create_dataset('reff', data=filter_state.get_effective_ssn())
                state_grp.attrs['timestamp'] = timestamp.isoformat()
                state_grp.attrs['cycle_count'] = filter_state.cycle_count

                # Save covariance
                sqrt_cov = filter_state.filter.get_sqrt_cov()
                cov_grp.create_dataset(
                    'sqrt_cov', data=sqrt_cov,
                    compression=self.compression,
                    compression_opts=self.compression_opts
                )
                cov_grp.attrs['rank'] = sqrt_cov.shape[1] if sqrt_cov.ndim > 1 else sqrt_cov.shape[0]

                # Save grid coordinates
                grid_grp.create_dataset('lat', data=filter_state.lat_grid)
                grid_grp.create_dataset('lon', data=filter_state.lon_grid)
                grid_grp.create_dataset('alt', data=filter_state.alt_grid)

                # Save smoother history if available
                if include_history and filter_state.state_history:
                    history_grp.create_dataset(
                        'states',
                        data=np.array(filter_state.state_history),
                        compression=self.compression,
                        compression_opts=self.compression_opts
                    )
                    if filter_state.sqrt_cov_history:
                        history_grp.create_dataset(
                            'sqrt_covs',
                            data=np.array(filter_state.sqrt_cov_history),
                            compression=self.compression,
                            compression_opts=self.compression_opts
                        )
                    history_grp.attrs['length'] = len(filter_state.state_history)
                else:
                    history_grp.attrs['length'] = 0

                # Save metadata
                meta_grp.attrs['mode'] = filter_state.current_mode.value
                meta_grp.attrs['version'] = CHECKPOINT_VERSION
                meta_grp.attrs['created'] = datetime.utcnow().isoformat()

                # Save filter statistics
                stats = filter_state.get_statistics()
                stats_grp = meta_grp.create_group('statistics')
                for key, value in stats.items():
                    if value is not None:
                        if isinstance(value, (int, float, bool)):
                            stats_grp.attrs[key] = value
                        elif isinstance(value, str):
                            stats_grp.attrs[key] = value

                # Save filter configuration
                config_grp = meta_grp.create_group('filter_config')
                config_grp.attrs['n_lat'] = filter_state.n_lat
                config_grp.attrs['n_lon'] = filter_state.n_lon
                config_grp.attrs['n_alt'] = filter_state.n_alt
                config_grp.attrs['uncertainty_threshold'] = filter_state.uncertainty_threshold
                config_grp.attrs['localization_radius_km'] = filter_state.localization_radius_km
                config_grp.attrs['max_history_length'] = filter_state.max_history_length

            # Atomic rename
            temp_path.rename(checkpoint_path)

            logger.info(f"Checkpoint saved: {checkpoint_path} ({checkpoint_path.stat().st_size / 1024:.1f} KB)")

            # Cleanup old checkpoints
            self._cleanup_old_checkpoints()

            return checkpoint_path

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}", exc_info=True)
            # Clean up temp file if it exists
            if temp_path.exists():
                temp_path.unlink()
            raise

    def load_checkpoint(self, filepath: Path) -> Dict[str, Any]:
        """
        Load filter state from checkpoint file.

        Args:
            filepath: Path to checkpoint file

        Returns:
            Dictionary with checkpoint data:
                - ne_grid: Electron density grid
                - reff: Effective sunspot number
                - sqrt_cov: Square-root covariance
                - lat_grid, lon_grid, alt_grid: Grid coordinates
                - state_history: Smoother state history (if available)
                - sqrt_cov_history: Smoother covariance history (if available)
                - mode: Operational mode
                - cycle_count: Filter cycle count
                - timestamp: Checkpoint timestamp
                - statistics: Filter statistics
                - config: Filter configuration
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")

        logger.info(f"Loading checkpoint: {filepath}")

        try:
            with h5py.File(filepath, 'r') as f:
                # Verify version
                version = f['metadata'].attrs.get('version', '0.0')
                if version != CHECKPOINT_VERSION:
                    logger.warning(f"Checkpoint version mismatch: {version} vs {CHECKPOINT_VERSION}")

                # Load state
                ne_grid = f['state/ne_grid'][:]
                reff = float(f['state/reff'][()])
                timestamp = f['state'].attrs['timestamp']
                cycle_count = int(f['state'].attrs['cycle_count'])

                # Load covariance
                sqrt_cov = f['covariance/sqrt_cov'][:]

                # Load grid coordinates
                lat_grid = f['grid/lat'][:]
                lon_grid = f['grid/lon'][:]
                alt_grid = f['grid/alt'][:]

                # Load smoother history
                state_history = []
                sqrt_cov_history = []
                history_length = f['history'].attrs.get('length', 0)

                if history_length > 0:
                    if 'states' in f['history']:
                        state_history = list(f['history/states'][:])
                    if 'sqrt_covs' in f['history']:
                        sqrt_cov_history = list(f['history/sqrt_covs'][:])

                # Load metadata
                mode = f['metadata'].attrs.get('mode', 'QUIET')

                # Load statistics
                statistics = {}
                if 'statistics' in f['metadata']:
                    for key in f['metadata/statistics'].attrs:
                        statistics[key] = f['metadata/statistics'].attrs[key]

                # Load config
                config = {}
                if 'filter_config' in f['metadata']:
                    for key in f['metadata/filter_config'].attrs:
                        config[key] = f['metadata/filter_config'].attrs[key]

            result = {
                'ne_grid': ne_grid,
                'reff': reff,
                'sqrt_cov': sqrt_cov,
                'lat_grid': lat_grid,
                'lon_grid': lon_grid,
                'alt_grid': alt_grid,
                'state_history': state_history,
                'sqrt_cov_history': sqrt_cov_history,
                'mode': mode,
                'cycle_count': cycle_count,
                'timestamp': timestamp,
                'statistics': statistics,
                'config': config
            }

            logger.info(f"Checkpoint loaded: {filepath} (cycle {cycle_count}, mode {mode})")
            return result

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}", exc_info=True)
            raise

    def get_latest_checkpoint(self) -> Optional[Path]:
        """
        Get the most recent checkpoint file.

        Returns:
            Path to latest checkpoint or None if no checkpoints exist
        """
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return None
        return checkpoints[0]  # List is sorted newest first

    def list_checkpoints(self) -> List[Path]:
        """
        List all checkpoint files, newest first.

        Returns:
            List of checkpoint paths, sorted by modification time (newest first)
        """
        pattern = f"{self.prefix}_*.h5"
        checkpoints = list(self.checkpoint_dir.glob(pattern))
        checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return checkpoints

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints exceeding max_checkpoints limit."""
        checkpoints = self.list_checkpoints()

        if len(checkpoints) > self.max_checkpoints:
            to_remove = checkpoints[self.max_checkpoints:]
            for checkpoint in to_remove:
                try:
                    checkpoint.unlink()
                    logger.debug(f"Removed old checkpoint: {checkpoint}")
                except Exception as e:
                    logger.warning(f"Failed to remove checkpoint {checkpoint}: {e}")

            logger.info(f"Cleaned up {len(to_remove)} old checkpoints")

    def get_checkpoint_info(self, filepath: Path) -> Dict[str, Any]:
        """
        Get metadata about a checkpoint without loading full state.

        Args:
            filepath: Path to checkpoint file

        Returns:
            Dictionary with checkpoint metadata
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")

        with h5py.File(filepath, 'r') as f:
            info = {
                'path': str(filepath),
                'size_kb': filepath.stat().st_size / 1024,
                'version': f['metadata'].attrs.get('version', 'unknown'),
                'created': f['metadata'].attrs.get('created', 'unknown'),
                'mode': f['metadata'].attrs.get('mode', 'unknown'),
                'timestamp': f['state'].attrs.get('timestamp', 'unknown'),
                'cycle_count': int(f['state'].attrs.get('cycle_count', 0)),
                'grid_shape': (
                    len(f['grid/lat']),
                    len(f['grid/lon']),
                    len(f['grid/alt'])
                ),
                'history_length': f['history'].attrs.get('length', 0)
            }

        return info

    def verify_checkpoint(self, filepath: Path) -> bool:
        """
        Verify checkpoint file integrity.

        Args:
            filepath: Path to checkpoint file

        Returns:
            True if checkpoint is valid, False otherwise
        """
        try:
            filepath = Path(filepath)
            if not filepath.exists():
                return False

            with h5py.File(filepath, 'r') as f:
                # Check required groups exist
                required_groups = ['state', 'covariance', 'grid', 'metadata']
                for group in required_groups:
                    if group not in f:
                        logger.warning(f"Missing group in checkpoint: {group}")
                        return False

                # Check required datasets
                if 'ne_grid' not in f['state']:
                    logger.warning("Missing ne_grid in checkpoint")
                    return False

                if 'sqrt_cov' not in f['covariance']:
                    logger.warning("Missing sqrt_cov in checkpoint")
                    return False

                # Check data integrity
                ne_grid = f['state/ne_grid'][:]
                if np.any(np.isnan(ne_grid)) or np.any(np.isinf(ne_grid)):
                    logger.warning("Invalid values in ne_grid")
                    return False

            return True

        except Exception as e:
            logger.warning(f"Checkpoint verification failed: {e}")
            return False


def restore_filter_from_checkpoint(
    filter_instance: 'AutoNVISFilter',
    checkpoint_data: Dict[str, Any]
) -> None:
    """
    Restore filter state from checkpoint data.

    Args:
        filter_instance: AutoNVISFilter instance to restore
        checkpoint_data: Data from CheckpointManager.load_checkpoint()
    """
    from .autonvis_filter import OperationalMode

    # Verify grid dimensions match
    config = checkpoint_data.get('config', {})
    if config:
        if (filter_instance.n_lat != config.get('n_lat') or
            filter_instance.n_lon != config.get('n_lon') or
            filter_instance.n_alt != config.get('n_alt')):
            raise ValueError(
                f"Grid dimensions mismatch: filter ({filter_instance.n_lat}, "
                f"{filter_instance.n_lon}, {filter_instance.n_alt}) vs "
                f"checkpoint ({config.get('n_lat')}, {config.get('n_lon')}, "
                f"{config.get('n_alt')})"
            )

    # Restore grid coordinates
    filter_instance.lat_grid = checkpoint_data['lat_grid']
    filter_instance.lon_grid = checkpoint_data['lon_grid']
    filter_instance.alt_grid = checkpoint_data['alt_grid']

    # Restore state to C++ filter
    ne_grid = checkpoint_data['ne_grid']
    sqrt_cov = checkpoint_data['sqrt_cov']

    # Flatten state for C++ StateVector
    state_flat = ne_grid.flatten('C')

    # Import C++ bindings
    import autonvis_srukf as srukf

    # Create state vector and set values
    state_vec = srukf.StateVector(
        filter_instance.n_lat,
        filter_instance.n_lon,
        filter_instance.n_alt
    )
    state_vec.from_numpy(state_flat)
    state_vec.set_reff(checkpoint_data['reff'])

    # Restore to filter
    filter_instance.filter.initialize(state_vec, sqrt_cov)

    # Restore smoother history
    filter_instance.state_history = list(checkpoint_data.get('state_history', []))
    filter_instance.sqrt_cov_history = list(checkpoint_data.get('sqrt_cov_history', []))

    # Restore mode
    mode_str = checkpoint_data.get('mode', 'QUIET')
    filter_instance.current_mode = OperationalMode(mode_str)

    # Restore cycle count
    filter_instance.cycle_count = checkpoint_data.get('cycle_count', 0)
    filter_instance.smoother_activation_count = checkpoint_data.get(
        'statistics', {}
    ).get('smoother_activation_count', 0)

    # Parse and restore timestamp
    timestamp_str = checkpoint_data.get('timestamp')
    if timestamp_str:
        try:
            filter_instance.last_update_time = datetime.fromisoformat(timestamp_str)
        except ValueError:
            filter_instance.last_update_time = datetime.utcnow()

    logger.info(
        f"Filter restored from checkpoint: cycle {filter_instance.cycle_count}, "
        f"mode {filter_instance.current_mode.value}"
    )
