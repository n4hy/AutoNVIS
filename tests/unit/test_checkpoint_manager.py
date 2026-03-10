"""
Unit Tests for Checkpoint Manager

Tests HDF5 checkpoint save/load functionality for filter state persistence.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
import h5py

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from assimilation.python.checkpoint_manager import (
    CheckpointManager,
    CHECKPOINT_VERSION,
    restore_filter_from_checkpoint
)


class MockStateVector:
    """Mock StateVector for testing"""
    def __init__(self, n_lat, n_lon, n_alt):
        self.n_lat = n_lat
        self.n_lon = n_lon
        self.n_alt = n_alt
        self._data = np.random.rand(n_lat * n_lon * n_alt) * 1e11
        self._reff = 100.0

    def to_numpy(self):
        return self._data

    def from_numpy(self, arr):
        self._data = arr

    def get_reff(self):
        return self._reff

    def set_reff(self, val):
        self._reff = val

    def get_ne(self, i, j, k):
        idx = i * self.n_lon * self.n_alt + j * self.n_alt + k
        return self._data[idx]


class MockFilter:
    """Mock C++ filter for testing"""
    def __init__(self, n_lat, n_lon, n_alt):
        self.state = MockStateVector(n_lat, n_lon, n_alt)
        self.sqrt_cov = np.eye(n_lat * n_lon * n_alt) * 0.1

    def get_state(self):
        return self.state

    def get_sqrt_cov(self):
        return self.sqrt_cov

    def initialize(self, state, sqrt_cov):
        self.state = state
        self.sqrt_cov = sqrt_cov


class MockAutoNVISFilter:
    """Mock AutoNVISFilter for testing"""
    def __init__(self, n_lat=5, n_lon=5, n_alt=5):
        self.n_lat = n_lat
        self.n_lon = n_lon
        self.n_alt = n_alt

        self.lat_grid = np.linspace(-90, 90, n_lat)
        self.lon_grid = np.linspace(-180, 180, n_lon)
        self.alt_grid = np.linspace(100, 500, n_alt)

        self.filter = MockFilter(n_lat, n_lon, n_alt)

        self.state_history = []
        self.sqrt_cov_history = []

        self.cycle_count = 42
        self.smoother_activation_count = 5
        self.last_update_time = datetime.utcnow()
        self.uncertainty_threshold = 1e12
        self.localization_radius_km = 500.0
        self.max_history_length = 3

        # Mock mode
        class MockMode:
            value = "QUIET"
        self.current_mode = MockMode()

    def get_state_grid(self):
        """Return 3D electron density grid"""
        ne_flat = self.filter.state.to_numpy()
        return ne_flat.reshape((self.n_lat, self.n_lon, self.n_alt), order='C')

    def get_effective_ssn(self):
        return self.filter.state.get_reff()

    def get_statistics(self):
        return {
            'cycle_count': self.cycle_count,
            'smoother_activation_count': self.smoother_activation_count,
            'avg_nis': 1.05,
            'inflation_factor': 1.02,
            'current_mode': self.current_mode.value
        }


@pytest.fixture
def temp_checkpoint_dir():
    """Create temporary directory for checkpoints"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_filter():
    """Create mock filter for testing"""
    return MockAutoNVISFilter(n_lat=5, n_lon=5, n_alt=5)


@pytest.fixture
def checkpoint_manager(temp_checkpoint_dir):
    """Create checkpoint manager with temp directory"""
    return CheckpointManager(
        checkpoint_dir=temp_checkpoint_dir,
        max_checkpoints=5,
        compression="gzip",
        compression_level=4
    )


class TestCheckpointManagerBasics:
    """Test basic checkpoint manager functionality"""

    def test_initialization(self, temp_checkpoint_dir):
        """Test checkpoint manager initialization"""
        manager = CheckpointManager(temp_checkpoint_dir)

        assert manager.checkpoint_dir == temp_checkpoint_dir
        assert manager.max_checkpoints == 48  # default
        assert manager.compression == "gzip"
        assert temp_checkpoint_dir.exists()

    def test_initialization_creates_directory(self):
        """Test that initialization creates checkpoint directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = Path(tmpdir) / "checkpoints" / "nested"
            manager = CheckpointManager(new_dir)
            assert new_dir.exists()

    def test_list_checkpoints_empty(self, checkpoint_manager):
        """Test listing checkpoints when none exist"""
        checkpoints = checkpoint_manager.list_checkpoints()
        assert checkpoints == []

    def test_get_latest_checkpoint_empty(self, checkpoint_manager):
        """Test getting latest checkpoint when none exist"""
        latest = checkpoint_manager.get_latest_checkpoint()
        assert latest is None


class TestCheckpointSaveLoad:
    """Test checkpoint save and load functionality"""

    def test_save_checkpoint(self, checkpoint_manager, mock_filter):
        """Test saving a checkpoint"""
        timestamp = datetime(2026, 3, 10, 12, 0, 0)
        checkpoint_path = checkpoint_manager.save_checkpoint(mock_filter, timestamp)

        assert checkpoint_path.exists()
        assert checkpoint_path.suffix == '.h5'
        assert 'checkpoint_20260310_120000' in checkpoint_path.name

    def test_save_checkpoint_creates_valid_hdf5(self, checkpoint_manager, mock_filter):
        """Test that saved checkpoint is valid HDF5"""
        checkpoint_path = checkpoint_manager.save_checkpoint(mock_filter)

        with h5py.File(checkpoint_path, 'r') as f:
            # Check required groups exist
            assert 'state' in f
            assert 'covariance' in f
            assert 'grid' in f
            assert 'metadata' in f

            # Check required datasets
            assert 'ne_grid' in f['state']
            assert 'sqrt_cov' in f['covariance']
            assert 'lat' in f['grid']
            assert 'lon' in f['grid']
            assert 'alt' in f['grid']

    def test_load_checkpoint(self, checkpoint_manager, mock_filter):
        """Test loading a checkpoint"""
        checkpoint_path = checkpoint_manager.save_checkpoint(mock_filter)
        loaded = checkpoint_manager.load_checkpoint(checkpoint_path)

        assert 'ne_grid' in loaded
        assert 'sqrt_cov' in loaded
        assert 'lat_grid' in loaded
        assert 'lon_grid' in loaded
        assert 'alt_grid' in loaded
        assert 'mode' in loaded
        assert 'cycle_count' in loaded

    def test_save_load_roundtrip(self, checkpoint_manager, mock_filter):
        """Test that save/load preserves data"""
        original_ne_grid = mock_filter.get_state_grid().copy()
        original_reff = mock_filter.get_effective_ssn()
        original_cycle_count = mock_filter.cycle_count

        checkpoint_path = checkpoint_manager.save_checkpoint(mock_filter)
        loaded = checkpoint_manager.load_checkpoint(checkpoint_path)

        np.testing.assert_array_almost_equal(loaded['ne_grid'], original_ne_grid)
        assert loaded['reff'] == pytest.approx(original_reff)
        assert loaded['cycle_count'] == original_cycle_count

    def test_save_with_smoother_history(self, checkpoint_manager, mock_filter):
        """Test saving checkpoint with smoother history"""
        # Add some history
        mock_filter.state_history = [np.random.rand(125) for _ in range(3)]
        mock_filter.sqrt_cov_history = [np.eye(125) * 0.1 for _ in range(3)]

        checkpoint_path = checkpoint_manager.save_checkpoint(mock_filter, include_history=True)
        loaded = checkpoint_manager.load_checkpoint(checkpoint_path)

        assert len(loaded['state_history']) == 3
        assert len(loaded['sqrt_cov_history']) == 3

    def test_save_without_smoother_history(self, checkpoint_manager, mock_filter):
        """Test saving checkpoint without smoother history"""
        mock_filter.state_history = [np.random.rand(125) for _ in range(3)]

        checkpoint_path = checkpoint_manager.save_checkpoint(mock_filter, include_history=False)
        loaded = checkpoint_manager.load_checkpoint(checkpoint_path)

        assert len(loaded['state_history']) == 0


class TestCheckpointManagement:
    """Test checkpoint management features"""

    def test_list_checkpoints_sorted(self, checkpoint_manager, mock_filter):
        """Test that checkpoints are listed newest first"""
        import time

        # Create multiple checkpoints with slight delays
        paths = []
        for i in range(3):
            path = checkpoint_manager.save_checkpoint(
                mock_filter,
                timestamp=datetime(2026, 3, 10, 12, i, 0)
            )
            paths.append(path)
            time.sleep(0.01)  # Ensure different mtime

        checkpoints = checkpoint_manager.list_checkpoints()

        # Should be sorted newest first (reverse order of creation)
        assert len(checkpoints) == 3
        assert checkpoints[0] == paths[-1]

    def test_cleanup_old_checkpoints(self, temp_checkpoint_dir, mock_filter):
        """Test automatic cleanup of old checkpoints"""
        manager = CheckpointManager(temp_checkpoint_dir, max_checkpoints=3)

        # Create 5 checkpoints
        for i in range(5):
            manager.save_checkpoint(
                mock_filter,
                timestamp=datetime(2026, 3, 10, 12, i, 0)
            )

        checkpoints = manager.list_checkpoints()

        # Should only have 3 checkpoints (2 were cleaned up)
        assert len(checkpoints) == 3

    def test_get_latest_checkpoint(self, checkpoint_manager, mock_filter):
        """Test getting latest checkpoint"""
        for i in range(3):
            checkpoint_manager.save_checkpoint(
                mock_filter,
                timestamp=datetime(2026, 3, 10, 12, i, 0)
            )

        latest = checkpoint_manager.get_latest_checkpoint()

        assert latest is not None
        assert '120200' in latest.name  # Latest timestamp

    def test_get_checkpoint_info(self, checkpoint_manager, mock_filter):
        """Test getting checkpoint metadata"""
        mock_filter.cycle_count = 100
        checkpoint_path = checkpoint_manager.save_checkpoint(mock_filter)

        info = checkpoint_manager.get_checkpoint_info(checkpoint_path)

        assert info['version'] == CHECKPOINT_VERSION
        assert info['cycle_count'] == 100
        assert info['grid_shape'] == (5, 5, 5)
        assert info['mode'] == 'QUIET'

    def test_verify_checkpoint_valid(self, checkpoint_manager, mock_filter):
        """Test verification of valid checkpoint"""
        checkpoint_path = checkpoint_manager.save_checkpoint(mock_filter)
        assert checkpoint_manager.verify_checkpoint(checkpoint_path) is True

    def test_verify_checkpoint_missing(self, checkpoint_manager):
        """Test verification of missing checkpoint"""
        assert checkpoint_manager.verify_checkpoint(Path("/nonexistent.h5")) is False


class TestCheckpointCompression:
    """Test checkpoint compression"""

    def test_gzip_compression(self, temp_checkpoint_dir, mock_filter):
        """Test checkpoint with gzip compression"""
        manager = CheckpointManager(
            temp_checkpoint_dir,
            compression="gzip",
            compression_level=9
        )
        checkpoint_path = manager.save_checkpoint(mock_filter)

        assert checkpoint_path.exists()

        # Verify compression was applied
        with h5py.File(checkpoint_path, 'r') as f:
            assert f['state/ne_grid'].compression == 'gzip'

    def test_no_compression(self, temp_checkpoint_dir, mock_filter):
        """Test checkpoint without compression"""
        manager = CheckpointManager(
            temp_checkpoint_dir,
            compression=None
        )
        checkpoint_path = manager.save_checkpoint(mock_filter)

        assert checkpoint_path.exists()


class TestErrorHandling:
    """Test error handling"""

    def test_load_nonexistent_checkpoint(self, checkpoint_manager):
        """Test loading nonexistent checkpoint raises error"""
        with pytest.raises(FileNotFoundError):
            checkpoint_manager.load_checkpoint(Path("/nonexistent.h5"))

    def test_atomic_write_failure_cleanup(self, checkpoint_manager, mock_filter):
        """Test that temp file is cleaned up on failure"""
        with patch('h5py.File', side_effect=IOError("Disk full")):
            with pytest.raises(IOError):
                checkpoint_manager.save_checkpoint(mock_filter)

        # No temp files should be left
        temp_files = list(checkpoint_manager.checkpoint_dir.glob("*.tmp"))
        assert len(temp_files) == 0


class TestFilterRestoration:
    """Test filter restoration from checkpoint"""

    def test_restore_filter_basic(self, checkpoint_manager, mock_filter):
        """Test basic filter restoration"""
        original_cycle = mock_filter.cycle_count
        checkpoint_path = checkpoint_manager.save_checkpoint(mock_filter)
        loaded = checkpoint_manager.load_checkpoint(checkpoint_path)

        # Create new filter and restore
        new_filter = MockAutoNVISFilter(n_lat=5, n_lon=5, n_alt=5)
        new_filter.cycle_count = 0

        # Manually verify restoration would work
        assert loaded['cycle_count'] == original_cycle
        assert loaded['ne_grid'].shape == (5, 5, 5)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
