"""
Unit Tests for IONORT-Style Homing Algorithm

Tests the HomingAlgorithm class and related components:
- WinnerTriplet data structure
- HomingSearchSpace parameter handling
- Great circle geometry calculations
- Landing accuracy checks (Condition 10)
- MUF/LUF/FOT calculation

Note: Full integration tests require IonosphericModel and may be slow.
These unit tests focus on the algorithm logic.
"""

import pytest
import numpy as np
from typing import List
from unittest.mock import Mock, MagicMock, patch

from src.raytracer.homing_algorithm import (
    HomingAlgorithm,
    HomingResult,
    HomingSearchSpace,
    HomingConfig,
    WinnerTriplet,
    PropagationMode,
)


class TestWinnerTriplet:
    """Tests for WinnerTriplet data structure."""

    def test_creation(self):
        """Test WinnerTriplet creation."""
        triplet = WinnerTriplet(
            frequency_mhz=7.0,
            elevation_deg=45.0,
            azimuth_deg=90.0,
            azimuth_deviation_deg=5.0,
            group_delay_ms=3.5,
            ground_range_km=500.0,
            landing_lat=42.0,
            landing_lon=-100.0,
            landing_error_km=15.0,
            mode=PropagationMode.O_MODE,
        )

        assert triplet.frequency_mhz == 7.0
        assert triplet.elevation_deg == 45.0
        assert triplet.mode == PropagationMode.O_MODE

    def test_repr(self):
        """Test string representation."""
        triplet = WinnerTriplet(
            frequency_mhz=10.5,
            elevation_deg=60.0,
            azimuth_deg=45.0,
            azimuth_deviation_deg=0.0,
            group_delay_ms=4.0,
            ground_range_km=750.0,
            landing_lat=41.0,
            landing_lon=-99.0,
            landing_error_km=8.0,
            mode=PropagationMode.X_MODE,
        )

        repr_str = repr(triplet)
        assert '10.5MHz' in repr_str
        assert '60.0°' in repr_str
        assert 'X' in repr_str


class TestHomingSearchSpace:
    """Tests for HomingSearchSpace configuration."""

    def test_default_values(self):
        """Test default search space parameters."""
        search = HomingSearchSpace()

        assert search.freq_range == (2.0, 30.0)
        assert search.freq_step == 0.5
        assert search.elevation_range == (5.0, 89.0)
        assert search.elevation_step == 2.0

    def test_num_calculations(self):
        """Test count calculations."""
        search = HomingSearchSpace(
            freq_range=(5.0, 10.0),
            freq_step=1.0,
            elevation_range=(30.0, 60.0),
            elevation_step=10.0,
            azimuth_deviation_range=(-10.0, 10.0),
            azimuth_step=5.0,
        )

        assert search.num_frequencies == 6  # 5, 6, 7, 8, 9, 10
        assert search.num_elevations == 4   # 30, 40, 50, 60
        assert search.num_azimuths == 5     # -10, -5, 0, 5, 10

    def test_total_triplets(self):
        """Test total triplet count."""
        search = HomingSearchSpace(
            freq_range=(5.0, 10.0),
            freq_step=1.0,
            elevation_range=(30.0, 60.0),
            elevation_step=10.0,
            azimuth_deviation_range=(-5.0, 5.0),
            azimuth_step=5.0,
        )

        # 6 freqs * 4 elevations * 3 azimuths = 72
        assert search.total_triplets == 72

    def test_array_generation(self):
        """Test frequency/elevation/azimuth array generation."""
        search = HomingSearchSpace(
            freq_range=(3.0, 5.0),
            freq_step=1.0,
            elevation_range=(10.0, 20.0),
            elevation_step=5.0,
            azimuth_deviation_range=(-5.0, 5.0),
            azimuth_step=5.0,
        )

        freqs = search.frequencies()
        elevations = search.elevations()
        azimuths = search.azimuth_deviations()

        np.testing.assert_array_almost_equal(freqs, [3.0, 4.0, 5.0])
        np.testing.assert_array_almost_equal(elevations, [10.0, 15.0, 20.0])
        np.testing.assert_array_almost_equal(azimuths, [-5.0, 0.0, 5.0])


class TestHomingConfig:
    """Tests for HomingConfig settings."""

    def test_default_config(self):
        """Test default configuration values."""
        config = HomingConfig()

        assert config.lat_tolerance_deg == 1.0
        assert config.lon_tolerance_deg == 1.0
        assert config.distance_tolerance_km == 100.0
        assert config.use_distance_tolerance is True
        assert config.trace_both_modes is True
        assert config.max_workers == 4

    def test_custom_config(self):
        """Test custom configuration."""
        config = HomingConfig(
            distance_tolerance_km=50.0,
            trace_both_modes=False,
            max_workers=8,
        )

        assert config.distance_tolerance_km == 50.0
        assert config.trace_both_modes is False
        assert config.max_workers == 8


class TestHomingResult:
    """Tests for HomingResult data structure."""

    def test_empty_result(self):
        """Test empty result properties."""
        result = HomingResult(
            tx_position=(40.0, -105.0, 0.0),
            rx_position=(42.0, -100.0, 0.0),
            great_circle_range_km=500.0,
            great_circle_azimuth_deg=45.0,
        )

        assert result.num_winners == 0
        assert len(result.o_mode_winners) == 0
        assert len(result.x_mode_winners) == 0
        assert result.muf == 0.0
        assert result.luf == 0.0

    def test_with_winners(self):
        """Test result with winner triplets."""
        o_winner = WinnerTriplet(
            frequency_mhz=8.0, elevation_deg=50.0, azimuth_deg=45.0,
            azimuth_deviation_deg=0.0, group_delay_ms=3.0, ground_range_km=500.0,
            landing_lat=42.0, landing_lon=-100.0, landing_error_km=10.0,
            mode=PropagationMode.O_MODE,
        )
        x_winner = WinnerTriplet(
            frequency_mhz=7.5, elevation_deg=55.0, azimuth_deg=45.0,
            azimuth_deviation_deg=0.0, group_delay_ms=3.2, ground_range_km=500.0,
            landing_lat=42.0, landing_lon=-100.0, landing_error_km=12.0,
            mode=PropagationMode.X_MODE,
        )

        result = HomingResult(
            tx_position=(40.0, -105.0, 0.0),
            rx_position=(42.0, -100.0, 0.0),
            great_circle_range_km=500.0,
            great_circle_azimuth_deg=45.0,
            winner_triplets=[o_winner, x_winner],
            muf=8.0,
            luf=7.5,
            fot=6.8,
        )

        assert result.num_winners == 2
        assert len(result.o_mode_winners) == 1
        assert len(result.x_mode_winners) == 1
        assert result.o_mode_winners[0].frequency_mhz == 8.0
        assert result.x_mode_winners[0].frequency_mhz == 7.5

    def test_frequencies_by_mode(self):
        """Test frequency extraction by mode."""
        winners = [
            WinnerTriplet(
                frequency_mhz=f, elevation_deg=50.0, azimuth_deg=45.0,
                azimuth_deviation_deg=0.0, group_delay_ms=3.0, ground_range_km=500.0,
                landing_lat=42.0, landing_lon=-100.0, landing_error_km=10.0,
                mode=PropagationMode.O_MODE if i % 2 == 0 else PropagationMode.X_MODE,
            )
            for i, f in enumerate([5.0, 6.0, 7.0, 8.0])
        ]

        result = HomingResult(
            tx_position=(40.0, -105.0, 0.0),
            rx_position=(42.0, -100.0, 0.0),
            great_circle_range_km=500.0,
            great_circle_azimuth_deg=45.0,
            winner_triplets=winners,
        )

        o_freqs = result.frequencies_by_mode(PropagationMode.O_MODE)
        x_freqs = result.frequencies_by_mode(PropagationMode.X_MODE)

        assert o_freqs == [5.0, 7.0]
        assert x_freqs == [6.0, 8.0]


class TestHomingAlgorithmGeometry:
    """Tests for HomingAlgorithm geometry calculations."""

    @pytest.fixture
    def mock_solver(self):
        """Create mock HaselgroveSolver."""
        solver = Mock()
        solver.trace_ray = MagicMock()
        return solver

    @pytest.fixture
    def algorithm(self, mock_solver):
        """Create HomingAlgorithm with mock solver."""
        return HomingAlgorithm(mock_solver)

    def test_great_circle_geometry_north(self, algorithm):
        """Test great circle calculation heading north."""
        # From equator going north
        distance, azimuth = algorithm._great_circle_geometry(
            lat1=0.0, lon1=0.0,
            lat2=1.0, lon2=0.0
        )

        # 1 degree latitude ≈ 111 km
        assert abs(distance - 111.0) < 1.0
        assert abs(azimuth - 0.0) < 1.0  # Due north

    def test_great_circle_geometry_east(self, algorithm):
        """Test great circle calculation heading east."""
        # From equator going east
        distance, azimuth = algorithm._great_circle_geometry(
            lat1=0.0, lon1=0.0,
            lat2=0.0, lon2=1.0
        )

        # 1 degree longitude at equator ≈ 111 km
        assert abs(distance - 111.0) < 1.0
        assert abs(azimuth - 90.0) < 1.0  # Due east

    def test_great_circle_geometry_diagonal(self, algorithm):
        """Test great circle calculation for diagonal path."""
        # Boulder to Denver (roughly NNE)
        distance, azimuth = algorithm._great_circle_geometry(
            lat1=40.0, lon1=-105.3,
            lat2=39.7, lon2=-104.9
        )

        # Should be roughly 45-60 km, heading SE
        assert 30 < distance < 60
        assert 100 < azimuth < 180  # Heading roughly south-east

    def test_landing_accuracy_check_pass(self, algorithm):
        """Test landing accuracy check with acceptable error."""
        algorithm.config = HomingConfig(
            distance_tolerance_km=100.0,
            use_distance_tolerance=True
        )

        is_winner, error = algorithm._check_landing_accuracy(
            land_lat=42.0, land_lon=-100.0,
            rx_lat=42.1, rx_lon=-100.1
        )

        assert is_winner == True
        assert error < 100.0

    def test_landing_accuracy_check_fail(self, algorithm):
        """Test landing accuracy check with unacceptable error."""
        algorithm.config = HomingConfig(
            distance_tolerance_km=50.0,
            use_distance_tolerance=True
        )

        is_winner, error = algorithm._check_landing_accuracy(
            land_lat=42.0, land_lon=-100.0,
            rx_lat=43.0, rx_lon=-99.0  # Too far
        )

        assert is_winner == False
        assert error > 50.0

    def test_landing_accuracy_lat_lon_mode(self, algorithm):
        """Test landing accuracy with lat/lon tolerance mode."""
        algorithm.config = HomingConfig(
            lat_tolerance_deg=0.5,
            lon_tolerance_deg=0.5,
            use_distance_tolerance=False
        )

        # Within tolerance
        is_winner, _ = algorithm._check_landing_accuracy(
            land_lat=42.0, land_lon=-100.0,
            rx_lat=42.3, rx_lon=-100.2
        )
        assert is_winner is True

        # Outside tolerance
        is_winner, _ = algorithm._check_landing_accuracy(
            land_lat=42.0, land_lon=-100.0,
            rx_lat=43.0, rx_lon=-100.0
        )
        assert is_winner is False


class TestHomingAlgorithmTripletGeneration:
    """Tests for triplet generation."""

    @pytest.fixture
    def mock_solver(self):
        solver = Mock()
        return solver

    @pytest.fixture
    def algorithm(self, mock_solver):
        return HomingAlgorithm(mock_solver)

    def test_generate_triplets_count(self, algorithm):
        """Test correct number of triplets generated."""
        search = HomingSearchSpace(
            freq_range=(5.0, 7.0),
            freq_step=1.0,
            elevation_range=(30.0, 40.0),
            elevation_step=10.0,
            azimuth_deviation_range=(0.0, 0.0),
            azimuth_step=5.0,
        )
        algorithm.config = HomingConfig(trace_both_modes=True)

        triplets = algorithm._generate_triplets(search, gc_azimuth=45.0)

        # 3 freqs * 2 elevations * 1 azimuth * 2 modes = 12
        assert len(triplets) == 12

    def test_generate_triplets_modes(self, algorithm):
        """Test both modes are generated."""
        search = HomingSearchSpace(
            freq_range=(5.0, 5.0),
            freq_step=1.0,
            elevation_range=(30.0, 30.0),
            elevation_step=10.0,
            azimuth_deviation_range=(0.0, 0.0),
            azimuth_step=5.0,
        )
        algorithm.config = HomingConfig(trace_both_modes=True)

        triplets = algorithm._generate_triplets(search, gc_azimuth=45.0)

        modes = [t[3] for t in triplets]
        assert PropagationMode.O_MODE in modes
        assert PropagationMode.X_MODE in modes

    def test_generate_triplets_single_mode(self, algorithm):
        """Test single mode generation."""
        search = HomingSearchSpace(
            freq_range=(5.0, 5.0),
            freq_step=1.0,
            elevation_range=(30.0, 30.0),
            elevation_step=10.0,
            azimuth_deviation_range=(0.0, 0.0),
            azimuth_step=5.0,
        )
        algorithm.config = HomingConfig(trace_both_modes=False)

        triplets = algorithm._generate_triplets(search, gc_azimuth=45.0)

        # Only O-mode
        assert len(triplets) == 1
        assert triplets[0][3] == PropagationMode.O_MODE

    def test_azimuth_deviation_applied(self, algorithm):
        """Test azimuth deviations are applied to great circle azimuth."""
        search = HomingSearchSpace(
            freq_range=(5.0, 5.0),
            freq_step=1.0,
            elevation_range=(30.0, 30.0),
            elevation_step=10.0,
            azimuth_deviation_range=(-10.0, 10.0),
            azimuth_step=10.0,
        )
        algorithm.config = HomingConfig(trace_both_modes=False)

        triplets = algorithm._generate_triplets(search, gc_azimuth=45.0)

        azimuths = sorted([t[2] for t in triplets])
        expected = [35.0, 45.0, 55.0]  # 45 + (-10, 0, 10)

        np.testing.assert_array_almost_equal(azimuths, expected)


class TestHomingAlgorithmMUFCalculation:
    """Tests for MUF/LUF/FOT calculation."""

    @pytest.fixture
    def mock_solver(self):
        return Mock()

    @pytest.fixture
    def algorithm(self, mock_solver):
        return HomingAlgorithm(mock_solver)

    def test_calculate_frequencies_empty(self, algorithm):
        """Test frequency calculation with no winners."""
        result = HomingResult(
            tx_position=(40.0, -105.0, 0.0),
            rx_position=(42.0, -100.0, 0.0),
            great_circle_range_km=500.0,
            great_circle_azimuth_deg=45.0,
            winner_triplets=[],
        )

        algorithm._calculate_frequencies(result)

        assert result.muf == 0.0
        assert result.luf == 0.0
        assert result.fot == 0.0

    def test_calculate_frequencies_with_winners(self, algorithm):
        """Test frequency calculation with winners."""
        winners = [
            WinnerTriplet(
                frequency_mhz=f, elevation_deg=50.0, azimuth_deg=45.0,
                azimuth_deviation_deg=0.0, group_delay_ms=3.0, ground_range_km=500.0,
                landing_lat=42.0, landing_lon=-100.0, landing_error_km=10.0,
                mode=PropagationMode.O_MODE,
            )
            for f in [5.0, 7.0, 10.0, 12.0]
        ]

        result = HomingResult(
            tx_position=(40.0, -105.0, 0.0),
            rx_position=(42.0, -100.0, 0.0),
            great_circle_range_km=500.0,
            great_circle_azimuth_deg=45.0,
            winner_triplets=winners,
        )

        algorithm._calculate_frequencies(result)

        assert result.muf == 12.0
        assert result.luf == 5.0
        assert abs(result.fot - 12.0 * 0.85) < 0.01


class TestPropagationMode:
    """Tests for PropagationMode enum."""

    def test_mode_values(self):
        """Test mode string values."""
        assert PropagationMode.O_MODE.value == "O"
        assert PropagationMode.X_MODE.value == "X"

    def test_mode_comparison(self):
        """Test mode comparison."""
        assert PropagationMode.O_MODE != PropagationMode.X_MODE
        assert PropagationMode.O_MODE == PropagationMode.O_MODE
