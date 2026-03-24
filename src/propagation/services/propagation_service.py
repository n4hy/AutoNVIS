"""
Propagation Service - Ray Tracing Integration for Auto-NVIS

This service integrates the native C++ ray tracer with the SR-UKF filter
to produce real-time LUF/MUF predictions and propagation products.

Also integrates the Vogler-Hoffmeyer HF channel model for realistic
communications simulation with time-varying fading effects.

Interfaces:
    - Input: Electron density grid from SR-UKF filter (via RabbitMQ)
    - Output: LUF/MUF products published to RabbitMQ
    - Configuration: Grid coordinates, transmitter location, frequencies
"""

import sys
import logging
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timezone
import numpy as np

# Import message queue for RabbitMQ integration
try:
    from ...common.message_queue import MessageQueueClient, Topics, Message
    MQ_AVAILABLE = True
except ImportError:
    try:
        from src.common.message_queue import MessageQueueClient, Topics, Message
        MQ_AVAILABLE = True
    except ImportError:
        MQ_AVAILABLE = False

# Add propagation module to path
sys.path.insert(0, '/home/n4hy/AutoNVIS/src/propagation/python')

from pharlap_replacement import RayTracer

# Import from products module
try:
    from ..products.luf_muf_calculator import LUFMUFCalculator, FrequencyRecommender
except ImportError:
    # Fallback for direct execution
    from pathlib import Path
    products_path = Path(__file__).parent.parent / "products"
    sys.path.insert(0, str(products_path))
    from luf_muf_calculator import LUFMUFCalculator, FrequencyRecommender

# Import channel model components
try:
    from channel_models import (
        VoglerHoffmeyerModel,
        RayToChannelMapper,
        ChannelConditions,
        IonosphericRegion,
        DisturbanceLevel
    )
    CHANNEL_MODEL_AVAILABLE = True
except ImportError:
    CHANNEL_MODEL_AVAILABLE = False


class PropagationService:
    """
    Service for real-time ionospheric propagation prediction.

    Integrates native C++ ray tracer with SR-UKF filter output to
    calculate LUF/MUF and generate frequency recommendations.
    """

    def __init__(
        self,
        tx_lat: float,
        tx_lon: float,
        tx_alt: float = 0.0,
        freq_min: float = 2.0,
        freq_max: float = 15.0,
        freq_step: float = 0.5,
        elevation_min: float = 70.0,
        elevation_max: float = 90.0,
        elevation_step: float = 2.0,
        azimuth_step: float = 15.0,
        absorption_threshold_db: float = 50.0,
        snr_threshold_db: float = 10.0,
        channel_model_enabled: bool = True,
        channel_sample_rate: float = 1e6
    ):
        """
        Initialize propagation service.

        Args:
            tx_lat: Transmitter latitude (degrees)
            tx_lon: Transmitter longitude (degrees)
            tx_alt: Transmitter altitude (km)
            freq_min: Minimum frequency to test (MHz)
            freq_max: Maximum frequency to test (MHz)
            freq_step: Frequency step size (MHz)
            elevation_min: Minimum elevation angle (degrees, NVIS: 70)
            elevation_max: Maximum elevation angle (degrees, NVIS: 90)
            elevation_step: Elevation step (degrees)
            azimuth_step: Azimuth step (degrees)
            absorption_threshold_db: Maximum acceptable absorption (dB)
            snr_threshold_db: Minimum acceptable SNR (dB)
            channel_model_enabled: Enable Vogler-Hoffmeyer channel model
            channel_sample_rate: Sample rate for channel model (Hz)
        """
        self.logger = logging.getLogger(__name__)

        # Transmitter configuration
        self.tx_lat = tx_lat
        self.tx_lon = tx_lon
        self.tx_alt = tx_alt

        # Frequency scan configuration
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.freq_step = freq_step

        # Ray tracing configuration (NVIS geometry)
        self.elevation_min = elevation_min
        self.elevation_max = elevation_max
        self.elevation_step = elevation_step
        self.azimuth_step = azimuth_step

        # Quality thresholds
        self.absorption_threshold_db = absorption_threshold_db
        self.snr_threshold_db = snr_threshold_db

        # Ray tracer instance (created when grid is provided)
        self.tracer: Optional[RayTracer] = None

        # LUF/MUF calculator
        self.calculator = LUFMUFCalculator(
            absorption_threshold_db=absorption_threshold_db,
            snr_threshold_db=snr_threshold_db
        )

        # Frequency recommender
        self.recommender = FrequencyRecommender()

        # Channel model components
        self.channel_model_enabled = channel_model_enabled and CHANNEL_MODEL_AVAILABLE
        self.channel_sample_rate = channel_sample_rate
        self._channel_model: Optional['VoglerHoffmeyerModel'] = None
        self._ray_mapper: Optional['RayToChannelMapper'] = None
        self._current_kp_index: float = 2.0  # Default Kp

        if self.channel_model_enabled:
            self._initialize_channel_model()

        self.logger.info(
            f"PropagationService initialized: TX=({tx_lat:.2f}, {tx_lon:.2f}), "
            f"Freq={freq_min}-{freq_max} MHz, Elev={elevation_min}-{elevation_max}°, "
            f"Channel model={'enabled' if self.channel_model_enabled else 'disabled'}"
        )

    def _initialize_channel_model(self) -> None:
        """Initialize the Vogler-Hoffmeyer channel model components."""
        if not CHANNEL_MODEL_AVAILABLE:
            self.logger.warning("Channel model not available - module not found")
            return

        try:
            self._channel_model = VoglerHoffmeyerModel(
                sample_rate=self.channel_sample_rate
            )
            self._ray_mapper = RayToChannelMapper(
                sample_rate=self.channel_sample_rate,
                max_modes=3
            )
            self.logger.info(
                f"Channel model initialized: {self._channel_model.name}, "
                f"sample_rate={self.channel_sample_rate/1e6:.3f} MHz"
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize channel model: {e}")
            self.channel_model_enabled = False

    def initialize_ray_tracer(
        self,
        ne_grid: np.ndarray,
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
        alt_grid: np.ndarray,
        xray_flux: float = 0.0
    ) -> None:
        """
        Initialize ray tracer with ionospheric grid from SR-UKF filter.

        Args:
            ne_grid: Electron density grid (n_lat, n_lon, n_alt) in el/m³
            lat_grid: Latitude grid (degrees)
            lon_grid: Longitude grid (degrees)
            alt_grid: Altitude grid (km)
            xray_flux: GOES X-ray flux for D-region absorption (W/m²)
        """
        self.logger.info(
            f"Initializing ray tracer with grid: "
            f"{ne_grid.shape[0]}×{ne_grid.shape[1]}×{ne_grid.shape[2]}"
        )

        # Validate grid dimensions
        if ne_grid.shape != (len(lat_grid), len(lon_grid), len(alt_grid)):
            raise ValueError(
                f"Grid shape mismatch: ne_grid={ne_grid.shape}, "
                f"expected ({len(lat_grid)}, {len(lon_grid)}, {len(alt_grid)})"
            )

        # Create ray tracer instance
        self.tracer = RayTracer(
            ne_grid=ne_grid,
            lat=lat_grid,
            lon=lon_grid,
            alt=alt_grid,
            xray_flux=xray_flux
        )

        # Log grid statistics
        ne_max = np.max(ne_grid)
        ne_mean = np.mean(ne_grid)
        self.logger.info(
            f"Ray tracer initialized: Ne_max={ne_max:.2e} el/m³, "
            f"Ne_mean={ne_mean:.2e} el/m³, X-ray={xray_flux:.2e} W/m²"
        )

    def calculate_luf_muf(self) -> Dict[str, Any]:
        """
        Calculate LUF/MUF and propagation products.

        Returns:
            Dictionary containing:
                - luf_mhz: Lowest Usable Frequency
                - muf_mhz: Maximum Usable Frequency
                - fot_mhz: Frequency of Optimum Traffic
                - usable_range_mhz: [LUF, MUF] tuple
                - blackout: True if no usable frequencies
                - coverage_stats: Ray statistics
                - frequency_recommendations: Recommended frequencies for ALE
                - timestamp_utc: Calculation time

        Raises:
            RuntimeError: If ray tracer not initialized
        """
        if self.tracer is None:
            raise RuntimeError("Ray tracer not initialized. Call initialize_ray_tracer() first.")

        self.logger.info("Calculating LUF/MUF coverage...")
        start_time = datetime.now(timezone.utc)

        # Calculate multi-frequency coverage
        coverage = self.tracer.calculate_coverage(
            tx_lat=self.tx_lat,
            tx_lon=self.tx_lon,
            freq_min=self.freq_min,
            freq_max=self.freq_max,
            freq_step=self.freq_step,
            snr_threshold_db=self.snr_threshold_db
        )

        # Extract LUF/MUF results
        luf_mhz = coverage['luf']
        muf_mhz = coverage['muf']
        fot_mhz = coverage['fot']  # FOT = 0.85 * MUF
        blackout = coverage['blackout']

        # Calculate coverage statistics
        total_rays = 0
        reflected_rays = 0
        usable_rays = 0
        total_absorption = 0.0

        for freq, paths in coverage['coverage_map'].items():
            total_rays += len(paths)
            for path in paths:
                if path['reflected']:
                    reflected_rays += 1
                    if path['absorption_db'] < self.absorption_threshold_db:
                        usable_rays += 1
                        total_absorption += path['absorption_db']

        avg_absorption_db = total_absorption / usable_rays if usable_rays > 0 else 0.0

        coverage_stats = {
            'total_rays': total_rays,
            'reflected_rays': reflected_rays,
            'usable_rays': usable_rays,
            'reflection_rate': reflected_rays / total_rays if total_rays > 0 else 0.0,
            'usability_rate': usable_rays / total_rays if total_rays > 0 else 0.0,
            'avg_absorption_db': avg_absorption_db
        }

        # Get frequency recommendations for ALE
        recommendations = self.recommender.recommend_frequencies(
            luf_mhz=luf_mhz,
            muf_mhz=muf_mhz,
            num_frequencies=5,
            strategy='distributed'
        )

        calculation_time = (datetime.now(timezone.utc) - start_time).total_seconds()

        self.logger.info(
            f"LUF/MUF calculated in {calculation_time:.2f}s: "
            f"LUF={luf_mhz:.2f} MHz, MUF={muf_mhz:.2f} MHz, "
            f"FOT={fot_mhz:.2f} MHz, Blackout={blackout}"
        )

        return {
            'luf_mhz': float(luf_mhz),
            'muf_mhz': float(muf_mhz),
            'fot_mhz': float(fot_mhz),
            'usable_range_mhz': [float(luf_mhz), float(muf_mhz)] if not blackout else None,
            'blackout': bool(blackout),
            'coverage_stats': coverage_stats,
            'frequency_recommendations': recommendations,
            'transmitter': {
                'latitude': self.tx_lat,
                'longitude': self.tx_lon,
                'altitude_km': self.tx_alt
            },
            'timestamp_utc': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            'calculation_time_sec': calculation_time
        }

    def calculate_nvis_coverage(self, freq_mhz: float) -> Dict[str, Any]:
        """
        Calculate NVIS coverage map for a single frequency.

        Args:
            freq_mhz: Frequency to calculate (MHz)

        Returns:
            Dictionary containing:
                - frequency_mhz: Operating frequency
                - ray_paths: List of ray path dictionaries
                - coverage_summary: Statistics about coverage
                - timestamp_utc: Calculation time
        """
        if self.tracer is None:
            raise RuntimeError("Ray tracer not initialized.")

        self.logger.info(f"Calculating NVIS coverage for {freq_mhz:.2f} MHz...")

        paths = self.tracer.trace_nvis(
            tx_lat=self.tx_lat,
            tx_lon=self.tx_lon,
            freq_mhz=freq_mhz,
            elevation_min=self.elevation_min,
            elevation_max=self.elevation_max,
            elevation_step=self.elevation_step,
            azimuth_step=self.azimuth_step
        )

        # Analyze coverage
        reflected = sum(1 for p in paths if p['reflected'])
        escaped = sum(1 for p in paths if p['escaped'])
        absorbed = sum(1 for p in paths if p['absorbed'])

        avg_range = np.mean([p['ground_range'] for p in paths if p['reflected']]) if reflected > 0 else 0.0
        max_range = np.max([p['ground_range'] for p in paths if p['reflected']]) if reflected > 0 else 0.0

        coverage_summary = {
            'total_rays': len(paths),
            'reflected': reflected,
            'escaped': escaped,
            'absorbed': absorbed,
            'reflection_rate': reflected / len(paths) if len(paths) > 0 else 0.0,
            'avg_ground_range_km': float(avg_range),
            'max_ground_range_km': float(max_range)
        }

        self.logger.info(
            f"Coverage calculated: {reflected}/{len(paths)} rays reflected, "
            f"avg range={avg_range:.1f} km"
        )

        return {
            'frequency_mhz': freq_mhz,
            'ray_paths': paths,
            'coverage_summary': coverage_summary,
            'transmitter': {
                'latitude': self.tx_lat,
                'longitude': self.tx_lon,
                'altitude_km': self.tx_alt
            },
            'timestamp_utc': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        }

    def get_configuration(self) -> Dict[str, Any]:
        """
        Get current propagation service configuration.

        Returns:
            Configuration dictionary
        """
        return {
            'transmitter': {
                'latitude': self.tx_lat,
                'longitude': self.tx_lon,
                'altitude_km': self.tx_alt
            },
            'frequency_scan': {
                'min_mhz': self.freq_min,
                'max_mhz': self.freq_max,
                'step_mhz': self.freq_step
            },
            'ray_geometry': {
                'elevation_min': self.elevation_min,
                'elevation_max': self.elevation_max,
                'elevation_step': self.elevation_step,
                'azimuth_step': self.azimuth_step
            },
            'thresholds': {
                'absorption_db': self.absorption_threshold_db,
                'snr_db': self.snr_threshold_db
            },
            'tracer_initialized': self.tracer is not None,
            'channel_model': {
                'enabled': self.channel_model_enabled,
                'available': CHANNEL_MODEL_AVAILABLE,
                'sample_rate_hz': self.channel_sample_rate,
                'model_name': self._channel_model.name if self._channel_model else None,
                'configured': self._channel_model.is_configured if self._channel_model else False
            }
        }

    def set_kp_index(self, kp_index: float) -> None:
        """
        Set the current Kp index for channel model configuration.

        Args:
            kp_index: Planetary K index (0-9)
        """
        self._current_kp_index = max(0.0, min(9.0, kp_index))
        self.logger.debug(f"Kp index set to {self._current_kp_index:.1f}")

    def apply_channel_effects(
        self,
        samples: np.ndarray,
        freq_mhz: float,
        kp_index: Optional[float] = None,
        use_ray_paths: bool = True
    ) -> np.ndarray:
        """
        Apply realistic channel fading to I/Q samples.

        Uses the Vogler-Hoffmeyer channel model configured from ray tracing
        results to apply:
            - Multipath delay spread
            - Doppler spread and shift
            - Time-varying Rayleigh/Rician fading
            - Optional spread-F effects

        Args:
            samples: Complex I/Q input samples
            freq_mhz: Operating frequency in MHz
            kp_index: Optional Kp index override (uses stored value if None)
            use_ray_paths: If True, derive channel from ray tracing

        Returns:
            Complex I/Q output samples with channel effects

        Raises:
            RuntimeError: If channel model not enabled or tracer not initialized

        Example:
            >>> output = service.apply_channel_effects(input_iq, freq_mhz=7.0)
        """
        if not self.channel_model_enabled:
            raise RuntimeError("Channel model not enabled")

        if self._channel_model is None:
            raise RuntimeError("Channel model not initialized")

        kp = kp_index if kp_index is not None else self._current_kp_index

        # Configure channel from ray tracing if available and requested
        if use_ray_paths and self.tracer is not None:
            self._configure_channel_from_rays(freq_mhz, kp)
        elif not self._channel_model.is_configured:
            # Use default configuration based on Kp
            self._configure_channel_from_conditions(kp)

        # Process samples through channel
        output = self._channel_model.process_samples(samples)

        return output

    def _configure_channel_from_rays(
        self,
        freq_mhz: float,
        kp_index: float
    ) -> None:
        """
        Configure channel model from ray tracing results.

        Args:
            freq_mhz: Operating frequency in MHz
            kp_index: Current Kp index
        """
        if self.tracer is None or self._ray_mapper is None:
            return

        # Trace rays for this frequency
        paths = self.tracer.trace_nvis(
            tx_lat=self.tx_lat,
            tx_lon=self.tx_lon,
            freq_mhz=freq_mhz,
            elevation_min=self.elevation_min,
            elevation_max=self.elevation_max,
            elevation_step=self.elevation_step,
            azimuth_step=self.azimuth_step
        )

        # Map rays to channel configuration
        mapped = self._ray_mapper.map_rays_to_channel(
            ray_paths=paths,
            kp_index=kp_index
        )

        # Create and configure channel from mapped config
        from channel_models.hifi import VoglerHoffmeyerChannel
        channel = VoglerHoffmeyerChannel(mapped.config)

        # Update internal model state to use this configuration
        self._channel_model._config = mapped.config
        self._channel_model._channel = channel

        self.logger.debug(
            f"Channel configured from {len(paths)} rays, "
            f"{len(mapped.config.modes)} modes, quality={mapped.mapping_quality:.2f}"
        )

    def _configure_channel_from_conditions(self, kp_index: float) -> None:
        """
        Configure channel model from conditions when rays unavailable.

        Args:
            kp_index: Current Kp index
        """
        if self._channel_model is None:
            return

        # Build conditions from Kp
        if kp_index < 2:
            disturbance = DisturbanceLevel.QUIET
        elif kp_index < 4:
            disturbance = DisturbanceLevel.MODERATE
        elif kp_index < 6:
            disturbance = DisturbanceLevel.DISTURBED
        else:
            disturbance = DisturbanceLevel.STORM

        conditions = ChannelConditions(
            region=IonosphericRegion.MIDLATITUDE,
            disturbance_level=disturbance,
            kp_index=kp_index,
            spread_f_present=(kp_index >= 5)
        )

        self._channel_model.configure(conditions)
        self.logger.debug(f"Channel configured from conditions: Kp={kp_index:.1f}")

    def get_channel_response(self) -> Optional[Dict[str, Any]]:
        """
        Get current channel model response parameters.

        Returns:
            Dictionary with channel response parameters, or None if not configured
        """
        if not self.channel_model_enabled or self._channel_model is None:
            return None

        if not self._channel_model.is_configured:
            return None

        response = self._channel_model.get_channel_response()
        return {
            'delay_us': response.delay_us,
            'delay_spread_us': response.delay_spread_us,
            'doppler_shift_hz': response.doppler_shift_hz,
            'doppler_spread_hz': response.doppler_spread_hz,
            'path_loss_db': response.path_loss_db,
            'fading_type': response.fading_type.value,
            'k_factor': response.k_factor,
            'mode_name': response.mode_name,
            'layer': response.layer
        }

    def reset_channel(self) -> None:
        """Reset channel model state for processing a new signal."""
        if self._channel_model is not None:
            self._channel_model.reset()
            self.logger.debug("Channel model reset")

    # =========================================================================
    # RabbitMQ Message Queue Integration
    # =========================================================================

    def connect_message_queue(
        self,
        host: str = "localhost",
        port: int = 5672,
        username: str = "guest",
        password: str = "guest"
    ) -> bool:
        """
        Connect to RabbitMQ message queue for publishing products.

        Args:
            host: RabbitMQ host
            port: RabbitMQ port
            username: RabbitMQ username
            password: RabbitMQ password

        Returns:
            True if connection successful, False otherwise
        """
        if not MQ_AVAILABLE:
            self.logger.warning("Message queue module not available")
            return False

        try:
            self._mq_client = MessageQueueClient(
                host=host,
                port=port,
                username=username,
                password=password
            )
            self.logger.info(f"Connected to RabbitMQ at {host}:{port}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to RabbitMQ: {e}")
            self._mq_client = None
            return False

    def publish_luf_muf(self, results: Dict[str, Any]) -> bool:
        """
        Publish LUF/MUF calculation results to message queue.

        Args:
            results: Results from calculate_luf_muf()

        Returns:
            True if published successfully
        """
        if not hasattr(self, '_mq_client') or self._mq_client is None:
            self.logger.debug("Message queue not connected, skipping publish")
            return False

        try:
            self._mq_client.publish(
                topic=Topics.OUT_LUF_MUF,
                data=results,
                source="propagation_service"
            )
            self.logger.info(
                f"Published LUF/MUF: LUF={results['luf_mhz']:.2f}, "
                f"MUF={results['muf_mhz']:.2f} MHz"
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to publish LUF/MUF: {e}")
            return False

    def publish_coverage(self, coverage: Dict[str, Any]) -> bool:
        """
        Publish NVIS coverage results to message queue.

        Args:
            coverage: Results from calculate_nvis_coverage()

        Returns:
            True if published successfully
        """
        if not hasattr(self, '_mq_client') or self._mq_client is None:
            return False

        try:
            # Simplify ray paths for transmission (remove large arrays)
            simplified_coverage = {
                'frequency_mhz': coverage['frequency_mhz'],
                'coverage_summary': coverage['coverage_summary'],
                'transmitter': coverage['transmitter'],
                'timestamp_utc': coverage['timestamp_utc']
            }

            self._mq_client.publish(
                topic=Topics.OUT_COVERAGE_MAP,
                data=simplified_coverage,
                source="propagation_service"
            )
            self.logger.info(
                f"Published coverage for {coverage['frequency_mhz']:.2f} MHz"
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to publish coverage: {e}")
            return False

    def publish_frequency_plan(self, recommendations: List[float]) -> bool:
        """
        Publish frequency plan/recommendations to message queue.

        Args:
            recommendations: List of recommended frequencies (MHz)

        Returns:
            True if published successfully
        """
        if not hasattr(self, '_mq_client') or self._mq_client is None:
            return False

        try:
            self._mq_client.publish(
                topic=Topics.OUT_FREQUENCY_PLAN,
                data={
                    'frequencies_mhz': recommendations,
                    'transmitter': {
                        'latitude': self.tx_lat,
                        'longitude': self.tx_lon
                    },
                    'timestamp_utc': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
                },
                source="propagation_service"
            )
            self.logger.info(f"Published frequency plan: {recommendations}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to publish frequency plan: {e}")
            return False

    def subscribe_grid_updates(
        self,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> None:
        """
        Subscribe to electron density grid updates from SR-UKF filter.

        Args:
            callback: Optional custom callback for grid updates.
                      If None, uses default handler that initializes ray tracer.
        """
        if not hasattr(self, '_mq_client') or self._mq_client is None:
            self.logger.warning("Message queue not connected")
            return

        def default_callback(message: 'Message'):
            """Default handler for grid updates."""
            self.logger.info(f"Received grid update from {message.source}")

            grid_data = message.data
            if all(k in grid_data for k in ['ne_grid', 'lat', 'lon', 'alt']):
                # Convert lists back to numpy arrays
                ne_grid = np.array(grid_data['ne_grid'])
                lat = np.array(grid_data['lat'])
                lon = np.array(grid_data['lon'])
                alt = np.array(grid_data['alt'])
                xray_flux = grid_data.get('xray_flux', 0.0)

                # Reinitialize ray tracer with new grid
                self.initialize_ray_tracer(ne_grid, lat, lon, alt, xray_flux)

                # Calculate and publish products
                try:
                    results = self.calculate_luf_muf()
                    self.publish_luf_muf(results)

                    # Also publish frequency plan
                    if not results['blackout']:
                        self.publish_frequency_plan(
                            results['frequency_recommendations']
                        )
                except Exception as e:
                    self.logger.error(f"Error processing grid update: {e}")

        handler = callback if callback else default_callback

        self._mq_client.subscribe(
            topic_pattern=Topics.PROC_GRID_READY,
            callback=handler
        )
        self.logger.info(f"Subscribed to grid updates on {Topics.PROC_GRID_READY}")

    def run_service(self) -> None:
        """
        Run as a message-driven service.

        Subscribes to grid updates and publishes propagation products.
        Blocks until interrupted.
        """
        if not hasattr(self, '_mq_client') or self._mq_client is None:
            raise RuntimeError("Message queue not connected. Call connect_message_queue() first.")

        self.logger.info("Starting propagation service...")
        self.subscribe_grid_updates()

        try:
            self._mq_client.start_consuming()
        except KeyboardInterrupt:
            self.logger.info("Propagation service stopped by user")
        finally:
            if self._mq_client:
                self._mq_client.close()

    def close(self) -> None:
        """Close message queue connection."""
        if hasattr(self, '_mq_client') and self._mq_client is not None:
            self._mq_client.close()
            self._mq_client = None
