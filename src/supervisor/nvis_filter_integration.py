"""
NVIS Filter Integration

Extends FilterOrchestrator with NVIS sounder observation handling.
Subscribes to RabbitMQ obs.nvis_sounder topic and integrates observations
into the SR-UKF filter cycle.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "assimilation" / "python"))

try:
    import autonvis_srukf
except ImportError:
    print("Warning: autonvis_srukf module not found. Ensure C++ module is built.")
    autonvis_srukf = None

from ..common.message_queue import MessageQueueClient, Topics, Message
from ..common.logging_config import ServiceLogger


class NVISFilterIntegration:
    """
    Integrates NVIS sounder observations with SR-UKF filter

    Responsibilities:
    - Subscribe to obs.nvis_sounder topic
    - Collect observations during cycle
    - Convert to C++ NVISMeasurement format
    - Create NVISSounderObservationModel
    - Call filter.update() with NVIS observations
    """

    def __init__(
        self,
        filter_orchestrator,
        mq_client: MessageQueueClient,
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
        alt_grid: np.ndarray
    ):
        """
        Initialize NVIS filter integration

        Args:
            filter_orchestrator: FilterOrchestrator instance
            mq_client: MessageQueueClient instance
            lat_grid: Latitude grid (degrees)
            lon_grid: Longitude grid (degrees)
            alt_grid: Altitude grid (km)
        """
        self.orchestrator = filter_orchestrator
        self.mq_client = mq_client
        self.lat_grid = lat_grid
        self.lon_grid = lon_grid
        self.alt_grid = alt_grid

        self.logger = ServiceLogger("nvis_filter_integration")

        # Observation buffer (cleared each cycle)
        self.observation_buffer: List[Dict[str, Any]] = []

        # Statistics
        self.total_observations = 0
        self.total_updates = 0
        self.total_rejected = 0

    def start_subscription(self):
        """Subscribe to NVIS observation topic"""
        self.mq_client.subscribe(
            topic_pattern=Topics.OBS_NVIS_SOUNDER,
            callback=self._on_nvis_observation,
            queue_name="filter_nvis_observations"
        )
        self.logger.info("Subscribed to NVIS observation topic")

    def _on_nvis_observation(self, message: Message):
        """
        Handle incoming NVIS observation

        Args:
            message: RabbitMQ message
        """
        try:
            data = message.data

            # Validate required fields
            required_fields = [
                'tx_latitude', 'tx_longitude', 'tx_altitude',
                'rx_latitude', 'rx_longitude', 'rx_altitude',
                'frequency', 'elevation_angle', 'azimuth', 'hop_distance',
                'signal_strength', 'group_delay', 'snr',
                'signal_strength_error', 'group_delay_error',
                'sounder_id', 'timestamp', 'is_o_mode'
            ]

            if all(field in data for field in required_fields):
                # Add to buffer
                self.observation_buffer.append(data)
                self.total_observations += 1

                self.logger.debug(
                    f"Buffered NVIS observation from {data['sounder_id']} "
                    f"(buffer size: {len(self.observation_buffer)})"
                )
            else:
                self.logger.warning("Invalid NVIS observation: missing fields")
                self.total_rejected += 1

        except Exception as e:
            self.logger.error(f"Error processing NVIS observation: {e}", exc_info=True)
            self.total_rejected += 1

    def collect_observations(self) -> List[Dict[str, Any]]:
        """
        Collect buffered observations and clear buffer

        Returns:
            List of NVIS observations
        """
        observations = self.observation_buffer.copy()
        self.observation_buffer = []
        return observations

    def create_observation_model(
        self,
        observations: List[Dict[str, Any]]
    ):
        """
        Create NVIS observation model from observations

        Args:
            observations: List of NVIS observation dicts

        Returns:
            Tuple of (model, obs_vector, obs_sqrt_cov)
        """
        if autonvis_srukf is None:
            raise RuntimeError("autonvis_srukf module not available")

        if len(observations) == 0:
            return None, None, None

        # Convert to C++ measurements
        cpp_measurements = []
        signal_errors = []
        delay_errors = []

        for obs in observations:
            meas = autonvis_srukf.NVISMeasurement()

            # Geometry
            meas.tx_latitude = obs['tx_latitude']
            meas.tx_longitude = obs['tx_longitude']
            meas.tx_altitude = obs['tx_altitude']
            meas.rx_latitude = obs['rx_latitude']
            meas.rx_longitude = obs['rx_longitude']
            meas.rx_altitude = obs['rx_altitude']

            # Propagation
            meas.frequency = obs['frequency']
            meas.elevation_angle = obs['elevation_angle']
            meas.azimuth = obs['azimuth']
            meas.hop_distance = obs['hop_distance']

            # Observables
            meas.signal_strength = obs['signal_strength']
            meas.group_delay = obs['group_delay']
            meas.snr = obs['snr']

            # Errors
            meas.signal_strength_error = obs['signal_strength_error']
            meas.group_delay_error = obs['group_delay_error']

            # Mode
            meas.is_o_mode = obs['is_o_mode']

            # Optional equipment parameters
            meas.tx_power = obs.get('tx_power', -100.0)
            meas.tx_antenna_gain = obs.get('tx_antenna_gain', -100.0)
            meas.rx_antenna_gain = obs.get('rx_antenna_gain', -100.0)

            cpp_measurements.append(meas)
            signal_errors.append(obs['signal_strength_error'])
            delay_errors.append(obs['group_delay_error'])

        # Create observation model
        model = autonvis_srukf.NVISSounderObservationModel(
            cpp_measurements,
            self.lat_grid.tolist(),
            self.lon_grid.tolist(),
            self.alt_grid.tolist()
        )

        # Build observation vector: [signal_1, ..., signal_N, delay_1, ..., delay_N]
        n_obs = len(observations)
        obs_vector = np.zeros(2 * n_obs)
        for i, obs in enumerate(observations):
            obs_vector[i] = obs['signal_strength']
            obs_vector[n_obs + i] = obs['group_delay']

        # Build observation error covariance (diagonal)
        obs_errors = np.array(signal_errors + delay_errors)
        obs_sqrt_cov = np.diag(obs_errors)

        self.logger.info(
            f"Created NVIS observation model: {n_obs} measurements, "
            f"{2*n_obs} observables"
        )

        return model, obs_vector, obs_sqrt_cov

    async def update_filter_with_nvis(self):
        """
        Update filter with buffered NVIS observations

        Returns:
            Number of observations processed
        """
        # Collect observations
        observations = self.collect_observations()

        if len(observations) == 0:
            self.logger.debug("No NVIS observations to process")
            return 0

        try:
            # Create observation model
            model, obs_vector, obs_sqrt_cov = self.create_observation_model(observations)

            if model is None:
                return 0

            # Update filter
            self.orchestrator.filter.filter.update(
                model,
                obs_vector,
                obs_sqrt_cov
            )

            self.total_updates += 1

            self.logger.info(
                f"Updated filter with {len(observations)} NVIS observations"
            )

            # Publish quality metrics to info gain analyzer (Phase 4)
            self._publish_quality_metrics(observations)

            return len(observations)

        except Exception as e:
            self.logger.error(f"Error updating filter with NVIS observations: {e}", exc_info=True)
            self.total_rejected += len(observations)
            return 0

    def _publish_quality_metrics(self, observations: List[Dict[str, Any]]):
        """
        Publish quality metrics for information gain analysis

        Args:
            observations: List of NVIS observations
        """
        try:
            # Extract quality metrics
            quality_data = {
                'timestamp': observations[0]['timestamp'] if observations else None,
                'n_observations': len(observations),
                'quality_tiers': {},
                'sounders': {}
            }

            # Group by quality tier
            for obs in observations:
                tier = obs.get('quality_tier', 'unknown')
                if tier not in quality_data['quality_tiers']:
                    quality_data['quality_tiers'][tier] = 0
                quality_data['quality_tiers'][tier] += 1

                # Group by sounder
                sounder_id = obs.get('sounder_id', 'unknown')
                if sounder_id not in quality_data['sounders']:
                    quality_data['sounders'][sounder_id] = {
                        'count': 0,
                        'quality_tier': tier,
                        'avg_snr': 0.0
                    }
                quality_data['sounders'][sounder_id]['count'] += 1
                quality_data['sounders'][sounder_id]['avg_snr'] += obs.get('snr', 0.0)

            # Average SNR
            for sounder_data in quality_data['sounders'].values():
                if sounder_data['count'] > 0:
                    sounder_data['avg_snr'] /= sounder_data['count']

            # Publish to quality topic
            self.mq_client.publish(
                topic=Topics.OBS_NVIS_QUALITY,
                data=quality_data,
                source="nvis_filter_integration"
            )

        except Exception as e:
            self.logger.error(f"Error publishing quality metrics: {e}", exc_info=True)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get NVIS integration statistics

        Returns:
            Statistics dictionary
        """
        return {
            'total_observations': self.total_observations,
            'total_updates': self.total_updates,
            'total_rejected': self.total_rejected,
            'buffer_size': len(self.observation_buffer)
        }


def extend_filter_orchestrator_with_nvis(
    orchestrator,
    mq_client: MessageQueueClient,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    alt_grid: np.ndarray
) -> NVISFilterIntegration:
    """
    Extend FilterOrchestrator with NVIS observation handling

    Args:
        orchestrator: FilterOrchestrator instance
        mq_client: MessageQueueClient instance
        lat_grid: Latitude grid (degrees)
        lon_grid: Longitude grid (degrees)
        alt_grid: Altitude grid (km)

    Returns:
        NVISFilterIntegration instance
    """
    integration = NVISFilterIntegration(
        orchestrator,
        mq_client,
        lat_grid,
        lon_grid,
        alt_grid
    )

    # Start subscription
    integration.start_subscription()

    return integration
