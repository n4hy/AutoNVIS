"""
NVIS Sounder Client - Main Orchestrator

Coordinates protocol adapters, quality assessment, aggregation,
and message queue publishing for NVIS sounder data ingestion.
"""

import asyncio
from typing import Dict, List
from .protocol_adapters import (
    BaseAdapter,
    TCPAdapter,
    HTTPAdapter,
    MQTTAdapter,
    EmailAdapter,
    NVISMeasurement
)
from .quality_assessor import QualityAssessor, QualityMetrics, QualityTier
from .adaptive_aggregator import AdaptiveAggregator, apply_rate_limiting
from ...common.config import AutoNVISConfig
from ...common.message_queue import MessageQueueClient, Topics
from ...common.logging_config import ServiceLogger


class NVISSounderClient:
    """
    Main orchestrator for NVIS sounder data ingestion

    Responsibilities:
    - Manage protocol adapters (TCP, HTTP, MQTT, Email)
    - Quality assessment of measurements
    - Adaptive aggregation for high-rate sounders
    - Rate limiting per tier
    - Publishing to message queue
    """

    def __init__(self, config: AutoNVISConfig):
        """
        Initialize NVIS sounder client

        Args:
            config: System configuration
        """
        self.config = config
        self.nvis_config = config.nvis_ingestion
        self.logger = ServiceLogger("nvis_sounder_client")

        # Initialize components
        self.quality_assessor = QualityAssessor(self.nvis_config)
        self.aggregator = AdaptiveAggregator(self.nvis_config)
        self.mq_client = MessageQueueClient(
            host=config.services.rabbitmq_host,
            port=config.services.rabbitmq_port,
            username=config.services.rabbitmq_user,
            password=config.services.rabbitmq_password
        )

        # Protocol adapters
        self.adapters: Dict[str, BaseAdapter] = {}
        self._initialize_adapters()

        # Runtime state
        self.running = False
        self.measurement_count = 0
        self.published_count = 0

    def _initialize_adapters(self):
        """Initialize enabled protocol adapters"""
        # TCP adapter
        if self.nvis_config.tcp.enabled:
            self.adapters['tcp'] = TCPAdapter(
                adapter_id='tcp',
                config={
                    'host': self.nvis_config.tcp.host,
                    'port': self.nvis_config.tcp.port
                }
            )
            self.logger.info("TCP adapter initialized")

        # HTTP adapter
        if self.nvis_config.http.enabled:
            self.adapters['http'] = HTTPAdapter(
                adapter_id='http',
                config={
                    'host': self.nvis_config.http.host,
                    'port': self.nvis_config.http.port
                }
            )
            self.logger.info("HTTP adapter initialized")

        # MQTT adapter
        if self.nvis_config.mqtt.enabled:
            self.adapters['mqtt'] = MQTTAdapter(
                adapter_id='mqtt',
                config={
                    'host': self.nvis_config.mqtt.host,
                    'port': self.nvis_config.mqtt.port,
                    'topic_prefix': self.nvis_config.mqtt.topic_prefix
                }
            )
            self.logger.info("MQTT adapter initialized")

        # Email adapter
        if self.nvis_config.email.enabled:
            self.adapters['email'] = EmailAdapter(
                adapter_id='email',
                config={
                    'imap_host': self.nvis_config.email.imap_host,
                    'imap_port': self.nvis_config.email.imap_port
                }
            )
            self.logger.info("Email adapter initialized")

        self.logger.info(f"Initialized {len(self.adapters)} protocol adapters")

    async def start(self):
        """Start NVIS sounder client"""
        self.running = True
        self.logger.info("Starting NVIS sounder client")

        # Start all adapters
        for adapter_id, adapter in self.adapters.items():
            await adapter.start()
            self.logger.info(f"Started adapter: {adapter_id}")

        # Start monitoring loop
        await self.run_monitoring_loop()

    async def stop(self):
        """Stop NVIS sounder client"""
        self.running = False
        self.logger.info("Stopping NVIS sounder client")

        # Stop all adapters
        for adapter_id, adapter in self.adapters.items():
            await adapter.stop()
            self.logger.info(f"Stopped adapter: {adapter_id}")

        # Close message queue
        self.mq_client.close()

    async def run_monitoring_loop(self):
        """
        Main ingestion loop

        Continuously monitors all adapters for measurements,
        assesses quality, aggregates, and publishes to queue.
        """
        self.logger.info("Starting monitoring loop")

        # Create tasks for each adapter
        adapter_tasks = []
        for adapter_id, adapter in self.adapters.items():
            task = asyncio.create_task(
                self._process_adapter_measurements(adapter_id, adapter)
            )
            adapter_tasks.append(task)

        # Wait for all adapter tasks
        try:
            await asyncio.gather(*adapter_tasks)
        except asyncio.CancelledError:
            self.logger.info("Monitoring loop cancelled")
        except Exception as e:
            self.logger.error(f"Monitoring loop error: {e}", exc_info=True)

    async def _process_adapter_measurements(
        self,
        adapter_id: str,
        adapter: BaseAdapter
    ):
        """
        Process measurements from a single adapter

        Args:
            adapter_id: Adapter identifier
            adapter: Protocol adapter instance
        """
        self.logger.info(f"Processing measurements from {adapter_id}")

        try:
            async for measurement in adapter.get_measurements():
                # Get sounder metadata
                metadata = adapter.get_sounder_metadata(measurement.sounder_id)

                # Assess quality
                metrics = self.quality_assessor.assess_measurement(
                    measurement, metadata
                )
                overall_score = metrics.overall_score(
                    self.quality_assessor.weights
                )
                tier = self.quality_assessor.assign_tier(metrics)

                # Map tier to error covariance
                errors = self.quality_assessor.map_to_error_covariance(tier)
                measurement.signal_strength_error = errors['signal_error_db']
                measurement.group_delay_error = errors['delay_error_ms']

                # Adaptive aggregation
                aggregated = self.aggregator.add_measurement(
                    measurement.sounder_id,
                    measurement,
                    overall_score
                )

                # Publish if ready (aggregated or pass-through)
                if aggregated:
                    self._publish_measurement(aggregated, tier, metrics)

                self.measurement_count += 1

                # Log progress periodically
                if self.measurement_count % 100 == 0:
                    self.logger.info(
                        f"Processed {self.measurement_count} measurements "
                        f"({self.published_count} published)"
                    )

        except Exception as e:
            self.logger.error(
                f"Error processing {adapter_id} measurements: {e}",
                exc_info=True
            )

    def _publish_measurement(
        self,
        measurement: NVISMeasurement,
        tier: QualityTier,
        metrics: QualityMetrics
    ):
        """
        Publish measurement to message queue

        Args:
            measurement: NVIS measurement
            tier: Quality tier
            metrics: Quality metrics
        """
        # Prepare data payload
        data = {
            # Geometry
            'tx_latitude': measurement.tx_latitude,
            'tx_longitude': measurement.tx_longitude,
            'tx_altitude': measurement.tx_altitude,
            'rx_latitude': measurement.rx_latitude,
            'rx_longitude': measurement.rx_longitude,
            'rx_altitude': measurement.rx_altitude,

            # Propagation
            'frequency': measurement.frequency,
            'elevation_angle': measurement.elevation_angle,
            'azimuth': measurement.azimuth,
            'hop_distance': measurement.hop_distance,

            # Observables
            'signal_strength': measurement.signal_strength,
            'group_delay': measurement.group_delay,
            'snr': measurement.snr,

            # Errors
            'signal_strength_error': measurement.signal_strength_error,
            'group_delay_error': measurement.group_delay_error,

            # Metadata
            'sounder_id': measurement.sounder_id,
            'timestamp': measurement.timestamp,
            'is_o_mode': measurement.is_o_mode,

            # Quality
            'quality_tier': tier.value,
            'quality_metrics': {
                'signal_quality': metrics.signal_quality,
                'calibration_quality': metrics.calibration_quality,
                'temporal_quality': metrics.temporal_quality,
                'spatial_quality': metrics.spatial_quality,
                'equipment_quality': metrics.equipment_quality,
                'historical_quality': metrics.historical_quality
            }
        }

        # Publish to observation topic
        self.mq_client.publish(
            topic=Topics.OBS_NVIS_SOUNDER,
            data=data,
            source=f"nvis_sounder_{measurement.sounder_id}"
        )

        self.published_count += 1

        self.logger.debug(
            f"Published measurement from {measurement.sounder_id} "
            f"(tier={tier.value}, freq={measurement.frequency}MHz)"
        )

    async def flush_pending_observations(self) -> int:
        """
        Flush all pending aggregated observations

        Should be called at end of filter cycle to ensure
        all buffered observations are processed.

        Returns:
            Number of observations flushed
        """
        aggregated = self.aggregator.flush_all_bins()

        for measurement in aggregated:
            # Re-assess quality for flushed measurements
            metadata = None  # TODO: Track metadata
            metrics = self.quality_assessor.assess_measurement(measurement, metadata)
            tier = self.quality_assessor.assign_tier(metrics)

            self._publish_measurement(measurement, tier, metrics)

        if len(aggregated) > 0:
            self.logger.info(f"Flushed {len(aggregated)} pending observations")

        return len(aggregated)


async def main():
    """Main entry point for standalone testing"""
    from ...common.config import get_config

    config = get_config()
    client = NVISSounderClient(config)

    try:
        await client.start()
    except KeyboardInterrupt:
        print("\nShutting down...")
        await client.stop()


if __name__ == '__main__':
    asyncio.run(main())
