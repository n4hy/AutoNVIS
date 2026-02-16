"""
Message Queue Abstraction Layer for Auto-NVIS

This module provides a unified interface for publishing and subscribing
to messages via RabbitMQ, abstracting the underlying implementation details.
"""

import json
import pika
import time
from typing import Callable, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from .logging_config import ServiceLogger


@dataclass
class Message:
    """Standard message format for Auto-NVIS"""
    topic: str
    timestamp: str
    source: str
    data: Dict[str, Any]

    def to_json(self) -> str:
        """Serialize message to JSON"""
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, json_str: str) -> 'Message':
        """Deserialize message from JSON"""
        data = json.loads(json_str)
        return cls(**data)


class MessageQueueClient:
    """
    RabbitMQ client for publishing and subscribing to messages
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5672,
        username: str = "guest",
        password: str = "guest",
        vhost: str = "/",
        exchange: str = "autonvis",
        exchange_type: str = "topic"
    ):
        """
        Initialize message queue client

        Args:
            host: RabbitMQ host
            port: RabbitMQ port
            username: RabbitMQ username
            password: RabbitMQ password
            vhost: RabbitMQ virtual host
            exchange: Exchange name
            exchange_type: Exchange type (topic, direct, fanout)
        """
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.vhost = vhost
        self.exchange = exchange
        self.exchange_type = exchange_type

        self.connection = None
        self.channel = None
        self.logger = ServiceLogger("message_queue")

        self._connect()

    def _connect(self, retry_count: int = 5, retry_delay: int = 5):
        """
        Establish connection to RabbitMQ with retry logic

        Args:
            retry_count: Number of connection attempts
            retry_delay: Delay between retries (seconds)
        """
        for attempt in range(retry_count):
            try:
                credentials = pika.PlainCredentials(self.username, self.password)
                parameters = pika.ConnectionParameters(
                    host=self.host,
                    port=self.port,
                    virtual_host=self.vhost,
                    credentials=credentials,
                    heartbeat=600,
                    blocked_connection_timeout=300
                )

                self.connection = pika.BlockingConnection(parameters)
                self.channel = self.connection.channel()

                # Declare exchange
                self.channel.exchange_declare(
                    exchange=self.exchange,
                    exchange_type=self.exchange_type,
                    durable=True
                )

                self.logger.info(f"Connected to RabbitMQ at {self.host}:{self.port}")
                return

            except pika.exceptions.AMQPConnectionError as e:
                self.logger.warning(
                    f"Connection attempt {attempt + 1}/{retry_count} failed: {e}"
                )
                if attempt < retry_count - 1:
                    time.sleep(retry_delay)
                else:
                    self.logger.error("Failed to connect to RabbitMQ", exc_info=True)
                    raise

    def publish(self, topic: str, data: Dict[str, Any], source: str = "unknown"):
        """
        Publish a message to a topic

        Args:
            topic: Topic routing key (e.g., 'obs.gnss_tec', 'wx.xray')
            data: Message payload
            source: Source identifier
        """
        message = Message(
            topic=topic,
            timestamp=datetime.utcnow().isoformat() + 'Z',
            source=source,
            data=data
        )

        try:
            self.channel.basic_publish(
                exchange=self.exchange,
                routing_key=topic,
                body=message.to_json(),
                properties=pika.BasicProperties(
                    delivery_mode=2,  # Persistent
                    content_type='application/json'
                )
            )

            self.logger.debug(f"Published message to {topic}", extra={'topic': topic})

        except Exception as e:
            self.logger.error(f"Failed to publish message to {topic}: {e}", exc_info=True)
            # Attempt to reconnect
            self._connect()
            raise

    def subscribe(
        self,
        topic_pattern: str,
        callback: Callable[[Message], None],
        queue_name: Optional[str] = None
    ):
        """
        Subscribe to messages matching a topic pattern

        Args:
            topic_pattern: Topic pattern (e.g., 'wx.*', 'obs.#')
            callback: Callback function to handle messages
            queue_name: Optional queue name (auto-generated if None)
        """
        # Declare queue
        if queue_name is None:
            result = self.channel.queue_declare(queue='', exclusive=True)
            queue_name = result.method.queue
        else:
            self.channel.queue_declare(queue=queue_name, durable=True)

        # Bind queue to exchange with topic pattern
        self.channel.queue_bind(
            exchange=self.exchange,
            queue=queue_name,
            routing_key=topic_pattern
        )

        self.logger.info(f"Subscribed to {topic_pattern} on queue {queue_name}")

        # Define message handler
        def on_message(ch, method, properties, body):
            try:
                message = Message.from_json(body.decode('utf-8'))
                callback(message)
                ch.basic_ack(delivery_tag=method.delivery_tag)

            except Exception as e:
                self.logger.error(f"Error processing message: {e}", exc_info=True)
                # Negative acknowledgment (requeue message)
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

        # Start consuming
        self.channel.basic_consume(
            queue=queue_name,
            on_message_callback=on_message
        )

    def start_consuming(self):
        """Start consuming messages (blocking)"""
        self.logger.info("Starting message consumption")
        try:
            self.channel.start_consuming()
        except KeyboardInterrupt:
            self.logger.info("Stopping message consumption")
            self.stop_consuming()

    def stop_consuming(self):
        """Stop consuming messages"""
        if self.channel:
            self.channel.stop_consuming()

    def close(self):
        """Close connection to RabbitMQ"""
        if self.connection and not self.connection.is_closed:
            self.connection.close()
            self.logger.info("Closed RabbitMQ connection")


# Topic naming conventions
class Topics:
    """Standard topic routing keys for Auto-NVIS"""

    # Space weather topics
    WX_XRAY = "wx.xray"
    WX_SOLAR_WIND = "wx.solar_wind"
    WX_GEOMAG = "wx.geomag"

    # Observation topics
    OBS_GNSS_TEC = "obs.gnss_tec"
    OBS_GLOTEC_MAP = "obs.glotec_map"
    OBS_IONOSONDE = "obs.ionosonde"
    OBS_NVIS_SOUNDER = "obs.nvis_sounder"
    OBS_NVIS_QUALITY = "obs.nvis_quality"

    # Processing topics
    PROC_STATE_UPDATE = "proc.state_update"
    PROC_GRID_READY = "proc.grid_ready"

    # Control topics
    CTRL_MODE_CHANGE = "ctrl.mode_change"
    CTRL_CYCLE_TRIGGER = "ctrl.cycle_trigger"

    # Output topics
    OUT_FREQUENCY_PLAN = "out.frequency_plan"
    OUT_COVERAGE_MAP = "out.coverage_map"
    OUT_ALERT = "out.alert"

    # Analysis topics
    ANALYSIS_INFO_GAIN = "analysis.info_gain"

    # Health monitoring
    HEALTH_STATUS = "health.status"
