"""
Pytest Configuration and Fixtures for Auto-NVIS Test Suite

Provides mock infrastructure for testing without external dependencies:
- MockMessageQueueClient: In-memory pub/sub for tests without RabbitMQ
- MockNVISSounderClient: Mock NVIS client for integration tests
- Common test fixtures and utilities
"""

import pytest
import asyncio
import threading
import time
from typing import Dict, List, Callable, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


@dataclass
class MockMessage:
    """Standard message format mirroring the real Message class"""
    topic: str
    timestamp: str
    source: str
    data: Dict[str, Any]

    def to_json(self) -> str:
        """Serialize message to JSON"""
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, json_str: str) -> 'MockMessage':
        """Deserialize message from JSON"""
        data = json.loads(json_str)
        return cls(**data)


class MockMessageQueueClient:
    """
    In-memory mock of MessageQueueClient for testing without RabbitMQ

    Provides:
    - Thread-safe pub/sub with topic routing
    - Wildcard topic matching (# and *)
    - Message delivery tracking
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
        """Initialize mock client (parameters for API compatibility)"""
        self.host = host
        self.port = port
        self.exchange = exchange
        self.exchange_type = exchange_type

        # In-memory message store
        self._subscribers: Dict[str, List[Callable]] = {}
        self._lock = threading.Lock()
        self._message_history: List[MockMessage] = []
        self._is_connected = True
        self._consuming = False

    def _connect(self, retry_count: int = 5, retry_delay: int = 5):
        """Mock connection (always succeeds)"""
        self._is_connected = True

    def _match_topic(self, pattern: str, topic: str) -> bool:
        """
        Match topic against pattern with AMQP-style wildcards

        - '#' matches zero or more dot-separated words
        - '*' matches exactly one word
        """
        pattern_parts = pattern.split('.')
        topic_parts = topic.split('.')

        i, j = 0, 0
        while i < len(pattern_parts) and j < len(topic_parts):
            if pattern_parts[i] == '#':
                # '#' matches remaining parts
                if i == len(pattern_parts) - 1:
                    return True
                # Try to match remaining pattern
                for k in range(j, len(topic_parts) + 1):
                    if self._match_topic('.'.join(pattern_parts[i+1:]), '.'.join(topic_parts[k:])):
                        return True
                return False
            elif pattern_parts[i] == '*':
                # '*' matches exactly one word
                i += 1
                j += 1
            elif pattern_parts[i] == topic_parts[j]:
                i += 1
                j += 1
            else:
                return False

        # Handle trailing '#'
        while i < len(pattern_parts) and pattern_parts[i] == '#':
            i += 1

        return i == len(pattern_parts) and j == len(topic_parts)

    def publish(self, topic: str, data: Dict[str, Any], source: str = "unknown"):
        """
        Publish a message to a topic

        Args:
            topic: Topic routing key
            data: Message payload
            source: Source identifier
        """
        if not self._is_connected:
            raise ConnectionError("Not connected to message queue")

        message = MockMessage(
            topic=topic,
            timestamp=datetime.utcnow().isoformat() + 'Z',
            source=source,
            data=data
        )

        with self._lock:
            self._message_history.append(message)

            # Deliver to matching subscribers
            for pattern, callbacks in self._subscribers.items():
                if self._match_topic(pattern, topic):
                    for callback in callbacks:
                        try:
                            # Run callback in a separate thread to avoid blocking
                            threading.Thread(
                                target=callback,
                                args=(message,),
                                daemon=True
                            ).start()
                        except Exception:
                            pass  # Ignore callback errors like the real client

    def subscribe(
        self,
        topic_pattern: str,
        callback: Callable[[MockMessage], None],
        queue_name: Optional[str] = None
    ):
        """
        Subscribe to messages matching a topic pattern

        Args:
            topic_pattern: Topic pattern with optional wildcards
            callback: Callback function for messages
            queue_name: Optional queue name (ignored in mock)
        """
        with self._lock:
            if topic_pattern not in self._subscribers:
                self._subscribers[topic_pattern] = []
            self._subscribers[topic_pattern].append(callback)

    def start_consuming(self):
        """Start consuming messages (blocks until stopped)"""
        self._consuming = True
        while self._consuming:
            time.sleep(0.1)

    def stop_consuming(self):
        """Stop consuming messages"""
        self._consuming = False

    def close(self):
        """Close connection"""
        self._is_connected = False
        self._consuming = False

    # Test helper methods

    def get_message_history(self) -> List[MockMessage]:
        """Get all published messages (for test verification)"""
        with self._lock:
            return list(self._message_history)

    def clear_message_history(self):
        """Clear message history"""
        with self._lock:
            self._message_history.clear()

    def get_subscriber_count(self, topic_pattern: str) -> int:
        """Get number of subscribers for a topic pattern"""
        with self._lock:
            return len(self._subscribers.get(topic_pattern, []))


@pytest.fixture
def mock_mq_client():
    """
    Fixture providing a MockMessageQueueClient instance

    Usage:
        def test_something(mock_mq_client):
            mock_mq_client.publish("test.topic", {"key": "value"}, "test")
    """
    client = MockMessageQueueClient()
    yield client
    client.close()


@pytest.fixture
def patch_message_queue():
    """
    Fixture that patches MessageQueueClient globally with MockMessageQueueClient

    Usage:
        def test_something(patch_message_queue):
            # Any code that imports MessageQueueClient will get MockMessageQueueClient
            from common.message_queue import MessageQueueClient
            client = MessageQueueClient()  # Returns MockMessageQueueClient
    """
    mock_client = MockMessageQueueClient()

    with patch('src.common.message_queue.MessageQueueClient', return_value=mock_client):
        yield mock_client

    mock_client.close()


@pytest.fixture
def mock_message_queue_factory():
    """
    Fixture providing a factory for creating mock MQ clients

    Usage:
        def test_something(mock_message_queue_factory):
            client1 = mock_message_queue_factory()
            client2 = mock_message_queue_factory()
    """
    clients = []

    def factory(**kwargs):
        client = MockMessageQueueClient(**kwargs)
        clients.append(client)
        return client

    yield factory

    # Cleanup
    for client in clients:
        client.close()


# NVIS Test Helpers

@pytest.fixture
def mock_sounder_metadata():
    """Fixture providing sample SounderMetadata for tests"""
    from src.ingestion.nvis.protocol_adapters.base_adapter import SounderMetadata

    return SounderMetadata(
        sounder_id='TEST_001',
        name='Test Sounder',
        operator='Test Operator',
        location='Test Location',
        latitude=40.0,
        longitude=-105.0,
        altitude=1500.0,
        equipment_type='professional',
        calibration_status='calibrated'
    )


@pytest.fixture
def mock_nvis_measurement():
    """Fixture providing sample NVISMeasurement for tests"""
    from src.ingestion.nvis.protocol_adapters.base_adapter import NVISMeasurement

    return NVISMeasurement(
        tx_latitude=40.0,
        tx_longitude=-105.0,
        tx_altitude=1500.0,
        rx_latitude=40.5,
        rx_longitude=-104.5,
        rx_altitude=1600.0,
        frequency=7.5,
        elevation_angle=85.0,
        azimuth=45.0,
        hop_distance=75.0,
        signal_strength=-85.0,
        group_delay=2.5,
        snr=20.0,
        signal_strength_error=2.0,
        group_delay_error=0.1,
        sounder_id='TEST_001',
        timestamp=datetime.utcnow().isoformat() + 'Z',
        is_o_mode=True
    )


@pytest.fixture
def multi_tier_sounders():
    """Create simulated multi-tier sounder network for testing"""
    from src.ingestion.nvis.protocol_adapters.base_adapter import SounderMetadata

    sounders = []

    # 2 PLATINUM sounders (professional research stations)
    for i in range(2):
        sounders.append(SounderMetadata(
            sounder_id=f'PLAT_{i+1:03d}',
            name=f'Professional Station {i+1}',
            operator='National Research Institute',
            location=f'Research Site {i+1}',
            latitude=35.0 + i * 5.0,
            longitude=-105.0 + i * 10.0,
            altitude=1500.0,
            equipment_type='professional',
            calibration_status='calibrated'
        ))

    # 5 GOLD sounders (university stations)
    for i in range(5):
        sounders.append(SounderMetadata(
            sounder_id=f'GOLD_{i+1:03d}',
            name=f'University Station {i+1}',
            operator='University Network',
            location=f'Campus {i+1}',
            latitude=40.0 + i * 3.0,
            longitude=-100.0 + i * 5.0,
            altitude=1200.0,
            equipment_type='research',
            calibration_status='calibrated'
        ))

    # 10 SILVER sounders (amateur club stations)
    for i in range(10):
        sounders.append(SounderMetadata(
            sounder_id=f'SILV_{i+1:03d}',
            name=f'Club Station {i+1}',
            operator='Ham Radio Club',
            location=f'Club Site {i+1}',
            latitude=42.0 + i * 2.0,
            longitude=-95.0 + i * 3.0,
            altitude=800.0,
            equipment_type='amateur_advanced',
            calibration_status='self_calibrated'
        ))

    # 20 BRONZE sounders (individual amateur stations)
    for i in range(20):
        sounders.append(SounderMetadata(
            sounder_id=f'BRON_{i+1:03d}',
            name=f'Amateur Station {i+1}',
            operator=f'Individual Operator {i+1}',
            location=f'Home QTH {i+1}',
            latitude=38.0 + i * 1.0,
            longitude=-90.0 + i * 2.0,
            altitude=500.0,
            equipment_type='amateur_basic',
            calibration_status='uncalibrated'
        ))

    return sounders


# Async test helpers

@pytest.fixture
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# Test configuration fixtures

@pytest.fixture
def test_config():
    """Create test configuration with AutoNVISConfig"""
    from src.common.config import (
        AutoNVISConfig, NVISIngestionConfig, NVISQualityTierConfig,
        ServiceConfig, GridConfig
    )

    nvis_config = NVISIngestionConfig(
        platinum=NVISQualityTierConfig(signal_error_db=2.0, delay_error_ms=0.1),
        gold=NVISQualityTierConfig(signal_error_db=4.0, delay_error_ms=0.5),
        silver=NVISQualityTierConfig(signal_error_db=8.0, delay_error_ms=2.0),
        bronze=NVISQualityTierConfig(signal_error_db=15.0, delay_error_ms=5.0),
        window_seconds=60,
        rate_threshold=60
    )

    return AutoNVISConfig(
        nvis_ingestion=nvis_config,
        services=ServiceConfig(),
        grid=GridConfig()
    )


# Helper for checking RabbitMQ availability (used for optional integration tests)

def rabbitmq_available() -> bool:
    """Check if RabbitMQ is available for integration tests"""
    try:
        from src.common.message_queue import MessageQueueClient
        client = MessageQueueClient()
        client.close()
        return True
    except Exception:
        return False


# Markers for conditional test execution

def pytest_configure(config):
    """Register custom markers"""
    config.addinivalue_line(
        "markers", "requires_rabbitmq: mark test as requiring RabbitMQ"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests that require RabbitMQ if not available"""
    if not rabbitmq_available():
        skip_rabbitmq = pytest.mark.skip(reason="RabbitMQ not available")
        for item in items:
            if "requires_rabbitmq" in item.keywords:
                item.add_marker(skip_rabbitmq)
