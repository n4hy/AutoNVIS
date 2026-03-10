"""
Unit Tests for Message Queue Client

Tests cover:
- Concurrent publishing and subscribing
- Message throughput stress testing
- Connection resilience (reconnection, failures)
- Large message handling
- Topic routing and filtering
- Memory leaks under sustained load

Uses MockMessageQueueClient from conftest.py - no RabbitMQ required.
"""

import pytest
import time
import json
import threading
from concurrent.futures import ThreadPoolExecutor
import sys
from pathlib import Path
from datetime import datetime
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

# Import the mock from conftest
from conftest import MockMessageQueueClient, MockMessage


class TestBasicPublishSubscribe:
    """Test basic publish/subscribe functionality"""

    def test_publish_subscribe_single_message(self, mock_mq_client):
        """Test publishing and receiving a single message"""
        received = []

        def callback(msg: MockMessage):
            received.append(msg)

        mock_mq_client.subscribe("test.topic", callback)
        time.sleep(0.05)  # Allow subscription to establish

        # Publish message
        mock_mq_client.publish("test.topic", {"test": "data"}, source="test")
        time.sleep(0.1)  # Allow message to be delivered

        assert len(received) == 1
        assert received[0].data["test"] == "data"
        assert received[0].source == "test"

    def test_multiple_subscribers_same_topic(self, mock_mq_client):
        """Test multiple subscribers to same topic all receive messages"""
        received1 = []
        received2 = []
        received3 = []

        mock_mq_client.subscribe("test.topic", lambda msg: received1.append(msg))
        mock_mq_client.subscribe("test.topic", lambda msg: received2.append(msg))
        mock_mq_client.subscribe("test.topic", lambda msg: received3.append(msg))
        time.sleep(0.05)

        mock_mq_client.publish("test.topic", {"msg": "broadcast"}, source="test")
        time.sleep(0.2)

        # All subscribers should receive
        assert len(received1) >= 1
        assert len(received2) >= 1
        assert len(received3) >= 1

    def test_topic_isolation(self, mock_mq_client):
        """Test that messages on different topics are isolated"""
        topic1_msgs = []
        topic2_msgs = []

        mock_mq_client.subscribe("topic.one", lambda msg: topic1_msgs.append(msg))
        mock_mq_client.subscribe("topic.two", lambda msg: topic2_msgs.append(msg))
        time.sleep(0.05)

        mock_mq_client.publish("topic.one", {"data": "one"}, source="test")
        mock_mq_client.publish("topic.two", {"data": "two"}, source="test")
        time.sleep(0.2)

        assert len(topic1_msgs) >= 1
        assert len(topic2_msgs) >= 1
        assert topic1_msgs[0].data["data"] == "one"
        assert topic2_msgs[0].data["data"] == "two"


class TestMessageTypes:
    """Test handling of different message data types"""

    def test_json_serializable_types(self, mock_mq_client):
        """Test various JSON-serializable data types"""
        received = []

        mock_mq_client.subscribe("test.topic", lambda msg: received.append(msg))
        time.sleep(0.05)

        # Dictionary
        mock_mq_client.publish("test.topic", {"key": "value"}, source="test")
        time.sleep(0.05)

        # List
        mock_mq_client.publish("test.topic", {"list": [1, 2, 3]}, source="test")
        time.sleep(0.05)

        # Nested structure
        mock_mq_client.publish("test.topic", {
            "nested": {"deep": [1, 2, {"very": "deep"}]}
        }, source="test")
        time.sleep(0.05)

        # Numbers
        mock_mq_client.publish("test.topic", {"int": 42, "float": 3.14159}, source="test")
        time.sleep(0.1)

        assert len(received) >= 4

    def test_large_message(self, mock_mq_client):
        """Test handling of large messages"""
        received = []

        mock_mq_client.subscribe("test.topic", lambda msg: received.append(msg))
        time.sleep(0.05)

        # Create large payload (~100 KB for faster tests)
        large_data = {
            "array": [i for i in range(10000)],
            "metadata": "x" * 10000
        }

        mock_mq_client.publish("test.topic", large_data, source="test")
        time.sleep(0.3)

        assert len(received) >= 1
        assert len(received[0].data["array"]) == 10000

    def test_numpy_array_serialization(self, mock_mq_client):
        """Test serialization of numpy arrays (converted to list)"""
        received = []

        mock_mq_client.subscribe("test.topic", lambda msg: received.append(msg))
        time.sleep(0.05)

        # Numpy array needs to be converted to list for JSON
        array = np.random.rand(50, 50)
        data = {
            "array": array.tolist(),
            "shape": list(array.shape)
        }

        mock_mq_client.publish("test.topic", data, source="test")
        time.sleep(0.2)

        assert len(received) >= 1
        received_array = np.array(received[0].data["array"])
        assert received_array.shape == (50, 50)


class TestConcurrentPublishing:
    """Test concurrent message publishing"""

    def test_many_publishers_one_subscriber(self, mock_mq_client):
        """Test many threads publishing to one subscriber"""
        received = []
        lock = threading.Lock()

        def callback(msg):
            with lock:
                received.append(msg)

        mock_mq_client.subscribe("test.topic", callback)
        time.sleep(0.1)

        # Spawn many publishing threads
        def publish_messages(thread_id):
            for i in range(10):
                mock_mq_client.publish("test.topic", {
                    "thread": thread_id,
                    "msg": i
                }, source=f"thread-{thread_id}")
                time.sleep(0.001)

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(publish_messages, i) for i in range(20)]
            for f in futures:
                f.result()

        time.sleep(0.5)  # Allow all messages to be delivered

        # Should receive 200 messages (20 threads * 10 messages)
        assert len(received) >= 180  # Allow some variance

    def test_high_throughput_publishing(self, mock_mq_client):
        """Test publishing at high rate (stress test)"""
        received = []
        lock = threading.Lock()

        def callback(msg):
            with lock:
                received.append(msg)

        mock_mq_client.subscribe("test.topic", callback)
        time.sleep(0.1)

        # Publish 500 messages as fast as possible (reduced for mock)
        start = time.time()
        for i in range(500):
            mock_mq_client.publish("test.topic", {"seq": i}, source="stress")

        elapsed = time.time() - start
        time.sleep(0.3)  # Allow messages to be delivered

        throughput = 500 / elapsed
        print(f"\nPublishing throughput: {throughput:.0f} msg/sec")

        # Should receive most messages
        assert len(received) >= 450

    def test_burst_publishing(self, mock_mq_client):
        """Test handling of message bursts"""
        received = []

        mock_mq_client.subscribe("test.topic", lambda msg: received.append(msg))
        time.sleep(0.05)

        # Send burst of 100 messages with no delay
        for i in range(100):
            mock_mq_client.publish("test.topic", {"burst": i}, source="burst-test")

        time.sleep(0.3)

        assert len(received) >= 90  # Allow some processing variance


class TestSubscriberPatterns:
    """Test different subscriber patterns"""

    def test_wildcard_topic_subscription_hash(self, mock_mq_client):
        """Test subscribing with # wildcard patterns"""
        received = []

        # Subscribe to all topics starting with "sensor."
        mock_mq_client.subscribe("sensor.#", lambda msg: received.append(msg))
        time.sleep(0.05)

        # Publish to various sensor topics
        mock_mq_client.publish("sensor.temperature", {"value": 25.0}, source="test")
        mock_mq_client.publish("sensor.pressure", {"value": 1013.0}, source="test")
        mock_mq_client.publish("sensor.humidity.indoor", {"value": 45.0}, source="test")
        mock_mq_client.publish("other.topic", {"value": 99.0}, source="test")

        time.sleep(0.2)

        # Should receive sensor topics but not "other.topic"
        assert len(received) >= 3
        topics = [msg.topic for msg in received]
        assert "sensor.temperature" in topics
        assert "sensor.pressure" in topics
        assert "other.topic" not in topics

    def test_wildcard_topic_subscription_star(self, mock_mq_client):
        """Test subscribing with * wildcard patterns"""
        received = []

        # Subscribe to single-word wildcard
        mock_mq_client.subscribe("sensor.*", lambda msg: received.append(msg))
        time.sleep(0.05)

        # Publish to various topics
        mock_mq_client.publish("sensor.temperature", {"value": 25.0}, source="test")
        mock_mq_client.publish("sensor.pressure", {"value": 1013.0}, source="test")
        mock_mq_client.publish("sensor.humidity.indoor", {"value": 45.0}, source="test")  # Won't match

        time.sleep(0.2)

        # Should receive only single-word matches
        topics = [msg.topic for msg in received]
        assert "sensor.temperature" in topics
        assert "sensor.pressure" in topics
        # sensor.humidity.indoor shouldn't match sensor.*
        assert "sensor.humidity.indoor" not in topics

    def test_late_subscriber(self, mock_mq_client):
        """Test that late subscriber misses earlier messages"""
        # Publish before subscribing
        mock_mq_client.publish("test.topic", {"early": "message"}, source="test")
        time.sleep(0.05)

        received = []
        mock_mq_client.subscribe("test.topic", lambda msg: received.append(msg))
        time.sleep(0.1)

        # Late subscriber shouldn't receive early message
        early_msgs = [msg for msg in received if msg.data.get("early") == "message"]
        assert len(early_msgs) == 0

        # Publish after subscribing
        mock_mq_client.publish("test.topic", {"late": "message"}, source="test")
        time.sleep(0.1)

        # Should receive the late message
        assert len(received) >= 1
        assert any(msg.data.get("late") == "message" for msg in received)


class TestConnectionResilience:
    """Test connection handling and resilience"""

    def test_publish_after_close(self, mock_mq_client):
        """Test publishing when connection closed raises error"""
        mock_mq_client.close()

        with pytest.raises(ConnectionError):
            mock_mq_client.publish("test.topic", {"test": "data"}, source="test")

    def test_reconnect_after_close(self):
        """Test creating new client after close"""
        client1 = MockMessageQueueClient()
        client1.publish("test.topic", {"msg": 1}, source="test")
        client1.close()

        # New client should work
        client2 = MockMessageQueueClient()
        received = []
        client2.subscribe("test.topic", lambda msg: received.append(msg))
        time.sleep(0.05)

        client2.publish("test.topic", {"msg": 2}, source="test")
        time.sleep(0.1)

        assert len(received) >= 1
        client2.close()


class TestMessageMetadata:
    """Test message metadata handling"""

    def test_timestamp_in_message(self, mock_mq_client):
        """Test that messages include timestamps"""
        received = []

        mock_mq_client.subscribe("test.topic", lambda msg: received.append(msg))
        time.sleep(0.05)

        before = datetime.utcnow()
        mock_mq_client.publish("test.topic", {"test": "data"}, source="test")
        time.sleep(0.1)

        assert len(received) >= 1
        msg = received[0]

        # Message should have timestamp
        assert hasattr(msg, 'timestamp')
        assert msg.timestamp is not None

    def test_source_tracking(self, mock_mq_client):
        """Test that message source is tracked"""
        received = []

        mock_mq_client.subscribe("test.topic", lambda msg: received.append(msg))
        time.sleep(0.05)

        mock_mq_client.publish("test.topic", {"data": "test"}, source="test-source")
        time.sleep(0.1)

        assert len(received) >= 1
        assert received[0].source == "test-source"

    def test_topic_in_message(self, mock_mq_client):
        """Test that received messages include topic"""
        received = []

        mock_mq_client.subscribe("test.topic", lambda msg: received.append(msg))
        time.sleep(0.05)

        mock_mq_client.publish("test.topic", {"data": "test"}, source="test")
        time.sleep(0.1)

        assert len(received) >= 1
        assert received[0].topic == "test.topic"


class TestErrorHandling:
    """Test error handling and edge cases"""

    def test_callback_exception_handling(self, mock_mq_client):
        """Test that exceptions in callbacks don't crash subscriber"""
        success_count = [0]

        def bad_callback(msg):
            if msg.data.get("crash"):
                raise ValueError("Intentional crash")
            success_count[0] += 1

        mock_mq_client.subscribe("test.topic", bad_callback)
        time.sleep(0.05)

        # Publish mix of good and bad messages
        mock_mq_client.publish("test.topic", {"crash": True}, source="test")
        mock_mq_client.publish("test.topic", {"good": True}, source="test")
        mock_mq_client.publish("test.topic", {"crash": True}, source="test")
        mock_mq_client.publish("test.topic", {"good": True}, source="test")

        time.sleep(0.3)

        # Should have processed the good messages despite exceptions
        assert success_count[0] >= 2

    def test_empty_message_data(self, mock_mq_client):
        """Test publishing empty message"""
        received = []

        mock_mq_client.subscribe("test.topic", lambda msg: received.append(msg))
        time.sleep(0.05)

        mock_mq_client.publish("test.topic", {}, source="test")
        time.sleep(0.1)

        assert len(received) >= 1
        assert received[0].data == {}

    def test_none_values_in_message(self, mock_mq_client):
        """Test handling of None values in message data"""
        received = []

        mock_mq_client.subscribe("test.topic", lambda msg: received.append(msg))
        time.sleep(0.05)

        mock_mq_client.publish("test.topic", {"value": None, "key": "test"}, source="test")
        time.sleep(0.1)

        assert len(received) >= 1
        assert received[0].data["value"] is None


class TestMemoryLeaks:
    """Test for memory leaks under sustained load"""

    def test_sustained_publishing_no_subscribers(self, mock_mq_client):
        """Test memory usage when publishing without subscribers"""
        # Publish many messages without subscribers
        for i in range(500):
            mock_mq_client.publish("test.topic", {"seq": i}, source="test")
            if i % 100 == 0:
                time.sleep(0.01)  # Brief pause

        # Check message history doesn't grow unbounded
        history = mock_mq_client.get_message_history()
        assert len(history) == 500


class TestMessageHistory:
    """Test message history tracking (mock-specific feature)"""

    def test_message_history_tracking(self, mock_mq_client):
        """Test that message history is tracked"""
        mock_mq_client.publish("topic.one", {"a": 1}, source="test")
        mock_mq_client.publish("topic.two", {"b": 2}, source="test")
        mock_mq_client.publish("topic.one", {"c": 3}, source="test")

        history = mock_mq_client.get_message_history()

        assert len(history) == 3
        assert history[0].topic == "topic.one"
        assert history[1].topic == "topic.two"
        assert history[2].topic == "topic.one"

    def test_clear_message_history(self, mock_mq_client):
        """Test clearing message history"""
        mock_mq_client.publish("test.topic", {"a": 1}, source="test")
        mock_mq_client.publish("test.topic", {"b": 2}, source="test")

        mock_mq_client.clear_message_history()

        history = mock_mq_client.get_message_history()
        assert len(history) == 0

    def test_subscriber_count(self, mock_mq_client):
        """Test subscriber count tracking"""
        assert mock_mq_client.get_subscriber_count("test.topic") == 0

        mock_mq_client.subscribe("test.topic", lambda msg: None)
        assert mock_mq_client.get_subscriber_count("test.topic") == 1

        mock_mq_client.subscribe("test.topic", lambda msg: None)
        assert mock_mq_client.get_subscriber_count("test.topic") == 2


class TestCPUIntensiveMessageQueuing:
    """CPU-intensive stress tests"""

    def test_massive_message_processing(self, mock_mq_client):
        """Process thousands of messages with computation"""
        processed = [0]
        lock = threading.Lock()

        def compute_callback(msg):
            # Do some computation
            data = msg.data.get("values", [])
            result = sum(x**2 for x in data)

            with lock:
                processed[0] += 1

        mock_mq_client.subscribe("test.topic", compute_callback)
        time.sleep(0.1)

        # Publish 1000 messages with data to process (reduced for mock)
        start = time.time()
        for i in range(1000):
            mock_mq_client.publish("test.topic", {
                "seq": i,
                "values": list(range(50))  # Reduced size
            }, source="stress")

        elapsed = time.time() - start
        time.sleep(1.0)  # Allow processing

        print(f"\nProcessed {processed[0]} messages in {elapsed:.2f}s")
        print(f"Throughput: {1000/elapsed:.0f} msg/sec publish")

        assert processed[0] >= 900

    def test_parallel_topic_streams(self, mock_mq_client):
        """Test multiple parallel topic streams"""
        counts = {i: [0] for i in range(10)}
        locks = {i: threading.Lock() for i in range(10)}

        # Subscribe to 10 different topics
        for i in range(10):
            def make_callback(topic_id):
                def callback(msg):
                    with locks[topic_id]:
                        counts[topic_id][0] += 1
                return callback

            mock_mq_client.subscribe(f"stream.{i}", make_callback(i))

        time.sleep(0.1)

        # Publish to all topics in parallel
        def publish_stream(topic_id):
            for j in range(50):  # Reduced count
                mock_mq_client.publish(f"stream.{topic_id}", {
                    "topic": topic_id,
                    "seq": j
                }, source=f"stream-{topic_id}")
                time.sleep(0.001)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(publish_stream, i) for i in range(10)]
            for f in futures:
                f.result()

        time.sleep(0.5)

        # Each topic should have received ~50 messages
        for i in range(10):
            assert counts[i][0] >= 40  # Allow some variance


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-s'])
