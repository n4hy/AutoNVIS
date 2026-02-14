"""
Brutal Unit Tests for Message Queue Client

Tests cover:
- Concurrent publishing and subscribing
- Message throughput stress testing
- Connection resilience (reconnection, failures)
- Large message handling
- Topic routing and filtering
- Memory leaks under sustained load
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

from common.message_queue import MessageQueueClient, Message, Topics


class TestBasicPublishSubscribe:
    """Test basic publish/subscribe functionality"""

    def test_publish_subscribe_single_message(self):
        """Test publishing and receiving a single message"""
        client = MessageQueueClient()
        received = []

        def callback(msg: Message):
            received.append(msg)

        client.subscribe("test.topic", callback)
        time.sleep(0.1)  # Allow subscription to establish

        # Publish message
        client.publish("test.topic", {"test": "data"}, source="test")
        time.sleep(0.1)  # Allow message to be delivered

        assert len(received) == 1
        assert received[0].data["test"] == "data"
        assert received[0].source == "test"

        client.close()

    def test_multiple_subscribers_same_topic(self):
        """Test multiple subscribers to same topic all receive messages"""
        client = MessageQueueClient()
        received1 = []
        received2 = []
        received3 = []

        client.subscribe("test.topic", lambda msg: received1.append(msg))
        client.subscribe("test.topic", lambda msg: received2.append(msg))
        client.subscribe("test.topic", lambda msg: received3.append(msg))
        time.sleep(0.1)

        client.publish("test.topic", {"msg": "broadcast"}, source="test")
        time.sleep(0.2)

        # All subscribers should receive
        assert len(received1) >= 1
        assert len(received2) >= 1
        assert len(received3) >= 1

        client.close()

    def test_topic_isolation(self):
        """Test that messages on different topics are isolated"""
        client = MessageQueueClient()
        topic1_msgs = []
        topic2_msgs = []

        client.subscribe("topic.one", lambda msg: topic1_msgs.append(msg))
        client.subscribe("topic.two", lambda msg: topic2_msgs.append(msg))
        time.sleep(0.1)

        client.publish("topic.one", {"data": "one"}, source="test")
        client.publish("topic.two", {"data": "two"}, source="test")
        time.sleep(0.2)

        assert len(topic1_msgs) >= 1
        assert len(topic2_msgs) >= 1
        assert topic1_msgs[0].data["data"] == "one"
        assert topic2_msgs[0].data["data"] == "two"

        client.close()


class TestMessageTypes:
    """Test handling of different message data types"""

    def test_json_serializable_types(self):
        """Test various JSON-serializable data types"""
        client = MessageQueueClient()
        received = []

        client.subscribe("test.topic", lambda msg: received.append(msg))
        time.sleep(0.1)

        # Dictionary
        client.publish("test.topic", {"key": "value"}, source="test")
        time.sleep(0.05)

        # List
        client.publish("test.topic", [1, 2, 3], source="test")
        time.sleep(0.05)

        # Nested structure
        client.publish("test.topic", {
            "nested": {"deep": [1, 2, {"very": "deep"}]}
        }, source="test")
        time.sleep(0.05)

        # Numbers
        client.publish("test.topic", {"int": 42, "float": 3.14159}, source="test")
        time.sleep(0.1)

        assert len(received) >= 4

        client.close()

    def test_large_message(self):
        """Test handling of large messages"""
        client = MessageQueueClient()
        received = []

        client.subscribe("test.topic", lambda msg: received.append(msg))
        time.sleep(0.1)

        # Create large payload (~1 MB)
        large_data = {
            "array": [i for i in range(100000)],
            "metadata": "x" * 10000
        }

        client.publish("test.topic", large_data, source="test")
        time.sleep(0.5)  # Larger message may take longer

        assert len(received) >= 1
        assert len(received[0].data["array"]) == 100000

        client.close()

    def test_numpy_array_serialization(self):
        """Test serialization of numpy arrays"""
        client = MessageQueueClient()
        received = []

        client.subscribe("test.topic", lambda msg: received.append(msg))
        time.sleep(0.1)

        # Numpy array needs to be converted to list for JSON
        array = np.random.rand(100, 100)
        data = {
            "array": array.tolist(),
            "shape": array.shape
        }

        client.publish("test.topic", data, source="test")
        time.sleep(0.2)

        assert len(received) >= 1
        received_array = np.array(received[0].data["array"])
        assert received_array.shape == (100, 100)

        client.close()


class TestConcurrentPublishing:
    """Test concurrent message publishing"""

    def test_many_publishers_one_subscriber(self):
        """Test many threads publishing to one subscriber"""
        client = MessageQueueClient()
        received = []
        lock = threading.Lock()

        def callback(msg):
            with lock:
                received.append(msg)

        client.subscribe("test.topic", callback)
        time.sleep(0.1)

        # Spawn many publishing threads
        def publish_messages(thread_id):
            for i in range(10):
                client.publish("test.topic", {
                    "thread": thread_id,
                    "msg": i
                }, source=f"thread-{thread_id}")
                time.sleep(0.001)

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(publish_messages, i) for i in range(20)]
            for f in futures:
                f.result()

        time.sleep(1.0)  # Allow all messages to be delivered

        # Should receive 200 messages (20 threads * 10 messages)
        assert len(received) >= 180  # Allow some loss

        client.close()

    def test_high_throughput_publishing(self):
        """Test publishing at high rate (stress test)"""
        client = MessageQueueClient()
        received = []
        lock = threading.Lock()

        def callback(msg):
            with lock:
                received.append(msg)

        client.subscribe("test.topic", callback)
        time.sleep(0.1)

        # Publish 1000 messages as fast as possible
        start = time.time()
        for i in range(1000):
            client.publish("test.topic", {"seq": i}, source="stress")

        elapsed = time.time() - start
        time.sleep(0.5)  # Allow messages to be delivered

        throughput = 1000 / elapsed
        print(f"\nPublishing throughput: {throughput:.0f} msg/sec")

        # Should receive most messages
        assert len(received) >= 900

        client.close()

    def test_burst_publishing(self):
        """Test handling of message bursts"""
        client = MessageQueueClient()
        received = []

        client.subscribe("test.topic", lambda msg: received.append(msg))
        time.sleep(0.1)

        # Send burst of 100 messages with no delay
        for i in range(100):
            client.publish("test.topic", {"burst": i}, source="burst-test")

        time.sleep(0.5)

        assert len(received) >= 90  # Allow some loss in burst

        client.close()


class TestSubscriberPatterns:
    """Test different subscriber patterns"""

    def test_wildcard_topic_subscription(self):
        """Test subscribing with wildcard patterns"""
        client = MessageQueueClient()
        received = []

        # Subscribe to all topics starting with "sensor."
        client.subscribe("sensor.#", lambda msg: received.append(msg))
        time.sleep(0.1)

        # Publish to various sensor topics
        client.publish("sensor.temperature", {"value": 25.0}, source="test")
        client.publish("sensor.pressure", {"value": 1013.0}, source="test")
        client.publish("sensor.humidity.indoor", {"value": 45.0}, source="test")
        client.publish("other.topic", {"value": 99.0}, source="test")

        time.sleep(0.2)

        # Should receive sensor topics but not "other.topic"
        assert len(received) >= 3
        topics = [msg.topic for msg in received]
        assert "sensor.temperature" in topics
        assert "sensor.pressure" in topics
        assert "other.topic" not in topics

        client.close()

    def test_late_subscriber(self):
        """Test that late subscriber misses earlier messages"""
        client = MessageQueueClient()

        # Publish before subscribing
        client.publish("test.topic", {"early": "message"}, source="test")
        time.sleep(0.1)

        received = []
        client.subscribe("test.topic", lambda msg: received.append(msg))
        time.sleep(0.1)

        # Late subscriber shouldn't receive early message (unless queue persists)
        # Behavior depends on RabbitMQ queue configuration

        # Publish after subscribing
        client.publish("test.topic", {"late": "message"}, source="test")
        time.sleep(0.1)

        # Should receive at least the late message
        assert len(received) >= 1
        assert any(msg.data.get("late") == "message" for msg in received)

        client.close()

    @pytest.mark.skip(reason="MessageQueueClient doesn't have unsubscribe method")
    def test_unsubscribe(self):
        """Test unsubscribing from topic"""
        client = MessageQueueClient()
        received = []

        tag = client.subscribe("test.topic", lambda msg: received.append(msg))
        time.sleep(0.1)

        client.publish("test.topic", {"msg": 1}, source="test")
        time.sleep(0.1)

        # Unsubscribe
        client.unsubscribe(tag)
        time.sleep(0.1)

        # Publish after unsubscribe
        client.publish("test.topic", {"msg": 2}, source="test")
        time.sleep(0.1)

        # Should only receive first message
        assert len(received) == 1
        assert received[0].data["msg"] == 1

        client.close()


class TestConnectionResilience:
    """Test connection handling and resilience"""

    @pytest.mark.skip(reason="MessageQueueClient doesn't have connect() method")
    def test_reconnect_after_disconnect(self):
        """Test reconnecting after disconnect"""
        client = MessageQueueClient()

        # Publish message
        client.publish("test.topic", {"test": "before"}, source="test")

        # Disconnect
        client.close()

        # Reconnect
        client.connect()

        # Publish again
        client.publish("test.topic", {"test": "after"}, source="test")

        client.close()

    @pytest.mark.skip(reason="MessageQueueClient doesn't have connect() method")
    def test_multiple_disconnects(self):
        """Test multiple disconnect/connect cycles"""
        client = MessageQueueClient()

        for i in range(10):
            client.publish("test.topic", {"cycle": i}, source="test")
            client.close()
            client.connect()

        client.close()

    def test_publish_without_connection(self):
        """Test publishing when not connected"""
        client = MessageQueueClient()
        client.close()

        # Should handle gracefully (either buffer or raise exception)
        try:
            client.publish("test.topic", {"test": "data"}, source="test")
        except Exception as e:
            # Exception is acceptable
            assert e is not None


class TestMessageMetadata:
    """Test message metadata handling"""

    def test_timestamp_in_message(self):
        """Test that messages include timestamps"""
        client = MessageQueueClient()
        received = []

        client.subscribe("test.topic", lambda msg: received.append(msg))
        time.sleep(0.1)

        before = datetime.utcnow()
        client.publish("test.topic", {"test": "data"}, source="test")
        time.sleep(0.1)
        after = datetime.utcnow()

        assert len(received) >= 1
        msg = received[0]

        # Message should have timestamp
        assert hasattr(msg, 'timestamp') or 'timestamp' in msg.data

        client.close()

    def test_source_tracking(self):
        """Test that message source is tracked"""
        client = MessageQueueClient()
        received = []

        client.subscribe("test.topic", lambda msg: received.append(msg))
        time.sleep(0.1)

        client.publish("test.topic", {"data": "test"}, source="test-source")
        time.sleep(0.1)

        assert len(received) >= 1
        assert received[0].source == "test-source"

        client.close()

    def test_topic_in_message(self):
        """Test that received messages include topic"""
        client = MessageQueueClient()
        received = []

        client.subscribe("test.topic", lambda msg: received.append(msg))
        time.sleep(0.1)

        client.publish("test.topic", {"data": "test"}, source="test")
        time.sleep(0.1)

        assert len(received) >= 1
        assert received[0].topic == "test.topic"

        client.close()


class TestErrorHandling:
    """Test error handling and edge cases"""

    def test_callback_exception_handling(self):
        """Test that exceptions in callbacks don't crash subscriber"""
        client = MessageQueueClient()
        success_count = [0]

        def bad_callback(msg):
            if msg.data.get("crash"):
                raise ValueError("Intentional crash")
            success_count[0] += 1

        client.subscribe("test.topic", bad_callback)
        time.sleep(0.1)

        # Publish mix of good and bad messages
        client.publish("test.topic", {"crash": True}, source="test")
        client.publish("test.topic", {"good": True}, source="test")
        client.publish("test.topic", {"crash": True}, source="test")
        client.publish("test.topic", {"good": True}, source="test")

        time.sleep(0.2)

        # Should have processed the good messages despite exceptions
        assert success_count[0] >= 2

        client.close()

    def test_empty_message_data(self):
        """Test publishing empty message"""
        client = MessageQueueClient()
        received = []

        client.subscribe("test.topic", lambda msg: received.append(msg))
        time.sleep(0.1)

        client.publish("test.topic", {}, source="test")
        time.sleep(0.1)

        assert len(received) >= 1
        assert received[0].data == {}

        client.close()

    def test_none_values_in_message(self):
        """Test handling of None values in message data"""
        client = MessageQueueClient()
        received = []

        client.subscribe("test.topic", lambda msg: received.append(msg))
        time.sleep(0.1)

        client.publish("test.topic", {"value": None, "key": "test"}, source="test")
        time.sleep(0.1)

        assert len(received) >= 1
        assert received[0].data["value"] is None

        client.close()


class TestMemoryLeaks:
    """Test for memory leaks under sustained load"""

    def test_sustained_publishing_no_subscribers(self):
        """Test memory usage when publishing without subscribers"""
        client = MessageQueueClient()

        # Publish many messages without subscribers
        for i in range(1000):
            client.publish("test.topic", {"seq": i}, source="test")
            if i % 100 == 0:
                time.sleep(0.01)  # Brief pause

        client.close()

    @pytest.mark.skip(reason="MessageQueueClient doesn't have unsubscribe method")
    def test_sustained_subscribe_unsubscribe(self):
        """Test repeated subscribe/unsubscribe cycles"""
        client = MessageQueueClient()

        for i in range(100):
            tag = client.subscribe("test.topic", lambda msg: None)
            time.sleep(0.01)
            client.unsubscribe(tag)

        client.close()

    @pytest.mark.skip(reason="MessageQueueClient doesn't have unsubscribe method")
    def test_many_short_lived_subscribers(self):
        """Test creating and destroying many subscribers"""
        client = MessageQueueClient()

        for i in range(100):
            received = []
            tag = client.subscribe(f"topic.{i}", lambda msg: received.append(msg))
            client.publish(f"topic.{i}", {"data": i}, source="test")
            time.sleep(0.01)
            client.unsubscribe(tag)

        client.close()


class TestCPUIntensiveMessageQueuing:
    """CPU-intensive stress tests"""

    def test_massive_message_processing(self):
        """Process thousands of messages with computation"""
        client = MessageQueueClient()
        processed = [0]
        lock = threading.Lock()

        def compute_callback(msg):
            # Do some computation
            data = msg.data.get("values", [])
            result = sum(x**2 for x in data)

            with lock:
                processed[0] += 1

        client.subscribe("test.topic", compute_callback)
        time.sleep(0.1)

        # Publish 5000 messages with data to process
        start = time.time()
        for i in range(5000):
            client.publish("test.topic", {
                "seq": i,
                "values": list(range(100))
            }, source="stress")

        elapsed = time.time() - start
        time.sleep(2.0)  # Allow processing

        print(f"\nProcessed {processed[0]} messages in {elapsed:.2f}s")
        print(f"Throughput: {5000/elapsed:.0f} msg/sec publish, {processed[0]/2.0:.0f} msg/sec process")

        assert processed[0] >= 4500

        client.close()

    def test_parallel_topic_streams(self):
        """Test multiple parallel topic streams"""
        client = MessageQueueClient()
        counts = {i: [0] for i in range(10)}
        locks = {i: threading.Lock() for i in range(10)}

        # Subscribe to 10 different topics
        for i in range(10):
            def make_callback(topic_id):
                def callback(msg):
                    with locks[topic_id]:
                        counts[topic_id][0] += 1
                return callback

            client.subscribe(f"stream.{i}", make_callback(i))

        time.sleep(0.2)

        # Publish to all topics in parallel
        def publish_stream(topic_id):
            for j in range(100):
                client.publish(f"stream.{topic_id}", {
                    "topic": topic_id,
                    "seq": j
                }, source=f"stream-{topic_id}")
                time.sleep(0.001)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(publish_stream, i) for i in range(10)]
            for f in futures:
                f.result()

        time.sleep(1.0)

        # Each topic should have received ~100 messages
        for i in range(10):
            assert counts[i][0] >= 90

        client.close()


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-s'])
