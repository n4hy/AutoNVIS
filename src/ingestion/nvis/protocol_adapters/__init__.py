"""
Protocol Adapters for NVIS Sounder Data Ingestion

Supports multiple ingestion protocols:
- TCP: Real-time socket streams
- HTTP: REST API submissions
- MQTT: IoT message broker
- Email: Low-rate amateur sounder submissions
"""

from .base_adapter import BaseAdapter, NVISMeasurement
from .tcp_adapter import TCPAdapter
from .http_adapter import HTTPAdapter
from .mqtt_adapter import MQTTAdapter
from .email_adapter import EmailAdapter

__all__ = [
    'BaseAdapter',
    'NVISMeasurement',
    'TCPAdapter',
    'HTTPAdapter',
    'MQTTAdapter',
    'EmailAdapter'
]
