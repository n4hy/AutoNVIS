"""
Centralized Logging Configuration for Auto-NVIS

This module provides structured logging with JSON formatting
for easy aggregation and analysis.
"""

import logging
import sys
import json
from datetime import datetime
from typing import Dict, Any
from pathlib import Path


class JSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON

        Args:
            record: Log record

        Returns:
            JSON string
        """
        log_data = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        # Add extra fields if present
        if hasattr(record, 'service'):
            log_data['service'] = record.service

        if hasattr(record, 'component'):
            log_data['component'] = record.component

        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        return json.dumps(log_data)


def setup_logging(
    service_name: str,
    log_level: str = "INFO",
    log_file: str = None,
    json_format: bool = True
) -> logging.Logger:
    """
    Set up logging for a service

    Args:
        service_name: Name of the service (e.g., 'ingestion', 'assimilation')
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path (None for stdout only)
        json_format: Use JSON formatting (True) or simple text (False)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(service_name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))

    if json_format:
        console_handler.setFormatter(JSONFormatter())
    else:
        simple_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(simple_format)

    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))

        if json_format:
            file_handler.setFormatter(JSONFormatter())
        else:
            file_handler.setFormatter(simple_format)

        logger.addHandler(file_handler)

    return logger


class ServiceLogger:
    """
    Wrapper for service-specific logging with extra context
    """

    def __init__(self, service_name: str, component: str = None):
        """
        Initialize service logger

        Args:
            service_name: Name of the service
            component: Optional component name within service
        """
        self.logger = logging.getLogger(service_name)
        self.service_name = service_name
        self.component = component

    def _add_context(self, extra: Dict[str, Any] = None) -> Dict[str, Any]:
        """Add service context to log extra fields"""
        context = {'service': self.service_name}
        if self.component:
            context['component'] = self.component
        if extra:
            context.update(extra)
        return context

    def debug(self, message: str, extra: Dict[str, Any] = None):
        """Log debug message"""
        self.logger.debug(message, extra=self._add_context(extra))

    def info(self, message: str, extra: Dict[str, Any] = None):
        """Log info message"""
        self.logger.info(message, extra=self._add_context(extra))

    def warning(self, message: str, extra: Dict[str, Any] = None):
        """Log warning message"""
        self.logger.warning(message, extra=self._add_context(extra))

    def error(self, message: str, extra: Dict[str, Any] = None, exc_info: bool = False):
        """Log error message"""
        self.logger.error(message, extra=self._add_context(extra), exc_info=exc_info)

    def critical(self, message: str, extra: Dict[str, Any] = None, exc_info: bool = False):
        """Log critical message"""
        self.logger.critical(message, extra=self._add_context(extra), exc_info=exc_info)


# Prometheus-compatible metrics logging
class MetricsLogger:
    """
    Log metrics in a format suitable for Prometheus scraping
    """

    def __init__(self, service_name: str):
        """
        Initialize metrics logger

        Args:
            service_name: Name of the service
        """
        self.logger = logging.getLogger(f"{service_name}.metrics")

    def log_metric(self, metric_name: str, value: float, labels: Dict[str, str] = None):
        """
        Log a metric value

        Args:
            metric_name: Name of the metric
            value: Metric value
            labels: Optional metric labels
        """
        metric_data = {
            'metric': metric_name,
            'value': value,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }

        if labels:
            metric_data['labels'] = labels

        self.logger.info(json.dumps(metric_data))

    def log_counter(self, name: str, increment: int = 1, labels: Dict[str, str] = None):
        """Log a counter increment"""
        self.log_metric(f"{name}_total", increment, labels)

    def log_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Log a gauge value"""
        self.log_metric(name, value, labels)

    def log_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """Log a histogram observation"""
        self.log_metric(f"{name}_seconds", value, labels)
