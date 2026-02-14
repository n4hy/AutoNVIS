"""
Performance Optimizations for Dashboard

Provides caching, compression, and performance utilities.
"""

import functools
import time
import gzip
import json
from typing import Any, Callable, Optional
from datetime import datetime, timedelta


class ResponseCache:
    """Simple time-based response cache"""

    def __init__(self, ttl_seconds: int = 60):
        """
        Initialize cache.

        Args:
            ttl_seconds: Time to live for cached entries
        """
        self.cache = {}
        self.ttl_seconds = ttl_seconds

    def get(self, key: str) -> Optional[Any]:
        """
        Get cached value.

        Args:
            key: Cache key

        Returns:
            Cached value or None if expired/missing
        """
        if key in self.cache:
            value, timestamp = self.cache[key]
            age = (datetime.utcnow() - timestamp).total_seconds()

            if age < self.ttl_seconds:
                return value
            else:
                # Expired, remove
                del self.cache[key]

        return None

    def set(self, key: str, value: Any):
        """
        Set cached value.

        Args:
            key: Cache key
            value: Value to cache
        """
        self.cache[key] = (value, datetime.utcnow())

    def clear(self):
        """Clear all cached entries"""
        self.cache.clear()

    def cleanup(self):
        """Remove expired entries"""
        now = datetime.utcnow()
        expired_keys = [
            key for key, (_, timestamp) in self.cache.items()
            if (now - timestamp).total_seconds() >= self.ttl_seconds
        ]

        for key in expired_keys:
            del self.cache[key]


# Global caches for different data types
grid_metadata_cache = ResponseCache(ttl_seconds=30)
parameter_map_cache = ResponseCache(ttl_seconds=60)
statistics_cache = ResponseCache(ttl_seconds=60)


def cached_response(cache: ResponseCache, key_func: Optional[Callable] = None):
    """
    Decorator for caching API responses.

    Args:
        cache: ResponseCache instance
        key_func: Function to generate cache key from args
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"

            # Try cache
            cached = cache.get(cache_key)
            if cached is not None:
                return cached

            # Execute function
            result = await func(*args, **kwargs)

            # Cache result
            cache.set(cache_key, result)

            return result

        return wrapper
    return decorator


def compress_json(data: dict) -> bytes:
    """
    Compress JSON data with gzip.

    Args:
        data: Dictionary to compress

    Returns:
        Compressed bytes
    """
    json_str = json.dumps(data)
    return gzip.compress(json_str.encode('utf-8'))


def decompress_json(compressed_data: bytes) -> dict:
    """
    Decompress gzipped JSON data.

    Args:
        compressed_data: Compressed bytes

    Returns:
        Decompressed dictionary
    """
    json_str = gzip.decompress(compressed_data).decode('utf-8')
    return json.loads(json_str)


class PerformanceMonitor:
    """Monitor API endpoint performance"""

    def __init__(self):
        self.metrics = {}

    def record(self, endpoint: str, duration: float):
        """
        Record endpoint execution time.

        Args:
            endpoint: Endpoint name
            duration: Execution time in seconds
        """
        if endpoint not in self.metrics:
            self.metrics[endpoint] = {
                'count': 0,
                'total_time': 0.0,
                'min_time': float('inf'),
                'max_time': 0.0
            }

        m = self.metrics[endpoint]
        m['count'] += 1
        m['total_time'] += duration
        m['min_time'] = min(m['min_time'], duration)
        m['max_time'] = max(m['max_time'], duration)

    def get_stats(self, endpoint: str) -> dict:
        """
        Get statistics for endpoint.

        Args:
            endpoint: Endpoint name

        Returns:
            Dictionary with statistics
        """
        if endpoint not in self.metrics:
            return {}

        m = self.metrics[endpoint]
        return {
            'count': m['count'],
            'avg_time': m['total_time'] / m['count'],
            'min_time': m['min_time'],
            'max_time': m['max_time'],
            'total_time': m['total_time']
        }

    def get_all_stats(self) -> dict:
        """Get statistics for all endpoints"""
        return {
            endpoint: self.get_stats(endpoint)
            for endpoint in self.metrics.keys()
        }


def timed_endpoint(monitor: PerformanceMonitor):
    """
    Decorator to monitor endpoint performance.

    Args:
        monitor: PerformanceMonitor instance
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start
                monitor.record(func.__name__, duration)

        return wrapper
    return decorator


# Global performance monitor
performance_monitor = PerformanceMonitor()


def optimize_numpy_array(arr):
    """
    Optimize numpy array for JSON serialization.

    Args:
        arr: Numpy array

    Returns:
        Optimized list (downsampled if too large)
    """
    import numpy as np

    # If array is large, downsample
    max_points = 10000
    total_points = arr.size

    if total_points > max_points:
        # Downsample
        step = int(np.ceil(total_points / max_points))
        if arr.ndim == 1:
            return arr[::step].tolist()
        elif arr.ndim == 2:
            return arr[::step, ::step].tolist()
        elif arr.ndim == 3:
            return arr[::step, ::step, ::step].tolist()

    return arr.tolist()


def batch_array_conversion(arrays: dict) -> dict:
    """
    Batch convert multiple numpy arrays to lists efficiently.

    Args:
        arrays: Dictionary of arrays

    Returns:
        Dictionary with converted arrays
    """
    return {
        key: optimize_numpy_array(arr) if hasattr(arr, 'tolist') else arr
        for key, arr in arrays.items()
    }
