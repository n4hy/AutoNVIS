"""
NTRIP Client for GNSS Data Streaming

Implements NTRIP (Networked Transport of RTCM via Internet Protocol) protocol
for receiving real-time GNSS correction data and observables from IGS stations.
"""

import asyncio
import aiohttp
import base64
from typing import Optional, Callable, Dict, Any
from datetime import datetime
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.common.logging_config import ServiceLogger


class NTRIPClient:
    """
    NTRIP client for connecting to GNSS data streams

    NTRIP is HTTP-based protocol for streaming RTCM data. Connection process:
    1. Send HTTP GET request with NTRIP headers
    2. Authenticate using HTTP Basic Auth
    3. Receive continuous binary stream of RTCM3 messages
    """

    def __init__(
        self,
        host: str,
        port: int,
        mountpoint: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        user_agent: str = "NTRIP AutoNVIS/1.0"
    ):
        """
        Initialize NTRIP client

        Args:
            host: NTRIP caster hostname (e.g., www.igs-ip.net)
            port: NTRIP caster port (typically 2101)
            mountpoint: Data stream mountpoint (e.g., RTCM3)
            username: Authentication username (optional)
            password: Authentication password (optional)
            user_agent: User agent string for NTRIP protocol
        """
        self.host = host
        self.port = port
        self.mountpoint = mountpoint
        self.username = username
        self.password = password
        self.user_agent = user_agent

        self.logger = ServiceLogger("ingestion", "ntrip_client")

        self._session: Optional[aiohttp.ClientSession] = None
        self._response: Optional[aiohttp.ClientResponse] = None
        self._connected = False
        self._reconnect_delay = 5  # seconds
        self._max_reconnect_attempts = 5

    def _build_request_headers(self) -> Dict[str, str]:
        """
        Build HTTP headers for NTRIP request

        Returns:
            Dictionary of HTTP headers
        """
        headers = {
            'User-Agent': self.user_agent,
            'Ntrip-Version': 'Ntrip/2.0',
            'Connection': 'close'
        }

        # Add Basic Authentication if credentials provided
        if self.username and self.password:
            credentials = f"{self.username}:{self.password}"
            encoded = base64.b64encode(credentials.encode()).decode()
            headers['Authorization'] = f'Basic {encoded}'

        return headers

    async def connect(self) -> bool:
        """
        Connect to NTRIP caster and establish data stream

        Returns:
            True if connection successful, False otherwise
        """
        url = f"http://{self.host}:{self.port}/{self.mountpoint}"

        self.logger.info(
            f"Connecting to NTRIP caster: {self.host}:{self.port}/{self.mountpoint}"
        )

        try:
            # Create session if not exists
            if self._session is None:
                timeout = aiohttp.ClientTimeout(total=None, connect=30, sock_read=60)
                self._session = aiohttp.ClientSession(timeout=timeout)

            headers = self._build_request_headers()

            # Make HTTP GET request
            self._response = await self._session.get(url, headers=headers)

            # Check response status
            if self._response.status == 200:
                self._connected = True
                self.logger.info(
                    f"Connected to NTRIP stream: {self.mountpoint}",
                    extra={
                        'host': self.host,
                        'port': self.port,
                        'mountpoint': self.mountpoint
                    }
                )
                return True

            elif self._response.status == 401:
                self.logger.error("NTRIP authentication failed (401 Unauthorized)")
                await self.disconnect()
                return False

            elif self._response.status == 404:
                self.logger.error(f"NTRIP mountpoint not found: {self.mountpoint}")
                await self.disconnect()
                return False

            else:
                self.logger.error(
                    f"NTRIP connection failed with status {self._response.status}"
                )
                await self.disconnect()
                return False

        except asyncio.TimeoutError:
            self.logger.error("NTRIP connection timeout")
            await self.disconnect()
            return False

        except aiohttp.ClientError as e:
            self.logger.error(f"NTRIP connection error: {e}", exc_info=True)
            await self.disconnect()
            return False

        except Exception as e:
            self.logger.error(f"Unexpected NTRIP connection error: {e}", exc_info=True)
            await self.disconnect()
            return False

    async def disconnect(self):
        """Close NTRIP connection and cleanup resources"""
        self._connected = False

        if self._response:
            self._response.close()
            self._response = None

        if self._session:
            await self._session.close()
            self._session = None

        self.logger.info("Disconnected from NTRIP stream")

    async def read_data(self, chunk_size: int = 1024) -> Optional[bytes]:
        """
        Read data chunk from NTRIP stream

        Args:
            chunk_size: Number of bytes to read per chunk

        Returns:
            Binary data chunk, or None if connection closed/error
        """
        if not self._connected or not self._response:
            self.logger.warning("Cannot read data: not connected to NTRIP stream")
            return None

        try:
            chunk = await self._response.content.read(chunk_size)

            if not chunk:
                # Empty chunk indicates stream closed
                self.logger.warning("NTRIP stream closed by server")
                self._connected = False
                return None

            return chunk

        except asyncio.TimeoutError:
            self.logger.error("Timeout reading from NTRIP stream")
            self._connected = False
            return None

        except aiohttp.ClientError as e:
            self.logger.error(f"Error reading NTRIP stream: {e}", exc_info=True)
            self._connected = False
            return None

        except Exception as e:
            self.logger.error(f"Unexpected error reading NTRIP stream: {e}", exc_info=True)
            self._connected = False
            return None

    async def read_stream(
        self,
        callback: Callable[[bytes], None],
        chunk_size: int = 1024
    ):
        """
        Continuously read NTRIP stream and process with callback

        Args:
            callback: Function to process each data chunk
            chunk_size: Number of bytes to read per chunk
        """
        reconnect_attempt = 0

        while True:
            # Connect if not connected
            if not self._connected:
                if reconnect_attempt >= self._max_reconnect_attempts:
                    self.logger.error(
                        f"Max reconnection attempts ({self._max_reconnect_attempts}) reached"
                    )
                    break

                if reconnect_attempt > 0:
                    self.logger.info(
                        f"Reconnection attempt {reconnect_attempt}/{self._max_reconnect_attempts}"
                    )
                    await asyncio.sleep(self._reconnect_delay)

                connected = await self.connect()

                if not connected:
                    reconnect_attempt += 1
                    continue

                reconnect_attempt = 0  # Reset on successful connection

            # Read data from stream
            try:
                chunk = await self.read_data(chunk_size)

                if chunk is None:
                    # Connection lost, will reconnect on next iteration
                    continue

                # Process data with callback
                callback(chunk)

            except Exception as e:
                self.logger.error(f"Error processing NTRIP data: {e}", exc_info=True)
                self._connected = False

    @property
    def is_connected(self) -> bool:
        """Check if currently connected to NTRIP stream"""
        return self._connected

    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()
