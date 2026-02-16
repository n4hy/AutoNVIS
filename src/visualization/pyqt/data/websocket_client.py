"""
WebSocket Client for Dashboard Connection

Connects to the AutoNVIS dashboard WebSocket for real-time data updates.
"""

from PyQt6.QtCore import QObject, QThread, pyqtSignal, QUrl, QTimer, QEventLoop
from PyQt6.QtWebSockets import QWebSocket
import json
from typing import Optional
import logging


class WebSocketWorker(QObject):
    """
    Worker object for WebSocket operations.

    Must be moved to a QThread to work properly.
    """

    # Connection signals
    connected = pyqtSignal()
    disconnected = pyqtSignal()
    error = pyqtSignal(str)

    # Data signals
    glotec_received = pyqtSignal(dict)
    xray_received = pyqtSignal(dict)
    xray_batch_received = pyqtSignal(dict)  # Historical batch
    grid_received = pyqtSignal(dict)
    spaceweather_received = pyqtSignal(dict)

    def __init__(self, url: str):
        super().__init__()
        self.url = url
        self.socket: Optional[QWebSocket] = None
        self.reconnect_timer: Optional[QTimer] = None
        self.running = False
        self.reconnect_delay = 5000  # ms
        self.logger = logging.getLogger("websocket_client")

    def start_connection(self):
        """Initialize and start the WebSocket connection."""
        self.running = True

        # Create socket (in the thread where this worker lives)
        self.socket = QWebSocket()

        # Connect socket signals
        self.socket.connected.connect(self._on_connected)
        self.socket.disconnected.connect(self._on_disconnected)
        self.socket.textMessageReceived.connect(self._on_message)
        self.socket.errorOccurred.connect(self._on_error)

        # Create reconnect timer
        self.reconnect_timer = QTimer(self)
        self.reconnect_timer.timeout.connect(self._try_connect)

        # Initial connection
        self._try_connect()

    def _try_connect(self):
        """Attempt to connect to WebSocket server."""
        if not self.running:
            return

        if self.reconnect_timer:
            self.reconnect_timer.stop()

        self.logger.info(f"Connecting to {self.url}")
        self.socket.open(QUrl(self.url))

    def _on_connected(self):
        """Handle successful connection."""
        self.logger.info("WebSocket connected")
        if self.reconnect_timer:
            self.reconnect_timer.stop()
        self.connected.emit()

    def _on_disconnected(self):
        """Handle disconnection."""
        self.logger.warning("WebSocket disconnected")
        self.disconnected.emit()

        # Schedule reconnect
        if self.running and self.reconnect_timer:
            self.logger.info(f"Reconnecting in {self.reconnect_delay}ms")
            self.reconnect_timer.start(self.reconnect_delay)

    def _on_error(self, error):
        """Handle WebSocket error."""
        error_msg = self.socket.errorString() if self.socket else str(error)
        self.logger.error(f"WebSocket error: {error_msg}")
        self.error.emit(error_msg)

    def _on_message(self, message: str):
        """Handle incoming WebSocket message."""
        try:
            data = json.loads(message)
            msg_type = data.get('type', '')

            if msg_type == 'glotec_update':
                self.glotec_received.emit(data.get('data', {}))
            elif msg_type == 'xray_update':
                self.xray_received.emit(data.get('data', {}))
            elif msg_type == 'xray_historical_batch':
                self.xray_batch_received.emit(data.get('data', {}))
            elif msg_type == 'grid_update':
                self.grid_received.emit(data.get('data', {}))
            elif msg_type in ('solar_wind_update', 'mode_change'):
                self.spaceweather_received.emit(data.get('data', {}))
            else:
                self.logger.debug(f"Received message type: {msg_type}")

        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error: {e}")
            self.error.emit(f"JSON decode error: {e}")

    def stop_connection(self):
        """Stop the WebSocket connection."""
        self.running = False

        if self.reconnect_timer:
            self.reconnect_timer.stop()

        if self.socket:
            self.socket.close()

    def is_connected(self) -> bool:
        """Check if currently connected."""
        if self.socket is None:
            return False
        return self.socket.state() == QWebSocket.State.ConnectedState


class DashboardWebSocketClient(QObject):
    """
    WebSocket client connecting to AutoNVIS dashboard.

    Uses a worker object moved to a thread for proper Qt threading.
    """

    # Forward signals from worker
    connected = pyqtSignal()
    disconnected = pyqtSignal()
    error = pyqtSignal(str)
    glotec_received = pyqtSignal(dict)
    xray_received = pyqtSignal(dict)
    xray_batch_received = pyqtSignal(dict)  # Historical batch
    grid_received = pyqtSignal(dict)
    spaceweather_received = pyqtSignal(dict)

    def __init__(self, url: str = "ws://localhost:8080/ws", parent=None):
        """
        Initialize WebSocket client.

        Args:
            url: Dashboard WebSocket URL
            parent: Parent QObject
        """
        super().__init__(parent)

        self.url = url
        self.thread: Optional[QThread] = None
        self.worker: Optional[WebSocketWorker] = None
        self.logger = logging.getLogger("websocket_client")

    def start(self):
        """Start the WebSocket client in a background thread."""
        if self.thread is not None and self.thread.isRunning():
            self.logger.warning("WebSocket client already running")
            return

        # Create thread and worker
        self.thread = QThread()
        self.worker = WebSocketWorker(self.url)

        # Move worker to thread
        self.worker.moveToThread(self.thread)

        # Connect thread signals
        self.thread.started.connect(self.worker.start_connection)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        # Forward worker signals
        self.worker.connected.connect(self.connected.emit)
        self.worker.disconnected.connect(self.disconnected.emit)
        self.worker.error.connect(self.error.emit)
        self.worker.glotec_received.connect(self.glotec_received.emit)
        self.worker.xray_received.connect(self.xray_received.emit)
        self.worker.xray_batch_received.connect(self.xray_batch_received.emit)
        self.worker.grid_received.connect(self.grid_received.emit)
        self.worker.spaceweather_received.connect(self.spaceweather_received.emit)

        # Start thread
        self.thread.start()
        self.logger.info("WebSocket client thread started")

    def stop(self):
        """Stop the WebSocket client."""
        if self.worker:
            self.worker.stop_connection()

        if self.thread and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait(5000)

        self.thread = None
        self.worker = None
        self.logger.info("WebSocket client stopped")

    def isRunning(self) -> bool:
        """Check if the client thread is running."""
        return self.thread is not None and self.thread.isRunning()

    def is_connected(self) -> bool:
        """Check if currently connected to WebSocket server."""
        if self.worker is None:
            return False
        return self.worker.is_connected()

    def send_message(self, message: dict):
        """
        Send a message to the server.

        Args:
            message: Dictionary to send as JSON
        """
        if self.worker and self.is_connected():
            # Need to invoke in worker's thread
            from PyQt6.QtCore import QMetaObject, Qt, Q_ARG
            QMetaObject.invokeMethod(
                self.worker.socket,
                "sendTextMessage",
                Qt.ConnectionType.QueuedConnection,
                Q_ARG(str, json.dumps(message))
            )
