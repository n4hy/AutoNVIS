/**
 * WebSocket Client for AutoNVIS Dashboard
 *
 * Manages WebSocket connection with automatic reconnection.
 * Handles real-time updates for all dashboard views.
 */

class WebSocketClient {
    constructor(url = null) {
        this.url = url || this._buildWebSocketURL();
        this.ws = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 10;
        this.reconnectDelay = 1000; // Start with 1 second
        this.maxReconnectDelay = 30000; // Max 30 seconds
        this.reconnectTimer = null;
        this.handlers = new Map();
        this.isConnecting = false;

        // Register default handlers
        this.onMessage = this._handleMessage.bind(this);
        this.onOpen = this._handleOpen.bind(this);
        this.onClose = this._handleClose.bind(this);
        this.onError = this._handleError.bind(this);
    }

    _buildWebSocketURL() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const host = window.location.host;
        return `${protocol}//${host}/ws`;
    }

    /**
     * Connect to WebSocket server
     */
    connect() {
        if (this.isConnecting || (this.ws && this.ws.readyState === WebSocket.OPEN)) {
            return;
        }

        this.isConnecting = true;

        try {
            console.log(`Connecting to WebSocket: ${this.url}`);
            this.ws = new WebSocket(this.url);

            this.ws.onopen = this.onOpen;
            this.ws.onclose = this.onClose;
            this.ws.onerror = this.onError;
            this.ws.onmessage = this.onMessage;

        } catch (error) {
            console.error('WebSocket connection error:', error);
            this.isConnecting = false;
            this._scheduleReconnect();
        }
    }

    /**
     * Disconnect from WebSocket server
     */
    disconnect() {
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }

        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }

        this.reconnectAttempts = 0;
    }

    /**
     * Send message to server
     */
    send(message) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(typeof message === 'string' ? message : JSON.stringify(message));
        } else {
            console.warn('WebSocket not connected, cannot send message');
        }
    }

    /**
     * Register handler for specific message type
     */
    on(messageType, handler) {
        if (!this.handlers.has(messageType)) {
            this.handlers.set(messageType, []);
        }
        this.handlers.get(messageType).push(handler);
    }

    /**
     * Unregister handler
     */
    off(messageType, handler) {
        if (this.handlers.has(messageType)) {
            const handlers = this.handlers.get(messageType);
            const index = handlers.indexOf(handler);
            if (index > -1) {
                handlers.splice(index, 1);
            }
        }
    }

    /**
     * Handle incoming message
     */
    _handleMessage(event) {
        try {
            const message = JSON.parse(event.data);
            const messageType = message.type;

            // Call registered handlers for this message type
            if (this.handlers.has(messageType)) {
                this.handlers.get(messageType).forEach(handler => {
                    try {
                        handler(message.data, message);
                    } catch (error) {
                        console.error(`Error in message handler for ${messageType}:`, error);
                    }
                });
            }

            // Call wildcard handlers (registered with '*')
            if (this.handlers.has('*')) {
                this.handlers.get('*').forEach(handler => {
                    try {
                        handler(message.data, message);
                    } catch (error) {
                        console.error('Error in wildcard message handler:', error);
                    }
                });
            }

        } catch (error) {
            console.error('Error parsing WebSocket message:', error);
        }
    }

    /**
     * Handle connection open
     */
    _handleOpen(event) {
        console.log('WebSocket connected');
        this.isConnecting = false;
        this.reconnectAttempts = 0;
        this.reconnectDelay = 1000;

        // Call registered connection handlers
        if (this.handlers.has('_connect')) {
            this.handlers.get('_connect').forEach(handler => handler());
        }
    }

    /**
     * Handle connection close
     */
    _handleClose(event) {
        console.log('WebSocket disconnected');
        this.isConnecting = false;
        this.ws = null;

        // Call registered disconnection handlers
        if (this.handlers.has('_disconnect')) {
            this.handlers.get('_disconnect').forEach(handler => handler());
        }

        // Attempt reconnection
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this._scheduleReconnect();
        } else {
            console.error('Max reconnection attempts reached');
        }
    }

    /**
     * Handle connection error
     */
    _handleError(event) {
        console.error('WebSocket error:', event);
        this.isConnecting = false;
    }

    /**
     * Schedule reconnection attempt
     */
    _scheduleReconnect() {
        if (this.reconnectTimer) {
            return; // Already scheduled
        }

        this.reconnectAttempts++;
        const delay = Math.min(
            this.reconnectDelay * Math.pow(1.5, this.reconnectAttempts - 1),
            this.maxReconnectDelay
        );

        console.log(`Reconnecting in ${delay / 1000}s (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);

        this.reconnectTimer = setTimeout(() => {
            this.reconnectTimer = null;
            this.connect();
        }, delay);
    }

    /**
     * Check connection status
     */
    isConnected() {
        return this.ws && this.ws.readyState === WebSocket.OPEN;
    }
}

// Create global WebSocket client instance
const ws = new WebSocketClient();

// Auto-connect on page load
window.addEventListener('load', () => {
    ws.connect();
});

// Reconnect on visibility change (when tab becomes visible)
document.addEventListener('visibilitychange', () => {
    if (!document.hidden && !ws.isConnected()) {
        ws.connect();
    }
});
