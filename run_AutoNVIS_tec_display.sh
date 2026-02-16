#!/bin/bash
#
# AutoNVIS TEC Display Launcher
# Starts all required services and the PyQt visualization app
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="config/local.yml"
VENV="venv/bin/activate"
PORT=8080  # TEC display uses port 8080 (Space Weather uses 8081)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=============================================="
echo "AutoNVIS TEC Display Launcher"
echo "=============================================="

# Activate virtual environment
if [ -f "$VENV" ]; then
    source "$VENV"
    echo -e "${GREEN}[OK]${NC} Virtual environment activated"
else
    echo -e "${RED}[ERROR]${NC} Virtual environment not found at $VENV"
    exit 1
fi

# Kill any existing processes on our port (but not port 8081 used by Space Weather)
if lsof -ti:$PORT > /dev/null 2>&1; then
    echo -e "${YELLOW}[WARN]${NC} Port $PORT in use, killing existing process..."
    lsof -ti:$PORT | xargs -r kill -9
    sleep 1
fi

# Check RabbitMQ
if systemctl is-active --quiet rabbitmq-server 2>/dev/null; then
    echo -e "${GREEN}[OK]${NC} RabbitMQ is running"
else
    echo -e "${YELLOW}[WARN]${NC} RabbitMQ not detected via systemctl, checking port..."
    if nc -z localhost 5672 2>/dev/null; then
        echo -e "${GREEN}[OK]${NC} RabbitMQ is running (Docker or manual)"
    else
        echo -e "${YELLOW}[WARN]${NC} RabbitMQ not running, attempting to start..."
        sudo systemctl start rabbitmq-server 2>/dev/null || true
        sleep 2
        if nc -z localhost 5672 2>/dev/null; then
            echo -e "${GREEN}[OK]${NC} RabbitMQ started"
        else
            echo -e "${RED}[ERROR]${NC} Failed to start RabbitMQ"
            echo "Please start RabbitMQ:"
            echo "  docker run -d -p 5672:5672 -p 15672:15672 rabbitmq:3-management"
            exit 1
        fi
    fi
fi

# Cleanup function
cleanup() {
    echo ""
    echo "Shutting down..."
    [ -n "$DASHBOARD_PID" ] && kill $DASHBOARD_PID 2>/dev/null
    [ -n "$INGESTION_PID" ] && kill $INGESTION_PID 2>/dev/null
    echo "Done."
    exit 0
}
trap cleanup SIGINT SIGTERM

# Start dashboard service (background)
echo "Starting dashboard service on port $PORT..."
python -m src.output.dashboard.main --config "$CONFIG" --port $PORT > /tmp/autonvis_dashboard.log 2>&1 &
DASHBOARD_PID=$!
sleep 3

if kill -0 $DASHBOARD_PID 2>/dev/null; then
    echo -e "${GREEN}[OK]${NC} Dashboard running (PID $DASHBOARD_PID, port $PORT)"
else
    echo -e "${RED}[ERROR]${NC} Dashboard failed to start. Check /tmp/autonvis_dashboard.log"
    cat /tmp/autonvis_dashboard.log
    exit 1
fi

# Check if ingestion is already running (shared with Space Weather display)
EXISTING_INGESTION=$(pgrep -f "src.ingestion.main" || true)
if [ -n "$EXISTING_INGESTION" ]; then
    echo -e "${GREEN}[OK]${NC} Ingestion already running (PID $EXISTING_INGESTION) - sharing with Space Weather"
    INGESTION_PID=""
else
    # Start ingestion service (background)
    echo "Starting ingestion service..."
    python -m src.ingestion.main --config "$CONFIG" > /tmp/autonvis_ingestion.log 2>&1 &
    INGESTION_PID=$!
    sleep 3

    if kill -0 $INGESTION_PID 2>/dev/null; then
        echo -e "${GREEN}[OK]${NC} Ingestion running (PID $INGESTION_PID)"
    else
        echo -e "${RED}[ERROR]${NC} Ingestion failed to start. Check /tmp/autonvis_ingestion.log"
        cat /tmp/autonvis_ingestion.log
        kill $DASHBOARD_PID 2>/dev/null
        exit 1
    fi
fi

echo ""
echo "=============================================="
echo "Services running. Waiting for first data..."
echo "GloTEC updates every 10 minutes from NOAA."
echo "=============================================="
echo ""
echo "Logs:"
echo "  Dashboard:  /tmp/autonvis_dashboard.log"
echo "  Ingestion:  /tmp/autonvis_ingestion.log"
echo ""
echo "Press Ctrl+C to stop all services."
echo ""

# Start PyQt app (foreground - blocks until user closes window)
python -m src.visualization.pyqt.main --ws-url ws://localhost:$PORT/ws

# Cleanup when PyQt app closes
cleanup
