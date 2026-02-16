#!/bin/bash
#
# AutoNVIS Space Weather Display Launcher
# Starts all required services and the PyQt visualization app for GOES X-ray
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="config/local.yml"
VENV="venv/bin/activate"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}=============================================="
echo "AutoNVIS Space Weather Display Launcher"
echo -e "==============================================${NC}"

# Activate virtual environment
if [ -f "$VENV" ]; then
    source "$VENV"
    echo -e "${GREEN}[OK]${NC} Virtual environment activated"
else
    echo -e "${RED}[ERROR]${NC} Virtual environment not found at $VENV"
    exit 1
fi

# Kill any existing processes on port 8080
if lsof -ti:8080 > /dev/null 2>&1; then
    echo -e "${YELLOW}[WARN]${NC} Port 8080 in use, killing existing process..."
    lsof -ti:8080 | xargs -r kill -9
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
        echo -e "${RED}[ERROR]${NC} RabbitMQ not running"
        echo "Please start RabbitMQ:"
        echo "  docker run -d -p 5672:5672 -p 15672:15672 rabbitmq:3-management"
        exit 1
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
echo "Starting dashboard service..."
python -m src.output.dashboard.main --config "$CONFIG" > /tmp/autonvis_spaceweather_dashboard.log 2>&1 &
DASHBOARD_PID=$!
sleep 3

if kill -0 $DASHBOARD_PID 2>/dev/null; then
    echo -e "${GREEN}[OK]${NC} Dashboard running (PID $DASHBOARD_PID, port 8080)"
else
    echo -e "${RED}[ERROR]${NC} Dashboard failed to start. Check /tmp/autonvis_spaceweather_dashboard.log"
    cat /tmp/autonvis_spaceweather_dashboard.log
    exit 1
fi

# Start ingestion service (background)
echo "Starting ingestion service (with 24h historical X-ray backfill)..."
python -m src.ingestion.main --config "$CONFIG" > /tmp/autonvis_spaceweather_ingestion.log 2>&1 &
INGESTION_PID=$!
sleep 3

if kill -0 $INGESTION_PID 2>/dev/null; then
    echo -e "${GREEN}[OK]${NC} Ingestion running (PID $INGESTION_PID)"
else
    echo -e "${RED}[ERROR]${NC} Ingestion failed to start. Check /tmp/autonvis_spaceweather_ingestion.log"
    cat /tmp/autonvis_spaceweather_ingestion.log
    kill $DASHBOARD_PID 2>/dev/null
    exit 1
fi

echo ""
echo -e "${CYAN}=============================================="
echo "Services running. Historical X-ray data loading..."
echo "GOES X-ray updates every 1 minute from NOAA."
echo -e "==============================================${NC}"
echo ""
echo "Flare Classes:"
echo "  A: < 1e-7 W/m² (Very Quiet)"
echo "  B: 1e-7 to 1e-6 W/m² (Quiet)"
echo "  C: 1e-6 to 1e-5 W/m² (Minor)"
echo "  M: 1e-5 to 1e-4 W/m² (Moderate - SHOCK mode trigger)"
echo "  X: > 1e-4 W/m² (Major)"
echo ""
echo "Logs:"
echo "  Dashboard:  /tmp/autonvis_spaceweather_dashboard.log"
echo "  Ingestion:  /tmp/autonvis_spaceweather_ingestion.log"
echo ""
echo "Press Ctrl+C to stop all services."
echo ""

# Start PyQt app (foreground - blocks until user closes window)
python -m src.visualization.pyqt.spaceweather.main --ws-url ws://localhost:8080/ws

# Cleanup when PyQt app closes
cleanup
