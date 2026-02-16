#!/bin/bash
#
# AutoNVIS Space Weather - Direct from NOAA
# Simple. No RabbitMQ. No Dashboard. Just data.
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

echo "=============================================="
echo "Space Weather - Direct from NOAA"
echo "=============================================="

python -m src.visualization.pyqt.spaceweather.main_direct
