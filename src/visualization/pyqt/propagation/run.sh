#!/bin/bash
#
# HF Propagation Conditions Display - Run Script
#
# Launches the HF Propagation Conditions Display application.
# Automatically activates the virtual environment if present.
#
# Usage: ./run.sh
#

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ -d "venv" ] && [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -z "$VIRTUAL_ENV" ]; then
    echo "Warning: No virtual environment found."
    echo "Run ./install.sh first, or ensure PyQt6, pyqtgraph, and aiohttp are installed."
    echo ""
fi

# Run the application
exec python -m propagation.main_direct "$@"
