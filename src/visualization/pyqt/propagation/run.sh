#!/bin/bash
#
# HF Propagation Conditions Display - Run Script
#
# Launches the HF Propagation Conditions Display application.
# Automatically activates the virtual environment if present.
#
# Usage: ./run.sh (can be run from any directory)
#

# Get the directory where this script is located (works from any cwd)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Determine which Python to use
if [ -f "$SCRIPT_DIR/venv/bin/python" ]; then
    # Use venv python directly (most reliable)
    PYTHON="$SCRIPT_DIR/venv/bin/python"
elif [ -n "$VIRTUAL_ENV" ] && command -v python &> /dev/null; then
    # Already in a virtual environment with python
    PYTHON="python"
elif command -v python3 &> /dev/null; then
    # Fall back to system python3
    PYTHON="python3"
    echo "Warning: No virtual environment found at $SCRIPT_DIR/venv"
    echo "Using system Python. If dependencies are missing, run:"
    echo "    cd $SCRIPT_DIR && ./install.sh"
    echo ""
else
    echo "Error: Python not found!"
    echo ""
    echo "Please install Python 3.9+ and run the installer:"
    echo "    cd $SCRIPT_DIR && ./install.sh"
    exit 1
fi

# Run from parent directory so Python can find the propagation package
cd "$SCRIPT_DIR/.."

# Run the application
exec "$PYTHON" -m propagation.main_direct "$@"
