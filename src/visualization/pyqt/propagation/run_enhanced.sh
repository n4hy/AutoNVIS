#!/bin/bash
#
# Enhanced HF Propagation Display Launcher
# Combines Standard Four + Advanced Ionospheric data sources
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Determine Python interpreter
if [ -f "$SCRIPT_DIR/venv/bin/python" ]; then
    PYTHON="$SCRIPT_DIR/venv/bin/python"
elif [ -n "$VIRTUAL_ENV" ] && command -v python &> /dev/null; then
    PYTHON="python"
elif command -v python3 &> /dev/null; then
    PYTHON="python3"
else
    echo "Error: Python not found."
    echo "Please install Python 3.9+ or run ./install.sh first."
    exit 1
fi

echo "Enhanced HF Propagation Display"
echo "================================"
echo "Data Sources:"
echo "  Standard Four: X-Ray, Kp, Proton, Solar Wind Bz"
echo "  Advanced:      F10.7, Ionosonde, HPI, D-RAP"
echo "  Predictions:   WSA-Enlil, Propagated Wind"
echo ""
echo "Using: $PYTHON"
echo ""

# Run from parent directory so module imports work
cd "$SCRIPT_DIR/.."
exec "$PYTHON" -m propagation.enhanced_main "$@"
