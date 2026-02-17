#!/bin/bash
#
# AutoNVIS HF Propagation Conditions - Direct from NOAA
# X-ray, Kp, Proton flux, and Solar Wind Bz in one view
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

echo "=============================================="
echo "HF Propagation Conditions - Direct from NOAA"
echo "=============================================="

python -m src.visualization.pyqt.propagation.main_direct
