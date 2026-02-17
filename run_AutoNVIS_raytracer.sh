#!/bin/bash
#
# AutoNVIS HF Ray Tracer - NVIS Propagation Analysis
# LUF/MUF, Ray Paths, Coverage Maps
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

echo "=============================================="
echo "HF Ray Tracer - NVIS Propagation Analysis"
echo "=============================================="

python -m src.visualization.pyqt.raytracer.main_direct
