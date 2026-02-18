#!/bin/bash
#
# AutoNVIS HF Ray Tracer - Native 3D Magnetoionic Ray Tracing
# Haselgrove equations, Appleton-Hartree, Chapman layers
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

echo "=============================================="
echo "HF Ray Tracer - Native 3D Magnetoionic"
echo "Haselgrove + Appleton-Hartree + Chapman Layers"
echo "=============================================="

python src/raytracer/display.py
