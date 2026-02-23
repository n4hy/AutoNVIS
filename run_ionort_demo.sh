#!/bin/bash
# IONORT Live Dashboard Launcher
# Usage: ./run_ionort_demo.sh [simple|full]

MODE=${1:-simple}

echo "==================================="
echo "  AutoNVIS IONORT Live Dashboard"
echo "  Version 0.3.1"
echo "==================================="
echo ""

if [ "$MODE" == "full" ]; then
    echo "Launching full dashboard with controls..."
    python3 scripts/ionort_live_demo.py "$@"
else
    echo "Launching simple dashboard..."
    python3 scripts/ionort_simple.py
fi
