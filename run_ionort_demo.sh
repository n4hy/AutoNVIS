#!/bin/bash
# IONORT Live Dashboard Launcher
# Usage: ./run_ionort_demo.sh [simple|full|live] [options]
#
# Modes:
#   simple  - Basic demo with minimal UI (default)
#   full    - Full dashboard with all controls
#   live    - Full dashboard with live ionospheric data enabled
#
# Options (for full/live modes):
#   --tx LAT,LON      Transmitter location (default: 40.0,-105.0 Boulder, CO)
#   --rx LAT,LON      Receiver location (default: 35.0,-106.0 Albuquerque, NM)
#   --freq MIN,MAX    Frequency range in MHz (default: 3,15)
#   --nvis            Enable NVIS mode (high elevation, short range)
#   --simulated       Use simulated data (no network required)

MODE=${1:-simple}
shift 2>/dev/null  # Remove mode from args, pass rest to script

echo "==================================="
echo "  AutoNVIS IONORT Live Dashboard"
echo "  Version 0.4.0"
echo "==================================="
echo ""

case "$MODE" in
    live)
        echo "Launching LIVE dashboard with real-time ionospheric data..."
        echo "  - GIRO ionosonde network for foF2/hmF2"
        echo "  - NOAA SWPC space weather monitoring"
        echo ""
        python3 scripts/ionort_live_demo.py --live "$@"
        ;;
    full)
        echo "Launching full dashboard with manual controls..."
        python3 scripts/ionort_live_demo.py "$@"
        ;;
    simple)
        echo "Launching simple dashboard..."
        python3 scripts/ionort_simple.py
        ;;
    help|--help|-h)
        echo "Usage: ./run_ionort_demo.sh [simple|full|live] [options]"
        echo ""
        echo "Modes:"
        echo "  simple    Basic demo with minimal UI (default)"
        echo "  full      Full dashboard with all controls"
        echo "  live      Full dashboard with LIVE ionospheric data"
        echo ""
        echo "Options (for full/live modes):"
        echo "  --tx LAT,LON      Transmitter location"
        echo "  --rx LAT,LON      Receiver location"
        echo "  --freq MIN,MAX    Frequency range in MHz"
        echo "  --nvis            NVIS mode (high elevation)"
        echo "  --simulated       Use simulated data (no network)"
        echo ""
        echo "Performance tuning:"
        echo "  --workers N       Number of parallel workers (default: all CPUs)"
        echo "  --step-km N       Integration step km (default: 1.0, try 2-5 for speed)"
        echo "  --freq-step N     Frequency step MHz (default: 1.0)"
        echo "  --elev-step N     Elevation step degrees (default: 10.0)"
        echo "  --max-hops N      Maximum ground reflections for multi-hop (default: 3)"
        echo "  --tolerance N     Landing tolerance km (default: 100, try 200-500 for long paths)"
        echo ""
        echo "Examples:"
        echo "  ./run_ionort_demo.sh simple"
        echo "  ./run_ionort_demo.sh live"
        echo "  ./run_ionort_demo.sh live --simulated"
        echo "  ./run_ionort_demo.sh live --tx 40.7,-74.0 --rx 38.9,-77.0"
        echo "  ./run_ionort_demo.sh full --nvis"
        echo ""
        echo "Fast run (larger steps, fewer rays):"
        echo "  ./run_ionort_demo.sh live --step-km 3 --freq-step 2 --elev-step 15"
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Use: ./run_ionort_demo.sh help"
        exit 1
        ;;
esac
