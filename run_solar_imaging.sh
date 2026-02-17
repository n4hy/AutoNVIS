#!/bin/bash
# Launch Solar Imaging Display

cd "$(dirname "$0")"
PYTHONPATH="src/visualization/pyqt:$PYTHONPATH" python -m solarimaging.main_direct "$@"
