#!/bin/bash
#
# HF Propagation Conditions Display - Installation Script
#
# This script creates a Python virtual environment and installs
# all required dependencies for the HF Propagation Conditions Display.
#
# Usage: ./install.sh
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored status messages
info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Header
echo ""
echo "=============================================="
echo "   HF Propagation Conditions - Installer"
echo "=============================================="
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check for Python 3
info "Checking Python installation..."

if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    # Check if 'python' is Python 3
    PYTHON_VERSION=$(python --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1)
    if [ "$PYTHON_VERSION" -ge 3 ]; then
        PYTHON_CMD="python"
    else
        error "Python 3 is required but only Python 2 was found."
        echo "Please install Python 3.9 or higher."
        exit 1
    fi
else
    error "Python is not installed."
    echo "Please install Python 3.9 or higher:"
    echo "  Ubuntu/Debian: sudo apt install python3 python3-pip python3-venv"
    echo "  Fedora:        sudo dnf install python3 python3-pip"
    echo "  Arch:          sudo pacman -S python python-pip"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d'.' -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d'.' -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]); then
    error "Python 3.9 or higher is required. Found: Python $PYTHON_VERSION"
    exit 1
fi

success "Found Python $PYTHON_VERSION"

# Check for venv module
info "Checking for venv module..."
if ! $PYTHON_CMD -m venv --help &> /dev/null; then
    error "Python venv module is not available."
    echo "Please install it:"
    echo "  Ubuntu/Debian: sudo apt install python3-venv"
    echo "  Fedora:        sudo dnf install python3-libs"
    exit 1
fi
success "venv module available"

# Check for pip
info "Checking for pip..."
if ! $PYTHON_CMD -m pip --version &> /dev/null; then
    warn "pip is not available. Attempting to install..."
    if ! $PYTHON_CMD -m ensurepip --upgrade &> /dev/null; then
        error "Could not install pip."
        echo "Please install pip manually:"
        echo "  Ubuntu/Debian: sudo apt install python3-pip"
        echo "  Fedora:        sudo dnf install python3-pip"
        exit 1
    fi
fi
success "pip available"

# Create virtual environment
info "Creating virtual environment..."
if [ -d "venv" ]; then
    warn "Virtual environment already exists. Removing old one..."
    rm -rf venv
fi

$PYTHON_CMD -m venv venv

if [ ! -f "venv/bin/activate" ]; then
    error "Failed to create virtual environment."
    exit 1
fi
success "Virtual environment created"

# Activate virtual environment
info "Activating virtual environment..."
source venv/bin/activate
success "Virtual environment activated"

# Upgrade pip
info "Upgrading pip..."
pip install --upgrade pip --quiet
success "pip upgraded"

# Install dependencies
info "Installing dependencies (this may take a moment)..."
echo ""

if pip install -r requirements.txt; then
    echo ""
    success "Dependencies installed successfully"
else
    echo ""
    error "Failed to install dependencies."
    echo ""
    echo "If you see errors about Qt libraries, install system dependencies:"
    echo "  Ubuntu/Debian: sudo apt install libxcb-xinerama0 libxcb-cursor0 libgl1-mesa-glx"
    echo "  Fedora:        sudo dnf install libxcb libxkbcommon mesa-libGL"
    exit 1
fi

# Verify installation
info "Verifying installation..."
if python -c "import PyQt6.QtWidgets; import pyqtgraph; import aiohttp; print('OK')" &> /dev/null; then
    success "All packages imported successfully"
else
    error "Package verification failed."
    echo "Try reinstalling: pip install --force-reinstall PyQt6 pyqtgraph aiohttp"
    exit 1
fi

# Deactivate virtual environment
deactivate

# Final message
echo ""
echo "=============================================="
echo -e "   ${GREEN}Installation Complete!${NC}"
echo "=============================================="
echo ""
echo "To run the HF Propagation Conditions Display:"
echo ""
echo "    ./run.sh"
echo ""
echo "Or manually:"
echo ""
echo "    source venv/bin/activate"
echo "    python -m propagation.main_direct"
echo ""
echo "For troubleshooting, see INSTALL.md"
echo ""
