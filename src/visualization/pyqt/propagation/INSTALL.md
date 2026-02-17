# HF Propagation Conditions Display - Linux Installation Guide

## Quick Start (TL;DR)

```bash
unzip hf-propagation-v1.1.0.zip
cd hf-propagation-v1.1.0
./install.sh
./run.sh
```

---

## System Requirements

### Minimum Requirements
- **OS**: Linux (Ubuntu 20.04+, Debian 11+, Fedora 35+, Arch, etc.)
- **Python**: 3.9 or higher
- **Memory**: 256 MB RAM
- **Display**: X11 or Wayland desktop environment
- **Network**: Internet connection for fetching NOAA data

### Recommended
- **Python**: 3.10 or higher
- **Memory**: 512 MB RAM
- **Display**: 1200x800 or higher resolution

---

## Pre-Installation: System Dependencies

### Ubuntu / Debian

```bash
# Update package list
sudo apt update

# Install Python and pip
sudo apt install python3 python3-pip python3-venv

# Install Qt dependencies (required for PyQt6)
sudo apt install libxcb-xinerama0 libxcb-cursor0 libgl1-mesa-glx

# Optional: Install for better font rendering
sudo apt install fonts-dejavu
```

### Fedora / RHEL / CentOS

```bash
# Install Python and pip
sudo dnf install python3 python3-pip

# Install Qt dependencies
sudo dnf install libxcb libxkbcommon mesa-libGL
```

### Arch Linux

```bash
# Install Python
sudo pacman -S python python-pip

# Qt dependencies are usually included, but if needed:
sudo pacman -S libxcb libxkbcommon mesa
```

### Verify Python Version

```bash
python3 --version
# Should output: Python 3.9.x or higher
```

---

## Installation Methods

### Method 1: Automated Installation (Recommended)

```bash
# 1. Extract the archive
unzip hf-propagation-v1.1.0.zip
cd hf-propagation-v1.1.0

# 2. Run the installer
./install.sh

# 3. Launch the application
./run.sh
```

The installer will:
- Create a Python virtual environment
- Install all required dependencies (PyQt6, pyqtgraph, aiohttp, numpy)
- Verify the installation

### Method 2: Manual Installation

```bash
# 1. Extract the archive
unzip hf-propagation-v1.1.0.zip
cd hf-propagation-v1.1.0

# 2. Create virtual environment
python3 -m venv venv

# 3. Activate virtual environment
source venv/bin/activate

# 4. Upgrade pip
pip install --upgrade pip

# 5. Install dependencies
pip install -r requirements.txt

# 6. Verify installation
python -c "import PyQt6; import pyqtgraph; import aiohttp; print('Installation successful!')"

# 7. Run the application
python -m propagation.main_direct
```

### Method 3: System-Wide Installation (Not Recommended)

```bash
# Install dependencies globally (may require sudo)
pip3 install --user PyQt6 pyqtgraph aiohttp numpy

# Run directly
cd hf-propagation-v1.1.0
python3 -m propagation.main_direct
```

---

## Running the Application

### Standard Launch

```bash
cd hf-propagation-v1.1.0
./run.sh
```

### With Logging

```bash
./run.sh 2>&1 | tee hf-propagation.log
```

### Background Launch

```bash
nohup ./run.sh > /dev/null 2>&1 &
```

---

## Creating a Desktop Shortcut

### Option 1: Application Menu Entry

Create file `~/.local/share/applications/hf-propagation.desktop`:

```ini
[Desktop Entry]
Name=HF Propagation Conditions
Comment=Real-time NOAA space weather monitoring for HF radio
Exec=/home/YOURUSERNAME/hf-propagation-v1.1.0/run.sh
Icon=applications-science
Terminal=false
Type=Application
Categories=Science;HamRadio;
Keywords=propagation;hf;radio;space;weather;noaa;
```

**Note**: Replace `/home/YOURUSERNAME/hf-propagation-v1.1.0` with the actual path where you extracted the application.

Then update the desktop database:
```bash
update-desktop-database ~/.local/share/applications/
```

### Option 2: Desktop Icon

```bash
# Copy the desktop file to your desktop
cp ~/.local/share/applications/hf-propagation.desktop ~/Desktop/

# Make it executable (if required by your desktop environment)
chmod +x ~/Desktop/hf-propagation.desktop
```

---

## Troubleshooting

### "No module named PyQt6"

```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall PyQt6
pip install --force-reinstall PyQt6
```

### "No module named pyqtgraph"

```bash
source venv/bin/activate
pip install --force-reinstall pyqtgraph
```

### "Could not load the Qt platform plugin 'xcb'"

Install missing X11 libraries:
```bash
# Ubuntu/Debian
sudo apt install libxcb-xinerama0 libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0

# Fedora
sudo dnf install libxcb libxkbcommon-x11
```

### "Cannot connect to X server"

Ensure you're running in a graphical environment:
```bash
echo $DISPLAY
# Should output something like ":0" or ":1"
```

If running over SSH, enable X11 forwarding:
```bash
ssh -X user@host
```

### No Data / Empty Plots

1. Check internet connectivity:
   ```bash
   curl -I https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json
   ```

2. Check firewall settings - ensure outbound HTTPS (port 443) is allowed to:
   - `services.swpc.noaa.gov`

3. Look for error messages in the terminal where you launched the app

### Application Crashes on Startup

Check for missing OpenGL:
```bash
# Ubuntu/Debian
sudo apt install libgl1-mesa-glx libgl1-mesa-dri

# Fedora
sudo dnf install mesa-libGL mesa-dri-drivers
```

---

## Uninstallation

```bash
# Simply delete the directory
rm -rf hf-propagation-v1.1.0

# Remove desktop shortcut (if created)
rm ~/.local/share/applications/hf-propagation.desktop
rm ~/Desktop/hf-propagation.desktop

# Update desktop database
update-desktop-database ~/.local/share/applications/
```

---

## Network Configuration

### Required Endpoints

The application fetches data from:

| Host | Port | Protocol | Purpose |
|------|------|----------|---------|
| services.swpc.noaa.gov | 443 | HTTPS | All space weather data |

### Proxy Configuration

If behind a corporate proxy, set environment variables before running:

```bash
export HTTP_PROXY=http://proxy.example.com:8080
export HTTPS_PROXY=http://proxy.example.com:8080
./run.sh
```

---

## Support

- **NOAA SWPC**: https://www.swpc.noaa.gov/
- **Space Weather Scales**: https://www.swpc.noaa.gov/noaa-scales-explanation
- **AutoNVIS Project**: Part of the AutoNVIS HF propagation research project
