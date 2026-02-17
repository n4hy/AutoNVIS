# Solar Imaging Display - Linux Installation Guide

## Quick Start (TL;DR)

```bash
unzip solar-imaging-display-v1.0.0.zip
cd solar-imaging-display-v1.0.0
./install.sh
./run.sh
```

---

## System Requirements

### Minimum Requirements
- **OS**: Linux (Ubuntu 20.04+, Debian 11+, Fedora 35+, Arch, etc.)
- **Python**: 3.9 or higher
- **Memory**: 512 MB RAM
- **Display**: X11 or Wayland desktop environment
- **Network**: Internet connection for fetching solar images

### Recommended
- **Python**: 3.10 or higher
- **Memory**: 1 GB RAM
- **Display**: 1920x1080 or higher resolution

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
unzip solar-imaging-display-v1.0.0.zip
cd solar-imaging-display-v1.0.0

# 2. Run the installer
./install.sh

# 3. Launch the application
./run.sh
```

The installer will:
- Create a Python virtual environment
- Install all required dependencies
- Verify the installation

### Method 2: Manual Installation

```bash
# 1. Extract the archive
unzip solar-imaging-display-v1.0.0.zip
cd solar-imaging-display-v1.0.0

# 2. Create virtual environment
python3 -m venv venv

# 3. Activate virtual environment
source venv/bin/activate

# 4. Upgrade pip
pip install --upgrade pip

# 5. Install dependencies
pip install -r requirements.txt

# 6. Verify installation
python -c "import PyQt6; import aiohttp; print('Installation successful!')"

# 7. Run the application
python -m solarimaging.main_direct
```

### Method 3: System-Wide Installation (Not Recommended)

```bash
# Install dependencies globally (may require sudo)
pip3 install --user PyQt6 aiohttp

# Run directly
cd solar-imaging-display-v1.0.0
python3 -m solarimaging.main_direct
```

---

## Running the Application

### Standard Launch

```bash
cd solar-imaging-display-v1.0.0
./run.sh
```

### With Logging

```bash
./run.sh 2>&1 | tee solar-imaging.log
```

### Background Launch

```bash
nohup ./run.sh > /dev/null 2>&1 &
```

---

## Creating a Desktop Shortcut

### Option 1: Application Menu Entry

Create file `~/.local/share/applications/solar-imaging.desktop`:

```ini
[Desktop Entry]
Name=Solar Imaging Display
Comment=Real-time multi-source solar imagery viewer
Exec=/home/YOURUSERNAME/solar-imaging-display-v1.0.0/run.sh
Icon=applications-science
Terminal=false
Type=Application
Categories=Science;Astronomy;
Keywords=solar;sun;space;weather;
```

**Note**: Replace `/home/YOURUSERNAME/solar-imaging-display-v1.0.0` with the actual path where you extracted the application.

Then update the desktop database:
```bash
update-desktop-database ~/.local/share/applications/
```

### Option 2: Desktop Icon

```bash
# Copy the desktop file to your desktop
cp ~/.local/share/applications/solar-imaging.desktop ~/Desktop/

# Make it executable (if required by your desktop environment)
chmod +x ~/Desktop/solar-imaging.desktop
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

### Images Not Loading

1. Check internet connectivity:
   ```bash
   curl -I https://services.swpc.noaa.gov/images/animations/suvi/primary/195/latest.png
   ```

2. Check firewall settings - ensure outbound HTTPS (port 443) is allowed to:
   - `services.swpc.noaa.gov`
   - `api.helioviewer.org`

### Application Crashes on Startup

Check for missing OpenGL:
```bash
# Ubuntu/Debian
sudo apt install libgl1-mesa-glx libgl1-mesa-dri

# Fedora
sudo dnf install mesa-libGL mesa-dri-drivers
```

### High Memory Usage

The application caches images in memory. If memory is constrained:
- Close other applications
- Reduce the number of tabs open simultaneously
- Images are ~100-700KB each, with 24 sources ≈ 10-15MB total

---

## Uninstallation

```bash
# Simply delete the directory
rm -rf solar-imaging-display-v1.0.0

# Remove desktop shortcut (if created)
rm ~/.local/share/applications/solar-imaging.desktop
rm ~/Desktop/solar-imaging.desktop

# Update desktop database
update-desktop-database ~/.local/share/applications/
```

---

## Network Configuration

### Required Endpoints

The application fetches data from:

| Host | Port | Protocol | Purpose |
|------|------|----------|---------|
| services.swpc.noaa.gov | 443 | HTTPS | GOES SUVI images |
| api.helioviewer.org | 443 | HTTPS | SDO/SOHO images |

### Proxy Configuration

If behind a corporate proxy, set environment variables before running:

```bash
export HTTP_PROXY=http://proxy.example.com:8080
export HTTPS_PROXY=http://proxy.example.com:8080
./run.sh
```

---

## Support

- **Issues**: https://github.com/anthropics/claude-code/issues
- **Solar Data**: https://www.swpc.noaa.gov/
- **Helioviewer API**: https://api.helioviewer.org/docs/v2/
