# HF Propagation Conditions Display

Real-time monitoring of the "Standard Four" NOAA space weather parameters that directly affect HF radio propagation.

## Features

- **Four-panel dashboard** displaying critical propagation indicators
- **NOAA R/G/S scale mapping** with color-coded severity
- **Real-time data** from GOES and DSCOVR satellites via NOAA SWPC
- **24-hour plots** with automatic data trimming
- **Overall HF conditions summary** (GOOD/MODERATE/FAIR/POOR)
- **Dark theme** optimized for radio shack environments
- **Standalone application** - no server required

## Installation

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

### Quick Install

```bash
# Extract and enter the distribution directory
cd hf-propagation-v1.0.0

# Run automated installer
./install.sh

# Launch the application
./run.sh
```

### Manual Install

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python -m propagation.main_direct
```

---

## The Standard Four Explained

These are the four primary NOAA space weather parameters that directly affect HF radio propagation. Understanding them helps you know when to expect good or poor conditions.

### 1. X-Ray Flux -> R-Scale (Radio Blackouts)

**What it is**: Solar X-ray flux measured by GOES satellites in the 0.1-0.8 nm band.

**Why it matters**: Solar flares emit intense X-rays that ionize the D-region of the ionosphere (60-90 km altitude). This increased ionization absorbs HF radio waves, causing blackouts on the sunlit side of Earth.

**NOAA R-Scale**:

| Scale | Flux (W/m²) | Flare Class | HF Impact |
|-------|-------------|-------------|-----------|
| R0 | < 10⁻⁶ | < C1 | None |
| R1 | 10⁻⁵ | M1 | Weak/minor degradation of HF on sunlit side |
| R2 | 5×10⁻⁵ | M5 | Limited blackout of HF on sunlit side |
| R3 | 10⁻⁴ | X1 | Wide area HF blackout for ~1 hour |
| R4 | 10⁻³ | X10 | HF blackout on most of sunlit Earth, 1-2 hours |
| R5 | 10⁻² | X20+ | Complete HF blackout on sunlit Earth, hours |

**Display**: Green plot (log scale), yellow line at M1, red line at X1.

---

### 2. Kp Index -> G-Scale (Geomagnetic Storms)

**What it is**: Planetary K-index, a measure of geomagnetic activity derived from ground magnetometer networks worldwide.

**Why it matters**: Geomagnetic storms disturb the ionosphere, particularly at high latitudes. During storms, the auroral oval expands equatorward, MUF drops, and absorption increases. Severe storms can make HF unusable for hours.

**NOAA G-Scale**:

| Scale | Kp | Impact on HF |
|-------|-----|--------------|
| G0 | 0-4 | Quiet - normal propagation |
| G1 | 5 | Minor - HF fades at higher latitudes |
| G2 | 6 | Moderate - HF propagation sporadic at higher latitudes |
| G3 | 7 | Strong - HF may be intermittent |
| G4 | 8 | Severe - HF propagation unreliable |
| G5 | 9 | Extreme - HF propagation may be impossible for 1-2 days |

**Display**: Blue plot (linear scale 0-9), yellow line at Kp 5, red line at Kp 7.

---

### 3. Proton Flux -> S-Scale (Solar Radiation Storms)

**What it is**: Flux of high-energy protons (>=10 MeV) measured by GOES satellites in particle flux units (pfu = particles/cm²/s/sr).

**Why it matters**: Energetic protons from solar events spiral down Earth's magnetic field lines and ionize the polar D-region, causing Polar Cap Absorption (PCA). HF signals crossing polar paths become severely attenuated.

**NOAA S-Scale**:

| Scale | Flux (pfu) | HF Impact |
|-------|------------|-----------|
| S0 | < 10 | None |
| S1 | 10 | Minor - HF fades at polar regions |
| S2 | 100 | Moderate - Small HF blackout at polar caps |
| S3 | 1,000 | Strong - HF blackout at polar regions |
| S4 | 10,000 | Severe - HF unusable in polar regions |
| S5 | 100,000 | Extreme - HF blackout in polar regions for days |

**Display**: Orange plot (log scale), yellow line at 10 pfu (S1 threshold).

---

### 4. Solar Wind Bz -> Storm Precursor

**What it is**: The north-south component of the interplanetary magnetic field (IMF) measured by the DSCOVR satellite at the L1 Lagrange point, ~1.5 million km sunward of Earth.

**Why it matters**: When Bz points southward (negative), it couples efficiently with Earth's northward magnetic field, allowing solar wind energy to enter the magnetosphere. Sustained southward Bz drives geomagnetic storms.

**Thresholds**:

| Bz (nT) | Status | Implication |
|---------|--------|-------------|
| > 0 | Northward | Favorable - energy transfer blocked |
| 0 to -10 | Southward | Watch - possible storm development |
| < -10 | Strong Southward | Warning - geomagnetic storm likely |

**Lead Time**: DSCOVR is ~30-60 minutes upstream of Earth. Southward Bz is a real-time predictor of imminent geomagnetic activity.

**Display**: Purple plot, gray line at 0 nT, red dashed line at -10 nT.

---

## Reading the Display

### Summary Bar

The top bar shows:
- **HF CONDITIONS**: Overall assessment (GOOD/MODERATE/FAIR/POOR)
- **R0/G0/S0**: Current NOAA scale values with color coding
- **Last update**: UTC timestamp of most recent data

### Condition Logic

| Status | Criteria |
|--------|----------|
| GOOD | All scales at 0 |
| MODERATE | Any scale at 1 |
| FAIR | Any scale at 2, or any scale at 1+ with Bz < -10 |
| POOR | Any scale at 4+, or any scale at 2+ with Bz < -10 |

### Plot Interpretation

- **Flat at bottom**: Quiet conditions
- **Rising toward thresholds**: Conditions degrading
- **Above yellow line**: Moderate impact expected
- **Above red line**: Significant impact expected

---

## Data Sources

All data is fetched directly from NOAA Space Weather Prediction Center:

| Parameter | Endpoint | Update Cadence |
|-----------|----------|----------------|
| X-Ray Flux | `xrays-7-day.json` | 1 minute |
| Kp Index | `planetary_k_index_1m.json` | 1 minute |
| Proton Flux | `integral-protons-1-day.json` | 5 minutes |
| Solar Wind | `mag-1-day.json` | 1 minute |

**Note**: The application fetches data every 60 seconds. Historical data (24 hours) is loaded on startup.

---

## Network Requirements

The application requires HTTPS access to:
- `services.swpc.noaa.gov` (port 443)

### Proxy Configuration

If behind a corporate proxy:

```bash
export HTTP_PROXY=http://proxy.example.com:8080
export HTTPS_PROXY=http://proxy.example.com:8080
./run.sh
```

---

## Troubleshooting

### Application Won't Start

```bash
# Verify Python version
python3 --version  # Should be 3.9+

# Verify dependencies
pip list | grep PyQt6
pip list | grep pyqtgraph
pip list | grep aiohttp

# Reinstall if needed
pip install --force-reinstall PyQt6 pyqtgraph aiohttp
```

### Plots Empty / No Data

1. Check internet connection:
   ```bash
   curl -I https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json
   ```

2. Check firewall allows HTTPS to `services.swpc.noaa.gov`

3. Check logs for errors in terminal

### "Could not load Qt platform plugin"

Install missing X11 libraries:

```bash
# Ubuntu/Debian
sudo apt install libxcb-xinerama0 libxcb-cursor0 libgl1-mesa-glx

# Fedora
sudo dnf install libxcb libxkbcommon mesa-libGL
```

---

## Technical Details

### Architecture

- **PyQt6**: Cross-platform GUI framework
- **pyqtgraph**: High-performance scientific plotting
- **aiohttp**: Async HTTP client for concurrent data fetching
- **QThread**: Background fetching without blocking UI

### Display Layout

```
+----------------------------------------------------------+
| HF CONDITIONS: [GOOD] | R0 G0 S0 | Last: 12:34:56 UTC    |
+----------------------------------------------------------+
| +------------------------+ +---------------------------+ |
| |   X-Ray Flux (R)       | |   Kp Index (G)            | |
| |   24h log plot         | |   24h linear plot         | |
| +------------------------+ +---------------------------+ |
| +------------------------+ +---------------------------+ |
| |   Proton Flux (S)      | |   Solar Wind Bz           | |
| |   24h log plot         | |   24h linear plot         | |
| +------------------------+ +---------------------------+ |
+----------------------------------------------------------+
```

---

## For HF Operators: Quick Reference

### Good Conditions (Work DX!)
- R0, G0, S0
- Bz northward or weakly southward
- Summary shows "GOOD"

### Marginal Conditions (Monitor Closely)
- R1 or G1 or S1
- Bz -5 to -10 nT
- Summary shows "MODERATE" or "FAIR"

### Poor Conditions (Consider QRT)
- R2+ or G3+ or S2+
- Bz < -10 nT sustained
- Summary shows "FAIR" or "POOR"

### Blackout Conditions (Daytime paths affected)
- R3+ (X-class flare)
- Expect 1-2 hours HF blackout on sunlit hemisphere

### PCA Event (Polar paths affected)
- S2+ (Proton event)
- Avoid polar/transpolar paths

---

## Version History

### 1.0.0 (2026-02-17)
- Initial release
- Standard Four monitoring (X-ray, Kp, Proton, Solar Wind)
- NOAA R/G/S scale mapping
- 24-hour historical plots
- Overall conditions summary

---

## License

This application is part of the AutoNVIS Project.

Space weather data is provided by NOAA SWPC and is in the public domain.

---

## References

- [NOAA Space Weather Scales](https://www.swpc.noaa.gov/noaa-scales-explanation)
- [SWPC Real-Time Data](https://www.swpc.noaa.gov/products-and-data)
- [Understanding Space Weather](https://www.swpc.noaa.gov/phenomena)
