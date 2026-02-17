# Solar Imaging Display

Real-time multi-source solar imagery viewer displaying images from five space-based solar observatories.

![Solar Imaging Display](https://www.swpc.noaa.gov/sites/default/files/images/u2/SUVI_montage.png)

## Features

- **24 solar image channels** from 5 observatories
- **Tabbed interface** organized by data source
- **Auto-refresh** with configurable update intervals
- **One-click download** of any image with automatic filename generation
- **Dark theme** optimized for space weather monitoring
- **Standalone application** - no server required

## Installation

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

### Quick Install

```bash
# Clone or extract the application
cd solar-imaging-display

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run the application
python -m solarimaging.main_direct
```

### Dependencies

```
PyQt6>=6.4.0
aiohttp>=3.8.0
```

### From ZIP Distribution

1. Extract the ZIP file to your desired location
2. Open a terminal/command prompt in the extracted folder
3. Run the installation commands above

## Usage

### Starting the Application

```bash
# From the distribution directory
python -m solarimaging.main_direct

# Or run directly
python solarimaging/main_direct.py
```

### Navigation

- **Tabs**: Click tabs to switch between observatories (GOES SUVI, SDO AIA, etc.)
- **Save**: Click the "Save" button on any panel to download that image
- **Refresh**: Click "Refresh All" in the toolbar to manually fetch all images
- **Tooltips**: Hover over any image panel to see wavelength descriptions

### Keyboard Shortcuts

- **Ctrl+Q**: Quit application
- **Tab/Shift+Tab**: Navigate between tabs

---

## Solar Image Sources

### GOES SUVI (Solar Ultraviolet Imager)

**Spacecraft**: GOES-16, GOES-18 (geostationary orbit, 35,786 km altitude)
**Operator**: NOAA Space Weather Prediction Center
**Update Cadence**: ~1 minute

SUVI continuously monitors the Sun in extreme ultraviolet (EUV) wavelengths to detect solar flares, track coronal holes, and observe active regions. As part of NOAA's operational space weather satellites, SUVI provides real-time data for space weather forecasting.

| Channel | Temperature | What It Shows |
|---------|-------------|---------------|
| **94 Å** | 6.3 MK | Flare plasma, very hot coronal loops |
| **131 Å** | 0.4 MK / 10 MK | Flare plasma and transition region |
| **171 Å** | 0.6 MK | Quiet corona, coronal loops |
| **195 Å** | 1.5 MK | Corona, hot active regions (standard view) |
| **284 Å** | 2.0 MK | Active regions, coronal holes |
| **304 Å** | 0.05 MK | Chromosphere, prominences, filaments |

---

### SDO AIA (Atmospheric Imaging Assembly)

**Spacecraft**: Solar Dynamics Observatory (geosynchronous orbit)
**Operator**: NASA Goddard Space Flight Center
**Update Cadence**: 12 seconds (displayed every 60 seconds)

AIA provides the highest-resolution full-disk solar images available, observing the Sun in 10 wavelength channels. SDO has been operational since 2010 and has revolutionized our understanding of solar dynamics.

| Channel | Ion | Temperature | What It Shows |
|---------|-----|-------------|---------------|
| **94 Å** | Fe XVIII | 6.3 MK | Hot flare plasma |
| **131 Å** | Fe VIII, XX, XXIII | 0.4 MK, 10 MK | Flares and transition region |
| **171 Å** | Fe IX | 0.6 MK | Quiet corona, coronal loops |
| **193 Å** | Fe XII, XXIV | 1.2 MK, 20 MK | Corona and flare plasma |
| **211 Å** | Fe XIV | 2.0 MK | Active region corona |
| **304 Å** | He II | 0.05 MK | Chromosphere, prominences |
| **335 Å** | Fe XVI | 2.5 MK | Active region corona |
| **1600 Å** | C IV + continuum | 0.1 MK | Upper photosphere, transition region |
| **1700 Å** | Continuum | 5,000 K | Temperature minimum |
| **4500 Å** | Continuum | 6,000 K | Photosphere (white light, sunspots) |

**Understanding EUV Wavelengths**: Each wavelength is dominated by emission from ions at specific temperatures. By observing multiple wavelengths, scientists can determine the temperature structure of the corona. Hotter plasma appears in shorter wavelengths (94 Å, 131 Å), while cooler plasma dominates longer wavelengths (304 Å, 171 Å).

---

### SDO HMI (Helioseismic and Magnetic Imager)

**Spacecraft**: Solar Dynamics Observatory
**Operator**: Stanford University / NASA
**Update Cadence**: 45 seconds (displayed every 60 seconds)

HMI measures the photospheric magnetic field and velocity field using the Zeeman effect in the Fe I absorption line at 6173 Å. Essential for tracking active regions, predicting flares, and understanding the solar magnetic cycle.

| Product | What It Shows |
|---------|---------------|
| **Continuum** | White-light image of the photosphere showing sunspots and solar granulation |
| **Magnetogram** | Line-of-sight magnetic field. White = positive (north) polarity, Black = negative (south) polarity. Gray = weak or transverse field |

**Reading Magnetograms**: Sunspots appear as bipolar regions (one white, one black pole). The magnetic complexity of an active region (how twisted and sheared the field lines are) correlates with flare probability. Delta-class regions, where opposite polarities are very close, produce the largest flares.

---

### SOHO LASCO (Large Angle Spectrometric Coronagraph)

**Spacecraft**: Solar and Heliospheric Observatory (L1 Lagrange point, 1.5 million km from Earth)
**Operator**: ESA/NASA
**Update Cadence**: ~12-20 minutes

LASCO blocks the bright solar disk with an occulting disk to reveal the faint outer corona. This is essential for detecting coronal mass ejections (CMEs) as they leave the Sun and propagate into the heliosphere toward Earth.

| Coronagraph | Field of View | What It Shows |
|-------------|---------------|---------------|
| **C2** | 2-6 solar radii | Inner corona, CME launch, streamers. Orange occulting disk |
| **C3** | 4-30 solar radii | Outer corona, CME propagation, solar wind. Blue occulting disk |

**CME Detection**: CMEs appear as expanding bright arcs or "halos" (when directed toward Earth) in coronagraph images. A CME visible in both C2 and C3 typically takes 1-4 days to reach Earth, depending on its speed (400-3000 km/s).

**Important**: The dark disk in the center is the occulting disk that blocks the Sun - it's not a solar eclipse! The white circle shows the actual size of the Sun behind the disk.

---

### SOHO EIT (Extreme ultraviolet Imaging Telescope)

**Spacecraft**: Solar and Heliospheric Observatory (L1)
**Operator**: ESA/NASA
**Update Cadence**: ~12 minutes

EIT has been observing the Sun since 1996, providing over 25 years of consistent solar imaging. While lower resolution than SDO/AIA, its location at L1 provides an uninterrupted view and serves as a backup to newer instruments.

| Channel | Temperature | What It Shows |
|---------|-------------|---------------|
| **171 Å** | 1.0 MK | Quiet corona (green colormap) |
| **195 Å** | 1.5 MK | Corona, active regions (standard view) |
| **284 Å** | 2.0 MK | Active region corona (yellow colormap) |
| **304 Å** | 0.08 MK | Chromosphere, prominences (orange/red) |

---

## Understanding Solar Features

### Active Regions
Regions of strong magnetic field, appearing bright in EUV and as sunspot groups in white light. Numbered by NOAA (e.g., AR 3234). Source of solar flares and CMEs.

### Coronal Holes
Dark regions in EUV images where magnetic field lines are "open" to interplanetary space. Source of high-speed solar wind streams that can cause geomagnetic storms 2-4 days after passing central meridian.

### Prominences / Filaments
Cool, dense plasma suspended in the corona by magnetic fields. Appear bright on the limb (prominences) and dark on the disk (filaments) in 304 Å. Can erupt and become CMEs.

### Solar Flares
Sudden brightening from magnetic reconnection. Classified by X-ray flux:
- **A, B, C**: Minor (no Earth impact)
- **M**: Moderate (radio blackouts)
- **X**: Major (severe radio blackouts, radiation storms)

### Coronal Mass Ejections (CMEs)
Massive expulsions of plasma and magnetic field. Visible in LASCO as expanding arcs. Earth-directed CMEs can cause geomagnetic storms (G1-G5) 1-4 days later.

---

## Data Sources & Attribution

### NOAA SWPC
- GOES SUVI imagery: https://www.swpc.noaa.gov/products/goes-solar-ultraviolet-imager-suvi
- Real-time space weather data: https://www.swpc.noaa.gov/

### NASA SDO
- SDO Mission: https://sdo.gsfc.nasa.gov/
- Data via Helioviewer.org API

### ESA/NASA SOHO
- SOHO Mission: https://soho.nascom.nasa.gov/
- Data via Helioviewer.org API

### Helioviewer
- API Documentation: https://api.helioviewer.org/docs/v2/
- Web Viewer: https://helioviewer.org/

---

## Troubleshooting

### Images Not Loading
- Check your internet connection
- Verify firewall allows HTTPS connections to:
  - `services.swpc.noaa.gov`
  - `api.helioviewer.org`

### Slow Updates
- NOAA SWPC images update every ~1 minute
- Helioviewer requests may take 10-30 seconds
- LASCO/EIT update every 12-20 minutes (this is their actual data cadence)

### Application Won't Start
```bash
# Verify Python version
python --version  # Should be 3.9+

# Verify dependencies
pip list | grep PyQt6
pip list | grep aiohttp

# Reinstall if needed
pip install --force-reinstall PyQt6 aiohttp
```

---

## Technical Details

### Architecture
- **PyQt6**: Cross-platform GUI framework
- **aiohttp**: Async HTTP client for concurrent image fetching
- **QThread**: Background fetching without blocking UI

### Update Intervals
- Fast sources (SUVI, AIA, HMI): 60 seconds
- Slow sources (LASCO, EIT): 15 minutes

### Image Sizes
- NOAA SWPC SUVI: ~700 KB PNG
- Helioviewer screenshots: ~50-100 KB PNG (512x512)

### Filename Convention
Downloaded images use the format:
```
{source_id}_{YYYYMMDD}_{HHMMSS}.png
```
Example: `suvi_195_20260217_161233.png`

---

## License

This application is part of the AutoNVIS Project.

Solar imagery is provided by NOAA, NASA, and ESA and is in the public domain.

---

## Version History

### 1.0.0 (2026-02-17)
- Initial release
- 24 image sources from 5 observatories
- Tabbed interface with download functionality
- Dark theme
