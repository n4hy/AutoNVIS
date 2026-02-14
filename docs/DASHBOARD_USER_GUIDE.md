# AutoNVIS Dashboard User Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Dashboard Views](#dashboard-views)
4. [Features](#features)
5. [API Reference](#api-reference)
6. [Troubleshooting](#troubleshooting)

---

## Introduction

The AutoNVIS Dashboard provides real-time visualization and control for the AutoNVIS ionospheric monitoring and NVIS propagation prediction system.

### Key Features

- **Real-time ionospheric visualization** with 3D electron density grids
- **Propagation prediction** with LUF/MUF/FOT tracking
- **Space weather monitoring** with X-ray flux and solar wind data
- **NVIS network analysis** with information gain computation
- **System control** for service management and configuration

### Technology Stack

- **Backend**: Python FastAPI, RabbitMQ, NumPy/SciPy
- **Frontend**: Plotly.js, Leaflet, Chart.js, Vanilla JavaScript
- **Communication**: WebSocket for real-time updates, REST API for queries

---

## Getting Started

### Starting the Dashboard

```bash
# Basic startup
python -m src.output.dashboard.main

# With custom configuration
python -m src.output.dashboard.main --config config/production.yml

# Custom host/port
python -m src.output.dashboard.main --host 0.0.0.0 --port 8080

# Without RabbitMQ (no real-time updates)
python -m src.output.dashboard.main --no-mq
```

### Accessing the Dashboard

Open your web browser and navigate to:
```
http://localhost:8080
```

### First-Time Setup

1. **Check WebSocket Connection**: Green indicator in sidebar = connected
2. **Verify Data Flow**: Grid age should be recent (<20 minutes)
3. **Navigate Views**: Use sidebar to explore different visualizations
4. **Enable Real-Time Updates**: Ensure RabbitMQ is running

---

## Dashboard Views

### 1. Overview (`/`)

**Summary dashboard with key metrics**

- **Grid Status**: Latest electron density grid age and quality
- **Frequency Plan**: Current LUF, MUF, FOT values
- **Space Weather**: X-ray flux and autonomous mode
- **Recent Alerts**: System alerts and notifications

**Use Cases**:
- Quick system health check
- At-a-glance operational status
- Entry point to specific visualizations

---

### 2. Ionosphere View (`/ionosphere`)

**3D electron density visualization**

#### Features

**Horizontal Slice (2D Heatmap)**
- Altitude slider: 60-600 km in 10 km steps
- Interactive zoom and pan
- Hover for exact Ne values at any lat/lon

**Vertical Profile**
- Specify latitude/longitude for profile location
- Automatic ionospheric layer detection (E, F1, F2)
- Critical frequencies displayed for each layer
- Chapman layer model visualization

**Parameter Maps**
- **foF2**: Critical frequency of F2 layer (MHz)
  - Computed: `f = 8.98 * sqrt(Ne_max)`
  - Typical range: 2-15 MHz
- **hmF2**: Height of F2 peak (km)
  - Typical range: 250-400 km
- **TEC**: Total Electron Content (TECU)
  - Computed: vertical integral of Ne
  - 1 TECU = 10^16 electrons/m²

#### Controls

- **Altitude Slider**: Change horizontal slice altitude
- **Profile Location**: Input lat/lon for vertical profile
- **Refresh Button**: Manually update all visualizations

#### Real-Time Updates

Visualizations automatically update when new grids arrive via WebSocket.

---

### 3. Propagation View (`/propagation`)

**NVIS propagation prediction analysis**

#### Features

**Frequency Plan Gauges**
- **LUF** (Lowest Usable Frequency): Minimum frequency for communication
- **MUF** (Maximum Usable Frequency): Maximum frequency before signal escapes
- **FOT** (Frequency of Optimum Traffic): Recommended frequency (85% of MUF)

**Time Series Chart**
- Historical LUF/MUF/FOT over 6-48 hours
- Interactive time range selector
- Hover for exact values and timestamps

**Coverage Map**
- Geographic coverage footprint
- Transmitter location marker
- SNR overlay (when available)

**Statistics**
- 24-hour min/max/mean values
- Usable bandwidth calculation
- Update frequency tracking

#### Recommended Usage

1. **Check FOT**: Use this frequency for reliable communication
2. **Monitor MUF trends**: Increasing MUF = improving conditions
3. **Watch LUF**: Frequencies below LUF will have high absorption
4. **Time series**: Identify diurnal patterns and trends

---

### 4. Space Weather View (`/spaceweather`)

**Solar activity and geomagnetic monitoring**

#### Features

**X-Ray Flux Chart**
- Real-time 0.1-0.8 nm X-ray flux from GOES satellite
- Flare classification thresholds (A, B, C, M, X)
- Automatic flare detection and labeling
- Logarithmic scale for wide dynamic range

**Flare Classes**
- **A-class**: < 10⁻⁷ W/m² (background)
- **B-class**: 10⁻⁷ to 10⁻⁶ W/m²
- **C-class**: 10⁻⁶ to 10⁻⁵ W/m² (minor)
- **M-class**: 10⁻⁵ to 10⁻⁴ W/m² (medium, affects polar regions)
- **X-class**: ≥ 10⁻⁴ W/m² (major, global impact)

**Solar Wind Parameters**
- **Speed** (km/s): Typical 300-500, CMEs can exceed 1000
- **Density** (p/cm³): Proton density, typical 5-10
- **Bz** (nT): North-south IMF component
  - Negative (southward) = geomagnetic disturbances likely
  - Positive (northward) = quiet conditions
- **Bt** (nT): Total IMF magnitude

**Autonomous Mode Timeline**
- **QUIET Mode**: Normal ionospheric conditions
- **SHOCK Mode**: Disturbed conditions (X-ray flux > M1)
- Mode change history with timestamps and reasons

**Alert Feed**
- Real-time system alerts
- Severity filtering (info, warning, error, critical)
- Timestamp and message details

#### Interpreting Data

**Good Conditions**:
- X-ray flux: A or B class
- Solar wind Bz: Positive (northward)
- Mode: QUIET

**Disturbed Conditions**:
- X-ray flux: M or X class (solar flare)
- Solar wind Bz: Strongly negative (< -5 nT)
- Mode: SHOCK
- Impact: Increased D-layer absorption, degraded HF propagation

---

### 5. Network View (`/network`)

**NVIS sounder network analysis**

#### Features

- Geographic map with sounder locations
- Quality tier visualization (platinum, gold, silver, bronze)
- Information gain analysis
- Coverage gap identification
- Optimal placement recommendations

**Quality Tiers**
- **Platinum**: Highest quality, lowest error
- **Gold**: Good quality
- **Silver**: Moderate quality
- **Bronze**: Basic quality

#### Metrics

- **Marginal Gain**: Information contribution of each sounder
- **Relative Contribution**: Percentage of total information
- **Observation Count**: Number of observations received
- **Coverage Analysis**: Geographic coverage assessment

---

### 6. Control View (`/control`)

**System management and configuration**

#### Features

**Filter Control**
- **Trigger Cycle**: Manually initiate filter update
- **Force Cycle**: Skip timing checks (use with caution)
- **Mode Switch**: Override autonomous mode (QUIET ↔ SHOCK)
- **Parameters**: View filter performance metrics

**Service Management**
- Start/stop/restart services
- Service status monitoring
- Last update timestamps

**Data Source Control**
- Enable/disable individual data sources:
  - GNSS TEC
  - Ionosonde
  - NVIS Sounders
  - GOES X-Ray
  - ACE Solar Wind

**Observations**
- Real-time observation counts by type
- Recent observations feed
- Type filtering

**System Statistics**
- Grids received
- Observations processed
- Updates and alerts
- Current mode and health

#### Safety Features

- **Confirmation modals** for all control actions
- **Status feedback** after each operation
- **Real-time updates** of system state

---

## Features

### Real-Time Updates

All visualizations update automatically via WebSocket when new data arrives:
- Grid updates → Ionosphere view refreshes
- Frequency plan updates → Propagation view updates
- X-ray data → Space weather view updates
- Mode changes → Notifications appear

### Interactive Visualizations

**Plotly Charts**:
- Zoom: Drag to select area
- Pan: Click and drag
- Reset: Double-click
- Hover: See exact values
- Download: Use toolbar button

**Leaflet Maps**:
- Zoom: Mouse wheel or +/- buttons
- Pan: Click and drag
- Markers: Click for details

### Responsive Design

Dashboard adapts to screen size:
- **Desktop**: Full sidebar, multi-column layouts
- **Tablet**: Collapsible sidebar, 2-column layouts
- **Mobile**: Hidden sidebar (toggle button), single column

### Sidebar Status Indicators

- **Latest Grid**: Age of most recent grid
  - Green: < 10 minutes old
  - Orange: 10-30 minutes old
  - Red: > 30 minutes old
- **Mode**: Current autonomous mode
- **Connection**: WebSocket status

---

## API Reference

### Base URL

```
http://localhost:8080/api
```

### Ionosphere Endpoints

```
GET /ionosphere/grid/metadata
GET /ionosphere/grid/full?max_age_seconds=1200
GET /ionosphere/slice/horizontal?altitude_km=300
GET /ionosphere/profile/vertical?latitude=40&longitude=-105
GET /ionosphere/parameters/fof2
GET /ionosphere/parameters/hmf2
GET /ionosphere/parameters/tec
GET /ionosphere/statistics
GET /ionosphere/history/grids?limit=100
```

### Propagation Endpoints

```
GET /propagation/frequency_plan/latest
GET /propagation/frequency_plan/history?hours=24
GET /propagation/coverage_map/latest
GET /propagation/luf?hours=24
GET /propagation/muf?hours=24
GET /propagation/fot?hours=24
GET /propagation/statistics
```

### Space Weather Endpoints

```
GET /spaceweather/xray/history?hours=24
GET /spaceweather/xray/latest
GET /spaceweather/solar_wind/latest
GET /spaceweather/solar_wind/history?hours=24
GET /spaceweather/mode/current
GET /spaceweather/mode/history?hours=24
GET /spaceweather/alerts/recent?hours=24&severity=warning
GET /spaceweather/statistics
```

### Control Endpoints

```
GET /control/services/status
POST /control/services/control
POST /control/filter/trigger_cycle
POST /control/filter/update_parameters
GET /control/filter/metrics
POST /control/mode/switch
POST /control/datasource/toggle
GET /control/observations/counts
GET /control/observations/recent?hours=1&obs_type=gnss_tec
GET /control/system/statistics
GET /control/health/check
```

### WebSocket

```
ws://localhost:8080/ws
```

**Message Types**:
- `grid_update`: New grid available
- `frequency_plan_update`: New propagation prediction
- `xray_update`: X-ray flux update
- `solar_wind_update`: Solar wind data update
- `mode_change`: Autonomous mode switch
- `alert`: System alert
- `observation_update`: New observation received

---

## Troubleshooting

### No Data Displayed

**Symptoms**: Empty charts, "No data available" messages

**Solutions**:
1. Check RabbitMQ is running: `docker ps | grep rabbitmq`
2. Verify services are publishing data
3. Check WebSocket connection (green indicator in sidebar)
4. Review logs: `/var/log/autonvis/dashboard.log`

### WebSocket Disconnected

**Symptoms**: Red connection indicator, no real-time updates

**Solutions**:
1. Refresh the page
2. Check network connectivity
3. Verify dashboard service is running
4. Check for firewall issues on port 8080

### Slow Performance

**Symptoms**: Laggy visualizations, slow page loads

**Solutions**:
1. Close other browser tabs
2. Reduce time range for historical data
3. Clear browser cache
4. Check server CPU/memory usage

### Stale Data

**Symptoms**: Grid age > 20 minutes, old timestamps

**Solutions**:
1. Check assimilation service is running
2. Verify observations are being received
3. Check filter cycle timing configuration
4. Review supervisor logs

### Control Actions Fail

**Symptoms**: Error messages when controlling services

**Solutions**:
1. Check you have proper permissions
2. Verify RabbitMQ is accepting messages
3. Check service is responding to commands
4. Review control topic subscriptions

### Charts Not Rendering

**Symptoms**: Blank spaces where charts should be

**Solutions**:
1. Check browser console for JavaScript errors
2. Verify Plotly.js loaded (check Network tab)
3. Clear browser cache
4. Try different browser

---

## Best Practices

### Daily Operations

1. **Morning Check**: View Overview for system status
2. **Check Grid Age**: Should be < 15 minutes during active hours
3. **Monitor Alerts**: Review Alert feed for issues
4. **Verify Mode**: Ensure mode matches space weather conditions

### Space Weather Events

1. **M-Class Flare**: Monitor propagation degradation
2. **X-Class Flare**: Expect SHOCK mode, HF blackouts possible
3. **CME Impact**: Watch solar wind Bz, prepare for disturbances
4. **Geomagnetic Storm**: Increased auroral absorption at high latitudes

### Network Optimization

1. **Review Information Gain**: Weekly check of sounder contributions
2. **Coverage Gaps**: Identify under-observed regions
3. **Placement Recommendations**: Consider adding sounders where suggested
4. **Quality Assessment**: Ensure sounders meet tier requirements

### System Maintenance

1. **Filter Performance**: Check NIS and uncertainty metrics
2. **Observation Rates**: Verify data sources are active
3. **Service Health**: All services should be running
4. **Cycle Timing**: Ensure regular updates (15-minute cycles)

---

## Keyboard Shortcuts

- **Ctrl + R**: Refresh current view
- **Esc**: Close modals/dialogs
- **Tab**: Navigate between input fields

---

## Contact & Support

For issues, feature requests, or questions:
- GitHub Issues: https://github.com/yourusername/autonvis/issues
- Documentation: https://autonvis.readthedocs.io
- Email: support@autonvis.example.com

---

## Version History

**v2.0.0** (Current)
- Complete GUI dashboard implementation
- Real-time ionosphere visualization
- Propagation prediction interface
- Space weather monitoring
- System control panel

**v1.0.0**
- Initial NVIS analytics API
- Basic network analysis
- REST endpoints only
