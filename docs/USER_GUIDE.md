# AutoNVIS NVIS System - User Guide

**Version**: 1.0
**Last Updated**: 2026-02-13

---

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Dashboard Usage](#dashboard-usage)
4. [Sounder Registration](#sounder-registration)
5. [Data Submission](#data-submission)
6. [Quality Management](#quality-management)
7. [Network Optimization](#network-optimization)
8. [Troubleshooting](#troubleshooting)
9. [API Reference](#api-reference)

---

## Introduction

The AutoNVIS NVIS (Near Vertical Incidence Skywave) System provides real-time ionospheric monitoring through a distributed network of sounders. The system automatically assesses observation quality, optimizes network placement, and provides analytics to help operators make data-driven decisions.

### Key Features

- **Inclusive Ingestion**: Accepts observations from all quality tiers (professional to amateur)
- **Quality Weighting**: Automatically weights observations based on assessed quality
- **Real-Time Analytics**: Live dashboard with network metrics and recommendations
- **Network Optimization**: Provides placement recommendations and upgrade suggestions
- **Flexible Protocols**: Supports TCP, HTTP, MQTT, and email data submission

---

## Getting Started

### Prerequisites

- Docker and docker-compose installed
- Network access to ports: 8080 (Dashboard), 8001 (TCP), 8002 (HTTP)
- (Optional) MQTT broker access for IoT sounders

### Quick Start with Docker

```bash
# Clone repository
git clone https://github.com/your-org/AutoNVIS.git
cd AutoNVIS

# Copy environment template
cp .env.example .env

# Edit configuration
nano .env

# Deploy all services
./scripts/deploy.sh production

# Check service health
./scripts/health_check.sh
```

### Accessing the Dashboard

Open your browser and navigate to:
```
http://localhost:8080
```

You should see the NVIS Analytics Dashboard with:
- Network map showing sounder locations
- Summary cards with key metrics
- Information gain charts
- Placement recommendations

---

## Dashboard Usage

### Overview Tab

The main dashboard displays:

**Summary Cards**:
- **Total Sounders**: Number of active sounders in network
- **Total Observations**: Observations received in last 24 hours
- **Information Gain**: Total uncertainty reduction from observations
- **Uncertainty Reduction**: Percentage improvement in state estimate

**Network Map**:
- Color-coded markers for sounder quality tiers:
  - 🔵 PLATINUM (professional, σ=2dB)
  - 🟡 GOLD (research, σ=4dB)
  - ⚪ SILVER (amateur advanced, σ=8dB)
  - 🟤 BRONZE (amateur basic, σ=15dB)
  - 🟣 Recommended placement locations

**Charts**:
- **Information Gain Bar Chart**: Top 10 contributors by relative contribution
- **Quality Tier Distribution**: Pie chart showing network composition

### Sounder Details

Click any sounder in the table to see:
- Sounder metadata (operator, location, equipment type)
- Observation count and quality metrics
- Information gain contribution
- Historical quality trends

### Recommendations Panel

**New Sounder Placement**:
Shows top 3 recommended locations for adding new sounders, with:
- Priority ranking
- Expected information gain
- Coverage gap score
- Redundancy score

**Equipment Upgrades**:
Lists sounders that would benefit most from equipment upgrades:
- Current tier → Recommended tier
- Expected improvement (%)
- Relative gain increase

---

## Sounder Registration

### Via HTTP API

```bash
curl -X POST http://localhost:8002/register \
  -H "Content-Type: application/json" \
  -d '{
    "sounder_id": "MY_STATION_001",
    "name": "My NVIS Sounder",
    "operator": "Your Name / Call Sign",
    "location": "City, State",
    "latitude": 40.0,
    "longitude": -105.0,
    "altitude": 1500.0,
    "equipment_type": "amateur_advanced",
    "calibration_status": "self_calibrated"
  }'
```

### Equipment Types

- `professional`: Research-grade equipment with professional calibration
- `research`: University/institutional equipment
- `amateur_advanced`: Well-maintained amateur equipment with calibration
- `amateur_basic`: Basic amateur equipment without calibration

### Calibration Status

- `calibrated`: Professional calibration within last 12 months
- `self_calibrated`: Self-calibration using known reference
- `uncalibrated`: No formal calibration

**Note**: Equipment type and calibration status affect initial quality assessment. Historical performance will adapt quality over time.

---

## Data Submission

### HTTP POST (Recommended for Low-Rate Sounders)

```bash
curl -X POST http://localhost:8002/measurement \
  -H "Content-Type: application/json" \
  -d '{
    "sounder_id": "MY_STATION_001",
    "timestamp": "2026-02-13T12:34:56Z",
    "tx_latitude": 40.0,
    "tx_longitude": -105.0,
    "tx_altitude": 1500.0,
    "rx_latitude": 40.5,
    "rx_longitude": -104.5,
    "rx_altitude": 1600.0,
    "frequency": 7.5,
    "elevation_angle": 85.0,
    "azimuth": 45.0,
    "hop_distance": 75.0,
    "signal_strength": -85.0,
    "group_delay": 2.5,
    "snr": 20.0,
    "is_o_mode": true
  }'
```

### TCP Stream (For High-Rate Sounders)

Connect to `tcp://localhost:8001` and send JSON lines:

```python
import socket
import json

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(('localhost', 8001))

measurement = {
    "sounder_id": "PROFESSIONAL_001",
    "timestamp": "2026-02-13T12:34:56Z",
    # ... other fields ...
}

sock.send((json.dumps(measurement) + '\n').encode())
sock.close()
```

### MQTT (For IoT Devices)

Publish to topic: `nvis/measurements/{sounder_id}`

```python
import paho.mqtt.client as mqtt
import json

client = mqtt.Client()
client.connect("mqtt.example.com", 1883)

measurement = {
    "sounder_id": "IOT_SOUNDER_001",
    # ... fields ...
}

client.publish(
    f"nvis/measurements/{measurement['sounder_id']}",
    json.dumps(measurement)
)
```

### Batch Submissions

For bulk historical data:

```bash
curl -X POST http://localhost:8002/batch \
  -H "Content-Type: application/json" \
  -d '{
    "sounder_id": "MY_STATION_001",
    "measurements": [
      { /* measurement 1 */ },
      { /* measurement 2 */ },
      { /* measurement 3 */ }
    ]
  }'
```

---

## Quality Management

### Understanding Quality Tiers

Your sounder is automatically assigned a quality tier based on:

1. **Signal Quality** (30%): SNR, consistency
2. **Calibration Quality** (20%): Known calibration status
3. **Temporal Quality** (15%): Observation rate and regularity
4. **Spatial Quality** (10%): Coverage gap filling
5. **Equipment Quality** (15%): Equipment type
6. **Historical Quality** (10%): Learned from past performance

### Quality Tier Characteristics

| Tier | Error (σ) | Typical Equipment | Observation Weight |
|------|-----------|-------------------|-------------------|
| PLATINUM | 2 dB | Professional research station | 57× vs BRONZE |
| GOLD | 4 dB | University station | 14× vs BRONZE |
| SILVER | 8 dB | Advanced amateur + calibration | 3.5× vs BRONZE |
| BRONZE | 15 dB | Basic amateur, uncalibrated | 1× (baseline) |

### Improving Your Quality Tier

**Immediate Actions**:
1. **Calibrate Equipment**: Use known reference signals
2. **Improve SNR**: Better antennas, reduce noise
3. **Increase Observation Rate**: More frequent measurements (up to threshold)

**Long-Term**:
- **Historical Performance**: System learns your quality over time
- **Consistent Operations**: Regular, reliable observations
- **Coverage Gaps**: Operating in undersampled regions increases spatial quality

**Check Your Quality**:
```bash
curl http://localhost:8080/api/nvis/sounder/MY_STATION_001
```

Response includes:
```json
{
  "sounder_id": "MY_STATION_001",
  "quality_tier": "silver",
  "quality_metrics": {
    "signal_quality": 0.7,
    "calibration_quality": 0.5,
    "temporal_quality": 0.6,
    "spatial_quality": 0.8,
    "equipment_quality": 0.4,
    "historical_quality": 0.65
  },
  "quality_score": 0.62,
  "relative_contribution": 0.05
}
```

---

## Network Optimization

### Finding Optimal Sounder Locations

Use the placement recommendation API:

```bash
curl "http://localhost:8080/api/nvis/placement/recommend?n_sounders=3&tier=gold"
```

Response:
```json
[
  {
    "priority": 1,
    "latitude": 42.5,
    "longitude": -95.3,
    "combined_score": 0.82,
    "expected_gain": 0.00145,
    "coverage_gap_score": 0.75,
    "redundancy_score": 0.08
  },
  {
    "priority": 2,
    "latitude": 38.2,
    "longitude": -88.7,
    "combined_score": 0.76,
    "expected_gain": 0.00121,
    "coverage_gap_score": 0.68,
    "redundancy_score": 0.12
  }
]
```

**Interpreting Scores**:
- **Expected Gain**: Information gain from adding sounder (higher is better)
- **Coverage Gap**: How much it fills undersampled regions (0-1, higher is better)
- **Redundancy**: Overlap with existing sounders (0-1, lower is better)

### "What-If" Analysis

Simulate adding a sounder at specific location:

```bash
curl -X POST "http://localhost:8080/api/nvis/placement/simulate?latitude=40.0&longitude=-100.0&tier=gold"
```

### Upgrade Recommendations

Check if upgrading equipment would be beneficial:

```bash
curl "http://localhost:8080/api/nvis/network/analysis"
```

Look for `recommendations.upgrades`:
```json
{
  "recommendations": {
    "upgrades": [
      {
        "sounder_id": "AMATEUR_HIGH_VOLUME_001",
        "current_tier": "bronze",
        "recommended_tier": "silver",
        "expected_improvement": 0.00234,
        "relative_improvement": 0.15
      }
    ]
  }
}
```

**When to Upgrade**:
- High observation volume (>50/hour) at low tier
- High `relative_improvement` (>10%)
- Operating in coverage gap region

---

## Troubleshooting

### My observations aren't showing up

**Check 1**: Verify sounder is registered
```bash
curl http://localhost:8080/api/nvis/sounders | grep "MY_STATION_001"
```

**Check 2**: Verify data format
- All required fields present
- Valid timestamp format (ISO 8601: `2026-02-13T12:34:56Z`)
- Elevation angle 70-90° (NVIS range)
- Frequency in HF range (3-30 MHz)

**Check 3**: Check service logs
```bash
docker-compose logs nvis-client | grep "MY_STATION_001"
```

### My quality tier is lower than expected

**Possible Causes**:
1. **Low SNR**: Check `snr` field in observations (should be >10 dB)
2. **Irregular Observations**: Temporal quality improves with consistent rate
3. **Uncalibrated**: Self-calibrate against known reference
4. **Historical Performance**: May take 24-48 hours to adapt if previously biased

**Action**: Check quality breakdown
```bash
curl http://localhost:8080/api/nvis/sounder/MY_STATION_001 | jq '.quality_metrics'
```

### High latency or timeouts

**Check System Health**:
```bash
./scripts/health_check.sh
```

**Check Resource Usage**:
```bash
docker stats
```

If memory > 80% or CPU > 90%, consider:
- Reduce observation rate
- Scale horizontally (multiple instances)
- Increase resources

### Dashboard not updating

**Check WebSocket Connection**:
- Open browser console (F12)
- Look for WebSocket connection status
- Should see: `WebSocket connected`

**Check RabbitMQ**:
```bash
docker-compose logs rabbitmq
```

**Force Refresh**:
- Dashboard auto-refreshes every 30 seconds
- Manual refresh: Click browser refresh button

---

## API Reference

### REST Endpoints

#### GET /api/nvis/sounders
List all sounders with current metrics

**Response**:
```json
[
  {
    "sounder_id": "STATION_001",
    "name": "Station 1",
    "latitude": 40.0,
    "longitude": -105.0,
    "quality_tier": "gold",
    "n_observations": 1234,
    "relative_contribution": 0.08,
    "marginal_gain": 0.00145
  }
]
```

#### GET /api/nvis/sounder/{id}
Get detailed information for specific sounder

#### GET /api/nvis/network/analysis
Comprehensive network analysis with recommendations

#### GET /api/nvis/placement/recommend
Get optimal placement recommendations
- Query params: `n_sounders` (default: 3), `tier` (default: gold)

#### POST /api/nvis/placement/simulate
Simulate adding sounder at location
- Query params: `latitude`, `longitude`, `tier`

#### GET /api/nvis/placement/heatmap
Get placement quality heatmap
- Query params: `resolution` (default: 20)

### WebSocket

**Connect**: `ws://localhost:8080/ws`

**Message Format**:
```json
{
  "type": "analysis_update",
  "data": {
    /* network analysis data */
  }
}
```

---

## Support

For technical support:
- GitHub Issues: https://github.com/your-org/AutoNVIS/issues
- Email: support@autonvis.org
- Documentation: https://docs.autonvis.org

---

**Happy Sounding! 📡**
