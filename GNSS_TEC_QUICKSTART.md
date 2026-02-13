# GNSS-TEC Quick Start Guide

## Installation

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure NTRIP Access** (if authentication required)
   ```bash
   export NTRIP_USER="your_username"
   export NTRIP_PASS="your_password"
   ```

3. **Verify Configuration**
   ```bash
   cat config/production.yml | grep -A 5 ntrip
   ```

## Running GNSS-TEC Ingestion

### Option 1: Standalone GNSS-TEC Client

```bash
python3 -m src.ingestion.gnss.gnss_tec_client
```

### Option 2: Full Ingestion Service (Recommended)

```bash
python3 -m src.ingestion.main
```

This starts all data ingestion services:
- GOES X-ray monitoring
- ACE solar wind monitoring
- GNSS-TEC monitoring

## Testing

### Run Unit Tests

```bash
# Run all GNSS-TEC tests
pytest tests/unit/test_gnss_tec.py -v

# Run specific test
pytest tests/unit/test_gnss_tec.py::TestTECCalculator::test_calculate_tec_from_pseudorange -v

# Run with coverage
pytest tests/unit/test_gnss_tec.py --cov=src.ingestion.gnss --cov-report=html
```

### Test Individual Components

**Test NTRIP Connection:**
```python
import asyncio
from src.ingestion.gnss.ntrip_client import NTRIPClient

async def test():
    client = NTRIPClient(
        host="www.igs-ip.net",
        port=2101,
        mountpoint="RTCM3"
    )

    connected = await client.connect()
    print(f"Connected: {connected}")

    if connected:
        chunk = await client.read_data(1024)
        print(f"Read {len(chunk) if chunk else 0} bytes")
        await client.disconnect()

asyncio.run(test())
```

**Test TEC Calculation:**
```python
from src.ingestion.gnss.tec_calculator import TECCalculator, GPS_L1_FREQ, GPS_L2_FREQ

calculator = TECCalculator()

# Simulate measurements with ionospheric delay
p1 = 20000000.0  # L1 pseudorange (meters)
p2 = 20000100.0  # L2 pseudorange (100m more delay)

tec = calculator.calculate_tec_from_pseudorange(p1, p2, GPS_L1_FREQ, GPS_L2_FREQ)
print(f"TEC: {tec:.2f} TECU")
```

## Monitoring

### Check Service Status

```bash
# View logs
tail -f logs/ingestion/gnss_tec.log

# View all ingestion logs
tail -f logs/ingestion/orchestrator.log
```

### Monitor Message Queue

```bash
# List queues
sudo rabbitmqctl list_queues

# Monitor messages
sudo rabbitmqctl list_consumers

# Check obs.gnss_tec topic
sudo rabbitmqctl list_bindings | grep obs.gnss_tec
```

### View Statistics

The client provides real-time statistics accessible via the `statistics` property:

```python
from src.ingestion.gnss.gnss_tec_client import GNSSTECClient

client = GNSSTECClient()
stats = client.statistics

print(f"RTCM messages: {stats['rtcm_messages_processed']}")
print(f"TEC published: {stats['tec_measurements_published']}")
print(f"Parser stats: {stats['rtcm_parser']}")
print(f"Calculator stats: {stats['tec_calculator']}")
```

## Troubleshooting

### Connection Timeout

**Symptom:**
```
NTRIP connection timeout
```

**Solution:**
- Check network connectivity: `ping www.igs-ip.net`
- Verify port 2101 is not blocked by firewall
- Try different NTRIP mountpoint

### Authentication Failed

**Symptom:**
```
NTRIP authentication failed (401 Unauthorized)
```

**Solution:**
- Verify credentials in environment variables
- Check if mountpoint requires authentication
- Some public mountpoints don't need credentials (try setting to None)

### No Data Published

**Symptom:**
```
Connected to NTRIP stream but no TEC measurements published
```

**Solution:**
1. Check if station position is being received (Type 1005 message)
2. Verify satellite positions are available (needs ephemeris)
3. Lower elevation mask for testing:
   ```python
   # In tec_calculator.py, temporarily change:
   MIN_ELEVATION_ANGLE = 5.0  # Instead of 10.0
   ```

### CRC Errors

**Symptom:**
```
RTCM3 CRC check failed
```

**Solution:**
- Network corruption - check connection quality
- Try different NTRIP mountpoint
- Verify RTCM3 version compatibility

## Next Steps

### Integration with Filter

The GNSS-TEC client is now publishing measurements to the message queue. To integrate with the SR-UKF filter:

1. **Verify message queue is receiving data:**
   ```bash
   sudo rabbitmqctl list_queues | grep gnss_tec
   ```

2. **Check filter orchestrator is running:**
   ```bash
   python3 -m src.supervisor.filter_orchestrator
   ```

3. **Monitor filter updates:**
   ```bash
   tail -f logs/supervisor/filter_orchestrator.log
   ```

### Production Deployment

For production use, consider these enhancements:

1. **Add pyrtcm library** for complete RTCM3 parsing:
   ```bash
   pip install pyrtcm
   ```

2. **Configure systemd service** (Linux):
   ```bash
   sudo systemctl enable autonvis-ingestion
   sudo systemctl start autonvis-ingestion
   ```

3. **Set up monitoring alerts** for:
   - Connection failures
   - Low data rate
   - Quality control rejections

4. **Implement ephemeris handling** for satellite positions:
   - Use IGS rapid ephemeris products
   - Or compute from broadcast ephemeris in RTCM messages

## File Locations

### Source Code
- `src/ingestion/gnss/ntrip_client.py` - NTRIP client
- `src/ingestion/gnss/rtcm3_parser.py` - RTCM3 parser
- `src/ingestion/gnss/tec_calculator.py` - TEC calculation
- `src/ingestion/gnss/gnss_tec_client.py` - Main client
- `src/ingestion/main.py` - Ingestion orchestrator

### Configuration
- `config/production.yml` - NTRIP settings

### Tests
- `tests/unit/test_gnss_tec.py` - Unit tests

### Documentation
- `docs/GNSS_TEC_IMPLEMENTATION.md` - Full implementation guide
- `GNSS_TEC_QUICKSTART.md` - This file

## Support

For detailed information, see `docs/GNSS_TEC_IMPLEMENTATION.md`.
