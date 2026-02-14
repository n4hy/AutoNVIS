# RabbitMQ Setup Guide

**Purpose**: Install and configure RabbitMQ for Auto-NVIS message queue
**Date**: 2026-02-13

---

## Quick Install (Ubuntu/Debian)

### Step 1: Install RabbitMQ

```bash
# Update package list
sudo apt update

# Install RabbitMQ server
sudo apt install -y rabbitmq-server

# Start RabbitMQ service
sudo systemctl start rabbitmq-server

# Enable auto-start on boot
sudo systemctl enable rabbitmq-server

# Check status
sudo systemctl status rabbitmq-server
```

**Expected Output**:
```
● rabbitmq-server.service - RabbitMQ broker
   Loaded: loaded (/lib/systemd/system/rabbitmq-server.service; enabled)
   Active: active (running) since Fri 2026-02-13 21:45:00 EST
```

### Step 2: Enable Management Plugin (Optional but Recommended)

```bash
# Enable web management interface
sudo rabbitmq-plugins enable rabbitmq_management

# Access web UI at: http://localhost:15672
# Default credentials: guest/guest
```

### Step 3: Create Auto-NVIS Config File

Create a config file to use localhost:

```bash
# Create config directory
mkdir -p config

# Create config file
cat > config/auto_nvis.yaml << 'EOF'
services:
  rabbitmq_host: localhost
  rabbitmq_port: 5672
  rabbitmq_user: guest
  rabbitmq_password: guest

grid:
  lat_min: -90.0
  lat_max: 90.0
  lat_step: 2.5
  lon_min: -180.0
  lon_max: 180.0
  lon_step: 5.0
  alt_min_km: 60.0
  alt_max_km: 600.0
  alt_step_km: 10.0

propagation:
  tx_lat: 40.0
  tx_lon: -105.0
  freq_min_mhz: 2.0
  freq_max_mhz: 15.0
EOF
```

### Step 4: Test Connection

```bash
# Test grid publisher
python3 src/propagation/test_grid_publisher.py
```

---

## Alternative: Docker (If you prefer containers)

### Run RabbitMQ in Docker

```bash
# Pull and run RabbitMQ with management plugin
docker run -d \
  --name rabbitmq \
  -p 5672:5672 \
  -p 15672:15672 \
  rabbitmq:3-management

# Check it's running
docker ps | grep rabbitmq

# View logs
docker logs rabbitmq
```

Then create config file as above with `rabbitmq_host: localhost`.

---

## Verify Installation

```bash
# Check RabbitMQ is listening
sudo netstat -tlnp | grep 5672

# Or using ss
ss -tlnp | grep 5672

# Expected:
# tcp  0  0  0.0.0.0:5672  0.0.0.0:*  LISTEN
```

**Access Web UI**: http://localhost:15672 (guest/guest)

---

## Troubleshooting

### Issue: "Connection refused"

**Solution**:
```bash
# Check if service is running
sudo systemctl status rabbitmq-server

# Restart if needed
sudo systemctl restart rabbitmq-server

# Check logs
sudo journalctl -u rabbitmq-server -n 50
```

### Issue: "Permission denied"

**Solution**:
```bash
# Check RabbitMQ user permissions
sudo rabbitmqctl list_users

# Add user if needed
sudo rabbitmqctl add_user autonvis password123
sudo rabbitmqctl set_permissions -p / autonvis ".*" ".*" ".*"

# Update config file with new credentials
```

### Issue: "Port already in use"

**Solution**:
```bash
# Find what's using port 5672
sudo lsof -i :5672

# Kill the process or change RabbitMQ port
```

---

## Configuration for Auto-NVIS

The Auto-NVIS system expects RabbitMQ configuration in `config/auto_nvis.yaml`:

```yaml
services:
  rabbitmq_host: localhost  # or "rabbitmq" for Docker
  rabbitmq_port: 5672
  rabbitmq_user: guest
  rabbitmq_password: guest
```

**Default values** (if no config file):
- Host: `rabbitmq` (Docker service name)
- Port: `5672`
- User: `guest`
- Password: `guest`

For **local development**, create `config/auto_nvis.yaml` with `rabbitmq_host: localhost`.

---

## Next Steps

After installation:

1. **Test publisher**:
   ```bash
   python3 src/propagation/test_grid_publisher.py
   ```

2. **Test subscriber** (in another terminal):
   ```bash
   python3 src/propagation/test_grid_subscription.py
   ```

3. **View messages** in web UI:
   - Go to http://localhost:15672
   - Click "Queues" tab
   - Look for `propagation_grid_subscriber`

---

## Production Deployment

For production, use Docker Compose:

```yaml
# docker-compose.yml
version: '3.8'

services:
  rabbitmq:
    image: rabbitmq:3-management
    ports:
      - "5672:5672"
      - "15672:15672"
    environment:
      RABBITMQ_DEFAULT_USER: autonvis
      RABBITMQ_DEFAULT_PASS: ${RABBITMQ_PASSWORD}
    volumes:
      - rabbitmq_data:/var/lib/rabbitmq
    restart: unless-stopped

volumes:
  rabbitmq_data:
```

Run with:
```bash
docker-compose up -d rabbitmq
```

---

## Summary

**Quick Setup**:
1. `sudo apt install rabbitmq-server`
2. `sudo systemctl start rabbitmq-server`
3. Create `config/auto_nvis.yaml` with `rabbitmq_host: localhost`
4. Test with `python3 src/propagation/test_grid_publisher.py`

**Status**: RabbitMQ ready for Auto-NVIS message queue! ✅
