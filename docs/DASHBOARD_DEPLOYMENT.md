# AutoNVIS Dashboard Deployment Guide

## Overview

This guide covers deploying the AutoNVIS Dashboard to production environments.

---

## Prerequisites

### System Requirements

- **OS**: Linux (Ubuntu 20.04+ recommended) or Docker
- **Python**: 3.8+
- **Memory**: 4GB minimum, 8GB recommended
- **CPU**: 2+ cores recommended
- **Disk**: 10GB minimum
- **Network**: Port 8080 accessible

### Dependencies

- RabbitMQ 3.8+
- Redis 6.0+ (optional, for caching)
- NVIS data sources running
- SR-UKF filter service running

---

## Installation

### 1. Clone Repository

```bash
cd /opt
git clone https://github.com/yourusername/autonvis.git
cd autonvis
```

### 2. Install Python Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure System

```bash
# Copy configuration template
cp config/production.yml.example config/production.yml

# Edit configuration
nano config/production.yml
```

**Key Configuration Items**:

```yaml
services:
  rabbitmq_host: "rabbitmq"  # or localhost
  rabbitmq_port: 5672
  rabbitmq_user: "guest"
  rabbitmq_password: "CHANGE_ME"

grid:
  lat_min: -90.0
  lat_max: 90.0
  lat_step: 2.5
  lon_min: -180.0
  lon_max: 180.0
  lon_step: 5.0
  alt_min: 60.0
  alt_max: 600.0
  alt_step: 10.0
```

---

## Deployment Options

### Option 1: Docker Compose (Recommended)

**Advantages**:
- Isolated environment
- Easy scaling
- Automatic restart
- Production-ready

#### docker-compose.yml

```yaml
version: '3.8'

services:
  dashboard:
    build:
      context: .
      dockerfile: docker/Dockerfile.dashboard
    ports:
      - "8080:8080"
    environment:
      - AUTONVIS_CONFIG=/config/production.yml
      - PYTHONUNBUFFERED=1
    volumes:
      - ./config:/config:ro
      - ./logs:/var/log/autonvis
    depends_on:
      - rabbitmq
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/api/control/health/check"]
      interval: 30s
      timeout: 10s
      retries: 3

  rabbitmq:
    image: rabbitmq:3.11-management
    ports:
      - "5672:5672"
      - "15672:15672"
    environment:
      - RABBITMQ_DEFAULT_USER=autonvis
      - RABBITMQ_DEFAULT_PASS=${RABBITMQ_PASSWORD}
    volumes:
      - rabbitmq_data:/var/lib/rabbitmq
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
    depends_on:
      - dashboard
    restart: unless-stopped

volumes:
  rabbitmq_data:
```

#### Deploy

```bash
# Set environment variables
export RABBITMQ_PASSWORD="your_secure_password"

# Build and start
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f dashboard
```

### Option 2: Systemd Service

**Advantages**:
- Native Linux integration
- System logging
- Automatic startup

#### Create Service File

```bash
sudo nano /etc/systemd/system/autonvis-dashboard.service
```

```ini
[Unit]
Description=AutoNVIS Dashboard Service
After=network.target rabbitmq-server.service
Requires=rabbitmq-server.service

[Service]
Type=simple
User=autonvis
Group=autonvis
WorkingDirectory=/opt/autonvis
Environment="AUTONVIS_CONFIG=/opt/autonvis/config/production.yml"
ExecStart=/opt/autonvis/venv/bin/python -m src.output.dashboard.main --host 0.0.0.0 --port 8080
Restart=always
RestartSec=10

# Logging
StandardOutput=append:/var/log/autonvis/dashboard.log
StandardError=append:/var/log/autonvis/dashboard_error.log

# Security
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/log/autonvis

[Install]
WantedBy=multi-user.target
```

#### Enable and Start

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable service
sudo systemctl enable autonvis-dashboard

# Start service
sudo systemctl start autonvis-dashboard

# Check status
sudo systemctl status autonvis-dashboard

# View logs
sudo journalctl -u autonvis-dashboard -f
```

### Option 3: Manual Deployment

```bash
# Navigate to project directory
cd /opt/autonvis

# Activate virtual environment
source venv/bin/activate

# Start dashboard
python -m src.output.dashboard.main \
  --host 0.0.0.0 \
  --port 8080 \
  --config config/production.yml
```

**Run in Background**:

```bash
nohup python -m src.output.dashboard.main \
  --host 0.0.0.0 \
  --port 8080 \
  --config config/production.yml \
  >> /var/log/autonvis/dashboard.log 2>&1 &
```

---

## Reverse Proxy Configuration

### Nginx

```nginx
# /etc/nginx/sites-available/autonvis

upstream dashboard {
    server localhost:8080;
}

server {
    listen 80;
    server_name autonvis.example.com;

    # Redirect to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name autonvis.example.com;

    # SSL certificates
    ssl_certificate /etc/nginx/ssl/autonvis.crt;
    ssl_certificate_key /etc/nginx/ssl/autonvis.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # Logging
    access_log /var/log/nginx/autonvis_access.log;
    error_log /var/log/nginx/autonvis_error.log;

    # Static files
    location /static/ {
        alias /opt/autonvis/src/output/dashboard/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # WebSocket
    location /ws {
        proxy_pass http://dashboard;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 3600s;
    }

    # API and pages
    location / {
        proxy_pass http://dashboard;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Compression
        gzip on;
        gzip_types application/json text/css application/javascript;
    }

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN";
    add_header X-Content-Type-Options "nosniff";
    add_header X-XSS-Protection "1; mode=block";
}
```

Enable site:

```bash
sudo ln -s /etc/nginx/sites-available/autonvis /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

---

## Security Configuration

### 1. Firewall

```bash
# Allow HTTP/HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Allow dashboard port (if direct access needed)
sudo ufw allow 8080/tcp

# Enable firewall
sudo ufw enable
```

### 2. SSL/TLS Certificates

**Let's Encrypt (Free)**:

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d autonvis.example.com
```

**Self-Signed (Development)**:

```bash
sudo mkdir -p /etc/nginx/ssl
sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout /etc/nginx/ssl/autonvis.key \
  -out /etc/nginx/ssl/autonvis.crt
```

### 3. Authentication (Optional)

Add basic auth to Nginx:

```bash
sudo apt install apache2-utils
sudo htpasswd -c /etc/nginx/.htpasswd admin
```

Update Nginx config:

```nginx
location / {
    auth_basic "AutoNVIS Dashboard";
    auth_basic_user_file /etc/nginx/.htpasswd;
    ...
}
```

---

## Monitoring & Logging

### Health Checks

```bash
# Dashboard health endpoint
curl http://localhost:8080/api/control/health/check

# Expected response:
# {"status": "healthy", "grid_age_seconds": 120, ...}
```

### Log Files

```bash
# Dashboard logs
tail -f /var/log/autonvis/dashboard.log

# Error logs
tail -f /var/log/autonvis/dashboard_error.log

# Nginx logs
tail -f /var/log/nginx/autonvis_access.log
tail -f /var/log/nginx/autonvis_error.log
```

### Prometheus Metrics (Optional)

Add to dashboard code:

```python
from prometheus_client import Counter, Histogram, generate_latest

# Metrics
api_requests = Counter('dashboard_api_requests_total', 'Total API requests')
api_latency = Histogram('dashboard_api_latency_seconds', 'API latency')

# Expose metrics endpoint
@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

### Alerting

Configure alerts for:
- Dashboard down (health check fails)
- High error rate (>5% of requests)
- Stale data (grid age > 30 minutes)
- RabbitMQ connection lost
- High memory usage (>80%)

---

## Performance Tuning

### 1. Python Settings

```python
# In main.py
uvicorn.run(
    app,
    host="0.0.0.0",
    port=8080,
    workers=4,  # Number of worker processes
    log_level="info",
    access_log=False,  # Disable for production
    limit_concurrency=1000,
    timeout_keep_alive=5
)
```

### 2. Caching

Enable response caching in production:

```python
from backend.performance import ResponseCache, cached_response

# 60-second cache for expensive operations
parameter_cache = ResponseCache(ttl_seconds=60)

@cached_response(parameter_cache)
async def get_fof2_map():
    # Expensive calculation cached for 60s
    ...
```

### 3. Database Connection Pool

If using PostgreSQL for history:

```python
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    "postgresql://user:pass@localhost/autonvis",
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=10
)
```

### 4. Compression

Enable gzip in Nginx (see config above) or in FastAPI:

```python
from fastapi.middleware.gzip import GZipMiddleware

app.add_middleware(GZipMiddleware, minimum_size=1000)
```

---

## Backup & Recovery

### 1. Configuration Backup

```bash
# Backup configuration
tar -czf autonvis-config-$(date +%Y%m%d).tar.gz config/

# Automated daily backup
echo "0 2 * * * /opt/autonvis/scripts/backup-config.sh" | crontab -
```

### 2. State Backup

Dashboard state is in-memory, but configuration and logs should be backed up:

```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backup/autonvis/$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR

# Configuration
cp -r /opt/autonvis/config $BACKUP_DIR/

# Logs (last 7 days)
find /var/log/autonvis -mtime -7 -type f -exec cp {} $BACKUP_DIR/ \;

# Compress
tar -czf $BACKUP_DIR.tar.gz $BACKUP_DIR
rm -rf $BACKUP_DIR

# Keep only last 30 days
find /backup/autonvis -name "*.tar.gz" -mtime +30 -delete
```

---

## Troubleshooting

### Dashboard Won't Start

```bash
# Check Python dependencies
pip list | grep fastapi

# Check port availability
sudo netstat -tlnp | grep 8080

# Check configuration syntax
python -c "from src.common.config import get_config; get_config('config/production.yml')"

# Check logs
tail -100 /var/log/autonvis/dashboard_error.log
```

### High Memory Usage

```bash
# Check process memory
ps aux | grep dashboard

# Limit workers if needed
# In systemd service:
Environment="UVICORN_WORKERS=2"

# Or reduce cache size in code
state = DashboardState(retention_hours=12)  # Reduce from 24
```

### WebSocket Issues

```bash
# Test WebSocket connection
wscat -c ws://localhost:8080/ws

# Check Nginx WebSocket config
sudo nginx -T | grep -A 10 "location /ws"

# Verify firewall allows WebSocket
sudo ufw status | grep 8080
```

---

## Upgrade Procedure

### 1. Backup Current Version

```bash
# Stop service
sudo systemctl stop autonvis-dashboard

# Backup
cp -r /opt/autonvis /opt/autonvis-backup-$(date +%Y%m%d)
```

### 2. Update Code

```bash
cd /opt/autonvis
git pull origin main
```

### 3. Update Dependencies

```bash
source venv/bin/activate
pip install -r requirements.txt --upgrade
```

### 4. Database Migrations (if any)

```bash
# Run migrations
alembic upgrade head
```

### 5. Restart Service

```bash
sudo systemctl start autonvis-dashboard
sudo systemctl status autonvis-dashboard
```

### 6. Verify

```bash
curl http://localhost:8080/api/control/health/check
```

---

## Production Checklist

Before going live:

- [ ] SSL/TLS certificates installed
- [ ] Firewall configured
- [ ] Reverse proxy configured (Nginx)
- [ ] Systemd service enabled
- [ ] Log rotation configured
- [ ] Monitoring set up
- [ ] Backups automated
- [ ] Health checks configured
- [ ] Performance tuning applied
- [ ] Security headers added
- [ ] Documentation updated
- [ ] Team trained on dashboard usage

---

## Contact

For deployment support:
- Documentation: https://autonvis.readthedocs.io
- Issues: https://github.com/yourusername/autonvis/issues
