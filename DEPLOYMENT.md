# AutoNVIS NVIS System - Deployment Guide

**Version**: 1.0
**Last Updated**: 2026-02-13

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Production Deployment](#production-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Configuration](#configuration)
6. [Monitoring](#monitoring)
7. [Backup and Recovery](#backup-and-recovery)
8. [Scaling](#scaling)
9. [Security](#security)
10. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

**Minimum** (Development/Testing):
- CPU: 4 cores
- RAM: 8 GB
- Disk: 50 GB
- OS: Ubuntu 20.04+ / Debian 11+ / RHEL 8+

**Recommended** (Production):
- CPU: 8+ cores
- RAM: 16+ GB
- Disk: 200 GB SSD
- OS: Ubuntu 22.04 LTS
- Network: 1 Gbps

### Software Dependencies

- Docker 24.0+
- docker-compose 2.20+
- Git 2.30+
- (Optional) kubectl 1.27+ for Kubernetes deployment

### Install Docker

```bash
# Ubuntu/Debian
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Verify
docker --version
docker-compose --version
```

---

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/your-org/AutoNVIS.git
cd AutoNVIS
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit with your settings
nano .env
```

**Key settings to change**:
- `RABBITMQ_PASSWORD`
- `POSTGRES_PASSWORD`
- `GRAFANA_PASSWORD`

### 3. Deploy Services

```bash
# Deploy all services
./scripts/deploy.sh production

# Or using Makefile
make deploy
```

### 4. Verify Deployment

```bash
# Check service health
make health

# Or manually
./scripts/health_check.sh
```

### 5. Access Services

- **Dashboard**: http://localhost:8080
- **Grafana**: http://localhost:3000 (admin/[GRAFANA_PASSWORD])
- **Prometheus**: http://localhost:9090
- **RabbitMQ**: http://localhost:15672 (autonvis/[RABBITMQ_PASSWORD])

---

## Production Deployment

### 1. Prepare Production Server

```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install dependencies
sudo apt-get install -y \
    docker.io \
    docker-compose \
    git \
    make \
    build-essential

# Configure firewall
sudo ufw allow 8080/tcp  # Dashboard
sudo ufw allow 8001/tcp  # TCP adapter
sudo ufw allow 8002/tcp  # HTTP adapter
sudo ufw allow 3000/tcp  # Grafana
sudo ufw enable
```

### 2. Configure Production Settings

Create `.env.production`:

```bash
ENVIRONMENT=production

# Use strong passwords
RABBITMQ_PASSWORD=$(openssl rand -base64 32)
POSTGRES_PASSWORD=$(openssl rand -base64 32)
GRAFANA_PASSWORD=$(openssl rand -base64 32)

# Production RabbitMQ
RABBITMQ_HOST=rabbitmq
RABBITMQ_PORT=5672

# Production PostgreSQL
POSTGRES_HOST=postgres
POSTGRES_PORT=5432

# Enable monitoring
ENABLE_METRICS=true

# Performance tuning
MAX_WORKERS=8
FILTER_CYCLE_INTERVAL=900

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
```

### 3. Deploy with Production Config

```bash
# Load production environment
cp .env.production .env

# Deploy
./scripts/deploy.sh production

# Verify
./scripts/health_check.sh
```

### 4. Configure SSL/TLS (Recommended)

Use nginx reverse proxy:

```bash
# Install nginx
sudo apt-get install nginx certbot python3-certbot-nginx

# Configure nginx
sudo nano /etc/nginx/sites-available/autonvis
```

nginx configuration:
```nginx
server {
    listen 80;
    server_name nvis.example.com;

    location / {
        proxy_pass http://localhost:8080;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/autonvis /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx

# Get SSL certificate
sudo certbot --nginx -d nvis.example.com
```

### 5. Set Up Systemd Service

Create `/etc/systemd/system/autonvis.service`:

```ini
[Unit]
Description=AutoNVIS NVIS System
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/autonvis
ExecStart=/usr/local/bin/docker-compose up -d
ExecStop=/usr/local/bin/docker-compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable autonvis
sudo systemctl start autonvis
```

---

## Kubernetes Deployment

### 1. Create Namespace

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: autonvis
```

```bash
kubectl apply -f k8s/namespace.yaml
```

### 2. Deploy StatefulSets

```yaml
# k8s/postgres-statefulset.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  namespace: autonvis
spec:
  serviceName: postgres
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15
        env:
        - name: POSTGRES_DB
          value: autonvis
        - name: POSTGRES_USER
          valueFrom:
            secretKeyRef:
              name: autonvis-secrets
              key: postgres-user
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: autonvis-secrets
              key: postgres-password
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
  volumeClaimTemplates:
  - metadata:
      name: postgres-storage
    spec:
      accessModes: [ "ReadWriteOnce" ]
      resources:
        requests:
          storage: 100Gi
```

### 3. Deploy Application

```yaml
# k8s/nvis-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nvis-client
  namespace: autonvis
spec:
  replicas: 2
  selector:
    matchLabels:
      app: nvis-client
  template:
    metadata:
      labels:
        app: nvis-client
    spec:
      containers:
      - name: nvis-client
        image: your-registry/autonvis-nvis:latest
        ports:
        - containerPort: 8001
        - containerPort: 8002
        env:
        - name: RABBITMQ_HOST
          value: rabbitmq
        - name: RABBITMQ_PASSWORD
          valueFrom:
            secretKeyRef:
              name: autonvis-secrets
              key: rabbitmq-password
```

### 4. Deploy Services

```yaml
# k8s/dashboard-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: dashboard
  namespace: autonvis
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8080
  selector:
    app: dashboard
```

### 5. Apply All

```bash
kubectl apply -f k8s/
kubectl get pods -n autonvis
```

---

## Configuration

### Grid Configuration

Edit `config/production.yml`:

```yaml
grid:
  latitude:
    min: 25.0
    max: 50.0
    points: 26
  longitude:
    min: -125.0
    max: -65.0
    points: 61
  altitude:
    min_km: 100.0
    max_km: 500.0
    points: 41
```

### Quality Tier Configuration

```yaml
nvis_ingestion:
  quality_tiers:
    platinum:
      signal_error_db: 2.0
      delay_error_ms: 0.1
    gold:
      signal_error_db: 4.0
      delay_error_ms: 0.5
    silver:
      signal_error_db: 8.0
      delay_error_ms: 2.0
    bronze:
      signal_error_db: 15.0
      delay_error_ms: 5.0
```

### Aggregation Settings

```yaml
nvis_ingestion:
  aggregation:
    window_seconds: 60
    rate_threshold: 60  # obs/hour triggers aggregation

  rate_limiting:
    max_obs_per_cycle:
      platinum: 50
      gold: 30
      silver: 15
      bronze: 5
```

---

## Monitoring

### Grafana Dashboards

1. Access Grafana: http://localhost:3000
2. Login: admin / [GRAFANA_PASSWORD]
3. Navigate to Dashboards → NVIS System Monitoring

**Key Panels**:
- Ingestion latency (avg, P99)
- Observation throughput
- Quality tier distribution
- Information gain
- Memory and CPU usage
- Filter cycle time
- Error rate

### Prometheus Metrics

Access raw metrics: http://localhost:9090

**Key Metrics**:
```
nvis_ingestion_latency_seconds
nvis_observations_total
nvis_observations_by_tier
nvis_information_gain_total
nvis_filter_cycle_duration_seconds
nvis_errors_total
```

### Alerting

Configure alerts in `docker/prometheus.yml`:

```yaml
rule_files:
  - /etc/prometheus/alerts.yml
```

Create `docker/prometheus/alerts.yml`:

```yaml
groups:
- name: nvis_alerts
  rules:
  - alert: HighIngestionLatency
    expr: histogram_quantile(0.99, rate(nvis_ingestion_latency_seconds_bucket[5m])) > 2
    for: 5m
    annotations:
      summary: "High ingestion latency detected"

  - alert: HighErrorRate
    expr: rate(nvis_errors_total[5m]) > 0.01
    for: 5m
    annotations:
      summary: "High error rate detected"
```

---

## Backup and Recovery

### Automated Backups

Set up cron job:

```bash
# Edit crontab
crontab -e

# Add daily backup at 2 AM
0 2 * * * /opt/autonvis/scripts/backup.sh /var/backups/autonvis
```

### Manual Backup

```bash
./scripts/backup.sh /path/to/backup/dir
```

### Recovery

```bash
# Extract backup
cd /var/backups/autonvis
tar -xzf autonvis_backup_20260213_020000.tar.gz

# Restore PostgreSQL
gunzip < postgres_backup.sql.gz | docker exec -i autonvis-postgres psql -U autonvis

# Restore Redis
docker cp redis_dump.rdb autonvis-redis:/data/dump.rdb
docker-compose restart redis

# Restore RabbitMQ definitions
curl -u autonvis:password -X POST \
  -H "Content-Type: application/json" \
  http://localhost:15672/api/definitions \
  -d @rabbitmq_definitions.json
```

---

## Scaling

### Horizontal Scaling

**NVIS Client** (multiple instances):

```yaml
# docker-compose.yml
nvis-client:
  deploy:
    replicas: 3
```

**Filter Orchestrator** (active-passive):
- Only one active instance (maintains filter state)
- Use leader election for high availability

**Dashboard** (multiple instances):

```yaml
dashboard:
  deploy:
    replicas: 2
  depends_on:
    - redis  # Shared session storage
```

### Vertical Scaling

Increase resources:

```yaml
nvis-client:
  deploy:
    resources:
      limits:
        cpus: '4'
        memory: 8G
      reservations:
        cpus: '2'
        memory: 4G
```

---

## Security

### Network Security

```bash
# Restrict external access
sudo ufw default deny incoming
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

### Container Security

```yaml
# Run as non-root
security_opt:
  - no-new-privileges:true
user: "1000:1000"

# Read-only root filesystem
read_only: true
tmpfs:
  - /tmp
```

### Secrets Management

Use Docker secrets or Kubernetes secrets:

```bash
# Create secrets
echo "supersecret" | docker secret create rabbitmq_password -

# Use in compose
services:
  nvis-client:
    secrets:
      - rabbitmq_password
```

---

## Troubleshooting

### Service Won't Start

```bash
# Check logs
docker-compose logs service-name

# Check resource usage
docker stats

# Restart service
docker-compose restart service-name
```

### High Memory Usage

```bash
# Check container memory
docker stats

# Increase limits
# Edit docker-compose.yml
services:
  nvis-client:
    mem_limit: 8g
```

### Database Connection Errors

```bash
# Check PostgreSQL
docker-compose logs postgres

# Verify connectivity
docker-compose exec filter-orchestrator \
  psql -h postgres -U autonvis -d autonvis
```

### RabbitMQ Queue Buildup

```bash
# Check queue depth
curl -u autonvis:password \
  http://localhost:15672/api/queues

# Purge queue (careful!)
curl -u autonvis:password -X DELETE \
  http://localhost:15672/api/queues/%2F/obs.nvis_sounder/contents
```

---

## Support

- **Documentation**: https://docs.autonvis.org
- **GitHub Issues**: https://github.com/your-org/AutoNVIS/issues
- **Email**: support@autonvis.org

---

**Deployment Complete! 🚀**
