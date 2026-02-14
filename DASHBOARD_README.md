# AutoNVIS Dashboard

Real-time ionospheric monitoring and NVIS propagation visualization system.

## Quick Start

### Using Docker (Recommended)

```bash
# Set RabbitMQ password
export RABBITMQ_PASSWORD="your_secure_password"

# Start services
docker-compose -f docker-compose.dashboard.yml up -d

# Check status
docker-compose -f docker-compose.dashboard.yml ps

# View logs
docker-compose -f docker-compose.dashboard.yml logs -f dashboard

# Access dashboard
open http://localhost:8080
```

### Manual Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Configure system
cp config/production.yml.example config/production.yml
# Edit config/production.yml with your settings

# Start dashboard
python -m src.output.dashboard.main
```

## Dashboard Views

1. **Overview** (`/`) - System summary and alerts
2. **Ionosphere** (`/ionosphere`) - 3D electron density visualization
3. **Propagation** (`/propagation`) - LUF/MUF/FOT frequency plans
4. **Space Weather** (`/spaceweather`) - X-ray flux and solar wind
5. **Network** (`/network`) - NVIS sounder analysis
6. **Control** (`/control`) - System management

## Features

- **Real-Time Updates**: WebSocket-based live data streaming
- **Interactive Visualizations**: Plotly.js 3D charts with zoom/pan
- **Ionospheric Parameters**: foF2, hmF2, TEC calculations
- **Propagation Prediction**: LUF/MUF/FOT tracking
- **Space Weather**: Flare detection and solar wind monitoring
- **System Control**: Service management and configuration

## API Endpoints

```
GET  /api/ionosphere/grid/metadata
GET  /api/ionosphere/slice/horizontal?altitude_km=300
GET  /api/ionosphere/profile/vertical?latitude=40&longitude=-105
GET  /api/ionosphere/parameters/fof2
GET  /api/propagation/frequency_plan/latest
GET  /api/spaceweather/xray/history?hours=24
POST /api/control/filter/trigger_cycle
...
```

Full API documentation: [docs/DASHBOARD_USER_GUIDE.md](docs/DASHBOARD_USER_GUIDE.md)

## Architecture

```
┌─────────────────┐
│   Web Browser   │
│  (Dashboard UI) │
└────────┬────────┘
         │ HTTP/WebSocket
┌────────┴────────┐
│  FastAPI Server │
│  (Dashboard)    │
└────────┬────────┘
         │ AMQP
┌────────┴────────┐
│    RabbitMQ     │
│  (Message Bus)  │
└────────┬────────┘
         │
    ┌────┴────┬────────┬──────────┐
    │         │        │          │
┌───┴──┐  ┌──┴───┐ ┌──┴────┐ ┌───┴────┐
│SR-UKF│  │ GNSS │ │ GOES  │ │  NVIS  │
│Filter│  │  TEC │ │ X-Ray │ │Sounders│
└──────┘  └──────┘ └───────┘ └────────┘
```

## Configuration

Edit `config/production.yml`:

```yaml
grid:
  lat_min: -90.0
  lat_max: 90.0
  lat_step: 2.5
  # ...

services:
  rabbitmq_host: "rabbitmq"
  rabbitmq_port: 5672
  # ...
```

## Monitoring

### Health Check

```bash
curl http://localhost:8080/api/control/health/check
```

### Logs

```bash
# Docker
docker-compose -f docker-compose.dashboard.yml logs -f

# Manual
tail -f /var/log/autonvis/dashboard.log
```

### Metrics

- Dashboard status: Sidebar connection indicator
- Grid age: Shown in Overview and Ionosphere views
- RabbitMQ: Management UI at http://localhost:15672

## Performance

- **Grid Updates**: < 100ms processing time
- **foF2 Calculation**: < 1s for full 73×73×55 grid
- **API Response**: < 500ms (excluding large grid transfers)
- **WebSocket Latency**: < 100ms for broadcasts

## Security

**Production Deployment**:
1. Use HTTPS (SSL/TLS certificates)
2. Set strong RabbitMQ password
3. Enable firewall (ufw/iptables)
4. Add authentication (Nginx basic auth or OAuth)
5. Regular security updates

See [docs/DASHBOARD_DEPLOYMENT.md](docs/DASHBOARD_DEPLOYMENT.md) for details.

## Testing

```bash
# Run integration tests
pytest tests/dashboard/test_integration.py -v

# Performance tests
pytest tests/dashboard/test_integration.py::TestPerformance -v
```

## Troubleshooting

### Dashboard Won't Start

```bash
# Check RabbitMQ
docker ps | grep rabbitmq

# Check logs
docker-compose -f docker-compose.dashboard.yml logs dashboard

# Verify configuration
python -c "from src.common.config import get_config; print(get_config())"
```

### No Data Displayed

1. Check WebSocket connection (green indicator in sidebar)
2. Verify SR-UKF filter is publishing grids
3. Check RabbitMQ queues: http://localhost:15672

### WebSocket Disconnected

1. Refresh browser page
2. Check Nginx WebSocket configuration
3. Verify firewall allows port 8080

See full troubleshooting guide in [docs/DASHBOARD_USER_GUIDE.md](docs/DASHBOARD_USER_GUIDE.md)

## Documentation

- **User Guide**: [docs/DASHBOARD_USER_GUIDE.md](docs/DASHBOARD_USER_GUIDE.md)
- **Deployment**: [docs/DASHBOARD_DEPLOYMENT.md](docs/DASHBOARD_DEPLOYMENT.md)
- **API Reference**: http://localhost:8080/docs (FastAPI auto-generated)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## License

See [LICENSE](LICENSE) file.

## Version

**v2.0.0** - Comprehensive GUI Dashboard Implementation

## Contact

- Issues: https://github.com/yourusername/autonvis/issues
- Documentation: https://autonvis.readthedocs.io
- Email: support@autonvis.example.com
