# Auto-NVIS Development Guide

This guide provides information for developers working on the Auto-NVIS system.

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.11+
- C++17 compiler (for local development)
- CMake 3.20+
- Git

### Initial Setup

1. **Clone the repository** (if not already done)
   ```bash
   git clone <repository-url>
   cd AutoNVIS
   ```

2. **Install Python dependencies** (for local development)
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Start services with Docker Compose**
   ```bash
   cd docker
   docker-compose up -d
   ```

4. **Check service health**
   ```bash
   docker-compose ps
   ```

### Access Points

- **RabbitMQ Management UI**: http://localhost:15672 (guest/guest)
- **Supervisor API**: http://localhost:8000
- **Output Dashboard**: http://localhost:8080
- **Redis**: localhost:6379

## Project Structure

```
AutoNVIS/
├── src/
│   ├── common/              # Shared utilities
│   │   ├── config.py        # Configuration management
│   │   ├── constants.py     # Physical constants
│   │   ├── geodesy.py       # Coordinate transformations
│   │   ├── logging_config.py # Structured logging
│   │   └── message_queue.py # Message queue abstraction
│   │
│   ├── ingestion/           # Data ingestion services
│   │   ├── space_weather/   # GOES, ACE clients
│   │   ├── gnss_tec/        # GNSS-TEC processing
│   │   ├── ionosonde/       # GIRO ionosonde client
│   │   └── common/          # Ingestion utilities
│   │
│   ├── assimilation/        # SR-UKF core (C++)
│   │   ├── include/         # Header files
│   │   ├── src/             # Implementation
│   │   ├── models/          # Physics models
│   │   ├── tests/           # Unit tests
│   │   └── CMakeLists.txt   # Build configuration
│   │
│   ├── supervisor/          # Autonomous control logic
│   ├── propagation/         # PHaRLAP integration
│   └── output/              # Output generation & dashboard
│
├── docker/                  # Docker configurations
├── config/                  # YAML configuration files
├── data/                    # Runtime data storage
├── tests/                   # Integration & validation tests
└── docs/                    # Documentation

```

## Development Workflow

### Running Individual Services

**Ingestion Service (Space Weather Monitor)**
```bash
python -m src.ingestion.main
```

**Supervisor Service**
```bash
python -m src.supervisor.main
```

**Output Service**
```bash
uvicorn src.output.web_dashboard.app:app --reload
```

### Building C++ Assimilation Core

```bash
cd src/assimilation
mkdir build && cd build
cmake ..
make -j$(nproc)
ctest  # Run unit tests
```

### Running Tests

**Python tests**
```bash
pytest tests/unit/
pytest tests/integration/
```

**C++ tests**
```bash
cd src/assimilation/build
ctest --verbose
```

## Configuration

Configuration is managed through YAML files in the `config/` directory:

- `production.yml`: Production configuration
- `development.yml`: Development overrides (create as needed)

Override configuration via environment variable:
```bash
export AUTONVIS_CONFIG=/path/to/config.yml
```

## Message Queue Topics

Standard topics defined in `src/common/message_queue.py`:

**Space Weather**
- `wx.xray` - GOES X-ray flux
- `wx.solar_wind` - ACE solar wind data
- `wx.geomag` - Geomagnetic indices

**Observations**
- `obs.gnss_tec` - GNSS-TEC measurements
- `obs.ionosonde` - Ionosonde parameters

**Control**
- `ctrl.mode_change` - Mode switching events
- `ctrl.cycle_trigger` - Cycle trigger commands

**Output**
- `out.frequency_plan` - Generated frequency plans
- `out.coverage_map` - Coverage maps
- `out.alert` - System alerts

## Logging

All services use structured JSON logging. Logs are sent to stdout and can be aggregated using tools like Promtail/Loki.

Example log entry:
```json
{
  "timestamp": "2026-02-11T12:00:00Z",
  "level": "INFO",
  "service": "ingestion",
  "component": "goes_xray",
  "message": "Fetched X-ray flux: M2.5",
  "module": "goes_xray_client",
  "function": "fetch_latest"
}
```

## Data Flow

1. **Ingestion** services fetch real-time data
2. **Publish** to message queue topics
3. **Supervisor** monitors space weather and triggers cycles
4. **SR-UKF** assimilates observations into state estimate
5. **PHaRLAP** performs ray tracing with updated ionosphere
6. **Output** generates frequency plans and coverage maps

## Debugging

**View RabbitMQ messages**
```bash
# Install rabbitmqadmin
wget http://localhost:15672/cli/rabbitmqadmin
chmod +x rabbitmqadmin

# List queues
./rabbitmqadmin list queues

# Get messages (non-destructive)
./rabbitmqadmin get queue=<queue_name> requeue=true
```

**View Redis state**
```bash
redis-cli
> KEYS *
> GET <key>
```

**View service logs**
```bash
docker-compose logs -f ingestion
docker-compose logs -f supervisor
docker-compose logs -f assimilation
```

## Next Steps

### Phase 1: Foundation (Completed)
- [x] Directory structure
- [x] Common utilities
- [x] Docker Compose configuration
- [x] Configuration management

### Phase 2: Data Ingestion (In Progress)
- [ ] Implement GOES X-ray client
- [ ] Implement ACE solar wind client
- [ ] Create data validators
- [ ] Build ingestion orchestrator

### Phase 3: Supervisor Logic (Upcoming)
- [ ] Mode controller
- [ ] System orchestrator
- [ ] Health monitor
- [ ] Alert generator

### Phase 4: SR-UKF Core (Upcoming)
- [ ] State vector implementation
- [ ] Sigma point generation
- [ ] Physics model interface
- [ ] SR-UKF algorithm
- [ ] IRI-2020 integration

## Contributing

1. Create a feature branch from `main`
2. Implement changes with tests
3. Run linting: `black src/` and `flake8 src/`
4. Run tests: `pytest`
5. Create pull request

## Resources

- [Plan Document](/.claude/plans/spicy-wishing-lecun.md)
- [AutoNVIS PDF](AutoNVIS.pdf) - System architecture and theory
- [README](README.md) - Project overview

## Support

For questions or issues, please consult the plan document or create an issue in the repository.
