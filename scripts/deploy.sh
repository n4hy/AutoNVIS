#!/bin/bash
# AutoNVIS Deployment Script
# Usage: ./scripts/deploy.sh [environment]

set -e

ENVIRONMENT="${1:-production}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "================================================"
echo "AutoNVIS Deployment Script"
echo "Environment: $ENVIRONMENT"
echo "================================================"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    print_error "docker-compose is not installed. Please install docker-compose first."
    exit 1
fi

# Load environment variables
if [ -f "$PROJECT_ROOT/.env.$ENVIRONMENT" ]; then
    print_info "Loading environment variables from .env.$ENVIRONMENT"
    export $(grep -v '^#' "$PROJECT_ROOT/.env.$ENVIRONMENT" | xargs)
else
    print_warning "No .env.$ENVIRONMENT file found. Using defaults."
fi

# Create necessary directories
print_info "Creating directories..."
mkdir -p "$PROJECT_ROOT/logs"
mkdir -p "$PROJECT_ROOT/data"
mkdir -p "$PROJECT_ROOT/docker/grafana/dashboards"
mkdir -p "$PROJECT_ROOT/docker/grafana/datasources"

# Stop existing containers
print_info "Stopping existing containers..."
cd "$PROJECT_ROOT"
docker-compose down || true

# Pull latest images (if using registry)
if [ "$ENVIRONMENT" = "production" ]; then
    print_info "Pulling latest images..."
    docker-compose pull || print_warning "Could not pull images, will build locally"
fi

# Build images
print_info "Building Docker images..."
docker-compose build

# Start services
print_info "Starting services..."
docker-compose up -d

# Wait for services to be healthy
print_info "Waiting for services to become healthy..."
sleep 10

# Check service health
print_info "Checking service health..."
services=("rabbitmq" "postgres" "redis" "nvis-client" "filter-orchestrator" "dashboard")
all_healthy=true

for service in "${services[@]}"; do
    if docker-compose ps | grep "$service" | grep -q "Up"; then
        print_info "✓ $service is running"
    else
        print_error "✗ $service is not running"
        all_healthy=false
    fi
done

if [ "$all_healthy" = true ]; then
    print_info "All services are running!"
    echo ""
    print_info "Service URLs:"
    echo "  - Dashboard:        http://localhost:8080"
    echo "  - Grafana:          http://localhost:3000 (admin/admin)"
    echo "  - Prometheus:       http://localhost:9090"
    echo "  - RabbitMQ:         http://localhost:15672 (autonvis/autonvis_secure_password)"
    echo ""
    print_info "Logs: docker-compose logs -f"
else
    print_error "Some services failed to start. Check logs with: docker-compose logs"
    exit 1
fi

# Run database migrations (if applicable)
if [ "$ENVIRONMENT" = "production" ]; then
    print_info "Running database migrations..."
    # docker-compose exec filter-orchestrator python -m alembic upgrade head || print_warning "Migrations not configured"
fi

# Display service status
echo ""
print_info "Service Status:"
docker-compose ps

echo ""
print_info "Deployment complete!"
