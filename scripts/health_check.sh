#!/bin/bash
# Health Check Script for AutoNVIS Services
# Usage: ./scripts/health_check.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_status() {
    local status=$1
    local message=$2
    if [ "$status" = "OK" ]; then
        echo -e "${GREEN}✓${NC} $message"
    elif [ "$status" = "WARN" ]; then
        echo -e "${YELLOW}⚠${NC} $message"
    else
        echo -e "${RED}✗${NC} $message"
    fi
}

echo "================================================"
echo "AutoNVIS Health Check"
echo "================================================"
echo ""

# Check Docker services
echo "Docker Services:"
if docker-compose -f "$PROJECT_ROOT/docker-compose.yml" ps | grep -q "Up"; then
    services=$(docker-compose -f "$PROJECT_ROOT/docker-compose.yml" ps --services)
    for service in $services; do
        status=$(docker-compose -f "$PROJECT_ROOT/docker-compose.yml" ps $service | grep "Up" || echo "Down")
        if echo "$status" | grep -q "Up"; then
            # Check if healthy
            health=$(docker inspect --format='{{.State.Health.Status}}' "autonvis-$service" 2>/dev/null || echo "unknown")
            if [ "$health" = "healthy" ] || [ "$health" = "unknown" ]; then
                print_status "OK" "$service (running)"
            else
                print_status "WARN" "$service (unhealthy: $health)"
            fi
        else
            print_status "FAIL" "$service (not running)"
        fi
    done
else
    print_status "FAIL" "No services running"
fi

echo ""

# Check API endpoints
echo "API Endpoints:"

# Dashboard
if curl -sf http://localhost:8080/health > /dev/null 2>&1; then
    print_status "OK" "Dashboard API (http://localhost:8080)"
else
    print_status "FAIL" "Dashboard API (http://localhost:8080)"
fi

# RabbitMQ Management
if curl -sf http://localhost:15672/api/health/checks/alarms > /dev/null 2>&1; then
    print_status "OK" "RabbitMQ Management (http://localhost:15672)"
else
    print_status "FAIL" "RabbitMQ Management (http://localhost:15672)"
fi

# Prometheus
if curl -sf http://localhost:9090/-/healthy > /dev/null 2>&1; then
    print_status "OK" "Prometheus (http://localhost:9090)"
else
    print_status "FAIL" "Prometheus (http://localhost:9090)"
fi

# Grafana
if curl -sf http://localhost:3000/api/health > /dev/null 2>&1; then
    print_status "OK" "Grafana (http://localhost:3000)"
else
    print_status "FAIL" "Grafana (http://localhost:3000)"
fi

echo ""

# Check resource usage
echo "Resource Usage:"

# Get container stats
stats=$(docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}" | grep autonvis || echo "")

if [ -n "$stats" ]; then
    echo "$stats"
else
    print_status "WARN" "No container stats available"
fi

echo ""

# Check disk usage
echo "Disk Usage:"
disk_usage=$(df -h "$PROJECT_ROOT" | tail -1 | awk '{print $5}' | sed 's/%//')
if [ "$disk_usage" -lt 80 ]; then
    print_status "OK" "Disk usage: ${disk_usage}%"
elif [ "$disk_usage" -lt 90 ]; then
    print_status "WARN" "Disk usage: ${disk_usage}% (consider cleanup)"
else
    print_status "FAIL" "Disk usage: ${disk_usage}% (critical)"
fi

echo ""

# Check logs for errors
echo "Recent Errors:"
error_count=$(docker-compose -f "$PROJECT_ROOT/docker-compose.yml" logs --tail=100 2>/dev/null | grep -i "error" | wc -l || echo "0")
if [ "$error_count" -eq 0 ]; then
    print_status "OK" "No errors in recent logs"
elif [ "$error_count" -lt 5 ]; then
    print_status "WARN" "$error_count errors in recent logs"
else
    print_status "FAIL" "$error_count errors in recent logs (check: docker-compose logs)"
fi

echo ""
echo "================================================"
echo "Health Check Complete"
echo "================================================"
