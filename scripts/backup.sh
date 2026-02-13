#!/bin/bash
# Backup Script for AutoNVIS Data
# Usage: ./scripts/backup.sh [backup_dir]

set -e

BACKUP_DIR="${1:-/var/backups/autonvis}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "================================================"
echo "AutoNVIS Backup Script"
echo "Backup Directory: $BACKUP_DIR"
echo "Timestamp: $TIMESTAMP"
echo "================================================"

# Create backup directory
mkdir -p "$BACKUP_DIR/$TIMESTAMP"

# Backup PostgreSQL database
echo "Backing up PostgreSQL database..."
docker exec autonvis-postgres pg_dump -U autonvis autonvis | gzip > "$BACKUP_DIR/$TIMESTAMP/postgres_backup.sql.gz"

# Backup RabbitMQ definitions
echo "Backing up RabbitMQ definitions..."
curl -u autonvis:autonvis_secure_password http://localhost:15672/api/definitions > "$BACKUP_DIR/$TIMESTAMP/rabbitmq_definitions.json"

# Backup Redis data
echo "Backing up Redis data..."
docker exec autonvis-redis redis-cli SAVE
docker cp autonvis-redis:/data/dump.rdb "$BACKUP_DIR/$TIMESTAMP/redis_dump.rdb"

# Backup configuration files
echo "Backing up configuration..."
cp -r "$PROJECT_ROOT/config" "$BACKUP_DIR/$TIMESTAMP/"

# Backup logs (last 7 days)
echo "Backing up recent logs..."
if [ -d "$PROJECT_ROOT/logs" ]; then
    find "$PROJECT_ROOT/logs" -mtime -7 -type f -exec cp {} "$BACKUP_DIR/$TIMESTAMP/logs/" \;
fi

# Create backup archive
echo "Creating backup archive..."
cd "$BACKUP_DIR"
tar -czf "autonvis_backup_$TIMESTAMP.tar.gz" "$TIMESTAMP"
rm -rf "$TIMESTAMP"

# Remove old backups (keep last 30 days)
echo "Cleaning up old backups..."
find "$BACKUP_DIR" -name "autonvis_backup_*.tar.gz" -mtime +30 -delete

echo ""
echo "Backup complete: $BACKUP_DIR/autonvis_backup_$TIMESTAMP.tar.gz"
echo "Backup size: $(du -h "$BACKUP_DIR/autonvis_backup_$TIMESTAMP.tar.gz" | cut -f1)"
