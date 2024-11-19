#!/bin/bash

# Exit on error
set -e

# Function to wait for a service to be ready
wait_for_service() {
    local host="$1"
    local port="$2"
    local service="$3"
    
    echo "Waiting for $service to be ready..."
    while ! nc -z "$host" "$port"; do
        sleep 1
    done
    echo "$service is ready!"
}

# Check if Redis is required and available
if [ "${CACHE_TYPE:-redis}" = "redis" ]; then
    wait_for_service "${CACHE_HOST:-localhost}" "${CACHE_PORT:-6379}" "Redis"
fi

# Apply any database migrations if needed
# python manage.py migrate --noinput

# Start Prometheus metrics server if enabled
if [ "${ENABLE_METRICS:-true}" = "true" ]; then
    prometheus_multiproc_dir=/tmp/prometheus-metrics
    mkdir -p "$prometheus_multiproc_dir"
    export prometheus_multiproc_dir
fi

# Start the application with Gunicorn
exec gunicorn \
    --bind "${HOST:-0.0.0.0}:${PORT:-8000}" \
    --workers "${WORKERS:-4}" \
    --worker-class uvicorn.workers.UvicornWorker \
    --timeout "${TIMEOUT:-300}" \
    --access-logfile - \
    --error-logfile - \
    --log-level "${LOG_LEVEL:-info}" \
    "src.main:app"
