# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p logs data models certs

# Set up configuration
COPY deployment/config.yaml /app/config/config.yaml

# Expose ports
EXPOSE 8000 9090

# Set up entrypoint
COPY deployment/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Run the application
ENTRYPOINT ["/entrypoint.sh"]
