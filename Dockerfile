# Base image with Python
FROM python:3.9-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Application
FROM python:3.9-slim

WORKDIR /app

# Copy Python packages from base stage
COPY --from=base /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=base /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ ./src/
COPY models/ ./models/
COPY data/processed/ ./data/processed/

# Create necessary directories
RUN mkdir -p models data/processed

# Expose ports
# 5000 for Flask API
# 8501 for Streamlit
EXPOSE 5000 8501

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/api/health')" || exit 1

# Copy startup script
COPY docker-entrypoint.sh /
RUN chmod +x /docker-entrypoint.sh

# Run both Flask and Streamlit
ENTRYPOINT ["/docker-entrypoint.sh"]