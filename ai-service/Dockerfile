# Python AI Service Dockerfile for RingRift

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    AI_SERVICE_PORT=8001

# Install compilers required by source-only networking dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install the supported base runtime. Optional PyG native extensions remain in
# requirements-gnn.txt and are intentionally excluded from the default image.
RUN pip install --no-cache-dir torch==2.13.0 && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Expose port (container-internal; host mapping is controlled by docker-compose)
EXPOSE 8001

# Health check – use the configured AI_SERVICE_PORT and httpx (already in requirements)
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import os, httpx; port = os.getenv('AI_SERVICE_PORT', '8001'); resp = httpx.get(f'http://localhost:{port}/health', timeout=2.0); raise SystemExit(0 if resp.status_code == 200 else 1)"

# Run the application (no hot-reload in containers; bind to AI_SERVICE_PORT)
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${AI_SERVICE_PORT}"]
