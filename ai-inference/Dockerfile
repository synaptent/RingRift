# RingRift AI Inference Service
# Minimal container for production AI move serving

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy ai-service (for imports) - only the app/ directory
COPY ai-service/app /app/ai-service/app

# Copy inference service
COPY ai-inference/inference /app/ai-inference/inference
COPY ai-inference/requirements.txt /app/ai-inference/

# Install inference requirements first (includes torch)
RUN pip install --no-cache-dir -r /app/ai-inference/requirements.txt

# Install additional ai-service dependencies needed for inference
# Skip optional GNN packages (torch-geometric, torch-scatter, torch-sparse)
RUN pip install --no-cache-dir \
    pyyaml \
    aiohttp \
    aiosqlite \
    scipy \
    psutil

# Copy pre-trained models (these should be baked in or mounted)
COPY ai-inference/models /app/ai-inference/models

# Set Python path to include ai-service
ENV PYTHONPATH=/app/ai-service:/app/ai-inference
ENV RINGRIFT_MODELS_PATH=/app/ai-inference/models

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Run the inference server
CMD ["python", "-m", "uvicorn", "inference.main:app", "--host", "0.0.0.0", "--port", "8001"]
