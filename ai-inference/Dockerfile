# RingRift AI Inference Service
# Minimal container for production AI move serving

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy ai-service (for imports) - only the app/ directory
COPY ai-service/app /app/ai-service/app
COPY ai-service/requirements.txt /app/ai-service/

# Copy inference service
COPY ai-inference/inference /app/ai-inference/inference
COPY ai-inference/requirements.txt /app/ai-inference/

# Install dependencies
# Note: ai-service requirements includes torch, numpy, etc.
RUN pip install --no-cache-dir -r /app/ai-service/requirements.txt
RUN pip install --no-cache-dir -r /app/ai-inference/requirements.txt

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
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
