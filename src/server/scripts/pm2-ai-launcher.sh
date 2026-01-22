#!/bin/bash
# Pre-startup script for AI service - kills orphaned processes on port 8765
#
# This script ensures the AI service can always bind to its port by:
# 1. Finding any processes using port 8765
# 2. Killing them (gracefully, then forcefully)
# 3. Waiting for the port to be released
# 4. Starting the AI service
#
# This prevents the "address already in use" error from orphaned uvicorn processes.

PORT=8765
MAX_WAIT=10

echo "[$(date)] AI Service launcher starting..."

# Find and kill any process using the port
PIDS=$(lsof -t -i:$PORT 2>/dev/null)
if [ -n "$PIDS" ]; then
    echo "[$(date)] Found processes on port $PORT: $PIDS"
    echo "[$(date)] Killing orphaned processes..."

    # Try graceful kill first
    echo "$PIDS" | xargs kill 2>/dev/null
    sleep 2

    # Force kill if still running
    REMAINING=$(lsof -t -i:$PORT 2>/dev/null)
    if [ -n "$REMAINING" ]; then
        echo "[$(date)] Force killing remaining processes: $REMAINING"
        echo "$REMAINING" | xargs kill -9 2>/dev/null
    fi

    # Wait for port to be released
    for i in $(seq 1 $MAX_WAIT); do
        if ! lsof -i:$PORT >/dev/null 2>&1; then
            echo "[$(date)] Port $PORT released after ${i}s"
            break
        fi
        sleep 1
    done
fi

# Verify port is free
if lsof -i:$PORT >/dev/null 2>&1; then
    echo "[$(date)] ERROR: Port $PORT still occupied after cleanup"
    lsof -i:$PORT
    exit 1
fi

echo "[$(date)] Port $PORT is free, starting AI service..."

# Activate virtual environment if it exists
if [ -f "/home/ubuntu/RingRift/ai-service/.venv/bin/activate" ]; then
    source /home/ubuntu/RingRift/ai-service/.venv/bin/activate
fi

# Start the AI service
cd /home/ubuntu/RingRift/ai-service
exec python -m uvicorn app.main:app --host 0.0.0.0 --port $PORT
