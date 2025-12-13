#!/bin/bash
# Run script for RingRift AI Service
# Starts the FastAPI server with hot reload

set -e  # Exit on error

echo "🚀 Starting RingRift AI Service..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run ./setup.sh first to create the environment."
    exit 1
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Prefer local CMA-ES trained heuristic weights when available.
# This keeps sandbox play aligned with the production ladder without requiring
# manual env-var setup for local development.
if [ -z "${RINGRIFT_TRAINED_HEURISTIC_PROFILES:-}" ] && [ -f "data/trained_heuristic_profiles.json" ]; then
    export RINGRIFT_TRAINED_HEURISTIC_PROFILES="$(pwd)/data/trained_heuristic_profiles.json"
fi

# Check if uvicorn is installed
if ! python -c "import uvicorn" &> /dev/null; then
    echo "❌ uvicorn not found in virtual environment!"
    echo "Please run ./setup.sh to install dependencies."
    exit 1
fi

# Start the server
echo "🌐 Starting AI service on http://localhost:8001"
echo "📚 API documentation: http://localhost:8001/docs"
echo "❤️  Health check: http://localhost:8001/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python -m uvicorn app.main:app --reload --port 8001 --host 0.0.0.0
