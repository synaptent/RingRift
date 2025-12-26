#!/bin/bash
# Setup script for RingRift AI Service
# Creates virtual environment and installs dependencies

set -e  # Exit on error

echo "🔧 Setting up RingRift AI Service..."

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.10 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "✅ Found Python $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📦 Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "To start the AI service, run:"
echo "  ./run.sh"
echo ""
echo "Or manually:"
echo "  source venv/bin/activate"
echo "  python -m uvicorn app.main:app --reload --port 8001"
echo ""
