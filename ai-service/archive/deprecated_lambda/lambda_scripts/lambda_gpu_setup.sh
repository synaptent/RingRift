#!/bin/bash
# Lambda Labs GPU Instance Setup Script
#
# This script sets up a fresh Lambda Labs GPU instance for RingRift AI training.
# Run this after SSH-ing into a new Lambda instance.
#
# Usage:
#   ssh lambda-gpu
#   curl -sL https://raw.githubusercontent.com/.../lambda_gpu_setup.sh | bash
#   # Or copy this script and run locally:
#   bash lambda_gpu_setup.sh
#
# Prerequisites:
#   - Lambda Labs instance with A10/A100 GPU
#   - Ubuntu 22.04 (Lambda's default)
#   - SSH access configured

set -e

echo "=============================================="
echo "RingRift Lambda Labs GPU Setup"
echo "=============================================="
echo ""

# Check if we're on a Lambda Labs instance
if ! command -v nvidia-smi &> /dev/null; then
    echo "WARNING: nvidia-smi not found. This might not be a GPU instance."
fi

# Show GPU info
echo "=== GPU Information ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "No GPU detected"
echo ""

# Update system
echo "=== Updating system packages ==="
sudo apt-get update -qq
sudo apt-get install -y -qq git curl wget htop tmux

# Check Python version (Lambda usually has Python 3.10+)
echo "=== Python Version ==="
python3 --version

# Create project directory
PROJECT_DIR="$HOME/ringrift"
echo "=== Setting up project directory: $PROJECT_DIR ==="
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# Clone repository if not exists
if [ ! -d "ai-service" ]; then
    echo "=== Cloning RingRift repository ==="
    # Option 1: Clone from GitHub (if public/have access)
    # git clone https://github.com/your-org/ringrift.git .

    # Option 2: Create placeholder structure for rsync
    mkdir -p ai-service/app ai-service/scripts ai-service/data/games ai-service/models
    echo "Created placeholder directories. Use rsync to sync code from local machine."
else
    echo "Repository already exists"
fi

cd ai-service

# Create virtual environment
echo "=== Setting up Python virtual environment ==="
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

# Install PyTorch with CUDA support
echo "=== Installing PyTorch with CUDA ==="
pip install --upgrade pip wheel setuptools

# Lambda Labs instances typically have CUDA 12.x
# Check CUDA version and install appropriate PyTorch
CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "")
echo "CUDA Driver: $CUDA_VERSION"

# Install PyTorch for CUDA 12.1 (common on Lambda)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA is available
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Install other dependencies
echo "=== Installing Python dependencies ==="
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    # Install common dependencies
    pip install numpy scipy cma tqdm pydantic
fi

# Create data directories
echo "=== Creating data directories ==="
mkdir -p data/games data/training models/nnue logs

# Set up environment variables
echo "=== Setting up environment ==="
cat > .env << 'EOF'
# RingRift AI Training Environment
RINGRIFT_SKIP_SHADOW_CONTRACTS=true
PYTHONPATH=/home/ubuntu/ringrift/ai-service
OMP_NUM_THREADS=4
MKL_NUM_THREADS=4
EOF

# Add to bashrc for persistence
if ! grep -q "ringrift" ~/.bashrc; then
    cat >> ~/.bashrc << 'EOF'

# RingRift AI Training
export RINGRIFT_SKIP_SHADOW_CONTRACTS=true
alias ringrift='cd ~/ringrift/ai-service && source venv/bin/activate'
alias gpu-watch='watch -n 1 nvidia-smi'
EOF
fi

echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Sync code from local machine:"
echo "     rsync -avz --exclude='venv' --exclude='*.db' --exclude='__pycache__' \\"
echo "       ~/Development/RingRift/ai-service/ lambda-gpu:~/ringrift/ai-service/"
echo ""
echo "  2. Sync training data:"
echo "     rsync -avz ~/Development/RingRift/ai-service/data/games/*.db \\"
echo "       lambda-gpu:~/ringrift/ai-service/data/games/"
echo ""
echo "  3. Activate environment:"
echo "     source venv/bin/activate"
echo ""
echo "  4. Run training:"
echo "     python app/training/train.py --device cuda --batch-size 512"
echo ""
echo "Useful commands:"
echo "  gpu-watch     - Monitor GPU usage"
echo "  ringrift      - CD to project and activate venv"
echo ""
