#!/bin/bash
# Sync RingRift AI code and data to Lambda Labs GPU instance
#
# Usage:
#   ./scripts/sync_to_lambda.sh           # Sync code only
#   ./scripts/sync_to_lambda.sh --data    # Sync code + training data
#   ./scripts/sync_to_lambda.sh --models  # Sync code + models
#   ./scripts/sync_to_lambda.sh --all     # Sync everything
#
# Prerequisites:
#   - SSH config entry for 'lambda-gpu' (see ~/.ssh/config)
#   - ssh-add ~/.ssh/id_ed25519 (load key with passphrase)

set -e

LAMBDA_HOST="${LAMBDA_HOST:-lambda-gpu}"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_DIR="~/ringrift/ai-service"

# Parse arguments
SYNC_DATA=false
SYNC_MODELS=false

for arg in "$@"; do
    case $arg in
        --data)
            SYNC_DATA=true
            ;;
        --models)
            SYNC_MODELS=true
            ;;
        --all)
            SYNC_DATA=true
            SYNC_MODELS=true
            ;;
        --help|-h)
            echo "Usage: $0 [--data] [--models] [--all]"
            echo ""
            echo "Options:"
            echo "  --data    Sync training data (*.db files)"
            echo "  --models  Sync model checkpoints"
            echo "  --all     Sync everything (code + data + models)"
            echo ""
            echo "Environment variables:"
            echo "  LAMBDA_HOST  Override SSH host (default: lambda-gpu)"
            exit 0
            ;;
    esac
done

echo "=============================================="
echo "Syncing RingRift to Lambda Labs"
echo "=============================================="
echo "Local:  $LOCAL_DIR"
echo "Remote: $LAMBDA_HOST:$REMOTE_DIR"
echo ""

# Check SSH connection
echo "=== Testing SSH connection ==="
if ! ssh -o BatchMode=yes -o ConnectTimeout=5 "$LAMBDA_HOST" echo "Connected" 2>/dev/null; then
    echo "ERROR: Cannot connect to $LAMBDA_HOST"
    echo ""
    echo "Make sure you've loaded your SSH key:"
    echo "  ssh-add ~/.ssh/id_ed25519"
    echo ""
    echo "Then try again."
    exit 1
fi

# Create remote directories
echo "=== Creating remote directories ==="
ssh "$LAMBDA_HOST" "mkdir -p $REMOTE_DIR/{app,scripts,data/games,data/training,models/nnue,logs,config}"

# Sync code (excluding large files and generated content)
echo "=== Syncing code ==="
rsync -avz --progress \
    --exclude='venv/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.git/' \
    --exclude='*.db' \
    --exclude='*.npz' \
    --exclude='*.pt' \
    --exclude='*.pth' \
    --exclude='*.nnue' \
    --exclude='logs/' \
    --exclude='data/games/' \
    --exclude='data/training/' \
    --exclude='models/' \
    --exclude='.env' \
    "$LOCAL_DIR/" "$LAMBDA_HOST:$REMOTE_DIR/"

# Sync training data if requested
if [ "$SYNC_DATA" = true ]; then
    echo ""
    echo "=== Syncing training data ==="

    # Sync game databases
    if ls "$LOCAL_DIR/data/games/"*.db 1>/dev/null 2>&1; then
        rsync -avz --progress \
            "$LOCAL_DIR/data/games/"*.db \
            "$LAMBDA_HOST:$REMOTE_DIR/data/games/"
    else
        echo "No .db files found in data/games/"
    fi

    # Sync NPZ training files
    if ls "$LOCAL_DIR/data/training/"*.npz 1>/dev/null 2>&1; then
        rsync -avz --progress \
            "$LOCAL_DIR/data/training/"*.npz \
            "$LAMBDA_HOST:$REMOTE_DIR/data/training/"
    else
        echo "No .npz files found in data/training/"
    fi
fi

# Sync models if requested
if [ "$SYNC_MODELS" = true ]; then
    echo ""
    echo "=== Syncing models ==="

    # Sync PyTorch models
    if ls "$LOCAL_DIR/models/"*.pt "$LOCAL_DIR/models/"*.pth 1>/dev/null 2>&1; then
        rsync -avz --progress \
            "$LOCAL_DIR/models/"*.pt \
            "$LOCAL_DIR/models/"*.pth \
            "$LAMBDA_HOST:$REMOTE_DIR/models/" 2>/dev/null || true
    fi

    # Sync NNUE models
    if [ -d "$LOCAL_DIR/models/nnue" ] && ls "$LOCAL_DIR/models/nnue/"* 1>/dev/null 2>&1; then
        rsync -avz --progress \
            "$LOCAL_DIR/models/nnue/" \
            "$LAMBDA_HOST:$REMOTE_DIR/models/nnue/"
    fi
fi

# Sync config files
echo ""
echo "=== Syncing config ==="
if [ -d "$LOCAL_DIR/config" ]; then
    rsync -avz --progress \
        "$LOCAL_DIR/config/" \
        "$LAMBDA_HOST:$REMOTE_DIR/config/"
fi

# Sync heuristic profiles
if [ -f "$LOCAL_DIR/data/trained_heuristic_profiles.json" ]; then
    rsync -avz --progress \
        "$LOCAL_DIR/data/trained_heuristic_profiles.json" \
        "$LAMBDA_HOST:$REMOTE_DIR/data/"
fi

echo ""
echo "=============================================="
echo "Sync Complete!"
echo "=============================================="
echo ""
echo "To start training on Lambda:"
echo "  ssh $LAMBDA_HOST"
echo "  cd ~/ringrift/ai-service"
echo "  source venv/bin/activate"
echo "  python app/training/train.py --device cuda --batch-size 512"
echo ""
