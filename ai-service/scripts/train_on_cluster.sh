#!/bin/bash
# Train models on Lambda cluster nodes

BOARD_TYPE=${1:-square8}
NUM_PLAYERS=${2:-2}
CLUSTER_NODE=${3:-lambda-gh200-h}

echo "Training ${BOARD_TYPE}_${NUM_PLAYERS}p on $CLUSTER_NODE"

# Copy training data to cluster
ssh $CLUSTER_NODE "mkdir -p ~/ringrift/ai-service/data/training"
rsync -avz data/training/canonical_*.npz $CLUSTER_NODE:~/ringrift/ai-service/data/training/

# Run training
ssh $CLUSTER_NODE "cd ~/ringrift/ai-service && source venv/bin/activate && \
  PYTHONPATH=. nohup python -m app.training.train \
    --data-path data/training/canonical_${BOARD_TYPE}_${NUM_PLAYERS}p_replay.npz \
    --board-type $BOARD_TYPE \
    --num-players $NUM_PLAYERS \
    --epochs 100 \
    --batch-size 64 \
    --save-path models/canonical_${BOARD_TYPE}_${NUM_PLAYERS}p.pth \
    --model-version v2 \
    --disable-circuit-breaker \
    > logs/train_canonical_${BOARD_TYPE}_${NUM_PLAYERS}p.log 2>&1 &"

echo "Training started on $CLUSTER_NODE"
