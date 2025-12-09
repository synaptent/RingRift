# Cloud Deployment for Neural Network Training Data Generation

This directory contains configuration and scripts for deploying distributed self-play workers on cloud platforms.

## Quick Start

### Local Testing

```bash
# Generate 100 games locally
python scripts/run_distributed_selfplay.py \
    --num-games 100 \
    --board-type square8 \
    --output file:///tmp/training_data.jsonl \
    --seed 42
```

### AWS EC2 Spot Instances

```bash
# Deploy 10 spot instances to generate 100K games
./cloud/aws/deploy-spot-fleet.sh \
    --workers 10 \
    --games-per-worker 10000 \
    --board-type square8 \
    --bucket ringrift-training-data
```

### Google Cloud Platform

```bash
# Deploy on GCE preemptible VMs
./cloud/gcp/deploy-preemptible.sh \
    --workers 10 \
    --games-per-worker 10000 \
    --board-type square8 \
    --bucket ringrift-training-data
```

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                      Job Coordinator                              │
│                     (Optional: Redis/SQS)                         │
└────────────────────────────┬─────────────────────────────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   Worker 1      │ │   Worker 2      │ │   Worker N      │
│  (Spot/Preempt) │ │  (Spot/Preempt) │ │  (Spot/Preempt) │
│                 │ │                 │ │                 │
│  - Self-play    │ │  - Self-play    │ │  - Self-play    │
│  - Extract      │ │  - Extract      │ │  - Extract      │
│    samples      │ │    samples      │ │    samples      │
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             ▼
                  ┌─────────────────────┐
                  │   Cloud Storage     │
                  │  (S3/GCS/Azure)     │
                  │                     │
                  │  ├── worker_01/     │
                  │  │   └── part_*.gz  │
                  │  ├── worker_02/     │
                  │  │   └── part_*.gz  │
                  │  └── ...            │
                  └─────────────────────┘
```

## Cost Estimates (AWS)

| Target     | Instance Type | Workers | Hours | Spot Cost |
| ---------- | ------------- | ------- | ----- | --------- |
| 10K games  | c6i.xlarge    | 2       | 1     | ~$0.10    |
| 100K games | c6i.xlarge    | 10      | 2.5   | ~$0.50    |
| 1M games   | c6i.xlarge    | 20      | 12.5  | ~$5.00    |

_Costs based on ~40 games/hour per vCPU for Square8_

## Training Sample Format

Each sample is a JSONL line:

```json
{
  "state": { ... full GameState ... },
  "outcome": 1.0,
  "board_type": "square8",
  "game_id": "worker_01_123_abc123",
  "move_number": 15,
  "ply_to_end": 20,
  "move": { ... move that was played ... },
  "metadata": {
    "worker_id": "worker_01",
    "engine_mode": "mixed",
    "difficulty_band": "light"
  }
}
```

## Aggregating Training Data

After workers complete, aggregate the data:

```bash
# Download from S3
aws s3 sync s3://bucket/prefix/selfplay ./training_data/

# Combine and shuffle
python scripts/aggregate_training_data.py \
    --input-dir ./training_data \
    --output ./combined_training.jsonl.gz \
    --shuffle
```

## Preemption Handling

Workers automatically checkpoint progress and resume:

```bash
# Worker with checkpointing enabled
python scripts/run_distributed_selfplay.py \
    --num-games 10000 \
    --checkpoint-interval 500 \
    --checkpoint-path /tmp/checkpoint.json \
    --output s3://bucket/prefix
```

When a spot instance is terminated:

1. SIGTERM triggers graceful shutdown
2. Buffered data is flushed to storage
3. Checkpoint is saved locally (or to persistent volume)
4. Next instance can resume from checkpoint

## Deployment Modes

All distributed training scripts support a `--mode` argument for selecting worker hosts:

| Mode     | Description                                     | Use Case                   |
| -------- | ----------------------------------------------- | -------------------------- |
| `local`  | Run on local machine only (no remote workers)   | Development, quick tests   |
| `lan`    | Use local Mac cluster workers from hosts config | Zero-cost distributed runs |
| `aws`    | Use AWS cloud workers                           | Cloud burst capacity       |
| `hybrid` | Use both LAN and AWS workers                    | Maximum parallelism        |

```bash
# Run distributed self-play on LAN cluster
python scripts/run_distributed_selfplay_soak.py --mode lan --num-games 100

# Run CMA-ES optimization on AWS
python scripts/run_cmaes_optimization.py --distributed --mode aws

# Run iterative CMA-ES with hybrid workers
python scripts/run_iterative_cmaes.py --distributed --mode hybrid
```

## Environment Variables

| Variable                         | Description              | Default             |
| -------------------------------- | ------------------------ | ------------------- |
| `WORKER_ID`                      | Unique worker identifier | Auto-generated UUID |
| `AWS_DEFAULT_REGION`             | AWS region for S3        | us-east-1           |
| `GOOGLE_APPLICATION_CREDENTIALS` | GCP service account      | -                   |
