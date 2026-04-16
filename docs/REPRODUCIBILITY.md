# Reproducing RingRift Training Results

This document records the exact commands, hyperparameters, and hardware used to produce the reported results.

The checked-in claim map is
[`docs/data/results_evidence_manifest.json`](/docs/data/results_evidence_manifest.json).
Use it to distinguish result numbers that are verifiable from repository files
from claims that require the S3 archive or live operator logs.

## Headline Result

**hex8_2p**: 1500 → 1979.8 Elo in 33 iterations using fixed learning rate self-play.

## Hardware

- **Training GPU**: NVIDIA GH200 96GB (Lambda Cloud, `gpu_1x_gh200`)
- **CPU**: 64 vCPUs, 432 GB RAM per node
- **Training time**: ~6 hours per iteration (4h selfplay + 1h training + 1h evaluation)
- **Total wall time for hex8_2p**: ~200 hours (33 iterations)

## Software

- Python 3.10 (Lambda Ubuntu 22.04)
- PyTorch 2.x with CUDA
- All other dependencies: `pip install -r requirements.txt`

## Exact Training Command

### hex8_2p (flagship result: 1500 → 1979.8 Elo, 7 promotions)

```bash
cd ai-service
export PYTHONPATH=.

python scripts/minimal_alphazero_loop.py \
  --model models/canonical_hex8_2p.pth \
  --board-type hex8 \
  --num-players 2 \
  --iterations 50 \
  --games-per-iter 100 \
  --selfplay-budget 200 \
  --eval-budget 128 \
  --lr 5e-5 \
  --lr-schedule fixed \
  --train-lr-scheduler none \
  --train-window 5 \
  --work-dir data/minimal_loop_hex8_2p
```

### square8_2p (second result: 1500 → 1697.3 Elo, 4 promotions)

```bash
python scripts/minimal_alphazero_loop.py \
  --model models/canonical_square8_2p.pth \
  --board-type square8 \
  --num-players 2 \
  --iterations 50 \
  --games-per-iter 100 \
  --selfplay-budget 128 \
  --eval-budget 128 \
  --lr 5e-5 \
  --lr-schedule fixed \
  --train-lr-scheduler none \
  --train-window 3 \
  --work-dir data/minimal_loop_square8_2p
```

## Critical Hyperparameters

| Parameter                   | Value                     | Why                                                             |
| --------------------------- | ------------------------- | --------------------------------------------------------------- |
| `--lr 5e-5`                 | Fixed at 5e-5             | sqrt_decay + cosine caused double LR decay that killed learning |
| `--lr-schedule fixed`       | No iteration-level decay  | Stable learning across all iterations                           |
| `--train-lr-scheduler none` | No epoch-level decay      | Prevents triple-decay (loop × epoch × optimizer)                |
| `--train-window 5`          | Last 5 iterations of data | Balances freshness vs volume                                    |
| `--selfplay-budget 200`     | 200 MCTS simulations      | Higher budget = better data quality                             |
| `--eval-budget 128`         | 128 MCTS simulations      | Sufficient for reliable evaluation                              |

## Evaluation Protocol

Staged evaluation with early exit:

| Stage | Games | Promote If | Reject If |
| ----- | ----- | ---------- | --------- |
| 1     | 50    | > 60%      | < 42%     |
| 2     | 100   | > 56%      | < 46%     |
| 3     | 200   | > 53%      | < 48%     |
| 4     | 400   | > 50.1%    | reject    |

For multiplayer (3p/4p), thresholds are lowered because evaluation pits 1 candidate vs (N-1) copies of the best model.

## Starting Model

The training starts from a randomly initialized model (`canonical_hex8_2p.pth`). The model architecture is:

- **HexNeuralNet_v2**: 12 SE residual blocks, 192 filters
- **Input**: 40 channels (10 base features × 4 history frames)
- **Policy head**: position-aware encoding for hex8 board (61 cells)
- **Value head**: 2-output softmax (player 1 win prob, player 2 win prob)
- **Parameters**: ~10M

## Archived Artifacts

All training artifacts are archived to S3:

```
s3://ringrift-models-20251214/archive/
├── gh200-8/          # hex8_2p (1979.8 Elo)
│   ├── models/best.pth
│   ├── metrics.jsonl
│   └── training_data/iter_*.npz
├── gh200-9/          # square8_2p (1697.3 Elo)
│   ├── models/best.pth
│   ├── metrics.jsonl
│   └── training_data/iter_*.npz
├── gh200-10/         # hex8_3p
├── gh200-12/         # square8_3p (1534.9 Elo)
└── gh200-14/         # square19_2p
```

To download and resume training:

```bash
aws s3 cp s3://ringrift-models-20251214/archive/gh200-8/models/best.pth \
  models/canonical_hex8_2p.pth

python scripts/minimal_alphazero_loop.py \
  --model models/canonical_hex8_2p.pth \
  --board-type hex8 --num-players 2 \
  --lr 5e-5 --lr-schedule fixed --train-lr-scheduler none \
  --work-dir data/resume_hex8_2p
```

## What Was Learned

1. **Fixed LR is critical**: sqrt_decay + cosine scheduler caused the hex8_2p plateau at 1967.6 for 8 iterations. Switching to fixed 5e-5 immediately produced a promotion.
2. **Multiplayer evaluation is unfair by design**: 1 candidate vs (N-1) best copies means the promotion threshold must be lowered for 3p/4p games.
3. **Data volume matters**: square8_3p with 50 games/iter (2,500 samples) couldn't learn. Increasing to 200 games/iter fixed the data starvation.
4. **GPU process contention is silent**: Two AI instances on one GPU can cause 10x slowdown without any error messages.
