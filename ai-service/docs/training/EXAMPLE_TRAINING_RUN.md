# Example Training Run

This document shows a complete example of training an NNUE policy model from scratch.

## Prerequisites

```bash
cd ai-service
source venv/bin/activate
pip install -r requirements.txt
```

## Step 1: Generate Self-Play Data

First, generate training games using self-play:

```bash
# Generate 1000 games for square8 2-player
python scripts/run_gpu_selfplay.py \
  --board-type square8 \
  --num-players 2 \
  --num-games 1000 \
  --engine-mode mixed \
  --output-dir data/selfplay/example_run
```

**Expected output:**

```
[2025-01-15 10:00:00] Starting selfplay: square8_2p, 1000 games
[2025-01-15 10:00:01] Using engine mode: mixed (heuristic + NNUE)
[2025-01-15 10:00:05] Game 1/1000 complete (87 moves, P1 wins)
[2025-01-15 10:00:09] Game 2/1000 complete (124 moves, P2 wins)
...
[2025-01-15 10:45:00] Completed 1000 games
[2025-01-15 10:45:00] Results: P1=487 wins, P2=513 wins, draws=0
[2025-01-15 10:45:00] Saved to: data/selfplay/example_run/games.db
```

## Step 2: Train the Model

Train an NNUE policy model on the generated data:

```bash
python scripts/train_nnue_policy.py \
  --board square8 \
  --num-players 2 \
  --db data/selfplay/example_run/games.db \
  --epochs 50 \
  --batch-size 1024 \
  --lr 0.001 \
  --output-dir models/nnue/example
```

**Expected output:**

```
[2025-01-15 11:00:00] Loading training data from data/selfplay/example_run/games.db
[2025-01-15 11:00:02] Loaded 1000 games, 87,432 positions
[2025-01-15 11:00:02] Model: NNUEPolicyNet(square8_2p), params=861,088
[2025-01-15 11:00:02] Training config: epochs=50, batch=1024, lr=0.001
[2025-01-15 11:00:02] Device: cuda:0 (NVIDIA H100)

Epoch 1/50:
  Train Loss: 4.2341 | Policy Acc: 12.3% | Value MSE: 0.892
  Val Loss: 4.1876 | Policy Acc: 13.1% | Value MSE: 0.871

Epoch 10/50:
  Train Loss: 2.8723 | Policy Acc: 34.7% | Value MSE: 0.432
  Val Loss: 2.9102 | Policy Acc: 33.2% | Value MSE: 0.456

Epoch 25/50:
  Train Loss: 1.9234 | Policy Acc: 52.8% | Value MSE: 0.234
  Val Loss: 2.0123 | Policy Acc: 50.1% | Value MSE: 0.267

Epoch 50/50:
  Train Loss: 1.4521 | Policy Acc: 65.3% | Value MSE: 0.156
  Val Loss: 1.5872 | Policy Acc: 62.7% | Value MSE: 0.189

[2025-01-15 11:15:00] Training complete!
[2025-01-15 11:15:00] Best model saved: models/nnue/example/best_model.pt
[2025-01-15 11:15:00] Final checkpoint: models/nnue/example/checkpoint_epoch50.pt
```

## Step 3: Evaluate the Model

Run an Elo tournament to evaluate model strength:

```bash
python scripts/run_model_elo_tournament.py \
  --board square8 \
  --num-players 2 \
  --model models/nnue/example/best_model.pt \
  --games-per-match 50
```

**Expected output:**

```
[2025-01-15 11:30:00] Starting Elo tournament: square8_2p
[2025-01-15 11:30:00] Contestants: trained_model, heuristic_baseline, random_baseline

Match 1: trained_model vs heuristic_baseline
  Games: 50 | Wins: 32 | Losses: 15 | Draws: 3

Match 2: trained_model vs random_baseline
  Games: 50 | Wins: 49 | Losses: 1 | Draws: 0

Match 3: heuristic_baseline vs random_baseline
  Games: 50 | Wins: 47 | Losses: 2 | Draws: 1

Final Elo Ratings:
  1. trained_model      : 1523 (+123 from baseline)
  2. heuristic_baseline : 1412
  3. random_baseline    : 1065
```

## Step 4: Deploy the Model

Copy the trained model to the production location:

```bash
cp models/nnue/example/best_model.pt models/nnue/nnue_policy_square8_2p.pt
```

The model will now be used automatically by the AI service.

## Quick Reference

### Training Commands

| Task              | Command                                                                               |
| ----------------- | ------------------------------------------------------------------------------------- |
| Generate data     | `python scripts/run_gpu_selfplay.py --board-type square8 --num-games 1000`            |
| Train model       | `python scripts/train_nnue_policy.py --board square8 --epochs 50`                     |
| Evaluate          | `python scripts/run_model_elo_tournament.py --board square8`                          |
| Transfer learning | `python scripts/train_nnue_policy.py --pretrained models/nnue/base.pt --freeze-value` |

### Expected Training Times

| Board Type | Games | Training Time (H100) | Training Time (RTX 4090) |
| ---------- | ----- | -------------------- | ------------------------ |
| square8    | 1K    | ~15 min              | ~25 min                  |
| square8    | 10K   | ~45 min              | ~90 min                  |
| square19   | 1K    | ~30 min              | ~60 min                  |
| hexagonal  | 1K    | ~35 min              | ~70 min                  |

### Troubleshooting

**Out of memory:**

```bash
# Reduce batch size
python scripts/train_nnue_policy.py --batch-size 512

# Enable gradient checkpointing
python scripts/train_nnue_policy.py --gradient-checkpointing
```

**Low accuracy:**

- Generate more training data (aim for 10K+ games)
- Train for more epochs (100-200)
- Use curriculum learning: `python scripts/curriculum_training.py --auto-progress`

**Model not improving:**

- Check learning rate (try 0.0001 or 0.01)
- Verify data quality: `python scripts/analyze_training_data.py --db data/games.db`
- Try transfer learning from a stronger model

## Next Steps

1. **Scale up**: Generate 10K-100K games for better models
2. **Multi-config**: Train models for different board types
3. **Distributed**: Use multiple GPUs with `--distributed` flag
4. **Continuous**: Set up unified AI loop for automated improvement

See [TRAINING_PIPELINE.md](TRAINING_PIPELINE.md) for advanced training options.
