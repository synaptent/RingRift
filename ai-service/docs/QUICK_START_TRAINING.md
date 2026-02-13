# Quick Start: Train a RingRift Model from Scratch

Train a neural network AI opponent locally in 5 steps. No cluster required.

## Prerequisites

```bash
cd ai-service
pip install -r requirements.txt
```

You need Python 3.10+ and PyTorch. A GPU is optional but speeds up training significantly.

## Step 1: Generate Selfplay Games

Generate games using the heuristic engine (fast, no neural network needed):

```bash
python scripts/selfplay.py \
  --board hex8 \
  --num-players 2 \
  --engine-mode heuristic-only \
  --num-games 100
```

This creates a SQLite database in `data/games/` with game records. For higher quality data, increase `--num-games` to 500-1000.

**Engine modes available:**

- `heuristic-only` - Fast, good for bootstrapping (recommended to start)
- `gumbel-mcts` - High quality training data (requires a trained model, slower)
- `mixed` - Mix of engine types for diversity

## Step 2: Export Training Data

Convert the game database to NPZ arrays for training:

```bash
python scripts/export_replay_dataset.py \
  --use-discovery \
  --board-type hex8 \
  --num-players 2 \
  --output data/training/quickstart_hex8_2p.npz
```

The `--use-discovery` flag automatically finds all databases matching the board type and player count. Alternatively, specify a database directly with `--db data/games/your_database.db`.

## Step 3: Train the Model

Train a neural network on the exported data:

```bash
python -m app.training.train \
  --board-type hex8 \
  --num-players 2 \
  --data-path data/training/quickstart_hex8_2p.npz \
  --epochs 10
```

The trainer auto-detects the model architecture and batch size. The best model checkpoint is saved to `models/`.

**Useful options:**

- `--epochs 20` - More training epochs (default: auto)
- `--batch-size 256` - Manual batch size (default: auto-tuned to GPU memory)
- `--model-version v2` - Specify architecture (default: auto-detected)
- `--save-path models/my_model.pth` - Custom save location

## Step 4: Evaluate the Model

Run a gauntlet evaluation against baseline opponents:

```bash
python scripts/quick_gauntlet.py \
  --model models/canonical_hex8_2p.pth \
  --board-type hex8 \
  --num-players 2 \
  --games 10
```

This tests the model against random and heuristic baselines. Passing thresholds:

- vs Random: 85% win rate
- vs Heuristic: 60% win rate

## Step 5: Iterate

To improve the model, generate more games using the trained model for selfplay:

```bash
# Use the trained model for higher-quality selfplay
python scripts/selfplay.py \
  --board hex8 \
  --num-players 2 \
  --engine-mode gumbel-mcts \
  --num-games 500

# Re-export with all games combined
python scripts/export_replay_dataset.py \
  --use-discovery \
  --board-type hex8 \
  --num-players 2 \
  --output data/training/hex8_2p_iter2.npz

# Train again
python -m app.training.train \
  --board-type hex8 \
  --num-players 2 \
  --data-path data/training/hex8_2p_iter2.npz
```

Each iteration should produce a stronger model. Use `scripts/elo_progress_report.py` to track Elo improvement over time.

## Board Types

You can train on any of the 4 board types with 2-4 players:

| Board     | Flag                                           | Cells | Notes                                      |
| --------- | ---------------------------------------------- | ----- | ------------------------------------------ |
| hex8      | `--board hex8` / `--board-type hex8`           | 61    | Fastest to train, good for experimentation |
| square8   | `--board square8` / `--board-type square8`     | 64    | Standard board                             |
| square19  | `--board square19` / `--board-type square19`   | 361   | Large, slower to train                     |
| hexagonal | `--board hexagonal` / `--board-type hexagonal` | 469   | Largest, needs many games                  |

## Cluster Training

For distributed training across multiple GPUs, see the [Distributed Training Architecture](architecture/DISTRIBUTED_TRAINING.md) documentation and the `master_loop.py` automation entry point.

```bash
# Full cluster automation
python scripts/master_loop.py
```

## Troubleshooting

**"No databases found"** during export: Make sure selfplay completed successfully and databases exist in `data/games/`.

**Out of memory during training**: Reduce batch size with `--batch-size 128` or use `--safe-mode`.

**Parity gate warnings**: Set `export RINGRIFT_ALLOW_PENDING_GATE=1` if you see "pending_gate" warnings (this is normal on machines without Node.js).

**PYTHONPATH errors**: Run from the `ai-service/` directory or set `PYTHONPATH=.`.
