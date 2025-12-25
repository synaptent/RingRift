# Database Module

This module provides SQLite-based storage for RingRift games, supporting training data generation, replay analysis, and parity validation.

## Table of Contents

1. [Overview](#overview)
2. [Core Components](#core-components)
   - [GameReplayDB](#gamereplaydb)
   - [UnifiedGameRecorder](#unifiedgamerecorder)
   - [Parity Validator](#parity-validator)
   - [Database Integrity](#database-integrity)
3. [Schema](#schema)
4. [Usage Examples](#usage-examples)
5. [Parity Validation](#parity-validation)
6. [NNUE Feature Caching](#nnue-feature-caching)
7. [Configuration](#configuration)
8. [Troubleshooting](#troubleshooting)

---

## Overview

The database module stores complete game replays for:

- **Training Data**: Games with moves, policies, and values for neural network training
- **Replay Analysis**: State reconstruction at any move for debugging
- **Parity Validation**: Cross-validation between TypeScript and Python engines
- **Quality Scoring**: Prioritization of high-quality training examples

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Database Architecture                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Selfplay Engine ──► UnifiedGameRecorder ──► GameReplayDB          │
│         │                     │                    │                │
│         │              (Parity Check)         (SQLite)              │
│         │                     │                    │                │
│         ▼                     ▼                    ▼                │
│   Game completed      TS Replay Harness      data/games/*.db        │
│                                                                     │
│   Training Pipeline ◄── export_replay_dataset.py ◄── Query games   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### GameReplayDB

The primary interface for storing and querying games:

```python
from app.db import GameReplayDB

# Open or create database
db = GameReplayDB("data/games/selfplay_hex8_2p.db")

# Query games
games = db.query_games(
    board_type="hex8",
    num_players=2,
    winner=0,  # Player 0 won
    limit=100,
)

# Get full game data
game_id = games[0]["game_id"]
initial_state = db.get_initial_state(game_id)
moves = db.get_moves(game_id)

# Reconstruct state at move 15
state_at_15 = db.get_state_at_move(game_id, move_number=15)

# Get game metadata
metadata = db.get_game_metadata(game_id)
print(f"Total moves: {metadata['total_moves']}")
print(f"Winner: {metadata['winner']}")
print(f"Quality: {metadata.get('quality_score', 'N/A')}")
```

#### Key Methods

| Method                 | Description                 |
| ---------------------- | --------------------------- |
| `query_games()`        | Filter games by criteria    |
| `get_initial_state()`  | Get starting state          |
| `get_moves()`          | Get all moves for a game    |
| `get_state_at_move()`  | Reconstruct state at move N |
| `get_game_metadata()`  | Get game summary data       |
| `count_games()`        | Count matching games        |
| `get_training_batch()` | Get batch for training      |

### UnifiedGameRecorder

High-level API for recording games with automatic features:

```python
from app.db import (
    UnifiedGameRecorder,
    RecordingConfig,
    RecordSource,
)

# Create recorder with config
config = RecordingConfig(
    board_type="hex8",
    num_players=2,
    source=RecordSource.GUMBEL_MCTS,
    enable_parity_check=True,
    enable_nnue_cache=True,
    snapshot_interval=20,
)

recorder = UnifiedGameRecorder(config)

# Record a game
game_id = recorder.record_game(
    initial_state=state,
    moves=moves,
    player_moves=player_moves,  # Per-turn info
    winner=winner,
    metadata={"model_version": "v2"},
)

print(f"Recorded game {game_id}")
```

#### Record Sources

| Source        | Description          | Quality Weight |
| ------------- | -------------------- | -------------- |
| `GUMBEL_MCTS` | Gumbel MCTS selfplay | High           |
| `NNUE_GUIDED` | NNUE-guided search   | Medium-High    |
| `HEURISTIC`   | Heuristic-only       | Low            |
| `HUMAN`       | Human gameplay       | Variable       |
| `REPLAY`      | Imported replay      | Variable       |

### Parity Validator

Cross-validates games against TypeScript engine:

```python
from app.db import (
    validate_game_parity,
    ParityMode,
    ParityDivergence,
    get_parity_mode,
)

# Check parity mode
mode = get_parity_mode()  # OFF, WARN, or STRICT

# Validate a game
try:
    result = validate_game_parity(
        game_id="abc123",
        db_path="data/games/selfplay.db",
    )
    if result.valid:
        print("Parity validated!")
    else:
        print(f"Divergence at move {result.divergence_move}")
except ParityValidationError as e:
    print(f"Validation failed: {e}")
```

#### Parity Modes

| Mode     | Behavior            | Use Case            |
| -------- | ------------------- | ------------------- |
| `OFF`    | Skip validation     | Production selfplay |
| `WARN`   | Log divergences     | Development         |
| `STRICT` | Raise on divergence | Testing/CI          |

Set via environment:

```bash
export RINGRIFT_PARITY_VALIDATION=strict
```

### Database Integrity

Tools for checking and repairing databases:

```python
from app.db import (
    check_database_integrity,
    check_and_repair_databases,
    recover_corrupted_database,
    get_database_stats,
)

# Check single database
is_valid = check_database_integrity("data/games/selfplay.db")

# Get stats
stats = get_database_stats("data/games/selfplay.db")
print(f"Games: {stats['game_count']}")
print(f"Size: {stats['size_mb']:.1f}MB")
print(f"Schema version: {stats['schema_version']}")

# Check and repair all databases in directory
results = check_and_repair_databases("data/games/")
for path, status in results.items():
    print(f"{path}: {status}")

# Recover corrupted database
if not is_valid:
    recovered = recover_corrupted_database(
        "data/games/corrupted.db",
        "data/games/recovered.db",
    )
```

---

## Schema

Current schema version: **11**

### Tables

#### `games`

Main table storing game metadata:

| Column             | Type    | Description          |
| ------------------ | ------- | -------------------- |
| `game_id`          | TEXT PK | UUID identifier      |
| `board_type`       | TEXT    | hex8, square8, etc.  |
| `num_players`      | INTEGER | 2-4 players          |
| `winner`           | INTEGER | Winning player index |
| `total_moves`      | INTEGER | Number of moves      |
| `quality_score`    | REAL    | Training quality 0-1 |
| `quality_category` | TEXT    | high/medium/low      |
| `metadata_json`    | TEXT    | Extended metadata    |

#### `moves`

Individual moves with policy data:

| Column              | Type    | Description         |
| ------------------- | ------- | ------------------- |
| `game_id`           | TEXT FK | Game reference      |
| `move_number`       | INTEGER | 0-indexed move      |
| `player_id`         | INTEGER | Player making move  |
| `move_json`         | TEXT    | Move data as JSON   |
| `move_probs`        | BLOB    | Policy distribution |
| `search_stats_json` | TEXT    | MCTS statistics     |

#### `snapshots`

State snapshots for efficient replay:

| Column        | Type    | Description               |
| ------------- | ------- | ------------------------- |
| `game_id`     | TEXT FK | Game reference            |
| `move_number` | INTEGER | Move number               |
| `state_json`  | TEXT    | Compressed state          |
| `state_hash`  | TEXT    | State hash for validation |

#### `game_nnue_features`

Pre-computed NNUE features for training:

| Column          | Type    | Description         |
| --------------- | ------- | ------------------- |
| `game_id`       | TEXT FK | Game reference      |
| `move_number`   | INTEGER | Move number         |
| `features`      | BLOB    | Compressed features |
| `policy_target` | BLOB    | Target policy       |
| `value_target`  | REAL    | Target value        |

### Schema Migrations

The database auto-migrates when opening older versions:

```python
db = GameReplayDB("old_database.db")
# Automatically migrates to SCHEMA_VERSION=11
```

---

## Usage Examples

### Recording a Selfplay Game

```python
from app.db import record_game_unified, RecordingConfig, RecordSource

# Configure recording
config = RecordingConfig(
    board_type="hex8",
    num_players=2,
    source=RecordSource.GUMBEL_MCTS,
    output_dir="data/games/",
    enable_parity_check=True,
)

# After game completes
game_id = record_game_unified(
    initial_state=game.initial_state,
    move_history=game.moves,
    winner=game.winner,
    config=config,
)
```

### Querying for Training Data

```python
from app.db import GameReplayDB

db = GameReplayDB("data/games/selfplay_hex8_2p.db")

# Get high-quality games for training
games = db.query_games(
    board_type="hex8",
    num_players=2,
    quality_category="high",
    limit=10000,
)

# Export training batch
for game in games:
    game_id = game["game_id"]
    moves = db.get_moves(game_id)
    for move in moves:
        features = db.get_nnue_features(game_id, move["move_number"])
        yield features, move["policy_target"], move["value_target"]
```

### Replaying a Game

```python
from app.db import GameReplayDB
from app.game_engine import GameEngine

db = GameReplayDB("data/games/selfplay.db")
game_id = "abc123"

# Get initial state and moves
state = db.get_initial_state(game_id)
moves = db.get_moves(game_id)

# Replay with validation
for i, move_data in enumerate(moves):
    move = Move.model_validate_json(move_data["move_json"])
    state = GameEngine.apply_move(state, move)

    # Verify against snapshot
    stored_hash = db.get_state_hash(game_id, i)
    computed_hash = compute_state_hash(state)
    assert stored_hash == computed_hash
```

### Batch Import

```python
from app.db import GameReplayDB, GameWriter

db = GameReplayDB("data/games/imported.db")

# Use writer context for efficient batch inserts
with db.writer() as writer:
    for game_data in import_source:
        writer.write_game(
            initial_state=game_data.initial_state,
            moves=game_data.moves,
            winner=game_data.winner,
            metadata=game_data.metadata,
        )

print(f"Imported {writer.games_written} games")
```

---

## Parity Validation

Parity validation ensures Python and TypeScript engines produce identical results.

### How It Works

1. Game is recorded with state hashes at each move
2. Parity validator replays game in TypeScript
3. Compares state hashes at each checkpoint
4. Reports first divergence if any

### Running Parity Checks

```bash
# Check single database
python scripts/check_ts_python_replay_parity.py \
  --db data/games/selfplay_hex8_2p.db \
  --sample 100

# Run canonical parity gate
python scripts/run_canonical_selfplay_parity_gate.py \
  --board-type hex8 --num-players 2

# Enable strict mode for selfplay
RINGRIFT_PARITY_VALIDATION=strict python scripts/selfplay.py ...
```

### Parity Divergence Output

```python
@dataclass
class ParityDivergence:
    game_id: str
    move_number: int
    python_hash: str
    typescript_hash: str
    python_state: dict
    typescript_state: dict
    move_at_divergence: Move
```

### Common Divergence Causes

| Cause               | Fix                       |
| ------------------- | ------------------------- |
| Float rounding      | Use integer arithmetic    |
| Map iteration order | Use sorted keys           |
| Random seed drift   | Verify RNG parity         |
| Rules difference    | Update Python to match TS |

---

## NNUE Feature Caching

Pre-compute and cache NNUE features for faster training:

```python
from app.db import (
    cache_nnue_features_for_game,
    cache_nnue_features_batch,
    GameReplayDB,
)

db = GameReplayDB("data/games/selfplay.db")

# Cache features for one game
cache_nnue_features_for_game(
    db=db,
    game_id="abc123",
    model_version="v2",
)

# Batch cache for all games
cache_nnue_features_batch(
    db_path="data/games/selfplay.db",
    model_version="v2",
    batch_size=100,
    num_workers=4,
)
```

### Feature Storage Format

Features are stored as compressed numpy arrays:

```python
# Writing
features_blob = gzip.compress(features.tobytes())

# Reading
features = np.frombuffer(
    gzip.decompress(features_blob),
    dtype=np.float32,
).reshape(expected_shape)
```

---

## Configuration

### Environment Variables

| Variable                     | Description                   | Default |
| ---------------------------- | ----------------------------- | ------- |
| `RINGRIFT_PARITY_VALIDATION` | Parity mode (off/warn/strict) | `off`   |
| `RINGRIFT_SNAPSHOT_INTERVAL` | Moves between snapshots       | `20`    |
| `RINGRIFT_DB_TIMEOUT`        | SQLite timeout seconds        | `30`    |
| `RINGRIFT_DB_BUSY_TIMEOUT`   | Busy timeout ms               | `30000` |

### Recording Configuration

```python
@dataclass
class RecordingConfig:
    board_type: str
    num_players: int
    source: RecordSource
    output_dir: str = "data/games/"
    enable_parity_check: bool = False
    enable_nnue_cache: bool = False
    snapshot_interval: int = 20
    quality_threshold: float = 0.5
```

---

## Troubleshooting

### Database Locked

```python
# Increase timeout
db = GameReplayDB("db.db", timeout=60.0)

# Or use exclusive lock
with exclusive_db_lock("db.db"):
    # Perform operations
```

### Corrupted Database

```bash
# Check integrity
python -c "
from app.db import check_database_integrity
print(check_database_integrity('data/games/corrupted.db'))
"

# Attempt recovery
python -c "
from app.db import recover_corrupted_database
recover_corrupted_database('corrupted.db', 'recovered.db')
"
```

### Schema Version Mismatch

```python
# Force upgrade
db = GameReplayDB("old.db", auto_migrate=True)

# Check version
stats = get_database_stats("old.db")
print(f"Version: {stats['schema_version']}")
```

### Large Query Performance

```python
# Use streaming for large exports
for game in db.iter_games(batch_size=100):
    process_game(game)

# Add appropriate indexes
db.execute("CREATE INDEX IF NOT EXISTS idx_quality ON games(quality_score)")
```

### Parity Validation Failures

```bash
# Get detailed divergence info
python scripts/check_ts_python_replay_parity.py \
  --db data/games/failing.db \
  --game-id abc123 \
  --verbose

# Export divergent state for analysis
python scripts/export_parity_fixture.py \
  --db data/games/failing.db \
  --game-id abc123 \
  --output fixtures/divergence_abc123.json
```

---

## See Also

- `docs/GAME_REPLAY_DATABASE_SPEC.md` - Full schema specification
- `app/training/README.md` - Training data export
- `app/rules/README.md` - Game rules engine
- `scripts/check_ts_python_replay_parity.py` - Parity testing script

---

_Last updated: December 2025_
