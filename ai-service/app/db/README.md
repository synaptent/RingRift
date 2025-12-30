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
    winner=1,  # playerNumber (1-based)
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

| Method                             | Description                            |
| ---------------------------------- | -------------------------------------- |
| `query_games()`                    | Filter games by metadata criteria      |
| `get_game_metadata()`              | Get game summary data                  |
| `get_initial_state()`              | Get starting state                     |
| `get_moves()`                      | Get all moves for a game               |
| `get_state_at_move()`              | Reconstruct state at move N            |
| `get_choices_at_move()`            | Get recorded player choices at a move  |
| `get_nnue_features()`              | Load cached NNUE features for a game   |
| `get_nnue_features_for_training()` | Stream cached features by board config |

### UnifiedGameRecorder

High-level API for recording games with automatic features:

```python
from app.db import (
    UnifiedGameRecorder,
    RecordingConfig,
    RecordSource,
)

config = RecordingConfig(
    board_type="hex8",
    num_players=2,
    source=RecordSource.SELF_PLAY,
    parity_mode="strict",
    snapshot_interval=20,
)

with UnifiedGameRecorder(config, initial_state) as recorder:
    for move, state_after in move_stream:
        recorder.add_move(move, state_after=state_after)
    recorder.finalize(final_state, extra_metadata={"model_id": "v2"})
```

For completed games already in memory, use the one-shot helper:

```python
from app.db import record_game_unified, RecordingConfig, RecordSource

game_id = record_game_unified(
    config=config,
    initial_state=initial_state,
    final_state=final_state,
    moves=moves,
    extra_metadata={"model_id": "v2"},
    with_parity_check=True,
)
```

Legacy one-shot helpers (still supported, but prefer `record_game_unified`):

```python
from app.db import (
    GameReplayDB,
    record_completed_game,
    record_completed_game_with_parity_check,
)

db = GameReplayDB("data/games/selfplay.db")

game_id = record_completed_game(
    db=db,
    initial_state=initial_state,
    final_state=final_state,
    moves=moves,
    metadata={"model_id": "v2"},
)

# Optional: validate parity immediately after recording
game_id = record_completed_game_with_parity_check(
    db=db,
    initial_state=initial_state,
    final_state=final_state,
    moves=moves,
    metadata={"model_id": "v2"},
)
```

#### Record Sources

| Source                    | Description                        |
| ------------------------- | ---------------------------------- |
| `RecordSource.SELF_PLAY`  | Self-play data collection          |
| `RecordSource.SOAK_TEST`  | Long-running soak tests            |
| `RecordSource.CMAES`      | CMA-ES optimization runs           |
| `RecordSource.GAUNTLET`   | Evaluation gauntlets               |
| `RecordSource.TOURNAMENT` | Tournament games                   |
| `RecordSource.TRAINING`   | Training data generation           |
| `RecordSource.MANUAL`     | Manual imports / ad-hoc recordings |

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

Additional parity controls:

- `RINGRIFT_PARITY_BACKEND` selects `auto`, `ts`, `python_only`, `ts_hashes`, or `skip`.
- `RINGRIFT_SKIP_PARITY` (`1`/`true`) disables parity validation entirely.
- `RINGRIFT_PARITY_DUMP_DIR` overrides the failure bundle output directory.

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

Current schema version: **15** (see `app/db/game_replay.py`).

For column-level definitions, see
[`../../docs/specs/GAME_REPLAY_DATABASE_SPEC.md`](../../docs/specs/GAME_REPLAY_DATABASE_SPEC.md).

### Tables

| Table                  | Purpose                                                             |
| ---------------------- | ------------------------------------------------------------------- |
| `games`                | Game metadata (status, winner, source, quality, parity status)      |
| `game_players`         | Per-player final stats (eliminated rings, territory, rings in hand) |
| `game_initial_state`   | Serialized initial `GameState`                                      |
| `game_moves`           | Ordered move list with metadata (0-based `move_number`)             |
| `game_state_snapshots` | Periodic state snapshots + hashes for validation/training           |
| `game_history_entries` | Optional before/after states + available moves for parity debugging |
| `game_choices`         | PlayerChoice payloads captured during decision phases               |
| `game_nnue_features`   | Cached NNUE features for training                                   |
| `schema_metadata`      | Schema version key/value                                            |
| `orphaned_games`       | Quarantine table for invalid/partial records                        |

Notes:

- `game_moves.move_number` is a 0-based storage index; `Move.moveNumber` stays 1-based.
- `game_history_entries` is only populated when `store_history_entries=True`.

### Schema Migrations

The database auto-migrates when opening older versions:

```python
db = GameReplayDB("old_database.db")
# Automatically migrates to SCHEMA_VERSION=15
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
    source=RecordSource.SELF_PLAY,
    db_dir="data/games/",
    parity_mode="strict",
)

# After game completes
game_id = record_game_unified(
    config=config,
    initial_state=game.initial_state,
    final_state=game.final_state,
    moves=game.moves,
    extra_metadata={"model_id": "v2"},
    with_parity_check=True,
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
    for move_num, player_persp, features, value in db.get_nnue_features(game_id):
        yield features, value, (game_id, move_num, player_persp)
```

### Replaying a Game

```python
from app.db import GameReplayDB
from app.game_engine import GameEngine
from app.models import Move

db = GameReplayDB("data/games/selfplay.db")
game_id = "abc123"

# Get initial state and moves
state = db.get_initial_state(game_id)
moves = db.get_moves(game_id)

# Replay with validation
for i, move_data in enumerate(moves):
    move = Move.model_validate_json(move_data["move_json"])
    state = GameEngine.apply_move(state, move)

# Or reconstruct directly
state_at_15 = db.get_state_at_move(game_id, move_number=15)
```

### Batch Import

```python
from app.db import GameReplayDB

db = GameReplayDB("data/games/imported.db")

for game_data in import_source:
    db.store_game(
        game_id=game_data.game_id,
        initial_state=game_data.initial_state,
        final_state=game_data.final_state,
        moves=game_data.moves,
        metadata=game_data.metadata,
        store_history_entries=False,
    )
```

---

## Parity Validation

Parity validation ensures Python and TypeScript engines produce identical results.

### How It Works

1. Game is recorded with the move list (and optional snapshots/history entries)
2. Parity validator replays the game in Python and TypeScript
3. Compares computed hashes, phase, player, and status at each move
4. Reports the first divergence if any

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
    db_path: str
    diverged_at: int
    mismatch_kinds: list[str]
    mismatch_context: str
    total_moves_python: int
    total_moves_ts: int
    python_summary: StateSummary | None
    ts_summary: StateSummary | None
    move_at_divergence: dict[str, Any] | None
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

If you need a single-call record + NNUE cache path, use:

```python
from app.db import GameReplayDB, record_completed_game_with_nnue_cache

db = GameReplayDB("data/games/selfplay.db")

game_id = record_completed_game_with_nnue_cache(
    db=db,
    initial_state=initial_state,
    final_state=final_state,
    moves=moves,
    metadata={"model_id": "v2"},
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

| Variable                     | Description                   | Default           |
| ---------------------------- | ----------------------------- | ----------------- |
| `RINGRIFT_PARITY_VALIDATION` | Parity mode (off/warn/strict) | `off`             |
| `RINGRIFT_PARITY_BACKEND`    | Parity backend selector       | `auto`            |
| `RINGRIFT_PARITY_DUMP_DIR`   | Parity failure dump directory | `parity_failures` |
| `RINGRIFT_SKIP_PARITY`       | Skip parity validation        | `false`           |
| `RINGRIFT_SNAPSHOT_INTERVAL` | Moves between snapshots       | `20`              |

SQLite timeout values are configured in `app/config/thresholds.py`
(`SQLITE_TIMEOUT`, `SQLITE_BUSY_TIMEOUT_MS`).

### Recording Configuration

```python
from dataclasses import dataclass, field
from app.db import RecordSource

@dataclass
class RecordingConfig:
    board_type: str
    num_players: int
    source: str = RecordSource.SELF_PLAY
    # Optional metadata
    difficulty: int | None = None
    engine_mode: str | None = None
    model_id: str | None = None
    generation: int | None = None
    candidate_id: str | None = None
    tags: list[str] = field(default_factory=list)
    # Database configuration
    db_path: str | None = None
    db_prefix: str = "selfplay"
    db_dir: str = "data/games"
    # Recording options
    store_history_entries: bool = True
    snapshot_interval: int | None = None
    parity_mode: str | None = None
    fsm_validation: bool = False
```

---

## Troubleshooting

### Database Locked

```python
# Use DELETE journal mode on NFS mounts
db = GameReplayDB("db.db", journal_mode="DELETE")

# Check write locks before syncing
from pathlib import Path
from app.db.write_lock import is_database_safe_to_sync
is_safe = is_database_safe_to_sync(Path("db.db"))
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

- `ai-service/docs/specs/GAME_REPLAY_DATABASE_SPEC.md` - Full schema specification
- `app/training/README.md` - Training data export
- `app/rules/README.md` - Game rules engine
- `scripts/check_ts_python_replay_parity.py` - Parity testing script

---

_Last updated: December 2025_
