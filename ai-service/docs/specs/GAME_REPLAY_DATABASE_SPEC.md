# Game Replay Database Specification

## Overview

This document specifies the schema and API for a database storing complete
RingRift games from self-play for training, analysis, and replay functionality.

The database supports:

- Full game replay with step-by-step navigation (forward/backward)
- Efficient querying by game metadata (board type, winner, outcome type, etc.)
- Training data extraction for neural network and heuristic optimization
- Integration with the sandbox UI for game replay and analysis

## Design Principles

1. **Completeness**: Store all data needed to reconstruct any game state at any turn
2. **Compactness**: Avoid redundancy; derive state from initial state + move history
3. **Queryability**: Index on useful dimensions (board type, player count, outcome)
4. **Extensibility**: Schema supports future metadata (AI profiles, evaluation scores)

**Current schema version:** 16 (see `SCHEMA_VERSION` in `app/db/game_replay.py`).

## Storage Format

### Option 1: SQLite Database (Recommended for local development)

A single SQLite database with the following tables:

### Option 2: JSONL Files with Index (Recommended for bulk storage)

- One JSONL file per board type with complete game records
- A separate SQLite index for fast metadata queries

---

## Schema Definition

### Table: `games`

Primary game metadata, one row per game.

| Column                   | Type             | Description                                                                                                                                                                                                                     |
| ------------------------ | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `game_id`                | TEXT PRIMARY KEY | Unique game identifier (UUID)                                                                                                                                                                                                   |
| `board_type`             | TEXT NOT NULL    | 'square8', 'square19', 'hex8', 'hexagonal'                                                                                                                                                                                      |
| `num_players`            | INTEGER NOT NULL | 2, 3, or 4                                                                                                                                                                                                                      |
| `rng_seed`               | INTEGER          | Seed used for any stochastic elements                                                                                                                                                                                           |
| `created_at`             | TIMESTAMP        | When the game was created                                                                                                                                                                                                       |
| `completed_at`           | TIMESTAMP        | When the game ended                                                                                                                                                                                                             |
| `game_status`            | TEXT NOT NULL    | GameStatus enum (`waiting`, `active`, `paused`, `abandoned`, `completed`, `finished`)                                                                                                                                           |
| `winner`                 | INTEGER          | Player number of winner (NULL for draws)                                                                                                                                                                                        |
| `termination_reason`     | TEXT             | 'ring_elimination', 'territory_control', 'last_player_standing', 'timeout', 'resignation', 'draw', 'abandonment'                                                                                                                |
| `total_moves`            | INTEGER NOT NULL | Number of moves in the game                                                                                                                                                                                                     |
| `total_turns`            | INTEGER NOT NULL | Number of full turn cycles                                                                                                                                                                                                      |
| `duration_ms`            | INTEGER          | Total game duration in milliseconds                                                                                                                                                                                             |
| `source`                 | TEXT             | Canonical: 'self_play', 'soak_test', 'cmaes', 'gauntlet', 'tournament', 'training', 'manual' (script-specific/legacy values may include 'online_game', 'manual_import', 'selfplay_soak', 'python-strict', 'cmaes_optimization') |
| `schema_version`         | INTEGER NOT NULL | Schema version for forward compatibility (current: 16)                                                                                                                                                                          |
| `time_control_type`      | TEXT             | Time control mode ('none', 'blitz', 'rapid', 'classical', etc.)                                                                                                                                                                 |
| `initial_time_ms`        | INTEGER          | Initial clock time per player in milliseconds                                                                                                                                                                                   |
| `time_increment_ms`      | INTEGER          | Increment added after each move in milliseconds                                                                                                                                                                                 |
| `metadata_json`          | TEXT             | JSON-encoded recording metadata (engine versions, tags, etc.)                                                                                                                                                                   |
| `quality_score`          | REAL             | Training quality score (higher is better)                                                                                                                                                                                       |
| `quality_category`       | TEXT             | Quality tier label                                                                                                                                                                                                              |
| `engine_mode`            | TEXT             | AI/engine mode label (fast filtering)                                                                                                                                                                                           |
| `opponent_type`          | TEXT             | Opponent category for diversity analysis (random, heuristic, mcts, nn_v2, etc.)                                                                                                                                                 |
| `opponent_model_id`      | TEXT             | Opponent model identifier/version (when opponent_type is neural)                                                                                                                                                                |
| `parity_status`          | TEXT             | 'passed', 'failed', 'error', 'pending', 'skipped'                                                                                                                                                                               |
| `parity_checked_at`      | TEXT             | Timestamp of last parity check                                                                                                                                                                                                  |
| `parity_divergence_move` | INTEGER          | Move index where parity first diverged                                                                                                                                                                                          |

Note: in-progress recordings may temporarily store `game_status = 'active'` as a placeholder
until `finalize()` persists the terminal status.

**Indexes:**

- `idx_games_board_type` on `board_type`
- `idx_games_winner` on `winner`
- `idx_games_termination` on `termination_reason`
- `idx_games_created` on `created_at`
- `idx_games_board_players` on (`board_type`, `num_players`)
- `idx_games_config` on (`board_type`, `num_players`, `game_status`)
- `idx_games_opponent_type` on `opponent_type`
- `idx_games_board_opponent` on (`board_type`, `opponent_type`)
- `idx_games_parity_status` on `parity_status`
- `idx_games_quality_board` on (`board_type`, `quality_score` DESC)

See `app/db/game_replay.py` for the full, current index list.

---

### Table: `game_players`

Per-player metadata for each game.

| Column                   | Type             | Description                                        |
| ------------------------ | ---------------- | -------------------------------------------------- |
| `game_id`                | TEXT NOT NULL    | FK to games.game_id                                |
| `player_number`          | INTEGER NOT NULL | 1, 2, 3, or 4                                      |
| `player_type`            | TEXT NOT NULL    | 'ai', 'human'                                      |
| `ai_type`                | TEXT             | 'heuristic', 'minimax', 'mcts', 'random', 'neural' |
| `ai_difficulty`          | INTEGER          | 1-10 difficulty level                              |
| `ai_profile_id`          | TEXT             | Heuristic weight profile or NN checkpoint ID       |
| `final_eliminated_rings` | INTEGER          | Rings eliminated by this player                    |
| `final_territory_spaces` | INTEGER          | Territory spaces controlled                        |
| `final_rings_in_hand`    | INTEGER          | Rings remaining in hand                            |

**Primary Key:** (`game_id`, `player_number`)

---

### Table: `game_initial_state`

Initial game state for reconstruction. Stored as JSON blob.

| Column               | Type              | Description                         |
| -------------------- | ----------------- | ----------------------------------- |
| `game_id`            | TEXT PRIMARY KEY  | FK to games.game_id                 |
| `initial_state_json` | TEXT NOT NULL     | Full GameState as JSON              |
| `compressed`         | INTEGER DEFAULT 0 | 0 = plain JSON, 1 = gzip-compressed |

---

### Table: `game_moves`

Move history with full metadata for replay.

| Column              | Type             | Description                                           |
| ------------------- | ---------------- | ----------------------------------------------------- |
| `game_id`           | TEXT NOT NULL    | FK to games.game_id                                   |
| `move_number`       | INTEGER NOT NULL | 0-based move sequence (distinct from Move.moveNumber) |
| `turn_number`       | INTEGER NOT NULL | Which turn this move belongs to                       |
| `player`            | INTEGER NOT NULL | Player who made the move                              |
| `phase`             | TEXT NOT NULL    | Game phase when move was made                         |
| `move_type`         | TEXT NOT NULL    | MoveType enum value                                   |
| `move_json`         | TEXT NOT NULL    | Full Move object as JSON                              |
| `timestamp`         | TIMESTAMP        | When move was made                                    |
| `think_time_ms`     | INTEGER          | AI think time or human decision time                  |
| `time_remaining_ms` | INTEGER          | Time remaining for the player (optional)              |
| `engine_eval`       | REAL             | Engine evaluation score (optional)                    |
| `engine_eval_type`  | TEXT             | Evaluation type label (optional)                      |
| `engine_depth`      | INTEGER          | Search depth (optional)                               |
| `engine_nodes`      | INTEGER          | Nodes searched (optional)                             |
| `engine_pv`         | TEXT             | Principal variation (optional)                        |
| `engine_time_ms`    | INTEGER          | Engine time per move (optional)                       |
| `move_probs`        | TEXT             | JSON soft policy targets (optional)                   |
| `search_stats_json` | TEXT             | JSON search diagnostics (optional)                    |

**Primary Key:** (`game_id`, `move_number`)

**Indexes:**

- `idx_moves_game_turn` on (`game_id`, `turn_number`)
- `idx_moves_lookup` on (`game_id`, `move_number`)

Note: `game_moves.move_number` is a 0-based storage index. The embedded
`Move.moveNumber` in `move_json` remains the canonical 1-based move number.

---

### Table: `game_state_snapshots`

Optional state snapshots at key points for fast seeking.

| Column        | Type              | Description                         |
| ------------- | ----------------- | ----------------------------------- |
| `game_id`     | TEXT NOT NULL     | FK to games.game_id                 |
| `move_number` | INTEGER NOT NULL  | Move number this snapshot is AFTER  |
| `state_json`  | TEXT NOT NULL     | Full GameState as JSON              |
| `compressed`  | INTEGER DEFAULT 0 | 0 = plain JSON, 1 = gzip-compressed |
| `state_hash`  | TEXT              | Optional hash of the snapshot state |

**Primary Key:** (`game_id`, `move_number`)

**Note:** Snapshots are created every N moves (default: 20) to allow fast
seeking without replaying from the beginning.

---

### Table: `game_choices`

Player choices during decision phases (line reward, ring elimination, etc.).

| Column                 | Type             | Description                                                                                                               |
| ---------------------- | ---------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `game_id`              | TEXT NOT NULL    | FK to games.game_id                                                                                                       |
| `move_number`          | INTEGER NOT NULL | Associated move number                                                                                                    |
| `choice_type`          | TEXT NOT NULL    | 'line_reward_option', 'ring_elimination', 'line_order', 'region_order', 'capture_direction' (legacy alias: 'line_reward') |
| `player`               | INTEGER NOT NULL | Player who made the choice                                                                                                |
| `options_json`         | TEXT NOT NULL    | Available options as JSON array                                                                                           |
| `selected_option_json` | TEXT NOT NULL    | Selected option as JSON                                                                                                   |
| `ai_reasoning`         | TEXT             | Optional: AI reasoning for the choice                                                                                     |

**Primary Key:** (`game_id`, `move_number`, `choice_type`)

---

### Table: `game_history_entries`

Structured per-move history entries used for validation, trace-style
replay, and TS↔Python parity debugging.

| Column                      | Type             | Description                                                              |
| --------------------------- | ---------------- | ------------------------------------------------------------------------ |
| `game_id`                   | TEXT NOT NULL    | FK to games.game_id                                                      |
| `move_number`               | INTEGER NOT NULL | Move number this entry is AFTER                                          |
| `player`                    | INTEGER NOT NULL | Player who made the move                                                 |
| `phase_before`              | TEXT NOT NULL    | Phase before the move                                                    |
| `phase_after`               | TEXT NOT NULL    | Phase after the move                                                     |
| `status_before`             | TEXT NOT NULL    | Game status before the move                                              |
| `status_after`              | TEXT NOT NULL    | Game status after the move                                               |
| `progress_before_json`      | TEXT NOT NULL    | Per-player progress snapshot before (JSON)                               |
| `progress_after_json`       | TEXT NOT NULL    | Per-player progress snapshot after (JSON)                                |
| `state_hash_before`         | TEXT             | Optional state hash before the move                                      |
| `state_hash_after`          | TEXT             | Optional state hash after the move                                       |
| `board_summary_before_json` | TEXT             | Optional compact board summary before (JSON)                             |
| `board_summary_after_json`  | TEXT             | Optional compact board summary after (JSON)                              |
| `state_before_json`         | TEXT             | Full GameState JSON before the move (v4+)                                |
| `state_after_json`          | TEXT             | Full GameState JSON after the move (v4+)                                 |
| `compressed_states`         | INTEGER          | 0 = uncompressed JSON, 1 = base64+gzip-compressed JSON                   |
| `available_moves_json`      | TEXT             | Optional JSON array of valid moves at `state_before` (v6+, parity debug) |
| `available_moves_count`     | INTEGER          | Optional count of valid moves at `state_before` (v6+)                    |
| `engine_eval`               | REAL             | Optional engine evaluation score at this step (v6+)                      |
| `engine_depth`              | INTEGER          | Optional engine search depth (v6+)                                       |
| `fsm_valid`                 | INTEGER          | 1 = valid, 0 = invalid, NULL = not checked (v7+)                         |
| `fsm_error_code`            | TEXT             | Error code if `fsm_valid = 0` (v7+)                                      |

**Primary Key:** (`game_id`, `move_number`)

---

### Table: `game_nnue_features`

Pre-computed NNUE feature vectors for training (v8+). These eliminate the need
to replay games during dataset generation.

| Column               | Type             | Description                                 |
| -------------------- | ---------------- | ------------------------------------------- |
| `game_id`            | TEXT NOT NULL    | FK to games.game_id                         |
| `move_number`        | INTEGER NOT NULL | Move number this feature vector is AFTER    |
| `player_perspective` | INTEGER NOT NULL | Perspective player number for rotation      |
| `features`           | BLOB NOT NULL    | Compressed float32 feature vector           |
| `value`              | REAL NOT NULL    | Win/loss label (-1, 0, +1)                  |
| `board_type`         | TEXT NOT NULL    | Board type for feature dimension validation |
| `feature_dim`        | INTEGER NOT NULL | Feature dimension for validation            |

**Primary Key:** (`game_id`, `move_number`, `player_perspective`)

---

### Table: `orphaned_games`

Quarantine table for games detected without move data (v14+).

| Column            | Type             | Description                        |
| ----------------- | ---------------- | ---------------------------------- |
| `game_id`         | TEXT PRIMARY KEY | Game ID flagged as orphaned        |
| `detected_at`     | TEXT NOT NULL    | Timestamp of detection             |
| `reason`          | TEXT             | Optional detection reason          |
| `original_status` | TEXT             | Game status at time of detection   |
| `board_type`      | TEXT             | Board type for the orphaned game   |
| `num_players`     | INTEGER          | Player count for the orphaned game |

---

## Data Structures (JSON Schemas)

### GameState JSON

Matches the existing Pydantic `GameState` model serialization with these fields:

```json
{
  "id": "game_id",
  "boardType": "square8",
  "rngSeed": 12345,
  "board": {
    "type": "square8",
    "size": 8,
    "stacks": { "3,3": {...}, ... },
    "markers": { "2,2": {...}, ... },
    "collapsedSpaces": { "1,1": 1, ... },
    "eliminatedRings": { "1": 5, "2": 3 },
    "formedLines": [],
    "territories": {}
  },
  "players": [...],
  "currentPhase": "movement",
  "currentPlayer": 1,
  "moveHistory": [],  // Empty in initial_state, populated in snapshots
  "timeControl": {...},
  "gameStatus": "active",
  "winner": null,
  "createdAt": "2024-12-01T00:00:00Z",
  "lastMoveAt": "2024-12-01T00:00:00Z",
  "maxPlayers": 2,
  "totalRingsInPlay": 36,
  "totalRingsEliminated": 0,
  "victoryThreshold": 18,
  "territoryVictoryThreshold": 33,
  "lpsRoundIndex": 0,
  "lpsCurrentRoundActorMask": {},
  "rulesOptions": { "swapRuleEnabled": false }
}
```

Note: `currentPlayer` uses the canonical playerNumber (1-based), not a 0-based index.

### Move JSON

Matches the existing Pydantic `Move` model serialization.

---

## API Specification

### Python API (ai-service)

```python
from typing import Optional, List, Iterator
from app.models import GameState, Move, BoardType

class GameReplayDB:
    """Database interface for game storage and replay."""

    def __init__(self, db_path: str):
        """Initialize database connection."""
        ...

    # Write operations
    def store_game(
        self,
        game_id: str,
        initial_state: GameState,
        final_state: GameState,
        moves: List[Move],
        choices: List[dict],
        metadata: dict,
    ) -> None:
        """Store a complete game with all associated data."""
        ...

    def store_game_incremental(
        self,
        game_id: str,
        initial_state: GameState,
    ) -> "GameWriter":
        """Begin incremental game storage (for live games)."""
        ...

    # Read operations
    def get_game_metadata(self, game_id: str) -> Optional[dict]:
        """Get game metadata without loading full state."""
        ...

    def get_initial_state(self, game_id: str) -> Optional[GameState]:
        """Get the initial game state."""
        ...

    def get_moves(
        self,
        game_id: str,
        start: int = 0,
        end: Optional[int] = None,
    ) -> List[Move]:
        """Get moves in a range."""
        ...

    def get_state_at_move(
        self,
        game_id: str,
        move_number: int,
        auto_inject: Optional[bool] = None,
    ) -> Optional[GameState]:
        """Reconstruct state at a specific move number."""
        ...

    def get_choices_at_move(
        self,
        game_id: str,
        move_number: int,
    ) -> List[dict]:
        """Get player choices made at a specific move."""
        ...

    # Query operations
    def query_games(
        self,
        board_type: Optional[BoardType] = None,
        num_players: Optional[int] = None,
        winner: Optional[int] = None,
        termination_reason: Optional[str] = None,
        source: Optional[str] = None,
        min_moves: Optional[int] = None,
        max_moves: Optional[int] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[dict]:
        """Query games by metadata filters."""
        ...

    def iterate_games(
        self,
        **filters,
    ) -> Iterator[tuple[dict, GameState, List[Move]]]:
        """Iterate over games matching filters (for bulk processing)."""
        ...

    # Maintenance
    def vacuum(self) -> None:
        """Optimize database storage."""
        ...

    def get_stats(self) -> dict:
        """Get database statistics."""
        ...


class GameWriter:
    """Incremental game writer for live games."""

    def add_move(self, move: Move) -> None:
        """Add a move to the game."""
        ...

    def add_choice(
        self,
        move_number: int,
        choice_type: str,
        options: List[dict],
        selected: dict,
        reasoning: Optional[str] = None,
    ) -> None:
        """Record a player choice."""
        ...

    def finalize(
        self,
        final_state: GameState,
        metadata: dict,
    ) -> None:
        """Finalize and close the game record."""
        ...

    def abort(self) -> None:
        """Abort an incomplete game."""
        ...
```

### TypeScript Integration (Sandbox Replay)

The sandbox UI consumes replays via the `/api/replay/*` REST API. The canonical
TypeScript shapes live in `src/client/types/replay.ts` and are wrapped by
`src/client/services/ReplayService.ts`.

```typescript
// GET /api/replay/games?board_type=...&num_players=...&limit=...
interface GameListResponse {
  games: GameMetadata[];
  total: number;
  hasMore: boolean;
}

// GET /api/replay/games/{gameId}
interface GameMetadata {
  gameId: string;
  boardType: BoardType;
  numPlayers: number;
  winner: number | null;
  terminationReason: string | null;
  totalMoves: number;
  totalTurns: number;
  createdAt: string;
  completedAt?: string | null;
  durationMs?: number | null;
  source?: string | null;
  timeControlType?: string;
  initialTimeMs?: number;
  timeIncrementMs?: number;
  metadata?: Record<string, unknown>;
  players?: PlayerMetadata[];
}

// GET /api/replay/games/{gameId}/state?move_number=N&legacy=false
interface ReplayState {
  gameState: GameState;
  moveNumber: number;
  totalMoves: number;
  engineEval?: number;
  enginePV?: string[];
}

// GET /api/replay/games/{gameId}/moves?start=0&end=100&limit=1000
interface MovesResponse {
  moves: MoveRecord[];
  hasMore: boolean;
}

// GET /api/replay/games/{gameId}/choices?move_number=N
interface ChoicesResponse {
  choices: ChoiceRecord[];
}

// POST /api/replay/games
interface StoreGameRequest {
  gameId?: string;
  initialState: GameState;
  finalState: GameState;
  moves: Move[];
  choices?: ChoiceRecord[];
  metadata?: Record<string, unknown>;
}

interface StoreGameResponse {
  gameId: string;
  totalMoves: number;
  success: boolean;
}

// GET /api/replay/stats
interface StatsResponse {
  totalGames: number;
  gamesByBoardType: Record<string, number>;
  gamesByStatus: Record<string, number>;
  gamesByTermination: Record<string, number>;
  totalMoves: number;
  schemaVersion: number;
}
```

---

## Integration with Self-Play

### Automatic Storage Hook

The self-play soak and pool generation scripts will automatically store
completed games:

```python
# In run_self_play_soak.py

def on_game_complete(
    game_id: str,
    initial_state: GameState,
    final_state: GameState,
    moves: List[Move],
    choices: List[dict],
    outcome: str,
    metadata: dict,
) -> None:
    """Called when a game completes successfully."""
    if db is not None and final_state.game_status in (GameStatus.FINISHED, GameStatus.COMPLETED):
        db.store_game(
            game_id=game_id,
            initial_state=initial_state,
            final_state=final_state,
            moves=moves,
            choices=choices,
            metadata={
                **metadata,
                "source": "self_play",
                "termination_reason": outcome,
            },
        )
```

### Validation Before Storage

Games are only stored if they meet these criteria:

1. `game_status` is `FINISHED` or `COMPLETED` (not `ACTIVE` or `ABANDONED`)
2. A winner is determined (or valid stalemate with tiebreaker)
3. Move history is non-empty and consistent
4. Final state passes invariant checks (S is monotonic, no invalid termination)

---

## Coverage & Health Targets

The schema above is only useful if the underlying databases have good,
representative coverage and structurally sound games. This section documents
the **“golden” baseline** we aim for in local development and CI, and the
scripts that help maintain it.

### Structural Coverage (RandomAI-only, fast)

For basic structural validation (schema correctness, TS↔Python replay parity,
invariant checks) we maintain a light-weight set of RandomAI games across all
board/player combinations using `scripts/run_self_play_soak.py` in
`random-only` mode.

Target baseline (per DB path):

| Board     | Players | Min finished games | Source            | DB path                               |
| --------- | ------- | ------------------ | ----------------- | ------------------------------------- |
| square8   | 2       | 20+                | `random_selfplay` | `data/games/selfplay_square8_2p.db`   |
| square8   | 3       | 15+                | `random_selfplay` | `data/games/selfplay_square8_3p.db`   |
| square8   | 4       | 10+                | `random_selfplay` | `data/games/selfplay_square8_4p.db`   |
| square19  | 2       | 15+                | `random_selfplay` | `data/games/selfplay_square19_2p.db`  |
| square19  | 3       | 10+                | `random_selfplay` | `data/games/selfplay_square19_3p.db`  |
| square19  | 4       | 10+                | `random_selfplay` | `data/games/selfplay_square19_4p.db`  |
| hexagonal | 2       | 15+                | `random_selfplay` | `data/games/selfplay_hexagonal_2p.db` |
| hexagonal | 3       | 10+                | `random_selfplay` | `data/games/selfplay_hexagonal_3p.db` |
| hexagonal | 4       | 10+                | `random_selfplay` | `data/games/selfplay_hexagonal_4p.db` |

These numbers are deliberately modest so they can be regenerated quickly on a
developer laptop while still providing coverage of:

- all three board types (`square8`, `square19`, `hexagonal`), and
- all supported player counts (2–4).

To (re)generate this structural baseline in one shot:

```bash
cd ai-service

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
  bash -c 'for board in square8 square19 hexagonal; do \
    for players in 2 3 4; do \
      python scripts/run_self_play_soak.py \
        --board-type "$board" \
        --num-players "$players" \
        --num-games 10 \
        --engine-mode random-only \
        --record-db "data/games/selfplay_${board}_${players}p.db"; \
    done; \
  done'
```

This loop calls `run_self_play_soak.py` for each `(board_type, num_players)`
pair using `RandomAI` only, so it avoids neural-network/MPS issues and remains
CPU-friendly.

### Training / Parity Coverage (mixed engine, CMA-ES)

For training and deeper parity analysis we additionally want:

1. **Mixed-engine self-play soaks** (2p, heuristic + neural ladder),
2. **CMA-ES evaluation games** (2p, heuristic‑only, multi-start).

We do not pin exact numbers here because they can vary by experiment, but a
useful **golden baseline** is:

| Board     | Profile        | Recommended finished games (2p) | Typical source                                         |
| --------- | -------------- | ------------------------------- | ------------------------------------------------------ |
| square8   | self-play soak | ≥ 100                           | `soak_test` (legacy: `selfplay_soak`, `python-strict`) |
| square19  | self-play soak | ≥ 50                            | `soak_test` (legacy: `selfplay_soak`)                  |
| hexagonal | self-play soak | ≥ 40 (when NN/MPS is stable)    | `soak_test` (legacy: `selfplay_soak`)                  |
| square8   | CMA-ES eval    | ≥ 100                           | `cmaes` (legacy: `cmaes_optimization`)                 |
| square19  | CMA-ES eval    | ≥ 60                            | `cmaes` (legacy: `cmaes_optimization`)                 |
| hexagonal | CMA-ES eval    | ≥ 40 (optional)                 | `cmaes` (legacy: `cmaes_optimization`)                 |

Small helper scripts exist for “quick but representative” coverage:

- `scripts/run_selfplay_matrix.sh` – mixed‑engine self-play soaks for
  2–4 players on `square8`/`square19`, recording to
  `data/games/selfplay_<board>_<players>p.db`.
- `scripts/run_cmaes_matrix.sh` – short CMA‑ES runs on `square8`/`square19`
  that record evaluation games to `logs/cmaes/runs/<run_id>/games.db`.

These are not a replacement for the full training runs in `AI_TRAINING_PLAN.md`
but provide a reproducible “good enough” replay corpus for:

- TS↔Python parity harnesses,
- regression tests on invariants, and
- ad‑hoc offline analysis.

### Health Checks (`db_health.*.json`)

The `scripts/cleanup_useless_replay_dbs.py` tool inspects all configured
`GameReplayDB` locations and produces a health summary JSON. This is the
canonical way to verify that databases meet the structural expectations above.

Example usage:

```bash
cd ai-service

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
  python scripts/cleanup_useless_replay_dbs.py \
    --summary-json db_health.current.json
```

The resulting `db_health.current.json` summarises, per DB:

- `total_games`, `board_type_counts`, `num_players_counts`
- `source_counts` (e.g. `self_play`, `soak_test`, `cmaes`; legacy sources may appear)
- `structure_counts` (`good`, `internal_inconsistent`, etc.)
- `termination_reason_counts`

Databases with **zero good games** or structurally inconsistent games are
reported as “useless” and can be safely regenerated. When promoting a replay
set to “golden”, we typically:

1. Regenerate the RandomAI matrix (run `scripts/run_self_play_soak.py` in
   `random-only` mode for each board/player pair, as in the structural
   coverage loop above),
2. Run a small mixed/CMA‑ES matrix (`run_selfplay_matrix.sh`,
   `run_cmaes_matrix.sh`) if needed,
3. Capture a snapshot as `db_health.golden.json` under version control.

---

### Parity & Debugging Tooling

On top of structural health, we use a TS↔Python replay parity harness and
debug helpers to validate that engines agree on the canonical rules sequence
for a given `GameReplayDB`.

Key entrypoints:

- `scripts/check_ts_python_replay_parity.py`
  - Compares Python `GameReplayDB.get_state_at_move` against the TS sandbox
    replay path (`scripts/selfplay-db-ts-replay.ts`) for each game.
  - Important flags:
    - `--db <path>` – restrict to a single DB.
    - `--compact` – emit one-line `SEMANTIC …` entries for divergences
      (greppable, no JSON summary).
    - `--emit-fixtures-dir <dir>` – write a compact JSON fixture per semantic
      divergence (db, game_id, diverged_at, summaries, canonical move). These
      fixtures are consumed by `tests/parity/test_replay_parity_fixtures_regression.py`.
    - `--emit-state-bundles-dir <dir>` – write a richer **state bundle** per
      semantic divergence, capturing full serialized TS and Python `GameState`
      JSON immediately before and at the first divergent step.

- `scripts/selfplay-db-ts-replay.ts`
  - Node/TS harness that replays a single DB game into `ClientSandboxEngine`
    with `traceMode=true` and prints per-step summaries.
  - Supports `--dump-state-at k1,k2,…` (or `RINGRIFT_TS_REPLAY_DUMP_STATE_AT_K`)
    to dump TS `GameState` JSON snapshots to `RINGRIFT_TS_REPLAY_DUMP_DIR` (or
    `./ts-replay-dumps` by default). This is what the parity harness uses under
    the hood when emitting state bundles.

- `ai-service/scripts/diff_state_bundle.py`
  - Convenience CLI for inspecting a single `.state_bundle.json` emitted by
    the parity harness:
    - Reconstructs the Python `GameState` via `deserialize_game_state`.
    - Compares players, stacks, and collapsed territory against the TS state
      (reusing the same parity summarizers as `app/db/parity_validator.py`).
    - Prints a concise structural diff summary plus per-player eliminations /
      territory / rings-in-hand.
  - Typical usage:

    ```bash
    cd ai-service

    # After running check_ts_python_replay_parity with --emit-state-bundles-dir
    PYTHONPATH=. python scripts/diff_state_bundle.py \
      --bundle parity_fixtures/state_bundles/selfplay_square19_parity__<gameId>__k142.state_bundle.json
    ```

This tooling is intended to make it easy to go from “parity summary says
there is a semantic divergence” to a concrete, inspectable pair of TS and
Python states (and the move that caused them) with a single CLI call.

## Replay Navigation in Sandbox

The sandbox UI supports game replay with these controls:

1. **Step Forward**: Apply next move, update state
2. **Step Backward**: Reconstruct state at previous move number
3. **Jump to Move**: Seek to any move number using snapshots + replay
4. **Play/Pause**: Automatic playback at configurable speed
5. **Choice Inspection**: View available choices and the selected option at decision points

### State Reconstruction Algorithm

```python
def get_state_at_move(game_id: str, target_move: int) -> GameState:
    """Reconstruct game state at a specific move."""

    # Find nearest snapshot before target
    snapshot = find_nearest_snapshot(game_id, target_move)

    if snapshot:
        state = GameState.model_validate_json(snapshot.state_json)
        start_move = snapshot.move_number + 1
    else:
        state = get_initial_state(game_id)
        start_move = 0

    # Replay moves from snapshot to target
    moves = get_moves(game_id, start=start_move, end=target_move + 1)
    for move in moves:
        state = GameEngine.apply_move(state, move)

    return state
```

Notes:

- Canonical DBs default to strict replay (no phase injection).
- Legacy DBs may opt in to phase injection via `auto_inject=True`.

---

## Storage Estimates

### Per-Game Storage

| Component                 | Size (approx)      |
| ------------------------- | ------------------ |
| Initial state JSON        | 2-10 KB            |
| Move (avg)                | 200-500 bytes      |
| Snapshot (every 20 moves) | 2-10 KB            |
| Choices                   | 100-500 bytes each |

### Estimated Database Sizes

| Games     | Avg Moves | Est. Size     |
| --------- | --------- | ------------- |
| 1,000     | 50        | 50-100 MB     |
| 10,000    | 50        | 500 MB - 1 GB |
| 100,000   | 50        | 5-10 GB       |
| 1,000,000 | 50        | 50-100 GB     |

---

## Migration Path

### Phase 1: Core Implementation

1. Implement SQLite schema and GameReplayDB class
2. Add storage hook to self-play soak script
3. Validate storage/retrieval with existing self-play runs

### Phase 2: Query and Analysis

1. Implement query_games and iterate_games
2. Add training data extraction utilities
3. Build CLI tools for database inspection

### Phase 3: Sandbox Integration

1. Add REST API endpoints for replay
2. Implement sandbox replay UI controls
3. Add choice inspection visualization

---

## File Locations

- Primary replay DBs live under `data/games/`:
  - `data/games/selfplay.db` (default `run_self_play_soak.py` output)
  - `data/games/selfplay_<board>_<players>p.db` (matrix/soak outputs)
  - `data/games/tournament_<board>_<players>p.db`
  - `data/games/gauntlet_<board>_<players>p.db`
  - `data/games/baseline_calibration_<board>_<players>p.db`
- Canonical DBs typically live under `data/selfplay/`:
  - `data/selfplay/canonical_<board>_<players>p.db`
  - `data/selfplay/canonical_<board>.db` (multi-player)
  - Legacy canonical DBs may still appear under `data/games/canonical_*.db`
- Session DBs:
  - `data/selfplay/unified_*/games.db`
  - `data/selfplay/p2p/**/games.db`
  - `data/selfplay/p2p_hybrid/**/games.db`
- Backup location: `data/games/backups/`
- Schema migrations: `app/db/migrations/`
- API implementation: `app/db/game_replay.py`

`app/utils/game_discovery.py` is the authoritative source for discovery patterns.

---

## Version History

| Version | Date       | Changes               |
| ------- | ---------- | --------------------- |
| 1.0     | 2024-12-01 | Initial specification |
