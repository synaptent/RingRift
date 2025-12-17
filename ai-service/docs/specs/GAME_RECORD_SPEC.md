# RingRift Game Record Specification

> **Doc Status (2025-12-14): Active (Phase 1–2 implemented)**  
> Core GameRecord types and JSONL format are implemented in both Python and TypeScript, and online games now populate canonical GameRecords via `GameRecordRepository` while self-play generators can emit `GameRecord` JSONL for training.

## Overview

This document specifies the canonical format for storing complete RingRift games with full replay capability. The format supports:

- Complete game reconstruction from initial state
- Forward/backward navigation through every turn action
- Human-readable algebraic notation
- Machine-readable numeric coordinates
- All board types (Square8, Square19, Hexagonal)
- Metadata for analysis, training, and archival

## 1. Game Record Schema

### 1.1 Top-Level Structure

```json
{
  "format_version": "1.0",
  "game_id": "uuid-v4-string",

  "metadata": {
    /* Game metadata */
  },
  "initial_state": {
    /* Starting position */
  },
  "moves": [
    /* Ordered list of all moves */
  ],
  "final_state": {
    /* Ending position */
  },
  "outcome": {
    /* Game result */
  },

  "annotations": {
    /* Optional analysis data */
  }
}
```

### 1.2 Metadata Block

```json
{
  "metadata": {
    "board_type": "square8" | "square19" | "hexagonal",
    "board_size": 8 | 19 | 25,  // hex uses radius-12 (25×25 grid)

    "players": [
      {
        "number": 1,
        "name": "Player One",
        "type": "human" | "ai",
        "ai_config": {  /* If AI */
          "difficulty": 5,
          "profile": "heuristic_v1_balanced",
          "seed": 12345
        }
      }
    ],

    "time_control": {
      "type": "standard" | "blitz" | "none",
      "initial_ms": 600000,
      "increment_ms": 0
    },

    "rules_version": "1.0",
    "victory_threshold": 5,
    "territory_threshold": 15,
    "swap_rule_enabled": true,

    "created_at": "2024-12-01T00:00:00Z",
    "source": "self_play" | "online" | "sandbox" | "import",
    "rng_seed": 42,

    "tags": ["training", "analysis", "tournament"]
  }
}
```

### 1.3 Move Record

Each move contains both algebraic and numeric representations:

```json
{
  "moves": [
    {
      "index": 0,
      "player": 1,
      "phase": "RING_PLACEMENT",
      "type": "PLACE_RING",

      "algebraic": "P1@d4",
      "notation": {
        "action": "place",
        "target": "d4",
        "count": 1
      },

      "numeric": {
        "type": "PLACE_RING",
        "position": { "row": 3, "col": 3 },
        "placement_count": 1
      },

      "timestamp_ms": 1234567890,
      "time_remaining_ms": 595000,
      "think_time_ms": 5000,

      "state_after_hash": "abc123..."
    }
  ]
}
```

### 1.4 State Snapshot

For initial and final states, and optionally at checkpoints:

```json
{
  "initial_state": {
    "board": {
      "stacks": {
        "d4": { "owner": 1, "height": 2 },
        "e5": { "owner": 2, "height": 1 }
      },
      "markers": {
        "c3": 1,
        "f6": 2
      },
      "collapsed_spaces": ["a1", "h8"],
      "eliminated_rings": {}
    },

    "players": [
      {
        "number": 1,
        "rings_in_hand": 18,
        "eliminated_rings": 0,
        "territory_spaces": 0
      },
      {
        "number": 2,
        "rings_in_hand": 18,
        "eliminated_rings": 0,
        "territory_spaces": 0
      }
    ],

    "current_player": 1,
    "current_phase": "RING_PLACEMENT",
    "move_number": 0,

    "zobrist_hash": "...",
    "lps_state": {
      /* LPS round tracking */
    }
  }
}
```

### 1.5 Outcome Block

```json
{
  "outcome": {
    "status": "finished" | "abandoned" | "timeout" | "error",
    "winner": 1 | 2 | null,
    "termination_reason": "ring_elimination" | "territory" | "resignation" | "timeout" | "stalemate" | "lps",

    "final_scores": {
      "player1": {
        "eliminated_rings": 5,
        "territory_spaces": 8,
        "rings_remaining": 13
      },
      "player2": {
        "eliminated_rings": 3,
        "territory_spaces": 4,
        "rings_remaining": 15
      }
    },

    "total_moves": 47,
    "total_time_ms": 125000
  }
}
```

## 2. Algebraic Notation (RRN - RingRift Notation)

### 2.1 Coordinate Systems

#### Square Boards (8x8, 19x19)

- Columns: a-h (8x8) or a-s (19x19)
- Rows: 1-8 (8x8) or 1-19 (19x19)
- Example: `d4`, `k10`, `s19`

#### Hexagonal Board (11-hex)

- Uses axial coordinates with letter-number format
- Center is `f6`
- Columns: a-k (11 columns)
- Rows: 1-11 with offset for hex grid
- Example: `f6`, `d4`, `h8`

### 2.2 Move Notation

#### Ring Placement

```
P[count]@[coord]    Place rings
P1@d4               Place 1 ring at d4
P2@e5               Place 2 rings at e5
P3@f6               Place 3 rings at f6
```

#### Ring Movement

```
[from]-[to]         Move stack
d4-e5               Move from d4 to e5
d4-e5x              Move with capture
d4-e5xf6            Chain capture continuing to f6
```

#### Capture Actions

```
[from]x[to]         Capture
d4xe5               Capture at e5 from d4
d4xe5xf6            Chain capture
```

#### Line Processing

```
L[line_id]:[action]         Line decision
L1:remove(d4-d8)            Remove line segment d4-d8
L1:keep                     Keep line (place markers)
L1:elim(2)@e5               Eliminate from player 2 at e5
```

#### Territory Processing

```
T[region_id]:elim(P)@[coord]   Territory elimination
T1:elim(1)@d4                   Eliminate P1 ring at d4
```

#### Swap Rule

```
SWAP                Accept swap (switch sides)
DECLINE             Decline swap offer
```

### 2.3 Example Game Fragment

```
1. P1@d4           P1@e5
2. P2@c3           P2@f6
3. d4-e4           e5-d5
4. P1@g4           P1@h5
5. e4-f4           d5-c5
6. f4xd5           c5xe4xg4    {chain capture}
7. L1:remove(d5-g5) elim(1)@h5
...
Result: 1-0 (ring elimination)
```

## 3. Numeric Coordinate Mapping

### 3.1 Square Board Mapping

```python
def algebraic_to_numeric_square(coord: str) -> tuple[int, int]:
    """Convert 'd4' to (row=3, col=3) for square boards."""
    col = ord(coord[0]) - ord('a')  # 0-indexed
    row = int(coord[1:]) - 1        # 0-indexed
    return (row, col)

def numeric_to_algebraic_square(row: int, col: int) -> str:
    """Convert (3, 3) to 'd4' for square boards."""
    return f"{chr(ord('a') + col)}{row + 1}"
```

### 3.2 Hexagonal Board Mapping

```python
def algebraic_to_numeric_hex(coord: str) -> tuple[int, int]:
    """Convert hex algebraic to axial coordinates."""
    col_letter = coord[0]
    row_num = int(coord[1:])

    col = ord(col_letter) - ord('a')  # 0-10
    # Hex row offset calculation
    row = row_num - 1

    return (row, col)

# Axial coordinate system for hex:
# q = column (0-10, left to right)
# r = row (0-10, top to bottom with offset)
```

### 3.3 Coordinate Lookup Tables

For unambiguous mapping, store lookup tables:

```json
{
  "coordinate_map": {
    "square8": {
      "a1": {"row": 0, "col": 0},
      "a2": {"row": 1, "col": 0},
      ...
      "h8": {"row": 7, "col": 7}
    },
    "hexagonal": {
      "a1": {"q": 0, "r": 0},
      "f6": {"q": 5, "r": 5},
      ...
    }
  }
}
```

## 4. Database Schema

### 4.1 Primary Tables

```sql
-- Game records
CREATE TABLE games (
    id UUID PRIMARY KEY,
    format_version VARCHAR(10) NOT NULL,
    board_type VARCHAR(20) NOT NULL,

    metadata JSONB NOT NULL,
    initial_state JSONB NOT NULL,
    final_state JSONB NOT NULL,
    outcome JSONB NOT NULL,

    winner INTEGER,
    termination_reason VARCHAR(50),
    total_moves INTEGER NOT NULL,

    created_at TIMESTAMP NOT NULL,
    source VARCHAR(20) NOT NULL,

    -- Indexes for common queries
    INDEX idx_board_type (board_type),
    INDEX idx_winner (winner),
    INDEX idx_created_at (created_at),
    INDEX idx_source (source)
);

-- Move records (separate for efficient querying)
CREATE TABLE moves (
    id SERIAL PRIMARY KEY,
    game_id UUID REFERENCES games(id),
    move_index INTEGER NOT NULL,

    player INTEGER NOT NULL,
    phase VARCHAR(30) NOT NULL,
    move_type VARCHAR(30) NOT NULL,

    algebraic VARCHAR(100) NOT NULL,
    numeric JSONB NOT NULL,

    state_after_hash VARCHAR(64),
    think_time_ms INTEGER,

    UNIQUE (game_id, move_index)
);

-- Tags for categorization
CREATE TABLE game_tags (
    game_id UUID REFERENCES games(id),
    tag VARCHAR(50) NOT NULL,
    PRIMARY KEY (game_id, tag)
);
```

### 4.2 JSONL File Format (Alternative)

For file-based storage (training data, exports):

```jsonl
{"format_version":"1.0","game_id":"...","metadata":{...},"moves":[...],"outcome":{...}}
{"format_version":"1.0","game_id":"...","metadata":{...},"moves":[...],"outcome":{...}}
```

## 5. Replay Support

### 5.1 State Reconstruction

To replay a game, start from `initial_state` and apply moves sequentially:

```python
def reconstruct_state_at_move(game_record: dict, move_index: int) -> GameState:
    """Reconstruct game state at a specific move."""
    state = deserialize_state(game_record["initial_state"])
    rules = DefaultRulesEngine()

    for i, move_data in enumerate(game_record["moves"]):
        if i >= move_index:
            break
        move = deserialize_move(move_data["numeric"])
        state = rules.apply_move(state, move)

    return state
```

### 5.2 Backward Navigation

For efficient backward navigation, cache states at checkpoints:

```python
CHECKPOINT_INTERVAL = 10  # Cache every 10 moves

def get_state_with_caching(game_record: dict, move_index: int, cache: dict) -> GameState:
    """Get state at move index, using cached checkpoints."""
    # Find nearest checkpoint before target
    checkpoint_idx = (move_index // CHECKPOINT_INTERVAL) * CHECKPOINT_INTERVAL

    if checkpoint_idx in cache:
        state = cache[checkpoint_idx].copy()
        start_idx = checkpoint_idx
    else:
        state = deserialize_state(game_record["initial_state"])
        start_idx = 0

    # Apply moves from checkpoint to target
    for i in range(start_idx, move_index):
        move = deserialize_move(game_record["moves"][i]["numeric"])
        state = rules.apply_move(state, move)

    # Cache this state if it's a checkpoint
    if move_index % CHECKPOINT_INTERVAL == 0:
        cache[move_index] = state.copy()

    return state
```

### 5.3 Sandbox Integration API

```typescript
interface GameReplayController {
  // Load game from database/file
  loadGame(gameId: string): Promise<void>;

  // Navigation
  goToMove(moveIndex: number): void;
  nextMove(): void;
  previousMove(): void;
  goToStart(): void;
  goToEnd(): void;

  // State access
  getCurrentState(): GameState;
  getCurrentMoveIndex(): number;
  getTotalMoves(): number;

  // Move info
  getMoveAtIndex(index: number): MoveRecord;
  getMoveAlgebraic(index: number): string;

  // Metadata
  getGameMetadata(): GameMetadata;
  getOutcome(): GameOutcome;
}
```

## 6. Implementation Phases

### Phase 1: Core Storage

- [x] Define TypeScript/Python types for game records
  - Python: `ai-service/app/models/game_record.py`
  - TypeScript: `src/shared/types/gameRecord.ts`
- [x] Implement serialization/deserialization
  - Python: `GameRecord.to_jsonl_line()` / `GameRecord.from_jsonl_line()`
  - TypeScript: `gameRecordToJsonlLine()` / `jsonlLineToGameRecord()`
- [x] Create JSONL export format
  - Canonical per-game JSONL schema shared between TS and Python.
- [x] Add to self-play scripts
  - Python training generator: `ai-service/app/training/generate_data.py --game-records-jsonl` writes one `GameRecord` JSONL line per completed Descent self-play game.
  - DB-backed export: `scripts/export-game-records-jsonl.ts` streams `GameRecord` JSONL from Postgres `Game` rows via `GameRecordRepository.exportAsJsonl()`.

### Phase 2: Database Integration

- [x] Create database schema
  - Prisma `Game` model extended with `recordMetadata Json?`, `finalScore Json?`, and `outcome String?` fields (see `prisma/schema.prisma`).
- [x] Implement GameRecordRepository
  - `src/server/services/GameRecordRepository.ts` provides `saveGameRecord`, `getGameRecord`, `listGameRecords`, `exportAsJsonl`, `countGameRecords`, and `deleteOldRecords`.
- [x] Add game storage to online game source
  - `GameSession.finishGameWithResult()` now calls `gameRecordRepository.saveGameRecord(...)` so completed online games populate `finalState`, `finalScore`, `outcome`, and `recordMetadata`.
- [ ] Add game storage to all game sources (self-play, sandbox)
  - Python self-play / CMA-ES harnesses record into SQLite via `GameReplayDB` + `record_completed_game` but do not yet pipe directly into the Postgres-backed `Game` table.
- [ ] Migration for existing games
  - Existing rows are backfilled lazily: `GameRecordRepository` treats missing `recordMetadata` / `finalScore` as absent rather than eagerly migrating all historical games.

### Phase 3: Notation & Display

- [x] Implement algebraic notation generator
- [x] Implement notation parser
- [x] Add coordinate conversion utilities
- [ ] Create move list display component

### Phase 4: Replay System

- [ ] Implement state reconstruction
- [ ] Add checkpoint caching
- [ ] Create replay controller
- [ ] Integrate with sandbox UI

### Phase 5: Analysis Tools

- [ ] Position search/filter
- [ ] Move statistics
- [ ] Opening book extraction
- [ ] Critical position tagging

## 7. File Locations

```
ai-service/
  app/
    models/
      game_record.py       # Core types
    storage/
      game_repository.py   # Database access
      game_exporter.py     # JSONL export
  scripts/
    export_games.py        # Batch export utility

src/shared/
  types/
    gameRecord.ts          # TypeScript types
  notation/
    algebraic.ts           # Notation utilities
    coordinates.ts         # Coordinate conversion

src/client/
  components/
    GameReplay/
      ReplayController.tsx
      MoveList.tsx
      NavigationControls.tsx
```

## 8. Versioning

The `format_version` field allows for schema evolution:

- `1.0`: Initial spec (this document)
- Future versions must be backward-compatible for reading
- Include migration utilities for older formats
