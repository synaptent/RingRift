# GameReplayDB Sandbox Integration Plan

> **Doc Status:** Draft (2025-12-02, partially implemented)
>
> This document outlines the plan to integrate the Python `GameReplayDB` (SQLite) with the TypeScript sandbox UI for game replay, analysis, and future training data visualization.
> Since this plan was written, the FastAPI replay API (`ai-service/app/routes/replay.py`), TypeScript `ReplayService` (`src/client/services/ReplayService.ts`), and sandbox replay panel (`ReplayPanel` and related components under `src/client/components/ReplayPanel/`) have been implemented; remaining work is primarily around schema evolution, additional analytics, and UX polish.
>
> **See Also:** [Unified Self-Play Game Recording Plan](/.claude/plans/memoized-cuddling-abelson.md) — A comprehensive plan (Track 11 in TODO.md) to record ALL self-play games across the codebase (CMA-ES optimization, tournaments, soak tests) to the GameReplayDB for sandbox replay, neural network training, and evaluation pool generation.

## Overview

### Current State

| Component                     | Status      | Location                                                                            |
| ----------------------------- | ----------- | ----------------------------------------------------------------------------------- |
| GameReplayDB (Python)         | Implemented | `ai-service/app/db/game_replay.py`                                                  |
| Self-play storage hook        | Implemented | `ai-service/scripts/run_self_play_soak.py`                                          |
| Schema v1/v2                  | Implemented | 6+ tables with time/control and engine eval fields (see `game_replay.py`)           |
| REST API for replay           | Implemented | `ai-service/app/routes/replay.py` (`/api/replay/*` endpoints)                       |
| Sandbox replay panel          | Implemented | `src/client/components/ReplayPanel/ReplayPanel.tsx` (used by `SandboxGameHost.tsx`) |
| TypeScript client integration | Implemented | `src/client/services/ReplayService.ts`, `src/client/types/replay.ts`                |

### Goals

1. **Sandbox Replay Panel**: Add a collapsible panel to the sandbox UI that exposes the game database (**implemented** via `ReplayPanel` in the `/sandbox` sidebar).
2. **Game Browser**: Enable browsing, filtering, and selecting games from the database (**implemented** via `GameFilters`/`GameList` and replay hooks).
3. **Playback Controls**: Step forward/backward through turn actions, or play like a movie at configurable speed (**implemented** via `PlaybackControls` and `useReplayPlayback` / `useReplayAnimation`).
4. **Unified Storage**: All AI self-play games (Python soak scripts, sandbox AI vs AI) stored in same format (Python self‑play paths implemented; sandbox AI‑vs‑AI storage wired via `ReplayService.storeGame` and `/api/replay/games` but optional in UI).
5. **Schema Migration**: Upgrade existing database entries when schema evolves (**implemented** in `GameReplayDB` with migration tests; future schema bumps follow the same pattern).
6. **Future-proof Schema**: Add fields for engine evaluation, principal variation, and time control (**partially implemented** in v2 schema; future extensions tracked in `ai-service/docs/GAME_REPLAY_DATABASE_SPEC.md`).

### Key Features

The sandbox replay panel will provide:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  📂 Game Database                                            [Collapse] │
├─────────────────────────────────────────────────────────────────────────┤
│  Filters: [Board ▼] [Players ▼] [Outcome ▼] [Source ▼]    [🔍 Search]  │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ Game ID      Board    Players  Winner  Moves  Date              │  │
│  │ abc123...    8×8      2        P1      47     2025-12-01 14:30  │  │
│  │ def456...    Hex      4        P3      89     2025-12-01 14:15  │  │
│  │ ghi789...    19×19    2        P2      156    2025-12-01 13:45  │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                              Page 1 of 42  [< >]       │
├─────────────────────────────────────────────────────────────────────────┤
│  ▶ Now Playing: abc123... (Move 23/47)                                 │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  [⏮] [◀] [▶ Play] [▶▶] [⏭]     Speed: [1x ▼]     [Jump to: ___] │  │
│  │  ═══════════════●══════════════════════════════════════════════  │  │
│  │  Move 23: P1 movement (3,4)→(5,4)              Eval: +0.32       │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Sandbox Replay vs. Self-Play Browser (Option A vs. Option B)

As of December 2025, there are **two complementary replay paths** in `/sandbox`:

- **Option B – Local self-play replay (always available when a self-play DB exists):**
  - The **Self-Play Browser** (`SelfPlayBrowser.tsx`) loads a game from a raw self-play SQLite DB.
  - `SandboxGameHost` seeds `ClientSandboxEngine` from the recorded initial state and replays the full canonical move list locally.
  - This path powers the history slider under the board even when the Python replay service is offline or pointing at a different DB.

- **Option A – GameReplayDB-backed replay panel (requires matching DB path):**
  - The **ReplayPanel** uses the Python `GameReplayDB` via `/api/replay/*` to drive the board from server-side snapshots.
  - When you load a self-play game via the Self-Play Browser, `SandboxGameHost` forwards the `gameId` into `ReplayPanel` as `requestedGameId`.
  - If `GAME_REPLAY_DB_PATH` points at a replay DB that actually contains that `gameId`, ReplayPanel auto-loads the game and enters **Replay Mode**.

#### Configuring `GAME_REPLAY_DB_PATH` for full Option A parity

To ensure the ReplayPanel can consume the **same games** you browse via the Self-Play Browser:

1. **Choose the target DB file** used by your self-play runs, for example:
   - `data/games/selfplay.db`
   - `ai-service/data/games/selfplay.db`
2. **Point the AI service at that file** by setting `GAME_REPLAY_DB_PATH` in the AI service environment:
   - See [`docs/ENVIRONMENT_VARIABLES.md`](../ENVIRONMENT_VARIABLES.md) for the full variable reference.
3. **Restart the AI service** after changing the env var so `/api/replay/*` sees the new DB.

When these paths line up:

- Self-Play Browser → "Load" provides:
  - Local engine replay (Option B) for the history slider, and
  - ReplayPanel auto-loading the same `gameId` from `GameReplayDB` (Option A).

If the configured replay DB does **not** contain the requested `gameId`, the ReplayPanel:

- Logs a console error for debugging, and
- Shows a small banner in browse mode:

> `Requested game not found in replay DB (check GAME_REPLAY_DB_PATH). Using local sandbox replay instead.`

This banner is the primary UX hint that Option B is active, but Option A could not engage due to a DB mismatch.

---

## Part 1: Schema Extensions

### Current Schema Gaps

The existing `GameReplayDB` schema (`ai-service/docs/GAME_REPLAY_DATABASE_SPEC.md`) is missing several fields needed for comprehensive game analysis:

| Missing Field       | Table        | Purpose                                      |
| ------------------- | ------------ | -------------------------------------------- |
| `time_remaining_ms` | `game_moves` | Clock state after each move                  |
| `time_increment_ms` | `games`      | Time control increment setting               |
| `initial_time_ms`   | `games`      | Initial time per player                      |
| `engine_eval`       | `game_moves` | Heuristic/neural evaluation score            |
| `engine_depth`      | `game_moves` | Search depth (if applicable)                 |
| `engine_pv`         | `game_moves` | Principal variation (predicted future moves) |
| `engine_nodes`      | `game_moves` | Nodes searched (for MCTS/minimax)            |

### Proposed Schema Extensions

#### 1. Time Control Fields

Add to `games` table:

```sql
ALTER TABLE games ADD COLUMN time_control_type TEXT;  -- 'none', 'fischer', 'simple'
ALTER TABLE games ADD COLUMN initial_time_ms INTEGER;
ALTER TABLE games ADD COLUMN time_increment_ms INTEGER;
```

Add to `game_moves` table:

```sql
ALTER TABLE game_moves ADD COLUMN time_remaining_ms INTEGER;  -- Clock after move
```

#### 2. Engine Evaluation Fields

Add to `game_moves` table:

```sql
ALTER TABLE game_moves ADD COLUMN engine_eval REAL;           -- Evaluation score (centipawns-equivalent or win%)
ALTER TABLE game_moves ADD COLUMN engine_eval_type TEXT;      -- 'heuristic', 'neural', 'mcts_winrate'
ALTER TABLE game_moves ADD COLUMN engine_depth INTEGER;       -- Search depth
ALTER TABLE game_moves ADD COLUMN engine_nodes INTEGER;       -- Nodes searched
ALTER TABLE game_moves ADD COLUMN engine_pv TEXT;             -- JSON array of predicted future moves (PV line)
ALTER TABLE game_moves ADD COLUMN engine_time_ms INTEGER;     -- Time spent computing this move
```

#### 3. Schema Version Bump

Update `SCHEMA_VERSION` from 1 to 2. Add migration logic in `GameReplayDB.__init__()`:

```python
def _migrate_schema(self, from_version: int) -> None:
    """Apply schema migrations from from_version to current."""
    if from_version < 2:
        # Add time control and engine evaluation columns
        self._execute_migration_v2()
```

### Multi-Player Support

The current schema already supports 3-4 players via:

- `games.num_players` field
- `game_players` table with `player_number` 1-4
- `game_moves.player` field

No changes needed for multi-player support.

---

## Part 2: REST API Design

### API Endpoints

The sandbox UI will consume game replays via a REST API served by the Python AI service.

#### Base URL: `/api/replay`

| Method | Endpoint                            | Description             |
| ------ | ----------------------------------- | ----------------------- |
| GET    | `/api/replay/games`                 | List games with filters |
| GET    | `/api/replay/games/:gameId`         | Get game metadata       |
| GET    | `/api/replay/games/:gameId/state`   | Get state at move N     |
| GET    | `/api/replay/games/:gameId/moves`   | Get moves in range      |
| GET    | `/api/replay/games/:gameId/choices` | Get choices for a move  |
| GET    | `/api/replay/stats`                 | Get database statistics |

#### Endpoint Details

##### `GET /api/replay/games`

Query games by metadata filters.

**Query Parameters:**

```typescript
interface GameQueryParams {
  board_type?: 'square8' | 'square19' | 'hexagonal';
  num_players?: 2 | 3 | 4;
  winner?: number;
  termination_reason?: 'ring_elimination' | 'territory' | 'last_player_standing' | 'stalemate';
  source?: 'self_play' | 'human' | 'tournament' | 'training';
  min_moves?: number;
  max_moves?: number;
  limit?: number; // default 20, max 100
  offset?: number;
}
```

**Response:**

```typescript
interface GameListResponse {
  games: GameMetadata[];
  total: number;
  hasMore: boolean;
}

interface GameMetadata {
  gameId: string;
  boardType: BoardType;
  numPlayers: number;
  winner: number | null;
  terminationReason: string;
  totalMoves: number;
  totalTurns: number;
  createdAt: string;
  completedAt: string;
  durationMs: number;
  source: string;
  players: PlayerMetadata[];
}

interface PlayerMetadata {
  playerNumber: number;
  playerType: 'ai' | 'human';
  aiType?: string;
  aiDifficulty?: number;
  finalEliminatedRings: number;
  finalTerritorySpaces: number;
}
```

##### `GET /api/replay/games/:gameId/state`

Reconstruct game state at a specific move.

**Query Parameters:**

```typescript
interface StateQueryParams {
  moveNumber: number; // 0 = initial state, N = after move N
}
```

**Response:**

```typescript
interface ReplayStateResponse {
  gameState: SerializedGameState; // Full GameState JSON
  moveNumber: number;
  totalMoves: number;
  availableChoices?: Choice[]; // Choices available at this position
  engineEval?: number; // Evaluation at this position
  enginePV?: string[]; // Predicted continuation
}
```

##### `GET /api/replay/games/:gameId/moves`

Get move history in a range.

**Query Parameters:**

```typescript
interface MovesQueryParams {
  start?: number; // default 0
  end?: number; // default all
}
```

**Response:**

```typescript
interface MovesResponse {
  moves: MoveRecord[];
  hasMore: boolean;
}

interface MoveRecord {
  moveNumber: number;
  turnNumber: number;
  player: number;
  phase: string;
  moveType: string;
  move: SerializedMove;
  timestamp: string;
  thinkTimeMs: number;
  timeRemainingMs?: number;
  engineEval?: number;
  engineDepth?: number;
  enginePV?: string[];
}
```

### FastAPI Implementation

Add routes to `ai-service/app/main.py`:

```python
from fastapi import APIRouter, Query, HTTPException
from app.db.game_replay import GameReplayDB

replay_router = APIRouter(prefix="/api/replay", tags=["replay"])

@replay_router.get("/games")
async def list_games(
    board_type: Optional[str] = None,
    num_players: Optional[int] = None,
    winner: Optional[int] = None,
    termination_reason: Optional[str] = None,
    source: Optional[str] = None,
    min_moves: Optional[int] = None,
    max_moves: Optional[int] = None,
    limit: int = Query(default=20, le=100),
    offset: int = 0,
):
    """List games matching filters."""
    db = get_replay_db()
    games = db.query_games(
        board_type=board_type,
        num_players=num_players,
        winner=winner,
        termination_reason=termination_reason,
        source=source,
        min_moves=min_moves,
        max_moves=max_moves,
        limit=limit + 1,  # Fetch one extra to check hasMore
        offset=offset,
    )
    has_more = len(games) > limit
    return {
        "games": games[:limit],
        "total": db.get_game_count(**filters),
        "hasMore": has_more,
    }

@replay_router.get("/games/{game_id}/state")
async def get_state_at_move(game_id: str, move_number: int = 0):
    """Get reconstructed game state at a specific move."""
    db = get_replay_db()
    state = db.get_state_at_move(game_id, move_number)
    if state is None:
        raise HTTPException(404, f"Game {game_id} not found or invalid move number")
    return {
        "gameState": state.model_dump(),
        "moveNumber": move_number,
        "totalMoves": db.get_game_metadata(game_id)["total_moves"],
    }
```

---

## Part 3: TypeScript Client Integration

### ReplayService

Create a TypeScript service to fetch replay data from the AI service:

**Location:** `src/client/services/ReplayService.ts`

```typescript
import type { GameState, BoardType } from '../../shared/types/game';

export interface GameMetadata {
  gameId: string;
  boardType: BoardType;
  numPlayers: number;
  winner: number | null;
  terminationReason: string;
  totalMoves: number;
  createdAt: string;
  players: PlayerMetadata[];
}

export interface ReplayState {
  gameState: GameState;
  moveNumber: number;
  totalMoves: number;
  engineEval?: number;
  enginePV?: string[];
}

export class ReplayService {
  private baseUrl: string;

  constructor(aiServiceUrl: string) {
    this.baseUrl = `${aiServiceUrl}/api/replay`;
  }

  async listGames(filters: GameQueryParams): Promise<GameListResponse> {
    const params = new URLSearchParams();
    Object.entries(filters).forEach(([k, v]) => {
      if (v !== undefined) params.set(k, String(v));
    });
    const res = await fetch(`${this.baseUrl}/games?${params}`);
    if (!res.ok) throw new Error(`Failed to list games: ${res.status}`);
    return res.json();
  }

  async getStateAtMove(gameId: string, moveNumber: number): Promise<ReplayState> {
    const res = await fetch(`${this.baseUrl}/games/${gameId}/state?moveNumber=${moveNumber}`);
    if (!res.ok) throw new Error(`Failed to get state: ${res.status}`);
    return res.json();
  }

  async getMoves(gameId: string, start = 0, end?: number): Promise<MovesResponse> {
    const params = new URLSearchParams({ start: String(start) });
    if (end !== undefined) params.set('end', String(end));
    const res = await fetch(`${this.baseUrl}/games/${gameId}/moves?${params}`);
    if (!res.ok) throw new Error(`Failed to get moves: ${res.status}`);
    return res.json();
  }
}
```

### Sandbox Replay UI Components

#### ReplayBrowser Component

A component to browse and select games for replay:

**Location:** `src/client/components/ReplayBrowser.tsx`

```typescript
interface ReplayBrowserProps {
  onSelectGame: (gameId: string) => void;
}

export function ReplayBrowser({ onSelectGame }: ReplayBrowserProps) {
  // Filter controls (board type, player count, outcome)
  // Paginated game list
  // Click to select game for replay
}
```

#### ReplayControls Component

Controls for stepping through a replay:

**Location:** `src/client/components/ReplayControls.tsx`

```typescript
interface ReplayControlsProps {
  currentMove: number;
  totalMoves: number;
  isPlaying: boolean;
  playbackSpeed: number;
  onStepForward: () => void;
  onStepBackward: () => void;
  onJumpToMove: (n: number) => void;
  onTogglePlay: () => void;
  onSetSpeed: (speed: number) => void;
}
```

#### ReplayHost Component

Host component that manages replay state:

**Location:** `src/client/pages/ReplayHost.tsx`

```typescript
export function ReplayHost({ gameId }: { gameId: string }) {
  const [currentMove, setCurrentMove] = useState(0);
  const [gameState, setGameState] = useState<GameState | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const replayService = useReplayService();

  // Load state when currentMove changes
  useEffect(() => {
    replayService.getStateAtMove(gameId, currentMove).then(({ gameState }) => {
      setGameState(deserializeGameState(gameState));
    });
  }, [gameId, currentMove]);

  // Render BoardView with gameState
  // Render ReplayControls
  // Render move list / evaluation graph
}
```

### Integration with Existing Sandbox

The replay panel integrates directly into `SandboxGameHost.tsx` as a collapsible side panel:

#### Panel Placement

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ SandboxGameHost                                                             │
├──────────────────────────────────┬──────────────────────────────────────────┤
│                                  │  Replay Panel (collapsible)              │
│                                  │  ┌────────────────────────────────────┐  │
│       BoardView                  │  │ [📂 Game Database]        [−]     │  │
│                                  │  │ ─────────────────────────────────  │  │
│                                  │  │ Filters...                        │  │
│                                  │  │ Game list...                      │  │
│                                  │  │ ─────────────────────────────────  │  │
│                                  │  │ Playback controls...              │  │
│                                  │  └────────────────────────────────────┘  │
│                                  │                                          │
│                                  │  Players panel                           │
│                                  │  Event log                               │
│                                  │  Phase guide                             │
├──────────────────────────────────┴──────────────────────────────────────────┤
```

#### Replay Mode State Machine

```
┌──────────────┐    Select game    ┌──────────────┐
│   BROWSING   │ ───────────────▶  │   LOADING    │
│              │                   │              │
└──────────────┘                   └──────────────┘
       ▲                                  │
       │                                  │ Load complete
       │ Close replay                     ▼
       │                           ┌──────────────┐
       └────────────────────────── │   PLAYING    │
                                   │              │
                                   └──────────────┘
                                          │
                                          │ "Fork from here"
                                          ▼
                                   ┌──────────────┐
                                   │  FORKED_GAME │
                                   │ (interactive)│
                                   └──────────────┘
```

#### Component Structure

```typescript
// New components for replay panel
src/client/components/
├── ReplayPanel/
│   ├── ReplayPanel.tsx           // Main container with collapse
│   ├── GameBrowser.tsx           // Filter bar + game list
│   ├── GameFilters.tsx           // Board type, players, outcome dropdowns
│   ├── GameList.tsx              // Paginated table of games
│   ├── PlaybackControls.tsx      // Transport + speed + scrubber
│   └── MoveInfo.tsx              // Current move details + eval
```

#### Key Interactions

1. **Browse Games**: Panel shows filterable list from database
2. **Select Game**: Click game row → load into replay mode
3. **Step Through**: Use transport controls or keyboard (←/→/Space)
4. **Movie Mode**: Play button auto-advances at selected speed
5. **Jump to Move**: Click scrubber or type move number
6. **Fork Game**: "Play from here" creates interactive game from current position
7. **Close Replay**: Return to normal sandbox mode

#### Keyboard Shortcuts (Replay Mode)

| Key          | Action                          |
| ------------ | ------------------------------- |
| `←` / `h`    | Step backward                   |
| `→` / `l`    | Step forward                    |
| `Space`      | Toggle play/pause               |
| `Home` / `0` | Jump to start                   |
| `End` / `$`  | Jump to end                     |
| `[`          | Decrease speed                  |
| `]`          | Increase speed                  |
| `f`          | Fork game from current position |
| `Escape`     | Close replay mode               |

---

## Part 4: Sandbox AI vs AI Storage

Currently, sandbox AI vs AI games are NOT stored to the GameReplayDB. This section outlines how to enable that.

### Option A: Post-hoc Storage via API

After a sandbox AI vs AI game completes, send the game record to the Python service:

```typescript
// In ClientSandboxEngine.ts or SandboxGameHost.tsx
async function onGameComplete(finalState: GameState) {
  if (config.recordGames && isAIvsAI) {
    await replayService.storeGame({
      initialState: savedInitialState,
      finalState,
      moves: moveHistory,
      choices: choiceHistory,
      metadata: { source: 'sandbox_ai_vs_ai', ... }
    });
  }
}
```

**New API endpoint:**

```
POST /api/replay/games
```

### Option B: Direct SQLite Access (Browser)

Use `sql.js` (SQLite compiled to WebAssembly) to write directly to an IndexedDB-backed SQLite database in the browser:

- Pro: Works offline, no server dependency
- Con: Games stored only locally, not shared with Python training

### Recommendation

Use **Option A** (post-hoc storage) for consistency with the Python GameReplayDB. This ensures all games are in the same format and available for training data extraction.

---

## Part 5: Schema Migration & Upgrading Existing Entries

### Migration Strategy

When the schema version changes, the `GameReplayDB` must:

1. **Detect version mismatch** on database open
2. **Apply incremental migrations** from current to target version
3. **Backfill nullable columns** with sensible defaults
4. **Re-derive computable fields** where possible

### Migration Framework

```python
# ai-service/app/db/game_replay.py

SCHEMA_VERSION = 2  # Increment when schema changes

class GameReplayDB:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Create or migrate schema to current version."""
        current_version = self._get_schema_version()

        if current_version == 0:
            # Fresh database
            self._create_schema_v2()
        elif current_version < SCHEMA_VERSION:
            # Migrate from current to target
            self._migrate_schema(current_version, SCHEMA_VERSION)
        elif current_version > SCHEMA_VERSION:
            raise ValueError(
                f"Database schema version {current_version} is newer than "
                f"supported version {SCHEMA_VERSION}. Please upgrade."
            )

    def _migrate_schema(self, from_version: int, to_version: int) -> None:
        """Apply incremental migrations."""
        for version in range(from_version + 1, to_version + 1):
            migration_method = getattr(self, f"_migrate_to_v{version}", None)
            if migration_method is None:
                raise ValueError(f"Missing migration method for v{version}")

            logger.info(f"Migrating database schema from v{version-1} to v{version}")
            migration_method()
            self._set_schema_version(version)

        logger.info(f"Schema migration complete: v{from_version} → v{to_version}")

    def _migrate_to_v2(self) -> None:
        """Add time control and engine evaluation columns."""
        cursor = self.conn.cursor()

        # Add columns to games table
        cursor.execute("ALTER TABLE games ADD COLUMN time_control_type TEXT DEFAULT 'none'")
        cursor.execute("ALTER TABLE games ADD COLUMN initial_time_ms INTEGER DEFAULT NULL")
        cursor.execute("ALTER TABLE games ADD COLUMN time_increment_ms INTEGER DEFAULT NULL")

        # Add columns to game_moves table
        cursor.execute("ALTER TABLE game_moves ADD COLUMN time_remaining_ms INTEGER DEFAULT NULL")
        cursor.execute("ALTER TABLE game_moves ADD COLUMN engine_eval REAL DEFAULT NULL")
        cursor.execute("ALTER TABLE game_moves ADD COLUMN engine_eval_type TEXT DEFAULT NULL")
        cursor.execute("ALTER TABLE game_moves ADD COLUMN engine_depth INTEGER DEFAULT NULL")
        cursor.execute("ALTER TABLE game_moves ADD COLUMN engine_nodes INTEGER DEFAULT NULL")
        cursor.execute("ALTER TABLE game_moves ADD COLUMN engine_pv TEXT DEFAULT NULL")
        cursor.execute("ALTER TABLE game_moves ADD COLUMN engine_time_ms INTEGER DEFAULT NULL")

        self.conn.commit()
        logger.info("Added time control and engine evaluation columns")
```

### Backfilling Existing Data

For games stored before schema v2, optional fields will be NULL. We can optionally run a backfill process to compute missing data:

```python
def backfill_engine_evaluations(
    db: GameReplayDB,
    evaluator: Callable[[GameState], float],
    batch_size: int = 100,
) -> None:
    """
    Backfill engine_eval for moves that don't have them.

    This is CPU-intensive and should be run as a background job.
    """
    cursor = db.conn.cursor()

    # Find games without evaluations
    cursor.execute("""
        SELECT DISTINCT gm.game_id
        FROM game_moves gm
        WHERE gm.engine_eval IS NULL
        LIMIT ?
    """, (batch_size,))

    game_ids = [row[0] for row in cursor.fetchall()]

    for game_id in game_ids:
        logger.info(f"Backfilling evaluations for game {game_id}")

        # Get initial state
        state = db.get_initial_state(game_id)
        if state is None:
            continue

        # Get all moves
        moves = db.get_moves(game_id)

        for i, move in enumerate(moves):
            # Apply move to get state
            state = apply_move(state, move)

            # Compute evaluation
            eval_score = evaluator(state)

            # Update database
            cursor.execute("""
                UPDATE game_moves
                SET engine_eval = ?, engine_eval_type = 'heuristic'
                WHERE game_id = ? AND move_number = ?
            """, (eval_score, game_id, i))

        db.conn.commit()
```

### CLI for Migration and Backfill

```bash
# Run schema migration (automatic on DB open, but can be triggered manually)
python -m ai-service.scripts.migrate_game_db --db data/games/selfplay.db

# Backfill evaluations for existing games
python -m ai-service.scripts.backfill_evaluations \
    --db data/games/selfplay.db \
    --batch-size 100 \
    --eval-type heuristic

# Verify migration status
python -m ai-service.scripts.migrate_game_db --db data/games/selfplay.db --status
```

### Migration Safety

1. **Backup first**: Always backup database before migration
2. **Atomic transactions**: Each migration step is a transaction
3. **Version tracking**: Schema version stored in metadata table
4. **Rollback support**: Keep migration reversal scripts for emergencies
5. **Dry-run mode**: Preview changes before applying

```python
def migrate_with_backup(db_path: str, dry_run: bool = False) -> None:
    """Safe migration with automatic backup."""
    backup_path = f"{db_path}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if not dry_run:
        shutil.copy2(db_path, backup_path)
        logger.info(f"Created backup at {backup_path}")

    db = GameReplayDB(db_path)

    if dry_run:
        logger.info("Dry run complete - no changes made")
    else:
        logger.info("Migration complete")
```

---

## Part 6: Implementation Phases

### Phase 1: Schema Migration & Backend API (2-3 days)

**Goal**: Extend schema and expose REST API for replay data

| Task | File(s)                               | Description                                              |
| ---- | ------------------------------------- | -------------------------------------------------------- |
| 1.1  | `app/db/game_replay.py`               | Add migration framework with version tracking            |
| 1.2  | `app/db/game_replay.py`               | Implement `_migrate_to_v2()` for new columns             |
| 1.3  | `app/db/game_replay.py`               | Update `store_game()`, `add_move()` to accept new fields |
| 1.4  | `scripts/migrate_game_db.py`          | CLI tool for manual migration & status check             |
| 1.5  | `scripts/backfill_evaluations.py`     | Backfill script for existing games                       |
| 1.6  | `app/routes/replay.py`                | Create FastAPI router with endpoints                     |
| 1.7  | `app/main.py`                         | Mount replay router, configure CORS                      |
| 1.8  | `tests/test_game_replay_migration.py` | Unit tests for migration                                 |

**Deliverables**:

- Schema auto-migrates on open
- `/api/replay/games`, `/api/replay/games/:id/state`, `/api/replay/games/:id/moves` working
- Existing databases upgraded transparently

### Phase 2: TypeScript Client & Types (1-2 days)

**Goal**: Create typed client for consuming replay API

| Task | File(s)                                | Description                                |
| ---- | -------------------------------------- | ------------------------------------------ |
| 2.1  | `src/client/services/ReplayService.ts` | HTTP client class                          |
| 2.2  | `src/client/types/replay.ts`           | TypeScript interfaces for API responses    |
| 2.3  | `src/client/hooks/useReplayService.ts` | React hook for service access              |
| 2.4  | `src/client/hooks/useReplayState.ts`   | State management for replay mode           |
| 2.5  | Environment config                     | Add `VITE_AI_SERVICE_URL` for API base URL |

**Deliverables**:

- Fully typed API client
- React hooks for replay functionality
- Error handling and loading states

### Phase 3: Sandbox Replay Panel UI (3-4 days)

**Goal**: Build the collapsible replay panel in sandbox

| Task | File(s)                                       | Description                                |
| ---- | --------------------------------------------- | ------------------------------------------ |
| 3.1  | `components/ReplayPanel/ReplayPanel.tsx`      | Main panel container with collapse         |
| 3.2  | `components/ReplayPanel/GameFilters.tsx`      | Filter dropdowns (board, players, outcome) |
| 3.3  | `components/ReplayPanel/GameList.tsx`         | Paginated game table with selection        |
| 3.4  | `components/ReplayPanel/PlaybackControls.tsx` | Transport buttons, speed, scrubber         |
| 3.5  | `components/ReplayPanel/MoveInfo.tsx`         | Current move details display               |
| 3.6  | `pages/SandboxGameHost.tsx`                   | Integrate ReplayPanel in sidebar           |
| 3.7  | `hooks/useReplayPlayback.ts`                  | Playback logic (step, auto-play, speed)    |
| 3.8  | Keyboard handlers                             | Arrow keys, space, number keys             |

**Deliverables**:

- Collapsible panel in sandbox sidebar
- Browse and filter games
- Select game to enter replay mode

### Phase 4: Playback & Interaction (2-3 days)

**Goal**: Full playback controls and keyboard navigation

| Task | File(s)                                    | Description                               |
| ---- | ------------------------------------------ | ----------------------------------------- |
| 4.1  | `hooks/useReplayPlayback.ts`               | Auto-play timer with configurable speed   |
| 4.2  | `hooks/useReplayKeyboard.ts`               | Keyboard shortcut handler                 |
| 4.3  | `components/ReplayPanel/Scrubber.tsx`      | Clickable timeline/progress bar           |
| 4.4  | `components/ReplayPanel/SpeedSelector.tsx` | Speed dropdown (0.5x, 1x, 2x, 4x)         |
| 4.5  | Move animation                             | Animate pieces during playback            |
| 4.6  | State caching                              | Cache fetched states for smooth scrubbing |

**Deliverables**:

- Step forward/backward with animation
- Auto-play at configurable speed
- Keyboard navigation
- Smooth scrubbing

### Phase 5: Game Storage & Fork (2 days)

**Goal**: Store sandbox games and fork from replay

| Task | File(s)                  | Description                            |
| ---- | ------------------------ | -------------------------------------- |
| 5.1  | `app/routes/replay.py`   | `POST /api/replay/games` endpoint      |
| 5.2  | `ClientSandboxEngine.ts` | Hook to store completed AI vs AI games |
| 5.3  | `SandboxGameHost.tsx`    | Toggle for "Record this game"          |
| 5.4  | `SandboxGameHost.tsx`    | "Fork from here" button in replay mode |
| 5.5  | `ClientSandboxEngine.ts` | `initFromReplayPosition()` method      |

**Deliverables**:

- Sandbox AI games automatically stored
- Fork replay into interactive game
- Round-trip: play → store → replay → fork

### Phase 6: Polish & Future Features (Ongoing)

| Feature            | Priority | Description                            |
| ------------------ | -------- | -------------------------------------- |
| Evaluation graph   | P2       | Line chart of engine_eval over moves   |
| Move annotations   | P2       | "Blunder", "Good move" markers         |
| Export games       | P2       | Download as PGN-like format            |
| Critical positions | P3       | Auto-detect interesting positions      |
| Compare games      | P3       | Side-by-side replay of two games       |
| Training export    | P3       | Export labeled data for neural network |

---

## Implementation Checklist

```
Phase 1: Schema & API
[ ] Migration framework
[ ] Schema v2 columns
[ ] Backfill script
[ ] REST endpoints
[ ] CORS config
[ ] API tests

Phase 2: TypeScript Client
[ ] ReplayService class
[ ] Type definitions
[ ] useReplayService hook
[ ] useReplayState hook
[ ] Environment config

Phase 3: Replay Panel UI
[ ] ReplayPanel container
[ ] GameFilters component
[ ] GameList component
[ ] PlaybackControls component
[ ] MoveInfo component
[ ] Sandbox integration

Phase 4: Playback
[ ] Auto-play timer
[ ] Keyboard shortcuts
[ ] Scrubber/timeline
[ ] Speed selector
[ ] Move animation
[ ] State caching

Phase 5: Storage & Fork
[ ] POST endpoint
[ ] Game recording hook
[ ] Recording toggle
[ ] Fork button
[ ] initFromReplayPosition
```

---

## Dependencies

### Python Side

- FastAPI router for replay endpoints
- Schema migration in `GameReplayDB`
- CORS configuration for sandbox access

### TypeScript Side

- `AI_SERVICE_URL` environment variable
- Fetch-based API client
- New components and pages

### Existing Code Affected

- `ai-service/app/db/game_replay.py` (schema changes)
- `ai-service/app/main.py` (add router)
- `src/client/pages/SandboxGameHost.tsx` (add replay mode)
- `src/client/sandbox/ClientSandboxEngine.ts` (game storage hook)

---

## Open Questions

1. **Authentication**: Should replay API require auth? (Probably not for local dev)
2. **CORS**: Need to configure AI service CORS for sandbox origin
3. **Large DBs**: How to handle databases with 100k+ games? (Pagination, lazy loading)
4. **Engine PV format**: How to serialize predicted future moves? (Algebraic notation or JSON Move objects)
5. **Time zones**: Store all timestamps in UTC?

---

## References

- [GAME_REPLAY_DATABASE_SPEC.md](ai-service/docs/GAME_REPLAY_DATABASE_SPEC.md) - Original spec
- [GameReplayDB Implementation](ai-service/app/db/game_replay.py) - Current code
- [Client Architecture](src/client/ARCHITECTURE.md) - Frontend patterns
- [TODO.md Track 5](TODO.md) - Persistence & Replays track
