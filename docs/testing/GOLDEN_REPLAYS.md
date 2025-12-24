# Golden Replay Tests

## Overview

Golden replay tests exercise curated games across the shared TypeScript engine and the Python rules engine. They replay recorded games and verify structural invariants at every move step on the TS side, and (for Python) use both shared JSON fixtures and GameReplayDB‑backed golden traces to enforce strict TS↔Python differential replay parity.

## Purpose

1. **Regression Prevention**: Detect unintended changes to game rules
2. **Cross-Engine Parity**: Verify TypeScript and Python engines behave identically
3. **Edge Case Coverage**: Exercise rare game situations (chain captures, territory splits, forced eliminations)
4. **Documentation**: Golden games serve as executable specifications

## Directory Structure

```
tests/
├── golden/
│   ├── goldenReplayHelpers.ts       # TS invariant checkers and utilities
│   └── goldenReplay.test.ts         # Jest test runner over JSON GameRecords
└── fixtures/
    └── golden-games/                # Golden game fixtures (GameRecord JSON / JSONL)

ai-service/tests/
├── golden/
│   └── test_golden_replay.py        # Python golden replay invariants over JSON fixtures
├── parity/
│   └── test_golden_replay_parity.py # TS↔Python differential replay over golden GameReplayDBs
└── fixtures/
    └── golden_games/                # Golden GameReplayDBs (.db) and/or JSON fixtures
```

## Structural Invariants

Each reconstructed move step is validated against six structural invariants (see `tests/golden/goldenReplayHelpers.ts` and `ai-service/tests/golden/test_golden_replay.py`).

### INV-BOARD-CONSISTENCY

- All stack keys in `board.stacks` map to positions that are valid for the given `boardType` and board size (via `BOARD_CONFIGS[boardType].size` and `isValidPosition`).
- All marker keys in `board.markers` map to valid board positions.
- Each stack has a non‑zero `stackHeight` (invalid empty stacks are rejected).

### INV-TURN-SEQUENCE

- `moveHistory.length` never decreases between consecutive reconstructed states.

### INV-PLAYER-RINGS

- For every player, `eliminatedRings` and `ringsInHand` are non‑negative.
- For every player, the total number of rings (`on‑board + eliminated + in hand`) never exceeds the starting `ringsPerPlayer` for that `boardType` (from `BOARD_CONFIGS[boardType].ringsPerPlayer`).

### INV-PHASE-VALID

- `currentPhase` is one of the canonical `GamePhase` values from `src/shared/types/game.ts`:
  - `ring_placement`, `movement`, `capture`, `chain_capture`, `line_processing`, `territory_processing`, `forced_elimination`, `game_over`.
- Higher‑level multi‑phase sequencing (movement → capture → chain_capture → line_processing → territory_processing → forced_elimination) is validated by contract vectors and snapshot parity tests rather than this structural check.

### INV-ACTIVE-PLAYER

- When `gameStatus === 'active'`, `currentPlayer` is a valid **0‑based** index into `players` (`0 <= currentPlayer < players.length`).

### INV-GAME-STATUS

- `gameStatus` uses the canonical `GameStatus` values from `src/shared/types/game.ts` (`waiting`, `active`, `finished`, `paused`, `abandoned`, `completed`).
- If `winner` is set on the final state, `gameStatus` must be `finished` or `completed`.
- Draws and certain aborted games are represented via `outcome === 'draw' | 'abandonment'` with `winner` omitted.

## Golden Game Fixture Formats

### TypeScript shared-engine fixtures (`tests/fixtures/golden-games/*.json[l]`)

TypeScript golden fixtures are JSON or JSONL files of `GameRecord` objects as defined in `src/shared/types/gameRecord.ts`. The type definition there is the Single Source of Truth; the example below is schematic:

```jsonc
{
  "id": "uuid",
  "boardType": "square8",
  "numPlayers": 2,
  "isRated": false,
  "players": [
    { "playerNumber": 0, "username": "P1", "playerType": "human" },
    { "playerNumber": 1, "username": "P2", "playerType": "ai" },
  ],
  "winner": 0,
  "outcome": "ring_elimination",
  "finalScore": {
    "ringsEliminated": { "0": 5, "1": 0 },
    "territorySpaces": { "0": 0, "1": 0 },
    "ringsRemaining": { "0": 0, "1": 5 },
  },
  "startedAt": "ISO timestamp",
  "endedAt": "ISO timestamp",
  "totalMoves": 42,
  "totalDurationMs": 300000,
  "moves": [
    {
      "moveNumber": 1,
      "player": 0,
      "type": "place_ring",
      "to": { "x": 3, "y": 3 },
      "thinkTimeMs": 500,
    },
  ],
  "metadata": {
    "recordVersion": "1.0.0",
    "createdAt": "ISO timestamp",
    "source": "self_play",
    "tags": ["category:chain_capture", "regression:issue-123"],
  },
}
```

#### Required fields (TS fixtures)

At minimum, TS golden fixtures must provide:

- Top‑level game configuration:
  - `id`
  - `boardType` (`"square8" | "square19" | "hex8" | "hexagonal"`)
  - `numPlayers`
  - `players[]` with `playerNumber`, `username`, `playerType`
- Result fields:
  - `winner` (0‑based index, optional for draws/abandonments)
  - `outcome` (`GameOutcome` union from `gameRecord.ts`)
  - `finalScore`
- Move history:
  - `moves[]` array of `MoveRecord` objects with `moveNumber`, `player`, `type`, and `thinkTimeMs` (plus `from` / `to` and other optional fields as needed for replay).
- Record metadata:
  - `metadata.recordVersion`, `metadata.createdAt`, `metadata.source`, `metadata.tags`.

Additional optional fields (`initialStateHash`, `finalStateHash`, `progressSnapshots`, etc.) are supported by the type and ignored by the golden replay helpers.

### Python golden DB fixtures (`ai-service/tests/fixtures/golden_games/*.db`)

Python parity tests (`ai-service/tests/parity/test_golden_replay_parity.py`) operate on SQLite `GameReplayDB` files promoted into `ai-service/tests/fixtures/golden_games/`. Each `.db` may contain multiple games; the parity test:

- Enumerates all `(db_path, game_id)` pairs.
- Replays each game via both the Python rules engine (`GameReplayDB` + `GameEngine`) and the TS replay script (`selfplay-db-ts-replay.ts`).
- Fails if any **differential replay divergences** are observed for a golden game.

The underlying schema and recording helpers are documented in:

- `docs/ENGINE_TOOLING_PARITY_RESEARCH_PLAN.md` (GameReplayDB overview and recording invariants).
- `ai-service/docs/GAME_RECORD_SPEC.md` (cross‑language GameRecord / GameReplayDB spec).

## Adding New Golden Games

### 1. Promote candidates into golden DBs (Python pipeline)

For large self‑play or CMA‑ES runs, start from GameReplayDBs and use the Python tools to triage and promote candidates:

- From `ai-service/`, list promising candidates:

  ```bash
  PYTHONPATH=. python scripts/find_golden_candidates.py \
    --db data/games/combined.db \
    --min-moves 40 \
    --output golden_candidates.json
  ```

- After choosing specific `(db_path, game_id)` pairs, copy them into golden fixture DBs:

  ```bash
  PYTHONPATH=. python scripts/extract_golden_games.py \
    --db data/games/combined.db \
    --game-id <id1> --game-id <id2> \
    --output tests/fixtures/golden_games/golden_line_territory.db
  ```

These `.db` files are consumed directly by `ai-service/tests/parity/test_golden_replay_parity.py`.

### 2. Export JSON GameRecord fixtures (TS shared engine)

To feed the TS shared‑engine golden tests from online games or from promoted golden DBs, export `GameRecord` JSON/JSONL and place it under `tests/fixtures/golden-games/`:

- From the project root, export records from Postgres:

  ```bash
  TS_NODE_PROJECT=tsconfig.server.json npx ts-node scripts/export-game-records-jsonl.ts \
    --output data/game_records.jsonl --board-type square8
  ```

- Alternatively, for ad‑hoc inspection in Node code, use `GameRecordRepository`:

  ```typescript
  import { gameRecordRepository } from 'src/server/services/GameRecordRepository';

  const record = await gameRecordRepository.getGameRecord(gameId);
  // Serialize `record` to JSON or JSONL under tests/fixtures/golden-games/
  ```

### 3. Add Metadata Tags

Tag games by category for coverage tracking:

- `category:chain_capture` - Games with chain capture sequences
- `category:territory_split` - Territory processing with splits
- `category:forced_elimination` - Forced elimination scenarios
- `category:victory_line` - Line victory games
- `category:victory_territory` - Territory victory games
- `category:swap_rule` - Games using swap rule
- `regression:issue-XXX` - Regression tests linked to issues

### 4. Place Fixture File

Save JSON/JSONL fixtures to `tests/fixtures/golden-games/<name>.json[l]`:

```bash
# Example naming convention:
# square8_2p_chain_capture_001.json
# hex19_4p_territory_split_001.json
```

### 5. Verify Tests Pass

```bash
# TypeScript
npm test -- tests/golden/goldenReplay.test.ts

# Python
cd ai-service && python -m pytest tests/golden/test_golden_replay.py -v
```

## Running Tests

### TypeScript (Jest)

```bash
# Run all golden replay tests
npm test -- tests/golden/

# Run with verbose output
npm test -- tests/golden/ --verbose

# Filter by pattern
npm test -- tests/golden/ -t "chain_capture"
```

### Python (pytest)

```bash
cd ai-service

# Run all golden replay tests
python -m pytest tests/golden/ -v

# Run with coverage
python -m pytest tests/golden/ --cov=app.rules

# Filter by marker
python -m pytest tests/golden/ -m "not slow"
```

## Coverage Goals

The test suite tracks coverage across:

| Category      | Target | Current Status | Description                                                              |
| ------------- | ------ | -------------- | ------------------------------------------------------------------------ |
| Board Types   | All    | ⚠️ Partial     | `square8` ✅, `square19` ✅, `hex8` ✅ (smoke), `hexagonal` ✅           |
| Player Counts | All    | ✅ Complete    | 2-player ✅, 3-player ✅, 4-player ✅                                    |
| Victory Types | All    | ⚠️ Partial     | `ring_elimination` ✅, `territory_control` ⚠️, `last_player_standing` ⚠️ |
| Edge Cases    | 80%+   | ⚠️ Unknown     | chain captures, territory splits, forced eliminations (need analysis)    |

### Current Golden Fixture Status (Dec 2025)

**Available candidates (from `find_golden_candidates.py`):** 29 total candidates

| Board Type | Players | Source DB             | Games | Finished Games        |
| ---------- | ------- | --------------------- | ----- | --------------------- |
| square19   | 2       | canonical_square19.db | 2     | 2 (with winners)      |
| square8    | 2       | selfplay.db           | 1     | 0                     |
| hexagonal  | 2       | golden_hexagonal.db   | 10    | 0 (max moves reached) |
| square8    | 3       | golden_3player.db     | 8     | 2 (with winners)      |
| square8    | 4       | golden_4player.db     | 8     | 0 (max moves reached) |

**Coverage status:**

- ✅ Hexagonal board games (10 games generated via minimal selfplay)
- ✅ Hex8 board games (smoke fixture in `tests/fixtures/golden-games/`)
- ✅ 3-player games (8 games, 2 with winners)
- ✅ 4-player games (8 games)
- ⚠️ Territory control victories (not yet verified)
- ⚠️ Last player standing victories (not yet verified)
- ❌ Pie rule / swap sides games

**Next steps:**

1. Export selected candidates to golden fixture format using `extract_golden_games.py`
2. Run games with higher max_moves or heuristic AI to get more decisive endings
3. Verify victory type coverage in promoted fixtures

## Troubleshooting

### Invariant Violation

If a test fails with an invariant violation:

1. Check the violation message for which invariant failed
2. Examine the move number and state at failure
3. Compare expected vs actual values
4. If the game rules changed intentionally, update the fixture

### Fixture Not Found

```
No golden game fixtures found
```

The test suite gracefully skips if no fixtures exist. Add fixtures to `tests/fixtures/golden-games/`.

### Cross-Engine Parity Failure

If TypeScript and Python produce different results:

1. Both engines should produce identical states at each move
2. Check for floating-point differences in territory calculations
3. Verify board coordinate systems match (x,y not row,col)
4. Review recent changes to either engine

## Integration with CI

Golden replay tests run as part of the standard test suite:

```yaml
# .github/workflows/ci.yml
- name: Run Golden Replay Tests
  run: |
    npm test -- tests/golden/
    cd ai-service && python -m pytest tests/golden/ -v
```

## Related Documentation

- [Game Record Types](../../src/shared/types/gameRecord.ts)
- [Engine Architecture](../../RULES_ENGINE_ARCHITECTURE.md)
- [Test Layers](../../tests/TEST_LAYERS.md)
- [Rules Scenario Matrix](../rules/RULES_SCENARIO_MATRIX.md)
- [Invariants & Parity Framework](../rules/INVARIANTS_AND_PARITY_FRAMEWORK.md)
