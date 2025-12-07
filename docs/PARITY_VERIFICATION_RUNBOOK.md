# Parity Verification Runbook

This runbook documents the TS↔Python parity verification system for RingRift. It covers running parity checks, interpreting results, debugging divergences, and maintaining canonical training data.

---

## 1. Overview

### 1.1 Purpose of TS/Python Parity

RingRift's AI training pipeline requires **identical engine behavior** between:

- **TypeScript (TS)**: The canonical rules engine used in the web client and server
- **Python (Py)**: The rules implementation used for AI training, self-play, and replay

When the engines diverge, training data becomes unreliable because:
- State hashes won't match between recorded positions and replayed positions
- AI move evaluations may be based on incorrect board states
- Self-play games may record invalid or impossible game trajectories

### 1.2 What "Parity" Means

Parity means: **same inputs → same outputs**

For any sequence of moves applied to an initial state:
- Both engines produce the same `currentPlayer` after each move
- Both engines produce the same `currentPhase` after each move  
- Both engines produce the same `gameStatus` after each move
- Both engines produce the same structural state hash

### 1.3 Current Parity Status

✅ **All board types are passing parity** (as of December 2025):

| Board Type | Status | Canonical DB |
|------------|--------|--------------|
| Square 8×8 | ✅ Passing | [`canonical_square8.db`](../ai-service/data/games/canonical_square8.db) |
| Square 19×19 | ✅ Passing | [`canonical_square19.db`](../ai-service/data/games/canonical_square19.db) |
| Hexagonal | ✅ Passing | `canonical_hex.db` (⚠️ deprecated R10 geometry) |

---

## 2. Canonical Data Locations

### 2.1 Directory Structure

```
ai-service/data/games/
├── canonical_square8.db                    # Canonical Square8 games
├── canonical_square8.db.parity_gate.json   # Parity gate results
├── canonical_square8.db.health.json        # DB health summary
├── canonical_square19.db                   # Canonical Square19 games
├── canonical_square19.db.parity_gate.json  # Parity gate results
├── db_health.canonical_square8.json        # Alternative health file
└── ...
```

### 2.2 Parity Gate JSON Format

Each canonical DB has an accompanying parity gate JSON:

```json
{
  "board_type": "square8",
  "db_path": "/path/to/canonical_square8.db",
  "num_games": 12,
  "seed": 42,
  "max_moves": 200,
  "soak_returncode": 0,
  "parity_summary": {
    "total_databases": 1,
    "total_games_checked": 12,
    "games_with_semantic_divergence": 0,
    "games_with_end_of_game_only_divergence": 0,
    "games_with_structural_issues": 0,
    "semantic_divergences": [],
    "structural_issues": [],
    "mismatch_counts_by_dimension": {}
  },
  "passed_canonical_parity_gate": true
}
```

**Key fields:**
- `passed_canonical_parity_gate`: **Must be `true`** for training use
- `games_with_semantic_divergence`: Should be `0`
- `games_with_structural_issues`: Should be `0`

### 2.3 Training Data Registry

The authoritative inventory of canonical vs legacy databases is maintained in:

📄 [`ai-service/TRAINING_DATA_REGISTRY.md`](../ai-service/TRAINING_DATA_REGISTRY.md)

**Data Classification:**

| Status | Meaning |
|--------|---------|
| `canonical` | Parity + canonical history gates passed - safe for training |
| `legacy_noncanonical` | Pre-parity-fix data - **DO NOT use for training** |
| `pending_gate` | Not yet validated - requires gate before training |

---

## 3. Running Parity Checks

### 3.1 Main Parity Check Script

The primary tool for checking TS↔Python parity:

```bash
cd ai-service
PYTHONPATH=. python scripts/check_ts_python_replay_parity.py --db <path>
```

**Example - Check a single DB:**
```bash
PYTHONPATH=. python scripts/check_ts_python_replay_parity.py \
  --db data/games/canonical_square8.db
```

**Output (JSON):**
```json
{
  "total_databases": 1,
  "total_games_checked": 12,
  "games_with_semantic_divergence": 0,
  "games_with_end_of_game_only_divergence": 0,
  "games_with_structural_issues": 0,
  "semantic_divergences": [],
  "structural_issues": [],
  "mismatch_counts_by_dimension": {}
}
```

### 3.2 Compact Output Mode

For grep-friendly single-line output:

```bash
PYTHONPATH=. python scripts/check_ts_python_replay_parity.py \
  --db data/games/my_selfplay.db \
  --compact
```

**Output format:**
```
SEMANTIC db=/path/to/db game=<game_id> diverged_at=50 py_phase=movement ts_phase=territory_processing ...
```

### 3.3 Parity Healthcheck (All Canonical DBs)

Run parity checks against all canonical databases:

```bash
cd ai-service
PYTHONPATH=. python scripts/run_parity_healthcheck.py
```

**Options:**
```bash
--suite contract_vectors_v2   # Run specific suite
--suite plateau_snapshots     # Run specific suite
--summary-json output.json    # Write results to file
--fail-on-mismatch            # Exit non-zero on any mismatch (for CI)
```

### 3.4 Generate New Canonical Self-Play + Gate

To generate a new canonical DB with automatic parity gating:

```bash
cd ai-service
PYTHONPATH=. python scripts/generate_canonical_selfplay.py \
  --board-type square8 \
  --num-games 32 \
  --db data/games/canonical_square8.db \
  --summary db_health.canonical_square8.json
```

A DB is eligible for `canonical` status **only if**:
- `canonical_ok` in summary is `true`
- `parity_gate.passed_canonical_parity_gate` is `true`
- `canonical_history.non_canonical_games == 0`

### 3.5 Tracing a Single Game

For detailed per-move debugging of a specific game:

```bash
PYTHONPATH=. python scripts/check_ts_python_replay_parity.py \
  --db data/games/my_selfplay.db \
  --trace-game <game_id> \
  --trace-max-k 100  # Optional: limit to first 100 moves
```

**Output:**
```
TRACE-HEADER db=... game=<id> structure=good total_moves_py=67 total_moves_ts=67
TRACE db=... game=<id> k=0 py_player=1 ts_player=1 py_phase=ring_placement ts_phase=ring_placement ...
TRACE db=... game=<id> k=1 py_player=1 ts_player=1 ...
...
```

---

## 4. Debugging Divergences

### 4.1 Debug Workflow Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  1. Run parity check with --emit-state-bundles-dir             │
│  2. Identify divergent game and k value                         │
│  3. Run diff_state_bundle.py on the bundle                      │
│  4. Analyze structural differences                              │
│  5. Identify which engine (TS or Python) is incorrect           │
│  6. Fix the engine implementation                               │
│  7. Re-run parity check to verify fix                           │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Step 1: Generate State Bundles

When divergences occur, generate rich debug bundles:

```bash
PYTHONPATH=. python scripts/check_ts_python_replay_parity.py \
  --db data/games/failing.db \
  --emit-state-bundles-dir parity_bundles/debug_session
```

This creates files like:
```
parity_bundles/debug_session/
├── failing__<game_id>__k50.state_bundle.json
├── failing.db__<game_id>__k49.ts_state.json
└── failing.db__<game_id>__k50.ts_state.json
```

### 4.3 Step 2: Diff the State Bundle

Analyze the divergence with the bundle diff tool:

```bash
PYTHONPATH=. python scripts/diff_state_bundle.py \
  --bundle parity_bundles/debug_session/failing__<game_id>__k50.state_bundle.json
```

**Sample output:**
```
Bundle: parity_bundles/debug_session/failing__game123__k50.state_bundle.json
DB:     /path/to/failing.db
Game:   game123
ts_k:   50
diverged_at: 50
mismatch_kinds: ['current_player', 'current_phase', 'game_status']
mismatch_context: post_move
available_ts_k_values: [49, 50]

Python state:
  phase=movement player=2 status=active
  stacks=12 collapsed=5 total_elims=8

TS state:
  phase=territory_processing player=1 status=completed
  stacks=12 collapsed=5 total_elims=8

=== Concise structural diff summary ===
  structural_diff=False
  reason=no structural differences

Players (PY vs TS):
  P1: PY={'elim': 4, 'terr': 5, 'hand': 0} TS={'elim': 4, 'terr': 5, 'hand': 0}
  P2: PY={'elim': 4, 'terr': 3, 'hand': 0} TS={'elim': 4, 'terr': 3, 'hand': 0}
```

### 4.4 Step 3: Interpret the Output

| Mismatch Kind | Typical Cause | Investigation Focus |
|---------------|---------------|---------------------|
| `current_phase` | Missing bookkeeping move injection | Check `_auto_inject_no_action_moves()` |
| `current_player` | Player rotation mismatch | Check turn advancement logic |
| `game_status` | Victory detection mismatch | Check terminal condition evaluation |
| `state_hash` | Structural board difference | Diff stacks, collapsed, players |

### 4.5 Generate Parity Fixtures for Regression

When you find a divergence, generate a compact fixture for regression testing:

```bash
PYTHONPATH=. python scripts/check_ts_python_replay_parity.py \
  --db data/games/failing.db \
  --emit-fixtures-dir parity_fixtures
```

This creates fixtures like:
```
parity_fixtures/failing__game123__k50.json
```

---

## 5. Key Concepts

### 5.1 The Parity Fix: Bookkeeping Move Injection

**Problem:** When replaying recorded games, Python would remain stuck in `line_processing` or `territory_processing` phases waiting for explicit moves that don't exist in legacy databases.

**Solution (RR-PARITY-FIX):** The Python [`GameReplayDB.get_state_at_move()`](../ai-service/app/db/game_replay.py:1143) method auto-injects bookkeeping moves to match TypeScript's replay behavior.

**Key code in [`game_replay.py`](../ai-service/app/db/game_replay.py:1238-1296):**

```python
def _auto_inject_no_action_moves(self, state: GameState) -> GameState:
    """Auto-inject NO_LINE_ACTION and NO_TERRITORY_ACTION bookkeeping moves.

    This helper matches TS's replay behavior where the orchestrator
    auto-generates these moves to complete turn traversal through
    phases that have no interactive options.
    """
    from app.game_engine import GameEngine, PhaseRequirementType

    while iterations < max_iterations:
        requirement = GameEngine.get_phase_requirement(state, state.current_player)
        
        if requirement.type == PhaseRequirementType.NO_LINE_ACTION_REQUIRED:
            bookkeeping = GameEngine.synthesize_bookkeeping_move(requirement, state)
            state = GameEngine.apply_move(state, bookkeeping, trace_mode=True)
        elif requirement.type == PhaseRequirementType.NO_TERRITORY_ACTION_REQUIRED:
            bookkeeping = GameEngine.synthesize_bookkeeping_move(requirement, state)
            state = GameEngine.apply_move(state, bookkeeping, trace_mode=True)
        else:
            break

    return state
```

### 5.2 Why Bookkeeping Moves Matter

The TypeScript [`SandboxOrchestratorAdapter`](../src/client/sandbox/SandboxOrchestratorAdapter.ts) automatically advances through phases with no interactive options during replay. Key logic:

```typescript
// RR-CANON-R076: Handle required no-action decisions from core layer.
if (
  decision.type === 'no_line_action_required' ||
  decision.type === 'no_territory_action_required'
) {
  if (this.skipTerritoryAutoResolve) {
    // Replay/traceMode: require explicit move from recording
    break;
  }
  // Live play: synthesize and auto-apply the required no-action move
  const noActionMove: Move = {
    type: moveType,
    player: decision.player,
    ...
  };
  result = runProcessTurn(workingState, noActionMove);
}
```

### 5.3 The RR-PARITY-FIX Tag Convention

Code related to parity fixes is tagged with `RR-PARITY-FIX` comments for easy searching:

```bash
grep -r "RR-PARITY-FIX" ai-service/
```

---

## 6. Parity Fixtures (Contract Vectors)

### 6.1 What Parity Fixtures Are

Parity fixtures are JSON snapshots capturing:
- Game ID and DB path where divergence occurred
- The exact move number (`k` value) of divergence
- Python and TS state summaries at that point
- Mismatch classification (phase, player, status, hash)

They serve as **regression tests** - once a divergence is fixed, the fixture ensures it stays fixed.

### 6.2 Fixture Location

```
ai-service/parity_fixtures/
├── selfplay_square8_2p__*.json
├── selfplay_square19_2p__*.json
├── selfplay_hexagonal_2p__*.json
└── state_bundles_hex_after_fix/   # Rich bundles for debugging
```

### 6.3 Fixture Format

```json
{
  "db_path": "/path/to/selfplay.db",
  "game_id": "3f9641f6-5f17-423d-b529-aa9fc054d786",
  "diverged_at": 50,
  "mismatch_kinds": ["current_player", "current_phase", "game_status"],
  "mismatch_context": "post_move",
  "total_moves_python": 67,
  "total_moves_ts": 67,
  "python_summary": {
    "current_phase": "movement",
    "current_player": 2,
    "game_status": "active",
    "move_index": 49,
    "state_hash": "b2c2f563cbbd5f92"
  },
  "ts_summary": {
    "current_phase": "territory_processing",
    "current_player": 1,
    "game_status": "completed",
    "move_index": 50,
    "state_hash": "cd457d73bc3b69b2"
  },
  "canonical_move_index": 49,
  "canonical_move": { /* Move JSON */ }
}
```

### 6.4 Generating New Fixtures

When investigating a failing DB:

```bash
PYTHONPATH=. python scripts/check_ts_python_replay_parity.py \
  --db data/games/failing.db \
  --emit-fixtures-dir parity_fixtures
```

### 6.5 Naming Convention

Fixtures follow this naming pattern:
```
{db_stem}__{game_id}__k{diverged_at}.json
```

Example: `selfplay_square8_2p__3f9641f6-5f17-423d-b529-aa9fc054d786__k50.json`

---

## 7. Parity CI Integration

### 7.1 Current Status

✅ **Parity CI is now automated** (PA-6 implemented December 2025).

Two levels of parity checks run in CI:

| Workflow | Trigger | What it checks |
|----------|---------|----------------|
| `parity-ci.yml` | PRs affecting engine code | TS↔Python replay parity on `canonical_square8.db` |
| `ci.yml` (`python-parity-healthcheck`) | All PRs | Contract vectors + plateau snapshots |

### 7.2 Parity CI Workflow (`parity-ci.yml`)

The dedicated parity CI gate runs on PRs that modify:
- `src/shared/engine/**` (TS engine)
- `src/client/sandbox/**` (TS sandbox)
- `ai-service/app/rules/**` (Python rules)
- `ai-service/app/game_engine.py` (Python engine)
- `ai-service/app/db/game_replay.py` (Python replay DB)
- Canonical databases and parity scripts

**What it does:**
1. Sets up Node.js (for TS replay harness) and Python
2. Runs [`check_ts_python_replay_parity.py`](../ai-service/scripts/check_ts_python_replay_parity.py) with `--fail-on-divergence`
3. Gates merges on zero semantic divergences
4. Uploads summary JSON as artifact for debugging

**Runtime:** ~5 minutes (using small `canonical_square8.db`)

### 7.3 Workflow Commands

**Check replay parity (CI-equivalent):**
```bash
cd ai-service
PYTHONPATH=. python scripts/check_ts_python_replay_parity.py \
  --db data/games/canonical_square8.db \
  --fail-on-divergence
```

**Check contract vectors + plateau snapshots (CI-equivalent):**
```bash
cd ai-service
PYTHONPATH=. python scripts/run_parity_healthcheck.py --fail-on-mismatch
```

### 7.4 Manual Pre-Merge Check

Before merging changes to rules/engine code, run both checks:

```bash
cd ai-service

# 1. Replay parity (TS↔Python on canonical DBs)
PYTHONPATH=. python scripts/check_ts_python_replay_parity.py \
  --db data/games/canonical_square8.db \
  --fail-on-divergence

# 2. Contract vectors + plateau snapshots
PYTHONPATH=. python scripts/run_parity_healthcheck.py --fail-on-mismatch
```

---

## 8. Troubleshooting Guide

### 8.1 "Phase Mismatch" (current_phase differs)

**Symptom:** Python shows `movement`, TS shows `territory_processing`

**Common causes:**
1. **Missing bookkeeping move injection** - Check `_auto_inject_no_action_moves()` is called
2. **Phase transition logic differs** - Compare phase state machines
3. **Legacy DB without explicit phase moves** - May need regeneration

**Fix approach:**
```bash
# Trace the specific game to see where phases diverge
PYTHONPATH=. python scripts/check_ts_python_replay_parity.py \
  --db failing.db \
  --trace-game <game_id>
```

### 8.2 "Player Mismatch" (current_player differs)

**Symptom:** Python shows player 2, TS shows player 1

**Common causes:**
1. **Turn advancement logic differs** - Check forced elimination rotation
2. **Player skip logic differs** - Check eliminated player handling
3. **Off-by-one in move counting** - Index alignment issue

**Fix approach:**
```bash
# Generate state bundle and compare player states
PYTHONPATH=. python scripts/check_ts_python_replay_parity.py \
  --db failing.db \
  --emit-state-bundles-dir debug_bundles

# Then diff the specific k value
PYTHONPATH=. python scripts/diff_state_bundle.py \
  --bundle debug_bundles/failing__<game_id>__k<N>.state_bundle.json
```

### 8.3 "Move Count Mismatch"

**Symptom:** `total_moves_python != total_moves_ts`

**Common causes:**
1. **TS or Python fails to apply some moves** - Check for validation errors
2. **Legacy move format incompatibility** - Check `normalizeRecordedMove()` in TS

**Fix approach:**
```bash
# Check compact output for the specific game
PYTHONPATH=. python scripts/check_ts_python_replay_parity.py \
  --db failing.db \
  --compact | grep <game_id>
```

### 8.4 "Hash Mismatch" (state_hash differs)

**Symptom:** Phase/player/status match but `state_hash` differs

**Common causes:**
1. **Board structure difference** - Different stacks, markers, or collapsed cells
2. **Player progress difference** - Different eliminated_rings or territory_spaces
3. **Hash computation difference** - Unlikely but check `hash_game_state()`

**Fix approach:**
```bash
# Use diff_state_bundle to see structural differences
PYTHONPATH=. python scripts/diff_state_bundle.py --bundle <bundle.json>

# Look for:
# - "stacks only in PY" or "stacks only in TS"
# - "collapsed cells with differing owners"
# - "players(elim/territory/hand) differ"
```

### 8.5 "Non-Canonical History" Structure

**Symptom:** `structure: non_canonical_history`

**Common causes:**
1. **Legacy DB with implicit phase transitions** - Must regenerate
2. **Recording pipeline bug** - Check `validate_canonical_move()` enforcement

**Fix approach:**
- Regenerate the DB using canonical self-play pipeline
- Check [`history_contract.py`](../ai-service/app/rules/history_contract.py) for valid phase/move-type pairs

---

## 9. Key Files Reference

### 9.1 Python Parity Infrastructure

| File | Purpose |
|------|---------|
| [`ai-service/app/db/game_replay.py`](../ai-service/app/db/game_replay.py) | GameReplayDB with parity fix (`_auto_inject_no_action_moves`) |
| [`ai-service/scripts/check_ts_python_replay_parity.py`](../ai-service/scripts/check_ts_python_replay_parity.py) | Main parity checker script |
| [`ai-service/scripts/diff_state_bundle.py`](../ai-service/scripts/diff_state_bundle.py) | State bundle differ |
| [`ai-service/scripts/run_parity_healthcheck.py`](../ai-service/scripts/run_parity_healthcheck.py) | Multi-suite parity healthcheck |
| [`ai-service/scripts/run_canonical_selfplay_parity_gate.py`](../ai-service/scripts/run_canonical_selfplay_parity_gate.py) | Generate + gate canonical DBs |
| [`ai-service/scripts/generate_canonical_selfplay.py`](../ai-service/scripts/generate_canonical_selfplay.py) | Unified canonical generator |
| [`ai-service/app/rules/history_contract.py`](../ai-service/app/rules/history_contract.py) | Canonical phase/move-type contract |
| [`ai-service/app/rules/history_validation.py`](../ai-service/app/rules/history_validation.py) | History validation helpers |

### 9.2 TypeScript Parity Infrastructure

| File | Purpose |
|------|---------|
| [`scripts/selfplay-db-ts-replay.ts`](../scripts/selfplay-db-ts-replay.ts) | TS replay harness for parity checks |
| [`src/client/sandbox/SandboxOrchestratorAdapter.ts`](../src/client/sandbox/SandboxOrchestratorAdapter.ts) | TS sandbox adapter with auto-advance |
| [`src/client/sandbox/ClientSandboxEngine.ts`](../src/client/sandbox/ClientSandboxEngine.ts) | TS sandbox engine |
| [`src/shared/engine/orchestration/turnOrchestrator.ts`](../src/shared/engine/orchestration/turnOrchestrator.ts) | Core TS turn orchestrator |

### 9.3 Documentation

| File | Purpose |
|------|---------|
| [`ai-service/TRAINING_DATA_REGISTRY.md`](../ai-service/TRAINING_DATA_REGISTRY.md) | Canonical vs legacy data inventory |
| [`ai-service/docs/GAME_REPLAY_DATABASE_SPEC.md`](../ai-service/docs/GAME_REPLAY_DATABASE_SPEC.md) | GameReplayDB schema spec |
| [`AGENTS.md`](../AGENTS.md) | Agent guidelines including parity rules |

---

## 10. Quick Reference Commands

### Check parity for a single DB
```bash
cd ai-service
PYTHONPATH=. python scripts/check_ts_python_replay_parity.py --db <path>
```

### Run all parity healthchecks
```bash
PYTHONPATH=. python scripts/run_parity_healthcheck.py --fail-on-mismatch
```

### Generate debug bundles for divergences
```bash
PYTHONPATH=. python scripts/check_ts_python_replay_parity.py \
  --db <path> \
  --emit-state-bundles-dir bundles/
```

### Diff a state bundle
```bash
PYTHONPATH=. python scripts/diff_state_bundle.py --bundle <path.state_bundle.json>
```

### Trace a specific game move-by-move
```bash
PYTHONPATH=. python scripts/check_ts_python_replay_parity.py \
  --db <path> \
  --trace-game <game_id>
```

### Generate canonical DB with parity gate
```bash
PYTHONPATH=. python scripts/generate_canonical_selfplay.py \
  --board-type square8 \
  --num-games 32 \
  --db data/games/canonical_square8.db \
  --summary db_health.json
```

---

_Last updated: 2025-12-07 (PA-6: Parity CI Gate automated)_