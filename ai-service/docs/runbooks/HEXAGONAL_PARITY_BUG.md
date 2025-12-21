# Hexagonal Board Parity Bug

**Status:** FIXED - Commit 7f43c368
**Resolution:** 2025-12-21 - Wrong attribute name in FSM
**Priority:** RESOLVED
**Discovered:** December 2025

---

## Problem Statement

Hexagonal board games exhibit phase divergence between Python and TypeScript implementations after `no_territory_action` moves. This causes parity validation failures and blocks training on hexagonal board variants.

## Symptoms

- Phase divergence at move ~k=989 in hexagonal games
- Python reports `territory_processing` phase
- TypeScript reports `forced_elimination` phase
- Occurs after `NO_TERRITORY_ACTION` bookkeeping move
- Square board (8x8) games work correctly

## Root Cause Analysis

### Phase Transition Logic

The divergence occurs in the phase state machine transition after territory processing:

1. Player completes territory phase with `NO_TERRITORY_ACTION`
2. Python transitions to `territory_processing` for next player
3. TypeScript transitions to `forced_elimination` check first
4. The `_should_enter_forced_elimination()` gate differs between implementations

### Suspected Locations

| File                             | Lines   | Concern                                              |
| -------------------------------- | ------- | ---------------------------------------------------- |
| `app/rules/phase_machine.py`     | 100-150 | `_did_process_territory_region()` may have edge case |
| `app/rules/fsm.py`               | 300-400 | FSM orchestration for hex boards                     |
| `app/ai/gpu_canonical_export.py` | 400-500 | GPU export may skip bookkeeping moves                |

### Hex-Specific Factors

1. **Larger board**: 469 spaces (hex) vs 64 spaces (square8) = longer games
2. **Territory geometry**: Hexagonal adjacency affects FE eligibility checks
3. **Ring distribution**: Different ring counts per player (96 vs 44)

## Reproduction Steps

```bash
# 1. Generate hex selfplay with parity validation
python scripts/generate_gumbel_selfplay.py \
  --board hexagonal \
  --num-players 2 \
  --games 10 \
  --validate-parity

# 2. Check for parity failures
ls -la parity_failures/canonical_hexagonal_*.json

# 3. Analyze specific failure
python scripts/analyze_parity_failures.py \
  parity_failures/canonical_hexagonal__*.json \
  --verbose
```

## Diagnostic Data

Parity failure bundles are stored in:

```
parity_failures/canonical_hexagonal__<uuid>__k<move>.parity_failure.json
```

Bundle contents:

- `game_id`: UUID of failed game
- `move_k`: Move index where divergence occurred
- `python_state`: Python game state snapshot
- `typescript_state`: TypeScript game state snapshot
- `last_move`: Move that triggered divergence
- `move_history`: Full move history up to failure point

## Fix Strategy

### Step 1: Reproduce Locally

Use the state bundle diff tool first, then optional ad-hoc diffing via debug_utils:

```bash
# Emit state bundles during parity check
python scripts/check_ts_python_replay_parity.py \
  --db ai-service/data/games/canonical_hexagonal.db \
  --emit-state-bundles-dir parity_bundles

# Diff the first divergent bundle
python scripts/diff_state_bundle.py \
  --bundle parity_bundles/<bundle>.state_bundle.json \
  --k <diverged_at>
```

If you need custom comparisons beyond the bundle diff:

```python
import json
from pathlib import Path

from app.db.game_replay import GameReplayDB
from app.utils.debug_utils import StateDiffer, load_ts_state_dump

bundle = json.load(open("parity_failures/<bundle>.parity_failure.json"))
db = GameReplayDB(bundle["db_path"])
py_state = db.get_state_at_move(bundle["game_id"], bundle["diverged_at"])
ts_state = load_ts_state_dump(Path("parity_failures/<bundle>.ts_state.json"))
diff = StateDiffer().diff_py_ts_state(py_state, ts_state)
print(diff)
```

### Step 2: Trace Phase Transitions

In `app/rules/phase_machine.py`:

```python
def advance_phases(input: PhaseTransitionInput) -> None:
    if input.trace_mode:
        print(f"[PHASE] Before: {input.game_state.current_phase}")
        print(f"[PHASE] Last move: {input.last_move.type}")
```

### Step 3: Fix Transition Logic

Based on root cause, fix will likely be in one of:

- `_should_enter_forced_elimination()` - add hex-specific gating
- `_did_process_territory_region()` - fix hex territory detection
- FSM orchestration - ensure bookkeeping moves are consistent

### Step 4: Validate Fix

```bash
# Run hex parity validation
pytest tests/parity/test_fsm_parity.py -k hex -v

# Generate fresh hex games
python scripts/generate_gumbel_selfplay.py \
  --board hexagonal \
  --games 100 \
  --validate-parity \
  --fail-fast
```

## Related Files

- `app/rules/phase_machine.py` - Phase state machine
- `app/rules/fsm.py` - Canonical FSM orchestrator
- `app/ai/gpu_canonical_export.py` - GPU parity validation
- `app/utils/debug_utils.py` - Debug utilities
- `tests/parity/test_fsm_parity.py` - FSM parity tests
- `docs/infrastructure/GPU_RULES_PARITY_AUDIT.md` - Parity audit notes

## References

- 7-phase state machine spec: `docs/specs/GAME_NOTATION_SPEC.md`
- RR-CANON compliance: FSM module is canonical
- TS implementation: `phaseStateMachine.ts`, `turnOrchestrator.ts`

## Success Criteria

- [ ] Hex games complete without phase divergence
- [ ] Parity validation passes for 1000+ hex games
- [ ] Training data can be generated for hex boards
- [ ] Model evaluation works on hex boards

---

## Investigation Notes (2025-12-21)

### Code Analysis

The phase transition logic was analyzed and appears correct:

**Python (`app/rules/phase_machine.py`)**:

- `NO_TERRITORY_ACTION` case (lines 502-511) calls `_on_territory_processing_complete()`
- `_on_territory_processing_complete()` (lines 187-214) checks `had_any_action` and `has_stacks`
- If `not had_any_action and has_stacks`, sets phase to `FORCED_ELIMINATION`
- Otherwise calls `_end_turn()` which sets phase to `RING_PLACEMENT`

**TypeScript (`src/shared/engine/orchestration/turnOrchestrator.ts`)**:

- After territory processing, calls `onTerritoryProcessingComplete()` (line 2929)
- If returns `'forced_elimination'`, calls `stateMachine.transitionTo('forced_elimination')` (line 2939)
- The logic mirrors Python exactly

### Mystery

The symptom (Python in `territory_processing`, TS in `forced_elimination`) should be impossible:

- If `_on_territory_processing_complete` runs, Python would be in either `FORCED_ELIMINATION` or `RING_PLACEMENT`
- Neither branch leaves Python in `territory_processing`

### Next Steps Required

1. **Reproduce locally**: Need to replay the specific game that failed at k=989
2. **Add tracing**: Insert debug logging in `_on_territory_processing_complete`
3. **Check compute_had_any_action_this_turn**: Verify it returns same value as TS
4. **Check player_has_stacks_on_board**: Verify hex-specific stack counting is correct

### Hypothesis

The divergence may be caused by:

- An exception being thrown silently before phase is updated
- A different code path being taken for hex (not yet identified)
- A race condition or state mutation issue in the parity checker itself

---

## ANM State Divergence (NEW ISSUE - 2025-12-21)

**Status:** INVESTIGATION IN PROGRESS
**Priority:** CRITICAL - Blocks hexagonal training
**Discovered:** After HEX-PARITY-01 fix

### Problem Statement

After fixing the territory phase bug (HEX-PARITY-01), 7 of 11 hexagonal games now fail with a NEW divergence:

- **Mismatch dimension:** `anm_state`
- **Phase:** `line_processing`
- **Python:** `is_anm: false` (finds interactive moves)
- **TypeScript:** `is_anm: true` (finds NO interactive moves)
- **State hashes MATCH** (board state is identical)

### Key Observation

State hashes matching proves the board state (markers, stacks, collapsed_spaces) is identical
between Python and TypeScript. The divergence is specifically in the ANM computation, not
the underlying game state.

### Investigation Findings (2025-12-21)

1. **Hash now includes `pendingLineRewardElimination`** - Both Python and TypeScript include
   this field in their hash computation (RR-PARITY-FIX-2025-12-21).

2. **Position key format is consistent** - Both use comma-separated format: `"0,0,-3"`.

3. **Line detection algorithms appear equivalent**:
   - Python: `BoardManager.find_all_lines()` → `_find_line_in_direction()`
   - TypeScript: `findAllLines()` → `findLineInDirection()`
   - Both use same direction vectors and minimum line length (4) for hexagonal.

4. **ANM check paths are identical**:
   - Both check `pending_line_reward_elimination` first
   - Both then call fresh line detection with `detect_now` mode
   - Both return True if lines found for current player

### Hypotheses

1. **Marker iteration difference** - TypeScript Map vs Python dict may iterate differently,
   affecting which markers are checked first and whether lines are found.

2. **Position z-coordinate handling** - Subtle difference in how z=0 vs undefined is handled
   during line stepping and marker lookup.

3. **State serialization issue** - Something in how the state is serialized for parity
   comparison may differ between Python's internal state and TypeScript's replay state.

### Diagnostic Tools Created

- `scripts/diagnose_anm_divergence.py` - Analyzes ANM divergences from parity gate data
- `scripts/run_hex_anm_diagnosis.sh` - Cluster script to generate and analyze fresh hex games

### Next Steps

1. **Run fresh parity check on cluster** with state bundle emission
2. **Add trace logging** to TypeScript's `findAllLines()` and Python's `find_all_lines()`
3. **Capture actual marker keys and positions** at divergence point
4. **Compare line-by-line** what each engine detects

### Sample Divergence

```json
{
  "game_id": "9db77afc-8585-4a31-a061-8d0c5b088ed2",
  "diverged_at": 837,
  "mismatch_kinds": ["anm_state"],
  "python_summary": {
    "current_phase": "line_processing",
    "is_anm": false,
    "state_hash": "2a2e232e45d3dd4f"
  },
  "ts_summary": {
    "current_phase": "line_processing",
    "is_anm": true,
    "state_hash": "2a2e232e45d3dd4f"
  }
}
```

---

**Last Updated:** December 21, 2025
**Status:** HEX-PARITY-01 FIXED - ANM divergence under investigation

---

## Resolution (2025-12-21)

### Root Cause Found

The bug was in `app/rules/fsm.py:_did_process_territory_region()`:

```python
# BUG: Was checking for wrong attribute
if hasattr(region, "positions"):  # ❌ Wrong!
    for pos in region.positions:

# FIX: Territory class uses 'spaces' not 'positions'
if hasattr(region, "spaces"):     # ✅ Correct!
    for pos in region.spaces:
```

The `Territory` class (in `app/models/core.py`) uses `spaces: list[Position]`, not `positions`. This caused `_did_process_territory_region()` to always skip the disconnected_regions check and return `False`.

### Impact

When a CHOOSE_TERRITORY_OPTION move was processed:

1. FSM checked if territory was actually collapsed
2. Due to wrong attribute name, check always failed
3. Python stayed in `territory_processing` phase
4. TypeScript correctly advanced to `forced_elimination`

### Fix Applied

Commit `7f43c368` corrects the attribute name from `.positions` to `.spaces`.

### Validation Required

```bash
# Run hex parity tests
pytest tests/parity/test_fsm_parity.py -k hex -v

# Generate hex games with parity validation
python scripts/generate_gumbel_selfplay.py \
  --board hexagonal \
  --games 100 \
  --validate-parity \
  --fail-fast
```

---

## Code Review Notes (2025-12-21)

### Phase Transition Logic Audit

Reviewed `app/rules/phase_machine.py` for NO_TERRITORY_ACTION handling:

1. **Line 502-511**: `NO_TERRITORY_ACTION` correctly delegates to `_on_territory_processing_complete()`
2. **Line 187-214**: `_on_territory_processing_complete()` logic is correct:
   - Computes `had_any_action` via `compute_had_any_action_this_turn()`
   - Computes `has_stacks` via `player_has_stacks_on_board()`
   - If `not had_any_action and has_stacks` → sets phase to `FORCED_ELIMINATION`
   - Otherwise → calls `GameEngine._end_turn()`

3. **Line 66-96**: `compute_had_any_action_this_turn()` walks move history correctly
4. **Line 99-109**: `player_has_stacks_on_board()` iterates stacks correctly

**Conclusion**: The Python code logic appears correct and board-agnostic. The bug likely requires:

- Live reproduction with tracing enabled
- Comparison of actual game state values between Python and TypeScript at divergence point
- Possible issues: silent exception, state mutation, or parity checker artifact
