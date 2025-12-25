# AI Pipeline Parity Fixes

> **Doc Status (2025-12-11): Historical record**
>
> This document tracks fixes that resolved TS↔Python replay parity failures at the time. Current parity status may differ; verify against the latest gate summaries and the parity runbook.

**Created:** 2025-12-11
**Last Updated:** 2025-12-11
**Related Documents:**

- [PYTHON_PARITY_REQUIREMENTS.md](../rules/PYTHON_PARITY_REQUIREMENTS.md)
- [KNOWN_ISSUES.md](../../KNOWN_ISSUES.md)
- [RULES_CANONICAL_SPEC.md](../../RULES_CANONICAL_SPEC.md)

---

## Executive Summary

The AI training pipeline requires TS↔Python parity for canonical game replay. Several divergences were discovered and fixed:

| Issue                        | Status   | Impact                                      |
| ---------------------------- | -------- | ------------------------------------------- |
| Early LPS victory detection  | ✅ Fixed | TS now counts total rings including buried  |
| Territory victory threshold  | ✅ Fixed | TS now counts from collapsedSpaces directly |
| Early victory in replay      | ✅ Fixed | TS replay terminates when victory detected  |
| Move count mismatch handling | ✅ Fixed | Parity checker accepts early termination    |

**Result (as of 2025-12-11):** All canonical parity tests pass (48 contract vectors, 9+ game replays; current v2 total = 90). For current status, see [`docs/runbooks/PARITY_VERIFICATION_RUNBOOK.md`](../runbooks/PARITY_VERIFICATION_RUNBOOK.md).

---

## 1. Fixes Applied

### 1.1 Early LPS (Last Player Standing) Victory Detection

**File:** `src/shared/engine/aggregates/VictoryAggregate.ts`

**Problem:** TS's Early LPS check only looked at `ringsInHand`, not total rings (including rings buried in opponent stacks). Python's `count_rings_in_play_for_player` correctly counts all rings.

**Root Cause:** The Early LPS check (RR-CANON-R172) requires checking if all other players have NO RINGS TOTAL (board + hand). A player's rings on board include ALL their rings, even if buried in stacks controlled by other players.

**Fix:** Added `countTotalRingsForPlayer` helper function:

```typescript
function countTotalRingsForPlayer(state: GameState, playerNumber: number): number {
  const player = state.players.find((p) => p.playerNumber === playerNumber);
  const ringsInHand = player?.ringsInHand ?? 0;

  let ringsOnBoard = 0;
  for (const stack of state.board.stacks.values()) {
    for (const ringOwner of stack.rings) {
      if (ringOwner === playerNumber) {
        ringsOnBoard++;
      }
    }
  }

  return ringsOnBoard + ringsInHand;
}
```

Updated Early LPS check to use this helper:

```typescript
if (playersWithStacks.size === 1) {
  const stackOwner = Array.from(playersWithStacks)[0];
  const othersHaveMaterial = players.some(
    (p) => p.playerNumber !== stackOwner && countTotalRingsForPlayer(state, p.playerNumber) > 0
  );
  if (!othersHaveMaterial) {
    return {
      isGameOver: true,
      winner: stackOwner,
      reason: 'last_player_standing',
      handCountsAsEliminated: false,
    };
  }
}
```

### 1.2 Territory Victory Threshold Check

**File:** `src/shared/engine/aggregates/VictoryAggregate.ts`

**Problem:** TS used `player.territorySpaces` field which could be stale after territory region processing moves. Python's `_check_victory` counts directly from `collapsed_spaces`.

**Root Cause:** The `territorySpaces` player field may not be updated atomically during territory processing, leading to race conditions where the count is checked before it's fully updated.

**Fix:** Changed to count directly from `collapsedSpaces` map:

```typescript
// 2) Territory-control victory: Use actual collapsed_spaces count
const territoryCounts = new Map<number, number>();
for (const owner of state.board.collapsedSpaces.values()) {
  territoryCounts.set(owner, (territoryCounts.get(owner) ?? 0) + 1);
}
for (const [playerId, count] of territoryCounts.entries()) {
  if (count >= state.territoryVictoryThreshold) {
    return {
      isGameOver: true,
      winner: playerId,
      reason: 'territory_control',
      handCountsAsEliminated: false,
    };
  }
}
```

### 1.3 Early Victory Detection in TS Replay

**File:** `scripts/selfplay-db-ts-replay.ts`

**Problem:** TS replay didn't check victory after each move like Python does. When a game reaches victory mid-replay, Python would stop but TS would continue applying moves.

**Fix:** Added `evaluateVictory` call after each move:

```typescript
const verdict = evaluateVictory(stateTyped);
const earlyVictoryDetected =
  verdict.isGameOver && verdict.winner !== undefined && stateTyped.gameStatus !== 'completed';

if (earlyVictoryDetected) {
  console.log(
    JSON.stringify({
      kind: 'ts-replay-early-victory',
      k: applied,
      winner: verdict.winner,
      reason: verdict.reason,
    })
  );
  // Mark game as completed
  mutableState.gameStatus = 'completed';
  mutableState.winner = verdict.winner;
  mutableState.currentPhase = 'game_over';
}
```

Added early return with `ts-replay-game-ended` event:

```typescript
if (earlyVictoryDetected) {
  console.log(JSON.stringify({
    kind: 'ts-replay-game-ended',
    appliedMoves: applied,
    remainingRecordedMoves: recordedMoves.length - applied,
    summary: summarizeState('game_ended', engine.getState() as GameState),
  }));
  return { state, appliedMoves: applied, ... };
}
```

### 1.4 Parity Checker: Handle Early Victory Termination

**File:** `ai-service/scripts/check_ts_python_replay_parity.py`

**Problem:** When TS terminates early due to victory detection, the move counts differ (e.g., Python DB has 120 moves, TS stops at 117). This was incorrectly flagged as a divergence.

**Fix 1:** Capture final summary from `ts-replay-game-ended`:

```python
elif kind == "ts-replay-game-ended":
    applied_moves = payload.get("appliedMoves")
    if applied_moves is not None:
        total_ts_moves = int(applied_moves)
    # Capture final summary for comparison
    summary = payload.get("summary") or {}
    k = total_ts_moves
    if k > 0 and k not in post_move_summaries:
        post_move_summaries[k] = StateSummary(
            move_index=k,
            current_player=summary.get("currentPlayer"),
            current_phase=summary.get("currentPhase"),
            game_status=_canonicalize_status(summary.get("gameStatus")),
            state_hash=summary.get("stateHash"),
            is_anm=summary.get("is_anm"),
        )
```

**Fix 2:** Accept move count difference when early victory is valid:

```python
if diverged_at is None and total_moves_py != total_moves_ts:
    ts_final_summary = ts_summaries.get(total_moves_ts)
    early_victory_acceptable = False

    if (ts_final_summary is not None
        and ts_final_summary.game_status == "completed"
        and total_moves_ts < total_moves_py):
        py_move_index = total_moves_ts - 1
        if py_move_index >= 0:
            py_final_summary = summarize_python_state(db, game_id, py_move_index)
            if (py_final_summary.game_status == "completed"
                and py_final_summary.state_hash == ts_final_summary.state_hash):
                early_victory_acceptable = True

    if not early_victory_acceptable:
        mismatch_kinds = ["move_count"]
```

**Fix 3:** Update divergence classification:

```python
# Only flag move_count if explicitly marked as mismatch
has_move_count_mismatch = "move_count" in (result.mismatch_kinds or [])
if result.diverged_at is not None or has_move_count_mismatch:
    # ... handle as divergence
```

---

## 2. Validation Results

### 2.1 Contract Vector Tests

```
48 passed, 36 skipped (multi-phase orchestrator tests)
```

### 2.2 Replay Parity Tests

Tested databases:

- `/tmp/soak_test.db` - 5 games, 0 divergences
- `/tmp/canonical_test_fresh.db` - 2 games, 0 divergences
- `/tmp/test_canonical_fixed7.db` - 2 games, 0 divergences

### 2.3 Early Victory Verification

Game `ed1d7d1e-8d2e-496c-bed8-83ae8cd2b493`:

- DB recorded moves: 120
- TS applied moves: 117 (stopped at Early LPS victory)
- Final state hash: `06506751f6e0ff0f` (matches Python)
- Result: **PASS** (acceptable early termination)

---

## 3. Architectural Considerations

### 3.1 Victory Detection as Single Source of Truth

The `evaluateVictory` function in `VictoryAggregate.ts` is now the canonical SSOT for victory detection in TS.

> **Note:** As of Dec 11, 2025, `victoryLogic.ts` is deprecated. All victory logic is consolidated in `src/shared/engine/aggregates/VictoryAggregate.ts`. Imports should use the shared engine index:
>
> ```typescript
> import { evaluateVictory, getLastActor } from '../../shared/engine';
> ```
>
> See [VICTORY_REFACTORING_PLAN.md](../architecture/VICTORY_REFACTORING_PLAN.md) for details.

It's used by:

- Turn orchestrator during live games
- Replay engine during parity validation
- Sandbox engine for client-side validation

**Recommendation:** Python's victory detection should mirror this exact logic. Consider extracting the victory detection rules into a shared specification that both implementations can reference.

### 3.2 State Hash for Parity Verification

The `stateHash` field provides a reliable fingerprint for comparing game states across implementations. It's computed from:

- Board topology (stacks, markers, collapsed spaces)
- Player state (rings in hand, eliminated rings, territory)
- Game progress (current player, phase, status)

**Recommendation:** Document the state hash computation algorithm to ensure TS and Python produce identical hashes for identical states.

### 3.3 Early Victory vs DB Recording

When a game is recorded to the DB, it should stop recording moves once victory is detected. The fact that some DBs have extra moves after victory suggests a recording bug in the selfplay pipeline.

**Recommendation:** Add victory detection to the selfplay recording pipeline to prevent recording moves after game end.

---

## 4. Future Work

1. **Audit Python victory detection** - Ensure Python's `_check_victory` matches the updated TS logic exactly
2. **State hash documentation** - Document the state hash algorithm for cross-language consistency
3. **Recording pipeline fix** - Stop recording moves after victory detection in selfplay
4. **Comprehensive parity gate** - Add early victory scenarios to the parity gate test suite
5. **Verify victoryLogic.ts removal** - Confirm no lingering imports (victoryLogic.ts removed Dec 2025)
6. **Extract common victory evaluation helper** - Consolidate duplicate victory+LPS evaluation in replay script

---

## 5. Related Canonical Rules

- **RR-CANON-R172:** Early Last-Player-Standing victory condition
- **RR-CANON-R100/R203:** Players with stacks can always act (ANM/FE machinery)
- **RR-CANON-R061 / R170 (rulebook §13.1):** Ring-elimination victory (`victoryThreshold`, scales with player count)
- **Rulebook §13.2:** Territory-control victory (> 50% board spaces)
- **Section 13.4:** Stalemate tie-break ladder

---

## Change Log

| Date       | Changes                                                                                         |
| ---------- | ----------------------------------------------------------------------------------------------- |
| 2025-12-11 | Initial document with all parity fixes                                                          |
| 2025-12-11 | Victory consolidation: Phases 1-3 complete (export routing, import updates, deprecation notice) |
