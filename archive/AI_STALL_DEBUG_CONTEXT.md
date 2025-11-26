# AI Stall Debugging - Current Context

## Summary of Work Completed

### 1. Log Suppression (COMPLETED ✓)

**Problem:** Console was flooded with hundreds of repetitive diagnostic logs when AI stalled.

**Solution Implemented:**

- Added `SANDBOX_NOOP_MAX_THRESHOLD = 10` to stop execution after 10 no-op turns
- Added early return check at start of `maybeRunAITurnSandbox()`
- Added `sandboxStallLoggingSuppressed` flag to disable all logging after threshold
- Logs single clear message: `"[Sandbox AI] Stopping AI execution after 10 consecutive no-op turns. Game is stalled."`

**Result:** Logs now cleanly stop after 10 iterations, making console output manageable.

---

## 2. Diagnostic Logging Enhancement (COMPLETED ✓)

**Improvements Made to `src/client/sandbox/sandboxAI.ts`:**

```typescript
// Added detailed JSON.stringify logging at key points:
1. Placement candidates before filtering
2. Ring supply calculation (NEW)
3. After multi-ring filtering
4. Move selection result
```

**Key logs now show:**

- Exact candidate counts
- Ring supply calculations
- Filter statistics
- Selected move details

---

## 3. Current Status

### What We Know:

1. **User's Original Logs (Hexagonal Board):**
   - Shows stall occurring during `ring_placement` phase
   - Player 4 stuck with 277 placement candidates
   - No "After multi-ring filtering" log appears → indicates early return
   - Game reaches 400+ consecutive no-ops before our fix

2. **Test Results (8x8 Compact Board):**
   - NO STALL OCCURS on 8x8 compact
   - All logs appear correctly including "Ring supply calculation" and "After multi-ring filtering"
   - Game progresses normally through placement phase

### Critical Observation:

**The stall appears to be BOARD-SPECIFIC** - occurs on hexagonal boards but NOT on 8x8 compact. This suggests:

- Possible geometry/coordinate handling difference
- Different board states trigger the bug
- Hexagonal-specific logic issue

---

## 4. Root Cause Hypothesis

### Original Theory (From User's Analysis):

**Double-filtering bug** - placement candidates filtered twice causing empty set:

1. First: `enumerateLegalRingPlacements()` returns 277 candidates
2. Second: Loop checks `hasAnyLegalMoveOrCaptureFrom` on hypothetical boards → filters out ALL candidates
3. Empty array causes no move selection → stall

### Evidence Supporting This:

- User's hexagonal logs show 277 candidates before filtering
- NO "After multi-ring filtering" log (code exits early somewhere)
- NO "Ring supply calculation" log appears in user's hexagonal logs
- This suggests code returns BEFORE reaching the ring supply calculation section

### New Discovery from Browser Test:

On 8x8 compact, the "Ring supply calculation" log DOES appear, meaning:

- The code successfully reaches past `placementCandidates.length === 0` check
- The code successfully reaches ring supply calculation
- Filtering completes normally

**Therefore:** The bug must be in code that executes BEFORE the ring supply calculation, and only triggers under hexagonal board conditions.

---

## 5. Suspected Code Sections

Looking at the execution flow in `sandboxAI.ts`:

```typescript
// This code runs BEFORE ring supply calculation:
if (placementCandidates.length === 0) {
  if (hasAnyActionFromStacks) {
    // Skip placement path
    return;
  }
  // Forced elimination path
  return;
}
```

**Theory:** On hexagonal boards, `placementCandidates.length === 0` might be TRUE even though we logged "count: 277". This could happen if:

1. The variable `placementCandidates` is being modified/reassigned
2. There's an async timing issue
3. Different code path executes between logging and the check

---

## Next Debugging Steps

### Immediate Actions:

1. ✅ Test on hexagonal board ("Full Hex" option) to reproduce stall
2. Add MORE granular logging between "before filtering" and "ring supply calculation"
3. Log right before the `if (placementCandidates.length === 0)` check
4. Add try-catch around critical sections to catch silent exceptions

### Code to Instrument:

```typescript
// Add after placementCandidates enumeration:
console.log('[DEBUG] After enumeration:', {
  count: placementCandidates.length,
  firstThree: placementCandidates.slice(0, 3).map((p) => positionToString(p)),
});

// Add right before the check:
console.log('[DEBUG] Before zero check:', {
  count: placementCandidates.length,
});

if (placementCandidates.length === 0) {
  console.log('[DEBUG] INSIDE zero length branch');
  // rest of code...
}
```

### Long-term Fix (Once Root Cause Found):

Based on the double-filtering hypothesis, the fix will likely involve:

- Removing redundant filtering logic
- Or fixing the logic that determines which candidates are "valid"
- Or adjusting when `hasAnyLegalMoveOrCaptureFrom` is called

---

## Files Modified

1. `src/client/sandbox/sandboxAI.ts` - Added log suppression and enhanced diagnostics

## Test Cases Needed

1. ✅ 8x8 Compact with 2 AI players (NO STALL)
2. ⏳ Full Hex with 4 AI players (REPRODUCE STALL - user reported this config)
3. ⏳ 19x19 Classic with 2 AI players (UNKNOWN)

---

## Key Code Locations

### `src/client/sandbox/sandboxAI.ts`

- Line ~430: `maybeRunAITurnSandbox()` function
- Line ~500: Placement candidate enumeration
- Line ~520: Zero-length check (suspected early return)
- Line ~595: Ring supply calculation
- Line ~620: Multi-ring filtering loop

### Related Files to Investigate:

- `src/client/sandbox/ClientSandboxEngine.ts` - Hook implementations
- `src/client/sandbox/sandboxPlacement.ts` - `enumerateLegalRingPlacements()`
- `src/client/sandbox/sandboxMovement.ts` - `hasAnyLegalMoveOrCaptureFrom()`

---

## Trace Data Analysis (from user)

Last successful turn before stall:

- Player 3 placed ring, moved to player 4
- Player 4 placed ring at (-1,9,-8)
- Game transitioned back to player 1 ring_placement
- **Then player 1 stalls with 257 candidates**

This suggests the stall begins AFTER a specific board configuration is reached, not from the start of the game.
