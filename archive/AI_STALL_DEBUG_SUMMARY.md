# AI Sandbox Stall Debug Summary

## Problem

When playing AI vs AI games in sandbox mode, the game stalls. The debug bar flashes briefly and repeatedly showing:

- Game stuck in `ring_placement` phase
- `placementCandidateCount: 237` (candidates exist initially)
- `lastAIMoveType: null` (no move actually executed)
- `consecutiveNoopAITurns` incrementing until stall detection triggers

## Investigation Done

### Files Analyzed

1. `src/client/sandbox/sandboxAI.ts` - Main AI turn logic
2. `src/shared/engine/localAIMoveSelection.ts` - Move selection algorithm
3. `src/client/sandbox/sandboxPlacement.ts` - Placement filtering logic

### Root Cause Hypothesis

**Double-filtering bug** in ring placement candidate generation:

1. `enumerateLegalRingPlacements()` filters all board positions using hypothetical board checks → 237 candidates remain
2. The AI turn code applies a **SECOND filter pass** checking `hasAnyLegalMoveOrCaptureFrom` on NEW hypothetical boards for each count (1-3 rings per placement)
3. This second pass may use stale/different board state, filtering out **ALL** remaining candidates
4. With empty candidates array, `chooseLocalMoveFromCandidates()` returns `null`
5. Code early-returns without executing any move → **STALL**

## Diagnostic Logging Added

I've added extensive logging to `sandboxAI.ts` that will output when stall diagnostics are enabled:

### Console Outputs to Watch For:

```javascript
// When placement candidates exist before secondary filtering
'[Sandbox AI Debug] Placement candidates before filtering:'
{
  count: 237,
  player: <playerNumber>,
  ringsInHand: <number>
}

// After the problematic multi-ring filtering loop
'[Sandbox AI Debug] After multi-ring filtering:'
{
  initialCandidates: 237,
  finalCandidates: <0 or >0>,  // ← KEY: if 0, this is the bug location
  filteredOut: <number>,
  hasSkipOption: <boolean>
}

// What move selection returns
'[Sandbox AI Debug] Move selection result:'
{
  selected: { type: 'place_ring', to: {...}, count: <number> } or null, // ← KEY: if null, selection failed
  candidatesLength: <number>,
  candidateTypes: [...]
}
```

### Error Messages to Watch For:

```javascript
// If selection succeeds but to field is missing
'[Sandbox AI] place_ring selected but to is missing:';

// If selection fails with candidates available
'[Sandbox AI] chooseLocalMoveFromCandidates returned null with <N> candidates';

// If unexpected move type
'[Sandbox AI] Unexpected move type in ring_placement: <type>';
```

## Next Steps - For You

### 1. Enable Diagnostics

Make sure stall diagnostics are enabled (should be by default in dev builds).

### 2. Reproduce the Stall

- Start an AI vs AI game in sandbox mode
- Wait for the stall to occur (debug bar flashing)

### 3. Capture Console Logs

- Open browser dev tools console (F12)
- Copy ALL console output, especially:
  - The `[Sandbox AI Debug]` messages
  - Any error messages
  - The final stall detection warning

### 4. Share the Logs

Create a new task and paste:

- The full console output
- Any additional error stack traces

## Expected Outcomes

### If Hypothesis is Correct:

You'll see:

```
[Sandbox AI Debug] Placement candidates before filtering: {count: 237, ...}
[Sandbox AI Debug] After multi-ring filtering: {
  initialCandidates: 237,
  finalCandidates: 0,  // ← ALL candidates filtered out!
  filteredOut: <large number>
}
```

### Potential Fixes Based on Findings:

1. **If all candidates filtered out**: Remove redundant second filtering pass
2. **If board state mismatch**: Ensure both filter passes use same board snapshot
3. **If move selection fails**: Add safety valve to use unfiltered candidates as fallback
4. **If `to` field missing**: Fix candidate construction to always populate `to`

## Code Changes Made

**File**: `src/client/sandbox/sandboxAI.ts`

**Changes**:

- Added logging before/after placement candidate filtering
- Added logging of move selection result
- Added error logging for null/malformed moves
- Added filtering statistics tracking
- Improved error messages for all failure paths

**No functional logic changed yet** - only diagnostics added to identify exact failure point.
