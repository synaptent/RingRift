# AI Stall Bug - Continued Debugging Session

## Current Status: STILL STALLED (Different Root Cause)

### Previous Fix Applied

Changed `stack.stackHeight` to `stack.rings.length` in ring counting logic (line 591 of sandboxAI.ts). This prevented over-counting by excluding caps from the ring count.

**Result:** Partial improvement - ring counts are now accurate, but stall still occurs.

---

## New Root Cause Identified

### Latest Console Output Analysis

Player 4 stalls when it reaches EXACTLY the 36-ring cap:

```
[Sandbox AI Debug] Placement candidates before filtering: {"count":267,"player":4,"ringsInHand":13,"hasAnyActionFromStacks":true,"playerStacksCount":8}
[Sandbox AI Debug] Ring supply calculation: {"player":4,"ringsOnBoard":36,"perPlayerCap":36,"remainingByCap":0,"remainingBySupply":13,"maxAvailableGlobal":0}
[Sandbox AI Debug] Early return: maxAvailableGlobal <= 0
```

**Key observations:**

1. ✅ Ring counting is now correct: `ringsOnBoard`:36 matches `perPlayerCap`:36
2. ❌ Player 4 has 13 rings in hand but can't place any (hit the cap)
3. ✅ Player 4 HAS legal moves available: `hasAnyActionFromStacks`:true with 8 stacks
4. ❌ Code does early return instead of skipping placement
5. ❌ Game stalls with consecutive no-ops

### The Bug

Lines 609-614 in `src/client/sandbox/sandboxAI.ts`:

```typescript
if (maxAvailableGlobal <= 0) {
  if (SANDBOX_AI_STALL_DIAGNOSTICS_ENABLED && !sandboxStallLoggingSuppressed) {
    console.warn('[Sandbox AI Debug] Early return: maxAvailableGlobal <= 0');
  }
  return; // ❌ WRONG: Should skip placement instead!
}
```

**Problem:** When `maxAvailableGlobal` is 0 (player at cap) but `hasAnyActionFromStacks` is true, the player should execute a `skip_placement` move to transition to movement phase. Instead, the code returns early, making no state changes, which triggers the stall detector.

---

## The Correct Fix

When a player hits their ring cap but has legal moves from their stacks, they should skip placement and move to the movement phase. The fix should be:

```typescript
if (maxAvailableGlobal <= 0) {
  // Player has hit their ring cap - check if they can skip placement
  if (hasAnyActionFromStacks) {
    // Generate and apply skip_placement move
    const stateForMove = hooks.getGameState();
    const moveNumber = stateForMove.history.length + 1;
    const sentinelTo = stacksForPlayer[0]?.position ?? ({ x: 0, y: 0 } as Position);

    const skipMove: Move = {
      id: '',
      type: 'skip_placement',
      player: current.playerNumber,
      from: undefined,
      to: sentinelTo,
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber,
    } as Move;

    await hooks.applyCanonicalMove(skipMove);
    lastAIMove = skipMove;
    hooks.setLastAIMove(lastAIMove);
    return;
  }

  // Otherwise, no moves available - try forced elimination
  const eliminated = hooks.maybeProcessForcedEliminationForCurrentPlayer();
  return;
}
```

---

## Code Context

### File: `src/client/sandbox/sandboxAI.ts`

### Function: `maybeRunAITurnSandbox()`

### Phase: `ring_placement`

### Lines: ~590-615

**Current flow:**

1. Calculate `ringsOnBoard` using `stack.rings.length` ✅ (fixed)
2. Calculate `maxAvailableGlobal` = min(cap remaining, rings in hand)
3. **If `maxAvailableGlobal` <= 0: early return ❌ (BUG)**
4. Generate placement candidates
5. Add skip_placement option if `hasAnyActionFromStack Ts`

**Correct flow should be:**

1. Calculate `ringsOnBoard` using `stack.rings.length` ✅
2. Calculate `maxAvailableGlobal`
3. **If `maxAvailableGlobal` <= 0 AND `hasAnyActionFromStacks`: skip placement ✅**
4. **If `maxAvailableGlobal` <= 0 AND NOT `hasAnyActionFromStacks`: forced elimination ✅**
5. Otherwise generate placement candidates normally

---

## Test Configuration That Reproduces Bug

- Board: "Full Hex" (hexagonal)
- Players: 4 AI players
- Stall occurs when any AI player reaches exactly 36 rings on board

---

## Previous Work Done

1. ✅ Added log suppression (10-turn threshold) to prevent console spam
2. ✅ Added enhanced diagnostic logging with JSON.stringify
3. ✅ Fixed ring counting from `stack.stackHeight` to `stack.rings.length`
4. ❌ Still need to fix early return logic when player hits cap

---

## Next Steps for Complete Fix

1. **Move the skip placement logic** to execute BEFORE the `maxAvailableGlobal <= 0` check returns
2. **Handle the at-cap scenario** by checking if player can skip (has legal moves)
3. **Test** on hexagonal board with 4 AI players to confirm no stall
4. **Verify** no regression on other board types

---

## Key Variables in Stall Scenario

- `current.ringsInHand`: 13 (rings still need to be placed)
- `ringsOnBoard`: 36 (exactly at cap)
- `perPlayerCap`: 36
- `remainingByCap`: 0 (at limit)
- `maxAvailableGlobal`: 0 (can't place more rings)
- `hasAnyActionFromStacks`: true (has 8 stacks with legal moves)
- **Expected behavior**: Skip placement and move
- **Actual behavior**: Early return → no-op → stall

---

## Related Code Patterns

The code ALREADY handles this scenario correctly a few lines above (lines 520-550) when `placementCandidates.length === 0`:

```typescript
if (placementCandidates.length === 0) {
  if (hasAnyActionFromStacks) {
    // Generate skip_placement move
    const skipMove: Move = { ... };
    await hooks.applyCanonicalMove(skipMove);
    return;
  }
  // Otherwise forced elimination
  const eliminated = hooks.maybeProcessForcedEliminationForCurrentPlayer();
  return;
}
```

We need the SAME logic for the `maxAvailableGlobal <= 0` scenario!
