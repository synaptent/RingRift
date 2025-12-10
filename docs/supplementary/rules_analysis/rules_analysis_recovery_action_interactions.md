# Recovery Action Interactions Analysis

> **Doc Status (2025-12-10): Active (derived analysis)**
>
> **Purpose:** Detailed analysis of how recovery action interacts with other game systems and its impact on game balance.
>
> **Canonical Source:** `RULES_CANONICAL_SPEC.md` R110–R115, `ringrift_complete_rules.md` §4.5

---

## 1. Recovery Action Summary

**Eligibility:** A player is eligible for recovery if ALL conditions hold:

1. They control **no stacks** on the board
2. They have **zero rings in hand**
3. They have at least one **marker** on the board
4. They have **buried rings** in opponent-controlled stacks

**Action:** Slide one marker to an adjacent empty cell. Legal if **either**:

- **(a) Line formation:** Completes a line of **at least** `lineLength` consecutive markers
- **(b) Fallback:** If no line-forming slide exists, any slide that **does not** cause territory disconnection

**Note:** Territory disconnection is **not** a valid criterion for recovery.

**Overlength Lines:** Overlength lines (longer than `lineLength`) **are permitted**. When an overlength line is formed, the player chooses:

- **Option 1:** Collapse all markers in the line to territory. Cost: 1 buried ring extraction.
- **Option 2:** Collapse exactly `lineLength` consecutive markers of the player's choice. Cost: 0 (no extraction required).

This mirrors normal line reward semantics (RR-CANON-R130–R134).

**Effect:**

1. Line of markers collapses → markers become collapsed spaces (territory)
2. For exact-length lines or Option 1: buried ring extracted → eliminated (credited to recovering player)
3. If collapse creates disconnected regions → territory cascade processing
4. Recovering player now has ring(s) in hand from any exhumed buried rings during cascade

**Critical Rule:** **Collapsed spaces never form part of a valid line.** Only markers can form lines.

---

## 2. Detailed System Interactions

### 2.1 LPS (Last Player Standing)

**Classification:** Recovery is **NOT** a "real action" for LPS purposes.

```
Real Actions = {placement, non-capture_movement, overtaking_capture}
NOT Real Actions = {recovery_slide, forced_elimination, skip_placement, no_* bookkeeping moves}
```

This creates strategic tension: rings in hand become a "survival budget" - players can use recovery moves but must place at least one ring every 2 rounds to avoid LPS loss.

**Impact on LPS Victory:**

| Scenario                                            | Without Recovery   | With Recovery                                       |
| --------------------------------------------------- | ------------------ | --------------------------------------------------- |
| Player A dominates, Player B has only FE            | B cannot block LPS | B cannot block LPS (FE ≠ real action)               |
| Player A dominates, Player B has recovery available | N/A                | B **cannot** block LPS with recovery (not real)     |
| Player B has recovery + rings in hand               | N/A                | B can block LPS by **placing a ring** every 2 turns |
| Only A has real actions for 2 rounds                | A wins LPS         | A wins LPS (recovery doesn't count)                 |

**Strategic Implications:**

- LPS victory is achievable against players with only markers + buried rings (no rings in hand)
- Players with rings in hand have a "survival budget" - must place periodically to reset LPS
- Recovery provides survival but not LPS defense; creates interesting resource management

**Example:**

```
Turn 100: A has 5 stacks, B has 0 stacks, 0 rings in hand
          B has 3 markers and 2 buried rings in A's stacks
          A: Takes real action → LPS round 1 begins (B has no real actions... OR DOES B?)

Question: Does B have a valid recovery slide?
- If YES: B is NOT "without real actions" - B has recovery available
- If NO valid slide exists: B truly has no real actions, LPS proceeds
```

**Key Insight:** LPS evaluation must check if temporarily eliminated players have valid recovery moves, not just whether they have markers + buried rings.

---

### 2.2 Forced Elimination (FE)

**Mutual Exclusivity:** FE and Recovery are **mutually exclusive** by definition:

- FE requires: controlling ≥1 stack
- Recovery requires: controlling 0 stacks

| Aspect          | Forced Elimination                         | Recovery Action                                       |
| --------------- | ------------------------------------------ | ----------------------------------------------------- |
| **Trigger**     | Has stacks but no legal moves              | Has NO stacks, has markers + buried rings             |
| **Cost Source** | Cap elimination (top rings of your stacks) | Buried ring extraction (rings inside opponent stacks) |
| **Result**      | Stack height decreases, may lose control   | Buried ring exhumed → eliminated (credited to you)    |
| **LPS Status**  | NOT a real action                          | NOT a real action                                     |
| **Mandatory?**  | Yes, when triggered                        | Yes, if it's your only legal action                   |

**State Transition:**

```
Active (has stacks)
    → FE loop (no legal moves, has stacks)
    → All stacks eliminated
        ├─ No markers OR no buried rings → Fully eliminated (skipped)
        └─ Has markers AND buried rings → Temporarily eliminated
            ├─ Valid recovery slide exists → Takes recovery (NOT real action)
            └─ No valid recovery slide → Effectively fully eliminated this turn
```

**Note:** Even when recovery is available, the player has no "real action" for LPS purposes. If they have rings in hand, they can place to create a real action.

---

### 2.3 ANM (Active-No-Moves)

**Updated Global Legal Action Set:**

```
Global Legal Actions = {
  placements,
  movements,
  captures,
  forced_elimination (if has stacks, no other moves),
  recovery_slides (if temporarily eliminated AND valid slide exists),
  line_decisions,
  territory_decisions
}
```

**Critical Distinction:** Having markers + buried rings is NECESSARY but NOT SUFFICIENT for recovery:

- Must also have either (a) a valid marker slide that completes at least `lineLength` with sufficient buried rings for cost, OR (b) any slide that does not cause territory disconnection
- If no such slide exists, player has no global legal actions → effectively fully eliminated

**ANM Invariant:**

```
INV-ACTIVE-NO-MOVES: For any ACTIVE state with currentPlayer P,
  P must have at least one global legal action available.

For temporarily eliminated players:
  IF hasMarkers(P) AND hasBuriedRings(P) THEN
    IF enumerateRecoverySlides(state, P).length > 0 THEN
      NOT_ANM (recovery available)
    ELSE
      Player should be skipped (no legal actions)
```

---

### 2.4 Turn Rotation

**Updated Skip Logic:**

```typescript
function shouldSkipPlayer(state: GameState, playerNumber: number): boolean {
  // Has turn-material (stacks or rings in hand)
  if (playerControlsAnyStack(state.board, playerNumber)) return false;
  if (player.ringsInHand > 0) return false;

  // Check for recovery eligibility AND availability
  if (isEligibleForRecovery(state, playerNumber)) {
    const recoveryMoves = enumerateRecoverySlides(state, playerNumber);
    if (recoveryMoves.length > 0) return false; // Don't skip
  }

  // No turn-material and no recovery → skip
  return true;
}
```

**Implications:**

- Turn rotation must enumerate recovery moves to determine if player is skipped
- Player's "alive" status depends on board geometry (can they form line or fallback slide?) AND buried ring count for line cost
- Multi-player games: "eliminated" players may remain active via recovery

---

### 2.5 Line Processing

**Line Length Requirements:**

| Line Type                    | Normal Line Processing       | Recovery Slide                                          |
| ---------------------------- | ---------------------------- | ------------------------------------------------------- |
| Exactly `lineLength`         | Collapse all, pay 1 ring/cap | **LEGAL** (1 buried ring)                               |
| Overlength (> `lineLength`)  | Option 1 or Option 2         | **LEGAL** (1 + N buried rings, where N = extra markers) |
| Underlength (< `lineLength`) | No collapse                  | **ILLEGAL**                                             |

**Overlength Recovery Cost:**

- Unlike normal play (which offers Option 1/Option 2 choice), recovery always collapses the **entire line**
- Cost scales with line length: `1 + max(0, actualLength - lineLength)` buried ring extractions
- This mirrors normal overlength Option 1 cost (1 ring/cap per extra marker)
- Player must have sufficient buried rings for the slide to be legal

**No Cascading Lines:**
When a line collapses:

1. Markers in line → become collapsed spaces (territory)
2. **Collapsed spaces CANNOT form part of any line**
3. Therefore: Line collapse CANNOT create new lines
4. Recovery → Line → Territory is the only cascade path

**Example - What CANNOT Happen:**

```
INCORRECT ASSUMPTION:
  Recovery completes line of 3 → collapse
  Collapse "reveals" adjacent markers → new line of 4 formed

CORRECT UNDERSTANDING:
  Recovery completes line of 3 → collapse
  Markers become collapsed spaces
  Collapsed spaces are NOT markers → cannot form new line
  Only territory cascade (disconnected regions) is possible
```

---

### 2.6 Territory Cascade

**Recovery → Line → Territory Flow:**

```
1. Recovery Slide (marker A moves to position B)
2. Line of at least lineLength completed (let L = actual length)
3. Line collapses → All L markers become collapsed spaces (territory)
4. Buried ring(s) extracted: 1 + max(0, L - lineLength) → Eliminated (credited to recovering player)
5. Check for disconnected regions created by collapse
6. IF regions found:
   a. For each region player chooses to claim:
      - Interior rings eliminated (credited to player)
      - Self-elimination cost: extract ANOTHER buried ring from stack OUTSIDE region
   b. Continue until no regions chosen or no buried rings remain outside
7. Player now has ring(s) in hand if any buried rings were exhumed during cascade
```

**Outside Stack Requirement:**
Territory self-elimination requires extracting buried ring from stack **outside** the region being claimed:

| Scenario                                     | Outcome                                |
| -------------------------------------------- | -------------------------------------- |
| All buried rings inside region being claimed | Cannot claim that region               |
| Buried rings exist in stacks outside region  | Can claim region (costs 1 buried ring) |
| Buried rings exhausted during cascade        | Must stop claiming regions             |

**Strategic Depth:**

- Claim order matters: small territories first to preserve buried rings?
- May need to leave some regions unclaimed to save buried rings for future recovery
- Number and location of buried rings determines claim budget

---

### 2.7 AI Implementation

**Move Generation:**

```python
def get_legal_moves(state: GameState, player: int) -> List[Move]:
    moves = []

    if player_has_turn_material(state, player):
        moves.extend(get_placement_moves(state, player))
        moves.extend(get_movement_moves(state, player))
        moves.extend(get_capture_moves(state, player))

    if needs_forced_elimination(state, player):
        moves.extend(get_fe_moves(state, player))

    # Recovery (if temporarily eliminated with valid slides)
    if is_eligible_for_recovery(state, player):
        moves.extend(get_recovery_moves(state, player))  # May be empty!

    return moves
```

**Evaluation Heuristics:**

| Factor                                           | Evaluation Impact                                 |
| ------------------------------------------------ | ------------------------------------------------- |
| Being temporarily eliminated with valid recovery | Less negative than fully eliminated               |
| Number of buried rings                           | Positive (recovery fuel + territory claim budget) |
| Marker positioning enabling recovery slides      | Positive (comeback potential)                     |
| Opponent's buried rings in your stacks           | Slight negative (gives them recovery potential)   |

**Search Considerations:**

- Must consider recovery moves even for "eliminated" opponents
- Cannot prune branches for players with markers + buried rings without checking slide validity
- Recovery can dramatically change game state (0 stacks → active with ring in hand)

---

## 3. Game Balance Analysis

### 3.1 Victory Condition Impact

**Projected Shift:**

| Victory Type     | Pre-Recovery | Post-Recovery | Change         |
| ---------------- | ------------ | ------------- | -------------- |
| Ring Elimination | ~40%         | ~42%          | ↑ Slightly     |
| Territory        | ~25%         | ~28%          | ↑ Slightly     |
| LPS              | ~30%         | ~18-22%       | ↓↓ Significant |
| Stalemate        | ~5%          | ~8-10%        | ↑ Slightly     |

**Rationale:**

- LPS decreases: temporarily eliminated players can block it
- Ring/Territory increase: games extend, reaching thresholds more often
- Stalemate increases: more players remain "active" longer, more likely to reach global stalemate

### 3.2 Strategic Dynamics

**New Strategic Elements:**

1. **Buried Ring Valuation**
   - Pre-recovery: "I lost material" (purely negative)
   - Post-recovery: "I have recovery potential" (conditionally positive)

2. **Marker Placement During Active Play**
   - Pre-recovery: For line threats and territory blocking
   - Post-recovery: Also for recovery positioning in losing positions

3. **Capture Decision Complexity**
   - Pre-recovery: Capture to gain material
   - Post-recovery: Capturing gives opponent buried rings (recovery potential)

4. **Endgame Changes**
   - Pre-recovery: Losing all stacks = effectively out
   - Post-recovery: Remain threat until markers OR buried rings exhausted

### 3.3 Player Count Considerations

**2-Player:**

- Recovery primarily helps losing player stay in game
- Dominant player must consider opponent's recovery potential
- "Checkmate-like" goal: eliminate recovery potential (remove markers or buried rings)

**3-4 Player:**

- Temporarily eliminated players can "kingmake" by blocking specific opponent's LPS
- Coalition dynamics: eliminating weakest player's recovery potential removes interference
- More chaotic endgames with multiple players capable of recovery

### 3.4 Balance Concerns

| Concern               | Risk       | Analysis                                                                                                |
| --------------------- | ---------- | ------------------------------------------------------------------------------------------------------- |
| Games never ending    | Low        | Recovery requires ≥lineLength + sufficient buried rings for cost; finite resources eventually exhausted |
| Recovery too powerful | Low        | Requires specific board state; overlength costs more buried rings                                       |
| Recovery too weak     | Medium     | May be rare in practice; monitor via telemetry                                                          |
| Cognitive load        | Medium     | Teaching materials + UI hints critical (especially overlength cost)                                     |
| LPS frustration       | Low-Medium | Dominant players may find LPS harder to close out                                                       |

---

## 4. Edge Cases

| Scenario                                                       | Outcome                                                             |
| -------------------------------------------------------------- | ------------------------------------------------------------------- |
| Recovery slide lands on opponent marker                        | **Illegal** - must land on empty cell                               |
| Recovery creates disconnected region                           | Process line collapse → check regions → territory cascade           |
| Recovery extracts ring, emptying stack                         | Stack removed; may create new disconnected regions                  |
| Player has markers but no valid line-forming or fallback slide | No recovery available; treated as fully eliminated for turn         |
| Overlength line but insufficient buried rings                  | That specific slide is **illegal**; try other slides or no recovery |
| Recovery during line processing phase                          | **Illegal** - recovery only during movement phase                   |
| Multiple buried rings in same stack                            | Can extract from any; bottommost ring of player's colour extracted  |
| Recovery + territory cascade exhausts all buried rings         | Recovery completes; further claims impossible                       |
| Line of 5 with `lineLength = 3`, player has 2 buried rings     | **Illegal** - needs 3 buried rings (1 + 2 overlength)               |

---

## 5. Implementation Checklist

- [ ] Add `recovery_slide` MoveType (TS + Python)
- [ ] Implement `isEligibleForRecovery()` predicate
- [ ] Implement `enumerateRecoverySlides()` - check for (a) lines ≥`lineLength` with sufficient buried rings, OR (b) any fallback slides (including those that disconnect territory)
- [ ] Implement `calculateRecoveryCost(lineLength, actualLineLength)` → number of buried rings required
- [ ] Update `hasAnyRealAction()` for LPS to include recovery
- [ ] Update turn rotation to check recovery availability
- [ ] Integrate into movement phase orchestration
- [ ] Add AI move generation for recovery (consider overlength cost/benefit)
- [ ] Update heuristic evaluation for buried rings (now also fuel for overlength)
- [ ] Add teaching content (TeachingOverlay, tips) - include overlength examples
- [ ] Add telemetry for recovery frequency and overlength usage

---

## 6. Related Documentation

- `RECOVERY_ACTION_IMPLEMENTATION_PLAN.md` - Implementation tasks
- `TODO.md` §3.1.1 - Recovery action task list
- `RULES_CANONICAL_SPEC.md` R110–R115 - Canonical rule definitions
- `ringrift_complete_rules.md` §4.5 - Player-facing rules
- `docs/rules/ACTIVE_NO_MOVES_BEHAVIOUR.md` ANM-SCEN-09 - ANM scenario
- `docs/UX_RULES_CONCEPTS_INDEX.md` - `recovery_action` concept entry
