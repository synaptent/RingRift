# Recovery Action Interactions Analysis

> **Doc Status (2025-12-10): Active (derived analysis)**
>
> **Purpose:** Detailed analysis of how recovery action interacts with other game systems and its impact on game balance.
>
> **Canonical Source:** `RULES_CANONICAL_SPEC.md` R110–R115, `ringrift_complete_rules.md` §4.5
>
> **Note:** This document includes some historical balance projections from early recovery work. Treat the numerical projections as non-normative; the rules semantics must match `RULES_CANONICAL_SPEC.md`.

---

## 1. Recovery Action Summary

**Eligibility:** A player is eligible for recovery if ALL conditions hold:

1. They control **no stacks** on the board
2. They have at least one **marker** on the board
3. They have **buried rings** in opponent-controlled stacks
4. **Note:** Eligibility is independent of rings in hand (rings in hand do not prevent recovery).

**Action:** Slide one marker to an adjacent destination cell. Legal if **either**:

- **(a) Line formation:** Completes a line of **at least** `lineLength` consecutive markers
- **(b) Fallback-class recovery:** If no line-forming recovery slide exists anywhere on the board, one of the following adjacent recovery actions is permitted:
  - **(b1) Fallback repositioning:** Slide to an adjacent empty cell (including slides that cause territory disconnection).
  - **(b2) Stack-strike:** Slide onto an adjacent stack; the marker is removed from play and the attacked stack's top ring is eliminated and credited to the recovering player.

**Overlength Lines:** Overlength lines (longer than `lineLength`) **are permitted**. When an overlength line is formed, the player chooses:

- **Option 1:** Collapse all markers in the line to territory. Cost: 1 buried ring extraction.
- **Option 2:** Collapse exactly `lineLength` consecutive markers of the player's choice. Cost: 0 (no extraction required).

This mirrors normal line reward semantics (RR-CANON-R130–R134).

**Effect:**

1. If `recoveryMode == 'line'`: collapse markers to Territory per the chosen Option 1/2.
2. Self-elimination cost:
   - Option 1 (exact-length or chosen for overlength): 1 buried ring extracted → eliminated (credited to recovering player).
   - Option 2 (overlength only): 0 buried rings extracted.
   - Fallback/stack-strike: 1 buried ring extracted → eliminated.
3. If the recovery slide (line or fallback) creates disconnected regions → territory cascade processing (paid via buried ring extraction per claimed region).
4. Recovery does **not** restore rings to hand: extracted buried rings are eliminated as cost; cascade processing may eliminate rings/award territory, but rings do not return to hand.

**Critical Rule:** **Collapsed spaces never form part of a valid line.** Only markers can form lines.

---

## 2. Detailed System Interactions

### 2.1 LPS (Last Player Standing)

**Classification:** Recovery is **NOT** a "real action" for LPS purposes.

```
Real Actions = {placement, non-capture_movement, overtaking_capture}
NOT Real Actions = {recovery_slide, forced_elimination, skip_placement, no_* bookkeeping moves}
```

This creates strategic tension: recovery can keep a player "alive" in turn rotation without preventing LPS; only placements, non-capture movements, and overtaking captures count as real actions.

**Impact on LPS Victory:**

| Scenario                                            | Without Recovery   | With Recovery                                                                                                                                             |
| --------------------------------------------------- | ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Player A dominates, Player B has only FE            | B cannot block LPS | B cannot block LPS (FE ≠ real action)                                                                                                                     |
| Player A dominates, Player B has recovery available | N/A                | B **cannot** block LPS with recovery (not real)                                                                                                           |
| Player B has recovery + rings in hand               | N/A                | B can block LPS if they have a legal placement/movement/capture available at the start of their turn (rings in hand often implies placement is available) |
| Only A has real actions for 2 rounds                | A wins LPS         | A wins LPS (recovery doesn't count)                                                                                                                       |

**Strategic Implications:**

- LPS victory is achievable against players with only markers + buried rings (no rings in hand)
- Players with rings in hand often retain ring-placement availability (a real action) and can therefore block LPS; once rings in hand run out and stacks are immobilised, recovery alone does not stop LPS.
- Recovery provides survival but not LPS defense; creates interesting resource management

**Example:**

```
Turn 100: A has 5 stacks, B has 0 stacks, 0 rings in hand
          B has 3 markers and 2 buried rings in A's stacks
          A: Takes real action → LPS round 1 begins (B has no real actions; may still have recovery)

Question: Does B have a valid recovery slide?
- If YES: B still has **no real actions** (recovery ≠ real), but B is not ANM and is not skipped (recovery provides a global legal action); LPS can still proceed.
- If NO valid slide exists: B truly has no real actions, LPS proceeds
```

**Key Insight:** Turn rotation must not skip temporarily eliminated players (buried rings still exist); recovery eligibility determines whether they have recovery moves available in `movement`. LPS real-action evaluation must **not** count recovery as a real action.

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
    → Loses all controlled stacks
        ├─ No rings anywhere → Permanently eliminated (skipped)
        └─ Still has rings somewhere (rings in hand and/or buried)
            ├─ Recovery-eligible (marker + buried ring) → may attempt recovery in `movement` (or skip recovery)
            └─ Not recovery-eligible or no legal recovery slide → forced no-op turns recorded via bookkeeping moves until state changes or global stalemate ends the game
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
  recovery_slides / skip_recovery (if recovery-eligible and in movement),
  line_decisions,
  territory_decisions
}
```

**Critical Distinction:** Having markers + buried rings is NECESSARY but NOT SUFFICIENT for recovery:

- Must also have either (a) a valid marker slide that completes at least `lineLength` (line recovery), OR (b) when no line-forming slide exists anywhere on the board, a fallback-class recovery action (adjacent reposition or stack-strike).
- If no such slide exists, the player has no recovery move available; this does **not** imply permanent elimination or turn skipping.

**ANM Invariant:**

```
INV-ACTIVE-NO-MOVES: For any ACTIVE state with currentPlayer P,
  P must have at least one global legal action available.

ANM is defined only for players with turn-material (RR‑CANON‑R202).
Players who lack turn-material but still have rings somewhere are not ANM; they remain in rotation and traverse phases via explicit bookkeeping moves, optionally taking recovery actions when eligible.
```

---

### 2.4 Turn Rotation

**Updated Skip Logic (RR‑CANON‑R201):**

```typescript
function shouldSkipPlayer(state: GameState, playerNumber: number): boolean {
  // Skip only permanently eliminated players (no rings anywhere).
  return !playerHasAnyRings(state, playerNumber);
}
```

**Implications:**

- Turn rotation must enumerate recovery moves to determine if player is skipped
- Player's "alive" status depends on board geometry (can they form line or fallback slide?) AND buried ring count for line cost
- Multi-player games: "eliminated" players may remain active via recovery

---

### 2.5 Line Processing

**Line Length Requirements:**

| Line Type                    | Normal Line Processing                  | Recovery Slide                                   |
| ---------------------------- | --------------------------------------- | ------------------------------------------------ |
| Exactly `lineLength`         | Collapse all, pay 1 ring from any stack | **LEGAL** (1 buried ring)                        |
| Overlength (> `lineLength`)  | Option 1 or Option 2                    | **LEGAL** (Option 1: 1 buried ring; Option 2: 0) |
| Underlength (< `lineLength`) | No collapse                             | **ILLEGAL**                                      |

**Line Elimination Cost (Normal Play):** Eliminate ONE ring from the top of any controlled stack (including standalone rings). Any controlled stack is an eligible target.

**Overlength Recovery Cost:**

- Recovery line mode uses the same Option 1 / Option 2 semantics as normal line rewards:
  - Option 1: collapse all markers in the line; cost is 1 buried ring extraction.
  - Option 2 (overlength only): collapse exactly `lineLength` markers; cost is 0.
- Exact-length recovery lines implicitly use Option 1.

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
2. If line mode: a line of at least lineLength is completed; Option 1/2 collapse is applied
3. Line self-elimination cost: Option 1 extracts 1 buried ring; Option 2 extracts 0
4. Check for disconnected regions created by the recovery slide (line or fallback)
5. IF regions found:
   a. For each region player chooses to claim:
      - Interior rings eliminated (credited to player)
      - Self-elimination cost: extract ANOTHER buried ring from stack OUTSIDE region
   b. Continue until no regions chosen or no buried rings remain outside
6. Note: extracted buried rings are eliminated; they do not return to hand
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

    # Recovery (if recovery-eligible; may still be empty if no slide is legal)
    if is_eligible_for_recovery(state, player):
        moves.extend(get_recovery_moves(state, player))  # May be empty!

    return moves
```

**Evaluation Heuristics:**

| Factor                                           | Evaluation Impact                                 |
| ------------------------------------------------ | ------------------------------------------------- |
| Being temporarily eliminated with valid recovery | Less negative than permanently eliminated         |
| Number of buried rings                           | Positive (recovery fuel + territory claim budget) |
| Marker positioning enabling recovery slides      | Positive (comeback potential)                     |
| Opponent's buried rings in your stacks           | Slight negative (gives them recovery potential)   |

**Search Considerations:**

- Must consider recovery moves even for "eliminated" opponents
- Cannot prune branches for players with markers + buried rings without checking slide validity
- Recovery can change game state even from zero stacks (marker relocation, eliminations, and potential territory claims)

---

## 3. Game Balance Analysis

### 3.1 Victory Condition Impact

**Projected Shift (historical / illustrative):**

| Victory Type     | Pre-Recovery | Post-Recovery | Change         |
| ---------------- | ------------ | ------------- | -------------- |
| Ring Elimination | ~40%         | ~42%          | ↑ Slightly     |
| Territory        | ~25%         | ~28%          | ↑ Slightly     |
| LPS              | ~30%         | ~18-22%       | ↓↓ Significant |
| Stalemate        | ~5%          | ~8-10%        | ↑ Slightly     |

**Rationale:**

- Recovery is **not** a real action for LPS, so recovery-only turns do not block LPS on their own; verify the net LPS mix via self-play.
- Game-length and victory-mix effects are best measured via the self-play statistics analyzer and recent distributed runs rather than treated as fixed projections.

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

- Temporarily eliminated players remain in rotation; recovery itself does **not** block LPS (not a real action), but regained placements/moves later can
- Coalition dynamics: eliminating weakest player's recovery potential removes interference
- More chaotic endgames with multiple players capable of recovery

### 3.4 Balance Concerns

| Concern               | Risk       | Analysis                                                                                                                                                         |
| --------------------- | ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Games never ending    | Low        | Recovery consumes finite buried-ring resources and/or triggers eliminations/territory; together with FE and RR-CANON termination invariants, progress is bounded |
| Recovery too powerful | Low        | Requires specific board state (marker geometry + buried rings); monitor via self-play telemetry                                                                  |
| Recovery too weak     | Medium     | May be rare in practice; monitor via telemetry                                                                                                                   |
| Cognitive load        | Medium     | Teaching materials + UI hints critical (eligibility, global fallback rule, and Option 1/2 choice)                                                                |
| LPS frustration       | Low-Medium | Dominant players may find LPS harder to close out                                                                                                                |

---

## 4. Edge Cases

| Scenario                                                       | Outcome                                                                                            |
| -------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| Recovery slide lands on any marker                             | **Illegal** - destination may not contain a marker                                                 |
| Recovery slide lands on a stack                                | **Legal only** for `recoveryMode == 'stack_strike'` (marker removed; attacked top ring eliminated) |
| Recovery creates disconnected region                           | Process line collapse → check regions → territory cascade                                          |
| Recovery extracts ring, emptying stack                         | Stack removed; may create new disconnected regions                                                 |
| Player has markers but no valid line-forming or fallback slide | No recovery move available; player remains in rotation unless permanently eliminated               |
| Recovery during line processing phase                          | **Illegal** - recovery only during movement phase                                                  |
| Multiple buried rings in same stack                            | Can extract from any; bottommost ring of player's colour extracted                                 |
| Recovery + territory cascade exhausts all buried rings         | Recovery completes; further claims impossible                                                      |
| Overlength line recovery                                       | Legal; Option 1 costs 1 buried ring extraction, Option 2 costs 0                                   |

---

## 5. Implementation Checklist

- [x] Add recovery MoveTypes (TS + Python): `recovery_slide`, `skip_recovery`
- [x] Implement eligibility + enumerate + apply (TS SSoT): `src/shared/engine/aggregates/RecoveryAggregate.ts`
- [x] Implement Python mirror semantics + parity coverage: `ai-service/app/rules/recovery.py`, `ai-service/tests/parity/test_recovery_parity.py`
- [x] Ensure LPS real-action detection excludes recovery (RR‑CANON‑R172)
- [x] Ensure turn rotation skips only permanently eliminated players (RR‑CANON‑R201)
- [ ] Add/extend teaching content and UX hints (see `docs/ux/UX_RULES_TEACHING_GAP_ANALYSIS.md`)

---

## 6. Related Documentation

- `RECOVERY_ACTION_IMPLEMENTATION_PLAN.md` - Implementation tasks
- `TODO.md` §3.1.1 - Recovery action task list
- `RULES_CANONICAL_SPEC.md` R110–R115 - Canonical rule definitions
- `ringrift_complete_rules.md` §4.5 - Player-facing rules
- `docs/rules/ACTIVE_NO_MOVES_BEHAVIOUR.md` ANM-SCEN-09 - ANM scenario
- `docs/UX_RULES_CONCEPTS_INDEX.md` - `recovery_action` concept entry
