# RingRift Rules Analysis: Termination & Tie-Breaking

**Date:** November 20, 2025
**Analyst:** Architect Mode

> **Purpose:** Supporting mathematical analysis of the RingRift rules, with a focus on the S‑invariant, termination, and tie‑breaking behaviour. This document explains _why_ the rules in [`ringrift_complete_rules.md`](ringrift_complete_rules.md) and [`ringrift_compact_rules.md`](ringrift_compact_rules.md) guarantee finite games and unique winners, but it is not itself a canonical rules specification.
> **Audience:** Engine implementers, AI authors, and rules maintainers who need to reason about progress and termination.
> **Relationship to other docs:** For authoritative rules semantics, see [`ringrift_complete_rules.md`](ringrift_complete_rules.md) and [`ringrift_compact_rules.md`](ringrift_compact_rules.md). For how specific rules/FAQ examples map into executable Jest suites, see [`RULES_SCENARIO_MATRIX.md`](RULES_SCENARIO_MATRIX.md).

## 1. Executive Summary

The RingRift rule set provides **strong mathematical guarantees** for both game termination and unique winner determination.

- **Termination** is guaranteed by a strictly increasing "Progress Metric" ($S$) bounded by finite board resources, coupled with a "No-Dead-Placement" rule that prevents infinite loops of non-progressive actions.
- **Tie-Breaking** is guaranteed to resolve to a unique winner through a hierarchical set of criteria, culminating in a temporal check ("Last Player to Move") that cannot be tied in a sequential turn-based game.

---

## 2. Termination Analysis

### 2.1 The Progress Metric ($S$)

As defined in the rules, we track the scalar value:
$$S = M + C + E$$
Where:

- $M$ = Total Markers on board
- $C$ = Total Collapsed spaces
- $E$ = Total Eliminated rings (all players)

**Theorem:** Every valid turn that involves a board interaction strictly increases $S$.

### 2.2 Proof by Move Type

| Action Type                   | Impact on $M$ | Impact on $C$ | Impact on $E$ | Net $\Delta S$ | Notes                                                                                    |
| :---------------------------- | :-----------: | :-----------: | :-----------: | :------------: | :--------------------------------------------------------------------------------------- |
| **Placement**                 |       0       |       0       |       0       |       0        | _See Section 2.3 below_                                                                  |
| **Standard Move**             |      +1       |       0       |       0       |     **+1**     | Leaves marker at start.                                                                  |
| **Move (Land on Own Marker)** |       0       |       0       |      +1       |     **+1**     | Start: +1M. Land: -1M, +1E.                                                              |
| **Overtaking Capture**        |      +1       |       0       |       0       |     **+1**     | Leaves marker at start. Captured ring stays in play ($E$ unchanged).                     |
| **Capture (Land on Marker)**  |       0       |       0       |      +1       |     **+1**     | Start: +1M. Land: -1M (marker removed), +1E (attacker cap top ring).                     |
| **Line Collapse**             |     -$k$      |     +$k$      |      +1       |     **+1**     | Markers convert to Collapsed ($\Delta M + \Delta C = 0$). Ring eliminated ($+1 E$).      |
| **Territory Disconnection**   |     -$k$      |     +$k$      |   +$r$ + 1    |   **+$r$+1**   | Markers convert ($\Delta M + \Delta C = 0$). Rings inside ($r$) + Self ($1$) eliminated. |
| **Forced Elimination**        |       0       |       0       |    +$cap$     |   **+$cap$**   | At least 1 ring eliminated.                                                              |
| **Recovery Slide**            |      +1       |       0       |       0       |     **+1**     | Leaves marker at start cell. Line processing may follow.                                 |

### 2.2.1 Recovery Action ($\Delta S$ Proof)

The **Recovery Action** (`RR-CANON-R110–R115`) is available when a player has:

- Zero stacks on board
- Zero rings in hand
- At least one marker AND at least one buried ring (opponent's stack over their ring)

A recovery slide moves a marker to complete a line. This action:

- Always leaves a marker at the start cell: $\Delta M = +1$
- May trigger line collapse (adds $+1$ to $E$), further increasing $S$

Since $\Delta S \geq +1$, recovery actions maintain the termination guarantee.

### 2.3 The Placement Loophole Closure

Placement itself does not increase $S$. However, the rules prevent "infinite placement" or "placement without progress" through two constraints:

1.  **Finite Resource:** Placement consumes `ringsInHand`. This is a strictly decreasing resource.
2.  **No-Dead-Placement Rule:** A player cannot place a ring unless that specific stack can immediately make a legal move.
    - _Consequence:_ Every turn involving placement **must** also involve a movement.
    - Since Movement increases $S$ (see table), **every turn involving placement increases $S$.**

### 2.4 The "Pass" Condition

If a player has no rings in hand, no stacks on board, or is otherwise unable to act (and has no stacks to force-eliminate), they must **Pass** (forfeit turn).

- $\Delta S = 0$.
- **Termination Condition:** If _all_ players Pass consecutively, the game ends immediately (Global Stalemate).

### 2.5 Conclusion on Termination

Since $S$ is strictly increasing for all active turns and bounded by $2 \times \text{Spaces} + \text{TotalRings}$, the number of active turns is finite. Since consecutive passes end the game, the game must terminate in finite time.

---

## 3. Tie-Breaking Analysis

In the event of a Global Stalemate (no legal moves for any player), the winner is determined by a strict hierarchy. We analyze this for uniqueness.

**Hierarchy:**

1.  **Most Collapsed Spaces** ($C_p$)
2.  **Most Eliminated Rings** ($E_p$) \*
    - _Note: Includes rings remaining in hand converted to eliminated._
3.  **Most Markers** ($M_p$)
4.  **Last Player to Complete a Valid Turn Action**

### 3.1 Uniqueness Proof

Criteria 1, 2, and 3 are state-based and can theoretically be tied (e.g., Player A and B both have 10 spaces, 5 eliminated, 2 markers).

**Criterion 4 is the "Sovereign Breaker":**

- The game is **sequential** and **turn-based**.
- Actions are discrete.
- For the game to reach a state where "No player can move", there must have been a transition from "Someone could move" to "No one can move".
- That transition was caused by exactly **one** player performing the final legal action (Movement, Capture, or Forced Elimination) that exhausted the last possibility for play (e.g., eliminating the last stack).
- Therefore, the "Last Player" is always unique.

**Edge Case: Game Starts in Stalemate?**

- If the board is empty and players have rings, Placement is legal.
- Therefore, at least one action must occur for the game to start.
- "Last Player" is always defined.

### 3.2 Strategic Implications

The tie-breaking order reinforces the game's identity:

1.  **Territory First:** Controlling the board is the primary goal.
2.  **Attrition Second:** Eliminating enemies (and emptying your hand) is secondary.
3.  **Presence Third:** Having markers (potential territory) is tertiary.
4.  **Tempo Last:** Being the one to "close" the game is the final resort.

## 4. Verification of Edge Cases

| Scenario              | Outcome                                                                                                                                                                                                                                                                                                                                                                           | Consistent? |
| :-------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------: |
| **Cyclic Captures**   | Allowed, but bounded by **finite capturable targets**. See Section 4.1 for deep analysis. Each capture consumes one target stack; the chain terminates when no more legal captures exist.                                                                                                                                                                                         |     ✅      |
| **180° Reversals**    | Allowed, but bounded by the **same three factors as cyclic captures** (see Section 4.1): (1) finite capturable targets, (2) cap depletion → control change via marker landings, and (3) board geometry constraints. Each reversal leaves a departure marker at each endpoint; landing on markers eliminates rings from attacker's cap until control changes or no targets remain. |     ✅      |
| **"Do Nothing" Turn** | Impossible. If you have stacks, you _must_ Move or Force Eliminate. If you have no stacks but have rings, you _must_ Place (and then Move). Only true "Pass" is when you have NO options, which leads to game end if universal.                                                                                                                                                   |     ✅      |

### 4.1 Deep Analysis: Cyclic Capture and 180° Reversal Termination

This section provides rigorous analysis of why chain captures (including cyclic captures and 180° reversals) are always bounded, even when marker landings offset height gains. Both cyclic captures (returning to previously-visited positions via multi-leg paths) and 180° reversals (back-and-forth captures along a line) are governed by **the same three termination mechanisms**.

**The question arises:** Can a chain capture cycle indefinitely if marker landings offset height gains?

#### 4.1.1 The Zero Net Height Growth Scenario

Consider a capture where:

- Attacker captures 1 ring (added to BOTTOM of stack): $\Delta H_{\text{capture}} = +1$
- Attacker lands on a marker (TOP ring of cap eliminated): $\Delta H_{\text{marker}} = -1$
- Net height change: $\Delta H = 0$

This appears to allow indefinite cycling. However, **termination is still guaranteed** via three mechanisms:

#### 4.1.2 Termination Mechanism 1: Finite Capturable Targets

**Key insight:** Captures target **stacks**, not markers. Each capture segment consumes exactly one enemy stack.

- A chain capture can only continue while there exists a legal capture from the current position
- Each capture removes one target stack from the board (the captured rings transfer to the attacker)
- Target stacks do not regenerate during a turn
- Therefore: Maximum chain length ≤ number of capturable stacks on board at chain start

**Proof:** Let $T$ = number of opponent stacks reachable via consecutive captures. After $T$ captures, no targets remain. The chain must terminate. $\square$

#### 4.1.3 Termination Mechanism 2: Control Change (Mixed-Color Stacks)

When P's stack captures Q's rings and lands on markers:

**Stack composition after capture:**

```
[P's cap rings (top)] + [Q's captured rings (bottom)]
```

**Marker landing effect:**

- Eliminates the TOP ring of cap (one of P's rings)
- If all of P's cap rings are eliminated, Q's captured ring becomes the new top
- Stack control transfers to Q
- P can no longer attack with this stack

**Example trace:**

1. P's stack: `[P, P]` (height 2, controlled by P)
2. P captures Q's single ring at distance 2
3. Stack: `[P, P, Q]` (height 3, controlled by P)
4. Landing on marker: eliminate top P → `[P, Q]` (controlled by P)
5. P captures Q's single ring at distance 2
6. Stack: `[P, Q, Q]` (height 3, controlled by P)
7. Landing on marker: eliminate top P → `[Q, Q]` (controlled by Q!)
8. P's chain terminates — the stack is no longer P's to control

**Termination bound:** P's original cap ring count bounds how many marker landings can occur before control changes.

#### 4.1.4 Termination Mechanism 3: Same-Color Analysis

**Q: What if all rings (attacker + targets) belong to the same player?**

This scenario is **impossible by game rules**:

- Players have distinct ring colors
- Captures target **opponent** stacks (controlled by different player)
- A player cannot capture their own stacks

The only way for an attacker's stack to contain only same-colored rings is if no captures have occurred (pure movement). Once captures begin, the stack necessarily contains mixed colors (attacker's cap + victim's rings).

**Conclusion:** The "same-color infinite cycle" scenario cannot arise.

#### 4.1.5 Path Geometry Constraints

Even with zero net height change, the **physical board constrains cyclic paths**:

1. **Departure markers accumulate:**
   - Each capture segment deposits a departure marker
   - These markers occupy cells along the path
   - Landing on these markers causes cap eliminations (accelerating control change)

2. **Movement direction constraints:**
   - Movement/capture must proceed in straight orthogonal or diagonal lines
   - Returning to a previously-visited position requires either:
     - A 180° reversal (deposits marker at each endpoint, causing eliminations)
     - A complex multi-leg path (each leg deposits markers)

3. **Board size limit:**
   - Even if height stays constant, the stack cannot reach targets beyond `BoardDiagonal`
   - Markers filling the board progressively reduce traversable paths

#### 4.1.6 Summary: Why Infinite Cycles Cannot Occur

| Factor             | Bound          | Primary Effect                              |
| ------------------ | -------------- | ------------------------------------------- |
| **Finite targets** | $T$ stacks     | Chain length ≤ $T$                          |
| **Cap depletion**  | $C$ cap rings  | ≤ $C$ marker landings before control change |
| **Board geometry** | $B$ board size | Max traversable distance = $B$              |

**Theorem:** For any chain capture sequence, termination occurs in at most $\min(T, C, B)$ capture segments, where:

- $T$ = initial count of reachable opponent stacks
- $C$ = attacker's initial cap ring count
- $B$ = board size parameter

Since all three quantities are finite and positive, **infinite cyclic captures are impossible**. $\square$

---

## 5. Final Verdict

The RingRift rule set is **robust**.

- **Infinite loops are impossible.**
- **Draws are impossible.**
- **Termination is guaranteed.**
