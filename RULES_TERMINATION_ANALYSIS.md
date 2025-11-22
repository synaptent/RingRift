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
| **Line Collapse**             |     -$k$      |     +$k$      |      +1       |     **+1**     | Markers convert to Collapsed ($\Delta M + \Delta C = 0$). Ring eliminated ($+1 E$).      |
| **Territory Disconnection**   |     -$k$      |     +$k$      |   +$r$ + 1    |   **+$r$+1**   | Markers convert ($\Delta M + \Delta C = 0$). Rings inside ($r$) + Self ($1$) eliminated. |
| **Forced Elimination**        |       0       |       0       |    +$cap$     |   **+$cap$**   | At least 1 ring eliminated.                                                              |

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

| Scenario              | Outcome                                                                                                                                                                                                                         | Consistent? |
| :-------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :---------: |
| **Cyclic Captures**   | Allowed, but limited by stack height (must move $\ge H$). $H$ increases with every capture. Eventually $H > \text{BoardSize}$, making moves impossible.                                                                         |     ✅      |
| **180° Reversals**    | Allowed, but leaves markers. Markers eventually fill the line or stack height limits movement.                                                                                                                                  |     ✅      |
| **"Do Nothing" Turn** | Impossible. If you have stacks, you _must_ Move or Force Eliminate. If you have no stacks but have rings, you _must_ Place (and then Move). Only true "Pass" is when you have NO options, which leads to game end if universal. |     ✅      |

## 5. Final Verdict

The RingRift rule set is **robust**.

- **Infinite loops are impossible.**
- **Draws are impossible.**
- **Termination is guaranteed.**
