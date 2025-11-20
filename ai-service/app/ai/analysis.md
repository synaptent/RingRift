# RingRift AI Analysis & Improvement Plan

## Game Characteristics vs Chess/Go

RingRift has several unique characteristics that distinguish it from Chess and Go, requiring specific AI adaptations:

1.  **Dynamic Board State (Stacks & Markers)**:
    *   Unlike Chess (pieces move) or Go (stones placed), RingRift involves *stacks* that grow/shrink and *markers* that flip ownership.
    *   **Implication**: The branching factor can be high due to placement options and multi-step chain captures. The state space is complex due to stack heights and marker configurations.

2.  **Mandatory Chain Captures**:
    *   Similar to Checkers/Draughts, but with more freedom (can stop if no captures available, but must continue if available).
    *   **Implication**: Search depth can be misleading. A single "move" might involve a long sequence of captures. The AI needs to evaluate the *end* of the chain, not intermediate states. Quiescence search is critical here.

3.  **Territory & Lines**:
    *   Territory formation is similar to Go but dynamic (markers can be flipped/removed).
    *   Lines are like Connect-4 or Gomoku but formed by markers left behind by movement.
    *   **Implication**: Heuristics need to value *potential* lines and territory, not just completed ones. The "influence" heuristic in the current implementation is a good start but could be more sophisticated (e.g., considering connectivity).

4.  **Resource Management (Rings in Hand)**:
    *   Finite resource pool.
    *   **Implication**: AI must balance placing new stacks vs moving existing ones. Running out of rings is a losing condition (elimination).

5.  **Victory Conditions**:
    *   Multiple paths to victory: Ring Elimination, Territory Control, Last Player Standing.
    *   **Implication**: The evaluation function must be a hybrid, weighing progress towards *all* victory conditions dynamically.

## Proposed Improvements

### 1. Enhanced Heuristics (HeuristicAI)

*   **Line Connectivity**: Instead of just counting markers, evaluate "connectedness" or "potential lines". A marker that can easily connect to others is more valuable.
*   **Territory Safety**: Evaluate how "safe" a region is from being invaded or having its border broken.
*   **Stack Mobility**: A stack that can reach many squares is valuable. A stack that is blocked is a liability.
*   **Cap Height Value**: High cap height is defensive (harder to capture) but also offensive (can capture taller stacks). The current heuristic values height, but maybe not cap height specifically enough.

### 2. MCTS Adaptations (MCTSAI)

*   **Playout Policy**: Random playouts are often weak in games with complex tactical sequences (like chain captures).
    *   *Improvement*: Use a "heavy" playout policy that prefers captures and line formations over random moves. This is similar to "decisive moves" in other games.
*   **RAVE (Rapid Action Value Estimation)**: The current implementation has a basic RAVE. We can tune the `rave_k` parameter or refine the move equivalence (e.g., moving *from* a square might be more relevant than the exact move).
*   **Tree Reuse**: Reuse the MCTS tree between moves (pruning the old root). This preserves statistics for relevant subtrees.

### 3. Search Improvements (MinimaxAI)

*   **Quiescence Search**: The current implementation has a basic QS. We should ensure it covers *all* "noisy" moves, specifically:
    *   Chain captures (crucial).
    *   Line formations (drastic state change).
    *   Territory claims.
*   **Move Ordering**: Improve move ordering for Alpha-Beta pruning.
    *   Prioritize captures (MVV-LVA: Most Valuable Victim - Least Valuable Aggressor).
    *   Prioritize moves that form lines.
    *   Use Killer Heuristic (moves that caused a cutoff at the same depth in sibling nodes).

### 4. Specific Rule Adaptations

*   **"No-Dead-Placement"**: The AI must be aware that placing a stack that has no legal moves is illegal. The move generator handles this, but the *evaluation* should penalize placements that have *few* future moves (low mobility).
*   **Forced Elimination**: The AI should avoid states where it is forced to eliminate its own rings due to being blocked. This is a form of "Zugzwang".

## Action Plan

1.  **Refine `HeuristicAI`**: Implement `_evaluate_line_connectivity` and `_evaluate_stack_mobility` with more depth.
2.  **Enhance `MCTSAI`**: Implement a weighted rollout policy (prefer captures/lines).
3.  **Optimize `MinimaxAI`**: Add Killer Heuristic to move ordering.

I will start by implementing the **Killer Heuristic** in `MinimaxAI` and refining the **Move Ordering**, as this usually yields the best performance/strength ratio improvement for Alpha-Beta search.