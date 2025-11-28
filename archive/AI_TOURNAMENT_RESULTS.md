> **Doc Status (2025-11-27): Archived (historical AI tournament results)**
>
> - Role: record of early AI tournament baselines (Random vs Heuristic vs MCTS vs Minimax) during initial development.
> - Superseded by: the active AI host docs [`AI_ARCHITECTURE.md`](../AI_ARCHITECTURE.md) and [`docs/AI_TRAINING_AND_DATASETS.md`](../docs/AI_TRAINING_AND_DATASETS.md) which describe current capabilities.
> - Not a semantics SSoT: see [`RULES_CANONICAL_SPEC.md`](../RULES_CANONICAL_SPEC.md) and [`docs/CANONICAL_ENGINE_API.md`](../docs/CANONICAL_ENGINE_API.md).
> - Related docs: [`archive/AI_TOURNAMENT_RESULTS_PHASE2.md`](./AI_TOURNAMENT_RESULTS_PHASE2.md), [`archive/AI_ASSESSMENT_REPORT.md`](./AI_ASSESSMENT_REPORT.md).

# AI Tournament Results (Baseline)

## Summary

The baseline tournament results are surprising and indicate significant issues with the "advanced" AIs (Minimax and MCTS). The Random AI performed best, which strongly suggests that the evaluation functions or search logic in the advanced AIs are flawed or that they are running out of time/crashing (though no crashes were observed in the logs).

## Results Table

| AI Type   | Wins | Losses | Draws | Score |
| --------- | ---- | ------ | ----- | ----- |
| Random    | 5    | ?      | ?     | 5     |
| Heuristic | 4    | ?      | ?     | 4     |
| MCTS      | 2    | ?      | ?     | 2     |
| Minimax   | 1    | ?      | ?     | 1     |

_Note: The score is total wins. The tournament structure was a round-robin with 2 games per matchup._

## Matchup Analysis

### Heuristic vs Random

- **Result:** Heuristic won 2-0.
- **Observation:** Heuristic AI seems to have basic competence over Random. It can form lines and claim territory.

### Minimax vs Random

- **Result:** Random won 2-0 (inferred from total score).
- **Observation:** Minimax lost to Random. This is a critical failure. It suggests Minimax is either:
  1.  Evaluating positions incorrectly (preferring losing states).
  2.  Not searching deep enough to see threats.
  3.  Pruning incorrectly.
  4.  Running out of time and defaulting to a bad move (though it should default to _some_ move).

### MCTS vs Random

- **Result:** Random won 2-0 (inferred).
- **Observation:** MCTS also lost to Random. Likely due to insufficient simulations or a poor default policy (Random) that doesn't guide the search effectively in the complex RingRift state space.

### Minimax vs Heuristic

- **Result:** Heuristic won 2-0 (inferred).
- **Observation:** The static evaluation of Heuristic AI beat the lookahead of Minimax. This confirms Minimax is actually _worse_ than just using the evaluation function directly (which Heuristic does). This points to a bug in the Minimax recursion or state update logic.

### MCTS vs Heuristic

- **Result:** Heuristic won 2-0.
- **Observation:** MCTS failed against Heuristic.

### MCTS vs Minimax

- **Result:** MCTS won 2-0.
- **Observation:** The battle of the broken AIs. MCTS won, possibly because Minimax is actively choosing bad moves.

## Diagnosis & Hypotheses

1.  **Minimax Recursion Bug:** The `_minimax` function might be flipping the maximizing/minimizing player incorrectly, causing it to choose moves that help the opponent.
2.  **State Mutation:** The `GameEngine.apply_move` might be mutating the state in a way that affects the parent state in the search tree, despite `copy.deepcopy`. (Checked code: `apply_move` does `deepcopy`, so this is less likely, but worth verifying).
3.  **Search Depth:** Minimax depth might be too shallow (1 or 2) to see anything useful, and the overhead of search prevents it from even doing the basic heuristic check effectively if it times out.
4.  **MCTS Simulation:** The random rollout in MCTS is likely too noisy to provide a reliable signal for RingRift, which requires precise sequences (chains).

## Action Plan

1.  **Fix Minimax:**
    - Verify the `maximizing_player` logic in `_minimax`.
    - Ensure `evaluate_position` returns scores from the perspective of the _maximizing player_ (or always Player 1, and Minimax adjusts). Currently `HeuristicAI.evaluate_position` returns positive for "this AI".
    - **CRITICAL:** In `MinimaxAI._minimax`, when calling recursively, we need to ensure we know _whose turn it is_ in the next state to set `maximizing_player` correctly.

2.  **Improve Heuristic:**
    - Since Heuristic is the only one winning against Random, its evaluation function is the "gold standard" right now. We should refine it further.

3.  **Fix MCTS:**
    - Increase simulation count/time.
    - Use Heuristic for rollout instead of Random (heavy, but better signal).
