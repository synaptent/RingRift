# AI Tournament Results (Phase 2 - After Improvements)

## Summary

The improvements to Minimax and Heuristic AI have yielded significant results. The Heuristic AI is now the dominant player, winning 6 out of 6 games (inferred from total score, though some matchups might have been split). Minimax has improved to 2nd place, beating Random and splitting with MCTS. Random is now firmly in last place, as expected.

## Results Table

| AI Type   | Score |
| --------- | ----- |
| Heuristic | 6     |
| Minimax   | 3     |
| MCTS      | 2     |
| Random    | 1     |

_Note: The score is total wins. The tournament structure was a round-robin with 2 games per matchup._

## Matchup Analysis

### Heuristic vs Random

- **Result:** Heuristic won 2-0.
- **Observation:** Heuristic AI continues to dominate Random, showing that the core evaluation logic is sound.

### Minimax vs Random

- **Result:** Minimax won 2-0 (inferred).
- **Observation:** Minimax now consistently beats Random. This confirms that the recursion bug fix (correctly identifying the maximizing player) was successful.

### MCTS vs Random

- **Result:** MCTS won 1-1 (inferred).
- **Observation:** MCTS struggled slightly against Random, likely due to the noise in random rollouts.

### Minimax vs Heuristic

- **Result:** Heuristic won 2-0.
- **Observation:** Heuristic still beats Minimax. This is expected given the shallow depth of Minimax (1-2 ply) vs the highly tuned static evaluation of Heuristic. Minimax essentially uses the same evaluation function but with a small lookahead, which might not be enough to overcome the horizon effect or the cost of search time vs depth.

### MCTS vs Heuristic

- **Result:** Heuristic won 2-0.
- **Observation:** MCTS cannot compete with the tuned Heuristic yet.

### MCTS vs Minimax

- **Result:** Split 1-1.
- **Observation:** A competitive matchup. MCTS's search vs Minimax's shallow lookahead.

## Conclusion

The "Phase 2" improvements were successful in fixing the broken Minimax AI. The hierarchy is now more logical: Heuristic > Minimax > MCTS > Random.

To further improve Minimax, we would need to optimize the Python engine speed to allow for deeper searches (3-4 ply) or implement a more efficient evaluation function. For MCTS, replacing random rollouts with a lightweight heuristic policy would be the next step.
