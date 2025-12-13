# AI Multiplayer Search (3P/4P)

This document tracks the current multi-player (3–4 player) search semantics for
the Python AI service and why they exist, so work is not duplicated and future
refactors stay debuggable.

## Goal

Replace “2-player only” fallbacks (1-ply heuristic) with a **stronger, consistent
multi-player search model** that remains easy to reason about while the
multi-player neural training pipeline matures.

## Current Approach: Paranoid Reduction (Root vs Coalition)

For 3P/4P games, search AIs use a classic **Paranoid** reduction:

- The AI’s `player_number` is the sole maximizing player (“root”).
- **All other players are treated as a single minimizing coalition.**

This is not a perfect model of multi-player incentives, but it is a large step
up from 1‑ply heuristics and keeps search behavior explainable and testable.

Key invariant:

- **Sign/role changes are driven by `current_player`**, not by depth parity.
  This matters because RingRift can legally yield consecutive actions by the
  same side (e.g., chain captures), and in 3P/4P multiple opponents can act
  consecutively while still being on the same coalition “side”.

## Implemented AIs

### MinimaxAI

- File: `ai-service/app/ai/minimax_ai.py`
- Multi-player: enabled via Paranoid semantics (root maximizes; any opponent
  minimizes), using `game_state.current_player == self.player_number` at each
  node.
- Leaf evaluation:
  - Uses NNUE when enabled and available; otherwise heuristic.
  - NNUE is instantiated with the inferred player count; if no `nnue_*_{3p,4p}`
    checkpoint exists, it safely falls back to heuristic evaluation.

### DescentAI (UBFM-style)

- File: `ai-service/app/ai/descent_ai.py`
- Multi-player: enabled via Paranoid semantics (root maximizes; any opponent
  minimizes).
- Neural evaluation:
  - Enabled for 3P/4P using the same Paranoid reduction as 2P, with
    NeuralNetAI framed against the **most threatening opponent**.
    (Still scalar‑head / 2P‑trained checkpoints; see limitations.)
- Heuristic scalarization:
  - Uses a “most threatening opponent” selector based on **victory progress**
    (max of territory-control, ring-elimination, and LPS proximity) so Paranoid
    search reacts to whichever canonical win path is closest.
  - LPS proximity respects the per-game `lpsRoundsRequired` threshold.

### MCTSAI

- File: `ai-service/app/ai/mcts_ai.py`
- Multi-player: enabled via Paranoid semantics.
- Neural evaluation:
  - Enabled for 3P/4P with Paranoid sign/backprop semantics and threat‑opponent
    framing in NeuralNetAI.
- Backpropagation semantics (important):
  - Values are treated as **side-to-move** (root vs coalition) values.
  - **Sign flips only when the turn switches between root and coalition**, not
    on every ply. This prevents incorrect value inversion when opponents act
    consecutively (3P/4P) or when the same player acts repeatedly (chain
    capture).

## Tests

- `ai-service/tests/test_multiplayer_ai_search.py`
  - Ensures Minimax/Descent/MCTS route to search (not 1‑ply fallback) for 3P.
  - Ensures MCTS backprop does **not** flip sign between two different opponent
    turns (coalition stays the same).
- `ai-service/tests/test_swap_search_ai.py`
  - Ensures search AIs can take the swap (pie rule) meta‑move in 2P.

## Known Limitations / Next Steps

1. **True multi-player neural evaluation**
   - Current 3P/4P search uses scalar head‑0 values with threat‑opponent
     globals. The long‑term target is a stable multi-player encoder +
     per‑player value interpretation (rank dist / MaxN‑style utilities) with
     canonical 3P/4P checkpoints.
2. **Threat modeling**
   - Victory-progress threat selection is now used for scalarization.
     Next step is to thread the same selector into neural-backed 3P/4P search
     once multi-player NN value heads are promoted.
3. **MaxN / vector-valued search**
   - Paranoid is a pragmatic interim. A MaxN-style value vector (or rank-aware
     scalar utility) would be the principled long-term approach once the value
     model is reliable and debuggable.
   - **Staged plan:** after the v2 multi-player encoder + **vector value head**
     are trained end-to-end on canonical data, add MaxN as an **optional**
     search mode. Keep Paranoid as the default for human-facing play and
     robustness against coalition/leader‑punish behaviour.
   - **Promotion criteria:** only promote MaxN for a board/player‑count tier if
     canonical tournaments show a clear strength win without introducing
     stalling or parity/correctness regressions.
4. **Best‑Reply Search (BRS) / Best‑Reply Reduction**
   - Potential intermediate between Paranoid and full MaxN: each opponent ply
     expands only that player’s single “best reply” (according to their own
     utility), rather than a full opponent branching factor.
   - **Why not now:** BRS is extremely evaluation‑dependent; with today’s scalar
     heuristics/threat proxies it is unstable and will under‑model implicit
     coalitions common in 3P/4P RingRift.
   - **When to add:** after vector utilities exist. Treat as a
     **config‑gated experiment** for large boards if MaxN is too expensive but
     Paranoid is too pessimistic. Promote only if tournaments show a durable
     strength/cost advantage.
