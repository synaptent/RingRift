> **Doc Status (2025-11-27): Archived (historical AI host code review, Python + client sandbox)**
>
> - Role: historical code review and optimization report for the Python AI microservice (`ai-service`) and the client-side sandbox AI. Retained for context on past refactors and performance work; not a live plan.
> - Superseded by: the active AI host docs [`AI_ARCHITECTURE.md`](../AI_ARCHITECTURE.md), [`ai-service/AI_ASSESSMENT_REPORT.md`](../ai-service/AI_ASSESSMENT_REPORT.md), [`ai-service/AI_IMPROVEMENT_PLAN.md`](../ai-service/AI_IMPROVEMENT_PLAN.md), and the training/meta docs [`docs/AI_TRAINING_AND_DATASETS.md`](../docs/AI_TRAINING_AND_DATASETS.md) and [`docs/AI_TRAINING_PREPARATION_GUIDE.md`](../docs/AI_TRAINING_PREPARATION_GUIDE.md).
> - Not a semantics or lifecycle SSoT: for rules semantics and lifecycle / API contracts, defer to the shared TypeScript rules engine under `src/shared/engine/**`, the engine contracts under `src/shared/engine/contracts/**`, the v2 contract vectors in `tests/fixtures/contract-vectors/v2/**`, [`RULES_CANONICAL_SPEC.md`](../RULES_CANONICAL_SPEC.md), [`ringrift_complete_rules.md`](../ringrift_complete_rules.md), [`RULES_ENGINE_ARCHITECTURE.md`](../RULES_ENGINE_ARCHITECTURE.md), [`RULES_IMPLEMENTATION_MAPPING.md`](../RULES_IMPLEMENTATION_MAPPING.md), and [`docs/CANONICAL_ENGINE_API.md`](../docs/CANONICAL_ENGINE_API.md).
> - Related docs: other archived AI-era reports in `archive/` (e.g. [`AI_ASSESSMENT_REPORT.md`](./AI_ASSESSMENT_REPORT.md), [`AI_IMPROVEMENT_PLAN.md`](./AI_IMPROVEMENT_PLAN.md), tournament summaries `AI_TOURNAMENT_RESULTS*.md`, and stall/debug write-ups `AI_STALL_*`).

# RingRift AI Engine Code Review & Optimization Report

## Executive Summary

This report provides a comprehensive analysis of the current AI implementations in the RingRift project, covering the Python-based AI Service (`HeuristicAI`, `MinimaxAI`, `MCTSAI`) and the Client-side Sandbox AI.

**Overall Assessment:**
Significant progress has been made in optimizing the AI engine. The critical performance bottleneck in `HeuristicAI` (mobility evaluation) has been resolved, and `MinimaxAI` now uses safer time management and improved move ordering. `MCTSAI` has been upgraded with batched inference. However, the Zobrist hashing implementation remains suboptimal (O(N) instead of O(1)), and state copying in MCTS still incurs overhead.

---

## 1. Heuristic AI (`ai-service/app/ai/heuristic_ai.py`)

### Status: Improved

### Completed Optimizations

- **Mobility Evaluation:** The expensive `get_valid_moves` call in `_evaluate_mobility` has been replaced with a lightweight "pseudo-mobility" heuristic. This iterates over stacks and checks adjacent positions directly, drastically reducing the computational cost of evaluating leaf nodes.
- **Partial Geometry Centralization:** The code now utilizes `BoardGeometry.get_adjacent_positions`, moving towards a centralized geometry logic.

### Remaining Issues

- **Hardcoded Weights:** Evaluation weights remain hardcoded class attributes, making dynamic tuning difficult.
- **Redundant Logic:** Some line-of-sight logic (`_get_visible_stacks`) still duplicates game engine logic.

### Future Recommendations

1.  **Dynamic Configuration:** Move weights to a configuration file or `AIConfig` object.
2.  **Full Geometry Centralization:** Refactor `_get_visible_stacks` to use `BoardGeometry`.

---

## 2. Minimax AI (`ai-service/app/ai/minimax_ai.py`)

### Status: Significantly Improved

### Completed Optimizations

- **Safe Time Management:** A time check has been added inside the recursive `_minimax` loop (every 1000 nodes), preventing Time-to-Move violations during deep searches.
- **Enhanced Move Ordering:** Moves are now sorted using lightweight heuristics (MVV-LVA approximation, noisy move priority) instead of full state evaluation, reducing the overhead at each node.
- **Optimized Quiescence Search:** The QS implementation now filters for "noisy" moves (`overtaking_capture`, `chain_capture`, etc.) and uses a simplified evaluation, mitigating the horizon effect without excessive cost.

### Critical Issue: Zobrist Hashing Performance

- **Current State:** While `ZobristHash` is integrated, the `_get_state_hash` method calls `compute_initial_hash`, which iterates over the entire board (O(N)) at every node.
- **Impact:** This negates much of the performance benefit of the transposition table, as hashing becomes a dominant cost.
- **Required Fix:** Implement **incremental hashing**. The hash should be updated in O(1) time during `apply_move` by XORing the features of the changed board positions, rather than recomputing from scratch.

---

## 3. MCTS AI (`ai-service/app/ai/mcts_ai.py`)

### Status: Improved

### Completed Optimizations

- **Batched Inference:** The implementation now collects leaf nodes and evaluates them in batches using the Neural Network. This significantly improves throughput, especially when using GPU acceleration.
- **Reduced Copying Overhead:** The explicit `copy.deepcopy` has been removed from the main loop. It now relies on `GameEngine.apply_move`, which performs a manual shallow copy with selective deep copying.

### Remaining Issues

- **State Copying Cost:** While better than `deepcopy`, `GameEngine.apply_move` still creates a new `GameState` object and deep copies mutable dictionaries (stacks, markers) for every simulation step. This remains an O(N) operation.
- **No Tree Reuse:** The MCTS tree is rebuilt from scratch for every move, discarding valuable statistics from previous searches.

### Future Recommendations

1.  **Incremental State Updates:** Refactor `GameEngine` or create a specialized `FastGameEngine` for MCTS that supports `do_move` / `undo_move` (in-place modification) to eliminate copying overhead.
2.  **Tree Persistence:** Implement a mechanism to cache and reuse the MCTS tree between moves, re-rooting it to the new game state.

---

## 4. Client Sandbox AI (`src/client/sandbox/sandboxAI.ts`)

### Status: Functional & Robust

### Completed Optimizations

- **Heuristic Strategy:** The `evaluateMove` function now includes basic heuristics (prioritizing captures, line formations, and territory claims), making the AI a more useful opponent for testing.
- **Structured Logic:** The AI now correctly handles different game phases (`ring_placement`, `line_processing`, `movement`) with specific logic for each, ensuring valid play.

---

## Final Verdict

The RingRift AI engine has moved from a "naive" implementation to a "structured and optimized" state. The most critical blocking issues (mobility performance, unsafe time limits) have been addressed.

**Next Priority:**
The immediate technical priority is fixing the **Zobrist Hashing implementation in MinimaxAI** to be truly incremental (O(1)). This is the single largest remaining "low-hanging fruit" for performance. Following that, implementing **Tree Reuse in MCTS** and **In-place State Updates** will be key for reaching high-level play strength.
