> **Doc Status (2025-12-14): Partially historical (AI host analysis, Python service only)**
>
> - Role: deep technical assessment of the Python AI microservice (agents, training flows, and Python `GameEngine`) from an earlier architecture revision.
> - Not a semantics or lifecycle SSoT: for rules semantics and lifecycle / API contracts, defer to the shared TypeScript rules engine under `src/shared/engine/**`, the engine contracts under `src/shared/engine/contracts/**`, the v2 contract vectors in `tests/fixtures/contract-vectors/v2/**`, [`RULES_CANONICAL_SPEC.md`](../RULES_CANONICAL_SPEC.md), [`../docs/rules/COMPLETE_RULES.md`](../docs/rules/COMPLETE_RULES.md), [`RULES_ENGINE_ARCHITECTURE.md`](../RULES_ENGINE_ARCHITECTURE.md), [`RULES_IMPLEMENTATION_MAPPING.md`](../RULES_IMPLEMENTATION_MAPPING.md), and [`docs/CANONICAL_ENGINE_API.md`](../docs/CANONICAL_ENGINE_API.md).
> - For the current AI architecture and improvement backlog, prefer [`AI_ARCHITECTURE.md`](../AI_ARCHITECTURE.md), this repository’s live [`AI_IMPROVEMENT_PLAN.md`](./AI_IMPROVEMENT_PLAN.md), and [`docs/AI_TRAINING_AND_DATASETS.md`](../docs/AI_TRAINING_AND_DATASETS.md).
> - Related docs: historical variants in [`archive/AI_ASSESSMENT_REPORT.md`](../archive/AI_ASSESSMENT_REPORT.md) and [`archive/AI_IMPROVEMENT_PLAN.md`](../archive/AI_IMPROVEMENT_PLAN.md), parity and invariants meta-docs such as [`docs/PYTHON_PARITY_REQUIREMENTS.md`](../docs/PYTHON_PARITY_REQUIREMENTS.md), [`docs/STRICT_INVARIANT_SOAKS.md`](../docs/STRICT_INVARIANT_SOAKS.md), [`tests/TEST_SUITE_PARITY_PLAN.md`](../tests/TEST_SUITE_PARITY_PLAN.md), and [`docs/PARITY_SEED_TRIAGE.md`](../docs/PARITY_SEED_TRIAGE.md).
>
> **2025-12-05 Update – Rules & Data Hygiene Progress:**
> Since this report was first written, several of the "Gaps & Simplifications" called out below have been addressed or scoped more narrowly:
>
> - Canonical rules have been tightened in [`RULES_CANONICAL_SPEC.md`](../RULES_CANONICAL_SPEC.md) with:
>   - Explicit line-processing semantics (RR-CANON-R120–R122) requiring **all** line effects (exact-length and overlength) to be driven by explicit `process_line` / `choose_line_reward` decisions.
>   - A new RR-CANON-R074 invariant that all turn actions and voluntary skips must be represented as explicit moves/choices in the game record (no implicit auto-actions).
> - The TS sandbox replay path (`ClientSandboxEngine` in trace mode) and the Python `GameEngine` now agree structurally on representative self-play games when driven from canonical recordings; auto-processing of single lines/territory regions is confined to UX-only paths.
> - A `run_canonical_selfplay_parity_gate.py` driver and `TRAINING_DATA_REGISTRY.md` have been added under `ai-service/` to:
>   - Generate fresh self-play GameReplayDBs per board type.
>   - Gate them on TS↔Python replay parity before they enter training pipelines.
>   - Classify legacy self-play DBs and NN checkpoints as `legacy_noncanonical` so they are **not** used for new training runs.
>     For an up-to-date view of which databases/models are considered canonical, see `ai-service/TRAINING_DATA_REGISTRY.md` and `ai-service/AI_IMPROVEMENT_PLAN.md` §1.3.6–1.3.7.

# AI Service Technical Assessment Report

## 1. Executive Summary

The `ai-service` module is operationally stable, with a functional FastAPI server and passing unit tests. The core AI components (Random, Heuristic, Minimax, MCTS, Descent, and Neural Network) are implemented and integrated. However, the Python-based `GameEngine` used for training and simulation is a re-implementation of the game rules that **currently** simplifies certain complex edge cases (specifically stalemate resolution and granular choices in line processing). These simplifications are no longer treated as acceptable shortcuts: the Python rules engine must be brought into **full parity** with the canonical TypeScript engine so that training and analysis always use spec-accurate rules. The training pipeline also lacks a robust orchestration system for continuous self-improvement.

## 2. Operational Stability

- **Status:** Stable
- **Verification:**
  - `run.sh` correctly sets up the environment and launches `uvicorn`.
  - `pytest` suite passes (23/23 tests), covering parity checks, engine correctness, and AI logic.
  - **Dependencies:** Pinned in `requirements.txt`.
    - _Warning:_ `pydantic` deprecation warnings are present.
    - _Warning:_ `torch.load` usage triggers security warnings regarding `weights_only=False`.

## 3. Architectural Analysis

### 3.1 AI Agents

- **RandomAI / HeuristicAI (baseline engines):**
  - **RandomAI (`random`):** Baseline stochastic policy used for very low difficulties and smoke tests.
  - **HeuristicAI (`heuristic`):** Rule-based evaluation using weighted factors (stack control, territory, mobility). Also used as a fallback evaluator in several agents.
- **MinimaxAI (`minimax`):**
  - **Implementation:** Alpha–beta minimax with move ordering (captures/lines/territory first), quiescence search, and a Zobrist-backed transposition table.
  - **Time behaviour:** Treats `AIConfig.think_time` as a hard search budget (no extra sleep), with explicit time checks in both the main search and quiescence recursion.
  - **Status:** Production-supported for mid–high difficulty levels, but still limited by Python-side state copying and non-incremental hashing.
- **MCTS (Monte Carlo Tree Search, `mcts`):**
  - **Implementation:** Standard PUCT algorithm with RAVE (Rapid Action Value Estimation) and batched neural network evaluation.
  - **Strengths:** Supports batched inference, integrates with the shared Neural Net for policy/value guidance.
  - **Weaknesses:**
    - **State Representation:** The `GameEngine` re-instantiates state objects frequently.
    - **Tree Reuse:** No persistent tree across moves; search starts fresh each request.
- **DescentAI (`descent`):**
  - **Implementation:** Descent / UBFM-style tree search that repeatedly extends the most promising line, logging features/targets for offline training.
  - **Neural Net Integration:** Uses the same shared CNN as MCTS for value/policy guidance where weights are available.
  - **Time behaviour:** Interprets `AIConfig.think_time` as a total search deadline and threads a `deadline` down the recursion so iterations stop conservatively once the budget is exhausted.
- **Neural Network:**
  - **Architecture:** ResNet-style CNN (RingRiftCNN) with adaptive pooling to support multiple board sizes (8x8, 19x19, Hex).
  - **Features:** 10-channel input (stacks, markers, collapsed, liberties, line potential) + global features.
  - **Status:** Implemented but relies on `torch.load` with security warnings. The same network is shared across board types and consumed by both `MCTSAI` and `DescentAI`.

### 3.2 Game Engine (Python)

The `ai-service` maintains its own Python implementation of the game rules (`app/game_engine/__init__.py`) separate from the main TypeScript server.

- **Compliance:**
  - **Placement:** Correctly implements "no-dead-placement" rule.
  - **Movement:** Handles `MOVE_STACK` and `OVERTAKING_CAPTURE`.
  - **Chain Capture:** Implements mandatory chain continuation logic.
  - **Territory:** Implements disconnection checks and self-elimination prerequisites.
- **Gaps & Simplifications (to be eliminated for full parity):**
  - **Stalemate Resolution:** The complex tie-breaking rules for "Global Stalemate" (Section 7.4 of Compact Rules) are not explicitly implemented. The engine currently relies on "no valid moves" to determine a loser, which may not correctly handle the specific "no stacks remain" draw/tiebreaker scenario.
  - **Line Processing (Option 2):** When a line is longer than required, the engine hardcodes the choice to collapse the _first_ 3 markers. A robust AI and training engine must be able to choose _which_ segment to collapse to maximize strategic advantage, matching the TypeScript implementation exactly.
  - **Forced Elimination:** The engine ends the turn immediately after a forced elimination. While compliant with "cycling through players", it relies on the next turn's logic to handle successive eliminations. This behaviour should be reviewed and aligned 1:1 with the TypeScript `GameEngine` so that both engines share identical stalemate and elimination semantics.

### 3.3 Training Pipeline

- **Data Generation:** `generate_data.py` runs self-play games using MCTS.
- **Data Augmentation:** Implements rotation and flipping for board symmetries.
- **Experience Replay:** Simple append-to-file mechanism. Loads the entire dataset into memory (`np.load`), which will become a bottleneck as the dataset grows.
- **Loop:** `train_loop.py` provides a basic skeleton but lacks versioning, automated evaluation (tournament), and model promotion logic.

## 4. Recommendations

### 4.1 Critical Fixes (Short Term)

1.  **Security:** Update `torch.load` calls to use `weights_only=True` or a safer serialization method.
2.  **Stalemate Logic:** Implement the specific tie-breaker logic for Global Stalemate in `GameEngine._check_victory` to ensure the AI learns to play for the tie-breaker when necessary.
3.  **Line Processing:** Expand `Move` structure and `GameEngine` to allow the AI to specify _which_ markers to collapse in Line Option 2, rather than defaulting to the first 3.

### 4.2 Strategic Improvements (Long Term)

1.  **Unified Engine:** Consider exposing the canonical TypeScript engine via a high-performance interface (e.g., Node.js native addon or sidecar) to avoid rule divergence between the Training Engine (Python) and the Game Server (TypeScript).
2.  **Scalable Training:** Refactor `RingRiftDataset` to use a disk-backed dataloader or a proper database (e.g., HDF5, LMDB) instead of loading all `.npy` files into RAM.
3.  **Continuous Integration:** Implement a `Tournament` class that pits the new model checkpoint against the previous best version, automatically promoting it only if it achieves a significant win rate (>55%).

## 5. Conclusion

The `ai-service` is a solid foundation. The MCTS and Neural Net implementations are technically sound. The primary risks lie in the potential for rule divergence in the Python `GameEngine` and the scalability of the data loading pipeline. Addressing the stalemate logic and line processing choices will ensure the AI learns the full depth of RingRift strategy.
