# AI Service Technical Assessment Report

## 1. Executive Summary
The `ai-service` module is operationally stable, with a functional FastAPI server and passing unit tests. The core AI components (MCTS, Neural Network, Heuristic) are implemented and integrated. However, the Python-based `GameEngine` used for training and simulation is a re-implementation of the game rules that simplifies certain complex edge cases (specifically stalemate resolution and granular choices in line processing). While sufficient for heuristic training, these simplifications may lead to strategy divergence from the canonical TypeScript engine. The training pipeline lacks a robust orchestration system for continuous self-improvement.

## 2. Operational Stability
- **Status:** Stable
- **Verification:**
  - `run.sh` correctly sets up the environment and launches `uvicorn`.
  - `pytest` suite passes (23/23 tests), covering parity checks, engine correctness, and AI logic.
  - **Dependencies:** Pinned in `requirements.txt`.
    - *Warning:* `pydantic` deprecation warnings are present.
    - *Warning:* `torch.load` usage triggers security warnings regarding `weights_only=False`.

## 3. Architectural Analysis

### 3.1 AI Agents
- **MCTS (Monte Carlo Tree Search):**
  - **Implementation:** Standard PUCT algorithm with RAVE (Rapid Action Value Estimation) and batched neural network evaluation.
  - **Strengths:** Supports batched inference, integrates with Neural Net for policy/value guidance.
  - **Weaknesses:**
    - **State Representation:** The `GameEngine` re-instantiates state objects frequently.
    - **Tree Reuse:** No persistent tree across moves; search starts fresh each request.
- **Neural Network:**
  - **Architecture:** ResNet-style CNN (RingRiftCNN) with adaptive pooling to support multiple board sizes (8x8, 19x19, Hex).
  - **Features:** 10-channel input (stacks, markers, collapsed, liberties, line potential) + global features.
  - **Status:** Implemented but relies on `torch.load` with security warnings.

### 3.2 Game Engine (Python)
The `ai-service` maintains its own Python implementation of the game rules (`app/game_engine.py`) separate from the main TypeScript server.
- **Compliance:**
  - **Placement:** Correctly implements "no-dead-placement" rule.
  - **Movement:** Handles `MOVE_STACK` and `OVERTAKING_CAPTURE`.
  - **Chain Capture:** Implements mandatory chain continuation logic.
  - **Territory:** Implements disconnection checks and self-elimination prerequisites.
- **Gaps & Simplifications:**
  - **Stalemate Resolution:** The complex tie-breaking rules for "Global Stalemate" (Section 7.4 of Compact Rules) are not explicitly implemented. The engine relies on "no valid moves" to determine a loser, which may not correctly handle the specific "no stacks remain" draw/tiebreaker scenario.
  - **Line Processing (Option 2):** When a line is longer than required, the engine hardcodes the choice to collapse the *first* 3 markers. A robust AI should be able to choose *which* segment to collapse to maximize strategic advantage.
  - **Forced Elimination:** The engine ends the turn immediately after a forced elimination. While compliant with "cycling through players", it relies on the next turn's logic to handle successive eliminations, which is acceptable but relies on the loop controller.

### 3.3 Training Pipeline
- **Data Generation:** `generate_data.py` runs self-play games using MCTS.
- **Data Augmentation:** Implements rotation and flipping for board symmetries.
- **Experience Replay:** Simple append-to-file mechanism. Loads the entire dataset into memory (`np.load`), which will become a bottleneck as the dataset grows.
- **Loop:** `train_loop.py` provides a basic skeleton but lacks versioning, automated evaluation (tournament), and model promotion logic.

## 4. Recommendations

### 4.1 Critical Fixes (Short Term)
1.  **Security:** Update `torch.load` calls to use `weights_only=True` or a safer serialization method.
2.  **Stalemate Logic:** Implement the specific tie-breaker logic for Global Stalemate in `GameEngine._check_victory` to ensure the AI learns to play for the tie-breaker when necessary.
3.  **Line Processing:** Expand `Move` structure and `GameEngine` to allow the AI to specify *which* markers to collapse in Line Option 2, rather than defaulting to the first 3.

### 4.2 Strategic Improvements (Long Term)
1.  **Unified Engine:** Consider exposing the canonical TypeScript engine via a high-performance interface (e.g., Node.js native addon or sidecar) to avoid rule divergence between the Training Engine (Python) and the Game Server (TypeScript).
2.  **Scalable Training:** Refactor `RingRiftDataset` to use a disk-backed dataloader or a proper database (e.g., HDF5, LMDB) instead of loading all `.npy` files into RAM.
3.  **Continuous Integration:** Implement a `Tournament` class that pits the new model checkpoint against the previous best version, automatically promoting it only if it achieves a significant win rate (>55%).

## 5. Conclusion
The `ai-service` is a solid foundation. The MCTS and Neural Net implementations are technically sound. The primary risks lie in the potential for rule divergence in the Python `GameEngine` and the scalability of the data loading pipeline. Addressing the stalemate logic and line processing choices will ensure the AI learns the full depth of RingRift strategy.