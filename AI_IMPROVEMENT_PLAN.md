# RingRift AI Improvement Plan

This document outlines a comprehensive, actionable plan to improve the AI players in RingRift. The current implementations provide a structural foundation but lack the depth and functionality required for competitive play.

## 1. Core Infrastructure Improvements (Prerequisite)

Before improving specific AI types, the underlying infrastructure for move generation and state simulation in the Python service needs to be robust.

- **Accurate Move Generation:**
  - **Current Status:** Relies on `RandomAI`'s simplified move generation.
  - **Action:** Port the critical logic from the TypeScript `RuleEngine` to Python. Specifically, ensure accurate handling of:
    - Movement constraints (stack height limits, blocking).
    - Capture mechanics (overtaking, chain captures).
    - Placement rules (rings in hand, valid spaces).
  - **Goal:** `get_valid_moves(game_state)` must return _all_ legal moves and _only_ legal moves.

- **State Transition (Simulation):**
  - **Current Status:** Missing. Minimax and MCTS cannot function without applying moves to look ahead.
  - **Action:** Implement a `apply_move(game_state, move) -> new_game_state` function in Python.
  - **Requirements:**
    - Must be efficient (avoid deep copying unnecessary data).
    - Must correctly update board state, player inventories, and turn counters.
    - Must handle "side effects" like captures and territory updates (or at least approximate them sufficiently for evaluation).

## 2. Heuristic AI Improvements

The Heuristic AI is the fallback for most other AIs. Improving it improves everything.

- **Refine Evaluation Function:**
  - **Line Potential:** Implement actual logic to detect partial lines (e.g., 3 or 4 markers in a row/diagonal).
  - **Mobility:** Replace the approximate "unblocked stacks" count with a true count of available moves (once the Move Generator is improved).
  - **Territory:** Improve the territory calculation to detect "eyes" or safe shapes, rather than just raw space count.
- **Dynamic Weights:**
  - **Action:** Adjust weights based on game phase.
    - _Early Game:_ Prioritize expansion and center control.
    - _Mid Game:_ Prioritize stack building and line formation.
    - _Late Game:_ Prioritize victory conditions (territory size, ring elimination).

## 3. Minimax AI Improvements

The current Minimax implementation is a placeholder that performs a depth-1 search.

- **Implement Recursion:**
  - **Action:** Uncomment and fix the `_minimax` recursive method.
  - **Logic:**
    - Base case: Depth 0 or Game Over -> Return `evaluate_position()`.
    - Recursive step: Generate moves -> Apply move -> Call `_minimax` with `depth - 1`.
- **Alpha-Beta Pruning:**
  - **Action:** Ensure alpha-beta values are correctly propagated to prune the search tree.
- **Iterative Deepening:**
  - **Action:** Instead of a fixed depth, run the search with increasing depth limits (1, 2, 3...) until the time limit is approached. This ensures the AI always has a "best move so far" if it runs out of time.
- **Transposition Table (Optional but Recommended):**
  - **Action:** Cache evaluation results for board states (using a Zobrist hash or similar) to avoid re-evaluating the same position reached via different move orders.

## 4. MCTS (Monte Carlo Tree Search) AI Improvements

The current MCTS implementation is a skeleton.

- **Implement the MCTS Loop:**
  - **Selection:** Implement UCT (Upper Confidence Bound for Trees) to balance exploration vs. exploitation.
  - **Expansion:** Add child nodes for valid moves from leaf nodes.
  - **Simulation (Rollout):** Play out the game from the leaf node to a terminal state (or a depth limit) using a lightweight policy (e.g., random moves).
  - **Backpropagation:** Update the win/visit stats up the tree.
- **Heuristic Initialization:**
  - **Action:** Use the Heuristic AI's evaluation to seed the value of new nodes, rather than starting them at zero. This guides the MCTS search toward promising areas faster.

## 5. Neural Network AI Improvements

The current implementation is a simple MLP with no training pipeline integration.

- **Architecture Upgrade:**
  - **Action:** Switch to a Convolutional Neural Network (CNN). RingRift is a grid-based game, making CNNs significantly more effective than fully connected layers.
  - **Input:** Represent the board as an $N \times N \times C$ tensor (where $C$ is channels for Player 1 stacks, Player 2 stacks, Markers, etc.).
- **Model Loading:**
  - **Action:** Implement functionality to load pre-trained PyTorch weights from a file.
- **Data Pipeline:**
  - **Action:** Create a script to record games (from Heuristic/Minimax vs. itself) to generate a dataset for initial training.

## 6. Integration & Testing

- **AI vs. AI Tournament:**
  - Create a script to run matches between different AI types/versions to empirically measure improvement.
- **Performance Profiling:**
  - Measure the time taken for `get_valid_moves` and `evaluate_position` to ensure the AI stays within time limits (e.g., 3 seconds per turn).

## Prioritized Roadmap

1.  **Phase 1: Foundation** - Implement Python `MoveGenerator` and `StateSimulator`.
2.  **Phase 2: Logic** - Fix Minimax recursion and MCTS loop.
3.  **Phase 3: Tuning** - Refine Heuristic weights and evaluation logic.
4.  **Phase 4: Advanced** - Implement CNN and training pipeline.
