# RingRift AI Service Technical Assessment

## Executive Summary

The `ai-service` module provides a solid foundation for a self-improving AI agent using Monte Carlo Tree Search (MCTS) and Deep Reinforcement Learning. The core infrastructure for self-play, data generation, and model training is operational. However, significant technical debt exists in the neural network training pipeline, specifically regarding history handling, and the game engine simulation simplifies some complex rule choices.

## 1. Operational Stability

*   **Status:** **Stable**
*   **Verification:** The environment setup (`setup.sh`, `run.sh`) and dependencies are correct. Unit tests for MCTS (`tests/test_mcts_ai.py`) are passing after fixing a mocking issue.
*   **Issues:**
    *   A runtime warning regarding model architecture mismatch exists but is handled gracefully (fallback to fresh weights).
    *   `pytest` invocation requires `python -m pytest` to correctly resolve imports.

## 2. AI Agent Architecture

### MCTS Agent (`mcts_ai.py`)
*   **Strengths:** Implements PUCT (Predictor + Upper Confidence Bound applied to Trees) with RAVE (Rapid Action Value Estimation) heuristics. Supports batched inference for efficiency.
*   **Weaknesses:**
    *   The fallback rollout policy (when no neural net is available) is a simple weighted random selection, which may be too weak for effective bootstrapping.
    *   Tree reuse is partially implemented but commented out or limited by the stateless API nature.

### Neural Network (`neural_net.py`)
*   **Strengths:** Uses a ResNet-style CNN architecture with adaptive pooling to support variable board sizes. Includes both value and policy heads.
*   **Weaknesses:**
    *   **Architecture Mismatch:** The saved model checkpoint (`ringrift_v1.pth`) has a different architecture (likely older) than the current code, causing a reset to random weights on load.
    *   **History Handling:** The model expects stacked history frames (current + 3 previous), but the training pipeline fails to provide them correctly (see below).

### Heuristic AI (`heuristic_ai.py`)
*   **Strengths:** Comprehensive set of weighted heuristics covering stack control, territory, mobility, and victory proximity.
*   **Weaknesses:** Relies on `RandomAI` for some helper methods, creating a slightly odd dependency structure.

## 3. Training Pipeline & Self-Improvement Loop

### Data Generation (`generate_data.py`)
*   **Status:** **Functional but Flawed**
*   **Critical Issue:** The feature extraction logic saves only the *current* state features (10 channels) to the dataset, ignoring the history buffer.
*   **Data Augmentation:** Rotation/flipping is implemented but skipped for policy training due to complexity in rotating move indices. This reduces data efficiency.

### Training Script (`train.py`)
*   **Status:** **Functional with Hacks**
*   **Critical Issue:** To compensate for the missing history in the dataset, the `RingRiftDataset` class duplicates the current frame 4 times to match the model's expected input shape (40 channels). **This effectively disables temporal learning**, preventing the AI from understanding move sequences or repetition.

### Self-Play Loop (`train_loop.py`)
*   **Status:** **Operational**
*   **Mechanism:** Iteratively generates data using MCTS (guided by the current model) and retrains the model.
*   **Experience Replay:** Implements a simple FIFO buffer (last 50k samples).

## 4. Rules Completeness (`game_engine.py`)

*   **Status:** **Mostly Complete**
*   **Implemented:** Ring placement, movement, capturing (including chains), line formation, territory claiming, forced elimination, and victory conditions.
*   **Simplifications:**
    *   **Line Formation:** Automatically chooses to collapse all markers and eliminate from the largest stack. In the actual game, players can choose "Option 2" (minimum collapse, no elimination). This simplification biases the AI against strategies that preserve board presence.
    *   **Territory Claim:** Automatically claims territory without user choice nuances.

## 5. Recommendations

### Priority 1: Fix Training Pipeline History
1.  **Update `generate_data.py`:** Ensure that `game_history` stores the full stacked feature tensor (40 channels) or that the dataset saves the sequence of states so `train.py` can reconstruct history on the fly.
2.  **Remove Hack in `train.py`:** Once data generation is fixed, remove the code that duplicates the current frame.

### Priority 2: Resolve Architecture Mismatch
1.  **Retrain from Scratch:** Given the architecture change and the history bug, the existing checkpoint is likely invalid or suboptimal. Start a fresh training run with the fixed pipeline.
2.  **Version Control:** Implement strict versioning for model architectures to prevent silent failures or mismatches.

### Priority 3: Enhance Game Engine
1.  **Implement Choices:** Update `GameEngine` to support branching for Line Formation and Territory Claim choices (e.g., return multiple valid moves for the same action but different options).
2.  **Policy Rotation:** Implement proper policy vector rotation in `augment_data` to enable data augmentation for policy training.

### Priority 4: Optimization
1.  **Parallel Self-Play:** The current generation is sequential. Implement parallel workers (using `multiprocessing` or Ray) to generate games faster.
2.  **Tree Reuse:** Implement a mechanism to serialize/cache the MCTS tree between moves in a game to improve search depth.
