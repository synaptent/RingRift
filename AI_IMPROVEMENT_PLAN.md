# RingRift AI Improvement Plan

## 1. Current State Analysis

### AI Service Architecture
- **Microservice**: Python-based FastAPI service (`ai-service`).
- **Communication**: REST API for move selection and evaluation.
- **Integration**: `AIEngine` in backend delegates to `AIServiceClient`.
- **Fallback**: Local heuristics in `AIEngine` if service is unavailable.

### AI Implementations
1. **Random AI**: Baseline for testing and low difficulty.
2. **Heuristic AI**: Rule-based evaluation using weighted factors (stack control, territory, etc.).
3. **MCTS AI**: Monte Carlo Tree Search with PUCT, using Neural Net for evaluation/policy if available.
4. **Neural Net AI**: CNN-based evaluation (value and policy heads).

### Neural Network Status
- **Architecture**: ResNet-style CNN (10 residual blocks).
- **Input**: 10-channel board representation + 10 global features.
- **Output**: Value (scalar) and Policy (probability distribution over ~55k moves).
- **Training**: Basic training loop implemented (`train.py`), but data generation (`generate_data.py`) needs improvement to use self-play with the current best model.

## 2. Identified Issues & Areas for Improvement

### A. Neural Network Training
- **Data Generation**: Currently uses Heuristic/MCTS with hardcoded difficulty. Should evolve to self-play (AlphaZero style).
- **Policy Target**: `generate_data.py` uses a one-hot vector for the selected move. It should ideally use the MCTS visit distribution for a richer training signal.
- **Feature Extraction**: Hardcoded in `NeuralNetAI`. Should be modularized and potentially optimized.

### B. MCTS Implementation
- **Simulation**: Currently uses a short weighted rollout or Neural Net evaluation. Full rollouts are too slow.
- **Parallelism**: MCTS is single-threaded. Python's GIL limits performance.
- **Search Depth**: Limited by Python execution speed.

### C. Integration
- **Latency**: HTTP overhead for every move.
- **State Synchronization**: Full `GameState` is sent in every request. Could be optimized with incremental updates or a stateful session.

## 3. Proposed Improvements

### Phase 1: Enhanced Training Pipeline (Immediate)
1.  **Self-Play Loop**: Update `generate_data.py` to use the latest model for self-play data generation.
2.  **Rich Policy Targets**: Modify MCTS to return visit counts/probabilities instead of just the best move.
3.  **Data Augmentation**: Implement dihedral symmetries (rotation/reflection) for board data to increase dataset size 8x.

### Phase 2: MCTS Optimization (Short-term)
1.  **Batched Inference**: Modify MCTS to evaluate leaf nodes in batches to maximize GPU/CPU throughput.
2.  **Tree Reuse**: Keep the MCTS tree between moves (pruning irrelevant branches) to save search time.

### Phase 3: Architecture Refinement (Medium-term)
1.  **Input Features**: Add history planes (last 3-5 moves) to capture dynamics (e.g., repetition).
2.  **Network Size**: Experiment with smaller/larger networks (MobileNet vs ResNet-50) for different difficulty levels.

### Phase 4: Production Readiness (Long-term)
1.  **Model Versioning**: Implement a system to manage and serve different model versions.
2.  **Async Inference**: Use an async task queue (e.g., Celery/Redis) for heavy AI computations to avoid blocking the API.

## 4. Action Plan

1.  **Fix Imports**: Ensure all scripts run correctly from the root directory (already started).
2.  **Update Data Generation**: Modify `generate_data.py` to support self-play with dynamic model loading.
3.  **Refine MCTS**: Update `MCTSAI` to expose visit distributions.
4.  **Run Training**: Execute a full training cycle (Generate -> Train -> Evaluate).
