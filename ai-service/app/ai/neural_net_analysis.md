# Neural Network AI Analysis

## Current Implementation Status

1.  **Integration**:
    *   The `NeuralNetAI` class is implemented in `ai-service/app/ai/neural_net.py`.
    *   It is integrated into the main application in `ai-service/app/main.py` via the `_create_ai_instance` factory.
    *   The `MCTSAI` class in `ai-service/app/ai/mcts_ai.py` attempts to use `NeuralNetAI` for evaluation if available.
    *   The TypeScript backend (`src/server/game/ai/AIEngine.ts`) and client (`src/server/services/AIServiceClient.ts`) are set up to request AI moves, including those from the neural network (via the `MCTS` type which uses it).

2.  **Architecture (`RingRiftCNN`)**:
    *   **Input**: 10-channel board representation (stacks, markers, collapsed spaces, liberties, line potential) + 10 global features (phase, rings in hand, eliminated rings, turn).
    *   **Backbone**: Residual Network (ResNet) with configurable number of blocks (default 10) and filters (default 128).
    *   **Heads**:
        *   **Value Head**: Outputs a scalar value (-1 to 1) representing the position evaluation.
        *   **Policy Head**: Outputs a probability distribution over ~55,000 possible moves.

3.  **Training (`ai-service/app/training/train.py`)**:
    *   **Data Loading**: Loads training data from a `.npy` file.
    *   **Loss Function**: MSE Loss for value prediction. **Missing Policy Loss**.
    *   **Optimizer**: Adam.
    *   **Loop**: Standard supervised learning loop.

## Identified Issues & Areas for Improvement

1.  **Missing Policy Training**:
    *   The current training script only calculates loss for the value head (`criterion = nn.MSELoss()`). The policy head output is ignored during training.
    *   **Fix**: Add Cross-Entropy Loss for the policy head and combine it with the value loss.

2.  **Data Generation**:
    *   The current training script generates dummy random data if the file is missing.
    *   **Improvement**: We need a robust self-play data generation script that uses MCTS (with the current best model or heuristics) to generate high-quality training games.

3.  **Model Architecture**:
    *   The policy head output size (~55,000) is very large and sparse.
    *   **Improvement**: Consider a more structured policy head (e.g., separate heads for move type, start position, end position) or masking invalid moves during training/inference. However, for now, the flat policy is standard for AlphaZero-like approaches, provided we mask invalid moves.

4.  **Input Features**:
    *   The current feature extraction is decent but could be enriched.
    *   **Improvement**: Add history planes (previous board states) to capture dynamics (though RingRift is mostly Markovian, history helps with repetition rules if any). Add "legal move" mask to the input or as a separate input to the policy head.

5.  **MCTS Integration**:
    *   The `MCTSAI` uses the neural net value but has a placeholder for policy priors.
    *   **Fix**: Ensure the policy output from the network is correctly mapped to the MCTS node priors.

## Proposed Plan

1.  **Update Training Script**: Implement a proper AlphaZero-style loss function (Value MSE + Policy Cross-Entropy + L2 Regularization).
2.  **Implement Self-Play Data Generation**: Create a script to run self-play games using MCTS and save the data (state, policy, value) for training.
3.  **Refine MCTS-Net Integration**: Ensure the policy logits are correctly used in the MCTS selection phase.