# Analysis of "A Simple AlphaZero" (arXiv:2008.01188v4)

## Key Insights

The paper "Learning to Play Two-Player Perfect-Information Games without Knowledge" (arXiv:2008.01188v4) proposes several techniques to improve reinforcement learning in games like Hex, surpassing Mohex 3HNN without expert knowledge.

1.  **Tree Learning (Tree Bootstrapping)**: Instead of learning only from the root value or the terminal value, the paper suggests learning from the values of all nodes in the search tree. This maximizes data efficiency.
2.  **Descent Search**: A modification of Unbounded Best-First Minimax (UBFM) called "Descent". It extends the best sequence of actions to terminal states during the learning process. This generates better quality data (closer to terminal values) for learning.
3.  **Reinforcement Heuristics**: Replacing the simple +1/-1 game outcome with richer heuristics like:
    *   **Depth Heuristic**: Rewards quick wins and slow defeats. Value = $P - p + 1$ (where $P$ is max moves, $p$ is current moves).
    *   **Scoring**: Using the game score (e.g., piece count difference) as the target value.
    *   **Mobility**: Maximizing available moves.
4.  **Ordinal Distribution**: A new action selection probability distribution based on the *rank* of the action values rather than their absolute values (like Softmax). This makes exploration more robust to value scale.
5.  **Completion**: A technique to handle resolved states (proven wins/losses) correctly during search, ensuring the agent prefers a proven win over a heuristic high value.

## Applicability to RingRift

1.  **Tree Learning**: Highly applicable. We are currently moving towards AlphaZero-style training which uses the root visit distribution (a form of tree learning). We can further enhance this by learning from subtree values if we use a minimax-based approach, but for MCTS, the visit distribution is the standard equivalent.
2.  **Descent Search**: This is an alternative to MCTS. Since we have already invested in MCTS, switching to Descent might be a larger architectural change. However, the core idea—extending rollouts to terminal states more often or using a "best-first" expansion—is valuable. Our current MCTS implementation is standard UCT.
3.  **Reinforcement Heuristics**: **Very applicable and easy to implement.** RingRift has a clear "score" (rings eliminated, territory). Using a "Depth Heuristic" (win fast) or "Score Heuristic" (maximize territory/rings) as the training target instead of just Win/Loss could significantly speed up learning.
    *   *Action*: Modify the reward signal in `generate_data.py` to include a depth penalty or score component.
4.  **Ordinal Distribution**: Applicable as an alternative to Softmax/PUCT exploration.
5.  **Completion**: Relevant for end-games. RingRift can have sudden death victories. Ensuring the AI respects "proven" paths is crucial.

## Selected Improvements for Implementation

We will focus on **Reinforcement Heuristics** as it provides a high impact-to-effort ratio and directly addresses the "sparse reward" problem in complex games.

**Plan:**
1.  **Implement Depth Heuristic**: Modify the reward function to be $R = \pm (1 + \lambda \cdot (MaxMoves - CurrentMove))$.
2.  **Implement Score Heuristic**: Include the "rings eliminated" count in the final reward.