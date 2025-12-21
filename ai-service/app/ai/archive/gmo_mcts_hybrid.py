"""GMO-MCTS Hybrid - Use GMO for move ordering in MCTS.

This module combines GMO's gradient-based move scoring with MCTS tree search.
GMO provides informed prior probabilities for MCTS, potentially improving search
efficiency by guiding exploration toward promising moves.

Key features:
1. GMO provides prior policy for MCTS
2. GMO uncertainty guides exploration vs exploitation
3. Optional GMO-based rollout policy
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import torch

from ..models import AIConfig, GameState, Move
from .base import BaseAI
from .gmo_ai import GMOAI, GMOConfig, estimate_uncertainty

logger = logging.getLogger(__name__)


@dataclass
class GMOMCTSConfig:
    """Configuration for GMO-MCTS hybrid."""
    # MCTS parameters
    num_simulations: int = 100
    c_puct: float = 1.5  # Exploration constant
    temperature: float = 1.0

    # GMO integration
    use_gmo_prior: bool = True  # Use GMO scores as MCTS prior
    use_gmo_rollout: bool = False  # Use GMO in rollouts (slower but more accurate)
    gmo_prior_weight: float = 0.7  # How much to weight GMO vs uniform prior
    uncertainty_exploration_bonus: float = 0.3  # Bonus for uncertain moves

    # GMO config
    gmo_config: GMOConfig | None = None

    # Device
    device: str = "cpu"


class MCTSNode:
    """Node in the MCTS search tree."""

    def __init__(
        self,
        state: GameState,
        parent: MCTSNode | None = None,
        move: Move | None = None,
        prior: float = 1.0,
    ):
        self.state = state
        self.parent = parent
        self.move = move  # Move that led to this node
        self.prior = prior

        self.children: dict[str, MCTSNode] = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False

    @property
    def q_value(self) -> float:
        """Average value of this node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, c_puct: float, parent_visits: int) -> float:
        """Upper Confidence Bound score for selection."""
        if self.visit_count == 0:
            return float('inf')

        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return self.q_value + exploration


class GMOMCTSHybrid(BaseAI):
    """GMO-MCTS Hybrid AI.

    Combines GMO's gradient-based move scoring with MCTS tree search.
    GMO provides prior probabilities to guide the MCTS search toward
    promising regions of the game tree.
    """

    def __init__(
        self,
        player_number: int,
        config: AIConfig,
        hybrid_config: GMOMCTSConfig | None = None,
    ):
        super().__init__(player_number, config)

        self.hybrid_config = hybrid_config or GMOMCTSConfig()
        self.device = torch.device(self.hybrid_config.device)

        # Initialize GMO for prior computation
        gmo_config = self.hybrid_config.gmo_config or GMOConfig(device=self.hybrid_config.device)
        self.gmo = GMOAI(
            player_number=player_number,
            config=config,
            gmo_config=gmo_config,
        )

    def load_gmo_checkpoint(self, checkpoint_path: str) -> None:
        """Load GMO model weights."""
        self.gmo.load_checkpoint(checkpoint_path)

    def _get_gmo_priors(
        self,
        state: GameState,
        legal_moves: list[Move],
    ) -> dict[str, float]:
        """Get GMO-based prior probabilities for moves.

        Returns dict mapping move strings to prior probabilities.
        """
        if not self.hybrid_config.use_gmo_prior:
            # Uniform prior
            uniform = 1.0 / len(legal_moves)
            return {self._move_key(m): uniform for m in legal_moves}

        # Get GMO scores for all moves
        scores = []
        with torch.no_grad():
            state_embed = self.gmo.state_encoder.encode_state(state)

            for move in legal_moves:
                move_embed = self.gmo.move_encoder.encode_move(move)

                # Get value + uncertainty score
                mean_val, _, var = estimate_uncertainty(
                    state_embed,
                    move_embed,
                    self.gmo.value_net,
                    self.gmo.gmo_config.mc_samples,
                )
                novelty = self.gmo.novelty_tracker.compute_novelty(move_embed)

                score = (
                    mean_val +
                    self.hybrid_config.uncertainty_exploration_bonus * torch.sqrt(var + 1e-8) +
                    self.gmo.gmo_config.gamma * novelty
                )
                scores.append(score.item())

        # Convert to probabilities via softmax
        scores = np.array(scores)
        exp_scores = np.exp(scores - scores.max())  # Numerical stability
        probs = exp_scores / exp_scores.sum()

        # Blend with uniform prior for robustness
        uniform = 1.0 / len(legal_moves)
        w = self.hybrid_config.gmo_prior_weight
        blended_probs = w * probs + (1 - w) * uniform

        return {self._move_key(m): p for m, p in zip(legal_moves, blended_probs, strict=False)}

    def _move_key(self, move: Move) -> str:
        """Create unique string key for a move."""
        from_key = move.from_pos.to_key() if move.from_pos else "none"
        to_key = move.to.to_key() if move.to else "none"
        return f"{move.type.value}_{from_key}_{to_key}"

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """Select child with highest UCB score."""
        best_score = float('-inf')
        best_child = None

        for child in node.children.values():
            score = child.ucb_score(
                self.hybrid_config.c_puct,
                node.visit_count,
            )
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def _expand(
        self,
        node: MCTSNode,
        legal_moves: list[Move],
        priors: dict[str, float],
    ) -> None:
        """Expand a node with all legal moves."""
        for move in legal_moves:
            move_key = self._move_key(move)
            new_state = self.rules_engine.apply_move(node.state, move)

            child = MCTSNode(
                state=new_state,
                parent=node,
                move=move,
                prior=priors.get(move_key, 1.0 / len(legal_moves)),
            )
            node.children[move_key] = child

        node.is_expanded = True

    def _simulate(self, state: GameState, max_moves: int = 50) -> float:
        """Run simulation from state to terminal.

        Returns value from perspective of current player.
        """
        from ..models import GameStatus

        current_state = state
        starting_player = state.current_player

        for _ in range(max_moves):
            if current_state.game_status != GameStatus.ACTIVE:
                break

            current_player = current_state.current_player
            legal_moves = self.rules_engine.get_valid_moves(current_state, current_player)
            if not legal_moves:
                break

            # Use GMO for rollout policy if enabled, otherwise random
            if self.hybrid_config.use_gmo_rollout:
                self.gmo.player_number = current_player
                move = self.gmo.select_move(current_state)
            else:
                move = np.random.choice(legal_moves)

            if move is None:
                break

            current_state = self.rules_engine.apply_move(current_state, move)

        # Evaluate terminal state
        if current_state.game_status == GameStatus.COMPLETED and current_state.winner:
            if current_state.winner == starting_player:
                return 1.0
            else:
                return -1.0
        return 0.0  # Draw or timeout

    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        """Backpropagate value up the tree."""
        current = node
        while current is not None:
            current.visit_count += 1
            # Flip value for opponent's perspective
            current.value_sum += value
            value = -value
            current = current.parent

    def select_move(self, game_state: GameState) -> Move | None:
        """Select move using GMO-guided MCTS."""
        from ..models import GameStatus

        # Use inherited rules_engine for consistency
        legal_moves = self.get_valid_moves(game_state)

        if not legal_moves:
            return None

        if len(legal_moves) == 1:
            return legal_moves[0]

        # Create root node
        root = MCTSNode(state=game_state)

        # Get GMO priors for root
        priors = self._get_gmo_priors(game_state, legal_moves)

        # Expand root
        self._expand(root, legal_moves, priors)

        # Run simulations
        for _ in range(self.hybrid_config.num_simulations):
            node = root

            # Selection: traverse tree to leaf
            while node.is_expanded and node.children:
                node = self._select_child(node)

            # Check if game ended
            if node.state.game_status != GameStatus.ACTIVE:
                # Terminal node - evaluate directly
                if node.state.winner == game_state.current_player:
                    value = 1.0
                elif node.state.winner:
                    value = -1.0
                else:
                    value = 0.0
            else:
                # Expand if not expanded
                node_current_player = node.state.current_player
                node_legal_moves = self.rules_engine.get_valid_moves(node.state, node_current_player)
                if node_legal_moves and not node.is_expanded:
                    node_priors = self._get_gmo_priors(node.state, node_legal_moves)
                    self._expand(node, node_legal_moves, node_priors)

                # Simulation
                value = self._simulate(node.state)

            # Backpropagation
            self._backpropagate(node, value)

        # Select move with most visits (or highest Q for low temperature)
        if self.hybrid_config.temperature < 0.01:
            # Greedy selection
            best_visits = -1
            best_move = None
            for _move_key, child in root.children.items():
                if child.visit_count > best_visits:
                    best_visits = child.visit_count
                    best_move = child.move
            return best_move
        else:
            # Sample proportional to visit counts
            moves = []
            visit_counts = []
            for child in root.children.values():
                moves.append(child.move)
                visit_counts.append(child.visit_count)

            visit_counts = np.array(visit_counts, dtype=np.float32)
            visit_counts = visit_counts ** (1.0 / self.hybrid_config.temperature)
            probs = visit_counts / visit_counts.sum()

            idx = np.random.choice(len(moves), p=probs)
            return moves[idx]

    def evaluate_position(self, game_state: GameState) -> float:
        """Evaluate the current position from this AI's perspective.

        Uses GMO value estimation.

        Args:
            game_state: Current game state

        Returns:
            Position evaluation from -1.0 (losing) to 1.0 (winning)
        """
        # Use GMO for evaluation
        with torch.no_grad():
            state_embed = self.gmo.state_encoder.encode_state(game_state)

            # Get legal moves and evaluate best one
            legal_moves = self.get_valid_moves(game_state)
            if not legal_moves:
                return 0.0

            best_value = float('-inf')
            for move in legal_moves[:5]:  # Check top 5 moves for speed
                move_embed = self.gmo.move_encoder.encode_move(move)
                mean_val, _, _ = estimate_uncertainty(
                    state_embed,
                    move_embed,
                    self.gmo.value_net,
                    self.gmo.gmo_config.mc_samples,
                )
                if mean_val.item() > best_value:
                    best_value = mean_val.item()

            return best_value


def create_gmo_mcts_hybrid(
    player_number: int,
    num_simulations: int = 100,
    device: str = "cpu",
    gmo_checkpoint: str | None = None,
) -> GMOMCTSHybrid:
    """Create a GMO-MCTS hybrid AI.

    Args:
        player_number: Player number (1-based)
        num_simulations: Number of MCTS simulations per move
        device: Device to use
        gmo_checkpoint: Optional path to GMO checkpoint

    Returns:
        Configured GMOMCTSHybrid instance
    """
    ai_config = AIConfig(difficulty=7)
    hybrid_config = GMOMCTSConfig(
        num_simulations=num_simulations,
        device=device,
    )

    ai = GMOMCTSHybrid(
        player_number=player_number,
        config=ai_config,
        hybrid_config=hybrid_config,
    )

    if gmo_checkpoint:
        ai.load_gmo_checkpoint(gmo_checkpoint)

    return ai
