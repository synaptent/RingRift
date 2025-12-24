"""GMO-Gumbel MCTS Hybrid - Use GMO value network with Gumbel MCTS search.

This module integrates GMO's value network with Gumbel MCTS's efficient tree search.
Unlike the regular GMO-MCTS hybrid, this uses Gumbel sampling + Sequential Halving
for more sample-efficient search.

Key features:
1. GMO provides value estimates at leaf nodes
2. GMO move scores converted to policy logits for Gumbel-Top-K sampling
3. Sequential Halving for efficient budget allocation
4. Optional uncertainty-based exploration bonus

This allows fair comparison between:
- CNN + Gumbel MCTS (existing production configuration)
- GMO + Gumbel MCTS (this hybrid)

Usage:
    from app.ai.gmo_gumbel_hybrid import GumbelMCTSGMOAI, create_gmo_gumbel_ai

    ai = create_gmo_gumbel_ai(
        player_number=1,
        gmo_checkpoint="models/gmo/gmo_best.pt",
        simulation_budget=150,
    )
    move = ai.select_move(game_state)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..models import AIConfig, BoardType, GameState, Move
from ..rules.mutable_state import MutableGameState
from .base import BaseAI
from .gmo_ai import GMOAI, GMOConfig, estimate_uncertainty
from .gumbel_common import GumbelAction  # Unified data structure

logger = logging.getLogger(__name__)


@dataclass
class GMOGumbelConfig:
    """Configuration for GMO-Gumbel MCTS hybrid."""
    # Gumbel MCTS parameters
    num_sampled_actions: int = 16
    simulation_budget: int = 150
    c_puct: float = 1.5

    # GMO integration
    use_uncertainty_exploration: bool = True
    uncertainty_weight: float = 0.5

    # GMO config
    gmo_config: GMOConfig | None = None

    # Device
    device: str = "cpu"


class GumbelMCTSGMOAI(BaseAI):
    """Gumbel MCTS with GMO value network.

    This hybrid AI uses:
    - GMO's value network for position evaluation
    - GMO's move scores as policy logits
    - Gumbel sampling for action selection
    - Sequential Halving for budget allocation

    This provides a fair comparison point with CNN + Gumbel MCTS.
    """

    def __init__(
        self,
        player_number: int,
        config: AIConfig,
        gumbel_config: GMOGumbelConfig | None = None,
    ):
        super().__init__(player_number, config)

        self.gumbel_config = gumbel_config or GMOGumbelConfig()
        self.device = torch.device(self.gumbel_config.device)

        # Initialize GMO
        gmo_config = self.gumbel_config.gmo_config or GMOConfig(
            device=self.gumbel_config.device
        )
        self.gmo = GMOAI(
            player_number=player_number,
            config=config,
            gmo_config=gmo_config,
        )

        # Store search actions for training data extraction
        self._last_search_actions: list[GumbelAction] | None = None

        logger.info(
            f"GumbelMCTSGMOAI(player={player_number}): initialized "
            f"(m={self.gumbel_config.num_sampled_actions}, "
            f"budget={self.gumbel_config.simulation_budget})"
        )

    def load_gmo_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Load GMO model weights."""
        self.gmo.load_checkpoint(Path(checkpoint_path))

    def _get_gmo_policy_logits(
        self,
        game_state: GameState,
        valid_moves: list[Move],
    ) -> np.ndarray:
        """Get policy logits from GMO move scores.

        Converts GMO's (value + uncertainty) scores into policy logits
        for Gumbel-Top-K sampling.
        """
        if not valid_moves:
            return np.array([])

        with torch.no_grad():
            state_embed = self.gmo.state_encoder.encode_state(game_state)

            scores = []
            for move in valid_moves:
                move_embed = self.gmo.move_encoder.encode_move(move)

                # Get value + uncertainty estimate
                mean_val, _, var = estimate_uncertainty(
                    state_embed,
                    move_embed,
                    self.gmo.value_net,
                    self.gmo.gmo_config.mc_samples,
                )

                # Combine value with uncertainty for exploration
                if self.gumbel_config.use_uncertainty_exploration:
                    score = mean_val + self.gumbel_config.uncertainty_weight * torch.sqrt(var + 1e-8)
                else:
                    score = mean_val

                scores.append(score.item())

        # Convert to logits (normalize to reasonable range)
        scores = np.array(scores, dtype=np.float32)

        # Scale scores to reasonable logit range
        if scores.std() > 0:
            logits = (scores - scores.mean()) / (scores.std() + 1e-8) * 2.0
        else:
            logits = scores

        return logits

    def _evaluate_leaf_gmo(
        self,
        game_state: GameState,
        is_opponent_perspective: bool,
    ) -> float:
        """Evaluate a leaf node using GMO's value network."""
        # Check for terminal state
        if game_state.game_status == "completed":
            if game_state.winner == self.player_number:
                return 1.0
            elif game_state.winner is not None:
                return -1.0
            return 0.0

        # Use GMO to evaluate position
        with torch.no_grad():
            state_embed = self.gmo.state_encoder.encode_state(game_state)

            # Get legal moves
            legal_moves = self.rules_engine.get_valid_moves(
                game_state, game_state.current_player
            )

            if not legal_moves:
                return 0.0

            # Evaluate best move to estimate position value
            best_value = float('-inf')
            for move in legal_moves[:5]:  # Check top 5 for speed
                move_embed = self.gmo.move_encoder.encode_move(move)
                mean_val, _, _ = estimate_uncertainty(
                    state_embed,
                    move_embed,
                    self.gmo.value_net,
                    self.gmo.gmo_config.mc_samples,
                )
                if mean_val.item() > best_value:
                    best_value = mean_val.item()

            value = best_value if best_value > float('-inf') else 0.0

        # Flip for opponent perspective
        if is_opponent_perspective:
            value = -value

        return value

    def _gumbel_top_k_sample(
        self,
        valid_moves: list[Move],
        policy_logits: np.ndarray,
    ) -> list[GumbelAction]:
        """Sample top-k actions using Gumbel-Top-K."""
        k = min(self.gumbel_config.num_sampled_actions, len(valid_moves))

        # Generate Gumbel noise
        seed = int(self.rng.randrange(0, 2**32 - 1))
        np_rng = np.random.default_rng(seed)
        uniform = np_rng.uniform(1e-10, 1.0 - 1e-10, size=len(valid_moves))
        gumbel_noise = -np.log(-np.log(uniform))

        # Perturb logits
        perturbed = policy_logits + gumbel_noise

        # Select top-k
        top_k_indices = np.argsort(perturbed)[-k:][::-1]

        actions = []
        for idx in top_k_indices:
            action = GumbelAction(
                move=valid_moves[idx],
                policy_logit=float(policy_logits[idx]),
                gumbel_noise=float(gumbel_noise[idx]),
                perturbed_value=float(perturbed[idx]),
            )
            actions.append(action)

        return actions

    def _sequential_halving(
        self,
        game_state: GameState,
        actions: list[GumbelAction],
    ) -> GumbelAction:
        """Run Sequential Halving to find the best action."""
        m = len(actions)
        if m == 1:
            return actions[0]

        num_phases = int(np.ceil(np.log2(m)))
        budget_per_phase = self.gumbel_config.simulation_budget // max(num_phases, 1)
        remaining = list(actions)

        for _ in range(num_phases):
            if len(remaining) == 1:
                break

            sims_per_action = max(1, budget_per_phase // len(remaining))

            # Simulate each action
            for action in remaining:
                value_sum = self._simulate_action(game_state, action.move, sims_per_action)
                action.visit_count += sims_per_action
                action.total_value += value_sum

            # Sort by completed Q-value and keep top half
            max_visits = max(a.visit_count for a in remaining)
            remaining.sort(key=lambda a: a.completed_q(max_visits), reverse=True)
            remaining = remaining[:max(1, len(remaining) // 2)]

        return remaining[0]

    def _simulate_action(
        self,
        game_state: GameState,
        action: Move,
        num_sims: int,
    ) -> float:
        """Simulate an action and return cumulative value."""
        # Apply action
        mstate = MutableGameState.from_immutable(game_state)
        undo = mstate.make_move(action)

        # Check for terminal
        if mstate.is_game_over():
            winner = mstate.winner
            if winner == self.player_number:
                value = 1.0
            elif winner is None:
                value = 0.0
            else:
                value = -1.0
            mstate.unmake_move(undo)
            return value * num_sims

        # Run simulations with random rollout + GMO leaf eval
        total_value = 0.0
        sim_state = mstate.to_immutable()

        for _ in range(num_sims):
            value = self._run_simulation(sim_state, max_depth=10)
            total_value += value

        mstate.unmake_move(undo)
        return total_value

    def _run_simulation(
        self,
        game_state: GameState,
        max_depth: int = 10,
    ) -> float:
        """Run a single simulation with random moves and GMO leaf eval."""
        mstate = MutableGameState.from_immutable(game_state)
        depth = 0

        while depth < max_depth and not mstate.is_game_over():
            valid_moves = self.rules_engine.get_valid_moves(
                mstate.to_immutable(), mstate.current_player
            )
            if not valid_moves:
                break

            # Random rollout policy
            move = self.rng.choice(valid_moves)
            mstate.make_move(move)
            depth += 1

        # Evaluate leaf with GMO
        if mstate.is_game_over():
            winner = mstate.winner
            if winner == self.player_number:
                return 1.0
            elif winner is None:
                return 0.0
            else:
                return -1.0

        is_opponent = mstate.current_player != self.player_number
        return self._evaluate_leaf_gmo(mstate.to_immutable(), is_opponent)

    def select_move(self, game_state: GameState) -> Move | None:
        """Select best move using GMO-Gumbel MCTS."""
        valid_moves = self.get_valid_moves(game_state)

        if not valid_moves:
            return None

        if len(valid_moves) == 1:
            self.move_count += 1
            return valid_moves[0]

        # Check for swap decision
        swap_move = self.maybe_select_swap_move(game_state, valid_moves)
        if swap_move is not None:
            self.move_count += 1
            return swap_move

        # Get GMO policy logits
        policy_logits = self._get_gmo_policy_logits(game_state, valid_moves)

        # Gumbel-Top-K sampling
        actions = self._gumbel_top_k_sample(valid_moves, policy_logits)

        if len(actions) == 1:
            actions[0].visit_count = 1
            self._last_search_actions = actions
            self.move_count += 1
            return actions[0].move

        # Sequential Halving
        best_action = self._sequential_halving(game_state, actions)

        self._last_search_actions = actions
        self.move_count += 1
        return best_action.move

    def evaluate_position(self, game_state: GameState) -> float:
        """Evaluate position using GMO."""
        is_opponent = game_state.current_player != self.player_number
        return self._evaluate_leaf_gmo(game_state, is_opponent)

    def get_visit_distribution(self) -> tuple[list[Move], list[float]]:
        """Extract normalized visit distribution from last search."""
        if self._last_search_actions is None:
            return [], []

        visited = [a for a in self._last_search_actions if a.visit_count > 0]
        if not visited:
            return [], []

        total = sum(a.visit_count for a in visited)
        if total == 0:
            return [], []

        moves = [a.move for a in visited]
        probs = [a.visit_count / total for a in visited]

        return moves, probs

    def __repr__(self) -> str:
        return (
            f"GumbelMCTSGMOAI(player={self.player_number}, "
            f"m={self.gumbel_config.num_sampled_actions}, "
            f"budget={self.gumbel_config.simulation_budget})"
        )


def create_gmo_gumbel_ai(
    player_number: int,
    gmo_checkpoint: str | Path | None = None,
    simulation_budget: int = 150,
    num_sampled_actions: int = 16,
    device: str = "cpu",
) -> GumbelMCTSGMOAI:
    """Create a GMO-Gumbel MCTS hybrid AI.

    Args:
        player_number: Player number (1-based)
        gmo_checkpoint: Path to GMO checkpoint
        simulation_budget: Total simulation budget
        num_sampled_actions: Number of actions for Gumbel-Top-K
        device: Device to use

    Returns:
        Configured GumbelMCTSGMOAI instance
    """
    ai_config = AIConfig(difficulty=8)
    gumbel_config = GMOGumbelConfig(
        num_sampled_actions=num_sampled_actions,
        simulation_budget=simulation_budget,
        device=device,
    )

    ai = GumbelMCTSGMOAI(
        player_number=player_number,
        config=ai_config,
        gumbel_config=gumbel_config,
    )

    if gmo_checkpoint:
        ai.load_gmo_checkpoint(gmo_checkpoint)

    return ai
