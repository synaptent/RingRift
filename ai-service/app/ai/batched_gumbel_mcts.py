"""Batched Gumbel MCTS for parallel game evaluation.

This module extends GumbelMCTSAI to run MCTS search on multiple games
simultaneously, batching NN evaluations across all games for 3-4x speedup.

Key optimization:
- Standard approach: Run MCTS for each game sequentially, each with its own NN calls
- Batched approach: Run MCTS for N games in parallel, share NN calls across all games

Instead of: N games × 4 phases × batch_size = N×4 NN forward passes
We get:     4 phases × (N × batch_size) = 4 NN forward passes (larger batches)

This dramatically improves GPU utilization since larger batches amortize
the kernel launch and memory transfer overhead.

Usage:
    batched_mcts = BatchedGumbelMCTS(
        board_type=BoardType.SQUARE8,
        num_players=2,
        neural_net=nn,
        batch_size=16,  # Number of games to run in parallel
    )

    # Get moves for 16 games at once
    moves = batched_mcts.select_moves_batch(game_states)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..models import BoardType, GameState, Move
from ..rules.mutable_state import MutableGameState
from .gumbel_mcts_ai import GumbelAction, GumbelMCTSAI, LeafEvalRequest, LeafEvaluationBuffer

if TYPE_CHECKING:
    from .neural_net import NeuralNetAI

logger = logging.getLogger(__name__)


@dataclass
class GameSearchState:
    """State of MCTS search for a single game."""

    game_idx: int
    game_state: GameState
    valid_moves: list[Move]
    actions: list[GumbelAction]  # Candidate actions after Gumbel-Top-K
    remaining_actions: list[GumbelAction]  # Actions still in contention
    best_move: Move | None = None
    is_complete: bool = False  # True when search is done


@dataclass
class BatchLeafEvalRequest:
    """Pending leaf evaluation for batch processing across games."""

    game_state: GameState
    is_opponent_perspective: bool
    game_idx: int  # Which game this belongs to
    action_idx: int  # Which action within the game
    simulation_idx: int


class BatchedGumbelMCTS:
    """Batched Gumbel MCTS that runs search on multiple games simultaneously.

    This class wraps GumbelMCTSAI but optimizes for multiple games by:
    1. Batching initial policy logit calls across all games
    2. Batching all leaf evaluations across all games in each phase
    3. Processing all games' Sequential Halving phases together
    """

    def __init__(
        self,
        board_type: BoardType,
        num_players: int,
        neural_net: NeuralNetAI | None = None,
        batch_size: int = 16,
        num_sampled_actions: int = 16,
        simulation_budget: int = 800,
        player_number: int = 1,
    ):
        """Initialize batched Gumbel MCTS.

        Args:
            board_type: Type of board (SQUARE8, HEX8, etc.)
            num_players: Number of players (2, 3, or 4)
            neural_net: Neural network for evaluation (optional)
            batch_size: Maximum number of games to run in parallel
            num_sampled_actions: K for Gumbel-Top-K sampling
            simulation_budget: Total simulations per action selection (800+ for quality)
            player_number: Player number this AI represents
        """
        self.board_type = board_type
        self.num_players = num_players
        self.neural_net = neural_net
        self.batch_size = batch_size
        self.num_sampled_actions = num_sampled_actions
        self.simulation_budget = simulation_budget
        self.player_number = player_number

        # Create base MCTS AI for utility functions
        from ..models import AIConfig, AIType
        config = AIConfig(
            ai_type=AIType.GUMBEL_MCTS,
            difficulty=7,
            gumbel_num_sampled_actions=num_sampled_actions,
            gumbel_simulation_budget=simulation_budget,
        )
        self._base_mcts = GumbelMCTSAI(
            player_number=player_number,
            config=config,
            board_type=board_type,
        )
        self._base_mcts.neural_net = neural_net

        # RNG for Gumbel noise
        self.rng = np.random.default_rng()

        logger.info(
            f"BatchedGumbelMCTS initialized: batch_size={batch_size}, "
            f"k={num_sampled_actions}, budget={simulation_budget}"
        )

    def select_moves_batch(
        self,
        game_states: list[GameState],
    ) -> list[Move | None]:
        """Select best moves for multiple games simultaneously.

        Args:
            game_states: List of game states (up to batch_size).

        Returns:
            List of selected moves (None if no valid moves for that game).
        """
        if not game_states:
            return []

        n_games = len(game_states)

        # Initialize search state for each game
        search_states = []
        for idx, game_state in enumerate(game_states):
            valid_moves = self._base_mcts.get_valid_moves(game_state)

            if not valid_moves:
                search_states.append(GameSearchState(
                    game_idx=idx,
                    game_state=game_state,
                    valid_moves=[],
                    actions=[],
                    remaining_actions=[],
                    best_move=None,
                    is_complete=True,
                ))
            elif len(valid_moves) == 1:
                search_states.append(GameSearchState(
                    game_idx=idx,
                    game_state=game_state,
                    valid_moves=valid_moves,
                    actions=[],
                    remaining_actions=[],
                    best_move=valid_moves[0],
                    is_complete=True,
                ))
            else:
                search_states.append(GameSearchState(
                    game_idx=idx,
                    game_state=game_state,
                    valid_moves=valid_moves,
                    actions=[],
                    remaining_actions=[],
                    is_complete=False,
                ))

        # Check if all games are already complete
        active_games = [s for s in search_states if not s.is_complete]
        if not active_games:
            return [s.best_move for s in search_states]

        # Step 1: Batch get policy logits for all active games
        self._batch_get_policy_logits(active_games)

        # Step 2: Gumbel-Top-K for each game
        for state in active_games:
            state.actions = self._gumbel_top_k_sample(state)
            state.remaining_actions = list(state.actions)

            if len(state.actions) == 1:
                state.best_move = state.actions[0].move
                state.is_complete = True

        # Update active games list
        active_games = [s for s in search_states if not s.is_complete]
        if not active_games:
            return [s.best_move for s in search_states]

        # Step 3: Batched Sequential Halving across all games
        self._batch_sequential_halving(active_games)

        # Extract results
        return [s.best_move for s in search_states]

    def _batch_get_policy_logits(self, search_states: list[GameSearchState]) -> None:
        """Batch evaluate policy logits for all games.

        Args:
            search_states: List of active game search states.
        """
        if self.neural_net is None:
            # Uniform logits when no neural net
            for state in search_states:
                state._policy_logits = np.zeros(len(state.valid_moves))
            return

        try:
            # Batch evaluate all game states
            game_states = [s.game_state for s in search_states]
            value_heads = [
                self._base_mcts._get_value_head(s.game_state)
                for s in search_states
            ]

            # If all value heads are the same, use single call
            if len(set(value_heads)) == 1:
                _, policies = self.neural_net.evaluate_batch(
                    game_states, value_head=value_heads[0]
                )
            else:
                # Evaluate each game separately (different value heads)
                policies = []
                for state, vh in zip(search_states, value_heads, strict=False):
                    _, policy = self.neural_net.evaluate_batch(
                        [state.game_state], value_head=vh
                    )
                    policies.append(policy[0] if len(policy) > 0 else np.zeros(0))

            # Extract logits for valid moves in each game
            for state_idx, state in enumerate(search_states):
                policy_vec = policies[state_idx] if state_idx < len(policies) else None

                if policy_vec is None or len(policy_vec) == 0:
                    state._policy_logits = np.zeros(len(state.valid_moves))
                    continue

                logits = []
                for move in state.valid_moves:
                    idx = self.neural_net.encode_move(move, state.game_state.board)
                    if idx >= 0 and idx < len(policy_vec):
                        prob = max(float(policy_vec[idx]), 1e-10)
                        logit = np.log(prob)
                    else:
                        logit = -10.0
                    logits.append(logit)

                state._policy_logits = np.array(logits, dtype=np.float32)

                # Apply Dirichlet noise
                state._policy_logits = self._base_mcts._apply_dirichlet_noise(
                    state._policy_logits, self.board_type
                )

        except (RuntimeError, ValueError, IndexError, TypeError) as e:
            logger.warning(f"BatchedGumbelMCTS: batch policy evaluation failed ({e})")
            for state in search_states:
                state._policy_logits = np.zeros(len(state.valid_moves))

    def _gumbel_top_k_sample(self, state: GameSearchState) -> list[GumbelAction]:
        """Sample top-k actions using Gumbel-Top-K for a single game.

        Args:
            state: Game search state with valid_moves and _policy_logits set.

        Returns:
            List of GumbelAction objects for the top-k actions.
        """
        k = min(self.num_sampled_actions, len(state.valid_moves))

        # Generate Gumbel noise
        uniform = self.rng.uniform(1e-10, 1.0 - 1e-10, size=len(state.valid_moves))
        gumbel_noise = -np.log(-np.log(uniform))

        # Perturb logits
        perturbed = state._policy_logits + gumbel_noise

        # Select top-k
        top_k_indices = np.argsort(perturbed)[-k:][::-1]

        actions = []
        for idx in top_k_indices:
            actions.append(GumbelAction(
                move=state.valid_moves[idx],
                policy_logit=float(state._policy_logits[idx]),
                gumbel_noise=float(gumbel_noise[idx]),
                perturbed_value=float(perturbed[idx]),
            ))

        return actions

    def _batch_sequential_halving(self, search_states: list[GameSearchState]) -> None:
        """Run Sequential Halving for all games with batched evaluation.

        This is the key optimization: collect leaf states from ALL games
        and evaluate them in a single batch.

        Args:
            search_states: List of active game search states.
        """
        # Determine max phases needed (based on max actions across games)
        max_actions = max(len(s.remaining_actions) for s in search_states)
        max_phases = int(np.ceil(np.log2(max_actions)))

        budget_per_phase = self.simulation_budget // max(max_phases, 1)

        for _phase in range(max_phases):
            # Check if all games are done
            active = [s for s in search_states if len(s.remaining_actions) > 1]
            if not active:
                break

            # Collect all leaf evaluation requests across all games
            all_requests: list[BatchLeafEvalRequest] = []
            action_values: dict[tuple[int, int], list[float]] = {}  # (game_idx, action_idx) -> values

            for state in active:
                sims_per_action = max(1, budget_per_phase // len(state.remaining_actions))

                for action_idx, action in enumerate(state.remaining_actions):
                    key = (state.game_idx, action_idx)
                    action_values[key] = []

                    # Apply move
                    mstate = MutableGameState.from_immutable(state.game_state)
                    undo = mstate.make_move(action.move)

                    # Check terminal
                    if mstate.is_game_over():
                        winner = mstate.winner
                        if winner == self.player_number:
                            v = 1.0
                        elif winner is None:
                            v = 0.0
                        else:
                            v = -1.0
                        action_values[key] = [v] * sims_per_action
                        mstate.unmake_move(undo)
                        continue

                    # Collect leaf states for simulations
                    for sim_idx in range(sims_per_action):
                        sim_mstate = MutableGameState.from_immutable(mstate.to_immutable())
                        leaf_state, is_terminal, terminal_value, is_opponent = (
                            self._base_mcts._collect_leaf_state(sim_mstate)
                        )

                        if is_terminal:
                            action_values[key].append(terminal_value)
                        else:
                            all_requests.append(BatchLeafEvalRequest(
                                game_state=leaf_state,
                                is_opponent_perspective=is_opponent,
                                game_idx=state.game_idx,
                                action_idx=action_idx,
                                simulation_idx=sim_idx,
                            ))

                    mstate.unmake_move(undo)

            # Batch evaluate all leaf states across all games
            if all_requests:
                self._batch_evaluate_leaves(all_requests, action_values)

            # Update action statistics and prune
            for state in active:
                for action_idx, action in enumerate(state.remaining_actions):
                    key = (state.game_idx, action_idx)
                    values = action_values.get(key, [])
                    action.visit_count += len(values)
                    action.total_value += sum(values)

                # Sort and keep top half
                max_visits = max(a.visit_count for a in state.remaining_actions)
                state.remaining_actions.sort(
                    key=lambda a: a.completed_q(max_visits),
                    reverse=True
                )
                state.remaining_actions = state.remaining_actions[:max(1, len(state.remaining_actions) // 2)]

        # Set best move for each game
        for state in search_states:
            if state.remaining_actions:
                state.best_move = state.remaining_actions[0].move
                state.is_complete = True

    def _batch_evaluate_leaves(
        self,
        requests: list[BatchLeafEvalRequest],
        action_values: dict[tuple[int, int], list[float]],
    ) -> None:
        """Batch evaluate all leaf states and update action values.

        Args:
            requests: All leaf evaluation requests across all games.
            action_values: Dictionary to update with evaluation results.
        """
        if not requests or self.neural_net is None:
            # Fallback: use heuristic or default value
            for req in requests:
                key = (req.game_idx, req.action_idx)
                action_values[key].append(0.0)
            return

        try:
            # Collect all game states
            game_states = [req.game_state for req in requests]

            # Batch evaluate (use default value head for simplicity)
            values, _ = self.neural_net.evaluate_batch(game_states)

            # Distribute results back
            for req, value in zip(requests, values, strict=False):
                # Flip perspective if needed
                v = float(value)
                if req.is_opponent_perspective:
                    v = -v

                key = (req.game_idx, req.action_idx)
                action_values[key].append(v)

        except (RuntimeError, ValueError, TypeError) as e:
            logger.warning(f"BatchedGumbelMCTS: batch leaf evaluation failed ({e})")
            for req in requests:
                key = (req.game_idx, req.action_idx)
                action_values[key].append(0.0)


def create_batched_gumbel_mcts(
    board_type: str | BoardType,
    num_players: int = 2,
    batch_size: int = 16,
    num_sampled_actions: int = 16,
    simulation_budget: int = 800,
    neural_net: NeuralNetAI | None = None,
    player_number: int = 1,
) -> BatchedGumbelMCTS:
    """Factory function to create a BatchedGumbelMCTS instance.

    Args:
        board_type: Board type string or enum.
        num_players: Number of players (2, 3, or 4).
        batch_size: Maximum number of games to run in parallel.
        num_sampled_actions: K for Gumbel-Top-K sampling.
        simulation_budget: Total simulations per action selection.
        neural_net: Neural network for evaluation.
        player_number: Player number this AI represents.

    Returns:
        Configured BatchedGumbelMCTS instance.
    """
    if isinstance(board_type, str):
        board_type_map = {
            "square8": BoardType.SQUARE8,
            "square19": BoardType.SQUARE19,
            "hex8": BoardType.HEX8,
            "hex": BoardType.HEXAGONAL,
            "hexagonal": BoardType.HEXAGONAL,
        }
        board_type = board_type_map.get(board_type.lower(), BoardType.SQUARE8)

    return BatchedGumbelMCTS(
        board_type=board_type,
        num_players=num_players,
        neural_net=neural_net,
        batch_size=batch_size,
        num_sampled_actions=num_sampled_actions,
        simulation_budget=simulation_budget,
        player_number=player_number,
    )
