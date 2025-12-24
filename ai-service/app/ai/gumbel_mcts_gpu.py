"""GPU-accelerated Gumbel MCTS for parallel game evaluation.

This module extends GumbelMCTSAI to leverage GPU infrastructure for simulation,
providing 2-3x additional speedup beyond batched NN evaluation.

Key optimizations (Phase 1 of GPU MCTS plan):
1. Use BatchGameState for GPU-based game simulation
2. Use GPU move generation instead of CPU rules engine
3. Batch all simulations for an action together

This module bridges the existing GPU parallel games infrastructure with MCTS
tree search, avoiding the need to copy states back to CPU for each simulation.

Usage:
    mcts = GumbelMCTSGPU(
        board_type=BoardType.SQUARE8,
        num_players=2,
        neural_net=nn,
        simulation_budget=800,
    )

    # Get move for a single game (integrates with existing code)
    move = mcts.select_move(game_state)

    # Or for batch of games
    moves = mcts.select_moves_batch(game_states)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from ..models import BoardType, GameState, Move
from .gpu_batch import get_device
from .gpu_batch_state import BatchGameState
from .gpu_game_types import GamePhase as GPUGamePhase, GameStatus
from .gumbel_mcts_ai import GumbelAction, GumbelMCTSAI

if TYPE_CHECKING:
    from .neural_net import NeuralNetAI

logger = logging.getLogger(__name__)

# Environment variable to disable GPU MCTS (fallback to CPU)
_GPU_MCTS_DISABLE = os.environ.get("RINGRIFT_GPU_MCTS_DISABLE", "").lower() in (
    "1", "true", "yes", "on"
)


@dataclass
class GPUSimulationRequest:
    """Request for GPU-batched simulation."""

    action_idx: int  # Index into actions list
    game_state: GameState  # State after taking action
    num_simulations: int  # How many simulations to run


@dataclass
class GPUSimulationResult:
    """Result of GPU-batched simulations for an action."""

    action_idx: int
    values: list[float]  # One value per simulation


class GPUSimulationEngine:
    """GPU-accelerated game simulation for MCTS.

    Uses BatchGameState and GPU move generation to run many game
    simulations in parallel, avoiding CPU rules engine overhead.
    """

    def __init__(
        self,
        board_type: BoardType,
        num_players: int = 2,
        device: torch.device | None = None,
        max_simulation_depth: int = 50,
    ):
        """Initialize GPU simulation engine.

        Args:
            board_type: Board type (SQUARE8, HEX8, etc.)
            num_players: Number of players (2, 3, or 4)
            device: GPU device (auto-detected if None)
            max_simulation_depth: Maximum moves per simulation
        """
        self.board_type = board_type
        self.num_players = num_players
        self.device = device or get_device()
        self.max_simulation_depth = max_simulation_depth

        # Map board type to string for BatchGameState
        self._board_type_str = {
            BoardType.SQUARE8: "square8",
            BoardType.SQUARE19: "square19",
            BoardType.HEX8: "hex8",
            BoardType.HEXAGONAL: "hexagonal",
        }.get(board_type, "square8")

        # Board size from type
        self._board_size = {
            BoardType.SQUARE8: 8,
            BoardType.SQUARE19: 19,
            BoardType.HEX8: 9,
            BoardType.HEXAGONAL: 25,
        }.get(board_type, 8)

        logger.debug(
            f"GPUSimulationEngine initialized: board={self._board_type_str}, "
            f"players={num_players}, device={self.device}"
        )

    def run_simulations_batch(
        self,
        requests: list[GPUSimulationRequest],
        player_to_evaluate: int,
    ) -> list[GPUSimulationResult]:
        """Run simulations for multiple actions in parallel on GPU.

        Args:
            requests: List of simulation requests (action + game state + count)
            player_to_evaluate: Player perspective for value calculation

        Returns:
            List of simulation results with values for each action.
        """
        if not requests:
            return []

        # Expand requests into individual simulations
        all_game_states: list[GameState] = []
        sim_to_action: list[int] = []  # Maps simulation index to action index

        for req in requests:
            for _ in range(req.num_simulations):
                all_game_states.append(req.game_state)
                sim_to_action.append(req.action_idx)

        if not all_game_states:
            return [GPUSimulationResult(req.action_idx, []) for req in requests]

        # Convert to BatchGameState for GPU simulation
        batch_state = BatchGameState.from_game_states(
            all_game_states,
            device=self.device,
        )

        # Run simulations until all games terminate or max depth
        values = self._run_gpu_simulations(batch_state, player_to_evaluate)

        # Group values by action
        results_by_action: dict[int, list[float]] = {
            req.action_idx: [] for req in requests
        }

        for sim_idx, value in enumerate(values):
            action_idx = sim_to_action[sim_idx]
            results_by_action[action_idx].append(value)

        return [
            GPUSimulationResult(req.action_idx, results_by_action[req.action_idx])
            for req in requests
        ]

    def _run_gpu_simulations(
        self,
        batch_state: BatchGameState,
        player_to_evaluate: int,
    ) -> list[float]:
        """Run simulations on GPU batch state.

        Uses GPU move generation and random move selection for fast playouts.

        Args:
            batch_state: GPU batch state with games to simulate
            player_to_evaluate: Player perspective for value calculation

        Returns:
            List of values (one per game in batch)
        """
        from .gpu_move_generation import (
            generate_placement_moves_batch,
            generate_movement_moves_batch,
            generate_capture_moves_batch,
        )
        from .gpu_move_application import (
            apply_placement_moves_batch,
            apply_movement_moves_batch,
            apply_capture_moves_batch,
        )
        from .gpu_selection import select_moves_vectorized

        batch_size = batch_state.batch_size
        board_size = batch_state.board_size

        for depth in range(self.max_simulation_depth):
            # Check which games are still active
            active_mask = batch_state.get_active_mask()
            if not active_mask.any():
                break

            # Get current phase for each game
            phases = batch_state.current_phase

            # Generate and select moves per phase
            # For simplicity, we use random move selection for playouts

            # Handle placement phase
            placement_mask = active_mask & (phases == GPUGamePhase.RING_PLACEMENT)
            if placement_mask.any():
                moves = generate_placement_moves_batch(batch_state, placement_mask)
                if moves.total_moves > 0:
                    # select_moves_vectorized expects: (moves, active_mask, board_size, temperature)
                    selected = select_moves_vectorized(
                        moves, placement_mask, board_size, temperature=1.0
                    )
                    apply_placement_moves_batch(batch_state, selected, moves)

            # Handle movement phase
            movement_mask = active_mask & (phases == GPUGamePhase.MOVEMENT)
            if movement_mask.any():
                moves = generate_movement_moves_batch(batch_state, movement_mask)
                if moves.total_moves > 0:
                    selected = select_moves_vectorized(
                        moves, movement_mask, board_size, temperature=1.0
                    )
                    apply_movement_moves_batch(batch_state, selected, moves)

            # Handle capture phase
            capture_mask = active_mask & (
                (phases == GPUGamePhase.CAPTURE) |
                (phases == GPUGamePhase.CHAIN_CAPTURE)
            )
            if capture_mask.any():
                moves = generate_capture_moves_batch(batch_state, capture_mask)
                if moves.total_moves > 0:
                    selected = select_moves_vectorized(
                        moves, capture_mask, board_size, temperature=1.0
                    )
                    apply_capture_moves_batch(batch_state, selected, moves)

        # Extract values from final states using vectorized operations
        # Avoid .item() calls for GPU efficiency
        winners = batch_state.winner  # Keep on GPU
        territory_counts = batch_state.territory_count

        # Compute values vectorized
        win_mask = (winners == player_to_evaluate)
        lose_mask = (winners > 0) & (winners != player_to_evaluate)

        # For unfinished games, use territory ratio
        my_territory = territory_counts[:, player_to_evaluate].float()
        total_territory = territory_counts[:, 1:self.num_players+1].sum(dim=1).float()
        territory_ratio = torch.where(
            total_territory > 0,
            2 * (my_territory / total_territory) - 1,
            torch.zeros_like(my_territory)
        )

        # Combine: win=1, lose=-1, else=territory_ratio
        values_tensor = torch.where(
            win_mask,
            torch.ones(batch_size, device=batch_state.device),
            torch.where(
                lose_mask,
                -torch.ones(batch_size, device=batch_state.device),
                territory_ratio
            )
        )

        return values_tensor.cpu().tolist()


class GumbelMCTSGPU(GumbelMCTSAI):
    """GPU-accelerated Gumbel MCTS that uses GPU for simulation phase.

    Extends GumbelMCTSAI to leverage BatchGameState for parallel simulation,
    providing 2-3x speedup over CPU-based simulation.

    The tree structure and selection logic remain on CPU (Phase 2 will
    add tensor-based tree storage for further acceleration).
    """

    def __init__(
        self,
        player_number: int,
        config: Any,
        board_type: BoardType,
        neural_net: NeuralNetAI | None = None,
        gpu_simulation: bool = True,
        device: torch.device | None = None,
    ):
        """Initialize GPU-accelerated Gumbel MCTS.

        Args:
            player_number: Player number this AI represents
            config: AI configuration
            board_type: Board type (SQUARE8, HEX8, etc.)
            neural_net: Neural network for policy/value evaluation
            gpu_simulation: Enable GPU simulation (disable for CPU fallback)
            device: GPU device (auto-detected if None)
        """
        super().__init__(player_number, config, board_type)
        self.neural_net = neural_net
        self.gpu_simulation = gpu_simulation and not _GPU_MCTS_DISABLE
        self.device = device or get_device()

        # Create GPU simulation engine
        if self.gpu_simulation:
            # Infer num_players from config if available
            num_players = getattr(config, 'num_players', 2)
            self._gpu_engine = GPUSimulationEngine(
                board_type=board_type,
                num_players=num_players,
                device=self.device,
            )
        else:
            self._gpu_engine = None

        logger.info(
            f"GumbelMCTSGPU initialized: gpu_simulation={self.gpu_simulation}, "
            f"device={self.device}"
        )

    def _run_simulations_for_action(
        self,
        action: GumbelAction,
        game_state: GameState,
        num_simulations: int,
        value_head: int | None = None,
    ) -> list[float]:
        """Run simulations for a single action.

        Overrides parent to use GPU simulation when enabled.

        Args:
            action: Action to evaluate
            game_state: Current game state
            num_simulations: Number of simulations to run
            value_head: Which value head to use for NN evaluation

        Returns:
            List of values from simulations
        """
        if not self.gpu_simulation or self._gpu_engine is None:
            # Fall back to CPU simulation
            return super()._run_simulations_for_action(
                action, game_state, num_simulations, value_head
            )

        # Apply action to get child state
        from ..rules.mutable_state import MutableGameState
        mstate = MutableGameState.from_immutable(game_state)
        mstate.make_move(action.move)
        child_state = mstate.to_immutable()

        # Use GPU engine for simulation
        request = GPUSimulationRequest(
            action_idx=0,
            game_state=child_state,
            num_simulations=num_simulations,
        )

        results = self._gpu_engine.run_simulations_batch(
            [request],
            player_to_evaluate=self.player_number,
        )

        return results[0].values if results else []

    def _run_simulations_batch(
        self,
        actions: list[GumbelAction],
        game_state: GameState,
        sims_per_action: int,
        value_head: int | None = None,
    ) -> dict[int, list[float]]:
        """Run simulations for multiple actions in parallel.

        Uses GPU batching to run all simulations together.

        Args:
            actions: Actions to evaluate
            game_state: Current game state
            sims_per_action: Number of simulations per action
            value_head: Which value head to use for NN evaluation

        Returns:
            Dict mapping action index to list of values
        """
        if not self.gpu_simulation or self._gpu_engine is None:
            # Fall back to sequential CPU simulation
            result = {}
            for i, action in enumerate(actions):
                result[i] = self._run_simulations_for_action(
                    action, game_state, sims_per_action, value_head
                )
            return result

        # Create requests for all actions
        from ..rules.mutable_state import MutableGameState

        requests = []
        for i, action in enumerate(actions):
            mstate = MutableGameState.from_immutable(game_state)
            mstate.make_move(action.move)
            child_state = mstate.to_immutable()

            requests.append(GPUSimulationRequest(
                action_idx=i,
                game_state=child_state,
                num_simulations=sims_per_action,
            ))

        # Run all simulations in batch
        results = self._gpu_engine.run_simulations_batch(
            requests,
            player_to_evaluate=self.player_number,
        )

        return {r.action_idx: r.values for r in results}


def create_gumbel_mcts_gpu(
    board_type: str | BoardType,
    num_players: int = 2,
    num_sampled_actions: int = 16,
    simulation_budget: int = 800,
    neural_net: NeuralNetAI | None = None,
    player_number: int = 1,
    gpu_simulation: bool = True,
    device: torch.device | None = None,
) -> GumbelMCTSGPU:
    """Factory function to create a GPU-accelerated Gumbel MCTS instance.

    Args:
        board_type: Board type string or enum.
        num_players: Number of players (2, 3, or 4).
        num_sampled_actions: K for Gumbel-Top-K sampling.
        simulation_budget: Total simulations per action selection.
        neural_net: Neural network for evaluation.
        player_number: Player number this AI represents.
        gpu_simulation: Enable GPU simulation.
        device: GPU device.

    Returns:
        Configured GumbelMCTSGPU instance.
    """
    from ..models import AIConfig, AIType

    if isinstance(board_type, str):
        board_type_map = {
            "square8": BoardType.SQUARE8,
            "square19": BoardType.SQUARE19,
            "hex8": BoardType.HEX8,
            "hex": BoardType.HEXAGONAL,
            "hexagonal": BoardType.HEXAGONAL,
        }
        board_type = board_type_map.get(board_type.lower(), BoardType.SQUARE8)

    config = AIConfig(
        ai_type=AIType.GUMBEL_MCTS,
        difficulty=7,
        gumbel_num_sampled_actions=num_sampled_actions,
        gumbel_simulation_budget=simulation_budget,
        num_players=num_players,
    )

    return GumbelMCTSGPU(
        player_number=player_number,
        config=config,
        board_type=board_type,
        neural_net=neural_net,
        gpu_simulation=gpu_simulation,
        device=device,
    )
