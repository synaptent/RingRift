"""Multi-Game Parallel Gumbel MCTS for high-throughput selfplay.

.. note::
    Consider using `GumbelSearchEngine` from `app.ai.gumbel_search_engine`
    for new code. It provides a unified interface to all Gumbel variants.

    Example:
        from app.ai.gumbel_search_engine import create_selfplay_engine
        engine = create_selfplay_engine(neural_net, board_type, num_players)
        results = engine.run_selfplay(num_games=64)

Runs Gumbel MCTS across multiple games simultaneously, batching NN evaluations
across ALL games' simulations for 10-20x speedup over sequential execution.

Key insight: Instead of batching within one game, batch across 64+ games.

Usage:
    from app.ai.multi_game_gumbel import MultiGameGumbelRunner

    runner = MultiGameGumbelRunner(
        num_games=64,
        simulation_budget=800,
        neural_net=my_nn,
    )

    results = runner.run_batch(num_games=64)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from app.ai.gumbel_common import GumbelAction
from app.models import (
    AIConfig,
    BoardType,
    GamePhase,
    GameState,
    GameStatus,
    Move,
)
from app.rules.default_engine import DefaultRulesEngine
from app.rules.mutable_state import MutableGameState
from app.rules.serialization import serialize_game_state
from app.training.initial_state import create_initial_state

if TYPE_CHECKING:
    from app.ai.neural_net import NeuralNetAI

logger = logging.getLogger(__name__)

# This module uses GumbelAction from gumbel_common.py with use_simple_additive=True
# for the completed_q calculation, trading accuracy for throughput in batch scenarios.


@dataclass
class MultiGameSearchState:
    """Search state for a single game in multi-game runner.

    This extends the search-only GameSearchState from gumbel_common.py
    with full game lifecycle tracking (moves_played, winner, etc.).
    """
    game_idx: int
    game_state: MutableGameState
    current_player: int
    actions: list[GumbelAction] = field(default_factory=list)
    remaining_actions: list[GumbelAction] = field(default_factory=list)
    phase: int = 0
    move_count: int = 0
    done: bool = False
    winner: int | None = None
    moves_played: list[dict] = field(default_factory=list)
    initial_state_serialized: dict | None = None  # For training data export


# Backward compatibility alias
GameSearchState = MultiGameSearchState


@dataclass
class LeafRequest:
    """Request to evaluate a leaf state."""
    game_idx: int
    action_idx: int
    simulation_idx: int
    game_state: GameState
    is_opponent_perspective: bool
    is_terminal: bool = False
    terminal_value: float = 0.0  # Value from perspective of current player at root


@dataclass
class GumbelGameResult:
    """Result of a completed multi-game Gumbel MCTS game.

    December 2025: Renamed from GameResult to avoid collision with
    app.training.selfplay_runner.GameResult (canonical for selfplay) and
    app.execution.game_executor.GameResult (canonical for execution).

    February 2026: Added mcts_policies for training data export.
    """
    game_idx: int
    winner: int | None
    status: str
    move_count: int
    moves: list[dict]
    duration_ms: float
    initial_state: dict | None = None  # Serialized initial state for training
    mcts_policies: list[dict[str, float]] = field(default_factory=list)  # Per-move visit distributions


# Backward-compat alias (deprecated Dec 2025)
GameResult = GumbelGameResult


class MultiGameGumbelRunner:
    """Run Gumbel MCTS across multiple games with batched evaluation.

    This class provides 10-20x speedup over sequential game execution by:
    1. Running 64+ games in parallel
    2. Synchronizing Sequential Halving phases across games
    3. Batching ALL leaf evaluations into single NN calls
    """

    def __init__(
        self,
        num_games: int = 64,
        simulation_budget: int = 800,
        num_sampled_actions: int = 16,
        board_type: BoardType = BoardType.SQUARE8,
        num_players: int = 2,
        neural_net: NeuralNetAI | None = None,
        device: str = "cuda",
        max_moves_per_game: int = 500,
        temperature: float = 1.0,
        temperature_threshold: int = 30,
    ):
        """Initialize multi-game Gumbel runner.

        Args:
            num_games: Number of games to run in parallel.
            simulation_budget: Simulations per move (800 for quality).
            num_sampled_actions: Actions to sample via Gumbel-Top-K.
            board_type: Board geometry.
            num_players: Players per game.
            neural_net: Neural network for evaluation.
            device: Compute device.
            max_moves_per_game: Max moves before declaring draw.
            temperature: Policy temperature for exploration.
            temperature_threshold: Move after which to reduce temperature.
        """
        self.num_games = num_games
        self.simulation_budget = simulation_budget
        self.num_sampled_actions = num_sampled_actions
        self.board_type = board_type
        self.num_players = num_players
        self.neural_net = neural_net
        self.device = device
        self.max_moves = max_moves_per_game
        self.temperature = temperature
        self.temperature_threshold = temperature_threshold

        # Rules engine
        self.engine = DefaultRulesEngine()

        # Statistics
        self.total_nn_calls = 0
        self.total_leaves_evaluated = 0

    def run_batch(
        self,
        num_games: int | None = None,
        initial_states: list[GameState] | None = None,
    ) -> list[GameResult]:
        """Run a batch of games to completion.

        Args:
            num_games: Number of games (uses self.num_games if None).
            initial_states: Optional starting states (creates new games if None).

        Returns:
            List of GameResult for each completed game.
        """
        start_time = time.time()
        num_games = num_games or self.num_games

        # Initialize game states
        if initial_states is None:
            game_states = self._create_initial_states(num_games)
        else:
            game_states = [self._init_game_search(i, s) for i, s in enumerate(initial_states)]

        # Main game loop
        active_games = game_states
        while active_games:
            # Step 1: For each active game needing a move, run Gumbel search
            games_needing_moves = [g for g in active_games if not g.done]

            if not games_needing_moves:
                break

            # Run synchronized Sequential Halving across all games
            self._run_synchronized_search(games_needing_moves)

            # Apply selected moves
            for game in games_needing_moves:
                if game.done:
                    continue
                self._apply_best_move(game)

                # Check for game end
                if game.game_state.is_game_over():
                    game.done = True
                    game.winner = game.game_state.winner
                elif game.move_count >= self.max_moves:
                    game.done = True
                    game.winner = None

            # Filter to still-active games
            active_games = [g for g in active_games if not g.done]

        # Build results
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000

        results = []
        for game in game_states:
            # Extract per-move MCTS policies for training pipeline
            policies = [
                m.get("mcts_policy", {}) for m in game.moves_played
            ]
            results.append(GameResult(
                game_idx=game.game_idx,
                winner=game.winner,
                status="completed" if game.done else "timeout",
                move_count=game.move_count,
                moves=game.moves_played,
                duration_ms=duration_ms / num_games,  # Approximate per-game
                initial_state=game.initial_state_serialized,
                mcts_policies=policies,
            ))

        logger.info(
            f"Completed {num_games} games in {duration_ms:.1f}ms, "
            f"NN calls: {self.total_nn_calls}, leaves: {self.total_leaves_evaluated}"
        )

        return results

    def _create_initial_states(self, num_games: int) -> list[GameSearchState]:
        """Create fresh game states."""
        states = []
        for i in range(num_games):
            # Create immutable initial state, then convert to mutable
            immutable = create_initial_state(
                board_type=self.board_type,
                num_players=self.num_players,
            )
            # Serialize initial state for training data export
            initial_serialized = serialize_game_state(immutable)
            mstate = MutableGameState.from_immutable(immutable)
            states.append(GameSearchState(
                game_idx=i,
                game_state=mstate,
                current_player=mstate.current_player,
                initial_state_serialized=initial_serialized,
            ))
        return states

    def _init_game_search(self, idx: int, state: GameState) -> GameSearchState:
        """Initialize search state from an existing game state."""
        mstate = MutableGameState.from_immutable(state)
        return GameSearchState(
            game_idx=idx,
            game_state=mstate,
            current_player=mstate.current_player,
        )

    def _run_synchronized_search(self, games: list[GameSearchState]) -> None:
        """Run Gumbel MCTS search synchronized across all games.

        All games run Sequential Halving together, phase by phase, enabling
        massive batching of NN evaluations.

        Optimization: Each action's child state is evaluated ONCE per phase,
        then the value is weighted by the simulation budget. This avoids
        O(sims) state copies while maintaining quality through proper
        Sequential Halving budget allocation.
        """
        # Step 1: Initialize search for each game (get legal moves, sample top-K)
        for game in games:
            self._init_search_for_game(game)

        # Step 2: Run Sequential Halving phases
        max_phases = int(np.ceil(np.log2(self.num_sampled_actions)))
        budget_per_phase = self.simulation_budget // max(max_phases, 1)

        for phase in range(max_phases):
            # Check if any games still have multiple candidates
            active = [g for g in games if len(g.remaining_actions) > 1]
            if not active:
                break

            # Collect ONE leaf per action (not per simulation) - massive optimization!
            all_leaves: list[LeafRequest] = []
            leaf_sims: list[int] = []  # Track how many sims this leaf represents

            for game in active:
                sims_per_action = max(1, budget_per_phase // len(game.remaining_actions))
                for action_idx, action in enumerate(game.remaining_actions):
                    # Create ONE leaf request per action
                    leaf = self._create_leaf_request(game, action_idx, 0)
                    if leaf:
                        all_leaves.append(leaf)
                        leaf_sims.append(sims_per_action)

            # Batch evaluate all leaves (one per action across all games)
            if all_leaves:
                values = self._batch_evaluate_leaves(all_leaves)

                # Build game_idx -> game lookup (games list may be filtered)
                game_lookup = {g.game_idx: g for g in active}

                # Distribute values back to actions, weighted by simulation count
                for leaf, value, sims in zip(all_leaves, values, leaf_sims):
                    game = game_lookup.get(leaf.game_idx)
                    if game and leaf.action_idx < len(game.remaining_actions):
                        action = game.remaining_actions[leaf.action_idx]
                        # Weight the value by simulation count
                        action.visit_count += sims
                        action.total_value += value * sims

            # Halve candidates for next phase
            for game in active:
                if len(game.remaining_actions) > 1:
                    max_visits = max(a.visit_count for a in game.remaining_actions)
                    game.remaining_actions.sort(
                        key=lambda a: a.completed_q(max_visits, use_simple_additive=True),
                        reverse=True
                    )
                    game.remaining_actions = game.remaining_actions[:max(1, len(game.remaining_actions) // 2)]

    def _init_search_for_game(self, game: GameSearchState) -> None:
        """Initialize Gumbel-Top-K action sampling for a game."""
        # Get legal moves
        immutable = game.game_state.to_immutable()
        legal_moves = self.engine.get_valid_moves(immutable, game.current_player)

        if not legal_moves:
            game.done = True
            return

        if len(legal_moves) == 1:
            # Only one move - no search needed
            game.actions = [GumbelAction.from_gumbel_score(legal_moves[0], 0.0)]
            game.remaining_actions = game.actions
            return

        # Get policy logits from NN (if available)
        if self.neural_net:
            try:
                _, policy = self.neural_net.evaluate_batch([immutable])
                policy_vec = policy[0] if len(policy) > 0 else None
            except Exception as e:
                logger.warning(f"Policy eval failed: {e}")
                policy_vec = None
        else:
            policy_vec = None

        # Sample Gumbel-Top-K
        k = min(self.num_sampled_actions, len(legal_moves))
        actions = []

        for move in legal_moves:
            # Get policy logit
            if policy_vec is not None and self.neural_net:
                idx = self.neural_net.encode_move(move, immutable.board)
                if idx >= 0 and idx < len(policy_vec):
                    logit = np.log(max(float(policy_vec[idx]), 1e-10))
                else:
                    logit = -10.0
            else:
                logit = 0.0  # Uniform prior

            # Add Gumbel noise
            gumbel = -np.log(-np.log(np.random.uniform() + 1e-10) + 1e-10)
            score = logit + gumbel

            actions.append(GumbelAction.from_gumbel_score(move, score))

        # Sort by Gumbel score (perturbed_value) and take top-K
        actions.sort(key=lambda a: a.perturbed_value, reverse=True)
        game.actions = actions[:k]
        game.remaining_actions = list(game.actions)

    def _create_leaf_request(
        self,
        game: GameSearchState,
        action_idx: int,
        simulation_idx: int,
    ) -> LeafRequest | None:
        """Create a leaf evaluation request for one simulation."""
        if action_idx >= len(game.remaining_actions):
            return None

        action = game.remaining_actions[action_idx]

        # Apply move to get resulting state
        try:
            # Copy state by converting to immutable and back
            immutable_parent = game.game_state.to_immutable()
            child_state = MutableGameState.from_immutable(immutable_parent)
            child_state.make_move(action.move)
            immutable = child_state.to_immutable()
        except Exception as e:
            logger.warning(f"Failed to apply move: {e}")
            return None

        # Check if terminal - compute value directly without NN
        if child_state.is_game_over():
            winner = child_state.winner
            if winner is None:
                # Draw
                terminal_value = 0.0
            elif winner == game.current_player:
                # Current player (at root) wins
                terminal_value = 1.0
            else:
                # Current player loses
                terminal_value = -1.0

            return LeafRequest(
                game_idx=game.game_idx,
                action_idx=action_idx,
                simulation_idx=simulation_idx,
                game_state=immutable,
                is_opponent_perspective=False,  # N/A for terminal
                is_terminal=True,
                terminal_value=terminal_value,
            )

        return LeafRequest(
            game_idx=game.game_idx,
            action_idx=action_idx,
            simulation_idx=simulation_idx,
            game_state=immutable,
            is_opponent_perspective=(
                child_state.current_player != game.current_player
            ),
        )

    def _batch_evaluate_leaves(self, leaves: list[LeafRequest]) -> list[float]:
        """Evaluate all leaves in a single batch.

        Terminal leaves use their precomputed terminal_value directly.
        Non-terminal leaves are sent to the neural network for evaluation.
        """
        if not leaves:
            return []

        # Separate terminal and non-terminal leaves
        non_terminal_indices = []
        non_terminal_leaves = []
        for i, leaf in enumerate(leaves):
            if not leaf.is_terminal:
                non_terminal_indices.append(i)
                non_terminal_leaves.append(leaf)

        # Initialize result with terminal values
        result = [0.0] * len(leaves)
        for i, leaf in enumerate(leaves):
            if leaf.is_terminal:
                result[i] = leaf.terminal_value

        # Evaluate non-terminal leaves with NN
        if non_terminal_leaves and self.neural_net:
            self.total_nn_calls += 1
            self.total_leaves_evaluated += len(non_terminal_leaves)

            states = [leaf.game_state for leaf in non_terminal_leaves]

            try:
                values, _ = self.neural_net.evaluate_batch(states)
            except Exception as e:
                logger.warning(f"Batch eval failed: {e}")
                values = [0.0] * len(non_terminal_leaves)

            # Apply perspective flipping and store results
            for idx, (orig_idx, leaf) in enumerate(zip(non_terminal_indices, non_terminal_leaves)):
                v = float(values[idx]) if idx < len(values) else 0.0
                if leaf.is_opponent_perspective:
                    v = -v
                result[orig_idx] = v

        return result

    def _apply_best_move(self, game: GameSearchState) -> None:
        """Apply the best action found by search."""
        if not game.remaining_actions:
            game.done = True
            return

        # Select best action
        best = game.remaining_actions[0]
        if len(game.remaining_actions) > 1:
            max_visits = max(a.visit_count for a in game.remaining_actions)
            best = max(
                game.remaining_actions,
                key=lambda a: a.completed_q(max_visits, use_simple_additive=True)
            )

        # Apply move
        try:
            game.game_state.make_move(best.move)
            game.move_count += 1

            # Build visit-fraction MCTS policy for training
            # Key format matches generate_gumbel_selfplay.py for pipeline compatibility
            total_visits = sum(a.visit_count for a in game.actions)
            mcts_policy: dict[str, float] = {}
            for a in game.actions:
                if a.visit_count > 0 and total_visits > 0:
                    key = a.move.type.value
                    if a.move.from_pos:
                        key += f"_{a.move.from_pos.x},{a.move.from_pos.y}"
                    if a.move.to:
                        key += f"_{a.move.to.x},{a.move.to.y}"
                    mcts_policy[key] = a.visit_count / total_visits

            # Record move with full policy info (format matches generate_gumbel_selfplay.py)
            move_data = best.move.model_dump(by_alias=True, exclude_none=True, mode="json")
            move_data["moveNumber"] = game.move_count
            move_data["mcts_policy"] = mcts_policy
            game.moves_played.append(move_data)

            # Update current player
            game.current_player = game.game_state.current_player

        except Exception as e:
            logger.warning(f"Failed to apply move: {e}")
            game.done = True
