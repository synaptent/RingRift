"""GPU-accelerated MCTS selfplay with full training data capture.

This module provides GPU MCTS selfplay that produces high-quality training data
with visit distributions (soft policy targets), state features, and value targets.

Key features:
- Uses MultiTreeMCTS for GPU-accelerated parallel MCTS
- Captures visit distributions as soft policy targets (not just hard move labels)
- Records full state features for each decision point
- Exports directly to NPZ format for training

December 2025: Initial implementation for canonical-quality GPU selfplay.

Usage:
    runner = GPUMCTSSelfplayRunner(config)
    samples = runner.run_batch(num_games=64)
    runner.export_to_npz("training_data.npz", samples)
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from ..models import BoardType, GameState, Move

logger = logging.getLogger(__name__)


@dataclass
class GPUMCTSSelfplayConfig:
    """Configuration for GPU MCTS selfplay."""

    # Board configuration
    board_type: str = "hex8"
    num_players: int = 2

    # MCTS configuration
    num_sampled_actions: int = 16  # K for Gumbel-Top-K
    simulation_budget: int = 64    # Simulations per move
    max_nodes: int = 1024          # Max tree nodes

    # Batch configuration
    batch_size: int = 64           # Games per batch
    max_moves_per_game: int = 300  # Safety limit

    # Feature encoding
    encoder_version: str = "v3"    # v2 or v3
    feature_version: int = 2
    history_length: int = 3

    # Neural network (optional - uses heuristic if None)
    model_path: str | None = None

    # Device
    device: str = "cuda"

    # Recording options
    record_state_snapshots: bool = True  # Store full states
    sample_every: int = 1                # Sample every N moves


@dataclass
class SelfplaySample:
    """A single training sample from selfplay."""

    features: np.ndarray           # (C, H, W) spatial features
    globals_vec: np.ndarray        # (G,) global features
    policy_indices: np.ndarray     # (K,) move indices with probability
    policy_values: np.ndarray      # (K,) corresponding probabilities
    value: float                   # Game outcome from this player's perspective
    player: int                    # Player number
    move_number: int               # Move index in game
    game_id: str                   # Game identifier


@dataclass
class GameRecord:
    """Complete record of a selfplay game."""

    game_id: str
    samples: list[SelfplaySample] = field(default_factory=list)
    winner: int | None = None
    total_moves: int = 0
    termination_reason: str = "normal"


class GPUMCTSSelfplayRunner:
    """GPU MCTS selfplay runner with full training data capture.

    This class runs selfplay games using GPU-accelerated MultiTreeMCTS,
    capturing high-quality training data with visit distributions.
    """

    def __init__(self, config: GPUMCTSSelfplayConfig):
        """Initialize the runner.

        Args:
            config: Selfplay configuration
        """
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        # Lazy-initialized components
        self._mcts: Any = None
        self._encoder: Any = None
        self._neural_net: Any = None
        self._engine: Any = None

        logger.info(
            f"GPUMCTSSelfplayRunner initialized: "
            f"board={config.board_type}, players={config.num_players}, "
            f"batch={config.batch_size}, device={self.device}"
        )

    def _init_components(self) -> None:
        """Lazy-initialize MCTS, encoder, and engine."""
        if self._mcts is not None:
            return

        from ..ai.tensor_gumbel_tree import MultiTreeMCTS, MultiTreeMCTSConfig
        from ..models import BoardType
        from ..rules.default_engine import DefaultRulesEngine
        from .encoding import get_encoder_for_board_type

        # Parse board type
        board_type_str = self.config.board_type.lower()
        if board_type_str == "hex8":
            board_type = BoardType.HEX8
            board_size = 9
        elif board_type_str == "hexagonal":
            board_type = BoardType.HEXAGONAL
            board_size = 25
        elif board_type_str == "square8":
            board_type = BoardType.SQUARE8
            board_size = 8
        elif board_type_str == "square19":
            board_type = BoardType.SQUARE19
            board_size = 19
        else:
            raise ValueError(f"Unknown board type: {board_type_str}")

        self._board_type = board_type
        self._board_size = board_size

        # Initialize MCTS
        mcts_config = MultiTreeMCTSConfig(
            num_sampled_actions=self.config.num_sampled_actions,
            simulation_budget=self.config.simulation_budget,
            max_nodes=self.config.max_nodes,
            max_actions=256,  # Max valid moves
            device=str(self.device),
        )
        self._mcts = MultiTreeMCTS(mcts_config)

        # Initialize encoder
        self._encoder = get_encoder_for_board_type(
            board_type,
            version=self.config.encoder_version,
            feature_version=self.config.feature_version,
        )

        # Initialize engine
        self._engine = DefaultRulesEngine()

        # Load neural network if specified
        if self.config.model_path and os.path.exists(self.config.model_path):
            self._load_neural_net()

        logger.info(
            f"Components initialized: mcts={type(self._mcts).__name__}, "
            f"encoder={type(self._encoder).__name__}"
        )

    def _load_neural_net(self) -> None:
        """Load neural network for MCTS evaluation."""
        try:
            from ..ai.neural_net import NeuralNetAI

            self._neural_net = NeuralNetAI(
                board_type=self._board_type,
                num_players=self.config.num_players,
                model_path=self.config.model_path,
            )
            logger.info(f"Loaded neural network from {self.config.model_path}")
        except Exception as e:
            logger.warning(f"Failed to load neural network: {e}")
            self._neural_net = None

    def run_batch(self, num_games: int | None = None) -> list[GameRecord]:
        """Run a batch of selfplay games.

        Args:
            num_games: Number of games to run (default: config.batch_size)

        Returns:
            List of GameRecord objects with training samples
        """
        self._init_components()

        num_games = num_games or self.config.batch_size

        from ..models import GameStatus
        from ..training.initial_state import create_initial_state

        # Also handle IN_PROGRESS if it exists
        try:
            IN_PROGRESS = GameStatus.IN_PROGRESS
        except AttributeError:
            IN_PROGRESS = None

        start_time = time.time()

        # Initialize games
        games: list[GameRecord] = []
        states: list["GameState"] = []
        state_histories: list[list[np.ndarray]] = []

        for i in range(num_games):
            game_id = f"gpu_mcts_{int(time.time())}_{i}"
            games.append(GameRecord(game_id=game_id))

            state = create_initial_state(
                board_type=self._board_type,
                num_players=self.config.num_players,
            )
            states.append(state)
            state_histories.append([])

        # Track active games
        active_mask = [True] * num_games
        move_counts = [0] * num_games

        # Main game loop
        while any(active_mask):
            # Collect active games
            active_indices = [i for i, a in enumerate(active_mask) if a]
            active_states = [states[i] for i in active_indices]

            if not active_states:
                break

            # Run MCTS for all active games
            try:
                moves, policies = self._mcts.search_batch(
                    active_states,
                    self._neural_net
                )
            except Exception as e:
                logger.error(f"MCTS search failed: {e}")
                break

            # Process results for each active game
            for batch_idx, game_idx in enumerate(active_indices):
                state = states[game_idx]
                move = moves[batch_idx]
                policy_dict = policies[batch_idx]

                # Record sample if this is a sampling point
                if move_counts[game_idx] % self.config.sample_every == 0:
                    sample = self._create_sample(
                        state=state,
                        policy_dict=policy_dict,
                        state_history=state_histories[game_idx],
                        game_id=games[game_idx].game_id,
                        move_number=move_counts[game_idx],
                    )
                    if sample is not None:
                        games[game_idx].samples.append(sample)

                # Apply move
                try:
                    new_state = self._engine.apply_move(state, move)
                    states[game_idx] = new_state
                    move_counts[game_idx] += 1

                    # Update feature history
                    if self._encoder is not None:
                        features, _ = self._encoder.encode_state(state)
                        state_histories[game_idx].append(features)
                        if len(state_histories[game_idx]) > self.config.history_length:
                            state_histories[game_idx].pop(0)

                    # Check termination (COMPLETED or not ACTIVE)
                    if new_state.game_status == GameStatus.COMPLETED:
                        active_mask[game_idx] = False
                        games[game_idx].winner = new_state.winner
                        games[game_idx].total_moves = move_counts[game_idx]
                        games[game_idx].termination_reason = "normal"

                        # Update value targets for all samples
                        self._assign_values(games[game_idx], new_state)
                    elif new_state.game_status not in (GameStatus.ACTIVE,) and (IN_PROGRESS is None or new_state.game_status != IN_PROGRESS):
                        # Other terminal states (draw, etc.)
                        active_mask[game_idx] = False
                        games[game_idx].total_moves = move_counts[game_idx]
                        games[game_idx].termination_reason = str(new_state.game_status.value)

                    # Safety limit
                    if move_counts[game_idx] >= self.config.max_moves_per_game:
                        active_mask[game_idx] = False
                        games[game_idx].total_moves = move_counts[game_idx]
                        games[game_idx].termination_reason = "move_limit"

                except Exception as e:
                    import traceback
                    logger.warning(f"Move application failed for game {game_idx}: {e}")
                    logger.debug(f"Traceback: {traceback.format_exc()}")
                    active_mask[game_idx] = False
                    games[game_idx].termination_reason = f"error: {e}"

        elapsed = time.time() - start_time
        total_samples = sum(len(g.samples) for g in games)
        completed = sum(1 for g in games if g.termination_reason == "normal")

        logger.info(
            f"Completed {completed}/{num_games} games in {elapsed:.1f}s, "
            f"{total_samples} samples, {total_samples/elapsed:.1f} samples/s"
        )

        return games

    def _create_sample(
        self,
        state: "GameState",
        policy_dict: dict[str, float],
        state_history: list[np.ndarray],
        game_id: str,
        move_number: int,
    ) -> SelfplaySample | None:
        """Create a training sample from current position.

        Args:
            state: Current game state
            policy_dict: MCTS policy distribution (move_key -> probability)
            state_history: Previous feature frames
            game_id: Game identifier
            move_number: Current move number

        Returns:
            SelfplaySample or None if encoding fails
        """
        if self._encoder is None:
            return None

        try:
            from ..ai.neural_net import encode_move_for_board

            # Encode current state
            features, globals_vec = self._encoder.encode_state(state)

            # Stack with history
            hist = list(state_history[-self.config.history_length:])
            while len(hist) < self.config.history_length:
                hist.insert(0, np.zeros_like(features))
            stacked_features = np.concatenate([features, *hist], axis=0)

            # Convert policy dict to indices and values
            # Build key->move lookup once (O(n) instead of O(n*m))
            valid_moves = self._engine.get_valid_moves(state, state.current_player)
            key_to_move = {self._move_to_key(m): m for m in valid_moves}

            policy_indices = []
            policy_values = []

            for move_key, prob in policy_dict.items():
                if prob <= 0:
                    continue
                move = key_to_move.get(move_key)
                if move is not None:
                    try:
                        idx = encode_move_for_board(move, state.board)
                        if idx >= 0:
                            policy_indices.append(idx)
                            policy_values.append(prob)
                    except Exception:
                        pass

            if not policy_indices:
                return None

            return SelfplaySample(
                features=stacked_features.astype(np.float32),
                globals_vec=globals_vec.astype(np.float32),
                policy_indices=np.array(policy_indices, dtype=np.int64),
                policy_values=np.array(policy_values, dtype=np.float32),
                value=0.0,  # Assigned later based on outcome
                player=state.current_player,
                move_number=move_number,
                game_id=game_id,
            )

        except Exception as e:
            logger.debug(f"Sample creation failed: {e}")
            return None

    def _move_to_key(self, move: "Move") -> str:
        """Convert a Move to a string key for policy dict matching.

        Must match MultiTreeMCTS._move_to_key() format exactly:
        - from_str is "none" when from_pos is None
        - Includes _{placement_count} suffix when present
        """
        from_str = f"{move.from_pos.x},{move.from_pos.y}" if move.from_pos else "none"
        to_str = f"{move.to.x},{move.to.y}" if move.to else "none"
        count_str = f"_{move.placement_count}" if hasattr(move, 'placement_count') and move.placement_count else ""
        return f"{move.type.value}_{from_str}_{to_str}{count_str}"

    def _assign_values(self, game: GameRecord, final_state: "GameState") -> None:
        """Assign value targets to all samples based on game outcome.

        Uses rank-aware values for multiplayer games.
        """
        from ..models import GameStatus

        if final_state.game_status != GameStatus.COMPLETED:
            # Draw or incomplete - all zeros
            for sample in game.samples:
                sample.value = 0.0
            return

        # Compute ranks from final state
        player_ranks = {}
        if final_state.winner:
            player_ranks[final_state.winner] = 1

        # Rank remaining players by score
        remaining = [p for p in final_state.players if p.player_number != final_state.winner]
        sorted_remaining = sorted(
            remaining,
            key=lambda p: (p.territory_spaces, -p.eliminated_rings),
            reverse=True
        )
        for rank, player in enumerate(sorted_remaining, start=2):
            player_ranks[player.player_number] = rank

        num_players = len(final_state.players)

        # Assign values based on rank
        for sample in game.samples:
            rank = player_ranks.get(sample.player, num_players)
            if num_players <= 1:
                sample.value = 0.0
            else:
                # Linear: 1st=+1, last=-1
                sample.value = 1.0 - 2.0 * (rank - 1) / (num_players - 1)

    def export_to_npz(
        self,
        output_path: str,
        games: list[GameRecord],
        encoder_version: str | None = None,
    ) -> int:
        """Export game samples to NPZ format for training.

        Args:
            output_path: Output file path
            games: List of game records to export
            encoder_version: Encoder version metadata (default: from config)

        Returns:
            Number of samples exported
        """
        # Collect all samples
        all_samples = []
        for game in games:
            all_samples.extend(game.samples)

        if not all_samples:
            logger.warning("No samples to export")
            return 0

        # Stack arrays
        features = np.stack([s.features for s in all_samples])
        globals_arr = np.stack([s.globals_vec for s in all_samples])
        values = np.array([s.value for s in all_samples], dtype=np.float32)

        # Policy arrays (variable length - pad to max)
        max_policy_len = max(len(s.policy_indices) for s in all_samples)
        policy_indices = np.zeros((len(all_samples), max_policy_len), dtype=np.int64)
        policy_values = np.zeros((len(all_samples), max_policy_len), dtype=np.float32)

        for i, s in enumerate(all_samples):
            policy_indices[i, :len(s.policy_indices)] = s.policy_indices
            policy_values[i, :len(s.policy_values)] = s.policy_values

        # Metadata
        effective_encoder = encoder_version or self.config.encoder_version

        # Save
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        np.savez_compressed(
            output_path,
            features=features,
            globals=globals_arr,
            values=values,
            policy_indices=policy_indices,
            policy_values=policy_values,
            # Metadata
            board_type=np.asarray(self.config.board_type),
            board_size=np.asarray(self._board_size),
            history_length=np.asarray(self.config.history_length),
            feature_version=np.asarray(self.config.feature_version),
            encoder_version=np.asarray(effective_encoder),
            in_channels=np.asarray(features.shape[1]),
            export_version=np.asarray("2.1"),
            source=np.asarray("gpu_mcts_selfplay"),
        )

        logger.info(f"Exported {len(all_samples)} samples to {output_path}")
        return len(all_samples)


def run_gpu_mcts_selfplay(
    board_type: str = "hex8",
    num_players: int = 2,
    num_games: int = 64,
    output_path: str | None = None,
    model_path: str | None = None,
    device: str = "cuda",
    encoder_version: str = "v3",
) -> list[GameRecord]:
    """Convenience function to run GPU MCTS selfplay.

    Args:
        board_type: Board type (hex8, hexagonal, square8, square19)
        num_players: Number of players
        num_games: Number of games to run
        output_path: Optional NPZ output path
        model_path: Optional neural network model path
        device: Device to use (cuda, cpu)
        encoder_version: Feature encoder version (v2, v3)

    Returns:
        List of game records
    """
    config = GPUMCTSSelfplayConfig(
        board_type=board_type,
        num_players=num_players,
        batch_size=num_games,
        model_path=model_path,
        device=device,
        encoder_version=encoder_version,
    )

    runner = GPUMCTSSelfplayRunner(config)
    games = runner.run_batch(num_games)

    if output_path:
        runner.export_to_npz(output_path, games)

    return games


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GPU MCTS Selfplay")
    parser.add_argument("--board-type", default="hex8", help="Board type")
    parser.add_argument("--num-players", type=int, default=2, help="Number of players")
    parser.add_argument("--num-games", type=int, default=64, help="Number of games")
    parser.add_argument("--output", help="Output NPZ path")
    parser.add_argument("--model", help="Neural network model path")
    parser.add_argument("--device", default="cuda", help="Device")
    parser.add_argument("--encoder-version", default="v3", help="Encoder version")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    games = run_gpu_mcts_selfplay(
        board_type=args.board_type,
        num_players=args.num_players,
        num_games=args.num_games,
        output_path=args.output,
        model_path=args.model,
        device=args.device,
        encoder_version=args.encoder_version,
    )

    print(f"Completed {len(games)} games")
    total_samples = sum(len(g.samples) for g in games)
    print(f"Total samples: {total_samples}")
