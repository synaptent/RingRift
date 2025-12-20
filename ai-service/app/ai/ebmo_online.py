"""EBMO Online Learning Module.

Implements continuous learning for EBMO without batch training:
- TD-Energy updates during gameplay
- Rolling buffer of recent games for stability
- Outcome-weighted contrastive loss
- Real-time weight updates after each game

Usage:
    from app.ai.ebmo_online import EBMOOnlineLearner

    learner = EBMOOnlineLearner(model, device='mps')

    # During play, record states and moves
    learner.record_transition(state, move, player, next_state)

    # After game, update with outcome
    learner.update_from_game(winner)
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models import GameState, Move
from .ebmo_network import ActionFeatureExtractor, EBMONetwork

logger = logging.getLogger(__name__)


@dataclass
class Transition:
    """A single state-action transition."""
    state: GameState
    move: Move
    player: int
    next_state: GameState | None
    energy: float | None = None  # Cached energy value


@dataclass
class GameRecord:
    """Complete record of a played game."""
    transitions: list[Transition]
    winner: int | None
    players: tuple[int, ...]


@dataclass
class EBMOOnlineConfig:
    """Configuration for online learning."""
    # Buffer settings
    buffer_size: int = 20  # Number of games to keep
    min_games_before_update: int = 1  # Minimum games before first update

    # Learning rate
    learning_rate: float = 1e-5  # Very low for stability

    # TD-Energy settings
    td_lambda: float = 0.9  # Eligibility trace decay
    gamma: float = 0.99  # Discount factor

    # Outcome weighting
    winner_energy_target: float = -1.0  # Target energy for winner's moves
    loser_energy_target: float = 1.0  # Target energy for loser's moves

    # Update settings
    batch_size: int = 8  # Games to sample per update
    gradient_clip: float = 1.0  # Gradient clipping norm

    # Loss weights
    td_weight: float = 0.5  # Weight for TD-energy loss
    outcome_weight: float = 0.5  # Weight for outcome-contrastive loss

    # Board settings
    board_size: int = 8


class EBMOOnlineLearner:
    """Online learning for EBMO using TD-Energy and outcome-based updates.

    Key features:
    1. TD-Energy: Updates energy to be consistent across trajectories
       E(s,a) should predict min E(s', a') over next state actions

    2. Outcome-weighted: Winner's moves -> low energy, loser's -> high energy

    3. Rolling buffer: Maintains last N games for stability

    4. Gradient accumulation: Accumulates gradients during game, applies after
    """

    def __init__(
        self,
        network: EBMONetwork,
        device: torch.device | str = 'cpu',
        config: EBMOOnlineConfig | None = None,
    ) -> None:
        """Initialize online learner.

        Args:
            network: EBMO network to train
            device: Device for computation
            config: Online learning configuration
        """
        self.network = network
        self.device = torch.device(device) if isinstance(device, str) else device
        self.config = config or EBMOOnlineConfig()

        # Move network to device and set to train mode
        self.network.to(self.device)

        # Feature extractor
        self.feature_extractor = ActionFeatureExtractor(self.config.board_size)

        # Optimizer with low learning rate for stability
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=1e-4,
        )

        # Rolling buffer of recent games
        self.game_buffer: deque[GameRecord] = deque(maxlen=self.config.buffer_size)

        # Current game being recorded
        self._current_transitions: list[Transition] = []
        self._current_players: set[int] = set()

        # Eligibility traces for TD updates
        self._eligibility_traces: dict[str, torch.Tensor] = {}

        # Statistics
        self.games_trained: int = 0
        self.total_updates: int = 0
        self.recent_losses: deque[float] = deque(maxlen=100)

    def record_transition(
        self,
        state: GameState,
        move: Move,
        player: int,
        next_state: GameState | None = None,
    ) -> None:
        """Record a state-action transition during gameplay.

        Call this after each move is made.

        Args:
            state: State before the move
            move: Move that was played
            player: Player who made the move
            next_state: State after the move (optional, for TD)
        """
        transition = Transition(
            state=state,
            move=move,
            player=player,
            next_state=next_state,
        )
        self._current_transitions.append(transition)
        self._current_players.add(player)

    def end_game(self, winner: int | None) -> None:
        """Signal end of game and add to buffer.

        Args:
            winner: Winning player number, or None for draw
        """
        if not self._current_transitions:
            return

        record = GameRecord(
            transitions=self._current_transitions.copy(),
            winner=winner,
            players=tuple(sorted(self._current_players)),
        )
        self.game_buffer.append(record)

        # Clear current game
        self._current_transitions = []
        self._current_players = set()

    def update_from_game(self, winner: int | None) -> dict[str, float]:
        """Complete game and run update step.

        This is the main API for online learning. Call after each game ends.

        Args:
            winner: Winning player number

        Returns:
            Dict with loss metrics
        """
        # Add current game to buffer
        self.end_game(winner)

        # Skip update if not enough games
        if len(self.game_buffer) < self.config.min_games_before_update:
            return {"status": "buffering", "buffer_size": len(self.game_buffer)}

        # Sample games for update
        games_to_sample = min(self.config.batch_size, len(self.game_buffer))
        sampled_games = list(self.game_buffer)[-games_to_sample:]

        # Compute and apply gradients
        metrics = self._update_from_games(sampled_games)

        self.games_trained += 1
        self.total_updates += 1

        return metrics

    def _update_from_games(self, games: list[GameRecord]) -> dict[str, float]:
        """Compute update from sampled games.

        Args:
            games: List of game records to learn from

        Returns:
            Dict with loss metrics
        """
        self.network.train()
        self.optimizer.zero_grad()

        total_td_loss = 0.0
        total_outcome_loss = 0.0
        total_samples = 0

        for game in games:
            if not game.transitions:
                continue

            # Compute losses for this game
            td_loss, outcome_loss, n_samples = self._compute_game_losses(game)

            total_td_loss += td_loss
            total_outcome_loss += outcome_loss
            total_samples += n_samples

        if total_samples == 0:
            self.network.eval()
            return {"status": "no_samples"}

        # Combine losses
        avg_td_loss = total_td_loss / total_samples
        avg_outcome_loss = total_outcome_loss / total_samples

        total_loss = (
            self.config.td_weight * avg_td_loss +
            self.config.outcome_weight * avg_outcome_loss
        )

        # Backward and clip gradients
        total_loss.backward()

        if self.config.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.network.parameters(),
                self.config.gradient_clip,
            )

        # Update weights
        self.optimizer.step()

        self.network.eval()

        # Track loss
        loss_value = total_loss.item()
        self.recent_losses.append(loss_value)

        return {
            "total_loss": loss_value,
            "td_loss": avg_td_loss.item() if isinstance(avg_td_loss, torch.Tensor) else avg_td_loss,
            "outcome_loss": avg_outcome_loss.item() if isinstance(avg_outcome_loss, torch.Tensor) else avg_outcome_loss,
            "samples": total_samples,
            "games": len(games),
        }

    def _compute_game_losses(
        self,
        game: GameRecord,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Compute TD and outcome losses for a single game.

        Args:
            game: Game record

        Returns:
            (td_loss, outcome_loss, num_samples)
        """
        td_losses = []
        outcome_losses = []

        for i, trans in enumerate(game.transitions):
            # Encode state and action
            state_embed = self.network.encode_state_from_game(
                trans.state,
                trans.player,
                self.device,
            )

            move_features = self.feature_extractor.extract_tensor(
                [trans.move],
                self.device,
            )
            action_embed = self.network.encode_action(move_features).squeeze(0)

            # Current energy
            current_energy = self.network.compute_energy(
                state_embed.unsqueeze(0),
                action_embed.unsqueeze(0),
            ).squeeze()

            # TD-Energy loss: E(s,a) should predict min E(s', a')
            if trans.next_state is not None and i + 1 < len(game.transitions):
                next_trans = game.transitions[i + 1]

                with torch.no_grad():
                    next_state_embed = self.network.encode_state_from_game(
                        trans.next_state,
                        next_trans.player,
                        self.device,
                    )

                    # Get legal moves from next state (simplified: use recorded move)
                    next_move_features = self.feature_extractor.extract_tensor(
                        [next_trans.move],
                        self.device,
                    )
                    next_action_embed = self.network.encode_action(next_move_features)

                    # Target: discounted next energy
                    next_energy = self.network.compute_energy(
                        next_state_embed.unsqueeze(0),
                        next_action_embed,
                    ).squeeze()

                    td_target = self.config.gamma * next_energy

                td_loss = (current_energy - td_target) ** 2
                td_losses.append(td_loss)

            # Outcome-weighted loss
            if game.winner is not None:
                if trans.player == game.winner:
                    target = self.config.winner_energy_target
                else:
                    target = self.config.loser_energy_target

                outcome_loss = (current_energy - target) ** 2
                outcome_losses.append(outcome_loss)

        # Aggregate losses
        total_td = sum(td_losses) if td_losses else torch.tensor(0.0, device=self.device)
        total_outcome = sum(outcome_losses) if outcome_losses else torch.tensor(0.0, device=self.device)
        n_samples = max(len(td_losses), len(outcome_losses), 1)

        return total_td, total_outcome, n_samples

    def td_step_update(
        self,
        state: GameState,
        move: Move,
        player: int,
        next_state: GameState,
        next_move: Move,
        reward: float = 0.0,
    ) -> float:
        """Single TD-style update step (optional, for per-move updates).

        This can be called after each move for more aggressive learning.
        Not recommended unless you need real-time adaptation.

        Args:
            state: Current state
            move: Action taken
            player: Acting player
            next_state: Resulting state
            next_move: Best/actual next move
            reward: Immediate reward (usually 0 except at game end)

        Returns:
            TD error (for logging)
        """
        self.network.train()

        # Encode current (s, a)
        state_embed = self.network.encode_state_from_game(state, player, self.device)
        move_features = self.feature_extractor.extract_tensor([move], self.device)
        action_embed = self.network.encode_action(move_features).squeeze(0)

        current_energy = self.network.compute_energy(
            state_embed.unsqueeze(0),
            action_embed.unsqueeze(0),
        ).squeeze()

        # Encode next (s', a')
        with torch.no_grad():
            next_player = next_state.current_player
            next_state_embed = self.network.encode_state_from_game(
                next_state, next_player, self.device
            )
            next_move_features = self.feature_extractor.extract_tensor([next_move], self.device)
            next_action_embed = self.network.encode_action(next_move_features).squeeze(0)

            next_energy = self.network.compute_energy(
                next_state_embed.unsqueeze(0),
                next_action_embed.unsqueeze(0),
            ).squeeze()

            # TD target: reward + gamma * next_energy
            td_target = reward + self.config.gamma * next_energy

        # TD error
        td_error = current_energy - td_target
        loss = td_error ** 2

        # Update with small learning rate
        self.optimizer.zero_grad()
        loss.backward()

        if self.config.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.network.parameters(),
                self.config.gradient_clip,
            )

        self.optimizer.step()
        self.network.eval()

        return td_error.item()

    def get_stats(self) -> dict[str, Any]:
        """Get training statistics.

        Returns:
            Dict with online learning stats
        """
        return {
            "games_trained": self.games_trained,
            "total_updates": self.total_updates,
            "buffer_size": len(self.game_buffer),
            "buffer_capacity": self.config.buffer_size,
            "avg_recent_loss": (
                sum(self.recent_losses) / len(self.recent_losses)
                if self.recent_losses else 0.0
            ),
            "learning_rate": self.config.learning_rate,
        }

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint
        """
        torch.save({
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'games_trained': self.games_trained,
            'total_updates': self.total_updates,
            'config': self.config,
        }, path)
        logger.info(f"Saved online learning checkpoint to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint.

        Args:
            path: Path to checkpoint
        """
        from app.utils.torch_utils import safe_load_checkpoint
        checkpoint = safe_load_checkpoint(path, map_location=str(self.device), warn_on_unsafe=False)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.games_trained = checkpoint.get('games_trained', 0)
        self.total_updates = checkpoint.get('total_updates', 0)
        logger.info(f"Loaded online learning checkpoint from {path}")


class EBMOOnlineAI:
    """EBMO AI with integrated online learning.

    Wraps an EBMO_AI and adds online learning capability.

    Usage:
        ai = EBMOOnlineAI(player_number=1, config=config, model_path=model_path)

        # Play games as normal
        move = ai.select_move(state)

        # After each game
        ai.end_game(winner)

        # Periodically save
        ai.save_checkpoint("models/ebmo_online.pt")
    """

    def __init__(
        self,
        player_number: int,
        config: Any,  # AIConfig
        model_path: str | None = None,
        online_config: EBMOOnlineConfig | None = None,
        enable_online_learning: bool = True,
    ) -> None:
        """Initialize online learning AI.

        Args:
            player_number: Player number
            config: AI configuration
            model_path: Path to initial model
            online_config: Online learning configuration
            enable_online_learning: Whether to enable learning
        """
        # Import here to avoid circular dependency
        from .ebmo_ai import EBMO_AI

        self.ai = EBMO_AI(player_number, config, model_path)
        self.enable_online_learning = enable_online_learning

        if enable_online_learning and self.ai.network is not None:
            self.learner = EBMOOnlineLearner(
                self.ai.network,
                device=self.ai.device,
                config=online_config,
            )
        else:
            self.learner = None

        self._last_state: GameState | None = None
        self._last_move: Move | None = None

    def select_move(self, game_state: GameState) -> Move | None:
        """Select move and record for learning.

        Args:
            game_state: Current state

        Returns:
            Selected move
        """
        move = self.ai.select_move(game_state)

        # Record transition for learning
        if self.learner is not None and move is not None:
            # Record previous transition with this state as next_state
            if self._last_state is not None and self._last_move is not None:
                self.learner.record_transition(
                    self._last_state,
                    self._last_move,
                    self._last_state.current_player,
                    game_state,
                )

            self._last_state = game_state
            self._last_move = move

        return move

    def end_game(self, winner: int | None) -> dict[str, float] | None:
        """End game and run learning update.

        Args:
            winner: Winning player

        Returns:
            Learning metrics, or None if learning disabled
        """
        # Record final transition
        if self.learner is not None and self._last_state is not None:
            self.learner.record_transition(
                self._last_state,
                self._last_move,
                self._last_state.current_player,
                None,  # No next state at game end
            )

        self._last_state = None
        self._last_move = None

        if self.learner is not None:
            return self.learner.update_from_game(winner)
        return None

    def reset_for_new_game(self, **kwargs) -> None:
        """Reset for new game."""
        self.ai.reset_for_new_game(**kwargs)
        self._last_state = None
        self._last_move = None

    def save_checkpoint(self, path: str) -> None:
        """Save model with online learning state."""
        if self.learner is not None:
            self.learner.save_checkpoint(path)

    def get_stats(self) -> dict[str, Any]:
        """Get combined stats."""
        stats = self.ai.get_stats()
        if self.learner is not None:
            stats["online_learning"] = self.learner.get_stats()
        return stats

    @property
    def player_number(self) -> int:
        return self.ai.player_number


__all__ = [
    "EBMOOnlineConfig",
    "EBMOOnlineLearner",
    "EBMOOnlineAI",
    "Transition",
    "GameRecord",
]
