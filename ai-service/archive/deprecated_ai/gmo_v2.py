"""GMO v2 - Enhanced Gradient Move Optimization AI.

.. deprecated:: December 2025
    GMO v2 is deprecated and will be removed in a future version.
    Use GNNPolicyNet or HybridPolicyNet from app.ai.neural_net instead.
    GNN-based approaches show better results than GMO variants.

An enhanced version of GMO with architectural improvements for stronger play.
Available as experimental AI at difficulty D18/D19 via the AI factory.

Improvements over GMO v1 (gmo_ai.py):
1. Larger encoders (256-dim vs 128-dim)
2. Ensemble optimization - multiple gradient paths with voting
3. Temperature scheduling by game phase (early/mid/late)
4. Learned projection network (instead of nearest-neighbor)
5. Attention-based state encoder for board feature relationships

Usage:
    # Via AI factory (recommended)
    from app.ai.factory import create_ai
    ai = create_ai(difficulty=18)  # GMO v2
    ai = create_ai(difficulty=19)  # GMO v2 with higher exploration

    # Direct instantiation
    from app.ai.gmo_v2 import GMOv2AI, GMOv2Config
    config = GMOv2Config(device="cuda")
    ai = GMOv2AI(gmo_config=config)

See also:
    - gmo_ai.py: Original GMO implementation (D13/D14/D17)
    - AI_ARCHITECTURE.md: Full algorithm documentation
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

warnings.warn(
    "app.ai.gmo_v2 is deprecated and will be removed in a future version. "
    "Use GNNPolicyNet or HybridPolicyNet from app.ai.neural_net instead.",
    DeprecationWarning,
    stacklevel=2,
)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.models import AIConfig, GameState, Move, MoveType
from app.utils.torch_utils import safe_load_checkpoint
from .base import BaseAI
from .gmo_ai import NoveltyTracker

logger = logging.getLogger(__name__)


# =============================================================================
# GMO v2 Configuration
# =============================================================================

@dataclass
class GMOv2Config:
    """Configuration for GMO v2 AI."""
    # Larger embedding dimensions (was 128)
    state_dim: int = 256
    move_dim: int = 256
    hidden_dim: int = 512

    # Optimization parameters
    top_k: int = 7  # More candidates (was 5)
    optim_steps: int = 15  # More steps (was 10)
    lr: float = 0.1

    # Information-theoretic parameters
    beta: float = 0.3  # Exploration coefficient
    gamma: float = 0.1  # Novelty coefficient

    # Ensemble parameters
    ensemble_size: int = 3  # Number of parallel optimization paths
    ensemble_voting: str = "soft"  # "hard" or "soft" voting

    # Temperature scheduling
    temp_early_game: float = 1.5  # More exploration early
    temp_mid_game: float = 1.0  # Balanced
    temp_late_game: float = 0.5  # More exploitation late
    early_game_threshold: int = 10  # Move count threshold
    late_game_threshold: int = 40  # Move count threshold

    # MC Dropout parameters
    dropout_rate: float = 0.15  # Slightly higher (was 0.1)
    mc_samples: int = 12  # More samples (was 10)

    # Novelty tracking
    novelty_memory_size: int = 2000  # Larger (was 1000)

    # Learned projection
    use_learned_projection: bool = True
    projection_hidden_dim: int = 256

    # Device
    device: str = "cpu"


# =============================================================================
# Attention-based State Encoder
# =============================================================================

class AttentionStateEncoder(nn.Module):
    """State encoder with self-attention for board features.

    Uses self-attention to capture long-range dependencies on the board,
    then projects to embedding space.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        board_size: int = 8,
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.board_size = board_size
        self.num_positions = board_size * board_size

        # Feature extraction (12 channels per position)
        self.feature_channels = 12
        self.position_features = self.feature_channels

        # Position embedding
        self.position_embed = nn.Embedding(self.num_positions, embed_dim)

        # Input projection (features per position -> embed_dim)
        self.input_proj = nn.Linear(self.feature_channels, embed_dim)

        # Self-attention layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Global pooling + output
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def _extract_board_features(self, state: GameState) -> torch.Tensor:
        """Extract per-position features from game state.

        Returns tensor of shape (num_positions, feature_channels).
        """
        board = state.board
        device = next(self.parameters()).device
        features = torch.zeros(
            self.num_positions,
            self.feature_channels,
            dtype=torch.float32,
            device=device,
        )

        territory_positions = set()
        for territory in board.territories.values():
            for space in territory.spaces:
                territory_positions.add(space.to_key())

        for idx in range(self.num_positions):
            row = idx // self.board_size
            col = idx % self.board_size
            pos_key = f"{col},{row}"

            # Get stack at position
            stack = board.stacks.get(pos_key)

            if stack and stack.stack_height > 0:
                # Stack height (normalized)
                features[idx, 0] = stack.stack_height / 5.0

                # Top ring owner
                if stack.rings:
                    top_owner = stack.rings[-1]
                    if 1 <= top_owner <= 4:
                        features[idx, top_owner] = 1.0

                # Stack control
                if 1 <= stack.controlling_player <= 4:
                    features[idx, 4 + stack.controlling_player] = 1.0

            # Marker presence
            marker = board.markers.get(pos_key)
            if marker:
                features[idx, 9] = 1.0
                features[idx, 10] = marker.player / 4.0

            # Territory info
            if pos_key in territory_positions:
                features[idx, 11] = 1.0

        return features

    def forward(self, state: GameState) -> torch.Tensor:
        """Encode game state to embedding vector."""
        # Extract features (num_positions, features)
        features = self._extract_board_features(state)
        features = features.unsqueeze(0)  # Add batch dim

        # Project features
        x = self.input_proj(features)  # (1, num_positions, embed_dim)

        # Add position embeddings
        positions = torch.arange(self.num_positions, device=features.device)
        pos_embed = self.position_embed(positions).unsqueeze(0)
        x = x + pos_embed

        # Self-attention
        x = self.transformer(x)  # (1, num_positions, embed_dim)

        # Global pooling
        x = x.transpose(1, 2)  # (1, embed_dim, num_positions)
        x = self.global_pool(x).squeeze(-1)  # (1, embed_dim)

        # Output projection
        x = self.output_proj(x)

        return x.squeeze(0)  # (embed_dim,)


# =============================================================================
# Larger Move Encoder
# =============================================================================

class MoveEncoderV2(nn.Module):
    """Enhanced move encoder with larger embeddings."""

    def __init__(
        self,
        embed_dim: int = 256,
        board_size: int = 8,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.board_size = board_size
        self.num_positions = board_size * board_size

        # Embedding tables (larger)
        self.move_type_embed = nn.Embedding(len(MoveType), embed_dim // 4)
        self.from_pos_embed = nn.Embedding(self.num_positions + 1, embed_dim // 4)
        self.to_pos_embed = nn.Embedding(self.num_positions + 1, embed_dim // 4)
        self.count_embed = nn.Embedding(6, embed_dim // 4)  # 0-5 rings

        # Projection with residual connection
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, move: Move) -> torch.Tensor:
        """Encode a move to embedding vector."""
        # Move type
        move_type_idx = list(MoveType).index(move.type) if move.type in MoveType else 0
        device = next(self.parameters()).device
        type_embed = self.move_type_embed(torch.tensor(move_type_idx, device=device))

        # From position
        if move.from_pos:
            from_idx = move.from_pos.y * self.board_size + move.from_pos.x
        else:
            from_idx = self.num_positions  # Null position
        from_embed = self.from_pos_embed(torch.tensor(from_idx, device=device))

        # To position
        if move.to:
            to_idx = move.to.y * self.board_size + move.to.x
        else:
            to_idx = self.num_positions
        to_embed = self.to_pos_embed(torch.tensor(to_idx, device=device))

        # Placement count
        count = min(getattr(move, 'placement_count', 0) or 0, 5)
        count_embed = self.count_embed(torch.tensor(count, device=device))

        # Concatenate and project
        combined = torch.cat([type_embed, from_embed, to_embed, count_embed])
        output = self.proj(combined) + combined  # Residual connection

        return output


# =============================================================================
# Value Network with Learned Projection
# =============================================================================

class GMOv2ValueNet(nn.Module):
    """Enhanced value network with optional learned projection.

    Can output both value prediction and move selection logits.
    """

    def __init__(
        self,
        state_dim: int = 256,
        move_dim: int = 256,
        hidden_dim: int = 512,
        dropout_rate: float = 0.15,
        use_learned_projection: bool = True,
        projection_hidden_dim: int = 256,
    ):
        super().__init__()

        self.use_learned_projection = use_learned_projection

        # Joint encoder for (state, move) -> value
        self.joint_encoder = nn.Sequential(
            nn.Linear(state_dim + move_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        # Value head
        self.value_head = nn.Linear(hidden_dim // 2, 1)

        # Uncertainty head (log variance)
        self.uncertainty_head = nn.Linear(hidden_dim // 2, 1)

        # Learned projection head (optimized_embed -> move logits)
        if use_learned_projection:
            self.projection_net = nn.Sequential(
                nn.Linear(move_dim, projection_hidden_dim),
                nn.ReLU(),
                nn.Linear(projection_hidden_dim, projection_hidden_dim),
                nn.ReLU(),
            )
            # Final layer outputs score for each move (computed dynamically)
            self.projection_score = nn.Linear(projection_hidden_dim + move_dim, 1)

    def forward(
        self,
        state_embed: torch.Tensor,
        move_embed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for value and uncertainty.

        Returns:
            (value, log_variance) tuple
        """
        joint = torch.cat([state_embed, move_embed], dim=-1)
        features = self.joint_encoder(joint)

        value = torch.tanh(self.value_head(features))
        log_var = self.uncertainty_head(features)

        return value.squeeze(-1), log_var.squeeze(-1)

    def project_to_moves(
        self,
        optimized_embed: torch.Tensor,
        legal_move_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Project optimized embedding to move selection scores.

        Args:
            optimized_embed: Optimized move embedding (move_dim,)
            legal_move_embeds: Embeddings of legal moves (num_moves, move_dim)

        Returns:
            Scores for each legal move (num_moves,)
        """
        if not self.use_learned_projection:
            # Fall back to cosine similarity
            return F.cosine_similarity(
                optimized_embed.unsqueeze(0),
                legal_move_embeds,
                dim=-1,
            )

        # Learned projection
        opt_features = self.projection_net(optimized_embed)  # (proj_hidden,)
        opt_features = opt_features.unsqueeze(0).expand(len(legal_move_embeds), -1)

        # Concatenate with each legal move embedding
        combined = torch.cat([opt_features, legal_move_embeds], dim=-1)
        scores = self.projection_score(combined).squeeze(-1)

        return scores


# =============================================================================
# GMO v2 AI
# =============================================================================

class GMOv2AI(BaseAI):
    """Enhanced Gradient Move Optimization AI (v2).

    Improvements:
    - Larger embeddings (256-dim)
    - Attention-based state encoder
    - Ensemble optimization with voting
    - Temperature scheduling by game phase
    - Learned projection network
    """

    def __init__(
        self,
        player_number: int,
        config: AIConfig,
        gmo_config: GMOv2Config | None = None,
    ):
        super().__init__(player_number, config)

        self.gmo_config = gmo_config or GMOv2Config()
        self.device = torch.device(self.gmo_config.device)

        board_size = 8  # Default to square8

        # Initialize enhanced networks
        self.state_encoder = AttentionStateEncoder(
            embed_dim=self.gmo_config.state_dim,
            board_size=board_size,
        ).to(self.device)

        self.move_encoder = MoveEncoderV2(
            embed_dim=self.gmo_config.move_dim,
            board_size=board_size,
        ).to(self.device)

        self.value_net = GMOv2ValueNet(
            state_dim=self.gmo_config.state_dim,
            move_dim=self.gmo_config.move_dim,
            hidden_dim=self.gmo_config.hidden_dim,
            dropout_rate=self.gmo_config.dropout_rate,
            use_learned_projection=self.gmo_config.use_learned_projection,
            projection_hidden_dim=self.gmo_config.projection_hidden_dim,
        ).to(self.device)

        # Novelty tracker with larger memory
        self.novelty_tracker = NoveltyTracker(
            memory_size=self.gmo_config.novelty_memory_size,
            embed_dim=self.gmo_config.move_dim,
        )

        self._is_trained = False

    def _get_exploration_temperature(self, game_state: GameState) -> float:
        """Get exploration temperature based on game phase."""
        move_count = len(game_state.move_history) if game_state.move_history else 0

        if move_count < self.gmo_config.early_game_threshold:
            return self.gmo_config.temp_early_game
        elif move_count > self.gmo_config.late_game_threshold:
            return self.gmo_config.temp_late_game
        else:
            # Linear interpolation
            progress = (move_count - self.gmo_config.early_game_threshold) / \
                      (self.gmo_config.late_game_threshold - self.gmo_config.early_game_threshold)
            return self.gmo_config.temp_mid_game - progress * \
                   (self.gmo_config.temp_mid_game - self.gmo_config.temp_late_game)

    def _estimate_uncertainty(
        self,
        state_embed: torch.Tensor,
        move_embed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Estimate value and uncertainty via MC Dropout.

        Returns:
            (mean_value, entropy, variance) tuple
        """
        self.value_net.train()  # Enable dropout

        values = []
        for _ in range(self.gmo_config.mc_samples):
            value, _ = self.value_net(state_embed, move_embed)
            values.append(value)

        self.value_net.eval()

        values_tensor = torch.stack(values)
        mean_value = values_tensor.mean()
        variance = values_tensor.var()
        entropy = 0.5 * torch.log(2 * np.pi * np.e * (variance + 1e-8))

        return mean_value, entropy, variance

    def _optimize_move_ensemble(
        self,
        state_embed: torch.Tensor,
        initial_embed: torch.Tensor,
        exploration_temp: float,
    ) -> list[torch.Tensor]:
        """Run ensemble of gradient optimization paths.

        Returns list of optimized embeddings from different paths.
        """
        optimized_embeds = []

        for path_idx in range(self.gmo_config.ensemble_size):
            # Add different random perturbations for diversity
            if path_idx > 0:
                noise = torch.randn_like(initial_embed) * 0.1
                move_embed = (initial_embed + noise).clone().requires_grad_(True)
            else:
                move_embed = initial_embed.clone().requires_grad_(True)

            optimizer = torch.optim.Adam([move_embed], lr=self.gmo_config.lr)

            for step in range(self.gmo_config.optim_steps):
                optimizer.zero_grad()

                # Estimate value and uncertainty
                mean_value, _entropy, variance = self._estimate_uncertainty(
                    state_embed, move_embed
                )

                # Novelty bonus
                novelty = self.novelty_tracker.compute_novelty(move_embed.detach())

                # Anneal exploration over steps
                step_factor = 1 - step / self.gmo_config.optim_steps
                exploration_weight = self.gmo_config.beta * step_factor * exploration_temp

                # Combined objective
                objective = mean_value + exploration_weight * torch.sqrt(variance + 1e-8) + \
                           self.gmo_config.gamma * novelty

                loss = -objective
                loss.backward()
                optimizer.step()

            optimized_embeds.append(move_embed.detach())

        return optimized_embeds

    def _ensemble_vote(
        self,
        optimized_embeds: list[torch.Tensor],
        state_embed: torch.Tensor,
        legal_move_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Combine ensemble predictions via voting.

        Returns scores for each legal move.
        """
        all_scores = []

        for opt_embed in optimized_embeds:
            scores = self.value_net.project_to_moves(opt_embed, legal_move_embeds)
            all_scores.append(scores)

        scores_tensor = torch.stack(all_scores)

        if self.gmo_config.ensemble_voting == "hard":
            # Hard voting: each path votes for its top choice
            votes = torch.zeros(len(legal_move_embeds))
            for scores in all_scores:
                top_idx = scores.argmax()
                votes[top_idx] += 1
            return votes
        else:
            # Soft voting: average scores
            return scores_tensor.mean(dim=0)

    def select_move(self, game_state: GameState) -> Move | None:
        """Select best move using enhanced GMO algorithm."""
        from ..game_engine import GameEngine

        current_player = game_state.current_player
        legal_moves = GameEngine.get_valid_moves(game_state, current_player)

        if not legal_moves:
            return None

        if len(legal_moves) == 1:
            return legal_moves[0]

        # Get exploration temperature based on game phase
        exploration_temp = self._get_exploration_temperature(game_state)

        # Encode state
        with torch.no_grad():
            state_embed = self.state_encoder(game_state).to(self.device)

        # Encode all legal moves
        move_embeds = []
        for move in legal_moves:
            with torch.no_grad():
                embed = self.move_encoder(move).to(self.device)
            move_embeds.append(embed)
        move_embeds_tensor = torch.stack(move_embeds)

        # Initial ranking
        initial_scores = []
        for me in move_embeds:
            mean_val, _, var = self._estimate_uncertainty(state_embed, me)
            novelty = self.novelty_tracker.compute_novelty(me)
            score = mean_val + self.gmo_config.beta * torch.sqrt(var + 1e-8) + \
                   self.gmo_config.gamma * novelty
            initial_scores.append(score.item())

        # Get top-k candidates
        top_k_indices = np.argsort(initial_scores)[-self.gmo_config.top_k:][::-1]

        best_move = None
        best_score = float('-inf')

        for idx in top_k_indices:
            # Ensemble optimization
            optimized_embeds = self._optimize_move_ensemble(
                state_embed,
                move_embeds[idx],
                exploration_temp,
            )

            # Ensemble voting to select from legal moves
            scores = self._ensemble_vote(
                optimized_embeds,
                state_embed,
                move_embeds_tensor,
            )

            # Select best projected move
            best_proj_idx = scores.argmax().item()

            # Evaluate projected move
            proj_value, _, proj_var = self._estimate_uncertainty(
                state_embed, move_embeds[best_proj_idx]
            )
            proj_score = proj_value + self.gmo_config.beta * torch.sqrt(proj_var + 1e-8)

            if proj_score.item() > best_score:
                best_score = proj_score.item()
                best_move = legal_moves[best_proj_idx]
                best_embed = move_embeds[best_proj_idx]

        # Update novelty tracker
        if best_move:
            self.novelty_tracker.add(best_embed)

        return best_move

    def evaluate_position(self, game_state: GameState) -> float:
        """Evaluate the current position from this AI's perspective.

        Uses the value network to estimate position value.

        Args:
            game_state: Current game state

        Returns:
            Position evaluation from -1.0 (losing) to 1.0 (winning)
        """
        with torch.no_grad():
            state_embed = self.state_encoder(game_state).to(self.device)
            # Use a null move embedding for position evaluation
            null_move = torch.zeros(self.gmo_config.embed_dim, device=self.device)
            value, _ = self.value_net(state_embed.unsqueeze(0), null_move.unsqueeze(0))
            return value.item()

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load trained model from checkpoint.

        Handles both the standard gmo_v2.py architecture and the simpler
        architecture used by train_gmo_v2.py training script.
        """
        checkpoint = safe_load_checkpoint(
            checkpoint_path,
            map_location=self.device,
            warn_on_unsafe=False,
        )

        if "state_encoder" in checkpoint:
            self.state_encoder.load_state_dict(checkpoint["state_encoder"])
        if "move_encoder" in checkpoint:
            self.move_encoder.load_state_dict(checkpoint["move_encoder"])
        if "value_net" in checkpoint:
            value_net_state = checkpoint["value_net"]

            # Check if this is from train_gmo_v2.py (uses 'net' instead of 'joint_encoder')
            if any(k.startswith("net.") for k in value_net_state.keys()):
                logger.info("Detected train_gmo_v2.py checkpoint format, adapting...")
                # Rebuild value_net with compatible architecture
                self.value_net = self._create_train_compatible_value_net(value_net_state)
            else:
                self.value_net.load_state_dict(value_net_state)

        self._is_trained = True
        logger.info(f"Loaded GMO v2 checkpoint from {checkpoint_path}")

    def _create_train_compatible_value_net(
        self, state_dict: dict[str, torch.Tensor]
    ) -> nn.Module:
        """Create a value net compatible with train_gmo_v2.py checkpoint.

        The training script uses a simpler architecture:
        - net: Sequential(Linear, ReLU, Dropout, Linear, ReLU, Dropout)
        - value_head: Linear
        - log_var_head: Linear
        """
        # Infer dimensions from state dict
        hidden_dim = state_dict["net.0.weight"].shape[0]  # 512
        state_move_dim = state_dict["net.0.weight"].shape[1]  # 512 (256+256)

        class TrainCompatValueNet(nn.Module):
            """Value net matching train_gmo_v2.py architecture."""

            def __init__(self, hidden_dim: int):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(state_move_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.15),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(0.15),
                )
                self.value_head = nn.Linear(hidden_dim // 2, 1)
                self.log_var_head = nn.Linear(hidden_dim // 2, 1)

            def forward(
                self, state_embed: torch.Tensor, move_embed: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor]:
                combined = torch.cat([state_embed, move_embed], dim=-1)
                h = self.net(combined)
                value = self.value_head(h).squeeze(-1)
                log_var = self.log_var_head(h).squeeze(-1)
                return value, log_var

            def project_to_moves(
                self,
                optimized_embed: torch.Tensor,
                legal_move_embeds: torch.Tensor,
            ) -> torch.Tensor:
                """Project optimized embedding to move scores via cosine similarity."""
                return F.cosine_similarity(
                    optimized_embed.unsqueeze(0),
                    legal_move_embeds,
                    dim=-1,
                )

        net = TrainCompatValueNet(hidden_dim)
        net.load_state_dict(state_dict)
        net.to(self.device)
        return net

    def save_checkpoint(self, checkpoint_path: str) -> None:
        """Save model to checkpoint."""
        checkpoint = {
            "state_encoder": self.state_encoder.state_dict(),
            "move_encoder": self.move_encoder.state_dict(),
            "value_net": self.value_net.state_dict(),
            "config": self.gmo_config,
        }
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved GMO v2 checkpoint to {checkpoint_path}")

    def get_parameters(self) -> list[torch.nn.Parameter]:
        """Get all trainable parameters."""
        params = []
        params.extend(self.state_encoder.parameters())
        params.extend(self.move_encoder.parameters())
        params.extend(self.value_net.parameters())
        return params

    # =========================================================================
    # Online Learning Support
    # =========================================================================

    def enable_online_learning(
        self,
        lr: float = 0.00005,  # Very conservative for v2
        buffer_size: int = 300,
        discount: float = 0.99,
        weight_decay: float = 0.02,
        max_grad_norm: float = 0.5,
    ) -> None:
        """Enable continuous learning during play.

        Uses conservative hyperparameters to prevent catastrophic forgetting
        while allowing incremental improvement.

        Args:
            lr: Learning rate for online updates (very low for stability)
            buffer_size: Size of experience replay buffer
            discount: Temporal discount factor
            weight_decay: L2 regularization strength
            max_grad_norm: Maximum gradient norm for clipping
        """
        self._online_buffer: list[tuple[torch.Tensor, torch.Tensor, float]] = []
        self._online_trajectory: list[tuple[torch.Tensor, torch.Tensor, float]] = []
        self._online_lr = lr
        self._online_buffer_size = buffer_size
        self._online_discount = discount
        self._online_weight_decay = weight_decay
        self._online_max_grad_norm = max_grad_norm

        self._online_optimizer = torch.optim.AdamW(
            self.get_parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        self._online_enabled = True
        logger.info(
            f"GMO v2 online learning enabled (lr={lr}, buffer={buffer_size})"
        )

    def disable_online_learning(self) -> None:
        """Disable continuous learning."""
        self._online_enabled = False
        self._online_buffer = []
        self._online_trajectory = []
        logger.info("GMO v2 online learning disabled")

    def is_learning_enabled(self) -> bool:
        """Check if online learning is active."""
        return getattr(self, '_online_enabled', False)

    def record_move(
        self,
        state: GameState,
        move: Move,
        predicted_value: float,
    ) -> None:
        """Record a move during gameplay for later learning."""
        if not self.is_learning_enabled():
            return

        with torch.no_grad():
            state_embed = self.state_encoder(state)
            move_embed = self.move_encoder(move)

        self._online_trajectory.append((
            state_embed.detach().cpu(),
            move_embed.detach().cpu(),
            predicted_value,
        ))

    def update_on_game_end(self, outcome: float) -> float:
        """Update model based on game outcome.

        Args:
            outcome: Game result from this player's perspective
                    (+1 win, 0 draw, -1 loss)

        Returns:
            Average loss for this update
        """
        if not self.is_learning_enabled() or not self._online_trajectory:
            return 0.0

        # Assign discounted rewards
        for i, (state_embed, move_embed, _) in enumerate(self._online_trajectory):
            progress = (i + 1) / len(self._online_trajectory)
            discounted_outcome = outcome * (0.5 + 0.5 * progress)
            self._online_buffer.append((state_embed, move_embed, discounted_outcome))

        # Trim buffer
        while len(self._online_buffer) > self._online_buffer_size:
            self._online_buffer.pop(0)

        # Clear trajectory
        self._online_trajectory = []

        # Perform update if buffer is large enough
        if len(self._online_buffer) < 16:
            return 0.0

        # Sample batch from buffer
        batch_size = min(32, len(self._online_buffer))
        indices = np.random.choice(len(self._online_buffer), batch_size, replace=False)

        state_embeds = torch.stack([
            self._online_buffer[i][0] for i in indices
        ]).to(self.device)
        move_embeds = torch.stack([
            self._online_buffer[i][1] for i in indices
        ]).to(self.device)
        outcomes = torch.tensor([
            self._online_buffer[i][2] for i in indices
        ], dtype=torch.float32, device=self.device)

        # Update
        self._online_optimizer.zero_grad()
        pred_values, pred_log_vars = self.value_net(state_embeds, move_embeds)

        # NLL loss with uncertainty
        variance = torch.exp(pred_log_vars) + 1e-8
        loss = (0.5 * torch.log(variance) +
                0.5 * ((outcomes - pred_values) ** 2) / variance).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.get_parameters(),
            self._online_max_grad_norm,
        )
        self._online_optimizer.step()

        return loss.item()


# =============================================================================
# Factory function
# =============================================================================

def create_gmo_v2(
    player_number: int,
    device: str = "cpu",
    checkpoint_path: str | None = None,
) -> GMOv2AI:
    """Create a GMO v2 AI instance.

    Args:
        player_number: Player number (1-based)
        device: Device to use ("cpu" or "cuda")
        checkpoint_path: Optional path to trained checkpoint

    Returns:
        Configured GMOv2AI instance
    """
    ai_config = AIConfig(difficulty=6)
    gmo_config = GMOv2Config(device=device)

    ai = GMOv2AI(
        player_number=player_number,
        config=ai_config,
        gmo_config=gmo_config,
    )

    if checkpoint_path:
        ai.load_checkpoint(checkpoint_path)

    return ai
