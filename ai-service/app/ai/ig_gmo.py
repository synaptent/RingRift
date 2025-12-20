"""Information-Gain GMO (IG-GMO) - Novel extension with MI-based exploration.

This module implements a research-grade extension of GMO that uses:
1. True mutual information for exploration (not just sqrt(variance))
2. Graph Neural Network state encoder for relational structure
3. Soft legality constraints during optimization

Key innovations over standard GMO:
- MI-based exploration: I(y; θ | s, a) = H(E_θ[p(y|s,a,θ)]) - E_θ[H(p(y|s,a,θ))]
- GNN state encoder captures board topology and spatial relationships
- Primal-dual legality optimization reduces projection loss

References:
- SPENs: Structured Prediction Energy Networks (Belanger & McCallum, 2016)
- MC Dropout as Bayesian approximation (Gal & Ghahramani, 2016)
- Action embeddings with NN projection (Dulac-Arnold et al., 2015)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models import AIConfig, GameState, Move
from .base import BaseAI
from .gmo_ai import GMOValueNetWithUncertainty, MoveEncoder, NoveltyTracker

logger = logging.getLogger(__name__)


@dataclass
class IGGMOConfig:
    """Configuration for Information-Gain GMO.

    Key differences from standard GMO:
    - Uses mutual information instead of variance for exploration
    - GNN-based state encoding
    - Soft legality constraints during optimization
    """
    # Embedding dimensions
    state_dim: int = 128
    move_dim: int = 128
    hidden_dim: int = 256

    # GNN parameters
    gnn_layers: int = 3
    gnn_heads: int = 4  # For graph attention

    # Optimization parameters
    top_k: int = 3
    optim_steps: int = 5
    lr: float = 0.1

    # Information-theoretic parameters (MI-based)
    beta: float = 0.2  # MI exploration coefficient
    gamma: float = 0.05  # Novelty coefficient

    # MC Dropout for Bayesian uncertainty
    dropout_rate: float = 0.1
    mc_samples: int = 10

    # Legality constraint parameters
    use_soft_legality: bool = True
    legality_weight: float = 1.0  # λ for legality penalty
    legality_temp: float = 0.1  # Temperature for soft legality

    # Novelty tracking
    novelty_memory_size: int = 1000

    # Device
    device: str = "cpu"


class GraphAttentionLayer(nn.Module):
    """Graph Attention Network layer for encoding board structure.

    Implements multi-head attention over graph neighbors.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        concat: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat

        # Per-head output dimension
        self.head_dim = out_features // num_heads if concat else out_features

        # Linear transformations for Q, K, V
        self.W_q = nn.Linear(in_features, self.head_dim * num_heads, bias=False)
        self.W_k = nn.Linear(in_features, self.head_dim * num_heads, bias=False)
        self.W_v = nn.Linear(in_features, self.head_dim * num_heads, bias=False)

        # Attention coefficients
        self.a = nn.Parameter(torch.zeros(num_heads, 2 * self.head_dim))
        nn.init.xavier_uniform_(self.a.view(num_heads, -1))

        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

        if concat:
            self.out_proj = nn.Linear(self.head_dim * num_heads, out_features)
        else:
            self.out_proj = nn.Linear(self.head_dim, out_features)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Node features (batch, num_nodes, in_features)
            adj: Adjacency matrix (batch, num_nodes, num_nodes)

        Returns:
            Updated node features (batch, num_nodes, out_features)
        """
        batch_size, num_nodes, _ = x.shape

        # Linear transformations
        Q = self.W_q(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        K = self.W_k(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        V = self.W_v(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)

        # Compute attention scores
        # (batch, nodes, heads, dim) -> (batch, heads, nodes, dim)
        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Mask non-adjacent nodes
        adj_expanded = adj.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        scores = scores.masked_fill(adj_expanded == 0, float('-inf'))

        # Softmax over neighbors
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, V)  # (batch, heads, nodes, dim)

        # Reshape and project
        if self.concat:
            out = out.permute(0, 2, 1, 3).contiguous()
            out = out.view(batch_size, num_nodes, -1)
        else:
            out = out.mean(dim=1)  # Average over heads

        out = self.out_proj(out)
        return out


class GNNStateEncoder(nn.Module):
    """Graph Neural Network encoder for game state.

    Treats the board as a graph where:
    - Nodes are board positions
    - Edges connect adjacent positions
    - Node features encode stack/marker/territory info
    """

    def __init__(
        self,
        output_dim: int = 128,
        hidden_dim: int = 64,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        board_size: int = 8,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.board_size = board_size
        self.num_positions = board_size * board_size

        # Node feature dimension (stack height, owner, marker, territory, etc.)
        self.node_feature_dim = 12

        # Initial embedding
        self.input_proj = nn.Linear(self.node_feature_dim, hidden_dim)

        # GNN layers
        self.gnn_layers = nn.ModuleList([
            GraphAttentionLayer(
                hidden_dim if i == 0 else hidden_dim,
                hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for i in range(num_layers)
        ])

        # Layer norms
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])

        # Global pooling and output
        self.global_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        # Pre-compute adjacency template for 8x8 board
        self._register_adjacency()

    def _register_adjacency(self):
        """Pre-compute adjacency matrix for board positions."""
        adj = torch.zeros(self.num_positions, self.num_positions)

        for i in range(self.num_positions):
            row, col = i // self.board_size, i % self.board_size

            # 8-connected neighbors (including diagonals)
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                        j = nr * self.board_size + nc
                        adj[i, j] = 1.0

        # Add self-loops
        adj = adj + torch.eye(self.num_positions)

        self.register_buffer('adj_template', adj)

    def _extract_node_features(self, state: GameState) -> torch.Tensor:
        """Extract per-node features from game state."""
        device = next(self.parameters()).device
        features = torch.zeros(
            self.num_positions,
            self.node_feature_dim,
            dtype=torch.float32,
            device=device,
        )

        board = state.board

        # Build territory position set
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

                # Top ring owner (one-hot for 4 players)
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

    def encode_state(self, state: GameState) -> torch.Tensor:
        """Encode game state to embedding vector."""
        # Extract node features
        node_features = self._extract_node_features(state)
        node_features = node_features.unsqueeze(0)  # Add batch dim

        # Get adjacency
        adj = self.adj_template.unsqueeze(0)

        # Initial projection
        x = self.input_proj(node_features)

        # GNN layers with residual connections
        for gnn_layer, layer_norm in zip(self.gnn_layers, self.layer_norms, strict=False):
            x_new = gnn_layer(x, adj)
            x = layer_norm(x + x_new)  # Residual + norm

        # Global pooling (mean over nodes)
        x = x.mean(dim=1)  # (batch, hidden_dim)

        # Output projection
        x = self.global_pool(x)

        return x.squeeze(0)  # Remove batch dim

    def forward(self, state: GameState) -> torch.Tensor:
        """Forward pass - alias for encode_state."""
        return self.encode_state(state)


def compute_mutual_information(
    state_embed: torch.Tensor,
    move_embed: torch.Tensor,
    value_net: nn.Module,
    mc_samples: int = 10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute mutual information for exploration.

    MI(y; θ | s, a) = H(E_θ[p(y|s,a,θ)]) - E_θ[H(p(y|s,a,θ))]

    For regression with Gaussian outputs:
    - First term: entropy of the predictive mean
    - Second term: expected entropy of individual predictions

    Returns:
        mean_value: Mean predicted value
        mutual_info: Mutual information estimate
        variance: Predictive variance
    """
    value_net.train()  # Enable dropout

    values = []
    log_vars = []

    for _ in range(mc_samples):
        value, log_var = value_net(state_embed, move_embed)
        values.append(value)
        log_vars.append(log_var)

    value_net.eval()

    values_tensor = torch.stack(values)
    log_vars_tensor = torch.stack(log_vars)

    # Mean prediction
    mean_value = values_tensor.mean()

    # Predictive variance (epistemic + aleatoric)
    epistemic_var = values_tensor.var() + 1e-8
    aleatoric_var = torch.exp(log_vars_tensor).mean()
    total_var = epistemic_var + aleatoric_var

    # Mutual information approximation for Gaussian
    # H(E[p]) - E[H(p)] ≈ 0.5 * log(1 + epistemic_var / aleatoric_var)
    # This measures how much the prediction varies due to model uncertainty
    mi = 0.5 * torch.log(1 + epistemic_var / (aleatoric_var + 1e-8))

    return mean_value, mi, total_var


class SoftLegalityPredictor(nn.Module):
    """Predicts legality probability for soft constraint optimization.

    Instead of hard projection to nearest legal move, we learn a
    differentiable legality function L(s, a) ∈ [0, 1].
    """

    def __init__(
        self,
        state_dim: int = 128,
        move_dim: int = 128,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + move_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        state_embed: torch.Tensor,
        move_embed: torch.Tensor,
    ) -> torch.Tensor:
        """Predict legality probability."""
        if state_embed.dim() == 1:
            state_embed = state_embed.unsqueeze(0)
        if move_embed.dim() == 1:
            move_embed = move_embed.unsqueeze(0)

        joint = torch.cat([state_embed, move_embed], dim=-1)
        return self.net(joint).squeeze(-1)


class IGGMO(BaseAI):
    """Information-Gain Gradient Move Optimization.

    Novel research extension with:
    1. MI-based exploration instead of UCB-style sqrt(var)
    2. GNN state encoder for graph structure
    3. Optional soft legality constraints
    """

    def __init__(
        self,
        player_number: int,
        config: AIConfig,
        ig_config: IGGMOConfig | None = None,
    ):
        super().__init__(player_number, config)

        self.ig_config = ig_config or IGGMOConfig()
        self.device = torch.device(self.ig_config.device)

        # GNN State encoder
        self.state_encoder = GNNStateEncoder(
            output_dim=self.ig_config.state_dim,
            hidden_dim=self.ig_config.hidden_dim // 2,
            num_layers=self.ig_config.gnn_layers,
            num_heads=self.ig_config.gnn_heads,
            dropout=self.ig_config.dropout_rate,
        ).to(self.device)

        # Move encoder (reuse from GMO)
        self.move_encoder = MoveEncoder(
            embed_dim=self.ig_config.move_dim,
        ).to(self.device)

        # Value network with uncertainty
        self.value_net = GMOValueNetWithUncertainty(
            state_dim=self.ig_config.state_dim,
            move_dim=self.ig_config.move_dim,
            hidden_dim=self.ig_config.hidden_dim,
            dropout_rate=self.ig_config.dropout_rate,
        ).to(self.device)

        # Optional soft legality predictor
        if self.ig_config.use_soft_legality:
            self.legality_net = SoftLegalityPredictor(
                state_dim=self.ig_config.state_dim,
                move_dim=self.ig_config.move_dim,
            ).to(self.device)
        else:
            self.legality_net = None

        # Novelty tracker
        self.novelty_tracker = NoveltyTracker(
            memory_size=self.ig_config.novelty_memory_size,
            embed_dim=self.ig_config.move_dim,
        )

    def _optimize_move_embedding(
        self,
        state_embed: torch.Tensor,
        initial_embed: torch.Tensor,
        legal_move_embeds: list[torch.Tensor],
    ) -> torch.Tensor:
        """Optimize move embedding with MI-based exploration.

        Objective: max E[V] + β * MI + γ * Novelty - λ * (1 - Legality)
        """
        move_embed = initial_embed.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([move_embed], lr=self.ig_config.lr)

        for _step in range(self.ig_config.optim_steps):
            optimizer.zero_grad()

            # Compute value and mutual information
            mean_value, mi, _ = compute_mutual_information(
                state_embed, move_embed, self.value_net,
                self.ig_config.mc_samples,
            )

            # Novelty bonus
            novelty = self.novelty_tracker.compute_novelty(move_embed)

            # MI-based objective
            objective = (
                mean_value +
                self.ig_config.beta * mi +
                self.ig_config.gamma * novelty
            )

            # Soft legality constraint (barrier method)
            if self.legality_net is not None:
                legality = self.legality_net(state_embed, move_embed)
                # Log barrier: -λ * log(L + ε)
                barrier = -self.ig_config.legality_weight * torch.log(
                    legality + self.ig_config.legality_temp
                )
                objective = objective - barrier

            # Maximize objective
            loss = -objective
            loss.backward()
            optimizer.step()

        return move_embed.detach()

    def select_move(self, game_state: GameState) -> Move | None:
        """Select move using IG-GMO algorithm."""
        from ..game_engine import GameEngine

        current_player = game_state.current_player
        legal_moves = GameEngine.get_valid_moves(game_state, current_player)

        if not legal_moves:
            return None

        if len(legal_moves) == 1:
            return legal_moves[0]

        # Encode state with GNN
        with torch.no_grad():
            state_embed = self.state_encoder.encode_state(game_state)

        # Encode all legal moves
        move_embeds = []
        for move in legal_moves:
            with torch.no_grad():
                embed = self.move_encoder.encode_move(move)
            move_embeds.append(embed)

        # Initial ranking with MI
        initial_scores = []
        for me in move_embeds:
            mean_val, mi, _ = compute_mutual_information(
                state_embed, me, self.value_net,
                self.ig_config.mc_samples,
            )
            novelty = self.novelty_tracker.compute_novelty(me)
            score = mean_val + self.ig_config.beta * mi + self.ig_config.gamma * novelty
            initial_scores.append(score.item())

        # Select top-k candidates
        top_k_indices = np.argsort(initial_scores)[-self.ig_config.top_k:][::-1]

        best_move = None
        best_score = float('-inf')
        best_embed = None

        for idx in top_k_indices:
            # Optimize move embedding
            optimized_embed = self._optimize_move_embedding(
                state_embed,
                move_embeds[idx],
                move_embeds,
            )

            # Project to nearest legal move
            similarities = torch.stack([
                F.cosine_similarity(optimized_embed.unsqueeze(0), me.unsqueeze(0))
                for me in move_embeds
            ])
            nearest_idx = similarities.argmax().item()

            # Evaluate projected move
            mean_val, mi, _ = compute_mutual_information(
                state_embed, move_embeds[nearest_idx], self.value_net,
                self.ig_config.mc_samples,
            )

            final_score = mean_val.item() + self.ig_config.beta * mi.item()

            if final_score > best_score:
                best_score = final_score
                best_move = legal_moves[nearest_idx]
                best_embed = move_embeds[nearest_idx]

        # Update novelty memory
        if best_embed is not None:
            self.novelty_tracker.add(best_embed)

        return best_move

    def evaluate_position(self, game_state: GameState) -> float:
        """Evaluate position using the value network.

        Returns average value across all legal moves weighted by their scores.
        """
        from ..game_engine import GameEngine

        current_player = game_state.current_player
        legal_moves = GameEngine.get_valid_moves(game_state, current_player)

        if not legal_moves:
            return 0.0

        # Encode state
        with torch.no_grad():
            state_embed = self.state_encoder.encode_state(game_state)

        # Get average value across legal moves
        total_value = 0.0
        for move in legal_moves:
            move_embed = self.move_encoder.encode_move(move)
            value, _ = self.value_net(state_embed, move_embed)
            total_value += value.item()

        return total_value / len(legal_moves)

    def save_checkpoint(self, path: Path) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'state_encoder': self.state_encoder.state_dict(),
            'move_encoder': self.move_encoder.state_dict(),
            'value_net': self.value_net.state_dict(),
            'ig_config': self.ig_config,
        }
        if self.legality_net is not None:
            checkpoint['legality_net'] = self.legality_net.state_dict()

        torch.save(checkpoint, path)
        logger.info(f"Saved IG-GMO checkpoint to {path}")

    def load_checkpoint(self, path: Path) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.state_encoder.load_state_dict(checkpoint['state_encoder'])
        self.move_encoder.load_state_dict(checkpoint['move_encoder'])
        self.value_net.load_state_dict(checkpoint['value_net'])

        if 'legality_net' in checkpoint and self.legality_net is not None:
            self.legality_net.load_state_dict(checkpoint['legality_net'])

        logger.info(f"Loaded IG-GMO checkpoint from {path}")


def create_ig_gmo(
    player_number: int,
    device: str = "cpu",
    checkpoint_path: Path | None = None,
) -> IGGMO:
    """Factory function for IG-GMO."""
    ai_config = AIConfig(difficulty=6)
    ig_config = IGGMOConfig(device=device)

    ai = IGGMO(
        player_number=player_number,
        config=ai_config,
        ig_config=ig_config,
    )

    if checkpoint_path and checkpoint_path.exists():
        ai.load_checkpoint(checkpoint_path)

    return ai
