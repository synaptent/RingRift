"""CAGE: Constraint-Aware Graph Energy-Based Move Optimization.

A novel game-playing architecture that combines:
1. Graph Neural Networks for board representation
2. Energy-based move optimization
3. Primal-dual legality constraints

Key innovations:
- Represents board as a graph (cells as nodes, adjacencies as edges)
- Learns legality constraints as part of the energy function
- Uses primal-dual optimization to stay on legal move manifold
- Enables interpretable energy decomposition

Usage:
    from app.ai.cage_network import CAGENetwork, CAGEConfig

    config = CAGEConfig()
    network = CAGENetwork(config)

    # Forward pass
    energies = network(state_graph, action_features)

    # Constrained optimization
    best_move = network.optimize_with_constraints(state, legal_moves)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models import (
    BoardType,
    GameState,
)

logger = logging.getLogger(__name__)


@dataclass
class CAGEConfig:
    """Configuration for CAGE network."""

    # Graph neural network
    node_feature_dim: int = 32  # Features per cell node
    edge_feature_dim: int = 8   # Features per edge
    gnn_hidden_dim: int = 128
    gnn_num_layers: int = 4
    gnn_num_heads: int = 4  # For attention-based aggregation

    # Action representation
    action_embed_dim: int = 64
    action_hidden_dim: int = 128

    # Energy network
    energy_hidden_dim: int = 128
    num_energy_layers: int = 3

    # Constraint network
    constraint_hidden_dim: int = 64
    num_constraint_types: int = 8  # Different legality violation types

    # Optimization
    optim_steps: int = 50
    optim_lr: float = 0.1
    dual_lr: float = 0.5  # Learning rate for dual variables
    constraint_penalty: float = 10.0

    # Board
    board_size: int = 8
    board_type: BoardType = BoardType.SQUARE8


class GraphAttentionLayer(nn.Module):
    """Graph attention layer with edge features."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        edge_features: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.out_per_head = out_features // num_heads

        # Linear transformations
        self.W_q = nn.Linear(in_features, out_features, bias=False)
        self.W_k = nn.Linear(in_features, out_features, bias=False)
        self.W_v = nn.Linear(in_features, out_features, bias=False)
        self.W_e = nn.Linear(edge_features, num_heads, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_features)

        # Skip connection projection
        self.skip_proj = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (N, in_features) node features
            edge_index: (2, E) edge indices
            edge_attr: (E, edge_features) edge features

        Returns:
            (N, out_features) updated node features
        """
        N = x.size(0)
        H = self.num_heads
        D = self.out_per_head

        # Compute Q, K, V
        Q = self.W_q(x).view(N, H, D)
        K = self.W_k(x).view(N, H, D)
        V = self.W_v(x).view(N, H, D)

        # Edge attention bias
        edge_bias = self.W_e(edge_attr)  # (E, H)

        # Compute attention scores for each edge
        src, dst = edge_index
        q_src = Q[src]  # (E, H, D)
        k_dst = K[dst]  # (E, H, D)

        # Scaled dot product attention
        attn_scores = (q_src * k_dst).sum(dim=-1) / math.sqrt(D)  # (E, H)
        attn_scores = attn_scores + edge_bias

        # Softmax over incoming edges (scatter_softmax)
        attn_weights = self._scatter_softmax(attn_scores, dst, N)

        # Aggregate
        v_dst = V[dst]  # (E, H, D)
        weighted_v = attn_weights.unsqueeze(-1) * v_dst  # (E, H, D)

        # Scatter sum to destination nodes
        out = torch.zeros(N, H, D, device=x.device)
        out.scatter_add_(0, dst.unsqueeze(-1).unsqueeze(-1).expand(-1, H, D), weighted_v)

        out = out.view(N, -1)  # (N, out_features)
        out = self.dropout(out)

        # Skip connection + layer norm
        out = self.layer_norm(out + self.skip_proj(x))

        return out

    def _scatter_softmax(
        self,
        scores: torch.Tensor,
        index: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """Softmax over scattered groups."""
        # For simplicity, use dense attention matrix (works for small graphs)
        # In production, use torch_scatter for efficiency
        scores_exp = scores.exp()
        denom = torch.zeros(num_nodes, scores.size(1), device=scores.device)
        denom.scatter_add_(0, index.unsqueeze(-1).expand(-1, scores.size(1)), scores_exp)
        return scores_exp / (denom[index] + 1e-10)


class GraphEncoder(nn.Module):
    """GNN encoder for board state."""

    def __init__(self, config: CAGEConfig):
        super().__init__()
        self.config = config

        # Initial node embedding
        self.node_embed = nn.Linear(config.node_feature_dim, config.gnn_hidden_dim)

        # GNN layers
        self.gnn_layers = nn.ModuleList([
            GraphAttentionLayer(
                config.gnn_hidden_dim,
                config.gnn_hidden_dim,
                config.edge_feature_dim,
                config.gnn_num_heads,
            )
            for _ in range(config.gnn_num_layers)
        ])

        # Global pooling + projection
        self.global_pool = nn.Sequential(
            nn.Linear(config.gnn_hidden_dim, config.gnn_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.gnn_hidden_dim, config.gnn_hidden_dim),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            node_features: (N, node_feature_dim) node features
            edge_index: (2, E) edge indices
            edge_attr: (E, edge_feature_dim) edge features
            batch: (N,) batch indices for batched graphs

        Returns:
            (node_embed, graph_embed) tuple
            - node_embed: (N, gnn_hidden_dim) per-node embeddings
            - graph_embed: (B, gnn_hidden_dim) per-graph embeddings
        """
        x = self.node_embed(node_features)

        for layer in self.gnn_layers:
            x = F.relu(layer(x, edge_index, edge_attr))

        node_embed = x

        # Global mean pooling
        if batch is None:
            graph_embed = x.mean(dim=0, keepdim=True)
        else:
            # Scatter mean over batch
            num_graphs = batch.max().item() + 1
            graph_embed = torch.zeros(num_graphs, x.size(1), device=x.device)
            counts = torch.zeros(num_graphs, device=x.device)
            graph_embed.scatter_add_(0, batch.unsqueeze(-1).expand(-1, x.size(1)), x)
            counts.scatter_add_(0, batch, torch.ones_like(batch, dtype=torch.float))
            graph_embed = graph_embed / (counts.unsqueeze(-1) + 1e-10)

        graph_embed = self.global_pool(graph_embed)

        return node_embed, graph_embed


class ConstraintNetwork(nn.Module):
    """Network that predicts legality constraint violations.

    For each action embedding, predicts whether it would violate
    various legality constraints. Used in primal-dual optimization.
    """

    def __init__(self, config: CAGEConfig):
        super().__init__()
        self.config = config

        input_dim = config.gnn_hidden_dim + config.action_embed_dim
        hidden = config.constraint_hidden_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, config.num_constraint_types),
            nn.Sigmoid(),  # Output in [0, 1], 0 = legal, 1 = illegal
        )

    def forward(
        self,
        graph_embed: torch.Tensor,
        action_embed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            graph_embed: (B, gnn_hidden_dim) graph embedding
            action_embed: (B, action_embed_dim) action embedding

        Returns:
            (B, num_constraint_types) constraint violation scores
        """
        combined = torch.cat([graph_embed, action_embed], dim=-1)
        return self.network(combined)


class CAGEEnergyHead(nn.Module):
    """Energy function with decomposed components.

    Computes:
    E(s, a) = E_base(s, a) + sum_i lambda_i * C_i(s, a)

    Where C_i are constraint violation terms and lambda_i are dual variables.
    """

    def __init__(self, config: CAGEConfig):
        super().__init__()
        self.config = config

        input_dim = config.gnn_hidden_dim + config.action_embed_dim
        hidden = config.energy_hidden_dim

        # Base energy network
        layers = []
        layers.append(nn.Linear(input_dim, hidden))
        layers.append(nn.ReLU())
        for _ in range(config.num_energy_layers - 2):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden, 1))
        self.base_energy = nn.Sequential(*layers)

        # Constraint network
        self.constraint_net = ConstraintNetwork(config)

    def forward(
        self,
        graph_embed: torch.Tensor,
        action_embed: torch.Tensor,
        dual_vars: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            graph_embed: (B, gnn_hidden_dim) graph embedding
            action_embed: (B, action_embed_dim) action embedding
            dual_vars: (num_constraint_types,) dual variables for constraints

        Returns:
            (total_energy, constraint_violations) tuple
            - total_energy: (B,) energy values
            - constraint_violations: (B, num_constraint_types) violation scores
        """
        combined = torch.cat([graph_embed, action_embed], dim=-1)

        # Base energy
        base_e = self.base_energy(combined).squeeze(-1)  # (B,)

        # Constraint violations
        violations = self.constraint_net(graph_embed, action_embed)  # (B, C)

        # Augmented energy with constraints
        if dual_vars is None:
            dual_vars = torch.ones(
                self.config.num_constraint_types,
                device=graph_embed.device,
            ) * self.config.constraint_penalty

        constraint_penalty = (violations * dual_vars).sum(dim=-1)  # (B,)

        total_energy = base_e + constraint_penalty

        return total_energy, violations


class CAGENetwork(nn.Module):
    """Complete CAGE network.

    Combines:
    - GraphEncoder: GNN for board state
    - ActionEncoder: MLP for action features
    - CAGEEnergyHead: Energy with constraint penalties
    """

    def __init__(self, config: CAGEConfig | None = None):
        super().__init__()
        self.config = config or CAGEConfig()

        self.graph_encoder = GraphEncoder(self.config)
        self.action_encoder = nn.Sequential(
            nn.Linear(14, self.config.action_hidden_dim),  # 14 = action feature dim
            nn.ReLU(),
            nn.Linear(self.config.action_hidden_dim, self.config.action_embed_dim),
        )
        self.energy_head = CAGEEnergyHead(self.config)

        # Dual variables (learnable during training, fixed during inference)
        self.register_buffer(
            'dual_vars',
            torch.ones(self.config.num_constraint_types) * self.config.constraint_penalty
        )

    def encode_graph(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode board as graph."""
        return self.graph_encoder(node_features, edge_index, edge_attr, batch)

    def encode_action(self, action_features: torch.Tensor) -> torch.Tensor:
        """Encode action to embedding."""
        return self.action_encoder(action_features)

    def compute_energy(
        self,
        graph_embed: torch.Tensor,
        action_embed: torch.Tensor,
    ) -> torch.Tensor:
        """Compute energy (base + constraint penalty)."""
        energy, _ = self.energy_head(graph_embed, action_embed, self.dual_vars)
        return energy

    def compute_energy_with_violations(
        self,
        graph_embed: torch.Tensor,
        action_embed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute energy and constraint violations."""
        return self.energy_head(graph_embed, action_embed, self.dual_vars)

    def primal_dual_optimize(
        self,
        graph_embed: torch.Tensor,
        legal_action_embeds: torch.Tensor,
        num_steps: int = 50,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Optimize action embedding using primal-dual method.

        Instead of pure gradient descent (which escapes manifold),
        we alternate:
        1. Primal step: minimize energy w.r.t. action embedding
        2. Dual step: update constraint penalties based on violations

        Args:
            graph_embed: (state_dim,) graph embedding
            legal_action_embeds: (M, action_dim) legal move embeddings
            num_steps: Number of optimization steps

        Returns:
            (best_action_idx, final_energy) tuple
        """
        M = legal_action_embeds.size(0)
        device = legal_action_embeds.device

        # Initialize with convex combination weights (simplex)
        weights = torch.zeros(M, device=device, requires_grad=True)

        # Local dual variables
        local_dual = self.dual_vars.clone()

        optimizer = torch.optim.Adam([weights], lr=self.config.optim_lr)

        best_energy = float('inf')
        best_idx = 0

        for _step in range(num_steps):
            optimizer.zero_grad()

            # Current action as convex combination
            probs = F.softmax(weights, dim=0)
            action_embed = (probs.unsqueeze(1) * legal_action_embeds).sum(dim=0)

            # Compute energy and violations
            energy, violations = self.energy_head(
                graph_embed.unsqueeze(0),
                action_embed.unsqueeze(0),
                local_dual,
            )

            # Primal step: minimize energy
            energy.backward()
            optimizer.step()

            # Dual step: increase penalty for violated constraints
            with torch.no_grad():
                local_dual = local_dual + self.config.dual_lr * violations.squeeze(0)
                local_dual = torch.clamp(local_dual, min=0.1, max=100.0)

            # Track best
            final_probs = F.softmax(weights, dim=0)
            best_current = final_probs.argmax().item()
            current_energy = self.compute_energy(
                graph_embed.unsqueeze(0),
                legal_action_embeds[best_current].unsqueeze(0),
            ).item()

            if current_energy < best_energy:
                best_energy = current_energy
                best_idx = best_current

        return best_idx, best_energy


def board_to_graph(
    game_state: GameState,
    player_number: int,
    board_size: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert board state to graph representation.

    Args:
        game_state: Game state
        player_number: Perspective player
        board_size: Board size

    Returns:
        (node_features, edge_index, edge_attr) tuple
    """
    num_nodes = board_size * board_size
    node_features = torch.zeros(num_nodes, 32)

    board = game_state.board

    def set_feature(x: int, y: int, idx: int, value: float = 1.0) -> None:
        if 0 <= x < board_size and 0 <= y < board_size:
            node_idx = y * board_size + x
            node_features[node_idx, idx] = value

    for key, stack in board.stacks.items():
        try:
            parts = key.split(",")
            x, y = int(parts[0]), int(parts[1])
        except (ValueError, IndexError):
            continue

        height = float(getattr(stack, "stack_height", len(stack.rings or [])))
        cap = float(getattr(stack, "cap_height", 0))
        controller = int(getattr(stack, "controlling_player", 0))

        set_feature(x, y, 0, 1.0)
        set_feature(x, y, 1, height / 5.0)
        set_feature(x, y, 2, cap / 5.0)
        if controller == player_number:
            set_feature(x, y, 3, 1.0)
        elif controller != 0:
            set_feature(x, y, 4, 1.0)

    for key, marker in board.markers.items():
        try:
            parts = key.split(",")
            x, y = int(parts[0]), int(parts[1])
        except (ValueError, IndexError):
            continue

        set_feature(x, y, 5, 1.0)
        if marker.player == player_number:
            set_feature(x, y, 6, 1.0)
        else:
            set_feature(x, y, 7, 1.0)

    for key, owner in board.collapsed_spaces.items():
        try:
            parts = key.split(",")
            x, y = int(parts[0]), int(parts[1])
        except (ValueError, IndexError):
            continue

        set_feature(x, y, 8, 1.0)
        if owner == player_number:
            set_feature(x, y, 9, 1.0)
        elif owner != 0:
            set_feature(x, y, 10, 1.0)

    # Edge index: 4-connectivity on square grids.
    edges = []
    edge_attrs = []
    dir_map = {
        (-1, 0): 1,
        (1, 0): 2,
        (0, -1): 3,
        (0, 1): 4,
    }

    for y in range(board_size):
        for x in range(board_size):
            node_idx = y * board_size + x
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= nx < board_size and 0 <= ny < board_size:
                    neighbor_idx = ny * board_size + nx
                    edges.append([node_idx, neighbor_idx])
                    edge_attr = torch.zeros(8)
                    edge_attr[0] = 1.0
                    edge_attr[dir_map[(dy, dx)]] = 1.0
                    edge_attrs.append(edge_attr)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.stack(edge_attrs) if edge_attrs else torch.zeros(0, 8)

    return node_features, edge_index, edge_attr


# Export for use
__all__ = [
    "CAGEConfig",
    "CAGEEnergyHead",
    "CAGENetwork",
    "ConstraintNetwork",
    "GraphEncoder",
    "board_to_graph",
]
