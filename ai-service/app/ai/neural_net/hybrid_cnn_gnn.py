#!/usr/bin/env python3
"""Hybrid CNN-GNN Architecture for RingRift.

Combines the strengths of both architectures:
- CNN: Fast local pattern recognition, well-optimized for GPUs
- GNN: Natural connectivity modeling, territory relationships

Architecture:
    Input (C, H, W)
        ↓
    CNN Backbone (ResNet blocks)
        ↓
    Feature Extraction per cell
        ↓
    Graph Construction (from CNN features)
        ↓
    GNN Message Passing (connectivity refinement)
        ↓
    Feature Fusion (CNN + GNN)
        ↓
    Policy & Value Heads

This approach is validated by MDPI 2024 research showing hybrid GNN-CNN
for Go outperformed pure CNN by capturing territory connectivity.

Usage:
    from app.ai.neural_net.hybrid_cnn_gnn import HybridPolicyNet

    model = HybridPolicyNet(
        in_channels=56,
        board_size=8,
        action_space_size=6158,
    )
    policy, value = model(features, globals_)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from torch import Tensor

logger = logging.getLogger(__name__)

# Check for PyTorch Geometric
try:
    from torch_geometric.nn import SAGEConv, global_mean_pool
    from torch_geometric.data import Data
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    logger.warning("PyTorch Geometric not installed - GNN layers disabled")


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        y = x.mean(dim=(2, 3))  # Global average pooling
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        return x * y.view(b, c, 1, 1)


class ResidualBlock(nn.Module):
    """Residual block with SE attention."""

    def __init__(self, channels: int, use_se: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels) if use_se else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.se(x)
        return F.relu(x + residual)


class CNNBackbone(nn.Module):
    """CNN backbone for local feature extraction."""

    def __init__(
        self,
        in_channels: int = 56,
        hidden_channels: int = 128,
        num_blocks: int = 6,
    ):
        super().__init__()

        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
        )

        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_channels)
            for _ in range(num_blocks)
        ])

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_conv(x)
        for block in self.blocks:
            x = block(x)
        return x


class GNNRefinement(nn.Module):
    """GNN layers for connectivity-aware refinement.

    Takes per-cell features from CNN and refines them using
    graph message passing over the board connectivity.
    """

    def __init__(
        self,
        in_features: int = 128,
        hidden_features: int = 128,
        num_layers: int = 3,
    ):
        super().__init__()

        if not HAS_PYG:
            self.enabled = False
            self.fallback = nn.Sequential(
                nn.Linear(in_features, hidden_features),
                nn.ReLU(),
                nn.Linear(hidden_features, hidden_features),
            )
            return

        self.enabled = True
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            in_dim = in_features if i == 0 else hidden_features
            self.layers.append(SAGEConv(in_dim, hidden_features))
            self.norms.append(nn.LayerNorm(hidden_features))

    def forward(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        batch: Tensor | None = None,
    ) -> Tensor:
        """Refine features using GNN message passing.

        Args:
            node_features: (num_nodes, in_features)
            edge_index: (2, num_edges)
            batch: (num_nodes,) batch assignment

        Returns:
            Refined features (num_nodes, hidden_features)
        """
        if not self.enabled:
            return self.fallback(node_features)

        h = node_features
        for layer, norm in zip(self.layers, self.norms):
            h_new = layer(h, edge_index)
            h_new = norm(h_new)
            h_new = F.relu(h_new)
            # Residual if dimensions match
            if h.shape[-1] == h_new.shape[-1]:
                h = h + h_new
            else:
                h = h_new

        return h


class HybridPolicyNet(nn.Module):
    """Hybrid CNN-GNN Policy/Value Network.

    Combines CNN local pattern recognition with GNN connectivity modeling.

    Args:
        in_channels: Input feature channels (default: 56 with history)
        global_features: Global feature dimension (default: 20)
        hidden_channels: CNN hidden channels (default: 128)
        cnn_blocks: Number of CNN residual blocks (default: 6)
        gnn_layers: Number of GNN message passing layers (default: 3)
        board_size: Board size for spatial dimensions (default: 8)
        action_space_size: Total action space (default: 6158)
        num_players: Number of players for value head (default: 4)
        is_hex: Whether board is hexagonal (default: False)
    """

    def __init__(
        self,
        in_channels: int = 56,
        global_features: int = 20,
        hidden_channels: int = 128,
        cnn_blocks: int = 6,
        gnn_layers: int = 3,
        board_size: int = 8,
        action_space_size: int = 6158,
        num_players: int = 4,
        is_hex: bool = False,
    ):
        super().__init__()

        self.board_size = board_size
        self.hidden_channels = hidden_channels
        self.is_hex = is_hex
        self.action_space_size = action_space_size

        # CNN backbone
        self.cnn = CNNBackbone(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_blocks=cnn_blocks,
        )

        # GNN refinement
        self.gnn = GNNRefinement(
            in_features=hidden_channels,
            hidden_features=hidden_channels,
            num_layers=gnn_layers,
        )

        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
        )

        # Global feature encoder
        self.global_encoder = nn.Sequential(
            nn.Linear(global_features, hidden_channels),
            nn.ReLU(),
        )

        # Policy head
        self.policy_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.policy_fc = nn.Linear(
            32 * board_size * board_size + hidden_channels,
            action_space_size
        )

        # Value head
        self.value_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, 4, 1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(4 * board_size * board_size + hidden_channels, 256),
            nn.ReLU(),
            nn.Linear(256, num_players),
            nn.Tanh(),
        )

        # Pre-compute edge index for board
        self._edge_index = self._build_edge_index()

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        """Initialize network weights using Kaiming/Xavier."""
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _build_edge_index(self) -> Tensor:
        """Build edge index for the board connectivity."""
        edges = []
        size = self.board_size

        if self.is_hex:
            # 6-connectivity for hex
            hex_dirs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
            for y in range(size):
                for x in range(size):
                    node_idx = y * size + x
                    for dx, dy in hex_dirs:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < size and 0 <= ny < size:
                            neighbor_idx = ny * size + nx
                            edges.append([node_idx, neighbor_idx])
        else:
            # 4-connectivity for square
            for y in range(size):
                for x in range(size):
                    node_idx = y * size + x
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < size and 0 <= ny < size:
                            neighbor_idx = ny * size + nx
                            edges.append([node_idx, neighbor_idx])

        return torch.tensor(edges, dtype=torch.long).t().contiguous()

    def forward(
        self,
        features: Tensor,
        globals_: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            features: (B, C, H, W) spatial features
            globals_: (B, G) global features

        Returns:
            Tuple of:
            - policy_logits: (B, action_space_size)
            - value: (B, num_players)
        """
        B = features.shape[0]
        device = features.device

        # CNN feature extraction
        cnn_features = self.cnn(features)  # (B, hidden, H, W)

        # Reshape for GNN: (B, hidden, H, W) -> (B*H*W, hidden)
        H, W = cnn_features.shape[2], cnn_features.shape[3]
        node_features = cnn_features.permute(0, 2, 3, 1)  # (B, H, W, hidden)
        node_features = node_features.reshape(B * H * W, -1)

        # Build batched edge index
        edge_index = self._edge_index.to(device)
        num_nodes_per_graph = H * W

        # Create batch-adjusted edge indices
        batch_edge_list = []
        batch_assignment = []
        for b in range(B):
            offset = b * num_nodes_per_graph
            batch_edge_list.append(edge_index + offset)
            batch_assignment.extend([b] * num_nodes_per_graph)

        batched_edge_index = torch.cat(batch_edge_list, dim=1)
        batch_tensor = torch.tensor(batch_assignment, device=device)

        # GNN refinement
        gnn_features = self.gnn(node_features, batched_edge_index, batch_tensor)

        # Reshape back to spatial: (B*H*W, hidden) -> (B, hidden, H, W)
        gnn_spatial = gnn_features.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        # Feature fusion: concatenate CNN and GNN features per cell
        # For now, use mean pooling approach
        combined = cnn_features + gnn_spatial  # Residual fusion

        # Global features
        g = self.global_encoder(globals_)

        # Policy head
        p = self.policy_conv(combined)
        p = p.flatten(1)
        p = torch.cat([p, g], dim=1)
        policy_logits = self.policy_fc(p)

        # Value head
        v = self.value_conv(combined)
        v = v.flatten(1)
        v = torch.cat([v, g], dim=1)
        value = self.value_fc(v)

        return policy_logits, value


def create_hybrid_model(
    board_type: str = "square8",
    variant: str = "standard",
    **kwargs,
) -> HybridPolicyNet:
    """Factory function for hybrid CNN-GNN models.

    Args:
        board_type: 'square8', 'square19', 'hex8', 'hexagonal'
        variant: 'standard', 'large', 'lite'
        **kwargs: Override parameters

    Returns:
        Configured HybridPolicyNet
    """
    configs = {
        "square8": {
            "board_size": 8,
            "action_space_size": 6158,
            "is_hex": False,
        },
        "square19": {
            "board_size": 19,
            "action_space_size": 67000,
            "is_hex": False,
        },
        "hex8": {
            "board_size": 9,
            "action_space_size": 3000,
            "is_hex": True,
        },
        "hexagonal": {
            "board_size": 25,
            "action_space_size": 25000,
            "is_hex": True,
        },
    }

    variants = {
        "lite": {"hidden_channels": 64, "cnn_blocks": 4, "gnn_layers": 2},
        "standard": {"hidden_channels": 128, "cnn_blocks": 6, "gnn_layers": 3},
        "large": {"hidden_channels": 192, "cnn_blocks": 8, "gnn_layers": 4},
    }

    config = configs.get(board_type, configs["square8"])
    config.update(variants.get(variant, variants["standard"]))
    config.update(kwargs)

    return HybridPolicyNet(**config)


__all__ = [
    "HybridPolicyNet",
    "CNNBackbone",
    "GNNRefinement",
    "create_hybrid_model",
]
