"""V5 Heavy Architecture: Maximum Strength Neural Network for RingRift.

This architecture combines the best features from all previous versions:
- SE blocks for channel attention (from v2/v3)
- Multi-head self-attention for long-range dependencies (from v4)
- Configurable heuristic features (21 fast or 49 full) with FiLM conditioning
- Optional GNN refinement for connectivity modeling (from Hybrid)
- Optional geometry encoding (distance-from-center, directional occupancy)
- Spatial policy heads for geometric consistency (from v3)
- Rank distribution value head for multiplayer outcomes (from v3)

Architecture:
    Input (C, H, W) + Global (G) + Heuristics (21 or 49)
        ↓
    [Optional] Geometry Encoding (add spatial channels)
        ↓
    Initial Conv (5x5, NAS-optimal)
        ↓
    Mixed Backbone:
    ├── SE Residual Blocks (first half)
    └── Attention Residual Blocks (second half)
        ↓
    [Optional] GNN Refinement (2-layer GraphSAGE)
        ↓
    Heuristic FiLM Conditioning (modulate features)
        ↓
    Spatial Policy Heads (v3-style)
        ↓
    Rank Distribution Value Head

Usage:
    from app.ai.neural_net.v5_heavy import RingRiftCNN_v5_Heavy

    # Standard mode (21 fast features)
    model = RingRiftCNN_v5_Heavy(board_size=8, num_players=2)

    # Maximum strength (49 full features + geometry encoding)
    model = RingRiftCNN_v5_Heavy(
        board_size=8,
        num_players=2,
        num_heuristics=49,
        use_geometry_encoding=True,
        use_gnn=True,
    )
    value, policy, rank_dist = model(features, globals_, heuristics)

Heuristic Features:
    - Fast mode (21): Aggregated component scores from HeuristicAI
      Extracted via: app.training.fast_heuristic_features.extract_heuristic_features()
    - Full mode (49): All 49 weight decomposition features
      Extracted via: app.training.fast_heuristic_features.extract_full_heuristic_features()

Architecture Version:
    v5.1.0 - Added configurable heuristics (21/49) and geometry encoding.

Estimated Parameters:
    - Standard (21 heuristics): ~6.2M params (~25MB)
    - Full (49 heuristics): ~6.5M params (~26MB)
    - Full + GNN: ~6.6M params (~27MB)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import AttentionResidualBlock, SEResidualBlock
from .constants import (
    MAX_DIST_SQUARE8,
    MAX_DIST_SQUARE19,
    NUM_LINE_DIRS,
    NUM_SQUARE_DIRS,
    POLICY_SIZE_8x8,
    POLICY_SIZE_19x19,
    SQUARE8_EXTRA_SPECIAL_BASE,
    SQUARE8_EXTRA_SPECIAL_SPAN,
    SQUARE8_LINE_CHOICE_BASE,
    SQUARE8_LINE_FORM_BASE,
    SQUARE8_MOVEMENT_BASE,
    SQUARE8_SKIP_PLACEMENT_IDX,
    SQUARE8_SKIP_RECOVERY_IDX,
    SQUARE8_SWAP_SIDES_IDX,
    SQUARE8_TERRITORY_CHOICE_BASE,
    SQUARE8_TERRITORY_CLAIM_BASE,
    SQUARE19_EXTRA_SPECIAL_BASE,
    SQUARE19_EXTRA_SPECIAL_SPAN,
    SQUARE19_LINE_CHOICE_BASE,
    SQUARE19_LINE_FORM_BASE,
    SQUARE19_MOVEMENT_BASE,
    SQUARE19_SKIP_PLACEMENT_IDX,
    SQUARE19_SKIP_RECOVERY_IDX,
    SQUARE19_SWAP_SIDES_IDX,
    SQUARE19_TERRITORY_CHOICE_BASE,
    SQUARE19_TERRITORY_CLAIM_BASE,
    TERRITORY_MAX_PLAYERS,
    TERRITORY_SIZE_BUCKETS,
)

if TYPE_CHECKING:
    from torch import Tensor

logger = logging.getLogger(__name__)

# Check for PyTorch Geometric (optional GNN support)
try:
    from torch_geometric.nn import SAGEConv
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    SAGEConv = None

# Heuristic feature counts
# Fast mode: 21 features from efficient component extraction
# Full mode: 49 features from linear weight decomposition
NUM_HEURISTIC_FEATURES_FAST = 21
NUM_HEURISTIC_FEATURES_FULL = 49
NUM_HEURISTIC_FEATURES = NUM_HEURISTIC_FEATURES_FAST  # Default for backwards compat

# Geometry encoding constants
NUM_GEOMETRY_CHANNELS = 10  # distance_from_center(1) + center_zone(1) + directional_occupancy(8)


class HeuristicEncoder(nn.Module):
    """Encoder for heuristic features with FiLM-style conditioning.

    Supports both fast (21) and full (49) heuristic feature modes:
    - Fast mode (21 features): Standard 2-layer encoder
    - Full mode (49 features): Deeper 3-layer encoder for richer representation

    Produces:
    1. A global embedding vector for concatenation with backbone features
    2. Scale and shift parameters for FiLM conditioning of spatial features

    FiLM (Feature-wise Linear Modulation):
        y = gamma * x + beta
    where gamma and beta are learned from heuristic features.
    """

    def __init__(
        self,
        num_heuristics: int = NUM_HEURISTIC_FEATURES,
        hidden_dim: int = 128,
        output_dim: int = 128,
        num_filters: int = 160,
    ):
        super().__init__()
        self.num_heuristics = num_heuristics
        self.output_dim = output_dim
        self.num_filters = num_filters

        # Use deeper encoder for full 49-feature mode
        use_deep_encoder = num_heuristics >= NUM_HEURISTIC_FEATURES_FULL

        if use_deep_encoder:
            # Deep encoder: 49 -> 256 -> 256 -> 128 (3 layers)
            self.encoder = nn.Sequential(
                nn.Linear(num_heuristics, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, output_dim),
                nn.LayerNorm(output_dim),
                nn.ReLU(),
            )
        else:
            # Standard encoder: 21 -> 128 -> 128 (2 layers)
            self.encoder = nn.Sequential(
                nn.Linear(num_heuristics, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.ReLU(),
            )

        # FiLM conditioning heads
        # Produce per-channel scale (gamma) and shift (beta)
        self.film_gamma = nn.Linear(output_dim, num_filters)
        self.film_beta = nn.Linear(output_dim, num_filters)

        # Initialize FiLM to identity (gamma=1, beta=0)
        nn.init.ones_(self.film_gamma.weight.data[:, :output_dim // 2])
        nn.init.zeros_(self.film_gamma.weight.data[:, output_dim // 2:])
        nn.init.zeros_(self.film_gamma.bias.data)
        nn.init.zeros_(self.film_beta.weight.data)
        nn.init.zeros_(self.film_beta.bias.data)

    def forward(self, heuristics: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Encode heuristics into embedding and FiLM parameters.

        Args:
            heuristics: [B, num_heuristics] heuristic feature values (21 or 49)

        Returns:
            embedding: [B, output_dim] global embedding
            gamma: [B, num_filters] FiLM scale parameters
            beta: [B, num_filters] FiLM shift parameters
        """
        # Normalize heuristic inputs (they can have very different scales)
        h_norm = heuristics / (heuristics.abs().max(dim=-1, keepdim=True).values + 1e-8)

        embedding = self.encoder(h_norm)
        gamma = 1.0 + 0.1 * self.film_gamma(embedding)  # Scale around 1.0
        beta = 0.1 * self.film_beta(embedding)  # Small shifts

        return embedding, gamma, beta


class GNNRefinement(nn.Module):
    """Optional GNN refinement layer for connectivity-aware features.

    Uses 2-layer GraphSAGE to refine spatial features based on
    board connectivity (4-connected for square, 6-connected for hex).
    """

    def __init__(
        self,
        in_features: int = 160,
        hidden_features: int = 160,
        num_layers: int = 2,
    ):
        super().__init__()

        if not HAS_PYG:
            self.enabled = False
            # Fallback: identity + small MLP
            self.fallback = nn.Sequential(
                nn.Conv2d(in_features, in_features, 1),
                nn.BatchNorm2d(in_features),
                nn.ReLU(),
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
        spatial_features: Tensor,
        edge_index: Tensor | None = None,
    ) -> Tensor:
        """Refine spatial features using GNN message passing.

        Args:
            spatial_features: [B, C, H, W] spatial feature maps
            edge_index: [2, num_edges] pre-computed edge indices

        Returns:
            Refined features [B, C, H, W]
        """
        if not self.enabled or edge_index is None:
            return self.fallback(spatial_features) if hasattr(self, 'fallback') else spatial_features

        B, C, H, W = spatial_features.shape
        device = spatial_features.device

        # Reshape: [B, C, H, W] -> [B*H*W, C]
        node_features = spatial_features.permute(0, 2, 3, 1).reshape(B * H * W, C)

        # Create batched edge indices
        num_nodes = H * W
        batch_edges = []
        for b in range(B):
            batch_edges.append(edge_index + b * num_nodes)
        batched_edge_index = torch.cat(batch_edges, dim=1)

        # Message passing
        h = node_features
        for layer, norm in zip(self.layers, self.norms):
            h_new = layer(h, batched_edge_index)
            h_new = norm(h_new)
            h_new = F.relu(h_new)
            # Residual connection
            if h.shape[-1] == h_new.shape[-1]:
                h = h + h_new
            else:
                h = h_new

        # Reshape back: [B*H*W, C] -> [B, C, H, W]
        return h.reshape(B, H, W, -1).permute(0, 3, 1, 2)


class GeometryEncoder(nn.Module):
    """Encodes spatial geometry features as additional input channels.

    Adds 10 geometry-aware channels to the input:
    - distance_from_center (1): Normalized Manhattan distance from board center
    - center_zone (1): Binary mask for cells within radius 2 of center
    - directional_occupancy (8): Per-direction neighbor occupancy masks

    These provide explicit spatial bias that helps the network understand
    positional importance without relying on implicit learning.
    """

    def __init__(self, board_size: int = 8):
        super().__init__()
        self.board_size = board_size
        self.center = board_size // 2
        self.max_dist = self.center * 2.0

        # Pre-compute static geometry features
        self._register_geometry_buffers()

    def _register_geometry_buffers(self) -> None:
        """Pre-compute and register geometry feature planes as buffers."""
        H = W = self.board_size
        center = self.center

        # Distance from center (normalized to [0, 1])
        y_coords = torch.arange(H).float().view(H, 1).expand(H, W)
        x_coords = torch.arange(W).float().view(1, W).expand(H, W)
        dist_from_center = (
            (y_coords - center).abs() + (x_coords - center).abs()
        ) / self.max_dist
        self.register_buffer("dist_from_center", dist_from_center.unsqueeze(0).unsqueeze(0))

        # Center zone mask (cells within radius 2 of center)
        center_zone = ((y_coords - center).abs() <= 2) & ((x_coords - center).abs() <= 2)
        self.register_buffer("center_zone", center_zone.float().unsqueeze(0).unsqueeze(0))

        # 8 directional shift masks for neighbor detection
        # Directions: NW, N, NE, W, E, SW, S, SE
        directions = [
            (-1, -1), (-1, 0), (-1, 1),  # NW, N, NE
            (0, -1),           (0, 1),   # W, E
            (1, -1),  (1, 0),  (1, 1),   # SW, S, SE
        ]
        self.directions = directions

    def forward(self, spatial_features: Tensor) -> Tensor:
        """Compute geometry encoding channels.

        Args:
            spatial_features: [B, C, H, W] input features (used to determine batch size)

        Returns:
            Geometry channels [B, 10, H, W] to concatenate with input
        """
        B, C, H, W = spatial_features.shape
        device = spatial_features.device

        # Static geometry features (broadcast to batch)
        dist_channel = self.dist_from_center.expand(B, -1, -1, -1).to(device)
        center_channel = self.center_zone.expand(B, -1, -1, -1).to(device)

        # Compute directional occupancy from spatial features
        # Use first channel as occupancy indicator (has_stack proxy)
        occupancy = (spatial_features[:, 0:1, :, :] > 0).float()

        # Shift occupancy in each direction to detect neighbors
        direction_channels = []
        for dy, dx in self.directions:
            shifted = torch.zeros_like(occupancy)
            # Compute valid source and target regions
            src_y_start = max(0, -dy)
            src_y_end = H - max(0, dy)
            src_x_start = max(0, -dx)
            src_x_end = W - max(0, dx)
            tgt_y_start = max(0, dy)
            tgt_y_end = H - max(0, -dy)
            tgt_x_start = max(0, dx)
            tgt_x_end = W - max(0, -dx)

            shifted[:, :, tgt_y_start:tgt_y_end, tgt_x_start:tgt_x_end] = \
                occupancy[:, :, src_y_start:src_y_end, src_x_start:src_x_end]
            direction_channels.append(shifted)

        dir_stack = torch.cat(direction_channels, dim=1)  # [B, 8, H, W]

        # Concatenate all geometry channels
        geometry = torch.cat([dist_channel, center_channel, dir_stack], dim=1)  # [B, 10, H, W]
        return geometry


class RingRiftCNN_v5_Heavy(nn.Module):
    """V5 Heavy Architecture: Maximum strength for RingRift.

    Combines all architectural innovations:
    - SE blocks + Multi-head attention (mixed backbone)
    - Configurable heuristic features (21 fast or 49 full) with FiLM conditioning
    - Optional geometry encoding (distance-from-center, directional occupancy)
    - Optional GNN refinement
    - Spatial policy heads
    - Rank distribution value head

    Args:
        board_size: Board dimension (8 or 19)
        in_channels: Base input channels per frame (default 14)
        global_features: Global feature dimension (default 20)
        num_heuristics: Heuristic feature dimension (21 for fast, 49 for full)
        use_geometry_encoding: Add geometry feature planes (default False)
        num_se_blocks: Number of SE residual blocks (default 6)
        num_attention_blocks: Number of attention blocks (default 5)
        num_filters: CNN channel count (default 160)
        history_length: Frames of history (default 3)
        policy_size: Policy output size (auto-detected if None)
        num_players: Number of players (default 4)
        use_gnn: Enable GNN refinement layer (default False)
        dropout: Dropout rate (default 0.1)
        initial_kernel_size: Initial conv kernel (default 5, NAS optimal)
    """

    ARCHITECTURE_VERSION = "v5.1.0"

    def __init__(
        self,
        board_size: int = 19,
        in_channels: int = 14,
        global_features: int = 20,
        num_heuristics: int = NUM_HEURISTIC_FEATURES,
        num_se_blocks: int = 6,
        num_attention_blocks: int = 5,
        num_filters: int = 160,
        history_length: int = 3,
        policy_size: int | None = None,
        num_players: int = 4,
        use_gnn: bool = False,
        use_geometry_encoding: bool = False,
        dropout: float = 0.1,
        initial_kernel_size: int = 5,
        num_attention_heads: int = 4,
        se_reduction: int = 16,
        num_ring_counts: int = 3,
        num_directions: int = NUM_SQUARE_DIRS,
        num_line_dirs: int = NUM_LINE_DIRS,
        territory_size_buckets: int = TERRITORY_SIZE_BUCKETS,
        territory_max_players: int = TERRITORY_MAX_PLAYERS,
    ) -> None:
        super().__init__()

        self.board_size = board_size
        self.num_filters = num_filters
        self.num_players = num_players
        self.global_features = global_features
        self.num_heuristics = num_heuristics
        self.use_gnn = use_gnn and HAS_PYG
        self.use_geometry_encoding = use_geometry_encoding
        self.dropout_rate = dropout

        # Determine max distance based on board size
        if board_size == 8:
            self.max_distance = MAX_DIST_SQUARE8
        else:
            self.max_distance = MAX_DIST_SQUARE19

        # Spatial policy dimensions
        self.num_ring_counts = num_ring_counts
        self.num_directions = num_directions
        self.num_line_dirs = num_line_dirs
        self.territory_size_buckets = territory_size_buckets
        self.territory_max_players = territory_max_players
        self.movement_channels = num_directions * self.max_distance
        self.territory_choice_channels = territory_size_buckets * territory_max_players

        # Input channels = base_channels * (history_length + 1) + optional geometry
        base_in_channels = in_channels * (history_length + 1)
        geometry_channels = NUM_GEOMETRY_CHANNELS if use_geometry_encoding else 0
        self.total_in_channels = base_in_channels + geometry_channels
        self.in_channels = self.total_in_channels
        self.base_in_channels = base_in_channels

        # === Geometry Encoder (optional) ===
        if use_geometry_encoding:
            self.geometry_encoder = GeometryEncoder(board_size=board_size)
        else:
            self.geometry_encoder = None

        # Determine policy size
        if policy_size is not None:
            self.policy_size = policy_size
        elif board_size == 8:
            self.policy_size = POLICY_SIZE_8x8
        elif board_size == 19:
            self.policy_size = POLICY_SIZE_19x19
        else:
            self.policy_size = POLICY_SIZE_8x8  # Fallback

        # Pre-compute policy index tensors
        self._register_policy_indices(board_size)

        # === Heuristic Encoder ===
        heuristic_embed_dim = 128
        self.heuristic_encoder = HeuristicEncoder(
            num_heuristics=num_heuristics,
            hidden_dim=128,
            output_dim=heuristic_embed_dim,
            num_filters=num_filters,
        )

        # === Initial Convolution (5x5, NAS optimal) ===
        self.conv1 = nn.Conv2d(
            self.total_in_channels,
            num_filters,
            kernel_size=initial_kernel_size,
            padding=initial_kernel_size // 2,
        )
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU()

        # === Mixed Backbone: SE + Attention ===
        # First half: SE blocks for channel attention
        self.se_blocks = nn.ModuleList([
            SEResidualBlock(num_filters, reduction=se_reduction)
            for _ in range(num_se_blocks)
        ])

        # Second half: Attention blocks for long-range dependencies
        self.attention_blocks = nn.ModuleList([
            AttentionResidualBlock(
                num_filters,
                num_heads=num_attention_heads,
                dropout=dropout,
            )
            for _ in range(num_attention_blocks)
        ])

        # === Optional GNN Refinement ===
        if self.use_gnn:
            self.gnn_refinement = GNNRefinement(
                in_features=num_filters,
                hidden_features=num_filters,
                num_layers=2,
            )
            self._edge_index = self._build_edge_index()
        else:
            self.gnn_refinement = None
            self._edge_index = None

        # === Value Head (Rank Distribution) ===
        # Concatenate: pooled_features + globals + heuristic_embedding
        value_input_dim = num_filters + global_features + heuristic_embed_dim
        value_hidden = 256

        self.value_fc1 = nn.Linear(value_input_dim, value_hidden)
        self.value_fc2 = nn.Linear(value_hidden, value_hidden)
        self.value_fc3 = nn.Linear(value_hidden, num_players)
        self.tanh = nn.Tanh()
        self.value_dropout = nn.Dropout(dropout)

        # Rank Distribution Head
        rank_hidden = value_hidden * 2
        self.rank_fc1 = nn.Linear(value_input_dim, rank_hidden)
        self.rank_fc2 = nn.Linear(rank_hidden, num_players * num_players)
        self.rank_softmax = nn.Softmax(dim=-1)

        # === Spatial Policy Heads (v3-style) ===
        self.placement_conv = nn.Conv2d(num_filters, num_ring_counts, kernel_size=1)
        self.movement_conv = nn.Conv2d(num_filters, self.movement_channels, kernel_size=1)
        self.line_form_conv = nn.Conv2d(num_filters, num_line_dirs, kernel_size=1)
        self.territory_claim_conv = nn.Conv2d(num_filters, 1, kernel_size=1)
        self.territory_choice_conv = nn.Conv2d(num_filters, self.territory_choice_channels, kernel_size=1)

        # Special actions FC (skip, swap, line choice, etc.)
        special_input_dim = num_filters + global_features + heuristic_embed_dim
        self.special_fc = nn.Linear(special_input_dim, 7)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _build_edge_index(self) -> torch.Tensor:
        """Build edge index for square board connectivity (4-connected)."""
        edges = []
        size = self.board_size

        for y in range(size):
            for x in range(size):
                node_idx = y * size + x
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < size and 0 <= ny < size:
                        neighbor_idx = ny * size + nx
                        edges.append([node_idx, neighbor_idx])

        return torch.tensor(edges, dtype=torch.long).t().contiguous()

    def _register_policy_indices(self, board_size: int) -> None:
        """Pre-compute index tensors for scattering spatial logits into flat policy."""
        H, W = board_size, board_size

        if board_size == 8:
            movement_base = SQUARE8_MOVEMENT_BASE
            line_form_base = SQUARE8_LINE_FORM_BASE
            territory_claim_base = SQUARE8_TERRITORY_CLAIM_BASE
            territory_choice_base = SQUARE8_TERRITORY_CHOICE_BASE
            max_dist = MAX_DIST_SQUARE8
        else:
            movement_base = SQUARE19_MOVEMENT_BASE
            line_form_base = SQUARE19_LINE_FORM_BASE
            territory_claim_base = SQUARE19_TERRITORY_CLAIM_BASE
            territory_choice_base = SQUARE19_TERRITORY_CHOICE_BASE
            max_dist = MAX_DIST_SQUARE19

        # Placement indices: [3, H, W] -> flat index
        placement_idx = torch.zeros(self.num_ring_counts, H, W, dtype=torch.long)
        for y in range(H):
            for x in range(W):
                for r in range(self.num_ring_counts):
                    placement_idx[r, y, x] = y * W * 3 + x * 3 + r
        self.register_buffer("placement_idx", placement_idx)

        # Movement indices
        movement_idx = torch.zeros(self.movement_channels, H, W, dtype=torch.long)
        for y in range(H):
            for x in range(W):
                for d in range(self.num_directions):
                    for dist_minus_1 in range(max_dist):
                        channel = d * max_dist + dist_minus_1
                        flat_idx = (
                            movement_base
                            + y * W * self.num_directions * max_dist
                            + x * self.num_directions * max_dist
                            + d * max_dist
                            + dist_minus_1
                        )
                        movement_idx[channel, y, x] = flat_idx
        self.register_buffer("movement_idx", movement_idx)

        # Line formation indices
        line_form_idx = torch.zeros(self.num_line_dirs, H, W, dtype=torch.long)
        for y in range(H):
            for x in range(W):
                for ld in range(self.num_line_dirs):
                    line_form_idx[ld, y, x] = line_form_base + y * W * self.num_line_dirs + x * self.num_line_dirs + ld
        self.register_buffer("line_form_idx", line_form_idx)

        # Territory claim indices
        territory_claim_idx = torch.zeros(1, H, W, dtype=torch.long)
        for y in range(H):
            for x in range(W):
                territory_claim_idx[0, y, x] = territory_claim_base + y * W + x
        self.register_buffer("territory_claim_idx", territory_claim_idx)

        # Territory choice indices
        territory_choice_idx = torch.zeros(self.territory_choice_channels, H, W, dtype=torch.long)
        for y in range(H):
            for x in range(W):
                for size_bucket in range(self.territory_size_buckets):
                    for player_idx in range(self.territory_max_players):
                        channel = size_bucket * self.territory_max_players + player_idx
                        flat_idx = (
                            territory_choice_base
                            + y * W * self.territory_choice_channels
                            + x * self.territory_choice_channels
                            + size_bucket * self.territory_max_players
                            + player_idx
                        )
                        territory_choice_idx[channel, y, x] = flat_idx
        self.register_buffer("territory_choice_idx", territory_choice_idx)

        # Store special action indices
        if board_size == 8:
            self.skip_placement_idx = SQUARE8_SKIP_PLACEMENT_IDX
            self.swap_sides_idx = SQUARE8_SWAP_SIDES_IDX
            self.skip_recovery_idx = SQUARE8_SKIP_RECOVERY_IDX
            self.line_choice_base = SQUARE8_LINE_CHOICE_BASE
            self.extra_special_base = SQUARE8_EXTRA_SPECIAL_BASE
            self.extra_special_span = SQUARE8_EXTRA_SPECIAL_SPAN
        else:
            self.skip_placement_idx = SQUARE19_SKIP_PLACEMENT_IDX
            self.swap_sides_idx = SQUARE19_SWAP_SIDES_IDX
            self.skip_recovery_idx = SQUARE19_SKIP_RECOVERY_IDX
            self.line_choice_base = SQUARE19_LINE_CHOICE_BASE
            self.extra_special_base = SQUARE19_EXTRA_SPECIAL_BASE
            self.extra_special_span = SQUARE19_EXTRA_SPECIAL_SPAN

    def _scatter_policy_logits(
        self,
        placement_logits: Tensor,
        movement_logits: Tensor,
        line_form_logits: Tensor,
        territory_claim_logits: Tensor,
        territory_choice_logits: Tensor,
        special_logits: Tensor,
    ) -> Tensor:
        """Scatter spatial policy logits into flat policy vector."""
        B = placement_logits.size(0)
        device = placement_logits.device
        dtype = placement_logits.dtype

        policy = torch.full((B, self.policy_size), -1e4, device=device, dtype=dtype)

        placement_flat = placement_logits.view(B, -1)
        placement_idx_flat = self.placement_idx.view(-1).expand(B, -1)
        policy.scatter_(1, placement_idx_flat, placement_flat)

        movement_flat = movement_logits.view(B, -1)
        movement_idx_flat = self.movement_idx.view(-1).expand(B, -1)
        policy.scatter_(1, movement_idx_flat, movement_flat)

        line_form_flat = line_form_logits.view(B, -1)
        line_form_idx_flat = self.line_form_idx.view(-1).expand(B, -1)
        policy.scatter_(1, line_form_idx_flat, line_form_flat)

        territory_claim_flat = territory_claim_logits.view(B, -1)
        territory_claim_idx_flat = self.territory_claim_idx.view(-1).expand(B, -1)
        policy.scatter_(1, territory_claim_idx_flat, territory_claim_flat)

        territory_choice_flat = territory_choice_logits.view(B, -1)
        territory_choice_idx_flat = self.territory_choice_idx.view(-1).expand(B, -1)
        policy.scatter_(1, territory_choice_idx_flat, territory_choice_flat)

        policy[:, self.skip_placement_idx] = special_logits[:, 0]
        policy[:, self.swap_sides_idx] = special_logits[:, 1]
        policy[:, self.skip_recovery_idx] = special_logits[:, 2]
        policy[:, self.line_choice_base : self.line_choice_base + 4] = special_logits[:, 3:7]
        if self.extra_special_span > 0:
            policy[
                :,
                self.extra_special_base : self.extra_special_base + self.extra_special_span,
            ] = special_logits[:, 0:1]

        return policy

    def forward(
        self,
        x: Tensor,
        globals_: Tensor,
        heuristics: Tensor | None = None,
        return_features: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, Tensor]:
        """Forward pass with heuristic feature conditioning.

        Args:
            x: Input features [B, in_channels, H, W] or [B, base_channels, H, W]
               If geometry encoding is enabled and base channels provided,
               geometry channels are automatically added.
            globals_: Global features [B, global_features]
            heuristics: Heuristic features [B, num_heuristics] (21 or 49)
                Optional, zeros if None.
            return_features: If True, also return backbone features

        Returns:
            value: [B, num_players] value predictions (tanh)
            policy: [B, policy_size] policy logits
            rank_dist: [B, num_players, num_players] rank distribution
            features (optional): [B, feature_dim] backbone features
        """
        B = x.shape[0]
        device = x.device

        # Handle missing heuristics (backward compatibility)
        if heuristics is None:
            heuristics = torch.zeros(B, self.num_heuristics, device=device, dtype=x.dtype)

        # === Geometry Encoding (optional) ===
        if self.geometry_encoder is not None:
            # Expect base input channels, will add geometry channels
            if x.shape[1] == self.base_in_channels:
                geometry_features = self.geometry_encoder(x)
                x = torch.cat([x, geometry_features], dim=1)
            # If already has geometry channels, skip encoding
            elif x.shape[1] != self.in_channels:
                raise RuntimeError(
                    f"Input channel mismatch in {self.__class__.__name__}.forward():\n"
                    f"  Input has {x.shape[1]} channels\n"
                    f"  Model expects {self.base_in_channels} (base) or {self.in_channels} (with geometry)"
                )

        # Input validation (for non-geometry mode)
        if self.geometry_encoder is None and x.shape[1] != self.in_channels:
            raise RuntimeError(
                f"Input channel mismatch in {self.__class__.__name__}.forward():\n"
                f"  Input has {x.shape[1]} channels\n"
                f"  Model expects {self.in_channels} channels"
            )

        # === Encode Heuristics ===
        h_embed, film_gamma, film_beta = self.heuristic_encoder(heuristics)

        # === Initial Convolution ===
        out = self.relu(self.bn1(self.conv1(x)))

        # === SE Blocks (first half) ===
        for block in self.se_blocks:
            out = block(out)

        # === FiLM Conditioning (after SE, before attention) ===
        # Apply heuristic-conditioned modulation
        out = film_gamma.unsqueeze(-1).unsqueeze(-1) * out + film_beta.unsqueeze(-1).unsqueeze(-1)

        # === Attention Blocks (second half) ===
        for block in self.attention_blocks:
            out = block(out)

        # === Optional GNN Refinement ===
        if self.gnn_refinement is not None:
            edge_index = self._edge_index.to(device) if self._edge_index is not None else None
            out = self.gnn_refinement(out, edge_index)

        # === Pooled Features for Value Head ===
        pooled = torch.mean(out, dim=[-2, -1])
        combined = torch.cat([pooled, globals_, h_embed], dim=1)

        # === Value Head (legacy tanh output) ===
        v = self.relu(self.value_fc1(combined))
        v = self.value_dropout(v)
        v = self.relu(self.value_fc2(v))
        v = self.value_dropout(v)
        value = self.tanh(self.value_fc3(v))

        # === Rank Distribution Head ===
        r = self.relu(self.rank_fc1(combined))
        r = self.value_dropout(r)
        rank_logits = self.rank_fc2(r).view(B, self.num_players, self.num_players)
        rank_dist = self.rank_softmax(rank_logits)

        # === Spatial Policy Heads ===
        placement_logits = self.placement_conv(out)
        movement_logits = self.movement_conv(out)
        line_form_logits = self.line_form_conv(out)
        territory_claim_logits = self.territory_claim_conv(out)
        territory_choice_logits = self.territory_choice_conv(out)

        special_logits = self.special_fc(combined)

        policy = self._scatter_policy_logits(
            placement_logits,
            movement_logits,
            line_form_logits,
            territory_claim_logits,
            territory_choice_logits,
            special_logits,
        )

        if return_features:
            return value, policy, rank_dist, combined

        return value, policy, rank_dist

    def forward_single(
        self,
        feature: np.ndarray,
        globals_vec: np.ndarray,
        heuristics_vec: np.ndarray | None = None,
        player_idx: int = 0,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Convenience method for single-sample inference.

        Args:
            feature: Board features [C, H, W]
            globals_vec: Global features [G]
            heuristics_vec: Heuristic features [49] (optional)
            player_idx: Which player's value to return

        Returns:
            Tuple of (value for player, policy logits, rank distribution)
        """
        self.eval()
        device = next(self.parameters()).device

        with torch.no_grad():
            x = torch.from_numpy(feature[None, ...]).float().to(device)
            g = torch.from_numpy(globals_vec[None, ...]).float().to(device)

            if heuristics_vec is not None:
                h = torch.from_numpy(heuristics_vec[None, ...]).float().to(device)
            else:
                h = None

            v, p, rank_dist = self.forward(x, g, h)

        return (
            float(v[0, player_idx].item()),
            p.cpu().numpy()[0],
            rank_dist.cpu().numpy()[0],
        )

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# === Hex V5 Architecture ===

class HexNeuralNet_v5_Heavy(RingRiftCNN_v5_Heavy):
    """V5 Heavy Architecture for hexagonal boards.

    Inherits from RingRiftCNN_v5_Heavy with hex-specific modifications:
    - 6-connected edge index for GNN
    - Hex-specific policy indices
    - Hex mask for valid cell identification

    Note: Currently uses square policy indexing - hex-specific spatial heads
    would require additional implementation similar to HexNeuralNet_v3.
    """

    ARCHITECTURE_VERSION = "v5.0.0-hex"

    def __init__(
        self,
        board_size: int = 9,
        hex_radius: int = 4,
        **kwargs,
    ):
        # Store hex-specific params before calling parent
        self.hex_radius = hex_radius

        # For hex, policy size and indices differ
        # This is a simplified version - full hex spatial heads would need more work
        super().__init__(board_size=board_size, **kwargs)

    def _build_edge_index(self) -> torch.Tensor:
        """Build edge index for hex board (6-connected)."""
        edges = []
        size = self.board_size

        # 6 hex directions
        hex_dirs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]

        for y in range(size):
            for x in range(size):
                node_idx = y * size + x
                for dx, dy in hex_dirs:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < size and 0 <= ny < size:
                        neighbor_idx = ny * size + nx
                        edges.append([node_idx, neighbor_idx])

        return torch.tensor(edges, dtype=torch.long).t().contiguous()


def create_v5_heavy_model(
    board_type: str = "square8",
    num_players: int = 2,
    use_gnn: bool = False,
    **kwargs,
) -> RingRiftCNN_v5_Heavy | HexNeuralNet_v5_Heavy:
    """Factory function for V5 Heavy models.

    Args:
        board_type: 'square8', 'square19', 'hex8', or 'hexagonal'
        num_players: Number of players (2-4)
        use_gnn: Enable GNN refinement layer
        **kwargs: Override parameters

    Returns:
        Configured V5 Heavy model
    """
    configs = {
        "square8": {"board_size": 8, "policy_size": POLICY_SIZE_8x8},
        "square19": {"board_size": 19, "policy_size": POLICY_SIZE_19x19},
        "hex8": {"board_size": 9, "hex_radius": 4, "is_hex": True},
        "hexagonal": {"board_size": 25, "hex_radius": 12, "is_hex": True},
    }

    config = configs.get(board_type, configs["square8"])
    is_hex = config.pop("is_hex", False)
    config.update(kwargs)
    config["num_players"] = num_players
    config["use_gnn"] = use_gnn

    if is_hex:
        return HexNeuralNet_v5_Heavy(**config)
    else:
        return RingRiftCNN_v5_Heavy(**config)


__all__ = [
    # Model classes
    "RingRiftCNN_v5_Heavy",
    "HexNeuralNet_v5_Heavy",
    # Encoder components
    "HeuristicEncoder",
    "GeometryEncoder",
    "GNNRefinement",
    # Factory function
    "create_v5_heavy_model",
    # Constants
    "NUM_HEURISTIC_FEATURES",
    "NUM_HEURISTIC_FEATURES_FAST",
    "NUM_HEURISTIC_FEATURES_FULL",
    "NUM_GEOMETRY_CHANNELS",
]
