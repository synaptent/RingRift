"""Hexagonal board CNN architectures for RingRift AI.

This module contains the CNN architecture classes for hexagonal boards:
- HexNeuralNet_v2: High-capacity SE architecture with hex masking (96GB systems)
- HexNeuralNet_v2_Lite: Memory-efficient SE architecture with hex masking (48GB systems)
- HexNeuralNet_v3: Spatial policy heads with SE backbone
- HexNeuralNet_v3_Lite: Memory-efficient spatial policy heads

Migrated from _neural_net_legacy.py as part of Phase 2 modularization.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import SEResidualBlock, create_hex_mask
from .constants import (
    HEX_BOARD_SIZE,
    HEX_MAX_DIST,
    HEX_MOVEMENT_BASE,
    HEX_SPECIAL_BASE,
    NUM_HEX_DIRS,
    P_HEX,
)


class HexNeuralNet_v2(nn.Module):
    """
    High-capacity CNN for hexagonal boards (96GB memory target).

    This architecture fixes the critical bug in HexNeuralNet where the policy
    head flattened full spatial features (80,000 dims) directly to policy logits,
    resulting in 7.35 billion parameters. The v2 architecture uses global average
    pooling before the policy FC layer, reducing parameters by 169×.

    Key improvements over HexNeuralNet:
    - Policy head uses global avg pool → FC (like RingRiftCNN_MPS)
    - 12 SE residual blocks with Squeeze-and-Excitation
    - 192 filters for richer hex representations
    - 14 base input channels capturing stack/cap/territory mechanics
    - 20 global features for multi-player state tracking
    - Multi-player value head with masked pooling for hex grid
    - 384-dim policy intermediate for better move discrimination

    Hex-specific features:
    - Automatic hex mask generation and caching
    - Masked global average pooling for valid cells only
    - 469 valid cells in 25×25 bounding box (radius 12)

    Input Feature Channels (14 base × 4 frames = 56 total):
        1-4: Per-player stack presence (binary, one per player)
        5-8: Per-player marker presence (binary)
        9: Stack height (normalized 0-1)
        10: Cap height (normalized)
        11: Collapsed territory (binary)
        12-14: Territory ownership channels

    Global Features (20):
        1-4: Rings in hand (per player)
        5-8: Eliminated rings (per player)
        9-12: Territory count (per player)
        13-16: Line count (per player)
        17: Current player indicator
        18: Game phase (early/mid/late)
        19: Total rings in play
        20: LPS threat indicator

    Memory profile (FP32):
    - Model weights: ~180 MB (vs ~29 GB in original!)
    - Per-model with activations: ~380 MB
    - Two models + MCTS: ~18 GB total

    Architecture Version:
        v2.0.0 - Fixed policy head, SE blocks, high-capacity for 96GB systems.
    """

    ARCHITECTURE_VERSION = "v2.0.0"

    def __init__(
        self,
        in_channels: int = 40,  # 10 base × 4 frames (hex uses fewer channels than square)
        global_features: int = 20,
        num_res_blocks: int = 12,
        num_filters: int = 192,
        board_size: int = HEX_BOARD_SIZE,
        policy_size: int = P_HEX,
        policy_intermediate: int = 384,
        value_intermediate: int = 128,
        num_players: int = 4,
        se_reduction: int = 16,
        hex_radius: int = 12,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.global_features = global_features
        self.num_res_blocks = num_res_blocks
        self.num_filters = num_filters
        self.board_size = board_size
        self.policy_size = policy_size
        self.num_players = num_players

        # Pre-compute hex validity mask
        self.register_buffer("hex_mask", create_hex_mask(hex_radius, board_size))

        # Shared backbone with SE blocks
        self.conv1 = nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU()
        self.res_blocks = nn.ModuleList(
            [SEResidualBlock(num_filters, reduction=se_reduction) for _ in range(num_res_blocks)]
        )

        # Multi-player value head with masked pooling
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 + global_features, value_intermediate)
        self.value_fc2 = nn.Linear(value_intermediate, num_players)
        self.tanh = nn.Tanh()

        # Policy head: 1×1 conv → global avg pool → FC (FIXED!)
        self.policy_conv = nn.Conv2d(num_filters, num_filters, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(num_filters)
        self.policy_fc1 = nn.Linear(num_filters + global_features, policy_intermediate)
        self.policy_fc2 = nn.Linear(policy_intermediate, policy_size)
        self.dropout = nn.Dropout(0.3)

    def _masked_global_avg_pool(
        self,
        x: torch.Tensor,
        hex_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Perform masked global average pooling over [H, W] for hex grid."""
        mask = hex_mask if hex_mask is not None else self.hex_mask
        if mask is None:
            return x.mean(dim=(2, 3))
        mask = mask.to(dtype=x.dtype, device=x.device)
        masked = x * mask
        num = masked.sum(dim=(2, 3))
        denom = mask.sum(dim=(2, 3)).clamp_min(1.0)
        return num / denom

    def forward(
        self,
        x: torch.Tensor,
        globals: torch.Tensor,
        hex_mask: torch.Tensor | None = None,
        return_features: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Apply hex mask to input to prevent information bleeding
        if hex_mask is not None:
            x = x * hex_mask.to(dtype=x.dtype, device=x.device)
        elif self.hex_mask is not None:
            x = x * self.hex_mask.to(dtype=x.dtype, device=x.device)

        # Backbone with SE blocks
        out = self.relu(self.bn1(self.conv1(x)))
        for block in self.res_blocks:
            out = block(out)

        # Global pooled features for optional return
        out_pooled = self._masked_global_avg_pool(out, hex_mask)

        # Multi-player value head with masked pooling
        v = self.value_conv(out)
        v = self.relu(self.value_bn(v))
        v_pooled = self._masked_global_avg_pool(v, hex_mask)
        v_cat = torch.cat([v_pooled, globals], dim=1)
        v_hidden = self.relu(self.value_fc1(v_cat))
        v_hidden = self.dropout(v_hidden)
        v_out = self.tanh(self.value_fc2(v_hidden))  # [B, num_players]

        # Policy head with masked global avg pool
        p = self.policy_conv(out)
        p = self.relu(self.policy_bn(p))
        p_pooled = self._masked_global_avg_pool(p, hex_mask)
        p_cat = torch.cat([p_pooled, globals], dim=1)
        p_hidden = self.relu(self.policy_fc1(p_cat))
        p_hidden = self.dropout(p_hidden)
        p_logits = self.policy_fc2(p_hidden)

        if return_features:
            return v_out, p_logits, out_pooled

        return v_out, p_logits


class HexNeuralNet_v2_Lite(nn.Module):
    """
    Memory-efficient CNN for hexagonal boards (48GB memory target).

    This architecture provides the same bug fix as HexNeuralNet_v2 but with
    reduced capacity for systems with limited memory (48GB). Suitable for
    running two instances simultaneously for comparison matches.

    Key trade-offs vs HexNeuralNet_v2:
    - 6 SE residual blocks (vs 12) - faster but shallower
    - 96 filters (vs 192) - smaller representations
    - 192-dim policy intermediate (vs 384)
    - 12 base input channels (vs 14) - reduced history
    - 3 history frames (vs 4) - reduced temporal context

    Hex-specific features:
    - Automatic hex mask generation and caching
    - Masked global average pooling for valid cells only
    - Input masking to prevent information bleeding
    - 469 valid cells in 25×25 bounding box (radius 12)

    Input Feature Channels (12 base × 3 frames = 36 total):
        1-4: Per-player stack presence (binary)
        5-8: Per-player marker presence (binary)
        9: Stack height (normalized)
        10: Cap height (normalized)
        11: Collapsed territory (binary)
        12: Current player territory

    Global Features (20):
        Same as HexNeuralNet_v2 for compatibility

    Memory profile (FP32):
    - Model weights: ~75 MB
    - Per-model with activations: ~150 MB
    - Two models + MCTS: ~10 GB total

    Architecture Version:
        v2.0.0-lite - SE blocks, hex masking, memory-efficient for 48GB.
    """

    ARCHITECTURE_VERSION = "v2.0.0-lite"

    def __init__(
        self,
        in_channels: int = 36,  # 12 base × 3 frames
        global_features: int = 20,
        num_res_blocks: int = 6,
        num_filters: int = 96,
        board_size: int = HEX_BOARD_SIZE,
        policy_size: int = P_HEX,
        policy_intermediate: int = 192,
        value_intermediate: int = 64,
        num_players: int = 4,
        se_reduction: int = 16,
        hex_radius: int = 12,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.global_features = global_features
        self.num_res_blocks = num_res_blocks
        self.num_filters = num_filters
        self.board_size = board_size
        self.policy_size = policy_size
        self.num_players = num_players

        # Pre-compute hex validity mask
        self.register_buffer("hex_mask", create_hex_mask(hex_radius, board_size))

        # Shared backbone with SE blocks
        self.conv1 = nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU()
        self.res_blocks = nn.ModuleList(
            [SEResidualBlock(num_filters, reduction=se_reduction) for _ in range(num_res_blocks)]
        )

        # Multi-player value head with masked pooling
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 + global_features, value_intermediate)
        self.value_fc2 = nn.Linear(value_intermediate, num_players)
        self.tanh = nn.Tanh()

        # Policy head: 1×1 conv → masked global avg pool → FC (FIXED!)
        self.policy_conv = nn.Conv2d(num_filters, num_filters, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(num_filters)
        self.policy_fc1 = nn.Linear(num_filters + global_features, policy_intermediate)
        self.policy_fc2 = nn.Linear(policy_intermediate, policy_size)
        self.dropout = nn.Dropout(0.3)

    def _masked_global_avg_pool(
        self,
        x: torch.Tensor,
        hex_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Perform masked global average pooling over [H, W] for hex grid."""
        mask = hex_mask if hex_mask is not None else self.hex_mask
        if mask is None:
            return x.mean(dim=(2, 3))
        mask = mask.to(dtype=x.dtype, device=x.device)
        masked = x * mask
        num = masked.sum(dim=(2, 3))
        denom = mask.sum(dim=(2, 3)).clamp_min(1.0)
        return num / denom

    def forward(
        self,
        x: torch.Tensor,
        globals: torch.Tensor,
        hex_mask: torch.Tensor | None = None,
        return_features: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Apply hex mask to input to prevent information bleeding
        if hex_mask is not None:
            x = x * hex_mask.to(dtype=x.dtype, device=x.device)
        elif self.hex_mask is not None:
            x = x * self.hex_mask.to(dtype=x.dtype, device=x.device)

        # Backbone with SE blocks
        out = self.relu(self.bn1(self.conv1(x)))
        for block in self.res_blocks:
            out = block(out)

        # Global pooled features for optional return
        out_pooled = self._masked_global_avg_pool(out, hex_mask)

        # Multi-player value head with masked pooling
        v = self.value_conv(out)
        v = self.relu(self.value_bn(v))
        v_pooled = self._masked_global_avg_pool(v, hex_mask)
        v_cat = torch.cat([v_pooled, globals], dim=1)
        v_hidden = self.relu(self.value_fc1(v_cat))
        v_hidden = self.dropout(v_hidden)
        v_out = self.tanh(self.value_fc2(v_hidden))  # [B, num_players]

        # Policy head with masked global avg pool
        p = self.policy_conv(out)
        p = self.relu(self.policy_bn(p))
        p_pooled = self._masked_global_avg_pool(p, hex_mask)
        p_cat = torch.cat([p_pooled, globals], dim=1)
        p_hidden = self.relu(self.policy_fc1(p_cat))
        p_hidden = self.dropout(p_hidden)
        p_logits = self.policy_fc2(p_hidden)

        if return_features:
            return v_out, p_logits, out_pooled

        return v_out, p_logits


class HexNeuralNet_v3(nn.Module):
    """
    V3 architecture with spatial policy heads for hexagonal boards.

    This architecture improves on V2 by using spatially-structured policy heads
    that preserve the geometric relationship between positions and actions,
    rather than collapsing everything through global average pooling.

    Key improvements over V2:
    - Spatial placement head: Conv1×1 → [B, 3, H, W] for (cell, ring_count) logits
    - Spatial movement head: Conv1×1 → [B, 144, H, W] for (cell, dir, dist) logits
    - Small FC for special actions (skip_placement only)
    - Logits are scattered into canonical P_HEX=91,876 flat policy vector
    - Preserves spatial locality during policy computation

    Why spatial heads are better:
    1. No spatial information loss - each cell produces its own policy logits
    2. Better gradient flow - actions at position (x,y) directly update features at (x,y)
    3. Reduced parameter count - Conv1×1 vs large FC layer
    4. Natural hex masking - invalid cells produce masked logits

    Policy Layout (P_HEX = 91,876):
        Placements:  [0, 1874]     = 25×25×3 = 1,875 (cell × ring_count)
        Movements:   [1875, 91874] = 25×25×6×24 = 90,000 (cell × dir × dist)
        Special:     [91875]       = 1 (skip_placement)

    Architecture Version:
        v3.0.0 - Spatial policy heads, SE backbone, MPS compatible.
    """

    ARCHITECTURE_VERSION = "v3.0.0"

    def __init__(
        self,
        in_channels: int = 64,  # 16 base × 4 frames for V3 encoder
        global_features: int = 20,  # V3 encoder provides 20 global features
        num_res_blocks: int = 12,
        num_filters: int = 192,
        board_size: int = HEX_BOARD_SIZE,
        policy_size: int = P_HEX,
        value_intermediate: int = 128,
        num_players: int = 4,
        se_reduction: int = 16,
        hex_radius: int = 12,
        num_ring_counts: int = 3,  # Ring count options (1, 2, 3)
        num_directions: int = NUM_HEX_DIRS,  # 6 hex directions
        max_distance: int | None = None,  # Computed from board_size if None
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.global_features = global_features
        self.num_res_blocks = num_res_blocks
        self.num_filters = num_filters
        self.board_size = board_size
        self.policy_size = policy_size
        self.num_players = num_players

        # Spatial policy dimensions
        self.num_ring_counts = num_ring_counts
        self.num_directions = num_directions
        # max_distance = board_size - 1: hex8 (9x9) uses 8, hexagonal (25x25) uses 24
        self.max_distance = max_distance if max_distance is not None else board_size - 1
        self.movement_channels = num_directions * self.max_distance

        # Compute layout spans dynamically based on board_size
        # This ensures hex8 (9x9) and hexagonal (25x25) use correct indices
        self.placement_span = board_size * board_size * num_ring_counts
        self.movement_span = board_size * board_size * num_directions * self.max_distance
        self.special_base = self.placement_span + self.movement_span

        # Pre-compute hex validity mask
        self.register_buffer("hex_mask", create_hex_mask(hex_radius, board_size))

        # Pre-compute index scatter tensors for policy assembly
        self._register_policy_indices(board_size)

        # Shared backbone with SE blocks (same as V2)
        self.conv1 = nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU()
        self.res_blocks = nn.ModuleList(
            [SEResidualBlock(num_filters, reduction=se_reduction) for _ in range(num_res_blocks)]
        )

        # Multi-player value head with masked pooling (same as V2)
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 + global_features, value_intermediate)
        self.value_fc2 = nn.Linear(value_intermediate, num_players)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.3)

        # === V3 Spatial Policy Heads ===
        # Placement head: produces logits for each (cell, ring_count) tuple
        # Output shape: [B, 3, 25, 25] → indices [0, 1874]
        self.placement_conv = nn.Conv2d(num_filters, num_ring_counts, kernel_size=1)

        # Movement head: produces logits for each (cell, dir, dist) tuple
        # Output shape: [B, 144, 25, 25] → indices [1875, 91874]
        self.movement_conv = nn.Conv2d(num_filters, self.movement_channels, kernel_size=1)

        # Special actions head: small FC for skip_placement
        # Uses global pooled features → single logit
        self.special_fc = nn.Linear(num_filters + global_features, 1)

    def _register_policy_indices(self, board_size: int) -> None:
        """
        Pre-compute index tensors for scattering spatial logits into flat policy.

        Placement indexing: idx = y * W * 3 + x * 3 + ring_count
        Movement indexing: idx = placement_span + y * W * dirs * dists + x * dirs * dists + dir * dists + (dist - 1)
        """
        H, W = board_size, board_size

        # Placement indices: [3, H, W] → flat index in [0, placement_span)
        placement_idx = torch.zeros(self.num_ring_counts, H, W, dtype=torch.long)
        for y in range(H):
            for x in range(W):
                for r in range(self.num_ring_counts):
                    placement_idx[r, y, x] = y * W * self.num_ring_counts + x * self.num_ring_counts + r
        self.register_buffer("placement_idx", placement_idx)

        # Movement indices: [144, H, W] → flat index in [placement_span, special_base)
        movement_idx = torch.zeros(self.movement_channels, H, W, dtype=torch.long)
        movement_base = self.placement_span
        for y in range(H):
            for x in range(W):
                for d in range(self.num_directions):
                    for dist_minus_1 in range(self.max_distance):
                        channel = d * self.max_distance + dist_minus_1
                        flat_idx = (
                            movement_base
                            + y * W * self.num_directions * self.max_distance
                            + x * self.num_directions * self.max_distance
                            + d * self.max_distance
                            + dist_minus_1
                        )
                        movement_idx[channel, y, x] = flat_idx
        self.register_buffer("movement_idx", movement_idx)

    def _masked_global_avg_pool(
        self,
        x: torch.Tensor,
        hex_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Perform masked global average pooling over [H, W] for hex grid."""
        mask = hex_mask if hex_mask is not None else self.hex_mask
        if mask is None:
            return x.mean(dim=(2, 3))
        mask = mask.to(dtype=x.dtype, device=x.device)
        masked = x * mask
        num = masked.sum(dim=(2, 3))
        denom = mask.sum(dim=(2, 3)).clamp_min(1.0)
        return num / denom

    def _scatter_policy_logits(
        self,
        placement_logits: torch.Tensor,
        movement_logits: torch.Tensor,
        special_logits: torch.Tensor,
        hex_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Scatter spatial policy logits into flat P_HEX policy vector.

        Args:
            placement_logits: [B, 3, H, W] placement logits
            movement_logits: [B, 144, H, W] movement logits
            special_logits: [B, 1] special action logits
            hex_mask: Optional [1, H, W] validity mask

        Returns:
            policy_logits: [B, P_HEX] flat policy vector
        """
        B = placement_logits.size(0)
        device = placement_logits.device
        dtype = placement_logits.dtype

        # Initialize flat policy with large negative (will be masked anyway)
        # Use -1e4 instead of -1e9 to avoid float16 overflow in mixed precision
        policy = torch.full((B, self.policy_size), -1e4, device=device, dtype=dtype)

        # Flatten spatial dimensions for scatter
        # placement_logits: [B, 3, H, W] → [B, 3*H*W]
        placement_flat = placement_logits.view(B, -1)
        placement_idx_flat = self.placement_idx.view(-1).expand(B, -1)

        # movement_logits: [B, 144, H, W] → [B, 144*H*W]
        movement_flat = movement_logits.view(B, -1)
        movement_idx_flat = self.movement_idx.view(-1).expand(B, -1)

        # Scatter placement and movement logits
        policy.scatter_(1, placement_idx_flat, placement_flat)
        policy.scatter_(1, movement_idx_flat, movement_flat)

        # Add special action logit at dynamically computed special_base
        policy[:, self.special_base : self.special_base + 1] = special_logits

        return policy

    def forward(
        self,
        x: torch.Tensor,
        globals: torch.Tensor,
        hex_mask: torch.Tensor | None = None,
        return_features: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with spatial policy heads.

        Args:
            x: Input features [B, in_channels, H, W]
            globals: Global features [B, global_features]
            hex_mask: Optional validity mask [1, H, W]
            return_features: If True, also return backbone features for auxiliary tasks

        Returns:
            value: [B, num_players] per-player win probability
            policy: [B, P_HEX] flat policy logits
            features (optional): [B, feat_dim] pooled backbone features (if return_features=True)
        """
        # Apply hex mask to input to prevent information bleeding
        mask = hex_mask if hex_mask is not None else self.hex_mask
        if mask is not None:
            x = x * mask.to(dtype=x.dtype, device=x.device)

        # Backbone with SE blocks
        out = self.relu(self.bn1(self.conv1(x)))
        for block in self.res_blocks:
            out = block(out)

        # === Value Head (same as V2) ===
        v = self.value_conv(out)
        v = self.relu(self.value_bn(v))
        v_pooled = self._masked_global_avg_pool(v, hex_mask)
        v_cat = torch.cat([v_pooled, globals], dim=1)
        v_hidden = self.relu(self.value_fc1(v_cat))
        v_hidden = self.dropout(v_hidden)
        v_out = self.tanh(self.value_fc2(v_hidden))  # [B, num_players]

        # === Spatial Policy Heads (V3) ===
        # Placement logits: [B, 3, H, W]
        placement_logits = self.placement_conv(out)

        # Movement logits: [B, 144, H, W]
        movement_logits = self.movement_conv(out)

        # Apply hex mask to spatial logits (invalid cells get -inf)
        if mask is not None:
            mask_expanded = mask.to(dtype=out.dtype, device=out.device)
            # Broadcast mask to all channels
            placement_logits = placement_logits * mask_expanded + (1.0 - mask_expanded) * (-1e9)
            movement_logits = movement_logits * mask_expanded + (1.0 - mask_expanded) * (-1e9)

        # Special action logits from pooled features
        out_pooled = self._masked_global_avg_pool(out, hex_mask)
        special_input = torch.cat([out_pooled, globals], dim=1)
        special_logits = self.special_fc(special_input)  # [B, 1]

        # Scatter into flat policy vector
        policy_logits = self._scatter_policy_logits(placement_logits, movement_logits, special_logits, hex_mask)

        if return_features:
            return v_out, policy_logits, v_cat

        return v_out, policy_logits


class HexNeuralNet_v3_Lite(nn.Module):
    """
    Memory-efficient V3 architecture with spatial policy heads (48GB target).

    Same spatial policy head design as HexNeuralNet_v3 but with reduced capacity:
    - 6 SE residual blocks (vs 12)
    - 96 filters (vs 192)
    - 3 history frames (vs 4)
    - 12 base input channels (vs 14)

    Architecture Version:
        v3.0.0-lite - Spatial policy heads, reduced capacity for 48GB systems.
    """

    ARCHITECTURE_VERSION = "v3.0.0-lite"

    def __init__(
        self,
        in_channels: int = 44,  # 12 base × 3 frames + 8 phase/chain planes
        global_features: int = 20,
        num_res_blocks: int = 6,
        num_filters: int = 96,
        board_size: int = HEX_BOARD_SIZE,
        policy_size: int = P_HEX,
        value_intermediate: int = 64,
        num_players: int = 4,
        se_reduction: int = 8,
        hex_radius: int = 12,
        num_ring_counts: int = 3,
        num_directions: int = NUM_HEX_DIRS,
        max_distance: int | None = None,  # Computed from board_size if None
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.global_features = global_features
        self.num_res_blocks = num_res_blocks
        self.num_filters = num_filters
        self.board_size = board_size
        self.policy_size = policy_size
        self.num_players = num_players

        # Spatial policy dimensions
        self.num_ring_counts = num_ring_counts
        self.num_directions = num_directions
        # max_distance = board_size - 1: hex8 (9x9) uses 8, hexagonal (25x25) uses 24
        self.max_distance = max_distance if max_distance is not None else board_size - 1
        self.movement_channels = num_directions * self.max_distance

        # Compute layout spans dynamically based on board_size
        # This ensures hex8 (9x9) and hexagonal (25x25) use correct indices
        self.placement_span = board_size * board_size * num_ring_counts
        self.movement_span = board_size * board_size * num_directions * self.max_distance
        self.special_base = self.placement_span + self.movement_span

        # Pre-compute hex validity mask
        self.register_buffer("hex_mask", create_hex_mask(hex_radius, board_size))

        # Pre-compute index scatter tensors
        self._register_policy_indices(board_size)

        # Shared backbone with SE blocks
        self.conv1 = nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU()
        self.res_blocks = nn.ModuleList(
            [SEResidualBlock(num_filters, reduction=se_reduction) for _ in range(num_res_blocks)]
        )

        # Multi-player value head
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 + global_features, value_intermediate)
        self.value_fc2 = nn.Linear(value_intermediate, num_players)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.3)

        # Spatial policy heads
        self.placement_conv = nn.Conv2d(num_filters, num_ring_counts, kernel_size=1)
        self.movement_conv = nn.Conv2d(num_filters, self.movement_channels, kernel_size=1)
        self.special_fc = nn.Linear(num_filters + global_features, 1)

    def _register_policy_indices(self, board_size: int) -> None:
        """Pre-compute index tensors for scattering spatial logits."""
        H, W = board_size, board_size

        placement_idx = torch.zeros(self.num_ring_counts, H, W, dtype=torch.long)
        for y in range(H):
            for x in range(W):
                for r in range(self.num_ring_counts):
                    placement_idx[r, y, x] = y * W * self.num_ring_counts + x * self.num_ring_counts + r
        self.register_buffer("placement_idx", placement_idx)

        movement_idx = torch.zeros(self.movement_channels, H, W, dtype=torch.long)
        movement_base = self.placement_span
        for y in range(H):
            for x in range(W):
                for d in range(self.num_directions):
                    for dist_minus_1 in range(self.max_distance):
                        channel = d * self.max_distance + dist_minus_1
                        flat_idx = (
                            movement_base
                            + y * W * self.num_directions * self.max_distance
                            + x * self.num_directions * self.max_distance
                            + d * self.max_distance
                            + dist_minus_1
                        )
                        movement_idx[channel, y, x] = flat_idx
        self.register_buffer("movement_idx", movement_idx)

    def _masked_global_avg_pool(
        self,
        x: torch.Tensor,
        hex_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Perform masked global average pooling over [H, W] for hex grid."""
        mask = hex_mask if hex_mask is not None else self.hex_mask
        if mask is None:
            return x.mean(dim=(2, 3))
        mask = mask.to(dtype=x.dtype, device=x.device)
        masked = x * mask
        num = masked.sum(dim=(2, 3))
        denom = mask.sum(dim=(2, 3)).clamp_min(1.0)
        return num / denom

    def _scatter_policy_logits(
        self,
        placement_logits: torch.Tensor,
        movement_logits: torch.Tensor,
        special_logits: torch.Tensor,
        hex_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Scatter spatial policy logits into flat P_HEX policy vector."""
        B = placement_logits.size(0)
        device = placement_logits.device
        dtype = placement_logits.dtype

        # Use -1e4 instead of -1e9 to avoid float16 overflow in mixed precision
        policy = torch.full((B, self.policy_size), -1e4, device=device, dtype=dtype)

        placement_flat = placement_logits.view(B, -1)
        placement_idx_flat = self.placement_idx.view(-1).expand(B, -1)

        movement_flat = movement_logits.view(B, -1)
        movement_idx_flat = self.movement_idx.view(-1).expand(B, -1)

        policy.scatter_(1, placement_idx_flat, placement_flat)
        policy.scatter_(1, movement_idx_flat, movement_flat)
        policy[:, self.special_base : self.special_base + 1] = special_logits

        return policy

    def forward(
        self,
        x: torch.Tensor,
        globals: torch.Tensor,
        hex_mask: torch.Tensor | None = None,
        return_features: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with spatial policy heads.

        Args:
            x: Input features [B, in_channels, H, W]
            globals: Global features [B, global_features]
            hex_mask: Optional validity mask [1, H, W]
            return_features: If True, also return backbone features for auxiliary tasks

        Returns:
            value: [B, num_players] per-player win probability
            policy: [B, P_HEX] flat policy logits
            features (optional): [B, feat_dim] pooled backbone features (if return_features=True)
        """
        mask = hex_mask if hex_mask is not None else self.hex_mask
        if mask is not None:
            x = x * mask.to(dtype=x.dtype, device=x.device)

        out = self.relu(self.bn1(self.conv1(x)))
        for block in self.res_blocks:
            out = block(out)

        # Value head
        v = self.value_conv(out)
        v = self.relu(self.value_bn(v))
        v_pooled = self._masked_global_avg_pool(v, hex_mask)
        v_cat = torch.cat([v_pooled, globals], dim=1)
        v_hidden = self.relu(self.value_fc1(v_cat))
        v_hidden = self.dropout(v_hidden)
        v_out = self.tanh(self.value_fc2(v_hidden))

        # Spatial policy heads
        placement_logits = self.placement_conv(out)
        movement_logits = self.movement_conv(out)

        if mask is not None:
            mask_expanded = mask.to(dtype=out.dtype, device=out.device)
            placement_logits = placement_logits * mask_expanded + (1.0 - mask_expanded) * (-1e9)
            movement_logits = movement_logits * mask_expanded + (1.0 - mask_expanded) * (-1e9)

        out_pooled = self._masked_global_avg_pool(out, hex_mask)
        special_input = torch.cat([out_pooled, globals], dim=1)
        special_logits = self.special_fc(special_input)

        policy_logits = self._scatter_policy_logits(placement_logits, movement_logits, special_logits, hex_mask)

        if return_features:
            return v_out, policy_logits, v_cat

        return v_out, policy_logits
