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

from .blocks import AttentionResidualBlock, SEResidualBlock, create_hex_mask
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
        # Input validation: fail fast on channel mismatch (Dec 2025)
        if x.shape[1] != self.in_channels:
            raise RuntimeError(
                f"Input channel mismatch in {self.__class__.__name__}.forward():\n"
                f"  Input has {x.shape[1]} channels\n"
                f"  Model expects {self.in_channels} channels\n"
                f"  This indicates encoder/model version mismatch.\n"
                f"  Check that data was exported with matching encoder version."
            )

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
        # Input validation: fail fast on channel mismatch (Dec 2025)
        if x.shape[1] != self.in_channels:
            raise RuntimeError(
                f"Input channel mismatch in {self.__class__.__name__}.forward():\n"
                f"  Input has {x.shape[1]} channels\n"
                f"  Model expects {self.in_channels} channels\n"
                f"  This indicates encoder/model version mismatch.\n"
                f"  Check that data was exported with matching encoder version."
            )

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

    ⚠️  WARNING: SPATIAL POLICY HEADS CAUSE LOSS EXPLOSION  ⚠️

    This architecture has a critical training bug that causes ~42 policy loss
    (expected ~2-5) and 40% win rate vs random opponents. The issue is that
    -1e9 masking of invalid cells creates log_softmax numerical instability.

    USE HexNeuralNet_v3_Flat INSTEAD for training. This class is preserved for:
    - Debugging and analysis of the spatial head issue
    - Loading existing V3 spatial checkpoints for inference

    Root cause:
    When invalid hex cells are masked to -1e9 and scattered into the flat policy
    vector, the ~90,000 masked entries dominate the softmax denominator, causing:
    - Valid actions to get log-prob ≈ -25
    - KL loss = 800 valid actions × 25 = 20,000+ loss per sample

    Original design intent (not working as intended):
    - Spatial placement head: Conv1×1 → [B, 3, H, W] for (cell, ring_count) logits
    - Spatial movement head: Conv1×1 → [B, 144, H, W] for (cell, dir, dist) logits
    - Small FC for special actions (skip_placement only)
    - Logits are scattered into canonical P_HEX=91,876 flat policy vector

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
        policy_size: int | None = None,  # Computed dynamically if None
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

        # Compute policy_size dynamically if not provided
        # hex8 (board_size=9): 9*9*3 + 9*9*6*8 + 1 = 243 + 3888 + 1 = 4132
        # hexagonal (board_size=25): 25*25*3 + 25*25*6*24 + 1 = 1875 + 90000 + 1 = 91876
        if policy_size is None:
            self.policy_size = self.special_base + 1  # +1 for special action
        else:
            self.policy_size = policy_size

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

        # Initialize flat policy with -1e9 to mask padding positions.
        # This ensures positions not covered by scatter (e.g., policy_size padding)
        # don't pollute the softmax.
        # NOTE: Use FP32 for initialization since -1e9 exceeds FP16 range (±65504).
        # We'll cast to the input dtype after scatter operations.
        policy = torch.full((B, self.policy_size), -1e9, device=device, dtype=torch.float32)

        # Flatten spatial dimensions for scatter
        # placement_logits: [B, 3, H, W] → [B, 3*H*W]
        # Cast to FP32 to match policy tensor dtype
        placement_flat = placement_logits.float().view(B, -1)
        placement_idx_flat = self.placement_idx.view(-1).expand(B, -1)

        # movement_logits: [B, 144, H, W] → [B, 144*H*W]
        movement_flat = movement_logits.float().view(B, -1)
        movement_idx_flat = self.movement_idx.view(-1).expand(B, -1)

        # Scatter placement and movement logits
        policy.scatter_(1, placement_idx_flat, placement_flat)
        policy.scatter_(1, movement_idx_flat, movement_flat)

        # Add special action logit at dynamically computed special_base
        policy[:, self.special_base : self.special_base + 1] = special_logits.float()

        # Cast back to input dtype for downstream operations
        return policy.to(dtype)

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
        # Input validation: fail fast on channel mismatch (Dec 2025)
        if x.shape[1] != self.in_channels:
            raise RuntimeError(
                f"Input channel mismatch in {self.__class__.__name__}.forward():\n"
                f"  Input has {x.shape[1]} channels\n"
                f"  Model expects {self.in_channels} channels\n"
                f"  This indicates encoder/model version mismatch.\n"
                f"  Check that data was exported with matching encoder version."
            )

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

        # Apply hex mask ALWAYS (training and inference).
        # Invalid hex corners must have large negative logits so they don't pollute
        # the softmax denominator. The encoder ensures targets only reference
        # valid hex positions, so masking during training is safe.
        # (Fixed Jan 2026 - previous Dec 2025 fix was backwards)
        # NOTE: Compute masking in FP32 since -1e9 exceeds FP16 range (±65504).
        # Keep logits in FP32 - the scatter function handles dtype internally.
        if mask is not None:
            mask_expanded = mask.to(dtype=torch.float32, device=out.device)
            placement_logits = placement_logits.float() * mask_expanded + (1.0 - mask_expanded) * (-1e9)
            movement_logits = movement_logits.float() * mask_expanded + (1.0 - mask_expanded) * (-1e9)

        # Special action logits from pooled features
        out_pooled = self._masked_global_avg_pool(out, hex_mask)
        special_input = torch.cat([out_pooled, globals], dim=1)
        special_logits = self.special_fc(special_input)  # [B, 1]

        # Scatter into flat policy vector
        policy_logits = self._scatter_policy_logits(placement_logits, movement_logits, special_logits, hex_mask)

        # Phase 2: Output validation during training to catch numerical issues early
        if self.training:
            # Check for NaN/Inf in policy outputs
            if torch.any(torch.isnan(policy_logits)) or torch.any(torch.isinf(policy_logits)):
                raise RuntimeError(
                    f"NaN/Inf detected in policy_logits during forward pass. "
                    f"Check backbone weights and input normalization. "
                    f"NaNs: {torch.isnan(policy_logits).sum().item()}, "
                    f"Infs: {torch.isinf(policy_logits).sum().item()}"
                )

            # Check for extreme logit values (excluding -1e9 masking)
            valid_mask = policy_logits > -1e8  # -1e9 is intentional invalid cell masking
            if torch.any(valid_mask):
                valid_logits = policy_logits[valid_mask]
                max_abs = valid_logits.abs().max().item()
                if max_abs > 1e6:
                    import warnings
                    warnings.warn(
                        f"Extreme policy logits in HexNeuralNet_v3 forward: "
                        f"max_abs={max_abs:.2e}. This may cause numerical issues in loss."
                    )

        if return_features:
            # Return backbone features (pooled), not value head features (v_cat)
            # out_pooled has shape [B, num_filters] which is expected by auxiliary tasks
            return v_out, policy_logits, out_pooled

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
        policy_size: int | None = None,  # Computed dynamically if None
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

        # Compute policy_size dynamically if not provided
        if policy_size is None:
            self.policy_size = self.special_base + 1  # +1 for special action
        else:
            self.policy_size = policy_size

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

        # Initialize with -1e9 to properly mask padding positions
        # NOTE: Use FP32 for initialization since -1e9 exceeds FP16 range (±65504).
        policy = torch.full((B, self.policy_size), -1e9, device=device, dtype=torch.float32)

        # Cast to FP32 to match policy tensor dtype
        placement_flat = placement_logits.float().view(B, -1)
        placement_idx_flat = self.placement_idx.view(-1).expand(B, -1)

        movement_flat = movement_logits.float().view(B, -1)
        movement_idx_flat = self.movement_idx.view(-1).expand(B, -1)

        policy.scatter_(1, placement_idx_flat, placement_flat)
        policy.scatter_(1, movement_idx_flat, movement_flat)
        policy[:, self.special_base : self.special_base + 1] = special_logits.float()

        # Cast back to input dtype for downstream operations
        return policy.to(dtype)

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

        # Apply hex mask ALWAYS (training and inference).
        # Invalid hex corners must have large negative logits so they don't pollute
        # the softmax denominator. The encoder ensures targets only reference
        # valid hex positions, so masking during training is safe.
        # (Fixed Jan 2026 - previous Dec 2025 fix was backwards)
        # NOTE: Compute masking in FP32 since -1e9 exceeds FP16 range (±65504).
        # Keep logits in FP32 - the scatter function handles dtype internally.
        if mask is not None:
            mask_expanded = mask.to(dtype=torch.float32, device=out.device)
            placement_logits = placement_logits.float() * mask_expanded + (1.0 - mask_expanded) * (-1e9)
            movement_logits = movement_logits.float() * mask_expanded + (1.0 - mask_expanded) * (-1e9)

        out_pooled = self._masked_global_avg_pool(out, hex_mask)
        special_input = torch.cat([out_pooled, globals], dim=1)
        special_logits = self.special_fc(special_input)

        policy_logits = self._scatter_policy_logits(placement_logits, movement_logits, special_logits, hex_mask)

        if return_features:
            # Return backbone features (pooled), not value head features (v_cat)
            # out_pooled has shape [B, num_filters] which is expected by auxiliary tasks
            return v_out, policy_logits, out_pooled

        return v_out, policy_logits


class HexNeuralNet_v3_Flat(nn.Module):
    """
    V3 backbone with flat policy heads - DEFAULT V3 ARCHITECTURE (Jan 2026).

    This architecture combines the SE backbone improvements from V3 with the
    flat policy head design from V2, avoiding the spatial policy head issues
    that cause loss explosion during training.

    This is now the default when using --model-version v3 because the spatial
    variant (HexNeuralNet_v3) has a critical training bug causing ~42 policy
    loss and 40% win rate vs random opponents.

    Why this class exists:
    V3's spatial policy heads mask invalid hex cells to -1e9, but when these
    values are scattered into the flat policy vector and log_softmax is applied,
    the ~90,000 masked entries dominate the softmax denominator, causing:
    - Valid actions to get log-prob ≈ -25
    - KL loss = 800 valid actions × 25 = 20,000+ loss per sample

    Solution:
    V3_Flat uses the same backbone as V3 but outputs policy through a standard
    FC layer like V2. This produces unbounded logits that work correctly with
    the standard cross-entropy loss where invalid actions are masked in the
    target distribution, not the model output.

    Key characteristics:
    - SE backbone from V3 (12 blocks, 192 filters, 64 input channels)
    - Flat policy head from V2 (global avg pool → FC → policy_size)
    - Compatible with standard training pipeline
    - No spatial action structure preserved (tradeoff for trainability)

    CLI usage:
    - --model-version v3        → Uses this class (flat, stable)
    - --model-version v3-flat   → Uses this class (alias for v3)
    - --model-version v3-spatial → Uses HexNeuralNet_v3 (broken, for debugging)

    Architecture Version:
        v3.1.0 - V3 backbone with flat policy heads for training compatibility.
    """

    ARCHITECTURE_VERSION = "v3.1.0-flat"

    def __init__(
        self,
        in_channels: int = 64,  # 16 base × 4 frames for V3 encoder
        global_features: int = 20,
        num_res_blocks: int = 12,
        num_filters: int = 192,
        board_size: int = HEX_BOARD_SIZE,
        policy_size: int = P_HEX,
        policy_intermediate: int = 384,  # V2-style policy intermediate
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

        # Shared backbone with SE blocks (same as V3)
        self.conv1 = nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU()
        self.res_blocks = nn.ModuleList(
            [SEResidualBlock(num_filters, reduction=se_reduction) for _ in range(num_res_blocks)]
        )

        # Multi-player value head with masked pooling (same as V3)
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 + global_features, value_intermediate)
        self.value_fc2 = nn.Linear(value_intermediate, num_players)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.3)

        # FLAT Policy head (V2-style, avoids spatial masking issues)
        self.policy_conv = nn.Conv2d(num_filters, num_filters, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(num_filters)
        self.policy_fc1 = nn.Linear(num_filters + global_features, policy_intermediate)
        self.policy_fc2 = nn.Linear(policy_intermediate, policy_size)

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
        """
        Forward pass with flat policy head (V2-style).

        Args:
            x: Input features [B, in_channels, H, W]
            globals: Global features [B, global_features]
            hex_mask: Optional validity mask [1, H, W]
            return_features: If True, also return backbone features

        Returns:
            value: [B, num_players] per-player win probability
            policy: [B, policy_size] flat policy logits (NO -1e9 masking)
            features (optional): [B, feat_dim] pooled backbone features
        """
        # Input validation
        if x.shape[1] != self.in_channels:
            raise RuntimeError(
                f"Input channel mismatch in {self.__class__.__name__}.forward():\n"
                f"  Input has {x.shape[1]} channels\n"
                f"  Model expects {self.in_channels} channels"
            )

        # Apply hex mask to input
        mask = hex_mask if hex_mask is not None else self.hex_mask
        if mask is not None:
            x = x * mask.to(dtype=x.dtype, device=x.device)

        # Backbone with SE blocks
        out = self.relu(self.bn1(self.conv1(x)))
        for block in self.res_blocks:
            out = block(out)

        # === Value Head (same as V3) ===
        v = self.value_conv(out)
        v = self.relu(self.value_bn(v))
        v_pooled = self._masked_global_avg_pool(v, hex_mask)
        v_cat = torch.cat([v_pooled, globals], dim=1)
        v_hidden = self.relu(self.value_fc1(v_cat))
        v_hidden = self.dropout(v_hidden)
        v_out = self.tanh(self.value_fc2(v_hidden))

        # === FLAT Policy Head (V2-style) ===
        # This avoids the spatial masking → scatter → log_softmax issue
        p = self.policy_conv(out)
        p = self.relu(self.policy_bn(p))
        p_pooled = self._masked_global_avg_pool(p, hex_mask)
        p_cat = torch.cat([p_pooled, globals], dim=1)
        p_hidden = self.relu(self.policy_fc1(p_cat))
        p_hidden = self.dropout(p_hidden)
        policy_logits = self.policy_fc2(p_hidden)

        if return_features:
            out_pooled = self._masked_global_avg_pool(out, hex_mask)
            return v_out, policy_logits, out_pooled

        return v_out, policy_logits


class HexNeuralNet_v4(nn.Module):
    """
    V4 architecture with NAS-optimized attention for hexagonal boards.

    This architecture applies the NAS-discovered improvements from RingRiftCNN_v4
    to the hexagonal board architecture, combining the spatial policy heads
    from V3 with the optimal structural choices found by evolutionary NAS.

    NAS-Discovered Improvements:
    - Multi-head self-attention (4 heads) instead of SE blocks
    - 13 residual blocks (vs 12 in v3)
    - 128 filters (vs 192 in v3, more efficient)
    - 5x5 initial kernel (vs 3x3, better spatial coverage)
    - Deeper value head (3 layers vs 2)
    - Lower dropout (0.08 vs 0.3)
    - Rank distribution head for multi-player games

    Preserved from V3:
    - Spatial policy heads (placement, movement, special)
    - Hex-specific masking and pooling
    - Game-specific action encoding

    Architecture Version:
        v4.0.0 - NAS-optimized attention architecture for hex boards.

    Performance Characteristics:
    - Slightly fewer parameters than v3 (128 vs 192 filters)
    - Better long-range pattern recognition (attention)
    - Improved training efficiency (deeper value head)
    """

    ARCHITECTURE_VERSION = "v4.0.0"

    def __init__(
        self,
        in_channels: int = 64,  # 16 base × 4 frames for V3 encoder
        global_features: int = 20,  # V3 encoder provides 20 global features
        num_res_blocks: int = 13,  # NAS optimal
        num_filters: int = 128,  # NAS optimal
        board_size: int = HEX_BOARD_SIZE,
        policy_size: int | None = None,  # Compute dynamically if not provided
        value_intermediate: int = 256,  # NAS optimal
        value_hidden: int = 256,  # NAS: deeper value head
        num_players: int = 4,
        num_attention_heads: int = 4,  # NAS optimal
        dropout: float = 0.08,  # NAS optimal
        initial_kernel_size: int = 5,  # NAS optimal
        hex_radius: int | None = None,  # Infer from board_size if not provided
        num_ring_counts: int = 3,  # Ring count options (1, 2, 3)
        num_directions: int = NUM_HEX_DIRS,  # 6 hex directions
        max_distance: int | None = None,  # Auto-detect based on board_size
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.global_features = global_features
        self.num_res_blocks = num_res_blocks
        self.num_filters = num_filters
        self.board_size = board_size
        self.num_players = num_players
        self.dropout_rate = dropout

        # Auto-detect hex_radius from board_size if not provided
        if hex_radius is None:
            hex_radius = (board_size - 1) // 2

        # Auto-detect max_distance based on board size:
        # - hex8 (board_size=9): max_distance = 8
        # - hexagonal (board_size=25): max_distance = 24
        if max_distance is None:
            max_distance = board_size - 1
        self.max_distance = max_distance

        # Spatial policy dimensions
        self.num_ring_counts = num_ring_counts
        self.num_directions = num_directions
        self.movement_channels = num_directions * max_distance

        # Compute policy layout spans dynamically (like V3)
        # This ensures correct policy_size for any board_size
        self.placement_span = board_size * board_size * num_ring_counts
        self.movement_span = board_size * board_size * num_directions * max_distance
        self.special_base = self.placement_span + self.movement_span

        # Compute policy_size dynamically if not provided
        if policy_size is None:
            policy_size = self.special_base + 1
        self.policy_size = policy_size

        # Validate policy_size is sufficient for computed spans
        required_policy_size = self.special_base + 1
        if self.policy_size < required_policy_size:
            raise ValueError(
                f"policy_size={self.policy_size} too small for board_size={board_size}. "
                f"Required: {required_policy_size} (placement={self.placement_span}, "
                f"movement={self.movement_span}, special=1). "
                f"This indicates a mismatch between board_size and policy_size parameters."
            )

        # Pre-compute hex validity mask
        self.register_buffer("hex_mask", create_hex_mask(hex_radius, board_size))

        # Pre-compute index scatter tensors for policy assembly
        self._register_policy_indices(board_size)

        # Initial convolution with larger kernel (NAS optimal: 5x5)
        self.conv1 = nn.Conv2d(
            in_channels,
            num_filters,
            kernel_size=initial_kernel_size,
            padding=initial_kernel_size // 2,
        )
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU()

        # Attention-enhanced residual blocks (NAS optimal)
        self.res_blocks = nn.ModuleList([
            AttentionResidualBlock(
                num_filters,
                num_heads=num_attention_heads,
                dropout=dropout,
            )
            for _ in range(num_res_blocks)
        ])

        # === Deeper Value Head (NAS optimal: 3 layers) ===
        self.value_fc1 = nn.Linear(num_filters + global_features, value_intermediate)
        self.value_fc2 = nn.Linear(value_intermediate, value_hidden)
        self.value_fc3 = nn.Linear(value_hidden, num_players)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

        # === Rank Distribution Head ===
        rank_dist_intermediate = value_intermediate
        self.rank_dist_fc1 = nn.Linear(num_filters + global_features, rank_dist_intermediate)
        self.rank_dist_fc2 = nn.Linear(rank_dist_intermediate, value_hidden)
        self.rank_dist_fc3 = nn.Linear(value_hidden, num_players * num_players)
        self.rank_softmax = nn.Softmax(dim=-1)

        # === V3 Spatial Policy Heads ===
        self.placement_conv = nn.Conv2d(num_filters, num_ring_counts, kernel_size=1)
        self.movement_conv = nn.Conv2d(num_filters, self.movement_channels, kernel_size=1)
        self.special_fc = nn.Linear(num_filters + global_features, 1)

    def _register_policy_indices(self, board_size: int) -> None:
        """
        Pre-compute index tensors for scattering spatial logits into flat policy.

        Hex8 (board_size=9):
          - Placement: 9 × 9 × 3 = 243
          - Movement base: 243
          - Movement: 9 × 9 × 6 × 8 = 3888
        Hexagonal (board_size=25):
          - Placement: 25 × 25 × 3 = 1875
          - Movement base: 1875 (HEX_MOVEMENT_BASE)
          - Movement: 25 × 25 × 6 × 24 = 90000
        """
        H, W = board_size, board_size

        # Compute placement span based on actual board size
        placement_span = H * W * self.num_ring_counts

        # Placement indices: [3, H, W] → flat index in [0, placement_span-1]
        placement_idx = torch.zeros(self.num_ring_counts, H, W, dtype=torch.long)
        for y in range(H):
            for x in range(W):
                for r in range(self.num_ring_counts):
                    placement_idx[r, y, x] = y * W * self.num_ring_counts + x * self.num_ring_counts + r
        self.register_buffer("placement_idx", placement_idx)

        # Movement indices: [movement_channels, H, W] → flat index in [placement_span, ...]
        movement_idx = torch.zeros(self.movement_channels, H, W, dtype=torch.long)
        for y in range(H):
            for x in range(W):
                for d in range(self.num_directions):
                    for dist_minus_1 in range(self.max_distance):
                        channel = d * self.max_distance + dist_minus_1
                        flat_idx = (
                            placement_span  # Movement base = after placements
                            + y * W * self.num_directions * self.max_distance
                            + x * self.num_directions * self.max_distance
                            + d * self.max_distance
                            + dist_minus_1
                        )
                        movement_idx[channel, y, x] = flat_idx
        self.register_buffer("movement_idx", movement_idx)

        # Store special base index for use in forward pass
        movement_span = H * W * self.num_directions * self.max_distance
        self.special_base = placement_span + movement_span

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
        Scatter spatial logits into flat policy vector using pre-computed indices.

        Args:
            placement_logits: [B, 3, H, W] placement logits
            movement_logits: [B, movement_channels, H, W] movement logits
            special_logits: [B, 1] special action logits
            hex_mask: [1, 1, H, W] hex validity mask

        Returns:
            policy: [B, P_HEX] flat policy logits
        """
        B = placement_logits.size(0)
        device = placement_logits.device
        dtype = placement_logits.dtype

        # Initialize policy with large negative (masked out)
        # NOTE: Use FP32 for initialization since -1e9 exceeds FP16 range (±65504).
        policy = torch.full((B, self.policy_size), -1e9, device=device, dtype=torch.float32)

        # Scatter placement logits: [B, 3, H, W] → [B, placement_span]
        # NOTE: We scatter ALL indices (including padding cells) to ensure v3-encoded
        # targets always have valid logits. The training loss will mask invalid moves.
        # VECTORIZED: Single scatter call instead of B*3 Python loop iterations
        # Cast to FP32 to match policy tensor dtype
        pl_values = placement_logits.float().view(B, -1)  # [B, 3*H*W]
        pl_idx_flat = self.placement_idx.reshape(-1)  # [3*H*W]
        pl_idx_expanded = pl_idx_flat.unsqueeze(0).expand(B, -1)  # [B, 3*H*W]
        policy.scatter_(1, pl_idx_expanded, pl_values)

        # Scatter movement logits: [B, movement_channels, H, W] → [B, movement_span]
        # VECTORIZED: Single scatter call instead of B*movement_channels Python loop iterations
        mv_values = movement_logits.float().view(B, -1)  # [B, C*H*W]
        mv_idx_flat = self.movement_idx.reshape(-1)  # [C*H*W]
        mv_idx_expanded = mv_idx_flat.unsqueeze(0).expand(B, -1)  # [B, C*H*W]
        policy.scatter_(1, mv_idx_expanded, mv_values)

        # Add special action logit at the computed special_base index
        policy[:, self.special_base] = special_logits.float().squeeze(-1)

        # Cast back to input dtype for downstream operations
        return policy.to(dtype)

    def forward(
        self,
        x: torch.Tensor,
        globals: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_features: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with spatial policy heads and NAS-optimized backbone.

        Args:
            x: [B, C, H, W] spatial features
            globals: [B, G] global features
            mask: [B, 1, H, W] optional action mask (for masking invalid cells)
            return_features: if True, also return intermediate features for auxiliary tasks

        Returns:
            v_out: [B, num_players] value predictions
            policy_logits: [B, P_HEX] policy logits
            features: (optional) [B, num_filters] intermediate features if return_features=True
        """
        hex_mask = self.hex_mask if mask is None else mask

        # Initial convolution with 5x5 kernel
        out = self.conv1(x)
        out = self.relu(self.bn1(out))

        # Attention residual blocks
        for block in self.res_blocks:
            out = block(out)

        # Global pooled features for heads
        out_pooled = self._masked_global_avg_pool(out, hex_mask)
        combined = torch.cat([out_pooled, globals], dim=1)

        # Deeper value head (3 layers)
        v_hidden = self.relu(self.value_fc1(combined))
        v_hidden = self.dropout(v_hidden)
        v_hidden = self.relu(self.value_fc2(v_hidden))
        v_hidden = self.dropout(v_hidden)
        v_out = self.tanh(self.value_fc3(v_hidden))

        # Spatial policy heads
        placement_logits = self.placement_conv(out)
        movement_logits = self.movement_conv(out)

        # Apply hex mask ALWAYS (training and inference).
        # Invalid hex corners must have large negative logits so they don't pollute
        # the softmax denominator. The encoder's _is_valid_hex_cell() ensures training
        # targets only have probability on valid hex positions, so this is safe.
        # (Fixed Jan 2026 - previous Dec 2025 fix was backwards)
        # NOTE: Compute masking in FP32 since -1e9 exceeds FP16 range (±65504).
        # Keep logits in FP32 - the scatter function handles dtype internally.
        if mask is not None:
            mask_expanded = mask.to(dtype=torch.float32, device=out.device)
            placement_logits = placement_logits.float() * mask_expanded + (1.0 - mask_expanded) * (-1e9)
            movement_logits = movement_logits.float() * mask_expanded + (1.0 - mask_expanded) * (-1e9)

        special_input = torch.cat([out_pooled, globals], dim=1)
        special_logits = self.special_fc(special_input)

        policy_logits = self._scatter_policy_logits(placement_logits, movement_logits, special_logits, hex_mask)

        if return_features:
            return v_out, policy_logits, out_pooled

        return v_out, policy_logits
