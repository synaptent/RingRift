"""Square board CNN architectures for RingRift AI.

This module contains the CNN architecture classes for square boards (8x8 and 19x19):
- RingRiftCNN_v2: High-capacity SE architecture (96GB systems)
- RingRiftCNN_v2_Lite: Memory-efficient SE architecture (48GB systems)
- RingRiftCNN_v3: Spatial policy heads with SE backbone
- RingRiftCNN_v3_Lite: Memory-efficient spatial policy heads
- RingRiftCNN_v4: NAS-discovered attention architecture

Migrated from _neural_net_legacy.py as part of Phase 2 modularization.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .blocks import AttentionResidualBlock, SEResidualBlock
from .constants import (
    MAX_DIST_SQUARE8,
    MAX_DIST_SQUARE19,
    NUM_LINE_DIRS,
    NUM_SQUARE_DIRS,
    POLICY_SIZE,
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
    POLICY_SIZE_8x8,
    POLICY_SIZE_19x19,
)


class RingRiftCNN_v2(nn.Module):
    """
    High-capacity CNN for 19x19 square boards (96GB memory target).

    This architecture is designed for maximum playing strength on systems
    with sufficient memory (96GB+) to run two instances simultaneously
    for comparison matches with MCTS search overhead.

    Key improvements over RingRiftCNN_MPS:
    - 12 SE residual blocks with Squeeze-and-Excitation for global patterns
    - 192 filters for richer representations
    - 14 base input channels capturing stack/cap/territory mechanics
    - 20 global features for multi-player state tracking
    - Multi-player value head (outputs per-player win probability)
    - 384-dim policy intermediate for better move discrimination

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
    - Model weights: ~150 MB
    - Per-model with activations: ~350 MB
    - Two models + MCTS: ~18 GB total

    Architecture Version:
        v2.0.0 - High-capacity SE architecture for 96GB systems.
    """

    ARCHITECTURE_VERSION = "v2.0.0"

    def __init__(
        self,
        board_size: int = 19,
        in_channels: int = 14,
        global_features: int = 20,
        num_res_blocks: int = 12,
        num_filters: int = 192,
        history_length: int = 3,
        policy_size: int | None = None,
        policy_intermediate: int = 384,
        value_intermediate: int = 128,
        num_players: int = 4,
        se_reduction: int = 16,
    ):
        super().__init__()
        self.board_size = board_size
        self.num_filters = num_filters
        self.num_players = num_players
        self.global_features = global_features

        # Input channels = base_channels * (history_length + 1)
        self.total_in_channels = in_channels * (history_length + 1)
        self.in_channels = self.total_in_channels  # For forward() validation

        # Initial convolution
        self.conv1 = nn.Conv2d(self.total_in_channels, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU()

        # SE-enhanced residual blocks for global pattern recognition
        self.res_blocks = nn.ModuleList(
            [SEResidualBlock(num_filters, reduction=se_reduction) for _ in range(num_res_blocks)]
        )

        # Multi-player value head (outputs per-player win probability)
        self.value_fc1 = nn.Linear(num_filters + global_features, value_intermediate)
        self.value_fc2 = nn.Linear(value_intermediate, num_players)
        self.tanh = nn.Tanh()

        # Policy head with larger intermediate
        self.policy_fc1 = nn.Linear(num_filters + global_features, policy_intermediate)
        if policy_size is not None:
            self.policy_size = policy_size
        elif board_size == 8:
            self.policy_size = POLICY_SIZE_8x8
        elif board_size == 19:
            self.policy_size = POLICY_SIZE_19x19
        else:
            self.policy_size = POLICY_SIZE
        self.policy_fc2 = nn.Linear(policy_intermediate, self.policy_size)
        self.dropout = nn.Dropout(0.3)

    def forward(
        self,
        x: torch.Tensor,
        globals: torch.Tensor,
        return_features: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input features [B, in_channels, H, W]
            globals: Global features [B, global_features]
            return_features: If True, also return backbone features for auxiliary tasks

        Returns:
            value: [B, num_players] value predictions
            policy: [B, policy_size] policy logits
            features (optional): [B, num_filters + global_features] backbone features
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

        # Backbone with SE blocks
        x = self.relu(self.bn1(self.conv1(x)))
        for block in self.res_blocks:
            x = block(x)

        # MPS-compatible global average pooling
        x = torch.mean(x, dim=[-2, -1])

        # Concatenate global features
        x = torch.cat((x, globals), dim=1)

        # Multi-player value head: outputs [batch, num_players]
        v = self.relu(self.value_fc1(x))
        v = self.dropout(v)
        value = self.tanh(self.value_fc2(v))  # [-1, 1] per player

        # Policy head
        p = self.relu(self.policy_fc1(x))
        p = self.dropout(p)
        policy = self.policy_fc2(p)

        if return_features:
            return value, policy, x
        return value, policy

    def forward_single(
        self, feature: np.ndarray, globals_vec: np.ndarray, player_idx: int = 0
    ) -> tuple[float, np.ndarray]:
        """Convenience method for single-sample inference.

        Args:
            feature: Board features [C, H, W]
            globals_vec: Global features [G]
            player_idx: Which player's value to return (default 0)

        Returns:
            Tuple of (value for player, policy logits)
        """
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(feature[None, ...]).float().to(next(self.parameters()).device)
            g = torch.from_numpy(globals_vec[None, ...]).float().to(next(self.parameters()).device)
            v, p = self.forward(x, g)
        return float(v[0, player_idx].item()), p.cpu().numpy()[0]


class RingRiftCNN_v2_Lite(nn.Module):
    """
    Memory-efficient CNN for 19x19 square boards (48GB memory target).

    This architecture is designed for systems with limited memory (48GB)
    while maintaining reasonable playing strength. Suitable for running
    two instances simultaneously for comparison matches.

    Key trade-offs vs RingRiftCNN_v2:
    - 6 SE residual blocks (vs 12) - faster but shallower
    - 96 filters (vs 192) - smaller representations
    - 192-dim policy intermediate (vs 384)
    - 12 base input channels (vs 14) - reduced history
    - 3 history frames (vs 4) - reduced temporal context

    Input Feature Channels (12 base × 3 frames = 36 total):
        1-4: Per-player stack presence (binary)
        5-8: Per-player marker presence (binary)
        9: Stack height (normalized)
        10: Cap height (normalized)
        11: Collapsed territory (binary)
        12: Current player territory

    Global Features (20):
        Same as RingRiftCNN_v2 for compatibility

    Memory profile (FP32):
    - Model weights: ~60 MB
    - Per-model with activations: ~130 MB
    - Two models + MCTS: ~8 GB total

    Architecture Version:
        v2.0.0-lite - Memory-efficient SE architecture for 48GB systems.
    """

    ARCHITECTURE_VERSION = "v2.0.0-lite"

    def __init__(
        self,
        board_size: int = 19,
        in_channels: int = 12,
        global_features: int = 20,
        num_res_blocks: int = 6,
        num_filters: int = 96,
        history_length: int = 2,
        policy_size: int | None = None,
        policy_intermediate: int = 192,
        value_intermediate: int = 64,
        num_players: int = 4,
        se_reduction: int = 16,
    ):
        super().__init__()
        self.board_size = board_size
        self.num_filters = num_filters
        self.num_players = num_players
        self.global_features = global_features

        self.total_in_channels = in_channels * (history_length + 1)
        self.in_channels = self.total_in_channels  # For forward() validation

        # Initial convolution
        self.conv1 = nn.Conv2d(self.total_in_channels, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU()

        # SE-enhanced residual blocks
        self.res_blocks = nn.ModuleList(
            [SEResidualBlock(num_filters, reduction=se_reduction) for _ in range(num_res_blocks)]
        )

        # Multi-player value head
        self.value_fc1 = nn.Linear(num_filters + global_features, value_intermediate)
        self.value_fc2 = nn.Linear(value_intermediate, num_players)
        self.tanh = nn.Tanh()

        # Policy head
        self.policy_fc1 = nn.Linear(num_filters + global_features, policy_intermediate)
        if policy_size is not None:
            self.policy_size = policy_size
        elif board_size == 8:
            self.policy_size = POLICY_SIZE_8x8
        elif board_size == 19:
            self.policy_size = POLICY_SIZE_19x19
        else:
            self.policy_size = POLICY_SIZE
        self.policy_fc2 = nn.Linear(policy_intermediate, self.policy_size)
        self.dropout = nn.Dropout(0.3)

    def forward(
        self,
        x: torch.Tensor,
        globals: torch.Tensor,
        return_features: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input features [B, in_channels, H, W]
            globals: Global features [B, global_features]
            return_features: If True, also return backbone features for auxiliary tasks

        Returns:
            value: [B, num_players] value predictions
            policy: [B, policy_size] policy logits
            features (optional): [B, num_filters + global_features] backbone features
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

        # Backbone with SE blocks
        x = self.relu(self.bn1(self.conv1(x)))
        for block in self.res_blocks:
            x = block(x)

        # MPS-compatible global average pooling
        x = torch.mean(x, dim=[-2, -1])

        # Concatenate global features
        x = torch.cat((x, globals), dim=1)

        # Multi-player value head: outputs [batch, num_players]
        v = self.relu(self.value_fc1(x))
        v = self.dropout(v)
        value = self.tanh(self.value_fc2(v))

        # Policy head
        p = self.relu(self.policy_fc1(x))
        p = self.dropout(p)
        policy = self.policy_fc2(p)

        if return_features:
            return value, policy, x
        return value, policy

    def forward_single(
        self, feature: np.ndarray, globals_vec: np.ndarray, player_idx: int = 0
    ) -> tuple[float, np.ndarray]:
        """Convenience method for single-sample inference."""
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(feature[None, ...]).float().to(next(self.parameters()).device)
            g = torch.from_numpy(globals_vec[None, ...]).float().to(next(self.parameters()).device)
            v, p = self.forward(x, g)
        return float(v[0, player_idx].item()), p.cpu().numpy()[0]


class RingRiftCNN_v3(nn.Module):
    """
    V3 architecture with spatial policy heads for square boards.

    This architecture improves on V2 by using spatially-structured policy heads
    that preserve the geometric relationship between positions and actions,
    rather than collapsing everything through global average pooling.

    Key improvements over V2:
    - Spatial placement head: Conv1×1 → [B, 3, H, W] for (cell, ring_count) logits
    - Spatial movement head: Conv1×1 → [B, 8*max_dist, H, W] for movement logits
    - Spatial line formation head: Conv1×1 → [B, 4, H, W] for line directions
    - Spatial territory claim head: Conv1×1 → [B, 1, H, W] for territory claims
    - Spatial territory choice head: Conv1×1 → [B, 32, H, W] for territory choice
    - Small FC for special actions (skip_placement, swap_sides, line_choice)
    - Preserves spatial locality during policy computation

    Architecture Version:
        v3.1.0 - Spatial policy heads, SE backbone, MPS compatible, rank distribution output.

    Rank Distribution Output (v3.1.0):
        The value head now outputs a rank probability distribution for each player:
        - Shape: [B, num_players, num_players] where rank_dist[b, p, r] = P(player p finishes at rank r)
        - Uses softmax over ranks (dim=-1) so each player's rank probabilities sum to 1
        - Ranks are 0-indexed: rank 0 = 1st place (winner), rank 1 = 2nd place, etc.
        - Also outputs legacy value for backward compatibility: [B, num_players] in [-1, 1]
    """

    ARCHITECTURE_VERSION = "v3.1.0"

    def __init__(
        self,
        board_size: int = 19,
        in_channels: int = 14,
        global_features: int = 20,
        num_res_blocks: int = 12,
        num_filters: int = 192,
        history_length: int = 3,
        policy_size: int | None = None,
        value_intermediate: int = 128,
        num_players: int = 4,
        se_reduction: int = 16,
        num_ring_counts: int = 3,
        num_directions: int = NUM_SQUARE_DIRS,
        num_line_dirs: int = NUM_LINE_DIRS,
        territory_size_buckets: int = TERRITORY_SIZE_BUCKETS,
        territory_max_players: int = TERRITORY_MAX_PLAYERS,
        # Backward compatibility: accept but ignore legacy params
        policy_intermediate: int | None = None,  # Deprecated in v3.1.0
    ) -> None:
        super().__init__()
        self.board_size = board_size
        self.num_filters = num_filters
        self.num_players = num_players
        self.global_features = global_features

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

        # Input channels = base_channels * (history_length + 1)
        self.total_in_channels = in_channels * (history_length + 1)
        self.in_channels = self.total_in_channels  # For forward() validation

        # Determine policy size
        if policy_size is not None:
            self.policy_size = policy_size
        elif board_size == 8:
            self.policy_size = POLICY_SIZE_8x8
        elif board_size == 19:
            self.policy_size = POLICY_SIZE_19x19
        else:
            self.policy_size = POLICY_SIZE

        # Pre-compute index scatter tensors for policy assembly
        self._register_policy_indices(board_size)

        # Initial convolution
        self.conv1 = nn.Conv2d(self.total_in_channels, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU()

        # SE-enhanced residual blocks
        self.res_blocks = nn.ModuleList(
            [SEResidualBlock(num_filters, reduction=se_reduction) for _ in range(num_res_blocks)]
        )

        # Multi-player value head (legacy, kept for backward compatibility)
        self.value_fc1 = nn.Linear(num_filters + global_features, value_intermediate)
        self.value_fc2 = nn.Linear(value_intermediate, num_players)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.3)

        # === V3.1 Rank Distribution Head ===
        rank_dist_intermediate = value_intermediate * 2
        self.rank_dist_fc1 = nn.Linear(num_filters + global_features, rank_dist_intermediate)
        self.rank_dist_fc2 = nn.Linear(rank_dist_intermediate, num_players * num_players)
        self.rank_softmax = nn.Softmax(dim=-1)

        # === V3 Spatial Policy Heads ===
        self.placement_conv = nn.Conv2d(num_filters, num_ring_counts, kernel_size=1)
        self.movement_conv = nn.Conv2d(num_filters, self.movement_channels, kernel_size=1)
        self.line_form_conv = nn.Conv2d(num_filters, num_line_dirs, kernel_size=1)
        self.territory_claim_conv = nn.Conv2d(num_filters, 1, kernel_size=1)
        self.territory_choice_conv = nn.Conv2d(num_filters, self.territory_choice_channels, kernel_size=1)
        self.special_fc = nn.Linear(num_filters + global_features, 7)

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

        # Placement indices: [3, H, W] → flat index
        placement_idx = torch.zeros(self.num_ring_counts, H, W, dtype=torch.long)
        for y in range(H):
            for x in range(W):
                for r in range(self.num_ring_counts):
                    placement_idx[r, y, x] = y * W * 3 + x * 3 + r
        self.register_buffer("placement_idx", placement_idx)

        # Movement indices: [movement_channels, H, W] → flat index
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

        # Line formation indices: [4, H, W] → flat index
        line_form_idx = torch.zeros(self.num_line_dirs, H, W, dtype=torch.long)
        for y in range(H):
            for x in range(W):
                for ld in range(self.num_line_dirs):
                    line_form_idx[ld, y, x] = line_form_base + y * W * self.num_line_dirs + x * self.num_line_dirs + ld
        self.register_buffer("line_form_idx", line_form_idx)

        # Territory claim indices: [1, H, W] → flat index
        territory_claim_idx = torch.zeros(1, H, W, dtype=torch.long)
        for y in range(H):
            for x in range(W):
                territory_claim_idx[0, y, x] = territory_claim_base + y * W + x
        self.register_buffer("territory_claim_idx", territory_claim_idx)

        # Territory choice indices: [32, H, W] → flat index
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
        placement_logits: torch.Tensor,
        movement_logits: torch.Tensor,
        line_form_logits: torch.Tensor,
        territory_claim_logits: torch.Tensor,
        territory_choice_logits: torch.Tensor,
        special_logits: torch.Tensor,
    ) -> torch.Tensor:
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
        x: torch.Tensor,
        globals: torch.Tensor,
        return_features: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with spatial policy heads and rank distribution output."""
        # Input validation: fail fast on channel mismatch (Dec 2025)
        if x.shape[1] != self.in_channels:
            raise RuntimeError(
                f"Input channel mismatch in {self.__class__.__name__}.forward():\n"
                f"  Input has {x.shape[1]} channels\n"
                f"  Model expects {self.in_channels} channels\n"
                f"  This indicates encoder/model version mismatch.\n"
                f"  Check that data was exported with matching encoder version."
            )

        out = self.relu(self.bn1(self.conv1(x)))
        for block in self.res_blocks:
            out = block(out)

        # Value head (legacy)
        v_pooled = torch.mean(out, dim=[-2, -1])
        v_cat = torch.cat([v_pooled, globals], dim=1)
        v_hidden = self.relu(self.value_fc1(v_cat))
        v_hidden = self.dropout(v_hidden)
        v_out = self.tanh(self.value_fc2(v_hidden))

        # Rank Distribution Head (V3.1)
        rank_hidden = self.relu(self.rank_dist_fc1(v_cat))
        rank_hidden = self.dropout(rank_hidden)
        rank_logits = self.rank_dist_fc2(rank_hidden)
        rank_logits = rank_logits.view(-1, self.num_players, self.num_players)
        rank_dist = self.rank_softmax(rank_logits)

        # Spatial policy heads
        placement_logits = self.placement_conv(out)
        movement_logits = self.movement_conv(out)
        line_form_logits = self.line_form_conv(out)
        territory_claim_logits = self.territory_claim_conv(out)
        territory_choice_logits = self.territory_choice_conv(out)

        special_input = torch.cat([v_pooled, globals], dim=1)
        special_logits = self.special_fc(special_input)

        policy_logits = self._scatter_policy_logits(
            placement_logits,
            movement_logits,
            line_form_logits,
            territory_claim_logits,
            territory_choice_logits,
            special_logits,
        )

        if return_features:
            return v_out, policy_logits, rank_dist, v_cat

        return v_out, policy_logits, rank_dist

    def forward_single(
        self, feature: np.ndarray, globals_vec: np.ndarray, player_idx: int = 0
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Convenience method for single-sample inference."""
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(feature[None, ...]).float().to(next(self.parameters()).device)
            g = torch.from_numpy(globals_vec[None, ...]).float().to(next(self.parameters()).device)
            v, p, rank_dist = self.forward(x, g)
        return float(v[0, player_idx].item()), p.cpu().numpy()[0], rank_dist.cpu().numpy()[0]


class RingRiftCNN_v3_Lite(nn.Module):
    """
    Memory-efficient V3 architecture with spatial policy heads (48GB target).

    Same spatial policy head design as RingRiftCNN_v3 but with reduced capacity:
    - 6 SE residual blocks (vs 12)
    - 96 filters (vs 192)
    - 3 history frames (vs 4)
    - 12 base input channels (vs 14)

    Architecture Version:
        v3.1.0-lite - Spatial policy heads, reduced capacity, rank distribution output.
    """

    ARCHITECTURE_VERSION = "v3.1.0-lite"

    def __init__(
        self,
        board_size: int = 19,
        in_channels: int = 12,
        global_features: int = 20,
        num_res_blocks: int = 6,
        num_filters: int = 96,
        history_length: int = 2,
        policy_size: int | None = None,
        value_intermediate: int = 64,
        num_players: int = 4,
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

        if board_size == 8:
            self.max_distance = MAX_DIST_SQUARE8
        else:
            self.max_distance = MAX_DIST_SQUARE19

        self.num_ring_counts = num_ring_counts
        self.num_directions = num_directions
        self.num_line_dirs = num_line_dirs
        self.territory_size_buckets = territory_size_buckets
        self.territory_max_players = territory_max_players
        self.movement_channels = num_directions * self.max_distance
        self.territory_choice_channels = territory_size_buckets * territory_max_players

        self.total_in_channels = in_channels * (history_length + 1)

        if policy_size is not None:
            self.policy_size = policy_size
        elif board_size == 8:
            self.policy_size = POLICY_SIZE_8x8
        elif board_size == 19:
            self.policy_size = POLICY_SIZE_19x19
        else:
            self.policy_size = POLICY_SIZE

        self._register_policy_indices(board_size)

        self.in_channels = self.total_in_channels  # For forward() validation
        self.conv1 = nn.Conv2d(self.total_in_channels, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU()

        self.res_blocks = nn.ModuleList(
            [SEResidualBlock(num_filters, reduction=se_reduction) for _ in range(num_res_blocks)]
        )

        self.value_fc1 = nn.Linear(num_filters + global_features, value_intermediate)
        self.value_fc2 = nn.Linear(value_intermediate, num_players)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.3)

        rank_dist_intermediate = value_intermediate * 2
        self.rank_dist_fc1 = nn.Linear(num_filters + global_features, rank_dist_intermediate)
        self.rank_dist_fc2 = nn.Linear(rank_dist_intermediate, num_players * num_players)
        self.rank_softmax = nn.Softmax(dim=-1)

        self.placement_conv = nn.Conv2d(num_filters, num_ring_counts, kernel_size=1)
        self.movement_conv = nn.Conv2d(num_filters, self.movement_channels, kernel_size=1)
        self.line_form_conv = nn.Conv2d(num_filters, num_line_dirs, kernel_size=1)
        self.territory_claim_conv = nn.Conv2d(num_filters, 1, kernel_size=1)
        self.territory_choice_conv = nn.Conv2d(num_filters, self.territory_choice_channels, kernel_size=1)
        self.special_fc = nn.Linear(num_filters + global_features, 7)

    def _register_policy_indices(self, board_size: int) -> None:
        """Pre-compute index tensors for policy assembly."""
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

        placement_idx = torch.zeros(self.num_ring_counts, H, W, dtype=torch.long)
        for y in range(H):
            for x in range(W):
                for r in range(self.num_ring_counts):
                    placement_idx[r, y, x] = y * W * 3 + x * 3 + r
        self.register_buffer("placement_idx", placement_idx)

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

        line_form_idx = torch.zeros(self.num_line_dirs, H, W, dtype=torch.long)
        for y in range(H):
            for x in range(W):
                for ld in range(self.num_line_dirs):
                    line_form_idx[ld, y, x] = line_form_base + y * W * self.num_line_dirs + x * self.num_line_dirs + ld
        self.register_buffer("line_form_idx", line_form_idx)

        territory_claim_idx = torch.zeros(1, H, W, dtype=torch.long)
        for y in range(H):
            for x in range(W):
                territory_claim_idx[0, y, x] = territory_claim_base + y * W + x
        self.register_buffer("territory_claim_idx", territory_claim_idx)

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
        placement_logits: torch.Tensor,
        movement_logits: torch.Tensor,
        line_form_logits: torch.Tensor,
        territory_claim_logits: torch.Tensor,
        territory_choice_logits: torch.Tensor,
        special_logits: torch.Tensor,
    ) -> torch.Tensor:
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
        x: torch.Tensor,
        globals: torch.Tensor,
        return_features: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with spatial policy heads and rank distribution output."""
        # Input validation: fail fast on channel mismatch (Dec 2025)
        if x.shape[1] != self.in_channels:
            raise RuntimeError(
                f"Input channel mismatch in {self.__class__.__name__}.forward():\n"
                f"  Input has {x.shape[1]} channels\n"
                f"  Model expects {self.in_channels} channels\n"
                f"  This indicates encoder/model version mismatch.\n"
                f"  Check that data was exported with matching encoder version."
            )

        out = self.relu(self.bn1(self.conv1(x)))
        for block in self.res_blocks:
            out = block(out)

        v_pooled = torch.mean(out, dim=[-2, -1])
        v_cat = torch.cat([v_pooled, globals], dim=1)
        v_hidden = self.relu(self.value_fc1(v_cat))
        v_hidden = self.dropout(v_hidden)
        v_out = self.tanh(self.value_fc2(v_hidden))

        rank_hidden = self.relu(self.rank_dist_fc1(v_cat))
        rank_hidden = self.dropout(rank_hidden)
        rank_logits = self.rank_dist_fc2(rank_hidden)
        rank_logits = rank_logits.view(-1, self.num_players, self.num_players)
        rank_dist = self.rank_softmax(rank_logits)

        placement_logits = self.placement_conv(out)
        movement_logits = self.movement_conv(out)
        line_form_logits = self.line_form_conv(out)
        territory_claim_logits = self.territory_claim_conv(out)
        territory_choice_logits = self.territory_choice_conv(out)

        special_input = torch.cat([v_pooled, globals], dim=1)
        special_logits = self.special_fc(special_input)

        policy_logits = self._scatter_policy_logits(
            placement_logits,
            movement_logits,
            line_form_logits,
            territory_claim_logits,
            territory_choice_logits,
            special_logits,
        )

        if return_features:
            return v_out, policy_logits, rank_dist, v_cat

        return v_out, policy_logits, rank_dist

    def forward_single(
        self, feature: np.ndarray, globals_vec: np.ndarray, player_idx: int = 0
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Convenience method for single-sample inference."""
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(feature[None, ...]).float().to(next(self.parameters()).device)
            g = torch.from_numpy(globals_vec[None, ...]).float().to(next(self.parameters()).device)
            v, p, rank_dist = self.forward(x, g)
        return float(v[0, player_idx].item()), p.cpu().numpy()[0], rank_dist.cpu().numpy()[0]


class RingRiftCNN_v4(nn.Module):
    """
    V4 architecture discovered by Neural Architecture Search (NAS).

    This architecture incorporates the optimal hyperparameters found by
    evolutionary NAS, combining the game-specific features of V3 (spatial
    policy heads, rank distribution) with NAS-optimized structural choices:

    NAS-Discovered Improvements:
    - Multi-head self-attention (4 heads) instead of SE blocks
    - 13 residual blocks (vs 12 in v3)
    - 128 filters (vs 192 in v3, more efficient)
    - 5x5 initial kernel (vs 3x3, better spatial coverage)
    - Deeper value head (3 layers vs 2)
    - Lower dropout (0.08 vs 0.3)

    Architecture Version:
        v4.0.0 - NAS-optimized attention architecture.
    """

    ARCHITECTURE_VERSION = "v4.0.0"

    def __init__(
        self,
        board_size: int = 19,
        in_channels: int = 14,
        global_features: int = 20,
        num_res_blocks: int = 13,
        num_filters: int = 128,
        history_length: int = 3,
        policy_size: int | None = None,
        value_intermediate: int = 256,
        value_hidden: int = 256,
        num_players: int = 4,
        num_attention_heads: int = 4,
        dropout: float = 0.08,
        initial_kernel_size: int = 5,
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
        self.num_res_blocks = num_res_blocks
        self.dropout_rate = dropout

        if board_size == 8:
            self.max_distance = MAX_DIST_SQUARE8
        else:
            self.max_distance = MAX_DIST_SQUARE19

        self.num_ring_counts = num_ring_counts
        self.num_directions = num_directions
        self.num_line_dirs = num_line_dirs
        self.territory_size_buckets = territory_size_buckets
        self.territory_max_players = territory_max_players
        self.movement_channels = num_directions * self.max_distance
        self.territory_choice_channels = territory_size_buckets * territory_max_players

        self.total_in_channels = in_channels * (history_length + 1)

        if policy_size is not None:
            self.policy_size = policy_size
        elif board_size == 8:
            self.policy_size = POLICY_SIZE_8x8
        elif board_size == 19:
            self.policy_size = POLICY_SIZE_19x19
        else:
            self.policy_size = POLICY_SIZE

        self._register_policy_indices(board_size)

        self.in_channels = self.total_in_channels  # For forward() validation

        # Initial convolution with larger kernel (NAS optimal: 5x5)
        self.conv1 = nn.Conv2d(
            self.total_in_channels,
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

        # Deeper Value Head (NAS optimal: 3 layers)
        self.value_fc1 = nn.Linear(num_filters + global_features, value_intermediate)
        self.value_fc2 = nn.Linear(value_intermediate, value_hidden)
        self.value_fc3 = nn.Linear(value_hidden, num_players)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

        # Rank Distribution Head
        rank_dist_intermediate = value_intermediate
        self.rank_dist_fc1 = nn.Linear(num_filters + global_features, rank_dist_intermediate)
        self.rank_dist_fc2 = nn.Linear(rank_dist_intermediate, value_hidden)
        self.rank_dist_fc3 = nn.Linear(value_hidden, num_players * num_players)
        self.rank_softmax = nn.Softmax(dim=-1)

        # Spatial Policy Heads
        self.placement_conv = nn.Conv2d(num_filters, num_ring_counts, kernel_size=1)
        self.movement_conv = nn.Conv2d(num_filters, self.movement_channels, kernel_size=1)
        self.line_form_conv = nn.Conv2d(num_filters, num_line_dirs, kernel_size=1)
        self.territory_claim_conv = nn.Conv2d(num_filters, 1, kernel_size=1)
        self.territory_choice_conv = nn.Conv2d(num_filters, self.territory_choice_channels, kernel_size=1)
        self.special_fc = nn.Linear(num_filters + global_features, 7)

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

        placement_idx = torch.zeros(self.num_ring_counts, H, W, dtype=torch.long)
        for y in range(H):
            for x in range(W):
                for r in range(self.num_ring_counts):
                    placement_idx[r, y, x] = y * W * 3 + x * 3 + r
        self.register_buffer("placement_idx", placement_idx)

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

        line_form_idx = torch.zeros(self.num_line_dirs, H, W, dtype=torch.long)
        for y in range(H):
            for x in range(W):
                for ld in range(self.num_line_dirs):
                    line_form_idx[ld, y, x] = line_form_base + y * W * self.num_line_dirs + x * self.num_line_dirs + ld
        self.register_buffer("line_form_idx", line_form_idx)

        territory_claim_idx = torch.zeros(1, H, W, dtype=torch.long)
        for y in range(H):
            for x in range(W):
                territory_claim_idx[0, y, x] = territory_claim_base + y * W + x
        self.register_buffer("territory_claim_idx", territory_claim_idx)

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
        placement_logits: torch.Tensor,
        movement_logits: torch.Tensor,
        line_form_logits: torch.Tensor,
        territory_claim_logits: torch.Tensor,
        territory_choice_logits: torch.Tensor,
        special_logits: torch.Tensor,
    ) -> torch.Tensor:
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
        x: torch.Tensor,
        globals: torch.Tensor,
        return_features: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with attention backbone and spatial policy heads."""
        # Input validation: fail fast on channel mismatch (Dec 2025)
        if x.shape[1] != self.in_channels:
            raise RuntimeError(
                f"Input channel mismatch in {self.__class__.__name__}.forward():\n"
                f"  Input has {x.shape[1]} channels\n"
                f"  Model expects {self.in_channels} channels\n"
                f"  This indicates encoder/model version mismatch.\n"
                f"  Check that data was exported with matching encoder version."
            )

        out = self.relu(self.bn1(self.conv1(x)))
        for block in self.res_blocks:
            out = block(out)

        # Deeper Value Head (3 layers)
        v_pooled = torch.mean(out, dim=[-2, -1])
        v_cat = torch.cat([v_pooled, globals], dim=1)
        v_hidden = self.relu(self.value_fc1(v_cat))
        v_hidden = self.dropout(v_hidden)
        v_hidden = self.relu(self.value_fc2(v_hidden))
        v_hidden = self.dropout(v_hidden)
        v_out = self.tanh(self.value_fc3(v_hidden))

        # Rank Distribution Head (3 layers)
        rank_hidden = self.relu(self.rank_dist_fc1(v_cat))
        rank_hidden = self.dropout(rank_hidden)
        rank_hidden = self.relu(self.rank_dist_fc2(rank_hidden))
        rank_hidden = self.dropout(rank_hidden)
        rank_logits = self.rank_dist_fc3(rank_hidden)
        rank_logits = rank_logits.view(-1, self.num_players, self.num_players)
        rank_dist = self.rank_softmax(rank_logits)

        # Spatial Policy Heads
        placement_logits = self.placement_conv(out)
        movement_logits = self.movement_conv(out)
        line_form_logits = self.line_form_conv(out)
        territory_claim_logits = self.territory_claim_conv(out)
        territory_choice_logits = self.territory_choice_conv(out)

        special_input = torch.cat([v_pooled, globals], dim=1)
        special_logits = self.special_fc(special_input)

        policy_logits = self._scatter_policy_logits(
            placement_logits,
            movement_logits,
            line_form_logits,
            territory_claim_logits,
            territory_choice_logits,
            special_logits,
        )

        if return_features:
            return v_out, policy_logits, rank_dist, v_cat

        return v_out, policy_logits, rank_dist

    def forward_single(
        self, feature: np.ndarray, globals_vec: np.ndarray, player_idx: int = 0
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Convenience method for single-sample inference."""
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(feature[None, ...]).float().to(next(self.parameters()).device)
            g = torch.from_numpy(globals_vec[None, ...]).float().to(next(self.parameters()).device)
            v, p, rank_dist = self.forward(x, g)
        return float(v[0, player_idx].item()), p.cpu().numpy()[0], rank_dist.cpu().numpy()[0]


# NOTE: RingRiftCNN_v5 (v5.0.0) was removed in Dec 2025 - it was dead code superseded by v5_heavy.
# Use RingRiftCNN_v5_Heavy from v5_heavy.py instead for maximum strength architecture.
