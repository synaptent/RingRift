"""Enhanced Neural Network Architecture for RingRift (V2).

This module implements the V2 neural network architecture from the Neural AI
Architecture document, featuring:

1. **42-channel board tensor input** (from NeuralEncoderV2)
2. **28-feature global vector input**
3. **Multiplayer value head** (per-player win probabilities)
4. **Auxiliary heads** (ownership, territory, line prediction)
5. **ResNet backbone** with optional MPS compatibility

Architecture Summary:
- Backbone: ResNet-style with configurable depth and width
- Value head: Vector output for 2-4 players (softmax for win probabilities)
- Policy head: Large fixed-size output with masking
- Auxiliary heads: Ownership (spatial), territory/line predictions

Training Targets:
- Value: Per-player expected outcome (from self-play)
- Policy: Move selection distribution (from MCTS visit counts)
- Ownership: Final board ownership per cell (auxiliary)
- Territory: Final territory count per player (auxiliary)
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple, NamedTuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


logger = logging.getLogger(__name__)

# Architecture constants
MAX_PLAYERS = 4
V2_BOARD_CHANNELS = 42
V2_GLOBAL_FEATURES = 28
MAX_BOARD_SIZE = 21  # Hex bounding box (radius 10)
POLICY_SIZE = 55000  # Fixed policy size for all board types


class ResidualBlock(nn.Module):
    """Residual block with two 3x3 convolutions and skip connection."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return F.relu(out)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        # Global average pooling
        y = x.view(b, c, -1).mean(dim=2)
        # FC layers with sigmoid
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        # Scale
        return x * y.view(b, c, 1, 1)


class ResidualBlockSE(nn.Module):
    """Residual block with Squeeze-and-Excitation."""

    def __init__(self, channels: int, se_reduction: int = 16):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels, se_reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = out + residual
        return F.relu(out)


@dataclass
class NetworkOutput:
    """Output from RingRiftCNN_V2 forward pass."""
    value: torch.Tensor          # [B, num_players] - win probabilities
    policy_logits: torch.Tensor  # [B, policy_size] - move logits
    ownership: Optional[torch.Tensor] = None     # [B, num_players, H, W]
    territory: Optional[torch.Tensor] = None     # [B, num_players]
    line_threat: Optional[torch.Tensor] = None   # [B, num_players]


class RingRiftCNN_V2(nn.Module):
    """Enhanced CNN architecture for RingRift with multiplayer support.

    This architecture is designed for:
    - 2-4 player games with vector value output
    - 42-channel board tensor input (from NeuralEncoderV2)
    - 28-feature global vector input
    - Auxiliary training targets for better representation learning

    Architecture Version: v2.0.0

    Key Features:
    - ResNet backbone with optional SE blocks
    - Per-player win probability value head
    - Large policy head with masking support
    - Auxiliary heads for ownership/territory/line prediction

    Usage:
        model = RingRiftCNN_V2(num_players=3, num_res_blocks=15)
        output = model(board_tensor, global_features)
        value = output.value  # [B, 3] win probabilities
        policy = output.policy_logits  # [B, 55000] raw logits
    """

    ARCHITECTURE_VERSION = "v2.0.0"

    def __init__(
        self,
        board_size: int = 8,
        in_channels: int = V2_BOARD_CHANNELS,
        global_features: int = V2_GLOBAL_FEATURES,
        num_players: int = 4,
        num_res_blocks: int = 15,
        num_filters: int = 256,
        policy_size: int = POLICY_SIZE,
        use_se_blocks: bool = False,
        use_auxiliary_heads: bool = True,
    ):
        """Initialize the V2 network.

        Args:
            board_size: Spatial dimension hint (actual size from input)
            in_channels: Number of input channels (42 for V2 encoder)
            global_features: Number of global features (28 for V2 encoder)
            num_players: Number of players (2-4)
            num_res_blocks: Number of residual blocks in backbone
            num_filters: Number of channels in backbone
            policy_size: Size of policy output
            use_se_blocks: Whether to use Squeeze-and-Excitation blocks
            use_auxiliary_heads: Whether to include auxiliary training heads
        """
        super().__init__()
        self.board_size = board_size
        self.in_channels = in_channels
        self.global_features = global_features
        self.num_players = min(num_players, MAX_PLAYERS)
        self.num_res_blocks = num_res_blocks
        self.num_filters = num_filters
        self.policy_size = policy_size
        self.use_auxiliary_heads = use_auxiliary_heads

        # Initial convolution
        self.conv1 = nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)

        # Residual blocks
        block_class = ResidualBlockSE if use_se_blocks else ResidualBlock
        self.res_blocks = nn.ModuleList([
            block_class(num_filters) for _ in range(num_res_blocks)
        ])

        # Value head: produces per-player win probabilities
        self.value_conv = nn.Conv2d(num_filters, 32, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(32)
        # After global pooling: 32 + global_features
        self.value_fc1 = nn.Linear(32 + global_features, 256)
        self.value_fc2 = nn.Linear(256, self.num_players)

        # Policy head: produces move logits
        self.policy_conv = nn.Conv2d(num_filters, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        # We use adaptive pooling to handle variable board sizes
        self.policy_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.policy_fc1 = nn.Linear(32 * 16 + global_features, 512)
        self.policy_fc2 = nn.Linear(512, policy_size)
        self.policy_dropout = nn.Dropout(0.3)

        # Auxiliary heads (optional, for training)
        if use_auxiliary_heads:
            # Ownership head: spatial prediction of who will own each cell
            self.ownership_conv = nn.Conv2d(num_filters, self.num_players, kernel_size=1)

            # Territory head: predicted final territory count per player
            self.territory_conv = nn.Conv2d(num_filters, 16, kernel_size=1)
            self.territory_fc = nn.Linear(16, self.num_players)

            # Line threat head: predicted line completion probability
            self.line_conv = nn.Conv2d(num_filters, 16, kernel_size=1)
            self.line_fc = nn.Linear(16, self.num_players)

    def forward(
        self,
        x: torch.Tensor,
        globals_vec: torch.Tensor,
        compute_auxiliary: bool = True,
    ) -> NetworkOutput:
        """Forward pass.

        Args:
            x: Board tensor [B, C, H, W] where C = 42
            globals_vec: Global features [B, 28]
            compute_auxiliary: Whether to compute auxiliary heads

        Returns:
            NetworkOutput with value, policy, and optional auxiliary outputs
        """
        batch_size = x.size(0)

        # Backbone
        out = F.relu(self.bn1(self.conv1(x)))
        for block in self.res_blocks:
            out = block(out)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.mean(dim=[2, 3])  # Global average pooling
        v = torch.cat([v, globals_vec], dim=1)
        v = F.relu(self.value_fc1(v))
        value = self.value_fc2(v)  # Raw logits for softmax

        # Apply softmax for win probabilities
        value = F.softmax(value, dim=1)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(out)))
        p = self.policy_pool(p)
        p = p.view(batch_size, -1)
        p = torch.cat([p, globals_vec], dim=1)
        p = F.relu(self.policy_fc1(p))
        p = self.policy_dropout(p)
        policy_logits = self.policy_fc2(p)

        # Auxiliary heads (optional)
        ownership = None
        territory = None
        line_threat = None

        if self.use_auxiliary_heads and compute_auxiliary:
            # Ownership: spatial prediction
            ownership = self.ownership_conv(out)
            ownership = F.softmax(ownership, dim=1)  # [B, num_players, H, W]

            # Territory: pooled prediction
            t = F.relu(self.territory_conv(out))
            t = t.mean(dim=[2, 3])
            territory = F.sigmoid(self.territory_fc(t))  # [B, num_players]

            # Line threat: pooled prediction
            l = F.relu(self.line_conv(out))
            l = l.mean(dim=[2, 3])
            line_threat = F.sigmoid(self.line_fc(l))  # [B, num_players]

        return NetworkOutput(
            value=value,
            policy_logits=policy_logits,
            ownership=ownership,
            territory=territory,
            line_threat=line_threat,
        )

    def forward_single(
        self,
        board_tensor: np.ndarray,
        global_features: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convenience method for single-sample inference.

        Args:
            board_tensor: [C, H, W] numpy array
            global_features: [28] numpy array

        Returns:
            (value, policy_logits) as numpy arrays
        """
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device
            x = torch.from_numpy(board_tensor[None, ...]).float().to(device)
            g = torch.from_numpy(global_features[None, ...]).float().to(device)
            output = self.forward(x, g, compute_auxiliary=False)
        return output.value.cpu().numpy()[0], output.policy_logits.cpu().numpy()[0]


class RingRiftCNN_V2_MPS(nn.Module):
    """MPS-compatible variant of RingRiftCNN_V2 for Apple Silicon.

    This model replaces AdaptiveAvgPool2d with manual pooling for MPS
    compatibility while maintaining the same architecture.

    Architecture Version: v2.0.0-mps
    """

    ARCHITECTURE_VERSION = "v2.0.0-mps"

    def __init__(
        self,
        board_size: int = 8,
        in_channels: int = V2_BOARD_CHANNELS,
        global_features: int = V2_GLOBAL_FEATURES,
        num_players: int = 4,
        num_res_blocks: int = 15,
        num_filters: int = 256,
        policy_size: int = POLICY_SIZE,
        use_se_blocks: bool = False,
        use_auxiliary_heads: bool = True,
    ):
        super().__init__()
        self.board_size = board_size
        self.in_channels = in_channels
        self.global_features = global_features
        self.num_players = min(num_players, MAX_PLAYERS)
        self.num_res_blocks = num_res_blocks
        self.num_filters = num_filters
        self.policy_size = policy_size
        self.use_auxiliary_heads = use_auxiliary_heads

        # Initial convolution
        self.conv1 = nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)

        # Residual blocks
        block_class = ResidualBlockSE if use_se_blocks else ResidualBlock
        self.res_blocks = nn.ModuleList([
            block_class(num_filters) for _ in range(num_res_blocks)
        ])

        # Value head
        self.value_conv = nn.Conv2d(num_filters, 32, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 + global_features, 256)
        self.value_fc2 = nn.Linear(256, self.num_players)

        # Policy head (MPS-compatible: no AdaptiveAvgPool2d)
        self.policy_conv = nn.Conv2d(num_filters, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        # MPS: Use global average pooling instead of adaptive
        self.policy_fc1 = nn.Linear(32 + global_features, 512)
        self.policy_fc2 = nn.Linear(512, policy_size)
        self.policy_dropout = nn.Dropout(0.3)

        # Auxiliary heads
        if use_auxiliary_heads:
            self.ownership_conv = nn.Conv2d(num_filters, self.num_players, kernel_size=1)
            self.territory_conv = nn.Conv2d(num_filters, 16, kernel_size=1)
            self.territory_fc = nn.Linear(16, self.num_players)
            self.line_conv = nn.Conv2d(num_filters, 16, kernel_size=1)
            self.line_fc = nn.Linear(16, self.num_players)

    def forward(
        self,
        x: torch.Tensor,
        globals_vec: torch.Tensor,
        compute_auxiliary: bool = True,
    ) -> NetworkOutput:
        batch_size = x.size(0)

        # Backbone
        out = F.relu(self.bn1(self.conv1(x)))
        for block in self.res_blocks:
            out = block(out)

        # Value head (MPS-compatible pooling)
        v = F.relu(self.value_bn(self.value_conv(out)))
        v = torch.mean(v, dim=[2, 3])  # Manual global average pooling
        v = torch.cat([v, globals_vec], dim=1)
        v = F.relu(self.value_fc1(v))
        value = F.softmax(self.value_fc2(v), dim=1)

        # Policy head (MPS-compatible pooling)
        p = F.relu(self.policy_bn(self.policy_conv(out)))
        p = torch.mean(p, dim=[2, 3])  # Manual global average pooling
        p = torch.cat([p, globals_vec], dim=1)
        p = F.relu(self.policy_fc1(p))
        p = self.policy_dropout(p)
        policy_logits = self.policy_fc2(p)

        # Auxiliary heads
        ownership = None
        territory = None
        line_threat = None

        if self.use_auxiliary_heads and compute_auxiliary:
            ownership = F.softmax(self.ownership_conv(out), dim=1)

            t = F.relu(self.territory_conv(out))
            t = torch.mean(t, dim=[2, 3])
            territory = F.sigmoid(self.territory_fc(t))

            l = F.relu(self.line_conv(out))
            l = torch.mean(l, dim=[2, 3])
            line_threat = F.sigmoid(self.line_fc(l))

        return NetworkOutput(
            value=value,
            policy_logits=policy_logits,
            ownership=ownership,
            territory=territory,
            line_threat=line_threat,
        )

    def forward_single(
        self,
        board_tensor: np.ndarray,
        global_features: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device
            x = torch.from_numpy(board_tensor[None, ...]).float().to(device)
            g = torch.from_numpy(global_features[None, ...]).float().to(device)
            output = self.forward(x, g, compute_auxiliary=False)
        return output.value.cpu().numpy()[0], output.policy_logits.cpu().numpy()[0]


def compute_v2_loss(
    output: NetworkOutput,
    target_value: torch.Tensor,
    target_policy: torch.Tensor,
    target_ownership: Optional[torch.Tensor] = None,
    target_territory: Optional[torch.Tensor] = None,
    target_line: Optional[torch.Tensor] = None,
    value_weight: float = 1.0,
    policy_weight: float = 1.0,
    ownership_weight: float = 0.5,
    territory_weight: float = 0.25,
    line_weight: float = 0.25,
) -> Tuple[torch.Tensor, dict]:
    """Compute training loss for RingRiftCNN_V2.

    Args:
        output: NetworkOutput from forward pass
        target_value: [B, num_players] target win probabilities
        target_policy: [B, policy_size] target policy distribution
        target_ownership: [B, num_players, H, W] optional ownership targets
        target_territory: [B, num_players] optional territory targets
        target_line: [B, num_players] optional line threat targets
        *_weight: Weight multipliers for each loss component

    Returns:
        (total_loss, loss_dict) where loss_dict contains individual losses
    """
    losses = {}

    # Value loss: cross-entropy for win probability
    value_loss = F.cross_entropy(
        output.value.log() + 1e-8,  # Numerical stability
        target_value,
    )
    losses['value'] = value_loss

    # Policy loss: cross-entropy with valid move masking
    # Mask is encoded in target_policy (zeros for invalid moves)
    log_probs = F.log_softmax(output.policy_logits, dim=1)
    policy_loss = -torch.sum(target_policy * log_probs, dim=1).mean()
    losses['policy'] = policy_loss

    total_loss = value_weight * value_loss + policy_weight * policy_loss

    # Auxiliary losses (optional)
    if output.ownership is not None and target_ownership is not None:
        ownership_loss = F.cross_entropy(
            output.ownership.log() + 1e-8,
            target_ownership,
        )
        losses['ownership'] = ownership_loss
        total_loss = total_loss + ownership_weight * ownership_loss

    if output.territory is not None and target_territory is not None:
        territory_loss = F.mse_loss(output.territory, target_territory)
        losses['territory'] = territory_loss
        total_loss = total_loss + territory_weight * territory_loss

    if output.line_threat is not None and target_line is not None:
        line_loss = F.mse_loss(output.line_threat, target_line)
        losses['line_threat'] = line_loss
        total_loss = total_loss + line_weight * line_loss

    losses['total'] = total_loss
    return total_loss, losses


def create_v2_model(
    num_players: int = 4,
    board_size: int = 8,
    use_mps: bool = False,
    use_se_blocks: bool = False,
    num_res_blocks: int = 15,
    num_filters: int = 256,
) -> nn.Module:
    """Factory function to create a V2 model.

    Args:
        num_players: Number of players (2-4)
        board_size: Spatial dimension hint
        use_mps: Whether to use MPS-compatible architecture
        use_se_blocks: Whether to use SE blocks
        num_res_blocks: Number of residual blocks
        num_filters: Number of filters in backbone

    Returns:
        Configured RingRiftCNN_V2 or RingRiftCNN_V2_MPS model
    """
    model_class = RingRiftCNN_V2_MPS if use_mps else RingRiftCNN_V2
    return model_class(
        board_size=board_size,
        num_players=num_players,
        num_res_blocks=num_res_blocks,
        num_filters=num_filters,
        use_se_blocks=use_se_blocks,
    )
