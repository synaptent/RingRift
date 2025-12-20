"""Neural network building blocks for RingRift CNN architectures.

This module contains reusable building blocks used across different
versions of the RingRift neural network:
- ResidualBlock: Basic residual block with skip connection
- SEResidualBlock: Squeeze-and-Excitation enhanced residual block
- AttentionResidualBlock: Residual block with multi-head self-attention
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Residual block with two 3x3 convolutions and skip connection."""

    def __init__(self, channels: int):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


class SEResidualBlock(nn.Module):
    """Squeeze-and-Excitation enhanced residual block for v2 architectures.

    SE blocks improve global pattern recognition by adaptively recalibrating
    channel-wise feature responses. This is particularly valuable for RingRift
    where global dependencies (territory connectivity, line formation) are critical.

    The SE mechanism:
    1. Squeeze: Global average pooling to get channel descriptors
    2. Excitation: FC layers to learn channel interdependencies
    3. Scale: Multiply original features by learned channel weights

    Adds ~1% parameter overhead but significantly improves pattern recognition.

    Reference: Hu et al., "Squeeze-and-Excitation Networks" (CVPR 2018)
    """

    def __init__(self, channels: int, reduction: int = 16):
        """
        Args:
            channels: Number of input/output channels
            reduction: Reduction ratio for SE bottleneck (default 16)
        """
        super(SEResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

        # Squeeze-and-Excitation layers
        reduced_channels = max(channels // reduction, 8)  # Minimum 8 channels
        self.se_fc1 = nn.Linear(channels, reduced_channels)
        self.se_fc2 = nn.Linear(reduced_channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Squeeze-and-Excitation
        # Squeeze: Global average pooling [B, C, H, W] -> [B, C]
        se = torch.mean(out, dim=[-2, -1])
        # Excitation: FC -> ReLU -> FC -> Sigmoid
        se = self.relu(self.se_fc1(se))
        se = torch.sigmoid(self.se_fc2(se))
        # Scale: Multiply features by channel attention
        out = out * se.unsqueeze(-1).unsqueeze(-1)

        out += residual
        out = self.relu(out)
        return out


class AttentionResidualBlock(nn.Module):
    """Residual block with multi-head self-attention.

    This block combines a standard residual convolution path with a spatial
    self-attention mechanism. The attention allows the network to capture
    long-range dependencies between board positions, which is critical for
    understanding territory connectivity and line formation patterns.

    NAS found this more effective than SE blocks for RingRift.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        dropout: float = 0.08,
        kernel_size: int = 3,
    ):
        super(AttentionResidualBlock, self).__init__()
        self.channels = channels
        self.num_heads = num_heads

        # Convolutional path
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(channels)

        # Multi-head self-attention
        # We use 1x1 convs for Q, K, V projections to maintain spatial structure
        self.head_dim = channels // num_heads
        assert channels % num_heads == 0, f"channels ({channels}) must be divisible by num_heads ({num_heads})"

        self.query = nn.Conv2d(channels, channels, kernel_size=1)
        self.key = nn.Conv2d(channels, channels, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.attention_out = nn.Conv2d(channels, channels, kernel_size=1)

        self.attention_dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm([channels])

        # Learnable mixing parameter for attention vs conv path
        self.attention_gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        residual = x

        # Convolutional path
        conv_out = self.relu(self.bn1(self.conv1(x)))
        conv_out = self.bn2(self.conv2(conv_out))

        # Attention path
        # Reshape for multi-head attention: [B, C, H, W] -> [B, heads, head_dim, H*W]
        q = self.query(x).view(B, self.num_heads, self.head_dim, H * W)
        k = self.key(x).view(B, self.num_heads, self.head_dim, H * W)
        v = self.value(x).view(B, self.num_heads, self.head_dim, H * W)

        # Compute attention: [B, heads, H*W, H*W]
        attn_weights = torch.matmul(q.transpose(-2, -1), k) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.attention_dropout(attn_weights)

        # Apply attention: [B, heads, head_dim, H*W]
        attn_out = torch.matmul(v, attn_weights.transpose(-2, -1))
        attn_out = attn_out.view(B, C, H, W)
        attn_out = self.attention_out(attn_out)

        # Gated combination of conv and attention paths
        gate = torch.sigmoid(self.attention_gate)
        out = conv_out + gate * attn_out

        # Residual connection
        out = out + residual
        out = self.relu(out)

        return out


def create_hex_mask(radius: int, bounding_size: int) -> torch.Tensor:
    """Create a hex board validity mask for the given radius.

    For a hex board embedded in a square bounding box, this creates a mask
    where valid hex cells are 1.0 and invalid (padding) cells are 0.0.

    Args:
        radius: Hex board radius (e.g., 12 for 469-cell board)
        bounding_size: Size of the square bounding box (e.g., 25)

    Returns:
        Tensor of shape [1, 1, bounding_size, bounding_size] with valid hex cells as 1.0
    """
    mask = torch.zeros(1, 1, bounding_size, bounding_size)
    center = bounding_size // 2

    for row in range(bounding_size):
        for col in range(bounding_size):
            # Convert to axial coordinates (q, r) centered at origin
            q = col - center
            r = row - center

            # Check if within hex radius using axial distance formula
            # For axial coords: distance = max(|q|, |r|, |q + r|)
            if max(abs(q), abs(r), abs(q + r)) <= radius:
                mask[0, 0, row, col] = 1.0

    return mask
