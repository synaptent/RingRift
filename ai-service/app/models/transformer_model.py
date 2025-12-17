"""
Transformer-based Models for RingRift AI.

Provides attention-based architectures for board game AI,
including hybrid CNN-Transformer models.
"""

import math
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


logger = logging.getLogger(__name__)


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


if TORCH_AVAILABLE:

    @dataclass
    class TransformerConfig:
        """Configuration for transformer model."""
        board_size: Tuple[int, int] = (8, 8)
        input_channels: int = 18
        embed_dim: int = 256
        num_heads: int = 8
        num_layers: int = 6
        mlp_ratio: float = 4.0
        dropout: float = 0.1
        attention_dropout: float = 0.1
        use_cls_token: bool = True
        use_positional_encoding: bool = True
        positional_encoding_type: str = "learned"  # learned, sinusoidal, 2d
        patch_size: int = 1  # For ViT-style patching
        num_policy_actions: int = 64
        use_auxiliary_heads: bool = False


    class PositionalEncoding2D(nn.Module):
        """2D positional encoding for board positions."""

        def __init__(self, embed_dim: int, board_size: Tuple[int, int]):
            super().__init__()
            self.embed_dim = embed_dim
            self.board_size = board_size

            # Create 2D positional embeddings
            h, w = board_size
            pos_h = torch.arange(h).unsqueeze(1).expand(h, w).reshape(-1)
            pos_w = torch.arange(w).unsqueeze(0).expand(h, w).reshape(-1)

            # Compute sinusoidal embeddings
            div_term = torch.exp(torch.arange(0, embed_dim // 2, 2) *
                                 -(math.log(10000.0) / (embed_dim // 2)))

            pe = torch.zeros(h * w, embed_dim)

            # Row encoding
            pe[:, 0:embed_dim // 2:2] = torch.sin(pos_h.unsqueeze(1) * div_term)
            pe[:, 1:embed_dim // 2:2] = torch.cos(pos_h.unsqueeze(1) * div_term)

            # Column encoding
            pe[:, embed_dim // 2::2] = torch.sin(pos_w.unsqueeze(1) * div_term)
            pe[:, embed_dim // 2 + 1::2] = torch.cos(pos_w.unsqueeze(1) * div_term)

            self.register_buffer('pe', pe)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Add positional encoding to input.

            Args:
                x: Input tensor of shape (batch, seq_len, embed_dim)

            Returns:
                Output with positional encoding added
            """
            return x + self.pe[:x.size(1)]


    class LearnedPositionalEncoding(nn.Module):
        """Learned positional encoding."""

        def __init__(self, embed_dim: int, max_positions: int):
            super().__init__()
            self.pos_embed = nn.Parameter(torch.randn(1, max_positions, embed_dim) * 0.02)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.pos_embed[:, :x.size(1)]


    class MultiHeadAttention(nn.Module):
        """Multi-head self-attention with optional relative position bias."""

        def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float = 0.0,
            use_relative_pos: bool = False,
            max_positions: int = 64
        ):
            super().__init__()
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.head_dim = embed_dim // num_heads
            self.scale = self.head_dim ** -0.5
            self.use_relative_pos = use_relative_pos

            self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
            self.proj = nn.Linear(embed_dim, embed_dim)
            self.dropout = nn.Dropout(dropout)

            if use_relative_pos:
                # Relative position bias (for 2D board)
                self.rel_pos_bias = nn.Parameter(
                    torch.randn(num_heads, 2 * max_positions - 1, 2 * max_positions - 1) * 0.02
                )

        def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            B, N, C = x.shape

            # Compute Q, K, V
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            # Scaled dot-product attention
            attn = (q @ k.transpose(-2, -1)) * self.scale

            # Add relative position bias
            if self.use_relative_pos and hasattr(self, 'rel_pos_bias'):
                # Simplified: use subset of bias table
                bias = self.rel_pos_bias[:, :N, :N]
                attn = attn + bias.unsqueeze(0)

            if mask is not None:
                attn = attn.masked_fill(mask == 0, float('-inf'))

            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)

            # Combine heads
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)

            return x


    class TransformerBlock(nn.Module):
        """Transformer encoder block with pre-normalization."""

        def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            mlp_ratio: float = 4.0,
            dropout: float = 0.1,
            attention_dropout: float = 0.1
        ):
            super().__init__()

            self.norm1 = nn.LayerNorm(embed_dim)
            self.attn = MultiHeadAttention(
                embed_dim, num_heads, attention_dropout
            )

            self.norm2 = nn.LayerNorm(embed_dim)
            mlp_hidden = int(embed_dim * mlp_ratio)
            self.mlp = nn.Sequential(
                nn.Linear(embed_dim, mlp_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden, embed_dim),
                nn.Dropout(dropout)
            )

        def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            # Pre-norm attention
            x = x + self.attn(self.norm1(x), mask)
            # Pre-norm MLP
            x = x + self.mlp(self.norm2(x))
            return x


    class BoardTransformer(nn.Module):
        """
        Pure Transformer model for board game AI.

        Treats each board position as a token.
        """

        def __init__(self, config: TransformerConfig):
            super().__init__()
            self.config = config

            h, w = config.board_size
            num_positions = h * w

            # Input projection
            self.input_proj = nn.Linear(config.input_channels, config.embed_dim)

            # CLS token for global representation
            if config.use_cls_token:
                self.cls_token = nn.Parameter(torch.randn(1, 1, config.embed_dim) * 0.02)
                num_positions += 1

            # Positional encoding
            if config.use_positional_encoding:
                if config.positional_encoding_type == "2d":
                    self.pos_encoder = PositionalEncoding2D(config.embed_dim, config.board_size)
                else:
                    self.pos_encoder = LearnedPositionalEncoding(
                        config.embed_dim, num_positions
                    )
            else:
                self.pos_encoder = None

            # Transformer blocks
            self.blocks = nn.ModuleList([
                TransformerBlock(
                    config.embed_dim,
                    config.num_heads,
                    config.mlp_ratio,
                    config.dropout,
                    config.attention_dropout
                )
                for _ in range(config.num_layers)
            ])

            self.norm = nn.LayerNorm(config.embed_dim)

            # Output heads
            self.policy_head = nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.ReLU(),
                nn.Linear(config.embed_dim, config.num_policy_actions)
            )

            self.value_head = nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim // 2),
                nn.ReLU(),
                nn.Linear(config.embed_dim // 2, 1),
                nn.Tanh()
            )

            self._init_weights()

        def _init_weights(self):
            """Initialize weights."""
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

        def forward(
            self,
            x: torch.Tensor,
            return_attention: bool = False
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Forward pass.

            Args:
                x: Input tensor of shape (batch, channels, height, width)
                return_attention: If True, return attention weights

            Returns:
                (policy_logits, value)
            """
            B, C, H, W = x.shape

            # Reshape to sequence: (batch, H*W, channels)
            x = x.view(B, C, H * W).transpose(1, 2)

            # Project to embedding dimension
            x = self.input_proj(x)

            # Add CLS token
            if self.config.use_cls_token:
                cls_tokens = self.cls_token.expand(B, -1, -1)
                x = torch.cat([cls_tokens, x], dim=1)

            # Add positional encoding
            if self.pos_encoder is not None:
                x = self.pos_encoder(x)

            # Transformer blocks
            for block in self.blocks:
                x = block(x)

            x = self.norm(x)

            # Use CLS token or mean pooling for global representation
            if self.config.use_cls_token:
                global_repr = x[:, 0]  # CLS token
            else:
                global_repr = x.mean(dim=1)  # Mean pooling

            # Output heads
            policy = self.policy_head(global_repr)
            value = self.value_head(global_repr)

            return policy, value

        def forward_with_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Forward pass returning intermediate features."""
            B, C, H, W = x.shape

            x = x.view(B, C, H * W).transpose(1, 2)
            x = self.input_proj(x)

            if self.config.use_cls_token:
                cls_tokens = self.cls_token.expand(B, -1, -1)
                x = torch.cat([cls_tokens, x], dim=1)

            if self.pos_encoder is not None:
                x = self.pos_encoder(x)

            for block in self.blocks:
                x = block(x)

            x = self.norm(x)

            if self.config.use_cls_token:
                global_repr = x[:, 0]
            else:
                global_repr = x.mean(dim=1)

            policy = self.policy_head(global_repr)
            value = self.value_head(global_repr)

            return policy, value, global_repr


    class ConvStem(nn.Module):
        """Convolutional stem for hybrid models."""

        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_layers: int = 3
        ):
            super().__init__()

            layers = []
            channels = in_channels

            for i in range(num_layers):
                out_ch = out_channels if i == num_layers - 1 else out_channels // 2
                layers.extend([
                    nn.Conv2d(channels, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True)
                ])
                channels = out_ch

            self.stem = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.stem(x)


    class HybridCNNTransformer(nn.Module):
        """
        Hybrid CNN-Transformer model.

        Uses CNN for local feature extraction and Transformer for global reasoning.
        """

        def __init__(self, config: TransformerConfig, cnn_layers: int = 4):
            super().__init__()
            self.config = config

            h, w = config.board_size
            num_positions = h * w

            # CNN stem
            self.cnn_stem = ConvStem(config.input_channels, config.embed_dim, cnn_layers)

            # CLS token
            if config.use_cls_token:
                self.cls_token = nn.Parameter(torch.randn(1, 1, config.embed_dim) * 0.02)
                num_positions += 1

            # Positional encoding
            self.pos_encoder = LearnedPositionalEncoding(config.embed_dim, num_positions)

            # Transformer blocks
            self.blocks = nn.ModuleList([
                TransformerBlock(
                    config.embed_dim,
                    config.num_heads,
                    config.mlp_ratio,
                    config.dropout,
                    config.attention_dropout
                )
                for _ in range(config.num_layers)
            ])

            self.norm = nn.LayerNorm(config.embed_dim)

            # Output heads
            self.policy_head = nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.ReLU(),
                nn.Linear(config.embed_dim, config.num_policy_actions)
            )

            self.value_head = nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim // 2),
                nn.ReLU(),
                nn.Linear(config.embed_dim // 2, 1),
                nn.Tanh()
            )

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            B = x.size(0)

            # CNN feature extraction
            x = self.cnn_stem(x)  # (B, embed_dim, H, W)

            # Reshape to sequence
            x = x.flatten(2).transpose(1, 2)  # (B, H*W, embed_dim)

            # Add CLS token
            if self.config.use_cls_token:
                cls_tokens = self.cls_token.expand(B, -1, -1)
                x = torch.cat([cls_tokens, x], dim=1)

            # Positional encoding
            x = self.pos_encoder(x)

            # Transformer
            for block in self.blocks:
                x = block(x)

            x = self.norm(x)

            # Global representation
            if self.config.use_cls_token:
                global_repr = x[:, 0]
            else:
                global_repr = x.mean(dim=1)

            policy = self.policy_head(global_repr)
            value = self.value_head(global_repr)

            return policy, value


    class EfficientBoardTransformer(nn.Module):
        """
        Efficient Transformer using linear attention.

        Uses performer-style linear attention for O(n) complexity.
        """

        def __init__(self, config: TransformerConfig):
            super().__init__()
            self.config = config

            h, w = config.board_size
            num_positions = h * w

            self.input_proj = nn.Linear(config.input_channels, config.embed_dim)

            if config.use_cls_token:
                self.cls_token = nn.Parameter(torch.randn(1, 1, config.embed_dim) * 0.02)
                num_positions += 1

            self.pos_encoder = LearnedPositionalEncoding(config.embed_dim, num_positions)

            # Use efficient attention blocks
            self.blocks = nn.ModuleList([
                EfficientTransformerBlock(
                    config.embed_dim,
                    config.num_heads,
                    config.mlp_ratio,
                    config.dropout
                )
                for _ in range(config.num_layers)
            ])

            self.norm = nn.LayerNorm(config.embed_dim)

            self.policy_head = nn.Linear(config.embed_dim, config.num_policy_actions)
            self.value_head = nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim // 4),
                nn.ReLU(),
                nn.Linear(config.embed_dim // 4, 1),
                nn.Tanh()
            )

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            B, C, H, W = x.shape

            x = x.view(B, C, H * W).transpose(1, 2)
            x = self.input_proj(x)

            if self.config.use_cls_token:
                cls_tokens = self.cls_token.expand(B, -1, -1)
                x = torch.cat([cls_tokens, x], dim=1)

            x = self.pos_encoder(x)

            for block in self.blocks:
                x = block(x)

            x = self.norm(x)

            if self.config.use_cls_token:
                global_repr = x[:, 0]
            else:
                global_repr = x.mean(dim=1)

            policy = self.policy_head(global_repr)
            value = self.value_head(global_repr)

            return policy, value


    class LinearAttention(nn.Module):
        """Linear attention using kernel feature maps."""

        def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
            super().__init__()
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.head_dim = embed_dim // num_heads

            self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
            self.proj = nn.Linear(embed_dim, embed_dim)
            self.dropout = nn.Dropout(dropout)

        def _feature_map(self, x: torch.Tensor) -> torch.Tensor:
            """Apply ELU + 1 feature map for positive features."""
            return F.elu(x) + 1

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            B, N, C = x.shape

            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            # Apply feature map
            q = self._feature_map(q)
            k = self._feature_map(k)

            # Linear attention: O(N * d^2) instead of O(N^2 * d)
            kv = torch.einsum('bhnd,bhnm->bhdm', k, v)
            qkv = torch.einsum('bhnd,bhdm->bhnm', q, kv)
            z = torch.einsum('bhnd,bhnd->bhn', q, k.sum(dim=2, keepdim=True).expand_as(q))
            z = z.unsqueeze(-1)

            x = qkv / (z + 1e-6)
            x = x.transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.dropout(x)

            return x


    class EfficientTransformerBlock(nn.Module):
        """Transformer block with linear attention."""

        def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            mlp_ratio: float = 4.0,
            dropout: float = 0.1
        ):
            super().__init__()

            self.norm1 = nn.LayerNorm(embed_dim)
            self.attn = LinearAttention(embed_dim, num_heads, dropout)

            self.norm2 = nn.LayerNorm(embed_dim)
            mlp_hidden = int(embed_dim * mlp_ratio)
            self.mlp = nn.Sequential(
                nn.Linear(embed_dim, mlp_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden, embed_dim),
                nn.Dropout(dropout)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
            return x


def create_model(
    model_type: str = "hybrid",
    board_size: Tuple[int, int] = (8, 8),
    input_channels: int = 18,
    embed_dim: int = 256,
    num_heads: int = 8,
    num_layers: int = 6,
    num_policy_actions: int = 64
) -> "nn.Module":
    """
    Create a transformer model.

    Args:
        model_type: 'pure', 'hybrid', or 'efficient'
        board_size: Board dimensions
        input_channels: Number of input feature planes
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        num_policy_actions: Number of policy output actions

    Returns:
        Transformer model
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available")

    config = TransformerConfig(
        board_size=board_size,
        input_channels=input_channels,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        num_policy_actions=num_policy_actions
    )

    if model_type == "pure":
        return BoardTransformer(config)
    elif model_type == "hybrid":
        return HybridCNNTransformer(config)
    elif model_type == "efficient":
        return EfficientBoardTransformer(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main():
    """Demonstrate transformer models."""
    if not TORCH_AVAILABLE:
        print("PyTorch not available")
        return

    import time

    print("=== Transformer Models Demo ===\n")

    # Test configurations
    batch_size = 8
    input_channels = 18
    board_size = (8, 8)
    num_actions = 64

    x = torch.randn(batch_size, input_channels, *board_size)

    models = {
        'Pure Transformer': create_model('pure', board_size, input_channels),
        'Hybrid CNN-Transformer': create_model('hybrid', board_size, input_channels),
        'Efficient Transformer': create_model('efficient', board_size, input_channels)
    }

    for name, model in models.items():
        model.eval()
        param_count = sum(p.numel() for p in model.parameters())

        # Warmup
        with torch.no_grad():
            _ = model(x)

        # Benchmark
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(100):
                policy, value = model(x)
        elapsed = time.perf_counter() - start

        print(f"{name}:")
        print(f"  Parameters: {param_count:,}")
        print(f"  Policy shape: {policy.shape}")
        print(f"  Value shape: {value.shape}")
        print(f"  Throughput: {100 * batch_size / elapsed:.1f} samples/sec")
        print()

    # Test forward_with_features
    print("=== Testing forward_with_features ===")
    pure_model = models['Pure Transformer']
    policy, value, features = pure_model.forward_with_features(x)
    print(f"Features shape: {features.shape}")


if __name__ == "__main__":
    main()
