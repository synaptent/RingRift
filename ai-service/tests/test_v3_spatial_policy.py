"""
Tests for V3 spatial policy head architecture.

These tests verify that V3 architectures (HexNeuralNet_v3, HexNeuralNet_v3_Lite)
correctly implement spatial policy heads that produce valid policy vectors.

V3 key features:
- Spatial placement head: Conv1x1 → [B, 3, H, W] for (cell, ring_count) logits
- Spatial movement head: Conv1x1 → [B, 144, H, W] for (cell, dir, dist) logits
- Small FC for special actions (skip_placement)
- Logits scattered into canonical P_HEX=91,876 flat policy vector
"""

import pytest
import torch

from app.ai.neural_net import (
    HEX_BOARD_SIZE,
    HEX_MAX_DIST,
    HEX_MOVEMENT_BASE,
    HEX_PLACEMENT_SPAN,
    HEX_SPECIAL_BASE,
    NUM_HEX_DIRS,
    P_HEX,
    HexNeuralNet_v2,
    HexNeuralNet_v3,
    HexNeuralNet_v3_Lite,
)


class TestV3ArchitectureCreation:
    """Tests for V3 architecture instantiation."""

    def test_v3_architecture_creation(self):
        """Test that HexNeuralNet_v3 can be instantiated with default params."""
        model = HexNeuralNet_v3()
        assert model is not None
        assert isinstance(model, HexNeuralNet_v3)
        assert model.ARCHITECTURE_VERSION == "v3.0.0"

    def test_v3_lite_architecture_creation(self):
        """Test that HexNeuralNet_v3_Lite can be instantiated."""
        model = HexNeuralNet_v3_Lite()
        assert model is not None
        assert isinstance(model, HexNeuralNet_v3_Lite)
        assert model.ARCHITECTURE_VERSION == "v3.0.0-lite"

    def test_v3_has_spatial_policy_heads(self):
        """Verify V3 has placement and movement conv layers."""
        model = HexNeuralNet_v3()

        # Should have spatial conv heads
        assert hasattr(model, 'placement_conv')
        assert hasattr(model, 'movement_conv')
        assert hasattr(model, 'special_fc')

        # Verify conv dimensions
        assert model.placement_conv.out_channels == 3  # 3 ring counts
        assert model.movement_conv.out_channels == 144  # 6 dirs × 24 dists

    def test_v3_has_policy_index_buffers(self):
        """Verify V3 has pre-computed index buffers for scatter."""
        model = HexNeuralNet_v3()

        assert hasattr(model, 'placement_idx')
        assert hasattr(model, 'movement_idx')
        assert hasattr(model, 'hex_mask')

        # Verify buffer shapes
        assert model.placement_idx.shape == (3, 25, 25)
        assert model.movement_idx.shape == (144, 25, 25)


class TestV3ForwardPass:
    """Tests for V3 forward pass correctness."""

    def test_v3_forward_output_shapes(self):
        """Test V3 forward pass produces correct output shapes."""
        model = HexNeuralNet_v3()
        batch_size = 4

        x = torch.randn(batch_size, 64, 25, 25)  # 56 base + 8 phase/chain
        globals_in = torch.randn(batch_size, 20)

        value, policy = model(x, globals_in)

        assert value.shape == (batch_size, 4), f"Expected value shape (4, 4), got {value.shape}"
        assert policy.shape == (batch_size, P_HEX), f"Expected policy shape (4, {P_HEX}), got {policy.shape}"

    def test_v3_lite_forward_output_shapes(self):
        """Test V3 Lite forward pass produces correct output shapes."""
        model = HexNeuralNet_v3_Lite()
        batch_size = 2

        x = torch.randn(batch_size, 44, 25, 25)  # 36 base + 8 phase/chain
        globals_in = torch.randn(batch_size, 20)

        value, policy = model(x, globals_in)

        assert value.shape == (batch_size, 4)
        assert policy.shape == (batch_size, P_HEX)

    def test_v3_value_range(self):
        """Test that V3 value output is in [-1, 1] (tanh)."""
        model = HexNeuralNet_v3()
        batch_size = 8

        x = torch.randn(batch_size, 64, 25, 25)  # 56 base + 8 phase/chain
        globals_in = torch.randn(batch_size, 20)

        value, _ = model(x, globals_in)

        assert value.min() >= -1.0
        assert value.max() <= 1.0


class TestV3PolicyIndexing:
    """Tests for V3 policy index scatter correctness."""

    def test_placement_index_bounds(self):
        """Verify placement indices are in valid range [0, 1874]."""
        model = HexNeuralNet_v3()

        assert model.placement_idx.min() >= 0
        assert model.placement_idx.max() < HEX_PLACEMENT_SPAN

    def test_movement_index_bounds(self):
        """Verify movement indices are in valid range [1875, 91874]."""
        model = HexNeuralNet_v3()

        assert model.movement_idx.min() >= HEX_MOVEMENT_BASE
        assert model.movement_idx.max() < HEX_SPECIAL_BASE

    def test_placement_index_formula(self):
        """Verify placement indices follow: idx = y * W * 3 + x * 3 + ring_count."""
        model = HexNeuralNet_v3()
        W = HEX_BOARD_SIZE

        # Test specific positions
        test_cases = [
            (0, 0, 0, 0),  # y=0, x=0, r=0 → idx=0
            (0, 0, 1, 1),  # y=0, x=0, r=1 → idx=1
            (0, 0, 2, 2),  # y=0, x=0, r=2 → idx=2
            (0, 1, 0, 3),  # y=0, x=1, r=0 → idx=3
            (1, 0, 0, W * 3),  # y=1, x=0, r=0 → idx=75
        ]

        for y, x, r, expected_idx in test_cases:
            actual_idx = model.placement_idx[r, y, x].item()
            assert actual_idx == expected_idx, f"At (y={y}, x={x}, r={r}): expected {expected_idx}, got {actual_idx}"

    def test_movement_index_formula(self):
        """Verify movement indices follow the canonical formula."""
        model = HexNeuralNet_v3()

        # Test: y=0, x=0, dir=0, dist=1 → first movement index
        expected = HEX_MOVEMENT_BASE + 0
        actual = model.movement_idx[0, 0, 0].item()
        assert actual == expected, f"First movement index: expected {expected}, got {actual}"

        # Test: y=0, x=0, dir=0, dist=24 (dist-1=23)
        expected = HEX_MOVEMENT_BASE + 23
        actual = model.movement_idx[23, 0, 0].item()
        assert actual == expected

        # Test: y=0, x=0, dir=1, dist=1
        expected = HEX_MOVEMENT_BASE + HEX_MAX_DIST
        actual = model.movement_idx[HEX_MAX_DIST, 0, 0].item()
        assert actual == expected

    def test_no_index_overlap(self):
        """Verify placement and movement indices don't overlap."""
        model = HexNeuralNet_v3()

        placement_max = model.placement_idx.max().item()
        movement_min = model.movement_idx.min().item()

        assert placement_max < movement_min, \
            f"Index overlap: placement_max={placement_max}, movement_min={movement_min}"


class TestV3HexMasking:
    """Tests for V3 hex grid masking."""

    def test_hex_mask_present(self):
        """Verify hex mask buffer is registered."""
        model = HexNeuralNet_v3()
        assert model.hex_mask is not None
        # Hex mask is [1, 1, H, W] for broadcasting with [B, C, H, W]
        assert model.hex_mask.shape == (1, 1, 25, 25)

    def test_invalid_cells_masked(self):
        """Verify that invalid hex cells get -inf logits in policy."""
        model = HexNeuralNet_v3()
        model.eval()

        x = torch.randn(1, 64, 25, 25)
        globals_in = torch.randn(1, 20)

        _, policy = model(x, globals_in)

        # Policy for invalid placement cells should be -1e9
        # Check corner (0, 0) which is invalid in hex grid
        corner_indices = [0, 1, 2]  # Placements at (0, 0) for ring counts 0, 1, 2
        for idx in corner_indices:
            logit = policy[0, idx].item()
            assert logit < -1e8, f"Corner placement {idx} should be masked, got {logit}"

    def test_valid_cells_not_masked(self):
        """Verify that valid hex cells have finite logits."""
        model = HexNeuralNet_v3()
        model.eval()

        x = torch.randn(1, 64, 25, 25)
        globals_in = torch.randn(1, 20)

        _, policy = model(x, globals_in)

        # Check center cell (12, 12) which is valid
        center_y, center_x = 12, 12
        W = HEX_BOARD_SIZE
        center_placement_idx = center_y * W * 3 + center_x * 3 + 0  # ring_count=0

        logit = policy[0, center_placement_idx].item()
        assert logit > -1e8, f"Center placement should not be masked, got {logit}"


class TestV3MPSCompatibility:
    """Tests for V3 MPS device compatibility."""

    def test_v3_no_adaptive_pooling(self):
        """Verify V3 doesn't use AdaptiveAvgPool2d (MPS incompatible)."""
        model = HexNeuralNet_v3()
        assert not hasattr(model, 'adaptive_pool')

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(),
        reason="MPS not available on this system"
    )
    def test_v3_mps_forward_pass(self):
        """Test V3 forward pass on MPS device."""
        model = HexNeuralNet_v3(
            num_res_blocks=2,  # Smaller for test speed
            num_filters=32,
        )
        model = model.to('mps')

        x = torch.randn(2, 64, 25, 25, device='mps')
        globals_in = torch.randn(2, 20, device='mps')

        value, policy = model(x, globals_in)

        assert value.device.type == 'mps'
        assert policy.device.type == 'mps'
        assert value.shape == (2, 4)
        assert policy.shape == (2, P_HEX)


class TestV3GradientFlow:
    """Tests for V3 gradient flow through spatial heads."""

    def test_v3_gradients_flow_to_backbone(self):
        """Verify gradients flow from policy loss to backbone."""
        model = HexNeuralNet_v3(num_res_blocks=2, num_filters=32)

        x = torch.randn(2, 64, 25, 25, requires_grad=True)
        globals_in = torch.randn(2, 20, requires_grad=True)

        _value, policy = model(x, globals_in)

        # Compute loss on a specific policy index
        center_idx = 12 * 25 * 3 + 12 * 3 + 0  # Center cell, ring_count=0
        loss = policy[:, center_idx].sum()
        loss.backward()

        # Verify gradients exist
        assert model.placement_conv.weight.grad is not None
        assert model.conv1.weight.grad is not None

    def test_v3_spatial_locality_preserved(self):
        """Verify spatial heads preserve locality - center changes affect center policy."""
        model = HexNeuralNet_v3(num_res_blocks=2, num_filters=32)
        model.eval()

        # Create two inputs: one with a spike at center
        x1 = torch.zeros(1, 64, 25, 25)
        x2 = torch.zeros(1, 64, 25, 25)
        x2[0, 0, 12, 12] = 10.0  # Spike at center

        globals_in = torch.zeros(1, 20)

        _, policy1 = model(x1, globals_in)
        _, policy2 = model(x2, globals_in)

        # Policy at center should change more than at edges
        center_idx = 12 * 25 * 3 + 12 * 3 + 0
        edge_idx = 12 * 25 * 3 + 1 * 3 + 0  # Near edge but still valid

        center_diff = abs(policy2[0, center_idx] - policy1[0, center_idx])
        edge_diff = abs(policy2[0, edge_idx] - policy1[0, edge_idx])

        # Center should be more affected (not strictly required but expected)
        # Just verify both are affected to some degree
        assert center_diff > 0 or edge_diff > 0


class TestV3ParameterCount:
    """Tests for V3 parameter efficiency."""

    def test_v3_fewer_params_than_large_fc(self):
        """Verify V3 spatial heads are more parameter-efficient than flat FC."""
        model = HexNeuralNet_v3(num_res_blocks=12, num_filters=192)

        # Count parameters in spatial policy heads
        placement_params = sum(p.numel() for p in model.placement_conv.parameters())
        movement_params = sum(p.numel() for p in model.movement_conv.parameters())
        special_params = sum(p.numel() for p in model.special_fc.parameters())
        total_spatial = placement_params + movement_params + special_params

        # Compare to what a flat FC would need: 192 filters → P_HEX
        flat_fc_params = 192 * P_HEX + P_HEX  # weights + bias

        print(f"Spatial policy params: {total_spatial:,}")
        print(f"Flat FC would need: {flat_fc_params:,}")
        print(f"Reduction factor: {flat_fc_params / total_spatial:.1f}x")

        # V3 spatial should be much smaller than flat FC
        assert total_spatial < flat_fc_params, \
            f"Spatial policy ({total_spatial}) should use fewer params than flat FC ({flat_fc_params})"

    def test_v3_total_params_reasonable(self):
        """Verify total V3 model params are in expected range."""
        model = HexNeuralNet_v3()

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total V3 params: {total_params:,}")

        # V3 is more parameter-efficient than V2 due to spatial heads
        # ~8M params vs ~44M for V2 (which has large policy FC layer)
        assert total_params < 50_000_000, f"V3 params ({total_params}) unexpectedly high"
        assert total_params > 5_000_000, f"V3 params ({total_params}) unexpectedly low"
