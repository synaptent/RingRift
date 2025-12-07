import unittest
import torch
import numpy as np
import sys
import os

# Ensure app package is importable
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from app.ai.neural_net import (
    RingRiftCNN_v2,
    RingRiftCNN_v2_Lite,
    HexNeuralNet_v2,
    HexNeuralNet_v2_Lite,
    P_HEX,
    POLICY_SIZE_8x8,
)


class TestModelArchitecture_v2(unittest.TestCase):
    """Architecture sanity checks for the V2 square-board CNN."""

    def setUp(self):
        # V2 models have multi-player value head (outputs 4 values)
        self.in_channels = 14
        self.global_features = 20
        self.history_length = 4
        self.model = RingRiftCNN_v2(
            board_size=8,
            in_channels=self.in_channels,
            global_features=self.global_features,
            num_res_blocks=2,  # Small for testing
            num_filters=32,    # Small for testing
            history_length=self.history_length,
        )

    def test_forward_pass_shapes(self):
        """Forward pass produces correct output shapes for V2 square model."""
        batch_size = 4
        # Input channels = in_channels * (history_length + 1)
        total_channels = self.in_channels * (self.history_length + 1)
        features = torch.randn(batch_size, total_channels, 8, 8)
        globals_vec = torch.randn(batch_size, self.global_features)

        value, policy = self.model(features, globals_vec)

        # V2 has multi-player value head: (batch, 4)
        self.assertEqual(value.shape, (batch_size, 4))
        self.assertEqual(policy.shape, (batch_size, POLICY_SIZE_8x8))

    def test_value_range(self):
        """Value output is constrained to [-1, 1] for V2 multi-player model (tanh)."""
        batch_size = 10
        total_channels = self.in_channels * (self.history_length + 1)
        features = torch.randn(batch_size, total_channels, 8, 8)
        globals_vec = torch.randn(batch_size, self.global_features)

        value, _ = self.model(features, globals_vec)

        # Multi-player value head uses tanh, so values are in [-1, 1] per player
        self.assertTrue(torch.all(value >= -1.0))
        self.assertTrue(torch.all(value <= 1.0))


class TestModelArchitecture_v2_Lite(unittest.TestCase):
    """Architecture sanity checks for the V2 Lite (memory-efficient) square-board CNN."""

    def setUp(self):
        self.in_channels = 12
        self.global_features = 20
        self.history_length = 3
        self.model = RingRiftCNN_v2_Lite(
            board_size=8,
            in_channels=self.in_channels,
            global_features=self.global_features,
            num_res_blocks=2,  # Small for testing
            num_filters=32,    # Small for testing
            history_length=self.history_length,
        )

    def test_forward_pass_shapes(self):
        """Forward pass produces correct output shapes for V2 Lite model."""
        batch_size = 4
        total_channels = self.in_channels * (self.history_length + 1)
        features = torch.randn(batch_size, total_channels, 8, 8)
        globals_vec = torch.randn(batch_size, self.global_features)

        value, policy = self.model(features, globals_vec)

        self.assertEqual(value.shape, (batch_size, 4))
        self.assertEqual(policy.shape, (batch_size, POLICY_SIZE_8x8))


class TestHexModelArchitecture_v2(unittest.TestCase):
    """Architecture sanity checks for the V2 hex-specific CNN."""

    def setUp(self):
        self.in_channels = 14
        self.global_features = 20
        self.model = HexNeuralNet_v2(
            in_channels=self.in_channels,
            global_features=self.global_features,
            num_res_blocks=2,
            num_filters=32,
            board_size=21,
            policy_size=P_HEX,
        )

    def test_hex_forward_pass_shapes(self):
        """Forward pass yields correct shapes for V2 hex model."""
        batch_size = 4
        features = torch.randn(batch_size, self.in_channels, 21, 21)
        globals_vec = torch.randn(batch_size, self.global_features)

        value, policy = self.model(features, globals_vec)

        # V2 has multi-player value head
        self.assertEqual(value.shape, (batch_size, 4))
        self.assertEqual(policy.shape, (batch_size, P_HEX))

    def test_hex_value_range(self):
        """Value head outputs are in [-1, 1] for V2 hex model (tanh)."""
        batch_size = 5
        features = torch.randn(batch_size, self.in_channels, 21, 21)
        globals_vec = torch.randn(batch_size, self.global_features)

        value, _ = self.model(features, globals_vec)

        # Multi-player value head uses tanh, so values are in [-1, 1] per player
        self.assertTrue(torch.all(value >= -1.0))
        self.assertTrue(torch.all(value <= 1.0))


class TestHexModelArchitecture_v2_Lite(unittest.TestCase):
    """Architecture sanity checks for the V2 Lite hex-specific CNN."""

    def setUp(self):
        self.in_channels = 12
        self.global_features = 20
        self.model = HexNeuralNet_v2_Lite(
            in_channels=self.in_channels,
            global_features=self.global_features,
            num_res_blocks=2,
            num_filters=32,
            board_size=21,
            policy_size=P_HEX,
        )

    def test_hex_forward_pass_shapes(self):
        """Forward pass yields correct shapes for V2 Lite hex model."""
        batch_size = 3
        features = torch.randn(batch_size, self.in_channels, 21, 21)
        globals_vec = torch.randn(batch_size, self.global_features)

        value, policy = self.model(features, globals_vec)

        self.assertEqual(value.shape, (batch_size, 4))
        self.assertEqual(policy.shape, (batch_size, P_HEX))


if __name__ == "__main__":
    unittest.main()
