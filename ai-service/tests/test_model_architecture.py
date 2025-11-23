import unittest
import torch
import numpy as np
import sys
import os

# Ensure app package is importable
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from app.ai.neural_net import RingRiftCNN, HexNeuralNet, P_HEX


class TestModelArchitecture(unittest.TestCase):
    """Architecture sanity checks for the square-board CNN."""

    def setUp(self):
        self.model = RingRiftCNN(
            board_size=8,
            in_channels=10,
            global_features=10,
            num_res_blocks=2,  # Small for testing
            num_filters=16,    # Small for testing
            history_length=3,
        )

    def test_forward_pass_shapes(self):
        """Forward pass produces correct output shapes for square model."""
        batch_size = 4
        # Input channels = 10 * (3 + 1) = 40
        features = torch.randn(batch_size, 40, 8, 8)
        globals_vec = torch.randn(batch_size, 10)

        value, policy = self.model(features, globals_vec)

        self.assertEqual(value.shape, (batch_size, 1))
        self.assertEqual(policy.shape, (batch_size, 55000))

    def test_value_range(self):
        """Value output is constrained to [-1, 1] for square model."""
        batch_size = 10
        features = torch.randn(batch_size, 40, 8, 8)
        globals_vec = torch.randn(batch_size, 10)

        value, _ = self.model(features, globals_vec)

        self.assertTrue(torch.all(value >= -1.0))
        self.assertTrue(torch.all(value <= 1.0))

    def test_forward_single(self):
        """The convenience forward_single method returns correct types/shapes."""
        features = np.random.randn(40, 8, 8).astype(np.float32)
        globals_vec = np.random.randn(10).astype(np.float32)

        value, policy = self.model.forward_single(features, globals_vec)

        self.assertIsInstance(value, float)
        self.assertIsInstance(policy, np.ndarray)
        self.assertEqual(policy.shape, (55000,))
        self.assertTrue(-1.0 <= value <= 1.0)


class TestHexModelArchitecture(unittest.TestCase):
    """Architecture sanity checks for the hex-specific CNN (HexNeuralNet)."""

    def setUp(self):
        # Use a small number of filters/blocks for test speed while keeping
        # the topology identical to the production design.
        self.in_channels = 10  # Example hex feature channels
        self.global_features = 10
        self.model = HexNeuralNet(
            in_channels=self.in_channels,
            global_features=self.global_features,
            num_res_blocks=2,
            num_filters=16,
            board_size=21,
            policy_size=P_HEX,
        )

    def test_hex_forward_pass_shapes_with_mask(self):
        """Forward pass with a full hex_mask yields correct shapes."""
        batch_size = 4
        features = torch.randn(batch_size, self.in_channels, 21, 21)
        globals_vec = torch.randn(batch_size, self.global_features)
        hex_mask = torch.ones(batch_size, 1, 21, 21)

        value, policy = self.model(features, globals_vec, hex_mask=hex_mask)

        self.assertEqual(value.shape, (batch_size, 1))
        self.assertEqual(policy.shape, (batch_size, P_HEX))

    def test_hex_forward_pass_shapes_without_mask(self):
        """Forward pass also works when no hex_mask is provided."""
        batch_size = 3
        features = torch.randn(batch_size, self.in_channels, 21, 21)
        globals_vec = torch.randn(batch_size, self.global_features)

        value, policy = self.model(features, globals_vec)

        self.assertEqual(value.shape, (batch_size, 1))
        self.assertEqual(policy.shape, (batch_size, P_HEX))

    def test_hex_value_range(self):
        """Value head outputs values constrained to [-1, 1] for hex model."""
        batch_size = 5
        features = torch.randn(batch_size, self.in_channels, 21, 21)
        globals_vec = torch.randn(batch_size, self.global_features)
        hex_mask = torch.ones(batch_size, 1, 21, 21)

        value, _ = self.model(features, globals_vec, hex_mask=hex_mask)

        self.assertTrue(torch.all(value >= -1.0))
        self.assertTrue(torch.all(value <= 1.0))


if __name__ == "__main__":
    unittest.main()
