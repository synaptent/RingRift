import unittest
import torch
import numpy as np
import sys
import os

# Ensure app package is importable
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from app.ai.neural_net import RingRiftCNN


class TestModelArchitecture(unittest.TestCase):
    def setUp(self):
        self.model = RingRiftCNN(
            board_size=8,
            in_channels=10,
            global_features=10,
            num_res_blocks=2,  # Small for testing
            num_filters=16,    # Small for testing
            history_length=3
        )

    def test_forward_pass_shapes(self):
        """Test that forward pass produces correct output shapes"""
        batch_size = 4
        # Input channels = 10 * (3 + 1) = 40
        features = torch.randn(batch_size, 40, 8, 8)
        globals_vec = torch.randn(batch_size, 10)
        
        value, policy = self.model(features, globals_vec)
        
        self.assertEqual(value.shape, (batch_size, 1))
        self.assertEqual(policy.shape, (batch_size, 55000))

    def test_value_range(self):
        """Test that value output is within [-1, 1]"""
        batch_size = 10
        features = torch.randn(batch_size, 40, 8, 8)
        globals_vec = torch.randn(batch_size, 10)
        
        value, _ = self.model(features, globals_vec)
        
        self.assertTrue(torch.all(value >= -1.0))
        self.assertTrue(torch.all(value <= 1.0))

    def test_forward_single(self):
        """Test the convenience forward_single method"""
        features = np.random.randn(40, 8, 8).astype(np.float32)
        globals_vec = np.random.randn(10).astype(np.float32)
        
        value, policy = self.model.forward_single(features, globals_vec)
        
        self.assertIsInstance(value, float)
        self.assertIsInstance(policy, np.ndarray)
        self.assertEqual(policy.shape, (55000,))
        self.assertTrue(-1.0 <= value <= 1.0)

if __name__ == '__main__':
    unittest.main()