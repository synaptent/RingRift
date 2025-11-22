import unittest
import torch
import numpy as np
import random
import sys
import os

# Ensure app package is importable
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from app.training.train import seed_all
from app.ai.neural_net import RingRiftCNN


class TestDeterminism(unittest.TestCase):
    def test_seed_determinism(self):
        """
        Verify that setting the seed produces identical results
        """
        seed = 42
        
        # Run 1
        seed_all(seed)
        model1 = RingRiftCNN(board_size=8, in_channels=10, global_features=10)
        input1 = torch.randn(4, 40, 8, 8)
        globals1 = torch.randn(4, 10)
        val1, pol1 = model1(input1, globals1)
        
        # Run 2
        seed_all(seed)
        model2 = RingRiftCNN(board_size=8, in_channels=10, global_features=10)
        input2 = torch.randn(4, 40, 8, 8)
        globals2 = torch.randn(4, 10)
        val2, pol2 = model2(input2, globals2)
        
        # Check equality
        self.assertTrue(torch.allclose(val1, val2))
        self.assertTrue(torch.allclose(pol1, pol2))
        
        # Check that inputs were generated identically
        self.assertTrue(torch.allclose(input1, input2))
        self.assertTrue(torch.allclose(globals1, globals2))
        
        # Check weights
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            self.assertTrue(torch.allclose(p1, p2))

if __name__ == '__main__':
    unittest.main()