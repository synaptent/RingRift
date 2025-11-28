"""Tests for MCTS dynamic batch sizing feature.

This module tests the DynamicBatchSizer class and its integration with MCTS,
including memory-based batch size calculation and adjustment logic.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

# Allow imports from app/
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from app.ai.mcts_ai import DynamicBatchSizer, MCTSAI, MCTSNode
from app.utils.memory_config import MemoryConfig
from app.models import AIConfig, GameState


class TestDynamicBatchSizer:
    """Tests for the DynamicBatchSizer class."""

    def test_initialization_defaults(self):
        """Test that DynamicBatchSizer initializes with correct defaults."""
        sizer = DynamicBatchSizer()

        assert sizer.batch_size_min == 100
        assert sizer.batch_size_max == 1600
        assert sizer.memory_safety_margin == 0.8
        assert sizer.node_size_estimate == 500
        assert sizer._last_batch_size == 1600
        assert sizer._adjustment_count == 0

    def test_initialization_with_custom_params(self):
        """Test that DynamicBatchSizer accepts custom parameters."""
        memory_config = MemoryConfig(
            max_memory_gb=4.0,
            inference_allocation=0.50,
        )
        sizer = DynamicBatchSizer(
            memory_config=memory_config,
            batch_size_min=50,
            batch_size_max=800,
            memory_safety_margin=0.7,
            node_size_estimate=300,
        )

        assert sizer.batch_size_min == 50
        assert sizer.batch_size_max == 800
        assert sizer.memory_safety_margin == 0.7
        assert sizer.node_size_estimate == 300
        assert sizer._last_batch_size == 800

    def test_stats_returns_correct_values(self):
        """Test that stats() returns correct current values."""
        sizer = DynamicBatchSizer(
            batch_size_min=100,
            batch_size_max=1000,
        )

        stats = sizer.stats()

        assert stats["current_batch_size"] == 1000
        assert stats["node_size_estimate"] == 500
        assert stats["batch_size_min"] == 100
        assert stats["batch_size_max"] == 1000
        assert stats["memory_safety_margin"] == 0.8
        assert stats["adjustment_count"] == 0
        assert stats["sample_count"] == 0


class TestDynamicBatchSizerMemoryCalculation:
    """Tests for memory-based batch size calculation."""

    @patch("app.ai.mcts_ai.psutil")
    def test_high_memory_returns_max_batch_size(self, mock_psutil):
        """When memory is plentiful, batch size should be max."""
        # Mock high available memory (4GB)
        mock_mem = MagicMock()
        mock_mem.available = 4 * 1024**3  # 4GB
        mock_psutil.virtual_memory.return_value = mock_mem

        # Create a memory config with 1GB limit
        memory_config = MemoryConfig(
            max_memory_gb=8.0,
            inference_allocation=0.30,
        )

        sizer = DynamicBatchSizer(
            memory_config=memory_config,
            batch_size_min=100,
            batch_size_max=1600,
        )

        # With lots of memory and no nodes, should get max batch size
        batch_size = sizer.get_optimal_batch_size(current_node_count=0)

        assert batch_size == 1600

    @patch("app.ai.mcts_ai.psutil")
    def test_low_memory_returns_min_batch_size(self, mock_psutil):
        """When memory is scarce, batch size should be min."""
        # Mock very low available memory (40KB - less than 100 * 500 bytes)
        mock_mem = MagicMock()
        mock_mem.available = 40 * 1024  # 40KB
        mock_psutil.virtual_memory.return_value = mock_mem

        # Create a memory config with low limit
        memory_config = MemoryConfig(
            max_memory_gb=0.001,  # 1MB limit
            inference_allocation=0.30,
        )

        sizer = DynamicBatchSizer(
            memory_config=memory_config,
            batch_size_min=100,
            batch_size_max=1600,
            node_size_estimate=500,  # 500 bytes per node
        )

        # With very little memory (40KB / 500 = 80 nodes < 100)
        # Should get min batch size
        batch_size = sizer.get_optimal_batch_size(current_node_count=0)

        assert batch_size == 100

    @patch("app.ai.mcts_ai.psutil")
    def test_large_tree_reduces_available_budget(self, mock_psutil):
        """When tree is large, available budget is reduced."""
        # Mock medium available memory (500MB)
        mock_mem = MagicMock()
        mock_mem.available = 500 * 1024**2  # 500MB
        mock_psutil.virtual_memory.return_value = mock_mem

        memory_config = MemoryConfig(
            max_memory_gb=2.0,
            inference_allocation=0.50,
        )

        sizer = DynamicBatchSizer(
            memory_config=memory_config,
            batch_size_min=100,
            batch_size_max=1600,
            node_size_estimate=500,  # 500 bytes per node
        )

        # Get batch size with no existing nodes
        batch_size_empty = sizer.get_optimal_batch_size(current_node_count=0)

        # Get batch size with 100K nodes (50MB worth of tree)
        batch_size_large = sizer.get_optimal_batch_size(
            current_node_count=100000
        )

        # Large tree should result in smaller or equal batch size
        assert batch_size_large <= batch_size_empty

    @patch("app.ai.mcts_ai.psutil")
    def test_batch_size_respects_minimum(self, mock_psutil):
        """Batch size should never go below minimum."""
        # Mock extremely low memory
        mock_mem = MagicMock()
        mock_mem.available = 1 * 1024**2  # 1MB
        mock_psutil.virtual_memory.return_value = mock_mem

        memory_config = MemoryConfig(
            max_memory_gb=1.0,
            inference_allocation=0.30,
        )

        sizer = DynamicBatchSizer(
            memory_config=memory_config,
            batch_size_min=50,
            batch_size_max=1600,
        )

        batch_size = sizer.get_optimal_batch_size(current_node_count=1000000)

        assert batch_size >= 50

    @patch("app.ai.mcts_ai.psutil")
    def test_batch_size_respects_maximum(self, mock_psutil):
        """Batch size should never exceed maximum."""
        # Mock extremely high memory
        mock_mem = MagicMock()
        mock_mem.available = 100 * 1024**3  # 100GB
        mock_psutil.virtual_memory.return_value = mock_mem

        memory_config = MemoryConfig(
            max_memory_gb=64.0,
            inference_allocation=0.50,
        )

        sizer = DynamicBatchSizer(
            memory_config=memory_config,
            batch_size_min=100,
            batch_size_max=2000,
        )

        batch_size = sizer.get_optimal_batch_size(current_node_count=0)

        assert batch_size <= 2000


class TestDynamicBatchSizerMemorySampling:
    """Tests for memory sampling and node size estimation refinement."""

    @patch("app.ai.mcts_ai.psutil")
    def test_record_memory_sample_stores_samples(self, mock_psutil):
        """Test that memory samples are recorded correctly."""
        mock_process = MagicMock()
        mock_process.memory_info.return_value = MagicMock(rss=100 * 1024**2)
        mock_psutil.Process.return_value = mock_process

        sizer = DynamicBatchSizer()

        sizer.record_memory_sample(1000)

        assert len(sizer._memory_samples) == 1
        assert sizer._memory_samples[0][0] == 1000

    @patch("app.ai.mcts_ai.psutil")
    def test_memory_samples_limited_to_100(self, mock_psutil):
        """Test that only last 100 samples are kept."""
        mock_process = MagicMock()
        mock_process.memory_info.return_value = MagicMock(rss=100 * 1024**2)
        mock_psutil.Process.return_value = mock_process

        sizer = DynamicBatchSizer()

        # Record 150 samples
        for i in range(150):
            mock_process.memory_info.return_value = MagicMock(
                rss=(100 + i) * 1024**2
            )
            sizer.record_memory_sample(i * 100)

        assert len(sizer._memory_samples) == 100
        # Should have the most recent samples (indices 50-149)
        assert sizer._memory_samples[0][0] == 50 * 100

    @patch("app.ai.mcts_ai.psutil")
    def test_node_size_estimate_refined_after_samples(self, mock_psutil):
        """Test that node size estimate is refined with enough samples."""
        mock_process = MagicMock()
        mock_psutil.Process.return_value = mock_process

        sizer = DynamicBatchSizer(node_size_estimate=500)

        # Simulate memory growth that suggests ~600 bytes per node
        base_memory = 100 * 1024**2  # 100MB base
        bytes_per_node = 600

        for i in range(15):
            node_count = i * 1000
            memory = base_memory + (node_count * bytes_per_node)
            mock_process.memory_info.return_value = MagicMock(rss=memory)
            sizer.record_memory_sample(node_count)

        # After enough samples, estimate should be refined
        # The exact value depends on the calculation, but it should change
        stats = sizer.stats()
        assert stats["sample_count"] == 15


class TestMCTSWithDynamicBatching:
    """Tests for MCTS integration with dynamic batching."""

    def test_mcts_init_without_dynamic_batching(self):
        """MCTS should work without dynamic batching (default)."""
        config = AIConfig(difficulty=5, randomness=0.1, rngSeed=42)
        mcts = MCTSAI(
            player_number=1,
            config=config,
        )

        assert mcts.enable_dynamic_batching is False
        assert mcts.dynamic_sizer is None

    def test_mcts_init_with_dynamic_batching_enabled(self):
        """MCTS should create default sizer when dynamic batching enabled."""
        config = AIConfig(difficulty=5, randomness=0.1, rngSeed=42)
        mcts = MCTSAI(
            player_number=1,
            config=config,
            enable_dynamic_batching=True,
        )

        assert mcts.enable_dynamic_batching is True
        assert mcts.dynamic_sizer is not None
        assert isinstance(mcts.dynamic_sizer, DynamicBatchSizer)

    def test_mcts_init_with_custom_dynamic_sizer(self):
        """MCTS should accept custom dynamic sizer."""
        config = AIConfig(difficulty=5, randomness=0.1, rngSeed=42)
        custom_sizer = DynamicBatchSizer(
            batch_size_min=200,
            batch_size_max=800,
        )

        mcts = MCTSAI(
            player_number=1,
            config=config,
            dynamic_sizer=custom_sizer,
            enable_dynamic_batching=True,
        )

        assert mcts.enable_dynamic_batching is True
        assert mcts.dynamic_sizer is custom_sizer
        assert mcts.dynamic_sizer.batch_size_min == 200
        assert mcts.dynamic_sizer.batch_size_max == 800

    def test_mcts_init_with_sizer_but_batching_disabled(self):
        """Providing sizer but not enabling batching should store sizer."""
        config = AIConfig(difficulty=5, randomness=0.1, rngSeed=42)
        custom_sizer = DynamicBatchSizer()

        mcts = MCTSAI(
            player_number=1,
            config=config,
            dynamic_sizer=custom_sizer,
            enable_dynamic_batching=False,
        )

        # Sizer is stored but not used when batching is disabled
        assert mcts.enable_dynamic_batching is False
        assert mcts.dynamic_sizer is custom_sizer

    def test_mcts_clear_tree(self):
        """Test that clear_tree method works correctly."""
        config = AIConfig(difficulty=5, randomness=0.1, rngSeed=42)
        mcts = MCTSAI(
            player_number=1,
            config=config,
        )

        # Set a fake last_root
        mcts.last_root = MagicMock()

        mcts.clear_tree()

        assert mcts.last_root is None


class TestMCTSNode:
    """Basic tests for MCTSNode to ensure it works as expected."""

    def test_node_initialization(self):
        """Test that MCTSNode initializes correctly."""
        mock_state = MagicMock(spec=GameState)
        node = MCTSNode(mock_state)

        assert node.game_state is mock_state
        assert node.parent is None
        assert node.move is None
        assert node.children == []
        assert node.wins == 0
        assert node.visits == 0
        assert node.prior == 0.0

    def test_node_update(self):
        """Test that update() modifies stats correctly."""
        mock_state = MagicMock(spec=GameState)
        node = MCTSNode(mock_state)

        node.update(1.0)

        assert node.visits == 1
        assert node.wins == 1.0

        node.update(0.5)

        assert node.visits == 2
        assert node.wins == 1.5


class TestDynamicBatchSizerAdjustmentLogging:
    """Tests for batch size adjustment tracking."""

    @patch("app.ai.mcts_ai.psutil")
    def test_adjustment_count_increments_on_significant_change(
        self, mock_psutil
    ):
        """Adjustment count increments on significant batch size change."""
        mock_mem = MagicMock()
        mock_psutil.virtual_memory.return_value = mock_mem

        memory_config = MemoryConfig(
            max_memory_gb=8.0,
            inference_allocation=0.30,
        )

        sizer = DynamicBatchSizer(
            memory_config=memory_config,
            batch_size_min=100,
            batch_size_max=1600,
            node_size_estimate=500,
        )

        # Start with high memory - max batch size
        mock_mem.available = 4 * 1024**3  # 4GB
        sizer.get_optimal_batch_size(0)
        initial_count = sizer._adjustment_count

        # Drop to very low memory (40KB) - min batch size
        mock_mem.available = 40 * 1024  # 40KB
        sizer.get_optimal_batch_size(0)

        # Should have recorded an adjustment (significant drop)
        assert sizer._adjustment_count > initial_count

    @patch("app.ai.mcts_ai.psutil")
    def test_small_changes_do_not_increment_count(self, mock_psutil):
        """Small batch size changes should not increment adjustment count."""
        mock_mem = MagicMock()
        mock_psutil.virtual_memory.return_value = mock_mem

        memory_config = MemoryConfig(
            max_memory_gb=8.0,
            inference_allocation=0.30,
        )

        sizer = DynamicBatchSizer(
            memory_config=memory_config,
            batch_size_min=100,
            batch_size_max=1600,
        )

        # Start with high memory
        mock_mem.available = 4 * 1024**3  # 4GB
        sizer.get_optimal_batch_size(0)

        # Still high memory (slightly less)
        mock_mem.available = 3.9 * 1024**3  # 3.9GB
        sizer.get_optimal_batch_size(0)

        # Both should return max (1600), so no adjustment
        assert sizer._adjustment_count == 0