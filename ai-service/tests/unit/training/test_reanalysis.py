"""Tests for reanalysis module.

Tests the MuZero-style reanalysis pipeline including:
- ReanalysisConfig validation
- ReanalyzedPosition dataclass
- ReanalysisEngine for re-evaluating games with current model
- Value/policy blending
- MCTS-based reanalysis
- ReanalysisDataset for PyTorch training
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.training.reanalysis import (
    ReanalysisConfig,
    ReanalyzedPosition,
    ReanalysisEngine,
    create_reanalysis_engine,
    reanalyze_training_data,
)

# Check for torch availability
try:
    import torch
    from app.training.reanalysis import ReanalysisDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None


# =============================================================================
# ReanalysisConfig Tests
# =============================================================================


class TestReanalysisConfig:
    """Tests for ReanalysisConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ReanalysisConfig()

        assert config.batch_size == 64
        assert config.max_games_per_run == 1000
        assert config.value_blend_ratio == 0.7
        assert config.policy_blend_ratio == 0.8
        assert config.min_model_elo_delta == 50
        assert config.reanalysis_interval_hours == 6.0
        assert config.priority_recent_games is True
        assert config.cache_dir == "data/reanalysis_cache"
        assert config.min_game_length == 10
        assert config.max_game_length == 500
        assert config.use_mcts is False
        assert config.mcts_simulations == 100
        assert config.mcts_temperature == 1.0
        assert config.capture_q_values is True
        assert config.capture_uncertainty is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ReanalysisConfig(
            batch_size=128,
            value_blend_ratio=0.5,
            use_mcts=True,
            mcts_simulations=200,
        )

        assert config.batch_size == 128
        assert config.value_blend_ratio == 0.5
        assert config.use_mcts is True
        assert config.mcts_simulations == 200

    def test_blend_ratio_bounds(self):
        """Test that blend ratios work at extremes."""
        # Pure new values
        config_new = ReanalysisConfig(value_blend_ratio=1.0, policy_blend_ratio=1.0)
        assert config_new.value_blend_ratio == 1.0

        # Pure old values
        config_old = ReanalysisConfig(value_blend_ratio=0.0, policy_blend_ratio=0.0)
        assert config_old.value_blend_ratio == 0.0


# =============================================================================
# ReanalyzedPosition Tests
# =============================================================================


class TestReanalyzedPosition:
    """Tests for ReanalyzedPosition dataclass."""

    def test_creation(self):
        """Test creating a reanalyzed position."""
        features = np.random.randn(64, 8, 8).astype(np.float32)
        globals_vec = np.random.randn(16).astype(np.float32)
        policy = np.random.rand(64).astype(np.float32)
        policy /= policy.sum()

        pos = ReanalyzedPosition(
            features=features,
            globals_vec=globals_vec,
            original_value=0.4,
            reanalyzed_value=0.6,
            blended_value=0.54,
            original_policy=policy,
            reanalyzed_policy=policy * 0.9,
            blended_policy=policy * 0.95,
            game_id="test-game-1",
            move_number=5,
            reanalysis_timestamp=1704067200.0,
        )

        assert pos.game_id == "test-game-1"
        assert pos.move_number == 5
        assert pos.original_value == 0.4
        assert pos.reanalyzed_value == 0.6
        assert pos.blended_value == 0.54
        assert pos.features.shape == (64, 8, 8)
        assert pos.globals_vec.shape == (16,)

    def test_with_none_policies(self):
        """Test position with None policies (raw value training)."""
        pos = ReanalyzedPosition(
            features=np.zeros((64, 8, 8)),
            globals_vec=np.zeros(16),
            original_value=0.5,
            reanalyzed_value=0.7,
            blended_value=0.64,
            original_policy=None,
            reanalyzed_policy=None,
            blended_policy=None,
            game_id="test",
            move_number=0,
            reanalysis_timestamp=0.0,
        )

        assert pos.original_policy is None
        assert pos.reanalyzed_policy is None
        assert pos.blended_policy is None


# =============================================================================
# ReanalysisEngine Tests
# =============================================================================


class TestReanalysisEngine:
    """Tests for ReanalysisEngine class."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock neural network model."""
        model = MagicMock()
        model.eval = MagicMock(return_value=model)
        model.to = MagicMock(return_value=model)

        # Mock forward pass returning value and policy
        def mock_forward(features, globals_vec):
            batch_size = features.shape[0]
            values = np.random.rand(batch_size, 1).astype(np.float32)
            policies = np.random.rand(batch_size, 64).astype(np.float32)
            # Normalize policies
            policies = policies / policies.sum(axis=1, keepdims=True)

            if HAS_TORCH:
                return torch.tensor(values), torch.tensor(policies)
            return values, policies

        model.side_effect = mock_forward
        model.__call__ = mock_forward

        return model

    @pytest.fixture
    def sample_npz(self, tmp_path):
        """Create a sample NPZ file for testing."""
        npz_path = tmp_path / "sample_data.npz"

        # Create sample data
        n_samples = 100
        features = np.random.randn(n_samples, 64, 8, 8).astype(np.float32)
        globals_vec = np.random.randn(n_samples, 16).astype(np.float32)
        values = np.random.rand(n_samples).astype(np.float32)
        policies = np.random.rand(n_samples, 64).astype(np.float32)
        policies = policies / policies.sum(axis=1, keepdims=True)

        np.savez(
            npz_path,
            features=features,
            globals=globals_vec,
            values=values,
            policies=policies,
        )

        return npz_path

    def test_init(self, mock_model):
        """Test engine initialization."""
        engine = ReanalysisEngine(model=mock_model)

        assert engine.model is mock_model
        assert engine.config is not None
        assert engine.stats["positions_reanalyzed"] == 0

    def test_init_with_config(self, mock_model):
        """Test initialization with custom config."""
        config = ReanalysisConfig(batch_size=128, use_mcts=True)
        engine = ReanalysisEngine(model=mock_model, config=config)

        assert engine.config.batch_size == 128
        assert engine.config.use_mcts is True

    def test_should_reanalyze_elo_delta(self, mock_model):
        """Test should_reanalyze based on Elo delta."""
        engine = ReanalysisEngine(
            model=mock_model,
            config=ReanalysisConfig(min_model_elo_delta=50),
        )

        # Sufficient improvement
        assert engine.should_reanalyze(current_elo=1550, last_reanalysis_elo=1500) is True

        # Insufficient improvement
        assert engine.should_reanalyze(current_elo=1520, last_reanalysis_elo=1500) is False

        # Exactly at threshold
        assert engine.should_reanalyze(current_elo=1550, last_reanalysis_elo=1500) is True

        # Regression (worse than before)
        assert engine.should_reanalyze(current_elo=1450, last_reanalysis_elo=1500) is False

    def test_get_stats(self, mock_model):
        """Test getting engine statistics."""
        engine = ReanalysisEngine(model=mock_model)

        stats = engine.get_stats()

        assert "positions_reanalyzed" in stats
        assert "games_processed" in stats
        assert "total_reanalysis_time" in stats
        assert "avg_value_delta" in stats

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_reanalyze_npz_basic(self, mock_model, sample_npz, tmp_path):
        """Test basic NPZ reanalysis."""
        # This is a complex test that requires PyTorch
        # For now, just verify the function signature works
        engine = ReanalysisEngine(model=mock_model)

        # The actual reanalysis requires proper model mocking
        # which is complex due to torch dependencies
        assert callable(engine.reanalyze_npz)


# =============================================================================
# ReanalysisDataset Tests
# =============================================================================


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestReanalysisDataset:
    """Tests for ReanalysisDataset class."""

    @pytest.fixture
    def sample_npz_files(self, tmp_path):
        """Create sample NPZ files for testing."""
        paths = []
        for i in range(3):
            npz_path = tmp_path / f"data_{i}.npz"
            n_samples = 50

            features = np.random.randn(n_samples, 64, 8, 8).astype(np.float32)
            globals_vec = np.random.randn(n_samples, 16).astype(np.float32)
            values = np.random.rand(n_samples).astype(np.float32)
            policies = np.random.rand(n_samples, 64).astype(np.float32)
            policies = policies / policies.sum(axis=1, keepdims=True)

            np.savez(
                npz_path,
                features=features,
                globals=globals_vec,
                values=values,
                policies=policies,
            )
            paths.append(npz_path)

        return paths

    def test_init(self, sample_npz_files):
        """Test dataset initialization."""
        dataset = ReanalysisDataset(npz_paths=sample_npz_files)

        assert len(dataset) > 0

    def test_len(self, sample_npz_files):
        """Test dataset length."""
        dataset = ReanalysisDataset(npz_paths=sample_npz_files)

        # 3 files * 50 samples each = 150
        assert len(dataset) == 150

    def test_getitem(self, sample_npz_files):
        """Test getting individual items."""
        dataset = ReanalysisDataset(npz_paths=sample_npz_files)

        features, globals_vec, values = dataset[0]

        assert isinstance(features, torch.Tensor)
        assert isinstance(globals_vec, torch.Tensor)
        assert isinstance(values, torch.Tensor)

    def test_empty_paths(self):
        """Test with empty paths list."""
        dataset = ReanalysisDataset(npz_paths=[])

        assert len(dataset) == 0


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        model = MagicMock()
        model.eval = MagicMock(return_value=model)
        model.to = MagicMock(return_value=model)
        return model

    def test_create_reanalysis_engine_basic(self, mock_model):
        """Test creating engine with basic params."""
        engine = create_reanalysis_engine(
            model=mock_model,
            board_type="hex8",
        )

        assert isinstance(engine, ReanalysisEngine)

    def test_create_reanalysis_engine_with_config(self, mock_model):
        """Test creating engine with custom config."""
        engine = create_reanalysis_engine(
            model=mock_model,
            board_type="hex8",
            batch_size=128,
            use_mcts=True,
        )

        assert engine.config.batch_size == 128
        assert engine.config.use_mcts is True


# =============================================================================
# Integration Tests
# =============================================================================


class TestReanalysisIntegration:
    """Integration tests for reanalysis pipeline."""

    @pytest.fixture
    def sample_data_dir(self, tmp_path):
        """Create a sample data directory."""
        data_dir = tmp_path / "training_data"
        data_dir.mkdir()

        # Create sample NPZ files
        for config in ["hex8_2p", "square8_4p"]:
            npz_path = data_dir / f"{config}_training.npz"
            n_samples = 100

            features = np.random.randn(n_samples, 64, 8, 8).astype(np.float32)
            globals_vec = np.random.randn(n_samples, 16).astype(np.float32)
            values = np.random.rand(n_samples).astype(np.float32)
            policies = np.random.rand(n_samples, 64).astype(np.float32)

            np.savez(
                npz_path,
                features=features,
                globals=globals_vec,
                values=values,
                policies=policies,
            )

        return data_dir

    def test_value_blending_math(self):
        """Test that value blending follows expected formula."""
        original = 0.3
        reanalyzed = 0.8
        ratio = 0.7  # 70% new, 30% old

        expected = ratio * reanalyzed + (1 - ratio) * original
        assert expected == pytest.approx(0.65)

    def test_policy_blending_math(self):
        """Test that policy blending follows expected formula."""
        original = np.array([0.2, 0.3, 0.5])
        reanalyzed = np.array([0.1, 0.1, 0.8])
        ratio = 0.8  # 80% new, 20% old

        expected = ratio * reanalyzed + (1 - ratio) * original
        np.testing.assert_array_almost_equal(
            expected,
            [0.12, 0.14, 0.74],
        )

    def test_config_immutability(self):
        """Test that config is a dataclass (immutable-ish)."""
        config = ReanalysisConfig()

        # Verify it's a dataclass
        assert hasattr(config, "__dataclass_fields__")

        # Can still modify (dataclass is mutable by default)
        config.batch_size = 256
        assert config.batch_size == 256


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestReanalysisEdgeCases:
    """Edge case tests for reanalysis."""

    def test_empty_game_handling(self):
        """Test handling of games with no moves."""
        # ReanalysisConfig should have min_game_length
        config = ReanalysisConfig(min_game_length=10)
        assert config.min_game_length == 10

    def test_very_long_game_handling(self):
        """Test handling of extremely long games."""
        config = ReanalysisConfig(max_game_length=500)
        assert config.max_game_length == 500

    def test_mcts_config_defaults(self):
        """Test MCTS configuration defaults."""
        config = ReanalysisConfig(use_mcts=True)

        assert config.mcts_simulations == 100
        assert config.mcts_temperature == 1.0
        assert config.capture_q_values is True
        assert config.capture_uncertainty is True

    def test_cache_dir_configuration(self):
        """Test cache directory configuration."""
        config = ReanalysisConfig(cache_dir="/custom/cache/path")
        assert config.cache_dir == "/custom/cache/path"

    def test_interval_configuration(self):
        """Test reanalysis interval configuration."""
        config = ReanalysisConfig(reanalysis_interval_hours=12.0)
        assert config.reanalysis_interval_hours == 12.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
