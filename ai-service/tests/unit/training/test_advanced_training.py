"""Tests for app/training/advanced_training.py.

Tests cover:
- LRFinderResult dataclass
- LRFinder class (learning rate finder)
- GradientCheckpointing class
- OpponentStats dataclass
- PFSPOpponentPool class (PFSP opponent management)
- PlateauConfig dataclass
- CMAESAutoTuner class
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from app.training.advanced_training import (
    CMAESAutoTuner,
    GradientCheckpointing,
    LRFinder,
    LRFinderResult,
    OpponentStats,
    PFSPOpponentPool,
    PlateauConfig,
)


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class TestLRFinderResult:
    """Tests for LRFinderResult dataclass."""

    def test_creation(self):
        """Test creating LRFinderResult."""
        result = LRFinderResult(
            lrs=[1e-5, 1e-4, 1e-3],
            losses=[1.0, 0.8, 0.5],
            suggested_lr=1e-4,
            min_lr=1e-7,
            max_lr=10.0,
            best_lr=1e-3,
            steepest_lr=1e-4,
        )

        assert result.suggested_lr == 1e-4
        assert len(result.lrs) == 3
        assert result.best_lr == 1e-3

    def test_empty_result(self):
        """Test creating empty LRFinderResult."""
        result = LRFinderResult(
            lrs=[],
            losses=[],
            suggested_lr=1e-3,
            min_lr=1e-7,
            max_lr=10.0,
            best_lr=1e-3,
            steepest_lr=1e-3,
        )

        assert len(result.lrs) == 0
        assert len(result.losses) == 0


class TestLRFinder:
    """Tests for LRFinder class."""

    @pytest.fixture
    def model(self):
        """Create a simple model."""
        return SimpleModel()

    @pytest.fixture
    def optimizer(self, model):
        """Create optimizer."""
        return torch.optim.SGD(model.parameters(), lr=1e-5)

    @pytest.fixture
    def criterion(self):
        """Create criterion."""
        return nn.CrossEntropyLoss()

    @pytest.fixture
    def finder(self, model, optimizer, criterion):
        """Create LRFinder."""
        return LRFinder(model, optimizer, criterion)

    def test_initialization(self, finder, model, optimizer):
        """Test LRFinder initialization."""
        assert finder.model is model
        assert finder.optimizer is optimizer
        assert finder._initial_model_state is not None
        assert finder._initial_optimizer_state is not None

    def test_analyze_results_empty(self, finder):
        """Test _analyze_results with empty data."""
        result = finder._analyze_results([], [], 1e-7, 10.0)

        assert result.suggested_lr == 1e-3
        assert len(result.lrs) == 0
        assert len(result.losses) == 0

    def test_analyze_results_finds_best(self, finder):
        """Test _analyze_results finds best LR."""
        lrs = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        losses = [1.0, 0.8, 0.3, 0.4, 2.0]

        result = finder._analyze_results(lrs, losses, 1e-7, 10.0)

        # Best LR should be at minimum loss (1e-3)
        assert result.best_lr == 1e-3

    def test_analyze_results_suggests_lower_lr(self, finder):
        """Test suggested LR is lower than steepest."""
        lrs = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        losses = [1.0, 0.9, 0.7, 0.3, 0.5, 5.0]

        result = finder._analyze_results(lrs, losses, 1e-7, 10.0)

        # Suggested should be one order of magnitude below steepest
        assert result.suggested_lr < result.steepest_lr

    def test_train_step_tensor_batch(self, finder, model):
        """Test _train_step with tensor batch."""
        model.train()
        batch = (
            torch.randn(4, 10),
            torch.randint(0, 5, (4,)),
        )

        loss = finder._train_step(batch)

        assert isinstance(loss, float)
        assert loss >= 0

    def test_train_step_dict_batch(self, finder, model):
        """Test _train_step with dict batch."""
        model.train()
        batch = {
            'features': torch.randn(4, 10),
            'labels': torch.randint(0, 5, (4,)),
        }

        loss = finder._train_step(batch)

        assert isinstance(loss, float)

    def test_restore_original_state(self, model, optimizer, criterion):
        """Test that original state is stored for restoration."""
        original_weight = model.fc1.weight.clone()

        finder = LRFinder(model, optimizer, criterion)

        # Modify model
        model.fc1.weight.data.fill_(0)

        # Verify state was captured before modification
        initial_weight = finder._initial_model_state['fc1.weight']
        assert torch.allclose(initial_weight, original_weight)


class TestGradientCheckpointing:
    """Tests for GradientCheckpointing class."""

    @pytest.fixture
    def model(self):
        """Create a simple model."""
        return SimpleModel()

    def test_initialization(self, model):
        """Test GradientCheckpointing initialization."""
        checkpointing = GradientCheckpointing(model)

        assert checkpointing.model is model
        assert not checkpointing.is_enabled
        assert checkpointing.checkpoint_layers is None

    def test_initialization_with_layers(self, model):
        """Test initialization with specific layers."""
        checkpointing = GradientCheckpointing(
            model,
            checkpoint_layers=['fc1', 'fc2'],
        )

        assert checkpointing.checkpoint_layers == ['fc1', 'fc2']

    def test_enable_disable(self, model):
        """Test enable and disable."""
        checkpointing = GradientCheckpointing(model)

        assert not checkpointing.is_enabled

        checkpointing.enable()
        assert checkpointing.is_enabled

        checkpointing.disable()
        assert not checkpointing.is_enabled

    def test_enable_idempotent(self, model):
        """Test that multiple enable calls are safe."""
        checkpointing = GradientCheckpointing(model)

        checkpointing.enable()
        checkpointing.enable()  # Should not raise

        assert checkpointing.is_enabled

    def test_disable_idempotent(self, model):
        """Test that multiple disable calls are safe."""
        checkpointing = GradientCheckpointing(model)

        checkpointing.enable()
        checkpointing.disable()
        checkpointing.disable()  # Should not raise

        assert not checkpointing.is_enabled

    def test_find_checkpoint_layers_auto(self, model):
        """Test auto-detection of checkpoint layers."""
        checkpointing = GradientCheckpointing(model)

        layers = checkpointing._find_checkpoint_layers()

        # Should find some layers
        assert isinstance(layers, list)

    def test_is_enabled_property(self, model):
        """Test is_enabled property."""
        checkpointing = GradientCheckpointing(model)

        assert checkpointing.is_enabled is False

        checkpointing._enabled = True
        assert checkpointing.is_enabled is True


class TestOpponentStats:
    """Tests for OpponentStats dataclass."""

    def test_default_values(self):
        """Test default values."""
        stats = OpponentStats(
            model_path="/models/gen1.pth",
            model_name="gen1",
        )

        assert stats.elo == 1500.0
        assert stats.games_played == 0
        assert stats.wins == 0
        assert stats.losses == 0
        assert stats.draws == 0
        assert stats.last_played is None
        assert stats.generation == 0
        assert stats.priority_score == 1.0

    def test_custom_values(self):
        """Test custom values."""
        now = datetime.now()
        stats = OpponentStats(
            model_path="/models/gen2.pth",
            model_name="gen2",
            elo=1650.0,
            games_played=50,
            wins=30,
            losses=15,
            draws=5,
            last_played=now,
            generation=2,
            priority_score=0.7,
        )

        assert stats.elo == 1650.0
        assert stats.games_played == 50
        assert stats.wins == 30
        assert stats.last_played == now


class TestPFSPOpponentPool:
    """Tests for PFSPOpponentPool class."""

    @pytest.fixture
    def pool(self):
        """Create empty pool."""
        return PFSPOpponentPool(max_pool_size=5)

    def test_initialization(self, pool):
        """Test pool initialization."""
        assert pool.max_pool_size == 5
        assert len(pool._opponents) == 0

    def test_add_opponent(self, pool):
        """Test adding opponent."""
        pool.add_opponent("/models/gen1.pth", elo=1500, generation=1, name="gen1")

        assert len(pool._opponents) == 1
        assert "/models/gen1.pth" in pool._opponents
        assert pool._opponents["/models/gen1.pth"].elo == 1500

    def test_add_duplicate_opponent(self, pool):
        """Test adding duplicate opponent is idempotent."""
        pool.add_opponent("/models/gen1.pth", elo=1500)
        pool.add_opponent("/models/gen1.pth", elo=1600)  # Should be ignored

        assert len(pool._opponents) == 1
        assert pool._opponents["/models/gen1.pth"].elo == 1500  # Original value

    def test_remove_opponent(self, pool):
        """Test removing opponent."""
        pool.add_opponent("/models/gen1.pth")
        pool.remove_opponent("/models/gen1.pth")

        assert len(pool._opponents) == 0

    def test_remove_nonexistent_opponent(self, pool):
        """Test removing nonexistent opponent is safe."""
        pool.remove_opponent("/models/nonexistent.pth")  # Should not raise

    def test_eviction_at_capacity(self, pool):
        """Test eviction when at capacity."""
        # Fill pool
        for i in range(5):
            pool.add_opponent(f"/models/gen{i}.pth", generation=i)

        assert len(pool._opponents) == 5

        # Add one more - should trigger eviction
        pool.add_opponent("/models/gen5.pth", generation=5)

        assert len(pool._opponents) == 5
        assert "/models/gen5.pth" in pool._opponents

    def test_sample_opponent_empty_pool(self, pool):
        """Test sampling from empty pool."""
        result = pool.sample_opponent()

        assert result is None

    def test_sample_opponent_uniform(self, pool):
        """Test uniform sampling."""
        pool.add_opponent("/models/gen1.pth")
        pool.add_opponent("/models/gen2.pth")

        result = pool.sample_opponent(strategy="uniform")

        assert result is not None
        assert result.model_path in ["/models/gen1.pth", "/models/gen2.pth"]

    def test_sample_opponent_elo_based(self, pool):
        """Test elo-based sampling."""
        pool.add_opponent("/models/gen1.pth", elo=1400)
        pool.add_opponent("/models/gen2.pth", elo=1500)
        pool.add_opponent("/models/gen3.pth", elo=2000)

        result = pool.sample_opponent(current_elo=1500, strategy="elo_based")

        assert result is not None

    def test_sample_opponent_pfsp(self, pool):
        """Test PFSP sampling."""
        pool.add_opponent("/models/gen1.pth", elo=1500)
        pool.add_opponent("/models/gen2.pth", elo=1600)

        result = pool.sample_opponent(strategy="pfsp")

        assert result is not None

    def test_sample_with_exclude(self, pool):
        """Test sampling with exclusions."""
        pool.add_opponent("/models/gen1.pth")
        pool.add_opponent("/models/gen2.pth")

        result = pool.sample_opponent(exclude=["/models/gen1.pth"])

        assert result is not None
        assert result.model_path == "/models/gen2.pth"

    def test_update_stats_win(self, pool):
        """Test updating stats for win."""
        pool.add_opponent("/models/gen1.pth")

        pool.update_stats("/models/gen1.pth", won=True)

        opp = pool._opponents["/models/gen1.pth"]
        assert opp.games_played == 1
        assert opp.wins == 1
        assert opp.losses == 0
        assert opp.last_played is not None

    def test_update_stats_loss(self, pool):
        """Test updating stats for loss."""
        pool.add_opponent("/models/gen1.pth")

        pool.update_stats("/models/gen1.pth", won=False)

        opp = pool._opponents["/models/gen1.pth"]
        assert opp.games_played == 1
        assert opp.wins == 0
        assert opp.losses == 1

    def test_update_stats_draw(self, pool):
        """Test updating stats for draw."""
        pool.add_opponent("/models/gen1.pth")

        pool.update_stats("/models/gen1.pth", won=False, drew=True)

        opp = pool._opponents["/models/gen1.pth"]
        assert opp.games_played == 1
        assert opp.draws == 1
        assert opp.losses == 0  # Draw doesn't count as loss

    def test_update_stats_elo_change(self, pool):
        """Test updating Elo."""
        pool.add_opponent("/models/gen1.pth", elo=1500)

        pool.update_stats("/models/gen1.pth", won=True, elo_change=16)

        assert pool._opponents["/models/gen1.pth"].elo == 1516

    def test_get_pool_stats_empty(self, pool):
        """Test pool stats for empty pool."""
        stats = pool.get_pool_stats()

        assert stats['size'] == 0

    def test_get_pool_stats(self, pool):
        """Test pool stats."""
        pool.add_opponent("/models/gen1.pth", elo=1400)
        pool.add_opponent("/models/gen2.pth", elo=1600)

        stats = pool.get_pool_stats()

        assert stats['size'] == 2
        assert stats['avg_elo'] == 1500
        assert stats['min_elo'] == 1400
        assert stats['max_elo'] == 1600

    def test_get_opponents(self, pool):
        """Test getting all opponents."""
        pool.add_opponent("/models/gen1.pth")
        pool.add_opponent("/models/gen2.pth")

        opponents = pool.get_opponents()

        assert len(opponents) == 2

    def test_save_and_load_pool(self, pool):
        """Test saving and loading pool."""
        pool.add_opponent("/models/gen1.pth", elo=1550, generation=1)
        pool.add_opponent("/models/gen2.pth", elo=1600, generation=2)

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name

        try:
            pool.save_pool(path)

            # Create new pool and load
            new_pool = PFSPOpponentPool()
            new_pool.load_pool(path)

            assert len(new_pool._opponents) == 2
            assert new_pool._opponents["/models/gen1.pth"].elo == 1550
            assert new_pool._opponents["/models/gen2.pth"].generation == 2
        finally:
            Path(path).unlink(missing_ok=True)


class TestPlateauConfig:
    """Tests for PlateauConfig dataclass."""

    def test_default_values(self):
        """Test default values."""
        config = PlateauConfig()

        assert config.patience == 10
        assert config.min_delta == 0.01
        assert config.metric == "elo"
        assert config.lookback == 20

    def test_custom_values(self):
        """Test custom values."""
        config = PlateauConfig(
            patience=5,
            min_delta=0.05,
            metric="loss",
            lookback=30,
        )

        assert config.patience == 5
        assert config.min_delta == 0.05
        assert config.metric == "loss"
        assert config.lookback == 30


class TestCMAESAutoTuner:
    """Tests for CMAESAutoTuner class."""

    @pytest.fixture
    def tuner(self):
        """Create CMAESAutoTuner."""
        return CMAESAutoTuner(
            cmaes_script="scripts/run_cmaes.py",
            board_type="square8",
            min_epochs_between_tuning=10,
            max_auto_tunes=3,
        )

    def test_initialization(self, tuner):
        """Test initialization."""
        assert tuner.board_type == "square8"
        assert tuner.min_epochs_between_tuning == 10
        assert tuner.max_auto_tunes == 3
        assert tuner._current_epoch == 0
        assert tuner._tune_count == 0

    def test_step_with_elo(self, tuner):
        """Test step with Elo metric."""
        tuner.step(current_elo=1500)

        assert tuner._current_epoch == 1
        assert len(tuner._metric_history) == 1

    def test_step_with_loss(self):
        """Test step with loss metric."""
        config = PlateauConfig(metric="loss")
        tuner = CMAESAutoTuner(plateau_config=config)

        tuner.step(current_loss=0.5)

        assert tuner._current_epoch == 1
        # Loss is negated (lower is better)
        assert next(iter(tuner._metric_history)) == -0.5

    def test_step_with_win_rate(self):
        """Test step with win rate metric."""
        config = PlateauConfig(metric="win_rate")
        tuner = CMAESAutoTuner(plateau_config=config)

        tuner.step(current_win_rate=0.65)

        assert tuner._current_epoch == 1
        assert next(iter(tuner._metric_history)) == 0.65

    def test_improvement_resets_counter(self, tuner):
        """Test that improvement resets epochs_without_improvement."""
        tuner.step(current_elo=1500)
        assert tuner._epochs_without_improvement == 0

        tuner.step(current_elo=1490)  # No improvement
        assert tuner._epochs_without_improvement == 1

        tuner.step(current_elo=1520)  # Improvement
        assert tuner._epochs_without_improvement == 0
        assert tuner._best_metric == 1520

    def test_no_improvement_increments_counter(self, tuner):
        """Test that no improvement increments counter."""
        tuner.step(current_elo=1500)
        tuner.step(current_elo=1499)  # Slightly worse
        tuner.step(current_elo=1500)  # Same (not min_delta better)

        assert tuner._epochs_without_improvement == 2

    def test_should_tune_respects_patience(self, tuner):
        """Test should_tune respects patience."""
        # Not enough epochs without improvement
        for i in range(5):
            tuner.step(current_elo=1500 - i)

        assert not tuner.should_tune()

    def test_should_tune_respects_min_epochs(self, tuner):
        """Test should_tune respects min_epochs_between_tuning."""
        # Trigger plateau
        tuner._epochs_without_improvement = tuner.plateau_config.patience + 1
        tuner._current_epoch = 15

        # Simulate a recent tune
        tuner._last_tune_epoch = 10  # Just 5 epochs ago (< min_epochs_between_tuning of 10)

        assert not tuner.should_tune()

    def test_should_tune_respects_max_tunes(self, tuner):
        """Test should_tune respects max_auto_tunes."""
        tuner._tune_count = tuner.max_auto_tunes

        assert not tuner.should_tune()

    def test_should_tune_when_tuning(self, tuner):
        """Test should_tune returns False when already tuning."""
        tuner._is_tuning = True

        assert not tuner.should_tune()

    def test_should_tune_positive(self):
        """Test should_tune returns True under right conditions."""
        config = PlateauConfig(patience=3)
        tuner = CMAESAutoTuner(
            plateau_config=config,
            min_epochs_between_tuning=2,
        )

        # Advance epochs and create plateau
        for _i in range(10):
            tuner.step(current_elo=1500)  # No improvement

        assert tuner.should_tune()


class TestPFSPPrioritySampling:
    """Tests for PFSP priority-based sampling logic."""

    def test_prioritizes_hard_opponents(self):
        """Test that PFSP prioritizes opponents we lose to."""
        pool = PFSPOpponentPool(
            hard_opponent_weight=1.0,
            diversity_weight=0.0,
            recency_weight=0.0,
            min_games_for_priority=2,
        )

        # Add two opponents with different win rates against them
        pool.add_opponent("/models/easy.pth")
        pool.add_opponent("/models/hard.pth")

        # Easy opponent: we win often
        for _ in range(10):
            pool.update_stats("/models/easy.pth", won=True)

        # Hard opponent: we lose often
        for _ in range(10):
            pool.update_stats("/models/hard.pth", won=False)

        # Sample many times and check distribution
        selections = {"easy": 0, "hard": 0}
        for _ in range(100):
            opp = pool.sample_opponent(strategy="pfsp")
            if "easy" in opp.model_path:
                selections["easy"] += 1
            else:
                selections["hard"] += 1

        # Hard opponent should be selected more often
        assert selections["hard"] > selections["easy"]

    def test_diversity_weights(self):
        """Test diversity weight affects sampling."""
        pool = PFSPOpponentPool(
            hard_opponent_weight=0.0,
            diversity_weight=1.0,
            recency_weight=0.0,
        )

        pool.add_opponent("/models/gen1.pth")
        pool.add_opponent("/models/gen2.pth")

        # Play many games against gen1
        for _ in range(20):
            pool.update_stats("/models/gen1.pth", won=True)

        # Sample many times
        selections = {"gen1": 0, "gen2": 0}
        for _ in range(100):
            opp = pool.sample_opponent(strategy="pfsp")
            if "gen1" in opp.model_path:
                selections["gen1"] += 1
            else:
                selections["gen2"] += 1

        # gen2 should be selected more often (less played = higher diversity)
        assert selections["gen2"] > selections["gen1"]


class TestLRFinderIntegration:
    """Integration tests for LRFinder."""

    def test_full_lr_find_cycle(self):
        """Test complete LR finding cycle."""
        model = SimpleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
        criterion = nn.CrossEntropyLoss()

        finder = LRFinder(model, optimizer, criterion)

        # Create simple data loader
        data = [
            (torch.randn(4, 10), torch.randint(0, 5, (4,)))
            for _ in range(20)
        ]

        class SimpleLoader:
            def __init__(self, data):
                self.data = data

            def __iter__(self):
                return iter(self.data)

            def __len__(self):
                return len(self.data)

        loader = SimpleLoader(data)

        result = finder.range_test(
            loader,
            min_lr=1e-6,
            max_lr=1.0,
            num_iter=10,
        )

        # Should have results
        assert len(result.lrs) > 0
        assert len(result.losses) > 0
        assert result.suggested_lr > 0
        assert result.best_lr > 0
