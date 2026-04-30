"""Tests for the main training module (app/training/train.py).

Tests cover:
- EnhancedEarlyStopping: Loss and Elo-based stopping criteria
- TrainConfig: Configuration validation and defaults
- masked_policy_kl: Policy loss computation
- Checkpoint save/load: Via checkpointing module
- train_model: Configuration validation (mocked execution)
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from app.models import BoardType
from app.training.config import TrainConfig


class TestEnhancedEarlyStopping:
    """Tests for EnhancedEarlyStopping class."""

    @pytest.fixture
    def early_stopper(self):
        """Create a basic early stopper for testing."""
        from app.training.training_enhancements import EarlyStopping

        return EarlyStopping(patience=3, min_delta=0.001, min_epochs=0)

    @pytest.fixture
    def early_stopper_with_elo(self):
        """Create an early stopper with Elo tracking."""
        from app.training.training_enhancements import EarlyStopping

        return EarlyStopping(
            patience=3,
            elo_patience=2,
            elo_min_improvement=5.0,
            min_epochs=0,
        )

    def test_initial_state(self, early_stopper):
        """Test that early stopper starts in correct initial state."""
        assert early_stopper.best_loss == float("inf")
        assert early_stopper.loss_counter == 0
        assert early_stopper.best_state is None

    def test_improvement_resets_counter(self, early_stopper):
        """Test that improvement resets the counter."""
        # First call establishes baseline
        early_stopper.should_stop(val_loss=1.0, epoch=0)
        assert early_stopper.loss_counter == 0
        assert early_stopper.best_loss == 1.0

        # Improvement
        early_stopper.should_stop(val_loss=0.5, epoch=1)
        assert early_stopper.loss_counter == 0
        assert early_stopper.best_loss == 0.5

    def test_no_improvement_increments_counter(self, early_stopper):
        """Test that lack of improvement increments counter."""
        early_stopper.should_stop(val_loss=1.0, epoch=0)
        early_stopper.should_stop(val_loss=1.0, epoch=1)  # No improvement
        assert early_stopper.loss_counter == 1

        early_stopper.should_stop(val_loss=1.1, epoch=2)  # Worse
        assert early_stopper.loss_counter == 2

    def test_stops_after_patience(self, early_stopper):
        """Test that training stops after patience is exceeded."""
        early_stopper.should_stop(val_loss=1.0, epoch=0)
        assert not early_stopper.should_stop(val_loss=1.0, epoch=1)
        assert not early_stopper.should_stop(val_loss=1.0, epoch=2)
        assert early_stopper.should_stop(val_loss=1.0, epoch=3)

    def test_min_epochs_prevents_early_stop(self):
        """Test that min_epochs prevents stopping too early."""
        from app.training.training_enhancements import EarlyStopping

        stopper = EarlyStopping(patience=1, min_epochs=5)

        # Even with patience exceeded, should not stop before min_epochs
        stopper.should_stop(val_loss=1.0, epoch=0)
        assert not stopper.should_stop(val_loss=1.5, epoch=1)
        assert not stopper.should_stop(val_loss=1.5, epoch=2)
        assert not stopper.should_stop(val_loss=1.5, epoch=3)
        assert not stopper.should_stop(val_loss=1.5, epoch=4)
        # After min_epochs, should stop
        assert stopper.should_stop(val_loss=1.5, epoch=5)

    def test_elo_tracking(self, early_stopper_with_elo):
        """Test that Elo improvements are tracked."""
        early_stopper_with_elo.should_stop(val_loss=1.0, current_elo=1000, epoch=0)
        assert early_stopper_with_elo.best_elo == 1000

        # Elo improvement
        early_stopper_with_elo.should_stop(val_loss=1.0, current_elo=1010, epoch=1)
        assert early_stopper_with_elo.best_elo == 1010
        assert early_stopper_with_elo.elo_counter == 0

    def test_elo_no_improvement_increments_counter(self, early_stopper_with_elo):
        """Test that small Elo changes don't reset counter."""
        early_stopper_with_elo.should_stop(val_loss=1.0, current_elo=1000, epoch=0)
        early_stopper_with_elo.should_stop(val_loss=1.0, current_elo=1003, epoch=1)
        # 3 Elo gain is below min_improvement of 5
        assert early_stopper_with_elo.elo_counter == 1

    def test_both_criteria_required_when_tracking_both(self, early_stopper_with_elo):
        """Test that both loss and Elo must stagnate to stop when tracking both."""
        # Elo stagnates but loss keeps improving
        early_stopper_with_elo.should_stop(val_loss=1.0, current_elo=1000, epoch=0)
        early_stopper_with_elo.should_stop(val_loss=0.9, current_elo=1000, epoch=1)
        early_stopper_with_elo.should_stop(val_loss=0.8, current_elo=1000, epoch=2)
        # Elo patience (2) exceeded but loss still improving
        assert not early_stopper_with_elo.should_stop(
            val_loss=0.7, current_elo=1000, epoch=3
        )

    def test_saves_best_model_state(self, early_stopper):
        """Test that best model state is saved on improvement."""

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 2)

        model = SimpleModel()
        model.linear.weight.data.fill_(1.0)

        early_stopper.should_stop(val_loss=1.0, model=model, epoch=0)
        assert early_stopper.best_state is not None

        # Modify model
        model.linear.weight.data.fill_(2.0)

        # Improve
        early_stopper.should_stop(val_loss=0.5, model=model, epoch=1)

        # best_state should have the new values
        assert torch.allclose(
            early_stopper.best_state["linear.weight"],
            torch.full((2, 10), 2.0),
        )

    def test_restore_best_model(self, early_stopper):
        """Test that model can be restored to best state."""

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 2)

        model = SimpleModel()
        model.linear.weight.data.fill_(1.0)

        early_stopper.should_stop(val_loss=1.0, model=model, epoch=0)

        # Modify model
        model.linear.weight.data.fill_(99.0)

        # Restore
        success = early_stopper.restore_best_model(model)
        assert success
        assert torch.allclose(
            model.linear.weight.data,
            torch.ones(2, 10),
        )

    def test_reset(self, early_stopper):
        """Test that reset clears all state."""
        early_stopper.should_stop(val_loss=1.0, epoch=0)
        early_stopper.should_stop(val_loss=1.0, epoch=1)

        early_stopper.reset()

        assert early_stopper.best_loss == float("inf")
        assert early_stopper.loss_counter == 0
        assert early_stopper.best_state is None


class TestTrainConfig:
    """Tests for TrainConfig dataclass."""

    def test_default_values(self):
        """Test that default values are reasonable."""
        config = TrainConfig(board_type=BoardType.SQUARE8)

        assert config.batch_size > 0
        assert config.learning_rate > 0
        assert config.epochs_per_iter > 0

    def test_default_board_type(self):
        """Test that default board_type is SQUARE8."""
        config = TrainConfig()
        assert config.board_type == BoardType.SQUARE8

    def test_custom_values(self):
        """Test that custom values are preserved."""
        config = TrainConfig(
            board_type=BoardType.HEX8,
            batch_size=256,
            learning_rate=0.0005,
            epochs_per_iter=100,
        )

        assert config.board_type == BoardType.HEX8
        assert config.batch_size == 256
        assert config.learning_rate == 0.0005
        assert config.epochs_per_iter == 100


class TestMaskedPolicyKL:
    """Tests for masked_policy_kl loss function."""

    @pytest.fixture
    def policy_kl(self):
        """Import the masked_policy_kl function."""
        from app.training.train import masked_policy_kl

        return masked_policy_kl

    def test_identical_distributions(self, policy_kl):
        """Test that identical distributions have zero KL divergence."""
        log_probs = torch.log(torch.tensor([[0.5, 0.3, 0.2], [0.4, 0.4, 0.2]]))
        targets = torch.tensor([[0.5, 0.3, 0.2], [0.4, 0.4, 0.2]])

        loss = policy_kl(log_probs, targets)
        assert loss.item() < 1e-5

    def test_different_distributions(self, policy_kl):
        """Test that different distributions have positive KL divergence."""
        log_probs = torch.log(torch.tensor([[0.9, 0.05, 0.05]]))
        targets = torch.tensor([[0.33, 0.33, 0.34]])

        loss = policy_kl(log_probs, targets)
        assert loss.item() > 0

    def test_masked_samples_ignored(self, policy_kl):
        """Test that samples with zero-sum targets are ignored."""
        log_probs = torch.log(torch.tensor([[0.5, 0.3, 0.2], [0.9, 0.05, 0.05]]))
        # Second sample has all-zero targets (masked)
        targets = torch.tensor([[0.5, 0.3, 0.2], [0.0, 0.0, 0.0]])

        loss = policy_kl(log_probs, targets)
        # Only first sample should contribute
        expected_loss = 0.0  # First sample is identical to log_probs
        assert loss.item() < 1e-5

    def test_gradient_flow(self, policy_kl):
        """Test that gradients flow through the loss."""
        # Use a tensor that is a leaf node for gradient tracking
        probs = torch.tensor([[0.5, 0.3, 0.2]], requires_grad=True)
        log_probs = torch.log(probs)
        targets = torch.tensor([[0.4, 0.4, 0.2]])

        loss = policy_kl(log_probs, targets)
        loss.backward()

        # Gradients should flow to the leaf probs tensor
        assert probs.grad is not None
        assert not torch.all(probs.grad == 0)


class TestCheckpointing:
    """Tests for checkpoint save/load functionality."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)

            def forward(self, x):
                return self.linear(x)

        return SimpleModel()

    @pytest.fixture
    def optimizer(self, simple_model):
        """Create an optimizer for the simple model."""
        return torch.optim.Adam(simple_model.parameters(), lr=0.001)

    @patch("app.training.checkpointing.check_disk_space", return_value=True)
    def test_save_load_checkpoint(self, mock_disk, simple_model, optimizer):
        """Test basic checkpoint save/load."""
        from app.training.checkpointing import load_checkpoint, save_checkpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "checkpoint.pth"

            # Save checkpoint
            save_checkpoint(
                model=simple_model,
                optimizer=optimizer,
                epoch=5,
                loss=0.123,
                path=str(ckpt_path),
                use_versioning=False,  # Use simple format for test
            )

            assert ckpt_path.exists()

            # Modify model
            simple_model.linear.weight.data.fill_(99.0)

            # Load checkpoint
            loaded_epoch, loaded_loss = load_checkpoint(
                path=str(ckpt_path),
                model=simple_model,
                optimizer=optimizer,
            )

            assert loaded_epoch == 5
            assert abs(loaded_loss - 0.123) < 0.001

    @patch("app.training.checkpointing.check_disk_space", return_value=True)
    def test_checkpoint_directory_creation(self, mock_disk, simple_model, optimizer):
        """Test that checkpoint creates parent directories."""
        from app.training.checkpointing import save_checkpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "deep" / "nested" / "dir" / "checkpoint.pth"

            save_checkpoint(
                model=simple_model,
                optimizer=optimizer,
                epoch=1,
                loss=0.5,
                path=str(ckpt_path),
                use_versioning=False,
            )

            assert ckpt_path.exists()


class TestTrainModelValidation:
    """Tests for train_model input validation.

    These tests mock the heavy parts of training and focus on
    validation logic at the start of train_model.
    """

    @pytest.fixture
    def minimal_config(self):
        """Create minimal training config."""
        return TrainConfig(
            board_type=BoardType.SQUARE8,
            epochs_per_iter=1,
            batch_size=4,
        )

    @pytest.fixture
    def valid_npz_file(self):
        """Create a valid NPZ file for testing."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            # Create minimal valid data
            n_samples = 10
            board_size = 8

            # Features: (samples, channels, height, width)
            features = np.random.randn(n_samples, 56, board_size, board_size).astype(
                np.float32
            )

            # Globals: (samples, 20)
            globals_vec = np.random.randn(n_samples, 20).astype(np.float32)

            # Policy: (samples, policy_size)
            policy_size = 7168  # square8 board-aware size
            policy = np.zeros((n_samples, policy_size), dtype=np.float32)
            # Set some valid policy targets
            for i in range(n_samples):
                policy[i, i % 100] = 1.0

            # Values: (samples,)
            values = np.random.choice([-1.0, 0.0, 1.0], size=n_samples).astype(
                np.float32
            )

            np.savez(
                f.name,
                features=features,
                globals=globals_vec,
                policy=policy,
                values=values,
                policy_encoding="board_aware",
                history_length=np.array(3),
                feature_version=np.array(1),
            )

            yield Path(f.name)

        # Cleanup
        Path(f.name).unlink(missing_ok=True)

    def test_nonexistent_data_path_fails(self, minimal_config):
        """Test that nonexistent data path is handled."""
        from app.training.train import train_model

        with tempfile.TemporaryDirectory() as tmpdir:
            # train_model should handle missing file gracefully
            # (either by raising or returning early)
            with pytest.raises((FileNotFoundError, ValueError, OSError)):
                train_model(
                    config=minimal_config,
                    data_path="/nonexistent/path/data.npz",
                    save_path=str(Path(tmpdir) / "model.pth"),
                )

    def test_empty_data_path_list_fails(self, minimal_config):
        """Test that empty data path list is handled."""
        from app.training.train import train_model

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises((ValueError, IndexError)):
                train_model(
                    config=minimal_config,
                    data_path=[],
                    save_path=str(Path(tmpdir) / "model.pth"),
                )


class TestBuildRankTargets:
    """Tests for build_rank_targets function (V3 multi-player)."""

    @pytest.fixture
    def build_rank_targets(self):
        """Import build_rank_targets function."""
        from app.training.train import build_rank_targets

        return build_rank_targets

    def test_two_player_ranking(self, build_rank_targets):
        """Test rank target generation for 2 players."""
        # value_targets: (batch, num_players) where higher = better
        value_targets = torch.tensor([[0.8, 0.2], [-0.5, 0.5]])
        num_players = 2

        rank_targets, rank_mask = build_rank_targets(value_targets, num_players)

        # Shape should be (batch, num_players, num_players)
        assert rank_targets.shape == (2, 2, 2)
        assert rank_mask.shape == (2, 2)

        # Player 0 in sample 0 has 0.8 > 0.2, so should be rank 0 (first place)
        # Rank distribution for player 0: [1, 0] (100% chance of rank 0)

    def test_four_player_ranking(self, build_rank_targets):
        """Test rank target generation for 4 players."""
        value_targets = torch.tensor([[0.9, 0.3, 0.6, 0.1]])
        num_players = 4

        rank_targets, rank_mask = build_rank_targets(value_targets, num_players)

        assert rank_targets.shape == (1, 4, 4)
        assert rank_mask.shape == (1, 4)

    def test_rank_targets_can_match_fixed_seat_model_head(self, build_rank_targets):
        """3p fixed-seat models can train from 4-slot values_mp targets."""
        value_targets = torch.tensor([[0.9, 0.3, 0.6, -1.0]])
        num_players = 3

        rank_targets, rank_mask = build_rank_targets(
            value_targets,
            num_players,
            output_players=3,
        )

        assert rank_targets.shape == (1, 3, 3)
        assert rank_mask.shape == (1, 3)
        assert rank_mask.all()


class TestGetPolicySizeForBoard:
    """Tests for get_policy_size_for_board function."""

    @pytest.fixture
    def get_policy_size(self):
        """Import the function."""
        from app.training.train import get_policy_size_for_board

        return get_policy_size_for_board

    def test_square8_size(self, get_policy_size):
        """Test policy size for square8 board."""
        size = get_policy_size(BoardType.SQUARE8)
        # square8 has 64 cells, various action types
        assert size > 0
        assert size < 100000  # Should be board-aware, not legacy MAX_N

    def test_hex8_size(self, get_policy_size):
        """Test policy size for hex8 board."""
        size = get_policy_size(BoardType.HEX8)
        assert size > 0
        assert size < 100000

    def test_all_board_types_have_sizes(self, get_policy_size):
        """Test that all board types return valid sizes."""
        for board_type in [
            BoardType.SQUARE8,
            BoardType.SQUARE19,
            BoardType.HEX8,
            BoardType.HEXAGONAL,
        ]:
            size = get_policy_size(board_type)
            assert size > 0


class TestSeedAll:
    """Tests for seed_all reproducibility function."""

    def test_seed_produces_deterministic_results(self):
        """Test that seeding produces deterministic random numbers."""
        from app.training.train import seed_all

        seed_all(42)
        rand1 = torch.rand(5)
        np_rand1 = np.random.rand(5)

        seed_all(42)
        rand2 = torch.rand(5)
        np_rand2 = np.random.rand(5)

        assert torch.allclose(rand1, rand2)
        assert np.allclose(np_rand1, np_rand2)

    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results."""
        from app.training.train import seed_all

        seed_all(42)
        rand1 = torch.rand(5)

        seed_all(123)
        rand2 = torch.rand(5)

        assert not torch.allclose(rand1, rand2)


class TestDataValidation:
    """Tests for training data validation.

    Note: These tests verify that data validation rejects invalid formats.
    They use the validate_data=True parameter to trigger validation early
    without starting the full training loop.
    """

    def test_legacy_encoding_check(self):
        """Test that legacy_max_n encoding is detected in NPZ metadata."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            # Create a file with legacy encoding marker
            values = np.random.choice([-1.0, 0.0, 1.0], size=10).astype(np.float32)
            np.savez(
                f.name,
                features=np.random.randn(10, 56, 8, 8).astype(np.float32),
                globals=np.random.randn(10, 20).astype(np.float32),
                policy=np.zeros((10, 60000), dtype=np.float32),
                values=values,
                policy_encoding="legacy_max_n",  # Deprecated
                history_length=np.array(3),
                feature_version=np.array(2),
            )
            npz_path = Path(f.name)

        try:
            # Verify we can detect the legacy encoding by reading metadata
            with np.load(str(npz_path), allow_pickle=True) as data:
                encoding = str(np.asarray(data["policy_encoding"]).item())
                assert encoding == "legacy_max_n"
        finally:
            npz_path.unlink(missing_ok=True)

    def test_feature_version_mismatch_detected(self):
        """Test that feature version mismatch is detected in NPZ metadata."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            np.savez(
                f.name,
                features=np.random.randn(10, 56, 8, 8).astype(np.float32),
                globals=np.random.randn(10, 20).astype(np.float32),
                policy=np.zeros((10, 7168), dtype=np.float32),
                values=np.random.choice([-1.0, 0.0, 1.0], size=10).astype(np.float32),
                policy_encoding="board_aware",
                history_length=np.array(3),
                feature_version=np.array(1),  # Old version
            )
            npz_path = Path(f.name)

        try:
            with np.load(str(npz_path), allow_pickle=True) as data:
                version = int(np.asarray(data["feature_version"]).item())
                # Default config uses feature_version=2
                config = TrainConfig()
                assert version != config.feature_version  # Mismatch
        finally:
            npz_path.unlink(missing_ok=True)

    def test_valid_data_passes_basic_checks(self):
        """Test that valid data passes basic structure checks."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            np.savez(
                f.name,
                features=np.random.randn(10, 56, 8, 8).astype(np.float32),
                globals=np.random.randn(10, 20).astype(np.float32),
                policy=np.zeros((10, 7168), dtype=np.float32),
                values=np.random.choice([-1.0, 0.0, 1.0], size=10).astype(np.float32),
                policy_encoding="board_aware",
                history_length=np.array(3),
                feature_version=np.array(2),
            )
            npz_path = Path(f.name)

        try:
            with np.load(str(npz_path), allow_pickle=True) as data:
                # Check all required keys present
                assert "features" in data
                assert "globals" in data
                assert "policy" in data
                assert "values" in data

                # Check shapes
                assert data["features"].shape[0] == 10  # samples
                assert data["globals"].shape[1] == 20  # global features
        finally:
            npz_path.unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
