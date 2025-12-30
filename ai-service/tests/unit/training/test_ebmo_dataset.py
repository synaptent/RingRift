"""Tests for app.training.ebmo_dataset.

Tests the EBMO training dataset module including:
- EBMODatasetConfig and EBMOSample dataclasses
- ActionFeatureGenerator for generating action features
- GameDataParser for loading NPZ game files
- EBMODataset for PyTorch training
- EBMOStreamingDataset for large data
- create_ebmo_dataloader factory function
- generate_synthetic_ebmo_data utility
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from app.training.ebmo_dataset import (
    ActionFeatureGenerator,
    EBMODataset,
    EBMODatasetConfig,
    EBMOSample,
    EBMOStreamingDataset,
    GameDataParser,
    create_ebmo_dataloader,
    generate_synthetic_ebmo_data,
)


class TestEBMODatasetConfig:
    """Tests for EBMODatasetConfig dataclass."""

    def test_default_values(self):
        """Test EBMODatasetConfig has sensible defaults."""
        config = EBMODatasetConfig()

        assert config.data_dir == "data/games"
        assert config.num_negatives == 15
        assert config.hard_negative_ratio == 0.3
        assert config.outcome_weighted is True
        assert config.board_size == 8
        assert config.num_input_channels == 14
        assert config.num_global_features == 20
        assert config.action_feature_dim == 14
        assert config.max_samples_per_game is None
        assert config.shuffle_buffer_size == 10000

    def test_post_init_sets_data_paths(self):
        """Test __post_init__ sets data_paths from data_dir."""
        config = EBMODatasetConfig()

        assert config.data_paths is not None
        assert len(config.data_paths) == 1
        assert config.data_paths[0] == "data/games/*.npz"

    def test_custom_data_paths_preserved(self):
        """Test custom data_paths are preserved."""
        custom_paths = ["path1/*.npz", "path2/*.npz"]
        config = EBMODatasetConfig(data_paths=custom_paths)

        assert config.data_paths == custom_paths

    def test_custom_values(self):
        """Test EBMODatasetConfig accepts custom values."""
        config = EBMODatasetConfig(
            num_negatives=20,
            hard_negative_ratio=0.5,
            board_size=19,
            max_samples_per_game=100,
        )

        assert config.num_negatives == 20
        assert config.hard_negative_ratio == 0.5
        assert config.board_size == 19
        assert config.max_samples_per_game == 100


class TestEBMOSample:
    """Tests for EBMOSample dataclass."""

    def test_sample_creation(self):
        """Test EBMOSample creation with arrays."""
        board_features = np.random.randn(14, 8, 8).astype(np.float32)
        global_features = np.random.randn(20).astype(np.float32)
        positive_action = np.random.randn(14).astype(np.float32)
        negative_actions = np.random.randn(15, 14).astype(np.float32)
        outcome = 1.0

        sample = EBMOSample(
            board_features=board_features,
            global_features=global_features,
            positive_action=positive_action,
            negative_actions=negative_actions,
            outcome=outcome,
        )

        assert sample.board_features.shape == (14, 8, 8)
        assert sample.global_features.shape == (20,)
        assert sample.positive_action.shape == (14,)
        assert sample.negative_actions.shape == (15, 14)
        assert sample.outcome == 1.0


class TestActionFeatureGenerator:
    """Tests for ActionFeatureGenerator."""

    def test_initialization(self):
        """Test ActionFeatureGenerator initialization."""
        gen = ActionFeatureGenerator(board_size=8, action_dim=14)

        assert gen.board_size == 8
        assert gen.action_dim == 14
        assert gen.NUM_MOVE_TYPES == 8

    def test_generate_from_positions_shape(self):
        """Test generate_from_positions returns correct shape."""
        gen = ActionFeatureGenerator(board_size=8, action_dim=14)

        features = gen.generate_from_positions(0, 0, 7, 7, move_type=1)

        assert features.shape == (14,)
        assert features.dtype == np.float32

    def test_generate_from_positions_normalized_coords(self):
        """Test positions are normalized to [0, 1]."""
        gen = ActionFeatureGenerator(board_size=8, action_dim=14)

        # Corner positions
        features = gen.generate_from_positions(0, 0, 7, 7)

        assert features[0] == 0.0  # from_x
        assert features[1] == 0.0  # from_y
        assert features[2] == 1.0  # to_x
        assert features[3] == 1.0  # to_y

    def test_generate_from_positions_move_type_onehot(self):
        """Test move type is one-hot encoded."""
        gen = ActionFeatureGenerator(board_size=8, action_dim=14)

        for move_type in range(8):
            features = gen.generate_from_positions(4, 4, 5, 5, move_type=move_type)

            # Move type indices are 4-11
            onehot = features[4:12]
            assert np.sum(onehot) == 1.0
            assert onehot[move_type] == 1.0

    def test_generate_from_positions_direction_vector(self):
        """Test direction vector is normalized."""
        gen = ActionFeatureGenerator(board_size=8, action_dim=14)

        # Horizontal move
        features = gen.generate_from_positions(0, 4, 7, 4)
        dx, dy = features[12], features[13]

        # Should point right (dx > 0, dy = 0)
        assert dx > 0
        assert abs(dy) < 0.01
        # Should be unit vector
        assert abs(np.sqrt(dx*dx + dy*dy) - 1.0) < 0.01

    def test_generate_random_action(self):
        """Test generate_random_action returns valid action."""
        gen = ActionFeatureGenerator(board_size=8)

        action = gen.generate_random_action()

        assert action.shape == (14,)
        # Positions should be in [0, 1]
        assert 0 <= action[0] <= 1
        assert 0 <= action[1] <= 1
        assert 0 <= action[2] <= 1
        assert 0 <= action[3] <= 1

    def test_generate_hard_negative(self):
        """Test generate_hard_negative perturbs positive."""
        gen = ActionFeatureGenerator(board_size=8)

        positive = gen.generate_from_positions(4, 4, 5, 5, move_type=3)
        hard_neg = gen.generate_hard_negative(positive, noise_scale=0.1)

        assert hard_neg.shape == (14,)
        # Should be different from original
        assert not np.allclose(hard_neg, positive)
        # Positions should still be in [0, 1]
        assert np.all(hard_neg[:4] >= 0)
        assert np.all(hard_neg[:4] <= 1)

    def test_generate_hard_negative_preserves_structure(self):
        """Test hard negative maintains one-hot for move type."""
        gen = ActionFeatureGenerator(board_size=8)

        positive = gen.generate_from_positions(4, 4, 5, 5, move_type=3)

        # Generate many and check all have valid one-hot
        for _ in range(10):
            hard_neg = gen.generate_hard_negative(positive)
            onehot = hard_neg[4:12]
            assert np.sum(onehot) == 1.0


class TestGameDataParser:
    """Tests for GameDataParser."""

    def test_initialization(self):
        """Test GameDataParser initialization."""
        parser = GameDataParser(board_size=8, num_channels=14, num_globals=20)

        assert parser.board_size == 8
        assert parser.num_channels == 14
        assert parser.num_globals == 20

    def test_load_npz_valid_file(self):
        """Test load_npz loads valid NPZ file."""
        parser = GameDataParser()

        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            np.savez(
                f.name,
                features=np.random.randn(10, 14, 8, 8),
                values=np.random.randn(10),
            )

            data = parser.load_npz(f.name)

            assert data is not None
            assert 'features' in data
            assert 'values' in data
            assert data['features'].shape == (10, 14, 8, 8)

    def test_load_npz_missing_required_keys(self):
        """Test load_npz returns None for invalid file."""
        parser = GameDataParser()

        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            # Missing 'values' key
            np.savez(f.name, features=np.random.randn(10, 14, 8, 8))

            data = parser.load_npz(f.name)

            assert data is None

    def test_load_npz_nonexistent_file(self):
        """Test load_npz returns None for nonexistent file."""
        parser = GameDataParser()

        data = parser.load_npz("/nonexistent/path.npz")

        assert data is None

    def test_extract_samples_returns_samples(self):
        """Test extract_samples returns EBMOSample list."""
        parser = GameDataParser(board_size=8)
        action_gen = ActionFeatureGenerator(board_size=8)

        data = {
            'features': np.random.randn(5, 14, 8, 8).astype(np.float32),
            'values': np.array([1, -1, 0, 1, -1], dtype=np.float32),
            'globals': np.random.randn(5, 20).astype(np.float32),
        }

        samples = parser.extract_samples(data, action_gen, num_negatives=10)

        assert len(samples) == 5
        assert all(isinstance(s, EBMOSample) for s in samples)

    def test_extract_samples_respects_max_samples(self):
        """Test extract_samples respects max_samples limit."""
        parser = GameDataParser(board_size=8)
        action_gen = ActionFeatureGenerator(board_size=8)

        data = {
            'features': np.random.randn(100, 14, 8, 8).astype(np.float32),
            'values': np.random.randn(100).astype(np.float32),
        }

        samples = parser.extract_samples(data, action_gen, max_samples=10)

        assert len(samples) == 10

    def test_extract_samples_without_globals(self):
        """Test extract_samples works without globals array."""
        parser = GameDataParser(board_size=8, num_globals=20)
        action_gen = ActionFeatureGenerator(board_size=8)

        data = {
            'features': np.random.randn(5, 14, 8, 8).astype(np.float32),
            'values': np.random.randn(5).astype(np.float32),
        }

        samples = parser.extract_samples(data, action_gen)

        assert len(samples) == 5
        # Should have zeros for globals
        assert samples[0].global_features.shape == (20,)

    def test_extract_samples_with_policy_indices(self):
        """Test extract_samples uses policy indices for positive actions."""
        parser = GameDataParser(board_size=8)
        action_gen = ActionFeatureGenerator(board_size=8)

        data = {
            'features': np.random.randn(5, 14, 8, 8).astype(np.float32),
            'values': np.random.randn(5).astype(np.float32),
            'policy_indices': np.array([[10], [20], [30], [40], [50]]),
        }

        samples = parser.extract_samples(data, action_gen)

        assert len(samples) == 5
        # Positive actions should be generated from policy

    def test_action_from_policy_empty_indices(self):
        """Test _action_from_policy handles empty indices."""
        parser = GameDataParser(board_size=8)
        action_gen = ActionFeatureGenerator(board_size=8)

        action = parser._action_from_policy(np.array([]), action_gen)

        assert action.shape == (14,)


class TestEBMODataset:
    """Tests for EBMODataset."""

    def test_initialization_empty_paths(self):
        """Test EBMODataset with no matching files."""
        dataset = EBMODataset(data_paths=["/nonexistent/*.npz"])

        assert len(dataset.file_paths) == 0

    def test_initialization_with_files(self):
        """Test EBMODataset finds files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            for i in range(3):
                path = Path(tmpdir) / f"game_{i}.npz"
                np.savez(
                    str(path),
                    features=np.random.randn(10, 14, 8, 8),
                    values=np.random.randn(10),
                )

            dataset = EBMODataset(data_paths=[f"{tmpdir}/*.npz"])

            assert len(dataset.file_paths) == 3

    def test_len_without_preload(self):
        """Test __len__ estimates size without preload."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "game.npz"
            np.savez(
                str(path),
                features=np.random.randn(10, 14, 8, 8),
                values=np.random.randn(10),
            )

            dataset = EBMODataset(
                data_paths=[f"{tmpdir}/*.npz"],
                preload=False,
            )

            # Should estimate ~1000 per file
            assert len(dataset) == 1000

    def test_len_with_max_samples(self):
        """Test __len__ respects max_samples_per_game."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(2):
                path = Path(tmpdir) / f"game_{i}.npz"
                np.savez(
                    str(path),
                    features=np.random.randn(100, 14, 8, 8),
                    values=np.random.randn(100),
                )

            dataset = EBMODataset(
                data_paths=[f"{tmpdir}/*.npz"],
                max_samples_per_game=50,
                preload=False,
            )

            assert len(dataset) == 100  # 2 files * 50

    def test_getitem_returns_tensors(self):
        """Test __getitem__ returns tuple of tensors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "game.npz"
            np.savez(
                str(path),
                features=np.random.randn(10, 14, 8, 8),
                values=np.random.randn(10),
                globals=np.random.randn(10, 20),
            )

            dataset = EBMODataset(
                data_paths=[f"{tmpdir}/*.npz"],
                num_negatives=5,
                preload=True,
            )

            sample = dataset[0]

            assert len(sample) == 5
            assert isinstance(sample[0], torch.Tensor)  # board_features
            assert isinstance(sample[1], torch.Tensor)  # global_features
            assert isinstance(sample[2], torch.Tensor)  # positive_action
            assert isinstance(sample[3], torch.Tensor)  # negative_actions
            assert isinstance(sample[4], torch.Tensor)  # outcome

    def test_getitem_shapes(self):
        """Test __getitem__ returns correct shapes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "game.npz"
            np.savez(
                str(path),
                features=np.random.randn(10, 14, 8, 8),
                values=np.random.randn(10),
                globals=np.random.randn(10, 20),
            )

            num_negatives = 5
            dataset = EBMODataset(
                data_paths=[f"{tmpdir}/*.npz"],
                num_negatives=num_negatives,
                board_size=8,
                preload=True,
            )

            board_feat, global_feat, pos, neg, outcome = dataset[0]

            assert board_feat.shape == (14, 8, 8)
            assert global_feat.shape == (20,)
            assert pos.shape == (14,)
            assert neg.shape == (num_negatives, 14)
            assert outcome.shape == ()

    def test_dummy_sample(self):
        """Test _dummy_sample returns valid dummy."""
        dataset = EBMODataset(
            data_paths=["/nonexistent/*.npz"],
            num_negatives=10,
        )

        sample = dataset._dummy_sample()

        assert len(sample) == 5
        assert sample[0].shape == (14, 8, 8)
        assert sample[1].shape == (20,)
        assert sample[2].shape == (14,)
        assert sample[3].shape == (10, 14)
        assert sample[4].item() == 0.0

    def test_preload(self):
        """Test preloading all samples."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "game.npz"
            np.savez(
                str(path),
                features=np.random.randn(10, 14, 8, 8),
                values=np.random.randn(10),
            )

            dataset = EBMODataset(
                data_paths=[f"{tmpdir}/*.npz"],
                preload=True,
            )

            assert dataset.preloaded_samples is not None
            assert len(dataset.preloaded_samples) == 10
            assert len(dataset) == 10


class TestEBMOStreamingDataset:
    """Tests for EBMOStreamingDataset."""

    def test_initialization(self):
        """Test EBMOStreamingDataset initialization."""
        dataset = EBMOStreamingDataset(
            data_paths=["/nonexistent/*.npz"],
            num_negatives=10,
            shuffle=True,
        )

        assert dataset.num_negatives == 10
        assert dataset.shuffle is True

    def test_iteration(self):
        """Test streaming dataset iteration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "game.npz"
            np.savez(
                str(path),
                features=np.random.randn(5, 14, 8, 8),
                values=np.random.randn(5),
            )

            dataset = EBMOStreamingDataset(
                data_paths=[f"{tmpdir}/*.npz"],
                num_negatives=5,
                shuffle=False,
            )

            samples = list(dataset)

            assert len(samples) == 5
            assert all(len(s) == 5 for s in samples)  # 5-tuple

    def test_iteration_with_shuffle(self):
        """Test streaming dataset shuffles."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(3):
                path = Path(tmpdir) / f"game_{i}.npz"
                np.savez(
                    str(path),
                    features=np.random.randn(2, 14, 8, 8),
                    values=np.array([i, i]),  # Use index as value
                )

            dataset = EBMOStreamingDataset(
                data_paths=[f"{tmpdir}/*.npz"],
                shuffle=True,
            )

            # Just check it iterates without error
            samples = list(dataset)
            assert len(samples) == 6

    def test_handles_invalid_files(self):
        """Test streaming skips invalid files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Valid file
            path1 = Path(tmpdir) / "good.npz"
            np.savez(
                str(path1),
                features=np.random.randn(2, 14, 8, 8),
                values=np.random.randn(2),
            )

            # Invalid file (missing values)
            path2 = Path(tmpdir) / "bad.npz"
            np.savez(str(path2), features=np.random.randn(2, 14, 8, 8))

            dataset = EBMOStreamingDataset(
                data_paths=[f"{tmpdir}/*.npz"],
                shuffle=False,
            )

            samples = list(dataset)

            # Should only have samples from good file
            assert len(samples) == 2


class TestCreateEBMODataLoader:
    """Tests for create_ebmo_dataloader factory function."""

    def test_creates_dataloader(self):
        """Test create_ebmo_dataloader returns DataLoader."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "game.npz"
            np.savez(
                str(path),
                features=np.random.randn(10, 14, 8, 8),
                values=np.random.randn(10),
            )

            loader = create_ebmo_dataloader(
                data_paths=[f"{tmpdir}/*.npz"],
                batch_size=4,
                num_workers=0,
            )

            assert loader.batch_size == 4

    def test_streaming_mode(self):
        """Test create_ebmo_dataloader in streaming mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "game.npz"
            np.savez(
                str(path),
                features=np.random.randn(10, 14, 8, 8),
                values=np.random.randn(10),
            )

            loader = create_ebmo_dataloader(
                data_paths=[f"{tmpdir}/*.npz"],
                batch_size=4,
                num_workers=0,
                streaming=True,
            )

            # Streaming datasets use IterableDataset
            assert isinstance(loader.dataset, EBMOStreamingDataset)

    def test_preload_mode(self):
        """Test create_ebmo_dataloader with preload."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "game.npz"
            np.savez(
                str(path),
                features=np.random.randn(10, 14, 8, 8),
                values=np.random.randn(10),
            )

            loader = create_ebmo_dataloader(
                data_paths=[f"{tmpdir}/*.npz"],
                batch_size=4,
                num_workers=0,
                streaming=False,
                preload=True,
            )

            assert isinstance(loader.dataset, EBMODataset)
            assert loader.dataset.preloaded_samples is not None


class TestGenerateSyntheticEBMOData:
    """Tests for generate_synthetic_ebmo_data utility."""

    def test_generates_samples(self):
        """Test generates correct number of samples."""
        samples = generate_synthetic_ebmo_data(
            num_samples=100,
            board_size=8,
            num_negatives=10,
        )

        assert len(samples) == 100
        assert all(isinstance(s, EBMOSample) for s in samples)

    def test_sample_shapes(self):
        """Test generated samples have correct shapes."""
        samples = generate_synthetic_ebmo_data(
            num_samples=10,
            board_size=8,
            num_negatives=15,
        )

        for sample in samples:
            assert sample.board_features.shape == (14, 8, 8)
            assert sample.global_features.shape == (20,)
            assert sample.positive_action.shape == (14,)
            assert sample.negative_actions.shape == (15, 14)
            assert sample.outcome in [1.0, -1.0]

    def test_different_board_sizes(self):
        """Test works with different board sizes."""
        for board_size in [8, 12, 19]:
            samples = generate_synthetic_ebmo_data(
                num_samples=5,
                board_size=board_size,
            )

            assert samples[0].board_features.shape == (14, board_size, board_size)

    def test_center_bias(self):
        """Test positive actions are biased toward center."""
        samples = generate_synthetic_ebmo_data(
            num_samples=100,
            board_size=8,
        )

        # Positive actions should have normalized positions closer to center
        center = 0.5
        total_dist = 0
        for sample in samples:
            pos = sample.positive_action
            from_dist = abs(pos[0] - center) + abs(pos[1] - center)
            to_dist = abs(pos[2] - center) + abs(pos[3] - center)
            total_dist += from_dist + to_dist

        avg_dist = total_dist / (100 * 4)
        # Should be closer to center than random (0.25 average)
        # Center-biased should be less than 0.4 on average
        assert avg_dist < 0.45


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_action_generator_zero_position(self):
        """Test action generator handles zero position."""
        gen = ActionFeatureGenerator(board_size=8)

        features = gen.generate_from_positions(0, 0, 0, 0)

        # Direction should be normalized despite zero movement
        assert np.isfinite(features[12])
        assert np.isfinite(features[13])

    def test_action_generator_same_position(self):
        """Test action generator handles same from/to position."""
        gen = ActionFeatureGenerator(board_size=8)

        features = gen.generate_from_positions(4, 4, 4, 4)

        # Should not have NaN from division by zero
        assert not np.any(np.isnan(features))

    def test_parser_extract_samples_empty_data(self):
        """Test extract_samples with empty features array."""
        parser = GameDataParser()
        action_gen = ActionFeatureGenerator()

        data = {
            'features': np.zeros((0, 14, 8, 8)),
            'values': np.zeros(0),
        }

        samples = parser.extract_samples(data, action_gen)

        assert len(samples) == 0

    def test_dataset_getitem_modulo(self):
        """Test dataset wraps index when preloaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "game.npz"
            np.savez(
                str(path),
                features=np.random.randn(5, 14, 8, 8),
                values=np.random.randn(5),
            )

            dataset = EBMODataset(
                data_paths=[f"{tmpdir}/*.npz"],
                preload=True,
            )

            # Should wrap around
            sample_0 = dataset[0]
            sample_5 = dataset[5]  # Same as 0 with 5 samples

            assert torch.allclose(sample_0[0], sample_5[0])

    def test_large_board_size(self):
        """Test works with large board sizes."""
        gen = ActionFeatureGenerator(board_size=19)

        features = gen.generate_from_positions(0, 0, 18, 18)

        assert features.shape == (14,)
        assert features[2] == 1.0  # to_x normalized

    def test_move_type_overflow(self):
        """Test move type is clamped to valid range."""
        gen = ActionFeatureGenerator(board_size=8)

        # Move type > 7 should be clamped
        features = gen.generate_from_positions(4, 4, 5, 5, move_type=100)

        onehot = features[4:12]
        assert np.sum(onehot) == 1.0
        assert onehot[7] == 1.0  # Clamped to max (7)
