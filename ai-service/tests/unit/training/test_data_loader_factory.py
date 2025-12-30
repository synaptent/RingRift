"""Tests for app.training.data_loader_factory.

Tests the data loader factory for neural network training including:
- DataLoaderConfig and DataLoaderResult dataclasses
- Platform-aware worker computation
- Auto-streaming detection based on file size
- Config key inference from file paths
- Curriculum weight computation
- Data path collection
- Data loader creation
- Dataset metadata validation
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.models import BoardType
from app.training.data_loader_factory import (
    AUTO_STREAMING_THRESHOLD_BYTES,
    DataLoaderConfig,
    DataLoaderResult,
    collect_data_paths,
    compute_curriculum_file_weights,
    compute_num_workers,
    create_data_loaders,
    create_standard_loaders,
    create_streaming_loaders,
    infer_config_key_from_path,
    should_use_streaming,
    validate_dataset_metadata,
)


class TestDataLoaderConfig:
    """Tests for DataLoaderConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = DataLoaderConfig()

        assert config.batch_size == 256
        assert config.use_streaming is False
        assert config.sampling_weights == 'uniform'
        assert config.augment_hex_symmetry is False
        assert config.multi_player is False
        assert config.filter_empty_policies is True
        assert config.seed == 42
        assert config.board_type == BoardType.SQUARE8
        assert config.policy_size == 512

    def test_distributed_defaults(self):
        """Test distributed training defaults."""
        config = DataLoaderConfig()

        assert config.distributed is False
        assert config.rank == 0
        assert config.world_size == 1

    def test_custom_values(self):
        """Test custom configuration values."""
        config = DataLoaderConfig(
            batch_size=512,
            use_streaming=True,
            sampling_weights='game_id',
            board_type=BoardType.HEXAGONAL,
            multi_player=True,
        )

        assert config.batch_size == 512
        assert config.use_streaming is True
        assert config.sampling_weights == 'game_id'
        assert config.board_type == BoardType.HEXAGONAL
        assert config.multi_player is True

    def test_curriculum_weights_config(self):
        """Test curriculum weights configuration."""
        weights = {"hex8_2p": 1.5, "square8_4p": 0.8}
        config = DataLoaderConfig(
            use_curriculum_weights=True,
            curriculum_weights=weights,
        )

        assert config.use_curriculum_weights is True
        assert config.curriculum_weights == weights

    def test_return_heuristics_config(self):
        """Test heuristics return configuration for v5 models."""
        config = DataLoaderConfig(return_heuristics=True)

        assert config.return_heuristics is True

    def test_data_path_types(self):
        """Test data_path accepts various types."""
        # Single string
        config1 = DataLoaderConfig(data_path="path/to/data.npz")
        assert config1.data_path == "path/to/data.npz"

        # List of strings
        config2 = DataLoaderConfig(data_path=["a.npz", "b.npz"])
        assert config2.data_path == ["a.npz", "b.npz"]

        # None
        config3 = DataLoaderConfig(data_path=None)
        assert config3.data_path is None


class TestDataLoaderResult:
    """Tests for DataLoaderResult dataclass."""

    def test_default_values(self):
        """Test default result values."""
        result = DataLoaderResult()

        assert result.train_loader is None
        assert result.val_loader is None
        assert result.train_sampler is None
        assert result.val_sampler is None
        assert result.train_size == 0
        assert result.val_size == 0
        assert result.use_streaming is False
        assert result.has_multi_player_values is False
        assert result.data_paths is None
        assert result.num_workers == 0

    def test_custom_values(self):
        """Test custom result values."""
        mock_loader = MagicMock()

        result = DataLoaderResult(
            train_loader=mock_loader,
            train_size=1000,
            val_size=200,
            use_streaming=True,
            has_multi_player_values=True,
        )

        assert result.train_loader is mock_loader
        assert result.train_size == 1000
        assert result.val_size == 200
        assert result.use_streaming is True
        assert result.has_multi_player_values is True


class TestComputeNumWorkers:
    """Tests for compute_num_workers function."""

    def test_explicit_config(self):
        """Test explicit num_workers configuration."""
        result = compute_num_workers(8)
        assert result == 8

        result = compute_num_workers(0)
        assert result == 0

    @patch.dict(os.environ, {"RINGRIFT_DATALOADER_WORKERS": "6"})
    def test_env_override(self):
        """Test environment variable override."""
        result = compute_num_workers(None)
        assert result == 6

        # Env var takes precedence even with explicit config
        result = compute_num_workers(4)
        assert result == 6

    @patch.dict(os.environ, {}, clear=True)
    @patch.object(sys, 'platform', 'darwin')
    def test_macos_default(self):
        """Test macOS defaults to 0 workers."""
        # Need to clear env var
        os.environ.pop("RINGRIFT_DATALOADER_WORKERS", None)

        result = compute_num_workers(None)
        assert result == 0

    @patch.dict(os.environ, {}, clear=True)
    @patch.object(sys, 'platform', 'linux')
    @patch('multiprocessing.cpu_count', return_value=16)
    def test_linux_default(self, mock_cpu):
        """Test Linux defaults to min(4, cpu_count//2)."""
        os.environ.pop("RINGRIFT_DATALOADER_WORKERS", None)

        result = compute_num_workers(None)
        assert result == 4  # min(4, 16//2) = 4

    @patch.dict(os.environ, {}, clear=True)
    @patch.object(sys, 'platform', 'linux')
    @patch('multiprocessing.cpu_count', return_value=4)
    def test_linux_low_cpu(self, mock_cpu):
        """Test Linux with few CPUs."""
        os.environ.pop("RINGRIFT_DATALOADER_WORKERS", None)

        result = compute_num_workers(None)
        assert result == 2  # min(4, 4//2) = 2


class TestShouldUseStreaming:
    """Tests for should_use_streaming function."""

    def test_below_threshold(self):
        """Test returns False when below threshold."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            # Write small file
            np.savez(f.name, data=np.zeros(100))
            f.close()

            try:
                result = should_use_streaming(f.name, None)
                assert result is False
            finally:
                os.unlink(f.name)

    def test_above_threshold(self):
        """Test returns True when above threshold."""
        # Use tiny threshold to trigger streaming
        result = should_use_streaming(
            "dummy.npz",
            None,
            threshold_bytes=0,  # Any file triggers
        )
        # File doesn't exist, so size is 0, equal to threshold
        assert result is False

    def test_multiple_files(self):
        """Test with multiple files summing size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = []
            for i in range(3):
                path = os.path.join(tmpdir, f"data{i}.npz")
                np.savez(path, data=np.zeros(100))
                paths.append(path)

            # Small threshold to trigger
            result = should_use_streaming(
                paths,
                None,
                threshold_bytes=100,  # Small enough to trigger
            )
            assert result is True

    def test_data_dir(self):
        """Test with data directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(2):
                path = os.path.join(tmpdir, f"data{i}.npz")
                np.savez(path, data=np.zeros(50))

            result = should_use_streaming(
                None,
                tmpdir,
                threshold_bytes=100,  # Should trigger
            )
            assert result is True

    def test_nonexistent_files(self):
        """Test with nonexistent files."""
        result = should_use_streaming(
            "/nonexistent/path.npz",
            None,
        )
        assert result is False


class TestInferConfigKeyFromPath:
    """Tests for infer_config_key_from_path function."""

    def test_hex8_2p(self):
        """Test hex8 2-player detection."""
        assert infer_config_key_from_path("hex8_2p_games.npz") == "hex8_2p"
        assert infer_config_key_from_path("/path/to/canonical_hex8_2p.npz") == "hex8_2p"
        assert infer_config_key_from_path("training_hex8_2p_v3.npz") == "hex8_2p"

    def test_square8_4p(self):
        """Test square8 4-player detection."""
        assert infer_config_key_from_path("square8_4p.npz") == "square8_4p"
        assert infer_config_key_from_path("canonical_square8_4p_20241225.npz") == "square8_4p"

    def test_hexagonal(self):
        """Test hexagonal board detection."""
        assert infer_config_key_from_path("hexagonal_3p_selfplay.npz") == "hexagonal_3p"
        assert infer_config_key_from_path("data/hexagonal_2p.npz") == "hexagonal_2p"

    def test_square19(self):
        """Test square19 board detection."""
        assert infer_config_key_from_path("square19_2p.npz") == "square19_2p"
        assert infer_config_key_from_path("square19_4p_training.npz") == "square19_4p"

    def test_case_insensitive(self):
        """Test case insensitivity."""
        assert infer_config_key_from_path("HEX8_2P.NPZ") == "hex8_2p"
        assert infer_config_key_from_path("SQUARE8_3P.npz") == "square8_3p"

    def test_unknown_pattern(self):
        """Test returns None for unknown patterns."""
        assert infer_config_key_from_path("random_data.npz") is None
        assert infer_config_key_from_path("training_v3.npz") is None
        assert infer_config_key_from_path("games_2player.npz") is None


class TestComputeCurriculumFileWeights:
    """Tests for compute_curriculum_file_weights function."""

    def test_matching_weights(self):
        """Test files matched to curriculum weights."""
        paths = [
            "/data/hex8_2p_games.npz",
            "/data/square8_4p.npz",
        ]
        curriculum = {
            "hex8_2p": 1.5,
            "square8_4p": 0.5,
        }

        result = compute_curriculum_file_weights(paths, curriculum)

        assert result["/data/hex8_2p_games.npz"] == 1.5
        assert result["/data/square8_4p.npz"] == 0.5

    def test_default_weight(self):
        """Test default weight for unknown configs."""
        paths = ["/data/unknown.npz"]
        curriculum = {"hex8_2p": 1.5}

        result = compute_curriculum_file_weights(paths, curriculum, default_weight=1.0)

        assert result["/data/unknown.npz"] == 1.0

    def test_mixed_weights(self):
        """Test mix of known and unknown configs."""
        paths = [
            "/data/hex8_2p.npz",
            "/data/unknown.npz",
            "/data/square8_4p.npz",
        ]
        curriculum = {"hex8_2p": 2.0}

        result = compute_curriculum_file_weights(paths, curriculum)

        assert result["/data/hex8_2p.npz"] == 2.0
        assert result["/data/unknown.npz"] == 1.0
        assert result["/data/square8_4p.npz"] == 1.0  # Known config but no weight

    def test_empty_inputs(self):
        """Test empty inputs."""
        assert compute_curriculum_file_weights([], {}) == {}
        assert compute_curriculum_file_weights([], {"hex8_2p": 1.0}) == {}


class TestCollectDataPaths:
    """Tests for collect_data_paths function."""

    def test_single_path(self):
        """Test single data path."""
        result = collect_data_paths("data.npz", None)
        assert result == ["data.npz"]

    def test_path_list(self):
        """Test list of data paths."""
        result = collect_data_paths(["a.npz", "b.npz"], None)
        assert result == ["a.npz", "b.npz"]

    def test_data_dir(self):
        """Test data directory glob."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for name in ["a.npz", "b.npz", "c.txt"]:
                Path(tmpdir, name).touch()

            result = collect_data_paths(None, tmpdir)

            assert len(result) == 2
            assert all(p.endswith(".npz") for p in result)

    def test_removes_duplicates(self):
        """Test duplicate removal."""
        result = collect_data_paths(["a.npz", "b.npz", "a.npz"], None)
        assert result == ["a.npz", "b.npz"]

    def test_preserves_order(self):
        """Test order preservation."""
        paths = ["c.npz", "a.npz", "b.npz"]
        result = collect_data_paths(paths, None)
        assert result == paths

    def test_none_inputs(self):
        """Test with None inputs."""
        result = collect_data_paths(None, None)
        assert result == []


class TestCreateStreamingLoaders:
    """Tests for create_streaming_loaders function."""

    def test_empty_paths(self):
        """Test with empty paths."""
        config = DataLoaderConfig()
        result = create_streaming_loaders(config, [])

        assert result.train_loader is None
        assert result.val_loader is None
        assert result.use_streaming is True

    @patch('app.training.data_loader_factory.get_sample_count', return_value=0)
    def test_no_samples(self, mock_count):
        """Test with zero samples."""
        config = DataLoaderConfig()
        result = create_streaming_loaders(config, ["fake.npz"])

        assert result.train_loader is None
        assert result.val_loader is None

    @patch('app.training.data_loader_factory.StreamingDataLoader')
    @patch('app.training.data_loader_factory.get_sample_count', return_value=1000)
    @patch('os.path.exists', return_value=True)
    def test_creates_loaders(self, mock_exists, mock_count, mock_loader_class):
        """Test streaming loaders are created."""
        mock_loader = MagicMock()
        mock_loader.has_multi_player_values = False
        mock_loader_class.return_value = mock_loader

        config = DataLoaderConfig(batch_size=64)
        result = create_streaming_loaders(config, ["data.npz"])

        assert result.use_streaming is True
        assert result.train_size == 800  # 80% of 1000
        assert result.val_size == 200  # 20% of 1000

    @patch('app.training.data_loader_factory.WeightedStreamingDataLoader')
    @patch('app.training.data_loader_factory.StreamingDataLoader')
    @patch('app.training.data_loader_factory.get_sample_count', return_value=1000)
    @patch('os.path.exists', return_value=True)
    def test_weighted_sampling(self, mock_exists, mock_count, mock_stream, mock_weighted):
        """Test weighted streaming loader is used."""
        mock_loader = MagicMock()
        mock_loader.has_multi_player_values = False
        mock_weighted.return_value = mock_loader
        mock_stream.return_value = mock_loader

        config = DataLoaderConfig(
            sampling_weights='game_id',
        )
        result = create_streaming_loaders(config, ["data.npz"])

        assert mock_weighted.called

    @patch('app.training.data_loader_factory.WeightedStreamingDataLoader')
    @patch('app.training.data_loader_factory.StreamingDataLoader')
    @patch('app.training.data_loader_factory.get_sample_count', return_value=1000)
    @patch('os.path.exists', return_value=True)
    def test_curriculum_weights(self, mock_exists, mock_count, mock_stream, mock_weighted):
        """Test curriculum weights are applied."""
        mock_loader = MagicMock()
        mock_loader.has_multi_player_values = False
        mock_weighted.return_value = mock_loader
        mock_stream.return_value = mock_loader

        config = DataLoaderConfig(
            use_curriculum_weights=True,
            curriculum_weights={"hex8_2p": 1.5},
        )
        result = create_streaming_loaders(config, ["hex8_2p.npz"])

        # Should use weighted loader due to curriculum weights
        assert mock_weighted.called


class TestCreateStandardLoaders:
    """Tests for create_standard_loaders function."""

    @patch('app.training.data_loader_factory.RingRiftDataset')
    @patch('app.training.data_loader_factory.random_split')
    def test_creates_loaders(self, mock_split, mock_dataset_class):
        """Test standard loaders are created."""
        # Mock dataset
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=1000)
        mock_dataset.has_multi_player_values = False
        mock_dataset.spatial_shape = (8, 8)  # Required for logging
        mock_dataset_class.return_value = mock_dataset

        # Mock split
        train_data = MagicMock()
        train_data.__len__ = MagicMock(return_value=800)
        val_data = MagicMock()
        val_data.__len__ = MagicMock(return_value=200)
        mock_split.return_value = (train_data, val_data)

        config = DataLoaderConfig(batch_size=64)
        result = create_standard_loaders(config, "data.npz")

        assert result.train_loader is not None
        assert result.val_loader is not None
        assert result.train_size == 800
        assert result.val_size == 200
        assert result.use_streaming is False

    @patch('app.training.data_loader_factory.RingRiftDataset')
    def test_empty_dataset(self, mock_dataset_class):
        """Test with empty dataset."""
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=0)
        mock_dataset_class.return_value = mock_dataset

        config = DataLoaderConfig()
        result = create_standard_loaders(config, "data.npz")

        assert result.train_loader is None
        assert result.val_loader is None

    @patch('app.training.data_loader_factory.WeightedRingRiftDataset')
    @patch('app.training.data_loader_factory.random_split')
    def test_weighted_dataset(self, mock_split, mock_weighted_class):
        """Test weighted dataset is used."""
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=1000)
        mock_dataset.has_multi_player_values = False
        mock_dataset.spatial_shape = (8, 8)  # Required for logging
        mock_dataset.sample_weights = None  # Avoid weighted sampler creation
        mock_weighted_class.return_value = mock_dataset

        train_data = MagicMock()
        train_data.__len__ = MagicMock(return_value=800)
        train_data.dataset = mock_dataset  # For hasattr checks
        val_data = MagicMock()
        val_data.__len__ = MagicMock(return_value=200)
        mock_split.return_value = (train_data, val_data)

        config = DataLoaderConfig(sampling_weights='game_id')
        result = create_standard_loaders(config, "data.npz")

        assert mock_weighted_class.called


class TestCreateDataLoaders:
    """Tests for create_data_loaders main entry point."""

    @patch('app.training.data_loader_factory.should_use_streaming', return_value=False)
    @patch('app.training.data_loader_factory.create_standard_loaders')
    @patch('os.path.exists', return_value=True)
    def test_uses_standard_loader(self, mock_exists, mock_standard, mock_streaming_check):
        """Test uses standard loaders when not streaming."""
        mock_standard.return_value = DataLoaderResult(train_size=100)

        config = DataLoaderConfig(data_path="data.npz")
        result = create_data_loaders(config)

        assert mock_standard.called
        assert result.train_size == 100

    @patch('app.training.data_loader_factory.should_use_streaming', return_value=True)
    @patch('app.training.data_loader_factory.create_streaming_loaders')
    @patch('app.training.data_loader_factory.collect_data_paths', return_value=["data.npz"])
    def test_uses_streaming_auto(self, mock_collect, mock_streaming, mock_check):
        """Test auto-enables streaming based on file size."""
        mock_streaming.return_value = DataLoaderResult(use_streaming=True)

        config = DataLoaderConfig(data_path="data.npz")
        result = create_data_loaders(config)

        assert mock_streaming.called
        assert result.use_streaming is True

    @patch('app.training.data_loader_factory.create_streaming_loaders')
    @patch('app.training.data_loader_factory.collect_data_paths', return_value=["data.npz"])
    def test_uses_streaming_explicit(self, mock_collect, mock_streaming):
        """Test uses streaming when explicitly enabled."""
        mock_streaming.return_value = DataLoaderResult(use_streaming=True)

        config = DataLoaderConfig(data_path="data.npz", use_streaming=True)
        result = create_data_loaders(config)

        assert mock_streaming.called

    @patch('app.training.data_loader_factory.should_use_streaming', return_value=False)
    @patch('os.path.exists', return_value=False)
    def test_nonexistent_path(self, mock_exists, mock_check):
        """Test returns empty result for nonexistent path."""
        config = DataLoaderConfig(data_path="nonexistent.npz")
        result = create_data_loaders(config)

        assert result.train_loader is None

    @patch('app.training.data_loader_factory.should_use_streaming', return_value=False)
    @patch('app.training.data_loader_factory.create_standard_loaders')
    @patch('os.path.exists', return_value=True)
    def test_list_path_uses_first(self, mock_exists, mock_standard, mock_check):
        """Test uses first path from list in standard mode."""
        mock_standard.return_value = DataLoaderResult()

        config = DataLoaderConfig(data_path=["a.npz", "b.npz"])
        create_data_loaders(config)

        # Should have called with first path
        mock_standard.assert_called_once()
        call_args = mock_standard.call_args[0]
        assert call_args[1] == "a.npz"


class TestValidateDatasetMetadata:
    """Tests for validate_dataset_metadata function."""

    def test_nonexistent_file(self):
        """Test returns empty dict for nonexistent file."""
        result = validate_dataset_metadata(
            "/nonexistent/path.npz",
            config_history_length=4,
            config_feature_version=1,
        )
        assert result == {}

    def test_validates_history_length(self):
        """Test history length validation."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            np.savez(f.name, features=np.zeros((10, 52)), history_length=np.array(8))
            f.close()

            try:
                with pytest.raises(ValueError) as exc_info:
                    validate_dataset_metadata(
                        f.name,
                        config_history_length=4,  # Mismatch!
                        config_feature_version=1,
                    )
                assert "history_length" in str(exc_info.value)
            finally:
                os.unlink(f.name)

    def test_validates_feature_version(self):
        """Test feature version validation."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            np.savez(f.name, features=np.zeros((10, 52)), feature_version=np.array(2))
            f.close()

            try:
                with pytest.raises(ValueError) as exc_info:
                    validate_dataset_metadata(
                        f.name,
                        config_history_length=4,
                        config_feature_version=1,  # Mismatch!
                    )
                assert "feature_version" in str(exc_info.value)
            finally:
                os.unlink(f.name)

    def test_extracts_metadata(self):
        """Test metadata extraction."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            np.savez(
                f.name,
                features=np.zeros((10, 52, 8, 8)),
                globals=np.zeros((10, 20)),
                history_length=np.array(4),
                feature_version=np.array(1),
            )
            f.close()

            try:
                result = validate_dataset_metadata(
                    f.name,
                    config_history_length=4,
                    config_feature_version=1,
                )

                assert result['in_channels'] == 52
                assert result['globals_dim'] == 20
                assert result['history_length'] == 4
                assert result['feature_version'] == 1
            finally:
                os.unlink(f.name)

    def test_validates_globals_dim(self):
        """Test globals dimension validation."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            np.savez(f.name, globals=np.zeros((10, 25)))  # Wrong dim
            f.close()

            try:
                with pytest.raises(ValueError) as exc_info:
                    validate_dataset_metadata(
                        f.name,
                        config_history_length=4,
                        config_feature_version=1,
                    )
                assert "globals dimension" in str(exc_info.value)
            finally:
                os.unlink(f.name)

    def test_validates_policy_encoding_v3(self):
        """Test policy encoding validation for v3 models."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            np.savez(f.name, policy_encoding=np.array("legacy_max_n"))
            f.close()

            try:
                with pytest.raises(ValueError) as exc_info:
                    validate_dataset_metadata(
                        f.name,
                        config_history_length=4,
                        config_feature_version=1,
                        model_version='v3',
                    )
                assert "legacy" in str(exc_info.value).lower()
            finally:
                os.unlink(f.name)

    def test_matching_metadata(self):
        """Test no error when metadata matches."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            np.savez(
                f.name,
                features=np.zeros((10, 52)),
                globals=np.zeros((10, 20)),
                history_length=np.array(4),
                feature_version=np.array(1),
            )
            f.close()

            try:
                result = validate_dataset_metadata(
                    f.name,
                    config_history_length=4,
                    config_feature_version=1,
                )
                # Should not raise
                assert 'history_length' in result
            finally:
                os.unlink(f.name)


class TestAutoStreamingThreshold:
    """Tests for AUTO_STREAMING_THRESHOLD_BYTES constant."""

    def test_default_value(self):
        """Test default threshold is 20GB."""
        expected = 20 * (1024 ** 3)
        # Only check if env var isn't set
        if "RINGRIFT_AUTO_STREAMING_THRESHOLD_GB" not in os.environ:
            assert AUTO_STREAMING_THRESHOLD_BYTES == expected

    @patch.dict(os.environ, {"RINGRIFT_AUTO_STREAMING_THRESHOLD_GB": "5"})
    def test_env_override(self):
        """Test environment variable overrides threshold."""
        # Re-import to pick up new env var
        from importlib import reload
        import app.training.data_loader_factory as dlf
        reload(dlf)

        expected = 5 * (1024 ** 3)
        assert dlf.AUTO_STREAMING_THRESHOLD_BYTES == expected


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_infer_config_key_special_characters(self):
        """Test config key inference with special characters."""
        # Should still work with paths containing special chars
        assert infer_config_key_from_path("/path/to/hex8_2p-v2.npz") == "hex8_2p"
        assert infer_config_key_from_path("data_hex8_3p_20241225.npz") == "hex8_3p"

    def test_collect_data_paths_empty_dir(self):
        """Test with empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = collect_data_paths(None, tmpdir)
            assert result == []

    def test_compute_num_workers_negative(self):
        """Test negative num_workers is used as-is."""
        # This might be invalid but shouldn't crash
        result = compute_num_workers(-1)
        assert result == -1

    def test_curriculum_weights_zero(self):
        """Test curriculum weights can be zero."""
        paths = ["/data/hex8_2p.npz"]
        curriculum = {"hex8_2p": 0.0}

        result = compute_curriculum_file_weights(paths, curriculum)
        assert result["/data/hex8_2p.npz"] == 0.0

    def test_should_use_streaming_empty_dir(self):
        """Test streaming check with empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = should_use_streaming(None, tmpdir)
            assert result is False
