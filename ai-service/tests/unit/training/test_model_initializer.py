"""Unit tests for model initialization module.

Tests the ModelInitializer class which handles model creation, weight loading,
and DDP wrapping for training. This is critical infrastructure for all training.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_default_config(self):
        """Test default ModelConfig values."""
        from app.coordination.types import BoardType
        from app.training.model_initializer import ModelConfig

        config = ModelConfig(board_type=BoardType.SQUARE8)
        assert config.board_type == BoardType.SQUARE8
        assert config.num_players == 2
        assert config.multi_player is False
        assert config.model_version == "v2"
        assert config.model_type == "cnn"
        assert config.num_res_blocks is None
        assert config.num_filters is None
        assert config.history_length == 3
        assert config.dropout == 0.08
        assert config.freeze_policy is False
        assert config.feature_version == 1
        assert config.policy_size == 0
        assert config.model_id == ""

    def test_config_with_custom_values(self):
        """Test ModelConfig with custom values."""
        from app.coordination.types import BoardType
        from app.training.model_initializer import ModelConfig

        config = ModelConfig(
            board_type=BoardType.HEX8,
            num_players=4,
            multi_player=True,
            model_version="v3",
            model_type="gnn",
            num_res_blocks=8,
            num_filters=128,
            history_length=5,
            dropout=0.1,
            freeze_policy=True,
            feature_version=2,
            policy_size=3782,
        )
        assert config.board_type == BoardType.HEX8
        assert config.num_players == 4
        assert config.multi_player is True
        assert config.model_version == "v3"
        assert config.model_type == "gnn"
        assert config.num_res_blocks == 8
        assert config.num_filters == 128
        assert config.history_length == 5
        assert config.dropout == 0.1
        assert config.freeze_policy is True
        assert config.feature_version == 2
        assert config.policy_size == 3782


class TestDatasetMetadata:
    """Tests for DatasetMetadata dataclass."""

    def test_default_metadata(self):
        """Test default DatasetMetadata values."""
        from app.training.model_initializer import DatasetMetadata

        metadata = DatasetMetadata()
        assert metadata.in_channels is None
        assert metadata.globals_dim is None
        assert metadata.policy_encoding is None
        assert metadata.history_length is None
        assert metadata.feature_version is None
        assert metadata.inferred_policy_size is None
        assert metadata.num_heuristics is None

    def test_metadata_with_values(self):
        """Test DatasetMetadata with values."""
        from app.training.model_initializer import DatasetMetadata

        metadata = DatasetMetadata(
            in_channels=64,
            globals_dim=20,
            policy_encoding="sparse",
            history_length=3,
            feature_version=2,
            inferred_policy_size=3782,
            num_heuristics=49,
        )
        assert metadata.in_channels == 64
        assert metadata.globals_dim == 20
        assert metadata.policy_encoding == "sparse"
        assert metadata.history_length == 3
        assert metadata.feature_version == 2
        assert metadata.inferred_policy_size == 3782
        assert metadata.num_heuristics == 49


class TestModelInitResult:
    """Tests for ModelInitResult dataclass."""

    def test_create_result(self):
        """Test creating ModelInitResult."""
        from app.training.model_initializer import ModelInitResult

        mock_model = MagicMock()
        result = ModelInitResult(
            model=mock_model,
            policy_size=3782,
            board_size=9,
            effective_blocks=6,
            effective_filters=96,
            feature_version=1,
            is_hex_model=True,
            hex_radius=4,
        )
        assert result.model == mock_model
        assert result.policy_size == 3782
        assert result.board_size == 9
        assert result.effective_blocks == 6
        assert result.effective_filters == 96
        assert result.feature_version == 1
        assert result.is_hex_model is True
        assert result.hex_radius == 4

    def test_result_defaults(self):
        """Test ModelInitResult default values."""
        from app.training.model_initializer import ModelInitResult

        result = ModelInitResult(
            model=MagicMock(),
            policy_size=128,
            board_size=8,
            effective_blocks=4,
            effective_filters=64,
            feature_version=1,
        )
        assert result.is_hex_model is False
        assert result.hex_radius is None


class TestModelInitializer:
    """Tests for ModelInitializer class."""

    @pytest.fixture
    def mock_device(self):
        """Create mock torch device."""
        import torch
        return torch.device("cpu")

    @pytest.fixture
    def default_config(self):
        """Create default ModelConfig."""
        from app.coordination.types import BoardType
        from app.training.model_initializer import ModelConfig

        return ModelConfig(board_type=BoardType.SQUARE8, num_players=2)

    def test_initializer_creation(self, default_config, mock_device):
        """Test creating ModelInitializer."""
        from app.training.model_initializer import ModelInitializer

        initializer = ModelInitializer(default_config, mock_device)
        assert initializer.config == default_config
        assert initializer.device == mock_device
        assert initializer.distributed is False
        assert initializer.is_main_process is True

    def test_initializer_distributed(self, default_config, mock_device):
        """Test initializer with distributed settings."""
        from app.training.model_initializer import ModelInitializer

        initializer = ModelInitializer(
            default_config,
            mock_device,
            distributed=True,
            is_main_process=False,
        )
        assert initializer.distributed is True
        assert initializer.is_main_process is False


class TestBoardSizeDetermination:
    """Tests for board size determination methods."""

    @pytest.fixture
    def mock_device(self):
        """Create mock torch device."""
        import torch
        return torch.device("cpu")

    def test_get_board_size_square8(self, mock_device):
        """Test board size for square8."""
        from app.coordination.types import BoardType
        from app.training.model_initializer import ModelConfig, ModelInitializer

        config = ModelConfig(board_type=BoardType.SQUARE8)
        initializer = ModelInitializer(config, mock_device)
        assert initializer._get_board_size() == 8

    def test_get_board_size_square19(self, mock_device):
        """Test board size for square19."""
        from app.coordination.types import BoardType
        from app.training.model_initializer import ModelConfig, ModelInitializer

        config = ModelConfig(board_type=BoardType.SQUARE19)
        initializer = ModelInitializer(config, mock_device)
        assert initializer._get_board_size() == 19

    def test_get_board_size_hex8(self, mock_device):
        """Test board size for hex8."""
        from app.coordination.types import BoardType
        from app.training.model_initializer import ModelConfig, ModelInitializer

        config = ModelConfig(board_type=BoardType.HEX8)
        initializer = ModelInitializer(config, mock_device)
        assert initializer._get_board_size() == 9  # HEX8_BOARD_SIZE

    def test_get_board_size_hexagonal(self, mock_device):
        """Test board size for large hexagonal."""
        from app.coordination.types import BoardType
        from app.training.model_initializer import ModelConfig, ModelInitializer

        config = ModelConfig(board_type=BoardType.HEXAGONAL)
        initializer = ModelInitializer(config, mock_device)
        assert initializer._get_board_size() == 25  # HEX_BOARD_SIZE

    def test_is_hex_model_true(self, mock_device):
        """Test hex model detection for hex boards."""
        from app.coordination.types import BoardType
        from app.training.model_initializer import ModelConfig, ModelInitializer

        for board_type in [BoardType.HEX8, BoardType.HEXAGONAL]:
            config = ModelConfig(board_type=board_type)
            initializer = ModelInitializer(config, mock_device)
            assert initializer._is_hex_model() is True

    def test_is_hex_model_false(self, mock_device):
        """Test hex model detection for square boards."""
        from app.coordination.types import BoardType
        from app.training.model_initializer import ModelConfig, ModelInitializer

        for board_type in [BoardType.SQUARE8, BoardType.SQUARE19]:
            config = ModelConfig(board_type=board_type)
            initializer = ModelInitializer(config, mock_device)
            assert initializer._is_hex_model() is False

    def test_get_hex_radius_hex8(self, mock_device):
        """Test hex radius for hex8."""
        from app.coordination.types import BoardType
        from app.training.model_initializer import ModelConfig, ModelInitializer

        config = ModelConfig(board_type=BoardType.HEX8)
        initializer = ModelInitializer(config, mock_device)
        assert initializer._get_hex_radius() == 4

    def test_get_hex_radius_hexagonal(self, mock_device):
        """Test hex radius for large hexagonal."""
        from app.coordination.types import BoardType
        from app.training.model_initializer import ModelConfig, ModelInitializer

        config = ModelConfig(board_type=BoardType.HEXAGONAL)
        initializer = ModelInitializer(config, mock_device)
        assert initializer._get_hex_radius() == 12


class TestEffectiveArchitecture:
    """Tests for effective architecture determination."""

    @pytest.fixture
    def mock_device(self):
        """Create mock torch device."""
        import torch
        return torch.device("cpu")

    def test_default_architecture(self, mock_device):
        """Test default blocks and filters."""
        from app.coordination.types import BoardType
        from app.training.model_initializer import ModelConfig, ModelInitializer

        config = ModelConfig(board_type=BoardType.SQUARE8)
        initializer = ModelInitializer(config, mock_device)
        blocks, filters = initializer._get_effective_architecture()
        assert blocks == 6  # Default
        assert filters == 96  # Default

    def test_custom_architecture(self, mock_device):
        """Test custom blocks and filters."""
        from app.coordination.types import BoardType
        from app.training.model_initializer import ModelConfig, ModelInitializer

        config = ModelConfig(
            board_type=BoardType.SQUARE8,
            num_res_blocks=10,
            num_filters=128,
        )
        initializer = ModelInitializer(config, mock_device)
        blocks, filters = initializer._get_effective_architecture()
        assert blocks == 10
        assert filters == 128


class TestPolicySizeDetermination:
    """Tests for policy size determination."""

    @pytest.fixture
    def mock_device(self):
        """Create mock torch device."""
        import torch
        return torch.device("cpu")

    def test_explicit_policy_size(self, mock_device):
        """Test explicit policy size from config."""
        from app.coordination.types import BoardType
        from app.training.model_initializer import ModelConfig, ModelInitializer

        config = ModelConfig(board_type=BoardType.SQUARE8, policy_size=256)
        initializer = ModelInitializer(config, mock_device)
        size = initializer._determine_policy_size(None, 8, False)
        assert size == 256

    def test_inferred_policy_size_from_metadata(self, mock_device):
        """Test policy size from dataset metadata."""
        from app.coordination.types import BoardType
        from app.training.model_initializer import (
            DatasetMetadata,
            ModelConfig,
            ModelInitializer,
        )

        config = ModelConfig(board_type=BoardType.SQUARE8)
        initializer = ModelInitializer(config, mock_device)
        metadata = DatasetMetadata(inferred_policy_size=512)
        size = initializer._determine_policy_size(metadata, 8, False)
        assert size == 512

    def test_default_policy_size_square(self, mock_device):
        """Test default policy size for square board."""
        from app.coordination.types import BoardType
        from app.training.model_initializer import ModelConfig, ModelInitializer

        config = ModelConfig(board_type=BoardType.SQUARE8)
        initializer = ModelInitializer(config, mock_device)
        size = initializer._determine_policy_size(None, 8, False)
        assert size == 8 * 8 * 2  # 128

    def test_default_policy_size_hex8(self, mock_device):
        """Test default policy size for hex8 board."""
        from app.coordination.types import BoardType
        from app.training.model_initializer import ModelConfig, ModelInitializer

        config = ModelConfig(board_type=BoardType.HEX8)
        initializer = ModelInitializer(config, mock_device)
        size = initializer._determine_policy_size(None, 9, True)
        assert size == 61 * 62  # hex8 default


class TestWeightLoading:
    """Tests for weight loading methods."""

    @pytest.fixture
    def mock_device(self):
        """Create mock torch device."""
        import torch
        return torch.device("cpu")

    @pytest.fixture
    def default_config(self):
        """Create default ModelConfig."""
        from app.coordination.types import BoardType
        from app.training.model_initializer import ModelConfig

        return ModelConfig(board_type=BoardType.SQUARE8, num_players=2)

    def test_load_init_weights_file_not_found(self, default_config, mock_device):
        """Test loading init weights from nonexistent file."""
        from app.training.model_initializer import ModelInitializer

        initializer = ModelInitializer(default_config, mock_device)
        mock_model = MagicMock()

        with pytest.raises(FileNotFoundError):
            initializer.load_init_weights(mock_model, "/nonexistent/path.pth")

    def test_load_checkpoint_no_file(self, default_config, mock_device):
        """Test loading checkpoint from nonexistent file (silent)."""
        from app.training.model_initializer import ModelInitializer

        initializer = ModelInitializer(default_config, mock_device)
        mock_model = MagicMock()

        # Should not raise - silent return
        initializer.load_checkpoint_weights(mock_model, "/nonexistent/path.pth")


class TestDDPWrapping:
    """Tests for DDP wrapping."""

    @pytest.fixture
    def mock_device(self):
        """Create mock torch device."""
        import torch
        return torch.device("cpu")

    @pytest.fixture
    def default_config(self):
        """Create default ModelConfig."""
        from app.coordination.types import BoardType
        from app.training.model_initializer import ModelConfig

        return ModelConfig(board_type=BoardType.SQUARE8, num_players=2)

    def test_wrap_ddp_disabled(self, default_config, mock_device):
        """Test DDP wrap when distributed is disabled."""
        from app.training.model_initializer import ModelInitializer

        initializer = ModelInitializer(
            default_config, mock_device, distributed=False
        )
        mock_model = MagicMock()

        result = initializer.wrap_ddp(mock_model)
        assert result is mock_model  # Returns same model


class TestValueHeadValidation:
    """Tests for value head validation."""

    @pytest.fixture
    def mock_device(self):
        """Create mock torch device."""
        import torch
        return torch.device("cpu")

    def test_validate_value_head_matching_players(self, mock_device):
        """Test validation passes with matching players."""
        from app.coordination.types import BoardType
        from app.training.model_initializer import ModelConfig, ModelInitializer

        config = ModelConfig(board_type=BoardType.SQUARE8, num_players=2)
        initializer = ModelInitializer(config, mock_device)

        mock_model = MagicMock()
        mock_model.num_players = 2
        mock_model.value_fc3.out_features = 2  # Match expected players

        # Should not raise
        initializer._validate_value_head(mock_model)

    def test_validate_value_head_mismatched_players(self, mock_device):
        """Test validation fails with mismatched players."""
        from app.coordination.types import BoardType
        from app.training.model_initializer import ModelConfig, ModelInitializer

        config = ModelConfig(board_type=BoardType.SQUARE8, num_players=2)
        initializer = ModelInitializer(config, mock_device)

        mock_model = MagicMock()
        mock_model.num_players = 4
        # Remove value_fc3 and value_fc2 to avoid second check
        del mock_model.value_fc3
        del mock_model.value_fc2

        with pytest.raises(ValueError, match="mismatch"):
            initializer._validate_value_head(mock_model)

    def test_validate_value_head_layer_mismatch(self, mock_device):
        """Test validation fails with mismatched layer output."""
        from app.coordination.types import BoardType
        from app.training.model_initializer import ModelConfig, ModelInitializer

        config = ModelConfig(board_type=BoardType.SQUARE8, num_players=2)
        initializer = ModelInitializer(config, mock_device)

        mock_model = MagicMock()
        # Remove num_players attribute
        del mock_model.num_players
        # Set mismatched value layer
        mock_model.value_fc3 = MagicMock()
        mock_model.value_fc3.out_features = 4

        with pytest.raises(ValueError, match="out_features"):
            initializer._validate_value_head(mock_model)


class TestFreezePolicy:
    """Tests for freeze policy functionality."""

    @pytest.fixture
    def mock_device(self):
        """Create mock torch device."""
        import torch
        return torch.device("cpu")

    def test_apply_freeze_policy(self, mock_device):
        """Test that freeze policy freezes non-value parameters."""
        from app.coordination.types import BoardType
        from app.training.model_initializer import ModelConfig, ModelInitializer

        config = ModelConfig(board_type=BoardType.SQUARE8, freeze_policy=True)
        initializer = ModelInitializer(config, mock_device)

        # Create mock model with parameters
        mock_model = MagicMock()
        mock_params = [
            ("conv1.weight", MagicMock()),
            ("conv1.bias", MagicMock()),
            ("value_fc1.weight", MagicMock()),
            ("value_fc1.bias", MagicMock()),
            ("policy_fc.weight", MagicMock()),
        ]

        # Set up requires_grad
        for name, param in mock_params:
            param.requires_grad = True

        mock_model.parameters.return_value = [p for _, p in mock_params]
        mock_model.named_parameters.return_value = mock_params

        initializer._apply_freeze_policy(mock_model)

        # Check that conv1 and policy_fc are frozen
        assert mock_params[0][1].requires_grad is False  # conv1.weight
        assert mock_params[1][1].requires_grad is False  # conv1.bias
        assert mock_params[4][1].requires_grad is False  # policy_fc.weight

        # Check that value_fc parameters are unfrozen
        assert mock_params[2][1].requires_grad is True  # value_fc1.weight
        assert mock_params[3][1].requires_grad is True  # value_fc1.bias


class TestCreateModelInitializer:
    """Tests for create_model_initializer factory function."""

    def test_create_with_defaults(self):
        """Test factory with default values."""
        import torch

        from app.coordination.types import BoardType
        from app.training.model_initializer import create_model_initializer

        device = torch.device("cpu")
        initializer = create_model_initializer(
            board_type=BoardType.SQUARE8,
            num_players=2,
            device=device,
        )
        assert initializer.config.board_type == BoardType.SQUARE8
        assert initializer.config.num_players == 2
        assert initializer.config.model_version == "v2"
        assert initializer.device == device

    def test_create_with_custom_values(self):
        """Test factory with custom values."""
        import torch

        from app.coordination.types import BoardType
        from app.training.model_initializer import create_model_initializer

        device = torch.device("cpu")
        initializer = create_model_initializer(
            board_type=BoardType.HEX8,
            num_players=4,
            device=device,
            model_version="v3",
            num_res_blocks=8,
            num_filters=128,
            distributed=True,
            is_main_process=False,
        )
        assert initializer.config.board_type == BoardType.HEX8
        assert initializer.config.num_players == 4
        assert initializer.config.model_version == "v3"
        assert initializer.config.num_res_blocks == 8
        assert initializer.config.num_filters == 128
        assert initializer.distributed is True
        assert initializer.is_main_process is False


class TestMetadataExtraction:
    """Tests for dataset metadata extraction."""

    @pytest.fixture
    def mock_device(self):
        """Create mock torch device."""
        import torch
        return torch.device("cpu")

    @pytest.fixture
    def default_config(self):
        """Create default ModelConfig."""
        from app.coordination.types import BoardType
        from app.training.model_initializer import ModelConfig

        return ModelConfig(board_type=BoardType.SQUARE8, num_players=2)

    def test_extract_metadata_missing_file(self, default_config, mock_device):
        """Test extraction from missing file."""
        from app.training.model_initializer import DatasetMetadata, ModelInitializer

        initializer = ModelInitializer(default_config, mock_device)

        with patch("app.training.model_initializer.safe_load_npz") as mock_npz:
            mock_npz.side_effect = OSError("File not found")
            metadata = initializer._extract_dataset_metadata("/nonexistent.npz")

        # Should return empty metadata
        assert metadata.in_channels is None
        assert metadata.globals_dim is None


class TestMetadataValidation:
    """Tests for metadata validation."""

    @pytest.fixture
    def mock_device(self):
        """Create mock torch device."""
        import torch
        return torch.device("cpu")

    def test_validate_history_length_mismatch(self, mock_device):
        """Test validation fails with history length mismatch."""
        from app.coordination.types import BoardType
        from app.training.model_initializer import (
            DatasetMetadata,
            ModelConfig,
            ModelInitializer,
        )

        config = ModelConfig(board_type=BoardType.SQUARE8, history_length=3)
        initializer = ModelInitializer(config, mock_device)
        metadata = DatasetMetadata(history_length=5, globals_dim=20)

        with pytest.raises(ValueError, match="history_length"):
            initializer._validate_metadata(metadata, "/test.npz")

    def test_validate_feature_version_mismatch(self, mock_device):
        """Test validation fails with feature version mismatch."""
        from app.coordination.types import BoardType
        from app.training.model_initializer import (
            DatasetMetadata,
            ModelConfig,
            ModelInitializer,
        )

        config = ModelConfig(board_type=BoardType.SQUARE8, feature_version=1)
        initializer = ModelInitializer(config, mock_device)
        metadata = DatasetMetadata(
            feature_version=2, history_length=3, globals_dim=20
        )

        with pytest.raises(ValueError, match="feature_version"):
            initializer._validate_metadata(metadata, "/test.npz")

    def test_validate_missing_globals(self, mock_device):
        """Test validation fails with missing globals."""
        from app.coordination.types import BoardType
        from app.training.model_initializer import (
            DatasetMetadata,
            ModelConfig,
            ModelInitializer,
        )

        config = ModelConfig(board_type=BoardType.SQUARE8)
        initializer = ModelInitializer(config, mock_device)
        metadata = DatasetMetadata(globals_dim=None)

        with pytest.raises(ValueError, match="globals"):
            initializer._validate_metadata(metadata, "/test.npz")

    def test_validate_matching_metadata(self, mock_device):
        """Test validation passes with matching metadata."""
        from app.coordination.types import BoardType
        from app.training.model_initializer import (
            DatasetMetadata,
            ModelConfig,
            ModelInitializer,
        )

        config = ModelConfig(
            board_type=BoardType.SQUARE8,
            history_length=3,
            feature_version=1,
        )
        initializer = ModelInitializer(config, mock_device)
        metadata = DatasetMetadata(
            history_length=3,
            feature_version=1,
            globals_dim=20,
        )

        # Should not raise
        initializer._validate_metadata(metadata, "/test.npz")
