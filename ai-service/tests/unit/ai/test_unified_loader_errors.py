"""Comprehensive error handling tests for UnifiedModelLoader.

Tests cover:
- Corrupted model file handling
- Missing required keys in state_dict
- Incompatible board type loading
- File not found errors
- Architecture mismatch detection
- GPU memory pressure scenarios
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
import torch.nn as nn

from app.ai.unified_loader import (
    InferredModelConfig,
    ModelArchitecture,
    UnifiedModelLoader,
    detect_architecture,
    infer_config_from_checkpoint,
)
from app.models import BoardType


class TestCorruptedFileHandling:
    """Tests for corrupted or malformed checkpoint files."""

    @patch("app.ai.unified_loader.UnifiedModelLoader._create_fresh_model")
    def test_empty_file_with_allow_fresh(self, mock_fresh):
        """Empty file should trigger fresh model creation when allow_fresh=True."""
        # Mock fresh model creation
        mock_model = MagicMock(spec=nn.Module)
        mock_fresh.return_value = Mock(
            model=mock_model,
            architecture=ModelArchitecture.CNN_V2,
            config=Mock(board_type=BoardType.SQUARE8),
        )

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            temp_path = Path(f.name)
            # Write invalid data
            f.write(b"not a pytorch checkpoint")

        try:
            loader = UnifiedModelLoader(device="cpu")
            loaded = loader.load(temp_path, board_type=BoardType.SQUARE8, allow_fresh=True)

            assert loaded.model is not None
            assert loaded.architecture == ModelArchitecture.CNN_V2
            mock_fresh.assert_called_once()
        finally:
            temp_path.unlink()

    def test_empty_file_without_allow_fresh(self):
        """Empty file should raise error when allow_fresh=False."""
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            temp_path = Path(f.name)
            f.write(b"corrupted data")

        try:
            loader = UnifiedModelLoader(device="cpu")
            with pytest.raises(Exception):  # Will raise during torch.load
                loader.load(temp_path, board_type=BoardType.SQUARE8, allow_fresh=False)
        finally:
            temp_path.unlink()

    def test_partially_corrupted_state_dict_strict_mode(self):
        """Partially corrupted state_dict should fail in strict mode."""
        # Create a checkpoint with some valid keys but missing critical ones
        partial_state = {
            "conv1.weight": torch.randn(96, 56, 3, 3),
            # Missing res_blocks and policy/value heads
        }

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            temp_path = Path(f.name)
            torch.save({"model_state_dict": partial_state}, f.name)

        try:
            loader = UnifiedModelLoader(device="cpu")
            with pytest.raises(RuntimeError):
                # Should fail when trying to load state dict
                loader.load(temp_path, board_type=BoardType.SQUARE8, strict=True, allow_fresh=False)
        finally:
            temp_path.unlink()

    def test_partially_corrupted_state_dict_non_strict_mode(self):
        """Partially corrupted state_dict should partially load in non-strict mode."""
        partial_state = {
            "conv1.weight": torch.randn(96, 56, 3, 3),
            "res_blocks.0.conv1.weight": torch.randn(96, 96, 3, 3),
        }

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            temp_path = Path(f.name)
            torch.save({"model_state_dict": partial_state}, f.name)

        try:
            loader = UnifiedModelLoader(device="cpu")
            # Non-strict mode should allow partial loading
            loaded = loader.load(temp_path, board_type=BoardType.SQUARE8, strict=False)
            assert loaded.model is not None
        finally:
            temp_path.unlink()

    @patch("app.ai.unified_loader.UnifiedModelLoader._create_fresh_model")
    @patch("app.utils.torch_utils.safe_load_checkpoint")
    def test_pickle_deserialization_error(self, mock_load, mock_fresh):
        """Should handle pickle deserialization errors gracefully."""
        mock_load.side_effect = RuntimeError("Pickle deserialization failed")

        # Mock fresh model creation
        mock_model = MagicMock(spec=nn.Module)
        mock_fresh.return_value = Mock(
            model=mock_model,
            architecture=ModelArchitecture.CNN_V2,
            config=Mock(board_type=BoardType.SQUARE8),
        )

        with tempfile.NamedTemporaryFile(suffix=".pth") as f:
            temp_path = Path(f.name)
            loader = UnifiedModelLoader(device="cpu")

            # With allow_fresh=True, should create fresh model
            loaded = loader.load(temp_path, board_type=BoardType.SQUARE8, allow_fresh=True)
            assert loaded.model is not None

            # With allow_fresh=False, should raise
            with pytest.raises(RuntimeError):
                loader.load(temp_path, board_type=BoardType.SQUARE8, allow_fresh=False)


class TestMissingRequiredKeys:
    """Tests for state_dict missing required keys."""

    def test_missing_conv1_in_cnn_architecture(self):
        """CNN architecture missing conv1 should be detected as UNKNOWN."""
        incomplete_state = {
            "res_blocks.0.conv1.weight": torch.randn(96, 96, 3, 3),
            "policy_fc.weight": torch.randn(7000, 384),
            # Missing conv1.weight
        }

        arch = detect_architecture(incomplete_state)
        # Without conv1, won't match CNN signature
        assert arch == ModelArchitecture.UNKNOWN

    def test_missing_accumulator_in_nnue(self):
        """NNUE architecture missing accumulator should not be detected as NNUE."""
        incomplete_state = {
            "hidden_blocks.0.weight": torch.randn(256, 256),
            "value_fc.weight": torch.randn(1, 256),
            # Missing accumulator
        }

        arch = detect_architecture(incomplete_state)
        assert arch != ModelArchitecture.NNUE_VALUE_ONLY
        assert arch != ModelArchitecture.NNUE_WITH_POLICY

    def test_missing_policy_head_keys(self):
        """State dict missing policy head keys should have default policy_size."""
        state = {
            "conv1.weight": torch.randn(96, 56, 3, 3),
            "res_blocks.0.conv1.weight": torch.randn(96, 96, 3, 3),
            "value_fc2.weight": torch.randn(2, 128),
            # No policy_fc keys
        }

        arch = detect_architecture(state)
        config = infer_config_from_checkpoint(state, arch)

        # Should use default policy_size
        assert config.policy_size == 4672  # Default value

    def test_missing_value_head_keys(self):
        """State dict missing value head should default to num_players=2."""
        state = {
            "conv1.weight": torch.randn(96, 56, 3, 3),
            "res_blocks.0.conv1.weight": torch.randn(96, 96, 3, 3),
            "policy_fc2.weight": torch.randn(7000, 384),
            # No value_fc keys
        }

        arch = detect_architecture(state)
        config = infer_config_from_checkpoint(state, arch)

        assert config.num_players == 2  # Default


class TestIncompatibleBoardTypeLoading:
    """Tests for loading models with incompatible board types."""

    def test_hex_model_loaded_as_square(self):
        """Loading hex model with square board type should detect mismatch."""
        # Create hex model state (has hex_mask)
        hex_state = {
            "conv1.weight": torch.randn(96, 56, 3, 3),
            "hex_mask": torch.ones(1, 1, 9, 9),
            "res_blocks.0.conv1.weight": torch.randn(96, 96, 3, 3),
            "policy_fc2.weight": torch.randn(4500, 384),
            "value_fc2.weight": torch.randn(2, 128),
        }

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            temp_path = Path(f.name)
            torch.save({"model_state_dict": hex_state}, f.name)

        try:
            loader = UnifiedModelLoader(device="cpu")
            loaded = loader.load(temp_path, board_type=BoardType.SQUARE8, strict=False)

            # Config should be inferred from checkpoint (HEX8), not override
            # The loader respects the checkpoint's architecture
            assert loaded.architecture in [ModelArchitecture.HEX_V2, ModelArchitecture.HEX_V3]
        finally:
            temp_path.unlink()

    def test_policy_size_mismatch_strict_mode(self):
        """Loading model with wrong policy size in strict mode should fail."""
        # Create model with legacy_max_n encoding (huge policy)
        wrong_policy_state = {
            "conv1.weight": torch.randn(96, 56, 3, 3),
            "res_blocks.0.conv1.weight": torch.randn(96, 96, 3, 3),
            "policy_fc2.weight": torch.randn(59000, 384),  # Legacy encoding
            "value_fc2.weight": torch.randn(2, 128),
        }

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            temp_path = Path(f.name)
            torch.save({"model_state_dict": wrong_policy_state}, f.name)

        try:
            loader = UnifiedModelLoader(device="cpu")
            # Should raise ModelConfigError in strict mode
            with pytest.raises(Exception):  # ModelConfigError
                loader.load(temp_path, board_type=BoardType.SQUARE8, strict=True)
        finally:
            temp_path.unlink()

    def test_policy_size_mismatch_non_strict_mode(self):
        """Loading model with wrong policy size in non-strict mode should warn."""
        # Create model with legacy encoding
        wrong_policy_state = {
            "conv1.weight": torch.randn(96, 56, 3, 3),
            "res_blocks.0.conv1.weight": torch.randn(96, 96, 3, 3),
            "policy_fc2.weight": torch.randn(59000, 384),  # Legacy
            "value_fc2.weight": torch.randn(2, 128),
        }

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            temp_path = Path(f.name)
            torch.save({"model_state_dict": wrong_policy_state}, f.name)

        try:
            loader = UnifiedModelLoader(device="cpu")
            # Should load but warn about mismatch
            loaded = loader.load(temp_path, board_type=BoardType.SQUARE8, strict=False)
            assert loaded.model is not None
            # Policy size will be mismatched
            assert loaded.config.policy_size == 59000
        finally:
            temp_path.unlink()


class TestFileNotFoundErrors:
    """Tests for missing checkpoint files."""

    @patch("app.ai.unified_loader.UnifiedModelLoader._create_fresh_model")
    def test_nonexistent_file_with_allow_fresh(self, mock_fresh):
        """Nonexistent file should create fresh model when allow_fresh=True."""
        # Mock fresh model creation
        mock_model = MagicMock(spec=nn.Module)
        mock_fresh.return_value = Mock(
            model=mock_model,
            architecture=ModelArchitecture.CNN_V2,
            config=Mock(board_type=BoardType.SQUARE8),
        )

        loader = UnifiedModelLoader(device="cpu")
        loaded = loader.load(
            "/nonexistent/path.pth",
            board_type=BoardType.SQUARE8,
            allow_fresh=True
        )

        assert loaded.model is not None
        assert loaded.architecture == ModelArchitecture.CNN_V2
        mock_fresh.assert_called_once()

    def test_nonexistent_file_without_allow_fresh(self):
        """Nonexistent file should raise FileNotFoundError when allow_fresh=False."""
        loader = UnifiedModelLoader(device="cpu")

        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/path.pth", allow_fresh=False)

    def test_directory_instead_of_file(self):
        """Passing directory instead of file should fail gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = UnifiedModelLoader(device="cpu")

            # Directory exists but is not a file
            with pytest.raises(Exception):
                loader.load(tmpdir, allow_fresh=False)


class TestArchitectureMismatchDetection:
    """Tests for architecture detection and mismatch scenarios."""

    def test_unknown_architecture_detection(self):
        """State dict with no recognizable pattern should be UNKNOWN."""
        unknown_state = {
            "mystery_layer.weight": torch.randn(100, 100),
            "enigma_block.bias": torch.randn(100),
        }

        arch = detect_architecture(unknown_state)
        assert arch == ModelArchitecture.UNKNOWN

    def test_nnue_vs_cnn_disambiguation(self):
        """NNUE and CNN should be correctly distinguished."""
        nnue_state = {
            "accumulator.weight": torch.randn(256, 100),
            "hidden_blocks.0.weight": torch.randn(256, 256),
        }

        cnn_state = {
            "conv1.weight": torch.randn(96, 56, 3, 3),
            "res_blocks.0.conv1.weight": torch.randn(96, 96, 3, 3),
        }

        assert detect_architecture(nnue_state) == ModelArchitecture.NNUE_VALUE_ONLY
        assert detect_architecture(cnn_state) == ModelArchitecture.CNN_V2

    def test_nnue_with_policy_vs_value_only(self):
        """NNUE with policy should be distinguished from value-only."""
        value_only_state = {
            "accumulator.weight": torch.randn(256, 100),
            "hidden_blocks.0.weight": torch.randn(256, 256),
            "value_fc.weight": torch.randn(1, 256),
        }

        with_policy_state = {
            "accumulator.weight": torch.randn(256, 100),
            "from_head.weight": torch.randn(1024, 256),
            "to_head.weight": torch.randn(1024, 256),
        }

        assert detect_architecture(value_only_state) == ModelArchitecture.NNUE_VALUE_ONLY
        assert detect_architecture(with_policy_state) == ModelArchitecture.NNUE_WITH_POLICY

    def test_hex_v2_vs_v3_detection(self):
        """Hex v2 and v3 should be correctly distinguished."""
        hex_v2_state = {
            "conv1.weight": torch.randn(96, 56, 3, 3),
            "hex_mask": torch.ones(1, 1, 9, 9),
            "res_blocks.0.conv1.weight": torch.randn(96, 96, 3, 3),
            "policy_fc2.weight": torch.randn(4500, 384),
        }

        hex_v3_state = {
            "conv1.weight": torch.randn(96, 56, 3, 3),
            "hex_mask": torch.ones(1, 1, 9, 9),
            "res_blocks.0.conv1.weight": torch.randn(96, 96, 3, 3),
            "placement_conv.weight": torch.randn(1, 96, 1, 1),
            "movement_conv.weight": torch.randn(1, 96, 1, 1),
        }

        assert detect_architecture(hex_v2_state) == ModelArchitecture.HEX_V2
        assert detect_architecture(hex_v3_state) == ModelArchitecture.HEX_V3

    def test_cnn_v2_vs_v3_vs_v4_detection(self):
        """CNN v2, v3, and v4 should be correctly distinguished."""
        cnn_v2_state = {
            "conv1.weight": torch.randn(96, 56, 3, 3),
            "res_blocks.0.conv1.weight": torch.randn(96, 96, 3, 3),
            "policy_fc2.weight": torch.randn(7000, 384),
        }

        cnn_v3_state = {
            "conv1.weight": torch.randn(96, 56, 3, 3),
            "res_blocks.0.conv1.weight": torch.randn(96, 96, 3, 3),
            "placement_conv.weight": torch.randn(1, 96, 1, 1),
            "movement_conv.weight": torch.randn(1, 96, 1, 1),
        }

        cnn_v4_state = {
            "conv1.weight": torch.randn(96, 56, 3, 3),
            "res_blocks.0.conv1.weight": torch.randn(96, 96, 3, 3),
            "res_blocks.0.attn.weight": torch.randn(96, 96),  # Has attention
        }

        assert detect_architecture(cnn_v2_state) == ModelArchitecture.CNN_V2
        assert detect_architecture(cnn_v3_state) == ModelArchitecture.CNN_V3
        assert detect_architecture(cnn_v4_state) == ModelArchitecture.CNN_V4


class TestGPUMemoryPressure:
    """Tests for GPU availability and memory pressure scenarios."""

    @patch("torch.cuda.is_available")
    def test_cuda_unavailable_falls_back_to_cpu(self, mock_cuda):
        """When CUDA is unavailable, should use CPU."""
        mock_cuda.return_value = False

        # Patch MPS check to avoid issues on systems without MPS
        with patch("torch.backends.mps.is_available", return_value=False):
            loader = UnifiedModelLoader()  # Auto-detect device
            assert loader.device == torch.device("cpu")

    @patch("torch.cuda.is_available")
    @patch("torch.backends.mps.is_available")
    @patch.dict("os.environ", {"RINGRIFT_FORCE_CPU": "0"}, clear=False)
    def test_mps_available_uses_mps(self, mock_mps, mock_cuda):
        """When MPS is available and CUDA is not, should use MPS."""
        mock_cuda.return_value = False
        mock_mps.return_value = True

        loader = UnifiedModelLoader()
        assert loader.device == torch.device("mps")

    @patch("torch.cuda.is_available")
    def test_explicit_cpu_device_override(self, mock_cuda):
        """Explicit CPU device should override CUDA availability."""
        mock_cuda.return_value = True

        loader = UnifiedModelLoader(device="cpu")
        assert loader.device == torch.device("cpu")

    def test_device_string_conversion(self):
        """Device string should be converted to torch.device."""
        loader = UnifiedModelLoader(device="cpu")
        assert isinstance(loader.device, torch.device)
        assert loader.device.type == "cpu"

    def test_device_object_passthrough(self):
        """torch.device object should be used as-is."""
        device = torch.device("cpu")
        loader = UnifiedModelLoader(device=device)
        assert loader.device == device


class TestModelInstantiationFailures:
    """Tests for model instantiation errors."""

    @patch("app.ai.unified_loader.UnifiedModelLoader._create_fresh_model")
    @patch("app.ai.unified_loader.UnifiedModelLoader._instantiate_model")
    def test_instantiation_error_with_allow_fresh(self, mock_instantiate, mock_fresh):
        """Instantiation error should create fresh model when allow_fresh=True."""
        mock_instantiate.side_effect = ValueError("Invalid architecture config")

        # Mock fresh model creation
        mock_model = MagicMock(spec=nn.Module)
        mock_fresh.return_value = Mock(
            model=mock_model,
            architecture=ModelArchitecture.CNN_V2,
            config=Mock(board_type=BoardType.SQUARE8),
        )

        state = {
            "conv1.weight": torch.randn(96, 56, 3, 3),
            "res_blocks.0.conv1.weight": torch.randn(96, 96, 3, 3),
        }

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            temp_path = Path(f.name)
            torch.save({"model_state_dict": state}, f.name)

        try:
            loader = UnifiedModelLoader(device="cpu")
            loaded = loader.load(temp_path, board_type=BoardType.SQUARE8, allow_fresh=True)
            assert loaded.model is not None
            mock_fresh.assert_called_once()
        finally:
            temp_path.unlink()

    @patch("app.ai.unified_loader.UnifiedModelLoader._instantiate_model")
    def test_instantiation_error_without_allow_fresh(self, mock_instantiate):
        """Instantiation error should raise when allow_fresh=False."""
        mock_instantiate.side_effect = ValueError("Invalid config")

        state = {
            "conv1.weight": torch.randn(96, 56, 3, 3),
            "res_blocks.0.conv1.weight": torch.randn(96, 96, 3, 3),
        }

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            temp_path = Path(f.name)
            torch.save({"model_state_dict": state}, f.name)

        try:
            loader = UnifiedModelLoader(device="cpu")
            with pytest.raises(ValueError):
                loader.load(temp_path, board_type=BoardType.SQUARE8, allow_fresh=False)
        finally:
            temp_path.unlink()


class TestConfigInference:
    """Tests for configuration inference from checkpoints."""

    def test_infer_board_type_from_policy_size(self):
        """Board type should be inferred from policy size."""
        state = {
            "policy_fc2.weight": torch.randn(7000, 384),  # SQUARE8 policy size
        }

        arch = ModelArchitecture.CNN_V2
        config = infer_config_from_checkpoint(state, arch)
        assert config.board_type == BoardType.SQUARE8

    def test_infer_board_type_from_hex_mask_size(self):
        """Board type should be inferred from hex_mask spatial dimensions."""
        state = {
            "hex_mask": torch.ones(1, 1, 25, 25),  # HEXAGONAL size
        }

        arch = ModelArchitecture.HEX_V2
        config = infer_config_from_checkpoint(state, arch)
        assert config.board_type == BoardType.HEXAGONAL
        assert config.hex_radius == 12

    def test_infer_num_players_from_value_head(self):
        """Number of players should be inferred from value head output size."""
        state = {
            "value_fc2.weight": torch.randn(4, 128),  # 4 players
        }

        arch = ModelArchitecture.CNN_V2
        config = infer_config_from_checkpoint(state, arch)
        assert config.num_players == 4

    def test_infer_num_res_blocks_from_state_dict(self):
        """Number of residual blocks should be counted from state dict keys."""
        state = {
            "res_blocks.0.conv1.weight": torch.randn(96, 96, 3, 3),
            "res_blocks.1.conv1.weight": torch.randn(96, 96, 3, 3),
            "res_blocks.2.conv1.weight": torch.randn(96, 96, 3, 3),
            "res_blocks.3.conv1.weight": torch.randn(96, 96, 3, 3),
            "res_blocks.4.conv1.weight": torch.randn(96, 96, 3, 3),
        }

        arch = ModelArchitecture.CNN_V2
        config = infer_config_from_checkpoint(state, arch)
        assert config.num_res_blocks == 5

    def test_infer_history_length_from_conv1_channels(self):
        """History length should be inferred from conv1 input channels."""
        # 56 total channels = 14 base * (3 history + 1 current)
        state = {
            "conv1.weight": torch.randn(96, 56, 3, 3),
        }

        arch = ModelArchitecture.CNN_V2
        config = infer_config_from_checkpoint(state, arch)
        assert config.input_channels == 14
        assert config.history_length == 3

    def test_detect_lite_variant_from_dimensions(self):
        """Lite variant should be detected from smaller dimensions."""
        lite_state = {
            "conv1.weight": torch.randn(96, 56, 3, 3),  # 96 filters
            "policy_fc1.weight": torch.randn(192, 1000),  # 192 intermediate
            "value_fc1.weight": torch.randn(64, 1000),  # 64 intermediate
        }

        arch = ModelArchitecture.CNN_V2
        config = infer_config_from_checkpoint(lite_state, arch)
        assert config.is_lite_variant is True

    def test_detect_full_variant_from_dimensions(self):
        """Full variant should be detected from larger dimensions."""
        full_state = {
            "conv1.weight": torch.randn(192, 56, 3, 3),  # 192 filters
            "policy_fc1.weight": torch.randn(384, 1000),  # 384 intermediate
            "value_fc1.weight": torch.randn(128, 1000),  # 128 intermediate
        }

        arch = ModelArchitecture.CNN_V2
        config = infer_config_from_checkpoint(full_state, arch)
        assert config.is_lite_variant is False


class TestCachingBehavior:
    """Tests for model caching functionality."""

    def test_cache_hit_returns_same_model(self):
        """Second load of same path should hit cache."""
        state = {
            "conv1.weight": torch.randn(96, 56, 3, 3),
            "res_blocks.0.conv1.weight": torch.randn(96, 96, 3, 3),
            "policy_fc2.weight": torch.randn(7000, 384),
            "value_fc2.weight": torch.randn(2, 128),
        }

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            temp_path = Path(f.name)
            torch.save({"model_state_dict": state}, f.name)

        try:
            loader = UnifiedModelLoader(device="cpu", cache_enabled=True)
            loaded1 = loader.load(temp_path, board_type=BoardType.SQUARE8)
            loaded2 = loader.load(temp_path, board_type=BoardType.SQUARE8)

            # Should be same cached object
            assert loaded1 is loaded2
        finally:
            temp_path.unlink()
            UnifiedModelLoader.clear_cache()

    def test_cache_disabled_loads_fresh(self):
        """With cache disabled, should load fresh each time."""
        state = {
            "conv1.weight": torch.randn(96, 56, 3, 3),
            "res_blocks.0.conv1.weight": torch.randn(96, 96, 3, 3),
            "policy_fc2.weight": torch.randn(7000, 384),
            "value_fc2.weight": torch.randn(2, 128),
        }

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            temp_path = Path(f.name)
            torch.save({"model_state_dict": state}, f.name)

        try:
            loader = UnifiedModelLoader(device="cpu", cache_enabled=False)
            loaded1 = loader.load(temp_path, board_type=BoardType.SQUARE8)
            loaded2 = loader.load(temp_path, board_type=BoardType.SQUARE8)

            # Should be different objects
            assert loaded1 is not loaded2
        finally:
            temp_path.unlink()

    def test_cache_eviction_on_max_size(self):
        """Cache should evict oldest entry when max size reached."""
        loader = UnifiedModelLoader(device="cpu", cache_enabled=True, max_cache_size=2)

        # Create 3 temporary checkpoints
        paths = []
        for i in range(3):
            state = {
                "conv1.weight": torch.randn(96, 56, 3, 3),
                "res_blocks.0.conv1.weight": torch.randn(96, 96, 3, 3),
                "policy_fc2.weight": torch.randn(7000, 384),
                "value_fc2.weight": torch.randn(2, 128),
            }
            f = tempfile.NamedTemporaryFile(suffix=f"_{i}.pth", delete=False)
            paths.append(Path(f.name))
            torch.save({"model_state_dict": state}, f.name)
            f.close()

        try:
            # Load all 3 (max size is 2, so first should be evicted)
            loader.load(paths[0], board_type=BoardType.SQUARE8)
            loader.load(paths[1], board_type=BoardType.SQUARE8)
            loader.load(paths[2], board_type=BoardType.SQUARE8)

            # Cache should only have 2 entries
            assert len(loader._cache) == 2
        finally:
            for p in paths:
                p.unlink()
            UnifiedModelLoader.clear_cache()

    def test_clear_cache(self):
        """clear_cache should empty the cache and free memory."""
        state = {
            "conv1.weight": torch.randn(96, 56, 3, 3),
            "res_blocks.0.conv1.weight": torch.randn(96, 96, 3, 3),
            "policy_fc2.weight": torch.randn(7000, 384),
            "value_fc2.weight": torch.randn(2, 128),
        }

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            temp_path = Path(f.name)
            torch.save({"model_state_dict": state}, f.name)

        try:
            loader = UnifiedModelLoader(device="cpu", cache_enabled=True)
            loader.load(temp_path, board_type=BoardType.SQUARE8)
            assert len(loader._cache) > 0

            UnifiedModelLoader.clear_cache()
            assert len(loader._cache) == 0
        finally:
            temp_path.unlink()
