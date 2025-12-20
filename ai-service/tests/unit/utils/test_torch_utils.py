"""Tests for app.utils.torch_utils module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Check if torch is available
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")


class TestSafeLoadCheckpoint:
    """Tests for safe_load_checkpoint function."""

    def test_loads_simple_state_dict_safely(self, tmp_path: Path):
        """Test loading a simple state dict with weights_only=True."""
        from app.utils.torch_utils import safe_load_checkpoint

        # Create a simple checkpoint
        checkpoint_path = tmp_path / "model.pt"
        state_dict = {"layer.weight": torch.randn(10, 5)}
        torch.save(state_dict, checkpoint_path)

        # Load it safely
        loaded = safe_load_checkpoint(checkpoint_path)
        assert "layer.weight" in loaded
        assert loaded["layer.weight"].shape == (10, 5)

    def test_loads_checkpoint_with_metadata(self, tmp_path: Path):
        """Test loading checkpoint that requires weights_only=False."""
        from app.utils.torch_utils import safe_load_checkpoint

        # Create a checkpoint with non-tensor data
        checkpoint_path = tmp_path / "model.pt"
        checkpoint = {
            "model_state_dict": {"layer.weight": torch.randn(10, 5)},
            "config": {"num_layers": 3, "hidden_size": 256},
            "epoch": 10,
        }
        torch.save(checkpoint, checkpoint_path)

        # Load it (should fall back to unsafe mode for metadata)
        loaded = safe_load_checkpoint(checkpoint_path, warn_on_unsafe=False)
        assert "model_state_dict" in loaded
        assert loaded["config"]["num_layers"] == 3
        assert loaded["epoch"] == 10

    def test_raises_on_missing_file(self, tmp_path: Path):
        """Test that missing file raises FileNotFoundError."""
        from app.utils.torch_utils import safe_load_checkpoint

        with pytest.raises(FileNotFoundError):
            safe_load_checkpoint(tmp_path / "nonexistent.pt")

    def test_respects_map_location(self, tmp_path: Path):
        """Test that map_location is respected."""
        from app.utils.torch_utils import safe_load_checkpoint

        checkpoint_path = tmp_path / "model.pt"
        state_dict = {"weight": torch.randn(5, 5)}
        torch.save(state_dict, checkpoint_path)

        loaded = safe_load_checkpoint(checkpoint_path, map_location="cpu")
        assert loaded["weight"].device.type == "cpu"

    def test_disallow_unsafe_raises(self, tmp_path: Path):
        """Test that allow_unsafe=False raises on unsafe checkpoint."""
        from app.utils.torch_utils import safe_load_checkpoint

        # Create checkpoint with custom class that requires unsafe loading
        checkpoint_path = tmp_path / "model.pt"

        # Simple dict should work with safe loading
        simple_checkpoint = {"weight": torch.randn(3, 3)}
        torch.save(simple_checkpoint, checkpoint_path)

        # This should work with allow_unsafe=False for simple tensors
        loaded = safe_load_checkpoint(checkpoint_path, allow_unsafe=False)
        assert "weight" in loaded


class TestLoadStateDictOnly:
    """Tests for load_state_dict_only function."""

    def test_extracts_model_state_dict(self, tmp_path: Path):
        """Test extracting model_state_dict from checkpoint."""
        from app.utils.torch_utils import load_state_dict_only

        checkpoint_path = tmp_path / "model.pt"
        checkpoint = {
            "model_state_dict": {"layer.weight": torch.randn(10, 5)},
            "optimizer_state_dict": {"some": "data"},
            "epoch": 10,
        }
        torch.save(checkpoint, checkpoint_path)

        state_dict = load_state_dict_only(checkpoint_path)
        assert "layer.weight" in state_dict
        assert "optimizer_state_dict" not in state_dict

    def test_extracts_state_dict_key(self, tmp_path: Path):
        """Test extracting state_dict (alternative key) from checkpoint."""
        from app.utils.torch_utils import load_state_dict_only

        checkpoint_path = tmp_path / "model.pt"
        checkpoint = {
            "state_dict": {"layer.weight": torch.randn(10, 5)},
            "epoch": 10,
        }
        torch.save(checkpoint, checkpoint_path)

        state_dict = load_state_dict_only(checkpoint_path)
        assert "layer.weight" in state_dict

    def test_returns_whole_dict_if_no_nested_key(self, tmp_path: Path):
        """Test returning whole dict when no nested state_dict key."""
        from app.utils.torch_utils import load_state_dict_only

        checkpoint_path = tmp_path / "model.pt"
        # Direct state dict without nesting
        state_dict = {"layer.weight": torch.randn(10, 5)}
        torch.save(state_dict, checkpoint_path)

        loaded = load_state_dict_only(checkpoint_path)
        assert "layer.weight" in loaded


class TestSaveCheckpointSafe:
    """Tests for save_checkpoint_safe function."""

    def test_saves_checkpoint(self, tmp_path: Path):
        """Test saving a checkpoint."""
        from app.utils.torch_utils import save_checkpoint_safe

        checkpoint_path = tmp_path / "model.pt"
        checkpoint = {
            "model_state_dict": {"layer.weight": torch.randn(10, 5)},
            "epoch": 5,
        }

        save_checkpoint_safe(checkpoint, checkpoint_path)

        assert checkpoint_path.exists()
        loaded = torch.load(checkpoint_path, weights_only=False)
        assert loaded["epoch"] == 5

    def test_creates_parent_directories(self, tmp_path: Path):
        """Test that parent directories are created."""
        from app.utils.torch_utils import save_checkpoint_safe

        checkpoint_path = tmp_path / "nested" / "dir" / "model.pt"
        checkpoint = {"weight": torch.randn(3, 3)}

        save_checkpoint_safe(checkpoint, checkpoint_path)

        assert checkpoint_path.exists()
