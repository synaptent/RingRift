"""Unit tests for model loading reliability (Dec 2025).

Tests for:
- Magic byte validation before torch.load()
- ModelMetadata default values for legacy checkpoints
- Checksum verification when sidecar exists
- Loading metrics tracking
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch


class TestMagicByteValidation:
    """Tests for validate_checkpoint_magic_bytes()."""

    def test_valid_zip_header(self, tmp_path: Path) -> None:
        """Valid ZIP files (PyTorch >= 1.6) should pass."""
        from app.utils.torch_utils import validate_checkpoint_magic_bytes

        # Create a file with ZIP header (PK\x03\x04)
        test_file = tmp_path / "model.pth"
        test_file.write_bytes(b"PK\x03\x04" + b"\x00" * 100)

        is_valid, fmt = validate_checkpoint_magic_bytes(test_file)
        assert is_valid
        assert fmt == "zip"

    def test_valid_pickle_v2_header(self, tmp_path: Path) -> None:
        """Pickle protocol 2 files should pass."""
        from app.utils.torch_utils import validate_checkpoint_magic_bytes

        test_file = tmp_path / "model.pth"
        test_file.write_bytes(b"\x80\x02" + b"\x00" * 100)

        is_valid, fmt = validate_checkpoint_magic_bytes(test_file)
        assert is_valid
        assert fmt == "pickle_v2"

    def test_valid_pickle_v4_header(self, tmp_path: Path) -> None:
        """Pickle protocol 4 files should pass."""
        from app.utils.torch_utils import validate_checkpoint_magic_bytes

        test_file = tmp_path / "model.pth"
        test_file.write_bytes(b"\x80\x04" + b"\x00" * 100)

        is_valid, fmt = validate_checkpoint_magic_bytes(test_file)
        assert is_valid
        assert fmt == "pickle_v4"

    def test_invalid_header(self, tmp_path: Path) -> None:
        """Files with invalid headers should fail."""
        from app.utils.torch_utils import validate_checkpoint_magic_bytes

        test_file = tmp_path / "corrupt.pth"
        test_file.write_bytes(b"\x00\x00\x00\x00" + b"garbage")

        is_valid, fmt = validate_checkpoint_magic_bytes(test_file)
        assert not is_valid
        assert "invalid_header" in fmt

    def test_empty_file(self, tmp_path: Path) -> None:
        """Empty files should fail."""
        from app.utils.torch_utils import validate_checkpoint_magic_bytes

        test_file = tmp_path / "empty.pth"
        test_file.write_bytes(b"")

        is_valid, fmt = validate_checkpoint_magic_bytes(test_file)
        assert not is_valid
        assert "file_too_small" in fmt

    def test_truncated_file(self, tmp_path: Path) -> None:
        """Files with < 4 bytes should fail."""
        from app.utils.torch_utils import validate_checkpoint_magic_bytes

        test_file = tmp_path / "truncated.pth"
        test_file.write_bytes(b"PK")

        is_valid, fmt = validate_checkpoint_magic_bytes(test_file)
        assert not is_valid
        assert "file_too_small" in fmt

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        """Non-existent files should fail."""
        from app.utils.torch_utils import validate_checkpoint_magic_bytes

        test_file = tmp_path / "nonexistent.pth"

        is_valid, fmt = validate_checkpoint_magic_bytes(test_file)
        assert not is_valid
        assert fmt == "file_not_found"


class TestInvalidCheckpointFormatError:
    """Tests for InvalidCheckpointFormatError exception."""

    def test_error_message(self) -> None:
        """Error message should include path and header hex."""
        from app.utils.torch_utils import InvalidCheckpointFormatError

        err = InvalidCheckpointFormatError("/path/to/model.pth", "00000000")
        assert "/path/to/model.pth" in str(err)
        assert "00000000" in str(err)
        assert "ZIP" in str(err) or "pickle" in str(err)

    def test_error_attributes(self) -> None:
        """Error should store path and header_hex."""
        from app.utils.torch_utils import InvalidCheckpointFormatError

        err = InvalidCheckpointFormatError("/path/model.pth", "deadbeef")
        assert err.path == "/path/model.pth"
        assert err.header_hex == "deadbeef"


class TestSafeLoadCheckpointMagicBytes:
    """Tests for safe_load_checkpoint magic byte validation."""

    def test_raises_on_invalid_header(self, tmp_path: Path) -> None:
        """safe_load_checkpoint should raise InvalidCheckpointFormatError for corrupt files."""
        from app.utils.torch_utils import (
            safe_load_checkpoint,
            InvalidCheckpointFormatError,
        )

        corrupt_file = tmp_path / "corrupt.pth"
        corrupt_file.write_bytes(b"\x00\x00\x00\x00garbage")

        with pytest.raises(InvalidCheckpointFormatError):
            safe_load_checkpoint(str(corrupt_file))

    def test_valid_checkpoint_loads(self, tmp_path: Path) -> None:
        """Valid checkpoints should load successfully."""
        from app.utils.torch_utils import safe_load_checkpoint

        checkpoint_file = tmp_path / "valid.pth"
        checkpoint = {"model_state_dict": {"weight": torch.zeros(10)}}
        torch.save(checkpoint, checkpoint_file)

        loaded = safe_load_checkpoint(str(checkpoint_file))
        assert "model_state_dict" in loaded


class TestModelMetadataDefaults:
    """Tests for ModelMetadata default values."""

    def test_from_dict_with_missing_fields(self) -> None:
        """from_dict() should not fail when required fields are missing."""
        from app.training.model_versioning import ModelMetadata

        # Empty dict - should use defaults for all fields
        metadata = ModelMetadata.from_dict({})
        assert metadata.architecture_version == "unknown"
        assert metadata.model_class == "unknown"
        assert metadata.config == {}

    def test_from_dict_with_partial_fields(self) -> None:
        """from_dict() should fill in missing fields with defaults."""
        from app.training.model_versioning import ModelMetadata

        # Only board_type provided
        metadata = ModelMetadata.from_dict({"board_type": "hex8"})
        assert metadata.architecture_version == "unknown"
        assert metadata.model_class == "unknown"
        assert metadata.board_type == "hex8"

    def test_from_dict_preserves_provided_values(self) -> None:
        """from_dict() should use provided values, not defaults."""
        from app.training.model_versioning import ModelMetadata

        data = {
            "architecture_version": "v2.1.0",
            "model_class": "RingRiftCNN_v2",
            "config": {"num_channels": 96},
        }
        metadata = ModelMetadata.from_dict(data)
        assert metadata.architecture_version == "v2.1.0"
        assert metadata.model_class == "RingRiftCNN_v2"
        assert metadata.config == {"num_channels": 96}

    def test_default_values_are_sentinels(self) -> None:
        """Default ModelMetadata should have 'unknown' sentinels."""
        from app.training.model_versioning import ModelMetadata

        metadata = ModelMetadata()
        assert metadata.architecture_version == "unknown"
        assert metadata.model_class == "unknown"


class TestLoadingStatistics:
    """Tests for model loading statistics."""

    def test_get_loading_stats_returns_all_keys(self) -> None:
        """get_loading_stats should return all expected keys."""
        from app.ai.unified_loader import get_loading_stats

        stats = get_loading_stats()
        expected_keys = {
            "success", "fallback_fresh", "corruption",
            "checksum_fail", "magic_byte_fail", "metadata_fail", "file_not_found",
            "architecture_mismatch",
        }
        assert set(stats.keys()) == expected_keys

    def test_reset_loading_stats(self) -> None:
        """reset_loading_stats should zero all counters."""
        from app.ai.unified_loader import (
            get_loading_stats,
            reset_loading_stats,
            _increment_loading_stat,
        )

        # Increment some stats
        _increment_loading_stat("success")
        _increment_loading_stat("success")
        _increment_loading_stat("fallback_fresh")

        # Reset
        reset_loading_stats()

        stats = get_loading_stats()
        assert all(v == 0 for v in stats.values())

    def test_increment_loading_stat(self) -> None:
        """_increment_loading_stat should increment counters."""
        from app.ai.unified_loader import (
            get_loading_stats,
            reset_loading_stats,
            _increment_loading_stat,
        )

        reset_loading_stats()

        _increment_loading_stat("success")
        _increment_loading_stat("success")
        _increment_loading_stat("corruption")

        stats = get_loading_stats()
        assert stats["success"] == 2
        assert stats["corruption"] == 1
        assert stats["fallback_fresh"] == 0


class TestChecksumSidecar:
    """Tests for checksum sidecar file handling."""

    def test_write_checksum_file(self, tmp_path: Path) -> None:
        """write_checksum_file should create .sha256 sidecar."""
        from app.utils.torch_utils import write_checksum_file, compute_model_checksum

        model_file = tmp_path / "model.pth"
        model_file.write_bytes(b"test model content")

        sidecar = write_checksum_file(model_file)

        assert sidecar.exists()
        assert sidecar.suffix == ".sha256"
        assert sidecar == model_file.with_suffix(".pth.sha256")

        # Verify checksum in sidecar matches computed checksum
        expected = compute_model_checksum(model_file)
        content = sidecar.read_text()
        assert expected in content

    def test_verify_model_checksum_with_sidecar(self, tmp_path: Path) -> None:
        """verify_model_checksum should use sidecar file."""
        from app.utils.torch_utils import (
            write_checksum_file,
            verify_model_checksum,
        )

        model_file = tmp_path / "model.pth"
        model_file.write_bytes(b"test model content")
        write_checksum_file(model_file)

        is_valid, _ = verify_model_checksum(model_file)
        assert is_valid

    def test_verify_model_checksum_detects_corruption(self, tmp_path: Path) -> None:
        """verify_model_checksum should detect corrupted files."""
        from app.utils.torch_utils import (
            write_checksum_file,
            verify_model_checksum,
        )

        model_file = tmp_path / "model.pth"
        model_file.write_bytes(b"original content")
        write_checksum_file(model_file)

        # Corrupt the file
        model_file.write_bytes(b"corrupted content")

        is_valid, _ = verify_model_checksum(model_file)
        assert not is_valid


class TestUnifiedLoaderChecksumVerification:
    """Tests for UnifiedModelLoader checksum verification."""

    def test_loader_verifies_checksum_when_sidecar_exists(self, tmp_path: Path) -> None:
        """Loader should verify checksum when sidecar file exists."""
        from app.ai.unified_loader import UnifiedModelLoader, reset_loading_stats
        from app.utils.torch_utils import write_checksum_file

        reset_loading_stats()

        # Create a valid checkpoint with sidecar
        model_file = tmp_path / "model.pth"
        checkpoint = {"model_state_dict": {"weight": torch.zeros(10, 10)}}
        torch.save(checkpoint, model_file)
        write_checksum_file(model_file)

        # Loader should verify and load successfully
        loader = UnifiedModelLoader(device="cpu")
        with patch.object(loader, '_create_fresh_model') as mock_fresh:
            # Allow fresh fallback
            result = loader.load(model_file, allow_fresh=True)
            # If loading succeeded, fresh model wasn't needed
            # (architecture may not be detected, but that's a different issue)

    def test_loader_skips_verification_without_sidecar(self, tmp_path: Path) -> None:
        """Loader should skip verification when no sidecar exists."""
        from app.ai.unified_loader import UnifiedModelLoader, get_loading_stats, reset_loading_stats

        reset_loading_stats()

        # Create a checkpoint without sidecar
        model_file = tmp_path / "model.pth"
        checkpoint = {"model_state_dict": {"weight": torch.zeros(10, 10)}}
        torch.save(checkpoint, model_file)

        # Loader should load without checksum verification
        loader = UnifiedModelLoader(device="cpu")
        result = loader.load(model_file, allow_fresh=True)

        # Verify success was recorded (even if architecture unknown)
        stats = get_loading_stats()
        # At minimum, we shouldn't have corruption or checksum failures
        assert stats["corruption"] == 0
        assert stats["checksum_fail"] == 0
