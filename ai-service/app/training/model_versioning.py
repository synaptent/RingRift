"""
Model Versioning System for RingRift Neural Networks

This module provides explicit architecture versioning and validation for model
checkpoints. It prevents silent fallback to fresh weights on architecture
mismatch by failing explicitly with informative errors.

This module handles CHECKPOINT INTEGRITY:
- Architecture version tracking with semantic versioning
- SHA256 checksum for weight integrity verification
- Full metadata storage with each checkpoint (config, training info)
- Migration utilities for legacy .pth files
- Backwards compatible loading with deprecation warnings

Works with model_registry.py which handles MODEL LIFECYCLE:
- Track models across development → staging → production stages
- Store training configurations and performance metrics
- Support promotion workflows

Typical usage:
    from app.training.model_versioning import (
        save_versioned_checkpoint,
        load_versioned_checkpoint,
        VersionMismatchError,
    )

    # Save checkpoint with full metadata
    save_versioned_checkpoint(
        model=network,
        path="models/my_model.pth",
        architecture_version="2.1.0",
        metadata={"board_type": "square8", "training_epochs": 100},
    )

    # Load with version validation
    try:
        state_dict, metadata = load_versioned_checkpoint(
            path="models/my_model.pth",
            expected_version="2.1.0",
        )
    except VersionMismatchError as e:
        print(f"Cannot load: {e}")
"""

import hashlib
import logging
import os
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================

# Import base exception from canonical source (December 2025 consolidation)
from app.errors import ModelVersioningError as _BaseModelVersioningError

# Re-export for backwards compatibility
ModelVersioningError = _BaseModelVersioningError


class VersionMismatchError(ModelVersioningError):
    """Raised when checkpoint version doesn't match current model."""

    def __init__(
        self,
        checkpoint_version: str,
        current_version: str,
        checkpoint_path: str,
        details: str | None = None,
    ):
        self.checkpoint_version = checkpoint_version
        self.current_version = current_version
        self.checkpoint_path = checkpoint_path
        self.details = details

        message = (
            f"Architecture version mismatch!\n"
            f"  Checkpoint version: {checkpoint_version}\n"
            f"  Current version: {current_version}\n"
            f"  Checkpoint path: {checkpoint_path}"
        )
        if details:
            message += f"\n  Details: {details}"

        super().__init__(message)


class ChecksumMismatchError(ModelVersioningError):
    """Raised when checkpoint checksum doesn't match stored value."""

    def __init__(
        self,
        expected: str,
        actual: str,
        checkpoint_path: str,
    ):
        self.expected = expected
        self.actual = actual
        self.checkpoint_path = checkpoint_path

        message = (
            f"Checkpoint integrity check failed!\n"
            f"  Expected checksum: {expected}\n"
            f"  Actual checksum: {actual}\n"
            f"  Checkpoint path: {checkpoint_path}\n"
            f"  The checkpoint file may be corrupted."
        )
        super().__init__(message)


class LegacyCheckpointError(ModelVersioningError):
    """Raised when trying to load a legacy checkpoint without migration."""

    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        message = (
            f"Legacy checkpoint detected without versioning metadata.\n"
            f"  Checkpoint path: {checkpoint_path}\n"
            f"  Use migrate_legacy_checkpoint() to upgrade, or set "
            f"strict=False to load with warnings."
        )
        super().__init__(message)


class ConfigMismatchError(ModelVersioningError):
    """Raised when model config doesn't match checkpoint config."""

    def __init__(
        self,
        mismatched_keys: dict[str, tuple[Any, Any]],
        checkpoint_path: str,
    ):
        self.mismatched_keys = mismatched_keys
        self.checkpoint_path = checkpoint_path

        details = "\n".join(
            f"    {key}: checkpoint={cp_val}, current={cur_val}"
            for key, (cp_val, cur_val) in mismatched_keys.items()
        )
        message = (
            f"Model configuration mismatch!\n"
            f"  Checkpoint path: {checkpoint_path}\n"
            f"  Mismatched config keys:\n{details}"
        )
        super().__init__(message)


# =============================================================================
# Metadata
# =============================================================================


@dataclass
class ModelMetadata:
    """
    Metadata stored with each versioned checkpoint.

    Attributes:
        architecture_version: Semantic version string (e.g., "v2.1.0")
        model_class: Fully qualified class name (e.g., "RingRiftCNN")
        config: Full model configuration for reconstruction
        training_info: Training details (epochs, samples, metrics)
        created_at: ISO 8601 timestamp of checkpoint creation
        checksum: SHA256 hash of the serialized state_dict bytes
        parent_checkpoint: Path to parent checkpoint (for lineage tracking)
    """
    architecture_version: str
    model_class: str
    config: dict[str, Any]
    training_info: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    checksum: str = ""
    parent_checkpoint: str | None = None

    # Use a method to get default timestamp to avoid mutable default
    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelMetadata":
        """Create from dictionary, ignoring unknown fields for forward compatibility."""
        # Filter to only known fields to handle future metadata additions gracefully
        known_fields = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered_data)

    def is_compatible_with(self, other: "ModelMetadata") -> bool:
        """
        Check if this metadata is compatible with another.

        Compatibility requires:
        - Same model class
        - Same or compatible architecture version (major.minor must match)
        """
        if self.model_class != other.model_class:
            return False

        # Parse versions for comparison
        try:
            self_parts = self._parse_version(self.architecture_version)
            other_parts = self._parse_version(other.architecture_version)

            # Major and minor version must match
            return (
                self_parts[0] == other_parts[0] and
                self_parts[1] == other_parts[1]
            )
        except ValueError:
            # If version parsing fails, require exact match
            return self.architecture_version == other.architecture_version

    @staticmethod
    def _parse_version(version: str) -> tuple[int, int, int]:
        """Parse version string like 'v2.1.0' or '2.1.0' into tuple."""
        v = version.lstrip('v')
        parts = v.split('.')
        if len(parts) != 3:
            raise ValueError(f"Invalid version format: {version}")
        return tuple(int(p) for p in parts)  # type: ignore


# =============================================================================
# Architecture Version Constants
# =============================================================================

# These should be imported from neural_net.py after we add them there
# For now, define defaults here

# Architecture version constants
# These match the ARCHITECTURE_VERSION class attributes in neural_net.py
RINGRIFT_CNN_V2_VERSION = "v2.0.0"
RINGRIFT_CNN_V3_VERSION = "v3.1.0"
RINGRIFT_CNN_V4_VERSION = "v4.0.0"
HEX_NEURAL_NET_V2_VERSION = "v2.0.0"
HEX_NEURAL_NET_V3_VERSION = "v3.0.0"

# Model class name to version mapping
MODEL_VERSIONS: dict[str, str] = {
    # Square board models
    "RingRiftCNN_v2": RINGRIFT_CNN_V2_VERSION,
    "RingRiftCNN_v2_Lite": "v2.0.0-lite",
    "RingRiftCNN_v3": RINGRIFT_CNN_V3_VERSION,
    "RingRiftCNN_v3_Lite": "v3.1.0-lite",
    "RingRiftCNN_v4": RINGRIFT_CNN_V4_VERSION,
    # Hex board models
    "HexNeuralNet_v2": HEX_NEURAL_NET_V2_VERSION,
    "HexNeuralNet_v2_Lite": "v2.0.0-lite",
    "HexNeuralNet_v3": HEX_NEURAL_NET_V3_VERSION,
    "HexNeuralNet_v3_Lite": "v3.0.0-lite",
}


def get_model_version(model: nn.Module) -> str:
    """Get the architecture version for a model instance."""
    class_name = model.__class__.__name__

    # Check if model has explicit version attribute
    if hasattr(model, 'ARCHITECTURE_VERSION'):
        version = model.ARCHITECTURE_VERSION
        if isinstance(version, str):
            return version

    # Fall back to registered versions
    return MODEL_VERSIONS.get(class_name, "v0.0.0")


def get_model_config(model: nn.Module) -> dict[str, Any]:
    """
    Extract configuration from a model instance.

    This extracts the key architectural parameters that define the model.
    """
    config: dict[str, Any] = {}
    class_name = model.__class__.__name__

    if class_name in ("RingRiftCNN_v2", "RingRiftCNN_v2_Lite", "RingRiftCNN_v3", "RingRiftCNN_v3_Lite"):
        # V2 models use SE residual blocks
        num_filters = 128
        if hasattr(model, "conv1"):
            conv1 = model.conv1
            if hasattr(conv1, "out_channels"):
                num_filters = conv1.out_channels

        num_res_blocks = 10
        if hasattr(model, "res_blocks"):
            res_blocks = model.res_blocks
            if hasattr(res_blocks, "__len__"):
                num_res_blocks = len(res_blocks)

        config = {
            "board_size": getattr(model, "board_size", 8),
            "total_in_channels": getattr(model, "total_in_channels", 40),
            # These are canonical constants in current encoders but we persist them
            # to make checkpoint compatibility checks explicit.
            "global_features": getattr(model, "global_features", 20),
            "history_length": getattr(model, "history_length", 3),
            "num_filters": num_filters,
            "num_res_blocks": num_res_blocks,
            "policy_size": getattr(model, "policy_size", 55000),
            # Critical for safe loading: value head output depends on num_players.
            "num_players": getattr(model, "num_players", 4),
        }

        # v3 spatial heads: persist key shape-defining dimensions for compatibility checks.
        if class_name in ("RingRiftCNN_v3", "RingRiftCNN_v3_Lite"):
            config.update(
                {
                    "max_distance": getattr(model, "max_distance", None),
                    "num_ring_counts": getattr(model, "num_ring_counts", None),
                    "num_directions": getattr(model, "num_directions", None),
                    "num_line_dirs": getattr(model, "num_line_dirs", None),
                    "territory_size_buckets": getattr(model, "territory_size_buckets", None),
                    "territory_max_players": getattr(model, "territory_max_players", None),
                }
            )
    elif class_name in ("HexNeuralNet_v2", "HexNeuralNet_v2_Lite"):
        config = {
            "in_channels": getattr(model, "in_channels", 14),
            "global_features": getattr(model, "global_features", 20),
            "num_res_blocks": getattr(model, "num_res_blocks", 10),
            "num_filters": getattr(model, "num_filters", 128),
            "board_size": getattr(model, "board_size", 25),  # Radius-12 hex: 25×25 frame
            "policy_size": getattr(model, "policy_size", 91876),  # P_HEX for radius-12
        }
    else:
        # Generic extraction for unknown models
        attrs = ['board_size', 'in_channels', 'num_filters', 'policy_size']
        for attr in attrs:
            if hasattr(model, attr):
                config[attr] = getattr(model, attr)

    feature_version = getattr(model, "feature_version", None)
    if feature_version is None:
        feature_version = getattr(model, "_ringrift_feature_version", None)
    if feature_version is not None:
        try:
            config["feature_version"] = int(feature_version)
        except (TypeError, ValueError):
            config["feature_version"] = feature_version

    return config


# =============================================================================
# Checksum Utilities
# =============================================================================


def compute_state_dict_checksum(state_dict: dict[str, torch.Tensor]) -> str:
    """
    Compute SHA256 checksum of a model's state_dict.

    This creates a deterministic hash of the model weights for integrity
    verification.
    """
    hasher = hashlib.sha256()

    # Sort keys for deterministic ordering
    for key in sorted(state_dict.keys()):
        value = state_dict[key]
        # Hash key name
        hasher.update(key.encode('utf-8'))
        # Hash value data - handle both tensors and scalar values
        if hasattr(value, 'cpu') and hasattr(value, 'numpy'):
            # PyTorch tensor
            hasher.update(value.cpu().numpy().tobytes())
        elif hasattr(value, 'tobytes'):
            # NumPy array
            hasher.update(value.tobytes())
        else:
            # Scalar or other type - convert to string representation
            hasher.update(str(value).encode('utf-8'))

    return hasher.hexdigest()


# =============================================================================
# Model Version Manager
# =============================================================================


class ModelVersionManager:
    """
    Manages versioned model checkpoints with metadata and validation.

    This class provides:
    - Saving checkpoints with full metadata (version, config, etc.)
    - Loading checkpoints with strict version validation
    - Compatibility checking before loading
    - Migration of legacy checkpoints to versioned format
    - Checksum verification for integrity

    Example usage::

        manager = ModelVersionManager()

        # Save checkpoint
        metadata = manager.create_metadata(model, training_info={
            "epochs": 100,
            "final_loss": 0.05,
        })
        manager.save_checkpoint(model, metadata, "model_v1.pth")

        # Load checkpoint (strict mode - fails on version mismatch)
        loaded_state, loaded_metadata = manager.load_checkpoint(
            "model_v1.pth",
            strict=True,
        )
        model.load_state_dict(loaded_state)

        # Check compatibility before loading
        if manager.validate_compatibility(current_config, "model_v1.pth"):
            state, meta = manager.load_checkpoint("model_v1.pth")
            model.load_state_dict(state)
    """

    # Keys used in versioned checkpoint files
    METADATA_KEY = "_versioning_metadata"
    STATE_DICT_KEY = "model_state_dict"
    OPTIMIZER_KEY = "optimizer_state_dict"
    SCHEDULER_KEY = "scheduler_state_dict"
    EPOCH_KEY = "epoch"
    LOSS_KEY = "loss"

    def __init__(self, default_device: torch.device | None = None):
        """
        Initialize the version manager.

        Args:
            default_device: Default device for loading checkpoints.
                If None, uses CPU.
        """
        self.default_device = default_device or torch.device("cpu")

    def create_metadata(
        self,
        model: nn.Module,
        training_info: dict[str, Any] | None = None,
        parent_checkpoint: str | None = None,
    ) -> ModelMetadata:
        """
        Create metadata for a model instance.

        Args:
            model: The model to create metadata for.
            training_info: Optional training information (epochs, loss, etc.)
            parent_checkpoint: Optional path to parent checkpoint for lineage.

        Returns:
            ModelMetadata instance with all fields populated.
        """
        state_dict = model.state_dict()

        return ModelMetadata(
            architecture_version=get_model_version(model),
            model_class=model.__class__.__name__,
            config=get_model_config(model),
            training_info=training_info or {},
            created_at=datetime.now(timezone.utc).isoformat(),
            checksum=compute_state_dict_checksum(state_dict),
            parent_checkpoint=parent_checkpoint,
        )

    def save_checkpoint(
        self,
        model: nn.Module,
        metadata: ModelMetadata,
        path: str,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        epoch: int | None = None,
        loss: float | None = None,
    ) -> None:
        """
        Save a versioned checkpoint with metadata.

        Args:
            model: Model to save.
            metadata: Pre-created metadata (or use create_metadata()).
            path: File path to save checkpoint.
            optimizer: Optional optimizer to save state.
            scheduler: Optional scheduler to save state.
            epoch: Optional current epoch number.
            loss: Optional current loss value.
        """
        # Ensure directory exists
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        # Prepare checkpoint dict - use Any for mixed value types
        checkpoint: dict[str, Any] = {
            self.STATE_DICT_KEY: model.state_dict(),
            self.METADATA_KEY: metadata.to_dict(),
        }

        if optimizer is not None:
            checkpoint[self.OPTIMIZER_KEY] = optimizer.state_dict()

        if scheduler is not None:
            checkpoint[self.SCHEDULER_KEY] = scheduler.state_dict()

        if epoch is not None:
            checkpoint[self.EPOCH_KEY] = epoch

        if loss is not None:
            checkpoint[self.LOSS_KEY] = loss

        # Update checksum in metadata before saving
        metadata.checksum = compute_state_dict_checksum(
            checkpoint[self.STATE_DICT_KEY]
        )
        checkpoint[self.METADATA_KEY] = metadata.to_dict()

        # Use atomic save pattern to prevent corruption
        # 1. Save to temp file
        # 2. Validate the temp file loads correctly
        # 3. Atomically rename to final path
        path_obj = Path(path)
        temp_path = path_obj.with_suffix('.pth.tmp')

        try:
            torch.save(checkpoint, temp_path)

            # Flush to disk (important for NFS and network filesystems)
            os.sync()

            # Validate the saved file before finalizing
            try:
                test_load = torch.load(temp_path, map_location='cpu', weights_only=False)
                if test_load is None or self.STATE_DICT_KEY not in test_load:
                    raise ValueError("Saved checkpoint is invalid or missing state_dict")
            except Exception as e:
                temp_path.unlink(missing_ok=True)
                raise RuntimeError(f"Post-save validation failed: {e}")

            # Atomic rename to final path
            temp_path.rename(path_obj)

        except Exception as e:
            # Clean up temp file on failure
            temp_path.unlink(missing_ok=True)
            raise RuntimeError(f"Failed to save checkpoint: {e}")

        logger.info(
            f"Saved versioned checkpoint to {path}\n"
            f"  Version: {metadata.architecture_version}\n"
            f"  Model: {metadata.model_class}\n"
            f"  Checksum: {metadata.checksum[:16]}..."
        )

    def load_checkpoint(
        self,
        path: str,
        strict: bool = True,
        expected_version: str | None = None,
        expected_class: str | None = None,
        verify_checksum: bool = True,
        device: torch.device | None = None,
    ) -> tuple[dict[str, torch.Tensor], ModelMetadata]:
        """
        Load a versioned checkpoint with validation.

        Args:
            path: Path to checkpoint file.
            strict: If True, raise errors on version/config mismatch.
                If False, log warnings but continue loading.
            expected_version: Expected architecture version (optional).
            expected_class: Expected model class name (optional).
            verify_checksum: Whether to verify checksum integrity.
            device: Device to load tensors to.

        Returns:
            Tuple of (state_dict, metadata).

        Raises:
            VersionMismatchError: If strict=True and version doesn't match.
            ChecksumMismatchError: If verify_checksum=True and checksum fails.
            LegacyCheckpointError: If strict=True and checkpoint is legacy.
            FileNotFoundError: If checkpoint file doesn't exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        device = device or self.default_device
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        # Check if this is a versioned checkpoint
        if self.METADATA_KEY not in checkpoint:
            return self._handle_legacy_checkpoint(checkpoint, path, strict)

        # Load metadata
        metadata = ModelMetadata.from_dict(checkpoint[self.METADATA_KEY])
        state_dict = checkpoint[self.STATE_DICT_KEY]

        # Verify checksum if requested
        if verify_checksum:
            actual_checksum = compute_state_dict_checksum(state_dict)
            if metadata.checksum and actual_checksum != metadata.checksum:
                raise ChecksumMismatchError(
                    expected=metadata.checksum,
                    actual=actual_checksum,
                    checkpoint_path=path,
                )

        # Version validation
        if expected_version is not None and metadata.architecture_version != expected_version:
            if strict:
                raise VersionMismatchError(
                    checkpoint_version=metadata.architecture_version,
                    current_version=expected_version,
                    checkpoint_path=path,
                )
            else:
                logger.warning(
                    f"Version mismatch: checkpoint="
                    f"{metadata.architecture_version}, "
                    f"expected={expected_version}. "
                    f"Loading anyway (strict=False)."
                )

        # Class validation
        if expected_class is not None and metadata.model_class != expected_class:
            if strict:
                raise VersionMismatchError(
                    checkpoint_version=f"{metadata.model_class}",
                    current_version=f"{expected_class}",
                    checkpoint_path=path,
                    details="Model class mismatch",
                )
            else:
                logger.warning(
                    f"Class mismatch: checkpoint={metadata.model_class}, "
                    f"expected={expected_class}. "
                    f"Loading anyway (strict=False)."
                )

        logger.info(
            f"Loaded versioned checkpoint from {path}\n"
            f"  Version: {metadata.architecture_version}\n"
            f"  Model: {metadata.model_class}\n"
            f"  Created: {metadata.created_at}"
        )

        return state_dict, metadata

    def _handle_legacy_checkpoint(
        self,
        checkpoint: dict[str, Any],
        path: str,
        strict: bool,
    ) -> tuple[dict[str, torch.Tensor], ModelMetadata]:
        """Handle loading of legacy (non-versioned) checkpoints."""
        if strict:
            raise LegacyCheckpointError(path)

        logger.warning(
            f"Loading legacy checkpoint without versioning metadata: {path}\n"
            "Consider migrating with migrate_legacy_checkpoint()."
        )

        # Try to extract state dict from various legacy formats
        if self.STATE_DICT_KEY in checkpoint:
            state_dict = checkpoint[self.STATE_DICT_KEY]
        elif isinstance(checkpoint, dict) and all(
            isinstance(v, torch.Tensor) for v in checkpoint.values()
        ):
            # Direct state dict format
            state_dict = checkpoint
        else:
            # Try common keys (including early-stopping checkpoint formats)
            for key in ['state_dict', 'model', 'net', 'best_state']:
                if key in checkpoint:
                    state_dict = checkpoint[key]
                    break
            else:
                state_dict = checkpoint

        # Create placeholder metadata for legacy checkpoint
        metadata = ModelMetadata(
            architecture_version="v0.0.0-legacy",
            model_class="Unknown",
            config={},
            training_info={"legacy": True},
            created_at=datetime.now(timezone.utc).isoformat(),
            checksum=compute_state_dict_checksum(state_dict),
        )

        return state_dict, metadata

    def validate_compatibility(
        self,
        current_config: dict[str, Any],
        checkpoint_path: str,
        strict_config: bool = False,
    ) -> bool:
        """
        Check if a checkpoint is compatible with the current architecture.

        Args:
            current_config: Current model configuration dict.
            checkpoint_path: Path to checkpoint to validate.
            strict_config: If True, all config values must match exactly.
                If False, only critical keys (tensor shapes) must match.

        Returns:
            True if compatible, False otherwise.
        """
        if not os.path.exists(checkpoint_path):
            return False

        try:
            metadata = self.get_metadata(checkpoint_path)
        except (ModelVersioningError, Exception) as e:
            logger.warning(f"Could not read metadata: {e}")
            return False

        checkpoint_config = metadata.config

        # Critical keys that affect tensor shapes and feature compatibility
        critical_keys = {
            'board_size', 'total_in_channels', 'in_channels',
            'num_filters', 'num_res_blocks', 'policy_size',
            'global_features', 'feature_version',  # feature_version affects encoding
        }

        keys_to_check = (
            set(current_config.keys())
            if strict_config
            else critical_keys & set(current_config.keys())
        )

        mismatches = {}
        for key in keys_to_check:
            if key in checkpoint_config and current_config[key] != checkpoint_config[key]:
                mismatches[key] = (
                    checkpoint_config[key],
                    current_config[key],
                )

        if mismatches:
            logger.warning(
                f"Config mismatches for {checkpoint_path}:\n"
                + "\n".join(
                    f"  {k}: checkpoint={cv}, current={curv}"
                    for k, (cv, curv) in mismatches.items()
                )
            )
            return False

        return True

    def get_metadata(self, path: str) -> ModelMetadata:
        """
        Extract metadata from a checkpoint without loading the full model.

        Args:
            path: Path to checkpoint file.

        Returns:
            ModelMetadata instance.

        Raises:
            LegacyCheckpointError: If checkpoint doesn't have metadata.
            FileNotFoundError: If file doesn't exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        # Load with weights_only=False to access metadata
        # Use CPU to avoid GPU memory usage for just reading metadata
        checkpoint = torch.load(
            path,
            map_location=torch.device("cpu"),
            weights_only=False,
        )

        if self.METADATA_KEY not in checkpoint:
            raise LegacyCheckpointError(path)

        return ModelMetadata.from_dict(checkpoint[self.METADATA_KEY])

    def migrate_legacy_checkpoint(
        self,
        legacy_path: str,
        output_path: str,
        model_class: str,
        config: dict[str, Any],
        architecture_version: str | None = None,
        training_info: dict[str, Any] | None = None,
    ) -> ModelMetadata:
        """
        Migrate a legacy checkpoint to versioned format.

        Args:
            legacy_path: Path to legacy .pth file.
            output_path: Path for new versioned checkpoint.
            model_class: Name of the model class (e.g., "RingRiftCNN").
            config: Model configuration dict.
            architecture_version: Version string (uses MODEL_VERSIONS if None).
            training_info: Optional training info to include.

        Returns:
            Created ModelMetadata.
        """
        if not os.path.exists(legacy_path):
            raise FileNotFoundError(
                f"Legacy checkpoint not found: {legacy_path}"
            )

        # Load legacy checkpoint
        checkpoint = torch.load(
            legacy_path,
            map_location=torch.device("cpu"),
            weights_only=False,
        )

        # Extract state dict
        if self.STATE_DICT_KEY in checkpoint:
            state_dict = checkpoint[self.STATE_DICT_KEY]
        elif isinstance(checkpoint, dict) and all(
            isinstance(v, torch.Tensor) for v in checkpoint.values()
        ):
            state_dict = checkpoint
        else:
            for key in ['state_dict', 'model', 'net']:
                if key in checkpoint:
                    state_dict = checkpoint[key]
                    break
            else:
                state_dict = checkpoint

        # Determine version
        if architecture_version is None:
            architecture_version = MODEL_VERSIONS.get(model_class, "v1.0.0")

        # Create metadata
        metadata = ModelMetadata(
            architecture_version=architecture_version,
            model_class=model_class,
            config=config,
            training_info=training_info or {
                "migrated_from": legacy_path,
                "migration_date": datetime.now(timezone.utc).isoformat(),
            },
            created_at=datetime.now(timezone.utc).isoformat(),
            checksum=compute_state_dict_checksum(state_dict),
            parent_checkpoint=legacy_path,
        )

        # Create versioned checkpoint
        versioned_checkpoint = {
            self.STATE_DICT_KEY: state_dict,
            self.METADATA_KEY: metadata.to_dict(),
        }

        # Preserve other fields from original checkpoint
        preserve_keys = [
            self.OPTIMIZER_KEY,
            self.SCHEDULER_KEY,
            self.EPOCH_KEY,
            self.LOSS_KEY,
        ]
        for key in preserve_keys:
            if key in checkpoint:
                versioned_checkpoint[key] = checkpoint[key]

        # Save with atomic pattern
        dir_path = os.path.dirname(output_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        output_path_obj = Path(output_path)
        temp_path = output_path_obj.with_suffix('.pth.tmp')

        try:
            torch.save(versioned_checkpoint, temp_path)
            os.sync()

            # Validate before finalizing
            test_load = torch.load(temp_path, map_location='cpu', weights_only=False)
            if test_load is None:
                raise ValueError("Migrated checkpoint is invalid")

            temp_path.rename(output_path_obj)
        except Exception as e:
            temp_path.unlink(missing_ok=True)
            raise RuntimeError(f"Failed to migrate checkpoint: {e}")

        logger.info(
            f"Migrated legacy checkpoint to versioned format:\n"
            f"  From: {legacy_path}\n"
            f"  To: {output_path}\n"
            f"  Version: {metadata.architecture_version}\n"
            f"  Class: {metadata.model_class}"
        )

        return metadata


# =============================================================================
# Convenience Functions
# =============================================================================


def load_model_with_validation(
    model: nn.Module,
    checkpoint_path: str,
    strict: bool = True,
    device: torch.device | None = None,
) -> tuple[nn.Module, ModelMetadata]:
    """
    Load checkpoint into model with version validation.

    This is a convenience function that validates the checkpoint
    against the model's expected version and config, then loads
    the weights.

    Args:
        model: Model instance to load weights into.
        checkpoint_path: Path to versioned checkpoint.
        strict: Whether to fail on version/config mismatch.
        device: Device to load to.

    Returns:
        Tuple of (model with loaded weights, metadata).

    Raises:
        VersionMismatchError: If strict and version mismatch.
        ConfigMismatchError: If strict and critical config mismatch.
    """
    manager = ModelVersionManager(default_device=device)

    expected_version = get_model_version(model)
    expected_class = model.__class__.__name__
    current_config = get_model_config(model)

    # Validate compatibility first
    is_compatible = manager.validate_compatibility(
        current_config, checkpoint_path, strict_config=False
    )

    if not is_compatible and strict:
        # Get metadata for detailed error
        try:
            metadata = manager.get_metadata(checkpoint_path)
            mismatches = {}
            for key in current_config:
                if key in metadata.config and current_config[key] != metadata.config[key]:
                    mismatches[key] = (
                        metadata.config[key],
                        current_config[key],
                    )
            if mismatches:
                raise ConfigMismatchError(mismatches, checkpoint_path)
        except LegacyCheckpointError:
            pass  # Will be caught below

    # Load checkpoint
    state_dict, metadata = manager.load_checkpoint(
        checkpoint_path,
        strict=strict,
        expected_version=expected_version,
        expected_class=expected_class,
        verify_checksum=True,
        device=device,
    )

    # Load into model
    model.load_state_dict(state_dict)

    return model, metadata


def save_model_checkpoint(
    model: nn.Module,
    path: str,
    training_info: dict[str, Any] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    epoch: int | None = None,
    loss: float | None = None,
    parent_checkpoint: str | None = None,
) -> ModelMetadata:
    """
    Save model checkpoint with automatic metadata creation.

    This is a convenience function that creates metadata from the
    model and saves a versioned checkpoint.

    Args:
        model: Model to save.
        path: Path to save checkpoint.
        training_info: Optional training information dict.
        optimizer: Optional optimizer state to include.
        scheduler: Optional scheduler state to include.
        epoch: Optional epoch number.
        loss: Optional loss value.
        parent_checkpoint: Optional path to parent checkpoint.

    Returns:
        Created ModelMetadata.
    """
    manager = ModelVersionManager()

    metadata = manager.create_metadata(
        model,
        training_info=training_info,
        parent_checkpoint=parent_checkpoint,
    )

    manager.save_checkpoint(
        model,
        metadata,
        path,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        loss=loss,
    )

    return metadata
