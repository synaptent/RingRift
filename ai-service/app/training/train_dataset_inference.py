"""Dataset metadata inference and validation for RingRift training pipeline.

Extracted from train.py (lines 1671-2326) to reduce train_model() complexity.
Consolidates heavily duplicated code between square, hex, and streaming paths
for reading NPZ metadata (history_length, feature_version, policy_encoding,
globals_dim validation) into shared helper functions.
"""
from __future__ import annotations

import dataclasses
import logging
import os
from typing import Any

import numpy as np

from app.utils.numpy_utils import safe_load_npz

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class DatasetInferenceResult:
    """Result of dataset metadata inference and validation."""
    board_size: int
    policy_size: int
    hex_in_channels: int
    hex_num_players: int
    use_hex_model: bool
    use_hex_v3: bool
    use_hex_v4: bool
    use_hex_v5: bool
    use_hex_v5_large: bool
    detected_num_heuristics: int | None
    config_feature_version: int
    hex_radius: int


@dataclasses.dataclass
class _NpzMetadata:
    """Metadata read from an NPZ file."""
    in_channels: int | None = None
    globals_dim: int | None = None
    policy_encoding: str | None = None
    history_length: int | None = None
    feature_version: int | None = None
    inferred_policy_size: int | None = None


def _read_npz_metadata(data_path_str: str, *, distributed: bool, is_main: bool) -> _NpzMetadata:
    """Read metadata fields from an NPZ file.

    This is the consolidated metadata reader that replaces the duplicated
    code between square and hex paths in train_model().
    """
    meta = _NpzMetadata()
    if not data_path_str or not os.path.exists(data_path_str):
        return meta

    try:
        with safe_load_npz(data_path_str, mmap_mode="r") as d:
            if "features" in d:
                feat_shape = d["features"].shape
                if len(feat_shape) >= 2:
                    meta.in_channels = int(feat_shape[1])

            if "globals" in d:
                glob_shape = d["globals"].shape
                if len(glob_shape) >= 2:
                    meta.globals_dim = int(glob_shape[1])

            if "policy_encoding" in d:
                try:
                    meta.policy_encoding = str(np.asarray(d["policy_encoding"]).item())
                except (ValueError, TypeError, AttributeError):
                    # Metadata field missing, wrong type, or empty array
                    pass

            if "history_length" in d:
                try:
                    meta.history_length = int(np.asarray(d["history_length"]).item())
                except (ValueError, TypeError, AttributeError):
                    # Metadata field missing, wrong type, or empty array
                    pass

            if "feature_version" in d:
                try:
                    meta.feature_version = int(np.asarray(d["feature_version"]).item())
                except (ValueError, TypeError, AttributeError):
                    # Metadata field missing, wrong type, or empty array
                    pass

            if "policy_indices" in d:
                pi = d["policy_indices"]
                max_idx = -1
                for i in range(len(pi)):
                    arr = np.asarray(pi[i])
                    if arr.size == 0:
                        continue
                    local_max = int(np.asarray(arr).max())
                    if local_max > max_idx:
                        max_idx = local_max
                if max_idx >= 0:
                    meta.inferred_policy_size = max_idx + 1
    except (OSError, KeyError, ValueError, AttributeError) as exc:
        # OSError: file I/O errors reading NPZ
        # KeyError: missing keys in dataset
        # ValueError: invalid array conversion
        # AttributeError: array method errors
        if not distributed or is_main:
            logger.warning(
                "Failed to read metadata from %s: %s",
                data_path_str,
                exc,
            )

    return meta


def _validate_history_length(
    *,
    dataset_history_length: int | None,
    config_history_length: int,
    data_path_str: str,
    distributed: bool,
    is_main: bool,
) -> None:
    """Validate that dataset history_length matches config."""
    if dataset_history_length is not None and dataset_history_length != config_history_length:
        raise ValueError(
            "Training history_length does not match dataset metadata.\n"
            f"  dataset={data_path_str}\n"
            f"  dataset_history_length={dataset_history_length}\n"
            f"  config.history_length={config_history_length}\n"
            "Regenerate the dataset with matching --history-length or "
            "update the training config."
        )
    elif (dataset_history_length is None and config_history_length != 3
          and (not distributed or is_main)):
        logger.warning(
            "Dataset %s missing history_length metadata; using "
            "config.history_length=%d. Ensure the dataset was built "
            "with matching history frames.",
            data_path_str,
            config_history_length,
        )


def _validate_feature_version(
    *,
    dataset_feature_version: int | None,
    config_feature_version: int,
    data_path_str: str,
    distributed: bool,
    is_main: bool,
) -> int:
    """Validate that dataset feature_version matches config.

    Returns the (possibly adjusted) config_feature_version.
    """
    if dataset_feature_version is not None and dataset_feature_version != config_feature_version:
        raise ValueError(
            "Training feature_version does not match dataset metadata.\n"
            f"  dataset={data_path_str}\n"
            f"  dataset_feature_version={dataset_feature_version}\n"
            f"  config_feature_version={config_feature_version}\n"
            "Regenerate the dataset with matching --feature-version or "
            "update the training config."
        )
    elif dataset_feature_version is None:
        if config_feature_version != 1:
            # Check for autonomous mode - fallback to v1 instead of failing
            autonomous_mode = os.environ.get("RINGRIFT_AUTONOMOUS_MODE", "").lower() in ("1", "true")
            if autonomous_mode:
                if not distributed or is_main:
                    logger.warning(
                        "[AUTONOMOUS] Dataset %s missing feature_version metadata. "
                        "Config requested v%d but falling back to v1 for compatibility.",
                        data_path_str,
                        config_feature_version,
                    )
                config_feature_version = 1
            else:
                raise ValueError(
                    "Dataset is missing feature_version metadata but training "
                    "was configured for feature_version="
                    f"{config_feature_version}.\n"
                    f"  dataset={data_path_str}\n"
                    "Regenerate the dataset with --feature-version or "
                    "set feature_version=1 to use legacy features."
                )
        if not distributed or is_main:
            logger.warning(
                "Dataset %s missing feature_version metadata; assuming legacy "
                "feature_version=1.",
                data_path_str,
            )
    return config_feature_version


def _validate_globals_dim(
    *,
    dataset_globals_dim: int | None,
    data_path_str: str,
    distributed: bool,
    is_main: bool,
    is_hex: bool = False,
) -> int | None:
    """Validate globals feature dimension.

    For hex boards, supports autonomous mode fallback to zeros.
    Returns the (possibly adjusted) dataset_globals_dim.
    """
    if dataset_globals_dim is None:
        if is_hex:
            # Check for autonomous mode - warn but continue with zero-filled globals
            autonomous_mode = os.environ.get("RINGRIFT_AUTONOMOUS_MODE", "").lower() in ("1", "true")
            if autonomous_mode:
                if not distributed or is_main:
                    logger.warning(
                        "[AUTONOMOUS] Dataset %s missing globals features. "
                        "Training will use zeros for globals (degraded quality). "
                        "Recommend regenerating dataset for best results.",
                        data_path_str,
                    )
                # Set a marker to inject zeros later - but for now, set to expected dim
                return 20  # Expected dimension, will be zero-filled
            else:
                raise ValueError(
                    "Dataset is missing globals features required for training.\n"
                    f"  dataset={data_path_str}\n"
                    "Regenerate the dataset with scripts/export_replay_dataset.py."
                )
        else:
            raise ValueError(
                "Dataset is missing globals features required for training.\n"
                f"  dataset={data_path_str}\n"
                "Regenerate the dataset with scripts/export_replay_dataset.py."
            )
    if dataset_globals_dim != 20:
        raise ValueError(
            "Dataset globals feature dimension does not match the CNN encoder.\n"
            f"  dataset={data_path_str}\n"
            f"  dataset_globals_dim={dataset_globals_dim}\n"
            "Regenerate the dataset with scripts/export_replay_dataset.py "
            "to produce 20 global features."
        )
    return dataset_globals_dim


def _validate_policy_encoding(
    *,
    policy_encoding: str | None,
    data_path_str: str,
    model_version: str,
    distributed: bool,
    is_main: bool,
    board_type_hint: str = "square8|square19",
) -> None:
    """Validate policy encoding - reject legacy_max_n for ALL model versions."""
    # CRITICAL: Reject legacy_max_n encoding for ALL model versions
    # Legacy encoding creates policy indices up to ~59K for square8, but models
    # should use board-aware encoding (7K for square8). Using legacy encoding
    # causes severe performance degradation as the model learns wrong action IDs.
    if policy_encoding == "legacy_max_n":
        raise ValueError(
            f"Dataset uses DEPRECATED legacy_max_n policy encoding.\n"
            f"  dataset={data_path_str}\n"
            f"  model_version={model_version}\n\n"
            "Legacy MAX_N encoding produces ~59K actions for square8 boards,\n"
            "but board-aware encoding produces only 7K actions.\n"
            "This mismatch causes models to output garbage predictions.\n\n"
            "Regenerate the dataset with board-aware encoding (now the default):\n"
            f"  PYTHONPATH=. python scripts/export_replay_dataset.py \\\n"
            f"    --db <canonical_db> --board-type <{board_type_hint}> \\\n"
            f"    --num-players <2|3|4> --output <path>.npz\n"
        )
    if policy_encoding is None and (not distributed or is_main):
        logger.warning(
            "Dataset %s missing policy_encoding metadata. Assuming board-aware "
            "encoding. If this dataset was exported with legacy MAX_N, "
            "regenerate with current export script (board-aware is now default).",
            data_path_str,
        )


def _validate_and_resolve_policy_size(
    *,
    inferred_size: int | None,
    config: Any,
    model_version: str,
    data_path_str: str,
    distributed: bool,
    is_main: bool,
    get_policy_size_for_board: Any,
    board_type_hint: str = "square8|square19",
    label: str = "",
) -> int:
    """Validate inferred policy size and return final policy_size.

    Validates that policy indices don't exceed the board-aware action space.
    """
    if inferred_size is not None:
        board_default_size = get_policy_size_for_board(config.board_type)
        # CRITICAL: For ALL model versions, validate that policy indices
        # don't exceed the board-aware action space. If they do, the data
        # was exported with legacy MAX_N encoding which is now deprecated.
        if inferred_size > board_default_size:
            raise ValueError(
                f"Dataset policy indices exceed the board-aware policy space.\n"
                f"  dataset={data_path_str}\n"
                f"  model_version={model_version}\n"
                f"  inferred_policy_size={inferred_size}\n"
                f"  board_default_policy_size={board_default_size}\n\n"
                "This indicates the dataset was exported with legacy MAX_N encoding,\n"
                f"which produces ~59K actions for square8 instead of 7K.\n\n"
                "Regenerate the dataset with the current export script:\n"
                f"  PYTHONPATH=. python scripts/export_replay_dataset.py \\\n"
                f"    --db <canonical_db> --board-type <{board_type_hint}> \\\n"
                f"    --num-players <2|3|4> --output <path>.npz\n"
            )
        # Use board-default size for consistent policy head dimensions
        policy_size = board_default_size
        if not distributed or is_main:
            logger.info(
                "Using board-default policy_size=%d for %s (dataset max index: %d)",
                policy_size,
                config.board_type.name,
                inferred_size,
            )
    else:
        policy_size = get_policy_size_for_board(config.board_type)
        log_suffix = f" ({label})" if label else ""
        if not distributed or is_main:
            logger.info(
                "Using board-default %spolicy_size=%d for board_type=%s%s",
                "hex " if label == "hex" else "",
                policy_size,
                config.board_type.name,
                log_suffix,
            )
    return policy_size


def _detect_heuristics_from_npz(
    *,
    data_path: str | list[str],
    model_version: str,
    distributed: bool,
    is_main: bool,
) -> int | None:
    """Detect heuristic feature count from NPZ for v5_heavy models.

    This enables automatic switching between fast (21) and full (49) modes.
    """
    v5_versions = ('v5', 'v5-gnn', 'v5-heavy', 'v5-heavy-large', 'v5-heavy-xl', 'v6', 'v6-xl')
    if model_version not in v5_versions:
        return None

    if isinstance(data_path, list):
        heuristic_check_path = data_path[0] if data_path else ""
    else:
        heuristic_check_path = data_path

    if not heuristic_check_path or not os.path.exists(heuristic_check_path):
        return None

    detected_num_heuristics: int | None = None
    try:
        with safe_load_npz(heuristic_check_path, mmap_mode="r") as d:
            if "heuristics" in d:
                heuristic_shape = d["heuristics"].shape
                if len(heuristic_shape) >= 2:
                    detected_num_heuristics = int(heuristic_shape[1])
                    if not distributed or is_main:
                        heuristic_mode = "full (49)" if detected_num_heuristics >= 49 else "fast (21)"
                        logger.info(
                            "Detected %d heuristic features (%s mode) from %s",
                            detected_num_heuristics,
                            heuristic_mode,
                            heuristic_check_path,
                        )
    except (OSError, KeyError, ValueError, AttributeError) as exc:
        # OSError: file I/O errors reading NPZ
        # KeyError: missing 'heuristics' in dataset
        # ValueError: invalid heuristic data
        # AttributeError: array method errors
        if not distributed or is_main:
            logger.warning(
                "Failed to detect heuristic count from %s: %s",
                heuristic_check_path,
                exc,
            )

    return detected_num_heuristics


def _infer_hex_in_channels(
    *,
    data_path: str | list[str],
    config: Any,
    model_version: str,
    use_hex_v3: bool,
    use_hex_v4: bool,
    distributed: bool,
    is_main: bool,
) -> int:
    """Infer hex input channels from dataset and validate compatibility."""
    if isinstance(data_path, list):
        data_path_str = data_path[0] if data_path else ""
    else:
        data_path_str = data_path

    inferred_in_channels = None
    if data_path_str and os.path.exists(data_path_str):
        try:
            with safe_load_npz(data_path_str, mmap_mode="r") as d:
                if "features" in d:
                    feat_shape = d["features"].shape
                    if len(feat_shape) >= 2:
                        inferred_in_channels = feat_shape[1]  # (N, C, H, W)
        except (OSError, KeyError, ValueError, AttributeError) as exc:
            # OSError: file I/O errors reading NPZ
            # KeyError: missing 'features' in dataset
            # ValueError: invalid array shape
            # AttributeError: shape attribute errors
            if not distributed or is_main:
                logger.warning(
                    "Failed to infer hex in_channels from %s: %s",
                    data_path_str,
                    exc,
                )

    hex_base_channels = 16 if (use_hex_v3 or use_hex_v4) else 10
    expected_in_channels = hex_base_channels * (config.history_length + 1)

    if inferred_in_channels is not None:
        if inferred_in_channels != expected_in_channels:
            # Determine which model version the data was encoded for
            if inferred_in_channels == 40:
                data_encoding = "V2 (10 base channels \u00d7 4 frames = 40)"
                compatible_versions = "v2"
            elif inferred_in_channels == 64:
                data_encoding = "V3/V4 (16 base channels \u00d7 4 frames = 64)"
                compatible_versions = "v3, v3-flat, v3-spatial, or v4"
            else:
                data_encoding = f"unknown ({inferred_in_channels} channels)"
                compatible_versions = "unknown"

            raise ValueError(
                f"\n{'='*70}\n"
                f"EARLY VALIDATION FAILURE: Dataset/model version mismatch\n"
                f"{'='*70}\n\n"
                f"Dataset: {data_path_str}\n"
                f"  - Has {inferred_in_channels} input channels ({data_encoding})\n"
                f"  - Compatible with: --model-version {compatible_versions}\n\n"
                f"Requested: --model-version {model_version}\n"
                f"  - Expects {expected_in_channels} input channels\n\n"
                f"SOLUTIONS:\n"
                f"  1. Use --model-version {compatible_versions} (recommended for existing data)\n"
                f"  2. Regenerate dataset with V3/V4 encoder for {model_version} models\n"
                f"{'='*70}"
            )
        hex_in_channels = inferred_in_channels
        if not distributed or is_main:
            logger.info(
                "Using inferred hex in_channels=%d from dataset %s",
                hex_in_channels,
                data_path_str,
            )
    else:
        # Fallback to computed value
        hex_in_channels = expected_in_channels

    return hex_in_channels


def infer_dataset_metadata(
    *,
    data_path: str | list[str],
    config: Any,  # TrainConfig
    num_players: int,
    model_version: str = "v2",
    multi_player: bool = False,
    use_streaming: bool = False,
    distributed: bool = False,
    is_main: bool = True,
    resume_path: str | None = None,
    num_filters: int | None = None,
    num_res_blocks: int | None = None,
    device: Any = None,
    # Import references
    BoardType: Any = None,
    HEX_BOARD_SIZE: int = 25,
    HEX8_BOARD_SIZE: int = 9,
    MAX_PLAYERS: int = 4,
    get_policy_size_for_board: Any = None,
    normalize_board_type: Any = None,
    validate_hex_policy_indices: Any = None,
    detect_tier_from_checkpoint: Any = None,
) -> DatasetInferenceResult:
    """Infer dataset metadata and validate compatibility with training config.

    This replaces the large if/elif blocks in train_model() that read NPZ
    metadata for square, hex, and streaming paths. The duplicated metadata
    reading code has been consolidated into shared helper functions.

    Args:
        data_path: Path(s) to training data.
        config: Training configuration with board_type, history_length, etc.
        num_players: Number of players.
        model_version: Model version string (v2, v3, v4, v5-heavy, etc.).
        multi_player: Whether multi-player mode is enabled.
        use_streaming: Whether streaming data loading is being used.
        distributed: Whether distributed training is enabled.
        is_main: Whether this is the main process.
        resume_path: Path to checkpoint being resumed from.
        num_filters: Number of CNN filters (None = use default).
        num_res_blocks: Number of residual blocks (None = use default).
        device: Torch device for checkpoint loading.
        BoardType: BoardType enum.
        HEX_BOARD_SIZE: Board size constant for hexagonal boards.
        HEX8_BOARD_SIZE: Board size constant for hex8 boards.
        MAX_PLAYERS: Maximum player count constant.
        get_policy_size_for_board: Function to get policy size.
        normalize_board_type: Function to normalize board type names.
        validate_hex_policy_indices: Function to validate hex policy indices.
        detect_tier_from_checkpoint: Function to detect model tier from checkpoint.

    Returns:
        DatasetInferenceResult with all inferred metadata.

    Raises:
        ValueError: If dataset is incompatible with training configuration.
    """
    # Determine canonical spatial board_size for the CNN from config.
    if config.board_type == BoardType.SQUARE19:
        board_size = 19
    elif config.board_type == BoardType.HEXAGONAL:
        board_size = HEX_BOARD_SIZE  # 25
    elif config.board_type == BoardType.HEX8:
        board_size = HEX8_BOARD_SIZE  # 9
    else:
        board_size = 8

    # Determine whether to use HexNeuralNet for hexagonal boards (including hex8)
    use_hex_model = config.board_type in (BoardType.HEXAGONAL, BoardType.HEX8)
    config_feature_version = int(getattr(config, "feature_version", 1) or 1)

    # Validate model_id matches board_type to prevent architecture mismatch errors (P0)
    if use_hex_model and "sq8" in config.model_id.lower():
        if resume_path:
            raise ValueError(
                f"Model ID '{config.model_id}' contains 'sq8' but board_type is "
                f"{config.board_type.name} which uses HexNeuralNet architecture. "
                "Use a model ID that reflects the hex board type (e.g., 'ringrift_hex_2p')."
            )
        else:
            board_prefix = "hex8" if config.board_type == BoardType.HEX8 else "hex"
            new_model_id = f"ringrift_{board_prefix}_{num_players}p"
            logger.warning(
                f"Model ID '{config.model_id}' is for sq8 but training {config.board_type.name}. "
                f"Using '{new_model_id}' instead."
            )
            config.model_id = new_model_id
    if not use_hex_model and ("hex" in config.model_id.lower() and "sq" not in config.model_id.lower()):
        if resume_path:
            raise ValueError(
                f"Model ID '{config.model_id}' appears to be for hex but board_type is "
                f"{config.board_type.name}. Use a model ID that matches the board type."
            )
        else:
            board_prefix = normalize_board_type(config.board_type)
            new_model_id = f"ringrift_{board_prefix}_{num_players}p"
            logger.warning(
                f"Model ID '{config.model_id}' is for hex but training {config.board_type.name}. "
                f"Using '{new_model_id}' instead."
            )
            config.model_id = new_model_id

    # Resolve first data path for metadata reading
    if isinstance(data_path, list):
        data_path_str = data_path[0] if data_path else ""
    else:
        data_path_str = data_path

    # Determine effective policy head size.
    policy_size: int
    if not use_hex_model and not use_streaming:
        # Non-hex, non-streaming: infer from the NPZ file if possible.
        meta = _read_npz_metadata(data_path_str, distributed=distributed, is_main=is_main)

        _validate_history_length(
            dataset_history_length=meta.history_length,
            config_history_length=config.history_length,
            data_path_str=data_path_str,
            distributed=distributed,
            is_main=is_main,
        )
        config_feature_version = _validate_feature_version(
            dataset_feature_version=meta.feature_version,
            config_feature_version=config_feature_version,
            data_path_str=data_path_str,
            distributed=distributed,
            is_main=is_main,
        )

        # Validate square-board in_channels
        expected_in_channels = 14 * (config.history_length + 1)
        if meta.in_channels is not None and meta.in_channels != expected_in_channels:
            raise ValueError(
                "Dataset feature channels do not match the square-board encoder.\n"
                f"  dataset={data_path_str}\n"
                f"  dataset_in_channels={meta.in_channels}\n"
                f"  expected_in_channels={expected_in_channels}\n"
                "Regenerate the dataset with scripts/export_replay_dataset.py "
                "or app.training.generate_data using the default CNN encoder."
            )

        _validate_globals_dim(
            dataset_globals_dim=meta.globals_dim,
            data_path_str=data_path_str,
            distributed=distributed,
            is_main=is_main,
            is_hex=False,
        )

        _validate_policy_encoding(
            policy_encoding=meta.policy_encoding,
            data_path_str=data_path_str,
            model_version=model_version,
            distributed=distributed,
            is_main=is_main,
            board_type_hint="square8|square19",
        )

        policy_size = _validate_and_resolve_policy_size(
            inferred_size=meta.inferred_policy_size,
            config=config,
            model_version=model_version,
            data_path_str=data_path_str,
            distributed=distributed,
            is_main=is_main,
            get_policy_size_for_board=get_policy_size_for_board,
            board_type_hint="square8|square19",
        )
    else:
        # Hex or streaming: try to infer from data, fall back to board defaults.
        inferred_hex_size: int | None = None
        if use_hex_model and not use_streaming:
            meta = _read_npz_metadata(data_path_str, distributed=distributed, is_main=is_main)

            _validate_history_length(
                dataset_history_length=meta.history_length,
                config_history_length=config.history_length,
                data_path_str=data_path_str,
                distributed=distributed,
                is_main=is_main,
            )
            config_feature_version = _validate_feature_version(
                dataset_feature_version=meta.feature_version,
                config_feature_version=config_feature_version,
                data_path_str=data_path_str,
                distributed=distributed,
                is_main=is_main,
            )
            _validate_globals_dim(
                dataset_globals_dim=meta.globals_dim,
                data_path_str=data_path_str,
                distributed=distributed,
                is_main=is_main,
                is_hex=True,
            )
            _validate_policy_encoding(
                policy_encoding=meta.policy_encoding,
                data_path_str=data_path_str,
                model_version=model_version,
                distributed=distributed,
                is_main=is_main,
                board_type_hint="hex8|hexagonal",
            )

            inferred_hex_size = meta.inferred_policy_size

        if inferred_hex_size is not None:
            policy_size = _validate_and_resolve_policy_size(
                inferred_size=inferred_hex_size,
                config=config,
                model_version=model_version,
                data_path_str=data_path_str,
                distributed=distributed,
                is_main=is_main,
                get_policy_size_for_board=get_policy_size_for_board,
                board_type_hint="hex8|hexagonal",
            )

            # Dec 2025: Validate policy indices for V3/V4 hex models
            # This catches encoding mismatches that would cause -1e9 masked logits
            if model_version in ('v3', 'v4') and data_path_str:
                try:
                    with safe_load_npz(data_path_str, mmap_mode="r") as d:
                        if "policy_indices" in d:
                            hex_board_size = 9 if config.board_type == BoardType.HEX8 else 25
                            hex_r = 4 if config.board_type == BoardType.HEX8 else 12
                            is_valid, errors = validate_hex_policy_indices(
                                d["policy_indices"],
                                board_size=hex_board_size,
                                hex_radius=hex_r,
                                sample_size=2000,
                            )
                            if not is_valid:
                                raise ValueError(
                                    f"Hex policy index validation failed for {model_version} model.\n"
                                    f"  dataset={data_path_str}\n\n"
                                    f"Errors:\n  " + "\n  ".join(errors) + "\n\n"
                                    "This indicates the dataset was exported with an older encoder "
                                    "that didn't validate hex cell positions.\n"
                                    "Regenerate the dataset with the current export script (Dec 2025+):\n"
                                    "  PYTHONPATH=. python scripts/export_replay_dataset.py \\\n"
                                    "    --db <canonical_db> --board-type <hex8|hexagonal> \\\n"
                                    "    --num-players <2|3|4> --board-aware-encoding --output <path>.npz\n"
                                )
                            elif not distributed or is_main:
                                logger.info(
                                    "Hex policy index validation passed for %s model",
                                    model_version.upper(),
                                )
                except (OSError, KeyError, ValueError, RuntimeError) as e:
                    # OSError: file I/O errors reading dataset
                    # KeyError: missing keys in validation data
                    # ValueError: validation check failures
                    # RuntimeError: validation logic errors
                    if "validation failed" in str(e):
                        raise
                    logger.warning(f"Could not validate hex policy indices: {e}")

        elif use_hex_model:
            # Use board-specific policy size (4500 for HEX8, 91876 for HEXAGONAL)
            policy_size = get_policy_size_for_board(config.board_type)
            if not distributed or is_main:
                logger.info(
                    "Using board-default hex policy_size=%d for board_type=%s",
                    policy_size,
                    config.board_type.name,
                )
        else:
            policy_size = get_policy_size_for_board(config.board_type)
            if not distributed or is_main:
                logger.info(
                    "Using board-default policy_size=%d for board_type=%s "
                    "(streaming path)",
                    policy_size,
                    config.board_type.name,
                )

    # December 2025: Auto-detect tier from checkpoint when resuming
    if resume_path is not None:
        detected = detect_tier_from_checkpoint(resume_path, device=device)
        if detected:
            ckpt_tier, ckpt_version, ckpt_filters, ckpt_blocks = detected
            current_filters = num_filters if num_filters is not None else 96
            current_blocks = num_res_blocks if num_res_blocks is not None else 6
            if (ckpt_version != model_version or
                    ckpt_filters != current_filters or
                    ckpt_blocks != current_blocks):
                if not distributed or is_main:
                    logger.warning(
                        f"Checkpoint architecture differs from requested settings. "
                        f"Requested: version={model_version}, filters={current_filters}, blocks={current_blocks}. "
                        f"Checkpoint: tier={ckpt_tier}, version={ckpt_version}, filters={ckpt_filters}, blocks={ckpt_blocks}. "
                        f"Using checkpoint architecture to ensure compatibility."
                    )
                model_version = ckpt_version
                num_filters = ckpt_filters
                num_res_blocks = ckpt_blocks

    hex_in_channels = 0
    hex_num_players = num_players
    hex_radius = 4 if config.board_type == BoardType.HEX8 else 12
    use_hex_v5 = bool(use_hex_model and model_version in ('v5', 'v5-gnn', 'v5-heavy'))
    use_hex_v5_large = bool(use_hex_model and model_version in ('v5-heavy-large', 'v5-heavy-xl', 'v6', 'v6-xl'))
    use_hex_v4 = bool(use_hex_model and model_version == 'v4')
    use_hex_v3 = bool(use_hex_model and model_version in ('v3', 'v3-flat', 'v3-spatial'))

    # Detect heuristic features
    detected_num_heuristics = _detect_heuristics_from_npz(
        data_path=data_path,
        model_version=model_version,
        distributed=distributed,
        is_main=is_main,
    )

    # Infer hex in_channels if hex model
    if use_hex_model:
        hex_in_channels = _infer_hex_in_channels(
            data_path=data_path,
            config=config,
            model_version=model_version,
            use_hex_v3=use_hex_v3,
            use_hex_v4=use_hex_v4,
            distributed=distributed,
            is_main=is_main,
        )
        hex_num_players = MAX_PLAYERS if multi_player else num_players

    return DatasetInferenceResult(
        board_size=board_size,
        policy_size=policy_size,
        hex_in_channels=hex_in_channels,
        hex_num_players=hex_num_players,
        use_hex_model=use_hex_model,
        use_hex_v3=use_hex_v3,
        use_hex_v4=use_hex_v4,
        use_hex_v5=use_hex_v5,
        use_hex_v5_large=use_hex_v5_large,
        detected_num_heuristics=detected_num_heuristics,
        config_feature_version=config_feature_version,
        hex_radius=hex_radius,
    )
