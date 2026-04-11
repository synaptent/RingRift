"""Training dataset metadata and loader preparation helpers.

This module extracts the large dataset metadata validation and dataloader
construction blocks from ``app.training.train`` without changing the supported
training entrypoints.
"""

from __future__ import annotations

import glob
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split

from app.training.data_loader import (
    StreamingDataLoader,
    WeightedStreamingDataLoader,
    get_sample_count,
)
from app.training.datasets import RingRiftDataset, WeightedRingRiftDataset
from app.training.distributed import (
    cleanup_distributed,
    get_distributed_sampler,
    get_rank,
    get_world_size,
)
from app.training.train_dataset_inference import infer_dataset_metadata
from app.utils.numpy_utils import safe_load_npz

logger = logging.getLogger(__name__)


@dataclass
class DatasetMetadataContext:
    """Resolved metadata needed to construct the training model and loaders."""

    board_size: int
    policy_size: int
    encoding_channels: int
    hex_num_players: int
    use_hex_model: bool
    use_hex_v3: bool
    use_hex_v4: bool
    use_hex_v5: bool
    use_hex_v5_large: bool
    detected_num_heuristics: int | None
    config_feature_version: int
    hex_radius: int


@dataclass
class TrainingDataPipelineContext:
    """Prepared datasets and loaders for a training run."""

    use_streaming: bool
    train_streaming_loader: StreamingDataLoader | WeightedStreamingDataLoader | None
    val_streaming_loader: StreamingDataLoader | None
    train_loader: DataLoader | None
    val_loader: DataLoader | None
    train_sampler: Any
    val_sampler: Any
    full_dataset: Any | None
    train_size: int
    val_size: int
    total_samples: int
    num_data_files: int
    value_only_training: bool


def validate_training_data(npz_path: Path, board_type: str, num_players: int) -> None:
    """Validate NPZ file before training. Raises ValueError on issues."""
    if not npz_path.exists():
        raise FileNotFoundError(f"Training data not found: {npz_path}")

    file_size = npz_path.stat().st_size
    if file_size < 1024:
        raise ValueError(
            f"Training data file too small ({file_size} bytes): {npz_path}. "
            "Likely corrupt or incomplete transfer."
        )

    data = np.load(npz_path, allow_pickle=True)
    required_keys = ["features", "values"]
    for key in required_keys:
        if key not in data:
            raise ValueError(f"NPZ missing required key: '{key}' in {npz_path}")

    sample_count = data["features"].shape[0]
    if sample_count < 100:
        raise ValueError(
            f"Too few samples in {npz_path}: {sample_count} (minimum 100). "
            "Collect more selfplay data before training."
        )

    if "board_type" in data:
        raw_bt = data["board_type"]
        meta_bt = str(raw_bt.item() if hasattr(raw_bt, "item") else raw_bt).lower().strip()
        expected_bt = board_type.lower().strip()
        if meta_bt != expected_bt:
            raise ValueError(
                f"Board type mismatch in {npz_path}: "
                f"file has '{meta_bt}', expected '{expected_bt}'"
            )

    logger.info(
        "Training data validated: %s (%s samples, board_type=%s, num_players=%s)",
        npz_path,
        sample_count,
        board_type,
        num_players,
    )


def prepare_dataset_metadata_context(
    *,
    data_path: str | list[str],
    config: Any,
    num_players: int,
    model_version: str,
    multi_player: bool,
    use_streaming: bool,
    distributed: bool,
    is_main: bool,
    resume_path: str | None,
    num_filters: int | None,
    num_res_blocks: int | None,
    device: torch.device,
    data_path_str: str,
    BoardType: Any,
    HEX_BOARD_SIZE: int,
    HEX8_BOARD_SIZE: int,
    MAX_PLAYERS: int,
    get_policy_size_for_board: Any,
    normalize_board_type: Any,
    validate_hex_policy_indices: Any,
    detect_tier_from_checkpoint: Any,
) -> DatasetMetadataContext:
    """Infer dataset metadata and validate architecture compatibility."""
    ds_result = infer_dataset_metadata(
        data_path=data_path,
        config=config,
        num_players=num_players,
        model_version=model_version,
        multi_player=multi_player,
        use_streaming=use_streaming,
        distributed=distributed,
        is_main=is_main,
        resume_path=resume_path,
        num_filters=num_filters,
        num_res_blocks=num_res_blocks,
        device=device,
        BoardType=BoardType,
        HEX_BOARD_SIZE=HEX_BOARD_SIZE,
        HEX8_BOARD_SIZE=HEX8_BOARD_SIZE,
        MAX_PLAYERS=MAX_PLAYERS,
        get_policy_size_for_board=get_policy_size_for_board,
        normalize_board_type=normalize_board_type,
        validate_hex_policy_indices=validate_hex_policy_indices,
        detect_tier_from_checkpoint=detect_tier_from_checkpoint,
    )

    try:
        from app.training.board_encoding_contract import (
            get_expected_channels,
            is_valid_channel_count,
        )

        contract_channels = get_expected_channels(config.board_type, model_version)
        if ds_result.encoding_channels > 0 and not is_valid_channel_count(ds_result.encoding_channels):
            raise ValueError(
                f"[ContractViolation] NPZ has {ds_result.encoding_channels} channels which is "
                f"not a known encoding. Expected {contract_channels} for "
                f"{config.board_type.name}/{model_version}. Known: 40 (hex v2), "
                "56 (square families), 64 (hex v3/v4/v5-heavy families)."
            )
        if ds_result.encoding_channels > 0 and ds_result.encoding_channels != contract_channels:
            logger.info(
                "[ContractInfo] NPZ has %d channels, contract expects %d for %s/%s - using NPZ value (cross-board encoding)",
                ds_result.encoding_channels,
                contract_channels,
                config.board_type.name,
                model_version,
            )
    except ValueError:
        raise
    except (ImportError, AttributeError, TypeError) as contract_err:
        logger.debug("[ContractCheck] Skipped: %s", contract_err)

    detected_num_heuristics = ds_result.detected_num_heuristics

    def _validate_architecture_data_compatibility() -> None:
        nonlocal detected_num_heuristics

        v5_heavy_versions = (
            "v5",
            "v5-gnn",
            "v5-heavy",
            "v5-heavy-large",
            "v5-heavy-xl",
            "v6",
            "v6-xl",
        )
        if not (ds_result.use_hex_v5 or model_version in v5_heavy_versions):
            return

        try:
            from app.training.encoder_registry import get_encoder_config

            board_type_name = (
                config.board_type.name if hasattr(config.board_type, "name") else str(config.board_type)
            )
            version_key = "v5-heavy"
            encoder_config = get_encoder_config(board_type_name, version_key)
        except (ValueError, ImportError):
            return

        if not encoder_config.requires_heuristics:
            return

        min_required = encoder_config.min_heuristic_features
        actual_heuristics = detected_num_heuristics or 0
        if actual_heuristics < min_required:
            version_names = {
                "v6": "V5-Heavy-Large (deprecated alias)",
                "v6-xl": "V5-Heavy-XL (deprecated alias)",
                "v5-heavy-large": "V5-Heavy-Large",
                "v5-heavy-xl": "V5-Heavy-XL",
            }
            version_name = version_names.get(model_version, "V5-Heavy")
            raise ValueError(
                f"\n{'=' * 70}\n"
                "ARCHITECTURE-DATA COMPATIBILITY ERROR\n"
                f"{'=' * 70}\n\n"
                f"Model: {version_name} (--model-version {model_version})\n"
                f"  - Requires at least {min_required} heuristic features\n\n"
                f"Dataset: {data_path_str if data_path_str else 'unknown'}\n"
                f"  - Has {actual_heuristics} heuristic features\n\n"
                "SOLUTIONS:\n"
                "  1. Re-export data with --full-heuristics flag:\n"
                "     python scripts/export_replay_dataset.py --full-heuristics ...\n"
                "  2. Use a different architecture that doesn't require heuristics:\n"
                "     --model-version v2 or --model-version v4\n"
                f"{'=' * 70}"
            )

        if is_main:
            logger.info(
                "Architecture validation passed: %s requires %s heuristics, dataset has %s",
                model_version,
                min_required,
                actual_heuristics,
            )

    if ds_result.use_hex_model or ds_result.use_hex_v5 or model_version in (
        "v5",
        "v5-gnn",
        "v5-heavy",
        "v5-heavy-large",
        "v5-heavy-xl",
        "v6",
        "v6-xl",
    ):
        _validate_architecture_data_compatibility()

    return DatasetMetadataContext(
        board_size=ds_result.board_size,
        policy_size=ds_result.policy_size,
        encoding_channels=ds_result.encoding_channels,
        hex_num_players=ds_result.hex_num_players,
        use_hex_model=ds_result.use_hex_model,
        use_hex_v3=ds_result.use_hex_v3,
        use_hex_v4=ds_result.use_hex_v4,
        use_hex_v5=ds_result.use_hex_v5,
        use_hex_v5_large=ds_result.use_hex_v5_large,
        detected_num_heuristics=detected_num_heuristics,
        config_feature_version=ds_result.config_feature_version,
        hex_radius=ds_result.hex_radius,
    )


def prepare_training_data_pipeline(
    *,
    config: Any,
    data_path: str | list[str],
    data_path_str: str,
    data_dir: str | None,
    augment_hex_symmetry: bool,
    use_streaming: bool,
    sampling_weights: str,
    multi_player: bool,
    enable_elo_weighting: bool,
    min_quality_score: float,
    discover_synced_data: bool,
    distributed: bool,
    is_main: bool,
    policy_size: int,
    use_hex_model: bool,
    use_hex_v3: bool,
    model_version: str,
    config_feature_version: int,
    auto_streaming_threshold_bytes: int,
    has_data_catalog: bool,
    get_data_catalog: Any,
) -> TrainingDataPipelineContext | None:
    """Prepare dataset loaders, streaming readers, and dataset statistics."""
    train_streaming_loader: StreamingDataLoader | WeightedStreamingDataLoader | None = None
    val_streaming_loader: StreamingDataLoader | None = None
    train_loader: DataLoader | None = None
    val_loader: DataLoader | None = None
    train_sampler = None
    val_sampler = None
    full_dataset: Any | None = None
    allow_empty_policies = bool(getattr(config, "allow_empty_policies", False))
    filter_empty_policies = not allow_empty_policies
    value_only_training = False
    total_samples = 0
    num_data_files = 0

    if not use_streaming:
        total_data_size = 0
        paths_to_check: list[str] = []
        if data_dir is not None:
            paths_to_check = glob.glob(os.path.join(data_dir, "*.npz"))
        elif isinstance(data_path, list):
            paths_to_check = data_path
        elif data_path:
            paths_to_check = [data_path]

        for path in paths_to_check:
            if os.path.exists(path):
                total_data_size += os.path.getsize(path)

        if total_data_size > auto_streaming_threshold_bytes:
            size_gb = total_data_size / (1024 ** 3)
            threshold_gb = auto_streaming_threshold_bytes / (1024 ** 3)
            if is_main:
                logger.warning(
                    "Auto-enabling streaming mode: dataset size %.1fGB exceeds threshold %.0fGB. "
                    "Set RINGRIFT_AUTO_STREAMING_THRESHOLD_GB to adjust or use --use-streaming explicitly.",
                    size_gb,
                    threshold_gb,
                )
            use_streaming = True

    data_paths: list[str] = []
    if discover_synced_data and has_data_catalog:
        try:
            catalog = get_data_catalog()
            discovered_paths = catalog.get_recommended_training_sources(
                target_games=100000,
                board_type=config.board_type.value if hasattr(config, "board_type") else None,
                num_players=config.num_players if hasattr(config, "num_players") else 2,
            )
            if discovered_paths:
                data_paths.extend([str(path) for path in discovered_paths])
                if is_main:
                    stats = catalog.get_stats()
                    logger.info(
                        "DataCatalog discovered %d sources with %s total games (avg quality: %.3f)",
                        len(discovered_paths),
                        stats.total_games,
                        stats.avg_quality_score,
                    )
        except (ImportError, AttributeError, OSError, ConnectionError) as exc:
            if is_main:
                logger.warning("DataCatalog discovery failed: %s", exc)

    if use_streaming:
        if data_dir is not None:
            npz_pattern = os.path.join(data_dir, "*.npz")
            data_paths.extend(sorted(glob.glob(npz_pattern)))
            if is_main:
                logger.info("Found %d .npz files in %s", len(data_paths), data_dir)
        elif isinstance(data_path, list):
            data_paths.extend(data_path)
        elif data_path:
            data_paths.append(data_path)

        seen: set[str] = set()
        unique_paths: list[str] = []
        for path in data_paths:
            if path not in seen:
                seen.add(path)
                unique_paths.append(path)
        data_paths = unique_paths

        if not data_paths:
            raise ValueError(
                "No data files found for streaming training. "
                "Ensure --data-path or --data-dir points to valid .npz files."
            )

        first_path = data_paths[0]
        dataset_history_length: int | None = None
        policy_encoding: str | None = None
        dataset_feature_version: int | None = None
        dataset_in_channels: int | None = None
        dataset_globals_dim: int | None = None
        dataset_encoder_type: str | None = None
        dataset_base_channels: int | None = None
        dataset_board_type_meta: str | None = None
        dataset_encoder_version: str | None = None
        dataset_in_channels_meta: int | None = None
        is_npz = bool(first_path and first_path.endswith(".npz"))

        try:
            if first_path and os.path.exists(first_path):
                with safe_load_npz(first_path, mmap_mode="r") as dataset:
                    if "features" in dataset:
                        feat_shape = dataset["features"].shape
                        if len(feat_shape) >= 2:
                            dataset_in_channels = int(feat_shape[1])
                    if "globals" in dataset:
                        global_shape = dataset["globals"].shape
                        if len(global_shape) >= 2:
                            dataset_globals_dim = int(global_shape[1])
                    for key, target, caster in [
                        ("policy_encoding", "policy_encoding", str),
                        ("history_length", "dataset_history_length", int),
                        ("feature_version", "dataset_feature_version", int),
                        ("encoder_type", "dataset_encoder_type", str),
                        ("base_channels", "dataset_base_channels", int),
                        ("board_type", "dataset_board_type_meta", str),
                        ("encoder_version", "dataset_encoder_version", str),
                        ("in_channels", "dataset_in_channels_meta", int),
                    ]:
                        if key not in dataset:
                            continue
                        try:
                            value = caster(np.asarray(dataset[key]).item())
                        except (ValueError, TypeError, AttributeError):
                            value = None
                        if target == "policy_encoding":
                            policy_encoding = value
                        elif target == "dataset_history_length":
                            dataset_history_length = value
                        elif target == "dataset_feature_version":
                            dataset_feature_version = value
                        elif target == "dataset_encoder_type":
                            dataset_encoder_type = value
                        elif target == "dataset_base_channels":
                            dataset_base_channels = value
                        elif target == "dataset_board_type_meta":
                            dataset_board_type_meta = value
                        elif target == "dataset_encoder_version":
                            dataset_encoder_version = value
                        elif target == "dataset_in_channels_meta":
                            dataset_in_channels_meta = value
        except (OSError, KeyError, ValueError) as exc:
            if is_main:
                logger.warning("Failed to read dataset metadata from %s: %s", first_path, exc)

        if dataset_history_length is not None and dataset_history_length != config.history_length:
            raise ValueError(
                "Training history_length does not match dataset metadata.\n"
                f"  dataset={first_path}\n"
                f"  dataset_history_length={dataset_history_length}\n"
                f"  config.history_length={config.history_length}\n"
                "Regenerate the dataset with matching --history-length or update the training config."
            )
        if dataset_history_length is None and config.history_length != 3 and is_main:
            logger.warning(
                "Dataset %s missing history_length metadata; using config.history_length=%d. "
                "Ensure the dataset was built with matching history frames.",
                first_path,
                config.history_length,
            )

        if dataset_feature_version is not None and dataset_feature_version != config_feature_version:
            raise ValueError(
                "Training feature_version does not match dataset metadata.\n"
                f"  dataset={first_path}\n"
                f"  dataset_feature_version={dataset_feature_version}\n"
                f"  config_feature_version={config_feature_version}\n"
                "Regenerate the dataset with matching --feature-version or update the training config."
            )
        if dataset_feature_version is None:
            if config_feature_version != 1:
                autonomous_mode = os.environ.get("RINGRIFT_AUTONOMOUS_MODE", "").lower() in ("1", "true")
                if autonomous_mode:
                    if is_main:
                        logger.warning(
                            "[AUTONOMOUS] Dataset %s missing feature_version metadata. Config requested v%d but falling back to v1 for compatibility.",
                            first_path,
                            config_feature_version,
                        )
                    config_feature_version = 1
                else:
                    raise ValueError(
                        "Dataset is missing feature_version metadata but training "
                        f"was configured for feature_version={config_feature_version}.\n"
                        f"  dataset={first_path}\n"
                        "Regenerate the dataset with --feature-version or set feature_version=1 to use legacy features."
                    )
            if is_main:
                logger.warning(
                    "Dataset %s missing feature_version metadata; assuming legacy feature_version=1.",
                    first_path,
                )

        if dataset_globals_dim is None:
            if is_npz:
                raise ValueError(
                    "Dataset is missing globals features required for training.\n"
                    f"  dataset={first_path}\n"
                    "Regenerate the dataset with scripts/export_replay_dataset.py."
                )
        elif dataset_globals_dim != 20:
            raise ValueError(
                "Dataset globals feature dimension does not match the CNN encoder.\n"
                f"  dataset={first_path}\n"
                f"  dataset_globals_dim={dataset_globals_dim}\n"
                "Regenerate the dataset with scripts/export_replay_dataset.py to produce 20 global features."
            )

        if dataset_in_channels is not None:
            if use_hex_model:
                hex_base = 16 if use_hex_v3 else 10
                expected_in_channels = hex_base * (config.history_length + 1)
                expected_encoder = "hex_v3" if use_hex_v3 else "hex_v2"
            else:
                expected_in_channels = 14 * (config.history_length + 1)
                expected_encoder = "square"

            if dataset_encoder_type and is_main:
                logger.info(
                    "Dataset encoder metadata: type=%s, base_channels=%s, in_channels=%s, board_type=%s",
                    dataset_encoder_type,
                    dataset_base_channels,
                    dataset_in_channels,
                    dataset_board_type_meta,
                )
            if dataset_encoder_version and is_main:
                logger.info(
                    "Dataset V2.1 metadata: encoder_version=%s, in_channels_meta=%s",
                    dataset_encoder_version,
                    dataset_in_channels_meta,
                )

            if dataset_in_channels_meta is not None and dataset_in_channels_meta != dataset_in_channels:
                raise ValueError(
                    "========================================\n"
                    "DATA INTEGRITY ERROR - METADATA MISMATCH\n"
                    "========================================\n"
                    f"Dataset in_channels metadata: {dataset_in_channels_meta}\n"
                    f"Actual feature shape:         {dataset_in_channels} channels\n"
                    f"Dataset:                      {first_path}\n\n"
                    "The export script recorded a channel count that doesn't match the actual feature tensor shape.\n"
                    "SOLUTION: Re-export the data with a fixed export script.\n"
                    "========================================"
                )

            if dataset_encoder_type and dataset_encoder_type != expected_encoder:
                raise ValueError(
                    "========================================\n"
                    "ENCODER TYPE MISMATCH - CANNOT TRAIN\n"
                    "========================================\n"
                    f"Dataset encoded with: {dataset_encoder_type}\n"
                    f"Model expects:        {expected_encoder}\n"
                    f"Model version:        {model_version}\n"
                    f"Dataset:              {first_path}\n\n"
                    "SOLUTION: Re-export data with --encoder-version matching model version\n"
                    f"  For v3 model: use --encoder-version v3\n"
                    f"  For v2 model: use --encoder-version v2\n"
                    "========================================"
                )

            if dataset_board_type_meta:
                dataset_board_upper = dataset_board_type_meta.upper()
                config_board_name = config.board_type.name
                if dataset_board_upper != config_board_name:
                    raise ValueError(
                        "========================================\n"
                        "BOARD TYPE MISMATCH - CANNOT TRAIN\n"
                        "========================================\n"
                        f"Dataset board type:   {dataset_board_type_meta}\n"
                        f"Training board type:  {config.board_type.name}\n"
                        f"Dataset:              {first_path}\n\n"
                        "Dataset and training board types must match.\n"
                        "========================================"
                    )

            if dataset_in_channels != expected_in_channels:
                encoder_info = ""
                if dataset_encoder_type:
                    encoder_info = f"  dataset_encoder_type={dataset_encoder_type}\n"
                    encoder_info += f"  dataset_base_channels={dataset_base_channels}\n"
                    if dataset_board_type_meta:
                        encoder_info += f"  dataset_board_type={dataset_board_type_meta}\n"
                raise ValueError(
                    "Dataset feature channels do not match the expected encoder.\n"
                    f"  dataset={first_path}\n"
                    f"  dataset_in_channels={dataset_in_channels}\n"
                    f"  expected_in_channels={expected_in_channels} ({expected_encoder})\n"
                    f"{encoder_info}"
                    f"Model expects {expected_encoder} encoder ({expected_in_channels} channels).\n"
                    "Solutions:\n"
                    "  1. Regenerate dataset with matching encoder version:\n"
                    f"     --encoder-version {'v3' if use_hex_v3 else 'v2'}\n"
                    "  2. Or use matching model version for your data:\n"
                    f"     --model-version {'v2' if dataset_in_channels == 40 else 'v3' if dataset_in_channels == 64 else 'unknown'}"
                )
        elif is_npz:
            raise ValueError(
                "Dataset is missing features required for training.\n"
                f"  dataset={first_path}\n"
                "Regenerate the dataset with scripts/export_replay_dataset.py."
            )

        if model_version in ("v3", "v4"):
            if policy_encoding == "legacy_max_n":
                raise ValueError(
                    f"Dataset uses legacy MAX_N policy encoding but --model-version={model_version} "
                    f"requires board-aware policy encoding.\n  dataset={first_path}\n"
                    "Regenerate the dataset with --board-aware-encoding."
                )
            if policy_encoding is None and is_main:
                logger.warning(
                    "Dataset %s missing policy_encoding metadata; assuming board-aware encoding for %s. "
                    "If this dataset was exported with legacy MAX_N, regenerate with --board-aware-encoding.",
                    first_path,
                    model_version,
                )

        total_samples = sum(get_sample_count(path) for path in data_paths if os.path.exists(path))
        num_data_files = len(data_paths)
        if total_samples == 0:
            if is_main:
                logger.warning("No samples found in data files; skipping.")
            if distributed:
                cleanup_distributed()
            return None

        if is_main:
            logger.info(
                "StreamingDataLoader: %d total samples across %d files",
                total_samples,
                len(data_paths),
            )

        val_split = 0.2
        val_samples = int(total_samples * val_split)
        train_samples = total_samples - val_samples
        if distributed:
            stream_rank = get_rank()
            stream_world_size = get_world_size()
        else:
            stream_rank = 0
            stream_world_size = 1

        if sampling_weights != "uniform":
            train_streaming_loader = WeightedStreamingDataLoader(
                data_paths=data_paths,
                batch_size=config.batch_size,
                shuffle=True,
                seed=config.seed,
                drop_last=False,
                policy_size=policy_size,
                rank=stream_rank,
                world_size=stream_world_size,
                filter_empty_policies=filter_empty_policies,
                sampling_weights=sampling_weights,
            )
            if is_main:
                logger.info(
                    "Using WeightedStreamingDataLoader with sampling_weights=%s",
                    sampling_weights,
                )
        else:
            train_streaming_loader = StreamingDataLoader(
                data_paths=data_paths,
                batch_size=config.batch_size,
                shuffle=True,
                seed=config.seed,
                drop_last=False,
                policy_size=policy_size,
                rank=stream_rank,
                world_size=stream_world_size,
                filter_empty_policies=filter_empty_policies,
            )

        val_streaming_loader = StreamingDataLoader(
            data_paths=data_paths,
            batch_size=config.batch_size,
            shuffle=False,
            seed=config.seed + 1000,
            drop_last=False,
            policy_size=policy_size,
            rank=stream_rank,
            world_size=stream_world_size,
            filter_empty_policies=filter_empty_policies,
        )

        if train_streaming_loader.has_multi_player_values and not multi_player and is_main:
            logger.info(
                "Dataset contains multi-player value vectors (values_mp). Consider using --multi-player flag for multi-player training."
            )
        if multi_player and not train_streaming_loader.has_multi_player_values:
            if is_main:
                logger.error(
                    "multi_player=True but streaming dataset does not contain 'values_mp' / 'num_players'. "
                    "Regenerate data with multi-player value targets or disable --multi-player."
                )
            if distributed:
                cleanup_distributed()
            raise ValueError("Multi-player training requested but streaming dataset lacks values_mp.")

        if not train_streaming_loader.has_policy:
            if is_main:
                logger.info(
                    "Dataset has no policy data - enabling value-only training mode (policy_weight=0). "
                    "Policy head will not be trained."
                )
            config.policy_weight = 0.0
            value_only_training = True

        return TrainingDataPipelineContext(
            use_streaming=True,
            train_streaming_loader=train_streaming_loader,
            val_streaming_loader=val_streaming_loader,
            train_loader=None,
            val_loader=None,
            train_sampler=None,
            val_sampler=None,
            full_dataset=None,
            train_size=train_samples,
            val_size=val_samples,
            total_samples=total_samples,
            num_data_files=num_data_files,
            value_only_training=value_only_training,
        )

    if isinstance(data_path, list):
        data_path_str = data_path[0] if data_path else ""

    use_heuristics = model_version in ("v5", "v5-gnn", "v5-heavy")
    if sampling_weights == "uniform":
        full_dataset = RingRiftDataset(
            data_path_str,
            board_type=config.board_type,
            augment_hex=augment_hex_symmetry,
            use_multi_player_values=multi_player,
            filter_empty_policies=filter_empty_policies,
            return_num_players=multi_player,
            return_heuristics=use_heuristics,
        )
        use_weighted_sampling = False
    else:
        full_dataset = WeightedRingRiftDataset(
            data_path_str,
            board_type=config.board_type,
            augment_hex=augment_hex_symmetry,
            weighting=sampling_weights,
            use_multi_player_values=multi_player,
            filter_empty_policies=filter_empty_policies,
            return_num_players=multi_player,
            return_heuristics=use_heuristics,
        )
        use_weighted_sampling = True

    elo_sample_weights: np.ndarray | None = None
    if enable_elo_weighting and data_path_str and os.path.exists(data_path_str):
        try:
            with safe_load_npz(data_path_str, mmap_mode="r") as dataset:
                if "opponent_elo" in dataset:
                    from app.training.elo_weighting import compute_elo_weights

                    opponent_elos = np.array(dataset["opponent_elo"])
                    elo_sample_weights = compute_elo_weights(
                        opponent_elos,
                        model_elo=1500.0,
                        elo_scale=400.0,
                        min_weight=0.2,
                        max_weight=3.0,
                    )
                    if is_main:
                        logger.info(
                            "ELO weighting enabled: %d samples, weight range [%.3f, %.3f]",
                            len(elo_sample_weights),
                            elo_sample_weights.min(),
                            elo_sample_weights.max(),
                        )
                elif is_main:
                    logger.info(
                        "ELO weighting requested but dataset lacks 'opponent_elo' field. "
                        "Regenerate with export_replay_dataset.py to include opponent ELO data."
                    )
        except (OSError, KeyError, ValueError) as exc:
            if is_main:
                logger.warning("Failed to load ELO weights: %s", exc)

    quality_sample_weights: np.ndarray | None = None
    if data_path_str and os.path.exists(data_path_str):
        try:
            with safe_load_npz(data_path_str, mmap_mode="r") as dataset:
                if "quality_score" in dataset:
                    quality_scores = np.array(dataset["quality_score"])
                    if min_quality_score > 0.0:
                        quality_mask = quality_scores >= min_quality_score
                        if is_main:
                            logger.info(
                                "Quality filtering: %d samples below threshold (%.2f) will be weighted to 0",
                                int(np.sum(~quality_mask)),
                                min_quality_score,
                            )
                        quality_sample_weights = np.where(quality_mask, quality_scores, 0.0)
                    else:
                        quality_sample_weights = quality_scores
                    if is_main:
                        nonzero = quality_sample_weights[quality_sample_weights > 0]
                        if len(nonzero) > 0:
                            logger.info(
                                "Quality weighting enabled: %d samples, weight range [%.3f, %.3f]",
                                len(quality_sample_weights),
                                nonzero.min(),
                                nonzero.max(),
                            )
                elif is_main:
                    logger.debug(
                        "Dataset lacks 'quality_score' field - quality weighting disabled. "
                        "Regenerate with export_replay_dataset.py to include quality data."
                    )
        except (OSError, KeyError, ValueError) as exc:
            if is_main:
                logger.warning("Failed to load quality scores: %s", exc)

    generator_elo_weights: np.ndarray | None = None
    if enable_elo_weighting and data_path_str and os.path.exists(data_path_str):
        try:
            with safe_load_npz(data_path_str, mmap_mode="r") as dataset:
                if "generator_elo" in dataset:
                    from app.training.elo_weighting import compute_generator_elo_weights

                    generator_elos = np.array(dataset["generator_elo"])
                    generator_elo_weights = compute_generator_elo_weights(
                        generator_elos,
                        baseline_elo=1000.0,
                        elo_scale=200.0,
                        min_weight=0.3,
                        max_weight=3.0,
                    )
                    if is_main:
                        logger.info(
                            "Generator Elo weighting enabled: %d samples, weight range [%.3f, %.3f]",
                            len(generator_elo_weights),
                            generator_elo_weights.min(),
                            generator_elo_weights.max(),
                        )
                elif is_main:
                    logger.debug(
                        "Dataset lacks 'generator_elo' field - generator Elo weighting disabled. "
                        "Regenerate with export_replay_dataset.py to include generator Elo data."
                    )
        except (OSError, KeyError, ValueError) as exc:
            if is_main:
                logger.warning("Failed to load generator Elo weights: %s", exc)

    if quality_sample_weights is not None and is_main:
        try:
            from app.training.improvement_optimizer import get_improvement_optimizer

            avg_quality = float(np.mean(quality_sample_weights[quality_sample_weights > 0]))
            optimizer_instance = get_improvement_optimizer()
            recommendation = optimizer_instance.record_data_quality(
                parity_success_rate=1.0,
                data_quality_score=avg_quality,
            )
            logger.info(
                "[ImprovementOptimizer] Recorded data quality: %.3f (signal: %s, threshold_adj: %.2f)",
                avg_quality,
                recommendation.signal.name,
                recommendation.threshold_adjustment,
            )
        except ImportError:
            pass
        except (AttributeError, TypeError) as exc:
            logger.debug("[ImprovementOptimizer] Failed to record data quality: %s", exc)

    if len(full_dataset) == 0:
        if is_main:
            logger.warning("Training dataset at %s is empty; skipping.", data_path_str)
        if distributed:
            cleanup_distributed()
        return None

    if multi_player and not getattr(full_dataset, "has_multi_player_values", False):
        if is_main:
            logger.error(
                "multi_player=True but dataset %s does not contain 'values_mp' / 'num_players'. "
                "Regenerate data with multi-player value targets or disable --multi-player.",
                data_path_str,
            )
        if distributed:
            cleanup_distributed()
        raise ValueError("Multi-player training requested but dataset lacks values_mp.")

    if not getattr(full_dataset, "has_policy", True):
        if is_main:
            logger.info(
                "Dataset has no policy data - enabling value-only training mode (policy_weight=0). "
                "Policy head will not be trained."
            )
        config.policy_weight = 0.0
        value_only_training = True

    shape = getattr(full_dataset, "spatial_shape", None)
    if shape is not None and is_main:
        height, width = shape
        logger.info("Dataset spatial feature shape inferred as %dx%d.", height, width)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    if distributed:
        train_sampler = get_distributed_sampler(train_dataset, shuffle=True)
        val_sampler = get_distributed_sampler(val_dataset, shuffle=False)
        env_workers = os.environ.get("RINGRIFT_DATALOADER_WORKERS")
        if env_workers is not None:
            num_loader_workers = int(env_workers)
        elif sys.platform == "darwin":
            num_loader_workers = 0
        else:
            import multiprocessing

            num_loader_workers = min(4, multiprocessing.cpu_count() // 2) if not use_streaming else 0
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            sampler=train_sampler,
            num_workers=num_loader_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=num_loader_workers,
            pin_memory=True,
        )
    else:
        use_any_weighting = (
            use_weighted_sampling
            or (elo_sample_weights is not None)
            or (quality_sample_weights is not None)
            or (generator_elo_weights is not None)
        )
        if use_any_weighting and isinstance(train_dataset, torch.utils.data.Subset):
            subset_indices = np.array(train_dataset.indices, dtype=np.int64)
            if use_weighted_sampling:
                base_dataset = cast(WeightedRingRiftDataset, train_dataset.dataset)
                if base_dataset.sample_weights is None:
                    train_weights_np = np.ones(len(train_dataset), dtype=np.float32)
                else:
                    train_weights_np = base_dataset.sample_weights[subset_indices].astype(np.float32)
            else:
                train_weights_np = np.ones(len(train_dataset), dtype=np.float32)

            if elo_sample_weights is not None:
                train_weights_np = train_weights_np * elo_sample_weights[subset_indices].astype(np.float32)
            if quality_sample_weights is not None:
                train_weights_np = train_weights_np * quality_sample_weights[subset_indices].astype(np.float32)
            if generator_elo_weights is not None:
                train_weights_np = train_weights_np * generator_elo_weights[subset_indices].astype(np.float32)

            if is_main:
                weight_sources: list[str] = []
                if use_weighted_sampling:
                    weight_sources.append("position")
                if elo_sample_weights is not None:
                    weight_sources.append("ELO")
                if quality_sample_weights is not None:
                    weight_sources.append("quality")
                if generator_elo_weights is not None:
                    weight_sources.append("generator_elo")
                nonzero = train_weights_np[train_weights_np > 0]
                if len(nonzero) > 0:
                    logger.info(
                        "Combined weights (%s): %d/%d samples with weight > 0, range [%.3f, %.3f]",
                        " * ".join(weight_sources),
                        len(nonzero),
                        len(train_weights_np),
                        nonzero.min(),
                        nonzero.max(),
                    )

            train_weights = torch.from_numpy(train_weights_np)
            train_sampler = WeightedRandomSampler(
                weights=train_weights,
                num_samples=len(train_dataset),
                replacement=True,
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                sampler=train_sampler,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
            )
        else:
            train_sampler = None
            train_loader = DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=True,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
            )

    return TrainingDataPipelineContext(
        use_streaming=use_streaming,
        train_streaming_loader=None,
        val_streaming_loader=None,
        train_loader=train_loader,
        val_loader=val_loader,
        train_sampler=train_sampler,
        val_sampler=val_sampler,
        full_dataset=full_dataset,
        train_size=train_size,
        val_size=val_size,
        total_samples=0,
        num_data_files=0,
        value_only_training=value_only_training,
    )


__all__ = [
    "DatasetMetadataContext",
    "TrainingDataPipelineContext",
    "prepare_dataset_metadata_context",
    "prepare_training_data_pipeline",
    "validate_training_data",
]
