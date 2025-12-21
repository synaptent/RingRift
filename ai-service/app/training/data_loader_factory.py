"""Data loader factory for RingRift neural network training.

This module centralizes data loading logic, extracting what was previously
~500 lines of data loading code from train_model().

December 2025: Extracted from train.py to improve modularity.
"""

from __future__ import annotations

import glob
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split

from app.models import BoardType
from app.training.data_loader import (
    StreamingDataLoader,
    WeightedStreamingDataLoader,
    get_sample_count,
)
from app.training.datasets import RingRiftDataset, WeightedRingRiftDataset

logger = logging.getLogger(__name__)

# Auto-streaming threshold: 20GB by default
AUTO_STREAMING_THRESHOLD_BYTES = int(
    os.environ.get("RINGRIFT_AUTO_STREAMING_THRESHOLD_GB", "20")
) * (1024 ** 3)


@dataclass
class DataLoaderConfig:
    """Configuration for data loader creation."""

    batch_size: int = 256
    use_streaming: bool = False
    sampling_weights: str = 'uniform'
    augment_hex_symmetry: bool = False
    multi_player: bool = False
    filter_empty_policies: bool = True
    seed: int = 42
    board_type: BoardType = BoardType.SQUARE8
    policy_size: int = 512

    # Distributed training
    distributed: bool = False
    rank: int = 0
    world_size: int = 1

    # Data paths
    data_path: str | list[str] | None = None
    data_dir: str | None = None


@dataclass
class DataLoaderResult:
    """Result of creating data loaders."""

    train_loader: DataLoader | StreamingDataLoader | None = None
    val_loader: DataLoader | StreamingDataLoader | None = None
    train_sampler: Any | None = None
    val_sampler: Any | None = None
    train_size: int = 0
    val_size: int = 0
    use_streaming: bool = False
    has_multi_player_values: bool = False
    data_paths: list[str] | None = None


def should_use_streaming(
    data_path: str | list[str] | None,
    data_dir: str | None,
    threshold_bytes: int = AUTO_STREAMING_THRESHOLD_BYTES,
) -> bool:
    """Check if streaming mode should be auto-enabled based on data size.

    Args:
        data_path: Path(s) to data files
        data_dir: Directory containing data files
        threshold_bytes: Size threshold for auto-streaming

    Returns:
        True if streaming should be used
    """
    total_size = 0
    paths_to_check: list[str] = []

    if data_dir is not None:
        npz_pattern = os.path.join(data_dir, "*.npz")
        paths_to_check = glob.glob(npz_pattern)
    elif isinstance(data_path, list):
        paths_to_check = data_path
    elif data_path:
        paths_to_check = [data_path]

    for p in paths_to_check:
        if os.path.exists(p):
            total_size += os.path.getsize(p)

    if total_size > threshold_bytes:
        size_gb = total_size / (1024 ** 3)
        threshold_gb = threshold_bytes / (1024 ** 3)
        logger.warning(
            f"Auto-enabling streaming mode: dataset size {size_gb:.1f}GB "
            f"exceeds threshold {threshold_gb:.0f}GB"
        )
        return True

    return False


def collect_data_paths(
    data_path: str | list[str] | None,
    data_dir: str | None,
) -> list[str]:
    """Collect all data paths for streaming mode.

    Args:
        data_path: Path(s) to data files
        data_dir: Directory containing data files

    Returns:
        List of unique data file paths
    """
    paths: list[str] = []

    if data_dir is not None:
        npz_pattern = os.path.join(data_dir, "*.npz")
        paths.extend(sorted(glob.glob(npz_pattern)))
        logger.info(f"Found {len(paths)} .npz files in {data_dir}")
    elif isinstance(data_path, list):
        paths.extend(data_path)
    elif data_path:
        paths.append(data_path)

    # Remove duplicates while preserving order
    seen = set()
    unique_paths = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            unique_paths.append(p)

    return unique_paths


def create_streaming_loaders(
    config: DataLoaderConfig,
    data_paths: list[str],
) -> DataLoaderResult:
    """Create streaming data loaders for large datasets.

    Args:
        config: Data loader configuration
        data_paths: List of data file paths

    Returns:
        DataLoaderResult with streaming loaders
    """
    if not data_paths:
        logger.warning("No data files found for streaming")
        return DataLoaderResult(use_streaming=True)

    # Get total sample count
    total_samples = sum(
        get_sample_count(p) for p in data_paths if os.path.exists(p)
    )

    if total_samples == 0:
        logger.warning("No samples found in data files")
        return DataLoaderResult(use_streaming=True)

    logger.info(
        f"StreamingDataLoader: {total_samples} total samples "
        f"across {len(data_paths)} files"
    )

    # Calculate split
    val_split = 0.2
    val_samples = int(total_samples * val_split)
    train_samples = total_samples - val_samples

    # Determine rank/world_size for distributed sharding
    stream_rank = config.rank if config.distributed else 0
    stream_world_size = config.world_size if config.distributed else 1

    # Create train loader
    if config.sampling_weights != 'uniform':
        train_loader = WeightedStreamingDataLoader(
            data_paths=data_paths,
            batch_size=config.batch_size,
            shuffle=True,
            seed=config.seed,
            drop_last=False,
            policy_size=config.policy_size,
            rank=stream_rank,
            world_size=stream_world_size,
            filter_empty_policies=config.filter_empty_policies,
            sampling_weights=config.sampling_weights,
        )
        logger.info(
            f"Using WeightedStreamingDataLoader with "
            f"sampling_weights={config.sampling_weights}"
        )
    else:
        train_loader = StreamingDataLoader(
            data_paths=data_paths,
            batch_size=config.batch_size,
            shuffle=True,
            seed=config.seed,
            drop_last=False,
            policy_size=config.policy_size,
            rank=stream_rank,
            world_size=stream_world_size,
            filter_empty_policies=config.filter_empty_policies,
        )

    # Create validation loader (always uniform sampling)
    val_loader = StreamingDataLoader(
        data_paths=data_paths,
        batch_size=config.batch_size,
        shuffle=False,
        seed=config.seed + 1000,
        drop_last=False,
        policy_size=config.policy_size,
        rank=stream_rank,
        world_size=stream_world_size,
        filter_empty_policies=config.filter_empty_policies,
    )

    # Check for multi-player values
    has_mp = getattr(train_loader, 'has_multi_player_values', False)

    return DataLoaderResult(
        train_loader=train_loader,
        val_loader=val_loader,
        train_size=train_samples,
        val_size=val_samples,
        use_streaming=True,
        has_multi_player_values=has_mp,
        data_paths=data_paths,
    )


def create_standard_loaders(
    config: DataLoaderConfig,
    data_path: str,
) -> DataLoaderResult:
    """Create standard (non-streaming) data loaders.

    Args:
        config: Data loader configuration
        data_path: Path to data file

    Returns:
        DataLoaderResult with standard loaders
    """
    # Create dataset
    if config.sampling_weights == 'uniform':
        full_dataset = RingRiftDataset(
            data_path,
            board_type=config.board_type,
            augment_hex=config.augment_hex_symmetry,
            use_multi_player_values=config.multi_player,
            filter_empty_policies=config.filter_empty_policies,
            return_num_players=config.multi_player,
        )
        use_weighted_sampling = False
    else:
        full_dataset = WeightedRingRiftDataset(
            data_path,
            board_type=config.board_type,
            augment_hex=config.augment_hex_symmetry,
            weighting=config.sampling_weights,
            use_multi_player_values=config.multi_player,
            filter_empty_policies=config.filter_empty_policies,
            return_num_players=config.multi_player,
        )
        use_weighted_sampling = True

    if len(full_dataset) == 0:
        logger.warning(f"Training dataset at {data_path} is empty")
        return DataLoaderResult()

    # Check for multi-player values
    has_mp = getattr(full_dataset, "has_multi_player_values", False)

    # Log spatial shape if available
    shape = getattr(full_dataset, "spatial_shape", None)
    if shape is not None:
        h, w = shape
        logger.info(f"Dataset spatial feature shape: {h}x{w}")

    # Split into train/val (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Create data loaders
    train_sampler = None
    val_sampler = None

    if config.distributed:
        from app.training.distributed import get_distributed_sampler

        train_sampler = get_distributed_sampler(train_dataset, shuffle=True)
        val_sampler = get_distributed_sampler(val_dataset, shuffle=False)

        # Determine number of workers
        env_workers = os.environ.get("RINGRIFT_DATALOADER_WORKERS")
        if env_workers is not None:
            num_workers = int(env_workers)
        elif sys.platform == "darwin":
            num_workers = 0  # macOS: mmap incompatible with multiprocessing
        else:
            import multiprocessing
            num_workers = min(4, multiprocessing.cpu_count() // 2)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=False,  # Sampler handles shuffling
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        # Non-distributed: optionally use weighted sampling
        if use_weighted_sampling and hasattr(train_dataset, 'dataset'):
            base_dataset = train_dataset.dataset
            if hasattr(base_dataset, 'sample_weights') and base_dataset.sample_weights is not None:
                subset_indices = np.array(train_dataset.indices, dtype=np.int64)
                train_weights = torch.from_numpy(
                    base_dataset.sample_weights[subset_indices].astype(np.float32)
                )
                train_sampler = WeightedRandomSampler(
                    weights=train_weights,
                    num_samples=len(train_dataset),
                    replacement=True,
                )

        if train_sampler is not None:
            train_loader = DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                sampler=train_sampler,
            )
        else:
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

    return DataLoaderResult(
        train_loader=train_loader,
        val_loader=val_loader,
        train_sampler=train_sampler,
        val_sampler=val_sampler,
        train_size=train_size,
        val_size=val_size,
        use_streaming=False,
        has_multi_player_values=has_mp,
    )


def create_data_loaders(config: DataLoaderConfig) -> DataLoaderResult:
    """Create appropriate data loaders based on configuration.

    This is the main entry point that decides between streaming and
    standard loaders.

    Args:
        config: Data loader configuration

    Returns:
        DataLoaderResult with created loaders
    """
    # Check if streaming should be auto-enabled
    use_streaming = config.use_streaming or should_use_streaming(
        config.data_path,
        config.data_dir,
    )

    if use_streaming:
        # Collect all data paths
        data_paths = collect_data_paths(config.data_path, config.data_dir)
        return create_streaming_loaders(config, data_paths)
    else:
        # Get single data path
        if isinstance(config.data_path, list):
            data_path = config.data_path[0] if config.data_path else ""
        else:
            data_path = config.data_path or ""

        if not data_path or not os.path.exists(data_path):
            logger.error(f"Data path does not exist: {data_path}")
            return DataLoaderResult()

        return create_standard_loaders(config, data_path)


def validate_dataset_metadata(
    data_path: str,
    config_history_length: int,
    config_feature_version: int,
    model_version: str = 'v2',
    distributed: bool = False,
    is_main_process: bool = True,
) -> dict[str, Any]:
    """Validate dataset metadata against training configuration.

    Args:
        data_path: Path to data file
        config_history_length: Expected history length
        config_feature_version: Expected feature version
        model_version: Model version being trained
        distributed: Whether running distributed
        is_main_process: Whether this is the main process

    Returns:
        Dictionary with extracted metadata

    Raises:
        ValueError: If metadata is incompatible
    """
    metadata = {}

    if not os.path.exists(data_path):
        return metadata

    try:
        with np.load(data_path, mmap_mode="r", allow_pickle=True) as d:
            # Extract metadata
            if "features" in d:
                feat_shape = d["features"].shape
                if len(feat_shape) >= 2:
                    metadata['in_channels'] = int(feat_shape[1])

            if "globals" in d:
                glob_shape = d["globals"].shape
                if len(glob_shape) >= 2:
                    metadata['globals_dim'] = int(glob_shape[1])

            if "policy_encoding" in d:
                try:
                    metadata['policy_encoding'] = str(np.asarray(d["policy_encoding"]).item())
                except Exception:
                    pass

            if "history_length" in d:
                try:
                    metadata['history_length'] = int(np.asarray(d["history_length"]).item())
                except Exception:
                    pass

            if "feature_version" in d:
                try:
                    metadata['feature_version'] = int(np.asarray(d["feature_version"]).item())
                except Exception:
                    pass

            if "policy_indices" in d:
                pi = d["policy_indices"]
                max_idx = -1
                for i in range(len(pi)):
                    arr = np.asarray(pi[i])
                    if arr.size == 0:
                        continue
                    local_max = int(arr.max())
                    if local_max > max_idx:
                        max_idx = local_max
                if max_idx >= 0:
                    metadata['inferred_policy_size'] = max_idx + 1

    except Exception as e:
        if is_main_process:
            logger.warning(f"Failed to read dataset metadata from {data_path}: {e}")
        return metadata

    # Validate history_length
    if 'history_length' in metadata:
        if metadata['history_length'] != config_history_length:
            raise ValueError(
                f"Training history_length ({config_history_length}) does not match "
                f"dataset metadata ({metadata['history_length']}).\n"
                f"  dataset={data_path}\n"
                "Regenerate the dataset with matching --history-length."
            )

    # Validate feature_version
    if 'feature_version' in metadata:
        if metadata['feature_version'] != config_feature_version:
            raise ValueError(
                f"Training feature_version ({config_feature_version}) does not match "
                f"dataset metadata ({metadata['feature_version']}).\n"
                f"  dataset={data_path}\n"
                "Regenerate the dataset with matching --feature-version."
            )

    # Validate globals
    if 'globals_dim' in metadata and metadata['globals_dim'] != 20:
        raise ValueError(
            f"Dataset globals dimension ({metadata['globals_dim']}) does not match "
            f"expected (20).\n"
            f"  dataset={data_path}\n"
            "Regenerate the dataset with scripts/export_replay_dataset.py."
        )

    # Validate policy encoding for v3/v4
    if model_version in ('v3', 'v4'):
        if metadata.get('policy_encoding') == "legacy_max_n":
            raise ValueError(
                f"Dataset uses legacy MAX_N policy encoding but --model-version={model_version} "
                "requires board-aware policy encoding.\n"
                f"  dataset={data_path}\n"
                "Regenerate the dataset with --board-aware-encoding."
            )

    return metadata


__all__ = [
    'AUTO_STREAMING_THRESHOLD_BYTES',
    'DataLoaderConfig',
    'DataLoaderResult',
    'collect_data_paths',
    'create_data_loaders',
    'create_standard_loaders',
    'create_streaming_loaders',
    'should_use_streaming',
    'validate_dataset_metadata',
]
