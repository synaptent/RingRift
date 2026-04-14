"""Training data export module.

This module provides a modular export pipeline for converting game databases
into NPZ training datasets. It replaces the monolithic export_replay_dataset.py
script with focused, testable components.

Components:
    - config: Configuration dataclasses (ExportConfig, FilterConfig, ExportResult)
    - game_processor: Database iteration and game filtering
    - sample_collector: Sample encoding and augmentation
    - array_builder: Array accumulation and stacking
    - npz_writer: NPZ output with validation

Usage:
    from app.training.export import (
        ExportConfig,
        FilterConfig,
        ExportResult,
        ArrayBuilder,
        Sample,
    )

    config = ExportConfig(
        board_type="hex8",
        num_players=2,
        output_path=Path("data/training/hex8_2p.npz"),
    )

    builder = ArrayBuilder()
    builder.add_sample(Sample(...))
    arrays = builder.build_arrays()
"""

from app.training.export.array_builder import (
    ArrayBuilder,
    BuiltArrays,
    Sample,
    load_existing_arrays,
    merge_built_arrays,
)
from app.training.export.config import (
    DB_LOCK_INITIAL_WAIT,
    DB_LOCK_MAX_RETRIES,
    DB_LOCK_MAX_WAIT,
    DISK_SPACE_SAFETY_MARGIN_MB,
    NPZ_COMPRESSION_RATIO,
    ExportConfig,
    ExportResult,
    FilterConfig,
)
from app.training.export.npz_writer import (
    NPZExportWriter,
    WriteResult,
    check_disk_space,
    estimate_npz_size,
    register_with_manifest,
)
from app.training.export.sample_collector import (
    CollectionResult,
    GameMetadata,
    SampleCollector,
    SampleCollectorConfig,
    create_collector,
)
from app.training.export.game_processor import (
    GameData,
    GameIterator,
    GameIteratorConfig,
    IterationStats,
    create_iterator,
)
from app.training.export.orchestrator import (
    ExportOrchestrator,
    ExportProgress,
    export_dataset,
)

__all__ = [
    # Configuration dataclasses
    "ExportConfig",
    "FilterConfig",
    "ExportResult",
    # Array building
    "ArrayBuilder",
    "BuiltArrays",
    "Sample",
    "load_existing_arrays",
    "merge_built_arrays",
    # Sample collection
    "SampleCollector",
    "SampleCollectorConfig",
    "CollectionResult",
    "GameMetadata",
    "create_collector",
    # NPZ writing
    "NPZExportWriter",
    "WriteResult",
    "check_disk_space",
    "estimate_npz_size",
    "register_with_manifest",
    # Game processing
    "GameIterator",
    "GameIteratorConfig",
    "GameData",
    "IterationStats",
    "create_iterator",
    # Orchestration
    "ExportOrchestrator",
    "ExportProgress",
    "export_dataset",
    # Constants
    "DB_LOCK_MAX_RETRIES",
    "DB_LOCK_INITIAL_WAIT",
    "DB_LOCK_MAX_WAIT",
    "DISK_SPACE_SAFETY_MARGIN_MB",
    "NPZ_COMPRESSION_RATIO",
]


def __dir__() -> list[str]:
    """Expose the intended training-export surface for discoverability."""

    return sorted(set(globals()) | set(__all__))
