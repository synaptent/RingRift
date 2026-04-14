"""Focused tests for app.training.export package exports."""

from __future__ import annotations

import importlib


def test_package_dir_lists_declared_training_export_surface() -> None:
    module = importlib.import_module("app.training.export")

    expected = [
        "ExportConfig",
        "FilterConfig",
        "ExportResult",
        "ArrayBuilder",
        "BuiltArrays",
        "Sample",
        "load_existing_arrays",
        "merge_built_arrays",
        "SampleCollector",
        "SampleCollectorConfig",
        "CollectionResult",
        "GameMetadata",
        "create_collector",
        "NPZExportWriter",
        "WriteResult",
        "check_disk_space",
        "estimate_npz_size",
        "register_with_manifest",
        "GameIterator",
        "GameIteratorConfig",
        "GameData",
        "IterationStats",
        "create_iterator",
        "ExportOrchestrator",
        "ExportProgress",
        "export_dataset",
        "DB_LOCK_MAX_RETRIES",
        "DB_LOCK_INITIAL_WAIT",
        "DB_LOCK_MAX_WAIT",
        "DISK_SPACE_SAFETY_MARGIN_MB",
        "NPZ_COMPRESSION_RATIO",
    ]

    assert module.__all__ == expected
    assert len(module.__all__) == len(set(module.__all__))

    for name in expected:
        assert hasattr(module, name)
        assert name in dir(module)
