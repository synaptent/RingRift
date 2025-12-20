#!/usr/bin/env python3
"""
Batch Checkpoint Migration Script

Migrates legacy checkpoint files to versioned format with full metadata.
Supports dry-run mode, in-place migration, and detailed reporting.

Usage:
    # Dry run to see what would be migrated
    python scripts/migrate_checkpoints.py models/ --dry-run

    # Migrate all legacy checkpoints in place
    python scripts/migrate_checkpoints.py models/ --in-place

    # Migrate to a separate output directory
    python scripts/migrate_checkpoints.py models/ --output migrated_models/

    # Specify model class for all files
    python scripts/migrate_checkpoints.py models/ --model-class RingRiftCNN_v2

December 2025 - RingRift AI Service
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from app.training.model_versioning import (
    ModelVersionManager,
    ModelMetadata,
    LegacyCheckpointError,
    MODEL_VERSIONS,
    compute_state_dict_checksum,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Model class inference patterns
MODEL_CLASS_PATTERNS = {
    "v2_lite": "RingRiftCNN_v2_Lite",
    "v2-lite": "RingRiftCNN_v2_Lite",
    "lite": "RingRiftCNN_v2_Lite",
    "v3_lite": "RingRiftCNN_v3_Lite",
    "v3-lite": "RingRiftCNN_v3_Lite",
    "v4": "RingRiftCNN_v4",
    "v3": "RingRiftCNN_v3",
    "v2": "RingRiftCNN_v2",
    "hex_v3_lite": "HexNeuralNet_v3_Lite",
    "hex_v3": "HexNeuralNet_v3",
    "hex_v2_lite": "HexNeuralNet_v2_Lite",
    "hex_v2": "HexNeuralNet_v2",
    "hex": "HexNeuralNet_v2",
    "square8": "RingRiftCNN_v2",
    "square19": "RingRiftCNN_v2",
}


@dataclass
class CheckpointInfo:
    """Information about a checkpoint file."""
    path: Path
    is_legacy: bool
    model_class: Optional[str] = None
    architecture_version: Optional[str] = None
    size_bytes: int = 0
    created_at: Optional[datetime] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": str(self.path),
            "is_legacy": self.is_legacy,
            "model_class": self.model_class,
            "architecture_version": self.architecture_version,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "error": self.error,
        }


@dataclass
class MigrationResult:
    """Result of a migration operation."""
    source_path: Path
    output_path: Optional[Path] = None
    success: bool = False
    skipped: bool = False
    skip_reason: Optional[str] = None
    error: Optional[str] = None
    metadata: Optional[ModelMetadata] = None


@dataclass
class MigrationReport:
    """Full migration report."""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    total_files: int = 0
    legacy_files: int = 0
    versioned_files: int = 0
    migrated_count: int = 0
    skipped_count: int = 0
    error_count: int = 0
    results: List[MigrationResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "total_files": self.total_files,
            "legacy_files": self.legacy_files,
            "versioned_files": self.versioned_files,
            "migrated_count": self.migrated_count,
            "skipped_count": self.skipped_count,
            "error_count": self.error_count,
            "results": [
                {
                    "source": str(r.source_path),
                    "output": str(r.output_path) if r.output_path else None,
                    "success": r.success,
                    "skipped": r.skipped,
                    "skip_reason": r.skip_reason,
                    "error": r.error,
                }
                for r in self.results
            ],
        }


def infer_model_class(path: Path) -> str:
    """Infer model class from filename patterns."""
    name = path.stem.lower()

    # Check patterns in order of specificity
    for pattern, model_class in MODEL_CLASS_PATTERNS.items():
        if pattern in name:
            return model_class

    # Default to RingRiftCNN_v2
    return "RingRiftCNN_v2"


def infer_config_from_state_dict(
    state_dict: Dict[str, torch.Tensor],
    model_class: str,
) -> Dict[str, Any]:
    """Infer model config from state dict tensor shapes."""
    config: Dict[str, Any] = {}

    # Extract num_filters from first conv layer
    if "conv1.weight" in state_dict:
        config["num_filters"] = state_dict["conv1.weight"].shape[0]

    # Count residual blocks
    res_block_count = 0
    for key in state_dict.keys():
        if key.startswith("res_blocks.") and ".conv1.weight" in key:
            idx = int(key.split(".")[1])
            res_block_count = max(res_block_count, idx + 1)
    if res_block_count > 0:
        config["num_res_blocks"] = res_block_count

    # Extract board_size from policy head if possible
    if "policy_conv.weight" in state_dict:
        # Policy conv output channels can hint at policy structure
        pass

    # Infer total_in_channels from first conv
    if "conv1.weight" in state_dict:
        config["total_in_channels"] = state_dict["conv1.weight"].shape[1]

    # Extract global features from value head
    if "value_fc1.weight" in state_dict:
        in_features = state_dict["value_fc1.weight"].shape[1]
        num_filters = config.get("num_filters", 128)
        global_features = in_features - num_filters
        if global_features > 0:
            config["global_features"] = global_features

    return config


def analyze_checkpoint(path: Path, manager: ModelVersionManager) -> CheckpointInfo:
    """Analyze a checkpoint file and return info."""
    info = CheckpointInfo(
        path=path,
        is_legacy=True,
        size_bytes=path.stat().st_size,
        created_at=datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc),
    )

    try:
        # Try to get metadata - if it exists, it's versioned
        metadata = manager.get_metadata(str(path))
        info.is_legacy = False
        info.model_class = metadata.model_class
        info.architecture_version = metadata.architecture_version
    except LegacyCheckpointError:
        # Legacy checkpoint - try to infer model class
        info.is_legacy = True
        info.model_class = infer_model_class(path)
        info.architecture_version = MODEL_VERSIONS.get(info.model_class, "v1.0.0")
    except Exception as e:
        info.error = str(e)

    return info


def migrate_checkpoint(
    source_path: Path,
    output_path: Path,
    model_class: str,
    config: Optional[Dict[str, Any]],
    manager: ModelVersionManager,
    dry_run: bool = False,
) -> MigrationResult:
    """Migrate a single checkpoint."""
    result = MigrationResult(source_path=source_path, output_path=output_path)

    if dry_run:
        result.success = True
        result.skipped = True
        result.skip_reason = "Dry run"
        return result

    try:
        # Load legacy checkpoint
        checkpoint = torch.load(
            str(source_path),
            map_location=torch.device("cpu"),
            weights_only=False,
        )

        # Check if already versioned
        if manager.METADATA_KEY in checkpoint:
            result.skipped = True
            result.skip_reason = "Already versioned"
            return result

        # Extract state dict
        if manager.STATE_DICT_KEY in checkpoint:
            state_dict = checkpoint[manager.STATE_DICT_KEY]
        elif isinstance(checkpoint, dict) and all(
            isinstance(v, torch.Tensor) for v in checkpoint.values()
        ):
            state_dict = checkpoint
        else:
            for key in ["state_dict", "model", "net"]:
                if key in checkpoint:
                    state_dict = checkpoint[key]
                    break
            else:
                state_dict = checkpoint

        # Infer config if not provided
        if config is None:
            config = infer_config_from_state_dict(state_dict, model_class)

        # Determine architecture version
        architecture_version = MODEL_VERSIONS.get(model_class, "v1.0.0")

        # Create metadata
        training_info: Dict[str, Any] = {
            "migrated_from": str(source_path),
            "migration_date": datetime.now(timezone.utc).isoformat(),
            "original_size_bytes": source_path.stat().st_size,
        }

        # Preserve epoch/loss from original if present
        if "epoch" in checkpoint:
            training_info["original_epoch"] = checkpoint["epoch"]
        if "loss" in checkpoint:
            training_info["original_loss"] = checkpoint["loss"]

        metadata = ModelMetadata(
            architecture_version=architecture_version,
            model_class=model_class,
            config=config,
            training_info=training_info,
            created_at=datetime.now(timezone.utc).isoformat(),
            checksum=compute_state_dict_checksum(state_dict),
            parent_checkpoint=str(source_path),
        )

        # Create versioned checkpoint
        versioned_checkpoint: Dict[str, Any] = {
            manager.STATE_DICT_KEY: state_dict,
            manager.METADATA_KEY: metadata.to_dict(),
        }

        # Preserve other fields
        for key in [manager.OPTIMIZER_KEY, manager.SCHEDULER_KEY, manager.EPOCH_KEY, manager.LOSS_KEY]:
            if key in checkpoint:
                versioned_checkpoint[key] = checkpoint[key]

        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save with atomic pattern
        temp_path = output_path.with_suffix(".pth.tmp")
        torch.save(versioned_checkpoint, temp_path)

        # Validate
        test_load = torch.load(temp_path, map_location="cpu", weights_only=False)
        if manager.METADATA_KEY not in test_load:
            temp_path.unlink()
            raise ValueError("Migration validation failed - metadata missing")

        # Atomic rename
        temp_path.rename(output_path)

        result.success = True
        result.metadata = metadata

    except Exception as e:
        result.error = str(e)
        logger.error(f"Failed to migrate {source_path}: {e}")

    return result


def scan_checkpoints(
    input_dir: Path,
    recursive: bool = True,
) -> List[Path]:
    """Scan directory for checkpoint files."""
    pattern = "**/*.pth" if recursive else "*.pth"
    paths = list(input_dir.glob(pattern))

    # Also check for .pt files
    pt_pattern = "**/*.pt" if recursive else "*.pt"
    paths.extend(input_dir.glob(pt_pattern))

    # Filter out temp files
    paths = [p for p in paths if not p.name.endswith(".tmp")]

    return sorted(paths)


def run_migration(
    input_dir: Path,
    output_dir: Optional[Path],
    model_class: Optional[str],
    config: Optional[Dict[str, Any]],
    in_place: bool,
    dry_run: bool,
    recursive: bool,
    skip_versioned: bool,
) -> MigrationReport:
    """Run the full migration process."""
    manager = ModelVersionManager()
    report = MigrationReport()

    # Scan for checkpoints
    checkpoint_paths = scan_checkpoints(input_dir, recursive)
    report.total_files = len(checkpoint_paths)

    logger.info(f"Found {report.total_files} checkpoint files")

    if report.total_files == 0:
        logger.warning(f"No checkpoint files found in {input_dir}")
        return report

    # Analyze each checkpoint
    checkpoint_infos: List[CheckpointInfo] = []
    for path in checkpoint_paths:
        info = analyze_checkpoint(path, manager)
        checkpoint_infos.append(info)

        if info.is_legacy:
            report.legacy_files += 1
        else:
            report.versioned_files += 1

    logger.info(
        f"Analysis: {report.legacy_files} legacy, "
        f"{report.versioned_files} already versioned"
    )

    # Migrate legacy checkpoints
    for info in checkpoint_infos:
        if not info.is_legacy:
            if skip_versioned:
                result = MigrationResult(
                    source_path=info.path,
                    skipped=True,
                    skip_reason="Already versioned",
                )
                report.results.append(result)
                report.skipped_count += 1
                continue

        if info.error:
            result = MigrationResult(
                source_path=info.path,
                error=info.error,
            )
            report.results.append(result)
            report.error_count += 1
            continue

        # Determine output path
        if in_place:
            out_path = info.path
        elif output_dir:
            rel_path = info.path.relative_to(input_dir)
            out_path = output_dir / rel_path
        else:
            # Add .versioned suffix
            out_path = info.path.with_suffix(".versioned.pth")

        # Use provided or inferred model class
        effective_model_class = model_class or info.model_class or "RingRiftCNN_v2"

        # Migrate
        result = migrate_checkpoint(
            source_path=info.path,
            output_path=out_path,
            model_class=effective_model_class,
            config=config,
            manager=manager,
            dry_run=dry_run,
        )
        report.results.append(result)

        if result.success and not result.skipped:
            report.migrated_count += 1
            logger.info(f"Migrated: {info.path} -> {out_path}")
        elif result.skipped:
            report.skipped_count += 1
            logger.debug(f"Skipped: {info.path} ({result.skip_reason})")
        else:
            report.error_count += 1

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Migrate legacy checkpoint files to versioned format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Dry run to preview what would be migrated
    python scripts/migrate_checkpoints.py models/ --dry-run

    # Migrate all legacy checkpoints in place (overwrites original files)
    python scripts/migrate_checkpoints.py models/ --in-place

    # Migrate to a separate output directory
    python scripts/migrate_checkpoints.py models/ --output migrated/

    # Force a specific model class for all files
    python scripts/migrate_checkpoints.py models/ --model-class RingRiftCNN_v3

    # Generate JSON report
    python scripts/migrate_checkpoints.py models/ --report migration_report.json
        """,
    )

    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing checkpoint files",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output directory for migrated checkpoints",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Migrate files in place (overwrite original)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without making changes",
    )
    parser.add_argument(
        "--model-class",
        type=str,
        choices=list(MODEL_VERSIONS.keys()),
        help="Force model class for all files",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="JSON string with model config overrides",
    )
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        default=True,
        help="Scan directories recursively (default: True)",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_false",
        dest="recursive",
        help="Only scan top-level directory",
    )
    parser.add_argument(
        "--skip-versioned",
        action="store_true",
        default=True,
        help="Skip already versioned checkpoints (default: True)",
    )
    parser.add_argument(
        "--force",
        action="store_false",
        dest="skip_versioned",
        help="Re-migrate versioned checkpoints",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Write JSON report to file",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate arguments
    if not args.input_dir.exists():
        parser.error(f"Input directory does not exist: {args.input_dir}")

    if args.in_place and args.output:
        parser.error("Cannot use --in-place with --output")

    if not args.in_place and not args.output and not args.dry_run:
        logger.warning(
            "Neither --in-place nor --output specified. "
            "Files will be saved with .versioned.pth suffix."
        )

    # Parse config if provided
    config = None
    if args.config:
        try:
            config = json.loads(args.config)
        except json.JSONDecodeError as e:
            parser.error(f"Invalid JSON config: {e}")

    # Run migration
    logger.info(f"Starting checkpoint migration from {args.input_dir}")
    if args.dry_run:
        logger.info("DRY RUN - no files will be modified")

    report = run_migration(
        input_dir=args.input_dir,
        output_dir=args.output,
        model_class=args.model_class,
        config=config,
        in_place=args.in_place,
        dry_run=args.dry_run,
        recursive=args.recursive,
        skip_versioned=args.skip_versioned,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("Migration Summary")
    print("=" * 60)
    print(f"Total files scanned:     {report.total_files}")
    print(f"Legacy checkpoints:      {report.legacy_files}")
    print(f"Already versioned:       {report.versioned_files}")
    print(f"Successfully migrated:   {report.migrated_count}")
    print(f"Skipped:                 {report.skipped_count}")
    print(f"Errors:                  {report.error_count}")
    print("=" * 60)

    # Save report if requested
    if args.report:
        with open(args.report, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        logger.info(f"Report saved to {args.report}")

    # Exit with error code if there were failures
    if report.error_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
