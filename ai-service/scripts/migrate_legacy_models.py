#!/usr/bin/env python3
"""Migrate legacy model checkpoints to the new registry format.

This script scans the models directory for checkpoints not yet registered
in the model registry and adds them with inferred metadata.

Usage:
    # Dry run (show what would be migrated)
    python scripts/migrate_legacy_models.py --dry-run

    # Execute migration
    python scripts/migrate_legacy_models.py --execute

    # Migrate specific model
    python scripts/migrate_legacy_models.py --model models/my_model.pt --execute
"""

import argparse
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

# Add ai-service to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class InferredConfig:
    """Configuration inferred from model checkpoint."""
    board_type: str = "unknown"
    num_players: int = 2
    model_version: str = "v1"
    architecture: str = "unknown"
    in_channels: int = 0
    policy_size: int = 0


def infer_config_from_checkpoint(path: Path) -> InferredConfig:
    """Infer model configuration from checkpoint state dict."""
    config = InferredConfig()

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        logger.debug(f"Could not load {path}: {e}")
        return config

    # Get state dict
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Check for config in checkpoint
    if "config" in checkpoint:
        ckpt_config = checkpoint["config"]
        if isinstance(ckpt_config, dict):
            config.board_type = ckpt_config.get("board_type", config.board_type)
            config.num_players = ckpt_config.get("num_players", config.num_players)
            config.model_version = ckpt_config.get("model_version", config.model_version)

    # Infer from state dict keys
    keys = list(state_dict.keys()) if isinstance(state_dict, dict) else []

    # Check for hex-specific keys
    if any("hex" in k.lower() for k in keys):
        config.architecture = "hex"
        if "hex_mask" in keys:
            config.architecture = "HexNeuralNet"

    # Check input channels from first conv layer
    for key in keys:
        if "conv1.weight" in key or "stem.0.weight" in key:
            weight = state_dict[key]
            if hasattr(weight, "shape") and len(weight.shape) >= 2:
                config.in_channels = weight.shape[1]
                # Infer version from channels
                if config.in_channels == 40:
                    config.model_version = "v2"
                elif config.in_channels == 64:
                    config.model_version = "v3"
                elif config.in_channels == 56:
                    config.model_version = "v2"  # Square board
            break

    # Check policy head size
    for key in keys:
        if "policy_head" in key and "weight" in key:
            weight = state_dict[key]
            if hasattr(weight, "shape"):
                config.policy_size = weight.shape[0]
            break

    # Infer board type from filename
    name = path.stem.lower()
    if "hex8" in name or "hex_" in name:
        config.board_type = "hex8"
    elif "hexagonal" in name:
        config.board_type = "hexagonal"
    elif "square19" in name or "sq19" in name:
        config.board_type = "square19"
    elif "square8" in name or "sq8" in name:
        config.board_type = "square8"

    # Infer num_players from filename
    match = re.search(r"(\d)p", name)
    if match:
        config.num_players = int(match.group(1))

    return config


def migrate_model(
    path: Path,
    registry,
    dry_run: bool = True,
) -> bool:
    """Migrate a single model to the registry.

    Returns True if migrated, False if skipped.
    """
    # Check if already registered
    try:
        existing = registry.get_model_by_path(str(path))
        if existing:
            logger.debug(f"Already registered: {path.name}")
            return False
    except Exception:
        pass

    # Infer configuration
    config = infer_config_from_checkpoint(path)

    # Generate model ID from filename
    model_id = path.stem
    # Clean up common patterns
    model_id = re.sub(r"_\d{8}_\d{6}", "", model_id)  # Remove timestamps
    model_id = re.sub(r"\.v\d+", "", model_id)  # Remove version suffixes

    if dry_run:
        logger.info(f"[DRY RUN] Would migrate: {path.name}")
        logger.info(f"  Model ID: {model_id}")
        logger.info(f"  Board: {config.board_type}, Players: {config.num_players}")
        logger.info(f"  Version: {config.model_version}, Channels: {config.in_channels}")
        return True

    # Register in registry
    try:
        from app.training.model_registry import ModelStage, ModelType

        version = registry.register_model(
            model_id=model_id,
            model_type=ModelType.CNN,
            checkpoint_path=str(path),
            config={
                "board_type": config.board_type,
                "num_players": config.num_players,
                "model_version": config.model_version,
                "in_channels": config.in_channels,
                "policy_size": config.policy_size,
            },
            stage=ModelStage.DEVELOPMENT,
            metrics={},
        )
        logger.info(f"Migrated: {path.name} -> {model_id}:v{version}")
        return True

    except Exception as e:
        logger.error(f"Failed to migrate {path.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Migrate legacy models to registry")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path(__file__).parent.parent / "models",
        help="Models directory to scan",
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="Migrate a specific model file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without executing",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute migration",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of models to migrate (0 = all)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load registry
    try:
        from app.training.model_registry import ModelRegistry
        registry_dir = args.models_dir.parent / "data" / "model_registry"
        registry = ModelRegistry(registry_dir)
    except Exception as e:
        logger.error(f"Could not load model registry: {e}")
        return 1

    dry_run = not args.execute

    if args.model:
        # Migrate single model
        if not args.model.exists():
            logger.error(f"Model not found: {args.model}")
            return 1
        migrated = migrate_model(args.model, registry, dry_run=dry_run)
        return 0 if migrated else 1

    # Scan models directory
    logger.info(f"Scanning {args.models_dir}...")

    patterns = ["*.pt", "*.pth", "*.ckpt"]
    model_files = []
    for pattern in patterns:
        model_files.extend(args.models_dir.glob(pattern))
        # Also scan subdirectories
        for subdir in args.models_dir.iterdir():
            if subdir.is_dir() and subdir.name not in ["archive", "__pycache__"]:
                model_files.extend(subdir.glob(pattern))

    logger.info(f"Found {len(model_files)} model files")

    # Migrate models
    migrated = 0
    skipped = 0
    errors = 0

    for i, path in enumerate(model_files):
        if args.limit > 0 and migrated >= args.limit:
            logger.info(f"Reached limit of {args.limit} migrations")
            break

        try:
            if migrate_model(path, registry, dry_run=dry_run):
                migrated += 1
            else:
                skipped += 1
        except Exception as e:
            logger.error(f"Error migrating {path.name}: {e}")
            errors += 1

    logger.info(f"\nMigration {'preview' if dry_run else 'complete'}:")
    logger.info(f"  Migrated: {migrated}")
    logger.info(f"  Skipped (already registered): {skipped}")
    logger.info(f"  Errors: {errors}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
