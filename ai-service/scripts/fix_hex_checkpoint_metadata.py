#!/usr/bin/env python3
"""Fix hex checkpoint metadata to use correct model_class.

Some hex checkpoints were saved with incorrect model_class metadata
(e.g., "RingRiftCNN_v2" instead of "HexNeuralNet_v2"). This causes
architecture mismatches when loading (value_fc1 in_features: 21 vs 212).

This script fixes the metadata and re-saves the checkpoints.

Usage:
    python scripts/fix_hex_checkpoint_metadata.py [--dry-run]
"""

import argparse
import os
import sys
from pathlib import Path

# Add ai-service to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from app.utils.torch_utils import safe_load_checkpoint

# Model class mapping for hex boards based on architecture version
HEX_MODEL_CLASSES = {
    "v2.0.0": "HexNeuralNet_v2",
    "v3.0.0": "HexNeuralNet_v3",
    "v4.0.0": "HexNeuralNet_v4",
}

SQUARE_MODEL_CLASSES = {"RingRiftCNN_v2", "RingRiftCNN_v2_Lite", "RingRiftCNN_v3", "RingRiftCNN_v3_Lite", "RingRiftCNN_v4"}


def infer_hex_model_class(checkpoint: dict) -> str | None:
    """Infer the correct hex model class from checkpoint weights.

    Returns the correct model class name or None if not determinable.
    """
    state_dict = checkpoint.get("model_state_dict", {})
    meta = checkpoint.get("_versioning_metadata", {})

    # Check value_fc1 in_features to determine architecture
    value_fc1 = state_dict.get("value_fc1.weight")
    if value_fc1 is None:
        return None

    in_features = value_fc1.shape[1]

    # HexNeuralNet_v2/v3: value_fc1 = Linear(1 + global_features, ...) = 21
    # HexNeuralNet_v4: value_fc1 = Linear(num_filters + global_features, ...) = 148
    # RingRiftCNN: value_fc1 = Linear(num_filters + global_features, ...) = 212

    if in_features == 21:
        # Check architecture version from metadata
        arch_version = meta.get("architecture_version", "v2.0.0")
        if arch_version.startswith("v3"):
            return "HexNeuralNet_v3"
        return "HexNeuralNet_v2"
    elif in_features == 148:
        return "HexNeuralNet_v4"

    return None


def fix_checkpoint(checkpoint_path: Path, dry_run: bool = False) -> bool:
    """Fix a single checkpoint's metadata.

    Fixes:
    1. model_class: RingRiftCNN_v2 -> HexNeuralNet_v2 for hex boards
    2. in_channels: Add if missing (inferred from conv1.weight)

    Returns True if the checkpoint was fixed, False if no fix needed.
    """
    print(f"\nProcessing: {checkpoint_path.name}")

    # Load checkpoint using safe loader
    try:
        checkpoint = safe_load_checkpoint(checkpoint_path, map_location="cpu", warn_on_unsafe=False)
    except Exception as e:
        print(f"  ERROR: Could not load checkpoint: {e}")
        return False

    meta = checkpoint.get("_versioning_metadata", {})
    current_class = meta.get("model_class")
    config = meta.get("config", {})
    board_type = config.get("board_type", "")
    state_dict = checkpoint.get("model_state_dict", {})

    # Check if this is a hex checkpoint
    is_hex = "hex" in board_type.lower()
    if not is_hex:
        print(f"  Skipping: not a hex checkpoint (board_type={board_type})")
        return False

    needs_fix = False

    # Fix 1: Check if model_class needs fixing
    if current_class in SQUARE_MODEL_CLASSES:
        correct_class = infer_hex_model_class(checkpoint)
        if correct_class is None:
            print(f"  ERROR: Could not infer correct model class")
            return False
        print(f"  Fix model_class: {current_class} -> {correct_class}")
        if not dry_run:
            meta["model_class"] = correct_class
        needs_fix = True
    else:
        print(f"  model_class={current_class} OK")

    # Fix 2: Add in_channels to config if missing
    conv1 = state_dict.get("conv1.weight")
    if conv1 is not None and hasattr(conv1, "shape"):
        inferred_in_channels = int(conv1.shape[1])
        if config.get("in_channels") is None:
            print(f"  Fix in_channels: None -> {inferred_in_channels}")
            if not dry_run:
                config["in_channels"] = inferred_in_channels
                meta["config"] = config
            needs_fix = True
        elif config.get("in_channels") != inferred_in_channels:
            print(f"  Fix in_channels: {config.get('in_channels')} -> {inferred_in_channels}")
            if not dry_run:
                config["in_channels"] = inferred_in_channels
                meta["config"] = config
            needs_fix = True
        else:
            print(f"  in_channels={inferred_in_channels} OK")

    if not needs_fix:
        print(f"  No fixes needed")
        return False

    if dry_run:
        print(f"  [DRY RUN] Would update checkpoint")
        return True

    # Update metadata
    checkpoint["_versioning_metadata"] = meta

    # Save updated checkpoint
    torch.save(checkpoint, checkpoint_path)
    print(f"  Updated and saved: {checkpoint_path}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Fix hex checkpoint metadata")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    parser.add_argument("--models-dir", type=str, default="models", help="Directory containing models")
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    if not models_dir.exists():
        print(f"ERROR: Models directory not found: {models_dir}")
        sys.exit(1)

    # Find hex checkpoints
    hex_patterns = ["canonical_hex*.pth", "hex*_v3.pth", "hex*_cluster.pth"]
    checkpoints = []
    for pattern in hex_patterns:
        checkpoints.extend(models_dir.glob(pattern))

    # Remove duplicates and symlinks
    checkpoints = [p for p in set(checkpoints) if p.is_file() and not p.is_symlink()]
    checkpoints = sorted(checkpoints)

    if not checkpoints:
        print("No hex checkpoints found to fix")
        sys.exit(0)

    print(f"Found {len(checkpoints)} hex checkpoint(s) to check")

    fixed_count = 0
    for ckpt in checkpoints:
        if fix_checkpoint(ckpt, args.dry_run):
            fixed_count += 1

    print(f"\n{'Would fix' if args.dry_run else 'Fixed'} {fixed_count} checkpoint(s)")


if __name__ == "__main__":
    main()
