#!/usr/bin/env python3
"""Validate NPZ training files for policy encoding issues.

This script checks NPZ training files for common issues:
1. Policy encoding type (legacy_max_n vs board_aware)
2. Policy indices within valid range for board type
3. Feature shape compatibility
4. Data quality metrics

Usage:
    python scripts/validate_npz_encoding.py data/training/square8_2p_v6.npz
    python scripts/validate_npz_encoding.py data/training/*.npz --board-type square8
    python scripts/validate_npz_encoding.py data/training/ --recursive
"""
from __future__ import annotations


import argparse
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np


class ValidationResult(NamedTuple):
    """Result of validating a training data file."""

    path: str
    valid: bool
    policy_encoding: str
    max_policy_index: int
    expected_policy_size: int
    sample_count: int
    issues: list[str]
    warnings: list[str]


# Board-aware policy sizes (must match constants.py)
BOARD_POLICY_SIZES = {
    "square8": 7000,
    "square19": 67000,
    "hex8": 4500,
    "hexagonal": 91876,
}


def validate_npz(path: str, expected_board_type: str | None = None) -> ValidationResult:
    """Validate a single NPZ training data file.

    Args:
        path: Path to the NPZ file
        expected_board_type: Expected board type (for validation)

    Returns:
        ValidationResult with details
    """
    issues = []
    warnings = []

    try:
        data = np.load(path, allow_pickle=True)
    except Exception as e:
        return ValidationResult(
            path=path,
            valid=False,
            policy_encoding="unknown",
            max_policy_index=0,
            expected_policy_size=0,
            sample_count=0,
            issues=[f"Failed to load: {e}"],
            warnings=[],
        )

    # Get metadata
    policy_encoding = "unknown"
    if "policy_encoding" in data:
        try:
            policy_encoding = str(np.asarray(data["policy_encoding"]).item())
        except (ValueError, TypeError, AttributeError):
            pass

    board_type = "unknown"
    if "board_type" in data:
        try:
            board_type = str(np.asarray(data["board_type"]).item())
        except (ValueError, TypeError, AttributeError):
            pass

    # Use expected board type if provided
    if expected_board_type:
        if board_type != "unknown" and board_type != expected_board_type:
            warnings.append(
                f"Board type mismatch: file has {board_type}, expected {expected_board_type}"
            )
        board_type = expected_board_type

    # Get expected policy size
    expected_policy_size = BOARD_POLICY_SIZES.get(board_type, 7000)

    # Get sample count
    sample_count = 0
    if "features" in data:
        sample_count = len(data["features"])
    elif "policy_indices" in data:
        sample_count = len(data["policy_indices"])

    # Compute max policy index
    max_policy_index = 0
    if "policy_indices" in data:
        policy_indices = data["policy_indices"]
        for i in range(len(policy_indices)):
            indices = policy_indices[i]
            if hasattr(indices, "__len__") and len(indices) > 0:
                try:
                    # Handle nested arrays
                    flat = np.asarray(indices).flatten()
                    if len(flat) > 0:
                        local_max = int(np.max(flat))
                        if local_max > max_policy_index:
                            max_policy_index = local_max
                except (ValueError, TypeError, IndexError):
                    pass

    # Check for issues
    if policy_encoding == "legacy_max_n":
        issues.append(
            f"DEPRECATED policy encoding: 'legacy_max_n' produces ~59K indices "
            f"but board-aware encoding for {board_type} expects {expected_policy_size}. "
            f"Regenerate with: python scripts/export_replay_dataset.py ..."
        )

    if max_policy_index >= expected_policy_size:
        issues.append(
            f"Policy index out of range: max={max_policy_index} >= expected={expected_policy_size}. "
            f"This data was likely exported with legacy_max_n encoding."
        )

    if policy_encoding == "unknown":
        warnings.append(
            "Missing policy_encoding metadata. Assuming board-aware encoding."
        )

    # Check feature shape
    if "features" in data:
        feat_shape = data["features"].shape
        if len(feat_shape) < 4:
            issues.append(f"Invalid feature shape: {feat_shape}, expected (N, C, H, W)")
        else:
            h, w = feat_shape[-2], feat_shape[-1]
            if board_type == "square8" and (h != 8 or w != 8):
                issues.append(f"Feature size mismatch: expected 8x8, got {h}x{w}")
            elif board_type == "square19" and (h != 19 or w != 19):
                issues.append(f"Feature size mismatch: expected 19x19, got {h}x{w}")

    # Check for NaN/Inf
    for key in ["features", "values", "globals"]:
        if key in data:
            arr = data[key]
            if np.issubdtype(arr.dtype, np.floating):
                nan_count = np.isnan(arr).sum()
                inf_count = np.isinf(arr).sum()
                if nan_count > 0:
                    issues.append(f"{key} contains {nan_count} NaN values")
                if inf_count > 0:
                    issues.append(f"{key} contains {inf_count} Inf values")

    data.close()

    valid = len(issues) == 0

    return ValidationResult(
        path=path,
        valid=valid,
        policy_encoding=policy_encoding,
        max_policy_index=max_policy_index,
        expected_policy_size=expected_policy_size,
        sample_count=sample_count,
        issues=issues,
        warnings=warnings,
    )


def find_npz_files(path: str, recursive: bool = False) -> list[str]:
    """Find all NPZ files in a path."""
    p = Path(path)
    if p.is_file() and p.suffix == ".npz":
        return [str(p)]
    elif p.is_dir():
        if recursive:
            return [str(f) for f in p.rglob("*.npz")]
        else:
            return [str(f) for f in p.glob("*.npz")]
    else:
        return []


def main():
    parser = argparse.ArgumentParser(
        description="Validate NPZ training data for policy encoding issues"
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="NPZ files or directories to validate",
    )
    parser.add_argument(
        "--board-type",
        choices=list(BOARD_POLICY_SIZES.keys()),
        help="Expected board type for validation",
    )
    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="Recursively search directories",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Only show errors",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with error code if any issues found",
    )

    args = parser.parse_args()

    # Collect all NPZ files
    npz_files = []
    for path in args.paths:
        npz_files.extend(find_npz_files(path, args.recursive))

    if not npz_files:
        print("No NPZ files found")
        return 1

    # Validate each file
    all_valid = True
    results = []

    for npz_path in sorted(npz_files):
        result = validate_npz(npz_path, args.board_type)
        results.append(result)

        if not result.valid:
            all_valid = False

        # Print result
        if not args.quiet or not result.valid:
            status = "✓ VALID" if result.valid else "✗ INVALID"
            print(f"\n{status}: {result.path}")
            print(f"  Samples: {result.sample_count:,}")
            print(f"  Encoding: {result.policy_encoding}")
            print(f"  Max policy index: {result.max_policy_index:,}")
            print(f"  Expected policy size: {result.expected_policy_size:,}")

            for issue in result.issues:
                print(f"  [ERROR] {issue}")
            for warning in result.warnings:
                print(f"  [WARN] {warning}")

    # Summary
    print(f"\n{'='*60}")
    valid_count = sum(1 for r in results if r.valid)
    invalid_count = len(results) - valid_count
    print(f"Summary: {valid_count} valid, {invalid_count} invalid out of {len(results)} files")

    if invalid_count > 0:
        print("\nTo fix invalid files, regenerate with board-aware encoding:")
        print("  PYTHONPATH=. python scripts/export_replay_dataset.py \\")
        print("    --db <canonical_db> --board-type <type> \\")
        print("    --num-players <n> --output <path>.npz")

    if args.strict and not all_valid:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
