#!/usr/bin/env python3
"""Export training data for all neural network architectures.

This script exports training data with the correct encoder for each architecture,
ensuring encoder/model channel count matches.

Architecture requirements:
- V2: 40 channels (10 base × 4 frames) - HexStateEncoder
- V3/V4: 64 channels (16 base × 4 frames) - HexStateEncoderV3 with feature_version=2
- V5-heavy: 56 channels (14 base × 4 frames) - HexStateEncoderV5

Usage:
    python scripts/export_for_all_architectures.py --board-type hex8 --num-players 2
    python scripts/export_for_all_architectures.py --board-type hex8 --num-players 2 --architectures v3,v4
    python scripts/export_for_all_architectures.py --config hex8_2p  # Shorthand

The script will:
1. Find all game databases for the configuration
2. Export separate NPZ files for each architecture with correct encoding
3. Verify channel counts match architecture requirements
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Architecture specifications
ARCHITECTURES = {
    "v2": {
        "channels": 40,
        "encoder_version": "v2",
        "base_channels": 10,
        "frames": 4,
        "description": "V2 standard (10 base × 4 frames)",
    },
    "v3": {
        "channels": 64,
        "encoder_version": "v3",
        "base_channels": 16,
        "frames": 4,
        "description": "V3/V4 enhanced (16 base × 4 frames)",
    },
    "v4": {
        "channels": 64,
        "encoder_version": "v3",  # V4 uses same encoder as V3
        "base_channels": 16,
        "frames": 4,
        "description": "V3/V4 enhanced (16 base × 4 frames)",
    },
    "v5-heavy": {
        "channels": 56,
        "encoder_version": "v3",  # V5-heavy uses V3 encoder with 14 base channels
        "base_channels": 14,
        "frames": 4,
        "description": "V5-heavy with heuristics (14 base × 4 frames)",
        "full_heuristics": True,
    },
}


def run_export(
    board_type: str,
    num_players: int,
    architecture: str,
    output_dir: str,
    use_discovery: bool = True,
    max_samples: int | None = None,
    verbose: bool = False,
) -> tuple[bool, str]:
    """Run export for a specific architecture.

    Returns:
        Tuple of (success, output_path or error_message)
    """
    arch_spec = ARCHITECTURES.get(architecture)
    if not arch_spec:
        return False, f"Unknown architecture: {architecture}"

    config_key = f"{board_type}_{num_players}p"
    output_filename = f"arch_comparison_{config_key}_{architecture.replace('-', '')}.npz"
    output_path = os.path.join(output_dir, output_filename)

    # Build export command
    cmd = [
        sys.executable, "-m", "scripts.export_replay_dataset",
        "--board-type", board_type,
        "--num-players", str(num_players),
        "--encoder-version", arch_spec["encoder_version"],
        "--output", output_path,
    ]

    if use_discovery:
        cmd.append("--use-discovery")

    if max_samples:
        cmd.extend(["--max-samples", str(max_samples)])

    # V5-heavy needs full heuristics
    if arch_spec.get("full_heuristics"):
        cmd.append("--full-heuristics")

    if verbose:
        print(f"[{architecture}] Running: {' '.join(cmd)}")

    # Feb 2026: Best-effort cross-process export coordination
    _config_key = f"{board_type}_{num_players}p"
    _release_slot = False
    try:
        from app.coordination.export_coordinator import get_export_coordinator
        _coord = get_export_coordinator()
        if not _coord.try_acquire(_config_key):
            return False, f"Cross-process export slot unavailable for {_config_key}"
        _release_slot = True
    except Exception:
        pass  # Fail open if coordinator unavailable

    try:
        result = subprocess.run(
            cmd,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )

        if result.returncode != 0:
            return False, f"Export failed: {result.stderr}"

        # Verify the output file exists and has correct channels
        if os.path.exists(output_path):
            return verify_npz_channels(output_path, arch_spec["channels"])
        else:
            return False, f"Output file not created: {output_path}"

    except subprocess.TimeoutExpired:
        return False, "Export timed out after 1 hour"
    except Exception as e:
        return False, f"Export error: {e}"
    finally:
        if _release_slot:
            try:
                _coord.release(_config_key)
            except Exception:
                pass


def verify_npz_channels(npz_path: str, expected_channels: int) -> tuple[bool, str]:
    """Verify NPZ has correct channel count."""
    try:
        import numpy as np

        data = np.load(npz_path, allow_pickle=True)

        # Check for features/boards key
        features_key = None
        for key in ["features", "boards"]:
            if key in data:
                features_key = key
                break

        if features_key is None:
            return False, f"No features/boards key in {npz_path}"

        features = data[features_key]
        actual_channels = features.shape[1] if features.ndim == 4 else features.shape[0]

        if actual_channels != expected_channels:
            return False, (
                f"Channel mismatch: {npz_path} has {actual_channels} channels, "
                f"expected {expected_channels}"
            )

        num_samples = len(features)
        return True, f"{npz_path} ({num_samples:,} samples, {actual_channels} channels)"

    except Exception as e:
        return False, f"Verification error: {e}"


def main():
    parser = argparse.ArgumentParser(
        description="Export training data for all neural network architectures"
    )

    # Configuration options
    parser.add_argument(
        "--config",
        type=str,
        help="Config shorthand (e.g., 'hex8_2p'). Parses board-type and num-players.",
    )
    parser.add_argument(
        "--board-type",
        type=str,
        choices=["hex8", "hexagonal", "square8", "square19"],
        help="Board type",
    )
    parser.add_argument(
        "--num-players",
        type=int,
        choices=[2, 3, 4],
        help="Number of players",
    )

    # Architecture selection
    parser.add_argument(
        "--architectures",
        type=str,
        default="v2,v3,v4,v5-heavy",
        help="Comma-separated list of architectures to export (default: all)",
    )

    # Export options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/training",
        help="Output directory for NPZ files",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum samples per export",
    )
    parser.add_argument(
        "--no-discovery",
        action="store_true",
        help="Don't use game discovery (require explicit --db)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Parse config shorthand
    if args.config:
        parts = args.config.replace("-", "_").split("_")
        if len(parts) >= 2:
            args.board_type = parts[0]
            args.num_players = int(parts[1].replace("p", ""))

    # Validate required args
    if not args.board_type or not args.num_players:
        parser.error("Either --config or both --board-type and --num-players required")

    # Parse architectures
    architectures = [a.strip() for a in args.architectures.split(",")]

    # Validate architectures
    invalid = [a for a in architectures if a not in ARCHITECTURES]
    if invalid:
        parser.error(f"Unknown architectures: {invalid}. Valid: {list(ARCHITECTURES.keys())}")

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Exporting {args.board_type}_{args.num_players}p for architectures: {architectures}")
    print(f"Output directory: {args.output_dir}")
    print()

    # Run exports
    results = {}
    for arch in architectures:
        print(f"[{arch}] Exporting with {ARCHITECTURES[arch]['description']}...")

        success, result = run_export(
            board_type=args.board_type,
            num_players=args.num_players,
            architecture=arch,
            output_dir=args.output_dir,
            use_discovery=not args.no_discovery,
            max_samples=args.max_samples,
            verbose=args.verbose,
        )

        results[arch] = (success, result)

        if success:
            print(f"[{arch}] ✓ {result}")
        else:
            print(f"[{arch}] ✗ {result}")
        print()

    # Summary
    print("=" * 60)
    print("EXPORT SUMMARY")
    print("=" * 60)

    successes = [a for a, (s, _) in results.items() if s]
    failures = [a for a, (s, _) in results.items() if not s]

    print(f"Successful: {len(successes)}/{len(architectures)}")
    if successes:
        for arch in successes:
            print(f"  ✓ {arch}: {results[arch][1]}")

    if failures:
        print(f"\nFailed: {len(failures)}/{len(architectures)}")
        for arch in failures:
            print(f"  ✗ {arch}: {results[arch][1]}")

    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
