#!/usr/bin/env python3
"""Merge weights from multiple CMA-ES training runs with the same player count.

This script combines optimized weights from multiple training runs into unified
profiles. Weights can ONLY be merged if they come from runs with the same
number of players, since different player counts require different strategies.

Usage:
    # Merge weights from multiple 2-player runs
    python scripts/merge_trained_weights.py \
        --input logs/cmaes/phase2_square8_2p/runs/*/best_weights.json \
        --output logs/cmaes/merged/2p_weights.json

    # Merge with fitness-weighted averaging
    python scripts/merge_trained_weights.py \
        --input logs/cmaes/run1/best_weights.json \
        --input logs/cmaes/run2/best_weights.json \
        --mode fitness-weighted \
        --output logs/cmaes/merged/weighted_ensemble.json

    # Create player-count-specific profiles from separate directories
    python scripts/merge_trained_weights.py \
        --input-2p logs/cmaes/phase2_square8_2p/runs/*/best_weights.json \
        --input-3p logs/cmaes/phase2_square8_3p/runs/*/best_weights.json \
        --input-4p logs/cmaes/phase2_square8_4p/runs/*/best_weights.json \
        --output logs/cmaes/merged/player_specific_profiles.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ai.heuristic_weights import BASE_V1_BALANCED_WEIGHTS


HeuristicWeights = dict[str, float]


def get_num_players_from_path(path: str) -> int | None:
    """Try to determine number of players from file path or run metadata.

    Looks for:
    1. run_meta.json in the same directory
    2. Pattern like '_2p', '_3p', '_4p' in the path
    """
    # Try to read run_meta.json
    run_dir = os.path.dirname(path)
    meta_path = os.path.join(run_dir, "run_meta.json")

    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                if "num_players" in meta:
                    return meta["num_players"]
        except Exception:
            pass

    # Try parent directory's run_meta.json
    parent_dir = os.path.dirname(run_dir)
    parent_meta_path = os.path.join(parent_dir, "run_meta.json")

    if os.path.exists(parent_meta_path):
        try:
            with open(parent_meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                if "num_players" in meta:
                    return meta["num_players"]
        except Exception:
            pass

    # Try to infer from path pattern
    path_lower = path.lower()
    if "_2p" in path_lower or "/2p/" in path_lower or "_2p/" in path_lower:
        return 2
    elif "_3p" in path_lower or "/3p/" in path_lower or "_3p/" in path_lower:
        return 3
    elif "_4p" in path_lower or "/4p/" in path_lower or "_4p/" in path_lower:
        return 4

    return None


def load_weights_file(path: str) -> tuple[HeuristicWeights, float, int, dict]:
    """Load weights from a best_weights.json or checkpoint file.

    Returns:
        Tuple of (weights dict, fitness score, num_players, full metadata)
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    weights = data.get("weights", data)
    fitness = data.get("fitness", 0.5)

    # Get num_players from data or infer from path
    num_players = data.get("num_players")
    if num_players is None:
        num_players = get_num_players_from_path(path)
    if num_players is None:
        num_players = 2  # Default assumption

    return weights, fitness, num_players, data


def expand_glob_patterns(patterns: list[str]) -> list[str]:
    """Expand glob patterns to actual file paths."""
    paths = []
    for pattern in patterns:
        expanded = glob.glob(pattern)
        if not expanded:
            print(f"Warning: No files match pattern '{pattern}'")
        paths.extend(expanded)
    return sorted(set(paths))


def merge_weights_average(
    weight_files: list[str],
    mode: str = "equal",
    expected_num_players: int | None = None,
) -> tuple[HeuristicWeights, int, dict]:
    """Merge multiple weight files using averaging.

    IMPORTANT: All weight files must be from runs with the same number of players.
    This function will raise an error if player counts don't match.

    Args:
        weight_files: List of paths to weight JSON files
        mode: "equal" for simple average, "fitness-weighted" for weighted by fitness
        expected_num_players: If provided, all files must match this player count

    Returns:
        Tuple of (merged weights, num_players, metadata about merge)

    Raises:
        ValueError: If weight files have different player counts
    """
    if not weight_files:
        raise ValueError("No weight files provided")

    all_weights: list[tuple[HeuristicWeights, float, int, str]] = []

    for path in weight_files:
        try:
            weights, fitness, num_players, _ = load_weights_file(path)
            all_weights.append((weights, fitness, num_players, path))
            print(f"  Loaded: {path} (fitness={fitness:.4f}, players={num_players})")
        except Exception as e:
            print(f"  Warning: Failed to load {path}: {e}")

    if not all_weights:
        raise ValueError("No valid weight files loaded")

    # Validate all files have same player count
    player_counts = set(np for _, _, np, _ in all_weights)

    if len(player_counts) > 1:
        raise ValueError(
            f"Cannot merge weights from runs with different player counts: {player_counts}. "
            f"Weights must come from runs with the same number of players. "
            f"Use --input-2p, --input-3p, --input-4p to specify separate runs for each player count."
        )

    detected_num_players = player_counts.pop()

    if expected_num_players is not None and detected_num_players != expected_num_players:
        raise ValueError(
            f"Expected {expected_num_players}-player weights but found {detected_num_players}-player weights"
        )

    # Calculate weights for averaging
    if mode == "fitness-weighted":
        total_fitness = sum(f for _, f, _, _ in all_weights)
        if total_fitness <= 0:
            # Fallback to equal weighting
            coefficients = [1.0 / len(all_weights)] * len(all_weights)
        else:
            coefficients = [f / total_fitness for _, f, _, _ in all_weights]
    else:  # equal
        coefficients = [1.0 / len(all_weights)] * len(all_weights)

    # Merge weights
    merged: HeuristicWeights = {}

    # Get all keys from all weight files
    all_keys = set()
    for weights, _, _, _ in all_weights:
        all_keys.update(weights.keys())

    for key in all_keys:
        weighted_sum = 0.0
        for (weights, _, _, _), coef in zip(all_weights, coefficients):
            value = weights.get(key, BASE_V1_BALANCED_WEIGHTS.get(key, 0.0))
            weighted_sum += value * coef
        merged[key] = weighted_sum

    metadata = {
        "merge_mode": mode,
        "num_players": detected_num_players,
        "source_files": [path for _, _, _, path in all_weights],
        "source_fitness": [f for _, f, _, _ in all_weights],
        "coefficients": coefficients,
        "timestamp": datetime.now().isoformat(),
    }

    return merged, detected_num_players, metadata


def create_player_specific_profiles(
    weights_2p: list[str] | None,
    weights_3p: list[str] | None,
    weights_4p: list[str] | None,
) -> tuple[dict[str, HeuristicWeights], dict]:
    """Create player-count-specific weight profiles.

    Each player count's weights are merged separately, ensuring no cross-
    contamination between different game modes.

    Args:
        weights_2p: List of weight files from 2-player training
        weights_3p: List of weight files from 3-player training
        weights_4p: List of weight files from 4-player training

    Returns:
        Tuple of (profiles dict, metadata)
    """
    profiles: dict[str, HeuristicWeights] = {}
    metadata: dict = {
        "timestamp": datetime.now().isoformat(),
        "profiles": {},
    }

    if weights_2p:
        print("\nMerging 2-player weights:")
        files = expand_glob_patterns(weights_2p)
        if files:
            merged, num_players, meta = merge_weights_average(files, mode="fitness-weighted", expected_num_players=2)
            profiles["heuristic_v1_2p"] = merged
            metadata["profiles"]["heuristic_v1_2p"] = meta

    if weights_3p:
        print("\nMerging 3-player weights:")
        files = expand_glob_patterns(weights_3p)
        if files:
            merged, num_players, meta = merge_weights_average(files, mode="fitness-weighted", expected_num_players=3)
            profiles["heuristic_v1_3p"] = merged
            metadata["profiles"]["heuristic_v1_3p"] = meta

    if weights_4p:
        print("\nMerging 4-player weights:")
        files = expand_glob_patterns(weights_4p)
        if files:
            merged, num_players, meta = merge_weights_average(files, mode="fitness-weighted", expected_num_players=4)
            profiles["heuristic_v1_4p"] = merged
            metadata["profiles"]["heuristic_v1_4p"] = meta

    return profiles, metadata


def main():
    parser = argparse.ArgumentParser(
        description="Merge weights from multiple CMA-ES training runs (same player count only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--input",
        "-i",
        action="append",
        dest="inputs",
        help="Input weight file(s) or glob pattern. All files must be from runs " "with the same number of players.",
    )
    parser.add_argument(
        "--input-2p",
        action="append",
        dest="inputs_2p",
        help="Input weight file(s) from 2-player training",
    )
    parser.add_argument(
        "--input-3p",
        action="append",
        dest="inputs_3p",
        help="Input weight file(s) from 3-player training",
    )
    parser.add_argument(
        "--input-4p",
        action="append",
        dest="inputs_4p",
        help="Input weight file(s) from 4-player training",
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=["equal", "fitness-weighted"],
        default="fitness-weighted",
        help="Merge mode (default: fitness-weighted)",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output path for merged weights JSON",
    )

    args = parser.parse_args()

    # Validate inputs
    has_general = args.inputs is not None
    has_player_specific = any([args.inputs_2p, args.inputs_3p, args.inputs_4p])

    if not has_general and not has_player_specific:
        parser.error("Must provide either --input or --input-2p/3p/4p")

    if has_general and has_player_specific:
        parser.error(
            "Cannot mix --input with --input-2p/3p/4p. "
            "Use --input for same-player-count runs, or "
            "--input-Np for player-specific profiles."
        )

    # Create output directory
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    if has_player_specific:
        # Player-specific mode - each player count merged separately
        profiles, metadata = create_player_specific_profiles(
            args.inputs_2p,
            args.inputs_3p,
            args.inputs_4p,
        )

        if not profiles:
            print("Error: No profiles created")
            sys.exit(1)

        output_data = {
            "profiles": {k: dict(v) for k, v in profiles.items()},
            "metadata": metadata,
        }

        print(f"\nCreated {len(profiles)} player-specific profiles:")
        for name, weights in profiles.items():
            print(f"  - {name}")
    else:
        # General merge mode - all files must have same player count
        files = expand_glob_patterns(args.inputs)
        if not files:
            print("Error: No input files found")
            sys.exit(1)

        print(f"Merging {len(files)} weight files (mode={args.mode}):")

        try:
            merged, num_players, metadata = merge_weights_average(files, mode=args.mode)
        except ValueError as e:
            print(f"\nError: {e}")
            sys.exit(1)

        output_data = {
            "weights": merged,
            "num_players": num_players,
            "fitness": sum(metadata["source_fitness"]) / len(metadata["source_fitness"]),
            "metadata": metadata,
        }

        print(f"\nMerged {len(files)} files for {num_players}-player games")

    # Write output
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nOutput written to: {args.output}")

    # Print usage hint
    print("\nTo load these profiles at runtime:")
    print(f"  export RINGRIFT_TRAINED_HEURISTIC_PROFILES={os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
