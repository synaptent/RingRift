#!/usr/bin/env python3
"""Post-training pipeline for validating and promoting heuristic weights.

This script automates the end-to-end process of:
1. Detecting completed training runs
2. Merging weights from multiple runs (if applicable)
3. Validating merged weights against current defaults
4. Promoting weights only if they demonstrate improvement

Designed to be called after training runs complete, either manually or via CI/CD.

Usage:
    # Process completed training runs in a directory
    python scripts/post_training_pipeline.py \
        --training-dir logs/cmaes/phase2_square8_2p \
        --num-players 2 \
        --validation-games 100

    # Process multiple player-count runs
    python scripts/post_training_pipeline.py \
        --training-dir-2p logs/cmaes/phase2_square8_2p \
        --training-dir-3p logs/cmaes/phase2_square8_3p \
        --training-dir-4p logs/cmaes/phase2_square8_4p \
        --validation-games 50

    # Dry run (validate but don't promote)
    python scripts/post_training_pipeline.py \
        --training-dir logs/cmaes/latest \
        --dry-run

    # Skip validation (just merge and promote)
    python scripts/post_training_pipeline.py \
        --training-dir logs/cmaes/validated_run \
        --skip-validation
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class TrainingRun:
    """Information about a completed training run."""

    run_dir: str
    best_weights_path: str
    fitness: float
    generation: int
    num_players: int
    timestamp: str


@dataclass
class PipelineResult:
    """Result of the post-training pipeline."""

    training_runs: list[TrainingRun]
    merged_weights_path: str | None
    validation_passed: bool
    promoted: bool
    promoted_path: str | None
    report: dict


def find_training_runs(training_dir: str) -> list[TrainingRun]:
    """Find all completed training runs in a directory."""
    runs = []

    # Look for best_weights.json files
    pattern = os.path.join(training_dir, "runs", "*", "best_weights.json")
    weight_files = glob.glob(pattern)

    if not weight_files:
        # Also try direct path
        direct_path = os.path.join(training_dir, "best_weights.json")
        if os.path.exists(direct_path):
            weight_files = [direct_path]

    for path in weight_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Try to determine num_players from run metadata
            run_dir = os.path.dirname(path)
            meta_path = os.path.join(run_dir, "run_meta.json")
            num_players = 2  # Default

            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    num_players = meta.get("num_players", 2)

            run = TrainingRun(
                run_dir=run_dir,
                best_weights_path=path,
                fitness=data.get("fitness", 0.5),
                generation=data.get("generation", 0),
                num_players=num_players,
                timestamp=data.get("timestamp", ""),
            )
            runs.append(run)
            print(f"  Found run: {path} (fitness={run.fitness:.4f})")

        except Exception as e:
            print(f"  Warning: Failed to load {path}: {e}")

    return sorted(runs, key=lambda r: r.fitness, reverse=True)


def merge_weights(
    runs: list[TrainingRun],
    output_path: str,
    num_players: int,
) -> str:
    """Merge weights from multiple runs using fitness-weighted averaging.

    All runs must have the same number of players - this is enforced by
    the merge_weights_average function.
    """
    from scripts.merge_trained_weights import merge_weights_average

    weight_files = [r.best_weights_path for r in runs]
    print(f"\nMerging {len(weight_files)} weight files for {num_players}-player games...")

    # The merge function validates all files have same player count
    merged, detected_players, metadata = merge_weights_average(
        weight_files,
        mode="fitness-weighted",
        expected_num_players=num_players,
    )

    # Create output structure
    output_data = {
        "weights": merged,
        "fitness": sum(r.fitness for r in runs) / len(runs),  # Average fitness
        "num_players": detected_players,
        "metadata": {
            **metadata,
            "pipeline_timestamp": datetime.now().isoformat(),
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"  Merged weights saved to: {output_path}")
    return output_path


def validate_weights(
    weights_path: str,
    num_players: int,
    num_games: int,
    min_win_rate: float = 0.52,
) -> tuple[bool, dict]:
    """Validate weights against defaults using the validation script."""
    print(f"\nValidating weights ({num_games} games, {num_players} players)...")

    # Import validation function
    from scripts.validate_and_promote_weights import validate_weights as do_validate

    from app.models import BoardType

    result = do_validate(
        weights_path,
        num_games,
        num_players,
        min_win_rate=min_win_rate,
        confidence=0.95,
        board_type=BoardType.SQUARE8,
    )

    passed = result.recommendation == "PROMOTE"
    return passed, result.to_dict()


def promote_weights(
    weights_path: str,
    num_players: int,
    output_path: str,
) -> str:
    """Promote validated weights to the trained profiles file."""
    from scripts.validate_and_promote_weights import promote_weights as do_promote

    profile_id = f"heuristic_v1_{num_players}p"
    return do_promote(weights_path, profile_id, output_path)


def run_pipeline(
    training_dirs: dict[int, str],
    validation_games: int = 50,
    min_win_rate: float = 0.52,
    output_dir: str = "data",
    dry_run: bool = False,
    skip_validation: bool = False,
) -> PipelineResult:
    """Run the complete post-training pipeline.

    Args:
        training_dirs: Map of num_players -> training directory
        validation_games: Number of games for validation
        min_win_rate: Minimum win rate to pass validation
        output_dir: Directory for output files
        dry_run: If True, don't actually promote weights
        skip_validation: If True, skip validation step

    Returns:
        PipelineResult with all information
    """
    print("=" * 60)
    print("POST-TRAINING PIPELINE")
    print("=" * 60)

    all_runs: list[TrainingRun] = []
    validation_results: dict[int, dict] = {}
    merged_paths: dict[int, str] = {}
    all_passed = True

    for num_players, training_dir in sorted(training_dirs.items()):
        print(f"\n--- Processing {num_players}-player training ---")
        print(f"Directory: {training_dir}")

        # Find training runs
        runs = find_training_runs(training_dir)
        if not runs:
            print(f"  No completed runs found in {training_dir}")
            continue

        all_runs.extend(runs)

        # Merge weights if multiple runs
        merged_path = os.path.join(output_dir, "pipeline", f"merged_{num_players}p_weights.json")

        if len(runs) > 1:
            merged_path = merge_weights(runs, merged_path, num_players)
        else:
            # Just copy the single run's weights
            merged_path = runs[0].best_weights_path

        merged_paths[num_players] = merged_path

        # Validate
        if not skip_validation:
            passed, result = validate_weights(
                merged_path,
                num_players,
                validation_games,
                min_win_rate,
            )
            validation_results[num_players] = result

            if not passed:
                all_passed = False
                print(f"  FAILED: {num_players}p weights did not pass validation")
        else:
            print(f"  Skipping validation (--skip-validation)")

    # Promote if all passed and not dry run
    promoted = False
    promoted_path = None

    if all_passed and not dry_run and merged_paths:
        print("\n--- Promoting validated weights ---")
        promoted_path = os.path.join(output_dir, "trained_heuristic_profiles.json")

        for num_players, weights_path in merged_paths.items():
            promote_weights(weights_path, num_players, promoted_path)

        promoted = True
        print(f"\nWeights promoted to: {promoted_path}")
        print("To activate, set environment variable:")
        print(f"  export RINGRIFT_TRAINED_HEURISTIC_PROFILES={os.path.abspath(promoted_path)}")

    elif dry_run:
        print("\n--- Dry run - weights NOT promoted ---")

    # Generate report
    report = {
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "validation_games": validation_games,
            "min_win_rate": min_win_rate,
            "dry_run": dry_run,
            "skip_validation": skip_validation,
        },
        "training_runs": [
            {
                "run_dir": r.run_dir,
                "fitness": r.fitness,
                "generation": r.generation,
                "num_players": r.num_players,
            }
            for r in all_runs
        ],
        "validation_results": validation_results,
        "merged_weights": merged_paths,
        "promoted": promoted,
        "promoted_path": promoted_path,
    }

    # Save report
    report_path = os.path.join(output_dir, "pipeline", "pipeline_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nPipeline report saved to: {report_path}")

    return PipelineResult(
        training_runs=all_runs,
        merged_weights_path=list(merged_paths.values())[0] if merged_paths else None,
        validation_passed=all_passed,
        promoted=promoted,
        promoted_path=promoted_path,
        report=report,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Post-training pipeline for validating and promoting weights",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--training-dir",
        "-t",
        help="Training directory to process (general, uses --num-players)",
    )
    parser.add_argument(
        "--training-dir-2p",
        help="Training directory for 2-player runs",
    )
    parser.add_argument(
        "--training-dir-3p",
        help="Training directory for 3-player runs",
    )
    parser.add_argument(
        "--training-dir-4p",
        help="Training directory for 4-player runs",
    )
    parser.add_argument(
        "--num-players",
        "-n",
        type=int,
        default=2,
        choices=[2, 3, 4],
        help="Number of players (for --training-dir)",
    )
    parser.add_argument(
        "--validation-games",
        "-g",
        type=int,
        default=50,
        help="Number of games for validation (default: 50)",
    )
    parser.add_argument(
        "--min-win-rate",
        type=float,
        default=0.52,
        help="Minimum win rate for promotion (default: 0.52)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="data",
        help="Output directory for results (default: data)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate but don't promote weights",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation step (just merge and promote)",
    )

    args = parser.parse_args()

    # Build training_dirs map
    training_dirs: dict[int, str] = {}

    if args.training_dir:
        training_dirs[args.num_players] = args.training_dir

    if args.training_dir_2p:
        training_dirs[2] = args.training_dir_2p
    if args.training_dir_3p:
        training_dirs[3] = args.training_dir_3p
    if args.training_dir_4p:
        training_dirs[4] = args.training_dir_4p

    if not training_dirs:
        parser.error("Must provide at least one --training-dir")

    # Run pipeline
    result = run_pipeline(
        training_dirs,
        validation_games=args.validation_games,
        min_win_rate=args.min_win_rate,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        skip_validation=args.skip_validation,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    print(f"  Training runs found: {len(result.training_runs)}")
    print(f"  Validation passed: {result.validation_passed}")
    print(f"  Weights promoted: {result.promoted}")

    if result.promoted_path:
        print(f"\n  To use promoted weights, set:")
        print(f"    export RINGRIFT_TRAINED_HEURISTIC_PROFILES={os.path.abspath(result.promoted_path)}")

    # Exit code for CI/CD
    if result.promoted:
        sys.exit(0)
    elif result.validation_passed and args.dry_run:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
