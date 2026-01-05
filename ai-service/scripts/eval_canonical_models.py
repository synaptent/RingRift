#!/usr/bin/env python3
"""Evaluate all canonical models and update Elo database.

January 4, 2026 (Session 17.21):
- Removed architecture filters (_v2, _v5) to evaluate ALL model architectures
- Added --architecture flag to filter by specific architecture if needed
- Added --all flag to include non-canonical models

For automated evaluation, use the daemon infrastructure instead:
- UNEVALUATED_MODEL_SCANNER: Scans for models without Elo ratings
- STALE_EVALUATION: Re-evaluates models with ratings >30 days old
- EVALUATION: Processes the evaluation queue
"""

import argparse
import re
import sys
from pathlib import Path

# Add ai-service to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.training.game_gauntlet import run_baseline_gauntlet, BaselineOpponent


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate canonical models")
    parser.add_argument(
        "--architecture", "-a",
        type=str,
        help="Filter to specific architecture (e.g., v2, v4, v5-heavy)"
    )
    parser.add_argument(
        "--all", "-A",
        action="store_true",
        help="Include non-canonical models (ringrift_best_*.pth, etc.)"
    )
    parser.add_argument(
        "--games", "-g",
        type=int,
        default=20,
        help="Games per opponent (default: 20)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List models without evaluating"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    models_dir = Path("models")

    # Get models based on flags
    if args.all:
        # Include all model files
        patterns = ["canonical_*.pth", "ringrift_best_*.pth", "*_trained_*.pth"]
        all_models = []
        for pattern in patterns:
            all_models.extend(models_dir.glob(pattern))
        canonical_models = sorted(set(all_models))
    else:
        # Canonical models only (but ALL architectures - no v2/v5 filter)
        canonical_models = sorted(models_dir.glob("canonical_*.pth"))

    # Filter by architecture if specified
    if args.architecture:
        arch = args.architecture.lower().replace("-", "_")
        canonical_models = [
            p for p in canonical_models
            if f"_{arch}" in p.name.lower() or p.name.lower().endswith(f"{arch}.pth")
        ]

    # Exclude backups
    canonical_models = [p for p in canonical_models if "_backup_" not in p.name]

    print(f"Found {len(canonical_models)} models to evaluate")

    # Dry run - just list models
    if args.dry_run:
        for model_path in canonical_models:
            print(f"  {model_path.name}")
        return

    results = {}
    for model_path in canonical_models:
        # Parse config from filename (handles architecture suffixes like _v2, _v5_heavy)
        name = model_path.stem  # e.g., canonical_hex8_2p_v2
        base_name = name.replace("canonical_", "").replace("ringrift_best_", "")

        # Extract board_type and num_players from base name
        # Pattern: {board_type}_{num_players}p[_architecture]
        match = re.match(r"(hex8|hexagonal|square8|square19)_(\d)p", base_name)
        if not match:
            print(f"Skipping {name}: could not parse config")
            continue

        board_type = match.group(1)
        num_players = int(match.group(2))

        print(f"\n{'='*60}")
        print(f"Evaluating {model_path.name}")
        print(f"Board: {board_type}, Players: {num_players}")
        print(f"{'='*60}")

        try:
            result = run_baseline_gauntlet(
                model_path=str(model_path),
                board_type=board_type,
                num_players=num_players,
                games_per_opponent=args.games,
                opponents=[BaselineOpponent.RANDOM, BaselineOpponent.HEURISTIC],
                store_results=True,
                check_baseline_gating=False,
            )
            print(f"\nResults for {name}:")
            print(f"  Total games: {result.total_games}")
            print(f"  Win rate: {result.win_rate:.1%}")
            print(f"  Estimated Elo: {result.estimated_elo:.0f}")
            print(f"  vs Random: {result.opponent_results.get('random', {})}")
            print(f"  vs Heuristic: {result.opponent_results.get('heuristic', {})}")

            results[name] = {
                "win_rate": result.win_rate,
                "elo": result.estimated_elo,
                "games": result.total_games,
            }
        except Exception as e:
            print(f"Error evaluating {name}: {e}")
            results[name] = {"error": str(e)}

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, res in results.items():
        if "error" in res:
            print(f"  {name}: ERROR - {res['error']}")
        else:
            print(f"  {name}: Elo={res['elo']:.0f}, WR={res['win_rate']:.1%}")


if __name__ == "__main__":
    main()
