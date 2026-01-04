#!/usr/bin/env python3
"""Evaluate all canonical models and update Elo database."""

import sys
from pathlib import Path

# Add ai-service to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.training.game_gauntlet import run_baseline_gauntlet, BaselineOpponent

def main():
    models_dir = Path("models")

    # Get canonical models (exclude backups and variants)
    canonical_models = [
        p for p in sorted(models_dir.glob("canonical_*.pth"))
        if "_backup_" not in p.name and "_v2" not in p.name and "_v5" not in p.name
    ]

    print(f"Found {len(canonical_models)} canonical models to evaluate")

    results = {}
    for model_path in canonical_models:
        # Parse config from filename
        name = model_path.stem  # e.g., canonical_hex8_2p
        parts = name.replace("canonical_", "").split("_")
        board_type = parts[0]
        num_players = int(parts[1].replace("p", ""))

        print(f"\n{'='*60}")
        print(f"Evaluating {model_path.name}")
        print(f"Board: {board_type}, Players: {num_players}")
        print(f"{'='*60}")

        try:
            result = run_baseline_gauntlet(
                model_path=str(model_path),
                board_type=board_type,
                num_players=num_players,
                games_per_opponent=20,
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
