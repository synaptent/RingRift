#!/usr/bin/env python3
"""Evaluate existing V4 models and register them in the Elo database.

This script retroactively evaluates V4 models that were trained but never
integrated into the Elo evaluation pipeline due to a broken connection
between the gauntlet runner and the Elo database.

Usage:
    # Evaluate all V4 models
    python scripts/evaluate_v4_models.py

    # Dry run (show what would be evaluated)
    python scripts/evaluate_v4_models.py --dry-run

    # Evaluate specific pattern
    python scripts/evaluate_v4_models.py --pattern "*sq8*v4*"

    # Skip already evaluated models
    python scripts/evaluate_v4_models.py --skip-existing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root
SCRIPT_DIR = Path(__file__).parent
AI_SERVICE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(AI_SERVICE_ROOT))

from scripts.lib.logging_config import setup_script_logging

logger = setup_script_logging("evaluate_v4_models")


def find_v4_models(models_dir: Path, pattern: str = "*v4*.pth") -> list[Path]:
    """Find all V4 model files."""
    models = list(models_dir.glob(pattern))
    # Filter out symlinks and non-existent
    models = [m for m in models if m.exists() and not m.is_symlink()]
    return sorted(models, key=lambda p: p.stat().st_mtime, reverse=True)


def get_evaluated_models(db) -> set[str]:
    """Get set of already-evaluated model names from Elo DB."""
    try:
        import sqlite3
        conn = sqlite3.connect(db.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT participant_id FROM elo_ratings WHERE games_played > 0")
        result = {row[0] for row in cursor.fetchall()}
        conn.close()
        return result
    except Exception as e:
        logger.warning(f"Could not get evaluated models: {e}")
        return set()


def parse_model_info(path: Path) -> dict:
    """Parse board type and player count from model filename."""
    name = path.stem.lower()
    info = {"board_type": "square8", "num_players": 2}

    if "sq8" in name or "square8" in name:
        info["board_type"] = "square8"
    elif "sq19" in name or "square19" in name:
        info["board_type"] = "square19"
    elif "hex8" in name:
        info["board_type"] = "hex8"
    elif "hex" in name:
        info["board_type"] = "hexagonal"

    for np in [2, 3, 4]:
        if f"_{np}p" in name or f"{np}p_" in name:
            info["num_players"] = np
            break

    return info


def evaluate_model(model_path: Path, games_per_baseline: int = 10) -> dict | None:
    """Run gauntlet evaluation for a single model."""
    from app.models import BoardType
    from scripts.baseline_gauntlet import run_gauntlet_for_model

    info = parse_model_info(model_path)

    model_dict = {
        "path": str(model_path),
        "name": model_path.stem,
        "type": "nn",
    }

    try:
        board_type = BoardType(info["board_type"])
    except ValueError:
        board_type = BoardType.SQUARE8

    logger.info(f"Evaluating {model_path.stem} ({info['board_type']}_{info['num_players']}p)")

    try:
        result = run_gauntlet_for_model(
            model=model_dict,
            num_games=games_per_baseline,
            board_type=board_type,
            num_players=info["num_players"],
            fast_mode=True,
        )

        return {
            "model_path": str(model_path),
            "model_name": model_path.stem,
            "vs_random": result.vs_random,
            "vs_heuristic": result.vs_heuristic,
            "vs_mcts": result.vs_mcts,
            "score": result.score,
            "games_played": result.games_played,
            "board_type": info["board_type"],
            "num_players": info["num_players"],
        }
    except Exception as e:
        logger.error(f"Failed to evaluate {model_path.stem}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate V4 models and update Elo database")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be evaluated")
    parser.add_argument("--pattern", default="*v4*.pth", help="Glob pattern for models")
    parser.add_argument("--skip-existing", action="store_true", help="Skip already evaluated models")
    parser.add_argument("--games", type=int, default=10, help="Games per baseline")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of models to evaluate")
    args = parser.parse_args()

    models_dir = AI_SERVICE_ROOT / "models"
    models = find_v4_models(models_dir, args.pattern)

    print(f"\nFound {len(models)} V4 models matching '{args.pattern}'")

    if args.skip_existing:
        from scripts.unified_promotion_daemon import EloIntegration
        elo = EloIntegration()
        evaluated = get_evaluated_models(elo.db)
        models = [m for m in models if m.stem not in evaluated]
        print(f"After skipping evaluated: {len(models)} models to evaluate")

    if args.limit:
        models = models[:args.limit]
        print(f"Limited to {args.limit} models")

    if args.dry_run:
        print("\n[DRY RUN] Would evaluate:")
        for m in models:
            info = parse_model_info(m)
            print(f"  - {m.stem} ({info['board_type']}_{info['num_players']}p)")
        return 0

    if not models:
        print("No models to evaluate")
        return 0

    # Import Elo integration
    from scripts.unified_promotion_daemon import EloIntegration
    elo = EloIntegration()

    print(f"\nEvaluating {len(models)} models with {args.games} games per baseline...")
    print("=" * 70)

    evaluated = 0
    errors = 0

    for idx, model_path in enumerate(models, 1):
        print(f"\n[{idx}/{len(models)}] {model_path.stem}")

        result = evaluate_model(model_path, args.games)
        if result:
            if elo.update_elo_from_gauntlet(result):
                evaluated += 1
                print(f"  vs_random: {result['vs_random']:.1%}, vs_heuristic: {result['vs_heuristic']:.1%}")
            else:
                errors += 1
        else:
            errors += 1

    print("\n" + "=" * 70)
    print(f"EVALUATION COMPLETE")
    print(f"  Evaluated: {evaluated}")
    print(f"  Errors: {errors}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
