#!/usr/bin/env python3
"""Run fresh Elo evaluations on cluster using GPU Gumbel MCTS.

January 2026: Created to refresh stale Elo entries with proper harness tracking.
Runs evaluation games on cluster GPU nodes, records results with composite
participant IDs that include harness type and simulation budget.

Usage:
    # Dry run for square8_2p
    python scripts/run_cluster_elo_refresh.py --configs square8_2p --dry-run

    # Run all configs with 100 games each
    python scripts/run_cluster_elo_refresh.py --configs all --games-per-config 100

    # Run specific configs with custom budget
    python scripts/run_cluster_elo_refresh.py \
        --configs hex8_2p,square8_2p \
        --harness gumbel_mcts \
        --budget 800 \
        --games-per-config 50
"""

import argparse
import asyncio
import logging
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# All 12 canonical configurations
ALL_CONFIGS = [
    "hex8_2p", "hex8_3p", "hex8_4p",
    "square8_2p", "square8_3p", "square8_4p",
    "square19_2p", "square19_3p", "square19_4p",
    "hexagonal_2p", "hexagonal_3p", "hexagonal_4p",
]

# Harness types supported
HARNESS_TYPES = ["gumbel_mcts", "minimax", "maxn", "policy_only", "heuristic"]

# Gumbel budget tiers
BUDGET_TIERS = {
    "throughput": 64,
    "standard": 150,
    "quality": 200,
    "ultimate": 800,
    "master": 1600,
}


@dataclass
class RefreshConfig:
    """Configuration for Elo refresh run."""
    configs: list[str] = field(default_factory=list)
    harness: str = "gumbel_mcts"
    budget: int = 800
    games_per_config: int = 100
    games_vs_random: int = 50
    games_vs_heuristic: int = 50
    dry_run: bool = False
    use_composite_ids: bool = True


@dataclass
class RefreshResult:
    """Result of a refresh run."""
    config_key: str
    composite_id: str
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    old_elo: float = 1500.0
    new_elo: float = 1500.0
    vs_random_win_rate: float = 0.0
    vs_heuristic_win_rate: float = 0.0


def make_composite_id(model_name: str, harness: str, budget: int) -> str:
    """Create composite participant ID with harness info.

    Format: model_name:harness_type:bN
    Examples:
        canonical_hex8_2p:gumbel_mcts:b800
        canonical_square8_2p:minimax:b150
    """
    return f"{model_name}:{harness}:b{budget}"


def parse_config_key(config_key: str) -> tuple[str, int]:
    """Parse config key into board_type and num_players.

    Examples:
        square8_2p -> (square8, 2)
        hexagonal_3p -> (hexagonal, 3)
    """
    parts = config_key.rsplit("_", 1)
    board_type = parts[0]
    num_players = int(parts[1].replace("p", ""))
    return board_type, num_players


def get_canonical_model_path(config_key: str) -> Path:
    """Get path to canonical model for a config."""
    model_dir = Path("models")
    patterns = [
        f"canonical_{config_key}.pth",
        f"ringrift_best_{config_key}.pth",
    ]
    for pattern in patterns:
        path = model_dir / pattern
        if path.exists():
            return path
    return model_dir / f"canonical_{config_key}.pth"


async def run_evaluation_games(
    config_key: str,
    model_path: Path,
    harness: str,
    budget: int,
    games_vs_random: int,
    games_vs_heuristic: int,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run evaluation games for a single config.

    Returns dict with:
        - games_played
        - wins, losses, draws
        - vs_random_results
        - vs_heuristic_results
    """
    board_type, num_players = parse_config_key(config_key)

    if dry_run:
        logger.info(f"[DRY RUN] Would run evaluation for {config_key}:")
        logger.info(f"  Model: {model_path}")
        logger.info(f"  Harness: {harness}")
        logger.info(f"  Budget: {budget}")
        logger.info(f"  Games vs random: {games_vs_random}")
        logger.info(f"  Games vs heuristic: {games_vs_heuristic}")
        return {
            "games_played": games_vs_random + games_vs_heuristic,
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "vs_random": {"wins": 0, "losses": 0, "games": games_vs_random},
            "vs_heuristic": {"wins": 0, "losses": 0, "games": games_vs_heuristic},
            "dry_run": True,
        }

    # Try to import game gauntlet
    try:
        from app.training.game_gauntlet import run_baseline_gauntlet, BaselineOpponent
    except ImportError:
        logger.error("Cannot import game_gauntlet - evaluation not available")
        return {"error": "game_gauntlet not available"}

    logger.info(f"Running evaluation for {config_key}...")
    logger.info(f"  Model: {model_path}")
    logger.info(f"  Harness: {harness}, Budget: {budget}")

    # Run games vs random
    random_results = {"wins": 0, "losses": 0, "games": games_vs_random}
    if games_vs_random > 0:
        try:
            result = await asyncio.to_thread(
                run_baseline_gauntlet,
                model_path=str(model_path),
                board_type=board_type,
                num_players=num_players,
                opponents=[BaselineOpponent.RANDOM],
                games_per_opponent=games_vs_random,
                harness_type=harness,
                verbose=True,
                check_baseline_gating=False,  # Skip gating for refresh
            )
            # Extract RANDOM-specific results from the gauntlet result
            # opponent_results uses string keys like "random", "heuristic"
            if hasattr(result, 'opponent_results') and "random" in result.opponent_results:
                opp_result = result.opponent_results["random"]
                random_results["wins"] = opp_result.get("wins", 0)
                random_results["losses"] = opp_result.get("losses", 0)
            else:
                # Fall back to total counts
                random_results["wins"] = getattr(result, 'total_wins', 0)
                random_results["losses"] = getattr(result, 'total_losses', 0)
            logger.info(f"  vs Random: {random_results['wins']}/{games_vs_random} wins")
        except Exception as e:
            logger.warning(f"  vs Random failed: {e}")

    # Run games vs heuristic
    heuristic_results = {"wins": 0, "losses": 0, "games": games_vs_heuristic}
    if games_vs_heuristic > 0:
        try:
            result = await asyncio.to_thread(
                run_baseline_gauntlet,
                model_path=str(model_path),
                board_type=board_type,
                num_players=num_players,
                opponents=[BaselineOpponent.HEURISTIC],
                games_per_opponent=games_vs_heuristic,
                harness_type=harness,
                verbose=True,
                check_baseline_gating=False,  # Skip gating for refresh
            )
            # Extract HEURISTIC-specific results from the gauntlet result
            # opponent_results uses string keys like "random", "heuristic"
            if hasattr(result, 'opponent_results') and "heuristic" in result.opponent_results:
                opp_result = result.opponent_results["heuristic"]
                heuristic_results["wins"] = opp_result.get("wins", 0)
                heuristic_results["losses"] = opp_result.get("losses", 0)
            else:
                # Fall back to total counts
                heuristic_results["wins"] = getattr(result, 'total_wins', 0)
                heuristic_results["losses"] = getattr(result, 'total_losses', 0)
            logger.info(f"  vs Heuristic: {heuristic_results['wins']}/{games_vs_heuristic} wins")
        except Exception as e:
            logger.warning(f"  vs Heuristic failed: {e}")

    total_wins = random_results["wins"] + heuristic_results["wins"]
    total_losses = random_results["losses"] + heuristic_results["losses"]
    total_games = games_vs_random + games_vs_heuristic

    return {
        "games_played": total_games,
        "wins": total_wins,
        "losses": total_losses,
        "draws": total_games - total_wins - total_losses,
        "vs_random": random_results,
        "vs_heuristic": heuristic_results,
    }


def record_elo_result(
    db_path: Path,
    composite_id: str,
    config_key: str,
    games_played: int,
    wins: int,
    losses: int,
    draws: int,
    harness: str,
    budget: int,
) -> float:
    """Record evaluation results to Elo database.

    Creates or updates composite participant ID with full harness metadata.
    Returns the new Elo rating.
    """
    board_type, num_players = parse_config_key(config_key)

    conn = sqlite3.connect(db_path)

    # Check if entry exists
    cur = conn.execute("""
        SELECT rating, games_played, wins, losses, draws
        FROM elo_ratings
        WHERE participant_id = ? AND board_type = ? AND num_players = ?
    """, (composite_id, board_type, num_players))

    row = cur.fetchone()

    if row:
        # Update existing entry
        old_rating, old_games, old_wins, old_losses, old_draws = row
        new_games = old_games + games_played
        new_wins = old_wins + wins
        new_losses = old_losses + losses
        new_draws = old_draws + draws

        # Simple Elo update based on win rate vs baseline
        # Baseline (random) is 400 Elo, Heuristic is ~1200 Elo
        win_rate = wins / games_played if games_played > 0 else 0.5
        expected_elo = 400 + (win_rate * 1200)  # Scale 400-1600 based on win rate
        new_rating = old_rating + 32 * (win_rate - 0.5)  # K=32 adjustment

        conn.execute("""
            UPDATE elo_ratings
            SET rating = ?, games_played = ?, wins = ?, losses = ?, draws = ?,
                last_update = ?
            WHERE participant_id = ? AND board_type = ? AND num_players = ?
        """, (new_rating, new_games, new_wins, new_losses, new_draws,
              time.time(), composite_id, board_type, num_players))

    else:
        # Create new entry
        # Start at 1500 and adjust based on win rate
        win_rate = wins / games_played if games_played > 0 else 0.5
        new_rating = 1500 + 100 * (win_rate - 0.5)  # Scale around 1500

        conn.execute("""
            INSERT INTO elo_ratings
            (participant_id, board_type, num_players, rating, games_played,
             wins, losses, draws, last_update)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (composite_id, board_type, num_players, new_rating, games_played,
              wins, losses, draws, time.time()))

    conn.commit()
    conn.close()

    logger.info(f"Recorded Elo for {composite_id}: {new_rating:.1f} ({games_played} games)")
    return new_rating


async def refresh_config(
    config_key: str,
    cfg: RefreshConfig,
    db_path: Path,
) -> RefreshResult:
    """Refresh Elo for a single config."""
    model_path = get_canonical_model_path(config_key)
    model_name = f"canonical_{config_key}"
    composite_id = make_composite_id(model_name, cfg.harness, cfg.budget)

    result = RefreshResult(
        config_key=config_key,
        composite_id=composite_id,
    )

    if not model_path.exists():
        logger.warning(f"Model not found: {model_path}")
        return result

    # Run evaluation games
    eval_results = await run_evaluation_games(
        config_key=config_key,
        model_path=model_path,
        harness=cfg.harness,
        budget=cfg.budget,
        games_vs_random=cfg.games_vs_random,
        games_vs_heuristic=cfg.games_vs_heuristic,
        dry_run=cfg.dry_run,
    )

    if "error" in eval_results:
        logger.error(f"Evaluation failed for {config_key}: {eval_results['error']}")
        return result

    result.games_played = eval_results.get("games_played", 0)
    result.wins = eval_results.get("wins", 0)
    result.losses = eval_results.get("losses", 0)
    result.draws = eval_results.get("draws", 0)

    if eval_results.get("vs_random", {}).get("games", 0) > 0:
        result.vs_random_win_rate = (
            eval_results["vs_random"]["wins"] /
            eval_results["vs_random"]["games"]
        )

    if eval_results.get("vs_heuristic", {}).get("games", 0) > 0:
        result.vs_heuristic_win_rate = (
            eval_results["vs_heuristic"]["wins"] /
            eval_results["vs_heuristic"]["games"]
        )

    # Record to Elo database (unless dry run)
    if not cfg.dry_run and result.games_played > 0:
        result.new_elo = record_elo_result(
            db_path=db_path,
            composite_id=composite_id,
            config_key=config_key,
            games_played=result.games_played,
            wins=result.wins,
            losses=result.losses,
            draws=result.draws,
            harness=cfg.harness,
            budget=cfg.budget,
        )

    return result


async def run_refresh(cfg: RefreshConfig, db_path: Path) -> list[RefreshResult]:
    """Run Elo refresh for all configured configs."""
    results = []

    for config_key in cfg.configs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Refreshing Elo for: {config_key}")
        logger.info(f"{'='*60}")

        try:
            result = await refresh_config(config_key, cfg, db_path)
            results.append(result)

            if result.games_played > 0:
                logger.info(f"Result for {config_key}:")
                logger.info(f"  Composite ID: {result.composite_id}")
                logger.info(f"  Games: {result.games_played}")
                logger.info(f"  Win rate: {(result.wins / result.games_played * 100):.1f}%")
                logger.info(f"  vs Random: {result.vs_random_win_rate*100:.1f}%")
                logger.info(f"  vs Heuristic: {result.vs_heuristic_win_rate*100:.1f}%")
                logger.info(f"  New Elo: {result.new_elo:.1f}")

        except Exception as e:
            logger.error(f"Failed to refresh {config_key}: {e}")
            results.append(RefreshResult(config_key=config_key, composite_id=""))

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run fresh Elo evaluations on cluster using GPU Gumbel MCTS"
    )
    parser.add_argument(
        "--configs",
        default="all",
        help="Configs to refresh (comma-separated or 'all')",
    )
    parser.add_argument(
        "--harness",
        choices=HARNESS_TYPES,
        default="gumbel_mcts",
        help="Harness type for evaluation (default: gumbel_mcts)",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=800,
        help="Simulation budget for Gumbel MCTS (default: 800)",
    )
    parser.add_argument(
        "--games-per-config",
        type=int,
        default=100,
        help="Total games per config (split between random/heuristic)",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/unified_elo.db"),
        help="Path to Elo database",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing",
    )
    parser.add_argument(
        "--use-composite-ids",
        action="store_true",
        default=True,
        help="Use composite participant IDs with harness info (default: True)",
    )

    args = parser.parse_args()

    # Parse configs
    if args.configs.lower() == "all":
        configs = ALL_CONFIGS
    else:
        configs = [c.strip() for c in args.configs.split(",")]

    # Validate configs
    for c in configs:
        if c not in ALL_CONFIGS:
            logger.warning(f"Unknown config: {c}")

    configs = [c for c in configs if c in ALL_CONFIGS]

    if not configs:
        logger.error("No valid configs specified")
        return 1

    # Split games between random and heuristic
    games_vs_random = args.games_per_config // 2
    games_vs_heuristic = args.games_per_config - games_vs_random

    cfg = RefreshConfig(
        configs=configs,
        harness=args.harness,
        budget=args.budget,
        games_per_config=args.games_per_config,
        games_vs_random=games_vs_random,
        games_vs_heuristic=games_vs_heuristic,
        dry_run=args.dry_run,
        use_composite_ids=args.use_composite_ids,
    )

    logger.info("=" * 70)
    logger.info("ELO REFRESH CONFIGURATION")
    logger.info("=" * 70)
    logger.info(f"Configs: {len(configs)} ({', '.join(configs[:5])}{'...' if len(configs) > 5 else ''})")
    logger.info(f"Harness: {cfg.harness}")
    logger.info(f"Budget: {cfg.budget} simulations")
    logger.info(f"Games per config: {cfg.games_per_config}")
    logger.info(f"  vs Random: {cfg.games_vs_random}")
    logger.info(f"  vs Heuristic: {cfg.games_vs_heuristic}")
    logger.info(f"Database: {args.db}")
    logger.info(f"Dry run: {cfg.dry_run}")
    logger.info("=" * 70)

    # Run refresh
    results = asyncio.run(run_refresh(cfg, args.db))

    # Summary
    print("\n" + "=" * 70)
    print("REFRESH SUMMARY")
    print("=" * 70)

    total_games = sum(r.games_played for r in results)
    total_wins = sum(r.wins for r in results)

    print(f"Configs processed: {len(results)}")
    print(f"Total games: {total_games}")
    print(f"Overall win rate: {(total_wins / total_games * 100) if total_games > 0 else 0:.1f}%")
    print()

    for r in results:
        if r.games_played > 0:
            win_rate = r.wins / r.games_played * 100
            print(f"  {r.config_key:20} {r.new_elo:7.1f} Elo  "
                  f"({r.games_played:3d} games, {win_rate:.0f}% wins)")
        else:
            print(f"  {r.config_key:20} FAILED")

    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
