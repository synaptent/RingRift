#!/usr/bin/env python3
"""Convert gauntlet results to Elo ratings.

Reads baseline_gauntlet_results.json and updates unified_elo.db with
calibrated Elo ratings based on performance against fixed baselines.

Usage:
    # Process gauntlet results and update Elo
    python scripts/gauntlet_to_elo.py --process

    # Dry run (show what would be updated)
    python scripts/gauntlet_to_elo.py --dry-run

    # Process specific config
    python scripts/gauntlet_to_elo.py --process --config square8_2p
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Optional

# Add project root
SCRIPT_DIR = Path(__file__).parent
AI_SERVICE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(AI_SERVICE_ROOT))

from app.tournament.unified_elo_db import EloDatabase
from scripts.lib.logging_config import setup_script_logging

logger = setup_script_logging("gauntlet_to_elo")

# Default paths
GAUNTLET_RESULTS_FILE = AI_SERVICE_ROOT / "data" / "baseline_gauntlet_results.json"
ELO_DB_PATH = AI_SERVICE_ROOT / "data" / "unified_elo.db"

# Baseline Elo anchors - fixed reference points
BASELINE_ELOS = {
    "random": 400,      # Random player anchor
    "heuristic": 1000,  # Heuristic AI
    "mcts": 1400,       # MCTS with 100 simulations
}


def estimate_elo_from_win_rate(win_rate: float, opponent_elo: float) -> float:
    """Estimate player Elo from win rate against opponent with known Elo.

    Uses the Elo expected score formula inverted:
    E = 1 / (1 + 10^((Rb - Ra) / 400))

    Solving for Ra:
    Ra = Rb - 400 * log10(1/E - 1)
    """
    if win_rate <= 0.01:
        # Very low win rate - estimate much lower than opponent
        return opponent_elo - 400
    if win_rate >= 0.99:
        # Very high win rate - estimate much higher than opponent
        return opponent_elo + 400

    # Standard Elo formula inversion
    return opponent_elo - 400 * math.log10(1 / win_rate - 1)


def calibrate_elo_from_gauntlet(
    vs_random: float,
    vs_heuristic: float,
    vs_mcts: float,
) -> float:
    """Calibrate Elo rating from gauntlet results against multiple baselines.

    Uses weighted average of Elo estimates from each baseline,
    with higher weight for more reliable estimates (mid-range win rates).
    """
    estimates = []
    weights = []

    # Estimate from each baseline
    for win_rate, _baseline, baseline_elo in [
        (vs_random, "random", BASELINE_ELOS["random"]),
        (vs_heuristic, "heuristic", BASELINE_ELOS["heuristic"]),
        (vs_mcts, "mcts", BASELINE_ELOS["mcts"]),
    ]:
        if win_rate > 0:  # Skip if no games played
            elo_est = estimate_elo_from_win_rate(win_rate, baseline_elo)
            estimates.append(elo_est)

            # Weight by reliability: mid-range win rates are more reliable
            # Peak weight at 50% win rate, lower at extremes
            reliability = 1 - abs(win_rate - 0.5) * 2
            reliability = max(0.1, reliability)  # Minimum weight
            weights.append(reliability)

    if not estimates:
        return 1500.0  # Default Elo if no data

    # Weighted average
    total_weight = sum(weights)
    calibrated_elo = sum(e * w for e, w in zip(estimates, weights, strict=False)) / total_weight

    # Clamp to reasonable range
    return max(200, min(2400, calibrated_elo))


def parse_config_from_path(model_path: str) -> tuple[str, int] | None:
    """Extract board_type and num_players from model path.

    Handles patterns like:
    - models/sq8_2p_nn_baseline.pth -> (square8, 2)
    - models/square8_2p_iter20.pth -> (square8, 2)
    - models/hexagonal_3p_model.pth -> (hexagonal, 3)
    """
    path = Path(model_path)
    name = path.stem.lower()

    # Board type patterns
    board_type = None
    if "sq8" in name or "square8" in name:
        board_type = "square8"
    elif "sq19" in name or "square19" in name:
        board_type = "square19"
    elif "hex" in name:
        board_type = "hexagonal"

    # Player count patterns
    num_players = None
    if "_2p" in name or "2p_" in name:
        num_players = 2
    elif "_3p" in name or "3p_" in name:
        num_players = 3
    elif "_4p" in name or "4p_" in name:
        num_players = 4

    if board_type and num_players:
        return board_type, num_players
    return None


def load_gauntlet_results(path: Path = GAUNTLET_RESULTS_FILE) -> list[dict[str, Any]]:
    """Load gauntlet results from JSON file."""
    if not path.exists():
        logger.warning(f"Gauntlet results file not found: {path}")
        return []

    with open(path) as f:
        data = json.load(f)

    return data.get("results", [])


def process_gauntlet_results(
    results: list[dict[str, Any]],
    db: EloDatabase,
    config_filter: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Process gauntlet results and update Elo database.

    Args:
        results: List of gauntlet result dicts
        db: EloDatabase instance
        config_filter: Optional config key to filter (e.g., "square8_2p")
        dry_run: If True, don't write to database

    Returns:
        Summary dict with processing stats
    """
    processed = 0
    skipped = 0
    errors = 0
    updates = []

    # Generate a gauntlet run ID
    run_id = f"gauntlet_{int(time.time())}"

    for result in results:
        model_path = result.get("model_path", "")

        # Parse config from path
        config = parse_config_from_path(model_path)
        if not config:
            logger.debug(f"Could not parse config from {model_path}")
            skipped += 1
            continue

        board_type, num_players = config
        config_key = f"{board_type}_{num_players}p"

        # Apply filter if specified
        if config_filter and config_key != config_filter:
            skipped += 1
            continue

        # Get win rates
        vs_random = result.get("vs_random", 0.0)
        vs_heuristic = result.get("vs_heuristic", 0.0)
        vs_mcts = result.get("vs_mcts", 0.0)

        # Calibrate Elo
        calibrated_elo = calibrate_elo_from_gauntlet(vs_random, vs_heuristic, vs_mcts)

        # Generate participant ID from model path
        model_name = Path(model_path).stem
        participant_id = f"model_{model_name}"

        updates.append({
            "participant_id": participant_id,
            "model_path": model_path,
            "board_type": board_type,
            "num_players": num_players,
            "vs_random": vs_random,
            "vs_heuristic": vs_heuristic,
            "vs_mcts": vs_mcts,
            "calibrated_elo": calibrated_elo,
        })

        if not dry_run:
            try:
                # Register participant
                db.register_participant(
                    participant_id=participant_id,
                    participant_type="model",
                    model_path=model_path,
                )

                # Get current rating
                current = db.get_rating(participant_id, board_type, num_players)

                # Update rating
                current.rating = calibrated_elo
                current.games_played = result.get("games_played", 0)
                # Approximate wins/losses from win rates
                games_per_baseline = current.games_played // 3 if current.games_played else 10
                current.wins = int((vs_random + vs_heuristic + vs_mcts) / 3 * games_per_baseline * 3)
                current.losses = current.games_played - current.wins

                db.update_rating(current)
                processed += 1

            except Exception as e:
                logger.error(f"Error updating {participant_id}: {e}")
                errors += 1
        else:
            processed += 1

    return {
        "run_id": run_id,
        "processed": processed,
        "skipped": skipped,
        "errors": errors,
        "updates": updates if dry_run else [],
        "dry_run": dry_run,
    }


def main():
    parser = argparse.ArgumentParser(description="Convert gauntlet results to Elo ratings")
    parser.add_argument("--process", action="store_true", help="Process gauntlet results")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be updated")
    parser.add_argument("--config", type=str, help="Filter by config (e.g., square8_2p)")
    parser.add_argument("--input", type=str, help="Input gauntlet results file")
    parser.add_argument("--db", type=str, help="Output Elo database path")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if not args.process and not args.dry_run:
        parser.print_help()
        return

    # Load results
    input_path = Path(args.input) if args.input else GAUNTLET_RESULTS_FILE
    results = load_gauntlet_results(input_path)

    if not results:
        print(f"No gauntlet results found in {input_path}")
        return

    print(f"Loaded {len(results)} gauntlet results")

    # Open database
    db_path = Path(args.db) if args.db else ELO_DB_PATH
    db = EloDatabase(db_path)

    # Process
    summary = process_gauntlet_results(
        results=results,
        db=db,
        config_filter=args.config,
        dry_run=args.dry_run,
    )

    # Print summary
    print()
    print("=" * 60)
    print("GAUNTLET TO ELO CONVERSION SUMMARY")
    print("=" * 60)
    print(f"Run ID: {summary['run_id']}")
    print(f"Mode: {'DRY RUN' if summary['dry_run'] else 'LIVE'}")
    print(f"Processed: {summary['processed']}")
    print(f"Skipped: {summary['skipped']}")
    print(f"Errors: {summary['errors']}")

    if args.dry_run and summary["updates"]:
        print()
        print("Would update the following ratings:")
        print("-" * 60)
        for u in sorted(summary["updates"], key=lambda x: x["calibrated_elo"], reverse=True)[:20]:
            print(f"  {u['participant_id'][:30]:<32} {u['board_type']}_{u['num_players']}p  Elo: {u['calibrated_elo']:.0f}")
            print(f"    vs_random: {u['vs_random']:.1%}  vs_heuristic: {u['vs_heuristic']:.1%}  vs_mcts: {u['vs_mcts']:.1%}")
        if len(summary["updates"]) > 20:
            print(f"  ... and {len(summary['updates']) - 20} more")


if __name__ == "__main__":
    main()
