#!/usr/bin/env python3
"""Automated model promotion pipeline.

Periodically checks for new trained models, evaluates them via Elo gauntlet,
and promotes top performers to production if they exceed the threshold.

Usage:
    # One-time evaluation
    python scripts/auto_model_promotion.py --check-once

    # Continuous monitoring
    python scripts/auto_model_promotion.py --daemon --interval 3600

    # Promote specific model
    python scripts/auto_model_promotion.py --promote model_id --board-type square8_2p
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.training.model_registry import ModelRegistry, ModelStage
from scripts.elo_promotion_gate import evaluate_promotion

logger = logging.getLogger(__name__)

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_DIR = AI_SERVICE_ROOT / "models" / "production"
ELO_DB_PATH = AI_SERVICE_ROOT / "data" / "unified_elo.db"

# Promotion thresholds
ELO_IMPROVEMENT_THRESHOLD = 50  # Must be 50+ Elo above current best
MIN_GAMES_FOR_PROMOTION = 20    # Minimum games before considering
CONFIDENCE_LEVEL = 0.90         # 90% confidence for promotion


def get_current_best(board_config: str) -> tuple[str | None, float]:
    """Get current best model and its Elo for a board config."""
    conn = sqlite3.connect(str(ELO_DB_PATH))
    cur = conn.cursor()

    # Parse board config
    parts = board_config.split("_")
    board_type = parts[0]
    num_players = int(parts[1].replace("p", ""))

    cur.execute("""
        SELECT participant_id, rating, games_played
        FROM elo_ratings
        WHERE board_type = ? AND num_players = ? AND games_played >= ?
        ORDER BY rating DESC LIMIT 1
    """, (board_type, num_players, MIN_GAMES_FOR_PROMOTION))

    row = cur.fetchone()
    conn.close()

    if row:
        return row[0], row[1]
    return None, 1500.0  # Default Elo if no best model


def get_candidates_for_promotion(board_config: str) -> list[dict]:
    """Get models that might be ready for promotion."""
    conn = sqlite3.connect(str(ELO_DB_PATH))
    cur = conn.cursor()

    parts = board_config.split("_")
    board_type = parts[0]
    num_players = int(parts[1].replace("p", ""))

    current_best_id, current_best_elo = get_current_best(board_config)
    threshold_elo = current_best_elo + ELO_IMPROVEMENT_THRESHOLD

    cur.execute("""
        SELECT participant_id, rating, games_played, wins, losses
        FROM elo_ratings
        WHERE board_type = ? AND num_players = ?
          AND games_played >= ?
          AND rating >= ?
          AND participant_id != ?
        ORDER BY rating DESC
    """, (board_type, num_players, MIN_GAMES_FOR_PROMOTION,
          threshold_elo, current_best_id or ""))

    candidates = []
    for row in cur.fetchall():
        candidates.append({
            "model_id": row[0],
            "elo": row[1],
            "games_played": row[2],
            "wins": row[3] or 0,
            "losses": row[4] or 0,
        })

    conn.close()
    return candidates


def evaluate_candidate(candidate: dict, current_best_elo: float) -> dict:
    """Evaluate if a candidate should be promoted."""
    result = evaluate_promotion(
        candidate_wins=candidate["wins"],
        candidate_losses=candidate["losses"],
        threshold=0.55,
        confidence=CONFIDENCE_LEVEL,
    )

    elo_diff = candidate["elo"] - current_best_elo

    return {
        **result,
        "candidate_id": candidate["model_id"],
        "elo": candidate["elo"],
        "elo_improvement": elo_diff,
        "games_played": candidate["games_played"],
    }


def promote_model(model_id: str, board_config: str, elo: float) -> bool:
    """Promote a model to production."""
    # Find the model file
    models_dir = AI_SERVICE_ROOT / "models"
    model_files = list(models_dir.glob(f"*{model_id}*.pth"))

    if not model_files:
        # Try finding by partial match
        model_files = list(models_dir.glob(f"*{model_id[:20]}*.pth"))

    if not model_files:
        logger.warning(f"Could not find model file for {model_id}")
        return False

    # Create production directory
    PRODUCTION_DIR.mkdir(parents=True, exist_ok=True)

    # Copy to production with versioned name
    src_file = model_files[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst_name = f"ringrift_v8_{board_config}_{timestamp}.pth"
    dst_file = PRODUCTION_DIR / dst_name

    shutil.copy2(src_file, dst_file)
    logger.info(f"Promoted {model_id} to {dst_file}")

    # Update model registry if available
    try:
        registry = ModelRegistry()
        registry.promote_model(model_id, ModelStage.PRODUCTION)
    except Exception as e:
        logger.warning(f"Could not update registry: {e}")

    # Write promotion record
    record = {
        "model_id": model_id,
        "board_config": board_config,
        "elo": elo,
        "promoted_at": datetime.now().isoformat(),
        "production_path": str(dst_file),
    }

    records_file = PRODUCTION_DIR / "promotion_log.jsonl"
    with open(records_file, "a") as f:
        f.write(json.dumps(record) + "\n")

    return True


def check_and_promote(board_configs: list[str] = None) -> list[dict]:
    """Check all board configs and promote worthy candidates."""
    if board_configs is None:
        board_configs = [
            "square8_2p", "square8_3p", "square8_4p",
            "square19_2p", "square19_3p", "square19_4p",
            "hexagonal_2p", "hexagonal_3p", "hexagonal_4p",
        ]

    promotions = []

    for config in board_configs:
        logger.info(f"Checking {config}...")
        current_best_id, current_best_elo = get_current_best(config)

        if current_best_id:
            logger.info(f"  Current best: {current_best_id[:30]} ({current_best_elo:.1f} Elo)")

        candidates = get_candidates_for_promotion(config)

        for candidate in candidates:
            eval_result = evaluate_candidate(candidate, current_best_elo)

            if eval_result.get("promote", False):
                logger.info(f"  Promoting {candidate['model_id'][:30]} "
                          f"(+{eval_result['elo_improvement']:.0f} Elo)")

                if promote_model(candidate["model_id"], config, candidate["elo"]):
                    promotions.append({
                        "board_config": config,
                        "model_id": candidate["model_id"],
                        "elo": candidate["elo"],
                        "improvement": eval_result["elo_improvement"],
                    })
            else:
                logger.debug(f"  Candidate {candidate['model_id'][:20]} not ready: "
                           f"{eval_result.get('reason', 'unknown')}")

    return promotions


def run_daemon(interval: int = 3600):
    """Run continuous promotion monitoring."""
    logger.info(f"Starting promotion daemon (interval: {interval}s)")

    while True:
        try:
            promotions = check_and_promote()
            if promotions:
                logger.info(f"Promoted {len(promotions)} models this cycle")
            else:
                logger.info("No models ready for promotion")
        except Exception as e:
            logger.error(f"Error in promotion cycle: {e}")

        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="Automated model promotion")
    parser.add_argument("--check-once", action="store_true",
                       help="Run one promotion check and exit")
    parser.add_argument("--daemon", action="store_true",
                       help="Run as continuous daemon")
    parser.add_argument("--interval", type=int, default=3600,
                       help="Check interval in seconds (default: 3600)")
    parser.add_argument("--promote", type=str,
                       help="Manually promote a specific model")
    parser.add_argument("--board-type", type=str,
                       help="Board config (e.g., square8_2p)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")

    args = parser.parse_args()

    from scripts.lib.logging_config import setup_script_logging
    global logger
    logger = setup_script_logging("auto_model_promotion")
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.promote:
        if not args.board_type:
            print("Error: --board-type required with --promote")
            sys.exit(1)

        # Get current Elo for the model
        conn = sqlite3.connect(str(ELO_DB_PATH))
        cur = conn.cursor()
        cur.execute("SELECT rating FROM elo_ratings WHERE participant_id LIKE ?",
                   (f"%{args.promote}%",))
        row = cur.fetchone()
        elo = row[0] if row else 1500.0
        conn.close()

        if promote_model(args.promote, args.board_type, elo):
            print(f"Promoted {args.promote}")
        else:
            print(f"Failed to promote {args.promote}")
            sys.exit(1)

    elif args.daemon:
        run_daemon(args.interval)

    else:
        promotions = check_and_promote()
        if promotions:
            print(f"\nPromoted {len(promotions)} models:")
            for p in promotions:
                print(f"  {p['board_config']}: {p['model_id'][:40]} (+{p['improvement']:.0f} Elo)")
        else:
            print("No models ready for promotion")


if __name__ == "__main__":
    main()
