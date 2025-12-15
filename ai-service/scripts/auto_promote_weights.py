#!/usr/bin/env python3
"""Auto-promote best weights and sync across cluster.

This script automates the final step of the AI improvement loop:
1. Query Elo leaderboard for best models
2. Update model symlinks in promoted/ directory
3. Sync models to cluster nodes via P2P orchestrator
4. Notify the system of promotions

Usage:
    # Run auto-promotion for all configs
    python scripts/auto_promote_weights.py --run

    # Dry run (show what would be promoted)
    python scripts/auto_promote_weights.py --dry-run

    # Continuous mode (run every N minutes)
    python scripts/auto_promote_weights.py --continuous --interval 30
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add ai-service to path
AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_SERVICE_ROOT))

# Paths
MODELS_DIR = AI_SERVICE_ROOT / "models"
PROMOTED_DIR = MODELS_DIR / "promoted"
ELO_DB_PATH = AI_SERVICE_ROOT / "data" / "unified_elo.db"
PROMOTION_STATE_PATH = AI_SERVICE_ROOT / "data" / "auto_promotion_state.json"

# P2P orchestrator for cluster sync
P2P_URL = os.environ.get("P2P_ORCHESTRATOR_URL", "http://localhost:8770")

# Minimum requirements for promotion
MIN_GAMES = 10  # Minimum Elo games to be considered
MIN_ELO = 1480  # Minimum Elo rating to promote (above baseline 1500 - 20)


def get_best_model(board_type: str, num_players: int) -> Optional[Dict]:
    """Get the best model from Elo leaderboard for a config."""
    if not ELO_DB_PATH.exists():
        return None

    conn = sqlite3.connect(str(ELO_DB_PATH))
    c = conn.cursor()
    c.execute("""
        SELECT model_id, rating, games_played, wins, losses
        FROM elo_ratings
        WHERE board_type = ? AND num_players = ?
        AND games_played >= ?
        ORDER BY rating DESC
        LIMIT 1
    """, (board_type, num_players, MIN_GAMES))

    row = c.fetchone()
    conn.close()

    if row:
        return {
            "model_id": row[0],
            "elo": row[1],
            "games": row[2],
            "wins": row[3],
            "losses": row[4],
        }
    return None


def find_model_path(model_id: str) -> Optional[Path]:
    """Find the actual .pth file for a model ID."""
    # Direct path
    direct = MODELS_DIR / f"{model_id}.pth"
    if direct.exists():
        return direct

    # Search for partial match
    for pth in MODELS_DIR.glob("*.pth"):
        if model_id in pth.name:
            return pth

    return None


def create_symlink(model_path: Path, symlink_name: str) -> bool:
    """Create symlink in promoted directory."""
    PROMOTED_DIR.mkdir(parents=True, exist_ok=True)
    symlink_path = PROMOTED_DIR / symlink_name

    # Remove existing
    if symlink_path.exists() or symlink_path.is_symlink():
        symlink_path.unlink()

    try:
        rel_path = os.path.relpath(model_path, PROMOTED_DIR)
        symlink_path.symlink_to(rel_path)
        return True
    except Exception as e:
        print(f"  Error creating symlink: {e}")
        return False


def sync_to_cluster() -> bool:
    """Trigger cluster sync via P2P orchestrator."""
    try:
        import requests
        response = requests.post(
            f"{P2P_URL}/api/sync_models",
            json={"action": "sync_promoted_models"},
            timeout=30
        )
        return response.status_code == 200
    except Exception as e:
        print(f"  Cluster sync failed: {e}")
        return False


def load_state() -> Dict:
    """Load previous promotion state."""
    if PROMOTION_STATE_PATH.exists():
        with open(PROMOTION_STATE_PATH) as f:
            return json.load(f)
    return {"promotions": {}, "last_run": None}


def save_state(state: Dict):
    """Save promotion state."""
    PROMOTION_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PROMOTION_STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def run_promotion(dry_run: bool = False, verbose: bool = True) -> Dict:
    """Run auto-promotion for all configurations."""
    configs = [
        ("square8", 2),
        ("square8", 3),
        ("square8", 4),
        ("square19", 2),
        ("hexagonal", 2),
    ]

    state = load_state()
    results = {"promoted": [], "skipped": [], "unchanged": []}

    if verbose:
        print(f"\n{'='*60}")
        print(f" Auto-Promote Best Models {'(DRY RUN)' if dry_run else ''}")
        print(f"{'='*60}")

    for board_type, num_players in configs:
        config_key = f"{board_type}_{num_players}p"
        if verbose:
            print(f"\n--- {config_key} ---")

        best = get_best_model(board_type, num_players)
        if not best:
            if verbose:
                print(f"  No eligible model (need {MIN_GAMES}+ games)")
            results["skipped"].append({"config": config_key, "reason": "no_data"})
            continue

        if best["elo"] < MIN_ELO:
            if verbose:
                print(f"  Best model Elo {best['elo']:.0f} < {MIN_ELO} threshold")
            results["skipped"].append({"config": config_key, "reason": "low_elo"})
            continue

        model_path = find_model_path(best["model_id"])
        if not model_path:
            if verbose:
                print(f"  Model file not found: {best['model_id']}")
            results["skipped"].append({"config": config_key, "reason": "file_not_found"})
            continue

        # Check if already promoted
        prev = state["promotions"].get(config_key)
        if prev and prev.get("model_id") == best["model_id"]:
            if verbose:
                print(f"  Already promoted: {best['model_id'][:40]}")
            results["unchanged"].append(config_key)
            continue

        # Promote
        symlink_name = f"{config_key}_best.pth"
        if verbose:
            print(f"  Promoting: {best['model_id'][:40]}")
            print(f"  Elo: {best['elo']:.0f} | Games: {best['games']} | W/L: {best['wins']}/{best['losses']}")

        if not dry_run:
            if create_symlink(model_path, symlink_name):
                state["promotions"][config_key] = {
                    "model_id": best["model_id"],
                    "elo": best["elo"],
                    "promoted_at": datetime.now().isoformat(),
                }
                results["promoted"].append(config_key)
                if verbose:
                    print(f"  ✓ Created symlink: {symlink_name}")
            else:
                results["skipped"].append({"config": config_key, "reason": "symlink_failed"})
        else:
            results["promoted"].append(config_key)

    if not dry_run:
        state["last_run"] = datetime.now().isoformat()
        save_state(state)

        # Sync to cluster if any promotions
        if results["promoted"]:
            if verbose:
                print(f"\nSyncing to cluster...")
            if sync_to_cluster():
                if verbose:
                    print("  ✓ Cluster sync triggered")
            else:
                if verbose:
                    print("  ✗ Cluster sync failed (will retry)")

    if verbose:
        print(f"\n{'='*60}")
        print(f" Summary: {len(results['promoted'])} promoted, {len(results['unchanged'])} unchanged, {len(results['skipped'])} skipped")
        print(f"{'='*60}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Auto-promote best AI models")
    parser.add_argument("--run", action="store_true", help="Run promotion")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be promoted")
    parser.add_argument("--continuous", action="store_true", help="Run continuously")
    parser.add_argument("--interval", type=int, default=30, help="Minutes between runs in continuous mode")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress output")

    args = parser.parse_args()
    verbose = not args.quiet

    if args.continuous:
        print(f"Starting continuous promotion (every {args.interval} minutes)...")
        while True:
            try:
                run_promotion(dry_run=False, verbose=verbose)
            except Exception as e:
                print(f"Error during promotion: {e}")
            time.sleep(args.interval * 60)
    elif args.run:
        run_promotion(dry_run=False, verbose=verbose)
    elif args.dry_run:
        run_promotion(dry_run=True, verbose=verbose)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
