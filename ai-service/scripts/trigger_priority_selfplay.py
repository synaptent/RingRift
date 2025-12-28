#!/usr/bin/env python3
"""Trigger priority selfplay for underserved configurations.

This script dispatches selfplay jobs to the P2P cluster for configurations
that have empty, corrupted, or sparse training databases.

Usage:
    # Check which configs need data (dry run)
    python scripts/trigger_priority_selfplay.py --dry-run

    # Trigger selfplay via P2P leader
    python scripts/trigger_priority_selfplay.py

    # Trigger specific configs only
    python scripts/trigger_priority_selfplay.py --configs hex8_4p,square8_2p

    # Use specific node
    python scripts/trigger_priority_selfplay.py --leader nebius-backbone-1
"""

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# Configs that need priority selfplay (from config analysis Dec 28, 2025)
PRIORITY_CONFIGS = {
    "hex8_4p": {"games": 500, "reason": "CORRUPTED - move data corruption"},
    "square8_2p": {"games": 500, "reason": "EMPTY - 0 games"},
    "square8_4p": {"games": 500, "reason": "EMPTY - 0 games"},
    "square19_2p": {"games": 300, "reason": "EMPTY - 0 games"},
    "square19_4p": {"games": 200, "reason": "SPARSE - 1 game"},
    "hexagonal_3p": {"games": 200, "reason": "SPARSE - minimal games"},
    "hexagonal_4p": {"games": 200, "reason": "SPARSE - minimal games"},
}


def check_database_status(config_key: str) -> dict:
    """Check the status of a canonical database."""
    board_type, num_players = config_key.rsplit("_", 1)
    num_players = int(num_players.rstrip("p"))

    db_path = Path(f"data/games/canonical_{board_type}_{num_players}p.db")

    status = {
        "config_key": config_key,
        "board_type": board_type,
        "num_players": num_players,
        "db_exists": db_path.exists(),
        "db_path": str(db_path),
        "game_count": 0,
        "completed_games": 0,
        "status": "UNKNOWN",
    }

    if not db_path.exists():
        status["status"] = "MISSING"
        return status

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Count total games
        cursor.execute("SELECT COUNT(*) FROM games")
        status["game_count"] = cursor.fetchone()[0]

        # Count completed games
        cursor.execute("SELECT COUNT(*) FROM games WHERE game_status = 'completed'")
        status["completed_games"] = cursor.fetchone()[0]

        conn.close()

        if status["completed_games"] == 0:
            status["status"] = "EMPTY"
        elif status["completed_games"] < 50:
            status["status"] = "SPARSE"
        elif status["completed_games"] < 200:
            status["status"] = "LOW"
        else:
            status["status"] = "OK"

    except Exception as e:
        status["status"] = f"ERROR: {e}"

    return status


def trigger_selfplay_via_p2p(
    leader_host: str,
    board_type: str,
    num_players: int,
    num_games: int,
    dry_run: bool = False,
) -> bool:
    """Trigger selfplay job via P2P leader HTTP endpoint."""
    import urllib.request
    import urllib.error

    url = f"http://{leader_host}:8770/jobs/selfplay"
    data = {
        "board_type": board_type,
        "num_players": num_players,
        "num_games": num_games,
        "priority": "high",
        "engine": "gumbel-mcts",
    }

    if dry_run:
        logger.info(f"  [DRY RUN] Would POST to {url}")
        logger.info(f"  [DRY RUN] Data: {json.dumps(data)}")
        return True

    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))
            if result.get("success"):
                job_id = result.get("job_id", "unknown")
                logger.info(f"  ✓ Job dispatched: {job_id}")
                return True
            else:
                logger.error(f"  ✗ Job failed: {result.get('error', 'unknown')}")
                return False

    except urllib.error.URLError as e:
        logger.error(f"  ✗ Connection failed: {e}")
        return False
    except Exception as e:
        logger.error(f"  ✗ Error: {e}")
        return False


def find_p2p_leader() -> str | None:
    """Try to find the current P2P leader."""
    import urllib.request
    import urllib.error

    # Try localhost first
    try:
        url = "http://localhost:8770/status"
        with urllib.request.urlopen(url, timeout=5) as response:
            data = json.loads(response.read().decode("utf-8"))
            leader = data.get("leader_id")
            if leader:
                return leader
    except Exception:
        pass

    # Try known stable nodes
    for host in ["nebius-backbone-1", "nebius-h100-3", "hetzner-cpu1"]:
        try:
            # Would need to resolve Tailscale IPs - skip for now
            pass
        except Exception:
            continue

    return None


def main():
    parser = argparse.ArgumentParser(description="Trigger priority selfplay for underserved configs")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--configs", type=str, help="Comma-separated list of configs to trigger")
    parser.add_argument("--leader", type=str, help="P2P leader host (e.g., nebius-backbone-1 or IP)")
    parser.add_argument("--check-only", action="store_true", help="Only check database status")
    args = parser.parse_args()

    # Determine which configs to process
    if args.configs:
        configs_to_process = [c.strip() for c in args.configs.split(",")]
    else:
        configs_to_process = list(PRIORITY_CONFIGS.keys())

    logger.info("=" * 60)
    logger.info("Priority Selfplay Trigger")
    logger.info("=" * 60)

    # Check status of all configs
    logger.info("\nDatabase Status:")
    logger.info("-" * 40)

    needs_selfplay = []
    for config_key in configs_to_process:
        status = check_database_status(config_key)
        games_needed = PRIORITY_CONFIGS.get(config_key, {}).get("games", 500)
        reason = PRIORITY_CONFIGS.get(config_key, {}).get("reason", "Unknown")

        status_icon = "✓" if status["status"] == "OK" else "✗"
        logger.info(
            f"  {status_icon} {config_key}: {status['completed_games']} games "
            f"({status['status']}) - {reason}"
        )

        if status["status"] in ("EMPTY", "SPARSE", "LOW", "MISSING"):
            needs_selfplay.append((config_key, status, games_needed))

    if args.check_only:
        logger.info("\n[Check only mode - not triggering selfplay]")
        return

    if not needs_selfplay:
        logger.info("\nAll configs have sufficient data!")
        return

    # Find leader
    leader = args.leader
    if not leader:
        logger.info("\nSearching for P2P leader...")
        leader = find_p2p_leader()
        if not leader:
            logger.error("Could not find P2P leader. Use --leader to specify.")
            logger.error("Example: --leader 100.64.x.x or --leader nebius-backbone-1")
            sys.exit(1)

    logger.info(f"\nUsing leader: {leader}")

    # Trigger selfplay for each config
    logger.info("\nTriggering selfplay jobs:")
    logger.info("-" * 40)

    success_count = 0
    for config_key, status, games_needed in needs_selfplay:
        board_type, num_players = config_key.rsplit("_", 1)
        num_players = int(num_players.rstrip("p"))

        logger.info(f"\n{config_key}: {games_needed} games")

        if trigger_selfplay_via_p2p(
            leader_host=leader,
            board_type=board_type,
            num_players=num_players,
            num_games=games_needed,
            dry_run=args.dry_run,
        ):
            success_count += 1

    logger.info("\n" + "=" * 60)
    logger.info(f"Summary: {success_count}/{len(needs_selfplay)} jobs triggered")

    if args.dry_run:
        logger.info("\n[DRY RUN - no actual jobs were dispatched]")
        logger.info("Remove --dry-run to actually trigger selfplay")


if __name__ == "__main__":
    main()
