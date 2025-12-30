#!/usr/bin/env python3
"""Run initial round-robin tournaments for all 12 configurations.

This script starts tournaments for each canonical model against baselines
(random, heuristic) to populate the Elo match history.

Usage:
    python scripts/run_initial_tournaments.py

    # Check status:
    curl http://localhost:8770/tournament/status | jq
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import requests

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# All 12 configurations
CONFIGS = [
    ("hex8", 2), ("hex8", 3), ("hex8", 4),
    ("square8", 2), ("square8", 3), ("square8", 4),
    ("square19", 2), ("square19", 3), ("square19", 4),
    ("hexagonal", 2), ("hexagonal", 3), ("hexagonal", 4),
]

# Default P2P endpoints to try
P2P_ENDPOINTS = [
    "http://127.0.0.1:8770",
    "http://localhost:8770",
    "http://100.107.168.125:8770",  # mac-studio
]


def find_leader(endpoints: list[str]) -> str | None:
    """Find the P2P leader endpoint."""
    for endpoint in endpoints:
        try:
            resp = requests.get(f"{endpoint}/status", timeout=5)
            if resp.ok:
                data = resp.json()
                leader_id = data.get("leader_id")
                if leader_id:
                    # If this node is leader, use it
                    if data.get("role") == "leader":
                        return endpoint
                    # Otherwise, find leader's endpoint from peers
                    peers = data.get("peers", {})
                    for peer_id, peer_info in peers.items():
                        if peer_id == leader_id:
                            leader_url = peer_info.get("url", "")
                            if leader_url:
                                return leader_url.rstrip("/")
                    # Fallback: use the first endpoint if it knows the leader
                    return endpoint
        except (requests.RequestException, ValueError, KeyError):
            continue
    return None


def run_tournament(endpoint: str, board_type: str, num_players: int, games_per_pairing: int = 5) -> str | None:
    """Start a tournament for the given configuration.

    Returns job_id if successful, None otherwise.
    """
    model_id = f"canonical_{board_type}_{num_players}p"
    agents = [model_id, "random", "heuristic"]

    try:
        resp = requests.post(
            f"{endpoint}/tournament/start",
            json={
                "board_type": board_type,
                "num_players": num_players,
                "agent_ids": agents,
                "games_per_pairing": games_per_pairing,
            },
            timeout=30,
        )

        if resp.ok:
            result = resp.json()
            job_id = result.get("job_id")
            if job_id:
                logger.info(f"Started tournament {job_id} for {model_id} ({len(agents)} agents, {games_per_pairing} games/pair)")
                return job_id
            else:
                logger.warning(f"Tournament started but no job_id returned: {result}")
                return None
        else:
            logger.error(f"Failed to start tournament for {model_id}: {resp.status_code} - {resp.text[:200]}")
            return None
    except (requests.RequestException, ValueError, KeyError) as e:
        logger.error(f"Error starting tournament for {model_id}: {e}")
        return None


def check_tournament_status(endpoint: str, job_ids: list[str]) -> dict:
    """Check status of running tournaments."""
    try:
        resp = requests.get(f"{endpoint}/tournament/status", timeout=10)
        if resp.ok:
            return resp.json()
    except (requests.RequestException, ValueError):
        pass
    return {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run initial tournaments for all 12 configurations")
    parser.add_argument("--endpoint", type=str, help="P2P endpoint (auto-detect if not specified)")
    parser.add_argument("--games-per-pairing", type=int, default=5, help="Games per model pairing (default: 5)")
    parser.add_argument("--configs", type=str, help="Comma-separated configs to run (e.g., hex8_2p,square8_4p)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    parser.add_argument("--status", action="store_true", help="Just show current tournament status")
    args = parser.parse_args()

    # Find P2P endpoint
    if args.endpoint:
        endpoint = args.endpoint
    else:
        endpoint = find_leader(P2P_ENDPOINTS)
        if not endpoint:
            logger.error("Could not find P2P cluster. Specify --endpoint manually.")
            return 1
        logger.info(f"Using P2P endpoint: {endpoint}")

    # If just checking status
    if args.status:
        status = check_tournament_status(endpoint, [])
        if status:
            logger.info(f"Active tournaments: {len(status)}")
            for job_id, state in status.items():
                completed = state.get("completed_matches", 0)
                total = state.get("total_matches", 0)
                status_str = state.get("status", "unknown")
                logger.info(f"  {job_id}: {completed}/{total} matches ({status_str})")
        else:
            logger.info("No active tournaments")
        return 0

    # Parse config filter
    if args.configs:
        filter_configs = set(args.configs.split(","))
        configs = [(bt, np) for bt, np in CONFIGS if f"{bt}_{np}p" in filter_configs]
        if not configs:
            logger.error(f"No matching configs found for: {args.configs}")
            return 1
    else:
        configs = CONFIGS

    if args.dry_run:
        logger.info("=== DRY RUN ===")
        for board_type, num_players in configs:
            model_id = f"canonical_{board_type}_{num_players}p"
            logger.info(f"Would start tournament for {model_id}")
        return 0

    # Start tournaments
    job_ids = []
    for board_type, num_players in configs:
        job_id = run_tournament(endpoint, board_type, num_players, args.games_per_pairing)
        if job_id:
            job_ids.append(job_id)
        time.sleep(2)  # Stagger starts to avoid overwhelming cluster

    logger.info(f"\nStarted {len(job_ids)}/{len(configs)} tournaments")

    if job_ids:
        logger.info("\nCheck status with:")
        logger.info(f"  curl {endpoint}/tournament/status | jq")
        logger.info(f"  python scripts/run_initial_tournaments.py --status")

    return 0 if len(job_ids) == len(configs) else 1


if __name__ == "__main__":
    sys.exit(main())
