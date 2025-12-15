#!/usr/bin/env python3
"""Cross-cluster Elo validation script.

Validates that Elo ratings are consistent across the cluster by:
1. Querying Elo leaderboards from multiple nodes
2. Comparing ratings to detect divergence
3. Alerting if ratings don't match

Usage:
    python scripts/validate_cluster_elo.py              # Run validation
    python scripts/validate_cluster_elo.py --fix        # Sync from Mac Studio to cluster
    python scripts/validate_cluster_elo.py --verbose    # Show detailed comparison
"""

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import urllib.request
import urllib.error

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]

# Mac Studio is the authoritative source for Elo ratings
MAC_STUDIO_ELO_URL = "http://100.107.168.125:8770/api/elo/leaderboard"
LOCAL_ELO_DB = AI_SERVICE_ROOT / "data" / "unified_elo.db"

# Cluster nodes to validate
CLUSTER_NODES = [
    ("mac-studio", "100.107.168.125"),
    ("lambda-h100", "209.20.157.81"),
    ("lambda-2xh100", "192.222.53.22"),
    ("lambda-gh200-a", "192.222.51.29"),
]

# Maximum allowed Elo difference before flagging divergence
ELO_DIVERGENCE_THRESHOLD = 50.0


def fetch_elo_from_node(host: str, port: int = 8770, timeout: int = 10) -> Optional[Dict[str, Any]]:
    """Fetch Elo leaderboard from a cluster node."""
    url = f"http://{host}:{port}/api/elo/leaderboard"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "RingRift-Validator/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.URLError as e:
        print(f"  Warning: Could not reach {host}:{port} - {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"  Warning: Invalid JSON from {host}:{port} - {e}")
        return None
    except Exception as e:
        print(f"  Warning: Error fetching from {host}:{port} - {e}")
        return None


def compare_leaderboards(
    source: Dict[str, Any],
    target: Dict[str, Any],
    threshold: float = ELO_DIVERGENCE_THRESHOLD
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Compare two Elo leaderboards.

    Returns (matches, diverged, missing_in_target).
    """
    matches = []
    diverged = []
    missing_in_target = []

    source_models = {m["model_id"]: m for m in source.get("leaderboard", [])}
    target_models = {m["model_id"]: m for m in target.get("leaderboard", [])}

    for model_id, source_entry in source_models.items():
        if model_id not in target_models:
            missing_in_target.append(source_entry)
            continue

        target_entry = target_models[model_id]
        source_elo = source_entry.get("elo", 1500)
        target_elo = target_entry.get("elo", 1500)
        diff = abs(source_elo - target_elo)

        if diff > threshold:
            diverged.append({
                "model_id": model_id,
                "source_elo": source_elo,
                "target_elo": target_elo,
                "diff": diff
            })
        else:
            matches.append({
                "model_id": model_id,
                "source_elo": source_elo,
                "target_elo": target_elo,
                "diff": diff
            })

    return matches, diverged, missing_in_target


def sync_elo_to_node(host: str, source_db: Path) -> bool:
    """Sync Elo database from source to a remote node."""
    try:
        cmd = [
            "scp",
            "-o", "ConnectTimeout=10",
            str(source_db),
            f"ubuntu@{host}:~/ringrift/ai-service/data/unified_elo.db"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return result.returncode == 0
    except Exception as e:
        print(f"  Error syncing to {host}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Cross-cluster Elo validation")
    parser.add_argument("--fix", action="store_true", help="Sync from Mac Studio to cluster")
    parser.add_argument("--verbose", action="store_true", help="Show detailed comparison")
    parser.add_argument("--threshold", type=float, default=ELO_DIVERGENCE_THRESHOLD,
                        help=f"Elo divergence threshold (default: {ELO_DIVERGENCE_THRESHOLD})")
    args = parser.parse_args()

    print("Cross-Cluster Elo Validation")
    print("=" * 50)

    # Fetch from authoritative source (Mac Studio)
    print(f"\nFetching authoritative Elo from Mac Studio...")
    source_elo = fetch_elo_from_node(MAC_STUDIO_ELO_URL.split("//")[1].split(":")[0])
    if not source_elo:
        print("ERROR: Could not reach Mac Studio (authoritative source)")
        return 1

    source_count = len(source_elo.get("leaderboard", []))
    print(f"  Source has {source_count} models in leaderboard")

    # Validate each cluster node
    print(f"\nValidating {len(CLUSTER_NODES)} cluster nodes...")
    all_valid = True
    nodes_to_sync = []

    for node_name, host in CLUSTER_NODES:
        if host == MAC_STUDIO_ELO_URL.split("//")[1].split(":")[0]:
            continue  # Skip source

        print(f"\n  Checking {node_name} ({host})...")
        node_elo = fetch_elo_from_node(host)

        if not node_elo:
            print(f"    Status: UNREACHABLE")
            nodes_to_sync.append((node_name, host))
            all_valid = False
            continue

        matches, diverged, missing = compare_leaderboards(source_elo, node_elo, args.threshold)

        if diverged or missing:
            all_valid = False
            nodes_to_sync.append((node_name, host))
            print(f"    Status: DIVERGED")
            print(f"    Matches: {len(matches)}, Diverged: {len(diverged)}, Missing: {len(missing)}")

            if args.verbose and diverged:
                print(f"    Diverged models:")
                for d in diverged[:5]:
                    print(f"      {d['model_id']}: {d['source_elo']:.0f} vs {d['target_elo']:.0f} (diff: {d['diff']:.0f})")
        else:
            print(f"    Status: OK ({len(matches)} models match)")

    # Summary
    print(f"\n{'=' * 50}")
    if all_valid:
        print("RESULT: All cluster nodes have consistent Elo ratings")
        return 0
    else:
        print(f"RESULT: {len(nodes_to_sync)} nodes have divergent Elo ratings")

        if args.fix and LOCAL_ELO_DB.exists():
            print(f"\nSyncing Elo database to divergent nodes...")
            for node_name, host in nodes_to_sync:
                print(f"  Syncing to {node_name}...")
                if sync_elo_to_node(host, LOCAL_ELO_DB):
                    print(f"    Success")
                else:
                    print(f"    Failed")
        elif args.fix:
            print(f"ERROR: Local Elo database not found: {LOCAL_ELO_DB}")
            return 1
        else:
            print("\nRun with --fix to sync Elo database from Mac Studio")

        return 1


if __name__ == "__main__":
    sys.exit(main())
