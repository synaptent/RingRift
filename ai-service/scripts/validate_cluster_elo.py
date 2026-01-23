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

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_SERVICE_ROOT))

# Import unified hosts module
try:
    from scripts.lib.hosts import get_active_hosts
    USE_LIB_HOSTS = True
except ImportError:
    USE_LIB_HOSTS = False

# Fallback to app.sync module
try:
    from app.sync.cluster_hosts import (
        get_active_nodes,
        get_coordinator_node,
        get_elo_sync_config,
    )
    USE_SHARED_CONFIG = True
except ImportError:
    USE_SHARED_CONFIG = False

LOCAL_ELO_DB = AI_SERVICE_ROOT / "data" / "unified_elo.db"


def get_cluster_nodes_for_validation() -> list[tuple[str, str]]:
    """Get cluster nodes for validation.

    Uses scripts.lib.hosts module (preferred), falls back to app.sync,
    then to direct YAML loading.
    """
    # Preferred: Use unified hosts module
    if USE_LIB_HOSTS:
        hosts = get_active_hosts()
        nodes = []
        for h in hosts:
            ip = h.tailscale_ip or h.ssh_host
            if ip and ip.startswith("100."):
                nodes.append((h.name, ip))
        return nodes

    # Fallback: Use app.sync module
    if USE_SHARED_CONFIG:
        nodes = get_active_nodes()
        return [(n.name, n.best_ip) for n in nodes if n.best_ip]

    # Last resort: Load YAML directly
    return _load_hosts_from_yaml()


def _load_hosts_from_yaml() -> list[tuple[str, str]]:
    """Load cluster nodes directly from YAML (legacy fallback)."""
    config_path = Path(__file__).parent.parent / "config" / "distributed_hosts.yaml"
    if not config_path.exists():
        print("[Validation] Warning: No config found at", config_path)
        return []

    try:
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}

        hosts = config.get("hosts", {})
        nodes = []
        for host_id, host_info in hosts.items():
            if host_info.get("status") != "ready":
                continue
            tailscale_ip = host_info.get("tailscale_ip") or host_info.get("ssh_host")
            if tailscale_ip and tailscale_ip.startswith("100."):
                nodes.append((host_id, tailscale_ip))

        return nodes
    except Exception as e:
        print(f"[Validation] Error loading config: {e}")
        return []


def get_coordinator_ip() -> str:
    """Get coordinator IP from shared config or config file."""
    if USE_SHARED_CONFIG:
        coord = get_coordinator_node()
        if coord and coord.best_ip:
            return coord.best_ip

    # Load from config file
    config_path = Path(__file__).parent.parent / "config" / "distributed_hosts.yaml"
    if not config_path.exists():
        print("[Validation] Warning: No config found, using empty coordinator IP")
        return ""

    try:
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}

        coordinator = config.get("elo_sync", {}).get("coordinator", "mac-studio")
        hosts = config.get("hosts", {})

        if coordinator in hosts:
            coord_info = hosts[coordinator]
            tailscale_ip = coord_info.get("tailscale_ip") or coord_info.get("ssh_host")
            if tailscale_ip and tailscale_ip.startswith("100."):
                return tailscale_ip

        return ""
    except Exception as e:
        print(f"[Validation] Error loading coordinator from config: {e}")
        return ""


def get_divergence_threshold() -> float:
    """Get divergence threshold from shared config or fallback."""
    if USE_SHARED_CONFIG:
        config = get_elo_sync_config()
        return float(config.divergence_threshold)
    return 50.0


# Legacy constants for backwards compatibility
MAC_STUDIO_ELO_URL = f"http://{get_coordinator_ip()}:8770/api/elo/leaderboard"
CLUSTER_NODES = get_cluster_nodes_for_validation()
ELO_DIVERGENCE_THRESHOLD = get_divergence_threshold()


def fetch_elo_from_node(host: str, port: int = 8770, timeout: int = 10) -> dict[str, Any] | None:
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
    source: dict[str, Any],
    target: dict[str, Any],
    threshold: float = ELO_DIVERGENCE_THRESHOLD
) -> tuple[list[dict], list[dict], list[dict]]:
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
    print("\nFetching authoritative Elo from Mac Studio...")
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
            print("    Status: UNREACHABLE")
            nodes_to_sync.append((node_name, host))
            all_valid = False
            continue

        matches, diverged, missing = compare_leaderboards(source_elo, node_elo, args.threshold)

        if diverged or missing:
            all_valid = False
            nodes_to_sync.append((node_name, host))
            print("    Status: DIVERGED")
            print(f"    Matches: {len(matches)}, Diverged: {len(diverged)}, Missing: {len(missing)}")

            if args.verbose and diverged:
                print("    Diverged models:")
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
            print("\nSyncing Elo database to divergent nodes...")
            for node_name, host in nodes_to_sync:
                print(f"  Syncing to {node_name}...")
                if sync_elo_to_node(host, LOCAL_ELO_DB):
                    print("    Success")
                else:
                    print("    Failed")
        elif args.fix:
            print(f"ERROR: Local Elo database not found: {LOCAL_ELO_DB}")
            return 1
        else:
            print("\nRun with --fix to sync Elo database from Mac Studio")

        return 1


if __name__ == "__main__":
    sys.exit(main())
