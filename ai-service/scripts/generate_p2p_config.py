#!/usr/bin/env python3
"""Generate P2P peer configuration from distributed_hosts.yaml.

This script reads distributed_hosts.yaml and generates:
1. The full peer list for P2P orchestrator --peers argument
2. A node.conf snippet with P2P_PEERS variable
3. Can update an existing node.conf file in place

Usage:
    # Print peer list
    python scripts/generate_p2p_config.py --peers

    # Print voter node IDs
    python scripts/generate_p2p_config.py --voters

    # Generate node.conf snippet
    python scripts/generate_p2p_config.py --node-conf

    # Update an existing node.conf file (for systemd integration)
    python scripts/generate_p2p_config.py --update-node-conf /etc/ringrift/node.conf
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

try:
    import yaml
except ImportError:
    print("Error: PyYAML not installed. Run: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

from scripts.lib.paths import CONFIG_DIR


def load_distributed_hosts(config_path: Optional[Path] = None) -> Dict:
    """Load distributed_hosts.yaml."""
    if config_path is None:
        config_path = CONFIG_DIR / "distributed_hosts.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def get_voters(hosts: Dict) -> List[str]:
    """Extract voter node IDs from hosts config."""
    voters = []
    for node_id, cfg in hosts.items():
        if not isinstance(cfg, dict):
            continue
        is_voter = cfg.get("p2p_voter", False)
        if is_voter is True or str(is_voter).lower() in {"true", "yes", "1"}:
            # Skip nodes with status != ready
            status = cfg.get("status", "ready")
            if status in {"ready", "setup"}:
                voters.append(node_id)
    return sorted(voters)


def get_peer_urls(hosts: Dict, port: int = 8770) -> List[str]:
    """Generate peer URLs from hosts config."""
    peers = []
    for node_id, cfg in hosts.items():
        if not isinstance(cfg, dict):
            continue

        # Skip offline/stopped/disabled nodes
        status = cfg.get("status", "ready")
        if status in {"offline", "stopped", "disabled"}:
            continue

        # Prefer tailscale_ip for reliable connectivity
        ip = cfg.get("tailscale_ip") or cfg.get("ssh_host")
        if ip:
            # Validate it looks like an IP (not a hostname like ssh6.vast.ai)
            if re.match(r'^\d+\.\d+\.\d+\.\d+$', str(ip)):
                peers.append(f"http://{ip}:{port}")

    return sorted(set(peers))


def get_voter_peer_urls(hosts: Dict, port: int = 8770) -> List[str]:
    """Generate peer URLs only for voter nodes."""
    voters = set(get_voters(hosts))
    peers = []
    for node_id, cfg in hosts.items():
        if node_id not in voters:
            continue
        if not isinstance(cfg, dict):
            continue

        ip = cfg.get("tailscale_ip") or cfg.get("ssh_host")
        if ip:
            if re.match(r'^\d+\.\d+\.\d+\.\d+$', str(ip)):
                peers.append(f"http://{ip}:{port}")

    return sorted(set(peers))


def update_node_conf(path: Path, peers: List[str]) -> None:
    """Update node.conf with P2P_PEERS variable."""
    if not path.exists():
        raise FileNotFoundError(f"node.conf not found: {path}")

    content = path.read_text()
    peers_str = ",".join(peers)

    # Check if P2P_PEERS already exists
    if re.search(r'^P2P_PEERS=', content, re.MULTILINE):
        # Update existing
        content = re.sub(
            r'^P2P_PEERS=.*$',
            f'P2P_PEERS={peers_str}',
            content,
            flags=re.MULTILINE
        )
    else:
        # Add new
        content = content.rstrip() + f"\nP2P_PEERS={peers_str}\n"

    path.write_text(content)
    print(f"Updated {path} with {len(peers)} peers")


def main():
    parser = argparse.ArgumentParser(description="Generate P2P peer configuration")
    parser.add_argument("--config", type=Path, help="Path to distributed_hosts.yaml")
    parser.add_argument("--port", type=int, default=8770, help="P2P port (default: 8770)")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--peers", action="store_true", help="Print all peer URLs")
    group.add_argument("--voter-peers", action="store_true", help="Print voter peer URLs only")
    group.add_argument("--voters", action="store_true", help="Print voter node IDs")
    group.add_argument("--node-conf", action="store_true", help="Print node.conf snippet")
    group.add_argument("--update-node-conf", type=Path, metavar="PATH",
                      help="Update existing node.conf file")

    args = parser.parse_args()

    try:
        data = load_distributed_hosts(args.config)
        hosts = data.get("hosts", {})

        if args.peers:
            peers = get_peer_urls(hosts, args.port)
            print(",".join(peers))

        elif args.voter_peers:
            peers = get_voter_peer_urls(hosts, args.port)
            print(",".join(peers))

        elif args.voters:
            voters = get_voters(hosts)
            print(",".join(voters))

        elif args.node_conf:
            peers = get_voter_peer_urls(hosts, args.port)
            print(f"P2P_PEERS={','.join(peers)}")

        elif args.update_node_conf:
            peers = get_voter_peer_urls(hosts, args.port)
            update_node_conf(args.update_node_conf, peers)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
