#!/usr/bin/env python3
"""
Discover and configure Mac nodes as coordinator-only P2P cluster members.

This script:
1. Discovers Macs on the local network via mDNS/Bonjour
2. Checks SSH accessibility
3. Updates to latest RingRift code
4. Configures as coordinator-only P2P nodes
5. Updates distributed_hosts.yaml

Usage:
    # Discover and list available Macs
    python scripts/setup_mac_coordinators.py --discover

    # Set up all accessible Macs as coordinators
    python scripts/setup_mac_coordinators.py --setup

    # Set up specific hosts
    python scripts/setup_mac_coordinators.py --setup --hosts "mac-studio,m1-pro"

    # Check status of configured Mac coordinators
    python scripts/setup_mac_coordinators.py --status

    # Dry run (show what would be done)
    python scripts/setup_mac_coordinators.py --setup --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

CONFIG_PATH = Path(__file__).parent.parent / "config" / "distributed_hosts.yaml"
SSH_TIMEOUT = 10
SETUP_TIMEOUT = 120

# Known Mac SSH configs from ~/.ssh/config
KNOWN_MAC_HOSTS = {
    "intel-mac": "10.0.0.89",
    "m1-pro": "10.0.0.170",
    "mac-studio": "100.107.168.125",  # Tailscale IP
}

# Default coordinator config template
COORDINATOR_TEMPLATE = {
    "ssh_user": "armand",
    "ssh_key": "~/.ssh/id_ed25519",
    "ringrift_path": "~/Development/RingRift/ai-service",
    "venv_activate": "source ~/Development/RingRift/ai-service/venv/bin/activate",
    "memory_gb": 16,
    "cpus": 8,
    "gpu": "none",
    "role": "coordinator",
    "status": "ready",
    "selfplay_enabled": False,
    "training_enabled": False,
    "gauntlet_enabled": False,
    "export_enabled": False,
    "p2p_enabled": True,
    "p2p_voter": True,  # Coordinator-only nodes can be voters
    "skip_sync_receive": True,  # Don't receive heavy data
}


@dataclass
class MacNode:
    """Represents a discovered Mac node."""

    name: str
    ip: str
    ssh_user: str = "armand"
    ssh_port: int = 22
    ssh_key: str | None = None
    accessible: bool = False
    chip: str = ""
    os_version: str = ""
    has_ringrift: bool = False
    ringrift_path: str = ""
    git_commit: str = ""
    p2p_running: bool = False
    error: str = ""

    def to_config(self) -> dict[str, Any]:
        """Convert to distributed_hosts.yaml config entry."""
        config = COORDINATOR_TEMPLATE.copy()
        config["ssh_host"] = self.ip
        config["ssh_user"] = self.ssh_user
        if self.ssh_key:
            config["ssh_key"] = self.ssh_key
        if self.ringrift_path:
            config["ringrift_path"] = self.ringrift_path
        if self.chip:
            # Detect Apple Silicon for MPS
            if "Apple" in self.chip and ("M1" in self.chip or "M2" in self.chip or "M3" in self.chip or "M4" in self.chip):
                config["gpu"] = f"{self.chip.split()[-2]} {self.chip.split()[-1]} (MPS)"
        return config


# =============================================================================
# Discovery
# =============================================================================


def discover_macs_bonjour() -> list[tuple[str, str]]:
    """Discover Macs via Bonjour/mDNS."""
    discovered = []

    try:
        # Use dns-sd to browse for SSH services
        proc = subprocess.run(
            ["dns-sd", "-B", "_ssh._tcp", "local"],
            capture_output=True,
            text=True,
            timeout=3,
        )
    except subprocess.TimeoutExpired:
        pass  # Expected - dns-sd runs forever
    except Exception as e:
        logger.debug(f"Bonjour discovery failed: {e}")

    # Parse mDNS results for Mac-like names
    # This is a simplified approach - real parsing would need dns-sd -L
    for name_pattern in ["MacBook", "iMac", "Mac-Pro", "Mac-Studio", "Mac-mini"]:
        try:
            # Try to resolve common Mac hostnames
            for suffix in ["", "-2", "-3", "-4"]:
                hostname = f"{name_pattern}{suffix}.local"
                result = subprocess.run(
                    ["ping", "-c", "1", "-W", "1", hostname],
                    capture_output=True,
                    timeout=2,
                )
                if result.returncode == 0:
                    # Extract IP from ping output
                    match = re.search(r"(\d+\.\d+\.\d+\.\d+)", result.stdout.decode())
                    if match:
                        ip = match.group(1)
                        if ip != "127.0.0.1":  # Skip localhost
                            discovered.append((hostname, ip))
        except Exception:
            pass

    return discovered


def discover_macs_known() -> list[tuple[str, str]]:
    """Get known Macs from SSH config and hardcoded list."""
    discovered = []

    # Parse ~/.ssh/config for Mac hosts
    ssh_config_path = Path.home() / ".ssh" / "config"
    if ssh_config_path.exists():
        try:
            content = ssh_config_path.read_text()
            current_host = None
            current_hostname = None

            for line in content.split("\n"):
                line = line.strip()
                if line.lower().startswith("host "):
                    if current_host and current_hostname:
                        if any(m in current_host.lower() for m in ["mac", "imac", "m1", "m2", "m3", "m4"]):
                            discovered.append((current_host, current_hostname))
                    current_host = line.split()[1] if len(line.split()) > 1 else None
                    current_hostname = None
                elif line.lower().startswith("hostname ") and current_host:
                    current_hostname = line.split()[1] if len(line.split()) > 1 else None

            # Don't forget last host
            if current_host and current_hostname:
                if any(m in current_host.lower() for m in ["mac", "imac", "m1", "m2", "m3", "m4"]):
                    discovered.append((current_host, current_hostname))
        except Exception as e:
            logger.debug(f"Failed to parse SSH config: {e}")

    # Add hardcoded known hosts
    for name, ip in KNOWN_MAC_HOSTS.items():
        if (name, ip) not in discovered:
            discovered.append((name, ip))

    return discovered


def discover_macs_tailscale() -> list[tuple[str, str]]:
    """Discover Macs via Tailscale."""
    discovered = []

    try:
        result = subprocess.run(
            ["tailscale", "status", "--json"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            peers = data.get("Peer", {})
            for peer_id, peer_info in peers.items():
                os_type = peer_info.get("OS", "").lower()
                if os_type in ("macos", "darwin"):
                    hostname = peer_info.get("HostName", "")
                    ips = peer_info.get("TailscaleIPs", [])
                    if hostname and ips:
                        # Use first Tailscale IP
                        discovered.append((hostname, ips[0]))
    except Exception as e:
        logger.debug(f"Tailscale discovery failed: {e}")

    return discovered


async def check_mac_accessibility(mac: MacNode) -> MacNode:
    """Check if a Mac is accessible and get its details."""
    try:
        # Build SSH command
        ssh_args = ["ssh", "-o", f"ConnectTimeout={SSH_TIMEOUT}", "-o", "BatchMode=yes"]
        if mac.ssh_key:
            ssh_args.extend(["-i", mac.ssh_key])
        ssh_args.append(f"{mac.ssh_user}@{mac.ip}")

        # Command to gather info
        check_cmd = """
echo "CHIP:$(sysctl -n machdep.cpu.brand_string 2>/dev/null || uname -m)"
echo "OS:$(sw_vers -productVersion 2>/dev/null)"
echo "USER:$(whoami)"
echo "HOST:$(hostname)"

# Check for RingRift
if [ -d ~/Development/RingRift/ai-service ]; then
    echo "RINGRIFT:~/Development/RingRift/ai-service"
    cd ~/Development/RingRift/ai-service
    echo "COMMIT:$(git rev-parse --short HEAD 2>/dev/null)"
    echo "BRANCH:$(git branch --show-current 2>/dev/null)"
elif [ -d ~/ringrift/ai-service ]; then
    echo "RINGRIFT:~/ringrift/ai-service"
    cd ~/ringrift/ai-service
    echo "COMMIT:$(git rev-parse --short HEAD 2>/dev/null)"
else
    echo "RINGRIFT:NONE"
fi

# Check P2P status
pgrep -f p2p_orchestrator >/dev/null && echo "P2P:RUNNING" || echo "P2P:STOPPED"
"""
        ssh_args.append(check_cmd)

        proc = await asyncio.create_subprocess_exec(
            *ssh_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=SSH_TIMEOUT + 5,
        )

        if proc.returncode == 0:
            mac.accessible = True
            output = stdout.decode()

            for line in output.split("\n"):
                if line.startswith("CHIP:"):
                    mac.chip = line[5:].strip()
                elif line.startswith("OS:"):
                    mac.os_version = line[3:].strip()
                elif line.startswith("RINGRIFT:"):
                    path = line[9:].strip()
                    if path != "NONE":
                        mac.has_ringrift = True
                        mac.ringrift_path = path
                elif line.startswith("COMMIT:"):
                    mac.git_commit = line[7:].strip()
                elif line.startswith("P2P:"):
                    mac.p2p_running = line[4:].strip() == "RUNNING"
        else:
            mac.error = stderr.decode().strip()[:100]

    except asyncio.TimeoutError:
        mac.error = "Connection timed out"
    except Exception as e:
        mac.error = str(e)[:100]

    return mac


# =============================================================================
# Setup
# =============================================================================


async def update_ringrift(mac: MacNode, dry_run: bool = False) -> bool:
    """Update RingRift to latest on a Mac."""
    if not mac.accessible or not mac.has_ringrift:
        return False

    logger.info(f"[{mac.name}] Updating RingRift...")

    cmd = f"cd {mac.ringrift_path} && git fetch origin && git pull --ff-only"

    if dry_run:
        logger.info(f"[DRY RUN] Would execute: {cmd}")
        return True

    try:
        ssh_args = ["ssh", "-o", f"ConnectTimeout={SSH_TIMEOUT}", "-o", "BatchMode=yes"]
        if mac.ssh_key:
            ssh_args.extend(["-i", mac.ssh_key])
        ssh_args.extend([f"{mac.ssh_user}@{mac.ip}", cmd])

        proc = await asyncio.create_subprocess_exec(
            *ssh_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=SETUP_TIMEOUT,
        )

        if proc.returncode == 0:
            logger.info(f"[{mac.name}] Updated successfully")
            return True
        else:
            logger.error(f"[{mac.name}] Update failed: {stderr.decode()[:200]}")
            return False

    except Exception as e:
        logger.error(f"[{mac.name}] Update error: {e}")
        return False


async def start_p2p_coordinator(mac: MacNode, dry_run: bool = False) -> bool:
    """Start P2P orchestrator in coordinator-only mode."""
    if not mac.accessible or not mac.has_ringrift:
        return False

    if mac.p2p_running:
        logger.info(f"[{mac.name}] P2P already running")
        return True

    logger.info(f"[{mac.name}] Starting P2P orchestrator as coordinator...")

    # Build startup command
    cmd = f"""
cd {mac.ringrift_path}
export PYTHONPATH=.
export RINGRIFT_NODE_ID={mac.name}
export RINGRIFT_IS_COORDINATOR=true
export RINGRIFT_P2P_COORDINATOR_ONLY=true

# Activate venv if exists
[ -f venv/bin/activate ] && source venv/bin/activate

# Start P2P in background
nohup python scripts/p2p_orchestrator.py \\
    --node-id {mac.name} \\
    --coordinator-only \\
    > logs/p2p_{mac.name}.log 2>&1 &

echo "Started with PID: $!"
sleep 2
pgrep -f p2p_orchestrator && echo "STARTED" || echo "FAILED"
"""

    if dry_run:
        logger.info(f"[DRY RUN] Would start P2P on {mac.name}")
        return True

    try:
        ssh_args = ["ssh", "-o", f"ConnectTimeout={SSH_TIMEOUT}", "-o", "BatchMode=yes"]
        if mac.ssh_key:
            ssh_args.extend(["-i", mac.ssh_key])
        ssh_args.extend([f"{mac.ssh_user}@{mac.ip}", cmd])

        proc = await asyncio.create_subprocess_exec(
            *ssh_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=SETUP_TIMEOUT,
        )

        output = stdout.decode()
        if "STARTED" in output:
            logger.info(f"[{mac.name}] P2P started successfully")
            return True
        else:
            logger.error(f"[{mac.name}] P2P start failed: {output} {stderr.decode()}")
            return False

    except Exception as e:
        logger.error(f"[{mac.name}] P2P start error: {e}")
        return False


def update_hosts_config(macs: list[MacNode], dry_run: bool = False) -> bool:
    """Update distributed_hosts.yaml with new Mac coordinators."""
    if not macs:
        return True

    try:
        # Load existing config
        with open(CONFIG_PATH) as f:
            config = yaml.safe_load(f)

        hosts = config.get("hosts", {})
        voters = config.get("p2p_voters", [])

        changes = []

        for mac in macs:
            if not mac.accessible:
                continue

            # Add or update host entry
            if mac.name not in hosts:
                hosts[mac.name] = mac.to_config()
                changes.append(f"Added: {mac.name}")
            else:
                # Update existing entry
                existing = hosts[mac.name]
                new_config = mac.to_config()
                for key in ["ssh_host", "chip", "role", "p2p_enabled", "p2p_voter"]:
                    if key in new_config:
                        existing[key] = new_config[key]
                changes.append(f"Updated: {mac.name}")

            # Add to voters if not already
            if mac.name not in voters:
                voters.append(mac.name)
                changes.append(f"Added voter: {mac.name}")

        if not changes:
            logger.info("No config changes needed")
            return True

        config["hosts"] = hosts
        config["p2p_voters"] = voters

        if dry_run:
            logger.info(f"[DRY RUN] Would make changes: {changes}")
            return True

        # Backup and write
        backup_path = CONFIG_PATH.with_suffix(f".yaml.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        import shutil
        shutil.copy(CONFIG_PATH, backup_path)

        with open(CONFIG_PATH, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Config updated: {changes}")
        return True

    except Exception as e:
        logger.error(f"Failed to update config: {e}")
        return False


# =============================================================================
# Main Commands
# =============================================================================


async def cmd_discover() -> list[MacNode]:
    """Discover all accessible Macs."""
    logger.info("Discovering Macs on network...")

    # Collect candidates from all sources
    candidates = set()
    candidates.update(discover_macs_bonjour())
    candidates.update(discover_macs_known())
    candidates.update(discover_macs_tailscale())

    logger.info(f"Found {len(candidates)} candidates")

    # Create MacNode objects and check accessibility
    macs = []
    for name, ip in candidates:
        mac = MacNode(name=name, ip=ip)

        # Try to find SSH key
        for key_path in ["~/.ssh/id_ed25519", "~/.ssh/id_cluster", "~/.ssh/id_rsa"]:
            expanded = os.path.expanduser(key_path)
            if os.path.exists(expanded):
                mac.ssh_key = key_path
                break

        macs.append(mac)

    # Check accessibility in parallel
    if macs:
        logger.info("Checking accessibility...")
        macs = await asyncio.gather(*[check_mac_accessibility(m) for m in macs])

    return list(macs)


async def cmd_status() -> None:
    """Show status of configured Mac coordinators."""
    logger.info("Checking status of Mac coordinators...")

    # Load config
    try:
        with open(CONFIG_PATH) as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return

    hosts = config.get("hosts", {})
    mac_hosts = {
        name: cfg
        for name, cfg in hosts.items()
        if "mac" in name.lower() or cfg.get("role") == "coordinator"
    }

    if not mac_hosts:
        logger.info("No Mac coordinators configured")
        return

    # Check each
    for name, cfg in mac_hosts.items():
        mac = MacNode(
            name=name,
            ip=cfg.get("ssh_host", ""),
            ssh_user=cfg.get("ssh_user", "armand"),
            ssh_key=cfg.get("ssh_key"),
        )
        mac = await check_mac_accessibility(mac)

        status = "ONLINE" if mac.accessible else "OFFLINE"
        p2p = "P2P Running" if mac.p2p_running else "P2P Stopped"
        chip = mac.chip if mac.chip else cfg.get("gpu", "unknown")

        print(f"  {name:20} [{status:7}] {chip:30} {p2p}")
        if mac.git_commit:
            print(f"  {' ':20} Commit: {mac.git_commit}")
        if mac.error:
            print(f"  {' ':20} Error: {mac.error}")


async def cmd_setup(hosts: list[str] | None, dry_run: bool = False) -> None:
    """Set up Macs as coordinator-only P2P nodes."""
    # Discover Macs
    macs = await cmd_discover()

    # Filter by requested hosts if specified
    if hosts:
        host_set = set(h.strip() for h in hosts)
        macs = [m for m in macs if m.name in host_set or m.ip in host_set]

    # Filter to accessible ones
    accessible = [m for m in macs if m.accessible]

    if not accessible:
        logger.error("No accessible Macs found")
        return

    logger.info(f"Setting up {len(accessible)} Mac(s) as coordinators...")

    # Print status
    print("\n" + "=" * 70)
    print("DISCOVERED MACS")
    print("=" * 70)
    for mac in macs:
        status = "ACCESSIBLE" if mac.accessible else f"OFFLINE: {mac.error}"
        ringrift = "RingRift OK" if mac.has_ringrift else "No RingRift"
        print(f"  {mac.name:20} {mac.ip:18} [{status}] {ringrift}")
    print()

    # Update RingRift on each
    for mac in accessible:
        if mac.has_ringrift:
            await update_ringrift(mac, dry_run)
        else:
            logger.warning(f"[{mac.name}] No RingRift installation - manual setup needed")

    # Start P2P
    for mac in accessible:
        if mac.has_ringrift:
            await start_p2p_coordinator(mac, dry_run)

    # Update config
    update_hosts_config(accessible, dry_run)

    print("\n" + "=" * 70)
    print("SETUP COMPLETE")
    print("=" * 70)
    print(f"Configured {len(accessible)} Mac coordinator(s)")
    if not dry_run:
        print(f"Config updated: {CONFIG_PATH}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Set up Mac nodes as P2P coordinators",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--discover",
        action="store_true",
        help="Discover Macs on network and show status",
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Set up discovered Macs as coordinators",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show status of configured Mac coordinators",
    )
    parser.add_argument(
        "--hosts",
        type=str,
        help="Comma-separated list of specific hosts to set up",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not any([args.discover, args.setup, args.status]):
        parser.print_help()
        return 1

    if args.discover:
        macs = asyncio.run(cmd_discover())
        print("\n" + "=" * 70)
        print("DISCOVERED MACS")
        print("=" * 70)
        for mac in macs:
            status = "ACCESSIBLE" if mac.accessible else "OFFLINE"
            print(f"  {mac.name:20} {mac.ip:18} [{status}]")
            if mac.accessible:
                print(f"  {' ':20} Chip: {mac.chip}")
                print(f"  {' ':20} OS: macOS {mac.os_version}")
                if mac.has_ringrift:
                    print(f"  {' ':20} RingRift: {mac.ringrift_path} @ {mac.git_commit}")
                print(f"  {' ':20} P2P: {'Running' if mac.p2p_running else 'Stopped'}")
            elif mac.error:
                print(f"  {' ':20} Error: {mac.error}")
            print()
        print(f"Total: {len(macs)} found, {sum(1 for m in macs if m.accessible)} accessible")
        return 0

    if args.status:
        asyncio.run(cmd_status())
        return 0

    if args.setup:
        hosts = args.hosts.split(",") if args.hosts else None
        asyncio.run(cmd_setup(hosts, args.dry_run))
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
