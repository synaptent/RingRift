#!/usr/bin/env python3
"""
Update SSH config with Tailscale IPs from distributed_hosts.yaml and live status.

This script:
1. Reads distributed_hosts.yaml to get the mapping of hostnames to Tailscale IPs
2. Optionally queries `tailscale status` for live IP verification
3. Updates ~/.ssh/config with the correct Tailscale IPs

Usage:
    python scripts/update_ssh_config_from_tailscale.py          # Preview changes
    python scripts/update_ssh_config_from_tailscale.py --apply  # Apply changes
    python scripts/update_ssh_config_from_tailscale.py --live   # Use live tailscale status
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


def parse_tailscale_status() -> dict[str, dict]:
    """Parse `tailscale status` output to get hostname -> IP mappings."""
    try:
        result = subprocess.run(
            ["tailscale", "status"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            print(f"Warning: tailscale status failed: {result.stderr}")
            return {}
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"Warning: Could not run tailscale status: {e}")
        return {}

    nodes = {}
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue

        # Format: 100.xxx.xxx.xxx    hostname    user@  os  status_info
        parts = line.split()
        if len(parts) < 3:
            continue

        ip = parts[0]
        hostname = parts[1]

        # Validate IP format
        if not ip.startswith("100."):
            continue

        # Parse status (active, idle, offline)
        status = "unknown"
        if "active" in line.lower():
            status = "active"
        elif "idle" in line.lower():
            status = "idle"
        elif "offline" in line.lower():
            status = "offline"

        nodes[hostname] = {
            "tailscale_ip": ip,
            "status": status,
        }

    return nodes


def load_distributed_hosts(yaml_path: Path) -> dict[str, dict]:
    """Load distributed_hosts.yaml and extract hostname -> tailscale_ip mapping."""
    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    hosts = config.get("hosts", {})
    result = {}

    for host_name, host_config in hosts.items():
        tailscale_ip = host_config.get("tailscale_ip")
        ssh_user = host_config.get("ssh_user", "ubuntu")
        ssh_key = host_config.get("ssh_key", "~/.ssh/id_cluster")
        ssh_port = host_config.get("ssh_port", 22)
        status = host_config.get("status", "unknown")

        if tailscale_ip:
            result[host_name] = {
                "tailscale_ip": tailscale_ip,
                "ssh_user": ssh_user,
                "ssh_key": ssh_key,
                "ssh_port": ssh_port,
                "status": status,
            }

    return result


def parse_ssh_config(config_path: Path) -> list[dict]:
    """Parse SSH config file into a list of host entries."""
    if not config_path.exists():
        return []

    entries = []
    current_entry = None

    with open(config_path) as f:
        for line in f:
            stripped = line.strip()

            # Skip empty lines and comments
            if not stripped or stripped.startswith("#"):
                if current_entry:
                    current_entry["trailing_comments"].append(line)
                continue

            # New Host entry
            if stripped.lower().startswith("host "):
                if current_entry:
                    entries.append(current_entry)
                host_pattern = stripped.split(None, 1)[1] if len(stripped.split()) > 1 else ""
                current_entry = {
                    "host": host_pattern,
                    "hostname": None,
                    "user": None,
                    "port": None,
                    "identityfile": None,
                    "other_options": [],
                    "raw_lines": [line],
                    "trailing_comments": [],
                }
            elif current_entry:
                current_entry["raw_lines"].append(line)
                # Parse option
                if stripped.lower().startswith("hostname "):
                    current_entry["hostname"] = stripped.split(None, 1)[1]
                elif stripped.lower().startswith("user "):
                    current_entry["user"] = stripped.split(None, 1)[1]
                elif stripped.lower().startswith("port "):
                    current_entry["port"] = stripped.split(None, 1)[1]
                elif stripped.lower().startswith("identityfile "):
                    current_entry["identityfile"] = stripped.split(None, 1)[1]
                else:
                    current_entry["other_options"].append((stripped.split()[0], stripped))

    if current_entry:
        entries.append(current_entry)

    return entries


def update_ssh_config(
    config_path: Path,
    host_mappings: dict[str, dict],
    dry_run: bool = True,
) -> list[str]:
    """Update SSH config with Tailscale IPs.

    Returns list of changes made.
    """
    entries = parse_ssh_config(config_path)
    changes = []

    # Create new config content
    new_lines = []

    # Add header comment
    new_lines.append(f"# Updated by update_ssh_config_from_tailscale.py on {datetime.now().isoformat()}\n")
    new_lines.append("# Tailscale IPs are stable across public IP changes\n")
    new_lines.append("\n")

    for entry in entries:
        host = entry["host"]

        # Check if this is a multi-host pattern (skip these)
        if " " in host or "*" in host:
            new_lines.extend(entry["raw_lines"])
            new_lines.extend(entry["trailing_comments"])
            continue

        # Check if we have a mapping for this host
        if host in host_mappings:
            mapping = host_mappings[host]
            new_ip = mapping["tailscale_ip"]
            old_ip = entry.get("hostname")

            if old_ip and old_ip != new_ip:
                changes.append(f"{host}: {old_ip} -> {new_ip}")

            # Rebuild entry with new IP
            new_lines.append(f"Host {host}\n")
            new_lines.append(f"    HostName {new_ip}\n")

            if entry.get("user"):
                new_lines.append(f"    User {entry['user']}\n")
            elif mapping.get("ssh_user"):
                new_lines.append(f"    User {mapping['ssh_user']}\n")

            if entry.get("port") and entry.get("port") != "22":
                new_lines.append(f"    Port {entry['port']}\n")
            elif mapping.get("ssh_port") and mapping.get("ssh_port") != 22:
                new_lines.append(f"    Port {mapping['ssh_port']}\n")

            if entry.get("identityfile"):
                new_lines.append(f"    IdentityFile {entry['identityfile']}\n")
            elif mapping.get("ssh_key"):
                new_lines.append(f"    IdentityFile {mapping['ssh_key']}\n")

            # Add other options
            for opt_name, opt_line in entry.get("other_options", []):
                if opt_name.lower() not in ("hostname", "user", "port", "identityfile"):
                    new_lines.append(f"    {opt_line}\n")

            new_lines.append("\n")
        else:
            # Keep entry as-is
            new_lines.extend(entry["raw_lines"])
            new_lines.extend(entry["trailing_comments"])

    if not dry_run and changes:
        # Backup existing config
        backup_path = config_path.with_suffix(".bak")
        shutil.copy2(config_path, backup_path)
        print(f"Backed up to {backup_path}")

        # Write new config
        with open(config_path, "w") as f:
            f.writelines(new_lines)
        print(f"Updated {config_path}")

    return changes


def main():
    parser = argparse.ArgumentParser(
        description="Update SSH config with Tailscale IPs from distributed_hosts.yaml"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes (default is dry-run)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use live tailscale status (merged with yaml config)",
    )
    parser.add_argument(
        "--ssh-config",
        type=Path,
        default=Path.home() / ".ssh" / "config",
        help="Path to SSH config file",
    )
    parser.add_argument(
        "--hosts-yaml",
        type=Path,
        default=Path(__file__).parent.parent / "config" / "distributed_hosts.yaml",
        help="Path to distributed_hosts.yaml",
    )
    args = parser.parse_args()

    # Load mappings from YAML
    print(f"Loading mappings from {args.hosts_yaml}")
    yaml_mappings = load_distributed_hosts(args.hosts_yaml)
    print(f"Found {len(yaml_mappings)} hosts with Tailscale IPs in YAML")

    # Optionally merge with live status
    if args.live:
        print("Querying live tailscale status...")
        live_mappings = parse_tailscale_status()
        print(f"Found {len(live_mappings)} nodes in tailscale status")

        # Merge - live takes precedence for active nodes
        for hostname, info in live_mappings.items():
            if info.get("status") == "active":
                # Try to match to a yaml host
                for yaml_host, yaml_info in yaml_mappings.items():
                    # Check if tailscale hostname matches or contains yaml host
                    if yaml_host in hostname or hostname in yaml_host:
                        # Update with live IP
                        yaml_info["tailscale_ip"] = info["tailscale_ip"]
                        yaml_info["live_status"] = info["status"]
                        print(f"  {yaml_host}: Updated from live status")
                        break

    # Show mappings
    print("\nHost -> Tailscale IP mappings:")
    for host, info in sorted(yaml_mappings.items()):
        status_indicator = "●" if info.get("status") == "ready" else "○"
        print(f"  {status_indicator} {host}: {info['tailscale_ip']}")

    # Update SSH config
    print(f"\n{'Dry run' if not args.apply else 'Applying'} changes to {args.ssh_config}")
    changes = update_ssh_config(
        args.ssh_config,
        yaml_mappings,
        dry_run=not args.apply,
    )

    if changes:
        print(f"\n{'Would update' if not args.apply else 'Updated'} {len(changes)} entries:")
        for change in changes:
            print(f"  {change}")
    else:
        print("\nNo changes needed - all entries up to date")

    if not args.apply and changes:
        print("\nRun with --apply to apply changes")

    return 0


if __name__ == "__main__":
    sys.exit(main())
