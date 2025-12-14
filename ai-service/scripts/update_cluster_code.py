#!/usr/bin/env python3
"""Update git code on all distributed cluster hosts.

This script reads config/distributed_hosts.yaml and runs git pull on each host.
Handles stashing local changes if needed.

Usage:
    # Check status only (dry-run)
    python scripts/update_cluster_code.py --status

    # Update all hosts
    python scripts/update_cluster_code.py

    # Update with auto-stash for local changes
    python scripts/update_cluster_code.py --auto-stash

    # Update specific hosts only
    python scripts/update_cluster_code.py --hosts lambda-h100 lambda-2xh100

    # Force reset to origin (dangerous - discards local changes)
    python scripts/update_cluster_code.py --force-reset
"""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class HostResult:
    """Result of updating a single host."""
    name: str
    success: bool
    commit: str
    message: str
    had_local_changes: bool = False


def load_hosts_config() -> dict[str, Any]:
    """Load distributed_hosts.yaml configuration."""
    try:
        import yaml
    except ImportError:
        print("ERROR: PyYAML required. Install with: pip install pyyaml")
        sys.exit(1)

    ai_service_dir = Path(__file__).resolve().parent.parent
    config_path = ai_service_dir / "config" / "distributed_hosts.yaml"

    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)

    data = yaml.safe_load(config_path.read_text()) or {}
    return data.get("hosts", {})


def build_ssh_cmd(host_config: dict[str, Any]) -> list[str]:
    """Build SSH command prefix for a host."""
    cmd = ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no"]

    # Add SSH key if specified
    if host_config.get("ssh_key"):
        key_path = Path(host_config["ssh_key"]).expanduser()
        cmd.extend(["-i", str(key_path)])

    # Add port if non-default
    port = host_config.get("ssh_port", 22)
    if port != 22:
        cmd.extend(["-p", str(port)])

    # Build user@host
    user = host_config.get("ssh_user", "root")
    host = host_config.get("ssh_host") or host_config.get("tailscale_ip")
    if not host:
        raise ValueError("No ssh_host or tailscale_ip defined")

    cmd.append(f"{user}@{host}")
    return cmd


def get_ringrift_path(host_config: dict[str, Any]) -> str:
    """Get the RingRift path on the remote host."""
    path = host_config.get("ringrift_path", "~/ringrift/ai-service")
    # Remove ai-service suffix to get root repo path
    if path.endswith("/ai-service"):
        path = path[:-11]
    elif path.endswith("ai-service"):
        path = path[:-10] or "~/ringrift"
    return path


async def run_ssh_command(ssh_cmd: list[str], remote_cmd: str, timeout: int = 60) -> tuple[bool, str]:
    """Run a command via SSH and return (success, output)."""
    full_cmd = ssh_cmd + [remote_cmd]
    try:
        proc = await asyncio.create_subprocess_exec(
            *full_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        output = stdout.decode("utf-8", errors="replace").strip()
        return proc.returncode == 0, output
    except asyncio.TimeoutError:
        return False, "SSH timeout"
    except Exception as e:
        return False, f"SSH error: {e}"


async def get_host_status(name: str, host_config: dict[str, Any]) -> HostResult:
    """Get git status for a single host."""
    try:
        ssh_cmd = build_ssh_cmd(host_config)
        path = get_ringrift_path(host_config)

        # Get current commit
        success, output = await run_ssh_command(ssh_cmd, f"cd {path} && git log --oneline -1")
        if not success:
            return HostResult(name=name, success=False, commit="", message=f"unreachable: {output}")

        commit = output.split()[0] if output else "unknown"
        commit_msg = " ".join(output.split()[1:]) if output else ""

        # Check for local changes
        _, status_output = await run_ssh_command(ssh_cmd, f"cd {path} && git status --porcelain")
        has_changes = bool(status_output.strip())

        return HostResult(
            name=name,
            success=True,
            commit=commit,
            message=commit_msg[:50],
            had_local_changes=has_changes,
        )
    except Exception as e:
        return HostResult(name=name, success=False, commit="", message=str(e))


async def update_host(
    name: str,
    host_config: dict[str, Any],
    auto_stash: bool = False,
    force_reset: bool = False,
) -> HostResult:
    """Update git on a single host."""
    try:
        ssh_cmd = build_ssh_cmd(host_config)
        path = get_ringrift_path(host_config)

        # Check for local changes first
        _, status_output = await run_ssh_command(ssh_cmd, f"cd {path} && git status --porcelain")
        has_changes = bool(status_output.strip())

        if has_changes:
            if force_reset:
                # Force reset to origin
                success, output = await run_ssh_command(
                    ssh_cmd,
                    f"cd {path} && git fetch origin main && git reset --hard origin/main",
                    timeout=120,
                )
                if not success:
                    return HostResult(name=name, success=False, commit="", message=f"reset failed: {output}", had_local_changes=True)
            elif auto_stash:
                # Stash changes
                success, _ = await run_ssh_command(ssh_cmd, f"cd {path} && git stash")
                if not success:
                    return HostResult(name=name, success=False, commit="", message="stash failed", had_local_changes=True)

                # Pull
                success, output = await run_ssh_command(ssh_cmd, f"cd {path} && git pull origin main", timeout=120)
                if not success:
                    return HostResult(name=name, success=False, commit="", message=f"pull failed: {output}", had_local_changes=True)

                # Try to restore stash (may conflict)
                await run_ssh_command(ssh_cmd, f"cd {path} && git stash pop || true")
            else:
                return HostResult(
                    name=name,
                    success=False,
                    commit="",
                    message="has local changes (use --auto-stash or --force-reset)",
                    had_local_changes=True,
                )
        else:
            # No local changes, just pull
            success, output = await run_ssh_command(ssh_cmd, f"cd {path} && git pull origin main", timeout=120)
            if not success:
                return HostResult(name=name, success=False, commit="", message=f"pull failed: {output}")

        # Get new commit
        success, output = await run_ssh_command(ssh_cmd, f"cd {path} && git log --oneline -1")
        if not success:
            return HostResult(name=name, success=False, commit="", message="updated but couldn't get commit")

        commit = output.split()[0] if output else "unknown"
        commit_msg = " ".join(output.split()[1:]) if output else ""

        return HostResult(
            name=name,
            success=True,
            commit=commit,
            message=commit_msg[:50],
            had_local_changes=has_changes,
        )
    except Exception as e:
        return HostResult(name=name, success=False, commit="", message=str(e))


async def main_async(args: argparse.Namespace) -> int:
    """Main async entry point."""
    hosts = load_hosts_config()

    if not hosts:
        print("No hosts configured in distributed_hosts.yaml")
        return 1

    # Filter hosts if specific ones requested
    if args.hosts:
        hosts = {k: v for k, v in hosts.items() if k in args.hosts}
        if not hosts:
            print(f"No matching hosts found for: {args.hosts}")
            return 1

    # Filter out disabled hosts
    hosts = {k: v for k, v in hosts.items() if v.get("status") != "disabled"}

    # Get local commit for reference
    local_result = subprocess.run(
        ["git", "log", "--oneline", "-1"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent.parent.parent,
    )
    local_commit = local_result.stdout.strip() if local_result.returncode == 0 else "unknown"

    print(f"Local commit: {local_commit}")
    print(f"Hosts to {'check' if args.status else 'update'}: {len(hosts)}")
    print()

    # Run operations in parallel
    if args.status:
        tasks = [get_host_status(name, config) for name, config in hosts.items()]
    else:
        tasks = [
            update_host(name, config, auto_stash=args.auto_stash, force_reset=args.force_reset)
            for name, config in hosts.items()
        ]

    results = await asyncio.gather(*tasks)

    # Print results
    success_count = 0
    behind_count = 0
    local_commit_short = local_commit.split()[0][:7] if local_commit else ""

    print(f"{'Host':<20} {'Status':<12} {'Commit':<10} {'Message'}")
    print("-" * 80)

    for result in sorted(results, key=lambda r: r.name):
        if result.success:
            status = "OK"
            # Compare first 7 chars of commit hashes
            if result.commit[:7] != local_commit_short:
                status = "BEHIND"
                behind_count += 1
            if result.had_local_changes:
                status += " (changes)"
            success_count += 1
        else:
            status = "FAILED"

        print(f"{result.name:<20} {status:<12} {result.commit:<10} {result.message}")

    print("-" * 80)
    print(f"Total: {len(results)} | Success: {success_count} | Behind: {behind_count} | Failed: {len(results) - success_count}")

    return 0 if success_count == len(results) else 1


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Update git code on all distributed cluster hosts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--status", action="store_true", help="Check status only, don't update")
    parser.add_argument("--auto-stash", action="store_true", help="Automatically stash local changes before pull")
    parser.add_argument("--force-reset", action="store_true", help="Force reset to origin/main (WARNING: discards local changes)")
    parser.add_argument("--hosts", nargs="+", help="Update only specific hosts (space-separated)")

    args = parser.parse_args()

    if args.force_reset and args.auto_stash:
        print("ERROR: Cannot use both --force-reset and --auto-stash")
        return 1

    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
