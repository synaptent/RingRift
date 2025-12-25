#!/usr/bin/env python3
"""
Deploy P2P service to all cluster nodes.

Handles:
- SSH with retry and fallback to Tailscale IP
- Deploy updated service file
- Fix node.conf if path is wrong
- Restart P2P service
- Verify health endpoint responds

Usage:
    python scripts/deploy_p2p_service.py                 # Deploy to all nodes
    python scripts/deploy_p2p_service.py --dry-run       # Show what would be done
    python scripts/deploy_p2p_service.py --node gh200-a  # Single node
    python scripts/deploy_p2p_service.py --verify-only   # Just check health
"""

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Optional

import yaml

# Add ai-service to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_hosts_config():
    """Load distributed_hosts.yaml configuration."""
    config_path = Path(__file__).parent.parent / "config" / "distributed_hosts.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_ssh_command(host_config: dict, host_name: str) -> tuple[list[str], str]:
    """Build SSH command for a host, returning (ssh_args, ip_used)."""
    ssh_host = host_config.get("ssh_host", "")
    tailscale_ip = host_config.get("tailscale_ip", "")
    ssh_user = host_config.get("ssh_user", "ubuntu")
    ssh_key = host_config.get("ssh_key", "~/.ssh/id_cluster")
    ssh_port = host_config.get("ssh_port", 22)

    ssh_key = os.path.expanduser(ssh_key)

    # Build base args
    ssh_args = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=10",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=3",
        "-i", ssh_key,
    ]

    if ssh_port != 22:
        ssh_args.extend(["-p", str(ssh_port)])

    return ssh_args, ssh_host, tailscale_ip, ssh_user


async def run_ssh_command(
    host_name: str,
    host_config: dict,
    command: str,
    timeout: int = 30,
    use_tailscale: bool = False
) -> tuple[bool, str]:
    """Run SSH command on a host, with fallback to Tailscale."""
    ssh_args, ssh_host, tailscale_ip, ssh_user = get_ssh_command(host_config, host_name)

    # Choose IP to use
    if use_tailscale and tailscale_ip:
        target_ip = tailscale_ip
    else:
        target_ip = ssh_host

    if not target_ip:
        return False, f"No SSH host configured for {host_name}"

    full_cmd = ssh_args + [f"{ssh_user}@{target_ip}", command]

    try:
        proc = await asyncio.create_subprocess_exec(
            *full_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        output = stdout.decode() + stderr.decode()

        if proc.returncode == 0:
            return True, output.strip()
        else:
            # If public IP failed, try Tailscale
            if not use_tailscale and tailscale_ip and proc.returncode in (255, 1):
                return await run_ssh_command(
                    host_name, host_config, command,
                    timeout=timeout, use_tailscale=True
                )
            return False, f"Exit code {proc.returncode}: {output.strip()}"

    except asyncio.TimeoutError:
        if not use_tailscale and tailscale_ip:
            return await run_ssh_command(
                host_name, host_config, command,
                timeout=timeout, use_tailscale=True
            )
        return False, f"Timeout after {timeout}s"
    except Exception as e:
        return False, str(e)


async def check_p2p_health(host_name: str, host_config: dict) -> tuple[bool, str]:
    """Check if P2P is responding on a host."""
    # First check if port is listening
    success, output = await run_ssh_command(
        host_name, host_config,
        "curl -s --connect-timeout 5 http://localhost:8770/health || echo FAIL",
        timeout=15
    )
    if success and "FAIL" not in output and output.strip():
        return True, output.strip()
    return False, output


async def fix_node_conf(host_name: str, host_config: dict, dry_run: bool = False) -> tuple[bool, str]:
    """Fix node.conf if RINGRIFT_PATH has trailing /ai-service."""
    # Check current config
    success, output = await run_ssh_command(
        host_name, host_config,
        "cat /etc/ringrift/node.conf 2>/dev/null || echo MISSING"
    )

    if not success or "MISSING" in output:
        return False, "node.conf missing or unreadable"

    # Check if path has the problematic /ai-service suffix
    if "/ai-service" in output and "RINGRIFT_PATH" in output:
        # Extract current path
        lines = output.split("\n")
        needs_fix = False
        for line in lines:
            if line.startswith("RINGRIFT_PATH=") and line.endswith("/ai-service"):
                needs_fix = True
                break

        if needs_fix:
            if dry_run:
                return True, "Would fix: remove trailing /ai-service from RINGRIFT_PATH"

            # Fix it
            fix_cmd = (
                "sudo sed -i 's|RINGRIFT_PATH=.*/ai-service$|RINGRIFT_PATH=/home/ubuntu/ringrift|' "
                "/etc/ringrift/node.conf"
            )
            success, output = await run_ssh_command(host_name, host_config, fix_cmd)
            if success:
                return True, "Fixed: removed trailing /ai-service"
            return False, f"Failed to fix: {output}"

    return True, "node.conf OK (no fix needed)"


async def deploy_service_file(host_name: str, host_config: dict, service_content: str, dry_run: bool = False) -> tuple[bool, str]:
    """Deploy the updated service file to a host."""
    if dry_run:
        return True, "Would deploy service file"

    # Create temp file and copy
    escaped_content = service_content.replace("'", "'\"'\"'")
    cmd = f"echo '{escaped_content}' | sudo tee /etc/systemd/system/ringrift-p2p.service > /dev/null"

    success, output = await run_ssh_command(host_name, host_config, cmd, timeout=30)
    if not success:
        return False, f"Failed to write service file: {output}"

    # Reload systemd
    success, output = await run_ssh_command(
        host_name, host_config,
        "sudo systemctl daemon-reload"
    )
    if not success:
        return False, f"Failed to reload systemd: {output}"

    return True, "Service file deployed"


async def restart_p2p_service(host_name: str, host_config: dict, dry_run: bool = False) -> tuple[bool, str]:
    """Restart the P2P service on a host."""
    if dry_run:
        return True, "Would restart P2P service"

    # Stop any existing
    await run_ssh_command(
        host_name, host_config,
        "sudo systemctl stop ringrift-p2p 2>/dev/null || true"
    )

    # Small delay
    await asyncio.sleep(1)

    # Start fresh
    success, output = await run_ssh_command(
        host_name, host_config,
        "sudo systemctl start ringrift-p2p"
    )
    if not success:
        return False, f"Failed to start: {output}"

    # Wait for service to initialize
    await asyncio.sleep(3)

    # Check status
    success, output = await run_ssh_command(
        host_name, host_config,
        "sudo systemctl is-active ringrift-p2p"
    )
    if success and "active" in output:
        return True, "Service started successfully"

    # Get failure reason
    _, logs = await run_ssh_command(
        host_name, host_config,
        "journalctl -u ringrift-p2p -n 20 --no-pager 2>/dev/null | tail -10"
    )
    return False, f"Service not active: {logs}"


async def deploy_to_node(
    host_name: str,
    host_config: dict,
    service_content: str,
    dry_run: bool = False,
    verify_only: bool = False
) -> dict:
    """Deploy P2P service to a single node."""
    result = {
        "host": host_name,
        "success": False,
        "steps": [],
        "healthy": False
    }

    # Skip coordinator nodes
    if host_config.get("role") == "coordinator":
        result["steps"].append(("skip", "Coordinator node - skipping"))
        result["success"] = True
        return result

    # Skip nodes without SSH
    if not host_config.get("ssh_host"):
        result["steps"].append(("skip", "No SSH host configured"))
        return result

    if verify_only:
        # Just check health
        healthy, msg = await check_p2p_health(host_name, host_config)
        result["healthy"] = healthy
        result["steps"].append(("health", msg))
        result["success"] = True
        return result

    # Step 1: Fix node.conf if needed
    success, msg = await fix_node_conf(host_name, host_config, dry_run)
    result["steps"].append(("fix_conf", msg))
    if not success:
        return result

    # Step 2: Deploy service file
    success, msg = await deploy_service_file(host_name, host_config, service_content, dry_run)
    result["steps"].append(("deploy", msg))
    if not success:
        return result

    # Step 3: Restart service
    success, msg = await restart_p2p_service(host_name, host_config, dry_run)
    result["steps"].append(("restart", msg))
    if not success:
        return result

    # Step 4: Verify health
    if not dry_run:
        await asyncio.sleep(5)  # Give service time to start
        healthy, msg = await check_p2p_health(host_name, host_config)
        result["healthy"] = healthy
        result["steps"].append(("health", msg))

    result["success"] = True
    return result


async def deploy_all(
    hosts_config: dict,
    service_content: str,
    dry_run: bool = False,
    verify_only: bool = False,
    target_node: Optional[str] = None,
    concurrency: int = 10
) -> list[dict]:
    """Deploy to all nodes with controlled concurrency."""
    hosts = hosts_config.get("hosts", {})

    if target_node:
        # Filter to specific node
        matching = {k: v for k, v in hosts.items() if target_node.lower() in k.lower()}
        if not matching:
            print(f"No hosts matching '{target_node}' found")
            return []
        hosts = matching

    semaphore = asyncio.Semaphore(concurrency)

    async def deploy_with_semaphore(name, config):
        async with semaphore:
            return await deploy_to_node(name, config, service_content, dry_run, verify_only)

    tasks = [deploy_with_semaphore(name, config) for name, config in hosts.items()]
    results = await asyncio.gather(*tasks)
    return list(results)


def print_results(results: list[dict], verify_only: bool = False):
    """Print deployment results in a readable format."""
    print("\n" + "=" * 60)
    if verify_only:
        print("P2P Health Check Results")
    else:
        print("P2P Deployment Results")
    print("=" * 60)

    # Group by status
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    healthy = [r for r in results if r.get("healthy")]
    skipped = [r for r in results if any("skip" in s[0] for s in r.get("steps", []))]

    # Print summary
    print(f"\nTotal nodes: {len(results)}")
    print(f"  Successful: {len(successful)}")
    print(f"  Failed: {len(failed)}")
    print(f"  Healthy: {len(healthy)}")
    print(f"  Skipped: {len(skipped)}")

    # Print failures first
    if failed:
        print("\n--- FAILED ---")
        for r in failed:
            print(f"\n{r['host']}:")
            for step, msg in r.get("steps", []):
                print(f"  [{step}] {msg}")

    # Print unhealthy nodes
    unhealthy = [r for r in results if r["success"] and not r.get("healthy") and "skip" not in str(r.get("steps", []))]
    if unhealthy and not verify_only:
        print("\n--- UNHEALTHY (deployed but not responding) ---")
        for r in unhealthy:
            print(f"\n{r['host']}:")
            for step, msg in r.get("steps", []):
                print(f"  [{step}] {msg}")

    # Print healthy nodes
    if verify_only:
        print("\n--- HEALTHY ---")
        for r in healthy:
            print(f"  ✓ {r['host']}")

        print("\n--- UNHEALTHY ---")
        for r in results:
            if not r.get("healthy") and "skip" not in str(r.get("steps", [])):
                print(f"  ✗ {r['host']}")


def main():
    parser = argparse.ArgumentParser(description="Deploy P2P service to cluster nodes")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    parser.add_argument("--verify-only", action="store_true", help="Only check P2P health, don't deploy")
    parser.add_argument("--node", type=str, help="Deploy to specific node (partial match)")
    parser.add_argument("--concurrency", type=int, default=10, help="Max concurrent deployments")
    args = parser.parse_args()

    # Load config
    hosts_config = load_hosts_config()

    # Load service file
    service_path = Path(__file__).parent.parent / "deploy" / "systemd" / "ringrift-p2p-universal.service"
    with open(service_path) as f:
        service_content = f.read()

    print(f"Loaded {len(hosts_config.get('hosts', {}))} hosts from config")
    if args.dry_run:
        print("DRY RUN - no changes will be made")
    if args.verify_only:
        print("VERIFY ONLY - checking P2P health")

    # Run deployment
    start = time.time()
    results = asyncio.run(deploy_all(
        hosts_config,
        service_content,
        dry_run=args.dry_run,
        verify_only=args.verify_only,
        target_node=args.node,
        concurrency=args.concurrency
    ))
    elapsed = time.time() - start

    # Print results
    print_results(results, verify_only=args.verify_only)
    print(f"\nCompleted in {elapsed:.1f}s")

    # Exit with error if any failures
    if any(not r["success"] for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
