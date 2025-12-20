#!/usr/bin/env python3
"""
Run GPU Tests on Cloud Instances

Deploys and runs GPU test suite on configured cloud GPU instances.
Results are collected and summarized locally.

Usage:
    # Run on all GPU instances
    PYTHONPATH=. python scripts/run_gpu_tests_cloud.py --mode full

    # Run on specific host
    PYTHONPATH=. python scripts/run_gpu_tests_cloud.py --host lambda-h100 --mode full

    # Quick test on best available GPU
    PYTHONPATH=. python scripts/run_gpu_tests_cloud.py --mode quick

    # List available GPU hosts
    PYTHONPATH=. python scripts/run_gpu_tests_cloud.py --list-hosts

    # Dry run (show commands without executing)
    PYTHONPATH=. python scripts/run_gpu_tests_cloud.py --dry-run
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, List

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# GPU hosts that should be tested (in order of preference)
GPU_HOST_PRIORITY = [
    "lambda-h100",      # H100 - best
    "lambda-2xh100",    # 2x H100 - if available
    "vast-5090-quad",   # 4x RTX 5090
    "vast-5090-dual",   # 2x RTX 5090
    "vast-5090-a",      # RTX 5090
    "vast-5090-b",      # RTX 5090
    "lambda-a10",       # A10
    "vast-3090-a",      # RTX 3090
    "vast-3090-b",      # RTX 3090
    "vast-3080-dual",   # 2x RTX 3080
    "mac-studio",       # M3 Max/Ultra (MPS)
]

# Minimum GPU memory (GB) for running tests
MIN_GPU_MEMORY = 8  # 8GB minimum


def load_hosts_config() -> dict[str, dict]:
    """Load host configuration from YAML file."""
    if not HAS_YAML:
        print("Error: PyYAML not installed. Install with: pip install pyyaml")
        return {}

    script_dir = os.path.dirname(os.path.abspath(__file__))
    ai_service_dir = os.path.dirname(script_dir)
    config_path = os.path.join(ai_service_dir, "config/distributed_hosts.yaml")

    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        return {}

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config.get("hosts", {})


def get_gpu_hosts(hosts: dict[str, dict]) -> list[str]:
    """Filter and sort hosts by GPU capability."""
    gpu_hosts = []

    for name, config in hosts.items():
        # Check if host has GPU
        gpu = config.get("gpu", "")
        if not gpu:
            continue

        # Check status
        status = config.get("status", "")
        if status not in ("ready", "setup"):
            continue

        gpu_hosts.append(name)

    # Sort by priority
    def priority_key(name):
        try:
            return GPU_HOST_PRIORITY.index(name)
        except ValueError:
            return 999

    return sorted(gpu_hosts, key=priority_key)


def build_ssh_command(host_config: dict) -> list[str]:
    """Build SSH command for a host."""
    ssh_cmd = ["ssh"]

    # Add SSH key if specified
    ssh_key = host_config.get("ssh_key", "")
    if ssh_key:
        ssh_key = os.path.expanduser(ssh_key)
        if os.path.exists(ssh_key):
            ssh_cmd.extend(["-i", ssh_key])

    # Add port if non-standard
    ssh_port = host_config.get("ssh_port", 22)
    if ssh_port != 22:
        ssh_cmd.extend(["-p", str(ssh_port)])

    # Add common options
    ssh_cmd.extend([
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=30",
    ])

    # Build target
    ssh_user = host_config.get("ssh_user", "ubuntu")
    ssh_host = host_config.get("tailscale_ip") or host_config.get("ssh_host")

    ssh_cmd.append(f"{ssh_user}@{ssh_host}")

    return ssh_cmd


def check_host_connectivity(host_name: str, host_config: dict) -> bool:
    """Check if host is reachable via SSH."""
    ssh_cmd = build_ssh_command(host_config)
    ssh_cmd.append("echo ok")

    try:
        result = subprocess.run(
            ssh_cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode == 0 and "ok" in result.stdout
    except Exception as e:
        print(f"  {host_name}: Connection failed - {e}")
        return False


def run_gpu_tests_on_host(
    host_name: str,
    host_config: dict,
    mode: str = "quick",
    dry_run: bool = False,
) -> dict:
    """Run GPU tests on a remote host."""
    print(f"\n{'='*60}")
    print(f"Running GPU tests on: {host_name}")
    print(f"  GPU: {host_config.get('gpu', 'Unknown')}")
    print(f"  Memory: {host_config.get('memory_gb', 'Unknown')} GB")
    print(f"{'='*60}")

    # Build remote command
    ringrift_path = host_config.get("ringrift_path", "~/ringrift/ai-service")
    ringrift_path = ringrift_path.rstrip("/")
    if not ringrift_path.endswith("ai-service"):
        ringrift_path = f"{ringrift_path}/ai-service"

    venv_activate = host_config.get("venv_activate", "")

    # Build the test command
    test_cmd = f"""
cd {ringrift_path} && \
git pull --quiet && \
{venv_activate if venv_activate else 'true'} && \
PYTHONPATH=. python scripts/test_gpu_all.py --{mode} --output-json /tmp/gpu_test_results.json 2>&1
"""

    ssh_cmd = build_ssh_command(host_config)
    ssh_cmd.append(test_cmd.strip())

    if dry_run:
        print(f"  [DRY RUN] Would execute:")
        print(f"  {' '.join(ssh_cmd[:5])} '...'")
        return {"status": "dry_run", "host": host_name}

    # Execute tests
    start_time = time.time()
    try:
        print(f"  Executing tests (this may take several minutes)...")
        result = subprocess.run(
            ssh_cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minute timeout
        )

        duration = time.time() - start_time

        # Print output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(f"STDERR: {result.stderr}")

        # Try to fetch results JSON
        results = {"status": "completed", "host": host_name, "duration": duration}

        if result.returncode == 0:
            results["passed"] = True
            print(f"\n  TESTS PASSED on {host_name} in {duration:.1f}s")
        else:
            results["passed"] = False
            print(f"\n  TESTS FAILED on {host_name} (exit code {result.returncode})")

        # Try to fetch detailed results
        try:
            fetch_cmd = build_ssh_command(host_config)
            fetch_cmd.append("cat /tmp/gpu_test_results.json")
            fetch_result = subprocess.run(fetch_cmd, capture_output=True, text=True, timeout=30)
            if fetch_result.returncode == 0:
                results["details"] = json.loads(fetch_result.stdout)
        except Exception:
            pass

        return results

    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT on {host_name} after 30 minutes")
        return {"status": "timeout", "host": host_name, "passed": False}
    except Exception as e:
        print(f"  ERROR on {host_name}: {e}")
        return {"status": "error", "host": host_name, "error": str(e), "passed": False}


def main():
    parser = argparse.ArgumentParser(description="Run GPU tests on cloud instances")
    parser.add_argument("--host", type=str, help="Specific host to test")
    parser.add_argument("--mode", choices=["quick", "full", "benchmark"], default="quick",
                       help="Test mode (default: quick)")
    parser.add_argument("--list-hosts", action="store_true", help="List available GPU hosts")
    parser.add_argument("--dry-run", action="store_true", help="Show commands without executing")
    parser.add_argument("--all", action="store_true", help="Run on all GPU hosts")
    parser.add_argument("--output", type=str, help="Output results to JSON file")

    args = parser.parse_args()

    # Load configuration
    hosts = load_hosts_config()
    if not hosts:
        sys.exit(1)

    gpu_hosts = get_gpu_hosts(hosts)

    if args.list_hosts:
        print("\nAvailable GPU hosts:")
        print("-" * 60)
        for name in gpu_hosts:
            config = hosts[name]
            gpu = config.get("gpu", "Unknown")
            memory = config.get("memory_gb", "?")
            status = config.get("status", "unknown")
            print(f"  {name:20} {gpu:30} {memory}GB  ({status})")
        return

    # Determine which hosts to test
    if args.host:
        if args.host not in hosts:
            print(f"Error: Host '{args.host}' not found in configuration")
            sys.exit(1)
        test_hosts = [args.host]
    elif args.all:
        test_hosts = gpu_hosts
    else:
        # Default: use first available GPU host
        test_hosts = gpu_hosts[:1] if gpu_hosts else []

    if not test_hosts:
        print("Error: No GPU hosts available for testing")
        sys.exit(1)

    print(f"\nGPU Test Deployment")
    print(f"  Mode: {args.mode}")
    print(f"  Hosts: {', '.join(test_hosts)}")
    print()

    # Check connectivity
    print("Checking host connectivity...")
    available_hosts = []
    for host_name in test_hosts:
        host_config = hosts[host_name]
        if check_host_connectivity(host_name, host_config):
            print(f"  {host_name}: OK")
            available_hosts.append(host_name)
        else:
            print(f"  {host_name}: UNREACHABLE")

    if not available_hosts:
        print("\nError: No hosts are reachable")
        sys.exit(1)

    # Run tests
    all_results = []
    for host_name in available_hosts:
        result = run_gpu_tests_on_host(
            host_name,
            hosts[host_name],
            mode=args.mode,
            dry_run=args.dry_run,
        )
        all_results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("CLOUD GPU TEST SUMMARY")
    print("=" * 60)

    passed_count = sum(1 for r in all_results if r.get("passed", False))
    total_count = len(all_results)

    for result in all_results:
        host = result.get("host", "unknown")
        status = result.get("status", "unknown")
        passed = result.get("passed", False)
        duration = result.get("duration", 0)

        status_str = "PASS" if passed else "FAIL"
        print(f"  {host:20} [{status_str}] ({status}, {duration:.1f}s)")

    print(f"\nTotal: {passed_count}/{total_count} hosts passed")

    # Save results
    if args.output:
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "mode": args.mode,
            "results": all_results,
            "summary": {
                "passed": passed_count,
                "total": total_count,
                "all_passed": passed_count == total_count,
            },
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"\nResults saved to: {args.output}")

    # Exit with appropriate code
    sys.exit(0 if passed_count == total_count else 1)


if __name__ == "__main__":
    main()
