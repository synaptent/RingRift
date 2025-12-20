#!/usr/bin/env python
"""Deploy and run LPS + rings ablation experiment across cluster hosts.

This script:
1. Deploys updated code to cluster machines via rsync
2. Runs ablation experiments in parallel across hosts
3. Collects and aggregates results

Usage:
    # Deploy code and run experiment
    python scripts/deploy_lps_ablation.py --deploy --run

    # Just run (code already deployed)
    python scripts/deploy_lps_ablation.py --run

    # Just deploy (manual run later)
    python scripts/deploy_lps_ablation.py --deploy

    # Collect results from previous run
    python scripts/deploy_lps_ablation.py --collect

Cluster hosts (configure via --hosts or RINGRIFT_CLUSTER_HOSTS env var):
    Default vast.ai instances are configured in CLUSTER_HOSTS below.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Default cluster hosts (SSH connection strings)
# Format: "user@host:port" or "user@host -p port"
CLUSTER_HOSTS = [
    # Vast.ai instances from user
    "root@211.21.106.81:35066",
    "root@79.112.1.66:20153",
    "root@211.21.106.81:36401",
    "root@116.102.207.205:24469",
]

# Remote paths
REMOTE_BASE = "/root/RingRift"
REMOTE_AI_SERVICE = f"{REMOTE_BASE}/ai-service"
REMOTE_RESULTS_DIR = f"{REMOTE_AI_SERVICE}/logs/lps_ablation"

# Local paths
LOCAL_AI_SERVICE = Path(__file__).parent.parent.resolve()
LOCAL_RESULTS_DIR = LOCAL_AI_SERVICE / "logs" / "lps_ablation_cluster"


@dataclass
class HostConfig:
    """Parsed host configuration."""
    user: str
    host: str
    port: int

    @classmethod
    def from_string(cls, s: str) -> HostConfig:
        """Parse host string like 'user@host:port' or 'user@host -p port'."""
        # Handle "user@host:port" format
        if ':' in s and '-p' not in s:
            user_host, port = s.rsplit(':', 1)
            user, host = user_host.split('@')
            return cls(user=user, host=host, port=int(port))

        # Handle "user@host -p port" format
        if '-p' in s:
            parts = s.split()
            user_host = parts[0]
            port_idx = parts.index('-p') + 1
            port = int(parts[port_idx])
            user, host = user_host.split('@')
            return cls(user=user, host=host, port=port)

        # Assume default port 22
        user, host = s.split('@')
        return cls(user=user, host=host, port=22)

    @property
    def ssh_target(self) -> str:
        return f"{self.user}@{self.host}"

    @property
    def ssh_args(self) -> list[str]:
        return ["-p", str(self.port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=30"]


def run_ssh(host: HostConfig, cmd: str, timeout: int = 300) -> tuple[int, str, str]:
    """Run SSH command and return (returncode, stdout, stderr)."""
    full_cmd = ["ssh", *host.ssh_args, host.ssh_target, cmd]
    try:
        result = subprocess.run(
            full_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"Timeout after {timeout}s"
    except Exception as e:
        return -1, "", str(e)


def run_rsync(host: HostConfig, local_path: str, remote_path: str) -> tuple[int, str, str]:
    """Run rsync to deploy files."""
    rsync_cmd = [
        "rsync", "-avz", "--delete",
        "-e", f"ssh -p {host.port} -o StrictHostKeyChecking=no -o ConnectTimeout=30",
        "--exclude", "__pycache__",
        "--exclude", "*.pyc",
        "--exclude", ".git",
        "--exclude", "venv",
        "--exclude", "*.egg-info",
        "--exclude", "data/games",
        "--exclude", "logs",
        local_path,
        f"{host.ssh_target}:{remote_path}"
    ]
    try:
        result = subprocess.run(rsync_cmd, capture_output=True, text=True, timeout=300)
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "rsync timeout"
    except Exception as e:
        return -1, "", str(e)


def deploy_via_tarball(host: HostConfig) -> tuple[str, bool, str]:
    """Deploy code via tarball (more reliable for restricted SSH connections)."""
    import tempfile
    print(f"  Deploying to {host.host}:{host.port} (via tarball)...")

    # Create tarball locally
    tarball_path = Path(tempfile.gettempdir()) / "ringrift_ai_deploy.tar.gz"

    # Create tarball excluding unnecessary files
    tar_cmd = [
        "tar", "-czf", str(tarball_path),
        "--exclude", "__pycache__",
        "--exclude", "*.pyc",
        "--exclude", ".git",
        "--exclude", "venv",
        "--exclude", "*.egg-info",
        "--exclude", "data/games",
        "--exclude", "logs",
        "-C", str(LOCAL_AI_SERVICE.parent),
        LOCAL_AI_SERVICE.name
    ]
    try:
        result = subprocess.run(tar_cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            return host.ssh_target, False, f"tar failed: {result.stderr}"
    except Exception as e:
        return host.ssh_target, False, f"tar error: {e}"

    # Create remote directory
    rc, _, stderr = run_ssh(host, f"mkdir -p {REMOTE_BASE}")
    if rc != 0:
        return host.ssh_target, False, f"mkdir failed: {stderr}"

    # SCP the tarball
    scp_cmd = [
        "scp",
        "-P", str(host.port),
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=30",
        str(tarball_path),
        f"{host.ssh_target}:/tmp/ringrift_ai_deploy.tar.gz"
    ]
    try:
        result = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            return host.ssh_target, False, f"scp failed: {result.stderr}"
    except subprocess.TimeoutExpired:
        return host.ssh_target, False, "scp timeout"
    except Exception as e:
        return host.ssh_target, False, f"scp error: {e}"

    # Extract on remote
    rc, _, stderr = run_ssh(
        host,
        f"cd {REMOTE_BASE} && rm -rf ai-service && tar -xzf /tmp/ringrift_ai_deploy.tar.gz && rm /tmp/ringrift_ai_deploy.tar.gz",
        timeout=120
    )
    if rc != 0:
        return host.ssh_target, False, f"extract failed: {stderr}"

    # Install dependencies
    rc, _, _ = run_ssh(
        host,
        f"cd {REMOTE_AI_SERVICE} && pip install -q -e . 2>/dev/null || true",
        timeout=120
    )

    return host.ssh_target, True, "Deployed (tarball)"


def check_host(host: HostConfig) -> tuple[str, bool, str]:
    """Check if host is reachable."""
    print(f"  Checking {host.host}:{host.port}...")
    rc, stdout, stderr = run_ssh(host, "echo ok", timeout=15)
    if rc == 0 and "ok" in stdout:
        return host.ssh_target, True, "OK"
    return host.ssh_target, False, stderr or "Connection failed"


def deploy_to_host(host: HostConfig) -> tuple[str, bool, str]:
    """Deploy code to a single host."""
    print(f"  Deploying to {host.host}:{host.port}...")

    # Create remote directory structure
    rc, _, stderr = run_ssh(host, f"mkdir -p {REMOTE_AI_SERVICE}")
    if rc != 0:
        return host.ssh_target, False, f"mkdir failed: {stderr}"

    # Rsync the ai-service directory
    local_path = str(LOCAL_AI_SERVICE) + "/"
    rc, _stdout, stderr = run_rsync(host, local_path, REMOTE_AI_SERVICE)
    if rc != 0:
        return host.ssh_target, False, f"rsync failed: {stderr}"

    # Install dependencies if needed
    rc, _, stderr = run_ssh(
        host,
        f"cd {REMOTE_AI_SERVICE} && pip install -q -e . 2>/dev/null || true",
        timeout=120
    )

    return host.ssh_target, True, "Deployed"


def run_experiment_on_host(
    host: HostConfig,
    board_type: str,
    lps_rounds: list[int],
    rings_values: list[str],
    num_games: int,
    engine_mode: str,
    seed: int,
) -> tuple[str, bool, str]:
    """Run ablation experiment on a single host."""
    print(f"  Running on {host.host}:{host.port} ({board_type})...")

    lps_args = " ".join(str(x) for x in lps_rounds)
    rings_args = " ".join(rings_values)

    cmd = (
        f"cd {REMOTE_AI_SERVICE} && "
        f"PYTHONPATH=. RINGRIFT_SKIP_SHADOW_CONTRACTS=true "
        f"python scripts/run_lps_ablation.py "
        f"--num-games {num_games} "
        f"--board-type {board_type} "
        f"--lps-rounds {lps_args} "
        f"--rings-per-player {rings_args} "
        f"--engine-mode {engine_mode} "
        f"--seed {seed} "
        f"--output-dir {REMOTE_RESULTS_DIR}"
    )

    # Longer timeout for actual experiment
    timeout = 3600 * 4  # 4 hours max
    rc, stdout, stderr = run_ssh(host, cmd, timeout=timeout)

    if rc != 0:
        return host.ssh_target, False, f"Experiment failed: {stderr[-500:] if stderr else 'unknown'}"

    # Extract results file path from output
    results_line = [l for l in stdout.split('\n') if 'Results saved to:' in l]
    if results_line:
        return host.ssh_target, True, results_line[0].split(': ')[-1].strip()

    return host.ssh_target, True, "Completed (no results file found)"


def collect_results(hosts: list[HostConfig]) -> dict[str, any]:
    """Collect results from all hosts."""
    LOCAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    for host in hosts:
        print(f"  Collecting from {host.host}:{host.port}...")

        # List result files
        rc, stdout, _ = run_ssh(host, f"ls {REMOTE_RESULTS_DIR}/*.json 2>/dev/null || true")
        if rc != 0 or not stdout.strip():
            print("    No results found")
            continue

        for remote_file in stdout.strip().split('\n'):
            if not remote_file:
                continue

            local_file = LOCAL_RESULTS_DIR / f"{host.host}_{host.port}_{Path(remote_file).name}"

            # SCP the file
            scp_cmd = [
                "scp",
                "-P", str(host.port),
                "-o", "StrictHostKeyChecking=no",
                f"{host.ssh_target}:{remote_file}",
                str(local_file)
            ]
            try:
                subprocess.run(scp_cmd, capture_output=True, timeout=60)
                print(f"    Downloaded: {local_file.name}")

                # Load and add to aggregated results
                with open(local_file) as f:
                    data = json.load(f)
                    all_results.append({
                        "host": f"{host.host}:{host.port}",
                        "file": str(local_file),
                        "data": data,
                    })
            except Exception as e:
                print(f"    Failed to download {remote_file}: {e}")

    return {"results": all_results, "collected_at": datetime.now().isoformat()}


def main():
    parser = argparse.ArgumentParser(description="Deploy and run LPS ablation on cluster")
    parser.add_argument("--deploy", action="store_true", help="Deploy code to hosts")
    parser.add_argument("--run", action="store_true", help="Run experiments on hosts")
    parser.add_argument("--collect", action="store_true", help="Collect results from hosts")
    parser.add_argument("--check", action="store_true", help="Check host connectivity")
    parser.add_argument(
        "--hosts",
        nargs="+",
        default=None,
        help="Override cluster hosts (user@host:port format)"
    )
    parser.add_argument(
        "--board-types",
        nargs="+",
        default=["square8", "square19", "hexagonal"],
        help="Board types to test"
    )
    parser.add_argument(
        "--lps-rounds",
        nargs="+",
        type=int,
        default=[2, 3],
        help="LPS rounds values"
    )
    parser.add_argument(
        "--rings-per-player",
        nargs="+",
        default=["default", "96", "120"],
        help="Rings per player values"
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=100,
        help="Games per condition per host"
    )
    parser.add_argument(
        "--engine-mode",
        default="heuristic-only",
        choices=["heuristic-only", "mcts-only", "random-only"],
        help="AI engine mode"
    )
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")

    args = parser.parse_args()

    # Parse hosts
    if args.hosts:
        host_strings = args.hosts
    elif os.getenv("RINGRIFT_CLUSTER_HOSTS"):
        host_strings = os.getenv("RINGRIFT_CLUSTER_HOSTS").split(",")
    else:
        host_strings = CLUSTER_HOSTS
    host_strings = [h.strip() for h in host_strings if h.strip()]
    hosts = [HostConfig.from_string(h) for h in host_strings]

    if not hosts:
        print("ERROR: No hosts configured")
        sys.exit(1)

    print(f"Cluster hosts ({len(hosts)}):")
    for h in hosts:
        print(f"  {h.user}@{h.host}:{h.port}")
    print()

    if args.check or (not args.deploy and not args.run and not args.collect):
        print("Checking host connectivity...")
        with ThreadPoolExecutor(max_workers=len(hosts)) as executor:
            futures = {executor.submit(check_host, h): h for h in hosts}
            for future in as_completed(futures):
                target, ok, msg = future.result()
                status = "✓" if ok else "✗"
                print(f"  {status} {target}: {msg}")

        if not args.deploy and not args.run and not args.collect:
            return

    if args.deploy:
        print("\nDeploying code to cluster (sequentially to avoid rate limits)...")
        for host in hosts:
            # Try tarball method first (more reliable for vast.ai)
            target, ok, msg = deploy_via_tarball(host)
            status = "✓" if ok else "✗"
            print(f"  {status} {target}: {msg}")
            if not ok:
                # Retry once after delay
                print("    Retrying in 10s...")
                time.sleep(10)
                target, ok, msg = deploy_via_tarball(host)
                status = "✓" if ok else "✗"
                print(f"  {status} {target} (retry): {msg}")
            time.sleep(3)  # Delay between hosts to avoid rate limits

    if args.run:
        print("\nRunning experiments...")
        # Distribute board types across hosts
        board_types = args.board_types
        assignments = []
        for i, board_type in enumerate(board_types):
            # Assign each board type to a different host (round-robin)
            host = hosts[i % len(hosts)]
            assignments.append((host, board_type))

        print("Assignments:")
        for host, bt in assignments:
            print(f"  {host.host}:{host.port} -> {bt}")

        with ThreadPoolExecutor(max_workers=len(assignments)) as executor:
            futures = {}
            for i, (host, board_type) in enumerate(assignments):
                # Vary seed by assignment to get different games
                seed = args.seed + i * 1000
                future = executor.submit(
                    run_experiment_on_host,
                    host,
                    board_type,
                    args.lps_rounds,
                    args.rings_per_player,
                    args.num_games,
                    args.engine_mode,
                    seed,
                )
                futures[future] = (host, board_type)

            for future in as_completed(futures):
                host, board_type = futures[future]
                target, ok, msg = future.result()
                status = "✓" if ok else "✗"
                print(f"  {status} {target} ({board_type}): {msg}")

    if args.collect:
        print("\nCollecting results...")
        results = collect_results(hosts)

        # Save aggregated results
        LOCAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        output_file = LOCAL_RESULTS_DIR / f"aggregated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nAggregated results saved to: {output_file}")


if __name__ == "__main__":
    main()
