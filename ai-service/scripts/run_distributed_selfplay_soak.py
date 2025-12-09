#!/usr/bin/env python
"""Distributed self-play soak runner for RingRift.

Runs mixed-AI self-play across multiple machines (local + remote SSH hosts)
for all board types and player counts. Records games to SQLite databases
for subsequent parity validation.

Supports multiple deployment modes:
- local:  Run only on localhost
- lan:    Run on localhost + local network Mac cluster
- aws:    Run on localhost + AWS staging server (square8 only - 16GB RAM limit)
- hybrid: Run on localhost + LAN + AWS (maximum parallelism)

Example usage (from ai-service/):

    # Run using LAN mode (local + Mac cluster)
    python scripts/run_distributed_selfplay_soak.py \
        --mode lan \
        --games-per-config 100 \
        --output-dir data/games/distributed_soak

    # Run with explicit host list
    python scripts/run_distributed_selfplay_soak.py \
        --games-per-config 100 \
        --hosts local,m1-pro \
        --output-dir data/games/distributed_soak

    # Run with AWS (square8 only due to 16GB memory limit)
    python scripts/run_distributed_selfplay_soak.py \
        --mode aws \
        --board-types square8 \
        --games-per-config 50

    # Hybrid mode for maximum parallelism
    python scripts/run_distributed_selfplay_soak.py \
        --mode hybrid \
        --games-per-config 100

    # Dry run to see what would be executed
    python scripts/run_distributed_selfplay_soak.py \
        --games-per-config 50 \
        --dry-run

After completion, run parity checks on all generated databases:

    python scripts/check_ts_python_replay_parity.py \
        --db data/games/distributed_soak/*.db
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# Board configurations with appropriate max moves
BOARD_CONFIGS: Dict[str, Dict[int, int]] = {
    # board_type: {num_players: max_moves}
    "square8": {2: 200, 3: 250, 4: 300},
    "square19": {2: 1200, 3: 1400, 4: 1600},
    "hexagonal": {2: 1200, 3: 1400, 4: 1600},
}

# Memory requirements per board type (in GB)
# Based on empirical testing:
# - 16GB machine: only 8x8 works reliably
# - 64GB machine: can run 19x19/hex with memory pressure
# - 96GB machine: runs everything comfortably
BOARD_MEMORY_REQUIREMENTS: Dict[str, int] = {
    "square8": 8,      # 8GB minimum for 8x8 games
    "square19": 48,    # 48GB minimum for 19x19 games (64GB machine has pressure)
    "hexagonal": 48,   # 48GB minimum for hex games
}

# Default config file paths (relative to ai-service/)
CONFIG_FILE_PATH = "config/distributed_hosts.yaml"
TEMPLATE_CONFIG_PATH = "config/distributed_hosts.template.yaml"

# Deployment modes
VALID_MODES = ["local", "lan", "aws", "hybrid"]


def load_remote_hosts(config_path: Optional[str] = None) -> Dict[str, Dict]:
    """Load remote host configuration from YAML file.

    Looks for config in:
    1. Explicitly provided path (--config flag)
    2. config/distributed_hosts.yaml (gitignored, local config)
    3. Falls back to empty dict if neither exists

    Copy config/distributed_hosts.template.yaml to config/distributed_hosts.yaml
    and fill in your actual host details.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ai_service_dir = os.path.dirname(script_dir)

    # Try explicit path first
    if config_path:
        if not os.path.isabs(config_path):
            config_path = os.path.join(ai_service_dir, config_path)
        paths_to_try = [config_path]
    else:
        paths_to_try = [
            os.path.join(ai_service_dir, CONFIG_FILE_PATH),
        ]

    for path in paths_to_try:
        if os.path.exists(path):
            if not HAS_YAML:
                print(f"Warning: PyYAML not installed. Install with: pip install pyyaml")
                print(f"         Cannot load config from {path}")
                return {}

            with open(path, "r") as f:
                config = yaml.safe_load(f)

            hosts = config.get("hosts", {})
            if hosts:
                print(f"Loaded {len(hosts)} remote host(s) from {path}")
            return hosts

    # No config found - print helpful message
    template_path = os.path.join(ai_service_dir, TEMPLATE_CONFIG_PATH)
    if os.path.exists(template_path):
        print(f"No host configuration found.")
        print(f"Copy {template_path}")
        print(f"  to {os.path.join(ai_service_dir, CONFIG_FILE_PATH)}")
        print(f"and fill in your host details.")

    return {}


# Remote hosts loaded at module init (can be overridden by load_remote_hosts)
REMOTE_HOSTS: Dict[str, Dict] = {}


def get_hosts_for_mode(mode: str, remote_hosts: Dict[str, Dict]) -> List[str]:
    """Get list of hosts based on deployment mode.

    Args:
        mode: One of 'local', 'lan', 'aws', 'hybrid'
        remote_hosts: Dict of configured remote hosts

    Returns:
        List of host names to use
    """
    hosts = ["local"]

    if mode == "local":
        return hosts

    # Identify LAN hosts (those without explicit ssh_user or with macOS-style paths)
    lan_hosts = []
    aws_hosts = []

    for name, config in remote_hosts.items():
        # AWS hosts typically have ssh_user set and use /home paths
        ssh_user = config.get("ssh_user", "")
        work_dir = config.get("ringrift_path", config.get("work_dir", ""))

        if ssh_user == "ubuntu" or "/home/" in work_dir:
            aws_hosts.append(name)
        else:
            lan_hosts.append(name)

    if mode == "lan":
        hosts.extend(lan_hosts)
    elif mode == "aws":
        hosts.extend(aws_hosts)
    elif mode == "hybrid":
        hosts.extend(lan_hosts)
        hosts.extend(aws_hosts)

    return hosts


def get_local_memory_gb() -> Tuple[int, int]:
    """Get total and available physical memory on local machine in GB.

    Returns:
        Tuple of (total_gb, available_gb)
    """
    total_gb = 8
    available_gb = 8

    try:
        # macOS: use sysctl for total
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            bytes_total = int(result.stdout.strip())
            total_gb = bytes_total // (1024 ** 3)

        # macOS: use vm_stat for available (free + inactive pages)
        result = subprocess.run(
            ["vm_stat"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            # Parse vm_stat output for page size and free/inactive pages
            page_size = 4096  # Default
            free_pages = 0
            inactive_pages = 0

            for line in result.stdout.split('\n'):
                if 'page size of' in line:
                    # Extract page size from first line
                    match = re.search(r'page size of (\d+)', line)
                    if match:
                        page_size = int(match.group(1))
                elif 'Pages free:' in line:
                    free_pages = int(line.split(':')[1].strip().rstrip('.'))
                elif 'Pages inactive:' in line:
                    inactive_pages = int(line.split(':')[1].strip().rstrip('.'))

            # Available = free + inactive (can be reclaimed)
            available_bytes = (free_pages + inactive_pages) * page_size
            available_gb = available_bytes // (1024 ** 3)

        return total_gb, available_gb

    except Exception:
        pass

    try:
        # Linux: read /proc/meminfo
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    total_gb = kb // (1024 * 1024)
                elif line.startswith("MemAvailable:"):
                    kb = int(line.split()[1])
                    available_gb = kb // (1024 * 1024)
        return total_gb, available_gb
    except Exception:
        pass

    # Fallback: assume 8GB total, 4GB available
    print("Warning: Could not detect local memory, assuming 8GB total, 4GB available")
    return 8, 4


def get_remote_memory_gb(host_name: str, host_config: Dict) -> Tuple[int, int]:
    """Get total and available physical memory on remote host in GB via SSH.

    Returns:
        Tuple of (total_gb, available_gb)
    """
    ssh_host = host_config["ssh_host"]
    ssh_key = host_config.get("ssh_key")

    # First check if memory_gb is in config (for total)
    if "memory_gb" in host_config:
        config_total = host_config["memory_gb"]
        # Still detect available if possible, but use config for total
    else:
        config_total = None

    ssh_cmd_base = ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes"]
    if ssh_key:
        ssh_cmd_base.extend(["-i", os.path.expanduser(ssh_key)])

    total_gb = 8
    available_gb = 4

    try:
        # Get total memory
        if config_total:
            total_gb = config_total
        else:
            ssh_cmd = ssh_cmd_base + [ssh_host, "sysctl -n hw.memsize 2>/dev/null || grep MemTotal /proc/meminfo 2>/dev/null | awk '{print $2 * 1024}'"]
            result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=15)
            if result.returncode == 0 and result.stdout.strip():
                bytes_total = int(result.stdout.strip())
                total_gb = bytes_total // (1024 ** 3)

        # Get available memory (free + inactive) via vm_stat on macOS
        # Script to extract free and inactive pages and calculate available GB
        vm_stat_script = '''
pagesize=$(sysctl -n hw.pagesize 2>/dev/null || echo 4096)
free=$(vm_stat 2>/dev/null | grep "Pages free:" | awk -F: '{gsub(/[^0-9]/,"",$2); print $2}')
inactive=$(vm_stat 2>/dev/null | grep "Pages inactive:" | awk -F: '{gsub(/[^0-9]/,"",$2); print $2}')
if [ -n "$free" ] && [ -n "$inactive" ]; then
    echo $(( (free + inactive) * pagesize / 1073741824 ))
elif [ -f /proc/meminfo ]; then
    grep MemAvailable /proc/meminfo | awk '{print int($2/1048576)}'
else
    echo 4
fi
'''
        ssh_cmd = ssh_cmd_base + [ssh_host, vm_stat_script]
        result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=15)
        if result.returncode == 0 and result.stdout.strip():
            available_gb = int(result.stdout.strip())

        return total_gb, available_gb

    except Exception as e:
        print(f"Warning: Could not detect memory for {host_name}: {e}")

    # Fallback
    if config_total:
        return config_total, config_total // 2
    print(f"Warning: Could not detect memory for {host_name}, assuming 8GB total, 4GB available")
    return 8, 4


@dataclass
class HostMemoryInfo:
    """Memory information for a host."""
    total_gb: int
    available_gb: int

    def __str__(self) -> str:
        return f"{self.total_gb}GB total, {self.available_gb}GB available"


# Cached host memory info (keyed by host name)
HOST_MEMORY_INFO_CACHE: Dict[str, HostMemoryInfo] = {}


def detect_all_host_memory(hosts: List[str]) -> Tuple[Dict[str, int], Dict[str, HostMemoryInfo]]:
    """Detect memory for all specified hosts.

    Returns:
        Tuple of:
        - Dict[host_name, total_memory_gb] for job assignment compatibility
        - Dict[host_name, HostMemoryInfo] for detailed reporting
    """
    global HOST_MEMORY_INFO_CACHE

    totals = {}
    details = {}

    for host in hosts:
        if host in HOST_MEMORY_INFO_CACHE:
            info = HOST_MEMORY_INFO_CACHE[host]
        else:
            if host == "local":
                total_gb, available_gb = get_local_memory_gb()
            elif host in REMOTE_HOSTS:
                total_gb, available_gb = get_remote_memory_gb(host, REMOTE_HOSTS[host])
            else:
                total_gb, available_gb = 8, 4  # Unknown host, assume minimal

            info = HostMemoryInfo(total_gb=total_gb, available_gb=available_gb)
            HOST_MEMORY_INFO_CACHE[host] = info

        totals[host] = info.total_gb
        details[host] = info

    return totals, details


def get_eligible_hosts_for_board(
    board_type: str,
    hosts: List[str],
    host_memory: Dict[str, int],
) -> List[str]:
    """Return list of hosts that have enough memory for the given board type."""
    required_memory = BOARD_MEMORY_REQUIREMENTS.get(board_type, 8)
    eligible = []

    for host in hosts:
        host_mem = host_memory.get(host, 8)
        if host_mem >= required_memory:
            eligible.append(host)

    return eligible


@dataclass
class JobConfig:
    """Configuration for a single self-play job."""
    job_id: str
    host: str
    board_type: str
    num_players: int
    num_games: int
    max_moves: int
    output_db: str
    log_jsonl: str
    seed: int


def generate_job_configs(
    games_per_config: int,
    hosts: List[str],
    output_dir: str,
    base_seed: int = 42,
    host_memory: Optional[Dict[str, int]] = None,
    allowed_board_types: Optional[List[str]] = None,
    allowed_num_players: Optional[List[int]] = None,
) -> List[JobConfig]:
    """Generate job configurations distributed across hosts based on memory capacity.

    Jobs are assigned only to hosts with sufficient memory for the board type.
    Memory requirements:
    - square8: 8GB minimum (runs on all machines)
    - square19/hexagonal: 48GB minimum (skips low-memory machines)

    If host_memory is not provided, all hosts are assumed to have sufficient memory.
    """
    jobs = []
    job_idx = 0

    for board_type, player_configs in BOARD_CONFIGS.items():
        if allowed_board_types and board_type not in allowed_board_types:
            continue
        # Get hosts eligible for this board type based on memory
        if host_memory:
            eligible_hosts = get_eligible_hosts_for_board(board_type, hosts, host_memory)
            if not eligible_hosts:
                required = BOARD_MEMORY_REQUIREMENTS.get(board_type, 8)
                print(f"Warning: No hosts have enough memory ({required}GB) for {board_type}")
                print(f"  Skipping {board_type} configurations")
                continue
        else:
            eligible_hosts = hosts

        # Calculate games per host for this board type
        num_hosts = len(eligible_hosts)
        games_per_host = games_per_config // num_hosts
        remainder = games_per_config % num_hosts

        for num_players, max_moves in player_configs.items():
            if allowed_num_players and num_players not in allowed_num_players:
                continue
            for host_idx, host in enumerate(eligible_hosts):
                # Distribute remainder across first hosts
                host_games = games_per_host + (1 if host_idx < remainder else 0)
                if host_games == 0:
                    continue

                config_id = f"{board_type}_{num_players}p"
                job_id = f"{config_id}_{host}_{job_idx}"

                # Different seed per job for variety
                job_seed = base_seed + job_idx * 1000

                jobs.append(JobConfig(
                    job_id=job_id,
                    host=host,
                    board_type=board_type,
                    num_players=num_players,
                    num_games=host_games,
                    max_moves=max_moves,
                    output_db=os.path.join(output_dir, f"selfplay_{config_id}_{host}.db"),
                    log_jsonl=os.path.join(output_dir, f"selfplay_{config_id}_{host}.jsonl"),
                    seed=job_seed,
                ))
                job_idx += 1

    return jobs


def build_soak_command(job: JobConfig, is_remote: bool = False) -> str:
    """Build the self-play soak command for a job."""
    cmd_parts = [
        "PYTHONPATH=.",
        "RINGRIFT_SKIP_SHADOW_CONTRACTS=true",
        "python",
        "scripts/run_self_play_soak.py",
        f"--num-games {job.num_games}",
        f"--board-type {job.board_type}",
        f"--num-players {job.num_players}",
        f"--max-moves {job.max_moves}",
        f"--seed {job.seed}",
        "--engine-mode mixed",
        "--difficulty-band canonical",
        f"--log-jsonl {job.log_jsonl}",
        f"--record-db {job.output_db}",
        "--verbose 10",
        "--gc-interval 5",
    ]

    # Add memory constraints for large boards
    if job.board_type in ("square19", "hexagonal"):
        cmd_parts.append("--memory-constrained")

    return " ".join(cmd_parts)


def run_local_job(job: JobConfig, ringrift_ai_dir: str) -> Tuple[str, bool, str]:
    """Run a self-play job on the local machine."""
    cmd = build_soak_command(job)

    print(f"[LOCAL] Starting job {job.job_id}: {job.num_games} games of "
          f"{job.board_type} {job.num_players}p")

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=ringrift_ai_dir,
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout
        )
        success = result.returncode == 0
        output = result.stdout + result.stderr

        if success:
            print(f"[LOCAL] Job {job.job_id} completed successfully")
        else:
            print(f"[LOCAL] Job {job.job_id} failed with code {result.returncode}")

        return job.job_id, success, output

    except subprocess.TimeoutExpired:
        print(f"[LOCAL] Job {job.job_id} timed out")
        return job.job_id, False, "Job timed out after 2 hours"
    except Exception as e:
        print(f"[LOCAL] Job {job.job_id} error: {e}")
        return job.job_id, False, str(e)


def run_remote_job(job: JobConfig, host_config: Dict) -> Tuple[str, bool, str]:
    """Run a self-play job on a remote machine via SSH."""
    ssh_host = host_config["ssh_host"]
    ssh_user = host_config.get("ssh_user")
    ringrift_path = host_config.get("ringrift_path") or host_config.get("work_dir", "~/Development/RingRift")
    venv_activate = host_config.get("venv_activate", "source venv/bin/activate")
    ssh_key = host_config.get("ssh_key")

    # Build SSH target (user@host or just host)
    ssh_target = f"{ssh_user}@{ssh_host}" if ssh_user else ssh_host

    # Build remote command
    soak_cmd = build_soak_command(job, is_remote=True)
    remote_cmd = f"cd {ringrift_path}/ai-service && {venv_activate} && {soak_cmd}"

    ssh_cmd = [
        "ssh",
        "-o", "ConnectTimeout=10",
        "-o", "ServerAliveInterval=60",
    ]
    if ssh_key:
        ssh_cmd.extend(["-i", os.path.expanduser(ssh_key)])
    ssh_cmd.extend([ssh_target, remote_cmd])

    print(f"[{ssh_host.upper()}] Starting job {job.job_id}: {job.num_games} games of "
          f"{job.board_type} {job.num_players}p")

    try:
        result = subprocess.run(
            ssh_cmd,
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout
        )
        success = result.returncode == 0
        output = result.stdout + result.stderr

        if success:
            print(f"[{ssh_host.upper()}] Job {job.job_id} completed successfully")
        else:
            print(f"[{ssh_host.upper()}] Job {job.job_id} failed with code {result.returncode}")

        return job.job_id, success, output

    except subprocess.TimeoutExpired:
        print(f"[{ssh_host.upper()}] Job {job.job_id} timed out")
        return job.job_id, False, "Job timed out after 2 hours"
    except Exception as e:
        print(f"[{ssh_host.upper()}] Job {job.job_id} error: {e}")
        return job.job_id, False, str(e)


def run_job(job: JobConfig, ringrift_ai_dir: str) -> Tuple[str, bool, str]:
    """Dispatch job to appropriate host."""
    if job.host == "local":
        return run_local_job(job, ringrift_ai_dir)
    elif job.host in REMOTE_HOSTS:
        return run_remote_job(job, REMOTE_HOSTS[job.host])
    else:
        return job.job_id, False, f"Unknown host: {job.host}"


def fetch_remote_results(jobs: List[JobConfig], output_dir: str) -> List[str]:
    """Fetch database files from remote hosts.

    Returns:
        List of successfully fetched local database paths.
    """
    fetched_dbs = []

    for job in jobs:
        if job.host != "local" and job.host in REMOTE_HOSTS:
            host_config = REMOTE_HOSTS[job.host]
            ssh_host = host_config["ssh_host"]
            ssh_user = host_config.get("ssh_user")
            ssh_key = host_config.get("ssh_key")
            ringrift_path = host_config.get("ringrift_path") or host_config.get("work_dir", "~/Development/RingRift")

            # Build SSH target (user@host or just host)
            ssh_target = f"{ssh_user}@{ssh_host}" if ssh_user else ssh_host

            remote_db = f"{ringrift_path}/ai-service/{job.output_db}"
            local_db = os.path.join(output_dir, os.path.basename(job.output_db))

            print(f"Fetching {remote_db} from {ssh_target}...")

            scp_cmd = ["scp"]
            if ssh_key:
                scp_cmd.extend(["-i", os.path.expanduser(ssh_key)])
            scp_cmd.extend([f"{ssh_target}:{remote_db}", local_db])

            try:
                subprocess.run(
                    scp_cmd,
                    check=True,
                    capture_output=True,
                    timeout=300,
                )
                print(f"  -> Saved to {local_db}")
                fetched_dbs.append(local_db)
            except subprocess.CalledProcessError as e:
                print(f"  -> Failed to fetch: {e}")
            except subprocess.TimeoutExpired:
                print(f"  -> Fetch timed out")
        elif job.host == "local":
            # Local jobs already have their DB in the output dir
            local_db = os.path.join(output_dir, os.path.basename(job.output_db))
            if os.path.exists(local_db):
                fetched_dbs.append(local_db)

    return fetched_dbs


def run_parity_checks(db_paths: List[str], ringrift_ai_dir: str) -> Tuple[int, int]:
    """Run parity checks on the given databases.

    Returns:
        Tuple of (passed_count, failed_count)
    """
    passed = 0
    failed = 0

    print("\nRunning parity checks on generated databases...")
    print("=" * 60)

    for db_path in db_paths:
        db_name = os.path.basename(db_path)
        print(f"\nChecking {db_name}...")

        try:
            result = subprocess.run(
                [
                    "python", "scripts/check_ts_python_replay_parity.py",
                    "--db", db_path,
                ],
                cwd=ringrift_ai_dir,
                capture_output=True,
                text=True,
                timeout=600,  # 10 min timeout per DB
                env={**os.environ, "PYTHONPATH": ".", "RINGRIFT_SKIP_SHADOW_CONTRACTS": "true"},
            )

            if result.returncode == 0:
                print(f"  ✓ PASSED")
                passed += 1
            else:
                print(f"  ✗ FAILED")
                # Print last few lines of error
                error_lines = result.stderr.strip().split('\n')[-5:]
                for line in error_lines:
                    print(f"    {line}")
                failed += 1

        except subprocess.TimeoutExpired:
            print(f"  ✗ TIMEOUT (exceeded 10 minutes)")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            failed += 1

    print()
    print("=" * 60)
    print(f"Parity check results: {passed} passed, {failed} failed")

    return passed, failed


def main():
    parser = argparse.ArgumentParser(
        description="Run distributed self-play soaks across multiple machines"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=VALID_MODES,
        default=None,
        help="Deployment mode: local, lan, aws, or hybrid. Overrides --hosts if specified.",
    )
    parser.add_argument(
        "--games-per-config",
        type=int,
        default=100,
        help="Number of games per (board_type, num_players) configuration",
    )
    parser.add_argument(
        "--board-types",
        type=str,
        default=None,
        help="Comma-separated board types to run (default: all)",
    )
    parser.add_argument(
        "--num-players",
        type=str,
        default=None,
        help="Comma-separated player counts to run (default: 2,3,4)",
    )
    parser.add_argument(
        "--hosts",
        type=str,
        default="local",
        help="Comma-separated list of hosts to use (local, m1-pro, aws-staging). Default: local. Ignored if --mode is specified.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/games/distributed_soak",
        help="Directory for output databases and logs",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="Base random seed for reproducibility",
    )
    parser.add_argument(
        "--max-parallel-per-host",
        type=int,
        default=2,
        help="Maximum parallel jobs per host (default: 2 to avoid memory exhaustion)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print job configurations without running",
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip fetching results from remote hosts",
    )
    parser.add_argument(
        "--run-parity",
        action="store_true",
        help="Run parity checks on all databases after fetching",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=f"Path to host configuration YAML file (default: {CONFIG_FILE_PATH})",
    )

    args = parser.parse_args()

    # Load remote host configuration
    global REMOTE_HOSTS
    REMOTE_HOSTS = load_remote_hosts(args.config)

    # Determine hosts based on mode or explicit --hosts
    if args.mode:
        hosts = get_hosts_for_mode(args.mode, REMOTE_HOSTS)
        print(f"Using mode '{args.mode}' with hosts: {', '.join(hosts)}")
    else:
        # Parse hosts from command line
        hosts = [h.strip() for h in args.hosts.split(",")]

    # Validate hosts
    for host in hosts:
        if host != "local" and host not in REMOTE_HOSTS:
            available = ["local"] + list(REMOTE_HOSTS.keys())
            print(f"Error: Unknown host '{host}'.")
            print(f"Available hosts: {', '.join(available)}")
            if not REMOTE_HOSTS:
                print(f"\nNo remote hosts configured. See: config/distributed_hosts.template.yaml")
            sys.exit(1)

    # Determine ai-service directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ringrift_ai_dir = os.path.dirname(script_dir)

    # Create output directory
    output_dir = os.path.join(ringrift_ai_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Detect memory on all hosts
    print("\nDetecting host memory...")
    host_memory, host_memory_details = detect_all_host_memory(hosts)
    print("Host memory configuration:")
    for host, mem_gb in sorted(host_memory.items(), key=lambda x: -x[1]):
        info = host_memory_details[host]
        eligible_boards = [
            board for board, req in BOARD_MEMORY_REQUIREMENTS.items()
            if mem_gb >= req
        ]
        print(f"  {host}: {info.total_gb}GB total, {info.available_gb}GB available -> eligible for: {', '.join(eligible_boards)}")
    print()

    # Parse filters
    allowed_board_types = None
    if args.board_types:
        allowed_board_types = [b.strip() for b in args.board_types.split(",") if b.strip()]
    allowed_num_players = None
    if args.num_players:
        allowed_num_players = [int(p.strip()) for p in args.num_players.split(",") if p.strip()]

    # Generate job configurations with memory-aware distribution
    jobs = generate_job_configs(
        games_per_config=args.games_per_config,
        hosts=hosts,
        output_dir=args.output_dir,
        base_seed=args.base_seed,
        host_memory=host_memory,
        allowed_board_types=allowed_board_types,
        allowed_num_players=allowed_num_players,
    )

    print(f"\n{'='*60}")
    print(f"Distributed Self-Play Soak Configuration")
    print(f"{'='*60}")
    print(f"Games per config: {args.games_per_config}")
    print(f"Hosts: {', '.join(hosts)}")
    print(f"Output directory: {output_dir}")
    print(f"Total jobs: {len(jobs)}")
    print(f"Total games: {sum(j.num_games for j in jobs)}")
    print()

    # Print job summary
    print("Job Summary:")
    print("-" * 60)
    for job in jobs:
        print(f"  {job.job_id}: {job.board_type} {job.num_players}p x{job.num_games} "
              f"-> {job.output_db}")
    print()

    if args.dry_run:
        print("DRY RUN - commands that would be executed:")
        print("-" * 60)
        for job in jobs:
            cmd = build_soak_command(job)
            if job.host == "local":
                print(f"[LOCAL] {cmd}")
            else:
                print(f"[{job.host.upper()}] ssh {job.host} 'cd ... && {cmd}'")
        return

    # Run jobs
    print(f"\nStarting distributed self-play at {datetime.now().isoformat()}")
    print("=" * 60)

    start_time = time.time()
    results = []

    # Group jobs by host for parallel execution
    jobs_by_host: Dict[str, List[JobConfig]] = {}
    for job in jobs:
        if job.host not in jobs_by_host:
            jobs_by_host[job.host] = []
        jobs_by_host[job.host].append(job)

    # Run jobs with limited parallelism per host
    with ThreadPoolExecutor(max_workers=len(hosts) * args.max_parallel_per_host) as executor:
        futures = {}
        for job in jobs:
            future = executor.submit(run_job, job, ringrift_ai_dir)
            futures[future] = job

        for future in as_completed(futures):
            job_id, success, output = future.result()
            results.append((job_id, success, output))

    elapsed = time.time() - start_time

    # Summary
    print()
    print("=" * 60)
    print(f"Distributed Self-Play Complete")
    print("=" * 60)
    print(f"Elapsed time: {elapsed/60:.1f} minutes")

    successful = sum(1 for _, s, _ in results if s)
    failed = len(results) - successful

    print(f"Successful jobs: {successful}/{len(results)}")
    if failed > 0:
        print(f"Failed jobs: {failed}")
        for job_id, success, output in results:
            if not success:
                print(f"  - {job_id}")

    # Fetch results from remote hosts
    fetched_dbs = []
    if not args.skip_fetch:
        print()
        print("Fetching results from remote hosts...")
        fetched_dbs = fetch_remote_results(jobs, output_dir)
        print(f"Fetched {len(fetched_dbs)} database(s) to {output_dir}")

    # Run parity checks if requested
    parity_passed = 0
    parity_failed = 0
    if args.run_parity and fetched_dbs:
        parity_passed, parity_failed = run_parity_checks(fetched_dbs, ringrift_ai_dir)

    # Write summary
    summary_path = os.path.join(output_dir, "distributed_soak_summary.json")
    summary = {
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": elapsed,
        "games_per_config": args.games_per_config,
        "hosts": hosts,
        "total_jobs": len(jobs),
        "successful_jobs": successful,
        "failed_jobs": failed,
        "fetched_dbs": len(fetched_dbs),
        "job_results": [
            {"job_id": jid, "success": s}
            for jid, s, _ in results
        ],
    }

    if args.run_parity:
        summary["parity_passed"] = parity_passed
        summary["parity_failed"] = parity_failed

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to: {summary_path}")

    print()
    if args.run_parity:
        if parity_failed > 0:
            print(f"WARNING: {parity_failed} database(s) failed parity checks!")
        else:
            print(f"All {parity_passed} database(s) passed parity checks.")
    else:
        print("Next steps:")
        print("  1. Run parity checks on generated databases:")
        print(f"     cd ai-service && python scripts/check_ts_python_replay_parity.py \\")
        print(f"         --db {args.output_dir}/*.db")
        print()
        print("  Or re-run with --run-parity to check automatically.")
    print()

    # Exit with error if jobs failed or parity checks failed
    exit_code = 0
    if failed > 0:
        exit_code = 1
    if args.run_parity and parity_failed > 0:
        exit_code = 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
