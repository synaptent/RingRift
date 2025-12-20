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
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# Import sync_lock for coordinated file transfers
try:
    from app.coordination.sync_lock import (
        acquire_sync_lock,
        release_sync_lock,
    )
    HAS_SYNC_LOCK = True
except ImportError:
    HAS_SYNC_LOCK = False

    def acquire_sync_lock(host: str, timeout: float = 30.0) -> bool:
        return True

    def release_sync_lock(host: str) -> None:
        pass

# Unified resource guard - 80% utilization limits (enforced 2025-12-16)
try:
    from app.utils.resource_guard import (
        LIMITS as RESOURCE_LIMITS,
        can_proceed as resource_can_proceed,
        check_disk_space,
        check_memory,
        require_resources,
    )
    HAS_RESOURCE_GUARD = True
except ImportError:
    HAS_RESOURCE_GUARD = False
    def resource_can_proceed(**kwargs):
        return True  # type: ignore
    def check_disk_space(*args, **kwargs):
        return True  # type: ignore
    def check_memory(*args, **kwargs):
        return True  # type: ignore
    def require_resources(*args, **kwargs):
        return True  # type: ignore
    RESOURCE_LIMITS = None  # type: ignore

# Unified selfplay configuration
try:
    from app.training.selfplay_config import SelfplayConfig, create_argument_parser
    HAS_SELFPLAY_CONFIG = True
except ImportError:
    HAS_SELFPLAY_CONFIG = False
    SelfplayConfig = None  # type: ignore
    create_argument_parser = None  # type: ignore

# Board configurations with appropriate max moves
BOARD_CONFIGS: dict[str, dict[int, int]] = {
    # board_type: {num_players: max_moves}
    "square8": {2: 400, 3: 600, 4: 800},
    "square19": {2: 2000, 3: 3000, 4: 4000},
    "hexagonal": {2: 2000, 3: 3000, 4: 4000},
    "hex8": {2: 400, 3: 600, 4: 800},  # Small hex board, similar to square8
}

# Memory requirements per board type (in GB)
# Based on empirical testing:
# - 16GB machine: only 8x8 works reliably
# - 64GB machine: can run 19x19/hex with memory pressure
# - 96GB machine: runs everything comfortably
BOARD_MEMORY_REQUIREMENTS: dict[str, int] = {
    "square8": 8,  # 8GB minimum for 8x8 games
    "square19": 48,  # 48GB minimum for 19x19 games (64GB machine has pressure)
    "hexagonal": 48,  # 48GB minimum for hex games
    "hex8": 8,  # 8GB minimum for small hex (similar to square8)
}

# Default config file paths (relative to ai-service/)
CONFIG_FILE_PATH = "config/distributed_hosts.yaml"
TEMPLATE_CONFIG_PATH = "config/distributed_hosts.template.yaml"

# Deployment modes
VALID_MODES = ["local", "lan", "aws", "hybrid"]


def load_remote_hosts(config_path: str | None = None) -> dict[str, dict]:
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
                print("Warning: PyYAML not installed. Install with: pip install pyyaml")
                print(f"         Cannot load config from {path}")
                return {}

            with open(path) as f:
                config = yaml.safe_load(f)

            hosts = config.get("hosts", {})
            if hosts:
                print(f"Loaded {len(hosts)} remote host(s) from {path}")
            return hosts

    # No config found - print helpful message
    template_path = os.path.join(ai_service_dir, TEMPLATE_CONFIG_PATH)
    if os.path.exists(template_path):
        print("No host configuration found.")
        print(f"Copy {template_path}")
        print(f"  to {os.path.join(ai_service_dir, CONFIG_FILE_PATH)}")
        print("and fill in your host details.")

    return {}


def _remote_ai_service_dir(host_config: dict) -> str:
    """Return the remote ai-service directory for a host config.

    Host configs sometimes specify the repo root (RingRift/) and sometimes the
    ai-service directory directly (RingRift/ai-service). Normalize both.
    """
    ringrift_path = host_config.get("ringrift_path") or host_config.get("work_dir", "~/Development/RingRift")
    path = str(ringrift_path).rstrip("/")
    if path.endswith("ai-service"):
        return path
    return f"{path}/ai-service"


def _quote_remote_path(path: str) -> str:
    """Quote a remote filesystem path for use in a bash command.

    Preserves ~$HOME expansion (ssh runs the command through a shell).
    """
    raw = str(path).strip()
    if raw == "~":
        return "$HOME"
    if raw.startswith("~/"):
        rest = raw[2:]
        # Quote only the suffix; allow $HOME to expand.
        return "$HOME/" + shlex.quote(rest)
    if "$" in raw:
        # Prefer double-quotes so $VAR expansion still works.
        escaped = raw.replace('"', '\\"')
        return '"' + escaped + '"'
    return shlex.quote(raw)


def _format_remote_venv_activate(venv_activate: str) -> str:
    """Best-effort venv activation that doesn't hard-fail when missing.

    Vast nodes often do not have a venv at the configured path; treat activation
    as optional so we can still run via system/conda python.
    """
    raw = (venv_activate or "").strip()
    if not raw:
        return ""

    try:
        parts = shlex.split(raw)
    except Exception:
        return f"({raw}) || true"

    if parts and parts[0] in ("source", ".") and len(parts) >= 2:
        activate_path = parts[1]
        quoted = _quote_remote_path(activate_path)
        # Some hosts (notably Vast.ai containers) can have an incomplete venv that
        # masks system-level deps. Activate only when it can import FastAPI (a
        # transitive dependency of the self-play harness).
        return (
            f"if [ -f {quoted} ]; then "
            f"{parts[0]} {quoted}; "
            "python3 -c 'import fastapi' >/dev/null 2>&1 || "
            "{ echo '[distributed-soak] venv missing deps; using system python' >&2; "
            "deactivate 2>/dev/null || true; }; "
            "fi"
        )

    return f"({raw}) || true"


# Remote hosts loaded at module init (can be overridden by load_remote_hosts)
REMOTE_HOSTS: dict[str, dict] = {}


def get_hosts_for_mode(mode: str, remote_hosts: dict[str, dict]) -> list[str]:
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


def get_local_memory_gb() -> tuple[int, int]:
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
            total_gb = bytes_total // (1024**3)

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

            for line in result.stdout.split("\n"):
                if "page size of" in line:
                    # Extract page size from first line
                    match = re.search(r"page size of (\d+)", line)
                    if match:
                        page_size = int(match.group(1))
                elif "Pages free:" in line:
                    free_pages = int(line.split(":")[1].strip().rstrip("."))
                elif "Pages inactive:" in line:
                    inactive_pages = int(line.split(":")[1].strip().rstrip("."))

            # Available = free + inactive (can be reclaimed)
            available_bytes = (free_pages + inactive_pages) * page_size
            available_gb = available_bytes // (1024**3)

        return total_gb, available_gb

    except Exception:
        pass

    try:
        # Linux: read /proc/meminfo
        with open("/proc/meminfo") as f:
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


def get_remote_memory_gb(host_name: str, host_config: dict) -> tuple[int, int]:
    """Get total and available physical memory on remote host in GB via SSH.

    Returns:
        Tuple of (total_gb, available_gb)
    """
    ssh_host = host_config["ssh_host"]
    ssh_user = host_config.get("ssh_user")
    ssh_port = host_config.get("ssh_port")
    ssh_key = host_config.get("ssh_key")
    ssh_target = f"{ssh_user}@{ssh_host}" if ssh_user else ssh_host

    # First check if memory_gb is in config (for total)
    if "memory_gb" in host_config:
        config_total = host_config["memory_gb"]
        # Still detect available if possible, but use config for total
    else:
        config_total = None

    ssh_cmd_base = ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes"]
    if ssh_port:
        ssh_cmd_base.extend(["-p", str(ssh_port)])
    if ssh_key:
        ssh_cmd_base.extend(["-i", os.path.expanduser(ssh_key)])

    total_gb = 8
    available_gb = 4

    try:
        # Get total memory
        if config_total:
            total_gb = config_total
        else:
            ssh_cmd = [*ssh_cmd_base, ssh_target, "sysctl -n hw.memsize 2>/dev/null || grep MemTotal /proc/meminfo 2>/dev/null | awk '{print $2 * 1024}'"]
            result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=15)
            if result.returncode == 0 and result.stdout.strip():
                bytes_total = int(result.stdout.strip())
                total_gb = bytes_total // (1024**3)

        # Get available memory (free + inactive) via vm_stat on macOS
        # Script to extract free and inactive pages and calculate available GB
        vm_stat_script = """
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
"""
        ssh_cmd = [*ssh_cmd_base, ssh_target, vm_stat_script]
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
HOST_MEMORY_INFO_CACHE: dict[str, HostMemoryInfo] = {}


def detect_all_host_memory(hosts: list[str]) -> tuple[dict[str, int], dict[str, HostMemoryInfo]]:
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


@dataclass
class HostDiskInfo:
    """Disk information for a host."""

    total_gb: int
    available_gb: int

    def __str__(self) -> str:
        return f"{self.total_gb}GB total, {self.available_gb}GB available"


# Cached host disk info (keyed by host name)
HOST_DISK_INFO_CACHE: dict[str, HostDiskInfo] = {}


def get_local_disk_gb(path: str) -> tuple[int, int]:
    """Get total and available disk on the filesystem backing `path` in GB."""
    try:
        usage = shutil.disk_usage(path)
        total_gb = usage.total // (1024**3)
        available_gb = usage.free // (1024**3)
        return int(total_gb), int(available_gb)
    except Exception as e:
        print(f"Warning: Could not detect local disk for {path}: {e}")
        return 0, 0


def get_remote_disk_gb(host_name: str, host_config: dict) -> tuple[int, int]:
    """Get total and available disk on the filesystem backing the remote ai-service dir in GB."""
    ssh_host = host_config["ssh_host"]
    ssh_user = host_config.get("ssh_user")
    ssh_port = host_config.get("ssh_port")
    ssh_key = host_config.get("ssh_key")
    ssh_target = f"{ssh_user}@{ssh_host}" if ssh_user else ssh_host
    ai_service_dir = _remote_ai_service_dir(host_config)

    ssh_cmd_base = ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes"]
    if ssh_port:
        ssh_cmd_base.extend(["-p", str(ssh_port)])
    if ssh_key:
        ssh_cmd_base.extend(["-i", os.path.expanduser(ssh_key)])

    try:
        df_script = (
            "set -euo pipefail; "
            f"cd {_quote_remote_path(ai_service_dir)}; "
            "df -Pk . | tail -n 1 | awk '{print $2\" \"$4}'"
        )
        ssh_cmd = [*ssh_cmd_base, ssh_target, f"bash -lc {shlex.quote(df_script)}"]
        result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=15)
        # Vast.ai nodes often print a login banner on stderr/stdout; extract the
        # final numeric payload line instead of hard-failing on extra output.
        if result.stdout:
            total_kb = None
            available_kb = None
            for line in result.stdout.splitlines():
                m = re.fullmatch(r"\s*(\d+)\s+(\d+)\s*", line)
                if not m:
                    continue
                total_kb = int(m.group(1))
                available_kb = int(m.group(2))
            if total_kb is not None and available_kb is not None:
                total_gb = total_kb // 1048576
                available_gb = available_kb // 1048576
                return int(total_gb), int(available_gb)

        raise ValueError(f"Unexpected df output: {result.stdout.strip()!r} {result.stderr.strip()!r}".strip())
    except Exception as e:
        print(f"Warning: Could not detect disk for {host_name}: {e}")
        return 0, 0


def detect_all_host_disk(
    hosts: list[str],
    ringrift_ai_dir: str,
) -> tuple[dict[str, int], dict[str, HostDiskInfo]]:
    """Detect disk for all specified hosts.

    Returns:
        Tuple of:
        - Dict[host_name, available_disk_gb]
        - Dict[host_name, HostDiskInfo] for detailed reporting
    """
    global HOST_DISK_INFO_CACHE

    available = {}
    details = {}

    for host in hosts:
        if host in HOST_DISK_INFO_CACHE:
            info = HOST_DISK_INFO_CACHE[host]
        else:
            if host == "local":
                total_gb, available_gb = get_local_disk_gb(ringrift_ai_dir)
            elif host in REMOTE_HOSTS:
                total_gb, available_gb = get_remote_disk_gb(host, REMOTE_HOSTS[host])
            else:
                total_gb, available_gb = 0, 0

            info = HostDiskInfo(total_gb=total_gb, available_gb=available_gb)
            HOST_DISK_INFO_CACHE[host] = info

        available[host] = info.available_gb
        details[host] = info

    return available, details


def get_eligible_hosts_for_board(
    board_type: str,
    hosts: list[str],
    host_memory: dict[str, int],
    *,
    host_disk_available: dict[str, int] | None = None,
    min_disk_gb: int = 0,
) -> list[str]:
    """Return list of hosts that have enough memory and disk for the given board type."""
    required_memory = BOARD_MEMORY_REQUIREMENTS.get(board_type, 8)
    eligible = []

    for host in hosts:
        host_mem = host_memory.get(host, 8)
        if host_mem >= required_memory:
            if host != "local" and min_disk_gb > 0 and host_disk_available is not None and host_disk_available.get(host, 0) < min_disk_gb:
                continue
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
    difficulty_band: str
    engine_mode: str
    output_db: str
    log_jsonl: str
    seed: int


def generate_job_configs(
    games_per_config: int,
    hosts: list[str],
    output_dir: str,
    base_seed: int = 42,
    difficulty_band: str = "light",
    engine_mode: str = "mixed",
    host_memory: dict[str, int] | None = None,
    host_disk_available: dict[str, int] | None = None,
    min_remote_free_disk_gb: int = 0,
    allowed_board_types: list[str] | None = None,
    allowed_num_players: list[int] | None = None,
) -> list[JobConfig]:
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
            eligible_hosts = get_eligible_hosts_for_board(
                board_type,
                hosts,
                host_memory,
                host_disk_available=host_disk_available,
                min_disk_gb=min_remote_free_disk_gb,
            )
            if not eligible_hosts:
                required = BOARD_MEMORY_REQUIREMENTS.get(board_type, 8)
                disk_note = ""
                if min_remote_free_disk_gb > 0:
                    disk_note = f" and at least {min_remote_free_disk_gb}GB free disk"
                print(f"Warning: No hosts have enough memory ({required}GB){disk_note} for {board_type}")
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

                jobs.append(
                    JobConfig(
                        job_id=job_id,
                        host=host,
                        board_type=board_type,
                        num_players=num_players,
                        num_games=host_games,
                        max_moves=max_moves,
                        difficulty_band=difficulty_band,
                        engine_mode=engine_mode,
                        output_db=os.path.join(output_dir, f"selfplay_{config_id}_{host}.db"),
                        log_jsonl=os.path.join(output_dir, f"selfplay_{config_id}_{host}.jsonl"),
                        seed=job_seed,
                    )
                )
                job_idx += 1

    return jobs


def build_soak_command(job: JobConfig, is_remote: bool = False) -> str:
    """Build the self-play soak command for a job."""
    python_exe = "python3" if is_remote else shlex.quote(sys.executable)
    # Distributed runs often execute on remote GPU hosts without Node.js/npx.
    # On-the-fly TS parity validation would fail there and prevent any games from
    # being recorded. We rely on the controller-side parity gate instead.
    parity_mode = os.getenv("RINGRIFT_PARITY_VALIDATION", "strict")
    if is_remote:
        parity_mode = os.getenv("RINGRIFT_REMOTE_PARITY_VALIDATION", "off")
    cmd_parts = [
        "PYTHONPATH=.",
        "RINGRIFT_SKIP_SHADOW_CONTRACTS=true",
        # Canonical self-play invariants / parity enforcement.
        # Keep these strict by default so distributed soaks produce debuggable,
        # parity-safe recordings (especially for training data).
        "RINGRIFT_STRICT_NO_MOVE_INVARIANT=1",
        f"RINGRIFT_PARITY_VALIDATION={parity_mode}",
        "RINGRIFT_FORCE_BOOKKEEPING_MOVES=1",
        # Keep OpenMP usage conservative for stability across hosts/containers.
        "OMP_NUM_THREADS=1",
        "MKL_NUM_THREADS=1",
    ]
    # Large-board heuristic evaluation can auto-enable multiprocessing and will
    # otherwise spawn `cpu_count()` worker processes (catastrophic on huge-core
    # hosts like Vast, and often slower due to overhead). Default to a safe cap
    # unless the caller explicitly configures `RINGRIFT_PARALLEL_WORKERS`.
    if job.board_type in ("square19", "hexagonal"):
        raw_parallel_workers = os.getenv("RINGRIFT_PARALLEL_WORKERS")
        if not (raw_parallel_workers or "").strip():
            cmd_parts.append("RINGRIFT_PARALLEL_WORKERS=8")
    # Propagate recovery-stack-strike flag across hosts when explicitly set.
    #
    # Canonical default is enabled; setting this to 0 is a non-canonical ablation
    # mode for baseline comparisons. For distributed runs we must make the
    # setting explicit to avoid mixing semantics across machines.
    raw_stack_strike = os.getenv("RINGRIFT_RECOVERY_STACK_STRIKE_V1")
    if raw_stack_strike is not None:
        raw_norm = raw_stack_strike.strip().lower()
        enabled = raw_norm in ("1", "true", "yes", "on")
        cmd_parts.append(f"RINGRIFT_RECOVERY_STACK_STRIKE_V1={'1' if enabled else '0'}")
    cmd_parts.extend([
        python_exe,
        "scripts/run_self_play_soak.py",
        f"--num-games {job.num_games}",
        f"--board-type {job.board_type}",
        f"--num-players {job.num_players}",
        f"--max-moves {job.max_moves}",
        f"--seed {job.seed}",
        f"--engine-mode {job.engine_mode}",
        f"--difficulty-band {job.difficulty_band}",
        f"--log-jsonl {job.log_jsonl}",
        f"--record-db {job.output_db}",
        "--fail-on-anomaly",
        "--verbose 10",
        "--gc-interval 5",
    ])

    # Add memory constraints for large boards
    if job.board_type in ("square19", "hexagonal"):
        cmd_parts.extend(["--streaming-record", "--intra-game-gc-interval 50"])
        cmd_parts.append("--memory-constrained")

    return " ".join(cmd_parts)


def run_local_job(job: JobConfig, ringrift_ai_dir: str, *, timeout_seconds: int) -> tuple[str, bool, str]:
    """Run a self-play job on the local machine."""
    cmd = build_soak_command(job)

    print(f"[LOCAL] Starting job {job.job_id}: {job.num_games} games of " f"{job.board_type} {job.num_players}p")

    try:
        run_kwargs = {
            "args": cmd,
            "shell": True,
            "cwd": ringrift_ai_dir,
            "capture_output": True,
            "text": True,
        }
        if timeout_seconds and timeout_seconds > 0:
            run_kwargs["timeout"] = int(timeout_seconds)
        result = subprocess.run(**run_kwargs)
        success = result.returncode == 0
        output = result.stdout + result.stderr

        if success:
            print(f"[LOCAL] Job {job.job_id} completed successfully")
        else:
            print(f"[LOCAL] Job {job.job_id} failed with code {result.returncode}")

        return job.job_id, success, output

    except subprocess.TimeoutExpired:
        print(f"[LOCAL] Job {job.job_id} timed out")
        return job.job_id, False, f"Job timed out after {timeout_seconds} seconds"
    except Exception as e:
        print(f"[LOCAL] Job {job.job_id} error: {e}")
        return job.job_id, False, str(e)


def run_remote_job(job: JobConfig, host_config: dict, *, timeout_seconds: int) -> tuple[str, bool, str]:
    """Run a self-play job on a remote machine via SSH."""
    ssh_host = host_config["ssh_host"]
    ssh_user = host_config.get("ssh_user")
    ssh_port = host_config.get("ssh_port")
    ai_service_dir = _remote_ai_service_dir(host_config)
    venv_activate = host_config.get("venv_activate", "")
    ssh_key = host_config.get("ssh_key")

    # Build SSH target (user@host or just host)
    ssh_target = f"{ssh_user}@{ssh_host}" if ssh_user else ssh_host

    # Build remote command
    soak_cmd = build_soak_command(job, is_remote=True)
    activate_cmd = _format_remote_venv_activate(venv_activate)
    remote_parts = [
        "set -euo pipefail",
        f"cd {_quote_remote_path(ai_service_dir)}",
    ]
    if activate_cmd:
        remote_parts.append(activate_cmd)
    remote_parts.append(soak_cmd)
    remote_script = "; ".join(remote_parts)
    remote_cmd = f"bash -lc {shlex.quote(remote_script)}"

    ssh_cmd = [
        "ssh",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "ServerAliveInterval=10",
        "-o",
        "ServerAliveCountMax=6",
        "-o",
        "BatchMode=yes",
    ]
    if ssh_port:
        ssh_cmd.extend(["-p", str(ssh_port)])
    if ssh_key:
        ssh_cmd.extend(["-i", os.path.expanduser(ssh_key)])
    ssh_cmd.extend([ssh_target, remote_cmd])

    print(
        f"[{ssh_host.upper()}] Starting job {job.job_id}: {job.num_games} games of "
        f"{job.board_type} {job.num_players}p"
    )

    try:
        run_args = {"capture_output": True, "text": True}
        if timeout_seconds and timeout_seconds > 0:
            run_args["timeout"] = int(timeout_seconds)
        result = subprocess.run(ssh_cmd, **run_args)
        success = result.returncode == 0
        output = (
            f"[ssh] target={ssh_target} returncode={result.returncode}\n"
            + (result.stdout or "")
            + (result.stderr or "")
        )

        if success:
            print(f"[{ssh_host.upper()}] Job {job.job_id} completed successfully")
        else:
            print(f"[{ssh_host.upper()}] Job {job.job_id} failed with code {result.returncode}")

        return job.job_id, success, output

    except subprocess.TimeoutExpired:
        print(f"[{ssh_host.upper()}] Job {job.job_id} timed out")
        return job.job_id, False, f"Job timed out after {timeout_seconds} seconds"
    except Exception as e:
        print(f"[{ssh_host.upper()}] Job {job.job_id} error: {e}")
        return job.job_id, False, str(e)


def run_job(job: JobConfig, ringrift_ai_dir: str, *, timeout_seconds: int) -> tuple[str, bool, str]:
    """Dispatch job to appropriate host."""
    if job.host == "local":
        return run_local_job(job, ringrift_ai_dir, timeout_seconds=timeout_seconds)
    elif job.host in REMOTE_HOSTS:
        return run_remote_job(job, REMOTE_HOSTS[job.host], timeout_seconds=timeout_seconds)
    else:
        return job.job_id, False, f"Unknown host: {job.host}"


def cleanup_remote_job_artifacts(
    job: JobConfig,
    host_config: dict,
    *,
    cleanup_jsonl: bool,
    timeout_seconds: int,
) -> bool:
    """Delete per-job artifacts on the remote host to conserve disk space."""
    ssh_host = host_config["ssh_host"]
    ssh_user = host_config.get("ssh_user")
    ssh_port = host_config.get("ssh_port")
    ssh_key = host_config.get("ssh_key")
    ai_service_dir = _remote_ai_service_dir(host_config)

    ssh_target = f"{ssh_user}@{ssh_host}" if ssh_user else ssh_host

    rm_paths = [
        job.output_db,
        f"{job.output_db}-wal",
        f"{job.output_db}-shm",
    ]
    if cleanup_jsonl:
        rm_paths.append(job.log_jsonl)

    rm_arg = shlex.join(rm_paths)
    remote_dir = os.path.dirname(job.output_db) or "."

    remote_parts = [
        "set -euo pipefail",
        f"cd {_quote_remote_path(ai_service_dir)}",
        f"rm -f {rm_arg}",
        f"rmdir {shlex.quote(remote_dir)} 2>/dev/null || true",
    ]
    remote_script = "; ".join(remote_parts)
    remote_cmd = f"bash -lc {shlex.quote(remote_script)}"

    ssh_cmd = [
        "ssh",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "ServerAliveInterval=60",
        "-o",
        "BatchMode=yes",
    ]
    if ssh_port:
        ssh_cmd.extend(["-p", str(ssh_port)])
    if ssh_key:
        ssh_cmd.extend(["-i", os.path.expanduser(ssh_key)])
    ssh_cmd.extend([ssh_target, remote_cmd])

    try:
        subprocess.run(
            ssh_cmd,
            check=True,
            capture_output=True,
            timeout=int(timeout_seconds) if timeout_seconds and timeout_seconds > 0 else None,
            text=True,
        )
        return True
    except Exception as e:
        print(f"  -> Remote cleanup failed for {job.job_id} ({ssh_host}): {e}")
        return False


def _fetch_single_job(
    job: JobConfig,
    output_dir: str,
    fetch_jsonl: bool,
    fetch_timeout_seconds: int,
    cleanup_remote: bool,
) -> str | None:
    """Fetch a single job's database from remote host. Thread-safe.

    Returns:
        Local database path if successful, None otherwise.
    """
    if job.host == "local":
        local_db = os.path.join(output_dir, os.path.basename(job.output_db))
        if os.path.exists(local_db):
            return local_db
        return None

    if job.host not in REMOTE_HOSTS:
        return None

    host_config = REMOTE_HOSTS[job.host]
    ssh_host = host_config["ssh_host"]
    ssh_user = host_config.get("ssh_user")
    ssh_port = host_config.get("ssh_port")
    ssh_key = host_config.get("ssh_key")
    ai_service_dir = _remote_ai_service_dir(host_config)

    # Build SSH target (user@host or just host)
    ssh_target = f"{ssh_user}@{ssh_host}" if ssh_user else ssh_host

    remote_db = f"{ai_service_dir}/{job.output_db}"
    local_db = os.path.join(output_dir, os.path.basename(job.output_db))

    print(f"[{job.host}] Fetching {remote_db}...")

    db_fetched = False
    jsonl_fetched = False

    # Acquire sync_lock for coordinated file transfers to this host
    sync_lock_acquired = acquire_sync_lock(ssh_host, timeout=60.0)
    if not sync_lock_acquired:
        print(f"[{job.host}] Warning: Could not acquire sync lock, proceeding anyway")

    try:
        # Fetch database
        scp_cmd = ["scp"]
        if ssh_key:
            scp_cmd.extend(["-i", os.path.expanduser(ssh_key)])
        if ssh_port:
            scp_cmd.extend(["-P", str(ssh_port)])
        scp_cmd.extend([f"{ssh_target}:{remote_db}", local_db])

        try:
            subprocess.run(
                scp_cmd,
                check=True,
                capture_output=True,
                timeout=int(fetch_timeout_seconds) if fetch_timeout_seconds and fetch_timeout_seconds > 0 else None,
            )
            print(f"[{job.host}] -> Saved to {local_db}")
            db_fetched = True
        except subprocess.CalledProcessError as e:
            print(f"[{job.host}] -> Failed to fetch: {e}")
        except subprocess.TimeoutExpired:
            print(f"[{job.host}] -> Fetch timed out")

        # Fetch JSONL if requested
        if fetch_jsonl:
            remote_jsonl = f"{ai_service_dir}/{job.log_jsonl}"
            local_jsonl = os.path.join(output_dir, os.path.basename(job.log_jsonl))
            print(f"[{job.host}] Fetching {remote_jsonl}...")

            scp_cmd = ["scp"]
            if ssh_key:
                scp_cmd.extend(["-i", os.path.expanduser(ssh_key)])
            if ssh_port:
                scp_cmd.extend(["-P", str(ssh_port)])
            scp_cmd.extend([f"{ssh_target}:{remote_jsonl}", local_jsonl])

            try:
                subprocess.run(
                    scp_cmd,
                    check=True,
                    capture_output=True,
                    timeout=int(fetch_timeout_seconds) if fetch_timeout_seconds and fetch_timeout_seconds > 0 else None,
                )
                print(f"[{job.host}] -> Saved JSONL to {local_jsonl}")
                jsonl_fetched = True
            except subprocess.CalledProcessError as e:
                print(f"[{job.host}] -> Failed to fetch JSONL: {e}")
            except subprocess.TimeoutExpired:
                print(f"[{job.host}] -> JSONL fetch timed out")
        else:
            jsonl_fetched = True
    finally:
        # Release sync_lock
        if sync_lock_acquired:
            release_sync_lock(ssh_host)

    # Cleanup remote if both succeeded
    if cleanup_remote and db_fetched and jsonl_fetched:
        print(f"[{job.host}] Cleaning up remote artifacts for {job.job_id}...")
        if cleanup_remote_job_artifacts(
            job,
            host_config,
            cleanup_jsonl=fetch_jsonl,
            timeout_seconds=max(30, int(fetch_timeout_seconds) if fetch_timeout_seconds else 60),
        ):
            print(f"[{job.host}] -> Remote cleanup complete")

    return local_db if db_fetched else None


def fetch_remote_results(
    jobs: list[JobConfig],
    output_dir: str,
    *,
    fetch_jsonl: bool,
    fetch_timeout_seconds: int,
    cleanup_remote: bool,
    parallel_workers: int = 8,
) -> list[str]:
    """Fetch database files from remote hosts in parallel.

    Uses ThreadPoolExecutor to download from multiple hosts simultaneously,
    significantly reducing total fetch time for large clusters.

    Args:
        parallel_workers: Number of parallel download threads (default 8).

    Returns:
        List of successfully fetched local database paths.
    """
    if not jobs:
        return []

    # Filter to remote jobs only for parallel fetching
    remote_jobs = [j for j in jobs if j.host != "local" and j.host in REMOTE_HOSTS]
    local_jobs = [j for j in jobs if j.host == "local"]

    fetched_dbs = []

    # Handle local jobs immediately
    for job in local_jobs:
        local_db = os.path.join(output_dir, os.path.basename(job.output_db))
        if os.path.exists(local_db):
            fetched_dbs.append(local_db)

    if not remote_jobs:
        return fetched_dbs

    print(f"\nFetching {len(remote_jobs)} remote databases with {parallel_workers} parallel workers...")

    # Parallel fetch for remote jobs
    with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
        futures = {
            executor.submit(
                _fetch_single_job,
                job,
                output_dir,
                fetch_jsonl,
                fetch_timeout_seconds,
                cleanup_remote,
            ): job
            for job in remote_jobs
        }

        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
                if result is not None:
                    fetched_dbs.append(result)
            except Exception as e:
                print(f"[{job.host}] Exception during fetch: {e}")

    print(f"Fetched {len(fetched_dbs)} databases total ({len(fetched_dbs) - len(local_jobs)} remote, {len([j for j in local_jobs if os.path.exists(os.path.join(output_dir, os.path.basename(j.output_db)))])} local)")
    return fetched_dbs


def run_parity_checks(db_paths: list[str], ringrift_ai_dir: str) -> tuple[int, int]:
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
                    sys.executable,
                    "scripts/check_ts_python_replay_parity.py",
                    "--db",
                    db_path,
                ],
                cwd=ringrift_ai_dir,
                capture_output=True,
                text=True,
                timeout=600,  # 10 min timeout per DB
                env={**os.environ, "PYTHONPATH": ".", "RINGRIFT_SKIP_SHADOW_CONTRACTS": "true"},
            )

            if result.returncode == 0:
                print("  ✓ PASSED")
                passed += 1
            else:
                print("  ✗ FAILED")
                # Print last few lines of error
                error_lines = result.stderr.strip().split("\n")[-5:]
                for line in error_lines:
                    print(f"    {line}")
                failed += 1

        except subprocess.TimeoutExpired:
            print("  ✗ TIMEOUT (exceeded 10 minutes)")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            failed += 1

    print()
    print("=" * 60)
    print(f"Parity check results: {passed} passed, {failed} failed")

    return passed, failed


def main():
    # Use unified parser if available, otherwise fall back to standard argparse
    if HAS_SELFPLAY_CONFIG and create_argument_parser is not None:
        parser = create_argument_parser(
            description="Run distributed self-play soaks across multiple machines",
            include_gpu=False,
            include_ramdrive=False,
        )
    else:
        parser = argparse.ArgumentParser(description="Run distributed self-play soaks across multiple machines")
        # Add base args that would come from unified parser
        parser.add_argument("--output-dir", type=str, default="data/games/distributed_soak")
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--difficulty-band", type=str, choices=["canonical", "light", "full"], default="light")
        parser.add_argument("--engine-mode", type=str, default="mixed")
        parser.add_argument("--max-parallel-per-host", type=int, default=2)
        parser.add_argument("--hosts", type=str, default="local")

    # Add soak-specific arguments
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
        "--base-seed",
        type=int,
        default=42,
        help="Base random seed for reproducibility",
    )
    parser.add_argument(
        "--soak-engine-mode",
        type=str,
        choices=[
            "mixed",
            "diverse",           # GPU-optimized diverse AI (all 11 types)
            "diverse-cpu",       # CPU-optimized diverse AI
            "descent-only",
            "heuristic-only",
            "minimax-only",
            "mcts-only",
            "nn-only",
            "gpu-minimax-only",
            "maxn-only",
            "brs-only",
            "policy-only",
            "gumbel-mcts-only",
        ],
        default="mixed",
        dest="soak_engine_mode",
        help="Engine mode for soak selfplay jobs (default: mixed).",
    )
    parser.add_argument(
        "--job-timeout-seconds",
        type=int,
        default=7200,
        help="Per-job wall-clock timeout in seconds (default: 7200). Use 0 to disable.",
    )
    parser.add_argument(
        "--fetch-timeout-seconds",
        type=int,
        default=900,
        help="Timeout for scp fetches in seconds (default: 900). Use 0 to disable.",
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
        "--fetch-jsonl",
        action="store_true",
        help="Also fetch per-job JSONL logs from remote hosts.",
    )
    parser.add_argument(
        "--cleanup-remote",
        action="store_true",
        help="After successfully fetching remote artifacts, delete them to conserve disk space.",
    )
    parser.add_argument(
        "--min-remote-free-disk-gb",
        type=int,
        default=2,
        help="Minimum free disk (GB) required on remote hosts to schedule jobs (default: 2).",
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

    parsed = parser.parse_args()

    # Create SelfplayConfig if available (for tracking/logging)
    if HAS_SELFPLAY_CONFIG and SelfplayConfig is not None:
        SelfplayConfig(
            board_type=getattr(parsed, "board", "square8"),
            num_players=2,  # Will vary per job
            num_games=parsed.games_per_config,
            output_dir=parsed.output_dir,
            seed=getattr(parsed, "seed", parsed.base_seed),
            difficulty_band=parsed.difficulty_band,
            hosts=getattr(parsed, "hosts", "local"),
            max_parallel_per_host=parsed.max_parallel_per_host,
            source="run_distributed_selfplay_soak.py",
            extra_options={
                "mode": parsed.mode,
                "board_types": parsed.board_types,
                "soak_engine_mode": parsed.soak_engine_mode,
                "job_timeout_seconds": parsed.job_timeout_seconds,
                "dry_run": parsed.dry_run,
            },
        )

    # Create backward-compatible args object
    args = type("Args", (), {
        "mode": parsed.mode,
        "games_per_config": parsed.games_per_config,
        "board_types": parsed.board_types,
        "num_players": getattr(parsed, "num_players", None),
        "hosts": getattr(parsed, "hosts", "local"),
        "output_dir": parsed.output_dir,
        "base_seed": parsed.base_seed,
        "difficulty_band": parsed.difficulty_band,
        "engine_mode": parsed.soak_engine_mode,
        "max_parallel_per_host": parsed.max_parallel_per_host,
        "job_timeout_seconds": parsed.job_timeout_seconds,
        "fetch_timeout_seconds": parsed.fetch_timeout_seconds,
        "dry_run": parsed.dry_run,
        "skip_fetch": parsed.skip_fetch,
        "fetch_jsonl": parsed.fetch_jsonl,
        "cleanup_remote": parsed.cleanup_remote,
        "min_remote_free_disk_gb": parsed.min_remote_free_disk_gb,
        "run_parity": parsed.run_parity,
        "config": parsed.config,
    })()

    # Entry point resource validation (enforced 2025-12-16)
    # Check local resources before starting distributed work
    if HAS_RESOURCE_GUARD and not args.dry_run and not resource_can_proceed(check_disk=True, check_mem=True):
        print("ERROR: Insufficient local resources to start distributed soak.")
        print("       Disk or memory usage exceeds 80% limit.")
        print("       Free up resources or use --dry-run to preview jobs.")
        sys.exit(1)

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
            available = ["local", *list(REMOTE_HOSTS.keys())]
            print(f"Error: Unknown host '{host}'.")
            print(f"Available hosts: {', '.join(available)}")
            if not REMOTE_HOSTS:
                print("\nNo remote hosts configured. See: config/distributed_hosts.template.yaml")
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
        eligible_boards = [board for board, req in BOARD_MEMORY_REQUIREMENTS.items() if mem_gb >= req]
        print(
            f"  {host}: {info.total_gb}GB total, {info.available_gb}GB available -> eligible for: {', '.join(eligible_boards)}"
        )
    print()

    # Detect disk on all hosts (ai-service filesystem)
    print("Detecting host disk free space...")
    host_disk_available, host_disk_details = detect_all_host_disk(hosts, ringrift_ai_dir)
    print("Host disk configuration (ai-service mount):")
    for host, info in sorted(host_disk_details.items(), key=lambda x: -x[1].available_gb):
        print(f"  {host}: {info.total_gb}GB total, {info.available_gb}GB available")
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
        difficulty_band=args.difficulty_band,
        engine_mode=args.engine_mode,
        host_memory=host_memory,
        host_disk_available=host_disk_available,
        min_remote_free_disk_gb=int(args.min_remote_free_disk_gb) if args.min_remote_free_disk_gb else 0,
        allowed_board_types=allowed_board_types,
        allowed_num_players=allowed_num_players,
    )

    print(f"\n{'='*60}")
    print("Distributed Self-Play Soak Configuration")
    print(f"{'='*60}")
    print(f"Games per config: {args.games_per_config}")
    print(f"Difficulty band: {args.difficulty_band}")
    print(f"Hosts: {', '.join(hosts)}")
    print(f"Output directory: {output_dir}")
    print(f"Total jobs: {len(jobs)}")
    print(f"Total games: {sum(j.num_games for j in jobs)}")
    print()

    # Print job summary
    print("Job Summary:")
    print("-" * 60)
    for job in jobs:
        print(f"  {job.job_id}: {job.board_type} {job.num_players}p x{job.num_games} " f"-> {job.output_db}")
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
    failures_dir = os.path.join(output_dir, "failures")
    os.makedirs(failures_dir, exist_ok=True)

    # Group jobs by host for parallel execution
    jobs_by_host: dict[str, list[JobConfig]] = {}
    for job in jobs:
        if job.host not in jobs_by_host:
            jobs_by_host[job.host] = []
        jobs_by_host[job.host].append(job)

    # Run jobs with limited parallelism per host
    with ThreadPoolExecutor(max_workers=len(hosts) * args.max_parallel_per_host) as executor:
        futures = {}
        for job in jobs:
            future = executor.submit(run_job, job, ringrift_ai_dir, timeout_seconds=args.job_timeout_seconds)
            futures[future] = job

        for future in as_completed(futures):
            job_id, success, output = future.result()
            results.append((job_id, success, output))
            if not success:
                job = futures.get(future)
                safe_name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", job_id)
                path = os.path.join(failures_dir, f"{safe_name}.log")
                try:
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(output)
                    host_label = getattr(job, "host", "?") if job else "?"
                    print(f"[FAILURE] wrote log: {path} (host={host_label})")
                except Exception as exc:
                    print(f"[FAILURE] could not write failure log for {job_id}: {exc}")

    elapsed = time.time() - start_time

    # Summary
    print()
    print("=" * 60)
    print("Distributed Self-Play Complete")
    print("=" * 60)
    print(f"Elapsed time: {elapsed/60:.1f} minutes")

    successful = sum(1 for _, s, _ in results if s)
    failed = len(results) - successful

    print(f"Successful jobs: {successful}/{len(results)}")
    if failed > 0:
        print(f"Failed jobs: {failed}")
        for job_id, success, _output in results:
            if not success:
                print(f"  - {job_id}")

    # Fetch results from remote hosts
    fetched_dbs = []
    if not args.skip_fetch:
        print()
        print("Fetching results from remote hosts...")
        fetched_dbs = fetch_remote_results(
            jobs,
            output_dir,
            fetch_jsonl=bool(args.fetch_jsonl),
            fetch_timeout_seconds=args.fetch_timeout_seconds,
            cleanup_remote=bool(args.cleanup_remote),
        )
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
        "job_results": [{"job_id": jid, "success": s} for jid, s, _ in results],
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
        print("     cd ai-service && python scripts/check_ts_python_replay_parity.py \\")
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
