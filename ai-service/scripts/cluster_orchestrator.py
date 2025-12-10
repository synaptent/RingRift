#!/usr/bin/env python3
"""Enhanced Cluster Orchestrator - Manages all compute resources with resource-aware scheduling.

Features:
- Monitors all hosts: 2 Lambda, 2 AWS, 3 Vast.ai, 3 local Macs
- Resource-aware job scheduling (CPU, GPU, RAM, disk)
- Adaptive task allocation based on current utilization
- Automatic job restart on crash
- Consolidated logging and status dashboard

Usage:
    python scripts/cluster_orchestrator.py
    python scripts/cluster_orchestrator.py --dry-run
    python scripts/cluster_orchestrator.py --status-only
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

LOG_DIR = Path(__file__).parent.parent / "logs" / "orchestrator"
LOG_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = LOG_DIR / "cluster_state.json"
LOCKFILE = Path("/tmp/cluster_orchestrator.lock")

# Mac Studio sync configuration
MAC_STUDIO_HOST = os.environ.get("MAC_STUDIO_HOST", "mac-studio")
MAC_STUDIO_DATA_DIR = "~/Development/RingRift/ai-service/data/games"
SYNC_INTERVAL = 6  # Sync every 6 iterations (30 minutes at 5-min interval)


@dataclass
class HostConfig:
    """Configuration for a compute host."""
    name: str
    ssh_host: str
    ssh_user: str = "ubuntu"
    ssh_port: int = 22
    ssh_key: Optional[str] = None
    ringrift_path: str = "~/ringrift/ai-service"
    memory_gb: int = 16
    cpus: int = 4
    has_gpu: bool = False
    gpu_name: str = ""
    role: str = "selfplay"  # selfplay, training, cmaes, all
    storage_type: str = "disk"  # disk or ram (vast.ai /dev/shm)
    min_selfplay_jobs: int = 2
    enabled: bool = True


@dataclass
class HostStatus:
    """Runtime status of a host."""
    name: str
    reachable: bool = False
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    disk_percent: float = 0.0
    gpu_percent: float = 0.0
    gpu_memory_percent: float = 0.0
    python_jobs: int = 0
    selfplay_jobs: int = 0
    training_jobs: int = 0
    cmaes_jobs: int = 0
    last_check: str = ""
    error: str = ""


@dataclass
class ClusterState:
    """Persistent cluster state for resumability."""
    iteration: int = 0
    total_jobs_started: int = 0
    total_restarts: int = 0
    host_statuses: Dict[str, dict] = field(default_factory=dict)
    last_sync: str = ""
    errors: List[str] = field(default_factory=list)


def log(msg: str, level: str = "INFO"):
    """Log message with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] [{level}] {msg}"
    print(line, flush=True)
    with open(LOG_DIR / "orchestrator.log", "a") as f:
        f.write(line + "\n")


def load_hosts_config() -> List[HostConfig]:
    """Load host configurations from YAML files."""
    hosts = []
    config_dir = Path(__file__).parent.parent / "config"

    # Load distributed_hosts.yaml
    distributed_file = config_dir / "distributed_hosts.yaml"
    if distributed_file.exists():
        with open(distributed_file) as f:
            data = yaml.safe_load(f) or {}

        for name, cfg in data.get("hosts", {}).items():
            if cfg.get("status") == "ssh_key_issue":
                continue  # Skip hosts with known issues
            # Ensure path includes ai-service
            ringrift_path = cfg.get("ringrift_path", "~/ringrift/ai-service")
            if not ringrift_path.endswith("ai-service"):
                ringrift_path = ringrift_path + "/ai-service"
            hosts.append(HostConfig(
                name=name,
                ssh_host=cfg.get("ssh_host", ""),
                ssh_user=cfg.get("ssh_user", "ubuntu"),
                ssh_port=cfg.get("ssh_port", 22),
                ssh_key=cfg.get("ssh_key"),
                ringrift_path=ringrift_path,
                memory_gb=cfg.get("memory_gb", 16),
                cpus=cfg.get("cpus", 4),
                has_gpu="gpu" in cfg,
                gpu_name=cfg.get("gpu", ""),
                role=cfg.get("role", "selfplay"),
                min_selfplay_jobs=max(1, cfg.get("memory_gb", 16) // 32),  # 1 job per 32GB
            ))

    # Load remote_hosts.yaml for additional config
    remote_file = config_dir / "remote_hosts.yaml"
    if remote_file.exists():
        with open(remote_file) as f:
            data = yaml.safe_load(f) or {}

        # Add standard hosts
        for name, cfg in data.get("standard_hosts", {}).items():
            # Check if already added
            if any(h.name == name for h in hosts):
                continue
            hosts.append(HostConfig(
                name=name,
                ssh_host=cfg.get("ssh_host", ""),
                ssh_key=cfg.get("ssh_key"),
                ringrift_path=cfg.get("remote_path", "~/ringrift/ai-service").replace("/data/games", ""),
                memory_gb=cfg.get("memory_gb", 16),
                cpus=cfg.get("cpus", 4),
                has_gpu=cfg.get("has_gpu", False),
                role=cfg.get("role", "selfplay"),
                min_selfplay_jobs=max(1, cfg.get("memory_gb", 16) // 32),
            ))

        # Add vast hosts
        for name, cfg in data.get("vast_hosts", {}).items():
            hosts.append(HostConfig(
                name=name,
                ssh_host=cfg.get("host", ""),
                ssh_user=cfg.get("user", "root"),
                ssh_port=cfg.get("port", 22),
                ringrift_path="~/ringrift/ai-service",
                memory_gb=cfg.get("memory_gb", 16),
                cpus=cfg.get("cpus", 4),
                has_gpu=cfg.get("has_gpu", False),
                role=cfg.get("role", "selfplay"),
                storage_type=cfg.get("storage_type", "disk"),
                min_selfplay_jobs=max(2, cfg.get("cpus", 4) // 48),  # 1 job per 48 CPUs for vast
            ))

    return hosts


def ssh_cmd(host: HostConfig, cmd: str, timeout: int = 30) -> Tuple[int, str, str]:
    """Execute SSH command on host."""
    ssh_args = [
        "ssh",
        "-o", "ConnectTimeout=10",
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=no",
    ]

    if host.ssh_port != 22:
        ssh_args.extend(["-p", str(host.ssh_port)])

    if host.ssh_key:
        key_path = os.path.expanduser(host.ssh_key)
        if os.path.exists(key_path):
            ssh_args.extend(["-i", key_path])

    ssh_args.append(f"{host.ssh_user}@{host.ssh_host}")
    ssh_args.append(cmd)

    try:
        result = subprocess.run(
            ssh_args,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired:
        return 1, "", "SSH timeout"
    except Exception as e:
        return 1, "", str(e)


def check_host_status(host: HostConfig) -> HostStatus:
    """Get comprehensive status of a host."""
    status = HostStatus(name=host.name, last_check=datetime.now().isoformat())

    # Check reachability
    code, out, err = ssh_cmd(host, "echo ok", timeout=15)
    if code != 0:
        status.error = err or "Connection failed"
        return status

    status.reachable = True

    # Get resource usage with single SSH command
    metrics_cmd = """
    # CPU usage
    cpu=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' 2>/dev/null || echo "0")

    # Memory usage
    mem=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}' 2>/dev/null || echo "0")

    # Disk usage
    disk=$(df -h / | tail -1 | awk '{print $5}' | tr -d '%' 2>/dev/null || echo "0")

    # GPU usage (if nvidia-smi available)
    if command -v nvidia-smi &> /dev/null; then
        gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "0")
        gpu_mem=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | awk -F', ' '{printf "%.1f", $1/$2*100}' || echo "0")
    else
        gpu="0"
        gpu_mem="0"
    fi

    # Job counts
    selfplay=$(pgrep -f "selfplay|run_hybrid" | wc -l)
    training=$(pgrep -f "train_nnue|train.py" | wc -l)
    cmaes=$(pgrep -f "cmaes" | wc -l)
    python_total=$(pgrep -f "python" | wc -l)

    echo "$cpu|$mem|$disk|$gpu|$gpu_mem|$selfplay|$training|$cmaes|$python_total"
    """

    code, out, err = ssh_cmd(host, metrics_cmd, timeout=30)
    if code == 0 and out:
        try:
            parts = out.strip().split("|")
            if len(parts) >= 9:
                status.cpu_percent = float(parts[0] or 0)
                status.memory_percent = float(parts[1] or 0)
                status.disk_percent = float(parts[2] or 0)
                status.gpu_percent = float(parts[3] or 0)
                status.gpu_memory_percent = float(parts[4] or 0)
                status.selfplay_jobs = int(parts[5] or 0)
                status.training_jobs = int(parts[6] or 0)
                status.cmaes_jobs = int(parts[7] or 0)
                status.python_jobs = int(parts[8] or 0)
        except (ValueError, IndexError) as e:
            status.error = f"Parse error: {e}"

    return status


def should_start_selfplay(host: HostConfig, status: HostStatus) -> int:
    """Determine how many selfplay jobs to start based on resources."""
    if not status.reachable:
        return 0

    # Check resource constraints
    if status.disk_percent > 90:
        log(f"{host.name}: Disk usage critical ({status.disk_percent}%), skipping selfplay", "WARN")
        return 0

    if status.memory_percent > 85:
        log(f"{host.name}: Memory usage high ({status.memory_percent}%), reducing jobs", "WARN")
        return max(0, host.min_selfplay_jobs - status.selfplay_jobs - 1)

    # Calculate jobs needed
    jobs_needed = max(0, host.min_selfplay_jobs - status.selfplay_jobs)

    # Adjust based on CPU headroom
    if status.cpu_percent > 80:
        jobs_needed = min(jobs_needed, 1)

    return jobs_needed


def start_selfplay_jobs(host: HostConfig, count: int, dry_run: bool = False) -> int:
    """Start selfplay jobs on a host. Returns number started."""
    if count <= 0:
        return 0

    configs = [
        {"board": "square8", "players": 2, "games": 1000},
        {"board": "square8", "players": 3, "games": 500},
        {"board": "square8", "players": 4, "games": 300},
        {"board": "hex", "players": 2, "games": 500},
        {"board": "square19", "players": 2, "games": 300},
    ]

    started = 0
    seed_base = int(time.time())

    for i in range(count):
        cfg = configs[i % len(configs)]
        seed = seed_base + i

        output_dir = f"data/selfplay/{cfg['board']}_{cfg['players']}p"
        log_file = f"/tmp/selfplay_{cfg['board']}_{cfg['players']}p_{seed}.log"

        cmd = f"""cd {host.ringrift_path} && \\
            mkdir -p {output_dir} && \\
            nohup python3 scripts/run_hybrid_selfplay.py \\
                --num-games {cfg['games']} \\
                --board-type {cfg['board']} \\
                --num-players {cfg['players']} \\
                --output-dir {output_dir} \\
                --seed {seed} \\
                > {log_file} 2>&1 &
        """

        if dry_run:
            log(f"[DRY-RUN] Would start on {host.name}: {cfg['board']} {cfg['players']}p")
            started += 1
        else:
            code, out, err = ssh_cmd(host, cmd, timeout=30)
            if code == 0:
                log(f"{host.name}: Started {cfg['board']} {cfg['players']}p selfplay")
                started += 1
            else:
                log(f"{host.name}: Failed to start selfplay: {err}", "ERROR")

    return started


def check_training_status(host: HostConfig, status: HostStatus) -> Optional[str]:
    """Check if training should be started/continued. Returns action or None."""
    if "training" not in host.role and host.role != "all":
        return None

    if not host.has_gpu:
        return None

    if status.training_jobs > 0:
        return "running"

    # Check if there's a model to continue from
    code, out, err = ssh_cmd(host, f"ls {host.ringrift_path}/models/nnue/*.pt 2>/dev/null | wc -l")
    model_exists = code == 0 and out.strip() != "0"

    if model_exists:
        return "start_improvement"
    else:
        return "start_nnue"


def start_training(host: HostConfig, action: str, dry_run: bool = False) -> bool:
    """Start training job based on action."""
    if action == "running":
        return True

    if action == "start_nnue":
        cmd = f"""cd {host.ringrift_path} && \\
            mkdir -p models/nnue logs/nnue && \\
            nohup python3 scripts/train_nnue.py \\
                --db data/games/*.db \\
                --board-type square8 --num-players 2 \\
                --epochs 50 --batch-size 1024 \\
                --save-path models/nnue/square8_2p_v1.pt \\
                > logs/nnue/train_overnight.log 2>&1 &
        """
    elif action == "start_improvement":
        cmd = f"""cd {host.ringrift_path} && \\
            mkdir -p logs/improvement && \\
            nohup python3 scripts/run_improvement_loop.py \\
                --board square8 --players 2 \\
                --iterations 100 --games-per-iter 50 \\
                --promotion-threshold 0.55 --resume \\
                > logs/improvement/overnight.log 2>&1 &
        """
    else:
        return False

    if dry_run:
        log(f"[DRY-RUN] Would start {action} on {host.name}")
        return True

    code, out, err = ssh_cmd(host, cmd, timeout=30)
    if code == 0:
        log(f"{host.name}: Started {action}")
        return True
    else:
        log(f"{host.name}: Failed to start {action}: {err}", "ERROR")
        return False


def print_status_dashboard(hosts: List[HostConfig], statuses: Dict[str, HostStatus]):
    """Print a nice status dashboard."""
    print("\n" + "=" * 80)
    print(f"CLUSTER STATUS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print(f"{'Host':<18} {'Status':<8} {'CPU%':<6} {'MEM%':<6} {'Disk%':<6} {'GPU%':<6} {'Jobs':<12}")
    print("-" * 80)

    total_selfplay = 0
    total_training = 0
    reachable_count = 0

    for host in hosts:
        status = statuses.get(host.name)
        if not status:
            print(f"{host.name:<18} {'?':<8}")
            continue

        if status.reachable:
            reachable_count += 1
            total_selfplay += status.selfplay_jobs
            total_training += status.training_jobs

            jobs_str = f"S:{status.selfplay_jobs} T:{status.training_jobs} C:{status.cmaes_jobs}"
            print(f"{host.name:<18} {'OK':<8} {status.cpu_percent:<6.1f} {status.memory_percent:<6.1f} "
                  f"{status.disk_percent:<6.1f} {status.gpu_percent:<6.1f} {jobs_str:<12}")
        else:
            print(f"{host.name:<18} {'DOWN':<8} {'-':<6} {'-':<6} {'-':<6} {'-':<6} {status.error[:20]}")

    print("-" * 80)
    print(f"TOTALS: {reachable_count}/{len(hosts)} hosts up | {total_selfplay} selfplay | {total_training} training")
    print("=" * 80 + "\n")


def load_state() -> ClusterState:
    """Load persistent state."""
    if STATE_FILE.exists():
        try:
            data = json.loads(STATE_FILE.read_text())
            return ClusterState(**data)
        except Exception as e:
            log(f"Failed to load state: {e}", "WARN")
    return ClusterState()


def save_state(state: ClusterState):
    """Save persistent state."""
    STATE_FILE.write_text(json.dumps({
        "iteration": state.iteration,
        "total_jobs_started": state.total_jobs_started,
        "total_restarts": state.total_restarts,
        "host_statuses": state.host_statuses,
        "last_sync": state.last_sync,
        "errors": state.errors[-100:],  # Keep last 100 errors
    }, indent=2))


def acquire_lock() -> bool:
    """Acquire lockfile to prevent multiple instances."""
    if LOCKFILE.exists():
        try:
            pid = int(LOCKFILE.read_text().strip())
            # Check if process is still running
            os.kill(pid, 0)
            return False  # Process still running
        except (ValueError, ProcessLookupError, PermissionError):
            pass  # Stale lockfile, proceed

    LOCKFILE.write_text(str(os.getpid()))
    return True


def release_lock():
    """Release lockfile."""
    try:
        LOCKFILE.unlink()
    except FileNotFoundError:
        pass


def sync_to_mac_studio(hosts: List[HostConfig], dry_run: bool = False) -> bool:
    """Sync selfplay data from all reachable hosts directly to Mac Studio.

    Uses this machine as a relay: pulls data from cloud hosts, then pushes to Mac Studio.
    This avoids storing large amounts of data on the laptop.
    """
    log("Starting data sync to Mac Studio...")

    # Check Mac Studio reachability
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes",
             MAC_STUDIO_HOST, "echo ok"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            log(f"Mac Studio unreachable: {result.stderr}", "ERROR")
            return False
    except Exception as e:
        log(f"Mac Studio connection failed: {e}", "ERROR")
        return False

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mac_studio_sync_dir = f"{MAC_STUDIO_DATA_DIR}/synced_{timestamp}"

    if not dry_run:
        # Create sync directory on Mac Studio
        subprocess.run(
            ["ssh", MAC_STUDIO_HOST, f"mkdir -p {mac_studio_sync_dir}"],
            capture_output=True, timeout=30
        )

    synced_count = 0

    # Sync from each cloud host (skip local Macs and Mac Studio itself)
    cloud_hosts = [h for h in hosts if h.enabled and
                   "mac" not in h.name.lower() and
                   h.ssh_host and h.role != "training"]

    for host in cloud_hosts:
        try:
            log(f"Syncing from {host.name} to Mac Studio...")

            # Create temp directory for relay
            import tempfile
            with tempfile.TemporaryDirectory() as tmp_dir:
                # Pull from cloud host
                rsync_src = [
                    "rsync", "-avz", "--progress",
                    "-e", f"ssh -p {host.ssh_port} -o ConnectTimeout=15"
                ]

                # Try to sync .db files
                db_cmd = rsync_src + [
                    f"{host.ssh_user}@{host.ssh_host}:{host.ringrift_path}/data/games/*.db",
                    f"{tmp_dir}/"
                ]

                if dry_run:
                    log(f"  [DRY-RUN] Would sync from {host.name}")
                    continue

                result = subprocess.run(db_cmd, capture_output=True, text=True, timeout=300)

                # Check if any files were synced
                synced_files = list(Path(tmp_dir).glob("*.db"))
                if not synced_files:
                    log(f"  {host.name}: No .db files found")
                    continue

                # Push to Mac Studio
                dest_subdir = f"{mac_studio_sync_dir}/{host.name}"
                subprocess.run(
                    ["ssh", MAC_STUDIO_HOST, f"mkdir -p {dest_subdir}"],
                    capture_output=True, timeout=30
                )

                push_cmd = [
                    "rsync", "-avz", "--progress",
                    f"{tmp_dir}/",
                    f"{MAC_STUDIO_HOST}:{dest_subdir}/"
                ]
                result = subprocess.run(push_cmd, capture_output=True, text=True, timeout=300)

                if result.returncode == 0:
                    log(f"  {host.name}: Synced {len(synced_files)} file(s) to Mac Studio")
                    synced_count += 1
                else:
                    log(f"  {host.name}: Failed to push to Mac Studio: {result.stderr}", "WARN")

        except subprocess.TimeoutExpired:
            log(f"  {host.name}: Sync timeout", "WARN")
        except Exception as e:
            log(f"  {host.name}: Sync error: {e}", "WARN")

    if synced_count > 0:
        log(f"Data sync complete: {synced_count} host(s) synced to Mac Studio")
    else:
        log("Data sync: No data synced (hosts may be unreachable or have no data)")

    return synced_count > 0


def main():
    parser = argparse.ArgumentParser(description="Enhanced Cluster Orchestrator")
    parser.add_argument("--dry-run", action="store_true", help="Don't execute commands")
    parser.add_argument("--status-only", action="store_true", help="Just show status and exit")
    parser.add_argument("--interval", type=int, default=300, help="Check interval in seconds")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--sync-now", action="store_true", help="Force sync to Mac Studio immediately")
    parser.add_argument("--no-sync", action="store_true", help="Disable periodic sync to Mac Studio")
    args = parser.parse_args()

    # Acquire lockfile to prevent multiple instances
    if not acquire_lock():
        log("Another instance is already running (lockfile exists with active PID)", "ERROR")
        sys.exit(1)

    # Register cleanup handlers
    def cleanup(signum=None, frame=None):
        log("Shutting down...")
        release_lock()
        if signum:
            sys.exit(0)

    atexit.register(release_lock)
    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)

    log("=" * 60)
    log("CLUSTER ORCHESTRATOR STARTING")
    log(f"PID: {os.getpid()}")
    log("=" * 60)

    hosts = load_hosts_config()
    log(f"Loaded {len(hosts)} host configurations")

    state = load_state()

    # Force sync if requested
    if args.sync_now:
        log("Forcing sync to Mac Studio...")
        if sync_to_mac_studio(hosts, args.dry_run):
            state.last_sync = datetime.now().isoformat()
            save_state(state)
        return

    while True:
        state.iteration += 1
        log(f"--- Iteration {state.iteration} ---")

        # Collect status from all hosts
        statuses: Dict[str, HostStatus] = {}
        for host in hosts:
            if not host.enabled:
                continue
            status = check_host_status(host)
            statuses[host.name] = status
            state.host_statuses[host.name] = {
                "reachable": status.reachable,
                "cpu_percent": status.cpu_percent,
                "memory_percent": status.memory_percent,
                "disk_percent": status.disk_percent,
                "selfplay_jobs": status.selfplay_jobs,
                "training_jobs": status.training_jobs,
                "last_check": status.last_check,
            }

        # Print dashboard
        print_status_dashboard(hosts, statuses)

        if args.status_only:
            break

        # Manage each host
        for host in hosts:
            if not host.enabled:
                continue

            status = statuses.get(host.name)
            if not status or not status.reachable:
                continue

            # Selfplay management
            jobs_to_start = should_start_selfplay(host, status)
            if jobs_to_start > 0:
                started = start_selfplay_jobs(host, jobs_to_start, args.dry_run)
                state.total_jobs_started += started
                state.total_restarts += started

            # Training management (only on GPU hosts)
            if host.has_gpu:
                action = check_training_status(host, status)
                if action and action != "running":
                    start_training(host, action, args.dry_run)

        # Periodic sync to Mac Studio (every SYNC_INTERVAL iterations)
        if not args.no_sync and state.iteration % SYNC_INTERVAL == 0:
            log(f"Starting periodic sync (every {SYNC_INTERVAL} iterations)...")
            if sync_to_mac_studio(hosts, args.dry_run):
                state.last_sync = datetime.now().isoformat()

        save_state(state)

        if args.once:
            break

        log(f"Sleeping {args.interval}s until next check...")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
