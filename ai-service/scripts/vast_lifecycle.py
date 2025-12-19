#!/usr/bin/env python3
"""Vast.ai Instance Lifecycle Manager - Automates instance management.

Handles:
1. Health checks on running instances
2. Restarting stuck/idle workers
3. Terminating unproductive instances
4. Collecting game data before termination

Usage:
    python scripts/vast_lifecycle.py --check       # Check instance health
    python scripts/vast_lifecycle.py --restart     # Restart stuck workers
    python scripts/vast_lifecycle.py --sync        # Sync data and terminate idle
    python scripts/vast_lifecycle.py --auto        # Full automation cycle

Designed to be run via cron every 30-60 minutes.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = AI_SERVICE_ROOT / "logs"
LOG_FILE = LOG_DIR / "vast_lifecycle.log"

if str(AI_SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_SERVICE_ROOT))

# Lambda target for data sync - prefer Tailscale IP for reliability
# Public IP (150.136.65.197) gets connection resets under load
LAMBDA_HOST = "100.91.25.13"  # Tailscale IP (was: lambda-a10 / 150.136.65.197)
LAMBDA_SSH_USER = "ubuntu"
LAMBDA_DB = "/home/ubuntu/ringrift/ai-service/data/games/selfplay.db"

# Rate limiting - prevent connection saturation on Lambda host
SYNC_DELAY_SECONDS = 5  # Delay between instance syncs

# Thresholds
MIN_GAMES_PER_HOUR = 5  # Minimum game production rate
IDLE_THRESHOLD_HOURS = 2  # Mark as idle if no games for this long
MAX_INSTANCE_AGE_HOURS = 48  # Consider termination after this time

# GPU to board type mapping - assign appropriate workloads based on GPU capability
# Updated 2025-12-18: Rebalanced to generate more square19/hexagonal data
GPU_BOARD_MAPPING = {
    # Small GPUs (<=8GB) - fast hex8 games (training data)
    "RTX 3070": "hex8",
    "RTX 2060S": "hex8",
    "RTX 2060 SUPER": "hex8",
    "RTX 3060 Ti": "hex8",
    "RTX 2080 Ti": "hex8",
    "RTX 3060": "hex8",
    # Mid-range GPUs (12-16GB) - split between square8 and square19
    "RTX 4060 Ti": "square8",  # Keep some on square8
    "RTX 4080S": "square19",   # Redirect to square19 (need more data)
    "RTX 4080 SUPER": "square19",
    "RTX 5080": "square19",    # Redirect to square19 (need more data)
    # High-end GPUs (24GB+) - large hexagonal boards
    "A40": "hexagonal",
    "RTX 5090": "hexagonal",
    "RTX 5070": "hexagonal",
    "A10": "hexagonal",
    "H100": "hexagonal",
}

# Default board type for unknown GPUs
DEFAULT_BOARD_TYPE = "square8"


def get_vast_instances() -> List[Dict]:
    """Dynamically get all running vast instances from vastai CLI."""
    try:
        # Try to find vastai executable
        vastai_paths = [
            "/Users/armand/.pyenv/versions/3.10.13/bin/vastai",
            "vastai",
            os.path.expanduser("~/.local/bin/vastai"),
        ]

        vastai_cmd = None
        for path in vastai_paths:
            try:
                result = subprocess.run([path, "--version"], capture_output=True, timeout=5)
                if result.returncode == 0:
                    vastai_cmd = path
                    break
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue

        if not vastai_cmd:
            logger.warning("vastai CLI not found, using fallback instance list")
            return _get_fallback_instances()

        # Get instances as JSON
        result = subprocess.run(
            [vastai_cmd, "show", "instances", "--raw"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            logger.warning(f"vastai show instances failed: {result.stderr}")
            return _get_fallback_instances()

        instances_data = json.loads(result.stdout)

        instances = []
        for inst in instances_data:
            if inst.get("actual_status") != "running":
                continue

            gpu_name = inst.get("gpu_name", "unknown")
            num_gpus = inst.get("num_gpus", 1)

            instance_id = str(inst.get("id", "unknown"))
            instances.append({
                "id": instance_id,
                "host": inst.get("ssh_host"),
                "port": inst.get("ssh_port"),
                "name": f"vast-{instance_id[:8]}",
                "gpu": f"{num_gpus}x {gpu_name}",
                "gpu_name": gpu_name,
                "num_gpus": num_gpus,
                "vcpus": inst.get("cpu_cores_effective", 0),
                "board_type": GPU_BOARD_MAPPING.get(gpu_name, DEFAULT_BOARD_TYPE),
            })

        logger.info(f"Discovered {len(instances)} running vast instances")
        return instances

    except Exception as e:
        logger.error(f"Error getting vast instances: {e}")
        return _get_fallback_instances()


def _get_fallback_instances() -> List[Dict]:
    """Fallback hardcoded instance list if vastai CLI unavailable."""
    return [
        {"host": "ssh5.vast.ai", "port": 14364, "name": "vast-3070a", "gpu": "RTX 3070", "board_type": "hex8"},
        {"host": "ssh2.vast.ai", "port": 14370, "name": "vast-2060s", "gpu": "RTX 2060S", "board_type": "hex8"},
        {"host": "ssh1.vast.ai", "port": 14400, "name": "vast-4060ti", "gpu": "RTX 4060 Ti", "board_type": "square8"},
        {"host": "ssh3.vast.ai", "port": 19766, "name": "vast-3060ti", "gpu": "RTX 3060 Ti", "board_type": "hex8"},
        {"host": "ssh2.vast.ai", "port": 19768, "name": "vast-4060ti-48", "gpu": "RTX 4060 Ti", "board_type": "square8"},
        {"host": "ssh3.vast.ai", "port": 19940, "name": "vast-4080s", "gpu": "2x RTX 4080S", "board_type": "square8"},
        {"host": "ssh1.vast.ai", "port": 19942, "name": "vast-5080", "gpu": "RTX 5080", "board_type": "square8"},
        {"host": "ssh7.vast.ai", "port": 10012, "name": "vast-3070b", "gpu": "RTX 3070", "board_type": "hex8"},
        {"host": "ssh9.vast.ai", "port": 10014, "name": "vast-2080ti", "gpu": "RTX 2080 Ti", "board_type": "hex8"},
        {"host": "ssh8.vast.ai", "port": 17016, "name": "vast-3060ti-512", "gpu": "2x RTX 3060 Ti", "board_type": "square8"},
        {"host": "ssh3.vast.ai", "port": 38740, "name": "vast-3060", "gpu": "4x RTX 3060", "board_type": "hex8"},
        {"host": "ssh8.vast.ai", "port": 38742, "name": "vast-a40", "gpu": "A40", "board_type": "hexagonal"},
        {"host": "ssh2.vast.ai", "port": 10042, "name": "vast-5070", "gpu": "4x RTX 5070", "board_type": "hexagonal"},
        {"host": "ssh1.vast.ai", "port": 15166, "name": "vast-5090", "gpu": "RTX 5090", "board_type": "hexagonal"},
        {"host": "ssh5.vast.ai", "port": 18168, "name": "vast-5090x8", "gpu": "8x RTX 5090", "board_type": "hexagonal"},
    ]

from scripts.lib.logging_config import setup_logging, get_logger

setup_logging(level="INFO", log_file=LOG_FILE)
logger = get_logger("vast_lifecycle")


def run_ssh_command(
    host: str, port: int, command: str, timeout: int = 30
) -> Tuple[bool, str]:
    """Run SSH command on remote host."""
    try:
        result = subprocess.run(
            [
                "ssh",
                "-o", "StrictHostKeyChecking=accept-new",
                "-o", f"ConnectTimeout={min(timeout, 10)}",
                "-p", str(port),
                f"root@{host}",
                command,
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode == 0, result.stdout.strip()
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, str(e)


def check_instance_health(instance: Dict) -> Dict:
    """Check health of a single Vast instance."""
    host, port = instance["host"], instance["port"]
    name = instance["name"]

    health = {
        "name": name,
        "host": host,
        "port": port,
        "reachable": False,
        "workers_running": 0,
        "tournament_running": 0,
        "training_running": 0,
        "games_count": 0,
        "last_game_age_hours": None,
        "status": "unknown",
    }

    # Check reachability
    success, _ = run_ssh_command(host, port, "echo ok", timeout=15)
    if not success:
        health["status"] = "unreachable"
        return health

    health["reachable"] = True

    # Check running selfplay workers
    success, output = run_ssh_command(
        host, port,
        "pgrep -fa 'diverse_selfplay|selfplay' | grep -v pgrep | wc -l",
        timeout=15,
    )
    if success:
        try:
            health["workers_running"] = int(output.strip())
        except ValueError:
            health["workers_running"] = 0

    # Check running tournament/evaluation processes (valuable work)
    success, output = run_ssh_command(
        host, port,
        "pgrep -fa 'elo_tournament|gauntlet|run_model_elo' | grep -v pgrep | wc -l",
        timeout=15,
    )
    if success:
        try:
            health["tournament_running"] = int(output.strip())
        except ValueError:
            health["tournament_running"] = 0

    # Check running training processes (valuable work)
    success, output = run_ssh_command(
        host, port,
        "pgrep -fa 'training_loop|train_model|run_tier_training' | grep -v pgrep | wc -l",
        timeout=15,
    )
    if success:
        try:
            health["training_running"] = int(output.strip())
        except ValueError:
            health["training_running"] = 0

    # Check game count and age (look in all possible DB locations)
    success, output = run_ssh_command(
        host, port,
        """python3 -c "
import sqlite3
import glob
from datetime import datetime
total = 0
min_age = 999
# Check multiple possible DB locations
paths = glob.glob('/root/ringrift/ai-service/data/games/*.db')
paths += glob.glob('/root/ringrift/ai-service/data/selfplay/canonical/*.db')
for path in paths:
    try:
        conn = sqlite3.connect(path)
        count = conn.execute('SELECT COUNT(*) FROM games').fetchone()[0]
        last = conn.execute('SELECT MAX(created_at) FROM games').fetchone()[0]
        conn.close()
        total += count
        if last:
            age = (datetime.utcnow() - datetime.fromisoformat(last.replace('Z',''))).total_seconds() / 3600
            min_age = min(min_age, age)
    except (sqlite3.Error, OSError, ValueError):
        pass
print(f'{total}|{min_age:.2f}')
" """,
        timeout=20,
    )
    if success and "|" in output:
        parts = output.split("|")
        try:
            health["games_count"] = int(parts[0])
            health["last_game_age_hours"] = float(parts[1])
        except (ValueError, IndexError):
            pass

    # Determine status - tournament/training work counts as valuable
    has_valuable_work = (
        health["workers_running"] > 0 or
        health["tournament_running"] > 0 or
        health["training_running"] > 0
    )

    if not has_valuable_work:
        health["status"] = "no_workers"
    elif health["tournament_running"] > 0:
        health["status"] = "tournament"  # Running Elo evaluation
    elif health["training_running"] > 0:
        health["status"] = "training"  # Running model training
    elif health["games_count"] == 0 and health["workers_running"] > 0:
        health["status"] = "no_games"  # Workers running but no games yet
    elif health["last_game_age_hours"] and health["last_game_age_hours"] > IDLE_THRESHOLD_HOURS:
        health["status"] = "idle"
    else:
        health["status"] = "healthy"

    return health


def sync_git_repo(instance: Dict) -> bool:
    """Sync git repository on a Vast instance."""
    host, port = instance["host"], instance["port"]
    name = instance.get("name", "unknown")

    # Canonical path for git repo
    repo_path = "/root/ringrift"

    logger.info(f"Syncing git repo on {name}...")

    # Check if repo exists
    success, output = run_ssh_command(
        host, port,
        f"test -d {repo_path}/.git && echo 'exists'",
        timeout=10,
    )

    if success and "exists" in output:
        # Repo exists, do git pull
        success, output = run_ssh_command(
            host, port,
            f"cd {repo_path} && git fetch origin && git reset --hard origin/main && git log -1 --oneline",
            timeout=60,
        )
        if success:
            logger.info(f"  {name}: Updated to {output.strip()}")
            return True
        else:
            logger.warning(f"  {name}: Git pull failed - {output}")
            return False
    else:
        # Clone fresh
        success, output = run_ssh_command(
            host, port,
            f"rm -rf {repo_path} && git clone --depth 1 https://github.com/an0mium/RingRift.git {repo_path} && "
            f"cd {repo_path} && git log -1 --oneline",
            timeout=120,
        )
        if success:
            logger.info(f"  {name}: Cloned fresh - {output.strip()}")
            # Set up venv after fresh clone
            run_ssh_command(
                host, port,
                f"cd {repo_path}/ai-service && python3 -m venv venv && "
                f"venv/bin/pip install -q -r requirements.txt",
                timeout=300,
            )
            return True
        else:
            logger.warning(f"  {name}: Git clone failed - {output}")
            return False


def restart_workers(instance: Dict, sync_code: bool = True) -> bool:
    """Restart selfplay workers on an instance."""
    host, port = instance["host"], instance["port"]
    board_type = instance.get("board_type", DEFAULT_BOARD_TYPE)
    gpu_name = instance.get("gpu", "unknown")
    name = instance.get("name", "unknown")

    logger.info(f"Restarting workers on {name} ({gpu_name}) with board_type={board_type}...")

    # Sync git repo first (ensures latest code)
    if sync_code:
        sync_git_repo(instance)

    # Kill existing workers
    run_ssh_command(host, port, "pkill -f 'generate_data|selfplay' || true", timeout=15)

    # Determine num_games based on board type (larger boards = fewer games)
    num_games = {"hex8": 2000, "square8": 1500, "hexagonal": 500}.get(board_type, 1000)

    # Canonical path for git repo
    path = "/root/ringrift/ai-service"

    # Verify path exists
    success, output = run_ssh_command(
        host, port,
        f"test -d {path} && echo 'found'",
        timeout=10,
    )
    if not success or "found" not in output:
        logger.warning(f"  {name}: Code path not found at {path}")
        return False

    # Determine engine based on board type and model availability
    # - square8: use mcts with neural network (GPU) if models exist
    # - hex8/hexagonal: use descent (CPU) until models are trained
    if board_type == "square8":
        engine = "mcts"
        model_arg = "--nn-model-id ringrift_v5_sq8_2p"  # Use latest square8 model
    else:
        engine = "descent"
        model_arg = ""

    # Start new workers with GPU-appropriate board type and engine
    # RINGRIFT_DISABLE_TORCH_COMPILE=1 avoids triton compilation issues on some vast images
    success, output = run_ssh_command(
        host, port,
        f"""cd {path} &&
        mkdir -p data/games logs models &&
        source venv/bin/activate 2>/dev/null || true &&
        PYTHONPATH=. RINGRIFT_DISABLE_TORCH_COMPILE=1 nohup python3 -m app.training.generate_data \\
            --board-type {board_type} --num-games {num_games} \\
            --engine {engine} {model_arg} \\
            --record-db data/games/selfplay_{board_type}_{name}.db \\
            > logs/selfplay_{board_type}.log 2>&1 &
        sleep 2 && pgrep -f generate_data | head -1
        """,
        timeout=90,  # Longer timeout for venv activation + process start
    )

    if success and output.strip():
        logger.info(f"  Started PID {output.strip()} on {name}")
        return True
    else:
        logger.warning(f"  Failed to start workers on {name}: {output}")
        return False


def sync_data_from_instance(instance: Dict) -> int:
    """Sync game data from instance to Lambda."""
    host, port = instance["host"], instance["port"]
    name = instance["name"]
    temp_dir = f"/tmp/vast_sync_{name}"

    logger.info(f"Syncing data from {name}...")

    # Create temp dir
    os.makedirs(temp_dir, exist_ok=True)

    # Find and download all DBs
    success, output = run_ssh_command(
        host, port,
        "find /root/ringrift/ai-service/data -name '*.db' -type f 2>/dev/null",
        timeout=15,
    )
    if not success:
        logger.warning(f"Failed to list DBs on {name}")
        return 0

    db_paths = [p.strip() for p in output.split("\n") if p.strip() and ".db" in p]
    if not db_paths:
        logger.info(f"No DBs found on {name}")
        return 0

    total_count = 0
    local_dbs = []

    for remote_path in db_paths:
        db_name = os.path.basename(remote_path)
        local_path = f"{temp_dir}/{db_name}"

        try:
            result = subprocess.run(
                [
                    "scp",
                    "-o", "StrictHostKeyChecking=accept-new",
                    "-o", "ConnectTimeout=10",
                    "-P", str(port),
                    f"root@{host}:{remote_path}",
                    local_path,
                ],
                capture_output=True,
                timeout=60,
            )
            if result.returncode == 0:
                # Count games
                try:
                    import sqlite3
                    conn = sqlite3.connect(local_path)
                    count = conn.execute("SELECT COUNT(*) FROM games").fetchone()[0]
                    conn.close()
                    if count > 0:
                        total_count += count
                        local_dbs.append(local_path)
                        logger.info(f"  {db_name}: {count} games")
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"Failed to download {db_name}: {e}")

    # Merge all DBs into Lambda
    if local_dbs:
        try:
            db_args = " ".join([f"--db {db}" for db in local_dbs])
            # First copy DBs to Lambda (using Tailscale IP for reliability)
            lambda_dest = f"{LAMBDA_SSH_USER}@{LAMBDA_HOST}"
            for local_db in local_dbs:
                subprocess.run(
                    ["scp", "-o", "ConnectTimeout=10", local_db, f"{lambda_dest}:/tmp/"],
                    timeout=60,
                )
            # Then merge
            remote_dbs = " ".join([f"--db /tmp/{os.path.basename(db)}" for db in local_dbs])
            subprocess.run(
                [
                    "ssh", "-o", "ConnectTimeout=10", lambda_dest,
                    f"cd /home/ubuntu/ringrift/ai-service && "
                    f"python3 scripts/merge_game_dbs.py --output {LAMBDA_DB} "
                    f"--dedupe-by-game-id {remote_dbs}",
                ],
                timeout=180,
            )
            logger.info(f"Merged {total_count} games from {name}")
        except Exception as e:
            logger.error(f"Merge error: {e}")

    # Cleanup
    try:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        pass

    return total_count


def run_health_check() -> List[Dict]:
    """Run health check on all instances."""
    logger.info("=" * 60)
    logger.info("VAST.AI INSTANCE HEALTH CHECK")
    logger.info("=" * 60)

    instances = get_vast_instances()
    logger.info(f"Checking {len(instances)} instances...")

    results = []
    for instance in instances:
        health = check_instance_health(instance)
        # Preserve instance metadata
        health["board_type"] = instance.get("board_type", DEFAULT_BOARD_TYPE)
        health["gpu"] = instance.get("gpu", "unknown")
        results.append(health)

        status_emoji = {
            "healthy": "✓",
            "tournament": "EVAL",  # Running Elo tournament evaluation
            "training": "TRAIN",  # Running model training
            "idle": "IDLE",
            "no_workers": "NO_WRK",
            "no_games": "NO_GAM",
            "unreachable": "OFFLN",
            "unknown": "?",
        }.get(health["status"], "?")

        logger.info(
            f"  {health['name']:<15} [{status_emoji:<6}] "
            f"{health.get('gpu', '?'):<20} "
            f"board={health.get('board_type', '?'):<10} "
            f"workers={health['workers_running']} "
            f"games={health['games_count']}"
        )

    return results


def run_restart_cycle(health_results: List[Dict]) -> int:
    """Restart workers on unhealthy instances.

    Skips instances that are running tournaments or training - these are doing
    valuable work and should not be interrupted.
    """
    restarted = 0
    for health in health_results:
        # Only restart truly idle/broken instances
        # Skip tournament/training - they're doing valuable work
        if health["status"] in ("no_workers", "idle", "no_games"):
            instance = {
                "host": health["host"],
                "port": health["port"],
                "name": health["name"],
                "board_type": health.get("board_type", DEFAULT_BOARD_TYPE),
                "gpu": health.get("gpu", "unknown"),
            }
            if restart_workers(instance):
                restarted += 1
    return restarted


def run_sync_cycle(health_results: List[Dict]) -> int:
    """Sync data from all reachable instances.

    Includes rate limiting (SYNC_DELAY_SECONDS between syncs) to prevent
    connection saturation on the Lambda aggregation host.
    """
    import time

    total_synced = 0
    sync_count = 0
    for health in health_results:
        if health["reachable"] and health["games_count"] > 0:
            # Rate limit: delay between syncs to prevent connection saturation
            if sync_count > 0 and SYNC_DELAY_SECONDS > 0:
                logger.debug(f"Rate limit: waiting {SYNC_DELAY_SECONDS}s before next sync")
                time.sleep(SYNC_DELAY_SECONDS)

            instance = {
                "host": health["host"],
                "port": health["port"],
                "name": health["name"],
            }
            synced = sync_data_from_instance(instance)
            total_synced += synced
            sync_count += 1

    logger.info(f"Total games synced: {total_synced} from {sync_count} instances")
    return total_synced


def run_auto_cycle():
    """Full automation cycle: check, restart, sync."""
    logger.info("=" * 60)
    logger.info(f"VAST.AI AUTO CYCLE - {datetime.now().isoformat()}")
    logger.info("=" * 60)

    # Health check
    health_results = run_health_check()

    # Sync data first (before any restarts)
    synced = run_sync_cycle(health_results)

    # Restart unhealthy instances
    restarted = run_restart_cycle(health_results)

    # Summary
    total = len(health_results)
    healthy = sum(1 for h in health_results if h["status"] == "healthy")
    tournament = sum(1 for h in health_results if h["status"] == "tournament")
    training = sum(1 for h in health_results if h["status"] == "training")
    unreachable = sum(1 for h in health_results if h["status"] == "unreachable")
    no_workers = sum(1 for h in health_results if h["status"] == "no_workers")
    idle = sum(1 for h in health_results if h["status"] == "idle")
    productive = healthy + tournament + training  # All doing valuable work

    # Count by board type
    board_counts = {}
    for h in health_results:
        bt = h.get("board_type", "unknown")
        board_counts[bt] = board_counts.get(bt, 0) + 1

    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info(f"  Total instances: {total}")
    logger.info(f"  Productive: {productive} (selfplay={healthy}, eval={tournament}, training={training})")
    logger.info(f"  Issues: unreachable={unreachable}, no_workers={no_workers}, idle={idle}")
    logger.info(f"  Restarted: {restarted}, Games synced: {synced}")
    logger.info(f"  Board distribution: {board_counts}")
    logger.info("=" * 60)


def run_code_sync() -> int:
    """Sync git repos on all reachable Vast instances."""
    logger.info("=" * 60)
    logger.info("VAST.AI CODE SYNC (GIT)")
    logger.info("=" * 60)

    instances = get_vast_instances()
    logger.info(f"Syncing code to {len(instances)} instances...")

    synced = 0
    for instance in instances:
        # First check if reachable
        success, _ = run_ssh_command(instance["host"], instance["port"], "echo ok", timeout=15)
        if not success:
            logger.warning(f"  {instance.get('name', 'unknown')}: unreachable")
            continue

        if sync_git_repo(instance):
            synced += 1

    logger.info(f"Code synced to {synced}/{len(instances)} instances")
    return synced


def main():
    parser = argparse.ArgumentParser(description="Vast.ai lifecycle manager")
    parser.add_argument("--check", action="store_true", help="Check instance health")
    parser.add_argument("--restart", action="store_true", help="Restart stuck workers")
    parser.add_argument("--sync", action="store_true", help="Sync data from instances")
    parser.add_argument("--sync-code", action="store_true", help="Sync git repos to latest")
    parser.add_argument("--auto", action="store_true", help="Full automation cycle")
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if args.auto:
        run_auto_cycle()
    elif args.check:
        run_health_check()
    elif args.restart:
        health = run_health_check()
        run_restart_cycle(health)
    elif args.sync:
        health = run_health_check()
        run_sync_cycle(health)
    elif args.sync_code:
        run_code_sync()
    else:
        # Default to check
        run_health_check()


if __name__ == "__main__":
    main()
