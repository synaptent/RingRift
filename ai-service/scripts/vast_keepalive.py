#!/usr/bin/env python3
"""Vast.ai Keepalive Manager - Keeps instances active and integrated in P2P network.

Uses vast CLI to:
1. Monitor instance status and prevent idle termination
2. Auto-restart stopped instances
3. Maintain P2P network connectivity
4. Sync code and restart workers on unhealthy instances

Usage:
    python scripts/vast_keepalive.py --status              # Check all instances
    python scripts/vast_keepalive.py --keepalive           # Send keepalive to all
    python scripts/vast_keepalive.py --restart-stopped     # Restart stopped instances
    python scripts/vast_keepalive.py --auto                # Full automation cycle
    python scripts/vast_keepalive.py --install-cron        # Install cron job locally

Designed to be run via cron every 15-30 minutes.
"""

import argparse
import contextlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

from scripts.lib.logging_config import setup_script_logging
from scripts.lib.paths import AI_SERVICE_ROOT, LOGS_DIR
from scripts.lib.ssh import run_vast_ssh_command

LOG_FILE = LOGS_DIR / "vast_keepalive.log"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

logger = setup_script_logging("vast_keepalive")

# Vastai CLI path
VASTAI_CMD = "/Users/armand/.pyenv/versions/3.10.13/bin/vastai"

# Keepalive settings
KEEPALIVE_COMMAND = "echo 'keepalive' > /tmp/keepalive_$(date +%s)"
IDLE_CHECK_MINUTES = 30
MIN_WORKERS_REQUIRED = 1


def run_vastai_command(args: list[str], timeout: int = 30) -> tuple[bool, str]:
    """Run a vastai CLI command."""
    try:
        result = subprocess.run(
            [VASTAI_CMD, *args],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode == 0, result.stdout.strip() or result.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except FileNotFoundError:
        return False, f"vastai not found at {VASTAI_CMD}"
    except Exception as e:
        return False, str(e)


def get_all_instances() -> list[dict]:
    """Get all Vast instances (running and stopped)."""
    success, output = run_vastai_command(["show", "instances", "--raw"])
    if not success:
        logger.error(f"Failed to get instances: {output}")
        return []

    try:
        return json.loads(output)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse instances JSON: {e}")
        return []


def get_instance_details(instance_id: str) -> dict | None:
    """Get detailed info for a single instance."""
    success, output = run_vastai_command(["show", "instance", instance_id, "--raw"])
    if success:
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            pass
    return None


def start_instance(instance_id: str) -> bool:
    """Start a stopped instance."""
    logger.info(f"Starting instance {instance_id}...")
    success, output = run_vastai_command(["start", "instance", instance_id])
    if success:
        logger.info(f"Started instance {instance_id}")
    else:
        logger.error(f"Failed to start {instance_id}: {output}")
    return success


def stop_instance(instance_id: str) -> bool:
    """Stop a running instance."""
    logger.info(f"Stopping instance {instance_id}...")
    success, output = run_vastai_command(["stop", "instance", instance_id])
    if success:
        logger.info(f"Stopped instance {instance_id}")
    else:
        logger.error(f"Failed to stop {instance_id}: {output}")
    return success


def send_keepalive(instance: dict) -> tuple[str, bool, str]:
    """Send keepalive to an instance to prevent idle termination."""
    inst_id = str(instance.get("id", "unknown"))
    status = instance.get("actual_status", "unknown")
    host = instance.get("ssh_host")
    port = instance.get("ssh_port")

    if status != "running":
        return inst_id, False, f"status={status}"

    if not host or not port:
        return inst_id, False, "no SSH info"

    # Send keepalive command - this prevents idle detection
    success, output = run_vast_ssh_command(
        host, port,
        KEEPALIVE_COMMAND,
        timeout=15,
    )

    if success:
        return inst_id, True, "alive"
    return inst_id, False, output


def check_instance_health(instance: dict) -> dict:
    """Check health of a single instance."""
    inst_id = str(instance.get("id", "unknown"))
    status = instance.get("actual_status", "unknown")
    host = instance.get("ssh_host")
    port = instance.get("ssh_port")
    gpu = instance.get("gpu_name", "unknown")
    cost = instance.get("dph_total", 0)

    health = {
        "id": inst_id,
        "status": status,
        "gpu": gpu,
        "cost_per_hour": cost,
        "reachable": False,
        "workers": 0,
        "tailscale_ip": None,
        "p2p_running": False,
        "selfplay_running": False,
        "games_count": 0,
    }

    if status != "running":
        return health

    if not host or not port:
        return health

    # Check reachability
    success, _ = run_vast_ssh_command(host, port, "echo ok", timeout=10)
    if not success:
        return health
    health["reachable"] = True

    # Check Tailscale IP
    success, output = run_vast_ssh_command(host, port, "tailscale ip -4 2>/dev/null", timeout=10)
    if success and output.startswith("100."):
        health["tailscale_ip"] = output.strip()

    # Check P2P
    success, output = run_vast_ssh_command(host, port, "pgrep -c -f p2p_orchestrator 2>/dev/null || echo 0", timeout=10)
    health["p2p_running"] = success and int(output.strip() or "0") > 0

    # Check selfplay/workers
    success, output = run_vast_ssh_command(
        host, port,
        "pgrep -c -f 'generate_data|selfplay|gauntlet' 2>/dev/null || echo 0",
        timeout=10,
    )
    if success:
        health["workers"] = int(output.strip() or "0")
        health["selfplay_running"] = health["workers"] > 0

    # Check game count
    success, output = run_vast_ssh_command(
        host, port,
        """python3 -c "
import sqlite3, glob
total=0
for db in glob.glob('/root/ringrift/ai-service/data/**/*.db', recursive=True):
    try:
        conn=sqlite3.connect(db)
        total+=conn.execute('SELECT COUNT(*) FROM games').fetchone()[0]
        conn.close()
    except (sqlite3.Error, OSError):
        pass
print(total)
" 2>/dev/null || echo 0""",
        timeout=20,
    )
    if success:
        with contextlib.suppress(ValueError):
            health["games_count"] = int(output.strip())

    return health


def restart_workers_on_instance(instance: dict) -> bool:
    """Restart selfplay workers on an instance."""
    inst_id = str(instance.get("id", "unknown"))
    host = instance.get("ssh_host")
    port = instance.get("ssh_port")
    gpu = instance.get("gpu_name", "unknown")

    if not host or not port:
        return False

    logger.info(f"Restarting workers on {inst_id} ({gpu})...")

    # Determine board type based on GPU
    gpu_lower = gpu.lower()
    if any(x in gpu_lower for x in ["3060", "3070", "2060", "2080"]):
        board_type = "hex8"
    elif any(x in gpu_lower for x in ["a40", "5090", "h100"]):
        board_type = "hexagonal"
    else:
        board_type = "square8"

    # Restart workers
    success, output = run_vast_ssh_command(
        host, port,
        f"""cd /root/ringrift/ai-service && \\
        source venv/bin/activate 2>/dev/null && \\
        pkill -f 'generate_data|selfplay' || true && \\
        sleep 2 && \\
        mkdir -p data/games logs && \\
        PYTHONPATH=. nohup python3 -m app.training.generate_data \\
            --board-type {board_type} --num-games 1000 --engine descent \\
            --record-db data/games/selfplay_{board_type}.db \\
            > logs/selfplay.log 2>&1 & \\
        sleep 3 && pgrep -f generate_data | head -1""",
        timeout=60,
    )

    if success and output.strip():
        logger.info(f"  Started workers on {inst_id} (PID: {output.strip()})")
        return True
    logger.warning(f"  Failed to start workers on {inst_id}: {output}")
    return False


def sync_code_on_instance(instance: dict) -> bool:
    """Sync git code on an instance."""
    inst_id = str(instance.get("id", "unknown"))
    host = instance.get("ssh_host")
    port = instance.get("ssh_port")

    if not host or not port:
        return False

    success, output = run_vast_ssh_command(
        host, port,
        "cd /root/ringrift && git fetch origin && git reset --hard origin/main 2>&1 | tail -1",
        timeout=60,
    )

    if success:
        logger.info(f"  Synced code on {inst_id}: {output}")
        return True
    logger.warning(f"  Failed to sync code on {inst_id}: {output}")
    return False


def run_status_check() -> list[dict]:
    """Check status of all instances."""
    logger.info("=" * 70)
    logger.info("VAST.AI KEEPALIVE STATUS CHECK")
    logger.info("=" * 70)

    instances = get_all_instances()
    if not instances:
        logger.warning("No instances found")
        return []

    logger.info(f"Found {len(instances)} instances")

    # Check health in parallel
    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(check_instance_health, inst): inst for inst in instances}
        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda x: x["id"])

    # Print table
    print(f"\n{'ID':<12} {'Status':<10} {'GPU':<15} {'$/hr':<6} {'Reach':^5} {'TS IP':<14} {'P2P':^3} {'Work':^4} {'Games'}")
    print("-" * 95)

    total_cost = 0.0
    for r in results:
        status_short = r["status"][:8] if r["status"] else "?"
        gpu_short = r["gpu"][:13] if r["gpu"] else "?"
        reach = "✓" if r["reachable"] else "✗"
        ts_ip = (r["tailscale_ip"] or "-")[:14]
        p2p = "✓" if r["p2p_running"] else "✗"
        work = str(r["workers"]) if r["workers"] else "-"
        games = str(r["games_count"]) if r["games_count"] else "-"
        cost = f"${r['cost_per_hour']:.2f}" if r["cost_per_hour"] else "-"

        if r["status"] == "running":
            total_cost += r.get("cost_per_hour", 0)

        print(f"{r['id']:<12} {status_short:<10} {gpu_short:<15} {cost:<6} {reach:^5} {ts_ip:<14} {p2p:^3} {work:^4} {games}")

    # Summary
    running = sum(1 for r in results if r["status"] == "running")
    stopped = sum(1 for r in results if r["status"] == "stopped")
    reachable = sum(1 for r in results if r["reachable"])
    p2p_ok = sum(1 for r in results if r["p2p_running"])
    selfplay_ok = sum(1 for r in results if r["selfplay_running"])

    print("-" * 95)
    print(f"Summary: {running} running, {stopped} stopped, {reachable} reachable, {p2p_ok} P2P, {selfplay_ok} selfplay")
    print(f"Total hourly cost: ${total_cost:.2f}/hr (${total_cost * 24:.2f}/day)")

    return results


def run_keepalive_cycle() -> int:
    """Send keepalive to all running instances."""
    logger.info("Sending keepalive to all instances...")

    instances = get_all_instances()
    running = [i for i in instances if i.get("actual_status") == "running"]

    if not running:
        logger.warning("No running instances")
        return 0

    alive_count = 0
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(send_keepalive, inst): inst for inst in running}
        for future in as_completed(futures):
            inst_id, success, msg = future.result()
            if success:
                alive_count += 1
                logger.debug(f"  ✓ {inst_id}: {msg}")
            else:
                logger.warning(f"  ✗ {inst_id}: {msg}")

    logger.info(f"Keepalive sent to {alive_count}/{len(running)} instances")
    return alive_count


def run_restart_stopped() -> int:
    """Restart all stopped instances."""
    instances = get_all_instances()
    stopped = [i for i in instances if i.get("actual_status") == "stopped"]

    if not stopped:
        logger.info("No stopped instances to restart")
        return 0

    logger.info(f"Restarting {len(stopped)} stopped instances...")

    restarted = 0
    for inst in stopped:
        inst_id = str(inst.get("id", ""))
        if inst_id and start_instance(inst_id):
            restarted += 1

    logger.info(f"Restarted {restarted}/{len(stopped)} instances")
    return restarted


def run_auto_cycle():
    """Full automation cycle."""
    logger.info("=" * 70)
    logger.info(f"VAST.AI AUTO CYCLE - {datetime.now().isoformat()}")
    logger.info("=" * 70)

    # 1. Status check
    health_results = run_status_check()

    # 2. Keepalive
    run_keepalive_cycle()

    # 3. Restart stopped instances
    run_restart_stopped()

    # 4. Fix unhealthy instances
    for health in health_results:
        if health["status"] != "running":
            continue

        if health["reachable"] and not health["selfplay_running"]:
            # Sync code and restart workers
            inst = next((i for i in get_all_instances() if str(i.get("id")) == health["id"]), None)
            if inst:
                sync_code_on_instance(inst)
                restart_workers_on_instance(inst)

    # 5. Ensure P2P is running
    logger.info("\nEnsuring P2P network...")
    _success, _ = subprocess.run(
        ["python", str(AI_SERVICE_ROOT / "scripts" / "vast_p2p_setup.py"), "--deploy-to-vast", "--components", "p2p"],
        capture_output=True,
        timeout=180,
    ).returncode == 0, ""

    logger.info("=" * 70)
    logger.info("AUTO CYCLE COMPLETE")
    logger.info("=" * 70)


def install_local_cron():
    """Install cron job to run keepalive every 15 minutes."""
    script_path = Path(__file__).resolve()
    cron_line = f"*/15 * * * * cd {AI_SERVICE_ROOT} && /usr/bin/python3 {script_path} --auto >> {LOG_FILE} 2>&1"

    print("Add this line to your crontab (crontab -e):")
    print()
    print(cron_line)
    print()
    print(f"Or run: (crontab -l 2>/dev/null; echo '{cron_line}') | crontab -")


def main():
    parser = argparse.ArgumentParser(description="Vast.ai Keepalive Manager")
    parser.add_argument("--status", action="store_true", help="Check all instances")
    parser.add_argument("--keepalive", action="store_true", help="Send keepalive to all")
    parser.add_argument("--restart-stopped", action="store_true", help="Restart stopped instances")
    parser.add_argument("--auto", action="store_true", help="Full automation cycle")
    parser.add_argument("--install-cron", action="store_true", help="Show cron installation")
    args = parser.parse_args()

    if args.status:
        run_status_check()
    elif args.keepalive:
        run_keepalive_cycle()
    elif args.restart_stopped:
        run_restart_stopped()
    elif args.auto:
        run_auto_cycle()
    elif args.install_cron:
        install_local_cron()
    else:
        run_status_check()


if __name__ == "__main__":
    main()
