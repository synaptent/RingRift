#!/usr/bin/env python3
"""Vast.ai P2P Manager - Manage P2P orchestrator on Vast instances.

This script provides reliable P2P orchestrator management for Vast.ai instances:
- Discovers running Vast instances via vastai CLI
- Starts/stops/restarts P2P orchestrator on instances
- Uses robust startup approach that survives SSH disconnection
- Monitors health and selfplay status

Usage:
    python scripts/vast_p2p_manager.py status          # Check all instances
    python scripts/vast_p2p_manager.py start           # Start P2P on all
    python scripts/vast_p2p_manager.py start --id 123  # Start on specific instance
    python scripts/vast_p2p_manager.py stop            # Stop P2P on all
    python scripts/vast_p2p_manager.py restart         # Restart P2P on all
"""

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class VastInstance:
    """Vast.ai instance info."""
    id: int
    ssh_host: str
    ssh_port: int
    gpu_name: str
    num_gpus: int
    status: str


def get_vast_instances() -> List[VastInstance]:
    """Get list of running Vast instances."""
    try:
        result = subprocess.run(
            ['vastai', 'show', 'instances', '--raw'],
            capture_output=True, text=True, timeout=30
        )
        instances = json.loads(result.stdout)
        return [
            VastInstance(
                id=inst.get('id'),
                ssh_host=inst.get('ssh_host', ''),
                ssh_port=inst.get('ssh_port', 22),
                gpu_name=inst.get('gpu_name', 'Unknown'),
                num_gpus=inst.get('num_gpus', 0) or 0,
                status=inst.get('actual_status', 'unknown'),
            )
            for inst in instances
            if inst.get('actual_status') == 'running' and inst.get('ssh_host')
        ]
    except Exception as e:
        print(f"Error getting Vast instances: {e}")
        return []


def run_ssh_command(inst: VastInstance, command: str, timeout: int = 30, retries: int = 2) -> Tuple[bool, str]:
    """Run SSH command on a Vast instance with retries."""
    ssh_cmd = [
        'ssh',
        '-o', 'StrictHostKeyChecking=no',
        '-o', 'ConnectTimeout=20',
        '-o', 'ServerAliveInterval=10',
        '-o', 'BatchMode=yes',
        '-p', str(inst.ssh_port),
        f'root@{inst.ssh_host}',
        command
    ]
    for attempt in range(retries + 1):
        try:
            result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=timeout)
            if result.returncode == 0:
                return True, result.stdout.strip()
            if attempt < retries:
                time.sleep(2)
        except subprocess.TimeoutExpired:
            if attempt < retries:
                time.sleep(2)
                continue
            return False, "Timeout"
        except Exception as e:
            if attempt < retries:
                time.sleep(2)
                continue
            return False, str(e)
    return False, "Failed after retries"


def check_p2p_health(inst: VastInstance) -> Tuple[bool, int, float]:
    """Check P2P health on instance. Returns (is_healthy, selfplay_jobs, disk_percent)."""
    ok, output = run_ssh_command(
        inst,
        'curl -s http://localhost:8770/health 2>/dev/null',
        timeout=20
    )
    if not ok or not output:
        return False, 0, 0.0

    try:
        health = json.loads(output)
        return (
            'healthy' in str(health),
            health.get('selfplay_jobs', 0),
            health.get('disk_percent', 0.0),
        )
    except:
        return False, 0, 0.0


def start_p2p(inst: VastInstance, skip_kill: bool = False) -> bool:
    """Start P2P orchestrator on instance using robust startup approach."""
    node_id = f"vast-{inst.id}"

    # Kill existing P2P if requested
    if not skip_kill:
        run_ssh_command(inst, 'pkill -f "p2p_orchestrator.py" 2>/dev/null || true', timeout=15, retries=1)
        time.sleep(2)

    # Use screen for robust process management (survives SSH disconnect)
    startup_cmd = f'''
screen -dmS p2p bash -c '
cd /root/ringrift/ai-service 2>/dev/null || cd ~/ringrift/ai-service
source venv/bin/activate 2>/dev/null || true
export PYTHONPATH="$PWD"
python scripts/p2p_orchestrator.py --node-id {node_id} --port 8770 2>&1 | tee /tmp/p2p.log
'
echo "STARTED"
'''

    ok, output = run_ssh_command(inst, startup_cmd, timeout=30, retries=1)
    if not ok:
        return False

    # Wait for startup with multiple health checks
    for i in range(4):  # Check up to 4 times over 20 seconds
        time.sleep(5)
        healthy, _, _ = check_p2p_health(inst)
        if healthy:
            return True

    return False


def stop_p2p(inst: VastInstance) -> bool:
    """Stop P2P orchestrator on instance."""
    ok, _ = run_ssh_command(
        inst,
        'pkill -f "p2p_orchestrator.py" 2>/dev/null; echo OK',
        timeout=15
    )
    return ok


def print_status(instances: List[VastInstance]):
    """Print status of all instances."""
    print(f"\n{'ID':<12} {'GPU':<20} {'#':<3} {'P2P':<6} {'Selfplay':<10} {'Disk':<6}")
    print("-" * 60)

    total_ok = 0
    total_selfplay = 0
    total_gpus = 0

    def check_one(inst):
        healthy, selfplay, disk = check_p2p_health(inst)
        return inst, healthy, selfplay, disk

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(check_one, inst) for inst in instances]
        for future in as_completed(futures):
            inst, healthy, selfplay, disk = future.result()
            total_gpus += inst.num_gpus

            if healthy:
                total_ok += 1
                total_selfplay += selfplay
                status = "OK"
            else:
                status = "OFF"

            print(f"{inst.id:<12} {inst.gpu_name[:18]:<20} {inst.num_gpus:<3} {status:<6} {selfplay if healthy else '-':<10} {f'{disk:.0f}%' if healthy else '-':<6}")

    print("-" * 60)
    print(f"P2P Running: {total_ok}/{len(instances)} | GPUs: {total_gpus} | Selfplay: {total_selfplay}")


def main():
    parser = argparse.ArgumentParser(description='Manage P2P orchestrator on Vast instances')
    parser.add_argument('action', choices=['status', 'start', 'stop', 'restart'],
                       help='Action to perform')
    parser.add_argument('--id', type=int, help='Specific instance ID (default: all)')
    parser.add_argument('--parallel', type=int, default=5, help='Parallel workers')
    parser.add_argument('--skip-running', action='store_true',
                       help='Only start on instances without running P2P (start action only)')
    args = parser.parse_args()

    print("Discovering Vast instances...")
    instances = get_vast_instances()

    if not instances:
        print("No running Vast instances found")
        return 1

    print(f"Found {len(instances)} running instances")

    # Filter to specific instance if requested
    if args.id:
        instances = [i for i in instances if i.id == args.id]
        if not instances:
            print(f"Instance {args.id} not found")
            return 1

    if args.action == 'status':
        print_status(instances)

    elif args.action == 'start':
        # Filter out already-running instances if --skip-running
        if args.skip_running:
            print("Checking which instances need P2P started...")
            offline_instances = []
            with ThreadPoolExecutor(max_workers=10) as executor:
                def check_one(inst):
                    healthy, _, _ = check_p2p_health(inst)
                    return inst, healthy
                futures = [executor.submit(check_one, inst) for inst in instances]
                for future in as_completed(futures):
                    inst, healthy = future.result()
                    if not healthy:
                        offline_instances.append(inst)
            instances = offline_instances
            if not instances:
                print("All instances already have P2P running")
                return 0

        print(f"\nStarting P2P on {len(instances)} instances...")

        def start_one(inst):
            ok = start_p2p(inst, skip_kill=args.skip_running)
            return inst.id, ok

        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = [executor.submit(start_one, inst) for inst in instances]
            for future in as_completed(futures):
                inst_id, ok = future.result()
                print(f"  {inst_id}: {'OK' if ok else 'FAILED'}")

        print("\nFinal status:")
        print_status(get_vast_instances())

    elif args.action == 'stop':
        print(f"\nStopping P2P on {len(instances)} instances...")

        def stop_one(inst):
            ok = stop_p2p(inst)
            return inst.id, ok

        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = [executor.submit(stop_one, inst) for inst in instances]
            for future in as_completed(futures):
                inst_id, ok = future.result()
                print(f"  {inst_id}: {'OK' if ok else 'FAILED'}")

    elif args.action == 'restart':
        print(f"\nRestarting P2P on {len(instances)} instances...")

        def restart_one(inst):
            # start_p2p already kills existing P2P by default
            ok = start_p2p(inst, skip_kill=False)
            return inst.id, ok

        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = [executor.submit(restart_one, inst) for inst in instances]
            for future in as_completed(futures):
                inst_id, ok = future.result()
                print(f"  {inst_id}: {'OK' if ok else 'FAILED'}")

        print("\nFinal status:")
        print_status(instances)

    return 0


if __name__ == '__main__':
    sys.exit(main())
