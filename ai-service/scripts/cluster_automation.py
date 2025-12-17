#!/usr/bin/env python3
"""Cluster Automation - Continuous orchestration for the training loop.

This script runs the full automation cycle:
1. Health check all instances (Vast + Lambda)
2. Restart idle/stuck workers
3. Distribute new models via aria2
4. Start gauntlets for underrepresented configs
5. Sync training data

Usage:
    # Run full cycle
    python scripts/cluster_automation.py --full

    # Quick health check only
    python scripts/cluster_automation.py --check

    # Install cron job (runs every 15 minutes)
    python scripts/cluster_automation.py --install-cron

Designed to run via cron every 15 minutes.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = AI_SERVICE_ROOT / "logs"
LOG_FILE = LOG_DIR / "cluster_automation.log"

# Thresholds
IDLE_THRESHOLD_MINUTES = 30
MIN_GPU_UTIL = 5  # Consider idle if below this

# GPU to config mapping for selfplay
GPU_CONFIG = {
    "RTX 3070": "square8_2p",
    "RTX 2060S": "square8_2p",
    "RTX 3060 Ti": "square8_3p",
    "RTX 2080 Ti": "square8_2p",
    "RTX 3060": "square8_4p",
    "RTX 4060 Ti": "square19_2p",
    "RTX 4080S": "square19_2p",
    "RTX 5080": "square19_3p",
    "RTX 5070": "square19_2p",
    "A40": "hexagonal_2p",
    "RTX 5090": "hexagonal_2p",
}

LOG_DIR.mkdir(parents=True, exist_ok=True)

# Unified logging setup
try:
    from app.core.logging_config import setup_logging
    logger = setup_logging("cluster_automation", log_file=LOG_FILE)
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)


def get_vast_instances() -> List[Dict]:
    """Get running Vast instances."""
    try:
        result = subprocess.run(
            ['vastai', 'show', 'instances', '--raw'],
            capture_output=True, text=True, timeout=30
        )
        return [
            {
                'id': str(i['id']),
                'name': f"vast-{i['id']}",
                'host': i['ssh_host'],
                'port': i['ssh_port'],
                'gpu': i.get('gpu_name', 'unknown'),
                'num_gpus': i.get('num_gpus', 1),
            }
            for i in json.loads(result.stdout)
            if i.get('actual_status') == 'running'
        ]
    except Exception as e:
        logger.error(f"Failed to get Vast instances: {e}")
        return []


def run_ssh(host: str, port: int, cmd: str, user: str = "root", timeout: int = 30) -> Tuple[bool, str]:
    """Run SSH command."""
    try:
        result = subprocess.run([
            'ssh', '-o', 'StrictHostKeyChecking=accept-new',
            '-o', f'ConnectTimeout={min(timeout, 10)}',
            '-o', 'BatchMode=yes',
            '-p', str(port),
            f'{user}@{host}',
            cmd
        ], capture_output=True, text=True, timeout=timeout)
        return result.returncode == 0, result.stdout.strip()
    except Exception as e:
        return False, str(e)


def check_instance(inst: Dict) -> Dict:
    """Check health of a single instance."""
    host, port = inst['host'], inst['port']
    name = inst['name']

    status = {
        'name': name,
        'reachable': False,
        'gpu_util': 0,
        'workers': 0,
        'p2p_running': False,
        'needs_restart': False,
    }

    ok, output = run_ssh(host, port, '''
        nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | head -1;
        pgrep -f "hybrid_selfplay|generate_data|selfplay" | wc -l;
        pgrep -f "p2p_orchestrator" | wc -l
    ''')

    if not ok:
        return status

    status['reachable'] = True
    lines = output.split('\n')

    try:
        status['gpu_util'] = int(lines[0].replace('%', '').strip()) if lines[0] else 0
    except ValueError:
        status['gpu_util'] = 0

    try:
        status['workers'] = int(lines[1].strip()) if len(lines) > 1 else 0
    except ValueError:
        status['workers'] = 0

    try:
        status['p2p_running'] = int(lines[2].strip()) > 0 if len(lines) > 2 else False
    except ValueError:
        status['p2p_running'] = False

    # Determine if restart needed
    if status['gpu_util'] < MIN_GPU_UTIL and status['workers'] < 2:
        status['needs_restart'] = True

    return status


def start_selfplay_on_instance(inst: Dict) -> Tuple[str, bool, str]:
    """Start selfplay on an instance."""
    host, port = inst['host'], inst['port']
    name = inst['name']
    gpu = inst.get('gpu', 'unknown')
    num_gpus = inst.get('num_gpus', 1)

    config = GPU_CONFIG.get(gpu, 'square8_2p')
    board, players = config.rsplit('_', 1)
    players = players.replace('p', '')
    games = 500 * num_gpus

    cmd = f'''
cd ~/ringrift/ai-service 2>/dev/null || cd /root/ringrift/ai-service || exit 1
source venv/bin/activate 2>/dev/null || true

pkill -f "generate_data.*random" 2>/dev/null || true

if pgrep -f "hybrid_selfplay" > /dev/null; then
    echo "already running"
    exit 0
fi

export PYTHONPATH=$PWD
mkdir -p data/selfplay logs

nohup python3 scripts/run_hybrid_selfplay.py \\
    --num-games {games} \\
    --board-type {board} \\
    --num-players {players} \\
    --auto-ramdrive \\
    --sync-interval 300 \\
    --sync-target data/selfplay/auto_{config}_{name} \\
    > logs/auto_selfplay_{config}.log 2>&1 &

sleep 2
echo "started {config}"
'''

    ok, output = run_ssh(host, port, cmd, timeout=45)
    if ok and ("started" in output.lower() or "already" in output.lower()):
        return name, True, output.split('\n')[-1]
    return name, False, output[-60:]


def run_health_check() -> List[Dict]:
    """Check health of all instances."""
    logger.info("=" * 60)
    logger.info("CLUSTER HEALTH CHECK")
    logger.info("=" * 60)

    instances = get_vast_instances()
    if not instances:
        logger.warning("No Vast instances found")
        return []

    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(check_instance, inst): inst for inst in instances}
        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda x: x['name'])

    for r in results:
        status = "✓" if r['reachable'] and not r['needs_restart'] else "⚠" if r['reachable'] else "✗"
        logger.info(f"  {status} {r['name']:<18} GPU:{r['gpu_util']:>3}% workers:{r['workers']} p2p:{'✓' if r['p2p_running'] else '✗'}")

    return results


def restart_idle_workers(health_results: List[Dict], instances: List[Dict]) -> int:
    """Restart workers on idle instances."""
    idle = [r for r in health_results if r['needs_restart'] and r['reachable']]
    if not idle:
        logger.info("No idle instances need restart")
        return 0

    logger.info(f"Restarting {len(idle)} idle instances...")

    # Map names to full instance info
    inst_map = {i['name']: i for i in instances}

    restarted = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(start_selfplay_on_instance, inst_map[r['name']]): r
            for r in idle if r['name'] in inst_map
        }
        for future in as_completed(futures):
            name, ok, msg = future.result()
            if ok:
                restarted += 1
                logger.info(f"  ✓ {name}: {msg}")
            else:
                logger.warning(f"  ✗ {name}: {msg}")

    return restarted


def run_full_cycle():
    """Run full automation cycle."""
    logger.info("=" * 60)
    logger.info(f"CLUSTER AUTOMATION - {datetime.now().isoformat()}")
    logger.info("=" * 60)

    instances = get_vast_instances()

    # 1. Health check
    health_results = run_health_check()

    # 2. Restart idle workers
    restarted = restart_idle_workers(health_results, instances)

    # 3. Summary
    healthy = sum(1 for r in health_results if r['reachable'] and not r['needs_restart'])
    unreachable = sum(1 for r in health_results if not r['reachable'])

    logger.info("=" * 60)
    logger.info(f"SUMMARY: {healthy}/{len(health_results)} healthy, "
                f"{unreachable} unreachable, {restarted} restarted")
    logger.info("=" * 60)


def install_cron():
    """Install cron job for automation."""
    script_path = Path(__file__).resolve()
    python_path = sys.executable

    cron_line = f"*/15 * * * * cd {AI_SERVICE_ROOT} && {python_path} {script_path} --full >> {LOG_FILE} 2>&1"

    print("Add this line to your crontab (crontab -e):")
    print()
    print(cron_line)
    print()
    print("This will run the automation every 15 minutes.")


def main():
    parser = argparse.ArgumentParser(description="Cluster automation")
    parser.add_argument("--check", action="store_true", help="Quick health check only")
    parser.add_argument("--full", action="store_true", help="Run full automation cycle")
    parser.add_argument("--install-cron", action="store_true", help="Show cron installation")
    args = parser.parse_args()

    if args.install_cron:
        install_cron()
    elif args.full:
        run_full_cycle()
    elif args.check:
        run_health_check()
    else:
        # Default to full cycle
        run_full_cycle()


if __name__ == "__main__":
    main()
