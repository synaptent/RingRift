#!/usr/bin/env python3
"""Distribute models to all cluster nodes using aria2 parallel downloads.

This script broadcasts new models from training nodes to all Vast instances
using aria2's multi-connection downloads for maximum speed.

Usage:
    # Distribute latest model to all nodes
    python scripts/distribute_models_aria2.py --latest

    # Distribute specific model
    python scripts/distribute_models_aria2.py --model models/sq8_2p_nn_baseline.pth

    # Distribute all models newer than 1 hour
    python scripts/distribute_models_aria2.py --since 1h
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]

# aria2 settings
ARIA2_DATA_PORT = 8766
ARIA2_CONNECTIONS = 16
ARIA2_SPLIT = 16


def _load_hosts_from_config():
    """Load source nodes from config/distributed_hosts.yaml."""
    config_path = Path(__file__).parent.parent / "config" / "distributed_hosts.yaml"
    if not config_path.exists():
        print("[Script] Warning: No config found")
        return []
    try:
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}

        # Extract training nodes (prefer Tailscale IPs)
        sources = []
        for name, host_config in config.get('hosts', {}).items():
            if host_config.get('status') == 'terminated':
                continue

            # Only include training nodes
            role = host_config.get('role', '')
            if 'training' not in role:
                continue

            # Get IPs
            ts_ip = host_config.get('tailscale_ip')
            ssh_host = host_config.get('ssh_host')
            user = host_config.get('ssh_user', 'ubuntu')

            if ts_ip or ssh_host:
                sources.append((name, user, ssh_host or '', ts_ip or ''))

        return sources
    except Exception as e:
        print(f"[Script] Error loading config: {e}")
        return []


# Source nodes for models (Lambda training nodes)
SOURCE_NODES = _load_hosts_from_config() or []


def get_vast_instances() -> list[dict]:
    """Get running Vast instances."""
    try:
        result = subprocess.run(
            ['vastai', 'show', 'instances', '--raw'],
            capture_output=True, text=True, timeout=30
        )
        return [
            {
                'name': f"vast-{i['id']}",
                'host': i['ssh_host'],
                'port': i['ssh_port'],
            }
            for i in json.loads(result.stdout)
            if i.get('actual_status') == 'running'
        ]
    except Exception:
        return []


def find_latest_models(source_host: str, source_user: str, max_age_hours: float = 24) -> list[str]:
    """Find latest models on a source node."""
    try:
        result = subprocess.run([
            'ssh', '-o', 'StrictHostKeyChecking=accept-new', '-o', 'ConnectTimeout=10',
            '-o', 'BatchMode=yes', f'{source_user}@{source_host}',
            f'find ~/ringrift/ai-service/models -name "*.pth" -mmin -{int(max_age_hours * 60)} -type f 2>/dev/null'
        ], capture_output=True, text=True, timeout=30)
        return [p.strip() for p in result.stdout.strip().split('\n') if p.strip()]
    except Exception:
        return []


def distribute_model_to_instance(
    model_path: str,
    source_urls: list[str],
    target_host: str,
    target_port: int,
    target_name: str,
) -> tuple[str, bool, str]:
    """Distribute a model to a single Vast instance using aria2."""
    model_name = os.path.basename(model_path)

    # Build aria2 command with multiple sources
    urls = ' '.join(source_urls)

    cmd = f'''
cd ~/ringrift/ai-service 2>/dev/null || cd /root/ringrift/ai-service || exit 1
mkdir -p models

# Check if model already exists
if [ -f "models/{model_name}" ]; then
    echo "Model already exists"
    exit 0
fi

# Download using aria2 with multiple connections
aria2c \\
    --max-connection-per-server={ARIA2_CONNECTIONS} \\
    --split={ARIA2_SPLIT} \\
    --min-split-size=1M \\
    --continue=true \\
    --allow-overwrite=true \\
    --dir=models \\
    --out={model_name} \\
    {urls} \\
    2>&1 | tail -3

if [ -f "models/{model_name}" ]; then
    echo "Downloaded {model_name}"
    ls -lh "models/{model_name}" | awk '{{print $5}}'
else
    echo "Download failed"
    exit 1
fi
'''

    try:
        result = subprocess.run([
            'ssh', '-o', 'StrictHostKeyChecking=accept-new', '-o', 'ConnectTimeout=15',
            '-o', 'BatchMode=yes', '-p', str(target_port), f'root@{target_host}',
            cmd
        ], capture_output=True, text=True, timeout=120)

        output = result.stdout.strip()
        if "downloaded" in output.lower() or "already exists" in output.lower():
            return target_name, True, output.split('\n')[-1]
        return target_name, False, (output or result.stderr)[-80:]
    except Exception as e:
        return target_name, False, str(e)[:80]


def distribute_models(
    model_paths: list[str],
    max_parallel: int = 8,
) -> dict[str, int]:
    """Distribute models to all Vast instances."""
    instances = get_vast_instances()
    if not instances:
        print("No Vast instances found")
        return {}

    # Build source URLs for each model
    source_urls = {}
    for model_path in model_paths:
        model_name = os.path.basename(model_path)
        urls = []
        for name, user, host, ts_ip in SOURCE_NODES:
            # Prefer Tailscale IP if available
            url_host = ts_ip or host
            urls.append(f"http://{url_host}:{ARIA2_DATA_PORT}/models/{model_name}")
        source_urls[model_path] = urls

    print(f"Distributing {len(model_paths)} models to {len(instances)} instances")

    results = {"success": 0, "failed": 0, "skipped": 0}

    for model_path in model_paths:
        model_name = os.path.basename(model_path)
        urls = source_urls[model_path]

        print(f"\n--- Distributing {model_name} ---")

        with ThreadPoolExecutor(max_workers=max_parallel) as executor:
            futures = {
                executor.submit(
                    distribute_model_to_instance,
                    model_path, urls,
                    inst['host'], inst['port'], inst['name']
                ): inst
                for inst in instances
            }

            for future in as_completed(futures):
                name, ok, msg = future.result()
                if "already exists" in msg.lower():
                    results["skipped"] += 1
                    print(f"  - {name}: skipped (exists)")
                elif ok:
                    results["success"] += 1
                    print(f"  ✓ {name}: {msg}")
                else:
                    results["failed"] += 1
                    print(f"  ✗ {name}: {msg}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Distribute models via aria2")
    parser.add_argument("--latest", action="store_true", help="Distribute latest models from all sources")
    parser.add_argument("--model", type=str, help="Specific model path to distribute")
    parser.add_argument("--since", type=str, default="24h", help="Models newer than this (e.g., 1h, 24h)")
    args = parser.parse_args()

    if args.model:
        model_paths = [args.model]
    else:
        # Parse time delta
        since = args.since.lower()
        if since.endswith('h'):
            hours = float(since[:-1])
        elif since.endswith('d'):
            hours = float(since[:-1]) * 24
        else:
            hours = 24

        # Find models from all sources
        model_paths = []
        for name, user, host, _ in SOURCE_NODES:
            paths = find_latest_models(host, user, hours)
            model_paths.extend(paths)
            if paths:
                print(f"Found {len(paths)} models on {name}")

        # Deduplicate by filename
        seen = set()
        unique_paths = []
        for p in model_paths:
            name = os.path.basename(p)
            if name not in seen:
                seen.add(name)
                unique_paths.append(p)
        model_paths = unique_paths

    if not model_paths:
        print("No models to distribute")
        return

    print(f"\n{'=' * 70}")
    print(f"DISTRIBUTING {len(model_paths)} MODELS VIA ARIA2")
    print(f"{'=' * 70}")

    results = distribute_models(model_paths)

    print(f"\n{'=' * 70}")
    print(f"RESULTS: {results.get('success', 0)} success, "
          f"{results.get('failed', 0)} failed, "
          f"{results.get('skipped', 0)} skipped")


if __name__ == "__main__":
    main()
