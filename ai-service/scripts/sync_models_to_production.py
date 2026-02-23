#!/usr/bin/env python3
"""Sync fresh canonical models from coordinator to production web server.

Usage (run from coordinator, e.g., local-mac):
    python scripts/sync_models_to_production.py

Compares model mtimes between local models/ directory and the production
server's /ai/model_freshness endpoint, then rsyncs stale models and triggers
a hot-reload via /ai/reload_models.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Production server configuration
PROD_HOST = os.environ.get("RINGRIFT_PROD_HOST", "54.198.219.106")
PROD_SSH_KEY = os.environ.get("RINGRIFT_PROD_SSH_KEY", os.path.expanduser("~/.ssh/ringrift-staging-key.pem"))
PROD_USER = os.environ.get("RINGRIFT_PROD_USER", "ubuntu")
PROD_AI_URL = os.environ.get("RINGRIFT_PROD_AI_URL", "http://localhost:8001")
PROD_MODELS_DIR = os.environ.get("RINGRIFT_PROD_MODELS_DIR", "/home/ubuntu/ringrift/ai-service/models")
STALE_THRESHOLD_HOURS = float(os.environ.get("RINGRIFT_STALE_THRESHOLD", "72"))

LOCAL_MODELS_DIR = Path(__file__).parent.parent / "models"


def get_prod_freshness() -> dict | None:
    """Query production /ai/model_freshness endpoint via SSH tunnel."""
    try:
        cmd = [
            "ssh", "-i", PROD_SSH_KEY,
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=10",
            f"{PROD_USER}@{PROD_HOST}",
            f"curl -s {PROD_AI_URL}/ai/model_freshness?stale_threshold_hours={STALE_THRESHOLD_HOURS}",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and result.stdout.strip():
            return json.loads(result.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError) as e:
        print(f"  Error querying production: {e}")
    return None


def sync_model(config_key: str) -> bool:
    """Rsync a single canonical model to production."""
    local_path = LOCAL_MODELS_DIR / f"canonical_{config_key}.pth"
    if not local_path.exists():
        print(f"  SKIP {config_key}: local model not found at {local_path}")
        return False

    remote_path = f"{PROD_USER}@{PROD_HOST}:{PROD_MODELS_DIR}/canonical_{config_key}.pth"
    cmd = [
        "rsync", "-avz", "--progress",
        "-e", f"ssh -i {PROD_SSH_KEY} -o StrictHostKeyChecking=no",
        str(local_path),
        remote_path,
    ]
    print(f"  Syncing {config_key} ({local_path.stat().st_size / 1024 / 1024:.1f} MB)...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode == 0:
        print(f"  OK: {config_key} synced")
        return True
    else:
        print(f"  FAIL: {config_key} - {result.stderr[:200]}")
        return False


def trigger_reload() -> bool:
    """Trigger model reload on production via /ai/reload_models."""
    admin_key = os.environ.get("RINGRIFT_PROD_ADMIN_KEY", "")
    header = f'-H "X-Admin-Key: {admin_key}"' if admin_key else ""
    try:
        cmd = [
            "ssh", "-i", PROD_SSH_KEY,
            "-o", "StrictHostKeyChecking=no",
            f"{PROD_USER}@{PROD_HOST}",
            f'curl -s -X POST {header} {PROD_AI_URL}/ai/reload_models',
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print(f"  Reload response: {result.stdout.strip()}")
            return True
    except (subprocess.TimeoutExpired, OSError) as e:
        print(f"  Reload error: {e}")
    return False


def main():
    print(f"=== RingRift Model Sync to Production ===")
    print(f"Production: {PROD_USER}@{PROD_HOST}")
    print(f"Stale threshold: {STALE_THRESHOLD_HOURS}h")
    print()

    # 1. Check production freshness
    print("1. Querying production model freshness...")
    freshness = get_prod_freshness()

    stale_configs = []
    if freshness and "models" in freshness:
        for config_key, info in freshness["models"].items():
            if isinstance(info, dict) and (info.get("stale") or info.get("error")):
                stale_configs.append(config_key)
                age = info.get("age_hours", "N/A")
                print(f"   STALE: {config_key} (age: {age}h)")
        if not stale_configs:
            print("   All models are fresh!")
            return
    else:
        print("   Could not query production - syncing all models")
        stale_configs = [
            f"{board}_{n}p"
            for board in ["hex8", "square8", "square19", "hexagonal"]
            for n in [2, 3, 4]
        ]

    # 2. Sync stale models
    print(f"\n2. Syncing {len(stale_configs)} stale models...")
    synced = 0
    for config_key in stale_configs:
        if sync_model(config_key):
            synced += 1

    # 3. Trigger reload
    if synced > 0:
        print(f"\n3. Triggering model reload ({synced} models synced)...")
        trigger_reload()
    else:
        print("\n3. No models synced, skipping reload")

    print(f"\nDone: {synced}/{len(stale_configs)} models synced to production")


if __name__ == "__main__":
    main()
