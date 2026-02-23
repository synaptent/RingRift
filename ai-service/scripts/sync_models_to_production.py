#!/usr/bin/env python3
"""Sync fresh canonical models from coordinator to production web server.

Usage (run from coordinator, e.g., local-mac):
    python scripts/sync_models_to_production.py

Compares model mtimes between local models/ directory and the production
server's /ai/model_freshness endpoint, then transfers stale models via
sftp (more reliable than rsync for this EC2 instance) and triggers
a hot-reload via /ai/reload_models.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Production server configuration
PROD_HOST = os.environ.get("RINGRIFT_PROD_HOST", "54.198.219.106")
PROD_TAILSCALE_HOST = os.environ.get("RINGRIFT_PROD_TS_HOST", "100.115.97.24")
PROD_SSH_KEY = os.environ.get("RINGRIFT_PROD_SSH_KEY", os.path.expanduser("~/.ssh/ringrift-staging-key.pem"))
PROD_USER = os.environ.get("RINGRIFT_PROD_USER", "ubuntu")
PROD_AI_URL = os.environ.get("RINGRIFT_PROD_AI_URL", "http://localhost:8765")
PROD_MODELS_DIR = os.environ.get("RINGRIFT_PROD_MODELS_DIR", "/home/ubuntu/ringrift/ai-service/models")
STALE_THRESHOLD_HOURS = float(os.environ.get("RINGRIFT_STALE_THRESHOLD", "72"))
SYNC_DELAY_SECONDS = float(os.environ.get("RINGRIFT_SYNC_DELAY", "3"))
MAX_RETRIES = int(os.environ.get("RINGRIFT_SYNC_RETRIES", "2"))

LOCAL_MODELS_DIR = Path(__file__).parent.parent / "models"


def _get_transfer_host() -> str:
    """Return Tailscale host if reachable, else fall back to public IP."""
    try:
        result = subprocess.run(
            ["ping", "-c", "1", "-W", "2", PROD_TAILSCALE_HOST],
            capture_output=True, timeout=5,
        )
        if result.returncode == 0:
            return PROD_TAILSCALE_HOST
    except (subprocess.TimeoutExpired, OSError):
        pass
    return PROD_HOST


def _ssh_cmd(host: str) -> list[str]:
    """Base SSH command fragments."""
    return [
        "ssh", "-i", PROD_SSH_KEY,
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=10",
        f"{PROD_USER}@{host}",
    ]


def get_prod_freshness() -> dict | None:
    """Query production /ai/model_freshness endpoint via SSH tunnel."""
    try:
        cmd = _ssh_cmd(PROD_HOST) + [
            f"curl -s {PROD_AI_URL}/ai/model_freshness?stale_threshold_hours={STALE_THRESHOLD_HOURS}",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and result.stdout.strip():
            return json.loads(result.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError) as e:
        print(f"  Error querying production: {e}")
    return None


def sync_model(config_key: str, host: str) -> bool:
    """Transfer a single canonical model to production via sftp."""
    local_path = LOCAL_MODELS_DIR / f"canonical_{config_key}.pth"
    if not local_path.exists():
        print(f"  SKIP {config_key}: local model not found at {local_path}")
        return False

    remote_path = f"{PROD_MODELS_DIR}/canonical_{config_key}.pth"
    size_mb = local_path.stat().st_size / 1024 / 1024
    print(f"  Syncing {config_key} ({size_mb:.1f} MB) via sftp to {host}...")

    # Use sftp batch mode (more reliable than rsync for this server)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sftp", delete=False) as f:
        f.write(f"put {local_path} {remote_path}\n")
        batch_file = f.name

    try:
        cmd = [
            "sftp", "-b", batch_file,
            "-i", PROD_SSH_KEY,
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=30",
            "-o", "ServerAliveInterval=15",
            f"{PROD_USER}@{host}",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            print(f"  OK: {config_key} synced")
            return True
        else:
            print(f"  FAIL: {config_key} - {result.stderr[:200]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  FAIL: {config_key} - transfer timed out")
        return False
    finally:
        os.unlink(batch_file)


def trigger_reload(host: str) -> bool:
    """Restart AI service on production to pick up new models."""
    try:
        cmd = _ssh_cmd(host) + ["pm2 restart ringrift-ai --update-env"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print(f"  Restarted ringrift-ai service")
            return True
    except (subprocess.TimeoutExpired, OSError) as e:
        print(f"  Reload error: {e}")
    return False


def main():
    transfer_host = _get_transfer_host()
    via = "Tailscale" if transfer_host == PROD_TAILSCALE_HOST else "public IP"

    print("=== RingRift Model Sync to Production ===")
    print(f"Production: {PROD_USER}@{PROD_HOST} (transfer via {via}: {transfer_host})")
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

    # 2. Sync stale models via sftp
    print(f"\n2. Syncing {len(stale_configs)} stale models...")
    synced = 0
    failed = []
    for i, config_key in enumerate(stale_configs):
        if i > 0:
            time.sleep(SYNC_DELAY_SECONDS)
        if sync_model(config_key, transfer_host):
            synced += 1
        else:
            failed.append(config_key)

    # Retry failed models
    for retry in range(MAX_RETRIES):
        if not failed:
            break
        retry_delay = SYNC_DELAY_SECONDS * (retry + 2)
        print(f"\n   Retry {retry + 1}/{MAX_RETRIES}: {len(failed)} models, {retry_delay}s delay...")
        time.sleep(retry_delay)
        still_failed = []
        for i, config_key in enumerate(failed):
            if i > 0:
                time.sleep(retry_delay)
            if sync_model(config_key, transfer_host):
                synced += 1
            else:
                still_failed.append(config_key)
        failed = still_failed

    # 3. Trigger reload
    if synced > 0:
        print(f"\n3. Reloading AI service ({synced} models synced)...")
        trigger_reload(PROD_HOST)
    else:
        print("\n3. No models synced, skipping reload")

    if failed:
        print(f"\n  WARNING: {len(failed)} models failed to sync: {', '.join(failed)}")

    print(f"\nDone: {synced}/{len(stale_configs)} models synced to production")


if __name__ == "__main__":
    main()
