#!/usr/bin/env python3
"""Model Sync via aria2 - Fast parallel model distribution across cluster.

DEPRECATED: Use sync_models.py instead, which provides:
- All aria2 functionality via coordination helpers
- Hash-based deduplication
- Bandwidth management and resource guards
- Lock management to prevent conflicts
- Both collect and distribute modes
- Cull manifest tracking for model lifecycle

Migration:
    # Sync all models (equivalent to --sync-to-all):
    python scripts/sync_models.py --sync --use-sync-coordinator

    # Check status:
    python scripts/sync_models.py --dry-run

    # Continuous daemon:
    python scripts/sync_models.py --daemon

This script will be removed in a future release.

---
Original description:
Uses aria2 for high-speed parallel downloads from multiple sources:
- 16 connections per server
- Multi-source downloads (metalink-style)
- Resume support for interrupted transfers
- Automatic source selection based on latency

Usage:
    python scripts/model_sync_aria2.py --sync-to-all       # Sync latest models to all nodes
    python scripts/model_sync_aria2.py --sync-from LEADER  # Pull models from leader
    python scripts/model_sync_aria2.py --status            # Show sync status
    python scripts/model_sync_aria2.py --serve             # Start local model server

Designed for distributing trained models across P2P cluster.
"""
import warnings

warnings.warn(
    "model_sync_aria2.py is deprecated. Use sync_models.py instead.",
    DeprecationWarning,
    stacklevel=2
)

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_SERVICE_ROOT))

LOG_DIR = AI_SERVICE_ROOT / "logs"
LOG_FILE = LOG_DIR / "model_sync.log"
MODELS_DIR = AI_SERVICE_ROOT / "data" / "models"

LOG_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

from scripts.lib.logging_config import setup_script_logging

logger = setup_script_logging("model_sync_aria2")

# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SyncConfig:
    """Model sync configuration."""
    aria2_connections: int = 16          # Connections per server
    aria2_split: int = 16                # Split file into N parts
    aria2_timeout: int = 300             # Download timeout (seconds)
    model_server_port: int = 8766        # HTTP port for serving models
    max_concurrent_syncs: int = 5        # Max parallel node syncs


# Model sources and sync targets are loaded from config/distributed_hosts.yaml
# See config/distributed_hosts.yaml.example for format
def _load_hosts_from_config():
    """Load hosts from config file."""
    config_path = Path(__file__).parent.parent / "config" / "distributed_hosts.yaml"
    example_path = config_path.with_suffix(".yaml.example")

    if not config_path.exists():
        print(f"[ModelSync] Warning: {config_path} not found")
        print(f"[ModelSync] Copy {example_path} to {config_path} and configure your hosts")
        return [], []

    try:
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}

        sources = []
        targets = []
        hosts = config.get("hosts", {})

        for name, info in hosts.items():
            host = info.get("tailscale_ip") or info.get("ssh_host")
            if not host:
                continue

            # Add as source if it can serve models
            if info.get("status") == "ready":
                sources.append({
                    "name": name,
                    "host": host,
                    "port": info.get("model_server_port", 8766),
                })

            # Add as sync target
            targets.append({
                "name": name,
                "host": host,
                "ssh_user": info.get("ssh_user", "ubuntu"),
            })

        return sources, targets
    except Exception as e:
        print(f"[ModelSync] Error loading config: {e}")
        return [], []

MODEL_SOURCES, SYNC_TARGETS = _load_hosts_from_config()


# =============================================================================
# Model Discovery
# =============================================================================

def get_local_models() -> list[dict]:
    """Get list of local model files with metadata."""
    models = []
    for model_file in MODELS_DIR.glob("*.pth"):
        stat = model_file.stat()
        models.append({
            "name": model_file.name,
            "path": str(model_file),
            "size": stat.st_size,
            "mtime": stat.st_mtime,
            "md5": _compute_md5(model_file),
        })
    return sorted(models, key=lambda m: -m["mtime"])


def _compute_md5(file_path: Path, chunk_size: int = 8192) -> str:
    """Compute MD5 hash of file."""
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            md5.update(chunk)
    return md5.hexdigest()


def get_latest_model(board_type: str | None = None) -> dict | None:
    """Get the latest model, optionally filtered by board type."""
    models = get_local_models()
    if board_type:
        models = [m for m in models if board_type in m["name"]]
    return models[0] if models else None


def probe_source(source: dict) -> tuple[str, bool, float]:
    """Probe a model source for availability and latency."""
    import time
    import urllib.request

    name = source["name"]
    url = f"http://{source['host']}:{source['port']}/"

    try:
        start = time.time()
        with urllib.request.urlopen(url, timeout=5) as resp:
            resp.read(1024)
        latency = time.time() - start
        return name, True, latency
    except Exception:
        return name, False, float("inf")


def get_best_sources(count: int = 3) -> list[dict]:
    """Get the best available model sources sorted by latency."""
    results = []
    with ThreadPoolExecutor(max_workers=len(MODEL_SOURCES)) as executor:
        futures = {executor.submit(probe_source, s): s for s in MODEL_SOURCES}
        for future in as_completed(futures):
            _name, available, latency = future.result()
            if available:
                source = futures[future]
                results.append({"source": source, "latency": latency})

    results.sort(key=lambda r: r["latency"])
    return [r["source"] for r in results[:count]]


# =============================================================================
# aria2 Integration
# =============================================================================

def download_with_aria2(
    urls: list[str],
    output_path: Path,
    config: SyncConfig,
) -> bool:
    """Download file using aria2 with multiple sources."""
    if not urls:
        logger.error("No URLs provided")
        return False

    # Build aria2 command
    cmd = [
        "aria2c",
        "--max-connection-per-server", str(config.aria2_connections),
        "--split", str(config.aria2_split),
        "--min-split-size", "1M",
        "--timeout", str(config.aria2_timeout),
        "--continue", "true",
        "--auto-file-renaming", "false",
        "--allow-overwrite", "true",
        "--dir", str(output_path.parent),
        "--out", output_path.name,
        "--console-log-level", "warn",
    ]

    # Add all URLs
    cmd.extend(urls)

    try:
        logger.info(f"Downloading to {output_path} from {len(urls)} sources...")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=config.aria2_timeout + 60,
        )
        if result.returncode == 0:
            logger.info(f"Download complete: {output_path}")
            return True
        else:
            logger.error(f"aria2 failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        logger.error("Download timed out")
        return False
    except FileNotFoundError:
        logger.error("aria2c not found - install with: apt install aria2")
        return False
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False


def sync_model_to_node(
    model: dict,
    target: dict,
    sources: list[dict],
    config: SyncConfig,
) -> tuple[str, bool, str]:
    """Sync a model to a target node."""
    name = target["name"]
    host = target["host"]
    user = target.get("ssh_user", "ubuntu")

    # Build source URLs
    urls = [f"http://{s['host']}:{s['port']}/models/{model['name']}" for s in sources]
    urls_str = " ".join(urls)

    # Remote aria2 command
    remote_cmd = f"""
cd ~/ringrift/ai-service/data/models &&
aria2c --max-connection-per-server={config.aria2_connections} \
    --split={config.aria2_split} \
    --min-split-size=1M \
    --continue=true \
    --auto-file-renaming=false \
    --allow-overwrite=true \
    --console-log-level=warn \
    {urls_str} \
    -o {model['name']} 2>&1 | tail -3
"""

    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", f"{user}@{host}", remote_cmd],
            capture_output=True,
            text=True,
            timeout=config.aria2_timeout + 30,
        )
        if result.returncode == 0:
            return name, True, "synced"
        else:
            return name, False, result.stderr[:100]
    except Exception as e:
        return name, False, str(e)[:100]


# =============================================================================
# Model Server
# =============================================================================

def start_model_server(port: int = 8766):
    """Start HTTP server to serve models."""
    import http.server
    import socketserver

    os.chdir(MODELS_DIR.parent)  # Serve from data/ directory

    handler = http.server.SimpleHTTPRequestHandler

    with socketserver.TCPServer(("", port), handler) as httpd:
        logger.info(f"Serving models on port {port}...")
        logger.info(f"Models directory: {MODELS_DIR}")
        httpd.serve_forever()


# =============================================================================
# Commands
# =============================================================================

def cmd_status():
    """Show model sync status."""
    logger.info("=" * 70)
    logger.info("MODEL SYNC STATUS")
    logger.info("=" * 70)

    # Local models
    models = get_local_models()
    print(f"\nLocal models ({len(models)}):")
    print(f"{'Name':<50} {'Size':<12} {'Modified'}")
    print("-" * 80)
    for m in models[:10]:
        size_mb = m["size"] / (1024 * 1024)
        mtime = datetime.fromtimestamp(m["mtime"]).strftime("%Y-%m-%d %H:%M")
        print(f"{m['name']:<50} {size_mb:>8.1f} MB  {mtime}")

    # Probe sources
    print("\nModel sources:")
    sources = get_best_sources(len(MODEL_SOURCES))
    for s in MODEL_SOURCES:
        available = any(src["name"] == s["name"] for src in sources)
        status = "OK" if available else "OFFLINE"
        print(f"  {s['name']:<20} {s['host']:<18} {status}")


def cmd_sync_to_all(config: SyncConfig, model_name: str | None = None):
    """Sync models to all target nodes."""
    logger.info("=" * 70)
    logger.info("SYNCING MODELS TO ALL NODES")
    logger.info("=" * 70)

    # Get model to sync
    if model_name:
        model_path = MODELS_DIR / model_name
        if not model_path.exists():
            logger.error(f"Model not found: {model_name}")
            return
        model = {
            "name": model_name,
            "path": str(model_path),
            "size": model_path.stat().st_size,
        }
    else:
        model = get_latest_model()
        if not model:
            logger.error("No models found to sync")
            return

    logger.info(f"Syncing model: {model['name']} ({model['size'] / 1024 / 1024:.1f} MB)")

    # Get best sources
    sources = get_best_sources(3)
    if not sources:
        logger.error("No model sources available")
        return

    logger.info(f"Using sources: {[s['name'] for s in sources]}")

    # Sync to all targets
    results = []
    with ThreadPoolExecutor(max_workers=config.max_concurrent_syncs) as executor:
        futures = {
            executor.submit(sync_model_to_node, model, target, sources, config): target
            for target in SYNC_TARGETS
        }
        for future in as_completed(futures):
            name, success, msg = future.result()
            status = "OK" if success else "FAIL"
            logger.info(f"  {name:<20} {status} - {msg}")
            results.append((name, success))

    # Summary
    succeeded = sum(1 for _, s in results if s)
    logger.info(f"\nSynced to {succeeded}/{len(results)} nodes")


def cmd_sync_from(source_name: str, config: SyncConfig):
    """Pull latest models from a specific source."""
    logger.info(f"Syncing from {source_name}...")

    # Find source
    source = next((s for s in MODEL_SOURCES if s["name"] == source_name), None)
    if not source:
        logger.error(f"Unknown source: {source_name}")
        return

    # List remote models
    import urllib.request

    try:
        with urllib.request.urlopen(f"http://{source['host']}:{source['port']}/models/", timeout=10) as resp:
            html = resp.read().decode()
            # Parse simple directory listing
            import re
            models = re.findall(r'href="([^"]+\.pth)"', html)
    except Exception as e:
        logger.error(f"Failed to list remote models: {e}")
        return

    if not models:
        logger.warning("No models found on remote")
        return

    logger.info(f"Found {len(models)} models on {source_name}")

    # Download latest
    latest = models[0]  # Assuming sorted by name/date
    url = f"http://{source['host']}:{source['port']}/models/{latest}"
    output_path = MODELS_DIR / latest

    success = download_with_aria2([url], output_path, config)
    if success:
        logger.info(f"Downloaded: {latest}")
    else:
        logger.error(f"Failed to download: {latest}")


def cmd_serve(port: int):
    """Start model server."""
    start_model_server(port)


def main():
    parser = argparse.ArgumentParser(description="Model Sync via aria2")
    parser.add_argument("--status", action="store_true", help="Show sync status")
    parser.add_argument("--sync-to-all", action="store_true", help="Sync to all nodes")
    parser.add_argument("--sync-from", type=str, metavar="SOURCE", help="Pull from source")
    parser.add_argument("--serve", action="store_true", help="Start model server")
    parser.add_argument("--model", type=str, help="Specific model to sync")
    parser.add_argument("--port", type=int, default=8766, help="Server port")

    args = parser.parse_args()
    config = SyncConfig()

    if args.status:
        cmd_status()
    elif args.sync_to_all:
        cmd_sync_to_all(config, args.model)
    elif args.sync_from:
        cmd_sync_from(args.sync_from, config)
    elif args.serve:
        cmd_serve(args.port)
    else:
        cmd_status()


if __name__ == "__main__":
    main()
