#!/usr/bin/env python3
"""Unified NN and NNUE model sync across all distributed hosts.

This is the canonical model sync script, consolidating features from:
- sync_models_to_cluster.py (cluster-aware, parallel scanning)
- sync_staging_ai_artifacts.py (tarball sync, daemon integration)
- sync_cluster_data.sh (resilient connections, Tailscale fallback)

Features:
- Uses app/distributed/hosts module for host configuration
- Parallel model discovery across cluster
- Hash-based deduplication to avoid re-syncing identical models
- Resilient SSH with Tailscale fallback
- Daemon integration hooks for continuous_improvement_daemon
- Supports both collect (from cluster to local) and distribute (local to cluster)

Usage:
    # Discover models on all hosts
    python scripts/sync_models.py --discover

    # Collect all models from cluster to local
    python scripts/sync_models.py --collect

    # Distribute local models to all hosts
    python scripts/sync_models.py --distribute

    # Sync (collect + distribute) with deduplication
    python scripts/sync_models.py --sync

    # Sync using SyncCoordinator (aria2/SSH/P2P + NFS-aware)
    python scripts/sync_models.py --sync --use-sync-coordinator

    # Include offline/disabled hosts
    python scripts/sync_models.py --sync --include-nonready

    # Dry run
    python scripts/sync_models.py --sync --dry-run

    # Daemon mode - sync every N minutes
    python scripts/sync_models.py --daemon --interval 30
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add ai-service to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Use canonical host configuration module
try:
    from app.distributed.hosts import (
        HostConfig,
        SSHExecutor,
        get_ssh_executor,
        load_ready_hosts,
        load_remote_hosts,
    )
    HOSTS_MODULE_AVAILABLE = True
except ImportError:
    HOSTS_MODULE_AVAILABLE = False
    HostConfig = None

# Import coordination helpers for sync lock and bandwidth management
from app.coordination.helpers import (
    acquire_sync_lock_safe,
    get_transfer_priorities,
    has_bandwidth_manager,
    has_sync_lock,
    release_bandwidth_safe,
    release_sync_lock_safe,
    request_bandwidth_safe,
)

HAS_SYNC_LOCK = has_sync_lock()
HAS_BANDWIDTH_MANAGER = has_bandwidth_manager()
TransferPriority = get_transfer_priorities()

# Unified resource checking utilities (80% max utilization)
try:
    from app.utils.resource_guard import (
        LIMITS as RESOURCE_LIMITS,
        check_disk_space as unified_check_disk,
        get_disk_usage as unified_get_disk_usage,
    )
    HAS_RESOURCE_GUARD = True
except ImportError:
    HAS_RESOURCE_GUARD = False
    unified_get_disk_usage = None
    unified_check_disk = None
    RESOURCE_LIMITS = None

# Bandwidth limit helper - Dec 2025: Enforce rsync --bwlimit from centralized config
try:
    from app.config.cluster_config import get_node_bandwidth_kbs
    HAS_BANDWIDTH_CONFIG = True
except ImportError:
    HAS_BANDWIDTH_CONFIG = False
    def get_node_bandwidth_kbs(node_name: str, config_path=None) -> int:
        return 100 * 1024  # Default 100 MB/s in KB/s

# Wrapper functions for backwards compatibility
def acquire_sync_lock(host: str, timeout: float = 30.0) -> bool:
    return acquire_sync_lock_safe(host, timeout)

def release_sync_lock(host: str) -> None:
    release_sync_lock_safe(host)

def request_bandwidth(host: str, mbps: float = 100.0, priority=None):
    return request_bandwidth_safe(host, mbps, priority)

def release_bandwidth(host: str) -> None:
    release_bandwidth_safe(host)

import contextlib

from scripts.lib.logging_config import setup_script_logging

# Import robust transfer utilities for connection reset handling
try:
    from scripts.lib.transfer import (
        TransferConfig,
        robust_push,
        chunked_push_progressive,
    )
    HAS_ROBUST_TRANSFER = True
except ImportError:
    HAS_ROBUST_TRANSFER = False
    robust_push = None
    chunked_push_progressive = None
    TransferConfig = None

logger = setup_script_logging("sync_models")

# Storage provider/NFS-aware helpers (optional)
try:
    from app.distributed.storage_provider import should_sync_to_node
    from app.metrics.orchestrator import record_nfs_skip
    HAS_STORAGE_PROVIDER = True
except (ImportError, ModuleNotFoundError):
    HAS_STORAGE_PROVIDER = False

    def should_sync_to_node(_target: str) -> bool:  # type: ignore[return-type]
        return True

    def record_nfs_skip(_category: str) -> None:
        return None


# ============================================
# Disk Usage Limits - from app.config.thresholds (canonical source)
# ============================================

try:
    from app.config.thresholds import DISK_SYNC_TARGET_PERCENT
    MAX_DISK_USAGE_PERCENT = float(os.environ.get("RINGRIFT_MAX_DISK_PERCENT", str(DISK_SYNC_TARGET_PERCENT)))
except ImportError:
    MAX_DISK_USAGE_PERCENT = float(os.environ.get("RINGRIFT_MAX_DISK_PERCENT", "70"))


# ============================================
# Retry Queue for Sync Failures (Dec 2025)
# ============================================

# Retry configuration
SYNC_MAX_RETRIES = int(os.environ.get("RINGRIFT_SYNC_MAX_RETRIES", "3"))
SYNC_BASE_DELAY_SECONDS = float(os.environ.get("RINGRIFT_SYNC_BASE_DELAY", "10.0"))

# Retry queue: list of (source_model, host_name, attempt_count, next_retry_time)
_RETRY_QUEUE: list[tuple[str, str, int, float]] = []


def _add_to_retry_queue(model_name: str, host_name: str, attempt: int) -> None:
    """Add a failed sync to the retry queue with exponential backoff.

    Dec 2025: Added to handle transient network failures gracefully.
    """
    if attempt >= SYNC_MAX_RETRIES:
        logger.warning(f"Max retries ({SYNC_MAX_RETRIES}) exceeded for {model_name} -> {host_name}")
        return

    # Exponential backoff: 10s, 20s, 40s
    delay = SYNC_BASE_DELAY_SECONDS * (2 ** attempt)
    next_retry = time.time() + delay
    _RETRY_QUEUE.append((model_name, host_name, attempt + 1, next_retry))
    logger.info(f"Queued retry {attempt + 1}/{SYNC_MAX_RETRIES} for {model_name} -> {host_name} in {delay}s")


def _process_retry_queue(host_loader, dry_run: bool = False) -> int:
    """Process pending retries from the retry queue.

    Returns number of successful retries.
    """
    if not _RETRY_QUEUE:
        return 0

    now = time.time()
    successful = 0
    remaining = []

    for model_name, host_name, attempt, next_retry in _RETRY_QUEUE:
        if now < next_retry:
            # Not ready for retry yet
            remaining.append((model_name, host_name, attempt, next_retry))
            continue

        # Find host config
        hosts = host_loader(include_nonready=True)
        host = next((h for h in hosts if h.name == host_name), None)
        if not host:
            logger.warning(f"Host {host_name} not found for retry")
            continue

        logger.info(f"Retry {attempt}/{SYNC_MAX_RETRIES}: {model_name} -> {host_name}")
        success, msg = sync_model_to_host(host, model_name, dry_run=dry_run, skip_retry_queue=True)

        if success:
            successful += 1
            logger.info(f"Retry succeeded: {model_name} -> {host_name}")
        else:
            # Re-queue for another retry if not at max
            _add_to_retry_queue(model_name, host_name, attempt)

    # Update queue with remaining items
    _RETRY_QUEUE.clear()
    _RETRY_QUEUE.extend(remaining)

    return successful


def get_retry_queue_size() -> int:
    """Get number of items pending retry."""
    return len(_RETRY_QUEUE)


def check_disk_usage(path: Path | None = None) -> tuple[bool, float]:
    """Check if disk has capacity for syncing.

    Uses unified resource_guard utilities when available for consistent
    80% max utilization enforcement (70% for disk).

    Args:
        path: Path to check disk usage for. Defaults to ROOT.

    Returns:
        Tuple of (has_capacity, current_usage_percent)
    """
    check_path = str(path) if path else str(ROOT)

    # Use unified utilities when available
    if HAS_RESOURCE_GUARD and unified_get_disk_usage is not None:
        try:
            percent, _, _ = unified_get_disk_usage(check_path)
            has_capacity = percent < MAX_DISK_USAGE_PERCENT
            if not has_capacity:
                logger.warning(f"Disk usage {percent:.1f}% exceeds limit {MAX_DISK_USAGE_PERCENT}%")
            return has_capacity, percent
        except (OSError, ValueError, TypeError):
            pass  # Fall through to original implementation

    # Fallback to original implementation
    try:
        usage = shutil.disk_usage(check_path)
        percent = 100.0 * usage.used / usage.total
        has_capacity = percent < MAX_DISK_USAGE_PERCENT
        if not has_capacity:
            logger.warning(f"Disk usage {percent:.1f}% exceeds limit {MAX_DISK_USAGE_PERCENT}%")
        return has_capacity, percent
    except Exception as e:
        logger.error(f"Failed to check disk usage: {e}")
        return True, 0.0  # Allow sync on error (fail open)


# ============================================
# Configuration
# ============================================

LOCAL_MODELS_DIR = ROOT / "models"
LOCAL_NNUE_DIR = ROOT / "models" / "nnue"
SYNC_STATE_PATH = ROOT / "data" / "model_sync_state.json"
MODEL_HASHES_PATH = ROOT / "data" / "model_hashes.json"

# Connection settings
SSH_TIMEOUT = 30
RSYNC_TIMEOUT = 300

# Board/player configurations
ALL_CONFIGS = [
    ("square8", 2), ("square8", 3), ("square8", 4),
    ("square19", 2), ("square19", 3), ("square19", 4),
    ("hexagonal", 2), ("hexagonal", 3), ("hexagonal", 4),
]


# ============================================
# Data Classes
# ============================================

@dataclass
class ModelInfo:
    """Information about a model file."""
    name: str
    path: str
    host: str
    size_bytes: int
    mtime: float
    model_type: str  # "nn" or "nnue"
    md5_hash: str | None = None
    board_type: str | None = None
    num_players: int | None = None
    version: str | None = None


@dataclass
class HostModelInventory:
    """Model inventory for a single host."""
    host_name: str
    nn_models: set[str] = field(default_factory=set)
    nnue_models: set[str] = field(default_factory=set)
    model_hashes: dict[str, str] = field(default_factory=dict)
    work_dir: str = ""
    reachable: bool = False
    error: str | None = None
    last_updated: str | None = None

    def total_models(self) -> int:
        return len(self.nn_models) + len(self.nnue_models)


@dataclass
class ClusterModelState:
    """Aggregated model state across the cluster."""
    all_nn_models: set[str] = field(default_factory=set)
    all_nnue_models: set[str] = field(default_factory=set)
    model_hashes: dict[str, str] = field(default_factory=dict)  # name -> md5
    host_inventories: dict[str, HostModelInventory] = field(default_factory=dict)
    canonical_models: dict[str, str] = field(default_factory=dict)  # config_key -> model_name
    last_sync: str | None = None
    sync_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "all_nn_models": sorted(self.all_nn_models),
            "all_nnue_models": sorted(self.all_nnue_models),
            "model_hashes": self.model_hashes,
            "canonical_models": self.canonical_models,
            "host_inventories": {
                name: {
                    "host_name": inv.host_name,
                    "nn_models": sorted(inv.nn_models),
                    "nnue_models": sorted(inv.nnue_models),
                    "work_dir": inv.work_dir,
                    "reachable": inv.reachable,
                    "error": inv.error,
                    "last_updated": inv.last_updated,
                }
                for name, inv in self.host_inventories.items()
            },
            "last_sync": self.last_sync,
            "sync_errors": self.sync_errors,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ClusterModelState:
        state = cls(
            all_nn_models=set(data.get("all_nn_models", [])),
            all_nnue_models=set(data.get("all_nnue_models", [])),
            model_hashes=data.get("model_hashes", {}),
            canonical_models=data.get("canonical_models", {}),
            last_sync=data.get("last_sync"),
            sync_errors=data.get("sync_errors", []),
        )
        for name, inv_data in data.get("host_inventories", {}).items():
            state.host_inventories[name] = HostModelInventory(
                host_name=inv_data["host_name"],
                nn_models=set(inv_data.get("nn_models", [])),
                nnue_models=set(inv_data.get("nnue_models", [])),
                work_dir=inv_data.get("work_dir", ""),
                reachable=inv_data.get("reachable", False),
                error=inv_data.get("error"),
                last_updated=inv_data.get("last_updated"),
            )
        return state

    def save(self, path: Path = SYNC_STATE_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path = SYNC_STATE_PATH) -> ClusterModelState:
        if path.exists():
            with open(path) as f:
                return cls.from_dict(json.load(f))
        return cls()


# ============================================
# Model Hashing for Deduplication
# ============================================

def compute_file_md5(path: Path, chunk_size: int = 8192) -> str:
    """Compute MD5 hash of a file."""
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            md5.update(chunk)
    return md5.hexdigest()


def load_model_hashes() -> dict[str, str]:
    """Load cached model hashes."""
    if MODEL_HASHES_PATH.exists():
        with open(MODEL_HASHES_PATH) as f:
            return json.load(f)
    return {}


def save_model_hashes(hashes: dict[str, str]):
    """Save model hashes to cache."""
    MODEL_HASHES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_HASHES_PATH, "w") as f:
        json.dump(hashes, f, indent=2)


def get_local_model_hash(model_name: str, model_type: str = "nn") -> str | None:
    """Get hash of local model, computing if not cached."""
    hashes = load_model_hashes()
    cache_key = f"{model_type}:{model_name}"

    if model_type == "nnue":
        path = LOCAL_NNUE_DIR / model_name
    else:
        path = LOCAL_MODELS_DIR / model_name

    if not path.exists():
        return None

    # Check if cached hash is still valid (based on mtime)
    stat = path.stat()
    cache_mtime_key = f"{cache_key}:mtime"
    cached_mtime = hashes.get(cache_mtime_key)

    if cache_key in hashes and cached_mtime and float(cached_mtime) >= stat.st_mtime:
        return hashes[cache_key]

    # Compute new hash
    file_hash = compute_file_md5(path)
    hashes[cache_key] = file_hash
    hashes[cache_mtime_key] = str(stat.st_mtime)
    save_model_hashes(hashes)

    return file_hash


# ============================================
# Model Name Parsing
# ============================================

def parse_model_name(name: str) -> tuple[str | None, int | None, str | None]:
    """Extract board_type, num_players, version from model name."""
    board_type = None
    num_players = None
    version = None

    name_lower = name.lower()

    # Board types
    if "sq8" in name_lower or "square8" in name_lower:
        board_type = "square8"
    elif "sq19" in name_lower or "square19" in name_lower:
        board_type = "square19"
    elif "hex" in name_lower:
        board_type = "hexagonal"

    # Player counts
    if "_2p" in name or "2p_" in name or "_2p." in name:
        num_players = 2
    elif "_3p" in name or "3p_" in name or "_3p." in name:
        num_players = 3
    elif "_4p" in name or "4p_" in name or "_4p." in name:
        num_players = 4

    # Version
    for v in ["v8", "v7", "v6", "v5", "v4", "v3", "v2", "v1"]:
        if v in name_lower:
            version = v
            break

    return board_type, num_players, version


# ============================================
# Local Model Discovery
# ============================================

def get_local_models() -> tuple[set[str], set[str], dict[str, ModelInfo]]:
    """Get all NN and NNUE models from local machine with metadata."""
    nn_models = set()
    nnue_models = set()
    model_info = {}

    if LOCAL_MODELS_DIR.exists():
        # NN models (.pth files, excluding checkpoints)
        for pth in LOCAL_MODELS_DIR.glob("*.pth"):
            if "checkpoint" in pth.name:
                continue
            nn_models.add(pth.name)
            board_type, num_players, version = parse_model_name(pth.stem)
            model_info[pth.name] = ModelInfo(
                name=pth.name,
                path=str(pth),
                host="local",
                size_bytes=pth.stat().st_size,
                mtime=pth.stat().st_mtime,
                model_type="nn",
                board_type=board_type,
                num_players=num_players,
                version=version,
            )

        # NNUE models
        if LOCAL_NNUE_DIR.exists():
            for pt in LOCAL_NNUE_DIR.glob("*.pt"):
                nnue_models.add(pt.name)
                board_type, num_players, version = parse_model_name(pt.stem)
                model_info[pt.name] = ModelInfo(
                    name=pt.name,
                    path=str(pt),
                    host="local",
                    size_bytes=pt.stat().st_size,
                    mtime=pt.stat().st_mtime,
                    model_type="nnue",
                    board_type=board_type,
                    num_players=num_players,
                    version=version,
                )

    return nn_models, nnue_models, model_info


# ============================================
# Remote Model Discovery
# ============================================

def get_remote_models(host: HostConfig) -> HostModelInventory:
    """Get model inventory from a remote host using SSHExecutor."""
    inventory = HostModelInventory(
        host_name=host.name,
        work_dir=host.work_directory if hasattr(host, 'work_directory') else "~/ringrift/ai-service",
    )

    try:
        executor = SSHExecutor(host)
        if not executor.is_alive():
            inventory.error = "unreachable"
            return inventory

        inventory.reachable = True

        # List NN models
        result = executor.run('ls -1 models/*.pth 2>/dev/null | xargs -I{} basename {}', timeout=30)
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                if line.endswith(".pth") and "checkpoint" not in line:
                    inventory.nn_models.add(line)

        # List NNUE models
        result = executor.run('ls -1 models/nnue/*.pt 2>/dev/null | xargs -I{} basename {}', timeout=30)
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                if line.endswith(".pt"):
                    inventory.nnue_models.add(line)

        inventory.last_updated = datetime.now().isoformat()

    except Exception as e:
        inventory.error = str(e)[:200]

    return inventory


# ============================================
# Cluster Scanning
# ============================================

def scan_cluster(hosts: dict[str, HostConfig], max_workers: int = 10) -> ClusterModelState:
    """Scan all hosts in parallel to build cluster model state."""
    state = ClusterModelState()

    # Get local models first
    local_nn, local_nnue, local_info = get_local_models()
    state.all_nn_models.update(local_nn)
    state.all_nnue_models.update(local_nnue)
    state.host_inventories["local"] = HostModelInventory(
        host_name="local",
        nn_models=local_nn,
        nnue_models=local_nnue,
        work_dir=str(ROOT),
        reachable=True,
        last_updated=datetime.now().isoformat(),
    )

    logger.info(f"Local models: {len(local_nn)} NN, {len(local_nnue)} NNUE")

    if not hosts:
        logger.warning("No remote hosts configured")
        state.last_sync = datetime.now().isoformat()
        return state

    # Scan remote hosts in parallel
    logger.info(f"Scanning {len(hosts)} remote hosts...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_host = {
            executor.submit(get_remote_models, host): name
            for name, host in hosts.items()
        }

        for future in as_completed(future_to_host):
            name = future_to_host[future]
            try:
                inventory = future.result()
                state.host_inventories[name] = inventory
                if inventory.reachable:
                    state.all_nn_models.update(inventory.nn_models)
                    state.all_nnue_models.update(inventory.nnue_models)
                    logger.info(f"  {name}: {len(inventory.nn_models)} NN, {len(inventory.nnue_models)} NNUE")
                else:
                    logger.warning(f"  {name}: UNREACHABLE - {inventory.error}")
            except Exception as e:
                logger.error(f"  {name}: ERROR - {e}")
                state.sync_errors.append(f"{name}: {e}")

    # Determine canonical models for each config
    for model_name, info in local_info.items():
        if info.board_type and info.num_players:
            key = f"{info.model_type}_{info.board_type}_{info.num_players}p"
            existing = state.canonical_models.get(key)
            if not existing or info.mtime > local_info.get(existing, ModelInfo("", "", "", 0, 0, "")).mtime:
                state.canonical_models[key] = model_name

    state.last_sync = datetime.now().isoformat()
    return state


# ============================================
# Model Sync Operations
# ============================================

def sync_model_to_host(
    host: HostConfig,
    model_name: str,
    model_type: str = "nn",
    dry_run: bool = False,
    skip_retry_queue: bool = False,
) -> tuple[bool, str]:
    """Sync a single model to a remote host.

    Dec 2025: Added retry queue - failed syncs are automatically queued for retry
    with exponential backoff (10s, 20s, 40s). Use skip_retry_queue=True to disable.
    """
    if HAS_STORAGE_PROVIDER and not should_sync_to_node(host.name):
        record_nfs_skip("models")
        return True, f"Skipped sync to {host.name} (shared NFS storage)"

    if model_type == "nnue":
        local_path = LOCAL_NNUE_DIR / model_name
        remote_dir = f"{host.work_directory}/models/nnue/"
    else:
        local_path = LOCAL_MODELS_DIR / model_name
        remote_dir = f"{host.work_directory}/models/"

    if not local_path.exists():
        return False, f"Local file not found: {local_path}"

    if dry_run:
        return True, f"Would sync {model_name} to {host.name}"

    # Acquire sync_lock for coordinated file transfer
    sync_lock_acquired = acquire_sync_lock(host.name, timeout=60.0)
    if not sync_lock_acquired:
        logger.warning(f"{host.name}: Could not acquire sync lock, proceeding anyway")

    # Request bandwidth allocation for model transfer
    bandwidth_allocated = False
    if HAS_BANDWIDTH_MANAGER:
        try:
            bandwidth_allocated = request_bandwidth(
                host.name, mbps=50.0, priority=TransferPriority.HIGH
            )
            if not bandwidth_allocated:
                logger.warning(f"{host.name}: Bandwidth unavailable, proceeding anyway")
        except Exception as e:
            logger.warning(f"{host.name}: Bandwidth request error: {e}")

    try:
        # Build rsync command with corruption-prevention options:
        # --partial: Keep partial transfers for resume
        # --partial-dir: Store partials separately to avoid corrupt files
        # --delay-updates: Atomic update - put files in place only after full transfer
        # --checksum: Verify integrity after transfer
        ssh_opts = [
            "-o", "ConnectTimeout=10",
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=no",
            "-o", "TCPKeepAlive=yes",           # Keep TCP connection alive
            "-o", "ServerAliveInterval=30",      # Send keepalive every 30s
            "-o", "ServerAliveCountMax=3",       # Allow 3 missed keepalives
        ]
        if host.ssh_key:
            ssh_opts.extend(["-i", host.ssh_key_path if hasattr(host, 'ssh_key_path') else os.path.expanduser(host.ssh_key)])
        if host.ssh_port and int(host.ssh_port) != 22:
            ssh_opts.extend(["-p", str(int(host.ssh_port))])

        # Dec 2025: Get bandwidth limit from centralized config
        bwlimit_kbs = get_node_bandwidth_kbs(host.name)

        rsync_cmd = [
            "rsync", "-avz", "--progress",
            "--partial",                    # Keep partial transfers
            "--partial-dir=.rsync-partial", # Store partials in hidden dir
            "--delay-updates",              # Atomic: put files in place at end
            "--checksum",                   # Verify integrity after transfer
            "--timeout=60",                 # Per-file I/O timeout (60s stall)
            f"--bwlimit={bwlimit_kbs}",     # Dec 2025: Enforce bandwidth limit
            "-e", f"ssh {' '.join(ssh_opts)}",
            str(local_path),
            f"{host.ssh_target}:{remote_dir}",
        ]

        # Dynamic timeout: 2 seconds per MB, minimum 120s, maximum 1800s
        file_size_mb = local_path.stat().st_size / (1024 * 1024) if local_path.exists() else 100
        dynamic_timeout = max(120, min(1800, int(60 + file_size_mb * 2)))

        try:
            result = subprocess.run(rsync_cmd, capture_output=True, text=True, timeout=dynamic_timeout)
            if result.returncode == 0:
                # Post-transfer integrity verification
                try:
                    from app.coordination.sync_integrity import verify_sync_integrity
                    report = verify_sync_integrity(
                        source=local_path,
                        target=None,  # Remote verification requires SSH
                        check_db=False,  # Skip PRAGMA check for .pth files
                    )
                    if not report.is_valid:
                        logger.warning(f"Post-transfer verification issue: {report.summary()}")
                except ImportError:
                    pass  # sync_integrity module not available
                except Exception as verify_err:
                    logger.debug(f"Verification skipped: {verify_err}")
                return True, f"Synced {model_name} to {host.name}"

            # Dec 2025: Check for connection reset errors and try robust fallback
            stderr_lower = result.stderr.lower() if result.stderr else ""
            is_connection_error = any(
                pattern in stderr_lower
                for pattern in ["connection reset", "broken pipe", "connection refused", "timed out"]
            )

            if is_connection_error and HAS_ROBUST_TRANSFER and robust_push is not None:
                logger.info(f"{host.name}: rsync failed with connection error, trying robust_push fallback")
                try:
                    ssh_port = int(host.ssh_port) if host.ssh_port else 22
                    remote_file = f"{remote_dir}{model_name}"
                    config = TransferConfig(
                        ssh_key=os.path.expanduser(host.ssh_key) if host.ssh_key else None,
                        timeout=dynamic_timeout,
                        bandwidth_kbps=bwlimit_kbs,
                    )
                    transfer_result = robust_push(
                        str(local_path),
                        host.ssh_host if hasattr(host, 'ssh_host') else host.name,
                        ssh_port,
                        remote_file,
                        config,
                    )
                    if transfer_result.success:
                        logger.info(f"{host.name}: robust_push succeeded via {transfer_result.method}")
                        return True, f"Synced {model_name} to {host.name} (via {transfer_result.method})"
                    else:
                        logger.warning(f"{host.name}: robust_push also failed: {transfer_result.error}")
                except Exception as fallback_err:
                    logger.warning(f"{host.name}: robust_push fallback failed: {fallback_err}")

            # Sync failed - add to retry queue if enabled
            if not skip_retry_queue:
                _add_to_retry_queue(model_name, host.name, 0)
            return False, result.stderr[:200]
        except subprocess.TimeoutExpired:
            # Dec 2025: Try robust_push fallback for timeout (may be connection issue)
            if HAS_ROBUST_TRANSFER and robust_push is not None:
                logger.info(f"{host.name}: rsync timeout, trying robust_push fallback")
                try:
                    ssh_port = int(host.ssh_port) if host.ssh_port else 22
                    remote_file = f"{remote_dir}{model_name}"
                    config = TransferConfig(
                        ssh_key=os.path.expanduser(host.ssh_key) if host.ssh_key else None,
                        timeout=dynamic_timeout,
                        bandwidth_kbps=bwlimit_kbs,
                    )
                    transfer_result = robust_push(
                        str(local_path),
                        host.ssh_host if hasattr(host, 'ssh_host') else host.name,
                        ssh_port,
                        remote_file,
                        config,
                    )
                    if transfer_result.success:
                        logger.info(f"{host.name}: robust_push succeeded via {transfer_result.method}")
                        return True, f"Synced {model_name} to {host.name} (via {transfer_result.method})"
                    else:
                        logger.warning(f"{host.name}: robust_push also failed: {transfer_result.error}")
                except Exception as fallback_err:
                    logger.warning(f"{host.name}: robust_push fallback failed: {fallback_err}")

            # Timeout - add to retry queue if enabled
            if not skip_retry_queue:
                _add_to_retry_queue(model_name, host.name, 0)
            return False, f"rsync timeout after {dynamic_timeout}s"
        except Exception as e:
            # Other error - add to retry queue if enabled
            if not skip_retry_queue:
                _add_to_retry_queue(model_name, host.name, 0)
            return False, str(e)[:200]
    finally:
        # Release bandwidth and sync_lock
        if bandwidth_allocated and HAS_BANDWIDTH_MANAGER:
            try:
                release_bandwidth(host.name)
            except Exception as e:
                logger.warning(f"{host.name}: Bandwidth release error: {e}")
        if sync_lock_acquired:
            release_sync_lock(host.name)


def collect_model_from_host(
    host: HostConfig,
    model_name: str,
    model_type: str = "nn",
    dry_run: bool = False,
) -> tuple[bool, str]:
    """Collect a model from a remote host to local."""
    if HAS_STORAGE_PROVIDER and not should_sync_to_node(host.name):
        record_nfs_skip("models")
        return True, f"Skipped collection from {host.name} (shared NFS storage)"

    # Check disk usage before collecting (70% limit enforced 2025-12-16)
    has_capacity, disk_percent = check_disk_usage()
    if not has_capacity:
        return False, f"Disk full ({disk_percent:.1f}%), skipping collect"

    if model_type == "nnue":
        remote_path = f"{host.work_directory}/models/nnue/{model_name}"
        local_dir = LOCAL_NNUE_DIR
    else:
        remote_path = f"{host.work_directory}/models/{model_name}"
        local_dir = LOCAL_MODELS_DIR

    local_dir.mkdir(parents=True, exist_ok=True)
    local_path = local_dir / model_name

    if dry_run:
        return True, f"Would collect {model_name} from {host.name}"

    # Acquire sync_lock for coordinated file transfer
    sync_lock_acquired = acquire_sync_lock(host.name, timeout=60.0)
    if not sync_lock_acquired:
        logger.warning(f"{host.name}: Could not acquire sync lock, proceeding anyway")

    # Request bandwidth allocation for model transfer
    bandwidth_allocated = False
    if HAS_BANDWIDTH_MANAGER:
        try:
            bandwidth_allocated = request_bandwidth(
                host.name, mbps=50.0, priority=TransferPriority.HIGH
            )
            if not bandwidth_allocated:
                logger.warning(f"{host.name}: Bandwidth unavailable, proceeding anyway")
        except Exception as e:
            logger.warning(f"{host.name}: Bandwidth request error: {e}")

    try:
        # Use SCP via SSHExecutor with temp file + validation pattern
        temp_path = local_path.with_suffix('.pth.tmp')
        try:
            executor = SSHExecutor(host)
            result = executor.scp_from(remote_path, str(temp_path), timeout=RSYNC_TIMEOUT)
            if result.returncode != 0:
                temp_path.unlink(missing_ok=True)
                return False, (result.stderr or "Unknown error")[:200]

            # Validate downloaded file before finalizing
            try:
                from app.utils.torch_utils import safe_load_checkpoint
                test_load = safe_load_checkpoint(temp_path, map_location='cpu')
                if test_load is None:
                    temp_path.unlink(missing_ok=True)
                    return False, "Downloaded model is corrupt (loads as None)"
            except Exception as e:
                temp_path.unlink(missing_ok=True)
                return False, f"Downloaded model is corrupt: {str(e)[:100]}"

            # Atomic rename to final path
            temp_path.rename(local_path)
            return True, f"Collected and validated {model_name} from {host.name}"

        except Exception as e:
            temp_path.unlink(missing_ok=True)
            return False, str(e)[:200]
    finally:
        # Release bandwidth and sync_lock
        if bandwidth_allocated and HAS_BANDWIDTH_MANAGER:
            try:
                release_bandwidth(host.name)
            except Exception as e:
                logger.warning(f"{host.name}: Bandwidth release error: {e}")
        if sync_lock_acquired:
            release_sync_lock(host.name)


def load_cluster_cull_manifests(hosts: dict[str, HostConfig]) -> set[str]:
    """Load cull manifests from all hosts and merge them.

    Returns combined set of all culled model IDs across the cluster.
    This ensures culled models don't get re-synced.
    """
    culled_ids: set[str] = set()

    # Load local manifest
    local_manifest = LOCAL_MODELS_DIR / "cull_manifest.json"
    if local_manifest.exists():
        try:
            with open(local_manifest) as f:
                data = json.load(f)
                culled_ids.update(data.get("archived_ids", []))
                logger.debug(f"Loaded {len(data.get('archived_ids', []))} culled models from local manifest")
        except Exception as e:
            logger.warning(f"Failed to load local cull manifest: {e}")

    # Load remote manifests
    for host_name, host in hosts.items():
        try:
            executor = get_ssh_executor()
            if not executor:
                continue

            # Try to fetch cull_manifest.json
            remote_path = f"{host.work_dir}/ai-service/models/cull_manifest.json"
            local_tmp = Path(f"/tmp/cull_manifest_{host_name}.json")

            result = executor.scp_from(remote_path, str(local_tmp), host=host, timeout=10)
            if result.returncode == 0 and local_tmp.exists():
                with open(local_tmp) as f:
                    data = json.load(f)
                    remote_culled = set(data.get("archived_ids", []))
                    culled_ids.update(remote_culled)
                    logger.debug(f"Loaded {len(remote_culled)} culled models from {host_name}")
                local_tmp.unlink()
        except Exception as e:
            logger.debug(f"Could not load cull manifest from {host_name}: {e}")

    if culled_ids:
        logger.info(f"[Sync] Loaded {len(culled_ids)} culled models from cluster manifests - will skip during sync")

    return culled_ids


def sync_missing_models(
    state: ClusterModelState,
    hosts: dict[str, HostConfig],
    dry_run: bool = False,
    collect_first: bool = True,
) -> tuple[int, int, list[str]]:
    """Sync missing models across the cluster with deduplication.

    Args:
        state: Current cluster model state
        hosts: Host configurations
        dry_run: If True, don't actually sync
        collect_first: If True, collect missing models to local first

    Returns:
        (collected_count, distributed_count, errors)
    """
    collected = 0
    distributed = 0
    errors = []

    # Check disk usage before collecting (70% limit enforced 2025-12-16)
    if collect_first and not dry_run:
        has_capacity, disk_percent = check_disk_usage()
        if not has_capacity:
            logger.warning(f"Disk at {disk_percent:.1f}%, skipping model collection phase")
            collect_first = False  # Skip collect, but still distribute
            errors.append(f"disk_full:{disk_percent:.1f}%")

    # Load culled models from cluster manifests to skip during sync
    culled_models = load_cluster_cull_manifests(hosts)

    local_inv = state.host_inventories.get("local")
    if not local_inv:
        return 0, 0, ["No local inventory found"]

    # Phase 1: Collect models we don't have locally
    if collect_first:
        models_to_collect_nn = state.all_nn_models - local_inv.nn_models
        models_to_collect_nnue = state.all_nnue_models - local_inv.nnue_models

        # Filter out culled models - don't re-sync them
        if culled_models:
            models_to_collect_nn = {m for m in models_to_collect_nn if m not in culled_models}
            models_to_collect_nnue = {m for m in models_to_collect_nnue if m not in culled_models}

        if models_to_collect_nn or models_to_collect_nnue:
            logger.info(f"Collecting {len(models_to_collect_nn)} NN, {len(models_to_collect_nnue)} NNUE models to local...")

            for model_name in models_to_collect_nn:
                # Find a host that has this model
                for host_name, inv in state.host_inventories.items():
                    if host_name == "local" or not inv.reachable:
                        continue
                    if model_name in inv.nn_models:
                        host = hosts.get(host_name)
                        if host:
                            success, msg = collect_model_from_host(host, model_name, "nn", dry_run)
                            if success:
                                collected += 1
                                logger.info(f"  {msg}")
                            else:
                                errors.append(msg)
                            break

            for model_name in models_to_collect_nnue:
                for host_name, inv in state.host_inventories.items():
                    if host_name == "local" or not inv.reachable:
                        continue
                    if model_name in inv.nnue_models:
                        host = hosts.get(host_name)
                        if host:
                            success, msg = collect_model_from_host(host, model_name, "nnue", dry_run)
                            if success:
                                collected += 1
                                logger.info(f"  {msg}")
                            else:
                                errors.append(msg)
                            break

    # Refresh local models after collection
    local_nn, local_nnue, _ = get_local_models()

    # Phase 2: Distribute local models to hosts that are missing them
    logger.info("Distributing models to cluster...")

    for host_name, inv in state.host_inventories.items():
        if host_name == "local" or not inv.reachable:
            continue

        host = hosts.get(host_name)
        if not host:
            continue

        missing_nn = local_nn - inv.nn_models
        missing_nnue = local_nnue - inv.nnue_models

        # Filter out culled models - don't distribute them
        if culled_models:
            missing_nn = {m for m in missing_nn if m not in culled_models}
            missing_nnue = {m for m in missing_nnue if m not in culled_models}

        if not missing_nn and not missing_nnue:
            logger.info(f"  {host_name}: up to date")
            continue

        logger.info(f"  {host_name}: syncing {len(missing_nn)} NN, {len(missing_nnue)} NNUE...")

        for model_name in missing_nn:
            success, msg = sync_model_to_host(host, model_name, "nn", dry_run)
            if success:
                distributed += 1
            else:
                errors.append(f"{host_name}: {model_name} - {msg}")

        for model_name in missing_nnue:
            success, msg = sync_model_to_host(host, model_name, "nnue", dry_run)
            if success:
                distributed += 1
            else:
                errors.append(f"{host_name}: {model_name} - {msg}")

    return collected, distributed, errors


# ============================================
# Daemon Integration
# ============================================

def sync_models_after_training(model_path: Path, model_type: str = "nn") -> bool:
    """Hook for continuous_improvement_daemon after training a new model.

    Distributes the new model to all GPU/training hosts.
    """
    if not HOSTS_MODULE_AVAILABLE:
        logger.warning("Hosts module not available, skipping sync")
        return False

    hosts = load_remote_hosts()
    training_hosts = {
        name: host for name, host in hosts.items()
        if "training" in (host.properties.get("role") or "") or
           "gpu" in (host.properties.get("role") or "").lower()
    }

    if not training_hosts:
        logger.warning("No training hosts found")
        return False

    model_name = model_path.name
    success_count = 0

    logger.info(f"Syncing new {model_type} model to {len(training_hosts)} training hosts...")

    for host_name, host in training_hosts.items():
        success, msg = sync_model_to_host(host, model_name, model_type)
        if success:
            success_count += 1
            logger.info(f"  {msg}")
        else:
            logger.warning(f"  {host_name}: {msg}")

    logger.info(f"Synced to {success_count}/{len(training_hosts)} hosts")
    return success_count > 0


def get_canonical_model_for_config(board_type: str, num_players: int, model_type: str = "nn") -> Path | None:
    """Get the canonical model for a specific configuration.

    Used by orchestrators to determine which model to use.
    """
    state = ClusterModelState.load()
    key = f"{model_type}_{board_type}_{num_players}p"

    if key in state.canonical_models:
        model_name = state.canonical_models[key]
        if model_type == "nnue":
            path = LOCAL_NNUE_DIR / model_name
        else:
            path = LOCAL_MODELS_DIR / model_name
        if path.exists():
            return path

    # Fallback: discover locally
    _local_nn, _local_nnue, local_info = get_local_models()
    for _name, info in local_info.items():
        if info.model_type == model_type and info.board_type == board_type and info.num_players == num_players:
            return Path(info.path)

    return None


# ============================================
# CLI
# ============================================

def print_summary(state: ClusterModelState):
    """Print summary of cluster model state."""
    print("\n" + "=" * 80)
    print(" Cluster Model Summary")
    print("=" * 80)

    print(f"\nTotal unique models: {len(state.all_nn_models)} NN, {len(state.all_nnue_models)} NNUE")

    print("\nModels per host:")
    for name, inv in sorted(state.host_inventories.items()):
        status = "OK" if inv.reachable else f"UNREACHABLE ({inv.error})"
        if inv.reachable:
            print(f"  {name:<25} NN: {len(inv.nn_models):>4}  NNUE: {len(inv.nnue_models):>2}  [{status}]")
        else:
            print(f"  {name:<25} [{status}]")

    if state.canonical_models:
        print("\nCanonical models (newest for each config):")
        for key in sorted(state.canonical_models.keys()):
            print(f"  {key:<30} {state.canonical_models[key]}")


def _run_async(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


def _sync_via_coordinator() -> bool:
    """Use SyncCoordinator for model sync (aria2/SSH/P2P + NFS-aware)."""
    try:
        from app.distributed.sync_coordinator import SyncCoordinator
    except Exception as e:
        logger.warning(f"SyncCoordinator unavailable: {e}")
        return False

    async def _do_sync():
        coordinator = SyncCoordinator.get_instance()
        return await coordinator.sync_models()

    try:
        stats = _run_async(_do_sync())
    except Exception as e:
        logger.warning(f"SyncCoordinator model sync failed: {e}")
        return False

    if stats:
        logger.info(
            f"SyncCoordinator: {stats.files_synced} models via {stats.transport_used} "
            f"({stats.bytes_transferred / (1024 * 1024):.1f}MB)"
        )
    return True


def run_daemon(
    interval_minutes: int = 30,
    use_sync_coordinator: bool = False,
    config_path: str | None = None,
    include_nonready: bool = False,
):
    """Run sync daemon that syncs every N minutes."""
    logger.info(f"Starting model sync daemon (interval: {interval_minutes}min)")

    shutdown = False

    def signal_handler(sig, frame):
        nonlocal shutdown
        logger.info("Shutdown signal received")
        shutdown = True

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    while not shutdown:
        try:
            logger.info("Running sync cycle...")
            if HOSTS_MODULE_AVAILABLE:
                hosts = (
                    load_remote_hosts(config_path=config_path)
                    if include_nonready
                    else load_ready_hosts(config_path=config_path)
                )
            else:
                hosts = {}
            coordinator_used = False
            if use_sync_coordinator:
                coordinator_used = _sync_via_coordinator()
            state = scan_cluster(hosts)
            collected, distributed, errors = sync_missing_models(
                state,
                hosts,
                collect_first=not coordinator_used,
            )
            state.save()

            logger.info(f"Sync complete: collected {collected}, distributed {distributed}, errors {len(errors)}")

            if errors:
                for err in errors[:5]:
                    logger.warning(f"  {err}")

        except Exception as e:
            logger.error(f"Sync cycle failed: {e}")

        # Wait for next cycle
        for _ in range(interval_minutes * 60):
            if shutdown:
                break
            time.sleep(1)

    logger.info("Daemon shutdown complete")


def main():
    parser = argparse.ArgumentParser(description="Sync NN/NNUE models across cluster")
    parser.add_argument("--discover", action="store_true", help="Discover models on all hosts")
    parser.add_argument("--collect", action="store_true", help="Collect models from cluster to local")
    parser.add_argument("--distribute", action="store_true", help="Distribute local models to cluster")
    parser.add_argument("--sync", action="store_true", help="Full sync (collect + distribute)")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon, syncing periodically")
    parser.add_argument("--interval", type=int, default=30, help="Daemon sync interval in minutes")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--host", type=str, help="Only sync to specific host")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--config", type=str, help="Path to distributed_hosts.yaml override")
    parser.add_argument(
        "--include-nonready",
        action="store_true",
        help="Include non-ready hosts (offline/disabled/unstable) in sync operations",
    )
    parser.add_argument("--no-lock", action="store_true", help="Skip singleton lock (for testing)")
    parser.add_argument(
        "--use-sync-coordinator",
        action="store_true",
        help=(
            "Use SyncCoordinator for model collection (aria2/SSH/P2P + NFS-aware). "
            "Distribution still uses SSH-based deduplication."
        ),
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Singleton lock to prevent multiple instances from stacking up
    # (cron runs every 30 min, but sync can take longer with slow hosts)
    lock_file = Path("/tmp/ringrift_sync_models.lock")
    lock_fd = None
    if not args.no_lock:
        import fcntl
        try:
            lock_fd = open(lock_file, "w")
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            lock_fd.write(str(os.getpid()))
            lock_fd.flush()
        except OSError:
            logger.warning("Another sync_models instance is already running, exiting")
            return 0  # Exit gracefully, not an error

    try:
        return _main_impl(args)
    finally:
        if lock_fd:
            import fcntl
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()
            with contextlib.suppress(OSError):
                lock_file.unlink()


def _main_impl(args):
    """Main implementation after lock is acquired."""
    if not HOSTS_MODULE_AVAILABLE:
        logger.error("app.distributed.hosts module not available")
        return 1

    # Load hosts
    hosts = (
        load_remote_hosts(config_path=args.config)
        if args.include_nonready
        else load_ready_hosts(config_path=args.config)
    )
    logger.info(f"Loaded {len(hosts)} remote hosts")

    if args.host:
        if args.host not in hosts:
            logger.error(
                f"Host '{args.host}' not found (use --include-nonready to allow offline/disabled hosts)"
            )
            return 1
        hosts = {args.host: hosts[args.host]}

    # Daemon mode
    if args.daemon:
        run_daemon(
            args.interval,
            use_sync_coordinator=args.use_sync_coordinator,
            config_path=args.config,
            include_nonready=args.include_nonready,
        )
        return 0

    coordinator_used = False
    if args.use_sync_coordinator and (args.collect or args.sync):
        if args.host:
            logger.warning("--host filtering is ignored when using SyncCoordinator")
        coordinator_used = _sync_via_coordinator()

    # Scan cluster
    state = scan_cluster(hosts)
    print_summary(state)

    if args.discover:
        state.save()
        return 0

    # Sync operations
    if args.sync or args.collect or args.distribute:
        collected, distributed, errors = sync_missing_models(
            state, hosts,
            dry_run=args.dry_run,
            collect_first=(args.sync or args.collect) and not coordinator_used,
        )

        print(f"\nSync results: collected {collected}, distributed {distributed}")
        if errors:
            print(f"Errors ({len(errors)}):")
            for err in errors[:10]:
                print(f"  - {err}")

        if not args.dry_run:
            state.save()

    return 0


if __name__ == "__main__":
    sys.exit(main())
