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

    # Dry run
    python scripts/sync_models.py --sync --dry-run

    # Daemon mode - sync every N minutes
    python scripts/sync_models.py --daemon --interval 30
"""
from __future__ import annotations

import argparse
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
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Add ai-service to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Use canonical host configuration module
try:
    from app.distributed.hosts import (
        HostConfig,
        load_remote_hosts,
        SSHExecutor,
        get_ssh_executor,
    )
    HOSTS_MODULE_AVAILABLE = True
except ImportError:
    HOSTS_MODULE_AVAILABLE = False
    HostConfig = None

# Import coordination helpers for sync lock and bandwidth management
from app.coordination.helpers import (
    has_sync_lock,
    acquire_sync_lock_safe,
    release_sync_lock_safe,
    has_bandwidth_manager,
    request_bandwidth_safe,
    release_bandwidth_safe,
    get_transfer_priorities,
)

HAS_SYNC_LOCK = has_sync_lock()
HAS_BANDWIDTH_MANAGER = has_bandwidth_manager()
TransferPriority = get_transfer_priorities()

# Unified resource checking utilities (80% max utilization)
try:
    from app.utils.resource_guard import (
        get_disk_usage as unified_get_disk_usage,
        check_disk_space as unified_check_disk,
        LIMITS as RESOURCE_LIMITS,
    )
    HAS_RESOURCE_GUARD = True
except ImportError:
    HAS_RESOURCE_GUARD = False
    unified_get_disk_usage = None
    unified_check_disk = None
    RESOURCE_LIMITS = None

# Wrapper functions for backwards compatibility
def acquire_sync_lock(host: str, timeout: float = 30.0) -> bool:
    return acquire_sync_lock_safe(host, timeout)

def release_sync_lock(host: str) -> None:
    release_sync_lock_safe(host)

def request_bandwidth(host: str, mbps: float = 100.0, priority=None):
    return request_bandwidth_safe(host, mbps, priority)

def release_bandwidth(host: str) -> None:
    release_bandwidth_safe(host)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================
# Disk Usage Limits (70% max enforced 2025-12-16)
# ============================================

MAX_DISK_USAGE_PERCENT = float(os.environ.get("RINGRIFT_MAX_DISK_PERCENT", "70"))


def check_disk_usage(path: Path = None) -> tuple[bool, float]:
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
        except Exception:
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
    md5_hash: Optional[str] = None
    board_type: Optional[str] = None
    num_players: Optional[int] = None
    version: Optional[str] = None


@dataclass
class HostModelInventory:
    """Model inventory for a single host."""
    host_name: str
    nn_models: Set[str] = field(default_factory=set)
    nnue_models: Set[str] = field(default_factory=set)
    model_hashes: Dict[str, str] = field(default_factory=dict)
    work_dir: str = ""
    reachable: bool = False
    error: Optional[str] = None
    last_updated: Optional[str] = None

    def total_models(self) -> int:
        return len(self.nn_models) + len(self.nnue_models)


@dataclass
class ClusterModelState:
    """Aggregated model state across the cluster."""
    all_nn_models: Set[str] = field(default_factory=set)
    all_nnue_models: Set[str] = field(default_factory=set)
    model_hashes: Dict[str, str] = field(default_factory=dict)  # name -> md5
    host_inventories: Dict[str, HostModelInventory] = field(default_factory=dict)
    canonical_models: Dict[str, str] = field(default_factory=dict)  # config_key -> model_name
    last_sync: Optional[str] = None
    sync_errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
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
    def from_dict(cls, data: Dict) -> "ClusterModelState":
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
    def load(cls, path: Path = SYNC_STATE_PATH) -> "ClusterModelState":
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


def load_model_hashes() -> Dict[str, str]:
    """Load cached model hashes."""
    if MODEL_HASHES_PATH.exists():
        with open(MODEL_HASHES_PATH) as f:
            return json.load(f)
    return {}


def save_model_hashes(hashes: Dict[str, str]):
    """Save model hashes to cache."""
    MODEL_HASHES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_HASHES_PATH, "w") as f:
        json.dump(hashes, f, indent=2)


def get_local_model_hash(model_name: str, model_type: str = "nn") -> Optional[str]:
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

def parse_model_name(name: str) -> Tuple[Optional[str], Optional[int], Optional[str]]:
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

def get_local_models() -> Tuple[Set[str], Set[str], Dict[str, ModelInfo]]:
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

def scan_cluster(hosts: Dict[str, HostConfig], max_workers: int = 10) -> ClusterModelState:
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
) -> Tuple[bool, str]:
    """Sync a single model to a remote host."""
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
        # Build rsync command
        ssh_opts = ["-o", "ConnectTimeout=10", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no"]
        if host.ssh_key:
            ssh_opts.extend(["-i", host.ssh_key_path if hasattr(host, 'ssh_key_path') else os.path.expanduser(host.ssh_key)])
        if host.ssh_port and int(host.ssh_port) != 22:
            ssh_opts.extend(["-p", str(int(host.ssh_port))])

        rsync_cmd = [
            "rsync", "-avz", "--progress",
            "-e", f"ssh {' '.join(ssh_opts)}",
            str(local_path),
            f"{host.ssh_target}:{remote_dir}",
        ]

        try:
            result = subprocess.run(rsync_cmd, capture_output=True, text=True, timeout=RSYNC_TIMEOUT)
            if result.returncode == 0:
                return True, f"Synced {model_name} to {host.name}"
            return False, result.stderr[:200]
        except subprocess.TimeoutExpired:
            return False, "rsync timeout"
        except Exception as e:
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
) -> Tuple[bool, str]:
    """Collect a model from a remote host to local."""
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
        # Use SCP via SSHExecutor
        try:
            executor = SSHExecutor(host)
            result = executor.scp_from(remote_path, str(local_path), timeout=RSYNC_TIMEOUT)
            if result.returncode == 0:
                return True, f"Collected {model_name} from {host.name}"
            return False, (result.stderr or "Unknown error")[:200]
        except Exception as e:
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


def load_cluster_cull_manifests(hosts: Dict[str, HostConfig]) -> Set[str]:
    """Load cull manifests from all hosts and merge them.

    Returns combined set of all culled model IDs across the cluster.
    This ensures culled models don't get re-synced.
    """
    culled_ids: Set[str] = set()

    # Load local manifest
    local_manifest = LOCAL_MODELS_DIR / "cull_manifest.json"
    if local_manifest.exists():
        try:
            with open(local_manifest, "r") as f:
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
                with open(local_tmp, "r") as f:
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
    hosts: Dict[str, HostConfig],
    dry_run: bool = False,
    collect_first: bool = True,
) -> Tuple[int, int, List[str]]:
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


def get_canonical_model_for_config(board_type: str, num_players: int, model_type: str = "nn") -> Optional[Path]:
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
    local_nn, local_nnue, local_info = get_local_models()
    for name, info in local_info.items():
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


def run_daemon(interval_minutes: int = 30):
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
            hosts = load_remote_hosts() if HOSTS_MODULE_AVAILABLE else {}
            state = scan_cluster(hosts)
            collected, distributed, errors = sync_missing_models(state, hosts)
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

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not HOSTS_MODULE_AVAILABLE:
        logger.error("app.distributed.hosts module not available")
        return 1

    # Load hosts
    hosts = load_remote_hosts()
    logger.info(f"Loaded {len(hosts)} remote hosts")

    if args.host:
        if args.host not in hosts:
            logger.error(f"Host '{args.host}' not found")
            return 1
        hosts = {args.host: hosts[args.host]}

    # Daemon mode
    if args.daemon:
        run_daemon(args.interval)
        return 0

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
            collect_first=args.sync or args.collect,
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
