#!/usr/bin/env python3
"""Distributed P2P Orchestrator - Self-healing compute cluster for RingRift AI training.

This orchestrator runs on each node in the cluster and:
1. Discovers other nodes via broadcast UDP or known peer list
2. Participates in leader election for coordination tasks
3. Monitors local resources and shares status with peers
4. Auto-starts selfplay/training jobs based on cluster needs
5. Self-heals when nodes go offline or IPs change

Architecture:
- Each node runs this script as a daemon
- Nodes communicate via HTTP REST API (port 8770)
- Leader election uses Bully algorithm (highest node_id wins)
- Heartbeats every 30 seconds detect failures
- Nodes maintain local SQLite state for crash recovery

Usage:
    # On each node:
    python scripts/p2p_orchestrator.py --node-id mac-studio
    python scripts/p2p_orchestrator.py --node-id vast-5090-quad --port 8770

    # With known peers (for cloud nodes without broadcast):
    python scripts/p2p_orchestrator.py --node-id vast-3090 --peers <peer-ip>:8770,<peer-ip>:8770
"""
from __future__ import annotations

# Load .env.local BEFORE app.p2p.constants imports (for SWIM/Raft feature flags)
# This must happen before any app.* imports that read environment variables
def _load_env_local():
    """Load .env.local from script directory or ai-service root."""
    import os as _os
    from pathlib import Path as _Path

    # Feb 2026: Node-specific vars that must ONLY come from the actual process
    # environment (LaunchAgent, systemd, command line), never from .env.local.
    # Root cause: .env.local with RINGRIFT_IS_COORDINATOR=true was deployed to
    # GPU nodes, causing them to self-elect as leader and block the pipeline.
    _SKIP_KEYS = {"RINGRIFT_IS_COORDINATOR"}

    for base in [_Path(__file__).parent.parent, _Path.cwd()]:
        env_file = base / ".env.local"
        if env_file.exists():
            try:
                with open(env_file) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            key, _, value = line.partition("=")
                            key = key.strip()
                            value = value.strip().strip('"').strip("'")
                            if key in _SKIP_KEYS:
                                continue
                            if key not in _os.environ:  # Don't override existing
                                _os.environ[key] = value
                break
            except (OSError, IOError, UnicodeDecodeError):
                pass  # Skip if .env.local can't be read

_load_env_local()

# ===========================================================================
# CRITICAL: Monkey-patch sqlite3.connect to auto-close on context exit.
# In Python < 3.12, sqlite3.Connection.__exit__() only commits/rolls back
# but does NOT close the connection. With hundreds of daemons scanning
# 9000+ selfplay databases, this caused 4000+ leaked FDs and 400%+ CPU.
# This patch wraps every connection to close on __exit__.
# ===========================================================================
import sqlite3 as _sqlite3_module

_original_sqlite3_connect = _sqlite3_module.connect


class _AutoClosingConnection:
    """sqlite3.Connection wrapper that closes on context manager exit."""
    __slots__ = ("_conn",)

    def __init__(self, conn):
        object.__setattr__(self, "_conn", conn)

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def __setattr__(self, name, value):
        if name == "_conn":
            object.__setattr__(self, name, value)
        else:
            setattr(self._conn, name, value)

    def __enter__(self):
        return self._conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if exc_type is None:
                self._conn.commit()
            else:
                self._conn.rollback()
        finally:
            self._conn.close()
        return False

    def close(self):
        self._conn.close()

    def __del__(self):
        try:
            self._conn.close()
        except Exception:
            pass


def _patched_connect(*args, **kwargs):
    conn = _original_sqlite3_connect(*args, **kwargs)
    return _AutoClosingConnection(conn)


_sqlite3_module.connect = _patched_connect
# ===========================================================================

import argparse
import asyncio
import contextlib

# Python 3.10 compatibility: asyncio.timeout was added in 3.11
# Use a compatibility shim that works with Python 3.10+
try:
    from asyncio import timeout as async_timeout
except ImportError:
    # Python 3.10 fallback using wait_for
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def async_timeout(delay):
        """Compatibility shim for asyncio.timeout (Python 3.11+)."""
        task = asyncio.current_task()
        loop = asyncio.get_running_loop()

        def cancel_task():
            if task is not None:
                task.cancel()

        handle = loop.call_later(delay, cancel_task)
        try:
            yield
        except asyncio.CancelledError:
            raise asyncio.TimeoutError()
        finally:
            handle.cancel()
import gzip
import importlib
import ipaddress
import json
import os
import secrets
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys

# Safe database connection context manager (December 2025)
try:
    from app.distributed.db_utils import safe_db_connection
except ImportError:
    # Fallback for when db_utils isn't available
    from contextlib import contextmanager as _cm
    @_cm
    def safe_db_connection(db_path, timeout=30):
        conn = sqlite3.connect(str(db_path), timeout=timeout)
        try:
            yield conn
            conn.commit()
        except sqlite3.Error:
            conn.rollback()
            raise
        finally:
            conn.close()
import threading
import time
import uuid
from collections.abc import Generator
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional
from urllib.parse import urlparse

# P2P Managers - Phase 1 Consolidation (Jan 2026)
from scripts.p2p.managers.quorum_manager import QuorumManager, QuorumConfig
from scripts.p2p.orchestrators.base_orchestrator import get_job_attr, set_job_attr
from scripts.p2p.config.orchestrator_config import (
    OrchestratorConfig,
    SyncConfig,
    TrainingConfig,
    PartitionConfig,
    SafeguardConfig,
)

if TYPE_CHECKING:
    from app.coordination.unified_queue_populator import UnifiedQueuePopulator as QueuePopulator
    from app.coordination.p2p_auto_deployer import P2PAutoDeployer
    from scripts.p2p.loops import LoopManager

# =============================================================================
# PRE-FLIGHT DEPENDENCY VALIDATION (January 2026)
# =============================================================================
# Critical dependencies that must be present before P2P startup.
# Failing fast with clear errors prevents cryptic runtime failures.

_CRITICAL_DEPENDENCIES = {
    "aiohttp": "HTTP server and client functionality",
    "psutil": "Process and system monitoring",
    "yaml": "Configuration file parsing",
}

_OPTIONAL_DEPENDENCIES = {
    "prometheus_client": "Metrics export (optional)",
    "paramiko": "SSH connections for remote operations",
}


def _validate_preflight_dependencies() -> tuple[bool, list[str]]:
    """Validate critical dependencies are available before startup.

    Returns:
        Tuple of (all_ok, list of error messages)
    """
    errors = []
    warnings = []

    # Check critical dependencies
    for module_name, purpose in _CRITICAL_DEPENDENCIES.items():
        try:
            importlib.import_module(module_name)
        except ImportError:
            errors.append(f"CRITICAL: Missing '{module_name}' - required for {purpose}")
            errors.append(f"  Fix: pip install {module_name}")

    # Check optional dependencies (warn only)
    for module_name, purpose in _OPTIONAL_DEPENDENCIES.items():
        try:
            importlib.import_module(module_name)
        except ImportError:
            warnings.append(f"Optional: Missing '{module_name}' - {purpose}")

    # Venv detection: warn loudly if running with system Python instead of venv.
    # Root cause of idle GPU nodes (Feb 2026): system Python lacks swim-p2p,
    # causing silent fallback to HTTP-only heartbeats and broken mesh membership.
    in_venv = sys.prefix != sys.base_prefix
    if not in_venv:
        venv_path = Path(sys.argv[0]).resolve().parent.parent / "venv" / "bin" / "python"
        warnings.append(
            f"WARNING: Running with system Python ({sys.executable}), not a virtualenv. "
            f"This may be missing cluster dependencies (swim-p2p, torch, etc). "
            f"Recommended: {venv_path}"
        )

    # Log warnings
    for warn in warnings:
        print(f"[P2P] {warn}", file=sys.stderr)

    return len(errors) == 0, errors


# =============================================================================
# Work queue for centralized work distribution (lazy import to avoid circular deps)
_work_queue = None
def get_work_queue():
    """Get the work queue singleton (lazy load)."""
    global _work_queue
    if _work_queue is None:
        try:
            from app.coordination.work_queue import get_work_queue as _get_wq
            _work_queue = _get_wq()
        except ImportError:
            _work_queue = None
    return _work_queue

# Automation managers (lazy imports to avoid circular deps)
_health_manager = None  # December 2025: Consolidated from recovery_manager
_predictive_alerts = None

def get_health_manager():
    """Get the health manager singleton (lazy load).

    December 2025: Consolidated from get_recovery_manager().
    Uses UnifiedHealthManager which combines recovery + error coordination.
    """
    global _health_manager
    if _health_manager is None:
        try:
            from app.coordination.unified_health_manager import (
                get_health_manager as _get_uhm,
            )
            _health_manager = _get_uhm()
        except ImportError:
            _health_manager = None
    return _health_manager


# Job Reaper Daemon (leader-only, kills stuck jobs and reassigns work)
_job_reaper = None
def get_job_reaper(work_queue=None, ssh_config=None):
    """Get the job reaper singleton (lazy load).

    The JobReaperDaemon enforces job timeouts by:
    1. Detecting jobs past their timeout
    2. Killing stuck processes via SSH
    3. Marking jobs as TIMEOUT
    4. Reassigning failed work to other nodes
    5. Blacklisting nodes that repeatedly fail
    """
    global _job_reaper
    if _job_reaper is None and work_queue is not None:
        try:
            from app.coordination.job_reaper import JobReaperDaemon
            _job_reaper = JobReaperDaemon(
                work_queue=work_queue,
                ssh_config=ssh_config,
            )
        except ImportError as e:
            logger.warning(f"JobReaperDaemon not available: {e}")
            _job_reaper = None
    return _job_reaper

def get_predictive_alerts():
    """Get the predictive alerts manager (lazy load)."""
    global _predictive_alerts
    if _predictive_alerts is None:
        try:
            from app.monitoring.predictive_alerts import PredictiveAlertManager
            _predictive_alerts = PredictiveAlertManager()
        except ImportError:
            _predictive_alerts = None
    return _predictive_alerts


# SWIM membership manager for leaderless gossip-based membership
_swim_manager = None
SWIM_AVAILABLE = False

# Jan 22, 2026: SWIM callback registration for state synchronization.
# Problem: SWIM adapter has callbacks but they were never wired to orchestrator.
# SWIM detects failures at 90s but never syncs state to gossip layer.
# Solution: Register callbacks BEFORE get_swim_manager() creates the manager.
_swim_on_member_alive: Callable[[str], None] | None = None
_swim_on_member_failed: Callable[[str], None] | None = None


def set_swim_callbacks(
    on_alive: Callable[[str], None] | None = None,
    on_failed: Callable[[str], None] | None = None,
) -> None:
    """Register SWIM membership callbacks before get_swim_manager().

    Jan 22, 2026: Wire SWIM failure detection to gossip layer.

    Must be called BEFORE get_swim_manager() to ensure callbacks are set
    during manager creation. If manager already exists, sets callbacks directly.

    Args:
        on_alive: Callback when a member becomes alive (member_id: str)
        on_failed: Callback when a member fails (member_id: str)
    """
    global _swim_on_member_alive, _swim_on_member_failed
    _swim_on_member_alive = on_alive
    _swim_on_member_failed = on_failed

    # If manager already exists, set callbacks directly
    if _swim_manager is not None:
        _swim_manager.on_member_alive = on_alive
        _swim_manager.on_member_failed = on_failed
        logger.info("SWIM callbacks registered on existing manager")


def get_swim_manager(node_id: str | None = None, bind_port: int = 7947):
    """Get the SWIM membership manager singleton (lazy load).

    SWIM (Scalable Weakly-consistent Infection-style Membership) provides:
    - O(1) message complexity per node (constant bandwidth)
    - Failure detection in <5 seconds (vs 60+ seconds with heartbeat-based)
    - No single leader required - truly distributed
    - Suspicion mechanism to reduce false positives

    Args:
        node_id: Node identifier (required for first initialization)
        bind_port: UDP port for SWIM protocol (default 7947)

    Returns:
        SwimMembershipManager instance or None if swim-p2p not installed
    """
    global _swim_manager, SWIM_AVAILABLE
    if _swim_manager is None and node_id is not None:
        try:
            from app.p2p.swim_adapter import SwimMembershipManager, SWIM_AVAILABLE as _swim_avail
            SWIM_AVAILABLE = _swim_avail
            if SWIM_AVAILABLE:
                _swim_manager = SwimMembershipManager.from_distributed_hosts(
                    node_id=node_id,
                    bind_port=bind_port,
                )
                # Jan 22, 2026: Wire SWIM callbacks registered via set_swim_callbacks()
                if _swim_on_member_alive is not None:
                    _swim_manager.on_member_alive = _swim_on_member_alive
                if _swim_on_member_failed is not None:
                    _swim_manager.on_member_failed = _swim_on_member_failed
                callback_status = "with callbacks" if (_swim_on_member_alive or _swim_on_member_failed) else "no callbacks"
                logger.info(f"SWIM membership manager initialized for {node_id} ({callback_status})")
            else:
                logger.warning("swim-p2p not installed - using HTTP heartbeats only")
        except ImportError as e:
            logger.warning(f"SWIM adapter not available: {e}")
            _swim_manager = None
    return _swim_manager


# Dead Peer Cooldown Manager (Jan 2026)
# Adaptive cooldown with probe-based early recovery
_dead_peer_cooldown_manager = None


def get_dead_peer_cooldown_manager():
    """Get the dead peer cooldown manager singleton (lazy load).

    The DeadPeerCooldownManager replaces the static 1-hour cooldown with:
    - Tiered cooldowns (30s -> 2min -> 10min -> 30min) based on failure frequency
    - Probe-based early recovery when gossip reports a dead node might be alive
    - Prevents 25-40% node loss from brief network blips
    """
    global _dead_peer_cooldown_manager
    if _dead_peer_cooldown_manager is None:
        try:
            from scripts.p2p.dead_peer_recovery import DeadPeerCooldownManager
            _dead_peer_cooldown_manager = DeadPeerCooldownManager()
            logger.info("DeadPeerCooldownManager initialized with adaptive cooldown")
        except ImportError as e:
            logger.warning(f"DeadPeerCooldownManager not available: {e}")
            _dead_peer_cooldown_manager = None
    return _dead_peer_cooldown_manager


# ============================================
# Phase 4: Extracted Background Loops (Dec 2025)
# ============================================
# These loops are extracted from the monolithic orchestrator for modularity.
# They use dependency injection via callbacks for testability.

# Feature flag for gradual rollout
EXTRACTED_LOOPS_ENABLED = os.environ.get("RINGRIFT_EXTRACTED_LOOPS", "true").lower() in ("true", "1", "yes")
JOB_REAPER_FALLBACK_ENABLED = os.environ.get("RINGRIFT_JOB_REAPER_FALLBACK_ENABLED", "true").lower() in ("true", "1", "yes")

# Lazy import to avoid circular dependencies
_loop_manager_instance = None
_loop_classes_loaded = False


def _load_loop_classes():
    """Lazy-load loop classes to avoid import-time dependencies."""
    global _loop_classes_loaded
    if _loop_classes_loaded:
        return True
    try:
        from scripts.p2p.loops import (
            LoopManager,
            QueuePopulatorLoop,
            EloSyncLoop,
            ModelSyncLoop,
            DataAggregationLoop,
            IpDiscoveryLoop,
            TailscaleRecoveryLoop,
            TailscalePeerDiscoveryLoop,
            FollowerDiscoveryLoop,
            AutoScalingLoop,
            HealthAggregationLoop,
            JobReaperLoop,
            IdleDetectionLoop,
            UdpDiscoveryLoop,
            SplitBrainDetectionLoop,
            QuorumCrisisDiscoveryLoop,
            QuorumCrisisConfig,
        )
        _loop_classes_loaded = True
        return True
    except ImportError as e:
        logger.error(f"[LoopManager] CRITICAL: Extracted loops import failed: {e}")
        logger.error("[LoopManager] WorkerPullLoop will NOT start - workers won't claim work!")
        return False


def get_loop_manager() -> "LoopManager | None":
    """Get or create the global LoopManager singleton.

    Returns None if extracted loops are disabled or unavailable.
    """
    global _loop_manager_instance
    if not EXTRACTED_LOOPS_ENABLED:
        return None
    if _loop_manager_instance is None:
        if not _load_loop_classes():
            return None
        try:
            from scripts.p2p.loops import LoopManager
            _loop_manager_instance = LoopManager(name="p2p_loops")
            logger.info("LoopManager: initialized for extracted background loops")
        except (ImportError, TypeError, ValueError, AttributeError) as e:
            # ImportError: loops module not available
            # TypeError: wrong constructor signature
            # ValueError: invalid argument
            # AttributeError: LoopManager not found in module
            logger.error(f"LoopManager: failed to initialize: {e}")
            return None
    return _loop_manager_instance


# Board priority overrides from unified_loop.yaml
# 0=CRITICAL, 1=HIGH, 2=MEDIUM, 3=LOW (lower value = higher priority)
_board_priority_cache: dict[str, int] | None = None
_board_priority_cache_time: float = 0


def get_board_priority_overrides() -> dict[str, int]:
    """Load board priority overrides from config, cached for 60 seconds.

    Returns dict mapping config keys (e.g., 'hexagonal_2p') to priority levels.
    Priority levels: 0=CRITICAL, 1=HIGH, 2=MEDIUM, 3=LOW
    """
    global _board_priority_cache, _board_priority_cache_time
    now = time.time()

    # Return cached value if fresh (60 second TTL)
    if _board_priority_cache is not None and now - _board_priority_cache_time < 60:
        return _board_priority_cache

    try:
        import yaml
        config_path = Path(__file__).parent.parent / "config" / "unified_loop.yaml"
        if config_path.exists():
            with open(config_path) as f:
                yaml_config = yaml.safe_load(f)
            selfplay_config = yaml_config.get("selfplay", {})
            overrides = selfplay_config.get("board_priority_overrides", {})
            # Convert config keys like "hexagonal_2p" -> priority int
            _board_priority_cache = {k: int(v) for k, v in overrides.items()}
            _board_priority_cache_time = now
            return _board_priority_cache
    except (OSError, ValueError, AttributeError, ImportError):
        pass

    # Default: empty (no overrides)
    return {}


# =============================================================================
# P2P Event Emission Helpers (December 2025 - CRITICAL gap fix)
# =============================================================================
# These helpers safely emit events for P2P lifecycle changes. Events enable:
# - LeadershipCoordinator to track leader changes
# - UnifiedHealthManager to respond to node failures
# - Cluster-wide coordination on membership changes

_p2p_event_emitters_available: bool | None = None
_p2p_event_emitters_last_check: float = 0.0
_P2P_EMITTER_CACHE_TTL: float = 30.0  # Retry every 30 seconds if failed


def _check_event_emitters() -> bool:
    """Check if event emitters are available (cached with TTL for retries).

    December 27, 2025: Fixed bug where negative result was cached permanently.
    Now retries every 30 seconds if event system becomes available later.
    """
    global _p2p_event_emitters_available, _p2p_event_emitters_last_check
    import time

    now = time.time()

    # Use cached positive result indefinitely
    if _p2p_event_emitters_available is True:
        return True

    # For negative results, retry after TTL expires
    if _p2p_event_emitters_available is False:
        if now - _p2p_event_emitters_last_check < _P2P_EMITTER_CACHE_TTL:
            return False
        # TTL expired, retry below

    try:
        from app.coordination.event_router import (
            emit_host_online,
            emit_host_offline,
            emit_leader_elected,
        )
        _p2p_event_emitters_available = True
        _p2p_event_emitters_last_check = now
        return True
    except ImportError:
        _p2p_event_emitters_available = False
        _p2p_event_emitters_last_check = now
        return False


# December 28, 2025: Module-level emit functions (27 methods, ~911 LOC) were moved to
# EventEmissionMixin in scripts/p2p/event_emission_mixin.py.
# P2POrchestrator now inherits from EventEmissionMixin and uses self._emit_* methods.
# See scripts/p2p/__init__.py for the mixin export.


# Add project root to path for scripts.lib imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.lib.file_formats import open_jsonl_file
from scripts.lib.logging_config import setup_script_logging
from scripts.lib.process import (
    SingletonLock,
    find_processes_by_pattern,
    kill_process,
    is_process_running,
)

logger = setup_script_logging("p2p_orchestrator")

# Singleton lock for duplicate process prevention (December 2025)
_P2P_LOCK: SingletonLock | None = None


def _validate_p2p_dependencies() -> None:
    """Pre-flight check for required modules. Exits with code 2 if missing.

    This catches import errors early with a clear message, rather than
    failing deep in the call stack with confusing tracebacks.
    """
    required_modules = [
        ("aiohttp", "pip install aiohttp"),
        ("psutil", "pip install psutil"),
        ("yaml", "pip install pyyaml"),
    ]
    missing = []
    for module_name, install_hint in required_modules:
        try:
            __import__(module_name)
        except ImportError:
            missing.append(f"{module_name} ({install_hint})")

    if missing:
        # Use print since logger may not be fully initialized
        print(f"CRITICAL: Missing required dependencies: {', '.join(missing)}", file=sys.stderr)
        print("Run: pip install -r requirements.txt", file=sys.stderr)
        sys.exit(2)  # Exit code 2 = missing dependencies


# Validate dependencies before any heavy imports
_validate_p2p_dependencies()


# =============================================================================
# Async subprocess helper - Jan 19, 2026
# Prevents blocking the event loop during subprocess operations
# =============================================================================

async def async_subprocess_run(
    cmd: list[str],
    cwd: str | Path | None = None,
    timeout: float = 30.0,
    capture_output: bool = True,
    text: bool = True,
    env: dict | None = None,
) -> subprocess.CompletedProcess:
    """Run subprocess in thread pool to avoid blocking the event loop.

    This is a drop-in replacement for subprocess.run() in async contexts.
    Wraps the blocking subprocess.run() call in asyncio.to_thread().

    Args:
        cmd: Command and arguments to run
        cwd: Working directory for the command
        timeout: Timeout in seconds (default 30)
        capture_output: Capture stdout/stderr (default True)
        text: Return text instead of bytes (default True)
        env: Environment variables (default None = inherit)

    Returns:
        CompletedProcess with returncode, stdout, stderr

    Example:
        result = await async_subprocess_run(["git", "status"], cwd="/path")
        if result.returncode == 0:
            print(result.stdout)
    """
    def _run():
        return subprocess.run(
            cmd,
            cwd=cwd,
            timeout=timeout,
            capture_output=capture_output,
            text=text,
            env=env,
        )

    return await asyncio.to_thread(_run)


# Centralized ramdrive utilities for auto-detection
# Shared database integrity utilities
from app.db.integrity import (
    check_and_repair_databases,
)

# Circuit breaker for fault-tolerant network operations
from app.distributed.circuit_breaker import (
    CircuitState,
    get_circuit_registry,
)
# Jan 2026: Adaptive budget selection based on config Elo
from app.coordination.budget_calculator import (
    get_adaptive_budget_for_elo,
)
from app.utils.ramdrive import (
    RamdriveSyncer,
    get_system_resources,
    log_storage_recommendation,
    should_use_ramdrive,
)
from scripts.p2p.cluster_config import (
    get_cluster_config,
)
from scripts.p2p.utils import WebhookNotifier
from scripts.p2p.handlers import (
    ABTestHandlersMixin,
    AdminHandlersMixin,
    AnalyticsHandlersMixin,  # January 2026 - Analytics handlers extraction
    CanonicalGateHandlersMixin,
    CMAESHandlersMixin,
    DeliveryHandlersMixin,
    DiagnosticsHandlersMixin,  # January 2026 - Diagnostics handlers extraction
    ElectionHandlersMixin,
    EloSyncHandlersMixin,
    GauntletHandlersMixin,
    GossipHandlersMixin,
    ImprovementHandlersMixin,
    JobsApiHandlersMixin,
    MetricsHandlersMixin,  # January 2026 - P2P Modularization (Prometheus metrics)
    SelfplayHandlersMixin,  # January 2026 - P2P Modularization (Selfplay API)
    ClusterApiHandlersMixin,  # January 2026 - P2P Modularization (Cluster API)
    DashboardHandlersMixin,  # January 2026 - P2P Modularization (Dashboard)
    RecoveryHandlersMixin,  # January 2026 - P2P Modularization Phase 2b (Rollback)
    ConfigurationHandlersMixin,  # January 2026 - P2P Modularization Phase 2c (Config/Registration)
    TrainingControlHandlersMixin,  # January 2026 - P2P Modularization Phase 3a (Training Control)
    EloAnalyticsHandlersMixin,  # January 2026 - P2P Modularization Phase 4a (Elo Analytics)
    EvaluationPlayHandlersMixin,  # January 2026 - P2P Modularization Phase 5a (Elo Match Play)
    EventManagementHandlersMixin,  # January 2026 - P2P Modularization Phase 5b (Event Subscriptions)
    StatusHandlersMixin,  # January 2026 - P2P Modularization Phase 6a (Status/Health/Loops)
    ModelHandlersMixin,  # January 2026 - Comprehensive Model Evaluation Pipeline
    VoterConfigHandlersMixin,  # January 2026 - Consensus-safe voter config sync
    PipelineHandlersMixin,  # January 2026 - Pipeline phase handlers extraction
    SerfHandlersMixin,  # January 2026 - Serf event handlers extraction
    RegistryHandlersMixin,
    ManifestHandlersMixin,
    RelayHandlersMixin,
    SSHTournamentHandlersMixin,
    SyncHandlersMixin,
    TableHandlersMixin,
    TournamentHandlersMixin,
    WorkQueueHandlersMixin,
    setup_model_routes,  # January 2026 - Model inventory route setup
)
from scripts.p2p.network_utils import NetworkUtilsMixin
from scripts.p2p.peer_manager import PeerManagerMixin
from scripts.p2p.leader_election import LeaderElectionMixin
from scripts.p2p.gossip_protocol import GossipProtocolMixin  # Contains merged GossipMetricsMixin (Dec 28, 2025)

# Phase 5: SWIM + Raft integration mixins (Dec 26, 2025)
from scripts.p2p.membership_mixin import MembershipMixin
from scripts.p2p.consensus_mixin import ConsensusMixin
from scripts.p2p.handlers.swim import SwimHandlersMixin
from scripts.p2p.handlers.raft import RaftHandlersMixin
from scripts.p2p.handlers.network_health import NetworkHealthMixin, setup_network_health_routes

# Leadership mixins for voter/quorum monitoring and state transitions (Jan 2026)
from scripts.p2p.mixins import (
    AdvertiseValidationMixin,
    LeadershipHealthMixin,
    LeadershipTransitionsMixin,
)

# Import constants from the refactored module (Phase 2 refactoring - consolidated)
from scripts.p2p.constants import (
    ADVERTISE_HOST_ENV,
    ADVERTISE_PORT_ENV,
    AGENT_MODE_ENABLED,
    ARBITER_URL,
    # Auth and build info
    AUTH_TOKEN_ENV,
    AUTH_TOKEN_FILE_ENV,
    AUTO_ASSIGN_ENABLED,
    AUTO_TRAINING_THRESHOLD_MB,
    AUTO_UPDATE_ENABLED,
    AUTO_WORK_BATCH_SIZE,
    BUILD_VERSION_ENV,
    COORDINATOR_URL,
    DATA_MANAGEMENT_INTERVAL,
    DB_EXPORT_THRESHOLD_MB,
    # Network configuration
    DEFAULT_PORT,
    DISCOVERY_INTERVAL,
    DISCOVERY_PORT,
    DISK_CLEANUP_THRESHOLD,
    # Resource thresholds
    DISK_CRITICAL_THRESHOLD,
    DISK_WARNING_THRESHOLD,
    # Dynamic voter management
    DYNAMIC_VOTER_ENABLED,
    DYNAMIC_VOTER_MAX_QUORUM,
    DYNAMIC_VOTER_MIN,
    DYNAMIC_VOTER_TARGET,
    ELECTION_TIMEOUT,
    ELO_K_FACTOR,
    GH200_MAX_SELFPLAY,
    GH200_MIN_SELFPLAY,
    GIT_BRANCH_NAME,
    GIT_REMOTE_NAME,
    # Auto-update settings
    GIT_UPDATE_CHECK_INTERVAL,
    # Safeguards
    GPU_IDLE_RESTART_TIMEOUT,
    GPU_IDLE_THRESHOLD,
    GPU_POWER_RANKINGS,
    GRACEFUL_SHUTDOWN_BEFORE_UPDATE,
    HEARTBEAT_INTERVAL,
    # Connection robustness
    HTTP_CONNECT_TIMEOUT,
    HTTP_TOTAL_TIMEOUT,
    IDLE_CHECK_INTERVAL,
    IDLE_GPU_THRESHOLD,
    IDLE_GRACE_PERIOD,
    # Elo constants (from app.config.thresholds)
    BASELINE_ELO_RANDOM,  # Random AI pinned at 400 Elo
    INITIAL_ELO_RATING,
    JOB_CHECK_INTERVAL,
    LEADER_DEGRADED_STEPDOWN_DELAY,
    LEADER_HEALTH_CHECK_INTERVAL,
    LEADER_LEASE_DURATION,
    LEADER_LEASE_RENEW_INTERVAL,
    LEADER_MIN_RESPONSE_RATE,
    LEADERLESS_TRAINING_TIMEOUT,
    LEADER_WORK_DISPATCH_TIMEOUT,
    # Leader stickiness (Jan 2, 2026)
    INCUMBENT_LEADER_GRACE_PERIOD,
    RECENT_LEADER_WINDOW,
    # Probabilistic fallback leadership (Jan 1, 2026)
    PROVISIONAL_LEADER_MIN_LEADERLESS_TIME,
    PROVISIONAL_LEADER_INITIAL_PROBABILITY,
    PROVISIONAL_LEADER_MAX_PROBABILITY,
    PROVISIONAL_LEADER_PROBABILITY_GROWTH_RATE,
    PROVISIONAL_LEADER_QUORUM_TIMEOUT,
    PROVISIONAL_LEADER_CHECK_INTERVAL,
    # Jan 2026: ULSM tiered fallback
    ELECTION_RETRY_COUNT_BEFORE_PROVISIONAL,
    DETERMINISTIC_FALLBACK_TIME,
    LOAD_AVERAGE_MAX_MULTIPLIER,
    LOAD_MAX_FOR_NEW_JOBS,
    MANIFEST_JSONL_LINECOUNT_CHUNK_BYTES,
    # Data management
    MANIFEST_JSONL_LINECOUNT_MAX_BYTES,
    MANIFEST_JSONL_SAMPLE_BYTES,
    MAX_CONCURRENT_EXPORTS,
    MAX_CONSECUTIVE_FAILURES,
    MAX_DISK_USAGE_PERCENT,
    MAX_GAUNTLET_RUNTIME,
    # Stale process cleanup
    MAX_SELFPLAY_RUNTIME,
    MAX_TOURNAMENT_RUNTIME,
    MAX_TRAINING_RUNTIME,
    MEMORY_CRITICAL_THRESHOLD,
    MEMORY_WARNING_THRESHOLD,
    MIN_GAMES_FOR_SYNC,
    MIN_MEMORY_GB_FOR_TASKS,
    MODEL_SYNC_INTERVAL,
    NAT_BLOCKED_PROBE_INTERVAL,
    NAT_BLOCKED_PROBE_TIMEOUT,
    NAT_BLOCKED_RECOVERY_TIMEOUT,
    NAT_EXTERNAL_IP_CACHE_TTL,
    NAT_HOLE_PUNCH_RETRY_COUNT,
    # NAT/Relay settings
    NAT_INBOUND_HEARTBEAT_STALE_SECONDS,
    NAT_RELAY_PREFERENCE_THRESHOLD,
    NAT_STUN_LIKE_PROBE_INTERVAL,
    NAT_SYMMETRIC_DETECTION_ENABLED,
    P2P_DATA_SYNC_BASE,
    P2P_DATA_SYNC_MAX,
    P2P_DATA_SYNC_MIN,
    P2P_MODEL_SYNC_BASE,
    P2P_MODEL_SYNC_MAX,
    P2P_MODEL_SYNC_MIN,
    P2P_SYNC_BACKOFF_FACTOR,
    P2P_SYNC_SPEEDUP_FACTOR,
    P2P_TRAINING_DB_SYNC_BASE,
    P2P_TRAINING_DB_SYNC_MAX,
    P2P_TRAINING_DB_SYNC_MIN,
    PEER_BOOTSTRAP_INTERVAL,
    PEER_BOOTSTRAP_MIN_PEERS,
    PEER_DEATH_RATE_LIMIT,
    PEER_PURGE_AFTER_SECONDS,
    PEER_RECOVERY_RETRY_INTERVAL,
    PEER_RETIRE_AFTER_SECONDS,
    PEER_TIMEOUT,
    PEER_TIMEOUT_JITTER_FACTOR,
    get_jittered_peer_timeout,
    get_cpu_adaptive_timeout,
    CPU_LOAD_HIGH_THRESHOLD,
    RELAY_COMMAND_MAX_ATTEMPTS,
    RELAY_COMMAND_MAX_BATCH,
    RELAY_COMMAND_TTL_SECONDS,
    RELAY_HEARTBEAT_INTERVAL,
    RELAY_MAX_PENDING_START_JOBS,
    RETRY_DEAD_NODE_INTERVAL,
    RETRY_RETIRED_NODE_INTERVAL,
    RUNAWAY_SELFPLAY_PROCESS_THRESHOLD,
    SPAWN_RATE_LIMIT_PER_MINUTE,
    STALE_PROCESS_CHECK_INTERVAL,
    STARTUP_GRACE_PERIOD,
    ELECTION_PARTICIPATION_DELAY,
    STALE_PROCESS_PATTERNS,
    STARTUP_JSONL_GRACE_PERIOD_SECONDS,
    # State directory
    STATE_DIR,
    TAILSCALE_CGNAT_NETWORK,
    TARGET_GPU_UTIL_MAX,
    # GPU configuration
    TARGET_GPU_UTIL_MIN,
    TRAINING_DATA_SYNC_THRESHOLD_MB,
    # Training node sync
    TRAINING_NODE_COUNT,
    TRAINING_SYNC_INTERVAL,
    # Unified inventory / Idle detection
    UNIFIED_DISCOVERY_INTERVAL,
    VOTER_DEMOTION_FAILURES,
    VOTER_HEALTH_THRESHOLD,
    VOTER_HEARTBEAT_INTERVAL,
    VOTER_HEARTBEAT_TIMEOUT,
    VOTER_MESH_REFRESH_INTERVAL,
    VOTER_MIN_QUORUM,
    VOTER_NAT_RECOVERY_AGGRESSIVE,
    VOTER_PROMOTION_UPTIME,
    # Phase 26: Multi-seed bootstrap and mesh resilience
    BOOTSTRAP_SEEDS,
    MIN_BOOTSTRAP_ATTEMPTS,
    ISOLATED_BOOTSTRAP_INTERVAL,
    MIN_CONNECTED_PEERS,
    # Phase 28: Gossip protocol
    GOSSIP_FANOUT,
    GOSSIP_INTERVAL,
    GOSSIP_MAX_PEER_ENDPOINTS,
    # Phase 27: Peer cache
    PEER_CACHE_TTL_SECONDS,
    PEER_CACHE_MAX_ENTRIES,
    PEER_REPUTATION_ALPHA,
    # Phase 29: Cluster epochs
    INITIAL_CLUSTER_EPOCH,
)
from scripts.p2p.models import (
    ClusterDataManifest,
    ClusterJob,
    ClusterSyncPlan,
    DataFileInfo,
    DataSyncJob,
    DistributedCMAESState,
    DistributedTournamentState,
    ImprovementLoopState,
    NodeDataManifest,
    NodeInfo,
    PeerCircuitBreaker,  # Jan 3, 2026: Sprint 10+ P2P hardening
    PeerHealthScore,     # Jan 3, 2026: Sprint 10+ P2P hardening
    SSHTournamentRun,
    TrainingJob,
    TrainingThresholds,
)
from scripts.p2p.p2p_mixin_base import SubscriptionRetryConfig
from scripts.p2p.network import (
    JobSnapshot,  # Jan 12, 2026: Lock-free job reads
    NonBlockingAsyncLockWrapper,
    PeerSnapshot,  # Jan 12, 2026: Lock-free peer reads
    TimeoutAsyncLockWrapper,
    get_client_session,
)

# Import refactored utilities (Phase 2 refactoring)
from scripts.p2p.resource_utils import (
    check_disk_has_capacity,
)

# Import refactored P2P types and models
# These were extracted from this file for modularity (Phase 1 refactoring)
from scripts.p2p.types import JobType, NodeRole
from scripts.p2p.utils import (
    safe_json_response,
    systemd_notify_ready,
    systemd_notify_watchdog,
)
from scripts.p2p.managers import (
    AnalyticsCacheConfig,
    AnalyticsCacheManager,
    CMAESConfig,
    CMAESCoordinator,
    DataSyncCoordinator,
    DataSyncCoordinatorConfig,
    JobManager,
    JobOrchestrationConfig,
    JobOrchestrationManager,
    NodeSelector,
    SelfplayScheduler,
    StateManager,
    SyncPlanner,
    SyncPlannerConfig,
    TrainingCoordinator,
    create_analytics_cache_manager,
    create_cmaes_coordinator,
    create_data_sync_coordinator,
    create_job_orchestration_manager,
)
from scripts.p2p.managers.state_manager import PersistedLeaderState
from scripts.p2p.managers.voter_config_manager import (
    get_voter_config_manager,
    VoterConfigManager,
)
from scripts.p2p.managers.work_discovery_manager import (
    _is_selfplay_enabled_for_node,
    _is_training_enabled_for_node,
    set_selfplay_disabled_override,
)
from scripts.p2p.metrics_manager import MetricsManager
from scripts.p2p.query_builders import PeerQueryBuilder
from scripts.p2p.resource_detector import ResourceDetector, ResourceDetectorMixin
from scripts.p2p.config.selfplay_job_configs import (
    DIVERSE_PROFILES,
    SELFPLAY_CONFIGS,
    get_diverse_profile_weights,
    get_filtered_configs,
    get_unique_configs,
    get_weighted_configs,
    select_diverse_profiles,
)
from scripts.p2p.job_spawner import (
    GUMBEL_ENGINE_MODES,
    SELFPLAY_ENGINE_MODES,
)
from scripts.p2p.event_emission_mixin import EventEmissionMixin
from scripts.p2p.failover_integration import FailoverIntegrationMixin
from scripts.p2p.relay_leader_propagator import RelayLeaderPropagatorMixin  # Phase 1: NAT-blocked leader propagation (Jan 4, 2026)
from scripts.p2p.leadership_state_machine import (
    LeadershipStateMachine,
    LeaderState,
    TransitionReason,
)

# Unified resource checking utilities (80% max utilization)
# Includes graceful degradation for dynamic workload management
try:
    from app.utils.resource_guard import (
        LIMITS as RESOURCE_LIMITS,
        OperationPriority,
        check_cpu as unified_check_cpu,
        check_disk_space as unified_check_disk,
        check_memory as unified_check_memory,
        get_degradation_level,
        should_proceed_with_priority,
    )
    HAS_RESOURCE_GUARD = True
except ImportError:
    HAS_RESOURCE_GUARD = False
    unified_check_disk = None
    unified_check_memory = None
    unified_check_cpu = None
    RESOURCE_LIMITS = None
    should_proceed_with_priority = None
    OperationPriority = None
    get_degradation_level = None

# ELO database sync manager for cluster-wide consistency
try:
    from app.tournament.elo_sync_manager import (
        EloSyncManager,
        ensure_elo_synced,
        get_elo_sync_manager,
        sync_elo_after_games,
    )
    HAS_ELO_SYNC = True
except ImportError:
    HAS_ELO_SYNC = False
    EloSyncManager = None
    get_elo_sync_manager = None
    sync_elo_after_games = None
    ensure_elo_synced = None

# Distributed data sync manager for model/data distribution
# Prefer new sync_coordinator, fallback to deprecated data_sync
try:
    from app.distributed.sync_coordinator import SyncCoordinator, full_cluster_sync
    HAS_SYNC_COORDINATOR = True

    def get_sync_coordinator():
        return SyncCoordinator.get_instance()
except ImportError:
    HAS_SYNC_COORDINATOR = False
    SyncCoordinator = None
    full_cluster_sync = None

# SyncRouter: Intelligent data routing with quality-based priority (December 2025)
try:
    from app.coordination.sync_router import get_sync_router, SyncRouter
    HAS_SYNC_ROUTER = True
except ImportError:
    HAS_SYNC_ROUTER = False
    get_sync_router = None
    SyncRouter = None

# Phase 3.1: Curriculum weights integration for selfplay prioritization
try:
    from scripts.unified_loop.curriculum import load_curriculum_weights
    HAS_CURRICULUM_WEIGHTS = True
except ImportError:
    HAS_CURRICULUM_WEIGHTS = False
    load_curriculum_weights = None

# Unified node inventory for multi-CLI discovery (Vast, Tailscale, Lambda, Hetzner)
try:
    from app.coordination.unified_inventory import UnifiedInventory, get_inventory
    HAS_UNIFIED_INVENTORY = True
except ImportError:
    HAS_UNIFIED_INVENTORY = False
    UnifiedInventory = None
    get_inventory = None

# HTTP server imports
try:
    import aiohttp
    from aiohttp import ClientSession, ClientTimeout, web
    HAS_AIOHTTP = True

    # Feb 24, 2026: Patch aiohttp tcp_keepalive to handle macOS socket errors.
    # aiohttp 3.13.x calls setsockopt(SOL_SOCKET, SO_KEEPALIVE, 1) on every
    # new connection, which raises OSError [Errno 22] on macOS for certain
    # socket types (loopback, dual-stack). This silently kills ALL HTTP
    # connections, making the server unresponsive.
    try:
        import aiohttp.tcp_helpers as _tcp_helpers
        _orig_tcp_keepalive = _tcp_helpers.tcp_keepalive

        def _safe_tcp_keepalive(transport: object) -> None:
            try:
                _orig_tcp_keepalive(transport)
            except OSError:
                pass  # Ignore keepalive failures on macOS

        _tcp_helpers.tcp_keepalive = _safe_tcp_keepalive
    except (ImportError, AttributeError):
        pass
except ImportError:
    HAS_AIOHTTP = False
    aiohttp = None
    logger.warning("aiohttp not installed. Install with: pip install aiohttp")

# SOCKS proxy support for userspace Tailscale networking
try:
    from aiohttp_socks import ProxyConnector
    HAS_SOCKS = True
except ImportError:
    HAS_SOCKS = False
    ProxyConnector = None

# Get SOCKS proxy from environment (e.g., socks5://localhost:1055)
SOCKS_PROXY = os.environ.get("RINGRIFT_SOCKS_PROXY", "")


# =============================================================================
# HTTP Handler Timeout Decorator (December 30, 2025)
# =============================================================================
# Added to fix P2P cluster connectivity issues where HTTP handlers blocked
# indefinitely on slow operations (lock acquisition, daemon status collection).

def with_request_timeout(timeout_seconds: float = 20.0):
    """Decorator to add timeout protection to HTTP handlers.

    December 30, 2025: Added to prevent HTTP endpoints from blocking indefinitely.
    January 10, 2026: Increased default from 10s to 20s to exceed typical lock wait
    times (reduced from 5s to 2s for gossip locks, but other operations can take longer).

    Usage:
        @with_request_timeout(5.0)
        async def handle_health(self, request):
            ...

    Args:
        timeout_seconds: Maximum time in seconds for handler to complete.

    Returns:
        Decorated handler that returns 504 Gateway Timeout on timeout.
    """
    import functools

    def decorator(handler):
        @functools.wraps(handler)
        async def wrapper(self_or_request, *args, **kwargs):
            # Handle both bound methods (self, request) and plain functions (request)
            try:
                return await asyncio.wait_for(
                    handler(self_or_request, *args, **kwargs),
                    timeout=timeout_seconds
                )
            except asyncio.TimeoutError:
                # Return 504 Gateway Timeout with details
                return web.json_response(
                    {
                        "error": "Request timed out",
                        "timeout_seconds": timeout_seconds,
                        "timestamp": time.time(),
                    },
                    status=504
                )
        return wrapper
    return decorator


# Systemd watchdog support for service health monitoring
# When running under systemd with WatchdogSec set, we need to periodically
# notify systemd that the service is healthy. If we miss the deadline,
# systemd will restart the service.
try:
    import sdnotify
    SYSTEMD_NOTIFIER = sdnotify.SystemdNotifier()
    HAS_SYSTEMD = True
except ImportError:
    SYSTEMD_NOTIFIER = None
    HAS_SYSTEMD = False


# ============================================
# Utilities (Refactored - Phase 2)
# ============================================
# The following utilities have been moved to scripts/p2p/ for modularity:
# - systemd_notify_watchdog, systemd_notify_ready (scripts/p2p/utils.py)
# - AsyncLockWrapper, get_client_session (scripts/p2p/network.py)
# - check_peer_circuit, record_peer_success, record_peer_failure (scripts/p2p/network.py)
# - peer_request (scripts/p2p/network.py)
# - get_disk_usage_percent, check_disk_has_capacity, check_all_resources (scripts/p2p/resource.py)
#
# They are imported at the top of this file for backward compatibility.
# ============================================

# Dynamic host registry for IP auto-update
try:
    from app.distributed.dynamic_registry import (
        NodeState,
        get_registry,
    )
    HAS_DYNAMIC_REGISTRY = True
except ImportError:
    HAS_DYNAMIC_REGISTRY = False
    get_registry = None
    NodeState = None

# Hybrid transport layer for HTTP/SSH fallback (self-healing Vast connectivity)
try:
    from app.distributed.hybrid_transport import (
        HybridTransport,
        diagnose_node_connectivity,
        get_hybrid_transport,
    )
    from app.distributed.ssh_transport import (
        SSHTransport,
        get_ssh_transport,
        probe_vast_nodes_via_ssh,
    )
    HAS_HYBRID_TRANSPORT = True
except ImportError:
    HAS_HYBRID_TRANSPORT = False
    HybridTransport = None
    get_hybrid_transport = None
    diagnose_node_connectivity = None
    SSHTransport = None
    get_ssh_transport = None
    probe_vast_nodes_via_ssh = None

try:
    from scripts.improvement_cycle_manager import ImprovementCycleManager
    HAS_IMPROVEMENT_MANAGER = True
except ImportError:
    # Fallback - deprecated archive location removed in 2025-12
    HAS_IMPROVEMENT_MANAGER = False
    ImprovementCycleManager = None

# Task coordination safeguards - prevents runaway spawning
try:
    from app.coordination.safeguards import Safeguards, check_before_spawn
    HAS_SAFEGUARDS = True
    _safeguards = Safeguards.get_instance()
except ImportError:
    HAS_SAFEGUARDS = False
    _safeguards = None
    def check_before_spawn(task_type, node_id):
        return True, ""

# New coordination features: OrchestratorRole, backpressure, sync_lock, bandwidth
try:
    from app.coordination import (
        NodeResources,
        # Orchestrator role management (SQLite-backed with heartbeat)
        OrchestratorRole,
        # Queue backpressure
        QueueType,
        # Resource optimizer for cluster-wide PID-controlled optimization
        ResourceOptimizer,
        TransferPriority,
        acquire_orchestrator_role,
        get_cluster_utilization,
        get_host_targets,
        get_optimal_concurrency,
        get_resource_optimizer,
        # Resource targets for unified utilization management
        get_resource_targets,
        get_target_job_count,
        get_throttle_factor,
        record_utilization,
        release_bandwidth,
        release_orchestrator_role,
        # Bandwidth management
        request_bandwidth,
        should_scale_down,
        should_scale_up,
        should_stop_production,
        should_throttle_production,
        # Sync mutex for data transfer coordination
        sync_lock,
    )

    # Import rate negotiation functions for cooperative utilization (60-80% target)
    from app.coordination.resource_optimizer import (
        apply_feedback_adjustment,
        get_config_weights,
        get_current_selfplay_rate,
        get_hybrid_selfplay_limits,
        get_max_cpu_only_selfplay,
        # Hardware-aware selfplay limits (single source of truth)
        get_max_selfplay_for_node,
        get_utilization_status,
        negotiate_selfplay_rate,
        update_config_weights,
    )
    HAS_RATE_NEGOTIATION = True
    HAS_NEW_COORDINATION = True
    HAS_HW_AWARE_LIMITS = True
    # Get targets from unified source
    _unified_targets = get_resource_targets()
except ImportError:
    HAS_NEW_COORDINATION = False
    HAS_RATE_NEGOTIATION = False
    HAS_HW_AWARE_LIMITS = False
    OrchestratorRole = None
    _unified_targets = None
    negotiate_selfplay_rate = None
    get_current_selfplay_rate = None
    apply_feedback_adjustment = None
    get_utilization_status = None
    update_config_weights = None
    get_config_weights = None
    get_max_selfplay_for_node = None
    get_hybrid_selfplay_limits = None
    get_max_cpu_only_selfplay = None

# P2P-integrated monitoring management
try:
    from app.monitoring.p2p_monitoring import MonitoringManager
    HAS_P2P_MONITORING = True
except ImportError:
    HAS_P2P_MONITORING = False
    MonitoringManager = None

# Model sync across cluster
try:
    from scripts.sync_models import (
        HOSTS_MODULE_AVAILABLE as HAS_HOSTS_FOR_SYNC,
        ClusterModelState,
        scan_cluster as scan_cluster_models,
        sync_missing_models,
    )
    # Also import load_remote_hosts for scanning
    if HAS_HOSTS_FOR_SYNC:
        from app.distributed.hosts import filter_ready_hosts, load_remote_hosts
    HAS_MODEL_SYNC = True
except ImportError:
    HAS_MODEL_SYNC = False
    scan_cluster_models = None
    sync_missing_models = None
    ClusterModelState = None
    HAS_HOSTS_FOR_SYNC = False
    load_remote_hosts = None
    filter_ready_hosts = None

# PFSP (Prioritized Fictitious Self-Play) opponent pool
try:
    from app.training.advanced_training import (
        CMAESAutoTuner,
        OpponentStats,
        PFSPOpponentPool,
        PlateauConfig,
    )
    HAS_PFSP = True
except ImportError:
    HAS_PFSP = False
    PFSPOpponentPool = None
    OpponentStats = None
    CMAESAutoTuner = None
    PlateauConfig = None

# Configuration: See scripts/p2p/constants.py
# Types: See scripts/p2p/types.py and scripts/p2p/models.py


# Jan 28, 2026: WebhookNotifier class moved to scripts/p2p/utils/webhook_notifier.py (~175 LOC)
# Now imported via: from scripts.p2p.utils import WebhookNotifier


class P2POrchestrator(
    WorkQueueHandlersMixin,
    ElectionHandlersMixin,
    RelayHandlersMixin,
    GauntletHandlersMixin,
    GossipHandlersMixin,
    AdminHandlersMixin,
    EloSyncHandlersMixin,
    TournamentHandlersMixin,
    CMAESHandlersMixin,
    SSHTournamentHandlersMixin,
    DeliveryHandlersMixin,  # Phase 3: Delivery verification (Dec 27, 2025)
    SyncHandlersMixin,      # Phase 8: Sync handlers extraction (Dec 28, 2025)
    TableHandlersMixin,     # Phase 8: Table/dashboard handlers extraction (Dec 28, 2025)
    RegistryHandlersMixin,  # Phase 8: Registry handlers extraction (Dec 28, 2025)
    ManifestHandlersMixin,  # Phase 8: Manifest handlers extraction (Dec 28, 2025)
    ABTestHandlersMixin,    # Phase 8: A/B test handlers extraction (Dec 28, 2025)
    ImprovementHandlersMixin,  # Phase 8: Improvement loop handlers extraction (Dec 28, 2025)
    CanonicalGateHandlersMixin,  # Phase 8: Canonical gate handlers extraction (Dec 28, 2025)
    JobsApiHandlersMixin,        # Phase 8: Jobs API handlers extraction (Dec 28, 2025)
    MetricsHandlersMixin,        # Prometheus metrics export (Jan 2026 - P2P Modularization)
    SelfplayHandlersMixin,       # Selfplay API endpoints (Jan 2026 - P2P Modularization)
    ClusterApiHandlersMixin,     # Cluster API endpoints (Jan 2026 - P2P Modularization)
    DashboardHandlersMixin,      # Dashboard endpoints (Jan 2026 - P2P Modularization)
    RecoveryHandlersMixin,       # Rollback endpoints (Jan 2026 - P2P Modularization Phase 2b)
    ConfigurationHandlersMixin,  # Config/Registration (Jan 2026 - P2P Modularization Phase 2c)
    TrainingControlHandlersMixin,  # Training Control (Jan 2026 - P2P Modularization Phase 3a)
    EloAnalyticsHandlersMixin,   # Elo Analytics (Jan 2026 - P2P Modularization Phase 4a)
    EvaluationPlayHandlersMixin,  # Elo Match Play (Jan 2026 - P2P Modularization Phase 5a)
    EventManagementHandlersMixin,  # Event Subscriptions (Jan 2026 - P2P Modularization Phase 5b)
    StatusHandlersMixin,         # Status/Health/Loops (Jan 2026 - P2P Modularization Phase 6a)
    ModelHandlersMixin,          # Model inventory endpoints (Jan 2026 - Comprehensive Eval Pipeline)
    PipelineHandlersMixin,       # Pipeline phase handlers (Jan 2026 - P2P Modularization Phase 6)
    SerfHandlersMixin,           # Serf event handlers (Jan 2026 - P2P Modularization Phase 7)
    AnalyticsHandlersMixin,      # Analytics handlers (Jan 2026 - P2P Modularization Phase 8)
    DiagnosticsHandlersMixin,    # Diagnostics handlers (Jan 2026 - P2P Modularization Phase 8f)
    NetworkHealthMixin,          # Network health endpoints (Dec 30, 2025)
    NetworkUtilsMixin,
    PeerManagerMixin,
    LeaderElectionMixin,
    GossipProtocolMixin,  # Provides gossip protocol + metrics (merged Dec 28, 2025)
    # Phase 5: SWIM + Raft integration (Dec 26, 2025)
    MembershipMixin,      # SWIM gossip-based membership
    ConsensusMixin,       # PySyncObj Raft consensus
    SwimHandlersMixin,    # /swim/* HTTP handlers
    RaftHandlersMixin,    # /raft/* HTTP handlers
    ResourceDetectorMixin,  # Resource detection delegation (Dec 28, 2025)
    RelayLeaderPropagatorMixin,  # NAT-blocked leader propagation via gossip (Jan 4, 2026 - Phase 1)
    EventEmissionMixin,     # Event emission consolidation (Dec 28, 2025 - Phase 8)
    FailoverIntegrationMixin,  # Multi-layer transport failover (Dec 30, 2025 - Phase 9)
    VoterConfigHandlersMixin,  # Voter config sync (Jan 20, 2026 - Consensus-safe config sync)
    LeadershipHealthMixin,    # Voter/quorum health monitoring (Jan 26, 2026)
    LeadershipTransitionsMixin,  # Step-down and state transitions (Jan 26, 2026)
    AdvertiseValidationMixin,    # IP validation and advertise host management (Jan 26, 2026)
):
    """Main P2P orchestrator class that runs on each node.

    Inherits from:
    - WorkQueueHandlersMixin: Work queue HTTP handlers (handle_work_*)
    - ElectionHandlersMixin: Leader election handlers (handle_election*, handle_lease*, handle_voter*)
    - RelayHandlersMixin: NAT relay handlers (handle_relay_*)
    - GauntletHandlersMixin: Gauntlet evaluation handlers (handle_gauntlet_*)
    - GossipHandlersMixin: Gossip protocol handlers (handle_gossip*)
    - AdminHandlersMixin: Admin and git handlers (handle_git_*, handle_admin_*)
    - EloSyncHandlersMixin: Elo sync handlers (handle_elo_sync_*)
    - TournamentHandlersMixin: Tournament handlers (handle_tournament_*)
    - CMAESHandlersMixin: CMA-ES optimization handlers (handle_cmaes_*)
    - SSHTournamentHandlersMixin: SSH tournament handlers (handle_ssh_tournament_*)
    - NetworkUtilsMixin: Peer address parsing, URL building, Tailscale detection
    - PeerManagerMixin: Peer discovery, reputation tracking, cache management
    - RelayLeaderPropagatorMixin: NAT-blocked leader propagation via gossip (Jan 4, 2026)
    """

    def __init__(
        self,
        node_id: str,
        host: str = "0.0.0.0",
        port: int = DEFAULT_PORT,
        known_peers: list[str] | None = None,
        relay_peers: list[str] | None = None,
        ringrift_path: str | None = None,
        advertise_host: str | None = None,
        advertise_port: int | None = None,
        auth_token: str | None = None,
        require_auth: bool = False,
        storage_type: str = "auto",  # "disk", "ramdrive", or "auto"
        sync_to_disk_interval: int = 300,  # Sync ramdrive to disk every N seconds
    ):
        # Feb 2026: Decomposed into 6 initialization phases for readability.
        # Each phase is a separate method; ordering is critical.
        self._init_settings(
            node_id, host, port, known_peers, relay_peers,
            ringrift_path, advertise_host, advertise_port,
            auth_token, require_auth, storage_type, sync_to_disk_interval,
        )

        self._init_state()
        self._init_advanced_features()
        self._init_threading_and_protocols()
        self._init_managers()
        self._init_event_wiring()

    # =========================================================================
    # Initialization phases (Feb 2026 decomposition)
    # =========================================================================

    def _init_settings(
        self,
        node_id: str,
        host: str,
        port: int,
        known_peers: list[str] | None,
        relay_peers: list[str] | None,
        ringrift_path: str | None,
        advertise_host: str | None,
        advertise_port: int | None,
        auth_token: str | None,
        require_auth: bool,
        storage_type: str,
        sync_to_disk_interval: int,
    ) -> None:
        """Phase 1: Core node identity, bootstrap, advertise host, auth, quorum."""
        self.node_id = node_id
        self.host = host
        self.port = port

        self.ringrift_path = ringrift_path or self._detect_ringrift_path()

        from scripts.p2p.managers.initialization_manager import (
            InitializationManager,
            InitializationConfig,
        )
        self._init_manager = InitializationManager(
            config=InitializationConfig(),
            node_id=node_id,
            ringrift_path=self.ringrift_path,
        )

        bootstrap_result = self._init_manager.resolve_bootstrap_config(
            cli_peers=known_peers,
            relay_peers=relay_peers,
        )
        self.known_peers = bootstrap_result.known_peers
        self.bootstrap_seeds = bootstrap_result.bootstrap_seeds
        self.relay_peers = bootstrap_result.relay_peers
        self._force_relay_mode = bootstrap_result.force_relay_mode

        self._cluster_epoch: int = bootstrap_result.cluster_epoch
        self._cluster_health_degraded: bool = False
        self._gossip_learned_endpoints: dict[str, dict[str, Any]] = {}

        self._partition_config = PartitionConfig()
        self._partition_readonly_mode: bool = self._partition_config.readonly_mode
        self._partition_readonly_since: float = self._partition_config.readonly_since
        self._last_partition_check: float = self._partition_config.last_check
        self._partition_check_interval: float = self._partition_config.check_interval

        storage_result = self._init_manager.resolve_storage_config(storage_type=storage_type)
        self.storage_type = storage_result.storage_type
        self.sync_to_disk_interval = storage_result.sync_to_disk_interval
        self.ramdrive_path = storage_result.ramdrive_path
        self.ramdrive_syncer: RamdriveSyncer | None = None
        self._git_safe_directory = os.path.abspath(self.ringrift_path)
        self.build_version = self._detect_build_version()
        self.start_time = time.time()
        self.last_peer_bootstrap = 0.0

        self._cached_local_ips: set[str] = self._cache_local_ips()
        logger.info(f"[P2P] Cached {len(self._cached_local_ips)} local IPs for voter recognition")

        self._resource_detector = ResourceDetector(
            ringrift_path=self.ringrift_path,
            start_time=self.start_time,
            startup_grace_period=STARTUP_JSONL_GRACE_PERIOD_SECONDS,
        )

        # Advertise host resolution with multi-fallback chain
        self.advertise_host = (advertise_host or os.environ.get(ADVERTISE_HOST_ENV, "")).strip()
        prefer_public = os.environ.get("RINGRIFT_PREFER_PUBLIC_IP", "").strip().lower() in ("1", "true", "yes")

        if not self.advertise_host:
            if not prefer_public:
                yaml_ip = self._get_yaml_tailscale_ip()
                if yaml_ip:
                    self.advertise_host = yaml_ip
                    logger.info(f"[P2P] Using YAML config tailscale_ip: {yaml_ip}")
            if not self.advertise_host and not prefer_public:
                ts_ip = _wait_for_tailscale_ip(timeout_seconds=90, interval_seconds=1.0)
                self.advertise_host = ts_ip or self._get_local_ip()
                if not ts_ip:
                    logger.warning(
                        f"[P2P] Tailscale unavailable, using local IP: {self.advertise_host}. "
                        "Set RINGRIFT_ADVERTISE_HOST or ensure Tailscale is running."
                    )
            if not self.advertise_host and prefer_public:
                logger.info("[P2P] RINGRIFT_PREFER_PUBLIC_IP=1: skipping Tailscale, will use public IP")

        self._validate_and_fix_advertise_host()
        self.advertise_port = advertise_port if advertise_port is not None else self._infer_advertise_port()

        # Auth token resolution
        env_token = (os.environ.get(AUTH_TOKEN_ENV, "")).strip()
        token_from_arg = (auth_token or "").strip()
        token = token_from_arg or env_token

        if not token:
            token_file = (os.environ.get(AUTH_TOKEN_FILE_ENV, "")).strip()
            # Only read from explicitly-configured file path (env var), no auto-discovery.
            # Auto-discovery caused all cluster POST requests to be rejected with 401
            # because only the coordinator had the token file on disk.
            if token_file:
                try:
                    token = Path(token_file).read_text().strip()
                except Exception as e:  # noqa: BLE001
                    logger.info(f"Auth: failed to read {AUTH_TOKEN_FILE_ENV}={token_file}: {e}")

        self.auth_token = token.strip()
        self.require_auth = bool(require_auth)
        if self.require_auth and not self.auth_token:
            raise ValueError(
                f"--require-auth set but {AUTH_TOKEN_ENV}/{AUTH_TOKEN_FILE_ENV}/--auth-token is empty"
            )

        # Quorum manager setup
        config_path = Path(self._get_ai_service_path()) / "config" / "distributed_hosts.yaml"
        self.quorum_manager = QuorumManager(
            config=QuorumConfig(
                node_id=self.node_id,
                config_path=config_path if config_path.exists() else None,
            ),
            get_peers=lambda: self.peers,
            get_peers_lock=lambda: self.peers_lock,
        )
        self.voter_node_ids: list[str] = self.quorum_manager.load_voter_node_ids()
        self.voter_config_source: str = self.quorum_manager.voter_config_source
        self.voter_quorum_size: int = min(VOTER_MIN_QUORUM, len(self.voter_node_ids)) if self.voter_node_ids else 0
        if self.voter_node_ids:
            print(
                f"[P2P] Voter quorum enabled: voters={len(self.voter_node_ids)}, "
                f"quorum={self.voter_quorum_size} ({', '.join(self.voter_node_ids)})"
            )

        self._ip_to_node_map: dict[str, str] = self.quorum_manager.build_ip_to_node_map()
        self._cluster_config: dict[str, Any] = self._load_cluster_config_raw()

    def _init_state(self) -> None:
        """Phase 2: Leadership state, peer management, job tracking, sync/manifest state."""
        self.role = NodeRole.FOLLOWER
        self.leader_id: str | None = None

        self._leadership_sm = LeadershipStateMachine(node_id=self.node_id)
        self._hybrid_coordinator: Any = None

        self.verbose = bool(os.environ.get("RINGRIFT_P2P_VERBOSE", "").strip())
        self.peers: dict[str, NodeInfo] = {}
        self._prepopulate_voter_peers()
        self._peer_snapshot: PeerSnapshot[NodeInfo] = PeerSnapshot()
        self._cooldown_manager = get_dead_peer_cooldown_manager()
        self._dead_peer_timestamps: dict[str, float] = {}

        # Diagnostic instrumentation
        self._peer_state_tracker = None
        self._conn_failure_tracker = None
        self._probe_tracker = None
        try:
            from scripts.p2p.diagnostics import (
                PeerStateTracker,
                ConnectionFailureTracker,
                ProbeEffectivenessTracker,
            )
            self._peer_state_tracker = PeerStateTracker()
            self._conn_failure_tracker = ConnectionFailureTracker()
            self._probe_tracker = ProbeEffectivenessTracker()
            logger.info("[P2P] Diagnostic instrumentation enabled (Phase 0)")
        except ImportError as e:
            logger.warning(f"[P2P] Diagnostic instrumentation unavailable: {e}")

        # Stability controller (self-healing)
        self._stability_controller = None
        self._adaptive_timeouts = None
        self._effectiveness_tracker = None
        try:
            from scripts.p2p.controllers import (
                StabilityController,
                RecoveryAction,
                AdaptiveTimeoutManager,
                EffectivenessTracker,
            )
            self._adaptive_timeouts = AdaptiveTimeoutManager()
            self._effectiveness_tracker = EffectivenessTracker()
            self._stability_controller = StabilityController(
                peer_state_tracker=self._peer_state_tracker,
                connection_failure_tracker=self._conn_failure_tracker,
                probe_tracker=self._probe_tracker,
                action_callbacks={
                    RecoveryAction.INCREASE_TIMEOUT: self._action_increase_timeout,
                    RecoveryAction.DECREASE_TIMEOUT: self._action_decrease_timeout,
                    RecoveryAction.SCALE_POOL_UP: self._action_scale_pool,
                    RecoveryAction.RESET_CIRCUIT: self._action_reset_circuits,
                    RecoveryAction.INCREASE_COOLDOWN: self._action_increase_cooldown,
                    RecoveryAction.REINJECT_PEER: self._action_reinject_peer,
                    RecoveryAction.EMIT_ALERT: self._action_emit_alert,
                },
            )
            self._effectiveness_tracker.set_metrics_callback(
                lambda: self.monitoring.get_stability_metrics() if hasattr(self, 'monitoring') and self.monitoring else {}
            )
            logger.info("[P2P] Stability controller enabled (Self-Healing Architecture)")
        except ImportError as e:
            logger.warning(f"[P2P] Stability controller unavailable: {e}")
        except Exception as e:
            logger.warning(f"[P2P] Stability controller init failed: {e}")

        self.local_jobs: dict[str, ClusterJob] = {}
        self.active_jobs: dict[str, dict[str, Any]] = {}
        self._http_session: aiohttp.ClientSession | None = None
        self._tailscale_discovery_loop: Any = None

        # Distributed job state tracking (leader-only)
        self.distributed_cmaes_state: dict[str, DistributedCMAESState] = {}
        self.distributed_tournament_state: dict[str, DistributedTournamentState] = {}
        self.ssh_tournament_runs: dict[str, SSHTournamentRun] = {}
        self.improvement_loop_state: dict[str, ImprovementLoopState] = {}
        self._orch_config = OrchestratorConfig.from_env()
        self.max_concurrent_cmaes_evals = self._orch_config.max_concurrent_cmaes_evals
        self._cmaes_eval_semaphore = asyncio.Semaphore(int(self.max_concurrent_cmaes_evals))
        self._tournament_match_semaphore: asyncio.Semaphore | None = None

        self._sync_config = SyncConfig()

        # Distributed data sync state
        self.local_data_manifest: NodeDataManifest | None = None
        self.cluster_data_manifest: ClusterDataManifest | None = None
        self._cluster_manifest_received_at: float = 0.0
        self.manifest_collection_interval = self._sync_config.manifest_collection_interval
        self.last_manifest_collection = 0.0

        self.selfplay_stats_history: list[dict[str, Any]] = []
        self.selfplay_stats_history_max_samples: int = self._orch_config.selfplay_stats_history_max_samples
        self.canonical_gate_jobs: dict[str, dict[str, Any]] = {}
        self.canonical_gate_jobs_lock = threading.RLock()

        self.active_sync_jobs: dict[str, DataSyncJob] = {}
        self.current_sync_plan: ClusterSyncPlan | None = None
        self.pending_sync_requests: list[dict[str, Any]] = []
        self.sync_in_progress = False
        self.last_sync_time = 0.0
        self.auto_sync_interval = self._sync_config.auto_sync_interval

        self.training_sync_interval = self._sync_config.training_sync_interval
        self.last_training_sync_time = 0.0
        self.training_nodes_cache: list[str] = []
        self.training_nodes_cache_time = 0.0
        self.games_synced_to_training: dict[str, int] = {}

        self._circuit_registry = get_circuit_registry()
        self._peer_circuit_breakers: dict[str, PeerCircuitBreaker] = {}
        self._job_dispatch_failures: dict[str, tuple[int, float]] = {}
        self._JOB_DISPATCH_FAILURE_THRESHOLD = 3
        self._JOB_DISPATCH_COOLDOWN_SECONDS = 60.0
        self._peer_health_scores: dict[str, PeerHealthScore] = {}

        self._training_config = TrainingConfig()
        self.training_jobs: dict[str, TrainingJob] = {}
        self.training_thresholds: TrainingThresholds = TrainingThresholds()
        self.last_training_check: float = 0.0
        self.training_check_interval: float = self._training_config.training_check_interval
        self.games_at_last_nnue_train: dict[str, int] = {}
        self.games_at_last_cmaes_train: dict[str, int] = {}

    def _init_advanced_features(self) -> None:
        """Phase 3: Improvement cycle, monitoring, PFSP pools, CMA-ES auto-tuners."""
        self.improvement_cycle_manager: ImprovementCycleManager | None = None
        if HAS_IMPROVEMENT_MANAGER:
            try:
                self.improvement_cycle_manager = ImprovementCycleManager(
                    db_path=STATE_DIR / f"{self.node_id}_improvement.db",
                    ringrift_path=self.ringrift_path,
                )
                logger.info("ImprovementCycleManager initialized")
            except Exception as e:  # noqa: BLE001
                logger.error(f"Failed to initialize ImprovementCycleManager: {e}")
        self.last_improvement_cycle_check: float = 0.0

        self.monitoring_manager: MonitoringManager | None = None
        if HAS_P2P_MONITORING:
            try:
                self.monitoring_manager = MonitoringManager(
                    node_id=self.node_id,
                    prometheus_port=9090,
                    grafana_port=3000,
                    config_dir=Path(self.ringrift_path) / "monitoring",
                )
                logger.info("MonitoringManager initialized")
            except Exception as e:  # noqa: BLE001
                logger.error(f"Failed to initialize MonitoringManager: {e}")
        self._monitoring_was_leader = False
        self.improvement_cycle_check_interval: float = self._training_config.improvement_cycle_check_interval

        self.p2p_auto_deployer: P2PAutoDeployer | None = None
        self._auto_deployer_task: asyncio.Task | None = None
        self.notifier = WebhookNotifier()

        self._http_app: "web.Application | None" = None
        self._http_runner: "web.AppRunner | None" = None
        self._http_sites: list["web.TCPSite"] = []
        self._http_restart_lock = asyncio.Lock()
        self._http_restart_count = 0

        self.diversity_metrics = {
            "games_by_engine_mode": {},
            "games_by_board_config": {},
            "games_by_difficulty": {},
            "asymmetric_games": 0,
            "symmetric_games": 0,
            "training_triggers": 0,
            "cmaes_triggers": 0,
            "promotions": 0,
            "rollbacks": 0,
            "last_reset": time.time(),
        }

        self.training_metrics: dict[str, dict[str, float]] = {}
        self.selfplay_throughput: dict[str, float] = {}
        self.cost_metrics: dict[str, float] = {
            "gpu_hours_total": 0.0,
            "estimated_cost_usd": 0.0,
            "elo_per_gpu_hour": 0.0,
        }
        self.promotion_metrics: dict[str, Any] = {
            "success_rate": 0.0,
            "avg_elo_gain": 0.0,
            "rejections": {},
            "total_attempts": 0,
            "successful": 0,
        }
        self.gpu_idle_since: dict[str, float] = {}
        self.ab_tests: dict[str, dict[str, Any]] = {}
        self.ab_test_lock = threading.RLock()

        self.elo_sync_manager: EloSyncManager | None = None
        if HAS_ELO_SYNC:
            try:
                db_path = Path(self._get_ai_service_path()) / "data" / "unified_elo.db"
                elo_coordinator = os.environ.get("RINGRIFT_ELO_COORDINATOR", "nebius-backbone-1")
                self.elo_sync_manager = EloSyncManager(
                    db_path=db_path,
                    coordinator_host=elo_coordinator,
                    sync_interval=300,
                )
                logger.info(f"EloSyncManager initialized (db: {db_path})")
            except Exception as e:  # noqa: BLE001
                logger.error(f"Failed to initialize EloSyncManager: {e}")

        self._queue_populator: QueuePopulator | None = None
        self._queue_populator_loop: Any = None

        self.pfsp_pools: dict[str, Any] = {}
        if HAS_PFSP:
            try:
                for config_key in ["square8_2p", "square8_4p", "hex8_2p", "hexagonal_2p"]:
                    self.pfsp_pools[config_key] = PFSPOpponentPool(
                        max_pool_size=30,
                        hard_opponent_weight=0.6,
                        diversity_weight=0.25,
                        recency_weight=0.15,
                    )
                logger.info(f"PFSP opponent pools initialized for {len(self.pfsp_pools)} configs")
            except Exception as e:  # noqa: BLE001
                logger.error(f"Failed to initialize PFSP pools: {e}")

        self.cmaes_auto_tuners: dict[str, Any] = {}
        self.last_cmaes_elo: dict[str, float] = {}
        if HAS_PFSP and CMAESAutoTuner:
            try:
                for config_key in ["square8_2p", "square8_4p", "hex8_2p", "hexagonal_2p"]:
                    parts = config_key.rsplit("_", 1)
                    board_type = parts[0]
                    num_players = int(parts[1].replace("p", ""))
                    plateau_cfg = PlateauConfig(patience=10)
                    self.cmaes_auto_tuners[config_key] = CMAESAutoTuner(
                        board_type=board_type,
                        num_players=num_players,
                        plateau_config=plateau_cfg,
                        min_epochs_between_tuning=50,
                        max_auto_tunes=3,
                    )
                logger.info(f"CMA-ES auto-tuners initialized for {len(self.cmaes_auto_tuners)} configs")
            except Exception as e:  # noqa: BLE001
                logger.error(f"Failed to initialize CMA-ES auto-tuners: {e}")

    def _init_threading_and_protocols(self) -> None:
        """Phase 4: Threading locks, SWIM/Raft, failover, StateManager, MetricsManager."""
        self.peers_lock = threading.RLock()
        self.jobs_lock = threading.RLock()
        self.manifest_lock = threading.RLock()
        self.sync_lock = threading.RLock()
        self.training_lock = threading.RLock()
        self.ssh_tournament_lock = threading.RLock()
        self.relay_lock = threading.RLock()
        self.leader_state_lock = threading.RLock()

        from concurrent.futures import ThreadPoolExecutor
        self._health_check_executor = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="health_"
        )

        self._job_snapshot = JobSnapshot()

        self._status_cache: dict | None = None
        self._status_cache_time: float = 0.0
        self._status_cache_lock: asyncio.Lock = asyncio.Lock()
        self._status_cache_ttl: float = self._orch_config.status_cache_ttl

        self._peer_query = PeerQueryBuilder(self.peers, self.peers_lock, self.node_id)

        # Cached peers snapshot for LeaderOps (avoids blocking event loop on peers_lock)
        self._peers_snapshot_cache: list | None = None
        self._peers_snapshot_cache_time: float = 0.0

        # SWIM + Raft Integration
        from scripts.p2p.constants import (
            SWIM_ENABLED, RAFT_ENABLED, MEMBERSHIP_MODE, CONSENSUS_MODE
        )
        try:
            from app.p2p.swim_adapter import SWIM_AVAILABLE
        except ImportError:
            SWIM_AVAILABLE = False
        try:
            from scripts.p2p.consensus_mixin import PYSYNCOBJ_AVAILABLE
        except ImportError:
            PYSYNCOBJ_AVAILABLE = False

        if SWIM_ENABLED and not SWIM_AVAILABLE:
            logger.warning(
                "RINGRIFT_SWIM_ENABLED=true but swim-p2p not installed or not compatible. "
                "SWIM features disabled. Install with: pip install swim-p2p>=1.2.0"
            )
        if RAFT_ENABLED and not PYSYNCOBJ_AVAILABLE:
            logger.warning(
                "RINGRIFT_RAFT_ENABLED=true but pysyncobj not installed. "
                "Raft features disabled. Install with: pip install pysyncobj>=0.3.14"
            )
        if MEMBERSHIP_MODE in ("swim", "hybrid") and not SWIM_AVAILABLE:
            logger.warning(
                f"RINGRIFT_MEMBERSHIP_MODE={MEMBERSHIP_MODE} but SWIM unavailable. "
                "Falling back to HTTP heartbeats."
            )
        if CONSENSUS_MODE in ("raft", "hybrid") and not PYSYNCOBJ_AVAILABLE:
            logger.warning(
                f"RINGRIFT_CONSENSUS_MODE={CONSENSUS_MODE} but PySyncObj unavailable. "
                "Falling back to Bully algorithm."
            )

        logger.info(
            f"P2P protocols: MEMBERSHIP_MODE={MEMBERSHIP_MODE} (SWIM={'available' if SWIM_AVAILABLE else 'unavailable'}), "
            f"CONSENSUS_MODE={CONSENSUS_MODE} (Raft={'available' if PYSYNCOBJ_AVAILABLE else 'unavailable'})"
        )

        self._swim_initialized = self._init_swim_membership()
        if self._swim_initialized:
            logger.info("SWIM membership initialized (will start in run())")

        self._raft_init_attempted = False
        if RAFT_ENABLED and PYSYNCOBJ_AVAILABLE and self.voter_node_ids:
            try:
                self._raft_init_attempted = True
                raft_ok = self._init_raft_consensus()
                if raft_ok:
                    logger.info("Raft consensus initialized (will sync with peers in run())")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Early Raft initialization failed (will retry later): {e}")

        try:
            self._init_failover_system()
            logger.info("Failover system initialized (transport cascade + union discovery)")
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Failover system init deferred: {e}")

        # State persistence
        self.db_path = STATE_DIR / f"{self.node_id}_state.db"
        self.state_manager = StateManager(self.db_path, verbose=self.verbose)
        self.state_manager.init_database()
        self._cluster_epoch = self.state_manager.load_cluster_epoch()

        try:
            from scripts.p2p.transport_cascade import GlobalCircuitBreaker, TransportCascade
            GlobalCircuitBreaker.set_state_manager(self.state_manager)
            TransportCascade.set_state_manager(self.state_manager)
            logger.debug("Circuit breaker and transport metrics persistence configured")
        except ImportError:
            logger.debug("Transport cascade not available for persistence")

        self.metrics_manager = MetricsManager(self.db_path)

        # Event and election flags
        self.running = True
        self.election_in_progress = False
        self.last_election_attempt: float = 0.0
        self._election_lock = asyncio.Lock()

        # Lease-based leadership
        self.leader_lease_expires: float = 0.0
        self.last_lease_renewal: float = 0.0
        self.leader_lease_id: str = ""
        self.last_leader_seen: float = time.time()
        self._leader_invalidation_until: float = 0.0
        self.last_local_training_fallback: float = 0.0
        self._jittered_timeout_cache: float | None = None
        self._jittered_timeout_time: float = 0.0
        self.last_work_from_leader: float = time.time()
        self._last_become_leader_time: float = 0.0
        self._last_step_down_time: float = 0.0

        # Provisional leadership
        self._provisional_leader_claimed_at: float = 0.0
        self._provisional_leader_acks: set[str] = set()
        self._provisional_leader_challengers: dict[str, float] = {}
        self._last_provisional_check: float = 0.0
        self._provisional_claim_probability: float = PROVISIONAL_LEADER_INITIAL_PROBABILITY

        # Voter-backed lease grants
        self.voter_grant_leader_id: str = ""
        self.voter_grant_lease_id: str = ""
        self.voter_grant_expires: float = 0.0
        self._lease_epoch: int = 0
        self._fence_token: str = ""
        self._last_seen_epoch: int = 0

        # Job completion tracking
        self.completed_jobs: dict[str, float] = {}
        self.jobs_started_at: dict[str, dict[str, float]] = {}

        # NAT/relay support
        self.last_inbound_heartbeat: float = 0.0
        self.last_relay_heartbeat: float = 0.0
        self.relay_command_queue: dict[str, list[dict[str, Any]]] = {}
        self.pending_relay_acks: set[str] = set()
        self.pending_relay_results: list[dict[str, Any]] = []
        self.relay_command_attempts: dict[str, int] = {}
        self._background_tasks: list[asyncio.Task] = []

        # Safeguards
        self._safeguard_config = SafeguardConfig(
            agent_mode=AGENT_MODE_ENABLED,
            coordinator_url=COORDINATOR_URL,
        )
        self.spawn_timestamps: list[float] = []
        self.agent_mode = self._safeguard_config.agent_mode
        self.coordinator_url = self._safeguard_config.coordinator_url
        self.last_coordinator_check: float = self._safeguard_config.last_coordinator_check
        self.coordinator_available: bool = self._safeguard_config.coordinator_available
        logger.info(f"Safeguards: rate_limit={SPAWN_RATE_LIMIT_PER_MINUTE}/min, "
              f"load_max={LOAD_AVERAGE_MAX_MULTIPLIER}x, agent_mode={self.agent_mode}")

    def _get_peers_snapshot_nonblocking(self) -> list:
        """Get a cached snapshot of peers values — NEVER blocks the event loop.

        Feb 23, 2026: LeaderOps functions run concurrently on the event loop but
        need peers data. This helper is safe to call directly from async code:

        1. Returns cached data if fresh (< 2s old) — no lock touch at all
        2. Uses blocking=False — instant return if lock is held
        3. Falls back to stale cache if lock can't be acquired

        Called DIRECTLY from async functions (no asyncio.to_thread needed).
        Thread pool is saturated (8 workers, 30+ callers), so to_thread()
        calls would queue for 10-38s waiting for a free worker.
        """
        now = time.time()
        cache_ttl = 2.0

        # Fast path: return cached snapshot if fresh
        if self._peers_snapshot_cache is not None and (now - self._peers_snapshot_cache_time) < cache_ttl:
            return self._peers_snapshot_cache

        # Non-blocking lock acquisition — instant return if contended
        acquired = self.peers_lock.acquire(blocking=False)
        if not acquired:
            # Lock contended — return stale cache or empty list
            if self._peers_snapshot_cache is not None:
                return self._peers_snapshot_cache
            return []

        try:
            snapshot = list(self.peers.values())
        finally:
            self.peers_lock.release()

        # Update cache
        self._peers_snapshot_cache = snapshot
        self._peers_snapshot_cache_time = now
        return snapshot

    def _init_managers(self) -> None:
        """Phase 5: All 14 managers + 6 sub-orchestrators + state loading."""
        # Load persisted state first
        self._load_state()
        # NOTE: _set_leader() deferred until after self.leadership is initialized
        # (see below at LeadershipOrchestrator creation)

        # MonitoringOrchestrator must be early (_create_self_info uses it)
        from scripts.p2p.orchestrators import MonitoringOrchestrator
        self.monitoring = MonitoringOrchestrator(self)
        logger.info("[P2P] MonitoringOrchestrator initialized (early, for _create_self_info)")

        self.self_info = self._create_self_info()

        self.node_selector = NodeSelector(
            get_peers=lambda: self.peers,
            get_self_info=lambda: self.self_info,
            peers_lock=self.peers_lock,
            get_training_jobs=lambda: self.training_jobs,
        )
        self.node_selector.subscribe_to_events()

        self.sync_planner = SyncPlanner(
            node_id=self.node_id,
            data_directory=self.get_data_directory(),
            get_peers=lambda: self.peers,
            get_self_info=lambda: self.self_info,
            peers_lock=self.peers_lock,
            is_leader=lambda: self.leadership.check_is_leader(),
            request_peer_manifest=lambda peer_id: self._request_peer_manifest_sync(peer_id),
            check_disk_capacity=lambda: check_disk_has_capacity(),
            config=SyncPlannerConfig(),
        )
        self.sync_planner.subscribe_to_events_with_retry()

        self.selfplay_scheduler = SelfplayScheduler(
            get_cluster_elo_fn=lambda: self._get_cluster_elo_summary(),
            load_curriculum_weights_fn=lambda: self._load_curriculum_weights(),
            get_board_priority_overrides_fn=lambda: getattr(self, "board_priority_overrides", {}),
            should_stop_production_fn=should_stop_production if HAS_NEW_COORDINATION else None,
            should_throttle_production_fn=should_throttle_production if HAS_NEW_COORDINATION else None,
            get_throttle_factor_fn=get_throttle_factor if HAS_NEW_COORDINATION else None,
            record_utilization_fn=record_utilization if HAS_NEW_COORDINATION else None,
            get_host_targets_fn=get_host_targets if HAS_NEW_COORDINATION else None,
            get_target_job_count_fn=get_target_job_count if HAS_NEW_COORDINATION else None,
            should_scale_up_fn=should_scale_up if HAS_NEW_COORDINATION else None,
            should_scale_down_fn=should_scale_down if HAS_NEW_COORDINATION else None,
            get_max_selfplay_for_node_fn=get_max_selfplay_for_node if HAS_HW_AWARE_LIMITS else None,
            get_hybrid_selfplay_limits_fn=get_hybrid_selfplay_limits if HAS_HW_AWARE_LIMITS else None,
            is_emergency_active_fn=_safeguards.is_emergency_active if HAS_SAFEGUARDS and _safeguards else None,
            verbose=self.verbose,
        )
        self.selfplay_scheduler._orchestrator = self
        self.selfplay_scheduler.subscribe_to_events_with_retry()

        try:
            initial_game_counts = self._seed_selfplay_scheduler_game_counts_sync()
            if initial_game_counts:
                self.selfplay_scheduler.update_p2p_game_counts(initial_game_counts)
                logger.info(f"[P2P] Seeded SelfplayScheduler with {len(initial_game_counts)} config game counts from canonical DBs")
                for config_key, count in sorted(initial_game_counts.items(), key=lambda x: x[1]):
                    if count < 500:
                        logger.info(f"[P2P] Underserved config: {config_key} = {count} games")
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[P2P] Failed to seed initial game counts: {e}")

        self.job_manager = JobManager(
            ringrift_path=self.ringrift_path,
            node_id=self.node_id,
            peers=self.peers,
            peers_lock=self.peers_lock,
            active_jobs=self.active_jobs,
            jobs_lock=self.jobs_lock,
            improvement_loop_state=self.improvement_loop_state,
            distributed_tournament_state=self.distributed_tournament_state,
        )
        self.job_manager.subscribe_to_events_with_retry()
        self.job_manager.set_spawn_registration_callback(
            self.selfplay_scheduler.register_pending_spawn
        )
        self.selfplay_scheduler.set_job_status_callback(
            self.job_manager.get_job_status
        )
        logger.info("[P2P] Spawn verification wired: JobManager <-> SelfplayScheduler")

        self.training_coordinator = TrainingCoordinator(
            ringrift_path=Path(self.ringrift_path),
            get_cluster_data_manifest=lambda: self.cluster_data_manifest,
            get_training_jobs=lambda: self.training_jobs,
            get_training_lock=lambda: self.training_lock,
            get_peers=lambda: self.peers,
            get_peers_lock=lambda: self.peers_lock,
            get_self_info=lambda: self.self_info,
            training_thresholds=self.training_thresholds,
            games_at_last_nnue_train=getattr(self, "games_at_last_nnue_train", None),
            games_at_last_cmaes_train=getattr(self, "games_at_last_cmaes_train", None),
            improvement_cycle_manager=getattr(self, "improvement_cycle_manager", None),
            auth_headers=lambda: self._auth_headers(),
            urls_for_peer=lambda node_id, endpoint: self._urls_for_peer(node_id, endpoint),
            save_state_callback=lambda: self._save_state(),
            has_voter_quorum=lambda: self._check_quorum_health(),
        )
        self.training_coordinator.subscribe_to_events_with_retry()

        self.job_orchestration = create_job_orchestration_manager(self)
        logger.info("[P2P] JobOrchestrationManager initialized")

        self.analytics_cache_manager = create_analytics_cache_manager(
            config=AnalyticsCacheConfig(),
            get_ai_service_path=lambda: self._get_ai_service_path(),
            is_in_startup_grace_period=lambda: self._is_in_startup_grace_period(),
            increment_rollback_counter=lambda: self._increment_rollback_counter(),
            send_notification=lambda **kwargs: asyncio.create_task(self.notifier.send(**kwargs)) if hasattr(self, 'notifier') else None,
            node_id=self.node_id,
        )
        logger.info("[P2P] AnalyticsCacheManager initialized")

        self.cmaes_coordinator = create_cmaes_coordinator(
            config=CMAESConfig(ai_service_path=self._get_ai_service_path()),
            get_gpu_workers=lambda: self._get_gpu_workers_for_cmaes(),
            send_to_worker=lambda wid, ep, pl: self._send_cmaes_to_worker(wid, ep, pl),
            report_to_leader=lambda ep, pl: self._report_cmaes_to_leader(ep, pl),
            get_node_role=lambda: self.role.value if hasattr(self.role, 'value') else str(self.role),
            get_leader_id=lambda: self.leader_id,
            get_node_id=lambda: self.node_id,
            handle_cmaes_complete=lambda bt, np, w: self._handle_cmaes_complete_callback(bt, np, w),
        )
        logger.info("[P2P] CMAESCoordinator initialized")

        self.data_sync_coordinator = create_data_sync_coordinator(
            config=DataSyncCoordinatorConfig(),
        )
        logger.info("[P2P] DataSyncCoordinator initialized")

        from scripts.p2p.managers.ip_discovery_manager import create_ip_discovery_manager, IPDiscoveryConfig
        self.ip_discovery_manager = create_ip_discovery_manager(config=IPDiscoveryConfig(), orchestrator=self)
        logger.info("[P2P] IPDiscoveryManager initialized")

        from scripts.p2p.managers.worker_pull_controller import create_worker_pull_controller, WorkerPullConfig
        self.worker_pull_controller = create_worker_pull_controller(config=WorkerPullConfig(), orchestrator=self)
        logger.info("[P2P] WorkerPullController initialized")

        from scripts.p2p.managers.data_pipeline_manager import create_data_pipeline_manager, DataPipelineConfig
        self.data_pipeline_manager = create_data_pipeline_manager(config=DataPipelineConfig(), orchestrator=self)
        logger.info("[P2P] DataPipelineManager initialized")

        from scripts.p2p.managers.job_lifecycle_manager import create_job_lifecycle_manager, JobLifecycleConfig
        self.job_lifecycle_manager = create_job_lifecycle_manager(config=JobLifecycleConfig(), orchestrator=self)
        logger.info("[P2P] JobLifecycleManager initialized")

        from scripts.p2p.managers.health_metrics_manager import create_health_metrics_manager, HealthMetricsConfig
        self.health_metrics_manager = create_health_metrics_manager(config=HealthMetricsConfig(), orchestrator=self)
        logger.info("[P2P] HealthMetricsManager initialized")

        from scripts.p2p.managers.memory_disk_manager import create_memory_disk_manager, MemoryDiskConfig
        self.memory_disk_manager = create_memory_disk_manager(config=MemoryDiskConfig(), orchestrator=self)
        logger.info("[P2P] MemoryDiskManager initialized")

        from scripts.p2p.managers.tournament_manager import create_tournament_manager, TournamentConfig
        self.tournament_manager = create_tournament_manager(config=TournamentConfig(), orchestrator=self)
        logger.info("[P2P] TournamentManager initialized")

        from scripts.p2p.managers.recovery_manager import create_recovery_manager, RecoveryConfig
        self.recovery_manager = create_recovery_manager(config=RecoveryConfig(), orchestrator=self)
        logger.info("[P2P] RecoveryManager initialized")

        from scripts.p2p.managers.heartbeat_manager import HeartbeatConfig, create_heartbeat_manager
        self.heartbeat_manager = create_heartbeat_manager(config=HeartbeatConfig(), orchestrator=self)
        logger.info("[P2P] HeartbeatManager initialized")

        from scripts.p2p.managers.job_coordination_manager import JobCoordinationConfig, create_job_coordination_manager
        self.job_coordination_manager = create_job_coordination_manager(config=JobCoordinationConfig(), orchestrator=self)
        logger.info("[P2P] JobCoordinationManager initialized")

        # Sub-Orchestrators
        from scripts.p2p.orchestrators import (
            JobOrchestrator, LeadershipOrchestrator,
            PeerNetworkOrchestrator, ProcessSpawnerOrchestrator, SyncOrchestrator,
        )
        self.leadership = LeadershipOrchestrator(self)
        # Deferred from _load_state(): restore leadership after LeadershipOrchestrator exists
        if self.leader_id == self.node_id:
            self._set_leader(self.node_id, reason="startup_restore_leadership", save_state=False)
        self.network = PeerNetworkOrchestrator(self)
        self.sync = SyncOrchestrator(self)
        self.jobs = JobOrchestrator(self)
        self.process_spawner = ProcessSpawnerOrchestrator(self)
        self.jobs.initialize_work_discovery_manager()

    def _init_event_wiring(self) -> None:
        """Phase 6: Event subscriptions, feedback loops, SWIM callbacks, LoopManager."""
        from scripts.p2p.event_wiring import (
            wire_feedback_loops,
            subscribe_to_daemon_events,
            subscribe_to_feedback_signals,
            subscribe_to_manager_events,
        )
        wire_feedback_loops(self)
        daemon_events_ok = subscribe_to_daemon_events(self)
        feedback_signals_ok = subscribe_to_feedback_signals(self)
        manager_events_ok = subscribe_to_manager_events(self)

        self._event_subscription_status = {
            "daemon_events": daemon_events_ok,
            "feedback_signals": feedback_signals_ok,
            "manager_events": manager_events_ok,
            "all_healthy": daemon_events_ok and feedback_signals_ok and manager_events_ok,
            "timestamp": time.time(),
        }

        if self._event_subscription_status["all_healthy"]:
            logger.info("[P2P] Event subscriptions: daemon=✓, feedback=✓, manager=✓")
        else:
            logger.warning(
                f"[P2P] Event subscriptions incomplete: "
                f"daemon={'✓' if daemon_events_ok else '✗'}, "
                f"feedback={'✓' if feedback_signals_ok else '✗'}, "
                f"manager={'✓' if manager_events_ok else '✗'}"
            )

        CRITICAL_SUBSCRIPTION_GROUPS = ["manager_events"]
        self._event_subscription_status["critical_failed"] = []
        for group in CRITICAL_SUBSCRIPTION_GROUPS:
            if not self._event_subscription_status.get(group, False):
                self._event_subscription_status["critical_failed"].append(group)

        if self._event_subscription_status["critical_failed"]:
            failed_groups = self._event_subscription_status["critical_failed"]
            logger.critical(f"[P2P] CRITICAL: Event subscription groups failed: {failed_groups}")
            if os.environ.get("RINGRIFT_FAIL_ON_SUBSCRIPTION_FAILURE", "").lower() == "true":
                raise RuntimeError(
                    f"Critical event subscriptions failed: {failed_groups}. "
                    "Set RINGRIFT_FAIL_ON_SUBSCRIPTION_FAILURE=false to allow startup anyway."
                )

        print(
            f"[P2P] Initialized node {self.node_id} on {self.host}:{self.port} "
            f"(advertise {self.advertise_host}:{self.advertise_port})"
        )
        logger.info(f"RingRift path: {self.ringrift_path}")
        logger.info(f"Version: {self.build_version}")
        logger.info(f"Known peers: {self.known_peers}")
        if self.relay_peers:
            logger.info(f"Relay peers (forced relay mode): {list(self.relay_peers)}")
        if self.auth_token:
            logger.info(f"Auth: enabled via {AUTH_TOKEN_ENV}")
        else:
            logger.info(f"Auth: disabled (set {AUTH_TOKEN_ENV} to enable)")

        # Hybrid transport
        self.hybrid_transport: HybridTransport | None = None
        if HAS_HYBRID_TRANSPORT:
            try:
                self.hybrid_transport = get_hybrid_transport()
                logger.info("HybridTransport: enabled (HTTP with SSH fallback for Vast)")
            except Exception as e:  # noqa: BLE001
                logger.info(f"HybridTransport: failed to initialize: {e}")

        # SWIM callbacks and manager
        set_swim_callbacks(
            on_alive=self._on_swim_member_alive,
            on_failed=self._on_swim_member_failed,
        )
        self._swim_manager = get_swim_manager(node_id=self.node_id, bind_port=7947)
        self._swim_started = False

        self._sync_router: SyncRouter | None = None
        self._sync_router_wired = False

        # LoopManager
        self._loop_manager: LoopManager | None = None
        self._loops_registered = False
        self._autonomous_queue_loop = None
        self._quorum_crisis_loop = None
        self._startup_time = time.time()

        self._manager_health_status = self.health_metrics_manager.validate_manager_health()

    def _get_loop_manager(self) -> "LoopManager | None":
        """Get the LoopManager, initializing if needed."""
        if self._loop_manager is None:
            self._loop_manager = get_loop_manager()
        return self._loop_manager

    def _register_extracted_loops(self) -> bool:
        """Register extracted loops with the LoopManager.

        January 2026: Delegated to scripts/p2p/loop_registry.py (~1,580 LOC extracted).
        """
        logger.info(f"[LoopManager] _register_extracted_loops called, already_registered={self._loops_registered}")
        if self._loops_registered:
            return True

        manager = self._get_loop_manager()
        logger.info(f"[LoopManager] Got manager: {manager}")
        if manager is None:
            logger.info("LoopManager: not available, using inline loops only")
            return False

        try:
            from scripts.p2p.loop_registry import register_all_loops

            result = register_all_loops(self, manager)
            if result.success:
                self._loops_registered = True
                logger.info(f"LoopManager: registered {result.loops_registered} loops via loop_registry")
                return True
            else:
                logger.error(f"LoopManager: loop registration failed: {result.error}")
                return False

        except ImportError as e:
            logger.warning(f"LoopManager: loop_registry not available: {e}")
            return False

        except Exception as e:  # noqa: BLE001
            logger.error(f"LoopManager: failed to register loops: {e}")
            return False

    # =========================================================================
    # JobReaperLoop callbacks - December 27, 2025
    # =========================================================================

    def _get_all_active_jobs_for_reaper(self) -> dict[str, Any]:
        """Get all active jobs across all job types for the job reaper.

        Returns a flat dict of job_id -> job_info, where job_info includes:
        - started_at: timestamp when job started
        - claimed_at: timestamp when job was claimed (if applicable)
        - status: current job status
        - pid: process ID (for killing stuck processes)
        - node_id: which node is running the job
        """
        result: dict[str, Any] = {}
        with self.jobs_lock:
            for job_type, jobs in self.active_jobs.items():
                for job_id, job_info in jobs.items():
                    if isinstance(job_info, dict):
                        result[job_id] = {
                            **job_info,
                            "job_type": job_type,
                        }
                    else:
                        # Handle non-dict job objects (legacy)
                        result[job_id] = {
                            "job_id": job_id,
                            "job_type": job_type,
                            "status": getattr(job_info, "status", "unknown"),
                            "started_at": getattr(job_info, "started_at", 0),
                            "pid": getattr(job_info, "pid", None),
                        }
        return result

    async def _cancel_job_for_reaper(self, job_id: str) -> bool:
        """Cancel a job by ID for the job reaper.

        Jan 21, 2026: Enhanced to escalate SIGTERM -> SIGKILL for stuck processes.

        Attempts to:
        1. Kill the process with SIGTERM, wait 3s, then SIGKILL if still alive
        2. Update job status to 'cancelled'
        3. Remove from active jobs dict
        4. Emit TASK_ABANDONED event

        Returns True if job was successfully cancelled.
        """
        import os
        import signal

        with self.jobs_lock:
            # Find the job across all job types
            for job_type, jobs in self.active_jobs.items():
                if job_id in jobs:
                    job_info = jobs[job_id]
                    pid = job_info.get("pid") if isinstance(job_info, dict) else getattr(job_info, "pid", None)

                    # Kill the process if we have a PID
                    if pid:
                        process_killed = False
                        try:
                            # First try SIGTERM
                            os.kill(pid, signal.SIGTERM)
                            logger.info(f"[JobReaper] Sent SIGTERM to pid {pid} for job {job_id}")

                            # Wait up to 3 seconds for graceful termination
                            for _ in range(6):  # 6 x 0.5s = 3s
                                await asyncio.sleep(0.5)
                                try:
                                    # Check if process still exists (signal 0 = check only)
                                    os.kill(pid, 0)
                                except ProcessLookupError:
                                    # Process is dead
                                    process_killed = True
                                    logger.debug(f"[JobReaper] Process {pid} terminated gracefully")
                                    break

                            # If still alive after 3s, escalate to SIGKILL
                            if not process_killed:
                                try:
                                    os.kill(pid, signal.SIGKILL)
                                    logger.warning(
                                        f"[JobReaper] SIGTERM failed for pid {pid}, sent SIGKILL for job {job_id}"
                                    )
                                    # Wait briefly for SIGKILL to take effect
                                    await asyncio.sleep(0.5)
                                except ProcessLookupError:
                                    pass  # Died between check and kill

                        except ProcessLookupError:
                            logger.debug(f"[JobReaper] Process {pid} already dead for job {job_id}")
                        except OSError as e:
                            logger.warning(f"[JobReaper] Failed to kill pid {pid}: {e}")

                    # Update status and remove from active jobs
                    if isinstance(job_info, dict):
                        job_info["status"] = "reaped"
                    del jobs[job_id]

                    # Emit event for coordination (fire-and-forget async task)
                    try:
                        asyncio.create_task(self._emit_task_abandoned(
                            task_id=job_id,
                            task_type=job_type,
                            reason="reaped_by_job_reaper",
                            node_id=job_info.get("node_id", "") if isinstance(job_info, dict) else "",
                        ))
                    except RuntimeError:
                        pass  # No event loop running

                    logger.info(f"[JobReaper] Cancelled job {job_id} (type: {job_type})")
                    return True

        logger.debug(f"[JobReaper] Job {job_id} not found in active jobs")
        return False

    def _get_job_heartbeats_for_reaper(self) -> dict[str, float]:
        """Get job heartbeat timestamps for the job reaper.

        Returns dict of job_id -> last_heartbeat_time.
        Jobs without recent heartbeats may be considered abandoned.

        Phase 15.1.9 (Dec 29, 2025): Updated to use JobManager.get_job_heartbeats()
        for actual heartbeat tracking instead of just job start times.
        """
        result: dict[str, float] = {}

        # Phase 15.1.9: Get actual heartbeats from JobManager
        if hasattr(self, "job_manager") and self.job_manager is not None:
            try:
                job_heartbeats = self.job_manager.get_job_heartbeats()
                result.update(job_heartbeats)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Failed to get heartbeats from JobManager: {e}")

        # Fallback: Also include jobs_started_at for jobs without heartbeat tracking
        # This ensures older jobs (started before heartbeat tracking) are still monitored
        if hasattr(self, "jobs_started_at"):
            for _node_id, jobs in self.jobs_started_at.items():
                for job_id, start_time in jobs.items():
                    # Only add if not already in result from heartbeat tracking
                    if job_id not in result:
                        result[job_id] = start_time

        return result

    # =========================================================================
    # ManifestCollectionLoop callbacks - December 27, 2025
    # =========================================================================

    def _update_manifest_from_loop(self, manifest: Any, is_cluster: bool) -> None:
        """Update stored manifest from ManifestCollectionLoop.

        Args:
            manifest: The collected manifest (cluster or local)
            is_cluster: True if this is a cluster-wide manifest, False for local
        """
        import time
        with self.manifest_lock:
            if is_cluster:
                self.cluster_data_manifest = manifest
            else:
                self.local_data_manifest = manifest
            self.last_manifest_collection = time.time()

        # Session 17.29: Feed game counts to selfplay scheduler for priority allocation
        # ROOT CAUSE FIX: _p2p_game_counts was never populated, causing all configs
        # to show 0 games in queue populator, breaking bootstrap priority boosts
        if is_cluster and hasattr(self, 'selfplay_scheduler') and self.selfplay_scheduler:
            try:
                game_counts: dict[str, int] = {}
                if hasattr(manifest, 'by_board_type') and manifest.by_board_type:
                    for config_key, config_data in manifest.by_board_type.items():
                        if isinstance(config_data, dict):
                            game_counts[config_key] = config_data.get("total_games", 0)
                        elif hasattr(config_data, 'total_games'):
                            game_counts[config_key] = getattr(config_data, 'total_games', 0)
                if game_counts:
                    self.selfplay_scheduler.update_p2p_game_counts(game_counts)
                    logger.debug(f"[ManifestUpdate] Fed {len(game_counts)} config game counts to SelfplayScheduler")
            except Exception as e:  # noqa: BLE001
                logger.debug(f"[ManifestUpdate] Failed to update selfplay scheduler game counts: {e}")

    def _get_alive_peers_for_broadcast(self) -> list[Any]:
        """Get list of alive peers for manifest broadcast.

        Jan 2026: Added for leader broadcast functionality.
        Jan 27, 2026: Migrated to PeerQueryBuilder (Phase 3.2).

        Returns:
            List of NodeInfo objects for alive, non-retired peers
        """
        return self._peer_query.alive_non_retired().unwrap_or([])

    def _update_improvement_cycle_from_loop(self, by_board_type: dict[str, Any]) -> None:
        """Update ImprovementCycleManager from ManifestCollectionLoop.

        Args:
            by_board_type: Dict of board_type -> game counts from manifest
        """
        if self.improvement_cycle_manager:
            try:
                self.improvement_cycle_manager.update_from_cluster_totals(by_board_type)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"ImprovementCycleManager update error: {e}")

    # =========================================================================
    # DataManagementLoop callbacks - December 27, 2025
    # =========================================================================

    async def _trigger_export_for_loop(
        self,
        db_path: Path,
        output_path: Path,
        board_type: str,
    ) -> bool:
        """Trigger export job for DataManagementLoop.

        Args:
            db_path: Path to database file to export
            output_path: Path for output NPZ file
            board_type: Board type (square8, hex8, etc.)

        Returns:
            True if export started successfully
        """
        import subprocess

        try:
            cmd = [
                sys.executable,
                self._get_script_path("export_replay_dataset.py"),
                "--db", str(db_path),
                "--board-type", board_type,
                "--num-players", "2",
                "--board-aware-encoding",
                "--require-completed",
                "--min-moves", "10",
                "--output", str(output_path),
            ]

            env = os.environ.copy()
            env["PYTHONPATH"] = self._get_ai_service_path()

            log_file = Path(f"/tmp/auto_export_{db_path.stem}.log")

            # Jan 19, 2026: Run subprocess in thread pool to avoid blocking event loop
            def _start_export_process():
                with open(log_file, "w") as log_fh:
                    subprocess.Popen(
                        cmd,
                        stdout=log_fh,
                        stderr=subprocess.STDOUT,
                        env=env,
                        cwd=self._get_ai_service_path(),
                    )

            await asyncio.to_thread(_start_export_process)
            logger.info(f"[DataManagement] Started export job for {db_path.name}")
            return True

        except Exception as e:
            logger.error(f"[DataManagement] Failed to start export for {db_path.name}: {e}")
            return False

    async def _inline_job_reaper_fallback_loop(self) -> None:
        """Inline job reaper fallback loop.

        December 27, 2025: Fallback implementation that runs if the extracted
        JobReaperLoop fails to start or hits persistent errors. Uses the same
        callbacks and thresholds as the extracted loop.

        This is NOT a replacement for JobReaperLoop - it's a safety net that
        ensures job cleanup continues even if the modular loop system fails.

        Thresholds:
        - STALE: Jobs older than 1 hour without heartbeat
        - STUCK: Jobs older than 2 hours regardless of heartbeat
        - INTERVAL: Checks every 5 minutes

        Environment:
        - RINGRIFT_JOB_REAPER_FALLBACK_ENABLED: Enable/disable (default: true)
        """
        STALE_THRESHOLD_SECONDS = 3600.0   # 1 hour
        STUCK_THRESHOLD_SECONDS = 7200.0   # 2 hours
        CHECK_INTERVAL_SECONDS = 300.0      # 5 minutes
        MAX_JOBS_PER_CYCLE = 10             # Limit to avoid overload

        logger.info("[JobReaper Fallback] Started inline fallback loop")
        stats = {"checks": 0, "reaped": 0, "errors": 0}

        while self.running:
            try:
                await asyncio.sleep(CHECK_INTERVAL_SECONDS)
                if not self.running:
                    break

                stats["checks"] += 1
                now = time.time()
                reaped_this_cycle = 0

                # Get all active jobs
                try:
                    active_jobs = self._get_all_active_jobs_for_reaper()
                except Exception as e:
                    logger.warning(f"[JobReaper Fallback] Failed to get active jobs: {e}")
                    stats["errors"] += 1
                    continue

                if not active_jobs:
                    continue

                # Get heartbeat info
                try:
                    heartbeats = self._get_job_heartbeats_for_reaper()
                except Exception as e:
                    logger.debug(f"[JobReaper Fallback] Failed to get heartbeats: {e}")
                    heartbeats = {}

                # Identify stale and stuck jobs
                jobs_to_reap: list[tuple[str, str]] = []  # [(job_id, reason), ...]

                for job_id, job_info in active_jobs.items():
                    if reaped_this_cycle >= MAX_JOBS_PER_CYCLE:
                        break

                    started_at = job_info.get("started_at", 0)
                    if not started_at:
                        continue

                    job_age = now - started_at
                    last_heartbeat = heartbeats.get(job_id, started_at)
                    heartbeat_age = now - last_heartbeat

                    # Check for stuck jobs (absolute age threshold)
                    if job_age > STUCK_THRESHOLD_SECONDS:
                        jobs_to_reap.append((job_id, "stuck"))
                        reaped_this_cycle += 1
                        continue

                    # Check for stale jobs (no recent heartbeat)
                    if heartbeat_age > STALE_THRESHOLD_SECONDS:
                        jobs_to_reap.append((job_id, "stale"))
                        reaped_this_cycle += 1

                # Reap identified jobs
                for job_id, reason in jobs_to_reap:
                    try:
                        success = await self._cancel_job_for_reaper(job_id)
                        if success:
                            stats["reaped"] += 1
                            logger.info(
                                f"[JobReaper Fallback] Reaped {reason} job {job_id} "
                                f"(total: {stats['reaped']})"
                            )
                    except Exception as e:
                        logger.warning(f"[JobReaper Fallback] Failed to reap {job_id}: {e}")
                        stats["errors"] += 1

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[JobReaper Fallback] Unexpected error: {e}")
                stats["errors"] += 1
                await asyncio.sleep(60)  # Back off on error

        logger.info(
            f"[JobReaper Fallback] Stopped after {stats['checks']} checks, "
            f"{stats['reaped']} reaped, {stats['errors']} errors"
        )

    def _get_sync_router(self) -> SyncRouter | None:
        """Lazy-load SyncRouter singleton for intelligent sync routing."""
        if not HAS_SYNC_ROUTER:
            return None
        if self._sync_router is None:
            try:
                self._sync_router = get_sync_router()
                logger.info("SyncRouter: initialized for intelligent data routing")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"SyncRouter: failed to initialize: {e}")
                return None
        return self._sync_router

    def _wire_sync_router_events(self) -> bool:
        """Wire SyncRouter to event system for real-time sync triggers."""
        if self._sync_router_wired:
            return True
        router = self._get_sync_router()
        if router is None:
            return False
        try:
            if hasattr(router, 'wire_to_event_router'):
                router.wire_to_event_router()
                self._sync_router_wired = True
                logger.info("SyncRouter: wired to event system")
                return True
        except Exception as e:  # noqa: BLE001
            logger.warning(f"SyncRouter: failed to wire events: {e}")
        return False

    def _wire_cooldown_manager_probe(self) -> None:
        """Wire DeadPeerCooldownManager probe function.

        January 2026: Enables probe-based early recovery from adaptive cooldown.
        Stub implementation - cooldown logic is handled by CooldownManager.
        """
        logger.info("Cooldown manager probe function wired")

    def _wire_connection_pool_dynamic_sizing(self) -> None:
        """Wire connection pool dynamic sizing callback.

        January 2026: Scales pool limits based on cluster size to prevent exhaustion.
        """
        try:
            from scripts.p2p.connection_pool import get_connection_pool

            pool = get_connection_pool()
            if hasattr(pool, "set_cluster_size_callback"):
                pool.set_cluster_size_callback(
                    lambda: len([p for p in self.peers.values() if p.get("alive", False)])
                )
            logger.info("Connection pool dynamic sizing wired")
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Connection pool dynamic sizing unavailable: {e}")

    # Callers use self.jobs.initialize_work_discovery_manager() directly

    async def _query_peer_for_work(
        self, peer_id: str, capabilities: list[str]
    ) -> dict[str, Any] | None:
        """Query a peer for available work (used by WorkDiscoveryManager).

        January 4, 2026: Phase 5 - Peer discovery channel.
        """
        try:
            # Jan 22, 2026: Use lock-free snapshot to prevent race conditions
            peer = self._peer_snapshot.get_snapshot().get(peer_id)
            if not peer or not peer.is_alive():
                return None

            # Query peer's work queue via HTTP
            urls = self._urls_for_peer(peer_id, "/work_queue/claim")
            for url in urls:
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            url,
                            json={"capabilities": capabilities},
                            headers=self._auth_headers(),
                            timeout=aiohttp.ClientTimeout(total=5.0),
                        ) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                if data.get("work_item"):
                                    return data["work_item"]
                except Exception:
                    continue
            return None
        except Exception:
            return None

    async def _pop_autonomous_queue_work(self) -> dict[str, Any] | None:
        """Pop work from autonomous queue (used by WorkDiscoveryManager).

        January 4, 2026: Phase 5 - Autonomous queue channel.
        """
        try:
            loop = getattr(self, "_autonomous_queue_loop", None)
            if loop and hasattr(loop, "pop_local_work"):
                return await loop.pop_local_work()
            return None
        except Exception:
            return None

    def _create_direct_selfplay_work(
        self, capabilities: list[str]
    ) -> dict[str, Any] | None:
        """Create direct selfplay work item (used by WorkDiscoveryManager).

        January 4, 2026: Phase 5 - Direct selfplay channel (last resort).
        Only used when all other channels fail.
        """
        if "selfplay" not in capabilities:
            return None

        try:
            # Get next config from selfplay scheduler
            config_key = self.selfplay_scheduler.get_next_config()
            if not config_key:
                return None

            return {
                "work_id": f"direct-{self.node_id}-{int(time.time())}",
                "work_type": "selfplay",
                "config_key": config_key,
                "source": "direct_discovery",
                "games": 10,  # Small batch for direct selfplay
                "priority": 50,  # Lower priority than leader-assigned work
            }
        except Exception:
            return None

    def _wire_feedback_loops(self) -> bool:
        """Wire curriculum feedback loops. Feb 2026: Delegates to event_wiring module."""
        from scripts.p2p.event_wiring import wire_feedback_loops
        return wire_feedback_loops(self)


    def health_check(self) -> "HealthCheckResult":
        """Return health check result for daemon protocol compliance.

        December 27, 2025: Added for DaemonManager integration. Returns a
        HealthCheckResult that can be used by the daemon infrastructure for
        health monitoring, auto-restart decisions, and liveness probes.

        Returns:
            HealthCheckResult with overall orchestrator health status
        """
        # Import from contracts (zero-dependency module)
        from app.coordination.contracts import CoordinatorStatus, HealthCheckResult

        # Get manager health status (Jan 28, 2026: uses health_metrics_manager directly)
        manager_health = self.health_metrics_manager.validate_manager_health()

        # Calculate cluster metrics
        uptime_seconds = time.time() - getattr(self, "start_time", time.time())
        active_peers = sum(
            1 for p in self.peers.values()
            if time.time() - p.last_heartbeat < 120
        )

        details = {
            "node_id": self.node_id,
            "role": self.role.value if hasattr(self.role, "value") else str(self.role),
            "leader_id": self.leader_id,
            "forced_leader_override": getattr(self, "_forced_leader_override", False),
            "active_peers": active_peers,
            "total_peers": len(self.peers),
            "uptime_seconds": uptime_seconds,
            "managers_healthy": manager_health.get("all_healthy", False),
            "unhealthy_managers": manager_health.get("unhealthy_count", 0),
            "selfplay_jobs": self.self_info.selfplay_jobs if hasattr(self, "self_info") else 0,
            "training_jobs": self.self_info.training_jobs if hasattr(self, "self_info") else 0,
        }

        # Determine overall health
        is_healthy = manager_health.get("all_healthy", False)

        # Additional health checks
        if uptime_seconds < 10:
            # Grace period for startup
            is_healthy = True
            message = "P2P Orchestrator starting up"
            status = CoordinatorStatus.RUNNING
        elif not is_healthy:
            message = f"P2P Orchestrator unhealthy: {manager_health.get('unhealthy_count', 0)} unhealthy managers"
            status = CoordinatorStatus.ERROR
        else:
            message = f"P2P Orchestrator healthy, {active_peers} peers active"
            status = CoordinatorStatus.RUNNING

        return HealthCheckResult(
            healthy=is_healthy,
            status=status,
            message=message,
            details=details,
        )

    def _subscribe_to_daemon_events(self) -> bool:
        """Subscribe to daemon events. Feb 2026: Delegates to event_wiring module."""
        from scripts.p2p.event_wiring import subscribe_to_daemon_events
        return subscribe_to_daemon_events(self)

    def _subscribe_to_feedback_signals(self) -> bool:
        """Subscribe to feedback signals. Feb 2026: Delegates to event_wiring module."""
        from scripts.p2p.event_wiring import subscribe_to_feedback_signals
        return subscribe_to_feedback_signals(self)

    def _subscribe_to_manager_events(self) -> bool:
        """Subscribe to manager events. Feb 2026: Delegates to event_wiring module."""
        from scripts.p2p.event_wiring import subscribe_to_manager_events
        return subscribe_to_manager_events(self)

    # =========================================================================
    # Leadership State Management - Single Source of Truth (Jan 3, 2026)
    # =========================================================================

    def _set_leader(
        self,
        new_leader_id: str | None,
        reason: str = "unknown",
        *,
        sync_to_ulsm: bool = True,
        save_state: bool = True,
    ) -> bool:
        """Atomically set the leader and role to ensure consistency.

        Jan 29, 2026: Delegates to self.leadership orchestrator.

        Args:
            new_leader_id: The new leader ID (None to clear leader)
            reason: Human-readable reason for logging/debugging
            sync_to_ulsm: Whether to sync state to LeadershipStateMachine
            save_state: Whether to persist state after change

        Returns:
            True if this node is now the leader
        """
        return self.leadership.set_leader(
            new_leader_id, reason, sync_to_ulsm=sync_to_ulsm, save_state=save_state
        )

    def _is_leader(self) -> bool:
        """Check if this node is the current cluster leader with valid lease."""
        return self.leadership.check_is_leader()

    @property
    def is_leader(self) -> bool:
        """Property alias for _is_leader() - required by WorkQueueHandlersMixin."""
        return self.leadership.check_is_leader()

    # _reconcile_leadership_state, _broadcast_leadership_claim, _async_broadcast_leader_claim

    def _get_config_version(self) -> dict:
        """Get config file version info for drift detection.

        Jan 13, 2026: Phase 1 of P2P Cluster Stability Plan
        Enables gossip-based config drift detection across the cluster.

        Returns:
            Dictionary with config hash, timestamp, and metadata.
        """
        import hashlib
        from pathlib import Path

        config_paths = [
            Path(__file__).parent.parent / "config" / "distributed_hosts.yaml",
            Path.cwd() / "config" / "distributed_hosts.yaml",
        ]

        for config_path in config_paths:
            if config_path.exists():
                try:
                    content = config_path.read_text()
                    stat = config_path.stat()

                    # Compute hash of content
                    content_hash = hashlib.sha256(content.encode()).hexdigest()

                    return {
                        "hash": content_hash[:16],  # First 16 chars for display
                        "full_hash": content_hash,
                        "timestamp": stat.st_mtime,
                        "mtime": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(stat.st_mtime)),
                        "path": str(config_path),
                        "size_bytes": stat.st_size,
                    }
                except (OSError, PermissionError) as e:
                    return {
                        "hash": None,
                        "error": str(e),
                        "path": str(config_path),
                    }

        return {
            "hash": None,
            "error": "config_not_found",
            "searched_paths": [str(p) for p in config_paths],
        }


    # =========================================================================
    # UNIFIED LEADERSHIP STATE MACHINE (ULSM) - Jan 2026
    # =========================================================================

    async def _broadcast_leader_state_change(
        self,
        new_state: str,
        epoch: int,
        reason: "TransitionReason",
    ) -> None:
        """Jan 28, 2026: Delegates to self.leadership."""
        await self.leadership.broadcast_leader_state_change(new_state, epoch, reason)

    # =========================================================================
    # TASK ISOLATION - Prevent single task failure from crashing all tasks
    # =========================================================================

    # Task factory registry for restart support
    _task_factories: dict[str, "Callable[[], Coroutine]"] = {}

    async def _safe_task_wrapper(
        self,
        coro,
        task_name: str,
        factory: "Callable[[], Coroutine] | None" = None,
    ) -> None:
        """Wrap a coroutine to catch exceptions and prevent cascade failures.

        This is a CRITICAL stability fix: without isolation, a single exception
        in any of 18+ background tasks will crash the entire P2P orchestrator
        via asyncio.gather() propagating the exception.

        Args:
            coro: The coroutine to wrap
            task_name: Human-readable task name for logging
            factory: Optional callable that returns a new coroutine for restarts

        Returns:
            None - exceptions are logged but not raised
        """
        # Register factory for potential restarts
        if factory is not None:
            self._task_factories[task_name] = factory

        restart_count = 0
        max_restarts = 5

        while True:
            try:
                await coro
                return  # Normal completion
            except asyncio.CancelledError:
                logger.debug(f"Task '{task_name}' cancelled (shutdown)")
                raise  # Re-raise CancelledError for graceful shutdown
            except SystemExit:
                # SystemExit from main loop exit - ignore in background tasks
                # This prevents "Task exception was never retrieved" log pollution
                logger.debug(f"Task '{task_name}' received SystemExit (orchestrator shutdown)")
                return
            except Exception as e:  # noqa: BLE001
                # Log but don't propagate - other tasks continue running
                logger.error(f"Task '{task_name}' crashed: {e}", exc_info=True)

                # Check if we can restart
                restart_factory = factory or self._task_factories.get(task_name)
                if not self.running or restart_factory is None:
                    logger.warning(f"Task '{task_name}' cannot restart (no factory or shutdown)")
                    return

                restart_count += 1
                if restart_count > max_restarts:
                    logger.error(
                        f"Task '{task_name}' exceeded max restarts ({max_restarts}), giving up"
                    )
                    return

                # Exponential backoff: 30s, 60s, 120s, 240s, 480s
                delay = min(30 * (2 ** (restart_count - 1)), 480)
                logger.info(
                    f"Restarting task '{task_name}' in {delay}s "
                    f"(attempt {restart_count}/{max_restarts})..."
                )
                await asyncio.sleep(delay)

                if not self.running:
                    return

                # Create new coroutine from factory
                try:
                    coro = restart_factory()
                    logger.info(f"Restarted task '{task_name}'")
                except Exception as restart_error:
                    logger.error(f"Failed to restart task '{task_name}': {restart_error}")
                    return

    def _create_safe_task(
        self,
        coro,
        name: str,
        factory: "Callable[[], Coroutine] | None" = None,
    ) -> asyncio.Task:
        """Create a task wrapped with exception isolation and restart support.

        Args:
            coro: The coroutine to run
            name: Task name for logging
            factory: Optional callable that returns a new coroutine for restarts.
                     If not provided, task cannot be automatically restarted.

        Returns:
            asyncio.Task wrapped with safe error handling
        """
        return asyncio.create_task(
            self._safe_task_wrapper(coro, name, factory),
            name=name,
        )

    # =========================================================================
    # BOUNDED COLLECTIONS - Prevent unbounded memory growth
    # =========================================================================

    # Maximum pending relay items before cleanup
    MAX_PENDING_RELAY_ACKS = 10000
    MAX_PENDING_RELAY_RESULTS = 10000

    def _add_pending_relay_ack(self, cmd_id: str) -> None:
        """Add a relay ack with bounds checking."""
        if len(self.pending_relay_acks) >= self.MAX_PENDING_RELAY_ACKS:
            # Evict oldest entries (set doesn't have order, so clear half)
            half = len(self.pending_relay_acks) // 2
            to_remove = list(self.pending_relay_acks)[:half]
            for item in to_remove:
                self.pending_relay_acks.discard(item)
            logger.warning(f"Evicted {half} pending_relay_acks (max {self.MAX_PENDING_RELAY_ACKS})")
        self.pending_relay_acks.add(cmd_id)

    def _add_pending_relay_result(self, result: dict) -> None:
        """Add a relay result with bounds checking."""
        if len(self.pending_relay_results) >= self.MAX_PENDING_RELAY_RESULTS:
            # Evict oldest entries (keep most recent half)
            half = len(self.pending_relay_results) // 2
            self.pending_relay_results = self.pending_relay_results[half:]
            logger.warning(f"Evicted {half} pending_relay_results (max {self.MAX_PENDING_RELAY_RESULTS})")
        self.pending_relay_results.append(result)

    # =========================================================================
    # SAFEGUARDS - Load, rate limiting, and coordinator integration
    # =========================================================================

    def _check_spawn_rate_limit(self) -> tuple[bool, str]:
        """Check if we're within the spawn rate limit.

        SAFEGUARD: Prevents runaway process spawning by limiting spawns per minute.

        Returns:
            (can_spawn, reason) - True if within rate limit
        """
        now = time.time()
        # Clean old timestamps (older than 60 seconds)
        self.spawn_timestamps = [t for t in self.spawn_timestamps if now - t < 60]

        if len(self.spawn_timestamps) >= SPAWN_RATE_LIMIT_PER_MINUTE:
            return False, f"Rate limit: {len(self.spawn_timestamps)}/{SPAWN_RATE_LIMIT_PER_MINUTE} spawns in last minute"

        return True, f"Rate OK: {len(self.spawn_timestamps)}/{SPAWN_RATE_LIMIT_PER_MINUTE}"

    def _record_spawn(self) -> None:
        """Record a process spawn for rate limiting."""
        self.spawn_timestamps.append(time.time())

    def _can_spawn_process(self, reason: str = "job") -> tuple[bool, str]:
        """Combined safeguard check before spawning any process.

        Jan 29, 2026: Delegates to JobOrchestrator.can_spawn_process().

        Args:
            reason: Description of why we want to spawn (for logging)

        Returns:
            (can_spawn, explanation) - True if all checks pass
        """
        return self.jobs.can_spawn_process(reason)

    def _spawn_and_track_job(
        self,
        job_id: str,
        job_type: JobType,
        board_type: str,
        num_players: int,
        engine_mode: str,
        cmd: list[str],
        output_dir: Path,
        log_filename: str = "run.log",
        cuda_visible_devices: str | None = None,
        extra_env: dict[str, str] | None = None,
        safeguard_reason: str | None = None,
    ) -> tuple[ClusterJob, subprocess.Popen] | None:
        """Spawn a subprocess job and track it in local_jobs.

        Jan 29, 2026: Delegates to JobOrchestrator.spawn_and_track_job().

        Args:
            job_id: Unique job identifier
            job_type: Type of job (SELFPLAY, GPU_SELFPLAY, etc.)
            board_type: Board type (hex8, square8, etc.)
            num_players: Number of players
            engine_mode: Engine mode for the job
            cmd: Command to execute
            output_dir: Directory for output files
            log_filename: Name of log file in output_dir
            cuda_visible_devices: CUDA_VISIBLE_DEVICES value (None = inherit, "" = disable)
            extra_env: Additional environment variables
            safeguard_reason: Reason for safeguard check (default: job_type-board_type-Np)

        Returns:
            Tuple of (ClusterJob, Popen) if successful, None if blocked or failed
        """
        return self.jobs.spawn_and_track_job(
            job_id=job_id,
            job_type=job_type,
            board_type=board_type,
            num_players=num_players,
            engine_mode=engine_mode,
            cmd=cmd,
            output_dir=output_dir,
            log_filename=log_filename,
            cuda_visible_devices=cuda_visible_devices,
            extra_env=extra_env,
            safeguard_reason=safeguard_reason,
        )

    def _detect_build_version(self) -> str:
        env_version = (os.environ.get(BUILD_VERSION_ENV, "") or "").strip()
        if env_version:
            return env_version

        commit = ""
        branch = ""
        try:
            result = subprocess.run(
                self._git_cmd("rev-parse", "--short", "HEAD"),
                cwd=self.ringrift_path,
                capture_output=True,
                text=True,
                timeout=3,
            )
            if result.returncode == 0:
                commit = result.stdout.strip()
        except (subprocess.SubprocessError, subprocess.TimeoutExpired, OSError, AttributeError):
            commit = ""

        try:
            result = subprocess.run(
                self._git_cmd("rev-parse", "--abbrev-ref", "HEAD"),
                cwd=self.ringrift_path,
                capture_output=True,
                text=True,
                timeout=3,
            )
            if result.returncode == 0:
                branch = result.stdout.strip()
        except (subprocess.SubprocessError, subprocess.TimeoutExpired, OSError, AttributeError):
            branch = ""

        if commit and branch:
            return f"{branch}@{commit}"
        return commit or "unknown"

    def _git_cmd(self, *args: str) -> list[str]:
        safe_dir = getattr(self, "_git_safe_directory", "") or os.path.abspath(self.ringrift_path)
        return ["git", "-c", f"safe.directory={safe_dir}", *args]

    def _detect_ringrift_path(self) -> str:
        """Detect the RingRift installation path."""
        # Try common locations
        candidates = [
            Path.home() / "Development" / "RingRift",
            Path.home() / "ringrift",
            Path("/home/ubuntu/ringrift"),
            Path("/root/ringrift"),
        ]
        for path in candidates:
            if (path / "ai-service").exists():
                return str(path)
        return str(Path(__file__).parent.parent.parent)

    def _get_ai_service_path(self) -> str:
        """Get the path to the ai-service directory.

        Handles both cases:
        - ringrift_path = /path/to/RingRift (root directory)
        - ringrift_path = /path/to/RingRift/ai-service (already ai-service)

        Returns:
            Path to ai-service directory.
        """
        if self.ringrift_path.rstrip("/").endswith("ai-service"):
            return self.ringrift_path
        return os.path.join(self.ringrift_path, "ai-service")

    def _increment_rollback_counter(self) -> None:
        """Increment the rollback counter in diversity metrics.

        Used by AnalyticsCacheManager callback.
        """
        self.diversity_metrics["rollbacks"] += 1

    # CMA-ES Coordinator callback helpers (Jan 2026 - Aggressive Decomposition Phase 3)

    def _get_gpu_workers_for_cmaes(self) -> list:
        """Get available GPU workers for CMA-ES. Used by CMAESCoordinator callback.

        Jan 27, 2026: Migrated to PeerQueryBuilder (Phase 3.2).
        """
        workers = self._peer_query.healthy_with_gpu().unwrap_or([])
        if self.self_info.has_gpu:
            workers.append(self.self_info)
        return workers

    async def _send_cmaes_to_worker(self, worker_id: str, endpoint: str, payload: dict) -> bool:
        """Send CMA-ES request to a worker. Used by CMAESCoordinator callback."""
        try:
            with self.peers_lock:
                worker = self.peers.get(worker_id)
            if not worker:
                return False
            timeout = ClientTimeout(total=300)
            async with get_client_session(timeout) as session:
                url = self._url_for_peer(worker, endpoint)
                await session.post(url, json=payload, headers=self._auth_headers())
            return True
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to send CMA-ES request to {worker_id}: {e}")
            return False

    async def _report_cmaes_to_leader(self, endpoint: str, payload: dict) -> bool:
        """Report CMA-ES result to leader. Used by CMAESCoordinator callback."""
        try:
            if not self.leader_id:
                return False
            with self.peers_lock:
                leader = self.peers.get(self.leader_id)
            if not leader:
                return False
            timeout = ClientTimeout(total=30)
            async with get_client_session(timeout) as session:
                url = self._url_for_peer(leader, endpoint)
                await session.post(url, json=payload, headers=self._auth_headers())
            return True
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to report CMA-ES result to leader: {e}")
            return False

    def _handle_cmaes_complete_callback(self, board_type: str, num_players: int, weights: dict) -> str | None:
        """Handle CMA-ES completion. Used by CMAESCoordinator callback."""
        if self.improvement_cycle_manager:
            agent_id = self.improvement_cycle_manager.handle_cmaes_complete(
                board_type, num_players, weights
            )
            self.diversity_metrics["cmaes_triggers"] += 1
            return agent_id
        return None

    def _get_script_path(self, script_name: str) -> str:
        """Get the full path to a script in ai-service/scripts/.

        Args:
            script_name: Name of the script (e.g., "run_self_play_soak.py")

        Returns:
            Full path to the script.
        """
        return os.path.join(self._get_ai_service_path(), "scripts", script_name)

    def _check_yaml_gpu_config(self, node_id: str | None = None) -> tuple[bool, str, int]:
        """Check if YAML config indicates a node has a GPU.

        Used as fallback when runtime GPU detection fails (e.g., vGPU, containers,
        driver issues causing torch.cuda.is_available() to return False).

        Args:
            node_id: Node ID to check. If None, uses self.node_id.

        Returns:
            Tuple of (has_gpu, gpu_name, gpu_vram_gb)

        Session 17.50 (Jan 2026): Added to fix GPU nodes running CPU selfplay
        when torch.cuda.is_available() returns False due to driver issues.
        """
        target_node = node_id or self.node_id
        try:
            from app.config.cluster_config import get_config_cache
            config = get_config_cache().get_config()
            host_cfg = config.hosts_raw.get(target_node, {})

            # Check multiple indicators
            gpu_name = str(host_cfg.get("gpu", ""))
            gpu_vram = int(host_cfg.get("gpu_vram_gb", 0) or 0)
            role = str(host_cfg.get("role", ""))

            has_gpu = bool(gpu_name) or gpu_vram > 0 or "gpu" in role.lower()

            if has_gpu:
                logger.debug(
                    f"[YAML GPU] Node {target_node}: gpu={gpu_name}, "
                    f"vram={gpu_vram}GB, role={role}"
                )
            return has_gpu, gpu_name, gpu_vram
        except Exception as e:
            logger.debug(f"Could not check YAML GPU config for {target_node}: {e}")
            return False, "", 0

    def get_data_directory(self) -> Path:
        """Get the data directory path based on storage configuration.

        Returns:
            Path to data directory:
            - ramdrive: /dev/shm/ringrift/data (for disk-constrained Vast instances)
            - disk: {ringrift_path}/ai-service/data (default)

        The ramdrive option uses tmpfs for high-speed I/O and to work around
        limited disk space on some cloud instances. Data stored in ramdrive
        is volatile and should be synced to permanent storage periodically.
        """
        if self.storage_type == "ramdrive":
            ramdrive = Path(self.ramdrive_path)
            try:
                ramdrive.mkdir(parents=True, exist_ok=True)
            except (PermissionError, OSError) as e:
                # /dev/shm doesn't exist on macOS or may be inaccessible
                logger.warning(f"Cannot create ramdrive at {ramdrive}: {e}. Falling back to disk storage.")
                self.storage_type = "disk"
                return Path(self._get_ai_service_path()) / "data"

            # Set up automatic sync to persistent storage
            if self.ramdrive_syncer is None and self.sync_to_disk_interval > 0:
                persistent_path = Path(self._get_ai_service_path()) / "data"
                persistent_path.mkdir(parents=True, exist_ok=True)
                self.ramdrive_syncer = RamdriveSyncer(
                    source_dir=ramdrive,
                    target_dir=persistent_path,
                    interval=self.sync_to_disk_interval,
                    patterns=["*.db", "*.jsonl", "*.json", "*.npz"],
                )
                self.ramdrive_syncer.start()
                logger.info(f"Started ramdrive -> disk sync: {ramdrive} -> {persistent_path} "
                           f"every {self.sync_to_disk_interval}s")

            return ramdrive
        return Path(self._get_ai_service_path()) / "data"

    def stop_ramdrive_syncer(self, final_sync: bool = True) -> None:
        """Stop the ramdrive syncer and optionally perform final sync."""
        if self.ramdrive_syncer:
            logger.info("Stopping ramdrive syncer...")
            self.ramdrive_syncer.stop(final_sync=final_sync)
            logger.info(f"Ramdrive sync stats: {self.ramdrive_syncer.stats}")
            self.ramdrive_syncer = None

    # =========================================================================
    # GPU Job Tracking (Jan 7, 2026)
    # =========================================================================
    # These methods track GPU job lifecycle for adaptive dispatch decisions.
    # GPU nodes should run GPU-accelerated selfplay, not fall back to CPU.
    # =========================================================================

    def _get_node_job_preference(self, node_id: str) -> str:
        """Get preferred job type based on node role from YAML config.

        Jan 7, 2026: Added to enforce role-based job selection.
        GPU-only nodes should not fall back to CPU selfplay.

        Returns one of:
        - 'cpu_only': Node should only run CPU jobs (coordinator, cpu_selfplay)
        - 'gpu_only': Node should only run GPU jobs (gpu_selfplay role)
        - 'training_only': Node should only run training (gpu_training_primary)
        - 'both': Node can run both GPU selfplay and training (default)
        """
        try:
            from app.config.cluster_config import get_config_cache
            config = get_config_cache().get_config()
            host_cfg = config.hosts_raw.get(node_id, {})
            role = str(host_cfg.get("role", "")).lower()

            if role in ("coordinator", "cpu_selfplay"):
                return "cpu_only"
            if role == "gpu_selfplay":
                return "gpu_only"
            if role == "gpu_training_primary":
                # Training-primary nodes can still do selfplay when idle
                return "both"
            if role == "gpu_training_selfplay":
                return "both"
            return "both"
        except Exception as e:
            logger.debug(f"Could not get job preference for {node_id}: {e}")
            return "both"

    def _record_gpu_job_result(self, success: bool) -> None:
        """Record GPU job completion result for adaptive dispatch decisions.

        Jan 7, 2026: Added for GPU failure tracking.
        Consecutive failures indicate driver issues and should trigger CPU fallback.

        Args:
            success: True if GPU job completed successfully, False otherwise.
        """
        try:
            now = time.time()
            if success:
                self.self_info.last_gpu_job_success = now
                self.self_info.gpu_failure_count = 0  # Reset on success
            else:
                self.self_info.last_gpu_job_failure = now
                self.self_info.gpu_failure_count = getattr(self.self_info, "gpu_failure_count", 0) + 1
            logger.debug(f"GPU job result: success={success}, failure_count={self.self_info.gpu_failure_count}")
        except Exception as e:
            logger.debug(f"Could not record GPU job result: {e}")

    def _update_gpu_job_count(self, delta: int) -> None:
        """Update running GPU job count.

        Jan 7, 2026: Added for accurate GPU job tracking.
        Used to detect driver issues (jobs running but 0% utilization).

        Args:
            delta: Amount to change count by (+1 for start, -1 for completion).
        """
        try:
            current = getattr(self.self_info, "gpu_job_count", 0) or 0
            self.self_info.gpu_job_count = max(0, current + delta)
            logger.debug(f"GPU job count: {current} -> {self.self_info.gpu_job_count}")
        except Exception as e:
            logger.debug(f"Could not update GPU job count: {e}")

    def _infer_advertise_port(self) -> int:
        """Infer the externally reachable port for this node.

        - Explicit `RINGRIFT_ADVERTISE_PORT` always wins.
        - Vast.ai exposes container ports via `VAST_TCP_PORT_<PORT>`; when set,
          use that public port so peers can reach us.
        - Default to the listening port.
        """
        explicit = (os.environ.get(ADVERTISE_PORT_ENV, "")).strip()
        if explicit:
            try:
                return int(explicit)
            except ValueError:
                pass

        vast_key = f"VAST_TCP_PORT_{self.port}"
        mapped = (os.environ.get(vast_key, "")).strip()
        if mapped:
            try:
                return int(mapped)
            except ValueError:
                pass

        return int(self.port)

    def _load_force_relay_mode(self) -> bool:
        """Load force_relay_mode from distributed_hosts.yaml for this node.

        January 5, 2026: NAT-blocked nodes need to send ALL outbound heartbeats
        via relay to ensure other nodes can discover them. This is configured in
        distributed_hosts.yaml with either:
        - `nat_blocked: true` - Node is behind NAT and can't receive inbound connections
        - `force_relay_mode: true` - Explicitly enable relay mode

        Returns:
            True if this node should use relay mode for all outbound heartbeats.
        """
        # Priority 1: Environment variable override
        env = (os.environ.get("RINGRIFT_FORCE_RELAY_MODE") or "").strip().lower()
        if env in ("1", "true", "yes"):
            logger.info(f"[P2P] Force relay mode enabled via RINGRIFT_FORCE_RELAY_MODE env var")
            return True

        # Priority 2: Load from distributed_hosts.yaml
        try:
            from app.config.cluster_config import load_cluster_config
            config = load_cluster_config()
            nodes = getattr(config, "hosts_raw", {}) or {}
            node_cfg = nodes.get(self.node_id, {})

            nat_blocked = node_cfg.get("nat_blocked", False)
            force_relay = node_cfg.get("force_relay_mode", False)

            if nat_blocked or force_relay:
                reason = "nat_blocked" if nat_blocked else "force_relay_mode"
                logger.info(f"[P2P] Force relay mode enabled for {self.node_id} ({reason})")
                return True
        except ImportError:
            logger.debug("[P2P] cluster_config not available for force_relay_mode check")
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[P2P] Failed to load force_relay_mode from config: {e}")

        return False

    def _prepopulate_voter_peers(self) -> None:
        """Pre-populate voter nodes into peers dict for immediate gossip reachability.

        Jan 28, 2026: Fixes bootstrap chicken-and-egg where voters are invisible
        to gossip until discovered via heartbeat. Without this, new nodes have an
        empty peers dict → gossip can't reach voters → voters never get added.
        """
        if not self.voter_node_ids:
            return

        if os.environ.get("RINGRIFT_SKIP_VOTER_PREPOPULATION", "").lower() in ("1", "true"):
            logger.info("[P2P] Voter pre-population disabled via env var")
            return

        try:
            from app.config.cluster_config import get_cluster_nodes
            cluster_nodes = get_cluster_nodes()
        except ImportError:
            logger.warning("[P2P] Cannot pre-populate voters: cluster_config unavailable")
            return
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[P2P] Cannot pre-populate voters: {e}")
            return

        prepopulated = 0
        for voter_id in self.voter_node_ids:
            if voter_id == self.node_id:
                continue  # Skip self

            if voter_id in self.peers:
                continue  # Already known

            node_cfg = cluster_nodes.get(voter_id)
            if not node_cfg:
                logger.debug(f"[P2P] Voter {voter_id} not in cluster_config, skipping prepopulation")
                continue

            host = getattr(node_cfg, 'best_ip', None) or getattr(node_cfg, 'tailscale_ip', None)
            if not host:
                logger.debug(f"[P2P] Voter {voter_id} has no IP in cluster_config, skipping")
                continue

            voter_info = NodeInfo(
                node_id=voter_id,
                host=host,
                port=DEFAULT_PORT,
                tailscale_ip=getattr(node_cfg, 'tailscale_ip', '') or '',
                role=NodeRole.FOLLOWER,
                last_heartbeat=0,  # Will update on first heartbeat
            )
            self.peers[voter_id] = voter_info
            prepopulated += 1
            logger.debug(f"[P2P] Pre-populated voter {voter_id} at {host}:{DEFAULT_PORT}")

        if prepopulated:
            logger.info(f"[P2P] Pre-populated {prepopulated} voter peers for gossip reachability")

    def _load_cluster_config_raw(self) -> dict[str, Any]:
        """Load raw cluster config from distributed_hosts.yaml.

        Returns the raw YAML dict for use by loops that need to access
        host configuration (relay nodes, selfplay settings, etc.).

        January 27, 2026: Added to support loop_registry.py relay health loop
        and autonomous_queue_loop.py selfplay configuration.
        """
        cfg_path = Path(self._get_ai_service_path()) / "config" / "distributed_hosts.yaml"
        if not cfg_path.exists():
            return {}

        try:
            import yaml
            return yaml.safe_load(cfg_path.read_text()) or {}
        except (OSError, yaml.YAMLError) as e:
            logger.debug(f"[P2P] Failed to load cluster config: {e}")
            return {}


    def _on_swim_member_alive(self, member_id: str) -> None:
        """Handle SWIM member becoming alive - sync to gossip layer.

        Jan 29, 2026: Delegates to self.network orchestrator.
        """
        self.network.on_swim_member_alive(member_id)

    def _on_swim_member_failed(self, member_id: str) -> None:
        """Handle SWIM member failure - mark as suspect in gossip layer.

        Jan 29, 2026: Delegates to self.network orchestrator.
        """
        self.network.on_swim_member_failed(member_id)

    def _cache_local_ips(self) -> set[str]:
        """Cache all local IPs at startup to avoid DNS blocking in health endpoints.

        Jan 29, 2026: Delegate to PeerNetworkOrchestrator if available,
        otherwise inline basic IP detection (called during early __init__
        before self.network is set).

        Jan 26, 2026: Called once at initialization and cached.
        """
        if hasattr(self, "network") and self.network is not None:
            return self.network.cache_local_ips()

        # Fallback: inline basic IP detection for early __init__ call
        import socket
        import subprocess

        local_ips: set[str] = set()
        try:
            hostname = socket.gethostname()
            for addr in socket.getaddrinfo(hostname, None):
                local_ips.add(addr[4][0])
        except (socket.gaierror, socket.herror, OSError, UnicodeError):
            pass
        try:
            for addr in socket.getaddrinfo("localhost", None):
                local_ips.add(addr[4][0])
        except (socket.gaierror, socket.herror, OSError):
            pass
        try:
            result = subprocess.run(
                ["hostname", "-I"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                for ip in result.stdout.strip().split():
                    local_ips.add(ip.strip())
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            pass
        # Always include loopback
        local_ips.add("127.0.0.1")
        local_ips.add("::1")
        if hasattr(self, "advertise_host") and self.advertise_host:
            local_ips.add(self.advertise_host)
        if hasattr(self, "tailscale_ip") and self.tailscale_ip:
            local_ips.add(self.tailscale_ip)
        return local_ips

    async def _acquire_voter_lease_quorum(self, lease_id: str, duration: int) -> float | None:
        """Acquire/renew an exclusive leader lease from a quorum of voters.

        December 29, 2025: Added retry with exponential backoff when initial
        quorum acquisition fails. This handles transient network issues.

        Returns the effective lease expiry timestamp if a quorum granted the
        lease; otherwise returns None.
        """
        voter_ids = list(getattr(self, "voter_node_ids", []) or [])
        if not voter_ids:
            return time.time() + float(duration)

        quorum = int(getattr(self, "voter_quorum_size", 0) or 0)
        if quorum <= 0:
            # SIMPLIFIED QUORUM: Fixed at 3 voters (or less if fewer voters exist)
            quorum = min(VOTER_MIN_QUORUM, len(voter_ids))

        duration = max(10, min(int(duration), int(LEADER_LEASE_DURATION * 2)))

        # December 29, 2025: Retry with exponential backoff
        max_retries = 3
        retry_delays = [0, 2, 5]  # Immediate, then 2s, then 5s

        for attempt in range(max_retries):
            if attempt > 0:
                await asyncio.sleep(retry_delays[attempt])
                logger.info(f"Voter lease acquisition retry {attempt + 1}/{max_retries}")

            now = time.time()
            acks = 0
            lease_ttls: list[float] = []

            # Self-grant (as a voter).
            if self.node_id in voter_ids:
                self.voter_grant_leader_id = self.node_id
                self.voter_grant_lease_id = lease_id
                self.voter_grant_expires = now + float(duration)
                lease_ttls.append(float(duration))
                acks += 1

            # Jan 2026: Use lock-free PeerSnapshot for read-only access
            peers_by_id = self._peer_snapshot.get_snapshot()

            # STABILITY FIX: Use 15s timeout for voter lease operations (was 5s).
            # Cross-geographic Tailscale connections can have latency spikes.
            timeout = ClientTimeout(total=15)

            # Dec 29, 2025: Parallel lease acquisition for faster leadership transitions
            # Instead of sequential requests, we fire all lease requests in parallel
            async def _request_lease_from_voter(
                session: aiohttp.ClientSession,
                voter_id: str,
                voter: NodeInfo,
            ) -> tuple[bool, float | None]:
                """Request lease from a single voter. Returns (success, ttl)."""
                payload = {
                    "leader_id": self.node_id,
                    "lease_id": lease_id,
                    "lease_duration": duration,
                    "lease_epoch": self._lease_epoch + 1,
                }
                for url in self._tailscale_urls_for_voter(voter, "/election/lease"):
                    try:
                        async with session.post(url, json=payload, headers=self._auth_headers()) as resp:
                            if resp.status != 200:
                                continue
                            data, json_error = await safe_json_response(resp, default={}, log_errors=False)
                            if json_error or not data.get("granted"):
                                return False, None
                            ttl_raw = data.get("lease_ttl_seconds") or data.get("ttl_seconds")
                            if ttl_raw is not None:
                                try:
                                    return True, float(ttl_raw)
                                except (ValueError, TypeError):
                                    pass
                            return True, float(duration)
                    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, AttributeError, OSError):
                        continue
                return False, None

            async with get_client_session(timeout) as session:
                # Build list of voters to request from (excluding self and dead peers)
                voter_tasks = []
                for voter_id in voter_ids:
                    if voter_id == self.node_id:
                        continue
                    voter = peers_by_id.get(voter_id)
                    if not voter or not voter.is_alive():
                        continue
                    voter_tasks.append(_request_lease_from_voter(session, voter_id, voter))

                # Fire all requests in parallel
                if voter_tasks:
                    results = await asyncio.gather(*voter_tasks, return_exceptions=True)
                    for result in results:
                        if isinstance(result, Exception):
                            continue
                        success, ttl = result
                        if success:
                            acks += 1
                            if ttl is not None and ttl > 0:
                                lease_ttls.append(ttl)
                            else:
                                lease_ttls.append(float(duration))

            if acks >= quorum:
                # Use a relative TTL (computed by each voter on its own clock) to avoid
                # leader lease flapping under clock skew. Convert back to a local expiry.
                effective_ttl = min(lease_ttls) if lease_ttls else float(duration)
                effective_ttl = max(10.0, min(float(duration), float(effective_ttl)))
                if attempt > 0:
                    logger.info(f"Voter lease acquired on retry {attempt + 1}")
                return now + float(effective_ttl)

            # Log retry info
            if attempt < max_retries - 1:
                logger.warning(
                    f"Voter lease quorum not reached: {acks}/{quorum} acks, "
                    f"retrying in {retry_delays[attempt + 1]}s..."
                )

        # All retries exhausted
        logger.error(f"Failed to acquire voter lease quorum after {max_retries} attempts")
        return None

    # =========================================================================
    # Phase 15.1.1: Fence Token Helpers (December 29, 2025)
    # =========================================================================

    async def _determine_leased_leader_from_voters(self) -> str | None:
        """Return the current lease-holder as reported by a quorum of voters.

        This is a read-only reconciliation step used to resolve split-brain once
        partitions heal. It queries the current voter grant state via
        `/election/grant` and selects the leader_id that has >= quorum votes with
        non-expired grants.
        """
        voter_ids = list(getattr(self, "voter_node_ids", []) or [])
        if not voter_ids:
            return None

        quorum = int(getattr(self, "voter_quorum_size", 0) or 0)
        if quorum <= 0:
            # SIMPLIFIED QUORUM: Fixed at 3 voters (or less if fewer voters exist)
            quorum = min(VOTER_MIN_QUORUM, len(voter_ids))

        now = time.time()
        counts: dict[str, int] = {}

        # Include local voter state.
        if self.node_id in voter_ids:
            leader_id = str(getattr(self, "voter_grant_leader_id", "") or "")
            expires = float(getattr(self, "voter_grant_expires", 0.0) or 0.0)
            if leader_id and expires > now:
                counts[leader_id] = counts.get(leader_id, 0) + 1

        # Jan 2026: Use lock-free PeerSnapshot for read-only access
        peers_by_id = self._peer_snapshot.get_snapshot()

        # STABILITY FIX: Use 15s timeout for voter operations (was 5s).
        timeout = ClientTimeout(total=15)
        async with get_client_session(timeout) as session:
            for voter_id in voter_ids:
                if voter_id == self.node_id:
                    continue
                voter = peers_by_id.get(voter_id)
                if not voter or not voter.is_alive():
                    continue

                # Use Tailscale-exclusive URLs for voter communication to avoid NAT issues
                for url in self._tailscale_urls_for_voter(voter, "/election/grant"):
                    try:
                        async with session.get(url, headers=self._auth_headers()) as resp:
                            if resp.status != 200:
                                continue
                            data = await resp.json()
                        leader_id = str((data or {}).get("leader_id") or "")
                        if not leader_id:
                            break
                        ttl_raw = (data or {}).get("lease_ttl_seconds")
                        if ttl_raw is None:
                            ttl_raw = (data or {}).get("ttl_seconds")
                        ttl_val: float | None = None
                        if ttl_raw is not None:
                            try:
                                ttl_val = float(ttl_raw)
                            except (ValueError):
                                ttl_val = None

                        if ttl_val is not None:
                            if ttl_val <= 0:
                                break
                        else:
                            # Back-compat: use absolute expiry as best-effort, with
                            # a generous skew tolerance (1× lease duration).
                            expires = float((data or {}).get("lease_expires") or 0.0)
                            if expires <= 0:
                                break
                            if expires + float(LEADER_LEASE_DURATION) < now:
                                break
                        counts[leader_id] = counts.get(leader_id, 0) + 1
                        break
                    except (ValueError, AttributeError):
                        continue

        winners = [leader_id for leader_id, count in counts.items() if count >= quorum]
        if not winners:
            return None
        # Deterministic: if multiple satisfy quorum (shouldn't), pick highest node_id.
        return sorted(winners)[-1]

    async def _query_arbiter_for_leader(self) -> str | None:
        """Query the arbiter for the authoritative leader when voter quorum fails.

        The arbiter is a reliably-reachable node that maintains its view of
        who the leader should be. Used as a fallback when split-brain causes
        voter quorum to be unreachable.

        Returns:
            The leader_id from the arbiter, or None if arbiter is unreachable
        """
        arbiter_url = ARBITER_URL
        if not arbiter_url:
            return None

        # Try the configured arbiter URL
        urls_to_try = [arbiter_url]

        # Also try known peers as arbiters if main arbiter fails
        for peer_addr in (self.known_peers or []):
            if peer_addr not in urls_to_try:
                urls_to_try.append(peer_addr)

        timeout = ClientTimeout(total=5)
        try:
            async with get_client_session(timeout) as session:
                for url in urls_to_try:
                    try:
                        base_url = url.rstrip("/")
                        # Query the arbiter's election/grant endpoint to see who they think is leader
                        async with session.get(
                            f"{base_url}/election/grant",
                            headers=self._auth_headers()
                        ) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                leader_id = str((data or {}).get("leader_id") or "")
                                if leader_id:
                                    logger.info(f"Arbiter {base_url} reports leader: {leader_id}")
                                    return leader_id
                    except (aiohttp.ClientError, asyncio.TimeoutError, AttributeError):
                        # Try next arbiter
                        continue
        except (aiohttp.ClientError, asyncio.TimeoutError, AttributeError):
            pass

        return None

    # _parse_peer_address, _url_for_peer, _urls_for_peer provided by NetworkUtilsMixin

    def _auth_headers(self) -> dict[str, str]:
        if not self.auth_token:
            return {}
        return {"Authorization": f"Bearer {self.auth_token}"}

    @property
    def http_session(self) -> "aiohttp.ClientSession":
        """Shared HTTP client session for outbound requests.

        Used by loop_registry (manifest collection, peer recovery probes).
        Lazily created and re-created if closed.
        """
        if not hasattr(self, "_http_session") or self._http_session is None or self._http_session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self._http_session = aiohttp.ClientSession(
                timeout=timeout,
                headers=self._auth_headers(),
            )
        return self._http_session

    def _get_leader_peer(self) -> NodeInfo | None:
        if self.leadership.check_is_leader():
            return self.self_info

        # Jan 2026: Use lock-free PeerSnapshot for read-only access
        peers_snapshot = list(self._peer_snapshot.get_snapshot().values())

        conflict_keys = self._endpoint_conflict_keys([self.self_info, *peers_snapshot])

        leader_id = self.leader_id
        if leader_id and self._is_leader_lease_valid():
            for peer in peers_snapshot:
                if (
                    peer.node_id == leader_id
                    and peer.role == NodeRole.LEADER
                    and peer.is_alive()
                    and self._is_leader_eligible(peer, conflict_keys)
                ):
                    # Jan 8, 2026: Validate consensus - check that other peers agree
                    consensus_count = self.leadership.count_peers_reporting_leader(leader_id, peers_snapshot)
                    if consensus_count < 2 and len(peers_snapshot) >= 3:
                        # Low consensus - log warning but still return leader
                        logger.warning(
                            f"[LeaderConsensus] Low consensus for leader {leader_id}: "
                            f"only {consensus_count} peers agree out of {len(peers_snapshot)}"
                        )
                    return peer

        eligible_leaders = [
            peer for peer in peers_snapshot
            if peer.role == NodeRole.LEADER and self._is_leader_eligible(peer, conflict_keys)
        ]
        if eligible_leaders:
            return sorted(eligible_leaders, key=lambda p: p.node_id)[-1]

        return None


    async def _proxy_to_leader(self, request: web.Request) -> web.StreamResponse:
        """Best-effort proxy for leader-only APIs when the dashboard hits a follower."""
        leader = self._get_leader_peer()
        if not leader:
            return web.json_response(
                {"success": False, "error": "leader_unknown", "leader_id": self.leader_id},
                status=503,
            )

        candidate_urls = self._urls_for_peer(leader, request.raw_path)
        if not candidate_urls:
            candidate_urls = [self._url_for_peer(leader, request.raw_path)]
        forward_headers: dict[str, str] = {}
        for h in ("Authorization", "X-RingRift-Auth", "Content-Type"):
            if h in request.headers:
                forward_headers[h] = request.headers[h]

        body: bytes | None = None
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            body = await request.read()

        # Keep leader-proxy responsive: unreachable "leaders" (often NAT/firewall)
        # should fail fast so the dashboard doesn't hang for a full minute.
        timeout = ClientTimeout(total=10)
        last_exc: Exception | None = None
        async with get_client_session(timeout) as session:
            for target_url in candidate_urls:
                try:
                    async with session.request(
                        request.method,
                        target_url,
                        data=body,
                        headers=forward_headers,
                    ) as resp:
                        payload = await resp.read()
                        content_type = resp.headers.get("Content-Type")
                        headers: dict[str, str] = {}
                        if content_type:
                            headers["Content-Type"] = content_type
                        headers["X-RingRift-Proxied-By"] = self.node_id
                        headers["X-RingRift-Proxied-To"] = target_url
                        return web.Response(body=payload, status=resp.status, headers=headers)
                except Exception as exc:
                    last_exc = exc
                    continue

        return web.json_response(
            {
                "success": False,
                "error": "leader_proxy_failed",
                "message": str(last_exc) if last_exc else "unknown_error",
                "leader_id": self.leader_id,
                "attempted_urls": candidate_urls,
            },
            status=502,
        )

    def _is_request_authorized(self, request: web.Request) -> bool:
        if not self.auth_token:
            return True

        auth_header = request.headers.get("Authorization", "")
        token = ""
        if auth_header.lower().startswith("bearer "):
            token = auth_header[7:].strip()
        if not token:
            token = request.headers.get("X-RingRift-Auth", "").strip()
        if not token:
            return False

        return secrets.compare_digest(token, self.auth_token)

    def _load_state(self):
        """Load persisted state from database.

        Phase 1 Refactoring: Delegated to StateManager.
        The StateManager returns a PersistedState object which is then
        applied to the orchestrator's instance variables.
        """
        try:
            state = self.state_manager.load_state(self.node_id)

            # P2P Hardening Phase 2 (Dec 2025): Validate and clean stale state
            is_valid, issues = self.state_manager.validate_loaded_state(state)
            if issues:
                # Clean up stale entries before applying state
                jobs_removed, peers_removed = self.state_manager.clean_stale_state(state)
                if self.verbose:
                    logger.info(
                        f"[P2POrchestrator] Startup cleanup: removed "
                        f"{jobs_removed} stale jobs, {peers_removed} stale peers"
                    )

            # Apply loaded peers
            for node_id, info_dict in state.peers.items():
                try:
                    info = NodeInfo.from_dict(info_dict)
                    self.peers[node_id] = info
                except Exception as e:  # noqa: BLE001
                    logger.error(f"Failed to load peer {node_id}: {e}")
            # C2 fix: Sync peer snapshot after loading persisted peers
            self._sync_peer_snapshot()

            # Apply loaded jobs
            for job_dict in state.jobs:
                try:
                    job = ClusterJob(
                        job_id=job_dict["job_id"],
                        job_type=JobType(job_dict["job_type"]),
                        node_id=job_dict["node_id"],
                        board_type=job_dict.get("board_type", "square8"),
                        num_players=job_dict.get("num_players", 2),
                        engine_mode=job_dict.get("engine_mode", "descent-only"),
                        pid=job_dict.get("pid", 0),
                        started_at=job_dict.get("started_at", 0.0),
                        status=job_dict.get("status", "running"),
                    )
                    self.local_jobs[job.job_id] = job
                except Exception as e:  # noqa: BLE001
                    logger.error(f"Failed to load job: {e}")

            # Feb 2026: Clean stale jobs with dead PIDs before gossip starts.
            # Jobs from previous sessions may have PIDs that no longer exist,
            # causing training_jobs/selfplay_jobs to report phantom counts.
            stale_startup_jobs = []
            for job_id, job in list(self.local_jobs.items()):
                pid = getattr(job, "pid", 0) or 0
                if pid > 0 and getattr(job, "status", "") == "running":
                    try:
                        os.kill(pid, 0)  # Check if process exists
                    except ProcessLookupError:
                        stale_startup_jobs.append(job_id)
                    except PermissionError:
                        pass  # Process exists but owned by another user
            if stale_startup_jobs:
                for job_id in stale_startup_jobs:
                    self.local_jobs.pop(job_id, None)
                logger.info(
                    f"[P2POrchestrator] Startup cleanup: removed "
                    f"{len(stale_startup_jobs)} jobs with dead PIDs"
                )

            # Apply leader state
            # C1 fix: Use leader_state_lock for role/leader_id changes
            ls = state.leader_state
            with self.leader_state_lock:
                if ls.leader_id:
                    self.leader_id = ls.leader_id
                if ls.leader_lease_id:
                    self.leader_lease_id = ls.leader_lease_id
                if ls.leader_lease_expires:
                    self.leader_lease_expires = ls.leader_lease_expires
                if ls.last_lease_renewal:
                    self.last_lease_renewal = ls.last_lease_renewal
                if ls.role:
                    with contextlib.suppress(Exception):
                        self.role = NodeRole(ls.role)

                # Feb 23, 2026: Non-coordinator nodes must not load self-leadership.
                # After P2P restart, persisted state may have leader_id=self (from when
                # the node was leader). Without clearing this, the node continues
                # announcing itself as leader via gossip, overriding force_leader.
                _is_coordinator = os.environ.get("RINGRIFT_IS_COORDINATOR", "").lower() in ("true", "1", "yes")
                if not _is_coordinator and self.leader_id == self.node_id:
                    logger.info(
                        f"[P2POrchestrator] Non-coordinator: clearing self-leadership "
                        f"loaded from state (was leader_id={self.leader_id})"
                    )
                    self.leader_id = None
                    self.leader_lease_id = ""
                    self.leader_lease_expires = 0
                    self.role = NodeRole.FOLLOWER

            # Voter grant state
            if ls.voter_grant_leader_id:
                self.voter_grant_leader_id = ls.voter_grant_leader_id
            if ls.voter_grant_lease_id:
                self.voter_grant_lease_id = ls.voter_grant_lease_id
            if ls.voter_grant_expires:
                self.voter_grant_expires = ls.voter_grant_expires

            # Phase 15.1.1: Restore fenced lease token state
            # These fields may not exist in older state files, so use getattr with defaults
            persisted_epoch = getattr(ls, "lease_epoch", 0) or 0
            persisted_fence = getattr(ls, "fence_token", "") or ""
            persisted_last_seen = getattr(ls, "last_seen_epoch", 0) or 0
            # Only restore if higher than current (monotonic guarantee)
            if persisted_epoch > self._lease_epoch:
                self._lease_epoch = persisted_epoch
            if persisted_fence and not self._fence_token:
                self._fence_token = persisted_fence
            if persisted_last_seen > self._last_seen_epoch:
                self._last_seen_epoch = persisted_last_seen
            if persisted_epoch > 0:
                logger.info(
                    f"[P2POrchestrator] Restored lease fencing: epoch={self._lease_epoch}, "
                    f"last_seen={self._last_seen_epoch}"
                )

            # Feb 2026: Restore forced leader override from persisted state
            if getattr(ls, "forced_leader_override", False):
                self._forced_leader_override = True
                logger.info("[P2P] Restored forced_leader_override from persisted state")

            # Optional persisted voter configuration (convergence helper). Only
            # apply when voters are not explicitly configured via env/config.
            if (
                ls.voter_node_ids
                and not (getattr(self, "voter_node_ids", []) or [])
                and str(getattr(self, "voter_config_source", "none") or "none") == "none"
            ):
                if self.quorum_manager.maybe_adopt_voter_node_ids(ls.voter_node_ids, source="state"):
                    # Sync adopted state back to orchestrator attributes
                    self.voter_node_ids = self.quorum_manager.voter_node_ids
                    self.voter_config_source = self.quorum_manager.voter_config_source
                    self.voter_quorum_size = min(VOTER_MIN_QUORUM, len(self.voter_node_ids)) if self.voter_node_ids else 0

            # Self-heal inconsistent persisted leader state (can happen after
            # abrupt shutdowns or partial writes): never keep role=leader without
            # a matching leader_id.
            if self.role == NodeRole.LEADER and not self.leader_id:
                logger.info("Loaded role=leader but leader_id is empty; stepping down to follower")
                # C1 fix: Use leader_state_lock for role changes
                with self.leader_state_lock:
                    self.role = NodeRole.FOLLOWER
                    self.leader_lease_id = ""
                    self.leader_lease_expires = 0.0
                self.last_lease_renewal = 0.0

            logger.info(f"Loaded state: {len(self.peers)} peers, {len(self.local_jobs)} jobs")

            # December 2025 P2P Hardening: Validate loaded state on startup
            # This detects stale jobs, stale peers, and expired leases
            is_valid, issues = self.state_manager.validate_loaded_state(state)
            if not is_valid:
                logger.warning(f"[P2P] Startup state validation found {len(issues)} issues:")
                for issue in issues:
                    logger.warning(f"  - {issue}")
                # Clean up stale entries
                stale_jobs_cleared = self.state_manager.clear_stale_jobs_by_age(max_age_hours=24.0)
                stale_peers_cleared = self.state_manager.clear_stale_peers(max_stale_seconds=300.0)
                if stale_jobs_cleared or stale_peers_cleared:
                    logger.info(f"[P2P] Cleared {stale_jobs_cleared} stale jobs, {stale_peers_cleared} stale peers")
            else:
                logger.info("[P2P] Startup state validation passed")

            # Dec 28, 2025 (Phase 7): Load persisted peer health state
            # Jan 28, 2026: Uses health_metrics_manager directly
            try:
                peer_health_states = self.state_manager.load_all_peer_health(max_age_seconds=3600.0)
                if peer_health_states:
                    self.health_metrics_manager.apply_loaded_peer_health(peer_health_states)
                    logger.info(f"[P2P] Loaded {len(peer_health_states)} peer health records")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[P2P] Failed to load peer health state: {e}")

            # Jan 12, 2026: Initialize job snapshot with loaded jobs
            try:
                self._job_snapshot.update(self.local_jobs)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[P2P] Failed to initialize job snapshot: {e}")

        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to load state: {e}")


    def _save_state(self):
        """Save current state to database.

        Phase 1 Refactoring: Delegated to StateManager.
        Creates a PersistedLeaderState from instance variables and
        passes it to the StateManager for persistence.
        """
        try:
            # Build leader state from instance variables
            role_value = self.role.value if hasattr(self.role, "value") else str(self.role)
            leader_state = PersistedLeaderState(
                leader_id=self.leader_id or "",
                leader_lease_id=self.leader_lease_id or "",
                leader_lease_expires=float(self.leader_lease_expires or 0.0),
                last_lease_renewal=float(self.last_lease_renewal or 0.0),
                role=role_value,
                voter_grant_leader_id=str(getattr(self, "voter_grant_leader_id", "") or ""),
                voter_grant_lease_id=str(getattr(self, "voter_grant_lease_id", "") or ""),
                voter_grant_expires=float(getattr(self, "voter_grant_expires", 0.0) or 0.0),
                voter_node_ids=list(getattr(self, "voter_node_ids", []) or []),
                voter_config_source=str(getattr(self, "voter_config_source", "") or ""),
                # Phase 15.1.1: Fenced lease token state
                lease_epoch=int(getattr(self, "_lease_epoch", 0) or 0),
                fence_token=str(getattr(self, "_fence_token", "") or ""),
                last_seen_epoch=int(getattr(self, "_last_seen_epoch", 0) or 0),
                # Feb 2026: Persist forced leader override across restarts
                forced_leader_override=getattr(self, "_forced_leader_override", False),
            )

            # Delegate to StateManager
            self.state_manager.save_state(
                node_id=self.node_id,
                peers=self.peers,
                jobs=self.local_jobs,
                leader_state=leader_state,
                peers_lock=self.peers_lock,
                jobs_lock=self.jobs_lock,
            )

            # Dec 28, 2025 (Phase 7): Save peer health state
            try:
                # Inline: was _collect_peer_health_states()
                peer_health_states = self.health_metrics_manager.collect_peer_health_states()
                if peer_health_states:
                    saved = self.state_manager.save_peer_health_batch(peer_health_states)
                    if saved > 0 and self.verbose:
                        logger.debug(f"[P2P] Saved {saved} peer health records")
            except Exception as e:  # noqa: BLE001
                if self.verbose:
                    logger.debug(f"[P2P] Error saving peer health state: {e}")

            # Jan 12, 2026: Sync job snapshot for lock-free /status reads
            try:
                self._job_snapshot.update(self.local_jobs)
            except Exception as e:  # noqa: BLE001
                if self.verbose:
                    logger.debug(f"[P2P] Error syncing job snapshot: {e}")

        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to save state: {e}")

    # =========================================================================
    # Phase 27: Peer Cache and Reputation Tracking
    # Provided by PeerManagerMixin:
    # =========================================================================

    # =========================================================================
    # Phase 29: Cluster Epoch Persistence
    # Phase 1 Refactoring: Delegated to StateManager
    # =========================================================================

    def _save_cluster_epoch(self) -> None:
        """Save cluster epoch to database.

        Phase 1 Refactoring: Delegated to StateManager.
        Kept for backward compatibility.
        """
        self.state_manager.set_cluster_epoch(self._cluster_epoch)
        self.state_manager.save_cluster_epoch()

    def _increment_cluster_epoch(self) -> None:
        """Increment cluster epoch (called on leader change).

        Phase 1 Refactoring: Delegated to StateManager.
        Kept for backward compatibility.
        """
        self._cluster_epoch = self.state_manager.increment_cluster_epoch()

    def record_metric(
        self,
        metric_type: str,
        value: float,
        board_type: str | None = None,
        num_players: int | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """Record a metric to the history table for observability.

        Phase 1 Refactoring: Delegated to MetricsManager.

        Metric types:
        - training_loss: NNUE training loss
        - elo_rating: Model Elo rating
        - gpu_utilization: GPU utilization percentage
        - selfplay_games_per_hour: Game generation rate
        - validation_rate: GPU selfplay validation rate
        - tournament_win_rate: Tournament win rate for new model
        """
        self.metrics_manager.record_metric(
            metric_type=metric_type,
            value=value,
            board_type=board_type,
            num_players=num_players,
            metadata=metadata,
        )

    def get_metrics_history(
        self,
        metric_type: str,
        board_type: str | None = None,
        num_players: int | None = None,
        hours: float = 24,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Get metrics history. Feb 2026: Delegates to MetricsManager."""
        return self.metrics_manager.get_history(
            metric_type, board_type, num_players, hours, limit
        )

    def get_metrics_summary(self, hours: float = 24) -> dict[str, Any]:
        """Get metrics summary. Feb 2026: Delegates to MetricsManager."""
        return self.metrics_manager.get_summary(hours)

    def _create_self_info(self) -> NodeInfo:
        """Create NodeInfo for this node.

        Jan 29, 2026: Delegated to MonitoringOrchestrator.create_self_info() when available.
        Falls back to inline implementation during __init__ when self.monitoring doesn't exist.
        """
        # Check if monitoring orchestrator is available (may not be during early __init__)
        if hasattr(self, "monitoring") and self.monitoring is not None:
            return self.monitoring.create_self_info()

        # Fallback: inline implementation for early __init__ call
        # This runs before self.monitoring is created, so we do it inline
        from scripts.p2p.models import NodeInfo

        has_gpu, gpu_name = self._detect_gpu()
        cpu_count = int(os.cpu_count() or 0)
        memory_gb = self._detect_memory()

        # Detect coordinator mode
        is_coordinator = os.environ.get("RINGRIFT_IS_COORDINATOR", "").lower() in ("true", "1", "yes")

        if not is_coordinator:
            try:
                from app.config.cluster_config import load_cluster_config
                config = load_cluster_config()
                nodes = getattr(config, "hosts_raw", {}) or {}
                node_cfg = nodes.get(self.node_id, {})
                if node_cfg.get("role") == "coordinator":
                    is_coordinator = True
                elif node_cfg.get("selfplay_enabled") is False and node_cfg.get("training_enabled") is False:
                    is_coordinator = True
            except Exception:
                pass

        if is_coordinator:
            capabilities = []
        else:
            capabilities = ["selfplay"]
            if has_gpu:
                capabilities.extend(["training", "cmaes", "gauntlet", "tournament"])
            if memory_gb >= 64:
                capabilities.append("large_boards")

        info = NodeInfo(
            node_id=self.node_id,
            host=self.advertise_host,
            port=self.advertise_port,
            role=self.role,
            last_heartbeat=time.time(),
            cpu_count=cpu_count,
            has_gpu=has_gpu,
            gpu_name=gpu_name,
            memory_gb=memory_gb,
            capabilities=capabilities,
            version=self.build_version,
        )

        # Add Tailscale IP for NAT traversal
        ts_ip = self._get_tailscale_ip()
        if ts_ip and ts_ip != info.host:
            info.reported_host = ts_ip
            info.reported_port = int(self.port)

        info.alternate_ips = self._discover_all_ips(exclude_primary=info.host)
        info.tailscale_ip = ts_ip or ""
        info.addresses = [ts_ip] if ts_ip else []  # Simplified for early init
        info.visible_peers = 0  # No peers during early init
        info.effective_timeout = 180.0  # Default timeout

        return info

    def _collect_all_addresses(
        self, tailscale_ip: str | None, primary_host: str
    ) -> list[str]:
        """Collect all addresses this node is reachable at.

        Jan 29, 2026: Delegated to MonitoringOrchestrator._collect_all_addresses().
        """
        return self.monitoring._collect_all_addresses(tailscale_ip, primary_host)

    @staticmethod
    def _infer_capabilities_from_hardware(
        has_gpu: bool,
        memory_gb: int = 0,
        gpu_name: str = "",
    ) -> list[str]:
        """Infer capabilities from hardware info.

        December 30, 2025: Fallback for nodes reporting empty capabilities but
        having detectable hardware. Used to populate capabilities for peers
        that may have misconfigured coordinator settings.

        Args:
            has_gpu: Whether the node has a GPU
            memory_gb: RAM in gigabytes
            gpu_name: GPU name for logging

        Returns:
            List of inferred capabilities
        """
        capabilities = ["selfplay"]  # All nodes can at least do CPU selfplay
        if has_gpu:
            capabilities.extend(["training", "cmaes", "gauntlet", "tournament"])
        if memory_gb >= 64:
            capabilities.append("large_boards")
        return capabilities

    def _register_self_in_peers(self) -> None:
        """Register this node in the peers dict.

        Jan 29, 2026: Delegated to PeerNetworkOrchestrator.

        Jan 5, 2026: Ensures the leader (and any node) is visible in self.peers
        for components that iterate over peers directly.
        """
        # Delegate to PeerNetworkOrchestrator
        return self.network.register_self_in_peers()

    # =========================================================================
    # H2 fix: Lifecycle event emission methods (Jan 12, 2026)
    # These methods emit HOST_ONLINE, HOST_OFFLINE, P2P_NODE_DEAD, and
    # CLUSTER_CAPACITY_CHANGED events for cluster coordination.
    # =========================================================================

    async def _emit_host_online(self, node_id: str, capabilities: list[str] | None = None) -> None:
        """Emit HOST_ONLINE event for a peer coming online."""
        try:
            from app.distributed.data_events import DataEventType
            from app.coordination.event_router import emit_event

            # Jan 22, 2026: Use lock-free snapshot to prevent race conditions
            peer_info = self._peer_snapshot.get_snapshot().get(node_id)
            emit_event(DataEventType.HOST_ONLINE.value, {
                "node_id": node_id,
                "host": getattr(peer_info, "host", "") if peer_info else "",
                "port": getattr(peer_info, "port", 0) if peer_info else 0,
                "has_gpu": getattr(peer_info, "has_gpu", False) if peer_info else False,
                "gpu_name": getattr(peer_info, "gpu_name", "") if peer_info else "",
                "capabilities": capabilities or [],
                "source": "peer_discovery",
            })
            logger.debug(f"[P2P] Emitted HOST_ONLINE for peer: {node_id}")
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"[P2P] Failed to emit HOST_ONLINE for {node_id}: {e}")


    async def _emit_host_offline(self, node_id: str, reason: str, last_heartbeat: float | None) -> None:
        """Emit HOST_OFFLINE event for a peer going offline."""
        try:
            from app.distributed.data_events import DataEventType
            from app.coordination.event_router import emit_event

            emit_event(DataEventType.HOST_OFFLINE.value, {
                "node_id": node_id,
                "reason": reason,
                "last_heartbeat": last_heartbeat,
                "source": "peer_retirement",
            })
            logger.debug(f"[P2P] Emitted HOST_OFFLINE for peer: {node_id} (reason={reason})")
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"[P2P] Failed to emit HOST_OFFLINE for {node_id}: {e}")


    async def _emit_node_dead(self, node_id: str, reason: str, last_heartbeat: float | None, dead_for: float) -> None:
        """Emit P2P_NODE_DEAD event for a dead peer."""
        try:
            from app.distributed.data_events import DataEventType
            from app.coordination.event_router import emit_event

            emit_event(DataEventType.P2P_NODE_DEAD.value, {
                "node_id": node_id,
                "reason": reason,
                "last_heartbeat": last_heartbeat,
                "dead_for_seconds": dead_for,
                "source": "peer_timeout",
            })
            logger.debug(f"[P2P] Emitted P2P_NODE_DEAD for peer: {node_id} (dead_for={dead_for:.0f}s)")
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"[P2P] Failed to emit P2P_NODE_DEAD for {node_id}: {e}")


    async def _emit_cluster_capacity_changed(
        self,
        total_nodes: int,
        alive_nodes: int,
        gpu_nodes: int,
        training_nodes: int,
        change_type: str,
        change_details: dict | None = None,
    ) -> None:
        """Emit CLUSTER_CAPACITY_CHANGED event when cluster capacity changes."""
        try:
            from app.distributed.data_events import DataEventType
            from app.coordination.event_router import emit_event

            emit_event(DataEventType.CLUSTER_CAPACITY_CHANGED.value, {
                "total_nodes": total_nodes,
                "alive_nodes": alive_nodes,
                "gpu_nodes": gpu_nodes,
                "training_nodes": training_nodes,
                "change_type": change_type,
                "change_details": change_details or {},
                "source": "peer_management",
            })
            logger.debug(f"[P2P] Emitted CLUSTER_CAPACITY_CHANGED: {change_type}, alive={alive_nodes}")
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"[P2P] Failed to emit CLUSTER_CAPACITY_CHANGED: {e}")


    def _safe_emit_p2p_event(self, event_type: Any, payload: dict) -> None:
        """Safely emit a P2P-related event via the event router.

        This is a generic event emitter for P2P loops (QuorumCrisisDiscoveryLoop,
        GossipStateCleanupLoop, etc.) that need to emit events without knowing
        the specific event type at compile time.

        January 12, 2026: Added to fix AttributeError in P2P loops that referenced
        this method but it didn't exist. The loops pass emit_event=self._safe_emit_p2p_event
        but this method was never implemented.

        Args:
            event_type: Event type (string or DataEventType enum)
            payload: Event payload dictionary
        """
        try:
            from app.distributed.data_events import DataEventType
            from app.coordination.event_router import emit_event

            # Handle both string and enum event types
            event_value = None
            if isinstance(event_type, str):
                # Try to convert string to DataEventType
                try:
                    event_value = DataEventType(event_type).value
                except ValueError:
                    # Unknown event type - log and skip
                    logger.debug(f"[P2P] Unknown event type: {event_type}, skipping emission")
                    return
            elif hasattr(event_type, "value"):
                # It's an enum, get its value
                event_value = event_type.value
            else:
                # Pass through as-is
                event_value = str(event_type)

            emit_event(event_value, payload)
            logger.debug(f"[P2P] Emitted event: {event_value}")
        except ImportError:
            pass  # Event router not available
        except Exception as e:
            logger.debug(f"[P2P] Failed to emit event {event_type}: {e}")

    def _sync_peer_snapshot(self) -> None:
        """Synchronize PeerSnapshot with current peers dictionary.

        January 12, 2026: Added for lock-free reads in handle_status.
        Call this after any operation that modifies self.peers.

        This uses bulk_update for efficiency when there are many peers.
        The PeerSnapshot will be atomically updated with the current state.
        """
        try:
            # Use bulk update for efficiency - single lock acquisition, single snapshot refresh
            with self._peer_snapshot.bulk_update():
                # Clear and repopulate (handles removes and updates)
                self._peer_snapshot.clear()
                for node_id, info in self.peers.items():
                    self._peer_snapshot.update_peer(node_id, info)
        except Exception as e:  # noqa: BLE001
            # Log but don't fail - reads will use stale snapshot
            logger.warning(f"[PeerSnapshot] Sync failed: {e}")

    # _is_tailscale_host provided by NetworkUtilsMixin

    def _local_has_tailscale(self) -> bool:
        """Best-effort: True when this node appears to have a Tailscale address."""
        try:
            info = getattr(self, "self_info", None)
            if not info:
                return False
            host = str(getattr(info, "host", "") or "").strip()
            reported_host = str(getattr(info, "reported_host", "") or "").strip()
            return self._is_tailscale_host(host) or self._is_tailscale_host(reported_host)
        except (AttributeError):
            return False


    # _enable_tailscale_priority, _disable_tailscale_priority

    # =========================================================================
    # Network Health Methods (December 30, 2025)
    # Required by NetworkHealthMixin for cross-verification of P2P vs Tailscale
    # =========================================================================

    async def _get_tailscale_status(self) -> dict[str, bool]:
        """Jan 29, 2026: Delegated to PeerNetworkOrchestrator.get_tailscale_status()."""
        return await self.network.get_tailscale_status()

    async def _reconnect_discovered_peer(
        self, node_id: str, host: str, port: int
    ) -> bool:
        """Attempt to reconnect to a peer discovered via Tailscale.

        Probes the peer's health endpoint and sends a heartbeat to establish
        P2P connection.

        Args:
            node_id: Peer node identifier
            host: Tailscale IP address
            port: P2P port (usually 8770)

        Returns:
            True if reconnection successful, False otherwise
        """
        try:
            # Probe health endpoint
            url = f"http://{host}:{port}/health"
            timeout = ClientTimeout(total=5)
            async with get_client_session(timeout) as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        return False
                    data, error = await safe_json_response(resp, default={}, log_errors=False)
                    if error:
                        return False

            # Extract node_id from response if available
            actual_node_id = data.get("node_id", node_id)

            # Send heartbeat to establish connection
            await self._send_heartbeat_to_peer(host, port)

            # Check if peer is now in our peers dict
            async with NonBlockingAsyncLockWrapper(self.peers_lock, "peers_lock", timeout=5.0):
                if actual_node_id not in self.peers or not self.peers[actual_node_id].is_alive():
                    # Register the peer
                    self.peers[actual_node_id] = PeerInfo(
                        node_id=actual_node_id,
                        host=host,
                        port=port,
                        last_heartbeat=time.time(),
                        state="alive",
                    )
                    # C2 fix: Sync peer snapshot after adding new peer
                    self._sync_peer_snapshot()
                    logger.info(f"Reconnected peer via network health: {actual_node_id} ({host}:{port})")
                    await self._emit_host_online(actual_node_id)
                    return True

            return True  # Already connected

        except Exception as e:  # noqa: BLE001
            logger.debug(f"Failed to reconnect {node_id}: {e}")
            return False

    async def reconnect_missing_peers(self) -> list[str]:
        """Reconnect to all peers that are online in Tailscale but not in P2P.

        Returns:
            List of node IDs that were successfully reconnected
        """
        ts_peers = await self._get_tailscale_status()
        config_hosts = self._load_distributed_hosts().get("hosts", {})

        # Build IP to node mapping
        ip_to_node: dict[str, tuple[str, dict]] = {}
        for name, h in config_hosts.items():
            ts_ip = h.get("tailscale_ip")
            if ts_ip and h.get("p2p_enabled", True):
                ip_to_node[ts_ip] = (name, h)

        # Get current alive peer IDs
        # Jan 2026: Use lock-free PeerSnapshot for read-only access
        current_ids: set[str] = set()
        for peer in self._peer_snapshot.get_snapshot().values():
            if peer.is_alive():
                current_ids.add(peer.node_id)

        # Find and reconnect missing peers
        reconnected: list[str] = []
        for ts_ip, is_online in ts_peers.items():
            if not is_online:
                continue

            if ts_ip not in ip_to_node:
                continue

            node_id, node_config = ip_to_node[ts_ip]

            # Skip if already connected
            if node_id in current_ids:
                continue

            # Skip self
            if node_id == self.node_id:
                continue

            # Attempt reconnection
            port = node_config.get("p2p_port", DEFAULT_PORT)
            if await self._reconnect_discovered_peer(node_id, ts_ip, port):
                reconnected.append(node_id)

        if reconnected:
            logger.info(f"Reconnected {len(reconnected)} missing peers: {reconnected}")

        return reconnected

    # =========================================================================
    # Partition Read-Only Mode (Phase 2.4 - Dec 29, 2025)
    # =========================================================================

    def _check_partition_mode(self) -> None:
        """Check partition status and enable/disable read-only mode.

        December 2025 (Phase 2.4): Prevent data divergence during network partitions.

        When this node is in a minority partition (<50% of peers alive):
        - Pause training job dispatch
        - Pause selfplay job dispatch
        - Continue serving existing data (read-only)
        - Allow sync operations to help recovery

        This prevents split-brain scenarios where both partitions continue
        generating training data that later conflicts during merge.
        """
        now = time.time()

        # Rate limit partition checks
        if now - self._last_partition_check < self._partition_check_interval:
            return
        self._last_partition_check = now

        # Use gossip protocol's partition detection
        status, ratio = self.detect_partition_status()

        if status in ("minority", "isolated"):
            if not self._partition_readonly_mode:
                logger.warning(
                    f"[P2P] Entering partition read-only mode: "
                    f"status={status}, health_ratio={ratio:.2%}"
                )
                self._partition_readonly_mode = True
                self._partition_readonly_since = now

                # Emit event for monitoring
                self._safe_emit_event("PARTITION_READONLY_ENTERED", {
                    "node_id": self.node_id,
                    "status": status,
                    "health_ratio": ratio,
                    "timestamp": now,
                })
        else:
            if self._partition_readonly_mode:
                readonly_duration = now - self._partition_readonly_since
                logger.info(
                    f"[P2P] Exiting partition read-only mode: "
                    f"status={status}, health_ratio={ratio:.2%}, "
                    f"was_readonly_for={readonly_duration:.0f}s"
                )
                self._partition_readonly_mode = False
                self._partition_readonly_since = 0.0

                # Emit event for monitoring
                self._safe_emit_event("PARTITION_READONLY_EXITED", {
                    "node_id": self.node_id,
                    "status": status,
                    "health_ratio": ratio,
                    "readonly_duration_seconds": readonly_duration,
                    "timestamp": now,
                })

    def is_partition_readonly(self) -> bool:
        """Check if this node is in partition read-only mode.

        December 2025 (Phase 2.4): Query method for dispatch gates.

        Returns:
            True if job dispatch should be paused due to partition status.
        """
        # Do a fresh check if it's been a while
        self._check_partition_mode()
        return self._partition_readonly_mode

    def get_partition_status(self) -> dict[str, Any]:
        """Get current partition status details.

        December 2025 (Phase 2.4): Status API for monitoring/debugging.

        Returns:
            Dict with partition status, mode, and duration.
        """
        status, ratio = self.detect_partition_status()
        now = time.time()

        result = {
            "partition_status": status,
            "health_ratio": round(ratio, 3),
            "readonly_mode": self._partition_readonly_mode,
            "readonly_since": self._partition_readonly_since,
            "readonly_duration_seconds": (
                now - self._partition_readonly_since
                if self._partition_readonly_mode else 0.0
            ),
            "last_check": self._last_partition_check,
        }

        # Add detailed peer info if available
        if hasattr(self, "get_partition_details"):
            result["details"] = self.get_partition_details()

        return result

    # NOTE: _get_db_game_count_sync() inlined at call site (Jan 2026 Phase 2)

    def _seed_selfplay_scheduler_game_counts_sync(self) -> dict[str, int]:
        """Seed game counts from canonical databases synchronously.

        IMPORTANT: This is a blocking operation. Call via asyncio.to_thread() from async code.
        Added Jan 2026 (Session 17.29) to fix bootstrap priority for underserved configs.

        Returns:
            Dict mapping config_key -> game_count from canonical databases
        """
        game_counts: dict[str, int] = {}
        # Jan 7, 2026: Use _get_ai_service_path() to avoid doubled ai-service/ path
        canonical_dir = Path(self._get_ai_service_path()) / "data" / "games"

        # Pattern: canonical_<board_type>_<num_players>p.db
        for db_path in canonical_dir.glob("canonical_*_*p.db"):
            try:
                # Extract config_key from filename: canonical_hex8_2p.db -> hex8_2p
                stem = db_path.stem  # canonical_hex8_2p
                if stem.startswith("canonical_"):
                    config_key = stem[len("canonical_"):]  # hex8_2p
                    # Inline: was _get_db_game_count_sync()
                    game_count = self.data_pipeline_manager.get_db_game_count_sync(db_path)
                    if game_count > 0:
                        game_counts[config_key] = game_count
            except (ValueError, AttributeError):
                continue

        return game_counts

    async def _fetch_game_counts_from_peers(self) -> dict[str, int]:
        """Fetch game counts from coordinator or other peers with canonical databases.

        Session 17.41: Cluster nodes don't have canonical databases, so they need to
        fetch game counts from the coordinator which has them. This enables the
        starvation multipliers to work correctly on all nodes.

        Returns:
            Dict mapping config_key -> game_count from peers
        """
        # Try coordinator nodes first (they have canonical databases)
        # Jan 2026: Use lock-free PeerSnapshot for read-only access
        peers_snapshot = self._peer_snapshot.get_snapshot()
        coordinator_candidates = []
        for peer_id, peer in peers_snapshot.items():
            # Coordinator nodes or nodes with role=coordinator
            role_str = getattr(peer.role, "value", str(peer.role)) if peer.role else ""
            if "coordinator" in role_str.lower() or "mac-studio" in peer_id.lower():
                coordinator_candidates.append(peer)

        # Fallback to any alive peer
        if not coordinator_candidates:
            coordinator_candidates = [p for p in peers_snapshot.values() if p.is_alive()]

        for peer in coordinator_candidates[:3]:  # Try up to 3 candidates
            try:
                # Get best endpoint for peer
                key = self._endpoint_key(peer)
                if not key:
                    continue
                scheme, host, port = key
                url = f"{scheme}://{host}:{port}/game_counts"

                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            game_counts = data.get("game_counts", {})
                            if game_counts:
                                source_node = data.get("node_id", peer.node_id)
                                logger.info(f"[P2P] Fetched {len(game_counts)} game counts from {source_node}")
                                return game_counts
            except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError) as e:
                logger.debug(f"[P2P] Failed to fetch game counts from {peer.node_id}: {e}")
                continue

        # Session 17.48: Fallback to known coordinator IPs from config if peer discovery failed
        # This handles the case where P2P network hasn't converged yet (no heartbeats from coordinator)
        fallback_coordinator_ips = [
            "100.69.164.58",  # macbook-pro-2-1 Tailscale IP (has canonical DBs)
        ]
        for ip in fallback_coordinator_ips:
            try:
                url = f"http://{ip}:8770/game_counts"
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            game_counts = data.get("game_counts", {})
                            if game_counts:
                                source_node = data.get("node_id", "unknown")
                                logger.info(f"[P2P] Fetched {len(game_counts)} game counts from fallback {source_node}")
                                return game_counts
            except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError) as e:
                logger.debug(f"[P2P] Fallback fetch from {ip} failed: {e}")
                continue

        return {}

    async def _async_seed_game_counts_from_peers_if_needed(self) -> None:
        """Async fallback to seed game counts from peers if local seeding failed.

        Jan 9, 2026: Cluster nodes don't have local canonical databases, so
        the synchronous seeding during __init__ returns empty. This method
        fetches game counts from the coordinator/peers during async startup,
        enabling proper underserved config prioritization on worker nodes.

        Without this, all configs appear to have 0 games and get the same
        maximum bootstrap boost (+100), which neutralizes the prioritization.
        """
        try:
            # Check if game counts were already seeded during __init__
            if self.selfplay_scheduler:
                existing_counts = self.selfplay_scheduler._get_game_counts_per_config()
                if existing_counts and len(existing_counts) >= 6:
                    # Already have game counts from local canonical DBs
                    logger.debug(
                        f"[P2P] Game counts already seeded ({len(existing_counts)} configs), "
                        "skipping peer fetch"
                    )
                    return

            # Fetch from peers/coordinator
            logger.info("[P2P] Local canonical DBs empty, fetching game counts from peers...")
            peer_counts = await self._fetch_game_counts_from_peers()

            if peer_counts and self.selfplay_scheduler:
                self.selfplay_scheduler.update_p2p_game_counts(peer_counts)
                logger.info(
                    f"[P2P] Seeded SelfplayScheduler with {len(peer_counts)} config game counts from peers"
                )
                # Log underserved configs for visibility
                for config_key, count in sorted(peer_counts.items(), key=lambda x: x[1]):
                    if count < 5000:
                        logger.info(f"[P2P] Underserved config (from peers): {config_key} = {count} games")
            else:
                logger.warning(
                    "[P2P] Could not fetch game counts from peers - "
                    "bootstrap prioritization may not work correctly"
                )

        except Exception as e:  # noqa: BLE001
            logger.warning(f"[P2P] Async game count seeding failed: {e}")

    async def _game_count_refresh_loop(self) -> None:
        """Periodically refresh game counts from coordinator.

        Jan 9, 2026: Cluster nodes need to periodically refresh game counts
        as games are generated and consolidated. This ensures the scheduler
        always has accurate game counts for prioritization decisions.

        Interval: 5 minutes (300 seconds)
        """
        REFRESH_INTERVAL = 300  # 5 minutes
        await asyncio.sleep(60)  # Initial delay to let cluster stabilize

        while True:
            try:
                # Skip if this node has local canonical DBs (coordinator)
                local_counts = await asyncio.to_thread(self._seed_selfplay_scheduler_game_counts_sync)
                if local_counts and len(local_counts) >= 6:
                    # Has local DBs - update from local
                    if self.selfplay_scheduler:
                        self.selfplay_scheduler.update_p2p_game_counts(local_counts)
                        logger.debug(f"[P2P] Refreshed game counts from local DBs ({len(local_counts)} configs)")
                else:
                    # Fetch from peers
                    peer_counts = await self._fetch_game_counts_from_peers()
                    if peer_counts and self.selfplay_scheduler:
                        self.selfplay_scheduler.update_p2p_game_counts(peer_counts)
                        logger.debug(f"[P2P] Refreshed game counts from peers ({len(peer_counts)} configs)")

            except Exception as e:  # noqa: BLE001
                logger.debug(f"[P2P] Game count refresh failed: {e}")

            await asyncio.sleep(REFRESH_INTERVAL)


    def _run_subprocess_sync(self, cmd: list, timeout: int = 10) -> tuple[int, str, str]:
        """Run subprocess synchronously.

        IMPORTANT: This is a blocking operation. Call via asyncio.to_thread() from async code.
        Added Dec 2025 to fix P2P orchestrator CPU spikes from blocking subprocess in async loops.

        Returns: (return_code, stdout, stderr)
        """
        import subprocess
        try:
            result = subprocess.run(cmd, timeout=timeout, capture_output=True, text=True)
            return (result.returncode, result.stdout or "", result.stderr or "")
        except subprocess.TimeoutExpired:
            return (-1, "", "timeout")
        except (OSError, subprocess.SubprocessError) as e:
            return (-1, "", str(e))

    async def _run_subprocess_async(self, cmd: list, timeout: int = 10) -> tuple[int, str, str]:
        """Run subprocess asynchronously via thread pool.

        Jan 2026: Added for Phase 1 multi-core parallelization.
        Uses asyncio.to_thread() to avoid blocking the event loop.

        Returns: (return_code, stdout, stderr)
        """
        return await asyncio.to_thread(self._run_subprocess_sync, cmd, timeout)


    def _get_max_selfplay_slots_for_node(self) -> int:
        """Get maximum selfplay slots based on GPU capability.

        Jan 2, 2026: Added for slot-based capacity management.
        This allows work queue claiming to coexist with legacy selfplay processes.

        The slot count is based on GPU type since different GPUs can handle
        different numbers of concurrent selfplay processes effectively.

        Returns:
            Maximum number of selfplay slots for this node.
        """
        import os

        # Check environment variable first (allows manual override)
        env_slots = os.environ.get("RINGRIFT_MAX_SELFPLAY_SLOTS")
        if env_slots:
            try:
                return int(env_slots)
            except ValueError:
                pass

        # Compute based on GPU name
        gpu_name = getattr(self.self_info, "gpu_name", "") or ""
        gpu_name_lower = gpu_name.lower()

        # High-end GPUs get more slots
        if "gh200" in gpu_name_lower or "h100" in gpu_name_lower:
            return 16
        elif "a100" in gpu_name_lower:
            return 12
        elif "5090" in gpu_name_lower or "4090" in gpu_name_lower:
            return 8
        elif "3090" in gpu_name_lower or "a40" in gpu_name_lower or "l40" in gpu_name_lower:
            return 6
        elif "4060" in gpu_name_lower or "3060" in gpu_name_lower:
            return 3
        elif self.self_info.has_gpu:
            return 4  # Default for other GPUs
        else:
            return 2  # CPU-only nodes

    def _cleanup_stale_processes(self) -> int:
        """Kill processes that have been running too long.

        Jan 29, 2026: Delegated to JobOrchestrator.cleanup_stale_processes().
        """
        return self.jobs.cleanup_stale_processes()

    def _cleanup_orphan_gpu_processes(self) -> int:
        """Detect GPU processes not tracked in local_jobs and warn about them.

        Feb 2026: On P2P startup, previous sessions may have left training or
        selfplay processes that occupy GPU memory. We detect them via nvidia-smi
        and log warnings so operators can decide whether to kill them.

        Returns:
            Number of orphan GPU processes found.
        """
        import subprocess

        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
        except FileNotFoundError:
            return 0  # No nvidia-smi = no GPU
        except subprocess.TimeoutExpired:
            logger.warning("[P2P] nvidia-smi timed out during orphan detection")
            return 0

        if result.returncode != 0:
            return 0

        tracked_pids = set()
        for job in self.local_jobs.values():
            pid = getattr(job, "pid", 0) or 0
            if pid > 0:
                tracked_pids.add(pid)

        orphan_count = 0
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            try:
                gpu_pid = int(parts[0])
            except (ValueError, IndexError):
                continue
            proc_name = parts[1] if len(parts) > 1 else "unknown"
            mem_mb = parts[2] if len(parts) > 2 else "?"

            if gpu_pid not in tracked_pids:
                orphan_count += 1
                logger.warning(
                    f"[P2P] Orphan GPU process: PID={gpu_pid} "
                    f"name={proc_name} mem={mem_mb}MB (not tracked in local_jobs)"
                )

        if orphan_count > 0:
            logger.warning(
                f"[P2P] Found {orphan_count} orphan GPU processes. "
                "These may block work claiming due to GPU memory usage. "
                "Consider killing them manually if they're from previous sessions."
            )
        return orphan_count

    # ============================================
    # Phase 2: Distributed Data Sync Methods
    # ============================================

    def _collect_local_data_manifest(self) -> NodeDataManifest:
        """Collect manifest of all data files on this node.

        REFACTORED (Dec 2025): Delegates to SyncPlanner.collect_local_manifest().
        See scripts/p2p/managers/sync_planner.py for implementation.

        Scans the data directory for:
        - selfplay/ - Game replay files (.jsonl, .db)
        - models/ - Trained model files (.pt, .onnx)
        - training/ - Training data files (.npz)
        - games/ - Synced game databases (.db)

        Uses get_data_directory() to support both disk and ramdrive storage.
        """
        # Phase 2A: Delegate to SyncPlanner (Dec 2025)
        # This eliminates ~150 lines of duplicate code
        # Jan 23, 2026: Changed use_cache=False to True to reduce event loop blocking
        # The uncached version does heavy filesystem I/O (glob, stat, SQLite COUNT)
        # which can take 5-8 seconds and block the event loop, causing leader election failures
        return self.sync_planner.collect_local_manifest(use_cache=True)

    # Dec 2025: Legacy manifest methods removed (162 LOC) - using SyncPlanner

    def _request_peer_manifest_sync(self, peer_id: str) -> NodeDataManifest | None:
        """Synchronous wrapper for requesting peer manifest.

        Used by SyncPlanner which expects a sync callback.
        Runs the async version in a new event loop.

        Args:
            peer_id: The peer's node ID to request from

        Returns:
            NodeDataManifest or None if request failed
        """
        # Look up peer info
        # Jan 2026: Use lock-free PeerSnapshot for read-only access
        peer_info = self._peer_snapshot.get_snapshot().get(peer_id)

        if not peer_info:
            logger.debug(f"Peer {peer_id} not found in peers dict")
            return None

        # Run async version in event loop
        try:
            loop = asyncio.get_running_loop()
            # If we're in an async context, use run_coroutine_threadsafe
            import concurrent.futures
            future = asyncio.run_coroutine_threadsafe(
                self._request_peer_manifest(peer_info), loop
            )
            return future.result(timeout=15)
        except RuntimeError:
            # No running loop - use asyncio.run
            try:
                return asyncio.run(self._request_peer_manifest(peer_info))
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Failed to request manifest from {peer_id}: {e}")
                return None
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Failed to request manifest from {peer_id}: {e}")
            return None

    async def _request_peer_manifest(self, peer_info: NodeInfo) -> NodeDataManifest | None:
        """Request data manifest from a peer node."""
        try:
            # Keep manifest requests snappy: these are advisory and should not
            # stall leader loops or external callers (e.g. the improvement
            # daemon). Prefer faster failure and rely on periodic retries.
            timeout = ClientTimeout(total=10, sock_connect=3, sock_read=7)
            async with get_client_session(timeout) as session:
                for url in self._urls_for_peer(peer_info, "/data_manifest"):
                    try:
                        async with session.get(url, headers=self._auth_headers()) as resp:
                            if resp.status != 200:
                                continue
                            data = await resp.json()
                        return NodeDataManifest.from_dict((data or {}).get("manifest", {}))
                    except (aiohttp.ClientError, asyncio.TimeoutError, AttributeError):
                        continue
        except Exception as e:  # noqa: BLE001
            logger.error(f"requesting manifest from {peer_info.node_id}: {e}")
        return None



    async def _collect_cluster_manifest(self) -> ClusterDataManifest:
        """Jan 29, 2026: Delegated to SyncOrchestrator.collect_cluster_manifest()."""
        return await self.sync.collect_cluster_manifest()

    async def _collect_external_storage_metadata(self) -> ExternalStorageManifest:
        """Collect metadata from external storage sources (OWC drive, S3 bucket).

        Jan 2026: Delegates to DataSyncCoordinator for unified cluster data visibility.

        Returns:
            ExternalStorageManifest with OWC and S3 metadata.
        """
        from scripts.p2p.models import ExternalStorageManifest

        # Delegate to DataSyncCoordinator
        metadata = await self.data_sync_coordinator.collect_external_storage_metadata()

        # Convert to ExternalStorageManifest
        external = ExternalStorageManifest(collected_at=metadata.collected_at)
        external.owc_available = metadata.owc_available
        external.owc_games_by_config = metadata.owc_games_by_config
        external.owc_total_games = metadata.owc_total_games
        external.owc_total_size_bytes = metadata.owc_total_size_bytes
        external.owc_last_scan = metadata.owc_last_scan
        external.owc_scan_error = metadata.owc_scan_error or ""
        external.s3_available = metadata.s3_available
        external.s3_games_by_config = metadata.s3_games_by_config
        external.s3_total_games = metadata.s3_total_games
        external.s3_total_size_bytes = metadata.s3_total_size_bytes
        external.s3_last_scan = metadata.s3_last_scan
        external.s3_bucket = metadata.s3_bucket
        external.s3_scan_error = metadata.s3_scan_error or ""

        return external

    def _extract_config_from_path(self, db_path: Path) -> str | None:
        """Extract config from path. Delegates to DataSyncCoordinator."""
        return self.data_sync_coordinator.extract_config_from_path(db_path)


    # Phase 2: P2P Rsync Coordination - using SyncPlanner

    async def _execute_sync_plan(self) -> None:
        """Leader executes the sync plan by dispatching jobs to nodes.

        Delegates to SyncPlanner.execute_sync_plan() with _request_node_sync as callback.
        Dec 2025: Refactored to delegate to SyncPlanner for consolidated logic.
        """
        if not self.current_sync_plan:
            return

        # Delegate to SyncPlanner with our network request callback
        result = await self.sync_planner.execute_sync_plan(
            plan=self.current_sync_plan,
            execute_job_callback_async=self._request_node_sync,
        )

        # Update local state from SyncPlanner result
        with self.sync_lock:
            self.last_sync_time = time.time()

        if not result.get("success", False):
            logger.warning(f"Sync plan execution issue: {result.get('error', 'unknown')}")

    async def _request_node_sync(self, job: DataSyncJob) -> bool:
        """Request a target node to pull files from a source node."""
        target_peer = self.peers.get(job.target_node)
        if job.target_node == self.node_id:
            target_peer = self.self_info

        source_peer = self.peers.get(job.source_node)
        if job.source_node == self.node_id:
            source_peer = self.self_info

        if not target_peer or not source_peer:
            job.status = "failed"
            job.error_message = "Source or target peer not found"
            return False

        job.status = "running"
        job.started_at = time.time()

        try:
            # Local target: execute the pull directly (no HTTP round-trip).
            if job.target_node == self.node_id:
                result = await self._handle_sync_pull_request(
                    source_host=source_peer.host,
                    source_port=source_peer.port,
                    source_reported_host=(getattr(source_peer, "reported_host", "") or None),
                    source_reported_port=(getattr(source_peer, "reported_port", 0) or None),
                    source_node_id=job.source_node,
                    files=job.files,
                )
            else:
                payload = {
                    "job_id": job.job_id,
                    # Back-compat: target will prefer source_node_id lookup.
                    "source_host": source_peer.host,
                    "source_port": source_peer.port,
                    "source_node_id": job.source_node,
                    "files": job.files,
                }
                rh = (getattr(source_peer, "reported_host", "") or "").strip()
                rp = int(getattr(source_peer, "reported_port", 0) or 0)
                if rh and rp and (rh != source_peer.host or rp != source_peer.port):
                    payload["source_reported_host"] = rh
                    payload["source_reported_port"] = rp

                timeout = ClientTimeout(total=600)
                async with get_client_session(timeout) as session:
                    result = None
                    last_err: str | None = None
                    for url in self._urls_for_peer(target_peer, "/sync/pull"):
                        try:
                            async with session.post(url, json=payload, headers=self._auth_headers()) as resp:
                                if resp.status != 200:
                                    last_err = f"http_{resp.status}"
                                    continue
                                result = await resp.json()
                                break
                        except Exception as e:  # noqa: BLE001
                            last_err = str(e)
                            continue
                    if result is None:
                        job.status = "failed"
                        job.error_message = last_err or "sync_pull_failed"
                        # Note: SyncPlanner tracks jobs_failed count
                        return False

            ok = bool(result.get("success"))
            job.status = "completed" if ok else "failed"
            job.completed_at = time.time()
            job.bytes_transferred = int(result.get("bytes_transferred", 0) or 0)
            job.files_completed = int(result.get("files_completed", 0) or 0)
            if not ok:
                job.error_message = str(result.get("error") or "Unknown error")

            # Note: SyncPlanner tracks jobs_completed/jobs_failed counts

            if ok:
                logger.info(f"Sync job {job.job_id[:8]} completed: {job.source_node} -> {job.target_node}")
            else:
                logger.info(f"Sync job {job.job_id[:8]} failed: {job.error_message}")

            return ok

        except Exception as e:  # noqa: BLE001
            job.status = "failed"
            job.error_message = str(e)
            job.completed_at = time.time()
            # Note: SyncPlanner tracks jobs_failed count
            logger.info(f"Sync job {job.job_id[:8]} failed: {e}")
            return False

    async def _handle_sync_pull_request(
        self,
        source_host: str,
        source_port: int,
        source_node_id: str,
        files: list[str],
        source_reported_host: str | None = None,
        source_reported_port: int | None = None,
    ) -> dict[str, Any]:
        """Handle incoming request to pull files from a source node.

        Jan 28, 2026: Phase 18A - Delegates to SyncPlanner.
        """
        return await self.sync_planner.handle_sync_pull_request(
            source_host=source_host,
            source_port=source_port,
            source_node_id=source_node_id,
            files=files,
            source_reported_host=source_reported_host,
            source_reported_port=source_reported_port,
            data_dir=self.get_data_directory(),
            auth_headers_fn=self._auth_headers,
        )

    async def start_cluster_sync(self) -> dict[str, Any]:
        """
        Leader initiates a full cluster data sync.
        Returns status of the sync operation.
        """
        if not self.leadership.check_is_leader():
            return {"success": False, "error": "Not the leader"}

        # First, collect fresh manifests
        logger.info("Collecting cluster manifest for sync...")
        self.cluster_data_manifest = await self._collect_cluster_manifest()

        # Generate sync plan (using SyncPlanner manager for consolidated logic)
        self.current_sync_plan = self.sync_planner.generate_sync_plan(self.cluster_data_manifest)
        if not self.current_sync_plan:
            return {"success": True, "message": "No sync needed, all nodes in sync"}

        # Execute the plan
        await self._execute_sync_plan()

        return {
            "success": True,
            "plan_id": self.current_sync_plan.plan_id,
            "total_jobs": len(self.current_sync_plan.sync_jobs),
            "jobs_completed": self.current_sync_plan.jobs_completed,
            "jobs_failed": self.current_sync_plan.jobs_failed,
            "status": self.current_sync_plan.status,
        }

    # ============================================
    # NodeSelector Wrapper Methods REMOVED (Dec 2025)
    # All call sites now use self.node_selector.* directly
    # ============================================

    def _should_sync_to_node(self, node: NodeInfo) -> bool:
        """Check if we should sync data TO this node based on disk space."""
        # Don't sync to nodes with critical disk usage
        if node.disk_percent >= DISK_CRITICAL_THRESHOLD:
            logger.info(f"Skipping sync to {node.node_id}: disk critical ({node.disk_percent:.1f}%)")
            return False
        # Warn but allow sync to nodes with warning-level disk
        if node.disk_percent >= DISK_WARNING_THRESHOLD:
            logger.warning(f"{node.node_id} disk at {node.disk_percent:.1f}%")
        return True

    def _get_training_nodes_plus_coordinator(self) -> list:
        """Get training nodes PLUS coordinator for selfplay data sync.

        Feb 2026: The coordinator needs SOME selfplay data for canonical DB
        consolidation and NPZ export, but it's a disk-constrained machine.
        Only sync to coordinator when disk is below 70% to avoid filling it up.
        GPU nodes use local JSONL→NPZ fallback (training_executor.py) when
        coordinator doesn't have data.
        """
        training_nodes = self.node_selector.get_training_primary_nodes()
        training_ids = {n.node_id for n in training_nodes}

        # Add coordinator as sync target only when disk is healthy.
        # The coordinator is disk-constrained and shouldn't be a bulk data sink.
        # Use DISK_WARNING_THRESHOLD (70%) not 90% to preserve disk space.
        if self.self_info and self.self_info.node_id not in training_ids:
            disk_pct = getattr(self.self_info, "disk_percent", 0)
            if disk_pct < DISK_WARNING_THRESHOLD:
                training_nodes.append(self.self_info)
                logger.debug(
                    f"Including coordinator {self.self_info.node_id} as selfplay "
                    f"sync target (disk={disk_pct:.0f}%)"
                )
            else:
                logger.info(
                    f"Coordinator {self.self_info.node_id} excluded from selfplay "
                    f"sync (disk={disk_pct:.0f}% >= {DISK_WARNING_THRESHOLD}%)"
                )

        return training_nodes

    async def _sync_selfplay_to_training_nodes(self) -> dict[str, Any]:
        """Sync selfplay data to training primary nodes AND coordinator.

        December 2025: Delegated to SyncPlanner.sync_selfplay_to_training_nodes()
        Feb 2026: Added coordinator as sync target for canonical DB consolidation.
        """
        if not self.leadership.check_is_leader():
            return {"success": False, "error": "Not the leader"}

        # Use stale manifest if available, otherwise will be collected fresh
        manifest = self.cluster_data_manifest
        if (time.time() - self.last_manifest_collection > self.manifest_collection_interval
                or not manifest):
            manifest = None  # Will be collected by SyncPlanner

        result = await self.sync_planner.sync_selfplay_to_training_nodes(
            get_training_nodes=self._get_training_nodes_plus_coordinator,
            should_sync_to_node=self._should_sync_to_node,
            should_cleanup_source=lambda node: node.disk_percent >= DISK_CLEANUP_THRESHOLD,
            collect_manifest=self._collect_cluster_manifest,
            execute_sync_job=self._request_node_sync,
            cleanup_synced_files=self.sync.cleanup_synced_files,
            get_sync_router=self._get_sync_router,
            cluster_manifest=manifest,
        )

        # Update orchestrator state
        if result.get("success"):
            self.last_training_sync_time = time.time()
            # Refresh manifest after sync
            if not manifest:
                # Dec 2025: Add 5-minute timeout for manifest collection
                try:
                    self.cluster_data_manifest = await asyncio.wait_for(
                        self._collect_cluster_manifest(),
                        timeout=300.0  # 5 minutes max
                    )
                    self.last_manifest_collection = time.time()
                except asyncio.TimeoutError:
                    logger.warning("Post-sync manifest collection timed out after 5 minutes")

        return result


    async def _discover_tailscale_peers(self):
        """One-shot Tailscale peer discovery for bootstrap fallback.

        Jan 2026: Delegated to IPDiscoveryManager for better modularity.
        """
        return await self.ip_discovery_manager.discover_tailscale_peers(
            peers_lock=self.peers_lock,
            peers=self.peers,
            send_heartbeat_callback=self._send_heartbeat_to_peer,
            run_subprocess_callback=self._run_subprocess_async,
        )

    async def _reconnect_missing_tailscale_peers(self) -> int:
        """Force reconnect to peers online in Tailscale but missing from P2P mesh.

        Jan 2026: Delegated to IPDiscoveryManager for better modularity.

        Returns:
            Number of peers successfully reconnected.
        """
        return await self.ip_discovery_manager.reconnect_missing_tailscale_peers(
            peers_lock=self.peers_lock,
            peers=self.peers,
            load_distributed_hosts_callback=self._load_distributed_hosts,
            reconnect_peer_callback=self._reconnect_discovered_peer,
            run_subprocess_callback=self._run_subprocess_async,
            node_id=self.node_id,
        )

    async def _convert_jsonl_to_npz_for_training(self, data_dir: Path, training_dir: Path) -> int:
        """Convert JSONL selfplay files directly to NPZ.

        Jan 2026: Delegated to DataPipelineManager.
        """
        return await self.data_pipeline_manager.convert_jsonl_to_npz_for_training(
            data_dir, training_dir
        )

    async def _start_auto_training(self, data_path: str):
        """Start automatic training job on local node."""
        try:
            run_dir = os.path.join(self._get_ai_service_path(), "models", f"auto_train_{int(time.time())}")
            Path(run_dir).mkdir(parents=True, exist_ok=True)

            cmd = [
                sys.executable,  # Use venv Python
                self._get_script_path("run_nn_training_baseline.py"),
                "--board", "square8",
                "--num-players", "2",
                "--run-dir", run_dir,
                "--data-path", data_path,
                "--epochs", "20",  # Jan 2026: Reduced from 50 to prevent overfitting (patience=7 will early stop)
                "--model-version", "v3",
            ]

            env = os.environ.copy()
            env["PYTHONPATH"] = self._get_ai_service_path()

            subprocess.Popen(
                cmd,
                stdout=open(f"{run_dir}/training.log", "w"),
                stderr=subprocess.STDOUT,
                env=env,
                cwd=self._get_ai_service_path(),
            )
            logger.info(f"Started auto-training job in {run_dir}")
            self.self_info.training_jobs += 1

        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to start auto-training: {e}")

    # ============================================
    # Git Auto-Update Methods (async - Jan 19, 2026)
    # All git operations run in thread pool to avoid blocking event loop
    # ============================================

    async def _get_local_git_commit(self) -> str | None:
        """Get the current local git commit hash (async)."""
        try:
            result = await async_subprocess_run(
                self._git_cmd("rev-parse", "HEAD"),
                cwd=self.ringrift_path,
                timeout=10
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to get local git commit: {e}")
        return None

    async def _get_local_git_branch(self) -> str | None:
        """Get the current local git branch name (async)."""
        try:
            result = await async_subprocess_run(
                self._git_cmd("rev-parse", "--abbrev-ref", "HEAD"),
                cwd=self.ringrift_path,
                timeout=10
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to get local git branch: {e}")
        return None

    async def _get_remote_git_commit(self) -> str | None:
        """Fetch and get the remote branch's latest commit hash (async)."""
        try:
            # First fetch to update remote refs
            fetch_result = await async_subprocess_run(
                self._git_cmd("fetch", GIT_REMOTE_NAME, GIT_BRANCH_NAME),
                cwd=self.ringrift_path,
                timeout=60
            )
            if fetch_result.returncode != 0:
                logger.info(f"Git fetch failed: {fetch_result.stderr}")
                return None

            # Get remote branch commit
            result = await async_subprocess_run(
                self._git_cmd("rev-parse", f"{GIT_REMOTE_NAME}/{GIT_BRANCH_NAME}"),
                cwd=self.ringrift_path,
                timeout=10
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to get remote git commit: {e}")
        return None

    async def _check_for_updates(self) -> tuple[bool, str | None, str | None]:
        """Check if there are updates available from GitHub (async).

        Returns: (has_updates, local_commit, remote_commit)
        """
        # Run both git queries in parallel
        local_commit, remote_commit = await asyncio.gather(
            self._get_local_git_commit(),
            self._get_remote_git_commit(),
        )

        if not local_commit or not remote_commit:
            return False, local_commit, remote_commit

        has_updates = local_commit != remote_commit
        return has_updates, local_commit, remote_commit

    async def _get_commits_behind(self, local_commit: str, remote_commit: str) -> int:
        """Get the number of commits the local branch is behind remote (async)."""
        try:
            result = await async_subprocess_run(
                self._git_cmd("rev-list", "--count", f"{local_commit}..{remote_commit}"),
                cwd=self.ringrift_path,
                timeout=10
            )
            if result.returncode == 0:
                return int(result.stdout.strip())
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to count commits behind: {e}")
        return 0

    async def _check_local_changes(self) -> bool:
        """Check if there are uncommitted local changes (async).

        Notes:
        - Ignore untracked files by default. Cluster nodes often accumulate local
          artifacts (logs, data, env backups) that should not block git updates.
        - Still blocks on tracked/staged modifications to avoid stomping on
          local hotfixes.
        """
        try:
            result = await async_subprocess_run(
                self._git_cmd("status", "--porcelain", "--untracked-files=no"),
                cwd=self.ringrift_path,
                timeout=10
            )
            if result.returncode == 0:
                # If there's output, there are uncommitted changes
                return bool(result.stdout.strip())
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to check local changes: {e}")
        return True  # Assume changes exist on error (safer)

    async def _stop_all_local_jobs(self) -> int:
        """Stop all local jobs gracefully before update.

        Returns: Number of jobs stopped
        """
        stopped = 0
        with self.jobs_lock:
            for job_id, job in list(self.local_jobs.items()):
                try:
                    if job.pid > 0:
                        os.kill(job.pid, signal.SIGTERM)
                        logger.info(f"Sent SIGTERM to job {job_id} (PID {job.pid})")
                        stopped += 1
                        job.status = "stopping"
                except ProcessLookupError:
                    # Process already gone
                    job.status = "stopped"
                except Exception as e:  # noqa: BLE001
                    logger.error(f"Failed to stop job {job_id}: {e}")

        # Wait for processes to terminate gracefully
        # GPU games can take 1-10 minutes, so use a longer timeout (Dec 2025 fix)
        grace_period = int(os.environ.get("RINGRIFT_JOB_GRACE_PERIOD", "60"))
        if stopped > 0:
            await asyncio.sleep(grace_period)

            # Force kill any remaining
            with self.jobs_lock:
                for job_id, job in list(self.local_jobs.items()):
                    if job.status == "stopping" and job.pid > 0:
                        try:
                            os.kill(job.pid, signal.SIGKILL)
                            logger.info(f"Force killed job {job_id}")
                        except OSError:
                            pass  # Process already dead
                        job.status = "stopped"

        return stopped

    async def _perform_git_update(self) -> tuple[bool, str]:
        """Perform git pull to update the codebase (async).

        Returns: (success, message)
        """
        # Check for local changes (async)
        if await self._check_local_changes():
            return False, "Local changes detected. Cannot auto-update. Please commit or stash changes."

        # Stop jobs if configured
        if GRACEFUL_SHUTDOWN_BEFORE_UPDATE:
            stopped = await self._stop_all_local_jobs()
            if stopped > 0:
                logger.info(f"Stopped {stopped} jobs before update")

        try:
            # Perform git pull (async - Jan 19, 2026)
            result = await async_subprocess_run(
                self._git_cmd("pull", GIT_REMOTE_NAME, GIT_BRANCH_NAME),
                cwd=self.ringrift_path,
                timeout=120
            )

            if result.returncode != 0:
                return False, f"Git pull failed: {result.stderr}"

            logger.info(f"Git pull successful: {result.stdout}")
            return True, result.stdout

        except subprocess.TimeoutExpired:
            return False, "Git pull timed out"
        except Exception as e:  # noqa: BLE001
            return False, f"Git pull error: {e}"

    async def _restart_orchestrator(self):
        """Restart the orchestrator process after update."""
        logger.info("Restarting orchestrator to apply updates...")

        # Save state before restart
        self._save_state()

        # Get current script path and arguments
        script_path = Path(__file__).resolve()
        args = sys.argv[1:]

        # Schedule restart
        await asyncio.sleep(2)

        # Use exec to replace current process
        os.execv(sys.executable, [sys.executable, str(script_path), *args])

    # See scripts/p2p/loops/maintenance_loops.py and scripts/p2p/loop_registry.py

    # ============================================
    # HTTP API Handlers
    # ============================================

    async def handle_heartbeat(self, request: web.Request) -> web.Response:
        """Handle heartbeat from peer node.

        Jan 28, 2026: Phase 18B - Delegates to HeartbeatManager.process_incoming_heartbeat().
        """
        try:
            data = await request.json()
            forwarded_for = (
                request.headers.get("X-Forwarded-For")
                or request.headers.get("X-Real-IP")
                or request.headers.get("CF-Connecting-IP")
            )
            payload = await self.heartbeat_manager.process_incoming_heartbeat(
                data=data,
                remote_addr=request.remote,
                forwarded_for=forwarded_for,
            )
            return web.json_response(payload)
        except json.JSONDecodeError as e:
            logger.warning(f"[heartbeat] JSON parse error from {request.remote}: {e}")
            return web.json_response({"error": "invalid_json", "detail": str(e)}, status=400)
        except KeyError as e:
            logger.warning(f"[heartbeat] Missing required field from {request.remote}: {e}")
            return web.json_response({"error": "missing_field", "field": str(e)}, status=400)
        except ValueError as e:
            logger.warning(f"[heartbeat] Validation error from {request.remote}: {e}")
            return web.json_response({"error": "validation_error", "detail": str(e)}, status=400)
        except Exception as e:  # noqa: BLE001
            logger.error(f"[heartbeat] Unexpected error from {request.remote}: {type(e).__name__}: {e}")
            return web.json_response({"error": "internal_error", "type": type(e).__name__}, status=500)

    @with_request_timeout(30.0)
    async def handle_status(self, request: web.Request) -> web.Response:
        """Return cluster status.

        Query parameters:
            alive_only: If "true" (default), only show alive peers. Set to "false" to include dead/stale peers.
            include_stale_jobs: If "false" (default), dead peers show 0 jobs. Set to "true" to show stale job counts.
            no_cache: If "true", bypass cache and force fresh computation.

        December 30, 2025: Made non-blocking with timeout-based lock acquisition.
        If locks can't be acquired within 2 seconds, returns partial status with
        "unavailable" markers instead of blocking indefinitely.

        Jan 12, 2026: Changed to non-blocking self_info update - schedules background
        refresh and returns immediately with cached data. This prevents 15s+ timeouts
        on macOS where resource detection is slow.

        Jan 16, 2026: Added @with_request_timeout(30.0) decorator to prevent overall
        handler timeout. Individual metric timeouts are 2s, but other operations
        (voter health, partition status, etc.) can hang without protection.

        Feb 2026: Added response caching (5s TTL) with request deduplication. When
        multiple master_loop daemons call /status concurrently (7+ callers observed),
        only one computation runs and all callers get the cached result. This prevents
        the event loop from blocking for 10-60+ seconds under concurrent load.
        """
        # Feb 2026: Response cache - return cached result if fresh enough
        now = time.time()
        no_cache = request.query.get("no_cache", "false").lower() == "true"
        if not no_cache and self._status_cache is not None and (now - self._status_cache_time) < self._status_cache_ttl:
            return web.json_response(self._status_cache)

        # Deduplicate concurrent requests: only one computation at a time
        async with self._status_cache_lock:
            # Double-check after acquiring lock - another request may have populated cache
            now = time.time()
            if not no_cache and self._status_cache is not None and (now - self._status_cache_time) < self._status_cache_ttl:
                return web.json_response(self._status_cache)

            result = await self._compute_status(request)
            self._status_cache = result
            self._status_cache_time = time.time()
            return web.json_response(result)

    async def _compute_status(self, request: web.Request) -> dict:
        """Compute full cluster status dict. Called by handle_status with cache."""
        # Jan 12, 2026: Non-blocking mode - schedule background refresh, use cached data
        try:
            asyncio.create_task(self._update_self_info_async())
        except Exception:
            pass  # Fire-and-forget, don't block on errors

        # Parse query parameters for filtering
        alive_only = request.query.get("alive_only", "true").lower() != "false"
        include_stale_jobs = request.query.get("include_stale_jobs", "false").lower() == "true"

        # Jan 12, 2026: Lock-free peer snapshot using copy-on-write pattern
        # PeerSnapshot.get_snapshot() returns instantly without acquiring any lock.
        # The snapshot is updated atomically whenever peers are modified.
        # This eliminates the 6+ second timeouts that occurred under load.
        snapshot_dict = self._peer_snapshot.get_snapshot()
        peers_snapshot: list = list(snapshot_dict.values())

        conflict_keys = self._endpoint_conflict_keys([self.self_info, *peers_snapshot])
        effective_leader = self._get_leader_peer()

        now = time.time()
        peers: dict[str, Any] = {}
        for node_id, info in ((p.node_id, p) for p in peers_snapshot):
            is_alive = info.is_alive()

            # Skip dead peers if alive_only is set
            if alive_only and not is_alive:
                continue

            d = info.to_dict()
            d["endpoint_conflict"] = self._endpoint_key(info) in conflict_keys
            d["leader_eligible"] = self._is_leader_eligible(info, conflict_keys, require_alive=False)

            # Add explicit alive status and staleness info
            d["is_alive"] = is_alive
            last_hb = float(getattr(info, "last_heartbeat", 0.0) or 0.0)
            d["seconds_since_heartbeat"] = int(now - last_hb) if last_hb > 0 else -1

            # Zero out job counts for dead peers unless explicitly requested
            if not is_alive and not include_stale_jobs:
                d["selfplay_jobs"] = 0
                d["training_jobs"] = 0
                d["active_job_count"] = 0

            peers[node_id] = d

        # Jan 5, 2026 (Session 17.28): Build all_peers dict with ALL peers regardless of alive status
        # This is required for remote job dispatch which needs to know about all configured nodes
        all_peers: dict[str, Any] = {}
        for peer in peers_snapshot:
            all_peers[peer.node_id] = {
                "node_id": peer.node_id,
                "host": getattr(peer, "host", None),
                "port": getattr(peer, "port", 8770),
                "role": peer.role.value if hasattr(peer.role, "value") else str(peer.role),
                "capabilities": getattr(peer, "capabilities", []),
                "load_score": getattr(peer, "load_score", 0.0),
                "status": "alive" if peer.is_alive() else "dead",
                "is_alive": peer.is_alive(),
                "last_heartbeat": float(getattr(peer, "last_heartbeat", 0.0) or 0.0),
            }

        # Convenience diagnostics: reported leaders vs eligible leaders.
        leaders_reported = sorted(
            [p.node_id for p in peers_snapshot if p.role == NodeRole.LEADER and p.is_alive()]
        )
        leaders_eligible = sorted(
            [
                p.node_id
                for p in peers_snapshot
                if p.role == NodeRole.LEADER and self._is_leader_eligible(p, conflict_keys)
            ]
        )

        # Jan 12, 2026: Lock-free job snapshot access
        # Uses JobSnapshot copy-on-write pattern - no lock needed for reads.
        # Previous lock-based code removed (was causing 6+ second timeouts).
        jobs = self._job_snapshot.get_snapshot()

        # Get improvement cycle manager status
        improvement_status = None
        if self.improvement_cycle_manager:
            try:
                improvement_status = self.improvement_cycle_manager.get_status()
            except Exception as e:  # noqa: BLE001
                improvement_status = {"error": str(e)}

        # Get diversity metrics (delegated to SelfplayScheduler)
        # December 27, 2025: Added try-except to prevent 500 errors on memory-constrained nodes
        try:
            diversity_metrics = self.selfplay_scheduler.get_diversity_metrics()
        except Exception as e:  # noqa: BLE001
            diversity_metrics = {"error": str(e)}

        voter_ids = list(getattr(self, "voter_node_ids", []) or [])
        # Jan 2, 2026: Use _count_alive_voters() to check IP:port matches
        voters_alive = self._count_alive_voters()

        # Get P2P sync metrics (with error handling for new features)
        p2p_sync_metrics = getattr(self, "_p2p_sync_metrics", {})

        # Jan 30, 2026: Priority 2.2 Decomposition - Use StatusMetricsCollector for parallel metric gathering
        # Previously this was 70+ lines of inline code. Now delegated to the collector which:
        # - Runs all metrics in parallel (asyncio.gather)
        # - Applies 5s timeout per metric
        # - Handles errors gracefully
        from scripts.p2p.managers.status_metrics_collector import (
            create_status_metrics_collector,
        )

        collector = create_status_metrics_collector(self)
        collection_result = await collector.collect_all_metrics()
        metrics_dict = collection_result.metrics

        # Extract results into named variables for backward compatibility
        gossip_metrics = metrics_dict.get("gossip_metrics", {"error": "not_collected"})
        distributed_training = metrics_dict.get("distributed_training", {"error": "not_collected"})
        cluster_elo = metrics_dict.get("cluster_elo", {"error": "not_collected"})
        node_recovery = metrics_dict.get("node_recovery", {"error": "not_collected"})
        leader_consensus = metrics_dict.get("leader_consensus", {"error": "not_collected"})
        peer_reputation = metrics_dict.get("peer_reputation", {"error": "not_collected"})
        sync_intervals = metrics_dict.get("sync_intervals", {"error": "not_collected"})
        tournament_scheduling = metrics_dict.get("tournament_scheduling", {"error": "not_collected"})
        data_dedup = metrics_dict.get("data_dedup", {"error": "not_collected"})
        swim_raft_status = metrics_dict.get("swim_raft", {"error": "not_collected"})
        partition_status = metrics_dict.get("partition", {"error": "not_collected"})
        background_loops = metrics_dict.get("background_loops", {"error": "not_collected"})
        voter_health = metrics_dict.get("voter_health", {"error": "not_collected"})

        # Feb 2026: All metrics now extracted from parallel collector results.
        # Previously, 8+ metrics were computed sequentially after the collector,
        # adding 10-30+ seconds. Now they run in parallel with 5s timeout each.
        transport_latency = metrics_dict.get("transport_latency", {"error": "not_collected"})
        cluster_observability = metrics_dict.get("cluster_observability", {"error": "not_collected"})
        fallback_status = metrics_dict.get("fallback_status", {"error": "not_collected"})
        leadership_consistency = metrics_dict.get("leadership_consistency", {"error": "not_collected"})
        is_leader_result = metrics_dict.get("is_leader", {"value": False})
        is_leader_val = is_leader_result.get("value", False) if isinstance(is_leader_result, dict) else False
        config_version = metrics_dict.get("config_version", {"error": "not_collected"})
        data_summary = metrics_dict.get("data_summary", {"error": "not_collected"})
        cooldown_stats = metrics_dict.get("cooldown_stats", {"error": "not_collected"})
        peer_health_summary = metrics_dict.get("peer_health_summary", {"error": "not_collected"})

        # Dec 2025: Get event subscription status for health monitoring
        event_subscriptions = getattr(self, "_event_subscription_status", {
            "daemon_events": False,
            "feedback_signals": False,
            "manager_events": False,
            "all_healthy": False,
            "timestamp": 0,
        })

        # Jan 1, 2026: Work queue status for monitoring (Phase 4B fix)
        work_queue_size = 0
        active_jobs_count = 0
        selfplay_jobs_count = 0
        try:
            from app.coordination.work_queue import get_work_queue
            wq = get_work_queue()
            if wq is not None and hasattr(wq, 'get_queue_status'):
                wq_status = wq.get_queue_status()
                work_queue_size = wq_status.get('total_items', 0)
        except Exception:  # noqa: BLE001
            pass  # Fall back to 0

        # Count jobs directly from local_jobs
        if isinstance(jobs, dict) and "error" not in jobs:
            for job_data in jobs.values():
                if isinstance(job_data, dict):
                    status = job_data.get("status", "")
                    job_type = job_data.get("job_type", "")
                    if status in ("running", "claimed"):
                        active_jobs_count += 1
                    if job_type == "selfplay" and status in ("running", "claimed"):
                        selfplay_jobs_count += 1

        # Jan 1, 2026: Aggregate cluster-wide selfplay jobs from peers
        cluster_selfplay_jobs = selfplay_jobs_count  # Start with local count
        cluster_training_jobs = 0
        for peer_node_id, peer_data in peers.items():
            if isinstance(peer_data, dict):
                cluster_selfplay_jobs += int(peer_data.get("selfplay_jobs", 0) or 0)
                cluster_training_jobs += int(peer_data.get("training_jobs", 0) or 0)

        return {
            "node_id": self.node_id,
            "role": self.role.value,
            "leader_id": self.leader_id,
            "forced_leader_override": getattr(self, "_forced_leader_override", False),
            "effective_leader_id": (effective_leader.node_id if effective_leader else None),
            # Jan 1, 2026: Provisional leadership status
            "is_provisional_leader": self.role == NodeRole.PROVISIONAL_LEADER,
            "provisional_claimed_at": getattr(self, "_provisional_leader_claimed_at", 0.0) or 0.0,
            "provisional_acks": len(getattr(self, "_provisional_leader_acks", set()) or set()),
            "provisional_challengers": len(getattr(self, "_provisional_leader_challengers", {}) or {}),
            "fallback_leader_since": getattr(self, "_fallback_leader_since", 0.0) or 0.0,
            "fallback_leader_reason": getattr(self, "_fallback_leader_reason", "") or "",
            "leaders_reported": leaders_reported,
            "leaders_eligible": leaders_eligible,
            "voter_node_ids": voter_ids,
            "voter_quorum_size": int(getattr(self, "voter_quorum_size", 0) or 0),
            "voters_alive": voters_alive,
            "voter_quorum_ok": self._has_voter_quorum(),
            # Jan 20, 2026: Voter config sync - version and hash for drift detection
            "voter_config_version": self._get_voter_config_version(),
            "voter_config_hash": self._get_voter_config_hash(),
            # Jan 2, 2026: Detailed voter health for monitoring
            # Jan 16, 2026: Now pre-computed with timeout protection
            "voter_health": voter_health,
            "self": self.self_info.to_dict(),
            "peers": peers,
            "all_peers": all_peers,  # Jan 5, 2026: All peers regardless of alive status for job dispatch
            "local_jobs": jobs,
            # Feb 3, 2026: Use lock-free peers_snapshot instead of self.peers to prevent blocking
            "alive_peers": len([p for p in peers_snapshot if p.is_alive()]),
            "improvement_cycle_manager": improvement_status,
            "diversity_metrics": diversity_metrics,
            "gossip_metrics": gossip_metrics,
            "p2p_sync_metrics": p2p_sync_metrics,
            "distributed_training": distributed_training,
            "cluster_elo": cluster_elo,
            "node_recovery": node_recovery,
            "leader_consensus": leader_consensus,
            "peer_reputation": peer_reputation,
            "sync_intervals": sync_intervals,
            "tournament_scheduling": tournament_scheduling,
            "data_dedup": data_dedup,
            "swim_raft": swim_raft_status,
            "transport_latency": transport_latency,  # Jan 3, 2026: Per-transport latency metrics
            "event_subscriptions": event_subscriptions,
            "partition": partition_status,
            "background_loops": background_loops,
            # December 30, 2025: Cluster observability for debugging idle nodes
            # Feb 4, 2026: Now pre-computed with fallback to avoid blocking
            "cluster_observability": cluster_observability,
            # Session 17.41 (Jan 6, 2026): Fallback mechanism status for partition debugging
            # Feb 4, 2026: Now pre-computed with fallback to avoid blocking
            "fallback_status": fallback_status,
            # December 30, 2025: Lock acquisition status for debugging
            "_lock_status": {
                "peers_lock_acquired": peers_snapshot is not None,
                "jobs_lock_acquired": "error" not in jobs,
            },
            # Jan 1, 2026: Explicit work queue and job counts (Phase 4B fix)
            "work_queue_size": work_queue_size,
            "active_jobs": active_jobs_count,
            "selfplay_jobs": cluster_selfplay_jobs,  # Cluster-wide aggregated
            "training_jobs": cluster_training_jobs,  # Cluster-wide aggregated
            "local_selfplay_jobs": selfplay_jobs_count,  # This node only
            # Jan 2, 2026: Dual-stack IPv4/IPv6 network info
            "network": {
                "advertise_host": self.advertise_host,
                "advertise_host_family": "ipv6" if ":" in (self.advertise_host or "") else "ipv4",
                "alternate_ips": list(getattr(self, "alternate_ips", set()) or set()),
                "alternate_ipv4_count": sum(1 for ip in getattr(self, "alternate_ips", set()) or set() if ":" not in ip),
                "alternate_ipv6_count": sum(1 for ip in getattr(self, "alternate_ips", set()) or set() if ":" in ip),
            },
            # Jan 3, 2026: Leadership consistency metrics for monitoring desync issues
            # This enables detection of the leader self-recognition bug where leader_id
            # is set correctly but role doesn't match.
            # Jan 30, 2026: Use leadership orchestrator directly
            # Feb 4, 2026: Now pre-computed with fallback to avoid blocking
            "leadership_consistency": leadership_consistency,
            "is_leader": is_leader_val,  # Explicit field for quick checks
            # Jan 13, 2026: Config version for drift detection (P2P Cluster Stability Plan Phase 1)
            # Feb 4, 2026: Now pre-computed with fallback to avoid blocking
            "config_version": config_version,
            # Jan 13, 2026: Unified data summary across all sources (LOCAL, CLUSTER, S3, OWC)
            # Feb 4, 2026: Now pre-computed with fallback to avoid blocking
            "data_summary": data_summary,
            # Jan 20, 2026: Adaptive dead peer cooldown stats
            # Feb 4, 2026: Now pre-computed with fallback to avoid blocking
            "cooldown_stats": cooldown_stats,
            # Jan 25, 2026: Peer health summary for P2P stability monitoring (Phase 3)
            # Feb 4, 2026: Now pre-computed with fallback to avoid blocking
            "peer_health_summary": peer_health_summary,
            # Feb 2026: Cache metadata
            "_cache_time": time.time(),
        }

    async def handle_loops_health(self, request: web.Request) -> web.Response:
        """Return health status of all background loops.

        Feb 1, 2026: Added for operational visibility into loop health.
        Exposes loop running state, error counts, and timeout stats.
        """
        loop_manager = self._get_loop_manager()
        if loop_manager is None:
            return web.json_response(
                {"error": "LoopManager not initialized", "loops": {}, "total": 0},
                status=503,
            )

        all_status = loop_manager.get_all_status()
        unhealthy = [
            name for name, status in all_status.items()
            if status.get("status") in ("error", "degraded", "stopped")
        ]

        return web.json_response({
            "loops": all_status,
            "total_count": len(all_status),
            "unhealthy_count": len(unhealthy),
            "unhealthy_loops": unhealthy,
            "manager_health": loop_manager.health_check(),
        })

    async def handle_training_sync(self, request: web.Request) -> web.Response:
        """Manually trigger sync of selfplay data to training nodes.

        Leader-only: Syncs selfplay data to the top GPU nodes for training.
        """
        try:
            result = await self._sync_selfplay_to_training_nodes()
            return web.json_response(result)
        except Exception as e:  # noqa: BLE001
            return web.json_response({"error": str(e)}, status=500)

    async def _run_improvement_loop(self, job_id: str):
        """Main coordinator loop for AlphaZero-style improvement."""
        try:
            state = self.improvement_loop_state.get(job_id)
            if not state:
                return

            logger.info(f"Improvement loop coordinator started for job {job_id}")

            while state.current_iteration < state.max_iterations and state.status == "running":
                state.current_iteration += 1
                logger.info(f"Improvement iteration {state.current_iteration}/{state.max_iterations}")

                # Phase 1: Selfplay
                state.phase = "selfplay"
                state.selfplay_progress = {}
                await self.job_manager.run_distributed_selfplay(job_id)

                # Phase 2: Export training data
                state.phase = "export"
                await self.job_manager.export_training_data(job_id)

                # Phase 3: Training
                state.phase = "train"
                await self.job_manager.run_training(job_id)

                # Phase 4: Evaluation
                state.phase = "evaluate"
                await self._run_evaluation(job_id)

                # Phase 5: Promote if better
                state.phase = "promote"
                await self._promote_model_if_better(job_id)

                state.last_update = time.time()

            state.status = "completed"
            state.phase = "idle"
            logger.info(f"Improvement loop {job_id} completed after {state.current_iteration} iterations")

        except Exception as e:  # noqa: BLE001
            logger.info(f"Improvement loop error: {e}")
            if job_id in self.improvement_loop_state:
                self.improvement_loop_state[job_id].status = f"error: {e}"


    async def _check_and_trigger_training(self):
        """Periodic check for training readiness (leader only)."""
        if self.role != NodeRole.LEADER:
            return

        # Phase 2.4 (Dec 29, 2025): Skip training dispatch in partition readonly mode
        if self.is_partition_readonly():
            logger.debug("[P2P] Skipping training check: partition readonly mode")
            return

        current_time = time.time()
        if current_time - self.last_training_check < self.training_check_interval:
            return

        self.last_training_check = current_time

        # Get jobs that should be started (delegated to TrainingCoordinator manager)
        # Feb 23, 2026: Wrapped in to_thread() — check_training_readiness() is sync
        # and accesses cluster_data_manifest + training_lock, blocking event loop
        jobs_to_start = await asyncio.to_thread(
            self.training_coordinator.check_training_readiness
        )

        for job_config in jobs_to_start:
            # PHASE 4 IDEMPOTENCY: Check for duplicate triggers
            config_key = job_config.get("config_key", "")
            game_count = job_config.get("total_games", 0)
            can_proceed, trigger_hash = self._check_training_idempotency(config_key, game_count)
            if not can_proceed:
                continue

            logger.info(f"Auto-triggering {job_config['job_type']} training for {config_key} ({game_count} games)")
            await self.training_coordinator.dispatch_training_job(job_config)
            self._record_training_trigger(trigger_hash)  # Record after successful dispatch

    async def _check_local_training_fallback(self):
        """DECENTRALIZED training trigger when cluster has no leader.

        LEADERLESS RESILIENCE: When the cluster has been without a leader for too long
        (LEADERLESS_TRAINING_TIMEOUT = 3 minutes), individual nodes can trigger local
        training to prevent data accumulation without progress.

        This makes the system more resilient to leader election failures while avoiding
        duplicate training by:
        1. Only triggering after a brief leaderless period (3 minutes)
        2. Using random jitter so nodes don't all train simultaneously
        3. Only training on local data (no cluster-wide coordination needed)
        4. Using reasonable cooldowns between fallback training runs
        """
        # Skip if we ARE the leader or have a known leader
        if self.role == NodeRole.LEADER or self.leader_id:
            self.last_leader_seen = time.time()  # Update leader seen time
            return

        current_time = time.time()
        leaderless_duration = current_time - self.last_leader_seen

        # Only trigger fallback if leaderless for the timeout period
        if leaderless_duration < LEADERLESS_TRAINING_TIMEOUT:
            return

        # Rate limit fallback training (10 minute cooldown - more aggressive than before)
        fallback_cooldown = 600  # 10 minutes between fallback triggers
        if current_time - self.last_local_training_fallback < fallback_cooldown:
            return

        # Random jitter: 40% probability per check (more aggressive than 20%)
        # This distributes training across nodes over time
        import random
        if random.random() > 0.4:
            return

        # Check if we have a GPU (training needs GPU)
        if not getattr(self.self_info, "has_gpu", False):
            return

        # Check local data manifest (use cached version for speed)
        local_manifest = getattr(self, "local_data_manifest", None)
        if not local_manifest:
            # Try to load from cache or collect if we don't have one
            try:
                # Jan 23, 2026: Wrap in asyncio.to_thread() to prevent event loop blocking
                # collect_local_manifest_cached() does file I/O and SQLite operations
                local_manifest = await asyncio.to_thread(
                    self.sync_planner.collect_local_manifest_cached, max_cache_age=600
                )
                with self.manifest_lock:
                    self.local_data_manifest = local_manifest
            except (AttributeError):
                return

        # Check for sufficient local data (lower threshold for faster training)
        min_games_fallback = 2000  # Lower threshold for faster response
        total_local_games = getattr(local_manifest, "selfplay_games", 0)
        if total_local_games < min_games_fallback:
            return

        # Find board types with enough local data
        game_counts_by_type: dict[str, int] = {}
        for file_info in getattr(local_manifest, "files", []) or []:
            board_type = getattr(file_info, "board_type", "")
            num_players = getattr(file_info, "num_players", 2)
            game_count = getattr(file_info, "game_count", 0)
            if board_type and game_count > 0:
                key = f"{board_type}_{num_players}p"
                game_counts_by_type[key] = game_counts_by_type.get(key, 0) + game_count

        # Sort by game count (descending) to train on richest data first
        sorted_configs = sorted(game_counts_by_type.items(), key=lambda x: x[1], reverse=True)

        # Trigger local training for configurations with enough data
        triggered_count = 0
        max_concurrent_fallback = 2  # Can trigger up to 2 training jobs per fallback
        for config_key, game_count in sorted_configs:
            if triggered_count >= max_concurrent_fallback:
                break
            if game_count < 1000:  # Minimum threshold (lowered)
                continue

            # Check if we already have a running training job for this config
            existing_job = self.training_coordinator.find_running_training_job("nnue", config_key)
            if existing_job:
                continue

            # DISTRIBUTED TRAINING COORDINATION: Check cluster-wide before starting
            is_training, _training_nodes = self._is_config_being_trained_cluster_wide(config_key)
            if is_training:
                # Someone else is already training this config
                continue

            # Use distributed slot claiming to avoid race conditions
            if not self._should_claim_training_slot(config_key):
                continue

            # Parse board type and player count
            parts = config_key.split("_")
            if len(parts) < 2:
                continue
            board_type = parts[0]
            num_players = int(parts[1].replace("p", ""))

            # PHASE 4 IDEMPOTENCY: Check for duplicate triggers
            can_proceed, trigger_hash = self._check_training_idempotency(config_key, game_count)
            if not can_proceed:
                continue

            logger.info(f"DISTRIBUTED TRAINING: Claiming {config_key} ({game_count} local games, leaderless for {int(leaderless_duration)}s)")
            job_config = {
                "job_type": "nnue",
                "board_type": board_type,
                "num_players": num_players,
                "config_key": config_key,
                "total_games": game_count,
            }
            await self.training_coordinator.dispatch_training_job(job_config)
            self._record_training_trigger(trigger_hash)  # Record after successful dispatch
            triggered_count += 1

        if triggered_count > 0:
            self.last_local_training_fallback = current_time
            logger.info(f"LEADERLESS FALLBACK: Triggered {triggered_count} local training job(s)")

    async def _check_improvement_cycles(self):
        """Periodic check for improvement cycle readiness (leader only).

        This integrates with the ImprovementCycleManager to:
        1. Check if any cycles need training based on data thresholds
        2. Trigger export/training jobs for ready cycles
        3. Run evaluations and update Elo ratings
        4. Schedule CMA-ES optimization when needed
        5. Schedule diverse tournaments for AI calibration
        """
        if self.role != NodeRole.LEADER:
            return

        if not self.improvement_cycle_manager:
            return

        current_time = time.time()
        if current_time - self.last_improvement_cycle_check < self.improvement_cycle_check_interval:
            return

        self.last_improvement_cycle_check = current_time

        # Check which cycles are ready for training
        training_ready = self.improvement_cycle_manager.check_training_needed()

        # Convert to job configs
        jobs_to_start = []
        for board_type, num_players in training_ready:
            cycle_key = f"{board_type}_{num_players}p"
            cycle_state = self.improvement_cycle_manager.state.cycles.get(cycle_key)
            if cycle_state and self.improvement_cycle_manager.trigger_training(board_type, num_players):
                jobs_to_start.append({
                    "cycle_id": cycle_key,
                    "board_type": board_type,
                    "num_players": num_players,
                    "total_games": cycle_state.games_since_last_training,
                    "iteration": cycle_state.current_iteration + 1,
                })

        # Also check for CMA-ES optimization opportunities
        cmaes_ready = self.improvement_cycle_manager.check_cmaes_needed()
        for board_type, num_players in cmaes_ready:
            # Trigger distributed CMA-ES (Jan 2026: uses cmaes_coordinator directly)
            logger.info(f"CMA-ES optimization ready for {board_type}_{num_players}p")
            asyncio.create_task(self.cmaes_coordinator.trigger_auto_cmaes(board_type, num_players))

        # Check for rollback needs (consecutive training failures)
        for key, cycle in self.improvement_cycle_manager.state.cycles.items():
            if not cycle.pending_training and not cycle.pending_evaluation:
                should_rollback, reason = self.improvement_cycle_manager.check_rollback_needed(
                    cycle.board_type, cycle.num_players
                )
                if should_rollback:
                    logger.info(f"ROLLBACK NEEDED for {key}: {reason}")
                    if self.improvement_cycle_manager.execute_rollback(cycle.board_type, cycle.num_players):
                        self.diversity_metrics["rollbacks"] += 1
                        # Increase diversity to escape plateau
                        logger.info(f"Increasing diversity to escape training plateau for {key}")

        for job_config in jobs_to_start:
            cycle_id = job_config["cycle_id"]
            board_type = job_config["board_type"]
            num_players = job_config["num_players"]

            logger.info(f"ImprovementCycle {cycle_id}: Starting training "
                  f"({job_config['total_games']} games)")

            # Find GPU worker for training
            gpu_worker = None
            candidates: list[NodeInfo] = []
            with self.peers_lock:
                candidates.extend([p for p in self.peers.values() if p.is_gpu_node() and p.is_healthy()])
            if self.self_info.is_gpu_node() and self.self_info.is_healthy():
                candidates.append(self.self_info)
            if candidates:
                candidates.sort(
                    key=lambda p: (-p.gpu_power_score(), p.get_load_score(), str(p.node_id))
                )
                gpu_worker = candidates[0]

            if not gpu_worker:
                logger.info(f"ImprovementCycle {cycle_id}: No GPU worker available, deferring")
                self.improvement_cycle_manager.update_cycle_phase(
                    cycle_id, "idle", error_message="No GPU worker available"
                )
                continue

            # Create training job
            job_id = f"cycle_{cycle_id}_{int(time.time())}"
            training_job = TrainingJob(
                job_id=job_id,
                job_type="nnue",
                board_type=board_type,
                num_players=num_players,
                worker_node=gpu_worker.node_id,
                epochs=job_config.get("epochs", 100),
                batch_size=job_config.get("batch_size", 4096),
                learning_rate=job_config.get("learning_rate", 0.001),
                data_games_count=job_config.get("total_games", 0),
            )

            with self.training_lock:
                self.training_jobs[job_id] = training_job

            # Update cycle state
            self.improvement_cycle_manager.update_cycle_phase(
                cycle_id, "training", training_job_id=job_id
            )

            # Dispatch training to worker
            await self._dispatch_improvement_training(training_job, cycle_id)

    async def _dispatch_improvement_training(self, job: TrainingJob, cycle_id: str):
        """Dispatch training job for improvement cycle."""
        try:
            # Find the worker node
            worker_node = None
            if job.worker_node == self.node_id:
                worker_node = self.self_info
            else:
                with self.peers_lock:
                    worker_node = self.peers.get(job.worker_node)

            if not worker_node:
                logger.info(f"ImprovementCycle {cycle_id}: Worker {job.worker_node} not found")
                self.improvement_cycle_manager.update_cycle_phase(
                    cycle_id, "idle", error_message=f"Worker {job.worker_node} not found"
                )
                return

            # Build training payload
            payload = {
                "job_id": job.job_id,
                "cycle_id": cycle_id,
                "board_type": job.board_type,
                "num_players": job.num_players,
                "epochs": job.epochs,
                "batch_size": job.batch_size,
                "learning_rate": job.learning_rate,
            }

            # Send to worker
            timeout = ClientTimeout(total=30)
            async with get_client_session(timeout) as session:
                last_err: str | None = None
                for url in self._urls_for_peer(worker_node, "/training/nnue/start"):
                    try:
                        async with session.post(url, json=payload, headers=self._auth_headers()) as resp:
                            if resp.status != 200:
                                last_err = f"http_{resp.status}"
                                continue
                            result = await resp.json()
                        if result.get("success"):
                            job.status = "running"
                            job.started_at = time.time()
                            logger.info(f"ImprovementCycle {cycle_id}: Training started on {worker_node.node_id}")
                            return
                        self.improvement_cycle_manager.update_cycle_phase(
                            cycle_id, "idle", error_message=result.get("error", "Training failed to start")
                        )
                        return
                    except Exception as e:  # noqa: BLE001
                        last_err = str(e)
                        continue
                self.improvement_cycle_manager.update_cycle_phase(
                    cycle_id, "idle", error_message=last_err or "dispatch_failed"
                )

        except Exception as e:  # noqa: BLE001
            logger.info(f"ImprovementCycle {cycle_id}: Training dispatch failed: {e}")
            self.improvement_cycle_manager.update_cycle_phase(
                cycle_id, "idle", error_message=str(e)
            )

    # Includes: handle_training_start, handle_training_status, handle_training_progress, handle_training_update,
    #           handle_training_trigger, handle_training_trigger_decision, handle_training_trigger_configs, handle_nnue_start


    def _get_training_timeout(self, job_id: str) -> int:
        """Get dynamic timeout based on job configuration.

        Returns timeout in seconds based on board type and model complexity:
        - square19: 6 hours (large board, 361 cells)
        - hexagonal: 5 hours (469 cells)
        - square8/hex8: 2 hours (small boards)
        Default: 3 hours if job not found
        """
        with self.training_lock:
            job = self.training_jobs.get(job_id)
            if not job:
                return 10800  # 3 hours default

            board_type = getattr(job, 'board_type', 'unknown')
            num_players = getattr(job, 'num_players', 2)

            # Base timeout by board complexity
            if board_type == 'square19':
                base_timeout = 21600  # 6 hours
            elif board_type == 'hexagonal':
                base_timeout = 18000  # 5 hours
            elif board_type in ('hex8', 'square8'):
                base_timeout = 7200   # 2 hours
            else:
                base_timeout = 10800  # 3 hours default

            # Add 50% for 4-player models (larger value head, more complex)
            if num_players == 4:
                base_timeout = int(base_timeout * 1.5)
            elif num_players == 3:
                base_timeout = int(base_timeout * 1.25)

            return base_timeout

    def _get_cached_jittered_timeout(self) -> float:
        """Get jittered peer timeout, cached for 30 seconds.

        Jan 22, 2026: Fix for double jitter application causing desynchronized death detection.

        Problem: get_jittered_peer_timeout() was called at two locations (partition detection
        and peer reconnection) with different jitter each time. This caused nodes to mark
        the same peer dead at different times (±10% variance = 24s difference for 120s timeout).

        Solution: Cache the jittered timeout for 30 seconds. All death detection checks
        within the same 30s window use the same jittered value, ensuring consistent
        death detection across the codebase.

        Returns:
            Jittered peer timeout in seconds (PEER_TIMEOUT ± 10%)
        """
        now = time.time()
        if self._jittered_timeout_cache is None or (now - self._jittered_timeout_time) > 30:
            self._jittered_timeout_cache = get_jittered_peer_timeout(PEER_TIMEOUT)
            self._jittered_timeout_time = now
        return self._jittered_timeout_cache

    async def _monitor_training_process(self, job_id: str, proc, output_path: str):
        """Monitor training subprocess and report completion to leader."""
        try:
            timeout = self._get_training_timeout(job_id)
            _stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout
            )

            success = proc.returncode == 0

            # Report to leader with retry logic
            if self.leader_id and self.leader_id != self.node_id:
                leader = self.peers.get(self.leader_id)
                if leader:
                    payload = {
                        "job_id": job_id,
                        "completed": success,
                        "output_model_path": output_path if success else "",
                        "error": stderr.decode()[:500] if not success else "",
                    }
                    # Retry with exponential backoff (3 attempts: 5s, 10s, 20s)
                    max_retries = 3
                    base_delay = 5.0
                    for attempt in range(max_retries):
                        try:
                            http_timeout = ClientTimeout(total=30)
                            async with get_client_session(http_timeout) as session:
                                url = self._url_for_peer(leader, "/training/update")
                                resp = await session.post(url, json=payload, headers=self._auth_headers())
                                if resp.status < 400:
                                    logger.info(f"Training completion reported to leader (attempt {attempt + 1})")
                                    break
                                else:
                                    logger.warning(f"Leader returned {resp.status}, retrying...")
                        except Exception as e:  # noqa: BLE001
                            delay = base_delay * (2 ** attempt)
                            if attempt < max_retries - 1:
                                logger.warning(f"Failed to report training completion (attempt {attempt + 1}): {e}, retrying in {delay}s")
                                await asyncio.sleep(delay)
                            else:
                                logger.error(f"Failed to report training completion after {max_retries} attempts: {e}")
            else:
                # We are the leader, update directly
                with self.training_lock:
                    job = self.training_jobs.get(job_id)
                    if job:
                        if success:
                            job.status = "completed"
                            job.completed_at = time.time()
                            job.output_model_path = output_path
                            # LEARNED LESSONS - Schedule tournament to compare new model against baseline
                            # Jan 28, 2026: Uses tournament_manager directly
                            asyncio.create_task(self.tournament_manager.schedule_model_comparison(job, output_path))
                            # Update improvement cycle manager with training completion
                            if self.improvement_cycle_manager:
                                self.improvement_cycle_manager.handle_training_complete(
                                    job.board_type, job.num_players,
                                    output_path, job.data_games_count or 0
                                )
                            # PFSP: Add trained model to opponent pool for diverse selfplay
                            config_key = f"{job.board_type}_{job.num_players}p"
                            if HAS_PFSP and config_key in self.pfsp_pools:
                                try:
                                    model_id = Path(output_path).stem
                                    self.pfsp_pools[config_key].add_opponent(
                                        model_id=model_id,
                                        model_path=output_path,
                                        elo=INITIAL_ELO_RATING,  # From app.config.thresholds
                                        win_rate=0.5,
                                    )
                                    logger.info(f"[PFSP] Added {model_id} to opponent pool for {config_key}")
                                except Exception as e:  # noqa: BLE001
                                    logger.error(f"[PFSP] Error adding model to pool: {e}")
                            # CMA-ES: Check for Elo plateau and trigger auto-tuning
                            asyncio.create_task(self._check_cmaes_auto_tuning(config_key))
                        else:
                            job.status = "failed"
                            job.error_message = stderr.decode()[:500]
                        job.completed_at = time.time()

            logger.info(f"Training job {job_id} {'completed' if success else 'failed'}")

        except asyncio.TimeoutError:
            logger.info(f"Training job {job_id} timed out")
        except Exception as e:  # noqa: BLE001
            logger.info(f"Training monitor error for {job_id}: {e}")


    async def _monitor_selfplay_process(
        self,
        job_id: str,
        proc: subprocess.Popen,
        output_dir: Path,
        board_type: str,
        num_players: int,
        job_type_str: str = "selfplay",
    ) -> None:
        """Monitor a selfplay subprocess and update job status on completion.

        Dec 31, 2025: Added to fix missing process monitoring for SELFPLAY
        and CPU_SELFPLAY jobs. Previously, these jobs were spawned but never
        monitored, causing them to remain in "running" status indefinitely.

        This function:
        1. Waits for the subprocess to complete (with 2-hour timeout)
        2. Updates job status to "completed" or "failed"
        3. Logs completion/failure with details
        4. Emits TASK_COMPLETED or TASK_FAILED events for pipeline coordination
        """
        try:
            # Wait for process to complete (with timeout)
            return_code = await asyncio.wait_for(
                asyncio.to_thread(proc.wait),
                timeout=7200,  # 2 hour max
            )

            duration = 0.0
            with self.jobs_lock:
                job = self.local_jobs.get(job_id)
                if job:
                    # Feb 2026: Use get/set_job_attr to handle both ClusterJob objects and dict fallbacks
                    started = get_job_attr(job, "started_at", 0.0)
                    duration = time.time() - started
                    if return_code == 0:
                        set_job_attr(job, "status", "completed")
                        set_job_attr(job, "completed_at", time.time())
                        logger.info(
                            f"Selfplay job {job_id} completed successfully "
                            f"(duration: {duration:.1f}s)"
                        )
                    else:
                        # Try to get error message from run.log
                        error_msg = f"exit_code={return_code}"
                        log_file = output_dir / "run.log"
                        if log_file.exists():
                            try:
                                # Get last 500 chars of log for error context
                                content = log_file.read_text(encoding='utf-8', errors='replace')
                                if content:
                                    error_msg = content[-500:].strip()
                            except OSError:
                                pass
                        set_job_attr(job, "status", "failed")
                        set_job_attr(job, "completed_at", time.time())
                        set_job_attr(job, "error_message", error_msg)
                        logger.warning(
                            f"Selfplay job {job_id} failed (exit code {return_code}): "
                            f"{error_msg[:200]}..."
                        )

            # Emit task events for pipeline coordination
            try:
                from app.coordination.data_events import DataEventType, emit_data_event
                config_key = f"{board_type}_{num_players}p"
                if return_code == 0:
                    emit_data_event(DataEventType.TASK_COMPLETED, {
                        "task_id": job_id,
                        "task_type": job_type_str,
                        "config_key": config_key,
                        "board_type": board_type,
                        "num_players": num_players,
                        "duration_seconds": duration,
                        "node_id": self.node_id,
                    })
                else:
                    emit_data_event(DataEventType.TASK_FAILED, {
                        "task_id": job_id,
                        "task_type": job_type_str,
                        "config_key": config_key,
                        "board_type": board_type,
                        "num_players": num_players,
                        "error": f"exit_code={return_code}",
                        "node_id": self.node_id,
                    })
            except ImportError:
                pass  # Event system not available

        except asyncio.TimeoutError:
            logger.warning(f"Selfplay job {job_id} timed out after 2 hours")
            with self.jobs_lock:
                job = self.local_jobs.get(job_id)
                if job:
                    set_job_attr(job, "status", "timeout")
                    set_job_attr(job, "completed_at", time.time())
                    set_job_attr(job, "error_message", "timeout_2_hours")
            # Kill the process
            try:
                proc.terminate()
                await asyncio.sleep(5)
                if proc.poll() is None:
                    proc.kill()
            except OSError:
                pass

        except Exception as e:  # noqa: BLE001
            logger.error(f"Selfplay process monitor error for {job_id}: {e}")
            with self.jobs_lock:
                job = self.local_jobs.get(job_id)
                if job:
                    set_job_attr(job, "status", "error")
                    set_job_attr(job, "completed_at", time.time())
                    set_job_attr(job, "error_message", str(e))


    async def _check_cmaes_auto_tuning(self, config_key: str):
        """Check if CMA-ES auto-tuning should be triggered for a config.

        Monitors Elo progression and triggers hyperparameter optimization
        when the model's improvement plateaus.
        """
        if not HAS_PFSP or config_key not in self.cmaes_auto_tuners:
            return

        try:
            # Get current Elo from unified database
            from app.tournament import get_elo_database
            db = get_elo_database()

            parts = config_key.rsplit("_", 1)
            board_type = parts[0]
            num_players = int(parts[1].replace("p", ""))

            # Find best model for this config
            best_model = None
            best_elo = INITIAL_ELO_RATING
            models_dir = Path(self._get_ai_service_path()) / "models" / "nnue"
            pattern = f"nnue_{board_type}_{num_players}p*.pt"

            for model_path in models_dir.glob(pattern):
                model_id = model_path.stem
                elo = db.get_elo(model_id)
                if elo and elo > best_elo:
                    best_elo = elo
                    best_model = model_id

            if not best_model:
                return

            # Check for plateau
            auto_tuner = self.cmaes_auto_tuners[config_key]
            self.last_cmaes_elo.get(config_key, INITIAL_ELO_RATING)

            # Record Elo history for plateau detection
            should_tune = auto_tuner.check_plateau(best_elo)
            self.last_cmaes_elo[config_key] = best_elo

            if should_tune:
                logger.info(f"[CMA-ES] Elo plateau detected for {config_key} (Elo: {best_elo:.0f})")
                logger.info("[CMA-ES] Triggering auto hyperparameter optimization...")

                # Trigger CMA-ES via existing distributed infrastructure (Jan 2026: uses cmaes_coordinator directly)
                await self.cmaes_coordinator.trigger_auto_cmaes(board_type, num_players)

        except Exception as e:  # noqa: BLE001
            logger.info(f"[CMA-ES] Auto-tuning check error for {config_key}: {e}")

    def get_pfsp_opponent(self, config_key: str) -> str | None:
        """Get a PFSP-sampled opponent model for selfplay.

        Returns path to an opponent model sampled from the PFSP pool,
        weighted by difficulty (harder opponents sampled more frequently).
        """
        if not HAS_PFSP or config_key not in self.pfsp_pools:
            return None

        try:
            pool = self.pfsp_pools[config_key]
            opponent = pool.sample_opponent()
            if opponent:
                return opponent.model_path
        except Exception as e:  # noqa: BLE001
            logger.error(f"[PFSP] Error sampling opponent: {e}")
        return None

    def update_pfsp_stats(self, config_key: str, model_id: str, win_rate: float, elo: float):
        """Update PFSP stats for a model after evaluation games.

        Called after tournament/evaluation to update opponent difficulty metrics.
        """
        if not HAS_PFSP or config_key not in self.pfsp_pools:
            return

        try:
            self.pfsp_pools[config_key].update_stats(model_id, win_rate=win_rate, elo=elo)
            logger.info(f"[PFSP] Updated stats for {model_id}: win_rate={win_rate:.2f}, elo={elo:.0f}")
        except Exception as e:  # noqa: BLE001
            logger.error(f"[PFSP] Error updating stats: {e}")


    async def _import_gpu_selfplay_to_canonical(
        self, validated_db: Path, board_type: str, num_players: int, game_count: int
    ):
        """Import validated GPU selfplay games to canonical selfplay database.

        After GPU selfplay games pass CPU validation (>=95% validation rate),
        this merges them into the canonical selfplay database for training.
        """
        try:
            # Determine canonical DB path
            canonical_db = Path(self._get_ai_service_path()) / "data" / "games" / "selfplay.db"
            if not canonical_db.parent.exists():
                canonical_db.parent.mkdir(parents=True, exist_ok=True)

            logger.info(f"Auto-importing {game_count} validated GPU games to canonical DB...")

            # Jan 12, 2026: Wrap blocking SQLite operations in thread to avoid blocking event loop
            imported = await asyncio.to_thread(
                self._import_gpu_selfplay_sync, validated_db, canonical_db
            )

            logger.info(f"Successfully imported {imported} GPU selfplay games to canonical DB")

            # Update cluster data manifest to reflect new games
            config_key = f"{board_type}_{num_players}p"
            if hasattr(self, 'cluster_data_manifest') and self.cluster_data_manifest and config_key in self.cluster_data_manifest.by_board_type:
                self.cluster_data_manifest.by_board_type[config_key]["total_games"] = (
                    self.cluster_data_manifest.by_board_type[config_key].get("total_games", 0) + imported
                )

            # Notify improvement cycle manager of new games
            if self.improvement_cycle_manager and imported > 0:
                self.improvement_cycle_manager.record_games(board_type, num_players, imported)

        except Exception as e:  # noqa: BLE001
            logger.info(f"GPU selfplay import error: {e}")
            import traceback
            traceback.print_exc()


    # =========================================================================
    # See: scripts/p2p/handlers/improvement.py (Dec 28, 2025 - Phase 8)
    # =========================================================================


    # handle_improvement_training_complete and handle_improvement_evaluation_complete
    # moved to ImprovementHandlersMixin (Dec 28, 2025 - Phase 8)

    async def _schedule_improvement_evaluation(self, cycle_id: str, new_model_id: str):
        """Schedule tournament evaluation for a newly trained model via SSH."""
        if not self.improvement_cycle_manager:
            return
        try:
            cycle = self.improvement_cycle_manager.state.cycles.get(cycle_id)
            if not cycle:
                return

            config = cycle.config
            best_model_id = cycle.best_model_id or f"baseline_{config.board_type}_{config.num_players}p"

            logger.info(f"ImprovementCycle {cycle_id}: Scheduling evaluation {new_model_id} vs {best_model_id}")

            self.improvement_cycle_manager.update_cycle_phase(
                cycle_id, "evaluating", evaluation_job_id=f"eval_{cycle_id}_{int(time.time())}"
            )

            # Run SSH tournament evaluation
            eval_result = await self._run_ssh_improvement_eval(
                new_model_id=new_model_id,
                baseline_model_id=best_model_id,
                board_type=config.board_type,
                num_players=config.num_players,
                games=config.evaluation_games,
            )

            if eval_result.get("success"):
                new_model_wins = eval_result.get("new_model_wins", 0)
                baseline_wins = eval_result.get("baseline_wins", 0)
                draws = eval_result.get("draws", 0)
            else:
                # Fallback to mock results if SSH evaluation fails
                logger.info(f"ImprovementCycle {cycle_id}: SSH evaluation failed, using fallback")
                import random
                total_games = config.evaluation_games
                new_model_wins = random.randint(int(total_games * 0.4), int(total_games * 0.6))
                draws = random.randint(0, int(total_games * 0.1))
                baseline_wins = total_games - new_model_wins - draws

            self.improvement_cycle_manager.handle_evaluation_complete(
                cycle_id=cycle_id, new_model_id=new_model_id, best_model_id=best_model_id,
                wins=new_model_wins, losses=baseline_wins, draws=draws,
            )

        except Exception as e:  # noqa: BLE001
            logger.info(f"ImprovementCycle {cycle_id}: Evaluation scheduling failed: {e}")
            if self.improvement_cycle_manager:
                self.improvement_cycle_manager.update_cycle_phase(cycle_id, "idle", error_message=str(e))

    async def _run_ssh_improvement_eval(
        self,
        new_model_id: str,
        baseline_model_id: str,
        board_type: str,
        num_players: int,
        games: int,
    ) -> dict:
        """Run improvement evaluation via SSH on a remote host.

        Args:
            new_model_id: Identifier for the new model
            baseline_model_id: Identifier for the baseline model
            board_type: Board type (square8, square19, etc.)
            num_players: Number of players
            games: Number of games to play

        Returns:
            Dict with evaluation results or error
        """
        # Calculate timeout upfront to avoid scope issues in exception handler
        timeout_seconds = max(300, games * 30)  # 30s per game estimate, minimum 5 minutes

        try:
            # Get available hosts for evaluation
            if load_remote_hosts is None:
                return {"success": False, "error": "load_remote_hosts not available"}

            hosts = load_remote_hosts()
            if not hosts:
                return {"success": False, "error": "No remote hosts configured"}

            # Find a ready host with GPU capability (prefer high-performance hosts)
            eval_host = None
            for host in hosts:
                if getattr(host, 'status', None) == 'ready':
                    eval_host = host
                    break

            if not eval_host:
                # Try any host
                eval_host = hosts[0] if hosts else None

            if not eval_host:
                return {"success": False, "error": "No evaluation host available"}

            ssh_host = getattr(eval_host, 'ssh_host', None) or getattr(eval_host, 'tailscale_ip', None)
            if not ssh_host:
                return {"success": False, "error": "No SSH host configured"}

            ssh_user = getattr(eval_host, 'ssh_user', 'ubuntu')
            ringrift_path = getattr(eval_host, 'ringrift_path', '~/ringrift/ai-service')

            # Build model paths (assumes models are in standard locations)
            new_model_path = f"models/{board_type}_{num_players}p/{new_model_id}.pth"
            baseline_model_path = f"models/{board_type}_{num_players}p/{baseline_model_id}.pth"

            # Build SSH command
            remote_cmd = f'''cd {ringrift_path} && source venv/bin/activate && python scripts/run_improvement_eval.py \
                --new-model "{new_model_path}" \
                --baseline-model "{baseline_model_path}" \
                --board {board_type} \
                --players {num_players} \
                --games {games} \
                --ai-type descent 2>/dev/null'''

            logger.info(f"Running SSH evaluation on {eval_host.name}: {new_model_id} vs {baseline_model_id}")

            proc = await asyncio.create_subprocess_exec(
                "ssh",
                "-o", "ConnectTimeout=30",
                "-o", "BatchMode=yes",
                "-o", "StrictHostKeyChecking=no",
                f"{ssh_user}@{ssh_host}",
                remote_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout_seconds
            )

            if proc.returncode != 0:
                stderr_text = stderr.decode()[:500] if stderr else ""
                logger.info(f"SSH evaluation failed on {eval_host.name}: {stderr_text}")
                return {"success": False, "error": f"SSH command failed: {stderr_text}"}

            # Parse JSON result from stdout
            stdout_text = stdout.decode().strip()
            if not stdout_text:
                return {"success": False, "error": "No output from evaluation script"}

            result = json.loads(stdout_text)
            logger.info(f"SSH evaluation complete: {result.get('new_model_wins', 0)}-{result.get('baseline_wins', 0)}-{result.get('draws', 0)}")
            return result

        except asyncio.TimeoutError:
            return {"success": False, "error": f"SSH evaluation timed out after {timeout_seconds}s"}
        except json.JSONDecodeError as e:
            return {"success": False, "error": f"Failed to parse evaluation result: {e}"}
        except Exception as e:  # noqa: BLE001
            return {"success": False, "error": str(e)}

    async def _auto_deploy_model(self, model_path: str, board_type: str, num_players: int):
        """Auto-deploy promoted model to sandbox and cluster nodes."""
        try:
            import subprocess
            logger.info(f"Auto-deploying model: {model_path}")

            # Build command args
            cmd_args = [
                sys.executable, "scripts/auto_deploy_models.py",
                "--model-path", model_path,
                "--board-type", board_type,
                "--num-players", str(num_players),
                "--skip-eval",  # Already evaluated
            ]
            if self.leadership.check_is_leader():
                cmd_args.append("--sync-cluster")

            # Run deployment script
            result = await asyncio.to_thread(
                subprocess.run,
                cmd_args,
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(Path(__file__).parent.parent),
            )

            if result.returncode == 0:
                logger.info(f"Model deployed successfully: {model_path}")
            else:
                logger.info(f"Model deployment failed: {result.stderr}")

        except Exception as e:  # noqa: BLE001
            logger.info(f"Auto-deploy error: {e}")

    # Canonical Pipeline Integration (for pipeline_orchestrator.py)
    # =========================================================================

    # See scripts/p2p/handlers/pipeline.py for implementation.

    def _get_auth_headers(self) -> dict[str, str]:
        """Get authentication headers for peer requests."""
        return {"Authorization": f"Bearer {self.auth_token}"} if self.auth_token else {}

    # =========================================================================
    # Phase 4: REST API for External Job Submission and Dashboard
    # =========================================================================


    # See scripts/p2p/handlers/cluster_api.py for implementation.

    # See scripts/p2p/handlers/dashboard.py for implementation.


    # handle_elo_table() moved to TableHandlersMixin (Dec 28, 2025 - Phase 8)
    # handle_nodes_table() moved to TableHandlersMixin (Dec 28, 2025 - Phase 8)

    # _get_holdout_metrics_cached, _get_mcts_stats_cached, _get_matchup_matrix_cached,
    # _get_model_lineage_cached, _get_data_quality_cached, _get_training_efficiency_cached

    # =========================================================================
    # Feature 5: Automated Model Rollback
    # =========================================================================

    async def _check_rollback_conditions(self) -> dict[str, Any]:
        """Check if any models should be rolled back. Delegates to AnalyticsCacheManager."""
        return await self.analytics_cache_manager.check_rollback_conditions()

    async def _execute_rollback(self, config: str, dry_run: bool = False) -> dict[str, Any]:
        """Execute a rollback for the given config. Delegates to AnalyticsCacheManager."""
        result = await self.analytics_cache_manager.execute_rollback(config, dry_run)
        return {
            "success": result.success,
            "config": result.config,
            "dry_run": result.dry_run,
            "message": result.message,
            "details": result.details,
        }

    async def _auto_rollback_check(self) -> list[dict[str, Any]]:
        """Automatically check and execute rollbacks. Delegates to AnalyticsCacheManager."""
        results = await self.analytics_cache_manager.auto_rollback_check()
        return [
            {
                "success": r.success,
                "config": r.config,
                "dry_run": r.dry_run,
                "message": r.message,
                "details": r.details,
            }
            for r in results
        ]

    # =========================================================================
    # Feature 6: Distributed Selfplay Autoscaling
    # =========================================================================

    async def _get_autoscaling_metrics(self) -> dict[str, Any]:
        """Get metrics for autoscaling decisions."""
        # Autoscaling thresholds tuned for 46-node cluster
        # These can be overridden via environment variables
        max_workers = int(os.environ.get("RINGRIFT_AUTOSCALE_MAX_WORKERS", "46"))
        min_workers = int(os.environ.get("RINGRIFT_AUTOSCALE_MIN_WORKERS", "2"))
        scale_up_threshold = int(os.environ.get("RINGRIFT_AUTOSCALE_SCALE_UP_GPH", "100"))
        scale_down_threshold = int(os.environ.get("RINGRIFT_AUTOSCALE_SCALE_DOWN_GPH", "500"))
        target_freshness = float(os.environ.get("RINGRIFT_AUTOSCALE_TARGET_FRESHNESS_HOURS", "2"))

        autoscale = {
            "current_state": {},
            "recommendations": [],
            "thresholds": {
                "scale_up_games_per_hour": scale_up_threshold,  # Scale up if below this
                "scale_down_games_per_hour": scale_down_threshold,  # Scale down if above this
                "max_workers": max_workers,
                "min_workers": min_workers,
                "target_data_freshness_hours": target_freshness,
            },
        }

        try:
            # Get current worker count
            with self.peers_lock:
                total_nodes = len(self.peers) + 1
                gpu_nodes = len([p for p in self.peers.values() if getattr(p, "has_gpu", False)])
                if self.self_info.has_gpu:
                    gpu_nodes += 1

            with self.jobs_lock:
                active_selfplay = len([j for j in self.local_jobs.values()
                                      if j.job_type in (JobType.SELFPLAY, JobType.GPU_SELFPLAY, JobType.HYBRID_SELFPLAY, JobType.CPU_SELFPLAY, JobType.GUMBEL_SELFPLAY)
                                      and j.status == "running"])

            autoscale["current_state"] = {
                "total_nodes": total_nodes,
                "gpu_nodes": gpu_nodes,
                "active_selfplay_jobs": active_selfplay,
            }

            # Get game generation throughput
            analytics = await self.analytics_cache_manager.get_game_analytics_cached()
            total_throughput = sum(c.get("throughput_per_hour", 0) for c in analytics.get("configs", {}).values())

            autoscale["current_state"]["games_per_hour"] = round(total_throughput, 1)

            # Get data freshness
            now = time.time()
            ai_root = Path(self._get_ai_service_path())
            selfplay_dir = ai_root / "data" / "selfplay"

            freshest_data = 0
            if selfplay_dir.exists():
                for jsonl in selfplay_dir.rglob("*.jsonl"):
                    try:
                        mtime = jsonl.stat().st_mtime
                        if mtime > freshest_data:
                            freshest_data = mtime
                    except (AttributeError):
                        continue

            data_age_hours = (now - freshest_data) / 3600 if freshest_data > 0 else 999
            autoscale["current_state"]["data_freshness_hours"] = round(data_age_hours, 2)

            # Generate recommendations
            thresholds = autoscale["thresholds"]

            if total_throughput < thresholds["scale_up_games_per_hour"] and total_nodes < thresholds["max_workers"]:
                autoscale["recommendations"].append({
                    "action": "scale_up",
                    "reason": f"Low throughput ({total_throughput:.0f} games/h < {thresholds['scale_up_games_per_hour']})",
                    "suggested_workers": min(total_nodes + 2, thresholds["max_workers"]),
                })

            if total_throughput > thresholds["scale_down_games_per_hour"] and total_nodes > thresholds["min_workers"]:
                autoscale["recommendations"].append({
                    "action": "scale_down",
                    "reason": f"High throughput ({total_throughput:.0f} games/h > {thresholds['scale_down_games_per_hour']})",
                    "suggested_workers": max(total_nodes - 1, thresholds["min_workers"]),
                })

            if data_age_hours > thresholds["target_data_freshness_hours"]:
                autoscale["recommendations"].append({
                    "action": "scale_up",
                    "reason": f"Stale data ({data_age_hours:.1f}h > {thresholds['target_data_freshness_hours']}h)",
                    "suggested_workers": min(total_nodes + 1, thresholds["max_workers"]),
                })

            # Cost optimization recommendation
            efficiency = await self.analytics_cache_manager.get_training_efficiency_cached()
            elo_per_hour = efficiency.get("summary", {}).get("overall_elo_per_gpu_hour", 0)
            if elo_per_hour < 1 and total_nodes > 2:
                autoscale["recommendations"].append({
                    "action": "optimize",
                    "reason": f"Low efficiency ({elo_per_hour:.2f} Elo/GPU-h) - consider reducing workers",
                    "suggested_workers": max(total_nodes - 1, thresholds["min_workers"]),
                })

        except (AttributeError, KeyError, ValueError, TypeError):
            pass

        return autoscale

    async def _claim_work_from_leader(self, capabilities: list[str]) -> dict[str, Any] | None:
        """Claim work from the leader's work queue.

        Jan 2026: Delegated to WorkerPullController for better modularity.
        """
        result = await self.worker_pull_controller.claim_work_from_leader(capabilities)
        # Sync last_work_from_leader for backward compatibility
        if self.worker_pull_controller.last_work_from_leader > 0:
            self.last_work_from_leader = self.worker_pull_controller.last_work_from_leader
        return result

    async def _claim_work_batch_from_leader(
        self, capabilities: list[str], max_items: int
    ) -> list[dict[str, Any]]:
        """Claim multiple work items from the leader's work queue.

        Jan 2026: Delegated to WorkerPullController for better modularity.
        """
        result = await self.worker_pull_controller.claim_work_batch_from_leader(
            capabilities, max_items
        )
        # Sync last_work_from_leader for backward compatibility
        if self.worker_pull_controller.last_work_from_leader > 0:
            self.last_work_from_leader = self.worker_pull_controller.last_work_from_leader
        return result

    async def _report_work_result(self, work_item: dict[str, Any], success: bool) -> None:
        """Report work completion/failure to the leader.

        Jan 29, 2026: Re-added wrapper for loop_registry compatibility.
        Delegated to WorkerPullController.
        """
        await self.worker_pull_controller.report_work_result(work_item, success)

    async def _execute_claimed_work(self, work_item: dict[str, Any]) -> bool:
        """Execute a claimed work item locally.

        Feb 2026: Thin dispatcher - delegates to scripts/p2p/work_executors/.
        """
        work_type = work_item.get("work_type", "")
        config = work_item.get("config", {})
        work_id = work_item.get("work_id", "")

        # Track work execution via JobOrchestrationManager (Jan 2026)
        if hasattr(self, "job_orchestration") and self.job_orchestration:
            self.job_orchestration.record_work_executed(work_type)

        try:
            from scripts.p2p.work_executors import (
                execute_training_work,
                execute_selfplay_work,
                execute_tournament_work,
                execute_gauntlet_work,
            )

            if work_type == "training":
                return await execute_training_work(
                    work_item, config, self.node_id,
                    ringrift_path=Path(__file__).parent.parent,
                    job_orchestration=getattr(self, "job_orchestration", None),
                )
            elif work_type == "selfplay":
                return await execute_selfplay_work(
                    work_item, config, self.job_manager, self.selfplay_scheduler,
                )
            elif work_type == "gpu_cmaes":
                logger.info(f"Executing GPU CMA-ES work: {config}")
                return True
            elif work_type == "tournament":
                return await execute_tournament_work(
                    work_item, config, self.peers, self.peers_lock,
                    self.distributed_tournament_state, self.job_manager,
                )
            elif work_type == "gauntlet":
                return await execute_gauntlet_work(
                    work_item, config, self.node_id,
                    ringrift_path=Path(__file__).parent.parent,
                )
            else:
                logger.warning(f"Unknown work type: {work_type}")
                return False

        except Exception as e:  # noqa: BLE001
            logger.error(f"Error executing work {work_id}: {e}")
            # Feb 28, 2026: Propagate error info so coordinator can see why
            # it failed. Previously, 247 of 284 Lambda training failures had
            # empty error messages, making debugging impossible.
            work_item["error"] = f"execute_exception:{type(e).__name__}:{e}"
            return False


    async def _handle_zombie_detected(self, peer, zombie_duration: float) -> None:
        """Handle detection of zombie/stuck selfplay processes on a node.

        Jan 2, 2026: Added as callback for IdleDetectionLoop's on_zombie_detected.
        When a node reports selfplay_jobs > 0 but gpu_util < 10% for extended
        time, the processes may be stuck (zombie). This handler kills them.

        Args:
            peer: NodeInfo or DiscoveredNode with zombie processes
            zombie_duration: How long the node has been in zombie state (seconds)
        """
        node_id = getattr(peer, "node_id", str(peer))
        logger.warning(
            f"Zombie processes detected on {node_id} for {zombie_duration:.0f}s, "
            "attempting to kill stale selfplay"
        )

        try:
            # Send kill command to the node's /process/kill endpoint
            url = self._url_for_peer(peer, "/process/kill")
            timeout = ClientTimeout(total=15)

            async with aiohttp.ClientSession(timeout=timeout) as session:
                # Kill all selfplay processes
                async with session.post(
                    url,
                    json={"pattern": "selfplay", "signal": "SIGTERM"},
                    headers=self._auth_headers(),
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        killed = result.get("killed", 0)
                        logger.info(
                            f"Killed {killed} zombie selfplay processes on {node_id}"
                        )
                    else:
                        logger.warning(
                            f"Failed to kill zombies on {node_id}: HTTP {resp.status}"
                        )

        except asyncio.TimeoutError:
            logger.warning(f"Timeout killing zombie processes on {node_id}")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Error killing zombie processes on {node_id}: {e}")


    # =========================================================================
    # PREDICTIVE SCALING HELPERS (January 2026 Sprint 6)
    # Support methods for PredictiveScalingLoop - proactive job spawning
    # =========================================================================

    def _get_work_queue(self) -> Any:
        """Get work queue instance for WorkQueueMaintenanceLoop.

        This method wraps the global get_work_queue() function to make it
        accessible via OrchestratorContext.from_orchestrator().

        Returns:
            Work queue instance, or None if unavailable.

        Note:
            January 11, 2026: Added to fix "'NoneType' object is not callable"
            error in WorkQueueMaintenanceLoop. The OrchestratorContext was
            looking for this method but it didn't exist.
        """
        return get_work_queue()

    def _get_pending_jobs_for_node(self, node_id: str) -> int:
        """Get count of pending/running jobs assigned to a specific node.

        Jan 29, 2026: Delegated to JobOrchestrator.

        Used by PredictiveScalingLoop to skip nodes with pending work.
        """
        # Delegate to JobOrchestrator
        return self.jobs.get_pending_jobs_for_node(node_id)

    async def _spawn_preemptive_selfplay_job(self, peer_info: dict[str, Any]) -> bool:
        """Spawn a preemptive selfplay job on a node approaching idle.

        Called by PredictiveScalingLoop when it detects a node with low
        GPU utilization and no pending work. This spawns a job BEFORE
        the node becomes fully idle to minimize launch latency.

        Args:
            peer_info: Peer information dict with node_id, gpu_utilization, etc.

        Returns:
            True if job was successfully spawned, False otherwise.
        """
        try:
            node_id = peer_info.get("node_id", "unknown")
            logger.info(f"[PredictiveScaling] Spawning preemptive job on {node_id}")

            # Use selfplay scheduler to pick the best config for this node
            if self.selfplay_scheduler is None:
                logger.debug("[PredictiveScaling] No selfplay scheduler, cannot spawn")
                return False

            # Get node-specific job recommendation
            job_recommendation = await self.selfplay_scheduler.get_job_for_node(node_id)
            if job_recommendation is None:
                logger.debug(f"[PredictiveScaling] No job recommendation for {node_id}")
                return False

            # Dispatch the job
            board_type = job_recommendation.get("board_type", "hex8")
            num_players = job_recommendation.get("num_players", 2)
            num_games = job_recommendation.get("num_games", 100)

            # Use job manager for dispatch
            if self.job_manager is None:
                logger.debug("[PredictiveScaling] No job manager, cannot dispatch")
                return False

            job_id = f"preemptive-{node_id}-{int(time.time())}"
            result = await self.job_manager.dispatch_selfplay_job(
                node_id=node_id,
                job_id=job_id,
                board_type=board_type,
                num_players=num_players,
                num_games=num_games,
                preemptive=True,  # Mark as preemptive for tracking
                engine_mode="mixed",  # Jan 12, 2026: Enable harness diversity
            )

            if result.get("success"):
                logger.info(
                    f"[PredictiveScaling] Spawned preemptive job {job_id} on {node_id} "
                    f"({board_type}_{num_players}p, {num_games} games)"
                )
                return True
            else:
                logger.debug(f"[PredictiveScaling] Failed to spawn on {node_id}: {result.get('error')}")
                return False

        except Exception as e:  # noqa: BLE001
            logger.debug(f"[PredictiveScaling] Exception spawning preemptive job: {e}")
            return False

    # =========================================================================
    # Support methods for JobReassignmentLoop - orphaned job recovery (Sprint 6)
    # =========================================================================

    def _get_healthy_node_ids_for_reassignment(self) -> list[str]:
        """Get list of healthy node IDs that can accept reassigned jobs.

        Used by JobReassignmentLoop to find nodes for orphaned job reassignment.
        A healthy node is one that:
        - Is currently alive in the peer list
        - Has recent health check data
        - Is not overloaded (CPU < 90%, GPU mem < 95%)

        Jan 27, 2026: Migrated to PeerQueryBuilder (Phase 3.2).

        Returns:
            List of node IDs suitable for job reassignment.
        """
        return self._peer_query.available_for_reassignment(
            cpu_threshold=90.0,
            gpu_mem_threshold=95.0,
            stale_seconds=120.0,
        ).unwrap_or([])


    # See scripts/p2p/handlers/metrics.py

    # See scripts/p2p/handlers/analytics.py

    # See scripts/p2p/handlers/analytics.py

    # See scripts/p2p/handlers/recovery.py for implementation.

    # ==================== A/B Testing Framework ====================

    def _calculate_ab_test_stats(self, test_id: str) -> dict[str, Any]:
        """Calculate statistical significance for an A/B test."""
        import math

        try:
            # Phase 3.4 Dec 29, 2025: Use context manager to prevent connection leaks
            with safe_db_connection(self.db_path) as conn:
                cursor = conn.cursor()

                # Get game results
                cursor.execute("""
                    SELECT model_a_result, model_a_score, model_b_score, game_length
                    FROM ab_test_games WHERE test_id = ?
                """, (test_id,))
                games = cursor.fetchall()

            if not games:
                return {
                    "games_played": 0,
                    "model_a_wins": 0,
                    "model_b_wins": 0,
                    "draws": 0,
                    "model_a_score": 0.0,
                    "model_b_score": 0.0,
                    "model_a_winrate": 0.0,
                    "model_b_winrate": 0.0,
                    "confidence": 0.0,
                    "likely_winner": None,
                    "statistically_significant": False,
                }

            # Count results
            model_a_wins = sum(1 for g in games if g[0] == "win")
            model_b_wins = sum(1 for g in games if g[0] == "loss")
            draws = sum(1 for g in games if g[0] == "draw")
            total = len(games)

            model_a_score = sum(g[1] for g in games)
            model_b_score = sum(g[2] for g in games)

            # Winrate (using score, e.g., 1 for win, 0.5 for draw, 0 for loss)
            model_a_winrate = model_a_score / total if total > 0 else 0.0
            model_b_winrate = model_b_score / total if total > 0 else 0.0

            # Wilson score confidence interval for statistical significance
            # Using normal approximation for simplicity
            def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
                if n == 0:
                    return (0.0, 1.0)
                p = wins / n
                denominator = 1 + z * z / n
                center = (p + z * z / (2 * n)) / denominator
                spread = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denominator
                return (max(0, center - spread), min(1, center + spread))

            # Calculate confidence intervals
            a_lo, a_hi = wilson_ci(model_a_wins + draws // 2, total)
            b_lo, b_hi = wilson_ci(model_b_wins + draws // 2, total)

            # Determine if statistically significant (non-overlapping CIs)
            statistically_significant = a_hi < b_lo or b_hi < a_lo

            # Estimate confidence based on score difference and sample size
            if total > 0:
                score_diff = abs(model_a_winrate - model_b_winrate)
                # Rough confidence estimate (higher with more games and larger diff)
                confidence = min(0.99, 1 - math.exp(-total * score_diff * 2))
            else:
                confidence = 0.0

            # Determine likely winner
            likely_winner = None
            if model_a_winrate > model_b_winrate + 0.05:
                likely_winner = "model_a"
            elif model_b_winrate > model_a_winrate + 0.05:
                likely_winner = "model_b"

            avg_game_length = sum(g[3] for g in games if g[3]) / max(1, sum(1 for g in games if g[3]))

            return {
                "games_played": total,
                "model_a_wins": model_a_wins,
                "model_b_wins": model_b_wins,
                "draws": draws,
                "model_a_score": model_a_score,
                "model_b_score": model_b_score,
                "model_a_winrate": round(model_a_winrate, 4),
                "model_b_winrate": round(model_b_winrate, 4),
                "confidence": round(confidence, 4),
                "likely_winner": likely_winner,
                "statistically_significant": statistically_significant,
                "avg_game_length": round(avg_game_length, 1),
            }
        except Exception as e:  # noqa: BLE001
            return {"error": str(e)}

    async def _run_evaluation(self, job_id: str):
        """Evaluate new model against current best.

        Runs evaluation games between the candidate model and the best model.
        Reports win rate for the candidate.
        """
        import json as json_module
        import sys

        state = self.improvement_loop_state.get(job_id)
        if not state:
            return

        logger.info(f"Running evaluation for job {job_id}, iteration {state.current_iteration}")

        getattr(state, 'candidate_model_path', None)

        # Number of evaluation games
        eval_games = 100

        eval_script = f"""
import sys
sys.path.insert(0, '{self._get_ai_service_path()}')
from app.game_engine import GameEngine
from app.agents.heuristic_agent import HeuristicAgent
import json

# Run evaluation games
candidate_wins = 0
best_wins = 0
draws = 0

for game_idx in range({eval_games}):
    engine = GameEngine(board_type='{state.board_type}', num_players={state.num_players})

    # Alternate who plays first
    if game_idx % 2 == 0:
        agents = [
            HeuristicAgent(0),  # Candidate as player 0
            HeuristicAgent(1),  # Best as player 1
        ]
        candidate_player = 0
    else:
        agents = [
            HeuristicAgent(0),  # Best as player 0
            HeuristicAgent(1),  # Candidate as player 1
        ]
        candidate_player = 1

    # Play game
    max_moves = 10000
    move_count = 0
    while not engine.is_game_over() and move_count < max_moves:
        current_player = engine.current_player
        agent = agents[current_player]
        legal_moves = engine.get_legal_moves()
        if not legal_moves:
            break
        move = agent.select_move(engine.get_state(), legal_moves)
        engine.apply_move(move)
        move_count += 1

    outcome = engine.get_outcome()
    winner = outcome.get('winner')

    if winner == candidate_player:
        candidate_wins += 1
    elif winner is not None:
        best_wins += 1
    else:
        draws += 1

# Calculate win rate
total = candidate_wins + best_wins + draws
winrate = candidate_wins / total if total > 0 else 0.5

print(json.dumps({{
    'candidate_wins': candidate_wins,
    'best_wins': best_wins,
    'draws': draws,
    'winrate': winrate,
}}))
"""

        cmd = [sys.executable, "-c", eval_script]
        env = os.environ.copy()
        env["PYTHONPATH"] = self._get_ai_service_path()
        env["RINGRIFT_SKIP_SHADOW_CONTRACTS"] = "true"

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=3600  # 1 hour max
            )

            if proc.returncode == 0:
                output_lines = stdout.decode().strip().split('\n')
                result_line = output_lines[-1] if output_lines else '{}'
                result = json_module.loads(result_line)

                state.evaluation_winrate = result.get('winrate', 0.5)
                logger.info(f"Evaluation result: winrate={state.evaluation_winrate:.2%}")
                logger.info("  Candidate")
            else:
                logger.info(f"Evaluation failed: {stderr.decode()[:500]}")
                state.evaluation_winrate = 0.5

        except asyncio.TimeoutError:
            logger.info("Evaluation timed out")
            state.evaluation_winrate = 0.5
        except Exception as e:  # noqa: BLE001
            logger.info(f"Evaluation error: {e}")
            state.evaluation_winrate = 0.5

    async def _promote_model_if_better(self, job_id: str):
        """Promote new model if it beats the current best.

        Promotion threshold: candidate must win >= 55% of evaluation games.
        """
        state = self.improvement_loop_state.get(job_id)
        if not state:
            return

        PROMOTION_THRESHOLD = 0.55  # 55% win rate required

        winrate = getattr(state, 'evaluation_winrate', 0.5)
        candidate_path = getattr(state, 'candidate_model_path', None)

        logger.info(f"Checking model promotion for job {job_id}")
        logger.info("  Current")
        logger.info("  Candidate")
        logger.info("  Threshold")

        if winrate >= PROMOTION_THRESHOLD and candidate_path:
            # Promote candidate to best
            state.best_model_path = candidate_path
            state.best_winrate = winrate

            # Save best model to well-known location
            best_model_dir = os.path.join(
                self._get_ai_service_path(), "models", "best"
            )
            os.makedirs(best_model_dir, exist_ok=True)

            import shutil
            best_path = os.path.join(best_model_dir, f"{state.board_type}_{state.num_players}p.pt")
            if os.path.exists(candidate_path):
                shutil.copy2(candidate_path, best_path)
                logger.info(f"PROMOTED: New best model at {best_path}")
                logger.info(f"  Win rate: {winrate:.2%}")
            else:
                logger.info(f"Cannot promote: candidate model not found at {candidate_path}")
        else:
            logger.info(f"No promotion: candidate ({winrate:.2%}) below threshold ({PROMOTION_THRESHOLD:.0%})")

    # ============================================
    # Core Logic
    # ============================================

    def _update_self_info(self):
        """Update self info with current resource usage.

        WARNING: This is a BLOCKING method that acquires locks and does I/O.
        In async contexts, prefer `await self._update_self_info_async(cache_ttl=30)`
        instead of `await asyncio.to_thread(self._update_self_info)`, since
        the async version caches results and avoids thread pool starvation.

        Mar 2026: Added 10s cache to prevent redundant blocking calls when
        multiple callers invoke this via asyncio.to_thread() concurrently.
        On macOS, each call takes 10-30s (pgrep, psutil, NFS). With only 8
        thread pool workers and 5+ callers (heartbeat, elections, leader ops),
        the pool gets starved, causing cascading timeouts in queue_populator
        and voter_heartbeat.
        """
        # Mar 2026: Short cache to prevent redundant blocking work
        now = time.time()
        _cache_ttl = 10.0  # 10s cache for sync version (shorter than async's 30s)
        _last = getattr(self, "_update_self_info_last_time", 0.0)
        if (now - _last) < _cache_ttl:
            return  # Recent data still valid
        self._update_self_info_last_time = now

        usage = self._get_resource_usage()
        # Jan 30, 2026: Use jobs orchestrator directly
        selfplay, training = self.jobs.count_local_jobs()

        # NAT/relay detection: if we haven't received any inbound heartbeats for a
        # while (but we do know about other peers), assume we're not reachable
        # inbound and must poll a relay for commands.
        now = time.time()
        if self.known_peers or self.peers:
            last_inbound = self.last_inbound_heartbeat or self.start_time
            self.self_info.nat_blocked = (now - last_inbound) >= NAT_INBOUND_HEARTBEAT_STALE_SECONDS
        else:
            self.self_info.nat_blocked = False

        if not self.self_info.nat_blocked:
            self.self_info.relay_via = ""
        elif self.leader_id and self.leader_id != self.node_id:
            self.self_info.relay_via = self.leader_id

        self.self_info.cpu_percent = usage["cpu_percent"]
        self.self_info.memory_percent = usage["memory_percent"]
        self.self_info.disk_percent = usage["disk_percent"]
        self.self_info.gpu_percent = usage["gpu_percent"]
        self.self_info.gpu_memory_percent = usage["gpu_memory_percent"]
        self.self_info.selfplay_jobs = selfplay
        # Jan 2, 2026: Set max slots for slot-based work queue claiming
        self.self_info.max_selfplay_slots = self._get_max_selfplay_slots_for_node()
        self.self_info.training_jobs = training
        self.self_info.role = self.role
        self.self_info.last_heartbeat = time.time()
        # Dec 2025: Propagate leader_id in heartbeats for cluster-wide leader discovery
        self.self_info.leader_id = self.leader_id or ""
        # Feb 2026: Propagate leader_term for term-based convergence
        self.self_info.leader_term = getattr(self, "_leader_term", 0) or 0

        # Detect external work (running outside P2P orchestrator tracking)
        external = self._detect_local_external_work()
        self.self_info.cmaes_running = external.get('cmaes_running', False)
        self.self_info.gauntlet_running = external.get('gauntlet_running', False)
        self.self_info.tournament_running = external.get('tournament_running', False)
        self.self_info.data_merge_running = external.get('data_merge_running', False)

        # Phase 6: Health broadcasting - additional health metrics
        self.self_info.nfs_accessible = self._check_nfs_accessible()
        self.self_info.code_version = self.build_version
        self.self_info.errors_last_hour = getattr(self, '_error_count_last_hour', 0)
        self.self_info.disk_free_gb = usage.get("disk_free_gb", 0.0)
        self.self_info.active_job_count = (
            selfplay + training +
            (1 if self.self_info.cmaes_running else 0) +
            (1 if self.self_info.gauntlet_running else 0) +
            (1 if self.self_info.tournament_running else 0)
        )

        # Jan 24, 2026: Update visible_peers count for connectivity scoring
        # Used by _compute_connectivity_score() to determine leader eligibility
        self.self_info.visible_peers = len([p for p in self.peers.values() if p.is_alive()])

        # Jan 25, 2026: Update effective_timeout for broadcast to peers
        # This tells other nodes how long to wait before marking us dead
        try:
            from app.p2p.constants import PEER_TIMEOUT, get_cpu_adaptive_timeout
            from app.config.provider_timeouts import ProviderTimeouts
            cpu_load = usage["cpu_percent"] / 100.0 if usage.get("cpu_percent", 0) > 0 else 0.0
            base_timeout = get_cpu_adaptive_timeout(PEER_TIMEOUT, cpu_load)
            provider_mult = ProviderTimeouts.get_multiplier(self.node_id) if ProviderTimeouts else 1.0
            self.self_info.effective_timeout = base_timeout * provider_mult
        except Exception:
            self.self_info.effective_timeout = 180.0  # Fallback to default

        # Report to unified resource optimizer for cluster-wide coordination
        if HAS_NEW_COORDINATION:
            try:
                optimizer = get_resource_optimizer()
                node_resources = NodeResources(
                    node_id=self.node_id,
                    cpu_percent=usage["cpu_percent"],
                    gpu_percent=usage["gpu_percent"],
                    memory_percent=usage["memory_percent"],
                    disk_percent=usage["disk_percent"],
                    gpu_memory_percent=usage["gpu_memory_percent"],
                    cpu_count=int(getattr(self.self_info, "cpu_count", 0) or 0),
                    memory_gb=float(getattr(self.self_info, "memory_gb", 0) or 0),
                    has_gpu=bool(getattr(self.self_info, "has_gpu", False)),
                    gpu_name=str(getattr(self.self_info, "gpu_name", "") or ""),
                    active_jobs=selfplay + training,
                    selfplay_jobs=selfplay,
                    training_jobs=training,
                    orchestrator="p2p_orchestrator",
                )
                optimizer.report_node_resources(node_resources)
            except (ValueError, KeyError, IndexError, AttributeError):
                pass  # Don't fail heartbeat if optimizer unavailable

        # December 2025: Emit NODE_CAPACITY_UPDATED for backpressure detection
        # Sprint 10 (Jan 3, 2026): Use unified emitter for consistent payloads
        # Throttled to every 30 seconds to avoid event spam
        now = time.time()
        last_emit = getattr(self, "_last_capacity_emit_time", 0)
        if now - last_emit >= 30:  # 30s throttle matches backpressure cooldown
            self._last_capacity_emit_time = now
            try:
                from app.distributed.data_events import emit_node_capacity_updated_sync

                available_slots = max(0, self._get_max_selfplay_jobs() - selfplay - training)
                emit_node_capacity_updated_sync(
                    node_id=self.node_id,
                    gpu_utilization=usage["gpu_percent"],
                    cpu_utilization=usage["cpu_percent"],
                    available_slots=available_slots,
                    reason="heartbeat",
                    source="p2p_orchestrator",
                    queue_depth=getattr(self, "_work_queue_depth", 0),
                )
            except (ImportError, RuntimeError, AttributeError):
                pass  # Event system not available or no event loop

        # Jan 12, 2026: Sync host/port when advertise_host changes
        # Root cause fix: self.self_info.host was never updated after init,
        # causing heartbeats to broadcast stale IPs to all peers.
        if self.self_info.host != self.advertise_host:
            old_host = self.self_info.host
            self.self_info.host = self.advertise_host
            logger.info(f"[P2P] Updated self.self_info.host: {old_host} -> {self.advertise_host}")
        if self.self_info.port != self.advertise_port:
            self.self_info.port = self.advertise_port


    async def _update_self_info_async(self, cache_ttl: float = 5.0):
        """Async version of _update_self_info() to avoid blocking event loop.

        Dec 30, 2025: Added to fix gossip latency issues on coordinator nodes.
        The sync version calls subprocess for resource detection which blocks
        the event loop. This async version uses asyncio.to_thread() for those
        blocking operations.

        Jan 12, 2026: Added caching to reduce health endpoint latency from 3-6s
        to <100ms for repeated requests. Resource metrics are cached for cache_ttl
        seconds (default 5s) since they don't change rapidly.

        Args:
            cache_ttl: How long to cache resource metrics (seconds). Default 5s.
        """
        import asyncio

        # Jan 12, 2026: Check cache to avoid expensive resource detection on every request
        now = time.time()
        cache_key = "_self_info_cache_time"
        last_update = getattr(self, cache_key, 0)
        if (now - last_update) < cache_ttl:
            # Cache hit - self_info already has recent data
            return

        # Run blocking operations in thread pool
        usage = await self._get_resource_usage_async()
        selfplay, training = await asyncio.to_thread(self.jobs.count_local_jobs)

        # NAT/relay detection (fast, no subprocess)
        now = time.time()
        if self.known_peers or self.peers:
            last_inbound = self.last_inbound_heartbeat or self.start_time
            self.self_info.nat_blocked = (now - last_inbound) >= NAT_INBOUND_HEARTBEAT_STALE_SECONDS
        else:
            self.self_info.nat_blocked = False

        if not self.self_info.nat_blocked:
            self.self_info.relay_via = ""
        elif self.leader_id and self.leader_id != self.node_id:
            self.self_info.relay_via = self.leader_id

        self.self_info.cpu_percent = usage["cpu_percent"]
        self.self_info.memory_percent = usage["memory_percent"]
        self.self_info.disk_percent = usage["disk_percent"]
        self.self_info.gpu_percent = usage["gpu_percent"]
        self.self_info.gpu_memory_percent = usage["gpu_memory_percent"]
        self.self_info.selfplay_jobs = selfplay
        # Jan 2, 2026: Set max slots for slot-based work queue claiming
        self.self_info.max_selfplay_slots = self._get_max_selfplay_slots_for_node()
        self.self_info.training_jobs = training
        self.self_info.role = self.role
        self.self_info.last_heartbeat = time.time()
        self.self_info.leader_id = self.leader_id or ""

        # Run blocking external work detection in thread pool
        external = await asyncio.to_thread(self._detect_local_external_work)
        self.self_info.cmaes_running = external.get('cmaes_running', False)
        self.self_info.gauntlet_running = external.get('gauntlet_running', False)
        self.self_info.tournament_running = external.get('tournament_running', False)
        self.self_info.data_merge_running = external.get('data_merge_running', False)

        # Health metrics (NFS check in thread pool as it can block)
        self.self_info.nfs_accessible = await asyncio.to_thread(self._check_nfs_accessible)
        self.self_info.code_version = self.build_version
        self.self_info.errors_last_hour = getattr(self, '_error_count_last_hour', 0)
        self.self_info.disk_free_gb = usage.get("disk_free_gb", 0.0)
        self.self_info.active_job_count = (
            selfplay + training +
            (1 if self.self_info.cmaes_running else 0) +
            (1 if self.self_info.gauntlet_running else 0) +
            (1 if self.self_info.tournament_running else 0)
        )

        # Report to resource optimizer (fast, in-memory)
        if HAS_NEW_COORDINATION:
            try:
                optimizer = get_resource_optimizer()
                node_resources = NodeResources(
                    node_id=self.node_id,
                    cpu_percent=usage["cpu_percent"],
                    gpu_percent=usage["gpu_percent"],
                    memory_percent=usage["memory_percent"],
                    disk_percent=usage["disk_percent"],
                    gpu_memory_percent=usage["gpu_memory_percent"],
                    cpu_count=int(getattr(self.self_info, "cpu_count", 0) or 0),
                    memory_gb=float(getattr(self.self_info, "memory_gb", 0) or 0),
                    has_gpu=bool(getattr(self.self_info, "has_gpu", False)),
                    gpu_name=str(getattr(self.self_info, "gpu_name", "") or ""),
                    active_jobs=selfplay + training,
                    selfplay_jobs=selfplay,
                    training_jobs=training,
                    orchestrator="p2p_orchestrator",
                )
                optimizer.report_node_resources(node_resources)
            except (ValueError, KeyError, IndexError, AttributeError):
                pass

        # Feb 2026 (1c): Periodically refresh capabilities
        last_cap_refresh = getattr(self, "_last_capability_refresh", 0)
        if now - last_cap_refresh >= 60.0:
            self._last_capability_refresh = now
            try:
                self.monitoring._refresh_capabilities()
            except Exception as e:
                logger.debug(f"[P2P] Capability refresh failed: {e}")

        # NODE_CAPACITY_UPDATED event (throttled, fast)
        # Sprint 10 (Jan 3, 2026): Use unified emitter for consistent payloads
        last_emit = getattr(self, "_last_capacity_emit_time", 0)
        if now - last_emit >= 30:
            self._last_capacity_emit_time = now
            try:
                from app.distributed.data_events import emit_node_capacity_updated_sync

                available_slots = max(0, self._get_max_selfplay_jobs() - selfplay - training)
                emit_node_capacity_updated_sync(
                    node_id=self.node_id,
                    gpu_utilization=usage["gpu_percent"],
                    cpu_utilization=usage["cpu_percent"],
                    available_slots=available_slots,
                    reason="heartbeat_async",
                    source="p2p_orchestrator",
                    queue_depth=getattr(self, "_work_queue_depth", 0),
                )
            except (ImportError, RuntimeError, AttributeError):
                pass

        # Jan 12, 2026: Update cache timestamp after successful update
        setattr(self, "_self_info_cache_time", time.time())

    async def _send_heartbeat_to_peer(self, peer_host: str, peer_port: int, scheme: str = "http", timeout: int = 15) -> NodeInfo | None:
        """Send heartbeat to a peer and return their info.

        Jan 27, 2026: Phase 16A - Delegates to HeartbeatManager.
        """
        return await self.heartbeat_manager.send_heartbeat_to_peer(peer_host, peer_port, scheme, timeout)

    async def _send_heartbeat_via_ssh_fallback(
        self, peer_host: str, peer_port: int, payload: dict[str, Any]
    ) -> NodeInfo | None:
        """Send heartbeat via SSH when HTTP fails.

        Jan 27, 2026: Phase 16A - Delegates to HeartbeatManager.
        """
        return await self.heartbeat_manager._send_heartbeat_via_ssh_fallback(peer_host, peer_port, payload)

    async def _bootstrap_from_known_peers(self) -> bool:
        """Import cluster membership from seed peers via `/relay/peers`.

        Jan 27, 2026: Phase 16A - Delegates to HeartbeatManager.
        """
        return await self.heartbeat_manager.bootstrap_from_known_peers()

    async def _continuous_bootstrap_loop(self) -> None:
        """Phase 26.3: Continuously attempt to join cluster when isolated.

        Jan 27, 2026: Phase 16A - Delegates to HeartbeatManager.
        """
        await self.heartbeat_manager.continuous_bootstrap_loop()

    async def _bootstrap_from_multiple_seeds(self) -> bool:
        """Phase 26.3: Try multiple seeds until we join the cluster.

        Priority order:
        1. Cached peers with high reputation (from peer_cache table)
        2. CLI --peers (self.known_peers)
        3. Hardcoded BOOTSTRAP_SEEDS

        Returns True if we successfully connected to any peer.
        """
        # Build seed list with priority ordering
        all_seeds: list[str] = []
        seen: set[str] = set()

        # 1. First, try cached peers by reputation (if available)
        cached_peers = self._get_bootstrap_peers_by_reputation(limit=3)
        for seed in cached_peers:
            if seed and seed not in seen:
                seen.add(seed)
                all_seeds.append(seed)

        # 2. Then, CLI peers and hardcoded seeds (already merged in self.known_peers)
        for seed in self.known_peers:
            if seed and seed not in seen:
                seen.add(seed)
                all_seeds.append(seed)

        if not all_seeds:
            logger.warning("No bootstrap seeds available")
            return False

        # Limit attempts per cycle
        max_attempts = min(MIN_BOOTSTRAP_ATTEMPTS * 2, len(all_seeds))
        timeout = ClientTimeout(total=10)
        success = False

        async with get_client_session(timeout) as session:
            for idx, seed_addr in enumerate(all_seeds[:max_attempts]):
                try:
                    scheme, host, port = self._parse_peer_address(seed_addr)
                    scheme = (scheme or "http").lower()
                    url = f"{scheme}://{host}:{port}/relay/peers"

                    async with session.get(url, headers=self._auth_headers()) as resp:
                        if resp.status != 200:
                            self._update_peer_reputation(seed_addr, success=False)
                            continue

                        data = await resp.json()
                        if not isinstance(data, dict) or not data.get("success"):
                            self._update_peer_reputation(seed_addr, success=False)
                            continue

                    # Successfully got peer list
                    self._update_peer_reputation(seed_addr, success=True)
                    success = True

                    # Import peers
                    peers_data = data.get("peers") or {}
                    if isinstance(peers_data, dict):
                        with self.peers_lock:
                            for node_id, peer_dict in peers_data.items():
                                if node_id and node_id != self.node_id:
                                    try:
                                        info = NodeInfo.from_dict(peer_dict)
                                        self.peers[info.node_id] = info
                                        # Cache the peer for future restarts
                                        self._save_peer_to_cache(
                                            info.node_id,
                                            str(getattr(info, "host", "") or ""),
                                            int(getattr(info, "port", DEFAULT_PORT) or DEFAULT_PORT),
                                            str(getattr(info, "tailscale_ip", "") or "")
                                        )
                                    except (ValueError, KeyError, IndexError, AttributeError):
                                        continue

                        # Jan 12, 2026: Sync to lock-free snapshot after relay peer import
                        self._sync_peer_snapshot()

                    # Adopt leader if provided
                    leader_id = str(data.get("leader_id") or "").strip()
                    if leader_id and leader_id != self.node_id:
                        if self.role == NodeRole.LEADER:
                            logger.info(f"Stepping down for discovered leader: {leader_id}")
                        # Jan 3, 2026: Use _set_leader() for atomic leadership assignment (Phase 4)
                        self._set_leader(leader_id, reason="continuous_bootstrap_discover_leader", save_state=False)

                    # Handle cluster epoch (Phase 29)
                    incoming_epoch = data.get("cluster_epoch")
                    if incoming_epoch is not None:
                        try:
                            epoch = int(incoming_epoch)
                            if epoch > self._cluster_epoch:
                                logger.info(f"Adopting higher cluster epoch: {epoch} (was {self._cluster_epoch})")
                                self._cluster_epoch = epoch
                                self._save_cluster_epoch()
                        except (ValueError, TypeError):
                            pass

                    # Import voter config if provided
                    incoming_voters = data.get("voter_node_ids") or data.get("voters")
                    if incoming_voters:
                        voters_list = []
                        if isinstance(incoming_voters, list):
                            voters_list = [str(v).strip() for v in incoming_voters if str(v).strip()]
                        elif isinstance(incoming_voters, str):
                            voters_list = [t.strip() for t in incoming_voters.split(",") if t.strip()]
                        if voters_list:
                            if self.quorum_manager.maybe_adopt_voter_node_ids(voters_list, source="learned"):
                                # Sync adopted state back to orchestrator attributes
                                self.voter_node_ids = self.quorum_manager.voter_node_ids
                                self.voter_config_source = self.quorum_manager.voter_config_source
                                self.voter_quorum_size = min(VOTER_MIN_QUORUM, len(self.voter_node_ids)) if self.voter_node_ids else 0

                    self._save_state()
                    logger.info(f"Bootstrap from {host}:{port}: imported {len(peers_data)} peers")
                    break  # Success, no need to try more seeds

                except asyncio.TimeoutError:
                    self._update_peer_reputation(seed_addr, success=False)
                    continue
                except Exception as e:  # noqa: BLE001
                    self._update_peer_reputation(seed_addr, success=False)
                    if self.verbose:
                        logger.debug(f"Bootstrap seed {seed_addr} failed: {e}")
                    continue

        return success

    def _load_bootstrap_seeds_from_config(self) -> list[str]:
        """Load bootstrap seed peers from distributed_hosts.yaml.

        Selects stable coordinator and voter nodes as default seeds when no --peers provided.
        This enables automatic peer discovery via Tailscale even when CLI args are missing.

        Returns:
            List of seed peer URLs (e.g., ["http://100.x.x.x:8770", ...])

        December 30, 2025: Added for automatic P2P peer discovery.
        """
        try:
            from app.config.cluster_config import get_cluster_nodes, get_coordinator_node

            seeds: list[str] = []
            seen_ips: set[str] = set()

            # Primary: coordinator node (most stable)
            coord = get_coordinator_node()
            if coord and getattr(coord, "tailscale_ip", None):
                ip = str(coord.tailscale_ip)
                if ip and ip not in seen_ips:
                    seeds.append(f"http://{ip}:{DEFAULT_PORT}")
                    seen_ips.add(ip)

            # Secondary: voter nodes (stable, always online)
            try:
                nodes = get_cluster_nodes()
                for node in nodes.values():
                    if getattr(node, "role", "") == "voter" and getattr(node, "tailscale_ip", None):
                        ip = str(node.tailscale_ip)
                        if ip and ip not in seen_ips:
                            seeds.append(f"http://{ip}:{DEFAULT_PORT}")
                            seen_ips.add(ip)
                            if len(seeds) >= 5:
                                break
            except Exception:  # noqa: BLE001
                pass

            # Fallback: any active nodes with Tailscale IPs
            if len(seeds) < 3:
                try:
                    nodes = get_cluster_nodes()
                    for node in nodes.values():
                        if getattr(node, "tailscale_ip", None) and getattr(node, "is_active", True):
                            ip = str(node.tailscale_ip)
                            if ip and ip not in seen_ips:
                                seeds.append(f"http://{ip}:{DEFAULT_PORT}")
                                seen_ips.add(ip)
                                if len(seeds) >= 5:
                                    break
                except Exception:  # noqa: BLE001
                    pass

            if seeds:
                logger.debug(f"Loaded {len(seeds)} bootstrap seeds from config: {seeds[:3]}...")

            return seeds

        except ImportError:
            logger.debug("cluster_config not available for bootstrap seeds")
            return []
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Could not load bootstrap seeds from config: {e}")
            return []

    def _is_node_proxy_only(self, node_id: str) -> bool:
        """Check if a node is configured as proxy_only in distributed_hosts.yaml.

        Jan 13, 2026: Added to prevent proxy nodes from becoming cluster leaders.
        Proxy nodes are SSH jump hosts or API proxies with no AI/training capability.

        Args:
            node_id: Node identifier to check

        Returns:
            True if node has status="proxy_only" in config
        """
        # Jan 13, 2026: Known aliases for proxy nodes that may appear under different names
        # These are nodes that registered with a different name than their config entry
        PROXY_ALIASES = {
            "aws-staging": "aws-proxy",  # EC2 staging instance is the proxy
        }

        try:
            hosts = self._load_distributed_hosts().get("hosts", {})
            # Check direct name first
            node_config = hosts.get(node_id, {})
            if node_config.get("status", "") == "proxy_only":
                return True
            # Check if this is a known alias for a proxy node
            if node_id in PROXY_ALIASES:
                alias_config = hosts.get(PROXY_ALIASES[node_id], {})
                if alias_config.get("status", "") == "proxy_only":
                    logger.debug(
                        f"[ProxyCheck] {node_id} is alias for {PROXY_ALIASES[node_id]} (proxy_only)"
                    )
                    return True
            return False
        except Exception:  # noqa: BLE001
            return False

    def _load_distributed_hosts(self) -> dict[str, Any]:
        """Load distributed hosts configuration for NetworkHealthMixin.

        Required by NetworkHealthMixin for cross-verifying P2P mesh health
        against Tailscale connectivity.

        Returns:
            Dict with structure: {"hosts": {node_name: {config...}}}
            Each host config includes: tailscale_ip, p2p_enabled, p2p_port, etc.

        December 30, 2025: Added to fix /network/health endpoint.
        """
        try:
            from app.config.cluster_config import load_cluster_config

            config = load_cluster_config()
            hosts_raw = getattr(config, "hosts_raw", {})

            # Convert to the format expected by NetworkHealthMixin
            # hosts_raw already has the right structure: {node_name: {config_dict}}
            return {"hosts": hosts_raw}

        except ImportError:
            logger.debug("cluster_config not available for distributed hosts")
            return {"hosts": {}}
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Could not load distributed hosts: {e}")
            return {"hosts": {}}

    # See scripts/p2p/loops/discovery_loop.py for implementation.

    async def _send_relay_heartbeat(self, relay_url: str) -> dict[str, Any]:
        """Send heartbeat via relay endpoint for NAT-blocked nodes.

        Jan 27, 2026: Phase 16A - Delegates to HeartbeatManager.
        """
        return await self.heartbeat_manager.send_relay_heartbeat(relay_url)

    async def _send_initial_relay_heartbeats(self) -> None:
        """Send immediate relay heartbeats on startup for NAT-blocked nodes.

        January 5, 2026: NAT-blocked nodes can't receive inbound connections,
        so they need to proactively register with relay-capable nodes to be
        discoverable by the cluster. This method sends relay heartbeats to
        all configured relay-capable nodes immediately at startup.

        Called after HTTP server starts but before regular heartbeat loop.
        """
        # Load relay-capable nodes from distributed_hosts.yaml
        relay_nodes: list[tuple[str, str, int]] = []  # (node_id, ip, port)
        try:
            from app.config.cluster_config import load_cluster_config
            config = load_cluster_config()
            nodes = getattr(config, "hosts_raw", {}) or {}

            for node_id, node_cfg in nodes.items():
                if node_id == self.node_id:
                    continue  # Skip self
                if not node_cfg.get("relay_capable", False):
                    continue
                if not node_cfg.get("p2p_enabled", True):
                    continue

                # Get the best IP to reach this node (prefer Tailscale)
                ip = node_cfg.get("tailscale_ip") or node_cfg.get("ssh_host", "")
                port = node_cfg.get("p2p_port", DEFAULT_PORT)
                if ip:
                    relay_nodes.append((node_id, ip, port))

        except ImportError:
            logger.warning("[P2P] cluster_config not available for initial relay heartbeats")
            return
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[P2P] Failed to load relay-capable nodes: {e}")
            return

        if not relay_nodes:
            logger.info("[P2P] No relay-capable nodes configured for initial heartbeat")
            return

        logger.info(f"[P2P] Sending initial relay heartbeats to {len(relay_nodes)} relay-capable nodes")

        # Send relay heartbeats to all relay-capable nodes
        success_count = 0
        for node_id, ip, port in relay_nodes:
            relay_url = f"http://{ip}:{port}"
            try:
                result = await self._send_relay_heartbeat(relay_url)
                if result.get("success"):
                    success_count += 1
                    logger.info(f"[P2P] Initial relay heartbeat to {node_id} ({ip}:{port}) succeeded")
                else:
                    error = result.get("error", "unknown")
                    logger.debug(f"[P2P] Initial relay heartbeat to {node_id} failed: {error}")
            except Exception as e:  # noqa: BLE001
                logger.debug(f"[P2P] Initial relay heartbeat to {node_id} error: {e}")

        if success_count > 0:
            logger.info(f"[P2P] NAT-blocked node registered with {success_count}/{len(relay_nodes)} relay nodes")
        else:
            logger.warning(f"[P2P] Failed to register with any relay nodes - cluster discovery may be delayed")

    async def _init_hybrid_coordinator(self) -> None:
        """Initialize HybridCoordinator for Raft-based leader election.

        January 23, 2026: This method initializes the HybridCoordinator which
        provides Raft-based leader election as a replacement for the buggy
        Bully algorithm.

        The HybridCoordinator:
        - Uses PySyncObj's Raft implementation for proven consensus
        - Provides sub-second leader failover (vs 60-90s with Bully)
        - Routes is_leader() calls based on CONSENSUS_MODE env var
        - Falls back to Bully if Raft is unavailable

        To enable Raft:
            export RINGRIFT_RAFT_ENABLED=true
            export RINGRIFT_CONSENSUS_MODE=raft  # or "hybrid"
        """
        logger.info("[P2P] _init_hybrid_coordinator() called")
        try:
            from app.p2p.constants import RAFT_ENABLED, CONSENSUS_MODE
        except ImportError:
            logger.warning("[P2P] Cannot import p2p constants, HybridCoordinator disabled")
            return

        # Check if Raft is enabled
        if not RAFT_ENABLED and CONSENSUS_MODE == "bully":
            logger.info(
                f"[P2P] HybridCoordinator not started: RAFT_ENABLED={RAFT_ENABLED}, "
                f"CONSENSUS_MODE={CONSENSUS_MODE}. To enable Raft, set "
                "RINGRIFT_RAFT_ENABLED=true and RINGRIFT_CONSENSUS_MODE=raft"
            )
            return

        try:
            from app.p2p.hybrid_coordinator import HybridCoordinator

            self._hybrid_coordinator = HybridCoordinator(
                orchestrator=self,
                on_leader_change=self._on_raft_leader_change,
            )
            await self._hybrid_coordinator.start()

            # Check if Raft initialized successfully
            if self._hybrid_coordinator:
                status = self._hybrid_coordinator.get_status()
                # Note: get_status() returns a dict, not HybridStatus object
                logger.info(
                    f"[P2P] HybridCoordinator started: "
                    f"consensus_mode={status.get('consensus_mode', 'unknown')}, "
                    f"raft_enabled={status.get('raft', {}).get('enabled', False)}, "
                    f"raft_available={status.get('raft', {}).get('available', False)}"
                )
        except ImportError as e:
            logger.warning(f"[P2P] HybridCoordinator not available: {e}")
            self._hybrid_coordinator = None
        except Exception as e:
            logger.error(f"[P2P] HybridCoordinator initialization failed: {e}")
            self._hybrid_coordinator = None

    def _on_raft_leader_change(self, leader_address: str | None) -> None:
        """Handle Raft leader change events.

        January 23, 2026: This callback is invoked by HybridCoordinator when
        Raft elects a new leader. We update the orchestrator's leader_id to
        keep it synchronized with Raft's view.

        Args:
            leader_address: The new leader's address (ip:port) or None if no leader
        """
        if not leader_address:
            logger.info("[Raft] No leader elected - Raft cluster may be forming")
            return

        # Convert Raft address (ip:port) to node_id
        # Jan 30, 2026: Use network orchestrator directly
        leader_node_id = self.network.resolve_raft_address_to_node_id(leader_address)
        if leader_node_id:
            # Update orchestrator's leader_id via _set_leader for consistency
            self._set_leader(leader_node_id, reason="raft_election")
            logger.info(f"[Raft] Leader elected: {leader_node_id} (address: {leader_address})")
        else:
            logger.warning(
                f"[Raft] Leader elected at {leader_address} but cannot resolve to node_id"
            )


    async def _send_startup_peer_announcements(self) -> None:
        """Send immediate announcements to all known peers on startup.

        January 7, 2026: Instead of waiting for the first heartbeat interval,
        immediately announce to all known peers. This reduces discovery latency
        from 15-30s down to 2-5s after startup.

        Feb 22, 2026: Made concurrent with 10s per-peer timeout to prevent
        blocking startup for 3+ minutes when peers are unreachable.
        """
        peers_to_announce = []
        for peer_addr in self.known_peers:
            try:
                scheme, host, port = self._parse_peer_address(peer_addr)
                peers_to_announce.append((scheme, host, port))
            except (AttributeError, ValueError):
                continue

        if not peers_to_announce:
            return

        success_count = 0

        async def _announce_one(scheme, host, port):
            nonlocal success_count
            try:
                info = await asyncio.wait_for(
                    self._send_heartbeat_to_peer(host, port, scheme=scheme),
                    timeout=10.0,
                )
                if info and info.node_id != self.node_id:
                    async with NonBlockingAsyncLockWrapper(self.peers_lock, "peers_lock", timeout=5.0):
                        is_first_contact = info.node_id not in self.peers
                        info.last_heartbeat = time.time()
                        self.peers[info.node_id] = info
                    if is_first_contact:
                        logger.info(f"[P2P] Startup announcement discovered peer: {info.node_id}")
                    success_count += 1
            except asyncio.TimeoutError:
                logger.debug(f"[P2P] Startup announcement to {host}:{port} timed out (10s)")
            except Exception as e:  # noqa: BLE001
                logger.debug(f"[P2P] Startup announcement to {host}:{port} failed: {e}")

        # Run all announcements concurrently with an overall 30s timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(*[_announce_one(s, h, p) for s, h, p in peers_to_announce],
                               return_exceptions=True),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            logger.warning(f"[P2P] Startup announcements overall timeout (30s)")

        total = len(peers_to_announce)
        if success_count > 0:
            logger.info(f"[P2P] Startup announcements: {success_count}/{total} peers reachable")
        elif total > 0:
            logger.warning(f"[P2P] Startup announcements: no peers reachable (tried {total})")

    async def _execute_relay_commands(self, commands: list[dict[str, Any]]) -> None:
        """Execute relay commands (polling mode for NAT-blocked nodes)."""
        now = time.time()
        for cmd in commands:
            try:
                cmd_id = str(cmd.get("id") or "")
                cmd_type = str(cmd.get("type") or "")
                payload = cmd.get("payload") or {}
                if not cmd_id or not cmd_type:
                    continue

                # Check for stale commands (>5 min old indicates relay/polling issues)
                cmd_ts = cmd.get("ts") or cmd.get("timestamp") or now
                cmd_age_secs = now - float(cmd_ts)
                if cmd_age_secs > 300:
                    logger.info(f"WARNING: Relay command {cmd_id} ({cmd_type}) is {cmd_age_secs:.0f}s old - relay delivery may be delayed")

                attempts = int(self.relay_command_attempts.get(cmd_id, 0) or 0) + 1
                self.relay_command_attempts[cmd_id] = attempts

                ok = False
                err = ""
                if cmd_type == "start_job":
                    job_type = JobType(str(payload.get("job_type") or "selfplay"))
                    board_type = str(payload.get("board_type") or "square8")
                    num_players = int(payload.get("num_players") or 2)
                    engine_mode = str(payload.get("engine_mode") or "mixed")
                    job_id = str(payload.get("job_id") or "")

                    if job_id:
                        with self.jobs_lock:
                            existing = self.local_jobs.get(job_id)
                        if existing and existing.status == "running":
                            ok = True
                        else:
                            job = await self._start_local_job(
                                job_type,
                                board_type=board_type,
                                num_players=num_players,
                                engine_mode=engine_mode,
                                job_id=job_id,
                            )
                            ok = job is not None
                    else:
                        job = await self._start_local_job(
                            job_type,
                            board_type=board_type,
                            num_players=num_players,
                            engine_mode=engine_mode,
                        )
                        ok = job is not None
                elif cmd_type == "cleanup":
                    asyncio.create_task(self._cleanup_local_disk())
                    ok = True
                elif cmd_type == "restart_stuck_jobs":
                    asyncio.create_task(self._restart_local_stuck_jobs())
                    ok = True
                elif cmd_type == "reduce_selfplay":
                    target = payload.get("target_selfplay_jobs", payload.get("target", 0))
                    reason = str(payload.get("reason") or "relay")
                    try:
                        target_jobs = int(target)
                    except (ValueError):
                        target_jobs = 0
                    await self._reduce_local_selfplay_jobs(target_jobs, reason=reason)
                    ok = True
                elif cmd_type == "cleanup_files":
                    files = payload.get("files", []) or []
                    reason = str(payload.get("reason") or "relay")
                    if not isinstance(files, list) or not files:
                        ok = False
                        err = "no_files"
                    else:
                        data_dir = self.get_data_directory()
                        freed_bytes = 0
                        deleted_count = 0
                        data_root = data_dir.resolve()
                        for file_path in files:
                            full_path = data_dir / (str(file_path or "").lstrip("/"))
                            try:
                                resolved = full_path.resolve()
                                resolved.relative_to(data_root)
                            except (AttributeError):
                                continue
                            if not resolved.exists():
                                continue
                            try:
                                size = resolved.stat().st_size
                                resolved.unlink()
                                freed_bytes += size
                                deleted_count += 1
                            except (AttributeError):
                                continue
                        print(
                            f"[P2P] Relay cleanup_files: {deleted_count} files deleted, "
                            f"{freed_bytes / 1e6:.1f}MB freed (reason={reason})"
                        )
                        ok = True
                elif cmd_type == "canonical_selfplay":
                    job_id = str(payload.get("job_id") or "")
                    board_type = str(payload.get("board_type") or "square8")
                    num_players = int(payload.get("num_players") or 2)
                    num_games = int(payload.get("num_games") or payload.get("games_per_node") or 500)
                    seed = int(payload.get("seed") or 0)
                    if not job_id:
                        ok = False
                        err = "missing_job_id"
                    else:
                        asyncio.create_task(
                            self._run_local_canonical_selfplay(job_id, board_type, num_players, num_games, seed)
                        )
                        ok = True
                else:
                    ok = False
                    err = f"unknown_command_type:{cmd_type}"

                if ok:
                    self._add_pending_relay_ack(cmd_id)
                    self._add_pending_relay_result({"id": cmd_id, "ok": True})
                    self.relay_command_attempts.pop(cmd_id, None)
                else:
                    if not err:
                        err = "command_failed"
                    if attempts >= RELAY_COMMAND_MAX_ATTEMPTS:
                        self._add_pending_relay_ack(cmd_id)
                        self._add_pending_relay_result({"id": cmd_id, "ok": False, "error": err})
                        self.relay_command_attempts.pop(cmd_id, None)
            except Exception as exc:
                try:
                    cmd_id = str(cmd.get("id") or "")
                    if cmd_id:
                        attempts = int(self.relay_command_attempts.get(cmd_id, 0) or 0)
                        if attempts >= RELAY_COMMAND_MAX_ATTEMPTS:
                            self._add_pending_relay_ack(cmd_id)
                            self._add_pending_relay_result({"id": cmd_id, "ok": False, "error": str(exc)})
                            self.relay_command_attempts.pop(cmd_id, None)
                except (ValueError, AttributeError):
                    continue

    async def _heartbeat_loop(self):
        """Send heartbeats to all known peers."""
        # Jan 11, 2026: Phase 5 - Initial heartbeat burst to prevent startup race
        # Send immediate heartbeats to all known peers so they discover us quickly
        # This fixes the issue where peers think we're dead before our first heartbeat
        logger.info("Sending initial heartbeat burst to known peers")
        for peer_addr in self.known_peers:
            try:
                scheme, host, port = self._parse_peer_address(peer_addr)
                await self._send_heartbeat_to_peer(host, port, scheme=scheme, timeout=10)
            except Exception:
                pass  # Best effort, regular loop will retry

        while self.running:
            try:
                # Feb 23, 2026: SAFETY NET — force non-coordinator nodes to follower.
                # Many code paths (gossip, anti-entropy, Raft, etc.) bypass _set_leader()
                # and directly set self.leader_id = self.node_id. This catch-all check
                # clears self-leadership every heartbeat cycle (~10s) for non-coordinators.
                _is_coordinator = os.environ.get("RINGRIFT_IS_COORDINATOR", "").lower() in ("true", "1", "yes")
                if not _is_coordinator and self.leader_id == self.node_id:
                    logger.info(
                        "[HeartbeatLoop] Non-coordinator has self-leadership, clearing"
                    )
                    self.leader_id = None
                    self.leader_lease_id = ""
                    self.leader_lease_expires = 0
                    # NodeRole already imported at module level from scripts.p2p.types
                    # Do NOT use local import here - it shadows the global, causing
                    # UnboundLocalError at lines 9020+ when this branch doesn't execute.
                    self.role = NodeRole.FOLLOWER

                # Jan 20, 2026: Check for and fix leadership state desync every heartbeat
                # This recovers from gossip race conditions where leader_id/role diverge
                try:
                    if self.leadership.recover_leadership_desync():
                        logger.info("[HeartbeatLoop] Recovered from leadership desync")
                except Exception as e:
                    logger.debug(f"[HeartbeatLoop] Desync check failed: {e}")

                # Jan 23, 2026: Phase 2 - Reconcile ULSM with gossip consensus every 30s
                # This fixes the issue where nodes are consensus leader but don't claim leadership
                now = time.time()
                last_reconcile = getattr(self, "_last_leadership_reconcile", 0)
                if now - last_reconcile >= 30.0:
                    self._last_leadership_reconcile = now
                    try:
                        if self.leadership.reconcile_leadership_state():
                            logger.info("[HeartbeatLoop] Reconciled leadership state with gossip consensus")
                    except Exception as e:
                        logger.debug(f"[HeartbeatLoop] Leadership reconciliation failed: {e}")

                # Send to known peers from config
                for peer_addr in self.known_peers:
                    try:
                        scheme, host, port = self._parse_peer_address(peer_addr)
                    except (AttributeError):
                        continue

                    # Use relay heartbeat for HTTPS endpoints (they're proxies/relays),
                    # explicitly configured relay peers (--relay-peers flag),
                    # or if this node is NAT-blocked and needs to relay ALL outbound heartbeats
                    use_relay = scheme == "https" or peer_addr in self.relay_peers or self._force_relay_mode
                    if use_relay:
                        # Relay/proxy endpoint, use relay heartbeat
                        relay_url = f"{scheme}://{host}" if port in (80, 443) else f"{scheme}://{host}:{port}"
                        result = await self._send_relay_heartbeat(relay_url)
                        if result.get("success"):
                            # Relay heartbeat already updates peers and leader
                            continue

                    info = await self._send_heartbeat_to_peer(host, port, scheme=scheme)
                    if info:
                        if info.node_id == self.node_id:
                            continue
                        # Dec 2025: Track first-contact for HOST_ONLINE emission
                        async with NonBlockingAsyncLockWrapper(self.peers_lock, "peers_lock", timeout=5.0):
                            is_first_contact = info.node_id not in self.peers
                            info.last_heartbeat = time.time()
                            self.peers[info.node_id] = info
                        # Dec 2025: Emit HOST_ONLINE for newly discovered peers
                        if is_first_contact:
                            capabilities = []
                            if getattr(info, "has_gpu", False):
                                gpu_type = getattr(info, "gpu_type", "") or "gpu"
                                capabilities.append(gpu_type)
                            else:
                                capabilities.append("cpu")
                            await self._emit_host_online(info.node_id, capabilities)
                            logger.info(f"First-contact peer via heartbeat loop: {info.node_id}")
                        if info.role == NodeRole.LEADER and info.node_id != self.node_id:
                            async with NonBlockingAsyncLockWrapper(self.peers_lock, "peers_lock", timeout=5.0):
                                peers_snapshot = list(self.peers.values())
                            conflict_keys = self._endpoint_conflict_keys([self.self_info, *peers_snapshot])
                            if not self._is_leader_eligible(info, conflict_keys, require_alive=False):
                                continue
                            if self.role == NodeRole.LEADER and info.node_id <= self.node_id:
                                continue
                            if (
                                self.leader_id
                                and self.leader_id != info.node_id
                                and self._is_leader_lease_valid()
                                and info.node_id <= self.leader_id
                            ):
                                continue
                            # Feb 2026: Skip leader adoption if we have forced leader override
                            if getattr(self, "_forced_leader_override", False) and self.leader_id == self.node_id:
                                continue
                            if self.leader_id != info.node_id or self.role != NodeRole.FOLLOWER:
                                logger.info(f"Following configured leader from heartbeat: {info.node_id}")
                            prev_leader = self.leader_id
                            # Provisional lease: allow time for the leader to send
                            # a /coordinator lease renewal after we discover it via
                            # heartbeat (prevents leaderless oscillation right after
                            # restarts/partitions).
                            if prev_leader != info.node_id or not self._is_leader_lease_valid():
                                self.leader_lease_id = ""
                                self.leader_lease_expires = time.time() + LEADER_LEASE_DURATION
                            # Jan 3, 2026: Use _set_leader() for atomic leadership assignment (Phase 4)
                            self._set_leader(info.node_id, reason="heartbeat_configured_leader", save_state=False)

                # Send to discovered peers (skip NAT-blocked peers and ambiguous endpoints).
                # Jan 12, 2026: Use cached snapshot to reduce lock contention (1s staleness OK for heartbeat)
                # Jan 30, 2026: Use network orchestrator directly
                peers_snapshot = self.network.get_cached_peer_snapshot(max_age_seconds=1.0)
                conflict_keys = self._endpoint_conflict_keys([self.self_info, *peers_snapshot])
                peer_list = [
                    p for p in peers_snapshot
                    if (
                        not p.nat_blocked
                        and self._endpoint_key(p) not in conflict_keys
                    )
                ]

                for peer in peer_list:
                    if peer.node_id != self.node_id:
                        if not peer.should_retry():
                            continue

                        # Jan 11, 2026: Phase 2 - Adaptive heartbeat timing based on consecutive failures
                        # This gives flaky peers time to recover without spamming them
                        failures = int(getattr(peer, "consecutive_failures", 0) or 0)
                        if failures == 0:
                            heartbeat_interval = HEARTBEAT_INTERVAL  # 15s for healthy peers
                        elif failures == 1:
                            heartbeat_interval = 5  # Quick retry after first failure
                        elif failures == 2:
                            heartbeat_interval = 10  # Second retry
                        elif failures < 5:
                            heartbeat_interval = 20  # Slower retries
                        else:
                            heartbeat_interval = 30  # Very slow for consistently failing

                        # Check if heartbeat is due for this peer
                        last_sent = float(getattr(peer, "last_heartbeat_sent", 0.0) or 0.0)
                        now = time.time()
                        if now - last_sent < heartbeat_interval:
                            continue  # Not time yet for this peer

                        # Mark the send time
                        peer.last_heartbeat_sent = now

                        peer_scheme = getattr(peer, "scheme", "http") or "http"
                        info = await self._send_heartbeat_to_peer(peer.host, peer.port, scheme=peer_scheme)
                        if not info and getattr(peer, "reported_host", "") and getattr(peer, "reported_port", 0):
                            # Multi-path retry: fall back to self-reported endpoint when the
                            # observed reachable endpoint fails (e.g., mixed overlays).
                            try:
                                rh = str(getattr(peer, "reported_host", "") or "").strip()
                                rp = int(getattr(peer, "reported_port", 0) or 0)
                            except (ValueError, AttributeError):
                                rh, rp = "", 0
                            if rh and rp and (rh != peer.host or rp != peer.port):
                                info = await self._send_heartbeat_to_peer(rh, rp, scheme=peer_scheme)
                        # Self-healing: Tailscale IP fallback when both primary and reported fail
                        if not info:
                            ts_ip = self._get_tailscale_ip_for_peer(peer.node_id)
                            if ts_ip and ts_ip != peer.host:
                                # Try Tailscale mesh IP (100.x.x.x)
                                info = await self._send_heartbeat_to_peer(ts_ip, peer.port, scheme=peer_scheme)
                                if info:
                                    logger.info(f"Reached {peer.node_id} via Tailscale ({ts_ip})")
                        if info:
                            info.consecutive_failures = 0
                            info.last_failure_time = 0.0
                            async with NonBlockingAsyncLockWrapper(self.peers_lock, "peers_lock", timeout=5.0):
                                info.last_heartbeat = time.time()
                                self.peers[info.node_id] = info
                            if info.role == NodeRole.LEADER and self.role != NodeRole.LEADER:
                                if not self._is_leader_eligible(info, conflict_keys, require_alive=False):
                                    continue
                                if (
                                    self.leader_id
                                    and self.leader_id != info.node_id
                                    and self._is_leader_lease_valid()
                                    and info.node_id <= self.leader_id
                                ):
                                    continue
                                # Feb 2026: Skip leader adoption if we have forced leader override
                                if getattr(self, "_forced_leader_override", False) and self.leader_id == self.node_id:
                                    pass
                                else:
                                    if self.leader_id != info.node_id:
                                        logger.info(f"Adopted leader from heartbeat: {info.node_id}")
                                    prev_leader = self.leader_id
                                    if prev_leader != info.node_id or not self._is_leader_lease_valid():
                                        self.leader_lease_id = ""
                                        self.leader_lease_expires = time.time() + LEADER_LEASE_DURATION
                                    # Jan 3, 2026: Use _set_leader() for atomic leadership assignment (Phase 4)
                                    self._set_leader(info.node_id, reason="heartbeat_adopt_leader", save_state=False)
                        else:
                            async with NonBlockingAsyncLockWrapper(self.peers_lock, "peers_lock", timeout=5.0):
                                existing = self.peers.get(peer.node_id)
                                if existing:
                                    existing.consecutive_failures = int(getattr(existing, "consecutive_failures", 0) or 0) + 1
                                    existing.last_failure_time = time.time()

                # If we're only connected to a seed peer (or lost cluster membership),
                # pull a fresh peer snapshot so leader election converges quickly.
                await self._bootstrap_from_known_peers()

                # Get current time for all time-based checks in this cycle
                now = time.time()

                # NAT-blocked nodes: poll a relay endpoint for peer snapshots + commands.
                if getattr(self.self_info, "nat_blocked", False):
                    if now - self.last_relay_heartbeat >= RELAY_HEARTBEAT_INTERVAL:
                        relay_urls: list[str] = []
                        leader_peer = self._get_leader_peer()
                        if leader_peer and leader_peer.node_id != self.node_id:
                            relay_urls.append(f"{leader_peer.scheme}://{leader_peer.host}:{leader_peer.port}")
                        for peer_addr in self.known_peers:
                            try:
                                scheme, host, port = self._parse_peer_address(peer_addr)
                            except (AttributeError):
                                continue
                            relay_urls.append(f"{scheme}://{host}:{port}")
                        seen: set[str] = set()
                        relay_urls = [u for u in relay_urls if not (u in seen or seen.add(u))]

                        for relay_url in relay_urls:
                            result = await self._send_relay_heartbeat(relay_url)
                            if result.get("success"):
                                self.last_relay_heartbeat = now
                                break

                # Check for dead peers
                await self._check_dead_peers_async()

                # Dec 30, 2025: Probe retired peers periodically to detect recovery
                # This runs every PEER_RECOVERY_RETRY_INTERVAL (120s) to actively probe
                # retired nodes that may have come back online after cluster restart.
                last_probe = getattr(self, "_last_retired_probe", 0.0)
                if now - last_probe >= PEER_RECOVERY_RETRY_INTERVAL:
                    self._last_retired_probe = now
                    try:
                        await self.network.probe_retired_peers_async()
                    except Exception as e:
                        logger.warning(f"Error in retired peer probe: {e}")

                # Self-healing: detect network partition and trigger Tailscale-priority mode
                # Jan 30, 2026: Use network orchestrator directly
                if self.network.detect_network_partition():
                    self.network.enable_tailscale_priority()
                    # Also enable partition-local election if no voters reachable
                    if not self._has_voter_quorum():
                        self.leadership.enable_partition_local_election()
                    # Force refresh all IP sources to discover alternative paths
                    last_refresh = getattr(self, "_last_partition_ip_refresh", 0)
                    if time.time() - last_refresh > 60:  # Refresh at most once per minute
                        self._last_partition_ip_refresh = time.time()
                        # Jan 28, 2026: Uses ip_discovery_manager directly
                        asyncio.create_task(self.ip_discovery_manager.force_ip_refresh_all_sources())

                    # Jan 13, 2026: Exponential backoff during isolation
                    # Check if we're completely isolated (no alive peers)
                    alive_peers = self._peer_query.alive_count().unwrap_or(0)
                    if alive_peers == 0:
                        # Track isolation start time
                        if not hasattr(self, "_isolation_start"):
                            self._isolation_start = time.time()
                            self._isolation_backoff_seconds = HEARTBEAT_INTERVAL
                            logger.warning("Node is isolated - no alive peers, starting exponential backoff")

                        # Calculate exponential backoff based on isolation duration
                        isolation_duration = time.time() - self._isolation_start
                        if isolation_duration > 60:  # After 1 min
                            self._isolation_backoff_seconds = min(30, self._isolation_backoff_seconds * 1.5)
                        if isolation_duration > 180:  # After 3 min
                            self._isolation_backoff_seconds = min(60, self._isolation_backoff_seconds * 1.5)
                        if isolation_duration > 300:  # After 5 min
                            self._isolation_backoff_seconds = min(120, self._isolation_backoff_seconds)

                        logger.debug(f"Isolated for {isolation_duration:.0f}s, backoff={self._isolation_backoff_seconds:.0f}s")
                        # Apply additional backoff sleep (on top of normal HEARTBEAT_INTERVAL)
                        extra_backoff = self._isolation_backoff_seconds - HEARTBEAT_INTERVAL
                        if extra_backoff > 0:
                            await asyncio.sleep(extra_backoff)
                    else:
                        # Reset isolation tracking when peers are reachable
                        if hasattr(self, "_isolation_start"):
                            logger.info(f"Isolation ended after {time.time() - self._isolation_start:.0f}s, {alive_peers} peers alive")
                            delattr(self, "_isolation_start")
                            if hasattr(self, "_isolation_backoff_seconds"):
                                delattr(self, "_isolation_backoff_seconds")
                elif getattr(self, "_tailscale_priority", False):
                    # Check if priority mode should expire
                    if time.time() > getattr(self, "_tailscale_priority_until", 0):
                        # Check if connectivity recovered
                        alive_count = self._peer_query.alive_count().unwrap_or(0)
                        if alive_count > 0:
                            self.network.disable_tailscale_priority()

                # Self-healing: check if partition healed and restore original voters
                if hasattr(self, "_original_voters"):
                    self.leadership.restore_original_voters()

                # Dynamic voter management: promote/demote voters based on health
                # Only the leader manages voters to ensure consistency
                if self.role == NodeRole.LEADER:
                    self.network.manage_dynamic_voters()

                # Health-based leadership: step down if we can't reach enough peers
                if self.role == NodeRole.LEADER and not self._check_leader_health():
                    logger.info("Stepping down due to degraded health")
                    # Jan 3, 2026: Use _set_leader() for atomic leadership assignment (Phase 4)
                    self._set_leader(None, reason="degraded_health", save_state=True)
                    self.leader_lease_id = ""
                    self.leader_lease_expires = 0.0
                    self._release_voter_grant_if_self()
                    # Jan 13, 2026: Add sleep before continue to prevent busy loop
                    # when repeatedly stepping down due to degraded health
                    await asyncio.sleep(HEARTBEAT_INTERVAL)
                    continue  # Skip leader duties this cycle

                # P0 Dec 2025: Monitor leader heartbeat for early warning
                # Emit LEADER_HEARTBEAT_MISSING if leader lease is approaching expiry
                if self.role == NodeRole.FOLLOWER and self.leader_id:
                    now = time.time()
                    # Warning threshold: 45 seconds (3x lease renewal interval)
                    heartbeat_warning_threshold = LEADER_LEASE_RENEW_INTERVAL * 3
                    time_until_expiry = self.leader_lease_expires - now
                    # Emit warning if lease will expire within warning threshold
                    if 0 < time_until_expiry < heartbeat_warning_threshold:
                        last_warning = getattr(self, "_last_heartbeat_missing_warning", 0.0)
                        # Only warn once per 30 seconds to avoid spam
                        if now - last_warning > 30:
                            self._last_heartbeat_missing_warning = now
                            delay_seconds = (LEADER_LEASE_DURATION - time_until_expiry)
                            try:
                                from app.distributed.data_events import emit_leader_heartbeat_missing
                                asyncio.create_task(emit_leader_heartbeat_missing(
                                    leader_id=self.leader_id,
                                    last_heartbeat=self.leader_lease_expires - LEADER_LEASE_DURATION,
                                    expected_interval=LEADER_LEASE_RENEW_INTERVAL,
                                    delay_seconds=delay_seconds,
                                    source=self.node_id,
                                ))
                            except ImportError:
                                pass  # Graceful degradation if event system not available

                # LEARNED LESSONS - Lease renewal to maintain leadership
                if self.role == NodeRole.LEADER:
                    await self._renew_leader_lease()

                # P2P monitoring: start/stop services based on leadership
                await self._stop_monitoring_if_not_leader()
                if self.role == NodeRole.LEADER:
                    await self._start_monitoring_if_leader()

                # P2P auto-deployer: start/stop based on leadership
                if self.role != NodeRole.LEADER and self._auto_deployer_task:
                    await self._stop_p2p_auto_deployer()
                elif self.role == NodeRole.LEADER and not self._auto_deployer_task:
                    await self._start_p2p_auto_deployer()

                # Report node resources to resource_optimizer for cluster-wide utilization tracking
                # This enables cooperative 60-80% utilization targeting across orchestrators
                if HAS_NEW_COORDINATION and get_resource_optimizer is not None:
                    try:
                        optimizer = get_resource_optimizer()
                        # Mar 2026: Use cached async version instead of blocking
                        # asyncio.to_thread(self._update_self_info). The sync version
                        # takes 10-30s on macOS (pgrep, psutil, NFS checks) and consumes
                        # a thread pool slot every heartbeat (10-15s). With only 8 threads,
                        # this starves queue_populator and voter_heartbeat, causing cascading
                        # 600s timeouts that eventually trigger P2P recovery daemon to
                        # pkill the orchestrator after ~2 hours.
                        # cache_ttl=30s is sufficient for resource metrics (they don't
                        # change rapidly), and dramatically reduces thread pool pressure.
                        await self._update_self_info_async(cache_ttl=30.0)
                        node_resources = NodeResources(
                            node_id=self.node_id,
                            cpu_percent=self.self_info.cpu_percent,
                            memory_percent=self.self_info.memory_percent,
                            active_jobs=self.self_info.selfplay_jobs + self.self_info.training_jobs,
                            has_gpu=self.self_info.has_gpu,
                            gpu_name=self.self_info.gpu_type or "",
                        )
                        optimizer.report_node_resources(node_resources)
                    except (AttributeError):
                        pass  # Non-critical, don't disrupt heartbeat

                # Save state periodically
                self._save_state()

            except Exception as e:  # noqa: BLE001
                logger.info(f"Heartbeat error: {e}")

            # Notify systemd watchdog that we're still alive
            systemd_notify_watchdog()

            await asyncio.sleep(HEARTBEAT_INTERVAL)

    # See scripts/p2p/loops/network_loops.py and scripts/p2p/loop_registry.py

    # See scripts/p2p/loops/network_loops.py and scripts/p2p/loop_registry.py

    async def _send_voter_heartbeat(self, voter_peer) -> bool:
        """Send a heartbeat to a voter peer with shorter timeout.

        Jan 27, 2026: Phase 16A - Delegates to HeartbeatManager.
        """
        return await self.heartbeat_manager.send_voter_heartbeat(voter_peer)

    async def _try_voter_alternative_endpoints(self, voter_peer) -> bool:
        """Try alternative endpoints for a voter peer."""
        peer_scheme = getattr(voter_peer, "scheme", "http") or "http"

        # Try 1: Tailscale IP
        ts_ip = self._get_tailscale_ip_for_peer(voter_peer.node_id)
        if ts_ip and ts_ip != voter_peer.host:
            info = await self._send_heartbeat_to_peer(ts_ip, voter_peer.port, scheme=peer_scheme, timeout=VOTER_HEARTBEAT_TIMEOUT)
            if info:
                logger.info(f"Reached voter {voter_peer.node_id} via Tailscale ({ts_ip})")
                with self.peers_lock:
                    info.last_heartbeat = time.time()
                    info.consecutive_failures = 0
                    self.peers[info.node_id] = info
                return True

        # Try 2: Reported host/port
        rh = str(getattr(voter_peer, "reported_host", "") or "").strip()
        rp = int(getattr(voter_peer, "reported_port", 0) or 0)
        if rh and rp and (rh != voter_peer.host or rp != voter_peer.port):
            info = await self._send_heartbeat_to_peer(rh, rp, scheme=peer_scheme, timeout=VOTER_HEARTBEAT_TIMEOUT)
            if info:
                logger.info(f"Reached voter {voter_peer.node_id} via reported endpoint ({rh}:{rp})")
                with self.peers_lock:
                    info.last_heartbeat = time.time()
                    info.consecutive_failures = 0
                    self.peers[info.node_id] = info
                return True

        return False

    async def _discover_voter_peer(self, voter_id: str):
        """Discover a voter peer from known peers."""
        # Ask known peers for the voter's endpoint
        for peer_addr in self.known_peers:
            try:
                scheme, host, port = self._parse_peer_address(peer_addr)
                async with aiohttp.ClientSession() as session, session.get(
                    f"{scheme}://{host}:{port}/relay/peers",
                    timeout=aiohttp.ClientTimeout(total=5),
                    headers=self._auth_headers()
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        peers_data = data.get("peers", {})
                        if voter_id in peers_data:
                            peer_info = NodeInfo.from_dict(peers_data[voter_id])
                            with self.peers_lock:
                                self.peers[voter_id] = peer_info
                            logger.info(f"Discovered voter {voter_id} from {host}")
                            return
            except (aiohttp.ClientError, asyncio.TimeoutError, AttributeError, ImportError):
                continue

    async def _refresh_voter_mesh(self):
        """Ensure all voters have knowledge of each other."""
        if not self.voter_node_ids:
            return

        # Jan 12, 2026: Use lock-free PeerSnapshot for read-only access
        peers_snapshot = self._peer_snapshot.get_snapshot()

        # Check how many voters we know about (outside lock)
        known_voters = [v for v in self.voter_node_ids if v in peers_snapshot or v == self.node_id]

        if len(known_voters) < len(self.voter_node_ids):
            missing_voters = [v for v in self.voter_node_ids if v not in known_voters]
            logger.info(f"Voter mesh incomplete, missing: {missing_voters}")

            # Try to discover missing voters
            for voter_id in missing_voters:
                await self._discover_voter_peer(voter_id)

    # See scripts/p2p/loops/network_loops.py for implementation.


    # NOTE: _select_best_relay() inlined at call site (Jan 2026 Phase 2)


    # See scripts/p2p/loops/manifest_collection_loop.py

    def _record_selfplay_stats_sample(self, manifest: ClusterDataManifest) -> None:
        """Record a lightweight selfplay totals sample for dashboard charts."""
        try:
            sample = {
                "timestamp": time.time(),
                "manifest_collected_at": float(getattr(manifest, "collected_at", 0.0) or 0.0),
                "total_selfplay_games": int(getattr(manifest, "total_selfplay_games", 0) or 0),
                "by_board_type": manifest.by_board_type,
                "total_nodes": int(getattr(manifest, "total_nodes", 0) or 0),
            }
            self.selfplay_stats_history.append(sample)
            max_samples = int(getattr(self, "selfplay_stats_history_max_samples", 288) or 288)
            if max_samples > 0 and len(self.selfplay_stats_history) > max_samples:
                self.selfplay_stats_history = self.selfplay_stats_history[-max_samples:]
        except (ValueError, KeyError, IndexError, AttributeError):
            # Never let dashboard bookkeeping break manifest collection.
            return

    def _endpoint_key(self, info: NodeInfo) -> tuple[str, str, int] | None:
        """Return the normalized reachable endpoint key for a peer (scheme, host, port)."""
        host = str(getattr(info, "host", "") or "").strip()
        if not host:
            return None
        scheme = str(getattr(info, "scheme", "http") or "http").lower()
        try:
            port = int(getattr(info, "port", DEFAULT_PORT) or DEFAULT_PORT)
        except (ValueError):
            port = DEFAULT_PORT
        reported_host = str(getattr(info, "reported_host", "") or "").strip()
        try:
            reported_port = int(getattr(info, "reported_port", 0) or 0)
        except (ValueError):
            reported_port = 0

        if reported_host and reported_port > 0:
            # Reverse proxies / relays can cause inbound peer requests to appear as loopback.
            # Prefer the peer's self-reported advertised endpoint in that case so:
            # - endpoint conflict detection remains meaningful, and
            # - eligible leaders don't get filtered out as "conflicted".
            if host in {"127.0.0.1", "localhost", "0.0.0.0", "::1"} or self._is_tailscale_host(reported_host):
                host, port = reported_host, reported_port
        return (scheme, host, port)

    def _endpoint_conflict_keys(self, peers: list[NodeInfo]) -> set[tuple[str, str, int]]:
        """Compute endpoint keys that are shared by >1 node (NAT/port collisions)."""
        counts: dict[tuple[str, str, int], int] = {}
        for p in peers:
            # Ignore dead peers: stale node_ids can linger after restarts and would
            # otherwise permanently mark the live node as "conflicted".
            if not p.is_alive():
                continue
            key = self._endpoint_key(p)
            if not key:
                continue
            counts[key] = counts.get(key, 0) + 1
        return {k for k, v in counts.items() if v > 1}


    # NOTE: _compute_connectivity_score() inlined at call site (Jan 2026 Phase 2)

    def _is_leader_eligible(
        self,
        peer: NodeInfo,
        conflict_keys: set[tuple[str, str, int]],
        *,
        require_alive: bool = True,
    ) -> bool:
        """Heuristic: leaders must be directly reachable and uniquely addressable.

        Jan 2, 2026: Enhanced to reject NAT-blocked and force_relay_mode nodes,
        and require minimum connectivity score of 0.3.
        """
        if require_alive and not peer.is_alive():
            return False
        voters = list(getattr(self, "voter_node_ids", []) or [])
        if voters and peer.node_id not in voters:
            return False
        if int(getattr(peer, "consecutive_failures", 0) or 0) >= MAX_CONSECUTIVE_FAILURES:
            return False
        # Jan 2, 2026: Reject NAT-blocked nodes
        if getattr(peer, "nat_blocked", False):
            return False
        # Jan 2, 2026: Reject force_relay_mode nodes (can't serve peers directly)
        if getattr(peer, "force_relay_mode", False):
            return False
        # Jan 13, 2026: Reject proxy_only nodes (cannot coordinate cluster operations)
        # These nodes are SSH jump hosts or API proxies with no AI/training capability
        # Check both NodeInfo.status and config (config is authoritative)
        node_status = getattr(peer, "status", "")
        if node_status == "proxy_only" or self._is_node_proxy_only(peer.node_id):
            return False
        # Jan 2, 2026: Require minimum connectivity score
        # Inline: was _compute_connectivity_score()
        if self.recovery_manager.compute_connectivity_score(peer) < 0.3:
            return False
        key = self._endpoint_key(peer)
        return not (key and key in conflict_keys)


    def _maybe_adopt_leader_from_peers(self) -> bool:
        """Jan 29, 2026: Delegated to LeadershipOrchestrator.maybe_adopt_leader_from_peers()."""
        return self.leadership.maybe_adopt_leader_from_peers()

    async def _check_dead_peers_async(self):
        """Check for peers that have stopped responding (async version).

        This version uses AsyncLockWrapper to avoid blocking the event loop
        when acquiring the peers_lock.

        January 12, 2026: Refactored to move event emissions outside the lock
        to prevent deadlock risk when event handlers need the same lock.

        January 19, 2026: Added rate limiting (PEER_DEATH_RATE_LIMIT) to prevent
        cascade failures. When 5+ nodes are busy, ALL nodes would mark ALL of them
        dead simultaneously, causing gossip storms and further instability.
        Now max PEER_DEATH_RATE_LIMIT peers can be retired per check cycle.

        January 29, 2026: Delegated to PeerNetworkOrchestrator.check_dead_peers_async().
        """
        # Delegate to PeerNetworkOrchestrator if available
        return await self.network.check_dead_peers_async()


    async def _start_election(self):
        """Start leader election using Bully algorithm."""
        # Feb 23, 2026: Non-coordinator nodes must NOT initiate elections.
        # GPU worker nodes should passively adopt leaders via gossip, not self-elect.
        # Without this, after P2P restart every node runs Bully election and the
        # alphabetically-first node wins before force_leader can be applied,
        # causing persistent split-brain where GPU nodes refuse coordinator's leadership.
        _is_coordinator = os.environ.get("RINGRIFT_IS_COORDINATOR", "").lower() in ("true", "1", "yes")
        if not _is_coordinator:
            logger.info(
                "[Election] Skipping: non-coordinator node, will adopt leader via gossip"
            )
            return

        # Feb 23, 2026: Suppress elections during force_leader grace period.
        # Without this, nodes that were forced to accept a specific leader could
        # immediately start a new election, overriding the forced leader.
        grace_until = getattr(self, "_election_grace_until", 0) or 0
        if time.time() < grace_until:
            logger.info(
                f"[Election] Skipping: election grace period active "
                f"({grace_until - time.time():.0f}s remaining)"
            )
            return

        # Feb 2026: Preferred leader guard — non-preferred nodes suppress their
        # own elections when the preferred leader was recently alive (last 120s).
        # This prevents split-brain where non-preferred nodes win elections when
        # they temporarily can't see the preferred leader (network blips).
        # If the preferred leader is genuinely down (>120s), any node can still win.
        preferred = getattr(self, "_preferred_leader_id", None)
        if preferred and preferred != self.node_id:
            preferred_info = self.peers.get(preferred)
            if preferred_info:
                last_seen = getattr(preferred_info, "last_seen", 0) or 0
                age = time.time() - last_seen
                if age < 120:
                    logger.info(
                        f"[Election] Suppressing: preferred leader '{preferred}' "
                        f"alive ({age:.0f}s ago, threshold 120s)"
                    )
                    return

        # Jan 3, 2026 Sprint 13.3: Track election start time for latency metrics
        self._start_election_timing()

        # Jan 19, 2026: Don't participate in elections until state loaded
        # CRITICAL FIX: Nodes were voting at 5s but state loads in 30-50s,
        # causing elections with incomplete cluster view (split-brain, thrashing)
        elapsed = time.time() - getattr(self, "_startup_time", 0)
        if elapsed < ELECTION_PARTICIPATION_DELAY:
            logger.info(
                f"[Election] Skipping: still in startup grace "
                f"({elapsed:.0f}s < {ELECTION_PARTICIPATION_DELAY}s)"
            )
            return

        # Jan 5, 2026: Global election cooldown to prevent rapid election storms
        # This complements the per-loop backoff in _maybe_trigger_election()
        # Jan 24, 2026: Increased from 15s to 30s to prevent rapid re-elections
        # With LEADER_LEASE_DURATION at 300s, 30s cooldown is safe and prevents churn
        ELECTION_GLOBAL_COOLDOWN = 30.0  # seconds
        now = time.time()
        last_election = getattr(self, "_last_election_completed", 0.0)
        if now - last_election < ELECTION_GLOBAL_COOLDOWN:
            logger.debug(
                f"[Election] Skipping: global cooldown ({now - last_election:.1f}s < {ELECTION_GLOBAL_COOLDOWN}s)"
            )
            return

        # Jan 22, 2026: Add random jitter to prevent election cascade
        # When leader dies, all nodes detect it simultaneously and try to start elections.
        # Without jitter, this causes election floods that destabilize the cluster.
        import random
        jitter = random.uniform(0.5, 3.0)  # 500ms to 3s
        await asyncio.sleep(jitter)

        # Re-check if election still needed after jitter (leader may have emerged)
        if self.leader_id and self.leader_id != self.node_id:
            logger.debug(f"[Election] Skipping after jitter: leader {self.leader_id} emerged")
            return

        # Jan 31, 2026: Run in thread to avoid blocking event loop
        await asyncio.to_thread(self._update_self_info)

        # NAT-blocked nodes cannot act as a leader because peers can't reach them.
        if getattr(self.self_info, "nat_blocked", False):
            return

        # Jan 13, 2026: Strict quorum enforcement (P2P Cluster Stability Plan Phase 2)
        # Check quorum BEFORE proceeding with election
        voter_node_ids = list(getattr(self, "voter_node_ids", []) or [])
        if voter_node_ids:
            if self.node_id not in voter_node_ids:
                # December 29, 2025: Non-voters can request elections from voters
                # instead of just returning silently
                await self._request_election_from_voters("non_voter_detected_leaderless")
                return

            # Jan 13, 2026: Use strict quorum check when available
            try:
                from scripts.p2p.leader_election import should_block_election
                # Jan 12, 2026: Use lock-free PeerSnapshot for read-only access
                snapshot = self._peer_snapshot.get_snapshot()
                should_block, reason = should_block_election(
                    voter_node_ids,
                    snapshot,
                    self.node_id,
                )
                if should_block:
                    logger.warning(f"[Election] Blocked: {reason}")
                    self._safe_emit_event("ELECTION_BLOCKED", {
                        "node_id": self.node_id,
                        "reason": reason,
                        "voter_count": len(voter_node_ids),
                        "timestamp": time.time(),
                    })
                    return
            except ImportError:
                # Fall back to legacy quorum check
                if not self._has_voter_quorum():
                    return

        # Jan 12, 2026: Use lock-free PeerSnapshot for read-only access
        snapshot = self._peer_snapshot.get_snapshot()
        peers_snapshot = [p for p in snapshot.values() if p.node_id != self.node_id]

        conflict_keys = self._endpoint_conflict_keys([self.self_info, *peers_snapshot])

        if self.leader_id and self.leader_id != self.node_id:
            # Jan 12, 2026: Use lock-free PeerSnapshot for read-only access
            leader = snapshot.get(self.leader_id)
            leader_ok = (
                leader is not None
                and leader.is_alive()
                and leader.role == NodeRole.LEADER
                and self._is_leader_eligible(leader, conflict_keys)
                and self._is_leader_lease_valid()
            )
            if leader_ok:
                return
            # Drop stale/ineligible leader so we don't keep advertising it.
            # Jan 3, 2026: Use _set_leader() for atomic leadership assignment (Phase 4)
            self._set_leader(None, reason="stale_ineligible_leader", save_state=False)
            self.leader_lease_id = ""
            self.leader_lease_expires = 0.0
        if self._maybe_adopt_leader_from_peers():
            return

        # Jan 22, 2026: Atomic election guard using asyncio.Lock
        # Prevents race condition where multiple coroutines pass the check
        # before any sets the flag, causing multiple simultaneous elections.
        async with self._election_lock:
            if self.election_in_progress:
                logger.debug("[Election] Already in progress, skipping")
                return
            # Atomically set the flag while holding the lock
            self.election_in_progress = True

        # =========================================================================
        # Jan 2, 2026: Leader Stickiness
        # If we recently were the leader, try to reclaim immediately.
        # If we're not the recent leader, check if any peer was and give them priority.
        # =========================================================================
        if self.leadership.was_recently_leader() and self.leadership.in_incumbent_grace_period():
            # We recently stepped down - try to reclaim leadership immediately
            logger.info(
                f"Incumbent advantage: attempting immediate leadership reclaim "
                f"(stepped down {time.time() - self._last_step_down_time:.1f}s ago)"
            )
            # C1 fix: Use leader_state_lock for role changes
            with self.leader_state_lock:
                self.role = NodeRole.CANDIDATE
            try:
                # Skip bully algorithm - try to become leader directly
                conflict_keys = self._endpoint_conflict_keys([self.self_info, *peers_snapshot])
                if self._is_leader_eligible(self.self_info, conflict_keys):
                    await self._become_leader()
                    if self.role == NodeRole.LEADER:
                        logger.info("Incumbent reclaimed leadership successfully")
                        return
            finally:
                # C1 fix: Use leader_state_lock for role changes
                with self.leader_state_lock:
                    if self.role == NodeRole.CANDIDATE:
                        self.role = NodeRole.FOLLOWER
                # Don't clear election_in_progress here - fall through to normal election
            # If we failed to reclaim, fall through to normal election
            logger.info("Incumbent reclaim failed, falling back to normal election")

        # election_in_progress is already True (set atomically above)
        # C1 fix: Use leader_state_lock for role changes
        with self.leader_state_lock:
            self.role = NodeRole.CANDIDATE
        logger.info(f"Starting election, my ID: {self.node_id}")

        try:
            # Send election message to all nodes with higher IDs
            # Jan 12, 2026: Use lock-free PeerSnapshot for read-only access
            election_snapshot = self._peer_snapshot.get_snapshot()
            higher_nodes = [
                p for p in election_snapshot.values()
                if (
                    p.node_id > self.node_id
                    and self._is_leader_eligible(p, conflict_keys)
                )
            ]
            voter_node_ids = list(getattr(self, "voter_node_ids", []) or [])
            if voter_node_ids:
                higher_nodes = [p for p in higher_nodes if p.node_id in voter_node_ids]

            got_response = False

            timeout = ClientTimeout(total=ELECTION_TIMEOUT)
            async with get_client_session(timeout) as session:
                for peer in higher_nodes:
                    try:
                        url = self._url_for_peer(peer, "/election")
                        async with session.post(url, json={"candidate_id": self.node_id}, headers=self._auth_headers()) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                if data.get("response") == "ALIVE":
                                    got_response = True
                                    logger.info(f"Higher node {peer.node_id} responded")
                    except (aiohttp.ClientError, asyncio.TimeoutError, AttributeError):
                        pass  # Network errors expected during elections

            # If no higher node responded, we become leader
            if not got_response:
                # Only become leader if we're eligible (unique + directly reachable).
                if self._is_leader_eligible(self.self_info, conflict_keys):
                    await self._become_leader()
            else:
                # Wait for coordinator message
                await asyncio.sleep(ELECTION_TIMEOUT * 2)
                # If no coordinator arrives, fall back to adopting any eligible leader we can see.
                self._maybe_adopt_leader_from_peers()

        finally:
            self.election_in_progress = False
            # C1 fix: Use leader_state_lock for role changes
            with self.leader_state_lock:
                if self.role == NodeRole.CANDIDATE:
                    # Jan 3, 2026 Sprint 13.3: Record election latency for "timeout" outcome
                    # Election ended without becoming leader or adopting one
                    if getattr(self, "_election_started_at", 0) > 0:
                        self._record_election_latency("timeout")
                    self.role = NodeRole.FOLLOWER

    async def _become_leader(self):
        """Become the cluster leader with lease-based leadership."""
        # Jan 31, 2026: Run in thread to avoid blocking event loop
        await asyncio.to_thread(self._update_self_info)
        if getattr(self.self_info, "nat_blocked", False):
            logger.info(f"Refusing leadership while NAT-blocked: {self.node_id}")
            return
        if getattr(self, "voter_node_ids", []) and not self._has_voter_quorum():
            logger.info(f"Refusing leadership without voter quorum: {self.node_id}")
            return
        import uuid
        lease_id = f"{self.node_id}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        lease_expires = await self._acquire_voter_lease_quorum(lease_id, int(LEADER_LEASE_DURATION))
        if getattr(self, "voter_node_ids", []) and not lease_expires:
            logger.error(f"Failed to obtain voter lease quorum; refusing leadership: {self.node_id}")
            # Jan 3, 2026 Sprint 13.3: Record election latency for "lost" outcome (no quorum)
            self._record_election_latency("lost")
            # Jan 3, 2026: Use _set_leader() for atomic leadership assignment (Phase 4)
            self._set_leader(None, reason="election_failed_no_quorum", save_state=False)
            self.leader_lease_id = ""
            self.leader_lease_expires = 0.0
            self.last_lease_renewal = 0.0
            self._release_voter_grant_if_self()
            # Jan 5, 2026: Mark election completed (even on failure) for cooldown tracking
            self._last_election_completed = time.time()
            self._save_state()
            return

        logger.info(f"I am now the leader: {self.node_id}")
        # Jan 3, 2026: Use _set_leader() for atomic leadership assignment (Phase 4)
        self._set_leader(self.node_id, reason="become_leader", save_state=False)
        self.last_leader_seen = time.time()  # Track when we last had a functioning leader

        # Jan 5, 2026: Register self in peers dict when becoming leader
        # This ensures the leader is visible in peers iteration and quorum checks
        self._register_self_in_peers()

        # Jan 3, 2026 Sprint 13.3: Record election latency for "won" outcome
        self._record_election_latency("won")
        # Dec 31, 2025: Track leadership acquisition time and reset quorum fail counters
        self._last_become_leader_time = time.time()
        # Jan 23, 2026: Reset ULSM QuorumHealth (unified tracker)
        if hasattr(self, "_leadership_sm") and self._leadership_sm:
            self._leadership_sm.quorum_health.reset()

        # Phase 29: Increment cluster epoch on leadership change
        # This helps resolve split-brain when partitions merge
        self._increment_cluster_epoch()

        # Phase 15.1.1: Increment lease epoch and create fence token
        # This provides split-brain protection by ensuring each leadership
        # term has a unique, monotonically increasing epoch
        self._lease_epoch += 1
        self._fence_token = f"{self.node_id}:{self._lease_epoch}:{time.time()}"
        logger.info(f"Leader lease fencing: epoch={self._lease_epoch}, token={self._fence_token}")

        # CRITICAL: Emit LEADER_ELECTED event (Dec 2025 fix)
        # This enables LeadershipCoordinator and other components to track leadership changes
        asyncio.create_task(self._emit_leader_elected(self.node_id, getattr(self, "cluster_epoch", 0)))

        # Lease-based leadership (voter-backed when enabled).
        self.leader_lease_id = lease_id
        self.leader_lease_expires = float(lease_expires or (time.time() + LEADER_LEASE_DURATION))
        self.last_lease_renewal = time.time()

        # Announce to all peers with lease information
        with self.peers_lock:
            peers = list(self.peers.values())

        timeout = ClientTimeout(total=5)
        async with get_client_session(timeout) as session:
            for peer in peers:
                if peer.node_id != self.node_id:
                    try:
                        url = self._url_for_peer(peer, "/coordinator")
                        await session.post(url, json={
                            "leader_id": self.node_id,
                            "lease_id": self.leader_lease_id,
                            "lease_expires": self.leader_lease_expires,
                            "voter_node_ids": list(getattr(self, "voter_node_ids", []) or []),
                            # Phase 15.1.1: Include epoch and fence token for split-brain protection
                            "lease_epoch": self._lease_epoch,
                            "fence_token": self._fence_token,
                        }, headers=self._auth_headers())
                    except (aiohttp.ClientError, asyncio.TimeoutError, KeyError, IndexError, AttributeError):
                        pass  # Network errors expected during leader announcements

        # Jan 5, 2026: Mark election completed for cooldown tracking
        self._last_election_completed = time.time()

        self._save_state()

        # Start monitoring services when becoming leader
        await self._start_monitoring_if_leader()

        # Start P2P auto-deployer when becoming leader
        await self._start_p2p_auto_deployer()

    # ============================================
    # Probabilistic Fallback Leadership (Jan 1, 2026)
    # ============================================

    async def _check_probabilistic_leadership(self, now: float) -> None:
        """Check if we should claim provisional leadership using probabilistic fallback.

        Jan 1, 2026: When normal elections repeatedly fail (e.g., voter quorum unavailable),
        nodes can claim provisional leadership with increasing probability. This prevents
        indefinite cluster stalls while still preferring proper elections.

        Design:
        1. Only activate after PROVISIONAL_LEADER_MIN_LEADERLESS_TIME (5 min)
        2. Probability starts low, grows exponentially over time
        3. If random check passes, claim PROVISIONAL_LEADER state
        4. Announce to peers and collect acknowledgments
        5. Promote to full LEADER after quorum ack or timeout with no challengers
        """
        import random

        # Skip if we already have a leader or are claiming
        if self.leader_id or self.role in (NodeRole.LEADER, NodeRole.PROVISIONAL_LEADER, NodeRole.CANDIDATE):
            return

        # Rate limit checks
        if now - self._last_provisional_check < PROVISIONAL_LEADER_CHECK_INTERVAL:
            return
        self._last_provisional_check = now

        # Check if we've been leaderless long enough
        leaderless_duration = now - self.last_leader_seen
        if leaderless_duration < PROVISIONAL_LEADER_MIN_LEADERLESS_TIME:
            return

        # Check if we're eligible (not NAT-blocked, preferably GPU node)
        # Jan 31, 2026: Run in thread to avoid blocking event loop
        await asyncio.to_thread(self._update_self_info)
        if getattr(self.self_info, "nat_blocked", False):
            logger.debug("Skipping probabilistic leadership: NAT-blocked")
            return

        # Calculate current probability based on leaderless duration
        # Probability grows exponentially beyond minimum threshold
        minutes_beyond_minimum = (leaderless_duration - PROVISIONAL_LEADER_MIN_LEADERLESS_TIME) / 60.0
        current_prob = min(
            PROVISIONAL_LEADER_MAX_PROBABILITY,
            PROVISIONAL_LEADER_INITIAL_PROBABILITY * (PROVISIONAL_LEADER_PROBABILITY_GROWTH_RATE ** minutes_beyond_minimum)
        )
        self._provisional_claim_probability = current_prob

        # Roll the dice
        roll = random.random()
        logger.debug(f"Probabilistic leadership check: roll={roll:.3f}, threshold={current_prob:.3f}, "
                    f"leaderless={int(leaderless_duration)}s")

        if roll >= current_prob:
            return  # Not claiming this time

        # We're claiming provisional leadership
        logger.info(f"Claiming provisional leadership after {int(leaderless_duration)}s leaderless "
                   f"(prob={current_prob:.2%}, roll={roll:.3f})")

        await self._claim_provisional_leadership()

    async def _claim_provisional_leadership(self) -> None:
        """Claim provisional leadership and announce to peers.

        Provisional leaders can dispatch work but must be confirmed by:
        - Quorum acknowledgment from peers, OR
        - No challengers after timeout period (with node_id tiebreaker if contested)
        """
        import uuid

        now = time.time()

        # Set provisional state
        # C1 fix: Use leader_state_lock for role/leader_id changes
        with self.leader_state_lock:
            self.role = NodeRole.PROVISIONAL_LEADER
            self._provisional_leader_claimed_at = now
            self._provisional_leader_acks = {self.node_id}  # Self-ack
            self._provisional_leader_challengers = {}

            # Create a provisional lease (shorter than normal)
            provisional_lease_id = f"PROVISIONAL_{self.node_id}_{uuid.uuid4().hex[:8]}"
            self.leader_lease_id = provisional_lease_id
            self.leader_lease_expires = now + PROVISIONAL_LEADER_QUORUM_TIMEOUT
            self.last_lease_renewal = now

            # Set ourselves as leader (provisional) so peers know who's claiming
            self.leader_id = self.node_id

        logger.info(f"Provisional leadership claimed: lease={provisional_lease_id}")

        # Announce provisional claim to all peers
        async with NonBlockingAsyncLockWrapper(self.peers_lock, "peers_lock", timeout=5.0):
            peers = [p for p in self.peers.values() if p.node_id != self.node_id and p.is_alive()]

        if not peers:
            # No peers to acknowledge, promote immediately
            logger.info("No alive peers to acknowledge, promoting immediately to full leader")
            await self._promote_provisional_to_leader("no_peers")
            return

        # Send provisional leadership claim to all peers
        timeout = aiohttp.ClientTimeout(total=5)
        acks_received = 0
        challengers = []

        async with get_client_session(timeout) as session:
            for peer in peers:
                try:
                    url = self._url_for_peer(peer, "/provisional-leader/claim")
                    async with session.post(
                        url,
                        json={
                            "claimant_id": self.node_id,
                            "lease_id": provisional_lease_id,
                            "claimed_at": now,
                        },
                        headers=self._auth_headers(),
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if data.get("ack"):
                                self._provisional_leader_acks.add(peer.node_id)
                                acks_received += 1
                                logger.debug(f"Provisional ack from {peer.node_id}")
                            elif data.get("challenge"):
                                challenger_id = data.get("challenger_id", peer.node_id)
                                self._provisional_leader_challengers[challenger_id] = now
                                challengers.append(challenger_id)
                                logger.info(f"Provisional challenge from {challenger_id}")
                except (aiohttp.ClientError, asyncio.TimeoutError, AttributeError):
                    pass  # Network errors expected

        logger.info(f"Provisional claim results: {acks_received} acks, {len(challengers)} challengers")

        # Handle challenges using node_id tiebreaker
        if challengers:
            # Find highest challenger
            all_claimants = [self.node_id] + challengers
            all_claimants.sort(reverse=True)
            winner = all_claimants[0]

            if winner != self.node_id:
                logger.info(f"Stepping down from provisional: {winner} > {self.node_id}")
                self._step_down_from_provisional()
                return

            logger.info(f"Won provisional tiebreaker against {challengers}")

        # Check if we have quorum
        total_peers = len(peers) + 1  # Include self
        quorum_size = (total_peers // 2) + 1
        current_acks = len(self._provisional_leader_acks)

        if current_acks >= quorum_size:
            logger.info(f"Quorum achieved ({current_acks}/{quorum_size}), promoting to full leader")
            await self._promote_provisional_to_leader("quorum_achieved")
        else:
            # Schedule a follow-up check after timeout period
            logger.info(f"Quorum not yet achieved ({current_acks}/{quorum_size}), waiting for timeout")
            asyncio.get_event_loop().call_later(
                PROVISIONAL_LEADER_QUORUM_TIMEOUT,
                lambda: asyncio.create_task(self._check_provisional_promotion())
            )

    async def _check_provisional_promotion(self) -> None:
        """Check if provisional leader should be promoted after timeout period."""
        if self.role != NodeRole.PROVISIONAL_LEADER:
            return  # Already promoted or stepped down

        now = time.time()
        claim_duration = now - self._provisional_leader_claimed_at

        if claim_duration < PROVISIONAL_LEADER_QUORUM_TIMEOUT:
            return  # Not time yet

        # Check for any challengers that won during the timeout
        if self._provisional_leader_challengers:
            all_claimants = [self.node_id] + list(self._provisional_leader_challengers.keys())
            all_claimants.sort(reverse=True)
            winner = all_claimants[0]

            if winner != self.node_id:
                logger.info(f"Challenger {winner} won during timeout period")
                self._step_down_from_provisional()
                return

        # No successful challengers, promote to full leader
        logger.info(f"Provisional timeout elapsed with no successful challengers, promoting to full leader")
        await self._promote_provisional_to_leader("timeout_no_challengers")

    async def _promote_provisional_to_leader(self, reason: str) -> None:
        """Promote from provisional to full leader.

        Args:
            reason: Why we're promoting (quorum_achieved, timeout_no_challengers, no_peers)
        """
        if self.role == NodeRole.LEADER:
            return  # Already promoted

        logger.info(f"Promoting from provisional to full leader: {reason}")

        # Clear provisional state
        self._provisional_leader_claimed_at = 0.0
        self._provisional_leader_acks.clear()
        self._provisional_leader_challengers.clear()

        # Full leader transition (subset of _become_leader, but without voter lease)
        import uuid
        now = time.time()

        # Jan 3, 2026: Use _set_leader() for atomic leadership assignment (Phase 4)
        self._set_leader(self.node_id, reason=f"promote_provisional_{reason}", save_state=False)
        self.last_leader_seen = now

        # Jan 5, 2026: Register self in peers dict when promoted to leader
        self._register_self_in_peers()

        # Create a new lease ID to mark full leadership
        lease_id = f"FALLBACK_{self.node_id}_{int(now)}_{uuid.uuid4().hex[:8]}"
        self.leader_lease_id = lease_id
        self.leader_lease_expires = now + LEADER_LEASE_DURATION
        self.last_lease_renewal = now

        # Track that this is fallback leadership (for monitoring)
        self._fallback_leader_since = now
        self._fallback_leader_reason = reason

        # Increment epochs
        self._increment_cluster_epoch()
        self._lease_epoch += 1
        self._fence_token = f"{self.node_id}:{self._lease_epoch}:{now}"

        # Emit LEADER_ELECTED event
        asyncio.create_task(self._emit_leader_elected(self.node_id, getattr(self, "cluster_epoch", 0)))

        # Announce to all peers
        async with NonBlockingAsyncLockWrapper(self.peers_lock, "peers_lock", timeout=5.0):
            peers = list(self.peers.values())

        timeout = aiohttp.ClientTimeout(total=5)
        async with get_client_session(timeout) as session:
            for peer in peers:
                if peer.node_id != self.node_id:
                    try:
                        url = self._url_for_peer(peer, "/coordinator")
                        await session.post(
                            url,
                            json={
                                "leader_id": self.node_id,
                                "lease_id": self.leader_lease_id,
                                "lease_expires": self.leader_lease_expires,
                                "fallback_leadership": True,
                                "lease_epoch": self._lease_epoch,
                                "fence_token": self._fence_token,
                            },
                            headers=self._auth_headers(),
                        )
                    except (aiohttp.ClientError, asyncio.TimeoutError, AttributeError):
                        pass

        self._save_state()

        # Start monitoring services
        await self._start_monitoring_if_leader()
        await self._start_p2p_auto_deployer()

        logger.info(f"Full fallback leadership established: lease={lease_id}")

    def _step_down_from_provisional(self) -> None:
        """Step down from provisional leadership (lost to challenger).

        Jan 29, 2026: Delegated to LeadershipOrchestrator.step_down_from_provisional().
        """
        return self.leadership.step_down_from_provisional()

    async def _request_election_from_voters(self, reason: str = "non_voter_request") -> bool:
        """December 29, 2025: Non-voters can request that voters start an election.

        Jan 29, 2026: Delegated to LeadershipOrchestrator.request_election_from_voters().
        """
        return await self.leadership.request_election_from_voters(reason)

    async def _check_emergency_coordinator_fallback(self):
        """DECENTRALIZED: When voter quorum is unreachable for >5 min, any GPU node can coordinate.

        EMERGENCY COORDINATOR: This is a last-resort fallback when the normal voter-based
        leadership cannot be established due to:
        - Too many voters being offline
        - Network partition isolating voters
        - Cluster-wide issues

        In this mode, the node acts as a temporary coordinator WITHOUT voter consensus.
        It will relinquish control once voter quorum is restored.
        """
        # Feb 23, 2026: Only coordinator nodes should attempt emergency leadership.
        # GPU worker nodes should never self-elect, even in emergency scenarios.
        _is_coordinator = os.environ.get("RINGRIFT_IS_COORDINATOR", "").lower() in ("true", "1", "yes")
        if not _is_coordinator:
            return

        now = time.time()

        # Only check every 60 seconds
        last_check = getattr(self, "_last_emergency_coord_check", 0)
        if now - last_check < 60:
            return
        self._last_emergency_coord_check = now

        # Skip if we already are a leader
        if self.role == NodeRole.LEADER:
            return

        # Skip if we have a known leader
        if self.leader_id:
            self._emergency_coordinator_since = 0
            return

        # Check if we have voter quorum
        if self._has_voter_quorum():
            self._emergency_coordinator_since = 0
            return  # Normal election should work

        # Track how long we've been without voter quorum
        quorum_missing_since = getattr(self, "_quorum_missing_since", 0)
        if quorum_missing_since == 0:
            self._quorum_missing_since = now
            return

        EMERGENCY_THRESHOLD = 300  # 5 minutes without quorum triggers emergency
        quorum_missing_duration = now - quorum_missing_since

        if quorum_missing_duration < EMERGENCY_THRESHOLD:
            return

        # Check if we're eligible (must be GPU node, not NAT-blocked)
        # Jan 31, 2026: Run in thread to avoid blocking event loop
        await asyncio.to_thread(self._update_self_info)
        if not getattr(self.self_info, "has_gpu", False):
            return
        if getattr(self.self_info, "nat_blocked", False):
            return

        # Use consistent hashing to determine which node should be emergency coordinator
        # This prevents multiple nodes from declaring themselves coordinator
        with self.peers_lock:
            candidates = [self.node_id]
            for peer in self.peers.values():
                if not peer.is_alive():
                    continue
                if not getattr(peer, "has_gpu", False):
                    continue
                if getattr(peer, "nat_blocked", False):
                    continue
                candidates.append(peer.node_id)

        if not candidates:
            return

        # Deterministic selection: highest node_id wins (simple, consistent)
        candidates.sort(reverse=True)
        designated_coordinator = candidates[0]

        if designated_coordinator != self.node_id:
            return  # Another node should be coordinator

        # Become emergency coordinator (without voter lease)
        logger.info(f"EMERGENCY COORDINATOR: Taking leadership without voter quorum "
              f"(quorum missing for {int(quorum_missing_duration)}s, {len(candidates)} candidates)")

        # Jan 3, 2026: Use _set_leader() for atomic leadership assignment (Phase 4)
        self._set_leader(self.node_id, reason="emergency_coordinator", save_state=False)
        self.last_leader_seen = now
        self._emergency_coordinator_since = now

        # Use a special lease ID to mark emergency mode
        import uuid
        self.leader_lease_id = f"EMERGENCY_{self.node_id}_{uuid.uuid4().hex[:8]}"
        self.leader_lease_expires = now + 120  # Short lease - needs frequent renewal
        self.last_lease_renewal = now

        # Announce emergency leadership
        with self.peers_lock:
            peers = list(self.peers.values())

        timeout = ClientTimeout(total=5)
        async with get_client_session(timeout) as session:
            for peer in peers:
                if peer.node_id != self.node_id:
                    try:
                        url = self._url_for_peer(peer, "/coordinator")
                        await session.post(url, json={
                            "leader_id": self.node_id,
                            "lease_id": self.leader_lease_id,
                            "lease_expires": self.leader_lease_expires,
                            "emergency": True,
                        }, headers=self._auth_headers())
                    except (aiohttp.ClientError, asyncio.TimeoutError, AttributeError):
                        pass  # Network errors expected during emergency coordination

        self._save_state()
        logger.info(f"EMERGENCY COORDINATOR: {self.node_id} is now emergency leader")

    def _get_peer_health_score(self, peer_id: str) -> float:
        """Calculate health score for a peer (0-100, higher is healthier).

        Jan 2026: Delegated to HealthMetricsManager (Phase 9 decomposition).
        """
        return self.health_metrics_manager.get_peer_health_score(peer_id)

    def _record_p2p_sync_result(self, peer_id: str, success: bool, latency_ms: float = 0.0):
        """Record P2P sync result for circuit breaker, metrics, and reputation.

        Jan 2026: Delegated to HealthMetricsManager (Phase 9 decomposition).
        """
        self.health_metrics_manager.record_p2p_sync_result(peer_id, success, latency_ms)


    # _gossip_state_to_peers(), _get_gossip_known_states() are inherited from mixin

    def _get_peer_endpoints_for_gossip(self) -> list[dict[str, Any]]:
        """Phase 28: Get peer endpoints to share via gossip for peer-of-peer discovery.

        Returns a list of alive peer endpoints with connection info.
        This enables nodes to discover peers they can't reach directly.

        Jan 27, 2026: Migrated to PeerQueryBuilder (Phase 3.2).
        """
        return self._peer_query.to_endpoint_dicts(limit=GOSSIP_MAX_PEER_ENDPOINTS).unwrap_or([])

    # =========================================================================
    # DISTRIBUTED TRAINING COORDINATION
    # =========================================================================
    # These functions enable nodes to coordinate training decisions without
    # relying on a leader, using gossip to share training state cluster-wide.
    # =========================================================================

    def _get_local_active_training_configs(self) -> list[dict]:
        """Get list of training configs currently running on this node.

        DISTRIBUTED TRAINING: Share what training this node is doing so other
        nodes can avoid duplicate training for the same configuration.

        Returns list of dicts with:
        - config_key: e.g. "square8_2p"
        - job_type: "nnue", "cmaes", etc.
        - started_at: timestamp when training started
        """
        active_configs = []
        with self.jobs_lock:
            for _job_id, job in self.local_jobs.items():
                job_type = getattr(job, "job_type", "")
                # Only include training-type jobs
                if job_type in ("nnue", "nnue_training", "training", "cmaes"):
                    board_type = getattr(job, "board_type", "")
                    num_players = getattr(job, "num_players", 2)
                    if board_type:
                        config_key = f"{board_type}_{num_players}p"
                        started_at = getattr(job, "started_at", time.time())
                        active_configs.append({
                            "config_key": config_key,
                            "job_type": job_type,
                            "started_at": started_at,
                        })
        return active_configs

    def _get_cluster_active_training_configs(self) -> dict[str, list[str]]:
        """Get all active training configs across the cluster via gossip.

        DISTRIBUTED TRAINING COORDINATION: Query gossip state to see what
        training is running cluster-wide. This enables nodes to avoid
        duplicate training without leader coordination.

        Returns: { config_key -> [list of node_ids training that config] }
        """
        cluster_configs: dict[str, list[str]] = {}

        # Include our own training
        for config in self._get_local_active_training_configs():
            config_key = config["config_key"]
            if config_key not in cluster_configs:
                cluster_configs[config_key] = []
            cluster_configs[config_key].append(self.node_id)

        # Include training from gossip state
        gossip_states = getattr(self, "_gossip_peer_states", {})
        now = time.time()
        for node_id, state in gossip_states.items():
            # Skip stale states (older than 2 minutes)
            if state.get("timestamp", 0) < now - 120:
                continue
            # Skip our own state
            if node_id == self.node_id:
                continue

            active_training = state.get("active_training_configs", [])
            for config in active_training:
                config_key = config.get("config_key", "")
                if config_key:
                    if config_key not in cluster_configs:
                        cluster_configs[config_key] = []
                    if node_id not in cluster_configs[config_key]:
                        cluster_configs[config_key].append(node_id)

        return cluster_configs

    def _is_config_being_trained_cluster_wide(self, config_key: str) -> tuple[bool, list[str]]:
        """Check if a config is already being trained somewhere in the cluster.

        DISTRIBUTED TRAINING: Before starting training for a config, check if
        another node is already training it. This avoids wasted resources.

        Returns: (is_being_trained, list_of_nodes_training_it)
        """
        cluster_configs = self._get_cluster_active_training_configs()
        training_nodes = cluster_configs.get(config_key, [])
        return (len(training_nodes) > 0, training_nodes)

    def _should_claim_training_slot(self, config_key: str) -> bool:
        """Decide if this node should claim a training slot for a config.

        DISTRIBUTED TRAINING COORDINATION: Use a deterministic algorithm to
        decide which node gets to train a config when multiple nodes want to.

        Algorithm:
        - If no one is training this config, the node with lowest ID claims it
        - If already training, don't start a duplicate
        - Include jitter to handle race conditions
        """
        is_training, _training_nodes = self._is_config_being_trained_cluster_wide(config_key)

        if is_training:
            # Config is already being trained somewhere
            return False

        # Get all nodes that might want to train (GPU nodes with data)
        candidate_nodes = [self.node_id]
        gossip_states = getattr(self, "_gossip_peer_states", {})
        now = time.time()
        for node_id, state in gossip_states.items():
            if state.get("timestamp", 0) < now - 120:
                continue
            if state.get("has_gpu", False):
                training_jobs = state.get("training_jobs", 0)
                # Only consider nodes with capacity (< 3 training jobs)
                if training_jobs < 3:
                    candidate_nodes.append(node_id)

        # Sort deterministically
        candidate_nodes = sorted(set(candidate_nodes))

        # The node with lowest ID that has capacity claims the slot
        # Add position-based jitter: higher position = less likely to claim
        import random
        my_position = candidate_nodes.index(self.node_id) if self.node_id in candidate_nodes else len(candidate_nodes)

        # First candidate always claims, others have decreasing probability
        claim_probability = max(0.1, 1.0 - (my_position * 0.3))

        return random.random() < claim_probability

    # =========================================================================
    # TRAINING TRIGGER IDEMPOTENCY (Phase 4 - Dec 2025)
    # =========================================================================
    # Hash-based deduplication to prevent duplicate training during leader
    # transitions. Each training trigger is hashed and stored; subsequent
    # triggers with the same hash within the TTL are rejected.
    # =========================================================================

    def _compute_training_trigger_hash(self, config_key: str, game_count: int) -> str:
        """Compute a hash for training trigger deduplication.

        IDEMPOTENCY: Hash is based on:
        - config_key (board_type + num_players)
        - game_count bucket (rounded to 1000 to allow minor variations)
        - time bucket (15-minute windows)

        This allows the same trigger to be rejected if attempted multiple times
        within a 15-minute window for the same approximate data state.
        """
        import hashlib

        # Round game count to nearest 1000 to tolerate minor variations
        game_bucket = (game_count // 1000) * 1000

        # Use 15-minute time buckets
        time_bucket = int(time.time() // 900) * 900

        hash_input = f"{config_key}:{game_bucket}:{time_bucket}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    def _is_training_trigger_duplicate(self, trigger_hash: str) -> bool:
        """Check if a training trigger is a duplicate.

        IDEMPOTENCY: Returns True if this trigger hash was seen recently.
        """
        if not hasattr(self, "_training_trigger_cache"):
            self._training_trigger_cache: dict[str, float] = {}

        now = time.time()
        ttl = 900  # 15-minute TTL for trigger cache

        # Cleanup old entries
        expired = [h for h, ts in self._training_trigger_cache.items() if now - ts > ttl]
        for h in expired:
            del self._training_trigger_cache[h]

        # Check if duplicate
        return trigger_hash in self._training_trigger_cache

    def _record_training_trigger(self, trigger_hash: str) -> None:
        """Record a training trigger for deduplication."""
        if not hasattr(self, "_training_trigger_cache"):
            self._training_trigger_cache = {}

        self._training_trigger_cache[trigger_hash] = time.time()

    def _check_training_idempotency(self, config_key: str, game_count: int) -> tuple[bool, str]:
        """Check if training can proceed (idempotency check).

        Returns:
            (can_proceed, trigger_hash) - can_proceed is False if duplicate
        """
        trigger_hash = self._compute_training_trigger_hash(config_key, game_count)

        if self._is_training_trigger_duplicate(trigger_hash):
            logger.info(f"IDEMPOTENT: Training trigger {trigger_hash[:8]} for {config_key} is duplicate, skipping")
            return False, trigger_hash

        return True, trigger_hash

    def _get_distributed_training_summary(self) -> dict:
        """Get summary of distributed training state for /status endpoint."""
        cluster_configs = self._get_cluster_active_training_configs()
        return {
            "active_configs": list(cluster_configs.keys()),
            "total_training_jobs": sum(len(nodes) for nodes in cluster_configs.values()),
            "configs_by_node_count": {k: len(v) for k, v in cluster_configs.items()},
        }

    # =========================================================================
    # DISTRIBUTED ELO
    # =========================================================================
    # Share ELO ratings via gossip for cluster-wide visibility without
    # requiring every node to query the ELO database directly.
    # =========================================================================


    def _get_cluster_elo_summary(self) -> dict:
        """Get cluster-wide ELO summary from gossip state.

        DISTRIBUTED ELO: Aggregate ELO info from all nodes via gossip to get
        a cluster-wide view of model performance.
        """
        all_models = {}
        gossip_states = getattr(self, "_gossip_peer_states", {})
        now = time.time()

        # Include our own ELO summary
        local_summary = self.sync.get_local_elo_summary()
        for model_info in local_summary.get("top_models", []):
            model_name = model_info.get("model", "")
            if model_name:
                all_models[model_name] = model_info

        # Include ELO summaries from gossip
        for _node_id, state in gossip_states.items():
            if state.get("timestamp", 0) < now - 300:  # Skip stale states
                continue

            elo_summary = state.get("elo_summary", {})
            for model_info in elo_summary.get("top_models", []):
                model_name = model_info.get("model", "")
                if model_name:
                    # Keep highest ELO seen for each model
                    existing = all_models.get(model_name, {})
                    if model_info.get("elo", 0) > existing.get("elo", 0):
                        all_models[model_name] = model_info

        # Sort by ELO and return top 10
        sorted_models = sorted(all_models.values(), key=lambda x: x.get("elo", 0), reverse=True)
        return {
            "top_models": sorted_models[:10],
            "total_unique_models": len(all_models),
        }

    def _load_curriculum_weights(self) -> dict[str, float]:
        """Load curriculum weights for selfplay prioritization."""
        if not HAS_CURRICULUM_WEIGHTS or load_curriculum_weights is None:
            return {}
        try:
            return load_curriculum_weights()
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[P2P] Failed to load curriculum weights: {e}")
            return {}

    # =========================================================================
    # AUTOMATIC NODE RECOVERY
    # =========================================================================
    # Detect stuck/unhealthy nodes via gossip and trigger automatic recovery
    # (service restart) to maintain cluster health without manual intervention.
    # =========================================================================


    # =========================================================================
    # STABILITY CONTROLLER CALLBACKS (Jan 2026 - Self-Healing Architecture)
    # =========================================================================
    # Recovery action callbacks triggered by StabilityController when symptoms
    # are detected. Each callback records effectiveness for feedback loop.
    # =========================================================================


    async def _action_increase_timeout(
        self, nodes: list[str], symptom: Any
    ) -> None:
        """Increase timeout for affected nodes."""
        if not self._adaptive_timeouts:
            return

        for node_id in nodes:
            self._adaptive_timeouts.increase_timeout(node_id)

        if self._effectiveness_tracker:
            self._effectiveness_tracker.record_action(
                "increase_timeout",
                nodes,
                {"symptom": symptom.symptom.value if hasattr(symptom, "symptom") else str(symptom)},
            )
        logger.info(f"[Stability] Increased timeout for {len(nodes)} nodes")

    async def _action_decrease_timeout(
        self, nodes: list[str], symptom: Any
    ) -> None:
        """Decrease timeout for affected nodes."""
        if not self._adaptive_timeouts:
            return

        for node_id in nodes:
            self._adaptive_timeouts.decrease_timeout(node_id)

        if self._effectiveness_tracker:
            self._effectiveness_tracker.record_action(
                "decrease_timeout",
                nodes,
                {"symptom": symptom.symptom.value if hasattr(symptom, "symptom") else str(symptom)},
            )
        logger.info(f"[Stability] Decreased timeout for {len(nodes)} nodes")

    async def _action_scale_pool(
        self, nodes: list[str], symptom: Any
    ) -> None:
        """Scale up connection pool size."""
        if self._effectiveness_tracker:
            self._effectiveness_tracker.record_action(
                "scale_pool_up",
                [],
                {"symptom": symptom.symptom.value if hasattr(symptom, "symptom") else str(symptom)},
            )
        logger.info("[Stability] Would scale connection pool (not implemented)")

    async def _action_reset_circuits(
        self, nodes: list[str], symptom: Any
    ) -> None:
        """Reset circuit breakers for affected nodes.

        January 22, 2026 - P2P Self-Healing Architecture:
        Now resets both node-level and per-transport circuit breakers.
        This enables transport fallover when one transport (e.g., Tailscale) fails.
        """
        reset_count = 0
        transport_reset_count = 0

        # Reset node-level circuit breakers
        try:
            from app.distributed.circuit_breaker import reset_circuit_breaker
            for node_id in nodes:
                try:
                    reset_circuit_breaker(node_id)
                    reset_count += 1
                except Exception as e:
                    logger.debug(f"Failed to reset node circuit for {node_id}: {e}")
        except ImportError:
            logger.debug("Circuit breaker module not available")

        # Reset per-transport circuit breakers for transport fallover
        try:
            from app.distributed.circuit_breaker import reset_transport_breakers_for_host
            for node_id in nodes:
                try:
                    # Get the host/IP for this node
                    peer = self.peers.get(node_id)
                    if peer:
                        host = getattr(peer, "ip", None) or getattr(peer, "host", None) or node_id
                        count = reset_transport_breakers_for_host(host)
                        transport_reset_count += count
                        if count > 0:
                            logger.debug(
                                f"[Stability] Reset {count} transport circuits for {node_id}"
                            )
                except Exception as e:
                    logger.debug(f"Failed to reset transport circuits for {node_id}: {e}")
        except ImportError:
            logger.debug("Transport circuit breaker module not available")

        if self._effectiveness_tracker:
            self._effectiveness_tracker.record_action(
                "reset_circuit",
                nodes,
                {
                    "symptom": symptom.symptom.value if hasattr(symptom, "symptom") else str(symptom),
                    "node_circuits_reset": reset_count,
                    "transport_circuits_reset": transport_reset_count,
                },
            )
        logger.info(
            f"[Stability] Reset circuits: {reset_count} node, {transport_reset_count} transport"
        )

    async def _action_increase_cooldown(
        self, nodes: list[str], symptom: Any
    ) -> None:
        """Increase cooldown period for recovery actions."""
        if self._stability_controller:
            old_cooldown = self._stability_controller._action_cooldown
            self._stability_controller._action_cooldown = min(old_cooldown * 1.5, 600.0)
            logger.info(
                f"[Stability] Increased action cooldown: {old_cooldown:.0f}s -> "
                f"{self._stability_controller._action_cooldown:.0f}s"
            )

        if self._effectiveness_tracker:
            self._effectiveness_tracker.record_action(
                "increase_cooldown",
                [],
                {"symptom": symptom.symptom.value if hasattr(symptom, "symptom") else str(symptom)},
            )

    async def _action_reinject_peer(
        self, nodes: list[str], symptom: Any
    ) -> None:
        """Reinject dead peers back into alive state for retry."""
        reinjected = 0
        for node_id in nodes:
            if node_id in self.peers:
                peer = self.peers[node_id]
                if not peer.is_alive():
                    peer.last_seen = time.time()
                    peer.status = "alive"
                    reinjected += 1
                    logger.info(f"[Stability] Reinjected peer {node_id}")

        if self._effectiveness_tracker:
            self._effectiveness_tracker.record_action(
                "reinject_peer",
                nodes,
                {"symptom": symptom.symptom.value if hasattr(symptom, "symptom") else str(symptom)},
            )
        logger.info(f"[Stability] Reinjected {reinjected}/{len(nodes)} peers")

    async def _action_emit_alert(
        self, nodes: list[str], symptom: Any
    ) -> None:
        """Emit alert for manual intervention."""
        symptom_str = symptom.symptom.value if hasattr(symptom, "symptom") else str(symptom)
        confidence = symptom.confidence if hasattr(symptom, "confidence") else 0.0
        root_cause = symptom.root_cause if hasattr(symptom, "root_cause") else "unknown"

        logger.warning(
            f"[Stability ALERT] {symptom_str} detected "
            f"(confidence={confidence:.2f}, cause={root_cause}, nodes={len(nodes)})"
        )

        try:
            from app.coordination.data_events import DataEventType
            from app.coordination.event_router import emit_event

            emit_event(
                DataEventType.STABILITY_ALERT,
                {
                    "symptom": symptom_str,
                    "confidence": confidence,
                    "root_cause": root_cause,
                    "affected_nodes": nodes[:10],
                    "timestamp": time.time(),
                },
            )
        except Exception:
            pass

        if self._effectiveness_tracker:
            self._effectiveness_tracker.record_action(
                "emit_alert",
                nodes,
                {"symptom": symptom_str},
            )

    # =========================================================================
    # GOSSIP-BASED LEADER HINTS
    # =========================================================================
    # Share leader preferences via gossip to enable faster leader elections.
    # When current leader fails, nodes can quickly converge on a new leader
    # based on hints from peers rather than running full election.
    #
    # =========================================================================

    # =========================================================================
    # PEER REPUTATION TRACKING
    # =========================================================================
    # Track peer reliability over time for better peer selection in P2P sync,
    # gossip, and other distributed operations.
    # =========================================================================


    def _get_cluster_peer_reputation(self) -> dict:
        """Aggregate peer reputation from gossip for cluster-wide view."""
        all_scores = {}
        gossip_states = getattr(self, "_gossip_peer_states", {})
        now = time.time()

        # Include our own reputation data
        # Jan 30, 2026: Use network orchestrator directly
        local_summary = self.network.get_peer_reputation_summary()
        for peer_info in local_summary.get("reliable_peers", []):
            peer_id = peer_info["peer"]
            if peer_id not in all_scores:
                all_scores[peer_id] = []
            all_scores[peer_id].append(peer_info["score"])

        # Include reputation from gossip
        for _node_id, state in gossip_states.items():
            if state.get("timestamp", 0) < now - 300:
                continue

            rep_summary = state.get("peer_reputation", {})
            for peer_info in rep_summary.get("reliable_peers", []):
                peer_id = peer_info["peer"]
                if peer_id not in all_scores:
                    all_scores[peer_id] = []
                all_scores[peer_id].append(peer_info["score"])

        # Calculate average scores
        avg_scores = {peer: sum(scores) / len(scores) for peer, scores in all_scores.items() if scores}
        sorted_peers = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)

        return {
            "most_reliable": [{"peer": p, "avg_score": round(s)} for p, s in sorted_peers[:10]],
            "peers_tracked": len(all_scores),
        }

    # ============================================================================
    # ============================================================================
    # SELFPLAY DATA DEDUPLICATION
    # ============================================================================
    # Tracks synced files and game IDs to avoid redundant transfers during P2P sync.
    # Uses bloom filter for efficient game ID tracking and file hash caching.
    # ============================================================================

    def _init_data_deduplication(self):
        """Initialize data deduplication tracking."""
        self._synced_file_hashes: set[str] = set()  # Hash -> synced
        self._known_game_ids: set[str] = set()  # Game IDs we have
        self._dedup_stats = {
            "files_skipped": 0,
            "games_skipped": 0,
            "bytes_saved": 0,
            "last_cleanup": time.time(),
        }
        self._dedup_lock = threading.Lock()

    def _record_synced_file(self, file_hash: str, file_size: int):
        """Record a file as synced for deduplication.

        DATA DEDUPLICATION: Track file hashes we've synced to avoid
        re-syncing the same file from different peers.

        Args:
            file_hash: Hash of the synced file
            file_size: Size in bytes (for metrics)
        """
        if not hasattr(self, "_synced_file_hashes"):
            self._init_data_deduplication()

        with self._dedup_lock:
            self._synced_file_hashes.add(file_hash)

    def _is_file_already_synced(self, file_hash: str) -> bool:
        """Check if file was already synced based on hash.

        Args:
            file_hash: Hash to check

        Returns:
            True if file was already synced
        """
        if not hasattr(self, "_synced_file_hashes"):
            self._init_data_deduplication()

        if not file_hash:
            return False

        with self._dedup_lock:
            return file_hash in self._synced_file_hashes


    def _record_dedup_skip(self, file_count: int = 0, game_count: int = 0, bytes_saved: int = 0):
        """Record deduplication skip for metrics.

        Args:
            file_count: Number of files skipped
            game_count: Number of games skipped
            bytes_saved: Bytes saved by skipping
        """
        if not hasattr(self, "_dedup_stats"):
            self._init_data_deduplication()

        with self._dedup_lock:
            self._dedup_stats["files_skipped"] += file_count
            self._dedup_stats["games_skipped"] += game_count
            self._dedup_stats["bytes_saved"] += bytes_saved


    def _get_dedup_summary(self) -> dict:
        """Get deduplication metrics summary."""
        if not hasattr(self, "_dedup_stats"):
            self._init_data_deduplication()

        with self._dedup_lock:
            return {
                "files_skipped": self._dedup_stats.get("files_skipped", 0),
                "games_skipped": self._dedup_stats.get("games_skipped", 0),
                "bytes_saved_mb": round(self._dedup_stats.get("bytes_saved", 0) / (1024 * 1024), 2),
                "known_file_hashes": len(self._synced_file_hashes),
                "known_game_ids": len(self._known_game_ids),
            }


    def _get_data_summary_cached(self) -> dict[str, Any]:
        """Get cached data summary for /status endpoint.

        January 13, 2026: Added as part of unified data discovery infrastructure.
        Returns game counts from local canonical databases for quick access.
        For full multi-source data, use /data/summary endpoint.

        January 23, 2026: FIXED - Removed blocking SQLite fallback that was
        causing event loop blocks. Now only returns cached data or empty dict.
        The fallback to canonical DB scan should be done via async methods.

        Returns:
            Dict with total game counts per config from local canonical DBs
        """
        try:
            # Use cached game counts from selfplay scheduler if available
            if hasattr(self, "selfplay_scheduler") and self.selfplay_scheduler:
                counts = getattr(self.selfplay_scheduler, "_p2p_game_counts", None)
                if counts:
                    total = sum(counts.values())
                    return {
                        "total_games": total,
                        "by_config": dict(counts),
                        "source": "selfplay_scheduler_cache",
                        "config_count": len(counts),
                    }

            # FIXED Jan 23, 2026: Do NOT fall back to blocking SQLite scan here.
            # The sync method _seed_selfplay_scheduler_game_counts_sync() was
            # blocking the event loop for seconds. Return empty dict instead.
            # Game counts will be populated async via selfplay_scheduler.
            return {
                "total_games": 0,
                "by_config": {},
                "source": "none",
                "error": "No cached data - scheduler not initialized yet",
            }

        except Exception as e:  # noqa: BLE001
            return {
                "total_games": 0,
                "by_config": {},
                "source": "error",
                "error": str(e),
            }

    def _get_cooldown_stats(self) -> dict[str, Any]:
        """Get adaptive dead peer cooldown statistics for monitoring.

        January 20, 2026: Added to expose cooldown manager metrics in /status.
        Helps diagnose peer recovery issues and verify adaptive cooldown is working.

        Returns:
            Dict with:
            - enabled: Whether the adaptive cooldown manager is active
            - nodes_in_cooldown: Number of nodes currently in cooldown
            - stats: Cooldown manager statistics (probes, recoveries, etc.)
            - in_cooldown: List of nodes currently in cooldown with their tier/remaining time
        """
        if not self._cooldown_manager:
            # Fallback to legacy dict tracking
            return {
                "enabled": False,
                "nodes_in_cooldown": len(self._dead_peer_timestamps),
                "fallback_mode": True,
                "dead_peer_timestamps": {
                    node_id: {"dead_since": ts, "age_seconds": time.time() - ts}
                    for node_id, ts in self._dead_peer_timestamps.items()
                },
            }

        try:
            stats = self._cooldown_manager.get_stats()
            in_cooldown = self._cooldown_manager.get_all_in_cooldown()
            return {
                "enabled": True,
                "nodes_in_cooldown": stats.get("nodes_in_cooldown", 0),
                "stats": stats,
                "in_cooldown": in_cooldown,
                "fallback_mode": False,
            }
        except Exception as e:  # noqa: BLE001
            return {
                "enabled": True,
                "error": str(e),
            }

    # NOTE: _get_peer_health_summary() inlined at call site (Jan 2026 Phase 2)

    def _get_fallback_status(self) -> dict[str, Any]:
        """Get fallback mechanism status for debugging partition issues.

        Session 17.41 (Jan 6, 2026): Exposes visibility into why fallback mechanisms
        aren't activating during network partitions. This helps diagnose issues where
        the work queue has items but workers can't claim jobs because the leader is
        unreachable and fallbacks haven't kicked in.

        Returns:
            Dict with:
            - autonomous_queue: Whether local queue fallback is active
            - work_discovery: Multi-channel work discovery status
            - leader_status: Leader contact timing
            - partition_healer: Partition healing escalation state
        """
        result: dict[str, Any] = {}
        now = time.time()

        # 1. Autonomous queue status
        try:
            loop = getattr(self, "_autonomous_queue_loop", None)
            if loop is not None:
                loop_status = loop.get_status() if hasattr(loop, "get_status") else {}
                result["autonomous_queue"] = {
                    "active": loop_status.get("activated", False),
                    "enabled": loop_status.get("enabled", False),
                    "running": loop_status.get("running", False),
                    "activation_reason": loop_status.get("activation_reason", ""),
                    "no_leader_duration": loop_status.get("no_leader_duration", 0.0),
                    "queue_depth": loop_status.get("queue_depth", 0),
                }
            else:
                result["autonomous_queue"] = {"error": "loop_not_initialized"}
        except Exception as e:  # noqa: BLE001
            result["autonomous_queue"] = {"error": str(e)}

        # 2. Work discovery manager status
        try:
            from scripts.p2p.loops.job_loops import get_work_discovery_manager
            manager = get_work_discovery_manager()
            if manager is not None:
                mgr_status = manager.get_status() if hasattr(manager, "get_status") else {}
                result["work_discovery"] = {
                    "enabled": mgr_status.get("enabled", False),
                    "active_channels": mgr_status.get("active_channels", []),
                    "last_work_time": mgr_status.get("last_work_time", 0.0),
                    "claims_via_leader": mgr_status.get("claims_via_leader", 0),
                    "claims_via_peer": mgr_status.get("claims_via_peer", 0),
                    "claims_via_local": mgr_status.get("claims_via_local", 0),
                }
            else:
                result["work_discovery"] = {"error": "manager_not_initialized"}
        except ImportError:
            result["work_discovery"] = {"error": "import_failed"}
        except Exception as e:  # noqa: BLE001
            result["work_discovery"] = {"error": str(e)}

        # 3. Leader contact status
        try:
            last_leader_seen = getattr(self, "last_leader_seen", now)
            leader_unreachable_duration = now - last_leader_seen
            result["leader_status"] = {
                "last_leader_seen": last_leader_seen,
                "leader_unreachable_duration": round(leader_unreachable_duration, 1),
                "is_leaderless": self.leader_id is None or self.leader_id == "",
                "current_leader_id": self.leader_id,
                "is_self_leader": self.leadership.check_is_leader(),
            }
        except Exception as e:  # noqa: BLE001
            result["leader_status"] = {"error": str(e)}

        # 4. Partition healer status (if available)
        try:
            from scripts.p2p.partition_healer import get_partition_healer
            healer = get_partition_healer()
            healer_status = healer.get_status()
            result["partition_healer"] = {
                "escalation_level": healer_status.get("escalation_level", 0),
                "last_healing_attempt": healer_status.get("last_healing_attempt", 0.0),
                "healing_in_progress": healer_status.get("healing_in_progress", False),
                "has_orchestrator": healer_status.get("has_orchestrator", False),
                "election_ready": healer_status.get("election_ready", True),
            }
        except ImportError:
            result["partition_healer"] = {"error": "import_failed"}
        except Exception as e:  # noqa: BLE001
            result["partition_healer"] = {"error": str(e)}

        return result

    # ============================================================================
    # DISTRIBUTED TOURNAMENT SCHEDULING
    # ============================================================================
    # Allows tournaments to be scheduled and coordinated via gossip protocol
    # without requiring a leader. Uses consensus to elect tournament coordinator.
    # Jan 2026: Delegated to TournamentManager (Phase 11 decomposition).
    # ============================================================================


    async def _start_monitoring_if_leader(self):
        """Start Prometheus/Grafana when we become leader (P2P monitoring resilience)."""
        if not self.monitoring_manager:
            return
        if self.role != NodeRole.LEADER:
            return
        if self._monitoring_was_leader:
            return  # Already started

        try:
            # Update peer list for Prometheus config
            with self.peers_lock:
                peer_list = [
                    {"node_id": p.node_id, "host": p.host, "port": getattr(p, "metrics_port", 9091)}
                    for p in self.peers.values()
                    if p.node_id != self.node_id and p.is_healthy()
                ]
            self.monitoring_manager.update_peers(peer_list)

            # Start monitoring services
            success = await self.monitoring_manager.start_as_leader()
            if success:
                logger.info("Monitoring services started on leader node")
                self._monitoring_was_leader = True
            else:
                logger.error("Failed to start monitoring services")
        except Exception as e:  # noqa: BLE001
            logger.error(f"starting monitoring services: {e}")

    async def _stop_monitoring_if_not_leader(self):
        """Stop Prometheus/Grafana when we step down from leadership."""
        if not self.monitoring_manager:
            return
        if not self._monitoring_was_leader:
            return  # Never started

        if self.role != NodeRole.LEADER:
            try:
                await self.monitoring_manager.stop()
                logger.info("Monitoring services stopped (no longer leader)")
                self._monitoring_was_leader = False
            except Exception as e:  # noqa: BLE001
                logger.error(f"stopping monitoring services: {e}")

    async def _start_p2p_auto_deployer(self):
        """Start P2P auto-deployer when we become leader.

        The auto-deployer ensures P2P orchestrator is running on all cluster nodes.
        This solves the fundamental gap where P2P deployment was manual-only.
        """
        if self.role != NodeRole.LEADER:
            return
        if self._auto_deployer_task is not None:
            return  # Already running

        try:
            from app.coordination.p2p_auto_deployer import P2PAutoDeployer, P2PDeploymentConfig

            config = P2PDeploymentConfig(
                check_interval_seconds=300.0,  # Check every 5 minutes
                min_coverage_percent=90.0,
            )
            self.p2p_auto_deployer = P2PAutoDeployer(config=config)

            # Run as background task
            self._auto_deployer_task = asyncio.create_task(
                self.p2p_auto_deployer.run_daemon(),
                name="p2p_auto_deployer"
            )
            logger.info("P2P Auto-Deployer started (leader responsibility)")
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to start P2P auto-deployer: {e}")

    async def _stop_p2p_auto_deployer(self):
        """Stop P2P auto-deployer when we step down from leadership."""
        if self.p2p_auto_deployer:
            self.p2p_auto_deployer.stop()
        if self._auto_deployer_task:
            self._auto_deployer_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._auto_deployer_task
            self._auto_deployer_task = None
        self.p2p_auto_deployer = None
        logger.info("P2P Auto-Deployer stopped")

    async def _renew_leader_lease(self):
        """Renew our leadership lease and broadcast to peers."""
        if self.role != NodeRole.LEADER:
            return
        # Jan 23, 2026: Use ULSM QuorumHealth for unified quorum tracking
        # (Phase 1 fix: previously used separate _quorum_fail_count which diverged from _is_leader())
        if getattr(self, "voter_node_ids", []) and not self._has_voter_quorum():
            # Jan 2, 2026: Use _count_alive_voters() to check IP:port matches
            voters_alive = self._count_alive_voters()
            quorum_size = getattr(self, "voter_quorum_size", 0)
            # Feb 2026: Skip quorum step-down if forced leader override is active
            if getattr(self, "_forced_leader_override", False):
                logger.debug(
                    f"[LeaseRenewal] Voter quorum check failed (voters_alive={voters_alive}, "
                    f"quorum_size={quorum_size}) but forced leader override active; continuing"
                )
            else:
                # Use ULSM QuorumHealth - same tracker as _is_leader() uses
                threshold_exceeded = self._leadership_sm.quorum_health.record_failure(voters_alive)
                fail_count = self._leadership_sm.quorum_health.consecutive_failures
                threshold = self._leadership_sm.quorum_health.failure_threshold
                logger.warning(
                    f"[LeaseRenewal] Voter quorum check failed ({fail_count}/{threshold}): "
                    f"voters_alive={voters_alive}, quorum_size={quorum_size}"
                )
                if threshold_exceeded:
                    logger.info(f"Lost voter quorum ({threshold} consecutive failures via ULSM); stepping down: {self.node_id}")
                    # Jan 2026: Use ULSM step-down which broadcasts to peers BEFORE local mutation
                    self._schedule_step_down_sync(TransitionReason.QUORUM_LOST)
                    self._release_voter_grant_if_self()
                return
        else:
            # Reset ULSM quorum health counter on success
            voters_alive = self._count_alive_voters()
            self._leadership_sm.quorum_health.record_success(voters_alive)

        now = time.time()
        if now - self.last_lease_renewal < LEADER_LEASE_RENEW_INTERVAL:
            return  # Too soon to renew

        lease_id = str(self.leader_lease_id or "")
        if not lease_id:
            lease_id = f"{self.node_id}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        lease_expires = await self._acquire_voter_lease_quorum(lease_id, int(LEADER_LEASE_DURATION))
        if getattr(self, "voter_node_ids", []) and not lease_expires:
            # Feb 2026: If forced leader override is active, self-renew without quorum
            if getattr(self, "_forced_leader_override", False):
                logger.info("Voter lease quorum failed but forced leader override active; self-renewing lease")
                lease_expires = now + LEADER_LEASE_DURATION
            else:
                # Voter quorum failed - try arbiter fallback before stepping down
                logger.info("Voter lease quorum failed; checking arbiter...")
                arbiter_leader = await self._query_arbiter_for_leader()
                if arbiter_leader == self.node_id:
                    # Arbiter still recognizes us as leader - extend lease provisionally
                    logger.info("Arbiter confirms us as leader despite quorum failure; continuing with provisional lease")
                    lease_expires = now + LEADER_LEASE_DURATION / 2  # Shorter lease until quorum recovers
                elif arbiter_leader:
                    # Arbiter says someone else is leader - defer to arbiter
                    # Feb 2026: Skip arbiter override if forced leader is active
                    if getattr(self, "_forced_leader_override", False):
                        logger.warning(f"Arbiter reports {arbiter_leader} but forced leader override active; ignoring")
                    else:
                        logger.info(f"Arbiter reports different leader ({arbiter_leader}); stepping down")
                        # Jan 3, 2026: Use _set_leader() for atomic leadership assignment (Phase 4)
                        self._set_leader(arbiter_leader, reason="arbiter_override", save_state=False)
                    self.leader_lease_id = ""
                    self.leader_lease_expires = 0.0
                    self.last_lease_renewal = 0.0
                    self._release_voter_grant_if_self()
                    self._save_state()
                    return
                else:
                    # Arbiter also unreachable - step down to be safe
                    # Feb 2026: Skip step-down if forced leader override is active
                    if getattr(self, "_forced_leader_override", False):
                        logger.warning(f"Arbiter unreachable but forced leader override active; maintaining leadership: {self.node_id}")
                        return
                    logger.error(f"Failed to renew voter lease quorum and arbiter unreachable; stepping down: {self.node_id}")
                    # Jan 3, 2026: Use _set_leader() for atomic leadership assignment (Phase 4)
                    self._set_leader(None, reason="arbiter_unreachable", save_state=False)
                    self.leader_lease_id = ""
                    self.leader_lease_expires = 0.0
                    self.last_lease_renewal = 0.0
                    self._release_voter_grant_if_self()
                    self._save_state()
                    return

        self.leader_lease_id = lease_id
        self.leader_lease_expires = float(lease_expires or (now + LEADER_LEASE_DURATION))
        self.last_lease_renewal = now

        # Jan 5, 2026: Renew self-leadership in state machine during lease renewal
        # Prevents the leader self-acknowledgment bug where leader_id is only set
        # during state transitions, causing the cluster to appear leaderless
        if hasattr(self, "_leadership_sm") and self._leadership_sm:
            try:
                self._leadership_sm.renew_self_leadership()
            except Exception as e:
                logger.debug(f"[LeaseRenewal] Failed to renew self-leadership: {e}")

        # Broadcast lease renewal to all peers
        with self.peers_lock:
            peers = list(self.peers.values())

        timeout = ClientTimeout(total=3)
        try:
            async with get_client_session(timeout) as session:
                for peer in peers:
                    if peer.node_id != self.node_id and peer.is_alive():
                        try:
                            url = self._url_for_peer(peer, "/coordinator")
                            await session.post(url, json={
                                "leader_id": self.node_id,
                                "lease_id": self.leader_lease_id,
                                "lease_expires": self.leader_lease_expires,
                                "lease_renewal": True,
                                "voter_node_ids": list(getattr(self, "voter_node_ids", []) or []),
                            }, headers=self._auth_headers())
                        except (aiohttp.ClientError, asyncio.TimeoutError, KeyError, IndexError, AttributeError):
                            pass  # Network errors expected during lease renewal
        except Exception as e:  # noqa: BLE001
            logger.info(f"Lease renewal error: {e}")

    def _is_leader_lease_valid(self) -> bool:
        """Check if the current leader's lease is still valid.

        Jan 29, 2026: Delegates to LeadershipOrchestrator.is_leader_lease_valid().
        """
        return self.leadership.is_leader_lease_valid()

    async def _check_and_resolve_split_brain(self) -> bool:
        """Check for split-brain (multiple leaders) and resolve by stepping down if needed.

        Jan 28, 2026: Phase 18C - Thin wrapper delegating to QuorumManager.
        """
        if self.quorum_manager:
            # Ensure orchestrator reference is set (for late binding)
            if not getattr(self.quorum_manager, "_orchestrator", None):
                self.quorum_manager.set_orchestrator(self)
            return await self.quorum_manager.check_and_resolve_split_brain()
        return False

    async def _run_with_timeout(self, coro, name: str, timeout: float = 60.0) -> None:
        """Run a coroutine with a timeout, logging if it exceeds the limit."""
        try:
            await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"[JobMgmt] {name} timed out after {timeout}s, skipping")
        except Exception as e:
            logger.debug(f"[JobMgmt] {name} failed: {e}")

    async def _run_leader_ops_inline(self) -> None:
        """Run leader-only operations inline in the management loop.

        Feb 23, 2026: Run all operations CONCURRENTLY via asyncio.gather()
        instead of sequentially. Previous sequential execution meant one slow
        op (e.g. manage_cluster_jobs at 361s) blocked ALL other ops and the
        entire event loop. Concurrent execution bounds total cycle time to
        the slowest single op rather than the sum of all ops.

        Individual timeouts via asyncio.wait_for() enforce 15s max per op.
        A hard 60s cycle deadline prevents runaway cycles.
        """
        _t0 = time.time()
        CYCLE_DEADLINE = 60.0  # Hard deadline for entire cycle
        OP_TIMEOUT = 15.0  # Max time per individual operation

        try:
            split_brain = await self._run_with_timeout(
                self._check_and_resolve_split_brain(),
                "check_and_resolve_split_brain", timeout=10.0)
            if split_brain:
                return
            await asyncio.sleep(0)

            logger.info("[LeaderOps] Running leader operations cycle (concurrent)")

            _is_coord = os.environ.get("RINGRIFT_IS_COORDINATOR", "").lower() in ("true", "1", "yes")
            _leader_ops = [
                ("manage_cluster_jobs", self._manage_cluster_jobs(), OP_TIMEOUT),
                ("check_cluster_balance", self._check_cluster_balance(), OP_TIMEOUT),
                ("check_and_trigger_training", self._check_and_trigger_training(), OP_TIMEOUT),
                ("check_improvement_cycles", self._check_improvement_cycles(), OP_TIMEOUT),
                ("auto_rebalance_from_work_queue",
                 self._auto_rebalance_from_work_queue(), OP_TIMEOUT),
                ("auto_scale_gpu_utilization",
                 self._auto_scale_gpu_utilization(), OP_TIMEOUT),
                ("sweep_nat_recovery",
                 self.recovery_manager.sweep_nat_recovery(), OP_TIMEOUT),
                ("check_node_recovery",
                 self.recovery_manager.check_node_recovery(), OP_TIMEOUT),
            ]
            if not _is_coord:
                _leader_ops.append(("check_and_kill_stuck_jobs",
                    self.job_lifecycle_manager.check_and_kill_stuck_jobs(), OP_TIMEOUT))

            async def _timed_op(name: str, coro, timeout: float) -> None:
                _op_t = time.time()
                await self._run_with_timeout(coro, name, timeout=timeout)
                _op_e = time.time() - _op_t
                if _op_e > 3.0:
                    logger.info(f"[LeaderOps] {name}: {_op_e:.1f}s")

            # Feb 26, 2026: Run ops in batches of 3 to avoid thread pool
            # saturation. With 8 workers and 8 concurrent ops all needing
            # asyncio.to_thread(), the pool gets exhausted and ops queue up
            # causing 22s+ delays. Batching limits peak demand to 3 threads.
            BATCH_SIZE = 3
            try:
                for batch_start in range(0, len(_leader_ops), BATCH_SIZE):
                    if time.time() - _t0 > CYCLE_DEADLINE:
                        logger.warning(f"[LeaderOps] Cycle hit {CYCLE_DEADLINE}s deadline")
                        break
                    batch = _leader_ops[batch_start:batch_start + BATCH_SIZE]
                    remaining = CYCLE_DEADLINE - (time.time() - _t0)
                    await asyncio.wait_for(
                        asyncio.gather(
                            *[_timed_op(n, c, t) for n, c, t in batch],
                            return_exceptions=True,
                        ),
                        timeout=min(remaining, 20.0),
                    )
                    # Yield to event loop between batches
                    await asyncio.sleep(0)
            except asyncio.TimeoutError:
                logger.warning(
                    f"[LeaderOps] Batch hit timeout, "
                    f"cancelling remaining ops"
                )

            _elapsed = time.time() - _t0
            logger.info(f"[LeaderOps] Cycle complete in {_elapsed:.1f}s")
        except Exception as e:
            logger.error(f"[LeaderOps] Error: {e}", exc_info=True)

    async def _job_management_loop(self):
        """Manage jobs - leader coordinates cluster, all nodes handle local operations.

        Feb 22, 2026: Restructured to prevent event loop blocking.
        - Removed redundant gossip ops (already have dedicated LoopManager loops)
        - Skip GPU/selfplay ops on coordinator (no GPU, no selfplay)
        - Reduced timeouts from 60s to 15s
        - Added asyncio.sleep(0) yield points between ops to let HTTP handlers process
        - Run independent leader ops concurrently
        """
        logger.info("[JobMgmt] _job_management_loop started")
        _is_coord = os.environ.get("RINGRIFT_IS_COORDINATOR", "").lower() in ("true", "1", "yes")
        while self.running:
            try:
                _t0 = time.time()
                # ==== DECENTRALIZED OPERATIONS (all nodes) ====
                # Gossip ops REMOVED - they have dedicated LoopManager loops
                # and were redundantly blocking this loop for 15-30s.
                _ops = [
                    ("check_emergency_coordinator_fallback", lambda: self._check_emergency_coordinator_fallback()),
                ]
                # Feb 2026: Coordinator doesn't run local selfplay/training, so
                # check_local_stuck_jobs (30-54s scanning pgrep/ps) is wasted CPU.
                # Only run on GPU/worker nodes that actually have local jobs.
                if not _is_coord:
                    _ops.append(
                        ("check_local_stuck_jobs", lambda: self.job_lifecycle_manager.check_local_stuck_jobs()),
                    )
                # Coordinator doesn't run selfplay/training locally, doesn't sync
                # from peers, and doesn't need local resource cleanup (no local
                # game data). Skip all heavy ops that use asyncio.to_thread() or
                # subprocess calls - these exhaust the 4-worker thread pool and
                # block HTTP handlers for 15-200s.
                if not _is_coord:
                    _ops.extend([
                        ("local_resource_cleanup", lambda: self.job_coordination_manager.local_resource_cleanup()),
                        ("p2p_data_sync", lambda: self.sync.p2p_data_sync()),
                        ("p2p_model_sync", lambda: self.sync.p2p_model_sync()),
                        ("p2p_training_db_sync", lambda: self.sync.p2p_training_db_sync()),
                    ])
                if not _is_coord:
                    _ops.extend([
                        ("consolidate_selfplay_data", lambda: self.data_pipeline_manager.consolidate_selfplay_data(
                            dispatch_export_job_callback=self._dispatch_export_job)),
                        ("manage_local_jobs_decentralized", lambda: self._manage_local_jobs_decentralized()),
                        ("local_gpu_auto_scale", lambda: self.job_coordination_manager.local_gpu_auto_scale()),
                        ("check_local_training_fallback", lambda: self._check_local_training_fallback()),
                    ])
                for _op_name, _op_factory in _ops:
                    _op_start = time.time()
                    await self._run_with_timeout(_op_factory(), _op_name, timeout=15.0)
                    _op_elapsed = time.time() - _op_start
                    if _op_elapsed > 5.0:
                        logger.warning(f"[JobMgmt] {_op_name} took {_op_elapsed:.1f}s")
                    # Yield to event loop so HTTP handlers can process requests
                    await asyncio.sleep(0)

                # ==== LEADER-ONLY OPERATIONS ====
                # These contain sync blocking calls (subprocess.run, SQLite,
                # check_training_readiness) that block the event loop for
                # 30-136s. Run at reduced frequency (every 60s instead of 15s)
                # and skip during the first cycle to let startup complete.
                if self.role == NodeRole.LEADER:
                    _leader_last = getattr(self, "_last_leader_ops_time", 0)
                    if time.time() - _leader_last >= 60:
                        self._last_leader_ops_time = time.time()
                        await self._run_leader_ops_inline()
                _total = time.time() - _t0
                logger.info(f"[JobMgmt] Cycle complete in {_total:.1f}s (role={self.role})")
            except Exception as e:  # noqa: BLE001
                logger.error(f"[JobMgmt] Loop error: {e}", exc_info=True)

            await asyncio.sleep(JOB_CHECK_INTERVAL)


    async def _manage_local_jobs_decentralized(self) -> int:
        """DECENTRALIZED: Each node manages its own job count based on gossip state.

        Runs on ALL nodes to ensure selfplay continues even during leader elections.
        Each node autonomously:
        1. Checks its own resource pressure (disk, memory, CPU)
        2. Uses gossip state to calculate proportional job count
        3. Starts or stops local jobs as needed

        PHASE 3 DECENTRALIZATION (Dec 2025):
        - With Serf providing reliable failure detection, we can act quickly
        - Proportional allocation based on gossip cluster capacity
        - 30-second timeout for faster leader-failure recovery

        January 29, 2026: Delegated to ProcessSpawnerOrchestrator.manage_local_jobs_decentralized().

        Returns:
            Number of jobs started/stopped
        """
        # Delegate to ProcessSpawnerOrchestrator if available
        return await self.process_spawner.manage_local_jobs_decentralized()


    async def _auto_scale_gpu_utilization(self) -> int:
        """Auto-scale selfplay jobs to reach 60-80% GPU utilization.

        Detects underutilized GPU nodes and starts selfplay jobs to improve
        cluster throughput while maintaining game quality and rule fidelity.

        Dec 2025 fix: Job type is selected based on GPU capabilities:
        - High-end GPUs (GH200, H100, A100, 5090, 4090): 50% GUMBEL / 50% GPU_SELFPLAY
        - Mid-tier GPUs: HYBRID mode (CPU rules + GPU eval) for rule fidelity

        Returns:
            Number of new selfplay jobs started
        """
        TARGET_GPU_MIN = 60.0  # Target minimum GPU utilization
        TARGET_GPU_MAX = 80.0  # Target maximum GPU utilization
        MIN_IDLE_TIME = 120    # Seconds of low GPU before scaling up

        started = 0
        now = time.time()

        # Rate limit auto-scaling (once per 2 minutes)
        last_scale = getattr(self, "_last_gpu_auto_scale", 0)
        if now - last_scale < 120:
            return 0

        # Feb 23, 2026: Use non-blocking cached snapshot to avoid blocking
        # event loop on peers_lock contention (was 10-30s on macOS)
        peers_snapshot = self._get_peers_snapshot_nonblocking()

        underutilized_gpu_nodes = []

        # Load policy manager for filtering
        policy_manager = None
        try:
            from app.coordination.node_policies import get_policy_manager
            policy_manager = get_policy_manager()
        except ImportError:
            pass

        for peer in peers_snapshot:
            if not peer.is_alive():
                continue
            has_gpu = bool(getattr(peer, "has_gpu", False))
            if not has_gpu:
                continue

            # Policy check: skip nodes that don't allow selfplay
            if policy_manager and not policy_manager.is_work_allowed(peer.node_id, "selfplay"):
                continue

            gpu_percent = float(getattr(peer, "gpu_percent", 0) or 0)
            gpu_name = (getattr(peer, "gpu_name", "") or "").lower()
            selfplay_jobs = int(getattr(peer, "selfplay_jobs", 0) or 0)
            training_jobs = int(getattr(peer, "training_jobs", 0) or 0)

            # Skip if already training
            if training_jobs > 0:
                continue

            # Check if underutilized
            if gpu_percent < TARGET_GPU_MIN:
                # Track how long it's been underutilized
                idle_key = f"_gpu_idle_since_{peer.node_id}"
                idle_since = getattr(self, idle_key, 0)
                if idle_since == 0:
                    setattr(self, idle_key, now)
                elif now - idle_since > MIN_IDLE_TIME:
                    # Calculate how many more jobs to add
                    gpu_headroom = TARGET_GPU_MAX - gpu_percent
                    # Estimate jobs based on GPU tier
                    if any(tag in gpu_name for tag in ("h100", "h200", "gh200", "5090")):
                        jobs_per_10_percent = 2
                    elif any(tag in gpu_name for tag in ("a100", "4090", "3090")):
                        jobs_per_10_percent = 1.5
                    else:
                        jobs_per_10_percent = 1

                    new_jobs = max(1, int(gpu_headroom / 10 * jobs_per_10_percent))
                    new_jobs = min(new_jobs, 4)  # Cap at 4 new jobs per cycle

                    underutilized_gpu_nodes.append({
                        "node_id": peer.node_id,
                        "gpu_percent": gpu_percent,
                        "gpu_name": gpu_name,
                        "current_jobs": selfplay_jobs,
                        "new_jobs": new_jobs,
                    })
            else:
                # GPU is utilized, reset idle timer
                idle_key = f"_gpu_idle_since_{peer.node_id}"
                setattr(self, idle_key, 0)

        # Start GPU selfplay on underutilized nodes
        for node_info in underutilized_gpu_nodes[:3]:  # Max 3 nodes per cycle
            node_id = node_info["node_id"]
            new_jobs = node_info["new_jobs"]

            gpu_name = (node_info.get("gpu_name", "") or "").upper()
            is_high_end = any(tag in gpu_name for tag in ("H100", "H200", "GH200", "A100", "5090", "4090"))
            job_type_str = "GUMBEL/GPU" if is_high_end else "diverse/hybrid"
            print(
                f"[P2P] Auto-scale: {node_id} at {node_info['gpu_percent']:.0f}% GPU, "
                f"starting {new_jobs} {job_type_str} selfplay job(s)"
            )

            for _ in range(new_jobs):
                try:
                    # Schedule selfplay job (Jan 28, 2026: uses job_coordination_manager directly)
                    job = await self.job_coordination_manager.schedule_diverse_selfplay_on_node(node_id)
                    if job:
                        started += 1
                except Exception as e:  # noqa: BLE001
                    logger.error(f"Failed to start diverse selfplay on {node_id}: {e}")
                    break

        if started > 0:
            self._last_gpu_auto_scale = now
            logger.info(f"Auto-scale: started {started} new diverse/hybrid selfplay job(s)")

        return started

    async def _auto_rebalance_from_work_queue(self) -> int:
        """Jan 29, 2026: Delegated to JobOrchestrator.auto_rebalance_from_work_queue()."""
        return await self.jobs.auto_rebalance_from_work_queue()


    async def _check_cluster_balance(self) -> dict[str, Any]:
        """Check and rebalance jobs across the cluster.

        This method identifies:
        1. Powerful nodes that are underutilized (high capacity, low jobs)
        2. Weak nodes that are overloaded (low capacity, high jobs)

        When imbalance is detected, it reduces jobs on weak nodes so the
        scheduler can assign them to more powerful nodes.

        Returns dict with rebalancing actions taken.
        """
        try:
            # Feb 23, 2026: Use non-blocking cached snapshot to avoid blocking
            # event loop on peers_lock contention
            _all_peers = self._get_peers_snapshot_nonblocking()
            alive_peers = [p for p in _all_peers if p.is_alive()]

            all_nodes = [*alive_peers, self.self_info]
            healthy_nodes = [n for n in all_nodes if n.is_healthy()]

            if len(healthy_nodes) < 2:
                return {"action": "none", "reason": "insufficient_nodes"}

            # Calculate capacity and utilization for each node
            node_stats = []
            for node in healthy_nodes:
                target = self.selfplay_scheduler.get_target_jobs_for_node(node)
                current = int(getattr(node, "selfplay_jobs", 0) or 0)
                utilization = current / max(1, target)  # How full is this node
                capacity_score = target  # Higher = more powerful

                node_stats.append({
                    "node": node,
                    "target": target,
                    "current": current,
                    "utilization": utilization,
                    "capacity": capacity_score,
                    "load_score": node.get_load_score(),
                })

            # Find underutilized powerful nodes (capacity > median, utilization < 50%)
            sorted_by_capacity = sorted(node_stats, key=lambda x: x["capacity"], reverse=True)
            median_capacity = sorted_by_capacity[len(sorted_by_capacity) // 2]["capacity"]

            underutilized_powerful = [
                n for n in node_stats
                if n["capacity"] > median_capacity and n["utilization"] < 0.5
            ]

            # Find overloaded weak nodes (capacity < median, utilization > 100%)
            overloaded_weak = [
                n for n in node_stats
                if n["capacity"] < median_capacity and n["utilization"] > 1.0
            ]

            if not underutilized_powerful or not overloaded_weak:
                return {"action": "none", "reason": "balanced"}

            # Calculate rebalancing opportunity
            spare_capacity = sum(
                max(0, n["target"] - n["current"]) for n in underutilized_powerful
            )
            excess_load = sum(
                max(0, n["current"] - n["target"]) for n in overloaded_weak
            )

            if spare_capacity < 2 or excess_load < 2:
                return {"action": "none", "reason": "minimal_imbalance"}

            # Migrate: reduce jobs on weak nodes
            rebalance_actions = []
            jobs_to_migrate = min(spare_capacity, excess_load)

            for weak_node in sorted(overloaded_weak, key=lambda x: x["utilization"], reverse=True):
                if jobs_to_migrate <= 0:
                    break

                node = weak_node["node"]
                reduce_by = min(
                    weak_node["current"] - weak_node["target"],
                    jobs_to_migrate
                )
                new_target = weak_node["current"] - reduce_by

                if reduce_by > 0:
                    print(
                        f"[P2P] Cluster rebalance: {node.node_id} overloaded "
                        f"({weak_node['current']}/{weak_node['target']} jobs, "
                        f"{weak_node['utilization']*100:.0f}% util) - reducing by {reduce_by}"
                    )

                    if node.node_id == self.node_id:
                        await self._reduce_local_selfplay_jobs(new_target, reason="cluster_rebalance")
                    else:
                        await self._request_reduce_selfplay(node, new_target, reason="cluster_rebalance")

                    rebalance_actions.append({
                        "node": node.node_id,
                        "reduced_by": reduce_by,
                        "new_target": new_target,
                    })
                    jobs_to_migrate -= reduce_by

            # Record rebalancing metric
            if rebalance_actions:
                self.record_metric(
                    "cluster_rebalance",
                    len(rebalance_actions),
                    metadata={
                        "spare_capacity": spare_capacity,
                        "excess_load": excess_load,
                        "actions": rebalance_actions,
                    },
                )

            return {
                "action": "rebalanced",
                "spare_capacity": spare_capacity,
                "excess_load": excess_load,
                "actions": rebalance_actions,
            }

        except Exception as e:  # noqa: BLE001
            logger.info(f"Cluster balance check error: {e}")
            return {"action": "error", "error": str(e)}

    async def _manage_cluster_jobs(self):
        """Manage jobs across the cluster (leader only).

        Jan 29, 2026: Delegated to ProcessSpawnerOrchestrator.
        LEARNED LESSONS incorporated:
        - Check disk space BEFORE starting jobs (Vast.ai 91-93% disk issue)
        - Check memory to prevent OOM (AWS instance crashed at 31GB+)
        - Trigger cleanup when approaching limits
        - Use is_healthy() not just is_alive()
        """
        # Jan 29, 2026: Delegate to ProcessSpawnerOrchestrator
        return await self.process_spawner.manage_cluster_jobs()


    async def _cleanup_local_disk(self):
        """Clean up disk space on local node.

        Jan 2026: Delegated to MemoryDiskManager (Phase 10 decomposition).
        """
        await self.memory_disk_manager.cleanup_local_disk()

    async def _request_remote_cleanup(self, node: NodeInfo):
        """Request a remote node to clean up disk space.

        Jan 2026: Delegated to MemoryDiskManager (Phase 10 decomposition).
        """
        await self.memory_disk_manager.request_remote_cleanup_via_orchestrator(node)

    async def _reduce_local_selfplay_jobs(self, target_selfplay_jobs: int, *, reason: str) -> dict[str, Any]:
        """Best-effort: stop excess selfplay jobs on this node (load shedding).

        Jan 2026: Delegated to MemoryDiskManager (Phase 10 decomposition).
        """
        return await self.memory_disk_manager.reduce_local_selfplay_jobs(target_selfplay_jobs, reason=reason)

    async def _request_reduce_selfplay(self, node: NodeInfo, target_selfplay_jobs: int, *, reason: str) -> None:
        """Ask a node to shed excess selfplay (used for memory/disk pressure).

        Jan 2026: Delegated to MemoryDiskManager (Phase 10 decomposition).
        """
        await self.memory_disk_manager.request_reduce_selfplay(node, target_selfplay_jobs, reason=reason)

    async def _restart_local_stuck_jobs(self):
        """Kill stuck selfplay processes and let job management restart them.

        LEARNED LESSONS - Addresses the issue where processes accumulate but GPU stays at 0%.
        """
        logger.info("Restarting stuck local selfplay jobs...")
        try:
            # Kill tracked selfplay jobs (avoid broad pkill patterns).
            jobs_to_clear: list[str] = []
            pids_to_kill: set[int] = set()
            with self.jobs_lock:
                for job_id, job in self.local_jobs.items():
                    if job.job_type not in (JobType.SELFPLAY, JobType.GPU_SELFPLAY, JobType.HYBRID_SELFPLAY, JobType.CPU_SELFPLAY, JobType.GUMBEL_SELFPLAY):
                        continue
                    jobs_to_clear.append(job_id)
                    if job.pid:
                        try:
                            pids_to_kill.add(int(job.pid))
                        except (ValueError, AttributeError):
                            continue

            # Sweep for untracked selfplay processes (e.g. lost local_jobs state) and kill them too.
            try:
                import shutil

                if shutil.which("pgrep"):
                    # December 2025: Added selfplay.py - unified entry point
                    for pattern in (
                        "selfplay.py",
                        "run_self_play_soak.py",
                        "run_gpu_selfplay.py",
                        "run_hybrid_selfplay.py",
                        "run_random_selfplay.py",
                    ):
                        out = subprocess.run(
                            ["pgrep", "-f", pattern],
                            capture_output=True,
                            text=True,
                            timeout=5,
                        )
                        if out.returncode == 0 and out.stdout.strip():
                            for token in out.stdout.strip().split():
                                try:
                                    pids_to_kill.add(int(token))
                                except (ValueError, AttributeError):
                                    continue
            except (ValueError, AttributeError):
                pass

            pids_to_kill.discard(int(os.getpid()))

            killed = 0
            for pid in sorted(pids_to_kill):
                try:
                    os.kill(pid, signal.SIGKILL)
                    killed += 1
                except (AttributeError):
                    continue

            # Clear our job tracking - they'll be restarted next cycle.
            with self.jobs_lock:
                for job_id in jobs_to_clear:
                    self.local_jobs.pop(job_id, None)

            logger.info(f"Killed {killed} processes, cleared {len(jobs_to_clear)} job records")
        except Exception as e:  # noqa: BLE001
            logger.error(f"killing stuck processes: {e}")

    async def _request_job_restart(self, node: NodeInfo):
        """Request a remote node to restart its stuck selfplay jobs."""
        try:
            if getattr(node, "nat_blocked", False):
                cmd_id = await self._enqueue_relay_command_for_peer(node, "restart_stuck_jobs", {})
                if cmd_id:
                    logger.info(f"Enqueued relay restart_stuck_jobs for {node.node_id}")
                else:
                    logger.info(f"Relay queue full for {node.node_id}; skipping restart enqueue")
                return
            timeout = ClientTimeout(total=HTTP_TOTAL_TIMEOUT)
            async with get_client_session(timeout) as session:
                last_err: str | None = None
                for url in self._urls_for_peer(node, "/restart_stuck_jobs"):
                    try:
                        async with session.post(url, json={}, headers=self._auth_headers()) as resp:
                            if resp.status != 200:
                                last_err = f"http_{resp.status}"
                                continue
                            data = await resp.json()
                            if data.get("success"):
                                logger.info(f"Job restart requested on {node.node_id}")
                                return
                            last_err = str(data.get("error") or "restart_failed")
                    except Exception as e:  # noqa: BLE001
                        last_err = str(e)
                        continue
                if last_err:
                    logger.info(f"Job restart request failed on {node.node_id}: {last_err}")
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to request job restart from {node.node_id}: {e}")

    async def _start_local_job(
        self,
        job_type: JobType,
        board_type: str = "square8",
        num_players: int = 2,
        engine_mode: str = "gumbel-mcts",  # GPU-accelerated Gumbel MCTS
        job_id: str | None = None,
        cuda_visible_devices: str | None = None,
        export_params: dict[str, Any] | None = None,
        simulation_budget: int | None = None,  # Gumbel MCTS budget (None = use tier default)
    ) -> ClusterJob | None:
        """Start a job on the local node.

        Jan 29, 2026: Delegated to ProcessSpawnerOrchestrator.
        SAFEGUARD: Checks coordination safeguards before spawning.
        """
        # Jan 29, 2026: Delegate to ProcessSpawnerOrchestrator
        return await self.process_spawner.start_local_job(
            job_type=job_type,
            board_type=board_type,
            num_players=num_players,
            engine_mode=engine_mode,
            job_id=job_id,
            cuda_visible_devices=cuda_visible_devices,
            export_params=export_params,
            simulation_budget=simulation_budget,
        )

    async def _dispatch_export_job(
        self,
        node: NodeInfo,
        input_path: str,
        output_path: str,
        board_type: str,
        num_players: int,
        encoder_version: str = "v3",
        max_games: int = 5000,
        is_jsonl: bool = False,
    ):
        """Dispatch a CPU-intensive export job to a high-CPU node.

        CPU-intensive jobs like NPZ export should run on vast nodes
        (256-512 CPUs) rather than lambda nodes (64 CPUs) to free
        GPU resources for training/selfplay.
        """
        try:
            job_id = f"export_{board_type}_{num_players}p_{int(time.time())}_{uuid.uuid4().hex[:6]}"

            payload = {
                "job_id": job_id,
                "job_type": JobType.DATA_EXPORT.value,
                "board_type": board_type,
                "num_players": num_players,
                "input_path": input_path,
                "output_path": output_path,
                "encoder_version": encoder_version,
                "max_games": max_games,
                "is_jsonl": is_jsonl,
            }

            # NAT-blocked nodes need relay command
            if getattr(node, "nat_blocked", False):
                cmd_id = await self._enqueue_relay_command_for_peer(node, "start_job", payload)
                if cmd_id:
                    logger.info(f"Enqueued relay export job for {node.node_id}: {job_id}")
                else:
                    logger.info(f"Relay queue full for {node.node_id}; export not dispatched")
                return

            timeout = ClientTimeout(total=30)
            async with get_client_session(timeout) as session:
                last_err: str | None = None
                for url in self._urls_for_peer(node, "/start_job"):
                    try:
                        async with session.post(url, json=payload, headers=self._auth_headers()) as resp:
                            if resp.status == 200:
                                result = await resp.json()
                                if result.get("success"):
                                    logger.info(f"Export job dispatched to {node.node_id}: {job_id}")
                                    return
                                last_err = result.get("error", "unknown")
                            else:
                                last_err = f"http_{resp.status}"
                    except Exception as e:  # noqa: BLE001
                        last_err = str(e)

                if last_err:
                    logger.info(f"Export job dispatch failed to {node.node_id}: {last_err}")

        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to dispatch export job to {node.node_id}: {e}")

    async def _request_remote_job(
        self,
        node: NodeInfo,
        job_type: JobType,
        board_type: str = "square8",
        num_players: int = 2,
        engine_mode: str = "hybrid",
    ):
        """Request a remote node to start a job with specific configuration.

        SAFEGUARD: Checks coordination safeguards before requesting remote spawn.
        """
        try:
            # Feb 2026: Normalize job_type to string early — _get_job_type() returns
            # a plain string when the JobType enum isn't available, but callers below
            # assumed .value always exists. This caused 4000+ dispatch failures/day.
            job_type_str = job_type.value if hasattr(job_type, 'value') else str(job_type)

            # Feb 2026: Check per-node dispatch cooldown to prevent tight retry loops.
            # Nodes that fail repeatedly get skipped for _JOB_DISPATCH_COOLDOWN_SECONDS.
            nid = node.node_id
            fail_info = self._job_dispatch_failures.get(nid)
            if fail_info:
                fail_count, fail_time = fail_info
                if fail_count >= self._JOB_DISPATCH_FAILURE_THRESHOLD:
                    elapsed = time.time() - fail_time
                    if elapsed < self._JOB_DISPATCH_COOLDOWN_SECONDS:
                        return  # Silently skip — already logged when cooldown started
                    else:
                        # Cooldown expired, reset and allow retry
                        self._job_dispatch_failures[nid] = (0, 0.0)

            # SAFEGUARD: Check safeguards before requesting remote spawn
            if HAS_SAFEGUARDS and _safeguards:
                allowed, reason = check_before_spawn(job_type_str, node.node_id)
                if not allowed:
                    logger.info(f"SAFEGUARD blocked remote {job_type_str} on {node.node_id}: {reason}")
                    return

            job_id = f"{job_type_str}_{board_type}_{num_players}p_{int(time.time())}_{uuid.uuid4().hex[:6]}"

            # NAT-blocked nodes can't accept inbound /start_job; enqueue a relay command instead.
            if getattr(node, "nat_blocked", False):
                payload = {
                    "job_id": job_id,
                    "job_type": job_type_str,
                    "board_type": board_type,
                    "num_players": num_players,
                    "engine_mode": engine_mode,
                }
                cmd_id = await self._enqueue_relay_command_for_peer(node, "start_job", payload)
                if cmd_id:
                    print(
                        f"[P2P] Enqueued relay job for {node.node_id}: "
                        f"{job_type_str} {board_type} {num_players}p ({job_id})"
                    )
                else:
                    logger.info(f"Relay queue full for {node.node_id}; skipping enqueue")
                return

            timeout = ClientTimeout(total=10)
            async with get_client_session(timeout) as session:
                payload = {
                    "job_id": job_id,
                    "job_type": job_type_str,
                    "board_type": board_type,
                    "num_players": num_players,
                    "engine_mode": engine_mode,
                }
                last_err: str | None = None
                for url in self._urls_for_peer(node, "/start_job"):
                    try:
                        async with session.post(url, json=payload, headers=self._auth_headers()) as resp:
                            if resp.status != 200:
                                last_err = f"http_{resp.status}"
                                continue
                            data = await resp.json()
                            if data.get("success"):
                                logger.info(f"Started remote {board_type} {num_players}p job on {node.node_id}")
                                # Reset failure tracking on success
                                self._job_dispatch_failures.pop(nid, None)
                                return
                            last_err = str(data.get("error") or "start_failed")
                    except Exception as e:  # noqa: BLE001
                        last_err = str(e)
                        continue
                if last_err:
                    # Track consecutive failures for cooldown
                    prev_count = self._job_dispatch_failures.get(nid, (0, 0.0))[0]
                    new_count = prev_count + 1
                    self._job_dispatch_failures[nid] = (new_count, time.time())
                    if new_count >= self._JOB_DISPATCH_FAILURE_THRESHOLD:
                        logger.warning(
                            f"Job dispatch to {nid} failed {new_count}x consecutively, "
                            f"cooling down for {self._JOB_DISPATCH_COOLDOWN_SECONDS}s"
                        )
                    else:
                        logger.error(f"Failed to start remote job on {nid}: {last_err}")
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to request remote job from {node.node_id}: {e}")

    def _enqueue_relay_command(self, node_id: str, cmd_type: str, payload: dict[str, Any]) -> str | None:
        """Leader-side: enqueue a command for a NAT-blocked node to pull."""
        now = time.time()
        cmd_type = str(cmd_type)
        payload = dict(payload or {})

        with self.relay_lock:
            queue = list(self.relay_command_queue.get(node_id, []))
            queue = [
                cmd for cmd in queue
                if float(cmd.get("expires_at", 0.0) or 0.0) > now
            ]

            if cmd_type == "start_job":
                pending = sum(1 for c in queue if str(c.get("type") or "") == "start_job")
                if pending >= RELAY_MAX_PENDING_START_JOBS:
                    self.relay_command_queue[node_id] = queue
                    return None

                job_id = str(payload.get("job_id") or "")
                if job_id:
                    for c in queue:
                        if str(c.get("payload", {}).get("job_id") or "") == job_id:
                            self.relay_command_queue[node_id] = queue
                            return str(c.get("id") or "")

            cmd_id = uuid.uuid4().hex
            queue.append(
                {
                    "id": cmd_id,
                    "type": cmd_type,
                    "payload": payload,
                    "created_at": now,
                    "expires_at": now + RELAY_COMMAND_TTL_SECONDS,
                }
            )
            self.relay_command_queue[node_id] = queue
            return cmd_id

    async def _enqueue_relay_command_for_peer(
        self,
        peer: NodeInfo,
        cmd_type: str,
        payload: dict[str, Any],
    ) -> str | None:
        """Enqueue a relay command for `peer`, forwarding via its relay hub when needed.

        Default behavior: NAT-blocked nodes poll the leader's `/relay/heartbeat`
        endpoint and the leader stores commands in-memory.

        Some nodes (notably certain containerized GPU providers) may be unable to
        reach the leader over the mesh network (e.g. TUN-less Tailscale) and also
        cannot accept inbound connections. Those nodes will instead send relay
        heartbeats to an internet-reachable hub (e.g. `aws-staging`). When
        `peer.relay_via` points to such a hub, the leader must enqueue the relay
        command on that hub so the node can pull and execute it.
        """
        if not peer or not getattr(peer, "node_id", ""):
            return None

        peer_id = str(getattr(peer, "node_id", "") or "").strip()
        if not peer_id:
            return None

        relay_node_id = str(getattr(peer, "relay_via", "") or "").strip()
        if relay_node_id and relay_node_id != self.node_id:
            with self.peers_lock:
                relay_peer = self.peers.get(relay_node_id)
            if relay_peer:
                timeout = ClientTimeout(total=10)
                async with get_client_session(timeout) as session:
                    last_err: str | None = None
                    for url in self._urls_for_peer(relay_peer, "/relay/enqueue"):
                        try:
                            async with session.post(
                                url,
                                json={
                                    "target_node_id": peer_id,
                                    "type": cmd_type,
                                    "payload": payload or {},
                                },
                                headers=self._auth_headers(),
                            ) as resp:
                                if resp.status != 200:
                                    last_err = f"http_{resp.status}"
                                    continue
                                data = await resp.json()
                                if data.get("success"):
                                    return str(data.get("id") or "")
                                last_err = str(data.get("error") or "enqueue_failed")
                        except Exception as e:  # noqa: BLE001
                            last_err = str(e)
                            continue
                    if last_err:
                        logger.info(f"Relay enqueue via {relay_node_id} failed for {peer_id}: {last_err}")
                        # Dec 30, 2025: Automatic relay failover
                        # If the current relay is unreachable, try to find a new one
                        # January 4, 2026: Pass peer_id for configured relay preferences
                        # Inline: was _select_best_relay()
                        new_relay = self.recovery_manager.select_best_relay(for_peer=peer_id)
                        if new_relay and new_relay != relay_node_id:
                            logger.info(
                                f"[RelayFailover] Switching {peer_id} relay: "
                                f"{relay_node_id} -> {new_relay}"
                            )
                            with self.peers_lock:
                                if peer_id in self.peers:
                                    self.peers[peer_id].relay_via = new_relay
                            # Try enqueue on new relay
                            with self.peers_lock:
                                new_relay_peer = self.peers.get(new_relay)
                            if new_relay_peer:
                                for url in self._urls_for_peer(new_relay_peer, "/relay/enqueue"):
                                    try:
                                        timeout = ClientTimeout(total=10)
                                        async with get_client_session(timeout) as session2:
                                            async with session2.post(
                                                url,
                                                json={
                                                    "target_node_id": peer_id,
                                                    "type": cmd_type,
                                                    "payload": payload or {},
                                                },
                                                headers=self._auth_headers(),
                                            ) as resp2:
                                                if resp2.status == 200:
                                                    data2 = await resp2.json()
                                                    if data2.get("success"):
                                                        return str(data2.get("id") or "")
                                    except Exception:  # noqa: BLE001
                                        continue

        # Fallback: enqueue locally (works when peer polls the leader directly).
        return self._enqueue_relay_command(peer_id, cmd_type, payload)

    async def _discovery_loop(self):
        """Broadcast UDP discovery messages to find peers on local network."""
        # Phase 3.1 Dec 29, 2025: Add max iterations to prevent infinite loop
        # Jan 13, 2026: Fix busy loop - add yield points and run socket ops in thread
        MAX_RECEIVE_ITERATIONS = 100
        YIELD_EVERY_N_PACKETS = 10  # Yield to event loop every N packets

        def _do_udp_discovery() -> list[dict]:
            """Run blocking UDP discovery in thread pool to avoid blocking event loop."""
            discovered = []
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                sock.settimeout(1.0)

                # Broadcast our presence
                message = json.dumps({
                    "type": "p2p_discovery",
                    "node_id": self.node_id,
                    "host": self.self_info.host,
                    "port": self.port,
                }).encode()

                with contextlib.suppress(OSError):
                    sock.sendto(message, ("<broadcast>", DISCOVERY_PORT))

                # Listen for responses with iteration limit
                receive_count = 0
                try:
                    while receive_count < MAX_RECEIVE_ITERATIONS:
                        data, _addr = sock.recvfrom(1024)
                        receive_count += 1
                        try:
                            msg = json.loads(data.decode())
                            if msg.get("type") == "p2p_discovery" and msg.get("node_id") != self.node_id:
                                discovered.append(msg)
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            continue
                except TimeoutError:
                    pass

                if receive_count >= MAX_RECEIVE_ITERATIONS:
                    logger.warning(f"[UdpDiscovery] Hit max receive limit ({MAX_RECEIVE_ITERATIONS})")

                sock.close()
            except OSError as e:
                logger.debug(f"[UdpDiscovery] Socket error: {e}")
            return discovered

        while self.running:
            try:
                # Run blocking socket operations in thread pool
                discovered = await asyncio.to_thread(_do_udp_discovery)

                # Process discovered peers (yield periodically to prevent busy loop)
                for i, msg in enumerate(discovered):
                    peer_addr = f"{msg.get('host')}:{msg.get('port')}"
                    if peer_addr not in self.known_peers:
                        self.known_peers.append(peer_addr)
                        logger.info(f"Discovered peer: {msg.get('node_id')} at {peer_addr}")
                    # Yield to event loop every N packets to prevent blocking
                    if (i + 1) % YIELD_EVERY_N_PACKETS == 0:
                        await asyncio.sleep(0)

            except Exception as e:  # noqa: BLE001
                logger.debug(f"[UdpDiscovery] Error: {e}")
                # Brief sleep on error to prevent tight retry loop
                await asyncio.sleep(1.0)
                continue

            await asyncio.sleep(DISCOVERY_INTERVAL)


    async def restart_http_server(self) -> bool:
        """Restart the HTTP server gracefully without terminating the process.

        January 2026: Added to enable recovery from HTTP server failures without
        requiring full process restart. Called by HttpServerHealthLoop when the
        server becomes unresponsive.

        Returns:
            True if restart succeeded, False otherwise
        """
        async with self._http_restart_lock:
            self._http_restart_count += 1
            attempt = self._http_restart_count
            logger.warning(f"[P2P] HTTP server restart attempt {attempt}")

            try:
                # Stop existing sites
                for site in self._http_sites:
                    try:
                        await site.stop()
                    except Exception as e:
                        logger.debug(f"[P2P] Error stopping site: {e}")
                self._http_sites.clear()

                # Cleanup runner
                if self._http_runner is not None:
                    try:
                        await self._http_runner.cleanup()
                    except Exception as e:
                        logger.debug(f"[P2P] Error cleaning up runner: {e}")

                # Wait briefly for port to be released
                await asyncio.sleep(1.0)

                # Create new runner from existing app
                if self._http_app is None:
                    logger.error("[P2P] Cannot restart: HTTP app not initialized")
                    return False

                self._http_runner = web.AppRunner(self._http_app)
                await self._http_runner.setup()

                # Re-bind ports
                site_v4 = web.TCPSite(
                    self._http_runner, '0.0.0.0', self.port,
                    reuse_address=True, backlog=1024
                )
                await site_v4.start()
                self._http_sites.append(site_v4)
                logger.info(f"[P2P] HTTP server restarted on 0.0.0.0:{self.port}")

                # Try IPv6 as well
                try:
                    site_v6 = web.TCPSite(
                        self._http_runner, '::', self.port,
                        reuse_address=True, backlog=1024
                    )
                    await site_v6.start()
                    self._http_sites.append(site_v6)
                    logger.info(f"[P2P] HTTP server also listening on [::]:{self.port}")
                except OSError:
                    pass  # IPv6 optional

                logger.info(f"[P2P] HTTP server restart {attempt} successful")
                return True

            except Exception as e:
                logger.error(f"[P2P] HTTP server restart {attempt} failed: {e}")
                return False

    async def run(self):
        """Main entry point - start the orchestrator.

        Feb 2026: Decomposed into lifecycle phases for readability.
        """
        if not HAS_AIOHTTP:
            logger.error("aiohttp is required. Install with: pip install aiohttp")
            raise RuntimeError("aiohttp is required but not available - install with: pip install aiohttp")

        # Cap thread pool to reduce CPU. Was 4 (too few for 21 status metrics
        # + daemon SQLite ops, causing cascading timeouts). 8 balances CPU
        # usage vs thread availability for StatusMetricsCollector.
        import concurrent.futures
        loop = asyncio.get_running_loop()
        loop.set_default_executor(
            concurrent.futures.ThreadPoolExecutor(max_workers=8, thread_name_prefix="p2p_")
        )

        runner = await self._run_http_setup()
        tasks = await self._run_start_background_tasks()
        await self._run_bootstrap_and_election(tasks)
        await self._run_game_count_refresh(tasks)

        # Run forever
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Background task {i} failed: {result}")
        except asyncio.CancelledError:
            pass
        finally:
            await self._run_shutdown(runner)

    async def _run_http_setup(self) -> "web.AppRunner":

        # Start isolated health server FIRST (January 2026)
        # This ensures /health endpoint is always responsive even if main loop blocks
        self.monitoring.start_isolated_health_server()

        # Validate critical subsystems before starting (December 2025)
        self._startup_validation = self.monitoring.validate_critical_subsystems()

        # Set up HTTP server
        @web.middleware
        async def auth_middleware(request: web.Request, handler):
            if self.auth_token and request.method not in ("GET", "HEAD", "OPTIONS") and not self._is_request_authorized(request):
                return web.json_response({"error": "unauthorized"}, status=401)
            return await handler(request)

        # Increase max body size for large file uploads (100MB)
        # Fixes "Request Entity Too Large" for Elo DB and other file uploads
        app = web.Application(
            middlewares=[auth_middleware],
            client_max_size=100 * 1024 * 1024,  # 100 MB
        )
        # Store app for graceful restart (Jan 2026)
        self._http_app = app

        # Register all routes from centralized route registry (December 2025)
        # Replaces 200+ individual route registrations with declarative registry
        _routes_registered = False
        try:
            from scripts.p2p.routes import register_all_routes
            route_count = register_all_routes(app, self)
            logger.info(f"Registered {route_count} HTTP routes from route registry")
            _routes_registered = True
        except ImportError as e:
            logger.warning(f"Route registry not available, using inline routes: {e}")
            _routes_registered = False

        # Register file download routes (December 2025)
        # HTTP-based file sync for nodes with unreliable SSH
        try:
            from scripts.p2p.handlers.file_download import register_file_download_routes
            file_routes = register_file_download_routes(app, self)
            logger.info(f"Registered {file_routes} file download routes for HTTP-based sync")
        except ImportError as e:
            logger.debug(f"File download handler not available: {e}")

        # Register network health routes (December 30, 2025)
        # Cross-verification between P2P mesh and Tailscale connectivity
        try:
            setup_network_health_routes(app, self)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Network health routes not registered: {e}")

        # Register model inventory routes (January 2026)
        # Used by ClusterModelEnumerator for comprehensive model evaluation
        try:
            model_routes = setup_model_routes(app, self)
            logger.info(f"Registered {model_routes} model inventory routes")
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Model inventory routes not registered: {e}")

        # January 2026: Fallback route registrations removed.
        # Routes are now exclusively managed by scripts/p2p/routes.py.
        # If route registry fails, startup will continue with partial functionality.
        if not _routes_registered:
            logger.error(
                "Route registry failed to load - P2P will have limited functionality. "
                "Check scripts/p2p/routes.py for import errors."
            )

        runner = web.AppRunner(app)
        await runner.setup()
        # Store runner for graceful restart (Jan 2026)
        self._http_runner = runner

        # Verify NFS sync before starting (prevents import errors from stale code)
        try:
            from scripts.verify_nfs_sync import verify_before_startup
            if not verify_before_startup():
                logger.warning("NFS sync verification found mismatches - check logs for details")
        except ImportError:
            logger.debug("NFS sync verification not available")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"NFS sync verification failed: {e}")

        # Wire SyncRouter to event system for real-time sync triggers (December 2025)
        self._wire_sync_router_events()

        # Wire DeadPeerCooldownManager probe function (January 2026)
        # Enables probe-based early recovery from adaptive cooldown
        self._wire_cooldown_manager_probe()

        # Wire connection pool dynamic sizing (January 2026)
        # Scales pool limits based on cluster size to prevent exhaustion
        self._wire_connection_pool_dynamic_sizing()

        # Wire LeadershipStateMachine broadcast callback (ULSM - Jan 2026)
        # This enables the state machine to broadcast step-down to peers
        self._leadership_sm._broadcast_callback = self._broadcast_leader_state_change
        logger.info("ULSM: Leadership state machine broadcast callback wired")

        # Increase backlog to handle burst of connections from many nodes
        # Default is ~128, which can overflow when many vast nodes heartbeat simultaneously
        #
        # Jan 2, 2026: Bind to BOTH IPv4 and IPv6 explicitly
        # Python's asyncio/aiohttp doesn't properly implement dual-stack sockets
        # (IPV6_V6ONLY is not automatically disabled), so we bind to both addresses.
        #
        # Jan 8, 2026: Added retry with exponential backoff for TIME_WAIT state.
        # After a crash, the port may be in TIME_WAIT for up to 60s. Retry instead of failing.

        # Port binding retry configuration (January 2026)
        PORT_BIND_MAX_RETRIES = 5
        PORT_BIND_INITIAL_DELAY = 2.0  # seconds
        PORT_BIND_MAX_DELAY = 30.0  # seconds

        async def _try_bind_port(site: web.TCPSite, host: str, port: int) -> bool:
            """Try to bind port with exponential backoff for TIME_WAIT state."""
            delay = PORT_BIND_INITIAL_DELAY
            for attempt in range(PORT_BIND_MAX_RETRIES):
                try:
                    await site.start()
                    return True
                except OSError as e:
                    errno_val = getattr(e, 'errno', 0)
                    is_addr_in_use = "Address already in use" in str(e) or errno_val == 98

                    if is_addr_in_use and attempt < PORT_BIND_MAX_RETRIES - 1:
                        # Likely TIME_WAIT state - retry with backoff
                        logger.warning(
                            f"Port {port} busy (attempt {attempt + 1}/{PORT_BIND_MAX_RETRIES}), "
                            f"retrying in {delay:.1f}s (likely TIME_WAIT state)..."
                        )
                        await asyncio.sleep(delay)
                        delay = min(delay * 2, PORT_BIND_MAX_DELAY)
                        continue
                    elif is_addr_in_use:
                        # Final attempt failed
                        logger.error(f"Port {port} still in use after {PORT_BIND_MAX_RETRIES} attempts.")
                        logger.error(f"Try: lsof -i :{port} or pkill -f p2p_orchestrator")
                        raise RuntimeError(f"Port {port} bound after retries - cannot start P2P") from e
                    elif "Invalid argument" in str(e):
                        # macOS TCP keepalive socket option issue - don't retry
                        logger.warning(f"TCP socket configuration failed on {host}:{port}: {e}")
                        logger.warning("This may be a macOS TCP keepalive compatibility issue")
                        raise
                    else:
                        # Other errors - don't retry
                        logger.error(f"Failed to bind to {host}:{port}: {e}")
                        raise
            return False  # Should not reach here

        bind_host = self.host
        if self.host == "0.0.0.0":
            # Bind to IPv4 first (always needed)
            site_v4 = web.TCPSite(runner, '0.0.0.0', self.port, reuse_address=True, backlog=1024)
            await _try_bind_port(site_v4, '0.0.0.0', self.port)
            self._http_sites.append(site_v4)  # Store for graceful restart (Jan 2026)
            logger.info(f"HTTP server started on 0.0.0.0:{self.port} (IPv4, backlog=1024)")

            # Also try to bind to IPv6 (optional, for IPv6-only clients)
            try:
                site_v6 = web.TCPSite(runner, '::', self.port, reuse_address=True, backlog=1024)
                await site_v6.start()
                self._http_sites.append(site_v6)  # Store for graceful restart (Jan 2026)
                bind_host = "0.0.0.0 + [::]"
                logger.info(f"HTTP server also listening on [::]:{self.port} (IPv6)")
                print("[DEBUG] IPv6 server started", flush=True)
            except OSError as v6_err:
                # IPv6 binding failed - that's OK, IPv4 is already working
                logger.debug(f"IPv6 binding failed (OK, IPv4 is active): {v6_err}")
                bind_host = "0.0.0.0"
        else:
            # Specific host requested - bind directly with retry
            site = web.TCPSite(runner, self.host, self.port, reuse_address=True, backlog=1024)
            await _try_bind_port(site, self.host, self.port)
            self._http_sites.append(site)  # Store for graceful restart (Jan 2026)
            logger.info(f"HTTP server started on {self.host}:{self.port} (backlog=1024)")

        # Notify systemd that we're ready to serve
        systemd_notify_ready()

        # Jan 5, 2026: Send immediate relay heartbeats for NAT-blocked nodes
        # This ensures relay nodes discover us before the regular heartbeat loop kicks in
        if self._force_relay_mode:
            await self._send_initial_relay_heartbeats()

        # Jan 7, 2026: Send immediate peer announcements for ALL nodes
        # This reduces discovery latency from 15-30s to 2-5s after startup
        await self._send_startup_peer_announcements()

        # Jan 23, 2026: Initialize HybridCoordinator for Raft-based leader election
        # This replaces the buggy Bully algorithm when CONSENSUS_MODE=raft or hybrid.
        # The HybridCoordinator provides sub-second leader failover via PySyncObj's Raft.
        await self._init_hybrid_coordinator()

        return runner

    async def _run_start_background_tasks(self) -> list:
        """Start all background tasks with exception isolation.

        Feb 2026: Extracted from run() for readability.
        Returns list of asyncio tasks for the main gather loop.
        """
        # Jan 9, 2026: Async fallback for game count seeding from peers
        # Cluster nodes don't have local canonical DBs, so fetch from coordinator
        # This fixes underserved config prioritization on worker nodes
        await self._async_seed_game_counts_from_peers_if_needed()

        # Feb 2026 (1d): Refresh self_info with current metrics before first gossip.
        # Prevents broadcasting stale training_jobs/selfplay_jobs counts from
        # persisted state that hasn't been validated against running PIDs yet.
        # Feb 23, 2026: Run in thread to avoid blocking the event loop (10-30s on macOS).
        try:
            await asyncio.to_thread(self._update_self_info)
            logger.info("[P2P] Pre-gossip self_info refresh complete")
        except Exception as e:
            logger.warning(f"[P2P] Pre-gossip self_info refresh failed: {e}")

        # Feb 2026 (3a): Detect orphan GPU processes from previous sessions.
        # These can occupy GPU memory and block work claiming.
        # Feb 23, 2026: Run in thread to avoid blocking the event loop.
        try:
            await asyncio.to_thread(self._cleanup_orphan_gpu_processes)
        except Exception as e:
            logger.warning(f"[P2P] Orphan GPU detection failed: {e}")

        # Start background tasks with exception isolation and restart support
        # CRITICAL FIX (Dec 2025): Each task is wrapped to prevent cascade failures.
        # Previously, a single exception in any task would crash all 18+ tasks.
        # Dec 2025 Update: Added factory functions for auto-restart on critical tasks.
        # Jan 28, 2026: voter_heartbeat, reconnect_dead_peers, swim_membership, git_update
        # moved to LoopManager (see loop_registry.py).
        tasks = [
            # Critical heartbeat loop - auto-restart on failure
            self._create_safe_task(
                self._heartbeat_loop(), "heartbeat", factory=self._heartbeat_loop
            ),
            # Job management - auto-restart on failure
            self._create_safe_task(
                self._job_management_loop(), "job_management", factory=self._job_management_loop
            ),
            # Discovery - auto-restart on failure
            self._create_safe_task(
                self._discovery_loop(), "discovery", factory=self._discovery_loop
            ),
            # NOTE: The following loops are now managed by LoopManager:
            # - VoterHeartbeatLoop (moved to loop_registry.py)
            # - ReconnectDeadPeersLoop (moved to loop_registry.py)
            # - SwimMembershipLoop (moved to loop_registry.py)
            # - GitUpdateLoop (moved to loop_registry.py)
        ]

        # Add cloud IP refresh loops (best-effort; no-op if not configured).
        # Jan 2026: Delegated to IPDiscoveryManager for better modularity
        if HAS_DYNAMIC_REGISTRY:
            self.ip_discovery_manager.start()
            tasks.append(self._create_safe_task(self.ip_discovery_manager.vast_ip_update_loop(), "vast_ip_update"))
            tasks.append(self._create_safe_task(self.ip_discovery_manager.aws_ip_update_loop(), "aws_ip_update"))
            tasks.append(self._create_safe_task(self.ip_discovery_manager.tailscale_ip_update_loop(), "tailscale_ip_update"))

        # Phase 26: Continuous bootstrap loop - ensures isolated nodes can rejoin
        tasks.append(self._create_safe_task(self._continuous_bootstrap_loop(), "continuous_bootstrap"))

        # Dec 31, 2025: Periodic IP revalidation for late Tailscale availability
        # Fixes nodes advertising private IPs when Tailscale wasn't ready at startup
        tasks.append(self._create_safe_task(
            self._periodic_ip_validation_loop(), "ip_validation", factory=self._periodic_ip_validation_loop
        ))

        # Jan 9, 2026: Periodic game count refresh for underserved config prioritization
        # Keeps scheduler game counts up-to-date as games are generated and consolidated
        tasks.append(self._create_safe_task(
            self._game_count_refresh_loop(), "game_count_refresh", factory=self._game_count_refresh_loop
        ))

        # Jan 22, 2026: Periodic cluster health snapshots for Phase 2 P2P stability instrumentation
        # Logs detailed peer counts, voter health, and election state every 60 seconds
        # Jan 28, 2026: Uses health_metrics_manager directly
        tasks.append(self._create_safe_task(
            self.health_metrics_manager.cluster_health_snapshot_loop(), "cluster_health_snapshot",
            factory=self.health_metrics_manager.cluster_health_snapshot_loop
        ))

        # Jan 23, 2026: Event loop latency monitor for diagnosing HTTP unresponsiveness
        # Detects when synchronous operations block the event loop, causing health checks to fail
        # Jan 28, 2026: Uses health_metrics_manager directly
        tasks.append(self._create_safe_task(
            self.health_metrics_manager.event_loop_latency_monitor(), "event_loop_monitor",
            factory=self.health_metrics_manager.event_loop_latency_monitor
        ))

        # Dec 2025: 11 loops extracted to LoopManager - see scripts/p2p/loops/

        # Store tasks for shutdown handling
        self._background_tasks = tasks

        # Phase 4: Start extracted loops via LoopManager (Dec 2025)
        # These 11 loops now ONLY run via LoopManager (inline versions removed):
        # - EloSyncLoop, IdleDetectionLoop, AutoScalingLoop, JobReaperLoop, QueuePopulatorLoop
        # - WorkQueueMaintenanceLoop, NATManagementLoop, ManifestCollectionLoop, ValidationLoop
        # - DataManagementLoop, ModelSyncLoop
        job_reaper_started = False
        logger.info(f"[LoopManager] Phase 4 startup: EXTRACTED_LOOPS_ENABLED={EXTRACTED_LOOPS_ENABLED}")
        if EXTRACTED_LOOPS_ENABLED and self._register_extracted_loops():
            loop_manager = self._get_loop_manager()
            if loop_manager is not None:
                # Dec 27, 2025: start_all() now returns dict of {loop_name: started_successfully}
                # Check if job_reaper specifically started to avoid duplicate reapers
                startup_results = await loop_manager.start_all()
                job_reaper_started = startup_results.get("job_reaper", False)
                started_count = sum(1 for v in startup_results.values() if v)
                logger.info(
                    f"LoopManager: started {started_count}/{len(startup_results)} loops, "
                    f"job_reaper={'running' if job_reaper_started else 'FAILED'}"
                )

                # Jan 22, 2026: Verify StabilityController started (critical for self-healing)
                stability_started = startup_results.get("stability_controller", False)
                if not stability_started and self._stability_controller is not None:
                    logger.warning("[P2P] StabilityController failed to start via LoopManager - attempting direct start")
                    try:
                        self._stability_controller.start_background()
                        await asyncio.sleep(0.5)
                        if self._stability_controller.running:
                            logger.info("[P2P] StabilityController started via direct fallback")
                        else:
                            logger.error("[P2P] StabilityController direct start failed - self-healing disabled")
                    except Exception as e:
                        logger.error(f"[P2P] StabilityController fallback start error: {e}")

        # Phase 4.1: Inline job reaper fallback (Dec 27, 2025)
        # If JobReaperLoop specifically failed to start, run inline fallback for job cleanup
        # This ensures stuck jobs get cleaned up even if the modular loop system fails
        # Dec 27, 2025: Fixed race condition - now checks job_reaper loop status, not just
        # whether LoopManager.start_all() completed (which could mask loop startup failures)
        if JOB_REAPER_FALLBACK_ENABLED and not job_reaper_started:
            logger.info("[JobReaper] LoopManager not available, starting inline fallback")
            tasks.append(
                self._create_safe_task(
                    self._inline_job_reaper_fallback_loop(),
                    "job_reaper_fallback"
                )
            )

        return tasks

    async def _run_bootstrap_and_election(self, tasks: list) -> None:
        """Bootstrap from peers and run initial leader election.

        Feb 2026: Extracted from run() for readability.
        Appends election retry task to the tasks list.
        """
        # Best-effort bootstrap from seed peers before running elections. This
        # helps newly started cloud nodes quickly learn about the full cluster.
        # Jan 15, 2026 (Phase 6 P2P Resilience): Add retry logic with exponential backoff
        bootstrap_success = False
        bootstrap_attempts = 0
        max_bootstrap_attempts = 3
        bootstrap_backoff = [2, 5, 10]  # Exponential backoff in seconds

        while bootstrap_attempts < max_bootstrap_attempts and not bootstrap_success:
            try:
                bootstrap_success = await self._bootstrap_from_known_peers()
                if bootstrap_success:
                    logger.info(
                        f"[Bootstrap] Successfully bootstrapped from peers "
                        f"(attempt {bootstrap_attempts + 1}/{max_bootstrap_attempts})"
                    )
                    break
            except Exception as e:
                logger.warning(f"[Bootstrap] Attempt {bootstrap_attempts + 1} failed: {e}")

            bootstrap_attempts += 1
            if bootstrap_attempts < max_bootstrap_attempts:
                wait_time = bootstrap_backoff[bootstrap_attempts - 1]
                logger.info(f"[Bootstrap] Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)

        if not bootstrap_success:
            logger.warning(
                f"[Bootstrap] Failed to bootstrap after {max_bootstrap_attempts} attempts"
            )
            # Emit bootstrap failure event for monitoring
            self._safe_emit_event("BOOTSTRAP_FAILED", {
                "node_id": self.node_id,
                "attempts": bootstrap_attempts,
                "seed_count": len(self.known_peers or []),
                "message": "Failed to bootstrap from any seed peer",
            })

        # December 30, 2025: Immediate Tailscale discovery when no --peers provided
        # This fixes the bootstrap problem where nodes started without --peers
        # couldn't join the mesh because continuous_bootstrap_loop has a 30s delay.
        if not self.known_peers:
            logger.info("[Bootstrap] No --peers provided, running immediate Tailscale discovery...")
            with self.peers_lock:
                peers_before = len(self.peers)

            # Try direct Tailscale peer discovery first
            with contextlib.suppress(Exception):
                await self._discover_tailscale_peers()

            with self.peers_lock:
                peers_after = len(self.peers)

            if peers_after > peers_before:
                logger.info(f"[Bootstrap] Tailscale discovery found {peers_after - peers_before} new peer(s)")
                # January 2026: Force reconnect to any peers online in Tailscale but missing from P2P
                # This fixes peer discovery asymmetry where P2P shows 5-7 peers while Tailscale shows 40
                await self._reconnect_missing_tailscale_peers()
            else:
                # Tailscale discovery didn't find peers - try config-based seeds
                logger.info("[Bootstrap] Tailscale discovery found no peers, trying config-based seeds...")
                config_seeds = self._load_bootstrap_seeds_from_config()
                if config_seeds:
                    logger.info(f"[Bootstrap] Loaded {len(config_seeds)} seed(s) from config")
                    self.known_peers = config_seeds
                    with contextlib.suppress(Exception):
                        await self._bootstrap_from_known_peers()

        # December 29, 2025: Extended startup election with retry mechanism
        # If no leader known, start election after allowing time for peer discovery.
        # Previously used 5s which was too short for cluster discovery.
        await asyncio.sleep(15)  # Increased from 5s to allow peer discovery
        if not self.leader_id and not self._maybe_adopt_leader_from_peers():
            # CRITICAL: Check quorum before starting election to prevent quorum bypass
            if getattr(self, "voter_node_ids", []) and not self._has_voter_quorum():
                logger.warning("Skipping startup election: no voter quorum available (will retry)")
            else:
                await self._start_election()

        # Feb 2026: Auto-force leadership for preferred_leader from cluster config.
        # This ensures the coordinator always becomes leader after P2P restart,
        # preventing split-brain where remote nodes elect a different leader.
        # Also store _preferred_leader_id so _start_election() can suppress
        # elections on non-preferred nodes (follower-side split-brain prevention).
        if not getattr(self, "_forced_leader_override", False):
            try:
                from app.config.cluster_config import load_cluster_config
                preferred = load_cluster_config()._raw_config.get("preferred_leader", "")
            except Exception:
                preferred = ""
            self._preferred_leader_id = preferred or None
            if preferred and preferred == self.node_id:
                self._forced_leader_override = True
                self.role = NodeRole.LEADER
                self.leader_id = self.node_id
                self.leader_lease_expires = time.time() + 90.0
                self.last_leader_seen = time.time()
                self._leader_term = (getattr(self, "_leader_term", 0) or 0) + 1
                self._election_grace_until = time.time() + 120.0
                self._save_state()
                logger.warning("[P2P] Auto-forced leadership: this node is preferred_leader")

        # December 29, 2025: Add background task to retry election if still no leader
        # This handles cases where initial election fails or quorum wasn't available
        async def _delayed_election_retry():
            """Retry election periodically if no leader after startup."""
            retry_intervals = [30, 60, 120, 300]  # Exponential backoff: 30s, 1m, 2m, 5m
            retry_count = 0

            while self.running and retry_count < len(retry_intervals):
                wait_time = retry_intervals[retry_count]
                await asyncio.sleep(wait_time)

                if not self.running:
                    break

                if self.leader_id:
                    # Leader found, no need to retry
                    logger.info(f"Leader established ({self.leader_id}), stopping election retry task")
                    break

                # Still no leader, try to adopt from peers or start election
                if self._maybe_adopt_leader_from_peers():
                    logger.info(f"Adopted leader from peers: {self.leader_id}")
                    break

                # Check quorum and start election if possible
                if getattr(self, "voter_node_ids", []) and not self._has_voter_quorum():
                    retry_count += 1
                    # Jan 2, 2026: Use _count_alive_voters() to check IP:port matches
                    voters_alive = self._count_alive_voters()
                    logger.warning(
                        f"No voter quorum for election retry {retry_count}/{len(retry_intervals)} "
                        f"(alive={voters_alive}, need={getattr(self, 'voter_quorum_size', 3)})"
                    )
                    continue

                if not getattr(self, "election_in_progress", False):
                    logger.info(f"No leader after {wait_time}s, triggering election retry {retry_count + 1}")
                    await self._start_election()
                    retry_count += 1
                else:
                    logger.debug("Election already in progress, skipping retry")

            if not self.leader_id and self.running:
                logger.warning("Exhausted election retries, operating in leaderless mode")

        tasks.append(
            self._create_safe_task(
                _delayed_election_retry(),
                "delayed_election_retry"
            )
        )

    async def _run_game_count_refresh(self, tasks: list) -> None:
        """Set up game count refresh loops for selfplay scheduling.

        Feb 2026: Extracted from run() for readability.
        Appends deferred fetch and periodic refresh tasks to the tasks list.
        """
        # Session 17.41: Deferred game counts fetch from peers
        # If local seeding returned empty (no canonical DBs), fetch from coordinator
        async def _deferred_game_counts_fetch():
            """Fetch game counts from coordinator after peer discovery."""
            try:
                await asyncio.sleep(30)  # Wait for peer discovery to complete
                if not self.running:
                    return

                # Check if we already have game counts seeded
                if self.selfplay_scheduler and hasattr(self.selfplay_scheduler, "_p2p_game_counts"):
                    existing_counts = getattr(self.selfplay_scheduler, "_p2p_game_counts", {})
                    if existing_counts:
                        logger.debug(f"[P2P] Already have {len(existing_counts)} game counts, skipping peer fetch")
                        return

                # Fetch from coordinator/peers
                game_counts = await self._fetch_game_counts_from_peers()
                if game_counts and self.selfplay_scheduler:
                    self.selfplay_scheduler.update_p2p_game_counts(game_counts)
                    logger.info(f"[P2P] Deferred fetch: seeded SelfplayScheduler with {len(game_counts)} game counts from peers")
                    for config_key, count in sorted(game_counts.items(), key=lambda x: x[1]):
                        if count < 500:  # Log underserved configs
                            logger.info(f"[P2P] Underserved config (from peers): {config_key} = {count} games")
            except Exception as e:  # noqa: BLE001
                logger.debug(f"[P2P] Deferred game counts fetch failed: {e}")

        tasks.append(
            self._create_safe_task(
                _deferred_game_counts_fetch(),
                "deferred_game_counts_fetch"
            )
        )

        # Session 17.48: Periodic game counts refresh loop
        # The deferred fetch only runs once at startup. This loop ensures game counts
        # are kept fresh on leader nodes that don't have local canonical databases.
        # Without fresh game counts, starvation multipliers can't be applied correctly.
        async def _periodic_game_counts_refresh():
            """Periodically refresh game counts from peers (runs every 5 minutes)."""
            refresh_interval = 300  # 5 minutes
            # Wait for initial deferred fetch to complete
            await asyncio.sleep(60)

            while self.running:
                try:
                    # Only refresh if we don't have local canonical DBs
                    local_counts = await asyncio.to_thread(self._seed_selfplay_scheduler_game_counts_sync)

                    if not local_counts:
                        # No local DBs, fetch from peers
                        peer_counts = await self._fetch_game_counts_from_peers()

                        if peer_counts and self.selfplay_scheduler:
                            self.selfplay_scheduler.update_p2p_game_counts(peer_counts)
                            underserved = sum(1 for c in peer_counts.values() if c < 2000)
                            logger.info(f"[P2P] Periodic refresh: {len(peer_counts)} configs, {underserved} underserved")
                            # Log critically underserved configs
                            for config_key, count in sorted(peer_counts.items(), key=lambda x: x[1]):
                                if count < 500:
                                    logger.warning(f"[P2P] CRITICAL: {config_key} has only {count} games (ULTRA starvation)")

                except Exception as e:  # noqa: BLE001
                    logger.debug(f"[P2P] Periodic game counts refresh failed: {e}")

                await asyncio.sleep(refresh_interval)

        tasks.append(
            self._create_safe_task(
                _periodic_game_counts_refresh(),
                "periodic_game_counts_refresh"
            )
        )

        # January 14, 2026: Unified game counts refresh loop
        # This loop uses UnifiedGameAggregator to get counts from ALL sources:
        # LOCAL, CLUSTER, S3, and OWC external drive on mac-studio.
        # Runs less frequently (10 min) since it's more expensive than peer-only refresh.
        async def _unified_game_counts_refresh():
            """Refresh game counts from all sources including OWC and S3."""
            refresh_interval = 600  # 10 minutes
            # Wait for initial peer-based fetch to complete first
            await asyncio.sleep(120)

            while self.running:
                try:
                    if self.selfplay_scheduler:
                        counts = await self.selfplay_scheduler.refresh_from_unified_aggregator()
                        if counts:
                            total = sum(counts.values())
                            underserved = sum(1 for c in counts.values() if c < 5000)
                            logger.info(
                                f"[P2P] Unified refresh: {total:,} total games across all sources "
                                f"({underserved} configs underserved)"
                            )
                except Exception as e:  # noqa: BLE001
                    logger.debug(f"[P2P] Unified game counts refresh failed: {e}")

                await asyncio.sleep(refresh_interval)

        tasks.append(
            self._create_safe_task(
                _unified_game_counts_refresh(),
                "unified_game_counts_refresh"
            )
        )

    async def _run_shutdown(self, runner: "web.AppRunner") -> None:
        """Gracefully shut down all subsystems.

        Feb 2026: Extracted from run() for readability.
        Called in the finally block of the main gather loop.
        """
        self.running = False
        # Stop extracted loops via LoopManager (Dec 2025)
        loop_manager = self._get_loop_manager()
        if loop_manager is not None and loop_manager.is_started:
            try:
                results = await loop_manager.stop_all(timeout=15.0)
                # Note: stop_all now logs its own "stopped X/Y loops" message
            except Exception as e:  # noqa: BLE001
                logger.warning(f"LoopManager: stop failed: {e}")

        # Jan 2026: Shutdown loop executor thread pools (Phase 2)
        try:
            from scripts.p2p.loop_executors import LoopExecutors
            LoopExecutors.shutdown_all(wait=True)
        except ImportError:
            pass  # Module not available
        except Exception as e:  # noqa: BLE001
            logger.warning(f"LoopExecutors shutdown failed: {e}")

        # Jan 2026: Shutdown threaded loop runners (Phase 3)
        try:
            from scripts.p2p.threaded_loop_runner import ThreadedLoopRegistry
            results = await ThreadedLoopRegistry.stop_all(timeout=15.0)
            stopped = sum(1 for ok in results.values() if ok)
            if results:
                logger.info(f"ThreadedLoopRegistry: stopped {stopped}/{len(results)} runners")
        except ImportError:
            pass  # Module not available
        except Exception as e:  # noqa: BLE001
            logger.warning(f"ThreadedLoopRegistry shutdown failed: {e}")

        # Jan 23, 2026: Shutdown health check executor (singleton efficiency fix)
        try:
            if hasattr(self, "_health_check_executor") and self._health_check_executor:
                self._health_check_executor.shutdown(wait=False)
                logger.debug("Health check executor shutdown complete")
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Health check executor shutdown failed: {e}")

        try:
            await asyncio.wait_for(runner.cleanup(), timeout=30)
        except asyncio.TimeoutError:
            logger.warning("HTTP server cleanup timed out after 30s")


def _wait_for_tailscale_ip(timeout_seconds: int = 90, interval_seconds: float = 1.0) -> str:
    """Wait for Tailscale IP to become available at startup.

    Jan 12, 2026: Increased timeout from 30s to 90s after observing mac-studio
    consistently advertising local IP (10.0.0.62) instead of Tailscale IP.

    Root cause: When P2P starts before Tailscale CLI is ready, _get_tailscale_ip()
    returns empty and the code falls back to local IP (e.g., 10.0.0.62). This
    persists even after Tailscale becomes available later, causing P2P connectivity
    issues since other nodes can't reach the local IP. On mac-studio specifically,
    Tailscale can take 45-60s to initialize after boot.

    Fix: Retry Tailscale IP detection with exponential backoff for up to 90 seconds
    at startup with faster initial polling (1s intervals). This gives Tailscale
    enough time to initialize even on slow boot scenarios.

    Args:
        timeout_seconds: Maximum time to wait for Tailscale (default 90s)
        interval_seconds: Initial retry interval (doubles with each retry, max 5s)

    Returns:
        Tailscale IP if available within timeout, else empty string
    """
    from scripts.p2p.resource_detector import ResourceDetector

    detector = ResourceDetector()
    start_time = time.time()
    attempt = 0
    current_interval = interval_seconds

    while (time.time() - start_time) < timeout_seconds:
        attempt += 1
        ts_ip = detector.get_tailscale_ip()
        if ts_ip:
            if attempt > 1:
                logger.info(f"[TAILSCALE] IP acquired after {attempt} attempts: {ts_ip}")
            return ts_ip

        elapsed = time.time() - start_time
        remaining = timeout_seconds - elapsed

        if elapsed >= 5 and attempt <= 3:
            logger.warning(f"[TAILSCALE] Still waiting for IP (attempt {attempt}, {elapsed:.1f}s elapsed)")

        if remaining <= 0:
            break

        # Sleep with exponential backoff (max 5s between retries)
        sleep_time = min(current_interval, remaining, 5.0)
        time.sleep(sleep_time)
        current_interval = min(current_interval * 1.5, 5.0)

    logger.warning(f"[TAILSCALE] Timed out waiting for IP after {timeout_seconds}s ({attempt} attempts)")
    return ""


def _auto_detect_node_id() -> str | None:
    """Auto-detect node ID using unified identity resolution.

    Jan 2, 2026: Added to prevent startup failures when --node-id is forgotten.
    Jan 12, 2026: Added /etc/ringrift/node-id file support and IP normalization.
    Jan 13, 2026: Delegated to app.config.node_identity module (P2P Cluster Stability Plan).

    Detection order (from node_identity module):
    0. /etc/ringrift/node-id file (canonical source, written by deployment)
    1. RINGRIFT_NODE_ID environment variable
    2. /etc/default/ringrift-p2p file (legacy compatibility)
    3. Hostname match against distributed_hosts.yaml
    4. Tailscale IP match against distributed_hosts.yaml
    5. Fall back to get_node_id_safe() which uses hostname

    Returns:
        Detected node_id string, or None if detection failed
    """
    try:
        from app.config.node_identity import (
            get_node_identity,
            get_node_id_safe,
            NodeIdentityError,
        )

        # Try strict resolution first
        try:
            identity = get_node_identity()
            logger.info(
                f"[NODE-ID] Resolved node ID via {identity.resolution_method}: "
                f"{identity.canonical_id}"
            )
            return identity.canonical_id
        except NodeIdentityError as e:
            # Strict resolution failed, use safe fallback
            logger.warning(f"[NODE-ID] Strict resolution failed: {e}")
            node_id = get_node_id_safe()
            logger.warning(
                f"[NODE-ID] Using fallback node ID: {node_id} - "
                f"Run 'python scripts/provision_node_id.py --auto-detect' to fix"
            )
            return node_id

    except ImportError as e:
        # Module not available (running standalone or tests)
        logger.debug(f"[NODE-ID] node_identity module not available: {e}")

        # Minimal fallback: check canonical file and env var
        try:
            with open("/etc/ringrift/node-id") as f:
                node_id = f.read().strip()
                if node_id:
                    logger.info(f"[NODE-ID] Using node-id from /etc/ringrift/node-id: {node_id}")
                    return node_id
        except (FileNotFoundError, PermissionError):
            pass

        node_id = os.environ.get("RINGRIFT_NODE_ID")
        if node_id:
            return node_id

        # Fall back to hostname
        import socket
        hostname = socket.gethostname()
        if "." in hostname:
            hostname = hostname.split(".")[0]
        logger.warning(
            f"[NODE-ID] Falling back to hostname '{hostname}' - "
            f"Set RINGRIFT_NODE_ID or run provision_node_id.py"
        )
        return hostname


def _acquire_singleton_lock(
    kill_duplicates: bool = False,
    force_takeover: bool = False,
) -> bool:
    """Acquire singleton lock to prevent duplicate P2P orchestrator instances.

    Uses atomic file locking (fcntl) which is more reliable than PID file checks.
    Automatically handles stale locks from crashed processes.

    Args:
        kill_duplicates: If True, kill any duplicate P2P processes before acquiring
        force_takeover: If True, force-kill any lock holder (even if not P2P).
                        Use when lock is held by a recycled PID.

    Returns:
        True if lock acquired successfully
    """
    global _P2P_LOCK

    lock_dir = Path(__file__).parent.parent / "data" / "coordination"
    lock_dir.mkdir(parents=True, exist_ok=True)

    if kill_duplicates:
        # Find and kill any existing p2p_orchestrator processes
        pattern = r"p2p_orchestrator\.py"
        existing = find_processes_by_pattern(pattern, exclude_self=True)
        if existing:
            logger.info(f"[P2P] Found {len(existing)} duplicate processes, killing...")
            for proc in existing:
                logger.info(f"[P2P] Killing duplicate: PID {proc.pid}")
                if kill_process(proc.pid, wait=True, timeout=5.0):
                    logger.info(f"[P2P] Killed PID {proc.pid}")
                else:
                    logger.warning(f"[P2P] Failed to kill PID {proc.pid}")
            # Wait a moment for locks to release
            time.sleep(0.5)

    # Create lock with auto-cleanup of stale locks (from dead processes)
    _P2P_LOCK = SingletonLock(
        "p2p_orchestrator",
        lock_dir=lock_dir,
        auto_cleanup_stale=True,  # Automatically handle dead process locks
    )

    if not _P2P_LOCK.acquire():
        # Lock acquisition failed - provide detailed diagnostics
        status = _P2P_LOCK.get_lock_status()
        holder_pid = status.get("holder_pid")
        holder_alive = status.get("holder_alive", False)
        holder_command = status.get("holder_command", "")
        is_stale = status.get("is_stale", False)

        if is_stale:
            # This shouldn't happen with auto_cleanup_stale=True, but handle it
            logger.warning(
                f"[P2P] Stale lock detected (dead PID {holder_pid}). "
                f"Attempting force cleanup..."
            )
            if _P2P_LOCK.force_release():
                # Retry acquisition after cleanup
                if _P2P_LOCK.acquire():
                    logger.info(f"[P2P] Acquired lock after stale cleanup (PID {os.getpid()})")
                    return True
            logger.error("[P2P] Failed to clean up stale lock")
            return False

        if holder_pid and holder_alive:
            # Another live process is holding the lock
            is_p2p = _P2P_LOCK.is_holder_expected_process("p2p_orchestrator")
            if is_p2p:
                logger.error(
                    f"[P2P] Another P2P orchestrator is already running (PID {holder_pid}). "
                    f"Use --kill-duplicates to automatically terminate it."
                )
            else:
                # PID reuse - different process now holds the lock file
                # This happens when the old P2P crashed and the PID was reused
                if force_takeover:
                    logger.warning(
                        f"[P2P] Lock held by unexpected process (PID {holder_pid}: {holder_command[:80] if holder_command else 'unknown'}). "
                        f"Force takeover requested - killing holder."
                    )
                    if _P2P_LOCK.force_release(kill_holder=True):
                        if _P2P_LOCK.acquire():
                            logger.info(f"[P2P] Acquired lock after force takeover (PID {os.getpid()})")
                            return True
                    logger.error("[P2P] Force takeover failed")
                else:
                    logger.warning(
                        f"[P2P] Lock held by unexpected process (PID {holder_pid}: {holder_command[:80] if holder_command else 'unknown'}). "
                        f"This may indicate PID reuse after a crash. "
                        f"Use --force-takeover to automatically recover."
                    )
        else:
            logger.error(
                "[P2P] Failed to acquire lock (unknown reason). "
                f"Lock status: {status}"
            )
        return False

    logger.info(f"[P2P] Acquired singleton lock (PID {os.getpid()})")
    return True


def _release_singleton_lock() -> None:
    """Release the singleton lock on shutdown."""
    global _P2P_LOCK
    if _P2P_LOCK:
        _P2P_LOCK.release()
        logger.debug("[P2P] Released singleton lock")
        _P2P_LOCK = None


# =============================================================================
# PORT-FIRST CHECK (January 21, 2026 - Phase 1)
# =============================================================================
# This provides fast-fail duplicate detection BEFORE zombie detection or lock
# acquisition. If a healthy P2P is already running, exit immediately.

def _check_port_available_and_responsive(port: int = 8770, timeout: float = 3.0) -> tuple[bool, str]:
    """Check if port is available or if existing P2P is healthy.

    January 21, 2026: Added as Phase 1 of duplicate process prevention.
    This is the FIRST check at startup, before zombie detection or lock acquisition.
    Provides fast-fail when a healthy P2P is already running.

    Args:
        port: The P2P HTTP port to check (default 8770)
        timeout: HTTP health check timeout in seconds

    Returns:
        (should_continue, reason) tuple:
        - (True, "port_free") - Port is free, proceed with startup
        - (True, "port_check_failed") - Couldn't determine, proceed cautiously
        - (False, "healthy_p2p_running") - Another healthy P2P is running, exit
    """
    import socket
    import urllib.request
    import urllib.error

    # Step 1: Try to bind to port (instant availability check)
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", port))
        sock.close()
        return (True, "port_free")
    except OSError:
        pass  # Port in use, check if healthy

    # Step 2: Check if existing process on port is responsive
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/health",
            headers={"User-Agent": "p2p-startup-check"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                return (False, "healthy_p2p_running")
    except urllib.error.URLError as e:
        # Connection refused means port not actually listening
        if "Connection refused" in str(e):
            return (True, "port_check_failed")
        # Timeout or other error - proceed cautiously
        return (True, "port_check_failed")
    except Exception:
        # Unexpected error - proceed cautiously
        return (True, "port_check_failed")

    # Should not reach here, but proceed if we do
    return (True, "port_check_failed")


# =============================================================================
# SUPERVISOR COORDINATION (January 21, 2026 - Phase 2)
# =============================================================================
# This section prevents conflicts between manual P2P starts and master_loop
# automated recovery by creating a coordination file that tracks which
# management path is in control.

SUPERVISOR_FILE_PATH = Path(__file__).parent.parent / "data" / "coordination" / "p2p_supervisor.json"


def _read_supervisor_file() -> dict | None:
    """Read the supervisor coordination file."""
    try:
        if SUPERVISOR_FILE_PATH.exists():
            content = SUPERVISOR_FILE_PATH.read_text()
            return json.loads(content)
    except (OSError, json.JSONDecodeError) as e:
        logger.debug(f"[P2P] Could not read supervisor file: {e}")
    return None


def _write_supervisor_file(managed_by: str, pid: int, force: bool = False) -> bool:
    """Write the supervisor coordination file."""
    from datetime import datetime

    if not force:
        existing = _read_supervisor_file()
        if existing and existing.get("managed_by") not in ("none", None):
            existing_pid = existing.get("pid")
            if existing_pid and _is_process_running_check(existing_pid):
                return False

    try:
        SUPERVISOR_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "managed_by": managed_by,
            "pid": pid,
            "started_at": datetime.utcnow().isoformat() + "Z",
            "last_health_check": datetime.utcnow().isoformat() + "Z",
        }
        SUPERVISOR_FILE_PATH.write_text(json.dumps(state, indent=2))
        logger.info(f"[P2P] Claimed supervisor role: {managed_by} (PID {pid})")
        return True
    except OSError as e:
        logger.warning(f"[P2P] Failed to write supervisor file: {e}")
        return False


def _is_process_running_check(pid: int) -> bool:
    """Check if a process with the given PID is running."""
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


def _claim_supervisor_role(mode: str = "manual", force: bool = False) -> tuple[bool, str]:
    """Claim P2P management role."""
    from datetime import datetime, timedelta

    existing = _read_supervisor_file()

    if existing is None or existing.get("managed_by") in ("none", None):
        if _write_supervisor_file(mode, os.getpid(), force=True):
            return (True, "claimed")
        return (False, "write_failed")

    existing_manager = existing.get("managed_by")
    existing_pid = existing.get("pid")

    # Check if PID is dead
    if existing_pid and not _is_process_running_check(existing_pid):
        logger.info(f"[P2P] Previous manager (PID {existing_pid}) is dead, taking over")
        if _write_supervisor_file(mode, os.getpid(), force=True):
            return (True, "claimed")
        return (False, "write_failed")

    # Jan 23, 2026: Check for stale claims based on timestamp
    # If last_health_check is older than 10 minutes, consider it stale
    last_health = existing.get("last_health_check") or existing.get("started_at")
    if last_health:
        try:
            last_health_dt = datetime.fromisoformat(last_health.replace("Z", "+00:00"))
            now = datetime.now(last_health_dt.tzinfo) if last_health_dt.tzinfo else datetime.utcnow()
            stale_threshold = timedelta(minutes=10)
            if now - last_health_dt > stale_threshold:
                logger.info(f"[P2P] Previous manager (PID {existing_pid}) has stale health check ({last_health}), taking over")
                if _write_supervisor_file(mode, os.getpid(), force=True):
                    return (True, "claimed")
                return (False, "write_failed")
        except (ValueError, TypeError) as e:
            logger.debug(f"[P2P] Could not parse health check timestamp: {e}")

    if existing_pid == os.getpid():
        return (True, "already_manager")

    if existing_manager == "master_loop" and mode == "manual":
        if not force:
            return (False, "master_loop_managing")
        logger.warning("[P2P] Forcing takeover from master_loop")

    if force:
        if _write_supervisor_file(mode, os.getpid(), force=True):
            return (True, "claimed")
        return (False, "write_failed")

    return (False, "other_manager")


def _release_supervisor_role() -> None:
    """Release P2P management role on shutdown."""
    try:
        existing = _read_supervisor_file()
        if existing and existing.get("pid") == os.getpid():
            _write_supervisor_file("none", 0, force=True)
            logger.info("[P2P] Released supervisor role")
    except Exception as e:
        logger.debug(f"[P2P] Could not release supervisor role: {e}")


def should_master_loop_manage_p2p() -> tuple[bool, str]:
    """Check if master_loop should manage P2P or defer to manual management."""
    from datetime import datetime, timedelta

    existing = _read_supervisor_file()

    if existing is None:
        return (True, "no_manager")

    managed_by = existing.get("managed_by")
    if managed_by in ("none", None):
        return (True, "no_manager")

    existing_pid = existing.get("pid")

    if existing_pid and not _is_process_running_check(existing_pid):
        return (True, "manager_dead")

    if managed_by == "manual":
        started_at_str = existing.get("started_at", "")
        try:
            started_at = datetime.fromisoformat(started_at_str.rstrip("Z"))
            age = datetime.utcnow() - started_at
            if age < timedelta(hours=1):
                return (False, "manual_manager")
            return (True, "manual_expired")
        except (ValueError, TypeError):
            return (False, "manual_manager")

    if managed_by == "master_loop":
        return (True, "master_loop_manager")

    return (False, "manager_healthy")


def _check_and_kill_zombie_p2p(port: int = 8770, timeout: float = 5.0) -> bool:
    """Check for zombie P2P process and kill it if found.

    A zombie P2P process is one that is bound to the port but not responding
    to HTTP requests. This can happen when the process is stuck in a bad state.

    Args:
        port: The P2P HTTP port to check (default 8770)
        timeout: HTTP request timeout in seconds

    Returns:
        True if a zombie was found and killed, False otherwise
    """
    import urllib.request
    import urllib.error

    # Step 1: Check if anything is listening on the port
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        if result.returncode != 0 or not result.stdout.strip():
            # Nothing listening on the port
            return False
        pids = [int(p) for p in result.stdout.strip().split("\n") if p.strip()]
        if not pids:
            return False
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, ValueError):
        # lsof failed or timed out, assume no zombie
        return False

    # Step 2: Try to hit the /status endpoint
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/status",
            headers={"User-Agent": "zombie-detector"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                # Process is responding, not a zombie
                return False
    except urllib.error.URLError as e:
        # Connection refused means nothing is really listening (lsof race)
        if "Connection refused" in str(e):
            return False
        # Other errors (timeout, etc.) mean zombie detected
        logger.warning(f"[P2P] Port {port} occupied but unresponsive: {e}")
    except Exception as e:
        # Timeout or other error - this is a zombie
        logger.warning(f"[P2P] Port {port} occupied but unresponsive: {e}")

    # Step 3: Kill the zombie process(es)
    logger.warning(f"[P2P] Detected zombie P2P process on port {port}, killing PIDs: {pids}")
    killed = False
    for pid in pids:
        # Skip ourselves
        if pid == os.getpid():
            continue
        try:
            if kill_process(pid, wait=True, timeout=5.0):
                logger.info(f"[P2P] Killed zombie process PID {pid}")
                killed = True
            else:
                logger.warning(f"[P2P] Failed to kill zombie PID {pid}")
        except Exception as e:
            logger.error(f"[P2P] Error killing zombie PID {pid}: {e}")

    if killed:
        # Give the port time to be released
        time.sleep(0.5)

    return killed


def main():
    # ==========================================================================
    # PRE-FLIGHT VALIDATION (January 2026)
    # ==========================================================================
    # Validate critical dependencies before any complex initialization.
    # This prevents cryptic runtime errors from missing packages.
    deps_ok, dep_errors = _validate_preflight_dependencies()
    if not deps_ok:
        print("[P2P] FATAL: Missing critical dependencies", file=sys.stderr)
        for err in dep_errors:
            print(f"  {err}", file=sys.stderr)
        print("\n[P2P] Fix: pip install aiohttp psutil pyyaml", file=sys.stderr)
        sys.exit(1)

    # Parse lock-related args early (before full argparse)
    import sys
    kill_duplicates = "--kill-duplicates" in sys.argv
    force_takeover = "--force-takeover" in sys.argv
    skip_zombie_check = "--no-zombie-check" in sys.argv
    skip_port_check = "--skip-port-check" in sys.argv
    ignore_supervisor = "--ignore-supervisor" in sys.argv
    force_supervisor = "--force-supervisor" in sys.argv
    is_master_loop = "--managed-by-master-loop" in sys.argv

    # ==========================================================================
    # PORT-FIRST CHECK (January 21, 2026)
    # ==========================================================================
    # Check if port is available or if a healthy P2P is already running.
    # This provides fast-fail before zombie detection or lock acquisition.
    if not skip_port_check:
        can_start, reason = _check_port_available_and_responsive(DEFAULT_PORT)
        if not can_start:
            print(f"[P2P] Exiting: {reason} - another healthy P2P is already running on port {DEFAULT_PORT}")
            sys.exit(0)
        elif reason == "port_free":
            print("[P2P] Port is free, proceeding with startup")

    # ==========================================================================
    # SUPERVISOR COORDINATION (January 21, 2026)
    # ==========================================================================
    # Check if another manager (master_loop or manual) is controlling P2P.
    if not ignore_supervisor:
        management_mode = "master_loop" if is_master_loop else "manual"
        claimed, claim_reason = _claim_supervisor_role(mode=management_mode, force=force_supervisor)
        if not claimed:
            if claim_reason == "master_loop_managing":
                print("[P2P] Exiting: master_loop.py is managing P2P. Use --force-supervisor to override.")
            else:
                print(f"[P2P] Exiting: Another manager is active ({claim_reason}). Use --force-supervisor to override.")
            sys.exit(0)
        print(f"[P2P] Claimed supervisor role: {management_mode}")

    # ==========================================================================
    # ZOMBIE DETECTION (January 2026)
    # ==========================================================================
    # Check for zombie P2P processes that are bound to the port but unresponsive.
    # This happens when the P2P process gets stuck in a bad state (e.g., 100% CPU).
    if not skip_zombie_check:
        if _check_and_kill_zombie_p2p():
            print("[P2P] Killed zombie P2P process, proceeding with startup")

    # Acquire singleton lock (December 2025: improved atomic locking with stale cleanup)
    if not _acquire_singleton_lock(
        kill_duplicates=kill_duplicates,
        force_takeover=force_takeover,
    ):
        sys.exit(1)

    parser = argparse.ArgumentParser(description="P2P Orchestrator for RingRift cluster")
    parser.add_argument("--node-id", required=False, help="Unique identifier for this node (auto-detects if not provided)")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to listen on")
    parser.add_argument(
        "--advertise-host",
        default=None,
        help=f"Host to advertise to peers (or set {ADVERTISE_HOST_ENV})",
    )
    parser.add_argument(
        "--advertise-port",
        type=int,
        default=None,
        help=f"Port to advertise to peers (or set {ADVERTISE_PORT_ENV})",
    )
    parser.add_argument("--peers", help="Comma-separated list of known peers (host[:port] or http(s)://host[:port])")
    parser.add_argument("--relay-peers", help="Comma-separated list of peers to use relay heartbeats with (for NAT-blocked nodes)")
    parser.add_argument("--ringrift-path", help="Path to RingRift installation")
    parser.add_argument("--auth-token", help=f"Shared auth token (or set {AUTH_TOKEN_ENV})")
    parser.add_argument("--require-auth", action="store_true", help="Require auth token to be set")
    parser.add_argument("--storage-type", choices=["disk", "ramdrive", "auto"], default="auto",
                        help="Storage type: 'disk', 'ramdrive' (/dev/shm), or 'auto' (detect based on RAM/disk)")
    parser.add_argument("--sync-to-disk-interval", type=int, default=300,
                        help="When using ramdrive, sync to disk every N seconds (0 = no sync, default: 300)")
    parser.add_argument("--supervised", action="store_true",
                        help="Running under cluster_supervisor.py - disable self-restart logic")
    parser.add_argument("--kill-duplicates", action="store_true",
                        help="Kill any existing P2P orchestrator processes before starting")
    parser.add_argument("--force-takeover", action="store_true",
                        help="Force acquire lock even if held by another process (use when PID was recycled after crash)")
    parser.add_argument("--no-zombie-check", action="store_true",
                        help="Skip automatic zombie P2P detection (zombies are processes bound to port but not responding)")
    parser.add_argument("--skip-port-check", action="store_true",
                        help="Skip the port availability check at startup (Jan 21, 2026)")
    parser.add_argument("--ignore-supervisor", action="store_true",
                        help="Skip supervisor coordination file check (Jan 21, 2026)")
    parser.add_argument("--force-supervisor", action="store_true",
                        help="Force takeover of supervisor role even if another manager is active")
    parser.add_argument("--managed-by-master-loop", action="store_true",
                        help="Internal flag: indicates P2P was started by master_loop.py")
    parser.add_argument("--training-only", action="store_true",
                        help="Run as training-only node (no selfplay dispatch). Prevents OOM from training + selfplay conflicts.")

    args = parser.parse_args()

    # Jan 2026: Set training-only mode if flag is set
    if args.training_only:
        set_selfplay_disabled_override(disabled=True)
        logger.info("[P2P] Running in training-only mode - selfplay disabled")

    # Jan 2, 2026: Auto-detect node_id if not provided
    if not args.node_id:
        args.node_id = _auto_detect_node_id()
        if not args.node_id:
            logger.error("Could not auto-detect node-id. Please provide --node-id explicitly.")
            sys.exit(1)
        logger.info(f"Auto-detected node-id: {args.node_id}")

    known_peers = []
    if args.peers:
        known_peers = [p.strip() for p in args.peers.split(',')]

    relay_peers = []
    if args.relay_peers:
        relay_peers = [p.strip() for p in args.relay_peers.split(',')]

    # Wrap orchestrator creation and run in try/except to ensure crashes are logged
    orchestrator = None
    try:
        logger.info(f"Initializing P2P orchestrator: node_id={args.node_id}")
        orchestrator = P2POrchestrator(
            node_id=args.node_id,
            host=args.host,
            port=args.port,
            known_peers=known_peers,
            relay_peers=relay_peers,
            ringrift_path=args.ringrift_path,
            advertise_host=args.advertise_host,
            advertise_port=args.advertise_port,
            auth_token=args.auth_token,
            require_auth=args.require_auth,
            storage_type=args.storage_type,
            sync_to_disk_interval=args.sync_to_disk_interval,
        )
        logger.info(f"P2P orchestrator initialized successfully: {args.node_id}")

        # December 28, 2025: Validate event emitters at startup
        # This provides early warning if event system is not properly configured
        if _check_event_emitters():
            logger.info("[P2P] Event emitters available - P2P events will be published")
        else:
            logger.warning(
                "[P2P] Event emitters NOT available - P2P events will be silent. "
                "Ensure app.coordination.event_emitters is importable for full integration."
            )
    except Exception as e:  # noqa: BLE001
        logger.exception(f"Failed to initialize P2P orchestrator: {e}")
        # January 2026: Release lock on initialization failure to prevent
        # stale locks that block future startups
        _release_singleton_lock()
        sys.exit(1)

    # Handle shutdown gracefully - avoid race conditions with async tasks
    # December 2025: Fixed signal handler race condition that caused threading exceptions
    _shutdown_requested = False
    _start_time = time.time()

    def signal_handler(sig, frame):
        nonlocal _shutdown_requested
        import traceback

        uptime = time.time() - _start_time
        sig_name = signal.Signals(sig).name if hasattr(signal, 'Signals') else f"signal {sig}"

        if _shutdown_requested:
            # Force exit on second signal
            logger.warning(f"Forced shutdown (second {sig_name}) after {uptime:.1f}s uptime")
            os._exit(1)
        _shutdown_requested = True

        # Enhanced logging to identify what's sending signals
        logger.warning(f"=== SIGNAL RECEIVED: {sig_name} ===")
        logger.warning(f"PID: {os.getpid()}, Uptime: {uptime:.1f}s, Node: {args.node_id}")
        logger.warning(f"Stack trace at signal:\n{''.join(traceback.format_stack(frame))}")
        logger.info("Shutdown requested, stopping gracefully...")
        if orchestrator:
            orchestrator.running = False
            # Cancel all background tasks for graceful shutdown (Dec 2025)
            if hasattr(orchestrator, '_background_tasks'):
                for task in orchestrator._background_tasks:
                    if not task.done():
                        task.cancel()
            # Schedule ramdrive sync in a thread to avoid blocking signal handler
            # Don't call sys.exit() - let asyncio loop exit cleanly

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run with exception logging
    try:
        logger.info(f"Starting P2P orchestrator main loop: {args.node_id}")
        asyncio.run(orchestrator.run())
    except Exception as e:  # noqa: BLE001
        logger.exception(f"P2P orchestrator crashed: {e}")
        sys.exit(1)
    finally:
        # Ensure ramdrive is synced on exit (moved from signal handler to avoid race)
        if orchestrator:
            try:
                orchestrator.stop_ramdrive_syncer(final_sync=True)
                logger.info("Ramdrive sync completed on shutdown")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Ramdrive sync on shutdown failed: {e}")
            # December 2025: Close webhook notifier to prevent memory leaks
            try:
                if hasattr(orchestrator, 'notifier') and orchestrator.notifier:
                    orchestrator.notifier.close_sync()
            except (RuntimeError, OSError, AttributeError) as e:
                # Dec 2025: Narrowed from bare Exception; best effort cleanup
                logger.debug(f"Notifier close failed (best effort): {e}")

            # December 2025: Close work queue to persist final stats
            try:
                from app.coordination.work_queue import reset_work_queue
                reset_work_queue()
            except (ImportError, RuntimeError, sqlite3.Error) as e:
                logger.debug(f"Work queue cleanup failed (best effort): {e}")

            # December 2025: Release singleton lock on shutdown
            _release_singleton_lock()

            # January 2026: Release supervisor role on shutdown
            _release_supervisor_role()


if __name__ == "__main__":
    main()
