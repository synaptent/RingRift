"""Startup infrastructure for P2P orchestrator.

This module contains all module-level functions, lazy singletons, imports, and
standalone utilities that were originally in p2p_orchestrator.py. These have ZERO
dependency on the P2POrchestrator class.

Extracted April 2026 as Target 1 of the P2P decomposition plan.

Note: The monkey-patches (_AutoClosingConnection, _load_env_local) remain in
p2p_orchestrator.py because they must execute at import time before any other
imports.
"""
from __future__ import annotations

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
from app.core.async_context import fire_and_forget

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
        config_path = Path(__file__).parent.parent.parent / "config" / "unified_loop.yaml"
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
# Note: __file__ is scripts/p2p/startup_infrastructure.py, so we need 3 levels up
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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
# Canonical definition is in scripts.p2p.entrypoint; re-exported here for compatibility.
from scripts.p2p.entrypoint import _P2P_LOCK  # noqa: F401


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
    HAS_HYBRID_TRANSPORT = True
except ImportError:
    HAS_HYBRID_TRANSPORT = False
    HybridTransport = None
    get_hybrid_transport = None
    diagnose_node_connectivity = None

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



# =============================================================================
# Standalone startup/entrypoint functions extracted to scripts/p2p/entrypoint.py
# (Target 5 of P2P decomposition). Re-exported here for backward compatibility.
# =============================================================================
from scripts.p2p.entrypoint import (  # noqa: F401, E402
    SUPERVISOR_FILE_PATH,
    _acquire_singleton_lock,
    _auto_detect_node_id,
    _check_and_kill_zombie_p2p,
    _check_port_available_and_responsive,
    _claim_supervisor_role,
    _is_process_running_check,
    _read_supervisor_file,
    _release_singleton_lock,
    _release_supervisor_role,
    _wait_for_tailscale_ip,
    _write_supervisor_file,
    should_master_loop_manage_p2p,
)

