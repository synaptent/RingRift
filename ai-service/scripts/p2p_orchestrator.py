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

# Increase file descriptor limit early — P2P opens many connections + SQLite DBs
import resource as _resource
_soft, _hard = _resource.getrlimit(_resource.RLIMIT_NOFILE)
_target = min(8192, _hard) if _hard != _resource.RLIM_INFINITY else 8192
if _soft < _target:
    _resource.setrlimit(_resource.RLIMIT_NOFILE, (_target, _hard))

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


class _AutoClosingConnection(_sqlite3_module.Connection):
    """sqlite3.Connection that closes on context manager exit.

    Preserve the sqlite3.Connection contract so downstream code and tests that
    perform isinstance checks or rely on connection attributes continue to work.
    """

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if exc_type is None:
                self.commit()
            else:
                self.rollback()
        finally:
            self.close()
        return False


def _patched_connect(*args, **kwargs):
    kwargs.setdefault("factory", _AutoClosingConnection)
    return _original_sqlite3_connect(*args, **kwargs)


_sqlite3_module.connect = _patched_connect
# ===========================================================================

# =============================================================================
# April 2026: Module-level imports, singletons, and standalone functions extracted
# to scripts/p2p/startup_infrastructure.py (Target 1 of P2P decomposition).
# All symbols are re-exported here for backward compatibility.
# =============================================================================
from scripts.p2p.startup_infrastructure import *  # noqa: F401,F403
from scripts.p2p.startup_infrastructure import (  # noqa: F401 — explicit re-export of underscore names
    _safeguards,
    _swim_manager,
    _swim_on_member_alive,
    _swim_on_member_failed,
    _unified_targets,
    _check_event_emitters,
    _validate_preflight_dependencies,
    _validate_p2p_dependencies,
    _wait_for_tailscale_ip,
    _auto_detect_node_id,
    _acquire_singleton_lock,
    _release_singleton_lock,
    _check_port_available_and_responsive,
    _read_supervisor_file,
    _write_supervisor_file,
    _is_process_running_check,
    _claim_supervisor_role,
    _release_supervisor_role,
    _check_and_kill_zombie_p2p,
    _P2P_LOCK,
    _is_selfplay_enabled_for_node,
    _is_training_enabled_for_node,
)
from scripts.p2p.mixins.autonomous_work_mixin import AutonomousWorkMixin
from scripts.p2p.mixins.game_count_mixin import GameCountMixin
from scripts.p2p.mixins.http_session_mixin import HttpSessionMixin
from scripts.p2p.mixins.initialization_phases_mixin import InitializationPhasesMixin
from scripts.p2p.mixins.peer_discovery_mixin import PeerDiscoveryMixin
from scripts.p2p.mixins.process_management_mixin import ProcessManagementMixin
from scripts.p2p.mixins.relay_command_execution_mixin import RelayCommandExecutionMixin
from scripts.p2p.mixins.runtime_lifecycle_mixin import RuntimeLifecycleMixin
from scripts.p2p.mixins.state_persistence_mixin import StatePersistenceMixin

# argparse and signal no longer needed here — main() moved to scripts/p2p/entrypoint.py

if TYPE_CHECKING:
    from app.coordination.unified_queue_populator import UnifiedQueuePopulator as QueuePopulator
    from app.coordination.p2p_auto_deployer import P2PAutoDeployer
    from scripts.p2p.loops import LoopManager


class P2POrchestrator(
    RuntimeLifecycleMixin,     # Runtime lifecycle and HTTP startup/shutdown (Apr 2026 - Part 2)
    InitializationPhasesMixin,  # Constructor phase and peer snapshot helpers (Apr 2026 - Part 2)
    StatePersistenceMixin,     # Persisted state, epoch, and metrics helpers (Apr 2026 - Part 3)
    PeerDiscoveryMixin,        # Peer discovery, partition, and voter prepopulation helpers (Apr 2026 - Part 3)
    ProcessManagementMixin,    # Process cleanup and subprocess helpers (Apr 2026 - Part 3)
    HttpSessionMixin,          # Shared HTTP session, auth, and leader proxy helpers (Apr 2026 - Part 3)
    GameCountMixin,            # Selfplay scheduler game-count refresh helpers (Apr 2026 - Part 3)
    AutonomousWorkMixin,       # Autonomous work discovery and predictive selfplay helpers (Apr 2026 - Part 3)
    RelayCommandExecutionMixin,  # Relay polling and stability action helpers (Apr 2026 - Part 3)
    StatusMonitoringMixin,     # Health/status/self-info methods (Apr 2026 - Phase 4)
    DataSyncMixin,             # Cluster data sync and dedup helpers (Apr 2026 - Phase 4)
    JobManagementMixin,        # Local/cluster job management helpers (Apr 2026 - Phase 4)
    CodeUpdateMixin,           # Git auto-update helpers (Apr 2026 - Phase 4)
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
    ElectionLogicMixin,        # Election, provisional leader, and lease logic (Apr 2026 - Target 2)
    HeartbeatLoopMixin,          # Heartbeat loop and bootstrap methods (Apr 2026 - Target 4)
    TrainingPipelineMixin,       # Training pipeline coordination (Apr 2026 - Target 3)
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
    - ElectionLogicMixin: Election, provisional leadership, and lease logic (Apr 2026 - Target 2)
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

    # Initialization and peer snapshot helpers are provided by InitializationPhasesMixin.


    # =========================================================================
    # JobReaperLoop callbacks - December 27, 2025
    # =========================================================================




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




    def _wire_feedback_loops(self) -> bool:
        """Wire curriculum feedback loops. Feb 2026: Delegates to event_wiring module."""
        from scripts.p2p.event_wiring import wire_feedback_loops
        return wire_feedback_loops(self)



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
            worker = self.get_peers_ro().get(worker_id)
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
            leader = self.get_peers_ro().get(self.leader_id)
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


    # =========================================================================
    # Phase 15.1.1: Fence Token Helpers (December 29, 2025)
    # =========================================================================

    # _parse_peer_address, _url_for_peer, _urls_for_peer provided by NetworkUtilsMixin












    # =========================================================================
    # Phase 27: Peer Cache and Reputation Tracking
    # Provided by PeerManagerMixin:
    # =========================================================================

    # =========================================================================
    # Phase 29: Cluster Epoch Persistence
    # Phase 1 Refactoring: Delegated to StateManager
    # =========================================================================






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


    # _is_tailscale_host provided by NetworkUtilsMixin



    # _enable_tailscale_priority, _disable_tailscale_priority

    # =========================================================================
    # Network Health Methods (December 30, 2025)
    # Required by NetworkHealthMixin for cross-verification of P2P vs Tailscale
    # =========================================================================




    # =========================================================================
    # Partition Read-Only Mode (Phase 2.4 - Dec 29, 2025)
    # =========================================================================




    # NOTE: _get_db_game_count_sync() inlined at call site (Jan 2026 Phase 2)











    async def _safe_startup_s3_push(
        self,
        push_fn: "Callable",
        models_dir: "Path | None",
    ) -> None:
        """Run stranded candidate S3 push in background without blocking startup.

        Mar 2026: Wraps push_stranded_candidates_to_s3() with error isolation
        and a short delay to let the HTTP server come up first.
        """
        try:
            # Small delay so we don't compete with other startup I/O
            await asyncio.sleep(10)
            results = await push_fn(models_dir)
            if results:
                pushed = sum(1 for v in results.values() if v)
                if pushed:
                    logger.info(
                        f"[StartupS3Push] Pushed {pushed} stranded candidate(s) to S3"
                    )
        except Exception as e:
            # Never crash startup over this
            logger.warning(f"[StartupS3Push] Background push failed: {e}")



    # ============================================
    # Phase 2: Distributed Data Sync Methods
    # ============================================


    # Dec 2025: Legacy manifest methods removed (162 LOC) - using SyncPlanner









    # Phase 2: P2P Rsync Coordination - using SyncPlanner





    # ============================================
    # NodeSelector Wrapper Methods REMOVED (Dec 2025)
    # All call sites now use self.node_selector.* directly
    # ============================================





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










    # See scripts/p2p/loops/maintenance_loops.py and scripts/p2p/loop_registry.py

    # ============================================
    # HTTP API Handlers
    # ============================================


    @with_request_timeout(30.0)




    # =============================================================================
    # April 2026: _run_improvement_loop extracted to TrainingPipelineMixin
    # (Target 3 of P2P decomposition).
    # =============================================================================


    # _check_and_trigger_training -> TrainingPipelineMixin (Apr 2026 - Target 3)

    # _check_local_training_fallback -> TrainingPipelineMixin (Apr 2026 - Target 3)

    # _check_improvement_cycles, _dispatch_improvement_training -> TrainingPipelineMixin (Apr 2026 - Target 3)

    # Includes: handle_training_start, handle_training_status, handle_training_progress, handle_training_update,
    #           handle_training_trigger, handle_training_trigger_decision, handle_training_trigger_configs, handle_nnue_start


    # _get_training_timeout -> TrainingPipelineMixin (Apr 2026 - Target 3)

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

    # _monitor_training_process -> TrainingPipelineMixin (Apr 2026 - Target 3)


    # _monitor_selfplay_process -> TrainingPipelineMixin (Apr 2026 - Target 3)


    # _check_cmaes_auto_tuning, get_pfsp_opponent, update_pfsp_stats -> TrainingPipelineMixin (Apr 2026 - Target 3)


    # _import_gpu_selfplay_to_canonical -> TrainingPipelineMixin (Apr 2026 - Target 3)

    # =========================================================================
    # See: scripts/p2p/handlers/improvement.py (Dec 28, 2025 - Phase 8)
    # =========================================================================

    # handle_improvement_training_complete and handle_improvement_evaluation_complete
    # moved to ImprovementHandlersMixin (Dec 28, 2025 - Phase 8)

    # _schedule_improvement_evaluation -> TrainingPipelineMixin (Apr 2026 - Target 3)
    # _run_ssh_improvement_eval -> TrainingPipelineMixin (Apr 2026 - Target 3)
    # _auto_deploy_model -> TrainingPipelineMixin (Apr 2026 - Target 3)

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









    # =========================================================================
    # PREDICTIVE SCALING HELPERS (January 2026 Sprint 6)
    # Support methods for PredictiveScalingLoop - proactive job spawning
    # =========================================================================




    # =========================================================================
    # Support methods for JobReassignmentLoop - orphaned job recovery (Sprint 6)
    # =========================================================================



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

    # _run_evaluation, _promote_model_if_better -> TrainingPipelineMixin (Apr 2026 - Target 3)

    # ============================================
    # Core Logic
    # ============================================




    # _send_heartbeat_to_peer, _send_heartbeat_via_ssh_fallback, _bootstrap_from_known_peers:
    # Moved to HeartbeatLoopMixin (Apr 2026 - Target 4)

    async def _continuous_bootstrap_loop(self) -> None:
        """Phase 26.3: Continuously attempt to join cluster when isolated.

        Jan 27, 2026: Phase 16A - Delegates to HeartbeatManager.
        """
        await self.heartbeat_manager.continuous_bootstrap_loop()

    # _bootstrap_from_multiple_seeds: Moved to HeartbeatLoopMixin (Apr 2026 - Target 4)

    # _load_bootstrap_seeds_from_config: Moved to HeartbeatLoopMixin (Apr 2026 - Target 4)

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

    # _send_initial_relay_heartbeats: Moved to HeartbeatLoopMixin (Apr 2026 - Target 4)

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


    # _send_startup_peer_announcements: Moved to HeartbeatLoopMixin (Apr 2026 - Target 4)


    # _heartbeat_loop: Moved to HeartbeatLoopMixin (Apr 2026 - Target 4)

    # See scripts/p2p/loops/network_loops.py and scripts/p2p/loop_registry.py

    # See scripts/p2p/loops/network_loops.py and scripts/p2p/loop_registry.py

    # _send_voter_heartbeat, _try_voter_alternative_endpoints, _discover_voter_peer,
    # _refresh_voter_mesh: Moved to HeartbeatLoopMixin (Apr 2026 - Target 4)

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

    # _endpoint_key, _endpoint_conflict_keys, _is_leader_eligible:
    # Moved to ElectionLogicMixin (Apr 2026 - Target 2)


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

    # _start_election(), _become_leader(), _check_probabilistic_leadership(),
    # _claim_provisional_leadership(), _check_provisional_promotion(),
    # _promote_provisional_to_leader(), _step_down_from_provisional(),
    # _request_election_from_voters(), _check_emergency_coordinator_fallback():
    # Moved to ElectionLogicMixin (Apr 2026 - Target 2)




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











    # NOTE: _get_peer_health_summary() inlined at call site (Jan 2026 Phase 2)


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
            peer_list = [
                {"node_id": p.node_id, "host": p.host, "port": getattr(p, "metrics_port", 9091)}
                for p in self.get_peers_list_ro()
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

    # _renew_leader_lease():
    # Moved to ElectionLogicMixin (Apr 2026 - Target 2)

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


    # Runtime lifecycle and HTTP restart helpers are provided by RuntimeLifecycleMixin.



# =============================================================================
# main() extracted to scripts/p2p/entrypoint.py (Target 5, P2P decomposition).
# Re-exported here for backward compatibility.
# =============================================================================
from scripts.p2p.entrypoint import main  # noqa: F401, E402


if __name__ == "__main__":
    main()
