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
from scripts.p2p.mixins.initialization_phases_mixin import InitializationPhasesMixin
from scripts.p2p.mixins.runtime_lifecycle_mixin import RuntimeLifecycleMixin

# argparse and signal no longer needed here — main() moved to scripts/p2p/entrypoint.py

if TYPE_CHECKING:
    from app.coordination.unified_queue_populator import UnifiedQueuePopulator as QueuePopulator
    from app.coordination.p2p_auto_deployer import P2PAutoDeployer
    from scripts.p2p.loops import LoopManager


class P2POrchestrator(
    RuntimeLifecycleMixin,     # Runtime lifecycle and HTTP startup/shutdown (Apr 2026 - Part 2)
    InitializationPhasesMixin,  # Constructor phase and peer snapshot helpers (Apr 2026 - Part 2)
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
                    config_key = str(job_type)
                    node_id = ""
                    if isinstance(job_info, dict):
                        node_id = str(job_info.get("node_id") or "")
                        config_key = str(job_info.get("config_key") or config_key)
                        if config_key == str(job_type):
                            board_type = job_info.get("board_type")
                            num_players = job_info.get("num_players")
                            if board_type and num_players is not None:
                                config_key = f"{board_type}_{num_players}p"

                    fire_and_forget(
                        self._emit_task_abandoned(
                            job_id=job_id,
                            config_key=config_key,
                            reason="reaped_by_job_reaper",
                            node_id=node_id,
                        ),
                        name=f"emit_task_abandoned:{job_id}",
                    )

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
            self._publish_peers_snapshot()
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

    # =========================================================================
    # Phase 15.1.1: Fence Token Helpers (December 29, 2025)
    # =========================================================================

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
            import time as _time

            timeout = aiohttp.ClientTimeout(total=30)
            self._http_session = aiohttp.ClientSession(
                timeout=timeout,
                headers=self._auth_headers(),
            )
            self._http_session_created_at = _time.time()
        return self._http_session

    @property
    def http_session_created_at(self) -> float:
        """Timestamp when the current HTTP session was created."""
        return getattr(self, "_http_session_created_at", 0.0)

    async def recreate_http_session(self) -> None:
        """Close the existing HTTP session and create a fresh one.

        March 2026: Called by HttpPoolMonitorLoop to prevent TIME_WAIT socket
        exhaustion during 7-day autonomous operation. After closing the old
        session, the next access to self.http_session will lazily create a new
        one via the property getter.
        """
        import time as _time

        old_session = getattr(self, "_http_session", None)
        if old_session is not None and not old_session.closed:
            try:
                await old_session.close()
                # Allow FIN/ACK handshake to complete
                await asyncio.sleep(0.25)
            except Exception as e:
                logger.debug(f"[P2P] Error closing old HTTP session: {e}")

        # Reset so the property creates a fresh session on next access
        self._http_session = None
        self._http_session_created_at = 0.0

        # Eagerly create the new session so callers don't hit a race
        _ = self.http_session

        logger.info(
            f"[P2P] HTTP session recreated at {_time.time():.0f}"
        )

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
            self._publish_peers_snapshot()

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
                    self._publish_peers_snapshot()
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

    def _reap_orphan_processes(self) -> int:
        """Kill orphan Python processes from previous P2P/master_loop runs.

        Mar 2026: When the P2P process restarts (via LaunchAgent KeepAlive or
        manual restart), child processes from the old run become orphans.
        These accumulate over time (148 zombies observed on mac-studio).

        This runs at startup and kills any Python processes that:
        1. Are children of PID 1 (reparented orphans)
        2. Match known RingRift process patterns (selfplay, train, gauntlet)
        3. Started before the current process

        Returns:
            Number of processes killed
        """
        import signal as _sig

        my_pid = os.getpid()
        my_start = os.path.getctime(f"/proc/{my_pid}") if os.path.exists(f"/proc/{my_pid}") else time.time()
        killed = 0

        try:
            import subprocess
            # Get all python processes with their PIDs and command lines
            result = subprocess.run(
                ["ps", "-eo", "pid,ppid,lstart,args"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                return 0

            ringrift_patterns = [
                "selfplay", "train", "gauntlet", "export_replay",
                "gpu_parallel_games", "game_gauntlet",
            ]

            for line in result.stdout.strip().split("\n")[1:]:  # Skip header
                parts = line.split()
                if len(parts) < 5:
                    continue
                try:
                    pid = int(parts[0])
                    ppid = int(parts[1])
                except ValueError:
                    continue

                if pid == my_pid:
                    continue

                cmd = " ".join(parts[4:]).lower()  # lstart takes 5 fields
                # Heuristic: skip lines where we can't find 'python' in the command
                # (ps output on macOS has variable-width lstart field)
                if "python" not in cmd:
                    continue

                # Only kill orphaned processes (ppid=1 means reparented)
                # and processes that match RingRift patterns
                is_orphan = ppid == 1
                is_ringrift = any(p in cmd for p in ringrift_patterns)

                if is_orphan and is_ringrift:
                    try:
                        os.kill(pid, _sig.SIGTERM)
                        killed += 1
                        logger.info(f"[OrphanReaper] Killed orphan PID {pid}: {cmd[:80]}")
                    except (OSError, ProcessLookupError):
                        pass

            if killed:
                logger.info(f"[OrphanReaper] Killed {killed} orphan processes at startup")
        except Exception as e:
            logger.debug(f"[OrphanReaper] Failed: {e}")

        return killed

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
            peers_ro = self.get_peers_ro()
            total_nodes = len(peers_ro) + 1
            gpu_nodes = len([p for p in peers_ro.values() if getattr(p, "has_gpu", False)])
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
            elif work_type == "hyperparam_sweep":
                # Mar 5, 2026: No handler exists for hyperparam_sweep — these items
                # were enqueued by the queue populator but never claimed because no
                # node reports this capability. Return True to mark them complete and
                # drain the 2 stuck items. New items are blocked by setting
                # max_pending_hyperparam_sweep=0 in unified_queue_populator.py.
                logger.info(f"[P2P] hyperparam_sweep work {work_id} completed (no-op: feature not implemented)")
                return True
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
                    fire_and_forget(
                        self._cleanup_local_disk(),
                        name=f"cleanup_local_disk:{self.node_id}",
                    )
                    ok = True
                elif cmd_type == "restart_stuck_jobs":
                    fire_and_forget(
                        self._restart_local_stuck_jobs(),
                        name=f"restart_stuck_jobs:{self.node_id}",
                    )
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
                        fire_and_forget(
                            self._run_local_canonical_selfplay(
                                job_id,
                                board_type,
                                num_players,
                                num_games,
                                seed,
                            ),
                            name=f"canonical_selfplay:{job_id}",
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
