"""Initialization phases mixin for the P2P orchestrator.

April 2026: Extracted from p2p_orchestrator.py (Part 2 target 11).
Contains the constructor phase helpers, peer snapshot helpers, and LoopManager
registration glue used during startup.
"""
from __future__ import annotations

from scripts.p2p.p2p_mixin_base import P2PMixinBase
from scripts.p2p.startup_infrastructure import *  # noqa: F401,F403
from scripts.p2p.startup_infrastructure import (  # noqa: F401
    _safeguards,
    _wait_for_tailscale_ip,
)


class InitializationPhasesMixin(P2PMixinBase):
    """Mixin for P2POrchestrator initialization phases."""

    MIXIN_TYPE = "initialization_phases"

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
            get_peers=self.get_peers_ro,  # Mar 2026: Lock-free snapshot
            get_peers_lock=None,  # Mar 2026: No lock needed with snapshot
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
        self._http_session_created_at: float = 0.0
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

        # Copy-on-write peers snapshot — updated after every peers dict mutation.
        # Readers use get_peers_ro() (dict) or get_peers_list_ro() (list) without locking.
        # CPython dict/list assignment is atomic (GIL), so concurrent reads are safe.
        self._peers_ro: dict = {}
        self._peers_list_ro: list = []
        # Legacy cached snapshot (kept for backward compatibility with external callers)
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

    def _publish_peers_snapshot(self) -> None:
        """Publish an immutable snapshot of peers after a mutation.

        MUST be called while holding peers_lock (or immediately after release).
        CPython dict/list assignment is atomic under GIL, so readers that call
        get_peers_ro() / get_peers_list_ro() never see a partially-updated state.

        Also syncs the PeerSnapshot object for consumers using get_snapshot().
        """
        self._peers_ro = dict(self.peers)
        self._peers_list_ro = list(self.peers.values())
        # Refresh the legacy cache used by _get_peers_snapshot_nonblocking
        self._peers_snapshot_cache = self._peers_list_ro
        self._peers_snapshot_cache_time = time.time()
        # Sync the PeerSnapshot object (used by status, elections, etc.)
        try:
            with self._peer_snapshot.bulk_update():
                self._peer_snapshot.clear()
                for node_id, info in self.peers.items():
                    self._peer_snapshot.update_peer(node_id, info)
        except Exception:  # noqa: BLE001
            pass  # PeerSnapshot sync is best-effort

    def get_peers_ro(self) -> dict:
        """Lock-free read-only snapshot of the peers dict.

        Returns a plain dict (not the live dict). Safe to iterate, read keys/values,
        and pass to functions without holding peers_lock. Values may be up to one
        mutation behind the live dict — this is acceptable for leader ops, health
        checks, status endpoints, and all read-only consumers.
        """
        return self._peers_ro

    def get_peers_list_ro(self) -> list:
        """Lock-free read-only snapshot of peers as a list of PeerInfo."""
        return self._peers_list_ro

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
            get_peers=self.get_peers_ro,
            get_self_info=lambda: self.self_info,
            peers_lock=None,  # Mar 2026: Lock-free via get_peers_ro snapshot
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
            send_notification=(
                lambda **kwargs: fire_and_forget(
                    self.notifier.send(**kwargs),
                    name=f"analytics_notification:{self.node_id}",
                )
                if hasattr(self, "notifier")
                else None
            ),
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
