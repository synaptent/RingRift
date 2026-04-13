"""Runtime lifecycle mixin for the P2P orchestrator.

April 2026: Extracted from p2p_orchestrator.py (Part 2 target 10).
Contains HTTP startup/restart, background task startup, bootstrap/election,
game-count refresh scheduling, and graceful shutdown orchestration.
"""
from __future__ import annotations

from scripts.p2p.p2p_mixin_base import P2PMixinBase
from scripts.p2p.startup_infrastructure import *  # noqa: F401,F403


class RuntimeLifecycleMixin(P2PMixinBase):
    """Mixin for P2POrchestrator runtime lifecycle methods."""

    MIXIN_TYPE = "runtime_lifecycle"

    async def _start_tcp_site_with_retry(
        self,
        runner: "web.AppRunner",
        host: str,
        port: int,
        *,
        reuse_address: bool = True,
        backlog: int = 1024,
        max_retries: int = 5,
        initial_delay: float = 2.0,
        max_delay: float = 30.0,
    ) -> "web.TCPSite":
        """Start a fresh TCP site with retry/backoff.

        ``aiohttp`` can partially register a ``TCPSite`` with its runner before
        surfacing an address-in-use error. Retrying ``start()`` on the same site
        instance then fails with "Site ... is already registered in runner", so
        each retry must create a fresh site object.
        """

        delay = initial_delay
        last_error: Exception | None = None

        for attempt in range(max_retries):
            site = web.TCPSite(runner, host, port, reuse_address=reuse_address, backlog=backlog)
            try:
                await site.start()
                return site
            except OSError as exc:
                last_error = exc
                errno_val = getattr(exc, "errno", 0)
                is_addr_in_use = "Address already in use" in str(exc) or errno_val == 98

                try:
                    await site.stop()
                except Exception:
                    pass

                if is_addr_in_use and attempt < max_retries - 1:
                    logger.warning(
                        f"Port {port} busy (attempt {attempt + 1}/{max_retries}), "
                        f"retrying in {delay:.1f}s (likely TIME_WAIT state)..."
                    )
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, max_delay)
                    continue
                if is_addr_in_use:
                    logger.error(f"Port {port} still in use after {max_retries} attempts.")
                    logger.error(f"Try: lsof -i :{port} or pkill -f p2p_orchestrator")
                    raise RuntimeError(f"Port {port} bound after retries - cannot start P2P") from exc
                if "Invalid argument" in str(exc):
                    logger.warning(f"TCP socket configuration failed on {host}:{port}: {exc}")
                    logger.warning("This may be a macOS TCP keepalive compatibility issue")
                else:
                    logger.error(f"Failed to bind to {host}:{port}: {exc}")
                raise

        raise RuntimeError(f"Failed to bind {host}:{port}") from last_error

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
                site_v4 = await self._start_tcp_site_with_retry(
                    self._http_runner,
                    "0.0.0.0",
                    self.port,
                    reuse_address=True,
                    backlog=1024,
                )
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

        # Size the default thread pool for asyncio.to_thread() callers.
        # History: 4 -> 8 (status metrics timeouts) -> 24 (queue_populator starvation).
        # With 100+ asyncio.to_thread() callers across loops/handlers/managers,
        # 8 workers causes cascading timeouts when multiple loops run concurrently.
        # 24 workers on 28-core mac-studio keeps CPU usage reasonable while
        # preventing thread pool starvation that blocks queue_populator.
        import concurrent.futures
        loop = asyncio.get_running_loop()
        loop.set_default_executor(
            concurrent.futures.ThreadPoolExecutor(max_workers=24, thread_name_prefix="p2p_")
        )

        # Mar 2026: Reap orphan processes from previous P2P/master_loop runs.
        # When LaunchAgent restarts P2P, old child processes (selfplay, training,
        # gauntlet) become orphans. Kill them on startup to prevent accumulation.
        await asyncio.to_thread(self._reap_orphan_processes)

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

        bind_host = self.host
        if self.host == "0.0.0.0":
            # Bind to IPv4 first (always needed)
            site_v4 = await self._start_tcp_site_with_retry(
                runner,
                "0.0.0.0",
                self.port,
                reuse_address=True,
                backlog=1024,
            )
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
            site = await self._start_tcp_site_with_retry(
                runner,
                self.host,
                self.port,
                reuse_address=True,
                backlog=1024,
            )
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

        # Mar 2026: Push any stranded candidate models to S3.
        # If P2P restarted between training completion and S3 push, candidate
        # models are stranded locally. Fire-and-forget so it doesn't block startup.
        try:
            from scripts.p2p.work_executors.training_executor import (
                push_stranded_candidates_to_s3,
            )
            models_dir = Path(self.ringrift_path) / "models" if self.ringrift_path else None
            self._startup_s3_push_task = fire_and_forget(
                self._safe_startup_s3_push(push_stranded_candidates_to_s3, models_dir),
                name="startup_s3_push",
            )
        except ImportError as e:
            logger.debug(f"[P2P] Startup S3 push not available: {e}")
        except Exception as e:
            logger.warning(f"[P2P] Failed to schedule startup S3 push: {e}")

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
            peers_before = len(self.get_peers_ro())

            # Try direct Tailscale peer discovery first
            with contextlib.suppress(Exception):
                await self._discover_tailscale_peers()

            peers_after = len(self.get_peers_ro())

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
