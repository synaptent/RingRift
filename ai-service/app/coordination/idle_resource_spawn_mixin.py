"""Spawn and job-distribution helpers for IdleResourceDaemon."""

from __future__ import annotations

import asyncio
import logging
import time

from app.coordination.event_utils import parse_config_key
from app.coordination.idle_resource_shared import (
    ClusterNode,
    HAS_BACKPRESSURE,
    HAS_CIRCUIT_BREAKER,
    HAS_INCOMPATIBILITY_EVENTS,
    HAS_JOB_SCHEDULER,
    HAS_SSH_FALLBACK,
    HAS_STALL_DETECTION,
    NodeStatus,
    SSHExecutor,
    ScheduledJob,
    JobPriority,
    emit_node_incompatible_with_workload,
    get_backpressure_monitor,
    get_configured_hosts,
    get_operation_breaker,
    get_scheduler,
    get_stall_detector,
)

logger = logging.getLogger(__name__)


class IdleResourceSpawnMixin:
    """Extracted helpers for IdleResourceDaemon."""

    async def _check_and_spawn(self) -> None:
        """Check for idle nodes and spawn selfplay jobs.

        December 2025 - Phase 2B.2: Dynamically scales spawning based on
        idle node count instead of fixed max of 4.

        December 2025 - Phase 2C.4: Integrates with SelfplayScheduler for
        priority-based config selection.
        """
        try:
            # December 2025: Broadcast local idle state to cluster
            await self._broadcast_local_state()

            # Update SelfplayScheduler priorities before spawning
            await self._update_scheduler_priorities()

            # Phase 21.5: Refresh backpressure signal for accurate spawn decisions
            await self._refresh_backpressure_signal()

            # December 2025: Enforce process limits before spawning
            # This actively kills excess processes on nodes with runaway counts
            await self._enforce_process_limits()

            # Get cluster status
            nodes = await self._get_cluster_nodes()

            if not nodes:
                logger.debug("No cluster nodes found")
                return

            # Get queue depth for scaling decisions
            queue_depth = await self._get_queue_depth()

            # Get dynamic max spawns based on current idle capacity
            max_spawns = self._get_dynamic_max_spawns()

            # Collect nodes that need spawning
            spawn_candidates = [
                node for node in nodes
                if self._should_spawn(node, queue_depth)
            ]

            if not spawn_candidates:
                return

            # Emit IDLE_RESOURCE_DETECTED events for each candidate (Dec 2025 Phase 2)
            # This allows SelfplayOrchestrator and other components to react
            await self._emit_idle_resource_events(spawn_candidates)

            # Log scaling decision
            logger.info(
                f"[IdleResourceDaemon] Spawn check: {len(spawn_candidates)} candidates, "
                f"max_spawns={max_spawns} (dynamic), queue_depth={queue_depth}"
            )

            # Spawn up to max_spawns jobs concurrently
            spawn_tasks = []
            for node in spawn_candidates[:max_spawns]:
                spawn_tasks.append(self._spawn_selfplay(node))

            if spawn_tasks:
                results = await asyncio.gather(*spawn_tasks, return_exceptions=True)
                successful = sum(1 for r in results if r is True)
                logger.info(
                    f"[IdleResourceDaemon] Spawned {successful}/{len(spawn_tasks)} jobs"
                )

        except Exception as e:
            logger.warning(f"Check and spawn error: {e}")

    async def _get_ssh_fallback_nodes(self, exclude: set[str]) -> list[NodeStatus]:
        """Get nodes via SSH for hosts not discovered via P2P.

        Dec 2025: Discovers nodes from distributed_hosts.yaml that aren't
        in the P2P cluster, checks their GPU status via SSH.

        Args:
            exclude: Set of node IDs/hosts to skip (already discovered via P2P).

        Returns:
            List of NodeStatus for SSH-discovered nodes with GPUs.
        """
        nodes: list[NodeStatus] = []

        if not HAS_SSH_FALLBACK or get_configured_hosts is None:
            return nodes

        try:
            configured_hosts = get_configured_hosts()
        except Exception as e:
            logger.debug(f"[IdleResourceDaemon] Failed to load configured hosts: {e}")
            return nodes

        # Filter to active hosts with GPUs that aren't already discovered
        candidates = [
            (name, host) for name, host in configured_hosts.items()
            if host.is_active
            and host.gpu  # Has GPU configured
            and name not in exclude
            and (host.best_ip is None or host.best_ip not in exclude)
        ]

        if not candidates:
            return nodes

        logger.debug(
            f"[IdleResourceDaemon] SSH fallback: checking {len(candidates)} nodes "
            f"not in P2P: {[n for n, _ in candidates[:5]]}..."
        )

        # Check nodes concurrently (limit concurrency to avoid overwhelming)
        semaphore = asyncio.Semaphore(5)

        async def check_node(name: str, host: ClusterNode) -> NodeStatus | None:
            async with semaphore:
                return await self._check_node_via_ssh(name, host)

        tasks = [check_node(name, host) for name, host in candidates]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, NodeStatus):
                nodes.append(result)
                self._update_node_state(result)
            elif isinstance(result, Exception):
                logger.debug(f"[IdleResourceDaemon] SSH check failed: {result}")

        if nodes:
            logger.info(
                f"[IdleResourceDaemon] SSH fallback discovered {len(nodes)} nodes: "
                f"{[n.node_id for n in nodes]}"
            )

        return nodes

    async def _check_node_via_ssh(
        self, name: str, host: ClusterNode
    ) -> NodeStatus | None:
        """Check a single node's GPU status via SSH.

        Args:
            name: Node name from config.
            host: ClusterNode with SSH connection info.

        Returns:
            NodeStatus if node is reachable and has GPU, None otherwise.
        """
        if SSHExecutor is None or host.best_ip is None:
            return None

        try:
            executor = SSHExecutor(
                host=host.best_ip,
                user=host.ssh_user,
                port=host.ssh_port,
                key_path=host.ssh_key,
                connect_timeout=5,
                max_retries=1,
            )

            # Quick GPU check via nvidia-smi
            result = await executor.run(
                "nvidia-smi --query-gpu=utilization.gpu,memory.total,memory.used "
                "--format=csv,noheader,nounits 2>/dev/null || echo 'no-gpu'",
                timeout=10,
            )

            if not result.success or "no-gpu" in result.stdout:
                return None

            # Parse nvidia-smi output: "util%, mem_total, mem_used"
            # Example: "5, 24576, 1234"
            lines = result.stdout.strip().split('\n')
            if not lines or not lines[0]:
                return None

            parts = lines[0].split(',')
            if len(parts) < 3:
                return None

            try:
                gpu_util = float(parts[0].strip())
                mem_total_mb = float(parts[1].strip())
                mem_used_mb = float(parts[2].strip())
            except ValueError:
                return None

            return NodeStatus(
                node_id=name,
                host=host.best_ip,
                gpu_utilization=gpu_util,
                gpu_memory_total_gb=mem_total_mb / 1024.0,
                gpu_memory_used_gb=mem_used_mb / 1024.0,
                last_seen=time.time(),
                active_jobs=0,  # Unknown via SSH
                provider=self._detect_provider(name),
            )

        except Exception as e:
            logger.debug(f"[IdleResourceDaemon] SSH check failed for {name}: {e}")
            return None

    async def _enforce_process_limits(self) -> None:
        """Kill excess selfplay processes on nodes exceeding limits.

        December 2025: Actively kills runaway selfplay processes to prevent
        resource exhaustion. Uses SSH fallback when P2P job tracking isn't accurate.

        Called before spawn decisions to maintain healthy process counts.
        """
        if not HAS_SSH_FALLBACK or SSHExecutor is None or get_configured_hosts is None:
            return

        max_per_node = self.config.max_selfplay_processes_per_node

        try:
            hosts = get_configured_hosts()
        except Exception as e:
            logger.debug(f"[IdleResourceDaemon] Failed to get cluster hosts: {e}")
            return

        for name, host in hosts.items():
            if host.best_ip is None:
                continue

            try:
                executor = SSHExecutor(
                    host=host.best_ip,
                    user=host.ssh_user,
                    port=host.ssh_port,
                    key_path=host.ssh_key,
                    connect_timeout=5,
                    max_retries=1,
                )

                # Count selfplay/gpu_parallel processes
                count_result = await executor.run(
                    "pgrep -c -f 'selfplay|gpu_parallel' 2>/dev/null || echo 0",
                    timeout=10,
                )

                if not count_result.success:
                    continue

                try:
                    process_count = int(count_result.stdout.strip())
                except ValueError:
                    continue

                if process_count <= max_per_node:
                    continue

                excess = process_count - max_per_node
                logger.warning(
                    f"[IdleResourceDaemon] Node {name} has {process_count} processes "
                    f"(max {max_per_node}), killing {excess} oldest"
                )

                # Kill oldest processes first (sorted by elapsed time)
                # ps -eo pid,etime,cmd sorts by start time, oldest first
                kill_cmd = (
                    f"ps -eo pid,etime,cmd --sort=etime 2>/dev/null | "
                    f"grep -E 'selfplay|gpu_parallel' | "
                    f"grep -v grep | "
                    f"head -n {excess} | "
                    f"awk '{{print $1}}' | "
                    f"xargs -r kill -9 2>/dev/null || true"
                )

                kill_result = await executor.run(kill_cmd, timeout=30)

                if kill_result.success:
                    logger.info(
                        f"[IdleResourceDaemon] Killed {excess} excess processes on {name}"
                    )
                    self._stats.failed_spawns += 0  # Track cleanup (no stat for this yet)
                else:
                    logger.warning(
                        f"[IdleResourceDaemon] Failed to kill processes on {name}: "
                        f"{kill_result.stderr}"
                    )

            except Exception as e:
                logger.debug(f"[IdleResourceDaemon] Process check failed for {name}: {e}")

    def _should_spawn(self, node: NodeStatus, queue_depth: int) -> bool:
        """Decide whether to spawn selfplay on a node."""
        now = time.time()

        # =======================================================================
        # P2P Node Health Check (December 2025 - Critical Gap Fix)
        # =======================================================================
        # Skip nodes that are marked unhealthy by P2P cluster health events
        unhealthy_nodes = getattr(self, "_unhealthy_nodes", set())
        if node.node_id in unhealthy_nodes:
            logger.debug(
                f"[IdleResourceDaemon] Skipping {node.node_id}: marked unhealthy by P2P"
            )
            return False

        # Also check by host if node_id doesn't match
        if node.host and node.host in unhealthy_nodes:
            logger.debug(
                f"[IdleResourceDaemon] Skipping {node.node_id} (host {node.host}): "
                f"marked unhealthy by P2P"
            )
            return False

        # =======================================================================
        # Selfplay Capability Check (December 2025 - Direct Dispatch Fix)
        # =======================================================================
        # Skip nodes that have selfplay disabled (e.g., GH200s used for training only)
        if not self._is_selfplay_capable(node.node_id):
            logger.debug(
                f"[IdleResourceDaemon] Skipping {node.node_id}: selfplay_enabled=False"
            )
            return False

        # =======================================================================
        # Incompatible Node Check (December 2025 - Phase 2 Training Loop Fix)
        # =======================================================================
        # Skip nodes that have been cached as incompatible (no compatible configs)
        # Clear cache if GPU VRAM has changed (node might now be compatible)
        if node.node_id in self._incompatible_nodes_cache:
            cached_vram, cached_time = self._incompatible_nodes_cache[node.node_id]
            current_vram = getattr(node, "gpu_memory_total_gb", 0.0)

            # Clear cache if GPU VRAM changed significantly (might now be compatible)
            if abs(current_vram - cached_vram) > 1.0:
                logger.info(
                    f"[IdleResourceDaemon] Clearing incompatibility cache for {node.node_id}: "
                    f"VRAM changed {cached_vram:.0f}GB -> {current_vram:.0f}GB"
                )
                del self._incompatible_nodes_cache[node.node_id]
            else:
                logger.debug(
                    f"[IdleResourceDaemon] Skipping {node.node_id}: cached as incompatible "
                    f"(VRAM={cached_vram:.0f}GB, cached {now - cached_time:.0f}s ago)"
                )
                return False

        # =======================================================================
        # Stall Detection Check (Phase 21.5 - December 2025)
        # =======================================================================
        # Skip nodes that are penalized due to previous job stalls
        if HAS_STALL_DETECTION and get_stall_detector:
            try:
                detector = get_stall_detector()
                if detector.is_node_penalized(node.node_id):
                    remaining = detector.get_penalty_remaining(node.node_id)
                    logger.debug(
                        f"[IdleResourceDaemon] Skipping {node.node_id}: "
                        f"stall penalty active ({remaining:.0f}s remaining)"
                    )
                    return False
                # Also check if node is unhealthy due to too many stalls
                if detector.is_node_unhealthy(node.node_id):
                    logger.warning(
                        f"[IdleResourceDaemon] Skipping {node.node_id}: "
                        f"marked unhealthy due to repeated stalls"
                    )
                    return False
            except Exception as e:
                logger.debug(f"[IdleResourceDaemon] Stall detector check failed: {e}")

        # =======================================================================
        # Unified Backpressure Check (Phase 21.5 - December 2025)
        # =======================================================================
        # Use unified backpressure signal for comprehensive pressure monitoring
        if HAS_BACKPRESSURE and get_backpressure_monitor:
            try:
                monitor = get_backpressure_monitor()
                # Use cached signal (non-blocking) since this is a sync method
                signal = monitor.get_cached_signal()
                if signal is not None:
                    if signal.should_pause:
                        logger.info(
                            f"[IdleResourceDaemon] Unified backpressure pause: "
                            f"pressure={signal.overall_pressure:.2f}, "
                            f"skipping spawn on {node.node_id}"
                        )
                        return False
                    elif signal.spawn_rate_multiplier < 0.5:
                        # Probabilistically skip based on spawn rate multiplier
                        import random
                        if random.random() > signal.spawn_rate_multiplier:
                            logger.debug(
                                f"[IdleResourceDaemon] Backpressure throttle: "
                                f"multiplier={signal.spawn_rate_multiplier:.2f}, "
                                f"skipping {node.node_id}"
                            )
                            return False
            except Exception as e:
                logger.debug(f"[IdleResourceDaemon] Backpressure check failed: {e}")

        # =======================================================================
        # Queue Backpressure Fallback (December 2025)
        # =======================================================================
        # Simple queue depth check as fallback if unified backpressure unavailable
        if queue_depth > self.config.max_queue_depth:
            logger.info(
                f"[IdleResourceDaemon] Queue backpressure: depth {queue_depth} > "
                f"max {self.config.max_queue_depth}, skipping spawn on {node.node_id}"
            )
            return False

        # Check training data backlog (prevent generating data faster than training)
        pending_hours = self._get_pending_training_hours()
        if pending_hours > self.config.max_pending_training_hours:
            logger.info(
                f"[IdleResourceDaemon] Training backlog: {pending_hours:.1f}h > "
                f"max {self.config.max_pending_training_hours}h, skipping spawn"
            )
            return False

        # Check if node is in backoff from previous failures (December 2025)
        if self._is_node_in_backoff(node.node_id):
            remaining = self._get_node_backoff_remaining(node.node_id)
            logger.debug(
                f"[IdleResourceDaemon] Skipping {node.node_id}: "
                f"in backoff for {remaining:.0f}s more"
            )
            return False

        # Check if node is idle long enough
        if node.idle_since <= 0:
            return False

        idle_duration = now - node.idle_since

        # Adjust threshold based on queue depth
        if queue_depth > self.config.high_queue_depth:
            # More aggressive spawning when queue is deep
            threshold = self.config.idle_threshold_percent * 3  # 15% (base 5% * 3)
            required_idle_time = self.config.idle_duration_seconds / 3  # 5 seconds
        elif queue_depth > self.config.medium_queue_depth:
            threshold = self.config.idle_threshold_percent * 2  # 10% (base 5% * 2)
            required_idle_time = self.config.idle_duration_seconds / 2  # 7.5 seconds
        else:
            threshold = self.config.idle_threshold_percent  # 5% (base threshold)
            required_idle_time = self.config.idle_duration_seconds  # 15 seconds

        # Check conditions
        if node.gpu_utilization > threshold:
            return False

        if idle_duration < required_idle_time:
            return False

        # Dec 26 2025: Enforce process limit - don't spawn if node at capacity
        # Note: active_jobs may be 0 for nodes where P2P tracking isn't perfect,
        # but this still protects against spawning on nodes that report high counts
        if node.active_jobs >= self.config.max_selfplay_processes_per_node:
            logger.debug(
                f"[IdleResourceDaemon] Node {node.node_id} at process limit "
                f"({node.active_jobs}/{self.config.max_selfplay_processes_per_node})"
            )
            return False

        return True

    def _select_config_for_gpu(self, gpu_memory_gb: float) -> str | None:
        """Select appropriate board config for GPU memory.

        December 2025 - Phase 2C.4: Now uses SelfplayScheduler priorities
        to select the highest-priority config that fits the GPU.

        December 2025 - Phase 2 Training Loop Fix: Returns None if no configs
        are compatible with this GPU, allowing caller to cache the node as
        incompatible and emit an event.

        Returns:
            config_key if a compatible config exists, None otherwise.
        """
        # Get configs that fit this GPU's memory
        valid_configs = {
            config_key for config_key, required_memory
            in self.config.gpu_memory_thresholds.items()
            if gpu_memory_gb >= required_memory
        }

        if not valid_configs:
            # No configs fit this GPU - return None to signal incompatibility
            logger.debug(
                f"[IdleResourceDaemon] No configs fit GPU with {gpu_memory_gb:.0f}GB VRAM "
                f"(min required: {min(self.config.gpu_memory_thresholds.values())}GB)"
            )
            return None

        # Try to get priority from SelfplayScheduler
        try:
            from app.coordination.selfplay_scheduler import get_selfplay_scheduler

            scheduler = get_selfplay_scheduler()
            # Get priority configs using public API (Dec 2025: replaced private access)
            # Uses cached priorities, safe for sync context
            sorted_priorities = scheduler.get_priority_configs_sync(
                filter_configs=valid_configs
            )

            # Return highest priority config that fits this GPU
            if sorted_priorities:
                config_key, priority_score = sorted_priorities[0]
                logger.debug(
                    f"[IdleResourceDaemon] Selected {config_key} "
                    f"(priority={priority_score:.2f}) for {gpu_memory_gb:.0f}GB GPU"
                )
                return config_key

        except ImportError:
            logger.debug("[IdleResourceDaemon] SelfplayScheduler not available, using memory-based selection")
        except Exception as e:
            logger.debug(f"[IdleResourceDaemon] SelfplayScheduler query failed: {e}")

        # Fallback: Sort by memory requirement descending, pick largest that fits
        sorted_configs = sorted(
            self.config.gpu_memory_thresholds.items(),
            key=lambda x: x[1],
            reverse=True
        )

        for config_key, required_memory in sorted_configs:
            if gpu_memory_gb >= required_memory:
                return config_key

        # No compatible config found
        return None

    async def _spawn_selfplay(self, node: NodeStatus) -> bool:
        """Spawn a selfplay job on the given node."""
        async with self._semaphore:
            self._stats.total_spawns += 1
            start_time = time.time()
            config_key = "unknown"
            games = self.config.default_games_per_spawn

            try:
                # Phase 4: Check circuit breaker before spawning
                # Prevents cascading failures when cluster operations are failing
                if HAS_CIRCUIT_BREAKER and get_operation_breaker:
                    breaker = get_operation_breaker()
                    if not breaker.can_execute("selfplay_spawn"):
                        logger.debug(
                            f"[IdleResourceDaemon] Circuit open for selfplay_spawn, "
                            f"skipping {node.node_id}"
                        )
                        return False

                config_key = self._select_config_for_gpu(node.gpu_memory_total_gb)

                # December 2025 - Phase 2 Training Loop Fix: Handle incompatible nodes
                if config_key is None:
                    gpu_vram = node.gpu_memory_total_gb
                    has_gpu = gpu_vram > 0

                    # Cache this node as incompatible (with timestamp for expiry)
                    self._incompatible_nodes_cache[node.node_id] = (gpu_vram, time.time())

                    # Emit event once per node (not on every cycle)
                    if HAS_INCOMPATIBILITY_EVENTS and emit_node_incompatible_with_workload:
                        try:
                            await emit_node_incompatible_with_workload(
                                node_id=node.node_id,
                                node_ip=getattr(node, "host", ""),
                                gpu_vram_gb=gpu_vram,
                                has_gpu=has_gpu,
                                reason="no_compatible_configs",
                                compatible_configs=[],
                                source="IdleResourceDaemon",
                            )
                        except (ImportError, RuntimeError, OSError, AttributeError, TypeError) as emit_err:
                            logger.debug(f"[IdleResourceDaemon] Failed to emit incompatibility event: {emit_err}")

                    logger.warning(
                        f"[IdleResourceDaemon] Node {node.node_id} has no compatible configs: "
                        f"GPU VRAM={gpu_vram:.0f}GB. Caching as incompatible."
                    )
                    self._stats.failed_spawns += 1
                    return False

                # P11-CRITICAL-2: Check free GPU memory before spawning
                # This prevents OOM errors by ensuring adequate VRAM headroom
                required_memory = self.config.gpu_memory_thresholds.get(config_key, 8)
                free_memory = node.gpu_memory_total_gb - node.gpu_memory_used_gb
                min_required = required_memory + self.config.min_free_gpu_memory_buffer_gb

                if free_memory < min_required:
                    logger.info(
                        f"[IdleResourceDaemon] Skipping {node.node_id}: insufficient VRAM. "
                        f"Free={free_memory:.1f}GB, required={min_required:.1f}GB "
                        f"(config={config_key} needs {required_memory}GB + "
                        f"{self.config.min_free_gpu_memory_buffer_gb}GB buffer)"
                    )
                    self._stats.failed_spawns += 1
                    return False

                # Get multiplier from FeedbackAccelerator
                try:
                    from app.training.feedback_accelerator import get_selfplay_multiplier
                    multiplier = get_selfplay_multiplier(config_key)
                    games = int(games * multiplier)
                except ImportError:
                    pass

                logger.info(
                    f"[IdleResourceDaemon] Spawning selfplay on {node.node_id}: "
                    f"config={config_key}, games={games}, "
                    f"gpu_memory={node.gpu_memory_total_gb:.0f}GB, "
                    f"free={free_memory:.1f}GB"
                )

                # Phase 21.2: Also schedule via PriorityJobScheduler for tracking
                if HAS_JOB_SCHEDULER and get_scheduler:
                    try:
                        scheduler = get_scheduler()
                        if scheduler:
                            # Parse config key for job config using canonical utility
                            parsed = parse_config_key(config_key)
                            board_type = parsed.board_type if parsed else config_key
                            num_players = parsed.num_players if parsed else 2

                            # Jan 5, 2026 (Phase 3): Use node's actual GPU capability
                            # CPU nodes (like Hetzner) can run heuristic selfplay
                            node_has_gpu = getattr(node, "gpu_memory_total_gb", 0) > 0
                            job = ScheduledJob(
                                job_type="selfplay",
                                priority=JobPriority.NORMAL,  # Idle-spawned jobs are normal priority
                                config={
                                    "board_type": board_type,
                                    "num_players": num_players,
                                    "games": games,
                                    "config_key": config_key,
                                    "source": "idle_resource_daemon",
                                },
                                host_preference=node.node_id,
                                requires_gpu=node_has_gpu,
                                estimated_duration_seconds=games * 10.0,  # ~10s per game estimate
                            )
                            scheduler.schedule(job)
                            logger.debug(
                                f"[IdleResourceDaemon] Scheduled job via JobScheduler: {config_key}"
                            )
                    except Exception as e:
                        logger.debug(f"[IdleResourceDaemon] JobScheduler integration failed: {e}")

                # Phase 21.5: Register job with stall detector for progress tracking
                job_id = f"selfplay_{node.node_id}_{config_key}_{int(start_time)}"
                if HAS_STALL_DETECTION and get_stall_detector:
                    try:
                        detector = get_stall_detector()
                        detector.register_job(job_id, node.node_id)
                    except Exception as e:
                        logger.debug(f"[IdleResourceDaemon] Stall detector registration failed: {e}")

                # Spawn via P2P job distribution
                success = await self._distribute_job(node, config_key, games)
                duration = time.time() - start_time

                if success:
                    self._stats.successful_spawns += 1
                    self._stats.games_spawned += games
                    self._stats.last_spawn_time = time.time()

                    # Record successful attempt (December 2025)
                    self._record_spawn_attempt(
                        node_id=node.node_id,
                        config_key=config_key,
                        games=games,
                        success=True,
                        duration=duration,
                    )

                    # Phase 4: Record circuit breaker success
                    if HAS_CIRCUIT_BREAKER and get_operation_breaker:
                        get_operation_breaker().record_success("selfplay_spawn")

                    # Phase 21.5: Mark job complete for stall detector (reduces node penalty)
                    if HAS_STALL_DETECTION and get_stall_detector:
                        try:
                            detector = get_stall_detector()
                            detector.complete_job(job_id, success=True)
                        except Exception as e:
                            logger.debug(f"[IdleResourceDaemon] Stall detector completion failed: {e}")

                    # Emit event
                    self._emit_spawn_event(node, config_key, games)

                    # Reset idle tracking for this node
                    if node.node_id in self._node_states:
                        self._node_states[node.node_id].idle_since = 0.0

                    return True
                else:
                    self._stats.failed_spawns += 1
                    # Record failed attempt (December 2025)
                    self._record_spawn_attempt(
                        node_id=node.node_id,
                        config_key=config_key,
                        games=games,
                        success=False,
                        error="P2P job distribution returned failure",
                        duration=duration,
                    )
                    # Phase 4: Record circuit breaker failure
                    if HAS_CIRCUIT_BREAKER and get_operation_breaker:
                        get_operation_breaker().record_failure("selfplay_spawn")

                    # Phase 21.5: Report stall to detector (applies node penalty)
                    if HAS_STALL_DETECTION and get_stall_detector:
                        try:
                            detector = get_stall_detector()
                            detector.report_stall(job_id, node.node_id, duration)
                        except Exception as e:
                            logger.debug(f"[IdleResourceDaemon] Stall detector report failed: {e}")

                    return False

            except Exception as e:
                self._stats.failed_spawns += 1
                self._stats.last_error = str(e)
                duration = time.time() - start_time
                # Record failed attempt with exception details (December 2025)
                self._record_spawn_attempt(
                    node_id=node.node_id,
                    config_key=config_key,
                    games=games,
                    success=False,
                    error=str(e),
                    duration=duration,
                )
                # Phase 4: Record circuit breaker failure on exception
                if HAS_CIRCUIT_BREAKER and get_operation_breaker:
                    get_operation_breaker().record_failure("selfplay_spawn")

                # Phase 21.5: Report stall on exception (applies node penalty)
                if HAS_STALL_DETECTION and get_stall_detector:
                    try:
                        detector = get_stall_detector()
                        detector.report_stall(job_id, node.node_id, duration)
                    except Exception as ex:
                        logger.debug(f"[IdleResourceDaemon] Stall detector report failed: {ex}")

                logger.error(f"Failed to spawn selfplay on {node.node_id}: {e}")
                return False

    async def _distribute_job(
        self,
        node: NodeStatus,
        config_key: str,
        games: int,
    ) -> bool:
        """Distribute a selfplay job to a node.

        Dec 2025: Added SSH fallback when P2P is unavailable. This allows
        spawning jobs on nodes even if their P2P daemon isn't running.
        """
        # Parse config key first (needed for both methods) using canonical utility
        parsed = parse_config_key(config_key)
        if not parsed:
            logger.warning(f"Invalid config key: {config_key}")
            return False
        board_type = parsed.board_type
        num_players = parsed.num_players

        # Phase 1: Try P2P first (preferred)
        p2p_success = await self._distribute_job_via_p2p(
            node, board_type, num_players, games
        )
        if p2p_success:
            return True

        # Phase 2: SSH fallback for nodes discovered via SSH
        if HAS_SSH_FALLBACK:
            return await self._distribute_job_via_ssh(
                node, board_type, num_players, games
            )

        return False

    async def _distribute_job_via_p2p(
        self,
        node: NodeStatus,
        board_type: str,
        num_players: int,
        games: int,
    ) -> bool:
        """Distribute job via P2P direct dispatch.

        Dec 29, 2025: Changed from work queue (submit_job) to direct dispatch.
        The work queue model doesn't work for selfplay because workers only
        pull when completely idle. Direct dispatch via /selfplay/start works
        immediately on the target node.
        """
        try:
            from app.coordination.p2p_integration import dispatch_selfplay_direct
            from app.config.cluster_config import get_p2p_port

            # Select engine based on board type for feasible throughput
            # Large boards (square19, hexagonal) use lighter engines
            if board_type in ("square19", "hexagonal"):
                import random
                if num_players >= 3:
                    engine_mode = random.choice(["heuristic-only", "brs", "maxn"])
                else:
                    engine_mode = random.choice(["heuristic-only", "policy-only"])
            else:
                engine_mode = "gumbel-mcts"  # GPU-accelerated Gumbel MCTS for small boards

            # Get host from node - prefer host attribute, fall back to node_id
            host = getattr(node, "host", None) or node.node_id
            port = getattr(node, "port", None) or get_p2p_port()

            # Direct dispatch to /selfplay/start endpoint
            result = await dispatch_selfplay_direct(
                target_node=node.node_id,
                host=host,
                port=port,
                board_type=board_type,
                num_players=num_players,
                num_games=games,
                engine_mode=engine_mode,
            )

            if result.success:
                logger.info(
                    f"[IdleResourceDaemon] Dispatched selfplay to {node.node_id}: "
                    f"{board_type}_{num_players}p, {games} games, job_id={result.job_id}"
                )
            return result.success

        except ImportError as e:
            logger.debug(f"P2P dispatch not available: {e}")
            return False
        except Exception as e:
            logger.debug(f"P2P job dispatch failed: {e}")
            return False

    async def _distribute_job_via_ssh(
        self,
        node: NodeStatus,
        board_type: str,
        num_players: int,
        games: int,
    ) -> bool:
        """Distribute job via SSH when P2P is unavailable.

        Dec 2025: SSH-based job spawn for nodes not in P2P cluster.
        Spawns selfplay as a background process on the remote node.

        Args:
            node: Target node info.
            board_type: Board type (e.g., 'hex8', 'square8').
            num_players: Number of players.
            games: Number of games to run.

        Returns:
            True if job was spawned successfully.
        """
        if not HAS_SSH_FALLBACK or SSHExecutor is None or get_configured_hosts is None:
            return False

        try:
            # Get SSH config for this node
            configured_hosts = get_configured_hosts()
            host_config = configured_hosts.get(node.node_id)

            if host_config is None:
                # Try to find by IP
                for name, cfg in configured_hosts.items():
                    if cfg.best_ip == node.host:
                        host_config = cfg
                        break

            if host_config is None or host_config.best_ip is None:
                logger.debug(
                    f"[IdleResourceDaemon] No SSH config for {node.node_id}, "
                    "cannot distribute via SSH"
                )
                return False

            executor = SSHExecutor(
                host=host_config.best_ip,
                user=host_config.ssh_user,
                port=host_config.ssh_port,
                key_path=host_config.ssh_key,
                connect_timeout=10,
                max_retries=2,
            )

            # Build selfplay command
            # Use nohup to detach from SSH session
            ringrift_path = host_config.ringrift_path or "~/ringrift/ai-service"

            # Expand ~ in path
            if ringrift_path.startswith("~"):
                ringrift_path = ringrift_path.replace("~", "$HOME", 1)

            selfplay_cmd = (
                f"cd {ringrift_path} && "
                f"PYTHONPATH=. nohup python scripts/selfplay.py "
                f"--board {board_type} --num-players {num_players} "
                f"--num-games {games} --engine gumbel-mcts "
                f"> /tmp/selfplay_{board_type}_{num_players}p_{int(time.time())}.log 2>&1 &"
            )

            logger.info(
                f"[IdleResourceDaemon] SSH spawn on {node.node_id}: "
                f"{board_type}_{num_players}p x{games} games"
            )

            result = await executor.run(selfplay_cmd, timeout=30)

            if result.success:
                logger.info(
                    f"[IdleResourceDaemon] SSH spawn successful on {node.node_id}"
                )
                return True
            else:
                logger.warning(
                    f"[IdleResourceDaemon] SSH spawn failed on {node.node_id}: "
                    f"{result.stderr}"
                )
                return False

        except Exception as e:
            logger.warning(f"[IdleResourceDaemon] SSH job distribution failed: {e}")
            return False
