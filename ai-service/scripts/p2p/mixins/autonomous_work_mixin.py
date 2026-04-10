"""Autonomous work discovery, claiming, execution, and predictive selfplay helpers.

April 2026: Extracted from p2p_orchestrator.py (Part 3 Phase 2 follow-up).
"""
from __future__ import annotations

from scripts.p2p.p2p_mixin_base import P2PMixinBase
from scripts.p2p.startup_infrastructure import *  # noqa: F401,F403


class AutonomousWorkMixin(P2PMixinBase):
    """Mixin for P2POrchestrator autonomous work discovery, claiming, execution, and predictive selfplay helpers."""

    MIXIN_TYPE = "autonomous_work"

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
