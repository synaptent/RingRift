"""Status Monitoring Mixin - health, status, and self-info endpoints.

April 2026: Extracted from p2p_orchestrator.py (Phase 4 task 18).
"""
from __future__ import annotations

import asyncio
import functools
import json
import logging
import time
from typing import Any

from aiohttp import web

from scripts.p2p.constants import NAT_INBOUND_HEARTBEAT_STALE_SECONDS
from scripts.p2p.p2p_mixin_base import P2PMixinBase
from scripts.p2p.types import NodeRole

try:
    from app.coordination.resource_optimizer import (
        NodeResources,
        get_resource_optimizer,
    )
    HAS_NEW_COORDINATION = True
except ImportError:
    NodeResources = None
    get_resource_optimizer = None
    HAS_NEW_COORDINATION = False

logger = logging.getLogger(__name__)


def with_request_timeout(timeout_seconds: float = 20.0):
    """Decorator to add timeout protection to HTTP handlers."""
    def decorator(handler):
        @functools.wraps(handler)
        async def wrapper(self_or_request, *args, **kwargs):
            try:
                return await asyncio.wait_for(
                    handler(self_or_request, *args, **kwargs),
                    timeout=timeout_seconds,
                )
            except asyncio.TimeoutError:
                return web.json_response(
                    {
                        "error": "Request timed out",
                        "timeout_seconds": timeout_seconds,
                        "timestamp": time.time(),
                    },
                    status=504,
                )
        return wrapper
    return decorator


class StatusMonitoringMixin(P2PMixinBase):
    """Mixin extracted from P2POrchestrator."""

    MIXIN_TYPE = "status_monitoring"

    node_id: str
    role: Any
    leader_id: str | None
    self_info: Any
    peers: dict[str, Any]
    known_peers: list[str]
    heartbeat_manager: Any
    health_metrics_manager: Any
    jobs: Any
    _peer_snapshot: Any
    _job_snapshot: Any

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
            fire_and_forget(
                self._update_self_info_async(),
                name=f"refresh_self_info:{self.node_id}",
            )
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
