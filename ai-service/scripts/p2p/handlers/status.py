"""Status and Health HTTP handlers for P2P orchestrator.

January 2026 - P2P Modularization Phase 6a

This mixin provides HTTP handlers for status monitoring, health checks,
loop management, and diagnostics endpoints.

Must be mixed into a class that provides:
- self.node_id: str
- self.self_info: NodeInfo
- self.peers: dict[str, NodeInfo]
- self.peers_lock: threading.RLock
- self.selfplay_scheduler: SelfplayScheduler (optional)
- self.hybrid_transport: HybridTransport (optional)
- self._peer_health_scores: dict
- self._peer_circuit_breakers: dict
- self._get_loop_manager(): LoopManager
- self._get_voter_promotion_cb(): CircuitBreaker
- self._update_self_info(): None
- self._seed_selfplay_scheduler_game_counts_sync(): dict
- self._fetch_game_counts_from_peers(): dict
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any

from aiohttp import web

from scripts.p2p.network import NonBlockingAsyncLockWrapper

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Optional imports for rate negotiation
try:
    from app.coordination.resource_optimizer import get_utilization_status
    HAS_RATE_NEGOTIATION = True
except ImportError:
    HAS_RATE_NEGOTIATION = False
    get_utilization_status = None

# Optional ClusterManifest for aggregated game counts
try:
    from pathlib import Path
    from app.distributed.cluster_manifest import ClusterManifest
    HAS_CLUSTER_MANIFEST = True
except ImportError:
    HAS_CLUSTER_MANIFEST = False
    ClusterManifest = None

# Optional async timeout
try:
    from async_timeout import timeout as async_timeout
except ImportError:
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def async_timeout(seconds: float):
        """Fallback async timeout context manager."""
        yield


class StatusHandlersMixin:
    """Mixin providing status and health monitoring HTTP handlers.

    Endpoints:
    - GET /health - Node health check for monitoring
    - GET /peer-health - Per-peer health scores and circuit breakers
    - GET /external-work - External work status across cluster
    - GET /game-counts - Game counts from canonical databases
    - POST /game-counts/refresh - Trigger game count refresh
    - POST /loops/restart/{name} - Restart a specific loop
    - POST /loops/restart_stopped - Restart all stopped loops
    - GET /loops/status - Get status of all background loops
    - GET /circuit-breakers/status - Get circuit breaker status
    - GET /connectivity/transport_stats - Get transport statistics
    - GET /dispatch/stats - Get work queue dispatch statistics
    """

    # Required attributes (provided by orchestrator)
    node_id: str
    self_info: Any  # NodeInfo
    peers: dict[str, Any]  # dict[str, NodeInfo]
    peers_lock: Any  # threading.RLock
    selfplay_scheduler: Any  # Optional[SelfplayScheduler]
    hybrid_transport: Any  # Optional[HybridTransport]
    _peer_health_scores: dict
    _peer_circuit_breakers: dict

    # Required methods (provided by orchestrator)
    def _get_loop_manager(self) -> Any:
        """Get the LoopManager instance."""
        ...

    def _get_voter_promotion_cb(self) -> Any:
        """Get the voter promotion circuit breaker."""
        ...

    def _update_self_info(self) -> None:
        """Update self_info with current node state."""
        ...

    async def _update_self_info_async(self) -> None:
        """Async version of _update_self_info."""
        ...

    def _seed_selfplay_scheduler_game_counts_sync(self) -> dict:
        """Get game counts from local canonical databases (sync)."""
        ...

    async def _fetch_game_counts_from_peers(self) -> dict:
        """Fetch game counts from peers."""
        ...

    async def handle_health(self, request: web.Request) -> web.Response:
        """GET /health - Handle health check request.

        Simple health endpoint for monitoring and load balancers.
        Returns node health status without full cluster state.
        Includes utilization status from resource_optimizer for cluster coordination.

        Jan 8, 2026: Added 15s timeout protection to prevent health endpoint from
        blocking indefinitely if resource detection hangs. Uses async version of
        _update_self_info to avoid blocking the event loop.

        Jan 12, 2026: Changed to non-blocking mode - returns cached data immediately
        while triggering background refresh. This prevents 15s timeouts on macOS where
        resource detection (torch import, subprocess calls) is slow.
        """
        try:
            # Jan 12, 2026: Non-blocking mode - schedule background refresh, return cached data
            # This prevents /health from blocking for 15s on macOS
            try:
                asyncio.create_task(self._update_self_info_async())
            except Exception:
                pass  # Fire-and-forget, don't block on errors

            # Return immediately with cached self_info data
            is_healthy = self.self_info.is_healthy()

            # Calculate uptime and leader status
            uptime_seconds = time.time() - getattr(self, "start_time", time.time())
            leader_last_seen = time.time() - getattr(self, "last_leader_seen", time.time())
            # Jan 12, 2026: Copy-on-write - snapshot for thread-safe iteration
            with self.peers_lock:
                peers_snapshot = list(self.peers.values())
            active_peers = sum(1 for p in peers_snapshot
                             if time.time() - p.last_heartbeat < 120)

            response = {
                "healthy": is_healthy,
                "node_id": self.node_id,
                "role": self.role.value,
                "disk_percent": self.self_info.disk_percent,
                "memory_percent": self.self_info.memory_percent,
                "cpu_percent": self.self_info.cpu_percent,
                "gpu_percent": self.self_info.gpu_percent,
                "gpu_memory_percent": self.self_info.gpu_memory_percent,
                "selfplay_jobs": self.self_info.selfplay_jobs,
                "training_jobs": self.self_info.training_jobs,
                # Cluster health for alerting
                "leader_id": self.leader_id,
                "leader_last_seen_seconds": leader_last_seen if self.leader_id else None,
                "active_peers": active_peers,
                "total_peers": len(peers_snapshot),
                "uptime_seconds": uptime_seconds,
                "timestamp": datetime.utcnow().isoformat(),
            }

            # Add cluster utilization status for cooperative 60-80% targeting
            if HAS_RATE_NEGOTIATION and get_utilization_status is not None:
                try:
                    util_status = get_utilization_status()
                    response["cluster_utilization"] = {
                        "cpu_util": util_status.get("cpu_util", 0),
                        "gpu_util": util_status.get("gpu_util", 0),
                        "selfplay_rate": util_status.get("current_rate", 1000),
                        "target_range": "60-80%",
                        "status": util_status.get("status", "unknown"),
                    }
                except (AttributeError):
                    pass

            return web.json_response(response)
        except Exception as e:  # noqa: BLE001
            return web.json_response({"error": str(e), "healthy": False}, status=500)

    async def handle_peer_health(self, request: web.Request) -> web.Response:
        """GET /peer-health - Return per-peer health scores and circuit breaker status.

        Jan 3, 2026: Sprint 10+ P2P hardening endpoint. Provides visibility into
        per-peer health metrics for monitoring and debugging:
        - PeerHealthScore: Composite health from success rate, latency, availability
        - PeerCircuitBreaker: Per-peer failure isolation state

        Query parameters:
            degraded_only: If "true", only return peers with health < 0.7
            include_breakers: If "true" (default), include circuit breaker state

        Returns:
            {
                "peers": {
                    "peer-id": {
                        "health": {...},     # PeerHealthScore.to_dict()
                        "circuit_breaker": {...}  # PeerCircuitBreaker state
                    }
                },
                "summary": {
                    "total_peers": 25,
                    "healthy": 20,
                    "degraded": 3,
                    "critical": 1,
                    "open_breakers": 1
                }
            }
        """
        degraded_only = request.query.get("degraded_only", "false").lower() == "true"
        include_breakers = request.query.get("include_breakers", "true").lower() != "false"

        result: dict[str, Any] = {}
        summary = {
            "total_peers": 0,
            "healthy": 0,
            "degraded": 0,
            "critical": 0,
            "open_breakers": 0,
        }

        # Get all known peers
        with self.peers_lock:
            peer_ids = list(self.peers.keys())

        for peer_id in peer_ids:
            health_score = self._peer_health_scores.get(peer_id)
            breaker = self._peer_circuit_breakers.get(peer_id)

            # Skip if no health data
            if not health_score and not breaker:
                continue

            # Filter by degraded_only
            if degraded_only and health_score and not health_score.is_degraded():
                continue

            summary["total_peers"] += 1

            peer_data: dict[str, Any] = {}

            if health_score:
                peer_data["health"] = health_score.to_dict()
                if health_score.is_critical():
                    summary["critical"] += 1
                elif health_score.is_degraded():
                    summary["degraded"] += 1
                else:
                    summary["healthy"] += 1

            if include_breakers and breaker:
                peer_data["circuit_breaker"] = {
                    "state": breaker.state,
                    "failure_count": breaker.failure_count,
                    "success_count": breaker.success_count,
                    "last_failure_time": breaker.last_failure_time,
                    "last_success_time": breaker.last_success_time,
                    "last_state_change": breaker.last_state_change,
                }
                if breaker.is_open():
                    summary["open_breakers"] += 1

            result[peer_id] = peer_data

        return web.json_response({
            "peers": result,
            "summary": summary,
            "timestamp": time.time(),
        })

    async def handle_external_work(self, request: web.Request) -> web.Response:
        """GET /external-work - Return external work status across the cluster.

        This endpoint shows work running outside P2P orchestrator tracking:
        - CMA-ES optimization jobs
        - Gauntlet runs
        - ELO tournaments
        - Data merge/aggregation

        Also identifies misrouted nodes (GPU nodes running CPU-bound work).
        """
        self._update_self_info()

        async with NonBlockingAsyncLockWrapper(self.peers_lock, "peers_lock", timeout=5.0):
            peers_snapshot = list(self.peers.values())

        # Collect external work info
        nodes_with_external = []
        misrouted_nodes = []

        # Check self
        self.self_info.to_dict()
        if self.self_info.has_external_work():
            nodes_with_external.append({
                'node_id': self.node_id,
                'cmaes': self.self_info.cmaes_running,
                'gauntlet': self.self_info.gauntlet_running,
                'tournament': self.self_info.tournament_running,
                'data_merge': self.self_info.data_merge_running,
                'gpu_percent': self.self_info.gpu_percent,
                'cpu_percent': self.self_info.cpu_percent,
            })
        if self.self_info.is_misrouted():
            misrouted_nodes.append({
                'node_id': self.node_id,
                'gpu_name': self.self_info.gpu_name,
                'gpu_percent': self.self_info.gpu_percent,
                'cpu_percent': self.self_info.cpu_percent,
                'external_work': {
                    'cmaes': self.self_info.cmaes_running,
                    'gauntlet': self.self_info.gauntlet_running,
                    'tournament': self.self_info.tournament_running,
                }
            })

        # Check peers
        for peer in peers_snapshot:
            if peer.has_external_work():
                nodes_with_external.append({
                    'node_id': peer.node_id,
                    'cmaes': peer.cmaes_running,
                    'gauntlet': peer.gauntlet_running,
                    'tournament': peer.tournament_running,
                    'data_merge': peer.data_merge_running,
                    'gpu_percent': peer.gpu_percent,
                    'cpu_percent': peer.cpu_percent,
                })
            if peer.is_misrouted():
                misrouted_nodes.append({
                    'node_id': peer.node_id,
                    'gpu_name': peer.gpu_name,
                    'gpu_percent': peer.gpu_percent,
                    'cpu_percent': peer.cpu_percent,
                    'external_work': {
                        'cmaes': peer.cmaes_running,
                        'gauntlet': peer.gauntlet_running,
                        'tournament': peer.tournament_running,
                    }
                })

        return web.json_response({
            'nodes_with_external_work': nodes_with_external,
            'misrouted_nodes': misrouted_nodes,
            'total_external_work': len(nodes_with_external),
            'total_misrouted': len(misrouted_nodes),
        })

    async def handle_game_counts(self, request: web.Request) -> web.Response:
        """GET /game-counts - Return game counts with cluster awareness.

        Session 17.41: Enables cluster nodes to fetch game counts from the coordinator.
        January 2026: Made cluster-aware using get_game_counts_cluster_aware() which
        aggregates data from cluster manifest, local DBs, OWC, and S3.

        Returns:
            JSON with config_key -> game_count mapping
        """
        try:
            # Try cluster-aware counts first (Jan 2026)
            try:
                from app.utils.game_discovery import get_game_counts_cluster_aware

                game_counts = await asyncio.to_thread(get_game_counts_cluster_aware)
                if game_counts and sum(game_counts.values()) > 0:
                    return web.json_response({
                        "game_counts": game_counts,
                        "node_id": self.node_id,
                        "source": "cluster_aware",
                        "count": len(game_counts),
                    })
            except ImportError:
                logger.debug("[P2P] get_game_counts_cluster_aware not available, falling back")
            except Exception as e:
                logger.debug(f"[P2P] Cluster-aware counts unavailable: {e}")

            # Fall back to local canonical databases
            game_counts = await asyncio.to_thread(self._seed_selfplay_scheduler_game_counts_sync)

            return web.json_response({
                "game_counts": game_counts,
                "node_id": self.node_id,
                "source": "canonical_databases",
                "count": len(game_counts),
            })
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[P2P] Failed to get game counts: {e}")
            return web.json_response({
                "game_counts": {},
                "node_id": self.node_id,
                "source": "error",
                "error": str(e),
            })

    async def handle_refresh_game_counts(self, request: web.Request) -> web.Response:
        """POST /refresh_game_counts - Trigger a refresh of game counts.

        Session 17.48: Added to allow manual/periodic refresh of game counts
        on cluster nodes that don't have local canonical databases.

        Jan 13, 2026: Fixed to use ClusterManifest for cluster-wide aggregated
        counts instead of just local canonical DBs. This ensures the scheduler
        makes decisions based on ALL data across the cluster, S3, and OWC.

        Returns:
            JSON with refresh status and new game count data
        """
        try:
            # Jan 13, 2026: First try ClusterManifest for aggregated cluster-wide counts
            if HAS_CLUSTER_MANIFEST:
                try:
                    manifest_db = Path("data/cluster_manifest.db")
                    if manifest_db.exists():
                        def _get_manifest_counts():
                            manifest = ClusterManifest(db_path=manifest_db)
                            configs = [
                                'hex8_2p','hex8_3p','hex8_4p',
                                'hexagonal_2p','hexagonal_3p','hexagonal_4p',
                                'square19_2p','square19_3p','square19_4p',
                                'square8_2p','square8_3p','square8_4p'
                            ]
                            counts = {}
                            for config in configs:
                                sources = manifest.get_total_games_across_sources(config)
                                total = sum(sources.values())
                                if total > 0:
                                    counts[config] = total
                            return counts

                        manifest_counts = await asyncio.to_thread(_get_manifest_counts)
                        if manifest_counts:
                            if self.selfplay_scheduler:
                                self.selfplay_scheduler.update_p2p_game_counts(manifest_counts)
                            logger.info(f"[P2P] Refreshed game counts from ClusterManifest: {len(manifest_counts)} configs")
                            return web.json_response({
                                "success": True,
                                "source": "cluster_manifest",
                                "game_counts": manifest_counts,
                                "count": len(manifest_counts),
                                "node_id": self.node_id,
                            })
                except Exception as e:
                    logger.debug(f"[P2P] ClusterManifest lookup failed, falling back: {e}")

            # Fall back to local canonical DBs
            local_counts = await asyncio.to_thread(self._seed_selfplay_scheduler_game_counts_sync)

            if local_counts:
                # We have local canonical DBs, use them
                if self.selfplay_scheduler:
                    self.selfplay_scheduler.update_p2p_game_counts(local_counts)
                return web.json_response({
                    "success": True,
                    "source": "local_canonical",
                    "game_counts": local_counts,
                    "count": len(local_counts),
                    "node_id": self.node_id,
                })

            # No local DBs, fetch from peers
            peer_counts = await self._fetch_game_counts_from_peers()

            if peer_counts and self.selfplay_scheduler:
                self.selfplay_scheduler.update_p2p_game_counts(peer_counts)
                logger.info(f"[P2P] Refreshed game counts: {len(peer_counts)} configs from peers")
                for config_key, count in sorted(peer_counts.items(), key=lambda x: x[1]):
                    if count < 500:
                        logger.info(f"[P2P] Underserved config (refreshed): {config_key} = {count} games")
                return web.json_response({
                    "success": True,
                    "source": "peers",
                    "game_counts": peer_counts,
                    "count": len(peer_counts),
                    "node_id": self.node_id,
                })

            return web.json_response({
                "success": False,
                "source": "none",
                "game_counts": {},
                "count": 0,
                "node_id": self.node_id,
                "error": "No game counts available from local or peers",
            })

        except Exception as e:  # noqa: BLE001
            logger.debug(f"[P2P] Failed to refresh game counts: {e}")
            return web.json_response({
                "success": False,
                "source": "error",
                "error": str(e),
                "node_id": self.node_id,
            }, status=500)

    async def handle_loop_restart(self, request: web.Request) -> web.Response:
        """POST /loops/restart/{name} - Restart a specific stopped loop.

        January 1, 2026: Added for 48h autonomous operation support.
        Allows restarting crashed/stopped loops without full P2P restart.
        """
        try:
            loop_name = request.match_info.get("name")
            if not loop_name:
                return web.json_response({"error": "Loop name required"}, status=400)

            loop_manager = self._get_loop_manager()
            if loop_manager is None:
                return web.json_response({
                    "error": "LoopManager not available"
                }, status=501)

            success = loop_manager.restart_loop(loop_name)
            if success:
                return web.json_response({
                    "success": True,
                    "message": f"Loop '{loop_name}' restart initiated",
                    "loop_name": loop_name,
                })
            else:
                return web.json_response({
                    "success": False,
                    "message": f"Could not restart loop '{loop_name}'",
                    "loop_name": loop_name,
                }, status=404)
        except Exception as e:  # noqa: BLE001
            logger.error(f"Loop restart error: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def handle_restart_stopped_loops(self, request: web.Request) -> web.Response:
        """POST /loops/restart_stopped - Restart all stopped but enabled loops.

        January 1, 2026: Added for 48h autonomous operation support.
        Automatically restarts all loops that have crashed/stopped unexpectedly.
        """
        try:
            loop_manager = self._get_loop_manager()
            if loop_manager is None:
                return web.json_response({
                    "error": "LoopManager not available"
                }, status=501)

            results = await loop_manager.restart_stopped_loops()
            restarted = [name for name, success in results.items() if success]
            failed = [name for name, success in results.items() if not success]

            return web.json_response({
                "success": True,
                "restarted": restarted,
                "failed": failed,
                "total_restarted": len(restarted),
                "total_failed": len(failed),
            })
        except Exception as e:  # noqa: BLE001
            logger.error(f"Restart stopped loops error: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def handle_loops_status(self, request: web.Request) -> web.Response:
        """GET /loops/status - Get status of all background loops.

        January 1, 2026: Added for monitoring and debugging loop health.
        """
        try:
            loop_manager = self._get_loop_manager()
            if loop_manager is None:
                return web.json_response({
                    "error": "LoopManager not available"
                }, status=501)

            all_status = loop_manager.get_all_status()
            health = loop_manager.health_check()

            # Summarize which loops are stopped
            stopped_loops = [
                name for name, status in all_status.items()
                if not status.get("running") and status.get("enabled")
            ]

            return web.json_response({
                "success": True,
                "health": health,
                "stopped_enabled_loops": stopped_loops,
                "loops": all_status,
            })
        except Exception as e:  # noqa: BLE001
            logger.error(f"Loops status error: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def handle_circuit_breaker_status(self, request: web.Request) -> web.Response:
        """GET /circuit-breakers/status - Get status of all circuit breakers.

        January 2026: Added for P2P reliability monitoring.

        Returns status of:
        - voter_promotion: Prevents voter churn during cluster instability
        - transport_cascade: Transport failover circuit breaker
        """
        try:
            status: dict[str, Any] = {"circuit_breakers": {}}

            # 1. Voter Promotion Circuit Breaker (from LeaderElectionMixin)
            try:
                cb = self._get_voter_promotion_cb()
                status["circuit_breakers"]["voter_promotion"] = cb.get_status()
            except AttributeError:
                status["circuit_breakers"]["voter_promotion"] = {"error": "Not available"}

            # 2. Global Transport Circuit Breaker (from transport_cascade)
            try:
                from scripts.p2p.transport_cascade import get_global_circuit_breaker
                gcb = get_global_circuit_breaker()
                stats = gcb.get_stats()
                # Add is_open for consistency with VoterPromotionCircuitBreaker
                stats["is_open"] = gcb.is_open
                status["circuit_breakers"]["transport_cascade"] = stats
            except (ImportError, AttributeError) as e:
                status["circuit_breakers"]["transport_cascade"] = {
                    "error": f"Not available: {e}"
                }

            # 3. Summary: count open circuit breakers
            open_count = 0
            for name, cb_status in status["circuit_breakers"].items():
                if isinstance(cb_status, dict) and cb_status.get("is_open"):
                    open_count += 1

            status["summary"] = {
                "total_circuit_breakers": len(status["circuit_breakers"]),
                "open_count": open_count,
                "health": "degraded" if open_count > 0 else "healthy",
            }

            return web.json_response(status)
        except Exception as e:  # noqa: BLE001
            logger.error(f"Circuit breaker status error: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def handle_node_circuit_breaker_metrics(self, request: web.Request) -> web.Response:
        """GET /circuit-breakers/nodes - Get per-node circuit breaker metrics.

        January 2026: Added for observability dashboard.

        Query params:
            format: "json" (default) or "prometheus"

        Returns per-node circuit breaker state, failure counts, and durations.
        """
        try:
            from app.coordination.node_circuit_breaker import get_node_circuit_breaker

            breaker = get_node_circuit_breaker()
            output_format = request.query.get("format", "json")

            if output_format == "prometheus":
                metrics_text = breaker.get_prometheus_metrics()
                return web.Response(
                    text=metrics_text,
                    content_type="text/plain; charset=utf-8",
                )
            else:
                metrics = breaker.get_metrics_dict()
                return web.json_response(metrics)

        except ImportError:
            return web.json_response({
                "available": False,
                "message": "Node circuit breaker not available",
            })
        except Exception as e:  # noqa: BLE001
            logger.error(f"Node circuit breaker metrics error: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def handle_transport_stats(self, request: web.Request) -> web.Response:
        """GET /connectivity/transport_stats - Get transport statistics for all nodes.

        Returns per-node transport preferences and success rates.
        """
        # Check if hybrid transport is available
        HAS_HYBRID_TRANSPORT = hasattr(self, 'hybrid_transport') and self.hybrid_transport is not None

        if not HAS_HYBRID_TRANSPORT:
            return web.json_response({
                "available": False,
                "message": "Hybrid transport not available",
            })

        try:
            stats = self.hybrid_transport.get_transport_stats()
            return web.json_response({
                "available": True,
                "node_count": len(stats),
                "nodes": stats,
            })
        except Exception as e:  # noqa: BLE001
            return web.json_response({"error": str(e)}, status=500)

    async def handle_dispatch_stats(self, request: web.Request) -> web.Response:
        """GET /dispatch/stats - Get work queue dispatch statistics.

        Jan 3, 2026: Added for observability into job dispatch filtering.
        Returns claim rejection stats showing why jobs aren't being dispatched
        to idle GPU nodes.
        """
        try:
            from app.coordination.work_queue import get_work_queue

            wq = get_work_queue()
            if wq is None:
                return web.json_response({
                    "available": False,
                    "message": "Work queue not initialized",
                })

            return web.json_response({
                "available": True,
                "claim_rejection_stats": wq.get_claim_rejection_stats(),
                "queue_stats": wq.get_queue_stats(),
            })
        except Exception as e:  # noqa: BLE001
            return web.json_response({"error": str(e)}, status=500)

    async def handle_work_queue_claim_stats(self, request: web.Request) -> web.Response:
        """GET /work_queue/claim_stats - Get enhanced claim rejection statistics.

        January 13, 2026: Added for infrastructure monitoring as part of the
        resilience plan. Returns enhanced stats with:
        - success_rate: Ratio of successful claims to total attempts
        - top_rejection_reason: Most common rejection reason
        - Detailed breakdown by rejection type

        This endpoint helps diagnose why the work queue accumulates items
        when circuit breakers block claims.
        """
        try:
            from app.coordination.work_queue import get_work_queue

            wq = get_work_queue()
            if wq is None:
                return web.json_response({
                    "available": False,
                    "message": "Work queue not initialized",
                })

            return web.json_response(wq.get_claim_rejection_stats_dict())
        except Exception as e:  # noqa: BLE001
            return web.json_response({"error": str(e)}, status=500)

    async def handle_data_summary(self, request: web.Request) -> web.Response:
        """GET /data/summary - Get unified game data summary across all sources.

        January 2026: Added as part of unified data discovery infrastructure.
        Returns game counts from all sources (LOCAL, CLUSTER, S3, OWC) for
        accurate data visibility across the cluster.

        Query Parameters:
            refresh: If "true", forces cache refresh
            local_only: If "true", skips remote sources

        Returns:
            JSON with:
            - timestamp: ISO timestamp
            - sources: Per-source game counts by config
            - totals: Total games per config across all sources
            - cache_age_seconds: Age of cached data
            - errors: List of any errors during aggregation
        """
        try:
            # Parse query params
            refresh = request.query.get("refresh", "").lower() == "true"
            local_only = request.query.get("local_only", "").lower() == "true"

            # Import aggregator
            from app.utils.unified_game_aggregator import (
                GameSourceConfig,
                UnifiedGameAggregator,
            )

            # Configure source inclusion
            if local_only:
                config = GameSourceConfig.local_only()
            else:
                config = GameSourceConfig.all_sources()

            aggregator = UnifiedGameAggregator(config)

            # Clear cache if refresh requested
            if refresh:
                aggregator.clear_cache()

            # Get all config counts
            counts = await aggregator.get_all_configs_counts()

            # Build response
            sources: dict[str, dict[str, int]] = {
                "local": {},
                "cluster": {},
                "s3": {},
                "owc": {},
            }
            totals: dict[str, int] = {}
            errors: list[dict] = []
            cache_age = 0.0

            for config_key, result in counts.items():
                # Per-source breakdown
                for source in ["local", "cluster", "s3", "owc"]:
                    count = result.sources.get(source, 0)
                    sources[source][config_key] = count

                # Total
                totals[config_key] = result.total_games

                # Track cache age
                if result.last_updated > 0:
                    age = time.time() - result.last_updated
                    if age > cache_age:
                        cache_age = age

                # Collect errors
                for error in result.errors:
                    errors.append({"config": config_key, "error": error})

            return web.json_response({
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "sources": sources,
                "totals": totals,
                "grand_total": {
                    "local": sum(sources["local"].values()),
                    "cluster": sum(sources["cluster"].values()),
                    "s3": sum(sources["s3"].values()),
                    "owc": sum(sources["owc"].values()),
                    "total": sum(totals.values()),
                },
                "cache_age_seconds": round(cache_age, 1),
                "errors": errors[:20],  # Limit errors
                "node_id": self.node_id,
            })

        except ImportError as e:
            return web.json_response({
                "error": f"UnifiedGameAggregator not available: {e}",
                "node_id": self.node_id,
            }, status=500)
        except Exception as e:  # noqa: BLE001
            logger.exception(f"[P2P] Failed to get data summary: {e}")
            return web.json_response({
                "error": str(e),
                "node_id": self.node_id,
            }, status=500)

    async def handle_parallelism_status(self, request: web.Request) -> web.Response:
        """GET /status/parallelism - Return parallelism infrastructure metrics.

        Jan 2026: Phase 5 of P2P multi-core parallelization.
        Exposes metrics for:
        - Thread pool utilization (Phase 2)
        - Threaded loop runners (Phase 3)
        - CPU affinity allocations (Phase 4)

        Returns:
            JSON with parallelism metrics for monitoring
        """
        response: dict[str, Any] = {
            "node_id": self.node_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        # Phase 2: Loop executor pools
        try:
            from scripts.p2p.loop_executors import LoopExecutors
            response["loop_executors"] = LoopExecutors.get_all_stats_summary()
        except ImportError:
            response["loop_executors"] = {"available": False}
        except Exception as e:
            response["loop_executors"] = {"error": str(e)}

        # Phase 3: Threaded loop runners
        try:
            from scripts.p2p.threaded_loop_runner import ThreadedLoopRegistry
            response["threaded_runners"] = ThreadedLoopRegistry.get_summary()
        except ImportError:
            response["threaded_runners"] = {"available": False}
        except Exception as e:
            response["threaded_runners"] = {"error": str(e)}

        # Phase 4: CPU affinity
        try:
            from scripts.p2p.cpu_affinity import get_affinity_manager
            manager = get_affinity_manager()
            response["cpu_affinity"] = manager.get_stats()
            response["cpu_affinity"]["allocations"] = manager.get_all_allocations()
        except ImportError:
            response["cpu_affinity"] = {"available": False}
        except Exception as e:
            response["cpu_affinity"] = {"error": str(e)}

        # Summary
        pools_enabled = response.get("loop_executors", {}).get("enabled", False)
        threads_enabled = response.get("threaded_runners", {}).get("enabled", False)
        affinity_enabled = response.get("cpu_affinity", {}).get("enabled", False)

        response["summary"] = {
            "loop_pools_enabled": pools_enabled,
            "threaded_loops_enabled": threads_enabled,
            "cpu_affinity_enabled": affinity_enabled,
            "all_features_enabled": pools_enabled and threads_enabled and affinity_enabled,
        }

        return web.json_response(response)
