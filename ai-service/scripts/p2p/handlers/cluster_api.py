"""Cluster API HTTP handlers for P2P orchestrator.

January 2026 - P2P Modularization Phase 1c

This mixin provides HTTP handlers for cluster-wide API endpoints including
status, health, and git update operations.

Must be mixed into a class that provides:
- self.node_id: str
- self.role: NodeRole
- self.leader_id: str
- self.build_version: str
- self.start_time: float
- self.self_info: NodeInfo
- self.peers: dict[str, NodeInfo]
- self.peers_lock: threading.RLock
- self.local_jobs: dict[str, Job]
- self.jobs_lock: threading.RLock
- self.training_jobs: dict[str, TrainingJob]
- self.training_lock: threading.RLock
- self.manifest_lock: threading.RLock
- self.local_data_manifest: Optional[LocalDataManifest]
- self.cluster_data_manifest: Optional[ClusterDataManifest]
- self.voter_node_ids: list[str]
- self.voter_quorum_size: int
- self._is_leader() -> bool
- self._get_leader_peer() -> Optional[NodeInfo]
- self._proxy_to_leader(request) -> web.Response
- self._update_self_info() -> None
- self._endpoint_key(peer) -> Optional[tuple[str, str, int]]
- self._has_voter_quorum() -> bool
- self._count_alive_voters() -> int
- self._urls_for_peer(peer, path) -> list[str]
- self._auth_headers() -> dict
- self._check_for_updates() -> tuple[bool, str, str]
- self._perform_git_update() -> tuple[bool, str]
- self._restart_orchestrator() -> None
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from aiohttp import ClientTimeout, web

from app.core.async_context import safe_create_task

if TYPE_CHECKING:
    from ..types import NodeInfo

logger = logging.getLogger(__name__)


# Import client session helper
def get_client_session(timeout: ClientTimeout | None = None):
    """Get aiohttp client session with optional timeout.

    This is a placeholder - the actual implementation should be imported
    from the orchestrator's module or a shared utility.
    """
    import aiohttp
    return aiohttp.ClientSession(timeout=timeout)


class ClusterApiHandlersMixin:
    """Mixin providing cluster API HTTP handlers.

    Endpoints:
    - GET /api/cluster/status - Comprehensive cluster status
    - POST /api/cluster/git_update - Leader-coordinated git updates
    - GET /cluster/health - Aggregated health from all nodes
    """

    async def handle_api_cluster_status(self, request: web.Request) -> web.Response:
        """Get comprehensive cluster status for external clients and dashboard."""
        try:
            if not self._is_leader() and request.query.get("local") != "1":
                proxied = await self._proxy_to_leader(request)
                if proxied.status not in (502, 503):
                    return proxied

            # Ensure local resource stats are fresh for dashboard consumers.
            # Feb 2026: Use asyncio.to_thread to avoid blocking the event loop
            # (_update_self_info calls subprocess.run for pgrep which blocks).
            with contextlib.suppress(Exception):
                await asyncio.to_thread(self._update_self_info)

            is_leader = self._is_leader()
            effective_leader = self._get_leader_peer()
            effective_leader_id = effective_leader.node_id if effective_leader else None
            last_known_leader_id = self.leader_id
            leader_id = effective_leader_id or last_known_leader_id

            # Collect peer info (dashboard-oriented shape)
            peers_info: list[dict[str, Any]] = []
            include_retired = request.query.get("include_retired") == "1"
            with self.peers_lock:
                peers_snapshot = dict(self.peers)
            for peer_id, peer in peers_snapshot.items():
                if getattr(peer, "retired", False) and not include_retired:
                    continue
                status = "offline" if not peer.is_alive() else "online"
                key = self._endpoint_key(peer)
                effective_scheme, effective_host, effective_port = (None, None, None)
                if key:
                    effective_scheme, effective_host, effective_port = key
                peers_info.append(
                    {
                        "node_id": peer_id,
                        "host": peer.host,
                        "port": peer.port,
                        "scheme": getattr(peer, "scheme", "http"),
                        "reported_host": getattr(peer, "reported_host", ""),
                        "reported_port": getattr(peer, "reported_port", 0),
                        "effective_scheme": effective_scheme,
                        "effective_host": effective_host,
                        "effective_port": effective_port,
                        "nat_blocked": bool(getattr(peer, "nat_blocked", False)),
                        "relay_via": getattr(peer, "relay_via", ""),
                        "role": peer.role.value if hasattr(peer.role, "value") else str(peer.role),
                        "version": getattr(peer, "version", ""),
                        "status": status,
                        "last_seen": peer.last_heartbeat,
                        "capabilities": list(peer.capabilities) if peer.capabilities else [],
                        "current_job": "",
                        "has_gpu": bool(peer.has_gpu),
                        "cpu_percent": peer.cpu_percent,
                        "memory_percent": peer.memory_percent,
                        "disk_percent": peer.disk_percent,
                        "gpu_percent": peer.gpu_percent,
                        "gpu_memory_percent": peer.gpu_memory_percent,
                        "selfplay_jobs": peer.selfplay_jobs,
                        "training_jobs": peer.training_jobs,
                    }
                )

            # Collect local job info
            with self.jobs_lock:
                jobs_snapshot = list(self.local_jobs.values())
            jobs_info: list[dict[str, Any]] = [
                {
                    "job_id": job.job_id,
                    "job_type": job.job_type.value if hasattr(job.job_type, "value") else str(job.job_type),
                    "status": job.status,
                    "node_id": job.node_id,
                    "board_type": job.board_type,
                    "num_players": job.num_players,
                    "engine_mode": job.engine_mode,
                    "pid": job.pid,
                    "started_at": job.started_at,
                }
                for job in jobs_snapshot
            ]

            # Collect training job info
            training_info: list[dict[str, Any]] = []
            with self.training_lock:
                for job_id, job in self.training_jobs.items():
                    training_info.append(
                        {
                            "job_id": job_id,
                            "job_type": job.job_type,
                            "status": job.status,
                            "board_type": job.board_type,
                            "num_players": job.num_players,
                            "assigned_worker": job.worker_node,
                            "created_at": job.created_at,
                            "started_at": job.started_at,
                            "completed_at": job.completed_at,
                            "output_model_path": job.output_model_path,
                            "error_message": job.error_message,
                        }
                    )

            # Collect data manifest info (lightweight dashboard summary)
            # NOTE: Never block on manifest collection here - use cached data only.
            with self.manifest_lock:
                local_manifest = self.local_data_manifest
                cluster_manifest = self.cluster_data_manifest

            manifest_info: dict[str, dict[str, Any]] = {}
            if cluster_manifest and getattr(cluster_manifest, "node_manifests", None):
                for node_id, node_manifest in cluster_manifest.node_manifests.items():
                    board_types = sorted(
                        {f.board_type for f in node_manifest.files if getattr(f, "board_type", "")}
                    )
                    manifest_info[node_id] = {
                        "game_count": node_manifest.selfplay_games,
                        "board_types": board_types,
                        "last_updated": node_manifest.collected_at,
                    }
            elif local_manifest:
                board_types = sorted(
                    {f.board_type for f in local_manifest.files if getattr(f, "board_type", "")}
                )
                manifest_info[local_manifest.node_id] = {
                    "game_count": local_manifest.selfplay_games,
                    "board_types": board_types,
                    "last_updated": local_manifest.collected_at,
                }

            voter_ids = list(getattr(self, "voter_node_ids", []) or [])
            # Jan 2, 2026: Use _count_alive_voters() to check IP:port matches
            voters_alive = self._count_alive_voters()

            self_payload = self.self_info.to_dict() if hasattr(self.self_info, "to_dict") else asdict(self.self_info)
            self_key = self._endpoint_key(self.self_info)
            if self_key:
                self_payload.update(
                    {
                        "effective_scheme": self_key[0],
                        "effective_host": self_key[1],
                        "effective_port": self_key[2],
                    }
                )

            return web.json_response({
                "success": True,
                "node_id": self.node_id,
                "role": self.role.value if hasattr(self.role, 'value') else str(self.role),
                "leader_id": leader_id,
                "effective_leader_id": effective_leader_id,
                "last_known_leader_id": last_known_leader_id,
                "is_leader": is_leader,
                "voter_node_ids": voter_ids,
                "voter_quorum_size": int(getattr(self, "voter_quorum_size", 0) or 0),
                "voters_alive": voters_alive,
                "voter_quorum_ok": self._has_voter_quorum(),
                "voter_config_source": str(getattr(self, "voter_config_source", "") or ""),
                "self": self_payload,
                "uptime_seconds": time.time() - self.start_time,
                "peers": peers_info,
                "peer_count": len(self.peers),
                "jobs": jobs_info,
                "job_count": len(jobs_info),
                "training_jobs": training_info,
                "training_job_count": len(training_info),
                "data_manifests": manifest_info,
                "timestamp": time.time(),
            })
        except Exception as e:  # noqa: BLE001
            return web.json_response({"success": False, "error": str(e)}, status=500)

    async def handle_api_cluster_git_update(self, request: web.Request) -> web.Response:
        """Leader-coordinated git updates for cluster nodes.

        Body (JSON):
            node_ids: list[str] | str (optional)
                If omitted, updates all known peers (online by default).
            include_self: bool (default False)
                If true and (node_ids omitted or includes this node_id), also update
                the leader node itself (performed last, triggers restart).
            include_offline: bool (default False)
                If true, attempt updates against offline peers as well.
            timeout_seconds: int (default 20, max 120)
                Per-peer request timeout.

        Notes:
            - This stops jobs and restarts orchestrators on nodes with updates
              available. Use with care.
        """
        try:
            if not self._is_leader() and request.query.get("local") != "1":
                return await self._proxy_to_leader(request)

            payload: dict[str, Any] = {}
            try:
                payload = await request.json()
            except (AttributeError):
                payload = {}

            node_ids_raw = payload.get("node_ids") or payload.get("nodes") or []
            node_ids: list[str] = []
            if isinstance(node_ids_raw, str):
                node_ids = [t.strip() for t in node_ids_raw.split(",") if t.strip()]
            elif isinstance(node_ids_raw, list):
                node_ids = [str(t).strip() for t in node_ids_raw if str(t).strip()]

            include_self = bool(payload.get("include_self", False))
            include_offline = bool(payload.get("include_offline", False))

            timeout_seconds = float(payload.get("timeout_seconds", 20) or 20)
            timeout_seconds = max(5.0, min(timeout_seconds, 120.0))

            with self.peers_lock:
                peers_by_id = dict(self.peers)

            targets: list[Any] = []  # NodeInfo type

            def should_include_peer(peer) -> bool:
                if peer.node_id == self.node_id:
                    return False
                return not (not include_offline and not peer.is_alive())

            if node_ids:
                for node_id in node_ids:
                    peer = peers_by_id.get(node_id)
                    if peer and should_include_peer(peer):
                        targets.append(peer)
            else:
                for peer in peers_by_id.values():
                    if should_include_peer(peer):
                        targets.append(peer)

            results: list[dict[str, Any]] = []
            timeout = ClientTimeout(total=timeout_seconds)
            async with get_client_session(timeout) as session:
                for peer in sorted(targets, key=lambda p: p.node_id):
                    peer_payload: dict[str, Any] = {
                        "node_id": peer.node_id,
                        "status": "online" if peer.is_alive() else "offline",
                        "success": False,
                        "attempted_urls": [],
                    }

                    if not include_offline and not peer.is_alive():
                        peer_payload["error"] = "offline"
                        results.append(peer_payload)
                        continue

                    last_error: str | None = None
                    for url in self._urls_for_peer(peer, "/git/update"):
                        peer_payload["attempted_urls"].append(url)
                        try:
                            async with session.post(url, json={}, headers=self._auth_headers()) as resp:
                                peer_payload["http_status"] = resp.status
                                try:
                                    data = await resp.json()
                                except (AttributeError):
                                    data = {"raw": await resp.text()}
                                peer_payload["response"] = data
                                if resp.status == 200:
                                    peer_payload["success"] = bool(data.get("success", True))
                                    break
                                last_error = (
                                    str(data.get("error") or "")
                                    or str(data.get("message") or "")
                                    or f"http_{resp.status}"
                                )
                        except Exception as exc:
                            last_error = str(exc)
                            continue

                    if last_error and not peer_payload.get("success"):
                        peer_payload["error"] = last_error

                    results.append(peer_payload)

            self_update: dict[str, Any] | None = None
            update_self = bool(include_self and (not node_ids or self.node_id in node_ids))
            if update_self:
                has_updates, local_commit, remote_commit = self._check_for_updates()
                if not has_updates:
                    self_update = {
                        "node_id": self.node_id,
                        "success": True,
                        "message": "Already up to date",
                        "local_commit": local_commit[:8] if local_commit else None,
                    }
                else:
                    success, message = await self._perform_git_update()
                    self_update = {
                        "node_id": self.node_id,
                        "success": success,
                        "message": message,
                        "old_commit": local_commit[:8] if local_commit else None,
                        "new_commit": remote_commit[:8] if remote_commit else None,
                    }
                    if success:
                        safe_create_task(self._restart_orchestrator(), name="cluster-restart-after-update")

            return web.json_response(
                {
                    "success": True,
                    "leader_id": self.node_id,
                    "updated_peers": results,
                    "self_update": self_update,
                    "timestamp": time.time(),
                }
            )
        except Exception as e:  # noqa: BLE001
            return web.json_response({"success": False, "error": str(e)}, status=500)

    async def handle_cluster_health(self, request: web.Request) -> web.Response:
        """Aggregate health status from all cluster nodes (leader-only).

        Phase 6: Health broadcasting - leader aggregates peer health data
        and reports unhealthy nodes for monitoring and alerting.
        """
        try:
            if not self._is_leader():
                return await self._proxy_to_leader(request)

            await asyncio.to_thread(self._update_self_info)

            # Collect health from all peers
            unhealthy_nodes = []
            code_version_mismatches = []
            disk_warnings = []
            memory_warnings = []
            nfs_issues = []

            my_version = self.build_version

            with self.peers_lock:
                peers_snapshot = list(self.peers.values())

            for peer in peers_snapshot:
                # Check for health issues
                issues = peer.get_health_issues()
                if issues:
                    unhealthy_nodes.append({
                        "node_id": peer.node_id,
                        "issues": [{"code": code, "description": desc} for code, desc in issues],
                    })

                # Check code version mismatch
                if peer.code_version and peer.code_version != my_version:
                    code_version_mismatches.append({
                        "node_id": peer.node_id,
                        "version": peer.code_version,
                        "leader_version": my_version,
                    })

                # Check disk warnings
                if peer.disk_percent >= 85:  # DISK_PRODUCTION_HALT_PERCENT from app.config.thresholds
                    disk_warnings.append({
                        "node_id": peer.node_id,
                        "disk_percent": peer.disk_percent,
                        "disk_free_gb": peer.disk_free_gb,
                    })

                # Check memory warnings
                if peer.memory_percent >= 85:
                    memory_warnings.append({
                        "node_id": peer.node_id,
                        "memory_percent": peer.memory_percent,
                    })

                # Check NFS issues
                if not peer.nfs_accessible:
                    nfs_issues.append({
                        "node_id": peer.node_id,
                    })

            # Calculate cluster health summary
            total_nodes = len(peers_snapshot) + 1  # Include self
            healthy_nodes = total_nodes - len(unhealthy_nodes)
            cluster_health_pct = (healthy_nodes / total_nodes * 100) if total_nodes > 0 else 0

            return web.json_response({
                "success": True,
                "timestamp": time.time(),
                "leader_id": self.node_id,
                "leader_version": my_version,
                "cluster_health": {
                    "total_nodes": total_nodes,
                    "healthy_nodes": healthy_nodes,
                    "unhealthy_nodes": len(unhealthy_nodes),
                    "health_percent": cluster_health_pct,
                },
                "issues": {
                    "unhealthy_nodes": unhealthy_nodes,
                    "code_version_mismatches": code_version_mismatches,
                    "disk_warnings": disk_warnings,
                    "memory_warnings": memory_warnings,
                    "nfs_issues": nfs_issues,
                },
            })
        except Exception as e:  # noqa: BLE001
            logger.error(f"Cluster health check error: {e}")
            return web.json_response({"error": str(e)}, status=500)
