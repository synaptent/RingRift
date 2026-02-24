"""Admin HTTP Handlers Mixin.

Provides HTTP endpoints for cluster administration and git repository management.
Supports remote code updates, health checks, and node configuration.

Usage:
    class P2POrchestrator(AdminHandlersMixin, ...):
        pass

Endpoints:
    GET /git/status - Get git status (local/remote commit, updates available)
    POST /git/update - Trigger git pull on this node
    POST /git/update-cluster - Trigger git pull on all cluster nodes
    GET /admin/health - Deep health check with subsystem status
    POST /admin/restart - Restart the P2P orchestrator
    GET /admin/config - Get current configuration settings

Auto-Update:
    When RINGRIFT_AUTO_UPDATE=true, nodes automatically pull changes
    when the leader detects new commits on the remote branch.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import subprocess
import time
from typing import TYPE_CHECKING, Any

from aiohttp import web

from app.core.async_context import safe_create_task
from scripts.p2p.handlers.base import BaseP2PHandler
from scripts.p2p.handlers.timeout_decorator import (
    handler_timeout,
    HANDLER_TIMEOUT_ADMIN,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Import constants
try:
    from scripts.p2p.constants import AUTO_UPDATE_ENABLED
except ImportError:
    AUTO_UPDATE_ENABLED = False

# Jan 22, 2026: Import canonical HTTP timeout for cross-cloud requests
try:
    from app.p2p.constants import HTTP_TOTAL_TIMEOUT
except ImportError:
    HTTP_TOTAL_TIMEOUT = 45  # Fallback to match constants.py default


class AdminHandlersMixin(BaseP2PHandler):
    """Mixin providing admin and git HTTP handlers.

    Inherits from BaseP2PHandler for consistent response formatting.

    Requires the implementing class to have:
    - ringrift_path: str
    - _get_local_git_commit() method
    - _get_local_git_branch() method
    - _check_local_changes() method
    - _check_for_updates() method
    - _get_commits_behind() method
    - _perform_git_update() method
    - _restart_orchestrator() method
    """

    # Type hints for IDE support
    ringrift_path: str

    @handler_timeout(HANDLER_TIMEOUT_ADMIN)
    async def handle_git_status(self, request: web.Request) -> web.Response:
        """Get git status for this node.

        Returns local/remote commit info and whether updates are available.
        """
        try:
            local_commit = self._get_local_git_commit()
            local_branch = self._get_local_git_branch()
            has_local_changes = self._check_local_changes()

            # Check for remote updates (this does a git fetch)
            has_updates, _, remote_commit = self._check_for_updates()
            commits_behind = 0
            if has_updates and local_commit and remote_commit:
                commits_behind = self._get_commits_behind(local_commit, remote_commit)

            return self.json_response({
                "local_commit": local_commit[:8] if local_commit else None,
                "local_commit_full": local_commit,
                "local_branch": local_branch,
                "remote_commit": remote_commit[:8] if remote_commit else None,
                "remote_commit_full": remote_commit,
                "has_updates": has_updates,
                "commits_behind": commits_behind,
                "has_local_changes": has_local_changes,
                "auto_update_enabled": AUTO_UPDATE_ENABLED,
                "ringrift_path": self.ringrift_path,
            })
        except Exception as e:
            return self.error_response(str(e), status=500)

    @handler_timeout(HANDLER_TIMEOUT_ADMIN)
    async def handle_git_update(self, request: web.Request) -> web.Response:
        """Manually trigger a git update on this node.

        This will stop jobs, pull updates, and restart the orchestrator.

        Dec 2025: Added optional auth token check for security.
        Set RINGRIFT_ADMIN_TOKEN to require auth, or leave unset for open access.
        """
        import os
        admin_token = os.environ.get("RINGRIFT_ADMIN_TOKEN")
        if admin_token:
            request_token = request.headers.get("X-Admin-Token", "")
            if request_token != admin_token:
                logger.warning("Unauthorized git update attempt")
                return self.error_response("Unauthorized", status=401)

        try:
            # Check for updates first
            has_updates, local_commit, remote_commit = self._check_for_updates()

            if not has_updates:
                return self.json_response({
                    "success": True,
                    "message": "Already up to date",
                    "local_commit": local_commit[:8] if local_commit else None,
                })

            # Perform the update
            success, message = await self._perform_git_update()

            if success:
                # Schedule restart
                safe_create_task(self._restart_orchestrator(), name="admin-restart-after-update")
                return self.json_response({
                    "success": True,
                    "message": "Update successful, restarting...",
                    "old_commit": local_commit[:8] if local_commit else None,
                    "new_commit": remote_commit[:8] if remote_commit else None,
                })
            else:
                return self.json_response({
                    "success": False,
                    "message": message,
                }, status=400)

        except Exception as e:
            return self.error_response(str(e), status=500)

    @handler_timeout(HANDLER_TIMEOUT_ADMIN)
    async def handle_admin_restart(self, request: web.Request) -> web.Response:
        """Force restart the orchestrator process.

        Useful after code updates when /git/update shows "already up to date"
        but the running process hasn't picked up the changes.

        Dec 2025: Added optional auth token check for security.
        Set RINGRIFT_ADMIN_TOKEN to require auth, or leave unset for open access.
        """
        import os
        admin_token = os.environ.get("RINGRIFT_ADMIN_TOKEN")
        if admin_token:
            request_token = request.headers.get("X-Admin-Token", "")
            if request_token != admin_token:
                logger.warning("Unauthorized admin restart attempt")
                return self.error_response("Unauthorized", status=401)

        try:
            logger.info("Admin restart requested via API")
            # Schedule restart (gives time to return response)
            safe_create_task(self._restart_orchestrator(), name="admin-restart-manual")
            return self.json_response({
                "success": True,
                "message": "Restart scheduled, process will restart in 2 seconds",
            })
        except Exception as e:
            return self.error_response(str(e), status=500)

    # =========================================================================
    # Peer Administration (Phase 8 - Dec 28, 2025)
    # =========================================================================

    @handler_timeout(HANDLER_TIMEOUT_ADMIN)
    async def handle_purge_retired_peers(self, request: web.Request) -> web.Response:
        """Purge retired peers from the cluster registry.

        Removes peers that have been marked as retired (dead/terminated instances)
        to clean up the peer list. This endpoint is unauthenticated for ease of
        admin access; it only cleans up stale entries, not active nodes.
        """
        try:
            # Import here to avoid circular imports
            from scripts.p2p.network import NonBlockingAsyncLockWrapper

            async with NonBlockingAsyncLockWrapper(self.peers_lock, "peers_lock", timeout=5.0):
                retired_peers = [
                    node_id for node_id, info in self.peers.items()
                    if getattr(info, "retired", False)
                ]

                if not retired_peers:
                    return web.json_response({
                        "success": True,
                        "purged_count": 0,
                        "message": "No retired peers to purge",
                    })

                for node_id in retired_peers:
                    del self.peers[node_id]
                    logger.info(f"Purged retired peer: {node_id}")

                logger.info(f"Purged {len(retired_peers)} retired peers")

            return web.json_response({
                "success": True,
                "purged_count": len(retired_peers),
                "purged_peers": retired_peers,
            })
        except Exception as e:  # noqa: BLE001
            return web.json_response({"error": str(e)}, status=500)

    @handler_timeout(HANDLER_TIMEOUT_ADMIN)
    async def handle_purge_stale_peers(self, request: web.Request) -> web.Response:
        """Purge stale peers based on heartbeat age.

        This is more aggressive than purge_retired - it removes any peer
        that hasn't sent a heartbeat in the specified threshold (default 1 hour).

        Query params:
            max_age: Maximum heartbeat age in seconds (default: 3600)
            dry_run: If 1, just report what would be purged without deleting
        """
        try:
            from scripts.p2p.network import NonBlockingAsyncLockWrapper

            max_age = int(request.query.get("max_age", "3600"))
            dry_run = request.query.get("dry_run", "0") == "1"
            now = time.time()

            stale_peers = []
            async with NonBlockingAsyncLockWrapper(self.peers_lock, "peers_lock", timeout=5.0):
                for node_id, info in self.peers.items():
                    if node_id == self.node_id:
                        continue  # Don't purge self
                    last_hb = getattr(info, "last_heartbeat", 0.0) or 0.0
                    age = now - last_hb
                    if age >= max_age:
                        stale_peers.append({
                            "node_id": node_id,
                            "age_seconds": int(age),
                            "last_heartbeat": last_hb,
                            "role": str(getattr(info, "role", "unknown")),
                            "nat_blocked": getattr(info, "nat_blocked", False),
                        })

            if not stale_peers:
                return web.json_response({
                    "success": True,
                    "purged_count": 0,
                    "message": f"No peers older than {max_age}s found",
                })

            purged_ids = []
            if not dry_run:
                async with NonBlockingAsyncLockWrapper(self.peers_lock, "peers_lock", timeout=5.0):
                    for peer in stale_peers:
                        node_id = peer["node_id"]
                        if node_id in self.peers:
                            del self.peers[node_id]
                            purged_ids.append(node_id)
                            logger.info(f"Purged stale peer: {node_id} (no heartbeat for {peer['age_seconds']}s)")

            return web.json_response({
                "success": True,
                "purged_count": len(purged_ids) if not dry_run else 0,
                "would_purge_count": len(stale_peers),
                "dry_run": dry_run,
                "max_age_seconds": max_age,
                "stale_peers": stale_peers,
                "purged_peers": purged_ids,
            })
        except Exception as e:  # noqa: BLE001
            return web.json_response({"error": str(e)}, status=500)

    @handler_timeout(HANDLER_TIMEOUT_ADMIN)
    async def handle_admin_unretire(self, request: web.Request) -> web.Response:
        """Unretire a specific peer node.

        This endpoint allows external systems (like vast_p2p_sync.py) to
        programmatically unretire nodes that are known to be active but were
        marked as retired due to temporary connectivity issues.

        Query params:
            node_id: The node ID to unretire (required)

        Returns:
            JSON with success status and node info
        """
        try:
            from scripts.p2p.network import NonBlockingAsyncLockWrapper

            node_id = request.query.get("node_id", "").strip()
            if not node_id:
                return web.json_response({
                    "error": "node_id parameter is required"
                }, status=400)

            async with NonBlockingAsyncLockWrapper(self.peers_lock, "peers_lock", timeout=5.0):
                if node_id not in self.peers:
                    # List available nodes for debugging
                    available = list(self.peers.keys())
                    return web.json_response({
                        "error": f"Node '{node_id}' not found in peer registry",
                        "available_nodes": available[:20],  # Limit to first 20
                        "total_nodes": len(available),
                    }, status=404)

                peer_info = self.peers[node_id]
                was_retired = getattr(peer_info, "retired", False)

                if not was_retired:
                    return web.json_response({
                        "success": True,
                        "message": f"Node '{node_id}' was not retired",
                        "already_active": True,
                    })

                # Unretire the node
                peer_info.retired = False
                peer_info.retired_at = 0.0

                # Also reset failure counters to give it a fresh start
                peer_info.consecutive_failures = 0
                peer_info.last_failure_time = 0.0

                logger.info(f"Unretired peer: {node_id} (admin request)")

                # Emit HOST_ONLINE event so SelfplayScheduler/SyncRouter detect recovered node
                capabilities = []
                if getattr(peer_info, "has_gpu", False):
                    gpu_type = getattr(peer_info, "gpu_type", "") or "gpu"
                    capabilities.append(gpu_type)
                else:
                    capabilities.append("cpu")

            # Emit event outside of lock
            if hasattr(self, "_emit_host_online"):
                await self._emit_host_online(node_id, capabilities)

            return web.json_response({
                "success": True,
                "message": f"Node '{node_id}' has been unretired",
                "node_id": node_id,
                "host": getattr(peer_info, "host", ""),
                "gpu_name": getattr(peer_info, "gpu_name", ""),
            })
        except Exception as e:  # noqa: BLE001
            return web.json_response({"error": str(e)}, status=500)

    @handler_timeout(HANDLER_TIMEOUT_ADMIN)
    async def handle_admin_reset_node_jobs(self, request: web.Request) -> web.Response:
        """Reset job counts for a specific node (for zombie cleanup).

        POST /admin/reset_node_jobs with JSON body:
            {"node_id": "node-id-to-reset"}

        Leader-only: Resets selfplay_jobs and training_jobs to 0 for a node.
        Use when zombie processes have been killed and job counts are stale.

        Returns:
            JSON with success status and updated node info
        """
        try:
            from scripts.p2p.network import NonBlockingAsyncLockWrapper
            from scripts.p2p.types import NodeRole

            # Leader-only endpoint
            if self.role != NodeRole.LEADER:
                return web.json_response({
                    "error": "This endpoint is only available on the cluster leader"
                }, status=403)

            data = await request.json()
            node_id = data.get("node_id", "").strip()
            if not node_id:
                return web.json_response({
                    "error": "node_id is required in request body"
                }, status=400)

            async with NonBlockingAsyncLockWrapper(self.peers_lock, "peers_lock", timeout=5.0):
                if node_id not in self.peers:
                    available = list(self.peers.keys())
                    return web.json_response({
                        "error": f"Node '{node_id}' not found in peer registry",
                        "available_nodes": available[:20],
                        "total_nodes": len(available),
                    }, status=404)

                peer_info = self.peers[node_id]

                # Get old values for logging
                old_selfplay = getattr(peer_info, "selfplay_jobs", 0)
                old_training = getattr(peer_info, "training_jobs", 0)

                # Reset job counts
                peer_info.selfplay_jobs = 0
                peer_info.training_jobs = 0

                logger.info(
                    f"Reset job counts for {node_id}: "
                    f"selfplay {old_selfplay}->0, training {old_training}->0 (admin request)"
                )

            return web.json_response({
                "success": True,
                "node_id": node_id,
                "message": f"Reset job counts for '{node_id}'",
                "previous_selfplay_jobs": old_selfplay,
                "previous_training_jobs": old_training,
                "current_selfplay_jobs": 0,
                "current_training_jobs": 0,
            })
        except Exception as e:  # noqa: BLE001
            logger.error(f"Error in handle_admin_reset_node_jobs: {e}")
            return web.json_response({"error": str(e)}, status=500)

    @handler_timeout(HANDLER_TIMEOUT_ADMIN)
    async def handle_admin_add_peer(self, request: web.Request) -> web.Response:
        """Add a peer to this node's peer list without restart.

        This endpoint enables partition healing - external systems can inject
        peers from other partitions to create bridges for gossip propagation.

        January 2026: Added for partition_healer.py integration.

        Request body:
            node_id: The node ID to add
            address: The IP/hostname of the peer
            port: The P2P port (default: 8770)

        Returns:
            JSON with success status and peer info
        """
        try:
            from scripts.p2p.network import NonBlockingAsyncLockWrapper

            data = await request.json()
            node_id = data.get("node_id", "").strip()
            address = data.get("address", "").strip()
            port = int(data.get("port", 8770))

            if not node_id:
                return web.json_response({
                    "error": "node_id is required in request body"
                }, status=400)

            if not address:
                return web.json_response({
                    "error": "address is required in request body"
                }, status=400)

            # Check if peer already exists
            async with NonBlockingAsyncLockWrapper(self.peers_lock, "peers_lock", timeout=5.0):
                already_exists = node_id in self.peers

            if already_exists:
                return web.json_response({
                    "success": True,
                    "node_id": node_id,
                    "message": f"Peer '{node_id}' already exists",
                    "already_existed": True,
                })

            # Try to connect to the peer to validate it's reachable
            import aiohttp
            peer_url = f"http://{address}:{port}/health"
            is_reachable = False
            peer_info_from_health = {}

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(peer_url, timeout=aiohttp.ClientTimeout(total=HTTP_TOTAL_TIMEOUT)) as resp:
                        if resp.status == 200:
                            is_reachable = True
                            try:
                                peer_info_from_health = await resp.json()
                            except Exception:
                                pass
            except Exception as e:
                logger.debug(f"Could not reach peer {node_id} at {address}:{port}: {e}")

            # Add peer to our list (even if not immediately reachable - gossip will sync)
            try:
                from scripts.p2p.models import NodeInfo
                from scripts.p2p.types import NodeRole

                new_peer = NodeInfo(
                    node_id=node_id,
                    host=address,
                    port=port,
                    scheme="http",
                    role=NodeRole.FOLLOWER,
                    last_heartbeat=time.time() if is_reachable else 0.0,
                    leader_id=peer_info_from_health.get("leader_id"),
                )

                async with NonBlockingAsyncLockWrapper(self.peers_lock, "peers_lock", timeout=5.0):
                    self.peers[node_id] = new_peer
                    logger.info(f"Added peer {node_id} at {address}:{port} (reachable={is_reachable})")

            except ImportError:
                # Fallback if models not available - just add raw dict
                async with NonBlockingAsyncLockWrapper(self.peers_lock, "peers_lock", timeout=5.0):
                    self.peers[node_id] = {
                        "node_id": node_id,
                        "host": address,
                        "port": port,
                        "last_heartbeat": time.time() if is_reachable else 0.0,
                    }
                    logger.info(f"Added peer {node_id} at {address}:{port} (dict mode)")

            return web.json_response({
                "success": True,
                "node_id": node_id,
                "address": address,
                "port": port,
                "is_reachable": is_reachable,
                "already_existed": False,
                "message": f"Peer '{node_id}' added successfully",
            })

        except Exception as e:  # noqa: BLE001
            logger.error(f"Error in handle_admin_add_peer: {e}")
            return web.json_response({"error": str(e)}, status=500)

    @handler_timeout(HANDLER_TIMEOUT_ADMIN)
    async def handle_admin_ping_work(self, request: web.Request) -> web.Response:
        """Respond to work acceptance probe for frozen leader detection.

        January 2, 2026: This endpoint is probed by voter nodes to verify
        the leader's event loop is responsive. A frozen leader may still
        respond to /status (heartbeat) but fail to process this request
        because its event loop is stuck.

        This is a lightweight endpoint that requires the event loop to:
        1. Accept the HTTP request
        2. Parse JSON body
        3. Create a simple async task
        4. Return response

        If the leader is frozen (deadlock, long sync, etc.), this will timeout
        while /status may still respond due to buffered TCP responses.

        Request body:
            probe_id: Unique probe identifier
            prober_node: Node ID of the prober
            timestamp: Probe timestamp

        Returns:
            JSON with pong response and processing time
        """
        start_time = time.time()
        try:
            data = await request.json()
            probe_id = data.get("probe_id", "unknown")
            prober_node = data.get("prober_node", "unknown")

            # Perform a minimal async operation to verify event loop is responsive
            # This creates a task and awaits it, which requires the event loop
            async def _verify_loop() -> bool:
                await asyncio.sleep(0)  # Yield to event loop
                return True

            loop_ok = await asyncio.wait_for(_verify_loop(), timeout=1.0)

            processing_time_ms = (time.time() - start_time) * 1000

            logger.debug(
                f"Ping work from {prober_node} (probe_id={probe_id}): "
                f"loop_ok={loop_ok}, processing_time={processing_time_ms:.1f}ms"
            )

            return web.json_response({
                "success": True,
                "pong": True,
                "probe_id": probe_id,
                "responder_node": getattr(self, "node_id", "unknown"),
                "processing_time_ms": processing_time_ms,
                "loop_responsive": loop_ok,
                "timestamp": time.time(),
            })

        except asyncio.TimeoutError:
            logger.warning(f"Event loop timeout in ping_work - possible frozen leader")
            return web.json_response({
                "success": False,
                "error": "event_loop_timeout",
                "processing_time_ms": (time.time() - start_time) * 1000,
            }, status=503)
        except Exception as e:  # noqa: BLE001
            logger.error(f"Error in handle_admin_ping_work: {e}")
            return web.json_response({
                "success": False,
                "error": str(e),
            }, status=500)

    @handler_timeout(HANDLER_TIMEOUT_ADMIN)
    async def handle_admin_clear_nat_blocked(self, request: web.Request) -> web.Response:
        """Clear NAT-blocked status on all or specific peers.

        Jan 2, 2026: Added to enable leader election when all voters are
        marked as nat_blocked due to voter heartbeat loop not finding peers
        by their IP:port keys.

        Query params:
            node_id: Optional. If specified, only clear for this node.
                     Otherwise clears all peers.
            voters_only: Optional. If "true", only clear for voters.

        Returns:
            JSON with count of cleared peers.
        """
        try:
            from scripts.p2p.network import NonBlockingAsyncLockWrapper

            node_id = request.query.get("node_id", "").strip()
            voters_only = request.query.get("voters_only", "").lower() == "true"

            cleared = []
            voter_ids = list(getattr(self, "voter_node_ids", []) or [])
            voter_ip_map = {}

            # Build IP to voter_id mapping for matching
            if voters_only and hasattr(self, "_build_voter_ip_mapping"):
                voter_ip_mapping = self._build_voter_ip_mapping()
                for vid, ips in voter_ip_mapping.items():
                    for ip in ips:
                        voter_ip_map[ip] = vid

            async with NonBlockingAsyncLockWrapper(self.peers_lock, "peers_lock", timeout=5.0):
                for peer_key, peer_info in list(self.peers.items()):
                    # Skip if filtering by specific node_id
                    if node_id and peer_key != node_id:
                        # Also try matching by IP for IP:port format keys
                        if ":" in peer_key:
                            peer_ip = peer_key.split(":")[0]
                            resolved = voter_ip_map.get(peer_ip, peer_key)
                            if resolved != node_id:
                                continue
                        else:
                            continue

                    # Skip if filtering by voters only
                    if voters_only:
                        is_voter = peer_key in voter_ids
                        if not is_voter and ":" in peer_key:
                            peer_ip = peer_key.split(":")[0]
                            is_voter = peer_ip in voter_ip_map
                        if not is_voter:
                            continue

                    # Clear NAT-blocked status
                    was_blocked = getattr(peer_info, "nat_blocked", False)
                    if was_blocked:
                        peer_info.nat_blocked = False
                        peer_info.nat_blocked_since = 0.0
                        peer_info.consecutive_failures = 0
                        cleared.append({
                            "peer_key": peer_key,
                            "node_id": getattr(peer_info, "node_id", peer_key),
                        })

            logger.info(f"Cleared NAT-blocked status on {len(cleared)} peers (admin request)")

            return web.json_response({
                "success": True,
                "cleared_count": len(cleared),
                "cleared_peers": cleared[:50],  # Limit for response size
                "voters_only": voters_only,
                "specific_node_id": node_id if node_id else None,
            })

        except Exception as e:  # noqa: BLE001
            logger.error(f"Error in handle_admin_clear_nat_blocked: {e}")
            return web.json_response({"error": str(e)}, status=500)

    @handler_timeout(HANDLER_TIMEOUT_ADMIN)
    async def handle_admin_deduplicate(self, request: web.Request) -> web.Response:
        """Trigger peer deduplication to remove duplicate node entries.

        Jan 13, 2026: Added to clean up duplicate peer entries where the same
        physical machine registered multiple times with different node_ids
        (e.g., after hostname changes or P2P restarts).

        Keeps the most recently active node per IP address (by last_heartbeat).

        Query params:
            dry_run: If "1", just report what would be removed without deleting

        Returns:
            JSON with deduplication results including removed peers.
        """
        try:
            dry_run = request.query.get("dry_run", "0") == "1"

            # Call the deduplication method
            if hasattr(self, "_deduplicate_peers"):
                if dry_run:
                    # For dry run, simulate deduplication without actually removing
                    from collections import defaultdict
                    ip_to_nodes: dict[str, list[tuple[str, float]]] = defaultdict(list)

                    with self.peers_lock:
                        for node_id, peer in self.peers.items():
                            if node_id == self.node_id:
                                continue
                            dedup_key = getattr(peer, "reported_host", None) or getattr(peer, "effective_host", None) or getattr(peer, "host", None)
                            if dedup_key and dedup_key not in ("127.0.0.1", ""):
                                freshness = getattr(peer, "last_heartbeat", 0) or getattr(peer, "last_seen", 0) or 0
                                ip_to_nodes[dedup_key].append((node_id, freshness))

                    duplicates = []
                    for ip, nodes in ip_to_nodes.items():
                        if len(nodes) > 1:
                            nodes.sort(key=lambda x: x[1], reverse=True)
                            keeper = nodes[0][0]
                            stale = [n[0] for n in nodes[1:]]
                            duplicates.append({
                                "ip": ip,
                                "keeping": keeper,
                                "would_remove": stale,
                            })

                    return web.json_response({
                        "success": True,
                        "dry_run": True,
                        "duplicate_groups": len(duplicates),
                        "would_remove_count": sum(len(d["would_remove"]) for d in duplicates),
                        "duplicates": duplicates,
                    })
                else:
                    # Actually run deduplication
                    # Jan 30, 2026: Use network orchestrator directly
                    removed_count = self.network.deduplicate_peers()
                    return web.json_response({
                        "success": True,
                        "dry_run": False,
                        "removed_count": removed_count,
                    })
            else:
                return web.json_response({
                    "error": "network.deduplicate_peers method not found",
                }, status=500)

        except Exception as e:  # noqa: BLE001
            logger.error(f"Error in handle_admin_deduplicate: {e}")
            return web.json_response({"error": str(e)}, status=500)

    @handler_timeout(HANDLER_TIMEOUT_ADMIN)
    async def handle_process_kill(self, request: web.Request) -> web.Response:
        """Kill processes matching a pattern on this node.

        Jan 21, 2026: Added to fix zombie process accumulation.
        This endpoint enables remote cleanup of stuck/zombie processes.
        Jan 28, 2026: Moved from p2p_orchestrator.py to AdminHandlersMixin.

        Request JSON:
            pattern: str - Process pattern to match (e.g., "selfplay", "gpu_selfplay")
            signal: str - Signal to send (default: "SIGKILL")

        Returns:
            JSON with killed count and details
        """
        try:
            data = await request.json()
            pattern = data.get("pattern", "")
            signal_name = data.get("signal", "SIGKILL").upper()

            if not pattern:
                return self.json_response(
                    {"error": "missing pattern", "killed": 0},
                    status=400,
                )

            # Validate signal
            signal_map = {
                "SIGTERM": "-15",
                "SIGKILL": "-9",
                "SIGHUP": "-1",
            }
            signal_flag = signal_map.get(signal_name, "-9")

            # Safety: only allow certain patterns to prevent accidents
            allowed_patterns = [
                "selfplay", "gpu_selfplay", "gumbel", "train", "gauntlet",
                "tournament", "export", "run_self_play", "run_gpu_selfplay",
                "run_hybrid_selfplay", "policy_only", "nnue",
            ]
            if not any(allowed in pattern.lower() for allowed in allowed_patterns):
                return self.json_response(
                    {"error": f"pattern not allowed: {pattern}", "killed": 0},
                    status=403,
                )

            if not shutil.which("pgrep") or not shutil.which("pkill"):
                return self.json_response(
                    {"error": "pgrep/pkill not available", "killed": 0},
                    status=500,
                )

            # Count matching processes first
            try:
                pgrep_result = await asyncio.to_thread(
                    subprocess.run,
                    ["pgrep", "-f", pattern],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if pgrep_result.returncode == 0 and pgrep_result.stdout.strip():
                    pids = [p.strip() for p in pgrep_result.stdout.strip().split() if p.strip()]
                    pid_count = len(pids)
                else:
                    pid_count = 0
                    pids = []
            except subprocess.TimeoutExpired:
                return self.json_response(
                    {"error": "pgrep timeout", "killed": 0},
                    status=500,
                )

            if pid_count == 0:
                return self.json_response({
                    "killed": 0,
                    "pattern": pattern,
                    "message": "no matching processes",
                })

            # Kill matching processes
            try:
                pkill_result = await asyncio.to_thread(
                    subprocess.run,
                    ["pkill", signal_flag, "-f", pattern],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                # pkill returns 0 if processes were killed, 1 if none matched
                killed = pid_count if pkill_result.returncode == 0 else 0
            except subprocess.TimeoutExpired:
                return self.json_response(
                    {"error": "pkill timeout", "killed": 0},
                    status=500,
                )

            logger.info(
                f"[ProcessKill] Killed {killed} processes matching '{pattern}' "
                f"with {signal_name} (pids: {pids[:10]}{'...' if len(pids) > 10 else ''})"
            )

            return self.json_response({
                "killed": killed,
                "pattern": pattern,
                "signal": signal_name,
                "pids": pids[:20],  # Limit returned PIDs
            })

        except json.JSONDecodeError:
            return self.json_response(
                {"error": "invalid JSON", "killed": 0},
                status=400,
            )
        except Exception as e:  # noqa: BLE001
            logger.error(f"[ProcessKill] Error: {e}")
            return self.json_response(
                {"error": str(e), "killed": 0},
                status=500,
            )
