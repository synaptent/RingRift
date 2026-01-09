"""Configuration HTTP handlers for P2P orchestrator.

January 2026 - P2P Modularization Phase 2c

This mixin provides HTTP handlers for node registration, config synchronization,
and connectivity diagnostics.

Must be mixed into a class that provides:
- self.is_leader: bool
- self.leader_id: str
- self.node_id: str
- self.peers_lock: threading.Lock
- self.peers: dict
- self.hybrid_transport: Optional[HybridTransport]
- self._send_heartbeat_to_peer(host, port) -> dict
"""
from __future__ import annotations

import logging

from aiohttp import web

logger = logging.getLogger(__name__)

# Try to import dynamic registry
try:
    from app.coordination.dynamic_registry import get_registry
    HAS_DYNAMIC_REGISTRY = True
except ImportError:
    HAS_DYNAMIC_REGISTRY = False
    get_registry = None  # type: ignore

# Try to import hybrid transport
try:
    from scripts.p2p.hybrid_transport import diagnose_node_connectivity
    HAS_HYBRID_TRANSPORT = True
except ImportError:
    HAS_HYBRID_TRANSPORT = False
    diagnose_node_connectivity = None  # type: ignore


class ConfigurationHandlersMixin:
    """Mixin providing configuration and registration HTTP handlers.

    Endpoints:
    - POST /register - Node self-registration for dynamic IP updates
    - POST /push_config - Push current config to all cluster nodes
    - GET /connectivity/diagnose/{node_id} - Diagnose connectivity to a specific node
    """

    async def handle_register(self, request: web.Request) -> web.Response:
        """POST /register - Node self-registration for dynamic IP updates.

        Nodes call this endpoint to announce their current IP address.
        Useful when Vast.ai instances restart and get new IPs.

        Request body:
        {
            "node_id": "vast-5090-quad",
            "host": "211.72.13.202",
            "port": 45875,
            "vast_instance_id": "28654132"  // optional
        }
        """
        if not HAS_DYNAMIC_REGISTRY:
            return web.json_response({
                "error": "Dynamic registry not available"
            }, status=501)

        try:
            data = await request.json()
            node_id = data.get("node_id")
            host = data.get("host")
            port = data.get("port", 22)
            vast_instance_id = data.get("vast_instance_id")
            tailscale_ip = data.get("tailscale_ip")

            if not node_id or not host:
                return web.json_response({
                    "error": "Missing required fields: node_id, host"
                }, status=400)

            registry = get_registry()
            success = registry.register_node(node_id, host, port, vast_instance_id, tailscale_ip=tailscale_ip)

            if success:
                logger.info(f"Node registered: {node_id} at {host}:{port}")
                return web.json_response({
                    "success": True,
                    "node_id": node_id,
                    "registered_host": host,
                    "registered_port": port,
                })
            else:
                return web.json_response({
                    "error": "Registration failed"
                }, status=500)

        except Exception as e:  # noqa: BLE001
            return web.json_response({"error": str(e)}, status=500)

    async def handle_push_config(self, request: web.Request) -> web.Response:
        """POST /push_config - Push current config to all cluster nodes.

        December 30, 2025: Added as part of distributed config sync infrastructure.
        Only the leader can push config to prevent conflicting updates.

        When called:
        1. Force reload local config to ensure we have latest
        2. Broadcast CONFIG_UPDATED event via gossip
        3. Nodes that receive the event will pull new config

        Returns:
            JSON response with status, config hash, and peer count
        """
        # Only leader can push config to prevent conflicting updates
        if not self.is_leader:
            return web.json_response({
                "error": "Only leader can push config",
                "leader_id": self.leader_id,
            }, status=403)

        try:
            # Force reload local config
            try:
                from app.config.cluster_config import get_config_cache, get_config_version

                cache = get_config_cache()
                cache.get_config(force_reload=True)
                version = get_config_version()
            except ImportError as e:
                return web.json_response({
                    "error": f"Config cache not available: {e}",
                }, status=500)
            except (OSError, ValueError) as e:
                return web.json_response({
                    "error": f"Config reload failed: {e}",
                }, status=500)

            # Broadcast CONFIG_UPDATED event via gossip
            peers_notified = 0
            try:
                from app.distributed.data_events import DataEventType, emit_data_event

                event_payload = {
                    "source_node": self.node_id,
                    "hash": version.content_hash,
                    "timestamp": version.timestamp,
                    "force_sync": True,  # Nodes should pull immediately
                }

                emit_data_event(
                    event_type=DataEventType.CONFIG_UPDATED,
                    payload=event_payload,
                    source="P2POrchestrator.push_config",
                )

                # Count alive peers
                with self.peers_lock:
                    peers_notified = sum(
                        1 for p in self.peers.values()
                        if p.get("status") == "alive"
                    )

                logger.info(
                    f"[P2POrchestrator] Config pushed: hash={version.content_hash}, "
                    f"peers_notified={peers_notified}"
                )

            except (ImportError, RuntimeError, OSError) as e:
                logger.warning(f"[P2POrchestrator] Event emission failed: {e}")
                # Still return success since config was reloaded locally

            return web.json_response({
                "status": "pushed",
                "config_hash": version.content_hash,
                "config_timestamp": version.timestamp,
                "peers_notified": peers_notified,
                "source_node": self.node_id,
            })

        except Exception as e:  # noqa: BLE001
            logger.error(f"[P2POrchestrator] push_config failed: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def handle_connectivity_diagnose(self, request: web.Request) -> web.Response:
        """GET /connectivity/diagnose/{node_id} - Diagnose connectivity to a specific node.

        Probes HTTP, Tailscale, and SSH transports and returns latency/reachability
        for each. Helps identify the best transport for communicating with a node.
        """
        node_id = request.match_info.get("node_id", "")
        if not node_id:
            return web.json_response({"error": "node_id required"}, status=400)

        # Find the node's address
        with self.peers_lock:
            peer = self.peers.get(node_id)

        if not peer:
            return web.json_response({
                "error": f"Node {node_id} not found in peers",
                "known_peers": list(self.peers.keys()),
            }, status=404)

        if not HAS_HYBRID_TRANSPORT or not self.hybrid_transport:
            # Fallback: just check if we can reach the node via HTTP
            try:
                info = await self._send_heartbeat_to_peer(peer.host, peer.port)
                return web.json_response({
                    "node_id": node_id,
                    "http_reachable": info is not None,
                    "hybrid_transport_available": False,
                })
            except Exception as e:  # noqa: BLE001
                return web.json_response({
                    "node_id": node_id,
                    "http_reachable": False,
                    "error": str(e),
                    "hybrid_transport_available": False,
                })

        try:
            diagnosis = await diagnose_node_connectivity(node_id, peer.host, peer.port)
            return web.json_response(diagnosis)
        except Exception as e:  # noqa: BLE001
            return web.json_response({"error": str(e)}, status=500)
