"""Registry HTTP Handlers Mixin.

Provides HTTP endpoints for dynamic IP registry management.
Handles node IP updates from cloud providers (Vast, AWS) and Tailscale.

Usage:
    class P2POrchestrator(RegistryHandlersMixin, ...):
        pass

Endpoints:
    GET /registry/status - Get dynamic registry status for all nodes
    POST /registry/update_vast - Refresh Vast instance IPs
    POST /registry/update_aws - Refresh AWS instance IPs
    POST /registry/update_tailscale - Discover Tailscale IPs
    POST /registry/save_yaml - Write dynamic IPs back to YAML config
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from aiohttp import web

from scripts.p2p.handlers.base import BaseP2PHandler

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Import dynamic registry
try:
    from app.distributed.dynamic_registry import (
        get_registry,
    )
    HAS_DYNAMIC_REGISTRY = True
except ImportError:
    HAS_DYNAMIC_REGISTRY = False
    get_registry = None


class RegistryHandlersMixin(BaseP2PHandler):
    """Mixin providing dynamic registry HTTP handlers.

    Inherits from BaseP2PHandler for consistent response formatting.
    """

    async def handle_registry_status(self, request: web.Request) -> web.Response:
        """GET /registry/status - Get dynamic registry status for all nodes.

        Returns current state of all nodes including:
        - Effective IP addresses (dynamic if registered)
        - Health state (online/degraded/offline)
        - Failure counters
        """
        if not HAS_DYNAMIC_REGISTRY:
            return web.json_response({
                "error": "Dynamic registry not available"
            }, status=501)

        try:
            registry = get_registry()
            nodes_status = registry.get_all_nodes_status()
            online_nodes = registry.get_online_nodes()

            return web.json_response({
                "total_nodes": len(nodes_status),
                "online_nodes": len(online_nodes),
                "online_node_ids": online_nodes,
                "nodes": nodes_status,
            })

        except Exception as e:  # noqa: BLE001
            return web.json_response({"error": str(e)}, status=500)

    async def handle_registry_update_vast(self, request: web.Request) -> web.Response:
        """POST /registry/update_vast - Refresh Vast instance IPs in the dynamic registry.

        Uses VAST_API_KEY when available, otherwise attempts the `vastai` CLI.
        """
        if not HAS_DYNAMIC_REGISTRY:
            return web.json_response({
                "error": "Dynamic registry not available"
            }, status=501)

        try:
            registry = get_registry()
            updated = await registry.update_vast_ips()

            return web.json_response({
                "success": True,
                "nodes_updated": updated,
            })

        except Exception as e:  # noqa: BLE001
            return web.json_response({"error": str(e)}, status=500)

    async def handle_registry_update_aws(self, request: web.Request) -> web.Response:
        """POST /registry/update_aws - Refresh AWS instance IPs in the dynamic registry.

        Uses the `aws` CLI and requires nodes to define `aws_instance_id` in
        distributed_hosts.yaml properties.
        """
        if not HAS_DYNAMIC_REGISTRY:
            return web.json_response({"error": "Dynamic registry not available"}, status=501)

        try:
            registry = get_registry()
            updated = await registry.update_aws_ips()
            return web.json_response({"success": True, "nodes_updated": updated})
        except Exception as e:  # noqa: BLE001
            return web.json_response({"error": str(e)}, status=500)

    async def handle_registry_update_tailscale(self, request: web.Request) -> web.Response:
        """POST /registry/update_tailscale - Discover Tailscale IPs in the dynamic registry.

        Uses `tailscale status --json` when available. No-op if `tailscale` is
        not installed or the node is not part of a Tailscale network.
        """
        if not HAS_DYNAMIC_REGISTRY:
            return web.json_response({"error": "Dynamic registry not available"}, status=501)

        try:
            registry = get_registry()
            updated = await registry.update_tailscale_ips()
            return web.json_response({"success": True, "nodes_updated": updated})
        except Exception as e:  # noqa: BLE001
            return web.json_response({"error": str(e)}, status=500)

    async def handle_registry_save_yaml(self, request: web.Request) -> web.Response:
        """POST /registry/save_yaml - Write dynamic IPs back to YAML config.

        Creates a backup before modifying. Only updates hosts where
        dynamic IP differs from static IP.
        """
        if not HAS_DYNAMIC_REGISTRY:
            return web.json_response({
                "error": "Dynamic registry not available"
            }, status=501)

        try:
            registry = get_registry()
            updated = registry.update_yaml_config()

            return web.json_response({
                "success": True,
                "config_updated": updated,
            })

        except Exception as e:  # noqa: BLE001
            return web.json_response({"error": str(e)}, status=500)
