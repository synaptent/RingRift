"""Cluster Node HTTP Handlers Mixin.

Provides HTTP endpoints for cluster node management and probing.

Usage:
    class P2POrchestrator(ClusterNodeHandlersMixin, ...):
        pass

Endpoints:
    GET /gpu-rankings - Get GPU power rankings for all cluster nodes
    POST /connectivity/probe_vast - Probe all Vast nodes via SSH

Requires the implementing class to have:
    - node_selector: NodeSelector for GPU node selection
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from aiohttp import web

from scripts.p2p.handlers.base import BaseP2PHandler

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Import HAS_HYBRID_TRANSPORT and probe_vast_nodes_via_ssh conditionally
try:
    from scripts.p2p.hybrid_transport import (
        HAS_HYBRID_TRANSPORT,
        probe_vast_nodes_via_ssh,
    )
except ImportError:
    HAS_HYBRID_TRANSPORT = False

    async def probe_vast_nodes_via_ssh():
        """Stub when hybrid transport not available."""
        return {}


# Import TRAINING_NODE_COUNT from orchestrator
try:
    from scripts.p2p_orchestrator import TRAINING_NODE_COUNT
except ImportError:
    TRAINING_NODE_COUNT = 3  # Default


class ClusterNodeHandlersMixin(BaseP2PHandler):
    """Mixin providing cluster node management HTTP handlers.

    Inherits from BaseP2PHandler for consistent response formatting.

    Requires the implementing class to have:
    - node_selector: NodeSelector for GPU node selection
    """

    # Type hints for IDE support
    node_selector: Any

    async def handle_gpu_rankings(self, request: web.Request) -> web.Response:
        """GET /gpu-rankings - Get GPU power rankings for all nodes in the cluster.

        Returns nodes sorted by GPU processing power for training priority.

        Returns:
            rankings: List of nodes with GPU rankings
            training_primary_nodes: List of primary training node IDs
            training_node_count: Configured number of training nodes
        """
        try:
            # Phase 2B: Direct calls to NodeSelector
            rankings = self.node_selector.get_training_nodes_ranked()
            training_nodes = self.node_selector.get_training_primary_nodes()

            return web.json_response({
                "rankings": rankings,
                "training_primary_nodes": [n.node_id for n in training_nodes],
                "training_node_count": TRAINING_NODE_COUNT,
            })
        except Exception as e:  # noqa: BLE001
            return web.json_response({"error": str(e)}, status=500)

    async def handle_probe_vast_nodes(self, request: web.Request) -> web.Response:
        """POST /connectivity/probe_vast - Probe all Vast nodes via SSH.

        Tests SSH connectivity to all vast-* nodes in the registry.
        Useful for diagnosing networking issues with Vast instances.

        Returns:
            total_nodes: Total number of Vast nodes
            reachable: Number of reachable nodes
            unreachable: Number of unreachable nodes
            nodes: Dict mapping node_id to {reachable, message}
        """
        if not HAS_HYBRID_TRANSPORT:
            return web.json_response({
                "error": "Hybrid transport not available"
            }, status=501)

        try:
            results = await probe_vast_nodes_via_ssh()
            reachable = sum(1 for r, _ in results.values() if r)

            return web.json_response({
                "total_nodes": len(results),
                "reachable": reachable,
                "unreachable": len(results) - reachable,
                "nodes": {
                    node_id: {"reachable": r, "message": msg}
                    for node_id, (r, msg) in results.items()
                },
            })
        except Exception as e:  # noqa: BLE001
            return web.json_response({"error": str(e)}, status=500)
