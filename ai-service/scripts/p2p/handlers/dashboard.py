"""Dashboard HTTP handlers for P2P orchestrator.

January 2026 - P2P Modularization Phase 2a

This mixin provides HTTP handlers for dashboard and UI endpoints including
HTML pages and resource optimization displays.

Must be mixed into a class that provides:
- self.node_id: str
- self.build_version: str (optional)
"""
from __future__ import annotations

import logging
from pathlib import Path

from aiohttp import web

logger = logging.getLogger(__name__)

# Try to import resource optimizer dependencies
try:
    from app.coordination.resource_optimizer import get_resource_optimizer
    from scripts.p2p.defaults import TARGET_GPU_UTIL_MIN, TARGET_GPU_UTIL_MAX
    HAS_RESOURCE_OPTIMIZER = True
except ImportError:
    HAS_RESOURCE_OPTIMIZER = False
    get_resource_optimizer = None  # type: ignore
    TARGET_GPU_UTIL_MIN = 60
    TARGET_GPU_UTIL_MAX = 80


class DashboardHandlersMixin:
    """Mixin providing dashboard and UI HTTP handlers.

    Endpoints:
    - GET / - Redirect to dashboard
    - GET /dashboard - Main dashboard HTML
    - GET /work_queue/dashboard - Work queue dashboard HTML
    - GET /resource/optimizer - Resource optimizer state and recommendations
    """

    async def handle_root(self, request: web.Request) -> web.Response:
        """GET / - Redirect to dashboard."""
        raise web.HTTPFound(location="/dashboard")

    async def handle_dashboard(self, request: web.Request) -> web.Response:
        """Serve the web dashboard HTML."""
        dashboard_path = Path(__file__).resolve().parent.parent / "dashboard_assets" / "dashboard.html"
        try:
            html = dashboard_path.read_text(encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            html = (
                "<!doctype html><html><body style='font-family:monospace'>"
                f"<h3>Dashboard asset unavailable</h3><pre>{e}</pre>"
                f"<pre>Expected: {dashboard_path}</pre>"
                "</body></html>"
            )
        headers = {
            # Avoid stale HTML across load balancers / browsers during rapid iteration.
            "Cache-Control": "no-store, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
            # Simple diagnostics (no secrets).
            "X-RingRift-Node-Id": str(getattr(self, "node_id", "") or ""),
            "X-RingRift-Build-Version": str(getattr(self, "build_version", "") or ""),
        }
        return web.Response(text=html, content_type="text/html", headers=headers)

    async def handle_work_queue_dashboard(self, request: web.Request) -> web.Response:
        """Serve the work queue dashboard HTML."""
        dashboard_path = Path(__file__).resolve().parent.parent / "dashboard_assets" / "work_queue_dashboard.html"
        try:
            html = dashboard_path.read_text(encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            html = (
                "<!doctype html><html><body style='font-family:monospace'>"
                f"<h3>Work Queue Dashboard unavailable</h3><pre>{e}</pre>"
                f"<pre>Expected: {dashboard_path}</pre>"
                "</body></html>"
            )
        headers = {
            "Cache-Control": "no-store, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
            "X-RingRift-Node-Id": str(getattr(self, "node_id", "") or ""),
        }
        return web.Response(text=html, content_type="text/html", headers=headers)

    async def handle_resource_optimizer(self, request: web.Request) -> web.Response:
        """GET /resource/optimizer - Resource optimizer state and recommendations.

        Returns cluster-wide utilization state, PID-controlled optimization
        recommendations, and target utilization ranges (60-80%).
        """
        try:
            if not HAS_RESOURCE_OPTIMIZER:
                return web.json_response({
                    "error": "Resource optimizer not available",
                    "available": False,
                })

            optimizer = get_resource_optimizer()
            cluster_state = optimizer.get_cluster_state()
            recommendation = optimizer.get_optimization_recommendation()
            metrics = optimizer.get_metrics_dict()

            return web.json_response({
                "available": True,
                "cluster_state": {
                    "total_cpu_util": round(cluster_state.total_cpu_util, 1),
                    "total_gpu_util": round(cluster_state.total_gpu_util, 1),
                    "total_memory_util": round(cluster_state.total_memory_util, 1),
                    "gpu_node_count": cluster_state.gpu_node_count,
                    "cpu_node_count": cluster_state.cpu_node_count,
                    "total_jobs": cluster_state.total_jobs,
                    "nodes": [n.to_dict() for n in cluster_state.nodes],
                },
                "recommendation": recommendation.to_dict(),
                "targets": {
                    "min": TARGET_GPU_UTIL_MIN,
                    "max": TARGET_GPU_UTIL_MAX,
                    "optimal": (TARGET_GPU_UTIL_MIN + TARGET_GPU_UTIL_MAX) // 2,
                },
                "metrics": metrics,
                "in_target_range": {
                    "cpu": TARGET_GPU_UTIL_MIN <= cluster_state.total_cpu_util <= TARGET_GPU_UTIL_MAX,
                    "gpu": TARGET_GPU_UTIL_MIN <= cluster_state.total_gpu_util <= TARGET_GPU_UTIL_MAX
                           if cluster_state.gpu_node_count > 0 else True,
                },
            })
        except Exception as e:  # noqa: BLE001
            return web.json_response({"error": str(e), "available": False}, status=500)
