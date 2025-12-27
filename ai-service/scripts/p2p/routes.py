"""P2P HTTP Route Registry (December 2025).

Centralized route definitions for the P2P orchestrator HTTP API.
Routes are organized by category matching the handler mixins in scripts/p2p/handlers/.

This module reduces p2p_orchestrator.py complexity by providing:
- Declarative route definitions
- Automatic route registration
- Route documentation via docstrings

Usage:
    from scripts.p2p.routes import register_all_routes

    app = web.Application(middlewares=[...])
    register_all_routes(app, orchestrator)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from aiohttp import web


@dataclass
class Route:
    """A single HTTP route definition."""

    method: str  # GET, POST, DELETE, etc.
    path: str
    handler_name: str  # Name of handler method on orchestrator

    def register(self, app: "web.Application", handler: Callable) -> None:
        """Register this route on the aiohttp application."""
        method_lower = self.method.lower()
        if method_lower == "get":
            app.router.add_get(self.path, handler)
        elif method_lower == "post":
            app.router.add_post(self.path, handler)
        elif method_lower == "delete":
            app.router.add_delete(self.path, handler)
        elif method_lower == "put":
            app.router.add_put(self.path, handler)
        elif method_lower == "patch":
            app.router.add_patch(self.path, handler)
        else:
            raise ValueError(f"Unsupported HTTP method: {self.method}")


# ===========================================================================
# Core Routes (status, heartbeat, health)
# ===========================================================================

CORE_ROUTES = [
    Route("POST", "/heartbeat", "handle_heartbeat"),
    Route("GET", "/status", "handle_status"),
    Route("GET", "/external_work", "handle_external_work"),
    Route("GET", "/health", "handle_health"),
    Route("GET", "/cluster/health", "handle_cluster_health"),
    Route("GET", "/", "handle_root"),
]

# ===========================================================================
# Work Queue Routes (centralized work distribution)
# ===========================================================================

WORK_QUEUE_ROUTES = [
    Route("POST", "/work/add", "handle_work_add"),
    Route("POST", "/work/add_batch", "handle_work_add_batch"),
    Route("GET", "/work/claim", "handle_work_claim"),
    Route("POST", "/work/start", "handle_work_start"),
    Route("POST", "/work/complete", "handle_work_complete"),
    Route("POST", "/work/fail", "handle_work_fail"),
    Route("GET", "/work/status", "handle_work_status"),
    Route("GET", "/work/populator", "handle_populator_status"),
    Route("GET", "/work/node/{node_id}", "handle_work_for_node"),
    Route("POST", "/work/cancel", "handle_work_cancel"),
    Route("GET", "/work/history", "handle_work_history"),
]

# ===========================================================================
# Election Routes (leader election)
# ===========================================================================

ELECTION_ROUTES = [
    Route("POST", "/election", "handle_election"),
    Route("POST", "/election/lease", "handle_lease_request"),
    Route("GET", "/election/grant", "handle_voter_grant_status"),
    Route("POST", "/election/reset", "handle_election_reset"),
    Route("POST", "/election/force_leader", "handle_election_force_leader"),
]

# ===========================================================================
# SWIM Routes (native SWIM gossip integration)
# ===========================================================================

SWIM_ROUTES = [
    Route("GET", "/swim/status", "handle_swim_status"),
    Route("GET", "/swim/members", "handle_swim_members"),
]

# ===========================================================================
# Raft Routes (PySyncObj consensus integration)
# ===========================================================================

RAFT_ROUTES = [
    Route("GET", "/raft/status", "handle_raft_status"),
    Route("GET", "/raft/work", "handle_raft_work_queue"),
    Route("GET", "/raft/jobs", "handle_raft_jobs"),
    Route("POST", "/raft/lock/{name}", "handle_raft_lock"),
    Route("DELETE", "/raft/lock/{name}", "handle_raft_unlock"),
]

# ===========================================================================
# Job Management Routes
# ===========================================================================

JOB_ROUTES = [
    Route("POST", "/coordinator", "handle_coordinator"),
    Route("POST", "/start_job", "handle_start_job"),
    Route("POST", "/stop_job", "handle_stop_job"),
    Route("POST", "/job/kill", "handle_job_kill"),
    Route("POST", "/cleanup", "handle_cleanup"),
    Route("POST", "/restart_stuck_jobs", "handle_restart_stuck_jobs"),
    Route("POST", "/reduce_selfplay", "handle_reduce_selfplay"),
    Route("POST", "/selfplay/start", "handle_selfplay_start"),
]

# ===========================================================================
# Admin Routes (git, registry, cleanup)
# ===========================================================================

ADMIN_ROUTES = [
    Route("GET", "/git/status", "handle_git_status"),
    Route("POST", "/git/update", "handle_git_update"),
    Route("POST", "/register", "handle_register"),
    Route("GET", "/registry/status", "handle_registry_status"),
    Route("POST", "/registry/update_vast", "handle_registry_update_vast"),
    Route("POST", "/registry/update_aws", "handle_registry_update_aws"),
    Route("POST", "/registry/update_tailscale", "handle_registry_update_tailscale"),
    Route("POST", "/registry/save_yaml", "handle_registry_save_yaml"),
    Route("GET", "/admin/purge_retired", "handle_purge_retired_peers"),
    Route("GET", "/admin/purge_stale", "handle_purge_stale_peers"),
    Route("POST", "/admin/unretire", "handle_admin_unretire"),
    Route("POST", "/admin/restart", "handle_admin_restart"),
]

# ===========================================================================
# Connectivity Routes (diagnosis, transport stats)
# ===========================================================================

CONNECTIVITY_ROUTES = [
    Route("GET", "/connectivity/diagnose/{node_id}", "handle_connectivity_diagnose"),
    Route("GET", "/connectivity/transport_stats", "handle_transport_stats"),
    Route("POST", "/connectivity/probe_vast", "handle_probe_vast_nodes"),
]

# ===========================================================================
# Gauntlet Routes (model evaluation)
# ===========================================================================

GAUNTLET_ROUTES = [
    Route("POST", "/gauntlet/execute", "handle_gauntlet_execute"),
    Route("GET", "/gauntlet/status", "handle_gauntlet_status"),
    Route("POST", "/gauntlet/quick-eval", "handle_gauntlet_quick_eval"),
]

# ===========================================================================
# Relay Routes (NAT-blocked node communication)
# ===========================================================================

RELAY_ROUTES = [
    Route("POST", "/relay/heartbeat", "handle_relay_heartbeat"),
    Route("GET", "/relay/peers", "handle_relay_peers"),
    Route("GET", "/relay/status", "handle_relay_status"),
    Route("POST", "/relay/enqueue", "handle_relay_enqueue"),
]

# ===========================================================================
# Gossip Routes (decentralized state sharing)
# ===========================================================================

GOSSIP_ROUTES = [
    Route("POST", "/gossip", "handle_gossip"),
    Route("POST", "/gossip/anti-entropy", "handle_gossip_anti_entropy"),
    Route("POST", "/serf/event", "handle_serf_event"),
]

# ===========================================================================
# Data Manifest Routes
# ===========================================================================

DATA_MANIFEST_ROUTES = [
    Route("GET", "/data_manifest", "handle_data_manifest"),
    Route("GET", "/cluster_data_manifest", "handle_cluster_data_manifest"),
    Route("POST", "/refresh_manifest", "handle_refresh_manifest"),
]

# ===========================================================================
# CMA-ES Routes (distributed hyperparameter optimization)
# ===========================================================================

CMAES_ROUTES = [
    Route("POST", "/cmaes/start", "handle_cmaes_start"),
    Route("POST", "/cmaes/evaluate", "handle_cmaes_evaluate"),
    Route("GET", "/cmaes/status", "handle_cmaes_status"),
    Route("POST", "/cmaes/result", "handle_cmaes_result"),
]

# ===========================================================================
# Tournament Routes (distributed round-robin)
# ===========================================================================

TOURNAMENT_ROUTES = [
    Route("POST", "/tournament/start", "handle_tournament_start"),
    Route("POST", "/tournament/match", "handle_tournament_match"),
    Route("POST", "/tournament/play_elo_match", "handle_play_elo_match"),
    Route("GET", "/tournament/status", "handle_tournament_status"),
    Route("POST", "/tournament/result", "handle_tournament_result"),
    Route("POST", "/tournament/ssh_start", "handle_ssh_tournament_start"),
    Route("GET", "/tournament/ssh_status", "handle_ssh_tournament_status"),
    Route("POST", "/tournament/ssh_cancel", "handle_ssh_tournament_cancel"),
]

# ===========================================================================
# Improvement Loop Routes
# ===========================================================================

IMPROVEMENT_ROUTES = [
    Route("POST", "/improvement/start", "handle_improvement_start"),
    Route("GET", "/improvement/status", "handle_improvement_status"),
    Route("POST", "/improvement/phase_complete", "handle_improvement_phase_complete"),
]

# ===========================================================================
# Sync Routes (P2P data synchronization)
# ===========================================================================

SYNC_ROUTES = [
    Route("POST", "/sync/start", "handle_sync_start"),
    Route("GET", "/sync/status", "handle_sync_status"),
    Route("POST", "/sync/pull", "handle_sync_pull"),
    Route("GET", "/sync/file", "handle_sync_file"),
    Route("POST", "/sync/job_update", "handle_sync_job_update"),
    Route("POST", "/sync/training", "handle_training_sync"),
]

# ===========================================================================
# GPU Ranking Routes
# ===========================================================================

GPU_ROUTES = [
    Route("GET", "/gpu/rankings", "handle_gpu_rankings"),
]

# ===========================================================================
# Cleanup Routes
# ===========================================================================

CLEANUP_ROUTES = [
    Route("POST", "/cleanup/files", "handle_cleanup_files"),
]

# ===========================================================================
# Event Subscription Routes
# ===========================================================================

SUBSCRIPTION_ROUTES = [
    Route("GET", "/subscriptions", "handle_subscriptions"),
]

# ===========================================================================
# Training Pipeline Routes
# ===========================================================================

TRAINING_ROUTES = [
    Route("POST", "/training/start", "handle_training_start"),
    Route("GET", "/training/status", "handle_training_status"),
    Route("POST", "/training/update", "handle_training_update"),
    Route("POST", "/training/nnue/start", "handle_nnue_start"),
    Route("POST", "/training/cmaes/start", "handle_cmaes_start_auto"),
]

# ===========================================================================
# Improvement Cycle Routes
# ===========================================================================

IMPROVEMENT_CYCLE_ROUTES = [
    Route("GET", "/improvement_cycles/status", "handle_improvement_cycles_status"),
    Route("GET", "/improvement_cycles/leaderboard", "handle_improvement_cycles_leaderboard"),
    Route("POST", "/improvement_cycles/training_complete", "handle_improvement_training_complete"),
    Route("POST", "/improvement_cycles/evaluation_complete", "handle_improvement_evaluation_complete"),
]

# ===========================================================================
# Metrics Routes (observability)
# ===========================================================================

METRICS_ROUTES = [
    Route("GET", "/metrics", "handle_metrics"),
    Route("GET", "/metrics/prometheus", "handle_metrics_prometheus"),
]

# ===========================================================================
# Pipeline Routes (canonical pipeline orchestration)
# ===========================================================================

PIPELINE_ROUTES = [
    Route("POST", "/pipeline/start", "handle_pipeline_start"),
    Route("GET", "/pipeline/status", "handle_pipeline_status"),
    Route("POST", "/pipeline/selfplay_worker", "handle_pipeline_selfplay_worker"),
]

# ===========================================================================
# REST API Routes (dashboard integration)
# ===========================================================================

API_ROUTES = [
    Route("GET", "/api/cluster/status", "handle_api_cluster_status"),
    Route("POST", "/api/cluster/git/update", "handle_api_cluster_git_update"),
    Route("GET", "/api/selfplay/stats", "handle_api_selfplay_stats"),
    Route("GET", "/api/elo/leaderboard", "handle_api_elo_leaderboard"),
]

# ===========================================================================
# Elo Routes (rating tables and sync)
# ===========================================================================

ELO_ROUTES = [
    Route("GET", "/elo/table", "handle_elo_table"),
    Route("GET", "/elo/history", "handle_elo_history"),
    Route("GET", "/elo/sync/status", "handle_elo_sync_status"),
    Route("POST", "/elo/sync/trigger", "handle_elo_sync_trigger"),
    Route("GET", "/elo/sync/db", "handle_elo_sync_download"),
    Route("POST", "/elo/sync/upload", "handle_elo_sync_upload"),
]

# ===========================================================================
# Dashboard Table Routes
# ===========================================================================

DASHBOARD_ROUTES = [
    Route("GET", "/nodes/table", "handle_nodes_table"),
    Route("GET", "/victory/table", "handle_victory_table"),
    Route("GET", "/games/analytics", "handle_games_analytics"),
    Route("GET", "/training/metrics", "handle_training_metrics"),
    Route("GET", "/holdout/metrics", "handle_holdout_metrics"),
    Route("GET", "/holdout/table", "handle_holdout_table"),
    Route("GET", "/mcts/stats", "handle_mcts_stats"),
    Route("GET", "/mcts/table", "handle_mcts_table"),
    Route("GET", "/matchups/matrix", "handle_matchup_matrix"),
    Route("GET", "/matchups/table", "handle_matchup_table"),
    Route("GET", "/models/lineage", "handle_model_lineage"),
    Route("GET", "/models/lineage/table", "handle_model_lineage_table"),
    Route("GET", "/data/quality", "handle_data_quality"),
    Route("GET", "/data/quality/table", "handle_data_quality_table"),
    Route("GET", "/data/quality/issues", "handle_data_quality_issues"),
    Route("GET", "/training/efficiency", "handle_training_efficiency"),
    Route("GET", "/training/efficiency/table", "handle_training_efficiency_table"),
    Route("GET", "/rollback/status", "handle_rollback_status"),
    Route("GET", "/rollback/candidates", "handle_rollback_candidates"),
    Route("POST", "/rollback/execute", "handle_rollback_execute"),
    Route("POST", "/rollback/auto", "handle_rollback_auto"),
    Route("GET", "/autoscale/metrics", "handle_autoscale_metrics"),
    Route("GET", "/autoscale/recommendations", "handle_autoscale_recommendations"),
]

# ===========================================================================
# All Route Groups
# ===========================================================================

ALL_ROUTE_GROUPS = {
    "core": CORE_ROUTES,
    "work_queue": WORK_QUEUE_ROUTES,
    "election": ELECTION_ROUTES,
    "swim": SWIM_ROUTES,
    "raft": RAFT_ROUTES,
    "job": JOB_ROUTES,
    "admin": ADMIN_ROUTES,
    "connectivity": CONNECTIVITY_ROUTES,
    "gauntlet": GAUNTLET_ROUTES,
    "relay": RELAY_ROUTES,
    "gossip": GOSSIP_ROUTES,
    "data_manifest": DATA_MANIFEST_ROUTES,
    "cmaes": CMAES_ROUTES,
    "tournament": TOURNAMENT_ROUTES,
    "improvement": IMPROVEMENT_ROUTES,
    "sync": SYNC_ROUTES,
    "gpu": GPU_ROUTES,
    "cleanup": CLEANUP_ROUTES,
    "subscription": SUBSCRIPTION_ROUTES,
    "training": TRAINING_ROUTES,
    "improvement_cycle": IMPROVEMENT_CYCLE_ROUTES,
    "metrics": METRICS_ROUTES,
    "pipeline": PIPELINE_ROUTES,
    "api": API_ROUTES,
    "elo": ELO_ROUTES,
    "dashboard": DASHBOARD_ROUTES,
}


def get_all_routes() -> list[Route]:
    """Get all routes as a flat list."""
    routes = []
    for group in ALL_ROUTE_GROUPS.values():
        routes.extend(group)
    return routes


def register_all_routes(app: "web.Application", orchestrator: object) -> int:
    """Register all P2P HTTP routes on the aiohttp application.

    Args:
        app: The aiohttp web application.
        orchestrator: The P2POrchestrator instance with handler methods.

    Returns:
        Number of routes registered.

    Raises:
        AttributeError: If orchestrator is missing a handler method.
    """
    count = 0
    missing_handlers = []

    for route in get_all_routes():
        handler = getattr(orchestrator, route.handler_name, None)
        if handler is None:
            missing_handlers.append(route.handler_name)
            continue
        route.register(app, handler)
        count += 1

    if missing_handlers:
        # Log but don't fail - some handlers may be added by optional mixins
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Missing {len(missing_handlers)} handlers: {missing_handlers[:5]}...")

    return count


def register_route_group(
    app: "web.Application",
    orchestrator: object,
    group_name: str
) -> int:
    """Register a specific group of routes.

    Args:
        app: The aiohttp web application.
        orchestrator: The P2POrchestrator instance.
        group_name: Name of the route group to register.

    Returns:
        Number of routes registered.
    """
    if group_name not in ALL_ROUTE_GROUPS:
        raise ValueError(f"Unknown route group: {group_name}")

    count = 0
    for route in ALL_ROUTE_GROUPS[group_name]:
        handler = getattr(orchestrator, route.handler_name, None)
        if handler is not None:
            route.register(app, handler)
            count += 1

    return count
