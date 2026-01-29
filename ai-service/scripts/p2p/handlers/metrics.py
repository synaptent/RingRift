"""P2P Metrics HTTP handlers.

January 2026: Extracted from p2p_orchestrator.py to reduce file size.

Endpoints:
- GET /metrics - Metrics with content negotiation (JSON or Prometheus format)
- GET /metrics/prometheus - Prometheus-compatible metrics export
- GET /pipeline/status - Current pipeline phase status

The handler accesses orchestrator state via `self.*` since it's designed
as a mixin that gets inherited by P2POrchestrator.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aiohttp import web

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def safe_db_connection(db_path: Path):
    """Context manager for safe database connections."""
    import contextlib

    @contextlib.contextmanager
    def _connection():
        conn = sqlite3.connect(str(db_path), timeout=5.0)
        try:
            yield conn
        finally:
            conn.close()

    return _connection()


# GPU hourly rates (Lambda Labs pricing as of Dec 2025)
GPU_HOURLY_RATES = {
    "GH200": 2.49, "H100": 2.49, "A100": 1.99, "A10": 0.75,
    "RTX_4090": 0.50, "RTX4090": 0.50, "4090": 0.50,
    "RTX_3090": 0.30, "RTX3090": 0.30, "3090": 0.30,
    "unknown": 0.50,
}


class MetricsHandlersMixin:
    """Mixin providing Prometheus metrics HTTP handlers.

    Must be mixed into a class that provides:
    - self.peers_lock, self.peers
    - self.jobs_lock, self.local_jobs
    - self.role, self.node_id, self.self_info
    - self.diversity_metrics, self.cluster_data_manifest
    - self.training_metrics, self.selfplay_throughput
    - self.cost_metrics, self.promotion_metrics
    - self.improvement_cycle_manager
    - self.is_leader, self.start_time
    - self.analytics_cache_manager (AnalyticsCacheManager instance)
    """

    # Required attributes (provided by orchestrator)
    analytics_cache_manager: Any  # AnalyticsCacheManager
    peers_lock: Any
    peers: dict
    jobs_lock: Any
    local_jobs: dict
    role: Any
    node_id: str
    self_info: Any
    diversity_metrics: dict
    cluster_data_manifest: Any
    training_metrics: dict
    selfplay_throughput: dict
    cost_metrics: dict
    promotion_metrics: dict
    improvement_cycle_manager: Any
    is_leader: bool
    start_time: float

    async def handle_metrics_prometheus(self, request: web.Request) -> web.Response:
        """GET /metrics/prometheus - Prometheus-compatible metrics export.

        Returns metrics in Prometheus text exposition format for scraping.
        """
        try:
            lines = []
            now = time.time()

            # Cluster metrics - Jan 12, 2026: Copy-on-write to reduce lock hold time
            with self.peers_lock:
                peers_snapshot = list(self.peers.values())
            alive_peers = len([p for p in peers_snapshot if p.is_alive()])
            total_peers = len(peers_snapshot)

            lines.append("# HELP ringrift_cluster_peers_total Total number of known peers")
            lines.append("# TYPE ringrift_cluster_peers_total gauge")
            lines.append(f"ringrift_cluster_peers_total {total_peers}")

            lines.append("# HELP ringrift_cluster_peers_alive Number of alive peers")
            lines.append("# TYPE ringrift_cluster_peers_alive gauge")
            lines.append(f"ringrift_cluster_peers_alive {alive_peers}")

            # Import NodeRole here to avoid circular imports at module level
            from scripts.p2p.models import NodeRole

            lines.append("# HELP ringrift_is_leader Whether this node is the leader")
            lines.append("# TYPE ringrift_is_leader gauge")
            lines.append(f"ringrift_is_leader {1 if self.role == NodeRole.LEADER else 0}")

            # Job counts
            from scripts.p2p.models import JobType
            with self.jobs_lock:
                selfplay_jobs = len([j for j in self.local_jobs.values()
                                    if j.job_type in (JobType.SELFPLAY, JobType.GPU_SELFPLAY, JobType.HYBRID_SELFPLAY, JobType.CPU_SELFPLAY, JobType.GUMBEL_SELFPLAY)
                                    and j.status == "running"])
                training_jobs = len([j for j in self.local_jobs.values()
                                    if j.job_type == JobType.TRAINING and j.status == "running"])

            lines.append("# HELP ringrift_selfplay_jobs_running Number of running selfplay jobs")
            lines.append("# TYPE ringrift_selfplay_jobs_running gauge")
            lines.append(f"ringrift_selfplay_jobs_running {selfplay_jobs}")

            lines.append("# HELP ringrift_training_jobs_running Number of running training jobs")
            lines.append("# TYPE ringrift_training_jobs_running gauge")
            lines.append(f"ringrift_training_jobs_running {training_jobs}")

            # Games per hour metric - aggregate from all peers
            lines.append("# HELP ringrift_selfplay_games_per_hour Estimated games generated per hour")
            lines.append("# TYPE ringrift_selfplay_games_per_hour gauge")

            # Calculate games/hour from peer data - reuse peers_snapshot from above
            total_cluster_selfplay_jobs = 0
            for peer in peers_snapshot:
                if peer.is_alive():
                    jobs = getattr(peer, 'selfplay_jobs', 0) or 0
                    total_cluster_selfplay_jobs += jobs

            # Estimate games/hour based on running jobs (rough heuristic: ~30 games/hour per job)
            estimated_games_per_hour = total_cluster_selfplay_jobs * 30
            lines.append(f"ringrift_selfplay_games_per_hour {estimated_games_per_hour}")

            # Also report total cluster selfplay jobs
            lines.append("# HELP ringrift_cluster_selfplay_jobs_total Total selfplay jobs across cluster")
            lines.append("# TYPE ringrift_cluster_selfplay_jobs_total gauge")
            lines.append(f"ringrift_cluster_selfplay_jobs_total {total_cluster_selfplay_jobs}")

            # Resource utilization - include node labels for all nodes
            lines.append("# HELP ringrift_cpu_percent CPU utilization percentage per node")
            lines.append("# TYPE ringrift_cpu_percent gauge")

            lines.append("# HELP ringrift_memory_percent Memory utilization percentage per node")
            lines.append("# TYPE ringrift_memory_percent gauge")

            lines.append("# HELP ringrift_disk_percent Disk utilization percentage per node")
            lines.append("# TYPE ringrift_disk_percent gauge")

            lines.append("# HELP ringrift_gpu_percent GPU utilization percentage per node")
            lines.append("# TYPE ringrift_gpu_percent gauge")

            lines.append("# HELP ringrift_selfplay_jobs Selfplay jobs per node")
            lines.append("# TYPE ringrift_selfplay_jobs gauge")

            lines.append("# HELP ringrift_node_alive Whether node is alive (1) or not (0)")
            lines.append("# TYPE ringrift_node_alive gauge")

            # Cluster cost metrics (for Grafana dashboards)
            lines.append("# HELP ringrift_cluster_node_up Whether cluster node is active (1=up, 0=down)")
            lines.append("# TYPE ringrift_cluster_node_up gauge")
            lines.append("# HELP ringrift_cluster_node_cost_per_hour Estimated hourly cost in USD")
            lines.append("# TYPE ringrift_cluster_node_cost_per_hour gauge")
            lines.append("# HELP ringrift_cluster_gpu_utilization GPU utilization as fraction (0-1)")
            lines.append("# TYPE ringrift_cluster_gpu_utilization gauge")
            lines.append("# HELP ringrift_cluster_cpu_utilization CPU utilization as fraction (0-1)")
            lines.append("# TYPE ringrift_cluster_cpu_utilization gauge")
            lines.append("# HELP ringrift_cluster_gpu_memory_used_bytes GPU memory used in bytes")
            lines.append("# TYPE ringrift_cluster_gpu_memory_used_bytes gauge")
            lines.append("# HELP ringrift_cluster_memory_used_bytes System memory used in bytes")
            lines.append("# TYPE ringrift_cluster_memory_used_bytes gauge")

            # Export self metrics with node label
            node_name = self.node_id or "unknown"
            cpu = getattr(self.self_info, 'cpu_percent', 0)
            mem = getattr(self.self_info, 'memory_percent', 0)
            disk = getattr(self.self_info, 'disk_percent', 0)
            gpu = getattr(self.self_info, 'gpu_percent', 0) if self.self_info.has_gpu else 0
            role = "leader" if self.role == NodeRole.LEADER else "worker"
            gpu_type = getattr(self.self_info, 'gpu_type', 'unknown') or 'unknown'
            # Normalize GPU type for lookup
            gpu_type_key = gpu_type.replace(' ', '_').upper() if gpu_type else 'unknown'
            hourly_cost = GPU_HOURLY_RATES.get(gpu_type_key, GPU_HOURLY_RATES.get(gpu_type, GPU_HOURLY_RATES['unknown']))
            gpu_mem_bytes = getattr(self.self_info, 'gpu_memory_used_bytes', 0) or 0
            sys_mem_bytes = getattr(self.self_info, 'memory_used_bytes', 0) or 0

            lines.append(f'ringrift_cpu_percent{{node="{node_name}",role="{role}"}} {cpu}')
            lines.append(f'ringrift_memory_percent{{node="{node_name}",role="{role}"}} {mem}')
            lines.append(f'ringrift_disk_percent{{node="{node_name}",role="{role}"}} {disk}')
            lines.append(f'ringrift_gpu_percent{{node="{node_name}",role="{role}"}} {gpu}')
            lines.append(f'ringrift_selfplay_jobs{{node="{node_name}",role="{role}"}} {selfplay_jobs}')
            lines.append(f'ringrift_node_alive{{node="{node_name}",role="{role}"}} 1')

            # Export cluster cost metrics for self (for Grafana cost dashboard)
            lines.append(f'ringrift_cluster_node_up{{node="{node_name}",gpu_type="{gpu_type}"}} 1')
            lines.append(f'ringrift_cluster_node_cost_per_hour{{node="{node_name}",gpu_type="{gpu_type}"}} {hourly_cost}')
            lines.append(f'ringrift_cluster_gpu_utilization{{node="{node_name}",gpu_type="{gpu_type}"}} {gpu / 100.0 if gpu else 0}')
            lines.append(f'ringrift_cluster_cpu_utilization{{node="{node_name}"}} {cpu / 100.0 if cpu else 0}')
            lines.append(f'ringrift_cluster_gpu_memory_used_bytes{{node="{node_name}",gpu_type="{gpu_type}"}} {gpu_mem_bytes}')
            lines.append(f'ringrift_cluster_memory_used_bytes{{node="{node_name}"}} {sys_mem_bytes}')

            # Export peer metrics with node labels
            # Jan 12, 2026: Copy-on-write - snapshot for thread-safe iteration
            with self.peers_lock:
                peers_items_snapshot = list(self.peers.items())
            for peer_id, peer in peers_items_snapshot:
                peer_name = peer_id or "unknown"
                peer_role = "worker"
                is_alive = 1 if peer.is_alive() else 0

                # Get peer resource info if available
                peer_cpu = getattr(peer, 'cpu_percent', 0) or 0
                peer_mem = getattr(peer, 'memory_percent', 0) or 0
                peer_gpu = getattr(peer, 'gpu_percent', 0) or 0
                peer_jobs = getattr(peer, 'selfplay_jobs', 0) or 0
                peer_gpu_type = getattr(peer, 'gpu_type', 'unknown') or 'unknown'
                peer_gpu_type_key = peer_gpu_type.replace(' ', '_').upper() if peer_gpu_type else 'unknown'
                peer_hourly_cost = GPU_HOURLY_RATES.get(peer_gpu_type_key, GPU_HOURLY_RATES.get(peer_gpu_type, GPU_HOURLY_RATES['unknown']))
                peer_gpu_mem = getattr(peer, 'gpu_memory_used_bytes', 0) or 0
                peer_sys_mem = getattr(peer, 'memory_used_bytes', 0) or 0

                lines.append(f'ringrift_cpu_percent{{node="{peer_name}",role="{peer_role}"}} {peer_cpu}')
                lines.append(f'ringrift_memory_percent{{node="{peer_name}",role="{peer_role}"}} {peer_mem}')
                lines.append(f'ringrift_gpu_percent{{node="{peer_name}",role="{peer_role}"}} {peer_gpu}')
                lines.append(f'ringrift_selfplay_jobs{{node="{peer_name}",role="{peer_role}"}} {peer_jobs}')
                lines.append(f'ringrift_node_alive{{node="{peer_name}",role="{peer_role}"}} {is_alive}')

                # Export cluster cost metrics for peer
                lines.append(f'ringrift_cluster_node_up{{node="{peer_name}",gpu_type="{peer_gpu_type}"}} {is_alive}')
                lines.append(f'ringrift_cluster_node_cost_per_hour{{node="{peer_name}",gpu_type="{peer_gpu_type}"}} {peer_hourly_cost if is_alive else 0}')
                lines.append(f'ringrift_cluster_gpu_utilization{{node="{peer_name}",gpu_type="{peer_gpu_type}"}} {peer_gpu / 100.0 if peer_gpu else 0}')
                lines.append(f'ringrift_cluster_cpu_utilization{{node="{peer_name}"}} {peer_cpu / 100.0 if peer_cpu else 0}')
                lines.append(f'ringrift_cluster_gpu_memory_used_bytes{{node="{peer_name}",gpu_type="{peer_gpu_type}"}} {peer_gpu_mem}')
                lines.append(f'ringrift_cluster_memory_used_bytes{{node="{peer_name}"}} {peer_sys_mem}')

            # Elo metrics with config labels
            try:
                from scripts.run_model_elo_tournament import ELO_DB_PATH, init_elo_database
                if ELO_DB_PATH and ELO_DB_PATH.exists():
                    db = init_elo_database()
                    conn = db._get_connection()
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT board_type, num_players, MAX(rating) as best_elo
                        FROM elo_ratings
                        WHERE games_played >= 10
                        GROUP BY board_type, num_players
                    """)
                    lines.append("# HELP ringrift_best_elo Best Elo rating per configuration")
                    lines.append("# TYPE ringrift_best_elo gauge")
                    for row in cursor.fetchall():
                        bt, np, elo = row
                        config = f"{bt}_{np}p"
                        lines.append(f'ringrift_best_elo{{config="{config}",board_type="{bt}",num_players="{np}"}} {elo}')
                    db.close()
            except (OSError, AttributeError, ImportError):
                pass

            # Diversity metrics
            if hasattr(self, 'diversity_metrics'):
                dm = self.diversity_metrics
                lines.append("# HELP ringrift_tournament_runs_total Total tournament runs")
                lines.append("# TYPE ringrift_tournament_runs_total counter")
                lines.append(f"ringrift_tournament_runs_total {dm.get('tournament_runs', 0)}")

                lines.append("# HELP ringrift_promotions_total Total model promotions")
                lines.append("# TYPE ringrift_promotions_total counter")
                lines.append(f"ringrift_promotions_total {dm.get('promotions', 0)}")

                lines.append("# HELP ringrift_rollbacks_total Total model rollbacks")
                lines.append("# TYPE ringrift_rollbacks_total counter")
                lines.append(f"ringrift_rollbacks_total {dm.get('rollbacks', 0)}")

                # GPU validation stats
                gpu_stats = dm.get('gpu_validation_stats', {})
                if gpu_stats:
                    lines.append("# HELP ringrift_gpu_games_validated_total Total GPU games validated")
                    lines.append("# TYPE ringrift_gpu_games_validated_total counter")
                    lines.append(f"ringrift_gpu_games_validated_total {gpu_stats.get('total_validated', 0)}")

                    lines.append("# HELP ringrift_gpu_games_failed_total Total GPU games failed validation")
                    lines.append("# TYPE ringrift_gpu_games_failed_total counter")
                    lines.append(f"ringrift_gpu_games_failed_total {gpu_stats.get('total_failed', 0)}")

            # Recent metrics from database (last hour averages)
            try:
                summary = self.get_metrics_summary(hours=1)
                metrics_data = summary.get("metrics", {})

                for metric_name, metric_info in metrics_data.items():
                    safe_name = metric_name.replace("-", "_").replace(".", "_")
                    if metric_info.get("latest") is not None:
                        lines.append(f"# HELP ringrift_{safe_name} Latest {metric_name} value")
                        lines.append(f"# TYPE ringrift_{safe_name} gauge")
                        lines.append(f"ringrift_{safe_name} {metric_info['latest']}")
            except (AttributeError):
                pass

            # Data manifest totals
            if hasattr(self, 'cluster_data_manifest') and self.cluster_data_manifest:
                for config_key, config_data in self.cluster_data_manifest.by_board_type.items():
                    total_games = config_data.get("total_games", 0)
                    parts = config_key.split("_")
                    if len(parts) >= 2:
                        board_type = parts[0]
                        num_players = parts[1].replace("p", "")
                        lines.append(f'ringrift_games_total{{board_type="{board_type}",num_players="{num_players}"}} {total_games}')

            # Add header for games total
            if hasattr(self, 'cluster_data_manifest') and self.cluster_data_manifest:
                lines.insert(-len(self.cluster_data_manifest.by_board_type),
                           "# HELP ringrift_games_total Total games per board configuration")
                lines.insert(-len(self.cluster_data_manifest.by_board_type),
                           "# TYPE ringrift_games_total gauge")

            # === CRITICAL SELF-IMPROVEMENT LOOP METRICS ===

            # Training Progress Metrics
            lines.append("# HELP ringrift_training_loss Current model training loss")
            lines.append("# TYPE ringrift_training_loss gauge")
            lines.append("# HELP ringrift_training_val_loss Current model validation loss")
            lines.append("# TYPE ringrift_training_val_loss gauge")
            lines.append("# HELP ringrift_training_epoch Current training epoch")
            lines.append("# TYPE ringrift_training_epoch gauge")
            if hasattr(self, 'training_metrics'):
                for config, metrics in self.training_metrics.items():
                    loss = metrics.get('loss', 0)
                    val_loss = metrics.get('val_loss', 0)
                    epoch = metrics.get('epoch', 0)
                    lines.append(f'ringrift_training_loss{{config="{config}"}} {loss}')
                    lines.append(f'ringrift_training_val_loss{{config="{config}"}} {val_loss}')
                    lines.append(f'ringrift_training_epoch{{config="{config}"}} {epoch}')

            # Data Freshness Metrics
            lines.append("# HELP ringrift_data_freshness_hours Age of newest training data in hours")
            lines.append("# TYPE ringrift_data_freshness_hours gauge")
            lines.append("# HELP ringrift_data_staleness_hours Age of oldest training data in hours")
            lines.append("# TYPE ringrift_data_staleness_hours gauge")
            try:
                selfplay_dir = Path("data/selfplay")
                if selfplay_dir.exists():
                    for config_dir in selfplay_dir.iterdir():
                        if config_dir.is_dir() and not config_dir.name.startswith('.'):
                            jsonl_files = list(config_dir.glob("*.jsonl"))
                            if jsonl_files:
                                newest = max(f.stat().st_mtime for f in jsonl_files)
                                oldest = min(f.stat().st_mtime for f in jsonl_files)
                                freshness_hours = (now - newest) / 3600
                                staleness_hours = (now - oldest) / 3600
                                config_name = config_dir.name
                                lines.append(f'ringrift_data_freshness_hours{{config="{config_name}"}} {freshness_hours:.2f}')
                                lines.append(f'ringrift_data_staleness_hours{{config="{config_name}"}} {staleness_hours:.2f}')
            except (OSError, AttributeError, ImportError):
                pass

            # Selfplay Throughput Metrics
            lines.append("# HELP ringrift_selfplay_games_per_hour Selfplay game generation rate")
            lines.append("# TYPE ringrift_selfplay_games_per_hour gauge")
            lines.append("# HELP ringrift_selfplay_games_total_24h Total games generated in last 24h")
            lines.append("# TYPE ringrift_selfplay_games_total_24h gauge")
            if hasattr(self, 'selfplay_throughput'):
                for config, rate in self.selfplay_throughput.items():
                    lines.append(f'ringrift_selfplay_games_per_hour{{config="{config}"}} {rate}')

            # Cost Efficiency Metrics
            lines.append("# HELP ringrift_gpu_hours_total Total GPU hours consumed")
            lines.append("# TYPE ringrift_gpu_hours_total counter")
            lines.append("# HELP ringrift_estimated_cost_usd Estimated cost in USD")
            lines.append("# TYPE ringrift_estimated_cost_usd gauge")
            lines.append("# HELP ringrift_elo_per_gpu_hour Elo improvement per GPU hour")
            lines.append("# TYPE ringrift_elo_per_gpu_hour gauge")
            if hasattr(self, 'cost_metrics'):
                gpu_hours = self.cost_metrics.get('gpu_hours_total', 0)
                cost_usd = self.cost_metrics.get('estimated_cost_usd', 0)
                elo_per_hour = self.cost_metrics.get('elo_per_gpu_hour', 0)
                lines.append(f"ringrift_gpu_hours_total {gpu_hours}")
                lines.append(f"ringrift_estimated_cost_usd {cost_usd}")
                lines.append(f"ringrift_elo_per_gpu_hour {elo_per_hour}")

            # Promotion Quality Metrics
            lines.append("# HELP ringrift_promotion_success_rate Promotion success rate (0-1)")
            lines.append("# TYPE ringrift_promotion_success_rate gauge")
            lines.append("# HELP ringrift_promotion_elo_gain Average Elo gain on successful promotion")
            lines.append("# TYPE ringrift_promotion_elo_gain gauge")
            lines.append("# HELP ringrift_promotion_rejections_total Total promotion rejections by reason")
            lines.append("# TYPE ringrift_promotion_rejections_total counter")
            if hasattr(self, 'promotion_metrics'):
                success_rate = self.promotion_metrics.get('success_rate', 0)
                avg_gain = self.promotion_metrics.get('avg_elo_gain', 0)
                lines.append(f"ringrift_promotion_success_rate {success_rate}")
                lines.append(f"ringrift_promotion_elo_gain {avg_gain}")
                for reason, count in self.promotion_metrics.get('rejections', {}).items():
                    lines.append(f'ringrift_promotion_rejections_total{{reason="{reason}"}} {count}')

            # Model Evaluation Quality Metrics
            lines.append("# HELP ringrift_eval_games_played Games played in model evaluation")
            lines.append("# TYPE ringrift_eval_games_played gauge")
            lines.append("# HELP ringrift_eval_confidence Evaluation confidence (0-1)")
            lines.append("# TYPE ringrift_eval_confidence gauge")
            lines.append("# HELP ringrift_elo_uncertainty Elo rating uncertainty margin")
            lines.append("# TYPE ringrift_elo_uncertainty gauge")
            try:
                from scripts.run_model_elo_tournament import ELO_DB_PATH, init_elo_database
                if ELO_DB_PATH and ELO_DB_PATH.exists():
                    db = init_elo_database()
                    conn = db._get_connection()
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT board_type, num_players,
                               AVG(games_played) as avg_games,
                               AVG(rating_deviation) as avg_rd
                        FROM elo_ratings
                        WHERE games_played >= 5
                        GROUP BY board_type, num_players
                    """)
                    for row in cursor.fetchall():
                        bt, np, avg_games, avg_rd = row
                        config = f"{bt}_{np}p"
                        confidence = max(0, min(1, 1 - (avg_rd / 350)))  # RD 350 = 0% confidence
                        lines.append(f'ringrift_eval_games_played{{config="{config}"}} {avg_games:.1f}')
                        lines.append(f'ringrift_eval_confidence{{config="{config}"}} {confidence:.3f}')
                        lines.append(f'ringrift_elo_uncertainty{{config="{config}"}} {avg_rd:.1f}')
                    db.close()
            except (OSError, AttributeError, ImportError):
                pass

            # Improvement Loop Health Metrics
            lines.append("# HELP ringrift_improvement_cycles_total Total improvement cycles completed")
            lines.append("# TYPE ringrift_improvement_cycles_total counter")
            lines.append("# HELP ringrift_last_improvement_hours Hours since last Elo improvement")
            lines.append("# TYPE ringrift_last_improvement_hours gauge")
            lines.append("# HELP ringrift_training_queue_size Number of configs awaiting training")
            lines.append("# TYPE ringrift_training_queue_size gauge")
            if hasattr(self, 'improvement_cycle_manager') and self.improvement_cycle_manager:
                icm = self.improvement_cycle_manager
                # Count total training iterations across all cycles
                cycles_completed = sum(c.current_iteration for c in icm.state.cycles.values())
                lines.append(f"ringrift_improvement_cycles_total {cycles_completed}")

            # Victory Type Metrics by board config
            lines.append("# HELP ringrift_victory_type_total Games won by victory type")
            lines.append("# TYPE ringrift_victory_type_total counter")
            try:
                victory_stats = await self.analytics_cache_manager.get_victory_type_stats()
                for (board_type, num_players, victory_type), count in victory_stats.items():
                    lines.append(
                        f'ringrift_victory_type_total{{board_type="{board_type}",num_players="{num_players}",victory_type="{victory_type}"}} {count}'
                    )
            except (AttributeError):
                pass

            # Game Analytics Metrics
            lines.append("# HELP ringrift_game_length_avg Average game length by config")
            lines.append("# TYPE ringrift_game_length_avg gauge")
            lines.append("# HELP ringrift_games_per_hour Game generation throughput")
            lines.append("# TYPE ringrift_games_per_hour gauge")
            lines.append("# HELP ringrift_opening_diversity Unique opening moves seen")
            lines.append("# TYPE ringrift_opening_diversity gauge")
            try:
                # Use cached analytics if available
                analytics = await self.analytics_cache_manager.get_game_analytics_cached()
                for config, stats in analytics.get("configs", {}).items():
                    parts = config.rsplit("_", 1)
                    if len(parts) == 2:
                        board_type = parts[0]
                        num_players = parts[1].replace("p", "")
                        lines.append(f'ringrift_game_length_avg{{board_type="{board_type}",num_players="{num_players}"}} {stats.get("avg_length", 0)}')
                        lines.append(f'ringrift_games_per_hour{{board_type="{board_type}",num_players="{num_players}"}} {stats.get("throughput_per_hour", 0)}')
                        lines.append(f'ringrift_opening_diversity{{board_type="{board_type}",num_players="{num_players}"}} {stats.get("opening_diversity", 0)}')
            except (AttributeError):
                pass

            # Best Elo by Config
            lines.append("# HELP ringrift_best_elo Best Elo rating by config")
            lines.append("# TYPE ringrift_best_elo gauge")
            lines.append("# HELP ringrift_elo_games_played Games played by best model")
            lines.append("# TYPE ringrift_elo_games_played gauge")
            try:
                ai_root = Path(self._get_ai_service_path())
                db_path = ai_root / "data" / "unified_elo.db"
                if not db_path.exists():
                    db_path = ai_root / "data" / "unified_elo.db"
                if db_path.exists():
                    # Use context manager to prevent connection leaks
                    with safe_db_connection(db_path) as conn:
                        cursor = conn.cursor()
                        # Check which column name is used (model_id vs participant_id)
                        cursor.execute("PRAGMA table_info(elo_ratings)")
                        columns = [col[1] for col in cursor.fetchall()]
                        id_col = "model_id" if "model_id" in columns else "participant_id"
                        cursor.execute(f"""
                            SELECT board_type, num_players, MAX(rating), {id_col}, games_played
                            FROM elo_ratings
                            WHERE games_played >= 10
                            GROUP BY board_type, num_players
                        """)
                        for row in cursor.fetchall():
                            bt, np, rating, model, games = row
                            lines.append(f'ringrift_best_elo{{board_type="{bt}",num_players="{np}",model="{model}"}} {rating:.1f}')
                            lines.append(f'ringrift_elo_games_played{{board_type="{bt}",num_players="{np}",model="{model}"}} {games}')
            except (OSError, KeyError, IndexError, AttributeError, ImportError, sqlite3.Error):
                pass

            # Training Loss Metrics (from latest training)
            lines.append("# HELP ringrift_training_loss Latest training loss")
            lines.append("# TYPE ringrift_training_loss gauge")
            lines.append("# HELP ringrift_training_epoch Current training epoch")
            lines.append("# TYPE ringrift_training_epoch gauge")
            try:
                training_metrics = await self.analytics_cache_manager.get_training_metrics_cached()
                for config, data in training_metrics.get("configs", {}).items():
                    parts = config.rsplit("_", 1)
                    if len(parts) == 2 and data.get("latest_loss"):
                        board_type = parts[0]
                        num_players = parts[1].replace("p", "")
                        lines.append(f'ringrift_training_loss{{board_type="{board_type}",num_players="{num_players}"}} {data["latest_loss"]}')
                        lines.append(f'ringrift_training_epoch{{board_type="{board_type}",num_players="{num_players}"}} {data.get("latest_epoch", 0)}')
            except (AttributeError):
                pass

            # === HOLDOUT VALIDATION METRICS ===
            lines.append("# HELP ringrift_holdout_games Number of games in holdout set")
            lines.append("# TYPE ringrift_holdout_games gauge")
            lines.append("# HELP ringrift_holdout_positions Number of positions in holdout set")
            lines.append("# TYPE ringrift_holdout_positions gauge")
            lines.append("# HELP ringrift_holdout_loss Model loss on holdout validation set")
            lines.append("# TYPE ringrift_holdout_loss gauge")
            lines.append("# HELP ringrift_holdout_accuracy Model accuracy on holdout validation set")
            lines.append("# TYPE ringrift_holdout_accuracy gauge")
            lines.append("# HELP ringrift_overfit_gap Gap between holdout and training loss (positive = overfitting)")
            lines.append("# TYPE ringrift_overfit_gap gauge")
            try:
                holdout_metrics = await self.analytics_cache_manager.get_holdout_metrics_cached()
                for config, data in holdout_metrics.get("configs", {}).items():
                    parts = config.rsplit("_", 1)
                    if len(parts) == 2:
                        board_type = parts[0]
                        num_players = parts[1].replace("p", "")
                        lines.append(f'ringrift_holdout_games{{board_type="{board_type}",num_players="{num_players}"}} {data.get("holdout_games", 0)}')
                        lines.append(f'ringrift_holdout_positions{{board_type="{board_type}",num_players="{num_players}"}} {data.get("holdout_positions", 0)}')
                        if data.get("holdout_loss") is not None:
                            lines.append(f'ringrift_holdout_loss{{board_type="{board_type}",num_players="{num_players}"}} {data["holdout_loss"]}')
                        if data.get("holdout_accuracy") is not None:
                            lines.append(f'ringrift_holdout_accuracy{{board_type="{board_type}",num_players="{num_players}"}} {data["holdout_accuracy"]}')
                        if data.get("overfit_gap") is not None:
                            lines.append(f'ringrift_overfit_gap{{board_type="{board_type}",num_players="{num_players}"}} {data["overfit_gap"]}')
            except (AttributeError):
                pass

            # === MCTS SEARCH STATISTICS ===
            lines.append("# HELP ringrift_mcts_avg_nodes Average MCTS nodes visited per move")
            lines.append("# TYPE ringrift_mcts_avg_nodes gauge")
            lines.append("# HELP ringrift_mcts_max_nodes Maximum MCTS nodes visited in a move")
            lines.append("# TYPE ringrift_mcts_max_nodes gauge")
            lines.append("# HELP ringrift_mcts_avg_depth Average MCTS search depth")
            lines.append("# TYPE ringrift_mcts_avg_depth gauge")
            lines.append("# HELP ringrift_mcts_max_depth Maximum MCTS search depth")
            lines.append("# TYPE ringrift_mcts_max_depth gauge")
            lines.append("# HELP ringrift_mcts_avg_time Average time per MCTS move (seconds)")
            lines.append("# TYPE ringrift_mcts_avg_time gauge")
            try:
                mcts_stats = await self.analytics_cache_manager.get_mcts_stats_cached()
                summary = mcts_stats.get("summary", {})
                if summary.get("avg_nodes_per_move"):
                    lines.append(f'ringrift_mcts_avg_nodes {summary["avg_nodes_per_move"]:.0f}')
                if summary.get("max_nodes_per_move"):
                    lines.append(f'ringrift_mcts_max_nodes {summary["max_nodes_per_move"]}')
                if summary.get("avg_search_depth"):
                    lines.append(f'ringrift_mcts_avg_depth {summary["avg_search_depth"]:.1f}')
                if summary.get("max_search_depth"):
                    lines.append(f'ringrift_mcts_max_depth {summary["max_search_depth"]}')
                if summary.get("avg_time_per_move"):
                    lines.append(f'ringrift_mcts_avg_time {summary["avg_time_per_move"]:.3f}')
                # Per-config MCTS stats
                for config, data in mcts_stats.get("configs", {}).items():
                    parts = config.rsplit("_", 1)
                    if len(parts) == 2:
                        board_type = parts[0]
                        num_players = parts[1].replace("p", "")
                        if data.get("avg_nodes"):
                            lines.append(f'ringrift_mcts_avg_nodes{{board_type="{board_type}",num_players="{num_players}"}} {data["avg_nodes"]:.0f}')
                        if data.get("avg_depth"):
                            lines.append(f'ringrift_mcts_avg_depth{{board_type="{board_type}",num_players="{num_players}"}} {data["avg_depth"]:.1f}')
            except (AttributeError):
                pass

            # === DATA QUALITY METRICS ===
            lines.append("# HELP ringrift_data_quality_games Total games analyzed for quality")
            lines.append("# TYPE ringrift_data_quality_games gauge")
            lines.append("# HELP ringrift_data_quality_short_rate Percentage of short games (<10 moves)")
            lines.append("# TYPE ringrift_data_quality_short_rate gauge")
            lines.append("# HELP ringrift_data_quality_issues Number of data quality issues detected")
            lines.append("# TYPE ringrift_data_quality_issues gauge")
            try:
                quality = await self._get_data_quality_cached()
                for config, data in quality.get("configs", {}).items():
                    parts = config.rsplit("_", 1)
                    if len(parts) == 2:
                        board_type = parts[0]
                        num_players = parts[1].replace("p", "")
                        lines.append(f'ringrift_data_quality_games{{board_type="{board_type}",num_players="{num_players}"}} {data.get("total_games", 0)}')
                        lines.append(f'ringrift_data_quality_short_rate{{board_type="{board_type}",num_players="{num_players}"}} {data.get("short_game_rate", 0)}')
                lines.append(f'ringrift_data_quality_issues {len(quality.get("issues", []))}')
            except (AttributeError):
                pass

            # === TRAINING EFFICIENCY METRICS ===
            lines.append("# HELP ringrift_gpu_hours_total Total GPU hours used for training")
            lines.append("# TYPE ringrift_gpu_hours_total gauge")
            lines.append("# HELP ringrift_elo_per_gpu_hour Elo points gained per GPU hour")
            lines.append("# TYPE ringrift_elo_per_gpu_hour gauge")
            lines.append("# HELP ringrift_training_cost_usd Estimated training cost in USD")
            lines.append("# TYPE ringrift_training_cost_usd gauge")
            try:
                efficiency = await self._get_training_efficiency_cached()
                for config, data in efficiency.get("configs", {}).items():
                    parts = config.rsplit("_", 1)
                    if len(parts) == 2:
                        board_type = parts[0]
                        num_players = parts[1].replace("p", "")
                        lines.append(f'ringrift_gpu_hours_total{{board_type="{board_type}",num_players="{num_players}"}} {data.get("gpu_hours", 0)}')
                        lines.append(f'ringrift_elo_per_gpu_hour{{board_type="{board_type}",num_players="{num_players}"}} {data.get("elo_per_gpu_hour", 0)}')
                        lines.append(f'ringrift_training_cost_usd{{board_type="{board_type}",num_players="{num_players}"}} {data.get("estimated_cost_usd", 0)}')
                summary = efficiency.get("summary", {})
                if summary:
                    lines.append(f'ringrift_gpu_hours_total {summary.get("total_gpu_hours", 0)}')
                    lines.append(f'ringrift_training_cost_usd {summary.get("total_estimated_cost_usd", 0)}')
            except (AttributeError):
                pass

            # === MODEL LINEAGE METRICS ===
            lines.append("# HELP ringrift_model_count Total number of trained models")
            lines.append("# TYPE ringrift_model_count gauge")
            lines.append("# HELP ringrift_model_generation Latest model generation per config")
            lines.append("# TYPE ringrift_model_generation gauge")
            try:
                lineage = await self._get_model_lineage_cached()
                lines.append(f'ringrift_model_count {lineage.get("total_models", 0)}')
                for config, data in lineage.get("configs", {}).items():
                    parts = config.rsplit("_", 1)
                    if len(parts) == 2:
                        board_type = parts[0]
                        num_players = parts[1].replace("p", "")
                        lines.append(f'ringrift_model_generation{{board_type="{board_type}",num_players="{num_players}"}} {data.get("latest_generation", 0)}')
            except (AttributeError):
                pass

            # === ROLLBACK STATUS METRICS ===
            lines.append("# HELP ringrift_rollback_candidates Number of configs recommended for rollback")
            lines.append("# TYPE ringrift_rollback_candidates gauge")
            try:
                rollback = await self._check_rollback_conditions()
                lines.append(f'ringrift_rollback_candidates {len(rollback.get("candidates", []))}')
            except (AttributeError):
                pass

            # === AUTOSCALING METRICS ===
            lines.append("# HELP ringrift_autoscale_suggested_workers Suggested worker count from autoscaling")
            lines.append("# TYPE ringrift_autoscale_suggested_workers gauge")
            lines.append("# HELP ringrift_cluster_games_per_hour Current cluster-wide game generation rate")
            lines.append("# TYPE ringrift_cluster_games_per_hour gauge")
            try:
                autoscale = await self.analytics_cache_manager.get_autoscaling_metrics()
                state = autoscale.get("current_state", {})
                lines.append(f'ringrift_cluster_games_per_hour {state.get("games_per_hour", 0)}')
                recs = autoscale.get("recommendations", [])
                if recs:
                    lines.append(f'ringrift_autoscale_suggested_workers {recs[0].get("suggested_workers", state.get("total_nodes", 1))}')
                else:
                    lines.append(f'ringrift_autoscale_suggested_workers {state.get("total_nodes", 1)}')
            except (AttributeError):
                pass

            # === P2P ENHANCEMENT METRICS ===

            # Adaptive Sync Intervals
            lines.append("# HELP ringrift_sync_interval_data Current data sync interval in seconds")
            lines.append("# TYPE ringrift_sync_interval_data gauge")
            lines.append("# HELP ringrift_sync_interval_model Current model sync interval in seconds")
            lines.append("# TYPE ringrift_sync_interval_model gauge")
            lines.append("# HELP ringrift_sync_activity_factor Cluster activity factor (lower = more active)")
            lines.append("# TYPE ringrift_sync_activity_factor gauge")
            try:
                sync_summary = self._get_sync_interval_summary()
                lines.append(f'ringrift_sync_interval_data {sync_summary.get("data_interval", 300)}')
                lines.append(f'ringrift_sync_interval_model {sync_summary.get("model_interval", 180)}')
                lines.append(f'ringrift_sync_activity_factor {sync_summary.get("activity_factor", 1.0)}')
            except (AttributeError):
                pass

            # Gossip Protocol Metrics
            lines.append("# HELP ringrift_gossip_messages_sent Total gossip messages sent")
            lines.append("# TYPE ringrift_gossip_messages_sent counter")
            lines.append("# HELP ringrift_gossip_messages_received Total gossip messages received")
            lines.append("# TYPE ringrift_gossip_messages_received counter")
            lines.append("# HELP ringrift_gossip_state_updates Total state updates from gossip")
            lines.append("# TYPE ringrift_gossip_state_updates counter")
            lines.append("# HELP ringrift_gossip_compression_ratio Gossip compression ratio (1.0 = 100% compressed)")
            lines.append("# TYPE ringrift_gossip_compression_ratio gauge")
            lines.append("# HELP ringrift_gossip_bytes_saved_kb Total bytes saved by compression")
            lines.append("# TYPE ringrift_gossip_bytes_saved_kb counter")
            try:
                gossip = self._get_gossip_metrics_summary()
                lines.append(f'ringrift_gossip_messages_sent {gossip.get("message_sent", 0)}')
                lines.append(f'ringrift_gossip_messages_received {gossip.get("message_received", 0)}')
                lines.append(f'ringrift_gossip_state_updates {gossip.get("state_updates", 0)}')
                lines.append(f'ringrift_gossip_compression_ratio {gossip.get("compression_ratio", 0)}')
                lines.append(f'ringrift_gossip_bytes_saved_kb {gossip.get("bytes_saved_kb", 0)}')
            except (AttributeError):
                pass

            # Leader Consensus Metrics
            lines.append("# HELP ringrift_leader_agreement Nodes agreeing on current leader")
            lines.append("# TYPE ringrift_leader_agreement gauge")
            try:
                consensus = self._get_cluster_leader_consensus()
                lines.append(f'ringrift_leader_agreement {consensus.get("leader_agreement", 0)}')
            except (AttributeError):
                pass

            # Data Deduplication Metrics
            lines.append("# HELP ringrift_dedup_files_skipped Files skipped due to deduplication")
            lines.append("# TYPE ringrift_dedup_files_skipped counter")
            lines.append("# HELP ringrift_dedup_bytes_saved_mb Megabytes saved by deduplication")
            lines.append("# TYPE ringrift_dedup_bytes_saved_mb gauge")
            lines.append("# HELP ringrift_dedup_known_hashes Number of file hashes tracked")
            lines.append("# TYPE ringrift_dedup_known_hashes gauge")
            try:
                dedup = self._get_dedup_summary()
                lines.append(f'ringrift_dedup_files_skipped {dedup.get("files_skipped", 0)}')
                lines.append(f'ringrift_dedup_bytes_saved_mb {dedup.get("bytes_saved_mb", 0)}')
                lines.append(f'ringrift_dedup_known_hashes {dedup.get("known_file_hashes", 0)}')
            except (AttributeError):
                pass

            # Tournament Scheduling Metrics
            lines.append("# HELP ringrift_tournament_proposals_pending Pending tournament proposals")
            lines.append("# TYPE ringrift_tournament_proposals_pending gauge")
            lines.append("# HELP ringrift_tournament_active Active distributed tournaments")
            lines.append("# TYPE ringrift_tournament_active gauge")
            try:
                # Jan 28, 2026: Uses tournament_manager directly
                tourney = self.tournament_manager.get_distributed_tournament_summary()
                lines.append(f'ringrift_tournament_proposals_pending {tourney.get("pending_proposals", 0)}')
                lines.append(f'ringrift_tournament_active {tourney.get("active_tournaments", 0)}')
            except (AttributeError):
                pass

            # Work Queue Metrics (leader only)
            from scripts.p2p.handlers.work_queue import get_work_queue
            wq = get_work_queue()
            if self.is_leader and wq:
                lines.append("# HELP ringrift_work_queue_pending Work items pending in queue")
                lines.append("# TYPE ringrift_work_queue_pending gauge")
                lines.append("# HELP ringrift_work_queue_running Work items currently running")
                lines.append("# TYPE ringrift_work_queue_running gauge")
                lines.append("# HELP ringrift_work_queue_total Total work items by status")
                lines.append("# TYPE ringrift_work_queue_total gauge")
                lines.append("# HELP ringrift_work_queue_by_type Work items by type and status")
                lines.append("# TYPE ringrift_work_queue_by_type gauge")
                lines.append("# HELP ringrift_work_queue_completed_total Total completed work items")
                lines.append("# TYPE ringrift_work_queue_completed_total counter")
                lines.append("# HELP ringrift_work_queue_failed_total Total failed work items")
                lines.append("# TYPE ringrift_work_queue_failed_total counter")
                lines.append("# HELP ringrift_work_queue_timeout_total Total timed out work items")
                lines.append("# TYPE ringrift_work_queue_timeout_total counter")
                lines.append("# HELP ringrift_work_queue_cancelled_total Total cancelled work items")
                lines.append("# TYPE ringrift_work_queue_cancelled_total counter")
                lines.append("# HELP ringrift_work_queue_avg_wait_seconds Average wait time in queue")
                lines.append("# TYPE ringrift_work_queue_avg_wait_seconds gauge")
                lines.append("# HELP ringrift_work_queue_avg_run_seconds Average run time for work items")
                lines.append("# TYPE ringrift_work_queue_avg_run_seconds gauge")

                try:
                    status = wq.get_queue_status()
                    by_status = status.get("by_status", {})

                    # Basic queue counts from by_status dict
                    pending_count = by_status.get("pending", 0)
                    running_count = by_status.get("running", 0) + by_status.get("claimed", 0)
                    lines.append(f"ringrift_work_queue_pending {pending_count}")
                    lines.append(f"ringrift_work_queue_running {running_count}")
                    lines.append(f'ringrift_work_queue_total{{status="pending"}} {pending_count}')
                    lines.append(f'ringrift_work_queue_total{{status="running"}} {running_count}')

                    # Count by work type from by_type dict
                    by_type = status.get("by_type", {})
                    for wtype, count in by_type.items():
                        lines.append(f'ringrift_work_queue_by_type{{work_type="{wtype}"}} {count}')

                    # Historical counts from database
                    history = wq.get_history(limit=1000)
                    completed_count = sum(1 for h in history if h.get("status") == "completed")
                    failed_count = sum(1 for h in history if h.get("status") == "failed")
                    timeout_count = sum(1 for h in history if h.get("status") == "timeout")
                    cancelled_count = sum(1 for h in history if h.get("status") == "cancelled")

                    lines.append(f"ringrift_work_queue_completed_total {completed_count}")
                    lines.append(f"ringrift_work_queue_failed_total {failed_count}")
                    lines.append(f"ringrift_work_queue_timeout_total {timeout_count}")
                    lines.append(f"ringrift_work_queue_cancelled_total {cancelled_count}")

                    # Calculate average wait and run times from completed items
                    wait_times = []
                    run_times = []
                    for h in history:
                        if h.get("status") == "completed":
                            created = h.get("created_at", 0)
                            claimed = h.get("claimed_at", 0)
                            completed = h.get("completed_at", 0)
                            if claimed and created:
                                wait_times.append(claimed - created)
                            if completed and claimed:
                                run_times.append(completed - claimed)

                    avg_wait = sum(wait_times) / len(wait_times) if wait_times else 0
                    avg_run = sum(run_times) / len(run_times) if run_times else 0
                    lines.append(f"ringrift_work_queue_avg_wait_seconds {avg_wait:.2f}")
                    lines.append(f"ringrift_work_queue_avg_run_seconds {avg_run:.2f}")

                except (AttributeError, KeyError, ValueError, TypeError):
                    # If work queue metrics fail, just skip them
                    pass

            # Uptime metric
            if hasattr(self, 'start_time'):
                uptime = now - self.start_time
                lines.append("# HELP ringrift_orchestrator_uptime_seconds Orchestrator uptime in seconds")
                lines.append("# TYPE ringrift_orchestrator_uptime_seconds gauge")
                lines.append(f"ringrift_orchestrator_uptime_seconds {uptime:.0f}")

            return web.Response(
                text="\n".join(lines) + "\n",
                content_type="text/plain",
                charset="utf-8",
            )

        except Exception as e:  # noqa: BLE001
            return web.json_response({"success": False, "error": str(e)})

    async def handle_metrics(self, request: web.Request) -> web.Response:
        """GET /metrics - Get metrics summary and history.

        Content negotiation:
        - Accept: text/plain -> Prometheus format (same as /metrics/prometheus)
        - Accept: application/json -> JSON format
        - Default (no header) -> Prometheus format for Prometheus scraper compatibility

        January 2026 - P2P Modularization Phase 4b
        """
        try:
            # Content negotiation for Prometheus compatibility
            accept = request.headers.get("Accept", "")
            # Prometheus sends "text/plain" or "application/openmetrics-text"
            # Also check for explicit format param
            format_param = request.query.get("format", "").lower()
            if format_param == "prometheus" or "text/plain" in accept or "openmetrics" in accept or not accept:
                # Return Prometheus format
                return await self.handle_metrics_prometheus(request)

            hours = float(request.query.get("hours", "24"))
            metric_type = request.query.get("type")
            board_type = request.query.get("board_type")
            num_players_str = request.query.get("num_players")
            num_players = int(num_players_str) if num_players_str else None

            if metric_type:
                # Get specific metric history
                history = self.get_metrics_history(
                    metric_type=metric_type,
                    board_type=board_type,
                    num_players=num_players,
                    hours=hours,
                )
                return web.json_response({
                    "success": True,
                    "metric_type": metric_type,
                    "period_hours": hours,
                    "count": len(history),
                    "history": history,
                })
            else:
                # Get summary of all metrics
                summary = self.get_metrics_summary(hours=hours)
                return web.json_response({
                    "success": True,
                    **summary,
                })

        except Exception as e:  # noqa: BLE001
            return web.json_response({"success": False, "error": str(e)})

    async def handle_pipeline_status(self, request: web.Request) -> web.Response:
        """GET /pipeline/status - Get current pipeline phase status.

        January 2026 - P2P Modularization Phase 4b
        """
        if not self._is_leader() and request.query.get("local") != "1":
            proxied = await self._proxy_to_leader(request)
            if proxied.status not in (502, 503):
                return proxied
        pipeline_status = getattr(self, '_pipeline_status', {})
        return web.json_response({
            "success": True,
            "node_id": self.node_id,
            "is_leader": self._is_leader(),
            "current_job": pipeline_status,
        })

    async def handle_games_analytics(self, request: web.Request) -> web.Response:
        """GET /games/analytics - Game statistics for dashboards.

        January 2026: Moved from p2p_orchestrator.py to MetricsHandlersMixin.

        Returns aggregated game analytics including:
        - Average game length by config
        - Victory type distribution
        - Games per hour throughput
        - Opening move diversity
        """
        import json
        from collections import defaultdict

        try:
            from scripts.lib.file_formats import open_jsonl_file
        except ImportError:
            # Fallback for missing module
            def open_jsonl_file(path):
                return open(path, "r")

        try:
            # Skip JSONL scanning during startup grace period
            if self._is_in_startup_grace_period():
                return web.json_response({"configs": {}, "message": "Startup in progress"})

            hours = int(request.query.get("hours", "24"))
            cutoff = time.time() - (hours * 3600)

            ai_root = Path(self._get_ai_service_path())
            data_dirs = [
                ai_root / "data" / "games" / "daemon_sync",
                ai_root / "data" / "selfplay",
            ]

            # Aggregation containers
            game_lengths: dict[str, list[int]] = defaultdict(list)
            victory_types: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
            games_by_hour: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
            opening_moves: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
            total_games = 0

            for data_dir in data_dirs:
                if not data_dir.exists():
                    continue
                for jsonl_path in data_dir.rglob("*.jsonl"):
                    try:
                        if jsonl_path.stat().st_mtime < cutoff:
                            continue
                        with open_jsonl_file(jsonl_path) as f:
                            for line in f:
                                try:
                                    game = json.loads(line)
                                    board_type = game.get("board_type", "unknown")
                                    num_players = game.get("num_players", 0)
                                    config = f"{board_type}_{num_players}p"

                                    # Game length
                                    length = game.get("length", 0)
                                    if length > 0:
                                        game_lengths[config].append(length)

                                    # Victory type
                                    vt = game.get("victory_type", "unknown")
                                    if vt:
                                        victory_types[config][vt] += 1

                                    # Games by hour (for throughput)
                                    moves = game.get("moves", [])
                                    if moves and len(moves) > 0:
                                        hour_bucket = int(jsonl_path.stat().st_mtime // 3600)
                                        games_by_hour[config][hour_bucket] += 1

                                    # Opening moves (first 3 moves)
                                    if moves and len(moves) >= 1:
                                        first_move = str(moves[0].get("action", ""))[:20]
                                        if first_move:
                                            opening_moves[config][first_move] += 1

                                    total_games += 1
                                except json.JSONDecodeError:
                                    continue
                    except (OSError, ValueError, KeyError):
                        continue

            # Build response
            analytics = {
                "period_hours": hours,
                "total_games": total_games,
                "configs": {}
            }

            for config in set(list(game_lengths.keys()) + list(victory_types.keys())):
                lengths = game_lengths.get(config, [])
                vt = dict(victory_types.get(config, {}))
                openings = dict(opening_moves.get(config, {}))

                # Calculate throughput (games/hour)
                hourly = games_by_hour.get(config, {})
                throughput = sum(hourly.values()) / max(len(hourly), 1) if hourly else 0

                analytics["configs"][config] = {
                    "games": len(lengths),
                    "avg_length": round(sum(lengths) / len(lengths), 1) if lengths else 0,
                    "min_length": min(lengths) if lengths else 0,
                    "max_length": max(lengths) if lengths else 0,
                    "victory_types": vt,
                    "throughput_per_hour": round(throughput, 1),
                    "opening_diversity": len(openings),
                    "top_openings": dict(sorted(openings.items(), key=lambda x: -x[1])[:5]),
                }

            return web.json_response(analytics)

        except Exception as e:  # noqa: BLE001
            return web.json_response({"error": str(e)})

    async def handle_training_metrics(self, request: web.Request) -> web.Response:
        """GET /training/metrics - Training loss and accuracy metrics.

        January 2026: Moved from p2p_orchestrator.py to MetricsHandlersMixin.

        Returns recent training metrics from log files.
        """
        import json
        import re

        try:
            ai_root = Path(self._get_ai_service_path())
            logs_dir = ai_root / "logs" / "training"

            metrics = {
                "configs": {},
                "latest_training": None,
            }

            if not logs_dir.exists():
                return web.json_response(metrics)

            # Find recent training logs
            log_files = sorted(logs_dir.glob("*.log"), key=lambda f: f.stat().st_mtime, reverse=True)[:10]

            for log_file in log_files:
                try:
                    content = log_file.read_text()

                    # Extract config from filename (e.g., train_square8_2p_20251214.log)
                    config_match = re.search(r"(square\d+|hexagonal|hex)_(\d+)p", log_file.name)
                    if not config_match:
                        continue
                    config = f"{config_match.group(1)}_{config_match.group(2)}p"

                    # Parse training metrics from log
                    loss_pattern = re.compile(
                        r"[Ee]poch\s+(\d+).*?loss[=:]\s*([\d.]+).*?"
                        r"(?:policy[_\s]?loss[=:]\s*([\d.]+))?.*?"
                        r"(?:value[_\s]?loss[=:]\s*([\d.]+))?"
                    )

                    epochs = []
                    for match in loss_pattern.finditer(content):
                        epoch = int(match.group(1))
                        total_loss = float(match.group(2))
                        policy_loss = float(match.group(3)) if match.group(3) else None
                        value_loss = float(match.group(4)) if match.group(4) else None
                        epochs.append({
                            "epoch": epoch,
                            "loss": total_loss,
                            "policy_loss": policy_loss,
                            "value_loss": value_loss,
                        })

                    if epochs:
                        metrics["configs"][config] = {
                            "log_file": log_file.name,
                            "epochs": epochs[-20:],  # Last 20 epochs
                            "latest_loss": epochs[-1]["loss"] if epochs else None,
                            "latest_epoch": epochs[-1]["epoch"] if epochs else None,
                        }
                        if not metrics["latest_training"]:
                            metrics["latest_training"] = {
                                "config": config,
                                "file": log_file.name,
                                "mtime": log_file.stat().st_mtime,
                            }

                except (OSError, ValueError, KeyError):
                    continue

            return web.json_response(metrics)

        except Exception as e:  # noqa: BLE001
            return web.json_response({"error": str(e)})
