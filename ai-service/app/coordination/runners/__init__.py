"""Runner category modules for daemon runner functions.

February 2026: Decomposed from daemon_runners.py into 6 category modules
for improved navigability and maintainability.

Modules:
    sync_runners: Sync + Event Processing daemons
    health_runners: Health & Monitoring daemons
    training_runners: Training, Pipeline, Evaluation & Promotion daemons
    distribution_runners: Distribution + Replication daemons
    resource_runners: Resource Management, Provider-Specific, Queue & Job daemons
    operations_runners: Feedback, Recovery, Miscellaneous, Data Integrity,
                        Cluster Availability, 48-Hour Autonomous, Data Availability daemons
"""

from __future__ import annotations

from app.coordination.runners._shared import wait_for_daemon as _wait_for_daemon


# Re-export all create_* functions from submodules
from app.coordination.runners.sync_runners import (  # noqa: E402, F401
    create_auto_sync,
    create_config_sync,
    create_config_validator,
    create_cross_process_poller,
    create_dlq_retry,
    create_elo_sync,
    create_ephemeral_sync,
    create_event_router,
    create_export_watchdog,
    create_gossip_sync,
    create_high_quality_sync,
    create_owc_import,
    create_owc_model_import,
    create_sync_coordinator,
    create_training_data_recovery,
    create_training_data_sync,
    create_training_node_watcher,
    create_training_watchdog,
    create_unevaluated_model_scanner,
    create_stale_evaluation,
    create_comprehensive_model_scan,
)
from app.coordination.runners.health_runners import (  # noqa: E402, F401
    create_canonical_model_watchdog,
    create_cluster_monitor,
    create_cluster_watchdog,
    create_coordinator_health_monitor,
    create_daemon_watchdog,
    create_health_check,
    create_health_server,
    create_model_performance_watchdog,
    create_node_health_monitor,
    create_pipeline_health_watchdog,
    create_quality_monitor,
    create_queue_monitor,
    create_system_health_monitor,
    create_work_queue_monitor,
)
from app.coordination.runners.training_runners import (  # noqa: E402, F401
    create_architecture_feedback,
    create_auto_export,
    create_auto_promotion,
    create_backlog_evaluation,
    create_continuous_training_loop,
    create_data_pipeline,
    create_elo_progress,
    create_evaluation_daemon,
    create_gauntlet_feedback,
    create_nnue_training,
    create_parity_validation,
    create_selfplay_scheduler,
    create_selfplay_coordinator,
    create_tournament_daemon,
    create_training_coordinator,
    create_training_trigger,
    create_unified_promotion,
)
from app.coordination.runners.distribution_runners import (  # noqa: E402, F401
    create_data_server,
    create_model_distribution,
    create_model_sync,
    create_npz_distribution,
    create_replication_monitor,
    create_replication_repair,
)
from app.coordination.runners.resource_runners import (  # noqa: E402, F401
    create_adaptive_resources,
    create_cluster_utilization_watchdog,
    create_idle_resource,
    create_job_scheduler,
    create_lambda_idle,
    create_multi_provider,
    create_node_recovery,
    create_queue_populator,
    create_resource_optimizer,
    create_utilization_optimizer,
    create_vast_idle,
)
from app.coordination.runners.operations_runners import (  # noqa: E402, F401
    create_availability_capacity_planner,
    create_availability_node_monitor,
    create_availability_provisioner,
    create_availability_recovery_engine,
    create_cache_coordination,
    create_cascade_training,
    create_cluster_consolidation,
    create_cluster_data_sync,
    create_comprehensive_consolidation,
    create_connectivity_recovery,
    create_coordinator_disk_manager,
    create_curriculum_integration,
    create_data_availability,
    create_data_cleanup,
    create_data_consolidation,
    create_disk_space_manager,
    create_distillation,
    create_dual_backup,
    create_external_drive_sync,
    create_fast_failure_detector,
    create_feedback_loop,
    create_integrity_check,
    create_maintenance,
    create_memory_monitor,
    create_metrics_analysis,
    create_node_availability,
    create_node_data_agent,
    create_npz_combination,
    create_online_merge,
    create_orphan_detection,
    create_owc_push,
    create_owc_sync_manager,
    create_p2p_auto_deploy,
    create_p2p_backend,
    create_p2p_recovery,
    create_per_orchestrator,
    create_production_game_import,
    create_progress_watchdog,
    create_reanalysis,
    create_recovery_orchestrator,
    create_s3_backup,
    create_s3_consolidation,
    create_s3_import,
    create_s3_node_sync,
    create_s3_push,
    create_s3_sync,
    create_socket_leak_recovery,
    create_stale_fallback,
    create_sync_push,
    create_tailscale_health,
    create_underutilization_recovery,
    create_unified_backup,
    create_unified_data_catalog,
    create_unified_data_plane,
    create_unified_data_sync_orchestrator,
    create_vast_cpu_pipeline,
    create_voter_health_monitor,
    create_pipeline_completeness_monitor,
)

__all__ = [name for name in globals() if name.startswith("create_")]


def __dir__() -> list[str]:
    """Expose only the public runner-factory surface for discoverability."""

    public_globals = {name for name in globals() if not name.startswith("_")}
    return sorted(public_globals | set(__all__))
