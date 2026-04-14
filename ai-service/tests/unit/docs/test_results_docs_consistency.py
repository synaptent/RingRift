"""Consistency checks for published results documentation.

These tests keep the short public results docs and the checked-in
machine-readable snapshot aligned on the current headline numbers.
"""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
DOCS_ROOT = REPO_ROOT / "docs"

TARGET_CONFIGS = ("hex8_2p", "square8_2p", "square8_3p")
OPERATOR_ENTRYPOINT_DOCS = (
    REPO_ROOT / "ai-service" / "README.md",
    REPO_ROOT / "ai-service" / "app" / "coordination" / "README.md",
    REPO_ROOT / "ai-service" / "app" / "training" / "README.md",
    REPO_ROOT / "ai-service" / "scripts" / "README.md",
    DOCS_ROOT / "ARCHITECTURE_OVERVIEW.md",
    DOCS_ROOT / "DEVELOPER_GUIDE.md",
    DOCS_ROOT / "REPOSITORY_MAP.md",
    DOCS_ROOT / "SCRIPT_INVENTORY.md",
    DOCS_ROOT / "architecture" / "MINIMAL_LOOP_CONTRACT.md",
    DOCS_ROOT / "architecture" / "TRAINING_INFRASTRUCTURE_STRATEGY.md",
)
INTEGRATION_README = REPO_ROOT / "ai-service" / "app" / "integration" / "README.md"
UTILS_README = REPO_ROOT / "ai-service" / "app" / "utils" / "README.md"
INTERFACES_README = REPO_ROOT / "ai-service" / "app" / "interfaces" / "README.md"
METRICS_README = REPO_ROOT / "ai-service" / "app" / "metrics" / "README.md"
MONITORING_README = REPO_ROOT / "ai-service" / "app" / "monitoring" / "README.md"
VALIDATION_README = REPO_ROOT / "ai-service" / "app" / "validation" / "README.md"
COORDINATION_DEPRECATED_README = REPO_ROOT / "ai-service" / "app" / "coordination" / "deprecated" / "README.md"
DISTRIBUTED_README = REPO_ROOT / "ai-service" / "app" / "distributed" / "README.md"
QUALITY_README = REPO_ROOT / "ai-service" / "app" / "quality" / "README.md"
GAME_ENGINE_README = REPO_ROOT / "ai-service" / "app" / "game_engine" / "README.md"
COORDINATION_CLUSTER_README = REPO_ROOT / "ai-service" / "app" / "coordination" / "cluster" / "README.md"
COORDINATION_PROVIDERS_README = REPO_ROOT / "ai-service" / "app" / "coordination" / "providers" / "README.md"
PROVIDERS_README = REPO_ROOT / "ai-service" / "app" / "providers" / "README.md"
COORDINATION_README = REPO_ROOT / "ai-service" / "app" / "coordination" / "README.md"
TRAINING_README = REPO_ROOT / "ai-service" / "app" / "training" / "README.md"
COORDINATION_EXPORT_TIERS_GUIDE = REPO_ROOT / "ai-service" / "app" / "coordination" / "EXPORT_TIERS.md"
TRAINING_ORCHESTRATOR_GUIDE = REPO_ROOT / "ai-service" / "app" / "training" / "ORCHESTRATOR_GUIDE.md"
COORDINATOR_GUIDE = REPO_ROOT / "ai-service" / "app" / "coordination" / "COORDINATOR_GUIDE.md"
COORDINATION_TRAINING_README = REPO_ROOT / "ai-service" / "app" / "coordination" / "training" / "README.md"
COORDINATION_DEPRECATION_GUIDE = REPO_ROOT / "ai-service" / "app" / "coordination" / "DEPRECATION_GUIDE.md"
DEPRECATED_TRAINING_README = REPO_ROOT / "ai-service" / "archive" / "deprecated_training" / "README.md"
AI_SERVICE_MIGRATION_GUIDE = REPO_ROOT / "ai-service" / "docs" / "MIGRATION_GUIDE.md"
UNIFIED_TRAINING_ORCHESTRATOR = REPO_ROOT / "ai-service" / "app" / "training" / "unified_orchestrator.py"
P2P_INTEGRATION = REPO_ROOT / "ai-service" / "app" / "integration" / "p2p_integration.py"
MODEL_LIFECYCLE = REPO_ROOT / "ai-service" / "app" / "integration" / "model_lifecycle.py"
ARCHIVED_TRAINING_ORCHESTRATOR = REPO_ROOT / "ai-service" / "archive" / "deprecated_training" / "orchestrated_training.py"
CONFIG_SOURCES_DOC = REPO_ROOT / "ai-service" / "docs" / "CONFIG_SOURCES.md"
DEPRECATION_ROADMAP_DOC = REPO_ROOT / "ai-service" / "docs" / "DEPRECATION_ROADMAP.md"
DEPRECATED_MODULES_MIGRATION_DOC = REPO_ROOT / "ai-service" / "docs" / "DEPRECATED_MODULES_MIGRATION.md"
MASTER_RUNBOOK_INDEX_DOC = REPO_ROOT / "ai-service" / "docs" / "runbooks" / "MASTER_RUNBOOK_INDEX.md"


def _extract_table_rows(path: Path) -> dict[str, list[str]]:
    rows: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cols = [col.strip().strip("`") for col in stripped.strip("|").split("|")]
        if not cols:
            continue
        config = cols[0]
        if config in TARGET_CONFIGS:
            rows[config] = cols
    return rows


def test_public_results_docs_match_results_snapshot() -> None:
    results_rows = _extract_table_rows(DOCS_ROOT / "RESULTS.md")
    research_rows = _extract_table_rows(DOCS_ROOT / "RESEARCH_SNAPSHOT.md")
    brief_rows = _extract_table_rows(DOCS_ROOT / "PROJECT_BRIEF.md")
    snapshot = json.loads((DOCS_ROOT / "data" / "results_snapshot.json").read_text(encoding="utf-8"))
    snapshot_rows = {
        item["config"]: (
            f"{float(item['best_elo']):.1f}",
            str(int(item["promotions"])),
        )
        for item in snapshot["headline"]
        if item["config"] in TARGET_CONFIGS
    }

    assert set(results_rows) == set(TARGET_CONFIGS)
    assert set(research_rows) == set(TARGET_CONFIGS)
    assert set(brief_rows) == set(TARGET_CONFIGS)
    assert set(snapshot_rows) == set(TARGET_CONFIGS)

    for config in TARGET_CONFIGS:
        results_best_elo = results_rows[config][2]
        results_promotions = results_rows[config][3]
        research_best_elo = research_rows[config][1]
        research_promotions = research_rows[config][2]
        brief_best_elo = brief_rows[config][1]
        brief_promotions = brief_rows[config][2]
        snapshot_best_elo, snapshot_promotions = snapshot_rows[config]

        assert results_best_elo == research_best_elo == brief_best_elo == snapshot_best_elo
        assert results_promotions == research_promotions == brief_promotions == snapshot_promotions


def test_current_status_doc_is_explicitly_historical() -> None:
    text = (DOCS_ROOT / "CURRENT_STATUS.md").read_text(encoding="utf-8")

    assert "# Current Status (Historical Snapshot)" in text
    assert "Historical snapshot date: April 10, 2026." in text
    assert "RESULTS.md" in text
    assert "RESEARCH_SNAPSHOT.md" in text
    assert "training_status.json" in text
    assert "DEVELOPER_GUIDE.md" in text
    assert "MINIMAL_LOOP_CONTRACT.md" in text
    assert "deploy_minimal_loops.sh" in text
    assert "progress.json" in text
    assert "metrics.jsonl" in text


def test_supported_operator_docs_share_minimal_loop_entrypoints() -> None:
    for path in OPERATOR_ENTRYPOINT_DOCS:
        text = path.read_text(encoding="utf-8")

        assert "deploy_minimal_loops.sh" in text, f"{path} should mention the supported canary rollout entrypoint"
        assert "progress.json" in text, f"{path} should mention the live trainer status file"
        assert "metrics.jsonl" in text, f"{path} should mention the durable trainer history log"


def test_todo_routes_readers_to_current_sources() -> None:
    text = (REPO_ROOT / "TODO.md").read_text(encoding="utf-8")

    assert "**Last Updated:** 2026-04-13" in text
    assert "Current Research And Runtime Sources" in text
    assert "RESULTS.md" in text
    assert "RESEARCH_SNAPSHOT.md" in text
    assert "CODEBASE_QUALITY_PROGRAM.md" in text
    assert "## Current Training Snapshot" not in text


def test_integration_readme_matches_supported_root_facade() -> None:
    text = INTEGRATION_README.read_text(encoding="utf-8")

    assert "PipelineFeedbackController" in text
    assert "FeedbackAction" in text
    assert "ModelLifecycleManager" in text
    assert "LifecycleConfig" in text
    assert "P2PIntegrationManager" in text
    assert "P2PIntegrationConfig" in text
    assert "EvaluationCurriculumBridge" in text
    assert "from app.integration import PipelineFeedback, FeedbackType" not in text
    assert "from app.integration import ModelLifecycle, ModelStage" not in text
    assert "from app.integration import P2PIntegration\n" not in text


def test_utils_readme_explains_supported_root_helpers() -> None:
    text = UTILS_README.read_text(encoding="utf-8")

    assert "The package root re-exports only the most common helpers." in text
    assert "app.utils.paths" in text
    assert "app.utils.resource_guard" in text
    assert "app.utils.canonical_naming" in text
    assert "app.utils.debug_utils" in text


def test_interfaces_readme_uses_current_hashing_example() -> None:
    text = INTERFACES_README.read_text(encoding="utf-8")

    assert "from app.core.zobrist import ZobristHash" in text
    assert "SearchCache" in text
    assert "from app.ai import MCTSNode" not in text
    assert "from app.zobrist import ZobristHasher" not in text


def test_metrics_readme_matches_supported_root_exports() -> None:
    text = METRICS_README.read_text(encoding="utf-8")

    assert "record_evaluation(" in text
    assert "record_pipeline_stage(" in text
    assert "record_pipeline_iteration(" in text
    assert "AI_ERRORS" in text
    assert "record_evaluation_result" not in text
    assert "ACTIVE_GAMES" not in text
    assert "record_job_completion" not in text


def test_monitoring_readme_separates_root_and_submodule_tools() -> None:
    text = MONITORING_README.read_text(encoding="utf-8")

    assert "MonitoringManager" in text
    assert "from app.monitoring.predictive_alerts import PredictiveAlertConfig, PredictiveAlertManager" in text
    assert "from app.monitoring.training_dashboard import DashboardServer, MetricsCollector" in text
    assert "from app.monitoring import P2PHealthMonitor" not in text
    assert "from app.monitoring import PredictiveAlertMonitor" not in text
    assert "from app.monitoring import TrainingDashboard" not in text


def test_validation_readme_matches_supported_validation_surface() -> None:
    text = VALIDATION_README.read_text(encoding="utf-8")

    assert "validate_all" in text
    assert "each_item" in text
    assert "is_non_negative" in text
    assert "is_instance" in text
    assert "is_one_of" not in text
    assert "each_value" not in text
    assert "each_key" not in text
    assert "pydantic_validator" not in text
    assert "strict=True" not in text


def test_deprecated_coordination_readme_matches_current_archive_state() -> None:
    text = COORDINATION_DEPRECATED_README.read_text(encoding="utf-8")

    assert "_deprecated_sync_coordinator.py" in text
    assert "app.coordination.event_router" in text
    assert "app.coordination.unified_health_manager" in text
    assert "app.coordination.auto_sync_daemon" in text
    assert "app.coordination.sync_router" in text
    assert "app.distributed.sync_coordinator" in text
    assert "archive/deprecated_coordination/README.md" not in text
    assert "app.coordination.core.events" not in text
    assert "from app.coordination.cluster import SyncScheduler" not in text


def test_distributed_readme_uses_real_circuit_breaker_reset_api() -> None:
    text = DISTRIBUTED_README.read_text(encoding="utf-8")

    assert "breaker = get_host_breaker()" in text
    assert 'breaker.get_state("100.x.x.1")' in text
    assert 'async with breaker.protected("100.x.x.1")' in text
    assert 'breaker.reset("100.x.x.1")' in text
    assert "breaker.reset_all()" in text
    assert 'get_host_breaker("100.x.x.1")' not in text
    assert "get_host_breaker(host)" not in text
    assert "reset_all_breakers" not in text


def test_quality_readme_uses_supported_training_entrypoints() -> None:
    text = QUALITY_README.read_text(encoding="utf-8")

    assert "from app.training.optimized_pipeline import get_optimized_pipeline" in text
    assert "pipeline = get_optimized_pipeline()" in text
    assert "pipeline.should_train(" in text
    assert "pipeline.run_training(" in text
    assert "canonical_square8_2p.db" in text
    assert "app.training.train_loop.run_training_loop" in text
    assert "from app.training import TrainingPipeline" not in text
    assert "data_filter=quality_filter" not in text


def test_game_engine_readme_matches_current_package_contract() -> None:
    text = GAME_ENGINE_README.read_text(encoding="utf-8")

    assert "from app.game_engine import GameEngine" in text
    assert "from app.game_engine import PhaseRequirement, PhaseRequirementType" in text
    assert "PhaseRequirementType.NO_TERRITORY_ACTION_REQUIRED" in text
    assert "eligible_positions=[]" in text
    assert "app._game_engine_legacy" in text
    assert "from app.board_manager import BoardManager" not in text
    assert "New code should use `app.board_manager.BoardManager` directly." not in text


def test_coordination_cluster_readme_matches_lazy_package_layout() -> None:
    text = COORDINATION_CLUSTER_README.read_text(encoding="utf-8")

    assert "from app.coordination.cluster import health" in text
    assert "health.get_cluster_health_summary()" in text
    assert "health.get_healthy_nodes()" in text
    assert "from app.coordination.cluster.health import (" in text
    assert "manager.health_check()" in text
    assert "cluster_transport.py" in text
    assert "p2p_backend.py" in text
    assert "check_node_health" not in text
    assert "| `transport.py`" not in text
    assert "| `p2p.py`" not in text


def test_coordination_providers_readme_matches_enum_root_api() -> None:
    text = COORDINATION_PROVIDERS_README.read_text(encoding="utf-8")

    assert "from app.coordination.providers import ProviderType, get_provider" in text
    assert "get_provider(ProviderType.VAST)" in text
    assert "await provider.list_instances()" in text
    assert "ProviderRegistry.get_for_node" in text
    assert "get_all_providers()" in text
    assert "get_available_gpus()" in text
    assert "async def get_instances" not in text
    assert "async def start_instance" not in text
    assert 'get_provider("lambda")' not in text
    assert "get_ssh_config" not in text
    assert "get_ringrift_path" not in text


def test_providers_readme_matches_current_manager_surface() -> None:
    text = PROVIDERS_README.read_text(encoding="utf-8")

    assert "`vast_manager.py`" in text
    assert "from app.providers import LambdaManager, VastManager" in text
    assert "vast_manager = VastManager()" in text
    assert "list_instances()" in text
    assert "get_instance(instance_id)" in text
    assert "check_health(instance)" in text
    assert "run_ssh_command(instance, command)" in text
    assert "get_instance_status(instance_id)" not in text
    assert "get_ssh_config(instance_id)" not in text


def test_coordination_readme_uses_current_facade_and_test_guidance() -> None:
    text = COORDINATION_README.read_text(encoding="utf-8")

    assert "lazy compatibility facade" in text
    assert "deploy_minimal_loops.sh" in text
    assert "tests/unit/coordination/test_package_exports.py" in text
    assert "tests/unit/coordination/test_import_hygiene.py" in text
    assert "mutants/tests/" not in text


def test_training_readme_distinguishes_local_and_supported_operator_paths() -> None:
    text = TRAINING_README.read_text(encoding="utf-8")

    assert "Local orchestration utility" in text
    assert "deploy_minimal_loops.sh" in text
    assert "minimal_alphazero_loop.py" in text
    assert "progress.json" in text
    assert "metrics.jsonl" in text
    assert "archive/deprecated_training/orchestrated_training.py" in text


def test_coordination_export_tiers_guide_matches_lazy_facade_state() -> None:
    text = COORDINATION_EXPORT_TIERS_GUIDE.read_text(encoding="utf-8")

    assert "594 exports in `__all__`, 251 LOC in `__init__.py`" in text
    assert "_exports_core.py" in text
    assert "tests/unit/coordination/test_package_exports.py" in text
    assert "tests/unit/coordination/test_import_hygiene.py" in text
    assert "568 exports" not in text
    assert "2,223 LOC" not in text
    assert "public.py" not in text


def test_training_orchestrator_guide_matches_archived_orchestrator_reality() -> None:
    text = TRAINING_ORCHESTRATOR_GUIDE.read_text(encoding="utf-8")

    assert "deploy_minimal_loops.sh" in text
    assert "minimal_alphazero_loop.py" in text
    assert "progress.json" in text
    assert "metrics.jsonl" in text
    assert "archive.deprecated_training.orchestrated_training" in text
    assert "from app.training import TrainingOrchestrator" in text
    assert "from app.training.orchestrated_training import TrainingOrchestrator" not in text
    assert "`app/training/orchestrated_training.py` → archive" not in text


def test_coordinator_guide_matches_current_orchestrator_and_singleton_guidance() -> None:
    text = COORDINATOR_GUIDE.read_text(encoding="utf-8")

    assert "deploy_minimal_loops.sh" in text
    assert "archive/deprecated_training/orchestrated_training.py" in text
    assert "@singleton" in text
    assert "prefer `@singleton`" in text
    assert "from app.training.orchestrated_training import TrainingOrchestrator" not in text


def test_coordination_training_readme_matches_current_package_exports() -> None:
    text = COORDINATION_TRAINING_README.read_text(encoding="utf-8")

    assert "get_training_coordinator" in text
    assert "get_unified_scheduler" in text
    assert "request_training_slot" in text
    assert "release_training_slot" in text
    assert "TRAINING_COMPLETED" in text
    assert "`TRAINING_COMPLETE` - Training finished successfully" not in text


def test_coordination_deprecation_guide_matches_current_sync_and_event_migrations() -> None:
    text = COORDINATION_DEPRECATION_GUIDE.read_text(encoding="utf-8")

    assert "app.coordination.sync_facade" in text
    assert "app.distributed.sync_coordinator" in text
    assert "from app.coordination.event_router import get_router, publish" in text
    assert "router = get_router()" in text
    assert "await publish(\"EVENT_TYPE\", {\"data\": \"value\"})" in text
    assert "thin root facade in `app.coordination.__init__` backed by `_exports_*.py`" in text
    assert "EventRouter.get_instance()" not in text
    assert "`SyncScheduler` (same file)" not in text
    assert "app/coordination/\n├── core/" not in text
    assert "https://github.com/anthropics/ringrift/issues" not in text


def test_deprecated_training_readme_uses_current_compatibility_import() -> None:
    text = DEPRECATED_TRAINING_README.read_text(encoding="utf-8")

    assert "archive/deprecated_training/orchestrated_training.py" in text
    assert "re-exported from\n`app.training`" in text
    assert "from app.training import TrainingOrchestrator, TrainingOrchestratorConfig" in text
    assert "app.training.orchestrated_training" in text
    assert "from app.training.orchestrated_training import TrainingOrchestrator" not in text


def test_ai_service_migration_guide_uses_current_training_compatibility_path() -> None:
    text = AI_SERVICE_MIGRATION_GUIDE.read_text(encoding="utf-8")

    assert "archive/deprecated_training/orchestrated_training.py" in text
    assert "from app.training import (" in text
    assert "The direct `app.training.orchestrated_training` module path has been removed" in text
    assert "Use that root-package compatibility import only for short-lived migrations." in text
    assert "from app.training.orchestrated_training import (" not in text


def test_training_docstrings_use_current_archived_orchestrator_story() -> None:
    unified_text = UNIFIED_TRAINING_ORCHESTRATOR.read_text(encoding="utf-8")
    p2p_text = P2P_INTEGRATION.read_text(encoding="utf-8")
    lifecycle_text = MODEL_LIFECYCLE.read_text(encoding="utf-8")
    archived_text = ARCHIVED_TRAINING_ORCHESTRATOR.read_text(encoding="utf-8")

    assert "archive/deprecated_training/orchestrated_training.py" in unified_text
    assert "app.training.train_loop.run_training_loop" in unified_text
    assert "TrainingOrchestrator (orchestrated_training.py)" not in unified_text
    assert "For higher-level pipeline orchestration, use TrainingOrchestrator." not in unified_text

    assert "archived compatibility re-export from app.training" in p2p_text
    assert "TrainingOrchestrator (orchestrated_training.py)" not in p2p_text

    assert "archived compatibility re-export from" in lifecycle_text
    assert "should only be used while migrating older code" in lifecycle_text
    assert "For training operations, use UnifiedTrainingOrchestrator or TrainingOrchestrator." not in lifecycle_text

    assert "from app.training import TrainingOrchestrator" in archived_text
    assert "from app.training.orchestrated_training import TrainingOrchestrator" not in archived_text


def test_ai_service_deprecation_docs_use_archived_training_path() -> None:
    config_text = CONFIG_SOURCES_DOC.read_text(encoding="utf-8")
    roadmap_text = DEPRECATION_ROADMAP_DOC.read_text(encoding="utf-8")
    migration_text = DEPRECATED_MODULES_MIGRATION_DOC.read_text(encoding="utf-8")

    assert "archive/deprecated_training/orchestrated_training.py" in config_text
    assert "archive/deprecated_training/orchestrated_training.py" in roadmap_text
    assert "app.training` compatibility re-export" in roadmap_text
    assert "archive/deprecated_training/orchestrated_training.py" in migration_text
    assert "app.training` compatibility re-export" in migration_text
    assert "| `orchestrated_training.py`" not in roadmap_text
    assert "| `orchestrated_training.py`" not in migration_text


def test_master_runbook_index_uses_current_coordination_helpers() -> None:
    text = MASTER_RUNBOOK_INDEX_DOC.read_text(encoding="utf-8")

    assert "from app.coordination import get_sync_scheduler" in text
    assert "print(get_sync_scheduler().get_stats())" in text
    assert "from app.coordination.event_router import get_event_stats" in text
    assert "print(get_event_stats())" in text
    assert "EventRouter.get_instance()" not in text
