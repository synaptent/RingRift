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
DEPRECATION_TRACKER_DOC = REPO_ROOT / "ai-service" / "docs" / "DEPRECATION_TRACKER.md"
CONSOLIDATION_STATUS_2025_12_19_DOC = REPO_ROOT / "ai-service" / "docs" / "CONSOLIDATION_STATUS_2025_12_19.md"
LEGACY_RULES_DIFF_DOC = REPO_ROOT / "ai-service" / "docs" / "specs" / "LEGACY_RULES_DIFF.md"
AI_NEURAL_NET_INIT = REPO_ROOT / "ai-service" / "app" / "ai" / "neural_net" / "__init__.py"
STRANDED_FEATURES_DOC = REPO_ROOT / "ai-service" / "docs" / "STRANDED_FEATURES.md"
TRAIN_REFACTORING_DOC = REPO_ROOT / "ai-service" / "app" / "training" / "TRAIN_REFACTORING.md"
TITANS_IMPLEMENTATION_PLAN_DOC = REPO_ROOT / "ai-service" / "docs" / "TITANS_IMPLEMENTATION_PLAN.md"
STRATEGIC_IMPROVEMENT_PLAN_DOC = REPO_ROOT / "ai-service" / "docs" / "STRATEGIC_IMPROVEMENT_PLAN_2025_12.md"
ARCHITECTURE_NAMING_DOC = REPO_ROOT / "ai-service" / "docs" / "architecture" / "ARCHITECTURE_NAMING.md"
CONSOLIDATION_ROADMAP_DOC = REPO_ROOT / "ai-service" / "docs" / "CONSOLIDATION_ROADMAP.md"
DEPRECATION_TIMELINE_DOC = REPO_ROOT / "ai-service" / "docs" / "DEPRECATION_TIMELINE.md"
CONSOLIDATION_STATUS_2025_12_28_DOC = REPO_ROOT / "ai-service" / "docs" / "CONSOLIDATION_STATUS_2025_12_28.md"
DEPRECATED_AI_README = REPO_ROOT / "ai-service" / "archive" / "deprecated_ai" / "README.md"
APP_DEPRECATION_AUDIT = REPO_ROOT / "ai-service" / "app" / "DEPRECATION_AUDIT.md"
RULES_ENGINE_SURFACE_AUDIT_DOC = REPO_ROOT / "ai-service" / "docs" / "RULES_ENGINE_SURFACE_AUDIT.md"
CODEBASE_QUALITY_REPORT_DOC = REPO_ROOT / "ai-service" / "docs" / "CODEBASE_QUALITY_REPORT.md"
CONSOLIDATION_STATUS_2025_12_19_DOC = REPO_ROOT / "ai-service" / "docs" / "CONSOLIDATION_STATUS_2025_12_19.md"
COORDINATION_ARCHITECTURE_DOC = REPO_ROOT / "ai-service" / "docs" / "COORDINATION_ARCHITECTURE.md"
EVENT_WIRING_GUIDE_DOC = REPO_ROOT / "ai-service" / "docs" / "EVENT_WIRING_GUIDE.md"
EVENT_WIRING_DIAGRAM_DOC = REPO_ROOT / "ai-service" / "docs" / "EVENT_WIRING_DIAGRAM.md"
EVENT_SYSTEM_REFERENCE_DOC = REPO_ROOT / "ai-service" / "docs" / "EVENT_SYSTEM_REFERENCE.md"
EVENT_CATALOG_DOC = REPO_ROOT / "ai-service" / "docs" / "EVENT_CATALOG.md"
EVENT_PAYLOAD_SCHEMAS_DOC = REPO_ROOT / "ai-service" / "docs" / "EVENT_PAYLOAD_SCHEMAS.md"
ADR_EVENT_DRIVEN_ARCHITECTURE_DOC = REPO_ROOT / "ai-service" / "docs" / "adr" / "ADR-001-event-driven-architecture.md"
INTEGRATION_CHECKLIST_DOC = REPO_ROOT / "ai-service" / "docs" / "INTEGRATION_CHECKLIST.md"
EVENT_HANDLER_PATTERNS_DOC = REPO_ROOT / "ai-service" / "docs" / "coordination" / "EVENT_HANDLER_PATTERNS.md"
EVENT_SUBSCRIPTION_MATRIX_ARCH_DOC = REPO_ROOT / "ai-service" / "docs" / "architecture" / "EVENT_SUBSCRIPTION_MATRIX.md"
EVENT_WIRING_DIAGRAM_ARCH_DOC = REPO_ROOT / "ai-service" / "docs" / "architecture" / "EVENT_WIRING_DIAGRAM.md"
EVENT_FLOW_INTEGRATION_ARCH_DOC = REPO_ROOT / "ai-service" / "docs" / "architecture" / "EVENT_FLOW_INTEGRATION.md"
P2P_MANAGER_INTEGRATION_DOC = REPO_ROOT / "ai-service" / "docs" / "p2p-manager-integration.md"
EVENT_NAMING_CONVENTION_DOC = REPO_ROOT / "ai-service" / "docs" / "EVENT_NAMING_CONVENTION.md"
EVENT_WIRING_VERIFICATION_RUNBOOK = REPO_ROOT / "ai-service" / "docs" / "runbooks" / "EVENT_WIRING_VERIFICATION.md"
COORDINATION_EVENT_SYSTEM_RUNBOOK = REPO_ROOT / "ai-service" / "docs" / "runbooks" / "COORDINATION_EVENT_SYSTEM.md"
PRIORITY_ACTION_PLAN_DOC = REPO_ROOT / "ai-service" / "docs" / "PRIORITY_ACTION_PLAN_2025_12_26.md"
COORDINATOR_EVENT_AUDIT_DOC = REPO_ROOT / "ai-service" / "docs" / "audits" / "COORDINATOR_EVENT_AUDIT.md"
INTEGRATION_MIGRATION_PLAN_DOC = REPO_ROOT / "ai-service" / "docs" / "roadmaps" / "INTEGRATION_MIGRATION_PLAN.md"
RESILIENT_TRANSFER_GUIDE_DOC = REPO_ROOT / "ai-service" / "docs" / "RESILIENT_TRANSFER_GUIDE.md"
MODEL_LIFECYCLE_DOC = REPO_ROOT / "ai-service" / "docs" / "MODEL_LIFECYCLE.md"
CLUSTER_DEPLOYMENT_RUNBOOK = REPO_ROOT / "ai-service" / "docs" / "runbooks" / "cluster_deployment.md"
DAEMON_REGISTRY_DOC = REPO_ROOT / "ai-service" / "docs" / "DAEMON_REGISTRY.md"
INTEGRATION_ASSESSMENT_DEC2025_DOC = REPO_ROOT / "ai-service" / "docs" / "planning" / "INTEGRATION_ASSESSMENT_DEC2025.md"
EXPERIMENTAL_AI_DOC = REPO_ROOT / "ai-service" / "docs" / "EXPERIMENTAL_AI.md"
TRAINING_EXPERIMENTAL_ALGORITHMS_DOC = REPO_ROOT / "ai-service" / "docs" / "training" / "EXPERIMENTAL_ALGORITHMS.md"
EBMO_RESULTS_DOC = REPO_ROOT / "ai-service" / "docs" / "EBMO_RESULTS.md"
PARITY_GATE_RESOLUTION_RUNBOOK = REPO_ROOT / "ai-service" / "docs" / "runbooks" / "PARITY_GATE_RESOLUTION.md"
PARITY_MISMATCH_DEBUG_RUNBOOK = REPO_ROOT / "ai-service" / "docs" / "runbooks" / "PARITY_MISMATCH_DEBUG.md"
AI_SERVICE_ARCHITECTURE_OVERVIEW_DOC = (
    REPO_ROOT / "ai-service" / "docs" / "architecture" / "ARCHITECTURE_OVERVIEW.md"
)
NEURAL_AI_ARCHITECTURE_DOC = REPO_ROOT / "ai-service" / "docs" / "architecture" / "NEURAL_AI_ARCHITECTURE.md"
AI_NEURAL_NET_ANALYSIS_DOC = REPO_ROOT / "ai-service" / "app" / "ai" / "neural_net_analysis.md"
AI_INIT_MODULE = REPO_ROOT / "ai-service" / "app" / "ai" / "__init__.py"
GENERATE_DATA_MODULE = REPO_ROOT / "ai-service" / "app" / "training" / "generate_data.py"
TOURNAMENT_DAEMON_MODULE = REPO_ROOT / "ai-service" / "app" / "coordination" / "tournament_daemon.py"


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
    assert (
        "TrainingOrchestrator` from the archived `archive/deprecated_training/orchestrated_training.py` implementation"
        in text
    )
    assert "from app.training import (" in text
    assert "The direct `app.training.orchestrated_training` module path has been removed" in text
    assert "Use that root-package compatibility import only for short-lived migrations." in text
    assert "from app.training.orchestrated_training import (" not in text
    assert "| `orchestrated_training.py`" not in text


def test_ai_service_migration_guide_uses_current_neural_net_facade_story() -> None:
    text = AI_SERVICE_MIGRATION_GUIDE.read_text(encoding="utf-8")

    assert "`app.ai.neural_net` facade + architecture submodules" in text
    assert "from app.ai.neural_net import create_model_for_board" in text
    assert "from app.models import BoardType" in text
    assert "model = create_model_for_board(" in text
    assert "RingRiftNet" not in text
    assert "neural_net/network.py" not in text
    assert "`app.ai._neural_net_legacy`" in text


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


def test_historical_planning_docs_use_archived_training_orchestrator_path() -> None:
    tracker_text = DEPRECATION_TRACKER_DOC.read_text(encoding="utf-8")
    status_text = CONSOLIDATION_STATUS_2025_12_19_DOC.read_text(encoding="utf-8")
    strategic_text = STRATEGIC_IMPROVEMENT_PLAN_DOC.read_text(encoding="utf-8")
    naming_text = ARCHITECTURE_NAMING_DOC.read_text(encoding="utf-8")
    roadmap_text = CONSOLIDATION_ROADMAP_DOC.read_text(encoding="utf-8")

    assert "archive/deprecated_training/orchestrated_training.py" in tracker_text
    assert "app.training` compatibility re-export" in tracker_text
    assert "archive/deprecated_training/orchestrated_training.py" in status_text
    assert "archived compatibility layer re-exported from `app.training`" in status_text
    assert "archive/deprecated_training/orchestrated_training.py" in strategic_text
    assert "re-exported from `app.training`" in strategic_text
    assert "archive/deprecated_training/orchestrated_training.py" in naming_text
    assert "app.training` compatibility re-export" in naming_text
    assert "archive/deprecated_training/orchestrated_training.py" in roadmap_text
    assert "compatibility re-export from `app.training`" in roadmap_text
    assert "app/training/orchestrated_training.py" not in status_text
    assert "app/training/orchestrated_training.py" not in strategic_text
    assert "app/training/orchestrated_training.py" not in roadmap_text
    assert "| `orchestrated_training.py`" not in tracker_text
    assert "| `orchestrated_training.py`" not in naming_text


def test_deprecation_timeline_matches_current_sync_and_game_engine_surfaces() -> None:
    text = DEPRECATION_TIMELINE_DOC.read_text(encoding="utf-8")

    assert "`app/coordination/sync_coordinator.py`      | Deprecated shim" in text
    assert "`auto_sync_daemon.py` + `sync_facade.py` + `app/distributed/sync_coordinator.py`" in text
    assert "`app/_game_engine_legacy.py` | Deprecated compatibility symlink" in text
    assert "`app.game_engine` stable facade" in text
    assert "Migrate remaining direct `app._game_engine_legacy` callers to `app.game_engine`" in text
    assert "archive/deprecated_ai/_game_engine_legacy.py" in text
    assert "from app._game_engine_legacy import GameEngine" in text
    assert "from app.game_engine import GameEngine" in text
    assert "Direct import from app._game_engine_legacy is deprecated." in text
    assert "DefaultRulesEngine" not in text
    assert "app/rules/default_engine.py" not in text


def test_sync_consolidation_status_tracks_current_shim_story() -> None:
    text = CONSOLIDATION_STATUS_2025_12_28_DOC.read_text(encoding="utf-8")

    assert "thin deprecated shim" in text
    assert "app/coordination/deprecated/_deprecated_sync_coordinator.py" in text
    assert "`auto_sync_daemon.py`, `sync_facade.py`, and `app/distributed/sync_coordinator.py`" in text
    assert "#### 1. Sync Coordinator Shim Retirement" in text
    assert "Rename `app/coordination/sync_coordinator.py`" not in text
    assert "rename to `sync_scheduler.py`" not in text


def test_deprecated_ai_readme_uses_current_game_engine_facade_story() -> None:
    text = DEPRECATED_AI_README.read_text(encoding="utf-8")

    assert "archived AI and engine modules kept for compatibility" in text
    assert "**Replacement**: Use `app.game_engine` instead." in text
    assert "from app._game_engine_legacy import GameEngine" in text
    assert "from app.game_engine import GameEngine" in text
    assert "`app/_game_engine_legacy.py` compatibility symlink" in text
    assert "`app/game_engine/__init__.py`" in text
    assert "app/rules/game_engine.py" not in text


def test_deprecated_ai_readme_uses_current_neural_net_facade_story() -> None:
    text = DEPRECATED_AI_README.read_text(encoding="utf-8")

    assert "supported public API for active code is now the stable `app.ai.neural_net`" in text
    assert "from app.ai._neural_net_legacy import NeuralNetAI, encode_move_for_board" in text
    assert "from app.ai.neural_net import NeuralNetAI, encode_move_for_board" in text
    assert "`app/ai/_neural_net_legacy.py` compatibility symlink" in text
    assert "`app/ai/neural_net/__init__.py`" in text
    assert "`app.ai.nnue_policy` directly" in text


def test_app_deprecation_audit_uses_archived_game_engine_path() -> None:
    text = APP_DEPRECATION_AUDIT.read_text(encoding="utf-8")

    assert "archive/deprecated_ai/_game_engine_legacy.py" in text
    assert "154 (via `app.game_engine`)" in text
    assert "Keep `app.game_engine` as the stable public surface" in text
    assert "Retire direct `app._game_engine_legacy` imports after all callers use `app.game_engine`" in text
    assert "`app.game_engine` facade" in text
    assert "Delete `_neural_net_legacy.py` and `_game_engine_legacy.py`" not in text


def test_app_deprecation_audit_uses_neural_net_facade_story() -> None:
    text = APP_DEPRECATION_AUDIT.read_text(encoding="utf-8")

    assert "| `app/ai/_neural_net_legacy.py` | DEPRECATED | `app.ai.neural_net` facade" in text
    assert "154 (via `app.ai.neural_net`)" in text
    assert "Retire direct `app.ai._neural_net_legacy` imports after callers use `app.ai.neural_net`" in text
    assert "| `app/ai/neural_net/` (package) | DEPRECATED" not in text
    assert "Delete `_neural_net_legacy.py`" not in text


def test_rules_engine_surface_audit_uses_current_game_engine_facade_story() -> None:
    text = RULES_ENGINE_SURFACE_AUDIT_DOC.read_text(encoding="utf-8")

    assert "`app/game_engine/` + `app/_game_engine_legacy.py` compatibility path" in text
    assert "backs the stable\n`app.game_engine` facade" in text
    assert "`app/_game_engine_legacy.py` | **Deprecated compatibility path**" in text
    assert "Keep callers on `app.game_engine` while decomposition continues" in text
    assert "remains the primary rules execution engine" not in text


def test_codebase_quality_report_uses_game_engine_facade_retirement_wording() -> None:
    text = CODEBASE_QUALITY_REPORT_DOC.read_text(encoding="utf-8")

    assert "Reduce direct imports behind `app.game_engine`, then retire compatibility path" in text
    assert "Archive (Python rules deprecated)" not in text


def test_codebase_quality_report_uses_neural_net_facade_retirement_wording() -> None:
    text = CODEBASE_QUALITY_REPORT_DOC.read_text(encoding="utf-8")

    assert "Reduce direct imports behind `app.ai.neural_net`, then retire compatibility path" in text
    assert "Archive or split by architecture" not in text


def test_consolidation_status_2025_12_19_uses_archived_game_engine_path() -> None:
    text = CONSOLIDATION_STATUS_2025_12_19_DOC.read_text(encoding="utf-8")

    assert "archive/deprecated_ai/_game_engine_legacy.py` via `app/_game_engine_legacy.py`" in text
    assert "Still backing 3 compatibility callers/facades" in text
    assert "stable `app.game_engine` facade" in text
    assert "NOT ready for compatibility retirement" in text


def test_consolidation_status_2025_12_19_uses_archived_neural_net_path() -> None:
    text = CONSOLIDATION_STATUS_2025_12_19_DOC.read_text(encoding="utf-8")

    assert "archive/deprecated_ai/_neural_net_legacy.py` via `app/ai/_neural_net_legacy.py`" in text
    assert "NOT ready for compatibility retirement" in text
    assert "`app.ai.neural_net` facade" in text


def test_legacy_rules_diff_points_to_archived_game_engine_path() -> None:
    text = LEGACY_RULES_DIFF_DOC.read_text(encoding="utf-8")

    assert "archive/deprecated_ai/_game_engine_legacy.py` via `app/_game_engine_legacy.py" in text
    assert "| Legacy engine" in text


def test_neural_net_package_docstring_uses_supported_facade_story() -> None:
    text = AI_NEURAL_NET_INIT.read_text(encoding="utf-8")

    assert "stable public surface for RingRift neural-network models" in text
    assert "Active code should import from" in text
    assert "app.ai._neural_net_legacy" in text
    assert "compatibility path" in text
    assert "archived monolithic implementation" in text


def test_neural_net_deprecation_docs_use_stable_package_replacements() -> None:
    timeline_text = DEPRECATION_TIMELINE_DOC.read_text(encoding="utf-8")
    tracker_text = DEPRECATION_TRACKER_DOC.read_text(encoding="utf-8")

    assert "`app/ai/_neural_net_legacy.py` | Deprecated compatibility symlink" in timeline_text
    assert "`app/ai/neural_net` stable facade" in timeline_text
    assert "`_neural_net_legacy.py`  | Dec 2025      | `app/ai/neural_net`" in tracker_text
    assert "`_game_engine_legacy.py` | Dec 2025      | `app/game_engine`" in tracker_text
    assert "app/ai/neural_net/*" not in timeline_text
    assert "app/ai/neural_net/*" not in tracker_text


def test_stranded_features_and_train_refactoring_use_neural_net_facade_story() -> None:
    stranded_text = STRANDED_FEATURES_DOC.read_text(encoding="utf-8")
    train_refactoring_text = TRAIN_REFACTORING_DOC.read_text(encoding="utf-8")

    assert "The `app.ai._neural_net_legacy` compatibility path now routes" in stranded_text
    assert "Retire behind `app.ai.neural_net` facade" in stranded_text
    assert "Migrated to `neural_net/` package" not in stranded_text
    assert "Active callers should stay on the `app.ai.neural_net` facade" in train_refactoring_text


def test_titans_plan_uses_supported_neural_net_surface() -> None:
    text = TITANS_IMPLEMENTATION_PLAN_DOC.read_text(encoding="utf-8")

    assert "Changes to `app/ai/neural_net/` architecture modules" in text
    assert "class NeuralNetWithMemory(nn.Module):" in text
    assert "def __init__(self, base_model: nn.Module, memory_config: MemoryConfig):" in text
    assert "create_model_for_board(...)" in text
    assert "Changes to `app/ai/ringrift_net.py`" not in text
    assert "app/ai/ringrift_net.py" not in text
    assert "RingRiftNetWithMemory" not in text


def test_event_system_docs_use_data_events_package_paths() -> None:
    coordination_text = COORDINATION_ARCHITECTURE_DOC.read_text(encoding="utf-8")
    wiring_text = EVENT_WIRING_GUIDE_DOC.read_text(encoding="utf-8")
    diagram_text = EVENT_WIRING_DIAGRAM_DOC.read_text(encoding="utf-8")
    reference_text = EVENT_SYSTEM_REFERENCE_DOC.read_text(encoding="utf-8")
    catalog_text = EVENT_CATALOG_DOC.read_text(encoding="utf-8")
    payload_text = EVENT_PAYLOAD_SCHEMAS_DOC.read_text(encoding="utf-8")
    adr_text = ADR_EVENT_DRIVEN_ARCHITECTURE_DOC.read_text(encoding="utf-8")

    assert "### 2. DataEventBus (`app/distributed/data_events/`)" in coordination_text
    assert "Add to `app/distributed/data_events/event_types.py`:" in wiring_text
    assert "Add to `app/distributed/data_events/emit.py`:" in wiring_text
    assert "`app/distributed/data_events/event_types.py`" in diagram_text
    assert "`app/distributed/data_events/event_types.py` - Event type definitions" in reference_text
    assert "re-exported via `app.distributed.data_events`" in catalog_text
    assert "Add to `DataEventType` enum in `app/distributed/data_events/event_types.py`" in catalog_text
    assert "`app/distributed/data_events/event_types.py` - DataEventType enum" in payload_text
    assert "`app/distributed/data_events/event_types.py`): 151 event types" in adr_text
    assert "app/distributed/data_events.py" not in coordination_text
    assert "app/distributed/data_events.py" not in wiring_text
    assert "app/distributed/data_events.py" not in diagram_text
    assert "app/distributed/data_events.py" not in reference_text
    assert "app/distributed/data_events.py" not in catalog_text
    assert "app/distributed/data_events.py" not in payload_text
    assert "app/distributed/data_events.py" not in adr_text


def test_event_integration_guides_use_data_events_package_paths() -> None:
    checklist_text = INTEGRATION_CHECKLIST_DOC.read_text(encoding="utf-8")
    patterns_text = EVENT_HANDLER_PATTERNS_DOC.read_text(encoding="utf-8")
    matrix_text = EVENT_SUBSCRIPTION_MATRIX_ARCH_DOC.read_text(encoding="utf-8")
    diagram_text = EVENT_WIRING_DIAGRAM_ARCH_DOC.read_text(encoding="utf-8")
    flow_text = EVENT_FLOW_INTEGRATION_ARCH_DOC.read_text(encoding="utf-8")

    assert "app/distributed/data_events/event_types.py" in checklist_text
    assert "app/distributed/data_events/emit.py" in checklist_text
    assert "app/distributed/data_events/__init__.py" in checklist_text
    assert "app/distributed/data_events/event_types.py" in patterns_text
    assert "app/distributed/data_events/event_types.py" in matrix_text
    assert "app/distributed/data_events/event_types.py" in diagram_text
    assert "app/distributed/data_events/event_types.py" in flow_text
    assert "app/coordination/data_events.py" not in checklist_text
    assert "app/coordination/data_events.py" not in patterns_text
    assert "app/coordination/data_events.py" not in matrix_text
    assert "app/coordination/data_events.py" not in diagram_text
    assert "app/coordination/data_events.py" not in flow_text


def test_p2p_manager_guide_uses_supported_circuit_breaker_surface() -> None:
    text = P2P_MANAGER_INTEGRATION_DOC.read_text(encoding="utf-8")

    assert "from app.distributed import (" in text
    assert "CircuitOpenError" in text
    assert "get_host_breaker" in text
    assert 'self._circuit_target = "my_manager_external"' in text
    assert "status = self._circuit.get_status(self._circuit_target)" in text
    assert "app/distributed/circuit_breaker.py" in text
    assert "app/coordination/circuit_breaker.py" not in text
    assert "CircuitBreakerConfig" not in text


def test_additional_event_docs_use_data_events_package_layout() -> None:
    naming_text = EVENT_NAMING_CONVENTION_DOC.read_text(encoding="utf-8")
    wiring_runbook_text = EVENT_WIRING_VERIFICATION_RUNBOOK.read_text(encoding="utf-8")
    event_system_text = COORDINATION_EVENT_SYSTEM_RUNBOOK.read_text(encoding="utf-8")
    priority_plan_text = PRIORITY_ACTION_PLAN_DOC.read_text(encoding="utf-8")
    audit_text = COORDINATOR_EVENT_AUDIT_DOC.read_text(encoding="utf-8")
    migration_text = INTEGRATION_MIGRATION_PLAN_DOC.read_text(encoding="utf-8")

    assert "app/distributed/data_events/event_types.py" in naming_text
    assert "ai-service/app/distributed/data_events/" in wiring_runbook_text
    assert "app/distributed/data_events/event_types.py" in event_system_text
    assert "re-exported via `app.distributed.data_events`" in event_system_text
    assert "app/distributed/data_events/event_types.py" in priority_plan_text
    assert "app/distributed/data_events/event_types.py" in audit_text
    assert "app/distributed/data_events/event_types.py" in migration_text
    assert "app/distributed/data_events.py" not in naming_text
    assert "app/distributed/data_events.py" not in wiring_runbook_text
    assert "app/distributed/data_events.py" not in event_system_text
    assert "app/distributed/data_events.py" not in priority_plan_text
    assert "app/distributed/data_events.py" not in audit_text
    assert "app/distributed/data_events.py" not in migration_text


def test_distribution_docs_use_unified_distribution_surfaces() -> None:
    transfer_text = RESILIENT_TRANSFER_GUIDE_DOC.read_text(encoding="utf-8")
    lifecycle_text = MODEL_LIFECYCLE_DOC.read_text(encoding="utf-8")
    deployment_text = CLUSTER_DEPLOYMENT_RUNBOOK.read_text(encoding="utf-8")
    registry_text = DAEMON_REGISTRY_DOC.read_text(encoding="utf-8")

    assert "app/coordination/unified_distribution_daemon.py" in transfer_text
    assert "DistributionConfig(" in transfer_text
    assert "get_distribution_daemon(config)" in transfer_text
    assert "bittorrent_threshold_bytes=50_000_000" in transfer_text
    assert "NPZ_EXPORT_COMPLETE" in transfer_text
    assert "ModelDistributionConfig(" not in transfer_text
    assert "ModelDistributionDaemon(config)" not in transfer_text

    assert "MODEL_DISTRIBUTION" in lifecycle_text
    assert "UnifiedDistributionDaemon" in lifecycle_text
    assert "NPZDistributionDaemon" not in lifecycle_text
    assert "ModelDistributionDaemon" not in lifecycle_text

    assert "`MODEL_DISTRIBUTION` daemon" in deployment_text
    assert "UnifiedDistributionDaemon" in deployment_text
    assert "ModelDistributionDaemon" not in deployment_text

    assert "`create_model_sync()` → Compatibility wrapper backed by `UnifiedDistributionDaemon`" in registry_text
    assert "`create_model_distribution()` → Starts the `MODEL_DISTRIBUTION` runner backed by `UnifiedDistributionDaemon`" in registry_text
    assert "`create_npz_distribution()` → Deprecated no-op compatibility runner" in registry_text
    assert "Creates `ModelSyncDaemon`" not in registry_text
    assert "Creates `ModelDistributionDaemon`" not in registry_text
    assert "Creates `NPZDistributionDaemon`" not in registry_text


def test_distribution_planning_and_migration_docs_use_unified_distribution_story() -> None:
    strategic_text = STRATEGIC_IMPROVEMENT_PLAN_DOC.read_text(encoding="utf-8")
    migration_text = DEPRECATED_MODULES_MIGRATION_DOC.read_text(encoding="utf-8")

    assert "UnifiedDistributionDaemon: Event-driven model sync on MODEL_PROMOTED" in strategic_text
    assert "ModelDistributionDaemon: Event-driven model sync on MODEL_PROMOTED" not in strategic_text

    assert "DistributionConfig" in migration_text
    assert "get_distribution_daemon(config)" in migration_text
    assert 'wait_for_model_distribution("square8", 2, timeout=300)' in migration_text
    assert 'check_model_availability("square8", 2)' in migration_text
    assert "Unified daemon handles both models and NPZ via MODEL_PROMOTED /" in migration_text
    assert "await daemon.distribute(DataType.MODEL" not in migration_text
    assert "await daemon.distribute(DataType.NPZ" not in migration_text


def test_data_events_tracker_and_assessment_use_package_layout() -> None:
    tracker_text = DEPRECATION_TRACKER_DOC.read_text(encoding="utf-8")
    assessment_text = INTEGRATION_ASSESSMENT_DEC2025_DOC.read_text(encoding="utf-8")

    assert "`data_events` package aliases" in tracker_text
    assert "`app/distributed/data_events/`" in tracker_text
    assert "Direct enum imports / package helpers" in tracker_text
    assert "`app/distributed/data_events/`" in assessment_text
    assert "`event_types.py`, `event_bus.py`, `emit.py`; 4,309 lines total" in assessment_text
    assert "`app/distributed/data_events.py`" not in tracker_text
    assert "`app/distributed/data_events.py`" not in assessment_text


def test_experimental_ai_docs_use_current_module_paths() -> None:
    experimental_text = EXPERIMENTAL_AI_DOC.read_text(encoding="utf-8")
    training_text = TRAINING_EXPERIMENTAL_ALGORITHMS_DOC.read_text(encoding="utf-8")
    ebmo_text = EBMO_RESULTS_DOC.read_text(encoding="utf-8")

    assert "app/ai/ebmo_online_learner.py" in experimental_text
    assert "archive/deprecated_ai/gmo_ai.py" in experimental_text
    assert "app/ai/archive/cage_ai.py" in experimental_text
    assert "app/ai/archive/cage_network.py" in experimental_text
    assert "app/ai/gpu_minimax_ai.py" in experimental_text
    assert "app/ai/ebmo_online.py" not in experimental_text
    assert "app/ai/gmo_network.py" not in experimental_text
    assert "app/ai/cage_ai.py" not in experimental_text
    assert "app/ai/cage_network.py" not in experimental_text
    assert "app/ai/gpu_minimax.py" not in experimental_text

    assert "from app.ai.gmo_ai import GMOAI" in training_text
    assert "ai = GMOAI(player_number=1, config=config)" in training_text
    assert "app/ai/archive/cage_ai.py" in training_text
    assert "app/ai/archive/cage_network.py" in training_text
    assert "from app.ai.archive.cage_network import CAGEConfig" in training_text
    assert "from app.ai.archive.cage_ai import CAGE_AI" in training_text
    assert "Archived under `app.ai.archive`" in training_text
    assert "from app.ai.gmo_ai import GMO_AI" not in training_text
    assert "app/ai/cage_ai.py" not in training_text
    assert "app/ai/cage_network.py" not in training_text

    assert "from app.ai.ebmo_online_learner import EBMOOnlineAI, EBMOOnlineConfig" in ebmo_text
    assert "`app/ai/ebmo_online_learner.py`" in ebmo_text
    assert "from app.ai.ebmo_online import EBMOOnlineAI, EBMOOnlineConfig" not in ebmo_text
    assert "`app/ai/ebmo_online.py`" not in ebmo_text


def test_parity_runbooks_use_live_rules_and_replay_paths() -> None:
    resolution_text = PARITY_GATE_RESOLUTION_RUNBOOK.read_text(encoding="utf-8")
    mismatch_text = PARITY_MISMATCH_DEBUG_RUNBOOK.read_text(encoding="utf-8")

    assert "../scripts/selfplay-db-ts-replay.ts" in resolution_text
    assert "app/board_manager.py" in resolution_text
    assert "app/rules/serialization.py" in resolution_text
    assert "src/shared/types/game.ts" in resolution_text
    assert "app/rules/phase_machine.py" in resolution_text
    assert "app/rules/fsm.py" in resolution_text
    assert "src/shared/engine/orchestration/turnOrchestrator.ts" in resolution_text
    assert "src/shared/engine/fsm/TurnStateMachine.ts" in resolution_text
    assert "app/rules/generators/territory.py" in resolution_text
    assert "app/rules/mutators/territory.py" in resolution_text
    assert "app/rules/validators/territory.py" in resolution_text
    assert "src/shared/engine/territoryDetection.ts" in resolution_text
    assert "src/shared/engine/territoryProcessing.ts" in resolution_text
    assert "src/shared/engine/aggregates/TerritoryAggregate.ts" in resolution_text
    assert "app/rules/generators/capture.py" in resolution_text
    assert "app/rules/mutators/capture.py" in resolution_text
    assert "app/rules/validators/capture.py" in resolution_text
    assert "app/rules/capture_chain.py" in resolution_text
    assert "src/shared/engine/captureLogic.ts" in resolution_text
    assert "src/shared/engine/aggregates/CaptureAggregate.ts" in resolution_text
    assert "app/rules/coordinate_transforms.py" not in resolution_text
    assert "app/rules/phase_transitions.py" not in resolution_text
    assert "app/rules/territory.py" not in resolution_text
    assert "app/rules/capture.py" not in resolution_text

    assert "../scripts/selfplay-db-ts-replay.ts" in mismatch_text
    assert "scripts/selfplay-db-ts-replay.py" not in mismatch_text
    assert "app/board_manager.py" in mismatch_text
    assert "app/rules/default_engine.py" in mismatch_text
    assert "app/rules/mutators/territory.py" in mismatch_text
    assert "app/rules/elimination.py" in mismatch_text
    assert "app/rules/global_actions.py" in mismatch_text
    assert "app/rules/generators/*.py" in mismatch_text
    assert "app/rules/validators/territory.py" in mismatch_text
    assert "app/rules/board_manager.py" not in mismatch_text
    assert "app/rules/scoring.py" not in mismatch_text
    assert "app/rules/anm_detection.py" not in mismatch_text
    assert "app/rules/move_generator.py" not in mismatch_text
    assert "app/rules/territory.py" not in mismatch_text


def test_ai_service_architecture_overview_uses_current_rules_layout() -> None:
    text = AI_SERVICE_ARCHITECTURE_OVERVIEW_DOC.read_text(encoding="utf-8")

    assert "`models/core.py`" in text
    assert "`rules/default_engine.py`" in text
    assert "`game_engine/__init__.py`" in text
    assert "`board_manager.py`" in text
    assert "`rules/mutators/territory.py` + `rules/validators/territory.py`" in text
    assert "`rules/elimination.py`" in text
    assert "| `game_state.py`" not in text
    assert "| `territory.py`" not in text
    assert "| `forced_elimination.py`" not in text


def test_neural_net_docs_use_package_facet_and_architecture_modules() -> None:
    architecture_text = NEURAL_AI_ARCHITECTURE_DOC.read_text(encoding="utf-8")
    analysis_text = AI_NEURAL_NET_ANALYSIS_DOC.read_text(encoding="utf-8")
    ai_init_text = AI_INIT_MODULE.read_text(encoding="utf-8")
    generate_data_text = GENERATE_DATA_MODULE.read_text(encoding="utf-8")
    tournament_text = TOURNAMENT_DAEMON_MODULE.read_text(encoding="utf-8")

    assert "ai-service/app/ai/neural_net/__init__.py" in architecture_text
    assert "app/ai/neural_net/hex_architectures.py" in architecture_text
    assert "app/ai/neural_net/model_factory.py" in architecture_text
    assert "app/ai/neural_ai.py" not in architecture_text
    assert "app/ai/neural_net.py" not in architecture_text

    assert "app/ai/neural_net/" in analysis_text
    assert "app/ai/_neural_net_legacy.py" in analysis_text
    assert "ai-service/app/ai/neural_net.py" not in analysis_text

    assert "neural_net/: Stable neural-network package facade and architecture modules" in ai_init_text
    assert "neural_net.py: Neural network models" not in ai_init_text

    assert "app.ai.neural_net" in generate_data_text
    assert "app/ai/neural_net.py" not in generate_data_text

    assert "app/ai/unified_factory.py" in tournament_text
    assert "app.ai.neural_net package facade" in tournament_text
    assert "app/ai/neural_net.py" not in tournament_text
