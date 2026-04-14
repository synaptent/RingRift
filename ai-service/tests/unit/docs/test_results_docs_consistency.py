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
