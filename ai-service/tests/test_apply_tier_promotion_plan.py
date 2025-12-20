"""Tests for the tier promotion-plan application helper."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.config.ladder_config import get_ladder_tier_config  # type: ignore[import]
from app.models import BoardType  # type: ignore[import]
import scripts.apply_tier_promotion_plan as apply_plan  # type: ignore[import]  # noqa: E402


def _make_promotion_plan(tier: str, candidate_id: str) -> dict[str, Any]:
    """Construct a minimal promotion plan JSON compatible with the helper."""
    difficulty = int(tier[1:])
    cfg = get_ladder_tier_config(
        difficulty=difficulty,
        board_type=BoardType.SQUARE8,
        num_players=2,
    )

    return {
        "tier": tier,
        "board_type": "square8",
        "num_players": 2,
        "current_model_id": cfg.model_id,
        "candidate_model_id": candidate_id,
        "decision": "promote",
        "timestamp": "2025-01-01T00:00:00Z",
        "reason": {
            "overall_pass": True,
            "use_candidate_artifact": True,
            "candidate_artifact_present": True,
            "candidate_artifact_loaded": None,
        },
    }


def test_apply_tier_promotion_plan_updates_registry_and_writes_artefacts(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Happy path: promote decision updates registry and writes outputs."""
    run_dir_path = Path(tmp_path)
    plan_path = run_dir_path / "promotion_plan.json"
    registry_path = run_dir_path / "tier_candidate_registry.square8_2p.json"

    candidate_id = "sq8_d4_candidate_test"
    plan_payload = _make_promotion_plan("D4", candidate_id)
    plan_path.write_text(json.dumps(plan_payload), encoding="utf-8")

    argv = [
        "--plan-path",
        os.fspath(plan_path),
        "--tier",
        "D4",
        "--board",
        "square8",
        "--num-players",
        "2",
        "--registry-path",
        os.fspath(registry_path),
    ]

    rc = apply_plan.main(argv)
    assert rc == 0

    # Registry should be written with a D4 entry for the candidate.
    assert registry_path.exists()
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    assert registry["board"] == "square8"
    assert registry["num_players"] == 2
    assert "D4" in registry["tiers"]
    candidates = registry["tiers"]["D4"]["candidates"]
    assert any(c.get("candidate_model_id") == candidate_id for c in candidates)

    # Summary and patch guide artefacts should be present.
    summary_path = run_dir_path / apply_plan.PROMOTION_SUMMARY_FILENAME
    patch_path = run_dir_path / apply_plan.PROMOTION_PATCH_GUIDE_FILENAME
    assert summary_path.exists()
    assert patch_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["tier"] == "D4"
    assert summary["board"] == "square8"
    assert summary["num_players"] == 2
    assert summary["candidate_model_id"] == candidate_id
    assert summary["registry_path"] == os.fspath(registry_path)

    patch_text = patch_path.read_text(encoding="utf-8")
    assert "ACTION REQUIRED (manual):" in patch_text
    assert "Change model_id" in patch_text


def test_apply_tier_promotion_plan_dry_run_does_not_write_registry(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """When --dry-run is set, the helper should not persist registry changes."""
    run_dir_path = Path(tmp_path)
    plan_path = run_dir_path / "promotion_plan.json"
    registry_path = run_dir_path / "tier_candidate_registry.square8_2p.json"

    candidate_id = "sq8_d2_candidate_test"
    plan_payload = _make_promotion_plan("D2", candidate_id)
    plan_payload["decision"] = "promote"
    plan_path.write_text(json.dumps(plan_payload), encoding="utf-8")

    argv = [
        "--plan-path",
        os.fspath(plan_path),
        "--tier",
        "D2",
        "--board",
        "square8",
        "--num-players",
        "2",
        "--registry-path",
        os.fspath(registry_path),
        "--dry-run",
    ]

    rc = apply_plan.main(argv)
    assert rc == 0

    # Registry file should not be created in dry-run mode.
    assert not registry_path.exists()

    # Summary and patch guide should still be written for operator visibility.
    summary_path = run_dir_path / apply_plan.PROMOTION_SUMMARY_FILENAME
    patch_path = run_dir_path / apply_plan.PROMOTION_PATCH_GUIDE_FILENAME
    assert summary_path.exists()
    assert patch_path.exists()

    patch_text = patch_path.read_text(encoding="utf-8")
    assert "NOTE: --dry-run was used; registry file was not written." in patch_text


def test_apply_tier_promotion_plan_rejects_on_ladder_mismatch(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Helper should fail fast when plan.current_model_id does not match ladder config."""
    run_dir_path = Path(tmp_path)
    plan_path = run_dir_path / "promotion_plan.json"
    registry_path = run_dir_path / "tier_candidate_registry.square8_2p.json"

    candidate_id = "sq8_d4_candidate_mismatch"
    plan_payload = _make_promotion_plan("D4", candidate_id)
    # Intentionally lie about the current_model_id to simulate a stale plan.
    plan_payload["current_model_id"] = "stale_prod_model"
    plan_path.write_text(json.dumps(plan_payload), encoding="utf-8")

    argv = [
        "--plan-path",
        os.fspath(plan_path),
        "--tier",
        "D4",
        "--board",
        "square8",
        "--num-players",
        "2",
        "--registry-path",
        os.fspath(registry_path),
    ]

    rc = apply_plan.main(argv)
    assert rc == 1

    # No registry or artefacts should be written when validation fails early.
    assert not registry_path.exists()
    summary_path = run_dir_path / apply_plan.PROMOTION_SUMMARY_FILENAME
    patch_path = run_dir_path / apply_plan.PROMOTION_PATCH_GUIDE_FILENAME
    assert not summary_path.exists()
    assert not patch_path.exists()


def test_apply_tier_promotion_plan_handles_reject_decision(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Reject decisions should record a rejected candidate without suggesting ladder edits."""
    run_dir_path = Path(tmp_path)
    plan_path = run_dir_path / "promotion_plan.json"
    registry_path = run_dir_path / "tier_candidate_registry.square8_2p.json"

    candidate_id = "sq8_d2_candidate_reject"
    plan_payload = _make_promotion_plan("D2", candidate_id)
    plan_payload["decision"] = "reject"
    plan_path.write_text(json.dumps(plan_payload), encoding="utf-8")

    argv = [
        "--plan-path",
        os.fspath(plan_path),
        "--tier",
        "D2",
        "--board",
        "square8",
        "--num-players",
        "2",
        "--registry-path",
        os.fspath(registry_path),
    ]

    rc = apply_plan.main(argv)
    assert rc == 0

    # Registry should be written with a rejected candidate entry.
    assert registry_path.exists()
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    candidates = registry["tiers"]["D2"]["candidates"]
    entry = next(c for c in candidates if c.get("candidate_model_id") == candidate_id)
    assert entry["status"] == "gated_reject"

    # Summary and patch guide should reflect a reject decision with no ladder edits.
    summary_path = run_dir_path / apply_plan.PROMOTION_SUMMARY_FILENAME
    patch_path = run_dir_path / apply_plan.PROMOTION_PATCH_GUIDE_FILENAME
    assert summary_path.exists()
    assert patch_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["decision"] == "reject"

    patch_text = patch_path.read_text(encoding="utf-8")
    assert "Decision is 'reject'; no ladder_config change is required." in patch_text
    assert "ACTION REQUIRED (manual):" not in patch_text


def test_apply_tier_promotion_plan_rejects_invalid_decision(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Helper should fail when promotion_plan.decision is neither promote nor reject."""
    run_dir_path = Path(tmp_path)
    plan_path = run_dir_path / "promotion_plan.json"
    registry_path = run_dir_path / "tier_candidate_registry.square8_2p.json"

    candidate_id = "sq8_d2_candidate_invalid"
    plan_payload = _make_promotion_plan("D2", candidate_id)
    plan_payload["decision"] = "maybe"  # Invalid decision value
    plan_path.write_text(json.dumps(plan_payload), encoding="utf-8")

    argv = [
        "--plan-path",
        os.fspath(plan_path),
        "--tier",
        "D2",
        "--board",
        "square8",
        "--num-players",
        "2",
        "--registry-path",
        os.fspath(registry_path),
    ]

    rc = apply_plan.main(argv)
    assert rc == 1

    # No registry or artefacts should be written on invalid decision.
    assert not registry_path.exists()
    summary_path = run_dir_path / apply_plan.PROMOTION_SUMMARY_FILENAME
    patch_path = run_dir_path / apply_plan.PROMOTION_PATCH_GUIDE_FILENAME
    assert not summary_path.exists()
    assert not patch_path.exists()


def test_apply_tier_promotion_plan_rejects_unsafe_promote_plan_by_default(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Promote decisions require candidate artefact enforcement by default."""
    run_dir_path = Path(tmp_path)
    plan_path = run_dir_path / "promotion_plan.json"
    registry_path = run_dir_path / "tier_candidate_registry.square8_2p.json"

    candidate_id = "sq8_d4_candidate_unsafe"
    plan_payload = _make_promotion_plan("D4", candidate_id)
    plan_payload["decision"] = "promote"
    plan_payload["reason"] = {"overall_pass": True}
    plan_path.write_text(json.dumps(plan_payload), encoding="utf-8")

    argv = [
        "--plan-path",
        os.fspath(plan_path),
        "--tier",
        "D4",
        "--board",
        "square8",
        "--num-players",
        "2",
        "--registry-path",
        os.fspath(registry_path),
    ]

    rc = apply_plan.main(argv)
    assert rc == 1
    assert not registry_path.exists()


def test_apply_tier_promotion_plan_allows_unsafe_plan_with_flag(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """--allow-unsafe-plan permits applying legacy promotion plans."""
    run_dir_path = Path(tmp_path)
    plan_path = run_dir_path / "promotion_plan.json"
    registry_path = run_dir_path / "tier_candidate_registry.square8_2p.json"

    candidate_id = "sq8_d4_candidate_unsafe_allowed"
    plan_payload = _make_promotion_plan("D4", candidate_id)
    plan_payload["decision"] = "promote"
    plan_payload["reason"] = {"overall_pass": True}
    plan_path.write_text(json.dumps(plan_payload), encoding="utf-8")

    argv = [
        "--plan-path",
        os.fspath(plan_path),
        "--tier",
        "D4",
        "--board",
        "square8",
        "--num-players",
        "2",
        "--registry-path",
        os.fspath(registry_path),
        "--allow-unsafe-plan",
    ]

    rc = apply_plan.main(argv)
    assert rc == 0
    assert registry_path.exists()
