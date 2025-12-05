"""Tests for the Square-8 2-player tier promotion registry helpers."""

from __future__ import annotations

import json
import os
from typing import Any, Dict

import pytest

from app.config.ladder_config import get_ladder_tier_config  # type: ignore[import]
from app.models import BoardType, AIType  # type: ignore[import]
from app.training.tier_promotion_registry import (  # type: ignore[import]
    _default_square8_two_player_registry,
    get_current_ladder_model_for_tier,
    load_square8_two_player_registry,
    record_promotion_plan,
    save_square8_two_player_registry,
    update_square8_two_player_registry_for_run,
)


def test_default_square8_two_player_registry_shape() -> None:
    """_default_square8_two_player_registry returns an empty but well-formed registry."""
    registry = _default_square8_two_player_registry()

    assert registry["board"] == "square8"
    assert registry["num_players"] == 2
    assert isinstance(registry["tiers"], dict)
    assert registry["tiers"] == {}


def test_load_and_save_square8_two_player_registry_roundtrip(tmp_path: pytest.TempPathFactory) -> None:
    """load_square8_two_player_registry and save_square8_two_player_registry round-trip JSON."""
    path = os.fspath(tmp_path.mktemp("registry") / "tier_registry.square8_2p.json")

    # When the file does not exist yet, load should return the default structure.
    loaded = load_square8_two_player_registry(path=path)
    assert loaded["board"] == "square8"
    assert loaded["num_players"] == 2
    assert loaded["tiers"] == {}

    # Mutate and save, then reload and ensure the change persists.
    loaded["tiers"]["D4"] = {"current": {"tier": "D4"}, "candidates": []}
    save_square8_two_player_registry(loaded, path=path)

    with open(path, "r", encoding="utf-8") as f:
        on_disk = json.load(f)

    assert on_disk["board"] == "square8"
    assert on_disk["num_players"] == 2
    assert "D4" in on_disk["tiers"]

    reloaded = load_square8_two_player_registry(path=path)
    assert reloaded == on_disk


@pytest.mark.parametrize("tier_name,difficulty", [("D2", 2), ("D4", 4), ("D6", 6), ("D8", 8)])
def test_get_current_ladder_model_for_tier_matches_ladder_config(
    tier_name: str, difficulty: int
) -> None:
    """get_current_ladder_model_for_tier mirrors ladder_config for square8 2p tiers."""
    summary = get_current_ladder_model_for_tier(tier_name)

    cfg = get_ladder_tier_config(
        difficulty=difficulty,
        board_type=BoardType.SQUARE8,
        num_players=2,
    )

    assert summary["tier"] == tier_name
    assert summary["difficulty"] == difficulty
    assert summary["board"] == "square8"
    assert summary["board_type"] == BoardType.SQUARE8.value
    assert summary["num_players"] == 2

    # Model and heuristic profile ids must match the ladder config.
    assert summary["model_id"] == cfg.model_id
    assert summary["heuristic_profile_id"] == cfg.heuristic_profile_id

    # ai_type is serialised via its Enum value.
    assert summary["ai_type"] == (cfg.ai_type.value if isinstance(cfg.ai_type, AIType) else str(cfg.ai_type))


def test_record_promotion_plan_creates_and_updates_candidate_entries(tmp_path: pytest.TempPathFactory) -> None:
    """record_promotion_plan should create or update candidate entries per promotion_plan."""
    registry: Dict[str, Any] = _default_square8_two_player_registry()

    run_dir = os.fspath(tmp_path.mktemp("run"))
    tier = "D4"
    candidate_id = "sq8_d4_candidate_v1"

    promotion_plan: Dict[str, Any] = {
        "tier": tier,
        "board_type": "square8",
        "num_players": 2,
        "candidate_model_id": candidate_id,
        "candidate_heuristic_profile_id": "sq8_heuristic_v2",
        "decision": "promote",
        "overall_pass": True,
    }

    entry = record_promotion_plan(
        registry=registry,
        tier=tier,
        candidate_id=candidate_id,
        run_dir=run_dir,
        promotion_plan=promotion_plan,
    )

    # Registry-level invariants
    assert registry["board"] == "square8"
    assert registry["num_players"] == 2
    assert "D4" in registry["tiers"]

    tier_block = registry["tiers"]["D4"]
    assert "current" in tier_block
    assert "candidates" in tier_block

    # Candidate entry fields
    assert entry["candidate_id"] == candidate_id
    assert entry["candidate_model_id"] == candidate_id
    assert entry["tier"] == "D4"
    assert entry["board"] == "square8"
    assert entry["num_players"] == 2
    assert entry["source_run_dir"] == os.fspath(run_dir)

    # Default artefact filenames
    assert entry["training_report"] == "training_report.json"
    assert entry["gate_report"] == "gate_report.json"
    assert entry["promotion_plan"] == "promotion_plan.json"

    # Model/profile ids from the promotion_plan
    assert entry["model_id"] == candidate_id
    assert entry["heuristic_profile_id"] == "sq8_heuristic_v2"

    # Status is derived from the decision field.
    assert entry["status"] == "gated_promote"

    # Calling record_promotion_plan again for the same candidate_id should update
    # the existing entry rather than appending a duplicate.
    promotion_plan_reject = dict(promotion_plan)
    promotion_plan_reject["decision"] = "reject"

    updated_entry = record_promotion_plan(
        registry=registry,
        tier=tier,
        candidate_id=candidate_id,
        run_dir=run_dir,
        promotion_plan=promotion_plan_reject,
    )

    assert updated_entry is entry
    assert updated_entry["status"] == "gated_reject"

    # Only a single candidate entry should exist for this tier.
    assert len(tier_block["candidates"]) == 1


def test_update_square8_two_player_registry_for_run_persists_to_disk(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """update_square8_two_player_registry_for_run should write an updated registry file."""
    registry_dir = tmp_path.mktemp("registry_run")
    registry_path = os.fspath(registry_dir / "tier_candidate_registry.square8_2p.json")

    tier = "D2"
    candidate_id = "sq8_d2_candidate_v1"
    run_dir = os.fspath(tmp_path.mktemp("run_dir"))

    promotion_plan: Dict[str, Any] = {
      "tier": tier,
      "board_type": "square8",
      "num_players": 2,
      "candidate_model_id": candidate_id,
      "decision": "promote",
      "overall_pass": True,
    }

    entry = update_square8_two_player_registry_for_run(
        tier=tier,
        candidate_id=candidate_id,
        run_dir=run_dir,
        promotion_plan=promotion_plan,
        path=registry_path,
    )

    # Registry file should now exist on disk.
    assert os.path.exists(registry_path)

    with open(registry_path, "r", encoding="utf-8") as f:
        persisted = json.load(f)

    assert persisted["board"] == "square8"
    assert persisted["num_players"] == 2
    assert "D2" in persisted["tiers"]
    candidates = persisted["tiers"]["D2"]["candidates"]
    assert len(candidates) == 1

    # The entry returned by the helper should match the on-disk candidate.
    persisted_entry = candidates[0]
    assert persisted_entry["candidate_id"] == entry["candidate_id"] == candidate_id
    assert persisted_entry["status"] == "gated_promote"

