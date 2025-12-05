from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from app.config import ladder_config
from app.models import BoardType, AIType

# Default registry location for Square-8 2-player tier candidates.
# This path is anchored at the ai-service repository root so that it
# remains stable regardless of the current working directory.
_AI_SERVICE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SQUARE8_2P_REGISTRY_PATH = os.fspath(
    _AI_SERVICE_ROOT / "config" / "tier_candidate_registry.square8_2p.json"
)


def _default_square8_two_player_registry() -> Dict[str, Any]:
    """Return an empty candidate registry for square8 2-player tiers."""
    return {
        "board": "square8",
        "num_players": 2,
        "tiers": {},
    }


def load_square8_two_player_registry(
    path: Optional[str] = None,
) -> Dict[str, Any]:
    """Load the Square-8 2-player candidate registry.

    When *path* is None, the default registry path under ``ai-service/config``
    is used. If the file does not exist, an empty default structure is
    returned instead of raising.
    """
    registry_path = path or DEFAULT_SQUARE8_2P_REGISTRY_PATH
    if not os.path.exists(registry_path):
        return _default_square8_two_player_registry()

    with open(registry_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_square8_two_player_registry(
    registry: Dict[str, Any],
    path: Optional[str] = None,
) -> None:
    """Persist the Square-8 2-player candidate registry to disk.

    The JSON is written with indentation and sorted keys for stable,
    human-auditable diffs.
    """
    registry_path = path or DEFAULT_SQUARE8_2P_REGISTRY_PATH
    registry_dir = os.path.dirname(registry_path)
    if registry_dir:
        os.makedirs(registry_dir, exist_ok=True)

    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, sort_keys=True)


def _status_from_decision(decision: str) -> str:
    """Map a promotion-plan decision string to a registry status value."""
    if decision.lower() == "promote":
        return "gated_promote"
    return "gated_reject"


def get_current_ladder_model_for_tier(tier: str) -> Dict[str, Any]:
    """Return a small summary of the live ladder assignment for *tier*.

    This helper is square8 2-player specific and uses the canonical
    ladder configuration as its single source of truth.
    """
    tier_name = str(tier).upper()
    if not tier_name.startswith("D") or not tier_name[1:].isdigit():
        raise ValueError(
            f"Unsupported tier name {tier!r}; expected like 'D4'."
        )

    difficulty = int(tier_name[1:])
    board_type = BoardType.SQUARE8
    num_players = 2

    cfg = ladder_config.get_ladder_tier_config(
        difficulty=difficulty,
        board_type=board_type,
        num_players=num_players,
    )

    return {
        "tier": tier_name,
        "difficulty": difficulty,
        "board": "square8",
        "board_type": board_type.value,
        "num_players": num_players,
        "model_id": cfg.model_id,
        "heuristic_profile_id": cfg.heuristic_profile_id,
        "ai_type": (
            cfg.ai_type.value
            if isinstance(cfg.ai_type, AIType)
            else str(cfg.ai_type)
        ),
        "ladder_source": (
            "app.config.ladder_config."
            "_build_default_square8_two_player_configs"
        ),
    }


def record_promotion_plan(
    registry: Dict[str, Any],
    tier: str,
    candidate_id: str,
    run_dir: str,
    promotion_plan: Dict[str, Any],
) -> Dict[str, Any]:
    """Add or update a candidate entry in the registry.

    Args:
        registry: In-memory registry structure to be updated.
        tier: Difficulty tier name (e.g. ``"D4"``).
        candidate_id: Candidate identifier. For square8 2p difficulty tiers
            this typically matches the ``candidate_model_id`` in the
            promotion plan.
        run_dir: Path to the training / gating run directory that produced
            the promotion plan.
        promotion_plan: Parsed ``promotion_plan.json`` payload.

    Returns:
        The candidate entry dictionary that was added or updated.
    """
    tier_name = str(tier).upper()
    status = _status_from_decision(
        str(promotion_plan.get("decision", "reject"))
    )
    board = registry.get("board") or "square8"
    num_players = int(registry.get("num_players") or 2)

    registry.setdefault("board", board)
    registry.setdefault("num_players", num_players)

    tiers = registry.setdefault("tiers", {})
    tier_block = tiers.setdefault(
        tier_name,
        {
            "current": get_current_ladder_model_for_tier(tier_name),
            "candidates": [],
        },
    )

    # Ensure the "current" block is kept in sync with the live ladder.
    tier_block["current"] = get_current_ladder_model_for_tier(tier_name)

    candidates = tier_block.setdefault("candidates", [])
    candidate_entry: Optional[Dict[str, Any]] = None

    for entry in candidates:
        if entry.get("candidate_id") == candidate_id or entry.get(
            "candidate_model_id"
        ) == candidate_id:
            candidate_entry = entry
            break

    if candidate_entry is None:
        candidate_entry = {
            "candidate_id": candidate_id,
            "candidate_model_id": candidate_id,
        }
        candidates.append(candidate_entry)

    candidate_entry["tier"] = tier_name
    candidate_entry["board"] = board
    candidate_entry["num_players"] = num_players
    candidate_entry["source_run_dir"] = os.fspath(run_dir)
    candidate_entry.setdefault("training_report", "training_report.json")
    candidate_entry.setdefault("gate_report", "gate_report.json")
    candidate_entry.setdefault("promotion_plan", "promotion_plan.json")

    # Track the model / profile identifiers when available.
    model_id = promotion_plan.get("candidate_model_id") or promotion_plan.get(
        "candidate_id"
    )
    if model_id is not None:
        candidate_entry["model_id"] = model_id
    heuristic_profile_id = promotion_plan.get("candidate_heuristic_profile_id")
    if heuristic_profile_id is not None:
        candidate_entry["heuristic_profile_id"] = heuristic_profile_id

    candidate_entry["status"] = status
    return candidate_entry


def update_square8_two_player_registry_for_run(
    tier: str,
    candidate_id: str,
    run_dir: str,
    promotion_plan: Dict[str, Any],
    path: Optional[str] = None,
) -> Dict[str, Any]:
    """Load, update, and persist the Square-8 2p tier candidate registry.

    This is a convenience helper intended for orchestration scripts
    such as ``run_full_tier_gating.py``. It:

    - Loads the existing registry (or creates a default if missing),
    - Records or updates the candidate entry for the given tier using
      :func:`record_promotion_plan`, and
    - Writes the updated registry back to disk.

    Returns:
        The candidate entry dictionary that was added or updated.
    """
    registry = load_square8_two_player_registry(path=path)
    entry = record_promotion_plan(
        registry=registry,
        tier=tier,
        candidate_id=candidate_id,
        run_dir=run_dir,
        promotion_plan=promotion_plan,
    )
    save_square8_two_player_registry(registry, path=path)
    return entry
