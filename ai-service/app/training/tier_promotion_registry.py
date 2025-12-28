"""Tier Promotion Registry - Model tier tracking and promotion management.

This module manages the tier candidate registry for trained models across all
board/player configurations. It tracks models that have qualified for different
strength tiers (Beginner, Intermediate, Advanced, Expert, Master) and their
promotion status.

Key Components:
    - Registry files stored in ``config/tier_registries/``
    - Per-configuration registries (e.g., ``tier_candidate_registry.hex8_2p.json``)
    - Promotion history tracking for audit trails

Usage:
    # Load registry for a specific configuration
    registry = load_registry("hex8", 2)

    # Register a model as a tier candidate
    register_tier_candidate(
        board="hex8",
        num_players=2,
        tier="Advanced",
        model_path="models/hex8_2p_v42.pth",
        elo_rating=1850,
    )

    # Get models at a specific tier
    candidates = get_tier_candidates("hex8", 2, "Advanced")

Integration:
    - Called by auto_promotion_daemon.py after gauntlet evaluation
    - Used by SelfplayScheduler for opponent selection
    - Supports TIER_PROMOTION events in the feedback loop

See Also:
    - docs/training/TIER_PROMOTION_SYSTEM.md for full documentation
    - app.coordination.auto_promotion_daemon for automated promotion
"""

from __future__ import annotations

import json
import os
from typing import Any

from app.config import ladder_config
from app.models import AIType, BoardType
from app.utils.paths import AI_SERVICE_ROOT

# Default registry location for Square-8 2-player tier candidates.
DEFAULT_SQUARE8_2P_REGISTRY_PATH = os.fspath(
    AI_SERVICE_ROOT / "config" / "tier_candidate_registry.square8_2p.json"
)

# Registry directory for all board/player configurations.
REGISTRY_DIR = AI_SERVICE_ROOT / "config" / "tier_registries"


def _get_registry_path(board: str, num_players: int) -> str:
    """Return the registry path for a specific board/player configuration."""
    return os.fspath(REGISTRY_DIR / f"tier_candidate_registry.{board}_{num_players}p.json")


def _board_str_to_enum(board: str) -> BoardType:
    """Convert board string to BoardType enum."""
    mapping = {
        "square8": BoardType.SQUARE8,
        "square19": BoardType.SQUARE19,
        "hexagonal": BoardType.HEXAGONAL,
        "hex": BoardType.HEXAGONAL,
    }
    return mapping.get(board.lower(), BoardType.SQUARE8)


def _default_square8_two_player_registry() -> dict[str, Any]:
    """Return an empty candidate registry for square8 2-player tiers."""
    return {
        "board": "square8",
        "num_players": 2,
        "tiers": {},
    }


def load_square8_two_player_registry(
    path: str | None = None,
) -> dict[str, Any]:
    """Load the Square-8 2-player candidate registry.

    When *path* is None, the default registry path under ``ai-service/config``
    is used. If the file does not exist, an empty default structure is
    returned instead of raising.
    """
    registry_path = path or DEFAULT_SQUARE8_2P_REGISTRY_PATH
    if not os.path.exists(registry_path):
        return _default_square8_two_player_registry()

    with open(registry_path, encoding="utf-8") as f:
        return json.load(f)


def save_square8_two_player_registry(
    registry: dict[str, Any],
    path: str | None = None,
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


def get_current_ladder_model_for_tier(tier: str) -> dict[str, Any]:
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
    registry: dict[str, Any],
    tier: str,
    candidate_id: str,
    run_dir: str,
    promotion_plan: dict[str, Any],
) -> dict[str, Any]:
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
    candidate_entry: dict[str, Any] | None = None

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
    promotion_plan: dict[str, Any],
    path: str | None = None,
) -> dict[str, Any]:
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


# =============================================================================
# Generic Multi-Config Registry API
# =============================================================================


def _default_config_registry(board: str, num_players: int) -> dict[str, Any]:
    """Return an empty candidate registry for a specific board/player config."""
    return {
        "board": board,
        "num_players": num_players,
        "tiers": {},
    }


def load_config_registry(
    board: str,
    num_players: int,
    path: str | None = None,
) -> dict[str, Any]:
    """Load the candidate registry for a specific board/player configuration.

    When *path* is None, the default registry path under ``ai-service/config/tier_registries``
    is used. If the file does not exist, an empty default structure is returned.

    Args:
        board: Board identifier (e.g., "square8", "square19", "hexagonal").
        num_players: Number of players (2, 3, or 4).
        path: Optional override path for the registry file.

    Returns:
        The registry dictionary for this configuration.
    """
    registry_path = path or _get_registry_path(board, num_players)
    if not os.path.exists(registry_path):
        return _default_config_registry(board, num_players)

    with open(registry_path, encoding="utf-8") as f:
        return json.load(f)


def save_config_registry(
    registry: dict[str, Any],
    board: str,
    num_players: int,
    path: str | None = None,
) -> None:
    """Persist the candidate registry for a specific board/player configuration.

    The JSON is written with indentation and sorted keys for stable,
    human-auditable diffs.

    Args:
        registry: The registry dictionary to save.
        board: Board identifier (e.g., "square8", "square19", "hexagonal").
        num_players: Number of players (2, 3, or 4).
        path: Optional override path for the registry file.
    """
    registry_path = path or _get_registry_path(board, num_players)
    registry_dir = os.path.dirname(registry_path)
    if registry_dir:
        os.makedirs(registry_dir, exist_ok=True)

    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, sort_keys=True)


def get_current_ladder_model_for_config_tier(
    tier: str,
    board: str,
    num_players: int,
) -> dict[str, Any]:
    """Return a small summary of the live ladder assignment for a tier.

    This is a generalized version of :func:`get_current_ladder_model_for_tier`
    that works for any board/player configuration.

    Args:
        tier: Tier name (e.g., "D4", "D6").
        board: Board identifier (e.g., "square8", "square19", "hexagonal").
        num_players: Number of players (2, 3, or 4).

    Returns:
        Dictionary with tier configuration details.
    """
    tier_name = str(tier).upper()
    if not tier_name.startswith("D") or not tier_name[1:].isdigit():
        raise ValueError(f"Unsupported tier name {tier!r}; expected like 'D4'.")

    difficulty = int(tier_name[1:])
    board_type = _board_str_to_enum(board)

    cfg = ladder_config.get_ladder_tier_config(
        difficulty=difficulty,
        board_type=board_type,
        num_players=num_players,
    )

    return {
        "tier": tier_name,
        "difficulty": difficulty,
        "board": board,
        "board_type": board_type.value,
        "num_players": num_players,
        "model_id": cfg.model_id,
        "heuristic_profile_id": cfg.heuristic_profile_id,
        "ai_type": (
            cfg.ai_type.value
            if isinstance(cfg.ai_type, AIType)
            else str(cfg.ai_type)
        ),
        "ladder_source": "app.config.ladder_config",
    }


def record_config_promotion_plan(
    registry: dict[str, Any],
    tier: str,
    candidate_id: str,
    run_dir: str,
    promotion_plan: dict[str, Any],
    board: str,
    num_players: int,
) -> dict[str, Any]:
    """Add or update a candidate entry in a config-specific registry.

    This is a generalized version of :func:`record_promotion_plan` that works
    for any board/player configuration.

    Args:
        registry: In-memory registry structure to be updated.
        tier: Difficulty tier name (e.g., "D4").
        candidate_id: Candidate identifier.
        run_dir: Path to the training/gating run directory.
        promotion_plan: Parsed promotion_plan.json payload.
        board: Board identifier (e.g., "square8").
        num_players: Number of players (2, 3, or 4).

    Returns:
        The candidate entry dictionary that was added or updated.
    """
    tier_name = str(tier).upper()
    status = _status_from_decision(
        str(promotion_plan.get("decision", "reject"))
    )

    registry.setdefault("board", board)
    registry.setdefault("num_players", num_players)

    tiers = registry.setdefault("tiers", {})
    tier_block = tiers.setdefault(
        tier_name,
        {
            "current": get_current_ladder_model_for_config_tier(tier_name, board, num_players),
            "candidates": [],
        },
    )

    # Ensure the "current" block is kept in sync with the live ladder.
    tier_block["current"] = get_current_ladder_model_for_config_tier(tier_name, board, num_players)

    candidates = tier_block.setdefault("candidates", [])
    candidate_entry: dict[str, Any] | None = None

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


def update_config_registry_for_run(
    tier: str,
    candidate_id: str,
    run_dir: str,
    promotion_plan: dict[str, Any],
    board: str,
    num_players: int,
    path: str | None = None,
) -> dict[str, Any]:
    """Load, update, and persist a config-specific tier candidate registry.

    This is a generalized version of :func:`update_square8_two_player_registry_for_run`
    that works for any board/player configuration.

    Args:
        tier: Difficulty tier name (e.g., "D4").
        candidate_id: Candidate identifier.
        run_dir: Path to the training/gating run directory.
        promotion_plan: Parsed promotion_plan.json payload.
        board: Board identifier (e.g., "square8", "square19", "hexagonal").
        num_players: Number of players (2, 3, or 4).
        path: Optional override path for the registry file.

    Returns:
        The candidate entry dictionary that was added or updated.
    """
    registry = load_config_registry(board, num_players, path=path)
    entry = record_config_promotion_plan(
        registry=registry,
        tier=tier,
        candidate_id=candidate_id,
        run_dir=run_dir,
        promotion_plan=promotion_plan,
        board=board,
        num_players=num_players,
    )
    save_config_registry(registry, board, num_players, path=path)
    return entry


def load_all_config_registries() -> dict[str, dict[str, Any]]:
    """Load all config registries that exist on disk.

    Returns:
        Dictionary mapping config key (e.g., "square8_2p") to registry contents.
    """
    from app.training.crossboard_strength import ALL_BOARD_CONFIGS, config_key

    registries = {}
    for board, num_players in ALL_BOARD_CONFIGS:
        key = config_key(board, num_players)
        registry = load_config_registry(board, num_players)
        if registry.get("tiers"):  # Only include non-empty registries
            registries[key] = registry
    return registries


def get_all_promoted_candidates() -> list[dict[str, Any]]:
    """Get all candidates that have been promoted across all configurations.

    Returns:
        List of candidate entries with status "gated_promote".
    """
    registries = load_all_config_registries()
    promoted = []

    for config_key, registry in registries.items():
        for tier_name, tier_data in registry.get("tiers", {}).items():
            for candidate in tier_data.get("candidates", []):
                if candidate.get("status") == "gated_promote":
                    # Add config context
                    candidate_with_context = {
                        **candidate,
                        "config_key": config_key,
                    }
                    promoted.append(candidate_with_context)

    return promoted
