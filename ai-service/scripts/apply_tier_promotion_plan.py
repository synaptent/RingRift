#!/usr/bin/env python
"""Apply a tier promotion plan to the Square-8 2-player candidate registry.

This helper reads a ``promotion_plan.json`` produced by
``scripts.run_full_tier_gating``, validates it against the live ladder
configuration, updates the Square-8 2-player candidate registry, and
emits a small ``promotion_summary.json`` plus a human-readable
``promotion_patch_guide.txt`` alongside the plan.

The script is intentionally conservative: when the promotion plan's view
of the current production model id does not match the ladder
configuration, it exits with a non-zero status instead of applying
changes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.config.ladder_config import (
    get_ladder_tier_config,
)
from app.models import BoardType
from app.training.tier_promotion_registry import (
    DEFAULT_SQUARE8_2P_REGISTRY_PATH,
    load_square8_two_player_registry,
    record_promotion_plan,
    save_square8_two_player_registry,
)

PROMOTION_SUMMARY_FILENAME = "promotion_summary.json"
PROMOTION_PATCH_GUIDE_FILENAME = "promotion_patch_guide.txt"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the promotion-plan applier."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate a tier promotion plan against the live ladder and "
            "update the Square-8 2-player candidate registry."
        ),
    )
    parser.add_argument(
        "--plan-path",
        required=True,
        help="Path to promotion_plan.json produced by run_full_tier_gating.",
    )
    parser.add_argument(
        "--board",
        default="square8",
        help="Board identifier (currently only 'square8' is supported).",
    )
    parser.add_argument(
        "--num-players",
        type=int,
        default=2,
        help="Number of players (currently only 2 is supported).",
    )
    parser.add_argument(
        "--registry-path",
        default=DEFAULT_SQUARE8_2P_REGISTRY_PATH,
        help=(
            "Path to the Square-8 2-player candidate registry JSON. When "
            "omitted, the default ai-service/config path is used."
        ),
    )
    parser.add_argument(
        "--tier",
        default=None,
        help=("Optional tier name (e.g. D4). When supplied, must match the " "tier recorded in the promotion plan."),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=("Do not write changes to the registry; still validate the plan " "and emit summary/patch guide."),
    )
    parser.add_argument(
        "--allow-unsafe-plan",
        action="store_true",
        help=(
            "Allow applying a promotion plan that was generated without "
            "candidate artefact enforcement (use_candidate_artifact=false). "
            "This is intended only for legacy plans or manual experiments; "
            "canonical promotions should be gated via run_full_tier_gating "
            "(non-demo) or run_tier_gate --use-candidate-artifact."
        ),
    )
    return parser.parse_args(argv)


def _load_promotion_plan(path: str) -> dict[str, Any]:
    """Load and parse a promotion_plan.json file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _normalise_board_label(board: str) -> str:
    """Return a normalised, case-insensitive board label."""
    return board.strip().lower()


def _board_to_enum(board_label: str) -> BoardType | None:
    """Map a textual board label to BoardType, restricting to square8."""
    norm = _normalise_board_label(board_label)
    if norm in {"square8", "sq8"}:
        return BoardType.SQUARE8
    return None


def _validate_plan_against_cli(
    plan: dict[str, Any],
    board: str,
    num_players: int,
    tier_override: str | None,
) -> tuple[str, str, int]:
    """Validate the plan against CLI inputs.

    Returns a triple of (tier, board_label, num_players). When validation
    fails, all three fields are returned as falsey values.
    """
    plan_tier = str(plan.get("tier") or "").upper()
    if tier_override is not None:
        tier_cli = str(tier_override).upper()
        if plan_tier and plan_tier != tier_cli:
            print(
                "Error: tier mismatch between CLI and promotion plan: " f"cli={tier_cli!r}, plan={plan_tier!r}",
            )
            return "", "", 0
        tier = tier_cli
    else:
        if not plan_tier:
            print("Error: promotion plan is missing required 'tier' field.")
            return "", "", 0
        tier = plan_tier

    board_cli = _normalise_board_label(board)
    plan_board_raw = plan.get("board_type") or plan.get("board")
    if plan_board_raw is not None:
        plan_board = _normalise_board_label(str(plan_board_raw))
        if plan_board != board_cli:
            print(
                "Error: board mismatch between CLI and promotion plan: " f"cli={board_cli!r}, plan={plan_board!r}",
            )
            return "", "", 0
        board_label = plan_board
    else:
        board_label = board_cli

    plan_num_players_raw = plan.get("num_players")
    if plan_num_players_raw is not None:
        plan_num_players = int(plan_num_players_raw)
        if plan_num_players != num_players:
            print(
                "Error: num_players mismatch between CLI and promotion plan: "
                f"cli={num_players}, plan={plan_num_players}",
            )
            return "", "", 0
        resolved_num_players = plan_num_players
    else:
        resolved_num_players = num_players

    return tier, board_label, resolved_num_players


def _validate_ladder_view(
    tier: str,
    board_type: BoardType,
    num_players: int,
    plan: dict[str, Any],
) -> dict[str, Any] | None:
    """Validate that the plan's current_model_id matches the ladder config.

    Returns a small dict describing the ladder view when validation
    succeeds, or None on failure.
    """
    difficulty_str = tier[1:]
    if not difficulty_str.isdigit():
        print(f"Error: unsupported tier name {tier!r}; expected like 'D4'.")
        return None

    difficulty = int(difficulty_str)
    current_from_plan = plan.get("current_model_id")
    candidate_model_id = plan.get("candidate_model_id")

    if candidate_model_id is None:
        print("Error: promotion plan is missing 'candidate_model_id'.")
        return None

    try:
        ladder_cfg = get_ladder_tier_config(
            difficulty=difficulty,
            board_type=board_type,
            num_players=num_players,
        )
    except Exception as exc:  # pragma: no cover - defensive
        print(
            "Error: could not resolve ladder tier config for "
            f"tier={tier!r}, board={board_type.value!r}, "
            f"num_players={num_players}: {exc}",
        )
        return None

    ladder_model_id = ladder_cfg.model_id
    if ladder_model_id != current_from_plan:
        print(
            "Error: promotion plan current_model_id does not match live "
            "ladder configuration: "
            f"plan={current_from_plan!r}, ladder={ladder_model_id!r}",
        )
        return None

    return {
        "difficulty": difficulty,
        "ladder_model_id": ladder_model_id,
        "candidate_model_id": candidate_model_id,
    }


def _write_promotion_summary(
    out_dir: str,
    tier: str,
    board_label: str,
    num_players: int,
    ladder_info: dict[str, Any],
    decision: str,
    registry_path: str,
    status_after_update: str,
) -> str:
    """Write promotion_summary.json and return its path."""
    summary = {
        "tier": tier,
        "board": board_label,
        "num_players": num_players,
        "current_model_id": ladder_info["ladder_model_id"],
        "candidate_model_id": ladder_info["candidate_model_id"],
        "decision": decision,
        "ladder_config_file": "ai-service/app/config/ladder_config.py",
        "registry_path": registry_path,
        "status_after_registry_update": status_after_update,
    }
    out_path = os.path.join(out_dir, PROMOTION_SUMMARY_FILENAME)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    return out_path


def _write_patch_guide(
    out_dir: str,
    tier: str,
    board_label: str,
    num_players: int,
    ladder_info: dict[str, Any],
    decision: str,
    registry_path: str,
    dry_run: bool,
) -> str:
    """Write a human-readable promotion_patch_guide.txt and return its path."""
    current_model_id = ladder_info["ladder_model_id"]
    candidate_model_id = ladder_info["candidate_model_id"]

    lines = []
    lines.append(f"Tier: {tier} ({board_label}, {num_players} players)")
    lines.append("")
    lines.append("Current ladder config:")
    lines.append("  File: ai-service/app/config/ladder_config.py")
    lines.append("  Function: _build_default_square8_two_player_configs")
    lines.append(f"  Current model_id: {current_model_id}")
    lines.append("")
    lines.append("Candidate:")
    lines.append(f"  candidate_model_id: {candidate_model_id}")
    lines.append(f"  decision from promotion_plan: {decision}")
    lines.append("")

    if decision == "promote":
        lines.append("ACTION REQUIRED (manual):")
        lines.append("  - Open ai-service/app/config/ladder_config.py")
        lines.append(
            "  - In _build_default_square8_two_player_configs(), locate the " f"LadderTierConfig for tier {tier}."
        )
        lines.append("  - Change model_id from " f'"{current_model_id}" to "{candidate_model_id}".')
        lines.append(
            "  - Commit this change with a message referencing the " "promotion plan and promotion_summary.json."
        )
    else:
        lines.append("ACTION:")
        lines.append("  - Decision is 'reject'; no ladder_config change is required.")
        lines.append("  - You may still keep the candidate in the registry for " "historical tracking.")

    lines.append("")
    lines.append("Registry:")
    lines.append(f"  - Updated registry at: {registry_path}")
    if dry_run:
        lines.append("  - NOTE: --dry-run was used; registry file was not written.")

    out_path = os.path.join(out_dir, PROMOTION_PATCH_GUIDE_FILENAME)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return out_path


def main(argv: list[str] | None = None) -> int:
    """Entry point for the tier promotion-plan application helper."""
    args = _parse_args(argv)

    plan_path = os.path.abspath(args.plan_path)
    if not os.path.exists(plan_path):
        print(f"Error: promotion plan path does not exist: {plan_path!r}")
        return 1

    try:
        plan = _load_promotion_plan(plan_path)
    except Exception as exc:
        print(f"Error: failed to load promotion plan JSON: {exc}")
        return 1

    tier, board_label, resolved_num_players = _validate_plan_against_cli(
        plan=plan,
        board=args.board,
        num_players=args.num_players,
        tier_override=args.tier,
    )
    if not tier:
        return 1

    board_enum = _board_to_enum(board_label)
    if board_enum is None or resolved_num_players != 2:
        print(
            "Error: only Square-8 2-player tiers are currently supported " "by this helper.",
        )
        return 1

    decision = str(plan.get("decision") or "").lower()
    if decision not in {"promote", "reject"}:
        print(
            "Error: promotion plan 'decision' must be 'promote' or " f"'reject', got {plan.get('decision')!r}.",
        )
        return 1
    if decision == "promote" and not args.allow_unsafe_plan:
        reason = plan.get("reason")
        reason_dict: dict[str, Any] = reason if isinstance(reason, dict) else {}
        used = bool(reason_dict.get("use_candidate_artifact", False))
        present = bool(reason_dict.get("candidate_artifact_present", False))
        loaded = reason_dict.get("candidate_artifact_loaded")

        if not used:
            print(
                "Error: refusing to apply a promotion plan that was generated "
                "without candidate artefact enforcement "
                "(reason.use_candidate_artifact=false).\n"
                "Regenerate the plan via:\n"
                "  - scripts/run_full_tier_gating.py (non-demo), or\n"
                "  - scripts/run_tier_gate.py --use-candidate-artifact\n"
                "To override (unsafe), re-run with --allow-unsafe-plan.",
            )
            return 1
        if not present:
            print(
                "Error: refusing to apply a promotion plan with "
                "reason.candidate_artifact_present=false.\n"
                "This usually indicates the candidate id was label-only and "
                "no promotable artefact was found.\n"
                "To override (unsafe), re-run with --allow-unsafe-plan.",
            )
            return 1
        if loaded is False:
            print(
                "Error: refusing to apply a promotion plan with "
                "reason.candidate_artifact_loaded=false.\n"
                "This indicates the candidate artefact exists but failed to "
                "load during gating.\n"
                "To override (unsafe), re-run with --allow-unsafe-plan.",
            )
            return 1

    ladder_info = _validate_ladder_view(
        tier=tier,
        board_type=board_enum,
        num_players=resolved_num_players,
        plan=plan,
    )
    if ladder_info is None:
        # Validation failure already reported.
        return 1

    registry_path = args.registry_path or DEFAULT_SQUARE8_2P_REGISTRY_PATH

    # Registry update (unless --dry-run is set).
    registry = load_square8_two_player_registry(registry_path)
    candidate_entry = record_promotion_plan(
        registry=registry,
        tier=tier,
        candidate_id=str(ladder_info["candidate_model_id"]),
        run_dir=os.path.dirname(plan_path),
        promotion_plan=plan,
    )
    if not args.dry_run:
        save_square8_two_player_registry(registry, registry_path)

    status_after_update = str(candidate_entry.get("status") or "")

    out_dir = os.path.dirname(plan_path)
    summary_path = _write_promotion_summary(
        out_dir=out_dir,
        tier=tier,
        board_label=board_label,
        num_players=resolved_num_players,
        ladder_info=ladder_info,
        decision=decision,
        registry_path=registry_path,
        status_after_update=status_after_update,
    )
    patch_guide_path = _write_patch_guide(
        out_dir=out_dir,
        tier=tier,
        board_label=board_label,
        num_players=resolved_num_players,
        ladder_info=ladder_info,
        decision=decision,
        registry_path=registry_path,
        dry_run=args.dry_run,
    )

    print(
        "Applied promotion plan for "
        f"tier={tier}, board={board_label}, "
        f"num_players={resolved_num_players}. Decision={decision}.",
    )
    print(f"  Registry path: {registry_path!r}")
    if args.dry_run:
        print("  Registry update: DRY-RUN (no file written).")
    else:
        print("  Registry update: written.")
    print(f"  Promotion summary: {summary_path!r}")
    print(f"  Patch guide: {patch_guide_path!r}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
