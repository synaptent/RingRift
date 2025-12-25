#!/usr/bin/env python
"""Tier gating CLI for RingRift difficulty tiers and heuristic eval pools.

This script has two related modes:

1) Difficulty-tier gating for the canonical ladder (e.g. D2, D4, D6, D8)
   using :mod:`app.training.tier_eval_runner` and
   :mod:`app.config.ladder_config`.

2) Backwards-compatible heuristic tier gating on eval pools using
   :class:`app.training.tier_eval_config.HeuristicTierSpec` and
   :mod:`app.training.eval_pools`.

The mode is selected by whether ``--tier`` (difficulty tier) or
``--tier-id`` (heuristic eval-pool tier id) is supplied.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.config.ladder_config import (
    get_ladder_tier_config,
)
from app.models import AIType
from app.training.eval_pools import (
    run_heuristic_tier_eval,
)
from app.training.tier_eval_config import (
    HEURISTIC_TIER_SPECS,
    TIER_EVAL_CONFIGS,
    HeuristicTierSpec,
    get_tier_config,
)
from app.training.tier_eval_runner import (
    TierEvaluationResult,
    run_tier_evaluation,
)


def _get_heuristic_tier_spec(tier_id: str) -> HeuristicTierSpec:
    """Lookup helper for heuristic eval-pool tiers."""
    for spec in HEURISTIC_TIER_SPECS:
        if spec.id == tier_id:
            return spec
    available = ", ".join(sorted(t.id for t in HEURISTIC_TIER_SPECS))
    raise SystemExit(f"Unknown heuristic tier id {tier_id!r}. " f"Available ids: {available}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a difficulty-tier gate (D2-D10) or a heuristic " "eval-pool tier gate and emit a JSON summary."
        ),
    )
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--tier",
        help="Difficulty tier name (e.g. D2-D10) for ladder gating.",
    )
    mode_group.add_argument(
        "--tier-id",
        help=("Heuristic eval-pool tier id " "(e.g. sq8_heuristic_baseline_v1)."),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Base RNG seed for reproducible evaluations (default: 1).",
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=None,
        help=(
            "Optional override for games per opponent/tier. When unset, the "
            "tier profile's configured num_games is used."
        ),
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help=(
            "Optional cap on games for heuristic eval-pool mode. When "
            "unset, falls back to --num-games or the tier's default."
        ),
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=None,
        help=(
            "Optional cap on moves per game for both difficulty-tier and "
            "heuristic-tier modes. When unset, the environment/theoretical "
            "defaults are used."
        ),
    )
    parser.add_argument(
        "--time-budget-ms",
        type=int,
        default=None,
        help=(
            "Optional wall-clock think-time budget override (ms) for "
            "difficulty-tier mode. When unset, uses the tier config."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help=("Optional path to write the JSON summary. When omitted, the " "summary is printed to stdout only."),
    )
    parser.add_argument(
        "--candidate-model-id",
        type=str,
        default=None,
        help=(
            "Candidate model identifier for difficulty-tier gating "
            "(e.g. checkpoint tag, profile id, or S3 key). Required when "
            "using --tier and ignored for --tier-id heuristic mode."
        ),
    )
    parser.add_argument(
        "--promotion-plan-out",
        type=str,
        default=None,
        help=(
            "Optional path to write a promotion descriptor JSON when using "
            "--tier. The descriptor records the current production model id, "
            "the candidate id, the PROMOTE/REJECT decision, and key "
            "metrics. When the candidate fails, a descriptor with "
            "decision='reject' is still written so CI can consume a "
            "uniform format."
        ),
    )
    parser.add_argument(
        "--use-candidate-artifact",
        action="store_true",
        help=(
            "When set, attempt to load the candidate artefact referenced by "
            "--candidate-model-id (heuristic profile id, NNUE model id, or "
            "neural checkpoint id). If the artefact cannot be resolved, an "
            "explicit failing gate criterion is recorded and overall_pass is "
            "forced to False so promotion plans do not accidentally promote "
            "a label-only candidate."
        ),
    )
    return parser.parse_args()


def _run_heuristic_mode(args: argparse.Namespace) -> int:
    """Legacy heuristic eval-pool tier gate (backwards compatible)."""
    tier_spec = _get_heuristic_tier_spec(args.tier_id)

    max_games = args.max_games
    if max_games is None:
        max_games = args.num_games

    result = run_heuristic_tier_eval(
        tier_spec=tier_spec,
        rng_seed=args.seed,
        max_games=max_games,
        max_moves_override=args.max_moves,
    )

    payload: dict[str, Any] = {
        "tier_id": result.get("tier_id"),
        "tier_name": result.get("tier_name"),
        "board_type": result.get("board_type"),
        "num_players": result.get("num_players"),
        "eval_pool_id": result.get("eval_pool_id"),
        "candidate_profile_id": result.get("candidate_profile_id"),
        "baseline_profile_id": result.get("baseline_profile_id"),
        "games_requested": result.get("games_requested"),
        "games_played": result.get("games_played"),
        "results": result.get("results"),
        "margins": result.get("margins"),
        "latency_ms": result.get("latency_ms"),
        "total_moves": result.get("total_moves"),
        "victory_reasons": result.get("victory_reasons"),
        "config": {
            "tier_spec_id": tier_spec.id,
            "tier_spec_name": tier_spec.name,
            "description": tier_spec.description,
        },
    }

    json_text = json.dumps(payload, indent=2, sort_keys=True)
    print(json_text)

    if args.output_json:
        out_path = os.path.abspath(args.output_json)
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(json_text)
        print(f"\nWrote tier gate report to {out_path}")

    return 0


def _print_difficulty_summary(
    tier_name: str,
    candidate_id: str,
    production_model_id: str | None,
    result: Any,
) -> None:
    """Print a concise human-readable summary for difficulty-tier gating."""
    print()
    print("=" * 72)
    print("DIFFICULTY TIER GATE SUMMARY")
    print("=" * 72)
    print(f"Tier:      {tier_name}")
    print(f"Candidate: {candidate_id}")
    board_str = f"{result.board_type.value}, {result.num_players} players"
    print(f"Board:     {board_str}")
    print(f"Games:     {result.total_games}")
    if production_model_id is not None:
        print(f"Production model id: {production_model_id}")
    print()

    print("Matchups:")
    for m in result.matchups:
        win_rate_pct = m.win_rate * 100.0
        header = f"  vs {m.opponent_id} " f"(difficulty {m.opponent_difficulty}, " f"ai={m.opponent_ai_type})"
        print(header)
        line = f"    W/D/L: {m.wins} / {m.draws} / {m.losses} " f"(win-rate: {win_rate_pct:.1f}%)"
        print(line)

    print()
    print("Gates:")
    for name, passed in result.criteria.items():
        if passed is None:
            status = "N/A"
        elif passed:
            status = "PASS"
        else:
            status = "FAIL"

        metric_key = name.replace("min_win_rate_", "win_rate_")
        metric_val = result.metrics.get(metric_key, None)
        if isinstance(metric_val, float):
            suffix = f" (observed={metric_val:.3f})"
        else:
            suffix = ""
        print(f"  {name}: {status}{suffix}")

    print()
    overall = "PASS" if result.overall_pass else "FAIL"
    decision = "PROMOTE" if result.overall_pass else "DO NOT PROMOTE"
    print(f"Overall: {overall}")
    print(f"Decision: {decision}")
    print("=" * 72)


def _run_difficulty_mode(args: argparse.Namespace) -> int:
    """Difficulty-tier gating on the canonical ladder (D2/D4/D6/D8)."""
    tier_name = args.tier.upper()
    if tier_name not in TIER_EVAL_CONFIGS:
        available = ", ".join(sorted(TIER_EVAL_CONFIGS.keys()))
        msg = f"Unknown difficulty tier '{args.tier}'. " f"Available tiers: {available}"
        raise SystemExit(msg)

    if not args.candidate_model_id:
        raise SystemExit("Difficulty-tier mode requires --candidate-model-id to be set.")

    tier_config = get_tier_config(tier_name)

    # Resolve current production LadderTierConfig for the candidate tier.
    try:
        production_ladder = get_ladder_tier_config(
            tier_config.candidate_difficulty,
            tier_config.board_type,
            tier_config.num_players,
        )
        current_model_id: str | None = production_ladder.model_id
    except Exception:
        production_ladder = None
        current_model_id = None

    candidate_override_id: str | None = None
    candidate_artifact_present: bool | None = None
    candidate_artifact_loaded: bool | None = None
    candidate_artifact_error: str | None = None

    if args.use_candidate_artifact and production_ladder is not None:
        candidate_id = str(args.candidate_model_id)
        ai_type = production_ladder.ai_type

        if ai_type == AIType.HEURISTIC:
            try:
                from app.ai.heuristic_weights import HEURISTIC_WEIGHT_PROFILES

                if candidate_id in HEURISTIC_WEIGHT_PROFILES:
                    candidate_override_id = candidate_id
            except Exception:
                candidate_override_id = None
        elif ai_type == AIType.MINIMAX:
            if bool(getattr(production_ladder, "use_neural_net", False)):
                import os as _os

                nnue_path = _os.path.join(
                    PROJECT_ROOT,
                    "models",
                    "nnue",
                    f"{candidate_id}.pt",
                )
                if _os.path.exists(nnue_path):
                    candidate_override_id = candidate_id
            else:
                candidate_override_id = candidate_id
        elif (
            ai_type == AIType.MCTS
            and not bool(getattr(production_ladder, "use_neural_net", False))
        ):
            candidate_override_id = candidate_id
        else:
            import os as _os

            for suffix in (".pth", "_mps.pth"):
                ckpt_path = _os.path.join(
                    PROJECT_ROOT,
                    "models",
                    f"{candidate_id}{suffix}",
                )
                if _os.path.exists(ckpt_path):
                    candidate_override_id = candidate_id
                    break

        candidate_artifact_present = candidate_override_id is not None

        # For neural tiers, ensure the checkpoint is loadable before running
        # a long evaluation (and before producing a promotion plan).
        if (
            candidate_override_id is not None
            and ai_type not in (AIType.HEURISTIC, AIType.MINIMAX)
            and bool(getattr(production_ladder, "use_neural_net", False))
        ):
            try:
                import os as _os

                # Preflight on CPU so we only validate checkpoint correctness.
                prev_force_cpu = _os.environ.get("RINGRIFT_FORCE_CPU")
                _os.environ["RINGRIFT_FORCE_CPU"] = "1"
                try:
                    from app.ai.neural_net import NeuralNetAI
                    from app.models import AIConfig

                    probe_cfg = AIConfig(
                        difficulty=tier_config.candidate_difficulty,
                        randomness=0.0,
                        think_time=0,
                        nn_model_id=candidate_override_id,
                        use_neural_net=True,
                        allow_fresh_weights=False,
                    )
                    NeuralNetAI(
                        player_number=1,
                        config=probe_cfg,
                        board_type=tier_config.board_type,
                    )
                    candidate_artifact_loaded = True
                finally:
                    if prev_force_cpu is None:
                        _os.environ.pop("RINGRIFT_FORCE_CPU", None)
                    else:
                        _os.environ["RINGRIFT_FORCE_CPU"] = prev_force_cpu
            except Exception as exc:
                candidate_artifact_loaded = False
                candidate_artifact_error = str(exc)

    result: TierEvaluationResult
    eval_error: str | None = None
    if args.use_candidate_artifact and production_ladder is not None:
        if candidate_artifact_present is False or candidate_artifact_loaded is False:
            result = TierEvaluationResult(
                tier_name=tier_config.tier_name,
                board_type=tier_config.board_type,
                num_players=tier_config.num_players,
                candidate_id=str(args.candidate_model_id),
                candidate_difficulty=tier_config.candidate_difficulty,
                total_games=0,
                overall_pass=False,
            )
        else:
            require_neural_env_set = False
            prev_require_neural = None
            if bool(getattr(production_ladder, "use_neural_net", False)):
                prev_require_neural = os.environ.get("RINGRIFT_REQUIRE_NEURAL_NET")
                os.environ["RINGRIFT_REQUIRE_NEURAL_NET"] = "1"
                require_neural_env_set = True
            try:
                result = run_tier_evaluation(
                    tier_config=tier_config,
                    candidate_id=args.candidate_model_id,
                    candidate_override_id=candidate_override_id,
                    seed=args.seed,
                    num_games_override=args.num_games,
                    time_budget_ms_override=args.time_budget_ms,
                    max_moves_override=args.max_moves,
                )
            except Exception as exc:
                eval_error = str(exc)
                result = TierEvaluationResult(
                    tier_name=tier_config.tier_name,
                    board_type=tier_config.board_type,
                    num_players=tier_config.num_players,
                    candidate_id=str(args.candidate_model_id),
                    candidate_difficulty=tier_config.candidate_difficulty,
                    total_games=0,
                    overall_pass=False,
                )
            finally:
                if require_neural_env_set:
                    if prev_require_neural is None:
                        os.environ.pop("RINGRIFT_REQUIRE_NEURAL_NET", None)
                    else:
                        os.environ["RINGRIFT_REQUIRE_NEURAL_NET"] = prev_require_neural
    else:
        try:
            result = run_tier_evaluation(
                tier_config=tier_config,
                candidate_id=args.candidate_model_id,
                candidate_override_id=None,
                seed=args.seed,
                num_games_override=args.num_games,
                time_budget_ms_override=args.time_budget_ms,
                max_moves_override=args.max_moves,
            )
        except Exception as exc:
            eval_error = str(exc)
            result = TierEvaluationResult(
                tier_name=tier_config.tier_name,
                board_type=tier_config.board_type,
                num_players=tier_config.num_players,
                candidate_id=str(args.candidate_model_id),
                candidate_difficulty=tier_config.candidate_difficulty,
                total_games=0,
                overall_pass=False,
            )

    if args.use_candidate_artifact:
        used = bool(candidate_artifact_present)
        result.criteria["candidate_artifact_present"] = used
        if candidate_artifact_loaded is not None:
            result.criteria["candidate_artifact_loaded"] = bool(candidate_artifact_loaded)
        if not used or candidate_artifact_loaded is False:
            result.overall_pass = False

    _print_difficulty_summary(
        tier_name=tier_name,
        candidate_id=args.candidate_model_id,
        production_model_id=current_model_id,
        result=result,
    )
    if candidate_artifact_error:
        print(f"\nCandidate artefact load error: {candidate_artifact_error}")
    if eval_error:
        print(f"\nEvaluation error: {eval_error}")

    # JSON summary (TierEvaluationResult.to_dict)
    payload = result.to_dict()
    if candidate_artifact_error:
        payload["candidate_artifact_error"] = candidate_artifact_error
    if eval_error:
        payload["evaluation_error"] = eval_error
    if current_model_id is not None:
        payload.setdefault("ladder", {})
        payload["ladder"]["current_model_id"] = current_model_id
    if args.candidate_model_id:
        payload.setdefault("ladder", {})
        payload["ladder"]["candidate_model_id"] = args.candidate_model_id
    payload.setdefault("ladder", {})
    payload["ladder"]["candidate_artifact_present"] = bool(
        candidate_artifact_present
    )
    payload["ladder"]["use_candidate_artifact"] = bool(args.use_candidate_artifact)
    if candidate_artifact_loaded is not None:
        payload["ladder"]["candidate_artifact_loaded"] = bool(candidate_artifact_loaded)

    if args.output_json:
        out_path = os.path.abspath(args.output_json)
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        print(f"\nWrote tier gate report to {out_path}")

    # Promotion descriptor
    if args.promotion_plan_out:
        ts = datetime.now(timezone.utc).isoformat()
        reason = {
            "overall_pass": result.overall_pass,
            "candidate_artifact_present": bool(candidate_artifact_present),
            "use_candidate_artifact": bool(args.use_candidate_artifact),
            "candidate_artifact_loaded": (
                bool(candidate_artifact_loaded) if candidate_artifact_loaded is not None else None
            ),
            "win_rate_vs_baseline": payload["metrics"].get("win_rate_vs_baseline"),
            "win_rate_vs_baseline_ci_low": payload["metrics"].get(
                "win_rate_vs_baseline_ci_low"
            ),
            "win_rate_vs_previous_tier": payload["metrics"].get("win_rate_vs_previous_tier"),
        }
        decision = "promote" if result.overall_pass else "reject"
        plan = {
            "tier": tier_name,
            "board_type": tier_config.board_type.value,
            "num_players": tier_config.num_players,
            "current_model_id": current_model_id,
            "candidate_model_id": args.candidate_model_id,
            "decision": decision,
            "timestamp": ts,
            "reason": reason,
        }

        out_path = os.path.abspath(args.promotion_plan_out)
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(plan, f, indent=2, sort_keys=True)
        print(f"\nWrote promotion plan to {out_path}")

    return 0


def main() -> int:
    args = parse_args()

    if args.tier:
        return _run_difficulty_mode(args)
    else:
        # Heuristic eval-pool mode (backwards compatible).
        return _run_heuristic_mode(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
