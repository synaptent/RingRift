#!/usr/bin/env python
"""Run evaluation tournaments on named state pools.

This harness plays short AI-vs-AI tournaments starting from mid/late-game
snapshots stored under ``ai-service/data/eval_pools/**`` and writes a
structured JSON report. It is intended for:

- Large-board evaluation (Square19, Hex).
- Multi-player evaluation (3p/4p) on Square8 and Square19.
- CI-safe demo runs via ``--demo`` with very small configs.

Pools are resolved via :mod:`app.training.eval_pools` and AI instances are
constructed from either the difficulty ladder
(:mod:`app.config.ladder_config`) or simple engine specs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.main import (  # noqa: E402
    _create_ai_instance,
    _get_difficulty_profile,
)
from app.config.ladder_config import (  # noqa: E402
    LadderTierConfig,
    get_ladder_tier_config,
)
from app.models import (  # noqa: E402
    AIConfig,
    AIType,
    BoardType,
    GameStatus,
)
from app.training.env import (  # noqa: E402
    TrainingEnvConfig,
    get_theoretical_max_moves,
    make_env,
)
from app.training.eval_pools import (  # noqa: E402
    EvalScenario,
    EvalPoolConfig,
    get_eval_pool_config,
    load_eval_pool,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the eval tournament harness."""
    parser = argparse.ArgumentParser(
        description=("Run evaluation tournaments for AI tiers/engines on named " "evaluation pools."),
    )
    parser.add_argument(
        "--pool",
        required=True,
        help=(
            "Logical eval pool name as defined in app.training.eval_pools. "
            "Examples: square19_2p_core, hex_4p_baseline, "
            "square8_3p_baseline."
        ),
    )
    parser.add_argument(
        "--tier",
        help=(
            "Difficulty tier id (e.g. D2, D4, D6, D8). When provided, AIs are "
            "constructed via ladder_config for this tier and the pool's "
            "(board_type, num_players)."
        ),
    )
    parser.add_argument(
        "--engine",
        help=(
            "Fallback engine spec when --tier is not supplied. Accepted "
            "values: random, heuristic, minimax, mcts, descent."
        ),
    )
    parser.add_argument(
        "--opponent",
        help=(
            "Optional opponent spec for 2-player pools: another tier id "
            "(e.g. D2) or an engine spec (random, heuristic, minimax, "
            "mcts, descent). When omitted, both sides use the primary "
            "spec. Ignored for multiplayer pools (num_players > 2)."
        ),
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=2,
        help=(
            "Number of games to play per scenario (default: 2). In --demo "
            "mode this is typically capped to a very small value."
        ),
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Directory where the JSON report will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional base RNG seed for deterministic tournaments.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help=("Enable CI-safe demo mode: limit scenarios/games and force " "small move budgets and think_time=0."),
    )
    return parser.parse_args(argv)


def _normalise_tier_name(tier: str) -> str:
    """Return a canonical tier name like ``'D4'`` or raise ValueError."""
    name = tier.strip().upper()
    if not name.startswith("D") or not name[1:].isdigit():
        raise ValueError(f"Unsupported tier name {tier!r}; expected something like 'D2'.")
    return name


def _build_ai_for_tier(
    tier_name: str,
    pool_cfg: EvalPoolConfig,
    player_number: int,
    rng_seed: int | None,
) -> Any:
    """Construct an AI instance from the ladder tier for this pool.

    We deliberately override ``think_time`` to 0 so that tournaments are
    cheap enough for CI and local smoke runs; search-based engines still
    see the tier's difficulty and randomness but run with a very small
    effective budget.
    """
    difficulty = int(tier_name[1:])
    try:
        tier_cfg: LadderTierConfig = get_ladder_tier_config(
            difficulty,
            pool_cfg.board_type,
            pool_cfg.num_players,
        )
    except KeyError as exc:  # pragma: no cover - defensive
        raise SystemExit(
            "No ladder tier configured for "
            f"difficulty={difficulty}, board_type={pool_cfg.board_type}, "
            f"num_players={pool_cfg.num_players}. "
            "Extend ladder_config or choose a different pool/tier."
        ) from exc

    nn_model_id: str | None = None
    if tier_cfg.ai_type in (AIType.MCTS, AIType.DESCENT):
        nn_model_id = tier_cfg.model_id

    cfg = AIConfig(
        difficulty=tier_cfg.difficulty,
        randomness=tier_cfg.randomness,
        think_time=0,
        rngSeed=rng_seed,
        heuristic_profile_id=tier_cfg.heuristic_profile_id,
        nn_model_id=nn_model_id,
    )
    return _create_ai_instance(tier_cfg.ai_type, player_number, cfg)


def _build_ai_from_engine_spec(
    engine: str,
    player_number: int,
    rng_seed: int | None,
) -> Any:
    """Construct an AI instance from a simple engine spec.

    The spec is one of: random, heuristic, minimax, mcts, descent.
    Difficulty and randomness are taken from the canonical difficulty
    profiles in :mod:`app.main`, but think_time is forced to 0 for
    evaluation.
    """
    name = engine.strip().lower()
    engine_map: dict[str, tuple[AIType, int]] = {
        "random": (AIType.RANDOM, 1),
        "heuristic": (AIType.HEURISTIC, 2),
        "minimax": (AIType.MINIMAX, 4),
        "mcts": (AIType.MCTS, 7),
        "descent": (AIType.DESCENT, 9),
    }
    if name not in engine_map:
        raise ValueError(f"Unknown engine spec {engine!r}; expected one of " f"{', '.join(sorted(engine_map.keys()))}.")
    ai_type, difficulty = engine_map[name]
    profile = _get_difficulty_profile(difficulty)
    cfg = AIConfig(
        difficulty=difficulty,
        randomness=profile["randomness"],
        think_time=0,
        rngSeed=rng_seed,
        heuristic_profile_id=None,
        nn_model_id=None,
    )
    return _create_ai_instance(ai_type, player_number, cfg)


def _build_ai_for_spec(
    spec: str,
    pool_cfg: EvalPoolConfig,
    player_number: int,
    rng_seed: int | None,
) -> Any:
    """Construct an AI from either a tier id or an engine spec string."""
    try:
        tier_name = _normalise_tier_name(spec)
    except ValueError:
        return _build_ai_from_engine_spec(spec, player_number, rng_seed)
    return _build_ai_for_tier(tier_name, pool_cfg, player_number, rng_seed)


def _effective_num_games(num_games: int, demo: bool) -> int:
    if num_games <= 0:
        num_games = 1
    if demo:
        return min(num_games, 2)
    return num_games


def _effective_max_moves(
    board_type: BoardType,
    num_players: int,
    demo: bool,
) -> int:
    base = get_theoretical_max_moves(board_type, num_players)
    if demo:
        # Keep demo runs bounded even on large boards.
        return min(base, 100)
    return base


def _build_report_filename(
    pool_cfg: EvalPoolConfig,
    tier: str | None,
    opponent: str | None,
    demo: bool,
) -> str:
    """Return a descriptive JSON filename for the report.

    Example (Square19 2p, D4 vs D2, demo):

        eval_results.square19_2p.D4_vs_D2.demo.json
    """
    board_label = pool_cfg.board_type.value
    players_label = f"{pool_cfg.num_players}p"
    prefix = f"eval_results.{board_label}_{players_label}."
    if tier:
        tier_part = _normalise_tier_name(tier)
    else:
        tier_part = "engine"
    if opponent:
        opp_part = f"_vs_{opponent}"
    else:
        opp_part = "_vs_self"
    demo_suffix = ".demo" if demo else ""
    return f"{prefix}{tier_part}{opp_part}{demo_suffix}.json"


def _run_tournament_on_pool(
    pool_name: str,
    pool_cfg: EvalPoolConfig,
    scenarios: list[EvalScenario],
    primary_spec: str,
    opponent_spec: str | None,
    num_games: int,
    base_seed: int,
    demo: bool,
) -> dict[str, Any]:
    """Core tournament loop over a single evaluation pool."""
    from app.training.env import RingRiftEnv  # noqa: E402

    effective_games = _effective_num_games(num_games, demo)
    max_moves = _effective_max_moves(
        pool_cfg.board_type,
        pool_cfg.num_players,
        demo,
    )

    env_cfg = TrainingEnvConfig(
        board_type=pool_cfg.board_type,
        num_players=pool_cfg.num_players,
        max_moves=max_moves,
        reward_mode="terminal",
    )
    env: RingRiftEnv = make_env(env_cfg)

    multiplayer = pool_cfg.num_players > 2
    if multiplayer and opponent_spec:
        # Simpler and well-defined semantics: for multiplayer pools we run
        # symmetric self-play where all seats use the primary spec. The
        # opponent spec is ignored but preserved in the report for clarity.
        print(
            "Warning: --opponent is ignored for multiplayer pools " "(num_players > 2); running symmetric evaluation.",
        )

    scenario_results: list[dict[str, Any]] = []
    total_games = 0
    total_moves_all = 0

    for s_idx, scenario in enumerate(scenarios):
        games_for_scenario = 0
        wins_by_player: dict[int, int] = {}
        victory_reasons: dict[str, int] = {}
        total_moves_for_scenario = 0

        player_numbers = [p.player_number for p in scenario.initial_state.players]

        for game_index in range(effective_games):
            game_seed = (base_seed + s_idx * 1_000_003 + game_index) & 0x7FFFFFFF

            # Build per-player AIs.
            ai_by_player: dict[int, Any] = {}
            if multiplayer:
                for pnum in player_numbers:
                    rng = game_seed + pnum * 97
                    ai_by_player[pnum] = _build_ai_for_spec(
                        primary_spec,
                        pool_cfg,
                        pnum,
                        rng,
                    )
            else:
                if len(player_numbers) != 2:
                    raise RuntimeError(
                        "2-player pool reported num_players=2 but GameState " f"has players={len(player_numbers)}",
                    )
                p1, p2 = player_numbers
                if opponent_spec:
                    # Alternate seats for fairness: primary plays as p1 on even
                    # games, p2 on odd games.
                    if game_index % 2 == 0:
                        primary_as = p1
                        opponent_as = p2
                    else:
                        primary_as = p2
                        opponent_as = p1
                    ai_by_player[primary_as] = _build_ai_for_spec(
                        primary_spec,
                        pool_cfg,
                        primary_as,
                        game_seed + primary_as * 97,
                    )
                    ai_by_player[opponent_as] = _build_ai_for_spec(
                        opponent_spec,
                        pool_cfg,
                        opponent_as,
                        game_seed + opponent_as * 97,
                    )
                else:
                    # Self-play: both seats use the same spec.
                    for pnum in player_numbers:
                        ai_by_player[pnum] = _build_ai_for_spec(
                            primary_spec,
                            pool_cfg,
                            pnum,
                            game_seed + pnum * 97,
                        )

            # Reset env and inject scenario snapshot as the starting state.
            env.reset(seed=game_seed)
            # mypy/pydantic: model_copy is available on pydantic v2 models.
            # The attribute is present at runtime even if type-checkers
            # disagree.
            env._state = scenario.initial_state.model_copy(
                deep=True,
            )
            env._move_count = 0

            moves_played = 0
            last_info: dict[str, Any] = {}
            while True:
                state = env.state
                if state.game_status != GameStatus.ACTIVE:
                    break

                legal_moves = env.legal_moves()
                if not legal_moves:
                    # Host-level ACTIVE-no-moves invariant should normally
                    # prevent this; treat as a structural termination.
                    break

                current_player = state.current_player
                ai = ai_by_player.get(current_player)
                if ai is None:
                    raise RuntimeError(
                        "No AI instance configured for player " f"{current_player} in tournament loop.",
                    )

                move = ai.select_move(state)
                if move is None:
                    # No move returned by AI – treat as immediate loss for the
                    # side to move by letting the environment handle the
                    # resulting terminal state.
                    state.game_status = GameStatus.COMPLETED
                    state.winner = None
                    break

                _next_state, _reward, done, info = env.step(move)
                last_info = info
                moves_played += 1
                if done:
                    break

            games_for_scenario += 1
            total_games += 1
            total_moves_for_scenario += moves_played
            total_moves_all += moves_played

            final_state = env.state
            winner = final_state.winner
            if winner is not None:
                wins_by_player[winner] = wins_by_player.get(winner, 0) + 1

            reason = last_info.get("victory_reason", "unknown")
            victory_reasons[reason] = victory_reasons.get(reason, 0) + 1

        avg_length = float(total_moves_for_scenario) / games_for_scenario if games_for_scenario > 0 else 0.0
        scenario_results.append(
            {
                "scenario_id": scenario.id,
                "scenario_index": s_idx,
                "games_played": games_for_scenario,
                "avg_game_length": avg_length,
                "wins_by_player": wins_by_player,
                "victory_reasons": victory_reasons,
            },
        )

    avg_game_length_all = float(total_moves_all) / total_games if total_games > 0 else 0.0

    report: dict[str, Any] = {
        "pool_name": pool_name,
        "board_type": pool_cfg.board_type.value,
        "num_players": pool_cfg.num_players,
        # Convenience fields: ``tier`` / ``opponent`` mirror the raw specs
        # passed on the CLI (tier id or engine name), while
        # ``primary_spec`` / ``opponent_spec`` remain the general identifiers
        # used internally.
        "tier": primary_spec,
        "opponent": opponent_spec,
        "primary_spec": primary_spec,
        "opponent_spec": opponent_spec,
        "mode": "demo" if demo else "full",
        "num_games_per_scenario": effective_games,
        "total_scenarios": len(scenarios),
        "total_games": total_games,
        "avg_game_length": avg_game_length_all,
        "results": scenario_results,
        "created_at": datetime.now(timezone.utc).isoformat() + "Z",
    }
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for the eval tournament harness."""
    args = parse_args(argv)

    if not args.tier and not args.engine:
        raise SystemExit("Must supply either --tier or --engine.")
    if args.tier and args.engine:
        raise SystemExit("Only one of --tier or --engine may be supplied.")

    pool_cfg = get_eval_pool_config(args.pool)

    primary_spec = args.tier or args.engine
    assert primary_spec is not None  # for type-checkers

    base_seed = args.seed if args.seed is not None else 1_234_567

    # In demo mode we keep the number of scenarios tiny to ensure fast runs.
    max_scenarios: int | None
    if args.demo:
        max_scenarios = 2
    else:
        max_scenarios = None

    scenarios = load_eval_pool(
        name=args.pool,
        max_scenarios=max_scenarios,
    )
    if not scenarios:
        raise SystemExit(
            f"Evaluation pool {args.pool!r} is empty; generate states first.",
        )

    os.makedirs(args.run_dir, exist_ok=True)

    report = _run_tournament_on_pool(
        pool_name=args.pool,
        pool_cfg=pool_cfg,
        scenarios=scenarios,
        primary_spec=primary_spec,
        opponent_spec=args.opponent,
        num_games=args.num_games,
        base_seed=base_seed,
        demo=bool(args.demo),
    )

    filename = _build_report_filename(
        pool_cfg=pool_cfg,
        tier=args.tier,
        opponent=args.opponent,
        demo=bool(args.demo),
    )
    out_path = os.path.join(args.run_dir, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print(f"Wrote eval tournament report to {out_path}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
