#!/usr/bin/env python
"""CI-friendly AI strength regression gate.

This script runs a small, bounded set of AI-vs-AI matchups on evaluation pools
and emits a stable JSON report plus an exit code suitable for CI:

- Exit 0: all configured matchups pass the gate.
- Exit 1: at least one matchup fails.

Design goals:
- Fast and deterministic in CI (fixed seeds, bounded scenarios, bounded moves).
- No dependency on external neural checkpoints (defaults to use_neural_net=False).
- Cross-board coverage via eval pools (square8 + square19 when available).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# CI/runtime safety defaults for this process.
# - Disable heuristic ProcessPool parallelism (it is expensive for small runs).
# - Default to FSM validation off unless explicitly enabled by the caller.
os.environ.setdefault("RINGRIFT_PARALLEL_MIN_MOVES", "100000")
os.environ.setdefault("RINGRIFT_FSM_VALIDATION_MODE", "off")

from app.main import _create_ai_instance, _get_difficulty_profile
from app.models import AIConfig, AIType, BoardType, GameState, GameStatus
from app.training.env import TrainingEnvConfig, make_env
from app.training.eval_pools import (
    EvalPoolConfig,
    EvalScenario,
    get_eval_pool_config,
    load_eval_pool,
)
from app.training.significance import wilson_score_interval


@dataclass(frozen=True)
class GateThreshold:
    min_win_rate: float
    require_significance: bool
    confidence: float


@dataclass(frozen=True)
class MatchupSpec:
    name: str
    candidate: str
    opponent: str
    pools: list[str]
    max_scenarios: int
    games_per_scenario: int
    max_moves: int
    think_time_ms: int
    use_neural_net: bool
    heuristic_move_sample_limit: int | None
    heuristic_eval_mode: str | None
    threshold: GateThreshold
    blocking: bool


def _normalise_board_label(board_type: BoardType) -> str:
    v = board_type.value
    return "hex" if v == "hexagonal" else v


def _resolve_ai_spec(spec: str) -> tuple[AIType, int]:
    """Resolve either an engine spec or a tier id like D8."""
    raw = spec.strip()
    if raw.upper().startswith("D") and raw[1:].isdigit():
        difficulty = int(raw[1:])
        profile = _get_difficulty_profile(difficulty)
        return profile["ai_type"], difficulty

    name = raw.lower()
    engine_map: dict[str, tuple[AIType, int]] = {
        "random": (AIType.RANDOM, 1),
        "heuristic": (AIType.HEURISTIC, 2),
        "minimax": (AIType.MINIMAX, 4),
        "mcts": (AIType.MCTS, 7),
        "descent": (AIType.DESCENT, 9),
    }
    if name not in engine_map:
        raise ValueError(
            f"Unknown AI spec {spec!r}; expected an engine spec "
            f"({', '.join(sorted(engine_map.keys()))}) or a tier id like 'D8'."
        )
    return engine_map[name]


def _build_ai(
    spec: str,
    *,
    player_number: int,
    rng_seed: int | None,
    think_time_ms: int,
    use_neural_net: bool,
    heuristic_move_sample_limit: int | None,
    heuristic_eval_mode: str | None,
) -> Any:
    ai_type, difficulty = _resolve_ai_spec(spec)
    profile = _get_difficulty_profile(difficulty)

    cfg = AIConfig(
        difficulty=difficulty,
        randomness=float(profile["randomness"]),
        think_time=int(think_time_ms),
        rngSeed=rng_seed,
        heuristic_profile_id=None,
        nn_model_id=None,
        use_neural_net=bool(use_neural_net),
        training_move_sample_limit=heuristic_move_sample_limit,
        heuristic_eval_mode=heuristic_eval_mode,
    )
    return _create_ai_instance(ai_type, player_number, cfg)


def _compute_gate(
    *,
    wins: int,
    losses: int,
    draws: int,
    threshold: GateThreshold,
) -> dict[str, Any]:
    decisive = wins + losses
    if decisive <= 0:
        win_rate = 0.5
        ci_low = None
        ci_high = None
    else:
        win_rate = wins / float(decisive)
        ci_low, ci_high = wilson_score_interval(
            wins,
            decisive,
            confidence=threshold.confidence,
        )

    passes = win_rate >= float(threshold.min_win_rate)
    if threshold.require_significance:
        # When no decisive games exist (all draws), we conservatively fail.
        if ci_low is None:
            passes = False
        else:
            passes = passes and (ci_low >= float(threshold.min_win_rate))

    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "decisive_games": decisive,
        "win_rate": win_rate,
        "confidence": threshold.confidence,
        "win_rate_ci_low": ci_low,
        "win_rate_ci_high": ci_high,
        "min_win_rate": threshold.min_win_rate,
        "require_significance": threshold.require_significance,
        "passes": passes,
    }


def _pool_exists(pool_name: str) -> bool:
    try:
        pool_cfg = get_eval_pool_config(pool_name)
    except KeyError:
        return False

    path = _pool_path(pool_cfg)
    return path.exists()


def _pool_path(pool_cfg: EvalPoolConfig) -> Path:
    # Mirror eval_pools path conventions without importing internal constants.
    base_dir = PROJECT_ROOT / "data" / "eval_pools"
    board_dir = _normalise_board_label(pool_cfg.board_type)
    filename = f"pool_{pool_cfg.pool_id}.jsonl"
    return base_dir / board_dir / filename


def _load_late_game_scenarios(
    pool_name: str,
    *,
    max_scenarios: int,
) -> list[EvalScenario]:
    """Load late-game snapshots from a pool.

    The pool JSONLs contain sequential snapshots from many games. For speed and
    decisiveness, we select (at most) one snapshot per detected game segment:
    the snapshot with the longest move_history within that segment. We then
    take the top-N segments by move_history length.
    """
    pool_cfg = get_eval_pool_config(pool_name)
    path = _pool_path(pool_cfg)

    best_by_segment: list[tuple[int, str]] = []
    best_moves = -1
    best_line = ""
    last_moves: int | None = None

    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            obj = json.loads(line)
            moves_len = len(obj.get("move_history") or [])

            # Heuristic: move_history length drops when a new game starts.
            if last_moves is not None and moves_len < last_moves:
                if best_moves >= 0 and best_line:
                    best_by_segment.append((best_moves, best_line))
                best_moves = -1
                best_line = ""

            if moves_len > best_moves:
                best_moves = moves_len
                best_line = line

            last_moves = moves_len

    if best_moves >= 0 and best_line:
        best_by_segment.append((best_moves, best_line))

    best_by_segment.sort(key=lambda t: t[0], reverse=True)
    selected = best_by_segment[: max(0, int(max_scenarios))]

    scenarios: list[EvalScenario] = []
    for idx, (moves_len, line) in enumerate(selected):
        state = GameState.model_validate_json(line)  # type: ignore[attr-defined]
        scenarios.append(
            EvalScenario(
                id=f"{pool_name}:late:{idx}",
                board_type=pool_cfg.board_type,
                num_players=pool_cfg.num_players,
                initial_state=state,
                metadata={
                    "pool_name": pool_name,
                    "pool_id": pool_cfg.pool_id,
                    "segment_rank": idx,
                    "snapshot_move_history_len": moves_len,
                },
            )
        )
    return scenarios


def _run_matchup_on_pool(
    *,
    pool_cfg: EvalPoolConfig,
    scenarios: list[EvalScenario],
    candidate_spec: str,
    opponent_spec: str,
    games_per_scenario: int,
    base_seed: int,
    think_time_ms: int,
    use_neural_net: bool,
    heuristic_move_sample_limit: int | None,
    heuristic_eval_mode: str | None,
    max_moves: int,
) -> dict[str, Any]:
    env = make_env(
        TrainingEnvConfig(
            board_type=pool_cfg.board_type,
            num_players=pool_cfg.num_players,
            max_moves=max_moves,
            reward_mode="terminal",
        )
    )

    wins = 0
    losses = 0
    draws = 0
    total_games = 0
    total_moves = 0
    victory_reasons: dict[str, int] = {}

    for s_idx, scenario in enumerate(scenarios):
        # Eval pools are expected to contain canonical GameState snapshots.
        player_numbers = [p.player_number for p in scenario.initial_state.players]
        if len(player_numbers) != 2:
            raise RuntimeError(
                f"Strength regression gate currently supports 2-player pools only "
                f"(pool={pool_cfg.name!r}, players={len(player_numbers)})."
            )
        p1, p2 = player_numbers

        for g_idx in range(games_per_scenario):
            game_seed = (base_seed + s_idx * 1_000_003 + g_idx) & 0x7FFFFFFF

            # Alternate seats for fairness.
            if g_idx % 2 == 0:
                candidate_as, opponent_as = p1, p2
            else:
                candidate_as, opponent_as = p2, p1

            ai_by_player: dict[int, Any] = {
                candidate_as: _build_ai(
                    candidate_spec,
                    player_number=candidate_as,
                    rng_seed=game_seed + candidate_as * 97,
                    think_time_ms=think_time_ms,
                    use_neural_net=use_neural_net,
                    heuristic_move_sample_limit=heuristic_move_sample_limit,
                    heuristic_eval_mode=heuristic_eval_mode,
                ),
                opponent_as: _build_ai(
                    opponent_spec,
                    player_number=opponent_as,
                    rng_seed=game_seed + opponent_as * 97,
                    think_time_ms=think_time_ms,
                    use_neural_net=use_neural_net,
                    heuristic_move_sample_limit=heuristic_move_sample_limit,
                    heuristic_eval_mode=heuristic_eval_mode,
                ),
            }

            env.reset(seed=game_seed)
            env._state = scenario.initial_state.copy(deep=True)  # Pydantic v1
            env._move_count = 0

            moves_played = 0
            last_info: dict[str, Any] = {}
            while True:
                state = env.state
                if state.game_status != GameStatus.ACTIVE:
                    break

                legal_moves = env.legal_moves()
                if not legal_moves:
                    break

                current_player = state.current_player
                ai = ai_by_player.get(current_player)
                if ai is None:
                    raise RuntimeError(f"Missing AI for player {current_player}")

                move = ai.select_move(state)
                if move is None:
                    # Treat failure to produce a move as a draw to avoid noisy
                    # seat-dependent bookkeeping; the healthcheck suite covers
                    # this more directly.
                    state.game_status = GameStatus.COMPLETED
                    state.winner = None
                    break

                _next_state, _reward, done, info = env.step(move)
                last_info = info
                moves_played += 1
                if done:
                    break

            total_games += 1
            total_moves += moves_played

            winner = env.state.winner
            if winner is None:
                draws += 1
            elif winner == candidate_as:
                wins += 1
            else:
                losses += 1

            reason = last_info.get("victory_reason", "unknown")
            victory_reasons[reason] = victory_reasons.get(reason, 0) + 1

    avg_len = float(total_moves) / float(total_games) if total_games > 0 else 0.0
    return {
        "pool_name": pool_cfg.name,
        "board_type": pool_cfg.board_type.value,
        "num_players": pool_cfg.num_players,
        "total_games": total_games,
        "avg_game_length": avg_len,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "victory_reasons": victory_reasons,
        "scenario_selection": "late_game_max_per_segment",
    }


def _default_matchups(mode: str) -> list[MatchupSpec]:
    require_sig = mode == "nightly"
    confidence = 0.95

    def _t(min_win_rate: float) -> GateThreshold:
        return GateThreshold(
            min_win_rate=min_win_rate,
            require_significance=require_sig,
            confidence=confidence,
        )

    if mode == "nightly":
        # More coverage for scheduled runs while keeping runtime bounded.
        return [
            MatchupSpec(
                name="heuristic_vs_random",
                candidate="heuristic",
                opponent="random",
                pools=["square8_2p_core", "square19_2p_core"],
                max_scenarios=6,
                games_per_scenario=6,
                max_moves=120,
                think_time_ms=5,
                use_neural_net=False,
                threshold=_t(0.55),
                heuristic_move_sample_limit=64,
                heuristic_eval_mode="light",
                blocking=True,
            ),
            MatchupSpec(
                name="mcts_vs_random",
                candidate="mcts",
                opponent="random",
                pools=["square8_2p_core"],
                max_scenarios=4,
                games_per_scenario=4,
                max_moves=120,
                think_time_ms=25,
                use_neural_net=False,
                threshold=_t(0.52),
                heuristic_move_sample_limit=None,
                heuristic_eval_mode=None,
                blocking=True,
            ),
            MatchupSpec(
                name="descent_vs_random",
                candidate="descent",
                opponent="random",
                pools=["square8_2p_core"],
                max_scenarios=5,
                games_per_scenario=3,
                max_moves=120,
                think_time_ms=25,
                use_neural_net=False,
                # Descent strength is monitored, but we keep this gate
                # conservative (no significance requirement) to avoid flakiness
                # when neural checkpoints are unavailable in CI contexts.
                threshold=GateThreshold(
                    min_win_rate=0.5,
                    require_significance=False,
                    confidence=confidence,
                ),
                heuristic_move_sample_limit=None,
                heuristic_eval_mode=None,
                blocking=False,
            ),
        ]

    # CI smoke: small but meaningful sample sizes.
    return [
        MatchupSpec(
            name="heuristic_vs_random",
            candidate="heuristic",
            opponent="random",
            pools=["square8_2p_core", "square19_2p_core"],
            max_scenarios=2,
            games_per_scenario=2,
            max_moves=100,
            think_time_ms=5,
            use_neural_net=False,
            threshold=_t(0.55),
            heuristic_move_sample_limit=64,
            heuristic_eval_mode="light",
            blocking=True,
        ),
        MatchupSpec(
            name="mcts_vs_random",
            candidate="mcts",
            opponent="random",
            pools=["square8_2p_core"],
            max_scenarios=2,
            games_per_scenario=2,
            max_moves=100,
            think_time_ms=25,
            use_neural_net=False,
            threshold=_t(0.52),
            heuristic_move_sample_limit=None,
            heuristic_eval_mode=None,
            blocking=True,
        ),
        MatchupSpec(
            name="descent_vs_random",
            candidate="descent",
            opponent="random",
            pools=["square8_2p_core"],
            max_scenarios=3,
            games_per_scenario=2,
            max_moves=100,
            think_time_ms=200,
            use_neural_net=False,
            threshold=_t(0.5),
            heuristic_move_sample_limit=None,
            heuristic_eval_mode=None,
            blocking=False,
        ),
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode",
        choices=["ci", "nightly"],
        default="ci",
        help="Preset mode controlling sample sizes and gating strictness.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=1_234_567,
        help="Base RNG seed for reproducible runs.",
    )
    p.add_argument(
        "--output-json",
        type=str,
        default="results/ai_strength_regression_gate.json",
        help="Path to write the JSON report.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    start = time.time()

    matchups = _default_matchups(args.mode)

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    matchups_report: dict[str, Any] = {}
    overall_pass = True
    skipped_pools: dict[str, str] = {}

    for matchup in matchups:
        pools_report: dict[str, Any] = {}
        total_wins = 0
        total_losses = 0
        total_draws = 0

        for pool_name in matchup.pools:
            if not _pool_exists(pool_name):
                skipped_pools[pool_name] = "pool_file_missing"
                continue

            pool_cfg = get_eval_pool_config(pool_name)
            scenarios = _load_late_game_scenarios(
                pool_name,
                max_scenarios=matchup.max_scenarios,
            )
            if not scenarios:
                skipped_pools[pool_name] = "pool_empty"
                continue

            pool_summary = _run_matchup_on_pool(
                pool_cfg=pool_cfg,
                scenarios=scenarios,
                candidate_spec=matchup.candidate,
                opponent_spec=matchup.opponent,
                games_per_scenario=matchup.games_per_scenario,
                base_seed=int(args.seed),
                think_time_ms=int(matchup.think_time_ms),
                use_neural_net=bool(matchup.use_neural_net),
                heuristic_move_sample_limit=matchup.heuristic_move_sample_limit,
                heuristic_eval_mode=matchup.heuristic_eval_mode,
                max_moves=int(matchup.max_moves),
            )
            pools_report[pool_name] = pool_summary
            total_wins += int(pool_summary["wins"])
            total_losses += int(pool_summary["losses"])
            total_draws += int(pool_summary["draws"])

        gate = _compute_gate(
            wins=total_wins,
            losses=total_losses,
            draws=total_draws,
            threshold=matchup.threshold,
        )

        passes = bool(gate["passes"])
        if matchup.blocking:
            overall_pass = overall_pass and passes

        matchups_report[matchup.name] = {
            "candidate": matchup.candidate,
            "opponent": matchup.opponent,
            "pools": matchup.pools,
            "blocking": matchup.blocking,
            "settings": {
                "max_scenarios": matchup.max_scenarios,
                "games_per_scenario": matchup.games_per_scenario,
                "max_moves": matchup.max_moves,
                "think_time_ms": matchup.think_time_ms,
                "use_neural_net": matchup.use_neural_net,
                },
            "gate": gate,
            "by_pool": pools_report,
            "passes": passes,
        }

    report: dict[str, Any] = {
        "mode": args.mode,
        "seed": int(args.seed),
        "overall_pass": overall_pass,
        "created_at": datetime.now(timezone.utc).isoformat() + "Z",
        "runtime_sec": round(time.time() - start, 3),
        "skipped_pools": skipped_pools,
        "matchups": matchups_report,
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    print(f"Wrote strength regression gate report to {out_path}")
    return 0 if overall_pass else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
