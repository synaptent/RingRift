#!/usr/bin/env python3
"""Measure natural per-seat win-rate asymmetry in multiplayer games.

Background
----------
The A1 quality-gate ``_check_seat_fairness`` in
``scripts/lib/model_quality_gate.py`` computes the max/min per-seat WR
ratio when a CANDIDATE rotates through every seat in staged_evaluate,
and emits ``SEAT_WR_IMBALANCE`` if the ratio exceeds 1.5.  The implicit
null hypothesis is "a fair candidate wins each seat with equal
probability".

That null is wrong for multiplayer games with turn-order effects.  A
single iter 4 metrics row on hex8_4p showed candidate seat1=15%, seat4=25%
— ratio 1.67 — and the gate fired.  But that same iter's SELFPLAY block
(same model on all 4 seats, 100 games) showed p1=17/25/28/30 (ratio
1.76), which is even higher.  In selfplay the players are identical, so
any seat WR difference is a property of the GAME (who moves first,
positional effects) not the model.

This script measures the natural per-seat WR distribution in
hex8_4p (and optionally other configs) using four independent, identical
``HeuristicAI`` players — no neural net, no seat-specific signal from
model training.  The result is the empirical null the fairness check
should be comparing against.

Output
------
Writes a JSON summary to ``data/seat_fairness_baseline/<config>.json``
and prints a table to stdout.

Safety
------
Read-only with respect to production.  Runs locally (CPU-only heuristic
play).  Never touches models/, never modifies ``minimal_alphazero_loop.py``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ai.heuristic_ai import HeuristicAI  # noqa: E402
from app.models import AIConfig, BoardType, GameStatus  # noqa: E402
from app.training.env import (  # noqa: E402
    TrainingEnvConfig,
    get_theoretical_max_moves,
    make_env,
)

BOARD_ENUM = {
    "hex8": BoardType.HEX8,
    "hexagonal": BoardType.HEXAGONAL,
    "square8": BoardType.SQUARE8,
    "square19": BoardType.SQUARE19,
}


def _make_heuristic_ai(player: int) -> HeuristicAI:
    """Build a default, deterministic (seed-controlled) HeuristicAI."""
    cfg = AIConfig(
        difficulty=6,
        randomness=0.0,
        use_neural_net=False,
    )
    return HeuristicAI(player, cfg)


def _make_env(board: BoardType, num_players: int):
    tmax = get_theoretical_max_moves(board, num_players)
    return make_env(
        TrainingEnvConfig(
            board_type=board,
            num_players=num_players,
            max_moves=int(tmax * 1.5),
        )
    )


def _play_game(env, ais: dict[int, HeuristicAI], seed: int) -> int | None:
    """Play one game, return winner seat (1..N) or None for draw/timeout."""
    state = env.reset(seed=seed)
    for ai in ais.values():
        if hasattr(ai, "reset_for_new_game"):
            try:
                ai.reset_for_new_game(rng_seed=seed & 0xFFFFFFFF)
            except Exception:
                pass
    move_count = 0
    max_moves = int(get_theoretical_max_moves(env.board_type, env.num_players) * 1.5)
    while state.game_status == GameStatus.ACTIVE and move_count < max_moves:
        cp = state.current_player
        ai = ais.get(cp)
        if ai is None:
            break
        ai.player_number = cp
        legal = env.legal_moves()
        if not legal:
            break
        mv = ai.select_move(state)
        if mv is None or mv not in legal:
            mv = legal[random.randint(0, len(legal) - 1)]
        state, _, done, _ = env.step(mv)
        move_count += 1
        if done:
            break
    if state.game_status == GameStatus.COMPLETED:
        return state.winner
    return None


def run_baseline(board: BoardType, num_players: int, num_games: int, base_seed: int) -> dict:
    env = _make_env(board, num_players)
    ais = {p: _make_heuristic_ai(p) for p in range(1, num_players + 1)}
    seat_wins: dict[int, int] = {p: 0 for p in range(1, num_players + 1)}
    total_draws = 0
    t0 = time.time()
    for i in range(num_games):
        gseed = (base_seed + i * 7919) & 0xFFFFFFFF
        winner = _play_game(env, ais, gseed)
        if winner is None:
            total_draws += 1
        else:
            seat_wins[winner] = seat_wins.get(winner, 0) + 1
        if (i + 1) % max(1, num_games // 10) == 0:
            elapsed = time.time() - t0
            print(
                f"  progress: {i + 1}/{num_games} games in {elapsed:.0f}s "
                f"({(i + 1) / elapsed:.2f} games/s)",
                flush=True,
            )
    elapsed = time.time() - t0

    # Statistics
    total_decided = sum(seat_wins.values())
    seat_wr = {p: (w / total_decided) if total_decided > 0 else 0.0 for p, w in seat_wins.items()}
    # vs uniform (1/N) expectation
    expected_wr = 1.0 / num_players
    chi2 = 0.0
    if total_decided > 0:
        for p, w in seat_wins.items():
            exp_w = expected_wr * total_decided
            if exp_w > 0:
                chi2 += (w - exp_w) ** 2 / exp_w
    ratio = max(seat_wr.values()) / max(min(seat_wr.values()), 1e-9) if seat_wins else 0.0

    return {
        "board_type": board.value,
        "num_players": num_players,
        "num_games": num_games,
        "draws": total_draws,
        "decided": total_decided,
        "elapsed_s": round(elapsed, 1),
        "seat_wins": seat_wins,
        "seat_wr": {p: round(wr, 4) for p, wr in seat_wr.items()},
        "max_min_ratio": round(ratio, 3),
        "chi2_vs_uniform": round(chi2, 3),
        "chi2_df": num_players - 1,
        "fairness_threshold_exceeded_at_1p5": ratio > 1.5,
        "expected_uniform_wr": round(expected_wr, 4),
        "ai_type": "heuristic_default",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board-type", default="hex8", choices=list(BOARD_ENUM))
    parser.add_argument("--num-players", type=int, default=4)
    parser.add_argument("--num-games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default="data/seat_fairness_baseline",
        help="Directory for JSON result files (created if missing).",
    )
    args = parser.parse_args()

    board = BOARD_ENUM[args.board_type]
    print(
        f"\n=== seat fairness baseline: {args.board_type} {args.num_players}p "
        f"({args.num_games} games, seed={args.seed}, ai=heuristic x{args.num_players}) ==="
    )
    result = run_baseline(board, args.num_players, args.num_games, args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.board_type}_{args.num_players}p_{args.num_games}g_seed{args.seed}.json"
    out_path.write_text(json.dumps(result, indent=2))

    # Pretty print
    print("\nResults:")
    print(f"  total games:       {result['num_games']}")
    print(f"  draws:             {result['draws']}")
    print(f"  elapsed:           {result['elapsed_s']}s")
    print(f"  expected uniform:  {result['expected_uniform_wr']:.3f}")
    print("  per-seat:")
    for p in sorted(result["seat_wins"]):
        w = result["seat_wins"][p]
        n = result["decided"] or 1
        wr = result["seat_wr"][p]
        # 95% Wilson CI
        denom = 1 + 3.8416 / n
        center = (wr + 1.9208 / n) / denom
        half = 1.96 * math.sqrt(wr * (1 - wr) / n + 0.9604 / (n**2)) / denom
        lo, hi = max(0.0, center - half), min(1.0, center + half)
        print(f"    seat {p}: {w}/{n} = {wr:.3f} (95% CI [{lo:.3f}, {hi:.3f}])")
    print(f"  max/min ratio:     {result['max_min_ratio']}  (threshold 1.5)")
    crit = {3: 7.815, 5: 9.488, 7: 11.07}.get(result["chi2_df"], None)
    crit_str = f", critical@5%={crit}" if crit else ""
    sig = result["chi2_vs_uniform"] > (crit or 0)
    print(
        f"  chi2 vs uniform:   {result['chi2_vs_uniform']} "
        f"(df={result['chi2_df']}{crit_str}) -> {'SIGNIFICANT' if sig else 'not significant'}"
    )
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
