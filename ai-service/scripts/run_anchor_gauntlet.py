#!/usr/bin/env python3
"""Run fixed-checkpoint anchor gauntlets for RingRift Elo calibration.

This script is intentionally separate from the promotion-ladder loop. The
minimal loop estimates progress from candidate-vs-current-best promotion
steps; this gauntlet replays fixed checkpoints against named anchors so public
claims can distinguish ladder Elo from calibrated pool evidence.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

if not os.environ.get("RINGRIFT_DISABLE_TORCH_COMPILE"):
    os.environ["RINGRIFT_DISABLE_TORCH_COMPILE"] = "1"

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ai.heuristic_ai import HeuristicAI
from app.ai.random_ai import RandomAI
from app.models import AIConfig, BoardType, GameStatus
from scripts import minimal_alphazero_loop as loop

logger = logging.getLogger("anchor_gauntlet")

BOARD_TYPE_MAP = {
    "hex8": BoardType.HEX8,
    "hexagonal": BoardType.HEXAGONAL,
    "square8": BoardType.SQUARE8,
    "square19": BoardType.SQUARE19,
}


@dataclass(frozen=True)
class Participant:
    """One named gauntlet participant."""

    name: str
    kind: str
    path: str | None = None


@dataclass(frozen=True)
class PairSpec:
    """A directed pair where ``a`` is scored against ``b``."""

    a: str
    b: str

    @property
    def key(self) -> str:
        return f"{self.a}__vs__{self.b}"


def _score_from_counts(a_wins: int, b_wins: int, draws: int) -> float:
    games = a_wins + b_wins + draws
    if games <= 0:
        return 0.5
    return (a_wins + 0.5 * draws) / games


def _elo_diff_from_score(score: float, *, clamp: float = 0.01) -> float:
    """Convert pairwise score into Elo difference.

    Scores at exactly 0 or 1 are clamped for finite reporting; callers should
    preserve the raw score alongside this value.
    """

    bounded = min(max(float(score), clamp), 1.0 - clamp)
    return 400.0 * math.log10(bounded / (1.0 - bounded))


def _wilson_interval(wins: float, games: int, *, z: float = 1.96) -> tuple[float, float]:
    """Return a Wilson interval for a binomial-style score."""

    if games <= 0:
        return (0.0, 1.0)
    p = float(wins) / games
    denom = 1.0 + z * z / games
    center = (p + z * z / (2.0 * games)) / denom
    margin = (
        z
        * math.sqrt((p * (1.0 - p) / games) + (z * z / (4.0 * games * games)))
        / denom
    )
    return (max(0.0, center - margin), min(1.0, center + margin))


def _parse_name_value(raw: str, *, flag: str) -> tuple[str, str]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(f"{flag} must be NAME=VALUE, got {raw!r}")
    name, value = raw.split("=", 1)
    name = name.strip()
    value = value.strip()
    if not name or not value:
        raise argparse.ArgumentTypeError(f"{flag} must be NAME=VALUE, got {raw!r}")
    return name, value


def _parse_model(raw: str) -> Participant:
    name, value = _parse_name_value(raw, flag="--model")
    return Participant(name=name, kind="model", path=value)


def _parse_baseline(raw: str) -> Participant:
    name, value = _parse_name_value(raw, flag="--baseline")
    kind = value.lower()
    if kind not in {"random", "heuristic"}:
        raise argparse.ArgumentTypeError(
            "--baseline kind must be one of: random, heuristic"
        )
    return Participant(name=name, kind=kind)


def _parse_pair(raw: str) -> PairSpec:
    if ":" not in raw:
        raise argparse.ArgumentTypeError(f"--pair must be A:B, got {raw!r}")
    a, b = raw.split(":", 1)
    a = a.strip()
    b = b.strip()
    if not a or not b or a == b:
        raise argparse.ArgumentTypeError(f"--pair must name two participants, got {raw!r}")
    return PairSpec(a=a, b=b)


def _parse_fixed_rating(raw: str) -> tuple[str, float]:
    name, value = _parse_name_value(raw, flag="--fixed-rating")
    try:
        return name, float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--fixed-rating value must be numeric, got {raw!r}"
        ) from exc


def _validate_participants(participants: list[Participant]) -> dict[str, Participant]:
    by_name: dict[str, Participant] = {}
    for participant in participants:
        if participant.name in by_name:
            raise ValueError(f"duplicate participant name: {participant.name}")
        by_name[participant.name] = participant
    if len(by_name) < 2:
        raise ValueError("anchor gauntlet requires at least two participants")
    return by_name


def _build_pairs(
    participants: dict[str, Participant],
    requested_pairs: list[PairSpec],
) -> list[PairSpec]:
    if requested_pairs:
        pairs = requested_pairs
    else:
        pairs = [PairSpec(a, b) for a, b in combinations(participants, 2)]

    for pair in pairs:
        if pair.a not in participants:
            raise ValueError(f"unknown participant in --pair: {pair.a}")
        if pair.b not in participants:
            raise ValueError(f"unknown participant in --pair: {pair.b}")
    return pairs


def _configure_minimal_loop(
    *,
    board_type: str,
    num_players: int,
    model_version: str,
    feature_version: int,
) -> None:
    if board_type not in BOARD_TYPE_MAP:
        valid = ", ".join(sorted(BOARD_TYPE_MAP))
        raise ValueError(f"unsupported board type {board_type!r}; expected one of {valid}")
    loop.BOARD_TYPE = board_type
    loop.BOARD_ENUM = BOARD_TYPE_MAP[board_type]
    loop.NUM_PLAYERS = int(num_players)
    loop.MODEL_VERSION = model_version
    loop.FEATURE_VERSION = int(feature_version)
    loop.MAX_MOVES = int(
        loop.get_theoretical_max_moves(loop.BOARD_ENUM, loop.NUM_PLAYERS) * 1.5
    )


def _make_ai(
    participant: Participant,
    player: int,
    *,
    budget: int,
    seed: int,
):
    if participant.kind == "model":
        if not participant.path:
            raise ValueError(f"model participant {participant.name!r} has no path")
        return loop._make_ai(player, participant.path, budget)

    if participant.kind == "random":
        return RandomAI(
            player,
            AIConfig(
                difficulty=1,
                randomness=1.0,
                rng_seed=seed,
                use_neural_net=False,
            ),
        )

    if participant.kind == "heuristic":
        return HeuristicAI(
            player,
            AIConfig(
                difficulty=2,
                randomness=0.0,
                rng_seed=seed,
                use_neural_net=False,
            ),
        )

    raise ValueError(f"unsupported participant kind: {participant.kind}")


def _play_pair_game(
    env,
    ai_cache: dict[tuple[str, int], Any],
    a: Participant,
    b: Participant,
    *,
    game_index: int,
    budget: int,
    seed_base: int,
) -> tuple[int | None, int]:
    num_players = env.num_players if hasattr(env, "num_players") else loop.NUM_PLAYERS
    a_player = (game_index % num_players) + 1
    gseed = (seed_base + game_index * 7919) & 0xFFFFFFFF

    ais = {}
    for player in range(1, num_players + 1):
        participant = a if player == a_player else b
        ai_seed = (gseed + player * 97_911) & 0xFFFFFFFF
        cache_key = (participant.name, player)
        if cache_key not in ai_cache:
            ai_cache[cache_key] = _make_ai(
                participant,
                player,
                budget=budget,
                seed=ai_seed,
            )
        ais[player] = ai_cache[cache_key]
        reset = getattr(ais[player], "reset_for_new_game", None)
        if callable(reset):
            reset(rng_seed=ai_seed)

    state = env.reset(seed=gseed)
    move_count = 0
    fallback_rng = random.Random((gseed ^ 0xA53A_9E11) & 0xFFFFFFFF)
    while state.game_status == GameStatus.ACTIVE and move_count < loop.MAX_MOVES:
        current_player = state.current_player
        ai = ais.get(current_player)
        if ai is None:
            break
        ai.player_number = current_player
        legal = env.legal_moves()
        if not legal:
            break
        selected = ai.select_move(state)
        if selected is None:
            break
        if selected not in legal:
            selected = legal[fallback_rng.randrange(len(legal))]
        state, _, done, _ = env.step(selected)
        move_count += 1
        if done:
            break

    winner = state.winner if state.game_status == GameStatus.COMPLETED else None
    return winner, a_player


def run_pair(
    a: Participant,
    b: Participant,
    *,
    games: int,
    budget: int,
    seed_base: int,
) -> dict[str, Any]:
    start = time.time()
    a_wins = 0
    b_wins = 0
    draws = 0
    seat_outcomes: dict[str, dict[str, int]] = {}
    env = loop._make_env()
    ai_cache: dict[tuple[str, int], Any] = {}

    for game_index in range(games):
        winner, a_player = _play_pair_game(
            env,
            ai_cache,
            a,
            b,
            game_index=game_index,
            budget=budget,
            seed_base=seed_base,
        )
        seat_key = str(a_player)
        seat_outcomes.setdefault(seat_key, {"a_wins": 0, "b_wins": 0, "draws": 0})
        if winner is None:
            draws += 1
            seat_outcomes[seat_key]["draws"] += 1
        elif winner == a_player:
            a_wins += 1
            seat_outcomes[seat_key]["a_wins"] += 1
        else:
            b_wins += 1
            seat_outcomes[seat_key]["b_wins"] += 1

        if (game_index + 1) % max(1, games // 5) == 0:
            logger.info(
                "%s vs %s %s/%s: %s-%s draws=%s",
                a.name,
                b.name,
                game_index + 1,
                games,
                a_wins,
                b_wins,
                draws,
            )

    score = _score_from_counts(a_wins, b_wins, draws)
    ci_low, ci_high = _wilson_interval(a_wins + 0.5 * draws, games)
    return {
        "a": a.name,
        "b": b.name,
        "games_played": games,
        "a_wins": a_wins,
        "b_wins": b_wins,
        "draws": draws,
        "score": score,
        "score_wilson_95": [ci_low, ci_high],
        "elo_diff_a_minus_b": _elo_diff_from_score(score),
        "elo_diff_a_minus_b_ci_95": [
            _elo_diff_from_score(ci_low),
            _elo_diff_from_score(ci_high),
        ],
        "seat_outcomes": seat_outcomes,
        "elapsed_s": time.time() - start,
    }


def _solve_anchored_ratings(
    participant_names: list[str],
    pair_results: list[dict[str, Any]],
    fixed_ratings: dict[str, float],
) -> dict[str, float]:
    """Estimate ratings from pairwise Elo deltas with fixed anchors.

    This is a lightweight least-squares bridge for operational calibration. It
    is not a substitute for a full Bradley-Terry/BayesElo model when a large
    rating pool exists, but it gives reproducible anchored estimates directly
    from mirrored gauntlet JSON.
    """

    if not fixed_ratings:
        return {}

    try:
        import numpy as np
    except ImportError:
        logger.warning("numpy unavailable; skipping anchored rating solve")
        return {}

    index = {name: i for i, name in enumerate(participant_names)}
    rows: list[list[float]] = []
    targets: list[float] = []
    weights: list[float] = []

    for result in pair_results:
        a = result["a"]
        b = result["b"]
        if a not in index or b not in index:
            continue
        row = [0.0] * len(index)
        row[index[a]] = 1.0
        row[index[b]] = -1.0
        rows.append(row)
        targets.append(float(result["elo_diff_a_minus_b"]))
        weights.append(max(1.0, math.sqrt(float(result.get("games_played", 1)))))

    for name, rating in fixed_ratings.items():
        if name not in index:
            raise ValueError(f"fixed rating names unknown participant: {name}")
        row = [0.0] * len(index)
        row[index[name]] = 1.0
        rows.append(row)
        targets.append(float(rating))
        weights.append(100.0)

    if not rows:
        return {}

    matrix = np.asarray(rows, dtype=float)
    target = np.asarray(targets, dtype=float)
    weight = np.asarray(weights, dtype=float)
    solution, *_ = np.linalg.lstsq(matrix * weight[:, None], target * weight, rcond=None)
    return {name: round(float(solution[index[name]]), 1) for name in participant_names}


def _load_existing_results(output_path: Path) -> dict[str, Any]:
    if not output_path.exists():
        return {}
    try:
        return json.loads(output_path.read_text())
    except (OSError, json.JSONDecodeError):
        logger.warning("Ignoring unreadable resume file: %s", output_path)
        return {}


def _write_output(output_path: Path, payload: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(output_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run fixed-checkpoint anchor gauntlets for Elo calibration."
    )
    parser.add_argument("--board-type", default="hex8", choices=sorted(BOARD_TYPE_MAP))
    parser.add_argument("--num-players", type=int, default=2)
    parser.add_argument("--model-version", default="v5-heavy")
    parser.add_argument("--feature-version", type=int, default=3)
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        type=_parse_model,
        metavar="NAME=PATH",
        help="Add a neural checkpoint participant.",
    )
    parser.add_argument(
        "--baseline",
        action="append",
        default=[],
        type=_parse_baseline,
        metavar="NAME=random|heuristic",
        help="Add a fixed baseline participant.",
    )
    parser.add_argument(
        "--pair",
        action="append",
        default=[],
        type=_parse_pair,
        metavar="A:B",
        help="Evaluate only this directed pair. Repeatable. Defaults to all pairs.",
    )
    parser.add_argument("--games", type=int, default=400)
    parser.add_argument("--budget", type=int, default=128)
    parser.add_argument("--seed-base", type=int, default=42_000)
    parser.add_argument(
        "--fixed-rating",
        action="append",
        default=[],
        type=_parse_fixed_rating,
        metavar="NAME=ELO",
        help="Pin a participant rating for anchored least-squares calibration.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = build_arg_parser().parse_args(argv)

    if args.games <= 0:
        raise SystemExit("--games must be positive")
    if args.num_players < 2:
        raise SystemExit("--num-players must be at least 2")

    participants = _validate_participants([*args.model, *args.baseline])
    pairs = _build_pairs(participants, args.pair)
    fixed_ratings = dict(args.fixed_rating)

    _configure_minimal_loop(
        board_type=args.board_type,
        num_players=args.num_players,
        model_version=args.model_version,
        feature_version=args.feature_version,
    )

    payload: dict[str, Any] = _load_existing_results(args.output) if args.resume else {}
    completed = dict(payload.get("pair_results_by_key", {}))
    payload.update(
        {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "purpose": (
                "Fixed-checkpoint anchor gauntlet for calibrating RingRift "
                "promotion-ladder Elo against stable anchors."
            ),
            "rating_warning": (
                "Gauntlet ratings are calibration estimates for this fixed pool. "
                "Keep them separate from promotion-ladder Elo."
            ),
            "config": {
                "board_type": args.board_type,
                "num_players": args.num_players,
                "model_version": args.model_version,
                "feature_version": args.feature_version,
                "games_per_pair": args.games,
                "budget": args.budget,
                "seed_base": args.seed_base,
            },
            "participants": [asdict(p) for p in participants.values()],
            "pairs": [{**asdict(pair), "key": pair.key} for pair in pairs],
            "fixed_ratings": fixed_ratings,
            "pair_results_by_key": completed,
        }
    )

    if args.dry_run:
        _write_output(args.output, payload)
        logger.info("Wrote dry-run gauntlet plan to %s", args.output)
        return 0

    for pair in pairs:
        if args.resume:
            existing = completed.get(pair.key)
            if existing and int(existing.get("games_played", 0)) >= args.games:
                logger.info("Skipping completed pair %s", pair.key)
                continue
        logger.info("Running pair %s: %s vs %s", pair.key, pair.a, pair.b)
        completed[pair.key] = run_pair(
            participants[pair.a],
            participants[pair.b],
            games=args.games,
            budget=args.budget,
            seed_base=args.seed_base,
        )
        payload["pair_results_by_key"] = completed
        payload["pair_results"] = list(completed.values())
        payload["calibrated_ratings"] = _solve_anchored_ratings(
            list(participants),
            list(completed.values()),
            fixed_ratings,
        )
        _write_output(args.output, payload)

    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    payload["pair_results"] = list(completed.values())
    payload["calibrated_ratings"] = _solve_anchored_ratings(
        list(participants),
        list(completed.values()),
        fixed_ratings,
    )
    _write_output(args.output, payload)
    logger.info("Wrote gauntlet results to %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
