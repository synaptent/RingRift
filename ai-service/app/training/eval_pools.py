"""Minimal loader for evaluation state pools (JSONL GameState snapshots).

The pools referenced here are intended primarily for heuristic training and
diagnostics in the multi-board, multi-start regime:

- The canonical ``"v1"`` pools for each board (Square8, Square19, Hexagonal)
  are biased toward **mid- and late-game** positions where heuristic features
  matter most. They are generated via long self-play soaks using
  :mod:`scripts.run_self_play_soak` with mid-game sampling.
- Multi-player variants use **distinct pool_ids** and directory roots so that
  2-player optimisation jobs never accidentally mix 3p/4p positions into their
  evaluation schedule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import argparse
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, quantiles

from app.ai.heuristic_ai import HeuristicAI  # type: ignore
from app.game_engine import GameEngine  # type: ignore
from app.models import (  # type: ignore
    AIConfig,
    BoardType,
    GameState,
    GameStatus,
)
from app.training.env import get_theoretical_max_moves
from app.training.seed_utils import seed_all
from app.training.tier_eval_config import (
    HEURISTIC_TIER_SPECS,
    HeuristicTierSpec,
)
from app.training.tournament import infer_victory_reason


# Canonical mapping from (BoardType, pool_id) to JSONL pool paths.
# 2-player CMA-ES / GA runs should use the ``"v1"`` pools below together with
# eval_mode="multi-start" so that evaluation is driven from fixed mid/late-game
# snapshots rather than only the empty starting position.
#
# Multi-player state pools are kept separate via explicit pool_ids so that
# callers must opt in to 3p/4p evaluation (for example,
# ``square19_3p_pool_v1``), and 2-player training code that passes the default
# pool_id="v1" will never see those states. Hex pools target the canonical
# radius-12 geometry; see ``HEX_DATA_DEPRECATION_NOTICE.md`` for details and
# generation commands.
POOL_PATHS: Dict[Tuple[BoardType, str], str] = {
    # Canonical 2-player evaluation pools (mid/late-game heavy).
    (BoardType.SQUARE8, "v1"): "data/eval_pools/square8/pool_v1.jsonl",
    (BoardType.SQUARE19, "v1"): "data/eval_pools/square19/pool_v1.jsonl",
    (BoardType.HEXAGONAL, "v1"): "data/eval_pools/hex/pool_v1.jsonl",
    # 3-player evaluation pools.
    (BoardType.SQUARE8, "3p_v1"): "data/eval_pools/square8_3p/pool_v1.jsonl",
    (BoardType.SQUARE19, "3p_v1"): "data/eval_pools/square19_3p/pool_v1.jsonl",
    (BoardType.HEXAGONAL, "3p_v1"): "data/eval_pools/hex_3p/pool_v1.jsonl",
    # 4-player evaluation pools.
    (BoardType.SQUARE8, "4p_v1"): "data/eval_pools/square8_4p/pool_v1.jsonl",
    (BoardType.SQUARE19, "4p_v1"): "data/eval_pools/square19_4p/pool_v1.jsonl",
    (BoardType.HEXAGONAL, "4p_v1"): "data/eval_pools/hex_4p/pool_v1.jsonl",
}


@dataclass(frozen=True)
class EvalPoolConfig:
    """Logical configuration for a named evaluation pool.

    This wraps the lower-level (BoardType, pool_id) mapping in
    :data:`POOL_PATHS` and adds ``num_players`` so higher-level tooling
    can refer to pools by a stable string id such as ``"square19_2p_core"``
    or ``"square8_3p_baseline"``.
    """

    name: str
    board_type: BoardType
    num_players: int
    pool_id: str


@dataclass(frozen=True)
class EvalScenario:
    """Single evaluation scenario drawn from an eval pool.

    Scenarios are thin wrappers over :class:`GameState` snapshots with a stable
    identifier and a small metadata bag. The underlying JSONL files remain the
    single source of truth for the actual positions.
    """

    id: str
    board_type: BoardType
    num_players: int
    initial_state: GameState
    metadata: Dict[str, Any]


# Stable registry of named evaluation pools used by tournaments and evaluation
# harnesses. This sits on top of :data:`POOL_PATHS` so that callers never need
# to hard-code (BoardType, pool_id, num_players) triples.
EVAL_POOLS: Dict[str, EvalPoolConfig] = {
    # Square8 2-player canonical core pool (mid/late-game states).
    "square8_2p_core": EvalPoolConfig(
        name="square8_2p_core",
        board_type=BoardType.SQUARE8,
        num_players=2,
        pool_id="v1",
    ),
    # Square19 2-player core pool.
    "square19_2p_core": EvalPoolConfig(
        name="square19_2p_core",
        board_type=BoardType.SQUARE19,
        num_players=2,
        pool_id="v1",
    ),
    # Hexagonal 2-player core pool targeting the radius-12 geometry. The
    # underlying JSONL file is not shipped in the repo yet; see
    # ``HEX_DATA_DEPRECATION_NOTICE.md`` for generation commands.
    "hex_2p_core": EvalPoolConfig(
        name="hex_2p_core",
        board_type=BoardType.HEXAGONAL,
        num_players=2,
        pool_id="v1",
    ),
    # Multiplayer Square8 pools.
    "square8_3p_baseline": EvalPoolConfig(
        name="square8_3p_baseline",
        board_type=BoardType.SQUARE8,
        num_players=3,
        pool_id="3p_v1",
    ),
    "square8_4p_baseline": EvalPoolConfig(
        name="square8_4p_baseline",
        board_type=BoardType.SQUARE8,
        num_players=4,
        pool_id="4p_v1",
    ),
    # Multiplayer Square19 pools.
    "square19_3p_baseline": EvalPoolConfig(
        name="square19_3p_baseline",
        board_type=BoardType.SQUARE19,
        num_players=3,
        pool_id="3p_v1",
    ),
    "square19_4p_baseline": EvalPoolConfig(
        name="square19_4p_baseline",
        board_type=BoardType.SQUARE19,
        num_players=4,
        pool_id="4p_v1",
    ),
    # Hexagonal multiplayer pools for the radius-12 geometry. As with
    # ``hex_2p_core``, the JSONL files must be generated out-of-band
    # before use.
    "hex_3p_baseline": EvalPoolConfig(
        name="hex_3p_baseline",
        board_type=BoardType.HEXAGONAL,
        num_players=3,
        pool_id="3p_v1",
    ),
    "hex_4p_baseline": EvalPoolConfig(
        name="hex_4p_baseline",
        board_type=BoardType.HEXAGONAL,
        num_players=4,
        pool_id="4p_v1",
    ),
}


def get_eval_pool_config(name: str) -> EvalPoolConfig:
    """Return the :class:`EvalPoolConfig` for a named evaluation pool.

    Raises
    ------
    KeyError
        If the pool name is unknown.
    """
    try:
        return EVAL_POOLS[name]
    except KeyError as exc:  # pragma: no cover - defensive
        available = ", ".join(sorted(EVAL_POOLS.keys()))
        raise KeyError(
            f"Unknown evaluation pool {name!r}. Available pools: {available}"
        ) from exc


def list_eval_pools() -> List[EvalPoolConfig]:
    """Return all registered evaluation pools in a deterministic order."""
    return sorted(
        EVAL_POOLS.values(),
        key=lambda cfg: (cfg.board_type.value, cfg.num_players, cfg.name),
    )


def load_eval_pool(
    name: str,
    max_scenarios: Optional[int] = None,
) -> List[EvalScenario]:
    """Load a named evaluation pool as a list of :class:`EvalScenario`.

    This is a thin wrapper over :func:`load_state_pool` that resolves
    board_type / pool_id / num_players from :data:`EVAL_POOLS` and assigns a
    stable per-scenario id of the form ``"{name}:{index}"``.
    """
    cfg = get_eval_pool_config(name)
    states = load_state_pool(
        board_type=cfg.board_type,
        pool_id=cfg.pool_id,
        max_states=max_scenarios,
        num_players=cfg.num_players,
    )
    scenarios: List[EvalScenario] = []
    for index, state in enumerate(states):
        scenario_id = f"{name}:{index}"
        metadata: Dict[str, Any] = {
            "pool_name": name,
            "pool_id": cfg.pool_id,
            "index": index,
        }
        scenarios.append(
            EvalScenario(
                id=scenario_id,
                board_type=cfg.board_type,
                num_players=cfg.num_players,
                initial_state=state,
                metadata=metadata,
            )
        )
    return scenarios


def load_state_pool(
    board_type: BoardType,
    pool_id: str = "v1",
    max_states: Optional[int] = None,
    num_players: Optional[int] = None,
) -> List[GameState]:
    """Load a deterministically ordered pool of GameState records.

    Parameters
    ----------
    board_type:
        BoardType for which to load a state pool.
    pool_id:
        Logical pool identifier (e.g. "v1" or a more specific label such
        as "square19_3p_pool_v1" for multi-player pools).
    max_states:
        Optional maximum number of states to load. If None, load all
        available states. If <= 0, return an empty list.
    num_players:
        Optional number of players to filter by. When provided, this
        helper enforces that every loaded state has exactly this many
        players (mirroring the strict board_type check below).

    Returns
    -------
    List[GameState]
        Parsed GameState instances in file order.

    Raises
    ------
    ValueError
        If no pool path is configured for (board_type, pool_id) or if the
        file contains states with a mismatched board_type.
    FileNotFoundError
        If the configured pool file does not exist.
    """
    key = (board_type, pool_id)
    if key not in POOL_PATHS:
        raise ValueError(
            f"No evaluation state pool configured for "
            f"board_type={board_type!r}, pool_id={pool_id!r}"
        )

    rel_path = POOL_PATHS[key]
    path = os.path.abspath(rel_path)

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"State pool file not found at {path!r} "
            f"for board_type={board_type!r}, pool_id={pool_id!r}"
        )

    states: List[GameState] = []
    if max_states is not None and max_states <= 0:
        return states

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if max_states is not None and len(states) >= max_states:
                break
            line = line.strip()
            if not line:
                continue

            state = GameState.model_validate_json(
                line  # type: ignore[attr-defined]
            )

            if state.board_type != board_type:
                raise ValueError(
                    "State pool contains state with mismatched board_type: "
                    f"expected={board_type!r}, got={state.board_type!r}"
                )

            if num_players is not None:
                actual_players = len(state.players)
                if actual_players != num_players:
                    raise ValueError(
                        "State pool contains state with mismatched "
                        f"num_players: expected={num_players!r}, "
                        f"got={actual_players!r}"
                    )

            states.append(state)

    return states


# ---------------------------------------------------------------------------
# Heuristic tiered evaluation on eval pools (square8-focused)
# ---------------------------------------------------------------------------


def _compute_margins(
    final_state: GameState,
    candidate_player: int,
    opponent_player: int,
) -> Dict[str, float]:
    """Compute simple ring/territory margins for a 2-player game."""
    # Rings eliminated (by causing player id string in eliminated_rings).
    rings_eliminated = final_state.board.eliminated_rings
    cand_rings = int(rings_eliminated.get(str(candidate_player), 0))
    opp_rings = int(rings_eliminated.get(str(opponent_player), 0))
    ring_margin = cand_rings - opp_rings

    # Territory spaces from Player models.
    cand_territory = 0
    opp_territory = 0
    for p in final_state.players:
        if p.player_number == candidate_player:
            cand_territory = int(p.territory_spaces)
        elif p.player_number == opponent_player:
            opp_territory = int(p.territory_spaces)
    territory_margin = cand_territory - opp_territory

    return {
        "ring_margin": float(ring_margin),
        "territory_margin": float(territory_margin),
    }


def run_heuristic_tier_eval(
    tier_spec: HeuristicTierSpec,
    rng_seed: int,
    max_games: Optional[int] = None,
    max_moves_override: Optional[int] = None,
    skip_shadow_contracts: bool = True,
) -> Dict[str, Any]:
    """Evaluate a heuristic profile against a baseline on an eval pool.

    This runner:
    - Loads a fixed pool of mid/late-game GameState snapshots via
      ``load_state_pool``.
    - For each game, samples a starting snapshot and plays a full 2-player
      game using HeuristicAI for both sides (candidate vs baseline).
    - Aggregates win/draw/loss counts, simple ring/territory margins, and
      basic latency statistics for the candidate side.

    The evaluation is square8-focused in this wave, but the implementation is
    board-agnostic and honours the BoardType and num_players from the tier
    specification.
    """
    if tier_spec.num_players != 2:
        raise ValueError(
            "Heuristic tier eval currently supports only 2-player tiers; "
            f"got num_players={tier_spec.num_players!r} "
            f"for tier {tier_spec.id!r}"
        )

    # Global seeding for reproducibility plus a dedicated RNG for sampling.
    seed_all(rng_seed)
    rng = random.Random(rng_seed)

    # Heuristic tier evaluation is a training/diagnostics harness and can
    # involve millions of apply_move calls. Mutator shadow contracts are a
    # valuable development-time tool but add substantial overhead; default to
    # skipping them here to keep eval runs tractable. Callers can opt back in
    # by passing skip_shadow_contracts=False.
    from app.rules.factory import get_rules_engine  # local import for tooling

    get_rules_engine(
        force_new=True,
        skip_shadow_contracts=bool(skip_shadow_contracts),
    )

    # Load the eval pool for this tier.
    states = load_state_pool(
        board_type=tier_spec.board_type,
        pool_id=tier_spec.eval_pool_id,
        num_players=tier_spec.num_players,
    )
    if not states:
        raise ValueError(
            f"Empty eval pool for board_type={tier_spec.board_type!r}, "
            f"pool_id={tier_spec.eval_pool_id!r}"
        )

    games_requested = tier_spec.num_games
    if max_games is not None:
        games_to_play = min(games_requested, max_games)
    else:
        games_to_play = games_requested

    max_moves = get_theoretical_max_moves(
        tier_spec.board_type,
        tier_spec.num_players,
    )
    if max_moves_override is not None and max_moves_override > 0:
        max_moves = min(max_moves, int(max_moves_override))

    wins = 0
    losses = 0
    draws = 0
    games_played = 0

    ring_margins: List[float] = []
    territory_margins: List[float] = []
    candidate_latencies_ms: List[float] = []
    victory_reasons: Dict[str, int] = {}
    total_moves = 0

    for game_index in range(games_to_play):
        games_played += 1

        # Alternate seats to reduce first-move bias.
        candidate_seat = 1 if (game_index % 2 == 0) else 2
        baseline_seat = 2 if candidate_seat == 1 else 1

        # Sample a starting snapshot from the pool and deep-copy it so that
        # per-game mutations do not affect subsequent games.
        base_state = rng.choice(states)
        game_state = base_state.model_copy(deep=True)

        # Per-game RNG seed to decorrelate AIs across games while remaining
        # reproducible under a fixed run seed.
        game_seed = (rng_seed * 1_000_003 + game_index) & 0x7FFFFFFF

        # Configure candidate and baseline heuristic AIs. Difficulty is kept
        # fixed; strength differences are expressed via profile ids.
        candidate_config = AIConfig(
            difficulty=5,
            randomness=0.0,
            think_time=0,
            rngSeed=game_seed,
            heuristic_profile_id=tier_spec.candidate_profile_id,
        )
        baseline_config = AIConfig(
            difficulty=5,
            randomness=0.0,
            think_time=0,
            rngSeed=game_seed,
            heuristic_profile_id=tier_spec.baseline_profile_id,
        )

        candidate_ai = HeuristicAI(candidate_seat, candidate_config)
        baseline_ai = HeuristicAI(baseline_seat, baseline_config)

        done = False
        moves_played = 0
        terminated_by_budget_only = False

        while not done and moves_played < max_moves:
            current_player = game_state.current_player
            if current_player == candidate_seat:
                current_ai = candidate_ai
            else:
                current_ai = baseline_ai

            # Ensure the AI's perspective matches the current player.
            current_ai.player_number = current_player

            t0 = time.perf_counter()
            move = current_ai.select_move(game_state)
            dt_ms = (time.perf_counter() - t0) * 1000.0

            if current_player == candidate_seat:
                candidate_latencies_ms.append(dt_ms)

            if move is None:
                # Per RR-CANON-R076: when get_valid_moves returns empty,
                # check for phase requirements that require bookkeeping moves
                # (NO_*_ACTION, FORCED_ELIMINATION, etc.)
                requirement = GameEngine.get_phase_requirement(
                    game_state,
                    current_player,
                )
                if requirement is not None:
                    # Synthesize the required bookkeeping move and continue
                    move = GameEngine.synthesize_bookkeeping_move(
                        requirement,
                        game_state,
                    )
                else:
                    # True "no moves" case: treat as an immediate loss for the
                    # side to move, mirroring the tournament harness semantics.
                    if current_player == candidate_seat:
                        winner = baseline_seat
                    else:
                        winner = candidate_seat

                    if winner == candidate_seat:
                        wins += 1
                    else:
                        losses += 1

                    # Use a synthetic victory reason to keep stats structured.
                    prev_no_moves = victory_reasons.get("no_moves", 0)
                    victory_reasons["no_moves"] = prev_no_moves + 1

                    # Margin from the current state.
                    margins = _compute_margins(
                        game_state,
                        candidate_seat,
                        baseline_seat,
                    )
                    ring_margins.append(margins["ring_margin"])
                    territory_margins.append(margins["territory_margin"])
                    break

            # Apply move via canonical GameEngine.
            game_state = GameEngine.apply_move(game_state, move)
            moves_played += 1

            if game_state.game_status != GameStatus.ACTIVE:
                done = True

        if not done and moves_played >= max_moves:
            # Max-move cutoff without a terminal engine state.
            terminated_by_budget_only = True
            done = True

        total_moves += moves_played

        # Determine winner and victory reason.
        winner = game_state.winner
        if terminated_by_budget_only and winner is None:
            reason = "max_moves"
        else:
            reason = infer_victory_reason(game_state)
        victory_reasons[reason] = victory_reasons.get(reason, 0) + 1

        if winner == candidate_seat:
            wins += 1
        elif winner == baseline_seat:
            losses += 1
        else:
            draws += 1

        margins = _compute_margins(game_state, candidate_seat, baseline_seat)
        ring_margins.append(margins["ring_margin"])
        territory_margins.append(margins["territory_margin"])

    avg_ring_margin: Optional[float] = None
    avg_territory_margin: Optional[float] = None
    if ring_margins:
        avg_ring_margin = float(mean(ring_margins))
    if territory_margins:
        avg_territory_margin = float(mean(territory_margins))

    mean_latency: Optional[float] = None
    p95_latency: Optional[float] = None
    if candidate_latencies_ms:
        mean_latency = float(mean(candidate_latencies_ms))
        # Use 95th percentile based on 20-quantiles (19/20 ≈ 0.95).
        try:
            p95_latency = float(quantiles(candidate_latencies_ms, n=20)[-1])
        except Exception:
            # Fallback to max when quantiles is unhappy with sample size.
            p95_latency = float(max(candidate_latencies_ms))

    result: Dict[str, Any] = {
        "tier_id": tier_spec.id,
        "tier_name": tier_spec.name,
        "board_type": tier_spec.board_type.value,
        "num_players": tier_spec.num_players,
        "eval_pool_id": tier_spec.eval_pool_id,
        "candidate_profile_id": tier_spec.candidate_profile_id,
        "baseline_profile_id": tier_spec.baseline_profile_id,
        "games_requested": games_requested,
        "games_played": games_played,
        "results": {
            "wins": wins,
            "losses": losses,
            "draws": draws,
        },
        "margins": {
            "ring_margin_mean": avg_ring_margin,
            "territory_margin_mean": avg_territory_margin,
        },
        "latency_ms": {
            "mean": mean_latency,
            "p95": p95_latency,
        },
        "total_moves": total_moves,
        "victory_reasons": dict(victory_reasons),
    }
    return result


def run_all_heuristic_tiers(
    tiers: List[HeuristicTierSpec],
    rng_seed: int,
    max_games: Optional[int] = None,
    max_moves_override: Optional[int] = None,
    tier_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run heuristic tier evaluation across a set of tiers.

    Args:
        tiers:
            List of HeuristicTierSpec entries to evaluate.
        rng_seed:
            Base RNG seed for reproducible runs. Per-tier seeds are derived
            from this value.
        max_games:
            Optional cap on games per tier. When provided, each tier's
            ``num_games`` is capped to this value.
        tier_ids:
            Optional list of tier ``id`` strings to filter the evaluation set.
    """
    if tier_ids is not None:
        wanted = {t_id.strip() for t_id in tier_ids if t_id.strip()}
        tiers = [t for t in tiers if t.id in wanted]

    if not tiers:
        raise ValueError("No heuristic tiers selected for evaluation")

    timestamp = datetime.now(timezone.utc)
    ts_str = timestamp.strftime("%Y%m%dT%H%M%SZ")
    run_id = f"heuristic_tier_eval_{ts_str}"

    tier_results: List[Dict[str, Any]] = []
    for idx, spec in enumerate(tiers):
        tier_seed = (rng_seed * 97_911 + idx * 1_000_003) & 0x7FFFFFFF
        tier_res = run_heuristic_tier_eval(
            tier_spec=spec,
            rng_seed=tier_seed,
            max_games=max_games,
            max_moves_override=max_moves_override,
        )
        tier_results.append(tier_res)

    git_commit = os.getenv("GIT_COMMIT") or os.getenv("RINGRIFT_GIT_COMMIT")

    report: Dict[str, Any] = {
        "run_id": run_id,
        "timestamp": timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rng_seed": rng_seed,
        "git_commit": git_commit,
        "board_types": sorted({t.board_type.value for t in tiers}),
        "tiers": tier_results,
    }
    return report


def _parse_heuristic_cli_args() -> argparse.Namespace:
    """Parse CLI arguments for heuristic tier evaluation on eval pools."""
    parser = argparse.ArgumentParser(
        description=(
            "Run heuristic tiered evaluation on eval pools "
            "(square8-focused initial slice)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Base RNG seed for reproducible evaluations (default: 1).",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help=(
            "Optional cap on games per tier. "
            "When unset, the per-tier num_games value is used."
        ),
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=None,
        help=(
            "Optional cap on moves per game. "
            "When unset, the theoretical per-board maximum is used."
        ),
    )
    parser.add_argument(
        "--tiers",
        type=str,
        default=None,
        help=(
            "Optional comma-separated list of heuristic tier ids to run. "
            f"Available ids: {', '.join(t.id for t in HEURISTIC_TIER_SPECS)}"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    args = _parse_heuristic_cli_args()
    tier_filter: Optional[List[str]] = None
    if args.tiers:
        tier_filter = [t.strip() for t in args.tiers.split(",") if t.strip()]

    report = run_all_heuristic_tiers(
        tiers=HEURISTIC_TIER_SPECS,
        rng_seed=args.seed,
        max_games=args.max_games,
        max_moves_override=args.max_moves,
        tier_ids=tier_filter,
    )

    out_dir = Path("results") / "ai_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"tier_eval_{ts}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote tier eval report to {out_path}")
