#!/usr/bin/env python
"""Policy-equivalence diagnostic harness for HeuristicAI.

This script compares the move choices of a baseline HeuristicAI against one
or more candidate weight profiles on a fixed pool of mid-game GameState
records (typically produced by ``scripts/run_self_play_soak.py``). For each
candidate we compute:

- the fraction of states where the selected move differs from the baseline,
- a simple L2 distance between weight vectors (in HEURISTIC_WEIGHT_KEYS
  order).

The script is intentionally self-contained and does not modify any training
or evaluation logic; it only reads existing state pools and weight JSON
files and writes a JSON summary under ``logs/diagnostics/``.

Typical usage includes:

- Pointing ``--weights-dir`` at a directory of CMA-ES / GA outputs
  (for example, ``logs/cmaes/runs/<run_id>/``) to compare several final
  candidates against the baseline.
- Reusing the same state pools that power multi-start evaluation so that
  strength and policy-difference measurements are aligned.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# Ensure app.* imports resolve when run from the ai-service root.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from app.models import (  # type: ignore  # noqa: E402
    AIConfig,
    BoardType,
    GameState,
    Move,
)
from app.ai.heuristic_ai import HeuristicAI  # type: ignore  # noqa: E402
from app.ai.heuristic_weights import (  # type: ignore  # noqa: E402
    BASE_V1_BALANCED_WEIGHTS,
    HEURISTIC_WEIGHT_KEYS,
    HeuristicWeights,
)


DEFAULT_STATE_POOL = os.path.join("data", "eval_pools", "square8", "pool_v1.jsonl")
DEFAULT_MAX_STATES = 300
DEFAULT_OUTPUT_DIR = os.path.join("logs", "diagnostics")


@dataclass
class CandidateProfile:
    """In-memory representation of a candidate heuristic weight profile."""

    id: str
    weights: HeuristicWeights
    meta: Dict[str, Any]


def load_state_pool(path: str, max_states: int) -> List[GameState]:
    """Load up to max_states GameState records from a JSONL pool.

    Each non-empty line is parsed via GameState.model_validate_json(...) and
    we assert that all states use BoardType.SQUARE8. Ordering is preserved
    (first N lines) for determinism.
    """
    states: List[GameState] = []
    if max_states <= 0:
        return states
    if not os.path.isfile(path):
        raise FileNotFoundError(f"State pool file not found: {path!r}")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if len(states) >= max_states:
                break
            line = line.strip()
            if not line:
                continue
            state = GameState.model_validate_json(line)  # type: ignore[attr-defined]
            if state.board_type != BoardType.SQUARE8:
                raise ValueError(
                    "State pool contains non-square8 state: "
                    f"{state.board_type!r}. This diagnostic currently "
                    "supports BoardType.SQUARE8 only."
                )
            states.append(state)
    return states


def _validate_weight_keys(
    weights: Mapping[str, float],
    baseline_keys: Iterable[str],
    context: str,
) -> None:
    """Ensure that weights has exactly the same key set as the baseline."""
    expected = set(baseline_keys)
    actual = set(weights.keys())
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"{context} weight keys mismatch baseline schema: "
            f"missing={missing}, extra={extra}"
        )


def _load_weights_payload(path: str) -> Tuple[HeuristicWeights, Dict[str, Any]]:
    """Load a weight dict and associated meta from a JSON file with 'weights'."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Weights file not found: {path!r}")
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    weights_obj = payload.get("weights")
    if not isinstance(weights_obj, dict):
        raise ValueError(
            f"Weights file {path!r} is missing a 'weights' object or it is not a dict"
        )
    weights: HeuristicWeights = {str(k): float(v) for k, v in weights_obj.items()}
    meta = payload.get("meta") or {}
    meta = dict(meta)
    meta.setdefault("path", path)
    return weights, meta


def load_candidate_profiles_from_dir(
    dir_path: str,
    pattern: str,
    baseline_keys: Iterable[str],
) -> List[CandidateProfile]:
    """Load all candidate profiles from JSON files under dir_path."""
    import glob

    if not os.path.isdir(dir_path):
        raise FileNotFoundError(
            f"Candidates dir does not exist or is not a directory: {dir_path!r}"
        )
    candidates: List[CandidateProfile] = []
    paths = sorted(glob.glob(os.path.join(dir_path, pattern)))
    for path in paths:
        if not os.path.isfile(path):
            continue
        weights, meta = _load_weights_payload(path)
        _validate_weight_keys(weights, baseline_keys, context=f"Candidate {path!r}")
        profile_id = (
            str(meta.get("id"))
            if "id" in meta
            else os.path.splitext(os.path.basename(path))[0]
        )
        meta.setdefault("path", path)
        candidates.append(
            CandidateProfile(
                id=profile_id,
                weights=weights,
                meta=meta,
            )
        )
    return candidates


def load_candidate_profiles_from_files(
    paths: Sequence[str],
    baseline_keys: Iterable[str],
) -> List[CandidateProfile]:
    """Load candidate profiles from an explicit list of JSON files."""
    candidates: List[CandidateProfile] = []
    for path in paths:
        weights, meta = _load_weights_payload(path)
        _validate_weight_keys(weights, baseline_keys, context=f"Candidate {path!r}")
        profile_id = (
            str(meta.get("id"))
            if "id" in meta
            else os.path.splitext(os.path.basename(path))[0]
        )
        meta.setdefault("path", path)
        candidates.append(
            CandidateProfile(
                id=profile_id,
                weights=weights,
                meta=meta,
            )
        )
    return candidates


def _move_signature(move: Optional[Move]) -> Optional[Tuple[str, Optional[str], Optional[str]]]:
    """Return a lightweight, comparable signature for a Move.

    The signature consists of (type.value, from_pos_key, to_pos_key).
    """
    if move is None:
        return None
    move_type = move.type.value if hasattr(move.type, "value") else str(move.type)
    from_key: Optional[str]
    if move.from_pos is not None:
        from_key = move.from_pos.to_key()
    else:
        from_key = None
    to_key: Optional[str]
    if move.to is not None:  # type: ignore[truthy-function]
        to_key = move.to.to_key()
    else:
        to_key = None
    return (str(move_type), from_key, to_key)


def compare_moves_on_states(
    baseline_weights: HeuristicWeights,
    candidate_weights: HeuristicWeights,
    states: Sequence[GameState],
) -> Dict[str, Any]:
    """Compare baseline vs candidate select_move decisions on states.

    The AIs are configured with think_time=0 and randomness=0.0 so that
    select_move is deterministic for a given state. For each state we set
    both AIs' player_number to state.current_player before querying moves.
    """
    # Instantiate baseline and candidate AIs once and override their weights.
    baseline_ai = HeuristicAI(
        1,
        AIConfig(
            difficulty=5,
            think_time=0,
            randomness=0.0,
            rngSeed=None,
            heuristic_profile_id=None,
        ),
    )
    candidate_ai = HeuristicAI(
        1,
        AIConfig(
            difficulty=5,
            think_time=0,
            randomness=0.0,
            rngSeed=None,
            heuristic_profile_id=None,
        ),
    )
    for name, value in baseline_weights.items():
        setattr(baseline_ai, name, float(value))
    for name, value in candidate_weights.items():
        setattr(candidate_ai, name, float(value))

    same_moves = 0
    different_moves = 0

    for state in states:
        # Ensure both AIs view the same side to move.
        current_player = state.current_player
        baseline_ai.player_number = current_player
        candidate_ai.player_number = current_player

        baseline_move = baseline_ai.select_move(state)
        candidate_move = candidate_ai.select_move(state)

        sig_baseline = _move_signature(baseline_move)
        sig_candidate = _move_signature(candidate_move)

        if sig_baseline is None and sig_candidate is None:
            # Both decline to move; treat as equivalent policy.
            same_moves += 1
            continue

        if sig_baseline == sig_candidate:
            same_moves += 1
        else:
            different_moves += 1

    total = same_moves + different_moves
    difference_rate = (different_moves / total) if total > 0 else 0.0

    # L2 distance in the canonical HEURISTIC_WEIGHT_KEYS order.
    sq_sum = 0.0
    for key in HEURISTIC_WEIGHT_KEYS:
        bw = float(baseline_weights.get(key, 0.0))
        cw = float(candidate_weights.get(key, 0.0))
        diff = cw - bw
        sq_sum += diff * diff
    weight_l2 = math.sqrt(sq_sum)

    return {
        "same_moves": same_moves,
        "different_moves": different_moves,
        "total_states": total,
        "difference_rate": difference_rate,
        "weight_l2": weight_l2,
    }


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare baseline HeuristicAI policy against candidate weight "
            "profiles on a fixed state pool."
        )
    )
    parser.add_argument(
        "--state-pool",
        type=str,
        default=DEFAULT_STATE_POOL,
        help=(
            "Path to JSONL file containing GameState records "
            f"(default: {DEFAULT_STATE_POOL})."
        ),
    )
    parser.add_argument(
        "--max-states",
        type=int,
        default=DEFAULT_MAX_STATES,
        help=(
            "Maximum number of states to sample from the pool "
            f"(default: {DEFAULT_MAX_STATES})."
        ),
    )
    parser.add_argument(
        "--baseline-weights",
        type=str,
        default="",
        help=(
            "Optional path to baseline weights JSON file with a 'weights' "
            "object. If omitted, BASE_V1_BALANCED_WEIGHTS is used."
        ),
    )
    parser.add_argument(
        "--candidates-dir",
        type=str,
        default="",
        help=(
            "Optional directory containing candidate weight JSON files "
            "(schema: {'weights': {...}})."
        ),
    )
    parser.add_argument(
        "--weights-dir",
        type=str,
        default="",
        help=(
            "Alias for --candidates-dir; if provided, all *.json files in "
            "this directory are treated as candidate weight profiles."
        ),
    )
    parser.add_argument(
        "--candidate-glob",
        type=str,
        default="*.json",
        help=("Glob pattern for files inside --candidates-dir (default: *.json)."),
    )
    parser.add_argument(
        "--candidate-weights",
        type=str,
        action="append",
        default=None,
        help=(
            "Optional explicit candidate weights JSON file; may be "
            "provided multiple times."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help=(
            "Optional explicit output path for the JSON summary. If "
            "omitted, a timestamped file is created under "
            f"{DEFAULT_OUTPUT_DIR}."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help=(
            "Seed recorded in the summary for reproducibility. The current "
            "implementation uses deterministic first-N sampling, so the "
            "seed does not affect behaviour yet."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)

    baseline_keys = list(BASE_V1_BALANCED_WEIGHTS.keys())

    # Load state pool.
    states = load_state_pool(args.state_pool, args.max_states)

    # Resolve baseline weights and metadata.
    if args.baseline_weights:
        baseline_weights, baseline_meta = _load_weights_payload(args.baseline_weights)
        _validate_weight_keys(
            baseline_weights,
            baseline_keys,
            context=f"Baseline {args.baseline_weights!r}",
        )
        baseline_id = os.path.splitext(os.path.basename(args.baseline_weights))[0]
        baseline_meta.setdefault("source", args.baseline_weights)
    else:
        baseline_weights = dict(BASE_V1_BALANCED_WEIGHTS)
        baseline_meta = {"source": "BASE_V1_BALANCED_WEIGHTS"}
        baseline_id = "baseline_v1_balanced"

    # Load candidate profiles from directory and/or explicit files.
    candidates: List[CandidateProfile] = []
    if args.candidates_dir:
        dir_candidates = load_candidate_profiles_from_dir(
            args.candidates_dir,
            args.candidate_glob,
            baseline_keys,
        )
        candidates.extend(dir_candidates)
    if args.weights_dir:
        dir_candidates = load_candidate_profiles_from_dir(
            args.weights_dir,
            args.candidate_glob,
            baseline_keys,
        )
        candidates.extend(dir_candidates)
    if args.candidate_weights:
        file_candidates = load_candidate_profiles_from_files(
            args.candidate_weights,
            baseline_keys,
        )
        candidates.extend(file_candidates)

    if not candidates:
        raise SystemExit(
            "No candidates found. Provide --candidates-dir and/or "
            "--candidate-weights."
        )

    created_at = datetime.utcnow().isoformat() + "Z"

    print("=== Policy-equivalence diagnostic ===")
    print(f"State pool: {args.state_pool} ({len(states)} states loaded)")
    print(f"Baseline:  {baseline_id}")
    print(f"Candidates: {len(candidates)}")
    print()

    results: List[Dict[str, Any]] = []
    for candidate in candidates:
        stats = compare_moves_on_states(
            baseline_weights=baseline_weights,
            candidate_weights=candidate.weights,
            states=states,
        )
        entry = {
            "id": candidate.id,
            "meta": candidate.meta,
            "stats": stats,
        }
        results.append(entry)

    summary: Dict[str, Any] = {
        "meta": {
            "created_at": created_at,
            "seed": args.seed,
            "state_pool": args.state_pool,
            "max_states": args.max_states,
            "baseline_id": baseline_id,
            "num_candidates": len(candidates),
        },
        "baseline": {
            "id": baseline_id,
            "meta": baseline_meta,
        },
        "candidates": results,
    }

    # Resolve output path.
    if args.output:
        output_path = args.output
    else:
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(
            DEFAULT_OUTPUT_DIR,
            f"policy_equivalence_{ts}.json",
        )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(
        "Diagnostic complete. Results written to "
        f"{output_path}"
    )


if __name__ == "__main__":  # pragma: no cover
    main()