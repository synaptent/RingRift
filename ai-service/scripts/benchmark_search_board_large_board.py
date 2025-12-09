#!/usr/bin/env python3
"""
Lightweight benchmark for large-board canonical vs SearchBoard (MutableGameState)
replay performance.

This script compares:

- Canonical replay: GameEngine.apply_move(..., trace_mode=True)
- SearchBoard replay: MutableGameState.make_move(...) + to_immutable()

on representative large-board parity fixtures (Square19 and Hex). It is intended
for manual, ad-hoc use and MUST NOT be wired into CI.

Usage (from ai-service root):

    python scripts/benchmark_search_board_large_board.py

You can adjust the number of iterations via the RINGRIFT_SB_BENCH_ITERS
environment variable (default: 50).
"""

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

# Ensure project root is on sys.path for direct script execution
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)

from app.db import GameReplayDB  # noqa: E402
from app.game_engine import GameEngine  # noqa: E402
from app.models import BoardType, GameState, Move  # noqa: E402
from app.rules.mutable_state import MutableGameState  # noqa: E402


AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]
PARITY_FIXTURES_DIR = AI_SERVICE_ROOT / "parity_fixtures"


# Representative large-board fixtures (same set used in
# test_search_board_parity.py for structural parity tests).
LARGE_BOARD_FIXTURES: List[str] = [
    ("selfplay_square19_2p__" "02aa8d91-47aa-4d3e-a506-cdd493bda33a__k118.json"),
    ("selfplay_hexagonal_2p__" "41c7c746-af99-48cb-8887-435b03b5eac7__k8.json"),
    ("selfplay_hexagonal_3p__" "1f7a10cc-e41c-48eb-80a9-8c4bfde8d3d0__k87.json"),
]


@dataclass
class ReplayBenchmarkResult:
    """Simple timing summary for a single replay benchmark."""

    fixture_name: str
    board_type: str
    mode: str  # "canonical" or "search_board"
    iterations: int
    total_time_sec: float

    @property
    def avg_ms(self) -> float:
        if self.iterations == 0:
            return 0.0
        return (self.total_time_sec / self.iterations) * 1000.0


def _load_fixture_payload(fixture_name: str) -> dict:
    """Load a parity fixture JSON payload.

    Returns the JSON object loaded from the fixture file. If the
    fixture (or its referenced DB) is missing, the caller should skip
    the benchmark for this fixture.
    """
    import json

    path = PARITY_FIXTURES_DIR / fixture_name
    if not path.exists():
        raise FileNotFoundError(f"Fixture not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    db_path = payload.get("db_path")
    game_id = payload.get("game_id")
    if not isinstance(db_path, str) or not isinstance(game_id, str):
        raise ValueError(f"Fixture {fixture_name} missing db_path/game_id")

    if "canonical_move_index" not in payload:
        # Fall back to diverged_at - 1 when only diverged_at is present.
        if "diverged_at" in payload:
            payload["canonical_move_index"] = payload["diverged_at"] - 1
        else:
            raise ValueError(f"Fixture {fixture_name} missing canonical_move_index " "and diverged_at")

    return payload


def _load_sequence_from_fixture(
    fixture_name: str,
) -> Tuple[GameState, List[Move], BoardType]:
    """Load initial state and move slice from a parity fixture."""
    payload = _load_fixture_payload(fixture_name)
    db_path = payload["db_path"]
    game_id = payload["game_id"]
    move_index = int(payload["canonical_move_index"])

    if not Path(db_path).exists():
        raise FileNotFoundError(f"GameReplayDB not found: {db_path}")

    db = GameReplayDB(db_path)
    initial_state = db.get_initial_state(game_id)
    if initial_state is None:
        raise RuntimeError(f"Initial state not found for game_id={game_id}")

    moves = db.get_moves(game_id, start=0, end=move_index + 1)
    return initial_state, moves, initial_state.board.type


def _benchmark_canonical_replay(
    initial_state: GameState,
    moves: List[Move],
    iterations: int,
) -> float:
    """Replay sequence via GameEngine.apply_move, return total time in seconds."""
    total = 0.0
    for _ in range(iterations):
        state = initial_state
        start = time.perf_counter()
        for mv in moves:
            state = GameEngine.apply_move(state, mv, trace_mode=True)
        total += time.perf_counter() - start
    return total


def _benchmark_search_board_replay(
    initial_state: GameState,
    moves: List[Move],
    iterations: int,
) -> float:
    """Replay sequence via MutableGameState.make_move, return total time."""
    total = 0.0
    for _ in range(iterations):
        mutable = MutableGameState.from_immutable(initial_state)
        start = time.perf_counter()
        for mv in moves:
            mutable.make_move(mv)
        # Force materialization to avoid the optimizer eliding work
        _ = mutable.to_immutable()
        total += time.perf_counter() - start
    return total


def run_benchmark_for_fixture(
    fixture_name: str,
    iterations: int,
) -> List[ReplayBenchmarkResult]:
    """Run canonical vs SearchBoard replay benchmarks for one fixture."""
    try:
        initial_state, moves, board_type = _load_sequence_from_fixture(fixture_name)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"  Skipping {fixture_name}: {exc}")
        return []

    print(f"  Fixture: {fixture_name} " f"(board_type={board_type.value}, moves={len(moves)})")

    # Canonical
    canon_total = _benchmark_canonical_replay(initial_state, moves, iterations)
    canon_res = ReplayBenchmarkResult(
        fixture_name=fixture_name,
        board_type=board_type.value,
        mode="canonical",
        iterations=iterations,
        total_time_sec=canon_total,
    )

    # SearchBoard / MutableGameState
    sb_total = _benchmark_search_board_replay(initial_state, moves, iterations)
    sb_res = ReplayBenchmarkResult(
        fixture_name=fixture_name,
        board_type=board_type.value,
        mode="search_board",
        iterations=iterations,
        total_time_sec=sb_total,
    )

    return [canon_res, sb_res]


def main() -> int:
    """Entry point for the large-board SearchBoard replay benchmark."""
    default_iters = 50
    env_val = os.getenv("RINGRIFT_SB_BENCH_ITERS")
    try:
        iterations = int(env_val) if env_val is not None else default_iters
    except ValueError:
        iterations = default_iters

    print("=" * 72)
    print("Large-Board SearchBoard Replay Benchmark")
    print("=" * 72)
    print(f"Iterations per mode: {iterations}")
    print("NOTE: This script is for manual benchmarking only and is NOT used in CI.")
    print()

    all_results: List[ReplayBenchmarkResult] = []

    for fixture_name in LARGE_BOARD_FIXTURES:
        print("-" * 72)
        results = run_benchmark_for_fixture(fixture_name, iterations)
        if not results:
            continue

        canon = next(r for r in results if r.mode == "canonical")
        sb = next(r for r in results if r.mode == "search_board")

        all_results.extend(results)

        if canon.total_time_sec > 0.0 and sb.total_time_sec > 0.0:
            speedup = canon.total_time_sec / sb.total_time_sec
        else:
            speedup = 0.0

        print(f"    Canonical:   {canon.avg_ms:.2f} ms / replay " f"(total {canon.total_time_sec:.3f}s)")
        print(f"    SearchBoard: {sb.avg_ms:.2f} ms / replay " f"(total {sb.total_time_sec:.3f}s)")
        print(f"    Speedup:     {speedup:.2f}x")
        print()

    if not all_results:
        print("No fixtures could be benchmarked (missing DBs or fixtures).")
        return 0

    print("=" * 72)
    print("Summary")
    print("=" * 72)
    by_board: dict[str, List[ReplayBenchmarkResult]] = {}
    for res in all_results:
        by_board.setdefault(res.board_type, []).append(res)

    for board_type, results in sorted(by_board.items()):
        canon_times = [r.total_time_sec for r in results if r.mode == "canonical" and r.total_time_sec > 0.0]
        sb_times = [r.total_time_sec for r in results if r.mode == "search_board" and r.total_time_sec > 0.0]
        if not canon_times or not sb_times:
            continue
        avg_canon = sum(canon_times) / len(canon_times)
        avg_sb = sum(sb_times) / len(sb_times)
        speedup = avg_canon / avg_sb if avg_sb > 0.0 else 0.0
        print(f"  Board {board_type}: average speedup " f"{speedup:.2f}x over {len(canon_times)} fixtures")

    print()
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
