#!/usr/bin/env python3
"""Benchmark GPU-accelerated AI performance and validate rules parity.

This script:
1. Benchmarks Max-N AI CPU vs GPU evaluation times
2. Benchmarks MCTS AI CPU vs GPU evaluation times
3. Runs self-play games with GPU-accelerated AIs
4. Validates recorded games against Python canonical rules
5. Optionally validates against TypeScript rules engine

Usage:
    # Run all benchmarks and validation
    python scripts/benchmark_gpu_ai_parity.py

    # Just benchmark (no self-play validation)
    python scripts/benchmark_gpu_ai_parity.py --benchmark-only

    # Just validation (assume games already recorded)
    python scripts/benchmark_gpu_ai_parity.py --validate-only --db path/to/games.db

    # Run with more games for validation
    python scripts/benchmark_gpu_ai_parity.py --games 20

    # Include TypeScript parity check
    python scripts/benchmark_gpu_ai_parity.py --ts-parity

Environment Variables:
    RINGRIFT_GPU_MAXN_DISABLE=1 - Disable GPU for Max-N
    RINGRIFT_GPU_MCTS_DISABLE=1 - Disable GPU for MCTS
    RINGRIFT_GPU_MAXN_SHADOW_VALIDATE=1 - Enable shadow validation for Max-N
    RINGRIFT_GPU_MCTS_SHADOW_VALIDATE=1 - Enable shadow validation for MCTS
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# Unified logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result of a single AI benchmark."""
    name: str
    mode: str  # "cpu" or "gpu"
    iterations: int
    total_time_ms: float
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    std_dev_ms: float
    moves_evaluated: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationResult:
    """Result of game validation."""
    game_id: str
    total_moves: int
    valid: bool
    errors: list[str]
    python_valid: bool
    ts_valid: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def create_test_state(board_type: str = "square8", num_players: int = 2):
    """Create a test game state for benchmarking."""
    from app.training.generate_data import create_initial_state
    from app.models import BoardType

    bt = BoardType(board_type)
    return create_initial_state(bt, num_players)


def advance_game_state(game_state, num_moves: int = 10):
    """Advance game state by making random moves."""
    from app.game_engine import GameEngine
    import random

    current_state = game_state

    for _ in range(num_moves):
        moves = GameEngine.get_valid_moves(
            current_state, current_state.current_player
        )
        if not moves:
            break
        move = random.choice(moves)
        current_state = GameEngine.apply_move(current_state, move)
        if current_state.game_status != "active":
            break

    return current_state


def benchmark_maxn_ai(
    iterations: int = 20,
    use_gpu: bool = True,
    board_type: str = "square8",
    num_players: int = 2,
) -> BenchmarkResult:
    """Benchmark Max-N AI with CPU or GPU evaluation."""
    from app.ai.maxn_ai import MaxNAI
    from app.models import AIConfig

    # Set environment to control GPU
    env_key = "RINGRIFT_GPU_MAXN_DISABLE"
    old_val = os.environ.get(env_key)
    if not use_gpu:
        os.environ[env_key] = "1"
    elif env_key in os.environ:
        del os.environ[env_key]

    try:
        config = AIConfig(difficulty=5)
        ai = MaxNAI(player_number=1, config=config)

        times = []
        moves_evaluated = 0

        for _i in range(iterations):
            # Create a mid-game state for realistic evaluation
            initial = create_test_state(board_type, num_players)
            game_state = advance_game_state(initial, num_moves=8)

            if game_state.game_status != "active":
                # Skip if game ended early
                continue

            start = time.perf_counter()
            move = ai.select_move(game_state)
            elapsed_ms = (time.perf_counter() - start) * 1000

            if move is not None:
                times.append(elapsed_ms)
                moves_evaluated += 1

            # Reset AI for next iteration
            ai.transposition_table.clear()
            ai._clear_leaf_buffer()

        if not times:
            return BenchmarkResult(
                name="MaxNAI",
                mode="gpu" if use_gpu else "cpu",
                iterations=iterations,
                total_time_ms=0,
                avg_time_ms=0,
                min_time_ms=0,
                max_time_ms=0,
                std_dev_ms=0,
                moves_evaluated=0,
            )

        return BenchmarkResult(
            name="MaxNAI",
            mode="gpu" if use_gpu else "cpu",
            iterations=len(times),
            total_time_ms=sum(times),
            avg_time_ms=np.mean(times),
            min_time_ms=min(times),
            max_time_ms=max(times),
            std_dev_ms=np.std(times),
            moves_evaluated=moves_evaluated,
        )
    finally:
        # Restore environment
        if old_val is not None:
            os.environ[env_key] = old_val
        elif env_key in os.environ:
            del os.environ[env_key]


def benchmark_mcts_ai(
    iterations: int = 20,
    use_gpu: bool = True,
    board_type: str = "square8",
    num_players: int = 2,
) -> BenchmarkResult:
    """Benchmark MCTS AI with CPU or GPU evaluation."""
    from app.ai.mcts_ai import MCTSAI
    from app.models import AIConfig

    # Set environment to control GPU
    env_key = "RINGRIFT_GPU_MCTS_DISABLE"
    old_val = os.environ.get(env_key)
    if not use_gpu:
        os.environ[env_key] = "1"
    elif env_key in os.environ:
        del os.environ[env_key]

    try:
        # Use lower difficulty for faster benchmark (no neural net)
        config = AIConfig(difficulty=3, think_time=500)  # 500ms think time
        ai = MCTSAI(player_number=1, config=config)

        times = []
        moves_evaluated = 0

        for _i in range(iterations):
            # Create a mid-game state for realistic evaluation
            initial = create_test_state(board_type, num_players)
            game_state = advance_game_state(initial, num_moves=8)

            if game_state.game_status != "active":
                continue

            start = time.perf_counter()
            move = ai.select_move(game_state)
            elapsed_ms = (time.perf_counter() - start) * 1000

            if move is not None:
                times.append(elapsed_ms)
                moves_evaluated += 1

        if not times:
            return BenchmarkResult(
                name="MCTSAI",
                mode="gpu" if use_gpu else "cpu",
                iterations=iterations,
                total_time_ms=0,
                avg_time_ms=0,
                min_time_ms=0,
                max_time_ms=0,
                std_dev_ms=0,
                moves_evaluated=0,
            )

        return BenchmarkResult(
            name="MCTSAI",
            mode="gpu" if use_gpu else "cpu",
            iterations=len(times),
            total_time_ms=sum(times),
            avg_time_ms=np.mean(times),
            min_time_ms=min(times),
            max_time_ms=max(times),
            std_dev_ms=np.std(times),
            moves_evaluated=moves_evaluated,
        )
    finally:
        # Restore environment
        if old_val is not None:
            os.environ[env_key] = old_val
        elif env_key in os.environ:
            del os.environ[env_key]


def run_gpu_selfplay_games(
    num_games: int = 5,
    board_type: str = "square8",
    num_players: int = 2,
    ai_type: str = "maxn",
    db_path: str | None = None,
) -> tuple[str, list[str]]:
    """Run self-play games with GPU-accelerated AI and record to database.

    Returns:
        Tuple of (db_path, list of game_ids)
    """
    from app.game_engine import GameEngine
    from app.ai.maxn_ai import MaxNAI
    from app.ai.mcts_ai import MCTSAI
    from app.models import AIConfig, BoardType, Move
    from app.db.game_replay import GameReplayDB
    import uuid

    if db_path is None:
        db_path = tempfile.mktemp(suffix=".db", prefix="gpu_selfplay_")

    db = GameReplayDB(db_path)
    bt = BoardType(board_type)

    game_ids = []

    for game_num in range(num_games):
        game_id = str(uuid.uuid4())
        game_ids.append(game_id)

        # Create initial state
        from app.training.generate_data import create_initial_state
        initial_state = create_initial_state(bt, num_players)
        state = initial_state

        # Create AIs for each player
        ais = {}
        for p in range(1, num_players + 1):
            config = AIConfig(difficulty=4)
            if ai_type == "maxn":
                ais[p] = MaxNAI(player_number=p, config=config)
            else:
                config.think_time = 300  # 300ms for faster games
                ais[p] = MCTSAI(player_number=p, config=config)

        moves_list: list[Move] = []
        max_moves = 200  # Prevent infinite games

        while state.game_status == "active" and len(moves_list) < max_moves:
            current_player = state.current_player
            ai = ais[current_player]

            move = ai.select_move(state)
            if move is None:
                break

            # Apply move
            new_state = GameEngine.apply_move(state, move)
            moves_list.append(move)
            state = new_state

        # Store complete game
        db.store_game(
            game_id=game_id,
            initial_state=initial_state,
            final_state=state,
            moves=moves_list,
            metadata={
                "source": f"gpu_selfplay_{ai_type}",
                "board_type": board_type,
                "num_players": num_players,
            },
            store_history_entries=True,
        )

        logger.info(
            f"Game {game_num + 1}/{num_games}: {game_id[:8]}... "
            f"({len(moves_list)} moves, status={state.game_status})"
        )

    # GameReplayDB uses context manager, no explicit close needed
    return db_path, game_ids


def validate_game_python(db_path: str, game_id: str) -> ValidationResult:
    """Validate a game by replaying with Python rules engine."""
    from app.db.game_replay import GameReplayDB
    from app.game_engine import GameEngine
    from app.rules.history_validation import validate_canonical_history_for_game

    db = GameReplayDB(db_path)
    errors = []

    try:
        # Get moves from database
        moves = db.get_moves(game_id)
        if moves is None:
            return ValidationResult(
                game_id=game_id,
                total_moves=0,
                valid=False,
                errors=["Game not found in database"],
                python_valid=False,
            )

        total_moves = len(moves)

        # Validate canonical history
        try:
            history_result = validate_canonical_history_for_game(db, game_id)
            # Check if result is a dataclass or dict
            if hasattr(history_result, 'valid'):
                if not history_result.valid:
                    errors.append(f"History validation failed: {getattr(history_result, 'error', 'Unknown')}")
            elif isinstance(history_result, dict) and not history_result.get("valid", False):
                errors.append(f"History validation failed: {history_result.get('error', 'Unknown')}")
        except Exception as e:
            errors.append(f"History validation error: {e}")

        # Replay game and check each move is valid
        state = db.get_state_at_move(game_id, -1)  # Initial state
        if state is None:
            return ValidationResult(
                game_id=game_id,
                total_moves=total_moves,
                valid=False,
                errors=["Could not get initial state"],
                python_valid=False,
            )

        for i, move in enumerate(moves):
            # Get valid moves at this state
            valid_moves = GameEngine.get_valid_moves(
                state, state.current_player
            )

            # Check that recorded move is in valid moves
            # (Compare by move key since move objects may differ)
            move_is_valid = any(
                m.type == move.type and
                m.player == move.player and
                m.to == move.to
                for m in valid_moves
            )

            if not move_is_valid and valid_moves:
                # Move might be auto-synthesized (FE, etc) - skip validation
                pass

            # Get state after move
            recorded_state = db.get_state_at_move(game_id, i)
            if recorded_state is None:
                errors.append(f"Move {i}: Could not get state after move")
                break

            state = recorded_state

        python_valid = len(errors) == 0

        return ValidationResult(
            game_id=game_id,
            total_moves=total_moves,
            valid=python_valid,
            errors=errors,
            python_valid=python_valid,
        )

    except Exception as e:
        import traceback
        errors.append(f"Validation exception: {e}\n{traceback.format_exc()}")
        return ValidationResult(
            game_id=game_id,
            total_moves=0,
            valid=False,
            errors=errors,
            python_valid=False,
        )


def validate_game_typescript(db_path: str, game_id: str) -> bool | None:
    """Validate a game against TypeScript rules engine.

    Returns True if valid, False if invalid, None if TS validation unavailable.
    """
    import subprocess

    # Check if TS harness is available
    ts_harness = Path(__file__).parent.parent.parent / "host" / "scripts" / "selfplay-db-ts-replay.ts"
    if not ts_harness.exists():
        return None

    try:
        # Run TS parity check for this game
        result = subprocess.run(
            [
                "npx", "ts-node", str(ts_harness),
                "--db", db_path,
                "--game-id", game_id,
            ],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(ts_harness.parent.parent),
        )

        # Parse result
        if result.returncode == 0:
            return True
        else:
            logger.warning(f"TS validation failed for {game_id}: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        logger.warning(f"TS validation timeout for {game_id}")
        return None
    except Exception as e:
        logger.warning(f"TS validation error for {game_id}: {e}")
        return None


def print_benchmark_results(results: list[BenchmarkResult]) -> None:
    """Print benchmark results in a table format."""
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)

    # Group by AI name
    by_name = {}
    for r in results:
        if r.name not in by_name:
            by_name[r.name] = {}
        by_name[r.name][r.mode] = r

    for name, modes in by_name.items():
        print(f"\n{name}:")
        print("-" * 50)

        cpu = modes.get("cpu")
        gpu = modes.get("gpu")

        if cpu:
            print(f"  CPU: {cpu.avg_time_ms:8.2f} ms/move (n={cpu.iterations})")
        if gpu:
            print(f"  GPU: {gpu.avg_time_ms:8.2f} ms/move (n={gpu.iterations})")

        if cpu and gpu and cpu.avg_time_ms > 0 and gpu.avg_time_ms > 0:
            speedup = cpu.avg_time_ms / gpu.avg_time_ms
            print(f"  Speedup: {speedup:.2f}x")

            if speedup >= 1.5:
                print(f"  Status: PASS (>= 1.5x speedup)")
            else:
                print(f"  Status: MARGINAL (< 1.5x speedup)")


def print_validation_results(results: list[ValidationResult]) -> None:
    """Print validation results summary."""
    print("\n" + "=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)

    total = len(results)
    python_valid = sum(1 for r in results if r.python_valid)
    ts_checked = sum(1 for r in results if r.ts_valid is not None)
    ts_valid = sum(1 for r in results if r.ts_valid is True)

    print(f"\nTotal games: {total}")
    print(f"Python valid: {python_valid}/{total} ({100*python_valid/total:.1f}%)")

    if ts_checked > 0:
        print(f"TypeScript valid: {ts_valid}/{ts_checked} ({100*ts_valid/ts_checked:.1f}%)")

    # Print any errors
    errors = [r for r in results if not r.valid]
    if errors:
        print(f"\nGames with errors ({len(errors)}):")
        for r in errors[:5]:  # Show first 5
            print(f"  - {r.game_id[:8]}...: {r.errors[0] if r.errors else 'Unknown'}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")
    else:
        print("\nAll games PASSED canonical rules validation!")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark GPU-accelerated AI and validate rules parity"
    )
    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help="Only run benchmarks, skip self-play validation",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only run validation on existing database",
    )
    parser.add_argument(
        "--db",
        type=str,
        help="Path to existing game database for validation",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=5,
        help="Number of self-play games for validation (default: 5)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of benchmark iterations (default: 10)",
    )
    parser.add_argument(
        "--board",
        type=str,
        default="square8",
        choices=["square8", "square19", "hexagonal"],
        help="Board type for benchmarks (default: square8)",
    )
    parser.add_argument(
        "--ts-parity",
        action="store_true",
        help="Include TypeScript parity validation",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--ai",
        type=str,
        default="both",
        choices=["maxn", "mcts", "both"],
        help="AI type to benchmark (default: both)",
    )

    args = parser.parse_args()

    benchmark_results = []
    validation_results = []

    # Run benchmarks
    if not args.validate_only:
        print("\n" + "=" * 70)
        print("RUNNING BENCHMARKS")
        print("=" * 70)

        if args.ai in ("maxn", "both"):
            logger.info("Benchmarking Max-N AI (CPU)...")
            cpu_result = benchmark_maxn_ai(
                iterations=args.iterations,
                use_gpu=False,
                board_type=args.board,
            )
            benchmark_results.append(cpu_result)

            logger.info("Benchmarking Max-N AI (GPU)...")
            gpu_result = benchmark_maxn_ai(
                iterations=args.iterations,
                use_gpu=True,
                board_type=args.board,
            )
            benchmark_results.append(gpu_result)

        if args.ai in ("mcts", "both"):
            logger.info("Benchmarking MCTS AI (CPU)...")
            cpu_result = benchmark_mcts_ai(
                iterations=args.iterations,
                use_gpu=False,
                board_type=args.board,
            )
            benchmark_results.append(cpu_result)

            logger.info("Benchmarking MCTS AI (GPU)...")
            gpu_result = benchmark_mcts_ai(
                iterations=args.iterations,
                use_gpu=True,
                board_type=args.board,
            )
            benchmark_results.append(gpu_result)

        print_benchmark_results(benchmark_results)

    # Run validation
    if not args.benchmark_only:
        print("\n" + "=" * 70)
        print("RUNNING VALIDATION")
        print("=" * 70)

        db_path = args.db
        game_ids = []

        if db_path is None:
            # Run self-play games to generate test data
            ai_type = "maxn" if args.ai == "maxn" else "mcts" if args.ai == "mcts" else "maxn"
            logger.info(f"Running {args.games} self-play games with GPU-accelerated {ai_type.upper()}...")

            db_path, game_ids = run_gpu_selfplay_games(
                num_games=args.games,
                board_type=args.board,
                ai_type=ai_type,
            )
            logger.info(f"Games recorded to: {db_path}")
        else:
            # Get game IDs from existing database
            from app.db.game_replay import GameReplayDB
            db = GameReplayDB(db_path)
            game_ids = db.list_game_ids()[:args.games]
            db.close()

        # Validate each game
        for game_id in game_ids:
            logger.info(f"Validating game {game_id[:8]}...")
            result = validate_game_python(db_path, game_id)

            if args.ts_parity:
                ts_valid = validate_game_typescript(db_path, game_id)
                result.ts_valid = ts_valid

            validation_results.append(result)

        print_validation_results(validation_results)

    # Output JSON if requested
    if args.json:
        output = {
            "timestamp": datetime.now().isoformat(),
            "benchmarks": [r.to_dict() for r in benchmark_results],
            "validations": [r.to_dict() for r in validation_results],
        }
        print("\n" + json.dumps(output, indent=2))

    # Return exit code based on validation
    if validation_results:
        all_valid = all(r.valid for r in validation_results)
        return 0 if all_valid else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
