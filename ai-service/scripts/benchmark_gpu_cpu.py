#!/usr/bin/env python3
"""Benchmark CPU vs GPU/Hybrid evaluation performance.

This script measures the performance difference between:
1. CPU-only evaluation (using GameEngine + heuristic scoring)
2. Hybrid evaluation (CPU rules + GPU heuristic scoring)
3. Pure GPU batch evaluation (if available)

This is the key benchmark for GPU Pipeline Phase 1 validation.
Target: Hybrid should achieve >= 3x speedup over CPU-only.

Usage:
    # Basic benchmark
    python scripts/benchmark_gpu_cpu.py

    # Extended benchmark with more iterations
    python scripts/benchmark_gpu_cpu.py --iterations 100

    # Specific board type
    python scripts/benchmark_gpu_cpu.py --board square8

    # Output JSON results
    python scripts/benchmark_gpu_cpu.py --json

Environment:
    RINGRIFT_SKIP_SHADOW_CONTRACTS=true (recommended for benchmarking)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set environment for consistent benchmarking
os.environ.setdefault("RINGRIFT_SKIP_SHADOW_CONTRACTS", "true")

import torch  # noqa: E402

from app.models import (  # noqa: E402
    BoardType,
    GameState,
    GamePhase,
    GameStatus,
    BoardState,
    Player,
    Position,
    RingStack,
    TimeControl,
)
from app.game_engine import GameEngine  # noqa: E402
from app.rules.core import BOARD_CONFIGS, get_victory_threshold, get_territory_victory_threshold  # noqa: E402

# Unified logging setup
from scripts.lib.logging_config import setup_script_logging

logger = setup_script_logging("benchmark_gpu_cpu")


# =============================================================================
# Data Classes for Results
# =============================================================================


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    name: str
    board_type: str
    num_players: int
    iterations: int
    total_time_ms: float
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    std_dev_ms: float
    throughput_per_sec: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ComparisonResult:
    """Comparison between CPU and GPU benchmarks."""

    board_type: str
    num_players: int
    cpu_avg_ms: float
    gpu_avg_ms: float
    speedup: float
    meets_phase1_target: bool  # >= 3x speedup

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkReport:
    """Full benchmark report."""

    timestamp: str
    device: str
    torch_version: str
    cuda_available: bool
    mps_available: bool
    results: List[BenchmarkResult]
    comparisons: List[ComparisonResult]
    phase1_gate_passed: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "device": self.device,
            "torch_version": self.torch_version,
            "cuda_available": self.cuda_available,
            "mps_available": self.mps_available,
            "results": [r.to_dict() for r in self.results],
            "comparisons": [c.to_dict() for c in self.comparisons],
            "phase1_gate_passed": self.phase1_gate_passed,
        }


# =============================================================================
# Game State Creation
# =============================================================================


def create_benchmark_state(
    board_type: BoardType,
    num_players: int = 2,
    num_stacks: int = 10,
    seed: int = 42,
) -> GameState:
    """Create a game state with some stacks for benchmarking.

    Creates a mid-game position with several stacks to provide
    realistic evaluation complexity.
    """
    import random
    random.seed(seed)

    config = BOARD_CONFIGS[board_type]
    board_size = config.size

    # Create players
    players = []
    for i in range(1, num_players + 1):
        players.append(Player(
            id=f"p{i}",
            username=f"Player{i}",
            type="ai",
            playerNumber=i,
            isReady=True,
            timeRemaining=600000,
            aiDifficulty=5,
            ringsInHand=config.rings_per_player - (num_stacks // num_players),
            eliminatedRings=0,
            territorySpaces=0,
        ))

    # Create stacks at random valid positions
    stacks = {}
    positions_used = set()

    for i in range(num_stacks):
        # Find valid position
        attempts = 0
        while attempts < 100:
            if board_type == BoardType.HEXAGONAL:
                # Hexagonal board has different valid positions
                x = random.randint(0, board_size - 1)
                y = random.randint(0, board_size - 1)
                # Simple validity check for hex
                if abs(x - board_size // 2) + abs(y - board_size // 2) <= board_size:
                    pos = (x, y)
                    if pos not in positions_used:
                        break
            else:
                x = random.randint(0, board_size - 1)
                y = random.randint(0, board_size - 1)
                pos = (x, y)
                if pos not in positions_used:
                    break
            attempts += 1

        if attempts >= 100:
            continue

        positions_used.add(pos)

        # Assign to alternating players
        player_num = (i % num_players) + 1
        key = f"{x},{y}"

        # Create stack with 1-3 rings
        height = random.randint(1, 3)
        rings = [player_num] * height

        stacks[key] = RingStack(
            position=Position(x=x, y=y),
            rings=rings,
            stackHeight=height,
            controllingPlayer=player_num,
            capHeight=height,
        )

    # Create board
    board = BoardState(
        type=board_type,
        size=board_size,
        stacks=stacks,
        markers={},
        collapsedSpaces={},
    )

    # Create time control
    time_control = TimeControl(
        initialTime=600000,
        increment=5000,
        type="fischer",
    )

    # Create game state
    now = datetime.now()
    state = GameState(
        id="benchmark-game",
        boardType=board_type,
        board=board,
        players=players,
        currentPhase=GamePhase.MOVEMENT if num_stacks > 0 else GamePhase.RING_PLACEMENT,
        currentPlayer=1,
        moveHistory=[],
        timeControl=time_control,
        spectators=[],
        gameStatus=GameStatus.ACTIVE,
        createdAt=now,
        lastMoveAt=now,
        isRated=False,
        maxPlayers=num_players,
        totalRingsInPlay=sum(len(s.rings) for s in stacks.values()),
        totalRingsEliminated=0,
        victoryThreshold=get_victory_threshold(board_type, num_players),
        territoryVictoryThreshold=get_territory_victory_threshold(board_type),
        chainCaptureState=None,
        mustMoveFromStackKey=None,
        zobristHash=None,
        lpsRoundIndex=0,
        lpsCurrentRoundActorMask={},
        lpsExclusivePlayerForCompletedRound=None,
        lpsCurrentRoundFirstPlayer=None,
        lpsConsecutiveExclusiveRounds=0,
        lpsConsecutiveExclusivePlayer=None,
    )

    return state


# =============================================================================
# Benchmark Functions
# =============================================================================


def benchmark_cpu_evaluation(
    states: List[GameState],
    iterations: int,
) -> BenchmarkResult:
    """Benchmark CPU-only heuristic evaluation."""
    from app.ai.heuristic_ai import HeuristicAI
    from app.models import AIConfig

    # Create AI instance for evaluation
    config = AIConfig(difficulty=5)
    ai = HeuristicAI(player_number=1, config=config)

    times = []

    for _ in range(iterations):
        start = time.perf_counter()

        for state in states:
            # Use the heuristic evaluation
            _ = ai.evaluate_position(state)

        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    import statistics

    return BenchmarkResult(
        name="CPU Heuristic",
        board_type=states[0].board_type.value if states else "unknown",
        num_players=len(states[0].players) if states else 0,
        iterations=iterations,
        total_time_ms=sum(times),
        avg_time_ms=statistics.mean(times),
        min_time_ms=min(times),
        max_time_ms=max(times),
        std_dev_ms=statistics.stdev(times) if len(times) > 1 else 0,
        throughput_per_sec=(len(states) * iterations) / (sum(times) / 1000),
    )


def benchmark_gpu_batch_evaluation(
    states: List[GameState],
    iterations: int,
    device: torch.device,
) -> BenchmarkResult:
    """Benchmark GPU batch heuristic evaluation."""
    from app.ai.gpu_parallel_games import BatchGameState, evaluate_positions_batch

    # Get default weights from gpu_parallel_games
    default_weights = {
        "WEIGHT_STACK_CONTROL": 1.0,
        "WEIGHT_STACK_HEIGHT": 0.5,
        "WEIGHT_CAP_HEIGHT": 0.75,
        "WEIGHT_TERRITORY": 2.0,
        "WEIGHT_RINGS_IN_HAND": 0.25,
        "WEIGHT_CENTER_CONTROL": 0.3,
        "WEIGHT_ADJACENCY": 0.2,
        "WEIGHT_LINE_PROGRESS": 1.5,
        "WEIGHT_VICTORY_PROXIMITY": 3.0,
    }

    # Convert states to batch format
    batch_states = []
    for state in states:
        batch_state = BatchGameState.from_single_game(state, device=device)
        batch_states.append(batch_state)

    # Merge into single batch (for simplicity, evaluate one at a time)
    times = []

    for _ in range(iterations):
        start = time.perf_counter()

        for batch_state in batch_states:
            # Evaluate using GPU batch function
            scores = evaluate_positions_batch(
                batch_state,
                weights=default_weights,
            )
            # Force synchronization
            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps":
                torch.mps.synchronize()

        end = time.perf_counter()
        times.append((end - start) * 1000)

    import statistics

    return BenchmarkResult(
        name=f"GPU Batch ({device.type})",
        board_type=states[0].board_type.value if states else "unknown",
        num_players=len(states[0].players) if states else 0,
        iterations=iterations,
        total_time_ms=sum(times),
        avg_time_ms=statistics.mean(times),
        min_time_ms=min(times),
        max_time_ms=max(times),
        std_dev_ms=statistics.stdev(times) if len(times) > 1 else 0,
        throughput_per_sec=(len(states) * iterations) / (sum(times) / 1000),
    )


def benchmark_hybrid_evaluation(
    states: List[GameState],
    iterations: int,
    device: torch.device,
) -> BenchmarkResult:
    """Benchmark hybrid CPU rules + GPU evaluation."""
    from app.ai.hybrid_gpu import HybridGPUEvaluator

    # Create hybrid evaluator
    board_size = 8 if states[0].board_type == BoardType.SQUARE8 else (19 if states[0].board_type == BoardType.SQUARE19 else 13)
    evaluator = HybridGPUEvaluator(device=device, board_size=board_size)

    times = []

    for _ in range(iterations):
        start = time.perf_counter()

        # Use evaluate_positions which is the main GPU evaluation method
        _ = evaluator.evaluate_positions(states, player_number=1)

        # Force synchronization
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

        end = time.perf_counter()
        times.append((end - start) * 1000)

    import statistics

    return BenchmarkResult(
        name=f"Hybrid ({device.type})",
        board_type=states[0].board_type.value if states else "unknown",
        num_players=len(states[0].players) if states else 0,
        iterations=iterations,
        total_time_ms=sum(times),
        avg_time_ms=statistics.mean(times),
        min_time_ms=min(times),
        max_time_ms=max(times),
        std_dev_ms=statistics.stdev(times) if len(times) > 1 else 0,
        throughput_per_sec=(len(states) * iterations) / (sum(times) / 1000),
    )


# =============================================================================
# Main Benchmark Runner
# =============================================================================


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def run_benchmarks(
    board_types: List[BoardType],
    num_players: int,
    num_states: int,
    iterations: int,
) -> BenchmarkReport:
    """Run all benchmarks and generate report."""

    device = get_device()
    logger.info(f"Using device: {device}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"MPS available: {torch.backends.mps.is_available()}")

    results = []
    comparisons = []

    for board_type in board_types:
        logger.info(f"\n{'='*60}")
        logger.info(f"Benchmarking {board_type.value} with {num_players} players")
        logger.info(f"{'='*60}")

        # Create test states
        states = [
            create_benchmark_state(board_type, num_players, num_stacks=10, seed=i)
            for i in range(num_states)
        ]
        logger.info(f"Created {len(states)} test states")

        # Warmup
        logger.info("Warming up...")
        _ = benchmark_cpu_evaluation(states[:1], 5)
        if device.type != "cpu":
            try:
                _ = benchmark_gpu_batch_evaluation(states[:1], 5, device)
            except Exception as e:
                logger.warning(f"GPU warmup failed: {e}")

        # CPU benchmark
        logger.info("Running CPU benchmark...")
        cpu_result = benchmark_cpu_evaluation(states, iterations)
        results.append(cpu_result)
        logger.info(f"  CPU avg: {cpu_result.avg_time_ms:.3f} ms ({cpu_result.throughput_per_sec:.1f}/sec)")

        # GPU batch benchmark
        gpu_result = None
        if device.type != "cpu":
            try:
                logger.info("Running GPU batch benchmark...")
                gpu_result = benchmark_gpu_batch_evaluation(states, iterations, device)
                results.append(gpu_result)
                logger.info(f"  GPU avg: {gpu_result.avg_time_ms:.3f} ms ({gpu_result.throughput_per_sec:.1f}/sec)")
            except Exception as e:
                logger.warning(f"GPU benchmark failed: {e}")

        # Hybrid benchmark
        hybrid_result = None
        try:
            logger.info("Running Hybrid benchmark...")
            hybrid_result = benchmark_hybrid_evaluation(states, iterations, device)
            results.append(hybrid_result)
            logger.info(f"  Hybrid avg: {hybrid_result.avg_time_ms:.3f} ms ({hybrid_result.throughput_per_sec:.1f}/sec)")
        except Exception as e:
            logger.warning(f"Hybrid benchmark failed: {e}")

        # Calculate speedups
        if gpu_result:
            gpu_speedup = cpu_result.avg_time_ms / gpu_result.avg_time_ms
            comparisons.append(ComparisonResult(
                board_type=board_type.value,
                num_players=num_players,
                cpu_avg_ms=cpu_result.avg_time_ms,
                gpu_avg_ms=gpu_result.avg_time_ms,
                speedup=gpu_speedup,
                meets_phase1_target=gpu_speedup >= 3.0,
            ))
            logger.info(f"  GPU Speedup: {gpu_speedup:.2f}x {'✓' if gpu_speedup >= 3.0 else '✗'}")

        if hybrid_result:
            hybrid_speedup = cpu_result.avg_time_ms / hybrid_result.avg_time_ms
            comparisons.append(ComparisonResult(
                board_type=board_type.value,
                num_players=num_players,
                cpu_avg_ms=cpu_result.avg_time_ms,
                gpu_avg_ms=hybrid_result.avg_time_ms,
                speedup=hybrid_speedup,
                meets_phase1_target=hybrid_speedup >= 3.0,
            ))
            logger.info(f"  Hybrid Speedup: {hybrid_speedup:.2f}x {'✓' if hybrid_speedup >= 3.0 else '✗'}")

    # Check if Phase 1 gate is passed
    phase1_passed = any(c.meets_phase1_target for c in comparisons)

    return BenchmarkReport(
        timestamp=datetime.now().isoformat(),
        device=str(device),
        torch_version=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        mps_available=torch.backends.mps.is_available(),
        results=results,
        comparisons=comparisons,
        phase1_gate_passed=phase1_passed,
    )


def print_report(report: BenchmarkReport) -> None:
    """Print human-readable report."""
    print("\n" + "="*70)
    print("GPU PIPELINE PHASE 1 BENCHMARK REPORT")
    print("="*70)
    print(f"Timestamp: {report.timestamp}")
    print(f"Device: {report.device}")
    print(f"PyTorch: {report.torch_version}")
    print(f"CUDA: {'Yes' if report.cuda_available else 'No'}")
    print(f"MPS: {'Yes' if report.mps_available else 'No'}")

    print("\n" + "-"*70)
    print("RESULTS")
    print("-"*70)
    print(f"{'Name':<25} {'Board':<10} {'Avg (ms)':<12} {'Throughput':<15}")
    print("-"*70)
    for r in report.results:
        print(f"{r.name:<25} {r.board_type:<10} {r.avg_time_ms:<12.3f} {r.throughput_per_sec:<15.1f}/sec")

    if report.comparisons:
        print("\n" + "-"*70)
        print("SPEEDUP COMPARISONS (vs CPU)")
        print("-"*70)
        print(f"{'Board':<10} {'CPU (ms)':<12} {'GPU (ms)':<12} {'Speedup':<10} {'Phase1 Target'}")
        print("-"*70)
        for c in report.comparisons:
            status = "✓ PASS" if c.meets_phase1_target else "✗ FAIL"
            print(f"{c.board_type:<10} {c.cpu_avg_ms:<12.3f} {c.gpu_avg_ms:<12.3f} {c.speedup:<10.2f}x {status}")

    print("\n" + "="*70)
    gate_status = "✓ PASSED" if report.phase1_gate_passed else "✗ FAILED"
    print(f"PHASE 1 GATE (>= 3x speedup): {gate_status}")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Benchmark CPU vs GPU evaluation")
    parser.add_argument("--iterations", type=int, default=20, help="Number of iterations per benchmark")
    parser.add_argument("--states", type=int, default=10, help="Number of game states to evaluate")
    parser.add_argument("--board", type=str, default="all",
                       choices=["all", "square8", "square19", "hexagonal"],
                       help="Board type to benchmark")
    parser.add_argument("--players", type=int, default=2, help="Number of players")
    parser.add_argument("--json", action="store_true", help="Output JSON results")
    parser.add_argument("--output", type=str, help="Output file for JSON results")

    args = parser.parse_args()

    # Determine board types
    if args.board == "all":
        board_types = [BoardType.SQUARE8, BoardType.SQUARE19, BoardType.HEXAGONAL]
    else:
        board_type_map = {
            "square8": BoardType.SQUARE8,
            "square19": BoardType.SQUARE19,
            "hexagonal": BoardType.HEXAGONAL,
        }
        board_types = [board_type_map[args.board]]

    # Run benchmarks
    report = run_benchmarks(
        board_types=board_types,
        num_players=args.players,
        num_states=args.states,
        iterations=args.iterations,
    )

    # Output results
    if args.json:
        result_json = json.dumps(report.to_dict(), indent=2)
        if args.output:
            with open(args.output, "w") as f:
                f.write(result_json)
            logger.info(f"Results written to {args.output}")
        else:
            print(result_json)
    else:
        print_report(report)

    # Return exit code based on Phase 1 gate
    return 0 if report.phase1_gate_passed else 1


if __name__ == "__main__":
    sys.exit(main())
