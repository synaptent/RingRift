#!/usr/bin/env python
"""Memory benchmark for heavy AI operations.

Benchmarks peak memory usage of RingRift AI types across different board types
and configurations. Designed to run on high-memory machines (Mac Studio 96GB).

This script:
1. Benchmarks AI move generation for different AI types (Minimax, MCTS, Descent)
2. Measures peak RSS during self-play games
3. Tests neural network operations (NNUE, full neural net)
4. Generates detailed memory profiles and reports

Example usage:
    # Quick benchmark on local machine
    python scripts/benchmark_ai_memory.py --quick

    # Full benchmark with all board types
    python scripts/benchmark_ai_memory.py --full

    # Specific AI types
    python scripts/benchmark_ai_memory.py --ai-types minimax,mcts --board-types square8

    # Remote execution on Mac Studio
    python scripts/benchmark_ai_memory.py --remote mac-studio --full
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Add ai-service to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.distributed.memory import (
    MemoryProfile,
    MemoryTracker,
    format_memory_profile,
    get_current_rss_mb,
    get_peak_rss_mb,
    write_memory_report,
)
from app.distributed.hosts import (
    detect_host_memory,
    get_ssh_executor,
    load_remote_hosts,
)

# Unified logging setup
from scripts.lib.logging_config import setup_script_logging

logger = setup_script_logging("benchmark_ai_memory")


# =============================================================================
# Benchmark Configurations
# =============================================================================


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""

    name: str
    ai_type: str  # minimax, mcts, descent, random
    difficulty: int
    board_type: str  # square8, square19, hexagonal
    num_players: int
    moves_to_benchmark: int = 10
    warmup_moves: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)


# Predefined benchmark configurations
QUICK_BENCHMARKS = [
    # Light benchmarks for quick testing
    BenchmarkConfig("minimax_d3_8x8", "minimax", 3, "square8", 2, moves_to_benchmark=5),
    BenchmarkConfig("minimax_d4_8x8_nnue", "minimax", 4, "square8", 2, moves_to_benchmark=5),
    BenchmarkConfig("mcts_d5_8x8", "mcts", 5, "square8", 2, moves_to_benchmark=5),
    BenchmarkConfig("mcts_d6_8x8_neural", "mcts", 6, "square8", 2, moves_to_benchmark=5),
]

STANDARD_BENCHMARKS = QUICK_BENCHMARKS + [
    # More thorough 8x8 tests
    BenchmarkConfig("minimax_d3_8x8_4p", "minimax", 3, "square8", 4),
    BenchmarkConfig("mcts_d7_8x8", "mcts", 7, "square8", 2),
    # Descent AI
    BenchmarkConfig("descent_d5_8x8", "descent", 5, "square8", 2),
]

FULL_BENCHMARKS = STANDARD_BENCHMARKS + [
    # 19x19 benchmarks (high memory)
    BenchmarkConfig("minimax_d3_19x19", "minimax", 3, "square19", 2, moves_to_benchmark=5),
    BenchmarkConfig("minimax_d4_19x19_nnue", "minimax", 4, "square19", 2, moves_to_benchmark=5),
    BenchmarkConfig("mcts_d5_19x19", "mcts", 5, "square19", 2, moves_to_benchmark=5),
    BenchmarkConfig("mcts_d6_19x19_neural", "mcts", 6, "square19", 2, moves_to_benchmark=5),
    # Hexagonal benchmarks (high memory)
    BenchmarkConfig("minimax_d3_hex", "minimax", 3, "hexagonal", 2, moves_to_benchmark=5),
    BenchmarkConfig("mcts_d5_hex", "mcts", 5, "hexagonal", 2, moves_to_benchmark=5),
]


# =============================================================================
# Benchmark Runner
# =============================================================================


def run_single_benchmark(config: BenchmarkConfig) -> MemoryProfile:
    """Run a single benchmark and return the memory profile.

    Args:
        config: Benchmark configuration

    Returns:
        MemoryProfile with peak memory usage
    """
    import uuid
    from datetime import datetime as dt
    from app.models import GameState, BoardType, GamePhase, GameStatus, Player, TimeControl, BoardState, AIConfig
    from app.ai.heuristic_ai import HeuristicAI
    from app.ai.minimax_ai import MinimaxAI
    from app.ai.mcts_ai import MCTSAI
    from app.ai.random_ai import RandomAI
    from app.rules.default_engine import DefaultRulesEngine

    AI_CLASSES = {
        "random": RandomAI,
        "heuristic": HeuristicAI,
        "minimax": MinimaxAI,
        "mcts": MCTSAI,
    }

    logger.info(f"Starting benchmark: {config.name}")
    logger.info(
        f"  AI: {config.ai_type} D{config.difficulty}, Board: {config.board_type}, Players: {config.num_players}"
    )

    # Force garbage collection before benchmark
    gc.collect()
    baseline_rss = get_current_rss_mb()
    logger.info(f"  Baseline RSS: {baseline_rss} MB")

    # Create board
    board_type = BoardType(config.board_type.lower())
    size = 8
    if board_type == BoardType.SQUARE19:
        size = 19
    elif board_type == BoardType.HEXAGONAL:
        size = 5

    board = BoardState(type=board_type, size=size, stacks={}, markers={}, collapsedSpaces={}, eliminatedRings={})

    # Create players
    players = []
    for p in range(1, config.num_players + 1):
        players.append(
            Player(
                id=f"player{p}",
                username=f"{config.ai_type}_L{config.difficulty}_P{p}",
                type="ai",
                playerNumber=p,
                isReady=True,
                timeRemaining=600000,
                aiDifficulty=config.difficulty,
                ringsInHand=20,
                eliminatedRings=0,
                territorySpaces=0,
            )
        )

    game_state = GameState(
        id=str(uuid.uuid4()),
        boardType=board_type,
        board=board,
        players=players,
        currentPhase=GamePhase.RING_PLACEMENT,
        currentPlayer=1,
        moveHistory=[],
        timeControl=TimeControl(initialTime=600, increment=5, type="standard"),
        gameStatus=GameStatus.ACTIVE,
        createdAt=dt.now(),
        lastMoveAt=dt.now(),
        isRated=False,
        maxPlayers=config.num_players,
        totalRingsInPlay=0,
        totalRingsEliminated=0,
        victoryThreshold=3,
        territoryVictoryThreshold=10,
        chainCaptureState=None,
        mustMoveFromStackKey=None,
        zobristHash=None,
    )

    # Create AI instances
    ai_class = AI_CLASSES.get(config.ai_type.lower())
    if not ai_class:
        raise ValueError(f"Unknown AI type: {config.ai_type}")

    ai_config = AIConfig(difficulty=config.difficulty)
    ais = {}
    for p in range(1, config.num_players + 1):
        ais[p] = ai_class(player_number=p, config=ai_config)

    rules_engine = DefaultRulesEngine()

    # Start memory tracking
    tracker = MemoryTracker(
        operation_name=config.name,
        sample_interval=0.5,
        use_tracemalloc=True,
        metadata={
            "ai_type": config.ai_type,
            "difficulty": config.difficulty,
            "board_type": config.board_type,
            "num_players": config.num_players,
        },
    )

    tracker.start()

    try:
        logger.info(
            f"  Game initialized, performing {config.warmup_moves} warmup + {config.moves_to_benchmark} benchmark moves"
        )

        total_moves = config.warmup_moves + config.moves_to_benchmark
        move_count = 0

        for i in range(total_moves):
            if game_state.game_status != GameStatus.ACTIVE:
                logger.info(f"  Game ended at move {i}")
                break

            # Get current AI
            current_player = game_state.current_player
            current_ai = ais[current_player]
            current_ai.player_number = current_player

            # Get AI move
            try:
                move = current_ai.select_move(game_state)
            except Exception as e:
                logger.warning(f"  AI move selection error: {e}")
                break

            if move is None:
                logger.warning(f"  No move available for player {current_player}")
                break

            # Apply move
            try:
                game_state = rules_engine.apply_move(game_state, move)
            except Exception as e:
                logger.warning(f"  Move application error: {e}")
                break

            move_count += 1

            if i >= config.warmup_moves and i % 2 == 0:
                current_rss = get_current_rss_mb()
                logger.info(f"  Move {i}: RSS = {current_rss} MB")

    except Exception as e:
        logger.error(f"  Benchmark failed: {e}")
        import traceback

        traceback.print_exc()

    finally:
        profile = tracker.stop()

    logger.info(f"  Completed: Peak RSS = {profile.peak_rss_mb} MB, Increase = {profile.memory_increase_mb} MB")

    # Force cleanup
    gc.collect()

    return profile


def run_benchmarks(
    configs: List[BenchmarkConfig],
    output_path: Optional[str] = None,
) -> List[MemoryProfile]:
    """Run multiple benchmarks and generate a report.

    Args:
        configs: List of benchmark configurations
        output_path: Optional path for JSON report

    Returns:
        List of MemoryProfile results
    """
    logger.info(f"Running {len(configs)} benchmarks")

    # Detect local memory
    memory_info = detect_host_memory("local")
    logger.info(f"Host memory: {memory_info}")

    profiles = []

    for i, config in enumerate(configs, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Benchmark {i}/{len(configs)}: {config.name}")
        logger.info(f"{'='*60}")

        try:
            profile = run_single_benchmark(config)
            profiles.append(profile)
        except Exception as e:
            logger.error(f"Benchmark {config.name} failed: {e}")
            continue

        # Brief pause between benchmarks for cleanup
        time.sleep(1)
        gc.collect()

    # Generate report
    logger.info(f"\n{'='*60}")
    logger.info("BENCHMARK RESULTS")
    logger.info(f"{'='*60}")

    for profile in profiles:
        logger.info(f"\n{format_memory_profile(profile)}")

    # Summary
    if profiles:
        max_peak = max(p.peak_rss_mb for p in profiles)
        max_increase = max(p.memory_increase_mb for p in profiles)
        logger.info(f"\nSummary:")
        logger.info(f"  Max peak RSS: {max_peak} MB")
        logger.info(f"  Max memory increase: {max_increase} MB")

    # Write report
    if output_path:
        write_memory_report(profiles, output_path, format="json")
        logger.info(f"\nReport written to: {output_path}")

    return profiles


def run_remote_benchmark(
    host_name: str,
    benchmark_type: str = "standard",
    output_dir: Optional[str] = None,
) -> None:
    """Run benchmarks on a remote host via SSH.

    Args:
        host_name: Name of configured remote host
        benchmark_type: "quick", "standard", or "full"
        output_dir: Local directory to copy results to
    """
    logger.info(f"Running {benchmark_type} benchmarks on remote host: {host_name}")

    executor = get_ssh_executor(host_name)
    if not executor:
        logger.error(f"Host not found: {host_name}")
        return

    if not executor.is_alive():
        logger.error(f"Cannot connect to host: {host_name}")
        return

    # Check remote memory
    memory_info = detect_host_memory(host_name)
    logger.info(f"Remote host memory: {memory_info}")

    # Run benchmark script remotely
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    remote_output = f"/tmp/memory_benchmark_{timestamp}.json"
    log_file = f"/tmp/memory_benchmark_{timestamp}.log"

    cmd = f"PYTHONPATH=. python scripts/benchmark_ai_memory.py --{benchmark_type} --output {remote_output}"

    logger.info(f"Starting remote benchmark: {cmd}")
    result = executor.run_async(cmd, log_file)

    if result.returncode != 0:
        logger.error(f"Failed to start remote benchmark: {result.stderr}")
        return

    logger.info(f"Remote benchmark started. Check {log_file} on {host_name}")
    logger.info(f"Results will be written to {remote_output}")

    # Copy results back when done (would need to poll/wait)
    if output_dir:
        logger.info(f"Copy results with: scp {host_name}:{remote_output} {output_dir}/")


# =============================================================================
# Neural Network Specific Benchmarks
# =============================================================================


def benchmark_nnue_inference(
    board_type: str = "square8",
    num_iterations: int = 1000,
) -> MemoryProfile:
    """Benchmark NNUE inference memory usage.

    Args:
        board_type: Board type to benchmark
        num_iterations: Number of inference iterations

    Returns:
        MemoryProfile
    """
    from app.models import BoardType
    from app.ai.nnue import load_nnue_model, RingRiftNNUE, get_feature_dim
    import numpy as np

    logger.info(f"Benchmarking NNUE inference: {board_type}, {num_iterations} iterations")

    bt = BoardType(board_type)
    feature_dim = get_feature_dim(bt)

    tracker = MemoryTracker(
        operation_name=f"nnue_inference_{board_type}",
        sample_interval=0.1,
        metadata={"board_type": board_type, "iterations": num_iterations},
    )

    tracker.start()

    try:
        # Load or create model
        model = load_nnue_model(bt, allow_fresh=True)

        # Run inference iterations
        for i in range(num_iterations):
            features = np.random.randn(feature_dim).astype(np.float32)
            _ = model.forward_single(features)

            if i % 100 == 0:
                logger.info(f"  Iteration {i}: RSS = {get_current_rss_mb()} MB")

    finally:
        profile = tracker.stop()

    return profile


def benchmark_nnue_training_step(
    board_type: str = "square8",
    batch_size: int = 256,
    num_batches: int = 100,
) -> MemoryProfile:
    """Benchmark NNUE training memory usage.

    Args:
        board_type: Board type to benchmark
        batch_size: Training batch size
        num_batches: Number of training batches

    Returns:
        MemoryProfile
    """
    import torch
    import torch.nn as nn
    from app.models import BoardType
    from app.ai.nnue import RingRiftNNUE, get_feature_dim

    logger.info(f"Benchmarking NNUE training: {board_type}, batch_size={batch_size}, {num_batches} batches")

    bt = BoardType(board_type)
    feature_dim = get_feature_dim(bt)

    tracker = MemoryTracker(
        operation_name=f"nnue_training_{board_type}",
        sample_interval=0.1,
        metadata={
            "board_type": board_type,
            "batch_size": batch_size,
            "num_batches": num_batches,
        },
    )

    tracker.start()

    try:
        # Create model and optimizer
        model = RingRiftNNUE(board_type=bt)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        model.train()

        # Run training batches
        for i in range(num_batches):
            # Generate random batch
            features = torch.randn(batch_size, feature_dim)
            targets = torch.randn(batch_size, 1)

            # Forward + backward
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                logger.info(f"  Batch {i}: RSS = {get_current_rss_mb()} MB, Loss = {loss.item():.4f}")

    finally:
        profile = tracker.stop()

    return profile


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Benchmark AI memory usage")

    # Benchmark type selection
    type_group = parser.add_mutually_exclusive_group()
    type_group.add_argument("--quick", action="store_true", help="Run quick benchmarks (light AI only)")
    type_group.add_argument("--standard", action="store_true", help="Run standard benchmarks")
    type_group.add_argument("--full", action="store_true", help="Run full benchmarks (all board types)")
    type_group.add_argument("--nnue", action="store_true", help="Benchmark NNUE specifically")

    # Custom configuration
    parser.add_argument("--ai-types", type=str, help="Comma-separated AI types (minimax,mcts,descent)")
    parser.add_argument("--board-types", type=str, help="Comma-separated board types (square8,square19,hexagonal)")
    parser.add_argument("--difficulties", type=str, help="Comma-separated difficulties (3,4,5,6)")

    # Remote execution
    parser.add_argument("--remote", type=str, metavar="HOST", help="Run on remote host")

    # Output
    parser.add_argument("--output", "-o", type=str, help="Output JSON report path")
    parser.add_argument("--output-dir", type=str, help="Output directory for remote results")

    args = parser.parse_args()

    # Remote execution
    if args.remote:
        benchmark_type = "standard"
        if args.quick:
            benchmark_type = "quick"
        elif args.full:
            benchmark_type = "full"
        run_remote_benchmark(args.remote, benchmark_type, args.output_dir)
        return

    # NNUE-specific benchmarks
    if args.nnue:
        profiles = []

        for bt in ["square8", "square19"]:
            # Inference benchmark
            profile = benchmark_nnue_inference(bt, num_iterations=500)
            profiles.append(profile)

            # Training benchmark
            profile = benchmark_nnue_training_step(bt, batch_size=256, num_batches=50)
            profiles.append(profile)

        for p in profiles:
            logger.info(f"\n{format_memory_profile(p)}")

        if args.output:
            write_memory_report(profiles, args.output)
        return

    # Custom configuration
    if args.ai_types or args.board_types or args.difficulties:
        ai_types = args.ai_types.split(",") if args.ai_types else ["minimax", "mcts"]
        board_types = args.board_types.split(",") if args.board_types else ["square8"]
        difficulties = [int(d) for d in args.difficulties.split(",")] if args.difficulties else [3, 5]

        configs = []
        for ai_type in ai_types:
            for board_type in board_types:
                for diff in difficulties:
                    name = f"{ai_type}_d{diff}_{board_type}"
                    configs.append(
                        BenchmarkConfig(
                            name=name,
                            ai_type=ai_type,
                            difficulty=diff,
                            board_type=board_type,
                            num_players=2,
                        )
                    )

        run_benchmarks(configs, args.output)
        return

    # Predefined benchmark sets
    if args.full:
        configs = FULL_BENCHMARKS
    elif args.standard:
        configs = STANDARD_BENCHMARKS
    else:  # Default to quick
        configs = QUICK_BENCHMARKS

    output_path = args.output
    if not output_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"/tmp/memory_benchmark_{timestamp}.json"

    run_benchmarks(configs, output_path)


if __name__ == "__main__":
    main()
