#!/usr/bin/env python3
"""Test parity and performance of vectorized movement generation.

Compares the new generate_movement_moves_batch_vectorized with the legacy
Python-loop implementation to ensure identical results.
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

from app.ai.gpu_parallel_games import (
    BatchGameState,
    generate_movement_moves_batch,
    generate_movement_moves_batch_vectorized,
    _generate_movement_moves_batch_legacy,
    generate_capture_moves_batch_vectorized,
    _generate_capture_moves_batch_legacy,
    generate_placement_moves_batch,
)
from app.ai.gpu_batch import get_device

# Unified logging setup
from scripts.lib.logging_config import setup_script_logging

logger = setup_script_logging("test_vectorized_movement")


def moves_to_set(moves):
    """Convert BatchMoves to a set of (game_idx, from_y, from_x, to_y, to_x) tuples."""
    if moves.total_moves == 0:
        return set()
    result = set()
    for i in range(moves.total_moves):
        result.add((
            moves.game_idx[i].item(),
            moves.from_y[i].item(),
            moves.from_x[i].item(),
            moves.to_y[i].item(),
            moves.to_x[i].item(),
        ))
    return result


def test_parity_single_state(state, round_num=0, move_type="movement") -> bool:
    """Test that vectorized and legacy produce same moves for a given state."""
    active_mask = state.get_active_mask()

    if move_type == "movement":
        # Run vectorized version
        t0 = time.perf_counter()
        moves_vec = generate_movement_moves_batch_vectorized(state, active_mask)
        t_vec = time.perf_counter() - t0

        # Run legacy version
        t0 = time.perf_counter()
        moves_leg = _generate_movement_moves_batch_legacy(state, active_mask)
        t_leg = time.perf_counter() - t0
    else:  # capture
        # Run vectorized version
        t0 = time.perf_counter()
        moves_vec = generate_capture_moves_batch_vectorized(state, active_mask)
        t_vec = time.perf_counter() - t0

        # Run legacy version
        t0 = time.perf_counter()
        moves_leg = _generate_capture_moves_batch_legacy(state, active_mask)
        t_leg = time.perf_counter() - t0

    # Convert to sets for comparison
    set_vec = moves_to_set(moves_vec)
    set_leg = moves_to_set(moves_leg)

    if set_vec != set_leg:
        logger.error(f"Round {round_num} ({move_type}): MISMATCH!")
        logger.error(f"  Vectorized: {len(set_vec)} moves")
        logger.error(f"  Legacy: {len(set_leg)} moves")

        only_vec = set_vec - set_leg
        only_leg = set_leg - set_vec

        if only_vec:
            logger.error(f"  Only in vectorized ({len(only_vec)}):")
            for m in list(only_vec)[:5]:
                logger.error(f"    {m}")
        if only_leg:
            logger.error(f"  Only in legacy ({len(only_leg)}):")
            for m in list(only_leg)[:5]:
                logger.error(f"    {m}")
        return False

    return True


def setup_test_state(batch_size: int, board_size: int, device: torch.device, seed: int = 42) -> BatchGameState:
    """Create a test state with some stacks on the board."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    state = BatchGameState.create_batch(
        batch_size=batch_size,
        board_size=board_size,
        num_players=2,
        device=device,
    )

    # Place some random stacks
    for g in range(batch_size):
        # Place 2-5 stacks per player
        for player in [1, 2]:
            num_stacks = np.random.randint(2, 6)
            for _ in range(num_stacks):
                y = np.random.randint(0, board_size)
                x = np.random.randint(0, board_size)

                # Skip if already occupied
                if state.stack_owner[g, y, x].item() != 0:
                    continue

                height = np.random.randint(1, 4)
                state.stack_owner[g, y, x] = player
                state.stack_height[g, y, x] = height
                state.cap_height[g, y, x] = min(height, np.random.randint(1, height + 1))

        # Maybe add some collapsed cells
        if np.random.random() < 0.3:
            num_collapsed = np.random.randint(1, 4)
            for _ in range(num_collapsed):
                y = np.random.randint(0, board_size)
                x = np.random.randint(0, board_size)
                if state.stack_owner[g, y, x].item() == 0:
                    state.is_collapsed[g, y, x] = True

        # Set current player
        state.current_player[g] = np.random.choice([1, 2])

    return state


def run_parity_tests(args, move_type="movement"):
    """Run parity tests between vectorized and legacy implementations."""
    device = get_device()
    logger.info(f"Device: {device}")
    logger.info(f"Running {args.rounds} {move_type} parity test rounds")

    passed = 0
    failed = 0

    for round_num in range(args.rounds):
        seed = args.seed + round_num
        state = setup_test_state(
            batch_size=args.batch_size,
            board_size=args.board_size,
            device=device,
            seed=seed,
        )

        if test_parity_single_state(state, round_num, move_type):
            passed += 1
        else:
            failed += 1
            if failed >= 5:
                logger.error("Too many failures, stopping")
                break

        if (round_num + 1) % 10 == 0:
            logger.info(f"  Round {round_num + 1}/{args.rounds}: {passed} passed, {failed} failed")

    logger.info(f"\n{move_type.capitalize()} Parity Results: {passed}/{passed + failed} passed")
    return failed == 0


def run_benchmark(args, move_type="movement"):
    """Benchmark vectorized vs legacy performance."""
    device = get_device()
    logger.info(f"Device: {device}")
    logger.info(f"Running {move_type} benchmark: batch_size={args.batch_size}, iterations={args.iterations}")

    # Create a stable test state
    state = setup_test_state(
        batch_size=args.batch_size,
        board_size=args.board_size,
        device=device,
        seed=args.seed,
    )
    active_mask = state.get_active_mask()

    if move_type == "movement":
        vec_func = generate_movement_moves_batch_vectorized
        leg_func = _generate_movement_moves_batch_legacy
    else:
        vec_func = generate_capture_moves_batch_vectorized
        leg_func = _generate_capture_moves_batch_legacy

    # Warmup
    for _ in range(3):
        _ = vec_func(state, active_mask)
        _ = leg_func(state, active_mask)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark vectorized
    t0 = time.perf_counter()
    for _ in range(args.iterations):
        _ = vec_func(state, active_mask)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_vec = time.perf_counter() - t0

    # Benchmark legacy
    t0 = time.perf_counter()
    for _ in range(args.iterations):
        _ = leg_func(state, active_mask)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_leg = time.perf_counter() - t0

    vec_per_iter = t_vec / args.iterations * 1000  # ms
    leg_per_iter = t_leg / args.iterations * 1000  # ms
    vec_games_per_sec = args.batch_size * args.iterations / t_vec
    leg_games_per_sec = args.batch_size * args.iterations / t_leg

    speedup = t_leg / t_vec if t_vec > 0 else float('inf')

    logger.info(f"\n=== {move_type.capitalize()} Benchmark Results ===")
    logger.info(f"Vectorized: {vec_per_iter:.2f} ms/iter, {vec_games_per_sec:.1f} games/sec")
    logger.info(f"Legacy:     {leg_per_iter:.2f} ms/iter, {leg_games_per_sec:.1f} games/sec")
    logger.info(f"Speedup:    {speedup:.2f}x")

    return speedup


def main():
    parser = argparse.ArgumentParser(description="Test vectorized movement and capture generation")
    parser.add_argument("--mode", choices=["parity", "benchmark", "both"], default="both")
    parser.add_argument("--move-type", choices=["movement", "capture", "all"], default="all")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--board-size", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    success = True
    move_types = ["movement", "capture"] if args.move_type == "all" else [args.move_type]

    for move_type in move_types:
        if args.mode in ["parity", "both"]:
            logger.info("\n" + "=" * 60)
            logger.info(f"{move_type.upper()} PARITY TESTS")
            logger.info("=" * 60)
            if not run_parity_tests(args, move_type):
                success = False

        if args.mode in ["benchmark", "both"]:
            logger.info("\n" + "=" * 60)
            logger.info(f"{move_type.upper()} BENCHMARK")
            logger.info("=" * 60)
            run_benchmark(args, move_type)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
