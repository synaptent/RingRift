#!/usr/bin/env python3
"""Benchmark vectorized vs legacy move application functions."""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

from app.ai.gpu_parallel_games import (
    BatchGameState,
    BatchMoves,
    generate_placement_moves_batch,
    generate_movement_moves_batch,
    generate_capture_moves_batch_vectorized,
    apply_placement_moves_batch_vectorized,
    apply_movement_moves_batch_vectorized,
    apply_capture_moves_batch_vectorized,
    _apply_placement_moves_batch_legacy,
    _apply_movement_moves_batch_legacy,
    _apply_capture_moves_batch_legacy,
)
from app.ai.gpu_batch import get_device

# Unified logging setup
try:
    from app.core.logging_config import setup_logging
    logger = setup_logging("benchmark_move_application", log_dir="logs")
except ImportError:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)


def clone_state(state: BatchGameState) -> BatchGameState:
    """Create a deep copy of BatchGameState by cloning all tensors."""
    new_state = BatchGameState.__new__(BatchGameState)

    # Copy all tensor attributes
    for attr in [
        'stack_owner', 'stack_height', 'cap_height', 'marker_owner',
        'territory_owner', 'is_collapsed', 'rings_in_hand', 'territory_count',
        'is_eliminated', 'eliminated_rings', 'buried_rings', 'rings_caused_eliminated',
        'current_player', 'current_phase', 'move_count', 'game_status', 'winner',
        'swap_offered', 'must_move_from_y', 'must_move_from_x',
        'lps_round_index', 'lps_current_round_first_player', 'lps_current_round_seen_mask',
        'lps_current_round_real_action_mask', 'lps_exclusive_player_for_completed_round',
        'lps_consecutive_exclusive_rounds', 'lps_consecutive_exclusive_player',
        'move_history',
    ]:
        setattr(new_state, attr, getattr(state, attr).clone())

    # Copy non-tensor attributes
    new_state.max_history_moves = state.max_history_moves
    new_state.lps_rounds_required = state.lps_rounds_required
    new_state.device = state.device
    new_state.batch_size = state.batch_size
    new_state.board_size = state.board_size
    new_state.num_players = state.num_players

    # Copy optional attributes if present
    for attr in ['reserves', 'field']:
        if hasattr(state, attr):
            val = getattr(state, attr)
            if val is not None and isinstance(val, torch.Tensor):
                setattr(new_state, attr, val.clone())
            else:
                setattr(new_state, attr, val)

    return new_state


def setup_placement_state(batch_size: int, board_size: int, device: torch.device, seed: int = 42):
    """Create state suitable for placement moves."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    state = BatchGameState.create_batch(
        batch_size=batch_size,
        board_size=board_size,
        num_players=2,
        device=device,
    )
    # Fresh state with reserves is ready for placement
    return state


def setup_movement_state(batch_size: int, board_size: int, device: torch.device, seed: int = 42):
    """Create state suitable for movement moves."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    state = BatchGameState.create_batch(
        batch_size=batch_size,
        board_size=board_size,
        num_players=2,
        device=device,
    )

    # Place stacks for each game to enable movement
    for g in range(batch_size):
        for player in [1, 2]:
            num_stacks = np.random.randint(3, 6)
            for _ in range(num_stacks):
                y = np.random.randint(0, board_size)
                x = np.random.randint(0, board_size)
                if state.stack_owner[g, y, x].item() != 0:
                    continue
                height = np.random.randint(1, 4)
                state.stack_owner[g, y, x] = player
                state.stack_height[g, y, x] = height
                state.cap_height[g, y, x] = min(height, np.random.randint(1, height + 1))

        state.current_player[g] = np.random.choice([1, 2])
        state.rings_in_hand[g, :] = 0  # No rings in hand to force movement phase

    return state


def setup_capture_state(batch_size: int, board_size: int, device: torch.device, seed: int = 42):
    """Create state suitable for capture moves."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    state = BatchGameState.create_batch(
        batch_size=batch_size,
        board_size=board_size,
        num_players=2,
        device=device,
    )

    # Place stacks in patterns that enable captures (adjacent opposing stacks)
    for g in range(batch_size):
        for player in [1, 2]:
            num_stacks = np.random.randint(3, 6)
            for _ in range(num_stacks):
                y = np.random.randint(0, board_size)
                x = np.random.randint(0, board_size)
                if state.stack_owner[g, y, x].item() != 0:
                    continue
                height = np.random.randint(1, 4)
                state.stack_owner[g, y, x] = player
                state.stack_height[g, y, x] = height
                state.cap_height[g, y, x] = min(height, np.random.randint(1, height + 1))

        state.current_player[g] = np.random.choice([1, 2])
        state.rings_in_hand[g, :] = 0  # No rings in hand

    return state


def benchmark_placement(args, device):
    """Benchmark placement move application."""
    logger.info(f"\n=== PLACEMENT APPLICATION BENCHMARK ===")

    vec_times = []
    leg_times = []

    for i in range(args.iterations):
        state = setup_placement_state(args.batch_size, args.board_size, device, seed=args.seed + i)
        active_mask = state.get_active_mask()
        moves = generate_placement_moves_batch(state, active_mask)

        if moves.total_moves == 0:
            continue

        # Generate random move indices
        game_counts = torch.zeros(args.batch_size, dtype=torch.long, device=device)
        for j in range(moves.total_moves):
            g = moves.game_idx[j].item()
            game_counts[g] += 1

        # Build valid move indices per game
        move_indices = torch.full((args.batch_size,), -1, dtype=torch.long, device=device)
        for g in range(args.batch_size):
            if game_counts[g] > 0:
                # Find first move for this game
                for j in range(moves.total_moves):
                    if moves.game_idx[j].item() == g:
                        move_indices[g] = j
                        break

        # Benchmark vectorized
        state_vec = clone_state(state)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        apply_placement_moves_batch_vectorized(state_vec, move_indices, moves)
        if device.type == "cuda":
            torch.cuda.synchronize()
        vec_times.append(time.perf_counter() - t0)

        # Benchmark legacy
        state_leg = clone_state(state)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _apply_placement_moves_batch_legacy(state_leg, move_indices, moves)
        if device.type == "cuda":
            torch.cuda.synchronize()
        leg_times.append(time.perf_counter() - t0)

    if not vec_times:
        logger.warning("No valid placement moves found")
        return 1.0

    vec_avg = np.mean(vec_times) * 1000
    leg_avg = np.mean(leg_times) * 1000
    speedup = leg_avg / vec_avg if vec_avg > 0 else float('inf')

    logger.info(f"  Vectorized: {vec_avg:.3f} ms avg")
    logger.info(f"  Legacy:     {leg_avg:.3f} ms avg")
    logger.info(f"  Speedup:    {speedup:.1f}x")

    return speedup


def benchmark_movement(args, device):
    """Benchmark movement move application."""
    logger.info(f"\n=== MOVEMENT APPLICATION BENCHMARK ===")

    vec_times = []
    leg_times = []

    for i in range(args.iterations):
        state = setup_movement_state(args.batch_size, args.board_size, device, seed=args.seed + i)
        active_mask = state.get_active_mask()
        moves = generate_movement_moves_batch(state, active_mask)

        if moves.total_moves == 0:
            continue

        # Build valid move indices per game
        move_indices = torch.full((args.batch_size,), -1, dtype=torch.long, device=device)
        for g in range(args.batch_size):
            for j in range(moves.total_moves):
                if moves.game_idx[j].item() == g:
                    move_indices[g] = j
                    break

        # Benchmark vectorized
        state_vec = clone_state(state)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        apply_movement_moves_batch_vectorized(state_vec, move_indices, moves)
        if device.type == "cuda":
            torch.cuda.synchronize()
        vec_times.append(time.perf_counter() - t0)

        # Benchmark legacy
        state_leg = clone_state(state)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _apply_movement_moves_batch_legacy(state_leg, move_indices, moves)
        if device.type == "cuda":
            torch.cuda.synchronize()
        leg_times.append(time.perf_counter() - t0)

    if not vec_times:
        logger.warning("No valid movement moves found")
        return 1.0

    vec_avg = np.mean(vec_times) * 1000
    leg_avg = np.mean(leg_times) * 1000
    speedup = leg_avg / vec_avg if vec_avg > 0 else float('inf')

    logger.info(f"  Vectorized: {vec_avg:.3f} ms avg")
    logger.info(f"  Legacy:     {leg_avg:.3f} ms avg")
    logger.info(f"  Speedup:    {speedup:.1f}x")

    return speedup


def benchmark_capture(args, device):
    """Benchmark capture move application."""
    logger.info(f"\n=== CAPTURE APPLICATION BENCHMARK ===")

    vec_times = []
    leg_times = []

    for i in range(args.iterations):
        state = setup_capture_state(args.batch_size, args.board_size, device, seed=args.seed + i)
        active_mask = state.get_active_mask()
        moves = generate_capture_moves_batch_vectorized(state, active_mask)

        if moves.total_moves == 0:
            continue

        # Build valid move indices per game
        move_indices = torch.full((args.batch_size,), -1, dtype=torch.long, device=device)
        for g in range(args.batch_size):
            for j in range(moves.total_moves):
                if moves.game_idx[j].item() == g:
                    move_indices[g] = j
                    break

        # Benchmark vectorized
        state_vec = clone_state(state)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        apply_capture_moves_batch_vectorized(state_vec, move_indices, moves)
        if device.type == "cuda":
            torch.cuda.synchronize()
        vec_times.append(time.perf_counter() - t0)

        # Benchmark legacy
        state_leg = clone_state(state)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _apply_capture_moves_batch_legacy(state_leg, move_indices, moves)
        if device.type == "cuda":
            torch.cuda.synchronize()
        leg_times.append(time.perf_counter() - t0)

    if not vec_times:
        logger.warning("No valid capture moves found")
        return 1.0

    vec_avg = np.mean(vec_times) * 1000
    leg_avg = np.mean(leg_times) * 1000
    speedup = leg_avg / vec_avg if vec_avg > 0 else float('inf')

    logger.info(f"  Vectorized: {vec_avg:.3f} ms avg")
    logger.info(f"  Legacy:     {leg_avg:.3f} ms avg")
    logger.info(f"  Speedup:    {speedup:.1f}x")

    return speedup


def main():
    parser = argparse.ArgumentParser(description="Benchmark move application functions")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--board-size", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    device = get_device()
    logger.info(f"Device: {device}")
    logger.info(f"Batch size: {args.batch_size}, iterations: {args.iterations}")

    speedups = {}
    speedups['placement'] = benchmark_placement(args, device)
    speedups['movement'] = benchmark_movement(args, device)
    speedups['capture'] = benchmark_capture(args, device)

    logger.info("\n" + "=" * 50)
    logger.info("SUMMARY")
    logger.info("=" * 50)
    for name, speedup in speedups.items():
        logger.info(f"  {name.capitalize()}: {speedup:.1f}x speedup")

    avg_speedup = np.mean(list(speedups.values()))
    logger.info(f"\n  Average: {avg_speedup:.1f}x speedup")


if __name__ == "__main__":
    main()
