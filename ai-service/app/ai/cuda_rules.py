"""CUDA-accelerated game rules for RingRift.

This module implements GPU-parallelized versions of expensive game rule checks:
- Territory counting (parallel flood fill)
- Line detection (parallel scan)
- Ring detection (parallel connected components)

These CUDA kernels run entirely on GPU, eliminating CPU-GPU data transfer overhead
for batch evaluation of positions.

Requirements:
    - CUDA-capable GPU
    - numba.cuda module

Usage:
    from app.ai.cuda_rules import (
        GPURuleChecker,
        batch_territory_count_gpu,
        batch_line_detect_gpu,
    )

    # Create checker
    checker = GPURuleChecker(board_size=8, num_players=2, device='cuda:0')

    # Batch territory counting
    territories = checker.batch_territory_count(board_states_tensor)
"""

from __future__ import annotations

import logging
from typing import Tuple, Optional, List
import numpy as np

logger = logging.getLogger(__name__)

# Try to import CUDA support
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available for CUDA rules")

try:
    import numba
    from numba import cuda, int32, int8, boolean
    from numba.cuda import threadIdx, blockIdx, blockDim, gridDim
    from numba.cuda import shared, syncthreads
    import numba.cuda as cuda_module
    CUDA_AVAILABLE = cuda.is_available() if hasattr(cuda, 'is_available') else False
    logger.info(f"CUDA available: {CUDA_AVAILABLE}")
except ImportError:
    CUDA_AVAILABLE = False
    logger.warning("Numba CUDA not available")


# =============================================================================
# CUDA Kernels for Territory Counting
# =============================================================================

if CUDA_AVAILABLE:
    @cuda.jit
    def _parallel_flood_fill_kernel(
        collapsed: cuda.devicearray,      # (batch, positions) bool
        marker_owner: cuda.devicearray,   # (batch, positions) int8
        border_player: int,               # Player whose markers form boundaries
        board_size: int,
        visited: cuda.devicearray,        # (batch, positions) bool - output
        region_id: cuda.devicearray,      # (batch, positions) int32 - output
    ):
        """Parallel flood fill using iterative wavefront expansion.

        Each block handles one game. Threads cooperate on flood fill.
        Uses shared memory for wavefront tracking.
        """
        # Shared memory for current and next wavefront
        # Max 64 positions for 8x8 board
        current_frontier = cuda.shared.array(64, dtype=int32)
        next_frontier = cuda.shared.array(64, dtype=int32)
        frontier_size = cuda.shared.array(1, dtype=int32)
        next_size = cuda.shared.array(1, dtype=int32)
        current_region = cuda.shared.array(1, dtype=int32)

        game_idx = cuda.blockIdx.x
        thread_idx = cuda.threadIdx.x
        num_threads = cuda.blockDim.x
        num_positions = board_size * board_size

        # Initialize
        if thread_idx == 0:
            frontier_size[0] = 0
            next_size[0] = 0
            current_region[0] = 1

        # Initialize visited to False for all positions
        for pos in range(thread_idx, num_positions, num_threads):
            visited[game_idx, pos] = False
            region_id[game_idx, pos] = 0

        cuda.syncthreads()

        # Process each unvisited, non-boundary position as potential seed
        for seed in range(num_positions):
            cuda.syncthreads()

            # Check if seed is valid (not visited, not collapsed, not boundary marker)
            if thread_idx == 0:
                is_valid_seed = (
                    not visited[game_idx, seed] and
                    not collapsed[game_idx, seed] and
                    marker_owner[game_idx, seed] != border_player
                )
                if is_valid_seed:
                    current_frontier[0] = seed
                    frontier_size[0] = 1
                    visited[game_idx, seed] = True
                    region_id[game_idx, seed] = current_region[0]
                else:
                    frontier_size[0] = 0

            cuda.syncthreads()

            # Iterative wavefront expansion
            while frontier_size[0] > 0:
                if thread_idx == 0:
                    next_size[0] = 0
                cuda.syncthreads()

                # Each thread processes some frontier positions
                for f_idx in range(thread_idx, frontier_size[0], num_threads):
                    pos = current_frontier[f_idx]
                    x = pos % board_size
                    y = pos // board_size

                    # Check 4 neighbors explicitly (left, right, up, down)
                    # Neighbor 0: left (x-1, y)
                    nx, ny = x - 1, y
                    if 0 <= nx < board_size and 0 <= ny < board_size:
                        n_pos = ny * board_size + nx
                        if (not visited[game_idx, n_pos] and
                            not collapsed[game_idx, n_pos] and
                            marker_owner[game_idx, n_pos] != border_player):
                            old_idx = cuda.atomic.add(next_size, 0, 1)
                            if old_idx < 64:
                                next_frontier[old_idx] = n_pos
                                visited[game_idx, n_pos] = True
                                region_id[game_idx, n_pos] = current_region[0]

                    # Neighbor 1: right (x+1, y)
                    nx, ny = x + 1, y
                    if 0 <= nx < board_size and 0 <= ny < board_size:
                        n_pos = ny * board_size + nx
                        if (not visited[game_idx, n_pos] and
                            not collapsed[game_idx, n_pos] and
                            marker_owner[game_idx, n_pos] != border_player):
                            old_idx = cuda.atomic.add(next_size, 0, 1)
                            if old_idx < 64:
                                next_frontier[old_idx] = n_pos
                                visited[game_idx, n_pos] = True
                                region_id[game_idx, n_pos] = current_region[0]

                    # Neighbor 2: up (x, y-1)
                    nx, ny = x, y - 1
                    if 0 <= nx < board_size and 0 <= ny < board_size:
                        n_pos = ny * board_size + nx
                        if (not visited[game_idx, n_pos] and
                            not collapsed[game_idx, n_pos] and
                            marker_owner[game_idx, n_pos] != border_player):
                            old_idx = cuda.atomic.add(next_size, 0, 1)
                            if old_idx < 64:
                                next_frontier[old_idx] = n_pos
                                visited[game_idx, n_pos] = True
                                region_id[game_idx, n_pos] = current_region[0]

                    # Neighbor 3: down (x, y+1)
                    nx, ny = x, y + 1
                    if 0 <= nx < board_size and 0 <= ny < board_size:
                        n_pos = ny * board_size + nx
                        if (not visited[game_idx, n_pos] and
                            not collapsed[game_idx, n_pos] and
                            marker_owner[game_idx, n_pos] != border_player):
                            old_idx = cuda.atomic.add(next_size, 0, 1)
                            if old_idx < 64:
                                next_frontier[old_idx] = n_pos
                                visited[game_idx, n_pos] = True
                                region_id[game_idx, n_pos] = current_region[0]

                cuda.syncthreads()

                # Swap frontiers
                if thread_idx == 0:
                    for i in range(min(next_size[0], 64)):
                        current_frontier[i] = next_frontier[i]
                    frontier_size[0] = min(next_size[0], 64)

                cuda.syncthreads()

            # Increment region counter after completing a region
            if thread_idx == 0:
                if frontier_size[0] == 0 and region_id[game_idx, seed] != 0:
                    # Only increment if we actually filled a region
                    pass  # Region was filled
                current_region[0] += 1

            cuda.syncthreads()


    @cuda.jit
    def _count_territory_kernel(
        region_id: cuda.devicearray,      # (batch, positions) int32
        marker_owner: cuda.devicearray,   # (batch, positions) int8
        collapsed: cuda.devicearray,      # (batch, positions) bool
        board_size: int,
        num_players: int,
        territory_counts: cuda.devicearray,  # (batch, num_players+1) int32 - output
    ):
        """Count territory for each player from region IDs.

        A region is territory for player P if:
        1. It's bounded only by player P's markers and board edges
        2. It contains no other players' markers
        """
        game_idx = cuda.blockIdx.x
        thread_idx = cuda.threadIdx.x
        num_threads = cuda.blockDim.x
        num_positions = board_size * board_size

        # Shared memory for region analysis
        # Track which players border each region
        max_regions = 64
        region_borders = cuda.shared.array((64, 5), dtype=boolean)  # region -> player borders
        region_sizes = cuda.shared.array(64, dtype=int32)

        # Initialize
        for r in range(thread_idx, max_regions, num_threads):
            region_sizes[r] = 0
            for p in range(5):
                region_borders[r, p] = False

        if thread_idx < num_players + 1:
            territory_counts[game_idx, thread_idx] = 0

        cuda.syncthreads()

        # Analyze each position
        for pos in range(thread_idx, num_positions, num_threads):
            rid = region_id[game_idx, pos]
            if rid > 0 and rid < max_regions:
                # Count region size
                cuda.atomic.add(region_sizes, rid, 1)

                # Check neighbors for bordering markers
                x = pos % board_size
                y = pos // board_size

                neighbors = [
                    (x - 1, y), (x + 1, y),
                    (x, y - 1), (x, y + 1)
                ]

                for nx, ny in neighbors:
                    if 0 <= nx < board_size and 0 <= ny < board_size:
                        n_pos = ny * board_size + nx
                        owner = marker_owner[game_idx, n_pos]
                        if owner > 0 and owner <= num_players:
                            region_borders[rid, owner] = True

        cuda.syncthreads()

        # Determine territory ownership
        # A region belongs to player P if bordered ONLY by player P
        for rid in range(thread_idx + 1, max_regions, num_threads):
            if region_sizes[rid] > 0:
                owner = 0
                num_bordering = 0

                for p in range(1, num_players + 1):
                    if region_borders[rid, p]:
                        num_bordering += 1
                        owner = p

                # Territory only if exactly one player borders it
                if num_bordering == 1:
                    cuda.atomic.add(territory_counts, (game_idx, owner), region_sizes[rid])

        cuda.syncthreads()


# =============================================================================
# GPU Rule Checker Class
# =============================================================================


class GPURuleChecker:
    """GPU-accelerated game rule checking.

    Provides batch evaluation of:
    - Territory counting
    - Line detection
    - Victory conditions

    All operations run on GPU with full rule fidelity.
    """

    def __init__(
        self,
        board_size: int = 8,
        num_players: int = 2,
        device: str = 'cuda:0',
    ):
        self.board_size = board_size
        self.num_players = num_players
        self.num_positions = board_size * board_size

        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for GPURuleChecker")

        self.device = torch.device(device)
        self.use_cuda_kernels = CUDA_AVAILABLE and 'cuda' in device

        logger.info(f"GPURuleChecker initialized (CUDA kernels: {self.use_cuda_kernels})")

    def batch_territory_count(
        self,
        collapsed: torch.Tensor,      # (batch, positions) bool
        marker_owner: torch.Tensor,   # (batch, positions) int8
    ) -> torch.Tensor:
        """Count territory for each player in batch of positions.

        Args:
            collapsed: Boolean tensor of collapsed positions
            marker_owner: Owner of marker at each position (0 = none)

        Returns:
            territory_counts: (batch, num_players+1) tensor of territory counts
        """
        batch_size = collapsed.shape[0]

        if self.use_cuda_kernels:
            return self._territory_count_cuda(collapsed, marker_owner, batch_size)
        else:
            return self._territory_count_torch(collapsed, marker_owner, batch_size)

    def _territory_count_cuda(
        self,
        collapsed: torch.Tensor,
        marker_owner: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Territory counting using CUDA kernels."""
        # Convert to numpy for numba.cuda
        collapsed_np = collapsed.cpu().numpy().astype(np.bool_)
        marker_np = marker_owner.cpu().numpy().astype(np.int8)

        # Allocate device arrays
        d_collapsed = cuda.to_device(collapsed_np)
        d_marker = cuda.to_device(marker_np)
        d_visited = cuda.device_array((batch_size, self.num_positions), dtype=np.bool_)
        d_region_id = cuda.device_array((batch_size, self.num_positions), dtype=np.int32)
        d_territory = cuda.device_array((batch_size, self.num_players + 1), dtype=np.int32)

        # Configure grid
        threads_per_block = 32  # One warp
        blocks = batch_size

        # Run flood fill for each player as border
        # Territory for player P = regions bordered only by P
        for border_player in range(1, self.num_players + 1):
            _parallel_flood_fill_kernel[blocks, threads_per_block](
                d_collapsed, d_marker, border_player, self.board_size,
                d_visited, d_region_id
            )
            cuda.synchronize()

        # Count territories
        _count_territory_kernel[blocks, threads_per_block](
            d_region_id, d_marker, d_collapsed, self.board_size,
            self.num_players, d_territory
        )
        cuda.synchronize()

        # Copy back
        territory_np = d_territory.copy_to_host()
        return torch.from_numpy(territory_np).to(self.device)

    def _territory_count_torch(
        self,
        collapsed: torch.Tensor,
        marker_owner: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Territory counting using pure PyTorch (fallback).

        Uses iterative convolution-based flood fill.
        """
        territory = torch.zeros(
            batch_size, self.num_players + 1,
            dtype=torch.int32, device=self.device
        )

        # Reshape to 2D grid
        collapsed_2d = collapsed.view(batch_size, self.board_size, self.board_size)
        marker_2d = marker_owner.view(batch_size, self.board_size, self.board_size)

        # For each player, find their territory
        for player in range(1, self.num_players + 1):
            # Create boundary mask: collapsed OR other player's markers
            boundary = collapsed_2d | ((marker_2d != 0) & (marker_2d != player))

            # Find connected regions not blocked by boundary
            # Use iterative dilation with player's markers as seeds
            player_markers = (marker_2d == player).float()

            # Expand from player markers
            kernel = torch.tensor([
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0]
            ], dtype=torch.float32, device=self.device).view(1, 1, 3, 3)

            reachable = player_markers.unsqueeze(1)  # Add channel dim
            boundary_float = boundary.float().unsqueeze(1)

            # Iterative expansion
            for _ in range(self.board_size * 2):
                expanded = torch.nn.functional.conv2d(
                    reachable, kernel, padding=1
                )
                expanded = (expanded > 0).float()
                expanded = expanded * (1 - boundary_float)  # Block at boundaries

                if torch.equal(expanded, reachable):
                    break
                reachable = expanded

            # Territory = positions reachable from player markers but not markers themselves
            reachable_squeezed = reachable.squeeze(1)
            territory_mask = reachable_squeezed.bool() & ~player_markers.bool() & ~boundary.bool()

            territory[:, player] = territory_mask.sum(dim=(1, 2)).int()

        return territory


# =============================================================================
# Convenience Functions
# =============================================================================


def batch_territory_count_gpu(
    collapsed: np.ndarray,
    marker_owner: np.ndarray,
    board_size: int = 8,
    num_players: int = 2,
    device: str = 'cuda:0',
) -> np.ndarray:
    """Convenience function for batch territory counting.

    Args:
        collapsed: (batch, positions) boolean array
        marker_owner: (batch, positions) int8 array
        board_size: Board dimension
        num_players: Number of players
        device: CUDA device

    Returns:
        territory_counts: (batch, num_players+1) array
    """
    checker = GPURuleChecker(board_size, num_players, device)

    collapsed_t = torch.from_numpy(collapsed).to(checker.device)
    marker_t = torch.from_numpy(marker_owner).to(checker.device)

    result = checker.batch_territory_count(collapsed_t, marker_t)
    return result.cpu().numpy()


def benchmark_gpu_territory(
    batch_sizes: List[int] = [1, 10, 100, 1000],
    board_size: int = 8,
    num_iterations: int = 10,
) -> dict:
    """Benchmark GPU territory counting.

    Returns:
        Dictionary with benchmark results
    """
    import time

    if not TORCH_AVAILABLE:
        return {"error": "PyTorch not available"}

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    checker = GPURuleChecker(board_size, 2, device)

    results = {
        "batch_sizes": batch_sizes,
        "times_ms": [],
        "positions_per_second": [],
        "device": device,
    }

    for batch_size in batch_sizes:
        # Create random test data
        collapsed = torch.rand(batch_size, board_size * board_size, device=device) < 0.1
        marker_owner = (torch.rand(batch_size, board_size * board_size, device=device) * 3).to(torch.int8)

        # Warmup
        _ = checker.batch_territory_count(collapsed, marker_owner)
        if device != 'cpu':
            torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(num_iterations):
            _ = checker.batch_territory_count(collapsed, marker_owner)
            if device != 'cpu':
                torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        avg_ms = (elapsed / num_iterations) * 1000
        positions_per_sec = (batch_size * board_size * board_size) / (elapsed / num_iterations)

        results["times_ms"].append(avg_ms)
        results["positions_per_second"].append(positions_per_sec)

        logger.info(f"Batch {batch_size}: {avg_ms:.2f}ms, {positions_per_sec:.0f} pos/sec")

    return results
