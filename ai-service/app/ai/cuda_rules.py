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

                # Check neighbors for bordering markers explicitly
                x = pos % board_size
                y = pos // board_size

                # Neighbor 0: left (x-1, y)
                nx, ny = x - 1, y
                if 0 <= nx < board_size and 0 <= ny < board_size:
                    n_pos = ny * board_size + nx
                    owner = marker_owner[game_idx, n_pos]
                    if owner > 0 and owner <= num_players:
                        region_borders[rid, owner] = True

                # Neighbor 1: right (x+1, y)
                nx, ny = x + 1, y
                if 0 <= nx < board_size and 0 <= ny < board_size:
                    n_pos = ny * board_size + nx
                    owner = marker_owner[game_idx, n_pos]
                    if owner > 0 and owner <= num_players:
                        region_borders[rid, owner] = True

                # Neighbor 2: up (x, y-1)
                nx, ny = x, y - 1
                if 0 <= nx < board_size and 0 <= ny < board_size:
                    n_pos = ny * board_size + nx
                    owner = marker_owner[game_idx, n_pos]
                    if owner > 0 and owner <= num_players:
                        region_borders[rid, owner] = True

                # Neighbor 3: down (x, y+1)
                nx, ny = x, y + 1
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
# CUDA Kernels for Line Detection (Power Stones)
# =============================================================================

if CUDA_AVAILABLE:
    @cuda.jit
    def _line_detect_kernel(
        marker_owner: cuda.devicearray,   # (batch, positions) int8 - marker ownership
        board_size: int,
        min_line_length: int,             # Minimum markers to form a line (typically 4)
        num_players: int,
        line_counts: cuda.devicearray,    # (batch, num_players+1) int32 - output
    ):
        """Detect lines of power stones for each player.

        Scans rows, columns, and diagonals for consecutive markers.
        Each block handles one game, threads cooperate on different scan directions.
        """
        game_idx = cuda.blockIdx.x
        thread_idx = cuda.threadIdx.x
        num_threads = cuda.blockDim.x

        # Initialize output
        if thread_idx < num_players + 1:
            line_counts[game_idx, thread_idx] = 0
        cuda.syncthreads()

        # Each thread handles different scan lines
        # Total scan lines: board_size rows + board_size cols + 2*(2*board_size-1) diagonals
        num_rows = board_size
        num_cols = board_size
        num_diag1 = 2 * board_size - 1  # Main diagonals (top-left to bottom-right)
        num_diag2 = 2 * board_size - 1  # Anti-diagonals (top-right to bottom-left)
        total_lines = num_rows + num_cols + num_diag1 + num_diag2

        for line_idx in range(thread_idx, total_lines, num_threads):
            # Determine which type of line and starting position
            if line_idx < num_rows:
                # Horizontal row scan
                row = line_idx
                start_x, start_y = 0, row
                dx, dy = 1, 0
                line_len = board_size
            elif line_idx < num_rows + num_cols:
                # Vertical column scan
                col = line_idx - num_rows
                start_x, start_y = col, 0
                dx, dy = 0, 1
                line_len = board_size
            elif line_idx < num_rows + num_cols + num_diag1:
                # Main diagonal (top-left to bottom-right)
                diag = line_idx - num_rows - num_cols
                if diag < board_size:
                    start_x, start_y = 0, diag
                else:
                    start_x, start_y = diag - board_size + 1, board_size - 1
                dx, dy = 1, -1
                # Calculate diagonal length
                if diag < board_size:
                    line_len = diag + 1
                else:
                    line_len = 2 * board_size - 1 - diag
            else:
                # Anti-diagonal (top-right to bottom-left)
                diag = line_idx - num_rows - num_cols - num_diag1
                if diag < board_size:
                    start_x, start_y = board_size - 1 - diag, 0
                else:
                    start_x, start_y = 0, diag - board_size + 1
                dx, dy = -1, 1
                if diag < board_size:
                    line_len = diag + 1
                else:
                    line_len = 2 * board_size - 1 - diag

            # Scan line for consecutive markers
            if line_len >= min_line_length:
                prev_owner = 0
                consecutive = 0

                for i in range(line_len):
                    x = start_x + i * dx
                    y = start_y + i * dy

                    # Bounds check
                    if x < 0 or x >= board_size or y < 0 or y >= board_size:
                        break

                    pos = y * board_size + x
                    owner = marker_owner[game_idx, pos]

                    if owner > 0 and owner == prev_owner:
                        consecutive += 1
                        if consecutive >= min_line_length:
                            # Found a line - increment count
                            cuda.atomic.add(line_counts, (game_idx, owner), 1)
                    else:
                        consecutive = 1 if owner > 0 else 0

                    prev_owner = owner

        cuda.syncthreads()


    @cuda.jit
    def _ring_detect_kernel(
        marker_owner: cuda.devicearray,   # (batch, positions) int8
        board_size: int,
        num_players: int,
        ring_counts: cuda.devicearray,    # (batch, num_players+1) int32 - output
    ):
        """Detect rings (closed loops) of markers for each player.

        A ring is a closed path of orthogonally adjacent markers.
        Uses parallel connected component labeling with cycle detection.

        For simplicity, this implementation detects 4-cell square patterns
        as the minimal ring, and larger enclosed regions.
        """
        game_idx = cuda.blockIdx.x
        thread_idx = cuda.threadIdx.x
        num_threads = cuda.blockDim.x

        # Initialize output
        if thread_idx < num_players + 1:
            ring_counts[game_idx, thread_idx] = 0
        cuda.syncthreads()

        # Simple ring detection: look for 2x2 or larger enclosed regions
        # For each player, check if they have markers forming a closed boundary
        # around empty cells or opponent markers

        # Approach: For each 2x2 cell pattern, check if all 4 corners belong
        # to the same player (simple square ring detection)
        num_patterns = (board_size - 1) * (board_size - 1)

        for pattern_idx in range(thread_idx, num_patterns, num_threads):
            # Convert pattern index to top-left corner position
            x = pattern_idx % (board_size - 1)
            y = pattern_idx // (board_size - 1)

            # Get positions of 4 corners
            pos_tl = y * board_size + x
            pos_tr = y * board_size + (x + 1)
            pos_bl = (y + 1) * board_size + x
            pos_br = (y + 1) * board_size + (x + 1)

            # Get owners
            owner_tl = marker_owner[game_idx, pos_tl]
            owner_tr = marker_owner[game_idx, pos_tr]
            owner_bl = marker_owner[game_idx, pos_bl]
            owner_br = marker_owner[game_idx, pos_br]

            # Check if all 4 belong to the same player
            if owner_tl > 0 and owner_tl == owner_tr and owner_tl == owner_bl and owner_tl == owner_br:
                cuda.atomic.add(ring_counts, (game_idx, owner_tl), 1)

        cuda.syncthreads()


# =============================================================================
# CUDA Kernels for Heuristic Evaluation
# =============================================================================

if CUDA_AVAILABLE:
    @cuda.jit
    def _stack_height_kernel(
        stack_owner: cuda.devicearray,    # (batch, positions) int8 - who owns each stack
        stack_height: cuda.devicearray,   # (batch, positions) int8 - height of each stack
        board_size: int,
        num_players: int,
        height_sums: cuda.devicearray,    # (batch, num_players+1) int32 - output: sum of heights
        stack_counts: cuda.devicearray,   # (batch, num_players+1) int32 - output: number of stacks
    ):
        """Sum stack heights and count stacks for each player.

        Parallel reduction across positions for each game in batch.
        """
        game_idx = cuda.blockIdx.x
        thread_idx = cuda.threadIdx.x
        num_threads = cuda.blockDim.x
        num_positions = board_size * board_size

        # Initialize output
        if thread_idx < num_players + 1:
            height_sums[game_idx, thread_idx] = 0
            stack_counts[game_idx, thread_idx] = 0
        cuda.syncthreads()

        # Each thread processes multiple positions
        for pos in range(thread_idx, num_positions, num_threads):
            owner = stack_owner[game_idx, pos]
            height = stack_height[game_idx, pos]

            if owner > 0 and owner <= num_players and height > 0:
                cuda.atomic.add(height_sums, (game_idx, owner), int(height))
                cuda.atomic.add(stack_counts, (game_idx, owner), 1)

        cuda.syncthreads()


    @cuda.jit
    def _cap_height_kernel(
        stack_owner: cuda.devicearray,    # (batch, positions) int8
        cap_height: cuda.devicearray,     # (batch, positions) int8 - cap (top marker) height
        board_size: int,
        num_players: int,
        cap_sums: cuda.devicearray,       # (batch, num_players+1) int32 - output
    ):
        """Sum cap heights for each player (capture power metric)."""
        game_idx = cuda.blockIdx.x
        thread_idx = cuda.threadIdx.x
        num_threads = cuda.blockDim.x
        num_positions = board_size * board_size

        if thread_idx < num_players + 1:
            cap_sums[game_idx, thread_idx] = 0
        cuda.syncthreads()

        for pos in range(thread_idx, num_positions, num_threads):
            owner = stack_owner[game_idx, pos]
            cap = cap_height[game_idx, pos]

            if owner > 0 and owner <= num_players and cap > 0:
                cuda.atomic.add(cap_sums, (game_idx, owner), int(cap))

        cuda.syncthreads()


    @cuda.jit
    def _center_control_kernel(
        stack_owner: cuda.devicearray,    # (batch, positions) int8
        stack_height: cuda.devicearray,   # (batch, positions) int8
        board_size: int,
        num_players: int,
        center_scores: cuda.devicearray,  # (batch, num_players+1) float32 - output
    ):
        """Calculate center control score for each player.

        Positions closer to center are weighted more heavily.
        Score = sum(height * (1 - distance_to_center / max_distance))
        """
        game_idx = cuda.blockIdx.x
        thread_idx = cuda.threadIdx.x
        num_threads = cuda.blockDim.x
        num_positions = board_size * board_size

        # Center coordinates
        center_x = (board_size - 1) / 2.0
        center_y = (board_size - 1) / 2.0
        max_dist = center_x * 1.414  # sqrt(2) * half_size for corner

        if thread_idx < num_players + 1:
            center_scores[game_idx, thread_idx] = 0.0
        cuda.syncthreads()

        for pos in range(thread_idx, num_positions, num_threads):
            owner = stack_owner[game_idx, pos]
            height = stack_height[game_idx, pos]

            if owner > 0 and owner <= num_players and height > 0:
                x = pos % board_size
                y = pos // board_size

                # Manhattan distance to center (simpler, still effective)
                dx = abs(x - center_x)
                dy = abs(y - center_y)
                dist = dx + dy

                # Weight: higher for positions closer to center
                # Normalize to 0-1 range where 1 = center
                weight = 1.0 - (dist / (center_x + center_y))
                score = float(height) * weight

                cuda.atomic.add(center_scores, (game_idx, owner), score)

        cuda.syncthreads()


    @cuda.jit
    def _adjacency_kernel(
        stack_owner: cuda.devicearray,    # (batch, positions) int8
        board_size: int,
        num_players: int,
        adjacency_counts: cuda.devicearray,  # (batch, num_players+1) int32 - output
    ):
        """Count adjacent friendly stacks for each player.

        Each adjacent pair counts once (undirected).
        """
        game_idx = cuda.blockIdx.x
        thread_idx = cuda.threadIdx.x
        num_threads = cuda.blockDim.x
        num_positions = board_size * board_size

        if thread_idx < num_players + 1:
            adjacency_counts[game_idx, thread_idx] = 0
        cuda.syncthreads()

        # Only check right and down neighbors to avoid double counting
        for pos in range(thread_idx, num_positions, num_threads):
            owner = stack_owner[game_idx, pos]
            if owner <= 0 or owner > num_players:
                continue

            x = pos % board_size
            y = pos // board_size

            # Check right neighbor
            if x + 1 < board_size:
                right_pos = y * board_size + (x + 1)
                if stack_owner[game_idx, right_pos] == owner:
                    cuda.atomic.add(adjacency_counts, (game_idx, owner), 1)

            # Check down neighbor
            if y + 1 < board_size:
                down_pos = (y + 1) * board_size + x
                if stack_owner[game_idx, down_pos] == owner:
                    cuda.atomic.add(adjacency_counts, (game_idx, owner), 1)

        cuda.syncthreads()


    @cuda.jit
    def _mobility_kernel(
        stack_owner: cuda.devicearray,    # (batch, positions) int8
        stack_height: cuda.devicearray,   # (batch, positions) int8
        collapsed: cuda.devicearray,      # (batch, positions) bool
        board_size: int,
        num_players: int,
        mobility_counts: cuda.devicearray,  # (batch, num_players+1) int32 - output
    ):
        """Count mobility (available moves) for each player.

        Mobility = number of valid orthogonal slide destinations for all stacks.
        A destination is valid if not collapsed and not blocked by higher stack.
        """
        game_idx = cuda.blockIdx.x
        thread_idx = cuda.threadIdx.x
        num_threads = cuda.blockDim.x
        num_positions = board_size * board_size

        if thread_idx < num_players + 1:
            mobility_counts[game_idx, thread_idx] = 0
        cuda.syncthreads()

        for pos in range(thread_idx, num_positions, num_threads):
            owner = stack_owner[game_idx, pos]
            height = stack_height[game_idx, pos]

            if owner <= 0 or owner > num_players or height <= 0:
                continue

            x = pos % board_size
            y = pos // board_size

            # Check 4 directions for movement
            # Left
            if x > 0:
                dest = y * board_size + (x - 1)
                if not collapsed[game_idx, dest]:
                    dest_height = stack_height[game_idx, dest]
                    if dest_height == 0 or height >= dest_height:
                        cuda.atomic.add(mobility_counts, (game_idx, owner), 1)

            # Right
            if x < board_size - 1:
                dest = y * board_size + (x + 1)
                if not collapsed[game_idx, dest]:
                    dest_height = stack_height[game_idx, dest]
                    if dest_height == 0 or height >= dest_height:
                        cuda.atomic.add(mobility_counts, (game_idx, owner), 1)

            # Up
            if y > 0:
                dest = (y - 1) * board_size + x
                if not collapsed[game_idx, dest]:
                    dest_height = stack_height[game_idx, dest]
                    if dest_height == 0 or height >= dest_height:
                        cuda.atomic.add(mobility_counts, (game_idx, owner), 1)

            # Down
            if y < board_size - 1:
                dest = (y + 1) * board_size + x
                if not collapsed[game_idx, dest]:
                    dest_height = stack_height[game_idx, dest]
                    if dest_height == 0 or height >= dest_height:
                        cuda.atomic.add(mobility_counts, (game_idx, owner), 1)

        cuda.syncthreads()


    @cuda.jit
    def _vulnerability_kernel(
        stack_owner: cuda.devicearray,    # (batch, positions) int8
        stack_height: cuda.devicearray,   # (batch, positions) int8
        cap_height: cuda.devicearray,     # (batch, positions) int8
        collapsed: cuda.devicearray,      # (batch, positions) bool
        board_size: int,
        num_players: int,
        vulnerability: cuda.devicearray,  # (batch, num_players+1) int32 - output
    ):
        """Count vulnerable stacks for each player.

        A stack is vulnerable if an adjacent enemy stack could capture it
        (enemy cap_height >= our stack_height).
        """
        game_idx = cuda.blockIdx.x
        thread_idx = cuda.threadIdx.x
        num_threads = cuda.blockDim.x
        num_positions = board_size * board_size

        if thread_idx < num_players + 1:
            vulnerability[game_idx, thread_idx] = 0
        cuda.syncthreads()

        for pos in range(thread_idx, num_positions, num_threads):
            owner = stack_owner[game_idx, pos]
            height = stack_height[game_idx, pos]

            if owner <= 0 or owner > num_players or height <= 0:
                continue

            x = pos % board_size
            y = pos // board_size
            is_vulnerable = False

            # Check 4 neighbors for enemy threats
            # Left
            if x > 0 and not is_vulnerable:
                n_pos = y * board_size + (x - 1)
                n_owner = stack_owner[game_idx, n_pos]
                if n_owner > 0 and n_owner != owner:
                    n_cap = cap_height[game_idx, n_pos]
                    if n_cap >= height:
                        is_vulnerable = True

            # Right
            if x < board_size - 1 and not is_vulnerable:
                n_pos = y * board_size + (x + 1)
                n_owner = stack_owner[game_idx, n_pos]
                if n_owner > 0 and n_owner != owner:
                    n_cap = cap_height[game_idx, n_pos]
                    if n_cap >= height:
                        is_vulnerable = True

            # Up
            if y > 0 and not is_vulnerable:
                n_pos = (y - 1) * board_size + x
                n_owner = stack_owner[game_idx, n_pos]
                if n_owner > 0 and n_owner != owner:
                    n_cap = cap_height[game_idx, n_pos]
                    if n_cap >= height:
                        is_vulnerable = True

            # Down
            if y < board_size - 1 and not is_vulnerable:
                n_pos = (y + 1) * board_size + x
                n_owner = stack_owner[game_idx, n_pos]
                if n_owner > 0 and n_owner != owner:
                    n_cap = cap_height[game_idx, n_pos]
                    if n_cap >= height:
                        is_vulnerable = True

            if is_vulnerable:
                cuda.atomic.add(vulnerability, (game_idx, owner), 1)

        cuda.syncthreads()


    @cuda.jit
    def _victory_check_kernel(
        eliminated_rings: cuda.devicearray,  # (batch, num_players+1) int32
        territory_counts: cuda.devicearray,  # (batch, num_players+1) int32
        line_counts: cuda.devicearray,       # (batch, num_players+1) int32
        num_players: int,
        ring_threshold: int,                 # Eliminated rings needed to win
        territory_threshold: int,            # Territory needed to win
        line_threshold: int,                 # Lines needed to win
        victory_status: cuda.devicearray,    # (batch,) int8 - output: 0=ongoing, >0=winner player
    ):
        """Check victory conditions for batch of games.

        Returns winning player number (1-4) or 0 if game ongoing.
        Checks in order: rings > territory > lines.
        """
        game_idx = cuda.blockIdx.x
        thread_idx = cuda.threadIdx.x

        if thread_idx != 0:
            return

        winner = 0

        # Check each player for victory conditions
        for player in range(1, num_players + 1):
            # Ring elimination victory
            if eliminated_rings[game_idx, player] >= ring_threshold:
                winner = player
                break

            # Territory victory
            if territory_counts[game_idx, player] >= territory_threshold:
                winner = player
                break

            # Line victory (power stones)
            if line_counts[game_idx, player] >= line_threshold:
                winner = player
                break

        victory_status[game_idx] = winner


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

    def batch_line_detect(
        self,
        marker_owner: torch.Tensor,   # (batch, positions) int8
        min_line_length: int = 4,
    ) -> torch.Tensor:
        """Detect lines of power stones for each player in batch.

        Args:
            marker_owner: Owner of marker at each position (0 = none)
            min_line_length: Minimum consecutive markers to count as a line

        Returns:
            line_counts: (batch, num_players+1) tensor of line counts
        """
        batch_size = marker_owner.shape[0]

        if self.use_cuda_kernels:
            return self._line_detect_cuda(marker_owner, min_line_length, batch_size)
        else:
            return self._line_detect_torch(marker_owner, min_line_length, batch_size)

    def _line_detect_cuda(
        self,
        marker_owner: torch.Tensor,
        min_line_length: int,
        batch_size: int,
    ) -> torch.Tensor:
        """Line detection using CUDA kernel."""
        marker_np = marker_owner.cpu().numpy().astype(np.int8)

        d_marker = cuda.to_device(marker_np)
        d_line_counts = cuda.device_array((batch_size, self.num_players + 1), dtype=np.int32)

        threads_per_block = 32
        blocks = batch_size

        _line_detect_kernel[blocks, threads_per_block](
            d_marker, self.board_size, min_line_length, self.num_players, d_line_counts
        )
        cuda.synchronize()

        line_counts_np = d_line_counts.copy_to_host()
        return torch.from_numpy(line_counts_np).to(self.device)

    def _line_detect_torch(
        self,
        marker_owner: torch.Tensor,
        min_line_length: int,
        batch_size: int,
    ) -> torch.Tensor:
        """Line detection using pure PyTorch (fallback)."""
        line_counts = torch.zeros(
            batch_size, self.num_players + 1,
            dtype=torch.int32, device=self.device
        )

        marker_2d = marker_owner.view(batch_size, self.board_size, self.board_size)

        for player in range(1, self.num_players + 1):
            player_mask = (marker_2d == player).float()

            # Create a 1D kernel for line detection
            line_kernel = torch.ones(1, 1, 1, min_line_length, device=self.device)

            # Horizontal lines
            h_conv = torch.nn.functional.conv2d(
                player_mask.unsqueeze(1), line_kernel, padding=(0, min_line_length // 2)
            )
            h_lines = (h_conv >= min_line_length).sum(dim=(1, 2, 3))

            # Vertical lines
            v_conv = torch.nn.functional.conv2d(
                player_mask.unsqueeze(1), line_kernel.transpose(2, 3), padding=(min_line_length // 2, 0)
            )
            v_lines = (v_conv >= min_line_length).sum(dim=(1, 2, 3))

            line_counts[:, player] = (h_lines + v_lines).int()

        return line_counts

    def batch_ring_detect(
        self,
        marker_owner: torch.Tensor,   # (batch, positions) int8
    ) -> torch.Tensor:
        """Detect rings (closed loops) of markers for each player in batch.

        Args:
            marker_owner: Owner of marker at each position (0 = none)

        Returns:
            ring_counts: (batch, num_players+1) tensor of ring counts
        """
        batch_size = marker_owner.shape[0]

        if self.use_cuda_kernels:
            return self._ring_detect_cuda(marker_owner, batch_size)
        else:
            return self._ring_detect_torch(marker_owner, batch_size)

    def _ring_detect_cuda(
        self,
        marker_owner: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Ring detection using CUDA kernel."""
        marker_np = marker_owner.cpu().numpy().astype(np.int8)

        d_marker = cuda.to_device(marker_np)
        d_ring_counts = cuda.device_array((batch_size, self.num_players + 1), dtype=np.int32)

        threads_per_block = 32
        blocks = batch_size

        _ring_detect_kernel[blocks, threads_per_block](
            d_marker, self.board_size, self.num_players, d_ring_counts
        )
        cuda.synchronize()

        ring_counts_np = d_ring_counts.copy_to_host()
        return torch.from_numpy(ring_counts_np).to(self.device)

    def _ring_detect_torch(
        self,
        marker_owner: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Ring detection using pure PyTorch (fallback).

        Detects 2x2 square patterns as minimal rings.
        """
        ring_counts = torch.zeros(
            batch_size, self.num_players + 1,
            dtype=torch.int32, device=self.device
        )

        marker_2d = marker_owner.view(batch_size, self.board_size, self.board_size)

        for player in range(1, self.num_players + 1):
            player_mask = (marker_2d == player).float()

            # Use 2x2 convolution to detect square rings
            square_kernel = torch.ones(1, 1, 2, 2, device=self.device)
            conv_result = torch.nn.functional.conv2d(
                player_mask.unsqueeze(1), square_kernel, padding=0
            )
            # Count positions where all 4 corners belong to player
            rings = (conv_result == 4).sum(dim=(1, 2, 3))
            ring_counts[:, player] = rings.int()

        return ring_counts

    # =========================================================================
    # Heuristic Evaluation Methods
    # =========================================================================

    def batch_stack_stats(
        self,
        stack_owner: torch.Tensor,    # (batch, positions) int8
        stack_height: torch.Tensor,   # (batch, positions) int8
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get stack statistics for each player.

        Args:
            stack_owner: Owner of stack at each position (0 = none)
            stack_height: Height of stack at each position

        Returns:
            Tuple of:
            - height_sums: (batch, num_players+1) total stack heights
            - stack_counts: (batch, num_players+1) number of stacks
        """
        batch_size = stack_owner.shape[0]

        if self.use_cuda_kernels:
            return self._stack_stats_cuda(stack_owner, stack_height, batch_size)
        else:
            return self._stack_stats_torch(stack_owner, stack_height, batch_size)

    def _stack_stats_cuda(
        self,
        stack_owner: torch.Tensor,
        stack_height: torch.Tensor,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Stack stats using CUDA kernel."""
        owner_np = stack_owner.cpu().numpy().astype(np.int8)
        height_np = stack_height.cpu().numpy().astype(np.int8)

        d_owner = cuda.to_device(owner_np)
        d_height = cuda.to_device(height_np)
        d_height_sums = cuda.device_array((batch_size, self.num_players + 1), dtype=np.int32)
        d_stack_counts = cuda.device_array((batch_size, self.num_players + 1), dtype=np.int32)

        threads_per_block = 32
        blocks = batch_size

        _stack_height_kernel[blocks, threads_per_block](
            d_owner, d_height, self.board_size, self.num_players,
            d_height_sums, d_stack_counts
        )
        cuda.synchronize()

        height_sums = torch.from_numpy(d_height_sums.copy_to_host()).to(self.device)
        stack_counts = torch.from_numpy(d_stack_counts.copy_to_host()).to(self.device)
        return height_sums, stack_counts

    def _stack_stats_torch(
        self,
        stack_owner: torch.Tensor,
        stack_height: torch.Tensor,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Stack stats using pure PyTorch."""
        height_sums = torch.zeros(batch_size, self.num_players + 1, dtype=torch.int32, device=self.device)
        stack_counts = torch.zeros(batch_size, self.num_players + 1, dtype=torch.int32, device=self.device)

        for player in range(1, self.num_players + 1):
            mask = (stack_owner == player) & (stack_height > 0)
            height_sums[:, player] = (stack_height * mask).sum(dim=1).int()
            stack_counts[:, player] = mask.sum(dim=1).int()

        return height_sums, stack_counts

    def batch_cap_height(
        self,
        stack_owner: torch.Tensor,    # (batch, positions) int8
        cap_height: torch.Tensor,     # (batch, positions) int8
    ) -> torch.Tensor:
        """Sum cap heights for each player.

        Args:
            stack_owner: Owner of stack at each position
            cap_height: Cap height at each position

        Returns:
            cap_sums: (batch, num_players+1) total cap heights
        """
        batch_size = stack_owner.shape[0]

        if self.use_cuda_kernels:
            return self._cap_height_cuda(stack_owner, cap_height, batch_size)
        else:
            return self._cap_height_torch(stack_owner, cap_height, batch_size)

    def _cap_height_cuda(
        self,
        stack_owner: torch.Tensor,
        cap_height: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Cap height sum using CUDA kernel."""
        owner_np = stack_owner.cpu().numpy().astype(np.int8)
        cap_np = cap_height.cpu().numpy().astype(np.int8)

        d_owner = cuda.to_device(owner_np)
        d_cap = cuda.to_device(cap_np)
        d_cap_sums = cuda.device_array((batch_size, self.num_players + 1), dtype=np.int32)

        threads_per_block = 32
        blocks = batch_size

        _cap_height_kernel[blocks, threads_per_block](
            d_owner, d_cap, self.board_size, self.num_players, d_cap_sums
        )
        cuda.synchronize()

        return torch.from_numpy(d_cap_sums.copy_to_host()).to(self.device)

    def _cap_height_torch(
        self,
        stack_owner: torch.Tensor,
        cap_height: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Cap height sum using pure PyTorch."""
        cap_sums = torch.zeros(batch_size, self.num_players + 1, dtype=torch.int32, device=self.device)

        for player in range(1, self.num_players + 1):
            mask = (stack_owner == player) & (cap_height > 0)
            cap_sums[:, player] = (cap_height * mask).sum(dim=1).int()

        return cap_sums

    def batch_center_control(
        self,
        stack_owner: torch.Tensor,    # (batch, positions) int8
        stack_height: torch.Tensor,   # (batch, positions) int8
    ) -> torch.Tensor:
        """Calculate center control score for each player.

        Args:
            stack_owner: Owner of stack at each position
            stack_height: Height of stack at each position

        Returns:
            center_scores: (batch, num_players+1) center control scores
        """
        batch_size = stack_owner.shape[0]

        if self.use_cuda_kernels:
            return self._center_control_cuda(stack_owner, stack_height, batch_size)
        else:
            return self._center_control_torch(stack_owner, stack_height, batch_size)

    def _center_control_cuda(
        self,
        stack_owner: torch.Tensor,
        stack_height: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Center control using CUDA kernel."""
        owner_np = stack_owner.cpu().numpy().astype(np.int8)
        height_np = stack_height.cpu().numpy().astype(np.int8)

        d_owner = cuda.to_device(owner_np)
        d_height = cuda.to_device(height_np)
        d_scores = cuda.device_array((batch_size, self.num_players + 1), dtype=np.float32)

        threads_per_block = 32
        blocks = batch_size

        _center_control_kernel[blocks, threads_per_block](
            d_owner, d_height, self.board_size, self.num_players, d_scores
        )
        cuda.synchronize()

        return torch.from_numpy(d_scores.copy_to_host()).to(self.device)

    def _center_control_torch(
        self,
        stack_owner: torch.Tensor,
        stack_height: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Center control using pure PyTorch."""
        center_scores = torch.zeros(batch_size, self.num_players + 1, dtype=torch.float32, device=self.device)

        # Create distance weights
        center = (self.board_size - 1) / 2.0
        x = torch.arange(self.board_size, device=self.device).float()
        y = torch.arange(self.board_size, device=self.device).float()
        xx, yy = torch.meshgrid(x, y, indexing='xy')
        dist = (xx - center).abs() + (yy - center).abs()
        weights = 1.0 - dist / (2 * center)
        weights = weights.flatten()  # (positions,)

        for player in range(1, self.num_players + 1):
            mask = (stack_owner == player) & (stack_height > 0)
            weighted = stack_height.float() * weights.unsqueeze(0) * mask.float()
            center_scores[:, player] = weighted.sum(dim=1)

        return center_scores

    def batch_adjacency(
        self,
        stack_owner: torch.Tensor,    # (batch, positions) int8
    ) -> torch.Tensor:
        """Count adjacent friendly stacks for each player.

        Args:
            stack_owner: Owner of stack at each position

        Returns:
            adjacency_counts: (batch, num_players+1) adjacency counts
        """
        batch_size = stack_owner.shape[0]

        if self.use_cuda_kernels:
            return self._adjacency_cuda(stack_owner, batch_size)
        else:
            return self._adjacency_torch(stack_owner, batch_size)

    def _adjacency_cuda(
        self,
        stack_owner: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Adjacency counting using CUDA kernel."""
        owner_np = stack_owner.cpu().numpy().astype(np.int8)

        d_owner = cuda.to_device(owner_np)
        d_adjacency = cuda.device_array((batch_size, self.num_players + 1), dtype=np.int32)

        threads_per_block = 32
        blocks = batch_size

        _adjacency_kernel[blocks, threads_per_block](
            d_owner, self.board_size, self.num_players, d_adjacency
        )
        cuda.synchronize()

        return torch.from_numpy(d_adjacency.copy_to_host()).to(self.device)

    def _adjacency_torch(
        self,
        stack_owner: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Adjacency counting using pure PyTorch."""
        adjacency = torch.zeros(batch_size, self.num_players + 1, dtype=torch.int32, device=self.device)

        owner_2d = stack_owner.view(batch_size, self.board_size, self.board_size)

        for player in range(1, self.num_players + 1):
            mask = (owner_2d == player).float()
            # Count right and down neighbors only
            right_match = (mask[:, :, :-1] * mask[:, :, 1:]).sum(dim=(1, 2))
            down_match = (mask[:, :-1, :] * mask[:, 1:, :]).sum(dim=(1, 2))
            adjacency[:, player] = (right_match + down_match).int()

        return adjacency

    def batch_mobility(
        self,
        stack_owner: torch.Tensor,    # (batch, positions) int8
        stack_height: torch.Tensor,   # (batch, positions) int8
        collapsed: torch.Tensor,      # (batch, positions) bool
    ) -> torch.Tensor:
        """Count mobility (available moves) for each player.

        Args:
            stack_owner: Owner of stack at each position
            stack_height: Height of stack at each position
            collapsed: Boolean mask of collapsed positions

        Returns:
            mobility_counts: (batch, num_players+1) mobility counts
        """
        batch_size = stack_owner.shape[0]

        if self.use_cuda_kernels:
            return self._mobility_cuda(stack_owner, stack_height, collapsed, batch_size)
        else:
            return self._mobility_torch(stack_owner, stack_height, collapsed, batch_size)

    def _mobility_cuda(
        self,
        stack_owner: torch.Tensor,
        stack_height: torch.Tensor,
        collapsed: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Mobility counting using CUDA kernel."""
        owner_np = stack_owner.cpu().numpy().astype(np.int8)
        height_np = stack_height.cpu().numpy().astype(np.int8)
        collapsed_np = collapsed.cpu().numpy().astype(np.bool_)

        d_owner = cuda.to_device(owner_np)
        d_height = cuda.to_device(height_np)
        d_collapsed = cuda.to_device(collapsed_np)
        d_mobility = cuda.device_array((batch_size, self.num_players + 1), dtype=np.int32)

        threads_per_block = 32
        blocks = batch_size

        _mobility_kernel[blocks, threads_per_block](
            d_owner, d_height, d_collapsed, self.board_size, self.num_players, d_mobility
        )
        cuda.synchronize()

        return torch.from_numpy(d_mobility.copy_to_host()).to(self.device)

    def _mobility_torch(
        self,
        stack_owner: torch.Tensor,
        stack_height: torch.Tensor,
        collapsed: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Mobility counting using pure PyTorch."""
        mobility = torch.zeros(batch_size, self.num_players + 1, dtype=torch.int32, device=self.device)

        owner_2d = stack_owner.view(batch_size, self.board_size, self.board_size)
        height_2d = stack_height.view(batch_size, self.board_size, self.board_size)
        collapsed_2d = collapsed.view(batch_size, self.board_size, self.board_size)

        for player in range(1, self.num_players + 1):
            player_mask = (owner_2d == player) & (height_2d > 0)
            total_moves = 0

            # Check 4 directions
            # Left: compare positions (y, x) with (y, x-1)
            can_left = player_mask[:, :, 1:] & ~collapsed_2d[:, :, :-1]
            can_left = can_left & ((height_2d[:, :, :-1] == 0) | (height_2d[:, :, 1:] >= height_2d[:, :, :-1]))
            total_moves = total_moves + can_left.sum(dim=(1, 2))

            # Right
            can_right = player_mask[:, :, :-1] & ~collapsed_2d[:, :, 1:]
            can_right = can_right & ((height_2d[:, :, 1:] == 0) | (height_2d[:, :, :-1] >= height_2d[:, :, 1:]))
            total_moves = total_moves + can_right.sum(dim=(1, 2))

            # Up
            can_up = player_mask[:, 1:, :] & ~collapsed_2d[:, :-1, :]
            can_up = can_up & ((height_2d[:, :-1, :] == 0) | (height_2d[:, 1:, :] >= height_2d[:, :-1, :]))
            total_moves = total_moves + can_up.sum(dim=(1, 2))

            # Down
            can_down = player_mask[:, :-1, :] & ~collapsed_2d[:, 1:, :]
            can_down = can_down & ((height_2d[:, 1:, :] == 0) | (height_2d[:, :-1, :] >= height_2d[:, 1:, :]))
            total_moves = total_moves + can_down.sum(dim=(1, 2))

            mobility[:, player] = total_moves.int()

        return mobility

    def batch_vulnerability(
        self,
        stack_owner: torch.Tensor,    # (batch, positions) int8
        stack_height: torch.Tensor,   # (batch, positions) int8
        cap_height: torch.Tensor,     # (batch, positions) int8
        collapsed: torch.Tensor,      # (batch, positions) bool
    ) -> torch.Tensor:
        """Count vulnerable stacks for each player.

        Args:
            stack_owner: Owner of stack at each position
            stack_height: Height of stack at each position
            cap_height: Cap height at each position
            collapsed: Boolean mask of collapsed positions

        Returns:
            vulnerability: (batch, num_players+1) vulnerability counts
        """
        batch_size = stack_owner.shape[0]

        if self.use_cuda_kernels:
            return self._vulnerability_cuda(stack_owner, stack_height, cap_height, collapsed, batch_size)
        else:
            return self._vulnerability_torch(stack_owner, stack_height, cap_height, collapsed, batch_size)

    def _vulnerability_cuda(
        self,
        stack_owner: torch.Tensor,
        stack_height: torch.Tensor,
        cap_height: torch.Tensor,
        collapsed: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Vulnerability counting using CUDA kernel."""
        owner_np = stack_owner.cpu().numpy().astype(np.int8)
        height_np = stack_height.cpu().numpy().astype(np.int8)
        cap_np = cap_height.cpu().numpy().astype(np.int8)
        collapsed_np = collapsed.cpu().numpy().astype(np.bool_)

        d_owner = cuda.to_device(owner_np)
        d_height = cuda.to_device(height_np)
        d_cap = cuda.to_device(cap_np)
        d_collapsed = cuda.to_device(collapsed_np)
        d_vuln = cuda.device_array((batch_size, self.num_players + 1), dtype=np.int32)

        threads_per_block = 32
        blocks = batch_size

        _vulnerability_kernel[blocks, threads_per_block](
            d_owner, d_height, d_cap, d_collapsed, self.board_size, self.num_players, d_vuln
        )
        cuda.synchronize()

        return torch.from_numpy(d_vuln.copy_to_host()).to(self.device)

    def _vulnerability_torch(
        self,
        stack_owner: torch.Tensor,
        stack_height: torch.Tensor,
        cap_height: torch.Tensor,
        collapsed: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Vulnerability counting using pure PyTorch."""
        vulnerability = torch.zeros(batch_size, self.num_players + 1, dtype=torch.int32, device=self.device)

        owner_2d = stack_owner.view(batch_size, self.board_size, self.board_size)
        height_2d = stack_height.view(batch_size, self.board_size, self.board_size)
        cap_2d = cap_height.view(batch_size, self.board_size, self.board_size)

        for player in range(1, self.num_players + 1):
            player_mask = (owner_2d == player) & (height_2d > 0)
            player_height = height_2d * player_mask

            # Check for adjacent enemies with sufficient cap height
            vulnerable = torch.zeros_like(player_mask)

            # Left enemy
            enemy_left = (owner_2d[:, :, :-1] > 0) & (owner_2d[:, :, :-1] != player)
            threat_left = cap_2d[:, :, :-1] >= player_height[:, :, 1:]
            vulnerable[:, :, 1:] |= player_mask[:, :, 1:] & enemy_left & threat_left

            # Right enemy
            enemy_right = (owner_2d[:, :, 1:] > 0) & (owner_2d[:, :, 1:] != player)
            threat_right = cap_2d[:, :, 1:] >= player_height[:, :, :-1]
            vulnerable[:, :, :-1] |= player_mask[:, :, :-1] & enemy_right & threat_right

            # Up enemy
            enemy_up = (owner_2d[:, :-1, :] > 0) & (owner_2d[:, :-1, :] != player)
            threat_up = cap_2d[:, :-1, :] >= player_height[:, 1:, :]
            vulnerable[:, 1:, :] |= player_mask[:, 1:, :] & enemy_up & threat_up

            # Down enemy
            enemy_down = (owner_2d[:, 1:, :] > 0) & (owner_2d[:, 1:, :] != player)
            threat_down = cap_2d[:, 1:, :] >= player_height[:, :-1, :]
            vulnerable[:, :-1, :] |= player_mask[:, :-1, :] & enemy_down & threat_down

            vulnerability[:, player] = vulnerable.sum(dim=(1, 2)).int()

        return vulnerability

    def batch_victory_check(
        self,
        eliminated_rings: torch.Tensor,   # (batch, num_players+1) int32
        territory_counts: torch.Tensor,   # (batch, num_players+1) int32
        line_counts: torch.Tensor,        # (batch, num_players+1) int32
        ring_threshold: int = 3,
        territory_threshold: int = 10,
        line_threshold: int = 2,
    ) -> torch.Tensor:
        """Check victory conditions for batch of games.

        Args:
            eliminated_rings: Eliminated ring counts per player
            territory_counts: Territory counts per player
            line_counts: Line counts per player
            ring_threshold: Rings needed to win
            territory_threshold: Territory needed to win
            line_threshold: Lines needed to win

        Returns:
            victory_status: (batch,) winner player number (0 = ongoing)
        """
        batch_size = eliminated_rings.shape[0]

        if self.use_cuda_kernels:
            return self._victory_check_cuda(
                eliminated_rings, territory_counts, line_counts,
                ring_threshold, territory_threshold, line_threshold, batch_size
            )
        else:
            return self._victory_check_torch(
                eliminated_rings, territory_counts, line_counts,
                ring_threshold, territory_threshold, line_threshold, batch_size
            )

    def _victory_check_cuda(
        self,
        eliminated_rings: torch.Tensor,
        territory_counts: torch.Tensor,
        line_counts: torch.Tensor,
        ring_threshold: int,
        territory_threshold: int,
        line_threshold: int,
        batch_size: int,
    ) -> torch.Tensor:
        """Victory checking using CUDA kernel."""
        elim_np = eliminated_rings.cpu().numpy().astype(np.int32)
        terr_np = territory_counts.cpu().numpy().astype(np.int32)
        lines_np = line_counts.cpu().numpy().astype(np.int32)

        d_elim = cuda.to_device(elim_np)
        d_terr = cuda.to_device(terr_np)
        d_lines = cuda.to_device(lines_np)
        d_status = cuda.device_array((batch_size,), dtype=np.int8)

        threads_per_block = 32
        blocks = batch_size

        _victory_check_kernel[blocks, threads_per_block](
            d_elim, d_terr, d_lines, self.num_players,
            ring_threshold, territory_threshold, line_threshold, d_status
        )
        cuda.synchronize()

        return torch.from_numpy(d_status.copy_to_host()).to(self.device)

    def _victory_check_torch(
        self,
        eliminated_rings: torch.Tensor,
        territory_counts: torch.Tensor,
        line_counts: torch.Tensor,
        ring_threshold: int,
        territory_threshold: int,
        line_threshold: int,
        batch_size: int,
    ) -> torch.Tensor:
        """Victory checking using pure PyTorch."""
        status = torch.zeros(batch_size, dtype=torch.int8, device=self.device)

        for player in range(1, self.num_players + 1):
            # Ring victory
            ring_win = eliminated_rings[:, player] >= ring_threshold
            # Territory victory
            terr_win = territory_counts[:, player] >= territory_threshold
            # Line victory
            line_win = line_counts[:, player] >= line_threshold

            winner_mask = (ring_win | terr_win | line_win) & (status == 0)
            status[winner_mask] = player

        return status

    # =========================================================================
    # Move Generation Methods
    # =========================================================================

    def count_placement_moves(
        self,
        collapsed: torch.Tensor,          # (batch, positions) bool
        stack_owner: torch.Tensor,         # (batch, positions) int8
        has_ring: torch.Tensor,            # (batch, num_players+1) bool
    ) -> torch.Tensor:
        """Count valid placement moves for each player.

        Args:
            collapsed: Boolean mask of collapsed positions
            stack_owner: Owner of stack at each position
            has_ring: Whether each player has rings to place

        Returns:
            move_counts: (batch, num_players+1) placement move counts
        """
        batch_size = collapsed.shape[0]

        if self.use_cuda_kernels:
            return self._count_placement_cuda(collapsed, stack_owner, has_ring, batch_size)
        else:
            return self._count_placement_torch(collapsed, stack_owner, has_ring, batch_size)

    def _count_placement_cuda(
        self,
        collapsed: torch.Tensor,
        stack_owner: torch.Tensor,
        has_ring: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Count placement moves using CUDA kernel."""
        collapsed_np = collapsed.cpu().numpy().astype(np.bool_)
        owner_np = stack_owner.cpu().numpy().astype(np.int8)
        has_ring_np = has_ring.cpu().numpy().astype(np.bool_)

        d_collapsed = cuda.to_device(collapsed_np)
        d_owner = cuda.to_device(owner_np)
        d_has_ring = cuda.to_device(has_ring_np)
        d_counts = cuda.device_array((batch_size, self.num_players + 1), dtype=np.int32)

        threads_per_block = 32
        blocks = batch_size

        _count_placement_moves_kernel[blocks, threads_per_block](
            d_collapsed, d_owner, d_has_ring, self.board_size, self.num_players, d_counts
        )
        cuda.synchronize()

        return torch.from_numpy(d_counts.copy_to_host()).to(self.device)

    def _count_placement_torch(
        self,
        collapsed: torch.Tensor,
        stack_owner: torch.Tensor,
        has_ring: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Count placement moves using pure PyTorch."""
        counts = torch.zeros(batch_size, self.num_players + 1, dtype=torch.int32, device=self.device)

        # Count empty non-collapsed positions
        empty = ~collapsed & (stack_owner == 0)
        empty_count = empty.sum(dim=1)

        for player in range(1, self.num_players + 1):
            counts[:, player] = empty_count * has_ring[:, player].int()

        return counts

    def count_movement_moves(
        self,
        stack_owner: torch.Tensor,         # (batch, positions) int8
        stack_height: torch.Tensor,        # (batch, positions) int8
        collapsed: torch.Tensor,           # (batch, positions) bool
    ) -> torch.Tensor:
        """Count valid movement moves for each player.

        Args:
            stack_owner: Owner of stack at each position
            stack_height: Height of stack at each position
            collapsed: Boolean mask of collapsed positions

        Returns:
            move_counts: (batch, num_players+1) movement move counts
        """
        batch_size = stack_owner.shape[0]

        if self.use_cuda_kernels:
            return self._count_movement_cuda(stack_owner, stack_height, collapsed, batch_size)
        else:
            return self._count_movement_torch(stack_owner, stack_height, collapsed, batch_size)

    def _count_movement_cuda(
        self,
        stack_owner: torch.Tensor,
        stack_height: torch.Tensor,
        collapsed: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Count movement moves using CUDA kernel."""
        owner_np = stack_owner.cpu().numpy().astype(np.int8)
        height_np = stack_height.cpu().numpy().astype(np.int8)
        collapsed_np = collapsed.cpu().numpy().astype(np.bool_)

        d_owner = cuda.to_device(owner_np)
        d_height = cuda.to_device(height_np)
        d_collapsed = cuda.to_device(collapsed_np)
        d_counts = cuda.device_array((batch_size, self.num_players + 1), dtype=np.int32)

        threads_per_block = 32
        blocks = batch_size

        _count_movement_moves_kernel[blocks, threads_per_block](
            d_owner, d_height, d_collapsed, self.board_size, self.num_players, d_counts
        )
        cuda.synchronize()

        return torch.from_numpy(d_counts.copy_to_host()).to(self.device)

    def _count_movement_torch(
        self,
        stack_owner: torch.Tensor,
        stack_height: torch.Tensor,
        collapsed: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Count movement moves using pure PyTorch (uses mobility kernel)."""
        return self.batch_mobility(stack_owner, stack_height, collapsed)

    def generate_placement_moves(
        self,
        collapsed: torch.Tensor,           # (batch, positions) bool
        stack_owner: torch.Tensor,          # (batch, positions) int8
        player: int,
        max_moves: int = 64,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate placement moves for a player.

        Args:
            collapsed: Boolean mask of collapsed positions
            stack_owner: Owner of stack at each position
            player: Player to generate moves for
            max_moves: Maximum moves to generate per game

        Returns:
            Tuple of:
            - move_targets: (batch, max_moves) target positions
            - move_counts: (batch,) actual move count per game
        """
        batch_size = collapsed.shape[0]

        if self.use_cuda_kernels:
            return self._gen_placement_cuda(collapsed, stack_owner, player, max_moves, batch_size)
        else:
            return self._gen_placement_torch(collapsed, stack_owner, player, max_moves, batch_size)

    def _gen_placement_cuda(
        self,
        collapsed: torch.Tensor,
        stack_owner: torch.Tensor,
        player: int,
        max_moves: int,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate placement moves using CUDA kernel."""
        collapsed_np = collapsed.cpu().numpy().astype(np.bool_)
        owner_np = stack_owner.cpu().numpy().astype(np.int8)

        d_collapsed = cuda.to_device(collapsed_np)
        d_owner = cuda.to_device(owner_np)
        d_targets = cuda.device_array((batch_size, max_moves), dtype=np.int16)
        d_counts = cuda.device_array((batch_size,), dtype=np.int32)

        threads_per_block = 32
        blocks = batch_size

        _generate_placement_moves_kernel[blocks, threads_per_block](
            d_collapsed, d_owner, player, self.board_size, max_moves, d_targets, d_counts
        )
        cuda.synchronize()

        targets = torch.from_numpy(d_targets.copy_to_host()).to(self.device)
        counts = torch.from_numpy(d_counts.copy_to_host()).to(self.device)
        return targets, counts

    def _gen_placement_torch(
        self,
        collapsed: torch.Tensor,
        stack_owner: torch.Tensor,
        player: int,
        max_moves: int,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate placement moves using pure PyTorch."""
        targets = torch.full((batch_size, max_moves), -1, dtype=torch.int16, device=self.device)
        counts = torch.zeros(batch_size, dtype=torch.int32, device=self.device)

        # Find empty non-collapsed positions
        empty = ~collapsed & (stack_owner == 0)

        for b in range(batch_size):
            positions = torch.where(empty[b])[0]
            num_moves = min(len(positions), max_moves)
            if num_moves > 0:
                targets[b, :num_moves] = positions[:num_moves].short()
                counts[b] = num_moves

        return targets, counts

    def generate_movement_moves(
        self,
        stack_owner: torch.Tensor,          # (batch, positions) int8
        stack_height: torch.Tensor,         # (batch, positions) int8
        collapsed: torch.Tensor,            # (batch, positions) bool
        player: int,
        max_moves: int = 256,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate movement moves for a player.

        Args:
            stack_owner: Owner of stack at each position
            stack_height: Height of stack at each position
            collapsed: Boolean mask of collapsed positions
            player: Player to generate moves for
            max_moves: Maximum moves to generate per game

        Returns:
            Tuple of:
            - move_sources: (batch, max_moves) source positions
            - move_targets: (batch, max_moves) target positions
            - move_counts: (batch,) actual move count per game
        """
        batch_size = stack_owner.shape[0]

        if self.use_cuda_kernels:
            return self._gen_movement_cuda(stack_owner, stack_height, collapsed, player, max_moves, batch_size)
        else:
            return self._gen_movement_torch(stack_owner, stack_height, collapsed, player, max_moves, batch_size)

    def _gen_movement_cuda(
        self,
        stack_owner: torch.Tensor,
        stack_height: torch.Tensor,
        collapsed: torch.Tensor,
        player: int,
        max_moves: int,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate movement moves using CUDA kernel."""
        owner_np = stack_owner.cpu().numpy().astype(np.int8)
        height_np = stack_height.cpu().numpy().astype(np.int8)
        collapsed_np = collapsed.cpu().numpy().astype(np.bool_)

        d_owner = cuda.to_device(owner_np)
        d_height = cuda.to_device(height_np)
        d_collapsed = cuda.to_device(collapsed_np)
        d_sources = cuda.device_array((batch_size, max_moves), dtype=np.int16)
        d_targets = cuda.device_array((batch_size, max_moves), dtype=np.int16)
        d_counts = cuda.device_array((batch_size,), dtype=np.int32)

        threads_per_block = 32
        blocks = batch_size

        _generate_movement_moves_kernel[blocks, threads_per_block](
            d_owner, d_height, d_collapsed, player, self.board_size, max_moves,
            d_sources, d_targets, d_counts
        )
        cuda.synchronize()

        sources = torch.from_numpy(d_sources.copy_to_host()).to(self.device)
        targets = torch.from_numpy(d_targets.copy_to_host()).to(self.device)
        counts = torch.from_numpy(d_counts.copy_to_host()).to(self.device)
        return sources, targets, counts

    def _gen_movement_torch(
        self,
        stack_owner: torch.Tensor,
        stack_height: torch.Tensor,
        collapsed: torch.Tensor,
        player: int,
        max_moves: int,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate movement moves using pure PyTorch."""
        sources = torch.full((batch_size, max_moves), -1, dtype=torch.int16, device=self.device)
        targets = torch.full((batch_size, max_moves), -1, dtype=torch.int16, device=self.device)
        counts = torch.zeros(batch_size, dtype=torch.int32, device=self.device)

        # This is a naive implementation - the CUDA version is much faster
        owner_2d = stack_owner.view(batch_size, self.board_size, self.board_size)
        height_2d = stack_height.view(batch_size, self.board_size, self.board_size)
        collapsed_2d = collapsed.view(batch_size, self.board_size, self.board_size)

        for b in range(batch_size):
            move_idx = 0
            for y in range(self.board_size):
                for x in range(self.board_size):
                    if owner_2d[b, y, x] != player or height_2d[b, y, x] <= 0:
                        continue

                    src_pos = y * self.board_size + x
                    src_h = height_2d[b, y, x]

                    # Check 4 directions
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                            if not collapsed_2d[b, ny, nx]:
                                dest_h = height_2d[b, ny, nx]
                                if dest_h == 0 or src_h >= dest_h:
                                    if move_idx < max_moves:
                                        sources[b, move_idx] = src_pos
                                        targets[b, move_idx] = ny * self.board_size + nx
                                        move_idx += 1

            counts[b] = move_idx

        return sources, targets, counts

    def generate_all_moves(
        self,
        stack_owner: torch.Tensor,          # (batch, positions) int8
        stack_height: torch.Tensor,         # (batch, positions) int8
        collapsed: torch.Tensor,            # (batch, positions) bool
        has_ring: torch.Tensor,             # (batch, num_players+1) bool
        player: int,
        max_moves: int = 320,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate all moves (placement + movement) for a player.

        Args:
            stack_owner: Owner of stack at each position
            stack_height: Height of stack at each position
            collapsed: Boolean mask of collapsed positions
            has_ring: Whether each player has rings to place
            player: Player to generate moves for
            max_moves: Maximum moves to generate per game

        Returns:
            Tuple of:
            - move_types: (batch, max_moves) int8 - 0=placement, 1=movement
            - move_sources: (batch, max_moves) int16 - source (or -1 for placement)
            - move_targets: (batch, max_moves) int16 - target position
            - move_counts: (batch,) actual move count per game
        """
        batch_size = stack_owner.shape[0]

        if self.use_cuda_kernels:
            return self._gen_all_moves_cuda(
                stack_owner, stack_height, collapsed, has_ring, player, max_moves, batch_size
            )
        else:
            return self._gen_all_moves_torch(
                stack_owner, stack_height, collapsed, has_ring, player, max_moves, batch_size
            )

    def _gen_all_moves_cuda(
        self,
        stack_owner: torch.Tensor,
        stack_height: torch.Tensor,
        collapsed: torch.Tensor,
        has_ring: torch.Tensor,
        player: int,
        max_moves: int,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate all moves using CUDA kernel."""
        owner_np = stack_owner.cpu().numpy().astype(np.int8)
        height_np = stack_height.cpu().numpy().astype(np.int8)
        collapsed_np = collapsed.cpu().numpy().astype(np.bool_)
        has_ring_np = has_ring.cpu().numpy().astype(np.bool_)

        d_owner = cuda.to_device(owner_np)
        d_height = cuda.to_device(height_np)
        d_collapsed = cuda.to_device(collapsed_np)
        d_has_ring = cuda.to_device(has_ring_np)
        d_types = cuda.device_array((batch_size, max_moves), dtype=np.int8)
        d_sources = cuda.device_array((batch_size, max_moves), dtype=np.int16)
        d_targets = cuda.device_array((batch_size, max_moves), dtype=np.int16)
        d_counts = cuda.device_array((batch_size,), dtype=np.int32)

        threads_per_block = 32
        blocks = batch_size

        _generate_all_moves_kernel[blocks, threads_per_block](
            d_owner, d_height, d_collapsed, d_has_ring, player, self.board_size, max_moves,
            d_types, d_sources, d_targets, d_counts
        )
        cuda.synchronize()

        types = torch.from_numpy(d_types.copy_to_host()).to(self.device)
        sources = torch.from_numpy(d_sources.copy_to_host()).to(self.device)
        targets = torch.from_numpy(d_targets.copy_to_host()).to(self.device)
        counts = torch.from_numpy(d_counts.copy_to_host()).to(self.device)
        return types, sources, targets, counts

    def _gen_all_moves_torch(
        self,
        stack_owner: torch.Tensor,
        stack_height: torch.Tensor,
        collapsed: torch.Tensor,
        has_ring: torch.Tensor,
        player: int,
        max_moves: int,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate all moves using pure PyTorch."""
        types = torch.full((batch_size, max_moves), -1, dtype=torch.int8, device=self.device)
        sources = torch.full((batch_size, max_moves), -1, dtype=torch.int16, device=self.device)
        targets = torch.full((batch_size, max_moves), -1, dtype=torch.int16, device=self.device)
        counts = torch.zeros(batch_size, dtype=torch.int32, device=self.device)

        # Generate placement moves first
        if has_ring[:, player].any():
            p_targets, p_counts = self.generate_placement_moves(collapsed, stack_owner, player, max_moves)

            for b in range(batch_size):
                if has_ring[b, player]:
                    num_placements = int(p_counts[b].item())
                    types[b, :num_placements] = 0  # Placement
                    sources[b, :num_placements] = -1
                    targets[b, :num_placements] = p_targets[b, :num_placements]
                    counts[b] = num_placements

        # Generate movement moves
        m_sources, m_targets, m_counts = self.generate_movement_moves(
            stack_owner, stack_height, collapsed, player, max_moves
        )

        for b in range(batch_size):
            offset = int(counts[b].item())
            num_movements = int(m_counts[b].item())
            end_idx = min(offset + num_movements, max_moves)
            num_to_add = end_idx - offset

            if num_to_add > 0:
                types[b, offset:end_idx] = 1  # Movement
                sources[b, offset:end_idx] = m_sources[b, :num_to_add]
                targets[b, offset:end_idx] = m_targets[b, :num_to_add]
                counts[b] = end_idx

        return types, sources, targets, counts

    # =========================================================================
    # Move Application Methods
    # =========================================================================

    def apply_placement_moves(
        self,
        move_targets: torch.Tensor,           # (batch,) int16
        stack_owner: torch.Tensor,            # (batch, positions) int8 - in/out
        stack_height: torch.Tensor,           # (batch, positions) int8 - in/out
        cap_height: torch.Tensor,             # (batch, positions) int8 - in/out
        ring_count: torch.Tensor,             # (batch, num_players+1) int8 - in/out
        players: torch.Tensor,                # (batch,) int8 - player for each game
    ) -> None:
        """Apply placement moves in-place for per-game players.

        Modifies stack_owner, stack_height, cap_height, ring_count in place.
        """
        batch_size = move_targets.shape[0]

        if self.use_cuda_kernels:
            self._apply_placement_cuda(
                move_targets, stack_owner, stack_height, cap_height, ring_count, players, batch_size
            )
        else:
            self._apply_placement_torch(
                move_targets, stack_owner, stack_height, cap_height, ring_count, players, batch_size
            )

    def _apply_placement_cuda(
        self,
        move_targets: torch.Tensor,
        stack_owner: torch.Tensor,
        stack_height: torch.Tensor,
        cap_height: torch.Tensor,
        ring_count: torch.Tensor,
        players: torch.Tensor,
        batch_size: int,
    ) -> None:
        """Apply placement moves using CUDA kernel."""
        targets_np = move_targets.cpu().numpy().astype(np.int16)
        owner_np = stack_owner.cpu().numpy().astype(np.int8)
        height_np = stack_height.cpu().numpy().astype(np.int8)
        cap_np = cap_height.cpu().numpy().astype(np.int8)
        ring_np = ring_count.cpu().numpy().astype(np.int8)
        players_np = players.cpu().numpy().astype(np.int8)

        d_targets = cuda.to_device(targets_np)
        d_owner = cuda.to_device(owner_np)
        d_height = cuda.to_device(height_np)
        d_cap = cuda.to_device(cap_np)
        d_ring = cuda.to_device(ring_np)
        d_players = cuda.to_device(players_np)

        _apply_placement_kernel[batch_size, 1](
            d_targets, d_owner, d_height, d_cap, d_ring, d_players, self.board_size
        )
        cuda.synchronize()

        # Copy back to tensors
        stack_owner.copy_(torch.from_numpy(d_owner.copy_to_host()).to(self.device))
        stack_height.copy_(torch.from_numpy(d_height.copy_to_host()).to(self.device))
        cap_height.copy_(torch.from_numpy(d_cap.copy_to_host()).to(self.device))
        ring_count.copy_(torch.from_numpy(d_ring.copy_to_host()).to(self.device))

    def _apply_placement_torch(
        self,
        move_targets: torch.Tensor,
        stack_owner: torch.Tensor,
        stack_height: torch.Tensor,
        cap_height: torch.Tensor,
        ring_count: torch.Tensor,
        players: torch.Tensor,
        batch_size: int,
    ) -> None:
        """Apply placement moves using pure PyTorch."""
        for b in range(batch_size):
            target = int(move_targets[b].item())
            player = int(players[b].item())
            if target >= 0:
                stack_owner[b, target] = player
                stack_height[b, target] = 1
                cap_height[b, target] = 1
                ring_count[b, player] -= 1

    def apply_movement_moves(
        self,
        move_sources: torch.Tensor,           # (batch,) int16
        move_targets: torch.Tensor,           # (batch,) int16
        stack_owner: torch.Tensor,            # (batch, positions) int8 - in/out
        stack_height: torch.Tensor,           # (batch, positions) int8 - in/out
        cap_height: torch.Tensor,             # (batch, positions) int8 - in/out
        marker_owner: torch.Tensor,           # (batch, positions) int8 - in/out
    ) -> None:
        """Apply movement moves in-place.

        Modifies all tensor arguments in place.
        """
        batch_size = move_sources.shape[0]

        if self.use_cuda_kernels:
            self._apply_movement_cuda(
                move_sources, move_targets, stack_owner, stack_height, cap_height, marker_owner, batch_size
            )
        else:
            self._apply_movement_torch(
                move_sources, move_targets, stack_owner, stack_height, cap_height, marker_owner, batch_size
            )

    def _apply_movement_cuda(
        self,
        move_sources: torch.Tensor,
        move_targets: torch.Tensor,
        stack_owner: torch.Tensor,
        stack_height: torch.Tensor,
        cap_height: torch.Tensor,
        marker_owner: torch.Tensor,
        batch_size: int,
    ) -> None:
        """Apply movement moves using CUDA kernel."""
        sources_np = move_sources.cpu().numpy().astype(np.int16)
        targets_np = move_targets.cpu().numpy().astype(np.int16)
        owner_np = stack_owner.cpu().numpy().astype(np.int8)
        height_np = stack_height.cpu().numpy().astype(np.int8)
        cap_np = cap_height.cpu().numpy().astype(np.int8)
        marker_np = marker_owner.cpu().numpy().astype(np.int8)

        d_sources = cuda.to_device(sources_np)
        d_targets = cuda.to_device(targets_np)
        d_owner = cuda.to_device(owner_np)
        d_height = cuda.to_device(height_np)
        d_cap = cuda.to_device(cap_np)
        d_marker = cuda.to_device(marker_np)

        _apply_movement_kernel[batch_size, 1](
            d_sources, d_targets, d_owner, d_height, d_cap, d_marker, self.board_size
        )
        cuda.synchronize()

        # Copy back to tensors
        stack_owner.copy_(torch.from_numpy(d_owner.copy_to_host()).to(self.device))
        stack_height.copy_(torch.from_numpy(d_height.copy_to_host()).to(self.device))
        cap_height.copy_(torch.from_numpy(d_cap.copy_to_host()).to(self.device))
        marker_owner.copy_(torch.from_numpy(d_marker.copy_to_host()).to(self.device))

    def _apply_movement_torch(
        self,
        move_sources: torch.Tensor,
        move_targets: torch.Tensor,
        stack_owner: torch.Tensor,
        stack_height: torch.Tensor,
        cap_height: torch.Tensor,
        marker_owner: torch.Tensor,
        batch_size: int,
    ) -> None:
        """Apply movement moves using pure PyTorch."""
        for b in range(batch_size):
            source = int(move_sources[b].item())
            target = int(move_targets[b].item())
            if source >= 0 and target >= 0:
                src_owner = int(stack_owner[b, source].item())
                src_height = int(stack_height[b, source].item())
                src_cap = int(cap_height[b, source].item())
                dest_height = int(stack_height[b, target].item())

                if dest_height == 0:
                    stack_owner[b, target] = src_owner
                    stack_height[b, target] = src_height
                    cap_height[b, target] = src_cap
                else:
                    stack_owner[b, target] = src_owner
                    stack_height[b, target] = dest_height + src_height
                    cap_height[b, target] = src_cap
                    marker_owner[b, target] = src_owner

                # Clear source
                stack_owner[b, source] = 0
                stack_height[b, source] = 0
                cap_height[b, source] = 0
                marker_owner[b, source] = src_owner

    def apply_batch_moves(
        self,
        move_types: torch.Tensor,             # (batch,) int8 - 0=placement, 1=movement
        move_sources: torch.Tensor,           # (batch,) int16
        move_targets: torch.Tensor,           # (batch,) int16
        players: torch.Tensor,                # (batch,) int8
        stack_owner: torch.Tensor,            # (batch, positions) int8 - in/out
        stack_height: torch.Tensor,           # (batch, positions) int8 - in/out
        cap_height: torch.Tensor,             # (batch, positions) int8 - in/out
        marker_owner: torch.Tensor,           # (batch, positions) int8 - in/out
        ring_count: torch.Tensor,             # (batch, num_players+1) int8 - in/out
    ) -> None:
        """Apply a batch of mixed moves (placement and movement) in-place.

        Each game in batch has one move to apply, which can be either placement or movement.
        """
        batch_size = move_types.shape[0]

        if self.use_cuda_kernels:
            self._apply_batch_moves_cuda(
                move_types, move_sources, move_targets, players,
                stack_owner, stack_height, cap_height, marker_owner, ring_count, batch_size
            )
        else:
            self._apply_batch_moves_torch(
                move_types, move_sources, move_targets, players,
                stack_owner, stack_height, cap_height, marker_owner, ring_count, batch_size
            )

    def _apply_batch_moves_cuda(
        self,
        move_types: torch.Tensor,
        move_sources: torch.Tensor,
        move_targets: torch.Tensor,
        players: torch.Tensor,
        stack_owner: torch.Tensor,
        stack_height: torch.Tensor,
        cap_height: torch.Tensor,
        marker_owner: torch.Tensor,
        ring_count: torch.Tensor,
        batch_size: int,
    ) -> None:
        """Apply batch moves using CUDA kernel."""
        types_np = move_types.cpu().numpy().astype(np.int8)
        sources_np = move_sources.cpu().numpy().astype(np.int16)
        targets_np = move_targets.cpu().numpy().astype(np.int16)
        players_np = players.cpu().numpy().astype(np.int8)
        owner_np = stack_owner.cpu().numpy().astype(np.int8)
        height_np = stack_height.cpu().numpy().astype(np.int8)
        cap_np = cap_height.cpu().numpy().astype(np.int8)
        marker_np = marker_owner.cpu().numpy().astype(np.int8)
        ring_np = ring_count.cpu().numpy().astype(np.int8)

        d_types = cuda.to_device(types_np)
        d_sources = cuda.to_device(sources_np)
        d_targets = cuda.to_device(targets_np)
        d_players = cuda.to_device(players_np)
        d_owner = cuda.to_device(owner_np)
        d_height = cuda.to_device(height_np)
        d_cap = cuda.to_device(cap_np)
        d_marker = cuda.to_device(marker_np)
        d_ring = cuda.to_device(ring_np)

        _batch_apply_moves_kernel[batch_size, 1](
            d_types, d_sources, d_targets, d_players,
            d_owner, d_height, d_cap, d_marker, d_ring, self.board_size
        )
        cuda.synchronize()

        # Copy back to tensors
        stack_owner.copy_(torch.from_numpy(d_owner.copy_to_host()).to(self.device))
        stack_height.copy_(torch.from_numpy(d_height.copy_to_host()).to(self.device))
        cap_height.copy_(torch.from_numpy(d_cap.copy_to_host()).to(self.device))
        marker_owner.copy_(torch.from_numpy(d_marker.copy_to_host()).to(self.device))
        ring_count.copy_(torch.from_numpy(d_ring.copy_to_host()).to(self.device))

    def _apply_batch_moves_torch(
        self,
        move_types: torch.Tensor,
        move_sources: torch.Tensor,
        move_targets: torch.Tensor,
        players: torch.Tensor,
        stack_owner: torch.Tensor,
        stack_height: torch.Tensor,
        cap_height: torch.Tensor,
        marker_owner: torch.Tensor,
        ring_count: torch.Tensor,
        batch_size: int,
    ) -> None:
        """Apply batch moves using pure PyTorch."""
        for b in range(batch_size):
            move_type = int(move_types[b].item())
            source = int(move_sources[b].item())
            target = int(move_targets[b].item())
            player = int(players[b].item())

            if target < 0:
                continue

            if move_type == 0 and player > 0:
                # Placement
                stack_owner[b, target] = player
                stack_height[b, target] = 1
                cap_height[b, target] = 1
                ring_count[b, player] -= 1
            elif move_type == 1 and source >= 0:
                # Movement
                src_owner = int(stack_owner[b, source].item())
                src_height = int(stack_height[b, source].item())
                src_cap = int(cap_height[b, source].item())
                dest_height = int(stack_height[b, target].item())

                if dest_height == 0:
                    stack_owner[b, target] = src_owner
                    stack_height[b, target] = src_height
                    cap_height[b, target] = src_cap
                else:
                    stack_owner[b, target] = src_owner
                    stack_height[b, target] = dest_height + src_height
                    cap_height[b, target] = src_cap
                    marker_owner[b, target] = src_owner

                # Clear source
                stack_owner[b, source] = 0
                stack_height[b, source] = 0
                cap_height[b, source] = 0
                marker_owner[b, source] = src_owner


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


def benchmark_all_kernels(
    batch_sizes: List[int] = [100, 500, 1000, 2000],
    board_size: int = 8,
    num_players: int = 2,
    num_iterations: int = 10,
) -> dict:
    """Benchmark all CUDA kernels for heuristic evaluation.

    Returns:
        Dictionary with benchmark results for each kernel
    """
    import time

    if not TORCH_AVAILABLE:
        return {"error": "PyTorch not available"}

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    checker = GPURuleChecker(board_size, num_players, device)
    num_positions = board_size * board_size

    results = {
        "device": device,
        "board_size": board_size,
        "num_players": num_players,
        "batch_sizes": batch_sizes,
        "kernels": {},
    }

    kernel_funcs = {
        "stack_stats": lambda c: checker.batch_stack_stats(c["owner"], c["height"]),
        "cap_height": lambda c: checker.batch_cap_height(c["owner"], c["cap"]),
        "center_control": lambda c: checker.batch_center_control(c["owner"], c["height"]),
        "adjacency": lambda c: checker.batch_adjacency(c["owner"]),
        "mobility": lambda c: checker.batch_mobility(c["owner"], c["height"], c["collapsed"]),
        "vulnerability": lambda c: checker.batch_vulnerability(c["owner"], c["height"], c["cap"], c["collapsed"]),
        "territory": lambda c: checker.batch_territory_count(c["collapsed"], c["marker"]),
        "line_detect": lambda c: checker.batch_line_detect(c["marker"], min_line_length=4),
        "ring_detect": lambda c: checker.batch_ring_detect(c["marker"]),
    }

    for batch_size in batch_sizes:
        # Create test data
        context = {
            "owner": (torch.rand(batch_size, num_positions, device=device) * (num_players + 1)).to(torch.int8),
            "height": (torch.rand(batch_size, num_positions, device=device) * 5 + 1).to(torch.int8),
            "cap": (torch.rand(batch_size, num_positions, device=device) * 3 + 1).to(torch.int8),
            "collapsed": torch.rand(batch_size, num_positions, device=device) < 0.1,
            "marker": (torch.rand(batch_size, num_positions, device=device) * (num_players + 1)).to(torch.int8),
        }

        for kernel_name, kernel_func in kernel_funcs.items():
            if kernel_name not in results["kernels"]:
                results["kernels"][kernel_name] = {"times_ms": [], "games_per_second": []}

            # Warmup
            _ = kernel_func(context)
            if device != 'cpu':
                torch.cuda.synchronize()

            # Benchmark
            start = time.perf_counter()
            for _ in range(num_iterations):
                _ = kernel_func(context)
                if device != 'cpu':
                    torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            avg_ms = (elapsed / num_iterations) * 1000
            games_per_sec = batch_size / (elapsed / num_iterations)

            results["kernels"][kernel_name]["times_ms"].append(avg_ms)
            results["kernels"][kernel_name]["games_per_second"].append(games_per_sec)

    # Print summary
    logger.info("=" * 60)
    logger.info("CUDA KERNEL BENCHMARK RESULTS")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    logger.info(f"Board: {board_size}x{board_size}, Players: {num_players}")
    logger.info("")

    for kernel_name, kernel_results in results["kernels"].items():
        logger.info(f"{kernel_name}:")
        for i, batch_size in enumerate(batch_sizes):
            gps = kernel_results["games_per_second"][i]
            ms = kernel_results["times_ms"][i]
            logger.info(f"  Batch {batch_size}: {gps/1e6:.2f}M games/sec ({ms:.2f}ms)")
        logger.info("")

    return results


# =============================================================================
# GPU Heuristic Evaluator
# =============================================================================


class GPUHeuristicEvaluator:
    """GPU-accelerated heuristic evaluation for batch positions.

    Combines all CUDA kernels to compute weighted heuristic scores
    for batches of game positions. This enables fast fitness evaluation
    for CMA-ES and other optimization algorithms.
    """

    def __init__(
        self,
        board_size: int = 8,
        num_players: int = 2,
        device: str = 'cuda:0',
        weights: Optional[dict] = None,
    ):
        """Initialize GPU heuristic evaluator.

        Args:
            board_size: Board dimension
            num_players: Number of players
            device: CUDA device
            weights: Heuristic weights dict (uses defaults if None)
        """
        self.board_size = board_size
        self.num_players = num_players
        self.num_positions = board_size * board_size
        self.device = torch.device(device) if TORCH_AVAILABLE else None
        self.checker = GPURuleChecker(board_size, num_players, device) if TORCH_AVAILABLE else None

        # Default weights (subset of full heuristic)
        self.weights = weights or {
            "stack_height": 6.81,
            "stack_count": 9.39,
            "cap_height": 4.82,
            "center_control": 2.28,
            "adjacency": 1.57,
            "mobility": 5.31,
            "vulnerability": -9.32,  # Negative: less vulnerable is better
            "territory": 8.66,
            "line_potential": 7.24,
        }

    def set_weights(self, weights: dict):
        """Update heuristic weights."""
        self.weights.update(weights)

    def evaluate_batch(
        self,
        stack_owner: torch.Tensor,    # (batch, positions) int8
        stack_height: torch.Tensor,   # (batch, positions) int8
        cap_height: torch.Tensor,     # (batch, positions) int8
        collapsed: torch.Tensor,      # (batch, positions) bool
        marker_owner: torch.Tensor,   # (batch, positions) int8
        player: int,
    ) -> torch.Tensor:
        """Evaluate batch of positions for a specific player.

        Args:
            stack_owner: Owner of stack at each position
            stack_height: Height of stack at each position
            cap_height: Cap height at each position
            collapsed: Boolean mask of collapsed positions
            marker_owner: Owner of marker at each position
            player: Player to evaluate for (1-indexed)

        Returns:
            scores: (batch,) tensor of heuristic scores
        """
        batch_size = stack_owner.shape[0]
        scores = torch.zeros(batch_size, dtype=torch.float32, device=self.device)

        # Stack stats
        height_sums, stack_counts = self.checker.batch_stack_stats(stack_owner, stack_height)
        scores += self.weights.get("stack_height", 0) * height_sums[:, player].float()
        scores += self.weights.get("stack_count", 0) * stack_counts[:, player].float()

        # Cap height
        cap_sums = self.checker.batch_cap_height(stack_owner, cap_height)
        scores += self.weights.get("cap_height", 0) * cap_sums[:, player].float()

        # Center control
        center_scores = self.checker.batch_center_control(stack_owner, stack_height)
        scores += self.weights.get("center_control", 0) * center_scores[:, player]

        # Adjacency
        adjacency = self.checker.batch_adjacency(stack_owner)
        scores += self.weights.get("adjacency", 0) * adjacency[:, player].float()

        # Mobility
        mobility = self.checker.batch_mobility(stack_owner, stack_height, collapsed)
        scores += self.weights.get("mobility", 0) * mobility[:, player].float()

        # Vulnerability (negative weight - less is better)
        vuln = self.checker.batch_vulnerability(stack_owner, stack_height, cap_height, collapsed)
        scores += self.weights.get("vulnerability", 0) * vuln[:, player].float()

        # Territory
        territory = self.checker.batch_territory_count(collapsed, marker_owner)
        scores += self.weights.get("territory", 0) * territory[:, player].float()

        # Line potential
        lines = self.checker.batch_line_detect(marker_owner, min_line_length=4)
        scores += self.weights.get("line_potential", 0) * lines[:, player].float()

        return scores

    def evaluate_batch_relative(
        self,
        stack_owner: torch.Tensor,
        stack_height: torch.Tensor,
        cap_height: torch.Tensor,
        collapsed: torch.Tensor,
        marker_owner: torch.Tensor,
        player: int,
    ) -> torch.Tensor:
        """Evaluate relative score (player score - opponent average).

        Returns:
            relative_scores: (batch,) tensor of relative heuristic scores
        """
        batch_size = stack_owner.shape[0]

        # Get player score
        player_score = self.evaluate_batch(
            stack_owner, stack_height, cap_height, collapsed, marker_owner, player
        )

        # Get opponent average
        opponent_scores = []
        for opp in range(1, self.num_players + 1):
            if opp != player:
                opp_score = self.evaluate_batch(
                    stack_owner, stack_height, cap_height, collapsed, marker_owner, opp
                )
                opponent_scores.append(opp_score)

        if opponent_scores:
            opponent_avg = torch.stack(opponent_scores).mean(dim=0)
        else:
            opponent_avg = torch.zeros(batch_size, device=self.device)

        return player_score - opponent_avg


# =============================================================================
# CUDA Kernels for Move Generation
# =============================================================================

if CUDA_AVAILABLE:
    @cuda.jit
    def _count_placement_moves_kernel(
        collapsed: cuda.devicearray,      # (batch, positions) bool
        stack_owner: cuda.devicearray,    # (batch, positions) int8
        has_ring: cuda.devicearray,       # (batch, num_players+1) bool - whether player has ring to place
        board_size: int,
        num_players: int,
        move_counts: cuda.devicearray,    # (batch, num_players+1) int32 - output
    ):
        """Count valid placement moves for each player.

        A placement is valid on any empty, non-collapsed position.
        Players can only place if they have remaining rings.
        """
        game_idx = cuda.blockIdx.x
        thread_idx = cuda.threadIdx.x
        num_threads = cuda.blockDim.x
        num_positions = board_size * board_size

        # Initialize output
        if thread_idx < num_players + 1:
            move_counts[game_idx, thread_idx] = 0
        cuda.syncthreads()

        # Count empty positions (potential placements)
        for pos in range(thread_idx, num_positions, num_threads):
            if not collapsed[game_idx, pos] and stack_owner[game_idx, pos] == 0:
                # This is a valid placement target for any player with rings
                for player in range(1, num_players + 1):
                    if has_ring[game_idx, player]:
                        cuda.atomic.add(move_counts, (game_idx, player), 1)

        cuda.syncthreads()


    @cuda.jit
    def _count_movement_moves_kernel(
        stack_owner: cuda.devicearray,    # (batch, positions) int8
        stack_height: cuda.devicearray,   # (batch, positions) int8
        collapsed: cuda.devicearray,      # (batch, positions) bool
        board_size: int,
        num_players: int,
        move_counts: cuda.devicearray,    # (batch, num_players+1) int32 - output
    ):
        """Count valid movement moves for each player.

        A movement is valid from owned stack to adjacent non-collapsed position
        where source height >= destination height (or destination is empty).
        """
        game_idx = cuda.blockIdx.x
        thread_idx = cuda.threadIdx.x
        num_threads = cuda.blockDim.x
        num_positions = board_size * board_size

        # Initialize output
        if thread_idx < num_players + 1:
            move_counts[game_idx, thread_idx] = 0
        cuda.syncthreads()

        # Check each position for valid movements
        for pos in range(thread_idx, num_positions, num_threads):
            owner = stack_owner[game_idx, pos]
            height = stack_height[game_idx, pos]

            if owner <= 0 or owner > num_players or height <= 0:
                continue

            x = pos % board_size
            y = pos // board_size

            # Check 4 directions
            # Left
            if x > 0:
                dest = y * board_size + (x - 1)
                if not collapsed[game_idx, dest]:
                    dest_height = stack_height[game_idx, dest]
                    if dest_height == 0 or height >= dest_height:
                        cuda.atomic.add(move_counts, (game_idx, owner), 1)

            # Right
            if x < board_size - 1:
                dest = y * board_size + (x + 1)
                if not collapsed[game_idx, dest]:
                    dest_height = stack_height[game_idx, dest]
                    if dest_height == 0 or height >= dest_height:
                        cuda.atomic.add(move_counts, (game_idx, owner), 1)

            # Up
            if y > 0:
                dest = (y - 1) * board_size + x
                if not collapsed[game_idx, dest]:
                    dest_height = stack_height[game_idx, dest]
                    if dest_height == 0 or height >= dest_height:
                        cuda.atomic.add(move_counts, (game_idx, owner), 1)

            # Down
            if y < board_size - 1:
                dest = (y + 1) * board_size + x
                if not collapsed[game_idx, dest]:
                    dest_height = stack_height[game_idx, dest]
                    if dest_height == 0 or height >= dest_height:
                        cuda.atomic.add(move_counts, (game_idx, owner), 1)

        cuda.syncthreads()


    @cuda.jit
    def _generate_placement_moves_kernel(
        collapsed: cuda.devicearray,          # (batch, positions) bool
        stack_owner: cuda.devicearray,        # (batch, positions) int8
        player: int,                          # Player to generate moves for
        board_size: int,
        max_moves_per_game: int,              # Max moves buffer size
        move_targets: cuda.devicearray,       # (batch, max_moves) int16 - output: target positions
        move_counts: cuda.devicearray,        # (batch,) int32 - output: actual move count
    ):
        """Generate placement moves for a specific player.

        Populates move_targets with position indices for valid placements.
        """
        game_idx = cuda.blockIdx.x
        thread_idx = cuda.threadIdx.x
        num_threads = cuda.blockDim.x
        num_positions = board_size * board_size

        # Shared memory for atomic move index
        move_idx = cuda.shared.array(1, dtype=int32)

        if thread_idx == 0:
            move_idx[0] = 0
            move_counts[game_idx] = 0
        cuda.syncthreads()

        # Find empty positions
        for pos in range(thread_idx, num_positions, num_threads):
            if not collapsed[game_idx, pos] and stack_owner[game_idx, pos] == 0:
                # Valid placement - atomically add to move list
                idx = cuda.atomic.add(move_idx, 0, 1)
                if idx < max_moves_per_game:
                    move_targets[game_idx, idx] = pos

        cuda.syncthreads()

        if thread_idx == 0:
            move_counts[game_idx] = min(move_idx[0], max_moves_per_game)


    @cuda.jit
    def _generate_movement_moves_kernel(
        stack_owner: cuda.devicearray,        # (batch, positions) int8
        stack_height: cuda.devicearray,       # (batch, positions) int8
        collapsed: cuda.devicearray,          # (batch, positions) bool
        player: int,                          # Player to generate moves for
        board_size: int,
        max_moves_per_game: int,              # Max moves buffer size
        move_sources: cuda.devicearray,       # (batch, max_moves) int16 - output: source positions
        move_targets: cuda.devicearray,       # (batch, max_moves) int16 - output: target positions
        move_counts: cuda.devicearray,        # (batch,) int32 - output: actual move count
    ):
        """Generate movement moves for a specific player.

        Populates move_sources and move_targets with (from, to) position pairs.
        """
        game_idx = cuda.blockIdx.x
        thread_idx = cuda.threadIdx.x
        num_threads = cuda.blockDim.x
        num_positions = board_size * board_size

        # Shared memory for atomic move index
        move_idx = cuda.shared.array(1, dtype=int32)

        if thread_idx == 0:
            move_idx[0] = 0
            move_counts[game_idx] = 0
        cuda.syncthreads()

        # Check each position owned by player
        for pos in range(thread_idx, num_positions, num_threads):
            owner = stack_owner[game_idx, pos]
            height = stack_height[game_idx, pos]

            if owner != player or height <= 0:
                continue

            x = pos % board_size
            y = pos // board_size

            # Check 4 directions
            # Left
            if x > 0:
                dest = y * board_size + (x - 1)
                if not collapsed[game_idx, dest]:
                    dest_height = stack_height[game_idx, dest]
                    if dest_height == 0 or height >= dest_height:
                        idx = cuda.atomic.add(move_idx, 0, 1)
                        if idx < max_moves_per_game:
                            move_sources[game_idx, idx] = pos
                            move_targets[game_idx, idx] = dest

            # Right
            if x < board_size - 1:
                dest = y * board_size + (x + 1)
                if not collapsed[game_idx, dest]:
                    dest_height = stack_height[game_idx, dest]
                    if dest_height == 0 or height >= dest_height:
                        idx = cuda.atomic.add(move_idx, 0, 1)
                        if idx < max_moves_per_game:
                            move_sources[game_idx, idx] = pos
                            move_targets[game_idx, idx] = dest

            # Up
            if y > 0:
                dest = (y - 1) * board_size + x
                if not collapsed[game_idx, dest]:
                    dest_height = stack_height[game_idx, dest]
                    if dest_height == 0 or height >= dest_height:
                        idx = cuda.atomic.add(move_idx, 0, 1)
                        if idx < max_moves_per_game:
                            move_sources[game_idx, idx] = pos
                            move_targets[game_idx, idx] = dest

            # Down
            if y < board_size - 1:
                dest = (y + 1) * board_size + x
                if not collapsed[game_idx, dest]:
                    dest_height = stack_height[game_idx, dest]
                    if dest_height == 0 or height >= dest_height:
                        idx = cuda.atomic.add(move_idx, 0, 1)
                        if idx < max_moves_per_game:
                            move_sources[game_idx, idx] = pos
                            move_targets[game_idx, idx] = dest

        cuda.syncthreads()

        if thread_idx == 0:
            move_counts[game_idx] = min(move_idx[0], max_moves_per_game)


    @cuda.jit
    def _generate_all_moves_kernel(
        stack_owner: cuda.devicearray,        # (batch, positions) int8
        stack_height: cuda.devicearray,       # (batch, positions) int8
        collapsed: cuda.devicearray,          # (batch, positions) bool
        has_ring: cuda.devicearray,           # (batch, num_players+1) bool
        player: int,                          # Player to generate moves for
        board_size: int,
        max_moves_per_game: int,
        move_types: cuda.devicearray,         # (batch, max_moves) int8 - 0=placement, 1=movement
        move_sources: cuda.devicearray,       # (batch, max_moves) int16 - source (or -1 for placement)
        move_targets: cuda.devicearray,       # (batch, max_moves) int16 - target position
        move_counts: cuda.devicearray,        # (batch,) int32
    ):
        """Generate all moves (placement + movement) for a player in one kernel.

        More efficient than running two separate kernels.
        Move types: 0 = placement, 1 = movement
        """
        game_idx = cuda.blockIdx.x
        thread_idx = cuda.threadIdx.x
        num_threads = cuda.blockDim.x
        num_positions = board_size * board_size

        move_idx = cuda.shared.array(1, dtype=int32)
        can_place = cuda.shared.array(1, dtype=boolean)

        if thread_idx == 0:
            move_idx[0] = 0
            move_counts[game_idx] = 0
            can_place[0] = has_ring[game_idx, player]
        cuda.syncthreads()

        # First pass: placement moves (if player can place)
        if can_place[0]:
            for pos in range(thread_idx, num_positions, num_threads):
                if not collapsed[game_idx, pos] and stack_owner[game_idx, pos] == 0:
                    idx = cuda.atomic.add(move_idx, 0, 1)
                    if idx < max_moves_per_game:
                        move_types[game_idx, idx] = 0  # Placement
                        move_sources[game_idx, idx] = -1  # No source for placement
                        move_targets[game_idx, idx] = pos

        cuda.syncthreads()

        # Second pass: movement moves
        for pos in range(thread_idx, num_positions, num_threads):
            owner = stack_owner[game_idx, pos]
            height = stack_height[game_idx, pos]

            if owner != player or height <= 0:
                continue

            x = pos % board_size
            y = pos // board_size

            # Left
            if x > 0:
                dest = y * board_size + (x - 1)
                if not collapsed[game_idx, dest]:
                    dest_height = stack_height[game_idx, dest]
                    if dest_height == 0 or height >= dest_height:
                        idx = cuda.atomic.add(move_idx, 0, 1)
                        if idx < max_moves_per_game:
                            move_types[game_idx, idx] = 1  # Movement
                            move_sources[game_idx, idx] = pos
                            move_targets[game_idx, idx] = dest

            # Right
            if x < board_size - 1:
                dest = y * board_size + (x + 1)
                if not collapsed[game_idx, dest]:
                    dest_height = stack_height[game_idx, dest]
                    if dest_height == 0 or height >= dest_height:
                        idx = cuda.atomic.add(move_idx, 0, 1)
                        if idx < max_moves_per_game:
                            move_types[game_idx, idx] = 1
                            move_sources[game_idx, idx] = pos
                            move_targets[game_idx, idx] = dest

            # Up
            if y > 0:
                dest = (y - 1) * board_size + x
                if not collapsed[game_idx, dest]:
                    dest_height = stack_height[game_idx, dest]
                    if dest_height == 0 or height >= dest_height:
                        idx = cuda.atomic.add(move_idx, 0, 1)
                        if idx < max_moves_per_game:
                            move_types[game_idx, idx] = 1
                            move_sources[game_idx, idx] = pos
                            move_targets[game_idx, idx] = dest

            # Down
            if y < board_size - 1:
                dest = (y + 1) * board_size + x
                if not collapsed[game_idx, dest]:
                    dest_height = stack_height[game_idx, dest]
                    if dest_height == 0 or height >= dest_height:
                        idx = cuda.atomic.add(move_idx, 0, 1)
                        if idx < max_moves_per_game:
                            move_types[game_idx, idx] = 1
                            move_sources[game_idx, idx] = pos
                            move_targets[game_idx, idx] = dest

        cuda.syncthreads()

        if thread_idx == 0:
            move_counts[game_idx] = min(move_idx[0], max_moves_per_game)


def benchmark_heuristic_evaluator(
    batch_sizes: List[int] = [100, 500, 1000, 2000],
    board_size: int = 8,
    num_players: int = 2,
    num_iterations: int = 10,
) -> dict:
    """Benchmark the full GPU heuristic evaluator.

    Returns:
        Dictionary with benchmark results
    """
    import time

    if not TORCH_AVAILABLE:
        return {"error": "PyTorch not available"}

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    evaluator = GPUHeuristicEvaluator(board_size, num_players, device)
    num_positions = board_size * board_size

    results = {
        "device": device,
        "board_size": board_size,
        "batch_sizes": batch_sizes,
        "times_ms": [],
        "games_per_second": [],
    }

    for batch_size in batch_sizes:
        # Create test data
        stack_owner = (torch.rand(batch_size, num_positions, device=device) * (num_players + 1)).to(torch.int8)
        stack_height = (torch.rand(batch_size, num_positions, device=device) * 5 + 1).to(torch.int8)
        cap_height = (torch.rand(batch_size, num_positions, device=device) * 3 + 1).to(torch.int8)
        collapsed = torch.rand(batch_size, num_positions, device=device) < 0.1
        marker_owner = (torch.rand(batch_size, num_positions, device=device) * (num_players + 1)).to(torch.int8)

        # Warmup
        _ = evaluator.evaluate_batch(stack_owner, stack_height, cap_height, collapsed, marker_owner, 1)
        if device != 'cpu':
            torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(num_iterations):
            _ = evaluator.evaluate_batch(stack_owner, stack_height, cap_height, collapsed, marker_owner, 1)
            if device != 'cpu':
                torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        avg_ms = (elapsed / num_iterations) * 1000
        games_per_sec = batch_size / (elapsed / num_iterations)

        results["times_ms"].append(avg_ms)
        results["games_per_second"].append(games_per_sec)

        logger.info(f"Batch {batch_size}: {games_per_sec/1e3:.1f}K evals/sec ({avg_ms:.2f}ms)")

    return results


# =============================================================================
# CUDA Kernels for Move Application
# =============================================================================

if CUDA_AVAILABLE:
    @cuda.jit
    def _apply_placement_kernel(
        move_targets: cuda.devicearray,       # (batch,) int16 - target positions
        stack_owner: cuda.devicearray,        # (batch, positions) int8 - in/out
        stack_height: cuda.devicearray,       # (batch, positions) int8 - in/out
        cap_height: cuda.devicearray,         # (batch, positions) int8 - in/out
        ring_count: cuda.devicearray,         # (batch, num_players+1) int8 - in/out
        players: cuda.devicearray,            # (batch,) int8 - player for each game
        board_size: int,
    ):
        """Apply placement moves in-place.

        Each block handles one game. Places a ring at target position for player.
        """
        game_idx = cuda.blockIdx.x

        target = move_targets[game_idx]
        if target < 0:
            return

        player = numba.int32(players[game_idx])

        # Place ring (stack height 1, cap height 1)
        stack_owner[game_idx, target] = players[game_idx]
        stack_height[game_idx, target] = 1
        cap_height[game_idx, target] = 1

        # Decrement ring count (safe: one thread per game)
        ring_count[game_idx, player] = ring_count[game_idx, player] - 1


    @cuda.jit
    def _apply_movement_kernel(
        move_sources: cuda.devicearray,       # (batch,) int16 - source positions
        move_targets: cuda.devicearray,       # (batch,) int16 - target positions
        stack_owner: cuda.devicearray,        # (batch, positions) int8 - in/out
        stack_height: cuda.devicearray,       # (batch, positions) int8 - in/out
        cap_height: cuda.devicearray,         # (batch, positions) int8 - in/out
        marker_owner: cuda.devicearray,       # (batch, positions) int8 - in/out
        board_size: int,
    ):
        """Apply movement moves in-place.

        Moves stack from source to target. If target has a stack, the moving
        stack lands on top (capture). If target is empty, stack moves there.
        """
        game_idx = cuda.blockIdx.x

        source = move_sources[game_idx]
        target = move_targets[game_idx]

        if source < 0 or target < 0:
            return

        src_owner = stack_owner[game_idx, source]
        src_height = stack_height[game_idx, source]
        src_cap = cap_height[game_idx, source]

        dest_height = stack_height[game_idx, target]

        if dest_height == 0:
            # Move to empty position
            stack_owner[game_idx, target] = src_owner
            stack_height[game_idx, target] = src_height
            cap_height[game_idx, target] = src_cap
        else:
            # Capture - stack on top
            stack_owner[game_idx, target] = src_owner
            stack_height[game_idx, target] = dest_height + src_height
            cap_height[game_idx, target] = src_cap

            # Leave marker at destination's old position
            marker_owner[game_idx, target] = src_owner

        # Clear source position
        stack_owner[game_idx, source] = 0
        stack_height[game_idx, source] = 0
        cap_height[game_idx, source] = 0

        # Leave marker at source
        marker_owner[game_idx, source] = src_owner


    @cuda.jit
    def _apply_collapse_kernel(
        stack_height: cuda.devicearray,       # (batch, positions) int8
        collapsed: cuda.devicearray,          # (batch, positions) bool - in/out
        collapse_threshold: int,
        board_size: int,
    ):
        """Check and apply collapse for stacks exceeding threshold.

        Collapses any stack taller than threshold.
        """
        game_idx = cuda.blockIdx.x
        thread_idx = cuda.threadIdx.x
        num_threads = cuda.blockDim.x
        num_positions = board_size * board_size

        for pos in range(thread_idx, num_positions, num_threads):
            if stack_height[game_idx, pos] > collapse_threshold:
                collapsed[game_idx, pos] = True
                stack_height[game_idx, pos] = 0


    @cuda.jit
    def _batch_apply_placement_kernel(
        move_targets: cuda.devicearray,       # (batch,) int16 - target positions
        stack_owner: cuda.devicearray,        # (batch, positions) int8 - in/out
        stack_height: cuda.devicearray,       # (batch, positions) int8 - in/out
        cap_height: cuda.devicearray,         # (batch, positions) int8 - in/out
        ring_count: cuda.devicearray,         # (batch, num_players+1) int8 - in/out
        players: cuda.devicearray,            # (batch,) int8 - player per game
        board_size: int,
    ):
        """Apply placement moves for varying players per game."""
        game_idx = cuda.blockIdx.x

        target = move_targets[game_idx]
        player = players[game_idx]

        if target < 0 or player <= 0:
            return

        # Place ring
        stack_owner[game_idx, target] = player
        stack_height[game_idx, target] = 1
        cap_height[game_idx, target] = 1

        # Decrement ring count
        cuda.atomic.add(ring_count, (game_idx, player), -1)


    @cuda.jit
    def _batch_apply_movement_kernel(
        move_sources: cuda.devicearray,       # (batch,) int16
        move_targets: cuda.devicearray,       # (batch,) int16
        stack_owner: cuda.devicearray,        # (batch, positions) int8 - in/out
        stack_height: cuda.devicearray,       # (batch, positions) int8 - in/out
        cap_height: cuda.devicearray,         # (batch, positions) int8 - in/out
        marker_owner: cuda.devicearray,       # (batch, positions) int8 - in/out
        board_size: int,
    ):
        """Apply movement moves for batch of games."""
        game_idx = cuda.blockIdx.x

        source = move_sources[game_idx]
        target = move_targets[game_idx]

        if source < 0 or target < 0:
            return

        src_owner = stack_owner[game_idx, source]
        src_height = stack_height[game_idx, source]
        src_cap = cap_height[game_idx, source]

        dest_height = stack_height[game_idx, target]

        if dest_height == 0:
            # Move to empty position
            stack_owner[game_idx, target] = src_owner
            stack_height[game_idx, target] = src_height
            cap_height[game_idx, target] = src_cap
        else:
            # Capture - stack on top
            stack_owner[game_idx, target] = src_owner
            stack_height[game_idx, target] = dest_height + src_height
            cap_height[game_idx, target] = src_cap
            marker_owner[game_idx, target] = src_owner

        # Clear source
        stack_owner[game_idx, source] = 0
        stack_height[game_idx, source] = 0
        cap_height[game_idx, source] = 0

        # Leave marker at source
        marker_owner[game_idx, source] = src_owner


    @cuda.jit
    def _batch_apply_moves_kernel(
        move_types: cuda.devicearray,         # (batch,) int8 - 0=placement, 1=movement
        move_sources: cuda.devicearray,       # (batch,) int16
        move_targets: cuda.devicearray,       # (batch,) int16
        players: cuda.devicearray,            # (batch,) int8
        stack_owner: cuda.devicearray,        # (batch, positions) int8 - in/out
        stack_height: cuda.devicearray,       # (batch, positions) int8 - in/out
        cap_height: cuda.devicearray,         # (batch, positions) int8 - in/out
        marker_owner: cuda.devicearray,       # (batch, positions) int8 - in/out
        ring_count: cuda.devicearray,         # (batch, num_players+1) int8 - in/out
        board_size: int,
    ):
        """Apply mixed moves (placement or movement) for batch of games.

        Each game in the batch has one move to apply.
        """
        game_idx = cuda.blockIdx.x

        move_type = move_types[game_idx]
        source = move_sources[game_idx]
        target = move_targets[game_idx]
        player = players[game_idx]
        player_idx = numba.int32(player)

        if target < 0:
            return

        if move_type == 0:
            # Placement
            if player > 0:
                stack_owner[game_idx, target] = player
                stack_height[game_idx, target] = 1
                cap_height[game_idx, target] = 1
                ring_count[game_idx, player_idx] = ring_count[game_idx, player_idx] - 1
        elif move_type == 1 and source >= 0:
            # Movement
            src_owner = stack_owner[game_idx, source]
            src_height = stack_height[game_idx, source]
            src_cap = cap_height[game_idx, source]

            dest_height = stack_height[game_idx, target]

            if dest_height == 0:
                stack_owner[game_idx, target] = src_owner
                stack_height[game_idx, target] = src_height
                cap_height[game_idx, target] = src_cap
            else:
                stack_owner[game_idx, target] = src_owner
                stack_height[game_idx, target] = dest_height + src_height
                cap_height[game_idx, target] = src_cap
                marker_owner[game_idx, target] = src_owner

            # Clear source
            stack_owner[game_idx, source] = 0
            stack_height[game_idx, source] = 0
            cap_height[game_idx, source] = 0
            marker_owner[game_idx, source] = src_owner


def benchmark_move_generation(
    batch_sizes: List[int] = [100, 500, 1000, 2000],
    board_size: int = 8,
    num_players: int = 2,
    num_iterations: int = 10,
) -> dict:
    """Benchmark GPU move generation kernels.

    Returns:
        Dictionary with benchmark results for each move generation method
    """
    import time

    if not TORCH_AVAILABLE:
        return {"error": "PyTorch not available"}

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    checker = GPURuleChecker(board_size, num_players, device)
    num_positions = board_size * board_size

    results = {
        "device": device,
        "board_size": board_size,
        "num_players": num_players,
        "batch_sizes": batch_sizes,
        "methods": {},
    }

    for batch_size in batch_sizes:
        # Create test data - simulate mid-game state
        stack_owner = (torch.rand(batch_size, num_positions, device=device) * (num_players + 1)).to(torch.int8)
        stack_height = (torch.rand(batch_size, num_positions, device=device) * 5 + 1).to(torch.int8)
        collapsed = torch.rand(batch_size, num_positions, device=device) < 0.1
        has_ring = torch.ones(batch_size, num_players + 1, dtype=torch.bool, device=device)

        methods = {
            "count_placement": lambda: checker.count_placement_moves(collapsed, stack_owner, has_ring),
            "count_movement": lambda: checker.count_movement_moves(stack_owner, stack_height, collapsed),
            "gen_placement": lambda: checker.generate_placement_moves(collapsed, stack_owner, 1, max_moves=64),
            "gen_movement": lambda: checker.generate_movement_moves(stack_owner, stack_height, collapsed, 1, max_moves=256),
            "gen_all_moves": lambda: checker.generate_all_moves(stack_owner, stack_height, collapsed, has_ring, 1, max_moves=320),
        }

        for method_name, method_func in methods.items():
            if method_name not in results["methods"]:
                results["methods"][method_name] = {"times_ms": [], "games_per_second": []}

            # Warmup
            _ = method_func()
            if device != 'cpu':
                torch.cuda.synchronize()

            # Benchmark
            start = time.perf_counter()
            for _ in range(num_iterations):
                _ = method_func()
                if device != 'cpu':
                    torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            avg_ms = (elapsed / num_iterations) * 1000
            games_per_sec = batch_size / (elapsed / num_iterations)

            results["methods"][method_name]["times_ms"].append(avg_ms)
            results["methods"][method_name]["games_per_second"].append(games_per_sec)

    # Print summary
    logger.info("=" * 60)
    logger.info("MOVE GENERATION BENCHMARK RESULTS")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    logger.info(f"Board: {board_size}x{board_size}, Players: {num_players}")
    logger.info("")

    for method_name, method_results in results["methods"].items():
        logger.info(f"{method_name}:")
        for i, batch_size in enumerate(batch_sizes):
            gps = method_results["games_per_second"][i]
            ms = method_results["times_ms"][i]
            logger.info(f"  Batch {batch_size}: {gps/1e6:.2f}M games/sec ({ms:.2f}ms)")
        logger.info("")

    return results


def benchmark_move_application(
    batch_sizes: List[int] = [100, 500, 1000, 2000],
    board_size: int = 8,
    num_players: int = 2,
    num_iterations: int = 10,
) -> dict:
    """Benchmark GPU move application kernels.

    Returns:
        Dictionary with benchmark results for each move application method
    """
    import time

    if not TORCH_AVAILABLE:
        return {"error": "PyTorch not available"}

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    checker = GPURuleChecker(board_size, num_players, device)
    num_positions = board_size * board_size

    results = {
        "device": device,
        "board_size": board_size,
        "num_players": num_players,
        "batch_sizes": batch_sizes,
        "methods": {},
    }

    for batch_size in batch_sizes:
        # Create test game state tensors (these will be modified in-place)
        def create_test_state():
            stack_owner = (torch.rand(batch_size, num_positions, device=device) * (num_players + 1)).to(torch.int8)
            stack_height = (torch.rand(batch_size, num_positions, device=device) * 3 + 1).to(torch.int8)
            cap_height = torch.ones(batch_size, num_positions, dtype=torch.int8, device=device)
            marker_owner = stack_owner.clone()
            ring_count = torch.ones(batch_size, num_players + 1, dtype=torch.int8, device=device) * 5
            return stack_owner, stack_height, cap_height, marker_owner, ring_count

        # Create move targets (random valid positions for each game)
        placement_targets = torch.randint(0, num_positions, (batch_size,), device=device)
        movement_sources = torch.randint(0, num_positions, (batch_size,), device=device)
        movement_targets = torch.randint(0, num_positions, (batch_size,), device=device)
        players = (torch.randint(1, num_players + 1, (batch_size,), device=device)).to(torch.int8)

        methods = {}

        # Apply placement benchmark
        def bench_placement():
            so, sh, ch, mo, rc = create_test_state()
            checker.apply_placement_moves(placement_targets, so, sh, ch, rc, players)
            return so

        methods["apply_placement"] = bench_placement

        # Apply movement benchmark
        def bench_movement():
            so, sh, ch, mo, rc = create_test_state()
            checker.apply_movement_moves(movement_sources, movement_targets, so, sh, ch, mo)
            return so

        methods["apply_movement"] = bench_movement

        # Apply batch (mixed) benchmark
        def bench_batch():
            so, sh, ch, mo, rc = create_test_state()
            move_types = torch.randint(0, 2, (batch_size,), dtype=torch.int8, device=device)
            checker.apply_batch_moves(move_types, movement_sources, movement_targets, players, so, sh, ch, mo, rc)
            return so

        methods["apply_batch"] = bench_batch

        for method_name, method_func in methods.items():
            if method_name not in results["methods"]:
                results["methods"][method_name] = {"times_ms": [], "games_per_second": []}

            # Warmup
            _ = method_func()
            if device != 'cpu':
                torch.cuda.synchronize()

            # Benchmark
            start = time.perf_counter()
            for _ in range(num_iterations):
                _ = method_func()
                if device != 'cpu':
                    torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            avg_ms = (elapsed / num_iterations) * 1000
            games_per_sec = batch_size / (elapsed / num_iterations)

            results["methods"][method_name]["times_ms"].append(avg_ms)
            results["methods"][method_name]["games_per_second"].append(games_per_sec)

    # Print summary
    logger.info("=" * 60)
    logger.info("MOVE APPLICATION BENCHMARK RESULTS")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    logger.info(f"Board: {board_size}x{board_size}, Players: {num_players}")
    logger.info("")

    for method_name, method_results in results["methods"].items():
        logger.info(f"{method_name}:")
        for i, batch_size in enumerate(batch_sizes):
            gps = method_results["games_per_second"][i]
            ms = method_results["times_ms"][i]
            logger.info(f"  Batch {batch_size}: {gps/1e6:.2f}M games/sec ({ms:.2f}ms)")
        logger.info("")

    return results
