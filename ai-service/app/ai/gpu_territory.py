"""GPU territory processing for parallel games.

This module provides territory processing functions for the GPU parallel games
system. Extracted from gpu_parallel_games.py for modularity.

December 2025: Extracted as part of R9 refactoring.

Territory processing per RR-CANON-R140-R146:
- R140: Find all maximal regions of non-collapsed cells
- R141: Check physical disconnection (single-color barrier)
- R142: Check color-disconnection (RegionColors ⊂ ActiveColors)
- R143: Self-elimination prerequisite
- R145: Region collapse and elimination
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from .gpu_parallel_games import BatchGameState


# =============================================================================
# Territory Processing (RR-CANON-R140-R146)
# =============================================================================

# Neighbor offsets for different board types
# Square boards: 4-connectivity (von Neumann)
SQUARE_DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Hexagonal boards in offset coordinates (odd-r layout):
# Even rows have different neighbor offsets than odd rows
HEX_EVEN_ROW_DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, 1)]
HEX_ODD_ROW_DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1)]


def _get_neighbors(
    y: int,
    x: int,
    board_size: int,
    is_hex: bool,
) -> list[tuple[int, int]]:
    """Get valid neighbor positions for a cell.

    Args:
        y: Row coordinate
        x: Column coordinate
        board_size: Size of the board grid
        is_hex: True for hexagonal boards, False for square boards

    Returns:
        List of (ny, nx) neighbor positions within board bounds
    """
    if is_hex:
        # Hexagonal boards use offset coordinates with row-dependent neighbors
        directions = HEX_EVEN_ROW_DIRECTIONS if y % 2 == 0 else HEX_ODD_ROW_DIRECTIONS
    else:
        directions = SQUARE_DIRECTIONS

    neighbors = []
    for dy, dx in directions:
        ny, nx = y + dy, x + dx
        if 0 <= ny < board_size and 0 <= nx < board_size:
            neighbors.append((ny, nx))
    return neighbors


def _is_hex_board(board_size: int) -> bool:
    """Check if board size indicates a hexagonal board.

    Hexagonal boards use sizes 25 (canonical hex) or 9 (hex8).
    Square boards use sizes 8 or 19.
    """
    return board_size in (25, 9)


def _find_eligible_territory_cap(
    state: BatchGameState,
    game_idx: int,
    player: int,
    excluded_positions: set | None = None,
) -> tuple[int, int, int] | None:
    """Find an eligible stack for territory self-elimination.

    Per RR-CANON-R145: All controlled stacks outside the region are eligible,
    including height-1 standalone rings.

    Optimized 2025-12-13: Use numpy to find eligible stack without .item() calls.

    Args:
        state: BatchGameState
        game_idx: Game index
        player: Player performing territory processing
        excluded_positions: Set of (y, x) positions to exclude (e.g., in region)

    Returns:
        Tuple of (y, x, cap_height) or None if no eligible stack
    """
    board_size = state.board_size
    if excluded_positions is None:
        excluded_positions = set()

    # Use numpy to find eligible stacks (avoids .item() per cell)
    stack_owner_np = state.stack_owner[game_idx].cpu().numpy()
    stack_height_np = state.stack_height[game_idx].cpu().numpy()

    # Per RR-CANON-R145: All controlled stacks are eligible (including height-1)
    # Eligible: player owns stack AND height >= 1
    eligible = (stack_owner_np == player) & (stack_height_np >= 1)

    # Apply exclusions if any
    if excluded_positions:
        for y, x in excluded_positions:
            if 0 <= y < board_size and 0 <= x < board_size:
                eligible[y, x] = False

    positions = np.argwhere(eligible)

    if len(positions) == 0:
        return None

    # Take first eligible position
    y, x = int(positions[0, 0]), int(positions[0, 1])
    height = int(stack_height_np[y, x])

    return (y, x, height)


def _find_all_regions(
    state: BatchGameState,
    game_idx: int,
) -> list[set[tuple[int, int]]]:
    """Find all maximal connected regions of non-collapsed cells (R140).

    Uses BFS to discover all connected regions of non-collapsed cells.
    A region is a maximal set of non-collapsed cells where each cell is connected
    to at least one other cell in the region.

    December 2025: Fixed to use proper neighbor connectivity:
    - Square boards: 4-connectivity (von Neumann)
    - Hexagonal boards: 6-connectivity with row-dependent offsets

    Optimized 2025-12-13: Use deque for O(1) queue ops and numpy for visited tracking.

    Args:
        state: BatchGameState
        game_idx: Game index

    Returns:
        List of regions, where each region is a set of (y, x) positions
    """
    board_size = state.board_size
    g = game_idx
    is_hex = _is_hex_board(board_size)

    # Non-collapsed cells are those that are not territory (collapsed spaces)
    non_collapsed = ~state.is_collapsed[g].cpu().numpy()

    # Use numpy array for visited tracking (faster than Python list of lists)
    visited = np.zeros((board_size, board_size), dtype=np.bool_)
    regions = []

    for start_y in range(board_size):
        for start_x in range(board_size):
            if visited[start_y, start_x] or not non_collapsed[start_y, start_x]:
                continue

            # BFS to find connected region using deque for O(1) popleft
            region = set()
            queue = deque([(start_y, start_x)])
            visited[start_y, start_x] = True

            while queue:
                y, x = queue.popleft()  # O(1) instead of O(n) with list.pop(0)
                region.add((y, x))

                # Use board-type-appropriate neighbor connectivity
                for ny, nx in _get_neighbors(y, x, board_size, is_hex):
                    if not visited[ny, nx] and non_collapsed[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))

            if region:
                regions.append(region)

    return regions


def _is_physically_disconnected(
    state: BatchGameState,
    game_idx: int,
    region: set[tuple[int, int]],
) -> tuple[bool, int | None]:
    """Check if a region is physically disconnected per R141.

    A region R is physically disconnected if every path from any cell in R to
    any non-collapsed cell outside R must cross:
    - Collapsed spaces (any color), and/or
    - Board edge (off-board), and/or
    - Markers belonging to exactly ONE player B (the border color)

    December 2025: Fixed to use proper neighbor connectivity for hexagonal boards.

    Optimized 2025-12-13: Use numpy arrays for membership checks.

    Args:
        state: BatchGameState
        game_idx: Game index
        region: Set of (y, x) positions in the region

    Returns:
        Tuple of (is_disconnected, border_player) where border_player is the
        single player B whose markers form the barrier (or None if not disconnected)
    """
    board_size = state.board_size
    g = game_idx
    is_hex = _is_hex_board(board_size)

    # Pre-extract all numpy arrays at once
    non_collapsed = ~state.is_collapsed[g].cpu().numpy()
    marker_owner_np = state.marker_owner[g].cpu().numpy() if hasattr(state, 'marker_owner') else None

    # Create numpy mask for region membership (faster than set for dense checks)
    region_mask = np.zeros((board_size, board_size), dtype=np.bool_)
    for y, x in region:
        region_mask[y, x] = True

    # Find outside_non_collapsed using numpy vectorized operation
    outside_mask = non_collapsed & ~region_mask
    outside_positions = np.argwhere(outside_mask)

    # If no cells outside region, region spans entire non-collapsed board
    if len(outside_positions) == 0:
        return (False, None)

    # BFS from region boundary to check what separates region from outside
    blocking_marker_players = set()

    # Find region boundary cells (region cells adjacent to non-region cells or board edge)
    region_boundary = set()
    for y, x in region:
        neighbors = _get_neighbors(y, x, board_size, is_hex)
        # Check if any neighbor is outside region or if we're at board edge
        has_outside_neighbor = any(not region_mask[ny, nx] for ny, nx in neighbors)
        # For hex boards, also check if we have fewer than expected neighbors (at edge)
        expected_neighbors = 6 if is_hex else 4
        at_edge = len(neighbors) < expected_neighbors
        if has_outside_neighbor or at_edge:
            region_boundary.add((y, x))

    # Check what separates region from outside by examining all neighbors of boundary cells
    # Use numpy array for visited tracking (faster than set for dense boards)
    visited = np.copy(region_mask)
    can_reach_outside = False

    for y, x in region_boundary:
        # Use board-type-appropriate neighbor connectivity
        for ny, nx in _get_neighbors(y, x, board_size, is_hex):
            if visited[ny, nx]:
                continue

            # Mark as visited to avoid rechecking
            visited[ny, nx] = True

            # Collapsed space - counts as barrier
            if not non_collapsed[ny, nx]:
                continue

            # Non-collapsed cell outside region
            if outside_mask[ny, nx]:
                # Check if there's a marker barrier
                cell_marker_owner = 0
                if marker_owner_np is not None:
                    cell_marker_owner = marker_owner_np[ny, nx]

                if cell_marker_owner > 0:
                    blocking_marker_players.add(int(cell_marker_owner))
                else:
                    # Empty cell or stack - we can reach outside without marker barrier
                    can_reach_outside = True

    # If we can reach outside directly (without crossing markers), not disconnected
    # Note: Even if there are ALSO markers in other directions, having ANY escape route
    # means the region is not physically disconnected.
    if can_reach_outside:
        return (False, None)

    # If blocking markers belong to multiple players, not physically disconnected
    if len(blocking_marker_players) > 1:
        return (False, None)

    # If exactly one player's markers form the barrier
    if len(blocking_marker_players) == 1:
        border_player = blocking_marker_players.pop()
        return (True, border_player)

    # Edge case: region is isolated by collapsed spaces and/or board edges only
    # This is physically disconnected with no border player
    # For territory purposes, any player who can claim it may do so
    return (True, None)


def _is_color_disconnected(
    state: BatchGameState,
    game_idx: int,
    region: set[tuple[int, int]],
) -> bool:
    """Check if a region is color-disconnected per R142.

    R is color-disconnected if RegionColors is a strict subset of ActiveColors.
    - ActiveColors: players with at least one ring anywhere on the board (any stack)
    - RegionColors: players controlling at least one stack (by top ring) in R

    Empty regions (no stacks) have RegionColors = empty set, which is always a strict
    subset of any non-empty ActiveColors, so they satisfy color-disconnection.

    Optimized 2025-12-13: Use numpy to compute colors without .item() calls.

    Args:
        state: BatchGameState
        game_idx: Game index
        region: Set of (y, x) positions in the region

    Returns:
        True if region is color-disconnected (eligible for processing)
    """
    g = game_idx

    # Use numpy to avoid .item() calls in board scan
    stack_owner_np = state.stack_owner[g].cpu().numpy()
    stack_height_np = state.stack_height[g].cpu().numpy()

    # Compute ActiveColors: unique owners with height > 0 across entire board
    active_mask = (stack_owner_np > 0) & (stack_height_np > 0)
    active_colors = set(stack_owner_np[active_mask].tolist())

    # If no active colors (empty board), no territory processing possible
    if not active_colors:
        return False

    # Compute RegionColors: players controlling stacks in the region
    region_colors = set()
    for y, x in region:
        owner = stack_owner_np[y, x]
        height = stack_height_np[y, x]
        if owner > 0 and height > 0:
            region_colors.add(int(owner))

    # R is color-disconnected if RegionColors < ActiveColors (strict subset)
    # This means RegionColors != ActiveColors AND RegionColors is subset of ActiveColors
    # Empty set is always a strict subset of non-empty set

    is_strict_subset = region_colors < active_colors  # Python set comparison
    return is_strict_subset


def compute_territory_batch(
    state: BatchGameState,
    game_mask: torch.Tensor | None = None,
    current_player_only: bool = False,
) -> dict[int, list[tuple[int, int, int]]]:
    """Compute and update territory claims (in-place).

    Per RR-CANON-R140-R146:
    - R140: Find all maximal regions of non-collapsed cells
    - R141: Check physical disconnection (all blocking markers belong to ONE player)
    - R142: Check color-disconnection (RegionColors is strict subset of ActiveColors)
    - R143: Self-elimination prerequisite (player must have eligible cap outside)
    - R145: Region collapse and elimination (collapse interior + border markers)

    This implementation correctly handles:
    1. Regions divided by collapsed spaces or single-color marker lines
    2. The single-color boundary requirement (R141)
    3. The color-disconnection criterion (R142)

    Cap eligibility is checked per RR-CANON-R145: all controlled stacks
    (including height-1 standalone rings) are eligible for territory elimination cost.

    Args:
        state: BatchGameState to modify
        game_mask: Mask of games to process
        current_player_only: If True, only process regions for the current_player of each game.
            This matches CPU semantics where territory is processed turn-by-turn.

    Returns:
        Dictionary mapping game_idx to list of (player, y, x) elimination positions.
        Each entry represents a self-elimination performed for a region.
    """
    elimination_positions: dict[int, list[tuple[int, int, int]]] = {}
    batch_size = state.batch_size
    board_size = state.board_size
    is_hex = _is_hex_board(board_size)

    if game_mask is None:
        game_mask = state.get_active_mask()

    for g in range(batch_size):
        if not game_mask[g]:
            continue

        # Determine which players to process for
        if current_player_only:
            target_players = [int(state.current_player[g].item())]
        else:
            target_players = list(range(1, state.num_players + 1))

        # R140: Find all maximal regions of non-collapsed cells
        all_regions = _find_all_regions(state, g)

        # If only one region, no territory processing possible
        # (entire non-collapsed board is connected)
        if len(all_regions) <= 1:
            continue

        # Pre-extract game arrays as numpy to avoid .item() calls in loops
        # (Optimized 2025-12-13)
        stack_height_np = state.stack_height[g].cpu().numpy()
        stack_owner_np = state.stack_owner[g].cpu().numpy()
        is_collapsed_np = state.is_collapsed[g].cpu().numpy()
        marker_owner_np = state.marker_owner[g].cpu().numpy() if hasattr(state, 'marker_owner') else None

        # Process each region
        # Track which regions have been processed to avoid double-processing
        processed_regions = set()

        # Iterate until no more regions can be processed
        # (processing a region may create new disconnected regions)
        max_iterations = 10  # Safety limit
        for _iteration in range(max_iterations):
            found_processable = False

            for region_idx, region in enumerate(all_regions):
                if region_idx in processed_regions:
                    continue

                # R141: Check physical disconnection
                is_disconnected, border_player = _is_physically_disconnected(state, g, region)
                if not is_disconnected:
                    continue

                # R142: Check color-disconnection
                if not _is_color_disconnected(state, g, region):
                    continue

                # Region is both physically and color-disconnected
                # Determine which player can process it

                # For each player who could claim this territory
                for player in target_players:
                    # Get positions in the region
                    region_positions = region

                    # R143: Find eligible cap for elimination (outside region)
                    eligible_cap = _find_eligible_territory_cap(
                        state, g, player, excluded_positions=region_positions
                    )

                    if eligible_cap is None:
                        continue

                    # Player can process this region
                    cap_y, cap_x, cap_height = eligible_cap

                    # R145: Process the region
                    # 1. Collapse interior (all cells in region become territory)
                    territory_count = 0
                    for y, x in region:
                        # Remove any stacks - use numpy for reading
                        sh = int(stack_height_np[y, x])
                        if sh > 0:
                            # Track which player lost the rings - use numpy
                            so = int(stack_owner_np[y, x])
                            if so > 0:
                                state.eliminated_rings[g, so] += sh
                            # Player processing territory CAUSED these eliminations (for victory)
                            state.rings_caused_eliminated[g, player] += sh
                            state.stack_height[g, y, x] = 0
                            state.stack_owner[g, y, x] = 0
                            # Update local numpy copy for consistency
                            stack_height_np[y, x] = 0
                            stack_owner_np[y, x] = 0

                        # Collapse the cell - use numpy for reading
                        if not is_collapsed_np[y, x]:
                            state.is_collapsed[g, y, x] = True
                            state.territory_owner[g, y, x] = player
                            territory_count += 1
                            is_collapsed_np[y, x] = True

                    # 2. Collapse border markers of single border color B (if applicable)
                    if border_player is not None and marker_owner_np is not None:
                        # Convert region to set once for O(1) lookup
                        region_set = set(region) if not isinstance(region, set) else region
                        # Find and collapse border markers - use numpy for reading
                        for y in range(board_size):
                            for x in range(board_size):
                                if marker_owner_np[y, x] == border_player:
                                    # Check if this marker is on the boundary of region
                                    # Use board-type-appropriate neighbor connectivity
                                    is_boundary = any(
                                        (ny, nx) in region_set
                                        for ny, nx in _get_neighbors(y, x, board_size, is_hex)
                                    )

                                    if is_boundary and not is_collapsed_np[y, x]:
                                        state.is_collapsed[g, y, x] = True
                                        state.territory_owner[g, y, x] = player
                                        state.marker_owner[g, y, x] = 0
                                        territory_count += 1
                                        is_collapsed_np[y, x] = True
                                        marker_owner_np[y, x] = 0

                    # 4. Mandatory self-elimination (eliminate cap)
                    state.stack_height[g, cap_y, cap_x] = 0
                    state.stack_owner[g, cap_y, cap_x] = 0
                    stack_height_np[cap_y, cap_x] = 0
                    stack_owner_np[cap_y, cap_x] = 0
                    # Player eliminates own rings for territory cap cost
                    state.eliminated_rings[g, player] += cap_height
                    # Player CAUSED these eliminations (self-elimination counts for victory)
                    state.rings_caused_eliminated[g, player] += cap_height

                    # Track elimination position for move recording
                    if g not in elimination_positions:
                        elimination_positions[g] = []
                    elimination_positions[g].append((player, cap_y, cap_x))

                    # Update territory count
                    state.territory_count[g, player] += territory_count

                    processed_regions.add(region_idx)
                    found_processable = True
                    break  # Move to next region

                if found_processable:
                    break

            if not found_processable:
                break

            # Recompute regions after processing (new disconnections may appear)
            all_regions = _find_all_regions(state, g)
            processed_regions = set()  # Reset since region indices changed
            # Re-extract numpy arrays since state changed
            stack_height_np = state.stack_height[g].cpu().numpy()
            stack_owner_np = state.stack_owner[g].cpu().numpy()
            is_collapsed_np = state.is_collapsed[g].cpu().numpy()
            marker_owner_np = state.marker_owner[g].cpu().numpy() if hasattr(state, 'marker_owner') else None

    return elimination_positions


__all__ = [
    '_find_all_regions',
    '_find_eligible_territory_cap',
    '_is_color_disconnected',
    '_is_physically_disconnected',
    'compute_territory_batch',
]
