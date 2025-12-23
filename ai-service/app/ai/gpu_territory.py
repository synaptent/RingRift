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

    # Get actual cap_height (not full stack height)
    cap_height_np = state.cap_height[game_idx].cpu().numpy()
    cap = int(cap_height_np[y, x])

    return (y, x, cap)


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


def _find_regions_with_border_color(
    state: BatchGameState,
    game_idx: int,
    border_color: int,
) -> list[tuple[set[tuple[int, int]], int]]:
    """Find regions where markers of border_color act as barriers.

    This mirrors CPU's BoardManager._find_regions_with_border_color:
    - Flood-fills regions while treating collapsed spaces AND markers of
      border_color as impassable boundaries
    - Returns regions that could be territory (will be filtered by caller)

    December 2025: Added to match CPU territory detection algorithm.

    Args:
        state: BatchGameState
        game_idx: Game index
        border_color: Player whose markers act as walls during flood-fill

    Returns:
        List of (region, border_color) tuples where region is a set of (y, x) positions
    """
    board_size = state.board_size
    g = game_idx
    is_hex = _is_hex_board(board_size)

    # Pre-extract numpy arrays
    non_collapsed = ~state.is_collapsed[g].cpu().numpy()
    marker_owner_np = state.marker_owner[g].cpu().numpy()

    # A cell is passable if: not collapsed AND not a marker of border_color
    passable = non_collapsed & (marker_owner_np != border_color)

    visited = np.zeros((board_size, board_size), dtype=np.bool_)
    regions = []

    for start_y in range(board_size):
        for start_x in range(board_size):
            if visited[start_y, start_x] or not passable[start_y, start_x]:
                continue

            # BFS flood-fill treating border_color markers as walls
            region = set()
            queue = deque([(start_y, start_x)])
            visited[start_y, start_x] = True

            while queue:
                y, x = queue.popleft()
                region.add((y, x))

                for ny, nx in _get_neighbors(y, x, board_size, is_hex):
                    if not visited[ny, nx] and passable[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))

            if region:
                regions.append((region, border_color))

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

    # Edge case: region is isolated by collapsed spaces and/or board edges only (no markers)
    # Per CPU semantics, this is NOT considered a disconnected territory region.
    # CPU's find_disconnected_regions iterates through marker colors as borders -
    # if there are no blocking markers, the region is not bounded by a single-color barrier.
    # December 2025: Fixed to match CPU behavior - require marker barrier for territory.
    return (False, None)


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
) -> tuple[dict[int, list[tuple[int, int, int, int, int]]], dict[int, tuple[int, int]]]:
    """Compute and update territory claims (in-place).

    Per RR-CANON-R140-R146:
    - R140: Find all maximal regions of non-collapsed cells
    - R141: Check physical disconnection (all blocking markers belong to ONE player)
    - R142: Check color-disconnection (RegionColors is strict subset of ActiveColors)
    - R143: Self-elimination prerequisite (player must have eligible cap outside)
    - R145: Region collapse and elimination (collapse interior + border markers)

    December 2025: Refactored to match CPU's BoardManager.find_disconnected_regions algorithm:
    - For each marker color, find regions treating that color's markers as barriers
    - Check each region for color-disconnection (missing some active player's stacks)
    - This correctly detects territory isolated by single-color marker lines

    Cap eligibility is checked per RR-CANON-R145: all controlled stacks
    (including height-1 standalone rings) are eligible for territory elimination cost.

    Args:
        state: BatchGameState to modify
        game_mask: Mask of games to process
        current_player_only: If True, only process regions for the current_player of each game.
            This matches CPU semantics where territory is processed turn-by-turn.

    Returns:
        Tuple of:
        - Dictionary mapping game_idx to list of (player, region_y, region_x, elim_y, elim_x)
          Each tuple pairs a territory region with its elimination stack for proper CHOOSE+ELIM recording
        - Dictionary mapping game_idx to (y, x) first region representative position (for backwards compat)
    """
    # Changed: territory_moves now includes region position for each elimination
    territory_moves: dict[int, list[tuple[int, int, int, int, int]]] = {}
    region_positions: dict[int, tuple[int, int]] = {}  # First region position per game
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

        # Pre-extract game arrays as numpy to avoid .item() calls in loops
        stack_height_np = state.stack_height[g].cpu().numpy()
        stack_owner_np = state.stack_owner[g].cpu().numpy()
        is_collapsed_np = state.is_collapsed[g].cpu().numpy()
        marker_owner_np = state.marker_owner[g].cpu().numpy() if hasattr(state, 'marker_owner') else None

        # CPU parity: Skip territory detection if only 1 or 0 active players
        # (matches CPU BoardManager.find_disconnected_regions early exit)
        active_mask = (stack_owner_np > 0) & (stack_height_np > 0)
        active_players = set(stack_owner_np[active_mask].tolist())
        if len(active_players) <= 1:
            continue

        # Collect all marker colors on the board (matches CPU algorithm)
        marker_colors = set()
        if marker_owner_np is not None:
            unique_markers = np.unique(marker_owner_np)
            marker_colors = {int(c) for c in unique_markers if c > 0}

        # If no marker colors, no territory detection possible via marker barriers
        # (CPU also requires marker barriers for territory - see _find_regions_with_border_color)
        if not marker_colors:
            continue

        # Find candidate regions for each marker color acting as barrier
        # This matches CPU's iteration through marker_colors
        candidate_regions: list[tuple[set[tuple[int, int]], int]] = []
        for border_color in marker_colors:
            regions_with_border = _find_regions_with_border_color(state, g, border_color)
            candidate_regions.extend(regions_with_border)

        # Filter to color-disconnected regions
        # A region is color-disconnected if RegionColors < ActiveColors (strict subset)
        eligible_regions: list[tuple[set[tuple[int, int]], int]] = []
        for region, border_player in candidate_regions:
            if _is_color_disconnected(state, g, region):
                eligible_regions.append((region, border_player))

        # If no eligible regions, no territory processing
        if not eligible_regions:
            continue

        # Process each eligible region
        # Track processed region representatives to avoid double-processing
        processed_representatives: set[tuple[int, int]] = set()

        max_iterations = 10  # Safety limit
        for _iteration in range(max_iterations):
            found_processable = False

            for region, border_player in eligible_regions:
                # Use first cell as representative to avoid re-processing same region
                rep = min(region)  # deterministic representative
                if rep in processed_representatives:
                    continue

                # For each player who could claim this territory
                for player in target_players:
                    # Get positions in the region (for exclusion check)
                    region_cells = region

                    # R143: Find eligible cap for elimination (outside region)
                    eligible_cap = _find_eligible_territory_cap(
                        state, g, player, excluded_positions=region_cells
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

                    # 2. Collapse border markers (CPU parity: collapses ALL connected markers)
                    # CPU algorithm:
                    #   Step 1: Find seed markers adjacent to region (von Neumann/territory adjacency)
                    #   Step 2: BFS through connected markers using Moore adjacency (8-dir for square)
                    # NOTE: CPU collapses all markers regardless of color, not just border_player.
                    # This differs from canonical spec but we match CPU for parity.
                    if marker_owner_np is not None:
                        region_set = set(region) if not isinstance(region, set) else region

                        # Step 1: Find seed markers adjacent to region (von Neumann = 4-dir)
                        # Territory adjacency uses von Neumann for square boards
                        von_neumann_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                        seed_markers: set[tuple[int, int]] = set()

                        for ry, rx in region:
                            for dy, dx in von_neumann_dirs:
                                ny, nx = ry + dy, rx + dx
                                if 0 <= ny < board_size and 0 <= nx < board_size:
                                    if (ny, nx) not in region_set and marker_owner_np[ny, nx] > 0:
                                        seed_markers.add((ny, nx))

                        # Step 2: BFS through connected markers using Moore adjacency (8-dir)
                        # For hex boards, CPU doesn't expand (matches TS)
                        border_markers = set(seed_markers)
                        if not is_hex and seed_markers:
                            moore_dirs = [
                                (-1, -1), (-1, 0), (-1, 1),
                                (0, -1),           (0, 1),
                                (1, -1),  (1, 0),  (1, 1)
                            ]
                            queue = list(seed_markers)
                            visited = set(seed_markers) | region_set

                            while queue:
                                cy, cx = queue.pop(0)
                                for dy, dx in moore_dirs:
                                    ny, nx = cy + dy, cx + dx
                                    if 0 <= ny < board_size and 0 <= nx < board_size:
                                        if (ny, nx) not in visited:
                                            visited.add((ny, nx))
                                            if marker_owner_np[ny, nx] > 0:
                                                border_markers.add((ny, nx))
                                                queue.append((ny, nx))

                        # Collapse all border markers
                        for y, x in border_markers:
                            if not is_collapsed_np[y, x]:
                                state.is_collapsed[g, y, x] = True
                                state.territory_owner[g, y, x] = player
                                state.marker_owner[g, y, x] = 0
                                territory_count += 1
                                is_collapsed_np[y, x] = True
                                marker_owner_np[y, x] = 0

                    # Track region representative position for CHOOSE_TERRITORY_OPTION recording
                    # Use first cell of region (deterministic) matching CPU's Territory.spaces[0]
                    if g not in region_positions:
                        region_positions[g] = rep  # rep is min(region), gives (y, x)

                    # Update territory count BEFORE checking victory
                    state.territory_count[g, player] += territory_count

                    # Check for territory victory AFTER region collapse but BEFORE self-elimination
                    # Per RR-CANON-R171/R062: territory victory if player has >= min AND > all opponents
                    # If victory achieved, skip self-elimination - game ends immediately (matches CPU)
                    total_spaces = board_size * board_size
                    min_threshold = total_spaces // state.num_players + 1
                    player_territory = int(state.territory_count[g, player].item())
                    opponent_total = 0
                    for opp in range(1, state.num_players + 1):
                        if opp != player:
                            opponent_total += int(state.territory_count[g, opp].item())

                    territory_victory = (player_territory >= min_threshold and player_territory > opponent_total)

                    if not territory_victory:
                        # 4. Mandatory self-elimination (eliminate cap)
                        # Per RR-CANON-R145, this is required when processing a region
                        # Skip only if territory victory achieved (game ends immediately)

                        # Get current stack state
                        current_height = int(state.stack_height[g, cap_y, cap_x].item())
                        new_height = current_height - cap_height

                        # Player eliminates own rings for territory cap cost
                        state.eliminated_rings[g, player] += cap_height
                        # Player CAUSED these eliminations (self-elimination counts for victory)
                        state.rings_caused_eliminated[g, player] += cap_height

                        if new_height > 0:
                            # There are buried rings - transfer ownership
                            ring_under = int(state.ring_under_cap[g, cap_y, cap_x].item())
                            new_owner = ring_under if ring_under > 0 else (1 if player == 2 else 2)

                            state.stack_height[g, cap_y, cap_x] = new_height
                            state.stack_owner[g, cap_y, cap_x] = new_owner
                            stack_height_np[cap_y, cap_x] = new_height
                            stack_owner_np[cap_y, cap_x] = new_owner

                            # Update ring_stack: shift remaining rings down (remove player's cap)
                            # Ring stack is stored bottom-to-top, so we need to keep bottom rings
                            if hasattr(state, 'ring_stack'):
                                for i in range(new_height):
                                    state.ring_stack[g, cap_y, cap_x, i] = state.ring_stack[g, cap_y, cap_x, i].item()
                                for i in range(new_height, current_height):
                                    state.ring_stack[g, cap_y, cap_x, i] = 0

                            # Compute new cap from remaining rings
                            from .gpu_move_generation import compute_cap_from_ring_stack
                            new_cap = compute_cap_from_ring_stack(state, g, cap_y, cap_x)
                            state.cap_height[g, cap_y, cap_x] = new_cap

                            # Update ring_under_cap from buried_at (find next buried player)
                            new_ring_under = 0
                            for p in range(1, state.num_players + 1):
                                if p != new_owner:
                                    if hasattr(state, 'buried_at') and state.buried_at[g, p, cap_y, cap_x].item() > 0:
                                        new_ring_under = p
                                        break
                            state.ring_under_cap[g, cap_y, cap_x] = new_ring_under

                            # Decrement buried_at for eliminated player if they had buried rings
                            if hasattr(state, 'buried_at'):
                                buried_count = int(state.buried_at[g, player, cap_y, cap_x].item())
                                if buried_count > 0:
                                    state.buried_at[g, player, cap_y, cap_x] = 0
                                    state.buried_rings[g, player] = max(0, state.buried_rings[g, player] - buried_count)
                        else:
                            # Stack fully eliminated
                            state.stack_height[g, cap_y, cap_x] = 0
                            state.stack_owner[g, cap_y, cap_x] = 0
                            state.cap_height[g, cap_y, cap_x] = 0
                            state.ring_under_cap[g, cap_y, cap_x] = 0
                            stack_height_np[cap_y, cap_x] = 0
                            stack_owner_np[cap_y, cap_x] = 0
                            if hasattr(state, 'ring_stack'):
                                state.ring_stack[g, cap_y, cap_x, :] = 0

                        # Track territory move for CHOOSE+ELIM recording
                        # Include region position so each elimination is paired with its territory
                        if g not in territory_moves:
                            territory_moves[g] = []
                        # rep is (y, x) tuple from min(region)
                        territory_moves[g].append((player, rep[0], rep[1], cap_y, cap_x))

                    processed_representatives.add(rep)
                    found_processable = True
                    break  # Move to next region (processed one region for this player)

                if found_processable:
                    break

            if not found_processable:
                break

            # Recompute eligible regions after processing (new disconnections may appear)
            # Re-extract numpy arrays since state changed
            stack_height_np = state.stack_height[g].cpu().numpy()
            stack_owner_np = state.stack_owner[g].cpu().numpy()
            is_collapsed_np = state.is_collapsed[g].cpu().numpy()
            marker_owner_np = state.marker_owner[g].cpu().numpy() if hasattr(state, 'marker_owner') else None

            # Recompute marker colors and candidate regions
            marker_colors = set()
            if marker_owner_np is not None:
                unique_markers = np.unique(marker_owner_np)
                marker_colors = {int(c) for c in unique_markers if c > 0}

            if not marker_colors:
                break

            candidate_regions = []
            for border_color in marker_colors:
                regions_with_border = _find_regions_with_border_color(state, g, border_color)
                candidate_regions.extend(regions_with_border)

            eligible_regions = []
            for region, border_player in candidate_regions:
                if _is_color_disconnected(state, g, region):
                    eligible_regions.append((region, border_player))

    return territory_moves, region_positions


__all__ = [
    '_find_all_regions',
    '_find_eligible_territory_cap',
    '_find_regions_with_border_color',
    '_is_color_disconnected',
    '_is_physically_disconnected',
    'compute_territory_batch',
]
