"""GPU line detection and processing for parallel games.

This module provides vectorized line detection and processing for the GPU
parallel games system. Extracted from gpu_parallel_games.py for modularity.

December 2025: Extracted as part of R5 refactoring.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import torch

from .gpu_game_types import DetectedLine, get_required_line_length

if TYPE_CHECKING:
    from .gpu_parallel_games import BatchGameState


# =============================================================================
# Line Detection (RR-CANON-R120)
# =============================================================================


def detect_lines_vectorized(
    state: "BatchGameState",
    player: int,
    game_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Vectorized line detection returning positions mask and line count.

    Per RR-CANON-R120: A line is a sequence of consecutive MARKERS (not stacks)
    of the same player, with length >= required_length.

    This is a fast vectorized implementation for MCTS. It returns:
    1. A mask of positions that are part of valid lines
    2. A count of positions per game that are in lines (proxy for line count)

    Args:
        state: Current batch game state
        player: Player number to detect lines for
        game_mask: Mask of games to check (optional)

    Returns:
        Tuple of:
        - in_line_mask: (batch_size, board_size, board_size) bool tensor
        - line_position_counts: (batch_size,) int tensor with count of line positions
    """
    device = state.device
    batch_size = state.batch_size
    board_size = state.board_size
    required_length = get_required_line_length(board_size, state.num_players)

    if game_mask is None:
        game_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

    # Per RR-CANON-R120: Lines are formed by MARKERS, not stacks
    # A marker at (y,x) can be part of a line only if no stack is there
    player_markers = (
        (state.marker_owner == player) &
        (state.stack_owner == 0) &
        game_mask.view(-1, 1, 1)
    )

    # Output mask: positions that are part of any line
    in_line_mask = torch.zeros_like(player_markers)

    # Check all 4 directions
    # For each direction, we check if there's a sequence of required_length markers

    # Note: Using explicit OR + assignment instead of |= due to MPS device quirk
    # Compute horizontal windows
    if board_size >= required_length:
        # Sum of required_length consecutive horizontal positions
        markers_float = player_markers.float()

        # Horizontal: sum along x-axis
        cumsum_h = markers_float.cumsum(dim=2)
        padded_cumsum_h = torch.cat([
            torch.zeros(batch_size, board_size, 1, device=device),
            cumsum_h
        ], dim=2)
        window_sums_h = padded_cumsum_h[:, :, required_length:] - padded_cumsum_h[:, :, :-required_length]
        complete_windows_h = (window_sums_h == required_length)
        for offset in range(required_length):
            if offset < complete_windows_h.shape[2]:
                start = offset
                end = board_size - required_length + offset + 1
                in_line_mask[:, :, start:end] = in_line_mask[:, :, start:end] | complete_windows_h

        # Vertical: sum along y-axis
        cumsum_v = markers_float.cumsum(dim=1)
        padded_cumsum_v = torch.cat([
            torch.zeros(batch_size, 1, board_size, device=device),
            cumsum_v
        ], dim=1)
        window_sums_v = padded_cumsum_v[:, required_length:, :] - padded_cumsum_v[:, :-required_length, :]
        complete_windows_v = (window_sums_v == required_length)
        for offset in range(required_length):
            if offset < complete_windows_v.shape[1]:
                start = offset
                end = board_size - required_length + offset + 1
                in_line_mask[:, start:end, :] = in_line_mask[:, start:end, :] | complete_windows_v

        # Diagonal (dy=1, dx=1)
        diag_mask = torch.ones(batch_size, board_size, board_size, dtype=torch.bool, device=device)
        for i in range(required_length):
            if i == 0:
                shifted = player_markers
            else:
                shifted = torch.zeros_like(player_markers)
                shifted[:, :-i, :-i] = player_markers[:, i:, i:]
            diag_mask = diag_mask & shifted

        diag_mask[:, board_size - required_length + 1:, :] = False
        diag_mask[:, :, board_size - required_length + 1:] = False

        valid_region = board_size - required_length + 1
        for i in range(required_length):
            y_start, y_end = i, valid_region + i
            x_start, x_end = i, valid_region + i
            in_line_mask[:, y_start:y_end, x_start:x_end] = \
                in_line_mask[:, y_start:y_end, x_start:x_end] | diag_mask[:, :valid_region, :valid_region]

        # Anti-diagonal (dy=1, dx=-1)
        anti_diag_mask = torch.ones(batch_size, board_size, board_size, dtype=torch.bool, device=device)
        for i in range(required_length):
            if i == 0:
                shifted = player_markers
            else:
                shifted = torch.zeros_like(player_markers)
                shifted[:, :-i, i:] = player_markers[:, i:, :-i]
            anti_diag_mask = anti_diag_mask & shifted

        anti_diag_mask[:, board_size - required_length + 1:, :] = False
        anti_diag_mask[:, :, :required_length - 1] = False

        valid_y = board_size - required_length + 1
        valid_x_start = required_length - 1
        for i in range(required_length):
            y_start, y_end = i, valid_y + i
            x_start, x_end = valid_x_start - i, board_size - i
            in_line_mask[:, y_start:y_end, x_start:x_end] = \
                in_line_mask[:, y_start:y_end, x_start:x_end] | anti_diag_mask[:, :valid_y, valid_x_start:]

    # Count line positions per game
    line_position_counts = in_line_mask.view(batch_size, -1).sum(dim=1)

    return in_line_mask, line_position_counts


def has_lines_batch_vectorized(
    state: "BatchGameState",
    player: int,
    game_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fast vectorized check for whether a player has any lines.

    This is the fastest path for MCTS victory checking - just returns
    a boolean tensor indicating which games have lines for this player.

    Args:
        state: Current batch game state
        player: Player number to check
        game_mask: Mask of games to check (optional)

    Returns:
        (batch_size,) bool tensor - True if player has at least one line
    """
    _, line_counts = detect_lines_vectorized(state, player, game_mask)
    return line_counts > 0


def detect_lines_with_metadata(
    state: "BatchGameState",
    player: int,
    game_mask: Optional[torch.Tensor] = None,
) -> List[List[DetectedLine]]:
    """Detect lines with full metadata including overlength status.

    Per RR-CANON-R120: A line for player P is a maximal sequence of positions
    where each position contains a MARKER of P (no stacks, no collapsed spaces).

    Returns structured line data including whether each line is overlength,
    enabling proper Option 1/2 handling per RR-CANON-R122.

    Args:
        state: Current batch game state
        player: Player number to detect lines for
        game_mask: Mask of games to check (optional)

    Returns:
        List of lists of DetectedLine objects, one list per game
    """
    batch_size = state.batch_size
    board_size = state.board_size
    num_players = state.num_players

    required_length = get_required_line_length(board_size, num_players)

    if game_mask is None:
        game_mask = torch.ones(batch_size, dtype=torch.bool, device=state.device)

    lines_per_game: List[List[DetectedLine]] = [[] for _ in range(batch_size)]

    # Early exit: use vectorized detection to quickly identify games WITH lines
    _, line_counts = detect_lines_vectorized(state, player, game_mask)
    games_with_lines = (line_counts > 0) & game_mask

    if not games_with_lines.any():
        return lines_per_game

    # Get marker mask for all games at once (batch, H, W)
    player_markers_batch = (state.marker_owner == player) & (state.stack_owner == 0)

    # 4 directions to check for lines
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

    # Only process games that have lines
    games_to_check = games_with_lines.nonzero(as_tuple=True)[0].tolist()

    for g in games_to_check:
        player_markers = player_markers_batch[g].cpu().numpy()
        assigned = np.zeros((board_size, board_size), dtype=np.bool_)

        for dy, dx in directions:
            for start_y in range(board_size):
                for start_x in range(board_size):
                    if assigned[start_y, start_x]:
                        continue
                    if not player_markers[start_y, start_x]:
                        continue

                    line_positions = [(start_y, start_x)]
                    y, x = start_y + dy, start_x + dx

                    while 0 <= y < board_size and 0 <= x < board_size:
                        if player_markers[y, x] and not assigned[y, x]:
                            line_positions.append((y, x))
                            y, x = y + dy, x + dx
                        else:
                            break

                    if len(line_positions) >= required_length:
                        for pos_y, pos_x in line_positions:
                            assigned[pos_y, pos_x] = True

                        lines_per_game[g].append(DetectedLine(
                            positions=line_positions,
                            length=len(line_positions),
                            is_overlength=(len(line_positions) > required_length),
                            direction=(dy, dx),
                        ))

    return lines_per_game


def detect_lines_batch(
    state: "BatchGameState",
    player: int,
    game_mask: Optional[torch.Tensor] = None,
) -> List[List[Tuple[int, int]]]:
    """Detect lines of consecutive same-owner MARKERS for a player.

    Per RR-CANON-R120: A line for player P is a maximal sequence of positions
    where each position contains a MARKER of P (no stacks, no collapsed spaces).

    Args:
        state: Current batch game state
        player: Player number to detect lines for
        game_mask: Mask of games to check (optional)

    Returns:
        List of lists of (y, x) tuples, one per game, containing all line positions
    """
    lines_with_meta = detect_lines_with_metadata(state, player, game_mask)

    lines_per_game = []
    for game_lines in lines_with_meta:
        all_positions = []
        for line in game_lines:
            all_positions.extend(line.positions)
        lines_per_game.append(all_positions)

    return lines_per_game


# =============================================================================
# Line Processing (RR-CANON-R121-R122)
# =============================================================================


def _eliminate_one_ring_from_any_stack(
    state: "BatchGameState",
    game_idx: int,
    player: int,
) -> bool:
    """Eliminate one ring from any controlled stack.

    Per RR-CANON-R122: Any controlled stack is eligible for line elimination,
    including height-1 standalone rings.

    Args:
        state: BatchGameState to modify
        game_idx: Game index
        player: Player performing elimination

    Returns:
        True if elimination was performed, False if no eligible stack found
    """
    stack_owner_np = state.stack_owner[game_idx].cpu().numpy()
    stack_height_np = state.stack_height[game_idx].cpu().numpy()

    eligible = (stack_owner_np == player) & (stack_height_np > 0)
    positions = np.argwhere(eligible)

    if len(positions) == 0:
        return False

    y, x = int(positions[0, 0]), int(positions[0, 1])
    stack_height = int(stack_height_np[y, x])

    state.stack_height[game_idx, y, x] = stack_height - 1
    state.eliminated_rings[game_idx, player] += 1
    state.rings_caused_eliminated[game_idx, player] += 1

    if stack_height - 1 == 0:
        state.stack_owner[game_idx, y, x] = 0

    return True


def process_lines_batch(
    state: "BatchGameState",
    game_mask: Optional[torch.Tensor] = None,
    option2_probability: float = 0.3,
) -> None:
    """Process formed marker lines for all players (in-place).

    Per RR-CANON-R121-R122:
    - Lines are formed by MARKERS (not stacks)
    - Exact-length lines: Collapse all markers, pay one ring elimination
    - Overlength lines (len > required): Player chooses Option 1 or Option 2
      - Option 1: Collapse ALL markers, pay one ring elimination
      - Option 2: Collapse exactly required_length markers, NO elimination cost

    Args:
        state: BatchGameState to modify
        game_mask: Mask of games to process
        option2_probability: Probability of choosing Option 2 for overlength lines
    """
    batch_size = state.batch_size
    board_size = state.board_size
    device = state.device

    if game_mask is None:
        game_mask = state.get_active_mask()

    required_length = get_required_line_length(board_size, state.num_players)

    max_lines_estimate = 100
    random_vals = torch.rand(max_lines_estimate, device=device).cpu().numpy()
    random_idx = 0

    for p in range(1, state.num_players + 1):
        lines_with_meta = detect_lines_with_metadata(state, p, game_mask)

        for g in range(batch_size):
            if not game_mask[g]:
                continue

            game_lines = lines_with_meta[g]
            if not game_lines:
                continue

            stack_owner_np = state.stack_owner[g].cpu().numpy()

            for line in game_lines:
                positions_to_collapse = line.positions

                if line.is_overlength:
                    use_option2 = random_vals[random_idx % max_lines_estimate] < option2_probability
                    random_idx += 1

                    if use_option2:
                        all_positions = line.positions
                        if len(all_positions) > required_length:
                            indices = torch.randperm(len(all_positions), device=device)[:required_length]
                            indices = indices.sort().values
                            positions_to_collapse = [all_positions[i] for i in indices.tolist()]
                        else:
                            positions_to_collapse = all_positions[:required_length]
                    else:
                        positions_to_collapse = line.positions
                        if (stack_owner_np == p).any():
                            _eliminate_one_ring_from_any_stack(state, g, p)
                else:
                    if (stack_owner_np == p).any():
                        _eliminate_one_ring_from_any_stack(state, g, p)

                for (y, x) in positions_to_collapse:
                    state.marker_owner[g, y, x] = 0
                    state.territory_owner[g, y, x] = p
                    state.is_collapsed[g, y, x] = True
                    state.territory_count[g, p] += 1


__all__ = [
    'detect_lines_vectorized',
    'has_lines_batch_vectorized',
    'detect_lines_with_metadata',
    'detect_lines_batch',
    'process_lines_batch',
]
