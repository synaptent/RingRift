"""GPU move application for parallel games.

This module provides move application functions for the GPU parallel games
system. Extracted from gpu_parallel_games.py for modularity.

December 2025: Extracted as part of R16 refactoring.

Move application per RR-CANON rules:
- R080: Placement moves
- R090-R092: Movement moves
- R100-R103: Capture moves
- R110-R115: Recovery slide moves
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

from .gpu_game_types import MoveType

if TYPE_CHECKING:
    from .gpu_parallel_games import BatchGameState
    from .gpu_move_generation import BatchMoves


# =============================================================================
# Vectorized Apply Functions (for move selection)
# =============================================================================


def apply_capture_moves_vectorized(
    state: "BatchGameState",
    selected_local_idx: torch.Tensor,
    moves: "BatchMoves",
    active_mask: torch.Tensor,
) -> None:
    """Apply capture moves in a vectorized manner.

    This applies selected capture moves for multiple games simultaneously,
    minimizing Python loops and .item() calls.

    Note: Some operations (like path marker flipping) still require iteration
    due to variable-length paths. This is a known limitation documented in
    GPU_PIPELINE_ROADMAP.md Section 2.2 (Irregular Data Access Patterns).

    Args:
        state: BatchGameState to modify
        selected_local_idx: (batch_size,) local move indices
        moves: BatchMoves containing capture moves
        active_mask: (batch_size,) bool tensor of games with captures to apply
    """
    device = state.device
    batch_size = state.batch_size

    # Identify games that have moves to apply
    has_selection = (selected_local_idx >= 0) & active_mask & (moves.moves_per_game > 0)

    if not has_selection.any():
        return

    # Compute global indices for selected moves
    global_idx = moves.move_offsets + selected_local_idx
    global_idx = torch.clamp(global_idx, 0, max(0, moves.total_moves - 1))

    # Get move data for all selected moves at once
    selected_from_y = moves.from_y[global_idx]
    selected_from_x = moves.from_x[global_idx]
    selected_to_y = moves.to_y[global_idx]
    selected_to_x = moves.to_x[global_idx]

    # Get current players for active games
    current_players = state.current_player

    # Apply moves game by game (some operations require iteration due to variable paths)
    # This is the minimal iteration - just for path processing
    game_indices = torch.where(has_selection)[0]

    for g in game_indices.tolist():
        from_y = selected_from_y[g].item()
        from_x = selected_from_x[g].item()
        to_y = selected_to_y[g].item()
        to_x = selected_to_x[g].item()
        player = current_players[g].item()
        mc = int(state.move_count[g].item())
        # Clear must-move constraint after the first movement/capture action
        # following a placement (RR-CANON-R090).
        state.must_move_from_y[g] = -1
        state.must_move_from_x[g] = -1

        # Record in history
        if mc < state.max_history_moves:
            state.move_history[g, mc, 0] = MoveType.CAPTURE
            state.move_history[g, mc, 1] = player
            state.move_history[g, mc, 2] = from_y
            state.move_history[g, mc, 3] = from_x
            state.move_history[g, mc, 4] = to_y
            state.move_history[g, mc, 5] = to_x
        state.move_count[g] += 1

        # Get attacker stack info at origin.
        attacker_height = int(state.stack_height[g, from_y, from_x].item())
        attacker_cap_height = int(state.cap_height[g, from_y, from_x].item())

        # Capture move representation:
        # - (from -> landing) is stored in BatchMoves
        # - The target stack is implicit as the first stack along the ray
        dy = 0 if to_y == from_y else (1 if to_y > from_y else -1)
        dx = 0 if to_x == from_x else (1 if to_x > from_x else -1)
        dist = max(abs(to_y - from_y), abs(to_x - from_x))

        target_y = None
        target_x = None
        for step in range(1, int(dist)):
            check_y = from_y + dy * step
            check_x = from_x + dx * step
            if state.stack_owner[g, check_y, check_x].item() != 0:
                target_y = check_y
                target_x = check_x
                break

        if target_y is None or target_x is None:
            # Defensive fallback: treat this as a movement to landing.
            state.stack_height[g, to_y, to_x] = attacker_height
            state.stack_owner[g, to_y, to_x] = player
            state.cap_height[g, to_y, to_x] = min(attacker_cap_height, attacker_height)
            state.stack_height[g, from_y, from_x] = 0
            state.stack_owner[g, from_y, from_x] = 0
            state.cap_height[g, from_y, from_x] = 0
            state.marker_owner[g, from_y, from_x] = player
            continue

        # Process markers along the full path (RR-CANON-R102 delegates to R092),
        # excluding the implicit target stack cell.
        for step in range(1, int(dist)):
            check_y = from_y + dy * step
            check_x = from_x + dx * step
            if check_y == target_y and check_x == target_x:
                continue

            marker_owner = state.marker_owner[g, check_y, check_x].item()
            if marker_owner == 0:
                continue
            if marker_owner != player:
                state.marker_owner[g, check_y, check_x] = player
                continue

            # Own marker on intermediate cell: collapse to territory.
            state.marker_owner[g, check_y, check_x] = 0
            if not state.is_collapsed[g, check_y, check_x].item():
                state.is_collapsed[g, check_y, check_x] = True
                state.territory_owner[g, check_y, check_x] = player
                state.territory_count[g, player] += 1

        # Landing marker interaction (RR-CANON-R102): remove any marker on landing
        # (do not collapse), then eliminate the top ring of the moving stack's cap.
        dest_marker = state.marker_owner[g, to_y, to_x].item()
        landing_ring_cost = 1 if dest_marker != 0 else 0
        if landing_ring_cost:
            state.marker_owner[g, to_y, to_x] = 0
            state.eliminated_rings[g, player] += 1
            state.rings_caused_eliminated[g, player] += 1

        # Pop the top ring from the implicit target and append it to the bottom
        # of the attacking stack (RR-CANON-R102). We do not store full ring
        # sequences on GPU; we approximate by updating stack/cap metadata and
        # tracking captured rings via buried_rings.
        target_owner = int(state.stack_owner[g, target_y, target_x].item())
        target_height = int(state.stack_height[g, target_y, target_x].item())
        target_cap_height = int(state.cap_height[g, target_y, target_x].item())

        # Target cell should not contain a marker; clear defensively.
        state.marker_owner[g, target_y, target_x] = 0

        new_target_height = max(0, target_height - 1)
        state.stack_height[g, target_y, target_x] = new_target_height
        if new_target_height <= 0:
            state.stack_owner[g, target_y, target_x] = 0
            state.cap_height[g, target_y, target_x] = 0
        else:
            new_target_cap = target_cap_height - 1
            if new_target_cap <= 0:
                new_target_cap = 1
            if new_target_cap > new_target_height:
                new_target_cap = new_target_height
            state.cap_height[g, target_y, target_x] = new_target_cap

        # Track captured ring as "buried" for the ring's owner (when capturing an opponent).
        if target_owner != 0 and target_owner != player:
            state.buried_rings[g, target_owner] += 1

        # Move attacker to landing and apply net height change:
        # +1 captured ring (to bottom) - landing marker elimination cost.
        new_height = attacker_height + 1 - landing_ring_cost
        state.stack_height[g, to_y, to_x] = new_height
        state.stack_owner[g, to_y, to_x] = player

        new_cap = attacker_cap_height - landing_ring_cost
        if new_cap <= 0:
            new_cap = 1
        if new_cap > new_height:
            new_cap = new_height
        state.cap_height[g, to_y, to_x] = new_cap

        # Clear origin stack and leave a departure marker (RR-CANON-R092).
        state.stack_height[g, from_y, from_x] = 0
        state.stack_owner[g, from_y, from_x] = 0
        state.cap_height[g, from_y, from_x] = 0
        state.marker_owner[g, from_y, from_x] = player


def apply_movement_moves_vectorized(
    state: "BatchGameState",
    selected_local_idx: torch.Tensor,
    moves: "BatchMoves",
    active_mask: torch.Tensor,
) -> None:
    """Apply movement moves in a vectorized manner.

    Similar to capture moves but without defender elimination.
    Still requires iteration for path marker processing.
    """
    device = state.device
    batch_size = state.batch_size

    has_selection = (selected_local_idx >= 0) & active_mask & (moves.moves_per_game > 0)

    if not has_selection.any():
        return

    global_idx = moves.move_offsets + selected_local_idx
    global_idx = torch.clamp(global_idx, 0, max(0, moves.total_moves - 1))

    selected_from_y = moves.from_y[global_idx]
    selected_from_x = moves.from_x[global_idx]
    selected_to_y = moves.to_y[global_idx]
    selected_to_x = moves.to_x[global_idx]

    current_players = state.current_player
    game_indices = torch.where(has_selection)[0]

    for g in game_indices.tolist():
        from_y = selected_from_y[g].item()
        from_x = selected_from_x[g].item()
        to_y = selected_to_y[g].item()
        to_x = selected_to_x[g].item()
        player = current_players[g].item()
        mc = int(state.move_count[g].item())
        state.must_move_from_y[g] = -1
        state.must_move_from_x[g] = -1

        # Record in history
        if mc < state.max_history_moves:
            state.move_history[g, mc, 0] = MoveType.MOVEMENT
            state.move_history[g, mc, 1] = player
            state.move_history[g, mc, 2] = from_y
            state.move_history[g, mc, 3] = from_x
            state.move_history[g, mc, 4] = to_y
            state.move_history[g, mc, 5] = to_x
        state.move_count[g] += 1

        moving_height = state.stack_height[g, from_y, from_x].item()
        moving_cap_height = state.cap_height[g, from_y, from_x].item()

        # Process markers along path (RR-CANON-R092):
        # - Flip opponent markers along the path
        # - Collapse own markers on intermediate cells to territory
        dy = 0 if to_y == from_y else (1 if to_y > from_y else -1)
        dx = 0 if to_x == from_x else (1 if to_x > from_x else -1)
        dist = max(abs(to_y - from_y), abs(to_x - from_x))

        for step in range(1, int(dist)):
            check_y = from_y + dy * step
            check_x = from_x + dx * step
            marker_owner = state.marker_owner[g, check_y, check_x].item()
            if marker_owner == 0:
                continue
            if marker_owner != player:
                state.marker_owner[g, check_y, check_x] = player
                continue

            # Own marker on intermediate cell: collapse to territory.
            state.marker_owner[g, check_y, check_x] = 0
            if not state.is_collapsed[g, check_y, check_x].item():
                state.is_collapsed[g, check_y, check_x] = True
                state.territory_owner[g, check_y, check_x] = player
                state.territory_count[g, player] += 1

        # Handle landing on ANY marker (own or opponent):
        # remove the marker (do not collapse), then eliminate the top cap ring.
        dest_marker = state.marker_owner[g, to_y, to_x].item()
        landing_ring_cost = 1 if dest_marker != 0 else 0
        if landing_ring_cost:
            state.marker_owner[g, to_y, to_x] = 0

        # Movement cannot land on stacks; destination is guaranteed empty by move generation.
        new_height = moving_height - landing_ring_cost

        # Track eliminated ring from landing cost
        if landing_ring_cost > 0:
            current_elim = state.eliminated_rings[g, player].item()
            state.eliminated_rings[g, player] = current_elim + landing_ring_cost
            # Player eliminates their own ring for landing cost (self-elimination counts for victory)
            state.rings_caused_eliminated[g, player] += landing_ring_cost

        # Update destination
        final_height = max(0, new_height)
        if final_height <= 0:
            # Landing cost can eliminate the final ring; the destination remains
            # empty (but may still become collapsed if the move landed on a marker).
            state.stack_height[g, to_y, to_x] = 0
            state.stack_owner[g, to_y, to_x] = 0
            state.cap_height[g, to_y, to_x] = 0
        else:
            state.stack_height[g, to_y, to_x] = final_height
            state.stack_owner[g, to_y, to_x] = player
            # Best-effort cap update (GPU does not track ring colors beyond capHeight metadata).
            new_cap = moving_cap_height - landing_ring_cost
            if new_cap <= 0:
                new_cap = 1
            if new_cap > final_height:
                new_cap = final_height
            state.cap_height[g, to_y, to_x] = new_cap

        # Clear origin
        state.stack_height[g, from_y, from_x] = 0
        state.stack_owner[g, from_y, from_x] = 0
        state.cap_height[g, from_y, from_x] = 0
        # Leave a marker on the departure space (RR-CANON-R092).
        state.marker_owner[g, from_y, from_x] = player


def apply_recovery_moves_vectorized(
    state: "BatchGameState",
    selected_local_idx: torch.Tensor,
    moves: "BatchMoves",
    active_mask: torch.Tensor,
) -> None:
    """Apply recovery slide moves in a fully vectorized manner.

    Optimized 2025-12-13: Eliminated Python loops and .item() calls.
    """
    device = state.device

    has_selection = (selected_local_idx >= 0) & active_mask & (moves.moves_per_game > 0)

    if not has_selection.any():
        return

    game_indices = torch.where(has_selection)[0]
    n_games = game_indices.shape[0]

    global_idx = moves.move_offsets[game_indices] + selected_local_idx[game_indices]
    global_idx = torch.clamp(global_idx, 0, max(0, moves.total_moves - 1))

    from_y = moves.from_y[global_idx].long()
    from_x = moves.from_x[global_idx].long()
    to_y = moves.to_y[global_idx].long()
    to_x = moves.to_x[global_idx].long()

    players = state.current_player[game_indices]

    # Record in history
    move_idx = state.move_count[game_indices]
    history_mask = move_idx < state.max_history_moves
    if history_mask.any():
        hist_games = game_indices[history_mask]
        hist_move_idx = move_idx[history_mask].long()
        hist_dtype = state.move_history.dtype
        state.move_history[hist_games, hist_move_idx, 0] = MoveType.RECOVERY_SLIDE
        state.move_history[hist_games, hist_move_idx, 1] = players[history_mask].to(hist_dtype)
        state.move_history[hist_games, hist_move_idx, 2] = from_y[history_mask].to(hist_dtype)
        state.move_history[hist_games, hist_move_idx, 3] = from_x[history_mask].to(hist_dtype)
        state.move_history[hist_games, hist_move_idx, 4] = to_y[history_mask].to(hist_dtype)
        state.move_history[hist_games, hist_move_idx, 5] = to_x[history_mask].to(hist_dtype)

    state.move_count[game_indices] += 1

    # Clear source marker
    state.marker_owner[game_indices, from_y, from_x] = 0

    # Check destination for stack-strike vs normal recovery
    dest_height = state.stack_height[game_indices, to_y, to_x]
    dest_owner = state.stack_owner[game_indices, to_y, to_x]
    is_stack_strike = (dest_height > 0) & (dest_owner > 0)

    # Handle stack-strike recovery (RR-CANON-R112(b2))
    if is_stack_strike.any():
        ss_games = game_indices[is_stack_strike]
        ss_to_y = to_y[is_stack_strike]
        ss_to_x = to_x[is_stack_strike]
        ss_players = players[is_stack_strike]
        ss_dest_owner = dest_owner[is_stack_strike]
        ss_dest_height = dest_height[is_stack_strike]
        ss_old_cap = state.cap_height[ss_games, ss_to_y, ss_to_x]

        # Update eliminated rings tracking
        ones = torch.ones(ss_games.shape[0], dtype=state.eliminated_rings.dtype, device=device)
        state.eliminated_rings.index_put_(
            (ss_games, ss_dest_owner.long()),
            ones,
            accumulate=True
        )
        state.rings_caused_eliminated.index_put_(
            (ss_games, ss_players.long()),
            ones,
            accumulate=True
        )

        # Update stack
        new_height = torch.clamp(ss_dest_height - 1, min=0)
        new_cap = torch.clamp(ss_old_cap - 1, min=1)
        new_cap = torch.minimum(new_cap, new_height)
        new_cap = torch.where(new_height > 0, new_cap, torch.zeros_like(new_cap))

        state.stack_height[ss_games, ss_to_y, ss_to_x] = new_height.to(state.stack_height.dtype)
        is_cleared = new_height == 0
        state.stack_owner[ss_games[is_cleared], ss_to_y[is_cleared], ss_to_x[is_cleared]] = 0
        state.cap_height[ss_games, ss_to_y, ss_to_x] = new_cap.to(state.cap_height.dtype)

    # Handle normal recovery slide
    is_normal = ~is_stack_strike
    if is_normal.any():
        nr_games = game_indices[is_normal]
        nr_to_y = to_y[is_normal]
        nr_to_x = to_x[is_normal]
        nr_players = players[is_normal]
        state.marker_owner[nr_games, nr_to_y, nr_to_x] = nr_players.to(state.marker_owner.dtype)

    # Deduct buried ring cost - only if player has buried rings
    current_buried = state.buried_rings[game_indices, players.long()]
    has_buried = current_buried > 0
    if has_buried.any():
        hb_games = game_indices[has_buried]
        hb_players = players[has_buried].long()
        neg_ones = torch.full((hb_games.shape[0],), -1, dtype=state.buried_rings.dtype, device=device)
        pos_ones = torch.ones(hb_games.shape[0], dtype=state.eliminated_rings.dtype, device=device)

        state.buried_rings.index_put_((hb_games, hb_players), neg_ones, accumulate=True)
        # Self-elimination for buried ring extraction (RR-CANON-R114/R115)
        state.eliminated_rings.index_put_((hb_games, hb_players), pos_ones, accumulate=True)
        state.rings_caused_eliminated.index_put_((hb_games, hb_players), pos_ones, accumulate=True)


def apply_no_action_moves_batch(
    state: "BatchGameState",
    mask: torch.Tensor,
) -> None:
    """Record a NO_ACTION move for each masked active game.

    This is used to avoid silent phase progression when a player has no
    legal action in an interactive phase (RR-CANON-R075).

    Optimized 2025-12-13: Eliminated Python loops and .item() calls.
    """
    active_mask = mask & state.get_active_mask()
    if not active_mask.any():
        return

    game_indices = torch.where(active_mask)[0]
    move_idx = state.move_count[game_indices]
    players = state.current_player[game_indices]

    # Record in history for games with space
    history_mask = move_idx < state.max_history_moves
    if history_mask.any():
        hist_games = game_indices[history_mask]
        hist_move_idx = move_idx[history_mask].long()
        hist_dtype = state.move_history.dtype
        state.move_history[hist_games, hist_move_idx, 0] = MoveType.NO_ACTION
        state.move_history[hist_games, hist_move_idx, 1] = players[history_mask].to(hist_dtype)
        state.move_history[hist_games, hist_move_idx, 2] = -1
        state.move_history[hist_games, hist_move_idx, 3] = -1
        state.move_history[hist_games, hist_move_idx, 4] = -1
        state.move_history[hist_games, hist_move_idx, 5] = -1

    state.move_count[game_indices] += 1


# =============================================================================
# Batch Apply Functions (main API)
# =============================================================================


def apply_placement_moves_batch_vectorized(
    state: "BatchGameState",
    move_indices: torch.Tensor,
    moves: "BatchMoves",
) -> None:
    """Vectorized placement move application.

    Args:
        state: BatchGameState to modify
        move_indices: (batch_size,) index into moves for each game's selected move
        moves: BatchMoves containing all candidate moves
    """
    device = state.device
    batch_size = state.batch_size
    active_mask = state.get_active_mask()

    # Filter to games with valid moves
    has_moves = moves.moves_per_game > 0
    valid_local_idx = move_indices < moves.moves_per_game
    process_mask = active_mask & has_moves & valid_local_idx

    if not process_mask.any():
        return

    # Get game indices to process
    game_indices = torch.where(process_mask)[0]
    n_games = game_indices.shape[0]

    # Compute global move indices
    global_indices = moves.move_offsets[game_indices] + move_indices[game_indices]

    # Gather move data
    y = moves.from_y[global_indices].long()
    x = moves.from_x[global_indices].long()
    move_type = moves.move_type[global_indices]
    players = state.current_player[game_indices]

    # Gather destination state
    dest_owner = state.stack_owner[game_indices, y, x]
    dest_height = state.stack_height[game_indices, y, x]
    dest_cap = state.cap_height[game_indices, y, x]

    # Record move history (for games with history space)
    move_idx = state.move_count[game_indices]
    history_mask = move_idx < state.max_history_moves
    if history_mask.any():
        hist_games = game_indices[history_mask]
        hist_move_idx = move_idx[history_mask].long()
        hist_y = y[history_mask]
        hist_x = x[history_mask]
        hist_players = players[history_mask]
        hist_move_type = move_type[history_mask]

        # Cast to match move_history dtype (int16)
        hist_dtype = state.move_history.dtype
        state.move_history[hist_games, hist_move_idx, 0] = hist_move_type.to(hist_dtype)
        state.move_history[hist_games, hist_move_idx, 1] = hist_players.to(hist_dtype)
        state.move_history[hist_games, hist_move_idx, 2] = hist_y.to(hist_dtype)
        state.move_history[hist_games, hist_move_idx, 3] = hist_x.to(hist_dtype)
        state.move_history[hist_games, hist_move_idx, 4] = hist_y.to(hist_dtype)  # to_y same for placement
        state.move_history[hist_games, hist_move_idx, 5] = hist_x.to(hist_dtype)  # to_x same for placement

    # Compute new values based on destination state
    is_empty = dest_height <= 0
    is_same_owner = dest_owner == players

    # New height: 1 if empty, else dest_height + 1 (clamped to 127)
    new_height = torch.where(is_empty, torch.ones_like(dest_height), torch.clamp(dest_height + 1, max=127))

    # New cap_height: 1 if empty or different owner, else dest_cap + 1 (clamped to 127)
    new_cap = torch.where(
        is_empty | ~is_same_owner,
        torch.ones_like(dest_cap),
        torch.clamp(dest_cap + 1, max=127)
    )

    # Update stack state (cast to match dtypes)
    state.stack_owner[game_indices, y, x] = players.to(state.stack_owner.dtype)
    state.stack_height[game_indices, y, x] = new_height.to(state.stack_height.dtype)
    state.cap_height[game_indices, y, x] = new_cap.to(state.cap_height.dtype)

    # Handle buried rings for opponent stacks - vectorized with index_put_
    is_opponent = ~is_empty & (dest_owner != 0) & (dest_owner != players)
    if is_opponent.any():
        opp_games = game_indices[is_opponent]
        opp_owners = dest_owner[is_opponent].long()
        opp_ones = torch.ones(opp_games.shape[0], dtype=state.buried_rings.dtype, device=device)
        state.buried_rings.index_put_(
            (opp_games, opp_owners),
            opp_ones,
            accumulate=True
        )

    # Update rings_in_hand - vectorized with index_put_
    neg_ones = torch.full((n_games,), -1, dtype=state.rings_in_hand.dtype, device=device)
    state.rings_in_hand.index_put_(
        (game_indices, players.long()),
        neg_ones,
        accumulate=True
    )

    # Update must_move_from (cast to match dtype)
    state.must_move_from_y[game_indices] = y.to(state.must_move_from_y.dtype)
    state.must_move_from_x[game_indices] = x.to(state.must_move_from_x.dtype)

    # Advance move counter
    state.move_count[game_indices] += 1


def _apply_placement_moves_batch_legacy(
    state: "BatchGameState",
    move_indices: torch.Tensor,
    moves: "BatchMoves",
) -> None:
    """Legacy Python-loop based placement application.

    Kept for debugging and comparison.
    """
    batch_size = state.batch_size
    active_mask = state.get_active_mask()

    for g in range(batch_size):
        if not active_mask[g]:
            continue

        if moves.moves_per_game[g] == 0:
            continue

        local_idx = move_indices[g].item()
        if local_idx >= moves.moves_per_game[g]:
            continue

        global_idx = moves.move_offsets[g] + local_idx

        y = moves.from_y[global_idx].item()
        x = moves.from_x[global_idx].item()
        player = state.current_player[g].item()
        move_type = moves.move_type[global_idx].item()

        move_idx = state.move_count[g].item()
        if move_idx < state.max_history_moves:
            state.move_history[g, move_idx, 0] = move_type
            state.move_history[g, move_idx, 1] = player
            state.move_history[g, move_idx, 2] = y
            state.move_history[g, move_idx, 3] = x
            state.move_history[g, move_idx, 4] = y
            state.move_history[g, move_idx, 5] = x

        dest_owner = int(state.stack_owner[g, y, x].item())
        dest_height = int(state.stack_height[g, y, x].item())
        dest_cap = int(state.cap_height[g, y, x].item())

        if dest_height <= 0:
            state.stack_owner[g, y, x] = player
            state.stack_height[g, y, x] = 1
            state.cap_height[g, y, x] = 1
        else:
            new_height = min(127, dest_height + 1)
            state.stack_owner[g, y, x] = player
            state.stack_height[g, y, x] = new_height
            if dest_owner == player:
                state.cap_height[g, y, x] = min(127, dest_cap + 1)
            else:
                state.cap_height[g, y, x] = 1

            if dest_owner not in (0, player):
                state.buried_rings[g, dest_owner] += 1
        state.rings_in_hand[g, player] -= 1
        state.must_move_from_y[g] = y
        state.must_move_from_x[g] = x
        state.move_count[g] += 1


def apply_placement_moves_batch(
    state: "BatchGameState",
    move_indices: torch.Tensor,
    moves: "BatchMoves",
) -> None:
    """Apply selected placement moves to batch state (in-place).

    Args:
        state: BatchGameState to modify
        move_indices: (batch_size,) index into moves for each game's selected move
        moves: BatchMoves containing all candidate moves
    """
    if os.environ.get("RINGRIFT_GPU_PLACEMENT_LEGACY", "0") == "1":
        _apply_placement_moves_batch_legacy(state, move_indices, moves)
    else:
        apply_placement_moves_batch_vectorized(state, move_indices, moves)


def apply_movement_moves_batch_vectorized(
    state: "BatchGameState",
    move_indices: torch.Tensor,
    moves: "BatchMoves",
) -> None:
    """Vectorized movement move application.

    Args:
        state: BatchGameState to modify
        move_indices: (batch_size,) index into moves for each game's selected move
        moves: BatchMoves containing all candidate moves
    """
    device = state.device
    batch_size = state.batch_size
    board_size = state.board_size
    active_mask = state.get_active_mask()

    # Filter to games with valid moves
    has_moves = moves.moves_per_game > 0
    valid_local_idx = move_indices < moves.moves_per_game
    process_mask = active_mask & has_moves & valid_local_idx

    if not process_mask.any():
        return

    # Get game indices to process
    game_indices = torch.where(process_mask)[0]
    n_games = game_indices.shape[0]

    # Compute global move indices
    global_indices = moves.move_offsets[game_indices] + move_indices[game_indices]

    # Gather move data
    from_y = moves.from_y[global_indices].long()
    from_x = moves.from_x[global_indices].long()
    to_y = moves.to_y[global_indices].long()
    to_x = moves.to_x[global_indices].long()
    move_type = moves.move_type[global_indices]
    players = state.current_player[game_indices]

    # Record move history
    move_idx = state.move_count[game_indices]
    history_mask = move_idx < state.max_history_moves
    if history_mask.any():
        hist_games = game_indices[history_mask]
        hist_move_idx = move_idx[history_mask].long()
        hist_dtype = state.move_history.dtype
        state.move_history[hist_games, hist_move_idx, 0] = move_type[history_mask].to(hist_dtype)
        state.move_history[hist_games, hist_move_idx, 1] = players[history_mask].to(hist_dtype)
        state.move_history[hist_games, hist_move_idx, 2] = from_y[history_mask].to(hist_dtype)
        state.move_history[hist_games, hist_move_idx, 3] = from_x[history_mask].to(hist_dtype)
        state.move_history[hist_games, hist_move_idx, 4] = to_y[history_mask].to(hist_dtype)
        state.move_history[hist_games, hist_move_idx, 5] = to_x[history_mask].to(hist_dtype)

    # Get moving stack info
    moving_height = state.stack_height[game_indices, from_y, from_x]
    moving_cap_height = state.cap_height[game_indices, from_y, from_x]

    # Compute direction and distance for each move
    dy = torch.sign(to_y - from_y)
    dx = torch.sign(to_x - from_x)
    dist = torch.maximum(torch.abs(to_y - from_y), torch.abs(to_x - from_x))
    max_dist = dist.max().item() if n_games > 0 else 0

    # Process markers along path (flip opposing markers)
    # Create path positions for all games: (n_games, max_dist-1)
    if max_dist > 1:
        steps = torch.arange(1, max_dist, device=device).view(1, -1)  # (1, max_dist-1)
        path_y = from_y.unsqueeze(1) + dy.unsqueeze(1) * steps  # (n_games, max_dist-1)
        path_x = from_x.unsqueeze(1) + dx.unsqueeze(1) * steps

        # Mask for valid path positions (step < dist, so we don't include destination)
        valid_path = steps < dist.unsqueeze(1)

        # Clamp for safe indexing
        path_y_safe = torch.clamp(path_y, 0, board_size - 1).long()
        path_x_safe = torch.clamp(path_x, 0, board_size - 1).long()

        # Get marker owners along path
        game_indices_exp = game_indices.unsqueeze(1).expand(-1, max_dist - 1)
        path_marker_owners = state.marker_owner[game_indices_exp, path_y_safe, path_x_safe]

        # Find opponent markers to flip (not 0 and not player)
        players_exp = players.unsqueeze(1).expand(-1, max_dist - 1)
        is_opponent_marker = (path_marker_owners != 0) & (path_marker_owners != players_exp) & valid_path

        # Flip opponent markers
        if is_opponent_marker.any():
            flip_games = game_indices_exp[is_opponent_marker]
            flip_y = path_y_safe[is_opponent_marker]
            flip_x = path_x_safe[is_opponent_marker]
            flip_players = players_exp[is_opponent_marker]
            state.marker_owner[flip_games, flip_y, flip_x] = flip_players.to(state.marker_owner.dtype)

    # Handle landing on ANY marker (per RR-CANON-R091/R092)
    # Landing on any marker (own or opponent) removes it and costs 1 ring (cap-elimination)
    dest_marker = state.marker_owner[game_indices, to_y, to_x]
    landing_on_any_marker = dest_marker != 0
    landing_on_own_marker = dest_marker == players
    landing_ring_cost = landing_on_any_marker.int()

    if landing_on_any_marker.any():
        marker_games = game_indices[landing_on_any_marker]
        marker_y = to_y[landing_on_any_marker]
        marker_x = to_x[landing_on_any_marker]
        # Remove the marker
        state.marker_owner[marker_games, marker_y, marker_x] = 0
        # Only collapse if landing on OWN marker
        own_marker_mask = landing_on_own_marker[landing_on_any_marker]
        if own_marker_mask.any():
            collapse_games = marker_games[own_marker_mask]
            collapse_y = marker_y[own_marker_mask]
            collapse_x = marker_x[own_marker_mask]
            state.is_collapsed[collapse_games, collapse_y, collapse_x] = True

    # Clear origin
    state.stack_owner[game_indices, from_y, from_x] = 0
    state.stack_height[game_indices, from_y, from_x] = 0
    state.cap_height[game_indices, from_y, from_x] = 0

    # Set destination
    new_height = torch.clamp(moving_height - landing_ring_cost, min=1)
    new_cap_height = torch.clamp(moving_cap_height - landing_ring_cost, min=1)
    state.stack_owner[game_indices, to_y, to_x] = players.to(state.stack_owner.dtype)
    state.stack_height[game_indices, to_y, to_x] = new_height.to(state.stack_height.dtype)
    state.cap_height[game_indices, to_y, to_x] = new_cap_height.to(state.cap_height.dtype)

    # Advance move counter only (NOT current_player - that's handled by END_TURN phase)
    # BUG FIX 2025-12-15: Removing player rotation here - it was causing players to
    # advance mid-turn, then advance again in END_TURN, resulting in players getting
    # multiple consecutive turns before opponents.
    state.move_count[game_indices] += 1


def _apply_movement_moves_batch_legacy(
    state: "BatchGameState",
    move_indices: torch.Tensor,
    moves: "BatchMoves",
) -> None:
    """Legacy Python-loop based movement application.

    Kept for debugging and comparison.
    """
    batch_size = state.batch_size
    active_mask = state.get_active_mask()

    for g in range(batch_size):
        if not active_mask[g]:
            continue

        if moves.moves_per_game[g] == 0:
            continue

        local_idx = move_indices[g].item()
        if local_idx >= moves.moves_per_game[g]:
            continue

        global_idx = moves.move_offsets[g] + local_idx

        from_y = moves.from_y[global_idx].item()
        from_x = moves.from_x[global_idx].item()
        to_y = moves.to_y[global_idx].item()
        to_x = moves.to_x[global_idx].item()
        player = state.current_player[g].item()
        move_type = moves.move_type[global_idx].item()

        move_idx = state.move_count[g].item()
        if move_idx < state.max_history_moves:
            state.move_history[g, move_idx, 0] = move_type
            state.move_history[g, move_idx, 1] = player
            state.move_history[g, move_idx, 2] = from_y
            state.move_history[g, move_idx, 3] = from_x
            state.move_history[g, move_idx, 4] = to_y
            state.move_history[g, move_idx, 5] = to_x

        moving_height = state.stack_height[g, from_y, from_x].item()
        moving_cap_height = state.cap_height[g, from_y, from_x].item()

        dy = 0 if to_y == from_y else (1 if to_y > from_y else -1)
        dx = 0 if to_x == from_x else (1 if to_x > from_x else -1)
        dist = max(abs(to_y - from_y), abs(to_x - from_x))

        for step in range(1, int(dist)):
            check_y = from_y + dy * step
            check_x = from_x + dx * step
            marker_owner = state.marker_owner[g, check_y, check_x].item()
            if marker_owner != 0 and marker_owner != player:
                state.marker_owner[g, check_y, check_x] = player

        dest_marker = state.marker_owner[g, to_y, to_x].item()
        landing_ring_cost = 0
        if dest_marker == player:
            landing_ring_cost = 1
            state.is_collapsed[g, to_y, to_x] = True
            state.marker_owner[g, to_y, to_x] = 0

        state.stack_owner[g, from_y, from_x] = 0
        state.stack_height[g, from_y, from_x] = 0
        state.cap_height[g, from_y, from_x] = 0

        new_height = max(1, moving_height - landing_ring_cost)
        new_cap_height = max(1, moving_cap_height - landing_ring_cost)
        state.stack_owner[g, to_y, to_x] = player
        state.stack_height[g, to_y, to_x] = new_height
        state.cap_height[g, to_y, to_x] = new_cap_height

        # Advance move counter only (NOT current_player - that's handled by END_TURN phase)
        # BUG FIX 2025-12-15: See apply_movement_moves_batch_vectorized for details
        state.move_count[g] += 1


def apply_movement_moves_batch(
    state: "BatchGameState",
    move_indices: torch.Tensor,
    moves: "BatchMoves",
) -> None:
    """Apply selected movement moves to batch state (in-place).

    Per RR-CANON-R090-R092:
    - Stack moves from origin to destination
    - Origin becomes empty
    - Destination gets merged stack (if own stack) or new stack
    - Markers along path: flip on pass, collapse cost on landing

    Args:
        state: BatchGameState to modify
        move_indices: (batch_size,) index into moves for each game's selected move
        moves: BatchMoves containing all candidate moves
    """
    if os.environ.get("RINGRIFT_GPU_MOVEMENT_APPLY_LEGACY", "0") == "1":
        _apply_movement_moves_batch_legacy(state, move_indices, moves)
    else:
        apply_movement_moves_batch_vectorized(state, move_indices, moves)


def apply_capture_moves_batch_vectorized(
    state: "BatchGameState",
    move_indices: torch.Tensor,
    moves: "BatchMoves",
) -> None:
    """Vectorized capture move application.

    Args:
        state: BatchGameState to modify
        move_indices: (batch_size,) index into moves for each game's selected move
        moves: BatchMoves containing all candidate moves
    """
    device = state.device
    batch_size = state.batch_size
    board_size = state.board_size
    active_mask = state.get_active_mask()

    # Filter to games with valid moves
    has_moves = moves.moves_per_game > 0
    valid_local_idx = move_indices < moves.moves_per_game
    process_mask = active_mask & has_moves & valid_local_idx

    if not process_mask.any():
        return

    # Get game indices to process
    game_indices = torch.where(process_mask)[0]
    n_games = game_indices.shape[0]

    # Compute global move indices
    global_indices = moves.move_offsets[game_indices] + move_indices[game_indices]

    # Gather move data
    from_y = moves.from_y[global_indices].long()
    from_x = moves.from_x[global_indices].long()
    to_y = moves.to_y[global_indices].long()
    to_x = moves.to_x[global_indices].long()
    move_type = moves.move_type[global_indices]
    players = state.current_player[game_indices]

    # Record move history
    move_idx = state.move_count[game_indices]
    history_mask = move_idx < state.max_history_moves
    if history_mask.any():
        hist_games = game_indices[history_mask]
        hist_move_idx = move_idx[history_mask].long()
        hist_dtype = state.move_history.dtype
        state.move_history[hist_games, hist_move_idx, 0] = move_type[history_mask].to(hist_dtype)
        state.move_history[hist_games, hist_move_idx, 1] = players[history_mask].to(hist_dtype)
        state.move_history[hist_games, hist_move_idx, 2] = from_y[history_mask].to(hist_dtype)
        state.move_history[hist_games, hist_move_idx, 3] = from_x[history_mask].to(hist_dtype)
        state.move_history[hist_games, hist_move_idx, 4] = to_y[history_mask].to(hist_dtype)
        state.move_history[hist_games, hist_move_idx, 5] = to_x[history_mask].to(hist_dtype)

    # Get attacker stack info
    attacker_height = state.stack_height[game_indices, from_y, from_x]
    attacker_cap_height = state.cap_height[game_indices, from_y, from_x]

    # Compute direction and distance for path processing
    # NOTE: `to` is the LANDING position, not the target. We must scan the ray to find the target.
    dy = torch.sign(to_y - from_y)
    dx = torch.sign(to_x - from_x)
    dist = torch.maximum(torch.abs(to_y - from_y), torch.abs(to_x - from_x))
    max_dist = dist.max().item() if n_games > 0 else 0

    # === BUGFIX 2025-12-17: Find actual target by scanning ray from->to ===
    # Move generation stores landing position in `to`, but target is the first
    # non-empty stack along the ray between from and landing.
    if max_dist > 0:
        steps = torch.arange(1, max_dist + 1, device=device).view(1, -1)  # (1, max_dist)
        ray_y = from_y.unsqueeze(1) + dy.unsqueeze(1) * steps  # (n_games, max_dist)
        ray_x = from_x.unsqueeze(1) + dx.unsqueeze(1) * steps  # (n_games, max_dist)

        # Clamp for safe indexing
        ray_y_safe = torch.clamp(ray_y, 0, board_size - 1).long()
        ray_x_safe = torch.clamp(ray_x, 0, board_size - 1).long()

        game_indices_exp = game_indices.unsqueeze(1).expand(-1, max_dist)
        ray_owners = state.stack_owner[game_indices_exp, ray_y_safe, ray_x_safe]

        # Steps within landing distance (exclusive of landing itself)
        dist_exp = dist.unsqueeze(1).expand(-1, max_dist)
        within_landing = steps < dist_exp

        # Find first non-zero stack along ray before landing
        ray_has_stack = (ray_owners != 0) & within_landing

        # Get index of first target (argmax returns first True)
        has_any_target = ray_has_stack.any(dim=1)
        target_step_idx = ray_has_stack.to(torch.int32).argmax(dim=1)  # (n_games,)

        # Compute target positions
        target_y = from_y + dy * (target_step_idx + 1)
        target_x = from_x + dx * (target_step_idx + 1)
    else:
        # Fallback: no distance, treat to as target
        target_y = to_y
        target_x = to_x
        has_any_target = torch.ones(n_games, dtype=torch.bool, device=device)

    # Get defender info from TARGET position (not landing)
    defender_owner = state.stack_owner[game_indices, target_y, target_x]
    defender_height = state.stack_height[game_indices, target_y, target_x]
    defender_cap_height = state.cap_height[game_indices, target_y, target_x]

    # Process markers along path (flip opposing markers)
    if max_dist > 1:
        steps = torch.arange(1, max_dist, device=device).view(1, -1)
        path_y = from_y.unsqueeze(1) + dy.unsqueeze(1) * steps
        path_x = from_x.unsqueeze(1) + dx.unsqueeze(1) * steps

        valid_path = steps < dist.unsqueeze(1)

        path_y_safe = torch.clamp(path_y, 0, board_size - 1).long()
        path_x_safe = torch.clamp(path_x, 0, board_size - 1).long()

        game_indices_exp_markers = game_indices.unsqueeze(1).expand(-1, max_dist - 1)
        path_marker_owners = state.marker_owner[game_indices_exp_markers, path_y_safe, path_x_safe]

        players_exp = players.unsqueeze(1).expand(-1, max_dist - 1)
        is_opponent_marker = (path_marker_owners != 0) & (path_marker_owners != players_exp) & valid_path

        if is_opponent_marker.any():
            flip_games = game_indices_exp_markers[is_opponent_marker]
            flip_y = path_y_safe[is_opponent_marker]
            flip_x = path_x_safe[is_opponent_marker]
            flip_players = players_exp[is_opponent_marker]
            state.marker_owner[flip_games, flip_y, flip_x] = flip_players.to(state.marker_owner.dtype)

    # Eliminate defender's ring (update tracking tensors) - vectorized using index_put_
    ones = torch.ones(n_games, dtype=state.eliminated_rings.dtype, device=device)
    state.eliminated_rings.index_put_(
        (game_indices, defender_owner.long()),
        ones,
        accumulate=True
    )
    state.rings_caused_eliminated.index_put_(
        (game_indices, players.long()),
        ones,
        accumulate=True
    )

    # === Update TARGET stack (reduce height by 1, capturing the top ring) ===
    new_target_height = torch.clamp(defender_height - 1, min=0)
    state.stack_height[game_indices, target_y, target_x] = new_target_height.to(state.stack_height.dtype)

    # Clear owner if height becomes 0
    target_is_empty = new_target_height == 0
    state.stack_owner[game_indices, target_y, target_x] = torch.where(
        target_is_empty,
        torch.zeros_like(defender_owner),
        defender_owner
    ).to(state.stack_owner.dtype)

    # Update target cap height
    new_target_cap = torch.clamp(defender_cap_height - 1, min=1)
    new_target_cap = torch.minimum(new_target_cap, new_target_height)
    new_target_cap = torch.where(target_is_empty, torch.zeros_like(new_target_cap), new_target_cap)
    state.cap_height[game_indices, target_y, target_x] = new_target_cap.to(state.cap_height.dtype)

    # Clear target marker
    state.marker_owner[game_indices, target_y, target_x] = 0

    # Track captured ring as buried for target owner
    target_owner_nonzero = defender_owner != 0
    if target_owner_nonzero.any():
        state.buried_rings.index_put_(
            (game_indices[target_owner_nonzero], defender_owner[target_owner_nonzero].long()),
            torch.ones(target_owner_nonzero.sum(), dtype=state.buried_rings.dtype, device=device),
            accumulate=True
        )

    # === Clear ORIGIN ===
    state.stack_owner[game_indices, from_y, from_x] = 0
    state.stack_height[game_indices, from_y, from_x] = 0
    state.cap_height[game_indices, from_y, from_x] = 0

    # Leave departure marker at origin (RR-CANON-R092)
    state.marker_owner[game_indices, from_y, from_x] = players.to(state.marker_owner.dtype)

    # === Move attacker to LANDING position ===
    # Check for marker at landing (RR-CANON-R102: landing marker costs 1 ring)
    landing_marker = state.marker_owner[game_indices, to_y, to_x]
    landing_ring_cost = (landing_marker != 0).to(torch.int32)

    # Clear landing marker if present
    state.marker_owner[game_indices, to_y, to_x] = torch.where(
        landing_marker != 0,
        torch.zeros_like(landing_marker),
        landing_marker
    )

    # Track eliminated ring from landing marker cost
    if landing_ring_cost.any():
        state.eliminated_rings.index_put_(
            (game_indices, players.long()),
            landing_ring_cost.to(state.eliminated_rings.dtype),
            accumulate=True
        )
        state.rings_caused_eliminated.index_put_(
            (game_indices, players.long()),
            landing_ring_cost.to(state.rings_caused_eliminated.dtype),
            accumulate=True
        )

    # Attacker gains captured ring (+1) minus landing marker cost
    new_height = torch.clamp(attacker_height + 1 - landing_ring_cost, min=1, max=5)
    new_cap_height = torch.clamp(attacker_cap_height - landing_ring_cost, min=1)
    new_cap_height = torch.minimum(new_cap_height, new_height)

    state.stack_owner[game_indices, to_y, to_x] = players.to(state.stack_owner.dtype)
    state.stack_height[game_indices, to_y, to_x] = new_height.to(state.stack_height.dtype)
    state.cap_height[game_indices, to_y, to_x] = new_cap_height.to(state.cap_height.dtype)

    # Place marker for attacker at landing
    state.marker_owner[game_indices, to_y, to_x] = players.to(state.marker_owner.dtype)

    # Clear must_move_from constraint after capture (RR-CANON-R090)
    # The player has fulfilled their movement obligation by capturing.
    state.must_move_from_y[game_indices] = -1
    state.must_move_from_x[game_indices] = -1

    # Advance move counter only (NOT current_player - that's handled by END_TURN phase)
    # BUG FIX 2025-12-15: Removing player rotation here - it was causing players to
    # advance mid-turn, then advance again in END_TURN, resulting in players getting
    # multiple consecutive turns before opponents.
    state.move_count[game_indices] += 1


def _apply_capture_moves_batch_legacy(
    state: "BatchGameState",
    move_indices: torch.Tensor,
    moves: "BatchMoves",
) -> None:
    """Legacy Python-loop based capture application.

    Kept for debugging and comparison.
    """
    batch_size = state.batch_size
    active_mask = state.get_active_mask()

    for g in range(batch_size):
        if not active_mask[g]:
            continue

        if moves.moves_per_game[g] == 0:
            continue

        local_idx = move_indices[g].item()
        if local_idx >= moves.moves_per_game[g]:
            continue

        global_idx = moves.move_offsets[g] + local_idx

        from_y = moves.from_y[global_idx].item()
        from_x = moves.from_x[global_idx].item()
        to_y = moves.to_y[global_idx].item()
        to_x = moves.to_x[global_idx].item()
        player = state.current_player[g].item()
        move_type = moves.move_type[global_idx].item()

        move_idx = state.move_count[g].item()
        if move_idx < state.max_history_moves:
            state.move_history[g, move_idx, 0] = move_type
            state.move_history[g, move_idx, 1] = player
            state.move_history[g, move_idx, 2] = from_y
            state.move_history[g, move_idx, 3] = from_x
            state.move_history[g, move_idx, 4] = to_y
            state.move_history[g, move_idx, 5] = to_x

        attacker_height = state.stack_height[g, from_y, from_x].item()
        attacker_cap_height = state.cap_height[g, from_y, from_x].item()

        defender_owner = state.stack_owner[g, to_y, to_x].item()
        defender_height = state.stack_height[g, to_y, to_x].item()

        dy = 0 if to_y == from_y else (1 if to_y > from_y else -1)
        dx = 0 if to_x == from_x else (1 if to_x > from_x else -1)
        dist = max(abs(to_y - from_y), abs(to_x - from_x))

        for step in range(1, int(dist)):
            check_y = from_y + dy * step
            check_x = from_x + dx * step
            marker_owner = state.marker_owner[g, check_y, check_x].item()
            if marker_owner != 0 and marker_owner != player:
                state.marker_owner[g, check_y, check_x] = player

        state.eliminated_rings[g, defender_owner] += 1
        state.rings_caused_eliminated[g, player] += 1

        state.stack_owner[g, from_y, from_x] = 0
        state.stack_height[g, from_y, from_x] = 0
        state.cap_height[g, from_y, from_x] = 0

        new_height = attacker_height + defender_height - 1
        new_cap_height = attacker_cap_height
        state.stack_owner[g, to_y, to_x] = player
        state.stack_height[g, to_y, to_x] = min(5, new_height)
        state.cap_height[g, to_y, to_x] = min(5, new_cap_height)

        state.marker_owner[g, to_y, to_x] = player

        # Advance move counter only (NOT current_player - that's handled by END_TURN phase)
        # BUG FIX 2025-12-15: See apply_capture_moves_batch_vectorized for details
        state.move_count[g] += 1


def apply_capture_moves_batch(
    state: "BatchGameState",
    move_indices: torch.Tensor,
    moves: "BatchMoves",
) -> None:
    """Apply selected capture moves to batch state (in-place).

    Per RR-CANON-R100-R103:
    - Attacker moves onto defender stack
    - Defender's top ring is eliminated
    - Stacks merge (attacker on top)
    - Control transfers to attacker

    Args:
        state: BatchGameState to modify
        move_indices: (batch_size,) index into moves for each game's selected move
        moves: BatchMoves containing all candidate moves
    """
    if os.environ.get("RINGRIFT_GPU_CAPTURE_APPLY_LEGACY", "0") == "1":
        _apply_capture_moves_batch_legacy(state, move_indices, moves)
    else:
        apply_capture_moves_batch_vectorized(state, move_indices, moves)


__all__ = [
    # Vectorized apply functions (for move selection)
    'apply_capture_moves_vectorized',
    'apply_movement_moves_vectorized',
    'apply_recovery_moves_vectorized',
    'apply_no_action_moves_batch',
    # Batch apply functions (main API)
    'apply_placement_moves_batch_vectorized',
    '_apply_placement_moves_batch_legacy',
    'apply_placement_moves_batch',
    'apply_movement_moves_batch_vectorized',
    '_apply_movement_moves_batch_legacy',
    'apply_movement_moves_batch',
    'apply_capture_moves_batch_vectorized',
    '_apply_capture_moves_batch_legacy',
    'apply_capture_moves_batch',
]
