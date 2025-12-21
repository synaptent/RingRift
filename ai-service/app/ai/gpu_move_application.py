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

from .gpu_game_types import GamePhase, MoveType

if TYPE_CHECKING:
    from .gpu_move_generation import BatchMoves
    from .gpu_parallel_games import BatchGameState


# =============================================================================
# Vectorized Apply Functions (for move selection)
# =============================================================================


def apply_capture_moves_vectorized(
    state: BatchGameState,
    selected_local_idx: torch.Tensor,
    moves: BatchMoves,
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

        # Record in history AFTER finding target
        # 9 columns: move_type, player, from_y, from_x, to_y, to_x, phase, capture_target_y, capture_target_x
        if mc < state.max_history_moves:
            state.move_history[g, mc, 0] = MoveType.CAPTURE
            state.move_history[g, mc, 1] = player
            state.move_history[g, mc, 2] = from_y
            state.move_history[g, mc, 3] = from_x
            state.move_history[g, mc, 4] = to_y
            state.move_history[g, mc, 5] = to_x
            state.move_history[g, mc, 6] = int(state.current_phase[g].item())
            # December 2025: Record capture target for canonical export
            state.move_history[g, mc, 7] = target_y if target_y is not None else -1
            state.move_history[g, mc, 8] = target_x if target_x is not None else -1
        state.move_count[g] += 1

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

        # December 2025: BUG FIX - When capturing the target's entire cap, ownership
        # transfers to the opponent.
        new_target_height = max(0, target_height - 1)
        state.stack_height[g, target_y, target_x] = new_target_height

        # Check if target's cap was fully captured
        target_cap_fully_captured = target_cap_height <= 1  # Cap will be 0 after -1

        if new_target_height <= 0:
            state.stack_owner[g, target_y, target_x] = 0
            state.cap_height[g, target_y, target_x] = 0
        elif target_cap_fully_captured:
            # Cap captured, ownership transfers to opponent
            opponent = 1 if target_owner == 2 else 2
            state.stack_owner[g, target_y, target_x] = opponent
            state.cap_height[g, target_y, target_x] = new_target_height
        else:
            # Cap not fully captured, defender keeps ownership
            new_target_cap = target_cap_height - 1
            if new_target_cap <= 0:
                new_target_cap = 1
            if new_target_cap > new_target_height:
                new_target_cap = new_target_height
            state.cap_height[g, target_y, target_x] = new_target_cap

        # Track captured ring as "buried" for the ring's owner (when capturing an opponent).
        if target_owner != 0 and target_owner != player:
            state.buried_rings[g, target_owner] += 1
            # December 2025: Track buried ring count at position for recovery extraction
            state.buried_at[g, target_owner, to_y, to_x] += 1

        # Move attacker to landing and apply net height change:
        # +1 captured ring (to bottom) - landing marker elimination cost.
        # December 2025: BUG FIX - When landing marker eliminates the attacker's entire cap,
        # ownership transfers to the target's original owner.
        new_height = attacker_height + 1 - landing_ring_cost
        state.stack_height[g, to_y, to_x] = new_height

        # Check if landing cost eliminated entire cap
        cap_fully_eliminated = landing_ring_cost >= attacker_cap_height
        # December 2025: Check if attacker has buried rings (opponent's rings under their cap)
        attacker_has_buried = attacker_cap_height < attacker_height
        buried_count = attacker_height - attacker_cap_height

        if state.num_players == 2 and cap_fully_eliminated and attacker_has_buried:
            # December 2025: BUG FIX - When cap is eliminated AND attacker has buried
            # rings, ownership transfers to the opponent (who owns those buried rings).
            # The remaining stack: captured ring (bottom) + buried opponent rings (now cap)
            opponent = 1 if player == 2 else 2
            new_owner = opponent
            new_cap = buried_count
            # The buried rings are now exposed - clear buried tracking
            buried_count_at_pos = state.buried_at[g, opponent, to_y, to_x].item()
            if buried_count_at_pos > 0:
                state.buried_at[g, opponent, to_y, to_x] = 0
                state.buried_rings[g, opponent] -= buried_count_at_pos
        elif cap_fully_eliminated:
            # Ownership transfers to target owner, new cap is all remaining rings
            new_owner = target_owner
            new_cap = new_height
        elif target_owner == player and attacker_cap_height == attacker_height:
            # SELF-CAPTURE without buried rings:
            # Per RR-CANON-R101/R102, captured ring goes to bottom of stack.
            # If attacker has no buried rings (cap == height), and target is same color,
            # the entire resulting stack is same color, so cap = new_height.
            new_owner = player
            new_cap = new_height
        else:
            # ENEMY CAPTURE or SELF-CAPTURE with buried rings:
            # Captured ring goes to bottom, doesn't extend the cap sequence from top.
            new_owner = player
            new_cap = attacker_cap_height - landing_ring_cost
            if new_cap <= 0:
                new_cap = 1
            if new_cap > new_height:
                new_cap = new_height

        state.stack_owner[g, to_y, to_x] = new_owner
        state.cap_height[g, to_y, to_x] = new_cap

        # Clear origin stack and leave a departure marker (RR-CANON-R092).
        state.stack_height[g, from_y, from_x] = 0
        state.stack_owner[g, from_y, from_x] = 0
        state.cap_height[g, from_y, from_x] = 0
        state.marker_owner[g, from_y, from_x] = player

        # December 2025: Move buried_at tracking from origin to landing
        # Any buried rings under the attacker move with it
        for p in range(1, state.num_players + 1):
            count = state.buried_at[g, p, from_y, from_x].item()
            if count > 0:
                state.buried_at[g, p, to_y, to_x] += count
                state.buried_at[g, p, from_y, from_x] = 0


def apply_movement_moves_vectorized(
    state: BatchGameState,
    selected_local_idx: torch.Tensor,
    moves: BatchMoves,
    active_mask: torch.Tensor,
) -> None:
    """Apply movement moves in a vectorized manner.

    Similar to capture moves but without defender elimination.
    Still requires iteration for path marker processing.
    """
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

        # Record in history (7 columns: move_type, player, from_y, from_x, to_y, to_x, phase)
        # CANONICAL: Always use MOVEMENT phase for movement moves
        if mc < state.max_history_moves:
            state.move_history[g, mc, 0] = MoveType.MOVEMENT
            state.move_history[g, mc, 1] = player
            state.move_history[g, mc, 2] = from_y
            state.move_history[g, mc, 3] = from_x
            state.move_history[g, mc, 4] = to_y
            state.move_history[g, mc, 5] = to_x
            state.move_history[g, mc, 6] = GamePhase.MOVEMENT
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

        # December 2025: Move buried_at tracking from origin to destination
        # Any buried rings under the stack move with it
        for p in range(1, state.num_players + 1):
            count = state.buried_at[g, p, from_y, from_x].item()
            if count > 0:
                state.buried_at[g, p, to_y, to_x] += count
                state.buried_at[g, p, from_y, from_x] = 0


def apply_recovery_moves_vectorized(
    state: BatchGameState,
    selected_local_idx: torch.Tensor,
    moves: BatchMoves,
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

    global_idx = moves.move_offsets[game_indices] + selected_local_idx[game_indices]
    global_idx = torch.clamp(global_idx, 0, max(0, moves.total_moves - 1))

    from_y = moves.from_y[global_idx].long()
    from_x = moves.from_x[global_idx].long()
    to_y = moves.to_y[global_idx].long()
    to_x = moves.to_x[global_idx].long()

    players = state.current_player[game_indices]

    # Record in history (7 columns: move_type, player, from_y, from_x, to_y, to_x, phase)
    # CANONICAL: Recovery slide is always in MOVEMENT phase (RR-CANON-R110-R115)
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
        # Recovery slides are part of MOVEMENT phase per history_contract.py
        # (recovery_slide is in movement phase allowed types, not a separate phase)
        state.move_history[hist_games, hist_move_idx, 6] = GamePhase.MOVEMENT

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
        # December 2025 BUG FIX: Change min=1 to min=0 to allow cap to reach 0
        # When cap reaches 0 but stack still has height, ownership needs recalculation
        new_cap = torch.clamp(ss_old_cap - 1, min=0)
        new_cap = torch.minimum(new_cap, new_height)
        new_cap = torch.where(new_height > 0, new_cap, torch.zeros_like(new_cap))

        state.stack_height[ss_games, ss_to_y, ss_to_x] = new_height.to(state.stack_height.dtype)
        is_cleared = new_height == 0
        state.stack_owner[ss_games[is_cleared], ss_to_y[is_cleared], ss_to_x[is_cleared]] = 0
        state.cap_height[ss_games, ss_to_y, ss_to_x] = new_cap.to(state.cap_height.dtype)

        # December 2025 BUG FIX: When cap becomes 0 but height > 0, the entire
        # cap was eliminated and a buried ring is now on top. Recalculate ownership
        # by finding who has buried rings at this position.
        needs_ownership_recalc = (new_cap == 0) & (new_height > 0)
        if needs_ownership_recalc.any():
            for i in torch.where(needs_ownership_recalc)[0]:
                g = ss_games[i].item()
                y_pos = ss_to_y[i].item()
                x_pos = ss_to_x[i].item()

                # Find new owner: check buried_at for each player
                new_owner = 0
                new_owner_cap = 0
                for p in range(1, state.num_players + 1):
                    if state.buried_at[g, p, y_pos, x_pos].item() > 0:
                        new_owner = p
                        # Conservative cap estimate: at least 1 ring
                        new_owner_cap = 1
                        break

                if new_owner > 0:
                    state.stack_owner[g, y_pos, x_pos] = new_owner
                    state.cap_height[g, y_pos, x_pos] = new_owner_cap
                    # Clear old owner's buried_at since they now control the stack
                    # (top ring is no longer buried)
                    # Note: We don't know exact cap, so we leave buried_at for now

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

        # December 2025 - Recovery fix: Also decrement the stack that contains the buried ring
        # For each game with a buried ring, find the stack position and reduce its height
        for i, (g, p) in enumerate(zip(hb_games.tolist(), hb_players.tolist(), strict=False)):
            # Find a position where this player has a buried ring
            buried_mask = state.buried_at[g, p] > 0  # (board_size, board_size)
            if buried_mask.any():
                # Get first buried position
                buried_indices = torch.where(buried_mask)
                if len(buried_indices[0]) > 0:
                    extraction_y = buried_indices[0][0].item()
                    extraction_x = buried_indices[1][0].item()

                    # Decrement stack height at this position
                    old_height = state.stack_height[g, extraction_y, extraction_x].item()
                    if old_height > 0:
                        new_height = old_height - 1
                        state.stack_height[g, extraction_y, extraction_x] = new_height

                        # If stack is now empty, clear owner
                        if new_height == 0:
                            state.stack_owner[g, extraction_y, extraction_x] = 0
                            state.cap_height[g, extraction_y, extraction_x] = 0
                            # Clear all buried_at for this empty position
                            for pp in range(1, state.num_players + 1):
                                state.buried_at[g, pp, extraction_y, extraction_x] = 0
                        else:
                            # December 2025 BUG FIX: Recalculate cap_height properly.
                            # When a non-owner's buried ring is extracted, the cap may INCREASE
                            # if all remaining buried rings belong to the owner (consecutive
                            # owner rings from top now extend further down).
                            #
                            # Check if there are non-owner buried rings remaining AFTER
                            # this extraction. Key insight: the extracting player may still
                            # have MORE buried rings at this position (we only extracted one).
                            # Check buried_at count > 0 to determine if player still has buried at this pos.
                            owner = int(state.stack_owner[g, extraction_y, extraction_x].item())
                            # Player still has buried at THIS position if count > 1 (we're about to decrement)
                            player_still_has_buried_here = state.buried_at[g, p, extraction_y, extraction_x].item() > 1
                            other_player_buried_remaining = False
                            for pp in range(1, state.num_players + 1):
                                if pp != owner and pp != p and state.buried_at[g, pp, extraction_y, extraction_x].item() > 0:
                                    other_player_buried_remaining = True
                                    break

                            # Include extracting player's remaining buried rings in the check
                            non_owner_buried_remaining = (
                                other_player_buried_remaining or
                                (p != owner and player_still_has_buried_here)
                            )

                            # Calculate new cap_height FIRST (before decrementing buried_at)
                            if non_owner_buried_remaining:
                                # Still have non-owner buried rings, cap stays same or decreases
                                old_cap = state.cap_height[g, extraction_y, extraction_x].item()
                                new_cap = min(old_cap, new_height)
                            elif owner == p:
                                # Owner extracted from own stack, use conservative approach
                                # (owner may still have more buried rings here)
                                old_cap = state.cap_height[g, extraction_y, extraction_x].item()
                                new_cap = min(old_cap, new_height)
                            else:
                                # Non-owner extracted and no non-owner buried rings remaining
                                # All remaining rings belong to owner, cap = height
                                new_cap = new_height
                            state.cap_height[g, extraction_y, extraction_x] = new_cap

                            # Decrement buried_at count for the extracted position.
                            # Now properly tracks multiple buried rings at same location.
                            state.buried_at[g, p, extraction_y, extraction_x] -= 1


def reset_capture_chain_batch(
    state: BatchGameState,
    mask: torch.Tensor,
) -> None:
    """Reset capture chain tracking for games where turn is ending.

    This should be called after captures complete when transitioning
    to LINE_PROCESSING or next player's turn.

    December 2025: Added for canonical CAPTURE/CHAIN_CAPTURE phase support.

    Args:
        state: BatchGameState to modify
        mask: (batch_size,) bool tensor of games to reset
    """
    active_mask = mask & state.get_active_mask()
    if not active_mask.any():
        return

    game_indices = torch.where(active_mask)[0]
    state.in_capture_chain[game_indices] = False
    state.capture_chain_depth[game_indices] = 0


def mark_real_action_batch(
    state: BatchGameState,
    mask: torch.Tensor,
) -> None:
    """Mark that a real action occurred for games in the mask.

    Real actions are: placement, movement, capture
    Non-real actions are: recovery, bookkeeping (NO_*_ACTION)

    December 2025: Added for FORCED_ELIMINATION detection (RR-CANON-R160).

    Args:
        state: BatchGameState to modify
        mask: (batch_size,) bool tensor of games with real action
    """
    active_mask = mask & state.get_active_mask()
    if not active_mask.any():
        return

    game_indices = torch.where(active_mask)[0]
    state.turn_had_real_action[game_indices] = True


def reset_turn_tracking_batch(
    state: BatchGameState,
    mask: torch.Tensor,
) -> None:
    """Reset per-turn tracking for games starting a new turn.

    Resets: turn_had_real_action, in_capture_chain, capture_chain_depth

    December 2025: Added for canonical phase tracking.

    Args:
        state: BatchGameState to modify
        mask: (batch_size,) bool tensor of games starting new turn
    """
    active_mask = mask & state.get_active_mask()
    if not active_mask.any():
        return

    game_indices = torch.where(active_mask)[0]
    state.turn_had_real_action[game_indices] = False
    state.in_capture_chain[game_indices] = False
    state.capture_chain_depth[game_indices] = 0


def check_and_apply_forced_elimination_batch(
    state: BatchGameState,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Check for and apply FORCED_ELIMINATION for games in the mask.

    FORCED_ELIMINATION occurs when (RR-CANON-R160):
    - Player had no real action this turn (no placement, movement, or capture)
    - Player still has stacks on the board

    If triggered, this:
    - Selects a target stack (smallest cap height, then first position per RR-CANON-R100)
    - Eliminates the entire cap from the target stack
    - Records a FORCED_ELIMINATION move with target position
    - Updates eliminated_rings and rings_caused_eliminated counters
    - Transitions phase to FORCED_ELIMINATION
    - May lead to player elimination if no rings remain

    December 2025: Added for canonical phase parity.
    December 2025: Fixed to actually apply elimination (was only recording move).

    Args:
        state: BatchGameState to modify
        mask: (batch_size,) bool tensor of games to check (usually after territory processing)

    Returns:
        (batch_size,) bool tensor of games where forced elimination was triggered
    """
    active_mask = mask & state.get_active_mask()
    if not active_mask.any():
        return torch.zeros(state.batch_size, dtype=torch.bool, device=state.device)

    game_indices = torch.where(active_mask)[0]

    # Check conditions: no real action AND player has stacks
    no_real_action = ~state.turn_had_real_action[game_indices]

    # Check if current player has any stacks
    players = state.current_player[game_indices].long()
    has_stacks = torch.zeros(len(game_indices), dtype=torch.bool, device=state.device)
    for i, g in enumerate(game_indices.tolist()):
        player = players[i].item()
        player_stacks = (state.stack_owner[g] == player).sum()
        has_stacks[i] = player_stacks > 0

    # Forced elimination triggers when: no real action AND has stacks
    triggers = no_real_action & has_stacks

    if not triggers.any():
        return torch.zeros(state.batch_size, dtype=torch.bool, device=state.device)

    # Apply forced elimination for triggered games
    triggered_games = game_indices[triggers]
    triggered_players = players[triggers]

    # For each triggered game, find target stack and apply elimination
    # Per RR-CANON-R100: Select stack with smallest positive cap height, then first position
    target_positions_y = torch.full((len(triggered_games),), -1, dtype=torch.int32, device=state.device)
    target_positions_x = torch.full((len(triggered_games),), -1, dtype=torch.int32, device=state.device)
    eliminated_counts = torch.zeros(len(triggered_games), dtype=torch.int32, device=state.device)

    for i, (g, player) in enumerate(zip(triggered_games.tolist(), triggered_players.tolist(), strict=False)):
        # Find all stacks owned by this player
        player_mask = state.stack_owner[g] == player
        if not player_mask.any():
            continue

        # Get positions and cap heights of player's stacks
        positions = torch.where(player_mask)
        ys = positions[0]
        xs = positions[1]
        cap_heights = state.cap_height[g, ys, xs]

        # Select stack with smallest positive cap height (per RR-CANON-R100)
        # If all caps are 0, select first stack
        positive_caps = cap_heights > 0
        if positive_caps.any():
            # Find minimum positive cap height
            positive_cap_values = cap_heights[positive_caps]
            min_cap = positive_cap_values.min()
            # Get indices where cap == min_cap among positive caps
            candidate_mask = (cap_heights == min_cap) & positive_caps
        else:
            # All caps are 0, select first stack (fallback per TS behavior)
            candidate_mask = torch.ones_like(cap_heights, dtype=torch.bool)

        # Take the first candidate (deterministic ordering)
        candidate_indices = torch.where(candidate_mask)[0]
        if len(candidate_indices) == 0:
            continue

        target_idx = candidate_indices[0].item()
        target_y = ys[target_idx].item()
        target_x = xs[target_idx].item()
        target_cap = int(state.cap_height[g, target_y, target_x].item())

        # Per TS parity: eliminate max(1, cap_height) rings
        rings_to_eliminate = max(1, target_cap)

        target_positions_y[i] = target_y
        target_positions_x[i] = target_x
        eliminated_counts[i] = rings_to_eliminate

        # Apply elimination: reduce cap and stack height
        current_cap = int(state.cap_height[g, target_y, target_x].item())
        current_height = int(state.stack_height[g, target_y, target_x].item())

        new_cap = max(0, current_cap - rings_to_eliminate)
        new_height = max(0, current_height - rings_to_eliminate)

        state.cap_height[g, target_y, target_x] = new_cap
        state.stack_height[g, target_y, target_x] = new_height

        # If stack is eliminated (height == 0), clear ownership
        if new_height == 0:
            state.stack_owner[g, target_y, target_x] = 0

        # Update elimination counters
        # Player loses their own rings (eliminated_rings tracks rings LOST by player)
        state.eliminated_rings[g, player] += rings_to_eliminate
        # Player causes elimination of their own rings (self-elimination counts for victory)
        state.rings_caused_eliminated[g, player] += rings_to_eliminate

    # Record FORCED_ELIMINATION move with target position
    move_idx = state.move_count[triggered_games]
    history_mask = move_idx < state.max_history_moves
    if history_mask.any():
        hist_games = triggered_games[history_mask]
        hist_move_idx = move_idx[history_mask].long()
        hist_dtype = state.move_history.dtype
        hist_target_y = target_positions_y[history_mask]
        hist_target_x = target_positions_x[history_mask]

        state.move_history[hist_games, hist_move_idx, 0] = MoveType.FORCED_ELIMINATION
        state.move_history[hist_games, hist_move_idx, 1] = triggered_players[history_mask].to(hist_dtype)
        state.move_history[hist_games, hist_move_idx, 2] = -1  # from_y (not used)
        state.move_history[hist_games, hist_move_idx, 3] = -1  # from_x (not used)
        state.move_history[hist_games, hist_move_idx, 4] = hist_target_y.to(hist_dtype)  # to_y: target stack
        state.move_history[hist_games, hist_move_idx, 5] = hist_target_x.to(hist_dtype)  # to_x: target stack
        state.move_history[hist_games, hist_move_idx, 6] = GamePhase.FORCED_ELIMINATION

    state.move_count[triggered_games] += 1
    state.current_phase[triggered_games] = GamePhase.FORCED_ELIMINATION

    # Build result mask
    result = torch.zeros(state.batch_size, dtype=torch.bool, device=state.device)
    result[triggered_games] = True
    return result


def apply_no_action_moves_batch(
    state: BatchGameState,
    mask: torch.Tensor,
) -> None:
    """Record a phase-specific NO_*_ACTION move for each masked active game.

    This is used to avoid silent phase progression when a player has no
    legal action in an interactive phase (RR-CANON-R075).

    December 2025: Updated to use canonical phase-specific move types:
    - RING_PLACEMENT → NO_PLACEMENT_ACTION
    - MOVEMENT → NO_MOVEMENT_ACTION
    - LINE_PROCESSING → NO_LINE_ACTION
    - TERRITORY_PROCESSING → NO_TERRITORY_ACTION
    - Others → NO_ACTION (fallback)

    Optimized 2025-12-13: Eliminated Python loops and .item() calls.
    """
    active_mask = mask & state.get_active_mask()
    if not active_mask.any():
        return

    game_indices = torch.where(active_mask)[0]
    move_idx = state.move_count[game_indices]
    players = state.current_player[game_indices]
    phases = state.current_phase[game_indices]

    # Map current phase to canonical NO_*_ACTION move type
    # Create tensor of move types based on phase
    move_types = torch.full_like(phases, MoveType.NO_ACTION, dtype=torch.int16)
    move_types[phases == GamePhase.RING_PLACEMENT] = MoveType.NO_PLACEMENT_ACTION
    move_types[phases == GamePhase.MOVEMENT] = MoveType.NO_MOVEMENT_ACTION
    move_types[phases == GamePhase.LINE_PROCESSING] = MoveType.NO_LINE_ACTION
    move_types[phases == GamePhase.TERRITORY_PROCESSING] = MoveType.NO_TERRITORY_ACTION
    # CAPTURE/CHAIN_CAPTURE phases use SKIP_CAPTURE
    move_types[phases == GamePhase.CAPTURE] = MoveType.SKIP_CAPTURE
    move_types[phases == GamePhase.CHAIN_CAPTURE] = MoveType.SKIP_CAPTURE
    # GPU internal recovery phase uses SKIP_RECOVERY (exported as movement).
    move_types[phases == GamePhase.RECOVERY] = MoveType.SKIP_RECOVERY

    # Record in history for games with space (7 columns: move_type, player, from_y, from_x, to_y, to_x, phase)
    history_mask = move_idx < state.max_history_moves
    if history_mask.any():
        hist_games = game_indices[history_mask]
        hist_move_idx = move_idx[history_mask].long()
        hist_dtype = state.move_history.dtype
        state.move_history[hist_games, hist_move_idx, 0] = move_types[history_mask].to(hist_dtype)
        state.move_history[hist_games, hist_move_idx, 1] = players[history_mask].to(hist_dtype)
        state.move_history[hist_games, hist_move_idx, 2] = -1
        state.move_history[hist_games, hist_move_idx, 3] = -1
        state.move_history[hist_games, hist_move_idx, 4] = -1
        state.move_history[hist_games, hist_move_idx, 5] = -1
        state.move_history[hist_games, hist_move_idx, 6] = phases[history_mask].to(hist_dtype)

    state.move_count[game_indices] += 1


# =============================================================================
# Batch Apply Functions (main API)
# =============================================================================


def apply_placement_moves_batch_vectorized(
    state: BatchGameState,
    move_indices: torch.Tensor,
    moves: BatchMoves,
) -> None:
    """Vectorized placement move application.

    Args:
        state: BatchGameState to modify
        move_indices: (batch_size,) index into moves for each game's selected move
        moves: BatchMoves containing all candidate moves
    """
    device = state.device
    active_mask = state.get_active_mask()

    # Filter to games with valid moves
    # Bug fix 2025-12-20: Check move_indices >= 0 to handle games that weren't
    # selected for this move type (selection returns -1 for inactive games)
    has_moves = moves.moves_per_game > 0
    valid_local_idx = (move_indices >= 0) & (move_indices < moves.moves_per_game)
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
    # 7 columns: move_type, player, from_y, from_x, to_y, to_x, phase
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
        state.move_history[hist_games, hist_move_idx, 6] = state.current_phase[hist_games].to(hist_dtype)

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
    # All rings in the previous owner's cap become buried when control flips.
    is_opponent = ~is_empty & (dest_owner != 0) & (dest_owner != players)
    if is_opponent.any():
        opp_games = game_indices[is_opponent]
        opp_owners = dest_owner[is_opponent].long()
        opp_y = y[is_opponent]
        opp_x = x[is_opponent]
        opp_caps = dest_cap[is_opponent].to(state.buried_rings.dtype)
        state.buried_rings.index_put_(
            (opp_games, opp_owners),
            opp_caps,
            accumulate=True
        )
        # Track buried ring count at position (December 2025 - recovery fix)
        # This enables recovery to correctly find and decrement stacks
        state.buried_at.index_put_(
            (opp_games, opp_owners, opp_y, opp_x),
            opp_caps.to(state.buried_at.dtype),
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
    state: BatchGameState,
    move_indices: torch.Tensor,
    moves: BatchMoves,
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
            state.move_history[g, move_idx, 6] = int(state.current_phase[g].item())

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
                state.buried_rings[g, dest_owner] += dest_cap
                # Track buried ring count at position (December 2025 - recovery fix)
                state.buried_at[g, dest_owner, y, x] += dest_cap
        state.rings_in_hand[g, player] -= 1
        state.must_move_from_y[g] = y
        state.must_move_from_x[g] = x
        state.move_count[g] += 1


def apply_placement_moves_batch(
    state: BatchGameState,
    move_indices: torch.Tensor,
    moves: BatchMoves,
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
    state: BatchGameState,
    move_indices: torch.Tensor,
    moves: BatchMoves,
) -> None:
    """Vectorized movement move application.

    Args:
        state: BatchGameState to modify
        move_indices: (batch_size,) index into moves for each game's selected move
        moves: BatchMoves containing all candidate moves
    """
    device = state.device
    board_size = state.board_size
    active_mask = state.get_active_mask()

    # Filter to games with valid moves
    # Bug fix 2025-12-20: Check move_indices >= 0 to handle games that weren't
    # selected for this move type (selection returns -1 for inactive games)
    has_moves = moves.moves_per_game > 0
    valid_local_idx = (move_indices >= 0) & (move_indices < moves.moves_per_game)
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

    # Record move history (7 columns: move_type, player, from_y, from_x, to_y, to_x, phase)
    # CANONICAL: Movements are always recorded with MOVEMENT phase, regardless of
    # current state phase. This ensures phase/move contract compliance.
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
        # Always use MOVEMENT phase for movement moves (canonical contract compliance)
        state.move_history[hist_games, hist_move_idx, 6] = GamePhase.MOVEMENT

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

        # Find own markers to collapse to territory (per RR-CANON-R090)
        is_own_marker = (path_marker_owners == players_exp) & valid_path
        if is_own_marker.any():
            collapse_games = game_indices_exp[is_own_marker]
            collapse_y = path_y_safe[is_own_marker]
            collapse_x = path_x_safe[is_own_marker]
            collapse_players = players_exp[is_own_marker]
            # Remove marker and convert to territory
            state.marker_owner[collapse_games, collapse_y, collapse_x] = 0
            state.is_collapsed[collapse_games, collapse_y, collapse_x] = True
            state.territory_owner[collapse_games, collapse_y, collapse_x] = collapse_players.to(
                state.territory_owner.dtype
            )
            # Update territory count per player
            # Use scatter_add to count collapsed markers per (game, player) pair
            for g, p in zip(collapse_games.tolist(), collapse_players.tolist(), strict=False):
                state.territory_count[g, p] += 1

    # Handle landing on ANY marker (per RR-CANON-R091/R092)
    # Landing on any marker (own or opponent) removes it and costs 1 ring (cap-elimination)
    dest_marker = state.marker_owner[game_indices, to_y, to_x]
    landing_on_any_marker = dest_marker != 0
    landing_ring_cost = landing_on_any_marker.int()

    if landing_on_any_marker.any():
        marker_games = game_indices[landing_on_any_marker]
        marker_y = to_y[landing_on_any_marker]
        marker_x = to_x[landing_on_any_marker]
        marker_players = players[landing_on_any_marker]
        # Remove the marker
        state.marker_owner[marker_games, marker_y, marker_x] = 0
        # Track eliminated rings from landing cost (cap-elimination)
        # The moving player loses a ring from their cap
        for g, p in zip(marker_games.tolist(), marker_players.tolist(), strict=False):
            state.eliminated_rings[g, p] += 1
            state.rings_caused_eliminated[g, p] += 1

    # Place departure marker at origin (per RR-CANON-R090)
    state.marker_owner[game_indices, from_y, from_x] = players.to(state.marker_owner.dtype)

    # Clear origin stack
    state.stack_owner[game_indices, from_y, from_x] = 0
    state.stack_height[game_indices, from_y, from_x] = 0
    state.cap_height[game_indices, from_y, from_x] = 0

    # Move buried_at tracking from origin to destination (December 2025 - recovery fix)
    # Buried rings move with the stack, so we need to update the position tracking
    # Add buried_at counts from origin to destination, then clear origin
    for p in range(1, state.num_players + 1):
        state.buried_at[game_indices, p, to_y, to_x] += state.buried_at[game_indices, p, from_y, from_x]
        state.buried_at[game_indices, p, from_y, from_x] = 0

    # Set destination
    # BUG FIX December 2025: Allow height to go to 0 (stack elimination) when landing on marker
    # Previously used min=1 which incorrectly prevented stack elimination.
    # Now we allow height 0, which means the stack is eliminated.
    new_height = torch.clamp(moving_height - landing_ring_cost, min=0)
    new_cap_height = torch.clamp(moving_cap_height - landing_ring_cost, min=0)

    # Handle stack elimination: if new_height is 0, clear the stack entirely
    is_eliminated = new_height == 0
    is_surviving = ~is_eliminated

    # For surviving stacks, set the destination
    if is_surviving.any():
        surv_games = game_indices[is_surviving]
        surv_to_y = to_y[is_surviving]
        surv_to_x = to_x[is_surviving]
        surv_players = players[is_surviving]
        surv_height = new_height[is_surviving]
        surv_cap = new_cap_height[is_surviving]

        # December 2025 BUG FIX: When cap is eliminated but stack survives (buried rings),
        # ownership must transfer to the buried ring owner. This happens when landing on
        # a marker costs the moving player's entire cap (e.g., h=2 c=1 lands on marker,
        # cap eliminated, only buried ring remains, ownership transfers to buried owner).
        needs_ownership_transfer = (surv_cap == 0) & (surv_height > 0)
        final_owners = surv_players.clone()
        final_caps = surv_cap.clone()

        if needs_ownership_transfer.any():
            transfer_mask = needs_ownership_transfer
            transfer_games = surv_games[transfer_mask]
            transfer_y = surv_to_y[transfer_mask]
            transfer_x = surv_to_x[transfer_mask]

            # Find the buried ring owner for each game needing transfer
            for idx in range(len(transfer_games)):
                g = transfer_games[idx].item()
                y_pos = transfer_y[idx].item()
                x_pos = transfer_x[idx].item()

                # Find which player has buried ring at this position
                for p in range(1, state.num_players + 1):
                    buried_count = state.buried_at[g, p, y_pos, x_pos].item()
                    if buried_count > 0:
                        # Get index in the transfer subset
                        surv_idx = torch.where(transfer_mask)[0][idx]
                        final_owners[surv_idx] = p
                        final_caps[surv_idx] = 1  # Exposed buried ring becomes the cap
                        # Decrement buried_at AND buried_rings since one ring is now exposed
                        state.buried_at[g, p, y_pos, x_pos] -= 1
                        state.buried_rings[g, p] -= 1
                        break

        state.stack_owner[surv_games, surv_to_y, surv_to_x] = final_owners.to(state.stack_owner.dtype)
        state.stack_height[surv_games, surv_to_y, surv_to_x] = surv_height.to(state.stack_height.dtype)
        state.cap_height[surv_games, surv_to_y, surv_to_x] = final_caps.to(state.cap_height.dtype)

    # For eliminated stacks, ensure destination remains empty
    if is_eliminated.any():
        elim_games = game_indices[is_eliminated]
        elim_to_y = to_y[is_eliminated]
        elim_to_x = to_x[is_eliminated]
        state.stack_owner[elim_games, elim_to_y, elim_to_x] = 0
        state.stack_height[elim_games, elim_to_y, elim_to_x] = 0
        state.cap_height[elim_games, elim_to_y, elim_to_x] = 0
        # Also clear buried_at AND decrement buried_rings since the stack is gone
        # BUG FIX 2025-12-20: buried_rings wasn't decremented when stacks were eliminated,
        # causing divergence with CPU buried ring counting
        for p in range(1, state.num_players + 1):
            # Check which games had buried rings for player p at this position
            buried_counts = state.buried_at[elim_games, p, elim_to_y, elim_to_x]
            has_buried = buried_counts > 0
            if has_buried.any():
                # Decrement buried_rings by the count at each position
                games_with_buried = elim_games[has_buried]
                counts_to_subtract = buried_counts[has_buried]
                for g, c in zip(games_with_buried.tolist(), counts_to_subtract.tolist(), strict=False):
                    state.buried_rings[g, p] -= c
            state.buried_at[elim_games, p, elim_to_y, elim_to_x] = 0

    # Clear must_move_from constraint after movement (RR-CANON-R090)
    # The player has fulfilled their movement obligation.
    # BUG FIX 2025-12-20: This was missing in the vectorized path, causing post-movement
    # capture checks to fail because they still looked at the old position.
    state.must_move_from_y[game_indices] = -1
    state.must_move_from_x[game_indices] = -1

    # Advance move counter only (NOT current_player - that's handled by END_TURN phase)
    # BUG FIX 2025-12-15: Removing player rotation here - it was causing players to
    # advance mid-turn, then advance again in END_TURN, resulting in players getting
    # multiple consecutive turns before opponents.
    state.move_count[game_indices] += 1


def _apply_movement_moves_batch_legacy(
    state: BatchGameState,
    move_indices: torch.Tensor,
    moves: BatchMoves,
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
            state.move_history[g, move_idx, 6] = int(state.current_phase[g].item())

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

        # December 2025 BUG FIX: Handle landing on ANY marker (own or opponent)
        # per RR-CANON-R091/R092. Previously only handled own markers.
        dest_marker = state.marker_owner[g, to_y, to_x].item()
        landing_ring_cost = 1 if dest_marker != 0 else 0
        if dest_marker != 0:
            state.marker_owner[g, to_y, to_x] = 0

        state.stack_owner[g, from_y, from_x] = 0
        state.stack_height[g, from_y, from_x] = 0
        state.cap_height[g, from_y, from_x] = 0

        # Move buried_at tracking from origin to destination (December 2025 - recovery fix)
        for p in range(1, state.num_players + 1):
            state.buried_at[g, p, to_y, to_x] += state.buried_at[g, p, from_y, from_x].item()
            state.buried_at[g, p, from_y, from_x] = 0

        # BUG FIX December 2025: Allow height to go to 0 (stack elimination) when landing on marker
        new_height = max(0, moving_height - landing_ring_cost)
        new_cap_height = max(0, moving_cap_height - landing_ring_cost)

        if new_height > 0:
            # December 2025 BUG FIX: When cap is eliminated but stack survives,
            # ownership must transfer to the buried ring owner.
            final_owner = player
            final_cap = new_cap_height
            if new_cap_height == 0:
                # Cap eliminated, find buried ring owner
                for p in range(1, state.num_players + 1):
                    buried_count = state.buried_at[g, p, to_y, to_x].item()
                    if buried_count > 0:
                        final_owner = p
                        final_cap = 1
                        state.buried_at[g, p, to_y, to_x] -= 1
                        state.buried_rings[g, p] -= 1
                        break
            state.stack_owner[g, to_y, to_x] = final_owner
            state.stack_height[g, to_y, to_x] = new_height
            state.cap_height[g, to_y, to_x] = final_cap
        else:
            # Stack is eliminated (landed on marker with height 1)
            state.stack_owner[g, to_y, to_x] = 0
            state.stack_height[g, to_y, to_x] = 0
            state.cap_height[g, to_y, to_x] = 0
            # Clear buried_at AND decrement buried_rings since the stack is gone
            for p in range(1, state.num_players + 1):
                buried_count = state.buried_at[g, p, to_y, to_x].item()
                if buried_count > 0:
                    state.buried_rings[g, p] -= buried_count
                state.buried_at[g, p, to_y, to_x] = 0

        # Advance move counter only (NOT current_player - that's handled by END_TURN phase)
        # BUG FIX 2025-12-15: See apply_movement_moves_batch_vectorized for details
        state.move_count[g] += 1


def apply_movement_moves_batch(
    state: BatchGameState,
    move_indices: torch.Tensor,
    moves: BatchMoves,
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
    state: BatchGameState,
    move_indices: torch.Tensor,
    moves: BatchMoves,
) -> None:
    """Vectorized capture move application with canonical phase tracking.

    December 2025: Updated for canonical CAPTURE/CHAIN_CAPTURE phases:
    - First capture uses OVERTAKING_CAPTURE and CAPTURE phase
    - Chain captures use CONTINUE_CAPTURE_SEGMENT and CHAIN_CAPTURE phase

    Args:
        state: BatchGameState to modify
        move_indices: (batch_size,) index into moves for each game's selected move
        moves: BatchMoves containing all candidate moves
    """
    device = state.device
    board_size = state.board_size
    active_mask = state.get_active_mask()

    # Filter to games with valid moves
    # Bug fix 2025-12-20: Check move_indices >= 0 to handle games that weren't
    # selected for this move type (selection returns -1 for inactive games)
    has_moves = moves.moves_per_game > 0
    valid_local_idx = (move_indices >= 0) & (move_indices < moves.moves_per_game)
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
    players = state.current_player[game_indices]

    # Determine canonical move type based on capture chain state
    # First capture → OVERTAKING_CAPTURE, chain captures → CONTINUE_CAPTURE_SEGMENT
    is_chain_capture = state.in_capture_chain[game_indices]
    canonical_move_type = torch.where(
        is_chain_capture,
        torch.full((n_games,), MoveType.CONTINUE_CAPTURE_SEGMENT, dtype=torch.int16, device=device),
        torch.full((n_games,), MoveType.OVERTAKING_CAPTURE, dtype=torch.int16, device=device),
    )

    # Determine canonical phase based on capture chain state and current game phase
    # Per RR-CANON phase machine:
    # - Direct capture during MOVEMENT phase → record with MOVEMENT phase
    # - First capture after entering CAPTURE phase (via MOVE_STACK) → CAPTURE phase
    # - Chain captures → CHAIN_CAPTURE phase
    #
    # The key distinction is: captures ARE valid during MOVEMENT phase, and when
    # taken directly (without prior MOVE_STACK), they should be recorded with
    # MOVEMENT phase. CAPTURE phase is only entered after a non-capture MOVE_STACK
    # lands where captures are available.
    is_first_capture_during_movement = ~is_chain_capture & (
        state.current_phase[game_indices] == GamePhase.MOVEMENT
    )
    canonical_phase = torch.where(
        is_chain_capture,
        torch.full((n_games,), GamePhase.CHAIN_CAPTURE, dtype=torch.int8, device=device),
        torch.where(
            is_first_capture_during_movement,
            torch.full((n_games,), GamePhase.MOVEMENT, dtype=torch.int8, device=device),
            torch.full((n_games,), GamePhase.CAPTURE, dtype=torch.int8, device=device),
        ),
    )

    # Record move history
    # 9 columns: move_type, player, from_y, from_x, to_y, to_x, phase, capture_target_y, capture_target_x
    # Note: capture_target columns added December 2025 for canonical export parity
    move_idx = state.move_count[game_indices]
    history_mask = move_idx < state.max_history_moves
    # Compute target positions BEFORE recording history (needed for columns 7, 8)
    # NOTE: `to` is the LANDING position, not the target. We must scan the ray to find the target.
    dy_hist = torch.sign(to_y - from_y)
    dx_hist = torch.sign(to_x - from_x)
    dist_hist = torch.maximum(torch.abs(to_y - from_y), torch.abs(to_x - from_x))
    max_dist_hist = dist_hist.max().item() if n_games > 0 else 0
    if max_dist_hist > 0:
        steps_hist = torch.arange(1, max_dist_hist + 1, device=device).view(1, -1)
        ray_y_hist = from_y.unsqueeze(1) + dy_hist.unsqueeze(1) * steps_hist
        ray_x_hist = from_x.unsqueeze(1) + dx_hist.unsqueeze(1) * steps_hist
        ray_y_safe_hist = torch.clamp(ray_y_hist, 0, board_size - 1).long()
        ray_x_safe_hist = torch.clamp(ray_x_hist, 0, board_size - 1).long()
        game_indices_exp_hist = game_indices.unsqueeze(1).expand(-1, max_dist_hist)
        ray_owners_hist = state.stack_owner[game_indices_exp_hist, ray_y_safe_hist, ray_x_safe_hist]
        dist_exp_hist = dist_hist.unsqueeze(1).expand(-1, max_dist_hist)
        within_landing_hist = steps_hist < dist_exp_hist
        ray_has_stack_hist = (ray_owners_hist != 0) & within_landing_hist
        target_step_idx_hist = ray_has_stack_hist.to(torch.int32).argmax(dim=1)
        target_y_hist = from_y + dy_hist * (target_step_idx_hist + 1)
        target_x_hist = from_x + dx_hist * (target_step_idx_hist + 1)
    else:
        target_y_hist = to_y
        target_x_hist = to_x
    if history_mask.any():
        hist_games = game_indices[history_mask]
        hist_move_idx = move_idx[history_mask].long()
        hist_dtype = state.move_history.dtype
        state.move_history[hist_games, hist_move_idx, 0] = canonical_move_type[history_mask].to(hist_dtype)
        state.move_history[hist_games, hist_move_idx, 1] = players[history_mask].to(hist_dtype)
        state.move_history[hist_games, hist_move_idx, 2] = from_y[history_mask].to(hist_dtype)
        state.move_history[hist_games, hist_move_idx, 3] = from_x[history_mask].to(hist_dtype)
        state.move_history[hist_games, hist_move_idx, 4] = to_y[history_mask].to(hist_dtype)
        state.move_history[hist_games, hist_move_idx, 5] = to_x[history_mask].to(hist_dtype)
        state.move_history[hist_games, hist_move_idx, 6] = canonical_phase[history_mask].to(hist_dtype)
        # December 2025: Record capture target for canonical export
        state.move_history[hist_games, hist_move_idx, 7] = target_y_hist[history_mask].to(hist_dtype)
        state.move_history[hist_games, hist_move_idx, 8] = target_x_hist[history_mask].to(hist_dtype)

    # Update capture chain tracking
    state.in_capture_chain[game_indices] = True
    state.capture_chain_depth[game_indices] += 1
    # After any capture, subsequent captures are chain captures (per CPU phase machine).
    # We transition to CHAIN_CAPTURE regardless of whether the first capture was
    # during MOVEMENT or CAPTURE phase. The chain capture loop in _step_movement_phase
    # will check for more captures and either continue the chain or advance phase.
    state.current_phase[game_indices] = GamePhase.CHAIN_CAPTURE

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
        target_step_idx = ray_has_stack.to(torch.int32).argmax(dim=1)  # (n_games,)

        # Compute target positions
        target_y = from_y + dy * (target_step_idx + 1)
        target_x = from_x + dx * (target_step_idx + 1)
    else:
        # Fallback: no distance, treat to as target
        target_y = to_y
        target_x = to_x

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

    # Note: Captured rings are BURIED, not ELIMINATED. They go under the attacker's stack
    # and can be liberated if the attacker's stack is later captured.
    # We track buried_rings below, not eliminated_rings here.

    # === Update TARGET stack (reduce height by 1, capturing the top ring) ===
    # December 2025: BUG FIX - When capturing the target's entire cap, ownership
    # transfers to the opponent. The captured ring was the top of the cap, so if
    # cap_height was 1 (only 1 ring of defender's color), all remaining rings
    # belong to the opponent.
    new_target_height = torch.clamp(defender_height - 1, min=0)
    state.stack_height[game_indices, target_y, target_x] = new_target_height.to(state.stack_height.dtype)

    # Check if target's cap was fully captured
    target_cap_fully_captured = defender_cap_height <= 1  # Cap will be 0 after -1
    target_is_empty = new_target_height == 0

    # Determine new owner: if cap fully captured and stack not empty, ownership transfers
    # In 2-player game, if defender was player P, opponent is 3-P (player 1 or 2)
    opponent = torch.where(
        defender_owner == 1,
        torch.full_like(defender_owner, 2),
        torch.ones_like(defender_owner)
    )
    new_target_owner = torch.where(
        target_is_empty,
        torch.zeros_like(defender_owner),  # Empty stack has no owner
        torch.where(
            target_cap_fully_captured,
            opponent,  # Cap captured, ownership transfers
            defender_owner  # Cap not fully captured, defender keeps ownership
        )
    )
    state.stack_owner[game_indices, target_y, target_x] = new_target_owner.to(state.stack_owner.dtype)

    # Update target cap height
    # If cap fully captured, new cap is all remaining rings (owned by opponent)
    new_target_cap = torch.where(
        target_cap_fully_captured,
        new_target_height,  # All remaining rings are opponent's cap
        torch.clamp(defender_cap_height - 1, min=1)
    )
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
            torch.ones(int(target_owner_nonzero.sum().item()), dtype=state.buried_rings.dtype, device=device),
            accumulate=True
        )
        # Track buried ring count at landing (December 2025 - recovery fix)
        # The captured ring goes under the attacker's stack at the landing position
        tnz_games = game_indices[target_owner_nonzero]
        tnz_owners = defender_owner[target_owner_nonzero].long()
        tnz_to_y = to_y[target_owner_nonzero]
        tnz_to_x = to_x[target_owner_nonzero]
        state.buried_at[tnz_games, tnz_owners, tnz_to_y, tnz_to_x] += 1

    # If target stack is eliminated, clear any buried_at at target position
    # (those buried rings are also eliminated when the stack is destroyed)
    # BUG FIX 2025-12-20: Also decrement buried_rings when stack is eliminated
    if target_is_empty.any():
        empty_games = game_indices[target_is_empty]
        empty_target_y = target_y[target_is_empty]
        empty_target_x = target_x[target_is_empty]
        for p in range(1, state.num_players + 1):
            # Check which games had buried rings for player p at this position
            buried_counts = state.buried_at[empty_games, p, empty_target_y, empty_target_x]
            has_buried = buried_counts > 0
            if has_buried.any():
                games_with_buried = empty_games[has_buried]
                counts_to_subtract = buried_counts[has_buried]
                for g, c in zip(games_with_buried.tolist(), counts_to_subtract.tolist(), strict=False):
                    state.buried_rings[g, p] -= c
            state.buried_at[empty_games, p, empty_target_y, empty_target_x] = 0

    # BUG FIX 2025-12-20: When ownership transfers due to cap capture, any previously
    # buried rings of the new owner are now exposed (they become the cap). Clear
    # buried_at and decrement buried_rings for these cases.
    # This happens when: target_cap_fully_captured AND the new owner had buried rings here
    if target_cap_fully_captured.any():
        cap_games = game_indices[target_cap_fully_captured]
        cap_target_y = target_y[target_cap_fully_captured]
        cap_target_x = target_x[target_cap_fully_captured]
        cap_new_owners = new_target_owner[target_cap_fully_captured]
        for i in range(cap_games.shape[0]):
            g = cap_games[i].item()
            y = cap_target_y[i].item()
            x = cap_target_x[i].item()
            new_owner = cap_new_owners[i].item()
            if new_owner > 0:  # Skip if no owner
                # Check if the new owner had buried rings here (now exposed)
                buried_count = state.buried_at[g, new_owner, y, x].item()
                if buried_count > 0:
                    state.buried_at[g, new_owner, y, x] -= 1
                    state.buried_rings[g, new_owner] -= 1

    # === Clear ORIGIN ===
    state.stack_owner[game_indices, from_y, from_x] = 0
    state.stack_height[game_indices, from_y, from_x] = 0
    state.cap_height[game_indices, from_y, from_x] = 0

    # Move buried_at tracking from attacker origin to landing (December 2025 - recovery fix)
    # Any buried rings in the attacker's stack move with it
    for p in range(1, state.num_players + 1):
        # Add origin counts to landing counts (from the captured ring added above)
        state.buried_at[game_indices, p, to_y, to_x] += state.buried_at[game_indices, p, from_y, from_x]
        state.buried_at[game_indices, p, from_y, from_x] = 0

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
    # December 2025: BUG FIX - When landing marker eliminates the attacker's entire cap,
    # ownership transfers to the target's original owner. The captured ring goes to the
    # bottom, so after cap elimination, all remaining rings are from the target's owner.
    new_height = torch.clamp(attacker_height + 1 - landing_ring_cost, min=1, max=5)

    # Check if cap is fully eliminated by landing cost
    cap_fully_eliminated = landing_ring_cost >= attacker_cap_height

    # Check if attacker has buried rings (opponent's rings under their cap)
    attacker_has_buried = attacker_cap_height < attacker_height

    # SELF-CAPTURE without buried rings:
    # If attacker has no buried rings (cap == height) and target is same owner,
    # the entire resulting stack is same color, so cap = new_height.
    is_self_capture_no_buried = (defender_owner == players) & ~attacker_has_buried

    # December 2025: BUG FIX - When cap is eliminated by landing cost AND attacker
    # has buried rings, ownership transfers to the opponent (who owns those buried
    # rings), NOT to the defender_owner. The remaining stack after cap elimination:
    # - Captured ring (target's color) at bottom
    # - Buried opponent rings become the new cap
    cap_elim_with_buried = cap_fully_eliminated & attacker_has_buried
    if state.num_players != 2:
        # For 3-4 players, buried ring ordering is not tracked on GPU; keep legacy owner.
        cap_elim_with_buried = torch.zeros_like(cap_elim_with_buried, dtype=torch.bool)

    # Calculate new cap height based on:
    # 1. Cap eliminated with buried: new_cap = buried_count = attacker_height - attacker_cap
    # 2. Cap eliminated no buried: new_cap = new_height (all target's color)
    # 3. Self-capture without buried rings: new_cap = new_height
    # 4. Otherwise: new_cap = attacker_cap - landing_cost
    buried_count = attacker_height - attacker_cap_height
    new_cap_height = torch.where(
        cap_elim_with_buried,
        buried_count,  # Buried opponent rings become the new cap
        torch.where(
            cap_fully_eliminated,
            new_height,  # All remaining rings are target's color
            torch.where(
                is_self_capture_no_buried,
                torch.clamp(new_height, min=1),  # Self-capture: cap = new_height
                torch.clamp(attacker_cap_height - landing_ring_cost, min=1)
            )
        )
    )
    new_cap_height = torch.minimum(new_cap_height, new_height)

    # Determine new owner
    # When cap eliminated with buried rings, opponent takes ownership
    opponent = torch.where(
        players == 1,
        torch.full_like(players, 2),
        torch.ones_like(players)
    )
    new_owner = torch.where(
        cap_elim_with_buried,
        opponent,  # Buried rings' owner (opponent of attacker)
        torch.where(
            cap_fully_eliminated,
            defender_owner,  # Target's original owner
            players  # Capturing player
        )
    )

    state.stack_owner[game_indices, to_y, to_x] = new_owner.to(state.stack_owner.dtype)
    state.stack_height[game_indices, to_y, to_x] = new_height.to(state.stack_height.dtype)
    state.cap_height[game_indices, to_y, to_x] = new_cap_height.to(state.cap_height.dtype)

    # December 2025: When cap is eliminated with buried rings, those rings are now
    # exposed (they became the cap). Decrement buried_at and buried_rings.
    if cap_elim_with_buried.any():
        elim_games = game_indices[cap_elim_with_buried]
        elim_to_y = to_y[cap_elim_with_buried]
        elim_to_x = to_x[cap_elim_with_buried]
        elim_opponent = opponent[cap_elim_with_buried]
        for i in range(elim_games.shape[0]):
            g = elim_games[i].item()
            y = elim_to_y[i].item()
            x = elim_to_x[i].item()
            opp = elim_opponent[i].item()
            buried_count = state.buried_at[g, opp, y, x].item()
            if buried_count > 0:
                state.buried_at[g, opp, y, x] -= 1
                state.buried_rings[g, opp] -= 1

    # Note: No marker at landing - there's a stack there. marker_owner should remain 0.
    # (Already cleared on line 1134-1138 if there was a landing marker)

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
    state: BatchGameState,
    move_indices: torch.Tensor,
    moves: BatchMoves,
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

        attacker_height = state.stack_height[g, from_y, from_x].item()
        attacker_cap_height = state.cap_height[g, from_y, from_x].item()

        # Note: Legacy capture uses to_y/to_x as target, not landing
        defender_owner = state.stack_owner[g, to_y, to_x].item()
        defender_height = state.stack_height[g, to_y, to_x].item()

        # Record in history (9 columns: move_type, player, from_y, from_x, to_y, to_x, phase, capture_target_y, capture_target_x)
        # December 2025: Added capture target columns for canonical export
        move_idx = state.move_count[g].item()
        if move_idx < state.max_history_moves:
            state.move_history[g, move_idx, 0] = move_type
            state.move_history[g, move_idx, 1] = player
            state.move_history[g, move_idx, 2] = from_y
            state.move_history[g, move_idx, 3] = from_x
            state.move_history[g, move_idx, 4] = to_y
            state.move_history[g, move_idx, 5] = to_x
            state.move_history[g, move_idx, 6] = int(state.current_phase[g].item())
            # Legacy capture: to_y/to_x IS the target (not landing)
            state.move_history[g, move_idx, 7] = to_y
            state.move_history[g, move_idx, 8] = to_x

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

        # Move buried_at tracking from attacker origin to target (December 2025 - recovery fix)
        # Note: Legacy capture merges at target position, not landing
        # Updated to count-based: add origin counts to target, then zero origin
        for p in range(1, state.num_players + 1):
            state.buried_at[g, p, to_y, to_x] = state.buried_at[g, p, to_y, to_x] + state.buried_at[g, p, from_y, from_x]
            state.buried_at[g, p, from_y, from_x] = 0

        new_height = attacker_height + defender_height - 1
        # SELF-CAPTURE without buried rings:
        if defender_owner == player and attacker_cap_height == attacker_height:
            # Per RR-CANON-R101/R102, captured ring goes to bottom of stack.
            # If attacker has no buried rings (cap == height), and target is same color,
            # the entire resulting stack is same color, so cap = new_height.
            new_cap_height = new_height
        else:
            # ENEMY CAPTURE or SELF-CAPTURE with buried rings:
            # Captured ring goes to BOTTOM, doesn't extend cap
            new_cap_height = attacker_cap_height
        state.stack_owner[g, to_y, to_x] = player
        state.stack_height[g, to_y, to_x] = min(5, new_height)
        state.cap_height[g, to_y, to_x] = min(5, new_cap_height)

        state.marker_owner[g, to_y, to_x] = player

        # Advance move counter only (NOT current_player - that's handled by END_TURN phase)
        # BUG FIX 2025-12-15: See apply_capture_moves_batch_vectorized for details
        state.move_count[g] += 1


def apply_capture_moves_batch(
    state: BatchGameState,
    move_indices: torch.Tensor,
    moves: BatchMoves,
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
    '_apply_capture_moves_batch_legacy',
    '_apply_movement_moves_batch_legacy',
    '_apply_placement_moves_batch_legacy',
    'apply_capture_moves_batch',
    'apply_capture_moves_batch_vectorized',
    # Vectorized apply functions (for move selection)
    'apply_capture_moves_vectorized',
    'apply_movement_moves_batch',
    'apply_movement_moves_batch_vectorized',
    'apply_movement_moves_vectorized',
    'apply_no_action_moves_batch',
    'apply_placement_moves_batch',
    # Batch apply functions (main API)
    'apply_placement_moves_batch_vectorized',
    'apply_recovery_moves_vectorized',
    # Canonical phase tracking (December 2025)
    'check_and_apply_forced_elimination_batch',
    'mark_real_action_batch',
    'reset_capture_chain_batch',
    'reset_turn_tracking_batch',
]
