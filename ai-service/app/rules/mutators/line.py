"""Line formation mutator - applies line processing moves.

This module implements the line formation and reward logic, extracted from
the legacy GameEngine to enable GameEngine deprecation (Phase 3i consolidation).

Canonical Spec References:
- RR-CANON-R120: Line formation conditions
- RR-CANON-R121: Marker collapse during line processing
- RR-CANON-R122: Ring return during collapse
- RR-CANON-R123: Line elimination reward (Option 1)

Architecture Note (2025-12):
    This mutator is now SELF-CONTAINED and does not delegate to GameEngine.
    It mirrors the TS LineMutator and lineProcessing semantics exactly.
"""

from app.board_manager import BoardManager
from app.core.zobrist import ZobristHash
from app.models import GameState, Move, MoveType, Position
from app.rules.core import get_effective_line_length
from app.rules.interfaces import Mutator


class LineMutator(Mutator):
    """Mutator for line formation and reward moves.

    Handles:
    - PROCESS_LINE: Full line collapse
    - CHOOSE_LINE_OPTION: Player choice (collapse all or minimum)
    - CHOOSE_LINE_REWARD: Legacy alias for CHOOSE_LINE_OPTION
    - LINE_FORMATION: Legacy move type
    """

    def apply(self, state: GameState, move: Move) -> None:
        """Apply line formation / reward moves.

        This mirrors the TS lineProcessing + LineMutator semantics used by the
        shared engine and fixtures:

        - Locate the line associated with this move using, in order of
          preference, `move.formed_lines`, `board.formed_lines`, or a fresh
          call to BoardManager.find_all_lines.
        - Determine whether we are applying the "collapse all" or
          "minimum collapse" choice.
        - Convert the chosen marker positions into collapsed territory owned
          by the moving player.
        - Increment the moving player's territory_spaces by the number of
          newly collapsed spaces.

        Elimination rewards are *not* applied here; they are modelled as
        explicit ELIMINATE_RINGS_FROM_STACK territory moves in parity
        fixtures.
        """
        board = state.board

        # 1. Locate the target line.
        target_line = None

        # Preferred source: line information carried on the Move itself.
        if getattr(move, "formed_lines", None):
            lines = list(move.formed_lines or [])
            if lines:
                target_line = lines[0]

        # Fallback: any pre-populated board.formed_lines entry whose first
        # position matches the move target.
        if target_line is None and getattr(board, "formed_lines", None):
            for line in board.formed_lines:
                if line.positions and line.positions[0].to_key() == move.to.to_key():
                    target_line = line
                    break

        # Final fallback: recompute from markers.
        if target_line is None:
            num_players = len(state.players)
            for line in BoardManager.find_all_lines(board, num_players):
                if line.positions and line.positions[0].to_key() == move.to.to_key():
                    target_line = line
                    break

        if target_line is None:
            raise ValueError(
                f"Cannot apply line formation - no target line found at {move.to.to_key()} "
                f"for player {move.player}"
            )

        # 2. Determine required minimum line length.
        num_players = len(state.players)
        required_len = get_effective_line_length(board.type, num_players)

        # 3. Decide which positions to collapse.
        positions_to_collapse: list[Position]

        if move.type in (MoveType.PROCESS_LINE, MoveType.LINE_FORMATION):
            # PROCESS_LINE / LINE_FORMATION always collapse the entire line.
            positions_to_collapse = list(target_line.positions)
        elif move.type in (MoveType.CHOOSE_LINE_OPTION, MoveType.CHOOSE_LINE_REWARD):
            # CHOOSE_LINE_OPTION is the canonical line option decision surface.
            # CHOOSE_LINE_REWARD is a legacy alias retained for replay.
            #
            # When collapsed_markers is provided, it encodes the user's choice;
            # otherwise fall back to legacy placement_count semantics:
            # 1 → minimum collapse, >1 → collapse all.
            markers_to_collapse = getattr(move, "collapsed_markers", None)
            if markers_to_collapse:
                positions_to_collapse = list(markers_to_collapse)
            else:
                option = move.placement_count or 1
                if option == 1:
                    positions_to_collapse = list(target_line.positions[:required_len])
                else:
                    positions_to_collapse = list(target_line.positions)
        else:
            # Fallback for unknown move types - collapse entire line
            positions_to_collapse = list(target_line.positions)

        # 4. Apply collapses. TS's LineAggregate increments territorySpaces
        # by the number of collapsed marker positions (collapsedKeys.size).
        # We mirror that here.
        #
        # IMPORTANT: TS's collapseLinePositions also returns rings from any
        # stacks on collapsed positions back to their owners' hands. We must
        # do the same for parity.
        seen_keys: set[str] = set()
        zobrist = ZobristHash()
        collapsed_count = 0

        for pos in positions_to_collapse:
            key = pos.to_key()
            if key in seen_keys:
                continue
            seen_keys.add(key)

            # Return rings from any stack at this position to their owners' hands.
            # This matches TS LineAggregate.collapseLinePositions behavior.
            stack = board.stacks.get(key)
            if stack and stack.rings:
                for ring in stack.rings:
                    # ring can be Ring object with .owner or just int
                    ring_owner = ring.owner if hasattr(ring, "owner") else ring
                    for player_state in state.players:
                        if player_state.player_number == ring_owner:
                            player_state.rings_in_hand += 1
                            break
                # Remove the stack
                del board.stacks[key]

            # Check if there was a marker there (should be for line formation)
            marker = board.markers.get(key)
            if marker and state.zobrist_hash is not None:
                marker_player = marker.player if hasattr(marker, "player") else marker
                state.zobrist_hash ^= zobrist.get_marker_hash(key, marker_player)

            BoardManager.set_collapsed_space(pos, move.player, board)

            if state.zobrist_hash is not None:
                state.zobrist_hash ^= zobrist.get_collapsed_hash(key)
            collapsed_count += 1

        # Update the player's territory_spaces to reflect newly collapsed spaces.
        # Territory increases when spaces collapse during line processing.
        if collapsed_count > 0:
            for player_state in state.players:
                if player_state.player_number == move.player:
                    player_state.territory_spaces += collapsed_count
                    break

        # RR-CANON-R123: Line elimination is a SEPARATE eliminate_rings_from_stack move.
        # Option 1 (collapse all) sets pending_line_reward_elimination = True.
        # Option 2 (minimum collapse) does NOT require elimination.
        # The actual elimination is applied via a follow-up eliminate_rings_from_stack move.
        is_option_1 = len(positions_to_collapse) >= len(target_line.positions)
        if is_option_1:
            # Check if player controls any stacks (has eligible elimination targets)
            player = move.player
            has_controlled_stack = any(
                stack.controlling_player == player and stack.stack_height > 0
                for stack in board.stacks.values()
            )
            if has_controlled_stack:
                # Set pending flag - requires follow-up eliminate_rings_from_stack move
                state.pending_line_reward_elimination = True

        # Update last_move_at timestamp
        state.last_move_at = move.timestamp
