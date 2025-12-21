"""Territory processing mutator - applies territory claim and elimination moves.

This module implements territory processing logic, extracted from the legacy
GameEngine to enable GameEngine deprecation (Phase 3i consolidation).

Canonical Spec References:
- RR-CANON-R145: Territory self-elimination prerequisite
- RR-CANON-R100: Forced elimination when no moves
- RR-CANON-R113/R114: Recovery buried ring extraction

Architecture Note (2025-12):
    This mutator is now SELF-CONTAINED and does not delegate to GameEngine.
    It mirrors the TS TerritoryAggregate and territoryProcessing semantics.
"""

from app.board_manager import BoardManager
from app.core.zobrist import ZobristHash
from app.models import GameState, Move, MoveType, Position
from app.rules.elimination import EliminationContext, is_stack_eligible_for_elimination
from app.rules.interfaces import Mutator


class TerritoryMutator(Mutator):
    """Mutator for territory processing and elimination moves.

    Handles:
    - CHOOSE_TERRITORY_OPTION: Player choice on disconnected region
    - TERRITORY_CLAIM: Legacy territory claim move
    - ELIMINATE_RINGS_FROM_STACK: Ring elimination (line/territory/forced/recovery)
    - FORCED_ELIMINATION: Legacy alias for elimination
    """

    def apply(self, state: GameState, move: Move) -> None:
        """Apply territory or elimination moves.

        This mirrors TS territoryProcessing + TerritoryAggregate semantics:
        - Territory claim: collapse disconnected region + border markers
        - Elimination: remove rings based on context (line/territory/forced/recovery)
        """
        if move.type == MoveType.ELIMINATE_RINGS_FROM_STACK:
            self._apply_forced_elimination(state, move)
        elif move.type == MoveType.FORCED_ELIMINATION:
            self._apply_forced_elimination(state, move)
        else:
            self._apply_territory_claim(state, move)

        state.last_move_at = move.timestamp

    def _apply_territory_claim(self, state: GameState, move: Move) -> None:
        """Apply territory claim move (TS-orchestrator-aligned).

        1) Identify the disconnected region associated with this move
        2) Verify region can be processed (self-elimination prerequisite)
        3) Eliminate all rings inside the region
        4) Collapse all spaces in region + border markers
        5) Update territory_spaces
        """
        board = state.board
        player = move.player

        # 1. Identify target region, preferring explicit decision geometry
        target_region = None
        move_key = move.to.to_key()

        # 1a. Prefer explicit disconnectedRegions geometry carried on the Move
        explicit_regions = list(getattr(move, "disconnected_regions", None) or [])
        if explicit_regions:
            for region in explicit_regions:
                if any(space.to_key() == move_key for space in region.spaces):
                    target_region = region
                    break
            if target_region is None:
                # Fall back to first region if no space matched
                target_region = explicit_regions[0]
        else:
            # 1b. Fallback: rediscover regions from the board state
            regions = BoardManager.find_disconnected_regions(board, player)
            for region in regions:
                if region.spaces and region.spaces[0].to_key() == move_key:
                    target_region = region
                    break

        if not target_region:
            raise ValueError(
                f"Cannot apply territory claim - no matching disconnected region "
                f"found for move at {move_key} by player {player}"
            )

        # 2. Check self-elimination prerequisite (RR-CANON-R145/R114)
        if not self._can_process_region(state, target_region, player):
            # TS treats non-processable regions as no-op
            return

        # 3. Gather border markers for territory elimination
        border_markers = BoardManager.get_border_marker_positions(
            target_region.spaces,
            board,
        )

        # 4. Eliminate all rings within the region
        zobrist = ZobristHash()
        for pos in target_region.spaces:
            while True:
                stack = BoardManager.get_stack(pos, board)
                if not stack or stack.stack_height == 0:
                    break
                self._eliminate_top_ring_at(state, pos, credited_player=player)

        # 5. Collapse all spaces in region to moving player's colour
        for pos in target_region.spaces:
            key = pos.to_key()
            marker = board.markers.get(key)
            if marker and state.zobrist_hash is not None:
                state.zobrist_hash ^= zobrist.get_marker_hash(key, marker.player)
            BoardManager.set_collapsed_space(pos, player, board)
            if state.zobrist_hash is not None:
                state.zobrist_hash ^= zobrist.get_collapsed_hash(key)

        # 6. Collapse all border markers to moving player's colour
        for pos in border_markers:
            key = pos.to_key()
            marker = board.markers.get(key)
            if marker and state.zobrist_hash is not None:
                state.zobrist_hash ^= zobrist.get_marker_hash(key, marker.player)
            BoardManager.set_collapsed_space(pos, player, board)
            if state.zobrist_hash is not None:
                state.zobrist_hash ^= zobrist.get_collapsed_hash(key)

        # 7. Update territory_spaces for the moving player
        spaces_gained = len(target_region.spaces) + len(border_markers)
        if spaces_gained > 0:
            for ps in state.players:
                if ps.player_number == player:
                    ps.territory_spaces += spaces_gained
                    break

    def _apply_forced_elimination(self, state: GameState, move: Move) -> None:
        """Apply forced elimination move.

        Context-dependent elimination (RR-CANON-R022, R122, R145, R100, R113):
        - 'line': Eliminate exactly ONE ring (RR-CANON-R122)
        - 'territory'/'forced'/None: Eliminate entire cap (RR-CANON-R145, R100)
        - 'recovery': Extract exactly ONE buried ring (RR-CANON-R113/R114)
        """
        board = state.board
        pos = move.to
        stack = board.stacks.get(pos.to_key())
        if not stack:
            raise ValueError(
                f"Cannot apply forced elimination - no stack at {pos.to_key()} "
                f"for player {move.player}"
            )

        # Determine rings to eliminate based on context
        elimination_context = getattr(move, "elimination_context", None)

        if elimination_context == "line":
            # RR-CANON-R122: Line elimination = exactly 1 ring
            rings_to_eliminate = 1
        elif elimination_context == "recovery":
            # RR-CANON-R113/R114: Extract buried ring
            self._extract_buried_ring_at(state, pos, credited_player=move.player)
            return
        else:
            # RR-CANON-R145/R100: Territory/forced = entire cap
            # Match TS: max(1, cap_height) for legacy states
            rings_to_eliminate = max(1, stack.cap_height or 0)

        for _ in range(rings_to_eliminate):
            self._eliminate_top_ring_at(state, pos, credited_player=move.player)

    def _can_process_region(
        self,
        state: GameState,
        region,
        player_number: int,
    ) -> bool:
        """Self-elimination prerequisite for territory processing.

        Mirrors TS canProcessTerritoryRegion:
        - Normal context (RR-CANON-R145): Player must have eligible cap target outside region
        - Recovery context (RR-CANON-R114): Player must have eligible buried ring target
        """
        board = state.board
        region_keys = {p.to_key() for p in region.spaces}

        if self._did_turn_include_recovery_slide(state, player_number):
            # RR-CANON-R114: Recovery context - check for buried ring targets
            for stack in board.stacks.values():
                if stack.position.to_key() in region_keys:
                    continue
                eligibility = is_stack_eligible_for_elimination(
                    rings=stack.rings,
                    controlling_player=stack.controlling_player,
                    context=EliminationContext.RECOVERY,
                    player=player_number,
                )
                if eligibility.eligible:
                    return True
            return False

        # Normal territory context - check for controlled stacks outside region
        player_stacks = BoardManager.get_player_stacks(board, player_number)
        stacks_outside = [
            stack
            for stack in player_stacks
            if stack.position.to_key() not in region_keys
        ]

        if len(stacks_outside) == 0:
            return False

        # RR-CANON-R145: Any controlled stack (including height-1) is eligible
        for stack in stacks_outside:
            eligibility = is_stack_eligible_for_elimination(
                rings=stack.rings,
                controlling_player=stack.controlling_player,
                context=EliminationContext.TERRITORY,
                player=player_number,
            )
            if eligibility.eligible:
                return True

        return False

    def _did_turn_include_recovery_slide(
        self,
        state: GameState,
        player_number: int,
    ) -> bool:
        """Check if current turn includes a RECOVERY_SLIDE move.

        Used to determine whether territory processing should use
        recovery self-elimination context (RR-CANON-R114).
        """
        for move in reversed(state.move_history):
            if move.player != player_number:
                break
            if move.type == MoveType.RECOVERY_SLIDE:
                return True
        return False

    def _eliminate_top_ring_at(
        self,
        state: GameState,
        position: Position,
        credited_player: int,
    ) -> None:
        """Eliminate exactly the top ring from stack at position.

        Updates board.eliminated_rings, total_rings_eliminated, and
        per-player eliminated_rings to match TS semantics.
        """
        board = state.board
        pos_key = position.to_key()
        stack = board.stacks.get(pos_key)
        if not stack or stack.stack_height == 0:
            return

        zobrist = ZobristHash()
        # Remove old stack hash
        if state.zobrist_hash is not None:
            state.zobrist_hash ^= zobrist.get_stack_hash(
                pos_key, stack.controlling_player, stack.stack_height, tuple(stack.rings)
            )

        stack = stack.model_copy(deep=True)
        board.stacks[pos_key] = stack

        stack.rings.pop()
        stack.stack_height -= 1

        # Update elimination counters
        player_id_str = str(credited_player)
        board.eliminated_rings[player_id_str] = (
            board.eliminated_rings.get(player_id_str, 0) + 1
        )
        state.total_rings_eliminated += 1

        for p in state.players:
            if p.player_number == credited_player:
                p.eliminated_rings += 1
                break

        if stack.stack_height == 0 or not stack.rings:
            # Stack is empty - remove it
            if pos_key in board.stacks:
                del board.stacks[pos_key]
        else:
            # Update controlling player and cap height
            stack.controlling_player = stack.rings[-1]
            h = 0
            for r in reversed(stack.rings):
                if r == stack.controlling_player:
                    h += 1
                else:
                    break
            stack.cap_height = h
            BoardManager.set_stack(position, stack, board)
            # Add new stack hash
            if state.zobrist_hash is not None:
                state.zobrist_hash ^= zobrist.get_stack_hash(
                    pos_key,
                    stack.controlling_player,
                    stack.stack_height,
                    tuple(stack.rings),
                )

    def _extract_buried_ring_at(
        self,
        state: GameState,
        position: Position,
        credited_player: int,
    ) -> None:
        """Extract (eliminate) exactly one buried ring from stack.

        Canonical recovery semantics (RR-CANON-R113/R114):
        - Extracted ring must belong to credited_player
        - Must be buried (not the top ring)
        - When multiple exist, extract the bottommost one
        """
        board = state.board
        pos_key = position.to_key()
        stack = board.stacks.get(pos_key)
        if not stack or stack.stack_height <= 1:
            raise ValueError(f"Cannot extract buried ring - no eligible stack at {pos_key}")

        # Find bottommost buried ring of credited_player (exclude top ring)
        extract_index = None
        for i, ring in enumerate(stack.rings[:-1]):
            if ring == credited_player:
                extract_index = i
                break

        if extract_index is None:
            raise ValueError(
                f"Cannot extract buried ring - stack at {pos_key} has no buried ring "
                f"of player {credited_player}"
            )

        zobrist = ZobristHash()
        # Remove old stack hash
        if state.zobrist_hash is not None:
            state.zobrist_hash ^= zobrist.get_stack_hash(
                pos_key, stack.controlling_player, stack.stack_height, tuple(stack.rings)
            )

        stack = stack.model_copy(deep=True)
        board.stacks[pos_key] = stack

        stack.rings.pop(extract_index)
        stack.stack_height -= 1

        # Update elimination counters
        player_id_str = str(credited_player)
        board.eliminated_rings[player_id_str] = (
            board.eliminated_rings.get(player_id_str, 0) + 1
        )
        state.total_rings_eliminated += 1

        for p in state.players:
            if p.player_number == credited_player:
                p.eliminated_rings += 1
                break

        # Recalculate controlling player and cap height
        stack.controlling_player = stack.rings[-1]
        h = 0
        for r in reversed(stack.rings):
            if r == stack.controlling_player:
                h += 1
            else:
                break
        stack.cap_height = h
        BoardManager.set_stack(position, stack, board)

        # Add new stack hash
        if state.zobrist_hash is not None:
            state.zobrist_hash ^= zobrist.get_stack_hash(
                pos_key,
                stack.controlling_player,
                stack.stack_height,
                tuple(stack.rings),
            )
