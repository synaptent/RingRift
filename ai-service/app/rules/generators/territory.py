"""Territory processing move generator.

This module implements territory processing move enumeration, extracted from
GameEngine._get_territory_processing_moves to establish SSoT.

Canonical Spec References:
- RR-CANON-R076: Interactive decision moves only
- RR-CANON-R114: Recovery context territory self-elimination
- RR-CANON-R143: Self-elimination prerequisite (any player can process any region)
- RR-CANON-R145: Normal territory self-elimination

Architecture Note (2025-12):
    This generator uses BoardManager.find_disconnected_regions (SSoT) for
    territory detection and creates Move objects for:
    - CHOOSE_TERRITORY_OPTION: Decision to process a specific region
    - SKIP_TERRITORY_PROCESSING: Voluntary skip when regions exist
    - ELIMINATE_RINGS_FROM_STACK: Follow-up elimination after region processing
"""

from __future__ import annotations  # Enable Python 3.10+ type hints on 3.9

from app.board_manager import BoardManager
from app.models import GameState, Move, MoveType, Position, Territory
from app.rules.elimination import EliminationContext, is_stack_eligible_for_elimination
from app.rules.interfaces import Generator


class TerritoryGenerator(Generator):
    """Generator for territory processing moves.

    Enumerates canonical territory-processing **decision** moves.

    Mirrors TS getValidTerritoryProcessingDecisionMoves and
    getValidEliminationDecisionMoves while obeying R076:

    - When a territory region was just processed this turn, emit the
      mandatory ELIMINATE_RINGS_FROM_STACK follow-up moves first
    - When eligible regions exist, emit CHOOSE_TERRITORY_OPTION moves
    - Include SKIP_TERRITORY_PROCESSING when regions are available
    - Return empty list when no eligible regions (requires NO_TERRITORY_ACTION)
    """

    def generate(self, state: GameState, player: int) -> list[Move]:
        """Generate all legal territory processing moves for the player.

        Args:
            state: Current game state
            player: Player number to generate moves for

        Returns:
            List of legal Move objects for territory processing
        """
        # Check if we need to enumerate elimination moves first
        elimination_moves = self._check_pending_elimination(state, player)
        if elimination_moves is not None:
            return elimination_moves

        return self._enumerate_territory_decisions(state, player)

    def _check_pending_elimination(
        self,
        state: GameState,
        player: int,
    ) -> list[Move] | None:
        """Check if mandatory elimination is pending after region processing.

        Returns:
            List of elimination moves if pending, None otherwise
        """
        last_move = state.move_history[-1] if state.move_history else None
        if last_move is None:
            return None

        if last_move.player != player:
            return None

        if last_move.type not in (
            MoveType.CHOOSE_TERRITORY_OPTION,
            MoveType.PROCESS_TERRITORY_REGION,  # legacy alias
        ):
            return None

        # Check that the move contains region data - matches TS behavior which
        # returns null from getPendingTerritorySelfEliminationRegion when empty
        processed_regions = getattr(last_move, "disconnected_regions", None) or ()
        if not processed_regions:
            return None

        # Elimination is pending after a territory region was processed
        move_number = len(state.move_history) + 1
        elimination_context = (
            "recovery"
            if self._did_current_turn_include_recovery_slide(state, player)
            else "territory"
        )

        # Get processed region keys to exclude from elimination targets
        processed_region_keys = {p.to_key() for p in processed_regions[0].spaces}

        board = state.board
        elimination_moves: list[Move] = []

        for pos_key, stack in board.stacks.items():
            if pos_key in processed_region_keys:
                continue

            if elimination_context == "recovery":
                # RR-CANON-R114: recovery-context territory self-elimination
                # is a buried-ring extraction. Stack need not be controlled.
                if stack.stack_height <= 1:
                    continue
                if player not in (stack.rings[:-1] or []):
                    continue
            else:
                # RR-CANON-R145: normal territory self-elimination requires
                # a controlled stack outside the region.
                if stack.controlling_player != player:
                    continue

            elimination_moves.append(
                Move(
                    id=f"eliminate-rings-from-stack-{move_number}-{pos_key}",
                    type=MoveType.ELIMINATE_RINGS_FROM_STACK,
                    player=player,
                    to=stack.position,
                    elimination_context=elimination_context,
                    timestamp=state.last_move_at,
                    thinkTime=0,
                    moveNumber=move_number,
                )
            )

        return elimination_moves

    def _enumerate_territory_decisions(
        self,
        state: GameState,
        player: int,
    ) -> list[Move]:
        """Enumerate territory decision moves (CHOOSE_TERRITORY_OPTION, SKIP).

        Per RR-CANON-R143: Any player can process ANY disconnected region
        as long as they satisfy the self-elimination prerequisite.
        """
        board = state.board
        regions = BoardManager.find_disconnected_regions(board, player)

        # Defensive filter: exclude non-marker-bordered regions with no stacks inside.
        # Per TS territoryDetection.ts lines 160-167, non-marker-bordered regions
        # require exactly 1 player represented. When there are no markers on the
        # board, ALL detected regions must have stacks inside. This guards against
        # detection bugs that might return empty regions.
        def _has_stacks_inside(region: Territory) -> bool:
            for space in region.spaces:
                if BoardManager.get_stack(space, board) is not None:
                    return True
            return False

        # Only apply the empty-region filter when no markers exist (all regions
        # must be non-marker-bordered in that case). When markers exist, some
        # empty regions may be valid (marker-bordered with controllingPlayer
        # assigned to the border color).
        has_markers = len(board.markers) > 0
        if has_markers:
            filtered_regions = regions or []
        else:
            filtered_regions = [r for r in (regions or []) if _has_stacks_inside(r)]

        # Filter to eligible regions (self-elimination prerequisite)
        eligible_regions = [
            region
            for region in filtered_regions
            if self._can_process_region(state, region, player)
        ]

        moves: list[Move] = []

        if eligible_regions:
            move_number = len(state.move_history) + 1

            for idx, region in enumerate(eligible_regions):
                rep = region.spaces[0]
                # Canonical guard: controlling_player should not be 0
                controlling_player = (
                    region.controlling_player
                    if region.controlling_player != 0
                    else player
                )
                # Mutate the region for downstream parity
                region.controlling_player = controlling_player

                safe_region = Territory(
                    spaces=region.spaces,
                    controlling_player=controlling_player,
                    is_disconnected=region.is_disconnected,
                )
                moves.append(
                    Move(
                        id=f"choose-territory-option-{idx}-{rep.to_key()}",
                        type=MoveType.CHOOSE_TERRITORY_OPTION,
                        player=player,
                        to=rep,
                        disconnected_regions=(safe_region,),
                        timestamp=state.last_move_at,
                        thinkTime=0,
                        moveNumber=move_number,
                    )
                )

            # RR-CANON-R075: Allow voluntary skip when regions exist
            moves.append(
                Move(
                    id=f"skip-territory-processing-{move_number}",
                    type=MoveType.SKIP_TERRITORY_PROCESSING,
                    player=player,
                    to=Position(x=0, y=0),
                    timestamp=state.last_move_at,
                    thinkTime=0,
                    moveNumber=move_number,
                )
            )

        # Empty list when no eligible regions
        return moves

    def _can_process_region(
        self,
        state: GameState,
        region: Territory,
        player: int,
    ) -> bool:
        """Check self-elimination prerequisite for territory processing.

        Mirrors TS canProcessTerritoryRegion:
        - Normal context (RR-CANON-R145): Need eligible cap target outside region
        - Recovery context (RR-CANON-R114): Need buried-ring extraction target
        """
        board = state.board
        region_keys = {p.to_key() for p in region.spaces}

        if self._did_current_turn_include_recovery_slide(state, player):
            # Recovery context: look for buried rings outside region
            for stack in board.stacks.values():
                if stack.position.to_key() in region_keys:
                    continue
                eligibility = is_stack_eligible_for_elimination(
                    rings=stack.rings,
                    controlling_player=stack.controlling_player,
                    context=EliminationContext.RECOVERY,
                    player=player,
                )
                if eligibility.eligible:
                    return True
            return False

        # Normal context: look for controlled stacks outside region
        player_stacks = BoardManager.get_player_stacks(board, player)
        stacks_outside = [
            stack for stack in player_stacks
            if stack.position.to_key() not in region_keys
        ]

        if not stacks_outside:
            return False

        for stack in stacks_outside:
            eligibility = is_stack_eligible_for_elimination(
                rings=stack.rings,
                controlling_player=stack.controlling_player,
                context=EliminationContext.TERRITORY,
                player=player,
            )
            if eligibility.eligible:
                return True

        return False

    def _did_current_turn_include_recovery_slide(
        self,
        state: GameState,
        player: int,
    ) -> bool:
        """Check if current turn includes a RECOVERY_SLIDE move.

        Used to determine whether territory processing should use the
        recovery self-elimination context (RR-CANON-R114).
        """
        for move in reversed(state.move_history):
            if move.player != player:
                break
            if move.type == MoveType.RECOVERY_SLIDE:
                return True
        return False
