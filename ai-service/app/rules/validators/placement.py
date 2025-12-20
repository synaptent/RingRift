from app.models import GamePhase, GameState, Move, MoveType
from app.rules.core import BOARD_CONFIGS
from app.rules.interfaces import Validator
from app.rules.placement import (
    PlacementContextPy,
    evaluate_skip_placement_eligibility_py,
    validate_placement_on_board_py,
)


class PlacementValidator(Validator):
    def validate(self, state: GameState, move: Move) -> bool:
        """Validate PLACE_RING and SKIP_PLACEMENT moves.

        Phase/turn checks remain here; board geometry, caps, and
        no-dead-placement semantics are delegated to the canonical
        helpers in app.rules.placement, mirroring the TS
        PlacementAggregate.validatePlacement / validatePlacementOnBoard
        split.
        """
        # 1. Phase check
        if state.current_phase != GamePhase.RING_PLACEMENT:
            return False

        # 1.5. Skip-placement handling
        if move.type == MoveType.SKIP_PLACEMENT:
            # Turn check
            if move.player != state.current_player:
                return False

            player = next(
                (p for p in state.players if p.player_number == move.player),
                None,
            )
            if not player:
                return False

            # Host-level tightening: skip-placement is only surfaced when
            # the player has at least one ring in hand. This mirrors the
            # backend/sandbox use of the shared skip-placement aggregate
            # plus a separate ringsInHand > 0 check.
            if player.rings_in_hand <= 0:
                return False

            result = evaluate_skip_placement_eligibility_py(state, move.player)
            return result.eligible

        # This validator only handles PLACE_RING / SKIP_PLACEMENT.
        if move.type != MoveType.PLACE_RING:
            return False

        # 2. Turn check
        if move.player != state.current_player:
            return False

        player = next(
            (p for p in state.players if p.player_number == move.player),
            None,
        )
        if not player:
            return False

        if move.to is None:
            return False

        count = move.placement_count or 1

        board_type = state.board.type
        board_config = BOARD_CONFIGS[board_type]

        ctx = PlacementContextPy(
            board_type=board_type,
            player=move.player,
            rings_in_hand=player.rings_in_hand,
            rings_per_player_cap=board_config.rings_per_player,
        )

        result = validate_placement_on_board_py(
            state.board,
            move.to,
            count,
            ctx,
        )

        return result.valid
