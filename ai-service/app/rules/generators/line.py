"""Line processing move generator.

This module implements line processing move enumeration, extracted from
GameEngine._get_line_processing_moves to establish SSoT.

Canonical Spec References:
- RR-CANON-R076: Interactive decision moves only
- RR-CANON-R120: Line formation conditions
- RR-CANON-R123: Line elimination reward (pending_line_reward_elimination)

Architecture Note (2025-12):
    This generator uses BoardManager.find_all_lines (SSoT) for line detection
    and creates Move objects for:
    - PROCESS_LINE: Decision to process a specific line
    - CHOOSE_LINE_OPTION: Choice between collapse-all and minimum-collapse
    - ELIMINATE_RINGS_FROM_STACK: Follow-up elimination when pending
"""

from app.board_manager import BoardManager
from app.models import GameState, Move, MoveType
from app.rules.core import get_effective_line_length
from app.rules.interfaces import Generator


class LineGenerator(Generator):
    """Generator for line processing moves.

    Enumerates canonical line-processing **decision** moves for a player.

    Per RR-CANON-R076, this returns **only interactive moves**:
    - One PROCESS_LINE move per player-owned line
    - CHOOSE_LINE_OPTION moves encoding collapse-all and minimum-collapse

    It does NOT fabricate NO_LINE_ACTION bookkeeping moves. When the
    player has no lines to process, generate() returns an empty list
    and get_phase_requirement surfaces NO_LINE_ACTION_REQUIRED.
    """

    def generate(self, state: GameState, player: int) -> list[Move]:
        """Generate all legal line processing moves for the player.

        Args:
            state: Current game state
            player: Player number to generate moves for

        Returns:
            List of legal Move objects for line processing
        """
        # RR-CANON-R123: If pending_line_reward_elimination, enumerate elimination targets
        if state.pending_line_reward_elimination:
            return self._enumerate_elimination_moves(state, player)

        return self._enumerate_line_decisions(state, player)

    def _enumerate_elimination_moves(
        self,
        state: GameState,
        player: int,
    ) -> list[Move]:
        """Enumerate eliminate_rings_from_stack moves for line processing.

        Per RR-CANON-R123: When pending_line_reward_elimination is True,
        the player must choose which stack to eliminate one ring from.
        All controlled stacks are eligible targets.

        Returns one ELIMINATE_RINGS_FROM_STACK move per eligible stack.
        """
        board = state.board
        moves: list[Move] = []

        for stack_key, stack in board.stacks.items():
            if stack.controlling_player == player and stack.stack_height > 0:
                pos = stack.position
                moves.append(
                    Move(
                        id=f"eliminate-line-{stack_key}",
                        type=MoveType.ELIMINATE_RINGS_FROM_STACK,
                        player=player,
                        to=pos,
                        eliminated_rings=({"player": player, "count": 1},),
                        elimination_context="line",
                        timestamp=state.last_move_at,
                        thinkTime=0,
                        moveNumber=len(state.move_history),
                    )
                )

        return moves

    def _enumerate_line_decisions(
        self,
        state: GameState,
        player: int,
    ) -> list[Move]:
        """Enumerate line decision moves (PROCESS_LINE and CHOOSE_LINE_OPTION).

        This mirrors TS enumerateChooseLineRewardMoves semantics:
        - If line_len < required_len: no choose-line-option moves
        - If line_len == required_len: single collapse-all move
        - If line_len > required_len: collapse-all + minimum-collapse variants
        """
        num_players = len(state.players)
        lines = BoardManager.find_all_lines(state.board, num_players)
        player_lines = [line for line in lines if line.player == player]

        if not player_lines:
            # No interactive line decisions – hosts must satisfy the phase via
            # a NO_LINE_ACTION_REQUIRED phase requirement.
            return []

        required_len = get_effective_line_length(state.board.type, num_players)
        moves: list[Move] = []
        move_number = len(state.move_history) + 1

        for idx, line in enumerate(player_lines):
            first_pos = line.positions[0]
            line_key = first_pos.to_key()

            # PROCESS_LINE decision for this line
            moves.append(
                Move(
                    id=f"process-line-{idx}-{line_key}",
                    type=MoveType.PROCESS_LINE,
                    player=player,
                    to=first_pos,
                    formed_lines=(line,),
                    timestamp=state.last_move_at,
                    thinkTime=0,
                    moveNumber=move_number,
                )
            )

            # CHOOSE_LINE_OPTION moves for lines that meet threshold
            line_len = len(line.positions)
            if line_len < required_len:
                continue

            # Exact-length line: single collapse-all variant
            if line_len == required_len:
                moves.append(
                    Move(
                        id=f"choose-line-option-{idx}-{line_key}-all",
                        type=MoveType.CHOOSE_LINE_OPTION,
                        player=player,
                        to=first_pos,
                        formed_lines=(line,),
                        collapsed_markers=tuple(line.positions),
                        timestamp=state.last_move_at,
                        thinkTime=0,
                        moveNumber=move_number,
                    )
                )
                continue

            # Overlength line (> required_len)
            # Option 1: collapse all markers
            moves.append(
                Move(
                    id=f"choose-line-option-{idx}-{line_key}-all",
                    type=MoveType.CHOOSE_LINE_OPTION,
                    player=player,
                    to=first_pos,
                    formed_lines=(line,),
                    collapsed_markers=tuple(line.positions),
                    timestamp=state.last_move_at,
                    thinkTime=0,
                    moveNumber=move_number,
                )
            )

            # Option 2: enumerate all valid minimum-collapse segments
            max_start = line_len - required_len
            for start in range(max_start + 1):
                segment = tuple(line.positions[start : start + required_len])
                moves.append(
                    Move(
                        id=f"choose-line-option-{idx}-{line_key}-min-{start}",
                        type=MoveType.CHOOSE_LINE_OPTION,
                        player=player,
                        to=first_pos,
                        formed_lines=(line,),
                        collapsed_markers=segment,
                        timestamp=state.last_move_at,
                        thinkTime=0,
                        moveNumber=move_number,
                    )
                )

        return moves
