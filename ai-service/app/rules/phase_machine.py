from __future__ import annotations

"""
Phase state machine for the Python GameEngine.

This module centralises phase and turn transitions, mirroring the
TypeScript phaseStateMachine + TurnOrchestrator behaviour:

- It is responsible for updating ``current_phase``, ``current_player``,
  and related per-turn bookkeeping after a move has been applied.
- It does **not** mutate board geometry or player inventories; those
  are handled by the move-application layer in ``GameEngine.apply_move``.
- It never fabricates Move objects; bookkeeping moves such as
  ``NO_*_ACTION`` and ``FORCED_ELIMINATION`` are surfaced via
  phase requirements and constructed by hosts.

All logic here operates directly on a mutable GameState instance.
"""

from dataclasses import dataclass

from app.models import GameState, GamePhase, GameStatus, Move, MoveType


@dataclass
class PhaseTransitionInput:
  """Input to the phase state machine."""

  game_state: GameState
  last_move: Move
  trace_mode: bool = False


def _is_no_action_bookkeeping_move(move_type: MoveType) -> bool:
  """Return True for forced no-op bookkeeping moves that do NOT count as actions.

  Mirrors the TS turnOrchestrator.isNoActionBookkeepingMove helper.
  """
  return move_type in {
    MoveType.NO_PLACEMENT_ACTION,
    MoveType.NO_MOVEMENT_ACTION,
    MoveType.NO_LINE_ACTION,
    MoveType.NO_TERRITORY_ACTION,
  }


def compute_had_any_action_this_turn(game_state: GameState) -> bool:
  """
  Python analogue of the TS computeHadAnyActionThisTurn helper.

  Walk backwards through move_history for the current_player until the player
  changes. Any move that is not a forced no-op bookkeeping move counts as an
  action. Voluntary skips (e.g. SKIP_TERRITORY_PROCESSING) and
  FORCED_ELIMINATION both count as actions; the various NO_*_ACTION moves do
  not.
  """
  current_player = game_state.current_player
  history = game_state.move_history

  for move in reversed(history):
    if move.player != current_player:
      break
    if not _is_no_action_bookkeeping_move(move.type):
      return True
  return False


def player_has_stacks_on_board(game_state: GameState, player: int) -> bool:
  """
  True if ``player`` currently controls at least one stack on the board.

  A stack is counted when stack_height > 0 and controlling_player == player.
  Mirrors the TS playerHasStacksOnBoard helper used for FE gating.
  """
  for stack in game_state.board.stacks.values():
    if stack.controlling_player == player and stack.stack_height > 0:
      return True
  return False


def _on_line_processing_complete(game_state: GameState, *, trace_mode: bool) -> None:
  """
  Shared helper for exiting LINE_PROCESSING.

  Decides whether to advance to TERRITORY_PROCESSING, enter FORCED_ELIMINATION,
  or end the turn (rotate to next player) using the same boolean contract as
  TurnStateMachine.onLineProcessingComplete on the TS side.
  """
  from app.game_engine import GameEngine

  current_player = game_state.current_player

  # Territory availability is defined by the presence of at least one
  # process_territory_region decision for the active player.
  territory_moves = GameEngine._get_territory_processing_moves(
    game_state,
    current_player,
  )
  has_territory_regions = bool(territory_moves)

  had_any_action = compute_had_any_action_this_turn(game_state)
  has_stacks = player_has_stacks_on_board(game_state, current_player)

  if has_territory_regions:
    GameEngine._advance_to_territory_processing(
      game_state,
      trace_mode=trace_mode,
    )
    return

  # No territory regions – decide between forced_elimination and normal turn end.
  if not had_any_action and has_stacks:
    game_state.current_phase = GamePhase.FORCED_ELIMINATION
    return

  # Normal turn end: rotate to the next player using the canonical _end_turn
  # helper so that FE gating and ANM invariants are applied consistently.
  GameEngine._end_turn(game_state, trace_mode=trace_mode)


def _on_territory_processing_complete(
  game_state: GameState, *, trace_mode: bool
) -> None:
  """
  Shared helper for exiting TERRITORY_PROCESSING.

  Mirrors TurnStateMachine.onTerritoryProcessingComplete + orchestrator
  behaviour:

  - If the active player had **no actions at all this turn** and still controls
    stacks, enter FORCED_ELIMINATION.
  - Otherwise, perform a full turn end via GameEngine._end_turn so that the
    next player's FE/ANM gating is applied consistently.
  """
  from app.game_engine import GameEngine

  current_player = game_state.current_player
  had_any_action = compute_had_any_action_this_turn(game_state)
  has_stacks = player_has_stacks_on_board(game_state, current_player)

  if not had_any_action and has_stacks:
    game_state.current_phase = GamePhase.FORCED_ELIMINATION
    return

  GameEngine._end_turn(game_state, trace_mode=trace_mode)


def advance_phases(inp: PhaseTransitionInput) -> None:
  """Advance phases and turn rotation for ``inp.game_state``.

  This mirrors the semantics of GameEngine._update_phase plus the helper
  transitions for line/territory/turn-end, but is factored into a
  dedicated module for clarity and easier testing.

  The function mutates ``inp.game_state`` in-place and does not return a
  value. Hosts should call victory and invariant checks separately after
  phase advancement.
  """
  # Local import to avoid circular imports at module load time.
  from app.game_engine import GameEngine

  game_state = inp.game_state
  last_move = inp.last_move
  trace_mode = inp.trace_mode

  current_player = game_state.current_player

  if last_move.type == MoveType.FORCED_ELIMINATION:
    # Per RR-CANON-R070 and the shared TS engine, explicit
    # FORCED_ELIMINATION is the seventh and final phase of a turn.
    # After applying the elimination move, the current player has no
    # further interactive actions this turn; we rotate directly to
    # the next active player (skipping players with no material) and
    # start their turn in ring_placement or movement.
    #
    # Victory detection is handled by _check_victory after this
    # phase update, mirroring the TS TurnOrchestrator flow where
    # forced_elimination is followed by a victory check and then
    # turn rotation.
    game_state.current_phase = GamePhase.FORCED_ELIMINATION
    GameEngine._rotate_to_next_active_player(game_state)

  elif last_move.type == MoveType.PLACE_RING:
    # After placement, decide whether to enter movement or, if no
    # movement/capture is available, advance directly to line
    # processing. Mirrors TurnEngine.advanceGameForCurrentPlayer.
    has_moves = GameEngine._has_valid_movements(
      game_state,
      current_player,
    )
    has_captures = GameEngine._has_valid_captures(
      game_state,
      current_player,
    )
    if has_moves or has_captures:
      game_state.current_phase = GamePhase.MOVEMENT
    else:
      GameEngine._advance_to_line_processing(
        game_state, trace_mode=trace_mode
      )

  elif last_move.type == MoveType.SKIP_PLACEMENT:
    # Skipping placement is only allowed when movement or capture is
    # already available. After a legal skip, always enter movement.
    game_state.current_phase = GamePhase.MOVEMENT

  elif last_move.type == MoveType.NO_PLACEMENT_ACTION:
    # Explicit no-op placement: always advance to MOVEMENT so that the
    # rest of the turn (movement/capture/line/territory/FE) can run.
    game_state.current_phase = GamePhase.MOVEMENT

  elif last_move.type == MoveType.MOVE_STACK:
    # After movement, check for captures from the landing position.
    # Per RR-CANON-R093, post-movement captures are only available
    # from the stack that just moved, at its landing position.
    # Clear any stale chain_capture_state from previous turns to
    # ensure the first capture is OVERTAKING_CAPTURE (initial), not
    # CONTINUE_CAPTURE_SEGMENT (continuation).
    game_state.chain_capture_state = None
    capture_moves = GameEngine._get_capture_moves(
      game_state, current_player
    )
    if capture_moves:
      game_state.current_phase = GamePhase.CAPTURE
    else:
      # No captures, go to line processing
      GameEngine._advance_to_line_processing(
        game_state, trace_mode=trace_mode
      )

  elif last_move.type == MoveType.NO_MOVEMENT_ACTION:
    # Movement phase was visited but no legal movement or capture
    # existed anywhere for the player. Per RR-CANON-R075, this is
    # recorded as an explicit NO_MOVEMENT_ACTION and we advance
    # directly to line_processing.
    GameEngine._advance_to_line_processing(
      game_state, trace_mode=trace_mode
    )

  elif last_move.type == MoveType.RECOVERY_SLIDE:
    # Recovery slide is a movement-phase real action. After applying it,
    # we proceed to line_processing to record phase traversal (even if
    # no further line decisions remain because collapse was applied).
    game_state.chain_capture_state = None
    GameEngine._advance_to_line_processing(
      game_state, trace_mode=trace_mode
    )

  elif last_move.type in (
    MoveType.OVERTAKING_CAPTURE,
    MoveType.CHAIN_CAPTURE,
    MoveType.CONTINUE_CAPTURE_SEGMENT,
  ):
    # Check for more captures (chain)
    capture_moves = GameEngine._get_capture_moves(
      game_state,
      current_player,
    )
    if capture_moves:
      # After the first capture, subsequent captures are chain captures
      game_state.current_phase = GamePhase.CHAIN_CAPTURE
    else:
      # End of chain
      GameEngine._advance_to_line_processing(
        game_state, trace_mode=trace_mode
      )

  elif last_move.type in (
    MoveType.PROCESS_LINE,
    MoveType.CHOOSE_LINE_REWARD,
    MoveType.LINE_FORMATION,
    MoveType.CHOOSE_LINE_OPTION,
  ):
    # After processing a line decision, check if there are more interactive
    # line-processing moves for the current player. If not, delegate to the
    # canonical post-line helper which decides between TERRITORY_PROCESSING,
    # FORCED_ELIMINATION, and turn-end based on hadAnyActionThisTurn and the
    # player's remaining material, mirroring the TS
    # TurnStateMachine.onLineProcessingComplete semantics.
    remaining_lines = [
      m for m in GameEngine._get_line_processing_moves(
        game_state,
        current_player,
      )
      if m.type != MoveType.NO_LINE_ACTION
    ]
    if remaining_lines:
      # Stay in line_processing; hosts will surface the next PROCESS_LINE or
      # CHOOSE_LINE_REWARD move.
      game_state.current_phase = GamePhase.LINE_PROCESSING
    else:
      _on_line_processing_complete(game_state, trace_mode=trace_mode)

  elif last_move.type == MoveType.NO_LINE_ACTION:
    # Forced no-op: player entered line_processing but had no lines to process.
    # Per RR-CANON-R075, this move marks that the phase was visited. After the
    # final line-phase move (including NO_LINE_ACTION), delegate to the shared
    # post-line helper so that we either:
    # - enter TERRITORY_PROCESSING when regions exist;
    # - enter FORCED_ELIMINATION when the player had no actions this turn but
    #   still controls stacks; or
    # - end the turn and rotate to the next player.
    _on_line_processing_complete(game_state, trace_mode=trace_mode)
    # If we landed in TERRITORY_PROCESSING with no regions, surface the
    # requirement so hosts record NO_TERRITORY_ACTION (prevents silent
    # territory skip leaving the next placement in the wrong phase).
    remaining_regions = GameEngine._get_territory_processing_moves(
      game_state,
      current_player,
    )
    if (
      game_state.current_phase == GamePhase.TERRITORY_PROCESSING
      and not remaining_regions
    ):
      game_state.current_phase = GamePhase.TERRITORY_PROCESSING  # explicit
    # If line processing had no interactive decisions and territory also has
    # no regions, make sure hosts see the no-territory requirement so they
    # record NO_TERRITORY_ACTION rather than leaving the recorder mid-turn.
    remaining_regions = GameEngine._get_territory_processing_moves(
      game_state,
      current_player,
    )
    if not remaining_regions and game_state.current_phase == GamePhase.TERRITORY_PROCESSING:
      # Explicitly set the phase; hosts will synthesize NO_TERRITORY_ACTION
      # via get_phase_requirement → synthesize_bookkeeping_move.
      game_state.current_phase = GamePhase.TERRITORY_PROCESSING

  elif last_move.type == MoveType.ELIMINATE_RINGS_FROM_STACK:
    # After an explicit ELIMINATE_RINGS_FROM_STACK decision, re-evaluate whether
    # more territory decisions remain for the **same player**. When no further
    # regions exist, delegate to the canonical post-territory helper so that we
    # either enter FORCED_ELIMINATION (ANM + material) or perform a full turn
    # end via GameEngine._end_turn, mirroring the TS orchestrator semantics for
    # territory completion.
    remaining_regions = GameEngine._get_territory_processing_moves(
      game_state,
      current_player,
    )

    if remaining_regions:
      # Stay in territory_processing and keep the current player; hosts will
      # surface the next PROCESS_TERRITORY_REGION decision.
      game_state.current_phase = GamePhase.TERRITORY_PROCESSING
    else:
      _on_territory_processing_complete(game_state, trace_mode=trace_mode)

  elif last_move.type == MoveType.PROCESS_TERRITORY_REGION:
    # After processing a disconnected territory region, re-evaluate whether
    # more territory decisions remain for the **same player**. This mirrors
    # the TS orchestrator which stays in territory_processing until all
    # regions are resolved.
    remaining_regions = GameEngine._get_territory_processing_moves(
      game_state,
      current_player,
    )

    if remaining_regions:
      # Stay in territory_processing and keep the current player; hosts will
      # surface the next PROCESS_TERRITORY_REGION decision.
      game_state.current_phase = GamePhase.TERRITORY_PROCESSING
    else:
      # No further territory decisions remain; delegate to the shared
      # post-territory helper so that we either:
      # - enter FORCED_ELIMINATION when the player had no actions this
      #   entire turn but still controls stacks; or
      # - end the turn and rotate to the next player.
      _on_territory_processing_complete(game_state, trace_mode=trace_mode)

  elif last_move.type == MoveType.NO_TERRITORY_ACTION:
    # Forced no-op: player entered territory_processing but had no eligible
    # regions. Per RR-CANON-R075, this move marks that the phase was visited.
    # After a NO_TERRITORY_ACTION, treat territory_processing as complete for
    # this player and delegate to the canonical post-territory helper.
    _on_territory_processing_complete(game_state, trace_mode=trace_mode)

  elif last_move.type in (
    MoveType.TERRITORY_CLAIM,
    MoveType.CHOOSE_TERRITORY_OPTION,
  ):
    # Territory option choices are phase-preserving; explicit turn
    # rotation is driven by PROCESS_TERRITORY_REGION / elimination.
    pass
