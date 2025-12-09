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
    # After processing a line decision, check if there are more lines
    # for the current player to process. If not, advance to territory
    # processing (which may end the turn if no territories either).
    # This mirrors the TS TurnEngine behaviour where line_processing
    # automatically advances when no further decisions remain.
    # NOTE: Filter out NO_LINE_ACTION - we only want actual line moves
    # (PROCESS_LINE, CHOOSE_LINE_REWARD) to keep us in line_processing.
    remaining_lines = [
      m for m in GameEngine._get_line_processing_moves(
        game_state,
        current_player,
      )
      if m.type != MoveType.NO_LINE_ACTION
    ]
    if not remaining_lines:
      GameEngine._advance_to_territory_processing(
        game_state, trace_mode=trace_mode
      )

  elif last_move.type == MoveType.NO_LINE_ACTION:
    # Forced no-op: player entered line_processing but had no lines
    # to process. Per RR-CANON-R075, this move marks that the phase
    # was visited. After NO_LINE_ACTION, advance to territory_processing.
    GameEngine._advance_to_territory_processing(
      game_state, trace_mode=trace_mode
    )

  elif last_move.type == MoveType.ELIMINATE_RINGS_FROM_STACK:
    # After ELIMINATE_RINGS_FROM_STACK, check if the game will end:
    # - Terminal: no stacks left AND no player has rings in hand
    #   → stay on current player, set phase to territory_processing
    # - Non-terminal: rotate to next player, set phase to ring_placement
    no_stacks_left = not game_state.board.stacks
    any_rings_in_hand = any(
      p.rings_in_hand > 0 for p in game_state.players
    )

    if no_stacks_left and not any_rings_in_hand:
      # Terminal case: game will end, preserve current player
      # and set phase to territory_processing (game over state)
      game_state.current_phase = GamePhase.TERRITORY_PROCESSING
      game_state.must_move_from_stack_key = None
    else:
      # Non-terminal case: perform a full turn end with ANM/FE gating.
      # This mirrors TS turnLogic.advanceTurnAndPhase (which applies
      # forced_elimination/no-action requirements rather than silently
      # rotating). Using _end_turn ensures the next seat either has a
      # legal action or will surface a forced_elimination/no_*_action
      # requirement instead of leaving the recorder in an ANM state.
      GameEngine._end_turn(game_state, trace_mode=trace_mode)

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
      # surface the next process_territory_region decision.
      game_state.current_phase = GamePhase.TERRITORY_PROCESSING
    else:
      # No further territory decisions remain; rotate to the next player
      # without applying forced elimination gating (mirrors TS turn end).
      GameEngine._rotate_to_next_active_player(game_state)

  elif last_move.type == MoveType.NO_TERRITORY_ACTION:
    # Forced no-op: player entered territory_processing but had no
    # eligible regions. Per RR-CANON-R075, this move marks that the
    # phase was visited. After a NO_TERRITORY_ACTION, rotate to the
    # next player just like after PROCESS_TERRITORY_REGION.
    GameEngine._rotate_to_next_active_player(game_state)

  elif last_move.type in (
    MoveType.TERRITORY_CLAIM,
    MoveType.CHOOSE_TERRITORY_OPTION,
  ):
    # Territory option choices are phase-preserving; explicit turn
    # rotation is driven by PROCESS_TERRITORY_REGION / elimination.
    pass
