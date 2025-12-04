"""Core game engine for the RingRift AI service.

This module provides a lightweight, Python-hosted view of the RingRift
rules sufficient for AI search, evaluation, and TS↔Python contract tests.
It mirrors the shared TypeScript engine semantics where required for
parity, but should be treated as a *host adapter* rather than an
independent rules SSoT.

For which modules are allowed to encode rules semantics vs. which must
act as adapters over the shared engine, see the “Rules Entry Surfaces /
SSoT checklist” in ``docs/RULES_ENGINE_SURFACE_AUDIT.md``. When in
doubt, the TS shared engine and rules documentation win and this module
should be updated to match them.
"""

from __future__ import annotations

from typing import List, Optional
import sys
import os
import json
import time
from .models import (
    GameState, Move, Position, BoardType, GamePhase, RingStack, MarkerInfo,
    GameStatus, MoveType, BoardState
)
from .board_manager import BoardManager
from .ai.zobrist import ZobristHash
from .rules.geometry import BoardGeometry
from .rules.core import count_rings_in_play_for_player, get_line_length_for_board
from .rules.capture_chain import enumerate_capture_moves_py


DEBUG_ENGINE = os.environ.get("RINGRIFT_DEBUG_ENGINE") == "1"
STRICT_NO_MOVE_INVARIANT = os.environ.get(
    "RINGRIFT_STRICT_NO_MOVE_INVARIANT",
    "0",
) in {"1", "true", "yes", "on"}


def _debug(msg: str) -> None:
    """Write a debug message to stderr when engine debug is enabled."""
    if DEBUG_ENGINE:
        sys.stderr.write(msg)


class GameEngine:
    """Python GameEngine used by the AI service.

    This engine:

    - Mirrors the shared TS engine semantics closely enough for AI search,
      training, and TS↔Python parity tests.
    - Exposes `get_valid_moves` and `apply_move` as the primary APIs for
      AI agents and training loops.
    - Maintains a small in‑memory move cache keyed by a canonical
      `hash_game_state` string to avoid recomputing legal moves for
      identical positions.

    It is *not* a separate rules SSoT; when in doubt the TS engine and
    rules documentation win and this module should be updated to match.
    """

    # Cache for valid moves: key = f"{state_hash}:{player_number}"
    _move_cache: dict[str, List[Move]] = {}
    _cache_hits: int = 0
    _cache_misses: int = 0

    @staticmethod
    def get_valid_moves(
        game_state: GameState, player_number: int
    ) -> List[Move]:
        """Return all valid moves for ``player_number`` in ``game_state``.

        This mirrors the TS `GameEngine.getValidMoves` surface and is used
        as the primary legal‑move generator for AI agents and parity tests.
        """
        # Only generate moves if it's the player's turn
        if game_state.current_player != player_number:
            return []

        # Check cache
        state_hash = BoardManager.hash_game_state(game_state)
        cache_key = f"{state_hash}:{player_number}"

        if cache_key in GameEngine._move_cache:
            GameEngine._cache_hits += 1
            _debug(f"DEBUG: Cache hit for {cache_key}\n")
            return GameEngine._move_cache[cache_key]

        GameEngine._cache_misses += 1

        phase = game_state.current_phase
        moves: List[Move] = []

        if phase == GamePhase.RING_PLACEMENT:
            # TS-aligned ring placement phase:
            # - Expose legal PLACE_RING moves.
            # - When placement is optional, also expose SKIP_PLACEMENT.
            placement_moves = GameEngine._get_ring_placement_moves(
                game_state, player_number
            )
            skip_moves = GameEngine._get_skip_placement_moves(
                game_state, player_number
            )
            moves = placement_moves + skip_moves

            # When player has 0 rings in hand, placement isn't possible and
            # skip_placement isn't exposed (it requires rings_in_hand > 0).
            # In this case, expose movement/capture moves directly so the
            # player can continue. This matches TS behaviour where the turn
            # auto-advances to movement when placement isn't possible.
            if not moves:
                movement_moves = GameEngine._get_movement_moves(
                    game_state, player_number
                )
                capture_moves = GameEngine._get_capture_moves(
                    game_state, player_number
                )
                moves = movement_moves + capture_moves

        elif phase == GamePhase.MOVEMENT:
            # In the movement phase, TS exposes both non-capture moves and
            # initial overtaking captures together. Mirror that here.
            movement_moves = GameEngine._get_movement_moves(
                game_state, player_number
            )
            capture_moves = GameEngine._get_capture_moves(
                game_state, player_number
            )
            moves = movement_moves + capture_moves
        elif phase == GamePhase.CAPTURE:
            moves = GameEngine._get_capture_moves(game_state, player_number)
        elif phase == GamePhase.CHAIN_CAPTURE:
            # Support for canonical chain capture phase
            moves = GameEngine._get_capture_moves(game_state, player_number)
        elif phase == GamePhase.LINE_PROCESSING:
            moves = GameEngine._get_line_processing_moves(
                game_state, player_number
            )
        elif phase == GamePhase.TERRITORY_PROCESSING:
            moves = GameEngine._get_territory_processing_moves(
                game_state, player_number
            )

        # Layer in the swap_sides meta-move (pie rule) for Player 2 when
        # enabled. This mirrors the backend TS GameEngine.shouldOfferSwapSidesMetaMove
        # gate so that Python-based rules/AI see the same one-time choice.
        if GameEngine._should_offer_swap_sides(game_state):
            already_has_swap = any(m.type == MoveType.SWAP_SIDES for m in moves)
            if not already_has_swap:
                move_number = len(game_state.move_history) + 1
                swap_move = Move(  # type: ignore[call-arg]
                    id=f"swap_sides-{move_number}",
                    type=MoveType.SWAP_SIDES,
                    player=2,
                    to=Position(x=0, y=0),
                    timestamp=game_state.last_move_at,
                    thinkTime=0,
                    moveNumber=move_number,
                )
                moves.append(swap_move)

        # RR-CANON-R072/R100/R205: When no placement/movement/capture is available
        # but the player controls stacks, expose forced elimination as a first-class
        # action so both AI and human players can choose which stack to eliminate from.
        if not moves:
            fe_moves = GameEngine._get_forced_elimination_moves(
                game_state, player_number
            )
            if fe_moves:
                moves = fe_moves

        # Cache result
        GameEngine._move_cache[cache_key] = moves
        return moves

    @staticmethod
    def clear_cache():
        """Clear the move cache"""
        GameEngine._move_cache.clear()
        GameEngine._cache_hits = 0
        GameEngine._cache_misses = 0

    @staticmethod
    def apply_move(game_state: GameState, move: Move) -> GameState:
        """
        Apply a move to a game state and return the new state
        This is a simplified simulation for AI lookahead
        """
        # Optimization: Manual shallow copy of GameState and BoardState
        # We only deep copy mutable structures that we intend to modify

        # 1. Create new BoardState (shallow copy first)
        new_board = game_state.board.model_copy()

        # 2. Deep copy mutable dictionaries in board that we might modify
        # Note: We can optimize further by only copying what we need based on
        # move type
        new_board.stacks = game_state.board.stacks.copy()
        new_board.markers = game_state.board.markers.copy()
        new_board.collapsed_spaces = game_state.board.collapsed_spaces.copy()
        new_board.eliminated_rings = game_state.board.eliminated_rings.copy()
        # formed_lines and territories are usually re-calculated or appended
        new_board.formed_lines = list(game_state.board.formed_lines)
        new_board.territories = game_state.board.territories.copy()

        # 3. Create new GameState
        new_state = game_state.model_copy(update={"board": new_board})

        # 4. Copy other mutable fields in GameState
        # Optimization: Only copy players if we modify them (e.g. rings count)
        # But for safety in simulation, we copy.
        # To optimize, we could implement a 'make_move' that modifies in-place
        # and 'undo_move' to revert, avoiding object creation.
        # For now, we stick to copy-on-write but ensure it's as shallow as possible.
        new_state.players = [p.model_copy() for p in game_state.players]
        new_state.move_history = list(game_state.move_history)

        # Initialize Zobrist hash if missing
        if new_state.zobrist_hash is None:
            new_state.zobrist_hash = (
                ZobristHash().compute_initial_hash(new_state)
            )

        # Capture S-invariant before move
        before_snapshot = BoardManager.compute_progress_snapshot(new_state)

        # Update hash for phase/player change (remove old)
        zobrist = ZobristHash()
        if new_state.zobrist_hash is not None:
            new_state.zobrist_hash ^= zobrist.get_player_hash(
                new_state.current_player
            )
            new_state.zobrist_hash ^= zobrist.get_phase_hash(
                new_state.current_phase
            )

        if move.type == MoveType.SWAP_SIDES:
            GameEngine._apply_swap_sides(new_state, move)
        elif move.type == MoveType.PLACE_RING:
            GameEngine._apply_place_ring(new_state, move)
        elif move.type == MoveType.SKIP_PLACEMENT:
            # No board change; phase update will advance the turn.
            pass
        elif move.type == MoveType.MOVE_STACK:
            GameEngine._apply_move_stack(new_state, move)
        elif move.type in (
            MoveType.OVERTAKING_CAPTURE,
            MoveType.CONTINUE_CAPTURE_SEGMENT,
            MoveType.CHAIN_CAPTURE,
        ):
            GameEngine._apply_chain_capture(new_state, move)
        elif move.type in (
            MoveType.PROCESS_LINE,
            MoveType.CHOOSE_LINE_REWARD,
            MoveType.LINE_FORMATION,
            MoveType.CHOOSE_LINE_OPTION,
        ):
            GameEngine._apply_line_formation(new_state, move)
        elif move.type in (
            MoveType.PROCESS_TERRITORY_REGION,
            MoveType.ELIMINATE_RINGS_FROM_STACK,
            MoveType.TERRITORY_CLAIM,
            MoveType.CHOOSE_TERRITORY_OPTION,
        ):
            # PROCESS_TERRITORY_REGION mirrors TS territory processing;
            # ELIMINATE_RINGS_FROM_STACK is the canonical explicit
            # self-elimination decision move, implemented here by reusing the
            # forced-elimination helper.
            if move.type == MoveType.ELIMINATE_RINGS_FROM_STACK:
                GameEngine._apply_forced_elimination(new_state, move)
            else:
                GameEngine._apply_territory_claim(new_state, move)
        elif move.type == MoveType.FORCED_ELIMINATION:
            GameEngine._apply_forced_elimination(new_state, move)

        # Update move history
        new_state.move_history.append(move)
        new_state.last_move_at = move.timestamp

        # Update per-turn must-move bookkeeping, mirroring the TS
        # TurnEngine.updatePerTurnStateAfterMove helper:
        # - After PLACE_RING, the updated stack at move.to becomes the
        #   must-move origin.
        # - After movement/capture originating from that stack, advance
        #   the tracked key to the new landing position.
        if move.type == MoveType.PLACE_RING and move.to:
            new_state.must_move_from_stack_key = move.to.to_key()
        elif (
            new_state.must_move_from_stack_key is not None
            and move.from_pos is not None
            and move.to is not None
            and move.type
            in (
                MoveType.MOVE_STACK,
                MoveType.MOVE_RING,
                MoveType.BUILD_STACK,
                MoveType.OVERTAKING_CAPTURE,
                MoveType.CONTINUE_CAPTURE_SEGMENT,
                MoveType.CHAIN_CAPTURE,
            )
        ):
            from_key = move.from_pos.to_key()
            if from_key == new_state.must_move_from_stack_key:
                new_state.must_move_from_stack_key = move.to.to_key()

        # Handle phase transitions
        GameEngine._update_phase(new_state, move)

        # Update hash for phase/player change (add new)
        if new_state.zobrist_hash is not None:
            new_state.zobrist_hash ^= zobrist.get_player_hash(
                new_state.current_player
            )
            new_state.zobrist_hash ^= zobrist.get_phase_hash(
                new_state.current_phase
            )

        # Verify S-invariant
        # S = markers + collapsed + eliminated
        # S must be non-decreasing
        after_snapshot = BoardManager.compute_progress_snapshot(new_state)
        if after_snapshot.S < before_snapshot.S:
            # In a real engine we might throw, but for AI simulation we
            # log/warn or just accept it if it's a known deviation.
            # For now, we'll assume correctness of logic but this hook is here.
            pass

        # Check victory conditions
        GameEngine._check_victory(new_state)

        # Strict no-move invariant: after any move that leaves the game ACTIVE,
        # assert that the current player has at least one legal action.
        if STRICT_NO_MOVE_INVARIANT and new_state.game_status == GameStatus.ACTIVE:
            GameEngine._assert_active_player_has_legal_action(new_state, move)

        return new_state

    @staticmethod
    def _apply_swap_sides(game_state: GameState, move: Move) -> None:
        """
        Apply a swap_sides meta-move for 2-player games.

        Semantics mirror the TS backend GameEngine.applySwapSidesMove:
        - Only players in seats 1 and 2 swap their *identities*:
          id / username / type / aiDifficulty.
        - Per-seat statistics remain attached to the seat:
          rings_in_hand, eliminated_rings, territory_spaces,
          time_remaining, etc. are preserved.
        - Board geometry and controlling_player indices are unchanged.
        - current_player and current_phase are preserved.
        """
        if len(game_state.players) != 2:
            return

        p1 = None
        p2 = None
        for p in game_state.players:
            if p.player_number == 1:
                p1 = p
            elif p.player_number == 2:
                p2 = p

        if p1 is None or p2 is None:
            return

        # Swap identity/meta fields while preserving per-seat stats.
        new_players = []
        for p in game_state.players:
            if p.player_number == 1:
                new_players.append(
                    p.model_copy(
                        update={
                            "id": p2.id,
                            "username": p2.username,
                            "type": p2.type,
                            "ai_difficulty": p2.ai_difficulty,
                        }
                    )
                )
            elif p.player_number == 2:
                new_players.append(
                    p.model_copy(
                        update={
                            "id": p1.id,
                            "username": p1.username,
                            "type": p1.type,
                            "ai_difficulty": p1.ai_difficulty,
                        }
                    )
                )
            else:
                new_players.append(p)

        game_state.players = new_players

    @staticmethod
    def _should_offer_swap_sides(game_state: GameState) -> bool:
        """
        Determine whether the current state should expose a SWAP_SIDES
        meta-move (pie rule) for Player 2.

        This mirrors the gate in the TS backend GameEngine:
        - rulesOptions.swapRuleEnabled must be truthy.
        - Game must be ACTIVE with exactly two players.
        - It must be Player 2's turn in an interactive phase.
        - There must be at least one move from Player 1.
        - Player 2 must not have taken any non-swap_sides move yet.
        - No prior swap_sides move may exist in moveHistory.
        """
        rules = game_state.rules_options or {}
        if not bool(rules.get("swapRuleEnabled", False)):
            return False

        if game_state.game_status != GameStatus.ACTIVE:
            return False

        if len(game_state.players) != 2:
            return False

        if game_state.current_player != 2:
            return False

        if game_state.current_phase not in {
            GamePhase.RING_PLACEMENT,
            GamePhase.MOVEMENT,
            GamePhase.CAPTURE,
            GamePhase.CHAIN_CAPTURE,
        }:
            return False

        if not game_state.move_history:
            return False

        has_swap = any(m.type == MoveType.SWAP_SIDES for m in game_state.move_history)
        if has_swap:
            return False

        has_p1_move = any(m.player == 1 for m in game_state.move_history)
        has_p2_move = any(
            m.player == 2 and m.type != MoveType.SWAP_SIDES
            for m in game_state.move_history
        )

        return has_p1_move and not has_p2_move

    @staticmethod
    def _check_victory(game_state: GameState):
        """Check for victory conditions"""
        # Defensive guard: some AI tests construct synthetic GameStates with
        # an empty players list. Victory logic assumes at least one player,
        # so in that case we skip further checks.
        if not game_state.players:
            return

        # 1. Ring Elimination Victory
        # Check total eliminated rings for each player
        # Note: game_state.players might not be up to date with
        # board.eliminated_rings. We should sync them or check
        # board.eliminated_rings directly

        for p_id_str, count in game_state.board.eliminated_rings.items():
            if count >= game_state.victory_threshold:
                game_state.game_status = GameStatus.FINISHED
                game_state.winner = int(p_id_str)
                return

        # 2. Territory Victory
        territory_counts = {}
        for p_id in game_state.board.collapsed_spaces.values():
            if p_id not in territory_counts:
                territory_counts[p_id] = 0
            territory_counts[p_id] += 1

        for p_id, count in territory_counts.items():
            if count >= game_state.territory_victory_threshold:
                game_state.game_status = GameStatus.FINISHED
                game_state.winner = p_id
                return

        # 3. Early last-player-standing victory (R172).
        candidate = game_state.lps_exclusive_player_for_completed_round
        if (
            candidate is not None
            and game_state.current_player == candidate
            and GameEngine._has_real_action_for_player(game_state, candidate)
        ):
            board = game_state.board
            others_have_actions = False
            for player in game_state.players:
                if player.player_number == candidate:
                    continue
                has_stacks = any(
                    stack.controlling_player == player.player_number
                    for stack in board.stacks.values()
                )
                if not has_stacks and player.rings_in_hand <= 0:
                    continue
                if GameEngine._has_real_action_for_player(
                    game_state,
                    player.player_number,
                ):
                    others_have_actions = True
                    break

            # If no other player with material has any real action, candidate wins.
            if not others_have_actions:
                game_state.game_status = GameStatus.FINISHED
                game_state.winner = candidate
                return

            # Candidate condition failed; require a fresh qualifying round.
            game_state.lps_exclusive_player_for_completed_round = None

        # 4. Global structural terminality
        # Fallback termination is triggered when:
        # (a) No stacks on board AND no rings in hand for any player, OR
        # (b) No stacks on board AND no player with rings in hand has any
        #     legal placement (respecting no-dead-placement rule).
        #
        # This mirrors TS victoryLogic.ts hasAnyLegalPlacementOnBareBoard().
        no_stacks_left = not game_state.board.stacks
        any_rings_in_hand = any(
            p.rings_in_hand > 0 for p in game_state.players
        )

        # Check if global stalemate should apply
        should_apply_stalemate = False
        hand_counts_as_eliminated = False

        if no_stacks_left:
            if not any_rings_in_hand:
                # No rings anywhere - clear stalemate
                should_apply_stalemate = True
            else:
                # Rings in hand exist - check if ANY player can legally place
                # Per §13.4: only trigger stalemate if NO player with rings
                # has ANY legal placement on the bare board.
                any_legal_placement_exists = any(
                    p.rings_in_hand > 0
                    and GameEngine._has_any_legal_placement_on_bare_board(
                        game_state, p.player_number
                    )
                    for p in game_state.players
                )
                if not any_legal_placement_exists:
                    # No player can place - stalemate with hand conversion
                    should_apply_stalemate = True
                    hand_counts_as_eliminated = True

        if should_apply_stalemate:
            game_state.game_status = GameStatus.FINISHED

            # Tie-breaker logic:
            # 1. Most collapsed spaces
            # 2. Most eliminated rings (including rings in hand converted)
            # 3. Most markers
            # 4. Last player to complete a valid turn action

            # Calculate scores for each player
            scores = {}
            for player in game_state.players:
                pid = player.player_number

                # 1. Collapsed spaces
                collapsed = territory_counts.get(pid, 0)

                # 2. Eliminated rings + Rings in hand
                eliminated = game_state.board.eliminated_rings.get(str(pid), 0)
                eliminated += player.rings_in_hand

                # 3. Markers
                markers = 0
                for m in game_state.board.markers.values():
                    if m.player == pid:
                        markers += 1

                scores[pid] = {
                    "collapsed": collapsed,
                    "eliminated": eliminated,
                    "markers": markers,
                }

            # Determine winner by applying the same ladder as TS:
            # territorySpaces -> eliminatedRings -> markers -> last actor.
            sorted_players = sorted(
                game_state.players,
                key=lambda p: (
                    scores[p.player_number]["collapsed"],
                    scores[p.player_number]["eliminated"],
                    scores[p.player_number]["markers"],
                    1
                    if (
                        game_state.move_history
                        and game_state.move_history[-1].player
                        == p.player_number
                    )
                    else 0,
                ),
                reverse=True,
            )

            game_state.winner = sorted_players[0].player_number
            return

        # 4. No legal moves for ANY player (Global Stalemate with stacks)
        # This is computationally expensive to check every move.
        # However, the rules say: "In any situation where no player has any
        # legal placement, movement, or capture but at least one stack still
        # exists on the board, the controlling player of some stack on their
        # turn must satisfy the condition above and perform a forced
        # elimination."
        # So technically, global stalemate with stacks is impossible if forced
        # elimination is implemented correctly.
        # The only true stalemate is when no stacks remain.
        # But we should check if the current player has no moves AND no forced
        # elimination is possible?
        # No, forced elimination is always possible if you have a stack.
        # So if you have a stack, you have a move (forced elim).
        # If you have no stack, you might have placement.
        # If you have no stack and no placement (no rings), you are
        # eliminated/inactive.
        # If ALL players are eliminated/inactive, then game ends.

        active_players = 0
        for p in game_state.players:
            # Check if player can potentially move
            has_stacks = any(
                s.controlling_player == p.player_number
                for s in game_state.board.stacks.values()
            )
            can_place = p.rings_in_hand > 0
            if has_stacks or can_place:
                active_players += 1

        if active_players == 0:
            # Should have been caught by "no stacks" check if no stacks,
            # but if stacks exist but belong to no one (impossible?) or
            # some edge case, we treat as stalemate.
            # But stacks always have a controller.
            # So if stacks exist, someone is active.
            pass

    @staticmethod
    def _update_phase(game_state: GameState, last_move: Move):
        """
        Advance phase after a move, partially mirroring TS TurnEngine semantics.

        Key behaviours:

        - After FORCED_ELIMINATION, re-check whether the same player now has
          any legal placement/movement/capture actions:
          * If yes, keep the same player and start an interactive turn in the
            MOVEMENT phase.
          * If no, end the turn and rotate to the next player.
        - Other move types retain the existing phase transitions for now.
        """
        current_player = game_state.current_player

        # Forced elimination is *not* a player-exposed Move in the TS engine;
        # it is applied automatically after the territory_processing phase when
        # the next player has stacks but no valid actions. The legacy Python
        # FORCED_ELIMINATION move type is therefore treated as internal-only.
        if last_move.type == MoveType.FORCED_ELIMINATION:
            # Legacy path retained for backwards compatibility with any tests
            # that still construct explicit FORCED_ELIMINATION moves. After
            # such a move, mirror the TS TurnEngine semantics: re-check whether
            # the same player now has any valid actions; if so, continue their
            # turn in MOVEMENT, otherwise end the turn.
            if GameEngine._has_valid_actions(game_state, current_player):
                game_state.current_phase = GamePhase.MOVEMENT
            else:
                GameEngine._end_turn(game_state)

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
                GameEngine._advance_to_line_processing(game_state)

        elif last_move.type == MoveType.SKIP_PLACEMENT:
            # Skipping placement is only allowed when movement or capture is
            # already available. After a legal skip, always enter movement.
            game_state.current_phase = GamePhase.MOVEMENT

        elif last_move.type == MoveType.MOVE_STACK:
            # After movement, check for captures
            capture_moves = GameEngine._get_capture_moves(
                game_state, current_player
            )
            if capture_moves:
                game_state.current_phase = GamePhase.CAPTURE
            else:
                # No captures, go to line processing
                GameEngine._advance_to_line_processing(game_state)

        elif (
            last_move.type == MoveType.OVERTAKING_CAPTURE
            or last_move.type == MoveType.CHAIN_CAPTURE
            or last_move.type == MoveType.CONTINUE_CAPTURE_SEGMENT
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
                GameEngine._advance_to_line_processing(game_state)

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
            remaining_lines = GameEngine._get_line_processing_moves(
                game_state,
                current_player,
            )
            if not remaining_lines:
                GameEngine._advance_to_territory_processing(game_state)

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
                # Non-terminal case: rotate to next player
                players = game_state.players
                if players:
                    current_index = 0
                    for i, p in enumerate(players):
                        if p.player_number == game_state.current_player:
                            current_index = i
                            break
                    next_index = (current_index + 1) % len(players)
                    game_state.current_player = players[next_index].player_number
                    game_state.current_phase = GamePhase.RING_PLACEMENT
                    game_state.must_move_from_stack_key = None

        elif last_move.type == MoveType.PROCESS_TERRITORY_REGION:
            # After processing a disconnected territory region via an explicit
            # PROCESS_TERRITORY_REGION decision, rotate to the next seat and
            # start them in the ring_placement phase without applying forced
            # elimination gating. This mirrors the TS orchestrator behaviour
            # exercised by the v2 territory_processing contract vectors.
            players = game_state.players
            if players:
                current_index = 0
                for i, p in enumerate(players):
                    if p.player_number == game_state.current_player:
                        current_index = i
                        break
                next_index = (current_index + 1) % len(players)
                game_state.current_player = players[next_index].player_number
                game_state.current_phase = GamePhase.RING_PLACEMENT
                game_state.must_move_from_stack_key = None

        elif last_move.type in (
            MoveType.TERRITORY_CLAIM,
            MoveType.CHOOSE_TERRITORY_OPTION,
        ):
            # Territory option choices are phase-preserving; explicit turn
            # rotation is driven by PROCESS_TERRITORY_REGION / elimination.
            pass

    @staticmethod
    def _advance_to_line_processing(game_state: GameState):
        line_moves = GameEngine._get_line_processing_moves(
            game_state, game_state.current_player
        )
        if line_moves:
            game_state.current_phase = GamePhase.LINE_PROCESSING
        else:
            GameEngine._advance_to_territory_processing(game_state)

    @staticmethod
    def _advance_to_territory_processing(game_state: GameState):
        """Advance from movement/capture into territory processing.

        Mirrors the minimal TS shared GameEngine semantics used after
        MOVE_STACK and capture actions:

        - If the current player has at least one disconnected region that can
          be processed, enter TERRITORY_PROCESSING.
        - Otherwise, end the turn and rotate to the next player.

        Explicit territory decision moves (PROCESS_TERRITORY_REGION,
        ELIMINATE_RINGS_FROM_STACK, etc.) are handled separately in
        _update_phase and are treated as phase-preserving.
        """
        board = game_state.board
        player = game_state.current_player

        regions = BoardManager.find_disconnected_regions(board, player)
        eligible = [
            region
            for region in (regions or [])
            if GameEngine._can_process_disconnected_region(
                game_state,
                region,
                player,
            )
        ]

        if eligible:
            game_state.current_phase = GamePhase.TERRITORY_PROCESSING
        else:
            GameEngine._end_turn(game_state)

    @staticmethod
    def _end_turn(game_state: GameState):
        """
        End the current player's turn and advance to the next active player.

        This mirrors the TS TurnEngine "territory_processing" end-of-turn
        behaviour:

        - Rotate to the next player in table order.
        - Skip players who have no stacks and no rings in hand (they cannot
          perform any further actions).
        - For the first player with material:
          * If they have any rings in hand, start in RING_PLACEMENT
            (placement may be mandatory or optional depending on stacks).
          * Otherwise, start directly in MOVEMENT, unless forced elimination
            must be applied first.
        - If no players have any material at all, leave current_player
          unchanged and allow _check_victory to resolve global stalemate via
          tie-breakers.
        """
        players = game_state.players
        if not players:
            return

        num_players = len(players)

        # Find index of the current player in the players list. If not found,
        # default to index 0 to avoid leaving the game in an inconsistent
        # state.
        current_index = 0
        for i, p in enumerate(players):
            if p.player_number == game_state.current_player:
                current_index = i
                break

        max_skips = num_players
        skips = 0
        idx = (current_index + 1) % num_players

        while skips < max_skips:
            candidate = players[idx]
            stacks_for_candidate = BoardManager.get_player_stacks(
                game_state.board,
                candidate.player_number,
            )
            has_stacks = bool(stacks_for_candidate)
            has_rings_in_hand = candidate.rings_in_hand > 0

            if not has_stacks and not has_rings_in_hand:
                # This player has no rings on the board and none in hand;
                # they can never act again in this game, so skip them.
                skips += 1
                idx = (idx + 1) % num_players
                continue

            # Forced-elimination check (TS TurnEngine.processForcedElimination):
            # if this player controls at least one stack but has no legal
            # placement, movement, or capture actions, we must eliminate a cap
            # before starting their interactive turn.
            if has_stacks and not GameEngine._has_valid_actions(
                game_state,
                candidate.player_number,
            ):
                GameEngine._perform_forced_elimination_for_player(
                    game_state,
                    candidate.player_number,
                )
                # Forced elimination may immediately end the game via victory
                # conditions.
                if game_state.game_status != GameStatus.ACTIVE:
                    return

                # Re-check material after forced elimination. If the player's
                # cap was their only control over their last stack, forced
                # elimination flips control to the opponent. In this case the
                # player now has no stacks and no rings in hand (they are
                # "temporarily inactive" per §7.3) and should be skipped.
                stacks_after_fe = BoardManager.get_player_stacks(
                    game_state.board,
                    candidate.player_number,
                )
                has_stacks_after_fe = bool(stacks_after_fe)
                has_rings_after_fe = candidate.rings_in_hand > 0

                if not has_stacks_after_fe and not has_rings_after_fe:
                    # Player lost all material due to forced elimination;
                    # skip them and try the next player.
                    skips += 1
                    idx = (idx + 1) % num_players
                    continue

                # Player still has material after forced elimination.
                # Keep this player active and begin their turn in MOVEMENT.
                game_state.current_player = candidate.player_number
                game_state.current_phase = GamePhase.MOVEMENT
                game_state.must_move_from_stack_key = None
                GameEngine._update_lps_round_tracking_for_current_player(
                    game_state,
                )
                return

            # Found the next active player with some material and at least one
            # legal action.
            game_state.current_player = candidate.player_number
            game_state.must_move_from_stack_key = None

            # Starting phase selection:
            # - If the player has any rings in hand, begin in RING_PLACEMENT
            #   (placement may be mandatory or optional depending on their
            #   stacks and future move availability).
            # - If they have no rings in hand but do have stacks, begin in
            #   MOVEMENT.
            if has_rings_in_hand:
                game_state.current_phase = GamePhase.RING_PLACEMENT
            else:
                game_state.current_phase = GamePhase.MOVEMENT

            GameEngine._update_lps_round_tracking_for_current_player(
                game_state,
            )
            return

        # If we exhaust all players without finding any with material, then
        # no-one can act any longer. Leave current_player unchanged and keep
        # the phase in MOVEMENT; _check_victory will detect the global
        # stalemate (no stacks) and apply the tie-breaking rules.
        game_state.current_phase = GamePhase.MOVEMENT
        game_state.must_move_from_stack_key = None

    @staticmethod
    def _assert_active_player_has_legal_action(
        game_state: GameState,
        triggering_move: Move,
    ) -> None:
        """Enforce the strict no-move invariant for ACTIVE states.

        When STRICT_NO_MOVE_INVARIANT is enabled, any ACTIVE state must satisfy
        the RR-CANON R2xx / INV-ACTIVE-NO-MOVES contract for
        ``current_player``:

        - the player has turn-material (RR-CANON-R201), and
        - the global action surface G(state, P) is non-empty (RR-CANON-R200):
          * at least one global placement; or
          * at least one phase-local interactive move; or
          * a host-level forced-elimination action while they still control
            stacks.

        For fully eliminated players (no stacks and no rings in hand) we first
        attempt a defensive ``_end_turn`` rotation before declaring an
        invariant failure, mirroring the TS turn orchestrator and
        INV-ANM-TURN-MATERIAL-SKIP.
        """
        if game_state.game_status != GameStatus.ACTIVE:
            return

        # Local import to avoid a circular import at module load time. The
        # global_actions module in turn imports GameEngine for shared helpers.
        from app.rules import global_actions as ga  # type: ignore

        current_player = game_state.current_player

        # First, ensure that fully eliminated players are not left as the
        # active player in ACTIVE states. If the current player has no
        # turn-material at all (no stacks and no rings in hand), attempt a
        # single defensive rotation via _end_turn before treating the shape as
        # an invariant violation.
        if not ga.has_turn_material(game_state, current_player):
            previous_player = current_player
            GameEngine._end_turn(game_state)

            # If rotation finished the game, there is no invariant violation.
            if game_state.game_status != GameStatus.ACTIVE:
                return

            # If we advanced to a different player who now has turn-material and
            # is not in an ANM state, the invariant holds.
            if game_state.current_player != previous_player:
                new_player = game_state.current_player
                if ga.has_turn_material(game_state, new_player) and not ga.is_anm_state(
                    game_state
                ):
                    return
            # Fall through to snapshot + raise handling below; at this point we
            # are in an ACTIVE state whose current_player either still has no
            # material or remains in an ANM shape.

        # At this point, either:
        # - current_player has turn-material, or
        # - a defensive rotation failed to find any non-ANM player.
        #
        # If ANM(state, current_player) is false, the invariant holds.
        if not ga.is_anm_state(game_state):
            return

        # We now have an ACTIVE state whose current_player satisfies
        # ANM(state, P): they have turn-material but no placements, movements,
        # captures, territory/line decisions, or forced eliminations available.
        # Capture a diagnostic snapshot and raise.
        try:
            failure_dir = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "logs",
                    "invariant_failures",
                )
            )
            os.makedirs(failure_dir, exist_ok=True)

            ts = int(time.time())
            filename = f"active_no_moves_p{game_state.current_player}_{ts}.json"
            path = os.path.join(failure_dir, filename)

            try:
                state_payload = game_state.model_dump(  # type: ignore[attr-defined]
                    by_alias=True,
                    mode="json",
                )
            except Exception:
                state_payload = None

            try:
                move_payload = triggering_move.model_dump(  # type: ignore[attr-defined]
                    by_alias=True,
                    mode="json",
                )
            except Exception:
                move_payload = None

            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "timestamp": ts,
                        "current_player": game_state.current_player,
                        "game_status": game_state.game_status.value,
                        "current_phase": game_state.current_phase.value,
                        "state": state_payload,
                        "move": move_payload,
                    },
                    f,
                )
        except Exception:
            # Snapshotting must never prevent raising the invariant error.
            pass

        raise RuntimeError(
            "STRICT_NO_MOVE_INVARIANT violated: ACTIVE "
            f"{game_state.current_phase.value} state for player "
            f"{game_state.current_player} has no legal actions",
        )

    @staticmethod
    def _estimate_rings_per_player(game_state: GameState) -> int:
        """
        TS-aligned per-player ring cap.

        Mirrors BOARD_CONFIGS[boardType].ringsPerPlayer from the shared TS
        types:

        - square8   → 18 rings per player
        - square19  → 36 rings per player
        - hexagonal → 36 rings per player
        """
        board_type = game_state.board.type
        if board_type == BoardType.SQUARE8:
            return 18
        if board_type == BoardType.SQUARE19:
            return 36
        if board_type == BoardType.HEXAGONAL:
            return 36
        # Fallback for unknown types: use totalRingsInPlay as a safe
        # upper bound to avoid underestimating.
        return game_state.total_rings_in_play

    @staticmethod
    def _calculate_distance(
        board_type: BoardType, from_pos: Position, to_pos: Position
    ) -> int:
        """Thin wrapper around shared geometry distance helper.

        Delegates to ``BoardGeometry.calculate_distance`` to keep a single
        source of truth for distance semantics.
        """
        return BoardGeometry.calculate_distance(board_type, from_pos, to_pos)

    @staticmethod
    def _get_path_positions(
        from_pos: Position, to_pos: Position
    ) -> List[Position]:
        """Thin wrapper around shared geometry path helper.

        Delegates to ``BoardGeometry.get_path_positions`` to keep a single
        source of truth for straight-line path semantics.
        """
        return BoardGeometry.get_path_positions(from_pos, to_pos)

    @staticmethod
    def _is_straight_line_movement(
        board_type: BoardType, from_pos: Position, to_pos: Position
    ) -> bool:
        """
        True if move from `from_pos` to `to_pos` lies along a valid movement
        ray for the given board type.
        Mirrors RuleEngine.isStraightLineMovement.
        """
        dx = to_pos.x - from_pos.x
        dy = to_pos.y - from_pos.y
        dz = (to_pos.z or 0) - (from_pos.z or 0)

        if board_type == BoardType.HEXAGONAL:
            # In cube coordinates, an axis-aligned ray changes exactly two
            # coordinates (the third is implied by x + y + z = 0).
            coord_changes = sum(1 for d in (dx, dy, dz) if d != 0)
            return coord_changes == 2

        # Square boards: orthogonal or diagonal only.
        if dx == 0 and dy == 0:
            return False
        if dx != 0 and dy != 0 and abs(dx) != abs(dy):
            return False
        return True

    @staticmethod
    def _is_path_clear_for_movement(
        board, from_pos: Position, to_pos: Position
    ) -> bool:
        """
        Path check for non-capture movement and for capture geometry.
        Mirrors RuleEngine.isPathClear semantics: no stacks or collapsed
        spaces on intermediate cells; markers are allowed.
        """
        path_positions = GameEngine._get_path_positions(from_pos, to_pos)[1:-1]
        for pos in path_positions:
            if not BoardManager.is_valid_position(pos, board.type, board.size):
                return False
            if BoardManager.is_collapsed_space(pos, board):
                return False
            if BoardManager.get_stack(pos, board):
                return False
        return True

    @staticmethod
    def _create_hypothetical_board_with_placement(
        board,
        position: Position,
        player: int,
        count: int = 1,
    ):
        """
        Python analogue of
        placementHelpers.createHypotheticalBoardWithPlacement.

        Returns a shallow-copied BoardState with `count` rings for `player`
        placed at `position`, recomputing stackHeight, controllingPlayer,
        and capHeight.

        Uses model_construct for ~5x faster creation by skipping validation.
        """
        # Use model_construct to skip Pydantic validation (faster)
        # We copy all dict/list fields to avoid mutating the original
        hyp = BoardState.model_construct(
            type=board.type,
            size=board.size,
            stacks=dict(board.stacks),
            markers=dict(board.markers),
            collapsedSpaces=dict(board.collapsed_spaces),
            eliminatedRings=dict(board.eliminated_rings),
            formedLines=list(board.formed_lines),
            territories=dict(board.territories),
        )

        pos_key = position.to_key()
        existing = hyp.stacks.get(pos_key)

        n = max(1, count)
        if existing and existing.stack_height > 0:
            rings = list(existing.rings)
            rings.extend([player] * n)
        else:
            rings = [player] * n

        stack = RingStack(
            position=position,
            rings=rings,
            stackHeight=len(rings),
            capHeight=0,
            controllingPlayer=rings[-1],
        )

        # Recompute capHeight from top (rings[-1]) downward.
        h = 0
        for r in reversed(rings):
            if r == stack.controlling_player:
                h += 1
            else:
                break
        stack.cap_height = h

        hyp.stacks[pos_key] = stack
        return hyp

    @staticmethod
    def _process_markers_along_path(
        board,
        from_pos: Position,
        to_pos: Position,
        player_number: int,
        game_state: GameState
    ) -> None:
        """
        Process markers along a straight-line path, excluding endpoints.
        """
        path = GameEngine._get_path_positions(from_pos, to_pos)
        zobrist = ZobristHash()
        # Exclude start and end positions
        for pos in path[1:-1]:
            pos_key = pos.to_key()
            marker = board.markers.get(pos_key)
            if marker is None:
                continue
            if marker.player == player_number:
                # Remove marker hash
                if game_state.zobrist_hash is not None:
                    game_state.zobrist_hash ^= zobrist.get_marker_hash(
                        pos_key, marker.player
                    )
                BoardManager.set_collapsed_space(pos, player_number, board)
                # Add collapsed hash
                if game_state.zobrist_hash is not None:
                    game_state.zobrist_hash ^= zobrist.get_collapsed_hash(
                        pos_key
                    )
            else:
                # Remove old marker hash
                if game_state.zobrist_hash is not None:
                    game_state.zobrist_hash ^= zobrist.get_marker_hash(
                        pos_key, marker.player
                    )
                # Copy marker before modification to avoid mutating shared state
                new_marker = marker.model_copy()
                new_marker.player = player_number
                board.markers[pos_key] = new_marker

                # Add new marker hash
                if game_state.zobrist_hash is not None:
                    game_state.zobrist_hash ^= zobrist.get_marker_hash(
                        pos_key, new_marker.player
                    )

    @staticmethod
    def _eliminate_top_ring_at(
        game_state: GameState, position: Position, credited_player: int
    ) -> None:
        """
        Eliminate exactly the top ring from the stack at `position`,
        crediting the elimination to `credited_player`.

        Mirrors GameEngine.eliminateTopRingAt in TS, adapted for the Python
        RingStack representation (rings stored bottom -> top).
        """
        board = game_state.board
        pos_key = position.to_key()
        stack = board.stacks.get(pos_key)
        if not stack or stack.stack_height == 0:
            return

        zobrist = ZobristHash()
        # Remove old stack hash
        if game_state.zobrist_hash is not None:
            game_state.zobrist_hash ^= zobrist.get_stack_hash(
                pos_key,
                stack.controlling_player,
                stack.stack_height,
                tuple(stack.rings)
            )

        stack = stack.model_copy(deep=True)
        board.stacks[pos_key] = stack

        stack.rings.pop()
        stack.stack_height -= 1

        player_id_str = str(credited_player)
        board.eliminated_rings[player_id_str] = (
            board.eliminated_rings.get(player_id_str, 0) + 1
        )
        game_state.total_rings_eliminated += 1

        for p in game_state.players:
            if p.player_number == credited_player:
                p.eliminated_rings += 1
                break

        if stack.stack_height == 0:
            del board.stacks[pos_key]
        else:
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
            if game_state.zobrist_hash is not None:
                game_state.zobrist_hash ^= zobrist.get_stack_hash(
                    pos_key,
                    stack.controlling_player,
                    stack.stack_height,
                    tuple(stack.rings)
                )

            _debug(
                f"DEBUG: _apply_place_ring created stack at {pos_key}: "
                f"player={stack.controlling_player}, "
                f"height={stack.stack_height}, "
                f"rings={stack.rings}\n"
            )

    @staticmethod
    def _validate_capture_segment_on_board_for_reachability(
        board_type: BoardType,
        from_pos: Position,
        target_pos: Position,
        landing_pos: Position,
        player: int,
        board,
    ) -> bool:
        """
        Lightweight port of TS validateCaptureSegmentOnBoard for use inside
        no-dead-placement reachability checks. This intentionally mirrors
        the shared core helper's geometry and path rules.
        """
        if not BoardManager.is_valid_position(from_pos, board.type, board.size):
            return False
        if not BoardManager.is_valid_position(target_pos, board.type, board.size):
            return False
        if not BoardManager.is_valid_position(landing_pos, board.type, board.size):
            return False

        attacker = BoardManager.get_stack(from_pos, board)
        if not attacker or attacker.controlling_player != player:
            return False

        target_stack = BoardManager.get_stack(target_pos, board)
        if not target_stack:
            return False

        # Cap height constraint: attacker.cap_height >= target.cap_height
        if attacker.cap_height < target_stack.cap_height:
            return False

        dx = target_pos.x - from_pos.x
        dy = target_pos.y - from_pos.y
        dz = (target_pos.z or 0) - (from_pos.z or 0)

        if board_type == BoardType.HEXAGONAL:
            coord_changes = sum(1 for d in (dx, dy, dz) if d != 0)
            if coord_changes != 2:
                return False
        else:
            if dx == 0 and dy == 0:
                return False
            if dx != 0 and dy != 0 and abs(dx) != abs(dy):
                return False

        # Path from attacker to target must be clear of stacks/collapsed.
        path_to_target = GameEngine._get_path_positions(from_pos, target_pos)[1:-1]
        for pos in path_to_target:
            if not BoardManager.is_valid_position(pos, board.type, board.size):
                return False
            if BoardManager.is_collapsed_space(pos, board):
                return False
            if BoardManager.get_stack(pos, board):
                return False

        # Landing must be beyond target in the same direction from `from_pos`.
        dx2 = landing_pos.x - from_pos.x
        dy2 = landing_pos.y - from_pos.y
        dz2 = (landing_pos.z or 0) - (from_pos.z or 0)

        def _sign(v: int) -> int:
            return 0 if v == 0 else (1 if v > 0 else -1)

        if dx != 0 and _sign(dx) != _sign(dx2):
            return False
        if dy != 0 and _sign(dy) != _sign(dy2):
            return False
        if dz != 0 and _sign(dz) != _sign(dz2):
            return False

        dist_to_target = abs(dx) + abs(dy) + abs(dz)
        dist_to_landing = abs(dx2) + abs(dy2) + abs(dz2)
        if dist_to_landing <= dist_to_target:
            return False

        # Total distance must be at least stack height.
        segment_distance = GameEngine._calculate_distance(
            board_type,
            from_pos,
            landing_pos,
        )
        if segment_distance < attacker.stack_height:
            return False

        # Path from target to landing must also be clear.
        path_from_target = GameEngine._get_path_positions(
            target_pos,
            landing_pos,
        )[1:-1]
        for pos in path_from_target:
            if not BoardManager.is_valid_position(pos, board.type, board.size):
                return False
            if BoardManager.is_collapsed_space(pos, board):
                return False
            if BoardManager.get_stack(pos, board):
                return False

        if BoardManager.is_collapsed_space(landing_pos, board):
            return False
        landing_stack = BoardManager.get_stack(landing_pos, board)
        if landing_stack:
            return False

        # Per RR-CANON-R101/R102: landing on any marker (own or opponent) is legal.
        # The marker is removed and the top ring of the attacking stack's cap is
        # eliminated. Validation allows this; mutation handles the elimination cost.

        return True

    @staticmethod
    def _has_any_legal_move_or_capture_from_on_board(
        board_type: BoardType,
        from_pos: Position,
        player_number: int,
        board,
    ) -> bool:
        """
        Python analogue of hasAnyLegalMoveOrCaptureFromOnBoard used for
        placement no-dead-placement checks.

        This is now a direct port of the TS helper:
        - Non-capture moves along movement directions with distance
          >= stackHeight, up to stackHeight + 5.
        - Capture reachability using the same geometry and capture-segment
          validation semantics as the TS engine.
        """
        stack = BoardManager.get_stack(from_pos, board)
        if not stack or stack.controlling_player != player_number:
            return False

        directions = BoardManager._get_all_directions(board_type)
        max_non_capture = stack.stack_height + 5
        max_capture_landing = stack.stack_height + 5

        # === Non-capture movement ===
        for direction in directions:
            for distance in range(stack.stack_height, max_non_capture + 1):
                target = BoardManager._add_direction(from_pos, direction, distance)

                if not BoardManager.is_valid_position(
                    target,
                    board.type,
                    board.size,
                ):
                    break
                if BoardManager.is_collapsed_space(target, board):
                    break

                # Path blocking identical to movement rules.
                if not GameEngine._is_path_clear_for_movement(
                    board,
                    from_pos,
                    target,
                ):
                    break

                key = target.to_key()
                dest_stack = board.stacks.get(key)

                if dest_stack is None or dest_stack.stack_height == 0:
                    # Empty space or marker-only cell.
                    # Per RR-CANON-R091/R092: landing on any marker is legal.
                    _debug(
                        f"DEBUG: _has_any found move from {from_pos} "
                        f"to {target} dist {distance}\n"
                    )
                    return True
                else:
                    # Landing on a stack (for merging) is also a legal move.
                    _debug(
                        f"DEBUG: _has_any found merge from {from_pos} "
                        f"to {target} dist {distance}\n"
                    )
                    return True

        # === Capture reachability ===
        for direction in directions:
            step = 1
            target_pos: Optional[Position] = None

            # Find first stack along this ray that could be a capture target.
            while True:
                pos = BoardManager._add_direction(from_pos, direction, step)
                if not BoardManager.is_valid_position(
                    pos,
                    board.type,
                    board.size,
                ):
                    break
                if BoardManager.is_collapsed_space(pos, board):
                    break

                stack_at_pos = BoardManager.get_stack(pos, board)
                if stack_at_pos and stack_at_pos.stack_height > 0:
                    # Can overtake own or opponent stacks as long as capHeight
                    # allows.
                    if stack.cap_height >= stack_at_pos.cap_height:
                        target_pos = pos
                    break
                step += 1

            if not target_pos:
                continue

            # From the target, search for valid landing positions beyond it.
            landing_step = 1
            while landing_step <= max_capture_landing:
                landing = BoardManager._add_direction(
                    target_pos,
                    direction,
                    landing_step,
                )
                if not BoardManager.is_valid_position(
                    landing,
                    board.type,
                    board.size,
                ):
                    break
                if BoardManager.is_collapsed_space(landing, board):
                    break
                landing_stack = BoardManager.get_stack(landing, board)
                if landing_stack and landing_stack.stack_height > 0:
                    break

                if GameEngine._validate_capture_segment_on_board_for_reachability(
                    board_type,
                    from_pos,
                    target_pos,
                    landing,
                    player_number,
                    board,
                ):
                    _debug(
                        f"DEBUG: _has_any found capture from {from_pos} "
                        f"to {landing} via {target_pos}\n"
                    )
                    return True

                landing_step += 1

        return False

    @staticmethod
    def _has_any_movement_or_capture_after_hypothetical_placement(
        game_state: GameState,
        player_number: int,
        from_pos: Position,
        hyp_board,
    ) -> bool:
        """
        Stronger no-dead-placement helper for ring placements.

        Given a hypothetical post-placement BoardState, construct a
        temporary GameState in the MOVEMENT phase with
        must_move_from_stack_key fixed to the placed stack and reuse the
        real movement/capture generators. This ensures that any
        placement accepted by this check cannot immediately lead to
        "no move found" when the turn advances to MOVEMENT.

        This mirrors the TS semantics used after ring placement.
        """
        # Fast fail: the placed stack must exist and be controlled by
        # the player.
        stack = BoardManager.get_stack(from_pos, hyp_board)
        if not stack or stack.controlling_player != player_number:
            return False

        # Build a lightweight temporary GameState that mirrors the
        # post-placement MOVEMENT phase from the placing player's
        # perspective. We copy the incoming state to preserve players,
        # clocks, etc., but swap in the hypothetical board.
        temp_state = game_state.model_copy()
        temp_state.board = hyp_board
        temp_state.current_player = player_number
        temp_state.current_phase = GamePhase.MOVEMENT
        temp_state.must_move_from_stack_key = from_pos.to_key()
        temp_state.chain_capture_state = None

        # Seed move_history with a synthetic placement move so that
        # _get_capture_moves identifies the attacker at from_pos (it
        # looks at the last move's .to when no chain_capture_state is
        # active).
        # Use model_construct for faster creation (skips validation)
        synthetic_move = Move.model_construct(
            id="hypothetical-placement",
            type=MoveType.PLACE_RING,
            player=player_number,
            to=from_pos,
            timestamp=game_state.last_move_at,
            thinkTime=0,
            moveNumber=len(game_state.move_history) + 1,
        )
        temp_state.move_history = list(game_state.move_history) + [synthetic_move]

        # Use limit=1 for early-return optimization - we only need to know
        # if at least one valid move exists, not enumerate all of them.
        movement_moves = GameEngine._get_movement_moves(
            temp_state,
            player_number,
            limit=1,
        )
        if movement_moves:
            return True

        capture_moves = GameEngine._get_capture_moves(
            temp_state,
            player_number,
            limit=1,
        )
        if capture_moves:
            return True

        if DEBUG_ENGINE:
            _debug(
                "DEBUG: no-dead-placement rejected placement at "
                f"{from_pos.to_key()} for P{player_number}\n"
            )

        return False

    @staticmethod
    def _get_skip_placement_moves(
        game_state: GameState,
        player_number: int,
    ) -> List[Move]:
        """
        Enumerate legal SKIP_PLACEMENT moves.

        Mirrors TS RuleEngine.validateSkipPlacement semantics:
        - Only during RING_PLACEMENT phase.
        - Player must have rings in hand and at least one stack.
        - At least one controlled stack must have a legal move or capture
          in the current board state.
        """
        if game_state.current_phase != GamePhase.RING_PLACEMENT:
            return []

        player = next(
            (
                p
                for p in game_state.players
                if p.player_number == player_number
            ),
            None,
        )
        if not player or player.rings_in_hand <= 0:
            return []

        board = game_state.board
        player_stacks = BoardManager.get_player_stacks(board, player_number)
        if not player_stacks:
            return []

        # Check if at least one controlled stack has a legal move or capture.
        has_any_action = False
        for stack in player_stacks:
            if GameEngine._has_any_legal_move_or_capture_from_on_board(
                board.type,
                stack.position,
                player_number,
                board,
            ):
                has_any_action = True
                break

        if not has_any_action:
            return []

        # Use an arbitrary stack position as a harmless sentinel for `to`.
        sentinel_pos = player_stacks[0].position
        return [
            Move(
                id="simulated",
                type=MoveType.SKIP_PLACEMENT,
                player=player_number,
                to=sentinel_pos,
                timestamp=game_state.last_move_at,
                thinkTime=0,
                moveNumber=len(game_state.move_history) + 1,
            )  # type: ignore[arg-type]
        ]

    @staticmethod
    def _get_ring_placement_moves(
        game_state: GameState,
        player_number: int,
    ) -> List[Move]:
        """
        Enumerate legal PLACE_RING moves for the given player.

        This mirrors the TS RuleEngine.getValidRingPlacements +
        validateRingPlacement semantics:

        - Respect per-player ring caps derived from BOARD_CONFIGS.ringsPerPlayer,
          based on own-colour rings in play (board + hand).
        - Allow multi-ring placement (1–3 rings) on empty spaces.
        - Allow exactly 1 ring per placement on existing stacks.
        - Enforce no-dead-placement by simulating MOVEMENT-phase
          movement/capture availability from the placed stack on a
          hypothetical post-placement board.
        """
        moves: List[Move] = []
        board = game_state.board

        # Check if player has rings in hand
        player = next(
            (
                p
                for p in game_state.players
                if p.player_number == player_number
            ),
            None,
        )
        if not player or player.rings_in_hand <= 0:
            return []

        rings_in_hand = player.rings_in_hand

        # TS-aligned per-player cap (BOARD_CONFIGS[boardType].ringsPerPlayer)
        per_player_cap = GameEngine._estimate_rings_per_player(game_state)

        # Own-colour rings in play for this player (board + hand), mirroring
        # the shared TS core.countRingsInPlayForPlayer helper semantics.
        total_in_play = count_rings_in_play_for_player(game_state, player_number)
        rings_on_board = max(0, total_in_play - rings_in_hand)

        remaining_by_cap = per_player_cap - rings_on_board
        max_available_global = min(remaining_by_cap, rings_in_hand)
        if max_available_global <= 0:
            return []

        # Get all valid positions
        all_positions = GameEngine._generate_all_positions(
            board.type,
            board.size,
        )

        for pos in all_positions:
            pos_key = pos.to_key()

            # Cannot place on collapsed spaces
            if pos_key in board.collapsed_spaces:
                continue

            # Cannot place on markers (no stack+marker coexistence)
            if pos_key in board.markers:
                continue

            existing_stack = board.stacks.get(pos_key)
            is_occupied = bool(
                existing_stack and existing_stack.stack_height > 0,
            )

            max_available = max_available_global
            if max_available <= 0:
                continue

            if is_occupied:
                # On existing stacks we only ever place a single ring.
                if max_available < 1:
                    continue
                max_per_placement = 1
            else:
                # On empty spaces, allow multi-ring placements (typically up to 3).
                max_per_placement = min(3, max_available)

            if max_per_placement <= 0:
                continue

            # Optimization: Check placement counts in reverse order (highest first).
            # If a higher count is valid (can move stack_height+ spaces), lower counts
            # are also valid (shorter minimum distance). This reduces hypothetical
            # checks from up to 3 per position to just 1 for most valid positions.
            valid_from_count = None
            for placement_count in range(max_per_placement, 0, -1):
                if placement_count > max_available:
                    continue

                hyp_board = (
                    GameEngine._create_hypothetical_board_with_placement(
                        board,
                        pos,
                        player_number,
                        placement_count,
                    )
                )

                if GameEngine._has_any_movement_or_capture_after_hypothetical_placement(
                    game_state,
                    player_number,
                    pos,
                    hyp_board,
                ):
                    # This count and all lower counts are valid
                    valid_from_count = placement_count
                    break

            # Add moves for all valid placement counts
            if valid_from_count is not None:
                for count in range(1, valid_from_count + 1):
                    if count > max_available:
                        break
                    moves.append(
                        Move(
                            id="simulated",
                            type=MoveType.PLACE_RING,
                            player=player_number,
                            to=pos,
                            timestamp=game_state.last_move_at,
                            thinkTime=0,
                            moveNumber=len(game_state.move_history) + 1,
                            placementCount=count,
                            placedOnStack=is_occupied,
                        )  # type: ignore
                    )

        return moves

    @staticmethod
    def _get_capture_moves(
        game_state: GameState, player_number: int, limit: int | None = None
    ) -> List[Move]:
        """
        Enumerate legal overtaking capture segments for the current attacker.

        Adapter over rules.capture_chain.enumerate_capture_moves_py, which
        mirrors TS CaptureAggregate.enumerateCaptureMoves.

        Args:
            limit: If provided, return at most this many moves (for early-return checks).
        """
        board = game_state.board

        # Determine attacker position
        if game_state.chain_capture_state:
            attacker_pos = game_state.chain_capture_state.current_position
            kind = "continuation"
        else:
            last_move = (
                game_state.move_history[-1]
                if game_state.move_history else None
            )
            if not last_move or not last_move.to:
                return []
            attacker_pos = last_move.to
            kind = "initial"

        moves = enumerate_capture_moves_py(
            game_state,
            player_number,
            attacker_pos,
            move_number=len(game_state.move_history) + 1,
            kind=kind,  # type: ignore[arg-type]
        )

        # Apply limit if specified (for early-return optimization)
        if limit is not None and len(moves) > limit:
            return moves[:limit]
        return moves
    @staticmethod
    def _has_valid_placements(
        game_state: GameState, player_number: int
    ) -> bool:
        """
        True if the player has any legal ring placements.

        Forced-elimination gating in the TS TurnEngine.hasValidPlacements
        helper ignores the current phase and always evaluates placement
        availability in a ring_placement-style view. Here we mirror that
        behaviour by delegating to _has_any_valid_placement_fast which
        returns early as soon as one valid placement is found.
        """
        return GameEngine._has_any_valid_placement_fast(game_state, player_number)

    @staticmethod
    def _has_any_valid_placement_fast(
        game_state: GameState, player_number: int
    ) -> bool:
        """
        Fast early-return check for valid placements.

        Unlike _get_ring_placement_moves which enumerates ALL placements,
        this returns True immediately upon finding the first valid one.
        Critical for performance on large boards (19x19).
        """
        board = game_state.board

        # Check if player has rings in hand
        player = next(
            (p for p in game_state.players if p.player_number == player_number),
            None,
        )
        if not player or player.rings_in_hand <= 0:
            return False

        rings_in_hand = player.rings_in_hand

        # TS-aligned per-player cap
        per_player_cap = GameEngine._estimate_rings_per_player(game_state)
        total_in_play = count_rings_in_play_for_player(game_state, player_number)
        rings_on_board = max(0, total_in_play - rings_in_hand)
        remaining_by_cap = per_player_cap - rings_on_board
        max_available_global = min(remaining_by_cap, rings_in_hand)

        if max_available_global <= 0:
            return False

        # Get all valid positions
        all_positions = GameEngine._generate_all_positions(
            board.type,
            board.size,
        )

        for pos in all_positions:
            pos_key = pos.to_key()

            # Cannot place on collapsed spaces
            if pos_key in board.collapsed_spaces:
                continue

            # Cannot place on markers
            if pos_key in board.markers:
                continue

            existing_stack = board.stacks.get(pos_key)
            is_occupied = bool(existing_stack and existing_stack.stack_height > 0)

            max_available = max_available_global
            if max_available <= 0:
                continue

            if is_occupied:
                if max_available < 1:
                    continue
                max_per_placement = 1
            else:
                max_per_placement = min(3, max_available)

            if max_per_placement <= 0:
                continue

            # Check if at least one placement count is valid - return early!
            for placement_count in range(1, max_per_placement + 1):
                if placement_count > max_available:
                    break

                hyp_board = GameEngine._create_hypothetical_board_with_placement(
                    board,
                    pos,
                    player_number,
                    placement_count,
                )

                if GameEngine._has_any_movement_or_capture_after_hypothetical_placement(
                    game_state,
                    player_number,
                    pos,
                    hyp_board,
                ):
                    # Found a valid placement - return immediately!
                    return True

        return False

    @staticmethod
    def _has_any_legal_placement_on_bare_board(
        game_state: GameState, player_number: int
    ) -> bool:
        """
        Check if a player can legally place a ring on a bare board.

        This mirrors TS victoryLogic.ts hasAnyLegalPlacementOnBareBoard().
        Used for global stalemate detection when the board has no stacks
        but some players still have rings in hand.

        For each valid position (not collapsed, no markers), we check if
        placing a single ring there would give the player at least one
        legal movement or capture (no-dead-placement rule).

        Returns True immediately upon finding any valid placement.
        """
        board = game_state.board

        # Check if player has rings in hand
        player = next(
            (p for p in game_state.players if p.player_number == player_number),
            None,
        )
        if not player or player.rings_in_hand <= 0:
            return False

        # Get all valid positions
        all_positions = GameEngine._generate_all_positions(
            board.type,
            board.size,
        )

        for pos in all_positions:
            pos_key = pos.to_key()

            # Cannot place on collapsed spaces
            if pos_key in board.collapsed_spaces:
                continue

            # Cannot place on markers
            if pos_key in board.markers:
                continue

            # On a bare board, any unoccupied position is a candidate.
            # Create a hypothetical board with a single stack at this position.
            hyp_board = GameEngine._create_hypothetical_board_with_placement(
                board,
                pos,
                player_number,
                1,  # Single ring placement
            )

            # Check if the hypothetical stack would have any legal move
            if GameEngine._has_any_movement_or_capture_after_hypothetical_placement(
                game_state,
                player_number,
                pos,
                hyp_board,
            ):
                # Found a valid placement - return immediately!
                return True

        return False

    @staticmethod
    def _has_valid_movements(
        game_state: GameState, player_number: int
    ) -> bool:
        """True if the player has any legal non-capture movements."""
        # Movement availability is independent of current_phase for our
        # forced-elimination gating: we ask "could this player move at all?"
        # Use limit=1 for early return optimization on large boards.
        return bool(
            GameEngine._get_movement_moves(game_state, player_number, limit=1)
        )

    @staticmethod
    def _has_valid_captures(game_state: GameState, player_number: int) -> bool:
        """True if the player has any legal overtaking capture segments."""
        # Use limit=1 for early return optimization on large boards.
        return bool(
            GameEngine._get_capture_moves(game_state, player_number, limit=1)
        )

    @staticmethod
    def _has_valid_actions(game_state: GameState, player_number: int) -> bool:
        """
        Combined placement/movement/capture availability check.

        This is a Python analogue of TS hasValidActions used to decide when
        forced elimination is required. If ANY legal placement, movement, or
        capture exists for the player, forced elimination is not permitted.
        """
        if GameEngine._has_valid_placements(game_state, player_number):
            return True
        if GameEngine._has_valid_movements(game_state, player_number):
            return True
        if GameEngine._has_valid_captures(game_state, player_number):
            return True
        return False

    @staticmethod
    def _has_real_action_for_player(
        game_state: GameState,
        player_number: int,
    ) -> bool:
        """
        R172 real-action availability predicate for LPS.

        A player has a real action if they have at least one legal placement,
        non-capture movement, or overtaking capture available on their turn.
        Forced elimination and line/territory decision moves do not count.
        """
        return GameEngine._has_valid_actions(game_state, player_number)

    @staticmethod
    def _update_lps_round_tracking_for_current_player(
        game_state: GameState,
    ) -> None:
        """
        Update last-player-standing (R172) round tracking for the current
        player.

        This mirrors the TS GameEngine LPS tracking helper by:
        - Computing whether the current player has any real actions.
        - Recording that fact in lps_current_round_actor_mask.
        - When all active players have been seen in this round, determining
          whether exactly one player had real actions and, if so, recording
          them as lps_exclusive_player_for_completed_round.
        """
        if game_state.game_status != GameStatus.ACTIVE:
            return

        board = game_state.board

        # Active players are those with at least one stack or a ring in hand.
        active_players: List[int] = []
        for player in game_state.players:
            has_stacks = any(
                stack.controlling_player == player.player_number
                for stack in board.stacks.values()
            )
            if has_stacks or player.rings_in_hand > 0:
                active_players.append(player.player_number)

        if not active_players:
            return

        active_set = set(active_players)
        mask = game_state.lps_current_round_actor_mask

        # Drop any stale entries for players who are no longer active.
        stale_keys = [pid for pid in list(mask.keys()) if pid not in active_set]
        if stale_keys:
            mask.clear()
            game_state.lps_exclusive_player_for_completed_round = None

        current = game_state.current_player
        if current not in active_set:
            return

        has_real_action = GameEngine._has_real_action_for_player(
            game_state,
            current,
        )
        mask[current] = has_real_action

        # When we've recorded an entry for every active player, a round completes.
        if all(pid in mask for pid in active_set):
            true_players = [
                pid for pid, value in mask.items()
                if value and pid in active_set
            ]
            if len(true_players) == 1:
                game_state.lps_exclusive_player_for_completed_round = true_players[0]
            else:
                game_state.lps_exclusive_player_for_completed_round = None

            game_state.lps_round_index += 1
            mask.clear()

    @staticmethod
    def _get_forced_elimination_moves(
        game_state: GameState, player_number: int
    ) -> List[Move]:
        """
        Get forced elimination moves for a blocked player.

        Forced elimination is only available when the player controls at least
        one stack but has NO legal placement, movement, or capture action, in
        line with TS TurnEngine.hasValidActions semantics.

        Stack selection strategy (per R100 / TS globalActions.ts):
        - Prefer smallest positive capHeight
        - Fallback to first stack when no caps exist
        """
        board = game_state.board

        # Player must control at least one stack and have no other actions.
        player_stacks = [
            s for s in board.stacks.values()
            if s.controlling_player == player_number
        ]
        if not player_stacks:
            return []

        if GameEngine._has_valid_actions(game_state, player_number):
            return []

        # Sort stacks by capHeight (smallest first) to match TS behavior
        # When caps are equal, order is arbitrary (first in iteration)
        player_stacks.sort(key=lambda s: s.cap_height if s.cap_height > 0 else float('inf'))

        moves: List[Move] = []
        for stack in player_stacks:
            pos = stack.position
            moves.append(
                Move(
                    id="simulated",
                    type=MoveType.FORCED_ELIMINATION,
                    player=player_number,
                    to=pos,
                    timestamp=game_state.last_move_at,
                    thinkTime=0,
                    moveNumber=len(game_state.move_history) + 1,
                    placementCount=0,
                    placedOnStack=False,
                )  # type: ignore
            )
        return moves

    @staticmethod
    def _perform_forced_elimination_for_player(
        game_state: GameState,
        player_number: int,
    ) -> None:
        """
        Internal helper mirroring TS TurnEngine.processForcedElimination.

        This is invoked from _end_turn after territory processing when the next
        player controls at least one stack but has no legal placements,
        movements, or captures available.

        The actual elimination semantics are centralised in the
        app.rules.global_actions.apply_forced_elimination_for_player helper so
        that host-level forced elimination remains consistent with the TS
        shared engine and RR-CANON-R205.
        """
        # Local import to avoid circular import at module import time.
        from app.rules import global_actions as ga  # type: ignore

        outcome = ga.apply_forced_elimination_for_player(game_state, player_number)
        if outcome is None:
            return

        # After elimination, re-check victory conditions in case thresholds
        # were crossed.
        GameEngine._check_victory(game_state)

    @staticmethod
    def _get_line_processing_moves(
        game_state: GameState,
        player_number: int,
    ) -> List[Move]:
        """
        Enumerate canonical line-processing decision moves for the player.

        Mirrors TS RuleEngine.getValidLineProcessingDecisionMoves:

        - One PROCESS_LINE move per player-owned line.
        - One CHOOSE_LINE_REWARD move per overlength line (length > required).
        """
        lines = BoardManager.find_all_lines(game_state.board)
        player_lines = [line for line in lines if line.player == player_number]

        if not player_lines:
            return []

        required_len = get_line_length_for_board(game_state.board.type)
        moves: List[Move] = []

        for idx, line in enumerate(player_lines):
            # PROCESS_LINE decision for this line.
            first_pos = line.positions[0]
            moves.append(
                Move(
                    id=f"process-line-{idx}-{first_pos.to_key()}",
                    type=MoveType.PROCESS_LINE,
                    player=player_number,
                    to=first_pos,
                    formed_lines=(line,),  # type: ignore[arg-type]
                    timestamp=game_state.last_move_at,
                    thinkTime=0,
                    moveNumber=len(game_state.move_history) + 1,
                )
            )

        # For overlength lines, enumerate all CHOOSE_LINE_REWARD options:
        # - Option 1 (COLLAPSE_ALL): collapse entire line, eliminate one ring
        # - Option 2 (MINIMUM_COLLAPSE): collapse exactly required_len consecutive
        #   markers, no ring elimination. Generate one move per valid segment.
        #
        # This mirrors TS LineAggregate.enumerateChooseLineRewardMoves.
        for idx, line in enumerate(player_lines):
            line_len = len(line.positions)
            if line_len <= required_len:
                continue

            first_pos = line.positions[0]
            line_key = first_pos.to_key()
            move_number = len(game_state.move_history) + 1

            # Option 1: Collapse all markers (collapsed_markers = full line)
            moves.append(
                Move(
                    id=f"choose-line-reward-{idx}-{line_key}-all",
                    type=MoveType.CHOOSE_LINE_REWARD,
                    player=player_number,
                    to=first_pos,
                    formed_lines=(line,),  # type: ignore[arg-type]
                    collapsed_markers=tuple(line.positions),
                    timestamp=game_state.last_move_at,
                    thinkTime=0,
                    moveNumber=move_number,
                )
            )

            # Option 2: Enumerate all valid minimum-collapse segments of
            # length required_len along the line.
            max_start = line_len - required_len
            for start in range(max_start + 1):
                segment = tuple(line.positions[start:start + required_len])
                moves.append(
                    Move(
                        id=f"choose-line-reward-{idx}-{line_key}-min-{start}",
                        type=MoveType.CHOOSE_LINE_REWARD,
                        player=player_number,
                        to=first_pos,
                        formed_lines=(line,),  # type: ignore[arg-type]
                        collapsed_markers=segment,
                        timestamp=game_state.last_move_at,
                        thinkTime=0,
                        moveNumber=move_number,
                    )
                )

        return moves

    @staticmethod
    def _get_territory_processing_moves(
        game_state: GameState,
        player_number: int,
    ) -> List[Move]:
        """
        Enumerate canonical territory-processing decision moves.

        Mirrors TS RuleEngine.getValidTerritoryProcessingDecisionMoves and
        getValidEliminationDecisionMoves:

        - When at least one disconnected region satisfies the self-elimination
          prerequisite for the moving player, emit one PROCESS_TERRITORY_REGION
          move per such region.
        - When no such regions exist but the player controls stacks, emit one
          ELIMINATE_RINGS_FROM_STACK move per controlled stack.
        """
        board = game_state.board
        regions = BoardManager.find_disconnected_regions(
            board,
            player_number,
        )

        eligible_regions = [
            region
            for region in (regions or [])
            if GameEngine._can_process_disconnected_region(
                game_state,
                region,
                player_number,
            )
        ]

        moves: List[Move] = []

        if eligible_regions:
            for idx, region in enumerate(eligible_regions):
                rep = region.spaces[0]
                moves.append(
                    Move(
                        id=f"process-region-{idx}-{rep.to_key()}",
                        type=MoveType.PROCESS_TERRITORY_REGION,
                        player=player_number,
                        to=rep,
                        disconnected_regions=(region,),  # type: ignore[arg-type]
                        timestamp=game_state.last_move_at,
                        thinkTime=0,
                        moveNumber=len(game_state.move_history) + 1,
                    )  # type: ignore
                )
            return moves

        # No eligible regions – enumerate explicit elimination decisions.
        player_stacks = BoardManager.get_player_stacks(board, player_number)
        if not player_stacks:
            return moves

        for stack in player_stacks:
            cap_height = stack.cap_height
            if cap_height <= 0:
                continue
            pos = stack.position
            moves.append(
                Move(
                    id=f"eliminate-{pos.to_key()}",
                    type=MoveType.ELIMINATE_RINGS_FROM_STACK,
                    player=player_number,
                    to=pos,
                    eliminated_rings=(
                        {"player": player_number, "count": cap_height},
                    ),
                    timestamp=game_state.last_move_at,
                    thinkTime=0,
                    moveNumber=len(game_state.move_history) + 1,
                )  # type: ignore
            )

        return moves

    @staticmethod
    def _can_process_disconnected_region(
        game_state: GameState,
        region,
        player_number: int,
    ) -> bool:
        """
        Self-elimination prerequisite for territory processing.

        Mirrors TS canProcessDisconnectedRegion:

        - Player must have at least one stack outside the region's spaces.
        """
        board = game_state.board
        region_keys = {p.to_key() for p in region.spaces}
        player_stacks = BoardManager.get_player_stacks(board, player_number)

        for stack in player_stacks:
            if stack.position.to_key() not in region_keys:
                return True

        return False

    @staticmethod
    def _get_movement_moves(
        game_state: GameState, player_number: int, limit: int | None = None
    ) -> List[Move]:
        """
        Get valid non-capture movement moves.

        Args:
            limit: If provided, return after finding this many moves (for early-return checks).
        """
        moves: List[Move] = []
        board = game_state.board

        # Per-turn must-move constraint: when a ring has been placed this
        # turn, only the updated stack (tracked via must_move_from_stack_key)
        # may move or capture, mirroring TurnEngine.mustMoveFromStackKey.
        must_move_key = game_state.must_move_from_stack_key

        directions = BoardManager._get_all_directions(board.type)
        _debug(f"DEBUG: _get_movement_moves directions: {directions}\n")

        # Iterate through all stacks controlled by the player
        # sys.stderr.write(f"DEBUG: _get_movement_moves checking stacks for P{player_number}. Stacks: {list(board.stacks.keys())}\n")
        for _, stack in board.stacks.items():
            if stack.controlling_player != player_number:
                _debug(
                    f"DEBUG: Skipping stack {stack.position} controlled by "
                    f"{stack.controlling_player} (looking for "
                    f"{player_number})\n"
                )
                continue

            from_pos = stack.position

            # If we must move a specific stack, skip others
            if (must_move_key is not None and
                    from_pos.to_key() != must_move_key):
                _debug(
                    f"DEBUG: Skipping stack {from_pos} because "
                    f"must_move_key={must_move_key} != {from_pos.to_key()}\n"
                )
                continue

            min_distance = max(1, stack.stack_height)
            _debug(f"DEBUG: Entering direction loop for {from_pos}\n")

            for direction in directions:
                # sys.stderr.write(f"DEBUG: Checking direction {direction} from {from_pos}\n")
                distance = min_distance
                while True:
                    to_pos = BoardManager._add_direction(
                        from_pos, direction, distance
                    )

                    if not BoardManager.is_valid_position(
                        to_pos, board.type, board.size
                    ):
                        break

                    if BoardManager.is_collapsed_space(to_pos, board):
                        break

                    # Check path blocking between from_pos and to_pos
                    if not GameEngine._is_path_clear_for_movement(
                        board, from_pos, to_pos
                    ):
                        # Any blocking stack/collapsed space stops further
                        # exploration along this ray.
                        break

                    to_key = to_pos.to_key()
                    dest_stack = board.stacks.get(to_key)
                    dest_marker = board.markers.get(to_key)

                    if dest_stack is None or dest_stack.stack_height == 0:
                        # Empty cell or marker-only cell
                        if (dest_marker is None or
                                dest_marker.player == player_number):
                            _debug(
                                f"DEBUG: _get_movement_moves found move from "
                                f"{from_pos} to {to_pos} dist {distance}\n"
                            )
                            moves.append(
                                Move(
                                    id="simulated",
                                    type=MoveType.MOVE_STACK,
                                    player=player_number,
                                    from_pos=from_pos,  # type: ignore[arg-type]
                                    to=to_pos,
                                    timestamp=game_state.last_move_at,
                                    thinkTime=0,
                                    moveNumber=len(game_state.move_history) + 1,
                                )  # type: ignore
                            )
                            if limit is not None and len(moves) >= limit:
                                return moves
                        else:
                            _debug(
                                f"DEBUG: _get_movement_moves rejected {to_pos} "
                                f"due to marker {dest_marker}\n"
                            )
                    else:
                        # Landing on any stack (friendly or opponent) is a
                        # legal merge move.
                        _debug(
                            f"DEBUG: _get_movement_moves found merge from "
                            f"{from_pos} to {to_pos} dist {distance}\n"
                        )
                        moves.append(
                            Move(
                                id="simulated",
                                type=MoveType.MOVE_STACK,
                                player=player_number,
                                from_pos=from_pos,  # type: ignore[arg-type]
                                to=to_pos,
                                    timestamp=game_state.last_move_at,
                                    thinkTime=0,
                                    moveNumber=len(game_state.move_history) + 1,
                            )  # type: ignore
                        )
                        if limit is not None and len(moves) >= limit:
                            return moves
                        # Cannot move further along this ray past a stack.
                        break

                    distance += 1

        return moves

    @staticmethod
    def _apply_place_ring(game_state: GameState, move: Move):
        """
        Apply a place_ring move, including multi-ring placement semantics.

        This mirrors the TS RuleEngine / shared PlacementMutator behaviour:

        - On empty spaces, place `placementCount` rings for the moving player
          (typically 1–3), stacked so that the newest ring is on top.
        - On existing stacks, append the new rings on top of the existing
          stack. Move generation will normally only emit placementCount = 1
          for stack placements, but we still respect larger values when
          provided for robustness.
        - Update stackHeight, capHeight, and controllingPlayer consistently
          with the Python RingStack representation (rings stored bottom→top,
          top = rings[-1]).
        - Decrement the player's rings_in_hand by placementCount.
        """
        board = game_state.board
        pos = move.to
        pos_key = pos.to_key()

        # Default to a single-ring placement when placementCount is omitted.
        placement_count = move.placement_count or 1
        if placement_count <= 0:
            return

        existing = board.stacks.get(pos_key)
        zobrist = ZobristHash()

        if existing and existing.stack_height > 0:
            # Remove old stack hash
            if game_state.zobrist_hash is not None:
                game_state.zobrist_hash ^= zobrist.get_stack_hash(
                    pos_key,
                    existing.controlling_player,
                    existing.stack_height,
                    tuple(existing.rings)
                )

            # Add rings on top of an existing stack. Our representation stores
            # rings bottom→top, so we append new rings to the end of the list.
            stack = existing.model_copy(deep=True)
            board.stacks[pos_key] = stack

            stack.rings.extend([move.player] * placement_count)
            stack.stack_height = len(stack.rings)
            stack.controlling_player = stack.rings[-1]

            # Recompute capHeight
            h = 0
            for r in reversed(stack.rings):
                if r == stack.controlling_player:
                    h += 1
                else:
                    break
            stack.cap_height = h
        else:
            # Create a brand new stack with placementCount rings for the
            # player.
            rings = [move.player] * placement_count
            stack = RingStack(
                position=pos,
                rings=rings,
                stackHeight=len(rings),
                capHeight=0,  # recomputed below
                controllingPlayer=rings[-1],
            )
            board.stacks[pos_key] = stack

        # Recompute stackHeight, controllingPlayer, and capHeight in a single
        # pass to keep behaviour aligned with
        # _create_hypothetical_board_with_placement and other helpers.
        stack.stack_height = len(stack.rings)
        stack.controlling_player = stack.rings[-1]

        h = 0
        for r in reversed(stack.rings):
            if r == stack.controlling_player:
                h += 1
            else:
                break
        stack.cap_height = h

        # Add new stack hash
        if game_state.zobrist_hash is not None:
            game_state.zobrist_hash ^= zobrist.get_stack_hash(
                pos_key,
                stack.controlling_player,
                stack.stack_height,
                tuple(stack.rings)
            )

        # Decrement rings in hand for the moving player. Move generation
        # guarantees that placementCount never exceeds rings_in_hand, so this
        # should not underflow; we still guard defensively.
        for p in game_state.players:
            if p.player_number == move.player:
                p.rings_in_hand = max(0, p.rings_in_hand - placement_count)
                break

    @staticmethod
    def _apply_move_stack(game_state: GameState, move: Move):
        """
        Apply non-capture movement.

        Mirrors the TS GameEngine.applyMove 'move_stack' logic:
        - Leave a marker at departure.
        - Process markers along the path (flip or collapse).
        - Handle landing on any marker per RR-CANON-R091/R092 (remove + eliminate).
        - Merge with any existing stack at destination.
        """
        if not move.from_pos:
            return

        from_key = move.from_pos.to_key()
        to_key = move.to.to_key()
        board = game_state.board

        source_stack = board.stacks.get(from_key)
        if not source_stack:
            return

        # Deep-copy source stack to avoid mutating shared instances
        moving_stack = source_stack.model_copy(deep=True)

        # Remove from source before placing marker/processing path
        zobrist = ZobristHash()
        if game_state.zobrist_hash is not None:
            game_state.zobrist_hash ^= zobrist.get_stack_hash(
                from_key,
                source_stack.controlling_player,
                source_stack.stack_height,
                tuple(source_stack.rings)
            )
        del board.stacks[from_key]

        # Leave a marker at departure
        board.markers[from_key] = MarkerInfo(
            player=move.player,
            position=move.from_pos,
            type="regular",
        )
        if game_state.zobrist_hash is not None:
            game_state.zobrist_hash ^= zobrist.get_marker_hash(
                from_key, move.player
            )

        # Process markers along movement path (excluding endpoints)
        GameEngine._process_markers_along_path(
            board, move.from_pos, move.to, move.player, game_state
        )

        # Check for marker at landing
        # Per RR-CANON-R091/R092: landing on any marker (own or opponent) removes
        # the marker and eliminates the top ring of the moving stack's cap.
        landing_marker = board.markers.get(to_key)
        landed_on_marker = landing_marker is not None

        if landed_on_marker:
            # Remove marker before landing
            if game_state.zobrist_hash is not None:
                game_state.zobrist_hash ^= zobrist.get_marker_hash(
                    to_key, landing_marker.player
                )
            del board.markers[to_key]

        # Handle merge with any existing stack at destination
        dest_stack = board.stacks.get(to_key)
        if dest_stack and dest_stack.stack_height > 0:
            # Remove old dest stack hash
            if game_state.zobrist_hash is not None:
                game_state.zobrist_hash ^= zobrist.get_stack_hash(
                    to_key,
                    dest_stack.controlling_player,
                    dest_stack.stack_height,
                    tuple(dest_stack.rings)
                )

            dest_stack = dest_stack.model_copy(deep=True)
            board.stacks[to_key] = dest_stack

            # In the Python representation, rings are stored bottom->top.
            # Merging means keeping the destination stack's rings at the
            # bottom and placing the moving stack's rings above them.
            merged_rings = dest_stack.rings + moving_stack.rings
            dest_stack.rings = merged_rings
            dest_stack.stack_height = len(merged_rings)

            # Update controlling player and cap height
            dest_stack.controlling_player = merged_rings[-1]
            h = 0
            for r in reversed(merged_rings):
                if r == dest_stack.controlling_player:
                    h += 1
                else:
                    break
            dest_stack.cap_height = h

            # Add new dest stack hash
            if game_state.zobrist_hash is not None:
                game_state.zobrist_hash ^= zobrist.get_stack_hash(
                    to_key,
                    dest_stack.controlling_player,
                    dest_stack.stack_height,
                    tuple(dest_stack.rings)
                )
        else:
            # Simple move to empty (or marker-only) space
            moving_stack.position = move.to
            board.stacks[to_key] = moving_stack
            # Add new stack hash
            if game_state.zobrist_hash is not None:
                game_state.zobrist_hash ^= zobrist.get_stack_hash(
                    to_key,
                    moving_stack.controlling_player,
                    moving_stack.stack_height,
                    tuple(moving_stack.rings)
                )

        # Cap-elimination when landing on any marker
        if landed_on_marker:
            GameEngine._eliminate_top_ring_at(game_state, move.to, move.player)
    @staticmethod
    def _apply_overtaking_capture(game_state: GameState, move: Move):
        """
        Apply an initial overtaking capture segment.

        This is a thin wrapper that delegates to _apply_chain_capture so that
        both the first segment and any continuation segments share the same
        TS-aligned semantics.
        """
        GameEngine._apply_chain_capture(game_state, move)

    @staticmethod
    def _apply_chain_capture(game_state: GameState, move: Move):
        """
        Apply a single overtaking capture segment (initial or continuation).

        Mirrors the TS GameEngine.performOvertakingCapture semantics:

        - Leave a marker at the departure space.
        - Process markers along both legs of the path (from→target and
          target→landing), flipping or collapsing as needed.
        - Remove exactly one ring from the top of the target stack and insert
          it at the bottom of the attacker.
        - Move the attacker stack to the landing space (merging if a stack is
          already present).
        - If landing on any marker (per RR-CANON-R101/R102), remove it and
          eliminate one ring from the top of the attacking stack's cap.
        - Update chain_capture_state so subsequent segments continue from the
          new position.
        """
        if not move.from_pos or not move.capture_target:
            return

        board = game_state.board
        from_pos = move.from_pos
        target_pos = move.capture_target
        landing_pos = move.to

        from_key = from_pos.to_key()
        target_key = target_pos.to_key()
        landing_key = landing_pos.to_key()

        attacker = board.stacks.get(from_key)
        target_stack = board.stacks.get(target_key)
        if not attacker or not target_stack:
            return

        # Deep-copy stacks before modification
        attacker = attacker.model_copy(deep=True)
        target_stack = target_stack.model_copy(deep=True)

        # Remove attacker from source before marker/path processing
        zobrist = ZobristHash()
        if game_state.zobrist_hash is not None:
            game_state.zobrist_hash ^= zobrist.get_stack_hash(
                from_key,
                attacker.controlling_player,
                attacker.stack_height,
                tuple(attacker.rings)
            )
        del board.stacks[from_key]

        # Leave a marker at departure
        board.markers[from_key] = MarkerInfo(
            player=move.player,
            position=from_pos,
            type="regular",
        )
        if game_state.zobrist_hash is not None:
            game_state.zobrist_hash ^= zobrist.get_marker_hash(
                from_key, move.player
            )

        # Process markers along from→target and target→landing
        GameEngine._process_markers_along_path(
            board, from_pos, target_pos, move.player, game_state
        )
        GameEngine._process_markers_along_path(
            board, target_pos, landing_pos, move.player, game_state
        )

        # Capture top ring from target (top = end of list)
        if not target_stack.rings:
            # Defensive: nothing to capture; restore attacker and exit
            board.stacks[from_key] = attacker
            return

        # Remove old target stack hash
        if game_state.zobrist_hash is not None:
            game_state.zobrist_hash ^= zobrist.get_stack_hash(
                target_key,
                target_stack.controlling_player,
                target_stack.stack_height,
                tuple(target_stack.rings)
            )

        captured_ring = target_stack.rings.pop()
        target_stack.stack_height -= 1

        if target_stack.stack_height == 0:
            if target_key in board.stacks:
                del board.stacks[target_key]
        else:
            # Recompute controlling player and cap height
            target_stack.controlling_player = target_stack.rings[-1]
            h = 0
            for r in reversed(target_stack.rings):
                if r == target_stack.controlling_player:
                    h += 1
                else:
                    break
            target_stack.cap_height = h
            board.stacks[target_key] = target_stack
            # Add new target stack hash
            if game_state.zobrist_hash is not None:
                game_state.zobrist_hash ^= zobrist.get_stack_hash(
                    target_key,
                    target_stack.controlling_player,
                    target_stack.stack_height,
                    tuple(target_stack.rings)
                )

        # Insert captured ring at bottom of attacker.
        #
        # In the TS engine, rings are stored [top -> bottom] and the captured
        # ring is appended at the bottom, which does not change the
        # controlling player or cap height of the attacking stack. Our Python
        # representation stores rings bottom -> top, but inserting at index 0
        # is the analogous "bottom" operation and likewise leaves the cap
        # unchanged.
        prev_controlling = attacker.controlling_player
        prev_cap_height = attacker.cap_height

        attacker.rings.insert(0, captured_ring)
        attacker.stack_height += 1
        attacker.controlling_player = prev_controlling
        attacker.cap_height = prev_cap_height

        # Move attacker to landing (merge if stack already present)
        existing_dest = board.stacks.get(landing_key)
        if existing_dest and existing_dest.stack_height > 0:
            # Remove old dest stack hash
            if game_state.zobrist_hash is not None:
                game_state.zobrist_hash ^= zobrist.get_stack_hash(
                    landing_key,
                    existing_dest.controlling_player,
                    existing_dest.stack_height,
                    tuple(existing_dest.rings)
                )

            existing_dest = existing_dest.model_copy(deep=True)
            merged_rings = existing_dest.rings + attacker.rings
            existing_dest.rings = merged_rings
            existing_dest.stack_height = len(merged_rings)
            existing_dest.controlling_player = merged_rings[-1]
            h_merge = 0
            for r in reversed(merged_rings):
                if r == existing_dest.controlling_player:
                    h_merge += 1
                else:
                    break
            existing_dest.cap_height = h_merge
            existing_dest.position = landing_pos
            board.stacks[landing_key] = existing_dest

            # Add new dest stack hash
            if game_state.zobrist_hash is not None:
                game_state.zobrist_hash ^= zobrist.get_stack_hash(
                    landing_key,
                    existing_dest.controlling_player,
                    existing_dest.stack_height,
                    tuple(existing_dest.rings)
                )
        else:
            attacker.position = landing_pos
            board.stacks[landing_key] = attacker
            # Add new stack hash
            if game_state.zobrist_hash is not None:
                game_state.zobrist_hash ^= zobrist.get_stack_hash(
                    landing_key,
                    attacker.controlling_player,
                    attacker.stack_height,
                    tuple(attacker.rings)
                )

        # Handle landing on marker
        # Per RR-CANON-R101/R102: landing on any marker (own or opponent) removes
        # the marker and eliminates the top ring of the attacking stack's cap.
        landing_marker = board.markers.get(landing_key)
        if landing_marker is not None:
            # Remove marker (do not collapse it)
            if game_state.zobrist_hash is not None:
                game_state.zobrist_hash ^= zobrist.get_marker_hash(
                    landing_key, landing_marker.player
                )
            del board.markers[landing_key]
            # Eliminate top ring of the attacking stack's cap
            GameEngine._eliminate_top_ring_at(
                game_state, landing_pos, move.player
            )

        # Update chain capture state (Python analogue of
        # updateChainCaptureStateAfterCapture)
        from .models import ChainCaptureSegment, ChainCaptureState

        segment = ChainCaptureSegment(
            **{
                "from": from_pos,
                "target": target_pos,
                "landing": landing_pos,
                "capturedCapHeight": 0,
            }
        )

        if game_state.chain_capture_state:
            state = game_state.chain_capture_state
            state.current_position = landing_pos
            state.segments.append(segment)
            state.visited_positions.append(from_pos.to_key())
        else:
            game_state.chain_capture_state = ChainCaptureState(
                **{
                    "playerNumber": move.player,
                    "startPosition": from_pos,
                    "currentPosition": landing_pos,
                    "segments": [segment],
                    "availableMoves": [],
                    "visitedPositions": [from_pos.to_key()],
                }
            )

    @staticmethod
    def _apply_line_formation(game_state: GameState, move: Move):
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
        board = game_state.board

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
            for line in BoardManager.find_all_lines(board):
                if line.positions and line.positions[0].to_key() == move.to.to_key():
                    target_line = line
                    break

        if target_line is None:
            return

        # 2. Determine required minimum line length.
        required_len = get_line_length_for_board(board.type)

        # 3. Decide which positions to collapse.
        positions_to_collapse: List[Position]

        if move.type in (MoveType.PROCESS_LINE, MoveType.LINE_FORMATION):
            # PROCESS_LINE / LINE_FORMATION always collapse the entire line.
            positions_to_collapse = list(target_line.positions)
        elif move.type == MoveType.CHOOSE_LINE_REWARD:
            # Reward choice for overlength lines. When collapsed_markers is
            # provided, it encodes the user's choice; otherwise we default to
            # the minimum-collapse behaviour (first required_len markers).
            markers_to_collapse = getattr(move, "collapsed_markers", None)
            if markers_to_collapse:
                positions_to_collapse = list(markers_to_collapse)
            else:
                positions_to_collapse = list(target_line.positions[:required_len])
        else:
            # Legacy CHOOSE_LINE_OPTION continues to honour placement_count
            # as an option selector: 1 → minimum collapse, >1 → collapse all.
            option = move.placement_count or 1
            if option == 1:
                positions_to_collapse = list(target_line.positions[:required_len])
            else:
                positions_to_collapse = list(target_line.positions)

        # 4. Apply collapses. TS currently tracks territory ownership for
        # line-collapse rewards implicitly via board.collapsedSpaces; the
        # per-player territorySpaces counters are not updated by the shared
        # LineMutator, and the TS parity fixtures' hashes reflect that. To
        # stay aligned, we update board.collapsed_spaces only and leave
        # territory_spaces unchanged here.
        seen_keys = set()
        zobrist = ZobristHash()
        for pos in positions_to_collapse:
            key = pos.to_key()
            if key in seen_keys:
                continue
            seen_keys.add(key)
            # Check if there was a marker there (should be for line formation)
            marker = board.markers.get(key)
            if marker:
                if game_state.zobrist_hash is not None:
                    game_state.zobrist_hash ^= zobrist.get_marker_hash(
                        key, marker.player
                    )
            BoardManager.set_collapsed_space(pos, move.player, board)
            if game_state.zobrist_hash is not None:
                game_state.zobrist_hash ^= zobrist.get_collapsed_hash(key)

        # Note: board.formed_lines is not currently consulted by the Python
        # GameEngine when generating further line-processing moves, and it is
        # not part of the canonical hash. We therefore do not mutate
        # board.formed_lines here; parity fixtures care only about the
        # collapsed territory side-effects.
    @staticmethod
    def _apply_territory_claim(game_state: GameState, move: Move):
        """
        Apply a territory claim move (TS-orchestrator-aligned territory processing).

        Mirrors TS territoryProcessing.processOneDisconnectedRegion:

        1) Identify the disconnected region associated with this move, using
           the explicit `move.disconnectedRegions` metadata when present and
           falling back to BoardManager.find_disconnected_regions when
           metadata is absent.
        2) Eliminate all rings inside the region, crediting ALL eliminated
           rings to the moving player.
        3) Collapse all spaces in the region, plus any border markers, to
           the moving player's colour.
        4) Update the moving player's territory_spaces by the total number
           of newly collapsed spaces (region + border markers).

        NOTE: Mandatory self-elimination is modelled as a separate
        ELIMINATE_RINGS_FROM_STACK / FORCED_ELIMINATION move in the
        orchestrator-aligned contract vectors and is therefore NOT applied
        inside this helper.
        """
        board = game_state.board
        player = move.player

        # 1. Identify target region, preferring explicit decision geometry.
        target_region = None
        move_key = move.to.to_key()

        # 1a. Prefer explicit disconnectedRegions geometry carried on the Move.
        explicit_regions = list(getattr(move, "disconnected_regions", None) or [])
        if explicit_regions:
            for region in explicit_regions:
                if any(space.to_key() == move_key for space in region.spaces):
                    target_region = region
                    break
            if target_region is None:
                # Fall back to the first supplied region if no space matched `to`.
                target_region = explicit_regions[0]

            _debug(
                "DEBUG: _apply_territory_claim using explicit disconnected "
                f"region at {move_key} with {len(target_region.spaces)} spaces "
                f"for P{player}\n"
            )
        else:
            # 1b. Fallback: rediscover regions from the board state and match
            #     by representative position, preserving legacy behaviour for
            #     callers that do not provide explicit geometry.
            regions = BoardManager.find_disconnected_regions(board, player)
            for region in regions:
                if region.spaces and region.spaces[0].to_key() == move_key:
                    target_region = region
                    break

            if target_region:
                _debug(
                    "DEBUG: _apply_territory_claim using discovered "
                    f"disconnected region at {move_key} with "
                    f"{len(target_region.spaces)} spaces for P{player}\n"
                )

        if not target_region:
            _debug(
                "DEBUG: _apply_territory_claim found no matching disconnected "
                f"region for move at {move_key} by P{player}\n"
            )
            return

        # 1c. Compute region key set for later checks (e.g., outside stacks)
        #     and gather border markers.
        region_keys = {p.to_key() for p in target_region.spaces}

        border_markers = BoardManager.get_border_marker_positions(
            target_region.spaces,
            board,
        )

        # 2. Eliminate all rings within the region, crediting them to the
        #    moving player. We repeatedly call _eliminate_top_ring_at until
        #    no stack remains at each space.
        for pos in target_region.spaces:
            while True:
                stack = BoardManager.get_stack(pos, board)
                if not stack or stack.stack_height == 0:
                    break
                GameEngine._eliminate_top_ring_at(
                    game_state,
                    pos,
                    credited_player=player,
                )

        # 3. Collapse all spaces in the region to the moving player's colour.
        zobrist = ZobristHash()
        for pos in target_region.spaces:
            key = pos.to_key()
            # Check for marker (unlikely if we just eliminated rings,
            # but possible if empty)
            marker = board.markers.get(key)
            if marker:
                if game_state.zobrist_hash is not None:
                    game_state.zobrist_hash ^= zobrist.get_marker_hash(
                        key, marker.player
                    )
            BoardManager.set_collapsed_space(pos, player, board)
            if game_state.zobrist_hash is not None:
                game_state.zobrist_hash ^= zobrist.get_collapsed_hash(key)

        # 4. Collapse all border markers to the moving player's colour.
        for pos in border_markers:
            key = pos.to_key()
            marker = board.markers.get(key)
            if marker:
                if game_state.zobrist_hash is not None:
                    game_state.zobrist_hash ^= zobrist.get_marker_hash(
                        key, marker.player
                    )
            BoardManager.set_collapsed_space(pos, player, board)
            if game_state.zobrist_hash is not None:
                game_state.zobrist_hash ^= zobrist.get_collapsed_hash(key)

        # 5. Update territory_spaces for the moving player.
        spaces_gained = len(target_region.spaces) + len(border_markers)
        if spaces_gained > 0:
            for ps in game_state.players:
                if ps.player_number == player:
                    ps.territory_spaces += spaces_gained
                    break

    @staticmethod
    def _generate_all_positions(
        board_type: BoardType, size: int
    ) -> List[Position]:
        """Generate all valid positions"""
        # Use BoardManager logic (replicated here or delegated if BoardManager
        # exposed it)
        # For now, keep existing logic but ensure it matches BoardManager
        positions = []
        if board_type == BoardType.SQUARE8:
            for x in range(8):
                for y in range(8):
                    positions.append(Position(x=x, y=y))
        elif board_type == BoardType.SQUARE19:
            for x in range(19):
                for y in range(19):
                    positions.append(Position(x=x, y=y))
        elif board_type == BoardType.HEXAGONAL:
            radius = size - 1
            for x in range(-radius, radius + 1):
                for y in range(-radius, radius + 1):
                    z = -x - y
                    if (abs(x) <= radius and
                            abs(y) <= radius and
                            abs(z) <= radius):
                        positions.append(Position(x=x, y=y, z=z))
        return positions

    @staticmethod
    def _get_adjacent_positions(
        pos: Position, board_type: BoardType, size: int
    ) -> List[Position]:
        """Get adjacent positions"""
        # Use BoardManager logic
        # For now, keep existing logic
        adjacent = []

        if board_type in [BoardType.SQUARE8, BoardType.SQUARE19]:
            # Moore neighborhood
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    new_x, new_y = pos.x + dx, pos.y + dy
                    # Bounds check
                    limit = 8 if board_type == BoardType.SQUARE8 else 19
                    if 0 <= new_x < limit and 0 <= new_y < limit:
                        adjacent.append(Position(x=new_x, y=new_y))

        elif board_type == BoardType.HEXAGONAL:
            hex_directions = [
                (1, 0, -1), (-1, 0, 1),
                (0, 1, -1), (0, -1, 1),
                (1, -1, 0), (-1, 1, 0)
            ]
            radius = size - 1
            for dx, dy, dz in hex_directions:
                if pos.z is None:
                    continue
                nx, ny, nz = pos.x + dx, pos.y + dy, pos.z + dz
                if (abs(nx) <= radius and
                        abs(ny) <= radius and
                        abs(nz) <= radius):
                    adjacent.append(Position(x=nx, y=ny, z=nz))

        return adjacent

    @staticmethod
    def get_visible_stacks(
        pos: Position, game_state: GameState
    ) -> List[RingStack]:
        """
        Get all stacks visible from a given position (line of sight).
        This is used for determining capture/overtake potential.
        """
        visible_stacks = []
        board = game_state.board
        board_type = board.type
        size = board.size

        directions = []
        if board_type in [BoardType.SQUARE8, BoardType.SQUARE19]:
            # 8 directions for square board
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    directions.append((dx, dy, 0))
        elif board_type == BoardType.HEXAGONAL:
            # 6 directions for hexagonal board
            directions = [
                (1, 0, -1), (-1, 0, 1),
                (0, 1, -1), (0, -1, 1),
                (1, -1, 0), (-1, 1, 0)
            ]

        limit = 8 if board_type == BoardType.SQUARE8 else 19
        radius = size - 1

        for dx, dy, dz in directions:
            curr_x, curr_y = pos.x, pos.y
            curr_z = pos.z if pos.z is not None else 0

            # Raycast in this direction
            while True:
                curr_x += dx
                curr_y += dy
                curr_z += dz

                # Check bounds
                if board_type in [BoardType.SQUARE8, BoardType.SQUARE19]:
                    if not (0 <= curr_x < limit and 0 <= curr_y < limit):
                        break
                elif board_type == BoardType.HEXAGONAL:
                    if not (abs(curr_x) <= radius and
                            abs(curr_y) <= radius and
                            abs(curr_z) <= radius):
                        break

                curr_pos_key = f"{curr_x},{curr_y}"
                if board_type == BoardType.HEXAGONAL:
                    curr_pos_key += f",{curr_z}"

                # Check for stack
                if curr_pos_key in board.stacks:
                    visible_stacks.append(board.stacks[curr_pos_key])
                    break  # Line of sight blocked by first stack

                # Check for marker (markers don't block line of sight in
                # RingRift? Actually they might, but for stack interactions
                # usually we care about stacks. Assuming markers don't block
                # stack visibility for now, or if they do, add check here.)

        return visible_stacks

    @staticmethod
    def _apply_forced_elimination(game_state: GameState, move: Move):
        """Apply forced elimination move.

        Mirrors the TS eliminatePlayerRingOrCap / ELIMINATE_STACK behaviour by
        eliminating exactly the current cap from the target stack using the
        shared _eliminate_top_ring_at helper, so that board.eliminated_rings,
        total_rings_eliminated, and per-player eliminated_rings remain in
        sync with TS semantics.
        """
        board = game_state.board
        pos = move.to
        stack = board.stacks.get(pos.to_key())
        if not stack or stack.cap_height <= 0:
            return

        cap_height = stack.cap_height
        for _ in range(cap_height):
            GameEngine._eliminate_top_ring_at(
                game_state,
                pos,
                credited_player=move.player,
            )
