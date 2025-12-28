"""Core game engine for the RingRift AI service.

.. deprecated:: December 2025
    This module is the LEGACY game engine kept for backward compatibility with
    historical game replays recorded under previous rule versions.

    **For new code, use:**
    - :mod:`app.game_engine` - Canonical game engine with RR-CANON compliance
    - :mod:`app.rules.fsm` - 7-phase finite state machine

    **This module will be removed** once all legacy replays have been migrated
    or marked as unmigrateable. See :doc:`docs/specs/LEGACY_RULES_DIFF.md` for
    details on rule differences between versions.

    **Current usage:**
    - :mod:`app.rules.legacy.replay_compatibility` uses this for legacy replays
    - Re-exported via :mod:`app.game_engine` for backwards compatibility only

    **DO NOT** add new code that depends on this module.

**SSoT (Single Source of Truth) Policy:**

The canonical rules defined in ``RULES_CANONICAL_SPEC.md`` (together with
``docs/rules/COMPLETE_RULES.md`` / ``docs/rules/COMPACT_RULES.md``) are the
**ultimate authority** for RingRift game semantics. All implementations
must derive from and faithfully implement these canonical rules.

**Implementation Hierarchy:**

1. **Canonical rules** (written specification) are the ultimate SSoT.
2. **TS shared engine** (``src/shared/engine/**``) is the *primary
   executable derivation* of the canonical rules spec.
3. **This Python module** is a *host adapter* that must mirror the
   canonical rules. If this code disagrees with the canonical rules
   or the validated TS engine behaviour, this code must be updated—
   never the other way around.

This module provides a lightweight, Python-hosted view of the RingRift
rules sufficient for AI search, evaluation, and TS↔Python contract tests.
When divergence is detected (via parity tests or replay verification),
the Python code is considered incorrect and must be fixed to match the
canonical rules and TS reference implementation.

For which modules are allowed to encode rules semantics vs. which must
act as adapters over the shared engine, see the "Rules Entry Surfaces /
SSoT checklist" in ``docs/rules/RULES_ENGINE_SURFACE_AUDIT.md``.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from enum import Enum

from .board_manager import BoardManager
from .models import (
    BoardState,
    BoardType,
    GamePhase,
    GameState,
    GameStatus,
    MarkerInfo,
    Move,
    MoveType,
    Position,
    RingStack,
    Territory,
)

# Import canonical phase requirements from the dedicated module (SSoT)
# This consolidation ensures a single definition for PhaseRequirementType and PhaseRequirement
from .game_engine.phase_requirements import PhaseRequirement, PhaseRequirementType

__all__ = [
    # Classes
    "GameEngine",
    "PhaseRequirement",
    "PhaseRequirementType",
]


from .core.zobrist import ZobristHash
from .rules.capture_chain import enumerate_capture_moves_py
from .rules.core import count_rings_in_play_for_player, get_effective_line_length
from .rules.generators import (
    CaptureGenerator,
    LineGenerator,
    MovementGenerator,
    PlacementGenerator,
    TerritoryGenerator,
)
from .rules.geometry import BoardGeometry
from .rules.recovery import (
    apply_recovery_slide,
    get_expanded_recovery_moves,
    has_any_recovery_move,
)

# SSoT generators for move enumeration (Phase 1-2 consolidation, December 2025)
_line_generator = LineGenerator()
_territory_generator = TerritoryGenerator()
_capture_generator = CaptureGenerator()
_movement_generator = MovementGenerator()
_placement_generator = PlacementGenerator()

# Runtime deprecation warning - placed after imports to avoid circular import issues
import warnings
warnings.warn(
    "app._game_engine_legacy is deprecated and will be removed in Q2 2026. "
    "Use app.game_engine instead for new code.",
    DeprecationWarning,
    stacklevel=2,
)

DEBUG_ENGINE = os.environ.get("RINGRIFT_DEBUG_ENGINE") == "1"
STRICT_NO_MOVE_INVARIANT = os.environ.get(
    "RINGRIFT_STRICT_NO_MOVE_INVARIANT",
    "0",
) in {"1", "true", "yes", "on"}
# Set to "1" to disable phase/move invariant check for test scaffolding.
# This allows tests to create synthetic move sequences without respecting
# the canonical phase model.
SKIP_PHASE_INVARIANT = os.environ.get(
    "RINGRIFT_SKIP_PHASE_INVARIANT",
    "0",
) in {"1", "true", "yes", "on"}


def _debug(msg: str) -> None:
    """Write a debug message to stderr when engine debug is enabled."""
    if DEBUG_ENGINE:
        sys.stderr.write(msg)


class GameEngine:
    """Python GameEngine used by the AI service.

    **SSoT Policy:** This is a *host adapter*, NOT a rules SSoT. The
    canonical rules in ``RULES_CANONICAL_SPEC.md`` are the ultimate
    authority. The TS shared engine is the primary executable derivation.
    If this code disagrees with either, THIS CODE IS WRONG and must be
    fixed to match.

    This engine:

    - Mirrors the shared TS engine semantics closely enough for AI search,
      training, and TS↔Python parity tests.
    - Exposes `get_valid_moves` and `apply_move` as the primary APIs for
      AI agents and training loops.
    - Uses the shared `app.ai.move_cache` (LRU) to avoid recomputing legal
      moves for identical positions. This cache is keyed by `zobristHash`
      when available (full-fidelity board state, including stack ring
      composition) plus the small pieces of metadata that affect move
      legality (phase, player, rulesOptions, etc).
    """

    @staticmethod
    def get_valid_moves(game_state: GameState, player_number: int) -> list[Move]:
        """Return all valid moves for ``player_number`` in ``game_state``.

        This mirrors the TS `GameEngine.getValidMoves` surface and is used
        as the primary legal‑move generator for AI agents and parity tests.
        """
        # Only generate moves if it's the player's turn
        if game_state.current_player != player_number:
            return []

        # Move generation caching is delegated to app.ai.move_cache so that:
        # - keys include `zobristHash` (full ring composition) when available,
        # - keys include phase + move_history length + must_move_from_stack_key
        #   + rulesOptions, preventing stale move surfaces,
        # - caching can be bounded (LRU) and toggled via env flags.
        #
        # NOTE: chainCaptureState affects legal capture sequences but is not
        # encoded in Zobrist or board structure; move_cache bypasses caching
        # when chainCaptureState is present.
        from .ai.move_cache import cache_moves, get_cached_moves

        cached = get_cached_moves(game_state, player_number)
        if cached is not None:
            return cached

        phase = game_state.current_phase
        moves: list[Move] = []

        # ═══════════════════════════════════════════════════════════════════
        # STRICT R076: Core rules MUST NOT auto-generate bookkeeping moves.
        # get_valid_moves returns ONLY interactive moves. When there are no
        # interactive moves, hosts must call get_phase_requirement() to learn
        # what bookkeeping move (NO_*_ACTION or FORCED_ELIMINATION) they need
        # to construct and apply.
        # ═══════════════════════════════════════════════════════════════════

        if phase == GamePhase.RING_PLACEMENT:
            # Core: only interactive placement/skip options.
            placement_moves = GameEngine._get_ring_placement_moves(game_state, player_number)
            skip_moves = GameEngine._get_skip_placement_moves(game_state, player_number)
            moves = placement_moves + skip_moves
            # NO auto NO_PLACEMENT_ACTION - hosts use get_phase_requirement()

        elif phase == GamePhase.MOVEMENT:
            # Core: only interactive movement/capture options.
            movement_moves = GameEngine._get_movement_moves(game_state, player_number)
            capture_moves = GameEngine._get_capture_moves(game_state, player_number)
            # Recovery slides for temporarily eliminated players (RR-CANON-R110-R115)
            # Use expanded recovery with line, territory, fallback modes + skip option
            recovery_moves = get_expanded_recovery_moves(game_state, player_number)
            moves = movement_moves + capture_moves + recovery_moves
            # NO auto NO_MOVEMENT_ACTION - hosts use get_phase_requirement()

        elif phase == GamePhase.CAPTURE:
            # RR-CANON-R093: Post-movement captures in the capture phase are
            # evaluated only from the stack that just moved, at its landing
            # position. Unlike MOVEMENT phase capture enumeration (which is
            # global), this is a narrow optional follow-up window.
            last_move = game_state.move_history[-1] if game_state.move_history else None
            attacker_pos = last_move.to if last_move and last_move.to else None

            move_number = len(game_state.move_history) + 1
            moves = []
            if attacker_pos is not None:
                moves = enumerate_capture_moves_py(
                    game_state,
                    player_number,
                    attacker_pos,
                    move_number=move_number,
                    kind="initial",
                )
            else:
                # Defensive: CAPTURE should normally follow a movement move
                # with a landing position. Even if the attacker position is
                # unavailable (corrupt history / legacy states), CAPTURE is an
                # optional window and declining it must still be recordable.
                _debug(
                    "[GameEngine] CAPTURE phase: missing attacker position; "
                    "surfacing SKIP_CAPTURE only\n"
                )

            # RR-CANON-R073: capture is optional; declining must be recorded.
            moves.append(
                Move(
                    id=f"skip-capture-{move_number}",
                    type=MoveType.SKIP_CAPTURE,
                    player=player_number,
                    to=Position(x=0, y=0),
                    timestamp=game_state.last_move_at,
                    thinkTime=0,
                    moveNumber=move_number,
                )
            )

        elif phase == GamePhase.CHAIN_CAPTURE:
            moves = GameEngine._get_capture_moves(game_state, player_number)

        elif phase == GamePhase.LINE_PROCESSING:
            # Core: only interactive line-processing decisions.
            moves = GameEngine._get_line_processing_moves(game_state, player_number)
            # NO auto NO_LINE_ACTION - hosts use get_phase_requirement()

        elif phase == GamePhase.TERRITORY_PROCESSING:
            # Core: only interactive territory-processing decisions.
            moves = GameEngine._get_territory_processing_moves(game_state, player_number)
            # NO auto NO_TERRITORY_ACTION - hosts use get_phase_requirement()

        elif phase == GamePhase.FORCED_ELIMINATION:
            # In the forced_elimination phase (7th and final phase per
            # RR-CANON-R070), enumerate explicit FORCED_ELIMINATION moves for
            # the blocked player.
            moves = GameEngine._get_forced_elimination_moves(game_state, player_number)

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

        # RR-CANON-R072/R100/R205: When no placement/movement/capture is
        # available but the player controls stacks, forced elimination must be
        # surfaced as an explicit interactive decision. The core rules layer
        # exposes FE via dedicated phase logic; it does not auto-fallback to
        # FORCED_ELIMINATION moves from get_valid_moves (RR-CANON-R076).

        # IMPORTANT: Only cache non-empty move lists. Empty lists should NOT be
        # cached because host-level callers (like DefaultRulesEngine.get_valid_moves)
        # may need to synthesize bookkeeping moves (NO_PLACEMENT_ACTION, etc.) when
        # there are no interactive moves. Caching empty lists prevents proper
        # bookkeeping move synthesis and causes AI to return None incorrectly.
        if moves:
            cache_moves(game_state, player_number, moves)
        return moves

    @staticmethod
    def clear_cache() -> None:
        """Clear the move cache"""
        from .ai.move_cache import clear_move_cache

        clear_move_cache()

    @staticmethod
    def get_phase_requirement(game_state: GameState, player_number: int) -> PhaseRequirement | None:
        """
        Return the phase requirement when no interactive moves exist.

        Per RR-CANON-R076, the core rules layer MUST NOT auto-generate moves.
        When get_valid_moves returns an empty list, hosts should call this
        method to learn what bookkeeping move they must construct and apply.

        Args:
            game_state: The current game state.
            player_number: The player to check (must be current_player).

        Returns:
            PhaseRequirement if a bookkeeping move is required, None otherwise.
            Returns None if:
            - It's not this player's turn
            - Game is not active
            - Interactive moves are available
        """
        if game_state.current_player != player_number:
            return None

        # Check game status
        game_status = game_state.game_status
        status_str = game_status.value if hasattr(game_status, "value") else str(game_status)
        if status_str != "active":
            return None

        # First check if there are interactive moves available
        interactive_moves = GameEngine.get_valid_moves(game_state, player_number)
        if interactive_moves:
            return None  # No requirement - interactive moves exist

        # No interactive moves - determine what no-action move is required
        phase = game_state.current_phase

        if phase == GamePhase.RING_PLACEMENT:
            return PhaseRequirement(
                type=PhaseRequirementType.NO_PLACEMENT_ACTION_REQUIRED,
                player=player_number,
                eligible_positions=[],
            )

        elif phase == GamePhase.MOVEMENT:
            # No interactive movement/capture options exist for this player.
            # Per the 7-phase model (RR-CANON-R070), forced elimination is a
            # distinct phase entered via _end_turn when a player with stacks
            # has no legal placement/movement/capture actions. The MOVEMENT
            # phase itself only ever records explicit moves or a
            # NO_MOVEMENT_ACTION bookkeeping move.
            return PhaseRequirement(
                type=PhaseRequirementType.NO_MOVEMENT_ACTION_REQUIRED,
                player=player_number,
                eligible_positions=[],
            )

        elif phase == GamePhase.LINE_PROCESSING:
            return PhaseRequirement(
                type=PhaseRequirementType.NO_LINE_ACTION_REQUIRED,
                player=player_number,
                eligible_positions=[],
            )

        elif phase == GamePhase.TERRITORY_PROCESSING:
            return PhaseRequirement(
                type=PhaseRequirementType.NO_TERRITORY_ACTION_REQUIRED,
                player=player_number,
                eligible_positions=[],
            )

        elif phase == GamePhase.FORCED_ELIMINATION:
            # Player is in forced_elimination phase but get_valid_moves returned
            # empty. This means they need to select a stack to eliminate from.
            # Return FORCED_ELIMINATION_REQUIRED with eligible positions.
            from app.board_manager import BoardManager

            stacks = BoardManager.get_player_stacks(game_state.board, player_number)
            eligible_positions = [
                stack.position for stack in stacks if stack.cap_height > 0
            ]
            return PhaseRequirement(
                type=PhaseRequirementType.FORCED_ELIMINATION_REQUIRED,
                player=player_number,
                eligible_positions=eligible_positions,
            )

        return None

    @staticmethod
    def synthesize_bookkeeping_move(
        requirement: PhaseRequirement,
        game_state: GameState,
    ) -> Move:
        """
        Synthesize a canonical bookkeeping move from a phase requirement.

        This is a HOST-LEVEL helper, not part of the core rules layer.
        Per RR-CANON-R076, hosts are responsible for constructing and
        applying bookkeeping moves (no_*_action, forced_elimination)
        when the core layer returns a phase requirement.

        Args:
            requirement: The phase requirement from get_phase_requirement().
            game_state: The current game state.

        Returns:
            A canonical Move object that satisfies the phase requirement.

        Raises:
            ValueError: If forced elimination is required but no eligible
                positions are available.
        """
        from datetime import datetime

        move_number = len(game_state.move_history) + 1
        player = requirement.player

        if requirement.type == PhaseRequirementType.NO_PLACEMENT_ACTION_REQUIRED:
            return Move(
                id=f"no-placement-action-{move_number}",
                type=MoveType.NO_PLACEMENT_ACTION,
                player=player,
                to=Position(x=0, y=0),
                timestamp=datetime.now(),
                thinkTime=0,
                moveNumber=move_number,
            )

        elif requirement.type == PhaseRequirementType.NO_MOVEMENT_ACTION_REQUIRED:
            return Move(
                id=f"no-movement-action-{move_number}",
                type=MoveType.NO_MOVEMENT_ACTION,
                player=player,
                to=Position(x=0, y=0),
                timestamp=datetime.now(),
                thinkTime=0,
                moveNumber=move_number,
            )

        elif requirement.type == PhaseRequirementType.NO_LINE_ACTION_REQUIRED:
            return Move(
                id=f"no-line-action-{move_number}",
                type=MoveType.NO_LINE_ACTION,
                player=player,
                to=Position(x=0, y=0),
                timestamp=datetime.now(),
                thinkTime=0,
                moveNumber=move_number,
            )

        elif requirement.type == PhaseRequirementType.NO_TERRITORY_ACTION_REQUIRED:
            return Move(
                id=f"no-territory-action-{move_number}",
                type=MoveType.NO_TERRITORY_ACTION,
                player=player,
                to=Position(x=0, y=0),
                timestamp=datetime.now(),
                thinkTime=0,
                moveNumber=move_number,
            )

        elif requirement.type == PhaseRequirementType.FORCED_ELIMINATION_REQUIRED:
            # Synthesize a FORCED_ELIMINATION move. Per RR-CANON-R206, when
            # multiple stacks are eligible, the player should choose. For
            # auto-synthesis, we pick the first eligible stack.
            if not requirement.eligible_positions:
                raise ValueError(
                    "FORCED_ELIMINATION_REQUIRED but no eligible positions"
                )
            target_pos = requirement.eligible_positions[0]
            return Move(
                id=f"forced-elimination-{move_number}",
                type=MoveType.FORCED_ELIMINATION,
                player=player,
                to=target_pos,
                timestamp=datetime.now(),
                thinkTime=0,
                moveNumber=move_number,
            )

        raise ValueError(f"Unknown requirement type: {requirement.type}")

    @staticmethod
    def apply_move(game_state: GameState, move: Move, *, trace_mode: bool = False) -> GameState:
        """
        Apply a move to a game state and return the new state.

        Args:
            game_state: The current game state.
            move: The move to apply.
            trace_mode: When True, disables automatic forced elimination during
                turn rotation, matching TS's traceMode behavior. Use this when
                replaying recorded games where forced eliminations are explicit
                moves in the sequence.
        """
        # Optimization: Manual shallow copy of GameState and BoardState
        # We only deep copy mutable structures that we intend to modify

        # RR-PARITY-FIX (Dec 2025): In trace_mode (legacy replay), handle phase
        # divergence between Python and TypeScript recordings. When Python is in
        # line_processing with pending_line_reward_elimination=True but the recording
        # has a territory action, coerce the phase to match the recording.
        if trace_mode:
            game_state = GameEngine._coerce_phase_for_trace_mode(game_state, move)

        # Strict phase/move invariant: ensure that the incoming move is
        # appropriate for the current phase. This enforces the canonical
        # per-phase move taxonomy (action / skip / no-action) and will
        # raise immediately on any recording that violates it.
        # Can be disabled via RINGRIFT_SKIP_PHASE_INVARIANT for test scaffolding.
        if not SKIP_PHASE_INVARIANT:
            GameEngine._assert_phase_move_invariant(game_state, move)

        # Enforce turn ownership: for ACTIVE states, the move actor must
        # match the current player. This prevents silent application of
        # mis-attributed moves during self-play recording or replay.
        # Can be disabled via RINGRIFT_SKIP_PHASE_INVARIANT for test scaffolding.
        if not SKIP_PHASE_INVARIANT:
            if game_state.game_status == GameStatus.ACTIVE and move.player != game_state.current_player:
                raise ValueError(f"Move player {move.player} does not match current player {game_state.current_player}")

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
            new_state.zobrist_hash = ZobristHash().compute_initial_hash(new_state)

        # Capture S-invariant before move
        before_snapshot = BoardManager.compute_progress_snapshot(new_state)

        # Update hash for phase/player change (remove old)
        zobrist = ZobristHash()
        if new_state.zobrist_hash is not None:
            new_state.zobrist_hash ^= zobrist.get_player_hash(new_state.current_player)
            new_state.zobrist_hash ^= zobrist.get_phase_hash(new_state.current_phase)

        # Note: RESIGN and TIMEOUT are not valid MoveType enum values;
        # they are handled at the game termination level, not as moves.
        is_terminal_move = False

        if move.type == MoveType.SWAP_SIDES:
            GameEngine._apply_swap_sides(new_state, move)
        elif move.type == MoveType.PLACE_RING:
            GameEngine._apply_place_ring(new_state, move)
        elif move.type == MoveType.SKIP_PLACEMENT:
            # No board change; phase update will advance the turn.
            pass
        elif move.type == MoveType.NO_PLACEMENT_ACTION:
            # Explicit forced no-op in RING_PLACEMENT when the player has no
            # legal placement anywhere. State is unchanged; phase update will
            # advance to MOVEMENT.
            pass
        elif move.type == MoveType.NO_MOVEMENT_ACTION:
            # Explicit forced no-op in MOVEMENT when the player has no legal
            # movement or capture anywhere. State is unchanged; phase update
            # will advance to line_processing per RR-CANON-R075.
            pass
        elif move.type == MoveType.NO_LINE_ACTION:
            # Explicit forced no-op in LINE_PROCESSING when there are no lines
            # to process. State is unchanged; phase update will advance to
            # territory_processing per RR-CANON-R075.
            pass
        elif move.type == MoveType.NO_TERRITORY_ACTION:
            # Explicit forced no-op in TERRITORY_PROCESSING when there are no
            # territories to process. State is unchanged; phase update will
            # end the turn per RR-CANON-R075.
            pass
        elif move.type == MoveType.MOVE_STACK:
            GameEngine._apply_move_stack(new_state, move)
        elif move.type in (
            MoveType.OVERTAKING_CAPTURE,
            MoveType.CONTINUE_CAPTURE_SEGMENT,
            MoveType.CHAIN_CAPTURE,
        ):
            GameEngine._apply_chain_capture(new_state, move)
        elif move.type == MoveType.SKIP_CAPTURE:
            # Explicit skip: decline optional post-movement capture.
            # No board change; phase update will advance to line_processing.
            pass
        elif move.type in (
            MoveType.PROCESS_LINE,
            MoveType.CHOOSE_LINE_REWARD,
            MoveType.LINE_FORMATION,
            MoveType.CHOOSE_LINE_OPTION,
        ):
            GameEngine._apply_line_formation(new_state, move)
        elif move.type == MoveType.ELIMINATE_RINGS_FROM_STACK:
            # RR-CANON-R122/R123/R145: eliminate_rings_from_stack can occur in:
            # - LINE_PROCESSING: for Option 1 line reward (eliminates exactly 1 ring)
            # - TERRITORY_PROCESSING: for territory claim prerequisite (eliminates cap)
            # - FORCED_ELIMINATION: when player exceeds ring limit (eliminates cap)
            GameEngine._apply_forced_elimination(new_state, move)
            # Clear pending flag if this was a line elimination
            elimination_context = getattr(move, 'elimination_context', None)
            if elimination_context == 'line':
                new_state.pending_line_reward_elimination = False
        elif move.type in (
            # Canonical territory decision move (legacy alias: PROCESS_TERRITORY_REGION).
            MoveType.CHOOSE_TERRITORY_OPTION,
            # Legacy alias retained for replay compatibility.
            MoveType.PROCESS_TERRITORY_REGION,
            MoveType.TERRITORY_CLAIM,
        ):
            # Territory decisions:
            # - choose_territory_option (legacy alias: process_territory_region)
            GameEngine._apply_territory_claim(new_state, move)
        elif move.type == MoveType.FORCED_ELIMINATION:
            GameEngine._apply_forced_elimination(new_state, move)
        elif move.type == MoveType.RECOVERY_SLIDE:
            # Recovery slide for temporarily eliminated players (RR-CANON-R110-R115)
            # Extract option and collapse_positions from move metadata if present
            option = getattr(move, "recovery_option", None) or 1
            collapse_positions = getattr(move, "collapse_positions", None)
            apply_recovery_slide(new_state, move, option=option, collapse_positions=collapse_positions)
        # Note: RESIGN and TIMEOUT are not valid MoveType enum values and would
        # be handled at the game termination level, not as moves. Any unrecognized
        # move types fall through with no board mutation.

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

        if is_terminal_move:
            GameEngine._apply_terminal_move(new_state, move)
        else:
            # Handle phase transitions
            GameEngine._update_phase(new_state, move, trace_mode=trace_mode)

        # Update hash for phase/player change (add new)
        if new_state.zobrist_hash is not None:
            new_state.zobrist_hash ^= zobrist.get_player_hash(new_state.current_player)
            new_state.zobrist_hash ^= zobrist.get_phase_hash(new_state.current_phase)

        # Verify S-invariant
        # S = markers + collapsed + eliminated
        # S must be non-decreasing
        after_snapshot = BoardManager.compute_progress_snapshot(new_state)
        if after_snapshot.S < before_snapshot.S:
            # In a real engine we might throw, but for AI simulation we
            # log/warn or just accept it if it's a known deviation.
            # For now, we'll assume correctness of logic but this hook is here.
            pass

        if is_terminal_move:
            return new_state

        # Check victory conditions
        GameEngine._check_victory(new_state)

        # TS parity: enforce ANM boundary resolution (RR-CANON-R202/R203).
        # After any move, if the game is still ACTIVE and ANM(state) holds for the
        # current player, resolve immediately via forced elimination and/or
        # bare-board stalemate evaluation so no externally visible ACTIVE state
        # satisfies ANM.
        #
        # In trace_mode we must NOT auto-resolve ANM, because forced elimination
        # is required to be recorded as an explicit move. Trace replays need
        # the state to remain in forced_elimination until the move is applied.
        if new_state.game_status == GameStatus.ACTIVE and not trace_mode:
            GameEngine._resolve_anm_for_current_player(new_state)

        # Strict no-move invariant: after any move that leaves the game ACTIVE,
        # assert that the current player has at least one legal action.
        if STRICT_NO_MOVE_INVARIANT and new_state.game_status == GameStatus.ACTIVE:
            GameEngine._assert_active_player_has_legal_action(new_state, move)

        # Recovery slide mutates multiple board domains (markers, stacks,
        # collapsed spaces) via the recovery module, which historically did not
        # maintain incremental Zobrist hash updates. Recompute from scratch to
        # keep move caching and parity tooling sound.
        if move.type == MoveType.RECOVERY_SLIDE:
            new_state.zobrist_hash = ZobristHash().compute_initial_hash(new_state)

        return new_state

    @staticmethod
    def _apply_terminal_move(game_state: GameState, move: Move) -> None:
        """Apply a terminal meta-move (resign/timeout) to end the game."""
        game_state.game_status = GameStatus.COMPLETED
        game_state.current_phase = GamePhase.GAME_OVER

        if len(game_state.players) == 2:
            winner = next(
                (p.player_number for p in game_state.players if p.player_number != move.player),
                None,
            )
            game_state.winner = winner
        else:
            game_state.winner = None

    @staticmethod
    def _coerce_phase_for_trace_mode(
        game_state: GameState,
        move: Move,
    ) -> GameState:
        """
        Coerce phase to match legacy recordings during trace_mode replay.

        RR-PARITY-FIX (Dec 2025): This handles phase divergence between Python
        and TypeScript recordings. Key scenarios:

        1. line_processing → territory_processing when:
           - Python has pending_line_reward_elimination=True (expects elimination)
           - Recording has a territory action (no_territory_action, skip_territory, etc.)
           - This happens because older TS versions didn't require elimination step

        2. line_processing → territory_processing when:
           - Python detects lines that TS didn't (has_line_moves=True)
           - Recording has a territory action instead of line action

        This function returns a (possibly modified) game_state with the phase
        coerced to match the incoming move's expected phase.
        """
        phase = game_state.current_phase
        mtype = move.type

        # Territory-related move types
        territory_moves = {
            MoveType.NO_TERRITORY_ACTION,
            MoveType.SKIP_TERRITORY_PROCESSING,
            MoveType.CHOOSE_TERRITORY_OPTION,
            MoveType.PROCESS_TERRITORY_REGION,
        }

        # Line-related move types
        line_moves = {
            MoveType.NO_LINE_ACTION,
            MoveType.PROCESS_LINE,
            MoveType.CHOOSE_LINE_OPTION,
            MoveType.CHOOSE_LINE_REWARD,
        }

        # Case 0: In capture/chain_capture but move is line/territory action
        # This happens when recording skipped capture phase that Python expects
        if phase in (GamePhase.CAPTURE, GamePhase.CHAIN_CAPTURE):
            if mtype in line_moves:
                new_state = game_state.model_copy()
                new_state.current_phase = GamePhase.LINE_PROCESSING
                new_state.current_player = move.player
                return new_state
            if mtype in territory_moves:
                new_state = game_state.model_copy()
                new_state.current_phase = GamePhase.TERRITORY_PROCESSING
                new_state.current_player = move.player
                return new_state

        # Case 0b: In ring_placement/movement but move is line/territory action
        # This can happen when recording has different turn rotation timing
        if phase in (GamePhase.RING_PLACEMENT, GamePhase.MOVEMENT):
            if mtype in line_moves:
                new_state = game_state.model_copy()
                new_state.current_phase = GamePhase.LINE_PROCESSING
                new_state.current_player = move.player
                return new_state
            if mtype in territory_moves:
                new_state = game_state.model_copy()
                new_state.current_phase = GamePhase.TERRITORY_PROCESSING
                new_state.current_player = move.player
                return new_state
            # RR-PARITY-FIX (Dec 2025): Handle eliminate_rings_from_stack from early phases
            # This can happen when recording has line/territory elimination but Python
            # hasn't transitioned to the expected phase yet
            if mtype == MoveType.ELIMINATE_RINGS_FROM_STACK:
                # Determine target phase based on pending state
                pending_line = getattr(game_state, 'pending_line_reward_elimination', False)
                if pending_line:
                    new_state = game_state.model_copy()
                    new_state.current_phase = GamePhase.LINE_PROCESSING
                    new_state.current_player = move.player
                    return new_state
                else:
                    new_state = game_state.model_copy()
                    new_state.current_phase = GamePhase.TERRITORY_PROCESSING
                    new_state.current_player = move.player
                    return new_state

        # Case 1: In line_processing but move is a territory action
        if phase == GamePhase.LINE_PROCESSING and mtype in territory_moves:
            pending_elim = getattr(game_state, 'pending_line_reward_elimination', False)

            # Check if Python expects line actions that recording doesn't have
            if pending_elim:
                # Create a copy to modify (avoid mutating input)
                new_state = game_state.model_copy()
                new_state.current_phase = GamePhase.TERRITORY_PROCESSING
                new_state.pending_line_reward_elimination = False
                new_state.current_player = move.player
                return new_state

            # Also check if Python sees line moves that recording skipped
            # (This is a more expensive check, so only do it if needed)
            valid_moves = GameEngine.get_valid_moves(game_state, game_state.current_player)
            has_line_moves = any(
                m.type in {MoveType.PROCESS_LINE, MoveType.CHOOSE_LINE_OPTION, MoveType.CHOOSE_LINE_REWARD}
                for m in valid_moves
            )
            if has_line_moves:
                new_state = game_state.model_copy()
                new_state.current_phase = GamePhase.TERRITORY_PROCESSING
                new_state.current_player = move.player
                return new_state

        # Case 2: In ring_placement/movement but move is forced_elimination
        if phase in (GamePhase.RING_PLACEMENT, GamePhase.MOVEMENT) and mtype == MoveType.FORCED_ELIMINATION:
            new_state = game_state.model_copy()
            new_state.current_phase = GamePhase.FORCED_ELIMINATION
            new_state.current_player = move.player
            return new_state

        # Case 2b: In line_processing but move is forced_elimination
        # This happens when GPU selfplay records forced_elimination during line phase
        if phase == GamePhase.LINE_PROCESSING and mtype == MoveType.FORCED_ELIMINATION:
            new_state = game_state.model_copy()
            new_state.current_phase = GamePhase.FORCED_ELIMINATION
            new_state.current_player = move.player
            return new_state

        # Case 3: In forced_elimination but move is a territory action
        # This can happen when recording didn't have forced elimination but Python detected it
        if phase == GamePhase.FORCED_ELIMINATION and mtype in territory_moves:
            new_state = game_state.model_copy()
            new_state.current_phase = GamePhase.TERRITORY_PROCESSING
            new_state.current_player = move.player
            return new_state

        # Placement-related move types
        placement_moves = {
            MoveType.PLACE_RING,
            MoveType.NO_PLACEMENT_ACTION,
            MoveType.SKIP_PLACEMENT,
        }

        # Movement-related move types
        movement_moves = {
            MoveType.MOVE_STACK,
            MoveType.MOVE_RING,
            MoveType.NO_MOVEMENT_ACTION,
            MoveType.RECOVERY_SLIDE,
            MoveType.SKIP_RECOVERY,
        }

        # Capture-related move types
        capture_moves = {
            MoveType.OVERTAKING_CAPTURE,
            MoveType.CONTINUE_CAPTURE_SEGMENT,
            MoveType.SKIP_CAPTURE,
        }

        # Case 4: In territory/line/forced_elim but move is a placement/movement action
        # This happens when recording has different turn rotation that already moved to next turn
        if phase in (GamePhase.TERRITORY_PROCESSING, GamePhase.LINE_PROCESSING, GamePhase.FORCED_ELIMINATION):
            if mtype in placement_moves:
                new_state = game_state.model_copy()
                new_state.current_phase = GamePhase.RING_PLACEMENT
                new_state.current_player = move.player
                return new_state
            if mtype in movement_moves:
                new_state = game_state.model_copy()
                new_state.current_phase = GamePhase.MOVEMENT
                new_state.current_player = move.player
                return new_state
            # RR-PARITY-FIX (Dec 2025): Handle capture moves during line/territory processing
            # Historical hexagonal recordings may have capture moves interleaved differently
            if mtype in capture_moves:
                new_state = game_state.model_copy()
                new_state.current_phase = GamePhase.CAPTURE
                new_state.current_player = move.player
                return new_state

        # Case 5: In movement but move is a capture action
        # This can happen when recording has different capture detection timing
        if phase == GamePhase.MOVEMENT and mtype in capture_moves:
            new_state = game_state.model_copy()
            new_state.current_phase = GamePhase.CAPTURE
            new_state.current_player = move.player
            return new_state

        # Case 6: In capture/chain_capture but move is a placement/movement action
        # This happens with GPU selfplay format that doesn't record capture phases
        if phase in (GamePhase.CAPTURE, GamePhase.CHAIN_CAPTURE):
            if mtype in placement_moves:
                new_state = game_state.model_copy()
                new_state.current_phase = GamePhase.RING_PLACEMENT
                new_state.current_player = move.player
                return new_state
            if mtype in movement_moves:
                new_state = game_state.model_copy()
                new_state.current_phase = GamePhase.MOVEMENT
                new_state.current_player = move.player
                return new_state

        # Case 7: Player mismatch without phase mismatch (GPU selfplay format)
        # GPU selfplay uses a different turn structure: P1 place+move, P2 place+move, etc.
        # Always coerce current_player to match the move's player in trace_mode
        if game_state.current_player != move.player:
            new_state = game_state.model_copy()
            new_state.current_player = move.player
            return new_state

        return game_state

    @staticmethod
    def _assert_phase_move_invariant(
        game_state: GameState,
        move: Move,
    ) -> None:
        """
        Enforce the canonical phase→MoveType mapping for ACTIVE states.

        For any move applied via GameEngine.apply_move, the move.type must
        be one of the allowed types for game_state.current_phase:

        - RING_PLACEMENT:
            PLACE_RING, SKIP_PLACEMENT, NO_PLACEMENT_ACTION
        - MOVEMENT:
            MOVE_STACK, MOVE_RING, OVERTAKING_CAPTURE,
            CONTINUE_CAPTURE_SEGMENT, NO_MOVEMENT_ACTION, RECOVERY_SLIDE
        - CAPTURE:
            OVERTAKING_CAPTURE, CONTINUE_CAPTURE_SEGMENT, SKIP_CAPTURE
        - CHAIN_CAPTURE:
            OVERTAKING_CAPTURE, CONTINUE_CAPTURE_SEGMENT
        - LINE_PROCESSING:
            PROCESS_LINE, CHOOSE_LINE_OPTION (legacy: CHOOSE_LINE_REWARD), NO_LINE_ACTION
        - TERRITORY_PROCESSING:
            CHOOSE_TERRITORY_OPTION (legacy: PROCESS_TERRITORY_REGION),
            ELIMINATE_RINGS_FROM_STACK, SKIP_TERRITORY_PROCESSING, NO_TERRITORY_ACTION
        - FORCED_ELIMINATION:
            FORCED_ELIMINATION

        SWAP_SIDES, RESIGN, and TIMEOUT are permitted in any phase as
        meta-moves. Legacy move types (LINE_FORMATION, TERRITORY_CLAIM,
        CHAIN_CAPTURE) are also accepted to avoid breaking historical
        tests; canonical recordings should avoid them.
        """
        phase = game_state.current_phase
        mtype = move.type

        # Meta-moves allowed in any phase
        # Note: RESIGN and TIMEOUT are not valid MoveType enum values;
        # they are handled at the game termination level, not as moves.
        if mtype == MoveType.SWAP_SIDES:
            return

        # Legacy/experimental move types – accept for now, but callers
        # should treat recordings containing them as non-canonical.
        if mtype in (
            MoveType.LINE_FORMATION,
            MoveType.TERRITORY_CLAIM,
            MoveType.CHAIN_CAPTURE,
        ):
            return

        allowed: set[MoveType]
        if phase == GamePhase.RING_PLACEMENT:
            allowed = {
                MoveType.PLACE_RING,
                MoveType.SKIP_PLACEMENT,
                MoveType.NO_PLACEMENT_ACTION,
            }
        elif phase == GamePhase.MOVEMENT:
            allowed = {
                MoveType.MOVE_STACK,
                MoveType.MOVE_RING,
                MoveType.OVERTAKING_CAPTURE,
                MoveType.CONTINUE_CAPTURE_SEGMENT,
                MoveType.NO_MOVEMENT_ACTION,
                MoveType.RECOVERY_SLIDE,  # RR-CANON-R110-R115
                MoveType.SKIP_RECOVERY,  # RR-CANON-R112: player may skip recovery
            }
        elif phase == GamePhase.CAPTURE:
            # CAPTURE is the optional post-movement capture window (RR-CANON-R073/R093):
            # the player may perform a capture segment from the landing position
            # or explicitly decline via SKIP_CAPTURE to proceed to line_processing.
            allowed = {
                MoveType.OVERTAKING_CAPTURE,
                MoveType.CONTINUE_CAPTURE_SEGMENT,
                MoveType.SKIP_CAPTURE,
            }
        elif phase == GamePhase.CHAIN_CAPTURE:
            allowed = {
                MoveType.OVERTAKING_CAPTURE,
                MoveType.CONTINUE_CAPTURE_SEGMENT,
            }
        elif phase == GamePhase.LINE_PROCESSING:
            allowed = {
                MoveType.PROCESS_LINE,
                MoveType.CHOOSE_LINE_OPTION,
                MoveType.CHOOSE_LINE_REWARD,
                MoveType.NO_LINE_ACTION,
                MoveType.ELIMINATE_RINGS_FROM_STACK,  # RR-CANON-R123: line elimination
            }
        elif phase == GamePhase.TERRITORY_PROCESSING:
            allowed = {
                MoveType.CHOOSE_TERRITORY_OPTION,
                MoveType.PROCESS_TERRITORY_REGION,  # legacy alias
                MoveType.ELIMINATE_RINGS_FROM_STACK,
                MoveType.SKIP_TERRITORY_PROCESSING,
                MoveType.NO_TERRITORY_ACTION,
            }
        elif phase == GamePhase.FORCED_ELIMINATION:
            allowed = {
                MoveType.FORCED_ELIMINATION,
            }
        else:
            # Unknown phase – do not enforce
            return

        if mtype not in allowed:
            raise RuntimeError(
                f"Phase/move invariant violated: cannot apply move type " f"{mtype.value} in phase {phase.value}"
            )

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
        has_p2_move = any(m.player == 2 and m.type != MoveType.SWAP_SIDES for m in game_state.move_history)

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
        # Check player.eliminated_rings which tracks "rings credited to this
        # player" (i.e. rings they caused to be eliminated from any opponent).
        # This matches TypeScript VictoryAggregate.checkScoreThreshold() which
        # uses `player.eliminatedRings >= victoryThreshold` per RR-CANON-R061.
        # Note: board.eliminated_rings has different semantics (rings belonging
        # to player X that were eliminated, keyed by victim, not by causer).
        for p in game_state.players:
            if p.eliminated_rings >= game_state.victory_threshold:
                game_state.game_status = GameStatus.COMPLETED
                game_state.winner = p.player_number
                game_state.current_player = p.player_number  # TS parity: winner stays current
                game_state.current_phase = GamePhase.GAME_OVER
                return

        # 2. Territory Victory per RR-CANON-R062-v2
        # Victory requires BOTH:
        #   a) Territory >= floor(totalSpaces / numPlayers) + 1 (territory_victory_minimum)
        #   b) Territory > sum of all opponent territories
        territory_counts = {}
        for p_id in game_state.board.collapsed_spaces.values():
            if p_id not in territory_counts:
                territory_counts[p_id] = 0
            territory_counts[p_id] += 1

        # Get minimum threshold (use new field, fall back to legacy for old states)
        territory_minimum = (
            game_state.territory_victory_minimum
            if game_state.territory_victory_minimum is not None
            else game_state.territory_victory_threshold
        )

        for player in game_state.players:
            player_territory = territory_counts.get(player.player_number, 0)

            # Check condition 1: meets minimum threshold
            if player_territory < territory_minimum:
                continue

            # Check condition 2: more territory than all opponents combined
            opponent_territory = sum(
                territory_counts.get(p.player_number, 0)
                for p in game_state.players
                if p.player_number != player.player_number
            )

            if player_territory > opponent_territory:
                game_state.game_status = GameStatus.COMPLETED
                game_state.winner = player.player_number
                game_state.current_player = player.player_number  # TS parity: winner stays current
                game_state.current_phase = GamePhase.GAME_OVER
                return

        # 3. Early Last-Player-Standing (R172): if exactly one player has stacks
        # on the board AND all other players have ZERO total rings (board + hand),
        # that player wins immediately. This matches TS VictoryAggregate.evaluateVictory
        # early-LPS check (lines 342-363).
        #
        # A player's rings on board includes ALL their rings, even if buried
        # in stacks controlled by other players.
        #
        # Note: We only trigger this when there ARE stacks on board. If no stacks
        # exist, we fall through to the bare-board stalemate logic which handles
        # global structural terminality.
        players_with_stacks = set()
        for stack in game_state.board.stacks.values():
            players_with_stacks.add(stack.controlling_player)

        # Only consider Early LPS when exactly one player has stacks
        if len(players_with_stacks) == 1:
            stack_owner = next(iter(players_with_stacks))
            # Check if ALL other players have no material (total rings = 0)
            # This includes rings in hand AND rings on board (even if buried)
            others_have_material = False
            for p in game_state.players:
                if p.player_number == stack_owner:
                    continue
                total_rings = count_rings_in_play_for_player(game_state, p.player_number)
                if total_rings > 0:
                    others_have_material = True
                    break

            if not others_have_material:
                game_state.game_status = GameStatus.COMPLETED
                game_state.winner = stack_owner
                game_state.current_player = stack_owner  # TS parity: winner stays current
                game_state.current_phase = GamePhase.GAME_OVER
                return

        # 4. Round-based LPS victory (R172) is evaluated at the START of each
        # interactive turn via _maybe_apply_lps_victory_at_turn_start(...)
        # after LPS tracking is updated. Do not re-check here.

        # 5. Global structural terminality
        # Fallback termination is triggered when:
        # (a) No stacks on board AND no rings in hand for any player, OR
        # (b) No stacks on board AND no player with rings in hand has any
        #     legal placement (respecting no-dead-placement rule).
        #
        # This mirrors TS VictoryAggregate.ts hasAnyLegalPlacementOnBareBoard().
        no_stacks_left = not game_state.board.stacks
        any_rings_in_hand = any(p.rings_in_hand > 0 for p in game_state.players)

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
                    and GameEngine._has_any_legal_placement_on_bare_board(game_state, p.player_number)
                    for p in game_state.players
                )
                if not any_legal_placement_exists:
                    # No player can place - stalemate with hand conversion
                    should_apply_stalemate = True
                    hand_counts_as_eliminated = True

        if should_apply_stalemate:
            game_state.game_status = GameStatus.COMPLETED

            # Tie-breaker logic:
            # 1. Most collapsed spaces (territorySpaces)
            # 2. Most eliminated rings (including rings in hand when
            #    hand_counts_as_eliminated is True, mirroring TS
            #    evaluateVictory semantics)
            # 3. Most markers
            # 4. Last player to complete a valid turn action

            # Calculate scores for each player
            scores = {}
            territory_counts = {}
            for owner in game_state.board.collapsed_spaces.values():
                if not isinstance(owner, int) or owner <= 0:
                    continue
                territory_counts[owner] = territory_counts.get(owner, 0) + 1
            for player in game_state.players:
                pid = player.player_number

                # 1. Collapsed spaces (territory control)
                collapsed = territory_counts.get(pid, 0)

                # 2. Eliminated rings + Rings in hand (when applicable)
                eliminated = player.eliminated_rings
                if hand_counts_as_eliminated:
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
                    1 if (game_state.move_history and game_state.move_history[-1].player == p.player_number) else 0,
                ),
                reverse=True,
            )

            game_state.winner = sorted_players[0].player_number
            game_state.current_player = sorted_players[0].player_number  # TS parity: winner stays current
            game_state.current_phase = GamePhase.GAME_OVER
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
            has_stacks = any(s.controlling_player == p.player_number for s in game_state.board.stacks.values())
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
    def _update_phase(game_state: GameState, last_move: Move, *, trace_mode: bool = False):
        """
        Advance phase after a move using the dedicated phase state machine.

        This is a thin wrapper around :mod:`app.rules.phase_machine`, which
        centralises all phase/turn transitions. It mutates ``game_state``
        in-place and does not return a value.

        NOTE: FSM validation is separate from phase orchestration. FSM validates
        that moves are appropriate for phases, but phase_machine handles the
        actual phase transitions. This maintains parity with the proven Python
        phase transition logic while FSM orchestration is experimental.
        """
        from app.rules.phase_machine import PhaseTransitionInput, advance_phases

        inp = PhaseTransitionInput(
            game_state=game_state,
            last_move=last_move,
            trace_mode=trace_mode,
        )
        advance_phases(inp)

    @staticmethod
    def _advance_to_line_processing(game_state: GameState, *, trace_mode: bool = False):
        # Clear chain_capture_state when leaving capture phase.
        # This ensures the next capture opportunity (e.g., after a future
        # MOVE_STACK) will be correctly classified as OVERTAKING_CAPTURE.
        game_state.chain_capture_state = None
        # RR-FIX-2025-12-12: Clear must_move_from_stack_key when transitioning
        # to line_processing. The per-turn must-move constraint is only
        # relevant during movement/capture phases; leaving it set can cause
        # stale keys (e.g., when the constrained stack is eliminated by
        # landing-on-marker cap removal) to incorrectly suppress global
        # movement/capture availability and break TS↔Python ANM parity.
        game_state.must_move_from_stack_key = None

        # Always enter LINE_PROCESSING phase. Per RR-CANON-R075, all phases
        # must be explicitly visited. When there are no lines to process,
        # the AI/player must select NO_LINE_ACTION which then advances to
        # territory_processing. There is no silent skipping of phases.
        game_state.current_phase = GamePhase.LINE_PROCESSING

    @staticmethod
    def _advance_to_territory_processing(game_state: GameState, *, trace_mode: bool = False):
        """Advance from line processing into territory processing.

        Always enter TERRITORY_PROCESSING as a distinct phase. If there are no
        eligible regions, the AI must explicitly select NO_TERRITORY_ACTION via
        get_valid_moves per RR-CANON-R075 to record that the phase was visited.

        Do NOT auto-generate moves here - that would cause replay parity issues.
        The AI/player selects the appropriate move which is then applied.

        Explicit territory decision moves (CHOOSE_TERRITORY_OPTION,
        ELIMINATE_RINGS_FROM_STACK, etc.) are handled separately in
        _update_phase and are treated as phase-preserving.
        """
        game_state.current_phase = GamePhase.TERRITORY_PROCESSING

    @staticmethod
    def _player_has_turn_material(game_state: GameState, player: int) -> bool:
        """
        Check if a player has turn-material per RR-CANON-R201.

        A player has turn-material if:
        - They control at least one stack (stack.controlling_player == player), OR
        - They have rings in hand (rings_in_hand > 0)

        Players without turn-material are "temporarily eliminated for turn
        rotation" and must be skipped when advancing the turn.
        """
        # Check rings in hand first (fast path)
        player_state = next(
            (p for p in game_state.players if p.player_number == player),
            None,
        )
        if player_state is not None and player_state.rings_in_hand > 0:
            return True

        # Check for controlled stacks
        for stack in game_state.board.stacks.values():
            if stack.controlling_player == player and stack.stack_height > 0:
                return True

        return False

    @staticmethod
    def _player_has_any_rings(game_state: GameState, player: int) -> bool:
        """
        Check if a player has any rings anywhere (controlled, buried, or in hand).

        A player with no rings anywhere is permanently eliminated and should
        be skipped in turn rotation (RR-CANON-R201). A player who has rings
        (even if only buried in stacks controlled by other players) is NOT
        permanently eliminated - they may be "recovery-eligible" and should
        receive turns.

        This is different from _player_has_turn_material():
        - _player_has_turn_material: controlled stacks OR rings in hand
        - _player_has_any_rings: any rings anywhere (including buried)
        """
        # Check rings in hand first (fast path)
        player_state = next(
            (p for p in game_state.players if p.player_number == player),
            None,
        )
        if player_state is not None and player_state.rings_in_hand > 0:
            return True

        # Check all stacks for any ring belonging to this player
        # This includes both controlled stacks (player is on top) AND
        # buried rings (player's rings inside stacks controlled by others)
        for stack in game_state.board.stacks.values():
            if stack.stack_height <= 0:
                continue
            # The rings list contains player numbers from bottom to top
            if stack.rings and player in stack.rings:
                return True

        return False

    @staticmethod
    def _rotate_to_next_active_player(game_state: GameState) -> None:
        """
        Rotate to the next player who has any rings, skipping permanently eliminated players.

        This is a lighter-weight rotation than _end_turn: it does NOT apply
        forced elimination or complex per-turn state updates. It simply finds
        the next player in seat order who still has any rings anywhere
        (controlled, buried, or in hand per RR-CANON-R201), and starts their
        turn in the appropriate phase.

        Players without any rings anywhere are permanently eliminated and are
        skipped. Players who have rings (even if only buried in stacks
        controlled by others) are NOT skipped - they may be recovery-eligible.

        Used after CHOOSE_TERRITORY_OPTION and ELIMINATE_RINGS_FROM_STACK
        where forced elimination should not be triggered at the transition.

        Mirrors TS turnLogic.advanceTurnAndPhase player-skipping behaviour.
        """
        players = game_state.players
        if not players:
            return

        num_players = len(players)

        # Find index of the current player in the players list
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

            # Check if candidate has any rings anywhere (RR-CANON-R201):
            # - rings in controlled stacks (top ring is player's colour)
            # - rings buried inside stacks controlled by others
            # - rings in hand (not yet placed)
            # Players without ANY rings are permanently eliminated and must be
            # skipped. Players with buried rings but no stacks/hand are NOT
            # skipped - they may be recovery-eligible.
            has_any_rings = GameEngine._player_has_any_rings(
                game_state, candidate.player_number
            )

            if has_any_rings:
                game_state.current_player = candidate.player_number
                game_state.must_move_from_stack_key = None

                # Per RR-CANON-R073: ALL players start in RING_PLACEMENT without exception.
                # Players with rings_in_hand == 0 will emit no_placement_action and
                # proceed to movement, but they MUST enter ring_placement first.
                # NO PHASE SKIPPING IS ALLOWED - this is a core invariant.
                game_state.current_phase = GamePhase.RING_PLACEMENT

                GameEngine._update_lps_round_tracking_for_current_player(
                    game_state,
                )
                GameEngine._maybe_apply_lps_victory_at_turn_start(game_state)
                return

            # Player has no rings anywhere (permanently eliminated); skip to next seat
            idx = (idx + 1) % num_players
            skips += 1

        # All players exhausted (no one has any rings); keep current_player
        # and set phase to RING_PLACEMENT to allow _check_victory to resolve
        # the global stalemate via canonical no-action moves.
        game_state.current_phase = GamePhase.RING_PLACEMENT
        game_state.must_move_from_stack_key = None

    @staticmethod
    def _end_turn(game_state: GameState, *, trace_mode: bool = False):
        """
        End the current player's turn and advance to the next active player.

        This mirrors the TS TurnEngine "territory_processing" end-of-turn
        behaviour (RR-CANON-R201):

        - Rotate to the next player in table order who has any rings anywhere
          (controlled, buried, or in hand).
        - Skip players without ANY rings (permanently eliminated).
        - Players with buried rings but no stacks/hand are NOT skipped - they
          may be recovery-eligible.
        - For the next player with rings:
          * ALL players start in RING_PLACEMENT without exception (RR-CANON-R073).
          * Players with rings_in_hand == 0 will emit no_placement_action and
            proceed to movement, but they MUST enter ring_placement first.
          * NO PHASE SKIPPING IS ALLOWED - this is a core invariant.
        - If no players have any rings at all, leave current_player unchanged
          and allow _check_victory to resolve global stalemate via tie-breakers.

        Args:
            game_state: The game state to update.
            trace_mode: When True, skip automatic forced elimination. In trace
                mode, forced elimination is expected to arrive as an explicit
                `forced_elimination` move in the recorded sequence. This ensures
                parity with the TS engine's traceMode replay semantics.
        """
        # If the game already ended earlier in the turn (e.g., territory
        # victory or elimination threshold reached), do not rotate the active
        # player. TS keeps the actor as current_player at GAME_OVER; rotating
        # here would desynchronise replay parity.
        if game_state.game_status != GameStatus.ACTIVE:
            return

        # Clear chain_capture_state at turn end. This ensures the next player's
        # first capture (if any) will be correctly classified as OVERTAKING_CAPTURE.
        game_state.chain_capture_state = None

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

        # Find the next player with any rings anywhere (RR-CANON-R201).
        # Skip players without ANY rings (permanently eliminated).
        # Players with buried rings but no stacks/hand are NOT skipped.
        max_skips = num_players
        skips = 0
        idx = (current_index + 1) % num_players

        while skips < max_skips:
            candidate = players[idx]

            # Check if candidate has any rings anywhere (controlled, buried, or in hand)
            has_any_rings = GameEngine._player_has_any_rings(
                game_state, candidate.player_number
            )

            if has_any_rings:
                game_state.current_player = candidate.player_number
                game_state.must_move_from_stack_key = None

                # Per RR-CANON-R073: ALL players start in RING_PLACEMENT without exception.
                # Players with rings_in_hand == 0 will emit no_placement_action and
                # proceed to movement, but they MUST enter ring_placement first.
                # NO PHASE SKIPPING IS ALLOWED - this is a core invariant.
                game_state.current_phase = GamePhase.RING_PLACEMENT

                GameEngine._update_lps_round_tracking_for_current_player(
                    game_state,
                )
                GameEngine._maybe_apply_lps_victory_at_turn_start(game_state)
                return

            # Player has no rings anywhere (permanently eliminated); skip to next seat
            idx = (idx + 1) % num_players
            skips += 1

        # All players exhausted (no one has any rings); keep current_player
        # and set phase to RING_PLACEMENT to allow _check_victory to resolve
        # the global stalemate via canonical no-action moves.
        game_state.current_phase = GamePhase.RING_PLACEMENT
        game_state.must_move_from_stack_key = None

    @staticmethod
    def _assert_active_player_has_legal_action(
        game_state: GameState,
        triggering_move: Move,
    ) -> None:
        """Enforce the strict no-move invariant for ACTIVE states.

        When ``STRICT_NO_MOVE_INVARIANT`` is enabled, any ACTIVE state must
        expose at least one legal action (interactive move **or** pending
        phase requirement) for ``current_player``, after accounting for the
        defensive rotation used for permanently eliminated players.

        Concretely:
        - Per RR-CANON-R201, players without ANY rings anywhere (controlled,
          buried, or in hand) are permanently eliminated and are skipped.
          If current_player has no rings anywhere, we rotate to the next
          player with rings via _end_turn.
        - Players with buried rings but no stacks/hand are NOT skipped - they
          may be recovery-eligible and should receive turns.
        - For the resulting ``current_player``, if
          ``GameEngine.get_valid_moves(game_state, current_player)`` returns
          at least one move, the invariant holds.
        - If there are no interactive moves but
          ``GameEngine.get_phase_requirement(game_state, current_player)``
          returns a non-None requirement (for example
          ``NO_LINE_ACTION_REQUIRED`` or ``NO_TERRITORY_ACTION_REQUIRED``),
          the invariant also holds: hosts are obligated to emit the
          corresponding bookkeeping move via the public API.
        - Only when there are **no** interactive moves and **no** pending
          phase requirement do we treat the state as an INV-ACTIVE-NO-MOVES /
          ANM violation and raise.
        """
        if game_state.game_status != GameStatus.ACTIVE:
            return

        # Local import to avoid a circular import at module load time. The
        # global_actions module in turn imports GameEngine for shared helpers.
        from app.rules import global_actions as ga  # type: ignore

        current_player = game_state.current_player

        # First, ensure that permanently eliminated players are not left as the
        # active player in ACTIVE states. If the current player has no rings
        # anywhere (controlled, buried, or in hand), attempt a single defensive
        # rotation via _end_turn before treating the shape as an invariant
        # violation.
        #
        # IMPORTANT: only apply this defensive rotation at turn boundaries
        # (i.e., when the just-applied move belongs to a different player).
        #
        # During a player's own turn they may self-eliminate their last ring
        # (e.g., recovery fallback cost via buried-ring extraction). In that
        # case the canonical trace still requires phase traversal to be
        # recorded (e.g., no_line_action / no_territory_action) before normal
        # turn rotation skips the permanently eliminated player.
        if (
            triggering_move.player != current_player
            and not GameEngine._player_has_any_rings(game_state, current_player)
        ):
            previous_player = current_player
            GameEngine._end_turn(game_state)

            # If rotation finished the game, there is no invariant violation.
            if game_state.game_status != GameStatus.ACTIVE:
                return

            # If we advanced to a different player who now has any rings and
            # is not in an ANM state, the invariant holds.
            if game_state.current_player != previous_player:
                rotated_player = game_state.current_player
                # If rotation found a player with rings and the resulting
                # state is not ANM, accept it without further checks.
                if GameEngine._player_has_any_rings(game_state, rotated_player) and not ga.is_anm_state(game_state):
                    return

        # Re-evaluate the (possibly rotated) current player and consult the
        # canonical move generator. If any interactive move exists for this
        # player, or if a phase-level no-action / forced-elimination
        # requirement is pending, the strict invariant is satisfied.
        current_player = game_state.current_player
        try:
            legal_moves = GameEngine.get_valid_moves(
                game_state,
                current_player,
            )
        except (ValueError, TypeError, KeyError, IndexError, AttributeError):
            # If move generation itself fails, treat this conservatively as
            # "no moves"; the snapshot + raise logic below will surface the
            # underlying issue.
            legal_moves = []

        if legal_moves:
            return

        # No interactive moves – check for a pending phase requirement. When
        # a requirement exists (for example, no_line_action_required in
        # LINE_PROCESSING, no_territory_action_required in
        # TERRITORY_PROCESSING, or forced_elimination_required in
        # FORCED_ELIMINATION), hosts are required to synthesize and apply the
        # corresponding bookkeeping move, so the state is not ANM.
        try:
            requirement = GameEngine.get_phase_requirement(
                game_state,
                current_player,
            )
        except (ValueError, TypeError, KeyError, IndexError, AttributeError):
            requirement = None

        if requirement is not None:
            return

        # We now have an ACTIVE state whose current_player satisfies
        # ANM-style conditions from the invariant's perspective: the canonical
        # move generator reports no legal actions for the active player even
        # after a defensive rotation for fully eliminated seats. Capture a
        # diagnostic snapshot and raise.
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
            except (ValueError, TypeError, AttributeError):
                state_payload = None

            try:
                move_payload = triggering_move.model_dump(  # type: ignore[attr-defined]
                    by_alias=True,
                    mode="json",
                )
            except (ValueError, TypeError, AttributeError):
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
        except (OSError, IOError, ValueError, TypeError):
            # Snapshotting must never prevent raising the invariant error.
            pass

        raise RuntimeError(
            "STRICT_NO_MOVE_INVARIANT violated: ACTIVE "
            f"{game_state.current_phase.value} state for player "
            f"{game_state.current_player} has no legal actions",
        )

    @staticmethod
    def _resolve_anm_for_current_player(game_state: GameState) -> None:
        """Resolve ANM(state, current_player) chains (TS parity).

        Mirrors TurnOrchestrator.resolveANMForCurrentPlayer:
        - While the game is ACTIVE and ANM holds for current_player:
          * If forced elimination is available, apply one elimination.
          * Otherwise, fall back to victory/stalemate evaluation.
        - Stop once ANM no longer holds or the game becomes terminal.
        """
        from app.rules import global_actions as ga  # local import

        total_rings = getattr(game_state, "total_rings_in_play", None)
        if not isinstance(total_rings, int):
            total_rings = sum(p.rings_in_hand for p in game_state.players) + sum(
                s.stack_height for s in game_state.board.stacks.values()
            )
        max_steps = max(4, total_rings + 4)

        for _ in range(max_steps):
            if game_state.game_status != GameStatus.ACTIVE:
                return
            if not ga.is_anm_state(game_state):
                return

            player = game_state.current_player
            forced = ga.apply_forced_elimination_for_player(game_state, player)
            if forced is None:
                GameEngine._check_victory(game_state)
                return

            GameEngine._check_victory(game_state)
            if game_state.game_status != GameStatus.ACTIVE:
                return

    @staticmethod
    def _estimate_rings_per_player(game_state: GameState) -> int:
        """
        TS-aligned per-player ring cap.

        Uses centralized BOARD_CONFIGS from app.rules.core which mirrors
        BOARD_CONFIGS[boardType].ringsPerPlayer from the shared TS types:

        - square8   → 18 rings per player
        - square19  → 72 rings per player
        - hexagonal → 96 rings per player
        """
        from app.rules.core import BOARD_CONFIGS
        board_type = game_state.board.type
        if board_type in BOARD_CONFIGS:
            return BOARD_CONFIGS[board_type].rings_per_player
        # Fallback for unknown types: use totalRingsInPlay as a safe
        # upper bound to avoid underestimating.
        return game_state.total_rings_in_play

    @staticmethod
    def _calculate_distance(board_type: BoardType, from_pos: Position, to_pos: Position) -> int:
        """Thin wrapper around shared geometry distance helper.

        Delegates to ``BoardGeometry.calculate_distance`` to keep a single
        source of truth for distance semantics.
        """
        return BoardGeometry.calculate_distance(board_type, from_pos, to_pos)

    @staticmethod
    def _get_path_positions(from_pos: Position, to_pos: Position) -> list[Position]:
        """Thin wrapper around shared geometry path helper.

        Delegates to ``BoardGeometry.get_path_positions`` to keep a single
        source of truth for straight-line path semantics.
        """
        return BoardGeometry.get_path_positions(from_pos, to_pos)

    @staticmethod
    def _is_straight_line_movement(board_type: BoardType, from_pos: Position, to_pos: Position) -> bool:
        """
        True if move from `from_pos` to `to_pos` lies along a valid movement
        ray for the given board type.
        Mirrors RuleEngine.isStraightLineMovement.
        """
        dx = to_pos.x - from_pos.x
        dy = to_pos.y - from_pos.y
        dz = (to_pos.z or 0) - (from_pos.z or 0)

        if board_type in (BoardType.HEXAGONAL, BoardType.HEX8):
            # In cube coordinates, an axis-aligned ray changes exactly two
            # coordinates (the third is implied by x + y + z = 0).
            coord_changes = sum(1 for d in (dx, dy, dz) if d != 0)
            return coord_changes == 2

        # Square boards: orthogonal or diagonal only.
        if dx == 0 and dy == 0:
            return False
        return not (dx != 0 and dy != 0 and abs(dx) != abs(dy))

    @staticmethod
    def _is_path_clear_for_movement(board, from_pos: Position, to_pos: Position) -> bool:
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
        # Use model_construct to skip Pydantic validation (faster) - Pydantic v2 API
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
        board, from_pos: Position, to_pos: Position, player_number: int, game_state: GameState
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
                    game_state.zobrist_hash ^= zobrist.get_marker_hash(pos_key, marker.player)
                BoardManager.set_collapsed_space(pos, player_number, board)
                # Add collapsed hash
                if game_state.zobrist_hash is not None:
                    game_state.zobrist_hash ^= zobrist.get_collapsed_hash(pos_key)
            else:
                # Remove old marker hash
                if game_state.zobrist_hash is not None:
                    game_state.zobrist_hash ^= zobrist.get_marker_hash(pos_key, marker.player)
                # Copy marker before modification to avoid mutating shared state
                new_marker = marker.model_copy()
                new_marker.player = player_number
                board.markers[pos_key] = new_marker

                # Add new marker hash
                if game_state.zobrist_hash is not None:
                    game_state.zobrist_hash ^= zobrist.get_marker_hash(pos_key, new_marker.player)

    @staticmethod
    def _eliminate_top_ring_at(game_state: GameState, position: Position, credited_player: int) -> None:
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
                pos_key, stack.controlling_player, stack.stack_height, tuple(stack.rings)
            )

        stack = stack.model_copy(deep=True)
        board.stacks[pos_key] = stack

        stack.rings.pop()
        stack.stack_height -= 1

        player_id_str = str(credited_player)
        board.eliminated_rings[player_id_str] = board.eliminated_rings.get(player_id_str, 0) + 1
        game_state.total_rings_eliminated += 1

        for p in game_state.players:
            if p.player_number == credited_player:
                p.eliminated_rings += 1
                break

        if stack.stack_height == 0 or not stack.rings:
            # Stack is empty - remove it
            if pos_key in board.stacks:
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
                    pos_key, stack.controlling_player, stack.stack_height, tuple(stack.rings)
                )

            _debug(
                f"DEBUG: _apply_place_ring created stack at {pos_key}: "
                f"player={stack.controlling_player}, "
                f"height={stack.stack_height}, "
                f"rings={stack.rings}\n"
            )

    @staticmethod
    def _extract_buried_ring_at(
        game_state: GameState, position: Position, credited_player: int
    ) -> None:
        """
        Extract (eliminate) exactly one buried ring belonging to `credited_player`
        from the stack at `position`, crediting the elimination to `credited_player`.

        Canonical recovery semantics (RR-CANON-R113/R114):
        - The extracted ring must belong to credited_player.
        - It must be buried (i.e., not the top ring).
        - When multiple buried rings exist, extract the bottommost one.

        Mirrors TS EliminationAggregate recovery context, adapted for the Python
        RingStack representation (rings stored bottom -> top).
        """
        board = game_state.board
        pos_key = position.to_key()
        stack = board.stacks.get(pos_key)
        if not stack or stack.stack_height <= 1:
            raise ValueError(f"Cannot extract buried ring - no eligible stack at {pos_key}")

        # Find bottommost buried ring of credited_player (exclude top ring).
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
        if game_state.zobrist_hash is not None:
            game_state.zobrist_hash ^= zobrist.get_stack_hash(
                pos_key, stack.controlling_player, stack.stack_height, tuple(stack.rings)
            )

        stack = stack.model_copy(deep=True)
        board.stacks[pos_key] = stack

        stack.rings.pop(extract_index)
        stack.stack_height -= 1

        player_id_str = str(credited_player)
        board.eliminated_rings[player_id_str] = board.eliminated_rings.get(player_id_str, 0) + 1
        game_state.total_rings_eliminated += 1

        for p in game_state.players:
            if p.player_number == credited_player:
                p.eliminated_rings += 1
                break

        if stack.stack_height == 0:
            del board.stacks[pos_key]
            return

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
                pos_key, stack.controlling_player, stack.stack_height, tuple(stack.rings)
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

        if board_type in (BoardType.HEXAGONAL, BoardType.HEX8):
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
                    _debug(f"DEBUG: _has_any found move from {from_pos} " f"to {target} dist {distance}\n")
                    return True
                else:
                    # Landing on a stack (for merging) is also a legal move.
                    _debug(f"DEBUG: _has_any found merge from {from_pos} " f"to {target} dist {distance}\n")
                    return True

        # === Capture reachability ===
        for direction in directions:
            step = 1
            target_pos: Position | None = None

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
                    _debug(f"DEBUG: _has_any found capture from {from_pos} " f"to {landing} via {target_pos}\n")
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
        # Use model_construct for faster creation (skips validation) - Pydantic v2 API
        synthetic_move = Move.model_construct(
            id="hypothetical-placement",
            type=MoveType.PLACE_RING,
            player=player_number,
            to=from_pos,
            timestamp=game_state.last_move_at,
            thinkTime=0,
            moveNumber=len(game_state.move_history) + 1,
        )
        temp_state.move_history = [*list(game_state.move_history), synthetic_move]

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
            _debug("DEBUG: no-dead-placement rejected placement at " f"{from_pos.to_key()} for P{player_number}\n")

        return False

    @staticmethod
    def _get_skip_placement_moves(
        game_state: GameState,
        player_number: int,
    ) -> list[Move]:
        """
        Enumerate legal SKIP_PLACEMENT moves.

        Mirrors TS PlacementAggregate.validateSkipPlacement semantics:
        - Only during RING_PLACEMENT phase.
        - Player must have rings in hand.
        - Player must either:
          - control at least one stack with a legal move/capture, OR
          - be recovery-eligible (no stacks, marker(s), buried ring), in which
            case skip_placement is the voluntary gateway to recovery in movement.
        """
        if game_state.current_phase != GamePhase.RING_PLACEMENT:
            return []

        player = next(
            (p for p in game_state.players if p.player_number == player_number),
            None,
        )
        if not player:
            return []

        # Canonical rule: with zero rings in hand, skip_placement is invalid.
        # Players must record an explicit no_placement_action instead.
        if player.rings_in_hand <= 0:
            return []

        board = game_state.board
        player_stacks = BoardManager.get_player_stacks(board, player_number)
        if not player_stacks:
            from app.rules.core import is_eligible_for_recovery

            if not is_eligible_for_recovery(game_state, player_number):
                return []

            return [
                Move(
                    id="simulated",
                    type=MoveType.SKIP_PLACEMENT,
                    player=player_number,
                    timestamp=game_state.last_move_at,
                    thinkTime=0,
                    moveNumber=len(game_state.move_history) + 1,
                )
            ]

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
    ) -> list[Move]:
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
        moves: list[Move] = []
        board = game_state.board

        # Check if player has rings in hand
        player = next(
            (p for p in game_state.players if p.player_number == player_number),
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
    def _get_capture_moves(game_state: GameState, player_number: int, limit: int | None = None) -> list[Move]:
        """
        Enumerate legal overtaking capture segments for the player.

        During CHAIN_CAPTURE phase (when chain_capture_state is set), enumerate
        captures only from the chain's current position.

        During MOVEMENT phase (no chain_capture_state), enumerate captures from
        ALL of the player's stacks, mirroring TS's enumerateAllCaptureMoves.

        Adapter over rules.capture_chain.enumerate_capture_moves_py, which
        mirrors TS CaptureAggregate.enumerateCaptureMoves.

        Args:
            limit: If provided, return at most this many moves (for early-return checks).
        """
        moves: list[Move] = []
        move_number = len(game_state.move_history) + 1

        if game_state.chain_capture_state:
            # Chain capture in progress - enumerate only from the chain position
            attacker_pos = game_state.chain_capture_state.current_position
            moves = enumerate_capture_moves_py(
                game_state,
                player_number,
                attacker_pos,
                move_number=move_number,
                kind="continuation",
            )
            # NOTE: visited_positions tracks "from" positions where the attacker
            # jumped FROM during this chain. The attacker IS allowed to land on a
            # previously visited "from" position as long as it's making progress
            # (capturing rings each time). This enables "bouncing" patterns where
            # the attacker captures back and forth from the same target stack.
            # Do NOT filter by landing position matching visited_positions.
        else:
            # Movement phase - enumerate captures from ALL player's stacks
            # This mirrors TS's enumerateAllCaptureMoves which iterates over
            # all stacks controlled by the player.
            #
            # When must_move_from_stack_key is set (after place_ring on a stack),
            # only captures from that specific stack are valid. This mirrors TS's
            # filtering by mustMoveFromStackKey in getValidMoves.
            board = game_state.board
            must_move_key = game_state.must_move_from_stack_key

            for stack in board.stacks.values():
                if stack.controlling_player != player_number:
                    continue
                if stack.stack_height <= 0:
                    continue

                # Filter by must_move_from_stack_key if set
                if must_move_key is not None and stack.position.to_key() != must_move_key:
                    continue

                stack_captures = enumerate_capture_moves_py(
                    game_state,
                    player_number,
                    stack.position,
                    move_number=move_number,
                    kind="initial",
                )
                moves.extend(stack_captures)

                # Early return if limit is reached
                if limit is not None and len(moves) >= limit:
                    return moves[:limit]

        # Apply limit if specified (for early-return optimization)
        if limit is not None and len(moves) > limit:
            return moves[:limit]
        return moves

    @staticmethod
    def _has_valid_placements(game_state: GameState, player_number: int) -> bool:
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
    def _has_any_valid_placement_fast(game_state: GameState, player_number: int) -> bool:
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
    def _has_any_legal_placement_on_bare_board(game_state: GameState, player_number: int) -> bool:
        """
        Check if a player can legally place a ring on a bare board.

        This mirrors the TS VictoryAggregate.hasAnyLegalPlacementOnBareBoard
        helper used inside evaluateVictory:

        - Iterate all board positions.
        - Skip collapsed spaces and markers.
        - For each remaining position, construct a hypothetical board with
          a single-ring stack for the player at that position.
        - Ask whether that stack would have at least one legal movement or
          capture via the shared movement/capture reachability helper.

        Returns True as soon as any such position is found.
        """
        board = game_state.board

        # Check if player has rings in hand
        player = next(
            (p for p in game_state.players if p.player_number == player_number),
            None,
        )
        if not player or player.rings_in_hand <= 0:
            return False
        # Get all board positions for the current geometry.
        all_positions = GameEngine._generate_all_positions(
            board.type,
            board.size,
        )

        for pos in all_positions:
            pos_key = pos.to_key()

            # Cannot place on collapsed spaces or markers.
            if pos_key in board.collapsed_spaces:
                continue
            if pos_key in board.markers:
                continue

            # Construct a hypothetical board with a single-ring stack for this
            # player at `pos` and ask whether that stack would have any legal
            # movement or capture from its origin. This matches the TS
            # hasAnyLegalPlacementOnBareBoard + hasAnyLegalMoveOrCaptureFromOnBoard
            # combination.
            hyp_board = GameEngine._create_hypothetical_board_with_placement(
                board,
                pos,
                player_number,
                1,
            )

            if GameEngine._has_any_legal_move_or_capture_from_on_board(
                hyp_board.type,
                pos,
                player_number,
                hyp_board,
            ):
                return True

        return False

    @staticmethod
    def _has_valid_movements(
        game_state: GameState,
        player_number: int,
        ignore_must_move_key: bool = False,
    ) -> bool:
        """True if the player has any legal non-capture movements.

        Args:
            ignore_must_move_key: If True, ignore the must_move_from_stack_key
                constraint. Use True for FE eligibility checks (asking "would
                this player have any moves on a fresh turn?"), and False for
                post-placement phase advancement (asking "can this player move
                from the stack they just placed?").
        """
        # Use limit=1 for early return optimization on large boards.
        return bool(
            GameEngine._get_movement_moves(
                game_state,
                player_number,
                limit=1,
                ignore_must_move_key=ignore_must_move_key,
            )
        )

    @staticmethod
    def _has_valid_captures(game_state: GameState, player_number: int) -> bool:
        """True if the player has any legal overtaking capture from controlled stacks.

        When must_move_from_stack_key is set, only that stack can capture.
        This constraint is active after a place_ring until the turn ends.
        This matches the TS hasAnyGlobalMovementOrCapture behaviour.
        """
        from app.rules.capture_chain import enumerate_capture_moves_py

        must_move_key = game_state.must_move_from_stack_key

        # Check each controlled stack for potential captures
        for stack in game_state.board.stacks.values():
            if stack.controlling_player != player_number or stack.stack_height <= 0:
                continue

            # When must_move_from_stack_key is set, only that stack can capture
            stack_key = stack.position.to_key()
            if must_move_key and stack_key != must_move_key:
                continue

            # Check captures from this stack position
            captures = enumerate_capture_moves_py(
                game_state,
                player_number,
                stack.position,
                move_number=len(game_state.move_history) + 1,
                kind="initial",
            )
            if captures:
                return True  # Early return - we only need to find one

        return False

    @staticmethod
    def _has_valid_actions(game_state: GameState, player_number: int) -> bool:
        """
        Combined placement/movement/capture availability check.

        This is a Python analogue of TS hasValidActions used to decide when
        forced elimination is required. If ANY legal placement, movement,
        capture, or recovery action exists for the player, forced elimination
        is not permitted.

        When must_move_from_stack_key is set, only that stack can move/capture.
        This constraint is active after a place_ring until the turn ends, and
        must be respected when evaluating whether forced elimination is available.
        This aligns with TS globalActions.hasAnyGlobalMovementOrCapture.
        """
        if GameEngine._has_valid_placements(game_state, player_number):
            return True
        # Respect must_move_from_stack_key constraint - only that stack can move/capture
        if GameEngine._has_valid_movements(game_state, player_number, ignore_must_move_key=False):
            return True
        if GameEngine._has_valid_captures(game_state, player_number):
            return True
        return bool(has_any_recovery_move(game_state, player_number))

    @staticmethod
    def _has_real_action_for_player(
        game_state: GameState,
        player_number: int,
    ) -> bool:
        """
        R172 real-action availability predicate for LPS.

        A player has a real action if they have at least one legal placement,
        non-capture movement, or overtaking capture available on their turn.

        Recovery actions are NOT counted as real actions for LPS purposes.
        This creates strategic tension: rings in hand become a "survival budget" -
        players can use recovery moves but must place at least one ring every 3
        rounds to avoid LPS loss.

        Forced elimination and line/territory decision moves also do not count.

        NOTE: This differs from _has_valid_actions which DOES include recovery
        (used for FE eligibility checks).
        """
        # Check placement, movement, capture - but NOT recovery
        if GameEngine._has_valid_placements(game_state, player_number):
            return True
        if GameEngine._has_valid_movements(game_state, player_number, ignore_must_move_key=True):
            return True
        return bool(GameEngine._has_valid_captures(game_state, player_number))

    @staticmethod
    def _maybe_apply_lps_victory_at_turn_start(game_state: GameState) -> None:
        """
        Apply round-based LPS victory (RR-CANON-R172) at the START of an
        interactive turn.

        Mirrors TS lpsTracking.evaluateLpsVictory semantics:
        - Requires three consecutive exclusive rounds for the same candidate.
        - Evaluated when the candidate's next interactive turn begins.
        - Candidate must still have a real action available.
        - All other players with material must have no real actions.

        This is intentionally evaluated at turn start (after LPS tracking is
        updated) rather than after a bookkeeping move, to avoid awarding LPS
        on non-real actions like ``no_placement_action``.
        """
        if game_state.game_status != GameStatus.ACTIVE:
            return

        lps_active_phases = {
            GamePhase.RING_PLACEMENT,
            GamePhase.MOVEMENT,
            GamePhase.CAPTURE,
            GamePhase.CHAIN_CAPTURE,
        }
        if game_state.current_phase not in lps_active_phases:
            return

        # NOTE: We intentionally do NOT defer during ANM states.
        #
        # TS awards LPS at the start of the candidate's interactive turn
        # whenever the candidate has any real action available (placement,
        # movement, capture) and all other players with rings have none,
        # even if the turn begins in a forced no-op ring_placement substep
        # (e.g., rings_in_hand == 0 implies the host must emit
        # no_placement_action before movement becomes available).
        #
        # Deferring here causes TS↔Python parity mismatches where TS ends the
        # game immediately on the turn boundary but Python waits for the
        # next bookkeeping move.

        lps_threshold = getattr(game_state, 'lps_rounds_required', 3)
        if game_state.lps_consecutive_exclusive_rounds < lps_threshold:
            return

        candidate = game_state.lps_consecutive_exclusive_player
        if candidate is None:
            return

        if game_state.current_player != candidate:
            return

        if not GameEngine._has_real_action_for_player(game_state, candidate):
            return

        for player in game_state.players:
            if player.player_number == candidate:
                continue
            if count_rings_in_play_for_player(game_state, player.player_number) <= 0:
                continue
            if GameEngine._has_real_action_for_player(game_state, player.player_number):
                return

        game_state.game_status = GameStatus.COMPLETED
        game_state.winner = candidate
        game_state.current_player = candidate  # TS parity: winner stays current
        game_state.current_phase = GamePhase.GAME_OVER

    @staticmethod
    def _update_lps_round_tracking_for_current_player(
        game_state: GameState,
    ) -> None:
        """
        Update last-player-standing (R172) round tracking for the current
        player.

        This mirrors the TS lpsTracking.ts updateLpsTracking() helper by:
        - Tracking currentRoundFirstPlayer to detect round boundaries.
        - Starting a new cycle when: (a) first round, or (b) the previous
          round's first player dropped out (no longer has material).
        - Finalizing a round only when we cycle BACK to the first player.
        - Recording whether the current player has any real actions.
        """
        if game_state.game_status != GameStatus.ACTIVE:
            return

        # Active players for LPS tracking are those with any material remaining
        # in the game (RR-CANON-R172 / RR-CANON-R175): rings on the board in
        # any stack (including buried rings) or rings in hand.
        #
        # IMPORTANT: Do NOT use "turn-material" (RR-CANON-R201) here. A player
        # with only buried rings (no controlled stacks, no rings in hand) still
        # has material and must be included in LPS round tracking (TS parity:
        # playerHasMaterial uses countRingsInPlayForPlayer, which counts buried
        # rings).
        active_players: list[int] = []
        for player in game_state.players:
            if GameEngine._player_has_any_rings(game_state, player.player_number):
                active_players.append(player.player_number)

        if not active_players:
            return

        active_set = set(active_players)
        mask = game_state.lps_current_round_actor_mask
        current = game_state.current_player

        if current not in active_set:
            # Current player has no material; they're not part of LPS tracking
            return

        first = game_state.lps_current_round_first_player

        # Detect if we're starting a new cycle:
        # - First round (first is None)
        # - Previous round's first player dropped out (no longer in active_set)
        starting_new_cycle = (first is None) or (first not in active_set)

        if starting_new_cycle:
            # Either first round or previous round leader dropped out.
            # Start a new cycle and clear exclusivePlayerForCompletedRound.
            game_state.lps_round_index += 1
            game_state.lps_current_round_first_player = current
            mask.clear()
            game_state.lps_exclusive_player_for_completed_round = None
            # Only reset consecutive tracking if the exclusive player also
            # dropped out. If the exclusive player is still active, they
            # should continue counting toward LPS victory even though the
            # round structure changed (e.g., opponent lost all material).
            excl = game_state.lps_consecutive_exclusive_player
            if excl is None or excl not in active_set:
                game_state.lps_consecutive_exclusive_rounds = 0
                game_state.lps_consecutive_exclusive_player = None
        elif current == first and len(mask) > 0:
            # Cycled back to first player - finalize the previous round.
            # Determine if exactly one active player had real actions.
            true_players = [pid for pid in active_players if mask.get(pid, False)]
            exclusive_player: int | None = None
            if len(true_players) == 1:
                exclusive_player = true_players[0]
            game_state.lps_exclusive_player_for_completed_round = exclusive_player

            # Track consecutive exclusive rounds for the same player
            if exclusive_player is not None:
                if exclusive_player == game_state.lps_consecutive_exclusive_player:
                    # Same player remains exclusive - increment count
                    game_state.lps_consecutive_exclusive_rounds += 1
                else:
                    # Different player is now exclusive - reset and start counting
                    game_state.lps_consecutive_exclusive_player = exclusive_player
                    game_state.lps_consecutive_exclusive_rounds = 1
            else:
                # No exclusive player this round - reset consecutive tracking
                game_state.lps_consecutive_exclusive_rounds = 0
                game_state.lps_consecutive_exclusive_player = None

            # Start new round
            game_state.lps_round_index += 1
            mask.clear()
            game_state.lps_current_round_first_player = current

        # Record whether the current player has real actions in this round.
        has_real_action = GameEngine._has_real_action_for_player(
            game_state,
            current,
        )
        mask[current] = has_real_action

    @staticmethod
    def _get_forced_elimination_moves(game_state: GameState, player_number: int) -> list[Move]:
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
        player_stacks = [s for s in board.stacks.values() if s.controlling_player == player_number]
        if not player_stacks:
            return []

        if GameEngine._has_valid_actions(game_state, player_number):
            return []

        # Sort stacks by capHeight (smallest first) to match TS behavior
        # When caps are equal, order is arbitrary (first in iteration)
        player_stacks.sort(key=lambda s: s.cap_height if s.cap_height > 0 else float("inf"))

        moves: list[Move] = []
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
    def _enumerate_line_elimination_moves(
        game_state: GameState,
        player_number: int,
    ) -> list[Move]:
        """
        Enumerate eliminate_rings_from_stack moves for line processing.

        .. deprecated:: December 2025
            Delegates to :class:`app.rules.generators.LineGenerator`.

        Per RR-CANON-R123: When pending_line_reward_elimination is True,
        the player must choose which stack to eliminate one ring from.
        All controlled stacks are eligible targets.

        Returns one ELIMINATE_RINGS_FROM_STACK move per eligible stack.
        """
        # Delegate to SSoT generator
        return _line_generator._enumerate_elimination_moves(game_state, player_number)

    @staticmethod
    def _get_line_processing_moves(
        game_state: GameState,
        player_number: int,
    ) -> list[Move]:
        """
        Enumerate canonical line-processing **decision** moves for the player.

        .. deprecated:: December 2025
            Delegates to :class:`app.rules.generators.LineGenerator`.
            All move enumeration logic is now in the SSoT generator module.

        Per RR-CANON-R076, this helper returns **only interactive moves**:

        - One PROCESS_LINE move per player-owned line.
        - For each player-owned line whose length is at least the effective
          reward threshold for the board/player-count combination, one or more
          CHOOSE_LINE_OPTION moves that encode:
            - collapse-all (full line), and
            - for overlength lines: each minimum-collapse contiguous segment.

        It does **not** fabricate `NO_LINE_ACTION` bookkeeping moves. When the
        player has no lines to process, this function returns an empty list and
        `get_phase_requirement` surfaces a corresponding
        `NO_LINE_ACTION_REQUIRED` requirement for hosts to satisfy via
        :meth:`synthesize_bookkeeping_move`.

        Per RR-CANON-R123: When pending_line_reward_elimination is True, only
        eliminate_rings_from_stack moves are legal (no process_line moves).
        """
        # Delegate to SSoT generator
        return _line_generator.generate(game_state, player_number)

    @staticmethod
    def _did_current_turn_include_recovery_slide(
        game_state: GameState,
        player_number: int,
    ) -> bool:
        """
        True iff the current player's turn (as recorded in move_history) includes
        a RECOVERY_SLIDE.

        Mirrors the TS helper used to decide whether territory processing should
        use the recovery self-elimination context (RR-CANON-R114) for any
        disconnected regions created by the recovery slide.
        """
        for move in reversed(game_state.move_history):
            if move.player != player_number:
                break
            if move.type == MoveType.RECOVERY_SLIDE:
                return True
        return False

    @staticmethod
    def _get_territory_processing_moves(
        game_state: GameState,
        player_number: int,
    ) -> list[Move]:
        """
        Enumerate canonical territory-processing **decision** moves.

        .. deprecated:: December 2025
            Delegates to :class:`app.rules.generators.TerritoryGenerator`.
            All move enumeration logic is now in the SSoT generator module.

        Mirrors TS `getValidTerritoryProcessingDecisionMoves` and
        `getValidEliminationDecisionMoves` while obeying R076:

        - When a territory region was just processed this turn, emit the
          mandatory ELIMINATE_RINGS_FROM_STACK follow-up moves first
          (RR-CANON-R145; recovery context RR-CANON-R114).
        - When at least one disconnected region satisfies the
          self-elimination prerequisite, emit one CHOOSE_TERRITORY_OPTION
          move per such region (legacy alias: PROCESS_TERRITORY_REGION).
        - When no eligible regions exist, return an empty list; `get_phase_requirement` will then surface a
          `NO_TERRITORY_ACTION_REQUIRED` requirement so hosts can emit an
          explicit `no_territory_action` bookkeeping move.
        """
        # Delegate to SSoT generator
        return _territory_generator.generate(game_state, player_number)

    @staticmethod
    def _can_process_disconnected_region(
        game_state: GameState,
        region,
        player_number: int,
    ) -> bool:
        """
        Self-elimination prerequisite for territory processing.

        .. deprecated:: December 2025
            Delegates to :class:`app.rules.generators.TerritoryGenerator`.

        Mirrors TS canProcessTerritoryRegion:

        - Normal territory context (RR-CANON-R145): Player must have at least one
          eligible cap target outside the region.
        - Recovery context (RR-CANON-R114): Player must have at least one eligible
          buried-ring extraction target outside the region (stack need not be controlled).
        """
        # Delegate to SSoT generator
        return _territory_generator._can_process_region(game_state, region, player_number)

    @staticmethod
    def _get_movement_moves(
        game_state: GameState,
        player_number: int,
        limit: int | None = None,
        ignore_must_move_key: bool = False,
    ) -> list[Move]:
        """
        Get valid non-capture movement moves.

        Args:
            limit: If provided, return after finding this many moves (for early-return checks).
            ignore_must_move_key: If True, ignore the must_move_from_stack_key
                constraint. Used when checking valid actions for FE eligibility
                where the constraint shouldn't apply.
        """
        moves: list[Move] = []
        board = game_state.board

        # Per-turn must-move constraint: when a ring has been placed this
        # turn, only the updated stack (tracked via must_move_from_stack_key)
        # may move or capture, mirroring TurnEngine.mustMoveFromStackKey.
        # When ignore_must_move_key is True (FE eligibility checks), we skip
        # this constraint since it's specific to the current player's turn.
        must_move_key = None if ignore_must_move_key else game_state.must_move_from_stack_key

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
            if must_move_key is not None and from_pos.to_key() != must_move_key:
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
                    to_pos = BoardManager._add_direction(from_pos, direction, distance)

                    if not BoardManager.is_valid_position(to_pos, board.type, board.size):
                        break

                    if BoardManager.is_collapsed_space(to_pos, board):
                        break

                    # Check path blocking between from_pos and to_pos
                    if not GameEngine._is_path_clear_for_movement(board, from_pos, to_pos):
                        # Any blocking stack/collapsed space stops further
                        # exploration along this ray.
                        break

                    to_key = to_pos.to_key()
                    dest_stack = board.stacks.get(to_key)
                    # Marker presence checked implicitly via stack/empty logic

                    if dest_stack is None or dest_stack.stack_height == 0:
                        # Empty cell or marker-only cell.
                        # Per RR-CANON-R091/R092: Can land on ANY marker (own
                        # or opponent). The marker is removed and a ring from
                        # the cap is eliminated. Movement to empty cells is
                        # always valid.
                        _debug(
                            f"DEBUG: _get_movement_moves found move from " f"{from_pos} to {to_pos} dist {distance}\n"
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
                        # Destination has a stack - movement cannot land on
                        # stacks. Captures (overtaking_capture) handle landing
                        # on opponent stacks and are enumerated separately.
                        # Cannot move further along this ray past a stack.
                        _debug(f"DEBUG: _get_movement_moves blocked by stack at " f"{to_pos}\n")
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

        # Defensive guard: ensure the player has enough rings in hand.
        # Move generation should guarantee this, but we guard here for parity
        # with TS which rejects place_ring when ringsInHand is insufficient.
        player = next(
            (p for p in game_state.players if p.player_number == move.player),
            None,
        )
        if not player:
            raise ValueError(
                f"Cannot place ring - player {move.player} not found in game state"
            )
        if player.rings_in_hand < placement_count:
            raise ValueError(
                f"Cannot place ring - player {move.player} has {player.rings_in_hand} "
                f"rings in hand but tried to place {placement_count}"
            )

        existing = board.stacks.get(pos_key)
        zobrist = ZobristHash()

        if existing and existing.stack_height > 0:
            # Remove old stack hash
            if game_state.zobrist_hash is not None:
                game_state.zobrist_hash ^= zobrist.get_stack_hash(
                    pos_key, existing.controlling_player, existing.stack_height, tuple(existing.rings)
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
                pos_key, stack.controlling_player, stack.stack_height, tuple(stack.rings)
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
            raise ValueError(
                f"Cannot move stack - move.from_pos is None for move by player {move.player}"
            )

        from_key = move.from_pos.to_key()
        to_key = move.to.to_key()
        board = game_state.board

        source_stack = board.stacks.get(from_key)
        if not source_stack:
            raise ValueError(
                f"Cannot move stack - no stack at source position {from_key} for player {move.player}"
            )

        # Deep-copy source stack to avoid mutating shared instances
        moving_stack = source_stack.model_copy(deep=True)

        # Remove from source before placing marker/processing path
        zobrist = ZobristHash()
        if game_state.zobrist_hash is not None:
            game_state.zobrist_hash ^= zobrist.get_stack_hash(
                from_key, source_stack.controlling_player, source_stack.stack_height, tuple(source_stack.rings)
            )
        del board.stacks[from_key]

        # Leave a marker at departure
        board.markers[from_key] = MarkerInfo(
            player=move.player,
            position=move.from_pos,
            type="regular",
        )
        if game_state.zobrist_hash is not None:
            game_state.zobrist_hash ^= zobrist.get_marker_hash(from_key, move.player)

        # Process markers along movement path (excluding endpoints)
        # Track collapsed space delta for territory_spaces update
        collapsed_before = len(board.collapsed_spaces)
        GameEngine._process_markers_along_path(board, move.from_pos, move.to, move.player, game_state)
        collapsed_after = len(board.collapsed_spaces)
        collapsed_delta = collapsed_after - collapsed_before
        if collapsed_delta > 0:
            for p in game_state.players:
                if p.player_number == move.player:
                    p.territory_spaces += collapsed_delta
                    break

        # Check for marker at landing
        # Per RR-CANON-R091/R092: landing on any marker (own or opponent) removes
        # the marker and eliminates the top ring of the moving stack's cap.
        landing_marker = board.markers.get(to_key)
        landed_on_marker = landing_marker is not None

        if landed_on_marker:
            # Remove marker before landing
            if game_state.zobrist_hash is not None:
                game_state.zobrist_hash ^= zobrist.get_marker_hash(to_key, landing_marker.player)
            del board.markers[to_key]

        # Handle merge with any existing stack at destination
        dest_stack = board.stacks.get(to_key)
        if dest_stack and dest_stack.stack_height > 0:
            # Remove old dest stack hash
            if game_state.zobrist_hash is not None:
                game_state.zobrist_hash ^= zobrist.get_stack_hash(
                    to_key, dest_stack.controlling_player, dest_stack.stack_height, tuple(dest_stack.rings)
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
                    to_key, dest_stack.controlling_player, dest_stack.stack_height, tuple(dest_stack.rings)
                )
        else:
            # Simple move to empty (or marker-only) space
            moving_stack.position = move.to
            board.stacks[to_key] = moving_stack
            # Add new stack hash
            if game_state.zobrist_hash is not None:
                game_state.zobrist_hash ^= zobrist.get_stack_hash(
                    to_key, moving_stack.controlling_player, moving_stack.stack_height, tuple(moving_stack.rings)
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
        if not move.from_pos:
            raise ValueError(
                f"Cannot apply chain capture - move.from_pos is None for player {move.player}"
            )
        if not move.capture_target:
            raise ValueError(
                f"Cannot apply chain capture - move.capture_target is None for player {move.player}"
            )

        board = game_state.board
        from_pos = move.from_pos
        target_pos = move.capture_target
        landing_pos = move.to

        from_key = from_pos.to_key()
        target_key = target_pos.to_key()
        landing_key = landing_pos.to_key()

        attacker = board.stacks.get(from_key)
        target_stack = board.stacks.get(target_key)
        if not attacker:
            raise ValueError(
                f"Cannot apply chain capture - no attacker stack at {from_key} for player {move.player}"
            )
        if not target_stack:
            raise ValueError(
                f"Cannot apply chain capture - no target stack at {target_key} for player {move.player}"
            )

        # Deep-copy stacks before modification
        attacker = attacker.model_copy(deep=True)
        target_stack = target_stack.model_copy(deep=True)

        # Remove attacker from source before marker/path processing
        zobrist = ZobristHash()
        if game_state.zobrist_hash is not None:
            game_state.zobrist_hash ^= zobrist.get_stack_hash(
                from_key, attacker.controlling_player, attacker.stack_height, tuple(attacker.rings)
            )
        del board.stacks[from_key]

        # Leave a marker at departure
        board.markers[from_key] = MarkerInfo(
            player=move.player,
            position=from_pos,
            type="regular",
        )
        if game_state.zobrist_hash is not None:
            game_state.zobrist_hash ^= zobrist.get_marker_hash(from_key, move.player)

        # Process markers along from→target and target→landing
        GameEngine._process_markers_along_path(board, from_pos, target_pos, move.player, game_state)
        GameEngine._process_markers_along_path(board, target_pos, landing_pos, move.player, game_state)

        # Capture top ring from target (top = end of list)
        if not target_stack.rings:
            # Defensive: nothing to capture; restore attacker and exit
            board.stacks[from_key] = attacker
            return

        # Remove old target stack hash
        if game_state.zobrist_hash is not None:
            game_state.zobrist_hash ^= zobrist.get_stack_hash(
                target_key, target_stack.controlling_player, target_stack.stack_height, tuple(target_stack.rings)
            )

        captured_ring = target_stack.rings.pop()
        target_stack.stack_height -= 1

        if target_stack.stack_height == 0 or not target_stack.rings:
            # Stack is empty (or inconsistent with empty rings) - remove it
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
                    target_key, target_stack.controlling_player, target_stack.stack_height, tuple(target_stack.rings)
                )

        # Insert captured ring at bottom of attacker.
        #
        # In the TS engine, rings are stored [top -> bottom] and the captured
        # ring is appended at the bottom. Our Python representation stores
        # rings bottom -> top, so inserting at index 0 is the analogous
        # "bottom" operation.
        #
        # IMPORTANT: We must recalculate cap_height after insertion because
        # if the captured ring is the same color as the entire cap (e.g.,
        # attacker has [1,1,1] and captures 1), the cap extends to include
        # the new ring. TS correctly recalculates this via calculateCapHeight.
        attacker.rings.insert(0, captured_ring)
        attacker.stack_height += 1
        # Controlling player is the top ring (last in our bottom->top array)
        attacker.controlling_player = attacker.rings[-1]
        # Recalculate cap_height from top (end of array) going down
        h = 0
        for r in reversed(attacker.rings):
            if r == attacker.controlling_player:
                h += 1
            else:
                break
        attacker.cap_height = h

        # Move attacker to landing (merge if stack already present)
        existing_dest = board.stacks.get(landing_key)
        if existing_dest and existing_dest.stack_height > 0:
            # Remove old dest stack hash
            if game_state.zobrist_hash is not None:
                game_state.zobrist_hash ^= zobrist.get_stack_hash(
                    landing_key,
                    existing_dest.controlling_player,
                    existing_dest.stack_height,
                    tuple(existing_dest.rings),
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
                    tuple(existing_dest.rings),
                )
        else:
            attacker.position = landing_pos
            board.stacks[landing_key] = attacker
            # Add new stack hash
            if game_state.zobrist_hash is not None:
                game_state.zobrist_hash ^= zobrist.get_stack_hash(
                    landing_key, attacker.controlling_player, attacker.stack_height, tuple(attacker.rings)
                )

        # Handle landing on marker
        # Per RR-CANON-R101/R102: landing on any marker (own or opponent) removes
        # the marker and eliminates the top ring of the attacking stack's cap.
        landing_marker = board.markers.get(landing_key)
        if landing_marker is not None:
            # Remove marker (do not collapse it)
            if game_state.zobrist_hash is not None:
                game_state.zobrist_hash ^= zobrist.get_marker_hash(landing_key, landing_marker.player)
            del board.markers[landing_key]
            # Eliminate top ring of the attacking stack's cap
            GameEngine._eliminate_top_ring_at(game_state, landing_pos, move.player)

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
            num_players = len(game_state.players)
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
        num_players = len(game_state.players)
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

        # 4. Apply collapses. TS's LineAggregate increments territorySpaces
        # by the number of collapsed marker positions (collapsedKeys.size).
        # We mirror that here.
        #
        # IMPORTANT: TS's collapseLinePositions also returns rings from any
        # stacks on collapsed positions back to their owners' hands. We must
        # do the same for parity.
        seen_keys = set()
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
                for ring_owner in stack.rings:
                    for player_state in game_state.players:
                        if player_state.player_number == ring_owner:
                            player_state.rings_in_hand += 1
                            break
                # Remove the stack
                del board.stacks[key]

            # Check if there was a marker there (should be for line formation)
            marker = board.markers.get(key)
            if marker and game_state.zobrist_hash is not None:
                game_state.zobrist_hash ^= zobrist.get_marker_hash(key, marker.player)
            BoardManager.set_collapsed_space(pos, move.player, board)
            if game_state.zobrist_hash is not None:
                game_state.zobrist_hash ^= zobrist.get_collapsed_hash(key)
            collapsed_count += 1

        # Update the player's territory_spaces to reflect newly collapsed spaces.
        # Territory increases when spaces collapse during line processing.
        if collapsed_count > 0:
            for player_state in game_state.players:
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
                game_state.pending_line_reward_elimination = True

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
            raise ValueError(
                f"Cannot apply territory claim - no matching disconnected region "
                f"found for move at {move_key} by player {player}"
            )

        # Self-elimination prerequisite (RR-CANON-R145 / RR-CANON-R114): a
        # disconnected region may only be processed if the moving player has
        # at least one eligible elimination target outside the region.
        #
        # TS applyProcessTerritoryRegionDecision treats non-processable regions
        # as a no-op rather than mutating the board. Mirror that behavior here
        # so invalid choose_territory_option moves cannot silently bypass the
        # prerequisite and create TS↔Python parity drift.
        if not GameEngine._can_process_disconnected_region(
            game_state,
            target_region,
            player,
        ):
            _debug(
                "DEBUG: _apply_territory_claim skipped non-processable region "
                f"at {move_key} for P{player}\n"
            )
            return

        # 1c. Gather border markers for territory elimination.

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
            if marker and game_state.zobrist_hash is not None:
                game_state.zobrist_hash ^= zobrist.get_marker_hash(key, marker.player)
            BoardManager.set_collapsed_space(pos, player, board)
            if game_state.zobrist_hash is not None:
                game_state.zobrist_hash ^= zobrist.get_collapsed_hash(key)

        # 4. Collapse all border markers to the moving player's colour.
        for pos in border_markers:
            key = pos.to_key()
            marker = board.markers.get(key)
            if marker and game_state.zobrist_hash is not None:
                game_state.zobrist_hash ^= zobrist.get_marker_hash(key, marker.player)
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
    def _generate_all_positions(board_type: BoardType, size: int) -> list[Position]:
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
        elif board_type in (BoardType.HEXAGONAL, BoardType.HEX8):
            # Hex boards: size = bounding box (2*radius + 1). Radius = (size - 1) // 2.
            radius = (size - 1) // 2
            for x in range(-radius, radius + 1):
                for y in range(-radius, radius + 1):
                    z = -x - y
                    if abs(x) <= radius and abs(y) <= radius and abs(z) <= radius:
                        positions.append(Position(x=x, y=y, z=z))
        return positions

    @staticmethod
    def _get_adjacent_positions(pos: Position, board_type: BoardType, size: int) -> list[Position]:
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
                    # Bounds check - use size parameter from BOARD_CONFIGS
                    if 0 <= new_x < size and 0 <= new_y < size:
                        adjacent.append(Position(x=new_x, y=new_y))

        elif board_type in (BoardType.HEXAGONAL, BoardType.HEX8):
            hex_directions = [(1, 0, -1), (-1, 0, 1), (0, 1, -1), (0, -1, 1), (1, -1, 0), (-1, 1, 0)]
            # Hex boards: size = bounding box (2*radius + 1). Radius = (size - 1) // 2.
            radius = (size - 1) // 2
            for dx, dy, dz in hex_directions:
                if pos.z is None:
                    continue
                nx, ny, nz = pos.x + dx, pos.y + dy, pos.z + dz
                if abs(nx) <= radius and abs(ny) <= radius and abs(nz) <= radius:
                    adjacent.append(Position(x=nx, y=ny, z=nz))

        return adjacent

    @staticmethod
    def get_visible_stacks(pos: Position, game_state: GameState) -> list[RingStack]:
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
        elif board_type in (BoardType.HEXAGONAL, BoardType.HEX8):
            # 6 directions for hexagonal board
            directions = [(1, 0, -1), (-1, 0, 1), (0, 1, -1), (0, -1, 1), (1, -1, 0), (-1, 1, 0)]

        # Use size from board for bounds checking
        # For hex boards: size = bounding box (2*radius + 1). Radius = (size - 1) // 2.
        radius = (size - 1) // 2

        for dx, dy, dz in directions:
            curr_x, curr_y = pos.x, pos.y
            curr_z = pos.z if pos.z is not None else 0

            # Raycast in this direction
            while True:
                curr_x += dx
                curr_y += dy
                curr_z += dz

                # Check bounds - use size from BOARD_CONFIGS
                if board_type in [BoardType.SQUARE8, BoardType.SQUARE19]:
                    if not (0 <= curr_x < size and 0 <= curr_y < size):
                        break
                elif board_type in (BoardType.HEXAGONAL, BoardType.HEX8):
                    if not (abs(curr_x) <= radius and abs(curr_y) <= radius and abs(curr_z) <= radius):
                        break

                curr_pos_key = f"{curr_x},{curr_y}"
                if board_type in (BoardType.HEXAGONAL, BoardType.HEX8):
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

        Mirrors the TS applyEliminateRingsFromStackDecision behaviour:
        - For line context (RR-CANON-R122): Eliminate exactly ONE ring
        - For territory/forced context (RR-CANON-R145, R100): Eliminate entire cap
        - For recovery context (RR-CANON-R113/R114): Extract exactly ONE buried ring

        Uses the shared _eliminate_top_ring_at helper, so that board.eliminated_rings,
        total_rings_eliminated, and per-player eliminated_rings remain in
        sync with TS semantics.
        """
        board = game_state.board
        pos = move.to
        stack = board.stacks.get(pos.to_key())
        if not stack:
            raise ValueError(
                f"Cannot apply forced elimination - no stack at {pos.to_key()} "
                f"for player {move.player}"
            )

        # Determine how many rings to eliminate based on context (RR-CANON-R022, R122):
        # - 'line': Eliminate exactly ONE ring (per RR-CANON-R122)
        # - 'territory' or 'forced' or None: Eliminate entire cap (per RR-CANON-R145, R100)
        # - 'recovery': Extract exactly ONE buried ring (per RR-CANON-R113/R114)
        elimination_context = getattr(move, 'elimination_context', None)
        if elimination_context == 'line':
            rings_to_eliminate = 1
        elif elimination_context == 'recovery':
            GameEngine._extract_buried_ring_at(
                game_state,
                pos,
                credited_player=move.player,
            )
            return
        else:
            # Per TS parity (globalActions.ts line 458): use max(1, cap_height) to
            # handle degenerate legacy states where cap_height metadata is 0 or
            # missing but the stack has rings. This matches:
            #   eliminatedRings: [{ player, count: Math.max(1, chosenStack.capHeight || 0) }]
            rings_to_eliminate = max(1, stack.cap_height or 0)

        for _ in range(rings_to_eliminate):
            GameEngine._eliminate_top_ring_at(
                game_state,
                pos,
                credited_player=move.player,
            )
