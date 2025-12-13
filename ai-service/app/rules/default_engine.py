from __future__ import annotations

import os
from typing import List, Mapping

from app.game_engine import PhaseRequirementType

from app.models import GameState, GameStatus, GamePhase, Move, MoveType

from .interfaces import RulesEngine, Validator, Mutator
from .validators.placement import PlacementValidator
from .validators.movement import MovementValidator
from .validators.capture import CaptureValidator
from .validators.line import LineValidator
from .validators.territory import TerritoryValidator

from .mutators.placement import PlacementMutator
from .mutators.movement import MovementMutator
from .mutators.capture import CaptureMutator
from .mutators.line import LineMutator
from .mutators.territory import TerritoryMutator
from .mutators.turn import TurnMutator


class DefaultRulesEngine(RulesEngine):
    """RulesEngine backed by the existing GameEngine implementation.

    This implementation mirrors the TS GameEngine architecture by maintaining
    lists of validators and mutators. Canonical semantics for move generation
    and full-state transitions are still owned by :class:`app.game_engine.GameEngine`.

    The engine currently runs in two layers:

    1. **Per-move mutator shadow contracts**
       For each move family (placement, movement, capture, line, territory),
       we apply the corresponding mutator to a copied state and assert that
       its *board + player* side-effects match the canonical
       ``GameEngine.apply_move`` result. This keeps the legacy engine as the
       single source of truth while continuously validating the mutators.

    2. **Optional mutator-first orchestration**
       When ``mutator_first`` is enabled (via constructor or the
       ``RINGRIFT_RULES_MUTATOR_FIRST`` env var), ``apply_move`` also runs a
       full mutator-driven state transition path that mirrors
       ``GameEngine.apply_move``: copy-on-write board/state, zobrist hashing,
       S-invariant snapshots, phase/turn/victory updates, and then asserts
       that its resulting state is in full lockstep with the canonical
       GameEngine result.

       Even in mutator-first mode, ``apply_move`` still *returns* the
       GameEngine-derived state; mutators remain shadow-only for now.
    """

    def __init__(
        self,
        mutator_first: bool | None = None,
        skip_shadow_contracts: bool | None = None,
    ):
        self.validators: List[Validator] = [
            PlacementValidator(),
            MovementValidator(),
            CaptureValidator(),
            LineValidator(),
            TerritoryValidator(),
        ]
        self.mutators: List[Mutator] = [
            PlacementMutator(),
            MovementMutator(),
            CaptureMutator(),
            LineMutator(),
            TerritoryMutator(),
            TurnMutator(),
        ]

        # Performance optimization: skip shadow contract validation.
        #
        # Shadow contracts run deep copies of state and compare mutator results
        # against GameEngine results for correctness validation. This is useful
        # during development but adds ~60-80% overhead for training/self-play.
        #
        # Set RINGRIFT_SKIP_SHADOW_CONTRACTS=true for training/benchmarking.
        skip_shadow_env = os.getenv(
            "RINGRIFT_SKIP_SHADOW_CONTRACTS", ""
        ).lower()
        if skip_shadow_contracts is not None:
            self._skip_shadow_contracts = skip_shadow_contracts
        else:
            self._skip_shadow_contracts = skip_shadow_env in {
                "1",
                "true",
                "yes",
                "on",
            }

        # Configuration for the optional mutator-first orchestration path.
        #
        # Mutator-first can be hard-gated per environment using the
        # RINGRIFT_SERVER_ENABLE_MUTATOR_FIRST flag. When this flag is not
        # truthy, mutator-first is disabled regardless of constructor
        # arguments or the per-service env flag below.
        server_flag = os.getenv(
            "RINGRIFT_SERVER_ENABLE_MUTATOR_FIRST",
            "",
        ).lower()
        server_allows = server_flag in {"1", "true", "yes", "on"}

        env_flag = os.getenv("RINGRIFT_RULES_MUTATOR_FIRST", "").lower()

        if not server_allows:
            # Hard gate: mutator-first is globally disabled in this
            # environment unless explicitly permitted by the server flag.
            self._mutator_first_enabled = False
        elif mutator_first is not None:
            self._mutator_first_enabled = mutator_first
        else:
            self._mutator_first_enabled = env_flag in {
                "1",
                "true",
                "yes",
                "on",
            }

        # Host-side bookkeeping enforcement (mirrors TS host behaviour).
        # When enabled we will auto-apply required no_* bookkeeping moves for
        # the **current actor** in line/territory phases immediately after the
        # interactive move, so turn rotation cannot skip required phase visits.
        self._force_bookkeeping_moves = os.getenv(
            "RINGRIFT_FORCE_BOOKKEEPING_MOVES", ""
        ).lower() in {"1", "true", "yes", "on"}

    def get_valid_moves(self, state: GameState, player: int) -> List[Move]:
        """Return all legal moves for ``player`` in ``state``.

        This is a **host-level** helper over the core GameEngine:

        - Delegates to :meth:`GameEngine.get_valid_moves` for interactive
          moves (placements, movements/captures, line/territory decisions,
          forced_elimination in the FE phase).
        - When no interactive moves exist for an ACTIVE state where
          ``player`` is the current player, calls
          :meth:`GameEngine.get_phase_requirement` and
          :meth:`GameEngine.synthesize_bookkeeping_move` to surface the
          required bookkeeping move (``NO_*_ACTION`` or
          ``FORCED_ELIMINATION``) as a single legal :class:`Move`.

        This mirrors the behaviour of the TS hosts and keeps the core
        rules layer free of auto-generated moves per RR-CANON-R076.
        """
        from app.game_engine import GameEngine, PhaseRequirement

        # Interactive moves from the core engine (no bookkeeping moves).
        moves = GameEngine.get_valid_moves(state, player)

        # If forced bookkeeping is enabled, and we're in a decision phase with
        # no interactive moves (or an explicit requirement), return exactly the
        # synthesized no_* bookkeeping move for the current actor. Do NOT
        # auto-apply it; the host must record it explicitly. Also, never surface
        # other move types in these phases when forced bookkeeping is on.
        # If we detect ANM (no moves, no requirement) for the current actor in
        # these phases, synthesize the required no_* action to keep the turn
        # canonical rather than aborting.
        if self._force_bookkeeping_moves:
            if (
                state.current_player == player
                and state.game_status == GameStatus.ACTIVE
                and state.current_phase
                in (GamePhase.LINE_PROCESSING, GamePhase.TERRITORY_PROCESSING)
            ):
                requirement = GameEngine.get_phase_requirement(state, player)
                bookkeeping_move = None
                if requirement and requirement.type in (
                    PhaseRequirementType.NO_LINE_ACTION_REQUIRED,
                    PhaseRequirementType.NO_TERRITORY_ACTION_REQUIRED,
                ):
                    bookkeeping_move = GameEngine.synthesize_bookkeeping_move(
                        requirement,
                        state,
                    )
                elif not moves:
                    req_type = (
                        PhaseRequirementType.NO_LINE_ACTION_REQUIRED
                        if state.current_phase == GamePhase.LINE_PROCESSING
                        else PhaseRequirementType.NO_TERRITORY_ACTION_REQUIRED
                    )
                    req = PhaseRequirement(
                        type=req_type,
                        player=player,
                        eligible_positions=[],
                    )
                    bookkeeping_move = GameEngine.synthesize_bookkeeping_move(
                        req,
                        state,
                    )
                # ANM fallback: no moves and no explicit requirement from core,
                # but forced bookkeeping is on — synthesize a no_* action to
                # keep the turn canonical instead of aborting.
                if bookkeeping_move is None and not moves:
                    req_type = (
                        PhaseRequirementType.NO_LINE_ACTION_REQUIRED
                        if state.current_phase == GamePhase.LINE_PROCESSING
                        else PhaseRequirementType.NO_TERRITORY_ACTION_REQUIRED
                    )
                    req = PhaseRequirement(
                        type=req_type,
                        player=player,
                        eligible_positions=[],
                    )
                    bookkeeping_move = GameEngine.synthesize_bookkeeping_move(
                        req,
                        state,
                    )
                if bookkeeping_move is not None:
                    return [bookkeeping_move]
                # If interactive decisions exist in these phases, keep them; but
                # block anything else (e.g., elimination/placement) from leaking in.
                moves = [
                    m
                    for m in moves
                    if (
                        state.current_phase == GamePhase.LINE_PROCESSING
                        and m.type
                        in {
                            MoveType.PROCESS_LINE,
                            MoveType.CHOOSE_LINE_OPTION,
                            MoveType.CHOOSE_LINE_REWARD,  # legacy alias
                        }
                    )
                    or (
                        state.current_phase == GamePhase.TERRITORY_PROCESSING
                        and m.type in {
                            MoveType.PROCESS_TERRITORY_REGION,
                            MoveType.CHOOSE_TERRITORY_OPTION,
                            MoveType.ELIMINATE_RINGS_FROM_STACK,
                            MoveType.SKIP_TERRITORY_PROCESSING,
                            MoveType.NO_TERRITORY_ACTION,
                        }
                    )
                ]

        if not moves:
            # No interactive moves – only attempt to synthesize bookkeeping
            # moves when this player is actually on turn in an ACTIVE game.
            if state.current_player == player:
                game_status = state.game_status
                status_str = (
                    game_status.value
                    if hasattr(game_status, "value")
                    else str(game_status)
                )
                if status_str == GameStatus.ACTIVE.value:
                    requirement = GameEngine.get_phase_requirement(
                        state,
                        player,
                    )
                    if requirement is not None:
                        bookkeeping_move = GameEngine.synthesize_bookkeeping_move(
                            requirement,
                            state,
                        )
                        # Return a single canonical bookkeeping move.
                        moves = [bookkeeping_move]
                    # Defensive: if we're in territory_processing with no moves
                    # and no requirement, synthesize NO_TERRITORY_ACTION when
                    # forced bookkeeping is on to avoid leaving the recorder in
                    # a mid-phase state.
                    elif (
                        self._force_bookkeeping_moves
                        and state.current_phase == GamePhase.TERRITORY_PROCESSING
                    ):
                        req = PhaseRequirement(
                            type=PhaseRequirementType.NO_TERRITORY_ACTION_REQUIRED,
                            player=player,
                            eligible_positions=[],
                        )
                        bookkeeping_move = GameEngine.synthesize_bookkeeping_move(
                            req,
                            state,
                        )
                        moves = [bookkeeping_move]

        # Defensive phase/move invariant: ensure every move we surface is
        # legal for the current phase, reusing the canonical engine guard.
        filtered_moves: list[Move] = []
        for move in moves:
            try:
                GameEngine._assert_phase_move_invariant(state, move)
                filtered_moves.append(move)
            except Exception:
                # Skip invalid moves instead of raising; caller will continue.
                continue
        moves = filtered_moves

        # Defensive phase-requirement consistency: if the core engine reports
        # a pending phase requirement, ensure we return exactly one matching
        # bookkeeping move.
        requirement = GameEngine.get_phase_requirement(state, player)
        if requirement is not None:
            expected = GameEngine.synthesize_bookkeeping_move(
                requirement,
                state,
            )
            if not moves:
                raise RuntimeError(
                    "DefaultRulesEngine.get_valid_moves: phase requirement "
                    f"exists ({requirement.type.value}) but no moves were "
                    "returned"
                )
            if len(moves) != 1 or moves[0].type != expected.type:
                raise RuntimeError(
                    "DefaultRulesEngine.get_valid_moves: inconsistent "
                    "bookkeeping move for requirement "
                    f"{requirement.type.value}: got {moves[0].type.value}, "
                    f"expected {expected.type.value}"
                )

        return moves

    def validate_move(self, state: GameState, move: Move) -> bool:
        """Validate if a specific move is legal in the current state.

        Dispatches to the appropriate validator based on move type.
        """

        # Special-case validation for the pie-rule meta-move. We treat
        # SWAP_SIDES as legal exactly when GameEngine.get_valid_moves would
        # surface it for the current state/player, mirroring the TS backend
        # GameEngine.shouldOfferSwapSidesMetaMove gate.
        if move.type == MoveType.SWAP_SIDES:
            from app.game_engine import GameEngine

            if move.player != 2:
                return False

            legal = GameEngine.get_valid_moves(state, move.player)
            return any(
                m.type == MoveType.SWAP_SIDES and m.player == move.player
                for m in legal
            )

        # Forced no-op placement action is a bookkeeping move that appears in
        # canonical recordings when a player enters RING_PLACEMENT but has no
        # legal placement anywhere. Hosts synthesize it from phase
        # requirements; treat it as valid without additional validation.
        if move.type == MoveType.NO_PLACEMENT_ACTION:
            return True

        # Forced no-op line action is a bookkeeping move that appears in
        # canonical recordings when a player enters LINE_PROCESSING with no
        # lines to process. Under R076, it is synthesized by hosts based on
        # phase requirements, not auto-generated by the core engine. Always
        # treat it as valid when present in a canonical recording.
        if move.type == MoveType.NO_LINE_ACTION:
            return True

        # Forced no-op movement action is a bookkeeping move that may appear
        # in canonical recordings when a player enters MOVEMENT but has no
        # legal movement or capture anywhere. Treat it as valid.
        if move.type == MoveType.NO_MOVEMENT_ACTION:
            return True

        # RR-CANON-R073: Post-movement capture is optional; the active player
        # may explicitly decline via SKIP_CAPTURE to proceed to line_processing.
        if move.type == MoveType.SKIP_CAPTURE:
            return state.current_phase == GamePhase.CAPTURE and move.player == state.current_player

        # Dispatch based on move type for all other moves
        if move.type in (MoveType.PLACE_RING, MoveType.SKIP_PLACEMENT):
            # PlacementValidator
            return self.validators[0].validate(state, move)
        if move.type == MoveType.MOVE_STACK:
            # MovementValidator
            return self.validators[1].validate(state, move)
        if move.type in (
            MoveType.OVERTAKING_CAPTURE,
            MoveType.CHAIN_CAPTURE,
            MoveType.CONTINUE_CAPTURE_SEGMENT,
        ):
            # CaptureValidator
            return self.validators[2].validate(state, move)
        if move.type in (
            MoveType.PROCESS_LINE,
            MoveType.CHOOSE_LINE_REWARD,
            MoveType.LINE_FORMATION,
            MoveType.CHOOSE_LINE_OPTION,
        ):
            # LineValidator
            return self.validators[3].validate(state, move)
        if move.type in (
            MoveType.PROCESS_TERRITORY_REGION,
            MoveType.ELIMINATE_RINGS_FROM_STACK,
            MoveType.TERRITORY_CLAIM,
            MoveType.CHOOSE_TERRITORY_OPTION,
        ):
            # TerritoryValidator
            return self.validators[4].validate(state, move)

        # Fallback for unknown move types
        return False

    def _apply_move_with_mutators(self, state: GameState, move: Move) -> GameState:
        """Mutator-first analogue of :meth:`GameEngine.apply_move`.

        This helper mirrors the canonical GameEngine orchestration while
        delegating board/player mutations to the specialised mutators:

        - Copy-on-write of the board and all mutable GameState fields.
        - Zobrist hash management (phase + player components).
        - S-invariant snapshots before and after the move (soft check only).
        - Dispatch of move families to Placement/Movement/Capture/Line/
          Territory mutators; FORCED_ELIMINATION is still handled via
          ``GameEngine._apply_forced_elimination`` to keep semantics aligned.
        - Per-turn bookkeeping (move_history, last_move_at,
          must_move_from_stack_key).
        - Phase/turn/victory updates via GameEngine helpers.

        The resulting state is expected to be in full lockstep with the
        canonical GameEngine.apply_move result; callers are responsible for
        asserting equivalence where appropriate.
        """
        from app.board_manager import BoardManager
        from app.game_engine import GameEngine
        from app.ai.zobrist import ZobristHash

        # --- 1. Copy-on-write for board and state ---
        board = state.board
        new_board = board.model_copy()
        new_board.stacks = dict(board.stacks)
        new_board.markers = dict(board.markers)
        new_board.collapsed_spaces = dict(board.collapsed_spaces)
        new_board.eliminated_rings = dict(board.eliminated_rings)
        new_board.formed_lines = list(board.formed_lines)
        new_board.territories = dict(board.territories)

        new_state = state.model_copy(update={"board": new_board})
        new_state.players = [p.model_copy() for p in state.players]
        new_state.move_history = list(state.move_history)

        # --- 2. Zobrist hash initialisation ---
        zobrist = ZobristHash()
        if new_state.zobrist_hash is None:
            new_state.zobrist_hash = zobrist.compute_initial_hash(new_state)

        # --- 3. Capture S-invariant before the move ---
        before_snapshot = BoardManager.compute_progress_snapshot(new_state)

        # --- 4. Remove current player/phase contributions from hash ---
        if new_state.zobrist_hash is not None:
            new_state.zobrist_hash ^= zobrist.get_player_hash(
                new_state.current_player
            )
            new_state.zobrist_hash ^= zobrist.get_phase_hash(
                new_state.current_phase
            )

        # --- 5. Dispatch to mutators / helpers by move type ---
        # SWAP_SIDES is a pure meta-move handled by the canonical
        # GameEngine.apply_move; mutators do not currently model seat/colour
        # swapping explicitly, so we skip mutator application for this move
        # type and rely solely on the canonical path.
        if move.type == MoveType.SWAP_SIDES:
            pass
        elif move.type == MoveType.PLACE_RING:
            PlacementMutator().apply(new_state, move)
        elif move.type == MoveType.SKIP_PLACEMENT:
            # No board change; phase update will advance the turn.
            pass
        elif move.type == MoveType.MOVE_STACK:
            MovementMutator().apply(new_state, move)
        elif move.type in (
            MoveType.OVERTAKING_CAPTURE,
            MoveType.CHAIN_CAPTURE,
            MoveType.CONTINUE_CAPTURE_SEGMENT,
        ):
            CaptureMutator().apply(new_state, move)
        elif move.type in (
            MoveType.PROCESS_LINE,
            MoveType.CHOOSE_LINE_REWARD,
            MoveType.LINE_FORMATION,
            MoveType.CHOOSE_LINE_OPTION,
        ):
            LineMutator().apply(new_state, move)
        elif move.type in (
            MoveType.PROCESS_TERRITORY_REGION,
            MoveType.ELIMINATE_RINGS_FROM_STACK,
            MoveType.TERRITORY_CLAIM,
            MoveType.CHOOSE_TERRITORY_OPTION,
        ):
            if move.type == MoveType.ELIMINATE_RINGS_FROM_STACK:
                # Preserve semantics by delegating elimination to the same
                # helper used by GameEngine.apply_move.
                GameEngine._apply_forced_elimination(new_state, move)
            else:
                TerritoryMutator().apply(new_state, move)
        elif move.type == MoveType.FORCED_ELIMINATION:
            GameEngine._apply_forced_elimination(new_state, move)

        # --- 6. Turn-level bookkeeping mirroring GameEngine.apply_move ---
        new_state.move_history.append(move)
        new_state.last_move_at = move.timestamp

        # Per-turn must-move tracking (TurnEngine.updatePerTurnStateAfterMove).
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

        # --- 7. Phase/turn transitions and victory checks ---
        GameEngine._update_phase(new_state, move)

        # Re-apply hash contributions for the new player/phase.
        if new_state.zobrist_hash is not None:
            new_state.zobrist_hash ^= zobrist.get_player_hash(
                new_state.current_player
            )
            new_state.zobrist_hash ^= zobrist.get_phase_hash(
                new_state.current_phase
            )

        # Soft S-invariant check after the move; we currently do not enforce
        # monotonicity here, but the snapshot is available for diagnostics.
        after_snapshot = BoardManager.compute_progress_snapshot(new_state)
        _ = after_snapshot  # noqa: F841 (reserved for future diagnostics)
        _ = before_snapshot  # noqa: F841

        GameEngine._check_victory(new_state)

        return new_state

    def apply_move(self, state: GameState, move: Move) -> GameState:
        """Apply ``move`` to ``state`` and return the resulting GameState.

        Canonical semantics are always provided by ``GameEngine.apply_move``,
        which owns the full turn/phase/victory orchestration and hashing.

        For each move family we also run the dedicated mutator on a deep-copied
        state and assert that its board / player side-effects stay in lockstep
        with the canonical engine. When ``mutator_first`` is enabled, we
        additionally run a full mutator-driven orchestration pass and assert
        full-state equivalence (board, players, current_player, phase, status,
        chain_capture_state, must_move_from_stack_key).
        """
        from app.game_engine import GameEngine

        # Canonical result: always computed via GameEngine.apply_move.
        next_via_engine = GameEngine.apply_move(state, move)

        # NOTE: Post-move bookkeeping (no_line_action, no_territory_action) is
        # intentionally NOT applied here. That's the host's responsibility
        # (e.g., RingRiftEnv.step() handles it and records the moves). Doing
        # it here would silently advance the phase without recording the moves,
        # breaking trace replay parity.

        # Fast path: skip shadow contracts for training/benchmarking.
        # Shadow contracts add ~60-80% overhead due to deep copies.
        if self._skip_shadow_contracts:
            return next_via_engine

        # --- Per-move mutator shadow contracts (board + players only) ---
        if move.type == MoveType.PLACE_RING:
            # Deep-copy the incoming state so the mutator can safely mutate
            # in-place without affecting callers or the canonical path.
            mutator_state = state.model_copy(deep=True)

            placement_mutator = PlacementMutator()
            placement_mutator.apply(mutator_state, move)

            # Contract: board + players must be identical to the canonical
            # GameEngine.apply_move result. We intentionally limit the
            # comparison to these domains; move history, phase, hashes, and
            # victory status are still owned by GameEngine.
            #
            # EXCEPTION: If GameEngine.apply_move triggered additional
            # automatic moves (e.g. FORCED_ELIMINATION) or side effects via
            # phase transitions that the atomic PlacementMutator does not
            # perform, we skip the strict comparison to avoid false positives.
            extra_moves = (
                len(next_via_engine.move_history) - len(state.move_history)
            )
            turn_ended = (
                next_via_engine.current_player != state.current_player
            )
            if extra_moves > 1 or turn_ended:
                # GameEngine performed extra moves (e.g. forced elimination)
                # or ended the turn. Divergence is expected.
                pass
            else:
                move_desc = self._describe_move(move)
                if mutator_state.board.stacks != next_via_engine.board.stacks:
                    details = self._diff_mapping_keys(
                        mutator_state.board.stacks,
                        next_via_engine.board.stacks,
                    )
                    raise RuntimeError(
                        "PlacementMutator diverged from GameEngine.apply_move "
                        "for PLACE_RING: board.stacks mismatch "
                        f"(move={move_desc}, "
                        f"mut={len(mutator_state.board.stacks)}, "
                        f"eng={len(next_via_engine.board.stacks)}, "
                        f"details={details})"
                    )
                if mutator_state.board.markers != next_via_engine.board.markers:
                    details = self._diff_mapping_keys(
                        mutator_state.board.markers,
                        next_via_engine.board.markers,
                    )
                    raise RuntimeError(
                        "PlacementMutator diverged from GameEngine.apply_move "
                        "for PLACE_RING: board.markers mismatch "
                        f"(move={move_desc}, "
                        f"mut={len(mutator_state.board.markers)}, "
                        f"eng={len(next_via_engine.board.markers)}, "
                        f"details={details})"
                    )
                if (
                    mutator_state.board.collapsed_spaces
                    != next_via_engine.board.collapsed_spaces
                ):
                    details = self._diff_mapping_keys(
                        mutator_state.board.collapsed_spaces,
                        next_via_engine.board.collapsed_spaces,
                    )
                    raise RuntimeError(
                        "PlacementMutator diverged from GameEngine.apply_move "
                        "for PLACE_RING: board.collapsed_spaces mismatch "
                        f"(move={move_desc}, "
                        f"mut={len(mutator_state.board.collapsed_spaces)}, "
                        f"eng={len(next_via_engine.board.collapsed_spaces)}, "
                        f"details={details})"
                    )
                if (
                    mutator_state.board.eliminated_rings
                    != next_via_engine.board.eliminated_rings
                ):
                    details = self._diff_mapping_keys(
                        mutator_state.board.eliminated_rings,
                        next_via_engine.board.eliminated_rings,
                    )
                    raise RuntimeError(
                        "PlacementMutator diverged from GameEngine.apply_move "
                        "for PLACE_RING: board.eliminated_rings mismatch "
                        f"(move={move_desc}, "
                        f"mut={len(mutator_state.board.eliminated_rings)}, "
                        f"eng={len(next_via_engine.board.eliminated_rings)}, "
                        f"details={details})"
                    )
                if mutator_state.players != next_via_engine.players:
                    raise RuntimeError(
                        "PlacementMutator diverged from GameEngine.apply_move "
                        "for PLACE_RING: players mismatch "
                        f"(move={move_desc}, "
                        f"mut={mutator_state.players}, "
                        f"eng={next_via_engine.players})"
                    )

        elif move.type == MoveType.MOVE_STACK:
            # Movement shadow contract: ensure MovementMutator applies the same
            # board / player side-effects as GameEngine._apply_move_stack /
            # GameEngine.apply_move for MOVE_STACK.
            mutator_state = state.model_copy(deep=True)
            movement_mutator = MovementMutator()
            movement_mutator.apply(mutator_state, move)

            extra_moves = (
                len(next_via_engine.move_history) - len(state.move_history)
            )
            turn_ended = (
                next_via_engine.current_player != state.current_player
            )
            # Check for forced elimination side-effects that might occur even if
            # the turn didn't end (e.g. P2 moves -> P1 skipped -> P2 forced
            # elimination).
            forced_elimination_occurred = (
                next_via_engine.total_rings_eliminated
                > mutator_state.total_rings_eliminated
            )

            if extra_moves > 1 or turn_ended or forced_elimination_occurred:
                pass
            else:
                move_desc = self._describe_move(move)
                if mutator_state.board.stacks != next_via_engine.board.stacks:
                    details = self._diff_mapping_keys(
                        mutator_state.board.stacks,
                        next_via_engine.board.stacks,
                    )
                    raise RuntimeError(
                        "MovementMutator diverged from GameEngine.apply_move "
                        "for MOVE_STACK: board.stacks mismatch "
                        f"(move={move_desc}, "
                        f"mut={len(mutator_state.board.stacks)}, "
                        f"eng={len(next_via_engine.board.stacks)}, "
                        f"details={details})"
                    )
                if mutator_state.board.markers != next_via_engine.board.markers:
                    details = self._diff_mapping_keys(
                        mutator_state.board.markers,
                        next_via_engine.board.markers,
                    )
                    raise RuntimeError(
                        "MovementMutator diverged from GameEngine.apply_move "
                        "for MOVE_STACK: board.markers mismatch "
                        f"(move={move_desc}, "
                        f"mut={len(mutator_state.board.markers)}, "
                        f"eng={len(next_via_engine.board.markers)}, "
                        f"details={details})"
                    )
                if (
                    mutator_state.board.collapsed_spaces
                    != next_via_engine.board.collapsed_spaces
                ):
                    details = self._diff_mapping_keys(
                        mutator_state.board.collapsed_spaces,
                        next_via_engine.board.collapsed_spaces,
                    )
                    raise RuntimeError(
                        "MovementMutator diverged from GameEngine.apply_move "
                        "for MOVE_STACK: board.collapsed_spaces mismatch "
                        f"(move={move_desc}, "
                        f"mut={len(mutator_state.board.collapsed_spaces)}, "
                        f"eng={len(next_via_engine.board.collapsed_spaces)}, "
                        f"details={details})"
                    )
                if (
                    mutator_state.board.eliminated_rings
                    != next_via_engine.board.eliminated_rings
                ):
                    details = self._diff_mapping_keys(
                        mutator_state.board.eliminated_rings,
                        next_via_engine.board.eliminated_rings,
                    )
                    raise RuntimeError(
                        "MovementMutator diverged from GameEngine.apply_move "
                        "for MOVE_STACK: board.eliminated_rings mismatch "
                        f"(move={move_desc}, "
                        f"mut={len(mutator_state.board.eliminated_rings)}, "
                        f"eng={len(next_via_engine.board.eliminated_rings)}, "
                        f"details={details})"
                    )
                if mutator_state.players != next_via_engine.players:
                    raise RuntimeError(
                        "MovementMutator diverged from GameEngine.apply_move "
                        "for MOVE_STACK: players mismatch "
                        f"(move={move_desc}, "
                        f"mut={mutator_state.players}, "
                        f"eng={next_via_engine.players})"
                    )

        elif move.type in (
            MoveType.OVERTAKING_CAPTURE,
            MoveType.CHAIN_CAPTURE,
            MoveType.CONTINUE_CAPTURE_SEGMENT,
        ):
            # Capture shadow contract: ensure CaptureMutator applies the same
            # board / player side-effects as GameEngine._apply_chain_capture /
            # GameEngine.apply_move for capture segments.
            mutator_state = state.model_copy(deep=True)
            capture_mutator = CaptureMutator()
            capture_mutator.apply(mutator_state, move)

            move_desc = self._describe_move(move)
            if mutator_state.board.stacks != next_via_engine.board.stacks:
                details = self._diff_mapping_keys(
                    mutator_state.board.stacks,
                    next_via_engine.board.stacks,
                )
                raise RuntimeError(
                    "CaptureMutator diverged from GameEngine.apply_move "
                    "for capture segment: board.stacks mismatch "
                    f"(move={move_desc}, "
                    f"mut={len(mutator_state.board.stacks)}, "
                    f"eng={len(next_via_engine.board.stacks)}, "
                    f"details={details})"
                )
            if mutator_state.board.markers != next_via_engine.board.markers:
                details = self._diff_mapping_keys(
                    mutator_state.board.markers,
                    next_via_engine.board.markers,
                )
                raise RuntimeError(
                    "CaptureMutator diverged from GameEngine.apply_move "
                    "for capture segment: board.markers mismatch "
                    f"(move={move_desc}, "
                    f"mut={len(mutator_state.board.markers)}, "
                    f"eng={len(next_via_engine.board.markers)}, "
                    f"details={details})"
                )
            if (
                mutator_state.board.collapsed_spaces
                != next_via_engine.board.collapsed_spaces
            ):
                details = self._diff_mapping_keys(
                    mutator_state.board.collapsed_spaces,
                    next_via_engine.board.collapsed_spaces,
                )
                raise RuntimeError(
                    "CaptureMutator diverged from GameEngine.apply_move "
                    "for capture segment: board.collapsed_spaces mismatch "
                    f"(move={move_desc}, "
                    f"mut={len(mutator_state.board.collapsed_spaces)}, "
                    f"eng={len(next_via_engine.board.collapsed_spaces)}, "
                    f"details={details})"
                )
            if (
                mutator_state.board.eliminated_rings
                != next_via_engine.board.eliminated_rings
            ):
                details = self._diff_mapping_keys(
                    mutator_state.board.eliminated_rings,
                    next_via_engine.board.eliminated_rings,
                )
                raise RuntimeError(
                    "CaptureMutator diverged from GameEngine.apply_move "
                    "for capture segment: board.eliminated_rings mismatch "
                    f"(move={move_desc}, "
                    f"mut={len(mutator_state.board.eliminated_rings)}, "
                    f"eng={len(next_via_engine.board.eliminated_rings)}, "
                    f"details={details})"
                )
            if mutator_state.players != next_via_engine.players:
                raise RuntimeError(
                    "CaptureMutator diverged from GameEngine.apply_move "
                    "for capture segment: players mismatch "
                    f"(move={move_desc}, "
                    f"mut={mutator_state.players}, "
                    f"eng={next_via_engine.players})"
                )

        elif move.type in (
            MoveType.PROCESS_LINE,
            MoveType.CHOOSE_LINE_REWARD,
            MoveType.LINE_FORMATION,
            MoveType.CHOOSE_LINE_OPTION,
        ):
            # Line-processing shadow contract: ensure LineMutator applies the
            # same board / player side-effects as
            # GameEngine._apply_line_formation / GameEngine.apply_move for
            # line-processing decisions.
            mutator_state = state.model_copy(deep=True)
            line_mutator = LineMutator()
            line_mutator.apply(mutator_state, move)

            # In some scenarios, a line-processing move can be the last
            # decision before the engine automatically advances into
            # territory processing and/or forced elimination for the *next*
            # player during GameEngine._update_phase + _end_turn. Those
            # host-level consequences may legitimately change stacks and
            # elimination counts beyond what LineMutator models (which is
            # strictly limited to _apply_line_formation side-effects).
            #
            # To avoid flagging these as false divergences, we relax the
            # per-move contract whenever we detect clear signs of extra
            # host-level work:
            # - The turn ended (current_player changed), or
            # - Additional automatic consequences increased
            #   total_rings_eliminated.
            extra_moves = (
                len(next_via_engine.move_history) - len(state.move_history)
            )
            turn_ended = next_via_engine.current_player != state.current_player
            forced_elimination_occurred = (
                next_via_engine.total_rings_eliminated
                > mutator_state.total_rings_eliminated
            )

            if extra_moves > 1 or turn_ended or forced_elimination_occurred:
                # GameEngine performed additional automatic work (e.g.
                # territory processing and/or forced elimination) after the
                # line, which LineMutator intentionally does not model.
                # Trust the canonical engine in these cases and skip strict
                # board/player parity for this move.
                pass
            else:
                move_desc = self._describe_move(move)
                if mutator_state.board.stacks != next_via_engine.board.stacks:
                    details = self._diff_mapping_keys(
                        mutator_state.board.stacks,
                        next_via_engine.board.stacks,
                    )
                    raise RuntimeError(
                        "LineMutator diverged from GameEngine.apply_move "
                        "for line-processing move: board.stacks mismatch "
                        f"(move={move_desc}, "
                        f"mut={len(mutator_state.board.stacks)}, "
                        f"eng={len(next_via_engine.board.stacks)}, "
                        f"details={details})"
                    )
                if mutator_state.board.markers != next_via_engine.board.markers:
                    details = self._diff_mapping_keys(
                        mutator_state.board.markers,
                        next_via_engine.board.markers,
                    )
                    raise RuntimeError(
                        "LineMutator diverged from GameEngine.apply_move "
                        "for line-processing move: board.markers mismatch "
                        f"(move={move_desc}, "
                        f"mut={len(mutator_state.board.markers)}, "
                        f"eng={len(next_via_engine.board.markers)}, "
                        f"details={details})"
                    )
                if (
                    mutator_state.board.collapsed_spaces
                    != next_via_engine.board.collapsed_spaces
                ):
                    details = self._diff_mapping_keys(
                        mutator_state.board.collapsed_spaces,
                        next_via_engine.board.collapsed_spaces,
                    )
                    raise RuntimeError(
                        "LineMutator diverged from GameEngine.apply_move "
                        "for line-processing move: board.collapsed_spaces mismatch "
                        f"(move={move_desc}, "
                        f"mut={len(mutator_state.board.collapsed_spaces)}, "
                        f"eng={len(next_via_engine.board.collapsed_spaces)}, "
                        f"details={details})"
                    )
                if (
                    mutator_state.board.eliminated_rings
                    != next_via_engine.board.eliminated_rings
                ):
                    details = self._diff_mapping_keys(
                        mutator_state.board.eliminated_rings,
                        next_via_engine.board.eliminated_rings,
                    )
                    raise RuntimeError(
                        "LineMutator diverged from GameEngine.apply_move "
                        "for line-processing move: board.eliminated_rings mismatch "
                        f"(move={move_desc}, "
                        f"mut={len(mutator_state.board.eliminated_rings)}, "
                        f"eng={len(next_via_engine.board.eliminated_rings)}, "
                        f"details={details})"
                    )
                if mutator_state.players != next_via_engine.players:
                    raise RuntimeError(
                        "LineMutator diverged from GameEngine.apply_move "
                        "for line-processing move: players mismatch "
                        f"(move={move_desc}, "
                        f"mut={mutator_state.players}, "
                        f"eng={next_via_engine.players})"
                    )

        elif move.type in (
            MoveType.PROCESS_TERRITORY_REGION,
            MoveType.ELIMINATE_RINGS_FROM_STACK,
            MoveType.TERRITORY_CLAIM,
            MoveType.CHOOSE_TERRITORY_OPTION,
        ):
            # Territory-processing shadow contract: ensure TerritoryMutator
            # applies the same board / player side-effects as
            # GameEngine._apply_territory_claim /
            # GameEngine._apply_forced_elimination as wired inside
            # GameEngine.apply_move for these move types.
            mutator_state = state.model_copy(deep=True)
            territory_mutator = TerritoryMutator()
            territory_mutator.apply(mutator_state, move)

            # GameEngine.apply_move may trigger additional host-level forced
            # elimination for the next player inside GameEngine._end_turn via
            # GameEngine._perform_forced_elimination_for_player. Those extra
            # eliminations are intentionally outside the per-move
            # TerritoryMutator's scope, so we detect them via a
            # total_rings_eliminated delta and skip strict board/player
            # parity in that case (mirroring the MOVE_STACK escape hatch).
            forced_elimination_occurred = (
                next_via_engine.total_rings_eliminated
                > mutator_state.total_rings_eliminated
            )

            move_desc = self._describe_move(move)
            if not forced_elimination_occurred:
                if mutator_state.board.stacks != next_via_engine.board.stacks:
                    details = self._diff_mapping_keys(
                        mutator_state.board.stacks,
                        next_via_engine.board.stacks,
                    )
                    raise RuntimeError(
                        "TerritoryMutator diverged from GameEngine.apply_move "
                        "for territory-processing move: board.stacks mismatch "
                        f"(move={move_desc}, "
                        f"mut={len(mutator_state.board.stacks)}, "
                        f"eng={len(next_via_engine.board.stacks)}, "
                        f"details={details})"
                    )
                if mutator_state.board.markers != next_via_engine.board.markers:
                    details = self._diff_mapping_keys(
                        mutator_state.board.markers,
                        next_via_engine.board.markers,
                    )
                    raise RuntimeError(
                        "TerritoryMutator diverged from GameEngine.apply_move "
                        "for territory-processing move: board.markers mismatch "
                        f"(move={move_desc}, "
                        f"mut={len(mutator_state.board.markers)}, "
                        f"eng={len(next_via_engine.board.markers)}, "
                        f"details={details})"
                    )
                if (
                    mutator_state.board.collapsed_spaces
                    != next_via_engine.board.collapsed_spaces
                ):
                    details = self._diff_mapping_keys(
                        mutator_state.board.collapsed_spaces,
                        next_via_engine.board.collapsed_spaces,
                    )
                    raise RuntimeError(
                        "TerritoryMutator diverged from GameEngine.apply_move "
                        "for territory-processing move: board.collapsed_spaces mismatch "
                        f"(move={move_desc}, "
                        f"mut={len(mutator_state.board.collapsed_spaces)}, "
                        f"eng={len(next_via_engine.board.collapsed_spaces)}, "
                        f"details={details})"
                    )
                if (
                    mutator_state.board.eliminated_rings
                    != next_via_engine.board.eliminated_rings
                ):
                    details = self._diff_mapping_keys(
                        mutator_state.board.eliminated_rings,
                        next_via_engine.board.eliminated_rings,
                    )
                    raise RuntimeError(
                        "TerritoryMutator diverged from GameEngine.apply_move "
                        "for territory-processing move: board.eliminated_rings mismatch "
                        f"(move={move_desc}, "
                        f"mut={len(mutator_state.board.eliminated_rings)}, "
                        f"eng={len(next_via_engine.board.eliminated_rings)}, "
                        f"details={details})"
                    )
                if mutator_state.players != next_via_engine.players:
                    raise RuntimeError(
                        "TerritoryMutator diverged from GameEngine.apply_move "
                        "for territory-processing move: players mismatch "
                        f"(move={move_desc}, "
                        f"mut={mutator_state.players}, "
                        f"eng={next_via_engine.players})"
                    )
            # When forced_elimination_occurred is True we intentionally relax the
            # per-move contract and trust the GameEngine's host-level
            # forced-elimination behaviour as canonical.

        # --- Optional mutator-first orchestration path ---
        if getattr(self, "_mutator_first_enabled", False):
            mutator_first_next = self._apply_move_with_mutators(state, move)
            move_desc = self._describe_move(move)

            # Compare key fields with the canonical result. Any divergence is
            # treated as a hard error to surface regressions early when the
            # flag is enabled in shadow mode.
            if mutator_first_next.board.stacks != next_via_engine.board.stacks:
                details = self._diff_mapping_keys(
                    mutator_first_next.board.stacks,
                    next_via_engine.board.stacks,
                )
                raise RuntimeError(
                    "Mutator-first apply_move diverged from GameEngine: "
                    "board.stacks mismatch "
                    f"(move={move_desc}, "
                    f"mut={len(mutator_first_next.board.stacks)}, "
                    f"eng={len(next_via_engine.board.stacks)}, "
                    f"details={details})"
                )
            if mutator_first_next.board.markers != next_via_engine.board.markers:
                details = self._diff_mapping_keys(
                    mutator_first_next.board.markers,
                    next_via_engine.board.markers,
                )
                raise RuntimeError(
                    "Mutator-first apply_move diverged from GameEngine: "
                    "board.markers mismatch "
                    f"(move={move_desc}, "
                    f"mut={len(mutator_first_next.board.markers)}, "
                    f"eng={len(next_via_engine.board.markers)}, "
                    f"details={details})"
                )
            if (
                mutator_first_next.board.collapsed_spaces
                != next_via_engine.board.collapsed_spaces
            ):
                details = self._diff_mapping_keys(
                    mutator_first_next.board.collapsed_spaces,
                    next_via_engine.board.collapsed_spaces,
                )
                raise RuntimeError(
                    "Mutator-first apply_move diverged from GameEngine: "
                    "board.collapsed_spaces mismatch "
                    f"(move={move_desc}, "
                    f"mut={len(mutator_first_next.board.collapsed_spaces)}, "
                    f"eng={len(next_via_engine.board.collapsed_spaces)}, "
                    f"details={details})"
                )
            if (
                mutator_first_next.board.eliminated_rings
                != next_via_engine.board.eliminated_rings
            ):
                details = self._diff_mapping_keys(
                    mutator_first_next.board.eliminated_rings,
                    next_via_engine.board.eliminated_rings,
                )
                raise RuntimeError(
                    "Mutator-first apply_move diverged from GameEngine: "
                    "board.eliminated_rings mismatch "
                    f"(move={move_desc}, "
                    f"mut={len(mutator_first_next.board.eliminated_rings)}, "
                    f"eng={len(next_via_engine.board.eliminated_rings)}, "
                    f"details={details})"
                )

            if mutator_first_next.players != next_via_engine.players:
                raise RuntimeError(
                    "Mutator-first apply_move diverged from GameEngine: "
                    "players mismatch (move="
                    f"{move_desc}, mut={mutator_first_next.players}, "
                    f"eng={next_via_engine.players})"
                )

            if (
                mutator_first_next.current_player
                != next_via_engine.current_player
            ):
                raise RuntimeError(
                    "Mutator-first apply_move diverged from GameEngine: "
                    "current_player mismatch (move="
                    f"{move_desc}, mut={mutator_first_next.current_player}, "
                    f"eng={next_via_engine.current_player})"
                )
            if (
                mutator_first_next.current_phase
                != next_via_engine.current_phase
            ):
                raise RuntimeError(
                    "Mutator-first apply_move diverged from GameEngine: "
                    "current_phase mismatch (move="
                    f"{move_desc}, mut={mutator_first_next.current_phase}, "
                    f"eng={next_via_engine.current_phase})"
                )
            if mutator_first_next.game_status != next_via_engine.game_status:
                raise RuntimeError(
                    "Mutator-first apply_move diverged from GameEngine: "
                    "game_status mismatch (move="
                    f"{move_desc}, mut={mutator_first_next.game_status}, "
                    f"eng={next_via_engine.game_status})"
                )
            if (
                mutator_first_next.chain_capture_state
                != next_via_engine.chain_capture_state
            ):
                raise RuntimeError(
                    "Mutator-first apply_move diverged from GameEngine: "
                    "chain_capture_state mismatch (move="
                    f"{move_desc}, mut={mutator_first_next.chain_capture_state}, "
                    f"eng={next_via_engine.chain_capture_state})"
                )
            if (
                mutator_first_next.must_move_from_stack_key
                != next_via_engine.must_move_from_stack_key
            ):
                raise RuntimeError(
                    "Mutator-first apply_move diverged from GameEngine: "
                    "must_move_from_stack_key mismatch (move="
                    f"{move_desc}, mut={mutator_first_next.must_move_from_stack_key}, "
                    f"eng={next_via_engine.must_move_from_stack_key})"
                )

        # Regardless of mutator-first being enabled or not, we always return
        # the canonical GameEngine state.
        return next_via_engine

    @staticmethod
    def _diff_mapping_keys(
        mut: Mapping[str, object],
        eng: Mapping[str, object],
        max_keys: int = 5,
    ) -> str:
        """Return a compact summary of key differences between two mappings.

        The result is designed for error messages and focuses on which keys
        differ, not full value dumps. It reports:

        - Keys only present in ``mut`` (``only_mut``)
        - Keys only present in ``eng`` (``only_eng``)
        - Keys present in both where the values differ (``changed``)

        Each bucket is truncated to ``max_keys`` entries with a ``(+ more)``
        suffix when additional keys exist.
        """
        mut_keys = set(mut.keys())
        eng_keys = set(eng.keys())

        only_mut_keys = sorted(mut_keys - eng_keys)
        only_eng_keys = sorted(eng_keys - mut_keys)
        changed_keys = sorted(
            key for key in mut_keys & eng_keys if mut.get(key) != eng.get(key)
        )

        def _fmt(keys: list[str]) -> str:
            display = list(keys)
            extra = ""
            if len(display) > max_keys:
                display = display[:max_keys]
                extra = ", (+ more)"
            inner = ", ".join(repr(k) for k in display)
            return f"[{inner}{extra}]"

        parts: list[str] = []
        if only_mut_keys:
            parts.append(f"only_mut={_fmt(only_mut_keys)}")
        if only_eng_keys:
            parts.append(f"only_eng={_fmt(only_eng_keys)}")
        if changed_keys:
            parts.append(f"changed={_fmt(changed_keys)}")

        return ", ".join(parts) if parts else "no_diff"

    @staticmethod
    def _describe_move(move: Move) -> str:
        """Return a compact string describing a move for diagnostics."""
        move_type = getattr(move.type, "name", str(move.type))
        from_key = move.from_pos.to_key() if move.from_pos else None
        to_key = move.to.to_key() if move.to else None
        return (
            f"id={move.id!r}, type={move_type}, player={move.player}, "
            f"from={from_key}, to={to_key}"
        )

    def _are_moves_equivalent(self, m1: Move, m2: Move) -> bool:
        """Helper to compare two moves for equivalence.

        This is currently used only for diagnostics and future-proofing; it is
        intentionally conservative and focuses on core identifying fields.
        """
        if m1.type != m2.type:
            return False
        if m1.player != m2.player:
            return False

        # Compare critical fields based on move type. This is a simplified
        # check; a full check would compare all fields.
        if m1.to and m2.to:
            if m1.to.to_key() != m2.to.to_key():
                return False
        elif m1.to != m2.to:  # One is None
            return False

        if m1.from_pos and m2.from_pos:
            if m1.from_pos.to_key() != m2.from_pos.to_key():
                return False
        elif m1.from_pos != m2.from_pos:
            return False

        return True
