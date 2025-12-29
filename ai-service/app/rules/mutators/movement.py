from app.models import GameState, Move
from app.rules.interfaces import Mutator

# from app.game_engine import GameEngine


class MovementMutator(Mutator):
    """Mutator that applies movement moves to game state.

    Delegates to GameEngine._apply_move_stack for the actual board mutation.
    Timeline bookkeeping (last_move_at, move_history) is handled by the
    orchestrating rules engine, not this mutator.
    """

    def apply(self, state: GameState, move: Move) -> None:
        # Delegate to GameEngine's static method
        from app.game_engine import GameEngine

        GameEngine._apply_move_stack(state, move)
        # Note: GameEngine._apply_move_stack does NOT update last_move_at
        # or move_history, so we must do it here if we want the mutator
        # to be fully self-contained. However, DefaultRulesEngine handles
        # history/timestamp updates centrally in _apply_move_with_mutators.
        #
        # The divergence error "board.stacks mismatch" suggests that
        # GameEngine._apply_move_stack might be doing something different
        # than what DefaultRulesEngine expects, OR that the mutator is
        # missing a side effect.
        #
        # Actually, looking at GameEngine._apply_move_stack, it modifies
        # the board in-place.
        #
        # The issue is likely that DefaultRulesEngine.apply_move calls
        # GameEngine.apply_move (which calls _apply_move_stack AND updates
        # history/timestamp/phase), and THEN calls MovementMutator.apply
        # on a copy.
        #
        # If GameEngine.apply_move does EXTRA things (like phase updates
        # or victory checks) that affect the board, and Mutator doesn't,
        # we get divergence.
        #
        # But _apply_move_stack is purely board-level.
        #
        # Wait, the error was:
        # RuntimeError: MovementMutator diverged ... board.stacks mismatch ...
        # details=only_mut=['3,6']
        #
        # This means the Mutator (which calls _apply_move_stack) resulted in
        # a stack at 3,6 that the Engine (GameEngine.apply_move) did NOT have.
        #
        # This implies GameEngine.apply_move -> _apply_move_stack did NOT
        # produce that stack, OR it produced it and then something else
        # removed it?
        #
        # Or maybe the Mutator is doing something EXTRA?
        #
        # The Mutator just calls GameEngine._apply_move_stack.
        # So if Mutator has it and Engine doesn't, it means
        # GameEngine.apply_move did NOT result in that stack being there.
        #
        # Ah, GameEngine.apply_move calls _apply_move_stack, THEN
        # _update_phase, THEN _check_victory.
        #
        # If _update_phase or _check_victory modifies the board (e.g.
        # forced elimination?), that would explain it.
        #
        # But the error was for MOVE_STACK.
        #
        # Let's look at the specific move: from=7,6 to=3,2.
        # Mutator result has stack at 3,6? Wait, 3,6 is neither from nor to.
        #
        # If 'only_mut' has '3,6', it means the Mutator state has a stack
        # at 3,6 that the Engine state does NOT.
        #
        # This suggests the Engine REMOVED the stack at 3,6.
        #
        # Why would the Engine remove a stack at 3,6 after a move from 7,6 to 3,2?
        #
        # Maybe 3,6 was the 'from' position? No, from is 7,6.
        # Maybe 3,6 was the 'to' position? No, to is 3,2.
        #
        # Wait, is 3,6 related to 3,2?
        #
        # If the Engine removed a stack at 3,6, it could be due to:
        # 1. Capture? (But this is MOVE_STACK)
        # 2. Territory processing? (But that happens in a separate phase)
        # 3. Forced elimination?
        #
        # If GameEngine.apply_move triggers _update_phase ->
        # _advance_to_territory_processing -> _end_turn ->
        # _perform_forced_elimination_for_player...
        #
        # If the move ended the turn, and the NEXT player had to force-eliminate,
        # that would modify the board in the Engine result.
        #
        # The DefaultRulesEngine says:
        # "EXCEPTION: If GameEngine.apply_move triggered additional
        # automatic moves (e.g. FORCED_ELIMINATION) or side effects via
        # phase transitions that the atomic PlacementMutator does not
        # perform, we skip the strict comparison to avoid false positives."
        #
        # It checks:
        # extra_moves = len(next_via_engine.move_history) - len(state.move_history)
        # turn_ended = next_via_engine.current_player != state.current_player
        # if extra_moves > 1 or turn_ended: pass
        #
        # So if the turn ended, it should have skipped the check.
        #
        # In the error case:
        # Player 2 moved.
        # If Player 2's turn ended, current_player would become 1.
        #
        # If the turn did NOT end, then current_player is still 2.
        #
        # If the turn didn't end, why did the board change?
        #
        # Maybe the move was invalid in the Engine but valid in the Mutator?
        # No, they use the same logic.
        #
        # Let's look at the error again:
        # move=id='simulated', type=MOVE_STACK, player=2, from=7,6, to=3,2
        # mut=6, eng=5 (stack counts)
        # only_mut=['3,6']
        #
        # So Mutator has a stack at 3,6. Engine does not.
        #
        # Was there a stack at 3,6 BEFORE the move?
        # If yes, Engine removed it.
        # If no, Mutator created it.
        #
        # 3,6 is not 7,6 (from) or 3,2 (to).
        #
        # Wait, could 3,6 be a typo for 7,6 or 3,2?
        # 7,6 -> 3,2 is a distance of 4 (dx=-4, dy=-4).
        # Path: 6,5 -> 5,4 -> 4,3.
        #
        # 3,6 is nowhere near that path.
        #
        # So it's likely a stack that existed before.
        #
        # If Engine removed it, and turn did NOT end...
        #
        # Did the move trigger a capture? No, MOVE_STACK.
        #
        # Did it trigger line processing?
        # _advance_to_line_processing -> _get_line_processing_moves.
        # If lines formed, phase becomes LINE_PROCESSING.
        # Board doesn't change yet.
        #
        # Did it trigger territory processing?
        # _advance_to_territory_processing -> find_disconnected_regions.
        # If regions found, phase becomes TERRITORY_PROCESSING.
        # Board doesn't change yet.
        #
        # If NO regions found -> _end_turn.
        #
        # If _end_turn is called, current_player changes.
        #
        # So if the check failed, it means current_player did NOT change, AND extra_moves <= 1.
        #
        # So the turn continued.
        #
        # If the turn continued, we are in MOVEMENT or CAPTURE or LINE_PROCESSING or TERRITORY_PROCESSING.
        #
        # If we are in MOVEMENT (e.g. after placement), no board change other than the move itself.
        #
        # Wait, is it possible that `GameEngine._apply_move_stack` ITSELF is buggy?
        #
        # It calls `_process_markers_along_path`.
        # It handles landing on own marker (self-elimination).
        #
        # If self-elimination happened at 3,2 (to_pos), that would affect the stack at 3,2.
        # But the diff is at 3,6.
        #
        # Let's verify if 3,6 is the 'from' position?
        # The move says from=7,6.
        #
        # Wait, I might be misinterpreting the coordinates or the diff.
        #
        # If `only_mut=['3,6']`, it means `mutator_state.board.stacks` has a
        # key '3,6' that `next_via_engine.board.stacks` does not.
        #
        # This implies the Engine deleted the stack at '3,6'.
        #
        # If the turn didn't end, how could a stack at 3,6 be deleted?
        #
        # Maybe `_apply_move_stack` has a side effect?
        #
        # Or maybe the Mutator is applying the move to a COPY, but the Engine is applying it to... a copy too.
        #
        # Let's look at `DefaultRulesEngine.apply_move`:
        # next_via_engine = GameEngine.apply_move(state, move)
        # mutator_state = state.model_copy(deep=True)
        # MovementMutator().apply(mutator_state, move)
        #
        # Both start from `state`.
        #
        # If `GameEngine.apply_move` removes a stack that `MovementMutator`
        # (which calls `_apply_move_stack`) does not...
        #
        # `GameEngine.apply_move` calls `_apply_move_stack`.
        #
        # So `next_via_engine` has the result of `_apply_move_stack`
        # PLUS `_update_phase` PLUS `_check_victory`.
        #
        # `mutator_state` has ONLY `_apply_move_stack`.
        #
        # So the difference MUST be in `_update_phase` or `_check_victory`.
        #
        # `_check_victory` checks for victory conditions. It doesn't modify the board (except maybe `game_status`).
        #
        # `_update_phase` advances phase. It calls `_advance_to_line_processing`
        # -> `_advance_to_territory_processing` -> `_end_turn`.
        #
        # If `_end_turn` is called, it might call `_perform_forced_elimination_for_player`.
        #
        # `_perform_forced_elimination_for_player` MODIFIES THE BOARD (eliminates rings).
        #
        # If forced elimination happens, the board changes.
        #
        # But if forced elimination happens, `_end_turn` was called.
        #
        # If `_end_turn` was called, `current_player` SHOULD change...
        # UNLESS the next player was forced to eliminate and then kept their turn?
        #
        # "If this player controls at least one stack but has no legal ...
        # actions, we must eliminate a cap ... Keep this player active and
        # begin their turn in MOVEMENT."
        #
        # AHA!
        #
        # If Player 1 moves, turn ends.
        # Player 2 is next.
        # Player 2 has stacks but no moves.
        # Player 2 is forced to eliminate.
        # Player 2 stays active (current_player is now 2).
        #
        # So `state.current_player` was 1.
        # `next_via_engine.current_player` is 2.
        #
        # So `turn_ended` should be True.
        #
        # Wait, in the error case:
        # "Starting game: RandomAI (P1) vs HeuristicAI (P2)"
        # "Game Over. Winner: Player 2"
        #
        # "Match 2/2"
        # "Starting game: HeuristicAI (P1) vs RandomAI (P2)"
        # "Error applying move: MovementMutator diverged ... player=2"
        #
        # So Player 2 (RandomAI) is moving.
        #
        # If Player 2 moves, and then the turn ends...
        # Next player is Player 1.
        #
        # If Player 1 is forced to eliminate, they stay active.
        # So `next_via_engine.current_player` becomes 1.
        #
        # `state.current_player` was 2.
        #
        # So `turn_ended` (next != state) is True.
        #
        # So the check SHOULD be skipped.
        #
        # Why wasn't it skipped?
        #
        # Maybe `next_via_engine.current_player` is STILL 2?
        #
        # If Player 2 moves, and `_update_phase` decides to KEEP Player 2 active?
        #
        # e.g. `MoveType.PLACE_RING` -> `_has_valid_movements` -> yes -> `GamePhase.MOVEMENT`. Player stays same.
        #
        # But this was `MOVE_STACK`.
        #
        # `MOVE_STACK` -> `_get_capture_moves`.
        # If captures exist -> `GamePhase.CAPTURE`. Player stays same.
        # If no captures -> `_advance_to_line_processing`.
        #
        # `_advance_to_line_processing` -> `_get_line_processing_moves`.
        # If lines -> `GamePhase.LINE_PROCESSING`. Player stays same.
        # If no lines -> `_advance_to_territory_processing`.
        #
        # `_advance_to_territory_processing` -> `find_disconnected_regions`.
        # If regions -> `GamePhase.TERRITORY_PROCESSING`. Player stays same.
        # If no regions -> `_end_turn`.
        #
        # So if `current_player` is unchanged, we must be in CAPTURE, LINE_PROCESSING, or TERRITORY_PROCESSING.
        #
        # In these phases, the board should NOT change (except for the move itself).
        #
        # So where did the stack at 3,6 go?
        #
        # If we are in `LINE_PROCESSING`, we just detected lines. We didn't process them yet.
        #
        # If we are in `TERRITORY_PROCESSING`, we just detected regions.
        #
        # So `GameEngine.apply_move` should match `_apply_move_stack`.
        #
        # UNLESS... `_apply_move_stack` itself has a bug where it modifies the board differently than `MovementMutator` expects?
        #
        # But `MovementMutator` CALLS `_apply_move_stack`. They are identical.
        #
        # So `GameEngine.apply_move` MUST be doing something extra.
        #
        # Is it possible `_check_victory` modifies the board?
        #
        # `_check_victory` reads `eliminated_rings`.
        #
        # Wait, `_check_victory` -> `game_state.game_status = FINISHED`.
        #
        # If game ends, `turn_ended` might be false (current_player doesn't change).
        #
        # But `extra_moves`?
        #
        # If game ends, no extra moves.
        #
        # Does `_check_victory` remove stacks? No.
        #
        # Let's look at the error again.
        # `mut=6, eng=5`.
        # Mutator has 6 stacks. Engine has 5.
        # Engine LOST a stack.
        #
        # And `only_mut=['3,6']`.
        # So the stack at 3,6 is missing in Engine.
        #
        # If `_apply_move_stack` was called by BOTH, then BOTH should have the same result from that call.
        #
        # So `GameEngine.apply_move` must have done something AFTER `_apply_move_stack` to remove the stack.
        #
        # The only thing that removes stacks is `_eliminate_top_ring_at` (if stack becomes empty) or `_apply_move_stack` (moving from).
        #
        # `_eliminate_top_ring_at` is called by:
        # - `_apply_place_ring` (no)
        # - `_apply_move_stack` (self-elimination) - but this is in both.
        # - `_apply_territory_claim` (not called here)
        # - `_perform_forced_elimination_for_player` (called by `_end_turn`)
        #
        # So it MUST be `_perform_forced_elimination_for_player`.
        #
        # This implies `_end_turn` was called.
        #
        # If `_end_turn` was called, `current_player` usually changes.
        #
        # UNLESS...
        #
        # "If we exhaust all players without finding any with material... Leave current_player unchanged... _check_victory will detect global stalemate".
        #
        # If global stalemate, `current_player` is unchanged.
        # But `_end_turn` doesn't remove stacks in that case.
        #
        # What if `_perform_forced_elimination_for_player` was called for the NEXT player, and that player happened to be the SAME player?
        #
        # e.g. 2-player game.
        # P2 moves. Turn ends.
        # Next is P1.
        # P1 has no material -> skipped.
        # Next is P2.
        # P2 has stacks but no moves -> Forced Elimination.
        #
        # So P2 moves, then P2 is forced to eliminate immediately.
        #
        # In this case, `current_player` starts as 2, and ends as 2.
        #
        # So `turn_ended` is False.
        #
        # And `extra_moves`?
        # `_perform_forced_elimination_for_player` calls `_apply_forced_elimination`.
        # `_apply_forced_elimination` calls `_eliminate_top_ring_at`.
        #
        # Does it add a move to history?
        #
        # `_apply_forced_elimination` does NOT add to `move_history`.
        # `_perform_forced_elimination_for_player` calls `_apply_forced_elimination`.
        #
        # `GameEngine.apply_move` adds the ORIGINAL move to history.
        #
        # But `_apply_forced_elimination` is usually a MoveType.
        #
        # Wait, `_apply_forced_elimination` (the static method) takes a `move` object.
        #
        # `_perform_forced_elimination_for_player` generates a `FORCED_ELIMINATION` move and calls `_apply_forced_elimination`.
        #
        # BUT it does NOT append it to `move_history`!
        #
        # `GameEngine.apply_move` appends the *input* move.
        #
        # So `extra_moves` is 0 (or 1 if we count the input move vs state).
        # `len(next_via_engine.move_history) - len(state.move_history)` is 1.
        #
        # So `extra_moves > 1` is False.
        #
        # So the check runs.
        #
        # And fails because the board has changed (forced elimination removed a stack).
        #
        # FIX: We need to detect if `_perform_forced_elimination_for_player` occurred.
        #
        # We can check if `total_rings_eliminated` changed more than expected?
        # Or check if `eliminated_rings` changed?
        #
        # Or simply relax the check in `DefaultRulesEngine` to account for this case.
        #
        # If `current_player` is same, but `eliminated_rings` changed in a way that `MovementMutator` didn't predict...
        #
        # `MovementMutator` predicts self-elimination (landing on own marker).
        #
        # If `next_via_engine` has MORE eliminated rings than `mutator_state`, it might be forced elimination.
        #
        # But `DefaultRulesEngine` is in `ai-service/app/rules/default_engine.py`.
        #
        # I should modify `DefaultRulesEngine` to handle this case.
        #
        # Specifically:
        # If `current_player` is unchanged, but we suspect a forced elimination loop (P2 -> P1 skip -> P2 forced elim), we should allow divergence.
        #
        # How to detect?
        #
        # If `next_via_engine` has a different `total_rings_eliminated` than `mutator_state`?
        #
        # Yes. `MovementMutator` handles self-elimination. So if `next_via_engine` has even MORE eliminated rings, it must be from forced elimination (since that's the only other automatic elimination source).
        #
        # So, in `DefaultRulesEngine.apply_move`, inside the `MOVE_STACK` block (and others), we should add a check.
        #
        # Actually, `DefaultRulesEngine` already has logic to skip checks:
        #
        # ```python
        #     extra_moves = (
        #         len(next_via_engine.move_history) - len(state.move_history)
        #     )
        #     turn_ended = (
        #         next_via_engine.current_player != state.current_player
        #     )
        #     if extra_moves > 1 or turn_ended:
        #         pass
        # ```
        #
        # I should update this condition.
        #
        # If `next_via_engine.total_rings_eliminated != mutator_state.total_rings_eliminated`, we should probably skip (assuming the mutator handles immediate effects correctly).
        #
        # Wait, `MovementMutator` handles self-elimination. So `mutator_state` WILL have increased elimination if self-elim occurred.
        #
        # So we should check if `next` has *more* elimination than `mutator`.
        #
        # `if next_via_engine.total_rings_eliminated > mutator_state.total_rings_eliminated:`
        #
        # This confirms extra elimination happened (forced).
        #
        # Let's modify `ai-service/app/rules/default_engine.py`.

        state.last_move_at = move.timestamp
