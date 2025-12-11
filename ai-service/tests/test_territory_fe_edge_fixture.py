from __future__ import annotations

"""
Territory / Forced-Elimination Edge Fixture
(canonical_square8_regen k90, k=89 pre-move).

This test mirrors the TS-side territory/FE edge-case tests built on
`territoryFeEdgeFixture.ts` and the parity bundle
`canonical_square8_regen__221a3037-3ca1-4d83-bccf-32d32edf0885__k90.state_bundle.json`.

Scope:

- Reconstruct the late-game square-8 GameState slice around ts_k=89 from the
  parity state bundle (using the Python contract deserializer).
- Define the same curated disconnected region for player 1 that the TS tests
  use to exercise self-elimination prerequisites and internal elimination
  crediting.
- Use the Python GameEngine and BoardManager to assert analogous invariants:
  - Self-elimination prerequisite depends on stacks outside the region.
  - Applying a PROCESS_TERRITORY_REGION decision over the curated region
    eliminates the internal stack, collapses the region to player 1, and
    credits all internal rings to player 1's elimination counters.

The goal is *not* to change Python semantics here, but to surface any
divergence from the TS/SSoT expectations as a regression test for later
H3/H5 work.
"""

import json
import os
import sys
from pathlib import Path

import pytest

# Ensure `app.*` imports resolve when running pytest from ai-service/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from app.board_manager import BoardManager  # noqa: E402
from app.game_engine import GameEngine  # noqa: E402
from app.models import (  # noqa: E402
    BoardType,
    GameState,
    Move,
    MoveType,
    Position,
    Territory,
)
from app.rules.serialization import deserialize_game_state  # noqa: E402


# Path to the canonical parity state bundle this fixture is derived from.
AI_SERVICE_ROOT = Path(__file__).resolve().parents[2]
BUNDLE_PATH = (
    AI_SERVICE_ROOT
    / "parity_failures"
    / (
        "canonical_square8_regen__"
        "221a3037-3ca1-4d83-bccf-32d32edf0885__k90.state_bundle.json"
    )
)

# Skip all tests in this module if the fixture bundle doesn't exist
# (it's gitignored and only generated during parity debugging)
pytestmark = pytest.mark.skipif(
    not BUNDLE_PATH.exists(),
    reason=f"Parity fixture bundle not found: {BUNDLE_PATH}",
)


def load_python_state_from_bundle(k: int = 89) -> GameState:
    """
    Load the Python-side contract GameState for a given ts_k from the bundle.

    The bundle format contains:
      - python_states: mapping from ts_k to serialized Python GameState
        (contract form)
      - ts_states:     mapping from ts_k to serialized TS GameState
        (contract form)

    We intentionally use python_states[k] here, mirroring the TS test fixture,
    which was reconstructed from python_states["89"].board.
    """
    with BUNDLE_PATH.open("r", encoding="utf-8") as f:
        bundle = json.load(f)

    py_states = bundle.get("python_states") or {}
    raw_state = py_states.get(str(k))
    if raw_state is None:
        raise KeyError(
            f"No python_states entry for k={k} in bundle {BUNDLE_PATH}"
        )

    state = deserialize_game_state(raw_state)
    return state


# Curated disconnected region for player 1, mirroring
# territoryFeEdgeRegionForPlayer1 from
# tests/fixtures/territoryFeEdgeFixture.ts.
TERRITORY_FE_EDGE_REGION_P1: Territory = Territory(
    spaces=[
        Position(x=6, y=1),
        Position(x=6, y=2),
        Position(x=6, y=3),
    ],
    controllingPlayer=1,
    isDisconnected=True,
)

# SQ8-A mini-region for player 1, derived from the same k89/k90 bundle board.
# Geometry mirrors territoryFeMiniRegionForPlayer1 on the TS side:
#   - (0,7): P1 stack, height 1
#   - (1,7): P2 stack, height 1
#   - (1,6): P2 stack, height 2
# None of these spaces are collapsed prior to processing.
TERRITORY_FE_MINI_REGION_P1: Territory = Territory(
    spaces=[
        Position(x=0, y=7),
        Position(x=1, y=7),
        Position(x=1, y=6),
    ],
    controllingPlayer=1,
    isDisconnected=True,
)


def _region_keys(region: Territory) -> set[str]:
    """Helper: return a set of position keys for the region's spaces."""
    return {pos.to_key() for pos in region.spaces}


class TestTerritoryFeEdgeFixture:
    """Python mirror of the TS-side territory/FE edge fixture tests."""

    def test_self_elimination_prerequisite_depends_on_stacks_outside_region(
        self,
    ) -> None:
        """
        Self-elimination prerequisite: player must have a stack outside region.

        Mirrors the TS test that uses canProcessTerritoryRegion over the
        curated region on the k89 board:

        - With the original board, player 1 has stacks both inside and outside
          the region, so the region is processable.
        - After removing all player-1 stacks outside the region, the region is
          no longer processable for player 1.
        """
        game_state = load_python_state_from_bundle(k=89)

        assert game_state.board.type == BoardType.SQUARE8
        assert game_state.current_phase.value == "territory_processing"
        assert game_state.current_player == 1

        region = TERRITORY_FE_EDGE_REGION_P1
        region_keys = _region_keys(region)

        # Collect all P1 stacks outside the curated region.
        p1_stacks = BoardManager.get_player_stacks(
            game_state.board,
            player_number=1,
        )
        outside_keys = [
            stack.position.to_key()
            for stack in p1_stacks
            if stack.position.to_key() not in region_keys
        ]

        # The fixture must have at least one P1 stack outside the region
        # for the self-elimination prerequisite to be meaningful.
        assert outside_keys, (
            "Expected at least one P1 stack outside the curated region"
        )

        # With the original board, the region should satisfy the
        # self-elimination prerequisite.
        can_process = GameEngine._can_process_disconnected_region(
            game_state,
            region,
            player_number=1,
        )
        assert can_process is True

        # Now construct a variant of the state where all P1 stacks outside the
        # curated region have been removed.
        board_no_outside = game_state.board.model_copy()
        # Follow GameEngine.apply_move pattern: copy stacks dict explicitly.
        board_no_outside.stacks = game_state.board.stacks.copy()
        for key in outside_keys:
            board_no_outside.stacks.pop(key, None)

        state_no_outside = game_state.model_copy(
            update={"board": board_no_outside},
        )

        can_process_after = GameEngine._can_process_disconnected_region(
            state_no_outside,
            region,
            player_number=1,
        )
        assert (
            can_process_after is False
        ), "Region should not be processable when P1 has no stacks outside it"

    def test_internal_eliminations_and_credit_on_k89_edge_fixture(
        self,
    ) -> None:
        """
        Internal eliminations + crediting on the k89 territory/FE edge fixture.
 
        Mirrors the TS test that uses applyTerritoryRegion over the curated
        region:
 
        - The tall mixed stack at (6,1) is fully eliminated.
        - All region spaces are collapsed to player 1.
        - All internal rings from the eliminated stack are credited to
          player 1.
        - The input GameState remains unchanged (immutability of apply_move).
        """
        base_state = load_python_state_from_bundle(k=89)
 
        assert base_state.board.type == BoardType.SQUARE8
        assert base_state.current_phase.value == "territory_processing"
        assert base_state.current_player == 1
 
        region = TERRITORY_FE_EDGE_REGION_P1
        internal_pos = Position(x=6, y=1)
        internal_key = internal_pos.to_key()
 
        internal_stack = base_state.board.stacks.get(internal_key)
        assert internal_stack is not None, (
            "Expected a stack at (6,1) in the fixture board"
        )
        internal_height = internal_stack.stack_height
        assert internal_height > 0
 
        # Player-1 elimination stats and global elimination count before
        # applying the move.
        before_p1_elims = 0
        for p in base_state.players:
            if p.player_number == 1:
                before_p1_elims = p.eliminated_rings
                break
 
        before_total_elims = base_state.total_rings_eliminated
 
        # Construct a PROCESS_TERRITORY_REGION move that carries the curated
        # region geometry explicitly via disconnectedRegions, so that
        # _apply_territory_claim processes exactly this region.
        move = Move(  # type: ignore[call-arg]
            id="process-territory-fe-edge-k89",
            type=MoveType.PROCESS_TERRITORY_REGION,
            player=base_state.current_player,
            to=internal_pos,
            disconnectedRegions=(region,),
            timestamp=base_state.last_move_at,
            thinkTime=0,
            moveNumber=len(base_state.move_history) + 1,
        )
 
        # Apply the move via the canonical GameEngine.apply_move surface, which
        # clones the input state and invokes _apply_territory_claim internally.
        next_state = GameEngine.apply_move(base_state, move)
 
        # Input state must remain unchanged.
        assert internal_key in base_state.board.stacks
        assert base_state.total_rings_eliminated == before_total_elims
 
        # Internal stack is removed on the resulting board.
        assert internal_key not in next_state.board.stacks
 
        # All region spaces are collapsed to player 1, with no stacks or
        # markers remaining on those cells.
        for pos in region.spaces:
            key = pos.to_key()
            owner = next_state.board.collapsed_spaces.get(key)
            assert owner == 1, (
                f"Expected collapsed owner 1 at {key}, got {owner!r}"
            )
            assert key not in next_state.board.stacks
            assert key not in next_state.board.markers
 
        # All rings from the internal stack are credited to player 1 as
        # eliminations, and the global elimination count increases by the same
        # amount. This mirrors the TS invariant that internal eliminations are
        # fully credited to the processing player.
        after_p1_elims = 0
        for p in next_state.players:
            if p.player_number == 1:
                after_p1_elims = p.eliminated_rings
                break
 
        after_total_elims = next_state.total_rings_eliminated
 
        assert after_p1_elims == before_p1_elims + internal_height, (
            "Expected all internal rings from (6,1) to be credited to "
            "player 1; "
            f"before={before_p1_elims}, internal_height={internal_height}, "
            f"after={after_p1_elims}"
        )
 
        assert after_total_elims == before_total_elims + internal_height, (
            "Expected total_rings_eliminated to increase by the internal "
            "stack height; "
            f"before={before_total_elims}, internal_height={internal_height}, "
            f"after={after_total_elims}"
        )
 
    def test_mini_region_self_elimination_prerequisite_depends_on_stacks_outside_region(  # noqa: E501
        self,
    ) -> None:
        """
        Self-elimination prerequisite for the SQ8-A mini-region.
 
        Mirrors the TS test that uses canProcessTerritoryRegion over the
        SQ8-A mini-region on the k89 board:
 
        - With the original board, player 1 has stacks both inside and
          outside the mini-region, so the region is processable.
        - After removing all player-1 stacks outside the mini-region,
          the region is no longer processable for player 1.
        """
        game_state = load_python_state_from_bundle(k=89)
 
        assert game_state.board.type == BoardType.SQUARE8
        assert game_state.current_phase.value == "territory_processing"
        assert game_state.current_player == 1
 
        region = TERRITORY_FE_MINI_REGION_P1
        region_keys = _region_keys(region)
 
        # Collect all P1 stacks outside the SQ8-A mini-region.
        p1_stacks = BoardManager.get_player_stacks(
            game_state.board,
            player_number=1,
        )
        outside_keys = [
            stack.position.to_key()
            for stack in p1_stacks
            if stack.position.to_key() not in region_keys
        ]
 
        # The fixture must have at least one P1 stack outside the mini-region
        # for the self-elimination prerequisite to be meaningful.
        assert outside_keys, (
            "Expected at least one P1 stack outside SQ8-A mini-region"
        )
 
        # With the original board, the mini-region should satisfy the
        # self-elimination prerequisite.
        can_process = GameEngine._can_process_disconnected_region(
            game_state,
            region,
            player_number=1,
        )
        assert can_process is True
 
        # Now construct a variant of the state where all P1 stacks outside the
        # mini-region have been removed.
        board_no_outside = game_state.board.model_copy()
        # Follow GameEngine.apply_move pattern: copy stacks dict explicitly.
        board_no_outside.stacks = game_state.board.stacks.copy()
        for key in outside_keys:
            board_no_outside.stacks.pop(key, None)
 
        state_no_outside = game_state.model_copy(
            update={"board": board_no_outside},
        )
 
        can_process_after = GameEngine._can_process_disconnected_region(
            state_no_outside,
            region,
            player_number=1,
        )
        assert (
            can_process_after is False
        ), (
            "SQ8-A mini-region should not be processable when P1 has no "
            "stacks outside it"
        )
 
    def test_mini_region_internal_eliminations_and_credit_on_k89_edge_fixture(
        self,
    ) -> None:
        """
        Internal eliminations + crediting on the SQ8-A mini-region fixture.
 
        Mirrors the TS test that uses applyTerritoryRegion over the SQ8-A
        mini-region:
 
        - All stacks on (0,7), (1,7), and (1,6) are eliminated.
        - All mini-region spaces are collapsed to player 1.
        - The total internal ring count across the mini-region is fully
          credited to player 1 and to the global elimination counter.
        - The input GameState remains unchanged (immutability of apply_move).
        """
        base_state = load_python_state_from_bundle(k=89)
 
        assert base_state.board.type == BoardType.SQUARE8
        assert base_state.current_phase.value == "territory_processing"
        assert base_state.current_player == 1
 
        region = TERRITORY_FE_MINI_REGION_P1
 
        # Representative internal stack at (1,6) used as the move target.
        internal_pos = Position(x=1, y=6)
        internal_key = internal_pos.to_key()
        internal_stack = base_state.board.stacks.get(internal_key)
        assert internal_stack is not None, (
            "Expected a stack at (1,6) in the SQ8-A mini-region"
        )
        internal_height = internal_stack.stack_height
        assert internal_height > 0
 
        # Total internal height across all region spaces in the mini-region,
        # computed directly from the fixture board (expected to be 4).
        total_internal_height = 0
        for pos in region.spaces:
            key = pos.to_key()
            stack = base_state.board.stacks.get(key)
            if stack is not None:
                total_internal_height += stack.stack_height
        assert total_internal_height >= internal_height
        assert total_internal_height > 0
 
        # Player-1 elimination stats and global elimination count before
        # applying the move.
        before_p1_elims = 0
        for p in base_state.players:
            if p.player_number == 1:
                before_p1_elims = p.eliminated_rings
                break
 
        before_total_elims = base_state.total_rings_eliminated
 
        # Construct a PROCESS_TERRITORY_REGION move that carries the SQ8-A
        # mini-region geometry explicitly via disconnectedRegions, so that
        # _apply_territory_claim processes exactly this mini-region.
        move = Move(  # type: ignore[call-arg]
            id="process-territory-fe-mini-region-k89",
            type=MoveType.PROCESS_TERRITORY_REGION,
            player=base_state.current_player,
            to=internal_pos,
            disconnectedRegions=(region,),
            timestamp=base_state.last_move_at,
            thinkTime=0,
            moveNumber=len(base_state.move_history) + 1,
        )
 
        # Apply the move via the canonical GameEngine.apply_move surface, which
        # clones the input state and invokes _apply_territory_claim internally.
        next_state = GameEngine.apply_move(base_state, move)
 
        # Input state must remain unchanged.
        assert internal_key in base_state.board.stacks
        assert base_state.total_rings_eliminated == before_total_elims
 
        # All mini-region stacks are removed on the resulting board and the
        # mini-region spaces are collapsed to player 1.
        for pos in region.spaces:
            key = pos.to_key()
            owner = next_state.board.collapsed_spaces.get(key)
            assert owner == 1, (
                f"Expected collapsed owner 1 at {key}, got {owner!r}"
            )
            assert key not in next_state.board.stacks
            assert key not in next_state.board.markers
 
        # All rings from stacks in the SQ8-A mini-region are credited to
        # player 1 as eliminations, and the global elimination count
        # increases by the same amount. This mirrors the TS invariant that
        # internal eliminations are fully credited to the processing player.
        after_p1_elims = 0
        for p in next_state.players:
            if p.player_number == 1:
                after_p1_elims = p.eliminated_rings
                break
 
        after_total_elims = next_state.total_rings_eliminated
 
        assert after_p1_elims == before_p1_elims + total_internal_height, (
            "Expected all internal rings from SQ8-A mini-region to be "
            "credited to player 1; "
            f"before={before_p1_elims}, "
            f"total_internal_height={total_internal_height}, "
            f"after={after_p1_elims}"
        )
 
        assert after_total_elims == (
            before_total_elims + total_internal_height
        ), (
            "Expected total_rings_eliminated to increase by "
            "total_internal_height; "
            f"before={before_total_elims}, "
            f"total_internal_height={total_internal_height}, "
            f"after={after_total_elims}"
        )
 
 
if __name__ == "__main__":  # pragma: no cover - manual debug helper
    # Allow running this module directly for quick iteration:
    #   cd ai-service && python -m pytest \
    #       tests/test_territory_fe_edge_fixture.py -q
    pytest.main([__file__])