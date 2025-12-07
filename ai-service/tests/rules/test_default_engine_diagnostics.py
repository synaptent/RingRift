import os
import sys
from datetime import datetime

import pytest

# Ensure app package is importable when running pytest from ai-service/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from app.models import GamePhase, Move, MoveType, Position, RingStack  # noqa: E402
from app.game_engine import GameEngine  # noqa: E402
from app.rules.default_engine import DefaultRulesEngine  # noqa: E402
from tests.rules.helpers import _make_base_game_state  # noqa: E402


# Skip shadow contract tests when RINGRIFT_SKIP_SHADOW_CONTRACTS is set.
_skip_if_shadow_contracts_disabled = pytest.mark.skipif(
    os.environ.get("RINGRIFT_SKIP_SHADOW_CONTRACTS") == "true",
    reason="Shadow contracts disabled via RINGRIFT_SKIP_SHADOW_CONTRACTS",
)


@_skip_if_shadow_contracts_disabled
def test_capture_mutator_shadow_contract_uses_diff_mapping_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CaptureMutator shadow contract should surface mapping-key diagnostics.

    This test deliberately forces a divergence between the CaptureMutator
    shadow path and the canonical GameEngine.apply_move result, then asserts
    that the raised RuntimeError includes details from _diff_mapping_keys.

    We patch CaptureMutator inside DefaultRulesEngine to be a no-op so that
    the mutator-side board state remains at the pre-move baseline, while the
    canonical engine still applies the full overtaking-capture semantics.
    """

    # Patch CaptureMutator in the DefaultRulesEngine module to be a no-op,
    # ensuring a deterministic divergence on board.stacks.
    class _NoOpCaptureMutator:  # pragma: no cover - behaviour tested via engine
        def apply(self, state, move) -> None:  # type: ignore[no-untyped-def]
            # Intentionally do nothing so the mutator-side board remains
            # unchanged while GameEngine.apply_move performs a real capture.
            return

    def _fake_diff_mapping_keys(
        mut: dict[str, object],
        eng: dict[str, object],
        max_keys: int = 5,
    ) -> str:
        # Sentinel value to assert that _diff_mapping_keys was invoked and
        # its result was threaded through to the error message.
        return "TEST-DETAILS-FROM-DIFF"

    monkeypatch.setattr(
        "app.rules.default_engine.CaptureMutator",
        _NoOpCaptureMutator,
    )
    monkeypatch.setattr(
        "app.rules.default_engine.DefaultRulesEngine._diff_mapping_keys",
        staticmethod(_fake_diff_mapping_keys),  # preserve staticmethod semantics
    )

    # Construct the same minimal overtaking-capture scenario used in the
    # equivalence tests to guarantee a valid capture move for player 1.
    base_state = _make_base_game_state()

    # Capture operations require MOVEMENT phase.
    base_state.current_phase = GamePhase.MOVEMENT

    attacker_pos = Position(x=0, y=0)
    attacker_key = attacker_pos.to_key()
    target_pos = Position(x=0, y=2)
    target_key = target_pos.to_key()
    landing_pos = Position(x=0, y=5)

    attacker_stack = RingStack(
        position=attacker_pos,
        rings=[1, 1],
        stackHeight=2,
        capHeight=2,
        controllingPlayer=1,
    )
    target_stack = RingStack(
        position=target_pos,
        rings=[2],
        stackHeight=1,
        capHeight=1,
        controllingPlayer=2,
    )

    base_state.board.stacks[attacker_key] = attacker_stack
    base_state.board.stacks[target_key] = target_stack

    now = datetime.now()
    move = Move(
        id="m-cap-diagnostics",
        type=MoveType.OVERTAKING_CAPTURE,
        player=1,
        from_pos=attacker_pos,
        to=landing_pos,
        capture_target=target_pos,
        timestamp=now,
        thinkTime=0,
        moveNumber=1,
    )

    # Sanity check: the canonical engine must accept and apply this move
    # without error so that the divergence comes solely from the mutator
    # shadow path.
    _ = GameEngine.apply_move(base_state, move)

    engine = DefaultRulesEngine()

    with pytest.raises(RuntimeError) as excinfo:
        engine.apply_move(base_state, move)

    message = str(excinfo.value)

    # The error should clearly indicate the CaptureMutator shadow contract
    # and the specific mapping field that diverged.
    assert "CaptureMutator diverged from GameEngine.apply_move" in message
    assert "board.stacks mismatch" in message

    # Most importantly, the message should include the sentinel details from
    # our patched _diff_mapping_keys implementation, proving that mapping
    # diagnostics are wired through the shadow contract.
    assert "TEST-DETAILS-FROM-DIFF" in message


@_skip_if_shadow_contracts_disabled
def test_territory_mutator_shadow_contract_uses_diff_mapping_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TerritoryMutator shadow contract should surface mapping-key diagnostics.

    This test mirrors the synthetic PROCESS_TERRITORY_REGION scenario from the
    equivalence suite but patches TerritoryMutator to introduce a bogus stack
    key, forcing a divergence on board.stacks. It also patches
    _diff_mapping_keys to return a sentinel string so we can assert that its
    output is threaded through to the error message.
    """

    from app.board_manager import BoardManager  # noqa: WPS433,E402
    from app.models import GamePhase, Territory  # noqa: WPS433,E402

    # Patch TerritoryMutator in the DefaultRulesEngine module so that it first
    # applies the real mutator behaviour, then introduces a bogus stack entry
    # to force a mapping mismatch.
    class _BogusTerritoryMutator:  # pragma: no cover - behaviour tested via engine
        def apply(self, state, move) -> None:  # type: ignore[no-untyped-def]
            from app.rules.mutators.territory import (  # noqa: WPS433,E402
                TerritoryMutator as _RealTerritoryMutator,
            )

            _RealTerritoryMutator().apply(state, move)

            bogus_pos = Position(x=9, y=9)
            bogus_key = bogus_pos.to_key()
            if bogus_key not in state.board.stacks:
                rings = [1]
                state.board.stacks[bogus_key] = RingStack(
                    position=bogus_pos,
                    rings=rings,
                    stackHeight=len(rings),
                    capHeight=len(rings),
                    controllingPlayer=1,
                )

    def _fake_diff_mapping_keys_territory(
        mut: dict[str, object],
        eng: dict[str, object],
        max_keys: int = 5,
    ) -> str:
        # Sentinel value to assert that _diff_mapping_keys was invoked for the
        # territory shadow contract.
        return "TERRITORY-DETAILS-FROM-DIFF"

    monkeypatch.setattr(
        "app.rules.default_engine.TerritoryMutator",
        _BogusTerritoryMutator,
    )
    monkeypatch.setattr(
        "app.rules.default_engine.DefaultRulesEngine._diff_mapping_keys",
        staticmethod(_fake_diff_mapping_keys_territory),
    )

    # Build the minimal disconnected-region scenario from the equivalence
    # tests to obtain a valid PROCESS_TERRITORY_REGION move.
    state = _make_base_game_state()
    state.current_phase = GamePhase.TERRITORY_PROCESSING

    board = state.board
    region_pos = Position(x=5, y=5)
    region_key = region_pos.to_key()

    region_stack = RingStack(
        position=region_pos,
        rings=[2],
        stackHeight=1,
        capHeight=1,
        controllingPlayer=2,
    )
    board.stacks[region_key] = region_stack

    outside_pos = Position(x=7, y=7)
    outside_key = outside_pos.to_key()
    p1_rings = [1, 1]
    outside_stack = RingStack(
        position=outside_pos,
        rings=p1_rings,
        stackHeight=len(p1_rings),
        capHeight=len(p1_rings),
        controllingPlayer=1,
    )
    board.stacks[outside_key] = outside_stack

    region_territory = Territory(
        spaces=[region_pos],
        controllingPlayer=1,
        isDisconnected=True,
    )

    monkeypatch.setattr(
        BoardManager,
        "find_disconnected_regions",
        staticmethod(lambda b, moving_player: [region_territory]),
    )
    monkeypatch.setattr(
        BoardManager,
        "get_border_marker_positions",
        staticmethod(lambda spaces, b: []),
    )

    territory_moves = GameEngine._get_territory_processing_moves(state, 1)
    assert territory_moves, "Expected at least one territory move"

    process_region_moves = [
        m for m in territory_moves if m.type == MoveType.PROCESS_TERRITORY_REGION
    ]
    assert process_region_moves, (
        "Expected at least one PROCESS_TERRITORY_REGION move from "
        "_get_territory_processing_moves",
    )
    move = process_region_moves[0]

    engine = DefaultRulesEngine()

    with pytest.raises(RuntimeError) as excinfo:
        engine.apply_move(state, move)

    message = str(excinfo.value)

    # The error should clearly indicate the TerritoryMutator shadow contract
    # and the specific mapping field that diverged.
    assert "TerritoryMutator diverged from GameEngine.apply_move" in message
    assert "board.stacks mismatch" in message

    # The message should include the sentinel details from our patched
    # _diff_mapping_keys implementation, proving that mapping diagnostics are
    # wired through the territory shadow contract as well.
    assert "TERRITORY-DETAILS-FROM-DIFF" in message
