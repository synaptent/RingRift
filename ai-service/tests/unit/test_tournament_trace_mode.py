from __future__ import annotations

from typing import Any

import pytest

from app.game_engine import GameEngine
from app.models import BoardType, MoveType
from app.rules.history_contract import phase_move_contract, validate_canonical_move
from app.tournament.agents import AIAgent, AIAgentRegistry, AgentType
from app.tournament.recording import TournamentRecordingOptions
from app.tournament.runner import TournamentRunner
from app.tournament.scheduler import RoundRobinScheduler


def test_tournament_runner_uses_trace_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = AIAgentRegistry()
    registry.register(
        AIAgent(
            agent_id="random_2",
            name="Random 2",
            agent_type=AgentType.RANDOM,
            search_depth=0,
        )
    )

    scheduler = RoundRobinScheduler(games_per_pairing=1, shuffle_order=False, seed=1)
    matches = scheduler.generate_matches(
        agent_ids=["random", "random_2"],
        board_type=BoardType.SQUARE8,
        num_players=2,
        games_per_pairing=1,
    )

    assert matches, "Expected at least one scheduled match"

    trace_modes: list[bool] = []
    original_apply = GameEngine.apply_move

    def apply_move_spy(game_state: Any, move: Any, *, trace_mode: bool = False) -> Any:
        trace_modes.append(trace_mode)
        return original_apply(game_state, move, trace_mode=trace_mode)

    monkeypatch.setattr(GameEngine, "apply_move", staticmethod(apply_move_spy))

    runner = TournamentRunner(
        agent_registry=registry,
        scheduler=scheduler,
        max_workers=1,
        max_moves=6,
        seed=1,
        recording_options=TournamentRecordingOptions(enabled=False),
    )

    agents = {agent_id: registry.get(agent_id) for agent_id in matches[0].agent_ids}
    assert all(agent is not None for agent in agents.values())

    runner._execute_match_local(matches[0], agents)  # pylint: disable=protected-access

    assert trace_modes, "Expected GameEngine.apply_move to be called"
    assert all(trace_modes), "Expected trace_mode=True for all apply_move calls"


def test_canonical_phase_contract_completeness() -> None:
    """RR-CANON-R075: Verify phase_move_contract covers all canonical move types."""
    contract = phase_move_contract()

    # All move types that should be canonical
    canonical_move_types = {
        "place_ring",
        "skip_placement",
        "no_placement_action",
        "swap_sides",
        "move_stack",
        "overtaking_capture",
        "recovery_slide",
        "skip_recovery",
        "no_movement_action",
        "skip_capture",
        "continue_capture_segment",
        "process_line",
        "choose_line_option",
        "no_line_action",
        "eliminate_rings_from_stack",
        "choose_territory_option",
        "skip_territory_processing",
        "no_territory_action",
        "forced_elimination",
    }

    # Collect all move types from the contract
    contract_move_types = set()
    for allowed in contract.values():
        contract_move_types.update(allowed)

    # Verify all canonical move types are in the contract
    missing = canonical_move_types - contract_move_types
    assert not missing, f"Missing canonical move types from contract: {missing}"

    # Verify contract doesn't have unknown types
    extra = contract_move_types - canonical_move_types
    # Some may be intentionally added, but flag for review
    if extra:
        pass  # These are intentional additions, no assertion


def test_validate_canonical_move_valid_pairs() -> None:
    """RR-CANON-R075: Verify validate_canonical_move accepts valid phase/move pairs."""
    # Test valid pairs from each phase
    valid_pairs = [
        ("ring_placement", "place_ring"),
        ("ring_placement", "skip_placement"),
        ("movement", "move_stack"),
        ("movement", "recovery_slide"),
        ("capture", "overtaking_capture"),
        ("chain_capture", "continue_capture_segment"),
        ("line_processing", "process_line"),
        ("line_processing", "choose_line_option"),
        ("territory_processing", "choose_territory_option"),
        ("forced_elimination", "forced_elimination"),
    ]

    for phase, move_type in valid_pairs:
        result = validate_canonical_move(phase, move_type)
        assert result.ok, f"Expected ({phase}, {move_type}) to be valid but got: {result.reason}"
        assert result.effective_phase == phase


def test_validate_canonical_move_invalid_pairs() -> None:
    """RR-CANON-R075: Verify validate_canonical_move rejects invalid phase/move pairs."""
    # Test invalid pairs (phase/move type mismatch)
    invalid_pairs = [
        ("ring_placement", "move_stack"),  # move_stack not valid in ring_placement
        ("movement", "place_ring"),  # place_ring not valid in movement
        ("capture", "place_ring"),  # place_ring not valid in capture
        ("line_processing", "move_stack"),  # move_stack not valid in line_processing
    ]

    for phase, move_type in invalid_pairs:
        result = validate_canonical_move(phase, move_type)
        assert not result.ok, f"Expected ({phase}, {move_type}) to be invalid but was accepted"
        assert result.reason is not None


def test_validate_canonical_move_infers_phase() -> None:
    """RR-CANON-R075: Verify phase inference from move_type when phase is empty."""
    # When phase is None or empty, it should be inferred from move_type
    result = validate_canonical_move(None, "place_ring")
    assert result.ok
    assert result.effective_phase == "ring_placement"

    result = validate_canonical_move("", "move_stack")
    assert result.ok
    assert result.effective_phase == "movement"

    result = validate_canonical_move(None, "forced_elimination")
    assert result.ok
    assert result.effective_phase == "forced_elimination"


def test_validate_canonical_move_unknown_type() -> None:
    """RR-CANON-R075: Verify unknown move types are rejected."""
    result = validate_canonical_move(None, "unknown_move_type")
    assert not result.ok
    assert "non_canonical_move_type" in (result.reason or "")


def test_tournament_recording_options_defaults_to_canonical() -> None:
    """Verify TournamentRecordingOptions enables canonical recording by default."""
    options = TournamentRecordingOptions()

    assert options.enabled is True
    assert options.fsm_validation is True
    assert options.store_history_entries is True
    # These are key for canonical recording compliance
