from __future__ import annotations

from typing import Any

import pytest

from app.game_engine import GameEngine
from app.models import BoardType
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
