from __future__ import annotations

from datetime import datetime, timezone

from app.ai.evaluation_provider import HeuristicEvaluator
from app.ai.fast_geometry import FastGeometry
from app.ai.heuristic_ai import HeuristicAI
from app.models import (
    AIConfig,
    BoardState,
    BoardType,
    GamePhase,
    GameState,
    GameStatus,
    Player,
    Position,
    RingStack,
    TimeControl,
)
from app.rules.core import (
    get_territory_victory_threshold,
    get_victory_threshold,
    infer_total_rings_in_play,
)


def _make_minimal_hex_state(*, stacks: dict[str, RingStack]) -> GameState:
    board_type = BoardType.HEXAGONAL
    num_players = 2
    now = datetime.now(timezone.utc)

    board = BoardState(
        type=board_type,
        size=13,
        stacks=stacks,
        markers={},
        collapsed_spaces={},
        eliminated_rings={},
        formed_lines=[],
        territories={},
    )

    players = [
        Player(
            id="p1",
            username="p1",
            type="ai",
            player_number=1,
            is_ready=True,
            time_remaining=0,
            ai_difficulty=2,
            rings_in_hand=0,
            eliminated_rings=0,
            territory_spaces=0,
        ),
        Player(
            id="p2",
            username="p2",
            type="ai",
            player_number=2,
            is_ready=True,
            time_remaining=0,
            ai_difficulty=2,
            rings_in_hand=0,
            eliminated_rings=0,
            territory_spaces=0,
        ),
    ]

    return GameState(
        id="game",
        board_type=board_type,
        rng_seed=1,
        board=board,
        players=players,
        current_phase=GamePhase.MOVEMENT,
        current_player=1,
        move_history=[],
        time_control=TimeControl(initial_time=0, increment=0, type="rapid"),
        spectators=[],
        game_status=GameStatus.ACTIVE,
        winner=None,
        created_at=now,
        last_move_at=now,
        is_rated=False,
        max_players=num_players,
        total_rings_in_play=infer_total_rings_in_play(players, board),
        total_rings_eliminated=0,
        victory_threshold=get_victory_threshold(board_type, num_players),
        territory_victory_threshold=get_territory_victory_threshold(board_type),
        lps_round_index=0,
        lps_current_round_actor_mask={},
        lps_exclusive_player_for_completed_round=None,
        lps_current_round_first_player=None,
        lps_consecutive_exclusive_rounds=0,
        lps_consecutive_exclusive_player=None,
        rules_options={"swapRuleEnabled": True},
    )


def test_fast_geometry_hex_uses_canonical_radius_12() -> None:
    geo = FastGeometry.get_instance()

    keys = geo.get_all_board_keys(BoardType.HEXAGONAL)
    assert len(keys) == 469
    assert "0,0,0" in keys

    assert geo.is_within_bounds_tuple(12, 0, -12, BoardType.HEXAGONAL)
    assert not geo.is_within_bounds_tuple(13, 0, -13, BoardType.HEXAGONAL)


def test_hex_visible_stacks_uses_cube_keys_for_heuristic_ai() -> None:
    stack = RingStack(
        position=Position(x=1, y=0, z=-1),
        rings=[1],
        stack_height=1,
        cap_height=1,
        controlling_player=1,
    )
    state = _make_minimal_hex_state(stacks={stack.position.to_key(): stack})

    ai = HeuristicAI(player_number=1, config=AIConfig(difficulty=2, randomness=0.0, rng_seed=1))
    ai._visible_stacks_cache = {}

    visible = ai._get_visible_stacks(Position(x=0, y=0, z=0), state)
    assert len(visible) == 1
    assert visible[0].position == stack.position


def test_hex_visible_stacks_uses_cube_keys_for_evaluator() -> None:
    stack = RingStack(
        position=Position(x=1, y=0, z=-1),
        rings=[2],
        stack_height=1,
        cap_height=1,
        controlling_player=2,
    )
    state = _make_minimal_hex_state(stacks={stack.position.to_key(): stack})

    evaluator = HeuristicEvaluator(player_number=1)
    visible = evaluator._get_visible_stacks(Position(x=0, y=0, z=0), state)
    assert len(visible) == 1
    assert visible[0].position == stack.position
