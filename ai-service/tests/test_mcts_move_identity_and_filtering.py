from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

from app.ai.mcts_ai import MCTSAI, MCTSNode, MCTSNodeLite, _moves_match
from app.models import AIConfig, BoardType, Move, MoveType, Position


class _StubBoard:
    def __init__(self, board_type: BoardType):
        self.type = board_type


class _StubState:
    def __init__(self, board_type: BoardType, current_player: int = 1):
        self.board = _StubBoard(board_type)
        self.current_player = current_player


def _place_ring(
    *,
    move_id: str,
    player: int,
    to: Position,
    placement_count: int,
) -> Move:
    ts = datetime.now(timezone.utc)
    return Move(
        id=move_id,
        type=MoveType.PLACE_RING,
        player=player,
        to=to,
        placement_count=placement_count,
        placed_on_stack=False,
        timestamp=ts,
        think_time=0,
        move_number=1,
    )


def test_moves_match_distinguishes_multi_ring_placements() -> None:
    pos = Position(x=3, y=4)
    move1 = _place_ring(move_id="simulated", player=1, to=pos, placement_count=1)
    move2 = _place_ring(move_id="simulated", player=1, to=pos, placement_count=2)
    assert _moves_match(move1, move2) is False


def test_update_node_policy_filters_only_exact_child_move_legacy() -> None:
    ai = MCTSAI(player_number=1, config=AIConfig(difficulty=1, use_neural_net=False))
    ai.rules_engine = MagicMock()

    pos = Position(x=1, y=2)
    move1 = _place_ring(move_id="simulated", player=1, to=pos, placement_count=1)
    move2 = _place_ring(move_id="simulated", player=1, to=pos, placement_count=2)
    ai.rules_engine.get_valid_moves.return_value = [move1, move2]

    state = _StubState(BoardType.SQUARE19, current_player=1)
    root = MCTSNode(state)
    root.children = [MCTSNode(state, parent=root, move=move2)]

    ai._update_node_policy_legacy(root, state, policy=[], use_hex_nn=False)

    assert move1 in root.untried_moves
    assert move2 not in root.untried_moves


def test_update_node_policy_filters_only_exact_child_move_incremental() -> None:
    ai = MCTSAI(player_number=1, config=AIConfig(difficulty=1, use_neural_net=False))
    ai.rules_engine = MagicMock()

    pos = Position(x=5, y=6)
    move1 = _place_ring(move_id="simulated", player=1, to=pos, placement_count=1)
    move2 = _place_ring(move_id="simulated", player=1, to=pos, placement_count=2)
    ai.rules_engine.get_valid_moves.return_value = [move1, move2]

    state = _StubState(BoardType.SQUARE19, current_player=1)
    root = MCTSNodeLite()
    root.children = [MCTSNodeLite(parent=root, move=move2)]

    ai._update_node_policy_lite(root, state, policy=[], use_hex_nn=False)

    assert move1 in root.untried_moves
    assert move2 not in root.untried_moves


def test_select_best_move_filters_invalid_multi_ring_child_legacy() -> None:
    ai = MCTSAI(player_number=1, config=AIConfig(difficulty=1, use_neural_net=False))

    pos = Position(x=7, y=8)
    move_valid = _place_ring(move_id="simulated", player=1, to=pos, placement_count=1)
    move_stale = _place_ring(move_id="simulated", player=1, to=pos, placement_count=2)

    state = _StubState(BoardType.SQUARE19, current_player=1)
    root = MCTSNode(state)
    child = MCTSNode(state, parent=root, move=move_stale)
    child.visits = 10
    root.children = [child]

    selected, _policy = ai._select_best_move_legacy(root, [move_valid], MagicMock())
    assert selected == move_valid


def test_select_best_move_filters_invalid_multi_ring_child_incremental() -> None:
    ai = MCTSAI(player_number=1, config=AIConfig(difficulty=1, use_neural_net=False))

    pos = Position(x=9, y=10)
    move_valid = _place_ring(move_id="simulated", player=1, to=pos, placement_count=1)
    move_stale = _place_ring(move_id="simulated", player=1, to=pos, placement_count=2)

    root = MCTSNodeLite()
    child = MCTSNodeLite(parent=root, move=move_stale)
    child.visits = 10
    root.children = [child]

    selected, _policy = ai._select_best_move_incremental(root, [move_valid], MagicMock())
    assert selected == move_valid

