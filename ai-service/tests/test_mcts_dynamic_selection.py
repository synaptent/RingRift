from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.ai.mcts_ai import MCTSAI, MCTSNode
from app.models import AIConfig, GamePhase

def test_dynamic_c_puct_scales_with_entropy() -> None:
    ai = MCTSAI(player_number=1, config=AIConfig(difficulty=5, use_neural_net=False))
    cpuct_uniform = ai._dynamic_c_puct(0, [0.25, 0.25, 0.25, 0.25])
    cpuct_peaky = ai._dynamic_c_puct(0, [1.0, 0.0, 0.0, 0.0])
    assert cpuct_uniform > cpuct_peaky


def test_rave_k_tapers_with_visits_and_difficulty() -> None:
    ai_low = MCTSAI(player_number=1, config=AIConfig(difficulty=5, use_neural_net=False))
    ai_high = MCTSAI(player_number=1, config=AIConfig(difficulty=8, use_neural_net=False))

    priors = [0.5, 0.5]
    k_low_0 = ai_low._rave_k_for_node(0, priors)
    k_high_0 = ai_high._rave_k_for_node(0, priors)
    assert k_high_0 < k_low_0

    k_low_200 = ai_low._rave_k_for_node(200, priors)
    assert k_low_200 < k_low_0


def test_fpu_reduction_phase_ordering() -> None:
    ai = MCTSAI(player_number=1, config=AIConfig(difficulty=5, use_neural_net=False))
    assert ai._fpu_reduction_for_phase(GamePhase.MOVEMENT) < ai._fpu_reduction_for_phase(
        GamePhase.TERRITORY_PROCESSING
    )


def test_node_fpu_affects_unvisited_q() -> None:
    mock_state = MagicMock()
    parent = MCTSNode(mock_state)
    parent.visits = 10
    parent.wins = 6.0  # parent_q = 0.6

    child_visited = MCTSNode(mock_state, parent=parent)
    child_visited.visits = 10
    child_visited.wins = 5.0  # q = 0.5

    child_unvisited = MCTSNode(mock_state, parent=parent)
    child_unvisited.visits = 0
    child_unvisited.wins = 0.0

    parent.children = [child_visited, child_unvisited]

    selected = parent.uct_select_child(
        c_puct=0.0,
        rave_k=0.0,
        fpu_reduction=0.2,
    )
    assert selected is child_visited


def test_node_values_convert_between_root_and_opponent_sides() -> None:
    """Child Q values are stored from the child-to-move perspective."""
    mock_state = MagicMock()
    parent = MCTSNode(mock_state)
    parent.to_move_is_root = True
    parent.visits = 10
    parent.wins = 0.0

    # Both children are opponent-to-move states; their wins/visits reflect
    # the opponent perspective and must be negated at the root.
    child_bad_for_root = MCTSNode(mock_state, parent=parent)
    child_bad_for_root.to_move_is_root = False
    child_bad_for_root.visits = 10
    child_bad_for_root.wins = 8.0  # opponent value = +0.8 => root value = -0.8

    child_less_bad_for_root = MCTSNode(mock_state, parent=parent)
    child_less_bad_for_root.to_move_is_root = False
    child_less_bad_for_root.visits = 10
    child_less_bad_for_root.wins = 2.0  # opponent value = +0.2 => root value = -0.2

    parent.children = [child_bad_for_root, child_less_bad_for_root]

    selected = parent.uct_select_child(c_puct=0.0, rave_k=0.0, fpu_reduction=0.0)
    assert selected is child_less_bad_for_root
