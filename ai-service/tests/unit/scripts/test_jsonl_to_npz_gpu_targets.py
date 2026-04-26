from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np

from app.models import BoardType
from app.training.initial_state import create_initial_state
from scripts import jsonl_to_npz


def _state_with_scores(scores: list[int]) -> SimpleNamespace:
    return SimpleNamespace(
        players=[
            SimpleNamespace(player_number=idx + 1, eliminated_rings=score, territory_spaces=0)
            for idx, score in enumerate(scores)
        ]
    )


def test_gpu_selfplay_multiplayer_targets_use_final_rankings(monkeypatch):
    initial_state = _state_with_scores([0, 0, 0])
    intermediate_state = _state_with_scores([1, 0, 0])
    final_state = _state_with_scores([3, 2, 1])
    replay_states = [intermediate_state, final_state]

    def fake_apply_move(state, move, trace_mode=False):
        assert trace_mode is True
        return replay_states.pop(0)

    monkeypatch.setattr(jsonl_to_npz.GameEngine, "apply_move", fake_apply_move)

    targets = jsonl_to_npz._compute_gpu_selfplay_value_targets(
        initial_state=initial_state,
        moves=[object(), object()],
        winner=1,
        num_players=3,
    )

    assert targets.rank_aware is True
    np.testing.assert_allclose(targets.values_vec, np.array([1.0, 0.0, -1.0, 0.0], dtype=np.float32))
    assert jsonl_to_npz._gpu_selfplay_scalar_value(targets, winner=1, perspective=2, num_players=3) == 0.0


def test_gpu_selfplay_multiplayer_targets_fall_back_when_replay_fails(monkeypatch):
    def fake_apply_move(state, move, trace_mode=False):
        raise ValueError("synthetic replay failure")

    monkeypatch.setattr(jsonl_to_npz.GameEngine, "apply_move", fake_apply_move)

    targets = jsonl_to_npz._compute_gpu_selfplay_value_targets(
        initial_state=_state_with_scores([0, 0, 0]),
        moves=[object()],
        winner=1,
        num_players=3,
    )

    assert targets.rank_aware is False
    np.testing.assert_allclose(targets.values_vec, np.array([1.0, -1.0, -1.0, 0.0], dtype=np.float32))
    assert jsonl_to_npz._gpu_selfplay_scalar_value(targets, winner=1, perspective=2, num_players=3) == -1.0


def test_gpu_selfplay_multiplayer_targets_warn_when_replay_falls_back(monkeypatch, caplog):
    def fake_apply_move(state, move, trace_mode=False):
        raise ValueError("synthetic replay failure")

    monkeypatch.setattr(jsonl_to_npz.GameEngine, "apply_move", fake_apply_move)
    caplog.set_level(logging.WARNING, logger="jsonl_to_npz")

    jsonl_to_npz._compute_gpu_selfplay_value_targets(
        initial_state=_state_with_scores([0, 0, 0]),
        moves=[jsonl_to_npz.parse_move({"type": "place_ring", "player": 1, "to": {"x": 0, "y": 0}})],
        winner=1,
        num_players=3,
        record={"game_id": "game-1", "board_type": "hex8"},
    )

    assert "GPU selfplay rank-aware replay failed" in caplog.text
    assert any(getattr(record, "exception", "") == "synthetic replay failure" for record in caplog.records)


def test_gpu_selfplay_multiplayer_targets_use_record_final_state_before_replay(monkeypatch):
    final_state = _state_with_scores([3, 2, 1])

    def fail_if_replayed(*_args, **_kwargs):
        raise AssertionError("record final_state should avoid replay for target computation")

    monkeypatch.setattr(jsonl_to_npz, "_final_state_from_gpu_record", lambda record: final_state)
    monkeypatch.setattr(jsonl_to_npz.GameEngine, "apply_move", fail_if_replayed)

    targets = jsonl_to_npz._compute_gpu_selfplay_value_targets(
        initial_state=_state_with_scores([0, 0, 0]),
        moves=[object()],
        winner=1,
        num_players=3,
        record={"game_id": "game-with-final-state"},
    )

    assert targets.rank_aware is True
    np.testing.assert_allclose(targets.values_vec, np.array([1.0, 0.0, -1.0, 0.0], dtype=np.float32))
    assert jsonl_to_npz._gpu_selfplay_scalar_value(targets, winner=1, perspective=2, num_players=3) == 0.0


def test_gpu_selfplay_multiplayer_targets_use_record_final_state_without_winner(monkeypatch):
    final_state = _state_with_scores([3, 2, 1])
    final_state.winner = None

    monkeypatch.setattr(jsonl_to_npz, "_final_state_from_gpu_record", lambda record: final_state)

    targets = jsonl_to_npz._compute_gpu_selfplay_value_targets(
        initial_state=_state_with_scores([0, 0, 0]),
        moves=[object()],
        winner=None,
        num_players=3,
        record={"game_id": "budget-cutoff-game"},
    )

    assert targets.rank_aware is True
    np.testing.assert_allclose(targets.values_vec, np.array([1.0, 0.0, -1.0, 0.0], dtype=np.float32))
    assert jsonl_to_npz._gpu_selfplay_scalar_value(targets, winner=None, perspective=2, num_players=3) == 0.0


def test_gpu_selfplay_no_winner_without_final_state_is_neutral(caplog):
    caplog.set_level(logging.WARNING, logger="jsonl_to_npz")

    targets = jsonl_to_npz._compute_gpu_selfplay_value_targets(
        initial_state=_state_with_scores([0, 0, 0]),
        moves=[object()],
        winner=None,
        num_players=3,
        record={"game_id": "no-final-state"},
    )

    assert targets.rank_aware is False
    np.testing.assert_allclose(targets.values_vec, np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32))
    assert "no winner and no usable final_state" in caplog.text


def test_rank_values_treat_exact_multiplayer_ties_as_neutral():
    final_state = _state_with_scores([1, 1, 1])
    final_state.winner = None

    values = jsonl_to_npz._rank_values_by_player(final_state, num_players=3)

    assert values == {1: 0.0, 2: 0.0, 3: 0.0}


def test_record_final_state_deserializes_camelcase_scores_for_rank_values():
    final_state = create_initial_state(board_type=BoardType.HEX8, num_players=3)
    final_state.winner = None
    final_state.players = [
        player.model_copy(update={"eliminated_rings": score, "territory_spaces": territory})
        for player, score, territory in zip(final_state.players, [4, 2, 1], [5, 7, 9], strict=True)
    ]
    record = {
        "game_id": "camel-final-state",
        "final_state": final_state.model_dump(by_alias=True, exclude_none=True, mode="json"),
    }

    loaded_final_state = jsonl_to_npz._final_state_from_gpu_record(record)
    assert loaded_final_state is not None

    targets = jsonl_to_npz._compute_gpu_selfplay_value_targets(
        initial_state=create_initial_state(board_type=BoardType.HEX8, num_players=3),
        moves=[object()],
        winner=None,
        num_players=3,
        record=record,
    )

    assert targets.rank_aware is True
    np.testing.assert_allclose(targets.values_vec, np.array([1.0, 0.0, -1.0, 0.0], dtype=np.float32))


def test_parse_move_preserves_camelcase_replay_metadata():
    move = jsonl_to_npz.parse_move(
        {
            "type": "place_ring",
            "player": 2,
            "to": {"x": 1, "y": 2},
            "phase": "ring_placement",
            "placedOnStack": True,
            "placementCount": 3,
            "minimumDistance": 2,
            "moveNumber": 7,
            "thinkTime": 12,
        }
    )

    assert move.phase == "ring_placement"
    assert move.placed_on_stack is True
    assert move.placement_count == 3
    assert move.minimum_distance == 2
    assert move.move_number == 7
