from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from scripts import jsonl_to_npz


def _state_with_scores(scores: list[int]) -> SimpleNamespace:
    return SimpleNamespace(
        players=[
            SimpleNamespace(player_number=idx + 1, eliminated_rings=score)
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
