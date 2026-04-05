"""Tests for cross-player checkpoint transfer helpers."""

from __future__ import annotations

import torch

from scripts.transfer_2p_to_4p import (
    infer_source_players,
    resize_value_head_bias,
    resize_value_head_weight,
)


def test_infer_source_players_uses_top_level_metadata_first():
    state_dict = {"value_fc2.weight": torch.randn(4, 8)}

    assert infer_source_players({"num_players": 3}, state_dict) == 3


def test_infer_source_players_falls_back_to_value_head_shape():
    state_dict = {"decoder.value_fc2.weight": torch.randn(4, 8)}

    assert infer_source_players({}, state_dict) == 4


def test_resize_value_head_weight_expands_preserving_existing_rows():
    weight = torch.arange(16, dtype=torch.float32).reshape(2, 8)

    resized = resize_value_head_weight(weight, source_players=2, target_players=4)

    assert resized.shape == (4, 8)
    assert torch.equal(resized[:2], weight)
    assert not torch.equal(resized[2], torch.zeros(8))
    assert not torch.equal(resized[3], torch.zeros(8))


def test_resize_value_head_weight_shrinks_by_truncating_rows():
    weight = torch.arange(32, dtype=torch.float32).reshape(4, 8)

    resized = resize_value_head_weight(weight, source_players=4, target_players=2)

    assert resized.shape == (2, 8)
    assert torch.equal(resized, weight[:2])


def test_resize_value_head_bias_handles_expand_and_shrink():
    bias = torch.tensor([1.0, 2.0, 3.0, 4.0])

    shrunk = resize_value_head_bias(bias, source_players=4, target_players=2)
    expanded = resize_value_head_bias(torch.tensor([1.0, 2.0]), source_players=2, target_players=4)

    assert torch.equal(shrunk, torch.tensor([1.0, 2.0]))
    assert expanded.shape == (4,)
    assert torch.equal(expanded[:2], torch.tensor([1.0, 2.0]))
