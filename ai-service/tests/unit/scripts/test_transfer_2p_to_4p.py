"""Regression tests for player-count transfer checkpoints."""

from __future__ import annotations

import torch

from app.ai.neural_net.hex_architectures import HexNeuralNet_v4
from app.utils.torch_utils import safe_load_checkpoint
from scripts.transfer_2p_to_4p import (
    infer_source_players,
    resize_rank_distribution_bias,
    resize_rank_distribution_weight,
    resize_value_head_bias,
    resize_value_head_weight,
    transfer_model_players,
)


def _small_hex_v4(num_players: int) -> HexNeuralNet_v4:
    return HexNeuralNet_v4(
        in_channels=64,
        global_features=20,
        num_res_blocks=1,
        num_filters=16,
        board_size=9,
        hex_radius=4,
        policy_size=None,
        num_players=num_players,
    )


def test_infer_source_players_uses_top_level_metadata_first():
    state_dict = {"value_fc2.weight": torch.randn(4, 8)}

    assert infer_source_players({"num_players": 3}, state_dict) == 3


def test_infer_source_players_falls_back_to_value_head_shape():
    state_dict = {"decoder.value_fc2.weight": torch.randn(4, 8)}

    assert infer_source_players({}, state_dict) == 4


def test_infer_source_players_checks_v4_value_fc3_before_intermediate_fc2():
    state_dict = {
        "value_fc2.weight": torch.randn(256, 256),
        "value_fc3.weight": torch.randn(3, 256),
    }

    assert infer_source_players({}, state_dict) == 3


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


def test_rank_distribution_resize_preserves_existing_block():
    old_weight = torch.arange(4 * 3, dtype=torch.float32).view(4, 3)
    old_bias = torch.arange(4, dtype=torch.float32)

    new_weight = resize_rank_distribution_weight(old_weight, 2, 3)
    new_bias = resize_rank_distribution_bias(old_bias, 2, 3)

    assert new_weight.shape == (9, 3)
    assert new_bias.shape == (9,)
    assert torch.equal(new_weight[0], old_weight[0])
    assert torch.equal(new_weight[1], old_weight[1])
    assert torch.equal(new_weight[3], old_weight[2])
    assert torch.equal(new_weight[4], old_weight[3])
    assert new_bias[0] == old_bias[0]
    assert new_bias[1] == old_bias[1]
    assert new_bias[3] == old_bias[2]
    assert new_bias[4] == old_bias[3]


def test_transfer_hex_v4_resizes_value_and_rank_heads_then_strict_loads(tmp_path):
    source = tmp_path / "hex8_2p_v4.pth"
    output = tmp_path / "hex8_4p_v4.pth"
    source_model = _small_hex_v4(num_players=2)
    source_sd = source_model.state_dict()

    torch.save(
        {
            "model_state_dict": source_sd,
            "_versioning_metadata": {
                "config": {
                    "board_type": "hex8",
                    "num_players": 2,
                    "num_res_blocks": 1,
                    "num_filters": 16,
                    "global_features": 20,
                },
            },
        },
        source,
    )

    transfer_model_players(
        str(source),
        str(output),
        "hex8",
        target_players=4,
        source_players=2,
    )

    checkpoint = safe_load_checkpoint(output, map_location="cpu")
    transferred_sd = checkpoint["model_state_dict"]

    assert transferred_sd["value_fc3.weight"].shape == (4, 256)
    assert transferred_sd["value_fc3.bias"].shape == (4,)
    assert transferred_sd["rank_dist_fc3.weight"].shape == (16, 256)
    assert transferred_sd["rank_dist_fc3.bias"].shape == (16,)
    assert checkpoint["_versioning_metadata"]["config"]["num_players"] == 4

    target_model = _small_hex_v4(num_players=4)
    target_model.load_state_dict(transferred_sd, strict=True)
