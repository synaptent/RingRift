"""Regression tests for runtime checkpoint contract inference."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import app.ai._neural_net_legacy as legacy_nn
import app.ai.neural_net.hex_architectures as hex_architectures
from app.models import BoardType
from app.ai._neural_net_legacy import _inspect_runtime_checkpoint_contract


pytestmark = pytest.mark.timeout(30)


def test_contract_infers_v4_multiplayer_value_head_shape(tmp_path):
    path = tmp_path / "candidate_3p_v4.pth"
    torch.save(
        {
            "model_state_dict": {
                "conv1.weight": torch.zeros(128, 64, 3, 3),
                "value_fc2.weight": torch.zeros(256, 256),
                "value_fc3.weight": torch.zeros(3, 256),
            },
            "_versioning_metadata": {"config": {"num_players": 2}},
        },
        path,
    )

    contract = _inspect_runtime_checkpoint_contract(path)

    assert contract["in_channels"] == 64
    assert contract["num_players"] == 3


def test_v4_runtime_init_uses_checkpoint_player_count(monkeypatch, tmp_path):
    path = tmp_path / "candidate_4p_v4.pth"
    torch.save(
        {
            "model_state_dict": {
                "conv1.weight": torch.zeros(128, 64, 3, 3),
                "value_fc3.weight": torch.zeros(4, 256),
            },
        },
        path,
    )
    captured: dict = {}

    class FakeV4Model:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def to(self, _device):
            return self

        def load_state_dict(self, _state_dict, strict):
            captured["strict"] = strict

        def eval(self):
            captured["eval"] = True

    monkeypatch.setattr(hex_architectures, "HexNeuralNet_v4", FakeV4Model)

    ai = legacy_nn.NeuralNetAI.__new__(legacy_nn.NeuralNetAI)
    ai.config = SimpleNamespace(nn_model_id=str(path), allow_fresh_weights=False)
    ai._base_dir = str(tmp_path)
    ai.device = "cpu"

    legacy_nn.NeuralNetAI._init_v4_model(ai, BoardType.HEX8)

    assert captured["num_players"] == 4
    assert captured["strict"] is True
    assert captured["eval"] is True


def test_contract_reads_feature_version_metadata(tmp_path):
    path = tmp_path / "candidate_v5_heavy_fv3.pth"
    torch.save(
        {
            "model_state_dict": {
                "conv1.weight": torch.zeros(160, 64, 3, 3),
                "value_fc3.weight": torch.zeros(2, 256),
            },
            "_versioning_metadata": {
                "config": {
                    "num_players": 2,
                    "feature_version": 3,
                },
            },
        },
        path,
    )

    contract = _inspect_runtime_checkpoint_contract(path)

    assert contract["in_channels"] == 64
    assert contract["num_players"] == 2
    assert contract["feature_version"] == 3


def test_contract_keeps_v2_value_fc2_player_inference(tmp_path):
    path = tmp_path / "candidate_v2_4p.pth"
    torch.save(
        {
            "model_state_dict": {
                "conv1.weight": torch.zeros(64, 40, 3, 3),
                "value_fc2.weight": torch.zeros(4, 128),
            },
        },
        path,
    )

    contract = _inspect_runtime_checkpoint_contract(path)

    assert contract["in_channels"] == 40
    assert contract["num_players"] == 4
