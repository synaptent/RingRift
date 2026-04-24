"""Regression tests for v5-heavy heuristic FiLM initialization."""

from __future__ import annotations

import torch

from app.ai.neural_net.v5_heavy import HexNeuralNet_v5_Heavy, HeuristicEncoder


def test_heuristic_encoder_film_starts_as_identity() -> None:
    torch.manual_seed(0)
    encoder = HeuristicEncoder(num_heuristics=49, output_dim=128, num_filters=160)
    heuristics = torch.randn(8, 49) * 0.05

    _, gamma, beta = encoder(heuristics)

    assert torch.allclose(gamma, torch.ones_like(gamma))
    assert torch.allclose(beta, torch.zeros_like(beta))


def test_hex_v5_heavy_fresh_forward_is_not_saturated() -> None:
    torch.manual_seed(0)
    model = HexNeuralNet_v5_Heavy(
        board_size=9,
        hex_radius=4,
        in_channels=64,
        global_features=20,
        num_filters=32,
        num_se_blocks=2,
        num_attention_blocks=1,
        num_attention_heads=4,
        policy_size=128,
        num_players=2,
        num_heuristics=49,
        dropout=0.0,
    )
    model.eval()

    features = torch.randn(4, 64, 9, 9) * 0.05
    globals_ = torch.zeros(4, 20)
    heuristics = torch.randn(4, 49) * 0.05

    with torch.no_grad():
        values, policy_logits, rank_dist = model(features, globals_, heuristics)

    assert torch.isfinite(values).all()
    assert torch.isfinite(policy_logits).all()
    assert torch.isfinite(rank_dist).all()
    assert values.abs().max().item() < 0.5
    assert policy_logits.std().item() < 2.0
