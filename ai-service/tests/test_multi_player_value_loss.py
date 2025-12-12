import pytest
import torch

from app.ai.neural_net import multi_player_value_loss


def test_multi_player_value_loss_masks_inactive_players_int() -> None:
    pred = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    target = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )

    # Only the first 2 player slots contribute.
    expected = ((pred[:, :2] - target[:, :2]) ** 2).sum() / (2 * 2)
    loss = multi_player_value_loss(pred, target, num_players=2)
    assert torch.isclose(loss, expected)


def test_multi_player_value_loss_masks_per_sample_tensor() -> None:
    pred = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
        ],
        dtype=torch.float32,
    )
    target = torch.zeros_like(pred)

    # Sample 0: 2p, Sample 1: 3p.
    num_players = torch.tensor([2, 3], dtype=torch.int64)

    sum0 = (pred[0, :2] ** 2).sum()
    sum1 = (pred[1, :3] ** 2).sum()
    expected = (sum0 + sum1) / (2 + 3)

    loss = multi_player_value_loss(pred, target, num_players=num_players)
    assert torch.isclose(loss, expected)


def test_multi_player_value_loss_rejects_shape_mismatch() -> None:
    pred = torch.zeros((2, 4), dtype=torch.float32)
    target = torch.zeros((2, 2), dtype=torch.float32)
    with pytest.raises(ValueError, match="share the same shape"):
        multi_player_value_loss(pred, target, num_players=2)

