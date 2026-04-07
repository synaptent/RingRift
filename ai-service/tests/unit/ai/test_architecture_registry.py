"""Regression tests for architecture_registry encoder lookup."""

from __future__ import annotations

from app.ai.neural_net.architecture_registry import get_encoder_for_model
from app.ai.neural_net.hex_architectures import HexNeuralNet_v4
from app.ai.neural_net.square_architectures import RingRiftCNN_v4
from app.ai.neural_net.v5_heavy import HexNeuralNet_v5_Heavy
from app.models import BoardType
from app.training.encoding import HexStateEncoderV3, SquareStateEncoder


def test_get_encoder_for_square_model_uses_square_encoder() -> None:
    model = RingRiftCNN_v4(board_size=8, num_players=2)

    encoder = get_encoder_for_model(model)

    assert isinstance(encoder, SquareStateEncoder)
    assert encoder.board_type == BoardType.SQUARE8
    assert encoder.board_size == 8
    assert encoder.feature_version == 2


def test_get_encoder_for_square19_model_preserves_board_size() -> None:
    model = RingRiftCNN_v4(board_size=19, num_players=2)

    encoder = get_encoder_for_model(model)

    assert isinstance(encoder, SquareStateEncoder)
    assert encoder.board_type == BoardType.SQUARE19
    assert encoder.board_size == 19
    assert encoder.feature_version == 2


def test_get_encoder_for_hex_heavy_model_uses_hex_v3_encoder() -> None:
    model = HexNeuralNet_v5_Heavy(board_size=9, num_players=2)

    encoder = get_encoder_for_model(model)

    assert isinstance(encoder, HexStateEncoderV3)
    assert encoder.board_size == 9
    assert encoder.feature_version == 2


def test_get_encoder_for_hex_v4_model_uses_hex_v3_encoder() -> None:
    model = HexNeuralNet_v4(board_size=9, num_players=2)

    encoder = get_encoder_for_model(model)

    assert isinstance(encoder, HexStateEncoderV3)
    assert encoder.board_size == 9
    assert encoder.feature_version == 2
