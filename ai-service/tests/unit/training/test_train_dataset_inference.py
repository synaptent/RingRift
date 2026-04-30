"""Regression tests for training dataset/model metadata inference."""

from __future__ import annotations

from types import SimpleNamespace


def test_hex_multiplayer_inference_keeps_requested_player_count():
    """Hex v4 3p training must build 3p heads, not MAX_PLAYERS heads."""
    from app.coordination.types import BoardType
    from app.training.train_dataset_inference import infer_dataset_metadata

    config = SimpleNamespace(
        board_type=BoardType.HEX8,
        history_length=3,
        feature_version=3,
        model_id="ringrift_hex8_3p",
    )

    result = infer_dataset_metadata(
        data_path="",
        config=config,
        num_players=3,
        model_version="v4",
        multi_player=True,
        use_streaming=True,
        BoardType=BoardType,
        get_policy_size_for_board=lambda _board_type: 4500,
        normalize_board_type=lambda board_type: board_type.name.lower(),
        validate_hex_policy_indices=lambda *_args, **_kwargs: (True, []),
        detect_tier_from_checkpoint=lambda *_args, **_kwargs: None,
    )

    assert result.use_hex_model is True
    assert result.use_hex_v4 is True
    assert result.hex_num_players == 3
    assert result.encoding_channels == 64
