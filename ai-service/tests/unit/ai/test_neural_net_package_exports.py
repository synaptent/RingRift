"""Focused tests for app.ai.neural_net package exports."""

from __future__ import annotations

import importlib


def test_package_dir_lists_key_neural_net_exports() -> None:
    module = importlib.import_module("app.ai.neural_net")

    expected = {
        "ActionEncoderHex",
        "ActionEncoderSquare",
        "ActionEncoderSquare8",
        "ActionEncoderSquare19",
        "AttentionResidualBlock",
        "BOARD_POLICY_SIZES",
        "BOARD_SPATIAL_SIZES",
        "DecodedPolicyIndex",
        "HeuristicEncoder",
        "HexNeuralNet_v3_Flat",
        "HexNeuralNet_v5_Heavy",
        "NeuralNetAI",
        "NUM_HEURISTIC_FEATURES",
        "POLICY_SIZE_HEX8",
        "ResidualBlock",
        "RingRiftCNN_v3_Flat",
        "RingRiftCNN_v5_Heavy",
        "SEResidualBlock",
        "UnifiedNeuralNetFactory",
        "_decode_move_square8",
        "_encode_move_square8",
        "clear_model_cache",
        "create_hex_mask",
        "create_model_for_board",
        "create_v5_heavy_model",
        "decode_move_for_board",
        "encode_move_for_board",
        "get_action_encoder",
        "get_cached_model_count",
        "get_memory_tier",
        "get_model_config_for_board",
        "get_policy_size_for_board",
        "get_spatial_size_for_board",
        "multi_player_value_loss",
        "rank_distribution_loss",
        "ranks_from_game_result",
        "transform_policy_index_square",
    }

    assert expected.issubset(set(module.__all__))
    assert len(module.__all__) == len(set(module.__all__))

    for name in expected:
        assert hasattr(module, name)
        assert name in dir(module)
