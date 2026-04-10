"""Tests for cross-board training/evaluation config coverage."""

from app.training.crossboard_strength import ALL_BOARD_CONFIGS, config_key


def test_all_board_configs_covers_all_12_canonical_configs():
    """Cross-board helpers should include every canonical board/player config."""
    expected = {
        ("square8", 2), ("square8", 3), ("square8", 4),
        ("square19", 2), ("square19", 3), ("square19", 4),
        ("hex8", 2), ("hex8", 3), ("hex8", 4),
        ("hexagonal", 2), ("hexagonal", 3), ("hexagonal", 4),
    }

    assert set(ALL_BOARD_CONFIGS) == expected
    assert {config_key(board, players) for board, players in ALL_BOARD_CONFIGS} == {
        "square8_2p", "square8_3p", "square8_4p",
        "square19_2p", "square19_3p", "square19_4p",
        "hex8_2p", "hex8_3p", "hex8_4p",
        "hexagonal_2p", "hexagonal_3p", "hexagonal_4p",
    }
