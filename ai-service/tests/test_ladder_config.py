import os
import sys

import pytest

# Ensure app package is importable when running tests directly.
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from app.config.ladder_config import (  # noqa: E402
    LadderTierConfig,
    get_ladder_tier_config,
)
from app.models import AIType, BoardType  # noqa: E402
from app.training.tier_eval_config import get_tier_config  # noqa: E402
from app.training import tier_eval_runner as runner  # noqa: E402
from app.ai.heuristic_ai import HeuristicAI  # noqa: E402


def test_square8_two_player_ladder_entries_exist() -> None:
    """Ladder configs exist for square8 2p D2/D4/D6/D8 with expected types."""
    expected = {
        2: AIType.HEURISTIC,
        4: AIType.MINIMAX,
        6: AIType.MCTS,
        8: AIType.MCTS,
    }

    for difficulty, expected_ai_type in expected.items():
        cfg = get_ladder_tier_config(difficulty, BoardType.SQUARE8, 2)
        assert isinstance(cfg, LadderTierConfig)
        assert cfg.difficulty == difficulty
        assert cfg.board_type == BoardType.SQUARE8
        assert cfg.num_players == 2
        assert cfg.ai_type == expected_ai_type
        assert cfg.think_time_ms >= 0
        assert 0.0 <= cfg.randomness <= 1.0


def test_ladder_configs_align_with_tier_eval_configs() -> None:
    """Ladder board/players match tier eval configs for D2/D4/D6/D8."""
    for tier_name in ("D2", "D4", "D6", "D8"):
        tier_cfg = get_tier_config(tier_name)
        ladder_cfg = get_ladder_tier_config(
            tier_cfg.candidate_difficulty,
            tier_cfg.board_type,
            tier_cfg.num_players,
        )
        assert ladder_cfg.board_type == tier_cfg.board_type
        assert ladder_cfg.num_players == tier_cfg.num_players
        assert ladder_cfg.difficulty == tier_cfg.candidate_difficulty


def test_tier_eval_runner_uses_ladder_tier_config(monkeypatch) -> None:
    """Tier eval uses LadderTierConfig for candidate behaviour."""
    tier_cfg = get_tier_config("D2")

    sentinel_randomness = 0.123
    sentinel_think_time = 987
    sentinel_profile = "heuristic_v1_2p_sentinel"

    ladder_cfg = LadderTierConfig(
        difficulty=tier_cfg.candidate_difficulty,
        board_type=tier_cfg.board_type,
        num_players=tier_cfg.num_players,
        ai_type=AIType.HEURISTIC,
        model_id=None,
        heuristic_profile_id=sentinel_profile,
        randomness=sentinel_randomness,
        think_time_ms=sentinel_think_time,
        notes="test override",
    )

    def _fake_get_ladder_tier_config(difficulty, board_type, num_players):
        assert difficulty == tier_cfg.candidate_difficulty
        assert board_type == tier_cfg.board_type
        assert num_players == tier_cfg.num_players
        return ladder_cfg

    monkeypatch.setattr(
        runner,
        "get_ladder_tier_config",
        _fake_get_ladder_tier_config,
    )

    ai = runner._create_ladder_ai_instance(  # type: ignore[attr-defined]
        tier_config=tier_cfg,
        difficulty=tier_cfg.candidate_difficulty,
        player_number=1,
        time_budget_ms=None,
        ai_type_override=None,
        rng_seed=42,
    )

    assert isinstance(ai, HeuristicAI)
    cfg = getattr(ai, "config", None)
    assert cfg is not None
    assert cfg.difficulty == tier_cfg.candidate_difficulty
    assert cfg.randomness == pytest.approx(sentinel_randomness)
    assert cfg.think_time == sentinel_think_time
    assert cfg.heuristic_profile_id == sentinel_profile
    assert cfg.rng_seed == 42


def test_multiboard_and_multiplayer_tier_eval_configs_exist() -> None:
    """Tier evaluation configs exist for multiboard and multiplayer tiers."""
    cases = [
        ("D2_SQ19_2P", BoardType.SQUARE19, 2, 2),
        ("D4_SQ19_2P", BoardType.SQUARE19, 2, 4),
        ("D2_HEX_2P", BoardType.HEXAGONAL, 2, 2),
        ("D4_HEX_2P", BoardType.HEXAGONAL, 2, 4),
        ("D2_SQ8_3P", BoardType.SQUARE8, 3, 2),
        ("D2_SQ8_4P", BoardType.SQUARE8, 4, 2),
    ]

    for tier_name, board_type, num_players, difficulty in cases:
        tier_cfg = get_tier_config(tier_name)
        assert tier_cfg.board_type == board_type
        assert tier_cfg.num_players == num_players
        assert tier_cfg.candidate_difficulty == difficulty
        assert tier_cfg.opponents, (
            f"{tier_name} should have at least one opponent"
        )
        # Thresholds are either unset or sensible probabilities.
        if tier_cfg.min_win_rate_vs_baseline is not None:
            assert 0.0 <= tier_cfg.min_win_rate_vs_baseline <= 1.0


def test_multiboard_and_multiplayer_ladder_entries_exist() -> None:
    """Ladder configs exist for multiboard and multiplayer tiers."""
    entries = [
        (2, BoardType.SQUARE19, 2),
        (4, BoardType.SQUARE19, 2),
        (2, BoardType.HEXAGONAL, 2),
        (4, BoardType.HEXAGONAL, 2),
        (2, BoardType.SQUARE8, 3),
        (2, BoardType.SQUARE8, 4),
    ]

    for difficulty, board_type, num_players in entries:
        cfg = get_ladder_tier_config(difficulty, board_type, num_players)
        assert isinstance(cfg, LadderTierConfig)
        assert cfg.difficulty == difficulty
        assert cfg.board_type == board_type
        assert cfg.num_players == num_players
        assert cfg.think_time_ms >= 0
        assert 0.0 <= cfg.randomness <= 1.0


def test_multiboard_ladder_aligns_with_tier_eval_configs() -> None:
    """Ensure ladder entries align with tier configs for multiboard tiers."""
    for tier_name in ("D2_SQ19_2P", "D2_HEX_2P"):
        tier_cfg = get_tier_config(tier_name)
        ladder_cfg = get_ladder_tier_config(
            tier_cfg.candidate_difficulty,
            tier_cfg.board_type,
            tier_cfg.num_players,
        )
        assert ladder_cfg.board_type == tier_cfg.board_type
        assert ladder_cfg.num_players == tier_cfg.num_players
        assert ladder_cfg.difficulty == tier_cfg.candidate_difficulty