"""Tier-aware evaluation and gating profiles for RingRift AI difficulty tiers.

This module defines small, data-driven configuration objects describing
how candidate AIs should be evaluated for promotion at each difficulty
tier. The intent is to keep all thresholds and opponent mixes in one
place so that tuning does not require code changes.

The initial configuration focuses on square8, 2-player games and
provides representative tiers (D2-D10). Additional tiers can
be added by extending the TIER_EVAL_CONFIGS mapping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from app.models import AIType, BoardType

TierRole = Literal["baseline", "previous_tier", "peer", "other"]


@dataclass(frozen=True)
class TierOpponentConfig:
    """Configuration for a single opponent used in tier evaluation."""

    id: str
    description: str
    difficulty: int
    ai_type: AIType | None = None
    role: TierRole = "baseline"
    weight: float = 1.0
    games: int | None = None


@dataclass(frozen=True)
class TierEvaluationConfig:
    """Evaluation profile for a single difficulty tier.

    Promotion Requirement (Dec 2025):
    D(n) must beat D(n-1) at 55%+ win rate in 2-player matches on BOTH
    square8 and hex8 boards. This is enforced via min_win_rate_vs_previous_tier.
    """

    tier_name: str
    display_name: str
    board_type: BoardType
    num_players: int
    num_games: int
    candidate_difficulty: int
    time_budget_ms: int | None
    opponents: list[TierOpponentConfig] = field(default_factory=list)
    min_win_rate_vs_baseline: float | None = None
    # Canonical threshold: D(n) must beat D(n-1) at 55%+ in 2p matches
    min_win_rate_vs_previous_tier: float = 0.55
    max_regression_vs_previous_tier: float | None = None
    # Confidence level used for Wilson lower-bound gating against baseline.
    # When set, min_win_rate_vs_baseline is applied to the lower bound of the
    # Wilson interval rather than the raw win rate.
    promotion_confidence: float = 0.95
    description: str = ""


TIER_EVAL_CONFIGS: dict[str, TierEvaluationConfig] = {}


def _build_default_configs() -> dict[str, TierEvaluationConfig]:
    """Return the built-in tier evaluation profiles.

    The defaults are intentionally modest and primarily intended as a
    starting point for calibration. They can be overridden or replaced
    in higher-level tooling by constructing new TierEvaluationConfig
    instances.
    """
    # Canonical square8 2-player tiers used for the primary production ladder.
    # Thresholds are tuned per tier based on observed performance and
    # calibration goals. They are not guaranteed to be monotonic.
    #
    # Baseline thresholds (vs random/heuristic) are set explicitly in the
    # per-tier configs below.
    configs: dict[str, TierEvaluationConfig] = {
        "D1": TierEvaluationConfig(
            tier_name="D1",
            display_name="D1 – random baseline (square8, 2p)",
            board_type=BoardType.SQUARE8,
            num_players=2,
            num_games=100,
            candidate_difficulty=1,
            time_budget_ms=None,
            opponents=[],
            min_win_rate_vs_baseline=None,
            max_regression_vs_previous_tier=None,
            description=(
                "Entry tier using pure random play. No gating required."
            ),
        ),
        "D2": TierEvaluationConfig(
            tier_name="D2",
            display_name="D2 – easy heuristic (square8, 2p)",
            board_type=BoardType.SQUARE8,
            num_players=2,
            num_games=200,
            candidate_difficulty=2,
            time_budget_ms=None,
            opponents=[
                TierOpponentConfig(
                    id="random_d1",
                    description="Random baseline (canonical difficulty 1)",
                    difficulty=1,
                    ai_type=AIType.RANDOM,
                    role="baseline",
                ),
                TierOpponentConfig(
                    id="tier_d1",
                    description="Previous tier reference (difficulty 1)",
                    difficulty=1,
                    ai_type=AIType.RANDOM,
                    role="previous_tier",
                ),
            ],
            min_win_rate_vs_baseline=0.60,
            max_regression_vs_previous_tier=0.05,
            description=(
                "Sanity-check that a difficulty-2 candidate clearly "
                "outperforms pure random play on square8, 2-player."
            ),
        ),
        "D3": TierEvaluationConfig(
            tier_name="D3",
            display_name="D3 – low heuristic (square8, 2p)",
            board_type=BoardType.SQUARE8,
            num_players=2,
            num_games=200,
            candidate_difficulty=3,
            time_budget_ms=None,
            opponents=[
                TierOpponentConfig(
                    id="random_d1",
                    description="Random baseline (canonical difficulty 1)",
                    difficulty=1,
                    ai_type=AIType.RANDOM,
                    role="baseline",
                ),
                TierOpponentConfig(
                    id="tier_d2",
                    description="Previous tier reference (difficulty 2)",
                    difficulty=2,
                    ai_type=None,
                    role="previous_tier",
                ),
            ],
            min_win_rate_vs_baseline=0.55,
            max_regression_vs_previous_tier=0.05,
            description=(
                "Low heuristic tier, must beat random and show improvement "
                "over difficulty-2."
            ),
        ),
        "D4": TierEvaluationConfig(
            tier_name="D4",
            display_name="D4 – mid heuristic (square8, 2p)",
            board_type=BoardType.SQUARE8,
            num_players=2,
            num_games=300,
            candidate_difficulty=4,
            time_budget_ms=None,
            opponents=[
                TierOpponentConfig(
                    id="random_d1",
                    description="Random baseline (canonical difficulty 1)",
                    difficulty=1,
                    ai_type=AIType.RANDOM,
                    role="baseline",
                ),
                TierOpponentConfig(
                    id="tier_d3",
                    description="Previous tier reference (difficulty 3)",
                    difficulty=3,
                    ai_type=None,
                    role="previous_tier",
                ),
            ],
            min_win_rate_vs_baseline=0.68,
            max_regression_vs_previous_tier=0.05,
            description=(
                "Mid-tier heuristic. Candidate should solidly beat random "
                "and outperform the difficulty-3 tier."
            ),
        ),
        "D5": TierEvaluationConfig(
            tier_name="D5",
            display_name="D5 – mid minimax (square8, 2p)",
            board_type=BoardType.SQUARE8,
            num_players=2,
            num_games=300,
            candidate_difficulty=5,
            time_budget_ms=None,
            opponents=[
                TierOpponentConfig(
                    id="random_d1",
                    description="Random baseline (canonical difficulty 1)",
                    difficulty=1,
                    ai_type=AIType.RANDOM,
                    role="baseline",
                ),
                TierOpponentConfig(
                    id="heuristic_d2",
                    description="Heuristic baseline at difficulty 2",
                    difficulty=2,
                    ai_type=AIType.HEURISTIC,
                    role="baseline",
                ),
                TierOpponentConfig(
                    id="tier_d4",
                    description="Previous tier reference (difficulty 4)",
                    difficulty=4,
                    ai_type=None,
                    role="previous_tier",
                ),
            ],
            min_win_rate_vs_baseline=0.60,
            max_regression_vs_previous_tier=0.05,
            description=(
                "Mid minimax tier. Must beat random and heuristic baselines, "
                "and outperform difficulty-4."
            ),
        ),
        "D6": TierEvaluationConfig(
            tier_name="D6",
            display_name="D6 – high heuristic (square8, 2p)",
            board_type=BoardType.SQUARE8,
            num_players=2,
            num_games=400,
            candidate_difficulty=6,
            time_budget_ms=None,
            opponents=[
                TierOpponentConfig(
                    id="random_d1",
                    description="Random baseline (canonical difficulty 1)",
                    difficulty=1,
                    ai_type=AIType.RANDOM,
                    role="baseline",
                ),
                TierOpponentConfig(
                    id="heuristic_d3",
                    description="Heuristic baseline at difficulty 3",
                    difficulty=3,
                    ai_type=AIType.HEURISTIC,
                    role="baseline",
                ),
                TierOpponentConfig(
                    id="tier_d5",
                    description="Previous tier reference (difficulty 5)",
                    difficulty=5,
                    ai_type=None,
                    role="previous_tier",
                ),
            ],
            min_win_rate_vs_baseline=0.72,
            max_regression_vs_previous_tier=0.05,
            description=(
                "High-difficulty heuristic tier. Must dominate random and "
                "mid-level heuristic baselines."
            ),
        ),
        "D7": TierEvaluationConfig(
            tier_name="D7",
            display_name="D7 – low MCTS (square8, 2p)",
            board_type=BoardType.SQUARE8,
            num_players=2,
            num_games=400,
            candidate_difficulty=7,
            time_budget_ms=None,
            opponents=[
                TierOpponentConfig(
                    id="random_d1",
                    description="Random baseline (canonical difficulty 1)",
                    difficulty=1,
                    ai_type=AIType.RANDOM,
                    role="baseline",
                ),
                TierOpponentConfig(
                    id="heuristic_d4",
                    description="Heuristic baseline at difficulty 4",
                    difficulty=4,
                    ai_type=AIType.HEURISTIC,
                    role="baseline",
                ),
                TierOpponentConfig(
                    id="tier_d6",
                    description="Previous tier reference (difficulty 6)",
                    difficulty=6,
                    ai_type=None,
                    role="previous_tier",
                ),
            ],
            min_win_rate_vs_baseline=0.65,
            max_regression_vs_previous_tier=0.05,
            description=(
                "Entry MCTS tier (heuristic-only). Must beat random and "
                "strong heuristic baselines."
            ),
        ),
        "D8": TierEvaluationConfig(
            tier_name="D8",
            display_name="D8 – strong MCTS (square8, 2p)",
            board_type=BoardType.SQUARE8,
            num_players=2,
            num_games=400,
            candidate_difficulty=8,
            time_budget_ms=None,
            opponents=[
                TierOpponentConfig(
                    id="random_d1",
                    description="Random baseline (canonical difficulty 1)",
                    difficulty=1,
                    ai_type=AIType.RANDOM,
                    role="baseline",
                ),
                TierOpponentConfig(
                    id="heuristic_d5",
                    description="Heuristic baseline at difficulty 5",
                    difficulty=5,
                    ai_type=AIType.HEURISTIC,
                    role="baseline",
                ),
                TierOpponentConfig(
                    id="tier_d7",
                    description="Previous tier reference (difficulty 7)",
                    difficulty=7,
                    ai_type=None,
                    role="previous_tier",
                ),
            ],
            min_win_rate_vs_baseline=0.75,
            max_regression_vs_previous_tier=0.05,
            description=(
                "Strong MCTS tier. Must beat random and high heuristic "
                "baselines. Threshold capped at 75% for neural model compatibility."
            ),
        ),
        "D9": TierEvaluationConfig(
            tier_name="D9",
            display_name="D9 – master Gumbel MCTS (square8, 2p)",
            board_type=BoardType.SQUARE8,
            num_players=2,
            num_games=400,
            candidate_difficulty=9,
            time_budget_ms=None,
            opponents=[
                TierOpponentConfig(
                    id="random_d1",
                    description="Random baseline (canonical difficulty 1)",
                    difficulty=1,
                    ai_type=AIType.RANDOM,
                    role="baseline",
                ),
                TierOpponentConfig(
                    id="heuristic_d6",
                    description="Heuristic baseline at difficulty 6",
                    difficulty=6,
                    ai_type=AIType.HEURISTIC,
                    role="baseline",
                ),
                TierOpponentConfig(
                    id="tier_d8",
                    description="Previous tier reference (difficulty 8)",
                    difficulty=8,
                    ai_type=None,
                    role="previous_tier",
                ),
            ],
            min_win_rate_vs_baseline=0.75,
            max_regression_vs_previous_tier=0.05,
            description=(
                "Master tier using Gumbel MCTS. Threshold calibrated at 75% "
                "based on observed neural model performance vs random (~70-76%)."
            ),
        ),
        "D10": TierEvaluationConfig(
            tier_name="D10",
            display_name="D10 – grandmaster Gumbel MCTS (square8, 2p)",
            board_type=BoardType.SQUARE8,
            num_players=2,
            num_games=400,
            candidate_difficulty=10,
            time_budget_ms=None,
            opponents=[
                TierOpponentConfig(
                    id="random_d1",
                    description="Random baseline (canonical difficulty 1)",
                    difficulty=1,
                    ai_type=AIType.RANDOM,
                    role="baseline",
                ),
                TierOpponentConfig(
                    id="heuristic_d7",
                    description="Heuristic baseline at difficulty 7",
                    difficulty=7,
                    ai_type=AIType.HEURISTIC,
                    role="baseline",
                ),
                TierOpponentConfig(
                    id="tier_d9",
                    description="Previous tier reference (difficulty 9)",
                    difficulty=9,
                    ai_type=None,
                    role="previous_tier",
                ),
            ],
            min_win_rate_vs_baseline=0.75,
            max_regression_vs_previous_tier=0.05,
            description=(
                "Grandmaster tier using maximum strength Gumbel MCTS. "
                "Threshold calibrated at 75% for neural model compatibility."
            ),
        ),
        "D11": TierEvaluationConfig(
            tier_name="D11",
            display_name="D11 – elite Gumbel MCTS (square8, 2p)",
            board_type=BoardType.SQUARE8,
            num_players=2,
            num_games=400,
            candidate_difficulty=11,
            time_budget_ms=None,
            opponents=[
                TierOpponentConfig(
                    id="random_d1",
                    description="Random baseline (canonical difficulty 1)",
                    difficulty=1,
                    ai_type=AIType.RANDOM,
                    role="baseline",
                ),
                TierOpponentConfig(
                    id="heuristic_d8",
                    description="Heuristic baseline at difficulty 8",
                    difficulty=8,
                    ai_type=AIType.HEURISTIC,
                    role="baseline",
                ),
                TierOpponentConfig(
                    id="tier_d10",
                    description="Previous tier reference (difficulty 10)",
                    difficulty=10,
                    ai_type=None,
                    role="previous_tier",
                ),
            ],
            min_win_rate_vs_baseline=0.75,
            max_regression_vs_previous_tier=0.05,
            description=(
                "Elite tier, highest difficulty. Uses maximum strength neural "
                "MCTS with extended search. Threshold at 75% for consistency."
            ),
        ),
    }

    # Experimental multi-board and multi-player tiers used for smoke tests and
    # early calibration on larger boards / additional player counts. These
    # profiles intentionally reuse conservative thresholds and the canonical
    # difficulty ladder so that wiring can be validated without committing to
    # final balance numbers.
    configs.update(
        {
            # Square19, 2-player tiers (easy heuristic / mid minimax).
            "D2_SQ19_2P": TierEvaluationConfig(
                tier_name="D2_SQ19_2P",
                display_name="D2 – easy heuristic (square19, 2p)",
                board_type=BoardType.SQUARE19,
                num_players=2,
                num_games=200,
                candidate_difficulty=2,
                time_budget_ms=None,
                opponents=[
                    TierOpponentConfig(
                        id="random_d1_sq19",
                        description=(
                            "Random baseline (canonical difficulty 1) "
                            "on square19, 2-player"
                        ),
                        difficulty=1,
                        ai_type=AIType.RANDOM,
                        role="baseline",
                    ),
                ],
                min_win_rate_vs_baseline=0.6,
                max_regression_vs_previous_tier=None,
                description=(
                    "Sanity-check tier ensuring a difficulty-2 candidate "
                    "comfortably beats pure random play on the larger "
                    "square19 board in 2-player games."
                ),
            ),
            "D4_SQ19_2P": TierEvaluationConfig(
                tier_name="D4_SQ19_2P",
                display_name="D4 – mid minimax (square19, 2p)",
                board_type=BoardType.SQUARE19,
                num_players=2,
                num_games=400,
                candidate_difficulty=4,
                time_budget_ms=None,
                opponents=[
                    TierOpponentConfig(
                        id="random_d1_sq19",
                        description=(
                            "Random baseline (canonical difficulty 1) "
                            "on square19, 2-player"
                        ),
                        difficulty=1,
                        ai_type=AIType.RANDOM,
                        role="baseline",
                    ),
                    TierOpponentConfig(
                        id="tier_d2_sq19",
                        description=(
                            "Previous tier reference using canonical "
                            "difficulty 2 profile on square19"
                        ),
                        difficulty=2,
                        ai_type=None,
                        role="previous_tier",
                    ),
                ],
                min_win_rate_vs_baseline=0.7,
                max_regression_vs_previous_tier=0.05,
                description=(
                    "Mid-tier square19 2p minimax gate. Candidate should "
                    "solidly beat random and not regress substantially "
                    "relative to the square19 difficulty-2 tier."
                ),
            ),
            # =================================================================
            # Hex8 (61 cells) 2-player tiers - REQUIRED for ladder validation
            # D(n) must beat D(n-1) at 55%+ on BOTH square8 and hex8
            # =================================================================
            "D2_HEX8_2P": TierEvaluationConfig(
                tier_name="D2_HEX8_2P",
                display_name="D2 – easy heuristic (hex8, 2p)",
                board_type=BoardType.HEX8,
                num_players=2,
                num_games=100,
                candidate_difficulty=2,
                time_budget_ms=None,
                opponents=[
                    TierOpponentConfig(
                        id="tier_d1_hex8",
                        description="D1 random baseline (hex8, 2p)",
                        difficulty=1,
                        ai_type=AIType.RANDOM,
                        role="previous_tier",
                    ),
                ],
                min_win_rate_vs_baseline=None,
                description="D2 must beat D1 at 55%+ on hex8 2p.",
            ),
            "D3_HEX8_2P": TierEvaluationConfig(
                tier_name="D3_HEX8_2P",
                display_name="D3 – low heuristic (hex8, 2p)",
                board_type=BoardType.HEX8,
                num_players=2,
                num_games=100,
                candidate_difficulty=3,
                time_budget_ms=None,
                opponents=[
                    TierOpponentConfig(
                        id="tier_d2_hex8",
                        description="D2 previous tier (hex8, 2p)",
                        difficulty=2,
                        ai_type=None,
                        role="previous_tier",
                    ),
                ],
                min_win_rate_vs_baseline=None,
                description="D3 must beat D2 at 55%+ on hex8 2p.",
            ),
            "D4_HEX8_2P": TierEvaluationConfig(
                tier_name="D4_HEX8_2P",
                display_name="D4 – mid heuristic (hex8, 2p)",
                board_type=BoardType.HEX8,
                num_players=2,
                num_games=100,
                candidate_difficulty=4,
                time_budget_ms=None,
                opponents=[
                    TierOpponentConfig(
                        id="tier_d3_hex8",
                        description="D3 previous tier (hex8, 2p)",
                        difficulty=3,
                        ai_type=None,
                        role="previous_tier",
                    ),
                ],
                min_win_rate_vs_baseline=None,
                description="D4 must beat D3 at 55%+ on hex8 2p.",
            ),
            "D5_HEX8_2P": TierEvaluationConfig(
                tier_name="D5_HEX8_2P",
                display_name="D5 – mid minimax (hex8, 2p)",
                board_type=BoardType.HEX8,
                num_players=2,
                num_games=100,
                candidate_difficulty=5,
                time_budget_ms=None,
                opponents=[
                    TierOpponentConfig(
                        id="tier_d4_hex8",
                        description="D4 previous tier (hex8, 2p)",
                        difficulty=4,
                        ai_type=None,
                        role="previous_tier",
                    ),
                ],
                min_win_rate_vs_baseline=None,
                description="D5 must beat D4 at 55%+ on hex8 2p.",
            ),
            "D6_HEX8_2P": TierEvaluationConfig(
                tier_name="D6_HEX8_2P",
                display_name="D6 – high heuristic (hex8, 2p)",
                board_type=BoardType.HEX8,
                num_players=2,
                num_games=100,
                candidate_difficulty=6,
                time_budget_ms=None,
                opponents=[
                    TierOpponentConfig(
                        id="tier_d5_hex8",
                        description="D5 previous tier (hex8, 2p)",
                        difficulty=5,
                        ai_type=None,
                        role="previous_tier",
                    ),
                ],
                min_win_rate_vs_baseline=None,
                description="D6 must beat D5 at 55%+ on hex8 2p.",
            ),
            "D7_HEX8_2P": TierEvaluationConfig(
                tier_name="D7_HEX8_2P",
                display_name="D7 – low MCTS (hex8, 2p)",
                board_type=BoardType.HEX8,
                num_players=2,
                num_games=100,
                candidate_difficulty=7,
                time_budget_ms=None,
                opponents=[
                    TierOpponentConfig(
                        id="tier_d6_hex8",
                        description="D6 previous tier (hex8, 2p)",
                        difficulty=6,
                        ai_type=None,
                        role="previous_tier",
                    ),
                ],
                min_win_rate_vs_baseline=None,
                description="D7 must beat D6 at 55%+ on hex8 2p.",
            ),
            "D8_HEX8_2P": TierEvaluationConfig(
                tier_name="D8_HEX8_2P",
                display_name="D8 – strong MCTS (hex8, 2p)",
                board_type=BoardType.HEX8,
                num_players=2,
                num_games=100,
                candidate_difficulty=8,
                time_budget_ms=None,
                opponents=[
                    TierOpponentConfig(
                        id="tier_d7_hex8",
                        description="D7 previous tier (hex8, 2p)",
                        difficulty=7,
                        ai_type=None,
                        role="previous_tier",
                    ),
                ],
                min_win_rate_vs_baseline=None,
                description="D8 must beat D7 at 55%+ on hex8 2p.",
            ),
            "D9_HEX8_2P": TierEvaluationConfig(
                tier_name="D9_HEX8_2P",
                display_name="D9 – master Gumbel MCTS (hex8, 2p)",
                board_type=BoardType.HEX8,
                num_players=2,
                num_games=100,
                candidate_difficulty=9,
                time_budget_ms=None,
                opponents=[
                    TierOpponentConfig(
                        id="tier_d8_hex8",
                        description="D8 previous tier (hex8, 2p)",
                        difficulty=8,
                        ai_type=None,
                        role="previous_tier",
                    ),
                ],
                min_win_rate_vs_baseline=None,
                description="D9 must beat D8 at 55%+ on hex8 2p.",
            ),
            "D10_HEX8_2P": TierEvaluationConfig(
                tier_name="D10_HEX8_2P",
                display_name="D10 – grandmaster Gumbel MCTS (hex8, 2p)",
                board_type=BoardType.HEX8,
                num_players=2,
                num_games=100,
                candidate_difficulty=10,
                time_budget_ms=None,
                opponents=[
                    TierOpponentConfig(
                        id="tier_d9_hex8",
                        description="D9 previous tier (hex8, 2p)",
                        difficulty=9,
                        ai_type=None,
                        role="previous_tier",
                    ),
                ],
                min_win_rate_vs_baseline=None,
                description="D10 must beat D9 at 55%+ on hex8 2p.",
            ),
            # =================================================================
            # Legacy hexagonal (469 cells) tiers - kept for compatibility
            # =================================================================
            "D2_HEX_2P": TierEvaluationConfig(
                tier_name="D2_HEX_2P",
                display_name="D2 – easy heuristic (hexagonal, 2p)",
                board_type=BoardType.HEXAGONAL,
                num_players=2,
                num_games=200,
                candidate_difficulty=2,
                time_budget_ms=None,
                opponents=[
                    TierOpponentConfig(
                        id="random_d1_hex",
                        description=(
                            "Random baseline (canonical difficulty 1) "
                            "on hexagonal, 2-player"
                        ),
                        difficulty=1,
                        ai_type=AIType.RANDOM,
                        role="baseline",
                    ),
                ],
                min_win_rate_vs_baseline=0.6,
                max_regression_vs_previous_tier=None,
                description=(
                    "Sanity-check tier for hexagonal 2-player games, "
                    "verifying that a difficulty-2 candidate outperforms "
                    "pure random play on the hex board."
                ),
            ),
            "D4_HEX_2P": TierEvaluationConfig(
                tier_name="D4_HEX_2P",
                display_name="D4 – mid minimax (hexagonal, 2p)",
                board_type=BoardType.HEXAGONAL,
                num_players=2,
                num_games=400,
                candidate_difficulty=4,
                time_budget_ms=None,
                opponents=[
                    TierOpponentConfig(
                        id="random_d1_hex",
                        description=(
                            "Random baseline (canonical difficulty 1) "
                            "on hexagonal, 2-player"
                        ),
                        difficulty=1,
                        ai_type=AIType.RANDOM,
                        role="baseline",
                    ),
                    TierOpponentConfig(
                        id="tier_d2_hex",
                        description=(
                            "Previous tier reference using canonical "
                            "difficulty 2 profile on hexagonal"
                        ),
                        difficulty=2,
                        ai_type=None,
                        role="previous_tier",
                    ),
                ],
                min_win_rate_vs_baseline=0.7,
                max_regression_vs_previous_tier=0.05,
                description=(
                    "Mid-tier hexagonal 2p minimax gate. Candidate should "
                    "solidly beat random and avoid major regression relative "
                    "to the hexagonal difficulty-2 tier."
                ),
            ),
            # Square8, 3-player and 4-player heuristic tiers. These reuse the
            # canonical difficulty-2 ladder but run with 3p/4p board configs
            # to provide stable evaluation entrypoints for multiplayer modes.
            "D2_SQ8_3P": TierEvaluationConfig(
                tier_name="D2_SQ8_3P",
                display_name="D2 – heuristic (square8, 3p)",
                board_type=BoardType.SQUARE8,
                num_players=3,
                num_games=200,
                candidate_difficulty=2,
                time_budget_ms=None,
                opponents=[
                    TierOpponentConfig(
                        id="random_d1_sq8_3p",
                        description=(
                            "Random baseline (canonical difficulty 1) "
                            "on square8, 3-player"
                        ),
                        difficulty=1,
                        ai_type=AIType.RANDOM,
                        role="baseline",
                    ),
                    TierOpponentConfig(
                        id="heuristic_d2_sq8_3p",
                        description=(
                            "Baseline heuristic profile at canonical "
                            "difficulty 2 for 3-player games"
                        ),
                        difficulty=2,
                        ai_type=AIType.HEURISTIC,
                        role="baseline",
                    ),
                ],
                min_win_rate_vs_baseline=0.55,
                max_regression_vs_previous_tier=None,
                description=(
                    "Experimental square8 3-player tier ensuring that the "
                    "difficulty-2 multiplayer heuristic comfortably beats "
                    "random and performs competitively against a low "
                    "difficulty heuristic baseline."
                ),
            ),
            "D2_SQ8_4P": TierEvaluationConfig(
                tier_name="D2_SQ8_4P",
                display_name="D2 – heuristic (square8, 4p)",
                board_type=BoardType.SQUARE8,
                num_players=4,
                num_games=200,
                candidate_difficulty=2,
                time_budget_ms=None,
                opponents=[
                    TierOpponentConfig(
                        id="random_d1_sq8_4p",
                        description=(
                            "Random baseline (canonical difficulty 1) "
                            "on square8, 4-player"
                        ),
                        difficulty=1,
                        ai_type=AIType.RANDOM,
                        role="baseline",
                    ),
                    TierOpponentConfig(
                        id="heuristic_d2_sq8_4p",
                        description=(
                            "Baseline heuristic profile at canonical "
                            "difficulty 2 for 4-player games"
                        ),
                        difficulty=2,
                        ai_type=AIType.HEURISTIC,
                        role="baseline",
                    ),
                ],
                min_win_rate_vs_baseline=0.55,
                max_regression_vs_previous_tier=None,
                description=(
                    "Experimental square8 4-player tier mirroring the 3p "
                    "configuration, intended primarily as a stable smoke "
                    "test entrypoint for 4-player evaluation."
                ),
            ),
        }
    )

    return configs


TIER_EVAL_CONFIGS = _build_default_configs()


def get_tier_config(tier_name: str) -> TierEvaluationConfig:
    """Return the TierEvaluationConfig for the given tier name.

    The lookup is case-insensitive and expects identifiers such as "D2",
    "d4", "D6", etc.
    """
    key = tier_name.upper()
    try:
        return TIER_EVAL_CONFIGS[key]
    except KeyError as exc:
        available = ", ".join(sorted(TIER_EVAL_CONFIGS.keys()))
        raise KeyError(
            f"Unknown tier '{tier_name}'. Available tiers: {available}"
        ) from exc


@dataclass(frozen=True)
class HeuristicTierSpec:
    """Minimal heuristic-tier specification for eval-pool based evaluation.

    This is intentionally small and square8-focused; higher level tooling can
    extend it over time as needed. All identifiers are strings so that CMA-ES
    and other tuning jobs can feed in weight-profile ids or parameter-set
    hashes without changing the harness.
    """

    id: str
    name: str
    board_type: BoardType
    num_players: int
    eval_pool_id: str
    num_games: int
    candidate_profile_id: str
    baseline_profile_id: str
    description: str = ""


# Square8-focused heuristic tiers for eval-pool based evaluation. These are
# deliberately conservative and can be extended in future waves. The default
# tiers assume the canonical v1 balanced heuristic weights; CMA-ES or GA jobs
# can point candidate_profile_id at alternative entries in
# HEURISTIC_WEIGHT_PROFILES without needing code changes.
HEURISTIC_TIER_SPECS: list[HeuristicTierSpec] = [
    HeuristicTierSpec(
        id="sq8_heuristic_baseline_v1",
        name="Square8 – heuristic_v1 vs baseline_v1 (eval pool v1)",
        board_type=BoardType.SQUARE8,
        num_players=2,
        eval_pool_id="v1",
        num_games=64,
        candidate_profile_id="baseline_v1_balanced",
        baseline_profile_id="baseline_v1_balanced",
        description=(
            "Sanity-check tier evaluating the canonical v1 balanced "
            "heuristic weights on the Square8 v1 eval pool."
        ),
    ),
]
