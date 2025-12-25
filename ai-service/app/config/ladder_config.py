"""Canonical difficulty ladder configuration for RingRift.

This module defines a small, serialisable mapping from logical difficulty
levels (1–10) to concrete AI configurations for specific board types and
player counts. It is the single source of truth for the production
difficulty ladder used by:

* The FastAPI service (/ai/move) when constructing AIs for live games.
* Tier evaluation and gating scripts when calibrating candidate models.

The initial slice focuses on square8, 2-player D2/D4/D6/D8 tiers.
Additional boards or player counts can be added incrementally.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from app.models import AIType, BoardType

LadderKey = tuple[int, BoardType, int]


@dataclass(frozen=True)
class LadderTierConfig:
    """Canonical assignment for a single difficulty tier.

    The (difficulty, board_type, num_players) triple identifies the tier.
    The remaining fields capture how that tier is realised in production.
    All fields are intentionally JSON-serialisable so that CI and offline
    tooling can persist and diff ladder assignments.
    """

    difficulty: int
    board_type: BoardType
    num_players: int
    ai_type: AIType
    model_id: str | None
    heuristic_profile_id: str | None
    randomness: float
    think_time_ms: int
    use_neural_net: bool = False
    notes: str = ""


def _build_default_square8_two_player_configs() -> dict[
    LadderKey, LadderTierConfig
]:
    """Return the built-in ladder assignments for square8 2-player tiers.

    These defaults mirror the canonical difficulty profiles in ``app.main``
    while threading through an explicit model / profile id so that
    calibration and promotion tooling can reason about assignments.

    Difficulty mapping:
    - D1: Random
    - D2: Heuristic
    - D3: Minimax (non-neural, heuristic eval only)
    - D4: Minimax (neural/NNUE)
    - D5: MCTS (non-neural, heuristic rollouts)
    - D6: MCTS (neural value/policy)
    - D7-8: MCTS (neural, higher budgets)
    - D9-10: Descent (AlphaZero-style, always neural)

    Strength differences between tiers are expressed via ``ai_type``,
    ``use_neural_net``, ``randomness`` and ``think_time_ms``.
    """
    return {
        # D1 – random baseline on square8, 2-player.
        (1, BoardType.SQUARE8, 2): LadderTierConfig(
            difficulty=1,
            board_type=BoardType.SQUARE8,
            num_players=2,
            ai_type=AIType.RANDOM,
            model_id=None,
            heuristic_profile_id=None,
            randomness=0.5,
            think_time_ms=150,
            use_neural_net=False,
            notes=(
                "Random baseline tier used for entry-level difficulty and "
                "gating baselines."
            ),
        ),
        # D2 – easy heuristic baseline on square8, 2-player.
        (2, BoardType.SQUARE8, 2): LadderTierConfig(
            difficulty=2,
            board_type=BoardType.SQUARE8,
            num_players=2,
            ai_type=AIType.HEURISTIC,
            model_id="heuristic_v1_sq8_2p",
            heuristic_profile_id="heuristic_v1_sq8_2p",
            randomness=0.3,
            think_time_ms=200,
            use_neural_net=False,
            notes=(
                "Easy square8 2p tier backed by v1 2-player heuristic "
                "weights. Intended as the first non-random production "
                "difficulty."
            ),
        ),
        # D3 – lower-mid minimax on square8, 2-player (non-neural).
        (3, BoardType.SQUARE8, 2): LadderTierConfig(
            difficulty=3,
            board_type=BoardType.SQUARE8,
            num_players=2,
            ai_type=AIType.MINIMAX,
            model_id="v1-minimax-3",
            heuristic_profile_id="heuristic_v1_sq8_2p",
            randomness=0.15,
            think_time_ms=1800,
            use_neural_net=False,
            notes=(
                "Lower-mid square8 2p tier using minimax with heuristic "
                "evaluation only (no neural net)."
            ),
        ),
        # D4 – mid minimax on square8, 2-player (neural/NNUE).
        (4, BoardType.SQUARE8, 2): LadderTierConfig(
            difficulty=4,
            board_type=BoardType.SQUARE8,
            num_players=2,
            ai_type=AIType.MINIMAX,
            model_id="nnue_square8_2p",
            heuristic_profile_id="heuristic_v1_sq8_2p",
            randomness=0.08,
            think_time_ms=2800,
            use_neural_net=True,
            notes=(
                "Mid square8 2p tier using minimax with NNUE neural "
                "evaluation for stronger positional assessment."
            ),
        ),
        # D5 – upper-mid Descent on square8, 2-player (neural).
        (5, BoardType.SQUARE8, 2): LadderTierConfig(
            difficulty=5,
            board_type=BoardType.SQUARE8,
            num_players=2,
            ai_type=AIType.DESCENT,
            model_id="ringrift_best_sq8_2p",
            heuristic_profile_id="heuristic_v1_sq8_2p",
            randomness=0.05,
            think_time_ms=4000,
            use_neural_net=True,
            notes=(
                "Upper-mid square8 2p tier using Descent with neural guidance. "
                "Replaced plain MCTS as Descent + NN performs better."
            ),
        ),
        # D6 – high Descent on square8, 2-player (neural).
        (6, BoardType.SQUARE8, 2): LadderTierConfig(
            difficulty=6,
            board_type=BoardType.SQUARE8,
            num_players=2,
            ai_type=AIType.DESCENT,
            model_id="ringrift_best_sq8_2p",
            heuristic_profile_id="heuristic_v1_sq8_2p",
            randomness=0.02,
            think_time_ms=5500,
            use_neural_net=True,
            notes=(
                "High square8 2p tier using Descent with neural guidance "
                "and larger search budget."
            ),
        ),
        # D7 – heuristic-only MCTS on square8, 2-player.
        (7, BoardType.SQUARE8, 2): LadderTierConfig(
            difficulty=7,
            board_type=BoardType.SQUARE8,
            num_players=2,
            ai_type=AIType.MCTS,
            model_id=None,
            heuristic_profile_id="heuristic_v1_sq8_2p",
            randomness=0.0,
            think_time_ms=7500,
            use_neural_net=False,
            notes=(
                "Expert square8 2p tier using MCTS without neural guidance."
            ),
        ),
        # D8 – strong expert MCTS on square8, 2-player (neural).
        (8, BoardType.SQUARE8, 2): LadderTierConfig(
            difficulty=8,
            board_type=BoardType.SQUARE8,
            num_players=2,
            ai_type=AIType.MCTS,
            model_id="ringrift_best_sq8_2p",
            heuristic_profile_id="heuristic_v1_sq8_2p",
            randomness=0.0,
            think_time_ms=9600,
            use_neural_net=True,
            notes=(
                "Strong expert square8 2p tier using MCTS with neural "
                "guidance and larger search budget."
            ),
        ),
        # D9 – master Gumbel MCTS on square8, 2-player.
        (9, BoardType.SQUARE8, 2): LadderTierConfig(
            difficulty=9,
            board_type=BoardType.SQUARE8,
            num_players=2,
            ai_type=AIType.GUMBEL_MCTS,
            model_id="ringrift_best_sq8_2p",
            heuristic_profile_id="heuristic_v1_sq8_2p",
            randomness=0.0,
            think_time_ms=12600,
            use_neural_net=True,
            notes=(
                "Master square8 2p tier using Gumbel MCTS with extended "
                "search budget (strongest per benchmarks)."
            ),
        ),
        # D10 – grandmaster Gumbel MCTS on square8, 2-player.
        (10, BoardType.SQUARE8, 2): LadderTierConfig(
            difficulty=10,
            board_type=BoardType.SQUARE8,
            num_players=2,
            ai_type=AIType.GUMBEL_MCTS,
            model_id="ringrift_best_sq8_2p",
            heuristic_profile_id="heuristic_v1_sq8_2p",
            randomness=0.0,
            think_time_ms=16000,
            use_neural_net=True,
            notes=(
                "Grandmaster square8 2p tier using maximum strength "
                "Gumbel MCTS configuration."
            ),
        ),
        # D11 – ultimate Gumbel MCTS on square8, 2-player.
        (11, BoardType.SQUARE8, 2): LadderTierConfig(
            difficulty=11,
            board_type=BoardType.SQUARE8,
            num_players=2,
            ai_type=AIType.GUMBEL_MCTS,
            model_id="ringrift_best_sq8_2p",
            heuristic_profile_id="heuristic_v1_sq8_2p",
            randomness=0.0,
            think_time_ms=60000,  # 60 seconds per move
            use_neural_net=True,
            notes=(
                "Ultimate square8 2p tier using Gumbel MCTS with 60s think time. "
                "Intended to be nearly unbeatable by humans."
            ),
        ),
    }


def _build_default_square19_two_player_configs() -> dict[
    LadderKey, LadderTierConfig
]:
    """Return ladder assignments for square19 2-player tiers.

    These entries use the same difficulty mapping as square8 but with
    adjusted think times for the larger board. Experimental tiers
    primarily intended for evaluation/gating and multi-board smoke tests.
    """
    return {
        # D2 – easy heuristic baseline on square19, 2-player.
        (2, BoardType.SQUARE19, 2): LadderTierConfig(
            difficulty=2,
            board_type=BoardType.SQUARE19,
            num_players=2,
            ai_type=AIType.HEURISTIC,
            model_id="heuristic_v1_sq19_2p",
            heuristic_profile_id="heuristic_v1_sq19_2p",
            randomness=0.3,
            think_time_ms=300,
            use_neural_net=False,
            notes="Easy square19 2p tier using heuristic evaluation.",
        ),
        # D3 – lower-mid Descent on square19, 2-player (neural).
        # NOTE: Minimax is too slow on 19x19, Descent + NN performs better.
        (3, BoardType.SQUARE19, 2): LadderTierConfig(
            difficulty=3,
            board_type=BoardType.SQUARE19,
            num_players=2,
            ai_type=AIType.DESCENT,
            model_id="ringrift_best_sq19_2p",
            heuristic_profile_id="heuristic_v1_sq19_2p",
            randomness=0.15,
            think_time_ms=2500,
            use_neural_net=True,
            notes="Lower-mid square19 2p tier using Descent + NN (minimax too slow).",
        ),
        # D4 – mid Descent on square19, 2-player (neural).
        # NOTE: Minimax is too slow on 19x19, Descent + NN performs better.
        (4, BoardType.SQUARE19, 2): LadderTierConfig(
            difficulty=4,
            board_type=BoardType.SQUARE19,
            num_players=2,
            ai_type=AIType.DESCENT,
            model_id="ringrift_best_sq19_2p",
            heuristic_profile_id="heuristic_v1_sq19_2p",
            randomness=0.08,
            think_time_ms=4000,
            use_neural_net=True,
            notes="Mid square19 2p tier using Descent + NN (minimax too slow).",
        ),
        # D5 – upper-mid Descent on square19, 2-player (neural).
        (5, BoardType.SQUARE19, 2): LadderTierConfig(
            difficulty=5,
            board_type=BoardType.SQUARE19,
            num_players=2,
            ai_type=AIType.DESCENT,
            model_id="ringrift_best_sq19_2p",
            heuristic_profile_id="heuristic_v1_sq19_2p",
            randomness=0.05,
            think_time_ms=6000,
            use_neural_net=True,
            notes="Upper-mid square19 2p tier using Descent + NN.",
        ),
        # D6 – high Descent on square19, 2-player (neural).
        (6, BoardType.SQUARE19, 2): LadderTierConfig(
            difficulty=6,
            board_type=BoardType.SQUARE19,
            num_players=2,
            ai_type=AIType.DESCENT,
            model_id="ringrift_best_sq19_2p",
            heuristic_profile_id="heuristic_v1_sq19_2p",
            randomness=0.02,
            think_time_ms=8000,
            use_neural_net=True,
            notes="High square19 2p tier using Descent with neural guidance.",
        ),
        # D7 – expert MCTS on square19, 2-player (neural).
        (7, BoardType.SQUARE19, 2): LadderTierConfig(
            difficulty=7,
            board_type=BoardType.SQUARE19,
            num_players=2,
            ai_type=AIType.MCTS,
            model_id="ringrift_best_sq19_2p",
            heuristic_profile_id="heuristic_v1_sq19_2p",
            randomness=0.0,
            think_time_ms=11000,
            use_neural_net=True,
            notes="Expert square19 2p tier using MCTS with neural guidance.",
        ),
        # D8 – strong expert MCTS on square19, 2-player (neural).
        (8, BoardType.SQUARE19, 2): LadderTierConfig(
            difficulty=8,
            board_type=BoardType.SQUARE19,
            num_players=2,
            ai_type=AIType.MCTS,
            model_id="ringrift_best_sq19_2p",
            heuristic_profile_id="heuristic_v1_sq19_2p",
            randomness=0.0,
            think_time_ms=14000,
            use_neural_net=True,
            notes="Strong expert square19 2p tier using MCTS + NN.",
        ),
        # D9 – master Gumbel MCTS on square19, 2-player.
        (9, BoardType.SQUARE19, 2): LadderTierConfig(
            difficulty=9,
            board_type=BoardType.SQUARE19,
            num_players=2,
            ai_type=AIType.GUMBEL_MCTS,
            model_id="ringrift_best_sq19_2p",
            heuristic_profile_id="heuristic_v1_sq19_2p",
            randomness=0.0,
            think_time_ms=18000,
            use_neural_net=True,
            notes="Master square19 2p tier using Gumbel MCTS with neural.",
        ),
        # D10 – grandmaster Gumbel MCTS on square19, 2-player.
        (10, BoardType.SQUARE19, 2): LadderTierConfig(
            difficulty=10,
            board_type=BoardType.SQUARE19,
            num_players=2,
            ai_type=AIType.GUMBEL_MCTS,
            model_id="ringrift_best_sq19_2p",
            heuristic_profile_id="heuristic_v1_sq19_2p",
            randomness=0.0,
            think_time_ms=24000,
            use_neural_net=True,
            notes="Grandmaster square19 2p tier using Gumbel MCTS.",
        ),
        # D11 – ultimate Gumbel MCTS on square19, 2-player.
        (11, BoardType.SQUARE19, 2): LadderTierConfig(
            difficulty=11,
            board_type=BoardType.SQUARE19,
            num_players=2,
            ai_type=AIType.GUMBEL_MCTS,
            model_id="ringrift_best_sq19_2p",
            heuristic_profile_id="heuristic_v1_sq19_2p",
            randomness=0.0,
            think_time_ms=90000,  # 90 seconds for larger board
            use_neural_net=True,
            notes="Ultimate square19 2p tier using Gumbel MCTS with 90s think time.",
        ),
    }


def _build_default_hex_two_player_configs() -> dict[
    LadderKey, LadderTierConfig
]:
    """Return ladder assignments for hexagonal 2-player tiers.

    These entries use the same difficulty mapping as square8 but with
    adjusted think times for the larger hex board. Experimental until
    dedicated hex-tuned weights are available.
    """
    return {
        # D2 – easy heuristic on hexagonal, 2-player.
        (2, BoardType.HEXAGONAL, 2): LadderTierConfig(
            difficulty=2,
            board_type=BoardType.HEXAGONAL,
            num_players=2,
            ai_type=AIType.HEURISTIC,
            model_id="heuristic_v1_hex_2p",
            heuristic_profile_id="heuristic_v1_hex_2p",
            randomness=0.3,
            think_time_ms=300,
            use_neural_net=False,
            notes="Easy hex 2p tier using heuristic evaluation.",
        ),
        # D3 – lower-mid Descent on hexagonal, 2-player (neural).
        # NOTE: Minimax is too slow on 469-space hex, Descent + NN performs better.
        (3, BoardType.HEXAGONAL, 2): LadderTierConfig(
            difficulty=3,
            board_type=BoardType.HEXAGONAL,
            num_players=2,
            ai_type=AIType.DESCENT,
            model_id="ringrift_best_hex_2p",
            heuristic_profile_id="heuristic_v1_hex_2p",
            randomness=0.15,
            think_time_ms=2500,
            use_neural_net=True,
            notes="Lower-mid hex 2p tier using Descent + NN (minimax too slow).",
        ),
        # D4 – mid Descent on hexagonal, 2-player (neural).
        # NOTE: Minimax is too slow on 469-space hex, Descent + NN performs better.
        (4, BoardType.HEXAGONAL, 2): LadderTierConfig(
            difficulty=4,
            board_type=BoardType.HEXAGONAL,
            num_players=2,
            ai_type=AIType.DESCENT,
            model_id="ringrift_best_hex_2p",
            heuristic_profile_id="heuristic_v1_hex_2p",
            randomness=0.08,
            think_time_ms=4000,
            use_neural_net=True,
            notes="Mid hex 2p tier using Descent + NN (minimax too slow).",
        ),
        # D5 – upper-mid Descent on hexagonal, 2-player (neural).
        (5, BoardType.HEXAGONAL, 2): LadderTierConfig(
            difficulty=5,
            board_type=BoardType.HEXAGONAL,
            num_players=2,
            ai_type=AIType.DESCENT,
            model_id="ringrift_best_hex_2p",
            heuristic_profile_id="heuristic_v1_hex_2p",
            randomness=0.05,
            think_time_ms=6000,
            use_neural_net=True,
            notes="Upper-mid hex 2p tier using Descent + NN.",
        ),
        # D6 – high Descent on hexagonal, 2-player (neural).
        (6, BoardType.HEXAGONAL, 2): LadderTierConfig(
            difficulty=6,
            board_type=BoardType.HEXAGONAL,
            num_players=2,
            ai_type=AIType.DESCENT,
            model_id="ringrift_best_hex_2p",
            heuristic_profile_id="heuristic_v1_hex_2p",
            randomness=0.02,
            think_time_ms=8000,
            use_neural_net=True,
            notes="High hex 2p tier using Descent with neural guidance.",
        ),
        # D7 – expert MCTS on hexagonal, 2-player (neural).
        (7, BoardType.HEXAGONAL, 2): LadderTierConfig(
            difficulty=7,
            board_type=BoardType.HEXAGONAL,
            num_players=2,
            ai_type=AIType.MCTS,
            model_id="ringrift_best_hex_2p",
            heuristic_profile_id="heuristic_v1_hex_2p",
            randomness=0.0,
            think_time_ms=11000,
            use_neural_net=True,
            notes="Expert hex 2p tier using MCTS with neural guidance.",
        ),
        # D8 – strong expert MCTS on hexagonal, 2-player (neural).
        (8, BoardType.HEXAGONAL, 2): LadderTierConfig(
            difficulty=8,
            board_type=BoardType.HEXAGONAL,
            num_players=2,
            ai_type=AIType.MCTS,
            model_id="ringrift_best_hex_2p",
            heuristic_profile_id="heuristic_v1_hex_2p",
            randomness=0.0,
            think_time_ms=14000,
            use_neural_net=True,
            notes="Strong expert hex 2p tier using MCTS + NN.",
        ),
        # D9 – master Gumbel MCTS on hexagonal, 2-player.
        (9, BoardType.HEXAGONAL, 2): LadderTierConfig(
            difficulty=9,
            board_type=BoardType.HEXAGONAL,
            num_players=2,
            ai_type=AIType.GUMBEL_MCTS,
            model_id="ringrift_best_hex_2p",
            heuristic_profile_id="heuristic_v1_hex_2p",
            randomness=0.0,
            think_time_ms=18000,
            use_neural_net=True,
            notes="Master hex 2p tier using Gumbel MCTS with neural.",
        ),
        # D10 – grandmaster Gumbel MCTS on hexagonal, 2-player.
        (10, BoardType.HEXAGONAL, 2): LadderTierConfig(
            difficulty=10,
            board_type=BoardType.HEXAGONAL,
            num_players=2,
            ai_type=AIType.GUMBEL_MCTS,
            model_id="ringrift_best_hex_2p",
            heuristic_profile_id="heuristic_v1_hex_2p",
            randomness=0.0,
            think_time_ms=24000,
            use_neural_net=True,
            notes="Grandmaster hex 2p tier using Gumbel MCTS.",
        ),
        # D11 – ultimate Gumbel MCTS on hexagonal, 2-player.
        (11, BoardType.HEXAGONAL, 2): LadderTierConfig(
            difficulty=11,
            board_type=BoardType.HEXAGONAL,
            num_players=2,
            ai_type=AIType.GUMBEL_MCTS,
            model_id="ringrift_best_hex_2p",
            heuristic_profile_id="heuristic_v1_hex_2p",
            randomness=0.0,
            think_time_ms=90000,  # 90 seconds for larger board
            use_neural_net=True,
            notes="Ultimate hex 2p tier using Gumbel MCTS with 90s think time.",
        ),
    }


def _build_default_generic_board_configs(
    board_type: BoardType,
    num_players: int,
    heuristic_profile_id: str,
    think_time_by_difficulty: dict[int, int],
) -> dict[LadderKey, LadderTierConfig]:
    """Return ladder assignments for (board_type, num_players) using shared defaults.

    These are intentionally conservative defaults for boards/player counts that
    do not yet have fully calibrated heuristic profiles or promoted neural
    checkpoints. They exist primarily so the training loop can promote
    per-tier model_id / heuristic_profile_id values via runtime overrides.

    NOTE: For square19 and hexagonal boards, D3-6 use Descent + NN instead of
    minimax/MCTS because minimax is prohibitively slow on larger boards and
    Descent + NN outperforms plain MCTS per benchmarks.
    """

    if num_players < 2 or num_players > 4:  # pragma: no cover - defensive
        raise ValueError("num_players must be 2, 3, or 4")

    best_model_prefix = {
        BoardType.SQUARE8: "ringrift_best_sq8",
        BoardType.SQUARE19: "ringrift_best_sq19",
        BoardType.HEXAGONAL: "ringrift_best_hex",
    }[board_type]

    nnue_model_id = f"nnue_{board_type.value}_{num_players}p"
    best_model_id = f"{best_model_prefix}_{num_players}p"

    def _think(difficulty: int, fallback: int) -> int:
        return int(think_time_by_difficulty.get(difficulty, fallback))

    # Large boards (sq19, hex) use Descent + NN for D3-6 (minimax too slow, Descent beats MCTS)
    is_large_board = board_type in (BoardType.SQUARE19, BoardType.HEXAGONAL)

    # For D3-4: large boards use Descent + NN, small boards use Minimax
    d3_d4_ai_type = AIType.DESCENT if is_large_board else AIType.MINIMAX
    d3_d4_note = " (Descent + NN - minimax too slow)" if is_large_board else ""

    return {
        (2, board_type, num_players): LadderTierConfig(
            difficulty=2,
            board_type=board_type,
            num_players=num_players,
            ai_type=AIType.HEURISTIC,
            model_id=heuristic_profile_id,
            heuristic_profile_id=heuristic_profile_id,
            randomness=0.3,
            think_time_ms=_think(2, 300),
            use_neural_net=False,
            notes=f"Easy {board_type.value} {num_players}p tier using heuristic evaluation.",
        ),
        (3, board_type, num_players): LadderTierConfig(
            difficulty=3,
            board_type=board_type,
            num_players=num_players,
            ai_type=d3_d4_ai_type,
            model_id=best_model_id if is_large_board else None,
            heuristic_profile_id=heuristic_profile_id,
            randomness=0.15,
            think_time_ms=_think(3, 2500),
            use_neural_net=is_large_board,
            notes=f"Lower-mid {board_type.value} {num_players}p tier{d3_d4_note}.",
        ),
        (4, board_type, num_players): LadderTierConfig(
            difficulty=4,
            board_type=board_type,
            num_players=num_players,
            ai_type=d3_d4_ai_type,
            model_id=best_model_id if is_large_board else nnue_model_id,
            heuristic_profile_id=heuristic_profile_id,
            randomness=0.08,
            think_time_ms=_think(4, 4000),
            use_neural_net=True,  # All D4 use neural (NNUE for small, Descent for large)
            notes=f"Mid {board_type.value} {num_players}p tier{d3_d4_note}.",
        ),
        # D5-6: Descent + NN for all boards (replaces plain MCTS per benchmarks)
        (5, board_type, num_players): LadderTierConfig(
            difficulty=5,
            board_type=board_type,
            num_players=num_players,
            ai_type=AIType.DESCENT,
            model_id=best_model_id,
            heuristic_profile_id=heuristic_profile_id,
            randomness=0.05,
            think_time_ms=_think(5, 6000),
            use_neural_net=True,
            notes=f"Upper-mid {board_type.value} {num_players}p tier using Descent + NN.",
        ),
        (6, board_type, num_players): LadderTierConfig(
            difficulty=6,
            board_type=board_type,
            num_players=num_players,
            ai_type=AIType.DESCENT,
            model_id=best_model_id,
            heuristic_profile_id=heuristic_profile_id,
            randomness=0.02,
            think_time_ms=_think(6, 8000),
            use_neural_net=True,
            notes=f"High {board_type.value} {num_players}p tier using Descent with neural.",
        ),
        # D7-8: MCTS + NN
        (7, board_type, num_players): LadderTierConfig(
            difficulty=7,
            board_type=board_type,
            num_players=num_players,
            ai_type=AIType.MCTS,
            model_id=best_model_id,
            heuristic_profile_id=heuristic_profile_id,
            randomness=0.0,
            think_time_ms=_think(7, 11000),
            use_neural_net=True,
            notes=f"Expert {board_type.value} {num_players}p tier using MCTS with neural.",
        ),
        (8, board_type, num_players): LadderTierConfig(
            difficulty=8,
            board_type=board_type,
            num_players=num_players,
            ai_type=AIType.MCTS,
            model_id=best_model_id,
            heuristic_profile_id=heuristic_profile_id,
            randomness=0.0,
            think_time_ms=_think(8, 14000),
            use_neural_net=True,
            notes=f"Strong expert {board_type.value} {num_players}p tier using MCTS + NN.",
        ),
        (9, board_type, num_players): LadderTierConfig(
            difficulty=9,
            board_type=board_type,
            num_players=num_players,
            ai_type=AIType.GUMBEL_MCTS,
            model_id=best_model_id,
            heuristic_profile_id=heuristic_profile_id,
            randomness=0.0,
            think_time_ms=_think(9, 18000),
            use_neural_net=True,
            notes=f"Master {board_type.value} {num_players}p tier using Gumbel MCTS (strongest).",
        ),
        (10, board_type, num_players): LadderTierConfig(
            difficulty=10,
            board_type=board_type,
            num_players=num_players,
            ai_type=AIType.GUMBEL_MCTS,
            model_id=best_model_id,
            heuristic_profile_id=heuristic_profile_id,
            randomness=0.0,
            think_time_ms=_think(10, 24000),
            use_neural_net=True,
            notes=f"Grandmaster {board_type.value} {num_players}p tier using max Gumbel MCTS.",
        ),
        (11, board_type, num_players): LadderTierConfig(
            difficulty=11,
            board_type=board_type,
            num_players=num_players,
            ai_type=AIType.GUMBEL_MCTS,
            model_id=best_model_id,
            heuristic_profile_id=heuristic_profile_id,
            randomness=0.0,
            think_time_ms=_think(11, 90000),  # 90s for ultimate
            use_neural_net=True,
            notes=f"Ultimate {board_type.value} {num_players}p tier using Gumbel MCTS extended.",
        ),
    }


def _build_default_square19_three_player_configs() -> dict[
    LadderKey, LadderTierConfig
]:
    """Return ladder assignments for square19 3-player tiers (default heuristics)."""
    return _build_default_generic_board_configs(
        board_type=BoardType.SQUARE19,
        num_players=3,
        heuristic_profile_id="heuristic_v1_sq19_3p",
        think_time_by_difficulty={
            2: 350,
            3: 2700,
            4: 4300,
            5: 6500,
            6: 9000,
            7: 12500,
            8: 15500,
            9: 20000,
            10: 26000,
            11: 90000,
        },
    )


def _build_default_square19_four_player_configs() -> dict[
    LadderKey, LadderTierConfig
]:
    """Return ladder assignments for square19 4-player tiers (default heuristics)."""
    return _build_default_generic_board_configs(
        board_type=BoardType.SQUARE19,
        num_players=4,
        heuristic_profile_id="heuristic_v1_sq19_4p",
        think_time_by_difficulty={
            2: 400,
            3: 3000,
            4: 4800,
            5: 7000,
            6: 10000,
            7: 14000,
            8: 17000,
            9: 22000,
            10: 28000,
            11: 90000,
        },
    )


def _build_default_hex_three_player_configs() -> dict[
    LadderKey, LadderTierConfig
]:
    """Return ladder assignments for hexagonal 3-player tiers (default heuristics)."""
    return _build_default_generic_board_configs(
        board_type=BoardType.HEXAGONAL,
        num_players=3,
        heuristic_profile_id="heuristic_v1_hex_3p",
        think_time_by_difficulty={
            2: 350,
            3: 2700,
            4: 4300,
            5: 6500,
            6: 9000,
            7: 12500,
            8: 15500,
            9: 20000,
            10: 26000,
            11: 90000,
        },
    )


def _build_default_hex_four_player_configs() -> dict[
    LadderKey, LadderTierConfig
]:
    """Return ladder assignments for hexagonal 4-player tiers (default heuristics)."""
    return _build_default_generic_board_configs(
        board_type=BoardType.HEXAGONAL,
        num_players=4,
        heuristic_profile_id="heuristic_v1_hex_4p",
        think_time_by_difficulty={
            2: 400,
            3: 3000,
            4: 4800,
            5: 7000,
            6: 10000,
            7: 14000,
            8: 17000,
            9: 22000,
            10: 28000,
            11: 90000,
        },
    )


def _build_default_square8_three_player_configs() -> dict[
    LadderKey, LadderTierConfig
]:
    """Return ladder assignments for square8 3-player tiers.

    These tiers use the same difficulty mapping as 2-player but with
    3-player-optimised heuristic profile where available.
    """
    return {
        # D2 – easy heuristic on square8, 3-player.
        (2, BoardType.SQUARE8, 3): LadderTierConfig(
            difficulty=2,
            board_type=BoardType.SQUARE8,
            num_players=3,
            ai_type=AIType.HEURISTIC,
            model_id="heuristic_v1_sq8_3p",
            heuristic_profile_id="heuristic_v1_sq8_3p",
            randomness=0.3,
            think_time_ms=250,
            use_neural_net=False,
            notes="Easy square8 3p tier using heuristic evaluation.",
        ),
        # D3 – lower-mid minimax on square8, 3-player (non-neural).
        (3, BoardType.SQUARE8, 3): LadderTierConfig(
            difficulty=3,
            board_type=BoardType.SQUARE8,
            num_players=3,
            ai_type=AIType.MINIMAX,
            model_id="v1-minimax-3p-3",
            heuristic_profile_id="heuristic_v1_sq8_3p",
            randomness=0.15,
            think_time_ms=2000,
            use_neural_net=False,
            notes="Lower-mid square8 3p tier using minimax (non-neural).",
        ),
        # D4 – mid minimax on square8, 3-player (neural/NNUE).
        (4, BoardType.SQUARE8, 3): LadderTierConfig(
            difficulty=4,
            board_type=BoardType.SQUARE8,
            num_players=3,
            ai_type=AIType.MINIMAX,
            model_id="nnue_square8_3p",
            heuristic_profile_id="heuristic_v1_sq8_3p",
            randomness=0.08,
            think_time_ms=3200,
            use_neural_net=True,
            notes="Mid square8 3p tier using minimax with NNUE.",
        ),
        # D5 – upper-mid MCTS on square8, 3-player (non-neural).
        (5, BoardType.SQUARE8, 3): LadderTierConfig(
            difficulty=5,
            board_type=BoardType.SQUARE8,
            num_players=3,
            ai_type=AIType.MCTS,
            model_id="v1-mcts-3p-5",
            heuristic_profile_id="heuristic_v1_sq8_3p",
            randomness=0.05,
            think_time_ms=4500,
            use_neural_net=False,
            notes="Upper-mid square8 3p tier using MCTS (non-neural).",
        ),
        # D6 – high MCTS on square8, 3-player (neural).
        (6, BoardType.SQUARE8, 3): LadderTierConfig(
            difficulty=6,
            board_type=BoardType.SQUARE8,
            num_players=3,
            ai_type=AIType.MCTS,
            model_id="ringrift_best_sq8_3p",
            heuristic_profile_id="heuristic_v1_sq8_3p",
            randomness=0.02,
            think_time_ms=6000,
            use_neural_net=True,
            notes="High square8 3p tier using MCTS with neural guidance.",
        ),
        # D7 – expert MCTS on square8, 3-player (neural).
        (7, BoardType.SQUARE8, 3): LadderTierConfig(
            difficulty=7,
            board_type=BoardType.SQUARE8,
            num_players=3,
            ai_type=AIType.MCTS,
            model_id="ringrift_best_sq8_3p",
            heuristic_profile_id="heuristic_v1_sq8_3p",
            randomness=0.0,
            think_time_ms=8500,
            use_neural_net=True,
            notes="Expert square8 3p tier using MCTS with neural guidance.",
        ),
        # D8 – strong expert MCTS on square8, 3-player (neural).
        (8, BoardType.SQUARE8, 3): LadderTierConfig(
            difficulty=8,
            board_type=BoardType.SQUARE8,
            num_players=3,
            ai_type=AIType.MCTS,
            model_id="ringrift_best_sq8_3p",
            heuristic_profile_id="heuristic_v1_sq8_3p",
            randomness=0.0,
            think_time_ms=11000,
            use_neural_net=True,
            notes="Strong expert square8 3p tier using MCTS with neural.",
        ),
        # D9 – master Descent on square8, 3-player.
        (9, BoardType.SQUARE8, 3): LadderTierConfig(
            difficulty=9,
            board_type=BoardType.SQUARE8,
            num_players=3,
            ai_type=AIType.DESCENT,
            model_id="ringrift_best_sq8_3p",
            heuristic_profile_id="heuristic_v1_sq8_3p",
            randomness=0.0,
            think_time_ms=14000,
            use_neural_net=True,
            notes="Master square8 3p tier using Descent with neural.",
        ),
        # D10 – grandmaster Descent on square8, 3-player.
        (10, BoardType.SQUARE8, 3): LadderTierConfig(
            difficulty=10,
            board_type=BoardType.SQUARE8,
            num_players=3,
            ai_type=AIType.DESCENT,
            model_id="ringrift_best_sq8_3p",
            heuristic_profile_id="heuristic_v1_sq8_3p",
            randomness=0.0,
            think_time_ms=18000,
            use_neural_net=True,
            notes="Grandmaster square8 3p tier using strongest Descent.",
        ),
        # D11 – ultimate Descent on square8, 3-player.
        (11, BoardType.SQUARE8, 3): LadderTierConfig(
            difficulty=11,
            board_type=BoardType.SQUARE8,
            num_players=3,
            ai_type=AIType.DESCENT,
            model_id="ringrift_best_sq8_3p",
            heuristic_profile_id="heuristic_v1_sq8_3p",
            randomness=0.0,
            think_time_ms=75000,  # 75s for 3-player
            use_neural_net=True,
            notes="Ultimate square8 3p tier using Descent with 75s think time.",
        ),
    }


def _build_default_square8_four_player_configs() -> dict[
    LadderKey, LadderTierConfig
]:
    """Return ladder assignments for square8 4-player tiers.

    These tiers use the same difficulty mapping as 2-player but with
    4-player-optimised heuristic profile where available.
    """
    return {
        # D2 – easy heuristic on square8, 4-player.
        (2, BoardType.SQUARE8, 4): LadderTierConfig(
            difficulty=2,
            board_type=BoardType.SQUARE8,
            num_players=4,
            ai_type=AIType.HEURISTIC,
            model_id="heuristic_v1_sq8_4p",
            heuristic_profile_id="heuristic_v1_sq8_4p",
            randomness=0.3,
            think_time_ms=300,
            use_neural_net=False,
            notes="Easy square8 4p tier using heuristic evaluation.",
        ),
        # D3 – lower-mid minimax on square8, 4-player (non-neural).
        (3, BoardType.SQUARE8, 4): LadderTierConfig(
            difficulty=3,
            board_type=BoardType.SQUARE8,
            num_players=4,
            ai_type=AIType.MINIMAX,
            model_id="v1-minimax-4p-3",
            heuristic_profile_id="heuristic_v1_sq8_4p",
            randomness=0.15,
            think_time_ms=2200,
            use_neural_net=False,
            notes="Lower-mid square8 4p tier using minimax (non-neural).",
        ),
        # D4 – mid minimax on square8, 4-player (neural/NNUE).
        (4, BoardType.SQUARE8, 4): LadderTierConfig(
            difficulty=4,
            board_type=BoardType.SQUARE8,
            num_players=4,
            ai_type=AIType.MINIMAX,
            model_id="nnue_square8_4p",
            heuristic_profile_id="heuristic_v1_sq8_4p",
            randomness=0.08,
            think_time_ms=3500,
            use_neural_net=True,
            notes="Mid square8 4p tier using minimax with NNUE.",
        ),
        # D5 – upper-mid MCTS on square8, 4-player (non-neural).
        (5, BoardType.SQUARE8, 4): LadderTierConfig(
            difficulty=5,
            board_type=BoardType.SQUARE8,
            num_players=4,
            ai_type=AIType.MCTS,
            model_id="v1-mcts-4p-5",
            heuristic_profile_id="heuristic_v1_sq8_4p",
            randomness=0.05,
            think_time_ms=5000,
            use_neural_net=False,
            notes="Upper-mid square8 4p tier using MCTS (non-neural).",
        ),
        # D6 – high MCTS on square8, 4-player (neural).
        (6, BoardType.SQUARE8, 4): LadderTierConfig(
            difficulty=6,
            board_type=BoardType.SQUARE8,
            num_players=4,
            ai_type=AIType.MCTS,
            model_id="ringrift_best_sq8_4p",
            heuristic_profile_id="heuristic_v1_sq8_4p",
            randomness=0.02,
            think_time_ms=6500,
            use_neural_net=True,
            notes="High square8 4p tier using MCTS with neural guidance.",
        ),
        # D7 – expert MCTS on square8, 4-player (neural).
        (7, BoardType.SQUARE8, 4): LadderTierConfig(
            difficulty=7,
            board_type=BoardType.SQUARE8,
            num_players=4,
            ai_type=AIType.MCTS,
            model_id="ringrift_best_sq8_4p",
            heuristic_profile_id="heuristic_v1_sq8_4p",
            randomness=0.0,
            think_time_ms=9000,
            use_neural_net=True,
            notes="Expert square8 4p tier using MCTS with neural guidance.",
        ),
        # D8 – strong expert MCTS on square8, 4-player (neural).
        (8, BoardType.SQUARE8, 4): LadderTierConfig(
            difficulty=8,
            board_type=BoardType.SQUARE8,
            num_players=4,
            ai_type=AIType.MCTS,
            model_id="ringrift_best_sq8_4p",
            heuristic_profile_id="heuristic_v1_sq8_4p",
            randomness=0.0,
            think_time_ms=12000,
            use_neural_net=True,
            notes="Strong expert square8 4p tier using MCTS with neural.",
        ),
        # D9 – master Descent on square8, 4-player.
        (9, BoardType.SQUARE8, 4): LadderTierConfig(
            difficulty=9,
            board_type=BoardType.SQUARE8,
            num_players=4,
            ai_type=AIType.DESCENT,
            model_id="ringrift_best_sq8_4p",
            heuristic_profile_id="heuristic_v1_sq8_4p",
            randomness=0.0,
            think_time_ms=15000,
            use_neural_net=True,
            notes="Master square8 4p tier using Descent with neural.",
        ),
        # D10 – grandmaster Descent on square8, 4-player.
        (10, BoardType.SQUARE8, 4): LadderTierConfig(
            difficulty=10,
            board_type=BoardType.SQUARE8,
            num_players=4,
            ai_type=AIType.DESCENT,
            model_id="ringrift_best_sq8_4p",
            heuristic_profile_id="heuristic_v1_sq8_4p",
            randomness=0.0,
            think_time_ms=20000,
            use_neural_net=True,
            notes="Grandmaster square8 4p tier using strongest Descent.",
        ),
        # D11 – ultimate Descent on square8, 4-player.
        (11, BoardType.SQUARE8, 4): LadderTierConfig(
            difficulty=11,
            board_type=BoardType.SQUARE8,
            num_players=4,
            ai_type=AIType.DESCENT,
            model_id="ringrift_best_sq8_4p",
            heuristic_profile_id="heuristic_v1_sq8_4p",
            randomness=0.0,
            think_time_ms=90000,  # 90s for 4-player
            use_neural_net=True,
            notes="Ultimate square8 4p tier using Descent with 90s think time.",
        ),
    }


_LADDER_TIER_CONFIGS: dict[LadderKey, LadderTierConfig] = {
    **_build_default_square8_two_player_configs(),
    **_build_default_square19_two_player_configs(),
    **_build_default_hex_two_player_configs(),
    **_build_default_square19_three_player_configs(),
    **_build_default_square19_four_player_configs(),
    **_build_default_hex_three_player_configs(),
    **_build_default_hex_four_player_configs(),
    **_build_default_square8_three_player_configs(),
    **_build_default_square8_four_player_configs(),
}


def get_ladder_tier_config(
    difficulty: int,
    board_type: BoardType,
    num_players: int,
) -> LadderTierConfig:
    """Return the LadderTierConfig for a given (difficulty, board, players).

    The lookup is exact on all three fields. Callers that want to fall back
    to legacy difficulty logic should catch KeyError and handle it
    explicitly.
    """
    key: LadderKey = (difficulty, board_type, num_players)
    try:
        return _LADDER_TIER_CONFIGS[key]
    except KeyError as exc:  # pragma: no cover - defensive error path
        available = ", ".join(
            sorted(
                f"(difficulty={d}, board_type={bt.value}, num_players={n})"
                for (d, bt, n) in _LADDER_TIER_CONFIGS
            )
        )
        raise KeyError(
            f"No ladder tier configured for difficulty={difficulty}, "
            f"board_type={board_type!r}, num_players={num_players}. "
            f"Available tiers: {available}"
        ) from exc


def list_ladder_tiers(
    board_type: BoardType | None = None,
    num_players: int | None = None,
) -> list[LadderTierConfig]:
    """Return all configured LadderTierConfig entries, optionally filtered."""
    configs = list(_LADDER_TIER_CONFIGS.values())
    if board_type is not None:
        configs = [c for c in configs if c.board_type == board_type]
    if num_players is not None:
        configs = [c for c in configs if c.num_players == num_players]
    # Deterministic ordering for debugging / tests.
    configs.sort(
        key=lambda c: (c.board_type.value, c.num_players, c.difficulty)
    )
    return configs


# =============================================================================
# Runtime Model Promotion API
# =============================================================================
# These functions allow the training loop to dynamically update which models
# are used for each difficulty tier without requiring code changes.

from app.utils.paths import AI_SERVICE_ROOT

# Runtime overrides loaded from JSON file
_RUNTIME_OVERRIDES: dict[LadderKey, dict[str, str]] = {}
_RUNTIME_OVERRIDES_PATH = AI_SERVICE_ROOT / "data" / "ladder_runtime_overrides.json"


def load_runtime_overrides() -> None:
    """Load runtime model/profile overrides from JSON file.

    This allows the training loop to promote new models without editing
    ladder_config.py. Overrides are stored in data/ladder_runtime_overrides.json.
    """
    global _RUNTIME_OVERRIDES

    if not _RUNTIME_OVERRIDES_PATH.exists():
        _RUNTIME_OVERRIDES = {}
        return

    try:
        with _RUNTIME_OVERRIDES_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)

        _RUNTIME_OVERRIDES = {}
        for key_str, overrides in data.items():
            # Parse key like "6_square8_2" -> (6, BoardType.SQUARE8, 2)
            parts = key_str.split("_")
            if len(parts) >= 3:
                difficulty = int(parts[0])
                board_type_str = parts[1]
                num_players = int(parts[2])

                # Convert board type string to enum
                board_type_map = {
                    "square8": BoardType.SQUARE8,
                    "square19": BoardType.SQUARE19,
                    "hexagonal": BoardType.HEXAGONAL,
                }
                if board_type_str in board_type_map:
                    key = (difficulty, board_type_map[board_type_str], num_players)
                    _RUNTIME_OVERRIDES[key] = overrides
    except Exception as e:
        print(f"Warning: Failed to load ladder runtime overrides: {e}")
        _RUNTIME_OVERRIDES = {}


def save_runtime_overrides() -> None:
    """Save runtime model/profile overrides to JSON file."""
    _RUNTIME_OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Convert keys to strings for JSON serialization
    data = {}
    for (difficulty, board_type, num_players), overrides in _RUNTIME_OVERRIDES.items():
        key_str = f"{difficulty}_{board_type.value}_{num_players}"
        data[key_str] = overrides

    with _RUNTIME_OVERRIDES_PATH.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def update_tier_model(
    difficulty: int,
    board_type: BoardType,
    num_players: int,
    model_id: str,
    heuristic_profile_id: str | None = None,
) -> bool:
    """Update the model used for a specific difficulty tier at runtime.

    This stores the override in a JSON file that is loaded on startup.
    The override takes precedence over the hardcoded defaults.

    Args:
        difficulty: Difficulty level (1-10)
        board_type: Board type enum
        num_players: Number of players
        model_id: New model ID to use (e.g., "ringrift_v6_sq8_2p")
        heuristic_profile_id: Optional new heuristic profile ID

    Returns:
        True if successful, False otherwise
    """
    key: LadderKey = (difficulty, board_type, num_players)

    # Verify the tier exists in base config
    if key not in _LADDER_TIER_CONFIGS:
        return False

    # Create or update override
    override: dict[str, str] = {"model_id": model_id}
    if heuristic_profile_id:
        override["heuristic_profile_id"] = heuristic_profile_id

    _RUNTIME_OVERRIDES[key] = override
    save_runtime_overrides()

    return True


def get_effective_ladder_config(
    difficulty: int,
    board_type: BoardType,
    num_players: int,
) -> LadderTierConfig:
    """Get ladder config with runtime overrides applied.

    This is the preferred way to get ladder config in production code,
    as it respects runtime model promotions from the training loop.
    """
    key: LadderKey = (difficulty, board_type, num_players)

    # Get base config
    base_config = get_ladder_tier_config(difficulty, board_type, num_players)

    # Check for runtime overrides
    if key in _RUNTIME_OVERRIDES:
        overrides = _RUNTIME_OVERRIDES[key]

        # Create new config with overrides applied
        return LadderTierConfig(
            difficulty=base_config.difficulty,
            board_type=base_config.board_type,
            num_players=base_config.num_players,
            ai_type=base_config.ai_type,
            model_id=overrides.get("model_id", base_config.model_id),
            heuristic_profile_id=overrides.get(
                "heuristic_profile_id", base_config.heuristic_profile_id
            ),
            randomness=base_config.randomness,
            think_time_ms=base_config.think_time_ms,
            use_neural_net=base_config.use_neural_net,
            notes=base_config.notes + " [runtime override]",
        )

    return base_config


def get_all_runtime_overrides() -> dict[str, dict[str, str]]:
    """Get all current runtime overrides for debugging/display."""
    result = {}
    for (difficulty, board_type, num_players), overrides in _RUNTIME_OVERRIDES.items():
        key_str = f"D{difficulty}_{board_type.value}_{num_players}p"
        result[key_str] = overrides
    return result


def update_tier_heuristic_profile(
    difficulty: int,
    board_type: BoardType,
    num_players: int,
    heuristic_profile_id: str,
) -> bool:
    """Update just the heuristic profile for a specific difficulty tier.

    This is used by CMAES auto-optimization to promote new heuristic weights
    without affecting the neural network model assignment.

    Args:
        difficulty: Difficulty level (1-10)
        board_type: Board type enum
        num_players: Number of players
        heuristic_profile_id: New heuristic profile ID (e.g., "heuristic_v2_sq8_2p")

    Returns:
        True if successful, False otherwise
    """
    key: LadderKey = (difficulty, board_type, num_players)

    # Verify the tier exists in base config
    if key not in _LADDER_TIER_CONFIGS:
        return False

    # Get existing override or create new one
    existing = _RUNTIME_OVERRIDES.get(key, {})
    existing["heuristic_profile_id"] = heuristic_profile_id

    _RUNTIME_OVERRIDES[key] = existing
    save_runtime_overrides()

    return True


def get_heuristic_tiers(board_type: BoardType, num_players: int) -> list[int]:
    """Get difficulty tiers that use heuristic evaluation for a board/player config.

    These are tiers D2-D5 which rely on heuristic weights (with or without NN).
    """
    heuristic_difficulties = []
    for difficulty in range(2, 6):  # D2-D5 use heuristics
        key = (difficulty, board_type, num_players)
        if key in _LADDER_TIER_CONFIGS:
            heuristic_difficulties.append(difficulty)
    return heuristic_difficulties


# Load overrides on module import
load_runtime_overrides()
