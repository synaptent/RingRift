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

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from app.models import AIType, BoardType


LadderKey = Tuple[int, BoardType, int]


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
    model_id: Optional[str]
    heuristic_profile_id: Optional[str]
    randomness: float
    think_time_ms: int
    use_neural_net: bool = False
    notes: str = ""


def _build_default_square8_two_player_configs() -> Dict[
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
        # D2 – easy heuristic baseline on square8, 2-player.
        (2, BoardType.SQUARE8, 2): LadderTierConfig(
            difficulty=2,
            board_type=BoardType.SQUARE8,
            num_players=2,
            ai_type=AIType.HEURISTIC,
            model_id="heuristic_v1_2p",
            heuristic_profile_id="heuristic_v1_2p",
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
            heuristic_profile_id="heuristic_v1_2p",
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
            model_id="v1-minimax-4-nnue",
            heuristic_profile_id="heuristic_v1_2p",
            randomness=0.08,
            think_time_ms=2800,
            use_neural_net=True,
            notes=(
                "Mid square8 2p tier using minimax with NNUE neural "
                "evaluation for stronger positional assessment."
            ),
        ),
        # D5 – upper-mid MCTS on square8, 2-player (non-neural).
        (5, BoardType.SQUARE8, 2): LadderTierConfig(
            difficulty=5,
            board_type=BoardType.SQUARE8,
            num_players=2,
            ai_type=AIType.MCTS,
            model_id="v1-mcts-5",
            heuristic_profile_id="heuristic_v1_2p",
            randomness=0.05,
            think_time_ms=4000,
            use_neural_net=False,
            notes=(
                "Upper-mid square8 2p tier using MCTS with heuristic "
                "rollouts only (no neural net)."
            ),
        ),
        # D6 – high MCTS on square8, 2-player (neural).
        (6, BoardType.SQUARE8, 2): LadderTierConfig(
            difficulty=6,
            board_type=BoardType.SQUARE8,
            num_players=2,
            ai_type=AIType.MCTS,
            model_id="ringrift_v4_sq8_2p",
            heuristic_profile_id="heuristic_v1_2p",
            randomness=0.02,
            think_time_ms=5500,
            use_neural_net=True,
            notes=(
                "High square8 2p tier using MCTS with neural value and "
                "policy guidance for improved search efficiency."
            ),
        ),
        # D7 – expert MCTS on square8, 2-player (neural).
        (7, BoardType.SQUARE8, 2): LadderTierConfig(
            difficulty=7,
            board_type=BoardType.SQUARE8,
            num_players=2,
            ai_type=AIType.MCTS,
            model_id="ringrift_v4_sq8_2p",
            heuristic_profile_id="heuristic_v1_2p",
            randomness=0.0,
            think_time_ms=7500,
            use_neural_net=True,
            notes=(
                "Expert square8 2p tier using MCTS with neural guidance "
                "and higher search budget."
            ),
        ),
        # D8 – strong expert MCTS on square8, 2-player (neural).
        (8, BoardType.SQUARE8, 2): LadderTierConfig(
            difficulty=8,
            board_type=BoardType.SQUARE8,
            num_players=2,
            ai_type=AIType.MCTS,
            model_id="ringrift_v4_sq8_2p",
            heuristic_profile_id="heuristic_v1_2p",
            randomness=0.0,
            think_time_ms=9600,
            use_neural_net=True,
            notes=(
                "Strong expert square8 2p tier using MCTS with neural "
                "guidance and large search budget."
            ),
        ),
        # D9 – master Descent on square8, 2-player.
        (9, BoardType.SQUARE8, 2): LadderTierConfig(
            difficulty=9,
            board_type=BoardType.SQUARE8,
            num_players=2,
            ai_type=AIType.DESCENT,
            model_id="ringrift_v4_sq8_2p",
            heuristic_profile_id="heuristic_v1_2p",
            randomness=0.0,
            think_time_ms=12600,
            use_neural_net=True,
            notes=(
                "Master square8 2p tier using Descent/UBFM search with "
                "AlphaZero-style neural policy and value guidance."
            ),
        ),
        # D10 – grandmaster Descent on square8, 2-player.
        (10, BoardType.SQUARE8, 2): LadderTierConfig(
            difficulty=10,
            board_type=BoardType.SQUARE8,
            num_players=2,
            ai_type=AIType.DESCENT,
            model_id="ringrift_v4_sq8_2p",
            heuristic_profile_id="heuristic_v1_2p",
            randomness=0.0,
            think_time_ms=16000,
            use_neural_net=True,
            notes=(
                "Grandmaster square8 2p tier using strongest Descent "
                "configuration with maximum search budget."
            ),
        ),
    }


def _build_default_square19_two_player_configs() -> Dict[
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
            model_id="heuristic_v1_square19_2p",
            heuristic_profile_id="heuristic_v1_square19_2p",
            randomness=0.3,
            think_time_ms=300,
            use_neural_net=False,
            notes="Easy square19 2p tier using heuristic evaluation.",
        ),
        # D3 – lower-mid minimax on square19, 2-player (non-neural).
        (3, BoardType.SQUARE19, 2): LadderTierConfig(
            difficulty=3,
            board_type=BoardType.SQUARE19,
            num_players=2,
            ai_type=AIType.MINIMAX,
            model_id="v1-minimax-square19-3",
            heuristic_profile_id="heuristic_v1_square19_2p",
            randomness=0.15,
            think_time_ms=2500,
            use_neural_net=False,
            notes="Lower-mid square19 2p tier using minimax (non-neural).",
        ),
        # D4 – mid minimax on square19, 2-player (neural/NNUE).
        (4, BoardType.SQUARE19, 2): LadderTierConfig(
            difficulty=4,
            board_type=BoardType.SQUARE19,
            num_players=2,
            ai_type=AIType.MINIMAX,
            model_id="v1-minimax-square19-4-nnue",
            heuristic_profile_id="heuristic_v1_square19_2p",
            randomness=0.08,
            think_time_ms=4000,
            use_neural_net=True,
            notes="Mid square19 2p tier using minimax with NNUE.",
        ),
        # D5 – upper-mid MCTS on square19, 2-player (non-neural).
        (5, BoardType.SQUARE19, 2): LadderTierConfig(
            difficulty=5,
            board_type=BoardType.SQUARE19,
            num_players=2,
            ai_type=AIType.MCTS,
            model_id="v1-mcts-square19-5",
            heuristic_profile_id="heuristic_v1_square19_2p",
            randomness=0.05,
            think_time_ms=6000,
            use_neural_net=False,
            notes="Upper-mid square19 2p tier using MCTS (non-neural).",
        ),
        # D6 – high MCTS on square19, 2-player (neural).
        (6, BoardType.SQUARE19, 2): LadderTierConfig(
            difficulty=6,
            board_type=BoardType.SQUARE19,
            num_players=2,
            ai_type=AIType.MCTS,
            model_id="v1-mcts-square19-6-neural",
            heuristic_profile_id="heuristic_v1_square19_2p",
            randomness=0.02,
            think_time_ms=8000,
            use_neural_net=True,
            notes="High square19 2p tier using MCTS with neural guidance.",
        ),
        # D7 – expert MCTS on square19, 2-player (neural).
        (7, BoardType.SQUARE19, 2): LadderTierConfig(
            difficulty=7,
            board_type=BoardType.SQUARE19,
            num_players=2,
            ai_type=AIType.MCTS,
            model_id="v1-mcts-square19-7-neural",
            heuristic_profile_id="heuristic_v1_square19_2p",
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
            model_id="v1-mcts-square19-8-neural",
            heuristic_profile_id="heuristic_v1_square19_2p",
            randomness=0.0,
            think_time_ms=14000,
            use_neural_net=True,
            notes="Strong expert square19 2p tier using MCTS with neural.",
        ),
        # D9 – master Descent on square19, 2-player.
        (9, BoardType.SQUARE19, 2): LadderTierConfig(
            difficulty=9,
            board_type=BoardType.SQUARE19,
            num_players=2,
            ai_type=AIType.DESCENT,
            model_id="v1-descent-square19-9",
            heuristic_profile_id="heuristic_v1_square19_2p",
            randomness=0.0,
            think_time_ms=18000,
            use_neural_net=True,
            notes="Master square19 2p tier using Descent with neural.",
        ),
        # D10 – grandmaster Descent on square19, 2-player.
        (10, BoardType.SQUARE19, 2): LadderTierConfig(
            difficulty=10,
            board_type=BoardType.SQUARE19,
            num_players=2,
            ai_type=AIType.DESCENT,
            model_id="v1-descent-square19-10",
            heuristic_profile_id="heuristic_v1_square19_2p",
            randomness=0.0,
            think_time_ms=24000,
            use_neural_net=True,
            notes="Grandmaster square19 2p tier using strongest Descent.",
        ),
    }


def _build_default_hex_two_player_configs() -> Dict[
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
            heuristic_profile_id="heuristic_v1_2p",
            randomness=0.3,
            think_time_ms=300,
            use_neural_net=False,
            notes="Easy hex 2p tier using heuristic evaluation.",
        ),
        # D3 – lower-mid minimax on hexagonal, 2-player (non-neural).
        (3, BoardType.HEXAGONAL, 2): LadderTierConfig(
            difficulty=3,
            board_type=BoardType.HEXAGONAL,
            num_players=2,
            ai_type=AIType.MINIMAX,
            model_id="v1-minimax-hex-3",
            heuristic_profile_id="heuristic_v1_2p",
            randomness=0.15,
            think_time_ms=2500,
            use_neural_net=False,
            notes="Lower-mid hex 2p tier using minimax (non-neural).",
        ),
        # D4 – mid minimax on hexagonal, 2-player (neural/NNUE).
        (4, BoardType.HEXAGONAL, 2): LadderTierConfig(
            difficulty=4,
            board_type=BoardType.HEXAGONAL,
            num_players=2,
            ai_type=AIType.MINIMAX,
            model_id="v1-minimax-hex-4-nnue",
            heuristic_profile_id="heuristic_v1_2p",
            randomness=0.08,
            think_time_ms=4000,
            use_neural_net=True,
            notes="Mid hex 2p tier using minimax with NNUE.",
        ),
        # D5 – upper-mid MCTS on hexagonal, 2-player (non-neural).
        (5, BoardType.HEXAGONAL, 2): LadderTierConfig(
            difficulty=5,
            board_type=BoardType.HEXAGONAL,
            num_players=2,
            ai_type=AIType.MCTS,
            model_id="v1-mcts-hex-5",
            heuristic_profile_id="heuristic_v1_2p",
            randomness=0.05,
            think_time_ms=6000,
            use_neural_net=False,
            notes="Upper-mid hex 2p tier using MCTS (non-neural).",
        ),
        # D6 – high MCTS on hexagonal, 2-player (neural).
        (6, BoardType.HEXAGONAL, 2): LadderTierConfig(
            difficulty=6,
            board_type=BoardType.HEXAGONAL,
            num_players=2,
            ai_type=AIType.MCTS,
            model_id="v1-mcts-hex-6-neural",
            heuristic_profile_id="heuristic_v1_2p",
            randomness=0.02,
            think_time_ms=8000,
            use_neural_net=True,
            notes="High hex 2p tier using MCTS with neural guidance.",
        ),
        # D7 – expert MCTS on hexagonal, 2-player (neural).
        (7, BoardType.HEXAGONAL, 2): LadderTierConfig(
            difficulty=7,
            board_type=BoardType.HEXAGONAL,
            num_players=2,
            ai_type=AIType.MCTS,
            model_id="v1-mcts-hex-7-neural",
            heuristic_profile_id="heuristic_v1_2p",
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
            model_id="v1-mcts-hex-8-neural",
            heuristic_profile_id="heuristic_v1_2p",
            randomness=0.0,
            think_time_ms=14000,
            use_neural_net=True,
            notes="Strong expert hex 2p tier using MCTS with neural.",
        ),
        # D9 – master Descent on hexagonal, 2-player.
        (9, BoardType.HEXAGONAL, 2): LadderTierConfig(
            difficulty=9,
            board_type=BoardType.HEXAGONAL,
            num_players=2,
            ai_type=AIType.DESCENT,
            model_id="v1-descent-hex-9",
            heuristic_profile_id="heuristic_v1_2p",
            randomness=0.0,
            think_time_ms=18000,
            use_neural_net=True,
            notes="Master hex 2p tier using Descent with neural.",
        ),
        # D10 – grandmaster Descent on hexagonal, 2-player.
        (10, BoardType.HEXAGONAL, 2): LadderTierConfig(
            difficulty=10,
            board_type=BoardType.HEXAGONAL,
            num_players=2,
            ai_type=AIType.DESCENT,
            model_id="v1-descent-hex-10",
            heuristic_profile_id="heuristic_v1_2p",
            randomness=0.0,
            think_time_ms=24000,
            use_neural_net=True,
            notes="Grandmaster hex 2p tier using strongest Descent.",
        ),
    }


def _build_default_square8_three_player_configs() -> Dict[
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
            model_id="heuristic_v1_3p",
            heuristic_profile_id="heuristic_v1_3p",
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
            heuristic_profile_id="heuristic_v1_3p",
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
            model_id="v1-minimax-3p-4-nnue",
            heuristic_profile_id="heuristic_v1_3p",
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
            heuristic_profile_id="heuristic_v1_3p",
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
            model_id="v1-mcts-3p-6-neural",
            heuristic_profile_id="heuristic_v1_3p",
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
            model_id="v1-mcts-3p-7-neural",
            heuristic_profile_id="heuristic_v1_3p",
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
            model_id="v1-mcts-3p-8-neural",
            heuristic_profile_id="heuristic_v1_3p",
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
            model_id="v1-descent-3p-9",
            heuristic_profile_id="heuristic_v1_3p",
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
            model_id="v1-descent-3p-10",
            heuristic_profile_id="heuristic_v1_3p",
            randomness=0.0,
            think_time_ms=18000,
            use_neural_net=True,
            notes="Grandmaster square8 3p tier using strongest Descent.",
        ),
    }


def _build_default_square8_four_player_configs() -> Dict[
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
            model_id="heuristic_v1_4p",
            heuristic_profile_id="heuristic_v1_4p",
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
            heuristic_profile_id="heuristic_v1_4p",
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
            model_id="v1-minimax-4p-4-nnue",
            heuristic_profile_id="heuristic_v1_4p",
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
            heuristic_profile_id="heuristic_v1_4p",
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
            model_id="v1-mcts-4p-6-neural",
            heuristic_profile_id="heuristic_v1_4p",
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
            model_id="v1-mcts-4p-7-neural",
            heuristic_profile_id="heuristic_v1_4p",
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
            model_id="v1-mcts-4p-8-neural",
            heuristic_profile_id="heuristic_v1_4p",
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
            model_id="v1-descent-4p-9",
            heuristic_profile_id="heuristic_v1_4p",
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
            model_id="v1-descent-4p-10",
            heuristic_profile_id="heuristic_v1_4p",
            randomness=0.0,
            think_time_ms=20000,
            use_neural_net=True,
            notes="Grandmaster square8 4p tier using strongest Descent.",
        ),
    }


_LADDER_TIER_CONFIGS: Dict[LadderKey, LadderTierConfig] = {
    **_build_default_square8_two_player_configs(),
    **_build_default_square19_two_player_configs(),
    **_build_default_hex_two_player_configs(),
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
                for (d, bt, n) in _LADDER_TIER_CONFIGS.keys()
            )
        )
        raise KeyError(
            f"No ladder tier configured for difficulty={difficulty}, "
            f"board_type={board_type!r}, num_players={num_players}. "
            f"Available tiers: {available}"
        ) from exc


def list_ladder_tiers(
    board_type: Optional[BoardType] = None,
    num_players: Optional[int] = None,
) -> List[LadderTierConfig]:
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
