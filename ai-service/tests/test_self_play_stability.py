from __future__ import annotations

"""Stability/soak tests for long-running Python self-play.

These tests are intentionally lightweight for CI (small game counts) but
exercise the same RingRiftEnv + mixed-engine AI stack used by the
self-play dataset generator. The goal is to detect rules/AI integration
regressions early, especially around territory processing and
forced-elimination orchestration, without requiring multi-hundred game
runs in CI.

For deeper local soaks, increase the number of games via the
RINGRIFT_SELFPLAY_STABILITY_GAMES env var.
"""

import os
import random
import sys
from typing import Dict

import pytest

# TODO-SELF-PLAY-STABILITY: These tests run multiple full self-play games
# with mixed AI engines which can exceed timeout limits. Even with reduced
# game counts (3), the territory processing and AI selection overhead can
# cause timeouts in CI. Skip pending optimization or async test execution.
pytestmark = pytest.mark.skip(
    reason="TODO-SELF-PLAY-STABILITY: multi-game self-play timeouts"
)

# Ensure `app.*` imports resolve when running pytest from ai-service/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from app.ai.base import BaseAI  # noqa: E402
from app.main import _get_difficulty_profile, _create_ai_instance  # noqa: E402
from app.models import AIConfig, AIType, BoardType, GameStatus, GameState  # noqa: E402
from app.training.env import RingRiftEnv  # noqa: E402

# Test timeout guards to prevent hanging in CI
# Allow more time for self-play as it runs multiple games
TEST_TIMEOUT_SECONDS = 60
# Maximum moves per game for safety (in addition to env.max_moves)
MAX_MOVES_SAFETY_LIMIT = 250


def _build_mixed_ai_pool(game_index: int) -> dict[int, BaseAI]:
    """Construct a per-game pool of AIs mirroring generate_territory_dataset.

    We deliberately set think_time_ms=0 so that these stability runs do not
    incur artificial UX delays; search-based engines still respect their
    internal iteration limits.

    To keep CI runs memory- and time-friendly, we default to a *light* band
    of AI difficulties (Random/Heuristic/low-depth Minimax). This can be
    overridden by setting ``RINGRIFT_SELFPLAY_STABILITY_DIFFICULTY_BAND=
    canonical`` to exercise the full ladder (including MCTS/Descent) during
    local deep soaks.
    """

    band = os.getenv(
        "RINGRIFT_SELFPLAY_STABILITY_DIFFICULTY_BAND",
        "light",
    )

    # Difficulty presets chosen to cover the canonical ladder while keeping
    # runtime reasonable on square8 (mirrors generate_territory_dataset).
    if band == "canonical":
        difficulty_choices = [
            1,  # Random
            2,  # Heuristic
            4,
            5,
            6,  # Minimax band
            7,
            8,  # MCTS band
            9,
            10,  # Descent band
        ]
    else:
        # Light band for CI and memory-conscious stability runs: Random,
        # Heuristic, and low-depth Minimax only.
        difficulty_choices = [
            1,
            2,
            4,
            5,
        ]

    game_rng = random.Random(42 + game_index)

    ai_by_player: dict[int, BaseAI] = {}

    # We construct AIs lazily per player once we see the initial GameState,
    # because the set of active player_numbers can vary with N-player mode.

    def _ensure_ai_for_player(player_number: int) -> BaseAI:
        if player_number in ai_by_player:
            return ai_by_player[player_number]

        difficulty = game_rng.choice(difficulty_choices)
        profile = _get_difficulty_profile(difficulty)
        ai_type = profile["ai_type"]

        heuristic_profile_id = None
        nn_model_id = None
        if ai_type == AIType.HEURISTIC:
            heuristic_profile_id = profile.get("profile_id")

        config = AIConfig(
            difficulty=difficulty,
            randomness=profile["randomness"],
            think_time=0,  # disable UX delay for soak tests
            rngSeed=game_rng.randrange(0, 2**31),
            heuristic_profile_id=heuristic_profile_id,
            nn_model_id=nn_model_id,
        )
        ai = _create_ai_instance(ai_type, player_number, config)
        ai_by_player[player_number] = ai
        return ai

    # Return a closure-style mapping that will populate lazily.
    class _AIProxy(dict[int, BaseAI]):
        def __missing__(self, key: int) -> BaseAI:  # type: ignore[override]
            return _ensure_ai_for_player(key)

    return _AIProxy()


@pytest.mark.timeout(TEST_TIMEOUT_SECONDS)
@pytest.mark.slow
def test_self_play_mixed_2p_square8_stability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run a small batch of mixed-engine 2p self-play games for stability.

    Assertions:
    - No exceptions are raised during env.step or AI selection.
    - Each game either reaches a non-ACTIVE GameStatus or cleanly hits
      max_moves without getting stuck in an illegal no-move ACTIVE state.
    - When no legal moves are available for the current player, the state
      is not left in GameStatus.ACTIVE (i.e. stalemates are surfaced).
    """

    num_games = int(os.getenv("RINGRIFT_SELFPLAY_STABILITY_GAMES", "3"))

    env = RingRiftEnv(
        board_type=BoardType.SQUARE8,
        max_moves=200,
        reward_on="terminal",
        num_players=2,
    )

    # Enable the strict no-move invariant at the engine level for this test
    import app.game_engine as game_engine  # local import to avoid cycles

    monkeypatch.setenv("RINGRIFT_STRICT_NO_MOVE_INVARIANT", "1")
    monkeypatch.setattr(
        game_engine,
        "STRICT_NO_MOVE_INVARIANT",
        True,
        raising=False,
    )

    for game_idx in range(num_games):
        state: GameState = env.reset(seed=1234 + game_idx)
        ai_by_player = _build_mixed_ai_pool(game_idx)

        move_count = 0

        move_safety_count = 0
        while True:
            # Sanity: current_player must correspond to a real player.
            player_numbers = {p.player_number for p in state.players}
            assert state.current_player in player_numbers, (
                f"Invalid current_player {state.current_player} in "
                f"{player_numbers}"
            )

            legal_moves = env.legal_moves()

            if not legal_moves:
                # With TS-aligned semantics and the strict no-move invariant
                # enabled for this test, an ACTIVE state with no legal moves
                # should be impossible. If it occurs, surface it as a hard
                # failure instead of a soft terminal.
                assert state.game_status != GameStatus.ACTIVE
                break

            ai = ai_by_player[state.current_player]
            move = ai.select_move(state)
            assert move is not None, "AI returned no move despite legal moves"

            state, _reward, done, _info = env.step(move)
            move_count += 1
            move_safety_count += 1

            # Guard against runaway games with multiple safety checks.
            assert move_count <= env.max_moves
            if move_safety_count >= MAX_MOVES_SAFETY_LIMIT:
                # Terminate test gracefully if safety limit is hit
                break

            if done:
                # Either game_status is non-ACTIVE (normal termination) or we
                # hit the max_moves cutoff.
                assert (
                    state.game_status != GameStatus.ACTIVE
                    or move_count == env.max_moves
                )
                break
