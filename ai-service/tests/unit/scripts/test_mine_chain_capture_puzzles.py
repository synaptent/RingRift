"""Tests for the chain-capture puzzle miner.

Games are generated in-process with the canonical training environment and
random move selection (no database, no network), then mined and validated.
"""

import json
import random

import pytest

from app.models import BoardType, GameStatus
from app.training.env import TrainingEnvConfig, make_env
from scripts.mine_chain_capture_puzzles import (
    SCHEMA_VERSION,
    chain_score,
    mine_games,
    validate_puzzle,
)


def _random_games(num_games: int, seed: int, board_type=BoardType.HEX8, num_players=2):
    """Play random games, returning (metadata, initial_state, moves) tuples."""
    rng = random.Random(seed)
    games = []
    for i in range(num_games):
        env = make_env(TrainingEnvConfig(board_type=board_type, num_players=num_players))
        initial = env.reset()
        moves = []
        state, done = initial, False
        while not done:
            legal = env.legal_moves()
            if not legal:
                break
            move = rng.choice(legal)
            moves.append(move)
            state, _r, done, _info = env.step(move)
        games.append(({"game_id": f"testgame_{seed}_{i}"}, initial, moves))
    return games


@pytest.fixture(scope="module")
def random_hex8_games():
    return _random_games(num_games=5, seed=1234)


class TestChainScore:
    def test_non_capture_moves_score_zero(self, random_hex8_games):
        _meta, initial, _moves = random_hex8_games[0]
        for move in initial.move_history or []:
            pass  # initial state has no history; scored below via legal moves
        from app.game_engine import GameEngine
        from app.models import MoveType

        legal = GameEngine.get_valid_moves(initial, initial.current_player)
        for move in legal:
            score, pv = chain_score(initial, move)
            if move.type != MoveType.OVERTAKING_CAPTURE:
                assert score == 0
                assert pv == [move]


class TestMineAndValidate:
    def test_mined_puzzles_are_well_formed_and_self_validating(self, random_hex8_games):
        puzzles = mine_games(
            random_hex8_games,
            max_puzzles=10,
            min_chain=1,
            min_margin=1,
            min_ply=4,
            max_per_game=3,
        )
        # Random hex8 games reliably contain capture opportunities; if this
        # ever yields zero the seed below needs adjusting, not the assert.
        assert puzzles, "expected at least one puzzle from 5 random games"
        for puzzle in puzzles:
            assert puzzle["schema_version"] == SCHEMA_VERSION
            assert puzzle["theme"] == "chain_capture"
            assert puzzle["board_type"] == "hex8"
            assert puzzle["num_players"] == 2
            assert puzzle["solution"]["margin"] >= 1
            assert puzzle["solution"]["score"] >= 1
            assert len(puzzle["solution"]["moves"]) >= 1
            # JSON round-trip must be lossless enough to re-validate.
            round_tripped = json.loads(json.dumps(puzzle))
            ok, reason = validate_puzzle(round_tripped)
            assert ok, f"{puzzle['id']}: {reason}"

    def test_validate_rejects_tampered_solution(self, random_hex8_games):
        puzzles = mine_games(
            random_hex8_games,
            max_puzzles=1,
            min_chain=1,
            min_margin=1,
            min_ply=4,
            max_per_game=3,
        )
        assert puzzles
        tampered = json.loads(json.dumps(puzzles[0]))
        tampered["solution"]["score"] += 1
        ok, reason = validate_puzzle(tampered)
        assert not ok
        assert "score" in reason

    def test_max_puzzles_cap(self, random_hex8_games):
        puzzles = mine_games(
            random_hex8_games,
            max_puzzles=2,
            min_chain=1,
            min_margin=1,
            min_ply=4,
            max_per_game=3,
        )
        assert len(puzzles) <= 2
