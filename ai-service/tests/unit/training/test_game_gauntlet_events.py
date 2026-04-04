from types import SimpleNamespace
from unittest.mock import MagicMock

from app.models import BoardType
from app.training import game_gauntlet


def test_evaluation_progress_payload_includes_config_key_and_board_type(monkeypatch):
    emitted_payloads = []
    import app.training.elo_recording as elo_recording

    monkeypatch.setattr(elo_recording, "record_gauntlet_match", lambda **kwargs: None)
    monkeypatch.setattr(
        game_gauntlet,
        "_publish_evaluation_progress",
        lambda **kwargs: emitted_payloads.append(kwargs),
    )

    monkeypatch.setattr(game_gauntlet, "create_neural_ai", lambda *args, **kwargs: object())
    monkeypatch.setattr(game_gauntlet, "create_baseline_ai", lambda *args, **kwargs: object())

    def fake_play_single_game(**kwargs):
        return SimpleNamespace(
            candidate_won=True,
            winner=1,
            victory_reason="test",
            move_count=1,
        )

    monkeypatch.setattr(game_gauntlet, "play_single_game", fake_play_single_game)

    game_gauntlet._evaluate_single_opponent(
        baseline=game_gauntlet.BaselineOpponent.RANDOM,
        model_path="dummy",
        board_type=BoardType.SQUARE19,
        games_per_opponent=1,
        num_players=2,
        verbose=False,
        model_getter=None,
        model_type="cnn",
        early_stopping=False,
        early_stopping_confidence=0.95,
        early_stopping_min_games=1,
        parallel_games=1,
    )

    assert emitted_payloads, "Expected evaluation progress events to be published"
    payload = emitted_payloads[-1]
    assert payload["board_type"] == BoardType.SQUARE19
    assert payload["baseline_name"] == "random"
    assert payload["games_per_opponent"] == 1
    assert payload["num_players"] == 2
    assert payload["result"]["wins"] == 1
    assert payload["result"]["games"] == 1
